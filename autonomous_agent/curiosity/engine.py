"""Curiosity：发现“值得探索的问题”，只产生 Thought，不执行行为。

Curiosity 的输出永远是一个候选问题（CREATE_THOUGHT），之后交给第三阶段的
思考文本引擎慢慢想。它不能发消息、不能改状态、不能绕过 Behavior Guard。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Iterable

from ..config import CuriosityConfig
from ..core.emotion import EmotionState
from ..memory.models import MemoryStatus, MemoryType, utcnow
from ..memory.store import MemoryStore
from ..memory.topics import Topic, TopicStatus, TopicStore
from ..thought.models import Thought, ThoughtStatus, ThoughtType
from ..thought.repository import ThoughtRepository
from ..thought.similarity import similarity
from .scoring import CuriosityCandidate, curiosity_score

logger = logging.getLogger("autonomous_agent.curiosity")

_ACTIVE_STATUSES = (
    ThoughtStatus.NEW,
    ThoughtStatus.QUEUED,
    ThoughtStatus.THINKING,
    ThoughtStatus.INSIGHT_FOUND,
    ThoughtStatus.FOLLOW_UP,
    ThoughtStatus.WAITING,
    ThoughtStatus.DRAWER,
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _snippet(text: str, limit: int = 40) -> str:
    cleaned = " ".join(str(text).split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1] + "…"


class CuriosityEngine:
    def __init__(
        self,
        config: CuriosityConfig,
        repository: ThoughtRepository,
        memories: MemoryStore,
        topics: TopicStore,
        *,
        metrics=None,
        clock=utcnow,
    ) -> None:
        self._config = config
        self._repo = repository
        self._memories = memories
        self._topics = topics
        self._metrics = metrics
        self._clock = clock

    # ------------------------------------------------------------------ 入口
    async def detect(self, *, now: datetime | None = None) -> list[CuriosityCandidate]:
        """收集候选问题，按分数从高到低返回。"""
        if not self._config.enabled:
            return []
        moment = now or self._clock()
        active = await self._repo.list(statuses=_ACTIVE_STATUSES, limit=200)
        candidates: list[CuriosityCandidate] = []
        candidates.extend(await self._from_conflicts(active, moment))
        candidates.extend(await self._from_unverified(active, moment))
        candidates.extend(await self._from_idle_topics(active, moment))
        if not candidates:
            candidates.extend(await self._exploration(active, moment))
        scored: list[tuple[float, CuriosityCandidate]] = []
        seen: list[str] = []
        for candidate in candidates[: self._config.max_curiosity_events_per_tick * 3]:
            if any(
                similarity(candidate.title, other) >= self._config.duplicate_threshold
                for other in seen
            ):
                await self._metric("curiosity.duplicate")
                continue
            seen.append(candidate.title)
            score = curiosity_score(candidate, self._config).value
            if score < self._config.min_score:
                await self._metric("curiosity.below_threshold")
                continue
            scored.append((score, candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [candidate for _, candidate in scored]

    async def step(
        self,
        *,
        now: datetime | None = None,
        emotion: EmotionState | None = None,
    ) -> list[Thought]:
        """把高分候选落成 Thought（受每 tick 上限约束）。"""
        if not self._config.enabled:
            return []
        moment = now or self._clock()
        candidates = await self.detect(now=moment)
        if not candidates:
            return []
        limit = max(0, int(self._config.max_new_thoughts_per_tick))
        created: list[Thought] = []
        for candidate in candidates:
            if len(created) >= limit:
                break
            thought = await self._materialize(candidate, moment, emotion)
            if thought is not None:
                created.append(thought)
        return created

    # ------------------------------------------------------------------ 候选源
    async def _from_conflicts(
        self, active: list[Thought], moment: datetime
    ) -> list[CuriosityCandidate]:
        out: list[CuriosityCandidate] = []
        for memory in await self._memories.find_conflicts():
            title = f"关于「{_snippet(memory.content)}」有两种说法，哪一种更可信？"
            out.append(
                CuriosityCandidate(
                    kind="contradiction",
                    title=title,
                    content=(
                        "记忆里出现了互相矛盾的说法。我需要比较它们的来源、"
                        "时间和可信度，而不是随便选一个。"
                    ),
                    topic_name=str(memory.metadata.get("topic") or ""),
                    source_type="MEMORY_CONFLICT",
                    source_id=memory.id,
                    importance=0.7,
                    uncertainty=0.8,
                    knowledge_gap=0.8,
                    future_value=0.3,
                )
            )
        return out

    async def _from_unverified(
        self, active: list[Thought], moment: datetime
    ) -> list[CuriosityCandidate]:
        out: list[CuriosityCandidate] = []
        memories = await self._memories.list(types=[MemoryType.INFERENCE], limit=30)
        for memory in memories:
            if memory.confidence >= 0.6:
                continue
            idle_days = max(
                0.0,
                (moment - (memory.updated_at or memory.created_at)).total_seconds() / 86400,
            )
            staleness = _clamp(idle_days / max(1.0, self._config.novelty_days))
            out.append(
                CuriosityCandidate(
                    kind="unverified",
                    title=f"「{_snippet(memory.content)}」真的成立吗？",
                    content="这是一条置信度不高的推测，值得找证据确认，或者降级它。",
                    topic_name=str(memory.metadata.get("topic") or ""),
                    source_type="MEMORY",
                    source_id=memory.id,
                    importance=_clamp(0.4 + memory.importance * 0.4),
                    uncertainty=_clamp(1.0 - memory.confidence),
                    novelty=_clamp(0.5 + (1.0 - staleness) * 0.3),
                    staleness=staleness,
                    knowledge_gap=0.7,
                    future_value=0.2,
                )
            )
        return out

    async def _from_idle_topics(
        self, active: list[Thought], moment: datetime
    ) -> list[CuriosityCandidate]:
        out: list[CuriosityCandidate] = []
        limit = max(0, int(self._config.max_reactivation_requests))
        for topic in await self._topics.all(limit=self._config.max_topic_expansion):
            if topic.status not in (TopicStatus.LOW_ACTIVITY, TopicStatus.DORMANT):
                continue
            if topic.importance < 0.5:
                continue
            if len(out) >= limit:
                break
            idle_days = max(0.0, (moment - topic.last_activity_at).total_seconds() / 86400)
            staleness = _clamp(idle_days / max(1.0, self._config.novelty_days))
            out.append(
                CuriosityCandidate(
                    kind="topic",
                    title=f"「{topic.name}」已经很久没有推进了，还值得继续吗？",
                    content=(
                        "这个话题曾经重要，但最近没有新的思考或记忆。"
                        "我需要判断它是否仍然值得投入。"
                    ),
                    topic_name=topic.name,
                    source_type="TOPIC",
                    source_id=topic.id,
                    importance=_clamp(0.4 + topic.importance * 0.5),
                    uncertainty=0.5,
                    novelty=_clamp(0.4 + (1.0 - staleness) * 0.4),
                    staleness=staleness,
                    knowledge_gap=0.6,
                    future_value=_clamp(0.2 + topic.thought_count * 0.05),
                )
            )
        return out

    async def _exploration(
        self, active: list[Thought], moment: datetime
    ) -> list[CuriosityCandidate]:
        """没有线索时的温和探索：只挑一个已有话题换个角度问。"""
        if any(
            thought.metadata.get("curiosity_kind") and thought.revisit_count == 0
            for thought in active
        ):
            # 已经有一个 Curiosity 念头还没被想过，先别再造新的。
            return []
        out: list[CuriosityCandidate] = []
        topics = await self._topics.context_topics(limit=self._config.max_topic_expansion)
        for topic in topics[:1]:
            out.append(
                CuriosityCandidate(
                    kind="exploration",
                    title=f"如果重新设计「{topic.name}」，最关键的限制是什么？",
                    content=(
                        "没有新的外部线索。可以从已有话题里换一个角度，"
                        "但不要为思考而思考；想不出所以然就放弃。"
                    ),
                    topic_name=topic.name,
                    source_type="TOPIC_EXPLORATION",
                    source_id=topic.id,
                    importance=_clamp(0.3 + topic.importance * 0.3),
                    uncertainty=0.6,
                    knowledge_gap=0.5,
                    future_value=0.1,
                )
            )
        return out

    # ------------------------------------------------------------------ 落库
    async def _materialize(
        self,
        candidate: CuriosityCandidate,
        moment: datetime,
        emotion: EmotionState | None,
    ) -> Thought | None:
        existing = await self._repo.list(statuses=_ACTIVE_STATUSES, limit=200)
        best = 0.0
        for thought in existing:
            score = similarity(candidate.title, f"{thought.title}\n{thought.content}")
            best = max(best, score)
        if best >= self._config.duplicate_threshold:
            await self._metric("curiosity.duplicate")
            return None
        for thought in existing:
            if (
                thought.metadata.get("curiosity_kind") == candidate.kind
                and thought.revisit_count == 0
                and similarity(candidate.title, thought.title) >= 0.6
            ):
                await self._metric("curiosity.loop_blocked")
                return None

        depth = 0
        if candidate.source_type == "THOUGHT":
            depth = 1
        curiosity_level = _clamp(
            (emotion.curiosity if emotion is not None else 0.3) * 0.5 + 0.5
        )
        thought = Thought.create(
            type=ThoughtType.CURIOSITY,
            title=candidate.title[:200],
            content=candidate.content,
            source=f"curiosity:{candidate.kind}",
            importance=_clamp(candidate.importance),
            curiosity=curiosity_level,
            relevance=_clamp(0.5 + candidate.relevance * 0.5),
            confidence=0.35,
            related_topics=(candidate.topic_name,) if candidate.topic_name else (),
            now=moment,
        ).with_changes(
            metadata={
                "curiosity_kind": candidate.kind,
                "curiosity_source_type": candidate.source_type,
                "curiosity_source_id": candidate.source_id,
                "curiosity_depth": depth,
                "topic": candidate.topic_name,
            }
        )
        created, merged = await self._repo.create(thought)
        if merged:
            await self._metric("curiosity.merged")
            return None
        await self._metric("curiosity.created")
        await self._link_topic(created, candidate.topic_name, moment)
        logger.info("Curiosity 产生 Thought %s（%s）", created.id, candidate.kind)
        return created

    async def _link_topic(
        self, thought: Thought, name: str, moment: datetime
    ) -> None:
        if not name:
            return
        topic = await self._topics.touch(name, importance=thought.importance, now=moment)
        await self._topics.record_thought(topic, thought_id=thought.id, now=moment)
        tags = tuple(dict.fromkeys((*thought.related_topics, topic.name)))[:8]
        await self._repo.save(thought.with_changes(related_topics=tags))

    async def _metric(self, name: str, **fields) -> None:
        if self._metrics is None:
            return
        try:
            await self._metrics.record(name, 1.0, dict(fields) or None)
        except Exception:  # noqa: BLE001 - 指标失败不影响主流程
            logger.debug("记录指标 %s 失败", name, exc_info=True)


__all__ = ["CuriosityEngine"]