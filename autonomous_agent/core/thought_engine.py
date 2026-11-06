"""自主思考引擎。

一次"思考"是程序编排的固定流程：

    AUTONOMOUS_TICK
      → ThoughtRepository 选出值得想的 Thought（程序评分）
      → 状态 QUEUED/WAITING/... → THINKING
      → 组装上下文（记忆、话题、最近对话、情绪）
      → LLM 输出 THINK JSON
      → Schema Validator → Thought Validator
      → 程序按 disposition 落地（记忆/日记/抽屉/联系意图）
      → Thought 进入新状态

LLM 只负责"想到什么"，不负责"要不要发""发去哪""能不能发"。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from ..config import AppConfig
from ..curiosity.engine import CuriosityEngine
from ..diary.repository import DiaryStore
from ..llm.adapter import (
    LLMAdapter,
    LLMError,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
)
from ..llm.budget import TokenBudget
from ..llm.prompts import render_think_system, render_think_user
from ..llm.schemas import ActionType
from ..llm.validator import OutputValidator, ValidationContext
from ..memory.models import Memory, MemoryStatus, MemoryType
from ..memory.reactivation import try_reactivate as try_reactivate_memories
from ..memory.store import MemoryLimitError, MemoryStore
from ..memory.topics import TopicStore
from ..memory.validator import (
    MemoryProposal,
    MemoryValidationError,
    MemoryValidator,
    text_keywords,
)
from ..contact.manager import ContactDecision, ContactManager
from ..database.database import MessageRepository
from ..thought.forgetting import ForgetResult, sweep_forgetting
from ..thought.models import (
    Disposition,
    NewQuestion,
    Thought,
    ThoughtStatus,
    ThoughtType,
    utcnow,
)
from ..thought.reactivation import try_reactivate as try_reactivate_thoughts
from ..thought.repository import ThoughtLimitError, ThoughtRepository
from ..thought.scoring import rank_thoughts
from ..thought.similarity import similarity
from ..thought.thinking_text import (
    ThinkingState,
    compress_thinking_text,
    parse_thinking_text,
    render_thinking_text,
    state_from_thought,
)
from ..thought.validator import ThoughtDecisionError, ThoughtOutcome, ThoughtValidator
from .emotion import EmotionModel, EmotionState

logger = logging.getLogger(__name__)

_QUESTION_MARKERS = ("?", "？", "为什么", "怎么", "是不是", "吗", "什么", "如何", "能不能")


@dataclass(frozen=True, slots=True)
class ContactIntent:
    thought_id: str
    conversation_id: str
    message: str
    urgency: float


@dataclass(frozen=True, slots=True)
class ThinkResult:
    kind: str  # "thought" | "nothing" | "error"
    thought: Thought | None = None
    outcome: ThoughtOutcome | None = None
    contact: ContactIntent | None = None
    denied_reason: str | None = None
    error: str | None = None
    skipped_reason: str | None = None
    forgot: tuple[ForgetResult, ...] = ()


class ThoughtEngine:
    def __init__(
        self,
        config: AppConfig,
        *,
        repository: ThoughtRepository,
        memories: MemoryStore,
        topics: TopicStore,
        diary: DiaryStore,
        contact: ContactManager,
        messages: MessageRepository,
        emotion_model: EmotionModel,
        metrics: Any = None,
        clock=utcnow,
        budget: TokenBudget | None = None,
    ) -> None:
        self._config = config
        self._cfg = config.thought
        self._repo = repository
        self._memories = memories
        self._topics = topics
        self._diary = diary
        self._contact = contact
        self._messages = messages
        self._emotion_model = emotion_model
        self._metrics = metrics
        self._clock = clock
        self._budget = budget
        self._budget_notified = False
        self._validator = ThoughtValidator(
            self._cfg,
            max_memory_candidates=self._config.memory.max_candidates_per_think,
        )
        self._memory_validator = MemoryValidator(self._config.memory)
        self._output = OutputValidator()
        self._curiosity = CuriosityEngine(
            self._config.curiosity,
            repository,
            memories,
            topics,
            metrics=metrics,
            clock=clock,
        )
        self._ticks = 0
        self._memory_ticks = 0
        self._emotion: EmotionState = emotion_model.initial()

    # ------------------------------------------------------------------ 情绪
    @property
    def emotion(self) -> EmotionState:
        return self._emotion

    def set_emotion(self, state: EmotionState) -> None:
        self._emotion = self._emotion_model.decay(state, now=self._clock())

    # ------------------------------------------------------------------ 摄取
    async def ingest_user_message(
        self,
        *,
        conversation_id: str,
        user_id: str,
        content: str,
        now: datetime | None = None,
    ) -> Thought | None:
        """用户消息进来：记记忆、补话题、必要时生成一条 Thought、尝试唤醒抽屉。"""
        moment = now or self._clock()
        if self._config.emotion.enabled:
            self._emotion = self._emotion_model.on_user_message(self._emotion, now=moment)

        await self._remember(
            Memory.create(
                type=MemoryType.EVENT,
                content=f"{user_id}: {content}",
                confidence=0.8,
                source=f"user:{user_id}",
                source_type="USER_MESSAGE",
                source_id=user_id,
                importance=0.4,
                keywords=_keywords(content),
                now=moment,
            )
        )
        if self._config.memory.enabled:
            revived_memories = await try_reactivate_memories(
                self._memories,
                content,
                self._config.memory,
                now=moment,
                metrics=self._metrics,
            )
            if revived_memories:
                await self._awaken_thoughts_for(revived_memories, moment)
        if not self._cfg.enabled:
            return None

        revived = await try_reactivate_thoughts(
            self._repo, content, self._cfg, now=moment, metrics=self._metrics
        )
        if revived:
            logger.info("新信息唤醒了 %s 条旧想法", len(revived))

        if not self._cfg.from_user_messages or not _looks_thoughtworthy(content):
            return None
        thought = Thought.create(
            type=ThoughtType.QUESTION,
            title=content.strip()[:60],
            content=content.strip(),
            source="USER_MESSAGE",
            importance=0.5,
            curiosity=0.5,
            now=moment,
        )
        construction = await self._create(thought)
        return construction[0] if construction else None

    # ------------------------------------------------------------------ 思考
    async def step(
        self,
        llm: LLMAdapter,
        *,
        conversation_id: str | None = None,
        now: datetime | None = None,
        silent: bool = False,
    ) -> ThinkResult:
        moment = now or self._clock()
        if not self._cfg.enabled:
            return ThinkResult("nothing", skipped_reason="thought_disabled")
        if self._budget is not None and not self._budget.allow():
            # 预算用完就别往下走了：状态流转、维护扫描都不产生价值。
            if not self._budget_notified:
                self._budget_notified = True
                await self._metric("llm.budget_skipped", 1.0, {"route": "think"})
                logger.warning("今日 token 预算已用完，暂停自主思考")
            return ThinkResult("nothing", skipped_reason="budget_exhausted")
        self._budget_notified = False
        self._ticks += 1
        forgot: tuple[ForgetResult, ...] = ()
        if self._cfg.maintenance_every_ticks > 0 and self._ticks % self._cfg.maintenance_every_ticks == 0:
            forgot = tuple(
                await sweep_forgetting(
                    self._repo, self._cfg, now=moment, metrics=self._metrics
                )
            )
        await self._maintain_topics(moment)
        await self._maintain_memories(moment)

        candidate = await self._pick(moment)
        if candidate is None:
            # 有念头在排队/等下次思考时，先让它们按自己的节奏走，别急着造新题。
            pending = await self._repo.count(
                [ThoughtStatus.QUEUED, ThoughtStatus.WAITING]
            )
            if pending == 0:
                # 真没存量念头了，才让 Curiosity 试着提一个值得探索的问题。
                created = await self._curiosity.step(now=moment, emotion=self._emotion)
                candidate = created[0] if created else None
        if candidate is None:
            return ThinkResult("nothing", skipped_reason="no candidate", forgot=forgot)

        thought = candidate
        try:
            if thought.status is not ThoughtStatus.QUEUED:
                thought = await self._repo.transition(
                    thought, ThoughtStatus.QUEUED, reason="scheduled", now=moment
                )
            thought = await self._repo.transition(
                thought, ThoughtStatus.THINKING, reason="thinking", now=moment
            )
        except Exception as exc:  # noqa: BLE001 - 状态异常不应拖垮 tick
            logger.warning("Thought %s 无法进入 THINKING: %s", thought.id, exc)
            return ThinkResult("error", thought=thought, error="STATE_ERROR", forgot=forgot)

        target_conversation = (
            conversation_id
            or str(thought.metadata.get("conversation_id") or "")
            or await self._latest_conversation()
        )
        state = self._read_state(thought)
        request = await self._build_request(thought, moment, target_conversation, state)
        try:
            response = await asyncio.wait_for(
                llm.complete(request), timeout=self._config.llm.request_timeout_seconds
            )
        except asyncio.TimeoutError:
            return await self._fail(thought, "LLM_TIMEOUT", moment, forgot)
        except LLMTimeoutError:
            return await self._fail(thought, "LLM_TIMEOUT", moment, forgot)
        except LLMError as exc:
            return await self._fail(thought, f"LLM_ERROR:{type(exc).__name__}", moment, forgot)

        # 先记账再校验：想法没通过校验也是花了钱的，不能漏掉。
        await self._record_tokens(request, response)

        ctx = ValidationContext(
            event_type="AUTONOMOUS_TICK",
            conversation_id=target_conversation or None,
            known_conversation_ids=frozenset(
                {target_conversation} if target_conversation else set()
            ),
            allowed_actions=frozenset({ActionType.THINK.value}),
        )
        decision, error = self._output.safe_validate(response.text, ctx)
        if error is not None:
            return await self._fail(thought, error.code, moment, forgot)

        try:
            outcome = self._validator.validate(decision, thought=thought)
        except ThoughtDecisionError as exc:
            return await self._fail(thought, exc.code, moment, forgot)

        result = await self._apply(
            outcome, thought, target_conversation, moment, forgot, state, silent=silent
        )
        if result.kind == "thought" and self._config.emotion.enabled:
            self._emotion = self._emotion_model.on_thought_completed(
                self._emotion, now=moment
            )
        return result

    # ------------------------------------------------------------------ 内部
    async def has_pending_work(self, now: datetime | None = None) -> bool:
        """还有念头可以立刻想吗？调度器据此决定用快节奏还是慢心跳。"""
        if not self._cfg.enabled:
            return False
        if self._budget is not None and not self._budget.allow():
            # 没预算的时候没必要 60 秒醒一次，回到慢心跳。
            return False
        moment = now or self._clock()
        return bool(await self._repo.candidates(moment, limit=1))

    async def _pick(self, moment: datetime) -> Thought | None:
        candidates = await self._repo.candidates(moment, limit=50)
        if not candidates:
            return None
        ranked = rank_thoughts(candidates, moment, config=self._cfg)
        top = ranked[0]
        # 分数太低就别硬想，留给维护/遗忘。
        if len(ranked) > 1 and top.importance < 0.05 and top.curiosity < 0.05:
            return None
        return top

    def _read_state(self, thought: Thought) -> ThinkingState:
        """读回上次的思考文本；读不到就从 Thought 的结论字段重建。"""
        try:
            text = self._repo.workspace.read_thinking_text(thought)
        except Exception as exc:  # noqa: BLE001 - 文件坏了不该挡住思考
            logger.warning("读取思考文本失败 %s: %s", thought.id, exc)
            text = None
        state = parse_thinking_text(text) if text else state_from_thought(thought)
        if state.turns <= 0:
            state.turns = thought.revisit_count
        return state

    async def _build_request(
        self,
        thought: Thought,
        moment: datetime,
        conversation_id: str | None,
        state: ThinkingState,
    ) -> LLMRequest:
        memories = await self._memories.search(
            f"{thought.title}\n{thought.content}",
            limit=self._cfg.memory_limit,
        )
        topics = [await self._topics.get(name) for name in thought.related_topics]
        topics = [topic for topic in topics if topic is not None]
        recent = (
            await self._messages.recent(conversation_id, limit=self._cfg.history_limit)
            if conversation_id
            else []
        )
        thinking_text = render_thinking_text(thought, state, now=moment)
        body = render_think_user(
            thought,
            now=moment,
            memories=memories,
            topics=topics,
            recent_messages=recent,
            emotion=self._emotion,
            thinking_text=thinking_text,
        )
        return LLMRequest(
            system_prompt=render_think_system(),
            messages=[LLMMessage(role="user", content=body)],
            metadata={"thought_id": thought.id, "kind": "think"},
        )

    async def _apply(
        self,
        outcome: ThoughtOutcome,
        thought: Thought,
        conversation_id: str | None,
        moment: datetime,
        forgot: tuple[ForgetResult, ...],
        state: ThinkingState,
        *,
        silent: bool = False,
    ) -> ThinkResult:
        next_check: datetime | None = None
        if outcome.next_check_minutes:
            next_check = moment + timedelta(minutes=outcome.next_check_minutes)
        elif outcome.disposition is Disposition.WAIT:
            next_check = moment + timedelta(minutes=60)
        elif outcome.disposition is Disposition.CONTINUE:
            # 接着刚才那步想：给一个短间隔，别让思考链断掉。
            next_check = moment + timedelta(minutes=self._cfg.continue_think_minutes)

        follow_up = outcome.disposition is Disposition.CONTINUE
        new_state = self._next_state(state, outcome, moment)
        changes: dict[str, Any] = {
            "insight": outcome.insight,
            "uncertainty": outcome.uncertainty,
            "revisit_count": thought.revisit_count + 1,
            "last_thought_at": moment,
            "next_check_at": next_check,
            "next_action": "THINK_AGAIN" if follow_up else outcome.disposition.value,
            "confidence": outcome.confidence,
            "metadata": {**dict(thought.metadata), "thinking": new_state.to_dict()},
        }
        try:
            updated = await self._repo.transition(
                thought,
                outcome.target_status,
                reason=f"think:{outcome.disposition.value}",
                changes=changes,
                now=moment,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Thought %s 落地状态失败: %s", thought.id, exc)
            return await self._fail(thought, "APPLY_FAILED", moment, forgot)

        await self._spawn_questions(updated, outcome.new_questions, moment)
        await self._touch_topics(updated, moment)

        try:
            text = render_thinking_text(updated, new_state, now=moment)
            text = compress_thinking_text(
                text, max_chars=self._cfg.thinking_text_max_chars
            )
            self._repo.workspace.write_thinking_text(updated, text)
        except Exception as exc:  # noqa: BLE001 - 文件失败以数据库为准
            logger.warning("写思考文本失败 %s: %s", updated.id, exc)
            await self._metric("thought.text_error", 1.0, {"thought_id": updated.id})

        if outcome.memory_candidates:
            updated = await self._save_memory_candidates(
                updated, outcome.memory_candidates, moment
            )

        if outcome.disposition in (Disposition.KEEP, Disposition.DIARY, Disposition.SPEAK, Disposition.FOLLOW_UP):
            remembered = await self._remember(
                Memory.create(
                    type=MemoryType.THOUGHT,
                    content=outcome.insight,
                    confidence=max(0.3, outcome.confidence),
                    source=f"thought:{updated.id}",
                    source_type="THOUGHT",
                    source_id=updated.id,
                    importance=updated.importance,
                    keywords=_keywords(outcome.insight),
                    now=moment,
                )
            )
            if remembered is not None:
                updated = await self._link_memory(updated, remembered, moment)

        if outcome.disposition is Disposition.DIARY and self._config.diary.enabled:
            entry = outcome.insight
            if outcome.uncertainty:
                entry = (
                    f"{entry}\n（以上是内部推断，尚未确认：{outcome.uncertainty}）"
                )
            try:
                await self._diary.append(entry, now=moment)
            except Exception as exc:  # noqa: BLE001 - 日记失败不影响思考
                logger.warning("写日记失败: %s", exc)

        if outcome.disposition is Disposition.SPEAK and outcome.contact_message:
            target = conversation_id or await self._latest_conversation()
            if not target:
                return ThinkResult(
                    "thought",
                    thought=updated,
                    outcome=outcome,
                    denied_reason="NO_CONVERSATION",
                    forgot=forgot,
                )
            decision: ContactDecision = await self._contact.evaluate(
                updated, target, outcome.contact_message, silent=silent, now=moment
            )
            if decision.allowed:
                return ThinkResult(
                    "thought",
                    thought=updated,
                    outcome=outcome,
                    contact=ContactIntent(
                        updated.id, target, outcome.contact_message, decision.urgency
                    ),
                    forgot=forgot,
                )
            await self._contact.record(
                updated,
                target,
                outcome.contact_message,
                status="DENIED",
                reason=decision.code,
                now=moment,
            )
            if self._config.emotion.enabled:
                self._emotion = self._emotion_model.on_contact_denied(self._emotion, now=moment)
            await self._metric("contact.denied", 1.0, {"code": decision.code})
            return ThinkResult(
                "thought",
                thought=updated,
                outcome=outcome,
                denied_reason=decision.code,
                forgot=forgot,
            )

        return ThinkResult("thought", thought=updated, outcome=outcome, forgot=forgot)

    def _next_state(
        self, state: ThinkingState, outcome: ThoughtOutcome, moment: datetime
    ) -> ThinkingState:
        history = list(state.history)
        history.append(
            f"{moment:%Y-%m-%d %H:%M} 第{state.turns + 1}次：{outcome.insight[:80]}"
        )
        history = history[-self._cfg.history_keep_turns :]
        known = _dedupe([*state.known, *outcome.known])[-8:]
        uncertainty = _dedupe(
            [*state.uncertainty, *([outcome.uncertainty] if outcome.uncertainty else [])]
        )[-5:]
        questions = [item.title for item in outcome.new_questions] or state.new_questions
        continuing = outcome.disposition is Disposition.CONTINUE
        resolved = outcome.disposition is Disposition.RESOLVED
        return ThinkingState(
            known=known,
            current_thinking=outcome.insight,
            insight=outcome.insight,
            uncertainty=uncertainty,
            new_questions=questions[-5:],
            related=list(state.related),
            next_step=outcome.next_step
            or ("继续思考" if continuing else outcome.disposition.value),
            continue_condition="还有待回答的问题，下一次 Tick 接着想" if continuing else "",
            stop_reason=(
                "已经想清楚，可以收尾"
                if resolved
                else ("" if continuing else f"按 disposition={outcome.disposition.value} 收尾")
            ),
            last_decision=outcome.disposition.value,
            turns=state.turns + 1,
            history=history,
        )

    async def _depth(self, thought: Thought) -> int:
        depth = 0
        current = thought
        seen: set[str] = set()
        while current.parent_thought_id and depth < 10:
            if current.id in seen:
                break
            seen.add(current.id)
            parent = await self._repo.get(current.parent_thought_id)
            if parent is None:
                break
            depth += 1
            current = parent
        return depth

    async def _ancestor_titles(self, thought: Thought) -> list[str]:
        titles: list[str] = []
        current = thought
        seen: set[str] = set()
        while current.parent_thought_id and len(titles) < 10:
            if current.id in seen:
                break
            seen.add(current.id)
            parent = await self._repo.get(current.parent_thought_id)
            if parent is None:
                break
            titles.append(parent.title)
            current = parent
        return titles

    async def _spawn_questions(
        self, parent: Thought, questions: tuple[NewQuestion, ...], moment: datetime
    ) -> None:
        if not questions:
            return
        if await self._depth(parent) >= self._cfg.max_thought_depth:
            await self._metric("thought.depth_limited", 1.0, {"thought_id": parent.id})
            return
        existing = await self._repo.children_of(parent.id)
        remaining = self._cfg.max_followups_per_thought - len(existing)
        if remaining <= 0:
            return
        ancestor_titles = await self._ancestor_titles(parent)
        for question in questions[:remaining]:
            if similarity(question.title, parent.title) >= 0.8:
                await self._metric("thought.cycle_skipped", 1.0, {"thought_id": parent.id})
                continue
            if any(similarity(question.title, title) >= 0.85 for title in ancestor_titles):
                await self._metric("thought.cycle_skipped", 1.0, {"thought_id": parent.id})
                continue
            child = Thought.create(
                type=ThoughtType.QUESTION,
                title=question.title,
                content=question.content,
                source=f"thought:{parent.id}",
                parent_thought_id=parent.id,
                importance=question.importance,
                curiosity=question.curiosity,
                related_topics=parent.related_topics,
                now=moment,
            )
            created = await self._create(child)
            if created and not created[1]:
                try:
                    await self._repo.add_relation(parent.id, created[0].id, "CHILD")
                except Exception:  # noqa: BLE001
                    pass

    async def _create(self, thought: Thought) -> tuple[Thought, bool] | None:
        try:
            return await self._repo.create(thought)
        except ThoughtLimitError as exc:
            logger.warning("Thought 数量达上限: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001 - 落库失败不该让 tick 崩
            logger.warning("创建 Thought 失败: %s", exc)
            return None

    async def _remember(self, memory: Memory) -> Memory | None:
        if not self._config.memory.enabled:
            return None
        try:
            return await self._memories.add(memory)
        except MemoryLimitError as exc:
            logger.info("记忆已达上限，跳过: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("写记忆失败: %s", exc)
        return None

    async def _save_memory_candidates(
        self,
        thought: Thought,
        raw: tuple[Any, ...],
        moment: datetime,
    ) -> Thought:
        """把 LLM 的 SAVE_MEMORY 意图交付 Memory Validator 裁决。"""
        if not self._config.memory.enabled:
            return thought
        try:
            candidates = self._memory_validator.parse_candidates(raw)
        except MemoryValidationError as exc:
            logger.info("记忆候选不合法: %s", exc)
            await self._metric("memory.invalid_candidate", 1.0, {"thought_id": thought.id})
            return thought
        for candidate in candidates:
            try:
                proposal: MemoryProposal = await self._memory_validator.evaluate(
                    candidate,
                    self._memories,
                    source_type="THOUGHT",
                    source_id=thought.id,
                    now=moment,
                )
            except Exception as exc:  # noqa: BLE001 - 单条失败不影响思考
                logger.warning("评估记忆候选失败: %s", exc)
                continue
            await self._metric(
                "memory.proposed",
                1.0,
                {"action": proposal.action, "thought_id": thought.id},
            )
            if proposal.action == "DO_NOT_SAVE":
                continue
            try:
                if proposal.action == "UPDATE" and proposal.target is not None:
                    memory = await self._merge_memory(proposal, moment)
                elif proposal.action == "CONFLICT" and proposal.target is not None:
                    await self._memories.mark_conflict(proposal.target)
                    memory = await self._create_memory(proposal, thought, moment)
                    if memory is not None:
                        await self._memories.add_relation(
                            memory.id, proposal.target.id, "CONTRADICTS"
                        )
                else:
                    memory = await self._create_memory(proposal, thought, moment)
            except Exception as exc:  # noqa: BLE001
                logger.warning("落地记忆候选失败: %s", exc)
                continue
            if memory is not None:
                thought = await self._link_memory(thought, memory, moment)
        return thought

    async def _create_memory(
        self, proposal: MemoryProposal, thought: Thought, moment: datetime
    ) -> Memory | None:
        candidate = proposal.candidate
        memory = Memory.create(
            type=candidate.type,
            content=candidate.content,
            confidence=candidate.confidence,
            importance=candidate.importance,
            relevance=candidate.relevance if candidate.relevance is not None else 0.5,
            source=f"thought:{thought.id}",
            source_type="THOUGHT",
            source_id=thought.id,
            keywords=candidate.keywords,
            now=moment,
        )
        saved = await self._remember(memory)
        if saved is None:
            return None
        await self._metric("memory.created", 1.0, {"memory_id": saved.id})
        return saved

    async def _merge_memory(
        self, proposal: MemoryProposal, moment: datetime
    ) -> Memory | None:
        target = proposal.target
        if target is None:
            return None
        candidate = proposal.candidate
        merged = target.with_changes(
            confidence=max(target.confidence, candidate.confidence),
            importance=max(target.importance, candidate.importance),
            frequency=target.frequency + 1,
            usage_count=target.usage_count + 1,
            last_used_at=moment,
            last_verified_at=moment,
            keywords=tuple(dict.fromkeys([*target.keywords, *candidate.keywords]))[:12],
            metadata={
                **dict(target.metadata),
                "merged_count": int(target.metadata.get("merged_count", 0)) + 1,
            },
        )
        await self._memories.save(merged)
        await self._metric("memory.updated", 1.0, {"memory_id": merged.id})
        return merged

    async def _link_memory(
        self, thought: Thought, memory: Memory, moment: datetime
    ) -> Thought:
        """Thought ↔ Memory 双向关系。"""
        try:
            await self._repo.add_relation(thought.id, memory.id, "MEMORY")
        except Exception as exc:  # noqa: BLE001
            logger.debug("写 Thought-Memory 关系失败: %s", exc)
        if memory.id in thought.related_memories:
            return thought
        links = tuple(dict.fromkeys([*thought.related_memories, memory.id]))[-12:]
        updated = thought.with_changes(related_memories=links, updated_at=moment)
        try:
            await self._repo.save(updated)
        except Exception as exc:  # noqa: BLE001
            logger.debug("更新 related_memories 失败: %s", exc)
            return thought
        return updated

    async def _awaken_thoughts_for(
        self, memories: list[Memory], moment: datetime
    ) -> None:
        """Memory 被重新激活时，把引用它的旧 Thought 也拉回队列。"""
        for memory in memories:
            try:
                thoughts = await self._repo.referencing_memory(memory.id, limit=5)
            except Exception as exc:  # noqa: BLE001
                logger.debug("查引用记忆的 Thought 失败: %s", exc)
                continue
            for thought in thoughts:
                if thought.status not in (
                    ThoughtStatus.DRAWER,
                    ThoughtStatus.EXPIRED,
                    ThoughtStatus.ARCHIVED,
                ):
                    continue
                try:
                    await self._repo.reactivate(
                        thought,
                        reason=f"memory:{memory.id}",
                        now=moment,
                        config=self._cfg,
                    )
                    await self._metric(
                        "thought.awakened_by_memory",
                        1.0,
                        {"thought_id": thought.id, "memory_id": memory.id},
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("唤醒 Thought 失败 %s: %s", thought.id, exc)

    async def _touch_topics(self, thought: Thought, moment: datetime) -> None:
        """思考推进时刷新相关话题的活跃度；归档话题不自动复活。"""
        if not self._config.topic.enabled:
            return
        names = list(thought.related_topics)[:2]
        if not names and thought.metadata.get("topic"):
            names = [str(thought.metadata["topic"])]
        for name in names:
            try:
                topic = await self._topics.touch(
                    name, importance=thought.importance, now=moment
                )
                if thought.revisit_count <= 1:
                    await self._topics.record_thought(
                        topic, thought_id=thought.id, now=moment
                    )
            except Exception as exc:  # noqa: BLE001
                logger.debug("刷新话题失败 %s: %s", name, exc)

    async def _maintain_topics(self, moment: datetime) -> None:
        """话题闲置太久就降级/归档，避免长期占用上下文。"""
        if not self._config.topic.enabled:
            return
        try:
            changed = await self._topics.sweep(now=moment)
        except Exception as exc:  # noqa: BLE001
            logger.warning("话题维护失败: %s", exc)
            return
        for topic in changed:
            await self._metric(
                "topic.demoted",
                1.0,
                {"topic": topic.name, "status": topic.status.value},
            )

    async def _maintain_memories(self, moment: datetime) -> None:
        cfg = self._config.memory
        if not cfg.enabled or cfg.consolidate_every_ticks <= 0:
            return
        self._memory_ticks += 1
        if self._memory_ticks % cfg.consolidate_every_ticks != 0:
            return
        try:
            swept = await self._memories.sweep(now=moment)
            if swept:
                await self._metric("memory.swept", float(len(swept)))
            merged = await self._memories.consolidate(now=moment)
            if merged:
                await self._metric("memory.consolidated", float(len(merged)))
        except Exception as exc:  # noqa: BLE001 - 维护失败不影响思考
            logger.warning("记忆维护失败: %s", exc)

    async def _fail(
        self,
        thought: Thought,
        code: str,
        moment: datetime,
        forgot: tuple[ForgetResult, ...],
    ) -> ThinkResult:
        await self._metric("thought.error", 1.0, {"code": code, "thought_id": thought.id})
        if self._config.emotion.enabled:
            self._emotion = self._emotion_model.on_thought_blocked(
                self._emotion, now=moment
            )
        try:
            await self._repo.transition(
                thought,
                ThoughtStatus.QUEUED,
                reason=f"retry:{code}",
                now=moment,
                changes={
                    "next_check_at": moment + timedelta(minutes=5),
                    "metadata": {
                        **dict(thought.metadata),
                        "last_error": code,
                    },
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("回退 Thought 失败: %s", exc)
        return ThinkResult("error", thought=thought, error=code, forgot=forgot)

    async def _latest_conversation(self) -> str | None:
        ids = await self._messages.conversation_ids()
        if not ids:
            return None
        return sorted(ids)[-1]

    async def _record_tokens(self, request: LLMRequest, response: LLMResponse) -> None:
        """把这次调用的 token 记进预算，顺手记一条指标。"""
        if self._budget is None:
            return
        try:
            used = self._budget.record_response(request, response)
        except Exception:  # noqa: BLE001 - 记账失败不该挡思考
            logger.warning("token 记账失败", exc_info=True)
            return
        await self._metric("llm.tokens", float(used), {"route": "think"})

    async def _metric(self, name: str, value: float, tags: dict[str, Any] | None = None) -> None:
        if self._metrics is None:
            return
        try:
            await self._metrics.record(name, value, tags or {})
        except Exception:  # noqa: BLE001
            pass


def _keywords(text: str, limit: int = 8) -> tuple[str, ...]:
    return text_keywords(text, limit=limit)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = value.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _looks_thoughtworthy(text: str) -> bool:
    value = text.strip()
    if not value:
        return False
    return any(marker in value for marker in _QUESTION_MARKERS)


__all__ = ["ContactIntent", "ThinkResult", "ThoughtEngine"]