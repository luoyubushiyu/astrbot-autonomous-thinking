"""Topic：长期关注的主题。

Topic ≠ Thought：

- Topic 是"长期关注什么"（游戏、编程、某个长期项目）；
- Thought 是"正在思考什么"（为什么自主思考系统容易无限递归）。

Topic 的活跃度与生命周期由程序计算，LLM 不能直接改状态。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Mapping

from ..config import TopicConfig
from ..database.database import Database
from .models import utcnow


class TopicStatus(str, Enum):
    NEW = "NEW"
    ACTIVE = "ACTIVE"
    INTERESTING = "INTERESTING"
    LOW_ACTIVITY = "LOW_ACTIVITY"
    DORMANT = "DORMANT"
    ARCHIVED = "ARCHIVED"

    @classmethod
    def parse(cls, value: Any) -> "TopicStatus":
        if isinstance(value, TopicStatus):
            return value
        text = str(value or "").strip().upper()
        try:
            return cls(text)
        except ValueError as exc:
            raise ValueError(f"未知的 Topic 状态: {value!r}") from exc


# 程序允许的状态迁移；没有列出的方向一律拒绝。
TOPIC_TRANSITIONS: dict[TopicStatus, frozenset[TopicStatus]] = {
    TopicStatus.NEW: frozenset(
        {TopicStatus.ACTIVE, TopicStatus.LOW_ACTIVITY, TopicStatus.ARCHIVED}
    ),
    TopicStatus.ACTIVE: frozenset(
        {
            TopicStatus.INTERESTING,
            TopicStatus.LOW_ACTIVITY,
            TopicStatus.DORMANT,
            TopicStatus.ARCHIVED,
        }
    ),
    TopicStatus.INTERESTING: frozenset(
        {
            TopicStatus.ACTIVE,
            TopicStatus.LOW_ACTIVITY,
            TopicStatus.DORMANT,
            TopicStatus.ARCHIVED,
        }
    ),
    TopicStatus.LOW_ACTIVITY: frozenset(
        {TopicStatus.ACTIVE, TopicStatus.DORMANT, TopicStatus.ARCHIVED}
    ),
    TopicStatus.DORMANT: frozenset({TopicStatus.ACTIVE, TopicStatus.ARCHIVED}),
    TopicStatus.ARCHIVED: frozenset({TopicStatus.ACTIVE}),
}


class TopicLifecycleError(ValueError):
    def __init__(self, source: TopicStatus, target: TopicStatus) -> None:
        super().__init__(f"Topic 不能从 {source.value} 转到 {target.value}")


@dataclass(frozen=True, slots=True)
class Topic:
    id: str
    name: str
    keywords: tuple[str, ...] = ()
    importance: float = 0.5
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    status: TopicStatus = TopicStatus.NEW
    last_activity_at: datetime | None = None
    thought_count: int = 0
    memory_count: int = 0

    def with_changes(self, **changes: Any) -> "Topic":
        from dataclasses import replace

        return replace(self, **changes)

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "keywords": json.dumps(list(self.keywords), ensure_ascii=False),
            "importance": float(self.importance),
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
            "metadata": json.dumps(dict(self.metadata), ensure_ascii=False),
            "status": self.status.value,
            "last_activity_at": _iso(self.last_activity_at) if self.last_activity_at else None,
            "thought_count": int(self.thought_count),
            "memory_count": int(self.memory_count),
        }


def _iso(value: datetime) -> str:
    return value.isoformat()


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _json_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    if isinstance(value, str) and value:
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return ()
        return tuple(str(item) for item in loaded) if isinstance(loaded, list) else ()
    return ()


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value:
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}
    return {}


def _normalize(text: str) -> str:
    keep = [ch for ch in text.lower().strip() if ch.isalnum()]
    return "".join(keep)[:200]


def _bigrams(text: str) -> set[str]:
    value = _normalize(text)
    if len(value) < 2:
        return {value} if value else set()
    return {value[i : i + 2] for i in range(len(value) - 1)}


def name_similarity(left: str, right: str) -> float:
    """Topic 名称相似度：字符 bigram 的 Jaccard，够用且不依赖模型。"""
    a = _bigrams(left)
    b = _bigrams(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class TopicStore:
    def __init__(
        self,
        db: Database,
        *,
        clock=utcnow,
        config: TopicConfig | None = None,
    ) -> None:
        self._db = db
        self._clock = clock
        self._config = config or TopicConfig()

    # ---------------------------------------------------------------- 查询
    async def all(
        self,
        *,
        limit: int = 20,
        statuses: Iterable[TopicStatus] | None = None,
    ) -> list[Topic]:
        if statuses is not None:
            wanted = [s.value if isinstance(s, TopicStatus) else str(s) for s in statuses]
            if not wanted:
                return []
            placeholders = ", ".join("?" for _ in wanted)
            rows = await self._db.fetch_all(
                "SELECT * FROM topics WHERE status IN ("
                + placeholders
                + ") ORDER BY importance DESC, last_activity_at DESC LIMIT ?",
                (*wanted, int(limit)),
            )
        else:
            rows = await self._db.fetch_all(
                "SELECT * FROM topics ORDER BY importance DESC, updated_at DESC LIMIT ?",
                (int(limit),),
            )
        return [self._from_row(row) for row in rows]

    async def context_topics(self, *, limit: int | None = None) -> list[Topic]:
        """只把仍有活力的话题放进上下文，休眠/归档的不占位置。"""
        return await self.all(
            limit=limit or self._config.max_context_topics,
            statuses=(
                TopicStatus.ACTIVE,
                TopicStatus.INTERESTING,
                TopicStatus.NEW,
            ),
        )

    async def get(self, name: str) -> Topic | None:
        row = await self._db.fetch_one(
            "SELECT * FROM topics WHERE name = ?", (name.strip(),)
        )
        return self._from_row(row) if row else None

    async def get_by_id(self, topic_id: str) -> Topic | None:
        row = await self._db.fetch_one(
            "SELECT * FROM topics WHERE id = ?", (topic_id,)
        )
        return self._from_row(row) if row else None

    async def find_similar_name(self, name: str) -> Topic | None:
        """按名称相似度找已有主题，避免"自主思考系统"和"自主思考系统 "各建一条。"""
        threshold = self._config.duplicate_threshold
        best: Topic | None = None
        best_score = 0.0
        for topic in await self.all(limit=100):
            score = name_similarity(name, topic.name)
            if score >= threshold and score > best_score:
                best, best_score = topic, score
        return best

    # ---------------------------------------------------------------- 写入
    async def resolve(
        self,
        name: str,
        *,
        keywords: Iterable[str] | None = None,
        importance: float = 0.5,
        now: datetime | None = None,
    ) -> Topic | None:
        """找已有主题或谨慎新建。

        一条 Thought 里出现一次关键词不足以建 Topic：只有重要度达到
        ``min_importance_to_create``、且没有高度相似的主题时才会新建。
        """
        name = name.strip()
        if not name:
            return None
        existing = await self.find_similar_name(name)
        if existing is not None:
            return await self.touch(
                existing.name, keywords=keywords, importance=importance, now=now
            )
        current = await self.get(name)
        if current is not None:
            return await self.touch(
                name, keywords=keywords, importance=importance, now=now
            )
        if importance < self._config.min_importance_to_create:
            return None
        return await self._create(name, keywords=keywords, importance=importance, now=now)

    async def touch(
        self,
        name: str,
        *,
        keywords: Iterable[str] | None = None,
        importance: float = 0.5,
        now: datetime | None = None,
    ) -> Topic:
        name = name.strip()
        if not name:
            raise ValueError("topic 名称不能为空")
        moment = now or self._clock()
        existing = await self.get(name)
        if existing is None:
            return await self._create(
                name, keywords=keywords, importance=importance, now=moment
            )
        merged_keywords = tuple(dict.fromkeys((*existing.keywords, *(keywords or ()))))
        status = existing.status
        if status is TopicStatus.NEW:
            status = TopicStatus.ACTIVE
        elif status in (TopicStatus.LOW_ACTIVITY, TopicStatus.DORMANT):
            status = TopicStatus.ACTIVE
        updated = existing.with_changes(
            keywords=merged_keywords[:16],
            importance=max(existing.importance, importance),
            status=status,
            last_activity_at=moment,
            updated_at=moment,
        )
        await self.save(updated)
        return updated

    async def _create(
        self,
        name: str,
        *,
        keywords: Iterable[str] | None,
        importance: float,
        now: datetime,
    ) -> Topic:
        topic = Topic(
            id=f"TP_{uuid.uuid4().hex[:12]}",
            name=name,
            keywords=tuple(dict.fromkeys(keywords or ()))[:16],
            importance=float(importance),
            created_at=now,
            updated_at=now,
            status=TopicStatus.NEW,
            last_activity_at=now,
            metadata={"mentions_count": 1},
        )
        await self.save(topic)
        return topic

    async def save(self, topic: Topic) -> None:
        row = topic.to_row()
        await self._db.execute(
            "INSERT INTO topics"
            " (id, name, keywords, importance, created_at, updated_at, metadata,"
            "  status, last_activity_at, thought_count, memory_count)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET"
            "  name = excluded.name, keywords = excluded.keywords,"
            "  importance = excluded.importance, updated_at = excluded.updated_at,"
            "  metadata = excluded.metadata, status = excluded.status,"
            "  last_activity_at = excluded.last_activity_at,"
            "  thought_count = excluded.thought_count,"
            "  memory_count = excluded.memory_count",
            (
                row["id"],
                row["name"],
                row["keywords"],
                row["importance"],
                row["created_at"],
                row["updated_at"],
                row["metadata"],
                row["status"],
                row["last_activity_at"],
                row["thought_count"],
                row["memory_count"],
            ),
        )

    async def record_thought(
        self,
        topic: Topic,
        *,
        thought_id: str = "",
        now: datetime | None = None,
    ) -> Topic:
        moment = now or self._clock()
        updated = topic.with_changes(
            thought_count=topic.thought_count + 1,
            last_activity_at=moment,
            updated_at=moment,
            status=(
                TopicStatus.ACTIVE
                if topic.status in (TopicStatus.NEW, TopicStatus.LOW_ACTIVITY, TopicStatus.DORMANT)
                else topic.status
            ),
        )
        await self.save(updated)
        return updated

    async def record_memory(
        self,
        topic: Topic,
        *,
        memory_id: str = "",
        now: datetime | None = None,
    ) -> Topic:
        moment = now or self._clock()
        updated = topic.with_changes(
            memory_count=topic.memory_count + 1,
            last_activity_at=moment,
            updated_at=moment,
        )
        await self.save(updated)
        return updated

    # ---------------------------------------------------------------- 生命周期
    def activity(self, topic: Topic, *, now: datetime | None = None) -> float:
        """活跃度 = 最近活动 + 积累量 + 重要度，全部由程序算。"""
        moment = now or self._clock()
        last = topic.last_activity_at or topic.updated_at or topic.created_at
        hours = max(0.0, (moment - last).total_seconds() / 3600.0)
        recency = max(0.0, 1.0 - hours / (max(self._config.dormant_days, 1.0) * 24.0))
        volume = min(1.0, (topic.thought_count * 0.15) + (topic.memory_count * 0.1))
        return _clamp(
            self._config.w_recency * recency
            + self._config.w_activity * volume
            + self._config.w_importance * topic.importance
        )

    async def set_status(
        self,
        topic: Topic,
        status: TopicStatus,
        *,
        now: datetime | None = None,
    ) -> Topic:
        if status is topic.status:
            return topic
        if status not in TOPIC_TRANSITIONS[topic.status]:
            raise TopicLifecycleError(topic.status, status)
        moment = now or self._clock()
        updated = topic.with_changes(status=status, updated_at=moment)
        await self.save(updated)
        return updated

    async def sweep(self, *, now: datetime | None = None) -> list[Topic]:
        """按闲置天数把话题降级：ACTIVE → LOW_ACTIVITY → DORMANT → ARCHIVED。"""
        moment = now or self._clock()
        changed: list[Topic] = []
        for topic in await self.all(limit=200, statuses=(
            TopicStatus.NEW,
            TopicStatus.ACTIVE,
            TopicStatus.INTERESTING,
            TopicStatus.LOW_ACTIVITY,
            TopicStatus.DORMANT,
        )):
            last = topic.last_activity_at or topic.updated_at or topic.created_at
            idle_days = max(0.0, (moment - last).total_seconds() / 86400.0)
            target: TopicStatus | None = None
            if topic.status in (TopicStatus.ACTIVE, TopicStatus.INTERESTING, TopicStatus.NEW):
                if idle_days >= self._config.dormant_days:
                    target = TopicStatus.DORMANT
                elif idle_days >= self._config.low_activity_days:
                    target = TopicStatus.LOW_ACTIVITY
            elif topic.status is TopicStatus.LOW_ACTIVITY:
                if idle_days >= self._config.dormant_days:
                    target = TopicStatus.DORMANT
            elif topic.status is TopicStatus.DORMANT:
                if idle_days >= self._config.archive_days:
                    target = TopicStatus.ARCHIVED
            if target is None:
                continue
            try:
                changed.append(await self.set_status(topic, target, now=moment))
            except TopicLifecycleError:
                continue
        return changed

    def _from_row(self, row: Mapping[str, Any]) -> Topic:
        status = TopicStatus.NEW
        try:
            status = TopicStatus.parse(row.get("status") or "NEW")
        except ValueError:
            status = TopicStatus.NEW
        return Topic(
            id=str(row["id"]),
            name=str(row["name"]),
            keywords=_json_list(row.get("keywords")),
            importance=float(row.get("importance") or 0.5),
            created_at=_parse_dt(row.get("created_at")) or utcnow(),
            updated_at=_parse_dt(row.get("updated_at")) or utcnow(),
            metadata=_json_dict(row.get("metadata")),
            status=status,
            last_activity_at=_parse_dt(row.get("last_activity_at")),
            thought_count=int(row.get("thought_count") or 0),
            memory_count=int(row.get("memory_count") or 0),
        )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = [
    "TOPIC_TRANSITIONS",
    "Topic",
    "TopicLifecycleError",
    "TopicStatus",
    "TopicStore",
    "name_similarity",
]