"""Thought 数据模型。

一个 Thought 代表"一个还没想完的问题/想法"。SQLite 是结构化事实来源，
文件只是它的可读镜像；两侧的一致性由 :class:`ThoughtRepository` 维护。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ThoughtType(str, Enum):
    QUESTION = "QUESTION"
    IDEA = "IDEA"
    PROBLEM = "PROBLEM"
    CURIOSITY = "CURIOSITY"
    OBSERVATION = "OBSERVATION"
    UNFINISHED_THOUGHT = "UNFINISHED_THOUGHT"
    REFLECTION = "REFLECTION"

    @classmethod
    def parse(cls, value: Any) -> "ThoughtType":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as exc:
            raise ValueError(f"未知 Thought 类型: {value!r}") from exc


class ThoughtStatus(str, Enum):
    NEW = "NEW"
    QUEUED = "QUEUED"
    THINKING = "THINKING"
    INSIGHT_FOUND = "INSIGHT_FOUND"
    FOLLOW_UP = "FOLLOW_UP"
    WAITING = "WAITING"
    RESOLVED = "RESOLVED"
    DRAWER = "DRAWER"
    DISCARDED = "DISCARDED"
    EXPIRED = "EXPIRED"
    ARCHIVED = "ARCHIVED"

    @classmethod
    def parse(cls, value: Any) -> "ThoughtStatus":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as exc:
            raise ValueError(f"未知 Thought 状态: {value!r}") from exc


class Disposition(str, Enum):
    """一次思考结束后的去向，由 LLM 建议、程序校验。"""

    KEEP = "KEEP"
    DRAWER = "DRAWER"
    DIARY = "DIARY"
    SPEAK = "SPEAK"
    BLACKBOX = "BLACKBOX"
    DISCARD = "DISCARD"
    WAIT = "WAIT"
    FOLLOW_UP = "FOLLOW_UP"
    CONTINUE = "CONTINUE"
    RESOLVED = "RESOLVED"

    @classmethod
    def parse(cls, value: Any) -> "Disposition":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as exc:
            raise ValueError(f"未知 disposition: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class NewQuestion:
    """一次思考产生的新问题，会被落成子 Thought。"""

    title: str
    content: str
    importance: float = 0.5
    curiosity: float = 0.5


@dataclass(frozen=True, slots=True)
class Thought:
    id: str
    type: ThoughtType
    title: str
    content: str
    source: str
    status: ThoughtStatus
    importance: float = 0.5
    curiosity: float = 0.5
    relevance: float = 0.5
    confidence: float = 0.5
    revisit_count: int = 0
    reactivation_count: int = 0
    related_memories: tuple[str, ...] = ()
    related_topics: tuple[str, ...] = ()
    parent_thought_id: str | None = None
    child_thought_ids: tuple[str, ...] = ()
    next_action: str | None = None
    insight: str | None = None
    uncertainty: str | None = None
    last_thought_at: datetime | None = None
    next_check_at: datetime | None = None
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    # -- 构造 -------------------------------------------------------------- #
    @classmethod
    def create(
        cls,
        *,
        type: ThoughtType | str,
        title: str,
        content: str = "",
        source: str = "system",
        status: ThoughtStatus | str = ThoughtStatus.NEW,
        importance: float = 0.5,
        curiosity: float = 0.5,
        relevance: float = 0.5,
        confidence: float = 0.5,
        parent_thought_id: str | None = None,
        child_thought_ids: tuple[str, ...] = (),
        related_memories: tuple[str, ...] = (),
        related_topics: tuple[str, ...] = (),
        next_check_at: datetime | None = None,
        now: datetime | None = None,
    ) -> "Thought":
        moment = now or utcnow()
        return cls(
            id=new_thought_id(moment),
            type=ThoughtType.parse(type),
            title=title,
            content=content,
            source=source,
            status=ThoughtStatus.parse(status),
            importance=_clamp01(importance),
            curiosity=_clamp01(curiosity),
            relevance=_clamp01(relevance),
            confidence=_clamp01(confidence),
            parent_thought_id=parent_thought_id,
            child_thought_ids=tuple(child_thought_ids),
            related_memories=tuple(related_memories),
            related_topics=tuple(related_topics),
            next_check_at=next_check_at,
            created_at=moment,
            updated_at=moment,
        )

    def with_changes(self, **changes: Any) -> "Thought":
        return replace(self, **changes)

    # -- 持久化 ------------------------------------------------------------ #
    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "status": self.status.value,
            "importance": self.importance,
            "curiosity": self.curiosity,
            "relevance": self.relevance,
            "confidence": self.confidence,
            "revisit_count": self.revisit_count,
            "reactivation_count": self.reactivation_count,
            "related_memories": json.dumps(list(self.related_memories), ensure_ascii=False),
            "related_topics": json.dumps(list(self.related_topics), ensure_ascii=False),
            "parent_thought_id": self.parent_thought_id,
            "child_thought_ids": json.dumps(list(self.child_thought_ids), ensure_ascii=False),
            "next_action": self.next_action,
            "insight": self.insight,
            "uncertainty": self.uncertainty,
            "last_thought_at": _iso(self.last_thought_at),
            "next_check_at": _iso(self.next_check_at),
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
            "metadata": json.dumps(dict(self.metadata), ensure_ascii=False),
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Thought":
        return cls(
            id=str(row["id"]),
            type=ThoughtType.parse(row["type"]),
            title=str(row["title"]),
            content=str(row["content"]),
            source=str(row["source"]),
            status=ThoughtStatus.parse(row["status"]),
            importance=float(row["importance"]),
            curiosity=float(row["curiosity"]),
            relevance=float(row["relevance"]),
            confidence=float(row["confidence"]),
            revisit_count=int(row["revisit_count"]),
            reactivation_count=int(row["reactivation_count"]),
            related_memories=tuple(_json_list(row["related_memories"])),
            related_topics=tuple(_json_list(row["related_topics"])),
            parent_thought_id=row["parent_thought_id"],
            child_thought_ids=tuple(_json_list(row.get("child_thought_ids"))),
            next_action=row["next_action"],
            insight=row["insight"],
            uncertainty=row["uncertainty"],
            last_thought_at=_parse_dt(row["last_thought_at"]),
            next_check_at=_parse_dt(row["next_check_at"]),
            created_at=_parse_dt(row["created_at"]) or utcnow(),
            updated_at=_parse_dt(row["updated_at"]) or utcnow(),
            metadata=_json_dict(row["metadata"]),
        )

    @classmethod
    def from_payload(cls, data: Mapping[str, Any]) -> "Thought":
        """从文件镜像 JSON 还原（列表字段可能是原生数组）。"""
        row = dict(data)
        for key in ("related_memories", "related_topics", "child_thought_ids"):
            value = row.get(key, [])
            if not isinstance(value, str):
                row[key] = json.dumps(list(value), ensure_ascii=False)
        metadata = row.get("metadata", {})
        if not isinstance(metadata, str):
            row["metadata"] = json.dumps(dict(metadata), ensure_ascii=False)
        return cls.from_row(row)

    def to_json(self) -> str:
        """给文件镜像用的 JSON 文本。"""
        return json.dumps(self.to_payload(), ensure_ascii=False, indent=2)

    def to_payload(self) -> dict[str, Any]:
        row = self.to_row()
        row["related_memories"] = list(self.related_memories)
        row["related_topics"] = list(self.related_topics)
        row["child_thought_ids"] = list(self.child_thought_ids)
        row["metadata"] = dict(self.metadata)
        return row


def new_thought_id(now: datetime | None = None) -> str:
    stamp = (now or utcnow()).strftime("%Y%m%d_%H%M%S")
    return f"TH_{stamp}_{uuid.uuid4().hex[:4]}"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        result = value
    else:
        try:
            result = datetime.fromisoformat(str(value))
        except ValueError:
            return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result


def _json_list(value: Any) -> list[str]:
    try:
        data = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return []
    if not isinstance(data, (list, tuple)):
        return []
    return [str(item) for item in data]


def _json_dict(value: Any) -> dict[str, Any]:
    try:
        data = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return {}
    return dict(data) if isinstance(data, Mapping) else {}


__all__ = [
    "Disposition",
    "NewQuestion",
    "Thought",
    "ThoughtStatus",
    "ThoughtType",
    "new_thought_id",
    "utcnow",
]