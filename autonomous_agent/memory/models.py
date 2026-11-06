"""Memory 数据模型。

Memory 是 Agent 对用户与世界的长期认识，与 Thought 严格分开：
Thought 是"正在想的东西"，Memory 是"记住的东西"。一条 Thought 不会因为
存在就自动变成 Memory，写入必须显式经过程序（Validator + Repository）。
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


class MemoryType(str, Enum):
    """记忆类型。

    INFERENCE 是推测，永远不能自动升级成 FACT。
    """

    FACT = "FACT"
    PREFERENCE = "PREFERENCE"
    EVENT = "EVENT"
    RELATIONSHIP = "RELATIONSHIP"
    TOPIC = "TOPIC"
    INFERENCE = "INFERENCE"
    THOUGHT = "THOUGHT"
    EPISODIC_MEMORY = "EPISODIC_MEMORY"
    SUMMARY_MEMORY = "SUMMARY_MEMORY"

    @classmethod
    def parse(cls, value: Any) -> "MemoryType":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ValueError(f"无效的 Memory 类型: {value!r}")
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"未知的 Memory 类型: {value!r}") from None


class MemoryStatus(str, Enum):
    """记忆的生命阶段。遗忘是降级而不是删除。"""

    ACTIVE = "ACTIVE"
    LOW_VALUE = "LOW_VALUE"
    SOFT_DELETED = "SOFT_DELETED"
    EXPIRED = "EXPIRED"
    BLACKBOX = "BLACKBOX"
    ARCHIVED = "ARCHIVED"

    @classmethod
    def parse(cls, value: Any) -> "MemoryStatus":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError:
            return cls.ACTIVE


STATUS_ORDER: tuple[MemoryStatus, ...] = (
    MemoryStatus.ACTIVE,
    MemoryStatus.LOW_VALUE,
    MemoryStatus.SOFT_DELETED,
    MemoryStatus.EXPIRED,
    MemoryStatus.BLACKBOX,
    MemoryStatus.ARCHIVED,
)

#: 能被正常检索、参与思考的记忆。
USABLE_STATUSES = frozenset({MemoryStatus.ACTIVE, MemoryStatus.LOW_VALUE})
#: 可以被重新激活的记忆。
REACTIVATABLE_STATUSES = frozenset(
    {MemoryStatus.SOFT_DELETED, MemoryStatus.EXPIRED, MemoryStatus.BLACKBOX}
)

SOURCE_TYPES = frozenset(
    {
        "USER_MESSAGE",
        "THOUGHT",
        "EVENT",
        "OBSERVATION",
        "CONSOLIDATION",
        "DIARY",
        "SYSTEM_OBSERVATION",
    }
)


def new_memory_id(now: datetime | None = None) -> str:
    moment = now or utcnow()
    return f"MEM_{moment.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"


def _parse_dt(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value:
        try:
            data = json.loads(value)
            return data if isinstance(data, dict) else {}
        except (ValueError, TypeError):
            return {}
    return {}


def _json_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item).strip())
    if isinstance(value, str) and value:
        try:
            data = json.loads(value)
        except (ValueError, TypeError):
            return ()
        return _json_list(data)
    return ()


def _infer_source_type(source: str) -> str:
    """旧数据只有 source 字符串，尽量还原它的类别。"""
    text = (source or "").strip()
    if text.startswith("user:"):
        return "USER_MESSAGE"
    if text.startswith("thought:"):
        return "THOUGHT"
    if text.startswith("diary:"):
        return "DIARY"
    if text.startswith("event:"):
        return "EVENT"
    if text.startswith("consolidation") or text == "consolidation":
        return "CONSOLIDATION"
    return "SYSTEM_OBSERVATION"


@dataclass(frozen=True, slots=True)
class Memory:
    id: str
    type: MemoryType
    content: str
    confidence: float = 0.5
    source: str = "system"
    importance: float = 0.5
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    last_verified_at: datetime | None = None
    expires_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    status: MemoryStatus = MemoryStatus.ACTIVE
    relevance: float = 0.5
    frequency: int = 1
    usage_count: int = 0
    future_value: float = 0.5
    relationship_value: float = 0.3
    last_used_at: datetime | None = None
    source_type: str = "SYSTEM_OBSERVATION"
    source_id: str = ""
    keywords: tuple[str, ...] = ()
    reactivation_count: int = 0
    last_reactivated_at: datetime | None = None
    conflict: bool = False

    @property
    def memory_id(self) -> str:
        return self.id

    @classmethod
    def create(
        cls,
        *,
        type: MemoryType | str,
        content: str,
        confidence: float = 0.5,
        source: str = "system",
        importance: float = 0.5,
        expires_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
        now: datetime | None = None,
        status: MemoryStatus | str = MemoryStatus.ACTIVE,
        relevance: float = 0.5,
        future_value: float = 0.5,
        relationship_value: float = 0.3,
        source_type: str | None = None,
        source_id: str = "",
        keywords: tuple[str, ...] | list[str] = (),
    ) -> "Memory":
        moment = now or utcnow()
        resolved_source = source_type or _infer_source_type(source)
        return cls(
            id=new_memory_id(moment),
            type=MemoryType.parse(type),
            content=content,
            confidence=max(0.0, min(1.0, confidence)),
            source=source,
            importance=max(0.0, min(1.0, importance)),
            created_at=moment,
            updated_at=moment,
            last_verified_at=moment,
            expires_at=expires_at,
            metadata=dict(metadata or {}),
            status=MemoryStatus.parse(status),
            relevance=max(0.0, min(1.0, relevance)),
            future_value=max(0.0, min(1.0, future_value)),
            relationship_value=max(0.0, min(1.0, relationship_value)),
            source_type=resolved_source if resolved_source in SOURCE_TYPES else "SYSTEM_OBSERVATION",
            source_id=source_id,
            keywords=tuple(str(item) for item in keywords if str(item).strip()),
        )

    def with_changes(self, **changes: Any) -> "Memory":
        return replace(self, **changes)

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "confidence": self.confidence,
            "source": self.source,
            "importance": self.importance,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
            "last_verified_at": _iso(self.last_verified_at),
            "expires_at": _iso(self.expires_at),
            "metadata": json.dumps(dict(self.metadata), ensure_ascii=False),
            "status": self.status.value,
            "relevance": self.relevance,
            "frequency": self.frequency,
            "usage_count": self.usage_count,
            "future_value": self.future_value,
            "relationship_value": self.relationship_value,
            "last_used_at": _iso(self.last_used_at),
            "source_type": self.source_type,
            "source_id": self.source_id,
            "keywords": json.dumps(list(self.keywords), ensure_ascii=False),
            "reactivation_count": self.reactivation_count,
            "last_reactivated_at": _iso(self.last_reactivated_at),
            "conflict": 1 if self.conflict else 0,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Memory":
        return cls(
            id=str(row["id"]),
            type=MemoryType.parse(row["type"]),
            content=str(row["content"]),
            confidence=float(row.get("confidence", 0.5)),
            source=str(row.get("source") or "system"),
            importance=float(row.get("importance", 0.5)),
            created_at=_parse_dt(row.get("created_at")) or utcnow(),
            updated_at=_parse_dt(row.get("updated_at")) or utcnow(),
            last_verified_at=_parse_dt(row.get("last_verified_at")),
            expires_at=_parse_dt(row.get("expires_at")),
            metadata=_json_dict(row.get("metadata")),
            status=MemoryStatus.parse(row.get("status", MemoryStatus.ACTIVE.value)),
            relevance=float(row.get("relevance", 0.5) or 0.5),
            frequency=int(row.get("frequency", 1) or 1),
            usage_count=int(row.get("usage_count", 0) or 0),
            future_value=float(row.get("future_value", 0.5) or 0.5),
            relationship_value=float(row.get("relationship_value", 0.3) or 0.3),
            last_used_at=_parse_dt(row.get("last_used_at")),
            source_type=str(row.get("source_type") or _infer_source_type(row.get("source") or "")),
            source_id=str(row.get("source_id") or ""),
            keywords=_json_list(row.get("keywords")),
            reactivation_count=int(row.get("reactivation_count", 0) or 0),
            last_reactivated_at=_parse_dt(row.get("last_reactivated_at")),
            conflict=bool(row.get("conflict", 0)),
        )


__all__ = [
    "Memory",
    "MemoryStatus",
    "MemoryType",
    "REACTIVATABLE_STATUSES",
    "SOURCE_TYPES",
    "STATUS_ORDER",
    "USABLE_STATUSES",
    "new_memory_id",
    "utcnow",
]