"""Memory 价值评分。

评分完全由程序计算：LLM 可以提供 importance / confidence 之类的输入，
但不能绕过这里的公式。遗忘与整理都以这个值为依据。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..config import MemoryConfig
from .models import Memory, utcnow


@dataclass(frozen=True, slots=True)
class MemoryValue:
    value: float
    base: float
    usage_factor: float
    recency_factor: float
    stale_penalty: float
    duplicate_penalty: float
    low_confidence_penalty: float

    def to_dict(self) -> dict[str, float]:
        return {
            "value": round(self.value, 4),
            "base": round(self.base, 4),
            "usage_factor": round(self.usage_factor, 4),
            "recency_factor": round(self.recency_factor, 4),
            "stale_penalty": round(self.stale_penalty, 4),
            "duplicate_penalty": round(self.duplicate_penalty, 4),
            "low_confidence_penalty": round(self.low_confidence_penalty, 4),
        }


def _days_between(later: datetime, earlier: datetime) -> float:
    return max(0.0, (later - earlier).total_seconds() / 86400.0)


def memory_value(
    memory: Memory,
    now: datetime | None = None,
    config: MemoryConfig | None = None,
) -> MemoryValue:
    cfg = config or MemoryConfig()
    moment = now or utcnow()

    base = (
        memory.importance
        * memory.confidence
        * (0.5 + 0.5 * memory.relevance)
        * (0.5 + 0.5 * memory.future_value)
    )
    usage_factor = 1.0 + min(
        0.5, 0.05 * memory.usage_count + 0.02 * max(0, memory.frequency - 1)
    )
    last_touch = memory.last_used_at or memory.last_verified_at or memory.updated_at
    idle = _days_between(moment, last_touch)
    recency_factor = max(0.25, 1.0 - 0.75 * min(1.0, idle / max(1.0, cfg.stale_days)))
    stale_penalty = cfg.penalty_stale * min(1.0, idle / max(1.0, cfg.stale_days))
    merged = int(memory.metadata.get("merged_count", 0) or 0)
    duplicate_penalty = cfg.penalty_duplicate * min(1.0, merged / 3.0)
    low_confidence_penalty = cfg.penalty_low_confidence * max(0.0, 1.0 - memory.confidence)

    raw = (
        base * usage_factor * recency_factor
        - stale_penalty
        - duplicate_penalty
        - low_confidence_penalty
    )
    value = max(0.0, min(1.5, raw))
    return MemoryValue(
        value=value,
        base=base,
        usage_factor=usage_factor,
        recency_factor=recency_factor,
        stale_penalty=stale_penalty,
        duplicate_penalty=duplicate_penalty,
        low_confidence_penalty=low_confidence_penalty,
    )


def value_of(
    memory: Memory,
    now: datetime | None = None,
    config: MemoryConfig | None = None,
) -> float:
    return memory_value(memory, now, config).value


__all__ = ["MemoryValue", "memory_value", "value_of"]