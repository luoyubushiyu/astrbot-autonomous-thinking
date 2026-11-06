"""Curiosity 评分：程序负责打分，LLM 只负责"想"。

Curiosity 不直接产生消息，也不执行任何外部行为；它只产生
"值得探索的问题"，最终由 ThoughtEngine 建 Thought 并进入思考文本。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ..config import CuriosityConfig


@dataclass(frozen=True, slots=True)
class CuriosityCandidate:
    """一个"值得探索"的候选问题。"""

    kind: str
    title: str
    content: str = ""
    topic_name: str = ""
    source_type: str = ""
    source_id: str = ""
    importance: float = 0.5
    uncertainty: float = 0.5
    novelty: float = 0.5
    relevance: float = 0.5
    knowledge_gap: float = 0.5
    future_value: float = 0.2
    relationship_value: float = 0.0
    repetition: float = 0.0
    staleness: float = 0.0
    cost: float = 0.0

    def with_changes(self, **changes: float | str) -> "CuriosityCandidate":
        from dataclasses import replace

        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class CuriosityScore:
    value: float
    parts: dict[str, float] = field(default_factory=dict)


def _c(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def curiosity_score(
    candidate: CuriosityCandidate,
    config: CuriosityConfig,
) -> CuriosityScore:
    """加权求和，再减重复/陈旧/成本三项惩罚。"""
    parts = {
        "novelty": config.w_novelty * _c(candidate.novelty),
        "uncertainty": config.w_uncertainty * _c(candidate.uncertainty),
        "importance": config.w_importance * _c(candidate.importance),
        "relevance": config.w_relevance * _c(candidate.relevance),
        "knowledge_gap": config.w_knowledge_gap * _c(candidate.knowledge_gap),
        "future": config.w_future * _c(candidate.future_value),
        "relationship": config.w_relationship * _c(candidate.relationship_value),
        "repetition": -config.penalty_repetition * _c(candidate.repetition),
        "stale": -config.penalty_stale * _c(candidate.staleness),
        "cost": -config.penalty_cost * _c(candidate.cost),
    }
    return CuriosityScore(value=round(sum(parts.values()), 4), parts=parts)


__all__ = ["CuriosityCandidate", "CuriosityScore", "curiosity_score"]