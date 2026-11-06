"""Thought Scoring。

选择"先想哪个"是程序的事，不是 LLM 的事：分数完全由可配置权重算出，
LLM 只能影响 Thought 自身的属性（importance / curiosity / relevance）。
"""

from __future__ import annotations

from datetime import datetime

from ..config import ThoughtConfig
from .models import Thought, ThoughtStatus, utcnow

_UNRESOLVED_BONUS = {
    ThoughtStatus.NEW: 1.0,
    ThoughtStatus.QUEUED: 1.0,
    ThoughtStatus.FOLLOW_UP: 0.9,
    ThoughtStatus.WAITING: 0.6,
    ThoughtStatus.INSIGHT_FOUND: 0.2,
    ThoughtStatus.RESOLVED: 0.0,
    ThoughtStatus.DRAWER: 0.0,
}


def score_thought(
    thought: Thought,
    now: datetime | None = None,
    config: ThoughtConfig | None = None,
) -> float:
    if config is None:
        config = ThoughtConfig()
    moment = now or utcnow()
    inactive_days = max(0.0, (moment - _latest_activity(thought)).total_seconds() / 86400)
    novelty = max(0.0, 1.0 - thought.revisit_count / max(1, config.novelty_revisits))
    relationship = min(
        1.0, (len(thought.related_memories) + len(thought.related_topics)) / 3.0
    )
    future = 1.0 if thought.next_check_at else 0.2
    repetition = min(1.0, thought.reactivation_count / 3.0)
    stale = min(1.0, inactive_days / max(0.1, config.stale_days))
    unresolved = _UNRESOLVED_BONUS.get(thought.status, 0.1)
    time_suitability = _time_suitability(thought, moment)
    cost = min(1.0, thought.revisit_count / 5.0)

    score = (
        config.w_importance * thought.importance
        + config.w_curiosity * thought.curiosity
        + config.w_relevance * thought.relevance
        + config.w_novelty * novelty
        + config.w_unresolved * unresolved
        + config.w_relationship * relationship
        + config.w_future * future
        + config.w_time_suitability * time_suitability
        - config.penalty_repetition * repetition
        - config.penalty_stale * stale
        - config.penalty_cost * cost
    )
    return round(score, 6)


def rank_thoughts(
    thoughts: list[Thought],
    now: datetime | None = None,
    config: ThoughtConfig | None = None,
) -> list[Thought]:
    if config is None:
        config = ThoughtConfig()
    moment = now or utcnow()
    return sorted(
        thoughts,
        key=lambda item: (score_thought(item, moment, config=config), item.updated_at),
        reverse=True,
    )


def _latest_activity(thought: Thought) -> datetime:
    return thought.last_thought_at or thought.updated_at


def _time_suitability(thought: Thought, now: datetime) -> float:
    """现在适不适合想它：到期或没有排期 = 1，排在未来越远越低。"""
    if thought.next_check_at is None or thought.next_check_at <= now:
        return 1.0
    hours = (thought.next_check_at - now).total_seconds() / 3600.0
    return max(0.0, 1.0 - hours / 24.0)


__all__ = ["rank_thoughts", "score_thought"]