"""Thought Forgetting / Decay。

不是"超过 X 天全删"，而是按价值衰减打分：越老、越久没碰、越不重要、
越没关联，越容易被判为 EXPIRED。EXPIRED 只是软删除，文件进
``blackbox/expired``，之后仍可能被新信息唤醒。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..config import ThoughtConfig
from .models import Thought, ThoughtStatus, utcnow
from .repository import ThoughtRepository

# 只遗忘"想过但没留住价值"的 Thought；NEW/QUEUED 还没被思考过，不参与。
SWEEPABLE = (
    ThoughtStatus.INSIGHT_FOUND,
    ThoughtStatus.FOLLOW_UP,
    ThoughtStatus.WAITING,
    ThoughtStatus.DRAWER,
)


@dataclass(frozen=True, slots=True)
class ForgetResult:
    thought_id: str
    forget_score: float


def forget_score(
    thought: Thought,
    now: datetime | None = None,
    config: ThoughtConfig | None = None,
) -> float:
    if config is None:
        config = ThoughtConfig()
    moment = now or utcnow()
    age_days = max(0.0, (moment - thought.created_at).total_seconds() / 86400)
    last = thought.last_thought_at or thought.updated_at
    inactivity_days = max(0.0, (moment - last).total_seconds() / 86400)

    score = (
        0.5 * min(3.0, age_days / max(0.1, config.forget_after_days))
        + 0.5 * min(3.0, inactivity_days / max(0.1, config.forget_inactivity_days))
        + 0.5 * (1.0 - thought.importance)
        + 0.3 * (1.0 - thought.relevance)
        + 0.05 * thought.reactivation_count
        - 0.08 * thought.revisit_count
        - 0.1 * thought.confidence
    )
    return round(score, 6)


async def sweep_forgetting(
    repo: ThoughtRepository,
    config: ThoughtConfig,
    *,
    now: datetime | None = None,
    limit: int = 50,
    metrics: Any = None,
) -> list[ForgetResult]:
    """扫描并软删除低价值 Thought，返回本次被遗忘的列表。"""
    moment = now or utcnow()
    expired: list[ForgetResult] = []
    thoughts = await repo.list(SWEEPABLE, limit=limit)
    for thought in thoughts:
        score = forget_score(thought, moment, config=config)
        if score <= config.forget_threshold:
            continue
        # 被反复唤醒/ revisit 过的 Thought 说明还有价值，阈值抬高。
        if thought.revisit_count >= 3 and thought.importance >= 0.7:
            continue
        await repo.transition(
            thought,
            ThoughtStatus.EXPIRED,
            reason=f"forgetting:{score:.2f}",
            now=moment,
        )
        expired.append(ForgetResult(thought.id, score))
        if metrics is not None:
            try:
                await metrics.record(
                    "thought.expired", 1.0, {"thought_id": thought.id, "score": score}
                )
            except Exception:  # noqa: BLE001 - 指标失败不影响遗忘
                pass
    return expired


__all__ = ["ForgetResult", "SWEEPABLE", "forget_score", "sweep_forgetting"]