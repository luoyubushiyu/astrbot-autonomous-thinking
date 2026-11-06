"""Thought Reactivation。

新信息出现时，程序先找"过去被放下/遗忘的 Thought 里有没有相关的"。
命中后把 DRAWER / EXPIRED / ARCHIVED 拉回 QUEUED，等待下一次思考。

复活有闸门：单条 Thought 的 ``max_reactivations`` 与
``reactivation_cooldown_seconds`` 防止无限复活。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..config import ThoughtConfig
from .models import Thought, ThoughtStatus, utcnow
from .repository import ThoughtRepository
from .similarity import similarity

REACTIVATABLE = (
    ThoughtStatus.DRAWER,
    ThoughtStatus.EXPIRED,
    ThoughtStatus.ARCHIVED,
)


async def find_candidates(
    repo: ThoughtRepository,
    text: str,
    config: ThoughtConfig,
    *,
    now: datetime | None = None,
    limit: int = 3,
) -> list[tuple[Thought, float]]:
    moment = now or utcnow()
    thoughts = await repo.list(REACTIVATABLE, limit=200)
    matches: list[tuple[Thought, float]] = []
    for thought in thoughts:
        score = similarity(text, f"{thought.title}\n{thought.content}")
        if score < config.reactivation_threshold:
            continue
        if _is_throttled(thought, moment, config):
            continue
        matches.append((thought, score))
    matches.sort(key=lambda item: item[1], reverse=True)
    return matches[:limit]


async def reactivate(
    repo: ThoughtRepository,
    thought: Thought,
    config: ThoughtConfig,
    *,
    now: datetime | None = None,
    reason: str = "reactivation",
    metrics: Any = None,
) -> Thought | None:
    moment = now or utcnow()
    if _is_throttled(thought, moment, config):
        return None
    metadata = {
        **dict(thought.metadata),
        "last_reactivated_at": moment.isoformat(),
        "last_reactivation_reason": reason,
    }
    updated = await repo.transition(
        thought,
        ThoughtStatus.QUEUED,
        reason=reason,
        now=moment,
        changes={
            "reactivation_count": thought.reactivation_count + 1,
            "next_check_at": None,
            "metadata": metadata,
        },
    )
    if metrics is not None:
        try:
            await metrics.record("thought.reactivated", 1.0, {"thought_id": thought.id})
        except Exception:  # noqa: BLE001
            pass
    return updated


async def try_reactivate(
    repo: ThoughtRepository,
    text: str,
    config: ThoughtConfig,
    *,
    now: datetime | None = None,
    reason: str = "new_information",
    metrics: Any = None,
) -> list[Thought]:
    candidates = await find_candidates(repo, text, config, now=now)
    revived: list[Thought] = []
    for thought, _score in candidates:
        updated = await reactivate(
            repo, thought, config, now=now, reason=reason, metrics=metrics
        )
        if updated is not None:
            revived.append(updated)
    return revived


def _is_throttled(thought: Thought, now: datetime, config: ThoughtConfig) -> bool:
    if thought.reactivation_count >= config.max_reactivations:
        return True
    last = thought.metadata.get("last_reactivated_at")
    if not last:
        return False
    try:
        last_dt = datetime.fromisoformat(str(last))
    except ValueError:
        return False
    if last_dt.tzinfo is None:
        from datetime import timezone

        last_dt = last_dt.replace(tzinfo=timezone.utc)
    elapsed = (now - last_dt).total_seconds()
    return elapsed < config.reactivation_cooldown_seconds


__all__ = [
    "REACTIVATABLE",
    "find_candidates",
    "reactivate",
    "try_reactivate",
]