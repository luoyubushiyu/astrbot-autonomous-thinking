"""Memory 重新激活。

软删除/过期的记忆不该永久死亡：新信息出现、且旧内容仍有价值时，
把它拉回 ACTIVE，再由上层决定要不要唤醒相关的 Thought。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from ..config import MemoryConfig
from ..thought.similarity import similarity
from .models import Memory, REACTIVATABLE_STATUSES, utcnow
from .scoring import value_of
from .store import MemoryStore


def _throttled(memory: Memory, config: MemoryConfig, now: datetime) -> bool:
    if memory.reactivation_count >= config.max_reactivations:
        return True
    if memory.last_reactivated_at is not None:
        elapsed = (now - memory.last_reactivated_at).total_seconds()
        if elapsed < config.reactivation_cooldown_seconds:
            return True
    return False


async def find_candidates(
    store: MemoryStore,
    text: str,
    config: MemoryConfig,
    *,
    now: datetime | None = None,
    limit: int = 3,
) -> list[Memory]:
    """找"跟当前信息相关、状态可恢复、仍有价值"的旧记忆。"""
    moment = now or utcnow()
    similar = await store.find_similar(
        text,
        threshold=config.reactivate_threshold,
        limit=max(limit * 3, 6),
        include_inactive=True,
    )
    found: list[Memory] = []
    for memory, score in similar:
        if memory.status not in REACTIVATABLE_STATUSES:
            continue
        if value_of(memory, moment, config) < config.reactivate_min_value:
            continue
        if _throttled(memory, config, moment):
            continue
        found.append(memory)
        if len(found) >= limit:
            break
    return found


async def reactivate(
    store: MemoryStore,
    memory: Memory,
    config: MemoryConfig,
    *,
    now: datetime | None = None,
    reason: str = "",
    metrics: Any | None = None,
) -> Memory:
    """恢复一条旧记忆，并记 metric。"""
    if memory.status not in REACTIVATABLE_STATUSES:
        raise ValueError(f"记忆状态 {memory.status.value} 不能被重新激活")
    if _throttled(memory, config, now or utcnow()):
        raise ValueError("记忆重新激活过于频繁，已进入冷却")
    restored = await store.restore(memory, now=now, reason=reason)
    if metrics is not None:
        try:
            await metrics.record("memory.reactivated", 1.0, {"memory_id": restored.id})
        except Exception:  # noqa: BLE001 - 指标失败不影响记忆恢复
            pass
    return restored


async def try_reactivate(
    store: MemoryStore,
    text: str,
    config: MemoryConfig,
    *,
    now: datetime | None = None,
    metrics: Any | None = None,
) -> list[Memory]:
    """检索并恢复所有合适的旧记忆（失败不影响主流程）。"""
    restored: list[Memory] = []
    for memory in await find_candidates(store, text, config, now=now):
        try:
            restored.append(
                await reactivate(
                    store,
                    memory,
                    config,
                    now=now,
                    reason="新信息与旧记忆相关",
                    metrics=metrics,
                )
            )
        except ValueError:
            continue
    return restored


__all__ = ["find_candidates", "reactivate", "try_reactivate"]