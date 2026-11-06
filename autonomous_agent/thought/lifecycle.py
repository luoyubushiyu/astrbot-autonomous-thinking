"""Thought 生命周期。

状态只能沿白名单迁移；``Disposition``（一次思考的结果去向）映射到目标状态。
非法迁移必须抛 :class:`ThoughtLifecycleError`，由调用方决定回滚或忽略。
"""

from __future__ import annotations

from .models import Disposition, ThoughtStatus

S = ThoughtStatus

STATUS_TRANSITIONS: dict[ThoughtStatus, frozenset[ThoughtStatus]] = {
    S.NEW: frozenset({S.QUEUED, S.WAITING, S.DRAWER, S.DISCARDED}),
    S.QUEUED: frozenset({S.THINKING, S.DRAWER, S.DISCARDED, S.ARCHIVED}),
    # THINKING -> QUEUED 是思考失败时的合法回滚。
    S.THINKING: frozenset(
        {
            S.QUEUED,
            S.INSIGHT_FOUND,
            S.FOLLOW_UP,
            S.WAITING,
            S.RESOLVED,
            S.DRAWER,
            S.DISCARDED,
        }
    ),
    S.INSIGHT_FOUND: frozenset(
        {
            S.QUEUED,
            S.FOLLOW_UP,
            S.WAITING,
            S.RESOLVED,
            S.DRAWER,
            S.DISCARDED,
            S.EXPIRED,
            S.ARCHIVED,
        }
    ),
    S.FOLLOW_UP: frozenset(
        {
            S.QUEUED,
            S.INSIGHT_FOUND,
            S.WAITING,
            S.RESOLVED,
            S.DRAWER,
            S.DISCARDED,
            S.EXPIRED,
        }
    ),
    S.WAITING: frozenset(
        {S.QUEUED, S.RESOLVED, S.DRAWER, S.EXPIRED, S.DISCARDED}
    ),
    S.RESOLVED: frozenset({S.ARCHIVED, S.DRAWER}),
    S.DRAWER: frozenset({S.QUEUED, S.EXPIRED, S.ARCHIVED, S.DISCARDED}),
    S.DISCARDED: frozenset({S.ARCHIVED}),
    S.EXPIRED: frozenset({S.QUEUED, S.ARCHIVED}),
    S.ARCHIVED: frozenset({S.QUEUED}),
}

# disposition -> 思考结束后的状态
DISPOSITION_TARGETS: dict[Disposition, ThoughtStatus] = {
    Disposition.KEEP: S.INSIGHT_FOUND,
    Disposition.DRAWER: S.DRAWER,
    Disposition.DIARY: S.RESOLVED,
    Disposition.SPEAK: S.INSIGHT_FOUND,
    Disposition.BLACKBOX: S.DISCARDED,
    Disposition.DISCARD: S.DISCARDED,
    Disposition.WAIT: S.WAITING,
    Disposition.FOLLOW_UP: S.FOLLOW_UP,
    # 继续想：文本保留，下一 Tick 再读；状态回到 QUEUED 以便被再次挑中。
    Disposition.CONTINUE: S.QUEUED,
    Disposition.RESOLVED: S.RESOLVED,
}


class ThoughtLifecycleError(ValueError):
    def __init__(self, source: ThoughtStatus, target: ThoughtStatus) -> None:
        super().__init__(f"非法的 Thought 状态迁移: {source.value} -> {target.value}")
        self.source = source
        self.target = target


def can_transition(source: ThoughtStatus, target: ThoughtStatus) -> bool:
    return target in STATUS_TRANSITIONS.get(source, frozenset())


def require_transition(source: ThoughtStatus, target: ThoughtStatus) -> None:
    if not can_transition(source, target):
        raise ThoughtLifecycleError(source, target)


def target_status_for(disposition: Disposition) -> ThoughtStatus:
    return DISPOSITION_TARGETS[disposition]


__all__ = [
    "DISPOSITION_TARGETS",
    "STATUS_TRANSITIONS",
    "ThoughtLifecycleError",
    "can_transition",
    "require_transition",
    "target_status_for",
]