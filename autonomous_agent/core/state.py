"""Agent 状态与状态机。

* Agent State 描述"Agent 现在处于什么运行态"，只由程序改写；
* Thought 状态与 Agent State 分离：每条想法的状态是 ``thought.models.ThoughtStatus``
  （存在 events/thoughts 表里），跟 Agent 状态机没有耦合；
* ``ThoughtState`` 只是早期占位枚举，保留兼容，不再驱动任何流程。

重启恢复规则集中在 :func:`recover_state`，避免"停在 THINKING 却再也动不了"
这类僵尸状态。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Mapping


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AgentState(str, Enum):
    """Agent 运行态。"""

    IDLE = "IDLE"
    OBSERVING = "OBSERVING"
    THINKING = "THINKING"
    WAITING = "WAITING"
    CONTACTING = "CONTACTING"
    SILENT = "SILENT"
    NIGHT = "NIGHT"
    SLEEPING = "SLEEPING"
    SAFE_IDLE = "SAFE_IDLE"

    @classmethod
    def parse(cls, value: str) -> "AgentState":
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(f"未知 Agent 状态: {value!r}") from exc


class ThoughtState(str, Enum):
    """早期占位枚举，保留以兼容旧数据。

    真正按想法追踪的状态是 ``thought.models.ThoughtStatus``（NEW/QUEUED/…），
    与 Agent State 完全分离；这里的值不再参与任何业务判断。
    """

    INACTIVE = "INACTIVE"
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"


# 允许的状态转移。任何状态都允许回到 IDLE / SAFE_IDLE（两跳兜底），
# 这样异常恢复与用户消息唤醒都不需要写一堆特例。
ALLOWED_TRANSITIONS: dict[AgentState, frozenset[AgentState]] = {
    AgentState.IDLE: frozenset(
        {
            AgentState.OBSERVING,
            AgentState.THINKING,
            AgentState.WAITING,
            AgentState.SILENT,
            AgentState.NIGHT,
            AgentState.SLEEPING,
            AgentState.CONTACTING,
            AgentState.SAFE_IDLE,
        }
    ),
    AgentState.OBSERVING: frozenset(
        {AgentState.THINKING, AgentState.IDLE, AgentState.SAFE_IDLE, AgentState.SILENT}
    ),
    AgentState.THINKING: frozenset(
        {
            AgentState.IDLE,
            AgentState.OBSERVING,
            AgentState.WAITING,
            AgentState.CONTACTING,
            AgentState.SILENT,
            AgentState.SAFE_IDLE,
        }
    ),
    AgentState.WAITING: frozenset(
        {
            AgentState.IDLE,
            AgentState.THINKING,
            AgentState.SILENT,
            AgentState.SLEEPING,
            AgentState.NIGHT,
            AgentState.SAFE_IDLE,
        }
    ),
    AgentState.CONTACTING: frozenset(
        {AgentState.IDLE, AgentState.WAITING, AgentState.SILENT, AgentState.SAFE_IDLE}
    ),
    AgentState.SILENT: frozenset(
        {
            AgentState.IDLE,
            AgentState.OBSERVING,
            AgentState.THINKING,
            AgentState.SLEEPING,
            AgentState.NIGHT,
            AgentState.SAFE_IDLE,
        }
    ),
    AgentState.NIGHT: frozenset({AgentState.SLEEPING, AgentState.IDLE, AgentState.SAFE_IDLE}),
    AgentState.SLEEPING: frozenset(
        {
            AgentState.IDLE,
            AgentState.OBSERVING,
            AgentState.NIGHT,
            AgentState.SAFE_IDLE,
        }
    ),
    AgentState.SAFE_IDLE: frozenset(
        {
            AgentState.IDLE,
            AgentState.OBSERVING,
            AgentState.THINKING,
            AgentState.SILENT,
            AgentState.SLEEPING,
        }
    ),
}

# 进程退出时不允许"留宿"的瞬态。
TRANSIENT_STATES = frozenset(
    {AgentState.OBSERVING, AgentState.THINKING, AgentState.CONTACTING}
)


class StateError(RuntimeError):
    """状态机错误基类。"""


class InvalidTransitionError(StateError):
    """不允许的状态转移。"""

    def __init__(self, current: AgentState, target: AgentState) -> None:
        super().__init__(f"不允许从 {current.value} 转移到 {target.value}")
        self.current = current
        self.target = target


@dataclass(frozen=True, slots=True)
class StateChange:
    previous: AgentState
    current: AgentState
    reason: str
    at: datetime


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """可持久化的状态快照。"""

    state: AgentState
    entered_at: datetime
    updated_at: datetime
    previous_state: AgentState | None = None
    thought_state: ThoughtState = ThoughtState.INACTIVE
    conversation_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        import json

        return {
            "state": self.state.value,
            "previous_state": self.previous_state.value if self.previous_state else None,
            "thought_state": self.thought_state.value,
            "entered_at": self.entered_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "conversation_id": self.conversation_id,
            "metadata": json.dumps(dict(self.metadata), ensure_ascii=False, default=str),
        }


_UNSET: object = object()


class AgentStateMachine:
    """内存状态机。持久化由 StateRepository 负责，二者不互相依赖。"""

    def __init__(
        self,
        initial: AgentState = AgentState.IDLE,
        *,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._clock = clock
        now = clock()
        self._state = initial
        self._previous: AgentState | None = None
        self._entered_at = now
        self._updated_at = now
        self._thought_state = ThoughtState.INACTIVE
        self._conversation_id: str | None = None
        self._metadata: dict[str, Any] = {}

    # -- 只读属性 ---------------------------------------------------------- #
    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def previous_state(self) -> AgentState | None:
        return self._previous

    @property
    def entered_at(self) -> datetime:
        return self._entered_at

    @property
    def updated_at(self) -> datetime:
        return self._updated_at

    @property
    def thought_state(self) -> ThoughtState:
        return self._thought_state

    @property
    def conversation_id(self) -> str | None:
        return self._conversation_id

    @property
    def metadata(self) -> Mapping[str, Any]:
        return dict(self._metadata)

    # -- 转移 -------------------------------------------------------------- #
    def can_transition(self, target: AgentState) -> bool:
        if target is self._state:
            return True
        return target in ALLOWED_TRANSITIONS[self._state]

    def transition(
        self,
        target: AgentState,
        reason: str = "",
        *,
        conversation_id: str | None | object = _UNSET,
        metadata: Mapping[str, Any] | None = None,
    ) -> StateChange:
        if not self.can_transition(target):
            raise InvalidTransitionError(self._state, target)

        now = self._clock()
        change = StateChange(
            previous=self._state, current=target, reason=reason, at=now
        )
        self._previous = self._state
        self._state = target
        if target is not change.previous:
            self._entered_at = now
        self._updated_at = now
        if conversation_id is not _UNSET:
            self._conversation_id = conversation_id  # type: ignore[assignment]
        if metadata is not None:
            self._metadata = dict(metadata)
        return change

    def set_thought_state(self, thought: ThoughtState) -> None:
        """仅供未来的 Thought Workspace 调用，本阶段不会被 Agent 使用。"""
        self._thought_state = thought
        self._updated_at = self._clock()

    def set_metadata(self, metadata: Mapping[str, Any]) -> None:
        """覆盖状态附带元信息（重启后恢复静默定时器用）。"""
        self._metadata = dict(metadata)
        self._updated_at = self._clock()

    def snapshot(self) -> StateSnapshot:
        return StateSnapshot(
            state=self._state,
            previous_state=self._previous,
            thought_state=self._thought_state,
            entered_at=self._entered_at,
            updated_at=self._updated_at,
            conversation_id=self._conversation_id,
            metadata=dict(self._metadata),
        )

    def restore(self, snapshot: StateSnapshot) -> None:
        """用快照覆盖内存状态（不做转移校验，只用于启动恢复）。"""
        self._state = snapshot.state
        self._previous = snapshot.previous_state
        self._thought_state = snapshot.thought_state
        self._entered_at = snapshot.entered_at
        self._updated_at = snapshot.updated_at
        self._conversation_id = snapshot.conversation_id
        self._metadata = dict(snapshot.metadata)


# --------------------------------------------------------------------------- #
# 重启恢复
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class RecoveryResult:
    snapshot: StateSnapshot
    changed: bool
    reason: str


def recover_state(
    snapshot: StateSnapshot,
    now: datetime | None = None,
) -> RecoveryResult:
    """按安全策略修正重启前的状态。

    * OBSERVING / THINKING / CONTACTING：进程在动作中途退出，无法判定是否
      已完成，统一落到 SAFE_IDLE 等新事件；
    * WAITING：等待计时依赖内存任务，重启即失效，回到 IDLE；
    * SILENT：只有 silent_until 仍在未来时才保持，否则回 IDLE；
    * NIGHT / SLEEPING：夜间状态不跨进程保留，回到 IDLE；
      如果仍是夜间，下一次 AUTONOMOUS_TICK 会重新进入 NIGHT。
    * 其余状态保持不变。
    """
    now = now or utcnow()
    state = snapshot.state

    if state in TRANSIENT_STATES:
        return _recovered(snapshot, AgentState.SAFE_IDLE, now, "瞬态状态重启恢复")

    if state is AgentState.WAITING:
        return _recovered(snapshot, AgentState.IDLE, now, "WAITING 计时器已随进程丢失")

    if state is AgentState.SILENT:
        until_raw = snapshot.metadata.get("silent_until")
        until = _parse_dt(until_raw)
        if until is not None and until > now:
            return RecoveryResult(snapshot=snapshot, changed=False, reason="静默期未结束")
        return _recovered(snapshot, AgentState.IDLE, now, "静默期已结束")

    if state in (AgentState.NIGHT, AgentState.SLEEPING):
        return _recovered(snapshot, AgentState.IDLE, now, "夜间状态重启恢复")

    return RecoveryResult(snapshot=snapshot, changed=False, reason="状态保持")


def _recovered(
    snapshot: StateSnapshot, target: AgentState, now: datetime, reason: str
) -> RecoveryResult:
    new_snapshot = replace(
        snapshot,
        state=target,
        previous_state=snapshot.state,
        entered_at=now,
        updated_at=now,
        thought_state=ThoughtState.INACTIVE,
    )
    return RecoveryResult(snapshot=new_snapshot, changed=True, reason=reason)


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None