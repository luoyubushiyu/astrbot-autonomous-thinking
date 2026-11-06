"""核心包。

这里刻意使用惰性导出：``database.database`` 需要引用 ``core.state``，而
``core.agent`` 又依赖数据库。如果本文件在导入时急切地拉起 Agent，就会形成
循环导入。因此所有名字都通过 ``__getattr__`` 按需加载。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "Agent": (".agent", "Agent"),
    "HandleResult": (".agent", "HandleResult"),
    "AgentContext": (".context", "AgentContext"),
    "ContextBuilder": (".context", "ContextBuilder"),
    "EmotionModel": (".emotion", "EmotionModel"),
    "EmotionState": (".emotion", "EmotionState"),
    "ContactIntent": (".thought_engine", "ContactIntent"),
    "ThinkResult": (".thought_engine", "ThinkResult"),
    "ThoughtEngine": (".thought_engine", "ThoughtEngine"),
    "Event": (".event", "Event"),
    "EventError": (".event", "EventError"),
    "EventType": (".event", "EventType"),
    "InvalidEventError": (".event", "InvalidEventError"),
    "ExecutionResult": (".executor", "ExecutionResult"),
    "Executor": (".executor", "Executor"),
    "BehaviorGuard": (".guard", "BehaviorGuard"),
    "GuardContext": (".guard", "GuardContext"),
    "GuardVerdict": (".guard", "GuardVerdict"),
    "AgentState": (".state", "AgentState"),
    "AgentStateMachine": (".state", "AgentStateMachine"),
    "InvalidTransitionError": (".state", "InvalidTransitionError"),
    "RecoveryResult": (".state", "RecoveryResult"),
    "StateChange": (".state", "StateChange"),
    "StateSnapshot": (".state", "StateSnapshot"),
    "ThoughtState": (".state", "ThoughtState"),
    "recover_state": (".state", "recover_state"),
    "utcnow": (".state", "utcnow"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = import_module(module_name, __name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value