"""行为守卫。

这是 LLM 与真实世界之间的最后一道门。即使 Validator 放行，也要在这里按
**事件来源 / 会话关系 / 频率 / 静默时段** 再判一次。

阶段一的核心限制：除回复当前用户消息外，不允许主动发消息。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Callable

from ..config import BehaviorConfig, parse_quiet_window
from ..contact.intent import ContactIntent
from ..llm.schemas import Action, ActionType
from .event import EventType
from .state import AgentState, utcnow


@dataclass(frozen=True, slots=True)
class GuardContext:
    event_type: EventType
    conversation_id: str | None
    known_conversation_ids: frozenset[str]
    agent_state: AgentState
    last_outgoing_at: datetime | None
    now: datetime
    contact_approved: bool = False


@dataclass(frozen=True, slots=True)
class GuardVerdict:
    allowed: bool
    reason: str
    code: str = "OK"

    @classmethod
    def allow(cls, reason: str = "允许") -> "GuardVerdict":
        return cls(allowed=True, reason=reason)

    @classmethod
    def deny(cls, code: str, reason: str) -> "GuardVerdict":
        return cls(allowed=False, reason=reason, code=code)


class BehaviorGuard:
    def __init__(
        self,
        config: BehaviorConfig,
        *,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._config = config
        self._clock = clock
        self._allowed = {item.upper() for item in config.allowed_actions}
        self._quiet_windows = [parse_quiet_window(w) for w in config.quiet_hours]

    # -- 查询 -------------------------------------------------------------- #
    def in_quiet_hours(self, now: datetime | None = None) -> bool:
        if not self._quiet_windows:
            return False
        current = (now or self._clock()).time()
        for sh, sm, eh, em in self._quiet_windows:
            start = time(hour=sh, minute=sm)
            end = time(hour=eh, minute=em)
            if start <= end:
                if start <= current < end:
                    return True
            elif current >= start or current < end:  # 跨午夜
                return True
        return False

    # -- 主检查 ------------------------------------------------------------ #
    def check(self, action: Action, ctx: GuardContext) -> GuardVerdict:
        if action.type.value not in self._allowed:
            return GuardVerdict.deny(
                "ACTION_NOT_ALLOWED", f"动作 {action.type.value} 未被配置允许"
            )

        if action.type is ActionType.DO_NOTHING:
            return GuardVerdict.allow("无副作用")

        if action.type is ActionType.SEND_MESSAGE:
            return self._check_send(action, ctx)

        if action.type is ActionType.WAIT:
            if self.in_quiet_hours(ctx.now):
                return GuardVerdict.allow("静默时段仍可等待")
            return GuardVerdict.allow("等待是安全动作")

        if action.type is ActionType.ENTER_SILENT:
            return GuardVerdict.allow("进入静默是安全动作")

        if action.type is ActionType.THINK:
            return GuardVerdict.allow("思考不直接产生外部副作用")

        return GuardVerdict.deny(
            "ACTION_NOT_IMPLEMENTED", f"动作 {action.type.value} 未实现"
        )  # pragma: no cover

    # -- 主动联系 ---------------------------------------------------------- #
    def check_contact(self, intent: ContactIntent, ctx: GuardContext) -> GuardVerdict:
        """CONTACT_INTENT 的第一道硬门：开关、状态、静默时段、频率、内容。

        更细的夜间/连续无回复/重复检查由 ContactManager 继续判。
        """
        if not self._config.allow_autonomous_contact:
            return GuardVerdict.deny("CONTACT_DISABLED", "主动联系总开关已关闭")

        if ctx.agent_state is AgentState.SILENT:
            return GuardVerdict.deny("SILENT_MODE", "静默状态下不主动联系")

        if not intent.conversation_id:
            return GuardVerdict.deny("MISSING_TARGET", "联系意图缺少会话 id")

        if self.in_quiet_hours(ctx.now) and self._config.quiet_blocks_send_message:
            return GuardVerdict.deny("QUIET_HOURS", "静默时段禁止主动联系")

        if ctx.last_outgoing_at is not None:
            elapsed = (ctx.now - ctx.last_outgoing_at).total_seconds()
            if elapsed < self._config.min_seconds_between_messages:
                return GuardVerdict.deny(
                    "RATE_LIMITED",
                    f"距上次发送仅 {elapsed:.1f}s，低于下限 "
                    f"{self._config.min_seconds_between_messages}s",
                )

        draft = intent.message_draft.strip()
        if not draft:
            return GuardVerdict.deny("EMPTY_MESSAGE", "联系内容为空")
        if len(draft) > self._config.max_message_chars:
            return GuardVerdict.deny("TOO_LONG", "联系内容超过长度限制")

        return GuardVerdict.allow("主动联系意图通过 Guard")

    # -- 发送 -------------------------------------------------------------- #
    def _check_send(self, action: Action, ctx: GuardContext) -> GuardVerdict:
        if (
            ctx.event_type is not EventType.USER_MESSAGE
            and not ctx.contact_approved
            and not self._config.allow_send_on_autonomous_tick
        ):
            return GuardVerdict.deny(
                "AUTONOMOUS_SEND_DISABLED",
                "非用户消息触发，且未经主动联系闸门批准",
            )

        target = action.parameters.get("conversation_id")
        if not isinstance(target, str) or not target:
            return GuardVerdict.deny("MISSING_TARGET", "缺少会话 id")

        if ctx.conversation_id and target != ctx.conversation_id:
            return GuardVerdict.deny(
                "TARGET_MISMATCH", "只允许回复当前事件所属会话"
            )
        if target not in ctx.known_conversation_ids:
            return GuardVerdict.deny("UNKNOWN_TARGET", f"未知会话: {target}")

        if self.in_quiet_hours(ctx.now) and self._config.quiet_blocks_send_message:
            return GuardVerdict.deny("QUIET_HOURS", "静默时段禁止发送消息")

        if ctx.last_outgoing_at is not None:
            elapsed = (ctx.now - ctx.last_outgoing_at).total_seconds()
            if elapsed < self._config.min_seconds_between_messages:
                return GuardVerdict.deny(
                    "RATE_LIMITED",
                    f"距上次发送仅 {elapsed:.1f}s，低于下限 "
                    f"{self._config.min_seconds_between_messages}s",
                )

        content = action.parameters.get("content")
        if isinstance(content, str) and len(content) > self._config.max_message_chars:
            return GuardVerdict.deny("TOO_LONG", "消息超过长度限制")

        return GuardVerdict.allow("满足发送条件")


__all__ = ["BehaviorGuard", "GuardContext", "GuardVerdict"]