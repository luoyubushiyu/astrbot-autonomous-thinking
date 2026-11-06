"""动作执行器。

LLM 的建议经过 Validator 与 BehaviorGuard 之后才会到这里。执行器**只**执行
一小组白名单动作，并且每次执行都落库/记指标。

执行器不抛业务异常：任何失败都转成 :class:`ExecutionResult` 的 ``error``，
由 Agent 决定是否进入 SAFE_IDLE。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping

from ..config import BehaviorConfig
from ..database.database import (
    MessageRepository,
    MetricsRepository,
    new_message_id,
)
from ..llm.schemas import Action, ActionType
from ..qq.adapter import OutgoingMessage, QQAdapter
from .state import utcnow

logger = logging.getLogger("autonomous_agent.executor")


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    executed: bool
    action: ActionType
    detail: str = ""
    sent_message_id: str | None = None
    wait_seconds: int | None = None
    silent_until: datetime | None = None
    error: str | None = None


class Executor:
    def __init__(
        self,
        *,
        qq: QQAdapter,
        messages: MessageRepository,
        metrics: MetricsRepository,
        behavior: BehaviorConfig,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._qq = qq
        self._messages = messages
        self._metrics = metrics
        self._behavior = behavior
        self._clock = clock

    async def execute(self, action: Action, ctx: Any) -> ExecutionResult:
        try:
            if action.type is ActionType.DO_NOTHING:
                return await self._do_nothing(action)
            if action.type is ActionType.SEND_MESSAGE:
                return await self._send_message(action)
            if action.type is ActionType.WAIT:
                return await self._wait(action)
            if action.type is ActionType.ENTER_SILENT:
                return await self._enter_silent(action)
            if action.type is ActionType.THINK:
                return await self._think(action)
        except Exception as exc:  # noqa: BLE001 - 执行失败必须转成结果
            logger.exception("动作执行异常: %s", action.type.value)
            await self._metric("action.failed", 1, {"action": action.type.value})
            return ExecutionResult(
                executed=False,
                action=action.type,
                detail="执行异常",
                error=f"{type(exc).__name__}: {exc}",
            )

        return ExecutionResult(
            executed=False,
            action=action.type,
            detail="未实现的动作",
            error="UNSUPPORTED_ACTION",
        )

    # -- 各动作 ------------------------------------------------------------ #
    async def _do_nothing(self, action: Action) -> ExecutionResult:
        await self._metric(
            "action.do_nothing", 1, {"reason": _short(action.parameters.get("reason"))}
        )
        return ExecutionResult(
            executed=True, action=action.type, detail="no-op"
        )

    async def _send_message(self, action: Action) -> ExecutionResult:
        conversation_id = str(action.parameters["conversation_id"])
        content = str(action.parameters["content"])
        reply_to = action.parameters.get("reply_to")
        outgoing = OutgoingMessage(
            conversation_id=conversation_id,
            content=content,
            reply_to=str(reply_to) if reply_to else None,
        )
        result = await self._qq.send_message(outgoing)
        if not result.ok:
            await self._metric("message.send_failed", 1, {"conversation": conversation_id})
            return ExecutionResult(
                executed=False,
                action=action.type,
                detail="QQ 发送失败",
                error=result.error or "SEND_FAILED",
            )

        message_id = result.message_id or new_message_id("out")
        await self._messages.add(
            conversation_id=conversation_id,
            message_id=message_id,
            role="assistant",
            content=content,
            created_at=self._clock(),
            metadata={"reply_to": reply_to} if reply_to else None,
        )
        await self._metric("message.sent", 1, {"conversation": conversation_id})
        return ExecutionResult(
            executed=True,
            action=action.type,
            detail="消息已发送",
            sent_message_id=message_id,
        )

    async def _wait(self, action: Action) -> ExecutionResult:
        seconds = int(action.parameters["seconds"])
        seconds = max(1, min(seconds, 3600))
        await self._metric("action.wait", 1, {"seconds": seconds})
        return ExecutionResult(
            executed=True,
            action=action.type,
            detail=f"等待 {seconds}s",
            wait_seconds=seconds,
        )

    async def _think(self, action: Action) -> ExecutionResult:
        # 真正的思考由 ThoughtEngine 完成；执行器只确认没有外部副作用。
        await self._metric("action.think", 1, {"thought": _short(action.parameters.get("thought_id"))})
        return ExecutionResult(executed=True, action=action.type, detail="thought handled by engine")

    async def _enter_silent(self, action: Action) -> ExecutionResult:
        minutes = int(action.parameters["minutes"])
        minutes = max(1, min(minutes, self._behavior.max_silent_minutes))
        until = self._clock() + timedelta(minutes=minutes)
        await self._metric("action.enter_silent", 1, {"minutes": minutes})
        return ExecutionResult(
            executed=True,
            action=action.type,
            detail=f"静默至 {until.isoformat()}",
            silent_until=until,
        )

    # -- 内部 -------------------------------------------------------------- #
    async def _metric(
        self,
        name: str,
        value: float = 1.0,
        tags: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            await self._metrics.record(name, value, tags)
        except Exception:  # noqa: BLE001 - 指标失败不影响主流程
            logger.warning("指标写入失败: %s", name, exc_info=True)


def _short(value: Any, limit: int = 200) -> str:
    text = str(value or "")
    return text[:limit]


__all__ = ["ExecutionResult", "Executor"]