"""上下文构建。

把"Agent 现在看到了什么"整理成两部分：
1. :class:`AgentContext` —— 程序内部使用的事实集合；
2. :class:`LLMRequest`  —— 只包含事实，不包含任何数据库/文件/API 句柄。

LLM 永远不会拿到 SQL、路径或 token 列表。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from ..database.database import MessageRecord, MessageRepository
from ..llm.adapter import LLMMessage, LLMRequest
from ..llm.schemas import DECISION_JSON_CONTRACT
from .event import Event, EventType
from .state import AgentState, AgentStateMachine, ThoughtState, utcnow

SYSTEM_PROMPT = """你是一个 QQ Bot 的决策模块，运行在“自主思考系统”里。

你的职责：根据当前状态、事件和最近对话，提出**一个**下一步建议。
你不是执行者：你没有数据库、文件系统、QQ API 或定时器的权限。
你无法发送消息——你只能建议程序去发送，程序会做权限检查。

必须遵守：
1. 只输出一个 JSON 对象，不要 Markdown 代码块、不要多余解释。
2. 不确定、信息不足、或没有必要时，输出 DO_NOTHING。
3. 不要编造用户、会话或消息 id。
4. 语气自然，像人在聊天，不要客服腔。
"""


@dataclass(frozen=True, slots=True)
class AgentContext:
    event: Event
    agent_state: AgentState
    thought_state: ThoughtState
    conversation_id: str | None
    recent_messages: tuple[MessageRecord, ...]
    known_conversation_ids: frozenset[str]
    now: datetime


class ContextBuilder:
    def __init__(
        self,
        messages: MessageRepository,
        *,
        history_limit: int = 10,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._messages = messages
        self._history_limit = history_limit
        self._clock = clock

    async def build(
        self,
        event: Event,
        machine: AgentStateMachine,
        *,
        conversation_id: str | None = None,
    ) -> AgentContext:
        conversation_id = conversation_id or event.conversation_id()
        known = await self._messages.conversation_ids()
        if event.conversation_id():
            known.add(event.conversation_id() or "")

        recent: tuple[MessageRecord, ...] = ()
        if conversation_id:
            recent = tuple(
                await self._messages.recent(conversation_id, self._history_limit)
            )

        return AgentContext(
            event=event,
            agent_state=machine.state,
            thought_state=machine.thought_state,
            conversation_id=conversation_id,
            recent_messages=recent,
            known_conversation_ids=frozenset(known),
            now=self._clock(),
        )

    def build_request(self, ctx: AgentContext) -> LLMRequest:
        system_prompt = f"{SYSTEM_PROMPT}\n{DECISION_JSON_CONTRACT}"
        return LLMRequest(
            system_prompt=system_prompt,
            messages=(LLMMessage(role="user", content=render_event(ctx)),),
            metadata={
                "event_id": ctx.event.id,
                "event_type": ctx.event.type.value,
                "agent_state": ctx.agent_state.value,
            },
        )


def render_event(ctx: AgentContext) -> str:
    """把上下文渲染成一段纯文本。"""
    event = ctx.event
    lines: list[str] = []
    lines.append(f"当前时间：{ctx.now.isoformat()}")
    lines.append(f"Agent 状态：{ctx.agent_state.value}")
    lines.append(f"思考态：{ctx.thought_state.value}")
    lines.append(f"事件类型：{event.type.value}")

    if event.type is EventType.USER_MESSAGE:
        payload = event.payload
        lines.append(f"会话 id：{payload.get('conversation_id')}")
        lines.append(f"发送者：{payload.get('user_id')}")
        if payload.get("is_group"):
            lines.append(f"群 id：{payload.get('group_id')}")
        lines.append("")
        lines.append("对方刚发来：")
        lines.append(str(payload.get("content", ""))[:1000])
    elif event.type is EventType.AUTONOMOUS_TICK:
        lines.append(
            f"这是第 {event.payload.get('tick')} 次自主心跳，没有新的用户消息。"
        )
        lines.append("你只能选择 DO_NOTHING、WAIT 或 ENTER_SILENT。")
    elif event.type is EventType.ALARM_TRIGGERED:
        lines.append(f"闹钟触发：{event.payload.get('alarm_id')}")
        lines.append("该事件只用于解除等待/静默，不需要主动联系。")
    elif event.type is EventType.CONTACT_TIMEOUT:
        lines.append("联系超时，对方没有回应。")
    elif event.type is EventType.BOT_START:
        lines.append("Bot 刚刚启动。")
    elif event.type is EventType.BOT_STOP:
        lines.append("Bot 即将停止。")
    else:  # pragma: no cover - 枚举已覆盖
        lines.append(f"事件内容：{dict(event.payload)}")

    if ctx.recent_messages:
        lines.append("")
        lines.append("最近对话（旧 → 新）：")
        for record in ctx.recent_messages:
            speaker = "我" if record.role == "assistant" else f"用户({record.user_id or '?'})"
            lines.append(f"- {speaker}: {record.content[:500]}")
    elif ctx.conversation_id:
        lines.append("")
        lines.append("最近对话：无")

    lines.append("")
    lines.append("请给出你的建议，只输出 JSON。")
    return "\n".join(lines)


__all__ = ["AgentContext", "ContextBuilder", "SYSTEM_PROMPT", "render_event"]