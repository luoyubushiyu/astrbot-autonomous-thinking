"""思考文本：一个 Thought 的"纸面状态"。

文档要求思考有连续状态：下一次思考先读回同一份文本，而不是每次从零开始。
这里定义文本的渲染、解析和压缩，全部围绕可读字段，不保存任何隐藏的思维链。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

from .models import Thought, utcnow

_SECTIONS = (
    "核心问题",
    "已知信息",
    "当前思考",
    "关键发现",
    "不确定",
    "新问题",
    "相关问题",
    "下一步",
    "继续思考条件",
    "停止原因",
)

_LIST_SECTIONS = frozenset({"已知信息", "不确定", "新问题", "相关问题"})


@dataclass(slots=True)
class ThinkingState:
    known: list[str] = field(default_factory=list)
    current_thinking: str = ""
    insight: str = ""
    uncertainty: list[str] = field(default_factory=list)
    new_questions: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    next_step: str = ""
    continue_condition: str = ""
    stop_reason: str = ""
    last_decision: str = ""
    turns: int = 0
    history: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "known": list(self.known),
            "current_thinking": self.current_thinking,
            "insight": self.insight,
            "uncertainty": list(self.uncertainty),
            "new_questions": list(self.new_questions),
            "related": list(self.related),
            "next_step": self.next_step,
            "continue_condition": self.continue_condition,
            "stop_reason": self.stop_reason,
            "last_decision": self.last_decision,
            "turns": self.turns,
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ThinkingState":
        if not isinstance(data, dict):
            return cls()
        return cls(
            known=[str(x) for x in data.get("known", []) if str(x).strip()],
            current_thinking=str(data.get("current_thinking", "")),
            insight=str(data.get("insight", "")),
            uncertainty=[str(x) for x in data.get("uncertainty", []) if str(x).strip()],
            new_questions=[str(x) for x in data.get("new_questions", []) if str(x).strip()],
            related=[str(x) for x in data.get("related", []) if str(x).strip()],
            next_step=str(data.get("next_step", "")),
            continue_condition=str(data.get("continue_condition", "")),
            stop_reason=str(data.get("stop_reason", "")),
            last_decision=str(data.get("last_decision", "")),
            turns=int(data.get("turns", 0) or 0),
            history=[str(x) for x in data.get("history", []) if str(x).strip()],
        )


def state_from_thought(thought: Thought) -> ThinkingState:
    state = ThinkingState.from_dict(thought.metadata.get("thinking"))
    if not state.current_thinking:
        state.current_thinking = thought.insight or ""
    if not state.insight:
        state.insight = thought.insight or ""
    if thought.uncertainty and not state.uncertainty:
        state.uncertainty = [thought.uncertainty]
    if thought.next_action and not state.next_step:
        state.next_step = thought.next_action
    return state


def render_thinking_text(thought: Thought, state: ThinkingState, *, now: datetime | None = None) -> str:
    moment = now or utcnow()

    def block(name: str, body: str) -> str:
        return f"{name}:\n{body.strip()}\n" if body.strip() else f"{name}:\n\n"

    def bullets(items: Iterable[str]) -> str:
        return "\n".join(f"- {item}" for item in items)

    lines = [
        f"ID: {thought.id}",
        f"类型: {thought.type.value}",
        f"标题: {thought.title}",
        f"创建时间: {thought.created_at.isoformat()}",
        f"最后思考时间: {moment.isoformat()}",
        f"状态: {thought.status.value}",
        f"已思考次数: {state.turns}",
        "",
        block("核心问题", thought.content or thought.title),
        block("已知信息", bullets(state.known)),
        block("当前思考", state.current_thinking),
        block("关键发现", state.insight),
        block("不确定", bullets(state.uncertainty)),
        block("新问题", bullets(state.new_questions)),
        block("相关问题", bullets(state.related)),
        block("下一步", state.next_step),
        block("继续思考条件", state.continue_condition),
        block("停止原因", state.stop_reason),
    ]
    if state.history:
        lines.append("历史摘要:")
        lines.extend(f"- {item}" for item in state.history[-8:])
        lines.append("")
    return "\n".join(lines)


def parse_thinking_text(text: str) -> ThinkingState:
    """从文本还原状态；格式损坏时尽量保留能读出来的部分。"""
    state = ThinkingState()
    current: str | None = None
    buffers: dict[str, list[str]] = {
        name: [] for name in (*_SECTIONS, "历史摘要")
    }

    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("已思考次数:"):
            try:
                state.turns = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
            continue
        header = line[:-1].strip() if line.endswith(":") else None
        if header in _SECTIONS:
            current = header
            continue
        if header == "历史摘要":
            current = "历史摘要"
            continue
        if current is None:
            continue
        buffers[current].append(line)

    def joined(name: str) -> str:
        return "\n".join(buffers[name]).strip()

    def items(name: str) -> list[str]:
        values = []
        for line in buffers[name]:
            stripped = line.strip()
            if stripped.startswith("- "):
                values.append(stripped[2:].strip())
        return [value for value in values if value]

    state.known = items("已知信息")
    state.current_thinking = joined("当前思考")
    state.insight = joined("关键发现")
    state.uncertainty = items("不确定")
    state.new_questions = items("新问题")
    state.related = items("相关问题")
    state.next_step = joined("下一步")
    state.continue_condition = joined("继续思考条件")
    state.stop_reason = joined("停止原因")
    history = []
    for line in buffers.get("历史摘要", []):
        stripped = line.strip()
        if stripped.startswith("- "):
            history.append(stripped[2:].strip())
    state.history = history
    return state


def compress_thinking_text(text: str, *, max_chars: int = 8000) -> str:
    """文本过长时保留头部的当前状态，压缩历史摘要。"""
    if len(text) <= max_chars:
        return text
    head, _, tail = text.partition("历史摘要:")
    keep = max(0, max_chars - len(head) - 200)
    if keep <= 0:
        return head
    summary = tail.strip()
    if len(summary) <= keep:
        return text
    return f"{head}历史摘要:\n{summary[-keep:]}\n"


__all__ = [
    "ThinkingState",
    "compress_thinking_text",
    "parse_thinking_text",
    "render_thinking_text",
    "state_from_thought",
]