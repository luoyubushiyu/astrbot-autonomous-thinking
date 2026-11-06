"""提示词模板。

分两类：
- 回复/决策：对用户消息做一次回复选择（只允许 DO_NOTHING/SEND_MESSAGE/WAIT/ENTER_SILENT）；
- 思考：对一条已选中的 Thought 做一次思考（只允许 THINK）。

程序不把"完整思维链"存回来，只保留 insight / uncertainty / 下一步。
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

from ..thought.models import Thought
from .schemas import DECISION_JSON_CONTRACT, THINK_JSON_CONTRACT

REPLY_SYSTEM_PROMPT = """\
你是一个 QQ 聊天机器人，正在和用户对话。
你只能输出一个 JSON 决策，不能直接发消息、不能读写数据库、不能调用工具。
你可以选择 DO_NOTHING、SEND_MESSAGE、WAIT、ENTER_SILENT。
不确定、没有合适的话、或者当前不该说话时，选择 DO_NOTHING。
不要编造事实；不要声称自己执行了任何操作。
"""

THINK_SYSTEM_PROMPT = """\
你是一个有自己节奏的 QQ 聊天机器人，现在在"独自思考"。
你会收到一条被程序选中的 Thought、相关记忆、最近对话和你当前的状态。
你的任务是想清楚这条 Thought，并决定怎么处理它。
你只能输出一个 JSON 决策，动作必须是 THINK。
不要输出完整思维链，只给结论、疑点和下一步。
disposition 的含义：
- CONTINUE：还没想完，保留这条思考，下一次 Tick 接着想
- KEEP：记下来，留着以后用
- DRAWER：先放抽屉，等有新信息再想
- DIARY：这值得写进今天的日记
- SPEAK：值得主动说给用户听（必须提供 contact_message）
- BLACKBOX：不想再看到它，封存
- DISCARD：没价值，丢掉
- WAIT：过一段时间再想（用 next_check_minutes）
- FOLLOW_UP：需要追问/后续行动
- RESOLVED：已经想清楚，可以收尾
一次只推进一步；没想完就用 CONTINUE，不要一次编完整条思维链。
"""


def render_reply_system(allowed_actions: Iterable[str]) -> str:
    allowed = ", ".join(sorted(allowed_actions))
    return f"{REPLY_SYSTEM_PROMPT}\n本次允许的动作：{allowed}。\n{DECISION_JSON_CONTRACT}"


def render_think_system() -> str:
    return f"{THINK_SYSTEM_PROMPT}\n{THINK_JSON_CONTRACT}"


def render_think_user(
    thought: Thought,
    *,
    now: datetime,
    memories: Iterable[object] = (),
    topics: Iterable[object] = (),
    recent_messages: Iterable[object] = (),
    emotion: object | None = None,
    thinking_text: str | None = None,
) -> str:
    lines = [
        f"现在：{now.isoformat()}",
        f"Thought id：{thought.id}",
        f"类型：{thought.type.value}",
        f"标题：{thought.title}",
        f"内容：{thought.content}",
        f"来源：{thought.source}",
        f"重要度：{thought.importance:.2f} 好奇度：{thought.curiosity:.2f} 相关度：{thought.relevance:.2f}",
        f"已思考次数：{thought.revisit_count} 复活次数：{thought.reactivation_count}",
    ]
    if thinking_text:
        lines.append("这是你上次留下的思考文本，请接着它继续，而不是从头重想：")
        lines.append(thinking_text.rstrip())
    if thought.insight and not thinking_text:
        lines.append(f"上次的结论：{thought.insight}")
    if thought.uncertainty:
        lines.append(f"上次的疑点：{thought.uncertainty}")

    memory_lines = [f"- {getattr(m, 'content', m)}" for m in memories]
    if memory_lines:
        lines.append("相关记忆：")
        lines.extend(memory_lines[:8])

    topic_lines = [f"- {getattr(t, 'name', t)}" for t in topics]
    if topic_lines:
        lines.append("相关话题：")
        lines.extend(topic_lines[:8])

    recent_lines = []
    for item in recent_messages:
        role = getattr(item, "role", "?")
        content = getattr(item, "content", "")
        recent_lines.append(f"- {role}: {content}")
    if recent_lines:
        lines.append("最近对话：")
        lines.extend(recent_lines[-6:])

    if emotion is not None:
        lines.append(
            "当前情绪："
            f"孤独 {getattr(emotion, 'loneliness', 0):.2f}，"
            f"好奇 {getattr(emotion, 'curiosity', 0):.2f}，"
            f"兴趣 {getattr(emotion, 'interest', 0):.2f}，"
            f"能量 {getattr(emotion, 'energy', 0):.2f}"
        )
    lines.append("请只输出 THINK 决策 JSON。")
    return "\n".join(lines)


__all__ = [
    "REPLY_SYSTEM_PROMPT",
    "THINK_SYSTEM_PROMPT",
    "render_reply_system",
    "render_think_system",
    "render_think_user",
]