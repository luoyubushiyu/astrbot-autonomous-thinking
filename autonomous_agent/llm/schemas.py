"""LLM 输出的动作定义。

LLM 能"说"的动作是有限集合。程序侧的 Schema Validator 与 Behavior Guard
只认这里登记过的动作和参数，LLM 不可能靠自由发挥拿到未声明的能力。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class ActionType(str, Enum):
    DO_NOTHING = "DO_NOTHING"
    SEND_MESSAGE = "SEND_MESSAGE"
    WAIT = "WAIT"
    ENTER_SILENT = "ENTER_SILENT"
    THINK = "THINK"

    @classmethod
    def parse(cls, value: Any) -> "ActionType":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ValueError(f"无效的动作类型: {value!r}")
        try:
            return cls(value.strip().upper())
        except ValueError:
            raise ValueError(f"未知动作: {value!r}") from None


class ParameterKind(str, Enum):
    STRING = "STRING"
    INTEGER = "INTEGER"
    BOOLEAN = "BOOLEAN"
    ARRAY = "ARRAY"


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    name: str
    kind: ParameterKind
    required: bool = False
    min_value: int | None = None
    max_value: int | None = None
    max_length: int | None = None


@dataclass(frozen=True, slots=True)
class Action:
    type: ActionType
    parameters: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def do_nothing(cls, reason: str = "no decision") -> "Action":
        return cls(ActionType.DO_NOTHING, {"reason": reason})


@dataclass(frozen=True, slots=True)
class Decision:
    reasoning: str
    action: Action
    confidence: float = 0.0

    @classmethod
    def do_nothing(cls, reason: str = "fallback", confidence: float = 0.0) -> "Decision":
        return cls(reasoning=reason, action=Action.do_nothing(reason), confidence=confidence)


ACTION_PARAMETER_SPECS: dict[ActionType, tuple[ParameterSpec, ...]] = {
    ActionType.DO_NOTHING: (
        ParameterSpec("reason", ParameterKind.STRING, required=False, max_length=500),
    ),
    ActionType.SEND_MESSAGE: (
        ParameterSpec("conversation_id", ParameterKind.STRING, required=True, max_length=128),
        ParameterSpec("content", ParameterKind.STRING, required=True, max_length=2000),
        ParameterSpec("reply_to", ParameterKind.STRING, required=False, max_length=128),
    ),
    ActionType.WAIT: (
        ParameterSpec(
            "seconds", ParameterKind.INTEGER, required=True, min_value=1, max_value=3600
        ),
        ParameterSpec("reason", ParameterKind.STRING, required=False, max_length=500),
    ),
    ActionType.ENTER_SILENT: (
        ParameterSpec(
            "minutes", ParameterKind.INTEGER, required=True, min_value=1, max_value=1440
        ),
        ParameterSpec("reason", ParameterKind.STRING, required=False, max_length=500),
    ),
    # 只由思考引擎使用，普通对话请求不会放行 THINK。
    ActionType.THINK: (
        ParameterSpec("thought_id", ParameterKind.STRING, required=True, max_length=64),
        ParameterSpec("insight", ParameterKind.STRING, required=True, max_length=2000),
        ParameterSpec("uncertainty", ParameterKind.STRING, required=False, max_length=500),
        ParameterSpec("disposition", ParameterKind.STRING, required=True, max_length=32),
        ParameterSpec("known", ParameterKind.ARRAY, required=False),
        ParameterSpec("new_questions", ParameterKind.ARRAY, required=False),
        ParameterSpec("next_step", ParameterKind.STRING, required=False, max_length=300),
        ParameterSpec("memory_candidates", ParameterKind.ARRAY, required=False),
        ParameterSpec(
            "next_check_minutes",
            ParameterKind.INTEGER,
            required=False,
            min_value=5,
            max_value=1440,
        ),
        ParameterSpec(
            "contact_message", ParameterKind.STRING, required=False, max_length=2000
        ),
    ),
}


DECISION_JSON_CONTRACT = """\
只输出一个 JSON 对象，不要 Markdown 代码块，不要任何额外文字。结构：
{
  "reasoning": "一句话说明为什么这么判断（不要写完整思维链）",
  "action": {
    "type": "DO_NOTHING | SEND_MESSAGE | WAIT | ENTER_SILENT",
    "parameters": { ... }
  },
  "confidence": 0.0
}
参数规则：
- DO_NOTHING: {"reason": "可选"}
- SEND_MESSAGE: {"conversation_id": "已有会话 id", "content": "回复内容", "reply_to": "可选"}
- WAIT: {"seconds": 1..3600, "reason": "可选"}
- ENTER_SILENT: {"minutes": 1..1440, "reason": "可选"}
不确定、缺少必要信息或没有合适的动作时，必须选择 DO_NOTHING。
"""


THINK_JSON_CONTRACT = """\
只输出一个 JSON 对象，不要 Markdown 代码块，不要任何额外文字。结构：
{
  "reasoning": "一句话说明这次思考的方向，不要写完整思维链",
  "action": {
    "type": "THINK",
    "parameters": {
      "thought_id": "本次思考的 Thought id",
      "insight": "这次想到的结果（结论/疑点/新角度）",
      "known": ["这次确认了的信息，可空"],
      "uncertainty": "可选：还不确定的地方",
      "disposition": "CONTINUE | KEEP | DRAWER | DIARY | SPEAK | BLACKBOX | DISCARD | WAIT | FOLLOW_UP | RESOLVED",
      "new_questions": [{"title": "...", "content": "...", "importance": 0.5}],
      "next_step": "下一步要做什么（一句话）",
      "next_check_minutes": 60,
      "contact_message": "仅 disposition=SPEAK 时提供",
      "memory_candidates": [
        {"type": "FACT | PREFERENCE | EVENT | RELATIONSHIP | TOPIC | INFERENCE | THOUGHT | EPISODIC_MEMORY | SUMMARY_MEMORY",
         "content": "值得长期记住的一句话",
         "confidence": 0.0,
         "importance": 0.0,
         "keywords": ["可选"]}
      ]
    }
  },
  "confidence": 0.0
}
规则：
- insight 必须具体，不要重复 Thought 原文；
- 还没想完时用 CONTINUE，程序会保留文本，下一次 Tick 再接着想（不要一次写完整条思维链）；
- 已有明确结论、不再需要继续时用 RESOLVED；
- 只有真的值得说给用户听才用 SPEAK，否则 KEEP/DRAWER/DIARY；
- 没有新问题就用空数组，最多 3 条；
- disposition=WAIT 时用 next_check_minutes 表示过多久再看；
- memory_candidates 是 SAVE_MEMORY 意图，只放"以后还用得上"的信息，最多 3 条，没有就给空数组；
- 拿不准的推测用 INFERENCE，不要写成 FACT；程序还会再校验一遍，不合格的不会入库。
"""


DEFAULT_DECISION_JSON = """\
{
  "reasoning": "当前没有足够信息，保持不动。",
  "action": {"type": "DO_NOTHING", "parameters": {"reason": "insufficient context"}},
  "confidence": 0.0
}"""


__all__ = [
    "ACTION_PARAMETER_SPECS",
    "Action",
    "ActionType",
    "DECISION_JSON_CONTRACT",
    "DEFAULT_DECISION_JSON",
    "Decision",
    "ParameterKind",
    "ParameterSpec",
    "THINK_JSON_CONTRACT",
]