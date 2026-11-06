"""LLM 输出的 Schema Validator。

LLM 的文本先变成 JSON，再过这里：
1. 解析失败 / 空响应 → 明确错误码；
2. 顶层结构、动作、参数类型/范围、允许的动作集合 → 结构校验；
3. 只会返回 ``Decision``，绝不把原始文本交给执行层。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from .schemas import (
    ACTION_PARAMETER_SPECS,
    Action,
    ActionType,
    Decision,
    ParameterKind,
)


class ValidationError(ValueError):
    code = "VALIDATION_ERROR"

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or message


class EmptyResponseError(ValidationError):
    code = "EMPTY_RESPONSE"


class JSONParseError(ValidationError):
    code = "JSON_PARSE_ERROR"


class SchemaValidationError(ValidationError):
    code = "SCHEMA_ERROR"


class UnknownActionError(ValidationError):
    code = "UNKNOWN_ACTION"


class InvalidParameterError(ValidationError):
    code = "INVALID_PARAMETER"


class UnknownIdError(ValidationError):
    code = "UNKNOWN_ID"


@dataclass(frozen=True, slots=True)
class ValidationContext:
    event_type: str
    conversation_id: str | None = None
    known_conversation_ids: frozenset[str] = field(default_factory=frozenset)
    max_message_chars: int = 500
    allowed_actions: frozenset[str] | None = None


class OutputValidator:
    def __init__(self, *, max_reasoning_chars: int = 2000) -> None:
        self._max_reasoning = max_reasoning_chars

    def validate(self, raw: str, ctx: ValidationContext) -> Decision:
        if raw is None or not str(raw).strip():
            raise EmptyResponseError("LLM 返回空内容")
        payload = self._parse(str(raw))
        self._validate_schema(payload)
        action = self._build_action(payload["action"], ctx)
        confidence = payload["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise SchemaValidationError("confidence 必须是数字")
        if not 0.0 <= float(confidence) <= 1.0:
            raise SchemaValidationError("confidence 必须落在 0..1")
        return Decision(
            reasoning=str(payload["reasoning"]).strip()[: self._max_reasoning],
            action=action,
            confidence=float(confidence),
        )

    def safe_validate(
        self, raw: str, ctx: ValidationContext
    ) -> tuple[Decision, ValidationError | None]:
        try:
            return self.validate(raw, ctx), None
        except ValidationError as exc:
            return Decision.do_nothing(reason=f"invalid LLM output: {exc.code}"), exc

    # ------------------------------------------------------------------ 解析
    def _parse(self, text: str) -> Mapping[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise JSONParseError(f"JSON 解析失败: {exc.msg}", detail=str(exc)) from exc
        if not isinstance(payload, Mapping):
            raise SchemaValidationError("顶层必须是 JSON 对象")
        return payload

    def _validate_schema(self, payload: Mapping[str, Any]) -> None:
        expected = {"reasoning", "action", "confidence"}
        missing = expected - set(payload)
        extra = set(payload) - expected
        if missing:
            raise SchemaValidationError(f"缺少字段: {', '.join(sorted(missing))}")
        if extra:
            raise SchemaValidationError(f"多余字段: {', '.join(sorted(extra))}")
        if not isinstance(payload["reasoning"], str):
            raise SchemaValidationError("reasoning 必须是字符串")
        action = payload["action"]
        if not isinstance(action, Mapping):
            raise SchemaValidationError("action 必须是对象")
        action_keys = set(action)
        if action_keys != {"type", "parameters"}:
            raise SchemaValidationError("action 只能包含 type 与 parameters")
        if not isinstance(action["type"], str):
            raise SchemaValidationError("action.type 必须是字符串")
        if not isinstance(action["parameters"], Mapping):
            raise SchemaValidationError("action.parameters 必须是对象")

    def _build_action(self, raw: Mapping[str, Any], ctx: ValidationContext) -> Action:
        try:
            action_type = ActionType.parse(raw["type"])
        except ValueError as exc:
            raise UnknownActionError(str(exc)) from exc
        if ctx.allowed_actions is not None and action_type.value not in ctx.allowed_actions:
            raise UnknownActionError(
                f"当前上下文不允许动作 {action_type.value}",
                detail="allowed=" + ",".join(sorted(ctx.allowed_actions)),
            )
        specs = ACTION_PARAMETER_SPECS[action_type]
        parameters = self._validate_parameters(action_type, specs, raw["parameters"], ctx)
        return Action(action_type, parameters)

    def _validate_parameters(
        self,
        action_type: ActionType,
        specs: tuple,
        raw: Mapping[str, Any],
        ctx: ValidationContext,
    ) -> dict[str, Any]:
        declared = {spec.name for spec in specs}
        extra = set(raw) - declared
        if extra:
            raise InvalidParameterError(
                f"{action_type.value} 出现未声明参数: {', '.join(sorted(extra))}"
            )
        cleaned: dict[str, Any] = {}
        for spec in specs:
            if spec.name not in raw:
                if spec.required:
                    raise InvalidParameterError(f"{action_type.value} 缺少参数 {spec.name}")
                continue
            value = raw[spec.name]
            cleaned[spec.name] = self._check_value(spec, value)
        if action_type is ActionType.SEND_MESSAGE:
            content = str(cleaned["content"]).strip()
            if not content:
                raise InvalidParameterError("SEND_MESSAGE content 不能为空")
            if len(content) > ctx.max_message_chars:
                raise InvalidParameterError(
                    f"SEND_MESSAGE content 超过 {ctx.max_message_chars} 字"
                )
            cleaned["content"] = content
            conversation_id = str(cleaned["conversation_id"])
            if conversation_id not in ctx.known_conversation_ids:
                raise UnknownIdError(
                    f"未知会话: {conversation_id}",
                    detail="known=" + ",".join(sorted(ctx.known_conversation_ids)),
                )
        return cleaned

    def _check_value(self, spec, value: Any) -> Any:
        if spec.kind is ParameterKind.STRING:
            if not isinstance(value, str):
                raise InvalidParameterError(f"{spec.name} 必须是字符串")
            if spec.max_length is not None and len(value) > spec.max_length:
                raise InvalidParameterError(f"{spec.name} 超过 {spec.max_length} 字符")
            return value
        if spec.kind is ParameterKind.INTEGER:
            if isinstance(value, bool) or not isinstance(value, int):
                raise InvalidParameterError(f"{spec.name} 必须是整数")
            if spec.min_value is not None and value < spec.min_value:
                raise InvalidParameterError(f"{spec.name} 不能小于 {spec.min_value}")
            if spec.max_value is not None and value > spec.max_value:
                raise InvalidParameterError(f"{spec.name} 不能大于 {spec.max_value}")
            return value
        if spec.kind is ParameterKind.BOOLEAN:
            if not isinstance(value, bool):
                raise InvalidParameterError(f"{spec.name} 必须是布尔值")
            return value
        if spec.kind is ParameterKind.ARRAY:
            if not isinstance(value, (list, tuple)):
                raise InvalidParameterError(f"{spec.name} 必须是数组")
            return list(value)
        raise InvalidParameterError(f"{spec.name} 类型未知")


__all__ = [
    "EmptyResponseError",
    "InvalidParameterError",
    "JSONParseError",
    "OutputValidator",
    "SchemaValidationError",
    "UnknownActionError",
    "UnknownIdError",
    "ValidationContext",
    "ValidationError",
]