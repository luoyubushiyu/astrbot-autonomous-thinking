"""事件层。

事件是 Agent 唯一的输入来源：QQ 消息、定时 tick、闹钟、启停信号都会先变成
``Event``，再由 Agent 统一消费。事件本身只做分发，不做推理。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EventType(str, Enum):
    USER_MESSAGE = "USER_MESSAGE"
    AUTONOMOUS_TICK = "AUTONOMOUS_TICK"
    BOT_START = "BOT_START"
    BOT_STOP = "BOT_STOP"
    ALARM_TRIGGERED = "ALARM_TRIGGERED"
    CONTACT_TIMEOUT = "CONTACT_TIMEOUT"


class EventError(ValueError):
    """事件错误基类。"""


class InvalidEventError(EventError):
    """事件结构不合法。"""


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _parse_iso(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str) or not value:
        raise InvalidEventError(f"无法解析时间: {value!r}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise InvalidEventError(f"无法解析时间: {value!r}") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class Event:
    type: EventType
    payload: Mapping[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    occurred_at: datetime = field(default_factory=utcnow)
    source: str = "system"

    def __post_init__(self) -> None:
        if not isinstance(self.type, EventType):
            object.__setattr__(self, "type", EventType(self.type))
        if not isinstance(self.payload, Mapping):
            raise InvalidEventError("event.payload 必须是 mapping")
        if self.occurred_at.tzinfo is None:
            object.__setattr__(
                self, "occurred_at", self.occurred_at.replace(tzinfo=timezone.utc)
            )
        _validate_payload(self.type, self.payload)

    # -- 序列化 ------------------------------------------------------------ #
    def to_row(self) -> dict[str, Any]:
        import json

        return {
            "event_id": self.id,
            "event_type": self.type.value,
            "source": self.source,
            "payload": json.dumps(dict(self.payload), ensure_ascii=False, default=str),
            "created_at": _iso(self.occurred_at),
        }

    def conversation_id(self) -> str | None:
        value = self.payload.get("conversation_id")
        return value if isinstance(value, str) and value else None

    # -- 工厂 -------------------------------------------------------------- #
    @classmethod
    def bot_start(cls) -> "Event":
        return cls(type=EventType.BOT_START, source="system")

    @classmethod
    def bot_stop(cls, reason: str = "normal") -> "Event":
        return cls(type=EventType.BOT_STOP, payload={"reason": reason}, source="system")

    @classmethod
    def user_message(
        cls,
        *,
        conversation_id: str,
        message_id: str,
        user_id: str,
        content: str,
        received_at: datetime | None = None,
        is_group: bool = False,
        group_id: str | None = None,
        source: str = "qq",
    ) -> "Event":
        return cls(
            type=EventType.USER_MESSAGE,
            occurred_at=received_at or utcnow(),
            source=source,
            payload={
                "conversation_id": conversation_id,
                "message_id": message_id,
                "user_id": user_id,
                "content": content,
                "is_group": bool(is_group),
                "group_id": group_id,
                "received_at": _iso(received_at or utcnow()),
            },
        )

    @classmethod
    def autonomous_tick(cls, tick: int, reason: str = "scheduled") -> "Event":
        return cls(
            type=EventType.AUTONOMOUS_TICK,
            payload={"tick": int(tick), "reason": reason},
            source="scheduler",
        )

    @classmethod
    def alarm_triggered(cls, alarm_id: str, reason: str = "", **extra: Any) -> "Event":
        payload = {"alarm_id": alarm_id, "reason": reason}
        payload.update(extra)
        return cls(type=EventType.ALARM_TRIGGERED, payload=payload, source="scheduler")

    @classmethod
    def contact_timeout(
        cls, conversation_id: str | None = None, waited_seconds: float = 0.0
    ) -> "Event":
        return cls(
            type=EventType.CONTACT_TIMEOUT,
            payload={
                "conversation_id": conversation_id,
                "waited_seconds": float(waited_seconds),
            },
            source="scheduler",
        )


def _require_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InvalidEventError(f"USER_MESSAGE 缺少字段 {key}")
    return value


def _validate_payload(event_type: EventType, payload: Mapping[str, Any]) -> None:
    """只校验真实使用到的字段，避免过度约束。"""
    if event_type is EventType.USER_MESSAGE:
        _require_str(payload, "conversation_id")
        _require_str(payload, "message_id")
        _require_str(payload, "user_id")
        if not isinstance(payload.get("content"), str):
            raise InvalidEventError("USER_MESSAGE 缺少字段 content")
        if "received_at" in payload:
            _parse_iso(payload["received_at"])
    elif event_type is EventType.AUTONOMOUS_TICK:
        tick = payload.get("tick", 0)
        if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
            raise InvalidEventError("AUTONOMOUS_TICK 的 tick 必须是非负整数")
    elif event_type is EventType.BOT_STOP:
        if not isinstance(payload.get("reason", ""), str):
            raise InvalidEventError("BOT_STOP 的 reason 必须是字符串")
    elif event_type is EventType.ALARM_TRIGGERED:
        _require_str(payload, "alarm_id")
    elif event_type is EventType.CONTACT_TIMEOUT:
        waited = payload.get("waited_seconds", 0.0)
        if isinstance(waited, bool) or not isinstance(waited, (int, float)):
            raise InvalidEventError("CONTACT_TIMEOUT 的 waited_seconds 必须是数字")
        if waited < 0:
            raise InvalidEventError("CONTACT_TIMEOUT 的 waited_seconds 不能为负")
        conversation_id = payload.get("conversation_id")
        if conversation_id is not None and not isinstance(conversation_id, str):
            raise InvalidEventError("CONTACT_TIMEOUT 的 conversation_id 必须是字符串")


def parse_event_payload(event_type: EventType, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """校验外部构造的 payload（供测试/适配器使用）。"""
    _validate_payload(event_type, payload)
    return payload