"""QQ 接入层。

Agent 核心只依赖这个抽象接口，不依赖任何具体 QQ 框架（NapCat / OneBot /
官方开放平台）。内置 ``MockQQAdapter``；接真实框架时实现 ``QQAdapter`` 再注入
``Agent(qq=...)`` 即可，核心代码不用动。
"""

from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from ..config import QQConfig
from ..monitoring import get_logger

logger = get_logger(__name__)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class IncomingMessage:
    conversation_id: str
    message_id: str
    user_id: str
    content: str
    received_at: datetime = field(default_factory=utcnow)
    is_group: bool = False
    group_id: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "IncomingMessage":
        received = payload.get("received_at")
        if isinstance(received, str):
            try:
                parsed = datetime.fromisoformat(received)
            except ValueError:
                parsed = utcnow()
        elif isinstance(received, datetime):
            parsed = received
        else:
            parsed = utcnow()
        return cls(
            conversation_id=str(payload["conversation_id"]),
            message_id=str(payload["message_id"]),
            user_id=str(payload["user_id"]),
            content=str(payload.get("content", "")),
            received_at=parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc),
            is_group=bool(payload.get("is_group", False)),
            group_id=payload.get("group_id"),
        )


@dataclass(frozen=True, slots=True)
class OutgoingMessage:
    conversation_id: str
    content: str
    reply_to: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SendResult:
    ok: bool
    message_id: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class UserInfo:
    user_id: str
    nickname: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GroupInfo:
    group_id: str
    name: str = ""
    member_count: int = 0
    raw: Mapping[str, Any] = field(default_factory=dict)


class QQAdapterError(RuntimeError):
    """QQ 适配器错误基类。"""


class QQSendError(QQAdapterError):
    """消息发送失败。"""


class QQAdapter(ABC):
    """QQ 侧能力抽象。"""

    name: str = "abstract"

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    @abstractmethod
    async def receive_message(self) -> IncomingMessage | None:
        """非阻塞地取一条新消息；没有则返回 None。"""

    @abstractmethod
    async def send_message(self, message: OutgoingMessage) -> SendResult:
        """发送消息。失败必须抛 ``QQAdapterError`` 子类或返回 ok=False。"""

    @abstractmethod
    async def get_user_info(self, user_id: str) -> UserInfo | None:
        """查询用户信息；查不到返回 None。"""

    @abstractmethod
    async def get_group_info(self, group_id: str) -> GroupInfo | None:
        """查询群信息；查不到返回 None。"""


class MockQQAdapter(QQAdapter):
    """内存 QQ 适配器，测试与离线演示使用。

    * ``push_incoming`` / ``push_text`` 往收件箱塞消息；
    * ``outbox`` 记录所有已发送消息；
    * ``send_failure`` 非空时发送会抛异常，用于测 SAFE_IDLE。
    """

    name = "mock"

    def __init__(self, *, self_id: str = "bot") -> None:
        self.self_id = self_id
        self._inbox: deque[IncomingMessage] = deque()
        self._lock = asyncio.Lock()
        self.outbox: list[OutgoingMessage] = []
        self.users: dict[str, UserInfo] = {}
        self.groups: dict[str, GroupInfo] = {}
        self.send_failure: Exception | None = None
        self.started = False

    # -- 生命周期 ---------------------------------------------------------- #
    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.started = False

    # -- 收 ---------------------------------------------------------------- #
    def push_incoming(self, message: IncomingMessage) -> None:
        self._inbox.append(message)

    def push_text(
        self,
        content: str,
        *,
        conversation_id: str = "mock-conversation",
        user_id: str = "mock-user",
        message_id: str | None = None,
        is_group: bool = False,
        group_id: str | None = None,
    ) -> IncomingMessage:
        message = IncomingMessage(
            conversation_id=conversation_id,
            message_id=message_id or f"in-{uuid.uuid4().hex}",
            user_id=user_id,
            content=content,
            is_group=is_group,
            group_id=group_id,
        )
        self.push_incoming(message)
        return message

    async def receive_message(self) -> IncomingMessage | None:
        async with self._lock:
            return self._inbox.popleft() if self._inbox else None

    # -- 发 ---------------------------------------------------------------- #
    async def send_message(self, message: OutgoingMessage) -> SendResult:
        if self.send_failure is not None:
            raise QQSendError(str(self.send_failure))
        if not message.content.strip():
            raise QQSendError("消息内容为空")
        self.outbox.append(message)
        logger.info(
            "Mock QQ 发送消息",
            extra={"conversation_id": message.conversation_id},
        )
        return SendResult(ok=True, message_id=f"out-{uuid.uuid4().hex}")

    # -- 资料 -------------------------------------------------------------- #
    async def get_user_info(self, user_id: str) -> UserInfo | None:
        return self.users.get(user_id)

    async def get_group_info(self, group_id: str) -> GroupInfo | None:
        return self.groups.get(group_id)

    # -- 测试辅助 ---------------------------------------------------------- #
    def clear(self) -> None:
        self._inbox.clear()
        self.outbox.clear()


def create_qq_adapter(config: QQConfig) -> QQAdapter:
    """按配置造适配器。

    目前只内置 mock；接真实框架时在这里加一个分支，或直接把实现了
    ``QQAdapter`` 的实例传进 ``Agent(qq=...)``，Agent 核心不需要改。
    """
    if config.adapter == "mock":
        return MockQQAdapter(self_id=config.self_id)
    raise QQAdapterError(
        f"没有内置该 QQ 适配器: {config.adapter}；"
        "请把实现了 QQAdapter 的实例直接传给 Agent(qq=...)"
    )


__all__ = [
    "GroupInfo",
    "IncomingMessage",
    "MockQQAdapter",
    "OutgoingMessage",
    "QQAdapter",
    "QQAdapterError",
    "QQSendError",
    "SendResult",
    "UserInfo",
    "create_qq_adapter",
]