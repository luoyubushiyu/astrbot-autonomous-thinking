"""QQ 接入层。"""

from __future__ import annotations

from .adapter import (
    GroupInfo,
    IncomingMessage,
    MockQQAdapter,
    OutgoingMessage,
    QQAdapter,
    QQAdapterError,
    QQSendError,
    SendResult,
    UserInfo,
    create_qq_adapter,
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