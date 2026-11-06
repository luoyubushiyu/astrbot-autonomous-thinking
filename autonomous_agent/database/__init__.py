"""数据库层。"""

from __future__ import annotations

from .database import (
    Database,
    DatabaseError,
    DatabaseNotConnectedError,
    EventRecord,
    EventRepository,
    MessageRecord,
    MessageRepository,
    MetricsRepository,
    StateRepository,
    new_message_id,
)
from .migrations import LATEST_VERSION, initialize_database, apply_migrations

__all__ = [
    "Database",
    "DatabaseError",
    "DatabaseNotConnectedError",
    "EventRecord",
    "EventRepository",
    "LATEST_VERSION",
    "MessageRecord",
    "MessageRepository",
    "MetricsRepository",
    "StateRepository",
    "apply_migrations",
    "initialize_database",
    "new_message_id",
]