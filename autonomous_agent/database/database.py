"""SQLite 持久化层。

只暴露四个仓储：状态、消息、事件、指标。业务层不直接写 SQL。

要点：
* 单连接 + 串行锁，避免 SQLite 多写冲突；
* 事务用 ``BEGIN IMMEDIATE``，异常自动回滚；事务期间不允许其它任务碰同一连接；
* ``PRAGMA user_version`` 驱动迁移；
* 所有 sqlite3 异常收敛成 ``DatabaseError``。
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Mapping, Sequence

from ..core.state import AgentState, StateSnapshot, ThoughtState
from ..monitoring import get_logger

logger = get_logger(__name__)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    raise ValueError(f"无法解析时间: {value!r}")


class DatabaseError(RuntimeError):
    """数据库操作失败。"""


class DatabaseNotConnectedError(DatabaseError):
    """未连接就执行操作。"""


# --------------------------------------------------------------------------- #
# 数据记录
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class MessageRecord:
    id: int
    conversation_id: str
    message_id: str
    role: str
    content: str
    created_at: datetime
    user_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EventRecord:
    id: int
    event_id: str
    event_type: str
    source: str
    payload: Mapping[str, Any]
    created_at: datetime
    status: str
    processed_at: datetime | None = None


# --------------------------------------------------------------------------- #
# 连接
# --------------------------------------------------------------------------- #
class Database:
    """异步门面，内部用 ``asyncio.to_thread`` 调同步 sqlite3。"""

    def __init__(self, path: str | Path, *, busy_timeout_ms: int = 5000) -> None:
        self._path = Path(path) if str(path) != ":memory:" else Path(":memory:")
        self._busy_timeout_ms = busy_timeout_ms
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        self._connect_lock = asyncio.Lock()
        self._tx_ctx: Any = None

    # -- 生命周期 ---------------------------------------------------------- #
    @property
    def path(self) -> Path:
        return self._path

    @property
    def connected(self) -> bool:
        return self._conn is not None

    async def connect(self) -> None:
        if self._conn is not None:
            return
        async with self._connect_lock:
            if self._conn is not None:
                return
            await asyncio.to_thread(self._connect_sync)

    def _connect_sync(self) -> None:
        conn: sqlite3.Connection | None = None
        try:
            if str(self._path) != ":memory:":
                self._path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                str(self._path),
                timeout=self._busy_timeout_ms / 1000,
                check_same_thread=False,
                isolation_level=None,  # 手动管理事务
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA busy_timeout = %d" % int(self._busy_timeout_ms))
            if str(self._path) != ":memory:":
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
            self._conn = conn
            logger.info("SQLite 已连接: %s", self._path)
        except sqlite3.Error as exc:
            if conn is not None:  # 连接已打开但 PRAGMA 失败，别把它泄漏出去
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
            raise DatabaseError(f"打开数据库失败 {self._path}: {exc}") from exc

    async def close(self) -> None:
        async with self._lock, self._connect_lock:
            conn, self._conn = self._conn, None
            if conn is None:
                return
            try:
                await asyncio.to_thread(conn.close)
                logger.info("SQLite 已关闭: %s", self._path)
            except sqlite3.Error as exc:  # pragma: no cover - 关闭失败很少见
                raise DatabaseError(f"关闭数据库失败: {exc}") from exc

    # -- 基础操作 ---------------------------------------------------------- #
    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise DatabaseNotConnectedError("数据库尚未连接")
        return self._conn

    def _tx_var(self) -> Any:
        if self._tx_ctx is None:
            from contextvars import ContextVar

            # 值是 (事务持有者任务, 嵌套深度)。用任务而不只是深度，避免子任务
            # 继承 ContextVar 后误以为自己也在事务里、绕过连接锁并发写。
            self._tx_ctx = ContextVar(f"db_tx_{id(self)}", default=(None, 0))
        return self._tx_ctx

    @staticmethod
    def _tx_conflict() -> DatabaseError:
        return DatabaseError(
            "事务进行中，禁止在其它任务里访问同一个数据库连接；"
            "请等事务结束后再顺序执行，或在事务内部直接 await"
        )

    async def _run(self, fn: Any, *args: Any) -> Any:
        owner, depth = self._tx_var().get()
        if depth > 0:
            if owner is asyncio.current_task():
                return await asyncio.to_thread(fn, *args)
            raise self._tx_conflict()
        async with self._lock:
            return await asyncio.to_thread(fn, *args)

    async def execute(self, sql: str, params: Sequence[Any] | Mapping[str, Any] = ()) -> int:
        return await self._run(self._execute_sync, sql, params)

    def _execute_sync(self, sql: str, params: Any) -> int:
        conn = self._require_conn()
        try:
            cur = conn.execute(sql, params)
            last_id = cur.lastrowid or 0
            rowcount = cur.rowcount
            cur.close()
        except sqlite3.Error as exc:
            raise DatabaseError(f"SQL 执行失败: {exc}\nSQL: {sql[:300]}") from exc
        if sql.lstrip().upper().startswith("INSERT"):
            return int(last_id)
        return int(rowcount)

    async def execute_many(
        self, sql: str, seq: Iterable[Sequence[Any] | Mapping[str, Any]]
    ) -> int:
        return await self._run(self._execute_many_sync, sql, list(seq))

    def _execute_many_sync(self, sql: str, seq: list[Any]) -> int:
        conn = self._require_conn()
        try:
            cur = conn.executemany(sql, seq)
            count = cur.rowcount
            cur.close()
        except sqlite3.Error as exc:
            raise DatabaseError(f"批量 SQL 执行失败: {exc}\nSQL: {sql[:300]}") from exc
        return int(count)

    async def fetch_one(
        self, sql: str, params: Sequence[Any] | Mapping[str, Any] = ()
    ) -> dict[str, Any] | None:
        return await self._run(self._fetch_one_sync, sql, params)

    def _fetch_one_sync(self, sql: str, params: Any) -> dict[str, Any] | None:
        conn = self._require_conn()
        try:
            cur = conn.execute(sql, params)
            row = cur.fetchone()
            cur.close()
        except sqlite3.Error as exc:
            raise DatabaseError(f"查询失败: {exc}\nSQL: {sql[:300]}") from exc
        return dict(row) if row is not None else None

    async def fetch_all(
        self, sql: str, params: Sequence[Any] | Mapping[str, Any] = ()
    ) -> list[dict[str, Any]]:
        return await self._run(self._fetch_all_sync, sql, params)

    def _fetch_all_sync(self, sql: str, params: Any) -> list[dict[str, Any]]:
        conn = self._require_conn()
        try:
            cur = conn.execute(sql, params)
            rows = cur.fetchall()
            cur.close()
        except sqlite3.Error as exc:
            raise DatabaseError(f"查询失败: {exc}\nSQL: {sql[:300]}") from exc
        return [dict(row) for row in rows]

    async def scalar(self, sql: str, params: Sequence[Any] | Mapping[str, Any] = ()) -> Any:
        row = await self.fetch_one(sql, params)
        if row is None:
            return None
        return next(iter(row.values()), None)

    async def user_version(self) -> int:
        value = await self.scalar("PRAGMA user_version")
        return int(value or 0)

    async def quick_check(self) -> str:
        value = await self.scalar("PRAGMA quick_check")
        return str(value or "unknown")

    # -- 事务 -------------------------------------------------------------- #
    @asynccontextmanager
    async def transaction(self) -> AsyncIterator["Database"]:
        var = self._tx_var()
        owner, depth = var.get()
        if depth > 0:
            if owner is not asyncio.current_task():
                raise self._tx_conflict()
            # 嵌套事务直接复用外层（SQLite 不支持真正的嵌套）。
            token = var.set((owner, depth + 1))
            try:
                yield self
            finally:
                var.reset(token)
            return

        async with self._lock:
            token = var.set((asyncio.current_task(), 1))
            try:
                await asyncio.to_thread(self._begin_sync)
                yield self
                await asyncio.to_thread(self._commit_sync)
            except BaseException:
                await asyncio.to_thread(self._rollback_sync)
                raise
            finally:
                var.reset(token)

    def _begin_sync(self) -> None:
        conn = self._require_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.Error as exc:
            raise DatabaseError(f"开启事务失败: {exc}") from exc

    def _commit_sync(self) -> None:
        conn = self._require_conn()
        try:
            conn.execute("COMMIT")
        except sqlite3.Error as exc:
            raise DatabaseError(f"提交事务失败: {exc}") from exc

    def _rollback_sync(self) -> None:
        conn = self._conn
        if conn is None:
            return
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            logger.warning("回滚事务失败（可能事务已结束）")


# --------------------------------------------------------------------------- #
# 仓储：Agent 状态
# --------------------------------------------------------------------------- #
class StateRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def load(self) -> StateSnapshot | None:
        row = await self._db.fetch_one("SELECT * FROM agent_state WHERE id = 1")
        if row is None:
            return None
        try:
            metadata = json.loads(row["metadata"] or "{}")
            if not isinstance(metadata, dict):
                metadata = {}
        except json.JSONDecodeError:
            logger.warning("agent_state.metadata 损坏，已按空对象处理")
            metadata = {}
        try:
            return StateSnapshot(
                state=AgentState.parse(row["state"]),
                previous_state=(
                    AgentState.parse(row["previous_state"])
                    if row["previous_state"]
                    else None
                ),
                thought_state=ThoughtState(row["thought_state"]),
                entered_at=_parse_dt(row["entered_at"]),
                updated_at=_parse_dt(row["updated_at"]),
                conversation_id=row["conversation_id"],
                metadata=metadata,
            )
        except (ValueError, TypeError) as exc:
            logger.error("agent_state 数据损坏，忽略并使用默认状态: %s", exc)
            return None

    async def save(self, snapshot: StateSnapshot) -> None:
        row = snapshot.to_row()
        await self._db.execute(
            """
            INSERT INTO agent_state
                (id, state, previous_state, thought_state, conversation_id,
                 entered_at, updated_at, metadata)
            VALUES
                (1, :state, :previous_state, :thought_state, :conversation_id,
                 :entered_at, :updated_at, :metadata)
            ON CONFLICT(id) DO UPDATE SET
                state = excluded.state,
                previous_state = excluded.previous_state,
                thought_state = excluded.thought_state,
                conversation_id = excluded.conversation_id,
                entered_at = excluded.entered_at,
                updated_at = excluded.updated_at,
                metadata = excluded.metadata
            """,
            row,
        )


# --------------------------------------------------------------------------- #
# 仓储：消息
# --------------------------------------------------------------------------- #
class MessageRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def add(
        self,
        *,
        conversation_id: str,
        message_id: str,
        role: str,
        content: str,
        user_id: str | None = None,
        created_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        """写入消息；message_id 已存在时静默跳过，返回是否真的插入。"""
        if role not in ("user", "assistant", "system"):
            raise ValueError(f"非法 role: {role!r}")
        sql = """
            INSERT OR IGNORE INTO messages
                (conversation_id, message_id, user_id, role, content, created_at, metadata)
            VALUES
                (:conversation_id, :message_id, :user_id, :role, :content, :created_at, :metadata)
        """
        async with self._db.transaction():
            await self._db.execute(
                sql,
                {
                    "conversation_id": conversation_id,
                    "message_id": message_id,
                    "user_id": user_id,
                    "role": role,
                    "content": content,
                    "created_at": _iso(created_at or utcnow()),
                    "metadata": json.dumps(dict(metadata or {}), ensure_ascii=False),
                },
            )
            inserted = int(await self._db.scalar("SELECT changes()") or 0)
        return inserted > 0

    async def recent(self, conversation_id: str, limit: int = 10) -> list[MessageRecord]:
        if limit <= 0:
            return []
        rows = await self._db.fetch_all(
            """
            SELECT * FROM messages
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        )
        rows.reverse()
        return [self._to_record(row) for row in rows]

    async def conversation_ids(self) -> set[str]:
        rows = await self._db.fetch_all(
            "SELECT DISTINCT conversation_id FROM messages"
        )
        return {row["conversation_id"] for row in rows if row["conversation_id"]}

    async def count(self) -> int:
        return int(await self._db.scalar("SELECT COUNT(*) FROM messages") or 0)

    @staticmethod
    def _to_record(row: Mapping[str, Any]) -> MessageRecord:
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except json.JSONDecodeError:
            metadata = {}
        return MessageRecord(
            id=int(row["id"]),
            conversation_id=str(row["conversation_id"]),
            message_id=str(row["message_id"]),
            role=str(row["role"]),
            content=str(row["content"]),
            created_at=_parse_dt(row["created_at"]),
            user_id=row["user_id"],
            metadata=metadata if isinstance(metadata, dict) else {},
        )


# --------------------------------------------------------------------------- #
# 仓储：事件
# --------------------------------------------------------------------------- #
class EventRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def record(self, *, event_id: str, event_type: str, source: str, payload: Mapping[str, Any], created_at: datetime) -> None:
        await self._db.execute(
            """
            INSERT OR IGNORE INTO events
                (event_id, event_type, source, payload, created_at, status)
            VALUES (?, ?, ?, ?, ?, 'PENDING')
            """,
            (
                event_id,
                event_type,
                source,
                json.dumps(dict(payload), ensure_ascii=False, default=str),
                _iso(created_at),
            ),
        )

    async def mark(self, event_id: str, status: str, *, processed_at: datetime | None = None) -> None:
        await self._db.execute(
            "UPDATE events SET status = ?, processed_at = ? WHERE event_id = ?",
            (status, _iso(processed_at or utcnow()), event_id),
        )

    async def recent(self, limit: int = 20) -> list[EventRecord]:
        rows = await self._db.fetch_all(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [self._to_record(row) for row in rows]

    async def count(self, event_type: str | None = None) -> int:
        if event_type:
            return int(
                await self._db.scalar(
                    "SELECT COUNT(*) FROM events WHERE event_type = ?", (event_type,)
                )
                or 0
            )
        return int(await self._db.scalar("SELECT COUNT(*) FROM events") or 0)

    @staticmethod
    def _to_record(row: Mapping[str, Any]) -> EventRecord:
        try:
            payload = json.loads(row["payload"] or "{}")
        except json.JSONDecodeError:
            payload = {}
        return EventRecord(
            id=int(row["id"]),
            event_id=str(row["event_id"]),
            event_type=str(row["event_type"]),
            source=str(row["source"]),
            payload=payload if isinstance(payload, dict) else {},
            created_at=_parse_dt(row["created_at"]),
            status=str(row["status"]),
            processed_at=(
                _parse_dt(row["processed_at"]) if row["processed_at"] else None
            ),
        )


# --------------------------------------------------------------------------- #
# 仓储：指标
# --------------------------------------------------------------------------- #
class MetricsRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def record(
        self,
        name: str,
        value: float = 1.0,
        tags: Mapping[str, Any] | None = None,
        *,
        recorded_at: datetime | None = None,
    ) -> None:
        await self._db.execute(
            "INSERT INTO metrics (name, value, tags, recorded_at) VALUES (?, ?, ?, ?)",
            (
                name,
                float(value),
                json.dumps(dict(tags or {}), ensure_ascii=False, default=str),
                _iso(recorded_at or utcnow()),
            ),
        )

    async def count(self, name: str) -> int:
        return int(
            await self._db.scalar("SELECT COUNT(*) FROM metrics WHERE name = ?", (name,))
            or 0
        )

    async def total(self, name: str) -> float:
        value = await self._db.scalar(
            "SELECT COALESCE(SUM(value), 0) FROM metrics WHERE name = ?", (name,)
        )
        return float(value or 0.0)

    async def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        return await self._db.fetch_all(
            "SELECT * FROM metrics ORDER BY id DESC LIMIT ?", (limit,)
        )


def new_message_id(prefix: str = "msg") -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


__all__ = [
    "Database",
    "DatabaseError",
    "DatabaseNotConnectedError",
    "EventRecord",
    "EventRepository",
    "MessageRecord",
    "MessageRepository",
    "MetricsRepository",
    "StateRepository",
    "new_message_id",
]