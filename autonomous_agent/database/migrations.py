"""数据库迁移。

用 ``PRAGMA user_version`` 记录已应用的版本，启动时把更大版本号的迁移按序
执行。每条迁移在自己的事务里，失败不会留下半张表。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..monitoring import get_logger
from .database import Database, DatabaseError

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    statements: tuple[str, ...]


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=1,
        name="bootstrap_schema",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS agent_state (
                id              INTEGER PRIMARY KEY CHECK (id = 1),
                state           TEXT    NOT NULL,
                previous_state  TEXT,
                thought_state   TEXT    NOT NULL DEFAULT 'INACTIVE',
                conversation_id TEXT,
                entered_at      TEXT    NOT NULL,
                updated_at      TEXT    NOT NULL,
                metadata        TEXT    NOT NULL DEFAULT '{}'
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT    NOT NULL,
                message_id      TEXT    NOT NULL UNIQUE,
                user_id         TEXT,
                role            TEXT    NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                content         TEXT    NOT NULL,
                created_at      TEXT    NOT NULL,
                metadata        TEXT    NOT NULL DEFAULT '{}'
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages (conversation_id, id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_messages_created
                ON messages (created_at)
            """,
            """
            CREATE TABLE IF NOT EXISTS events (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id     TEXT    NOT NULL UNIQUE,
                event_type   TEXT    NOT NULL,
                source       TEXT    NOT NULL,
                payload      TEXT    NOT NULL DEFAULT '{}',
                created_at   TEXT    NOT NULL,
                processed_at TEXT,
                status       TEXT    NOT NULL DEFAULT 'PENDING'
                    CHECK (status IN ('PENDING', 'PROCESSED', 'FAILED', 'SKIPPED'))
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_events_type_created
                ON events (event_type, created_at)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_events_status
                ON events (status)
            """,
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                value       REAL    NOT NULL DEFAULT 1.0,
                tags        TEXT    NOT NULL DEFAULT '{}',
                recorded_at TEXT    NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_metrics_name_recorded
                ON metrics (name, recorded_at)
            """,
        ),
    ),
    Migration(
        version=2,
        name="thought_workspace_and_memory",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS thoughts (
                id                 TEXT PRIMARY KEY,
                type               TEXT    NOT NULL,
                title              TEXT    NOT NULL,
                content            TEXT    NOT NULL,
                source             TEXT    NOT NULL,
                status             TEXT    NOT NULL,
                importance         REAL    NOT NULL DEFAULT 0.5,
                curiosity          REAL    NOT NULL DEFAULT 0.5,
                relevance          REAL    NOT NULL DEFAULT 0.5,
                confidence         REAL    NOT NULL DEFAULT 0.5,
                revisit_count      INTEGER NOT NULL DEFAULT 0,
                reactivation_count INTEGER NOT NULL DEFAULT 0,
                related_memories   TEXT    NOT NULL DEFAULT '[]',
                related_topics     TEXT    NOT NULL DEFAULT '[]',
                parent_thought_id  TEXT,
                next_action        TEXT,
                insight            TEXT,
                uncertainty        TEXT,
                last_thought_at    TEXT,
                next_check_at      TEXT,
                created_at         TEXT    NOT NULL,
                updated_at         TEXT    NOT NULL,
                metadata           TEXT    NOT NULL DEFAULT '{}'
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_thoughts_status
                ON thoughts (status, updated_at)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_thoughts_parent
                ON thoughts (parent_thought_id)
            """,
            """
            CREATE TABLE IF NOT EXISTS thought_relations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                from_id    TEXT    NOT NULL,
                to_id      TEXT    NOT NULL,
                kind       TEXT    NOT NULL,
                created_at TEXT    NOT NULL,
                UNIQUE (from_id, to_id, kind)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_thought_relations_to
                ON thought_relations (to_id, kind)
            """,
            """
            CREATE TABLE IF NOT EXISTS memories (
                id              TEXT PRIMARY KEY,
                type            TEXT    NOT NULL,
                content         TEXT    NOT NULL,
                confidence      REAL    NOT NULL DEFAULT 0.5,
                source          TEXT    NOT NULL DEFAULT 'system',
                importance      REAL    NOT NULL DEFAULT 0.5,
                created_at      TEXT    NOT NULL,
                updated_at      TEXT    NOT NULL,
                last_verified_at TEXT,
                expires_at      TEXT,
                metadata        TEXT    NOT NULL DEFAULT '{}'
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_memories_type
                ON memories (type, updated_at)
            """,
            """
            CREATE TABLE IF NOT EXISTS topics (
                id         TEXT PRIMARY KEY,
                name       TEXT    NOT NULL UNIQUE,
                keywords   TEXT    NOT NULL DEFAULT '[]',
                importance REAL    NOT NULL DEFAULT 0.5,
                created_at TEXT    NOT NULL,
                updated_at TEXT    NOT NULL,
                metadata   TEXT    NOT NULL DEFAULT '{}'
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS contact_attempts (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                thought_id      TEXT,
                conversation_id TEXT    NOT NULL,
                content         TEXT    NOT NULL,
                urgency         REAL    NOT NULL DEFAULT 0.0,
                importance      REAL    NOT NULL DEFAULT 0.5,
                status          TEXT    NOT NULL
                    CHECK (status IN ('SENT', 'DENIED', 'DELAYED')),
                reason          TEXT,
                created_at      TEXT    NOT NULL,
                replied_at      TEXT
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_contact_attempts_created
                ON contact_attempts (created_at)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_contact_attempts_conversation
                ON contact_attempts (conversation_id, created_at)
            """,
            """
            CREATE TABLE IF NOT EXISTS diary (
                day        TEXT PRIMARY KEY,
                content    TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
        ),
    ),
    Migration(
        version=3,
        name="thought_child_ids",
        statements=(
            # Thought Tree 的子节点列表（JSON 数组）。父节点 id 已在 v2。
            """
            ALTER TABLE thoughts
                ADD COLUMN child_thought_ids TEXT NOT NULL DEFAULT '[]'
            """,
        ),
    ),
    Migration(
        version=4,
        name="memory_lifecycle_and_relations",
        statements=(
            "ALTER TABLE memories ADD COLUMN status TEXT NOT NULL DEFAULT 'ACTIVE'",
            "ALTER TABLE memories ADD COLUMN relevance REAL NOT NULL DEFAULT 0.5",
            "ALTER TABLE memories ADD COLUMN frequency INTEGER NOT NULL DEFAULT 1",
            "ALTER TABLE memories ADD COLUMN usage_count INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE memories ADD COLUMN future_value REAL NOT NULL DEFAULT 0.5",
            "ALTER TABLE memories ADD COLUMN relationship_value REAL NOT NULL DEFAULT 0.3",
            "ALTER TABLE memories ADD COLUMN last_used_at TEXT",
            "ALTER TABLE memories ADD COLUMN source_type TEXT NOT NULL DEFAULT 'SYSTEM_OBSERVATION'",
            "ALTER TABLE memories ADD COLUMN source_id TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE memories ADD COLUMN keywords TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE memories ADD COLUMN reactivation_count INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE memories ADD COLUMN last_reactivated_at TEXT",
            "ALTER TABLE memories ADD COLUMN conflict INTEGER NOT NULL DEFAULT 0",
            """
            CREATE INDEX IF NOT EXISTS idx_memories_status
                ON memories (status, updated_at)
            """,
            """
            CREATE TABLE IF NOT EXISTS memory_relations (
                from_id    TEXT NOT NULL,
                to_id      TEXT NOT NULL,
                kind       TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (from_id, to_id, kind)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_memory_relations_to
                ON memory_relations (to_id)
            """,
        ),
    ),
    Migration(
        version=5,
        name="topic_lifecycle",
        statements=(
            "ALTER TABLE topics ADD COLUMN status TEXT NOT NULL DEFAULT 'NEW'",
            "ALTER TABLE topics ADD COLUMN last_activity_at TEXT",
            "ALTER TABLE topics ADD COLUMN thought_count INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE topics ADD COLUMN memory_count INTEGER NOT NULL DEFAULT 0",
            """
            CREATE INDEX IF NOT EXISTS idx_topics_status
                ON topics (status, last_activity_at)
            """,
        ),
    ),
    Migration(
        version=6,
        name="contact_intent_tracking",
        statements=(
            "ALTER TABLE contact_attempts ADD COLUMN intent_id TEXT",
            "ALTER TABLE contact_attempts ADD COLUMN topic TEXT",
            "ALTER TABLE contact_attempts ADD COLUMN message_hash TEXT",
            "ALTER TABLE contact_attempts ADD COLUMN reply_received INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE contact_attempts ADD COLUMN response_time_seconds REAL",
            "ALTER TABLE contact_attempts ADD COLUMN blocked_reason TEXT",
            """
            CREATE INDEX IF NOT EXISTS idx_contact_intent
                ON contact_attempts (intent_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_contact_topic
                ON contact_attempts (topic, created_at)
            """,
        ),
    ),
    )

LATEST_VERSION: int = max(m.version for m in MIGRATIONS)


async def current_version(db: Database) -> int:
    return await db.user_version()


async def apply_migrations(db: Database) -> list[int]:
    """应用所有未执行的迁移，返回本次实际执行的版本号列表。"""
    if not db.connected:
        raise RuntimeError("apply_migrations 需要先 connect 数据库")

    version = await current_version(db)
    if version > LATEST_VERSION:
        raise DatabaseError(
            f"数据库 schema 版本 v{version} 高于本程序支持的 v{LATEST_VERSION}；"
            "请升级程序，或换一个数据库文件"
        )

    applied: list[int] = []
    for migration in MIGRATIONS:
        if migration.version <= version:
            continue
        async with db.transaction():
            # 进事务后再读一次版本：另一个进程可能刚做完同一条迁移。
            latest = await current_version(db)
            if latest >= migration.version:
                version = latest
                logger.info("迁移 v%d 已应用，跳过", migration.version)
                continue
            for statement in migration.statements:
                await db.execute(statement)
            # PRAGMA 不支持参数绑定，版本号是内部整数，安全。
            await db.execute(f"PRAGMA user_version = {int(migration.version)}")
        applied.append(migration.version)
        logger.info(
            "数据库迁移完成: v%d %s", migration.version, migration.name
        )
        version = migration.version

    if not applied:
        logger.debug("数据库已是最新版本 v%d", version)
    return applied


async def initialize_database(db: Database) -> Database:
    """连接 + 迁移，一步到位。"""
    await db.connect()
    await apply_migrations(db)
    return db


__all__ = [
    "LATEST_VERSION",
    "MIGRATIONS",
    "Migration",
    "apply_migrations",
    "current_version",
    "initialize_database",
]