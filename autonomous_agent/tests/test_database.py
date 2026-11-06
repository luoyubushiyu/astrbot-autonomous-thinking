from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from autonomous_agent.core.state import AgentState, StateSnapshot, ThoughtState
from autonomous_agent.database.database import (
    Database,
    DatabaseError,
    DatabaseNotConnectedError,
    EventRepository,
    MessageRepository,
    MetricsRepository,
    StateRepository,
)
from autonomous_agent.database.migrations import LATEST_VERSION, initialize_database


class DatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = Database(str(Path(self._tmp.name) / "test.db"))
        await initialize_database(self.db)

    async def asyncTearDown(self) -> None:
        await self.db.close()

    # -- schema ------------------------------------------------------------ #
    async def test_migration_version(self) -> None:
        self.assertEqual(await self.db.user_version(), LATEST_VERSION)

    async def test_idempotent_migrations(self) -> None:
        from autonomous_agent.database.migrations import apply_migrations

        self.assertEqual(await apply_migrations(self.db), [])

    async def test_future_schema_version_refused(self) -> None:
        path = Path(self._tmp.name) / "future.db"
        db = Database(str(path))
        await db.connect()
        await db.execute(f"PRAGMA user_version = {LATEST_VERSION + 1}")
        await db.close()

        db = Database(str(path))
        with self.assertRaises(DatabaseError):
            await initialize_database(db)
        await db.close()

    async def test_tables_exist(self) -> None:
        expected = {"agent_state", "messages", "events", "metrics"}
        rows = await self.db.fetch_all(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
        names = {row["name"] for row in rows}
        self.assertTrue(expected.issubset(names))

    # -- 状态 -------------------------------------------------------------- #
    async def test_state_save_and_load(self) -> None:
        repo = StateRepository(self.db)
        self.assertIsNone(await repo.load())

        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        snapshot = StateSnapshot(
            state=AgentState.WAITING,
            entered_at=now,
            updated_at=now,
            previous_state=AgentState.THINKING,
            thought_state=ThoughtState.INACTIVE,
            conversation_id="c1",
            metadata={"wait_until": "2026-01-01T00:10:00+00:00"},
        )
        await repo.save(snapshot)
        loaded = await repo.load()
        assert loaded is not None
        self.assertIs(loaded.state, AgentState.WAITING)
        self.assertIs(loaded.previous_state, AgentState.THINKING)
        self.assertEqual(loaded.conversation_id, "c1")
        self.assertEqual(loaded.metadata["wait_until"], "2026-01-01T00:10:00+00:00")

    async def test_state_upsert_single_row(self) -> None:
        repo = StateRepository(self.db)
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for state in (AgentState.IDLE, AgentState.THINKING, AgentState.SAFE_IDLE):
            await repo.save(
                StateSnapshot(state=state, entered_at=now, updated_at=now)
            )
        count = await self.db.scalar("SELECT COUNT(*) FROM agent_state")
        self.assertEqual(count, 1)

    # -- 消息 -------------------------------------------------------------- #
    async def test_message_insert_and_dedupe(self) -> None:
        repo = MessageRepository(self.db)
        first = await repo.add(
            conversation_id="c1", message_id="m1", role="user", content="你好"
        )
        second = await repo.add(
            conversation_id="c1", message_id="m1", role="user", content="你好"
        )
        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(await repo.count(), 1)

    async def test_message_recent_order(self) -> None:
        repo = MessageRepository(self.db)
        for i in range(3):
            await repo.add(
                conversation_id="c1",
                message_id=f"m{i}",
                role="user",
                content=f"第{i}条",
            )
        recent = await repo.recent("c1", 2)
        self.assertEqual([m.content for m in recent], ["第1条", "第2条"])
        self.assertEqual(await repo.conversation_ids(), {"c1"})

    async def test_invalid_role(self) -> None:
        repo = MessageRepository(self.db)
        with self.assertRaises(ValueError):
            await repo.add(
                conversation_id="c1", message_id="m1", role="hacker", content="x"
            )

    # -- 事件 / 指标 -------------------------------------------------------- #
    async def test_event_record_and_mark(self) -> None:
        repo = EventRepository(self.db)
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        await repo.record(
            event_id="e1",
            event_type="BOT_START",
            source="system",
            payload={"reason": "test"},
            created_at=now,
        )
        await repo.mark("e1", "PROCESSED")
        events = await repo.recent(1)
        self.assertEqual(events[0].event_id, "e1")
        self.assertEqual(events[0].status, "PROCESSED")
        self.assertEqual(await repo.count("BOT_START"), 1)

    async def test_metrics(self) -> None:
        repo = MetricsRepository(self.db)
        await repo.record("test.counter", 2, {"k": "v"})
        await repo.record("test.counter", 3)
        self.assertEqual(await repo.count("test.counter"), 2)
        self.assertEqual(await repo.total("test.counter"), 5.0)

    # -- 事务 -------------------------------------------------------------- #
    async def test_transaction_commit(self) -> None:
        async with self.db.transaction():
            await self.db.execute(
                "INSERT INTO metrics (name, value, tags, recorded_at) "
                "VALUES ('tx', 1, '{}', '2026-01-01T00:00:00+00:00')"
            )
        self.assertEqual(await self.db.scalar("SELECT COUNT(*) FROM metrics"), 1)

    async def test_transaction_rollback(self) -> None:
        with self.assertRaises(RuntimeError):
            async with self.db.transaction():
                await self.db.execute(
                    "INSERT INTO metrics (name, value, tags, recorded_at) "
                    "VALUES ('tx', 1, '{}', '2026-01-01T00:00:00+00:00')"
                )
                raise RuntimeError("boom")
        self.assertEqual(await self.db.scalar("SELECT COUNT(*) FROM metrics"), 0)

    async def test_nested_transaction_reuses_outer(self) -> None:
        async with self.db.transaction():
            async with self.db.transaction():
                await self.db.execute(
                    "INSERT INTO metrics (name, value, tags, recorded_at) "
                    "VALUES ('nested', 1, '{}', '2026-01-01T00:00:00+00:00')"
                )
        self.assertEqual(await self.db.scalar("SELECT COUNT(*) FROM metrics"), 1)

    async def test_nested_rollback_rolls_back_outer(self) -> None:
        with self.assertRaises(RuntimeError):
            async with self.db.transaction():
                await self.db.execute(
                    "INSERT INTO metrics (name, value, tags, recorded_at) "
                    "VALUES ('nested', 1, '{}', '2026-01-01T00:00:00+00:00')"
                )
                async with self.db.transaction():
                    raise RuntimeError("inner")
        self.assertEqual(await self.db.scalar("SELECT COUNT(*) FROM metrics"), 0)

    async def test_cross_task_access_inside_transaction_is_refused(self) -> None:
        async with self.db.transaction():
            with self.assertRaises(DatabaseError):
                await asyncio.create_task(self.db.scalar("SELECT 1"))
            await self.db.execute(
                "INSERT INTO metrics (name, value, tags, recorded_at) "
                "VALUES ('after', 1, '{}', '2026-01-01T00:00:00+00:00')"
            )
        self.assertEqual(await self.db.scalar("SELECT COUNT(*) FROM metrics"), 1)

    async def test_transaction_rollback_covers_state_and_messages(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with self.assertRaises(RuntimeError):
            async with self.db.transaction():
                await StateRepository(self.db).save(
                    StateSnapshot(state=AgentState.WAITING, entered_at=now, updated_at=now)
                )
                await MessageRepository(self.db).add(
                    conversation_id="c1", message_id="m1", role="user", content="hi"
                )
                raise RuntimeError("boom")
        self.assertIsNone(await StateRepository(self.db).load())
        self.assertEqual(await MessageRepository(self.db).count(), 0)

    # -- 连接与恢复 --------------------------------------------------------- #
    async def test_close_rejects_queries_and_reopen_keeps_data(self) -> None:
        await self.db.execute(
            "INSERT INTO metrics (name, value, tags, recorded_at) "
            "VALUES ('persist', 1, '{}', '2026-01-01T00:00:00+00:00')"
        )
        path = self.db.path
        await self.db.close()
        with self.assertRaises(DatabaseNotConnectedError):
            await self.db.scalar("SELECT 1")
        await self.db.connect()
        self.assertEqual(await self.db.scalar("SELECT COUNT(*) FROM metrics"), 1)
        self.assertEqual(self.db.path, path)

    async def test_concurrent_connect_is_harmless(self) -> None:
        path = Path(self._tmp.name) / "concurrent.db"
        db = Database(str(path))
        await asyncio.gather(db.connect(), db.connect(), db.connect())
        self.assertTrue(db.connected)
        self.assertEqual(await db.scalar("SELECT 1"), 1)
        await db.close()
        await db.close()  # 重复关闭不报错

    async def test_failed_migration_is_atomic(self) -> None:
        import autonomous_agent.database.migrations as migrations_mod

        path = Path(self._tmp.name) / "broken.db"
        db = Database(str(path))
        bad = migrations_mod.Migration(
            version=migrations_mod.LATEST_VERSION + 1,
            name="broken",
            statements=(
                "CREATE TABLE should_rollback (id INTEGER)",
                "INSERT INTO table_that_does_not_exist VALUES (1)",
            ),
        )
        migrations_mod.MIGRATIONS = migrations_mod.MIGRATIONS + (bad,)
        try:
            with self.assertRaises(DatabaseError):
                await initialize_database(db)
            self.assertEqual(await db.user_version(), LATEST_VERSION)
            tables = await db.fetch_all(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
            names = {row["name"] for row in tables}
            self.assertNotIn("should_rollback", names)
            self.assertIn("agent_state", names)
        finally:
            migrations_mod.MIGRATIONS = migrations_mod.MIGRATIONS[:-1]
            await db.close()

    async def test_restart_restores_state(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        await StateRepository(self.db).save(
            StateSnapshot(
                state=AgentState.SILENT,
                entered_at=now,
                updated_at=now,
                conversation_id="c9",
                metadata={"silent_until": "2026-01-01T01:00:00+00:00"},
            )
        )
        await self.db.close()
        await self.db.connect()
        loaded = await StateRepository(self.db).load()
        assert loaded is not None
        self.assertIs(loaded.state, AgentState.SILENT)
        self.assertEqual(loaded.conversation_id, "c9")

    async def test_invalid_event_status_rejected(self) -> None:
        repo = EventRepository(self.db)
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        await repo.record(
            event_id="e1", event_type="BOT_START", source="system", payload={}, created_at=now
        )
        with self.assertRaises(DatabaseError):
            await repo.mark("e1", "NOT_A_STATUS")


if __name__ == "__main__":
    unittest.main()