"""主动联系闸门测试：每一条硬限制都要能独立拦住发送。"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from autonomous_agent.config import ContactConfig
from autonomous_agent.contact.manager import ContactManager
from autonomous_agent.database.database import Database
from autonomous_agent.database.migrations import initialize_database
from autonomous_agent.thought.models import Thought, ThoughtType

NOW = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)


def make_thought() -> Thought:
    return Thought.create(type=ThoughtType.QUESTION, title="想找你聊聊")


class ContactManagerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self._tmp.name) / "agent.db")
        await initialize_database(self.db)
        self.addAsyncCleanup(self._cleanup)

    async def _cleanup(self) -> None:
        await self.db.close()
        self._tmp.cleanup()

    def manager(self, **overrides) -> ContactManager:
        config = ContactConfig()
        config.minimum_interval_seconds = 0
        config.cooldown_seconds = 0
        config.consecutive_no_reply_limit = 10
        config.hourly_limit = 10
        config.daily_limit = 10
        config.night_hours = []
        for key, value in overrides.items():
            setattr(config, key, value)
        return ContactManager(self.db, config, clock=lambda: NOW)

    async def test_disabled_denies(self) -> None:
        manager = self.manager(enabled=False)
        decision = await manager.evaluate(make_thought(), "conv-1", "在吗", now=NOW)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.code, "CONTACT_DISABLED")

    async def test_night_time_denies(self) -> None:
        manager = self.manager(night_hours=["00:00-23:59"])
        decision = await manager.evaluate(make_thought(), "conv-1", "在吗", now=NOW)
        self.assertEqual(decision.code, "NIGHT_TIME")

    async def test_hourly_limit(self) -> None:
        manager = self.manager(hourly_limit=1)
        thought = make_thought()
        await manager.record(thought, "conv-1", "第一句", status="SENT", now=NOW - timedelta(minutes=30))
        decision = await manager.evaluate(thought, "conv-1", "第二句", now=NOW)
        self.assertEqual(decision.code, "HOURLY_LIMIT")

    async def test_consecutive_no_reply(self) -> None:
        manager = self.manager(consecutive_no_reply_limit=2)
        thought = make_thought()
        for index in range(2):
            await manager.record(
                thought,
                "conv-1",
                f"第 {index} 句",
                status="SENT",
                now=NOW - timedelta(minutes=30 - index),
            )
        decision = await manager.evaluate(thought, "conv-1", "又来了", now=NOW)
        self.assertEqual(decision.code, "CONSECUTIVE_NO_REPLY")

    async def test_reply_resets_no_reply_counter(self) -> None:
        manager = self.manager(consecutive_no_reply_limit=2)
        thought = make_thought()
        for index in range(2):
            await manager.record(
                thought, "conv-1", f"第 {index} 句", status="SENT", now=NOW - timedelta(minutes=30)
            )
        await manager.mark_replied("conv-1", now=NOW - timedelta(minutes=20))
        decision = await manager.evaluate(thought, "conv-1", "新的内容", now=NOW)
        self.assertTrue(decision.allowed)

    async def test_minimum_interval(self) -> None:
        manager = self.manager(minimum_interval_seconds=1800)
        thought = make_thought()
        await manager.record(thought, "conv-1", "刚说过", status="SENT", now=NOW - timedelta(seconds=60))
        decision = await manager.evaluate(thought, "conv-1", "不一样的内容", now=NOW)
        self.assertEqual(decision.code, "MIN_INTERVAL")

    async def test_duplicate_message(self) -> None:
        manager = self.manager(duplicate_threshold=0.7)
        thought = make_thought()
        await manager.record(
            thought, "conv-1", "今天天气真不错呀", status="SENT", now=NOW - timedelta(minutes=5)
        )
        decision = await manager.evaluate(thought, "conv-1", "今天天气真不错呀", now=NOW)
        self.assertEqual(decision.code, "DUPLICATE_MESSAGE")

    async def test_thought_repeat(self) -> None:
        manager = self.manager(max_attempts_per_thought=1)
        thought = make_thought()
        await manager.record(
            thought, "conv-1", "完全不同的第一句", status="SENT", now=NOW - timedelta(minutes=5)
        )
        decision = await manager.evaluate(thought, "conv-1", "完全不同的话", now=NOW)
        self.assertEqual(decision.code, "THOUGHT_REPEAT")

    async def test_clean_attempt_allowed(self) -> None:
        manager = self.manager()
        decision = await manager.evaluate(make_thought(), "conv-1", "想问你个事", now=NOW)
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.code, "CONTACT_ALLOWED")

    async def test_denied_attempts_still_count_for_cooldown(self) -> None:
        manager = self.manager(cooldown_seconds=600)
        thought = make_thought()
        await manager.record(
            thought, "conv-1", "被拒的话", status="DENIED", reason="QUIET", now=NOW - timedelta(seconds=10)
        )
        decision = await manager.evaluate(thought, "conv-1", "另一句", now=NOW)
        self.assertEqual(decision.code, "COOLDOWN")


if __name__ == "__main__":
    unittest.main()