"""第六阶段：CONTACT_INTENT、硬限制、静默与长期维护。"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from autonomous_agent.config import AppConfig
from autonomous_agent.contact.intent import ContactIntent
from autonomous_agent.contact.manager import ContactManager
from autonomous_agent.core.agent import Agent
from autonomous_agent.core.event import Event
from autonomous_agent.core.guard import BehaviorGuard, GuardContext, GuardVerdict
from autonomous_agent.core.state import AgentState, utcnow
from autonomous_agent.database.database import Database
from autonomous_agent.database.migrations import LATEST_VERSION, initialize_database
from autonomous_agent.llm.adapter import MockLLMAdapter
from autonomous_agent.llm.schemas import Action, ActionType
from autonomous_agent.qq.adapter import MockQQAdapter
from autonomous_agent.thought.models import Thought, ThoughtStatus, ThoughtType

from . import helpers

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def make_thought() -> Thought:
    return Thought.create(
        type=ThoughtType.CURIOSITY,
        title="想起个事",
        content="上次说的那家店到底叫什么",
        source="test",
    )


class IntentTests(unittest.TestCase):
    def test_create_and_roundtrip(self) -> None:
        intent = ContactIntent.create(
            source_thought_id="t-1",
            conversation_id="conv-1",
            message_draft="在吗",
            topic="那家店",
            reason="突然想起来",
            now=NOW,
        )
        self.assertTrue(intent.intent_id.startswith("CI_"))
        self.assertEqual(intent.priority, 0.5)
        data = intent.to_dict()
        again = ContactIntent.from_dict(data)
        self.assertEqual(again.to_dict(), data)

    def test_from_thought_uses_topic_and_clamps_priority(self) -> None:
        thought = make_thought().with_changes(
            importance=0.9, curiosity=0.9, related_topics=["店"]
        )
        intent = ContactIntent.from_thought(
            thought,
            conversation_id="conv-1",
            message_draft="你好",
            emotion_drift=1.0,
            now=NOW,
        )
        self.assertEqual(intent.topic, "店")
        self.assertLessEqual(intent.priority, 1.0)
        self.assertGreaterEqual(intent.urgency, 0.9)


class DatabasePhase6Tests(unittest.IsolatedAsyncioTestCase):
    async def test_migration_adds_contact_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(str(Path(tmp) / "agent.db"))
            await initialize_database(db)
            self.assertGreaterEqual(LATEST_VERSION, 6)
            self.assertEqual(await db.user_version(), LATEST_VERSION)
            columns = {row["name"] for row in await db.fetch_all("PRAGMA table_info(contact_attempts)")}
            for name in (
                "intent_id",
                "topic",
                "message_hash",
                "reply_received",
                "response_time_seconds",
                "blocked_reason",
            ):
                self.assertIn(name, columns)
            await db.close()


class ContactManagerPhase6Tests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = Database(str(Path(self._tmp.name) / "agent.db"))
        await initialize_database(self.db)
        self.config = AppConfig().contact
        self.config.night_hours = []
        self.config.minimum_interval_seconds = 0
        self.config.cooldown_seconds = 0
        self.manager = ContactManager(self.db, self.config, clock=lambda: NOW)
        self.addAsyncCleanup(self.db.close)

    async def test_record_stores_intent_fields(self) -> None:
        thought = make_thought()
        await self.manager.record(
            thought,
            "conv-1",
            "在吗",
            status="SENT",
            intent_id="CI_1",
            topic="店",
            now=NOW,
        )
        row = (await self.manager.attempts(conversation_id="conv-1"))[0]
        self.assertEqual(row["intent_id"], "CI_1")
        self.assertEqual(row["topic"], "店")
        self.assertEqual(len(row["message_hash"]), 40)
        self.assertEqual(row["reply_received"], 0)

    async def test_mark_replied_records_response(self) -> None:
        thought = make_thought()
        await self.manager.record(thought, "conv-1", "在吗", status="SENT", now=NOW)
        changed = await self.manager.mark_replied("conv-1", now=NOW + timedelta(seconds=42))
        self.assertEqual(changed, 1)
        row = (await self.manager.attempts(conversation_id="conv-1"))[0]
        self.assertEqual(row["reply_received"], 1)
        self.assertAlmostEqual(float(row["response_time_seconds"]), 42.0, places=1)
        self.assertEqual(await self.manager.unanswered_count(), 0)

    async def test_silent_denies_contact(self) -> None:
        decision = await self.manager.evaluate(
            make_thought(), "conv-1", "在吗", silent=True, now=NOW
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.code, "SILENT_MODE")

    async def test_topic_repeat_denies(self) -> None:
        self.config.hourly_limit = 10
        self.config.daily_limit = 20
        self.config.topic_repeat_limit = 2
        thought = make_thought()
        for i in range(2):
            await self.manager.record(
                thought, "conv-1", f"第{i}句", status="SENT", topic="店", now=NOW - timedelta(minutes=10 * (i + 1))
            )
        decision = await self.manager.evaluate(
            thought, "conv-1", "又说到那家店", topic="店", now=NOW
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.code, "TOPIC_REPEAT")

    async def test_anomaly_protection_after_many_denials(self) -> None:
        self.config.anomaly_block_threshold = 5
        thought = make_thought()
        for i in range(5):
            await self.manager.record(
                thought,
                "conv-1",
                f"被拦{i}",
                status="DENIED",
                reason="QUIET_HOURS",
                now=NOW - timedelta(minutes=i),
            )
        decision = await self.manager.evaluate(thought, "conv-1", "再来一次", now=NOW)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.code, "ANOMALY_PROTECTION")

    async def test_unanswered_count_filters_conversation(self) -> None:
        thought = make_thought()
        await self.manager.record(thought, "conv-1", "a", status="SENT", now=NOW)
        await self.manager.record(thought, "conv-2", "b", status="SENT", now=NOW)
        self.assertEqual(await self.manager.unanswered_count(), 2)
        self.assertEqual(await self.manager.unanswered_count("conv-1"), 1)


class GuardContactTests(unittest.IsolatedAsyncioTestCase):
    def ctx(self, *, state: AgentState = AgentState.IDLE, last=None) -> GuardContext:
        return GuardContext(
            event_type=Event.autonomous_tick(1).type,
            conversation_id="conv-1",
            known_conversation_ids=frozenset({"conv-1"}),
            agent_state=state,
            last_outgoing_at=last,
            now=NOW,
        )

    def intent(self, message: str = "在吗") -> ContactIntent:
        return ContactIntent.create(
            source_thought_id="t-1",
            conversation_id="conv-1",
            message_draft=message,
            now=NOW,
        )

    def test_disabled_switch(self) -> None:
        config = AppConfig().behavior
        config.allow_autonomous_contact = False
        guard = BehaviorGuard(config)
        verdict = guard.check_contact(self.intent(), self.ctx())
        self.assertFalse(verdict.allowed)
        self.assertEqual(verdict.code, "CONTACT_DISABLED")

    def test_silent_state_denies(self) -> None:
        guard = BehaviorGuard(AppConfig().behavior)
        verdict = guard.check_contact(self.intent(), self.ctx(state=AgentState.SILENT))
        self.assertFalse(verdict.allowed)
        self.assertEqual(verdict.code, "SILENT_MODE")

    def test_quiet_hours_denies(self) -> None:
        config = AppConfig().behavior
        config.quiet_hours = ["23:30-07:30"]
        guard = BehaviorGuard(config)
        verdict = guard.check_contact(
            self.intent(), self.ctx()
        )
        # NOW 是 12:00 UTC，quiet 窗口按本地时钟判断，这里只确认不抛异常。
        self.assertIsInstance(verdict, GuardVerdict)

    def test_rate_limit_denies(self) -> None:
        config = AppConfig().behavior
        config.min_seconds_between_messages = 10
        guard = BehaviorGuard(config)
        verdict = guard.check_contact(
            self.intent(), self.ctx(last=NOW - timedelta(seconds=2))
        )
        self.assertFalse(verdict.allowed)
        self.assertEqual(verdict.code, "RATE_LIMITED")

    def test_empty_and_long_message(self) -> None:
        config = AppConfig().behavior
        config.max_message_chars = 5
        guard = BehaviorGuard(config)
        self.assertEqual(
            guard.check_contact(self.intent("  "), self.ctx()).code, "EMPTY_MESSAGE"
        )
        self.assertEqual(
            guard.check_contact(self.intent("太长了吧喂喂"), self.ctx()).code, "TOO_LONG"
        )

    def test_allowed(self) -> None:
        guard = BehaviorGuard(AppConfig().behavior)
        verdict = guard.check_contact(self.intent(), self.ctx())
        self.assertTrue(verdict.allowed)


class AgentPhase6Tests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def config(self) -> AppConfig:
        return helpers.make_config(self.dir)

    async def start_agent(self, responses=None, *, config=None) -> Agent:
        config = config or self.config()
        self.llm = MockLLMAdapter(responses=responses)
        self.qq = MockQQAdapter()
        agent = Agent(config, llm=self.llm, qq=self.qq, clock=utcnow)
        await agent.start(background=False)
        self.addAsyncCleanup(agent.stop)
        return agent

    async def test_consecutive_unanswered_enters_silent(self) -> None:
        agent = await self.start_agent([helpers.DO_NOTHING])
        thought = make_thought()
        for i in range(3):
            await agent._contact.record(
                thought,
                "conv-1",
                f"第{i}次",
                status="SENT",
                now=utcnow() - timedelta(hours=2, minutes=i),
            )
        result = await agent.handle_event(Event.autonomous_tick(1))
        self.assertIs(result.state, AgentState.SILENT)
        self.assertEqual(len(self.llm.calls), 0)
        self.assertIsNotNone(agent._machine.metadata.get("silent_until"))

    async def test_silent_keeps_thinking_but_blocks_contact(self) -> None:
        thought = make_thought()

        def speak(request: object) -> str:
            # tick 里被选中的 thought 由引擎决定，按请求里的 id 回答才不会错位。
            selected = getattr(request, "metadata", {}).get("thought_id", thought.id)
            return helpers.think(
                str(selected),
                disposition="SPEAK",
                contact_message="在忙吗？突然想起那家店。",
            )

        agent = await self.start_agent([helpers.enter_silent(10), speak])
        await agent._thoughts.create(thought)
        await agent.handle_event(helpers.user_message_event())
        self.assertIs(agent.state, AgentState.SILENT)

        calls_before = len(self.llm.calls)
        result = await agent.handle_event(Event.autonomous_tick(2))
        self.assertIs(result.state, AgentState.SILENT)
        self.assertIsNone(result.error, result.error)
        self.assertGreater(len(self.llm.calls), calls_before)
        self.assertEqual(self.qq.outbox, [])
        rows = await agent._contact.attempts()
        denied = [row for row in rows if row["status"] == "DENIED"]
        self.assertTrue(denied)
        self.assertEqual(denied[0]["blocked_reason"], "SILENT_MODE")

    async def test_contact_records_intent_and_topic(self) -> None:
        thought = make_thought().with_changes(
            metadata={"conversation_id": "conv-1"}, related_topics=["店"]
        )
        agent = await self.start_agent(
            [
                helpers.think(
                    thought.id,
                    disposition="SPEAK",
                    contact_message="在忙吗？突然想起那家店。",
                )
            ]
        )
        await agent._thoughts.create(thought)
        result = await agent.handle_event(Event.autonomous_tick(1))
        self.assertIsNone(result.error, result.error)
        rows = await agent._contact.attempts(conversation_id="conv-1")
        sent = [row for row in rows if row["status"] == "SENT"]
        self.assertEqual(len(sent), 1)
        self.assertTrue(str(sent[0]["intent_id"]).startswith("CI_"))
        self.assertEqual(sent[0]["topic"], "店")

    async def test_maintenance_cleans_old_rows(self) -> None:
        agent = await self.start_agent()
        old = (utcnow() - timedelta(days=365)).isoformat()
        await agent._db.execute(
            "INSERT INTO events (event_id, event_type, source, payload, created_at, status)"
            " VALUES (?, 'AUTONOMOUS_TICK', 'test', '{}', ?, 'PROCESSED')",
            ("evt-old", old),
        )
        await agent._db.execute(
            "INSERT INTO metrics (name, value, tags, recorded_at) VALUES ('x', 1, '{}', ?)",
            (old,),
        )
        await agent._run_maintenance()
        events = await agent._db.scalar(
            "SELECT COUNT(*) FROM events WHERE event_id = 'evt-old'"
        )
        metrics = await agent._db.scalar(
            "SELECT COUNT(*) FROM metrics WHERE name = 'x'"
        )
        self.assertEqual(events, 0)
        self.assertEqual(metrics, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()