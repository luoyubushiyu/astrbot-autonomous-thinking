from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from autonomous_agent.config import AppConfig
from autonomous_agent.core.agent import Agent
from autonomous_agent.core.event import Event, EventType
from autonomous_agent.core.state import (
    AgentState,
    StateSnapshot,
    ThoughtState,
    utcnow,
)
from autonomous_agent.database.database import Database, DatabaseError
from autonomous_agent.database.migrations import LATEST_VERSION, initialize_database
from autonomous_agent.llm.adapter import LLMTimeoutError, MockLLMAdapter
from autonomous_agent.llm.schemas import ActionType
from autonomous_agent.qq.adapter import MockQQAdapter

from autonomous_agent.thought.models import Thought, ThoughtStatus, ThoughtType

from . import helpers


class AgentTestBase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def config(self, **tweaks) -> AppConfig:
        config = helpers.make_config(self.dir)
        for key, value in tweaks.items():
            if key == "quiet_hours":
                config.behavior.quiet_hours = value
            elif key == "timeout":
                config.llm.request_timeout_seconds = value
            else:  # pragma: no cover
                raise KeyError(key)
        return config

    async def start_agent(
        self,
        responses=None,
        *,
        config: AppConfig | None = None,
        llm: MockLLMAdapter | None = None,
        qq: MockQQAdapter | None = None,
        clock=None,
    ) -> Agent:
        config = config or self.config()
        self.llm = llm or MockLLMAdapter(responses=responses)
        self.qq = qq or MockQQAdapter()
        agent = Agent(config, llm=self.llm, qq=self.qq, clock=clock or utcnow)
        await agent.start(background=False)
        self.addAsyncCleanup(agent.stop)
        return agent


class StartupTests(AgentTestBase):
    async def test_start_enters_idle_and_records_event(self) -> None:
        agent = await self.start_agent()
        self.assertTrue(agent.running)
        self.assertIs(agent.state, AgentState.IDLE)

        events = await agent._events.recent(5)
        self.assertTrue(any(e.event_type == "BOT_START" for e in events))
        self.assertEqual(events[0].status, "PROCESSED")

    async def test_state_persisted_to_database(self) -> None:
        agent = await self.start_agent()
        snapshot = await agent._states.load()
        assert snapshot is not None
        self.assertIs(snapshot.state, AgentState.IDLE)
        thought = await agent.stats()
        self.assertEqual(thought["running"], True)

    async def test_restart_recovers_transient_state(self) -> None:
        # 模拟上一次进程在 THINKING 时被杀死
        db = Database(str(self.dir / "agent.db"))
        await initialize_database(db)
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        await db.close()

        # 用同一路径重新打开并写入瞬态快照
        db = Database(str(self.dir / "agent.db"))
        await initialize_database(db)
        from autonomous_agent.database.database import StateRepository

        await StateRepository(db).save(
            StateSnapshot(
                state=AgentState.THINKING,
                entered_at=now,
                updated_at=now,
                previous_state=AgentState.OBSERVING,
            )
        )
        await db.close()

        agent = await self.start_agent()
        self.assertIs(agent.state, AgentState.SAFE_IDLE)

        await agent.stop()
        db = Database(str(self.dir / "agent.db"))
        await initialize_database(db)
        from autonomous_agent.database.database import StateRepository as Repo

        loaded = await Repo(db).load()
        await db.close()
        assert loaded is not None
        self.assertIs(loaded.state, AgentState.SAFE_IDLE)

    async def test_start_failure_releases_database(self) -> None:
        # 模拟数据库被更高版本的程序写过：启动要报错，且不能留下打开的连接。
        path = self.dir / "agent.db"
        db = Database(str(path))
        await db.connect()
        await db.execute(f"PRAGMA user_version = {LATEST_VERSION + 1}")
        await db.close()

        agent = Agent(self.config(), llm=MockLLMAdapter())
        with self.assertRaises(DatabaseError):
            await agent.start(background=False)
        self.assertFalse(agent.running)


class UserMessageTests(AgentTestBase):
    async def test_reply_with_send_message(self) -> None:
        agent = await self.start_agent([helpers.send_message()])
        result = await agent.handle_event(helpers.user_message_event())

        self.assertIsNone(result.error)
        assert result.decision is not None
        self.assertIs(result.decision.action.type, ActionType.SEND_MESSAGE)
        self.assertIs(result.state, AgentState.IDLE)
        self.assertEqual(len(self.qq.outbox), 1)
        self.assertEqual(self.qq.outbox[0].conversation_id, "conv-1")
        self.assertEqual(self.qq.outbox[0].content, "在的呀")

        messages = await agent._messages.recent("conv-1")
        self.assertEqual([m.role for m in messages], ["user", "assistant"])

    async def test_do_nothing(self) -> None:
        agent = await self.start_agent([helpers.DO_NOTHING])
        result = await agent.handle_event(helpers.user_message_event())

        assert result.decision is not None
        self.assertIs(result.decision.action.type, ActionType.DO_NOTHING)
        self.assertIs(result.state, AgentState.IDLE)
        self.assertEqual(self.qq.outbox, [])

    async def test_duplicate_message_ignored(self) -> None:
        agent = await self.start_agent([helpers.send_message()])
        event = helpers.user_message_event()
        await agent.handle_event(event)
        await agent.handle_event(event)

        self.assertEqual(len(self.qq.outbox), 1)
        self.assertEqual(await agent._messages.count(), 2)  # user + assistant

    async def test_wait_then_alarm(self) -> None:
        agent = await self.start_agent([helpers.wait(5)])
        result = await agent.handle_event(helpers.user_message_event())
        self.assertIs(result.state, AgentState.WAITING)
        self.assertIn("wait_until", agent._machine.metadata)
        self.assertIn("alarm_id", agent._machine.metadata)

        alarm = Event.alarm_triggered(
            agent._machine.metadata["alarm_id"], "wait_elapsed"
        )
        result = await agent.handle_event(alarm)
        self.assertIs(result.state, AgentState.IDLE)

    async def test_enter_silent(self) -> None:
        agent = await self.start_agent([helpers.enter_silent(10)])
        result = await agent.handle_event(helpers.user_message_event())
        self.assertIs(result.state, AgentState.SILENT)
        self.assertIn("silent_until", agent._machine.metadata)
        self.assertIn("alarm_id", agent._machine.metadata)

        await agent.handle_event(
            Event.alarm_triggered(agent._machine.metadata["alarm_id"], "silent_elapsed")
        )
        self.assertIs(agent.state, AgentState.IDLE)


class AlarmTests(AgentTestBase):
    async def test_stale_alarm_does_not_clear_new_wait(self) -> None:
        agent = await self.start_agent([helpers.wait(5), helpers.wait(5)])
        await agent.handle_event(helpers.user_message_event())
        old_alarm = agent._machine.metadata["alarm_id"]

        await agent.handle_event(
            helpers.user_message_event(message_id="m-2", content="再等等")
        )
        new_alarm = agent._machine.metadata["alarm_id"]
        self.assertNotEqual(old_alarm, new_alarm)

        result = await agent.handle_event(
            Event.alarm_triggered(old_alarm, "wait_elapsed")
        )
        self.assertIs(result.state, AgentState.WAITING)
        self.assertGreaterEqual(await agent._metrics.count("alarm.stale"), 1)

        result = await agent.handle_event(
            Event.alarm_triggered(new_alarm, "wait_elapsed")
        )
        self.assertIs(result.state, AgentState.IDLE)


class SilentResumeTests(AgentTestBase):
    async def _seed_silent(
        self, *, seconds: float, alarm_id: str | None = "silent:seeded"
    ) -> None:
        from autonomous_agent.database.database import StateRepository

        now = datetime.now(timezone.utc)
        metadata: dict[str, object] = {
            "silent_until": (now + timedelta(seconds=seconds)).isoformat()
        }
        if alarm_id is not None:
            metadata["alarm_id"] = alarm_id
        db = Database(str(self.dir / "agent.db"))
        await initialize_database(db)
        await StateRepository(db).save(
            StateSnapshot(
                state=AgentState.SILENT,
                entered_at=now,
                updated_at=now,
                metadata=metadata,
            )
        )
        await db.close()

    async def test_restored_silent_timer_resumes(self) -> None:
        await self._seed_silent(seconds=0.4)
        agent = Agent(self.config(), llm=MockLLMAdapter(), qq=MockQQAdapter())
        await agent.start(background=True)
        self.addAsyncCleanup(agent.stop)
        self.assertIs(agent.state, AgentState.SILENT)

        await asyncio.sleep(1.0)
        self.assertIs(agent.state, AgentState.IDLE)

    async def test_restored_silent_without_alarm_id_gets_one(self) -> None:
        await self._seed_silent(seconds=30, alarm_id=None)
        agent = Agent(self.config(), llm=MockLLMAdapter(), qq=MockQQAdapter())
        await agent.start(background=True)
        self.addAsyncCleanup(agent.stop)

        self.assertIs(agent.state, AgentState.SILENT)
        self.assertTrue(agent._machine.metadata.get("alarm_id"))


class AutonomousTickTests(AgentTestBase):
    async def test_tick_do_nothing(self) -> None:
        agent = await self.start_agent([helpers.DO_NOTHING])
        result = await agent.handle_event(Event.autonomous_tick(1))

        assert result.decision is not None
        self.assertIs(result.decision.action.type, ActionType.DO_NOTHING)
        self.assertIs(result.state, AgentState.IDLE)
        self.assertEqual(self.qq.outbox, [])

    async def test_tick_without_thoughts_skips_llm_and_never_sends(self) -> None:
        agent = await self.start_agent([helpers.send_message()])
        await agent._messages.add(
            conversation_id="conv-1", message_id="m-old", role="user", content="旧消息"
        )
        calls_before = len(agent._llm.calls)

        result = await agent.handle_event(Event.autonomous_tick(1))
        assert result.decision is not None
        self.assertIs(result.decision.action.type, ActionType.DO_NOTHING)
        self.assertIs(result.state, AgentState.IDLE)
        self.assertEqual(len(agent._llm.calls), calls_before)
        self.assertEqual(self.qq.outbox, [])

    async def test_tick_still_thinks_while_silent(self) -> None:
        agent = await self.start_agent([helpers.enter_silent(10)])
        await agent.handle_event(helpers.user_message_event())
        self.assertIs(agent.state, AgentState.SILENT)

        calls_before = len(self.llm.calls)
        result = await agent.handle_event(Event.autonomous_tick(2))
        self.assertIs(result.state, AgentState.SILENT)
        # 静默只停主动联系，不停思考：LLM 仍被调用。
        self.assertGreater(len(self.llm.calls), calls_before)
        self.assertEqual(self.qq.outbox, [])


class FailureFallbackTests(AgentTestBase):
    async def test_llm_timeout_falls_back_to_safe_idle(self) -> None:
        agent = await self.start_agent(
            config=self.config(timeout=0.05),
            llm=MockLLMAdapter(responses=[helpers.send_message()], delay_seconds=5),
        )
        result = await agent.handle_event(helpers.user_message_event())

        self.assertEqual(result.error, "LLM_TIMEOUT")
        self.assertIs(result.state, AgentState.SAFE_IDLE)
        self.assertEqual(self.qq.outbox, [])

    async def test_llm_exception_falls_back(self) -> None:
        agent = await self.start_agent([LLMTimeoutError("boom")])
        result = await agent.handle_event(helpers.user_message_event())
        self.assertIn("LLM_ERROR", result.error or "")
        self.assertIs(result.state, AgentState.SAFE_IDLE)
        self.assertEqual(self.qq.outbox, [])

    async def test_empty_response_falls_back(self) -> None:
        agent = await self.start_agent([""])
        result = await agent.handle_event(helpers.user_message_event())
        self.assertEqual(result.error, "EMPTY_RESPONSE")
        self.assertIs(result.state, AgentState.SAFE_IDLE)

    async def test_invalid_json_falls_back(self) -> None:
        agent = await self.start_agent(["这不是 JSON"])
        result = await agent.handle_event(helpers.user_message_event())
        self.assertEqual(result.error, "JSON_PARSE_ERROR")
        self.assertIs(result.state, AgentState.SAFE_IDLE)
        assert result.decision is not None
        self.assertIs(result.decision.action.type, ActionType.DO_NOTHING)

    async def test_invalid_schema_falls_back(self) -> None:
        agent = await self.start_agent(['{"action":{"type":"DO_NOTHING"}}'])
        result = await agent.handle_event(helpers.user_message_event())
        self.assertEqual(result.error, "SCHEMA_ERROR")
        self.assertIs(result.state, AgentState.SAFE_IDLE)

    async def test_unknown_conversation_id_falls_back(self) -> None:
        agent = await self.start_agent([helpers.send_message("conv-999")])
        result = await agent.handle_event(helpers.user_message_event())
        self.assertEqual(result.error, "UNKNOWN_ID")
        self.assertIs(result.state, AgentState.SAFE_IDLE)
        self.assertEqual(self.qq.outbox, [])

    async def test_invalid_parameter_falls_back(self) -> None:
        agent = await self.start_agent([helpers.wait(seconds=0)])
        result = await agent.handle_event(helpers.user_message_event())
        self.assertEqual(result.error, "INVALID_PARAMETER")
        self.assertIs(result.state, AgentState.SAFE_IDLE)
        assert result.decision is not None
        self.assertIs(result.decision.action.type, ActionType.DO_NOTHING)
        self.assertEqual(self.qq.outbox, [])

    async def test_send_failure_falls_back(self) -> None:
        agent = await self.start_agent([helpers.send_message()])
        self.qq.send_failure = RuntimeError("network down")
        result = await agent.handle_event(helpers.user_message_event())
        self.assertIsNotNone(result.error)
        self.assertIs(result.state, AgentState.SAFE_IDLE)

    async def test_recovers_after_safe_idle(self) -> None:
        llm = MockLLMAdapter(
            responses=[LLMTimeoutError("boom"), helpers.DO_NOTHING]
        )
        agent = await self.start_agent(llm=llm)
        await agent.handle_event(helpers.user_message_event())
        self.assertIs(agent.state, AgentState.SAFE_IDLE)

        result = await agent.handle_event(
            helpers.user_message_event(message_id="m-2", content="还在吗")
        )
        self.assertIs(result.state, AgentState.IDLE)
        self.assertIsNone(result.error)


class GuardTests(AgentTestBase):
    async def test_rate_limit(self) -> None:
        agent = await self.start_agent(
            [helpers.send_message(), helpers.send_message()]
        )
        await agent.handle_event(helpers.user_message_event())
        result = await agent.handle_event(
            helpers.user_message_event(message_id="m-2", content="再说一句")
        )
        assert result.verdict is not None
        self.assertEqual(result.verdict.code, "RATE_LIMITED")
        self.assertEqual(len(self.qq.outbox), 1)

    async def test_quiet_hours_block_send(self) -> None:
        agent = await self.start_agent(
            [helpers.send_message()], config=self.config(quiet_hours=["00:00-23:59"])
        )
        result = await agent.handle_event(helpers.user_message_event())
        assert result.verdict is not None
        self.assertEqual(result.verdict.code, "QUIET_HOURS")
        self.assertEqual(self.qq.outbox, [])
        self.assertIs(result.state, AgentState.IDLE)


class ConcurrencyTests(AgentTestBase):
    async def test_concurrent_events_are_serialized(self) -> None:
        agent = await self.start_agent([helpers.DO_NOTHING] * 5)
        events = [
            helpers.user_message_event(message_id=f"m-{i}", content=f"hi {i}")
            for i in range(5)
        ]
        results = await asyncio.gather(
            *(agent.handle_event(event) for event in events)
        )
        self.assertTrue(all(r.error is None for r in results))
        self.assertIs(agent.state, AgentState.IDLE)
        self.assertEqual(await agent._messages.count(), 5)


class RestartTests(AgentTestBase):
    async def test_restart_keeps_messages_and_state(self) -> None:
        config = self.config()
        qq1 = MockQQAdapter()
        first = Agent(config, llm=MockLLMAdapter([helpers.send_message()]), qq=qq1)
        await first.start(background=False)
        await first.handle_event(helpers.user_message_event())
        await first.stop()
        self.assertEqual(len(qq1.outbox), 1)

        qq2 = MockQQAdapter()
        second = Agent(config, llm=MockLLMAdapter([helpers.DO_NOTHING]), qq=qq2)
        await second.start(background=False)
        self.addAsyncCleanup(second.stop)
        self.assertIs(second.state, AgentState.IDLE)
        self.assertEqual(await second._messages.count(), 2)


class NightStateTests(AgentTestBase):
    def _clock(self, now: datetime):
        holder = {"now": now}
        return (lambda: holder["now"]), holder

    async def test_night_then_sleeping_then_wake(self) -> None:
        tz = datetime.now().astimezone().tzinfo
        clock, holder = self._clock(datetime(2026, 1, 1, 23, 30, tzinfo=tz))
        config = self.config()
        config.contact.night_hours = ["23:00-07:00"]
        agent = await self.start_agent([helpers.DO_NOTHING], config=config, clock=clock)

        await agent.handle_event(Event.autonomous_tick(1))
        self.assertIs(agent.state, AgentState.NIGHT)
        self.assertEqual(len(self.llm.calls), 0)

        await agent.handle_event(Event.autonomous_tick(2))
        self.assertIs(agent.state, AgentState.SLEEPING)

        holder["now"] = datetime(2026, 1, 2, 8, 0, tzinfo=tz)
        await agent.handle_event(Event.autonomous_tick(3))
        self.assertIs(agent.state, AgentState.IDLE)

        recent = (await agent.stats())["recent_states"]
        visited = [item["to"] for item in recent]
        self.assertIn("NIGHT", visited)
        self.assertIn("SLEEPING", visited)

    async def test_user_message_wakes_sleeping(self) -> None:
        tz = datetime.now().astimezone().tzinfo
        clock, holder = self._clock(datetime(2026, 1, 1, 23, 30, tzinfo=tz))
        config = self.config()
        config.contact.night_hours = ["23:00-07:00"]
        agent = await self.start_agent(
            [helpers.send_message()], config=config, clock=clock
        )
        await agent.handle_event(Event.autonomous_tick(1))
        await agent.handle_event(Event.autonomous_tick(2))
        self.assertIs(agent.state, AgentState.SLEEPING)

        result = await agent.handle_event(helpers.user_message_event())
        self.assertIsNone(result.error)
        self.assertIs(agent.state, AgentState.IDLE)
        self.assertEqual(len(self.qq.outbox), 1)

    async def test_no_night_window_keeps_idle(self) -> None:
        clock, _ = self._clock(datetime(2026, 1, 1, 3, 0, tzinfo=timezone.utc))
        agent = await self.start_agent([helpers.DO_NOTHING], clock=clock)
        await agent.handle_event(Event.autonomous_tick(1))
        self.assertIs(agent.state, AgentState.IDLE)


class ContactingStateTests(AgentTestBase):
    async def test_contact_transition_is_recorded(self) -> None:
        config = self.config()
        thought = Thought.create(
            type=ThoughtType.CURIOSITY,
            title="想起个事",
            content="上次说的那家店到底叫什么",
            source="test",
        ).with_changes(metadata={"conversation_id": "conv-1"})
        agent = await self.start_agent(
            [helpers.think(
                thought.id,
                disposition="SPEAK",
                contact_message="在忙吗？突然想起上次说的那家店。",
            )],
            config=config,
        )
        await agent._thoughts.create(thought)

        result = await agent.handle_event(Event.autonomous_tick(1))
        self.assertIsNone(result.error, result.error)
        self.assertEqual(len(self.qq.outbox), 1)
        recent = (await agent.stats())["recent_states"]
        pairs = [(item["from"], item["to"]) for item in recent]
        self.assertIn(("CONTACTING", "IDLE"), pairs)


class ContactTimeoutTests(AgentTestBase):
    async def test_due_timeout_reported_once(self) -> None:
        agent = await self.start_agent()
        thought = Thought.create(
            type=ThoughtType.CURIOSITY, title="想起个事", content="还在吗", source="test"
        )
        created = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        await agent._contact.record(
            thought, "conv-1", "在忙吗？", status="SENT", now=created
        )

        event = await agent._due_contact_timeout()
        assert event is not None
        self.assertIs(event.type, EventType.CONTACT_TIMEOUT)
        self.assertEqual(event.payload["conversation_id"], "conv-1")
        self.assertGreaterEqual(event.payload["waited_seconds"], 1800.0)
        # 同一条联系只报一次。
        self.assertIsNone(await agent._due_contact_timeout())

    async def test_replied_attempt_is_not_due(self) -> None:
        agent = await self.start_agent()
        thought = Thought.create(
            type=ThoughtType.CURIOSITY, title="想起个事", content="还在吗", source="test"
        )
        created = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        await agent._contact.record(
            thought, "conv-1", "在忙吗？", status="SENT", now=created
        )
        await agent._contact.mark_replied(
            "conv-1", now=created + timedelta(minutes=5)
        )
        self.assertIsNone(await agent._due_contact_timeout())

    async def test_contact_timeout_clears_waiting(self) -> None:
        agent = await self.start_agent()
        agent._machine.transition(AgentState.WAITING)
        await agent._persist_state()

        result = await agent.handle_event(Event.contact_timeout("conv-1", 1800.0))
        self.assertIsNone(result.error, result.error)
        self.assertIs(agent.state, AgentState.IDLE)
        self.assertEqual(await agent._metrics.total("contact.timeout"), 1)


if __name__ == "__main__":
    unittest.main()