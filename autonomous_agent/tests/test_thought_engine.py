"""思考引擎端到端测试（不经过 Agent 的完整事件流）。"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from autonomous_agent.config import AppConfig
from autonomous_agent.contact.manager import ContactManager
from autonomous_agent.core.emotion import EmotionModel
from autonomous_agent.core.thought_engine import ThoughtEngine
from autonomous_agent.database.database import Database, MessageRepository, MetricsRepository
from autonomous_agent.database.migrations import initialize_database
from autonomous_agent.diary.repository import DiaryStore
from autonomous_agent.llm.adapter import MockLLMAdapter
from autonomous_agent.memory.store import MemoryStore
from autonomous_agent.memory.topics import TopicStore
from autonomous_agent.tests import helpers
from autonomous_agent.thought.models import Disposition, ThoughtStatus, ThoughtType
from autonomous_agent.thought.repository import ThoughtRepository
from autonomous_agent.thought.workspace import ThoughtWorkspace

NOW = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)


class ThoughtEngineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.db = Database(self.root / "agent.db")
        await initialize_database(self.db)
        self.workspace = ThoughtWorkspace(self.root / "thinking")
        self.workspace.ensure_layout()
        self.repo = ThoughtRepository(self.db, self.workspace)
        self.memories = MemoryStore(self.db)
        self.topics = TopicStore(self.db)
        self.diary = DiaryStore(self.db, self.workspace.diary_dir)
        self.messages = MessageRepository(self.db)
        self.metrics = MetricsRepository(self.db)
        self.config = AppConfig()
        self.config.contact.minimum_interval_seconds = 0
        self.config.contact.cooldown_seconds = 0
        self.contact = ContactManager(self.db, self.config.contact)
        self.engine = ThoughtEngine(
            self.config,
            repository=self.repo,
            memories=self.memories,
            topics=self.topics,
            diary=self.diary,
            contact=self.contact,
            messages=self.messages,
            emotion_model=EmotionModel(self.config.emotion),
            metrics=self.metrics,
            clock=lambda: NOW,
        )
        self.addAsyncCleanup(self._cleanup)

    async def _cleanup(self) -> None:
        await self.db.close()
        self._tmp.cleanup()

    async def test_user_question_creates_thought(self) -> None:
        thought = await self.engine.ingest_user_message(
            conversation_id="conv-1",
            user_id="user-1",
            content="为什么天空是蓝的？",
            now=NOW,
        )
        assert thought is not None
        self.assertIs(thought.type, ThoughtType.QUESTION)
        self.assertEqual(await self.repo.count(), 1)
        self.assertEqual(await self.memories.count(), 1)

    async def test_plain_message_does_not_create_thought(self) -> None:
        thought = await self.engine.ingest_user_message(
            conversation_id="conv-1",
            user_id="user-1",
            content="早",
            now=NOW,
        )
        self.assertIsNone(thought)
        self.assertEqual(await self.repo.count(), 0)

    async def test_step_applies_keep_and_saves_memory(self) -> None:
        thought = await self.engine.ingest_user_message(
            conversation_id="conv-1",
            user_id="user-1",
            content="为什么天空是蓝的？",
            now=NOW,
        )
        assert thought is not None
        llm = MockLLMAdapter([helpers.think(thought.id, insight="因为瑞利散射")])

        result = await self.engine.step(llm, conversation_id="conv-1", now=NOW)
        self.assertEqual(result.kind, "thought")
        assert result.thought is not None
        self.assertIs(result.thought.status, ThoughtStatus.INSIGHT_FOUND)
        self.assertEqual(result.thought.revisit_count, 1)
        self.assertIsNotNone(await self.repo.get(thought.id))
        memories = await self.memories.list()
        self.assertTrue(any("瑞利散射" in item.content for item in memories))
        self.assertEqual(len(llm.calls), 1)

    async def test_step_without_candidates_skips_llm(self) -> None:
        llm = MockLLMAdapter([helpers.DO_NOTHING])
        result = await self.engine.step(llm, conversation_id="conv-1", now=NOW)
        self.assertEqual(result.kind, "nothing")
        self.assertEqual(llm.calls, [])

    async def test_bad_llm_output_returns_to_queue(self) -> None:
        thought = await self.engine.ingest_user_message(
            conversation_id="conv-1",
            user_id="user-1",
            content="为什么？",
            now=NOW,
        )
        assert thought is not None
        llm = MockLLMAdapter(["这不是 JSON"])

        result = await self.engine.step(llm, conversation_id="conv-1", now=NOW)
        self.assertEqual(result.error, "JSON_PARSE_ERROR")
        reloaded = await self.repo.get(thought.id)
        assert reloaded is not None
        self.assertIs(reloaded.status, ThoughtStatus.QUEUED)
        self.assertEqual(reloaded.metadata.get("last_error"), "JSON_PARSE_ERROR")
        assert reloaded.next_check_at is not None
        self.assertGreater(reloaded.next_check_at, NOW + timedelta(minutes=4))

    async def test_unknown_thought_id_rejected(self) -> None:
        thought = await self.engine.ingest_user_message(
            conversation_id="conv-1",
            user_id="user-1",
            content="为什么？",
            now=NOW,
        )
        assert thought is not None
        llm = MockLLMAdapter([helpers.think("TH_does_not_exist")])

        result = await self.engine.step(llm, conversation_id="conv-1", now=NOW)
        self.assertEqual(result.error, "INVALID_THOUGHT_ID")
        reloaded = await self.repo.get(thought.id)
        assert reloaded is not None
        self.assertIs(reloaded.status, ThoughtStatus.QUEUED)

    async def test_wait_disposition_sets_next_check(self) -> None:
        thought = await self.engine.ingest_user_message(
            conversation_id="conv-1",
            user_id="user-1",
            content="为什么会这样？",
            now=NOW,
        )
        assert thought is not None
        llm = MockLLMAdapter(
            [helpers.think(thought.id, disposition="WAIT", next_check_minutes=30)]
        )
        result = await self.engine.step(llm, conversation_id="conv-1", now=NOW)
        assert result.thought is not None
        self.assertIs(result.thought.status, ThoughtStatus.WAITING)
        self.assertEqual(result.thought.next_check_at, NOW + timedelta(minutes=30))

    async def test_continue_reschedules_soon_and_waits_its_turn(self) -> None:
        thought = await self.engine.ingest_user_message(
            conversation_id="conv-1",
            user_id="user-1",
            content="这件事我想得更深一点？",
            now=NOW,
        )
        assert thought is not None
        llm = MockLLMAdapter([helpers.think(thought.id, disposition="CONTINUE")])
        result = await self.engine.step(llm, conversation_id="conv-1", now=NOW)
        assert result.thought is not None
        self.assertIs(result.thought.status, ThoughtStatus.QUEUED)
        self.assertEqual(
            result.thought.next_check_at, NOW + timedelta(minutes=3)
        )
        # 到点前不抢跑，到点后重新进候选。
        self.assertEqual(await self.repo.candidates(NOW), [])
        later = await self.repo.candidates(NOW + timedelta(minutes=4))
        self.assertEqual([item.id for item in later], [thought.id])

    async def test_has_pending_work_tracks_the_pool(self) -> None:
        self.assertFalse(await self.engine.has_pending_work(NOW))
        thought = await self.engine.ingest_user_message(
            conversation_id="conv-1",
            user_id="user-1",
            content="为什么天空是蓝的？",
            now=NOW,
        )
        assert thought is not None
        self.assertTrue(await self.engine.has_pending_work(NOW))

    async def test_diary_disposition_writes_file(self) -> None:
        thought = await self.engine.ingest_user_message(
            conversation_id="conv-1",
            user_id="user-1",
            content="为什么人会觉得孤独？",
            now=NOW,
        )
        assert thought is not None
        llm = MockLLMAdapter(
            [helpers.think(thought.id, insight="孤独是想要连接的信号", disposition="DIARY")]
        )
        await self.engine.step(llm, conversation_id="conv-1", now=NOW)
        content = await self.diary.read("2026-01-01")
        assert content is not None
        self.assertIn("孤独是想要连接的信号", content)

    async def test_speak_denied_when_contact_disabled(self) -> None:
        self.config.contact.enabled = False
        thought = await self.engine.ingest_user_message(
            conversation_id="conv-1",
            user_id="user-1",
            content="她为什么不回我？",
            now=NOW,
        )
        assert thought is not None
        llm = MockLLMAdapter(
            [
                helpers.think(
                    thought.id,
                    disposition="SPEAK",
                    contact_message="在忙吗？突然想到你。",
                )
            ]
        )
        result = await self.engine.step(llm, conversation_id="conv-1", now=NOW)
        self.assertIsNone(result.contact)
        self.assertEqual(result.denied_reason, "CONTACT_DISABLED")
        attempts = await self.contact.attempts(conversation_id="conv-1")
        self.assertEqual(attempts[0]["status"], "DENIED")

    async def test_speak_returns_contact_intent(self) -> None:
        thought = await self.engine.ingest_user_message(
            conversation_id="conv-1",
            user_id="user-1",
            content="要不要约她出来？",
            now=NOW,
        )
        assert thought is not None
        llm = MockLLMAdapter(
            [
                helpers.think(
                    thought.id,
                    disposition="SPEAK",
                    contact_message="在忙吗？突然想到你。",
                )
            ]
        )
        result = await self.engine.step(llm, conversation_id="conv-1", now=NOW)
        assert result.contact is not None
        self.assertEqual(result.contact.conversation_id, "conv-1")
        self.assertEqual(result.contact.thought_id, thought.id)
        self.assertIs(result.outcome.disposition, Disposition.SPEAK)

    async def test_new_questions_spawn_children(self) -> None:
        thought = await self.engine.ingest_user_message(
            conversation_id="conv-1",
            user_id="user-1",
            content="为什么？",
            now=NOW,
        )
        assert thought is not None
        llm = MockLLMAdapter(
            [
                helpers.think(
                    thought.id,
                    new_questions=[{"title": "那怎么办？", "content": "继续想"}],
                )
            ]
        )
        await self.engine.step(llm, conversation_id="conv-1", now=NOW)
        questions = await self.repo.list(statuses=[ThoughtStatus.NEW])
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0].parent_thought_id, thought.id)
        relations = await self.repo.relations_of(thought.id)
        kinds = [row["kind"] for row in relations]
        self.assertIn("CHILD", kinds)


if __name__ == "__main__":
    unittest.main()