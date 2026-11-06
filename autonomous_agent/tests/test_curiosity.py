"""Curiosity / Topic / Emotion / Diary 的第五阶段测试。"""

from __future__ import annotations

import re
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from autonomous_agent.config import AppConfig
from autonomous_agent.contact.manager import ContactManager
from autonomous_agent.core.emotion import EmotionModel
from autonomous_agent.core.thought_engine import ThoughtEngine
from autonomous_agent.curiosity.engine import CuriosityEngine
from autonomous_agent.curiosity.scoring import CuriosityCandidate, curiosity_score
from autonomous_agent.database.database import Database, MessageRepository, MetricsRepository
from autonomous_agent.database.migrations import initialize_database
from autonomous_agent.diary.repository import DiaryStore
from autonomous_agent.llm.adapter import MockLLMAdapter
from autonomous_agent.memory.models import Memory, MemoryType
from autonomous_agent.memory.store import MemoryStore
from autonomous_agent.memory.topics import TopicStatus, TopicStore, name_similarity
from autonomous_agent.tests import helpers
from autonomous_agent.thought.models import Thought, ThoughtStatus, ThoughtType
from autonomous_agent.thought.repository import ThoughtRepository
from autonomous_agent.thought.workspace import ThoughtWorkspace

NOW = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)


class CuriosityBase(unittest.IsolatedAsyncioTestCase):
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
        self.config = helpers.make_config(self.root)
        self.config.diary.enabled = True
        self.contact = ContactManager(self.db, self.config.contact)
        self.curiosity = CuriosityEngine(
            self.config.curiosity, self.repo, self.memories, self.topics, clock=lambda: NOW
        )
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

    async def _idle_topic(self, name: str = "自主思考系统") -> None:
        topic = await self.topics.touch(name, importance=0.8, now=NOW - timedelta(days=40))
        await self.topics.save(
            topic.with_changes(
                status=TopicStatus.LOW_ACTIVITY,
                last_activity_at=NOW - timedelta(days=40),
                thought_count=2,
            )
        )


class CuriosityDetectionTests(CuriosityBase):
    async def test_conflict_memory_creates_candidate(self) -> None:
        memory = await self.memories.add(
            Memory.create(
                type=MemoryType.FACT,
                content="主人不喜欢咖啡",
                confidence=0.8,
                source="user:1",
            )
        )
        await self.memories.mark_conflict(memory)
        candidates = await self.curiosity.detect(now=NOW)
        kinds = {candidate.kind for candidate in candidates}
        self.assertIn("contradiction", kinds)

    async def test_low_confidence_inference_creates_candidate(self) -> None:
        await self.memories.add(
            Memory.create(
                type=MemoryType.INFERENCE,
                content="主人可能更喜欢晚上聊天",
                confidence=0.4,
                source="thought:TH_1",
            )
        )
        candidates = await self.curiosity.detect(now=NOW)
        self.assertTrue(any(c.kind == "unverified" for c in candidates))

    async def test_idle_topic_creates_candidate(self) -> None:
        await self._idle_topic()
        candidates = await self.curiosity.detect(now=NOW)
        self.assertTrue(any(c.kind == "topic" for c in candidates))

    async def test_no_signal_returns_nothing(self) -> None:
        candidates = await self.curiosity.detect(now=NOW)
        self.assertEqual(candidates, [])

    async def test_topic_name_similarity_dedup(self) -> None:
        self.assertGreater(name_similarity("自主思考系统", "自主思考系统！"), 0.8)
        first = await self.topics.resolve("自主思考系统", importance=0.9, now=NOW)
        second = await self.topics.resolve("自主思考系统！", importance=0.9, now=NOW)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first.id, second.id)

    async def test_topic_needs_importance_to_create(self) -> None:
        created = await self.topics.resolve("随手一提", importance=0.1, now=NOW)
        self.assertIsNone(created)


class CuriosityMaterializeTests(CuriosityBase):
    async def test_step_creates_thought_once(self) -> None:
        await self._idle_topic()
        created = await self.curiosity.step(now=NOW)
        self.assertEqual(len(created), 1)
        thought = created[0]
        self.assertIs(thought.type, ThoughtType.CURIOSITY)
        self.assertTrue(thought.source.startswith("curiosity:"))
        self.assertEqual(await self.repo.count(), 1)
        again = await self.curiosity.step(now=NOW)
        self.assertEqual(again, [])
        self.assertEqual(await self.repo.count(), 1)

    async def test_step_respects_new_thought_cap(self) -> None:
        theme = self.config.curiosity
        theme.max_new_thoughts_per_tick = 1
        await self._idle_topic("自主思考系统")
        await self._idle_topic("长期记忆")
        created = await self.curiosity.step(now=NOW)
        self.assertLessEqual(len(created), 1)

    async def test_low_score_is_rejected(self) -> None:
        theme = self.config.curiosity
        theme.min_score = 9.0
        await self._idle_topic()
        self.assertEqual(await self.curiosity.step(now=NOW), [])

    async def test_score_disabled_flag(self) -> None:
        candidate = CuriosityCandidate(kind="topic", title="t", importance=1.0)
        cfg = self.config.curiosity
        self.assertGreater(curiosity_score(candidate, cfg).value, 0)


class EmotionTests(unittest.IsolatedAsyncioTestCase):
    async def test_delta_is_capped(self) -> None:
        config = AppConfig()
        config.emotion.max_emotion_delta_per_event = 0.05
        model = EmotionModel(config.emotion)
        state = model.initial()
        after = model.on_user_message(state)
        self.assertLessEqual(abs(after.interest - state.interest), 0.05 + 1e-9)
        self.assertLessEqual(abs(after.loneliness - state.loneliness), 0.05 + 1e-9)

    async def test_new_finding_hook_rewards_curiosity(self) -> None:
        config = AppConfig()
        model = EmotionModel(config.emotion)
        before = model.initial()
        after = model.on_new_finding(before)
        self.assertGreater(after.curiosity, before.curiosity)

    async def test_disabled_emotion_returns_same_state(self) -> None:
        config = AppConfig()
        config.emotion.enabled = False
        model = EmotionModel(config.emotion)
        before = model.initial()
        self.assertIs(model.on_user_message(before), before)


class EngineIntegrationTests(CuriosityBase):
    def _llm_answering_keep(self) -> MockLLMAdapter:
        pattern = re.compile(r"Thought id：(\S+)")

        def respond(request):
            match = pattern.search(request.messages[-1].content)
            assert match, "提示词里没有 Thought id"
            return helpers.think(match.group(1), disposition="KEEP")

        return MockLLMAdapter(responses=[respond])

    async def test_tick_grows_thought_from_curiosity(self) -> None:
        await self._idle_topic()
        llm = self._llm_answering_keep()
        result = await self.engine.step(llm, now=NOW)
        self.assertEqual(result.kind, "thought")
        self.assertEqual(len(llm.calls), 1)
        thoughts = await self.repo.list()
        self.assertEqual(len(thoughts), 1)
        self.assertIs(thoughts[0].type, ThoughtType.CURIOSITY)
        self.assertTrue(thoughts[0].source.startswith("curiosity:"))

    async def test_tick_does_nothing_without_signals(self) -> None:
        llm = MockLLMAdapter(responses=[helpers.DO_NOTHING])
        result = await self.engine.step(llm, now=NOW)
        self.assertEqual(result.kind, "nothing")
        self.assertEqual(len(llm.calls), 0)

    async def test_tick_keeps_one_llm_call(self) -> None:
        await self._idle_topic()
        llm = self._llm_answering_keep()
        await self.engine.step(llm, now=NOW)
        self.assertEqual(len(llm.calls), 1)
        await self.engine.step(llm, now=NOW + timedelta(minutes=5))
        self.assertLessEqual(len(llm.calls), 2)


class DiaryUncertaintyTests(CuriosityBase):
    async def test_diary_keeps_uncertainty_marker(self) -> None:
        thought, _ = await self.repo.create(
            Thought.create(
                type=ThoughtType.QUESTION,
                title="主人喜欢咖啡吗",
                content="还没有确认，只是猜测。",
                status=ThoughtStatus.QUEUED,
                now=NOW,
            )
        )
        self.assertIsNotNone(thought)
        llm = MockLLMAdapter(
            responses=[
                helpers.think(
                    thought.id,
                    insight="大概喜欢吧",
                    disposition="DIARY",
                    uncertainty="没有直接证据",
                )
            ]
        )
        result = await self.engine.step(llm, now=NOW)
        self.assertEqual(result.kind, "thought")
        day = NOW.date().isoformat()
        text = await self.diary.read(day)
        self.assertIsNotNone(text)
        self.assertIn("尚未确认", str(text))
        self.assertIn("没有直接证据", str(text))

    async def test_curiosity_never_sends_messages(self) -> None:
        await self._idle_topic()
        outbox = _OutboxProbe()
        self.assertFalse(hasattr(outbox, "sent"))
        await self.curiosity.step(now=NOW)
        self.assertEqual(await self.repo.count(), 1)


class _OutboxProbe:
    """占位：Curiosity 手里根本没有 QQ 适配器。"""


if __name__ == "__main__":
    unittest.main()