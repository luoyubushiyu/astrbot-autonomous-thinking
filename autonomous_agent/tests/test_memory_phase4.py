"""第四阶段：Memory 生命周期、价值、遗忘、整理、重新激活。"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from autonomous_agent.config import AppConfig
from autonomous_agent.contact.manager import ContactManager
from autonomous_agent.core.emotion import EmotionModel
from autonomous_agent.core.thought_engine import ThoughtEngine
from autonomous_agent.database.database import (
    Database,
    MessageRepository,
    MetricsRepository,
)
from autonomous_agent.database.migrations import initialize_database
from autonomous_agent.diary.repository import DiaryStore
from autonomous_agent.llm.adapter import MockLLMAdapter
from autonomous_agent.memory.models import Memory, MemoryStatus, MemoryType
from autonomous_agent.memory.reactivation import reactivate
from autonomous_agent.memory.scoring import value_of
from autonomous_agent.memory.store import MemoryStore
from autonomous_agent.memory.topics import TopicStore
from autonomous_agent.memory.validator import MemoryCandidate, MemoryValidator
from autonomous_agent.tests import helpers
from autonomous_agent.thought.models import Thought, ThoughtStatus, ThoughtType
from autonomous_agent.thought.repository import ThoughtRepository
from autonomous_agent.thought.workspace import ThoughtWorkspace

NOW = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)


async def _store(tmp_path: Path, **overrides) -> tuple[Database, MemoryStore]:
    db = Database(tmp_path / "agent.db")
    await initialize_database(db)
    config = AppConfig().memory
    for key, value in overrides.items():
        setattr(config, key, value)
    return db, MemoryStore(db, clock=lambda: NOW, config=config)


def _memory(content: str, **kwargs) -> Memory:
    defaults = dict(
        type=MemoryType.FACT,
        content=content,
        confidence=0.8,
        importance=0.6,
        source="user:user-1",
        source_type="USER_MESSAGE",
        source_id="user-1",
        now=NOW,
    )
    defaults.update(kwargs)
    return Memory.create(**defaults)


class MemoryUnitTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.db, self.store = await _store(self.root)
        self.addAsyncCleanup(self._cleanup)

    async def _cleanup(self) -> None:
        await self.db.close()
        self._tmp.cleanup()

    async def test_value_rewards_importance_and_recency(self) -> None:
        fresh = _memory("今天学了点新东西", confidence=0.9, importance=0.9)
        stale = _memory(
            "很久以前的一件小事",
            confidence=0.2,
            importance=0.1,
            now=NOW - timedelta(days=120),
        )
        self.assertGreater(value_of(fresh, NOW), value_of(stale, NOW))

    async def test_soft_delete_ladder_and_restore(self) -> None:
        memory = await self.store.add(_memory("喜欢喝热牛奶"))
        for status in (
            MemoryStatus.LOW_VALUE,
            MemoryStatus.SOFT_DELETED,
            MemoryStatus.EXPIRED,
        ):
            memory = await self.store.set_status(memory, status)
        self.assertIs(memory.status, MemoryStatus.EXPIRED)
        # 软删除后不再进入常规检索
        self.assertEqual(await self.store.count(), 1)
        self.assertEqual(await self.store.search("热牛奶"), [])
        restored = await self.store.restore(memory, reason="用户又提到了")
        self.assertIs(restored.status, MemoryStatus.ACTIVE)
        self.assertEqual(restored.reactivation_count, 1)
        self.assertTrue(await self.store.search("热牛奶"))

    async def test_set_status_cannot_skip_steps(self) -> None:
        memory = await self.store.add(_memory("一步一级"))
        with self.assertRaises(ValueError):
            await self.store.set_status(memory, MemoryStatus.EXPIRED)

    async def test_inference_never_upgraded_to_fact(self) -> None:
        memory = await self.store.add(
            _memory("她可能喜欢猫", type=MemoryType.INFERENCE, confidence=0.6)
        )
        await self.store.save(memory.with_changes(type=MemoryType.FACT))
        stored = await self.store.get(memory.id)
        assert stored is not None
        self.assertIs(stored.type, MemoryType.INFERENCE)

    async def test_duplicate_candidate_updates_instead_of_create(self) -> None:
        await self.store.add(_memory("我喜欢在周末去公园"))
        validator = MemoryValidator()
        proposal = await validator.evaluate(
            MemoryCandidate(
                type=MemoryType.FACT,
                content="我喜欢在周末去公园",
                confidence=0.9,
                importance=0.7,
            ),
            self.store,
            source_type="USER_MESSAGE",
            source_id="user-1",
            now=NOW,
        )
        self.assertEqual(proposal.action, "UPDATE")
        assert proposal.target is not None
        self.assertEqual(proposal.target.content, "我喜欢在周末去公园")

    async def test_conflict_detected_not_overwritten(self) -> None:
        await self.store.add(_memory("我住在上海"))
        validator = MemoryValidator()
        proposal = await validator.evaluate(
            MemoryCandidate(
                type=MemoryType.FACT,
                content="我不住在上海",
                confidence=0.8,
            ),
            self.store,
            source_type="USER_MESSAGE",
            source_id="user-2",
            now=NOW,
        )
        self.assertEqual(proposal.action, "CONFLICT")
        stored = await self.store.list()
        self.assertEqual(stored[0].content, "我住在上海")

    async def test_thought_fact_downgraded_to_inference(self) -> None:
        validator = MemoryValidator()
        proposal = await validator.evaluate(
            MemoryCandidate(
                type=MemoryType.FACT,
                content="她应该很喜欢下雨天",
                confidence=0.8,
            ),
            self.store,
            source_type="THOUGHT",
            source_id="TH_1",
            now=NOW,
        )
        self.assertEqual(proposal.action, "CREATE")
        self.assertIs(proposal.candidate.type, MemoryType.INFERENCE)

    async def test_weak_candidate_is_not_saved(self) -> None:
        validator = MemoryValidator()
        proposal = await validator.evaluate(
            MemoryCandidate(type=MemoryType.FACT, content="嗯", confidence=0.9),
            self.store,
            source_type="USER_MESSAGE",
            source_id="user-1",
            now=NOW,
        )
        self.assertEqual(proposal.action, "DO_NOT_SAVE")

    async def test_sweep_degrades_worthless_memory(self) -> None:
        memory = await self.store.add(
            _memory(
                "早就没用的东西",
                confidence=0.2,
                importance=0.1,
                now=NOW - timedelta(days=180),
            )
        )
        changed = await self.store.sweep(now=NOW)
        self.assertIn(memory.id, [item.id for item in changed])
        stored = await self.store.get(memory.id)
        assert stored is not None
        self.assertIs(stored.status, MemoryStatus.EXPIRED)

    async def test_consolidate_keeps_one_copy(self) -> None:
        first = await self.store.add(_memory("常去的咖啡馆在路口"))
        second = await self.store.add(
            _memory("常去的咖啡馆在路口", confidence=0.9, importance=0.8),
            )
        # add() 本身会去重，这里造一条绕过 add 的重复条目
        duplicate = second.with_changes(id="MEM_duplicate", content="常去的咖啡馆在路口")
        await self.store.save(duplicate)
        archived = await self.store.consolidate(now=NOW)
        self.assertIn(duplicate.id, [item.id for item in archived])
        survivor = await self.store.get(first.id)
        assert survivor is not None
        self.assertEqual(await self.store.count(statuses=[MemoryStatus.ACTIVE]), 1)
        self.assertTrue(await self.store.relations_of(first.id))

    async def test_reactivation_throttled(self) -> None:
        config = AppConfig().memory
        config.max_reactivations = 1
        config.reactivation_cooldown_seconds = 3600
        memory = await self.store.add(_memory("旧爱好：拼图"))
        memory = await self.store.set_status(memory, MemoryStatus.LOW_VALUE)
        memory = await self.store.set_status(memory, MemoryStatus.SOFT_DELETED)
        memory = await self.store.set_status(memory, MemoryStatus.EXPIRED)
        restored = await reactivate(self.store, memory, config, now=NOW, reason="又提起")
        self.assertIs(restored.status, MemoryStatus.ACTIVE)
        again = await self.store.set_status(restored, MemoryStatus.LOW_VALUE)
        again = await self.store.set_status(again, MemoryStatus.SOFT_DELETED)
        again = await self.store.set_status(again, MemoryStatus.EXPIRED)
        with self.assertRaises(ValueError):
            await reactivate(self.store, again, config, now=NOW + timedelta(hours=2))


class EngineMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.db = Database(self.root / "agent.db")
        await initialize_database(self.db)
        self.workspace = ThoughtWorkspace(self.root / "thinking")
        self.workspace.ensure_layout()
        self.repo = ThoughtRepository(self.db, self.workspace, clock=lambda: NOW)
        self.config = AppConfig()
        self.config.contact.minimum_interval_seconds = 0
        self.config.contact.cooldown_seconds = 0
        self.memories = MemoryStore(self.db, clock=lambda: NOW, config=self.config.memory)
        self.topics = TopicStore(self.db, clock=lambda: NOW)
        self.diary = DiaryStore(self.db, self.workspace.diary_dir, clock=lambda: NOW)
        self.messages = MessageRepository(self.db)
        self.metrics = MetricsRepository(self.db)
        self.contact = ContactManager(
            self.db, self.config.contact, clock=lambda: NOW
        )
        self.engine = ThoughtEngine(
            self.config,
            repository=self.repo,
            memories=self.memories,
            topics=self.topics,
            diary=self.diary,
            contact=self.contact,
            messages=self.messages,
            emotion_model=EmotionModel(self.config.emotion, clock=lambda: NOW),
            metrics=self.metrics,
            clock=lambda: NOW,
        )
        self.addAsyncCleanup(self._cleanup)

    async def _cleanup(self) -> None:
        await self.db.close()
        self._tmp.cleanup()

    async def test_think_candidates_are_validated_and_linked(self) -> None:
        thought = await self.engine.ingest_user_message(
            conversation_id="conv-1",
            user_id="user-1",
            content="为什么天空是蓝的？",
            now=NOW,
        )
        assert thought is not None
        llm = MockLLMAdapter(
            [
                helpers.think(
                    thought.id,
                    insight="和瑞利散射有关",
                    disposition="KEEP",
                    memory_candidates=[
                        {
                            "type": "INFERENCE",
                            "content": "天空呈蓝色和瑞利散射有关",
                            "confidence": 0.7,
                            "importance": 0.6,
                        }
                    ],
                )
            ]
        )
        await self.engine.step(llm, conversation_id="conv-1", now=NOW)

        memories = await self.memories.list(include_all=True)
        self.assertEqual(len(memories), 3)  # EVENT + THOUGHT 结论 + 候选
        candidates = [
            memory
            for memory in memories
            if memory.source_type == "THOUGHT" and memory.type is not MemoryType.THOUGHT
        ]
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].source_id, thought.id)

        updated = await self.repo.get(thought.id)
        assert updated is not None
        self.assertIn(candidates[0].id, updated.related_memories)
        relations = await self.repo.relations_of(thought.id)
        self.assertIn(
            candidates[0].id,
            [row["to_id"] for row in relations if row["kind"] == "MEMORY"],
        )

    async def test_repeated_thinking_does_not_duplicate_memory(self) -> None:
        thought = await self.engine.ingest_user_message(
            conversation_id="conv-1",
            user_id="user-1",
            content="为什么猫喜欢纸箱？",
            now=NOW,
        )
        assert thought is not None
        candidate = {
            "type": "INFERENCE",
            "content": "猫喜欢纸箱可能是因为有安全感",
            "confidence": 0.7,
            "importance": 0.6,
        }
        first = MockLLMAdapter(
            [
                helpers.think(
                    thought.id,
                    insight="纸箱让猫有安全感",
                    disposition="KEEP",
                    memory_candidates=[candidate],
                )
            ]
        )
        await self.engine.step(first, conversation_id="conv-1", now=NOW)

        updated = await self.repo.get(thought.id)
        assert updated is not None
        queued = await self.repo.transition(
            updated, ThoughtStatus.QUEUED, reason="再想想", now=NOW
        )
        second = MockLLMAdapter(
            [
                helpers.think(
                    queued.id,
                    insight="纸箱让猫有安全感",
                    disposition="KEEP",
                    memory_candidates=[candidate],
                )
            ]
        )
        await self.engine.step(second, conversation_id="conv-1", now=NOW)
        memories = await self.memories.list(include_all=True)
        matching = [
            memory
            for memory in memories
            if memory.source_type == "THOUGHT"
            and memory.type is not MemoryType.THOUGHT
        ]
        self.assertEqual(len(matching), 1)

    async def test_reactivated_memory_awakens_thought(self) -> None:
        memory = await self.memories.add(
            _memory(
                "我喜欢在雨天散步",
                confidence=0.9,
                importance=0.8,
                now=NOW - timedelta(days=5),
            )
        )
        thought = Thought.create(
            type=ThoughtType.CURIOSITY,
            title="雨天散步",
            content="为什么有人喜欢雨天散步",
            source="USER_MESSAGE",
            status=ThoughtStatus.DRAWER,
            now=NOW - timedelta(days=4),
        )
        await self.repo.save(thought)
        await self.repo.save(
            thought.with_changes(related_memories=(memory.id,))
        )
        # 把记忆一路降档到 EXPIRED
        for status in (
            MemoryStatus.LOW_VALUE,
            MemoryStatus.SOFT_DELETED,
            MemoryStatus.EXPIRED,
        ):
            memory = await self.memories.set_status(memory, status)

        await self.engine.ingest_user_message(
            conversation_id="conv-1",
            user_id="user-1",
            content="我还是喜欢在雨天散步",
            now=NOW,
        )
        restored = await self.memories.get(memory.id)
        assert restored is not None
        self.assertIs(restored.status, MemoryStatus.ACTIVE)
        revived = await self.repo.get(thought.id)
        assert revived is not None
        self.assertIs(revived.status, ThoughtStatus.QUEUED)


if __name__ == "__main__":
    unittest.main()