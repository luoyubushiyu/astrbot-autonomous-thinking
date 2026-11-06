"""Memory / Topic / Diary 测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autonomous_agent.config import AppConfig
from autonomous_agent.database.database import Database
from autonomous_agent.database.migrations import initialize_database
from autonomous_agent.diary.repository import DiaryStore
from autonomous_agent.memory.models import Memory, MemoryType
from autonomous_agent.memory.store import MemoryLimitError, MemoryStore
from autonomous_agent.memory.topics import TopicStore


class MemoryStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.db = Database(self.root / "agent.db")
        await initialize_database(self.db)
        self.store = MemoryStore(
            self.db, max_memories=5, duplicate_threshold=0.85
        )
        self.addAsyncCleanup(self._cleanup)

    async def _cleanup(self) -> None:
        await self.db.close()
        self._tmp.cleanup()

    async def test_add_and_get(self) -> None:
        memory = Memory.create(
            type=MemoryType.FACT, content="用户喜欢猫", confidence=0.8
        )
        saved = await self.store.add(memory)
        loaded = await self.store.get(saved.id)
        assert loaded is not None
        self.assertEqual(loaded.content, "用户喜欢猫")
        self.assertIs(loaded.type, MemoryType.FACT)

    async def test_duplicate_merges_and_keeps_type(self) -> None:
        first = Memory.create(
            type=MemoryType.INFERENCE, content="用户可能喜欢猫", confidence=0.4
        )
        await self.store.add(first)
        second = Memory.create(
            type=MemoryType.FACT, content="用户可能喜欢猫", confidence=0.9
        )
        merged = await self.store.add(second)
        self.assertEqual(merged.id, first.id)
        self.assertIs(merged.type, MemoryType.INFERENCE)
        self.assertAlmostEqual(merged.confidence, 0.9)
        self.assertEqual(await self.store.count(), 1)

    async def test_limit_raises(self) -> None:
        for index in range(5):
            await self.store.add(
                Memory.create(type=MemoryType.FACT, content=f"独立的记忆 {index}")
            )
        with self.assertRaises(MemoryLimitError):
            await self.store.add(
                Memory.create(type=MemoryType.FACT, content="完全不同的新内容")
            )

    async def test_search_ranks_by_similarity(self) -> None:
        await self.store.add(
            Memory.create(type=MemoryType.PREFERENCE, content="用户喜欢喝咖啡")
        )
        await self.store.add(
            Memory.create(type=MemoryType.EVENT, content="今天下雨了")
        )
        results = await self.store.search("咖啡")
        self.assertEqual(results[0].content, "用户喜欢喝咖啡")


class TopicStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self._tmp.name) / "agent.db")
        await initialize_database(self.db)
        self.topics = TopicStore(self.db)
        self.addAsyncCleanup(self._cleanup)

    async def _cleanup(self) -> None:
        await self.db.close()
        self._tmp.cleanup()

    async def test_touch_creates_and_merges(self) -> None:
        first = await self.topics.touch("猫", keywords=["橘猫"], importance=0.4)
        second = await self.topics.touch("猫", keywords=["三花"], importance=0.8)
        self.assertEqual(first.id, second.id)
        self.assertEqual(set(second.keywords), {"橘猫", "三花"})
        self.assertAlmostEqual(second.importance, 0.8)
        self.assertEqual(len(await self.topics.all()), 1)


class DiaryStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.db = Database(self.root / "agent.db")
        await initialize_database(self.db)
        self.diary = DiaryStore(self.db, self.root / "thinking" / "diary")
        self.addAsyncCleanup(self._cleanup)

    async def _cleanup(self) -> None:
        await self.db.close()
        self._tmp.cleanup()

    async def test_append_writes_file_and_db(self) -> None:
        day = await self.diary.append("今天想通了一件事")
        content = await self.diary.read(day)
        assert content is not None
        self.assertIn("今天想通了一件事", content)
        path = self.root / "thinking" / "diary" / f"{day}.md"
        self.assertTrue(path.is_file())
        self.assertIn("今天想通了一件事", path.read_text(encoding="utf-8"))

    async def test_append_is_additive(self) -> None:
        day = await self.diary.append("第一句", now=None)
        await self.diary.append("第二句")
        content = await self.diary.read(day)
        assert content is not None
        self.assertIn("第一句", content)
        self.assertIn("第二句", content)

    async def test_empty_entry_rejected(self) -> None:
        with self.assertRaises(ValueError):
            await self.diary.append("   ")


class EmotionTests(unittest.TestCase):
    def test_user_message_moves_emotion(self) -> None:
        from autonomous_agent.core.emotion import EmotionModel
        from autonomous_agent.config import EmotionConfig

        model = EmotionModel(EmotionConfig())
        state = model.initial()
        after = model.on_user_message(state)
        self.assertLess(after.loneliness, state.loneliness)
        self.assertGreater(after.interest, state.interest)

    def test_values_stay_in_range(self) -> None:
        from autonomous_agent.core.emotion import EmotionState

        state = EmotionState.from_dict({"loneliness": 5, "energy": -3})
        self.assertEqual(state.loneliness, 1.0)
        self.assertEqual(state.energy, 0.0)


if __name__ == "__main__":
    unittest.main()