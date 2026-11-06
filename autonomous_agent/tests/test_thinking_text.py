"""思考文本（thinking text）测试：多轮连续、写回、守卫。"""

from __future__ import annotations

import json
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
from autonomous_agent.thought.models import Disposition, Thought, ThoughtStatus, ThoughtType
from autonomous_agent.thought.repository import ThoughtRepository
from autonomous_agent.thought.thinking_text import (
    ThinkingState,
    compress_thinking_text,
    parse_thinking_text,
    render_thinking_text,
)
from autonomous_agent.thought.workspace import ThoughtWorkspace

NOW = datetime(2026, 1, 2, 10, 0, tzinfo=timezone.utc)


def think_response(
    thought_id: str,
    *,
    insight: str,
    disposition: str = "CONTINUE",
    known: list[str] | None = None,
    uncertainty: str | None = None,
    new_questions: list[dict] | None = None,
    next_step: str = "",
) -> str:
    parameters = {
        "thought_id": thought_id,
        "insight": insight,
        "disposition": disposition,
        "known": known or [],
        "new_questions": new_questions or [],
        "next_step": next_step,
    }
    if uncertainty:
        parameters["uncertainty"] = uncertainty
    return json.dumps(
        {
            "reasoning": "测试",
            "action": {"type": "THINK", "parameters": parameters},
            "confidence": 0.7,
        },
        ensure_ascii=False,
    )


class ThinkingTextTests(unittest.IsolatedAsyncioTestCase):
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

    async def _seed(self, title: str = "为什么天空是蓝的") -> Thought:
        thought, _ = await self.repo.create(
            Thought.create(
                type=ThoughtType.QUESTION,
                title=title,
                content=title,
                source="TEST",
                now=NOW,
            )
        )
        return thought

    # ---------------------------------------------------------------- 纯函数
    def test_roundtrip(self) -> None:
        thought = Thought.create(
            type=ThoughtType.QUESTION,
            title="问题",
            content="问题内容",
            now=NOW,
        )
        state = ThinkingState(
            known=["一", "二"],
            current_thinking="当前想法",
            insight="关键发现",
            uncertainty=["疑点"],
            new_questions=["子问题"],
            next_step="下一步",
            continue_condition="还有没想完的",
            stop_reason="",
            last_decision="CONTINUE",
            turns=2,
            history=["第一次思考"],
        )
        text = render_thinking_text(thought, state, now=NOW)
        self.assertIn("核心问题:", text)
        self.assertIn("当前思考:", text)
        parsed = parse_thinking_text(text)
        self.assertEqual(parsed.known, ["一", "二"])
        self.assertEqual(parsed.current_thinking, "当前想法")
        self.assertEqual(parsed.uncertainty, ["疑点"])
        self.assertEqual(parsed.history, ["第一次思考"])

    def test_compress_keeps_head(self) -> None:
        state = ThinkingState(current_thinking="x" * 500, history=["h" * 500] * 20)
        thought = Thought.create(type=ThoughtType.IDEA, title="t", content="c", now=NOW)
        text = render_thinking_text(thought, state, now=NOW)
        compressed = compress_thinking_text(text, max_chars=1200)
        self.assertLessEqual(len(compressed), len(text))
        self.assertIn("核心问题:", compressed)

    # ---------------------------------------------------------------- 连续思考
    async def test_multiround_continue_reads_and_writes_text(self) -> None:
        thought = await self._seed()
        llm = MockLLMAdapter(
            [
                think_response(
                    thought.id,
                    insight="先拆开：蓝光波长更短，更容易散射",
                    known=["天空不是自己发蓝光"],
                    next_step="查瑞利散射",
                ),
                think_response(
                    thought.id,
                    insight="确认是瑞利散射，短波被散射得更多",
                    disposition="KEEP",
                    known=["瑞利散射与波长四次方成反比"],
                    next_step="可以停在结论上",
                ),
            ]
        )

        first = await self.engine.step(llm, now=NOW)
        self.assertEqual(first.kind, "thought")
        self.assertIsNotNone(first.outcome)
        self.assertIs(first.outcome.disposition, Disposition.CONTINUE)
        self.assertIs(first.thought.status, ThoughtStatus.QUEUED)
        self.assertEqual(first.thought.next_action, "THINK_AGAIN")
        text = self.workspace.read_thinking_text(first.thought)
        self.assertIsNotNone(text)
        self.assertIn("先拆开", text)
        self.assertIn("历史摘要:", text)

        second = await self.engine.step(llm, now=NOW + timedelta(minutes=4))
        self.assertEqual(second.kind, "thought")
        self.assertIs(second.outcome.disposition, Disposition.KEEP)
        text2 = self.workspace.read_thinking_text(second.thought)
        self.assertIn("确认是瑞利散射", text2)
        self.assertIn("先拆开", text2)  # 上一轮的摘要还在
        state = parse_thinking_text(text2)
        self.assertEqual(state.turns, 2)
        self.assertIn("先拆开", " ".join(state.history))

    async def test_missing_text_is_rebuilt(self) -> None:
        thought = await self._seed()
        llm = MockLLMAdapter([think_response(thought.id, insight="新的想法")])
        await self.engine.step(llm, now=NOW)
        path = self.workspace.text_path_for(thought)
        path.unlink()
        llm2 = MockLLMAdapter([think_response(thought.id, insight="从数据库重建")])
        result = await self.engine.step(llm2, now=NOW + timedelta(minutes=4))
        self.assertEqual(result.kind, "thought")
        text = self.workspace.read_thinking_text(result.thought)
        self.assertIn("从数据库重建", text)

    # ---------------------------------------------------------------- 守卫
    async def test_depth_limit_stops_expansion(self) -> None:
        self.config.thought.max_thought_depth = 1
        parent = await self._seed("父问题")
        llm = MockLLMAdapter(
            [
                think_response(
                    parent.id,
                    insight="先想一层",
                    new_questions=[{"title": "子问题A", "content": "子问题A的内容"}],
                )
            ]
        )
        await self.engine.step(llm, now=NOW)
        children = await self.repo.children_of(parent.id)
        self.assertEqual(len(children), 1)
        child = children[0]
        llm2 = MockLLMAdapter(
            [
                think_response(
                    child.id,
                    insight="再想一层",
                    new_questions=[{"title": "孙问题", "content": "孙问题内容"}],
                )
            ]
        )
        await self.engine.step(llm2, now=NOW)
        self.assertEqual(len(await self.repo.children_of(child.id)), 0)

    async def test_similar_question_is_not_spawned(self) -> None:
        parent = await self._seed("为什么天空是蓝的")
        llm = MockLLMAdapter(
            [
                think_response(
                    parent.id,
                    insight="绕回来",
                    new_questions=[
                        {"title": "为什么天空是蓝的", "content": "为什么天空是蓝的"}
                    ],
                )
            ]
        )
        await self.engine.step(llm, now=NOW)
        self.assertEqual(len(await self.repo.children_of(parent.id)), 0)

    async def test_bad_json_keeps_text_and_requeues(self) -> None:
        thought = await self._seed()
        good = MockLLMAdapter([think_response(thought.id, insight="先记一步")])
        await self.engine.step(good, now=NOW)
        before = self.workspace.read_thinking_text(thought)

        bad = MockLLMAdapter(["not json at all"])
        result = await self.engine.step(bad, now=NOW + timedelta(minutes=4))
        self.assertEqual(result.kind, "error")
        self.assertEqual(result.error, "JSON_PARSE_ERROR")
        refreshed = await self.repo.get(thought.id)
        self.assertIs(refreshed.status, ThoughtStatus.QUEUED)
        self.assertEqual(self.workspace.read_thinking_text(refreshed), before)

    async def test_latest_conversation_is_used(self) -> None:
        await self.messages.add(
            conversation_id="conv-9",
            message_id="m-9",
            role="user",
            content="在吗",
            created_at=NOW,
        )
        thought = await self._seed()
        llm = MockLLMAdapter([think_response(thought.id, insight="用的会话")])
        result = await self.engine.step(llm, now=NOW)
        self.assertEqual(result.kind, "thought")

    async def test_text_follows_status_directory(self) -> None:
        thought = await self._seed()
        llm = MockLLMAdapter([think_response(thought.id, insight="先放一边", disposition="KEEP")])
        result = await self.engine.step(llm, now=NOW)
        active_text = self.root / "thinking" / "thinking" / "active" / f"{thought.id}.txt"
        self.assertTrue(active_text.exists())

        moved = await self.repo.transition(result.thought, ThoughtStatus.DRAWER, reason="test")
        drawer_text = self.root / "thinking" / "drawer" / f"{thought.id}.txt"
        self.assertTrue(drawer_text.exists())
        self.assertFalse(active_text.exists())
        self.assertIn("先放一边", drawer_text.read_text(encoding="utf-8"))

    async def test_resolved_disposition(self) -> None:
        thought = await self._seed()
        llm = MockLLMAdapter(
            [think_response(thought.id, insight="想清楚了", disposition="RESOLVED")]
        )
        result = await self.engine.step(llm, now=NOW)
        self.assertEqual(result.kind, "thought")
        self.assertIs(result.thought.status, ThoughtStatus.RESOLVED)


if __name__ == "__main__":
    unittest.main()