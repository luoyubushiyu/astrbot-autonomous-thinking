"""token 预算：估算、按天清零、持久化、以及引擎侧的刹车。"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from autonomous_agent.config import AppConfig, ConfigError, load_config
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
from autonomous_agent.llm.adapter import LLMMessage, LLMRequest, LLMResponse, MockLLMAdapter
from autonomous_agent.llm.budget import (
    TokenBudget,
    estimate_request_tokens,
    estimate_tokens,
)
from autonomous_agent.memory.store import MemoryStore
from autonomous_agent.memory.topics import TopicStore
from autonomous_agent.thought.repository import ThoughtRepository
from autonomous_agent.thought.workspace import ThoughtWorkspace

NOW = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)


class Clock:
    """可以拨的时钟。"""

    def __init__(self, moment: datetime = NOW) -> None:
        self.moment = moment

    def __call__(self) -> datetime:
        return self.moment

    def advance(self, **kwargs) -> None:
        self.moment = self.moment + timedelta(**kwargs)


def make_request(text: str = "在吗") -> LLMRequest:
    return LLMRequest(
        system_prompt="系统提示",
        messages=[LLMMessage(role="user", content=text)],
    )


class EstimateTests(unittest.TestCase):
    def test_empty_text(self) -> None:
        self.assertEqual(estimate_tokens(""), 0)

    def test_chinese_costs_more_than_ascii(self) -> None:
        zh = estimate_tokens("你好世界" * 100)
        en = estimate_tokens("abcd" * 100)
        self.assertGreater(zh, en)
        self.assertGreater(zh, 0)

    def test_request_includes_system_and_messages(self) -> None:
        request = make_request("一二三四五")
        self.assertGreater(estimate_request_tokens(request), estimate_tokens("一二三四五"))


class TokenBudgetTests(unittest.TestCase):
    def test_unlimited_always_allows(self) -> None:
        budget = TokenBudget(daily_limit=0, clock=Clock())
        budget.record(input_tokens=10**9, output_tokens=10**9)
        self.assertTrue(budget.allow())
        self.assertFalse(budget.snapshot().exhausted)

    def test_blocks_after_limit(self) -> None:
        budget = TokenBudget(daily_limit=100, clock=Clock())
        self.assertTrue(budget.allow())
        budget.record(input_tokens=80, output_tokens=30)
        self.assertFalse(budget.allow())
        snapshot = budget.snapshot()
        self.assertEqual(snapshot.used, 110)
        self.assertEqual(snapshot.remaining, 0)
        self.assertTrue(snapshot.exhausted)

    def test_day_rollover_resets(self) -> None:
        clock = Clock()
        budget = TokenBudget(daily_limit=100, clock=clock)
        budget.record(input_tokens=100, output_tokens=0)
        self.assertFalse(budget.allow())
        clock.advance(days=1)
        self.assertTrue(budget.allow())
        self.assertEqual(budget.snapshot().used, 0)

    def test_persists_across_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "token_budget.json"
            first = TokenBudget(daily_limit=500, state_path=path, clock=Clock())
            first.record(input_tokens=100, output_tokens=20)
            second = TokenBudget(daily_limit=500, state_path=path, clock=Clock())
            self.assertEqual(second.snapshot().used, 120)
            self.assertEqual(second.snapshot().calls, 1)

    def test_stale_state_file_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "token_budget.json"
            path.write_text(
                json.dumps({"day": "2025-12-31", "input_tokens": 9999, "output_tokens": 0}),
                encoding="utf-8",
            )
            budget = TokenBudget(daily_limit=100, state_path=path, clock=Clock())
            self.assertEqual(budget.snapshot().used, 0)
            self.assertTrue(budget.allow())

    def test_uses_usage_from_provider(self) -> None:
        budget = TokenBudget(daily_limit=1000, clock=Clock())
        response = LLMResponse(
            text="好",
            raw={"usage": {"prompt_tokens": 700, "completion_tokens": 40}},
        )
        used = budget.record_response(make_request(), response)
        self.assertEqual(used, 740)
        self.assertEqual(budget.snapshot().output_tokens, 40)

    def test_falls_back_to_estimate(self) -> None:
        budget = TokenBudget(daily_limit=10**6, clock=Clock())
        response = LLMResponse(text="好的呀")
        used = budget.record_response(make_request("今天吃什么"), response)
        self.assertGreater(used, 0)
        self.assertEqual(budget.snapshot().calls, 1)


class ConfigTests(unittest.TestCase):
    def test_budget_section_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps({"budget": {"max_daily_tokens": 1234}}), encoding="utf-8"
            )
            config = load_config(path, env={})
            self.assertEqual(config.budget.max_daily_tokens, 1234)
            self.assertTrue(config.budget.state_file.endswith("token_budget.json"))

    def test_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text("{}", encoding="utf-8")
            config = load_config(path, env={"AGENT_MAX_DAILY_TOKENS": "8000"})
            self.assertEqual(config.budget.max_daily_tokens, 8000)

    def test_negative_limit_rejected(self) -> None:
        config = AppConfig()
        config.budget.max_daily_tokens = -1
        with self.assertRaises(ConfigError):
            config.validate()


class EngineBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.db = Database(self.root / "agent.db")
        await initialize_database(self.db)
        workspace = ThoughtWorkspace(self.root / "thinking")
        workspace.ensure_layout()
        self.config = AppConfig()
        self.config.contact.minimum_interval_seconds = 0
        self.config.contact.cooldown_seconds = 0
        self.budget = TokenBudget(daily_limit=100, clock=Clock())
        self.engine = ThoughtEngine(
            self.config,
            repository=ThoughtRepository(self.db, workspace),
            memories=MemoryStore(self.db),
            topics=TopicStore(self.db),
            diary=DiaryStore(self.db, workspace.diary_dir),
            contact=ContactManager(self.db, self.config.contact),
            messages=MessageRepository(self.db),
            emotion_model=EmotionModel(self.config.emotion),
            metrics=MetricsRepository(self.db),
            clock=Clock(),
            budget=self.budget,
        )
        self.addAsyncCleanup(self._cleanup)

    async def _cleanup(self) -> None:
        await self.db.close()
        self._tmp.cleanup()

    async def test_step_skips_when_budget_spent(self) -> None:
        self.budget.record(input_tokens=100, output_tokens=0)
        llm = MockLLMAdapter(responses=["不该被调用"])
        result = await self.engine.step(llm)
        self.assertEqual(result.kind, "nothing")
        self.assertEqual(result.skipped_reason, "budget_exhausted")
        self.assertEqual(llm.calls, [])

    async def test_has_pending_work_false_when_budget_spent(self) -> None:
        self.budget.record(input_tokens=100, output_tokens=0)
        self.assertFalse(await self.engine.has_pending_work())


class AgentBudgetTests(unittest.IsolatedAsyncioTestCase):
    """回复路径也要被预算挡住，不然一轮对话就把额度花光。"""

    async def asyncSetUp(self) -> None:
        from autonomous_agent.qq.adapter import MockQQAdapter
        from autonomous_agent.tests import helpers

        self._helpers = helpers
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.config = helpers.make_config(self.root)
        self.config.budget.max_daily_tokens = 100
        self.config.budget.state_file = str(self.root / "token_budget.json")
        self._qq = MockQQAdapter()

    async def test_reply_skipped_when_budget_spent(self) -> None:
        from autonomous_agent.core.agent import Agent

        TokenBudget(
            daily_limit=100,
            state_path=self.config.budget.state_file,
        ).record(input_tokens=100, output_tokens=0)

        llm = MockLLMAdapter(responses=[self._helpers.send_message()])
        agent = Agent(self.config, llm=llm, qq=self._qq)
        await agent.start(background=False)
        self.addAsyncCleanup(agent.stop)

        result = await agent.handle_event(self._helpers.user_message_event())
        self.assertEqual(result.error, "BUDGET_EXHAUSTED")
        self.assertEqual(llm.calls, [])


if __name__ == "__main__":
    unittest.main()