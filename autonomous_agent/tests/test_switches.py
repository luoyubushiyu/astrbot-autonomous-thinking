"""阶段二开关：thought/memory/emotion/diary 的 enabled 必须真的生效。"""

from __future__ import annotations

import unittest
from pathlib import Path

from autonomous_agent.config import load_config
from autonomous_agent.core.agent import Agent
from autonomous_agent.core.event import Event
from autonomous_agent.llm.adapter import MockLLMAdapter
from autonomous_agent.qq.adapter import MockQQAdapter
from autonomous_agent.thought.models import Thought, ThoughtStatus, ThoughtType

from . import helpers


def _user(content: str, message_id: str = "m-1") -> Event:
    return helpers.user_message_event(conversation_id="conv-1", message_id=message_id, content=content)


class EnabledFlagsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.tmp = Path(self._tmp.name)
        self.agents: list[Agent] = []

    async def asyncTearDown(self) -> None:
        for agent in self.agents:
            if agent.running:
                await agent.stop()
        self._tmp.cleanup()

    async def _start(self, config, responses=None) -> tuple[Agent, MockQQAdapter, MockLLMAdapter]:
        qq = MockQQAdapter(self_id="bot-1")
        llm = MockLLMAdapter(responses or [], default_response=helpers.DO_NOTHING)
        agent = Agent(config, llm=llm, qq=qq)
        await agent.start(background=False)
        self.agents.append(agent)
        return agent, qq, llm

    async def test_thought_disabled_still_records_memory(self) -> None:
        config = helpers.make_config(self.tmp)
        config.thought.enabled = False
        agent, _qq, _llm = await self._start(config)

        await agent.handle_event(_user("为什么人要睡觉？"))

        self.assertEqual(await agent._thoughts.count(), 0)
        self.assertGreaterEqual(await agent._memories.count(), 1)

        tick = await agent.handle_event(Event.autonomous_tick(1))
        self.assertIsNone(tick.error)

    async def test_memory_disabled(self) -> None:
        config = helpers.make_config(self.tmp)
        config.memory.enabled = False
        agent, _qq, _llm = await self._start(config)

        await agent.handle_event(_user("为什么人要睡觉？"))

        self.assertEqual(await agent._memories.count(), 0)
        self.assertEqual(await agent._thoughts.count(), 1)

    async def test_emotion_disabled_keeps_baseline_and_metadata_clean(self) -> None:
        config = helpers.make_config(self.tmp)
        config.emotion.enabled = False
        agent, _qq, _llm = await self._start(config)

        await agent.handle_event(_user("在吗"))

        stats = await agent.stats()
        self.assertEqual(stats["emotion"]["loneliness"], config.emotion.baseline)
        snapshot = await agent._states.load()
        self.assertNotIn("emotion", snapshot.metadata or {})

    async def test_diary_disabled_skips_but_thought_settles(self) -> None:
        config = helpers.make_config(self.tmp)
        config.diary.enabled = False
        thought = Thought.create(
            type=ThoughtType.IDEA, title="写点什么", importance=0.8
        )
        agent, _qq, llm = await self._start(config)
        created, _ = await agent._thoughts.create(thought)
        await agent._thoughts.transition(created, ThoughtStatus.QUEUED)
        llm._responses.append(
            helpers.think(created.id, insight="今天适合写日记", disposition="DIARY")
        )

        result = await agent.handle_event(Event.autonomous_tick(1))

        self.assertIsNone(result.error)
        updated = await agent._thoughts.get(created.id)
        self.assertIs(updated.status, ThoughtStatus.RESOLVED)
        self.assertEqual(await agent._diary.recent(days=7), [])


class ExampleConfigTests(unittest.TestCase):
    def test_example_config_is_valid(self) -> None:
        path = Path(__file__).resolve().parents[2] / "config.example.json"
        self.assertTrue(path.is_file(), path)
        config = load_config(path)
        config.validate()
        self.assertTrue(Path(config.thought.workspace_dir).is_absolute())
        self.assertIn("THINK", config.behavior.allowed_actions)