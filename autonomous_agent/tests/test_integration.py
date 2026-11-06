"""阶段一～六的端到端冒烟测试：真的走一遍流水线，不 mock 内部模块。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autonomous_agent.core.agent import Agent
from autonomous_agent.core.event import Event
from autonomous_agent.core.state import AgentState, utcnow
from autonomous_agent.llm.adapter import MockLLMAdapter
from autonomous_agent.qq.adapter import MockQQAdapter
from autonomous_agent.thought.models import Thought, ThoughtType

from . import helpers


def speak(request: object) -> str:
    thought_id = getattr(request, "metadata", {}).get("thought_id", "")
    return helpers.think(
        str(thought_id),
        disposition="SPEAK",
        insight="想把这件事说给她听",
        contact_message="上次说的那家店，我查到了。",
    )


class FullPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.config = helpers.make_config(self.root)
        self.config.contact.consecutive_no_reply_limit = 3
        self.config.behavior.min_seconds_between_messages = 0
        self.thought = Thought.create(
            type=ThoughtType.CURIOSITY,
            title="那家店叫什么",
            content="上次聊到的那家店，名字忘了。",
            source="test",
            importance=0.9,
            curiosity=0.9,
            related_topics=("店",),
        ).with_changes(metadata={"conversation_id": "conv-1"})

    async def asyncTearDown(self) -> None:
        if getattr(self, "agent", None) is not None:
            await self.agent.stop()
        self._tmp.cleanup()

    async def _start(self, responses: list[object]) -> Agent:
        self.llm = MockLLMAdapter(responses=responses)  # type: ignore[arg-type]
        self.qq = MockQQAdapter()
        self.agent = Agent(self.config, llm=self.llm, qq=self.qq, clock=utcnow)
        await self.agent.start(background=False)
        await self.agent._thoughts.create(self.thought)
        return self.agent

    async def test_user_message_then_contact_then_reply(self) -> None:
        agent = await self._start(
            [
                helpers.send_message(content="在的呀"),
                speak,
                helpers.send_message(content="嗯，说嘛"),
            ]
        )

        first = await agent.handle_event(helpers.user_message_event())
        self.assertIsNone(first.error)
        self.assertEqual(len(self.qq.outbox), 1)
        self.assertIn("在的", self.qq.outbox[0].content)

        # 消息、记忆、话题、情绪都已经落库（阶段一/四/五）。
        messages = await agent._messages.recent("conv-1", limit=10)
        self.assertTrue(any(m.content == "在吗" for m in messages))
        self.assertGreater(await agent._memories.count(), 0)
        self.assertGreater(agent._engine.emotion.interest, 0.3)

        # 自主 tick：选中念头 → THINK(SPEAK) → 联系意图 → 护栏/限制 → QQ。
        tick = await agent.handle_event(Event.autonomous_tick(2))
        self.assertIsNone(tick.error)
        self.assertEqual(len(self.qq.outbox), 2)
        self.assertIn("那家店", self.qq.outbox[1].content)

        rows = await agent._contact.attempts()
        sent = [row for row in rows if row["status"] == "SENT"]
        self.assertEqual(len(sent), 1)
        self.assertTrue(sent[0]["intent_id"])
        self.assertEqual(sent[0]["topic"], "店")
        self.assertTrue(sent[0]["message_hash"])
        self.assertEqual(sent[0]["reply_received"], 0)

        # 用户回复：记 reply_received 与响应时间，反馈回情绪。
        second = await agent.handle_event(
            Event.user_message(
                conversation_id="conv-1", message_id="m-2", user_id="user-1", content="说嘛"
            )
        )
        self.assertIsNone(second.error)
        rows = await agent._contact.attempts()
        replied = [row for row in rows if row["status"] == "SENT"][0]
        self.assertEqual(replied["reply_received"], 1)
        self.assertIsNotNone(replied["response_time_seconds"])
        self.assertGreaterEqual(replied["response_time_seconds"], 0.0)
        self.assertEqual(await agent._contact.unanswered_count(), 0)

        # 重启：数据库/Thought 文件/联系记录都能恢复，状态不倒退。
        await agent.stop()
        self.agent = None
        self.llm = MockLLMAdapter(responses=[helpers.DO_NOTHING])
        self.qq = MockQQAdapter()
        agent = Agent(self.config, llm=self.llm, qq=self.qq, clock=utcnow)
        self.agent = agent
        await agent.start(background=False)
        self.assertIs(agent.state, AgentState.IDLE)
        self.assertEqual(await agent._contact.unanswered_count(), 0)
        restored = [r for r in await agent._contact.attempts() if r["status"] == "SENT"]
        self.assertEqual(len(restored), 1)

    async def test_consecutive_no_reply_enters_silent_and_keeps_thinking(self) -> None:
        self.config.contact.consecutive_no_reply_limit = 2
        self.config.contact.silent_after_consecutive = 2
        self.config.contact.minimum_interval_seconds = 0
        agent = await self._start([speak])

        # 先攒两条没人回复的 SENT（模拟之前几轮主动联系都石沉大海）。
        for _ in range(2):
            await agent._contact.record(
                self.thought, "conv-1", "在吗", status="SENT", reason="test"
            )
        self.assertEqual(await agent._contact.unanswered_count("conv-1"), 2)

        # 下一个 tick 先判连续未回复，直接进 SILENT，不再联系。
        result = await agent.handle_event(Event.autonomous_tick(2))
        self.assertIs(result.state, AgentState.SILENT)
        self.assertIsNotNone(agent._machine.metadata.get("silent_until"))
        self.assertEqual(self.qq.outbox, [])

        # SILENT 里照样思考，只是联系意图被拦成 DENIED(SILENT_MODE)。
        calls_before = len(self.llm.calls)
        await agent.handle_event(Event.autonomous_tick(3))
        self.assertGreater(len(self.llm.calls), calls_before)
        self.assertIs(agent.state, AgentState.SILENT)
        rows = await agent._contact.attempts()
        denied = [r for r in rows if r["status"] == "DENIED"]
        self.assertTrue(denied)
        self.assertEqual(denied[0]["blocked_reason"], "SILENT_MODE")


if __name__ == "__main__":
    unittest.main()