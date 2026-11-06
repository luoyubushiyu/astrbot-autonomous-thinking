"""七、QQ Adapter：接口解耦、Mock 行为、自定义适配器可插拔。"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from collections import deque
from pathlib import Path

from autonomous_agent import __file__ as package_file
from autonomous_agent.config import QQConfig
from autonomous_agent.core.agent import Agent
from autonomous_agent.llm.adapter import MockLLMAdapter
from autonomous_agent.qq.adapter import (
    GroupInfo,
    IncomingMessage,
    MockQQAdapter,
    OutgoingMessage,
    QQAdapter,
    QQAdapterError,
    QQSendError,
    SendResult,
    UserInfo,
    create_qq_adapter,
)
from autonomous_agent.tests import helpers


class EchoAdapter(QQAdapter):
    """一个完全独立的适配器：只实现接口，不继承 Mock。"""

    name = "echo"

    def __init__(self) -> None:
        self.inbox: deque[IncomingMessage] = deque()
        self.sent: list[OutgoingMessage] = []
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def receive_message(self) -> IncomingMessage | None:
        return self.inbox.popleft() if self.inbox else None

    async def send_message(self, message: OutgoingMessage) -> SendResult:
        self.sent.append(message)
        return SendResult(ok=True, message_id=f"echo-{len(self.sent)}")

    async def get_user_info(self, user_id: str) -> UserInfo | None:
        return UserInfo(user_id=user_id, nickname="echo-user")

    async def get_group_info(self, group_id: str) -> GroupInfo | None:
        return GroupInfo(group_id=group_id, name="echo-group", member_count=2)


class InterfaceTests(unittest.TestCase):
    def test_interface_cannot_be_instantiated(self) -> None:
        with self.assertRaises(TypeError):
            QQAdapter()  # type: ignore[abstract]

    def test_partial_implementation_is_rejected(self) -> None:
        class Partial(QQAdapter):
            async def receive_message(self):
                return None

        with self.assertRaises(TypeError):
            Partial()  # type: ignore[abstract]

    def test_factory_returns_mock(self) -> None:
        adapter = create_qq_adapter(QQConfig(adapter="mock", self_id="bot-9"))
        self.assertIsInstance(adapter, MockQQAdapter)
        self.assertEqual(adapter.self_id, "bot-9")

    def test_factory_rejects_unknown(self) -> None:
        with self.assertRaises(QQAdapterError):
            create_qq_adapter(QQConfig(adapter="napcat"))


class DecouplingTests(unittest.TestCase):
    def test_core_does_not_import_concrete_adapter(self) -> None:
        core = Path(package_file).parent / "core"
        for path in core.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("MockQQAdapter", text, path.name)
            self.assertNotIn("OpenAICompatibleAdapter", text, path.name)

    def test_qq_layer_does_not_import_agent(self) -> None:
        qq = Path(package_file).parent / "qq"
        for path in qq.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("core.agent", text, path.name)
            self.assertNotIn("from ..core", text, path.name)


class MockAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.adapter = MockQQAdapter()

    async def test_receive_order_and_empty(self) -> None:
        self.assertIsNone(await self.adapter.receive_message())
        first = self.adapter.push_text("一")
        second = self.adapter.push_text("二")
        self.assertEqual((await self.adapter.receive_message()).message_id, first.message_id)
        self.assertEqual((await self.adapter.receive_message()).message_id, second.message_id)
        self.assertIsNone(await self.adapter.receive_message())

    async def test_send_records_outbox(self) -> None:
        result = await self.adapter.send_message(
            OutgoingMessage(conversation_id="c1", content="你好")
        )
        self.assertTrue(result.ok)
        self.assertEqual(self.adapter.outbox[0].content, "你好")

    async def test_send_rejects_empty_content(self) -> None:
        with self.assertRaises(QQSendError):
            await self.adapter.send_message(
                OutgoingMessage(conversation_id="c1", content="   ")
            )

    async def test_send_failure_raises(self) -> None:
        self.adapter.send_failure = RuntimeError("网络断了")
        with self.assertRaises(QQSendError):
            await self.adapter.send_message(
                OutgoingMessage(conversation_id="c1", content="你好")
            )

    async def test_user_and_group_info(self) -> None:
        self.assertIsNone(await self.adapter.get_user_info("u1"))
        self.assertIsNone(await self.adapter.get_group_info("g1"))
        self.adapter.users["u1"] = UserInfo(user_id="u1", nickname="小鲸")
        self.adapter.groups["g1"] = GroupInfo(group_id="g1", name="群", member_count=3)
        self.assertEqual((await self.adapter.get_user_info("u1")).nickname, "小鲸")
        self.assertEqual((await self.adapter.get_group_info("g1")).member_count, 3)

    async def test_clear(self) -> None:
        self.adapter.push_text("x")
        await self.adapter.send_message(
            OutgoingMessage(conversation_id="c1", content="y")
        )
        self.adapter.clear()
        self.assertIsNone(await self.adapter.receive_message())
        self.assertEqual(self.adapter.outbox, [])


class PluggableAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.config = helpers.make_config(Path(self._tmp.name))

    async def test_custom_adapter_handles_user_message(self) -> None:
        adapter = EchoAdapter()
        agent = Agent(
            self.config,
            llm=MockLLMAdapter([helpers.send_message(conversation_id="conv-1")]),
            qq=adapter,
        )
        await agent.start(background=False)
        self.assertTrue(adapter.started)
        await agent.handle_event(helpers.user_message_event(conversation_id="conv-1"))
        self.assertEqual([m.content for m in adapter.sent], ["在的呀"])
        self.assertEqual((await agent.stats())["state"], "IDLE")
        await agent.stop()
        self.assertTrue(adapter.stopped)

    async def test_custom_adapter_through_poll_loop(self) -> None:
        adapter = EchoAdapter()
        self.config.qq.enabled = True
        self.config.qq.poll_interval_seconds = 0.02
        agent = Agent(
            self.config,
            llm=MockLLMAdapter(
                [helpers.send_message(conversation_id="conv-1", content="收到")]
            ),
            qq=adapter,
        )
        await agent.start(background=True)
        adapter.inbox.append(
            IncomingMessage(
                conversation_id="conv-1",
                message_id="m-poll-1",
                user_id="u1",
                content="在吗",
            )
        )
        for _ in range(100):
            if adapter.sent:
                break
            await asyncio.sleep(0.02)
        self.assertEqual([m.content for m in adapter.sent], ["收到"])
        await agent.stop()


if __name__ == "__main__":
    unittest.main()