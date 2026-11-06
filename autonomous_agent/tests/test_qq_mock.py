from __future__ import annotations

import unittest

from autonomous_agent.qq.adapter import (
    IncomingMessage,
    MockQQAdapter,
    OutgoingMessage,
    QQSendError,
    UserInfo,
    GroupInfo,
)


class MockQQTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.qq = MockQQAdapter(self_id="bot-1")
        await self.qq.start()

    async def asyncTearDown(self) -> None:
        await self.qq.stop()

    async def test_receive_pushed_text(self) -> None:
        pushed = self.qq.push_text("你好", conversation_id="c1", user_id="u1")
        received = await self.qq.receive_message()
        assert received is not None
        self.assertEqual(received.message_id, pushed.message_id)
        self.assertEqual(received.conversation_id, "c1")
        self.assertEqual(received.content, "你好")

    async def test_receive_empty(self) -> None:
        self.assertIsNone(await self.qq.receive_message())

    async def test_send_message(self) -> None:
        result = await self.qq.send_message(
            OutgoingMessage(conversation_id="c1", content="在的")
        )
        self.assertTrue(result.ok)
        self.assertIsNotNone(result.message_id)
        self.assertEqual(len(self.qq.outbox), 1)
        self.assertEqual(self.qq.outbox[0].content, "在的")

    async def test_send_failure(self) -> None:
        self.qq.send_failure = RuntimeError("network down")
        with self.assertRaises(QQSendError):
            await self.qq.send_message(
                OutgoingMessage(conversation_id="c1", content="x")
            )
        self.assertEqual(self.qq.outbox, [])

    async def test_send_empty_content(self) -> None:
        with self.assertRaises(QQSendError):
            await self.qq.send_message(
                OutgoingMessage(conversation_id="c1", content="   ")
            )

    async def test_profile_lookup(self) -> None:
        self.qq.users["u1"] = UserInfo(user_id="u1", nickname="小明")
        self.qq.groups["g1"] = GroupInfo(group_id="g1", name="测试群")
        self.assertEqual((await self.qq.get_user_info("u1")).nickname, "小明")  # type: ignore[union-attr]
        self.assertEqual((await self.qq.get_group_info("g1")).name, "测试群")  # type: ignore[union-attr]
        self.assertIsNone(await self.qq.get_user_info("nobody"))

    async def test_from_payload(self) -> None:
        message = IncomingMessage.from_payload(
            {
                "conversation_id": "c1",
                "message_id": "m1",
                "user_id": "u1",
                "content": "hi",
                "received_at": "2026-01-01T00:00:00+00:00",
                "is_group": True,
                "group_id": "g1",
            }
        )
        self.assertEqual(message.message_id, "m1")
        self.assertTrue(message.is_group)
        self.assertEqual(message.group_id, "g1")


if __name__ == "__main__":
    unittest.main()