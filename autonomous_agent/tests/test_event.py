from __future__ import annotations

import unittest

from autonomous_agent.core.event import (
    Event,
    EventType,
    InvalidEventError,
    parse_event_payload,
)


class EventTests(unittest.TestCase):
    def test_user_message_factory(self) -> None:
        event = Event.user_message(
            conversation_id="c1",
            message_id="m1",
            user_id="u1",
            content="你好",
            is_group=True,
            group_id="g1",
        )
        self.assertIs(event.type, EventType.USER_MESSAGE)
        self.assertEqual(event.conversation_id(), "c1")
        self.assertEqual(event.payload["content"], "你好")
        self.assertTrue(event.payload["is_group"])
        self.assertEqual(event.source, "qq")

    def test_autonomous_tick_factory(self) -> None:
        event = Event.autonomous_tick(3)
        self.assertIs(event.type, EventType.AUTONOMOUS_TICK)
        self.assertEqual(event.payload["tick"], 3)

    def test_negative_tick_rejected(self) -> None:
        with self.assertRaises(InvalidEventError):
            Event.autonomous_tick(-1)

    def test_missing_content_rejected(self) -> None:
        with self.assertRaises(InvalidEventError):
            Event(type=EventType.USER_MESSAGE, payload={"conversation_id": "c1"})

    def test_alarm_requires_id(self) -> None:
        with self.assertRaises(InvalidEventError):
            Event(type=EventType.ALARM_TRIGGERED, payload={})

    def test_to_row(self) -> None:
        event = Event.bot_start()
        row = event.to_row()
        self.assertEqual(row["event_id"], event.id)
        self.assertIn("created_at", row)

    def test_parse_event_payload(self) -> None:
        event = Event.bot_stop()
        parsed = parse_event_payload(event.type, event.payload)
        self.assertEqual(parsed["reason"], "normal")

    def test_all_six_types_construct(self) -> None:
        events = [
            Event.user_message(
                conversation_id="c1", message_id="m1", user_id="u1", content="嗨"
            ),
            Event.autonomous_tick(1),
            Event.bot_start(),
            Event.bot_stop("test"),
            Event.alarm_triggered("alarm-1"),
            Event.contact_timeout("c1", 120),
        ]
        self.assertEqual({e.type for e in events}, set(EventType))

    def test_bot_stop_reason_type_rejected(self) -> None:
        with self.assertRaises(InvalidEventError):
            Event(type=EventType.BOT_STOP, payload={"reason": 1})

    def test_contact_timeout_factory(self) -> None:
        event = Event.contact_timeout("c1", 90)
        self.assertIs(event.type, EventType.CONTACT_TIMEOUT)
        self.assertEqual(event.conversation_id(), "c1")
        self.assertEqual(event.payload["waited_seconds"], 90.0)

    def test_contact_timeout_negative_waited_rejected(self) -> None:
        with self.assertRaises(InvalidEventError):
            Event(type=EventType.CONTACT_TIMEOUT, payload={"waited_seconds": -1})

    def test_contact_timeout_bad_conversation_rejected(self) -> None:
        with self.assertRaises(InvalidEventError):
            Event(
                type=EventType.CONTACT_TIMEOUT,
                payload={"conversation_id": 3, "waited_seconds": 1},
            )

    def test_each_type_row_roundtrip_fields(self) -> None:
        for event in (
            Event.user_message(
                conversation_id="c1", message_id="m1", user_id="u1", content="嗨"
            ),
            Event.autonomous_tick(2),
            Event.bot_start(),
            Event.bot_stop(),
            Event.alarm_triggered("a1"),
            Event.contact_timeout(),
        ):
            row = event.to_row()
            self.assertEqual(row["event_type"], event.type.value)
            self.assertTrue(row["payload"])


if __name__ == "__main__":
    unittest.main()