from __future__ import annotations

import unittest

from autonomous_agent.llm.schemas import ActionType
from autonomous_agent.llm.validator import (
    EmptyResponseError,
    InvalidParameterError,
    JSONParseError,
    OutputValidator,
    SchemaValidationError,
    UnknownActionError,
    UnknownIdError,
    ValidationContext,
)

CTX = ValidationContext(
    event_type="USER_MESSAGE",
    conversation_id="conv-1",
    known_conversation_ids=frozenset({"conv-1"}),
    max_message_chars=50,
)


def payload(action: str, parameters: dict | None = None) -> str:
    import json

    return json.dumps(
        {
            "reasoning": "因为所以",
            "action": {"type": action, "parameters": parameters or {}},
            "confidence": 0.5,
        },
        ensure_ascii=False,
    )


class ValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = OutputValidator()

    def test_valid_send_message(self) -> None:
        decision = self.validator.validate(
            payload(
                "SEND_MESSAGE", {"conversation_id": "conv-1", "content": "好的"}
            ),
            CTX,
        )
        self.assertIs(decision.action.type, ActionType.SEND_MESSAGE)
        self.assertEqual(decision.action.parameters["content"], "好的")

    def test_code_fence(self) -> None:
        raw = "```json\n" + payload("DO_NOTHING") + "\n```"
        decision = self.validator.validate(raw, CTX)
        self.assertIs(decision.action.type, ActionType.DO_NOTHING)

    def test_empty_response(self) -> None:
        with self.assertRaises(EmptyResponseError):
            self.validator.validate("   ", CTX)

    def test_bad_json(self) -> None:
        with self.assertRaises(JSONParseError):
            self.validator.validate("not-json", CTX)

    def test_missing_reasoning(self) -> None:
        with self.assertRaises(SchemaValidationError):
            self.validator.validate('{"action":{"type":"DO_NOTHING"}}', CTX)

    def test_unknown_action(self) -> None:
        with self.assertRaises(UnknownActionError):
            self.validator.validate(payload("DELETE_DATABASE"), CTX)

    def test_unknown_top_level_field(self) -> None:
        raw = (
            '{"reasoning":"x","action":{"type":"DO_NOTHING"},'
            '"confidence":0.1,"shell":"rm -rf"}'
        )
        with self.assertRaises(SchemaValidationError):
            self.validator.validate(raw, CTX)

    def test_confidence_out_of_range(self) -> None:
        raw = '{"reasoning":"x","action":{"type":"DO_NOTHING"},"confidence":2}'
        with self.assertRaises(SchemaValidationError):
            self.validator.validate(raw, CTX)

    def test_missing_send_target(self) -> None:
        with self.assertRaises(InvalidParameterError):
            self.validator.validate(
                payload("SEND_MESSAGE", {"content": "hi"}), CTX
            )

    def test_content_too_long(self) -> None:
        with self.assertRaises(InvalidParameterError):
            self.validator.validate(
                payload(
                    "SEND_MESSAGE",
                    {"conversation_id": "conv-1", "content": "长" * 100},
                ),
                CTX,
            )

    def test_unknown_conversation_id(self) -> None:
        with self.assertRaises(UnknownIdError):
            self.validator.validate(
                payload(
                    "SEND_MESSAGE",
                    {"conversation_id": "conv-999", "content": "hi"},
                ),
                CTX,
            )

    def test_wait_range(self) -> None:
        with self.assertRaises(InvalidParameterError):
            self.validator.validate(payload("WAIT", {"seconds": 0}), CTX)

    def test_extra_parameter_rejected(self) -> None:
        with self.assertRaises(InvalidParameterError):
            self.validator.validate(
                payload("WAIT", {"seconds": 5, "sql": "DROP TABLE users"}), CTX
            )

    def test_safe_validate_falls_back(self) -> None:
        decision, error = self.validator.safe_validate("nope", CTX)
        self.assertIs(decision.action.type, ActionType.DO_NOTHING)
        self.assertIsInstance(error, JSONParseError)

    def test_safe_validate_ok_has_no_error(self) -> None:
        decision, error = self.validator.safe_validate(payload("DO_NOTHING"), CTX)
        self.assertIs(decision.action.type, ActionType.DO_NOTHING)
        self.assertIsNone(error)


if __name__ == "__main__":
    unittest.main()