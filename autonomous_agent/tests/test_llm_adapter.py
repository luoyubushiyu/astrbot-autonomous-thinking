from __future__ import annotations

import time
import unittest

from autonomous_agent.llm.adapter import (
    LLMConnectionError,
    LLMProtocolError,
    LLMRequest,
    LLMTimeoutError,
    LLMMessage,
    MockLLMAdapter,
    OpenAICompatibleAdapter,
)


def make_request() -> LLMRequest:
    return LLMRequest(
        system_prompt="系统提示",
        messages=[LLMMessage(role="user", content="在吗")],
    )


class MockAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_queued_responses_in_order(self) -> None:
        adapter = MockLLMAdapter(responses=["第一个", "第二个"])
        self.assertEqual((await adapter.complete(make_request())).text, "第一个")
        self.assertEqual((await adapter.complete(make_request())).text, "第二个")

    async def test_exhausted_queue_uses_default(self) -> None:
        adapter = MockLLMAdapter(responses=["唯一"])
        await adapter.complete(make_request())
        text = (await adapter.complete(make_request())).text
        self.assertIn("DO_NOTHING", text)
        self.assertEqual(adapter.remaining, 0)
        self.assertEqual(len(adapter.calls), 2)

    async def test_exception_response_is_raised(self) -> None:
        adapter = MockLLMAdapter(responses=[LLMTimeoutError("boom")])
        with self.assertRaises(LLMTimeoutError):
            await adapter.complete(make_request())

    async def test_none_response_is_empty_text(self) -> None:
        adapter = MockLLMAdapter(responses=[None])
        self.assertEqual((await adapter.complete(make_request())).text, "")

    async def test_callable_response(self) -> None:
        adapter = MockLLMAdapter(responses=[lambda request: f"echo:{request.messages[0].content}"])
        self.assertEqual(
            (await adapter.complete(make_request())).text, "echo:在吗"
        )


class OpenAIAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.adapter = OpenAICompatibleAdapter(
            base_url="http://example.invalid/v1", api_key="k", model="cfg-model"
        )

    async def test_parses_string_content(self) -> None:
        self.adapter._post_json = lambda payload: {  # type: ignore[method-assign]
            "model": "real-model",
            "choices": [{"message": {"content": "ok"}}],
        }
        response = await self.adapter.complete(make_request())
        self.assertEqual(response.text, "ok")
        self.assertEqual(response.model, "real-model")
        self.assertEqual(response.provider, "openai_compatible")

    async def test_parses_segmented_content(self) -> None:
        self.adapter._post_json = lambda payload: {  # type: ignore[method-assign]
            "choices": [{"message": {"content": [{"text": "a"}, {"text": "b"}]}}]
        }
        self.assertEqual((await self.adapter.complete(make_request())).text, "ab")

    async def test_missing_model_falls_back_to_config(self) -> None:
        self.adapter._post_json = lambda payload: {  # type: ignore[method-assign]
            "choices": [{"message": {"content": "ok"}}]
        }
        response = await self.adapter.complete(make_request())
        self.assertEqual(response.model, "cfg-model")

    async def test_bad_payloads_raise_protocol_error(self) -> None:
        bad_payloads = (
            {"choices": []},
            {"choices": "nope"},
            {"choices": [{}]},
            {"choices": [{"message": {"content": 1}}]},
        )
        for payload in bad_payloads:
            self.adapter._post_json = lambda _p, _r=payload: _r  # type: ignore[method-assign]
            with self.assertRaises(LLMProtocolError, msg=str(payload)):
                await self.adapter.complete(make_request())

    async def test_timeout(self) -> None:
        self.adapter._timeout = 0.05

        def slow(_payload: object) -> dict:
            time.sleep(0.3)
            return {"choices": [{"message": {"content": "late"}}]}

        self.adapter._post_json = slow  # type: ignore[method-assign]
        with self.assertRaises(LLMTimeoutError):
            await self.adapter.complete(make_request())

    async def test_missing_api_key(self) -> None:
        adapter = OpenAICompatibleAdapter(
            base_url="http://example.invalid/v1", api_key="", model="m"
        )
        with self.assertRaises(LLMConnectionError):
            await adapter.complete(make_request())

    async def test_json_mode_requests_json_object(self) -> None:
        seen: list[dict] = []

        def fake(payload: dict) -> dict:
            seen.append(dict(payload))
            return {"choices": [{"message": {"content": "{}"}}]}

        self.adapter._post_json = fake  # type: ignore[method-assign]
        await self.adapter.complete(make_request())
        self.assertEqual(seen[0]["response_format"], {"type": "json_object"})

    async def test_json_mode_can_be_disabled(self) -> None:
        seen: list[dict] = []

        def fake(payload: dict) -> dict:
            seen.append(dict(payload))
            return {"choices": [{"message": {"content": "{}"}}]}

        adapter = OpenAICompatibleAdapter(
            base_url="http://example.invalid/v1",
            api_key="k",
            model="m",
            json_mode=False,
        )
        adapter._post_json = fake  # type: ignore[method-assign]
        await adapter.complete(make_request())
        self.assertNotIn("response_format", seen[0])

    async def test_provider_name_is_passed_through(self) -> None:
        adapter = OpenAICompatibleAdapter(
            base_url="http://example.invalid/v1",
            api_key="k",
            model="deepseek-chat",
            provider="deepseek",
        )
        adapter._post_json = lambda payload: {  # type: ignore[method-assign]
            "choices": [{"message": {"content": "{}"}}]
        }
        response = await adapter.complete(make_request())
        self.assertEqual(response.provider, "deepseek")


if __name__ == "__main__":
    unittest.main()