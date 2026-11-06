"""LLM 适配层。

Agent 核心只依赖 :class:`LLMAdapter` 抽象，不关心具体厂商。提供：

* ``MockLLMAdapter`` —— 测试与离线运行用，可控地返回任意文本/异常/延迟；
* ``OpenAICompatibleAdapter`` —— 对接 DeepSeek / OpenAI / 任何兼容
  ``/chat/completions`` 的服务，仅用标准库 urllib，不引入额外依赖。

约定的契约：模型必须返回一段 JSON 文本，解析与校验由 ``llm.validator`` 负责。
所有异常都收敛到 ``LLMError`` 子类，Agent 只需兜底成 DO_NOTHING。
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Any, Awaitable, Callable, Mapping

from ..config import LLMConfig
from ..monitoring import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class LLMMessage:
    role: str  # system | user | assistant
    content: str


@dataclass(frozen=True, slots=True)
class LLMRequest:
    system_prompt: str
    messages: tuple[LLMMessage, ...]
    temperature: float | None = None
    max_tokens: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_payload(self, model: str, default_temperature: float, default_max_tokens: int) -> dict[str, Any]:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend({"role": m.role, "content": m.content} for m in self.messages)
        return {
            "model": model,
            "messages": messages,
            "temperature": (
                self.temperature if self.temperature is not None else default_temperature
            ),
            "max_tokens": (
                self.max_tokens if self.max_tokens is not None else default_max_tokens
            ),
            "stream": False,
        }


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model: str = ""
    provider: str = ""
    elapsed_seconds: float = 0.0
    raw: Mapping[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 异常
# --------------------------------------------------------------------------- #
class LLMError(RuntimeError):
    """LLM 调用失败基类。"""


class LLMTimeoutError(LLMError):
    """调用超时。"""


class LLMConnectionError(LLMError):
    """网络不可达 / HTTP 错误。"""


class LLMProtocolError(LLMError):
    """返回体不符合接口协议。"""


# --------------------------------------------------------------------------- #
# 接口
# --------------------------------------------------------------------------- #
class LLMAdapter(ABC):
    """统一接口。新增厂商只需实现 ``complete``。"""

    provider: str = "abstract"
    model: str = ""

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """返回结构化文本。失败必须抛 ``LLMError`` 子类。"""

    async def aclose(self) -> None:  # pragma: no cover - 默认无需清理
        return None


# --------------------------------------------------------------------------- #
# Mock
# --------------------------------------------------------------------------- #
def default_do_nothing_response() -> str:
    return (
        '{"reasoning":"没有需要执行的动作","action":{"type":"DO_NOTHING",'
        '"parameters":{"reason":"default"}},"confidence":0.0}'
    )


MockReply = str | Exception | Callable[[LLMRequest], Awaitable[str] | str] | None


class MockLLMAdapter(LLMAdapter):
    """测试用适配器。

    ``responses`` 里的元素按顺序消费：

    * ``str``  —— 直接作为模型输出返回（空串也算，用于测空内容）；
    * ``Exception`` —— 抛出（用于测超时/网络错误）；
    * ``callable`` —— 同步或异步，返回值当输出；
    * ``None`` —— 返回空字符串。

    队列耗尽后重复 ``default_response``。
    """

    provider = "mock"

    def __init__(
        self,
        responses: Sequence[MockReply] | None = None,
        *,
        model: str = "mock-model",
        delay_seconds: float = 0.0,
        default_response: str | None = None,
    ) -> None:
        self.model = model
        self._responses: list[MockReply] = list(responses or [])
        self._delay = delay_seconds
        self._default = (
            default_response if default_response is not None else default_do_nothing_response()
        )
        self.calls: list[LLMRequest] = []

    @property
    def remaining(self) -> int:
        return len(self._responses)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        started = time.monotonic()
        if self._delay:
            await asyncio.sleep(self._delay)

        reply: MockReply = self._responses.pop(0) if self._responses else self._default
        if isinstance(reply, Exception):
            raise reply
        if reply is None:
            text = ""
        elif callable(reply):
            outcome = reply(request)
            text = await outcome if asyncio.iscoroutine(outcome) else outcome
        else:
            text = reply

        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.provider,
            elapsed_seconds=time.monotonic() - started,
        )


# --------------------------------------------------------------------------- #
# OpenAI 兼容
# --------------------------------------------------------------------------- #
class OpenAICompatibleAdapter(LLMAdapter):
    """兼容 ``POST {base_url}/chat/completions`` 的实现。

    DeepSeek 的 base_url 用 ``https://api.deepseek.com/v1``，
    OpenAI 用 ``https://api.openai.com/v1``。
    """

    provider = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 30.0,
        temperature: float = 0.7,
        max_output_tokens: int = 512,
        json_mode: bool = True,
        provider: str = "openai_compatible",
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.model = model
        self.provider = provider
        self._timeout = timeout_seconds
        self._temperature = temperature
        self._max_tokens = max_output_tokens
        self._json_mode = json_mode
        self._extra_headers = dict(extra_headers or {})

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if not self._api_key:
            raise LLMConnectionError("未配置 API Key（llm.api_key / AGENT_LLM_API_KEY）")

        payload = request.to_payload(
            model=self.model,
            default_temperature=self._temperature,
            default_max_tokens=self._max_tokens,
        )
        if self._json_mode:
            # OpenAI / DeepSeek 都支持这个字段；不支持的兼容层可以关掉。
            payload["response_format"] = {"type": "json_object"}
        started = time.monotonic()
        try:
            data = await asyncio.wait_for(
                asyncio.to_thread(self._post_json, payload), timeout=self._timeout
            )
        except asyncio.TimeoutError as exc:
            raise LLMTimeoutError(f"LLM 请求超时（{self._timeout}s）") from exc
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - 统一收敛
            raise LLMConnectionError(f"LLM 请求失败: {exc}") from exc

        return LLMResponse(
            text=self._extract_text(data),
            model=str(data.get("model") or self.model),
            provider=self.provider,
            elapsed_seconds=time.monotonic() - started,
            raw=data if isinstance(data, Mapping) else {},
        )

    # -- 内部 -------------------------------------------------------------- #
    def _post_json(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        headers.update(self._extra_headers)
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:  # noqa: BLE001 - 读错误体失败无所谓
                pass
            raise LLMConnectionError(f"HTTP {exc.code}: {detail or exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise LLMConnectionError(f"网络错误: {exc.reason}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMProtocolError(f"响应不是合法 JSON: {raw[:200]!r}") from exc
        if not isinstance(data, Mapping):
            raise LLMProtocolError("响应顶层不是 object")
        return data

    @staticmethod
    def _extract_text(data: Mapping[str, Any]) -> str:
        choices = data.get("choices")
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
            raise LLMProtocolError("响应缺少 choices")
        first = choices[0]
        message = first.get("message") if isinstance(first, Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if isinstance(content, str):
            return content
        if isinstance(content, list):  # 某些兼容层返回分段内容
            parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, Mapping) and isinstance(part.get("text"), str)
            ]
            return "".join(parts)
        raise LLMProtocolError("choices[0].message.content 不是文本")


# --------------------------------------------------------------------------- #
# 工厂
# --------------------------------------------------------------------------- #
def create_llm_adapter(config: LLMConfig) -> LLMAdapter:
    """按配置构造适配器。未知 provider 已在 config 校验阶段拦截。"""
    if config.provider in ("deepseek", "openai", "openai_compatible"):
        return OpenAICompatibleAdapter(
            base_url=config.base_url,
            api_key=config.api_key,
            model=config.model,
            timeout_seconds=config.request_timeout_seconds,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            json_mode=config.json_mode,
            provider=config.provider,
            extra_headers=config.extra_headers,
        )
    if config.provider == "mock":
        logger.warning("使用 Mock LLM Adapter，不会产生真实推理结果")
        return MockLLMAdapter()
    raise LLMConnectionError(f"不支持的 LLM provider: {config.provider}")