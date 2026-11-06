"""Token 预算。

按天累计模型用量，花满之后调用方就不要再发请求了。

* 响应里带 ``usage`` 就记真实数字（OpenAI / DeepSeek 都放在这里）；
* 没带就按字符数估：中文按 0.7 token/字，其它按 3.5 字符/token；
* 计数落在一个小 JSON 文件里，进程重启不丢，跨天自动清零。

预算只管"花不花得起"，不关心花在哪。调用方发请求前问 :meth:`TokenBudget.allow`，
拿到响应后调 :meth:`TokenBudget.record_response` 记账。
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .adapter import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

# 中日韩统一表意文字（含扩展 A 与兼容区）。这几个区之外的符号按 ASCII 估。
_CJK_RANGES = ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF))

_CJK_TOKENS_PER_CHAR = 0.7
_OTHER_CHARS_PER_TOKEN = 3.5
# 每条消息的角色标记、分隔符之类，OpenAI 兼容接口都会算进去。
_PER_MESSAGE_OVERHEAD = 4


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return any(start <= code <= end for start, end in _CJK_RANGES)


def estimate_tokens(text: str) -> int:
    """粗估一段文本的 token 数。只用来兜底，有真实 usage 时不用它。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    other = len(text) - cjk
    return max(1, math.ceil(cjk * _CJK_TOKENS_PER_CHAR + other / _OTHER_CHARS_PER_TOKEN))


def estimate_request_tokens(request: LLMRequest) -> int:
    total = estimate_tokens(request.system_prompt)
    for message in request.messages:
        total += estimate_tokens(message.content) + _PER_MESSAGE_OVERHEAD
    return total


def _usage_from_raw(raw: Mapping[str, Any] | None) -> tuple[int, int] | None:
    """从厂商响应里取 ``(输入 token, 输出 token)``，取不到返回 None。"""
    if not isinstance(raw, Mapping):
        return None
    usage = raw.get("usage")
    if not isinstance(usage, Mapping):
        return None
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    if not isinstance(prompt, (int, float)) or not isinstance(completion, (int, float)):
        return None
    if prompt < 0 or completion < 0:
        return None
    return int(prompt), int(completion)


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    """某一天的用量快照，给日志和状态接口看。"""

    day: str
    input_tokens: int
    output_tokens: int
    calls: int
    daily_limit: int

    @property
    def used(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def remaining(self) -> int:
        if self.daily_limit <= 0:
            return -1
        return max(0, self.daily_limit - self.used)

    @property
    def exhausted(self) -> bool:
        return self.daily_limit > 0 and self.used >= self.daily_limit

    def to_dict(self) -> dict[str, Any]:
        return {
            "day": self.day,
            "used": self.used,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "calls": self.calls,
            "daily_limit": self.daily_limit,
            "remaining": self.remaining,
            "exhausted": self.exhausted,
        }


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TokenBudget:
    """按天计数的 token 配额。``daily_limit <= 0`` 表示不限制。"""

    def __init__(
        self,
        *,
        daily_limit: int = 0,
        state_path: str | Path | None = None,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self._limit = max(0, int(daily_limit))
        self._path = Path(state_path) if state_path else None
        self._clock = clock
        self._day = self._clock().date().isoformat()
        self._input = 0
        self._output = 0
        self._calls = 0
        self._warned = False
        self._load()

    # ------------------------------------------------------------------ 查询
    @property
    def enabled(self) -> bool:
        return self._limit > 0

    @property
    def daily_limit(self) -> int:
        return self._limit

    def snapshot(self) -> BudgetSnapshot:
        self._roll()
        return BudgetSnapshot(
            day=self._day,
            input_tokens=self._input,
            output_tokens=self._output,
            calls=self._calls,
            daily_limit=self._limit,
        )

    def allow(self) -> bool:
        """今天还能再发一次请求吗？"""
        self._roll()
        if self._limit <= 0:
            return True
        if self._input + self._output < self._limit:
            return True
        if not self._warned:
            self._warned = True
            logger.warning(
                "今日 token 预算已用完（已用 %d / 上限 %d），暂停 LLM 调用",
                self._input + self._output,
                self._limit,
            )
        return False

    # ------------------------------------------------------------------ 记账
    def record(self, *, input_tokens: int, output_tokens: int) -> None:
        self._roll()
        self._input += max(0, int(input_tokens))
        self._output += max(0, int(output_tokens))
        self._calls += 1
        self._save()

    def record_response(self, request: LLMRequest, response: LLMResponse) -> int:
        """按一次真实调用记账，返回这次算出来的消耗。"""
        usage = _usage_from_raw(response.raw)
        if usage is None:
            usage = (
                estimate_request_tokens(request),
                estimate_tokens(response.text),
            )
        self.record(input_tokens=usage[0], output_tokens=usage[1])
        return usage[0] + usage[1]

    def reset(self) -> None:
        """清零当天计数（测试与手动复位用）。"""
        self._input = 0
        self._output = 0
        self._calls = 0
        self._warned = False
        self._save()

    # ------------------------------------------------------------------ 内部
    def _roll(self) -> None:
        today = self._clock().date().isoformat()
        if today == self._day:
            return
        self._day = today
        self._input = 0
        self._output = 0
        self._calls = 0
        self._warned = False
        self._save()

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("token 预算文件读取失败 %s: %s", self._path, exc)
            return
        if not isinstance(data, Mapping) or data.get("day") != self._day:
            # 隔天的旧账不清零会一直卡着。
            return
        self._input = max(0, int(data.get("input_tokens") or 0))
        self._output = max(0, int(data.get("output_tokens") or 0))
        self._calls = max(0, int(data.get("calls") or 0))

    def _save(self) -> None:
        if self._path is None:
            return
        payload = {
            "day": self._day,
            "input_tokens": self._input,
            "output_tokens": self._output,
            "calls": self._calls,
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("token 预算写入失败 %s: %s", self._path, exc)


__all__ = [
    "BudgetSnapshot",
    "TokenBudget",
    "estimate_request_tokens",
    "estimate_tokens",
]