"""Memory Validator。

LLM 只能提出"想记住什么"，这里负责判断它配不配成为 Memory：
类型合法性、推测不升级、重复合并、冲突标记、证据与置信度门槛。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Mapping, Sequence

from ..config import MemoryConfig
from ..thought.similarity import normalize, similarity
from .models import Memory, MemoryType, SOURCE_TYPES, utcnow
from .store import MemoryStore

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")

_NEGATIONS = ("不", "没", "别", "非", "未必", "并不", "不太", "不是")


class MemoryValidationError(ValueError):
    """候选记忆结构不合法。"""


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    type: MemoryType
    content: str
    confidence: float = 0.5
    importance: float = 0.5
    relevance: float | None = None
    keywords: tuple[str, ...] = ()
    evidence: str = ""


@dataclass(frozen=True, slots=True)
class MemoryProposal:
    """一次保存意图的处理结果。"""

    action: str  # CREATE / UPDATE / CONFLICT / DO_NOT_SAVE
    candidate: MemoryCandidate
    reason: str
    target: Memory | None = None
    conflict_with: Memory | None = None


def text_keywords(text: str, limit: int = 8) -> tuple[str, ...]:
    tokens = [token for token in _TOKEN_RE.findall(text or "") if token.strip()]
    seen: list[str] = []
    for token in tokens:
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.append(lowered)
        if len(seen) >= limit:
            break
    return tuple(seen)


def _has_negation(text: str) -> bool:
    return any(marker in (text or "") for marker in _NEGATIONS)


def looks_conflicting(a: str, b: str, threshold: float = 0.3) -> bool:
    """主题相似但肯定/否定倾向相反，算冲突而不是重复。"""
    if similarity(a, b) < threshold:
        return False
    return _has_negation(a) != _has_negation(b)


def _clamp(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


class MemoryValidator:
    def __init__(self, config: MemoryConfig | None = None) -> None:
        self._cfg = config or MemoryConfig()

    # -- 解析 -------------------------------------------------------------- #
    def parse_candidates(
        self,
        raw: Sequence[Mapping[str, Any]] | None,
    ) -> list[MemoryCandidate]:
        if not raw:
            return []
        candidates: list[MemoryCandidate] = []
        for index, item in enumerate(raw, start=1):
            if not isinstance(item, Mapping):
                raise MemoryValidationError(f"第 {index} 条记忆候选不是对象")
            content = str(item.get("content", "")).strip()
            if not content:
                raise MemoryValidationError(f"第 {index} 条记忆候选缺少 content")
            try:
                memory_type = MemoryType.parse(item.get("type", "THOUGHT"))
            except ValueError as exc:
                raise MemoryValidationError(str(exc)) from None
            keywords = item.get("keywords") or text_keywords(content)
            candidates.append(
                MemoryCandidate(
                    type=memory_type,
                    content=content[:500],
                    confidence=_clamp(item.get("confidence", 0.5)),
                    importance=_clamp(item.get("importance", 0.5)),
                    relevance=(
                        _clamp(item["relevance"])
                        if item.get("relevance") is not None
                        else None
                    ),
                    keywords=tuple(str(k) for k in keywords if str(k).strip())[:8],
                    evidence=str(item.get("evidence", "") or ""),
                )
            )
        return candidates

    # -- 评估 -------------------------------------------------------------- #
    async def evaluate(
        self,
        candidate: MemoryCandidate,
        store: MemoryStore,
        *,
        source_type: str,
        source_id: str = "",
        now: datetime | None = None,
    ) -> MemoryProposal:
        moment = now or utcnow()
        if source_type not in SOURCE_TYPES:
            source_type = "SYSTEM_OBSERVATION"

        memory_type = candidate.type
        reason = ""
        if source_type == "THOUGHT" and memory_type is MemoryType.FACT:
            memory_type = MemoryType.INFERENCE
            reason = "来自思考的断言降级为 INFERENCE"
        if memory_type is MemoryType.INFERENCE and candidate.confidence > 0.85:
            candidate = replace(candidate, confidence=0.85)
        if len(normalize(candidate.content)) < 4:
            return MemoryProposal("DO_NOT_SAVE", candidate, "内容太短，不值得记住")
        if candidate.confidence < 0.35:
            return MemoryProposal("DO_NOT_SAVE", candidate, "置信度太低")
        if not candidate.keywords:
            candidate = replace(candidate, keywords=text_keywords(candidate.content))

        window = max(0.5, self._cfg.duplicate_threshold - 0.05)
        similar = await store.find_similar(
            candidate.content, threshold=0.3, limit=5, include_inactive=False
        )
        for existing, score in similar:
            if looks_conflicting(existing.content, candidate.content):
                return MemoryProposal(
                    "CONFLICT",
                    candidate,
                    f"与已有记忆冲突（相似度 {score:.2f}）",
                    target=existing,
                    conflict_with=existing,
                )
            if score >= window:
                return MemoryProposal(
                    "UPDATE",
                    candidate,
                    f"高度重复（相似度 {score:.2f}），合并进已有记忆",
                    target=existing,
                )

        return MemoryProposal(
            "CREATE",
            replace(candidate, type=memory_type),
            reason or "新的长期记忆",
        )

    async def evaluate_many(
        self,
        candidates: Sequence[MemoryCandidate],
        store: MemoryStore,
        *,
        source_type: str,
        source_id: str = "",
        now: datetime | None = None,
    ) -> list[MemoryProposal]:
        proposals: list[MemoryProposal] = []
        for candidate in candidates:
            proposals.append(
                await self.evaluate(
                    candidate,
                    store,
                    source_type=source_type,
                    source_id=source_id,
                    now=now,
                )
            )
        return proposals


__all__ = [
    "MemoryCandidate",
    "MemoryProposal",
    "MemoryValidationError",
    "MemoryValidator",
    "looks_conflicting",
    "text_keywords",
]