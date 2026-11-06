"""Thought Validator：语义层校验。

Schema Validator 只保证"形状对"；这里保证"这次思考对当前 Thought 有意义"：

* thought_id 必须是本次选中的那条；
* disposition 合法，且目标状态在白名单内；
* 新问题数量/结构受限；
* SPEAK 必须带 contact_message，非 SPEAK 不允许夹带消息。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..config import ThoughtConfig
from ..llm.schemas import ActionType, Decision
from .lifecycle import ThoughtLifecycleError, require_transition, target_status_for
from .models import Disposition, NewQuestion, Thought


class ThoughtDecisionError(ValueError):
    code = "THOUGHT_DECISION_ERROR"

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class ThoughtOutcome:
    thought_id: str
    insight: str
    known: tuple[str, ...]
    next_step: str
    uncertainty: str | None
    disposition: Disposition
    new_questions: tuple[NewQuestion, ...]
    next_check_minutes: int | None
    contact_message: str | None
    reasoning: str
    confidence: float
    target_status: Any
    memory_candidates: tuple[Mapping[str, Any], ...] = ()


class ThoughtValidator:
    def __init__(
        self,
        config: ThoughtConfig,
        *,
        max_insight_chars: int = 2000,
        max_memory_candidates: int = 3,
    ) -> None:
        self._config = config
        self._max_insight = max_insight_chars
        self._max_memory_candidates = max_memory_candidates

    def validate(self, decision: Decision, *, thought: Thought) -> ThoughtOutcome:
        if decision.action.type is not ActionType.THINK:
            raise ThoughtDecisionError(
                "NOT_THINK", f"期望 THINK，收到 {decision.action.type.value}"
            )
        params: Mapping[str, Any] = decision.action.parameters

        thought_id = params.get("thought_id")
        if thought_id != thought.id:
            raise ThoughtDecisionError(
                "INVALID_THOUGHT_ID",
                f"thought_id 不匹配: {thought_id!r} != {thought.id!r}",
            )

        insight = params.get("insight")
        if not isinstance(insight, str) or not insight.strip():
            raise ThoughtDecisionError("EMPTY_INSIGHT", "insight 不能为空")
        if len(insight) > self._max_insight:
            raise ThoughtDecisionError("INSIGHT_TOO_LONG", "insight 过长")

        uncertainty_raw = params.get("uncertainty")
        uncertainty: str | None = None
        if uncertainty_raw is not None:
            if not isinstance(uncertainty_raw, str):
                raise ThoughtDecisionError("INVALID_UNCERTAINTY", "uncertainty 必须是字符串")
            uncertainty = uncertainty_raw.strip() or None

        known = self._parse_strings(params.get("known"), "INVALID_KNOWN", max_items=5, max_len=200)

        next_step_raw = params.get("next_step")
        if next_step_raw is not None and not isinstance(next_step_raw, str):
            raise ThoughtDecisionError("INVALID_NEXT_STEP", "next_step 必须是字符串")
        next_step = (next_step_raw or "").strip()[:300]

        try:
            disposition = Disposition.parse(params.get("disposition"))
        except ValueError as exc:
            raise ThoughtDecisionError("INVALID_DISPOSITION", str(exc)) from exc

        target = target_status_for(disposition)
        try:
            require_transition(thought.status, target)
        except ThoughtLifecycleError as exc:
            raise ThoughtDecisionError("INVALID_TRANSITION", str(exc)) from exc

        questions = self._parse_questions(params.get("new_questions"))

        minutes_raw = params.get("next_check_minutes")
        minutes: int | None = None
        if minutes_raw is not None:
            if isinstance(minutes_raw, bool) or not isinstance(minutes_raw, int):
                raise ThoughtDecisionError(
                    "INVALID_NEXT_CHECK", "next_check_minutes 必须是整数"
                )
            if not 5 <= minutes_raw <= 1440:
                raise ThoughtDecisionError(
                    "INVALID_NEXT_CHECK", "next_check_minutes 必须落在 5..1440"
                )
            minutes = minutes_raw

        contact_message = params.get("contact_message")
        if disposition is Disposition.SPEAK:
            if not isinstance(contact_message, str) or not contact_message.strip():
                raise ThoughtDecisionError(
                    "MISSING_CONTACT_MESSAGE", "SPEAK 必须提供 contact_message"
                )
        elif contact_message:
            raise ThoughtDecisionError(
                "UNEXPECTED_CONTACT_MESSAGE", "只有 SPEAK 才能提供 contact_message"
            )

        memory_candidates = self._parse_memory_candidates(params.get("memory_candidates"))

        return ThoughtOutcome(
            thought_id=thought.id,
            insight=insight.strip(),
            known=known,
            next_step=next_step,
            uncertainty=uncertainty,
            disposition=disposition,
            new_questions=questions,
            next_check_minutes=minutes,
            contact_message=contact_message.strip() if isinstance(contact_message, str) else None,
            reasoning=decision.reasoning,
            confidence=decision.confidence,
            target_status=target,
            memory_candidates=memory_candidates,
        )

    def _parse_memory_candidates(self, raw: Any) -> tuple[Mapping[str, Any], ...]:
        if raw is None:
            return ()
        if not isinstance(raw, (list, tuple)):
            raise ThoughtDecisionError(
                "INVALID_MEMORY_CANDIDATES", "memory_candidates 必须是数组"
            )
        if len(raw) > self._max_memory_candidates:
            raise ThoughtDecisionError(
                "TOO_MANY_MEMORY_CANDIDATES",
                f"memory_candidates 最多 {self._max_memory_candidates} 条",
            )
        candidates: list[Mapping[str, Any]] = []
        for item in raw:
            if not isinstance(item, Mapping):
                raise ThoughtDecisionError(
                    "INVALID_MEMORY_CANDIDATES", "记忆候选必须是 object"
                )
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ThoughtDecisionError(
                    "INVALID_MEMORY_CANDIDATES", "记忆候选缺少 content"
                )
            candidates.append(dict(item))
        return tuple(candidates)

    def _parse_strings(
        self, raw: Any, code: str, *, max_items: int, max_len: int
    ) -> tuple[str, ...]:
        if raw is None:
            return ()
        if not isinstance(raw, (list, tuple)):
            raise ThoughtDecisionError(code, "必须是字符串数组")
        values: list[str] = []
        for item in raw[:max_items]:
            if not isinstance(item, str):
                raise ThoughtDecisionError(code, "数组元素必须是字符串")
            text = item.strip()
            if text:
                values.append(text[:max_len])
        return tuple(values)

    def _parse_questions(self, raw: Any) -> tuple[NewQuestion, ...]:
        if raw is None:
            return ()
        if not isinstance(raw, (list, tuple)):
            raise ThoughtDecisionError("INVALID_QUESTIONS", "new_questions 必须是数组")
        if len(raw) > self._config.max_new_questions:
            raise ThoughtDecisionError(
                "TOO_MANY_QUESTIONS",
                f"new_questions 最多 {self._config.max_new_questions} 条",
            )
        questions: list[NewQuestion] = []
        for item in raw:
            if not isinstance(item, Mapping):
                raise ThoughtDecisionError("INVALID_QUESTIONS", "问题必须是 object")
            title = item.get("title")
            content = item.get("content")
            if not isinstance(title, str) or not title.strip():
                raise ThoughtDecisionError("INVALID_QUESTIONS", "问题缺少 title")
            if not isinstance(content, str) or not content.strip():
                raise ThoughtDecisionError("INVALID_QUESTIONS", "问题缺少 content")
            questions.append(
                NewQuestion(
                    title=title.strip()[:200],
                    content=content.strip()[:2000],
                    importance=_number(item.get("importance"), 0.5),
                    curiosity=_number(item.get("curiosity"), 0.5),
                )
            )
        return tuple(questions)


def _number(value: Any, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return max(0.0, min(1.0, float(value)))


__all__ = ["ThoughtDecisionError", "ThoughtOutcome", "ThoughtValidator"]