"""Emotion 基础。

情绪只影响"倾向"，不直接决定发不发消息：
- 孤独感升高 → 更容易产生"想联系"的念头；
- 兴趣/能量降低 → 减少主动行为；
- 所有数值都夹在 0..1，并随时间衰减回 baseline。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Mapping

from ..config import EmotionConfig


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


_FIELDS = ("loneliness", "curiosity", "interest", "energy", "mood", "attachment")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True, slots=True)
class EmotionState:
    loneliness: float = 0.3
    curiosity: float = 0.5
    interest: float = 0.5
    energy: float = 0.5
    mood: float = 0.5
    attachment: float = 0.3
    updated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {name: getattr(self, name) for name in _FIELDS}
        payload["updated_at"] = self.updated_at.isoformat() if self.updated_at else None
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "EmotionState":
        if not payload:
            return cls()
        kwargs: dict[str, Any] = {}
        for name in _FIELDS:
            raw = payload.get(name)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                kwargs[name] = _clamp(float(raw))
        updated = payload.get("updated_at")
        if isinstance(updated, str) and updated:
            try:
                parsed = datetime.fromisoformat(updated)
                kwargs["updated_at"] = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        return cls(**kwargs)

    def drift(self) -> float:
        """孤独 + 想聊的倾向，给思考引擎做加权。"""
        return _clamp(0.5 * self.loneliness + 0.3 * self.curiosity + 0.2 * self.interest)


class EmotionModel:
    def __init__(self, config: EmotionConfig, *, clock=utcnow) -> None:
        self._config = config
        self._clock = clock

    def initial(self) -> EmotionState:
        base = self._config.baseline
        return EmotionState(
            loneliness=base,
            curiosity=base,
            interest=base,
            energy=base,
            mood=base,
            attachment=base,
            updated_at=self._clock(),
        )

    def decay(self, state: EmotionState, *, now: datetime | None = None) -> EmotionState:
        """随时间慢慢回到 baseline；这里不做精细物理模型。"""
        if not self._config.enabled:
            return state
        moment = now or self._clock()
        base = self._config.baseline
        rate = self._config.decay
        last = state.updated_at or moment
        hours = max(0.0, (moment - last).total_seconds() / 3600.0)
        factor = min(1.0, rate * hours) if hours else 0.0
        values = {
            name: _clamp(getattr(state, name) + (base - getattr(state, name)) * factor)
            for name in _FIELDS
        }
        return replace(state, updated_at=moment, **values)

    def on_user_message(self, state: EmotionState, *, now: datetime | None = None) -> EmotionState:
        return self._change(
            state,
            {"loneliness": -0.15, "interest": 0.1, "attachment": 0.05, "mood": 0.05},
            now=now,
        )

    def on_ignored(self, state: EmotionState, *, now: datetime | None = None) -> EmotionState:
        return self._change(
            state, {"loneliness": 0.05, "interest": -0.05}, now=now
        )

    def on_contact_denied(self, state: EmotionState, *, now: datetime | None = None) -> EmotionState:
        return self._change(state, {"energy": -0.03, "mood": -0.03}, now=now)

    def on_thought_completed(
        self, state: EmotionState, *, now: datetime | None = None
    ) -> EmotionState:
        return self._change(
            state, {"interest": 0.06, "mood": 0.05, "curiosity": 0.03}, now=now
        )

    def on_thought_blocked(
        self, state: EmotionState, *, now: datetime | None = None
    ) -> EmotionState:
        return self._change(state, {"interest": -0.04, "mood": -0.03}, now=now)

    def on_new_finding(
        self, state: EmotionState, *, now: datetime | None = None
    ) -> EmotionState:
        return self._change(
            state, {"curiosity": 0.08, "interest": 0.04, "mood": 0.04}, now=now
        )

    def _change(
        self,
        state: EmotionState,
        deltas: Mapping[str, float],
        *,
        now: datetime | None = None,
    ) -> EmotionState:
        """单次事件只允许小幅变化，防止一条消息把情绪拉满。"""
        if not self._config.enabled:
            return state
        cap = self._config.max_emotion_delta_per_event
        values = {
            name: _clamp(getattr(state, name) + max(-cap, min(cap, deltas.get(name, 0.0))))
            for name in _FIELDS
        }
        return replace(state, updated_at=now or self._clock(), **values)


__all__ = ["EmotionModel", "EmotionState"]