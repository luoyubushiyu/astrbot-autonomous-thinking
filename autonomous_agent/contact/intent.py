"""CONTACT_INTENT：LLM 只能说「我想说这句话」，发不发由程序决定。

一个 intent 只是建议，真正发送前要过 Behavior Guard 的硬限制和
ContactManager 的频率/重复/冷却检查。这里不执行任何外部行为。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping
import secrets

from ..thought.models import Thought


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_intent_id(now: datetime | None = None) -> str:
    moment = now or utcnow()
    return f"CI_{moment.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(2)}"


@dataclass(frozen=True, slots=True)
class ContactIntent:
    """一条「值得说点什么」的意图。

    ``message_draft`` 只是建议文本，Guard/Executor 有权不发送或改写长度。
    """

    intent_id: str
    source_thought_id: str
    conversation_id: str
    topic: str
    reason: str
    message_draft: str
    priority: float = 0.5
    urgency: float = 0.5
    created_at: datetime = field(default_factory=utcnow)

    @classmethod
    def create(
        cls,
        *,
        source_thought_id: str,
        conversation_id: str,
        message_draft: str,
        topic: str = "",
        reason: str = "",
        priority: float = 0.5,
        urgency: float = 0.5,
        now: datetime | None = None,
    ) -> "ContactIntent":
        moment = now or utcnow()
        return cls(
            intent_id=new_intent_id(moment),
            source_thought_id=source_thought_id,
            conversation_id=conversation_id,
            topic=topic.strip(),
            reason=reason.strip(),
            message_draft=message_draft,
            priority=_clamp(priority),
            urgency=_clamp(urgency),
            created_at=moment,
        )

    @classmethod
    def from_thought(
        cls,
        thought: Thought,
        *,
        conversation_id: str,
        message_draft: str,
        topic: str = "",
        reason: str = "",
        emotion_drift: float = 0.0,
        now: datetime | None = None,
    ) -> "ContactIntent":
        urgency = max(float(thought.importance), float(thought.curiosity))
        # 情绪只抬优先级，不改变能不能发。
        priority = min(1.0, urgency + max(0.0, min(0.4, emotion_drift * 0.2)))
        topic = topic or (thought.related_topics[0] if thought.related_topics else "")
        return cls.create(
            source_thought_id=thought.id,
            conversation_id=conversation_id,
            message_draft=message_draft,
            topic=topic,
            reason=reason or thought.title,
            priority=priority,
            urgency=urgency,
            now=now,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "source_thought_id": self.source_thought_id,
            "conversation_id": self.conversation_id,
            "topic": self.topic,
            "reason": self.reason,
            "message_draft": self.message_draft,
            "priority": self.priority,
            "urgency": self.urgency,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContactIntent":
        created = data.get("created_at")
        if isinstance(created, str):
            parsed = datetime.fromisoformat(created)
            created_at = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        elif isinstance(created, datetime):
            created_at = created
        else:
            created_at = utcnow()
        return cls(
            intent_id=str(data.get("intent_id") or new_intent_id(created_at)),
            source_thought_id=str(data.get("source_thought_id") or ""),
            conversation_id=str(data.get("conversation_id") or ""),
            topic=str(data.get("topic") or ""),
            reason=str(data.get("reason") or ""),
            message_draft=str(data.get("message_draft") or ""),
            priority=_clamp(float(data.get("priority") or 0.5)),
            urgency=_clamp(float(data.get("urgency") or 0.5)),
            created_at=created_at,
        )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = ["ContactIntent", "new_intent_id", "utcnow"]