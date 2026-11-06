"""主动联系闸门（硬限制）。

LLM 只能表达"想联系"（CONTACT_INTENT / THINK 的 disposition），能不能真发由这里决定：
夜间、静默、频率、冷却、连续无回复、重复内容、同话题反复、同一条 Thought
反复联系，全部拦掉。这些判断不接受 LLM 的输入。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha1
from typing import Any

from ..config import ContactConfig
from ..core.timewindow import in_window as _shared_in_window
from ..database.database import Database
from ..thought.models import Thought
from ..thought.similarity import similarity


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class ContactDecision:
    allowed: bool
    code: str
    detail: str
    urgency: float = 0.5

    @classmethod
    def allow(cls, detail: str = "ok", urgency: float = 0.5) -> "ContactDecision":
        return cls(True, "CONTACT_ALLOWED", detail, urgency)

    @classmethod
    def deny(cls, code: str, detail: str) -> "ContactDecision":
        return cls(False, code, detail)


class ContactManager:
    def __init__(
        self,
        db: Database,
        config: ContactConfig,
        *,
        clock=utcnow,
    ) -> None:
        self._db = db
        self._config = config
        self._clock = clock

    # ------------------------------------------------------------------ 判断
    async def evaluate(
        self,
        thought: Thought,
        conversation_id: str,
        message: str,
        *,
        topic: str = "",
        silent: bool = False,
        now: datetime | None = None,
    ) -> ContactDecision:
        moment = now or self._clock()
        cfg = self._config
        if not cfg.enabled:
            return ContactDecision.deny("CONTACT_DISABLED", "主动联系已关闭")
        if silent:
            return ContactDecision.deny("SILENT_MODE", "当前处于静默模式")
        if any(_in_window(spec, moment) for spec in cfg.night_hours):
            return ContactDecision.deny("NIGHT_TIME", "现在是夜间模式")

        denied = await self._recent_denials(conversation_id, moment)
        if len(denied) >= cfg.anomaly_block_threshold:
            return ContactDecision.deny(
                "ANOMALY_PROTECTION", "短时间内被拦截次数异常，先停一轮"
            )

        sent = await self._recent_attempts(conversation_id, statuses=("SENT",))
        unanswered = [row for row in sent if not row.get("replied_at")]
        if len(unanswered) >= cfg.consecutive_no_reply_limit:
            return ContactDecision.deny(
                "CONSECUTIVE_NO_REPLY", "连续多次主动联系没有回复，先降低频率"
            )

        hour_ago = moment - timedelta(hours=1)
        if sum(1 for row in sent if _dt(row.get("created_at")) >= hour_ago) >= cfg.hourly_limit:
            return ContactDecision.deny("HOURLY_LIMIT", "一小时内联系次数已达上限")
        day_ago = moment - timedelta(days=1)
        if sum(1 for row in sent if _dt(row.get("created_at")) >= day_ago) >= cfg.daily_limit:
            return ContactDecision.deny("DAILY_LIMIT", "一天内联系次数已达上限")

        if sent:
            last = _dt(sent[0].get("created_at"))
            if (moment - last).total_seconds() < cfg.minimum_interval_seconds:
                return ContactDecision.deny("MIN_INTERVAL", "距上次联系时间太短")

        attempts = await self._recent_attempts(conversation_id, limit=20)
        if attempts:
            last_any = _dt(attempts[0].get("created_at"))
            if (moment - last_any).total_seconds() < cfg.cooldown_seconds:
                return ContactDecision.deny("COOLDOWN", "冷却中")

        for row in sent[: cfg.duplicate_window]:
            if similarity(message, str(row.get("content") or "")) >= cfg.duplicate_threshold:
                return ContactDecision.deny("DUPLICATE_MESSAGE", "最近说过几乎一样的话")

        if topic:
            repeated = await self._recent_topic_attempts(conversation_id, topic, moment)
            if len(repeated) >= cfg.topic_repeat_limit:
                return ContactDecision.deny("TOPIC_REPEAT", "同一话题最近说得太多了")

        thought_attempts = [
            row for row in unanswered if str(row.get("thought_id") or "") == thought.id
        ]
        if len(thought_attempts) >= cfg.max_attempts_per_thought:
            return ContactDecision.deny("THOUGHT_REPEAT", "同一条想法反复联系过太多次")

        urgency = max(thought.importance, thought.curiosity)
        return ContactDecision.allow("满足主动联系条件", urgency=urgency)

    # ------------------------------------------------------------------ 记录
    async def record(
        self,
        thought: Thought,
        conversation_id: str,
        message: str,
        *,
        status: str,
        reason: str = "",
        intent_id: str | None = None,
        topic: str | None = None,
        blocked_reason: str | None = None,
        now: datetime | None = None,
    ) -> str:
        moment = now or self._clock()
        rowid = await self._db.execute(
            """
            INSERT INTO contact_attempts
                (thought_id, conversation_id, content, urgency, importance,
                 status, reason, created_at, replied_at,
                 intent_id, topic, message_hash, reply_received,
                 response_time_seconds, blocked_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, 0, NULL, ?)
            """,
            (
                thought.id,
                conversation_id,
                message,
                max(thought.importance, thought.curiosity),
                thought.importance,
                status,
                reason,
                moment.isoformat(),
                intent_id,
                topic,
                _message_hash(message),
                blocked_reason if blocked_reason is not None else reason,
            ),
        )
        return f"CA_{rowid}"

    async def mark_replied(
        self, conversation_id: str, *, now: datetime | None = None
    ) -> int:
        moment = now or self._clock()
        return int(
            await self._db.execute(
                "UPDATE contact_attempts"
                " SET replied_at = ?, reply_received = 1,"
                " response_time_seconds = (julianday(?) - julianday(created_at)) * 86400.0"
                " WHERE conversation_id = ? AND replied_at IS NULL AND status = 'SENT'",
                (moment.isoformat(), moment.isoformat(), conversation_id),
            )
            or 0
        )

    async def unanswered_count(self, conversation_id: str | None = None) -> int:
        """还未收到回复的 SENT 数量；不传会话 id 时统计全部。"""
        if conversation_id:
            return int(
                await self._db.scalar(
                    "SELECT COUNT(*) FROM contact_attempts"
                    " WHERE status = 'SENT' AND replied_at IS NULL"
                    " AND conversation_id = ?",
                    (conversation_id,),
                )
                or 0
            )
        return int(
            await self._db.scalar(
                "SELECT COUNT(*) FROM contact_attempts"
                " WHERE status = 'SENT' AND replied_at IS NULL"
            )
            or 0
        )

    async def due_timeout(
        self, now: datetime, timeout_seconds: float
    ) -> tuple[str, str | None, float] | None:
        """最早一条还没被回复的 SENT，等够 ``timeout_seconds`` 就返回。"""
        if timeout_seconds <= 0:
            return None
        row = await self._db.fetch_one(
            "SELECT id, conversation_id, created_at FROM contact_attempts"
            " WHERE status = 'SENT' AND replied_at IS NULL"
            " ORDER BY created_at ASC LIMIT 1"
        )
        if row is None:
            return None
        created = _dt(row.get("created_at"))
        if created is None:
            return None
        waited = (now - created).total_seconds()
        if waited < timeout_seconds:
            return None
        conversation_id = row.get("conversation_id")
        return (
            f"CA_{row['id']}",
            str(conversation_id) if conversation_id else None,
            waited,
        )

    async def attempts(self, *, conversation_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        if conversation_id:
            return await self._db.fetch_all(
                "SELECT * FROM contact_attempts WHERE conversation_id = ?"
                " ORDER BY created_at DESC LIMIT ?",
                (conversation_id, int(limit)),
            )
        return await self._db.fetch_all(
            "SELECT * FROM contact_attempts ORDER BY created_at DESC LIMIT ?", (int(limit),)
        )

    async def _recent_attempts(
        self,
        conversation_id: str,
        *,
        statuses: tuple[str, ...] | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        params: list[Any] = [conversation_id]
        sql = "SELECT * FROM contact_attempts WHERE conversation_id = ?"
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            sql += f" AND status IN ({placeholders})"
            params.extend(statuses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        return await self._db.fetch_all(sql, tuple(params))

    async def _recent_denials(
        self, conversation_id: str, moment: datetime
    ) -> list[dict[str, Any]]:
        since = moment - timedelta(minutes=self._config.anomaly_window_minutes)
        return await self._db.fetch_all(
            "SELECT id FROM contact_attempts WHERE conversation_id = ?"
            " AND status = 'DENIED' AND created_at >= ?"
            " ORDER BY created_at DESC LIMIT ?",
            (conversation_id, since.isoformat(), int(self._config.anomaly_block_threshold)),
        )

    async def _recent_topic_attempts(
        self, conversation_id: str, topic: str, moment: datetime
    ) -> list[dict[str, Any]]:
        since = moment - timedelta(minutes=self._config.topic_repeat_window_minutes)
        return await self._db.fetch_all(
            "SELECT id FROM contact_attempts WHERE conversation_id = ?"
            " AND status = 'SENT' AND topic = ? AND created_at >= ?"
            " ORDER BY created_at DESC LIMIT ?",
            (
                conversation_id,
                topic,
                since.isoformat(),
                int(self._config.topic_repeat_limit),
            ),
        )


def _message_hash(message: str) -> str:
    return sha1(message.strip().encode("utf-8")).hexdigest()


def _dt(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _in_window(spec: str | None, now: datetime) -> bool:
    return _shared_in_window(spec, now)


__all__ = ["ContactDecision", "ContactManager", "utcnow"]