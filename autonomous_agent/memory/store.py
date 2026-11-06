"""Memory 存取。

负责落库、检索、去重合并与生命阶段管理；"该不该记"由上层（Validator）判断。
所有遗忘都是状态降级，记录永远留在库里，可恢复。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Mapping

from ..config import MemoryConfig
from ..database.database import Database
from ..thought.similarity import similarity
from .models import (
    Memory,
    MemoryStatus,
    MemoryType,
    STATUS_ORDER,
    USABLE_STATUSES,
    utcnow,
)


class MemoryLimitError(RuntimeError):
    """记忆总量超过上限。"""


class MemoryRelationError(ValueError):
    """记忆关系非法。"""


RELATION_KINDS = ("RELATED", "SUPPORTS", "CONTRADICTS", "MERGED_INTO", "REACTIVATES")


class MemoryStore:
    """长期记忆仓储。"""

    def __init__(
        self,
        db: Database,
        *,
        clock: Any = utcnow,
        max_memories: int = 2000,
        duplicate_threshold: float = 0.9,
        config: MemoryConfig | None = None,
    ) -> None:
        self._db = db
        self._clock = clock
        self._max = max_memories
        self._duplicate_threshold = duplicate_threshold
        self._config = config or MemoryConfig()

    # -- 读取 -------------------------------------------------------------- #
    async def get(self, memory_id: str) -> Memory | None:
        row = await self._db.fetch_one(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        )
        return Memory.from_row(row) if row else None

    @staticmethod
    def _status_clause(
        statuses: Iterable[MemoryStatus | str] | None,
        include_all: bool,
    ) -> tuple[str, list[Any]]:
        if statuses is not None:
            values = [MemoryStatus.parse(s).value for s in statuses]
        elif include_all:
            return "", []
        else:
            values = [status.value for status in USABLE_STATUSES]
        if not values:
            return " WHERE 1 = 0", []
        placeholders = ",".join("?" for _ in values)
        return f" WHERE status IN ({placeholders})", list(values)

    async def list(
        self,
        *,
        types: Iterable[MemoryType] | None = None,
        limit: int = 50,
        statuses: Iterable[MemoryStatus | str] | None = None,
        include_all: bool = False,
    ) -> list[Memory]:
        sql = "SELECT * FROM memories"
        params: list[Any] = []
        clause, params = self._status_clause(statuses, include_all)
        sql += clause
        if types is not None:
            values = [MemoryType.parse(t).value for t in types]
            if not values:
                return []
            placeholders = ",".join("?" for _ in values)
            sql += (" AND" if clause else " WHERE") + f" type IN ({placeholders})"
            params.extend(values)
        sql += " ORDER BY importance DESC, updated_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = await self._db.fetch_all(sql, tuple(params))
        return [Memory.from_row(row) for row in rows]

    async def recent(self, *, limit: int = 20, include_all: bool = False) -> list[Memory]:
        clause, params = self._status_clause(None, include_all)
        rows = await self._db.fetch_all(
            f"SELECT * FROM memories{clause} ORDER BY updated_at DESC LIMIT ?",
            (*params, max(1, int(limit))),
        )
        return [Memory.from_row(row) for row in rows]

    async def count(
        self,
        *,
        statuses: Iterable[MemoryStatus | str] | None = None,
        include_all: bool = True,
    ) -> int:
        clause, params = self._status_clause(statuses, include_all)
        value = await self._db.scalar(f"SELECT COUNT(*) FROM memories{clause}", tuple(params))
        return int(value or 0)

    async def count_by_status(self) -> dict[str, int]:
        rows = await self._db.fetch_all(
            "SELECT status, COUNT(*) AS n FROM memories GROUP BY status"
        )
        return {str(row["status"]): int(row["n"]) for row in rows}

    async def find_similar(
        self,
        content: str,
        *,
        threshold: float | None = None,
        limit: int = 5,
        same_type: MemoryType | str | None = None,
        include_inactive: bool = False,
    ) -> list[tuple[Memory, float]]:
        clause, params = self._status_clause(None, include_inactive)
        sql = f"SELECT * FROM memories{clause}"
        values: list[Any] = list(params)
        if same_type is not None:
            sql += (" AND" if clause else " WHERE") + " type = ?"
            values.append(MemoryType.parse(same_type).value)
        rows = await self._db.fetch_all(sql, tuple(values))
        floor = self._duplicate_threshold if threshold is None else threshold
        scored: list[tuple[Memory, float]] = []
        for row in rows:
            try:
                memory = Memory.from_row(row)
            except (ValueError, KeyError):
                continue
            score = similarity(content, memory.content)
            if score >= floor:
                scored.append((memory, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(1, int(limit))]

    async def search(self, text: str, *, limit: int = 5) -> list[Memory]:
        """按相似度 + 重要度取相关记忆，只给 Context Builder 一小部分。"""
        rows = await self._db.fetch_all(
            "SELECT * FROM memories WHERE status IN ('ACTIVE', 'LOW_VALUE') "
            "ORDER BY updated_at DESC LIMIT 500"
        )
        now = self._clock()
        scored: list[tuple[float, Memory]] = []
        for row in rows:
            try:
                memory = Memory.from_row(row)
            except (ValueError, KeyError):
                continue
            if memory.expires_at is not None and memory.expires_at <= now:
                continue
            score = similarity(text, memory.content) + 0.2 * memory.importance
            scored.append((score, memory))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [memory for _, memory in scored[: max(1, int(limit))]]

    # -- 写入 -------------------------------------------------------------- #
    async def add(self, memory: Memory) -> Memory:
        """新增记忆；内容高度重复时合并进已有记忆。"""
        similar = await self.find_similar(
            memory.content,
            threshold=self._duplicate_threshold,
            limit=1,
        )
        if similar:
            existing, _ = similar[0]
            merged = self._merge(existing, memory)
            await self.save(merged)
            return merged
        if await self.count(include_all=True) >= self._max:
            raise MemoryLimitError(f"记忆数量已达上限 {self._max}")
        await self.save(memory)
        return memory

    def _merge(self, existing: Memory, incoming: Memory) -> Memory:
        moment = self._clock()
        metadata = dict(existing.metadata)
        metadata["merged_count"] = int(metadata.get("merged_count", 0)) + 1
        merged_type = existing.type
        if existing.type is MemoryType.INFERENCE:
            merged_type = MemoryType.INFERENCE
        return existing.with_changes(
            type=merged_type,
            confidence=max(existing.confidence, incoming.confidence),
            importance=max(existing.importance, incoming.importance),
            frequency=existing.frequency + max(1, incoming.frequency),
            keywords=tuple(dict.fromkeys(existing.keywords + incoming.keywords)),
            updated_at=moment,
            last_verified_at=moment,
            metadata=metadata,
        )

    async def save(self, memory: Memory) -> None:
        row = memory.to_row()
        existing = await self.get(memory.id)
        if (
            existing is not None
            and existing.type is MemoryType.INFERENCE
            and memory.type is MemoryType.FACT
        ):
            # INFERENCE 永远不能自动升级成 FACT：静默保持原类型。
            memory = memory.with_changes(type=MemoryType.INFERENCE)
            row = memory.to_row()
        await self._db.execute(
            """
            INSERT INTO memories
                (id, type, content, confidence, source, importance,
                 created_at, updated_at, last_verified_at, expires_at, metadata,
                 status, relevance, frequency, usage_count, future_value,
                 relationship_value, last_used_at, source_type, source_id,
                 keywords, reactivation_count, last_reactivated_at, conflict)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                type = excluded.type,
                content = excluded.content,
                confidence = excluded.confidence,
                source = excluded.source,
                importance = excluded.importance,
                updated_at = excluded.updated_at,
                last_verified_at = excluded.last_verified_at,
                expires_at = excluded.expires_at,
                metadata = excluded.metadata,
                status = excluded.status,
                relevance = excluded.relevance,
                frequency = excluded.frequency,
                usage_count = excluded.usage_count,
                future_value = excluded.future_value,
                relationship_value = excluded.relationship_value,
                last_used_at = excluded.last_used_at,
                source_type = excluded.source_type,
                source_id = excluded.source_id,
                keywords = excluded.keywords,
                reactivation_count = excluded.reactivation_count,
                last_reactivated_at = excluded.last_reactivated_at,
                conflict = excluded.conflict
            """,
            tuple(row[key] for key in (
                "id", "type", "content", "confidence", "source", "importance",
                "created_at", "updated_at", "last_verified_at", "expires_at",
                "metadata", "status", "relevance", "frequency", "usage_count",
                "future_value", "relationship_value", "last_used_at",
                "source_type", "source_id", "keywords", "reactivation_count",
                "last_reactivated_at", "conflict",
            )),
        )

    upsert = save

    async def touch(self, memory: Memory) -> Memory:
        moment = self._clock()
        updated = memory.with_changes(updated_at=moment, last_verified_at=moment)
        await self.save(updated)
        return updated

    async def mark_used(self, memory: Memory, *, now: datetime | None = None) -> Memory:
        """被检索/引用一次：使用次数 +1，最近使用时间刷新。"""
        moment = now or self._clock()
        updated = memory.with_changes(
            usage_count=memory.usage_count + 1,
            last_used_at=moment,
            frequency=memory.frequency + 1,
        )
        await self.save(updated)
        return updated

    # -- 状态阶梯 ---------------------------------------------------------- #
    async def set_status(
        self,
        memory: Memory,
        status: MemoryStatus | str,
        *,
        now: datetime | None = None,
    ) -> Memory:
        """只允许向前降一级，或恢复为 ACTIVE。"""
        target = MemoryStatus.parse(status)
        moment = now or self._clock()
        current = STATUS_ORDER.index(memory.status)
        wanted = STATUS_ORDER.index(target)
        if target is not MemoryStatus.ACTIVE and wanted != current + 1:
            raise ValueError(
                f"记忆状态只能逐级降级或恢复：{memory.status.value} -> {target.value}"
            )
        updated = memory.with_changes(status=target, updated_at=moment)
        await self.save(updated)
        return updated

    async def restore(
        self, memory: Memory, *, now: datetime | None = None, reason: str = ""
    ) -> Memory:
        """把软删除/过期的记忆拉回 ACTIVE，并记一次重新激活。"""
        moment = now or self._clock()
        metadata = dict(memory.metadata)
        if reason:
            metadata["last_reactivation_reason"] = reason
        updated = memory.with_changes(
            status=MemoryStatus.ACTIVE,
            reactivation_count=memory.reactivation_count + 1,
            last_reactivated_at=moment,
            updated_at=moment,
            metadata=metadata,
        )
        await self.save(updated)
        return updated

    # -- 关系 -------------------------------------------------------------- #
    async def add_relation(
        self,
        from_id: str,
        to_id: str,
        kind: str = "RELATED",
        *,
        now: datetime | None = None,
    ) -> None:
        if kind not in RELATION_KINDS:
            raise MemoryRelationError(f"未知的记忆关系类型: {kind}")
        if from_id == to_id:
            raise MemoryRelationError("记忆不能和自己建立关系")
        await self._db.execute(
            """
            INSERT OR IGNORE INTO memory_relations (from_id, to_id, kind, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (from_id, to_id, kind, (now or self._clock()).isoformat()),
        )

    async def relations_of(self, memory_id: str) -> list[tuple[str, str]]:
        rows = await self._db.fetch_all(
            "SELECT from_id, to_id, kind FROM memory_relations "
            "WHERE from_id = ? OR to_id = ?",
            (memory_id, memory_id),
        )
        return [(str(row["from_id"]), str(row["to_id"]), str(row["kind"])) for row in rows]

    async def find_conflicts(self) -> list[Memory]:
        rows = await self._db.fetch_all(
            "SELECT * FROM memories WHERE conflict = 1 AND status IN ('ACTIVE', 'LOW_VALUE')"
        )
        return [Memory.from_row(row) for row in rows]

    async def mark_conflict(self, memory: Memory, *, flag: bool = True) -> Memory:
        updated = memory.with_changes(conflict=flag, updated_at=self._clock())
        await self.save(updated)
        return updated

    async def _step_to(
        self, memory: Memory, target: MemoryStatus, moment: datetime
    ) -> Memory:
        current = memory
        while STATUS_ORDER.index(current.status) < STATUS_ORDER.index(target):
            next_status = STATUS_ORDER[STATUS_ORDER.index(current.status) + 1]
            current = await self.set_status(current, next_status, now=moment)
        return current

    # -- 衰减与整理 -------------------------------------------------------- #
    async def sweep(self, *, now: datetime | None = None, limit: int = 200) -> list[Memory]:
        """按价值把记忆逐级降档（ACTIVE→LOW_VALUE→SOFT_DELETED→EXPIRED）。"""
        from .scoring import value_of

        moment = now or self._clock()
        changed: list[Memory] = []
        for memory in await self.list(limit=limit):
            value = value_of(memory, moment, self._config)
            target: MemoryStatus | None = None
            if value < self._config.expire_threshold:
                target = MemoryStatus.EXPIRED
            elif value < self._config.soft_delete_threshold:
                target = MemoryStatus.SOFT_DELETED
            elif value < self._config.low_value_threshold:
                target = MemoryStatus.LOW_VALUE
            if target is None or target is memory.status:
                continue
            changed.append(await self._step_to(memory, target, moment))
        return changed

    async def consolidate(
        self, *, now: datetime | None = None, limit: int = 200
    ) -> list[Memory]:
        """保守整理：同类型且几乎相同的记忆合并，旧条目归档，不删。"""
        moment = now or self._clock()
        memories = await self.list(limit=limit)
        consumed: set[str] = set()
        archived: list[Memory] = []
        for index, older in enumerate(memories):
            if older.id in consumed:
                continue
            for newer in memories[index + 1 :]:
                if newer.id in consumed:
                    continue
                if older.type is not newer.type:
                    continue
                if older.conflict or newer.conflict:
                    continue
                if (
                    similarity(older.content, newer.content)
                    < self._config.consolidate_threshold
                ):
                    continue
                merged = older.with_changes(
                    confidence=max(older.confidence, newer.confidence),
                    importance=max(older.importance, newer.importance),
                    frequency=older.frequency + newer.frequency,
                    keywords=tuple(
                        dict.fromkeys([*older.keywords, *newer.keywords])
                    )[:12],
                    updated_at=moment,
                    last_verified_at=moment,
                    metadata={
                        **dict(older.metadata),
                        "consolidated": int(older.metadata.get("consolidated", 0)) + 1,
                    },
                )
                await self.save(merged)
                await self.add_relation(merged.id, newer.id, "MERGED_INTO", now=moment)
                archived.append(
                    await self._step_to(newer, MemoryStatus.ARCHIVED, moment)
                )
                consumed.add(newer.id)
        return archived


__all__ = [
    "MemoryLimitError",
    "MemoryRelationError",
    "MemoryStore",
    "RELATION_KINDS",
]