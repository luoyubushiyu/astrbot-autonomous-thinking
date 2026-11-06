"""ThoughtRepository：Thought 的唯一入口。

业务代码不直接碰文件、不直接写 SQL——都从这里走，保证：

* SQLite 与文件镜像一致（写入时双写，启动时 ``sync`` 修复）；
* 状态迁移经过 :mod:`thought.lifecycle` 白名单；
* 查重、数量上限这类"程序规则"不交给 LLM。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Iterable, Sequence

from ..config import ThoughtConfig
from ..database.database import Database
from .lifecycle import require_transition
from .models import Thought, ThoughtStatus, _parse_dt, utcnow
from .similarity import similarity
from .workspace import ThoughtWorkspace

logger = logging.getLogger("autonomous_agent.thought")

_ACTIVE_STATUSES = (
    ThoughtStatus.NEW,
    ThoughtStatus.QUEUED,
    ThoughtStatus.THINKING,
    ThoughtStatus.INSIGHT_FOUND,
    ThoughtStatus.FOLLOW_UP,
    ThoughtStatus.WAITING,
)
# 查重时忽略"已经明确出局"的 Thought。
_DUPLICATE_IGNORED = frozenset(
    {ThoughtStatus.DISCARDED, ThoughtStatus.EXPIRED, ThoughtStatus.ARCHIVED}
)


class ThoughtLimitError(RuntimeError):
    """Thought 数量达到上限。"""


class ThoughtDuplicateError(RuntimeError):
    """同样内容的 Thought 已存在。"""

    def __init__(self, thought_id: str) -> None:
        super().__init__(f"重复的 Thought: {thought_id}")
        self.thought_id = thought_id


class ThoughtReactivationError(RuntimeError):
    """不满足唤醒条件（次数、冷却、状态）。"""

    def __init__(self, thought_id: str, reason: str) -> None:
        super().__init__(f"Thought {thought_id} 无法唤醒: {reason}")
        self.thought_id = thought_id
        self.reason = reason


class ThoughtRepository:
    def __init__(
        self,
        db: Database,
        workspace: ThoughtWorkspace,
        *,
        clock: Any = utcnow,
        max_thoughts: int = 500,
        duplicate_threshold: float = 0.6,
    ) -> None:
        self._db = db
        self._workspace = workspace
        self._clock = clock
        self._max_thoughts = max_thoughts
        self._duplicate_threshold = duplicate_threshold

    @property
    def workspace(self) -> ThoughtWorkspace:
        """思考文本的读写口；业务层只通过 Repository 碰文件。"""
        return self._workspace

    # -- 读 ---------------------------------------------------------------- #
    async def get(self, thought_id: str) -> Thought | None:
        row = await self._db.fetch_one(
            "SELECT * FROM thoughts WHERE id = ?", (thought_id,)
        )
        return Thought.from_row(row) if row else None

    async def list(
        self,
        statuses: Sequence[ThoughtStatus] | None = None,
        *,
        limit: int = 100,
    ) -> list[Thought]:
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            rows = await self._db.fetch_all(
                f"SELECT * FROM thoughts WHERE status IN ({placeholders}) "
                "ORDER BY updated_at DESC LIMIT ?",
                tuple(s.value for s in statuses) + (limit,),
            )
        else:
            rows = await self._db.fetch_all(
                "SELECT * FROM thoughts ORDER BY updated_at DESC LIMIT ?", (limit,)
            )
        return [Thought.from_row(row) for row in rows]

    async def candidates(self, now: datetime, *, limit: int = 50) -> list[Thought]:
        """状态上"可以开始思考"的 Thought。"""
        rows = await self._db.fetch_all(
            "SELECT * FROM thoughts WHERE status IN ('NEW', 'FOLLOW_UP') "
            "OR (status = 'QUEUED' "
            "    AND (next_check_at IS NULL OR next_check_at <= ?)) "
            "OR (status = 'WAITING' "
            "    AND (next_check_at IS NULL OR next_check_at <= ?)) "
            "ORDER BY updated_at ASC LIMIT ?",
            (now.isoformat(), now.isoformat(), limit),
        )
        return [Thought.from_row(row) for row in rows]

    async def count(self, statuses: Sequence[ThoughtStatus] | None = None) -> int:
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            value = await self._db.scalar(
                f"SELECT COUNT(*) FROM thoughts WHERE status IN ({placeholders})",
                tuple(s.value for s in statuses),
            )
        else:
            value = await self._db.scalar("SELECT COUNT(*) FROM thoughts")
        return int(value or 0)

    async def count_by_status(self) -> dict[str, int]:
        rows = await self._db.fetch_all(
            "SELECT status, COUNT(*) AS n FROM thoughts GROUP BY status"
        )
        return {str(row["status"]): int(row["n"]) for row in rows}

    async def all_ids(self) -> set[str]:
        rows = await self._db.fetch_all("SELECT id FROM thoughts")
        return {str(row["id"]) for row in rows}

    async def referencing_memory(self, memory_id: str, *, limit: int = 20) -> list[Thought]:
        """找 related_memories 里提到某条 Memory 的 Thought。"""
        rows = await self._db.fetch_all(
            "SELECT * FROM thoughts WHERE related_memories LIKE ? "
            "ORDER BY updated_at DESC LIMIT ?",
            (f'%"{memory_id}"%', limit),
        )
        return [Thought.from_row(row) for row in rows]

    # -- 写 ---------------------------------------------------------------- #
    async def create(
        self,
        thought: Thought,
        *,
        merge_duplicates: bool = True,
    ) -> tuple[Thought, bool]:
        """保存新 Thought。返回 ``(thought, merged)``。

        同内容（相似度 >= 阈值）的活跃 Thought 已存在时默认合并到旧的那条，
        避免无限创建重复文件。
        """
        total = await self.count()
        if total >= self._max_thoughts:
            raise ThoughtLimitError(
                f"Thought 数量已达上限 {self._max_thoughts}，先让遗忘机制清理"
            )

        existing = await self._find_duplicate(thought)
        if existing is not None:
            if not merge_duplicates:
                raise ThoughtDuplicateError(existing.id)
            merged = existing.with_changes(
                importance=max(existing.importance, thought.importance),
                curiosity=max(existing.curiosity, thought.curiosity),
                relevance=max(existing.relevance, thought.relevance),
                updated_at=self._clock(),
                metadata={
                    **dict(existing.metadata),
                    "merged_count": int(existing.metadata.get("merged_count", 0)) + 1,
                },
            )
            await self.save(merged)
            await self.add_relation(existing.id, thought.id, "MERGED_INPUT")
            return merged, True

        await self.save(thought)
        await self._link_parent(thought)
        return thought, False

    async def save(self, thought: Thought) -> None:
        row = thought.to_row()
        columns = ", ".join(row.keys())
        placeholders = ", ".join("?" for _ in row)
        updates = ", ".join(f"{key} = excluded.{key}" for key in row if key != "id")
        await self._db.execute(
            f"INSERT INTO thoughts ({columns}) VALUES ({placeholders}) "
            f"ON CONFLICT(id) DO UPDATE SET {updates}",
            tuple(row.values()),
        )
        try:
            self._workspace.write(thought)
        except OSError:
            logger.warning("Thought 文件写入失败: %s", thought.id, exc_info=True)

    async def transition(
        self,
        thought: Thought,
        target: ThoughtStatus,
        *,
        reason: str = "",
        changes: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> Thought:
        require_transition(thought.status, target)
        moment = now or self._clock()
        updated = thought.with_changes(
            status=target,
            updated_at=moment,
        )
        if changes:
            updated = updated.with_changes(**changes)
        if reason and "next_action" not in (changes or {}):
            updated = updated.with_changes(next_action=reason)
        await self.save(updated)
        return updated

    async def read(self, thought_id: str) -> Thought | None:
        """按 id 读取；等价于 :meth:`get`，保留更直白的名字。"""
        return await self.get(thought_id)

    async def update(self, thought: Thought, **changes: Any) -> Thought:
        """更新一条 Thought（可同时改字段），返回保存后的对象。"""
        if changes:
            thought = thought.with_changes(updated_at=self._clock(), **changes)
        await self.save(thought)
        return thought

    async def move(
        self,
        thought: Thought,
        status: ThoughtStatus,
        *,
        reason: str = "",
        now: datetime | None = None,
    ) -> Thought:
        """把 Thought 移到另一个状态（文件会跟着搬家）。"""
        return await self.transition(thought, status, reason=reason, now=now)

    async def archive(
        self, thought: Thought, *, reason: str = "", now: datetime | None = None
    ) -> Thought:
        return await self.transition(
            thought, ThoughtStatus.ARCHIVED, reason=reason, now=now
        )

    async def delete(
        self, thought: Thought, *, reason: str = "", now: datetime | None = None
    ) -> Thought:
        """软删除：只进 blackbox/discarded，从不物理删。"""
        return await self.transition(
            thought, ThoughtStatus.DISCARDED, reason=reason, now=now
        )

    async def restore(
        self, thought: Thought, *, reason: str = "", now: datetime | None = None
    ) -> Thought:
        if thought.status in (
            ThoughtStatus.DISCARDED,
            ThoughtStatus.EXPIRED,
            ThoughtStatus.ARCHIVED,
            ThoughtStatus.DRAWER,
        ):
            return await self.transition(
                thought, ThoughtStatus.QUEUED, reason=reason, now=now
            )
        return thought

    async def reactivate(
        self,
        thought: Thought,
        *,
        reason: str = "",
        now: datetime | None = None,
        config: ThoughtConfig | None = None,
    ) -> Thought:
        """从抽屉/过期/归档里把 Thought 拉回队列，并记一次唤醒。"""
        from ..config import ThoughtConfig as _Config

        moment = now or self._clock()
        if config is not None and thought.reactivation_count >= config.max_reactivations:
            raise ThoughtReactivationError(thought.id, "唤醒次数已达上限")
        if config is not None and thought.reactivation_count:
            last = thought.metadata.get("last_reactivated_at")
            elapsed = (moment - _parse_dt(last)).total_seconds() if last else None
            if elapsed is not None and elapsed < config.reactivation_cooldown_seconds:
                raise ThoughtReactivationError(thought.id, "还在冷却期")
        if thought.status not in (
            ThoughtStatus.DRAWER,
            ThoughtStatus.EXPIRED,
            ThoughtStatus.ARCHIVED,
        ):
            raise ThoughtReactivationError(thought.id, f"状态 {thought.status.value} 不能唤醒")
        updated = await self.transition(
            thought,
            ThoughtStatus.QUEUED,
            reason=reason or "reactivated",
            now=moment,
            changes={
                "reactivation_count": thought.reactivation_count + 1,
                "metadata": {
                    **dict(thought.metadata),
                    "last_reactivated_at": moment.isoformat(),
                },
            },
        )
        return updated

    # -- Thought Tree ------------------------------------------------------ #
    async def parent_of(self, thought_id: str) -> Thought | None:
        thought = await self.get(thought_id)
        if thought is None or not thought.parent_thought_id:
            return None
        return await self.get(thought.parent_thought_id)

    async def children_of(self, thought_id: str) -> list[Thought]:
        rows = await self._db.fetch_all(
            "SELECT * FROM thoughts WHERE parent_thought_id = ? "
            "ORDER BY created_at ASC",
            (thought_id,),
        )
        known = {str(row["id"]) for row in rows}
        thought = await self.get(thought_id)
        if thought is not None:
            for child_id in thought.child_thought_ids:
                if child_id in known:
                    continue
                child = await self.get(child_id)
                if child is not None:
                    rows.append(child.to_row())
                    known.add(child_id)
        return [Thought.from_row(row) for row in rows]

    async def _related_by_kind(self, thought_id: str, kind: str) -> list[Thought]:
        rows = await self._db.fetch_all(
            "SELECT t.* FROM thought_relations r JOIN thoughts t "
            "ON t.id = CASE WHEN r.from_id = ? THEN r.to_id ELSE r.from_id END "
            "WHERE r.kind = ? AND (r.from_id = ? OR r.to_id = ?)",
            (thought_id, kind, thought_id, thought_id),
        )
        return [Thought.from_row(row) for row in rows]

    async def related_of(self, thought_id: str) -> list[Thought]:
        return await self._related_by_kind(thought_id, "RELATED")

    async def supports_of(self, thought_id: str) -> list[Thought]:
        return await self._related_by_kind(thought_id, "SUPPORTS")

    async def contradicts_of(self, thought_id: str) -> list[Thought]:
        return await self._related_by_kind(thought_id, "CONTRADICTS")

    async def tree(self, thought_id: str) -> dict[str, Any]:
        """父/子/相关/支持/矛盾，一次查全。"""
        parent = await self.parent_of(thought_id)
        return {
            "thought_id": thought_id,
            "parent": parent,
            "children": await self.children_of(thought_id),
            "related": await self.related_of(thought_id),
            "supports": await self.supports_of(thought_id),
            "contradicts": await self.contradicts_of(thought_id),
        }

    async def add_relation(self, from_id: str, to_id: str, kind: str = "RELATED") -> None:
        await self._db.execute(
            "INSERT OR IGNORE INTO thought_relations (from_id, to_id, kind, created_at) "
            "VALUES (?, ?, ?, ?)",
            (from_id, to_id, kind, self._clock().isoformat()),
        )

    async def relations_of(self, thought_id: str) -> list[dict[str, str]]:
        rows = await self._db.fetch_all(
            "SELECT from_id, to_id, kind FROM thought_relations "
            "WHERE from_id = ? OR to_id = ?",
            (thought_id, thought_id),
        )
        return [dict(row) for row in rows]

    async def _link_parent(self, thought: Thought) -> None:
        if not thought.parent_thought_id:
            return
        parent = await self.get(thought.parent_thought_id)
        if parent is None:
            return
        if thought.id not in parent.child_thought_ids:
            await self.save(
                parent.with_changes(
                    child_thought_ids=parent.child_thought_ids + (thought.id,),
                    updated_at=self._clock(),
                )
            )
        await self.add_relation(parent.id, thought.id, "CHILD")

    # -- 一致性 ------------------------------------------------------------ #
    async def sync(self) -> None:
        """按数据库重建文件镜像，并清掉数据库里不存在的孤儿文件。"""
        self._workspace.ensure_layout()
        thoughts = await self.list(limit=10_000)
        for thought in thoughts:
            try:
                self._workspace.write(thought)
            except OSError:
                logger.warning("Thought 文件修复失败: %s", thought.id, exc_info=True)
        known = {thought.id for thought in thoughts}
        removed = self._workspace.prune_orphans(known)
        if removed:
            logger.info("清理了 %d 个孤儿 Thought 文件", removed)

    async def repair(self) -> dict[str, int]:
        """修复数据库与文件镜像的不一致。

        * 数据库有、文件没有 → 按数据库重写；
        * 文件有、数据库没有 → 尝试导入；解析失败则隔离到 blackbox/discarded；
        * 数据库里已不存在的孤儿文件 → 清掉。
        """
        self._workspace.ensure_layout()
        report = {"restored": 0, "imported": 0, "quarantined": 0}
        thoughts = await self.list(limit=100_000)
        for thought in thoughts:
            if self._workspace.path_for(thought).exists():
                continue
            try:
                self._workspace.write(thought)
                report["restored"] += 1
            except OSError:
                logger.warning("Thought 文件修复失败: %s", thought.id, exc_info=True)

        known = {thought.id for thought in thoughts}
        for path in self._workspace.list_files():
            if path.stem in known:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                orphan = Thought.from_payload(payload)
            except (OSError, ValueError, KeyError, TypeError):
                self._workspace.quarantine(path)
                report["quarantined"] += 1
                continue
            try:
                await self.save(orphan)
                report["imported"] += 1
            except Exception:  # noqa: BLE001 - 坏文件不该让修复中断
                self._workspace.quarantine(path)
                report["quarantined"] += 1
        return report

    # -- 内部 -------------------------------------------------------------- #
    async def _find_duplicate(self, thought: Thought) -> Thought | None:
        rows = await self._db.fetch_all(
            "SELECT * FROM thoughts WHERE status NOT IN ('DISCARDED', 'EXPIRED', 'ARCHIVED') "
            "ORDER BY updated_at DESC LIMIT 200"
        )
        fresh = f"{thought.title}\n{thought.content}"
        for row in rows:
            existing = Thought.from_row(row)
            if existing.id == thought.id:
                continue
            if existing.status in _DUPLICATE_IGNORED:
                continue
            if similarity(f"{existing.title}\n{existing.content}", fresh) >= self._duplicate_threshold:
                return existing
        return None


__all__ = [
    "ThoughtDuplicateError",
    "ThoughtLimitError",
    "ThoughtRepository",
    "_ACTIVE_STATUSES",
]