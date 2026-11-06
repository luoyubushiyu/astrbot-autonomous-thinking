"""Thought Workspace 的模型、生命周期、评分、遗忘与唤醒测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from autonomous_agent.config import AppConfig
from autonomous_agent.database.database import Database
from autonomous_agent.database.migrations import initialize_database
from autonomous_agent.llm.schemas import Action, ActionType, Decision
from autonomous_agent.thought.forgetting import forget_score, sweep_forgetting
from autonomous_agent.thought.lifecycle import (
    ThoughtLifecycleError,
    can_transition,
    require_transition,
    target_status_for,
)
from autonomous_agent.thought.models import (
    Disposition,
    NewQuestion,
    Thought,
    ThoughtStatus,
    ThoughtType,
    utcnow,
)
from autonomous_agent.thought.reactivation import try_reactivate
from autonomous_agent.thought.repository import (
    ThoughtLimitError,
    ThoughtReactivationError,
    ThoughtRepository,
)
from autonomous_agent.thought.scoring import rank_thoughts, score_thought
from autonomous_agent.thought.validator import ThoughtDecisionError, ThoughtValidator
from autonomous_agent.thought.workspace import ThoughtWorkspace


class LifecycleTests(unittest.TestCase):
    def test_transition_table(self) -> None:
        self.assertTrue(can_transition(ThoughtStatus.NEW, ThoughtStatus.QUEUED))
        self.assertTrue(can_transition(ThoughtStatus.DRAWER, ThoughtStatus.QUEUED))
        self.assertFalse(can_transition(ThoughtStatus.ARCHIVED, ThoughtStatus.THINKING))

    def test_require_transition_raises(self) -> None:
        with self.assertRaises(ThoughtLifecycleError):
            require_transition(ThoughtStatus.NEW, ThoughtStatus.RESOLVED)

    def test_disposition_targets(self) -> None:
        self.assertIs(target_status_for(Disposition.DRAWER), ThoughtStatus.DRAWER)
        self.assertIs(target_status_for(Disposition.BLACKBOX), ThoughtStatus.DISCARDED)
        self.assertIs(target_status_for(Disposition.SPEAK), ThoughtStatus.INSIGHT_FOUND)


class ScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig().thought
        self.now = utcnow()

    def test_importance_and_curiosity_raise_score(self) -> None:
        low = Thought.create(
            type=ThoughtType.IDEA,
            title="低",
            importance=0.1,
            curiosity=0.1,
            now=self.now,
        )
        high = Thought.create(
            type=ThoughtType.IDEA,
            title="高",
            importance=0.9,
            curiosity=0.9,
            now=self.now,
        )
        self.assertGreater(
            score_thought(high, self.now, self.config),
            score_thought(low, self.now, self.config),
        )

    def test_stale_thought_loses_points(self) -> None:
        fresh = Thought.create(type=ThoughtType.IDEA, title="新", now=self.now)
        stale = fresh.with_changes(
            created_at=self.now - timedelta(days=30),
            updated_at=self.now - timedelta(days=30),
        )
        self.assertGreater(
            score_thought(fresh, self.now, self.config),
            score_thought(stale, self.now, self.config),
        )

    def test_rank_orders_by_score(self) -> None:
        high = Thought.create(
            type=ThoughtType.IDEA, title="高", importance=0.9, now=self.now
        )
        low = Thought.create(
            type=ThoughtType.IDEA, title="低", importance=0.1, now=self.now
        )
        ranked = rank_thoughts([low, high], self.now, self.config)
        self.assertEqual(ranked[0].id, high.id)


class ValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig().thought
        self.validator = ThoughtValidator(self.config)
        self.thought = Thought.create(
            type=ThoughtType.QUESTION, title="问题", status=ThoughtStatus.THINKING
        )

    def _decision(self, **parameters) -> Decision:
        params = {"thought_id": self.thought.id, "insight": "结论", "disposition": "KEEP"}
        params.update(parameters)
        return Decision("思考", Action(ActionType.THINK, params), 0.6)

    def test_accepts_keep(self) -> None:
        outcome = self.validator.validate(self._decision(), thought=self.thought)
        self.assertIs(outcome.disposition, Disposition.KEEP)
        self.assertIs(outcome.target_status, ThoughtStatus.INSIGHT_FOUND)

    def test_rejects_empty_insight(self) -> None:
        with self.assertRaises(ThoughtDecisionError):
            self.validator.validate(self._decision(insight="  "), thought=self.thought)

    def test_rejects_bad_disposition(self) -> None:
        with self.assertRaises(ThoughtDecisionError):
            self.validator.validate(
                self._decision(disposition="PARTY"), thought=self.thought
            )

    def test_speak_requires_contact_message(self) -> None:
        with self.assertRaises(ThoughtDecisionError):
            self.validator.validate(
                self._decision(disposition="SPEAK"), thought=self.thought
            )

    def test_contact_message_only_for_speak(self) -> None:
        with self.assertRaises(ThoughtDecisionError):
            self.validator.validate(
                self._decision(disposition="KEEP", contact_message="多余的"),
                thought=self.thought,
            )

    def test_too_many_questions(self) -> None:
        questions = [
            {"title": f"q{i}", "content": "c"} for i in range(self.config.max_new_questions + 1)
        ]
        with self.assertRaises(ThoughtDecisionError):
            self.validator.validate(
                self._decision(new_questions=questions), thought=self.thought
            )


class RepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.db = Database(self.root / "agent.db")
        await initialize_database(self.db)
        self.workspace = ThoughtWorkspace(self.root / "thinking")
        self.workspace.ensure_layout()
        self.repo = ThoughtRepository(
            self.db, self.workspace, max_thoughts=5, duplicate_threshold=0.85
        )
        self.addAsyncCleanup(self._cleanup)

    async def _cleanup(self) -> None:
        await self.db.close()
        self._tmp.cleanup()

    async def test_create_writes_file_and_row(self) -> None:
        thought = Thought.create(type=ThoughtType.IDEA, title="一个念头", content="内容")
        saved, merged = await self.repo.create(thought)
        self.assertFalse(merged)
        self.assertEqual((await self.repo.get(saved.id)).title, "一个念头")
        self.assertTrue(self.workspace.path_for(saved).is_file())

    async def test_duplicate_merges(self) -> None:
        first = Thought.create(
            type=ThoughtType.QUESTION, title="为什么天是蓝的", content="为什么天是蓝的"
        )
        await self.repo.create(first)
        second = Thought.create(
            type=ThoughtType.QUESTION, title="为什么天是蓝的", content="为什么天是蓝的"
        )
        saved, merged = await self.repo.create(second)
        self.assertTrue(merged)
        self.assertEqual(saved.id, first.id)
        self.assertEqual(await self.repo.count(), 1)

    async def test_drawer_moves_file(self) -> None:
        thought = Thought.create(type=ThoughtType.IDEA, title="先放抽屉")
        saved, _ = await self.repo.create(thought)
        active = self.workspace.path_for(saved)
        moved = await self.repo.transition(saved, ThoughtStatus.DRAWER)
        self.assertFalse(active.exists())
        self.assertTrue(self.workspace.path_for(moved).is_file())
        self.assertEqual(moved.status, ThoughtStatus.DRAWER)

    async def test_limit_raises(self) -> None:
        for index in range(5):
            await self.repo.create(
                Thought.create(type=ThoughtType.IDEA, title=f"念头 {index}")
            )
        with self.assertRaises(ThoughtLimitError):
            await self.repo.create(Thought.create(type=ThoughtType.IDEA, title="多余的"))

    async def test_forgetting_expires_old_thought(self) -> None:
        now = utcnow()
        old = Thought.create(
            type=ThoughtType.IDEA,
            title="很久以前的想法",
            importance=0.1,
            relevance=0.1,
            confidence=0.1,
            now=now - timedelta(days=90),
        ).with_changes(
            created_at=now - timedelta(days=90),
            updated_at=now - timedelta(days=90),
        )
        saved, _ = await self.repo.create(old)
        saved = saved.with_changes(status=ThoughtStatus.INSIGHT_FOUND)
        await self.repo.save(saved)
        self.assertGreater(forget_score(saved, now, AppConfig().thought), 1.0)
        result = await sweep_forgetting(self.repo, AppConfig().thought, now=now)
        self.assertIn(saved.id, [item.thought_id for item in result])
        self.assertEqual((await self.repo.get(saved.id)).status, ThoughtStatus.EXPIRED)

    async def test_reactivation_brings_drawer_back(self) -> None:
        now = utcnow()
        thought = Thought.create(
            type=ThoughtType.QUESTION,
            title="要不要学游泳",
            content="要不要学游泳",
            now=now - timedelta(days=10),
        )
        saved, _ = await self.repo.create(thought)
        await self.repo.transition(saved, ThoughtStatus.DRAWER, now=now - timedelta(days=9))
        revived = await try_reactivate(
            self.repo, "要不要去学游泳", AppConfig().thought, now=now
        )
        self.assertEqual(len(revived), 1)
        self.assertEqual(revived[0].status, ThoughtStatus.QUEUED)
        self.assertEqual(revived[0].reactivation_count, 1)

    async def test_sync_restores_missing_file(self) -> None:
        thought = Thought.create(type=ThoughtType.IDEA, title="文件丢了")
        saved, _ = await self.repo.create(thought)
        self.workspace.path_for(saved).unlink()
        await self.repo.sync()
        self.assertTrue(self.workspace.path_for(saved).is_file())

    async def test_candidates_include_due_waiting(self) -> None:
        now = utcnow()
        thought = Thought.create(
            type=ThoughtType.QUESTION, title="等一会", next_check_at=now - timedelta(minutes=1)
        )
        saved, _ = await self.repo.create(thought)
        await self.repo.transition(saved, ThoughtStatus.WAITING, now=now)
        candidates = await self.repo.candidates(now)
        self.assertEqual([item.id for item in candidates], [saved.id])


class WorkspaceTests(unittest.TestCase):
    def test_layout_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = ThoughtWorkspace(Path(tmp) / "thinking")
            workspace.ensure_layout()
            self.assertTrue((workspace.root / "thinking" / "active").is_dir())
            self.assertTrue((workspace.diary_dir).is_dir())

    def test_json_roundtrip(self) -> None:
        thought = Thought.create(type=ThoughtType.IDEA, title="往返")
        payload = json.loads(thought.to_json())
        self.assertEqual(payload["title"], "往返")
        self.assertEqual(Thought.from_row(thought.to_row()).id, thought.id)


class RepositoryApiTests(unittest.IsolatedAsyncioTestCase):
    """第二阶段文档要求的 Repository 接口、Thought Tree 与一致性修复。"""

    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.db = Database(self.root / "agent.db")
        await initialize_database(self.db)
        self.workspace = ThoughtWorkspace(self.root / "thinking")
        self.workspace.ensure_layout()
        self.repo = ThoughtRepository(self.db, self.workspace)
        self.addAsyncCleanup(self._cleanup)

    async def _cleanup(self) -> None:
        await self.db.close()
        self._tmp.cleanup()

    async def test_read_update_move_archive_delete_restore(self) -> None:
        saved, _ = await self.repo.create(
            Thought.create(type=ThoughtType.IDEA, title="接口齐全")
        )
        self.assertEqual((await self.repo.read(saved.id)).id, saved.id)

        updated = await self.repo.update(saved, importance=0.9)
        self.assertEqual(updated.importance, 0.9)

        queued = await self.repo.move(updated, ThoughtStatus.QUEUED, reason="排进队列")
        self.assertEqual(queued.status, ThoughtStatus.QUEUED)
        self.assertTrue(self.workspace.path_for(queued).is_file())

        archived = await self.repo.archive(queued, reason="归档")
        self.assertEqual(archived.status, ThoughtStatus.ARCHIVED)

        restored = await self.repo.restore(archived)
        self.assertEqual(restored.status, ThoughtStatus.QUEUED)

        discarded = await self.repo.delete(restored, reason="不要了")
        self.assertEqual(discarded.status, ThoughtStatus.DISCARDED)
        # 软删除：数据库行和文件都还在，只是搬进黑盒。
        self.assertIsNotNone(await self.repo.get(discarded.id))
        self.assertTrue(self.workspace.path_for(discarded).is_file())

    async def test_reactivate_checks_limit_and_cooldown(self) -> None:
        now = utcnow()
        thought = Thought.create(
            type=ThoughtType.IDEA,
            title="被放抽屉",
            now=now,
        )
        saved, _ = await self.repo.create(thought)
        await self.repo.transition(saved, ThoughtStatus.DRAWER, now=now)

        with self.assertRaises(ThoughtReactivationError):
            await self.repo.reactivate(
                saved.with_changes(
                    status=ThoughtStatus.DRAWER,
                    reactivation_count=3,
                ),
                now=now,
                config=AppConfig().thought,
            )

        cooled = saved.with_changes(
            status=ThoughtStatus.DRAWER,
            reactivation_count=1,
            metadata={"last_reactivated_at": now.isoformat()},
        )
        with self.assertRaises(ThoughtReactivationError):
            await self.repo.reactivate(
                cooled,
                now=now + timedelta(minutes=1),
                config=AppConfig().thought,
            )

        revived = await self.repo.reactivate(
            saved.with_changes(status=ThoughtStatus.DRAWER),
            now=now,
            config=AppConfig().thought,
        )
        self.assertEqual(revived.status, ThoughtStatus.QUEUED)
        self.assertEqual(revived.reactivation_count, 1)

    async def test_tree_parent_children_and_relations(self) -> None:
        parent, _ = await self.repo.create(
            Thought.create(type=ThoughtType.QUESTION, title="Q1")
        )
        child, _ = await self.repo.create(
            Thought.create(
                type=ThoughtType.QUESTION, title="Q2", parent_thought_id=parent.id
            )
        )
        other, _ = await self.repo.create(
            Thought.create(type=ThoughtType.IDEA, title="相关")
        )
        await self.repo.add_relation(parent.id, other.id, "RELATED")
        await self.repo.add_relation(child.id, other.id, "SUPPORTS")
        await self.repo.add_relation(child.id, other.id, "CONTRADICTS")

        reloaded_parent = await self.repo.get(parent.id)
        self.assertIn(child.id, reloaded_parent.child_thought_ids)

        self.assertEqual((await self.repo.parent_of(child.id)).id, parent.id)
        self.assertEqual([t.id for t in await self.repo.children_of(parent.id)], [child.id])
        self.assertIn(other.id, [t.id for t in await self.repo.related_of(parent.id)])
        self.assertIn(other.id, [t.id for t in await self.repo.supports_of(child.id)])
        self.assertIn(other.id, [t.id for t in await self.repo.contradicts_of(child.id)])

        tree = await self.repo.tree(child.id)
        self.assertEqual(tree["parent"].id, parent.id)
        self.assertTrue(tree["supports"])

    async def test_repair_imports_orphan_and_quarantines_garbage(self) -> None:
        thought = Thought.create(type=ThoughtType.IDEA, title="数据库里有的")
        saved, _ = await self.repo.create(thought)
        self.workspace.path_for(saved).unlink()

        orphan = Thought.create(type=ThoughtType.IDEA, title="只剩文件的")
        orphan_path = self.workspace.path_for(orphan)
        orphan_path.write_text(orphan.to_json(), encoding="utf-8")

        broken = self.workspace.subdir_for(ThoughtStatus.NEW) / "TH_broken.json"
        broken.write_text("{ not json", encoding="utf-8")

        report = await self.repo.repair()
        self.assertTrue(self.workspace.path_for(saved).is_file())
        self.assertIsNotNone(await self.repo.get(orphan.id))
        self.assertEqual(report["imported"], 1)
        self.assertGreaterEqual(report["restored"], 1)
        self.assertEqual(report["quarantined"], 1)
        self.assertFalse(broken.exists())


class ScoringPhaseTwoTests(unittest.TestCase):
    def test_time_suitability_prefers_due_thoughts(self) -> None:
        now = utcnow()
        due = Thought.create(
            type=ThoughtType.IDEA, title="现在就想", next_check_at=now - timedelta(minutes=1)
        )
        later = due.with_changes(
            id="TH_later", next_check_at=now + timedelta(hours=20)
        )
        config = AppConfig().thought
        self.assertGreater(
            score_thought(due, now, config), score_thought(later, now, config)
        )

    def test_revisit_cost_lowers_score(self) -> None:
        now = utcnow()
        fresh = Thought.create(type=ThoughtType.IDEA, title="第一次")
        old = fresh.with_changes(id="TH_old", revisit_count=5)
        config = AppConfig().thought
        self.assertGreater(
            score_thought(fresh, now, config), score_thought(old, now, config)
        )


if __name__ == "__main__":
    unittest.main()