"""Thought Workspace 文件镜像。

目录结构（可读载体）::

    autonomous_thinking/
    ├── thinking/active/      NEW / QUEUED / THINKING / INSIGHT_FOUND / FOLLOW_UP
    ├── thinking/pending/     WAITING
    ├── drawer/               DRAWER
    ├── diary/                日记（Markdown）
    ├── spoken/               决定主动表达时保存的文本
    ├── blackbox/discarded/   DISCARDED
    ├── blackbox/expired/     EXPIRED
    └── archive/              ARCHIVED / RESOLVED

SQLite 仍是事实来源；文件写入失败只会记录日志（由 Repository 处理），
读取永远以数据库为准。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .models import Thought, ThoughtStatus

_ACTIVE = ("thinking", "active")
_PENDING = ("thinking", "pending")
_DRAWER = ("drawer",)
_DISCARDED = ("blackbox", "discarded")
_EXPIRED = ("blackbox", "expired")
_ARCHIVE = ("archive",)

_STATUS_DIRS: dict[ThoughtStatus, tuple[str, ...]] = {
    ThoughtStatus.NEW: _ACTIVE,
    ThoughtStatus.QUEUED: _ACTIVE,
    ThoughtStatus.THINKING: _ACTIVE,
    ThoughtStatus.INSIGHT_FOUND: _ACTIVE,
    ThoughtStatus.FOLLOW_UP: _ACTIVE,
    ThoughtStatus.WAITING: _PENDING,
    ThoughtStatus.RESOLVED: _ACTIVE,
    ThoughtStatus.DRAWER: _DRAWER,
    ThoughtStatus.DISCARDED: _DISCARDED,
    ThoughtStatus.EXPIRED: _EXPIRED,
    ThoughtStatus.ARCHIVED: _ARCHIVE,
}

_THOUGHT_DIRS = (_ACTIVE, _PENDING, _DRAWER, _DISCARDED, _EXPIRED, _ARCHIVE)


class ThoughtWorkspace:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # -- 目录 -------------------------------------------------------------- #
    def ensure_layout(self) -> None:
        for parts in (
            _ACTIVE,
            _PENDING,
            _DRAWER,
            ("diary",),
            ("spoken",),
            _DISCARDED,
            _EXPIRED,
            _ARCHIVE,
        ):
            (self.root.joinpath(*parts)).mkdir(parents=True, exist_ok=True)

    def subdir_for(self, status: ThoughtStatus) -> Path:
        return self.root.joinpath(*_STATUS_DIRS[status])

    def path_for(self, thought: Thought, status: ThoughtStatus | None = None) -> Path:
        return self.subdir_for(status or thought.status) / f"{thought.id}.json"

    def text_path_for(self, thought: Thought, status: ThoughtStatus | None = None) -> Path:
        return self.subdir_for(status or thought.status) / f"{thought.id}.txt"

    # -- 思考文本 ---------------------------------------------------------- #
    def write_thinking_text(self, thought: Thought, text: str) -> Path:
        """把"纸面思考状态"写到 Thought 同目录下的 .txt。"""
        self.ensure_layout()
        target = self.text_path_for(thought)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".txt.tmp")
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, target)
        for parts in _THOUGHT_DIRS:
            candidate = self.root.joinpath(*parts) / f"{thought.id}.txt"
            if candidate != target:
                try:
                    candidate.unlink()
                except FileNotFoundError:
                    continue
        return target

    def read_thinking_text(self, thought: Thought) -> str | None:
        candidates = [self.text_path_for(thought)]
        candidates.extend(
            self.root.joinpath(*parts) / f"{thought.id}.txt"
            for parts in _THOUGHT_DIRS
            if parts != _STATUS_DIRS[thought.status]
        )
        for path in candidates:
            try:
                return path.read_text(encoding="utf-8")
            except OSError:
                continue
        return None

    def remove_thinking_text(self, thought: Thought) -> None:
        for parts in _THOUGHT_DIRS:
            candidate = self.root.joinpath(*parts) / f"{thought.id}.txt"
            try:
                candidate.unlink()
            except OSError:
                continue

    @property
    def diary_dir(self) -> Path:
        return self.root / "diary"

    @property
    def spoken_dir(self) -> Path:
        return self.root / "spoken"

    # -- 读写 -------------------------------------------------------------- #
    def write(self, thought: Thought) -> Path:
        self.ensure_layout()
        target = self.path_for(thought)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = thought.to_json()
        temp = target.with_suffix(".json.tmp")
        temp.write_text(payload, encoding="utf-8")
        os.replace(temp, target)
        self._remove_stale(thought.id, keep=target)
        self._follow_text(thought)
        return target

    def _follow_text(self, thought: Thought) -> None:
        """思考文本跟着 .json 走：换目录时一起搬，不留在旧状态目录里。"""
        target = self.text_path_for(thought)
        for parts in _THOUGHT_DIRS:
            source = self.root.joinpath(*parts) / f"{thought.id}.txt"
            if source == target or not source.exists():
                continue
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)
            except OSError:
                continue

    def remove(self, thought: Thought) -> None:
        self._remove_stale(thought.id, keep=None)

    def _remove_stale(self, thought_id: str, *, keep: Path | None) -> None:
        for parts in _THOUGHT_DIRS:
            candidate = self.root.joinpath(*parts) / f"{thought_id}.json"
            if candidate == keep:
                continue
            try:
                candidate.unlink()
            except FileNotFoundError:
                continue

    def list_files(self) -> list[Path]:
        files: list[Path] = []
        for parts in _THOUGHT_DIRS:
            folder = self.root.joinpath(*parts)
            if folder.is_dir():
                files.extend(sorted(folder.glob("*.json")))
        return files

    def quarantine(self, path: Path) -> Path:
        """把无法解析/无法入库的文件挪进 blackbox/discarded，不直接删。"""
        self.ensure_layout()
        target = self.subdir_for(ThoughtStatus.DISCARDED) / f"{path.stem}.orphan{path.suffix}"
        try:
            path.replace(target)
        except OSError:
            target = path
        return target

    def prune_orphans(self, known_ids: set[str]) -> int:
        """删除数据库里已经不存在的镜像文件，返回删除数量。"""
        removed = 0
        for path in self.list_files():
            if path.stem in known_ids:
                continue
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            removed += 1
        return removed

    def write_spoken(self, thought_id: str, text: str) -> Path:
        self.ensure_layout()
        target = self.spoken_dir / f"{thought_id}.txt"
        target.write_text(text, encoding="utf-8")
        return target


__all__ = ["ThoughtWorkspace"]