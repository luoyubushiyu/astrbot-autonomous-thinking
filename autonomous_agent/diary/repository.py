"""Diary：按天写的轻量日志。

只记"今天发生了什么、想到了什么"，不是完整 CoT。
文件是 ``diary/YYYY-MM-DD.md``，数据库 ``diary`` 表只是索引/镜像。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ..database.database import Database


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DiaryStore:
    def __init__(
        self,
        db: Database,
        diary_dir: Path | str,
        *,
        clock=utcnow,
        max_entry_chars: int = 2000,
    ) -> None:
        self._db = db
        self._dir = Path(diary_dir)
        self._clock = clock
        self._max = max_entry_chars

    def _day(self, now: datetime | None = None) -> str:
        return (now or self._clock()).strftime("%Y-%m-%d")

    def path_for(self, day: str) -> Path:
        return self._dir / f"{day}.md"

    async def append(self, text: str, *, now: datetime | None = None) -> str:
        moment = now or self._clock()
        day = self._day(moment)
        entry = text.strip()
        if not entry:
            raise ValueError("日记内容不能为空")
        entry = entry[: self._max]
        existing = await self.read(day) or ""
        block = f"{existing.rstrip()}\n\n- {moment.strftime('%H:%M')} {entry}".strip()
        await self.save(day, block, now=moment)
        return day

    async def read(self, day: str) -> str | None:
        row = await self._db.fetch_one("SELECT content FROM diary WHERE day = ?", (day,))
        if row is not None:
            return str(row["content"])
        path = self.path_for(day)
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8")
            except OSError:
                return None
        return None

    async def recent(self, *, days: int = 7) -> list[tuple[str, str]]:
        rows = await self._db.fetch_all(
            "SELECT day, content FROM diary ORDER BY day DESC LIMIT ?", (int(days),)
        )
        return [(str(row["day"]), str(row["content"])) for row in rows]

    async def save(self, day: str, content: str, *, now: datetime | None = None) -> None:
        moment = now or self._clock()
        await self._db.execute(
            "INSERT INTO diary (day, content, updated_at) VALUES (?, ?, ?)"
            " ON CONFLICT(day) DO UPDATE SET content = excluded.content,"
            " updated_at = excluded.updated_at",
            (day, content, moment.isoformat()),
        )
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(day)
        try:
            tmp = path.with_suffix(".md.tmp")
            tmp.write_text(content + "\n", encoding="utf-8")
            tmp.replace(path)
        except OSError:
            # 文件只是镜像；写失败不影响主流程。
            pass


__all__ = ["DiaryStore", "utcnow"]