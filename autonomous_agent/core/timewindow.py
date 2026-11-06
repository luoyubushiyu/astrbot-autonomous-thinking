"""时间窗口判断（夜间/静默时段共用）。"""

from __future__ import annotations

from datetime import datetime

from ..config import parse_quiet_window


def in_window(spec: str | None, now: datetime) -> bool:
    """判断 ``now`` 是否落在 ``HH:MM-HH:MM`` 窗口内；支持跨午夜。

    窗口按机器本地时间比较；传入带时区的时间会先转成当地时间。
    """
    if not spec:
        return False
    if now.tzinfo is not None:
        now = now.astimezone()
    try:
        sh, sm, eh, em = parse_quiet_window(spec)
    except ValueError:
        return False
    minutes = now.hour * 60 + now.minute
    start = sh * 60 + sm
    end = eh * 60 + em
    if start == end:
        return True
    if start < end:
        return start <= minutes < end
    return minutes >= start or minutes < end


def in_any_window(specs, now: datetime) -> bool:
    return any(in_window(spec, now) for spec in specs)


__all__ = ["in_any_window", "in_window"]