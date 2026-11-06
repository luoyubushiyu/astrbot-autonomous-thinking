"""结构化日志。

控制台给人看，文件给机器看（JSON Lines）。两边都不依赖第三方库。
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import LoggingConfig

_ROOT_LOGGER = "autonomous_agent"
_HANDLER_MARK = "_autonomous_agent_handler"

# 允许通过 logger.info("...", extra={...}) 带出的结构化字段。
_EXTRA_FIELDS = (
    "event_type",
    "event_id",
    "conversation_id",
    "action",
    "state",
    "latency_ms",
    "decision",
    "error",
    "reason",
)


class JsonLineFormatter(logging.Formatter):
    """一行一个 JSON 对象。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in _EXTRA_FIELDS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    """人类可读的单行日志。"""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = []
        for key in _EXTRA_FIELDS:
            value = getattr(record, key, None)
            if value is not None:
                extras.append(f"{key}={value}")
        if extras:
            base = f"{base} | {' '.join(extras)}"
        return base


def setup_logging(config: LoggingConfig, *, root_name: str = _ROOT_LOGGER) -> None:
    """安装日志 handler。重复调用会先移除旧 handler，避免重复输出。"""
    logger = logging.getLogger(root_name)
    logger.setLevel(_level_value(config.level))
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            logger.removeHandler(handler)
            handler.close()

    if config.console:
        console = logging.StreamHandler(stream=sys.stderr)
        console.setFormatter(ConsoleFormatter())
        console.setLevel(_level_value(config.level))
        setattr(console, _HANDLER_MARK, True)
        logger.addHandler(console)

    if config.file:
        path = Path(config.file)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                path,
                maxBytes=max(0, config.max_bytes),
                backupCount=max(0, config.backup_count),
                encoding="utf-8",
            )
        except OSError as exc:  # 日志落盘失败不能拖垮主流程
            logger.warning("日志文件不可用，已退化为仅控制台输出: %s", exc)
        else:
            file_handler.setFormatter(
                JsonLineFormatter() if config.json_lines else ConsoleFormatter()
            )
            file_handler.setLevel(_level_value(config.level))
            setattr(file_handler, _HANDLER_MARK, True)
            logger.addHandler(file_handler)

    if not logger.handlers:
        logger.addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """返回包内 logger；传入 ``__name__`` 即可。"""
    if name == _ROOT_LOGGER or name.startswith(f"{_ROOT_LOGGER}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER}.{name}")


def _level_value(level: str | int) -> int:
    if isinstance(level, int):
        return level
    value = logging.getLevelName(str(level).upper())
    return value if isinstance(value, int) else logging.INFO