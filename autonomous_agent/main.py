"""CLI 入口：python -m autonomous_agent.main --config config.json"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
from pathlib import Path
from typing import Sequence

from .config import ConfigError, load_config
from .core.agent import Agent
from .database.database import DatabaseError
from .monitoring.logger import setup_logging

logger = logging.getLogger("autonomous_agent.main")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autonomous-agent",
        description="QQ Bot 自主思考系统（第一阶段：基础骨架）",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="配置文件路径，默认 $AGENT_CONFIG 或项目根 config.json",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="覆盖配置中的日志级别，如 DEBUG",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="启动后立即停止，用于自检",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="打印脱敏后的配置并退出",
    )
    return parser


async def async_main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config or os.environ.get("AGENT_CONFIG") or None)
    except ConfigError as exc:
        print(f"配置错误: {exc}")
        return 2

    logging_config = config.logging
    if args.log_level:
        logging_config.level = args.log_level.upper()
    setup_logging(logging_config)

    if args.print_config:
        print(json.dumps(config.to_dict(mask_secrets=True), ensure_ascii=False, indent=2))
        return 0

    agent = Agent(config)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, agent.request_stop)
        except (NotImplementedError, RuntimeError):  # Windows 部分环境不支持
            pass

    if args.once:
        try:
            await agent.start(background=False)
            stats = await agent.stats()
        except DatabaseError as exc:
            print(f"数据库错误: {exc}")
            return 3
        finally:
            if agent.running:
                await agent.stop()
        print("自检完成:", json.dumps(stats, ensure_ascii=False, default=str))
        return 0

    logger.info("启动 Agent（provider=%s, qq=%s）", config.llm.provider, config.qq.adapter)
    try:
        await agent.run()
    except DatabaseError as exc:
        print(f"数据库错误: {exc}")
        return 3
    except asyncio.CancelledError:  # pragma: no cover
        pass
    finally:
        if agent.running:
            await agent.stop()
    logger.info("Agent 已退出")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return asyncio.run(async_main(argv))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())