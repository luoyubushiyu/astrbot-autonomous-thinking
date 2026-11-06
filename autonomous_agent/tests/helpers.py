"""测试公共工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autonomous_agent.config import AppConfig


def make_config(tmp_path: Path) -> AppConfig:
    """一套不碰真实文件、不开后台循环的配置。"""
    config = AppConfig()
    config.database.path = str(tmp_path / "agent.db")
    config.logging.file = None
    config.logging.console = False
    config.scheduler.enabled = False
    config.qq.enabled = False
    config.llm.request_timeout_seconds = 1.0
    config.llm.provider = "mock"
    config.thought.workspace_dir = str(tmp_path / "thinking")
    config.contact.minimum_interval_seconds = 0
    config.contact.cooldown_seconds = 0
    # 夜间窗口默认关掉：单测跑在任意时刻，不能因为真实时钟到了夜里就进 NIGHT。
    config.contact.night_hours = []
    return config


def decision(
    action: str,
    parameters: dict[str, Any] | None = None,
    *,
    reasoning: str = "测试决策",
    confidence: float = 0.5,
) -> str:
    return json.dumps(
        {
            "reasoning": reasoning,
            "action": {"type": action, "parameters": parameters or {}},
            "confidence": confidence,
        },
        ensure_ascii=False,
    )


DO_NOTHING = decision("DO_NOTHING", {"reason": "安静待着"}, confidence=0.1)


def send_message(conversation_id: str = "conv-1", content: str = "在的呀") -> str:
    return decision(
        "SEND_MESSAGE",
        {"conversation_id": conversation_id, "content": content},
        confidence=0.9,
    )


def wait(seconds: int = 5) -> str:
    return decision("WAIT", {"seconds": seconds}, confidence=0.4)


def enter_silent(minutes: int = 10) -> str:
    return decision("ENTER_SILENT", {"minutes": minutes}, confidence=0.4)


def think(
    thought_id: str,
    *,
    insight: str = "想明白了一点东西",
    disposition: str = "KEEP",
    uncertainty: str = "",
    new_questions: list[dict[str, Any]] | None = None,
    next_check_minutes: int | None = None,
    contact_message: str | None = None,
    memory_candidates: list[dict[str, Any]] | None = None,
    confidence: float = 0.6,
) -> str:
    parameters: dict[str, Any] = {
        "thought_id": thought_id,
        "insight": insight,
        "disposition": disposition,
    }
    if uncertainty:
        parameters["uncertainty"] = uncertainty
    if new_questions is not None:
        parameters["new_questions"] = new_questions
    if next_check_minutes is not None:
        parameters["next_check_minutes"] = next_check_minutes
    if contact_message is not None:
        parameters["contact_message"] = contact_message
    if memory_candidates is not None:
        parameters["memory_candidates"] = memory_candidates
    return decision("THINK", parameters, confidence=confidence)


def user_message_event(
    *,
    conversation_id: str = "conv-1",
    message_id: str = "m-1",
    user_id: str = "user-1",
    content: str = "在吗",
):
    from autonomous_agent.core.event import Event

    return Event.user_message(
        conversation_id=conversation_id,
        message_id=message_id,
        user_id=user_id,
        content=content,
    )