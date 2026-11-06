"""自主思考系统 —— 第一阶段：基础工程骨架。

本包只提供可长期运行、可恢复、可扩展的 Agent 基础设施：
事件队列、状态机、LLM 结构化输出校验、行为守卫、执行器、SQLite 持久化。

Thought Workspace / Emotion / Curiosity / 主动联系 均**不在本阶段范围内**。
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]