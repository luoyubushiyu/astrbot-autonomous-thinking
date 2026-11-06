"""Thought Workspace：自主思考空间。

这一层只处理"想法"本身：类型、生命周期、文件镜像、评分、遗忘、唤醒。
它不关心 QQ、LLM 或事件，因此可以独立测试。
"""

from .lifecycle import (
    DISPOSITION_TARGETS,
    STATUS_TRANSITIONS,
    ThoughtLifecycleError,
    can_transition,
    require_transition,
)
from .models import (
    Disposition,
    NewQuestion,
    Thought,
    ThoughtStatus,
    ThoughtType,
    new_thought_id,
    utcnow,
)
from .repository import (
    ThoughtDuplicateError,
    ThoughtLimitError,
    ThoughtReactivationError,
    ThoughtRepository,
)
from .scoring import score_thought
from .validator import ThoughtDecisionError, ThoughtOutcome, ThoughtValidator
from .workspace import ThoughtWorkspace

__all__ = [
    "DISPOSITION_TARGETS",
    "STATUS_TRANSITIONS",
    "Disposition",
    "NewQuestion",
    "Thought",
    "ThoughtDecisionError",
    "ThoughtDuplicateError",
    "ThoughtLifecycleError",
    "ThoughtLimitError",
    "ThoughtOutcome",
    "ThoughtReactivationError",
    "ThoughtRepository",
    "ThoughtStatus",
    "ThoughtType",
    "ThoughtValidator",
    "ThoughtWorkspace",
    "can_transition",
    "new_thought_id",
    "require_transition",
    "score_thought",
    "utcnow",
]