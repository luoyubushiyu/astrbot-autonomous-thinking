"""配置层。

设计要点：
* 使用 dataclass 定义强类型配置，加载时做键名与类型校验；
* 支持 JSON 配置文件 + 环境变量覆盖（优先级：环境变量 > 配置文件 > 默认值）；
* 相对路径以配置文件所在目录为基准解析，避免 ``python -m`` 与直接运行时
  写库/写日志落到不同目录。
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, get_type_hints


class ConfigError(ValueError):
    """配置内容非法。"""


# --------------------------------------------------------------------------- #
# 配置分节
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class DatabaseConfig:
    """SQLite 配置。"""

    path: str = "data/agent.db"
    busy_timeout_ms: int = 5000
    store_events: bool = True
    # 长期运行清理：已处理事件与旧指标保留天数。
    event_retention_days: int = 30
    metrics_retention_days: int = 90


@dataclass(slots=True)
class LLMConfig:
    """LLM 接入配置。provider 决定使用哪个 Adapter 实现。"""

    provider: str = "mock"  # mock | deepseek | openai | openai_compatible
    base_url: str = "https://api.deepseek.com/v1"
    api_key: str = ""
    model: str = "deepseek-chat"
    request_timeout_seconds: float = 30.0
    temperature: float = 0.7
    max_output_tokens: int = 512
    json_mode: bool = True
    extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class BudgetConfig:
    """token 用量上限。

    0 表示不限。``state_file`` 留空时放在数据库同目录，重启后当天用量接着算。
    """

    max_daily_tokens: int = 0
    state_file: str = ""


@dataclass(slots=True)
class QQConfig:
    """QQ 侧配置。第一阶段只需要 Mock。"""

    adapter: str = "mock"  # mock
    self_id: str = "bot"
    poll_interval_seconds: float = 1.0
    enabled: bool = True


@dataclass(slots=True)
class SchedulerConfig:
    """调度器配置。第一阶段只有心跳 tick，没有主动联系调度。"""

    enabled: bool = True
    autonomous_tick_seconds: float = 300.0
    # 有念头等着想的时候，缩短步长，让思考链接得上；空池子才回到 300 秒。
    active_tick_seconds: float = 60.0
    max_wait_seconds: int = 3600
    safe_idle_alert_seconds: float = 0.0


@dataclass(slots=True)
class ThoughtConfig:
    """Thought Workspace 与自主思考循环的配置。

    ``workspace_dir`` 是思考空间的文件目录；SQLite 才是结构化事实的来源，
    文件只是可读镜像（ThoughtRepository 负责保持两边一致）。
    """

    enabled: bool = True
    workspace_dir: str = "autonomous_thinking"
    max_active_thoughts: int = 200
    max_thoughts: int = 500
    max_new_questions: int = 3
    from_user_messages: bool = True
    history_limit: int = 8
    memory_limit: int = 5
    # Thought Scoring 的权重，全部可调。
    w_importance: float = 1.0
    w_curiosity: float = 0.8
    w_relevance: float = 0.6
    w_novelty: float = 0.3
    w_unresolved: float = 0.5
    w_relationship: float = 0.2
    w_future: float = 0.2
    w_time_suitability: float = 0.15
    penalty_repetition: float = 0.4
    penalty_stale: float = 0.6
    penalty_cost: float = 0.1
    novelty_revisits: int = 2
    stale_days: float = 7.0
    # 遗忘（软删除，先 EXPIRED 再归档）。
    forget_after_days: float = 30.0
    forget_inactivity_days: float = 14.0
    forget_threshold: float = 1.1
    # 重新唤醒。
    reactivation_threshold: float = 0.35
    reactivation_cooldown_seconds: float = 3600.0
    max_reactivations: int = 3
    # 维护节奏：每 N 次 tick 扫一次遗忘；0 表示只手动触发。
    maintenance_every_ticks: int = 10
    # 思考文本：一次 Tick 只推进一步；文本只保留摘要，不存完整思维链。
    max_thought_depth: int = 3
    max_followups_per_thought: int = 3
    # CONTINUE 之后隔多久接着想：太密会烧 token，太疏思考链就断了。
    continue_think_minutes: float = 3.0
    thinking_text_max_chars: int = 8000
    history_keep_turns: int = 6


@dataclass(slots=True)
class MemoryConfig:
    """长期记忆的存取、价值评分、遗忘与重新激活配置。

    所有权重与阈值都由程序计算，LLM 只能提供辅助信息。
    """

    enabled: bool = True
    retrieval_limit: int = 5
    duplicate_threshold: float = 0.9
    max_memories: int = 2000
    # 一次思考最多提交几条 SAVE_MEMORY 意图
    max_candidates_per_think: int = 3
    # 价值评分权重
    w_importance: float = 1.0
    w_confidence: float = 0.9
    w_relevance: float = 0.6
    w_future: float = 0.3
    w_relationship: float = 0.2
    penalty_stale: float = 0.5
    penalty_duplicate: float = 0.3
    penalty_low_confidence: float = 0.3
    stale_days: float = 45.0
    # 状态阶梯阈值：value <= 阈值时降一级
    low_value_threshold: float = 0.12
    soft_delete_threshold: float = 0.07
    expire_threshold: float = 0.03
    # 整理
    consolidate_threshold: float = 0.92
    consolidate_every_ticks: int = 20
    # 重新激活保护
    reactivate_threshold: float = 0.45
    reactivate_min_value: float = 0.05
    max_reactivations: int = 2
    reactivation_cooldown_seconds: float = 3600.0


@dataclass(slots=True)
class ContactConfig:
    """主动联系的硬限制。这些数值不允许 LLM 修改。"""

    enabled: bool = True
    hourly_limit: int = 2
    daily_limit: int = 6
    minimum_interval_seconds: float = 1800.0
    consecutive_no_reply_limit: int = 3
    cooldown_seconds: float = 600.0
    night_hours: list[str] = field(default_factory=lambda: ["23:30-07:30"])
    duplicate_threshold: float = 0.8
    duplicate_window: int = 5
    max_attempts_per_thought: int = 2
    timeout_seconds: float = 1800.0
    # 同一话题连续主动联系的限制
    topic_repeat_limit: int = 2
    topic_repeat_window_minutes: int = 1440
    # 短时间内被挡太多次，视为异常，先停一轮
    anomaly_block_threshold: int = 5
    anomaly_window_minutes: int = 60
    # 连续没回复到这个数，就自动进入 SILENT（只停联系，不停思考）
    silent_after_consecutive: int = 3
    silent_minutes: float = 720.0


@dataclass(slots=True)
class EmotionConfig:
    """模拟情绪：只影响倾向，不直接触发行为。"""

    enabled: bool = True
    baseline: float = 0.3
    decay: float = 0.05
    # 单次事件最多允许改变多少，防止一条消息把 mood 从 0 拉到 1。
    max_emotion_delta_per_event: float = 0.15


@dataclass(slots=True)
class TopicConfig:
    """长期关注的主题：活跃度、生命周期与去重。"""

    enabled: bool = True
    duplicate_threshold: float = 0.8
    min_importance_to_create: float = 0.55
    # 活跃度评分权重（程序计算，LLM 不参与）。
    w_recency: float = 0.5
    w_activity: float = 0.3
    w_importance: float = 0.2
    # 生命周期阈值（天）：多久没活动算低活跃/休眠/归档。
    low_activity_days: float = 3.0
    dormant_days: float = 14.0
    archive_days: float = 45.0
    max_context_topics: int = 8


@dataclass(slots=True)
class CuriosityConfig:
    """Curiosity Engine：把"值得探索"变成 CREATE_THOUGHT 候选。"""

    enabled: bool = True
    # 硬上限，程序强制，LLM 改不了。
    max_curiosity_events_per_tick: int = 2
    max_new_thoughts_per_tick: int = 1
    max_reactivation_requests: int = 1
    # 一次 Tick 最多顺着几个话题找线索。
    max_topic_expansion: int = 3
    # 低于这个分数就不要为了"显得自主"硬造念头。
    min_score: float = 0.25
    novelty_days: float = 14.0
    duplicate_threshold: float = 0.75
    # 评分权重与惩罚。
    w_novelty: float = 0.8
    w_uncertainty: float = 0.7
    w_importance: float = 0.9
    w_relevance: float = 0.5
    w_knowledge_gap: float = 0.8
    w_future: float = 0.2
    w_relationship: float = 0.2
    penalty_repetition: float = 0.6
    penalty_stale: float = 0.3
    penalty_cost: float = 0.1


@dataclass(slots=True)
class DiaryConfig:
    enabled: bool = True
    max_entry_chars: int = 2000


@dataclass(slots=True)
class BehaviorConfig:
    """行为限制，交给 BehaviorGuard 执行。"""

    allowed_actions: list[str] = field(
        default_factory=lambda: [
            "DO_NOTHING",
            "SEND_MESSAGE",
            "WAIT",
            "ENTER_SILENT",
            "THINK",
        ]
    )
    max_message_chars: int = 500
    min_seconds_between_messages: float = 3.0
    quiet_hours: list[str] = field(default_factory=list)  # 例：["23:30-07:30"]
    quiet_blocks_send_message: bool = True
    allow_send_on_autonomous_tick: bool = False
    # 主动联系总开关（在 hard limit 之外由 Guard 再确认一次）
    allow_autonomous_contact: bool = True
    max_silent_minutes: int = 720


@dataclass(slots=True)
class LoggingConfig:
    """日志配置。"""

    level: str = "INFO"
    file: str | None = "logs/agent.log"
    json_lines: bool = True
    console: bool = True
    # 轮转：单文件上限与保留份数
    max_bytes: int = 5 * 1024 * 1024
    backup_count: int = 3


@dataclass(slots=True)
class AppConfig:
    """应用总配置。"""

    agent_name: str = "自主思考系统"
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    qq: QQConfig = field(default_factory=QQConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    thought: ThoughtConfig = field(default_factory=ThoughtConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    contact: ContactConfig = field(default_factory=ContactConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    topic: TopicConfig = field(default_factory=TopicConfig)
    curiosity: CuriosityConfig = field(default_factory=CuriosityConfig)
    diary: DiaryConfig = field(default_factory=DiaryConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self, *, mask_secrets: bool = True) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        if mask_secrets and data["llm"]["api_key"]:
            data["llm"]["api_key"] = "***"
        return data

    def validate(self) -> None:
        """校验配置；不合法时抛 :class:`ConfigError`。"""
        _validate(self)


# --------------------------------------------------------------------------- #
# 环境变量覆盖表
# --------------------------------------------------------------------------- #
_ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "AGENT_LLM_PROVIDER": ("llm", "provider"),
    "AGENT_LLM_BASE_URL": ("llm", "base_url"),
    "AGENT_LLM_API_KEY": ("llm", "api_key"),
    "AGENT_LLM_MODEL": ("llm", "model"),
    "AGENT_LLM_TIMEOUT_SECONDS": ("llm", "request_timeout_seconds"),
    "AGENT_LLM_JSON_MODE": ("llm", "json_mode"),
    "AGENT_MAX_DAILY_TOKENS": ("budget", "max_daily_tokens"),
    "AGENT_BUDGET_STATE_FILE": ("budget", "state_file"),
    "AGENT_DB_PATH": ("database", "path"),
    "AGENT_LOG_LEVEL": ("logging", "level"),
    "AGENT_LOG_FILE": ("logging", "file"),
    "AGENT_QQ_ADAPTER": ("qq", "adapter"),
    "AGENT_SCHEDULER_ENABLED": ("scheduler", "enabled"),
    "AGENT_ACTIVE_TICK_SECONDS": ("scheduler", "active_tick_seconds"),
    "AGENT_THOUGHT_ENABLED": ("thought", "enabled"),
    "AGENT_WORKSPACE_DIR": ("thought", "workspace_dir"),
    "AGENT_CONTINUE_THINK_MINUTES": ("thought", "continue_think_minutes"),
    "AGENT_CONTACT_ENABLED": ("contact", "enabled"),
    "AGENT_CONTACT_HOURLY_LIMIT": ("contact", "hourly_limit"),
    "AGENT_CONTACT_DAILY_LIMIT": ("contact", "daily_limit"),
    "AGENT_CONTACT_TIMEOUT_SECONDS": ("contact", "timeout_seconds"),
    "AGENT_CONTACT_TOPIC_REPEAT_LIMIT": ("contact", "topic_repeat_limit"),
    "AGENT_CONTACT_SILENT_AFTER": ("contact", "silent_after_consecutive"),
    "AGENT_CONTACT_SILENT_MINUTES": ("contact", "silent_minutes"),
    "AGENT_ALLOW_AUTONOMOUS_CONTACT": ("behavior", "allow_autonomous_contact"),
    "AGENT_MEMORY_ENABLED": ("memory", "enabled"),
    "AGENT_EMOTION_ENABLED": ("emotion", "enabled"),
    "AGENT_EMOTION_MAX_DELTA": ("emotion", "max_emotion_delta_per_event"),
    "AGENT_TOPIC_ENABLED": ("topic", "enabled"),
    "AGENT_CURIOSITY_ENABLED": ("curiosity", "enabled"),
    "AGENT_CURIOSITY_MIN_SCORE": ("curiosity", "min_score"),
    "AGENT_DIARY_ENABLED": ("diary", "enabled"),
}

# DeepSeek 常用变量名，作为别名一起识别。
_ENV_ALIASES: dict[str, str] = {
    "DEEPSEEK_API_KEY": "AGENT_LLM_API_KEY",
    "OPENAI_API_KEY": "AGENT_LLM_API_KEY",
}


# --------------------------------------------------------------------------- #
# 加载
# --------------------------------------------------------------------------- #
def _coerce_primitive(value: Any, expected: Any, dotpath: str) -> Any:
    """对基础类型做严格校验；容器与嵌套 dataclass 交给递归处理。"""
    if expected is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{dotpath} 需要 bool，收到 {type(value).__name__}")
        return value
    if expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{dotpath} 需要 int，收到 {type(value).__name__}")
        return value
    if expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{dotpath} 需要 number，收到 {type(value).__name__}")
        return float(value)
    if expected is str:
        if not isinstance(value, str):
            raise ConfigError(f"{dotpath} 需要 str，收到 {type(value).__name__}")
        return value
    return value


def _build_dataclass(cls: type, data: Mapping[str, Any], dotpath: str) -> Any:
    if not isinstance(data, Mapping):
        raise ConfigError(f"{dotpath} 需要 object，收到 {type(data).__name__}")

    hints = get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"{dotpath} 存在未知配置项: {', '.join(sorted(unknown))}")

    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        expected = hints[f.name]
        path = f"{dotpath}.{f.name}"
        if dataclasses.is_dataclass(expected):
            kwargs[f.name] = _build_dataclass(expected, value, path)
        else:
            kwargs[f.name] = _coerce_primitive(value, expected, path)
    return cls(**kwargs)


def _apply_env(config: AppConfig, env: Mapping[str, str]) -> AppConfig:
    """返回被环境变量覆盖后的新配置对象。"""
    sections = {f.name: getattr(config, f.name) for f in dataclasses.fields(config)}
    changed = False

    for env_name, (section, field_name) in _ENV_OVERRIDES.items():
        raw = env.get(env_name)
        if raw is None and env_name in ("AGENT_LLM_API_KEY",):
            for alias, target in _ENV_ALIASES.items():
                if target == env_name and env.get(alias):
                    raw = env[alias]
                    break
        if raw is None:
            continue

        section_obj = sections[section]
        expected = get_type_hints(type(section_obj))[field_name]
        value: Any = raw
        if expected is bool:
            value = raw.strip().lower() in ("1", "true", "yes", "on")
        elif expected is int:
            try:
                value = int(raw)
            except ValueError as exc:
                raise ConfigError(f"环境变量 {env_name} 需要 int，收到 {raw!r}") from exc
        elif expected is float:
            try:
                value = float(raw)
            except ValueError as exc:
                raise ConfigError(f"环境变量 {env_name} 需要 number，收到 {raw!r}") from exc

        sections[section] = dataclasses.replace(section_obj, **{field_name: value})
        changed = True

    if not changed:
        return config

    rebuilt = dataclasses.replace(config, **sections)
    return rebuilt


def _resolve_paths(config: AppConfig, base_dir: Path) -> AppConfig:
    db_path = Path(config.database.path).expanduser()
    if not db_path.is_absolute():
        db_path = (base_dir / db_path).resolve()
    db_cfg = dataclasses.replace(config.database, path=str(db_path))

    budget_file = config.budget.state_file
    if budget_file:
        bp = Path(budget_file).expanduser()
        if not bp.is_absolute():
            bp = (base_dir / bp).resolve()
        budget_file = str(bp)
    else:
        # 默认跟库放一起，免得写库和记账落到不同目录。
        budget_file = str(db_path.with_name("token_budget.json"))
    budget_cfg = dataclasses.replace(config.budget, state_file=budget_file)

    log_file = config.logging.file
    if log_file:
        lp = Path(log_file).expanduser()
        if not lp.is_absolute():
            lp = (base_dir / lp).resolve()
        log_file = str(lp)
    log_cfg = dataclasses.replace(config.logging, file=log_file)

    workspace = Path(config.thought.workspace_dir).expanduser()
    if not workspace.is_absolute():
        workspace = (base_dir / workspace).resolve()
    thought_cfg = dataclasses.replace(config.thought, workspace_dir=str(workspace))

    return dataclasses.replace(
        config,
        database=db_cfg,
        budget=budget_cfg,
        logging=log_cfg,
        thought=thought_cfg,
    )


def default_config_path() -> Path:
    """默认配置文件位置（项目根目录下的 config.json）。"""
    return Path(__file__).resolve().parent / "config.json"


# 常用厂商只差 base_url 与默认模型，配置里写个名字就够了。
_LLM_PRESETS: dict[str, tuple[str, str]] = {
    "deepseek": ("https://api.deepseek.com/v1", "deepseek-chat"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini"),
}
_LLM_DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
_LLM_DEFAULT_MODEL = "deepseek-chat"


def _apply_llm_presets(config: AppConfig) -> AppConfig:
    """按 provider 填入 base_url / model 的默认值。

    只替换仍是内置默认值的字段，用户显式写过的地址和模型不动。
    """
    preset = _LLM_PRESETS.get(config.llm.provider)
    if preset is None:
        return config
    base_url, model = preset
    changes: dict[str, Any] = {}
    if config.llm.base_url in ("", _LLM_DEFAULT_BASE_URL):
        changes["base_url"] = base_url
    if config.llm.model in ("", _LLM_DEFAULT_MODEL):
        changes["model"] = model
    if not changes:
        return config
    return dataclasses.replace(config, llm=dataclasses.replace(config.llm, **changes))


def load_config(
    path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> AppConfig:
    """加载配置。

    ``path`` 为空时依次尝试：环境变量 AGENT_CONFIG、项目根 config.json、内置默认值。
    显式传入的路径不存在会抛 ConfigError，避免静默用错配置。
    """
    env = os.environ if env is None else env
    explicit = path is not None
    if path is None:
        path = env.get("AGENT_CONFIG") or default_config_path()

    config_path = Path(path).expanduser()
    base_dir = config_path.resolve().parent

    raw: dict[str, Any] = {}
    if config_path.exists():
        try:
            text = config_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"无法读取配置文件 {config_path}: {exc}") from exc
        if text.strip():
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"配置文件不是合法 JSON（{config_path}）: {exc}") from exc
            if not isinstance(raw, dict):
                raise ConfigError(f"配置文件顶层必须是 object（{config_path}）")
    elif explicit:
        raise ConfigError(f"配置文件不存在: {config_path}")
    else:
        base_dir = Path.cwd()

    config = _build_dataclass(AppConfig, raw, "config") if raw else AppConfig()
    config = _apply_env(config, env)
    config = _apply_llm_presets(config)
    config = _resolve_paths(config, base_dir)
    _validate(config)
    return config


def _validate(config: AppConfig) -> None:
    if config.llm.provider not in ("mock", "deepseek", "openai", "openai_compatible"):
        raise ConfigError(
            f"llm.provider 不支持 {config.llm.provider!r}，"
            "可选 mock / deepseek / openai / openai_compatible"
        )
    if config.llm.request_timeout_seconds <= 0:
        raise ConfigError("llm.request_timeout_seconds 必须大于 0")
    if config.llm.max_output_tokens <= 0:
        raise ConfigError("llm.max_output_tokens 必须大于 0")
    if config.budget.max_daily_tokens < 0:
        raise ConfigError("budget.max_daily_tokens 不能为负（0 表示不限）")
    if config.llm.provider != "mock":
        if not config.llm.base_url.strip():
            raise ConfigError("llm.base_url 不能为空")
        if not config.llm.model.strip():
            raise ConfigError("llm.model 不能为空")
    if config.qq.adapter not in ("mock",):
        raise ConfigError(f"qq.adapter 第一阶段只支持 mock，收到 {config.qq.adapter!r}")
    if config.qq.poll_interval_seconds <= 0:
        raise ConfigError("qq.poll_interval_seconds 必须大于 0")
    if config.database.event_retention_days < 0:
        raise ConfigError("database.event_retention_days 不能为负")
    if config.database.metrics_retention_days < 0:
        raise ConfigError("database.metrics_retention_days 不能为负")
    if config.scheduler.autonomous_tick_seconds <= 0:
        raise ConfigError("scheduler.autonomous_tick_seconds 必须大于 0")
    if config.scheduler.active_tick_seconds <= 0:
        raise ConfigError("scheduler.active_tick_seconds 必须大于 0")
    if config.scheduler.active_tick_seconds > config.scheduler.autonomous_tick_seconds:
        raise ConfigError("scheduler.active_tick_seconds 不能大于 autonomous_tick_seconds")
    if config.scheduler.max_wait_seconds <= 0:
        raise ConfigError("scheduler.max_wait_seconds 必须大于 0")
    if config.behavior.max_message_chars <= 0:
        raise ConfigError("behavior.max_message_chars 必须大于 0")
    if config.behavior.max_silent_minutes <= 0:
        raise ConfigError("behavior.max_silent_minutes 必须大于 0")
    if config.behavior.min_seconds_between_messages < 0:
        raise ConfigError("behavior.min_seconds_between_messages 不能为负")
    for item in config.behavior.allowed_actions:
        if not isinstance(item, str) or not item:
            raise ConfigError("behavior.allowed_actions 只能包含非空字符串")
    for window in config.behavior.quiet_hours:
        parse_quiet_window(window)

    thought = config.thought
    if not thought.workspace_dir or not str(thought.workspace_dir).strip():
        raise ConfigError("thought.workspace_dir 不能为空")
    for name in (
        "max_active_thoughts",
        "max_thoughts",
        "max_new_questions",
        "history_limit",
        "memory_limit",
    ):
        if getattr(thought, name) <= 0:
            raise ConfigError(f"thought.{name} 必须大于 0")
    for name in (
        "w_importance",
        "w_curiosity",
        "w_relevance",
        "w_novelty",
        "w_unresolved",
        "w_relationship",
        "w_future",
        "w_time_suitability",
        "penalty_repetition",
        "penalty_stale",
        "penalty_cost",
    ):
        if getattr(thought, name) < 0:
            raise ConfigError(f"thought.{name} 不能为负")
    if thought.novelty_revisits <= 0:
        raise ConfigError("thought.novelty_revisits 必须大于 0")
    if thought.stale_days <= 0:
        raise ConfigError("thought.stale_days 必须大于 0")
    if thought.forget_after_days <= 0 or thought.forget_inactivity_days <= 0:
        raise ConfigError("thought.forget_*_days 必须大于 0")
    if thought.forget_threshold <= 0:
        raise ConfigError("thought.forget_threshold 必须大于 0")
    if not 0.0 <= thought.reactivation_threshold <= 1.0:
        raise ConfigError("thought.reactivation_threshold 必须落在 [0, 1]")
    if thought.reactivation_cooldown_seconds < 0:
        raise ConfigError("thought.reactivation_cooldown_seconds 不能为负")
    if thought.max_reactivations < 0:
        raise ConfigError("thought.max_reactivations 不能为负")
    if thought.maintenance_every_ticks < 0:
        raise ConfigError("thought.maintenance_every_ticks 不能为负")
    if thought.max_thought_depth <= 0:
        raise ConfigError("thought.max_thought_depth 必须大于 0")
    if thought.max_followups_per_thought <= 0:
        raise ConfigError("thought.max_followups_per_thought 必须大于 0")
    if thought.continue_think_minutes <= 0:
        raise ConfigError("thought.continue_think_minutes 必须大于 0")
    if thought.thinking_text_max_chars < 1000:
        raise ConfigError("thought.thinking_text_max_chars 不能小于 1000")
    if thought.history_keep_turns <= 0:
        raise ConfigError("thought.history_keep_turns 必须大于 0")

    memory = config.memory
    if not 0.0 <= memory.duplicate_threshold <= 1.0:
        raise ConfigError("memory.duplicate_threshold 必须落在 [0, 1]")
    if memory.retrieval_limit <= 0 or memory.max_memories <= 0:
        raise ConfigError("memory.retrieval_limit / max_memories 必须大于 0")
    for name in (
        "w_importance",
        "w_confidence",
        "w_relevance",
        "w_future",
        "w_relationship",
        "penalty_stale",
        "penalty_duplicate",
        "penalty_low_confidence",
        "stale_days",
    ):
        if getattr(memory, name) < 0:
            raise ConfigError(f"memory.{name} 不能为负")
    for name in (
        "duplicate_threshold",
        "low_value_threshold",
        "soft_delete_threshold",
        "expire_threshold",
        "consolidate_threshold",
        "reactivate_threshold",
        "reactivate_min_value",
    ):
        if not 0.0 <= getattr(memory, name) <= 1.0:
            raise ConfigError(f"memory.{name} 必须落在 [0, 1]")
    if not (memory.expire_threshold <= memory.soft_delete_threshold <= memory.low_value_threshold):
        raise ConfigError(
            "memory 状态阈值必须满足 expire_threshold <= soft_delete_threshold <= low_value_threshold"
        )
    if memory.max_candidates_per_think < 0:
        raise ConfigError("memory.max_candidates_per_think 不能为负")
    if memory.max_reactivations < 1:
        raise ConfigError("memory.max_reactivations 必须大于 0")
    if memory.reactivation_cooldown_seconds < 0:
        raise ConfigError("memory.reactivation_cooldown_seconds 不能为负")
    if memory.consolidate_every_ticks < 0:
        raise ConfigError("memory.consolidate_every_ticks 不能为负")

    contact = config.contact
    for name in ("hourly_limit", "daily_limit", "max_attempts_per_thought"):
        if getattr(contact, name) < 0:
            raise ConfigError(f"contact.{name} 不能为负")
    for name in (
        "minimum_interval_seconds",
        "cooldown_seconds",
        "consecutive_no_reply_limit",
    ):
        if getattr(contact, name) < 0:
            raise ConfigError(f"contact.{name} 不能为负")
    if contact.daily_limit and contact.hourly_limit > contact.daily_limit:
        raise ConfigError("contact.hourly_limit 不能大于 contact.daily_limit")
    if not 0.0 <= contact.duplicate_threshold <= 1.0:
        raise ConfigError("contact.duplicate_threshold 必须落在 [0, 1]")
    if contact.duplicate_window <= 0:
        raise ConfigError("contact.duplicate_window 必须大于 0")
    if contact.timeout_seconds < 60:
        raise ConfigError("contact.timeout_seconds 不能小于 60 秒")
    if contact.topic_repeat_limit < 1:
        raise ConfigError("contact.topic_repeat_limit 必须大于 0")
    if contact.topic_repeat_window_minutes <= 0:
        raise ConfigError("contact.topic_repeat_window_minutes 必须大于 0")
    if contact.anomaly_block_threshold < 1:
        raise ConfigError("contact.anomaly_block_threshold 必须大于 0")
    if contact.anomaly_window_minutes <= 0:
        raise ConfigError("contact.anomaly_window_minutes 必须大于 0")
    if contact.silent_after_consecutive < 1:
        raise ConfigError("contact.silent_after_consecutive 必须大于 0")
    if contact.silent_minutes <= 0:
        raise ConfigError("contact.silent_minutes 必须大于 0")
    for window in contact.night_hours:
        parse_quiet_window(window, label="contact.night_hours")

    if not 0.0 <= config.emotion.baseline <= 1.0:
        raise ConfigError("emotion.baseline 必须落在 [0, 1]")
    if not 0.0 <= config.emotion.decay <= 1.0:
        raise ConfigError("emotion.decay 必须落在 [0, 1]")
    if not 0.0 < config.emotion.max_emotion_delta_per_event <= 1.0:
        raise ConfigError("emotion.max_emotion_delta_per_event 必须落在 (0, 1]")

    topic = config.topic
    if not 0.0 <= topic.duplicate_threshold <= 1.0:
        raise ConfigError("topic.duplicate_threshold 必须落在 [0, 1]")
    if not 0.0 <= topic.min_importance_to_create <= 1.0:
        raise ConfigError("topic.min_importance_to_create 必须落在 [0, 1]")
    for name in ("w_recency", "w_activity", "w_importance"):
        if getattr(topic, name) < 0:
            raise ConfigError(f"topic.{name} 不能为负")
    if topic.low_activity_days < 0 or topic.dormant_days < 0 or topic.archive_days < 0:
        raise ConfigError("topic 的活跃度天数不能为负")
    if not (topic.low_activity_days <= topic.dormant_days <= topic.archive_days):
        raise ConfigError(
            "topic 天数必须满足 low_activity_days <= dormant_days <= archive_days"
        )
    if topic.max_context_topics <= 0:
        raise ConfigError("topic.max_context_topics 必须大于 0")

    curiosity = config.curiosity
    for name in (
        "max_curiosity_events_per_tick",
        "max_new_thoughts_per_tick",
        "max_reactivation_requests",
        "max_topic_expansion",
        "novelty_days",
        "w_novelty",
        "w_uncertainty",
        "w_importance",
        "w_relevance",
        "w_knowledge_gap",
        "w_future",
        "w_relationship",
        "penalty_repetition",
        "penalty_stale",
        "penalty_cost",
    ):
        if getattr(curiosity, name) < 0:
            raise ConfigError(f"curiosity.{name} 不能为负")
    if not 0.0 <= curiosity.min_score <= 1.0:
        raise ConfigError("curiosity.min_score 必须落在 [0, 1]")
    if not 0.0 <= curiosity.duplicate_threshold <= 1.0:
        raise ConfigError("curiosity.duplicate_threshold 必须落在 [0, 1]")

    if config.diary.max_entry_chars <= 0:
        raise ConfigError("diary.max_entry_chars 必须大于 0")


def parse_quiet_window(spec: str, *, label: str = "quiet_hours") -> tuple[int, int, int, int]:
    """把 ``HH:MM-HH:MM`` 解析成 (sh, sm, eh, em)，供 guard 复用。"""
    try:
        start, end = spec.split("-", 1)
        sh, sm = (int(x) for x in start.strip().split(":"))
        eh, em = (int(x) for x in end.strip().split(":"))
    except (ValueError, AttributeError) as exc:
        raise ConfigError(f"{label} 格式应为 HH:MM-HH:MM，收到 {spec!r}") from exc
    if not (0 <= sh < 24 and 0 <= sm < 60 and 0 <= eh < 24 and 0 <= em < 60):
        raise ConfigError(f"{label} 时间越界: {spec!r}")
    return sh, sm, eh, em