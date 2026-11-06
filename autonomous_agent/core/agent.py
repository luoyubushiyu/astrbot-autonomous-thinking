"""Agent 编排器：事件 → 上下文 → LLM → JSON → 校验 → 守卫 → 执行。

设计约束（第一阶段）：
* LLM 只产出建议，不接触数据库、文件、QQ API 或调度器；
* 任何 LLM 侧失败都安全回退到 DO_NOTHING；
* 任何程序侧异常都让 Agent 进入 SAFE_IDLE 并落库；
* 重启时把上次的瞬态状态恢复成 SAFE_IDLE，而不是继续假装在思考。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from ..config import AppConfig
from ..contact.intent import ContactIntent as ContactProposal
from ..contact.manager import ContactManager
from ..database.database import (
    Database,
    EventRepository,
    MessageRepository,
    MetricsRepository,
    StateRepository,
)
from ..database.migrations import initialize_database
from ..diary.repository import DiaryStore
from ..llm.adapter import LLMAdapter, LLMError, create_llm_adapter
from ..llm.budget import TokenBudget
from ..llm.schemas import Action, ActionType, Decision
from ..llm.validator import OutputValidator, ValidationContext
from ..memory.store import MemoryStore
from ..memory.topics import TopicStore
from ..qq.adapter import IncomingMessage, QQAdapter, create_qq_adapter
from ..thought.repository import ThoughtRepository
from ..thought.workspace import ThoughtWorkspace
from .context import ContextBuilder
from .emotion import EmotionModel, EmotionState
from .event import Event, EventType
from .executor import ExecutionResult, Executor
from .guard import BehaviorGuard, GuardContext, GuardVerdict
from .state import (
    AgentState,
    AgentStateMachine,
    InvalidTransitionError,
    StateSnapshot,
    ThoughtState,
    recover_state,
    utcnow,
)
from .thought_engine import ThinkResult, ThoughtEngine
from .timewindow import in_any_window

logger = logging.getLogger("autonomous_agent.agent")

_KEEP = object()

_MAINTENANCE_EVERY_TICKS = 20


def think_reason(result: ThinkResult) -> str:
    """给 ContactIntent 一个可读的原因，不参与硬限制判断。"""
    if result.outcome is not None and result.outcome.insight:
        return result.outcome.insight
    return ""


@dataclass(frozen=True, slots=True)
class HandleResult:
    event_id: str
    event_type: EventType
    state: AgentState
    decision: Decision | None = None
    verdict: GuardVerdict | None = None
    execution: ExecutionResult | None = None
    error: str | None = None


class Agent:
    def __init__(
        self,
        config: AppConfig,
        *,
        database: Database | None = None,
        llm: LLMAdapter | None = None,
        qq: QQAdapter | None = None,
        clock: Callable[[], datetime] = utcnow,
        history_limit: int = 10,
    ) -> None:
        self._config = config
        self._clock = clock
        self._db = database or Database(
            config.database.path, busy_timeout_ms=config.database.busy_timeout_ms
        )
        self._llm = llm or create_llm_adapter(config.llm)
        self._qq = qq or create_qq_adapter(config.qq)

        self._states = StateRepository(self._db)
        self._messages = MessageRepository(self._db)
        self._events = EventRepository(self._db)
        self._metrics = MetricsRepository(self._db)

        self._machine = AgentStateMachine(clock=clock)
        self._validator = OutputValidator()
        self._guard = BehaviorGuard(config.behavior, clock=clock)
        self._context = ContextBuilder(
            self._messages, history_limit=history_limit, clock=clock
        )
        self._executor = Executor(
            qq=self._qq,
            messages=self._messages,
            metrics=self._metrics,
            behavior=config.behavior,
            clock=clock,
        )

        self._workspace = ThoughtWorkspace(config.thought.workspace_dir)
        self._thoughts = ThoughtRepository(
            self._db,
            self._workspace,
            clock=clock,
            max_thoughts=config.thought.max_thoughts,
        )
        self._memories = MemoryStore(
            self._db,
            clock=clock,
            max_memories=config.memory.max_memories,
            duplicate_threshold=config.memory.duplicate_threshold,
            config=config.memory,
        )
        self._topics = TopicStore(self._db, clock=clock, config=config.topic)
        self._diary = DiaryStore(
            self._db,
            self._workspace.diary_dir,
            clock=clock,
            max_entry_chars=config.diary.max_entry_chars,
        )
        self._contact = ContactManager(self._db, config.contact, clock=clock)
        self._emotion_model = EmotionModel(config.emotion, clock=clock)
        self._budget = TokenBudget(
            daily_limit=config.budget.max_daily_tokens,
            state_path=config.budget.state_file or None,
            clock=clock,
        )
        self._engine = ThoughtEngine(
            config,
            repository=self._thoughts,
            memories=self._memories,
            topics=self._topics,
            diary=self._diary,
            contact=self._contact,
            messages=self._messages,
            emotion_model=self._emotion_model,
            metrics=self._metrics,
            clock=clock,
            budget=self._budget,
        )

        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=1000)
        self._stop_event = asyncio.Event()
        self._event_lock = asyncio.Lock()
        self._worker_task: asyncio.Task[None] | None = None
        self._tick_task: asyncio.Task[None] | None = None
        self._poll_task: asyncio.Task[None] | None = None
        self._wake_task: asyncio.Task[None] | None = None

        self._started = False
        self._background = True
        self._tick_index = 0
        self._last_outgoing_at: datetime | None = None
        self._recent_states: deque[dict[str, Any]] = deque(maxlen=50)
        self._announced_timeouts: set[str] = set()

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    @property
    def state(self) -> AgentState:
        return self._machine.state

    @property
    def running(self) -> bool:
        return self._started

    async def start(self, *, background: bool = True) -> None:
        if self._started:
            return
        try:
            await initialize_database(self._db)
            await self._restore_state()
            await self._persist_state()
            self._workspace.ensure_layout()
            if self._config.thought.enabled:
                try:
                    # repair 比 sync 多做两件事：导入"有文件没数据库"的 Thought，
                    # 把解析不了的文件隔离而不是直接删。
                    await self._thoughts.repair()
                except Exception:  # noqa: BLE001 - 文件修复失败不阻断启动
                    logger.warning("Thought 文件修复失败", exc_info=True)
            self._restore_emotion()
            await self._qq.start()

            self._started = True
            self._background = background
            self._stop_event.clear()

            await self.handle_event(Event.bot_start())

            if not background:
                return
            self._worker_task = asyncio.create_task(
                self._worker_loop(), name="agent-worker"
            )
            if self._config.scheduler.enabled:
                self._tick_task = asyncio.create_task(
                    self._tick_loop(), name="agent-tick"
                )
            if self._config.qq.enabled:
                self._poll_task = asyncio.create_task(
                    self._poll_loop(), name="agent-qq-poll"
                )
            await self._resume_silent_alarm()
        except BaseException:
            # 启动半途失败时不能留下开着的数据库连接或孤儿任务。
            self._started = False
            tasks = [
                task
                for task in (
                    self._worker_task,
                    self._tick_task,
                    self._poll_task,
                    self._wake_task,
                )
                if task is not None
            ]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self._worker_task = None
            self._tick_task = None
            self._poll_task = None
            self._wake_task = None
            try:
                await self._qq.stop()
            except Exception:  # noqa: BLE001 - 清理失败不掩盖原始异常
                logger.warning("启动失败后停止 QQ 适配器失败", exc_info=True)
            try:
                await self._db.close()
            except Exception:  # noqa: BLE001
                logger.warning("启动失败后关闭数据库失败", exc_info=True)
            raise

    async def stop(self) -> None:
        if not self._started:
            return
        self._stop_event.set()
        await self._cancel_task("_wake_task")
        tasks = [
            task
            for task in (self._worker_task, self._tick_task, self._poll_task)
            if task is not None
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._worker_task = None
        self._tick_task = None
        self._poll_task = None

        await self.handle_event(Event.bot_stop())
        await self._persist_state()
        await self._llm.aclose()
        await self._qq.stop()
        await self._db.close()
        self._started = False

    async def run(self) -> None:
        await self.start(background=True)
        try:
            await self._stop_event.wait()
        finally:
            await self.stop()

    def request_stop(self) -> None:
        """线程/信号安全的停止请求；由 ``run`` 消费。"""
        self._stop_event.set()

    # ------------------------------------------------------------------ #
    # 入口：提交与处理
    # ------------------------------------------------------------------ #
    async def submit_event(self, event: Event) -> None:
        if not self._started:
            raise RuntimeError("Agent 尚未启动")
        if not self._background:
            raise RuntimeError("Agent 以 background=False 启动，不支持事件队列")
        await self._queue.put(event)

    async def handle_event(self, event: Event) -> HandleResult:
        """串行处理一个事件（测试与 BOT_START/STOP 直接使用）。

        同一 Agent 上的事件处理彼此互斥：状态机与落库都不是可重入的，
        并发调用会把 OBSERVING/THINKING 之类的状态搅在一起。
        """
        async with self._event_lock:
            return await self._handle_event(event)

    async def _handle_event(self, event: Event) -> HandleResult:
        if self._config.database.store_events:
            try:
                await self._events.record(
                    event_id=event.id,
                    event_type=event.type.value,
                    source=event.source,
                    payload=event.payload,
                    created_at=event.occurred_at,
                )
            except Exception:  # noqa: BLE001 - 记录失败不阻断处理
                logger.warning("事件落库失败: %s", event.id, exc_info=True)

        status = "PROCESSED"
        try:
            result = await self._dispatch(event)
            if result.error:
                status = "FAILED"
        except asyncio.CancelledError:
            status = "SKIPPED"
            raise
        except Exception as exc:  # noqa: BLE001 - 兜底：进 SAFE_IDLE
            logger.exception("事件处理异常: %s", event.type.value)
            await self._safe_idle(f"unhandled:{type(exc).__name__}")
            result = HandleResult(
                event_id=event.id,
                event_type=event.type,
                state=self._machine.state,
                error=f"{type(exc).__name__}: {exc}",
            )
            status = "FAILED"

        if self._config.database.store_events:
            try:
                await self._events.mark(event.id, status)
            except Exception:  # noqa: BLE001
                logger.warning("事件状态更新失败: %s", event.id, exc_info=True)
        return result

    async def _dispatch(self, event: Event) -> HandleResult:
        if event.type is EventType.BOT_START:
            return await self._on_bot_start(event)
        if event.type is EventType.BOT_STOP:
            return await self._on_bot_stop(event)
        if event.type is EventType.USER_MESSAGE:
            return await self._on_user_message(event)
        if event.type is EventType.AUTONOMOUS_TICK:
            return await self._on_autonomous_tick(event)
        if event.type is EventType.ALARM_TRIGGERED:
            return await self._on_alarm(event)
        if event.type is EventType.CONTACT_TIMEOUT:
            return await self._on_contact_timeout(event)
        raise ValueError(f"未支持的事件类型: {event.type}")

    # ------------------------------------------------------------------ #
    # 事件处理器
    # ------------------------------------------------------------------ #
    async def _on_bot_start(self, event: Event) -> HandleResult:
        await self._metric("bot.started", 1, {"state": self._machine.state.value})
        await self._load_last_outgoing()
        return HandleResult(
            event_id=event.id, event_type=event.type, state=self._machine.state
        )

    async def _on_bot_stop(self, event: Event) -> HandleResult:
        await self._metric("bot.stopped", 1, {"state": self._machine.state.value})
        if self._machine.state in (AgentState.THINKING, AgentState.CONTACTING):
            await self._safe_idle("bot_stop_during_transient")
        return HandleResult(
            event_id=event.id, event_type=event.type, state=self._machine.state
        )

    async def _on_user_message(self, event: Event) -> HandleResult:
        try:
            incoming = IncomingMessage.from_payload(event.payload)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"USER_MESSAGE 字段不完整: {exc}") from exc

        conversation_id = incoming.conversation_id
        self._goto(
            AgentState.OBSERVING,
            "收到用户消息",
            conversation_id=conversation_id,
        )

        inserted = await self._messages.add(
            conversation_id=conversation_id,
            message_id=incoming.message_id,
            role="user",
            content=incoming.content,
            user_id=incoming.user_id,
            created_at=incoming.received_at,
            metadata={"is_group": incoming.is_group, "group_id": incoming.group_id},
        )
        if not inserted:
            logger.info("重复消息已忽略: %s", incoming.message_id)
            self._goto(AgentState.IDLE, "重复消息", conversation_id=conversation_id)
            await self._persist_state()
            return HandleResult(
                event_id=event.id, event_type=event.type, state=self._machine.state
            )

        replied = await self._contact.mark_replied(conversation_id)
        if replied:
            await self._metric("contact.replied", float(replied))
        try:
            await self._engine.ingest_user_message(
                conversation_id=conversation_id,
                user_id=incoming.user_id,
                content=incoming.content,
            )
        except Exception:  # noqa: BLE001 - 记忆/想法失败不该拖垮回复
            logger.warning("摄取用户消息失败", exc_info=True)
        await self._persist_emotion()

        decision, verdict, execution, error = await self._think_and_act(
            event, conversation_id=conversation_id
        )
        return await self._settle(
            event,
            decision=decision,
            verdict=verdict,
            execution=execution,
            error=error,
            conversation_id=conversation_id,
        )

    async def _on_autonomous_tick(self, event: Event) -> HandleResult:
        self._tick_index = max(
            self._tick_index, int(event.payload.get("tick", self._tick_index + 1))
        )
        if self._machine.state is AgentState.WAITING:
            await self._metric("tick.skipped", 1, {"state": self._machine.state.value})
            return HandleResult(
                event_id=event.id, event_type=event.type, state=self._machine.state
            )

        # 夜间：IDLE → NIGHT → SLEEPING，天亮后自动醒回 IDLE。
        night = in_any_window(self._config.contact.night_hours, self._clock())
        state = self._machine.state
        if state in (AgentState.NIGHT, AgentState.SLEEPING):
            if night:
                if state is AgentState.NIGHT:
                    self._goto(AgentState.SLEEPING, "夜深了，进入睡眠", conversation_id=None)
                    await self._persist_state()
                    await self._metric("state.sleeping", 1)
                await self._metric("tick.skipped", 1, {"state": self._machine.state.value})
                return HandleResult(
                    event_id=event.id, event_type=event.type, state=self._machine.state
                )
            self._goto(AgentState.IDLE, "夜间结束，自然醒来", conversation_id=None)
            await self._persist_state()
            await self._metric("state.wake", 1, {"from": state.value})
            state = self._machine.state
        elif night and state in (AgentState.IDLE, AgentState.SAFE_IDLE):
            self._goto(AgentState.NIGHT, "进入夜间模式", conversation_id=None)
            await self._persist_state()
            await self._metric("state.night", 1)
            return HandleResult(
                event_id=event.id,
                event_type=event.type,
                state=self._machine.state,
                decision=Decision.do_nothing("夜间休息，不主动思考"),
            )

        # 连续主动联系都没人回：降低 tendency，自动进入 SILENT。
        # SILENT 只停联系，不停思考。
        if state in (AgentState.IDLE, AgentState.SAFE_IDLE):
            unanswered = await self._contact.unanswered_count()
            if unanswered >= self._config.contact.silent_after_consecutive:
                await self._enter_silent("consecutive_no_reply")
                return HandleResult(
                    event_id=event.id,
                    event_type=event.type,
                    state=self._machine.state,
                    decision=Decision.do_nothing("连续没有回复，进入静默"),
                )

        silent = self._machine.state is AgentState.SILENT
        result = await self._engine.step(
            self._llm,
            conversation_id=self._machine.conversation_id,
            silent=silent,
        )
        await self._persist_emotion()
        if result.error:
            await self._metric("thought.error", 1, {"code": result.error})
            logger.warning("自主思考失败: %s", result.error)
            return HandleResult(
                event_id=event.id,
                event_type=event.type,
                state=self._machine.state,
                error=result.error,
            )
        if result.contact is not None:
            if silent:
                return await self._block_contact_in_silent(event, result)
            return await self._send_contact(event, result)

        decision = Decision.do_nothing(
            reason=result.skipped_reason or "这一轮没有值得处理的想法"
        )
        if self._machine.state is not AgentState.SILENT:
            self._goto(
                AgentState.IDLE, "自主心跳结束", conversation_id=event.conversation_id()
            )
        await self._persist_state()
        return HandleResult(
            event_id=event.id,
            event_type=event.type,
            state=self._machine.state,
            decision=decision,
        )

    async def _send_contact(self, event: Event, result: ThinkResult) -> HandleResult:
        """把"想联系"变成一次真正的主动发送；三道闸门都过才发。"""
        intent = result.contact
        thought = result.thought
        if intent is None or thought is None:
            return HandleResult(
                event_id=event.id, event_type=event.type, state=self._machine.state
            )
        topic = thought.related_topics[0] if thought.related_topics else ""
        proposal = ContactProposal.from_thought(
            thought,
            conversation_id=intent.conversation_id,
            message_draft=intent.message,
            topic=topic,
            reason=think_reason(result),
        )
        self._goto(AgentState.CONTACTING, "准备主动联系", conversation_id=intent.conversation_id)
        await self._persist_state()

        known = await self._messages.conversation_ids()
        known.add(intent.conversation_id)
        guard_verdict = self._guard.check_contact(
            proposal,
            GuardContext(
                event_type=event.type,
                conversation_id=intent.conversation_id,
                known_conversation_ids=frozenset(known),
                agent_state=self._machine.state,
                last_outgoing_at=self._last_outgoing_at,
                now=self._clock(),
            ),
        )
        if not guard_verdict.allowed:
            return await self._record_blocked_contact(
                event, proposal, thought, guard_verdict.code, guard_verdict.reason
            )

        decision = await self._contact.evaluate(
            thought,
            proposal.conversation_id,
            proposal.message_draft,
            topic=proposal.topic,
            silent=self._machine.state is AgentState.SILENT,
            now=self._clock(),
        )
        if not decision.allowed:
            return await self._record_blocked_contact(
                event, proposal, thought, decision.code, decision.detail
            )

        action = Action(
            ActionType.SEND_MESSAGE,
            {
                "conversation_id": proposal.conversation_id,
                "content": proposal.message_draft,
            },
        )
        execution = await self._executor.execute(action, None)
        if execution.error:
            await self._contact.record(
                thought,
                proposal.conversation_id,
                proposal.message_draft,
                status="DENIED",
                reason=execution.error,
                intent_id=proposal.intent_id,
                topic=proposal.topic,
                blocked_reason=execution.error,
            )
            await self._metric("contact.send_failed", 1, {"code": execution.error})
            self._goto(AgentState.IDLE, "主动联系发送失败", conversation_id=proposal.conversation_id)
            await self._persist_state()
            return HandleResult(
                event_id=event.id,
                event_type=event.type,
                state=self._machine.state,
                execution=execution,
            )

        await self._contact.record(
            thought,
            proposal.conversation_id,
            proposal.message_draft,
            status="SENT",
            intent_id=proposal.intent_id,
            topic=proposal.topic,
        )
        self._last_outgoing_at = self._clock()
        await self._metric("contact.sent", 1, {"thought": thought.id})
        self._goto(AgentState.IDLE, "主动联系完成", conversation_id=proposal.conversation_id)
        await self._persist_state()
        return HandleResult(
            event_id=event.id,
            event_type=event.type,
            state=self._machine.state,
            decision=Decision(
                reasoning=result.outcome.insight if result.outcome else "主动联系",
                action=action,
                confidence=result.outcome.confidence if result.outcome else 0.0,
            ),
            execution=execution,
        )

    async def _record_blocked_contact(
        self,
        event: Event,
        proposal: ContactProposal,
        thought: Any,
        code: str,
        detail: str,
    ) -> HandleResult:
        await self._contact.record(
            thought,
            proposal.conversation_id,
            proposal.message_draft,
            status="DENIED",
            reason=code,
            intent_id=proposal.intent_id,
            topic=proposal.topic,
            blocked_reason=code,
        )
        await self._metric("contact.blocked", 1, {"code": code})
        logger.info("主动联系被拦截(%s): %s", code, detail)
        if self._machine.state is AgentState.SILENT:
            # SILENT 中拦截联系后继续静默，不切走状态。
            await self._persist_state()
            return HandleResult(
                event_id=event.id,
                event_type=event.type,
                state=self._machine.state,
                decision=Decision.do_nothing(f"静默中不联系：{detail}"),
            )
        self._goto(AgentState.IDLE, "主动联系被拒绝", conversation_id=proposal.conversation_id)
        await self._persist_state()
        return HandleResult(
            event_id=event.id,
            event_type=event.type,
            state=self._machine.state,
            decision=Decision.do_nothing(f"行为被拒绝：{detail}"),
            verdict=GuardVerdict.deny(code, detail),
        )

    async def _block_contact_in_silent(
        self, event: Event, result: ThinkResult
    ) -> HandleResult:
        intent = result.contact
        thought = result.thought
        if intent is None or thought is None:
            return HandleResult(
                event_id=event.id, event_type=event.type, state=self._machine.state
            )
        topic = thought.related_topics[0] if thought.related_topics else ""
        proposal = ContactProposal.from_thought(
            thought,
            conversation_id=intent.conversation_id,
            message_draft=intent.message,
            topic=topic,
            reason=think_reason(result),
        )
        return await self._record_blocked_contact(
            event, proposal, thought, "SILENT_MODE", "静默状态下不主动联系"
        )

    async def _enter_silent(self, reason: str) -> None:
        """自动进入 SILENT：只少说话，不间断思考。"""
        now = self._clock()
        minutes = self._config.contact.silent_minutes
        until = now + timedelta(minutes=minutes)
        alarm_id = f"silent:{uuid.uuid4().hex}"
        self._goto(
            AgentState.SILENT,
            f"静默：{reason}",
            conversation_id=self._machine.conversation_id,
            metadata={
                "silent_until": until.isoformat(),
                "alarm_id": alarm_id,
                "silent_reason": reason,
            },
        )
        self._schedule_alarm(
            minutes * 60.0, alarm_id=alarm_id, reason="silent_elapsed"
        )
        await self._persist_state()
        await self._metric("contact.silent_entered", 1, {"reason": reason})

    async def _on_alarm(self, event: Event) -> HandleResult:
        fired = event.payload.get("alarm_id")
        await self._metric("alarm.triggered", 1, {"alarm_id": fired})
        if self._machine.state in (
            AgentState.WAITING,
            AgentState.SILENT,
            AgentState.SLEEPING,
        ):
            expected = self._machine.metadata.get("alarm_id")
            if expected and fired != expected:
                # 旧等待/静默留下的闹钟晚了半步，不能清掉新的等待。
                await self._metric("alarm.stale", 1, {"alarm_id": fired})
                logger.info("忽略过期闹钟: %s（当前 %s）", fired, expected)
                return HandleResult(
                    event_id=event.id,
                    event_type=event.type,
                    state=self._machine.state,
                )
            self._goto(AgentState.IDLE, "闹钟解除等待/静默", metadata={})
            await self._persist_state()
        return HandleResult(
            event_id=event.id, event_type=event.type, state=self._machine.state
        )

    async def _on_contact_timeout(self, event: Event) -> HandleResult:
        await self._metric("contact.timeout", 1)
        if self._machine.state is AgentState.WAITING:
            self._goto(AgentState.IDLE, "联系超时")
            await self._persist_state()
        return HandleResult(
            event_id=event.id, event_type=event.type, state=self._machine.state
        )

    async def _due_contact_timeout(self) -> Event | None:
        """有一条主动联系等太久没人回，就产出一个 CONTACT_TIMEOUT 事件。

        同一条联系只报一次；回复后 ``mark_replied`` 会把它排除在查询外。
        """
        pending = await self._contact.due_timeout(
            self._clock(), self._config.contact.timeout_seconds
        )
        if pending is None:
            return None
        attempt_id, conversation_id, waited = pending
        if attempt_id in self._announced_timeouts:
            return None
        self._announced_timeouts.add(attempt_id)
        if len(self._announced_timeouts) > 500:
            self._announced_timeouts.clear()
        return Event.contact_timeout(
            conversation_id=conversation_id, waited_seconds=waited
        )

    async def _check_contact_timeouts(self) -> None:
        event = await self._due_contact_timeout()
        if event is None:
            return
        await self._metric("contact.timeout_event", 1)
        await self.submit_event(event)

    # ------------------------------------------------------------------ #
    # 思考 → 校验 → 守卫 → 执行
    # ------------------------------------------------------------------ #
    async def _think_and_act(
        self, event: Event, *, conversation_id: str | None
    ) -> tuple[Decision, GuardVerdict, ExecutionResult | None, str | None]:
        self._goto(AgentState.THINKING, "开始思考", conversation_id=conversation_id)
        await self._persist_state()

        context = await self._context.build(
            event, self._machine, conversation_id=conversation_id
        )
        request = self._context.build_request(context)

        raw: str | None = None
        error: str | None = None
        if not self._budget.allow():
            # 预算用完就不问了，直接走安全回退。
            error = "BUDGET_EXHAUSTED"
            await self._metric(
                "llm.budget_skipped", 1, {"route": "reply", "event": event.type.value}
            )
            logger.warning("今日 token 预算已用完，跳过这次回复")
        else:
            try:
                response = await asyncio.wait_for(
                    self._llm.complete(request),
                    timeout=self._config.llm.request_timeout_seconds,
                )
                raw = response.text
                await self._metric(
                    "llm.latency",
                    response.elapsed_seconds,
                    {"provider": response.provider, "model": response.model},
                )
                await self._record_tokens(request, response, route="reply")
            except asyncio.TimeoutError:
                error = "LLM_TIMEOUT"
                await self._metric("llm.timeout", 1, {"event": event.type.value})
                logger.warning("LLM 调用超时，安全回退")
            except LLMError as exc:
                error = f"LLM_ERROR:{type(exc).__name__}"
                await self._metric("llm.error", 1, {"code": type(exc).__name__})
                logger.warning("LLM 调用失败: %s", exc)

        decision: Decision | None = None
        if error is None:
            vctx = ValidationContext(
                event_type=event.type.value,
                conversation_id=conversation_id,
                known_conversation_ids=context.known_conversation_ids,
                max_message_chars=self._config.behavior.max_message_chars,
                allowed_actions=frozenset(
                    item
                    for item in self._config.behavior.allowed_actions
                    if item.upper() != ActionType.THINK.value
                ),
            )
            decision, verr = self._validator.safe_validate(raw or "", vctx)
            if verr is not None:
                error = verr.code
                await self._metric(
                    "llm.invalid_output", 1, {"code": verr.code}
                )
                logger.warning("LLM 输出不合格(%s): %s", verr.code, verr.detail)

        if decision is None:
            decision = Decision.do_nothing(f"安全回退：{error or 'UNKNOWN'}")

        verdict = self._guard.check(
            decision.action,
            GuardContext(
                event_type=event.type,
                conversation_id=conversation_id,
                known_conversation_ids=context.known_conversation_ids,
                agent_state=self._machine.state,
                last_outgoing_at=self._last_outgoing_at,
                now=self._clock(),
            ),
        )
        if not verdict.allowed:
            await self._metric(
                "guard.denied",
                1,
                {"code": verdict.code, "action": decision.action.type.value},
            )
            logger.info("行为被拒绝(%s): %s", verdict.code, verdict.reason)
            decision = Decision.do_nothing(f"行为被拒绝：{verdict.reason}")

        execution = await self._executor.execute(decision.action, context)
        return decision, verdict, execution, error

    async def _settle(
        self,
        event: Event,
        *,
        decision: Decision | None,
        verdict: GuardVerdict | None,
        execution: ExecutionResult | None,
        error: str | None,
        conversation_id: str | None,
    ) -> HandleResult:
        if error:
            await self._safe_idle(f"思考失败：{error}")
            return HandleResult(
                event_id=event.id,
                event_type=event.type,
                state=self._machine.state,
                decision=decision,
                verdict=verdict,
                execution=execution,
                error=error,
            )

        if execution is not None and execution.error:
            await self._safe_idle(f"执行失败：{execution.error}")
            return HandleResult(
                event_id=event.id,
                event_type=event.type,
                state=self._machine.state,
                decision=decision,
                verdict=verdict,
                execution=execution,
                error=execution.error,
            )

        action_type = (
            decision.action.type if decision is not None else ActionType.DO_NOTHING
        )
        now = self._clock()
        if action_type is ActionType.SEND_MESSAGE:
            self._last_outgoing_at = now
            self._goto(
                AgentState.IDLE,
                "回复完成",
                conversation_id=conversation_id,
            )
        elif action_type is ActionType.WAIT:
            seconds = (execution.wait_seconds if execution else 1) or 1
            wait_until = now + timedelta(seconds=seconds)
            alarm_id = f"wait:{event.id}"
            self._goto(
                AgentState.WAITING,
                f"等待 {seconds}s",
                conversation_id=conversation_id,
                metadata={
                    "wait_until": wait_until.isoformat(),
                    "alarm_id": alarm_id,
                },
            )
            self._schedule_alarm(seconds, alarm_id=alarm_id, reason="wait_elapsed")
        elif action_type is ActionType.ENTER_SILENT:
            until = execution.silent_until if execution else None
            until = until or (now + timedelta(minutes=1))
            delay = max(1.0, (until - now).total_seconds())
            alarm_id = f"silent:{event.id}"
            self._goto(
                AgentState.SILENT,
                "进入静默",
                conversation_id=conversation_id,
                metadata={
                    "silent_until": until.isoformat(),
                    "alarm_id": alarm_id,
                },
            )
            self._schedule_alarm(delay, alarm_id=alarm_id, reason="silent_elapsed")
        else:
            self._goto(
                AgentState.IDLE,
                "无需行动",
                conversation_id=conversation_id,
            )

        self._machine.set_thought_state(ThoughtState.INACTIVE)
        await self._persist_state()
        return HandleResult(
            event_id=event.id,
            event_type=event.type,
            state=self._machine.state,
            decision=decision,
            verdict=verdict,
            execution=execution,
            error=None,
        )

    # ------------------------------------------------------------------ #
    # 长期维护
    # ------------------------------------------------------------------ #
    async def _run_maintenance(self) -> None:
        """低频清理：事件/指标保留期、Thought 文件修复、SQLite 优化。

        失败只记日志，不影响主循环。
        """
        now = self._clock()
        event_cutoff = (
            now - timedelta(days=self._config.database.event_retention_days)
        ).isoformat()
        metrics_cutoff = (
            now - timedelta(days=self._config.database.metrics_retention_days)
        ).isoformat()
        try:
            await self._db.execute(
                "DELETE FROM events WHERE status != 'PENDING' AND created_at < ?",
                (event_cutoff,),
            )
            await self._db.execute(
                "DELETE FROM metrics WHERE recorded_at < ?", (metrics_cutoff,)
            )
            await self._db.execute("PRAGMA optimize")
        except Exception:  # noqa: BLE001
            logger.warning("数据库维护失败", exc_info=True)
        try:
            report = await self._thoughts.repair()
        except Exception:  # noqa: BLE001
            logger.warning("Thought 文件修复失败", exc_info=True)
        else:
            if any(int(v) for v in report.values()):
                await self._metric(
                    "maintenance.repair",
                    float(sum(int(v) for v in report.values())),
                )
        await self._metric("maintenance.run", 1)

    # ------------------------------------------------------------------ #
    # 后台循环
    # ------------------------------------------------------------------ #
    async def _worker_loop(self) -> None:
        while True:
            try:
                event = await self._queue.get()
            except asyncio.CancelledError:
                return
            try:
                await self.handle_event(event)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - handle_event 已兜底
                logger.exception("事件队列处理异常")

    async def _tick_loop(self) -> None:
        idle_interval = self._config.scheduler.autonomous_tick_seconds
        active_interval = self._config.scheduler.active_tick_seconds
        while True:
            # 有活儿就快点想，池子空了就回到慢心跳，省点 token 也让思考链接得上。
            try:
                pending = await self._engine.has_pending_work()
            except Exception:  # noqa: BLE001 - 判断失败就按慢节奏来
                logger.warning("检查待思考念头失败，按空闲节奏继续", exc_info=True)
                pending = False
            interval = active_interval if pending else idle_interval
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
                return
            except asyncio.TimeoutError:
                pass
            self._tick_index += 1
            try:
                await self.submit_event(
                    Event.autonomous_tick(self._tick_index, reason="scheduled")
                )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("提交 AUTONOMOUS_TICK 失败")
            try:
                await self._check_contact_timeouts()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("检查联系超时失败")
            if self._tick_index % _MAINTENANCE_EVERY_TICKS == 0:
                try:
                    await self._run_maintenance()
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001
                    logger.exception("长期维护失败")

    async def _poll_loop(self) -> None:
        interval = self._config.qq.poll_interval_seconds
        backoff = interval
        while True:
            try:
                incoming = await self._qq.receive_message()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - QQ 适配器异常要重试
                logger.warning("QQ 接收异常，%.1fs 后重试", backoff, exc_info=True)
                await asyncio.sleep(backoff)
                backoff = min(max(backoff * 2, 1.0), 30.0)
                continue
            backoff = interval
            if incoming is None:
                await asyncio.sleep(interval)
                continue
            try:
                await self.submit_event(
                    Event.user_message(
                        conversation_id=incoming.conversation_id,
                        message_id=incoming.message_id,
                        user_id=incoming.user_id,
                        content=incoming.content,
                        received_at=incoming.received_at,
                        is_group=incoming.is_group,
                        group_id=incoming.group_id,
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("提交 USER_MESSAGE 失败")

    # ------------------------------------------------------------------ #
    # 状态工具
    # ------------------------------------------------------------------ #
    def _goto(
        self,
        target: AgentState,
        reason: str,
        *,
        conversation_id: str | None | object = _KEEP,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._machine.state is target:
            return
        if not self._machine.can_transition(target):
            if self._machine.state is not AgentState.IDLE and self._machine.can_transition(
                AgentState.IDLE
            ):
                self._machine.transition(AgentState.IDLE, f"reset:{reason}")
            if not self._machine.can_transition(target):
                raise InvalidTransitionError(self._machine.state, target)
        kwargs: dict[str, Any] = {"metadata": metadata}
        if conversation_id is not _KEEP:
            kwargs["conversation_id"] = conversation_id
        self._machine.transition(target, reason, **kwargs)
        self._recent_states.append(
            {
                "from": self._machine.previous_state.value
                if self._machine.previous_state
                else None,
                "to": target.value,
                "reason": reason[:120],
                "at": self._machine.updated_at.isoformat(),
            }
        )

    async def _safe_idle(self, reason: str) -> None:
        logger.warning("进入 SAFE_IDLE: %s", reason)
        now = self._clock()
        if self._machine.state is not AgentState.SAFE_IDLE:
            try:
                if self._machine.can_transition(AgentState.SAFE_IDLE):
                    self._machine.transition(AgentState.SAFE_IDLE, reason)
                else:
                    self._machine.restore(
                        StateSnapshot(
                            state=AgentState.SAFE_IDLE,
                            entered_at=now,
                            updated_at=now,
                            previous_state=self._machine.state,
                            thought_state=ThoughtState.INACTIVE,
                            metadata={"reason": reason[:200]},
                        )
                    )
            except InvalidTransitionError:
                self._machine.restore(
                    StateSnapshot(
                        state=AgentState.SAFE_IDLE,
                        entered_at=now,
                        updated_at=now,
                        previous_state=self._machine.state,
                        thought_state=ThoughtState.INACTIVE,
                        metadata={"reason": reason[:200]},
                    )
                )
        self._machine.set_thought_state(ThoughtState.INACTIVE)
        await self._persist_state()
        await self._metric("agent.safe_idle", 1, {"reason": reason[:200]})

    async def _persist_state(self) -> None:
        try:
            await self._states.save(self._machine.snapshot())
        except Exception:  # noqa: BLE001 - 状态落库失败要可见但不致命
            logger.exception("状态持久化失败")

    async def _restore_state(self) -> None:
        snapshot = await self._states.load()
        if snapshot is None:
            logger.info("首次启动，使用初始状态 IDLE")
            return
        recovery = recover_state(snapshot, self._clock())
        self._machine.restore(recovery.snapshot)
        logger.info(
            "状态恢复: %s -> %s (%s)",
            snapshot.state.value,
            recovery.snapshot.state.value,
            recovery.reason,
        )

    def _restore_emotion(self) -> None:
        payload = self._machine.metadata.get("emotion")
        if isinstance(payload, dict):
            try:
                self._engine.set_emotion(EmotionState.from_dict(payload))
            except Exception:  # noqa: BLE001 - 坏数据退回 baseline
                logger.warning("情绪状态无法恢复，使用 baseline", exc_info=True)

    async def _persist_emotion(self) -> None:
        if not self._config.emotion.enabled:
            return
        metadata = dict(self._machine.metadata)
        metadata["emotion"] = self._engine.emotion.to_dict()
        self._machine.set_metadata(metadata)
        await self._persist_state()

    async def _resume_silent_alarm(self) -> None:
        if self._machine.state is not AgentState.SILENT:
            return
        raw = self._machine.metadata.get("silent_until")
        if not raw:
            return
        try:
            deadline = datetime.fromisoformat(str(raw))
        except ValueError:
            logger.warning("SILENT 的 silent_until 无法解析，改回 IDLE: %r", raw)
            self._goto(AgentState.IDLE, "静默时间损坏", metadata={})
            await self._persist_state()
            return
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        delay = (deadline - self._clock()).total_seconds()
        if delay <= 0:
            self._goto(AgentState.IDLE, "静默已结束", metadata={})
            await self._persist_state()
            return
        alarm_id = self._machine.metadata.get("alarm_id")
        if not alarm_id:
            metadata = dict(self._machine.metadata)
            alarm_id = f"silent:{uuid.uuid4().hex}"
            metadata["alarm_id"] = alarm_id
            self._machine.set_metadata(metadata)
            await self._persist_state()
        self._schedule_alarm(delay, alarm_id=str(alarm_id), reason="silent_resumed")
        logger.info("恢复静默定时器: %.1fs 后解除（%s）", delay, alarm_id)

    async def _load_last_outgoing(self) -> None:
        try:
            row = await self._db.fetch_one(
                "SELECT MAX(created_at) AS last_at FROM messages WHERE role = 'assistant'"
            )
        except Exception:  # noqa: BLE001
            return
        if not row or not row["last_at"]:
            return
        try:
            self._last_outgoing_at = datetime.fromisoformat(str(row["last_at"]))
        except ValueError:
            self._last_outgoing_at = None

    def _schedule_alarm(
        self, delay_seconds: float, *, alarm_id: str, reason: str
    ) -> None:
        if self._wake_task is not None and not self._wake_task.done():
            self._wake_task.cancel()

        async def _fire() -> None:
            try:
                await asyncio.sleep(max(0.0, delay_seconds))
            except asyncio.CancelledError:
                return
            try:
                await self.submit_event(
                    Event.alarm_triggered(alarm_id=alarm_id, reason=reason)
                )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("提交闹钟事件失败")

        self._wake_task = asyncio.create_task(_fire(), name=f"alarm:{alarm_id}")

    async def _cancel_task(self, name: str) -> None:
        task: asyncio.Task[None] | None = getattr(self, name)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            setattr(self, name, None)

    async def _metric(
        self,
        name: str,
        value: float = 1.0,
        tags: dict[str, Any] | None = None,
    ) -> None:
        try:
            await self._metrics.record(name, value, tags)
        except Exception:  # noqa: BLE001
            logger.warning("指标写入失败: %s", name, exc_info=True)

    async def _record_tokens(self, request, response, *, route: str) -> None:
        """把一次调用的 token 记进预算，顺手记一条指标。"""
        try:
            used = self._budget.record_response(request, response)
        except Exception:  # noqa: BLE001 - 记账失败不该挡住主流程
            logger.warning("token 记账失败", exc_info=True)
            return
        await self._metric("llm.tokens", float(used), {"route": route})

    async def stats(self) -> dict[str, Any]:
        return {
            "state": self._machine.state.value,
            "previous_state": (
                self._machine.previous_state.value
                if self._machine.previous_state
                else None
            ),
            "thought_state": self._machine.thought_state.value,
            "conversation_id": self._machine.conversation_id,
            "running": self._started,
            "ticks": self._tick_index,
            "messages": await self._messages.count(),
            "events": await self._events.count(),
            "thoughts": await self._safe_count(self._thoughts.count),
            "thoughts_by_status": await self._safe_count(self._thoughts.count_by_status),
            "memories": await self._safe_count(self._memories.count),
            "emotion": self._engine.emotion.to_dict(),
            "budget": self._budget.snapshot().to_dict(),
            "recent_states": list(self._recent_states)[-10:],
        }

    async def _safe_count(self, func) -> Any:
        try:
            return await func()
        except Exception:  # noqa: BLE001 - 统计失败不影响心跳
            return None


__all__ = ["Agent", "HandleResult"]