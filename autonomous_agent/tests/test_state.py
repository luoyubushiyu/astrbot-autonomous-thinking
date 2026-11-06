from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from autonomous_agent.core.state import (
    ALLOWED_TRANSITIONS,
    AgentState,
    AgentStateMachine,
    InvalidTransitionError,
    StateSnapshot,
    ThoughtState,
    recover_state,
)


def snapshot(state: AgentState, **metadata) -> StateSnapshot:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return StateSnapshot(
        state=state, entered_at=now, updated_at=now, metadata=metadata
    )


class StateMachineTests(unittest.TestCase):
    def test_initial_state(self) -> None:
        machine = AgentStateMachine()
        self.assertIs(machine.state, AgentState.IDLE)
        self.assertIs(machine.thought_state, ThoughtState.INACTIVE)

    def test_allowed_transition(self) -> None:
        machine = AgentStateMachine()
        change = machine.transition(AgentState.OBSERVING, "收到消息")
        self.assertIs(change.previous, AgentState.IDLE)
        self.assertIs(machine.state, AgentState.OBSERVING)

    def test_invalid_transition(self) -> None:
        machine = AgentStateMachine()
        machine.transition(AgentState.NIGHT)
        with self.assertRaises(InvalidTransitionError):
            machine.transition(AgentState.OBSERVING)

    def test_same_state_is_allowed(self) -> None:
        machine = AgentStateMachine()
        machine.transition(AgentState.SAFE_IDLE)
        machine.transition(AgentState.SAFE_IDLE)

    def test_every_state_has_transitions(self) -> None:
        self.assertEqual(set(ALLOWED_TRANSITIONS), set(AgentState))
        for state, targets in ALLOWED_TRANSITIONS.items():
            with self.subTest(state=state):
                self.assertTrue(targets, f"{state} 没有任何出口")

    def test_night_sleeping_cycle(self) -> None:
        machine = AgentStateMachine()
        machine.transition(AgentState.NIGHT)
        machine.transition(AgentState.SLEEPING)
        machine.transition(AgentState.NIGHT)
        machine.transition(AgentState.IDLE)

    def test_snapshot_roundtrip(self) -> None:
        machine = AgentStateMachine()
        machine.transition(AgentState.THINKING, "thinking", conversation_id="c1")
        machine.set_thought_state(ThoughtState.PENDING)
        restored = AgentStateMachine()
        restored.restore(machine.snapshot())
        self.assertIs(restored.state, AgentState.THINKING)
        self.assertIs(restored.thought_state, ThoughtState.PENDING)
        self.assertEqual(restored.conversation_id, "c1")


class RecoveryTests(unittest.TestCase):
    def test_transient_becomes_safe_idle(self) -> None:
        for state in (AgentState.OBSERVING, AgentState.THINKING, AgentState.CONTACTING):
            with self.subTest(state=state):
                result = recover_state(snapshot(state))
                self.assertTrue(result.changed)
                self.assertIs(result.snapshot.state, AgentState.SAFE_IDLE)

    def test_waiting_becomes_idle(self) -> None:
        result = recover_state(snapshot(AgentState.WAITING))
        self.assertIs(result.snapshot.state, AgentState.IDLE)

    def test_silent_kept_when_in_future(self) -> None:
        now = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        snap = snapshot(
            AgentState.SILENT,
            silent_until=(now + timedelta(minutes=5)).isoformat(),
        )
        result = recover_state(snap, now)
        self.assertFalse(result.changed)
        self.assertIs(result.snapshot.state, AgentState.SILENT)

    def test_silent_expired(self) -> None:
        now = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        snap = snapshot(
            AgentState.SILENT,
            silent_until=(now - timedelta(minutes=5)).isoformat(),
        )
        result = recover_state(snap, now)
        self.assertTrue(result.changed)
        self.assertIs(result.snapshot.state, AgentState.IDLE)

    def test_night_becomes_idle(self) -> None:
        result = recover_state(snapshot(AgentState.NIGHT))
        self.assertIs(result.snapshot.state, AgentState.IDLE)

    def test_sleeping_becomes_idle(self) -> None:
        result = recover_state(snapshot(AgentState.SLEEPING))
        self.assertTrue(result.changed)
        self.assertIs(result.snapshot.state, AgentState.IDLE)

    def test_idle_kept(self) -> None:
        result = recover_state(snapshot(AgentState.IDLE))
        self.assertFalse(result.changed)


if __name__ == "__main__":
    unittest.main()