#!/usr/bin/env python3
"""
Pure-FSM + action-executor asyncio event loop for multi-protocol stacks (PPP, TCP, …).

FSM implementations return ``(next_state, actions)`` only; this class performs I/O.

Author: deviprasad
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import logging
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

log = logging.getLogger("fsm_orchestrator")


@dataclass(frozen=True)
class Event:
    """External stimulus routed to one protocol FSM."""

    protocol_id: str
    event_type: str
    payload: Optional[bytes] = None


@dataclass(frozen=True)
class Action:
    """Side-effect requested by an FSM (executed only by the orchestrator)."""

    action_type: str
    target_protocol: str
    payload: Any = None


class FsmInstance(Protocol):
    """Minimal FSM surface: current state + pure transition function."""

    state: str

    def handle_event(self, event: Event) -> Tuple[str, List[Action]]:
        """Return ``(next_state, actions)`` without performing side effects."""


class FSMOrchestrator:
    """
    Single-threaded asyncio event loop: ``asyncio.PriorityQueue`` (lower int =
    higher priority), protocol registry, and action dispatcher (TX, timers).
    """

    def __init__(
        self,
        *,
        virtual_tx: Optional[Callable[[str, bytes], None]] = None,
    ) -> None:
        self._prio_queue: asyncio.PriorityQueue[Tuple[int, int, Event]] = asyncio.PriorityQueue()
        self._seq = itertools.count()
        self._protocols: Dict[str, Any] = {}
        self._timers: Dict[str, List[asyncio.Task[None]]] = {}
        self._virtual_tx = virtual_tx
        self._running = False

    def register_protocol(self, protocol_id: str, fsm_instance: Any) -> None:
        self._protocols[protocol_id] = fsm_instance

    def enqueue_event(self, event: Event, *, priority: int = 100) -> None:
        """Lower ``priority`` value is dequeued earlier (0 = highest)."""
        if not self._running:
            raise RuntimeError("enqueue_event called before run_forever task is active")
        cnt = next(self._seq)
        self._prio_queue.put_nowait((priority, cnt, event))

    async def _execute_action(self, action: Action) -> None:
        if action.action_type == "TX_PACKET":
            data = action.payload if isinstance(action.payload, (bytes, bytearray)) else b""
            if self._virtual_tx is not None:
                self._virtual_tx(action.target_protocol, bytes(data))
            else:
                log.info("TX_PACKET [%s] %d bytes (no virtual_tx hook)", action.target_protocol, len(data))
        elif action.action_type == "START_TIMER":
            delay_s = float(action.payload) if action.payload is not None else 1.0
            proto = action.target_protocol
            orch = self

            async def _fire() -> None:
                await asyncio.sleep(delay_s)
                try:
                    orch.enqueue_event(Event(proto, "TIMER_EXPIRED", None), priority=10)
                except RuntimeError:
                    log.debug("timer %s: orchestrator stopped, drop TIMER_EXPIRED", proto)

            task = asyncio.create_task(_fire(), name=f"timer-{proto}")
            self._timers.setdefault(proto, []).append(task)
            log.debug("START_TIMER %s %.3fs", proto, delay_s)
        elif action.action_type == "STOP_TIMER":
            tasks = self._timers.pop(action.target_protocol, [])
            for t in tasks:
                t.cancel()
            log.debug("STOP_TIMER %s cancelled %d task(s)", action.target_protocol, len(tasks))
        else:
            log.warning("unknown action_type %r", action.action_type)

    async def run_forever(self) -> None:
        """Drain the event queue until cancelled."""
        self._running = True
        log.info("FSMOrchestrator event loop started")
        try:
            while True:
                _prio, _cnt, event = await self._prio_queue.get()
                fsm = self._protocols.get(event.protocol_id)
                if fsm is None:
                    log.error("unknown protocol_id %r", event.protocol_id)
                    continue
                old = getattr(fsm, "state", "")
                try:
                    new_state, actions = fsm.handle_event(event)
                except Exception:
                    log.exception("handle_event failed for %s", event.protocol_id)
                    continue
                setattr(fsm, "state", new_state)
                log.info(
                    "[%s] State: %s -> %s | Trigger: %s | Emitted %d Actions",
                    event.protocol_id,
                    old,
                    new_state,
                    event.event_type,
                    len(actions),
                )
                for act in actions:
                    await self._execute_action(act)
        finally:
            for tasks in self._timers.values():
                for t in tasks:
                    t.cancel()
            self._timers.clear()
            self._running = False


class _StubFsm:
    """Minimal FSM for demos: toggles between IDLE and OPEN on USER_OPEN / Close."""

    def __init__(self) -> None:
        self.state = "IDLE"

    def handle_event(self, event: Event) -> Tuple[str, List[Action]]:
        if event.event_type == "USER_OPEN" and self.state == "IDLE":
            return "OPEN", [Action("START_TIMER", "demo", 0.5)]
        if event.event_type == "TIMER_EXPIRED" and self.state == "OPEN":
            return "IDLE", [Action("TX_PACKET", "demo", b"hello")]
        if event.event_type == "RX_PACKET":
            return self.state, []
        return self.state, []


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="FSM asyncio orchestrator (demo loop)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    async def _demo() -> None:
        orch = FSMOrchestrator(virtual_tx=lambda p, b: log.info("virtual TX %s %r", p, b))
        orch.register_protocol("demo", _StubFsm())
        task = asyncio.create_task(orch.run_forever())
        await asyncio.sleep(0)  # let run_forever set _running
        orch.enqueue_event(Event("demo", "USER_OPEN", None), priority=5)
        await asyncio.sleep(1.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    try:
        asyncio.run(_demo())
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
