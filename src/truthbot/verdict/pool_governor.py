"""P120 PR-2: the in-process resource monitor that sizes the adaptive retrieval
pool. It is sampled SYNCHRONOUSLY at claim-admission boundaries — there is no
background thread or daemon (memory ``no-unattended-watchers``): the pool driver
asks the governor for a target before admitting each claim, and the governor reads
the cc-host pressure flag + its own backoff state on that call.

Two signal families govern the pool:
  * **Pi health** — ``~/.config/cc-host/pressure.json`` (level ok/warn/critical +
    MemAvailable). ``ok`` grows claims-in-flight toward the ceiling; ``warn`` pares
    to serial; ``critical`` pauses new admissions until it clears (bounded wait).
  * **API / subscription headroom** — R1's Claude Max usage-limit signal drops R1
    from the active researcher set for a cool-down (the run carries on R1-less on
    R2+retry-R3); provider 429s on R2/R3 pare that lane and claims-in-flight.

Backoff state is mutated from retriever worker threads (via ``note_*`` callbacks)
and read from the driver thread, so it is guarded by a lock. ``now_fn``/``sleep_fn``/
``pressure_path`` are injectable so the whole thing tests offline with no real clock,
sleep, or pressure file.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_PRESSURE_PATH = os.path.expanduser("~/.config/cc-host/pressure.json")


class CriticalPressureTimeout(RuntimeError):
    """Raised when the Pi stays at ``critical`` pressure past ``pressure_wait_s``.
    The driver turns completed packs into a clean journaled stop (nothing lost)."""


@dataclass
class PoolGovernor:
    pool_start: int = 1                 # starting claims-in-flight (serial default)
    pool_max: int = 3                   # hard ceiling the monitor may grow to
    r1_cli_cap: int = 2                 # max concurrent claude CLI workers (Max + RSS)
    adaptive: bool = True               # False → freeze at pool_start (ablation)
    mem_floor_mb: int = 2000            # pare below this MemAvailable even at ok
    pressure_stale_s: int = 5400        # older pressure.json → treat as warn
    pressure_wait_s: int = 600          # max block on critical before a clean stop
    r1_cooldown_s: int = 900            # how long R1 stays dropped after a Max signal
    pressure_path: str = DEFAULT_PRESSURE_PATH
    now_fn: Callable[[], float] = time.time
    sleep_fn: Callable[[float], None] = time.sleep

    # ── internal state ────────────────────────────────────────────────────────
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    r1_gate: threading.Semaphore = field(init=False, repr=False)
    _r1_dropped_until: float = field(default=0.0, repr=False)
    _lane_backoff: dict = field(default_factory=dict, repr=False)   # lane -> until_ts
    _events: Counter = field(default_factory=Counter, repr=False)
    _max_in_flight: int = field(default=0, repr=False)

    def __post_init__(self):
        self.r1_gate = threading.Semaphore(max(1, int(self.r1_cli_cap)))

    # ── API/subscription signals (called from retriever worker threads) ────────

    def note_max_signal(self) -> None:
        """R1 hit a Claude Max usage/rate limit → drop R1 for a cool-down."""
        with self._lock:
            self._r1_dropped_until = self.now_fn() + self.r1_cooldown_s
            self._events["r1_max_drop"] += 1
        logger.warning("pool governor: R1 dropped for %ds after a Max signal",
                       self.r1_cooldown_s)

    def note_429(self, lane: str) -> None:
        """A provider 429 on R2/R3 → pare that lane for a short cool-down."""
        with self._lock:
            self._lane_backoff[lane] = self.now_fn() + 60.0
            self._events[f"429_{lane}"] += 1
        logger.warning("pool governor: %s 429 → lane pared for 60s", lane)

    def _bump(self, key: str) -> None:
        """Thread-safe telemetry counter bump (call OUTSIDE an already-held lock —
        threading.Lock is not reentrant)."""
        with self._lock:
            self._events[key] += 1

    def r1_available(self) -> bool:
        with self._lock:
            dropped = self._r1_dropped_until > self.now_fn()
        return not dropped

    def active_retrievers(self, pool: Sequence) -> list:
        """Filter ``pool`` to the researchers currently in play — drops R1 while it
        is in a Max cool-down (jackie's rule: carry on R1-less, re-admit on recovery).
        Never returns empty: if R1 is the only member and it's dropped, R1 rides
        anyway (better to try than to build nothing)."""
        if self.r1_available():
            return list(pool)
        from truthbot.verify.retrievers import ClaudeWorkerRetriever
        kept = [r for r in pool if not isinstance(r, ClaudeWorkerRetriever)]
        if kept:
            self._bump("r1_skipped_claim")
            return kept
        return list(pool)

    # ── Pi-health read ─────────────────────────────────────────────────────────

    def read_pressure(self) -> dict:
        """Parse the cc-host pressure flag. Missing / unparseable / STALE → a
        synthesized ``warn`` (fail-safe: never assume the box is healthy when we
        can't tell)."""
        p = Path(self.pressure_path)
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"level": "warn", "mem_avail_mb": None, "reason": "pressure.json unreadable"}
        ts = doc.get("ts")
        if not isinstance(ts, (int, float)) or (self.now_fn() - ts) > self.pressure_stale_s:
            return {"level": "warn", "mem_avail_mb": doc.get("mem_avail_mb"),
                    "reason": "pressure.json stale"}
        return doc

    def target_claims_in_flight(self) -> int:
        """Claims to have in flight right now, in ``[1, pool_max]``. ``--no-adaptive``
        freezes it at ``pool_start``; otherwise Pi health decides: critical/warn → 1,
        low memory → 1, ok → grow to ``pool_max``."""
        if not self.adaptive:
            return max(1, min(self.pool_start, self.pool_max))
        doc = self.read_pressure()
        level = doc.get("level", "warn")
        mem = doc.get("mem_avail_mb")
        if level == "critical":
            return 1
        if level == "warn":
            self._bump("warn_pares")
            return 1
        if isinstance(mem, (int, float)) and mem < self.mem_floor_mb:
            self._bump("mem_floor_pares")
            return 1
        # healthy: grow toward the ceiling, but respect any lane in 429 backoff.
        target = self.pool_max
        if self._lane_in_backoff():
            target = max(1, target - 1)
        return max(1, min(target, self.pool_max))

    def _lane_in_backoff(self) -> bool:
        now = self.now_fn()
        with self._lock:
            return any(until > now for until in self._lane_backoff.values())

    def wait_if_critical(self) -> None:
        """Block while the Pi is at ``critical`` pressure, polling until it clears or
        ``pressure_wait_s`` elapses (then raise ``CriticalPressureTimeout`` so the
        driver stops cleanly with everything journaled). No-op when not critical."""
        if not self.adaptive:
            return
        waited = 0.0
        poll = 15.0
        while self.read_pressure().get("level") == "critical":
            if waited == 0.0:
                self._bump("critical_pauses")
                logger.warning("pool governor: critical Pi pressure — pausing new "
                               "claim admissions (draining in-flight)")
            if waited >= self.pressure_wait_s:
                raise CriticalPressureTimeout(
                    f"Pi at critical pressure > {self.pressure_wait_s}s — stopping "
                    f"Phase R with completed packs journaled")
            self.sleep_fn(poll)
            waited += poll

    # ── telemetry ──────────────────────────────────────────────────────────────

    def observe_in_flight(self, n: int) -> None:
        with self._lock:
            if n > self._max_in_flight:
                self._max_in_flight = n

    def telemetry(self) -> dict:
        with self._lock:
            events = dict(self._events)
            max_in_flight = self._max_in_flight
        return {
            "pool_start": self.pool_start, "pool_max": self.pool_max,
            "r1_cli_cap": self.r1_cli_cap, "adaptive": self.adaptive,
            "max_in_flight": max_in_flight,
            "events": events,
        }
