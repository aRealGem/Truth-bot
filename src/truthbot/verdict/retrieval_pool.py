"""P120 PR-2: the adaptive parallel retrieval pool primitives.

Two levels of concurrency, both governed by :class:`~truthbot.verdict.pool_governor.PoolGovernor`:

* **L1 — ``parallel_shortlists``**: run a claim's R1/R2/R3 researcher shortlists
  CONCURRENTLY instead of serially (the ~3x win — a claim's wall-clock drops from
  the SUM of the researcher latencies to the MAX). R1 (the ``claude`` CLI worker)
  additionally acquires the governor's ``r1_gate`` so concurrent CLI subprocesses
  stay under the Claude Max / RSS cap. Wired into ``build_evidence_pack_v2`` as its
  ``shortlist_runner``; results are returned in pool order so labels line up.

* **L2 — ``build_packs_pooled``**: run several claims' pack builds in flight at once,
  bounded by ``governor.target_claims_in_flight()`` (re-sampled at each admission, so
  the pool grows/pares with Pi pressure). Used by ``retrieval_phase.build_packs_phase``.

Retrievers are blocking I/O (subprocess for R1, HTTP for R2/R3), so both levels use
``ThreadPoolExecutor`` — the same primitive the legacy engine already uses.
"""
from __future__ import annotations

import contextvars
import logging
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Callable, Optional, Sequence

logger = logging.getLogger(__name__)


def parallel_shortlists(pool: Sequence, call: Callable[[object], list], *,
                        governor=None, max_workers: Optional[int] = None) -> list:
    """L1 fan-out: call every retriever in ``pool`` concurrently, return results in
    pool order. ``call(retriever) -> list`` is expected to be fail-soft already (the
    evidence builder wraps each call in try/except). R1 workers acquire
    ``governor.r1_gate`` (the CLI-worker cap) before running."""
    if not pool:
        return []
    from truthbot.verify.retrievers import ClaudeWorkerRetriever

    workers = max_workers or len(pool)

    def _wrapped(r):
        if governor is not None and isinstance(r, ClaudeWorkerRetriever):
            with governor.r1_gate:            # cap concurrent claude CLI subprocesses
                return call(r)
        return call(r)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        # ThreadPoolExecutor does NOT propagate contextvars, so without this the
        # workers lose the caller's claim/run context and every metered
        # retrieval record would be attributed to no claim at all.
        #
        # A fresh Context per submit is required: Context.run raises
        # "cannot enter context - already entered" if one Context object is
        # entered by two threads at once, so hoisting a single copy_context()
        # out of the loop would crash intermittently under fan-out.
        futs = [ex.submit(contextvars.copy_context().run, _wrapped, r)
                for r in pool]
        return [f.result() for f in futs]     # pool order preserved


def build_packs_pooled(
    todo: list[dict],
    pack_builder: Callable[[str, str, str], object],
    governor,
    *,
    on_pack: Callable[[str, object], None],
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """L2 fan-out: build ``todo`` claims' packs with several in flight at once.

    The executor is created at the hard ceiling (``governor.pool_max``); actual
    admission is gated by ``governor.target_claims_in_flight()``, re-sampled before
    each submission so the pool grows/pares at runtime. ``on_pack(sid, pack)`` is
    invoked (on the DRIVER thread — safe for file journaling and dict writes) as each
    pack completes; ``on_progress(done, n, sid)`` is optional CLI logging.

    A ``pack_builder`` exception propagates after in-flight builds are drained and
    journaled via ``on_pack`` — so completed work is never lost (resume continues).
    ``governor.wait_if_critical()`` may raise ``CriticalPressureTimeout`` to stop the
    phase cleanly under sustained critical Pi pressure.
    """
    packs: dict = {}
    n = len(todo)
    if n == 0:
        return packs

    ceiling = max(1, int(governor.pool_max))
    it = iter(todo)
    inflight: dict = {}          # future -> sid
    done_count = 0
    exhausted = False

    ex = ThreadPoolExecutor(max_workers=ceiling)
    try:
        while inflight or not exhausted:
            # Admit up to the current adaptive target (pauses here on critical).
            if not exhausted:
                governor.wait_if_critical()
                target = governor.target_claims_in_flight()
                while not exhausted and len(inflight) < target:
                    c = next(it, None)
                    if c is None:
                        exhausted = True
                        break
                    # Same per-submit context copy as the L1 fan-out above, and
                    # for the same reason: threads do not inherit contextvars.
                    # This level is where run_id is lost — it is bound by
                    # telemetry_run_context on the DRIVER thread, while
                    # claim_id is bound inside the worker by the pack builder.
                    # So without this, split-mode retrieval rows land with a
                    # correct claim_id and run_id=None at every pool size,
                    # including pool=1, and nothing ties a row to its run.
                    fut = ex.submit(contextvars.copy_context().run,
                                    pack_builder, c["sid"], c["text"],
                                    c.get("context", ""))
                    inflight[fut] = c["sid"]
            governor.observe_in_flight(len(inflight))
            if not inflight:
                break
            finished, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in finished:
                sid = inflight.pop(fut)
                pack = fut.result()          # raises → drained below, then propagates
                packs[sid] = pack
                on_pack(sid, pack)
                done_count += 1
                if on_progress is not None:
                    on_progress(done_count, n, sid)
    except BaseException:
        # Drain whatever else already finished so its spend is journaled too, then
        # let the original exception propagate (partial-result semantics).
        for fut in list(inflight):
            if fut.done() and not fut.cancelled():
                try:
                    pack = fut.result()
                except BaseException:
                    continue
                sid = inflight.pop(fut)
                packs[sid] = pack
                on_pack(sid, pack)
        raise
    finally:
        ex.shutdown(wait=True)
    return packs
