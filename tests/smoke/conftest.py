"""
Shared fixtures + helpers for live smoke tests.

The smoke suite has two phases that talk to each other through a
**manifest file** on disk:

    metrics/smoke/manifest.json

``test_smoke_submit.py`` writes entries into the manifest once each
provider is submitted (or complete, for the live providers). The
``test_smoke_reconcile.py`` tests later read those entries to find
the ``run_id`` for each batch provider and poll until it completes.

The manifest is deliberately a single JSON blob (not JSONL) because
these tests rewrite it in place and a single load + dump is simpler
than keyed JSONL line surgery. Contention is a non-issue: the two
test files are run in serial phases, never in parallel.

Provider SLAs + the AUTOMATED_WATCH_CAP_S constant below are the
source-of-truth timeouts; see ``tests/smoke/README.md`` for the
reasoning.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import pytest

from truthbot.models import Claim


# ---------------------------------------------------------------------------
# Fixed test data
# ---------------------------------------------------------------------------

# Two claims with opposite truth values. Any reasonable fact-checker should
# label these correctly on the first try, with zero ambiguity.
CLAIM_TRUE_TEXT = "The United States landed astronauts on the Moon in 1969."
CLAIM_FALSE_TEXT = "The Eiffel Tower is located in Berlin, Germany."


# Additional trivial claims used by the paginated smoke. Each is paired with
# its expected truth polarity so a single list comprehension produces both
# the Claim objects and the truth-pattern bool list used for assertions.
# Kept deliberately unambiguous so multi-chunk validation isn't confounded
# by verdict noise on hard content.
CLAIM_EXTRAS: list[tuple[str, bool]] = [
    ("Water boils at 100 degrees Celsius at standard atmospheric pressure.", True),
    ("The Great Wall of China is visible from the Moon with the naked eye.", False),
    ("The Pacific Ocean is the largest ocean on Earth.", True),
]


# ---------------------------------------------------------------------------
# SLA + timeout constants
# ---------------------------------------------------------------------------
#
# Both Anthropic and OpenAI expire batches server-side 24h after creation;
# past that, the job is unrecoverable regardless of client tooling.
#
# Our AUTOMATED cap is 2.5 h: the "longest reasonable" line past which we
# stop waiting on autopilot. The manifest stays on disk, so operators can
# manually resume with ``truthbot batch reconcile <run_id>`` any time up
# to the 24 h vendor cutoff (or with TRUTHBOT_SMOKE_TIMEOUT_<PROVIDER> set
# higher, opt-in, for another pytest-driven wait).
#
# Env overrides (seconds):
#   TRUTHBOT_SMOKE_TIMEOUT_ANTHROPIC_BATCH
#   TRUTHBOT_SMOKE_TIMEOUT_OPENAI_BATCH
#   TRUTHBOT_SMOKE_TIMEOUT_XAI_LIVE
#   TRUTHBOT_SMOKE_TIMEOUT_GEMINI_LIVE
# ---------------------------------------------------------------------------

AUTOMATED_WATCH_CAP_S: int = int(2.5 * 60 * 60)   # 9000s = 2.5 h
VENDOR_EXPIRY_S: int = 24 * 60 * 60               # 86400s = 24 h

_DEFAULT_TIMEOUTS_S: dict[str, int] = {
    "anthropic_batch": AUTOMATED_WATCH_CAP_S,
    "openai_batch":    AUTOMATED_WATCH_CAP_S,
    "xai_live":        3 * 60,
    "gemini_live":     3 * 60,
}


def provider_timeout_s(provider: str) -> int:
    """Per-provider wall-clock cap, with env override."""
    default = _DEFAULT_TIMEOUTS_S[provider]
    env_var = f"TRUTHBOT_SMOKE_TIMEOUT_{provider.upper()}"
    raw = os.environ.get(env_var)
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        raise RuntimeError(
            f"{env_var} must be an integer number of seconds, got {raw!r}"
        )
    if parsed <= 0:
        raise RuntimeError(f"{env_var} must be positive, got {parsed}")
    return parsed


# ---------------------------------------------------------------------------
# Claim + filesystem fixtures
# ---------------------------------------------------------------------------


def _mk_claim(text: str, transcript_id: str = "smoke-transcript") -> Claim:
    return Claim(
        transcript_id=transcript_id,
        text=text,
        speaker="Smoke Test",
        context=text,
        is_checkable=True,
    )


@pytest.fixture
def two_claims() -> list[Claim]:
    return [_mk_claim(CLAIM_TRUE_TEXT), _mk_claim(CLAIM_FALSE_TEXT)]


@pytest.fixture
def five_claims() -> list[Claim]:
    """
    Five trivial claims for the paginated smoke (3 TRUE + 2 FALSE).

    With ``claims_per_request=2`` this yields 3 chunks (2 + 2 + 1) on the
    batch providers, which is the smallest N that exercises both an even
    split and the odd-remainder tail of the chunking loop.
    """
    texts = [CLAIM_TRUE_TEXT, CLAIM_FALSE_TEXT] + [t for t, _ in CLAIM_EXTRAS]
    return [_mk_claim(text) for text in texts]


@pytest.fixture
def five_claims_truth_pattern() -> list[bool]:
    """``True``/``False`` per index, aligned with ``five_claims``."""
    return [True, False] + [expected for _, expected in CLAIM_EXTRAS]


def chunk_claims(
    claims: list[Claim],
    truth_pattern: list[bool],
    chunk_size: int,
) -> list[tuple[list[Claim], list[bool]]]:
    """
    Partition ``claims`` + ``truth_pattern`` into fixed-size chunks.

    Returns a list of ``(chunk_claims, chunk_truth_pattern)`` tuples. The
    final tuple may hold fewer than ``chunk_size`` items when
    ``len(claims)`` doesn't divide evenly.

    Used by the paginated-multi smoke tests to manually drive
    ``adapter.call_multi`` N times over five claims (e.g. chunk_size=2
    yields 2+2+1, chunk_size=4 yields 4+1, chunk_size=6 yields one
    5-claim chunk).
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if len(claims) != len(truth_pattern):
        raise ValueError(
            f"claims / truth_pattern length mismatch: "
            f"{len(claims)} vs {len(truth_pattern)}"
        )
    return [
        (claims[i : i + chunk_size], truth_pattern[i : i + chunk_size])
        for i in range(0, len(claims), chunk_size)
    ]


@pytest.fixture(scope="session")
def smoke_metrics_dir() -> Path:
    """
    The on-repo metrics directory the smoke suite writes into.

    We deliberately use the repo-root ``metrics/smoke/`` rather than
    ``tmp_path``: the whole point of the two-phase design is that
    ``test_smoke_reconcile.py`` can find the batch IDs from a prior
    ``test_smoke_submit.py`` run, potentially in a different pytest
    invocation. ``tmp_path`` is session-scoped and would defeat that.
    """
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "metrics" / "smoke"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def manifest_path(smoke_metrics_dir: Path) -> Path:
    return smoke_metrics_dir / "manifest.json"


# ---------------------------------------------------------------------------
# Manifest IO
# ---------------------------------------------------------------------------


def load_manifest(path: Path) -> dict[str, Any]:
    """
    Read the on-disk manifest, or return an empty dict if it doesn't exist.

    Schema (per-provider):
        {
          "anthropic": {"run_id": "...", "batch_id": "...", "status": "submitted",
                        "chunk_size": 2, "submitted_at": 1714.., ...},
          "openai":    {"run_id": "...", "batch_id": "...", ...},
          "xai":       {"status": "complete", "verdicts": [...], ...},
          "gemini":    {"status": "complete", "verdicts": [...], ...}
        }
    """
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def update_manifest(path: Path, provider: str, entry: dict[str, Any]) -> None:
    """Merge ``entry`` into the manifest under ``provider``."""
    current = load_manifest(path)
    existing = current.get(provider) or {}
    existing.update(entry)
    current[provider] = existing
    save_manifest(path, current)


# ---------------------------------------------------------------------------
# Env + label helpers
# ---------------------------------------------------------------------------


def require_key(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} not set; cannot run live smoke")
    return value


def is_true_label(label) -> bool:
    from truthbot.models import VerdictLabel
    return label in {VerdictLabel.TRUE, VerdictLabel.MOSTLY_TRUE}


def is_false_label(label) -> bool:
    from truthbot.models import VerdictLabel
    return label in {
        VerdictLabel.FALSE,
        VerdictLabel.MISLEADING,
        VerdictLabel.EXAGGERATED,
    }


# ---------------------------------------------------------------------------
# Poll-cycle helpers (used by the reconcile suite)
# ---------------------------------------------------------------------------


def poll_interval_for_elapsed(elapsed_s: float) -> int:
    """
    Return the poll interval (seconds) to use for the next cycle, based on
    how long we've already waited. Gets coarser over time so we don't
    hammer vendor status endpoints during the long tail.
    """
    if elapsed_s < 5 * 60:
        return 60
    if elapsed_s < 30 * 60:
        return 120
    return 300


def print_poll_line(provider: str, status: str, elapsed_s: float) -> None:
    """
    Single-line progress print in a fixed format so the terminal file
    is easy to tail + grep from the monitoring loop.
    """
    mm = int(elapsed_s // 60)
    ss = int(elapsed_s % 60)
    stamp = time.strftime("%H:%M:%S")
    print(
        f"[smoke] {stamp}  {provider}  status={status}  elapsed={mm:02d}m{ss:02d}s",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Summary writer (used by both phases)
# ---------------------------------------------------------------------------


def _run_reconcile_n(
    provider: str,
    adapter,
    *,
    manifest_path: Path,
    metrics_dir: Path,
    claims: list,
    truth_pattern: list[bool],
    manifest_key: str,
    timeout_s: int,
    summary_mode: str = "batch",
) -> None:
    """
    Generic poll → reconcile → assert for one batch provider.

    Reads the ``manifest_key`` entry from the manifest (NOT ``provider``,
    since the paginated smoke uses suffixed keys like ``anthropic_pg``
    while still routing to the ``anthropic`` provider code path). If the
    manifest entry is missing, SKIPS rather than failing.

    ``claims`` and ``truth_pattern`` must be the same length; each claim
    is matched to its bundle by ``claim.text`` and the bundle's
    ``model_verdicts`` must contain at least one label matching the
    expected truth polarity for that index.
    """
    import pytest

    from truthbot.verify.batch import BatchDispatcher, reconcile_run
    from truthbot.verify.engine import VerificationEngine

    assert len(claims) == len(truth_pattern), (
        f"claims ({len(claims)}) and truth_pattern ({len(truth_pattern)}) "
        f"must align"
    )

    manifest = load_manifest(manifest_path)
    entry = manifest.get(manifest_key)
    if not entry or not entry.get("run_id") or not entry.get("batch_id"):
        pytest.skip(
            f"{manifest_key}: no submit entry in manifest {manifest_path} "
            f"(run the corresponding submit test first)"
        )

    run_id = entry["run_id"]
    batch_id = entry["batch_id"]
    submitted_at = entry.get("submitted_at") or time.time()

    dispatcher = BatchDispatcher(metrics_dir)
    t_start = time.monotonic()
    elapsed = 0.0
    last_status = "pending"

    print(
        f"[smoke] {manifest_key}: polling run_id={run_id} batch_id={batch_id} "
        f"cap={timeout_s}s",
        flush=True,
    )

    while True:
        last_status = dispatcher.poll(run_id)
        elapsed = time.monotonic() - t_start
        print_poll_line(manifest_key, last_status, elapsed)

        if last_status in ("complete", "failed", "missing"):
            break
        if elapsed >= timeout_s:
            update_manifest(
                manifest_path,
                manifest_key,
                {
                    "status": "pending_at_cap",
                    "last_status": last_status,
                    "automated_cap_hit_at": time.time(),
                    "elapsed_at_cap_s": round(elapsed, 1),
                },
            )
            pytest.fail(
                f"{manifest_key}: still {last_status!r} after automated cap "
                f"of {timeout_s}s ({timeout_s / 3600:.2f}h). "
                f"run_id={run_id}, batch_id={batch_id}. "
                f"Resume manually with: truthbot batch reconcile {run_id}"
            )

        time.sleep(poll_interval_for_elapsed(elapsed))

    if last_status != "complete":
        update_manifest(
            manifest_path,
            manifest_key,
            {
                "status": last_status,
                "terminal_at": time.time(),
                "elapsed_s": round(elapsed, 1),
            },
        )
        pytest.fail(
            f"{manifest_key}: batch terminated as {last_status!r} after "
            f"{elapsed:.0f}s. run_id={run_id}, batch_id={batch_id}."
        )

    engine = VerificationEngine(run_id=run_id, inject_evidence=False)
    result = reconcile_run(
        metrics_dir,
        run_id,
        adapters_by_name={provider: adapter},
        engine=engine,
    )
    assert result["status"] == "complete", (
        f"{manifest_key}: reconcile_run returned {result['status']}, "
        f"expected complete"
    )
    bundles = result["bundles"]
    assert len(bundles) == len(claims), (
        f"{manifest_key}: expected {len(claims)} bundles, got {len(bundles)}"
    )

    by_text = {b.claim.text: b for b in bundles}
    mismatches: list[tuple[int, str, list]] = []
    for i, (claim, expected_true) in enumerate(zip(claims, truth_pattern)):
        bundle = by_text.get(claim.text)
        assert bundle is not None, (
            f"{manifest_key}: no bundle for claim {i}: {claim.text!r}"
        )
        labels = [mv.label for mv in bundle.model_verdicts]
        check = is_true_label if expected_true else is_false_label
        if not any(check(lbl) for lbl in labels):
            mismatches.append((i, claim.text, [lbl.value for lbl in labels]))

    assert not mismatches, (
        f"{manifest_key}: {len(mismatches)} claim(s) did not match truth "
        f"pattern. mismatches: {mismatches}"
    )

    descriptor = result["descriptor"] or {}
    provider_entry: dict[str, Any] = (descriptor.get("provider_jobs") or {}).get(
        provider, {}
    )

    total_elapsed = time.time() - submitted_at
    verdict_rows = [
        {
            "claim": b.claim.text,
            "consensus_label": b.consensus.consensus_label.value,
            "model_labels": [
                {"model": mv.adapter_name, "label": mv.label.value}
                for mv in b.model_verdicts
            ],
        }
        for b in bundles
    ]

    update_manifest(
        manifest_path,
        manifest_key,
        {
            "status": "complete",
            "completed_at": time.time(),
            "elapsed_total_s": round(total_elapsed, 1),
            "elapsed_poll_s": round(elapsed, 1),
            "verdicts": verdict_rows,
            "chunk_size": provider_entry.get("chunk_size"),
            "request_count": provider_entry.get("request_count"),
        },
    )
    append_smoke_summary(
        metrics_dir,
        manifest_key,
        summary_mode,
        wall_clock_s=total_elapsed,
        claim_count=len(claims),
        request_count=provider_entry.get("request_count", 1),
        verdicts=verdict_rows,
        notes=f"run_id={run_id}; batch_id={batch_id}",
    )


def append_smoke_summary(
    status_path: Path,
    provider: str,
    mode: str,
    *,
    wall_clock_s: float,
    claim_count: int,
    request_count: int,
    verdicts: Iterable[dict],
    notes: str = "",
) -> None:
    """Write one row to a structured JSONL summary that the status task reads."""
    path = status_path / "smoke_summary.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "provider": provider,
                    "mode": mode,
                    "wall_clock_s": round(wall_clock_s, 2),
                    "claim_count": claim_count,
                    "request_count": request_count,
                    "verdicts": list(verdicts),
                    "notes": notes,
                }
            )
            + "\n"
        )
