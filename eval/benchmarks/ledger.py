"""
Independent run ledger for truth-bot proxy runs.

One JSONL row per run, appended after `hm.run(...)`: the client config + test-run
settings, the run outcome, and the cost LiteLLM reported. Kept independent of the
HydraMind engine (the engine stays pure) — the runner calls `append_run` itself.

Cost source of truth is the manifest's proxy-reported per-call cost
(`cost_source="proxy"`, captured from the `x-litellm-response-cost` header in
`hydramind/transport.extract_cost`). The row records `cost_source_tally` so a
table-estimated total is never mistaken for a LiteLLM-authoritative one.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

DEFAULT_PATH = "metrics/spend_ledger/truthbot.jsonl"


def build_record(manifest, config: Optional[dict] = None,
                 notes: Optional[dict] = None, ts: Optional[str] = None,
                 run_id: Optional[str] = None) -> dict:
    """Assemble a ledger row from a finished RunManifest (+ optional strategy
    notes and run config). Pure — no I/O — so it is trivially testable."""
    config = config or {}
    notes = notes or {}
    spec = manifest.resolved_spec or {}
    cost_cfg = spec.get("cost", {}) or {}
    return {
        "ts": ts or datetime.now(timezone.utc).isoformat(),
        "run_id": run_id or uuid.uuid4().hex[:12],
        "client": manifest.project,          # spend-attribution identity (truth-bot)
        "key_label": config.get("key_label"),
        "strategy": manifest.strategy,
        "roster": spec.get("roster_name"),
        "task": manifest.task,
        "n_items": manifest.n_items,
        "dataset_hash": manifest.dataset_hash,
        "budget_ceiling_usd": cost_cfg.get("ceiling_usd"),
        "base_url": config.get("base_url"),
        "result": {
            "halted": manifest.halted,
            "halt_reason": manifest.halt_reason,
            "flagged": notes.get("flagged"),
            "split_rate": notes.get("split_rate"),
            "escalation": manifest.escalation or None,
        },
        "cost": {
            "total_cost_usd": manifest.total_cost_usd,
            "tokens_in": manifest.total_tokens_in,
            "tokens_out": manifest.total_tokens_out,
            "cost_source_tally": manifest.cost_source_tally,
            "lanes": manifest.to_spend_records(),
        },
        "model_mismatches": manifest.model_mismatches(),
    }


def append_run(path, manifest, config: Optional[dict] = None,
               notes: Optional[dict] = None, ts: Optional[str] = None,
               run_id: Optional[str] = None) -> dict:
    """Append one run's ledger row to the JSONL at `path` and return it."""
    rec = build_record(manifest, config=config, notes=notes, ts=ts, run_id=run_id)
    d = os.path.dirname(str(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec
