"""shared_pack_v2 wiring (P67.9): trio → consolidator pack builder, the T2.4
one-retry quality gate, and adjudicate's forced-Unverifiable partition.

Offline: fake retrievers only — no network, no CLI, no spend."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from truthbot.models import Evidence, SourceTier
from truthbot.verdict import adjudicator
from truthbot.verdict.consolidator import GATE_INSUFFICIENT
from truthbot.verdict.evidence_pack_v2 import build_evidence_pack_v2
from truthbot.verdict.speech_context import register_speech_date

SID = "pytest_v2:0001"
UTT = datetime(2026, 2, 24).date()
register_speech_date("pytest_v2", UTT)


def _ev(url, *, tier=SourceTier.GOVERNMENT, supports=True, day=20):
    return Evidence(claim_id="", source_name="R", source_url=url,
                    source_tier=tier, snippet="stat page",
                    supports_claim=supports,
                    published_at=datetime(2026, 2, day, tzinfo=timezone.utc))


class _Retriever:
    """Scripted retriever: pops one shortlist per call, records contexts."""

    def __init__(self, label, shortlists):
        self.label = label
        self.shortlists = list(shortlists)
        self.contexts = []

    def shortlist(self, claim_text, *, context="", utterance=None, window=None):
        self.contexts.append(context)
        return self.shortlists.pop(0) if self.shortlists else []


def test_v2_pack_builds_without_retry_when_quota_met():
    r1 = _Retriever("R1", [[_ev("https://bls.gov/a"), _ev("https://bea.gov/b", supports=False)]])
    r2 = _Retriever("R2", [[_ev("https://apnews.com/c", tier=SourceTier.WIRE)]])
    pack = build_evidence_pack_v2(SID, "unemployment fell", (r1, r2))
    assert pack.gate_code == ""
    assert pack.ids == ["E1", "E2", "E3"]
    assert len(r1.contexts) == 1 and len(r2.contexts) == 1   # no retry pass
    # provenance survives into the pack items (I5 checked inside the builder)
    assert all(it.sha256 and it.retrieved_at for it in pack.items)


def test_v2_gate_retries_once_then_forces_code():
    r1 = _Retriever("R1", [[], []])                    # nothing, twice
    r2 = _Retriever("R2", [[_ev("https://blog.example.com/x", tier=SourceTier.OTHER)], []])
    pack = build_evidence_pack_v2(SID, "claim", (r1, r2))
    assert pack.gate_code == GATE_INSUFFICIENT
    assert len(r1.contexts) == 2 and len(r2.contexts) == 2   # exactly ONE retry
    assert "TARGETED RE-RETRIEVAL" in r1.contexts[1]         # retry is targeted


def test_v2_retry_can_rescue_quota():
    rescue = [_ev("https://bls.gov/a"), _ev("https://treasury.gov/b", supports=False)]
    r1 = _Retriever("R1", [[], rescue])
    pack = build_evidence_pack_v2(SID, "claim", (r1,))
    assert pack.gate_code == "" and len(pack.items) == 2


def test_v2_dead_retriever_is_soft():
    class Boom:
        label = "R3"

        def shortlist(self, *a, **kw):
            raise RuntimeError("lane down")

    r1 = _Retriever("R1", [[_ev("https://bls.gov/a"),
                            _ev("https://bea.gov/b", supports=False)]])
    pack = build_evidence_pack_v2(SID, "claim", (r1, Boom()))
    assert pack.gate_code == "" and len(pack.items) == 2


# ── adjudicate partition: gated claims never reach the panel ─────────────────

class _PanelHM:
    def __init__(self):
        self.panel_item_ids = None

    def run(self, task, items, strategy, *, roster=None, tune=None, rc_id=None):
        from hydramind import ItemResult, StrategyResultKind
        self.panel_item_ids = [i["item_id"] for i in items]
        out = [ItemResult(i["item_id"], StrategyResultKind.RESOLVED,
                          {"verdict": "TRUE", "citations": ["E1"], "confidence": 0.9},
                          {"votes": {"TRUE": 2}})
               for i in items]
        return SimpleNamespace(items=out, notes=None), SimpleNamespace(total_cost_usd=0.01)


def _pack_builder_gating(gated_sids):
    def build(sid, text, context):
        from truthbot.verdict.evidence_pack import EvidencePack, PackItem
        items = [] if sid in gated_sids else [PackItem(
            pack_id="E1", source_name="BLS", source_url="https://bls.gov/x",
            tier=SourceTier.GOVERNMENT, snippet="s", retrieved_at="2026-02-24T00:00:00+00:00",
            sha256="0" * 64, supports_claim=True)]
        return EvidencePack(sid=sid, window=None, items=items,
                            gate_code=GATE_INSUFFICIENT if sid in gated_sids else "")
    return build


def test_adjudicate_forces_uv_for_gated_packs_without_panel_spend():
    hm = _PanelHM()
    claims = [{"sid": "pytest_v2:0001", "text": "a", "context": ""},
              {"sid": "pytest_v2:0002", "text": "b", "context": ""}]
    rows, manifest, notes = adjudicator.adjudicate(
        hm, claims, pack_builder=_pack_builder_gating({"pytest_v2:0002"}),
        two_stage=False)
    assert hm.panel_item_ids == ["pytest_v2:0001"]           # gated sid never ran
    by = {r["sid"]: r for r in rows}
    forced = by["pytest_v2:0002"]
    assert forced["status"] == "resolved" and forced["verdict"] == "UNVERIFIABLE"
    assert forced["provenance_code"] == GATE_INSUFFICIENT
    assert by["pytest_v2:0001"]["verdict"] == "TRUE"
    assert notes["evidence_mode"] == "shared_pack_v2"
    assert notes["gate_forced_unverifiable"] == ["pytest_v2:0002"]
    assert notes["open_book"] is True


def test_adjudicate_all_gated_skips_panel_entirely():
    hm = _PanelHM()
    claims = [{"sid": "pytest_v2:0003", "text": "c", "context": ""}]
    rows, manifest, notes = adjudicator.adjudicate(
        hm, claims, pack_builder=_pack_builder_gating({"pytest_v2:0003"}),
        two_stage=False)
    assert hm.panel_item_ids is None                          # hm.run never called
    assert manifest is None
    assert rows[0]["verdict"] == "UNVERIFIABLE"
    # callers read cost via getattr(manifest, "total_cost_usd", 0.0) — safe
    assert float(getattr(manifest, "total_cost_usd", 0.0) or 0.0) == 0.0


def test_retry_retrievers_join_only_the_rescue_round():
    """Grok-fallback (2026-07-24): the retry pool is consulted ONLY when the
    first-pass pack fails quota — quota-met claims never touch it."""
    grok_calls = []

    class _Grok:
        label = "R3"

        def shortlist(self, claim_text, *, context="", utterance=None, window=None):
            grok_calls.append(claim_text)
            return [_ev("https://apnews.com/rescue", tier=SourceTier.WIRE,
                        supports=False)]

    # quota met first pass → grok never called
    r1 = _Retriever("R1", [[_ev("https://bls.gov/a"),
                            _ev("https://bea.gov/b", supports=False)]])
    pack = build_evidence_pack_v2(SID, "healthy claim", (r1,),
                                  retry_retrievers=(r1, _Grok()))
    assert pack.gate_code == "" and grok_calls == []

    # quota unmet → grok joins the one retry and can rescue the pack
    r1_thin = _Retriever("R1", [[_ev("https://bls.gov/a")], []])
    pack = build_evidence_pack_v2(SID, "thin claim", (r1_thin,),
                                  retry_retrievers=(r1_thin, _Grok()))
    assert grok_calls == ["thin claim"]
    assert pack.gate_code == ""              # rescued: 1 gov supports + 1 wire refutes
    assert any("apnews" in it.source_url for it in pack.items)
