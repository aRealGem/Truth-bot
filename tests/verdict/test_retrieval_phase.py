"""P120 B1 phase-split: Phase R pack building + journaling, and parity with the
inline (pre-P120) fused path. Fully offline — the retriever trio is a deterministic
fake pack_builder, so split vs inline is a byte-for-byte behavior comparison.
"""
from __future__ import annotations

from truthbot.models import SourceTier, VerdictLabel
from truthbot.verdict import publish_pipeline as pp
from truthbot.verdict import retrieval_phase as rp
from truthbot.verdict.consolidator import GATE_INSUFFICIENT
from truthbot.verdict.evidence_pack import EvidencePack, PackItem


# ── fakes ────────────────────────────────────────────────────────────────────

def _sentences(n):
    return [
        {"sid": f"sp:{i:04d}", "text": f"Metric {i} rose by {i} percent in 2026.",
         "context": f"|| Metric {i} rose by {i} percent in 2026. ||"}
        for i in range(n)
    ]


def _fake_classify_all_checkworthy(sents):
    return [{"sid": s["sid"], "label": "check-worthy", "source": "A2",
             "text": s["text"], "context": s["context"]} for s in sents]


def _pack(sid):
    return EvidencePack(sid=sid, window=None, items=[
        PackItem(pack_id="E1", source_name="BLS", source_url="https://bls.gov/x",
                 tier=SourceTier.GOVERNMENT, snippet="snip",
                 retrieved_at="2026-01-01T00:00:00+00:00", sha256="x")])


class _CountingBuilder:
    """Deterministic fake retriever trio: one call per sid, records the sids."""

    def __init__(self, gate_sids=()):
        self.calls: list[str] = []
        self.gate_sids = set(gate_sids)

    def __call__(self, sid, text, context):
        self.calls.append(sid)
        if sid in self.gate_sids:
            return EvidencePack(sid=sid, window=None, items=[],
                                gate_code=GATE_INSUFFICIENT)
        return _pack(sid)


def _adjudicate_using(pack_builder):
    """Fake adjudicate that mirrors the real one's shape: it CALLS the pack_builder
    per claim (so swapping in packs_only_builder is what's under test) and derives a
    row from the resulting pack (gated → Unverifiable, else True)."""
    def adj(chunk):
        rows, packs = [], {}
        for c in chunk:
            pack = pack_builder(c["sid"], c["text"], c.get("context", ""))
            packs[c["sid"]] = pack
            gated = bool(getattr(pack, "gate_code", ""))
            rows.append({
                "sid": c["sid"], "status": "resolved",
                "verdict": "UNVERIFIABLE" if gated else "TRUE",
                "confidence": 0.9, "citations": [it.pack_id for it in pack.items],
                "reasoning": "ok", "votes": {"TRUE": 3}})
        return rows, {"packs": packs, "cost_usd": 0.1}
    return adj


# ── build_packs_phase ─────────────────────────────────────────────────────────

def test_build_packs_phase_builds_each_claim_once():
    claims = pp.claims_from_queue(_fake_classify_all_checkworthy(_sentences(4)))
    b = _CountingBuilder()
    prog = []
    packs = rp.build_packs_phase(claims, b, on_progress=lambda i, n, s: prog.append((i, n)))
    assert sorted(b.calls) == [f"sp:{i:04d}" for i in range(4)]     # exactly once each
    assert set(packs) == {f"sp:{i:04d}" for i in range(4)}
    assert prog == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_build_packs_phase_journals_and_resumes_zero_rebuild(tmp_path):
    claims = pp.claims_from_queue(_fake_classify_all_checkworthy(_sentences(3)))
    jp = tmp_path / "s_packs.jsonl"

    b1 = _CountingBuilder()
    packs1 = rp.build_packs_phase(claims, b1, journal_path=jp)
    assert sorted(b1.calls) == [f"sp:{i:04d}" for i in range(3)]
    assert jp.exists()

    # Reload the journal and resume — nothing is rebuilt (spend already banked).
    loaded = pp.load_packs_journal(jp)
    assert set(loaded) == {f"sp:{i:04d}" for i in range(3)}
    b2 = _CountingBuilder()
    packs2 = rp.build_packs_phase(claims, b2, journal_path=jp, resume_packs=loaded)
    assert b2.calls == []                    # full resume → zero retrieval
    assert set(packs2) == set(packs1)


def test_build_packs_phase_partial_resume_builds_only_missing(tmp_path):
    claims = pp.claims_from_queue(_fake_classify_all_checkworthy(_sentences(3)))
    jp = tmp_path / "s_packs.jsonl"
    # Pretend the first claim was already built+journaled in a prior run.
    pp.append_packs_journal(jp, "sp:0000", _pack("sp:0000"))
    loaded = pp.load_packs_journal(jp)

    b = _CountingBuilder()
    packs = rp.build_packs_phase(claims, b, journal_path=jp, resume_packs=loaded)
    assert b.calls == ["sp:0001", "sp:0002"]     # only the two un-journaled sids
    assert set(packs) == {f"sp:{i:04d}" for i in range(3)}


# ── packs journal round-trip ──────────────────────────────────────────────────

def test_packs_journal_roundtrip_preserves_gate_code(tmp_path):
    jp = tmp_path / "g_packs.jsonl"
    pp.append_packs_journal(jp, "sp:0000", _pack("sp:0000"))
    pp.append_packs_journal(jp, "sp:0001",
                            EvidencePack(sid="sp:0001", window=None, items=[],
                                         gate_code=GATE_INSUFFICIENT))
    loaded = pp.load_packs_journal(jp)
    # good pack: evidence reconstructed, no gate
    assert loaded["sp:0000"].gate_code == ""
    assert [it.source_url for it in loaded["sp:0000"].items] == ["https://bls.gov/x"]
    # gate-failed pack: gate_code SURVIVES the round-trip so it still forces UV
    assert loaded["sp:0001"].gate_code == GATE_INSUFFICIENT
    assert loaded["sp:0001"].items == []


# ── packs_only_builder ────────────────────────────────────────────────────────

def test_packs_only_builder_returns_prebuilt_and_gates_on_miss():
    packs = {"sp:0000": _pack("sp:0000")}
    lookup = rp.packs_only_builder(packs)
    assert lookup("sp:0000", "t", "c") is packs["sp:0000"]        # exact hit, no copy
    miss = lookup("sp:9999", "t", "c")                            # never built
    assert miss.gate_code == GATE_INSUFFICIENT and miss.items == []


# ── prebuilt_layer_a bypass ───────────────────────────────────────────────────

def test_prebuilt_layer_a_bypasses_layer_a():
    sents = _sentences(3)
    la = pp.run_layer_a(sents, classify_fn=_fake_classify_all_checkworthy)

    called = []
    def boom(sents):
        called.append(1)
        raise AssertionError("Layer A must not re-run when prebuilt is supplied")

    res = pp.run_pca_verify(
        sents, layer_a_fn=boom, adjudicate_fn=_adjudicate_using(_CountingBuilder()),
        chunk_size=2, prebuilt_layer_a=la)
    assert called == []
    assert res.n_check_worthy == 3


# ── parity: split == inline ───────────────────────────────────────────────────

def test_split_matches_inline_bundles():
    sents = _sentences(5)

    # INLINE: retrieval happens inside the panel loop (real builder in adjudicate).
    b_inline = _CountingBuilder(gate_sids={"sp:0003"})
    res_inline = pp.run_pca_verify(
        sents, layer_a_fn=_fake_classify_all_checkworthy,
        adjudicate_fn=_adjudicate_using(b_inline), chunk_size=2)

    # SPLIT: Phase R builds all packs up front; Phase P adjudicates by lookup.
    b_split = _CountingBuilder(gate_sids={"sp:0003"})
    la = pp.run_layer_a(sents, classify_fn=_fake_classify_all_checkworthy)
    claims = pp.claims_from_queue(la.check_worthy_queue)
    packs = rp.build_packs_phase(claims, b_split)
    res_split = pp.run_pca_verify(
        sents, layer_a_fn=_fake_classify_all_checkworthy,
        adjudicate_fn=_adjudicate_using(rp.packs_only_builder(packs)),
        chunk_size=2, prebuilt_layer_a=la)

    ids = [f"sp:{i:04d}" for i in range(5)]
    # identical bundle order + verdicts (incl. the gated sp:0003 → Unverifiable)
    assert [b.consensus.claim_id for b in res_inline.bundles] == ids
    assert ([b.consensus.claim_id for b in res_split.bundles]
            == [b.consensus.claim_id for b in res_inline.bundles])
    assert ([b.consensus.consensus_label for b in res_split.bundles]
            == [b.consensus.consensus_label for b in res_inline.bundles])
    assert res_split.bundles[3].consensus.consensus_label is VerdictLabel.UNVERIFIABLE
    # split retrieved each claim EXACTLY once (Phase R) and never again in Phase P
    assert sorted(b_split.calls) == ids


# ── PR-2: pool seam ───────────────────────────────────────────────────────────

def test_build_evidence_pack_v2_uses_shortlist_runner():
    # The pool injects a concurrent runner; confirm build_evidence_pack_v2 routes its
    # per-retriever calls through whatever runner is passed (default stays serial).
    from datetime import date

    from truthbot.verdict import evidence_pack_v2, speech_context

    # the lookup key is the sid's SPEECH PREFIX ("rt", not "rt:0000") — the
    # old full-sid registration was dead weight the fail-open era gate never
    # noticed; fail-closed (remediation v2, 1.3) does.
    speech_context.register_speech_date("rt", date(2026, 1, 1))
    seen = {"pools": 0}

    class _FakeR:
        def __init__(self, label):
            self.label = label

        def shortlist(self, claim_text, *, context="", utterance=None, window=None):
            return []

    def runner(pool, call):
        seen["pools"] += 1
        return [call(r) for r in pool]

    evidence_pack_v2.build_evidence_pack_v2(
        "rt:0000", "some claim", (_FakeR("A"), _FakeR("B")),
        context="", shortlist_runner=runner)
    assert seen["pools"] >= 1                 # runner drove the fan-out


def test_build_packs_phase_governor_matches_serial(tmp_path):
    # L2 pool path (governor set) must produce the SAME packs as the serial path.
    from truthbot.verdict.pool_governor import PoolGovernor

    claims = pp.claims_from_queue(_fake_classify_all_checkworthy(_sentences(6)))

    serial = rp.build_packs_phase(claims, _CountingBuilder())

    p = tmp_path / "pressure.json"
    p.write_text('{"level":"ok","mem_avail_mb":8000,"ts":1000}')
    gov = PoolGovernor(pressure_path=str(p), now_fn=lambda: 1000.0,
                       sleep_fn=lambda s: None, pool_max=3)
    jp = tmp_path / "pooled_packs.jsonl"
    pooled = rp.build_packs_phase(claims, _CountingBuilder(), journal_path=jp,
                                  governor=gov)

    assert set(pooled) == set(serial)
    # journaled every sid exactly once despite the concurrency
    loaded = pp.load_packs_journal(jp)
    assert set(loaded) == set(serial)


def test_packs_journal_persists_pre_cap_pool_when_larger(tmp_path):
    # PR-A2.2: a pack whose builder discarded candidates at the cap journals
    # the full pre-cap pool alongside the capped evidence; packs without a
    # meaningful pool journal exactly as before (no "pool" key), and the
    # loader (which reads only "evidence") is unaffected either way.
    import json
    jp = tmp_path / "p_packs.jsonl"
    capped = _pack("sp:0000")
    pool_items = list(capped.items) + [
        PackItem(pack_id=f"E{i}", source_name="R1",
                 source_url=f"https://extra{i}.gov/x", tier=SourceTier.GOVERNMENT,
                 snippet="s", retrieved_at="2026-01-01T00:00:00Z", sha256="h")
        for i in range(2, 5)]
    with_pool = EvidencePack(sid="sp:0000", window=None,
                             items=list(capped.items), pool=pool_items)
    pp.append_packs_journal(jp, "sp:0000", with_pool)
    pp.append_packs_journal(jp, "sp:0001", _pack("sp:0001"))

    recs = [json.loads(l) for l in jp.read_text().splitlines()]
    assert len(recs[0]["pool"]) == len(pool_items)
    assert len(recs[0]["evidence"]) == len(capped.items)
    assert "pool" not in recs[1]

    loaded = pp.load_packs_journal(jp)
    assert [it.source_url for it in loaded["sp:0000"].items] == \
        [it.source_url for it in capped.items]
