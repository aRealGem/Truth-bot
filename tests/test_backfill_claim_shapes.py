"""Claim-shape backfill (scripts/backfill_claim_shapes.py) + the
phase3_rebuild --shapes-sidecar merge — offline, $0.

The classifier is stubbed with canned A2 rows; no HydraMind, no proxy, no
key. The phase3 plan-mode / refusal paths run against a fake artifact in a
tmp pca_runs dir (the module globals are monkeypatched), so nothing touches
the real metrics/ tree.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)      # must import clean with no key present
    return mod


p3 = _load("phase3_rebuild")
bf = _load("backfill_claim_shapes")


# ── helpers ──────────────────────────────────────────────────────────────────

def _claim(sid, text="We convened a summit.", shape=None, context="ctx"):
    la = {"label": "check-worthy", "source": "A2", "claim_type": "other"}
    if shape:
        la["claim_shape"] = shape
    return {"sid": sid, "text": text, "context": context, "layer_a": la}


def _art(claims, run_id="old-run-id"):
    rows = [{"sid": c["sid"], "verdict": "TRUE", "split": False}
            for c in claims]
    return {"run_id": run_id, "meta": {"speech_id": "x"},
            "claims": claims, "rows": rows, "characterization": [],
            "evidence": {}}


def _stub_classify(shapes_by_sid, label_by_sid=None, seen=None):
    """Canned-classifier lane: returns A2-contract rows for each sentence."""
    def fn(sentences):
        if seen is not None:
            seen.extend(s["sid"] for s in sentences)
        out = []
        for s in sentences:
            label = (label_by_sid or {}).get(s["sid"], "check-worthy")
            out.append({"sid": s["sid"], "label": label,
                        "claim_type": "other" if label == "check-worthy" else None,
                        "claim_shape": (shapes_by_sid.get(s["sid"])
                                        if label == "check-worthy" else None),
                        "confidence": 0.9, "rationale": "canned",
                        "text": s["text"], "context": s.get("context", "")})
        return out
    return fn


# ── backfill: sidecar schema + tally ─────────────────────────────────────────

def test_sidecar_written_with_schema(tmp_path):
    art = _art([_claim("t:1"), _claim("t:2"), _claim("t:3", shape="c-eval")])
    path = tmp_path / "shapes_backfill_t.json"
    doc = bf.run_backfill("t", art, path,
                          _stub_classify({"t:1": "c-exist", "t:2": "c-third"}),
                          pause_s=0)
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk == doc
    assert doc["schema"] == "truthbot-shape-backfill v1"
    assert doc["speech_id"] == "t"
    assert doc["source_run"] == "old-run-id"
    assert doc["classifier"] == bf.CLASSIFIER_ID
    # only the SHAPELESS claims are classified; t:3 keeps its artifact shape
    assert doc["shapes"] == {"t:1": "c-exist", "t:2": "c-third"}
    assert doc["warnings"] == []
    assert bf.shape_tally(doc) == {"c-exist": 1, "c-third": 1}


def test_shapeless_claims_scope():
    claims = [_claim("t:1"), _claim("t:2", shape="c-eval"),
              {"sid": "t:3", "text": "no layer_a at all", "context": ""}]
    assert [c["sid"] for c in bf.shapeless_claims(claims)] == ["t:1", "t:3"]


# ── backfill: resume skips done sids (never re-spends) ───────────────────────

def test_resume_skips_done_sids(tmp_path):
    art = _art([_claim("t:1"), _claim("t:2")])
    path = tmp_path / "sc.json"
    done = bf.new_sidecar("t", "old-run-id")
    done["shapes"]["t:1"] = "c-exist"
    bf.write_sidecar(path, done)

    seen: list[str] = []
    doc = bf.run_backfill("t", art, path,
                          _stub_classify({"t:2": "c-count"}, seen=seen),
                          pause_s=0)
    assert seen == ["t:2"]                          # t:1 never re-classified
    assert doc["shapes"] == {"t:1": "c-exist", "t:2": "c-count"}


def test_resume_refuses_mismatched_sidecar(tmp_path):
    path = tmp_path / "sc.json"
    bf.write_sidecar(path, bf.new_sidecar("other_speech", "old-run-id"))
    with pytest.raises(ValueError, match="speech_id"):
        bf.run_backfill("t", _art([_claim("t:1")]), path,
                        _stub_classify({}), pause_s=0)
    bf.write_sidecar(path, bf.new_sidecar("t", "DIFFERENT-run"))
    with pytest.raises(ValueError, match="source_run"):
        bf.run_backfill("t", _art([_claim("t:1")]), path,
                        _stub_classify({}), pause_s=0)


# ── backfill: non-check-worthy label keeps the claim, warns ──────────────────

def test_non_checkworthy_kept_with_warning(tmp_path):
    art = _art([_claim("t:1"), _claim("t:2")])
    path = tmp_path / "sc.json"
    doc = bf.run_backfill(
        "t", art, path,
        _stub_classify({"t:1": "c-exist"}, label_by_sid={"t:2": "opinion"}),
        pause_s=0)
    # the claim is KEPT (present in the sidecar) at shape "" — never dropped
    assert doc["shapes"] == {"t:1": "c-exist", "t:2": ""}
    assert len(doc["warnings"]) == 1
    assert "t:2" in doc["warnings"][0] and "opinion" in doc["warnings"][0]
    assert bf.shape_tally(doc) == {"c-exist": 1, "(none — legacy)": 1}


def test_checkworthy_without_shape_kept_with_warning(tmp_path):
    art = _art([_claim("t:1")])
    doc = bf.run_backfill("t", art, tmp_path / "sc.json",
                          _stub_classify({}), pause_s=0)   # shape → None
    assert doc["shapes"] == {"t:1": ""}
    assert any("no claim_shape" in w for w in doc["warnings"])


def test_missing_row_fails_loud(tmp_path):
    art = _art([_claim("t:1")])
    with pytest.raises(RuntimeError, match="no row for t:1"):
        bf.run_backfill("t", art, tmp_path / "sc.json", lambda s: [],
                        pause_s=0)


# ── backfill: shape-lint validation on the sidecar ───────────────────────────

def test_lint_forces_ministerial_with_causal_tokens(tmp_path):
    # a c-exist claim whose text carries a superlative → lint forces c-eval
    art = _art([_claim("t:1", text="We launched the largest program ever.")])
    doc = bf.run_backfill("t", art, tmp_path / "sc.json",
                          _stub_classify({"t:1": "c-exist"}), pause_s=0)
    assert doc["shapes"]["t:1"] == "c-eval"


def test_lint_pass_clears_out_of_vocab_resumed_shape(tmp_path):
    art = _art([_claim("t:1")])
    path = tmp_path / "sc.json"
    stale = bf.new_sidecar("t", "old-run-id")
    stale["shapes"]["t:1"] = "c-bogus"              # hand-edited/corrupt entry
    bf.write_sidecar(path, stale)
    doc = bf.run_backfill("t", art, path, _stub_classify({}), pause_s=0)
    assert doc["shapes"]["t:1"] == ""
    assert any("out-of-vocabulary" in w for w in doc["warnings"])


def test_estimate_cost_is_positive_and_offline():
    est = bf.estimate_cost([_claim("t:1"), _claim("t:2")])
    assert 0 < est < 0.01                            # two haiku calls


# ── phase3_rebuild: sidecar merge ────────────────────────────────────────────

def test_merge_fills_blanks_never_overrides():
    claims = [_claim("t:1"),                          # blank → filled
              _claim("t:2", shape="c-eval"),          # artifact shape wins
              {"sid": "t:3", "text": "x", "context": ""},  # no layer_a → filled
              _claim("t:4")]                          # blank, not in sidecar
    n = p3.merge_sidecar_shapes(
        claims, {"t:1": "c-exist", "t:2": "c-third", "t:3": "c-count"})
    assert n == 2
    assert claims[0]["layer_a"]["claim_shape"] == "c-exist"
    assert claims[1]["layer_a"]["claim_shape"] == "c-eval"   # NOT overridden
    assert claims[2]["layer_a"]["claim_shape"] == "c-count"
    assert "claim_shape" not in claims[3]["layer_a"]


def test_load_sidecar_shapes_validates_and_drops_empty(tmp_path):
    path = tmp_path / "sc.json"
    doc = bf.new_sidecar("t", "run-1")
    doc["shapes"] = {"t:1": "c-exist", "t:2": ""}    # "" = warned/legacy claim
    bf.write_sidecar(path, doc)
    assert p3.load_sidecar_shapes(path, "t", "run-1") == {"t:1": "c-exist"}
    with pytest.raises(ValueError, match="speech_id"):
        p3.load_sidecar_shapes(path, "other", "run-1")
    with pytest.raises(ValueError, match="source_run"):
        p3.load_sidecar_shapes(path, "t", "run-2")
    path.write_text(json.dumps({"schema": "nope", "speech_id": "t"}))
    with pytest.raises(ValueError, match="schema"):
        p3.load_sidecar_shapes(path, "t", "run-1")


# ── phase3_rebuild: --go refusal / bypass / plan print ───────────────────────

def test_shape_refusal_logic():
    msg = p3.shape_refusal(0, 182, False)
    assert msg and "LEGACY" in msg and "--legacy-quota-ok" in msg
    assert "--shapes-sidecar" in msg or "backfill_claim_shapes" in msg
    assert p3.shape_refusal(100, 182, False)          # partial coverage refuses
    assert p3.shape_refusal(182, 182, False) is None  # fully shaped → clear
    assert p3.shape_refusal(0, 182, True) is None     # deliberate legacy run


@pytest.fixture
def fake_speech(tmp_path, monkeypatch):
    """A fake trump_2026 artifact in a tmp pca_runs dir; phase3_rebuild's
    module globals are pointed at it. Registry cleared around the test."""
    from truthbot.verdict import shape_registry
    run_id = p3.SPEECHES["trump_2026"]["run_id"]
    claims = [_claim(f"trump_2026:{i:04d}") for i in range(3)]
    art_dir = tmp_path / "pca_runs"
    art_dir.mkdir()
    (art_dir / f"{run_id}.json").write_text(
        json.dumps(_art(claims, run_id=run_id)), encoding="utf-8")
    monkeypatch.setattr(p3, "PCA_RUNS_DIR", art_dir)
    monkeypatch.setattr(p3, "journal_paths",
                        lambda s: (tmp_path / f"{s}.jsonl",
                                   tmp_path / f"{s}_packs.jsonl"))
    shape_registry.clear()
    yield run_id
    shape_registry.clear()


def _sidecar_for(tmp_path, run_id, shapes):
    doc = bf.new_sidecar("trump_2026", run_id)
    doc["shapes"] = shapes
    path = tmp_path / "shapes_backfill_trump_2026.json"
    bf.write_sidecar(path, doc)
    return path


def test_plan_mode_prints_sidecar_count(fake_speech, tmp_path, monkeypatch,
                                        capsys):
    path = _sidecar_for(tmp_path, fake_speech,
                        {"trump_2026:0000": "c-exist",
                         "trump_2026:0001": "c-third"})
    monkeypatch.setattr(sys, "argv",
                        ["phase3_rebuild.py", "--speech", "trump_2026",
                         "--shapes-sidecar", str(path)])
    p3.main()
    out = capsys.readouterr().out
    assert "claim shapes registered: 2/3 (2 from sidecar)" in out
    from truthbot.verdict import shape_registry
    assert shape_registry.shape_for("trump_2026:0000") == "c-exist"
    assert shape_registry.shape_for("trump_2026:0002") == ""    # still legacy


def test_plan_mode_without_sidecar_shows_legacy(fake_speech, monkeypatch,
                                                capsys):
    monkeypatch.setattr(sys, "argv",
                        ["phase3_rebuild.py", "--speech", "trump_2026"])
    p3.main()
    out = capsys.readouterr().out
    assert "claim shapes registered: 0/3 (0 from sidecar)" in out
    assert "legacy quota" in out


def test_go_refused_without_sidecar_for_shapeless_speech(fake_speech,
                                                         monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["phase3_rebuild.py", "--speech", "trump_2026",
                         "--go", "--budget", "5"])
    with pytest.raises(SystemExit) as ei:
        p3.main()
    assert "LEGACY" in str(ei.value)                  # shape refusal, no spend


def test_go_refused_with_partial_sidecar(fake_speech, tmp_path, monkeypatch):
    path = _sidecar_for(tmp_path, fake_speech, {"trump_2026:0000": "c-exist"})
    monkeypatch.setattr(sys, "argv",
                        ["phase3_rebuild.py", "--speech", "trump_2026",
                         "--go", "--budget", "5",
                         "--shapes-sidecar", str(path)])
    with pytest.raises(SystemExit) as ei:
        p3.main()
    assert "2/3 claims have no claim shape" in str(ei.value)
    assert "LEGACY" in str(ei.value)


def test_legacy_quota_ok_bypasses_shape_refusal(fake_speech, monkeypatch):
    # With the bypass, main() gets PAST the shape refusal and into
    # run_rebuild's key gate (no key in env → BLOCKED, still $0).
    for var in ("LITELLM_TRUTHBOT_KEY", "LITELLM_PCA_KEY", "LITELLM_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(sys, "argv",
                        ["phase3_rebuild.py", "--speech", "trump_2026",
                         "--go", "--budget", "5", "--legacy-quota-ok"])
    with pytest.raises(SystemExit) as ei:
        p3.main()
    assert "BLOCKED" in str(ei.value)                 # key gate, not the refusal


def test_full_sidecar_clears_refusal_then_key_gate(fake_speech, tmp_path,
                                                   monkeypatch):
    for var in ("LITELLM_TRUTHBOT_KEY", "LITELLM_PCA_KEY", "LITELLM_KEY"):
        monkeypatch.delenv(var, raising=False)
    path = _sidecar_for(tmp_path, fake_speech,
                        {"trump_2026:0000": "c-exist",
                         "trump_2026:0001": "c-third",
                         "trump_2026:0002": "c-eval"})
    monkeypatch.setattr(sys, "argv",
                        ["phase3_rebuild.py", "--speech", "trump_2026",
                         "--go", "--budget", "5",
                         "--shapes-sidecar", str(path)])
    with pytest.raises(SystemExit) as ei:
        p3.main()
    assert "BLOCKED" in str(ei.value)                 # refusal cleared; $0 gate
