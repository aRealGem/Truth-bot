"""Offline tests for the Layer B verdict scorer — closed-book abstention semantics."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "eval" / "benchmarks" / "scorer"))

import score_verdict as sv


def _p(status, verdict):
    return {"status": status, "verdict": verdict}


def test_committed_excludes_unverifiable_and_unresolved():
    assert sv._pred_verdict(_p("resolved", "TRUE")) == "TRUE"
    assert sv._pred_verdict(_p("resolved", "MISLEADING")) == "MISLEADING"
    assert sv._pred_verdict(_p("resolved", "UNVERIFIABLE")) is None   # abstention
    assert sv._pred_verdict(_p("disagreement", None)) is None
    assert sv._pred_verdict(_p("no_label", None)) is None


def test_scoring_buckets():
    gold = {"a": "TRUE", "b": "MISLEADING", "c": "UNVERIFIABLE", "d": "FALSE", "e": "TRUE"}
    preds = {
        "a": _p("resolved", "TRUE"),          # hit
        "b": _p("resolved", "FALSE"),         # miss (committed but wrong)
        "c": _p("resolved", "UNVERIFIABLE"),  # abstain_ok (gold is UNVERIFIABLE)
        "d": _p("resolved", "UNVERIFIABLE"),  # abstain_gap (decidable gold, abstained)
        "e": _p("disagreement", None),        # abstain_gap (unresolved)
    }
    rep = sv.score_verdicts(gold, preds)
    assert rep["n"] == 5
    assert rep["hit"] == 1 and rep["miss"] == 1
    assert rep["decided"] == 2 and rep["decided_accuracy"] == 0.5
    assert rep["coverage"] == 2 / 5
    assert rep["abstain_ok"] == 1 and rep["abstain_gap"] == 2
    # confusion: gold MISLEADING predicted FALSE; gold FALSE abstained
    assert rep["confusion"]["MISLEADING"]["FALSE"] == 1
    assert rep["confusion"]["FALSE"]["ABSTAIN"] == 1
    assert rep["confusion"]["TRUE"]["TRUE"] == 1


def test_ignores_sids_without_prediction():
    gold = {"a": "TRUE", "missing": "FALSE"}
    rep = sv.score_verdicts(gold, {"a": _p("resolved", "TRUE")})
    assert rep["n"] == 1 and rep["hit"] == 1


def test_loaders_roundtrip(tmp_path):
    g = tmp_path / "g.jsonl"
    g.write_text('{"sid":"x","gold_verdict":"TRUE"}\n')
    assert sv.load_gold(g) == {"x": "TRUE"}
    p_json = tmp_path / "p.json"
    p_json.write_text('[{"sid":"x","status":"resolved","verdict":"TRUE"}]')
    assert sv.load_preds(p_json)["x"]["verdict"] == "TRUE"
    p_jsonl = tmp_path / "p.jsonl"
    p_jsonl.write_text('{"sid":"x","status":"resolved","verdict":"FALSE"}\n')
    assert sv.load_preds(p_jsonl)["x"]["verdict"] == "FALSE"


def test_seed_gold_scores_against_devlot_artifact():
    """The committed 3-row seed against the committed dev-lot verdicts: the model
    decides the general-knowledge historical claim (NATO) and abstains on the two
    that need statistical/contextual evidence — the expected closed-book shape."""
    gold = sv.load_gold()
    artifact = _ROOT / "eval" / "benchmarks" / "examples" / "layerb-devlot-verdicts.json"
    if not artifact.exists():          # artifact only exists after a live dev-lot run
        return
    rep = sv.score_verdicts(gold, sv.load_preds(artifact))
    assert rep["n"] == 3
    assert rep["decided_accuracy"] == 1.0      # NATO/TRUE decided correctly
    assert rep["abstain_gap"] == 2             # deficit + vax stats abstained (→ Layer C)
