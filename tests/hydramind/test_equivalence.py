"""Unit-test the equivalence diff logic (the live L-P/L-B run is infra-gated)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval" / "benchmarks"))
import equivalence


def test_diff_detects_and_clears():
    lp = {"a": {"label": "check-worthy"}, "b": {"label": "opinion"}}
    lb_same = {"a": {"label": "check-worthy"}, "b": {"label": "opinion"}}
    lb_diff = {"a": {"label": "check-worthy"}, "b": {"label": "unimportant"}}
    assert equivalence.diff_outputs(lp, lb_same) == []
    m = equivalence.diff_outputs(lp, lb_diff)
    assert len(m) == 1 and m[0]["item_id"] == "b" and m[0]["L-B"] == "unimportant"
