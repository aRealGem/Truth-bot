"""`held` removes a run from head resolution (FR-0901-04)."""
import json
import pytest
from truthbot.publish.heads import publishing_heads

GEN = "g1"


def _mk(tmp, runs, arts):
    (tmp / "methodology_manifest.json").write_text(
        json.dumps({"current_generation": GEN, "runs": runs}), encoding="utf-8")
    for rid, (sid, parent) in arts.items():
        meta = {"speech_id": sid}
        if parent:
            meta["rebuild_of"] = parent
        (tmp / f"{rid}.json").write_text(
            json.dumps({"meta": meta, "evidence": {"s:1": []}}), encoding="utf-8")
    return tmp


def test_a_held_run_is_not_a_head(tmp_path):
    d = _mk(tmp_path,
            {"r1": {"generation": GEN}, "r2": {"generation": GEN, "held": "why"}},
            {"r1": ("kept", None), "r2": ("dropped", None)})
    heads = publishing_heads(d)
    assert set(heads) == {"kept"}


def test_a_speech_whose_runs_are_all_held_is_absent_not_an_error(tmp_path):
    """Not-published-right-now is a normal state, not a broken lineage."""
    d = _mk(tmp_path, {"r1": {"generation": GEN, "held": "why"}},
            {"r1": ("gone", None)})
    assert publishing_heads(d) == {}


def test_a_held_parent_still_resolves_to_its_unheld_rebuild_child(tmp_path):
    """Holding a superseded parent must not orphan the child that replaced it."""
    d = _mk(tmp_path,
            {"p": {"generation": GEN, "held": "superseded"},
             "c": {"generation": GEN}},
            {"p": ("sp", None), "c": ("sp", "p")})
    heads = publishing_heads(d)
    assert set(heads) == {"sp"}
    assert heads["sp"].stem == "c"


def test_published_is_not_consulted(tmp_path):
    """`published` is a historical marker; the live presidential heads on main
    carry published:false and must still resolve."""
    d = _mk(tmp_path, {"r1": {"generation": GEN, "published": False}},
            {"r1": ("sp", None)})
    assert set(publishing_heads(d)) == {"sp"}
