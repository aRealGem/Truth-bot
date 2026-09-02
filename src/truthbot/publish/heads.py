"""Deterministic publishing-head resolution — the ONE head selector (F4).

``speech_id -> the artifact a publish would render``. Every consumer that has
to agree on "which run is the head" imports from here: the renderer
(``scripts/rerender_pca_site.py``), the score-propagation merge
(``scripts/propagate_rescores.py``), the DC-6 packager
(``scripts/dc6_package.py``), the era-lint and agreed-verdict audits, and the
tests. Single-sourced so the render, the merge, and the checks cannot silently
disagree about the head.

Why not mtime. The head used to be "newest evidence-bearing artifact per speech
by ``st_mtime``, last wins". mtime is not a property of the artifact — a fresh
``git clone`` stamps every file with the checkout time, so the ordering becomes
undefined and ``tests/test_propagate_rescores`` was non-deterministic on a clean
checkout. The head is now resolved from data that travels with the repo.

The rule. Each rebuild writes a NEW run whose ``meta.rebuild_of`` points at the
head it derived from, and the manifest records its ``generation`` (see
``scripts/propagate_rescores.py`` and ``methodology_manifest.json``). Within the
sub-graph of artifacts at the manifest's ``current_generation``, the head is the
unique leaf of the ``rebuild_of`` DAG: the one current-generation artifact that
no other current-generation artifact names as its ``rebuild_of`` parent.
Restricting to the current generation keeps superseded and no-evidence roots
(older or orphaned runs of the same speech) from resurrecting a stale report —
the same failure the mtime rule was guarding against, now handled by lineage
rather than by filesystem timestamps.

There must be exactly one such leaf per speech; a fork or a dangling chain is a
build fault, not something to resolve by picking one, so this raises.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PCA_RUNS_DIR = REPO / "metrics" / "pca_runs"
MANIFEST_NAME = "methodology_manifest.json"


def _load_manifest(runs_dir: Path) -> dict:
    try:
        return json.loads((runs_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def _evidence_artifacts(runs_dir: Path) -> dict[str, tuple[str, str | None, Path]]:
    """``run_id -> (speech_id, rebuild_of, path)`` for evidence-bearing runs.

    Iterated in sorted-name order so any diagnostics are stable across
    filesystems — the resolution itself does not depend on iteration order.
    """
    arts: dict[str, tuple[str, str | None, Path]] = {}
    for p in sorted(runs_dir.glob("*.json")):
        if p.name == MANIFEST_NAME:
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if "evidence" not in d:
            continue
        meta = d.get("meta") or {}
        arts[p.stem] = (meta.get("speech_id") or p.stem,
                        meta.get("rebuild_of"), p)
    return arts


def publishing_heads(runs_dir: Path | None = None) -> dict[str, Path]:
    """``speech_id -> the artifact a publish would render`` — THE head selection.

    Deterministic and checkout-stable: the leaf of the ``rebuild_of`` DAG per
    speech, restricted to the manifest's ``current_generation``. See the module
    docstring for the rationale. Raises ``SystemExit`` if any speech does not
    have exactly one head (a lineage fork or a broken chain).
    """
    runs_dir = Path(runs_dir) if runs_dir is not None else PCA_RUNS_DIR
    manifest = _load_manifest(runs_dir)
    current_gen = manifest.get("current_generation")
    runs_meta = manifest.get("runs") or {}
    arts = _evidence_artifacts(runs_dir)

    # Candidate universe: evidence-bearing artifacts recorded at the current
    # generation. Without a manifest generation to key on there is no
    # deterministic lineage to resolve, so fail loudly rather than guess.
    if not current_gen:
        raise SystemExit(
            "publishing_heads: methodology_manifest has no current_generation; "
            "cannot resolve heads deterministically")
    # HELD runs leave the candidate universe BEFORE leaf resolution, so a held
    # artifact can neither be selected as a head nor make its speech look like
    # a lineage fork. A speech whose current-generation runs are ALL held simply
    # has no head: it is absent from the map rather than raising, because "not
    # published right now" is a normal state, not a broken lineage.
    # This is the ONE place the filter lives -- consumers do not each re-filter.
    # (`published` is deliberately NOT consulted: it is a historical marker of
    # what a publish once emitted, and the live obama/biden/trump heads carry
    # published:false on main.)
    cur = {rid for rid in arts
           if runs_meta.get(rid, {}).get("generation") == current_gen
           and not runs_meta.get(rid, {}).get("held")}

    by_speech: dict[str, list[str]] = defaultdict(list)
    for rid in cur:
        by_speech[arts[rid][0]].append(rid)
    # Parents named within the current-generation sub-graph; a leaf is a
    # current-generation artifact that is nobody's rebuild_of parent.
    parents = {arts[rid][1] for rid in cur if arts[rid][1]}

    heads: dict[str, Path] = {}
    for sid, rids in by_speech.items():
        leaves = sorted(rid for rid in rids if rid not in parents)
        if len(leaves) != 1:
            raise SystemExit(
                f"publishing_heads: {sid} has {len(leaves)} rebuild_of DAG "
                f"leaves at generation {current_gen!r} ({leaves}); the lineage "
                "must have exactly one head per speech")
        heads[sid] = arts[leaves[0]][2]
    return heads


def renderer_selection(runs_dir: Path | None = None) -> dict[str, str]:
    """``speech_id -> run id`` the renderer will choose — stems, not paths.

    A thin projection of :func:`publishing_heads` for callers that assert on run
    ids (the DC-6 packager). Sharing the resolver is the point: the assertion
    that the staged render consumed the five heads is only meaningful if it asks
    the same function the renderer does.
    """
    return {sid: path.stem for sid, path in publishing_heads(runs_dir).items()}
