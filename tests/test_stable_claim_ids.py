"""Wave 2, lane 1 — deep links stop rotating on every republish.

``Claim.id`` defaults to a fresh ``uuid4``. That is right for a claim being
extracted for the first time and wrong for one being REBUILT from a stored
artifact, which is what the publish path does: ``site.py`` renders evidence
anchors as ``#ev-{claim.id}-E5``, so a new uuid per rebuild rotated every deep
link into every evidence pack, every time a report was republished.

The sid is already the stable key — artifacts store evidence under it and
``rows`` are keyed by it — so the id is derived from it instead.

What this file pins:
  * the derivation is deterministic and safe for BOTH uses (URL fragment and
    filename), since ``claim.id`` is also ``claims/{id}.html``;
  * no two sids in the real corpus collide, because a collision would silently
    merge two claims' deep links — the failure this lane exists to prevent;
  * the property that actually matters: rebuilding the same claim twice yields
    the same anchor, which is what "stable" has to mean operationally.

Offline — committed artifacts only, no network, no model.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from truthbot.models import (
    Claim,
    assert_stable_ids_unique,
    stable_claim_id,
)
from truthbot.verdict.bridge import _build_claim

RUNS = Path(__file__).resolve().parents[1] / "metrics" / "pca_runs"
HEADS = (
    "91dd7a34-7a3c-4f40-bcdc-276b2cb15d26",  # trump_2026
    "ddb05ee3-7d9c-4b2c-beaf-e197b9354379",  # biden_2022
    "2cbda3e4-c578-442a-aee7-c5c28a388048",  # obama_2014
    "49b2e3e8-1667-4460-8989-b265914d4450",  # clinton_1998
    "5c923c25-b065-4a9f-80bf-d23db4f9bcd1",  # gwbush_2006
)

#: The sanitiser site.py applies when building ``#ev-{id}-E5``. If the id is
#: already anchor-safe this is the identity, which is the point.
_SITE_ANCHOR_SANITISER = re.compile(r"[^A-Za-z0-9_-]")


def _all_sids() -> list[str]:
    sids: list[str] = []
    for head in HEADS:
        path = RUNS / f"{head}.json"
        if path.exists():
            sids += [c["sid"] for c in json.loads(path.read_text())["claims"]]
    return sids


# ── derivation ──────────────────────────────────────────────────────────────

def test_derivation_is_deterministic() -> None:
    assert stable_claim_id("trump_2026:0054") == stable_claim_id("trump_2026:0054")


def test_derivation_is_readable_not_opaque() -> None:
    """A debuggable anchor beats a hash: you can read the sid out of the URL."""
    assert stable_claim_id("trump_2026:0054") == "trump_2026-0054"


@pytest.mark.parametrize("sid", ["trump_2026:0054", "biden_2022:0169",
                                 "gwbush_2006:0133"])
def test_id_is_safe_as_both_url_fragment_and_filename(sid: str) -> None:
    """``claim.id`` is used as ``#ev-{id}-E5`` AND as ``claims/{id}.html``."""
    cid = stable_claim_id(sid)
    assert _SITE_ANCHOR_SANITISER.sub("-", cid) == cid, "not anchor-safe"
    assert "/" not in cid and ":" not in cid and cid not in (".", "..")
    assert cid == Path(cid).name, "not usable as a bare filename"


def test_empty_or_junk_sid_raises_rather_than_yielding_a_bare_dash() -> None:
    """Failing loudly beats every junk sid sharing the id '-'."""
    for bad in ("", "   ", ":", "::", "--"):
        with pytest.raises(ValueError):
            stable_claim_id(bad)


# ── collision safety over the REAL corpus ───────────────────────────────────

def test_no_collisions_across_the_whole_corpus() -> None:
    sids = _all_sids()
    assert sids, "no committed heads found — this test would pass vacuously"
    mapping = assert_stable_ids_unique(sids)
    assert len(set(mapping.values())) == len(set(sids))


def test_collision_is_detected_rather_than_silently_merged() -> None:
    """The guard has to actually fire, or it is decoration."""
    with pytest.raises(ValueError, match="collision"):
        assert_stable_ids_unique(["speech:0001", "speech-0001"])


# ── the property that matters ───────────────────────────────────────────────

def test_rebuilding_the_same_claim_twice_gives_the_same_anchor() -> None:
    """Two renders of one claim must produce one URL. This is the whole lane."""
    src = {"text": "More Americans are working today than at any time.",
           "context": "ctx"}
    first = _build_claim("trump_2026:0054", src)
    second = _build_claim("trump_2026:0054", src)
    assert first.id == second.id
    assert f"ev-{first.id}" == "ev-trump_2026-0054"


def test_the_default_factory_still_randomises_for_fresh_claims() -> None:
    """Only the REBUILD path is pinned; first-time extraction still gets a uuid.

    If this ever fails, two genuinely different new claims could share an id."""
    a = Claim(transcript_id="t", text="x")
    b = Claim(transcript_id="t", text="x")
    assert a.id != b.id


def test_distinct_claims_keep_distinct_anchors() -> None:
    a = _build_claim("trump_2026:0054", {"text": "a"})
    b = _build_claim("trump_2026:0055", {"text": "b"})
    assert a.id != b.id


def test_every_corpus_claim_rebuilds_to_its_own_anchor() -> None:
    """End to end over the real corpus, through the actual publish-path call."""
    sids = _all_sids()
    ids = [_build_claim(sid, {"text": "t"}).id for sid in sids]
    assert len(set(ids)) == len(set(sids))
