"""Render-side pins for the reason-coded species (step 6 / Wave A A3).

The render keys on the recorded axis ONLY — the map built by
``build_reason_pills`` from the fail-closed registries. Pins: the map is
exactly the A1 render set; the pill carries the CODE with the verbatim ratified
copy (+ shared footer) as tooltip and visible note; the gate-withheld copy is
suppressed on a reason-coded card (two species, one card, never both);
``reason_code_2`` (audit-only) can never render; the M-6 genre-property
disclosure derives from the same map and is silent when the species is absent.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from truthbot.models import (
    Claim,
    Confidence,
    ConsensusVerdict,
    ModelVerdict,
    VerdictBundle,
    VerdictLabel,
    VerdictProvenance,
    stable_claim_id,
)
from truthbot.publish.site import (
    GATE_INSUFFICIENT,
    SiteReport,
    _claim_card,
    _verdict_panel,
    build_reason_pills,
    set_corpus_genre_rates,
    set_reason_pills,
)

REPO = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(autouse=True)
def _clean_reason_pills():
    """Every test starts and ends with an empty module map."""
    set_reason_pills(None)
    set_corpus_genre_rates(None)
    yield
    set_reason_pills(None)
    set_corpus_genre_rates(None)


def _bundle(label: VerdictLabel = VerdictLabel.UNVERIFIABLE,
            *, gate: str = GATE_INSUFFICIENT) -> VerdictBundle:
    claim = Claim(transcript_id="t", text="A private moment nobody recorded.",
                  speaker="Barack Obama", context="ctx", category="anecdote",
                  is_checkable=True)
    mv = ModelVerdict(adapter_name="panel", model_id="m", claim_id=claim.id,
                      label=label, confidence=Confidence.HIGH, explanation="r")
    consensus = ConsensusVerdict(
        claim_id=claim.id, model_verdicts=[mv], consensus_label=label,
        consensus_verdict=label.value, confidence=Confidence.HIGH,
        agreement=True, consensus_strength="strong", explanation="x",
        provenance=VerdictProvenance(layer_a_label="check-worthy",
                                     layer_a_source="A2",
                                     layer_a_claim_type="statistical",
                                     evidence_gate=gate))
    return VerdictBundle(claim=claim, speaker="Barack Obama",
                         date_str="2014-01-28", model_verdicts=[mv],
                         consensus=consensus, sources_consulted=[])


def _site_report(bundles: list[VerdictBundle], *,
                 speech_id: str = "") -> SiteReport:
    return SiteReport(
        report_id="00000000-1111-2222-3333-444444444444",
        speaker="Barack Obama", role="President",
        date=datetime(2014, 1, 28, tzinfo=timezone.utc), venue="U.S. Capitol",
        transcript_source_url="", bundles=bundles, speech_id=speech_id)


# ── the map is the A1 render set ─────────────────────────────────────────────

def test_build_reason_pills_is_the_render_set():
    pills = build_reason_pills(REPO)
    assert len(pills) == 33
    # keyed by stable claim id, never by sid
    assert stable_claim_id("trump_2026:0153") in pills
    assert all("/" not in k and ":" not in k for k in pills)
    # spot-check a ratified assignment + the verbatim copy contract
    entry = pills[stable_claim_id("trump_2026:0153")]
    assert entry["code"] == "PRIVATE-EVENT"
    assert entry["copy"].endswith(
        "This label is re-reviewed if a qualifying source or measure is "
        "identified.")
    # the reclassified-out rows can never enter the map (A1 gate)
    assert stable_claim_id("biden_2022:0194") not in pills
    assert stable_claim_id("trump_2026:0106") not in pills
    # duals carry ONLY the primary; the audit-only secondary never enters
    dual = pills[stable_claim_id("trump_2026:0482")]
    assert dual["code"] == "PRIVATE-EVENT"
    assert "INTENT" not in dual.values()


# ── claim card ───────────────────────────────────────────────────────────────

def test_reason_pill_and_verbatim_copy_render_on_a_coded_card():
    b = _bundle()
    copy = ("This claim describes private circumstances or events that left "
            "no public record, so no qualifying source could confirm or "
            "refute it. This label is re-reviewed if a qualifying source or "
            "measure is identified.")
    set_reason_pills({b.claim.id: {"sid": "trump_2026:0153",
                                   "code": "PRIVATE-EVENT", "copy": copy}})
    html = _claim_card(b, 1, 1)
    assert "reason-code-pill" in html
    assert "PRIVATE-EVENT" in html
    assert "reason-code-note" in html
    # verbatim copy, footer included (rendered escaped; this fragment has no
    # HTML-escaping characters so it must appear byte-for-byte)
    assert ("no qualifying source could confirm or refute it. This label is "
            "re-reviewed if a qualifying source or measure is identified."
            in html)
    # the OTHER species' copy is suppressed on this card
    assert "No verdict was reached." not in html


def test_reason_code_2_never_renders():
    b = _bundle()
    # the map carries only the primary by construction; a well-formed dual
    # renders its primary and nothing else
    set_reason_pills({b.claim.id: {"sid": "trump_2026:0482",
                                   "code": "PRIVATE-EVENT", "copy": "c."}})
    html = _claim_card(b, 1, 1)
    assert "PRIVATE-EVENT" in html
    assert "INTENT" not in html  # 0482's audit-only secondary


def test_uncoded_card_keeps_the_gate_withheld_copy():
    b = _bundle()  # gate-insufficient, NOT in the map
    html = _claim_card(b, 1, 1)
    assert "No verdict was reached." in html
    assert "reason-code-pill" not in html


# ── M-6 genre-property disclosure ────────────────────────────────────────────

def test_genre_note_states_the_rate_and_the_rank():
    """Rate-based trigger + copy (owner-ratified 2026-08-24).

    The note is no longer a count share ("2 of the corpus's 3"). It states this
    speech's RATE, its rank among the speeches checked, and the median -- the
    speeches differ several-fold in length, so a count share tracked how much a
    speaker said as much as how they said it.
    """
    b1, b2 = _bundle(), _bundle()
    other = _bundle(VerdictLabel.TRUE, gate="")
    # Real speech_ids: the table is partitioned by authored class, so the
    # comparison set has to be a real class with at least _GENRE_NOTE_MIN_SPEECHES
    # members. These three are all presidential_address.
    set_reason_pills({
        b1.claim.id: {"sid": "clinton_1998:1", "code": "INTENT", "copy": "c."},
        b2.claim.id: {"sid": "clinton_1998:2", "code": "INTENT", "copy": "c."},
        "elsewhere": {"sid": "obama_2014:3", "code": "INTENT", "copy": "c."},
    })
    # clinton: 2 coded / 4 checked = 50.0%   obama: 1 / 10 = 10.0%   biden: 0 / 10 = 0.0%
    # -> clinton is top; median over [0.0, 10.0, 50.0] = 10.0
    set_corpus_genre_rates({"clinton_1998": {"checked": 4},
                            "obama_2014": {"checked": 10},
                            "biden_2022": {"checked": 10}})
    html = _verdict_panel(_site_report([b1, b2, other], speech_id="clinton_1998"))
    assert "vp-genre-note" in html
    assert "Of this speech's 4 checked claims, 2 (50.0%)" in html
    # the class label is authored and rides in the rank statement
    assert ("the highest rate of the three presidential addresses checked "
            "(median 10.0%)") in html
    # sentence 2 is verbatim and always rides with the rank statement
    assert "rhetorical genre" in html
    assert "not a finding about the speaker" in html
    # the note is NOT collapsible (M-6): a bare <p>, never a <summary>
    assert '<p class="vp-genre-note">' in html
    assert "vp-genre-summary" not in html


def test_genre_note_absent_when_this_speech_is_not_the_top_rate():
    """Only the highest-rate speech carries the note."""
    b1 = _bundle()
    set_reason_pills({
        b1.claim.id: {"sid": "clinton_1998:1", "code": "INTENT", "copy": "c."},
        "a": {"sid": "obama_2014:1", "code": "INTENT", "copy": "c."},
        "b": {"sid": "obama_2014:2", "code": "INTENT", "copy": "c."},
    })
    # clinton: 1/10 = 10%   obama: 2/4 = 50% -> obama wins, so clinton stays silent
    set_corpus_genre_rates({"clinton_1998": {"checked": 10},
                            "obama_2014": {"checked": 4},
                            "biden_2022": {"checked": 10}})
    html = _verdict_panel(_site_report([b1], speech_id="clinton_1998"))
    assert "vp-genre-note" not in html


def test_genre_note_renders_on_every_speech_tied_for_top():
    """On an exact tie for highest rate, all tied speeches carry it."""
    b1 = _bundle()
    set_reason_pills({
        b1.claim.id: {"sid": "clinton_1998:1", "code": "INTENT", "copy": "c."},
        "a": {"sid": "obama_2014:1", "code": "INTENT", "copy": "c."},
    })
    # clinton and obama tie at 10.0%; biden makes the class big enough to rank in.
    set_corpus_genre_rates({"clinton_1998": {"checked": 10},
                            "obama_2014": {"checked": 10},
                            "biden_2022": {"checked": 10}})
    html = _verdict_panel(_site_report([b1], speech_id="clinton_1998"))
    assert "vp-genre-note" in html
    assert "the highest rate of the three presidential addresses checked" in html


def test_genre_note_absent_when_species_absent():
    html = _verdict_panel(_site_report([_bundle(VerdictLabel.TRUE, gate="")]))
    assert "vp-genre-note" not in html


def test_genre_note_absent_without_a_corpus_rate_map():
    """Fail closed: a caller that does not supply the corpus map gets no note,
    rather than a rank decided from one report in isolation."""
    b1 = _bundle()
    set_reason_pills({b1.claim.id: {"sid": "mine:1", "code": "INTENT",
                                    "copy": "c."}})
    set_corpus_genre_rates(None)
    html = _verdict_panel(_site_report([b1], speech_id="mine"))
    assert "vp-genre-note" not in html
