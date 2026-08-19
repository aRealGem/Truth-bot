"""The abstention chip and its consistency-gate parser change in LOCKSTEP.

Wave A A3 renamed the chip class (vp-selfsource-chip -> vp-abstention-chip:
since D17-d it decomposes ALL honest abstentions, not just the self-sourced
sub-state) and taught ``consistency.check_report_page`` the chip's ACTUAL
grammar. The old parser expected the pre-D17-d two-term copy, so any page
carrying the gate term silently never matched — the gate was a no-op exactly
where it mattered. These tests pin that the gate now parses what site.py emits.
"""
from __future__ import annotations

from truthbot.publish.consistency import check_report_page


def _claims():
    def c(strict, verdict, *, selfsrc=False, gate="", anecdote=False):
        return {
            "coarse_strict_label": strict,
            "consensus_verdict": verdict,
            "provenance": {
                "self_sourced_only": selfsrc,
                "evidence_gate": gate,
                "layer_a_claim_type":
                    "personal-anecdote" if anecdote else "statistical",
            },
        }
    return [
        c("True", "True"),
        c("True", "True"),
        c("Unverifiable", "Unverifiable", selfsrc=True,
          gate="insufficient-qualifying-evidence"),
        c("Unverifiable", "Unverifiable",
          gate="insufficient-qualifying-evidence"),
        c("Unverifiable", "Unverifiable"),  # other: uv, no gate code
    ]


def _page(chip_text: str) -> str:
    return f'<p class="vp-abstention-chip" title="t">{chip_text}</p>'


_CHIP_OK = ("2 decided · 1 unverified — self-sourced only · "
            "1 insufficient qualifying evidence retrieved · "
            "1 unverifiable — other")


def _chip_violations(page, claims):
    report = {"id": "r", "claim_count": len(claims)}
    return [v for v in check_report_page(page, report, claims)
            if "abstention chip" in v]


def test_gate_parses_the_current_chip_grammar():
    assert _chip_violations(_page(_CHIP_OK), _claims()) == []


def test_gate_catches_a_wrong_count():
    bad = _CHIP_OK.replace("2 decided", "3 decided")
    vios = _chip_violations(_page(bad), _claims())
    assert vios and any("derived" in v for v in vios)


def test_gate_rejects_an_unknown_term():
    vios = _chip_violations(_page("2 decided · 3 mystery things"), _claims())
    assert vios and any("unparseable" in v for v in vios)


def test_anecdote_substate_is_excluded_from_the_gate_term():
    claims = _claims()
    # flip the plain-gate row to a personal anecdote: it leaves the gate term
    # and lands in "other", exactly as the chip's own predicate chain does
    claims[3]["provenance"]["layer_a_claim_type"] = "personal-anecdote"
    chip = ("2 decided · 1 unverified — self-sourced only · "
            "2 unverifiable — other")
    assert _chip_violations(_page(chip), claims) == []
