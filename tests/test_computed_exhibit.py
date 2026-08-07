"""Computed exhibits (A8 / R-2) — arithmetic, admissibility, render, CI.

Three separate guarantees, deliberately kept apart:

* the ARITHMETIC is re-derived from the COMMITTED inputs on every run, with no
  network anywhere near it — so the default suite is offline and fast;
* the NETWORK re-fetch of the pinned ALFRED vintage is marked ``network`` and
  deselected by default, so a silent upstream revision is caught by an opt-in
  run rather than by turning CI into a flaky internet client;
* ADMISSIBILITY — never on a C-EVAL judgment — is asserted at the attach point
  and at the render point, because a rule enforced in only one place is a rule
  someone routes around.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

_SPEC = importlib.util.spec_from_file_location(
    "build_computed_exhibit", REPO / "scripts" / "build_computed_exhibit.py")
bce = importlib.util.module_from_spec(_SPEC)
sys.modules["build_computed_exhibit"] = bce
_SPEC.loader.exec_module(bce)          # must import clean with no key present

from truthbot.publish import computed_exhibit as ce  # noqa: E402

EXHIBIT_PATH = REPO / "metrics" / "computed_exhibits" / "cpilfesl_q4_2025_annualized.json"


@pytest.fixture(scope="module")
def exhibit() -> dict:
    return json.loads(EXHIBIT_PATH.read_text("utf-8"))


# ── the committed exhibit + its arithmetic (offline, every run) ──────────────

def test_committed_exhibit_carries_everything_r2_requires(exhibit):
    assert exhibit["series"] == "CPILFESL"
    assert exhibit["source"] == "ALFRED"
    assert exhibit["vintage_date"] == "2026-02-24"
    assert exhibit["formula"] == "(Dec/Sep)^4 - 1"
    assert exhibit["claim_ref"] == "trump_2026:0031"
    assert exhibit["inputs"] == {"2025-09-01": 330.418, "2025-12-01": 331.814}
    assert exhibit["result"] == 0.01701


def test_ci_recomputes_the_arithmetic_from_the_committed_inputs(exhibit):
    """Pure arithmetic, no network: (331.814/330.418)^4 - 1 = 1.701%. If
    someone edits the stored result without editing the inputs — or edits an
    input without rerunning the builder — this fails."""
    recomputed = bce.recompute(exhibit)
    assert round(recomputed, 5) == exhibit["result"]
    assert f"{recomputed * 100:.3f}%" == "1.701%"


def test_annualized_compounds_a_three_month_change_to_a_year():
    assert bce.annualized(100.0, 100.0) == 0.0
    # a 1% quarterly rise compounds to ~4.06% annualized, not 4.00%
    assert round(bce.annualized(100.0, 101.0), 5) == 0.04060
    with pytest.raises(ValueError):
        bce.annualized(0.0, 101.0)


def test_the_vintage_pin_is_load_bearing(exhibit):
    """The pre-revision 2026-02-09 vintage gives 330.542 → 331.860 = 1.605%,
    ~10bp below the pinned answer. This is why R-2 requires the vintage on the
    page: without it the arithmetic is reproducible only by luck."""
    pre_revision = bce.annualized(330.542, 331.860)
    assert f"{pre_revision * 100:.3f}%" == "1.605%"
    assert abs(pre_revision - exhibit["result"]) > 0.0009


def test_builder_check_mode_verifies_the_committed_file_offline(capsys):
    assert bce.main(["--check"]) == 0
    assert "OK" in capsys.readouterr().out


def test_builder_check_mode_fails_a_tampered_exhibit(tmp_path, exhibit):
    bad = dict(exhibit, result=0.99)
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(bad), "utf-8")
    assert bce.main(["--check", "--out", str(path)]) == 1


def test_alfred_url_pins_the_vintage():
    url = bce.alfred_url()
    assert "alfredgraph.csv" in url and "id=CPILFESL" in url
    assert "vintage_date=2026-02-24" in url


def test_parse_levels_reads_positionally_and_skips_gaps():
    """ALFRED names the value column after the series AND vintage
    (CPILFESL_20260224), and this vintage has an empty October cell."""
    csv_text = ("observation_date,CPILFESL_20260224\n"
                "2025-09-01,330.418\n"
                "2025-10-01,\n"
                "2025-12-01,331.814\n")
    got = bce.parse_levels(csv_text, ("2025-09-01", "2025-12-01"))
    assert got == {"2025-09-01": 330.418, "2025-12-01": 331.814}
    with pytest.raises(ValueError):
        bce.parse_levels(csv_text, ("2025-10-01",))
    with pytest.raises(ValueError):
        bce.parse_levels("nope,at,all\n1,2,3\n", ("2025-09-01",))


# ── admissibility: numeric claim-vs-series only, NEVER a C-EVAL judgment ─────

def test_exhibit_is_never_attached_to_a_c_eval_judgment(exhibit):
    """The rule the module exists to hold. C-EVAL is the evaluative shape —
    the argument is about what the words mean, not what the number is —
    and attaching five decimal places of arithmetic to it launders a
    judgement call into a computation."""
    prov: dict = {}
    with pytest.raises(ce.InadmissibleExhibit):
        ce.attach(prov, exhibit, claim_shape="c-eval")
    assert prov == {}                       # nothing written on refusal
    assert not ce.is_admissible(exhibit, claim_shape="c-eval")
    assert not ce.is_admissible(exhibit, claim_shape="C-EVAL")
    # and the renderer refuses independently of the attach point
    assert ce.exhibit_html(exhibit, claim_shape="c-eval") == ""


def test_exhibit_attaches_to_a_numeric_claim_vs_series_comparison(exhibit):
    for shape in ("", "c-count", "c-third"):
        prov = ce.attach({}, exhibit, claim_shape=shape)
        assert prov["computed_exhibit"]["result"] == exhibit["result"]
        assert ce.is_admissible(exhibit, claim_shape=shape)


def test_attach_rejects_a_malformed_exhibit(exhibit):
    for broken in (dict(exhibit, formula=""),
                   dict(exhibit, vintage_date=""),
                   dict(exhibit, inputs={"2025-12-01": 331.814})):
        with pytest.raises(ce.InadmissibleExhibit):
            ce.attach({}, broken)
        assert not ce.is_well_formed(broken)


def test_attach_writes_through_to_the_provenance_model(exhibit):
    from truthbot.models import VerdictProvenance

    prov = VerdictProvenance()
    assert prov.computed_exhibit == {}      # default: renders exactly today
    ce.attach(prov, exhibit, claim_shape="c-count")
    assert prov.computed_exhibit["vintage_date"] == "2026-02-24"


# ── render (R-2: formula + BOTH levels + vintage, all visible) ───────────────

def test_render_shows_the_formula_both_levels_and_the_vintage(exhibit):
    html = ce.exhibit_html(exhibit)
    assert "Computed exhibit" in html
    assert "(Dec/Sep)^4 - 1" in html
    assert "1.701%" in html
    for level in ("330.418", "331.814"):    # BOTH inputs, not just the answer
        assert level in html
    for day in ("2025-09-01", "2025-12-01"):
        assert day in html
    assert "2026-02-24" in html             # the vintage
    assert "CPILFESL" in html and "ALFRED" in html


def test_render_is_empty_for_a_claim_with_no_exhibit():
    assert ce.exhibit_html(None) == ""
    assert ce.exhibit_html({}) == ""
    assert ce.exhibit_html({"series": "X"}) == ""


def test_claim_card_renders_the_exhibit_only_when_one_is_attached(exhibit):
    from truthbot.publish import site

    def _card(shape: str, attach_it: bool):
        bundle = _bundle()
        bundle.consensus.provenance.layer_a_claim_shape = shape
        if attach_it:
            bundle.consensus.provenance.computed_exhibit = dict(exhibit)
        return site._claim_card(bundle, 1, 1)

    assert "computed-exhibit" not in _card("c-count", attach_it=False)
    with_exhibit = _card("c-count", attach_it=True)
    assert "computed-exhibit" in with_exhibit
    assert "330.418" in with_exhibit and "331.814" in with_exhibit
    assert "2026-02-24" in with_exhibit
    # Belt-and-braces: even a bundle that somehow carries an exhibit on a
    # C-EVAL judgment must not render one.
    assert "computed-exhibit" not in _card("c-eval", attach_it=True)


def test_the_exhibit_css_ships_with_the_site():
    from truthbot.publish import site

    assert ".computed-exhibit" in site.CSS
    assert ".ce-vintage" in site.CSS


def _bundle():
    from truthbot.models import (
        Claim,
        Confidence,
        ConsensusVerdict,
        ModelVerdict,
        VerdictBundle,
        VerdictLabel,
    )

    claim = Claim(transcript_id="t",
                  text="In the last three months of 2025 it was down to 1.7 percent.")
    mvs = [ModelVerdict(adapter_name="pca", model_id="reconciled",
                        claim_id=claim.id, label=VerdictLabel.TRUE,
                        confidence=Confidence.HIGH,
                        explanation="Three-month annualized core CPI was ~1.7%.")]
    consensus = ConsensusVerdict(
        claim_id=claim.id, model_verdicts=mvs,
        consensus_label=VerdictLabel.TRUE, consensus_verdict="True",
        confidence=Confidence.HIGH, agreement=True,
        consensus_strength="strong", explanation="Synthetic.",
        coarse_strict_label="True", coarse_strict_strength="strong")
    return VerdictBundle(claim=claim, speaker="Synthetic Speaker",
                         date_str="2026-02-24", model_verdicts=mvs,
                         consensus=consensus)


# ── the network arm: opt-in, so an upstream revision still gets caught ───────

@pytest.mark.network
def test_alfred_still_serves_the_pinned_vintage_levels(exhibit):
    """Re-fetches the pinned vintage URL and asserts the SAME levels. ALFRED
    vintages are supposed to be immutable; if this ever fails, the exhibit's
    premise — that a dated vintage is a stable citation — has broken, and the
    published number needs re-ratifying at DC-B1. Free and keyless, but
    deselected by default: `pytest -m network`."""
    fetched = bce.fetch_exhibit()
    assert fetched["inputs"] == exhibit["inputs"]
    assert fetched["result"] == exhibit["result"]
    assert fetched["vintage_date"] == exhibit["vintage_date"]
