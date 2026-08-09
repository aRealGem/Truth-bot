"""D15 utterance-derivative exclusion — rule table, flag, and quota effect.

Every rule is exercised on its own so a failure names the rule that broke, and
the NAMED fixtures from the directive are frozen here with the REAL urls,
snippets and dates carried by the five rebuilt runs (trump_2026:0469 E5/E7/E9,
the American Presidency Project archive rows, the Weekly Compilation issue).

The flag is the load-bearing assertion in this file: with
``TRUTHBOT_D15_UTTERANCE_RECORD`` unset — which is production — the
consolidator must behave bit-for-bit as it does today. Design/ratification
note: ``docs/decisions/D15-utterance-derivative.md``.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from truthbot.models import Evidence, SourceTier
from truthbot.verdict import era_lint, speech_context, utterance_record as ur
from truthbot.verdict.consolidator import GATE_INSUFFICIENT, consolidate
from truthbot.verdict.evidential_role import EvidentialRole

TRUMP_UTT = date(2026, 2, 24)          # trump_2026 SOTU
CLINTON_UTT = date(1998, 1, 27)        # clinton_1998 SOTU
OBAMA_UTT = date(2014, 1, 28)          # obama_2014 SOTU

# ── the real corpus rows the directive names ────────────────────────────────
E7_DCPD_URL = ("https://www.govinfo.gov/content/pkg/DCPD-202600136/pdf/"
               "DCPD-202600136.pdf")
E7_DCPD_SNIPPET = ("[2026-02-24] Official White House DCPD transcript of the "
                   "State of the Union including the quoted line about Sarah "
                   "Beckstrom.")
E9_CREC_URL = ("https://www.govinfo.gov/content/pkg/CREC-2026-02-24/pdf/"
               "CREC-2026-02-24.pdf")
E9_CREC_SNIPPET = ("[2026-02-24] Congressional Record (Senate) showing the "
                   "same State of the Union language as delivered publicly.")
E5_AP_URL = "https://apnews.com/article/fba1273cd9046e3d4c9a503297feb561"
# NOTE the curly apostrophe — it is what the stored snippet actually carries.
E5_AP_SNIPPET = ("[2026-02-25] AP recap of State of the Union moment — "
                 "documents President’s wording and the Beckstrom family "
                 "appearing in gallery.")
UCSB_ADDRESS_URL = ("https://www.presidency.ucsb.edu/documents/address-before-"
                    "joint-session-the-congress-the-state-the-union-21")
WCPD_URL = ("https://www.govinfo.gov/content/pkg/WCPD-1998-02-02/pdf/"
            "WCPD-1998-02-02-Pg129-2.pdf")
#: An ordinary statistical PDF on the SAME host as the transcript — govinfo is
#: not the signal, the package identity is.
GOVINFO_STATS_URL = ("https://www.govinfo.gov/content/pkg/BUDGET-1998-APP/pdf/"
                     "BUDGET-1998-APP-1-11.pdf")


# ── the named fixtures ──────────────────────────────────────────────────────

def test_trump_0469_e7_dcpd_transcript_is_an_utterance_record() -> None:
    assert ur.utterance_record_rule(
        E7_DCPD_URL, E7_DCPD_SNIPPET, speech_date=TRUMP_UTT,
        item_date=TRUMP_UTT) == ur.RULE_DCPD


def test_trump_0469_e9_congressional_record_is_an_utterance_record() -> None:
    assert ur.utterance_record_rule(
        E9_CREC_URL, E9_CREC_SNIPPET, speech_date=TRUMP_UTT,
        item_date=TRUMP_UTT) == ur.RULE_CREC


def test_presidency_ucsb_presidential_document_is_an_utterance_record() -> None:
    assert ur.utterance_record_rule(
        UCSB_ADDRESS_URL, "American Presidency Project archive copy.",
        speech_date=OBAMA_UTT, item_date=OBAMA_UTT) == ur.RULE_UCSB


def test_congressional_record_from_an_unrelated_date_is_not_swept_in() -> None:
    """The whole point of the date anchor: the Record is published EVERY
    sitting day, and only the day of the address is the address."""
    other = ("https://www.govinfo.gov/content/pkg/CREC-1998-03-19/pdf/"
             "CREC-1998-03-19.pdf")
    assert ur.utterance_record_rule(other, "Congressional Record, House.",
                                    speech_date=CLINTON_UTT,
                                    item_date=date(1998, 3, 19)) == ""
    # Even the very next day's Record is a different day's business.
    next_day = ("https://www.govinfo.gov/content/pkg/CREC-1998-01-28/pdf/"
                "CREC-1998-01-28.pdf")
    assert ur.utterance_record_rule(next_day, "Congressional Record, House.",
                                    speech_date=CLINTON_UTT,
                                    item_date=date(1998, 1, 28)) == ""


def test_ordinary_govinfo_statistical_pdf_is_not_an_utterance_record() -> None:
    assert ur.utterance_record_rule(
        GOVINFO_STATS_URL,
        "Budget of the United States Government, FY1998 Appendix — Department "
        "of Labor training appropriations by account.",
        speech_date=CLINTON_UTT, item_date=CLINTON_UTT) == ""


# ── rule by rule ────────────────────────────────────────────────────────────

def test_dcpd_rule_needs_the_right_year_and_a_date_inside_the_grace_day() -> None:
    assert ur.dcpd_package_year(E7_DCPD_URL) == 2026
    assert ur.dcpd_package_year(GOVINFO_STATS_URL) is None
    # The morning-after filing is still this speech (biden's DCPD-202200127
    # arrives with both 03-01 and 03-02).
    assert ur.utterance_record_rule(E7_DCPD_URL, "", speech_date=TRUMP_UTT,
                                    item_date=date(2026, 2, 25)) == ur.RULE_DCPD
    # Two days out is a different presidential document.
    assert ur.utterance_record_rule(E7_DCPD_URL, "", speech_date=TRUMP_UTT,
                                    item_date=date(2026, 2, 26)) == ""
    # A DCPD document from a previous year cannot be this address.
    prior = ("https://www.govinfo.gov/content/pkg/DCPD-202500023/pdf/"
             "DCPD-202500023.pdf")
    assert ur.utterance_record_rule(prior, "", speech_date=TRUMP_UTT,
                                    item_date=date(2025, 1, 4)) == ""
    # No usable item date -> no match. A miss, on purpose.
    assert ur.utterance_record_rule(E7_DCPD_URL, "", speech_date=TRUMP_UTT,
                                    item_date=None) == ""


def test_crec_date_is_read_from_the_package_id_not_the_metadata() -> None:
    """Retrievers disagree about ``published_at`` on the same CREC PDF, so the
    rule reads GPO's own package id instead."""
    assert ur.crec_package_date(E9_CREC_URL) == TRUMP_UTT
    congress_gov = ("https://www.congress.gov/109/crec/2006/01/31/152/9/"
                    "modified/CREC-2006-01-31-pt1-PgS366-3.htm")
    assert ur.crec_package_date(congress_gov) == date(2006, 1, 31)
    # The BOUND Record carries no date -> deliberately never matches.
    bound = ("https://www.govinfo.gov/content/pkg/GPO-CRECB-1998-pt1/pdf/"
             "GPO-CRECB-1998-pt1-1-2.pdf")
    assert ur.crec_package_date(bound) is None
    assert ur.utterance_record_rule(bound, "", speech_date=CLINTON_UTT,
                                    item_date=CLINTON_UTT) == ""
    # An impossible date in a well-formed id is refused, not raised.
    assert ur.crec_package_date("https://x/CREC-2026-13-45/y.pdf") is None
    # ...and the metadata date is irrelevant once the package id is right.
    assert ur.utterance_record_rule(E9_CREC_URL, "", speech_date=TRUMP_UTT,
                                    item_date=None) == ur.RULE_CREC


def test_wcpd_rule_covers_the_speech_week_only() -> None:
    assert ur.wcpd_package_date(WCPD_URL) == date(1998, 2, 2)
    assert ur.utterance_record_rule(WCPD_URL, "", speech_date=CLINTON_UTT,
                                    item_date=None) == ur.RULE_WCPD
    older = ("https://www.govinfo.gov/content/pkg/WCPD-1997-08-11/pdf/"
             "WCPD-1997-08-11-Pg1192.pdf")
    assert ur.utterance_record_rule(older, "", speech_date=CLINTON_UTT,
                                    item_date=None) == ""
    # An issue dated BEFORE the speech cannot contain it.
    earlier = ("https://www.govinfo.gov/content/pkg/WCPD-1998-01-26/pdf/"
               "WCPD-1998-01-26.pdf")
    assert ur.utterance_record_rule(earlier, "", speech_date=CLINTON_UTT,
                                    item_date=None) == ""


def test_ucsb_rule_needs_host_path_date_and_the_address_named() -> None:
    assert ur.is_presidency_ucsb_document(UCSB_ADDRESS_URL) is True
    assert ur.is_presidency_ucsb_document(
        "https://www.presidency.ucsb.edu/node/305034") is True
    assert ur.is_presidency_ucsb_document(
        "https://www.presidency.ucsb.edu/") is False
    assert ur.is_presidency_ucsb_document(
        "https://www.presidency.ucsb.example.com/documents/x") is False
    # Same host, same DAY, but a press release about something else — the APP
    # archives every presidential document, so naming the address is required.
    other_same_day = ("https://www.presidency.ucsb.edu/documents/fact-sheet-"
                      "the-state-the-economy-0")
    assert ur.utterance_record_rule(
        other_same_day, "Fact sheet on the economy.",
        speech_date=date(2006, 1, 31), item_date=date(2006, 1, 31)) == ""
    # Right document, wrong day -> a different year's address.
    assert ur.utterance_record_rule(UCSB_ADDRESS_URL, "",
                                    speech_date=OBAMA_UTT,
                                    item_date=date(2013, 2, 12)) == ""


def test_recap_language_needs_two_independent_cues() -> None:
    assert ur.has_recap_language(E5_AP_SNIPPET) is True
    # Recap phrasing alone is not enough...
    assert ur.has_recap_language("Full transcript of the earnings call.") is False
    # ...and naming the speech alone is not enough either.
    assert ur.has_recap_language(
        "Unemployment fell in the month of the State of the Union.") is False


def test_trump_0469_e5_ap_recap_classifies_on_language(monkeypatch) -> None:
    assert ur.utterance_record_rule(
        E5_AP_URL, E5_AP_SNIPPET, speech_date=TRUMP_UTT,
        item_date=date(2026, 2, 25)) == ur.RULE_RECAP
    # Recap language on a PRE-speech item cannot be a recap of it.
    assert ur.utterance_record_rule(
        E5_AP_URL, E5_AP_SNIPPET, speech_date=TRUMP_UTT,
        item_date=date(2026, 2, 23)) == ""


def test_nothing_matches_without_a_registered_speech_date() -> None:
    """No anchor, no exclusion — the rule refuses to guess."""
    for url, snip in ((E7_DCPD_URL, E7_DCPD_SNIPPET),
                      (E9_CREC_URL, E9_CREC_SNIPPET),
                      (WCPD_URL, ""), (E5_AP_URL, E5_AP_SNIPPET)):
        assert ur.utterance_record_rule(url, snip, speech_date=None,
                                        item_date=TRUMP_UTT) == ""


def test_role_string_and_window_constants_cannot_drift() -> None:
    assert EvidentialRole.UTTERANCE_RECORD.value == ur.ROLE == "utterance-record"
    # The recap band IS the speaker's fair-game window; restated in the leaf
    # module, pinned here.
    assert ur.RECAP_WINDOW_DAYS == era_lint.FAIR_GAME_DAYS


# ── the flag ────────────────────────────────────────────────────────────────

def test_flag_is_on_by_default_and_reads_the_env_at_call_time(monkeypatch) -> None:
    """RATIFIED 2026-08-09: unset means ON. Before that date this asserted the
    opposite, and the inversion is the whole content of the ratification — so
    it is asserted here rather than left implicit in the module constant."""
    assert ur.DEFAULT_ENABLED is True
    assert ur.RATIFIED == "2026-08-09"

    monkeypatch.delenv(ur.FLAG_ENV, raising=False)
    assert ur.flag_enabled() is True
    # Empty is not an override — it is "say nothing", which means the default.
    monkeypatch.setenv(ur.FLAG_ENV, "")
    assert ur.flag_enabled() is True
    # An explicit value overrides in BOTH directions. Anything unrecognised
    # reads as OFF, so a typo fails toward the pre-ratification gate rather
    # than toward silently keeping the new one.
    for off in ("0", "false", "no", "off", "maybe"):
        monkeypatch.setenv(ur.FLAG_ENV, off)
        assert ur.flag_enabled() is False
    for on in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv(ur.FLAG_ENV, on)
        assert ur.flag_enabled() is True


# ── consolidator effect ─────────────────────────────────────────────────────

def _ev(url: str, tier: SourceTier, snippet: str, when: date,
        supports: bool | None = True) -> Evidence:
    return Evidence(claim_id="c", source_name="R1", source_url=url,
                    source_tier=tier, snippet=snippet, supports_claim=supports,
                    published_at=datetime(when.year, when.month, when.day,
                                          tzinfo=timezone.utc))


@pytest.fixture()
def trump_0469_pack():
    """The two GOVERNMENT utterance records plus ONE genuine outside source —
    the shape that decides whether the claim can witness itself."""
    return [
        _ev("https://www.npr.org/2025/11/27/nx-s1-5622955/national-guard",
            SourceTier.ESTABLISHED, "NPR confirms the death.", date(2025, 11, 27)),
        _ev(E7_DCPD_URL, SourceTier.GOVERNMENT, E7_DCPD_SNIPPET, TRUMP_UTT),
        _ev(E9_CREC_URL, SourceTier.GOVERNMENT, E9_CREC_SNIPPET, TRUMP_UTT),
    ]


def _consolidate(pack, **kw):
    speech_context.register_speech_date("trump_2026", TRUMP_UTT)
    return consolidate("trump_2026:0469", [("stored", pack)],
                       utterance=TRUMP_UTT,
                       window=(date(2025, 1, 1), date(2026, 3, 3)), **kw)


def test_flag_off_leaves_the_gate_exactly_where_it_is(trump_0469_pack,
                                                      monkeypatch) -> None:
    """The override still reproduces the pre-ratification gate bit-for-bit —
    which is what makes a regression bisectable without reverting code."""
    monkeypatch.setenv(ur.FLAG_ENV, "0")
    res = _consolidate(trump_0469_pack)
    assert res.quota_met is True and res.gate_code == ""
    assert res.utterance_records == []
    assert [it.utterance_rule for it in res.items] == ["", "", ""]
    assert [it.role for it in res.items] == ["", "", ""]
    assert "role" not in res.to_payload()[1]


def test_flag_on_strips_the_quota_credit_from_the_speech_records(
        trump_0469_pack, monkeypatch) -> None:
    monkeypatch.setenv(ur.FLAG_ENV, "1")
    res = _consolidate(trump_0469_pack)
    # NPR alone is left, so the quota (2) is unmet and the claim is gated.
    assert res.quota_met is False
    assert res.gate_code == GATE_INSUFFICIENT
    assert [r["rule"] for r in res.utterance_records] == [ur.RULE_DCPD,
                                                          ur.RULE_CREC]


def test_the_excluded_records_are_still_kept_and_still_displayed(
        trump_0469_pack, monkeypatch) -> None:
    """Quota credit 0, display allowed — provenance survives the exclusion."""
    monkeypatch.setenv(ur.FLAG_ENV, "1")
    res = _consolidate(trump_0469_pack)
    assert len(res.items) == 3
    payload = res.to_payload()
    assert [p.get("role") for p in payload] == [None, ur.ROLE, ur.ROLE]
    assert payload[1]["tier"] == "Government"    # not demoted, just uncredited


def test_the_explicit_argument_is_the_same_switch(trump_0469_pack,
                                                  monkeypatch) -> None:
    """``utterance_record=`` overrides the env in BOTH directions, so the $0
    measurement and the tests never depend on ambient environment."""
    monkeypatch.setenv(ur.FLAG_ENV, "0")
    assert _consolidate(trump_0469_pack, utterance_record=True).quota_met is False
    monkeypatch.setenv(ur.FLAG_ENV, "1")
    assert _consolidate(trump_0469_pack, utterance_record=False).quota_met is True


def test_lenient_era_mode_cannot_launder_a_transcript_into_a_credit(
        monkeypatch) -> None:
    """Lenient mode credits a contemporaneous GOVERNMENT item even with a null
    stance — exactly the door a same-day transcript would walk through."""
    monkeypatch.setenv(ur.FLAG_ENV, "1")
    speech_context.register_speech_date("clinton_1998", CLINTON_UTT)
    pack = [
        _ev("https://www.govinfo.gov/content/pkg/CREC-1998-01-27/pdf/"
            "CREC-1998-01-27.pdf", SourceTier.GOVERNMENT,
            "Congressional Record for the day.", CLINTON_UTT, supports=None),
        _ev(WCPD_URL, SourceTier.GOVERNMENT, "Weekly Compilation.",
            date(1998, 2, 2), supports=None),
    ]
    res = consolidate("clinton_1998:0101", [("stored", pack)],
                      utterance=CLINTON_UTT,
                      window=(date(1996, 1, 1), date(1998, 3, 1)),
                      era_mode="lenient")
    assert [r["rule"] for r in res.utterance_records] == [ur.RULE_CREC,
                                                          ur.RULE_WCPD]
    assert res.quota_met is False


def test_the_d11_role_table_does_not_overwrite_an_utterance_record(
        monkeypatch) -> None:
    """A transcript is also the speaker's own record; the STRONGER label wins,
    and the role-aware quota must not hand it a primary-record slot."""
    from truthbot.verify.principals import PrincipalRelation

    monkeypatch.setenv(ur.FLAG_ENV, "1")
    pack = [_ev(E7_DCPD_URL, SourceTier.GOVERNMENT, E7_DCPD_SNIPPET, TRUMP_UTT)]
    res = _consolidate(pack, claim_shape="c-count",
                       relation_of=lambda ev: PrincipalRelation.SELF)
    assert [it.role for it in res.items] == [ur.ROLE]
    assert res.role_tally == {ur.ROLE: 1}
    assert res.quota_met is False
