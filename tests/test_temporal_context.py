"""
Unit tests for the Phase 1 temporal-grounding stack.

Covers:
  * ``terms.lookup`` + ``TermRecord.display`` + ``expected_claim_window``
  * ``build_temporal_preamble`` end-to-end including speaker-alias cases,
    missing-field fall-throughs, and stability of the preamble format.
  * ``scan_text`` / ``apply_temporal_flags`` heuristic boundaries.
  * Payload-level integration: ``build_user_message`` and
    ``build_multi_user_message`` emit the preamble exactly once per payload
    and include the expected era anchor.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from truthbot.models import Claim, Confidence, ModelVerdict, VerdictLabel
from truthbot.verify.adapters.base import build_multi_user_message, build_user_message
from truthbot.verify.context import (
    apply_temporal_flags,
    build_temporal_preamble,
    scan_text,
)
from truthbot.verify.context import terms as terms_registry


# ── terms registry ────────────────────────────────────────────────────────────


class TestTermsRegistry:
    def test_lookup_trump_ii_matches_post_inauguration_date(self) -> None:
        rec = terms_registry.lookup("Donald Trump", date(2025, 3, 1))
        assert rec is not None
        assert rec.presidency_number == 47
        assert rec.term_index == 2
        assert rec.start_date == date(2025, 1, 20)

    def test_lookup_trump_i_matches_2019_date(self) -> None:
        rec = terms_registry.lookup("Donald Trump", date(2019, 6, 1))
        assert rec is not None
        assert rec.presidency_number == 45
        assert rec.term_index == 1

    def test_lookup_biden_matches_mid_term(self) -> None:
        rec = terms_registry.lookup("Joe Biden", date(2023, 6, 15))
        assert rec is not None
        assert rec.presidency_number == 46
        assert rec.start_date == date(2021, 1, 20)
        assert rec.end_date == date(2025, 1, 20)

    def test_lookup_handles_aliased_speakers(self) -> None:
        assert terms_registry.lookup(
            "President Joseph R. Biden Jr.", date(2023, 1, 1)
        ) is not None
        assert terms_registry.lookup(
            "President Donald J. Trump", date(2026, 2, 24)
        ).presidency_number == 47  # type: ignore[union-attr]

    def test_lookup_returns_none_for_pre_registry_dates(self) -> None:
        assert terms_registry.lookup("Donald Trump", date(1999, 1, 1)) is None

    def test_lookup_returns_none_for_empty_speaker(self) -> None:
        assert terms_registry.lookup("", date(2026, 1, 1)) is None

    def test_lookup_returns_none_for_non_date(self) -> None:
        assert terms_registry.lookup("Donald Trump", "not-a-date") is None

    def test_inauguration_day_belongs_to_incoming_president(self) -> None:
        """2025-01-20 is Trump-II's first day, not Biden's last."""
        rec = terms_registry.lookup("Donald Trump", date(2025, 1, 20))
        assert rec is not None and rec.term_index == 2
        rec_biden = terms_registry.lookup("Joe Biden", date(2025, 1, 20))
        assert rec_biden is None

    def test_display_format(self) -> None:
        rec = terms_registry.lookup("Donald Trump", date(2026, 2, 24))
        assert rec is not None
        assert rec.display == (
            "Donald Trump — 47th U.S. President, 2nd term "
            "(inaugurated 2025-01-20)"
        )

    def test_expected_claim_window(self) -> None:
        start, end = terms_registry.expected_claim_window(date(2026, 2, 24))
        assert start == date(2024, 1, 1)
        assert end == date(2026, 5, 1)


# ── build_temporal_preamble ──────────────────────────────────────────────────


def _sotu_claim(speech: datetime | None = None, speaker: str = "Donald Trump") -> Claim:
    return Claim(
        transcript_id="t",
        text="A claim.",
        speaker=speaker,
        speech_date=speech,
    )


class TestTemporalPreamble:
    def test_full_context_sotu_2026(self) -> None:
        claim = _sotu_claim(datetime(2026, 2, 24))
        preamble = build_temporal_preamble(claim, today=date(2026, 4, 24))
        assert "Today's date: 2026-04-24" in preamble
        assert "Speech date: 2026-02-24" in preamble
        assert "Expected evidence window: 2024-01-01 -> 2026-05-01" in preamble
        assert "Speaker: Donald Trump" in preamble
        assert "47th U.S. President, 2nd term" in preamble
        assert "inaugurated 2025-01-20" in preamble
        assert "PRIMARY EVIDENCE" in preamble
        assert "war game" in preamble  # guards C3 (Midnight Hammer-style rejection)
        assert "search results win" in preamble

    def test_unknown_speaker_omits_office_line(self) -> None:
        claim = _sotu_claim(datetime(2026, 2, 24), speaker="Unknown")
        preamble = build_temporal_preamble(claim, today=date(2026, 4, 24))
        assert "Speaker:" not in preamble
        assert "Office/term" not in preamble
        # Still has date anchors + rules.
        assert "Speech date: 2026-02-24" in preamble
        assert "PRIMARY EVIDENCE" in preamble

    def test_no_speech_date_omits_window_and_term(self) -> None:
        claim = _sotu_claim(speech=None, speaker="Donald Trump")
        preamble = build_temporal_preamble(claim, today=date(2026, 4, 24))
        assert "Speech date" not in preamble
        assert "Expected evidence window" not in preamble
        assert "Office/term" not in preamble
        # Today's date still present so model at least knows the calendar.
        assert "Today's date: 2026-04-24" in preamble

    def test_preamble_ends_with_blank_line_for_clean_concat(self) -> None:
        claim = _sotu_claim(datetime(2026, 2, 24))
        preamble = build_temporal_preamble(claim, today=date(2026, 4, 24))
        assert preamble.endswith("\n")

    # ── 2026-05-23 substance-track additions ────────────────────────────────
    # Pin the "verify, don't dismiss" + "invoke search before deciding" rules
    # that were added in response to the temporal-regressions 0/4 first run.
    # The 0/4 came from OpenAI + Gemini returning Unverifiable on post-cutoff
    # claims without ever invoking search. Rules 5 and 6 close that loophole.

    def test_preamble_pins_verify_dont_dismiss_rule(self) -> None:
        """Rule 5: a claim's date being past training cutoff is NOT grounds
        for Unverifiable. Pinned because removing it would silently re-open
        the temporal-dismissal regression.
        """
        claim = _sotu_claim(datetime(2026, 2, 24))
        preamble = build_temporal_preamble(claim, today=date(2026, 4, 24))
        assert "VERIFY, DON'T DISMISS" in preamble
        assert "is NOT grounds for 'Unverifiable'" in preamble

    def test_preamble_pins_unverifiable_reserved_for(self) -> None:
        """Rule 5 also enumerates what Unverifiable IS legitimately for —
        so the model doesn't read 'don't dismiss' as 'never return Unverifiable'.
        """
        claim = _sotu_claim(datetime(2026, 2, 24))
        preamble = build_temporal_preamble(claim, today=date(2026, 4, 24))
        assert "reserved for claims the" in preamble
        assert "web genuinely cannot resolve" in preamble

    def test_preamble_pins_concrete_failure_mode_example(self) -> None:
        """Rule 5 carries a worked example using today's date. The example
        wording is the most likely surface to drift; pin both the framing
        and that today's date appears inside it (proves it's not a constant).
        """
        claim = _sotu_claim(datetime(2026, 2, 24))
        preamble = build_temporal_preamble(claim, today=date(2026, 4, 24))
        assert "2026-04-24" in preamble
        assert "future event" in preamble  # the wrong reasoning we're naming

    def test_preamble_pins_invoke_search_before_deciding_rule(self) -> None:
        """Rule 6: a 'humble' Unverifiable without having invoked search is
        a contract violation. This is the single most direct counter to the
        regression-set's 0-tool-calls failure mode.
        """
        claim = _sotu_claim(datetime(2026, 2, 24))
        preamble = build_temporal_preamble(claim, today=date(2026, 4, 24))
        assert "INVOKE SEARCH BEFORE DECIDING" in preamble
        assert "contract violation" in preamble


# ── validator (scan_text + apply_temporal_flags) ─────────────────────────────


class TestTemporalValidator:
    speech = date(2026, 2, 24)

    def test_flags_wrong_term_reference(self) -> None:
        f = scan_text("Per 2017 BLS data, inflation was 2.1%.", self.speech)
        assert f.is_flagged
        assert 2017 in f.flagged_years
        assert "TEMPORAL_MISMATCH" in (f.format_flag() or "")

    def test_no_flag_for_in_window_years(self) -> None:
        f = scan_text("November 2025 CPI came in at 2.6%.", self.speech)
        assert not f.is_flagged
        assert 2025 in f.in_window_years

    def test_gray_zone_year_not_flagged_not_counted_in_window(self) -> None:
        """2021 sits between lookback_floor (2021) and window_start (2022).

        Not flagged (too recent to be wrong-term); not in-window either.
        Intentional tolerance buffer.
        """
        f = scan_text("Prices eased in 2021.", self.speech)
        assert not f.is_flagged

    def test_mixed_reference_is_flagged_with_in_window_note(self) -> None:
        f = scan_text("Compared to 2017, 2025 is better.", self.speech)
        assert f.is_flagged
        flag = f.format_flag() or ""
        assert "2017" in flag
        assert "Also cited in-window: 2025" in flag

    def test_apply_temporal_flags_noop_without_speech_date(self) -> None:
        claim = _sotu_claim(speech=None, speaker="Donald Trump")
        verdict = ModelVerdict(
            adapter_name="test",
            model_id="m",
            claim_id=claim.id,
            label=VerdictLabel.FALSE,
            confidence=Confidence.HIGH,
            explanation="Per 2017 data...",
        )
        apply_temporal_flags(verdict, claim)
        assert verdict.temporal_flags == []

    def test_apply_temporal_flags_attaches_and_is_idempotent(self) -> None:
        claim = _sotu_claim(datetime(2026, 2, 24))
        verdict = ModelVerdict(
            adapter_name="test",
            model_id="m",
            claim_id=claim.id,
            label=VerdictLabel.FALSE,
            confidence=Confidence.HIGH,
            explanation="In 2017 the figure was 4.4; in 2018 it was 3.9.",
        )
        apply_temporal_flags(verdict, claim)
        assert len(verdict.temporal_flags) == 1
        apply_temporal_flags(verdict, claim)  # idempotent
        assert len(verdict.temporal_flags) == 1

    # ── Phase 1c refinement: false-positive regressions from v-p1-p2 ────

    def test_year_appearing_in_claim_text_is_exempt_from_flagging(self) -> None:
        """v-p1-p2 case c33522d3: claim says 'referencing the year 1900'.

        Before the refinement: validator flagged models for citing 1900.
        After: 1900 is a claim-referenced year and must not flag.
        """
        claim_text = (
            "Trump claims the murder rate reached its lowest number in "
            "over 125 years, specifically referencing the year 1900."
        )
        explanation = (
            "Compared with historical baselines going back to 1900, "
            "the 2025 rate is the lowest in a century."
        )
        f = scan_text(explanation, self.speech, claim_text=claim_text)
        assert not f.is_flagged, f"unexpected flagged_years: {f.flagged_years}"

    def test_historical_window_phrase_extends_lookback_floor(self) -> None:
        """v-p1-p2 case 9f978c90: claim says 'lowest in more than five years'.

        The 'five years' phrasing legitimizes citing 2020 as the baseline
        for a 2026 speech. Before: 2020 flagged. After: in-comparison-window.
        """
        claim_text = (
            "Core inflation driven down to its lowest level in more than "
            "five years within 12 months."
        )
        explanation = (
            "Comparing 2020's 1.4% core CPI against 2025's 2.6% reading..."
        )
        f = scan_text(explanation, self.speech, claim_text=claim_text)
        assert not f.is_flagged

    def test_decades_phrase_extends_lookback_floor(self) -> None:
        claim_text = "First border wall construction in four decades."
        explanation = "Construction paused around 1990 per DHS historical records."
        f = scan_text(explanation, self.speech, claim_text=claim_text)
        assert not f.is_flagged

    def test_wrong_term_still_flags_when_no_historical_framing(self) -> None:
        """Negative control: the C10 pattern we DO want to catch is unchanged.

        A claim with no historical-comparison phrasing and no in-text
        year reference should still flag a 2017 citation.
        """
        claim_text = (
            "Trump claims the border is the most secure it has ever been."
        )
        explanation = "Per 2017 CBP encounters data, apprehensions were 310K."
        f = scan_text(explanation, self.speech, claim_text=claim_text)
        assert f.is_flagged
        assert 2017 in f.flagged_years

    def test_claim_year_exempt_but_other_deep_past_still_flags(self) -> None:
        """If the claim mentions 1900 but the explanation also cites 1975
        without a matching historical frame, 1975 should still flag."""
        claim_text = (
            "Lowest murder rate in 125 years, specifically referencing 1900."
        )
        explanation = "Per 1900 baseline and 1975 CDC data, the 2025 rate is low."
        # claim_lookback = 125 -> floor = 2026-125 = 1901, so 1975 is inside
        # the historical window and exempt. This confirms 'in N years'
        # phrases widen the window generously.
        f = scan_text(explanation, self.speech, claim_text=claim_text)
        assert not f.is_flagged

    def test_open_ended_record_levels_suppresses_flag(self) -> None:
        """v-p1-p2 case aed0b384: claim says 'inflation at record levels'.

        Models correctly cite 1920 and 1980 as historical inflation peaks
        to show 2025's 3.0% is NOT a record. 'Record levels' is an
        unbounded historical claim — no year should flag.
        """
        claim_text = (
            "Trump claims that when he last spoke in the chamber 12 months "
            "prior, he had inherited a nation with inflation at record levels."
        )
        explanation = (
            "The highest recorded inflation in U.S. history was 23.7% in "
            "June 1920. By January 2025 the rate was 3.0%, far from records."
        )
        f = scan_text(explanation, self.speech, claim_text=claim_text)
        assert not f.is_flagged, f"flagged: {f.flagged_years}"

    def test_open_ended_all_time_high_suppresses_flag(self) -> None:
        claim_text = "Border encounters at an all-time high."
        explanation = "Historical CBP data from 1960 onward shows..."
        f = scan_text(explanation, self.speech, claim_text=claim_text)
        assert not f.is_flagged

    def test_open_ended_most_in_history_suppresses_flag(self) -> None:
        claim_text = "Most border arrests in U.S. history."
        explanation = "Comparing 1954 Operation Wetback figures..."
        f = scan_text(explanation, self.speech, claim_text=claim_text)
        assert not f.is_flagged

    def test_open_ended_does_not_exempt_purely_wrong_term_anchoring(self) -> None:
        """Negative control: even if the claim has 'record' framing, a
        model citing a single deep-past year with NO in-window grounding
        is still suspicious. We accept this as a known permissive-mode
        false negative in exchange for eliminating the false positives
        the v-p1-p2 run exposed. Documents the trade-off explicitly.
        """
        claim_text = "Unprecedented economic growth."
        explanation = "Per 2017 GDP data, growth was 2.3%."
        f = scan_text(explanation, self.speech, claim_text=claim_text)
        # Permissive: we allow this through because 'unprecedented' is
        # an open-ended historical claim. If the model is wrong-term
        # anchored, the family-aware consensus layer (Phase 3c) will
        # catch it via disagreement with other providers.
        assert not f.is_flagged

    def test_historical_lookback_n_years_does_not_leak_in_window_years(self) -> None:
        """The extended floor must not inflate in_window_years reporting."""
        claim_text = "Lowest in 125 years."
        explanation = "1900 baseline vs 2025 level."
        f = scan_text(explanation, self.speech, claim_text=claim_text)
        # 1900 is in the extended historical window but NOT in the
        # [speech_year-4, speech_year+1] reporting window.
        assert 1900 not in f.in_window_years
        assert 2025 in f.in_window_years

    def test_apply_scans_caveats_too(self) -> None:
        claim = _sotu_claim(datetime(2026, 2, 24))
        verdict = ModelVerdict(
            adapter_name="test",
            model_id="m",
            claim_id=claim.id,
            label=VerdictLabel.TRUE,
            confidence=Confidence.HIGH,
            explanation="Current data confirms it.",
            caveats="Pre-2019 methodology differed.",
        )
        apply_temporal_flags(verdict, claim)
        assert verdict.temporal_flags  # 2019 < lookback_floor 2021


# ── payload-level integration ────────────────────────────────────────────────


class TestPayloadIntegration:
    def _claims(self, n: int = 10) -> list[Claim]:
        return [
            Claim(
                transcript_id="t",
                text=f"Claim {i} body.",
                speaker="Donald Trump",
                speech_date=datetime(2026, 2, 24),
            )
            for i in range(n)
        ]

    def test_single_claim_payload_contains_preamble_once(self) -> None:
        claim = self._claims(1)[0]
        msg = build_user_message(claim, [], inject_evidence=False)
        assert msg.count("TEMPORAL CONTEXT") == 1
        assert "47th U.S. President, 2nd term" in msg
        assert msg.startswith("TEMPORAL CONTEXT")

    def test_multi_claim_payload_contains_preamble_once(self) -> None:
        """10-claim integration sentinel: preamble appears exactly once at top.

        Critical: preamble is amortized across the batch (one copy for N
        claims), not N copies — matches the prompt-cache goal of keeping the
        per-claim token cost of temporal anchoring at 1/N.
        """
        claims = self._claims(10)
        msg = build_multi_user_message(claims, {}, inject_evidence=False)
        assert msg.count("TEMPORAL CONTEXT") == 1
        assert msg.count("47th U.S. President, 2nd term") == 1
        # Every claim body still present.
        for c in claims:
            assert c.id in msg

    def test_multi_claim_payload_preamble_precedes_claim_index_header(self) -> None:
        claims = self._claims(3)
        msg = build_multi_user_message(claims, {}, inject_evidence=False)
        preamble_pos = msg.index("TEMPORAL CONTEXT")
        claims_header_pos = msg.index("You will verify")
        assert preamble_pos < claims_header_pos


# ── regression sentinels for Phase 1a wiring ─────────────────────────────────


def test_preamble_does_not_live_in_system_prompt() -> None:
    """Caching regression guard: SYNTHESIS_SYSTEM must stay byte-stable per day."""
    from truthbot.verify.adapters.base import SYNTHESIS_SYSTEM

    # No date strings, no 'Today' markers, no claim-scoped tokens.
    assert "Today's date" not in SYNTHESIS_SYSTEM
    assert "Speech date" not in SYNTHESIS_SYSTEM
    assert "TEMPORAL CONTEXT" not in SYNTHESIS_SYSTEM


def test_preamble_is_dated_per_call() -> None:
    """Sanity: ``today`` override makes the preamble output date-controllable.

    If this breaks, something pinned today's date at import time — which would
    silently stale the anchor on long-running processes.
    """
    c = _sotu_claim(datetime(2026, 2, 24))
    assert "2025-12-31" in build_temporal_preamble(c, today=date(2025, 12, 31))
    assert "2026-01-02" in build_temporal_preamble(c, today=date(2026, 1, 2))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
