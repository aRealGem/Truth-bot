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
