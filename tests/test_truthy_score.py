"""
tests/test_truthy_score.py

Full test suite for truthbot.truthy.truthy_score.
Covers all required scenarios from the subagent spec.
"""
import json
import pytest
from truthbot.truthy import evaluate_truthy, TruthyVerdict, Rating
from truthbot.truthy.truthy_score import (
    HAPPY_SCORE_MIN, SAD_SCORE_MAX, POF_VETO_COUNT,
    RATING_POINTS,
)

# ---------------------------------------------------------------------------
# 1. All six Rating values coerce correctly from strings
# ---------------------------------------------------------------------------

class TestRatingCoercion:
    def test_true(self):
        assert Rating("true") == Rating.TRUE

    def test_mostly_true(self):
        assert Rating("mostly_true") == Rating.MOSTLY_TRUE

    def test_half_true(self):
        assert Rating("half_true") == Rating.HALF_TRUE

    def test_mostly_false(self):
        assert Rating("mostly_false") == Rating.MOSTLY_FALSE

    def test_false(self):
        assert Rating("false") == Rating.FALSE

    def test_pants_on_fire(self):
        assert Rating("pants_on_fire") == Rating.PANTS_ON_FIRE

    def test_all_six_covered(self):
        """Paranoia check: Rating has exactly 6 members."""
        assert len(Rating) == 6


# ---------------------------------------------------------------------------
# 2. Empty input → mood='iffy' and non-empty reasoning
# ---------------------------------------------------------------------------

class TestEmptyInput:
    def test_empty_mood_is_iffy(self):
        v = evaluate_truthy([])
        assert v.mood == "iffy"

    def test_empty_reasoning_nonempty(self):
        v = evaluate_truthy([])
        assert isinstance(v.reasoning, str) and len(v.reasoning) > 0

    def test_empty_claim_count_zero(self):
        v = evaluate_truthy([])
        assert v.claim_count == 0

    def test_empty_pof_zero(self):
        v = evaluate_truthy([])
        assert v.pants_on_fire_count == 0

    def test_empty_score_zero(self):
        v = evaluate_truthy([])
        assert v.score == 0.0


# ---------------------------------------------------------------------------
# 3. Each mood band is reachable
# ---------------------------------------------------------------------------

class TestMoodBands:
    def test_happy_reachable(self):
        # All TRUE → score = +1.0 → happy with 0 PoF
        v = evaluate_truthy([Rating.TRUE] * 4)
        assert v.mood == "happy"

    def test_iffy_reachable(self):
        # Mix that lands between thresholds
        ratings = [Rating.TRUE, Rating.HALF_TRUE, Rating.MOSTLY_FALSE, Rating.HALF_TRUE]
        v = evaluate_truthy(ratings)
        assert v.mood == "iffy"

    def test_sad_reachable(self):
        # Enough FALSE/PANTS_ON_FIRE to go sad
        ratings = [Rating.FALSE, Rating.FALSE, Rating.PANTS_ON_FIRE, Rating.MOSTLY_FALSE]
        v = evaluate_truthy(ratings)
        assert v.mood == "sad"


# ---------------------------------------------------------------------------
# 4. Pants-on-Fire veto: exactly 2 PoF → sad; 1 PoF alone does NOT trigger
# ---------------------------------------------------------------------------

class TestPantsOnFireVeto:
    def test_one_pof_does_not_veto(self):
        """1 PoF with an otherwise high score should NOT hit the veto."""
        # TRUE*4 + PANTS_ON_FIRE: score = (4*2 + 1*(-3)) / (2*5) = 5/10 = 0.50
        # score >= HAPPY_SCORE_MIN but pof=1 → should be iffy, not sad via veto
        ratings = [Rating.TRUE] * 4 + [Rating.PANTS_ON_FIRE]
        v = evaluate_truthy(ratings)
        assert v.pants_on_fire_count == 1
        # The veto requires POF_VETO_COUNT (2); 1 PoF should NOT trigger sad via veto
        # (it might be iffy or could be sad due to score, but NOT via the PoF veto)
        # Score = (8 - 3) / 10 = 0.50 → score >= HAPPY_SCORE_MIN, but pof != 0 → iffy
        assert v.mood == "iffy"

    def test_two_pof_triggers_veto(self):
        """2 PoF must trigger sad regardless of score."""
        # TRUE*4 + PoF*2: score = (8-6)/12 = +0.167 (well above SAD_SCORE_MAX)
        ratings = [Rating.TRUE] * 4 + [Rating.PANTS_ON_FIRE] * 2
        v = evaluate_truthy(ratings)
        assert v.pants_on_fire_count == 2
        assert v.mood == "sad"

    def test_three_pof_also_veto(self):
        """3 PoF also triggers sad."""
        ratings = [Rating.TRUE] * 6 + [Rating.PANTS_ON_FIRE] * 3
        v = evaluate_truthy(ratings)
        assert v.mood == "sad"

    def test_pof_veto_count_constant_is_two(self):
        assert POF_VETO_COUNT == 2


# ---------------------------------------------------------------------------
# 5. Happy requires BOTH score >= 0.50 AND zero PoF
# ---------------------------------------------------------------------------

class TestHappyConditions:
    def test_high_score_zero_pof_is_happy(self):
        # All TRUE → score=1.0, pof=0 → happy
        v = evaluate_truthy([Rating.TRUE] * 5)
        assert v.mood == "happy"
        assert v.score >= HAPPY_SCORE_MIN
        assert v.pants_on_fire_count == 0

    def test_high_score_one_pof_is_iffy_not_happy(self):
        # TRUE*4 + PoF*1 → score=0.50, pof=1 → iffy
        ratings = [Rating.TRUE] * 4 + [Rating.PANTS_ON_FIRE]
        v = evaluate_truthy(ratings)
        assert v.score >= HAPPY_SCORE_MIN
        assert v.pants_on_fire_count == 1
        assert v.mood == "iffy"

    def test_score_below_threshold_is_not_happy(self):
        # Score just below HAPPY_SCORE_MIN → iffy (not happy)
        # Need score = 0.49ish with 0 PoF
        # TRUE*5 + HALF_TRUE*1: points=10, n=6 → score=10/12 ≈ 0.833, too high
        # MOSTLY_TRUE*1 + HALF_TRUE*1: points=1, n=2, score=0.25 → iffy
        ratings = [Rating.MOSTLY_TRUE, Rating.HALF_TRUE]
        v = evaluate_truthy(ratings)
        assert v.pants_on_fire_count == 0
        assert v.score < HAPPY_SCORE_MIN
        assert v.mood == "iffy"


# ---------------------------------------------------------------------------
# 6. to_dict() is JSON-serializable and round-trips cleanly
# ---------------------------------------------------------------------------

class TestToDictSerialisation:
    def test_to_dict_json_serializable(self):
        v = evaluate_truthy([Rating.TRUE, Rating.MOSTLY_FALSE, Rating.HALF_TRUE])
        d = v.to_dict()
        serialized = json.dumps(d)   # must not raise
        assert isinstance(serialized, str)

    def test_to_dict_roundtrip(self):
        v = evaluate_truthy([Rating.TRUE, Rating.FALSE, Rating.PANTS_ON_FIRE])
        d = v.to_dict()
        roundtripped = json.loads(json.dumps(d))
        assert roundtripped["mood"] == d["mood"]
        assert roundtripped["score"] == d["score"]
        assert roundtripped["claim_count"] == d["claim_count"]
        assert roundtripped["pants_on_fire_count"] == d["pants_on_fire_count"]
        assert roundtripped["tallies"] == d["tallies"]
        assert roundtripped["reasoning"] == d["reasoning"]

    def test_to_dict_keys_present(self):
        v = evaluate_truthy([Rating.TRUE])
        d = v.to_dict()
        required_keys = {"score", "mood", "claim_count", "pants_on_fire_count", "tallies", "reasoning"}
        assert required_keys.issubset(d.keys())

    def test_to_dict_empty_input(self):
        v = evaluate_truthy([])
        d = v.to_dict()
        serialized = json.dumps(d)
        rt = json.loads(serialized)
        assert rt["mood"] == "iffy"
        assert rt["claim_count"] == 0


# ---------------------------------------------------------------------------
# 7. Invariant: mood is always one of {"happy", "iffy", "sad"}
# ---------------------------------------------------------------------------

VALID_MOODS = {"happy", "iffy", "sad"}

class TestMoodInvariant:
    @pytest.mark.parametrize("ratings", [
        [],
        [Rating.TRUE],
        [Rating.FALSE],
        [Rating.PANTS_ON_FIRE],
        [Rating.PANTS_ON_FIRE, Rating.PANTS_ON_FIRE],
        [Rating.TRUE] * 10,
        [Rating.FALSE] * 10,
        [Rating.HALF_TRUE] * 5,
        [Rating.TRUE, Rating.MOSTLY_TRUE, Rating.HALF_TRUE, Rating.MOSTLY_FALSE, Rating.FALSE, Rating.PANTS_ON_FIRE],
        [Rating.MOSTLY_TRUE] * 3 + [Rating.MOSTLY_FALSE] * 3,
    ])
    def test_mood_always_valid(self, ratings):
        v = evaluate_truthy(ratings)
        assert v.mood in VALID_MOODS

    def test_mood_from_string_ratings(self):
        ratings = ["true", "mostly_true", "half_true", "mostly_false", "false", "pants_on_fire"]
        v = evaluate_truthy(ratings)
        assert v.mood in VALID_MOODS


# ---------------------------------------------------------------------------
# 8 & 9. Boundary conditions on score thresholds
# ---------------------------------------------------------------------------

class TestScoreBoundaries:
    def _make_score(self, target_score: float) -> list:
        """
        Build a rating list that produces exactly target_score.
        score = total_points / (2 * n)  → total_points = target_score * 2 * n
        We use n=10 and mix TRUE(+2) / FALSE(-2) to hit integer targets,
        then fall back to exact construction.
        """
        raise NotImplementedError("Use hand-crafted fixtures instead.")

    def test_sad_boundary_exact(self):
        """score exactly at SAD_SCORE_MAX (-0.30) → sad."""
        # score = total_points / (2*n) = -0.30
        # total_points = -0.30 * 2 * n
        # Choose n=10: total_points = -6
        # Mix: x TRUE(+2) + y FALSE(-2) + z HALF_TRUE(0)
        # We need sum = -6 with n=10
        # e.g. 2 TRUE (+4), 5 FALSE (-10), 3 HALF_TRUE (0) → sum = -6, n=10 ✓
        ratings = [Rating.TRUE] * 2 + [Rating.FALSE] * 5 + [Rating.HALF_TRUE] * 3
        v = evaluate_truthy(ratings)
        assert abs(v.score - (-0.30)) < 1e-9, f"Expected -0.30, got {v.score}"
        assert v.mood == "sad"

    def test_just_above_sad_boundary_is_iffy(self):
        """score just above SAD_SCORE_MAX → iffy (assuming 0 PoF and score < HAPPY_SCORE_MIN)."""
        # score = -0.25: total_points = -5 with n=10
        # 2 TRUE (+4), 1 MOSTLY_TRUE (+1), 5 FALSE (-10), 2 HALF_TRUE (0) → sum=-5, n=10 ✓
        ratings = [Rating.TRUE] * 2 + [Rating.MOSTLY_TRUE] * 1 + [Rating.FALSE] * 5 + [Rating.HALF_TRUE] * 2
        v = evaluate_truthy(ratings)
        assert v.score > SAD_SCORE_MAX
        assert v.pants_on_fire_count == 0
        assert v.mood == "iffy"

    def test_happy_boundary_exact(self):
        """score exactly at HAPPY_SCORE_MIN (0.50) with zero PoF → happy."""
        # score = 0.50: total_points = 0.50 * 2 * n
        # n=10: total_points = 10
        # 5 TRUE (+10), 5 HALF_TRUE (0) → sum=10, n=10, pof=0 ✓
        ratings = [Rating.TRUE] * 5 + [Rating.HALF_TRUE] * 5
        v = evaluate_truthy(ratings)
        assert abs(v.score - 0.50) < 1e-9, f"Expected 0.50, got {v.score}"
        assert v.pants_on_fire_count == 0
        assert v.mood == "happy"

    def test_just_below_happy_boundary_is_iffy(self):
        """score just below HAPPY_SCORE_MIN with 0 PoF → iffy."""
        # score ≈ 0.45: total_points = 9 with n=10
        # 4 TRUE (+8), 1 MOSTLY_TRUE (+1), 5 HALF_TRUE (0) → sum=9, n=10 ✓
        # score = 9/20 = 0.45
        ratings = [Rating.TRUE] * 4 + [Rating.MOSTLY_TRUE] * 1 + [Rating.HALF_TRUE] * 5
        v = evaluate_truthy(ratings)
        assert v.score < HAPPY_SCORE_MIN
        assert v.pants_on_fire_count == 0
        assert v.mood == "iffy"
