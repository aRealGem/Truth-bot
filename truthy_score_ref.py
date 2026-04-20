"""
truthy_score.py — aggregate fact-check claims into Truthy M. McTruthface's mood.

Claim ratings follow PolitiFact's 6-tier scale. Each claim gets weighted points;
the normalized aggregate maps to Truthy's mood, with severity vetoes that catch
bad-faith deception even when the raw percentage looks "mostly okay."

Design rationale:
    * Pure percentage thresholds don't distinguish "Half True" from "Pants on
      Fire," so a speech with one egregious lie buried in mostly-true framing
      could score as "happy." The weighted scale fixes that.
    * The PANTS_ON_FIRE_VETO rule catches the classic propaganda move of
      burying one big lie in a sea of truthful padding.
    * Thresholds are calibrated so "happy" is rare and meaningful. If Truthy
      is always sad, the signal is dead.

Tune constants below if you want to recalibrate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Literal


class Rating(str, Enum):
    """PolitiFact Truth-O-Meter 6-tier rating."""
    TRUE = "true"
    MOSTLY_TRUE = "mostly_true"
    HALF_TRUE = "half_true"
    MOSTLY_FALSE = "mostly_false"
    FALSE = "false"
    PANTS_ON_FIRE = "pants_on_fire"


# Points per rating. Pants-on-Fire is penalized extra for bad faith.
RATING_POINTS: dict[Rating, int] = {
    Rating.TRUE:          +2,
    Rating.MOSTLY_TRUE:   +1,
    Rating.HALF_TRUE:      0,
    Rating.MOSTLY_FALSE:  -1,
    Rating.FALSE:         -2,
    Rating.PANTS_ON_FIRE: -3,
}

Mood = Literal["happy", "iffy", "sad"]

# --- Thresholds (the only knobs; keep symmetric across all speakers) --------
HAPPY_SCORE_MIN = 0.50   # normalized score required for happy
SAD_SCORE_MAX   = -0.30  # normalized score at/below which we go sad
POF_VETO_COUNT  = 2      # this many Pants on Fire forces sad, score be damned


@dataclass
class TruthyVerdict:
    """Aggregate verdict for a speech or set of claims."""
    score: float                              # normalized, range ~[-1.5, +1.0]
    mood: Mood                                # "happy" | "iffy" | "sad"
    claim_count: int
    pants_on_fire_count: int
    tallies: dict[str, int] = field(default_factory=dict)
    reasoning: str = ""                       # human-readable explanation

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 3),
            "mood": self.mood,
            "claim_count": self.claim_count,
            "pants_on_fire_count": self.pants_on_fire_count,
            "tallies": self.tallies,
            "reasoning": self.reasoning,
        }


def _coerce_rating(r) -> Rating:
    if isinstance(r, Rating):
        return r
    if isinstance(r, str):
        return Rating(r.strip().lower())
    raise ValueError(f"Cannot coerce {r!r} to Rating")


def evaluate_truthy(ratings: Iterable) -> TruthyVerdict:
    """
    Aggregate a sequence of claim ratings into Truthy's mood.

    Args:
        ratings: iterable of Rating enums or their string values, e.g.
                 ['true', 'mostly_true', 'half_true', 'pants_on_fire', ...]

    Returns:
        TruthyVerdict — includes normalized score, mood, tallies, reasoning.
        Pass .mood straight into the mascot via setState(verdict.mood).
    """
    rating_list = [_coerce_rating(r) for r in ratings]
    n = len(rating_list)

    if n == 0:
        return TruthyVerdict(
            score=0.0, mood="iffy", claim_count=0,
            pants_on_fire_count=0, tallies={},
            reasoning="No claims evaluated — Truthy has nothing to judge.",
        )

    tallies = {r.value: 0 for r in Rating}
    for r in rating_list:
        tallies[r.value] += 1

    pof = tallies[Rating.PANTS_ON_FIRE.value]
    total_points = sum(RATING_POINTS[r] for r in rating_list)
    score = total_points / (2 * n)   # normalize by max-possible positive

    # Mood assignment with severity vetoes ---------------------------------
    if pof >= POF_VETO_COUNT:
        mood: Mood = "sad"
        reasoning = (
            f"{pof} Pants-on-Fire rating(s) triggered the bad-faith veto "
            f"(aggregate score was {score:+.2f})."
        )
    elif score <= SAD_SCORE_MAX:
        mood = "sad"
        reasoning = (
            f"Aggregate score {score:+.2f} at or below sad threshold "
            f"({SAD_SCORE_MAX:+.2f})."
        )
    elif score >= HAPPY_SCORE_MIN and pof == 0:
        mood = "happy"
        reasoning = (
            f"Score {score:+.2f} meets happy threshold with zero fabrications."
        )
    else:
        mood = "iffy"
        if score >= HAPPY_SCORE_MIN:
            reasoning = (
                f"Score {score:+.2f} would qualify as happy, but "
                f"{pof} Pants-on-Fire rating(s) vetoed it down to iffy."
            )
        else:
            reasoning = f"Mixed aggregate score {score:+.2f} — iffy."

    return TruthyVerdict(
        score=score,
        mood=mood,
        claim_count=n,
        pants_on_fire_count=pof,
        tallies=tallies,
        reasoning=reasoning,
    )


# -- demo / sanity check ---------------------------------------------------
if __name__ == "__main__":
    import json

    scenarios = {
        "Honest policy speech": [
            Rating.TRUE, Rating.TRUE, Rating.MOSTLY_TRUE, Rating.MOSTLY_TRUE,
            Rating.HALF_TRUE, Rating.MOSTLY_TRUE,
        ],
        "Typical mixed speech": [
            Rating.TRUE, Rating.MOSTLY_TRUE, Rating.HALF_TRUE, Rating.HALF_TRUE,
            Rating.MOSTLY_FALSE, Rating.FALSE, Rating.MOSTLY_TRUE,
        ],
        "Mostly-true speech with ONE big lie": [
            Rating.TRUE, Rating.TRUE, Rating.MOSTLY_TRUE, Rating.MOSTLY_TRUE,
            Rating.PANTS_ON_FIRE,
        ],
        "Two big lies (PoF veto)": [
            Rating.TRUE, Rating.MOSTLY_TRUE, Rating.MOSTLY_TRUE,
            Rating.PANTS_ON_FIRE, Rating.PANTS_ON_FIRE,
        ],
        "Deception-heavy speech": [
            Rating.FALSE, Rating.MOSTLY_FALSE, Rating.FALSE, Rating.HALF_TRUE,
            Rating.PANTS_ON_FIRE, Rating.MOSTLY_FALSE,
        ],
        "Empty": [],
    }

    for name, ratings in scenarios.items():
        v = evaluate_truthy(ratings)
        print(f"\n=== {name} ===")
        print(json.dumps(v.to_dict(), indent=2))
