"""
Named rosters (design §3.3 rotation / P96.2 role assignment) — map PCA seats to
concrete model aliases, with a per-model `roles_allowed` guard.

A roster maps seat → alias (or a list of aliases for a critic panel). Seat codes:
  P = proposer,  C = critic,  A = arbiter.

`ROLES_ALLOWED` restricts which seats a model may occupy (generalizes I1). It is a
HARD guard: a roster that seats grok as proposer/arbiter, or dsv4-flash outside
critic, fails validation. Models absent from the table are unrestricted.
Incomplete rosters (a seat left "TBD") load but are marked not-runnable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .invariants import InvariantError

SPECS_DIR = Path(__file__).parent / "specs"
SEAT_CODES = {"P": "proposer", "C": "critic", "A": "arbiter"}

# Model → set of seats it may occupy. Absent ⇒ any seat.
ROLES_ALLOWED: dict[str, set[str]] = {
    "grok": {"critic"},          # never proposes or arbitrates (I1, Principle 2)
    "dsv4-flash": {"critic"},    # China-origin: critic-only (Western-audit doctrine)
}

_TBD = {"TBD", "tbd", None, ""}


class RosterRoleError(InvariantError):
    code = "ROSTER"


@dataclass
class Roster:
    name: str
    seats: dict[str, list[str]]     # seat -> list of aliases (single seat ⇒ len 1)
    complete: bool                  # False if any seat is TBD

    def model_for(self, seat: str) -> list[str]:
        return self.seats.get(seat, [])


def _normalize(raw_seats: dict) -> tuple[dict[str, list[str]], bool]:
    seats: dict[str, list[str]] = {}
    complete = True
    for code, seat in SEAT_CODES.items():
        val = raw_seats.get(seat, raw_seats.get(code))
        if isinstance(val, list):
            seats[seat] = [v for v in val if v not in _TBD]
            if not val or len(seats[seat]) != len(val):
                complete = False
        elif val in _TBD:
            seats[seat] = []
            complete = False
        else:
            seats[seat] = [val]
    return seats, complete


def validate_roster(roster: Roster) -> None:
    """Hard guard: every concrete model must be allowed in the seat it occupies."""
    for seat, models in roster.seats.items():
        for m in models:
            allowed = ROLES_ALLOWED.get(m)
            if allowed is not None and seat not in allowed:
                raise RosterRoleError(
                    f"roster '{roster.name}': model '{m}' not allowed in seat "
                    f"'{seat}' (allowed: {sorted(allowed)})")


def load_rosters(path: str | Path = SPECS_DIR / "rosters.yaml") -> dict[str, Roster]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    out: dict[str, Roster] = {}
    for name, seats in raw.items():
        norm, complete = _normalize(seats or {})
        r = Roster(name=name, seats=norm, complete=complete)
        validate_roster(r)          # validates concrete seats even if incomplete
        out[name] = r
    return out


def get_roster(name: str, path: str | Path = SPECS_DIR / "rosters.yaml") -> Roster:
    rs = load_rosters(path)
    if name not in rs:
        raise KeyError(f"unknown roster '{name}'; known: {sorted(rs)}")
    r = rs[name]
    if not r.complete:
        raise RosterRoleError(f"roster '{name}' is incomplete (a seat is TBD); "
                              f"not runnable until filled.")
    return r
