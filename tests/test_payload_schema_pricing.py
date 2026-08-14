"""D17-c wave 2 — a cost constant may not price a payload it never measured.

The failure this exists to prevent already happened. ``PACK_REUSE_USD_MEASURED``
was measured on packs whose items carried ``{id, source, tier, url, snippet}``.
D17-c added ``series_rows``, which turned out to be 91.2% of the panel payload —
an 11.3x inflation overall and 31x on ``trump_2026:0054`` alone. Priced with the
old constant, the escalation ran 8.2x over: $0.3266 against a $0.0396 estimate.

The constant did not drift. It was correct for what it measured and was applied
to something else. So the fix is not a better number, it is a refusal.

REFUSE, not warn: a warning at 21:00 on a long run is a line of scrollback.
"""
from __future__ import annotations

import pytest

from truthbot import costs


def _item(**kw) -> dict:
    d = {"id": "E1", "source": "FRED", "tier": "Government",
         "url": "https://x.gov/a", "snippet": "…"}
    d.update(kw)
    return d


def test_a_plain_pack_reads_as_the_schema_the_constant_measured() -> None:
    assert costs.payload_schema_for([_item(), _item()]) == \
        costs.PAYLOAD_SCHEMA_PACK_V2


def test_one_item_with_series_rows_changes_the_schema() -> None:
    """One is enough: the cost is driven by the rows that ARE there."""
    items = [_item(), _item(series_rows={"series_id": "CE16OV",
                                         "rows": [{"period": "2026-01-01"}]})]
    assert costs.payload_schema_for(items) == costs.PAYLOAD_SCHEMA_SERIES_V1


def test_pricing_refuses_the_exact_mistake_that_was_made() -> None:
    """PACK_REUSE_USD_MEASURED against a series-bearing payload."""
    items = [_item(series_rows={"series_id": "PAYEMS", "rows": [{"v": 1}]})]
    with pytest.raises(costs.PayloadSchemaMismatch, match="Re-measure"):
        costs.check_constant_applies("PACK_REUSE_USD_MEASURED", items)


def test_pricing_allows_the_constant_on_the_shape_it_measured() -> None:
    costs.check_constant_applies("PACK_REUSE_USD_MEASURED", [_item()])
    costs.check_constant_applies(
        "SERIES_PAYLOAD_USD_PER_CLAIM",
        [_item(series_rows={"series_id": "X", "rows": [{"v": 1}]})])


def test_the_series_constant_is_refused_on_a_plain_payload() -> None:
    """Symmetric: over-pricing a plain pack is also a category error, and it
    would quietly inflate every future estimate."""
    with pytest.raises(costs.PayloadSchemaMismatch):
        costs.check_constant_applies("SERIES_PAYLOAD_USD_PER_CLAIM", [_item()])


def test_an_unregistered_constant_is_refused_outright() -> None:
    """A constant that cannot name what it measured is a proxy, not a
    measurement — the proposed S-12 rule, enforced at the one place it bites."""
    with pytest.raises(costs.PayloadSchemaMismatch, match="declares no payload"):
        costs.check_constant_applies("SOME_UNMEASURED_GUESS", [_item()])


def test_the_measured_series_constant_matches_the_ledger() -> None:
    """The number is ledger-derived and must stay reconcilable: $0.3266 over
    3 claims and 104,547 payload characters."""
    assert costs.SERIES_PAYLOAD_USD_PER_CLAIM == pytest.approx(0.3266 / 3, abs=1e-4)
    assert costs.SERIES_PAYLOAD_USD_PER_KCHAR == pytest.approx(
        0.3266 / 104_547 * 1000, abs=1e-5)


def test_the_series_constant_is_far_above_the_one_it_replaces() -> None:
    """Guards against someone 'tidying' the new constant back toward the old
    band: they differ by 8.2x for a reason, and that reason is the payload."""
    assert costs.SERIES_PAYLOAD_USD_PER_CLAIM > costs.PACK_REUSE_USD_MEASURED[1] * 5
