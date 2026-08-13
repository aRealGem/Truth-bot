"""Wave 2, lane 2 — series rows as data, and as something a reader can check.

Stage A appended the observations to the item's snippet. That was right for a
MEASUREMENT — the flip census had to change one variable against the shipped
baseline, and the wire shape was not that variable — and wrong to ship: the
rows arrive as prose the scorer has to parse back out, and a reader gets
nothing.

Lane 2 carries them structurally on ``series_rows``, through the pack, into the
payload, and into the rendered page. What this file pins:

  * the payload key is a WHITELIST, so a golden gaining a field cannot silently
    grow every scoring prompt;
  * a non-series item is byte-unchanged, because most items are non-series and
    a regression there would be invisible and everywhere;
  * the window's own limits travel WITH the rows — how many of the full table
    are shown, the predicate that chose them, and the mismatch warning — since
    a window is only honest if what it excluded is visible beside what it kept;
  * ``window_period_mismatch`` renders as a VISIBLE warning, not a data field
    nobody reads.

Offline — no network, no model.
"""
from __future__ import annotations

import json

from truthbot.models import Evidence
from truthbot.publish.site import _series_rows_html, _sources_consulted_html
from truthbot.verify.relevance import score_payload_ex

ROWS = {
    "series_id": "CPIAUCSL",
    "rows": [{"period": "2013-11-01", "value": "233.069"},
             {"period": "2013-12-01", "value": "233.049"}],
    "window_start": "2011-12-01", "window_end": "2013-12-01",
    "rows_shown": 25, "total_rows_in_full_table": 804,
    "vintage_as_of": "2014-01-28",
    "units": None,
    "units_unavailable_because": "the authorized CSV endpoint carries no units field",
    "full_table": "https://fred.stlouisfed.org/series/CPIAUCSL",
    "selection_predicate": "25 observations <= 2014-01-28, rule 'default trailing 25'",
}


def _ev(**kw) -> Evidence:
    base = dict(claim_id="c", source_name="FRED",
                source_url="https://fred.stlouisfed.org/series/CPIAUCSL",
                snippet="stored snippet")
    base.update(kw)
    return Evidence(**base)


# ── the payload ─────────────────────────────────────────────────────────────

def test_series_rows_ride_structurally_not_in_the_snippet() -> None:
    ev = _ev(series_rows=ROWS)
    payload, meta = score_payload_ex("claim", [ev], None)
    item = json.loads(payload)["items"][0]
    assert item["snippet"] == "stored snippet", "rows must not touch the snippet"
    assert item["series_rows"]["series_id"] == "CPIAUCSL"
    assert meta[0]["has_series_rows"] is True


def test_non_series_item_payload_is_unchanged() -> None:
    """Most items are non-series; a regression here would be everywhere."""
    payload, meta = score_payload_ex("claim", [_ev()], None)
    item = json.loads(payload)["items"][0]
    assert "series_rows" not in item
    assert set(item) == {"i", "source", "snippet"}
    assert meta[0]["has_series_rows"] is False


def test_payload_keys_are_whitelisted() -> None:
    """A golden gaining a field must not silently grow every scoring prompt."""
    noisy = dict(ROWS, fixture_sha256="deadbeef", role="wave1",
                 claim_sid="obama_2014:0189", window_selection={"x": 1})
    payload, _ = score_payload_ex("claim", [_ev(series_rows=noisy)], None)
    sent = json.loads(payload)["items"][0]["series_rows"]
    for leaked in ("fixture_sha256", "role", "claim_sid", "window_selection"):
        assert leaked not in sent


def test_window_limits_travel_with_the_rows() -> None:
    """Rows without their bounds are a quotation without an ellipsis."""
    payload, _ = score_payload_ex("claim", [_ev(series_rows=ROWS)], None)
    sent = json.loads(payload)["items"][0]["series_rows"]
    assert sent["rows_shown"] == 25
    assert sent["total_rows_in_full_table"] == 804
    assert "selection_predicate" in sent
    assert "units_unavailable_because" in sent, "a null unit needs its reason"


def test_the_snippet_cap_does_not_clip_series_rows() -> None:
    """The cap governs PROSE. A table clipped mid-row still parses and lies."""
    many = dict(ROWS, rows=[{"period": f"20{i:02d}-01-01", "value": str(i)}
                            for i in range(99)])
    payload, meta = score_payload_ex("claim", [_ev(series_rows=many)], 400)
    sent = json.loads(payload)["items"][0]["series_rows"]
    assert len(sent["rows"]) == 99
    assert meta[0]["chars_truncated"] == 0


# ── the render ──────────────────────────────────────────────────────────────

def test_render_shows_what_was_left_out() -> None:
    html = _series_rows_html(ROWS)
    assert "25 of 804 rows in the full table" in html
    assert "2011-12-01 to 2013-12-01" in html
    assert "Selected by:" in html
    assert "fred.stlouisfed.org" in html


def test_render_collapses_a_long_window_but_says_how_much() -> None:
    many = dict(ROWS, rows=[{"period": f"{1939 + i}-01-01", "value": str(i)}
                            for i in range(60)])
    html = _series_rows_html(many)
    assert html.count("<tr>") - 1 == 14, "inline rows are bounded"
    assert "46 more observations" in html, "and the remainder is disclosed"


def test_mismatch_renders_as_a_visible_warning() -> None:
    """The R2 case: these rows cannot settle this claim, and saying so in a
    data field nobody opens is not disclosure."""
    html = _series_rows_html(dict(ROWS, window_period_mismatch=True,
                                  window_period_mismatch_note="Does not reach."))
    assert "series-mismatch" in html
    assert "Does not reach." in html


def test_units_absence_is_stated_not_omitted() -> None:
    html = _series_rows_html(ROWS)
    assert "Units unavailable" in html
    html_with = _series_rows_html(dict(ROWS, units="Index 1982-84=100"))
    assert "Units: Index 1982-84=100" in html_with


def test_non_series_item_renders_nothing() -> None:
    assert _series_rows_html(None) == ""
    assert _series_rows_html({}) == ""
    assert _series_rows_html({"series_id": "X", "rows": []}) == ""


def test_rows_reach_the_real_pack_renderer() -> None:
    """End to end: the field has to survive the path, not just exist."""
    html = _sources_consulted_html(
        [{"id": "E4", "source": "FRED", "url": "https://x.gov/a",
          "tier": "Government", "snippet": "s", "series_rows": ROWS}],
        anchor_base="ev-obama_2014-0189")
    assert "series-rows" in html
    assert "25 of 804 rows" in html
