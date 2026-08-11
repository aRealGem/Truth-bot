"""Computed exhibits — showing the arithmetic, with the vintage attached (A8 / R-2).

A fact-check that says "the right figure is 1.7%" and stops has asked the
reader to trust it. A computed exhibit shows the work instead: the formula,
both input levels, and the data vintage those levels came from. Owner ruling
R-2 requires ALL THREE on the page — the vintage is not metadata, it is part of
the answer. The same ``(Dec/Sep)^4 - 1`` over the same two months returns
1.701% on the 2026-02-24 vintage and 1.605% on the 2026-02-09 pre-revision
vintage, about 10 basis points apart. An exhibit that hid which one it ran on
would be reproducible only by coincidence.

ADMISSIBILITY — the load-bearing constraint
-------------------------------------------
A computed exhibit is admissible ONLY for a numeric claim-vs-series
comparison: a claim that states a number, checked against a published data
series. It is NEVER admissible on a C-EVAL judgment.

The reason is not stylistic. C-EVAL is the evaluative shape — "we ended DEI in
America", "this was the greatest recovery in history" — where the disagreement
is about what the words mean, not about what the number is. Bolting exact
arithmetic onto an evaluative call launders a judgement call into a
computation: the page would show five decimal places of rigour standing under
a conclusion that arithmetic cannot reach. :func:`attach` refuses, and a test
holds the refusal.

The renderer is total and defaults to nothing: an empty exhibit dict renders
the empty string, so every claim that has no exhibit renders exactly as it
does today.
"""
from __future__ import annotations

from typing import Any, Optional

#: Claim shapes a computed exhibit may NEVER be attached to (see module
#: docstring). C-EVAL is the evaluative shape; arithmetic cannot settle it.
INADMISSIBLE_SHAPES = frozenset({"c-eval"})

#: Fields R-2 requires to be present AND rendered.
REQUIRED_FIELDS = ("series", "source", "vintage_date", "inputs", "formula",
                   "result")


class InadmissibleExhibit(ValueError):
    """Raised when an exhibit is attached where arithmetic cannot decide."""


def is_well_formed(exhibit: Optional[dict]) -> bool:
    """True when the exhibit carries everything R-2 needs to render: formula,
    at least two input levels, and a vintage date."""
    if not exhibit:
        return False
    if any(not exhibit.get(f) for f in REQUIRED_FIELDS):
        return False
    return len(exhibit.get("inputs") or {}) >= 2


def is_admissible(exhibit: Optional[dict], *, claim_shape: str = "") -> bool:
    """Well-formed AND on a shape arithmetic can actually settle."""
    if not is_well_formed(exhibit):
        return False
    return (claim_shape or "").strip().lower() not in INADMISSIBLE_SHAPES


def attach(provenance: Any, exhibit: dict, *, claim_shape: str = "") -> Any:
    """Attach ``exhibit`` to a :class:`VerdictProvenance` (or a plain dict).

    Refuses on an inadmissible shape — the guard lives here, at the single
    write point, rather than in the renderer, so an exhibit can never reach a
    C-EVAL page by some other route."""
    shape = (claim_shape or "").strip().lower()
    if shape in INADMISSIBLE_SHAPES:
        raise InadmissibleExhibit(
            f"computed exhibit is not admissible on claim_shape={shape!r}: "
            "arithmetic cannot settle an evaluative claim")
    if not is_well_formed(exhibit):
        raise InadmissibleExhibit(
            "computed exhibit is missing required fields "
            f"{[f for f in REQUIRED_FIELDS if not (exhibit or {}).get(f)]} "
            "or has fewer than two input levels")
    if isinstance(provenance, dict):
        provenance["computed_exhibit"] = dict(exhibit)
    else:
        provenance.computed_exhibit = dict(exhibit)
    return provenance


def _fmt_result(result: Any) -> str:
    try:
        return f"{float(result) * 100:.3f}%"
    except (TypeError, ValueError):  # pragma: no cover — defensive
        return str(result)


def _comparison_html(exhibit: dict, esc) -> str:
    """The optional SECOND computed row: the same series, vintage and formula
    over the adjacent window (R-1, 2026-08-10).

    A claim like "it was down to 1.7 percent" has a DIRECTIONAL element, and a
    single window's rate cannot settle direction — only compare it to something.
    Left to the panel, "down" rides on model arithmetic. A second row keeps it
    on the same published series: it is the same class of evidence as the first,
    not a new kind of claim, so it earns its place in the exhibit rather than in
    the rationale.

    Absent on every exhibit that has no directional element, and renders "" —
    so this changes no existing page."""
    comp = exhibit.get("comparison") or {}
    if not comp.get("formula") or comp.get("result") is None:
        return ""
    inputs = comp.get("inputs") or {}
    rows = "".join(
        f'<li><span class="ce-date">{esc(str(day))}</span>'
        f'<span class="ce-level">{esc(str(inputs[day]))}</span></li>'
        for day in sorted(inputs))
    delta = comp.get("delta_pp")
    delta_html = ""
    if delta is not None:
        try:
            delta_html = (f'<span class="ce-delta">'
                          f'{float(delta):+.2f} pp</span>')
        except (TypeError, ValueError):  # pragma: no cover — defensive
            delta_html = ""
    return (
        '<div class="ce-comparison">'
        f'  <p class="ce-comparison-label">{esc(str(comp.get("label") or ""))}'
        f'</p>'
        f'  <p class="ce-formula">{esc(str(comp["formula"]))} '
        f'= <strong>{esc(_fmt_result(comp.get("result")))}</strong>'
        f'{delta_html}</p>'
        f'  <ul class="ce-inputs">{rows}</ul>'
        '</div>')


def exhibit_html(exhibit: Optional[dict], *, claim_shape: str = "",
                 esc=None) -> str:
    """The exhibit block: a badge plus the derivation.

    Renders the formula, BOTH input levels, and the vintage date — the three
    things R-2 requires visible. Returns "" for a missing, malformed, or
    inadmissible exhibit, so callers need no guard of their own.

    ``esc`` is the caller's HTML escaper (site.py passes its ``_esc``); it
    defaults to :func:`html.escape` so the module is usable standalone."""
    if not is_admissible(exhibit, claim_shape=claim_shape):
        return ""
    if esc is None:
        from html import escape as esc  # noqa: N806 (local alias by design)

    inputs = exhibit["inputs"]
    rows = "".join(
        f'<li><span class="ce-date">{esc(str(day))}</span>'
        f'<span class="ce-level">{esc(str(inputs[day]))}</span></li>'
        for day in sorted(inputs))
    source_url = str(exhibit.get("source_url") or "")
    vintage = esc(str(exhibit["vintage_date"]))
    vintage_html = (
        f'<a href="{esc(source_url)}" rel="nofollow noopener">{vintage}</a>'
        if source_url else vintage)
    note = str(exhibit.get("note") or "")
    note_html = f'<p class="ce-note">{esc(note)}</p>' if note else ""
    return (
        '<section class="computed-exhibit" '
        'title="Computed exhibit: the arithmetic behind this comparison, '
        'pinned to a dated data vintage.">'
        '  <div class="ce-head">'
        '    <span class="ce-badge">Computed exhibit</span>'
        f'    <span class="ce-series">{esc(str(exhibit["source"]))} '
        f'{esc(str(exhibit["series"]))}</span>'
        '  </div>'
        f'  <p class="ce-formula">{esc(str(exhibit["formula"]))} '
        f'= <strong>{esc(_fmt_result(exhibit.get("result")))}</strong></p>'
        f'  <ul class="ce-inputs">{rows}</ul>'
        f'{_comparison_html(exhibit, esc)}'
        f'  <p class="ce-vintage">Data vintage: {vintage_html}</p>'
        f'{note_html}'
        '</section>'
    )
