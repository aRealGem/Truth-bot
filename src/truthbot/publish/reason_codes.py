"""Reason codes for the SUBSTANTIVE species — why the public record cannot reach a claim.

    {"schema": "truthbot-reason-codes v1",
     "shared_footer": "...",
     "codes": [{"code": "INTENT", "renders": true, "copy": "...",
                "precedents": [{"sid": "gwbush_2006:0033", "role": "precedent"}]},
               ...]}

WHAT THESE ARE, AND WHAT THEY ARE NOT
-------------------------------------
A reason code explains why NO SOURCE COULD SETTLE a claim. It applies only to
rows recorded as ``undecidable-from-public-record``.

It is NOT the gate-withheld copy and must never be shown in its place. Those are
two different statements and conflating them is the specific defect this whole
axis exists to prevent:

  gate-withheld  — we did not retrieve enough qualifying evidence. A statement
                   about OUR pack. The claim may be perfectly checkable.
  reason-coded   — the public record does not reach it. A statement about the
                   WORLD, and it is the stronger claim, so it needs the higher
                   bar (owner ratification) before it is ever shown.

The two species therefore never share wording. This module holds the coded copy
and nothing else holds it, so the wording cannot drift between surfaces.

COPY IS OWNER-APPROVED AND VERBATIM. It is data, not a constant, precisely so
that changing it is a reviewable diff to ``data/reason_codes.json`` rather than
an edit buried in a renderer.

FAIL CLOSED
-----------
An unknown code raises rather than degrading to blank. A row labelled with a
code nobody defined would otherwise render as an unexplained assertion that a
claim is beyond checking, which is exactly the failure this axis must not have.

``UNCODED`` is a pipeline STATE, not a label: it has no copy and never renders.
:func:`renderable` excludes it, so a caller cannot accidentally present it.
"""
from __future__ import annotations

import json
from pathlib import Path

SCHEMA = "truthbot-reason-codes v1"

#: The state that means "no code fits yet" — never rendered, never labelled.
STATE_ONLY = "UNCODED"

_REQUIRED = ("code", "renders")


class ReasonCodeError(ValueError):
    """reason_codes.json is malformed — fail the build, don't guess."""


def load_reason_codes(path: Path) -> dict:
    """Load + validate the registry. Missing file → empty registry.

    Returns ``{"shared_footer": str, "codes": {code: entry}}``."""
    path = Path(path)
    if not path.exists():
        return {"shared_footer": "", "codes": {}}
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise ReasonCodeError(f"{path}: unknown schema {doc.get('schema')!r}")

    codes: dict[str, dict] = {}
    for e in doc.get("codes") or []:
        missing = [k for k in _REQUIRED if e.get(k) is None]
        if missing:
            raise ReasonCodeError(
                f"{path}: entry {e.get('code', '?')} missing {missing}")
        code = e["code"]
        if code in codes:
            raise ReasonCodeError(f"{path}: duplicate code {code!r}")
        # A renderable code with no copy would show the reader an empty
        # explanation for the strongest claim the system makes.
        if e["renders"] and not (e.get("copy") or "").strip():
            raise ReasonCodeError(
                f"{path}: {code} renders but carries no copy")
        if not e["renders"] and (e.get("copy") or "").strip():
            raise ReasonCodeError(
                f"{path}: {code} does not render but carries copy — a "
                "non-rendering state must not accumulate reader-facing prose")
        codes[code] = e
    return {"shared_footer": doc.get("shared_footer") or "", "codes": codes}


def known(registry: dict) -> set[str]:
    """Every code the registry defines, including the non-rendering state."""
    return set(registry.get("codes") or {})


def renderable(registry: dict) -> set[str]:
    """Codes that may be shown to a reader — excludes ``UNCODED`` by construction."""
    return {c for c, e in (registry.get("codes") or {}).items() if e.get("renders")}


def copy_for(registry: dict, code: str) -> str:
    """Reader-facing copy for ``code``, with the shared footer appended.

    Raises on an unknown code and on the non-rendering state, so a caller
    cannot present either as an explanation."""
    entry = (registry.get("codes") or {}).get(code)
    if entry is None:
        raise ReasonCodeError(f"unknown reason code {code!r}")
    if not entry.get("renders"):
        raise ReasonCodeError(
            f"{code} is a pipeline state, not a label — it has no copy and "
            "must never be rendered")
    footer = registry.get("shared_footer") or ""
    body = entry["copy"]
    return f"{body} {footer}".strip() if footer else body
