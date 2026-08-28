"""Reader feedback: a prefilled form link, and deliberately nothing more.

    {"schema": "truthbot-reader-feedback v1",
     "form_url": "https://docs.google.com/forms/d/e/<ID>/viewform",
     "entries": {"claim_url": "123", "claim_id": "456", ...}}

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This builds a URL. It does not submit anything, and nothing in the published
site may. The control is a plain ``<a>``: the reader taps it, a form opens in a
new tab with the claim already identified, and they type only their opinion.

That shape was chosen over an inline one-tap rating on purpose. A fact-checking
site that fires a request every time someone reacts to a verdict has a
credibility problem, and the site currently makes ZERO data requests of any kind
(only Google Fonts, an asset fetch). Nothing here may change that: no ``fetch``,
no ``sendBeacon``, no ``<form>``, no page-load traffic. The reader's first
outbound request happens only if they choose to send feedback.

It also collects prose rather than counts, because "was this useful, and where
is it wrong" is the question being asked, and a thumbs tally cannot answer it.

FAIL CLOSED
-----------
Unconfigured means INVISIBLE, not broken and not empty. A missing file, an empty
``form_url``, or a missing ``claim_url`` entry all yield ``""`` from
:func:`prefill_url`, and the renderer then emits no element at all — not a
disabled control, not a placeholder. The rendered HTML is byte-identical to a
build without this feature, which is what lets the code ship before the form
exists.

An unknown schema raises instead: that is a malformed config, not an absent one,
and guessing at it would be how a broken link reaches readers.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote, urlencode

SCHEMA = "truthbot-reader-feedback v1"

#: Fields offered to the form, in a FIXED order. Explicit rather than dict
#: iteration so the generated query string is stable across runs and versions —
#: the rendered site is byte-reproducibility checked in CI.
FIELD_ORDER: tuple[str, ...] = (
    "claim_url", "claim_id", "claim_text", "verdict", "speaker", "speech_date",
)

#: Without this we cannot tell which claim a response is about, so an otherwise
#: complete config that omits it is treated as unconfigured.
REQUIRED_FIELD = "claim_url"

#: Claim text is prefilled as a courtesy, not as the record. Measured over the
#: published corpus (n=529): mean 117 chars, p95 227, max 502 — only 5 claims
#: (0.9%) exceed this limit. Truncation costs those five nothing, because
#: ``claim_url`` in the same response resolves the full text.
CLAIM_TEXT_LIMIT = 300

_EMPTY: dict = {"form_url": "", "entries": {}}


class ReaderFeedbackError(ValueError):
    """reader_feedback.json is malformed — fail the build, don't guess."""


def load_config(path: Path) -> dict:
    """Load + validate the config. Missing file → unconfigured (not an error).

    A missing file is the normal state for an installed package with no repo
    ``data/`` directory, so it must not raise.
    """
    path = Path(path)
    if not path.exists():
        return dict(_EMPTY)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise ReaderFeedbackError(f"{path}: unknown schema {doc.get('schema')!r}")
    entries = doc.get("entries") or {}
    if not isinstance(entries, dict):
        raise ReaderFeedbackError(f"{path}: 'entries' must be an object")
    unknown = sorted(set(entries) - set(FIELD_ORDER))
    if unknown:
        raise ReaderFeedbackError(
            f"{path}: unknown entry field(s) {unknown}; known: {list(FIELD_ORDER)}")
    return {"form_url": str(doc.get("form_url") or ""),
            "entries": {k: str(v or "") for k, v in entries.items()}}


def is_configured(cfg: dict) -> bool:
    """True when a link can actually be built."""
    return bool((cfg or {}).get("form_url")
                and (cfg or {}).get("entries", {}).get(REQUIRED_FIELD))


def truncate(text: str, limit: int = CLAIM_TEXT_LIMIT) -> str:
    """Collapse whitespace, then cut on a word boundary with an ellipsis.

    Whitespace is normalised first so the same claim yields the same string
    regardless of how the source happened to wrap it.
    """
    t = " ".join(str(text or "").split())
    if len(t) <= limit:
        return t
    cut = t[:limit].rsplit(" ", 1)[0] or t[:limit]
    return cut.rstrip(" ,;:—-") + "…"


def prefill_url(cfg: dict, **values: str) -> str:
    """Build the prefilled form URL, or ``""`` when unconfigured.

    Percent-encoding happens HERE and HTML-escaping happens in the caller, in
    that order. Reversed, an ampersand would be escaped to ``&amp;`` and then
    percent-encoded into ``%26amp%3B``, silently corrupting every field after
    the first.
    """
    if not is_configured(cfg):
        return ""
    entries = cfg.get("entries", {})
    pairs: list[tuple[str, str]] = []
    for field in FIELD_ORDER:
        entry_id = entries.get(field)
        value = values.get(field)
        if not entry_id or not value:
            continue
        if field == "claim_text":
            value = truncate(value)
        pairs.append((f"entry.{entry_id}", str(value)))
    if not pairs:
        return ""
    # quote_via=quote with safe="" yields %20 for space rather than "+", which
    # is only meaningful inside form-encoded bodies and is ambiguous in a URL.
    qs = urlencode(pairs, quote_via=quote, safe="")
    return f"{cfg['form_url']}?usp=pp_url&{qs}"
