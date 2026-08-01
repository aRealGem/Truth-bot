"""R4 — deterministic newspaper-archive retriever (P132 / D12, approved 2026-08-01).

The corpus mission (P131, Nixon 1974 → Obama 2014) starves without
era-contemporaneous press coverage: the LLM retrievers search the live web,
which under-serves pre-web decades. R4 queries the **NYT Article Search API**
— free key, searchable back to 1851, returns headline + abstract + lead
paragraph + publication date — and emits a standard shortlist for the
deterministic consolidator. NYT is S3 (Established), so with stance attached
these items genuinely credit the T2.4 quota.

Two cheap-LLM assists are borrowed from :mod:`truthbot.verify.relevance`
(both fail SOFT and are speaker-blind):
  * ``generate_queries`` — targeted keyword queries instead of the raw claim
    sentence (archive search engines do poorly on rhetoric);
  * ``score_evidence`` — populates ``relevance_score``/``supports_claim`` so
    archival items can bear on the core assertion (a dateline alone cannot).

Source-survey outcome recorded on P132 (verified 2026-08-01): Chronicling
America ends ~1963 (copyright) — no use for this corpus; UPI's robots.txt
allows browsing but Disallows ``/search`` and ``/archive/search``, so UPI is
excluded from automated retrieval (a polite date-window crawl is a possible
v2); ProQuest/TimesMachine/newspapers.com are paid, deferred.

Terms: the NYT API requires attribution ("Data provided by The New York
Times") and non-commercial use — evidence lists link nytimes.com directly;
add the footer attribution line before the first PUBLISHED run that carries
R4 items. Rate limits: 500 requests/day, 5/min → ``_PACE_S`` seconds between
calls and a soft per-process request budget.

Key: ``NYT_API_KEY`` env. Missing → empty shortlists with one warning
(fail-soft, same contract as a dead R1/R2/R3 lane).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Optional

from truthbot.models import Evidence
from truthbot.verify.source_tiers import classify_tier

logger = logging.getLogger(__name__)

_NYT_SEARCH = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
_PACE_S = 12.5           # 5 req/min ceiling → ≥12s spacing
_DAILY_BUDGET = 450      # soft cap under the 500/day limit
_QUERIES_PER_CLAIM = 2
_DOCS_PER_QUERY = 6
_SHORTLIST_CAP = 8

#: Lenient-mode fallback: pre-web packs deliberately pass utterance/window as
#: None to retrievers (their prompts would hard-scope dates), but the era
#: brief in the context carries "speech given on YYYY-MM-DD" — for an archive
#: API a date window is the whole point, so R4 recovers it from there.
_CTX_DATE = re.compile(r"speech given on (\d{4}-\d{2}-\d{2})")


def _era_window(utterance: Optional[date], window, context: str
                ) -> Optional[tuple[date, date]]:
    if window:
        return window
    utt = utterance
    if utt is None:
        m = _CTX_DATE.search(context or "")
        if m:
            utt = date.fromisoformat(m.group(1))
    if utt is None:
        return None
    # Era-contemporaneous reporting: claims reference the recent past; coverage
    # up to the fair-game edge. Mirrors expected_claim_window's spirit without
    # importing sid-keyed context (R4 never sees the sid).
    return (utt - timedelta(days=730), utt + timedelta(days=7))


@dataclass
class NytArchiveRetriever:
    """NYT Article Search shortlist producer (Retriever protocol)."""

    label: str = "R4-nyt-archive"
    api_key: str = ""
    llm: Optional[Callable] = None          # LlmFn; None → build lazily
    # Injectable GET for offline tests: (url) -> parsed JSON dict.
    http_get: Optional[Callable[[str], dict]] = None
    requests_made: int = field(default=0, init=False)
    _last_call: float = field(default=0.0, init=False)
    _warned_no_key: bool = field(default=False, init=False)

    def _key(self) -> str:
        return self.api_key or os.environ.get("NYT_API_KEY", "")

    def _llm(self):
        if self.llm is None:
            from truthbot.verify.relevance import build_proxy_llm
            self.llm = build_proxy_llm() or False   # False = tried, unavailable
        return self.llm or None

    def _get(self, url: str) -> dict:
        if self.http_get is not None:
            return self.http_get(url)
        wait = _PACE_S - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _queries(self, claim_text: str, context: str) -> list[str]:
        llm = self._llm()
        if llm is not None:
            try:
                from truthbot.verify.relevance import generate_queries
                qs = generate_queries(llm, claim_text, context=context)
                if qs:
                    return qs[:_QUERIES_PER_CLAIM]
            except Exception as exc:                      # fail SOFT
                logger.warning("%s: query-gen failed (%s)", self.label, exc)
        # Deterministic fallback: the claim sentence minus quote punctuation.
        return [re.sub(r"[\"“”]", "", claim_text).strip()[:120]]

    def shortlist(self, claim_text: str, *, context: str = "",
                  utterance: Optional[date] = None,
                  window: Optional[tuple[date, date]] = None) -> list[Evidence]:
        if not self._key():
            if not self._warned_no_key:
                logger.warning("%s: NYT_API_KEY not set — lane inactive", self.label)
                self._warned_no_key = True
            return []
        if self.requests_made >= _DAILY_BUDGET:
            logger.warning("%s: daily request budget exhausted", self.label)
            return []
        win = _era_window(utterance, window, context)
        out: list[Evidence] = []
        seen: set[str] = set()
        for q in self._queries(claim_text, context):
            params = {"q": q, "sort": "relevance", "api-key": self._key()}
            if win:
                params["begin_date"] = win[0].strftime("%Y%m%d")
                params["end_date"] = win[1].strftime("%Y%m%d")
            url = _NYT_SEARCH + "?" + urllib.parse.urlencode(params)
            try:
                doc = self._get(url)
                self.requests_made += 1
            except urllib.error.HTTPError as exc:
                logger.warning("%s: HTTP %s for query %r", self.label,
                               getattr(exc, "code", "?"), q)
                if getattr(exc, "code", None) == 429:
                    break                                  # back off entirely
                continue
            except Exception as exc:
                logger.warning("%s: fetch failed (%s)", self.label, exc)
                continue
            for d in (doc.get("response") or {}).get("docs", [])[:_DOCS_PER_QUERY]:
                ev = self._doc_to_evidence(d)
                if ev is None or ev.source_url in seen:
                    continue
                seen.add(ev.source_url)
                out.append(ev)
            if len(out) >= _SHORTLIST_CAP:
                break
        out = out[:_SHORTLIST_CAP]
        llm = self._llm()
        if out and llm is not None:
            try:
                from truthbot.verify.relevance import score_evidence
                score_evidence(llm, claim_text, out)       # stance in place
                out.sort(key=lambda e: -(e.relevance_score or 0.0))
            except Exception as exc:                       # fail SOFT
                logger.warning("%s: stance scoring failed (%s)", self.label, exc)
        return out

    def _doc_to_evidence(self, d: dict) -> Optional[Evidence]:
        url = (d.get("web_url") or "").strip()
        if not url:
            return None
        headline = ((d.get("headline") or {}).get("main") or "").strip()
        body = (d.get("abstract") or d.get("lead_paragraph")
                or d.get("snippet") or "").strip()
        pub_raw = (d.get("pub_date") or "")[:10]
        try:
            pub = (date.fromisoformat(pub_raw) if pub_raw else None)
        except ValueError:
            pub = None
        snippet = " — ".join(s for s in (headline, body) if s)[:400]
        if pub_raw:
            snippet = f"[{pub_raw}] {snippet}"
        from datetime import datetime, timezone
        return Evidence(
            claim_id="", source_name=self.label,
            source_url=url, source_tier=classify_tier(url),
            snippet=snippet, supports_claim=None,
            published_at=(datetime(pub.year, pub.month, pub.day,
                                   tzinfo=timezone.utc) if pub else None),
        )
