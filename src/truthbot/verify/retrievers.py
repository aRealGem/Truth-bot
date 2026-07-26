"""Evidence-v2 retrievers — R1 / R2 / R3 shortlist producers (P67.8 / T2.5).

Design note: wiki ``projects:truthbot:evidence-v2-design``. Each retriever
searches the open web its own way and emits an ordered SHORTLIST of
candidate evidence for one claim; the union of shortlists feeds the
deterministic consolidator (:mod:`truthbot.verdict.consolidator`), which
does all selection — a retriever ranks only its own list.

* **R1** — Claude Opus through **Lane-Worker**: the ``claude`` CLI in
  headless mode on the Max-subscription login (zero marginal cost).
  ``ANTHROPIC_API_KEY`` is STRIPPED from the subprocess environment so the
  CLI cannot silently fall back to API billing (CW-12 scoped the key into
  the repo ``.env``; the worker must not inherit it).
* **R2** — GPT with native browsing via the OpenAI Responses API
  ``web_search`` tool (the lane ``verify.adapters.openai`` already uses).
  Model defaults to ``gpt-5.5`` per the roster plan, falling back down
  ``_R2_FALLBACKS`` when the account lacks the primary.
* **R3** — pending decision D1 (Grok live search vs an independent search
  API via Lane-Tools); :class:`PendingDecisionError` until decided.

Retrieval-side T2.1/T2.2 enforcement lives here: prompts are built ONLY
from the claim text/context and era (speaker-blind by construction, and the
contamination guard in the pilot harness asserts no gold-label text ever
enters a prompt); fact-checker URLs are dropped on the way out with a log
line (the consolidator filters again).
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable, Optional, Protocol, Sequence

from truthbot.models import Evidence, SourceTier
from truthbot.verdict.era_lint import FAIR_GAME_DAYS, fair_game_end
from truthbot.verify.factcheck_exclusion import is_excluded_factchecker
from truthbot.verify.sources.brave import classify_tier

logger = logging.getLogger(__name__)

SHORTLIST_N = 8

# P120 PR-2: rate-limit / subscription-quota detection for the adaptive pool.
# Retrievers stay fail-soft (return []); when an ``on_rate_limit`` callback is wired
# (by the pool governor) they ALSO signal it so the pool can pare/drop the lane. The
# claude CLI (R1) has no structured error, so we sniff its stderr/stdout for the
# usage-limit phrasings; R2/R3 raise urllib HTTPError 429.
_RATE_LIMIT_RE = re.compile(
    r"usage limit|rate[ _-]?limit|too many requests|quota|overloaded|\b429\b", re.I)


def _looks_rate_limited(text: str) -> bool:
    return bool(text and _RATE_LIMIT_RE.search(text))


def _is_http_429(exc: Exception) -> bool:
    return isinstance(exc, urllib.error.HTTPError) and getattr(exc, "code", None) == 429


class PendingDecisionError(RuntimeError):
    """The retriever seat is blocked on an open roster decision."""


class ContaminationError(AssertionError):
    """Gold-label text reached a retriever prompt (T2.6 hard guard)."""


class Retriever(Protocol):
    label: str

    def shortlist(self, claim_text: str, *, context: str = "",
                  utterance: Optional[date] = None,
                  window: Optional[tuple[date, date]] = None) -> list[Evidence]:
        ...


def build_retrieval_prompt(claim_text: str, *, context: str = "",
                           utterance: Optional[date] = None,
                           window: Optional[tuple[date, date]] = None,
                           n: int = SHORTLIST_N) -> str:
    """Shared retrieval prompt. Speaker-blind by construction: only the claim
    text, optional context, and era dates go in — never a speaker name field
    and never any gold material (the pilot harness asserts this)."""
    era = ""
    if window:
        era = f"Evidence must be published between {window[0]} and {window[1]}. "
    if utterance:
        era += (f"The claim was made on {utterance}; STRONGLY prefer sources "
                f"published on or before {fair_game_end(utterance)} "
                f"(utterance + {FAIR_GAME_DAYS} days — later items will be "
                f"discarded).")
    ctx = f"\nContext (surrounding transcript, non-evidentiary): {context[:400]}" if context else ""
    return (
        "You are an evidence retriever for a fact-checking pipeline. "
        "Search the web and return the BEST candidate sources for judging "
        "this claim as of the date it was made.\n\n"
        f"CLAIM: {claim_text}{ctx}\n\n"
        f"{era}\n\n"
        "Rules:\n"
        "- Prefer primary/official sources (government statistics, agency "
        "documents), then wire services, then established outlets.\n"
        "- NEVER return fact-checking organizations' pages (PolitiFact, "
        "FactCheck.org, Snopes, FullFact, AFP/Reuters/AP/WaPo fact-check "
        "sections) — this pipeline must reach its own verdict.\n"
        "- Return article/data pages, not homepages or search/listing pages.\n"
        f"- Up to {n} items, YOUR best first.\n\n"
        "Respond with STRICT JSON only:\n"
        '{"items": [{"url": "...", "date": "YYYY-MM-DD or null", '
        '"stance": "supports|refutes|context", '
        '"one_line_why": "<= 25 words"}]}'
    )


def _parse_shortlist_json(text: str) -> list[dict]:
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return []
    try:
        return list(json.loads(m.group(0)).get("items") or [])
    except (json.JSONDecodeError, AttributeError):
        return []


def items_to_evidence(items: Sequence[dict], *, retriever_label: str) -> list[Evidence]:
    """Model-emitted shortlist items → Evidence, with retrieval-side T2.1
    enforcement (fact-checker URLs dropped here AND at consolidation)."""
    out: list[Evidence] = []
    for it in items:
        url = str(it.get("url") or "").strip()
        if not url.startswith("http"):
            continue
        if is_excluded_factchecker(url):
            logger.info("%s: dropped fact-checker URL at retrieval (T2.1): %s",
                        retriever_label, url)
            continue
        stance = str(it.get("stance") or "").strip().lower()
        supports = {"supports": True, "refutes": False}.get(stance)
        pub = None
        raw_date = str(it.get("date") or "").strip()
        if raw_date and raw_date.lower() != "null":
            try:
                pub = datetime.fromisoformat(raw_date[:10]).replace(tzinfo=timezone.utc)
            except ValueError:
                pub = None
        why = str(it.get("one_line_why") or "").strip()[:300]
        snippet = f"[{raw_date[:10]}] {why}" if pub else why
        out.append(Evidence(
            claim_id="", source_name=retriever_label,
            source_url=url, source_tier=classify_tier(url),
            snippet=snippet, supports_claim=supports,
            published_at=pub,
        ))
    return out


# ── R1: Claude Opus via Lane-Worker (claude CLI, subscription auth) ──────────


@dataclass
class ClaudeWorkerRetriever:
    label: str = "R1-opus-worker"
    model: str = "opus"
    timeout_s: int = 420
    # P120 PR-2: called (if set) when the worker output looks like a Max
    # usage/rate-limit hit, so the pool governor can drop R1 for a cool-down.
    on_rate_limit: Optional[Callable[[], None]] = None

    def shortlist(self, claim_text: str, *, context: str = "",
                  utterance: Optional[date] = None,
                  window: Optional[tuple[date, date]] = None) -> list[Evidence]:
        prompt = build_retrieval_prompt(claim_text, context=context,
                                        utterance=utterance, window=window)
        env = dict(os.environ)
        # Lane-Worker runs on the Max-subscription login — never API billing.
        env.pop("ANTHROPIC_API_KEY", None)
        try:
            proc = subprocess.run(
                ["claude", "-p", prompt, "--output-format", "json",
                 "--model", self.model, "--allowedTools", "WebSearch"],
                capture_output=True, text=True, timeout=self.timeout_s, env=env)
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("%s: worker invocation failed (%s)", self.label, exc)
            return []
        if proc.returncode != 0:
            logger.warning("%s: worker exit %d: %s", self.label,
                           proc.returncode, proc.stderr[-300:])
            if self.on_rate_limit and _looks_rate_limited(
                    (proc.stderr or "") + (proc.stdout or "")):
                logger.warning("%s: worker output looks like a Max usage/rate "
                               "limit — signaling pool backoff", self.label)
                self.on_rate_limit()
            return []
        try:
            envelope = json.loads(proc.stdout)
            text = envelope.get("result") or ""
        except json.JSONDecodeError:
            text = proc.stdout
        return items_to_evidence(_parse_shortlist_json(text),
                                 retriever_label=self.label)


# ── R2: GPT with native browsing (Responses API web_search) ──────────────────

_R2_FALLBACKS = ("gpt-5.4", "gpt-4o")


@dataclass
class OpenAIBrowsingRetriever:
    label: str = "R2-gpt-browsing"
    model: str = ""
    timeout_s: int = 300
    on_rate_limit: Optional[Callable[[], None]] = None   # P120 PR-2: 429 → pool pare

    def _models(self) -> list[str]:
        primary = (self.model or os.environ.get("TRUTHBOT_R2_MODEL") or "gpt-5.5")
        chain = [primary] + [m for m in _R2_FALLBACKS if m != primary]
        return chain

    def _post(self, model: str, prompt: str) -> dict:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY not set (R2 lane)")
        body = json.dumps({
            "model": model,
            "input": prompt,
            "tools": [{"type": "web_search"}],
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/responses", data=body,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _output_text(doc: dict) -> str:
        parts: list[str] = []
        for item in doc.get("output") or []:
            for c in item.get("content") or []:
                if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                    parts.append(str(c.get("text") or ""))
        return "\n".join(parts) or str(doc.get("output_text") or "")

    def shortlist(self, claim_text: str, *, context: str = "",
                  utterance: Optional[date] = None,
                  window: Optional[tuple[date, date]] = None) -> list[Evidence]:
        prompt = build_retrieval_prompt(claim_text, context=context,
                                        utterance=utterance, window=window)
        last_err: Exception | None = None
        for model in self._models():
            # An EMPTY parse from a successful call is a soft failure, not an
            # answer (P67.9 mini pilot: gpt-5-mini came back empty on 2/15
            # claims where gpt-5.5 never did — refusal/format flakes, since
            # even unsourceable claims normally yield *something*). Retry the
            # same model once (pennies at mini rates), then fall down the
            # chain like a POST failure would.
            for attempt in (1, 2):
                try:
                    doc = self._post(model, prompt)
                except Exception as exc:  # noqa: BLE001 — fall down the chain
                    logger.warning("%s: model %s failed (%s)", self.label, model, exc)
                    last_err = exc
                    if self.on_rate_limit and _is_http_429(exc):
                        self.on_rate_limit()
                    break                      # POST failure → next model
                usage = doc.get("usage") or {}
                logger.info("%s: model=%s tokens in/out %s/%s", self.label, model,
                            usage.get("input_tokens"), usage.get("output_tokens"))
                items = items_to_evidence(
                    _parse_shortlist_json(self._output_text(doc)),
                    retriever_label=self.label)
                if items:
                    return items
                logger.warning("%s: model %s returned an empty shortlist "
                               "(attempt %d/2)", self.label, model, attempt)
        logger.warning("%s: all models failed or empty (%s)", self.label, last_err)
        return []


# ── R3: pending decision D1 ──────────────────────────────────────────────────


@dataclass
class GrokSearchRetriever:
    """R3 — Grok Live Search via the xAI API (decision D1, resolved
    2026-07-22: jackie's roster ruling put Grok in the stack; the on-file
    recommendation was Grok, and ``XAI_API_KEY`` ships in ``~/.env``).

    Era discipline is enforced twice: the Live Search ``from_date``/
    ``to_date`` parameters bound what Grok may search (to_date = fair-game
    end), and the shared conversion/consolidation filters re-check every
    returned item like any other retriever."""
    label: str = "R3-grok-search"
    model: str = ""
    timeout_s: int = 300
    on_rate_limit: Optional[Callable[[], None]] = None   # P120 PR-2: 429 → pool pare

    def _post(self, model: str, prompt: str, tool: dict) -> dict:
        key = os.environ.get("XAI_API_KEY")
        if not key:
            raise EnvironmentError("XAI_API_KEY not set (R3 lane)")
        # xAI Agent Tools API (the 2026 replacement for the deprecated
        # search_parameters Live Search — the old field now 410s). The
        # /v1/responses envelope mirrors OpenAI's, so R2's output parsing
        # is reused verbatim.
        body = json.dumps({
            "model": model,
            "input": prompt,
            "tools": [tool],
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.x.ai/v1/responses", data=body,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def shortlist(self, claim_text: str, *, context: str = "",
                  utterance: Optional[date] = None,
                  window: Optional[tuple[date, date]] = None) -> list[Evidence]:
        prompt = build_retrieval_prompt(claim_text, context=context,
                                        utterance=utterance, window=window)
        tool: dict = {"type": "web_search"}
        if window:
            tool["from_date"] = window[0].isoformat()
        if utterance:
            tool["to_date"] = fair_game_end(utterance).isoformat()
        elif window:
            tool["to_date"] = window[1].isoformat()
        model = self.model or os.environ.get("TRUTHBOT_R3_MODEL") or "grok-4"
        try:
            doc = self._post(model, prompt, tool)
        except Exception as exc:  # noqa: BLE001 — fail soft like the other seats
            logger.warning("%s: model %s failed (%s)", self.label, model, exc)
            if self.on_rate_limit and _is_http_429(exc):
                self.on_rate_limit()
            return []
        usage = doc.get("usage") or {}
        logger.info("%s: model=%s tokens in/out %s/%s", self.label,
                    doc.get("model", model), usage.get("input_tokens"),
                    usage.get("output_tokens"))
        return items_to_evidence(
            _parse_shortlist_json(OpenAIBrowsingRetriever._output_text(doc)),
            retriever_label=self.label)


# ── T2.6 contamination guard (harness assertion, not a convention) ───────────


def assert_no_contamination(prompt: str, gold_fragments: Sequence[str]) -> None:
    """Hard assertion that no gold-label material entered a retriever prompt.
    ``gold_fragments`` are verdicts/rationales from the fixture; any
    (case-insensitive) occurrence in ``prompt`` is a ContaminationError."""
    p = (prompt or "").lower()
    for frag in gold_fragments:
        f = (frag or "").strip().lower()
        if len(f) >= 12 and f in p:
            raise ContaminationError(
                f"gold fragment leaked into retriever prompt: {frag[:60]!r}")
