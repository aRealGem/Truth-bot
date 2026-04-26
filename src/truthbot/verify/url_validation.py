"""
URL reachability check for model-cited sources (Phase 3b).

Model ``web_search`` outputs occasionally include hallucinated, expired,
or paywall-wrapped URLs. This module HEAD-checks each URL in
``ModelVerdict.web_sources`` so the publish layer can surface
unreachable citations to readers instead of rendering them as trusted.

Design goals
------------
1. **Predictable latency.** Synchronous concurrency via a small thread
   pool; one HTTP client; explicit timeouts. No asyncio dance.
2. **Cacheable.** Results persist to ``metrics/url_cache.jsonl`` with a
   TTL. Reruns of the publish pipeline are nearly free for URLs seen
   recently.
3. **Graceful degradation.** Any failure — DNS, connect, TLS, timeout,
   redirect loop, 4xx/5xx — marks the URL unreachable but never raises
   out of the checker. A caller that cannot reach the outside network
   (e.g. pytest in a sandbox) must be able to short-circuit via
   ``enabled=False``.
4. **Test-safe.** No network I/O at import time. All HTTP goes through a
   single ``_request`` seam that ``respx`` can patch.
5. **Never silently strip.** The checker returns ``UrlCheckResult`` per
   URL; consumers decide how to render (strike-through, badge, drop,
   etc.). Stripping without an audit trail would hide real signal about
   which models hallucinate.

Public API
----------
* ``UrlCheckResult``                     — dataclass returned for each URL
* ``check_url(url, ...) -> UrlCheckResult``
* ``check_urls_bulk(urls, ...) -> dict[str, UrlCheckResult]``
* ``UrlCache.load(path)`` / ``UrlCache.save(path)``
* ``annotate_verdicts(verdicts, cache, ...)`` — attach reachability info
  to each ``ModelVerdict`` without mutating ``web_sources``.
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

from truthbot.models import ModelVerdict

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 5.0
_DEFAULT_MAX_WORKERS = 8
_DEFAULT_TTL_DAYS = 14
_DEFAULT_USER_AGENT = (
    "TruthBot/0.1 (+https://github.com/jackiem/Truth-bot) URLCheck/1.0"
)

# HEAD can be rejected by servers that happily return 200 on GET —
# notably many .gov, press sites, and CDN-fronted articles. When HEAD
# comes back in this status set we retry with GET (range-limited) to
# avoid false-negative reachability flags.
_HEAD_RETRY_WITH_GET_STATUSES = frozenset({403, 405, 501})


# ── Result dataclass ────────────────────────────────────────────────────


# Status / failure-mode classifier
# --------------------------------
# Empirically, many "unreachable" results from ``check_url`` come back as
# 403 from ``.gov``, ``.mil``, or major newsrooms that aggressively block
# automated HEAD/GET. These URLs are almost always real — the bot
# just can't see them without a browser User-Agent. Publish-layer
# rendering must distinguish that case from a genuine 404 / DNS failure
# / malformed URL, so readers aren't told a real NYT article is dead
# because we couldn't bypass a WAF.

_TRUSTED_BOT_BLOCK_DOMAINS = (
    ".gov", ".mil",
    "apnews.com", "reuters.com", "bbc.", "npr.org",
    "nytimes.com", "washingtonpost.com", "wsj.com", "bloomberg.com",
    "cbsnews.com", "nbcnews.com", "abcnews.go.com", "cnn.com",
    "axios.com", "politico.com", "foxnews.com",
)


def classify_failure(result: "UrlCheckResult") -> str:
    """Bucket a non-reachable result for publish-layer rendering.

    Returns one of:
      * ``"ok"``           — reachable (``status`` is 2xx/3xx).
      * ``"bot-blocked"``  — 401/403 from a trusted domain; URL is very
        likely real, the checker was blocked by a WAF or rate limiter.
        Publish should render these as trusted even though
        ``reachable`` is False.
      * ``"malformed"``    — invalid scheme / broken URL.
      * ``"dead-4xx"``     — 404/410 etc.; path does not exist.
      * ``"cert-error"``   — TLS cert failure.
      * ``"dns"``          — DNS resolution failed.
      * ``"transient"``    — timeout / 5xx / connection reset.
      * ``"unknown"``      — anything else.
    """
    if result.reachable:
        return "ok"
    status = result.status
    url_lower = (result.url or "").lower()
    trusted = any(d in url_lower for d in _TRUSTED_BOT_BLOCK_DOMAINS)
    if status in (401, 403) and trusted:
        return "bot-blocked"
    err = (result.error or "").lower()
    if err.startswith("invalid-scheme"):
        return "malformed"
    if status in (404, 410):
        return "dead-4xx"
    if "certificate" in err or "ssl" in err:
        return "cert-error"
    if "nodename nor servname" in err or "name or service not known" in err:
        return "dns"
    if "timeout" in err or (status is not None and 500 <= status < 600):
        return "transient"
    return "unknown"


@dataclass(frozen=True)
class UrlCheckResult:
    """Outcome of a single URL reachability check."""

    url: str
    reachable: bool
    status: Optional[int] = None
    """HTTP status code that decided the outcome (``None`` on pre-HTTP failure)."""

    error: Optional[str] = None
    """Short human-readable error (``None`` on success)."""

    method_used: str = "HEAD"
    """``"HEAD"`` or ``"GET"``; GET is only used when HEAD was rejected."""

    checked_at: str = ""
    """ISO-8601 timestamp of the check. Populated by callers / cache."""

    final_url: Optional[str] = None
    """Final URL after following redirects (differs from ``url`` only on 3xx)."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def failure_class(self) -> str:
        """``classify_failure(self)`` convenience accessor."""
        return classify_failure(self)

    @property
    def likely_real(self) -> bool:
        """True when the URL is either reachable or bot-blocked from a
        trusted domain. Publish layer uses this to decide whether to
        display a "dead link" warning or render the URL as-is with a
        quiet ``bot-blocked`` annotation."""
        return self.reachable or self.failure_class == "bot-blocked"


# ── Cache ───────────────────────────────────────────────────────────────


@dataclass
class UrlCache:
    """Simple TTL-aware URL reachability cache persisted as JSONL."""

    entries: dict[str, UrlCheckResult] = field(default_factory=dict)
    ttl_days: int = _DEFAULT_TTL_DAYS

    def get(self, url: str, *, now: Optional[datetime] = None) -> Optional[UrlCheckResult]:
        """Return a cached result if present AND not expired."""
        rec = self.entries.get(url)
        if rec is None:
            return None
        if not rec.checked_at:
            return rec  # cache migration path; treat as fresh-enough
        try:
            checked = datetime.fromisoformat(rec.checked_at)
        except ValueError:
            return None
        cutoff = (now or datetime.utcnow()) - timedelta(days=self.ttl_days)
        if checked < cutoff:
            return None
        return rec

    def put(self, result: UrlCheckResult) -> None:
        self.entries[result.url] = result

    # ── I/O ──────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Path, *, ttl_days: int = _DEFAULT_TTL_DAYS) -> "UrlCache":
        cache = cls(ttl_days=ttl_days)
        if not path.exists():
            return cache
        try:
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("UrlCache: skipping malformed row in %s", path)
                        continue
                    url = row.get("url")
                    if not url:
                        continue
                    cache.entries[url] = UrlCheckResult(
                        url=url,
                        reachable=bool(row.get("reachable", False)),
                        status=row.get("status"),
                        error=row.get("error"),
                        method_used=row.get("method_used", "HEAD"),
                        checked_at=row.get("checked_at", ""),
                        final_url=row.get("final_url"),
                    )
        except OSError:
            logger.warning("UrlCache: failed to read %s", path, exc_info=True)
        return cache

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as f:
            for url in sorted(self.entries):
                f.write(json.dumps(self.entries[url].to_dict()) + "\n")
        tmp.replace(path)


# ── HTTP seam (single point of interception for tests) ──────────────────


def _request(
    url: str,
    *,
    method: str,
    timeout: float,
    user_agent: str,
) -> tuple[int, Optional[str]]:
    """Issue a single HTTP request. Returns ``(status_code, final_url)``.

    Raises on any pre-HTTP failure (connection, timeout, DNS, TLS). Callers
    translate exceptions into ``UrlCheckResult.reachable=False``.

    Isolated so tests can ``respx.mock`` this entire module's network I/O.
    """
    import httpx

    headers = {
        "User-Agent": user_agent,
        "Accept": "*/*",
    }
    with httpx.Client(
        follow_redirects=True,
        timeout=timeout,
        headers=headers,
    ) as client:
        resp = client.request(method, url)
        final = str(resp.url) if str(resp.url) != url else None
        return resp.status_code, final


# ── Single-URL check ────────────────────────────────────────────────────


def check_url(
    url: str,
    *,
    timeout: float = _DEFAULT_TIMEOUT_S,
    user_agent: str = _DEFAULT_USER_AGENT,
    _now: Optional[datetime] = None,
) -> UrlCheckResult:
    """HEAD (and GET fallback) a URL; return a ``UrlCheckResult``.

    Never raises. Caller can rely on the ``reachable`` flag unconditionally.
    """
    checked_at = (_now or datetime.utcnow()).isoformat()
    if not url or not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return UrlCheckResult(
            url=url or "",
            reachable=False,
            error="invalid-scheme",
            checked_at=checked_at,
        )

    try:
        status, final = _request(url, method="HEAD", timeout=timeout, user_agent=user_agent)
    except Exception as exc:
        return UrlCheckResult(
            url=url,
            reachable=False,
            error=f"head:{type(exc).__name__}:{exc}",
            method_used="HEAD",
            checked_at=checked_at,
        )

    if status in _HEAD_RETRY_WITH_GET_STATUSES:
        try:
            status, final = _request(
                url, method="GET", timeout=timeout, user_agent=user_agent
            )
            return UrlCheckResult(
                url=url,
                reachable=200 <= status < 400,
                status=status,
                error=None if 200 <= status < 400 else f"http-{status}",
                method_used="GET",
                checked_at=checked_at,
                final_url=final,
            )
        except Exception as exc:
            return UrlCheckResult(
                url=url,
                reachable=False,
                error=f"get:{type(exc).__name__}:{exc}",
                method_used="GET",
                checked_at=checked_at,
            )

    return UrlCheckResult(
        url=url,
        reachable=200 <= status < 400,
        status=status,
        error=None if 200 <= status < 400 else f"http-{status}",
        method_used="HEAD",
        checked_at=checked_at,
        final_url=final,
    )


# ── Bulk check ──────────────────────────────────────────────────────────


def check_urls_bulk(
    urls: Iterable[str],
    *,
    timeout: float = _DEFAULT_TIMEOUT_S,
    max_workers: int = _DEFAULT_MAX_WORKERS,
    user_agent: str = _DEFAULT_USER_AGENT,
    cache: Optional[UrlCache] = None,
    on_result: Optional[Any] = None,
) -> dict[str, UrlCheckResult]:
    """Check many URLs with bounded concurrency; optionally update ``cache``.

    Deduplicates the input; preserves no particular order in the return.
    ``on_result`` is a callable ``(UrlCheckResult) -> None`` invoked as
    each result completes (useful for progress bars).
    """
    unique = sorted({u for u in urls if isinstance(u, str) and u})
    out: dict[str, UrlCheckResult] = {}

    # Serve cache hits without spawning workers.
    pending: list[str] = []
    for url in unique:
        cached = cache.get(url) if cache else None
        if cached is not None:
            out[url] = cached
            if on_result is not None:
                on_result(cached)
        else:
            pending.append(url)

    if not pending:
        return out

    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                check_url, url, timeout=timeout, user_agent=user_agent
            ): url
            for url in pending
        }
        for fut in as_completed(futures):
            url = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                result = UrlCheckResult(
                    url=url,
                    reachable=False,
                    error=f"pool:{type(exc).__name__}:{exc}",
                    checked_at=datetime.utcnow().isoformat(),
                )
            with lock:
                out[url] = result
                if cache is not None:
                    cache.put(result)
                if on_result is not None:
                    on_result(result)
    return out


# ── Verdict-level helper ────────────────────────────────────────────────


@dataclass
class VerdictUrlAudit:
    """Summary of reachability audit attached to a ``ModelVerdict``.

    Stored outside the Pydantic model so existing serializers don't care
    about Phase 3b. Publish-layer rendering consults this dict via the
    ``verdict.claim_id``/``adapter_name`` key. If the renderer is updated,
    it can wire directly.
    """

    checked: list[str] = field(default_factory=list)
    reachable: list[str] = field(default_factory=list)
    unreachable: list[str] = field(default_factory=list)
    results: dict[str, UrlCheckResult] = field(default_factory=dict)


def annotate_verdicts(
    verdicts: Iterable[ModelVerdict],
    *,
    cache: Optional[UrlCache] = None,
    timeout: float = _DEFAULT_TIMEOUT_S,
    max_workers: int = _DEFAULT_MAX_WORKERS,
    user_agent: str = _DEFAULT_USER_AGENT,
) -> dict[tuple[str, str], VerdictUrlAudit]:
    """Run ``check_urls_bulk`` across every URL in ``verdicts.web_sources``.

    Returns a mapping keyed by ``(claim_id, adapter_name)`` so the publish
    layer can look up reachability per-verdict. Verdicts are never
    mutated — the audit is a separate sidecar.
    """
    verdicts_list = list(verdicts)
    all_urls: set[str] = set()
    for mv in verdicts_list:
        for url in mv.web_sources or []:
            if isinstance(url, str) and url:
                all_urls.add(url)

    results = check_urls_bulk(
        all_urls,
        timeout=timeout,
        max_workers=max_workers,
        user_agent=user_agent,
        cache=cache,
    )

    audit: dict[tuple[str, str], VerdictUrlAudit] = {}
    for mv in verdicts_list:
        a = VerdictUrlAudit()
        for url in mv.web_sources or []:
            if not isinstance(url, str) or not url:
                continue
            a.checked.append(url)
            res = results.get(url)
            if res is None:
                continue
            a.results[url] = res
            if res.reachable:
                a.reachable.append(url)
            else:
                a.unreachable.append(url)
        audit[(mv.claim_id, mv.adapter_name)] = a
    return audit


# ── Sidecar filtering (Layer 3 — anti-hallucination) ───────────────────


# Categories that we treat as "the URL exists, just not reachable from
# this checker"; publish should render them with an "unverified" badge,
# not strip them.
_KEEP_AS_UNVERIFIED = frozenset({"bot-blocked", "transient"})
# Categories where the URL almost certainly does not point at a real
# resource. Publish should drop these from the rendered citation list.
_STRIP_AS_BROKEN = frozenset({"dead-4xx", "malformed", "dns", "cert-error"})


def filter_sidecar_row(
    row: dict[str, Any],
    *,
    results: dict[str, "UrlCheckResult"],
) -> dict[str, Any]:
    """Apply Layer 3 anti-hallucination filtering to a single sidecar row.

    Returns a new row dict (the input is not mutated) with three new
    fields populated based on the URL reachability classifications in
    ``results``:

      * ``verified_sources``   — URLs classified ``ok``.
      * ``unverified_sources`` — URLs classified ``bot-blocked`` or
        ``transient``. Likely real, but the checker could not confirm;
        the publish layer should render with a muted/badged style.
      * ``broken_sources``     — URLs classified ``dead-4xx``,
        ``malformed``, ``dns``, or ``cert-error``. Almost certainly
        fabricated or rotted; publish should not render these.

    The row's ``web_sources`` is rewritten to ``verified + unverified``
    so existing publish-layer code keeps working — broken URLs are
    silently removed from there. ``model_reported_sources`` is left
    untouched for the audit trail.
    """
    out = dict(row)
    sources = [u for u in (row.get("web_sources") or []) if isinstance(u, str) and u]

    verified: list[str] = []
    unverified: list[str] = []
    broken: list[str] = []
    classification: dict[str, str] = {}

    for url in sources:
        res = results.get(url)
        if res is None:
            # Unknown — treat as unverified rather than silently strip.
            unverified.append(url)
            classification[url] = "unknown"
            continue
        cls = res.failure_class
        classification[url] = cls
        if cls == "ok":
            verified.append(url)
        elif cls in _KEEP_AS_UNVERIFIED:
            unverified.append(url)
        elif cls in _STRIP_AS_BROKEN:
            broken.append(url)
        else:
            unverified.append(url)

    out["verified_sources"] = verified
    out["unverified_sources"] = unverified
    out["broken_sources"] = broken
    out["web_sources"] = verified + unverified
    out["url_filter_classification"] = classification
    return out


def filter_sidecar(
    in_path: Path,
    out_path: Path,
    *,
    cache: "UrlCache | None" = None,
    timeout: float = _DEFAULT_TIMEOUT_S,
    max_workers: int = _DEFAULT_MAX_WORKERS,
) -> dict[str, int]:
    """Apply ``filter_sidecar_row`` to every row of a sidecar file.

    HEAD-checks each unique URL once (using ``cache`` when available)
    and rewrites the sidecar to ``out_path``. Returns a stats dict
    summarising the counts written across all rows.
    """
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    rows: list[dict[str, Any]] = []
    urls: set[str] = set()
    with in_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("filter_sidecar: skipping malformed row")
                continue
            rows.append(row)
            for u in row.get("web_sources") or []:
                if isinstance(u, str) and u:
                    urls.add(u)

    results = check_urls_bulk(
        urls, timeout=timeout, max_workers=max_workers, cache=cache
    )

    stats = {"rows": len(rows), "verified": 0, "unverified": 0, "broken": 0}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            cleaned = filter_sidecar_row(row, results=results)
            stats["verified"] += len(cleaned["verified_sources"])
            stats["unverified"] += len(cleaned["unverified_sources"])
            stats["broken"] += len(cleaned["broken_sources"])
            f.write(json.dumps(cleaned) + "\n")
    tmp.replace(out_path)
    return stats


def classify_verdicts_in_place(
    verdicts: Iterable[ModelVerdict],
    *,
    cache: Optional[UrlCache] = None,
    timeout: float = _DEFAULT_TIMEOUT_S,
    max_workers: int = _DEFAULT_MAX_WORKERS,
    user_agent: str = _DEFAULT_USER_AGENT,
) -> dict[str, int]:
    """HEAD-check every URL across ``verdicts`` and write the result onto
    each verdict's ``url_classifications`` dict (Layer 4 backbone).

    Unlike ``filter_sidecar`` which writes a separate JSONL file, this
    helper mutates the verdicts in place so they can flow straight to
    the publish layer. Reuses ``check_urls_bulk`` (and therefore the
    URL cache) so a previously-checked URL is free.

    Returns a stats dict ``{verified, unverified, broken}`` summed
    across all verdicts.
    """
    verdicts_list = list(verdicts)
    all_urls: set[str] = set()
    for mv in verdicts_list:
        for url in mv.web_sources or []:
            if isinstance(url, str) and url:
                all_urls.add(url)

    if not all_urls:
        return {"verified": 0, "unverified": 0, "broken": 0}

    results = check_urls_bulk(
        all_urls,
        timeout=timeout,
        max_workers=max_workers,
        user_agent=user_agent,
        cache=cache,
    )

    stats = {"verified": 0, "unverified": 0, "broken": 0}
    for mv in verdicts_list:
        cls_map = dict(mv.url_classifications or {})
        for url in mv.web_sources or []:
            if not isinstance(url, str) or not url:
                continue
            res = results.get(url)
            if res is None:
                continue
            cls = res.failure_class
            cls_map[url] = cls
            if cls == "ok":
                stats["verified"] += 1
            elif cls in _STRIP_AS_BROKEN:
                stats["broken"] += 1
            else:
                stats["unverified"] += 1
        mv.url_classifications = cls_map
    return stats


__all__ = [
    "UrlCheckResult",
    "UrlCache",
    "VerdictUrlAudit",
    "check_url",
    "check_urls_bulk",
    "annotate_verdicts",
    "classify_verdicts_in_place",
    "filter_sidecar",
    "filter_sidecar_row",
]
