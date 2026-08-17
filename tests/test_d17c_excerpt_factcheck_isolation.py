"""D17-c Stage 0, item 8 — the excerpt path cannot carry a fact-checker verdict.

Series-row excerpting (D17-c) puts fetched rows into the scoring payload. That
is a new door into the model's context, and P67.7 spent a PR closing the old
one: the fact-checker exclusion blocklist exists so that a verdict someone else
already reached cannot be laundered into ours as evidence.

The claim under test is that the new door reuses the old lock rather than
routing around it. It rests on two gates, and this file asserts both plus the
seams between them:

* **Gate 1 — the excerpt source allowlist.** Stage A may only excerpt a URL
  that ``classify_ex`` admits. That allowlist fails closed, so a source has to
  be vouched for by name; no fact-checker is.
* **Gate 2 — the exclusion blocklist.** Independently, ``is_excluded_factchecker``
  rejects fact-check *paths* on any host at all, so even an allowlisted agency
  that published a fact-check page could not contribute one.

Neither gate is trusted alone: the point of the pair is that admitting a
fact-checker verdict would take a simultaneous failure of both. Every test here
is offline — registry data and shipped artifacts only, no network, no model.
"""
from __future__ import annotations

import json

import pytest

from truthbot.verify.factcheck_exclusion import (
    _blocklist,
    blocked_domains,
    factcheck_exclusion_reason,
    is_excluded_factchecker,
)
from truthbot.verify.statistical_agency import classify_ex, load_registry


def _agency_domains() -> tuple[str, ...]:
    return tuple(sorted(load_registry().entries_by_domain))


# ── Gate 1: no fact-checker can be an excerpt source ────────────────────────

@pytest.mark.parametrize("domain", blocked_domains())
def test_blocked_factcheckers_are_not_excerptable_sources(domain: str) -> None:
    """Direction A. Every blocklisted fact-checker fails the excerpt
    allowlist, so Stage A would never fetch it in the first place."""
    allowed, reason = classify_ex(f"https://{domain}/some/article")
    assert not allowed, f"{domain} is excerptable: {reason}"


@pytest.mark.parametrize(
    "url",
    [
        "https://www.reuters.com/fact-check/some-claim",
        "https://apnews.com/hub/ap-fact-check/some-claim",
        "https://www.washingtonpost.com/politics/fact-checker/some-claim",
        "https://www.cbsnews.com/news/fact-check-some-claim/",
        "https://www.nbcnews.com/politics/fact-check-some-claim",
    ],
)
def test_factcheck_verticals_are_not_excerptable_sources(url: str) -> None:
    """The vertical and path-prefix rules cover outlets whose *domain* is
    perfectly legitimate. Those URLs must still fail the excerpt allowlist —
    a general-news host is not a statistical agency either."""
    assert is_excluded_factchecker(url), f"{url} escaped the blocklist"
    assert not classify_ex(url)[0], f"{url} is excerptable"


# ── Gate 2: the two registries do not overlap ───────────────────────────────

def test_no_statistical_agency_is_a_blocked_factchecker() -> None:
    """Direction B. The allowlist and the blocklist are disjoint at the domain
    level. If they ever overlapped, one of the two registries would be wrong
    about the same host and the resolution order would decide by accident."""
    overlap = set(_agency_domains()).intersection(blocked_domains())
    assert not overlap, f"a host is on both registries: {sorted(overlap)}"


@pytest.mark.parametrize("domain", _agency_domains())
def test_agency_domains_are_not_swept_up_by_the_blocklist(domain: str) -> None:
    """The converse failure mode: a blocklist rule so broad it suppresses the
    very series Stage A exists to fetch. A bare agency host must be clean."""
    assert not is_excluded_factchecker(f"https://{domain}/"), domain


def test_the_blocklist_allowlist_carve_out_is_not_an_excerpt_source() -> None:
    """The blocklist has exactly one allowlist entry — a state-government page
    whose path merely contains 'factcheck'. It is the one place where a
    ``fact-check`` path is deliberately NOT excluded, so it is the natural
    place for a seam. There is none: the host is not an excerpt source."""
    carve_outs = _blocklist().allowlist
    assert carve_outs, "expected at least one allowlist carve-out to pin"
    agency = set(_agency_domains())
    for domain, prefix in carve_outs:
        url = f"https://{domain}{prefix}/anything"
        assert factcheck_exclusion_reason(url) == "", f"{url} unexpectedly blocked"
        assert domain not in agency, f"{domain} is both carve-out and agency"
        assert not classify_ex(url)[0], f"{url} is excerptable"


# ── Defense in depth: both gates hold independently ─────────────────────────

@pytest.mark.parametrize("domain", _agency_domains())
def test_a_factcheck_path_on_an_agency_host_is_still_excluded(domain: str) -> None:
    """The hypothetical that motivates keeping gate 2: an allowlisted agency
    publishes a page at a ``/fact-check/`` path. Gate 1 admits the host, so
    only the generic path regex stands between that page and the payload. It
    holds — and it holds for every agency, not just the ones we thought of."""
    for path in ("/fact-check/claim", "/factcheck/claim", "/news/fact_check/x"):
        url = f"https://{domain}{path}"
        assert is_excluded_factchecker(url), url


def test_the_two_gates_are_independent_not_one_rule_twice() -> None:
    """A guard on the guards. If some refactor collapsed the two registries
    into one, every test above would still pass while the redundancy this file
    documents had quietly gone. Each gate must reject something the other
    admits."""
    only_gate_1 = "https://www.politifact.com/factchecks/2026/jan/01/x/"
    only_gate_2 = "https://www.bls.gov/fact-check/jobs"
    assert is_excluded_factchecker(only_gate_1) and not classify_ex(only_gate_1)[0]
    assert is_excluded_factchecker(only_gate_2) and classify_ex(only_gate_2)[0]


# ── The corpus, not just the registries ─────────────────────────────────────

def test_no_shipped_evidence_url_is_both_excerptable_and_a_factchecker() -> None:
    """The empirical form of the invariant, over every evidence URL in the
    five shipped publishing heads. The registry tests prove the rule; this
    proves the rule was actually true of the corpus D17-c will operate on."""
    heads = pytest.importorskip("truthbot.publish.heads")
    paths = heads.publishing_heads()
    if not paths:
        pytest.skip("no publishing heads available in this tree")

    seen = both = excerptable = 0
    for path in paths.values():
        with open(path) as fh:
            doc = json.load(fh)
        for pack in (doc.get("evidence") or {}).values():
            for item in pack or []:
                url = item.get("source_url") or ""
                if not url:
                    continue
                seen += 1
                allowed = classify_ex(url)[0]
                excerptable += allowed
                if allowed and is_excluded_factchecker(url):
                    both += 1

    assert seen, "expected evidence URLs in the shipped heads"
    assert excerptable, "expected some excerptable statistical-agency items"
    assert both == 0, f"{both} of {seen} URLs are excerptable AND fact-checkers"
