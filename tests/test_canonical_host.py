"""The published site must name ONE canonical host, and it must be the real one.

Every absolute self-link on the site -- canonical, og:url, og:image, feed.xml
entries, and the reader-feedback prefill -- derives from a single base URL. When
the site moved to GitHub Pages on 2026-08-29 and that base was left pointing at
the old githack mirror, the live pages spent a day telling crawlers and social
platforms that their real home was somewhere else. Nothing failed; the site just
quietly disavowed itself.

The two defaults are held in separate modules on purpose (the render layer keeps
its zero-config-import convention), which is exactly how they drift apart. Both
are pinned here.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from truthbot.config import settings
from truthbot.publish.site import _DEFAULT_SITE_URL, _site_url

REPO = Path(__file__).resolve().parents[1]

EXPECTED_HOST = "https://arealgem.github.io/Truth-bot/site-pca"

#: Searched for stale references.
_SCAN_DIRS = ("src", "scripts", "tests", "docs")
_SCAN_FILES = (".env.example", "pyproject.toml", "README.md")

#: DELIBERATELY EXCLUDED, not an oversight. STATUS.md is an append-only session
#: log, PROJECT_BOARD.md records the history of a backlog item, and metrics/
#: holds dated review documents. All three are RECORDS of what was true at the
#: time. Rewriting them to match today would destroy the history that makes them
#: worth keeping -- the same reason already-collected feedback responses keep
#: their githack claim URLs.
#: This file excludes itself for the obvious reason: it is where the
#: forbidden string is defined.
_EXCLUDED = ("STATUS.md", "PROJECT_BOARD.md", "metrics",
             "tests/test_canonical_host.py")

_STALE = "githack"


def test_the_two_defaults_agree():
    """They live in different modules, so nothing but a test keeps them equal."""
    assert _DEFAULT_SITE_URL == EXPECTED_HOST
    assert settings.site_url == EXPECTED_HOST
    assert _site_url() == EXPECTED_HOST


def _candidate_files():
    for d in _SCAN_DIRS:
        for p in (REPO / d).rglob("*"):
            if p.is_file() and p.suffix in {".py", ".md", ".toml", ".yml", ".yaml"}:
                yield p
    for f in _SCAN_FILES:
        p = REPO / f
        if p.exists():
            yield p


def test_no_stale_host_in_code_or_docs():
    """One live reference is allowed: the comment explaining why this matters."""
    hits = []
    for p in _candidate_files():
        rel = p.relative_to(REPO)
        if any(str(rel).startswith(x) for x in _EXCLUDED):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if _STALE in line and "the githack mirror" not in line:
                hits.append(f"{rel}:{i}: {line.strip()[:100]}")
    assert not hits, "stale publication host still referenced:\n" + "\n".join(hits)


@pytest.mark.skipif(not (REPO / "site-pca" / "index.html").exists(),
                    reason="site-pca not rendered")
def test_no_stale_host_in_the_rendered_site():
    """The rendered tree is regenerated, never hand-edited, so a hit here means
    the render was not re-run after the host changed."""
    hits = [str(p.relative_to(REPO))
            for p in (REPO / "site-pca").rglob("*")
            if p.is_file() and p.suffix in {".html", ".xml"}
            and _STALE in p.read_text(encoding="utf-8", errors="replace")]
    assert not hits, (
        f"{len(hits)} rendered file(s) still carry the old host; regenerate "
        f"site-pca/. First few: {hits[:5]}")
