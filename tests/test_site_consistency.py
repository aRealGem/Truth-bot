"""Run the build-time consistency checker over the committed ``site-pca/``
tree so hand-typed or drifted figures cannot merge (T0.8; this is the test the
``consistency.py`` docstring has promised since P67.4 — added in PR-A2.0
together with the distribution-sum invariants)."""
from __future__ import annotations

from pathlib import Path

import pytest

from truthbot.publish.consistency import check_site

_SITE = Path(__file__).resolve().parent.parent / "site-pca"


@pytest.mark.skipif(not (_SITE / "data" / "reports.json").exists(),
                    reason="site-pca tree not present")
def test_committed_site_has_no_consistency_violations() -> None:
    violations = check_site(_SITE)
    assert violations == [], "\n".join(violations)
