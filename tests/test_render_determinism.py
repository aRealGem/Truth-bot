"""A2 (Wave A) acceptance: the site render is byte-reproducible.

Renders the publishing heads twice with a frozen clock (SOURCE_DATE_EPOCH) and
two different PYTHONHASHSEED values, and asserts the two trees are byte-for-byte
identical. Skips when no publishing-head artifacts are present (a fresh checkout
without metrics/pca_runs/), so it never fails for a data-absent environment.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_render_determinism", REPO / "scripts" / "check_render_determinism.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.acceptance
def test_render_is_byte_reproducible():
    from truthbot.publish.heads import publishing_heads
    if not publishing_heads():
        pytest.skip("no publishing-head artifacts under metrics/pca_runs/")
    diffs = _load_checker().check()
    assert diffs == [], "render is not byte-reproducible:\n" + "\n".join(diffs)
