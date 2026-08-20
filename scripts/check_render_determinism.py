#!/usr/bin/env python3
"""A2 (Wave A) determinism check: render the publishing heads twice with a
frozen clock and assert the two site trees are byte-identical.

The render is made reproducible by two changes on this branch:
  * ``report_id`` is content-derived (was ``uuid4()`` -> rotated every render);
  * every render timestamp honours ``SOURCE_DATE_EPOCH`` (reproducible-builds
    convention) via ``truthbot.publish.site._reproducible_now``.

With ``SOURCE_DATE_EPOCH`` set, two independent renders of the same inputs are
byte-for-byte identical. This script proves it -- $0, offline, no model calls.

    python scripts/check_render_determinism.py      # exit 0 if identical, 1 if not

Also importable: ``check() -> list[str]`` returns the differing paths ([] == ok).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# A fixed instant so both renders stamp the same time. The exact value is
# irrelevant; it only has to be identical across the two runs.
_EPOCH = "1600000000"


def _head_paths() -> list[str]:
    from truthbot.publish.heads import publishing_heads
    return [str(p) for _sid, p in sorted(publishing_heads().items())]


def _render(dest: Path, heads: list[str], hashseed: str) -> None:
    env = dict(os.environ, SOURCE_DATE_EPOCH=_EPOCH, PYTHONHASHSEED=hashseed)
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "rerender_pca_site.py"),
         *heads, "--site-root", str(dest)],
        check=True, cwd=str(REPO), env=env, stdout=subprocess.DEVNULL,
    )


def _tree(root: Path) -> set[Path]:
    return {p.relative_to(root) for p in root.rglob("*") if p.is_file()}


def check() -> list[str]:
    """Render twice (with two different hash seeds, to catch set-order leaks)
    and return the list of differing paths. Empty list == byte-identical."""
    heads = _head_paths()
    if not heads:
        raise SystemExit("no publishing heads found under metrics/pca_runs/")
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "a", Path(d) / "b"
        _render(a, heads, hashseed="0")
        _render(b, heads, hashseed="1")  # different seed on purpose
        fa, fb = _tree(a), _tree(b)
        diffs = [f"only in A: {p}" for p in sorted(fa - fb)]
        diffs += [f"only in B: {p}" for p in sorted(fb - fa)]
        for rel in sorted(fa & fb):
            if (a / rel).read_bytes() != (b / rel).read_bytes():
                diffs.append(f"differs: {rel}")
        return diffs


def main() -> int:
    diffs = check()
    if diffs:
        print(f"RENDER NOT REPRODUCIBLE — {len(diffs)} difference(s):")
        for d in diffs:
            print(f"  · {d}")
        return 1
    print("render is byte-reproducible: two renders (differing hash seeds) are identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
