#!/usr/bin/env python3
"""
Pre-push secret/PII scanner for the C1 benchmark trees.

Scans given paths (recursively) for anything resembling a leaked credential or
PII. Skips .git, build caches, and the scanner's own source. Exit code 1 if any
hit is found — wire it as a pre-push gate.

Usage: secret_scan.py <path> [<path> ...]
"""
from __future__ import annotations
import re, sys, os
from pathlib import Path

RX = [
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"), "anthropic-key"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "github-pat"),
    (re.compile(r"ghp_[A-Za-z0-9]{30,}"), "github-classic-pat"),
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}"), "openai-key"),
    (re.compile(r"AIza[A-Za-z0-9_\-]{20,}"), "google-key"),
    (re.compile(r"xai-[A-Za-z0-9]{20,}"), "xai-key"),
    (re.compile(r"sk-or-v1-[A-Za-z0-9]{20,}"), "openrouter-key"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private-key"),
    (re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "email"),
    (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"), "phone"),
]
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv"}
SKIP_NAMES = {"secret_scan.py", "build_claim_set.py"}  # contain the patterns themselves


def scan_file(p: Path):
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    hits = []
    for i, line in enumerate(txt.splitlines(), 1):
        for rx, tag in RX:
            for m in rx.findall(line):
                hits.append((i, tag, (m if isinstance(m, str) else m[0])[:16]))
    return hits


def main(argv):
    if not argv:
        print("usage: secret_scan.py <path> ...", file=sys.stderr); return 2
    total = 0
    for root in argv:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if fn in SKIP_NAMES:
                    continue
                p = Path(dirpath) / fn
                for ln, tag, frag in scan_file(p):
                    print(f"HIT {p}:{ln} [{tag}] {frag}...")
                    total += 1
    print(f"scanned: {argv}")
    print(f"HITS: {total if total else 'none'}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
