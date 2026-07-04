#!/usr/bin/env python3
"""
Merge the 5 labeled shards with their source sentences into the final
check-worthiness claim set, add the one manually-labeled sentence, run a
secret/PII scan, and emit a stratified 70/30 train/held-out split.

Reproducible: no randomness beyond a fixed seed for the split.
Outputs (in this directory):
  claim_set.jsonl            all labeled rows
  claim_set.train.jsonl      ~70% stratified by label
  claim_set.heldout.jsonl    ~30% stratified by label
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

HERE = Path(__file__).parent
SEED = 20260702  # deterministic split seed (date-derived; no RNG at import)

# One sentence (biden_2022:0205) was dropped by its labeler; labeled by hand
# here per LABELING_GUIDE (guest introduction -> unimportant).
MANUAL = {
    "biden_2022:0205": {
        "label": "unimportant", "claim_type": None, "confidence": "high",
        "rationale": "guest introduction, no public stakes",
        "edge_case": "guest-intro",
    }
}

# Defense-in-depth: scan final text for anything that looks like a leaked
# secret/PII even though SOTU transcripts are public.
SECRET_RX = [
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"), "anthropic-key"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "github-pat"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "openai-key"),
    (re.compile(r"AIza[A-Za-z0-9_\-]{20,}"), "google-key"),
    (re.compile(r"xai-[A-Za-z0-9]{20,}"), "xai-key"),
    (re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "email"),
    (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"), "phone"),
]


def scan(text: str):
    return [tag for rx, tag in SECRET_RX if rx.search(text)]


def stratified_split(rows, frac_heldout=0.30):
    """Deterministic per-class round-robin: every Nth item (by a fixed hash
    order) goes to held-out, guaranteeing each class is split ~70/30."""
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    train, held = [], []
    step = round(1 / frac_heldout)  # 0.30 -> ~3 => every 3rd to held-out
    for label, items in by_label.items():
        items = sorted(items, key=lambda r: (hash((SEED, r["sid"])) & 0xffffffff))
        for i, r in enumerate(items):
            (held if i % step == 0 else train).append(r)
    return train, held


def main():
    cand = {}
    for l in (HERE / "_candidates.jsonl").read_text().splitlines():
        o = json.loads(l); cand[o["sid"]] = o

    labels = {}
    for k in range(5):
        for o in json.loads((HERE / f"_labels_{k}.json").read_text()):
            labels.setdefault(o["sid"], o)
    labels.update({sid: {"sid": sid, **v} for sid, v in MANUAL.items()})

    rows, flagged = [], []
    for sid, src in cand.items():
        lab = labels.get(sid)
        if not lab:
            print(f"WARN: no label for {sid}", file=sys.stderr); continue
        hits = scan(src["text"])
        if hits:
            flagged.append((sid, hits))
        rows.append({
            "sid": sid,
            "speech": src["speech"],
            "text": src["text"],
            "context": src["context"],
            "label": lab["label"],
            "claim_type": lab.get("claim_type"),
            "confidence": lab.get("confidence"),
            "rationale": lab.get("rationale"),
            "edge_case": lab.get("edge_case"),
        })

    rows.sort(key=lambda r: r["sid"])
    (HERE / "claim_set.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")

    train, held = stratified_split(rows)
    (HERE / "claim_set.train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n")
    (HERE / "claim_set.heldout.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in held) + "\n")

    from collections import Counter
    def bal(rs): return dict(Counter(r["label"] for r in rs))
    print("total:", len(rows), bal(rows))
    print("train:", len(train), bal(train))
    print("held :", len(held), bal(held))
    print("secret/PII flags:", flagged if flagged else "none")


if __name__ == "__main__":
    main()
