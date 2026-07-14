#!/usr/bin/env python3
"""
Gold-side helper for sotu_gold_fixture_2026-07-10.json (Tasks 3 & 4).

READ-ONLY over the corpus. Does two things and prints a report:

  (3) Span resolution — for each fixture claim, locate the sentence in the in-repo
      corpus (_sentences.jsonl, keyed by sid) whose text contains ALL anchor_keywords
      (case-insensitive, unordered). Record the sid + char span (first→last anchor)
      into the sibling offsets file. A claim is resolved in its DECLARED speech; if
      not found there we probe the other speech and flag it (catches misattribution).

  (4) Attribution audit — for all 277 claim_set.jsonl records, confirm the `speech`
      field matches the sid prefix and the record text matches the _sentences.jsonl
      entry for that sid. Reports match/miss. Also reports where trump2026-05
      (Dominican officers) actually lives.

Nothing here writes to the corpus or the 277-set; it only writes the offsets sibling.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
FIXTURE = HERE / "sotu_gold_fixture_2026-07-10.json"
OFFSETS = HERE / "sotu_gold_fixture_2026-07-10.offsets.json"
SENTENCES = HERE / "_sentences.jsonl"
CLAIM_SET = HERE / "claim_set.jsonl"

_SPEECH = {"biden2022": "biden_2022", "trump2026": "trump_2026"}   # fixture -> sid prefix


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _anchors_in(text: str, anchors: list[str]) -> dict | None:
    """All anchors present (case-insensitive substring)? Return per-anchor offsets."""
    low = text.lower()
    hits = {}
    for a in anchors:
        i = low.find(a.lower())
        if i < 0:
            return None
        hits[a] = [i, i + len(a)]
    return hits


def resolve_spans(fixture: dict, sents: list[dict]) -> tuple[dict, list[str]]:
    by_speech: dict[str, list[dict]] = {}
    for s in sents:
        by_speech.setdefault(s["speech"], []).append(s)
    offsets, misses = {}, []
    for c in fixture["claims"]:
        cid, anchors = c["claim_id"], c["anchor_keywords"]
        declared = _SPEECH[c["speech"]]
        speech_order = (declared, *[p for p in by_speech if p != declared])
        found = None
        # Global preference: an exact SENTENCE-level match anywhere beats a
        # PARAGRAPH-level (context-window) match, so we don't resolve onto a neighbour.
        for field in ("text", "context"):
            for pref in speech_order:
                for s in by_speech.get(pref, []):
                    hits = _anchors_in(s[field], anchors)
                    if hits:
                        starts = [v[0] for v in hits.values()]
                        ends = [v[1] for v in hits.values()]
                        found = {"claim_id": cid, "resolved_sid": s["sid"],
                                 "resolved_speech": pref, "declared_speech": declared,
                                 "in_declared_speech": pref == declared,
                                 "resolution_level": "sentence" if field == "text" else "paragraph",
                                 "resolved_field": field,
                                 "char_start": min(starts), "char_end": max(ends),
                                 "matched_text": s[field], "anchor_offsets": hits}
                        break
                if found:
                    break
            if found:
                break
        if found:
            offsets[cid] = found
            if not found["in_declared_speech"]:
                misses.append(f"{cid}: anchors NOT in declared {declared}; resolved in "
                              f"{found['resolved_speech']} at {found['resolved_sid']} (MISATTRIBUTION)")
        else:
            offsets[cid] = {"claim_id": cid, "resolved_sid": None,
                            "declared_speech": declared, "anchors": anchors}
            misses.append(f"{cid}: anchors did not resolve in ANY speech")
    return offsets, misses


def attribution_audit(claim_set: list[dict], sents: list[dict]) -> dict:
    sent_by_sid = {s["sid"]: s for s in sents}
    n = speech_mismatch = text_mismatch = missing_sid = 0
    anomalies = []
    for r in claim_set:
        n += 1
        sid = r["sid"]
        pref = sid.split(":")[0]
        if r.get("speech") != pref:
            speech_mismatch += 1
            anomalies.append(f"{sid}: speech={r.get('speech')} != sid prefix {pref}")
        s = sent_by_sid.get(sid)
        if s is None:
            missing_sid += 1
            anomalies.append(f"{sid}: no matching sentence in _sentences.jsonl")
        elif s["text"].strip() != r["text"].strip():
            text_mismatch += 1
            anomalies.append(f"{sid}: text differs from corpus sentence")
    return {"n": n, "speech_mismatch": speech_mismatch, "text_mismatch": text_mismatch,
            "missing_sid": missing_sid,
            "match": n - speech_mismatch - text_mismatch - missing_sid,
            "anomalies": anomalies}


def main() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    sents = _load_jsonl(SENTENCES)
    claim_set = _load_jsonl(CLAIM_SET)

    offsets, misses = resolve_spans(fixture, sents)
    OFFSETS.write_text(json.dumps(offsets, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    resolved = sum(1 for v in offsets.values() if v.get("resolved_sid"))
    in_declared = sum(1 for v in offsets.values() if v.get("in_declared_speech"))
    print("## Task 3 — span resolution")
    print(f"  resolved: {resolved}/{len(fixture['claims'])}  "
          f"(in declared speech: {in_declared}/{len(fixture['claims'])})")
    for m in misses:
        print(f"  FLAG {m}")
    print(f"  -> offsets written: {OFFSETS.name}")

    audit = attribution_audit(claim_set, sents)
    print("\n## Task 4 — attribution audit (277-set, read-only)")
    print(f"  match {audit['match']}/{audit['n']}  | speech_mismatch={audit['speech_mismatch']} "
          f"text_mismatch={audit['text_mismatch']} missing_sid={audit['missing_sid']}")
    for a in audit["anomalies"][:20]:
        print(f"  ANOMALY {a}")
    dom = [r["sid"] for r in claim_set
           if "Dominican" in r["text"] or ("patrol" in r["text"].lower() and "officer" in r["text"].lower())]
    print(f"  trump2026-05 (Dominican officers) lives in 277-set at: {dom} "
          f"— {'correctly under biden_2022 (no repo fix needed)' if all(s.startswith('biden_2022') for s in dom) else 'MISATTRIBUTED in repo'}")


if __name__ == "__main__":
    main()
