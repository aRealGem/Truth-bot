#!/usr/bin/env python3
"""
G1 (Layer A) evaluation harness.

Composed A1+A2 recall_cw / macro-F1 on the HELDOUT split is the real gate. That
requires A2 live (a proxy virtual key from the repo .env) and is read ONCE per
release candidate (I6). Until A2 is live this harness runs the A1-only analysis
on the TRAIN split (183) to calibrate the prefilter thresholds — train is for
tuning; heldout stays sealed behind HeldoutGuard.

Usage:
  run_g1.py a1-train                 # tune A1 on train (no LLM, no heldout)
  run_g1.py composed --rc <id>       # A1+A2 on heldout (BLOCKED without proxy key)
"""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from truthbot.checkworthy import prefilter  # noqa: E402
from hydramind.invariants import HeldoutGuard, I6HeldoutReuseError  # noqa: E402

HERE = Path(__file__).parent
TRAIN = HERE / "claim-set" / "claim_set.train.jsonl"
HELDOUT = HERE / "claim-set" / "claim_set.heldout.jsonl"


def load(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def a1_train():
    rows = load(TRAIN)
    scored = [(r, prefilter.score(r["text"])) for r in rows]
    n_cw = sum(1 for r, _ in scored if r["label"] == "check-worthy")
    n = len(scored)
    print(f"# A1-only calibration on TRAIN ({n} sentences, {n_cw} check-worthy)")
    print(f"# check-worthy = positive; opinion+unimportant = negative\n")

    # binary detector sweep
    print("tau   recall_cw  prec_cw   F1_cw   pred_pos")
    best = None
    for i in range(30, 81, 5):
        tau = i / 100
        tp = sum(1 for r, s in scored if s >= tau and r["label"] == "check-worthy")
        fp = sum(1 for r, s in scored if s >= tau and r["label"] != "check-worthy")
        fn = sum(1 for r, s in scored if s < tau and r["label"] == "check-worthy")
        rec = tp / (tp + fn) if (tp + fn) else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        print(f"{tau:.2f}   {rec:6.3f}    {prec:6.3f}   {f1:5.3f}   {tp+fp}")
        if rec >= 0.90 and (best is None or f1 > best[3]):
            best = (tau, rec, prec, f1)

    # routing view: pick tau_low (drop floor) preserving recall, tau_high (auto-pass)
    print("\n# routing analysis: DROP (s<tau_low) / AMBIGUOUS→A2 / PASS (s>=tau_high)")
    print("tau_low  cw_recall_preserved  budget_dropped")
    tau_low_pick = 0.30
    for i in range(30, 61, 5):
        tl = i / 100
        cw_dropped = sum(1 for r, s in scored if s < tl and r["label"] == "check-worthy")
        dropped = sum(1 for _, s in scored if s < tl)
        preserved = 1 - (cw_dropped / n_cw if n_cw else 0)
        print(f"{tl:.2f}     {preserved:6.3f}              {dropped/n:6.3f}")
        if preserved >= 0.95:
            tau_low_pick = tl

    tau_high_pick = best[0] if best else 0.70
    amb = sum(1 for _, s in scored if tau_low_pick <= s < tau_high_pick)
    print(f"\n# recommended operating point (train): tau_low={tau_low_pick:.2f} "
          f"tau_high={tau_high_pick:.2f}")
    print(f"#   A2 load (ambiguous band) = {amb}/{n} = {amb/n:.2f} of sentences")
    if best:
        print(f"#   best binary tau={best[0]:.2f} recall_cw={best[1]:.3f} "
              f"prec_cw={best[2]:.3f} F1={best[3]:.3f}")
    print("#   HELDOUT NOT READ (sealed for the A1+A2 composed RC run — I6).")


_LABELS = ["check-worthy", "opinion", "unimportant"]


def _metrics(pairs):
    """pairs: list[(gold, pred)] -> (accuracy, per-class PRF, macro_f1, confusion)."""
    conf = {g: {p: 0 for p in _LABELS} for g in _LABELS}
    for g, p in pairs:
        conf[g][p] = conf[g].get(p, 0) + 1
    n = len(pairs)
    correct = sum(conf[l][l] for l in _LABELS)
    prf = {}
    f1s = []
    for l in _LABELS:
        tp = conf[l][l]
        fp = sum(conf[g][l] for g in _LABELS) - tp
        fn = sum(conf[l][p] for p in _LABELS) - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        prf[l] = (prec, rec, f1)
        f1s.append(f1)
    return correct / n, prf, sum(f1s) / len(f1s), conf


def composed(rc_id, tau_high=0.65):
    import os
    from hydramind import HydraMind
    from hydramind.transport import Transport, ProxyCompletion
    from truthbot.checkworthy import classifier, prefilter

    guard = HeldoutGuard()
    try:
        guard.read("claim_set.heldout", rc_id)      # I6 — once per RC
    except I6HeldoutReuseError as e:
        print(f"REFUSED: {e}"); return
    if not os.environ.get("LITELLM_KEY"):
        print("BLOCKED: LITELLM_KEY not in env; source the repo .env first."); return

    rows = load(HELDOUT)
    sents = [{"sid": r["sid"], "text": r["text"], "context": r.get("context", "")} for r in rows]
    gold = {r["sid"]: r["label"] for r in rows}

    hm = HydraMind.from_specs_dir(transport=Transport(
        completion_fn=ProxyCompletion(response_parser=classifier.parse_a2)))
    a2, manifest = classifier.classify(hm, sents)          # A2 on all (full 3-way labels)
    a2_label = {v["sid"]: v["label"] for v in a2}

    # composed = A1 PASS forces check-worthy (budget path); else A2's label.
    a1_pass = {s["sid"]: (prefilter.score(s["text"]) >= tau_high) for s in sents}
    composed_pred = {sid: ("check-worthy" if a1_pass[sid] else a2_label[sid]) for sid in gold}

    mism = manifest.model_mismatches()
    print(f"# G1 composed A1+A2 on HELDOUT (RC={rc_id}, n={len(gold)}) — read once (I6)")
    if mism:
        print(f"!! MODEL FALLBACK DETECTED (fails G5/equivalence): {mism[:5]}")
    else:
        print("# model-fallback guard: PASS (all calls returned the requested family)")

    for name, pred in [("A2-only", a2_label), ("composed(A1+A2)", composed_pred)]:
        pairs = [(gold[s], pred[s]) for s in gold]
        acc, prf, macro, conf = _metrics(pairs)
        print(f"\n## {name}: acc={acc:.3f}  macro-F1={macro:.3f}  "
              f"recall_cw={prf['check-worthy'][1]:.3f}  prec_cw={prf['check-worthy'][0]:.3f}")
        print("   confusion (gold→pred):")
        print("             " + "  ".join(f"{l[:5]:>7}" for l in _LABELS))
        for g in _LABELS:
            print(f"   {g:12} " + "  ".join(f"{conf[g][p]:7d}" for p in _LABELS))

    print(f"\n# provisional bars: recall_cw ≥ 0.90, macro-F1 ≥ 0.75 (record, don't chase)")
    print(f"# lane tally: {manifest.lane_tally}  tokens_in={manifest.total_tokens_in} "
          f"tokens_out={manifest.total_tokens_out}")
    Path(HERE / "examples" / "manifest.heldout.json").write_text(manifest.to_json())
    print(f"# manifest → examples/manifest.heldout.json")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "a1-train"
    if cmd == "a1-train":
        a1_train()
    elif cmd == "composed":
        rc = sys.argv[sys.argv.index("--rc") + 1] if "--rc" in sys.argv else "rc0"
        composed(rc)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
