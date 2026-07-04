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


def composed(rc_id):
    guard = HeldoutGuard()
    try:
        guard.read("claim_set.heldout", rc_id)
    except I6HeldoutReuseError as e:
        print(f"REFUSED: {e}"); return
    print("BLOCKED: composed A1+A2 heldout run needs A2 live (proxy virtual key "
          "from the repo .env / CW-12). Heldout intentionally not opened here.\n"
          "When the key is present:\n"
          "  from truthbot.checkworthy import classifier, pipeline\n"
          "  hm = HydraMind.load(); res = pipeline.run_layer_a(heldout, \n"
          "       classify_fn=lambda ss: classifier.classify(hm, ss)[0], full_speech=True)\n"
          "  then score with scorer/score.py claims.")


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
