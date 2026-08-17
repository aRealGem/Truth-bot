#!/usr/bin/env python3
"""D17-d triage — what would it take to decide the 128 withheld claims? ($0)

Every gate-withheld claim across the five speeches, classified by WHAT WOULD
DECIDE IT. The gate withheld a verdict because the retrieved pack never met the
Tier-1..3 bearing quota; that is a statement about our retrieval, and this asks
what retrieval would close it.

Four classes:
  series-core     the checkable core resolves against a statistical series we
                  can name — the D17-c mechanism already works on these
  web-tier1       needs Tier-1..3 web retrieval; no series will settle it
  compound-split  a checkable core is buried inside a compound utterance and
                  needs segmentation before anything can retrieve for it
  substantive     genuinely undecidable — a private moment, an unmeasured
                  population, an attribution of motive or intent

DESK WORK, NOT RETRIEVAL. Classifications come from the claim text and general
knowledge of the sources; nothing is fetched and no model is called. Where the
class is genuinely arguable the ``why`` says so — this is input to an owner
decision about scope, not a verdict about a claim.

A NOTE ON THE BIGGEST GROUP. Most of the withheld claims are the human stories
around a State of the Union: guests in the gallery, valor citations, victims.
Many are documentable (a Purple Heart is in a record; a crash was reported) and
sit in web-tier1. Others are private by nature — what a doctor said in a room,
what someone thought they were feeling — and no amount of retrieval reaches
them. Separating those two is most of the value here: the first group is a
backlog, the second is honest permanent abstention, and today the page says the
same thing about both.

Usage (repo root):
  PYTHONPATH=src .venv/bin/python scripts/build_d17d_triage.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "metrics" / "pca_runs"
OUT = REPO / "metrics" / "remediation_v2" / "d17d_triage.json"

HEADS = {
    "trump_2026": "799e71b6-2480-50ca-870e-1a95f0d0d5fe",
    "biden_2022": "c156d8f9-be85-5263-92a1-c08949afdedd",
    "obama_2014": "70748500-315a-5664-8474-c6632de57816",
    "clinton_1998": "d7ee7340-c07d-55da-b9db-9397d7141c35",
    "gwbush_2006": "6df77093-e328-596e-bfd5-afabd08a1679",
}

GATE = "insufficient-qualifying-evidence"

#: Measured on the d17c-wave2 escape run — the ONLY lane with a real number.
SERIES_USD_PER_CLAIM = 0.1089

# ── the classification ──────────────────────────────────────────────────────
# (class, series ids or [], why)

S, W, C, U = "series-core", "web-tier1", "compound-split", "substantive"

T = {
 "0035": (S, ["MORTGAGE30US"], "30-year fixed mortgage rate; 'lowest in four years' is a direct min-over-window on the series, the same shape D17-c already decides."),
 "0040": (S, ["SP500", "DJIA", "NASDAQCOM"], "Count of record closes since a dated start — arithmetic over an index series. Which index is unstated, so the count depends on picking one; say so when scoping."),
 "0043": (S, ["ROWFDNQ027S", "BOGZ1FL063164003Q"], "'New investment in the United States' most plausibly maps to foreign direct investment inflows. The mapping is ARGUABLE — the claim does not say which measure — and a wrong series would answer a different question."),
 "0057": (C, [], "Two claims in one sentence: a regulation count (Federal Register / Unified Agenda, web) and 'lifted 2.4 million Americans' out of something unnamed (a series, once the something is identified). Neither is retrievable until they are separated."),
 "0090": (W, [], "Whether the 2028 Olympic torch is lit in America — an IOC host-city decision. Documented, and no series touches it."),
 "0098": (W, [], "WWII service at 17 in the Pacific: service records and unit histories are documentary."),
 "0099": (W, [], "Participation in the Battle of Manila — unit records."),
 "0100": (W, [], "Wounded at Luzon — casualty and service records."),
 "0102": (W, [], "Purple Heart and Bronze Star are formal awards with citations on record."),
 "0106": (U, [], "'I was there.' Unfalsifiable as stated and carries no checkable content on its own."),
 "0110": (U, [], "What she thought she was feeling. A private mental state; no retrieval reaches it."),
 "0111": (W, [], "A named Coast Guard rescue swimmer performing a documented rescue — USCG records and contemporaneous reporting."),
 "0130": (C, [], "The urging is trivially true; the embedded superlative ('largest tax cuts in American history') is the checkable core and needs CBO/Treasury scoring against prior cuts. Segment first."),
 "0137": (U, [], "A named individual's felt financial position, plus attribution to a policy. Neither half is retrievable."),
 "0153": (U, [], "A private conversation with Michael Dell. Nothing public could settle what was said. NOTE: this claim currently renders as gate-withheld because the structured code cannot tell it apart from an under-retrieved claim — it is the clearest case for the substantive class existing."),
 "0161": (U, [], "A conditional projection about future account balances, not an assertion of present fact."),
 "0255": (U, [], "A couple's private fertility history."),
 "0256": (W, [], "A named drug's price is documentable (list price, GoodRx, CMS), though what THIS individual paid may not be. Arguable between web-tier1 and substantive; scope the drug price, not the person."),
 "0279": (U, [], "One buyer's bidding history across 20 homes. No public record aggregates this."),
 "0291": (S, ["SP500", "WILL5000IND"], "'Typical 401(k) balance up $30,000' is index-driven, but the authoritative figure is survey data (EBRI/Fidelity), not FRED. Series can bound it; only the survey can confirm it. Flagged as arguable."),
 "0325": (W, [], "A specific 2024 crash involving a named child — police, NTSB and contemporaneous reporting."),
 "0326": (W, [], "Immigration status and CDL issuance for a named driver — court filings and state DMV records."),
 "0327": (U, [], "What doctors said about a prognosis, in private."),
 "0328": (U, [], "Same private prognosis."),
 "0329": (U, [], "A child's current developmental state — no public record."),
 "0334": (U, [], "'Many, if not most' of an unmeasured population. English proficiency of undocumented immigrants is not measured in a way that could confirm or refute this quantifier."),
 "0340": (W, [], "A named 2023 homicide victim — contemporaneous reporting."),
 "0341": (W, [], "The circumstances of the death as reported — police and press."),
 "0342": (W, [], "The killer's prior arrest and immigration status — court records."),
 "0343": (C, [], "Compound: 'her mother is in the gallery' (present, trivially checkable) plus 'why we are deporting' with an embedded record-removals claim that IS checkable against ICE removal statistics. Segment, then the removals core becomes series-or-agency retrievable. Cross-refs the logged utterance-segmentation item."),
 "0379": (W, [], "A polling figure with a stated number — public polls, if one matching the description exists."),
 "0400": (W, [], "A named minor and a Virginia school district's actions — court filings and reporting."),
 "0402": (W, [], "A runaway incident tied to the same documented case."),
 "0403": (W, [], "A judge's custody decision — court record."),
 "0404": (W, [], "Placement in a state facility — court and agency records."),
 "0405": (W, [], "A full-ride scholarship is documentable via the institution or reporting."),
 "0450": (U, [], "The expressions on people's faces. Not a checkable proposition."),
 "0469": (W, [], "A named service member's death in a documented incident."),
 "0482": (U, [], "What doctors thought and what a mother said, in a hospital room."),
 "0487": (U, [], "A private conversation the speaker had that night."),
 "0514": (U, [], "'Thirty-five million people said' — an unmeasured mass attribution, wrapped around a counterfactual about what would have happened. Neither half is decidable."),
 "0569": (W, [], "Whether a warning was delivered after a named operation — reporting and official statements."),
 "0638": (U, [], "A family's closeness. Not a checkable proposition."),
 "0643": (W, [], "A named person's release from detention and presence at the address — documentable."),
 "0652": (W, [], "Valor deeds of a named service member — the citation is the record."),
 "0659": (W, [], "Slover: wounds sustained under fire, described in the valor citation."),
 "0660": (W, [], "Same citation — number and effect of the wounds."),
 "0664": (W, [], "Aircraft manoeuvre under fire — citation and after-action record."),
 "0665": (W, [], "The landing and its location — same citation."),
 "0666": (U, [], "That the mission's success and the crew's lives HINGED on one man is a causal attribution, not an event. The citation can evidence what he did, not that everything turned on it. Same conflation family as ruling (d)."),
 "0667": (U, [], "What everybody in the helicopter knew. A group's inner awareness."),
 "0683": (W, [], "A 1952 Korean-war dogfight involving a named pilot — service records."),
 "0685": (W, [], "First aerial combat, outnumbered, led the engagement — citation and records."),
 "0686": (W, [], "Whether the story was classified for 50 years — declassification records."),
}

B = {
 "0017": (W, [], "Civilians blocking tanks in Ukraine — extensively documented in wire reporting."),
 "0019": (W, [], "A direct quotation from Zelenskyy's address to the European Parliament — the transcript settles it."),
 "0045": (U, [], "'Isolated from the world more than ever' is an evaluative comparison with no defined measure."),
 "0051": (W, [], "Whether DOJ stood up such a task force — the department's own announcements."),
 "0100": (U, [], "The speaker's childhood memory of his father leaving for work."),
 "0124": (U, [], "'Used to have the best' infrastructure is evaluative. Rankings exist (WEF) but the claim names no measure or date; arguable against web-tier1 if a measure is stipulated."),
 "0137": (W, [], "A stated programme scope — 65,000 miles and 1,500 bridges — checkable against DOT/FHWA announcements."),
 "0146": (W, [], "A specific site near Columbus and its acreage — local reporting and Intel's own filings."),
 "0154": (C, [], "Compound: a private remark by Intel's CEO (not retrievable) wrapped around an investment figure that IS (Intel announcements, SEC filings). Segment and keep the figure."),
 "0168": (W, [], "GM's $7bn investment and job figure, and whether it is the largest in its history — company statements and filings."),
 "0171": (W, [], "A direct quotation from a named senator."),
 "0194": (U, [], "'Top business leaders and most Americans support my plan' — an unmeasured quantifier over two undefined populations."),
 "0200": (W, [], "Insulin manufacturing cost per vial — published cost-of-production studies."),
 "0211": (W, [], "ACA premium savings under the American Rescue Plan — HHS/CMS analyses. Borderline series-core if a specific premium series is stipulated."),
 "0284": (W, [], "Pfizer's monthly antiviral output — company and HHS statements."),
 "0285": (W, [], "Whether the Test-to-Treat initiative launched as described — HHS."),
 "0362": (W, [], "Endorsements from named organisations — their own statements."),
 "0366": (W, [], "Joint patrols with Mexico and Guatemala — DHS/State announcements."),
 "0373": (U, [], "'Supported by everyone from X to Y' is rhetorical breadth, not a countable claim."),
 "0376": (W, [], "The constitutional holding of Roe — the opinion itself."),
 "0420": (W, [], "Beau Biden's rank and service — public record."),
 "0427": (W, [], "Service as a combat medic in Kosovo and Iraq — service records. NOTE: the surrounding passage refers to more than one person; confirm the referent before retrieving."),
 "0431": (U, [], "What he loved doing with his daughter."),
 "0437": (W, [], "VA programmes linking toxic exposure to benefits decisions — VA publications."),
}

O = {
 "0001": (W, [], "A composite narrative opening. The embedded 'more than eight million' jobs figure is series-checkable, but the sentence as constructed is scene-setting; arguable as compound-split."),
 "0004": (U, [], "A rural doctor writing one prescription. No public record."),
 "0013": (W, [], "Business-leader survey rankings on investment destination — the surveys are published (e.g. A.T. Kearney FDI Confidence Index)."),
 "0045": (W, [], "Joining Forces hiring/training totals — the initiative's own reporting."),
 "0055": (W, [], "'Companies say they intend to hire' — hiring-intention surveys (NFIB, Duke CFO)."),
 "0070": (W, [], "SBA lending volume by administration — SBA data; borderline series-core if an SBA series is stipulated."),
 "0082": (W, [], "Announced natural-gas-driven factory investment — industry and company announcements."),
 "0114": (W, [], "One named firm's headcount — company statements and local reporting."),
 "0121": (W, [], "The count of people whose unemployment insurance lapsed — DOL figures. Borderline series-core (ICSA/CCSA measure a different thing)."),
 "0123": (U, [], "A named individual's family circumstances."),
 "0125": (U, [], "That she put herself through college."),
 "0126": (U, [], "That she had never collected unemployment benefits — a negative about one person's history."),
 "0153": (W, [], "How many states raised pre-K funding — NIEER's annual yearbook tracks exactly this."),
 "0158": (W, [], "Whether named companies committed via the FCC initiative — FCC and company announcements."),
 "0189": (S, ["CPIAUCSL", "FEDMINNFRWG"], "Real value of the federal minimum wage against a 1980s baseline: nominal minimum wage deflated by CPI. NOTE: D17-c already attached CPIAUCSL here and the window did NOT reach the Reagan-era baseline — this is the recorded window_period_mismatch. Deciding it needs the named-anchor work, not just the series."),
 "0198": (W, [], "'Helps about half of all parents at some point' — programme participation studies; the population and period are unstated, which is what makes it hard."),
 "0207": (W, [], "MyRA's guarantee structure — Treasury's own programme documentation."),
 "0255": (W, [], "Troop levels in Iraq and Afghanistan at a dated point — DoD deployment figures. Borderline series-core; DoD publishes counts, not a FRED series."),
 "0283": (W, [], "Whether Iran had begun eliminating its higher-enriched stockpile — IAEA verification reporting."),
}

C_ = {
 "0027": (W, [], "Whether six presidents warned about deficits — the State of the Union corpus itself is the record."),
 "0029": (S, ["FYFSD"], "First balanced budget in 30 years — the federal surplus/deficit series decides it directly. Same series D17-c already used on biden_2022:0245."),
 "0077": (W, [], "Class-size reduction from a teacher-hiring programme — NCES and programme reporting."),
 "0090": (W, [], "A count of 240 trade agreements — USTR records."),
 "0101": (W, [], "Dislocated-worker training funding doubling since 1993 — DOL budget appropriations."),
 "0107": (W, [], "'Hundreds of new trade agreements' — same USTR record, weaker quantifier."),
 "0132": (C, [], "'We have also met that goal, two years ahead of schedule' — the goal is named in a previous sentence, so this cannot be retrieved for until the utterance is segmented and the referent bound."),
 "0134": (U, [], "A named individual's 13-year welfare history."),
 "0135": (U, [], "Her current job."),
 "0195": (W, [], "Whether it was the largest antidrug budget — ONDCP budget summaries."),
 "0210": (U, [], "'NATO contained communism and kept America and Europe secure' — a historical evaluation, not a checkable event."),
 "0211": (W, [], "Whether the three named countries adopted democratic government — documented."),
 "0225": (W, [], "A named soldier present as a guest — reporting and service records."),
 "0226": (W, [], "His father's Vietnam decorations — award records."),
 "0227": (W, [], "College in Colorado then Army service — records."),
 "0236": (W, [], "Whether four named former Joint Chiefs chairmen endorsed as described — their statements."),
 "0240": (U, [], "How Saddam Hussein 'spent the better part of a decade' — an evaluative characterisation of intent and priorities."),
 "0241": (W, [], "UNSCOM inspection and destruction findings — the commission's own reports."),
 "0243": (U, [], "'I speak for everyone in this chamber' — a claim about others' agreement."),
 "0350": (U, [], "'Only a handful of physicists used the web' at a past date. Directionally famous but no measurement exists for the period; arguable against web-tier1 via CERN/NSF histories."),
 "0358": (W, [], "ISS participation by 16 countries beginning in 1998 — NASA and partner-agency records."),
}

G = {
 "0024": (W, [], "Count of democracies in 1945 — Freedom House and Polity datasets, though the count depends on the definition used."),
 "0025": (W, [], "Count of democracies today — same datasets, same definitional caveat."),
 "0027": (W, [], "Share of world population in democratic nations — Freedom House plus population data."),
 "0033": (U, [], "The aim of an adversary. An attribution of intent, which evidence of actions cannot establish. Same conflation family as ruling (d)."),
 "0134": (S, ["PAYEMS"], "4.6 million jobs over a dated window is a direct difference on nonfarm payrolls. The comparison to Japan and the EU needs OECD data, so the second half is web-tier1 — arguably compound."),
 "0147": (W, [], "$880bn of tax relief over five years — Treasury/CBO scoring."),
 "0155": (W, [], "A count of programmes cut or eliminated — the budget documents."),
 "0156": (W, [], "Projected $14bn savings — budget documents; forward-looking, so partially a projection."),
 "0189": (W, [], "A 22 percent increase in clean-energy research funding — DOE budget request."),
 "0217": (W, [], "Youth drug use down 19 percent since 2001 — Monitoring the Future / NSDUH. Survey series, not FRED, so not series-core as defined here."),
}

CLASSIFIED = {"trump_2026": T, "biden_2022": B, "obama_2014": O,
              "clinton_1998": C_, "gwbush_2006": G}

#: Flagged separately — NOT a gate-withheld claim, and no change is proposed.
READJUDICATION = {
    "sid": "trump_2026:0466",
    "page_position": "#118",
    "verdict": "TRUE",
    "confidence": 0.9,
    "decidability_class": "readjudication-candidate",
    "why": (
        "The claim's core is causal — 'ALL BECAUSE she wore the uniform, she "
        "was shot' asserts the shooter's MOTIVE. The panel's stated reasoning "
        "verified that she was shot while on duty in uniform, which is a "
        "weakened paraphrase: evidence of the shooting does not reach why the "
        "shooter fired. Same conflation family as ruling (d) — a decision "
        "procedure that settles one proposition being credited with settling a "
        "neighbouring, harder one. No change now; logged for owner-visible "
        "re-adjudication in D17-d."),
}


def build() -> dict:
    claims = []
    per_speech = Counter()
    for speech, run in sorted(HEADS.items()):
        doc = json.loads((RUNS / f"{run}.json").read_text(encoding="utf-8"))
        texts = {c["sid"]: c["text"] for c in doc["claims"]}
        table = CLASSIFIED[speech]
        for row in doc["rows"]:
            if row.get("provenance_code") != GATE:
                continue
            sid = row["sid"]
            key = sid.split(":")[1]
            per_speech[speech] += 1
            cls, series, why = table.get(
                key, ("substantive", [], "UNCLASSIFIED — not reached by the "
                                         "desk pass; treat as unknown."))
            claims.append({
                "sid": sid, "speech": speech, "text": texts.get(sid, ""),
                "decidability_class": cls,
                "candidate_series": series,
                "why": why,
            })
    totals = Counter(c["decidability_class"] for c in claims)
    n_series = totals.get("series-core", 0)
    doc = {
        "schema": "truthbot-d17d-triage v1",
        "generated_from": {sp: rid for sp, rid in sorted(HEADS.items())},
        "method": ("$0 desk classification of every gate-withheld claim "
                   "(provenance_code == insufficient-qualifying-evidence) in "
                   "all five speeches. No retrieval, no model call. Input to "
                   "an owner scope decision, not a verdict."),
        "gate_withheld_per_speech": dict(sorted(per_speech.items())),
        "gate_withheld_total": sum(per_speech.values()),
        "class_totals": dict(totals.most_common()),
        "cost": {
            "series_lane_usd_per_claim": SERIES_USD_PER_CLAIM,
            "series_lane_claims": n_series,
            "series_lane_projected_usd": round(n_series * SERIES_USD_PER_CLAIM, 4),
            "basis": ("measured on the d17c-wave2 escape run "
                      "($0.003124/kchar, $0.1089/claim)"),
            "web_tier1_lane_usd": None,
            "web_tier1_note": (
                "UNPRICED. No measured constant exists for a retrieval-bearing "
                "lane on these packs, and S-12 forbids borrowing one — a "
                "per-claim number measured on a different payload is what ran "
                "the escalation 8.2x over. Needs its own $0 estimate pass "
                "before it is costed."),
            "compound_split_lane_usd": None,
            "compound_split_note": (
                "UNPRICED, and blocked before it is priced: segmentation has to "
                "land before retrieval can be scoped for these claims."),
            "substantive_lane_usd": 0.0,
            "substantive_note": (
                "$0 by definition — these are permanent honest abstentions. "
                "The work is display, not retrieval: they should read as "
                "'cannot be verified', not as an unfinished backlog item."),
        },
        "readjudication_candidates": [READJUDICATION],
        "claims": claims,
    }
    return doc


def main() -> int:
    doc = build()
    OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print("gate-withheld per speech:")
    for sp, n in doc["gate_withheld_per_speech"].items():
        print(f"  {sp:<14}{n:>4}")
    print(f"  {'TOTAL':<14}{doc['gate_withheld_total']:>4}")
    print("\nclass totals:")
    for cls, n in doc["class_totals"].items():
        print(f"  {cls:<16}{n:>4}")
    c = doc["cost"]
    print(f"\nseries lane: {c['series_lane_claims']} claims x "
          f"${c['series_lane_usd_per_claim']} = "
          f"${c['series_lane_projected_usd']}")
    print("web-tier1 lane: UNPRICED · compound-split: UNPRICED · "
          "substantive: $0 (display only)")
    unclassified = [c_["sid"] for c_ in doc["claims"]
                    if "UNCLASSIFIED" in c_["why"]]
    if unclassified:
        print(f"\nUNCLASSIFIED ({len(unclassified)}): {unclassified}")
    print(f"\n-> {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
