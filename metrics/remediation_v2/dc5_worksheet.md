# DC-5 worksheet — Phase-2 $0 regeneration dry-run

Generated from `bbe4a25922b8` on branch remediation-v2. No model calls; published artifacts + journaled pools re-judged under the new deterministic rules.

Rules: removals: fc-excluded (factcheck_exclusion_reason), era-violation (BOTH coded expected_claim_window AND fair-game utterance+7d), mutable-endpoint (is_mutable_latest), s5-capped (POLITICAL survivors beyond first 3 in candidate order). post-speech (utterance < d <= +7d) items become context-only, not removed. Credit rule: >=2 bearing, non-post-speech survivors with NEW tier in {Government, Wire, Established}; POLITICAL never credits. Pool candidate sets used where journaled (clinton_1998, gwbush_2006, obama_2014 rescue sids); trump/biden are pack-only — losses measurable, gains need retrieval.

## Per report

| report | claims | decided | losing items | cited losses | rationale-only losses | context-only cited | would-gate | items removed | post-speech items | tier flips |
|---|---|---|---|---|---|---|---|---|---|---|
| trump_2026 | 183 | 168 | 52 | 31 | 0 | 40 | 54 | 103 | 180 | 357 |
| biden_2022 | 111 | 101 | 20 | 13 | 0 | 19 | 22 | 36 | 94 | 207 |
| obama_2014 | 96 | 85 | 40 | 17 | 0 | 20 | 8 | 84 | 72 | 60 |
| clinton_1998 | 92 | 83 | 20 | 12 | 0 | 23 | 19 | 53 | 127 | 263 |
| gwbush_2006 | 48 | 42 | 21 | 9 | 0 | 9 | 7 | 39 | 58 | 182 |

**Site-wide:** 530 claims (479 decided) — 153 lose >=1 pack item, 82 lose a CITED item, 110 decided claims would now gate to Unverifiable (credits < 2). 315 items removed (fc-excluded: 15, era-violation: 48, mutable-endpoint: 37, s5-capped: 215); 531 items de-credited as post-speech context-only; 1069 tier flips.

## Scope option (a) — minimal

every sid with cited-item-lost OR would-gate: **166 sids**

- **biden_2022** (33): 0019, 0031, 0038, 0045, 0046, 0051, 0079, 0100, 0125, 0137, 0148, 0167, 0194, 0200, 0211, 0242, 0250, 0251, 0284, 0285, 0288, 0289, 0322, 0332, 0362, 0365, 0368, 0385, 0397, 0412, 0420, 0428, 0437
- **clinton_1998** (26): 0006, 0021, 0025, 0026, 0028, 0029, 0035, 0038, 0055, 0083, 0090, 0107, 0131, 0132, 0134, 0135, 0136, 0146, 0167, 0199, 0225, 0279, 0289, 0290, 0300, 0313
- **gwbush_2006** (14): 0033, 0047, 0052, 0057, 0106, 0134, 0147, 0155, 0156, 0171, 0183, 0187, 0189, 0248
- **obama_2014** (22): 0055, 0065, 0070, 0088, 0093, 0095, 0121, 0125, 0127, 0134, 0157, 0158, 0177, 0189, 0214, 0221, 0225, 0247, 0255, 0257, 0284, 0319
- **trump_2026** (71): 0010, 0017, 0019, 0020, 0022, 0023, 0024, 0029, 0032, 0034, 0035, 0040, 0043, 0045, 0055, 0057, 0098, 0100, 0109, 0113, 0114, 0130, 0135, 0137, 0161, 0186, 0205, 0206, 0208, 0248, 0252, 0255, 0257, 0281, 0310, 0311, 0319, 0325, 0326, 0327, 0328, 0334, 0340, 0356, 0374, 0402, 0409, 0422, 0428, 0436, 0457, 0466, 0487, 0497, 0509, 0556, 0559, 0569, 0586, 0594, 0614, 0616, 0638, 0652, 0659, 0660, 0664, 0665, 0667, 0684, 0685
