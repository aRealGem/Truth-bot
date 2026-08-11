# T-1 — D15 x wave intersections

_Generated 2026-08-09T22:58:54.216696+00:00 · $0 (set arithmetic over committed artifacts, no model calls)._

## The question

A claim D15 newly gates is Unverifiable by the gate, deterministically, for free. Sending it to an adjudication panel buys nothing. So every claim in BOTH the wave and D15's newly-gated set is a claim the wave can drop.

## Intersections

| Intersection | Size |
| --- | ---: |
| D15 newly-gated (50) n released (33) | **12** |
| D15 newly-gated n named extras (6) | **1** |
| D15 newly-gated n D16(alpha) released (2) | **0** |

### The 12 released claims D15 re-gates for free

| sid | D15 rule(s) that fired |
| --- | --- |
| `biden_2022:0146` | crec-congressional-record |
| `biden_2022:0154` | crec-congressional-record, dcpd-daily-compilation |
| `clinton_1998:0090` | crec-congressional-record, presidency-ucsb-address, recap-language, wcpd-weekly-compilation |
| `clinton_1998:0350` | crec-congressional-record, recap-language |
| `obama_2014:0153` | recap-language |
| `obama_2014:0255` | recap-language |
| `trump_2026:0279` | crec-congressional-record, dcpd-daily-compilation |
| `trump_2026:0325` | dcpd-daily-compilation, presidency-ucsb-address |
| `trump_2026:0329` | crec-congressional-record, dcpd-daily-compilation, recap-language |
| `trump_2026:0379` | crec-congressional-record, dcpd-daily-compilation |
| `trump_2026:0402` | crec-congressional-record, dcpd-daily-compilation, recap-language |
| `trump_2026:0405` | crec-congressional-record, dcpd-daily-compilation |

### Named extras D15 re-gates for free

| sid | D15 rule(s) that fired |
| --- | --- |
| `trump_2026:0343` | crec-congressional-record, dcpd-daily-compilation |

D15 and D16(alpha) do not collide: neither claim D16 releases is one D15 takes back. The two rules are pulling on different claims.

## Wave size

- Planned gross: **41** (33 released + 6 named extras + 2 from D16(alpha))
- Already answered by D15, no panel needed: **-13**
- **CEILING: 28 claims.**

This is a ceiling, not an estimate. D15 releases nothing in any speech or vintage, so re-gating with both rules active can only subtract from the released set. Nothing downstream adds a claim to the wave.

## Cross-checks

The original `33 + 6 + 2` assumed the three lists were disjoint. They are — verified, not assumed:

- released n named extras: empty
- released n D16(alpha): empty
- named extras n D16(alpha): empty
- D15 released nothing anywhere (the ceiling argument leans on this): `True`

## Vintage caveat

measure_d15.py loads the B1a sidecar only; regate_from_rescore.py merges B2 over B1a. T-4 re-runs both rules over the merged vintage and settles the difference.

