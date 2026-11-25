# The frozen take + maker-exit rule on five other days — v3, 2026-11-25

**Answer.** It holds. On the five days not used to develop the rule (Oct 23, Oct 31, Nov 1–3; 408 complete markets):

- **Conservative setting** (strict take fills + the measured 58 ms order latency): **+1,901.65 USD, +4.66 per market,
  positive on 5/5 days** and in 293/408 markets. That is
  81% of the in-sample +5.76 per market.
- **Lenient frozen simulator:** +5,923.13 (+14.52 per market,
  104% of in-sample), positive on 5/5 days.
- **Combined with in-sample:** every one of the 9 days is positive under the conservative setting.

Readings computed: holds out of sample True; similar size True.

These are consumed project days, so this is out-of-sample for this rule, not untouched-day validation. Days from Nov 4 on (reserved
for the pending forward races) and the sealed Oct 28–30 were not read.

## What was run

Amendment v3 ([amendment_v3_other_days.md](amendment_v3_other_days.md)), frozen 99ac28e53 before any v3 data.

- **Rule and simulators:** the rule is unchanged (ee0f16ee6). The two bound settings are from the review.
- **Binance input:** change points from the same builder. It reproduced the original Oct 24 data exactly (97,270 / 97,270) and
  kept no row before a segment start (446 flush rows dropped, including sealed Oct 30's).
- **Population:** every market through the unchanged loader. The Nov 3 23:55 market is censored because its tail sits in a forward
  file.

## Results

| Setting | END net USD | Per market | In-sample per market (Oct 24–27) | Days positive | Markets positive | Top-10 share | Without young exit value | Mean abs position |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LENIENT40 | +5,923.13 | +14.52 | +14.00 | 5/5 | 362/408 | 10% | +5,569.97 | 7.6 |
| STRICT58 | +1,901.65 | +4.66 | +5.76 | 5/5 | 293/408 | 13% | +1,579.23 | 4.1 |

By day:

| Day | Markets | Complete | LENIENT40 net | per market | STRICT58 net | per market |
|---|---:|---:|---:|---:|---:|---:|
| 20261023 | 288 | 70 | +788.86 | +11.27 | +300.15 | +4.29 |
| 20261031 | 288 | 102 | +1,537.24 | +15.07 | +597.00 | +5.85 |
| 20261101 | 288 | 88 | +1,084.56 | +12.32 | +380.32 | +4.32 |
| 20261102 | 288 | 72 | +1,260.10 | +17.50 | +281.35 | +3.91 |
| 20261103 | 288 | 76 | +1,252.36 | +16.48 | +342.84 | +4.51 |

## Integrity

- **Checks:** 8/8 pass ([v3_checks.json](v3_checks.json)):
  - Binance stems = the allow-list;
  - delays exactly 40 / 58 ms;
  - strict cap respected;
  - all IOC fills re-derived;
  - all exit posts on the closing side at rung 0;
  - ledger identities;
  - triggers recomputed;
  - the no-trigger falsifier.
- **Operational stops, both disclosed and committed:**
  - The declared CPU gate stopped the first run after Oct 23 (projection 1,211 > 1,100 s). Resource-only amendment R1 raised the
    cap to 1,300 s and reused Oct 23's ledgers after verifying them against a fresh census.
  - The first resume stopped at the 2.5 GiB memory self-guard because it held full ledgers. After the fix, the 4 ledgers written
    before that stop were recomputed byte-identical.
- **CPU:** 1,037.5 s for the run.

## Limits (unchanged from the review)

- **Population:** complete markets only (24–35 % per day), which carry about 2× the per-share edge of the rest.
- **Simulator:** the replay's passive fill model for the exits.
- **Latency:** Tokyo latency is assumed (decision on the 50 ms grid).
- **Untested:** competition below the snapshot resolution, and size beyond 50 shares.
- **Live path:** no IOC path in pmmmrust yet.
- **Checklist:** the full FINDINGS C1–C3 / C12 / C15 checklist is still open.

## Next (USER's call)

1. **Untouched days.** The reserved forward days (Nov 4 →) under the strict + 58 ms setting. This consumes them for the forward
   races, so it is the USER's decision.
2. **The full results checklist** (M 250 on the exits, cancel-pending, WS / E2 split).
3. **Live facts with the exec session:** a true IOC / marketable path, real latency, and a Tokyo shadow log of how fast stale
   quotes disappear.
