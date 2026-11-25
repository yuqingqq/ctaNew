# Take + maker exit on all complete markets — v2, 2026-11-25

**Answer.** On all **281 complete markets** of consumed Oct 24–27 (v1 had 19), the take + guarded maker exit earns
**+3,933.54 USD**, +14.00 per market. It is positive on 4/4 days, in
**256/281 markets** and in every opening hour of the day; the top 10 markets carry only
11% of the absolute per-market values. HOLD earns about the same total (+3,926.65), but in
fewer markets (199/281) and with about 4× the inventory. The 262 new markets earn
+14.18 per market, against +11.50 on the 19 v1 markets: v1 was
representative. Development evidence on consumed days; not validation.

Freeze 0e7882007 (v2 amendment + harness) before the census. Strategy code unchanged from v1 (ee0f16ee6).

## Population (census, outcome-blind)

281 complete, 516 missing_endpoint, and
355 stale_future_book, of all 1152 markets. The rule was
"all complete markets"; the projection fitted the 900 CPU s budget. Complete markets are those whose book stays two-sided and fresh
to 298 s, i.e. mostly still contested at the close. Markets that are decided early (one-sided near the end) cannot be replayed under
the current END-mark rule. This is the main limit on generality.

## Results

| Arm | END net USD | Per market | Completed turnover | Open at END | Taker fees | Days positive | Markets positive | Top-10 share | Mean abs position |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PASSIVE_EXIT | +3,933.54 | +14.00 | +4,687.49 | -6.36 | -747.59 | 4/4 | 256/281 | 11% | 7.4 |
| HOLD | +3,926.65 | +13.97 | +3,056.56 | +1,453.87 | -583.78 | 4/4 | 199/281 | 12% | 30.0 |

By day:

| Day | Complete markets | PASSIVE_EXIT | HOLD |
|---|---:|---:|---:|
| 20261024 | 81 | +834.73 | +954.02 |
| 20261025 | 72 | +939.18 | +861.69 |
| 20261026 | 70 | +1,250.44 | +1,211.39 |
| 20261027 | 58 | +909.20 | +899.54 |

By opening hour (PASSIVE_EXIT):

| Opening hour (UTC) | PASSIVE_EXIT net USD |
|---|---:|
| 00 | +208.91 |
| 01 | +113.54 |
| 02 | +175.16 |
| 03 | +188.32 |
| 04 | +123.89 |
| 05 | +222.15 |
| 06 | +158.07 |
| 07 | +208.68 |
| 08 | +120.98 |
| 09 | +284.40 |
| 10 | +163.39 |
| 11 | +80.07 |
| 12 | +147.63 |
| 13 | +73.85 |
| 14 | +65.40 |
| 15 | +66.99 |
| 16 | +181.26 |
| 17 | +279.39 |
| 18 | +154.14 |
| 19 | +130.70 |
| 20 | +335.86 |
| 21 | +124.14 |
| 22 | +221.11 |
| 23 | +105.51 |

- **Readings, computed:**
  - positive True, robust (≥ 3/4 days) True;
  - beats HOLD True (+6.90);
  - positive without the value of exit fills on orders < 100 ms old True (+4,054.33);
  - new markets positive True.
- **Activity:** takes filled 3,575 (121,486 shares); maker exits 89,984 shares,
  of which 18,095 on orders < 100 ms old.

## Integrity

- **Checks:** 5/5 pass ([v2_checks.json](v2_checks.json)):
  - the 38 v1 parity-gated ledgers are exactly equal (cache bits, identity, fills, orders, commands, grid, signals, IOC results,
    guard log without the logged age);
  - all exit posts are on the closing side at rung 0;
  - all IOC fills re-derived with fees;
  - ledger and FIFO identities on all 562 ledgers;
  - triggers recomputed.
- **CPU:** census 84 s, run 652 s, on the claude slot.

## Limits and next steps

- **Limits:**
  - consumed days only;
  - complete (contested-to-the-end) markets only;
  - values at the 298 s mark, though positions are small, so settlement differs little;
  - the replay's fill model (young exits are shown not to drive the result);
  - Tokyo latency is assumed;
  - sub-frame competition and capacity beyond 50 shares are untested;
  - there is no live IOC path yet.
- **Next (USER's call):**
  1. untouched days with the frozen rule;
  2. a valuation for decided markets (e.g. settlement), to cover the other 76 %;
  3. an IOC path in pmmmrust for a Tokyo shadow or canary.
