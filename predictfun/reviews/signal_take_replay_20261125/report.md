# Signal take added to our strategy — 19-market replay, 2026-11-25

**Answer.** Adding the take helps the **wide-band** strategy and not the base one.

- **Wide band** (cap 1,000, soft band 500): END net improves +289.97 USD over 19 markets, positive on 4/4 days, 12/19 markets better. Completed turnover is +277.83: the taken inventory is sold back passively at better prices.
- **Base band** (cap 200, soft 100): -28.06 USD, 2/4 days positive, 9/19 markets. A 50-share take pushes the position into the 100-share soft band, which changes the passive trading enough to lose more than the takes earn.

Both strategies remain loss-making overall on these markets. Consumed days, development evidence only: no validation, no promotion, no live claim.

## What was run

Frozen at 7e37888c8 before any take-arm replay ([contract.md](contract.md)):

- **Markets and timing:** the same 19 complete consumed markets, opening-only gate and ASYNC_FAST10 passive timing as the host experiment (inventory_exit_exception_20261125).
- **Host ledgers:** reused by hash; the take-disabled branch reproduced them exactly on 3 declared markets × 2 hosts.
- **Take:** when the pressure guard's own signal turns ON after 2 s quiet, an IOC in its direction: buy YES on the ask side, sell YES (native BUY NO) on the bid side. 50 shares within the hard cap; effective 40 ms after the decision tick; fills only at the venue best inside the limit with depth not reused; 200 bps × min(p, 1−p) fee.
- **Afterwards:** the unchanged host rules manage the inventory.

## Results

| Host | Host END net | With takes | Delta | Delta completed turnover | Delta remaining inventory | Taker fees | Triggers / sent / filled | Mean abs inventory | Both-side uptime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE | -916.05 | -944.11 | -28.06 | -9.76 | +4.86 | -23.16 | 376 / 197 / 135 | 72 → 70 | 42.8% → 45.7% |
| WIDE | -852.33 | -562.36 | +289.97 | +277.83 | +30.77 | -18.63 | 376 / 148 / 100 | 197 → 196 | 89.0% → 87.3% |

By day:

| Day | BASE delta | WIDE delta | Markets |
|---|---:|---:|---:|
| 20261024 | -66.37 | +22.31 | 7 |
| 20261025 | +28.09 | +84.53 | 5 |
| 20261026 | +18.63 | +124.73 | 5 |
| 20261027 | -8.41 | +58.41 | 2 |

The take fills themselves, per filled share after the fee (venue mid):

| Host | Filled take shares | +1 s | +3 s | +10 s | END |
|---|---:|---:|---:|---:|---:|
| BASE | 4,252 | +4.46 | +4.30 | +4.94 | +4.66 |
| WIDE | 3,153 | +6.26 | +6.23 | +7.44 | +7.31 |

- **Readings, computed:**
  - WIDE delta positive True, robust (≥ 3/4 days) True;
  - BASE delta positive False, robust False;
  - take +3 s markout positive under both hosts (True / True);
  - gain only in leftover inventory: WIDE False.
- **Concentration:** top-3 markets carry 35% of WIDE's absolute per-market changes; without 3789484 WIDE is still +252.63.

## Why many triggers were not taken

Of 376 triggers per arm:

- **Own-cross refusals:** 156 (BASE) and 228 (WIDE) were refused because our own resting order on the other side sat at or through the take price. When BTC jumps up the guard is pulling our asks, but they are still live while the cancel is pending. Buying at the ask would cross them, so the take is refused.
- **Room:** 23 BASE triggers had no room under the cap.

Of the IOCs sent:

- **BASE:** 75 filled fully, 60 partly, 57 found the price gone.
- **WIDE:** 47, 53 and 45.

Self-cross protection is correct live behaviour (pending-cancel orders can still trade). A rule that pulls our opposite quotes earlier, or takes only when none rest at the take price, is a separate candidate.

## Per market

| Host | Day | Market | Host END net | With takes | Delta | Takes filled/sent/triggers |
|---|---:|---:|---:|---:|---:|---:|
| BASE | 20261024 | 3697902 | -42.48 | -41.61 | +0.87 | 1/1/5 |
| BASE | 20261024 | 3699278 | -67.19 | -53.52 | +13.67 | 7/11/21 |
| BASE | 20261024 | 3700070 | +13.22 | +4.08 | -9.14 | 3/4/7 |
| BASE | 20261024 | 3702960 | +5.41 | -1.16 | -6.57 | 2/4/9 |
| BASE | 20261024 | 3704004 | -53.56 | -72.81 | -19.26 | 1/3/7 |
| BASE | 20261024 | 3712725 | -109.64 | -130.39 | -20.75 | 7/14/25 |
| BASE | 20261024 | 3716727 | -57.14 | -82.34 | -25.20 | 6/7/15 |
| BASE | 20261025 | 3723774 | +2.63 | +5.73 | +3.11 | 5/6/15 |
| BASE | 20261025 | 3734864 | -110.35 | -114.74 | -4.39 | 3/5/9 |
| BASE | 20261025 | 3737919 | -30.86 | -8.67 | +22.19 | 3/6/10 |
| BASE | 20261025 | 3763401 | -78.05 | -92.12 | -14.07 | 8/12/19 |
| BASE | 20261025 | 3776314 | -49.92 | -28.67 | +21.25 | 2/2/3 |
| BASE | 20261026 | 3789484 | +9.73 | +5.81 | -3.92 | 21/32/68 |
| BASE | 20261026 | 3803131 | +57.17 | +26.74 | -30.43 | 14/18/31 |
| BASE | 20261026 | 3804696 | -86.43 | -60.05 | +26.39 | 10/13/28 |
| BASE | 20261026 | 3838531 | -72.71 | -46.62 | +26.09 | 11/17/31 |
| BASE | 20261026 | 3850428 | -50.17 | -49.67 | +0.50 | 3/6/10 |
| BASE | 20261027 | 3877472 | -136.46 | -132.72 | +3.74 | 24/30/45 |
| BASE | 20261027 | 3899224 | -59.25 | -71.40 | -12.15 | 4/6/18 |
| WIDE | 20261024 | 3697902 | -113.70 | -113.70 | +0.00 | 0/1/5 |
| WIDE | 20261024 | 3699278 | -217.15 | -185.96 | +31.19 | 5/8/21 |
| WIDE | 20261024 | 3700070 | +87.52 | +86.75 | -0.77 | 2/2/7 |
| WIDE | 20261024 | 3702960 | +24.38 | +36.29 | +11.92 | 2/4/9 |
| WIDE | 20261024 | 3704004 | -101.03 | -115.14 | -14.11 | 1/4/7 |
| WIDE | 20261024 | 3712725 | -39.28 | -50.06 | -10.78 | 3/10/25 |
| WIDE | 20261024 | 3716727 | +255.86 | +260.73 | +4.87 | 5/5/15 |
| WIDE | 20261025 | 3723774 | +48.30 | +88.21 | +39.92 | 5/6/15 |
| WIDE | 20261025 | 3734864 | -194.25 | -177.29 | +16.96 | 3/5/9 |
| WIDE | 20261025 | 3737919 | +120.96 | +118.24 | -2.72 | 2/2/10 |
| WIDE | 20261025 | 3763401 | -215.82 | -175.72 | +40.11 | 8/9/19 |
| WIDE | 20261025 | 3776314 | -149.22 | -158.95 | -9.73 | 3/3/3 |
| WIDE | 20261026 | 3789484 | +248.80 | +286.15 | +37.35 | 15/22/68 |
| WIDE | 20261026 | 3803131 | -22.52 | +2.92 | +25.45 | 8/13/31 |
| WIDE | 20261026 | 3804696 | -179.58 | -134.02 | +45.56 | 10/14/28 |
| WIDE | 20261026 | 3838531 | -78.69 | -81.39 | -2.70 | 5/10/31 |
| WIDE | 20261026 | 3850428 | -47.23 | -28.16 | +19.07 | 3/4/10 |
| WIDE | 20261027 | 3877472 | -225.39 | -179.70 | +45.69 | 15/20/45 |
| WIDE | 20261027 | 3899224 | -54.28 | -41.56 | +12.72 | 5/6/18 |


## Integrity

- **Parity:** host parity was exact on all 6 declared runs.
- **Checks:** 6/6 pass ([checks.json](checks.json)):
  - triggers recomputed by a plain loop;
  - every sent limit is the known frame's best, with the 40 ms delay;
  - all 235 filled IOCs re-derived from venue frames, with fees;
  - ledger and FIFO identities on all 38 take ledgers;
  - synthetic sent / pending / no_room / near_end / disabled cases;
  - a corruption refusal.
- **CPU:** the run took 306 s, within the 450 s cap, on the claude slot.

## Next (USER's call)

1. **Own-cross handling.** With the take on, cancel opposite quotes at the trigger and send the take once they are gone, or only take when no own quote is in the way. That is the largest blocked share.
2. **Host choice.** The take needs inventory room: it works with the wide band and not the base band. Any adoption ties the take to the wider band and must be judged on the full book.
3. **Validation.** Untouched days before any live use.
