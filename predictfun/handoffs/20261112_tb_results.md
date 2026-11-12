# TB: v3 without its quoting limits, and "create the level" (results, 2026-11-12)

Session ce68ebe0. USER: "We can remove the quoting related limits first" / "we need to get front position in the queue" / "We can
test, but finding missing information or explain the move will be more important".

Script and run:
- Script `research/pf_tb_check.py`, driver `research/run_tb.sh` (72a03e1a, committed before the first run).
- Run 17:21-18:52Z in the claude slot, sharing it with MOVE.

Setup:
- Engine: the frozen `pf_step4_ladder_v3.py`, never edited. FV3's settings, 9 consumed replay days, arms set through PF_ARMS only.
- **B0:** v3 without CAP10, G6 and RO60. Quotes stop at 60 s. The venue rules (XMIN1, XNA, XV) are kept.
- **IM95:** B0 plus the engine's improve rule. When the known spread is 2 ticks or more and v >= the minute's p95 line, it quotes one tick
  inside. Our order is then alone at a new price.
- Timings J / X / X250.

Controls passed:
- The in-sample arm noCAP_S60 is reproduced row for row on 10-23 and 11-03.
- The limits withhold nothing in B0 or IM95.
- The improve rule is engaged in IM95 only, about 5,900 decisions a day.
- The age bins add up.

## Answer

Fills in the first 60 s, $/day at 1 s:

| Timing, arm | All | Without young fills | B1a M 0 / M 50 | Settlement (ties 50) | Shares/day | Cancel-pending | Shares with <= 5 sh ahead | Median queue ahead |
|---|---|---|---|---|---|---|---|---|
| J v3 | 50.27 | 9.73 | 8.76 / 7.61 | 90.60 | 3,418 | 41.3% | 4.5% | 315 |
| J B0 (limits off) | 66.90 | 14.23 | 12.18 / 10.51 | 50.65 | 4,358 | 38.4% | 5.0% | 308 |
| J IM95 | 80.01 | 15.82 | 14.06 / 12.25 | 87.28 | 5,603 | 41.3% | 18.7% | 208 |
| X v3 | 39.46 | 8.81 | 7.43 / 6.16 | 50.39 | 2,876 | 40.5% | 2.6% | 429 |
| X B0 | 48.98 | 12.25 | 9.84 / 7.84 | 32.11 | 3,581 | 38.2% | 3.2% | 418 |
| X IM95 | 55.16 | 12.66 | 10.59 / 8.54 | 65.70 | 4,392 | 41.2% | 10.4% | 332 |
| X250 B0 | -39.54 | -13.17 | -8.91 / -8.19 | -19.90 | 3,815 | 37.0% | 2.1% | 456 |
| X250 IM95 | -39.02 | -13.17 | -8.82 / -8.24 | 14.74 | 4,498 | 40.4% | 9.3% | 363 |

Readings:
- **R360 HIGHER:** with the limits off, the value without young fills is +$4.5 a day (J, 8 of 9 days) and +$3.4 (X, 7 of 9). It is
  about a quarter more volume.
  - Settlement falls by $40 (J) / $18 (X): more inventory is carried to the end.
  - Settlement is noisy at 9 days, and v3 chose CAP10 on these days.
- **R361 NOT-SHOWN:** improving inside a spread of 2+ ticks adds +$1.6 (J) / +$0.4 (X) a day without young fills, on 6 of 9 days at
  both timings.
- **R362 NOT-SHOWN:** at the 250 ms fill clock improving adds $0.00.
- **R363 FRONT:** improving does buy queue place. With 5 sh or less ahead at activation: 18.7% of filled shares against 5.0% (J). The
  median queue ahead falls from 308 to 208 sh. Volume is +29%.

## What it means

1. **The quoting limits cost a few dollars a day of fill value in the first minute.** Without them, the timing work measures entries,
   not the cap. Inventory then needs its own control, judged at settlement.
2. **Creating the level gets us to the front, but the front alone barely pays.**
   - The extra front fills are worth about as much as the fills they replace at 1 s.
   - Settlement is higher (+$37 / +$34) but not tested by day.
   - This matches MOVE: the edge at the front is avoiding the trades that take the whole level, which needs the pull decision. Being
     first is not enough.
3. Everything still loses at the 250 ms fill clock.

## Limits

- The replay fill model (rule M) is unvalidated (B1a).
- Consumed days, first minute only. v3's parameters were chosen on these days.
- Spreads of 2+ ticks are rare in the first minute, so the improve rule acts on a small share of the time.

Outputs: `data/predict_fun/research/tb_20261112/` (tb.json / tb.txt, logs); engine runs under `step4/ladder_tokyoreal_custom_e2_preds_t60k_tb*`;
copies in `reviews/tb_20261112/`.
