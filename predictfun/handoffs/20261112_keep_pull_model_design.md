# Keep-or-pull model for a resting order: review of the current guard, and a design to test (2026-11-12)

Session ce68ebe0. USER: "We should align the target first, we rest orders there, and the model will predict if we keep there or we
pull to avoid adverse selections. Review the model first, design it properly to test". Design only; nothing here has been run.
Every choice below is fixed before any label or table is seen. The script that implements a stage declares its readings and is
committed before its first run.

## 1. Review: what the current guard predicts

Sources:
- `freeze/pm_ladder_v1/params.json` (v3 uses the same guard);
- `research/pf_badfill_grid_tx.py` (the label);
- `research/pf_guard_freeze.py` (the fit);
- `step4/minute_lines_t60k.json` (the lines).

| | Current guard (v1 / v3) |
|---|---|
| Row | Every 500 ms of the live 0-240 s window, per side. 8 read days, 1,071,151 rows. |
| Label | v = sg (mid(t_sweep + 1 s) - p) if the known best p is traded THROUGH in (tau + 22, tau + 522] ms, else 0. |
| Inputs | 32 market features: Binance returns, flow, imbalance and activity; Predict spread, imbalance, L1 / L3 sizes, mid changes, pending flow; t_rel, sigma, basis, gap. |
| Model | LightGBM regression (lr 0.05, 31 leaves, min 500 per leaf, 300 rounds), label clipped +-10. v = model - c60, where c60 is the 60% quantile of in-sample predictions. |
| Use | Quote and keep a side only while v > the per-minute PM90 line (0.049 / 0.062 / 0.079 / 0.092 c): the top 10% of moments of each minute, set in-sample on the 8 days. |

In plain words, the guard answers: "for an order at the back of the queue, how much would a sweep in the next half second cost at
1 s?". It then admits the best tenth of moments. That is an admission filter for posting. It is not the question a resting order
asks.

### Misalignment with keep-or-pull for a resting order

| # | The guard assumes | Keep-or-pull needs | Evidence |
|---|---|---|---|
| 1 Position | Back of the queue: only a trade through the level fills us | The order's real place. Near the front, trades AT our price fill it: small benign trades, and the start of sweeps | 61% of our 11-08 at-best shares came from trades at the price (B12 / B12e / B12f); B21's resting orders fill with 15-25% of the level ahead |
| 2 Counterfactual | Pull = 0 | Pulling loses the place. We re-enter at the back once the signal clears, and the benign fills that would have come at the front are lost | B21: pulling on thinning or thin side is worse than never pulling at the 250 ms clock |
| 3 Horizon | One 500 ms exposure, 1 s markout | Long enough for a pulled order to regain its place (seconds); 1 s and 5 s markouts | B16 / B16c: the horizon flips signs |
| 4 Latency | 22 ms cancel inside the label | Fills before the cancel lands happen whichever we choose; value only what comes after t + L | CLAUDE.md rule 7 |
| 5 Inputs | Market state only | Also our order (size ahead, place, age, ticks behind the best, recent fills) and our level's dynamics (thinning over 100-500 ms, trades at our price) | B18: 50% of our fills come while our level thins |
| 6 Decision | A rank: the top decile of moments | A priced boundary: pull when keeping is expected to be worth less than pulling | Guard status review 11-11 17:42Z: PM90 mixes admission with retaining an order |
| 7 Population | Every moment, a fresh order on each row | The moments when our order is actually resting, on the neutral never-pull path (not the fills a policy let through) | CLAUDE.md rule 1 |

What the tests so far say:
- The guard is a reasonable admission filter for v3.
- As a pull signal for a resting order it is the best of the five tried (+1.0 c/sh against never pulling, B21), but not enough.
- Speed does not change that (B21b).
- Retrained variants did not move the decision: B14 (pressure features), B16 (back-of-queue resting label) and B16c (combined
  horizons). All of them kept the guard's question.

## 2. Design: the keep-or-pull model

**Decision.** At each 50 ms step while our order rests on a side (live, no cancel in flight), we keep or pull.
- The policy rests at the known best by default, as B21's rest lane: held while within 3 ticks; re-placed at the back once the pull
  state has been clear for 250 ms.
- It pulls when the model's predicted Delta is below 0.

**Target.** Delta(t) = Q_keep(t) - Q_pull(t): one decision away from the never-pull reference path.
- Q_keep is the value of the fills our order gets in (t + L_c, t + H] if it keeps resting. This is the never-pull path itself,
  re-placements after full fills included.
- Q_pull is the value of the fills in (t + L_c, t + H] if we cancel at t and then follow the rest policy. The order comes back once
  the pull state has been clear for 250 ms, at the back of the then-known best, and rests to t + H. It is simulated as a fork from t
  on the same market.
- Fills in (t, t + L_c] happen on both paths because the cancel has not landed, so they are excluded. L_c = 22 ms, or 95 ms after
  the create while it is pending, as the replay.
- Value per fill = sg (mid(anchor + h) - price) x shares, with h = 1 s (primary) and 5 s. The label is in c per share of our 5-sh
  order.
- H = 5 s (primary): the re-entry plus the time to regain the place. H = 2 s and 10 s are reported, not selected.
- Fill clock: one label per trade-time shift (0 / 50 / 250 ms). The model trains on 50 ms and is evaluated at all three.
- Queue rule M (the engine's) is primary. Rule L (no credit for cancels ahead, conservative) runs beside it.
- Rows within H of the window's end are truncated, with a status.

**Rows.**
- Every decision step on the never-pull path where the order rests.
- Training uses every 250 ms (memory); evaluation decides at every 50 ms step.
- Rows overlap in their futures, so no table ever sums labels. Value is measured only by simulating the policy.

**Inputs** (all known at t on the live clock; our own fills with the 100 ms own-fill delay):
- market: the 32 guard features, the guard value v itself (so the model nests the guard), and B13a's 13 pressure features;
- our order: size ahead (estimated from the public book, the rule M view), shown size at our price, place = ahead / shown, age,
  ticks behind the best, shares filled, time since our last known fill;
- our level: shown size change over 100 / 250 / 500 ms, trades at our price in the last 250 ms / 1 s, the opposite best's size
  change, the level's age;
- sides pooled: directional features signed by side, and the side as a feature.

**Model.**
- LightGBM regression on Delta at 1 s, with the guard's pinned parameters (pf_model_spec PARAMS, 300 rounds) and the label clipped
  +-10. A 5 s model runs beside it.
- The boundary is fixed: pull when Delta-hat < 0. A grid of thresholds is printed, never selected.

## 3. Test

**Data.**
- The order-level lane simulator (strategy review 6.5, B20, B21) is ported from the 11-08 cohort to the 9 replay days: 10-23..27,
  10-31, 11-01..03.
- Every day has the inputs it reads: step1 books (5 levels), legs_timed2 / legs_tk, clock_rl_d2.json, step5 rows_rl_d2_pk and
  preds_t60k.
- Windows cover the live 0-240 s.
- All 9 days are consumed, so this is exploratory.

**Evaluation**, leaving one day out (fit on 8, score the 9th):
- E1, prediction: out-of-fold rank-IC of Delta-hat against Delta per day, compared with a guard-only model (Delta on v alone). This
  measures skill beyond the guard (rule 9).
- E2, decision (deciding): simulate the policy on the held-out day, with Delta-hat computed along the policy's own path (our order
  state as it evolves).
  - Comparators: never pull (Z-none), the guard as a pull signal (Z-guard, B21's best), v3 (J-v3), and a random pull matched on
    pulls per window-side-minute (200 draws).
  - Metrics: $/day and c/sh at 1 s and 5 s (60 s and settlement beside), under M 0 / 50 / 250 and L 50; U 50 as the
    front-of-queue bound; presence, place at fill, pulls per minute.
- Ceiling: the same policy driven by the realised Delta (perfect foresight). That is what a perfect keep-or-pull model could give.

**Readings** (exact rules fixed in the script; 9 consumed days, intervals clustered by day):
- R240, label: mean Delta on the never-pull path, the share of rows with Delta < 0, and the share of the path's loss in those rows.
- R241, ceiling: is the perfect-foresight policy positive at M 0 and M 50? If not, stop: no model can rescue resting by default
  with this target.
- R242, skill: out-of-fold IC above the guard-only model's on >= 7 of 9 days.
- R243, decision: $/day at 1 s and 5 s above Z-none, Z-guard and the matched random p95, at M 0 and M 50, on >= 7 of 9 days; and
  c/sh > 0 at M 0 and M 50.
- R244, against v3: the 6 $/day cells (1 s / 5 s x M 0 / 50 / 250). BEATS-V3 if 6 of 6.

**Controls and falsifiers** (any failure means no table):
- On 11-08 the keep path reproduces B21's Z-none fill for fill.
- Fork cases with known answers:
  - a pull before a sweep avoids it;
  - a pull loses a later benign fill;
  - the re-entry joins at the back;
  - a fill inside the cancel latency counts on both paths.
- Planted errors must break those cases: no cooldown, re-entry at the old place, latency fills credited to the pull.
- The ported simulator passes, on every day, the J-v3 checks it passes on 11-08, with status counts per day.
- A shuffled-label falsifier (labels shuffled across windows within each day) must fail R242 and R243.

## 4. Stages and gates

| Stage | What | Gate to go on |
|---|---|---|
| S1 | Label builder and forks on 11-08: parity with B21's Z-none, fork cases, label stats, perfect-foresight ceiling | Ceiling positive at M 0 and M 50, else stop |
| S2 | Port the lane simulator's inputs to the 9 days | Per-day statuses, J-v3 checks pass |
| S3 | Labels and inputs on the 9 days (0-240 s, 250 ms training rows) | Memory and time measured on a pilot day first (4 GiB cap, self-guard at 50%) |
| S4 | Leave-one-day-out fits, E1 / E2 with the controls; R240-R244 | R242 and R243 |
| S5 | If it passes: freeze (builder committed, receipt, number of candidates), then a forward test on >= 5 untouched days after the v3 block (11-14 onward) | CLAUDE.md rules 11-12 |

## 5. Limits

- The replay's queue and fill model (rule M) is not validated (B1a), and the labels inherit it. Rule L runs as the conservative
  variant; B1 (live fills with the A4b capture) calibrates it.
- The fill clock is unknown, hence three label worlds.
- Our order is not in the public book, so other makers' reaction to it is not modelled.
- One 5-sh order per side. CAP10 / RO60, MICRO / G4 and Binance's hard guards are not in the lane simulator.
- The target is one decision away from never pull. The learned policy pulls more than its reference, so a second iteration
  (relabelling on the learned policy's path) may be needed.
- Consumed days: a pass here makes a candidate, nothing more.
