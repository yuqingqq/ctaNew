# Danger-pull model: rest at the best, pull on a 250 ms danger signal - design (2026-11-12)

Session ce68ebe0. USER: "We plan this well first" / "5s will be mostly noise. Can we try 250ms instead. Generally, we post there,
check danger signal to pull right? We use Binance signals, and predict fun's feed will be like 50ms stale". This is a plan only;
nothing has been run.

Why this replaces keep-or-pull S4 (9578ae84, diagnostic 5a494858):
- S4's label compared keeping with pull-and-re-enter. Both lose about 3 c/sh per fill, so most of its sign was luck on the
  re-entered order.
- Its 5 s window mixed in fills that later decisions control.
- 72.5% of its pulls fell on quiet moments.
- Here the target is our resting order alone, over the next 250 ms only. The pull means "be out while the danger lasts", so there
  is no churn.

## 1. Policy shape

- Rest one 5-sh order per side at the known best, held within 3 ticks: B21's rest lane, unchanged.
- At every 50 ms step, a danger score for the next 250 ms.
  - While the score is above the threshold: cancel if resting, and place nothing.
  - Re-place once it has been below for 250 ms (REST_COOL).
  - This is the guard trigger's structure, so there is no pull-re-enter loop.
- The score uses market inputs only (primary). It is then a fixed series per window-side and does not depend on our path.
  - In S4, our order's own state (behind, age, shown size) drove most false pulls.
  - An own-order variant is reported, not primary.

## 2. Clocks (the simulator's, unchanged)

- Predict.fun: known = max(update + 13.2 ms, receipt - day shift) + 2 ms. That is 53 ms stale at the median ("about 50 ms").
- Binance: 1.5 ms in.
- Our latencies: place 40 ms, cancel 22 ms, pending-cancel 95 ms, own fill known 100 ms. Fill clocks M 0 / 50 / 250 ms.

## 3. Target

- Rows: steps where our order rests on the never-pull path (B21's Z-none, the neutral reference path), one every 250 ms, at M 50.
  - Label windows then do not overlap: each fill belongs to one row.
- Label: D(t) = value of our order's fills in (t + 22 ms, t + 272 ms], in c per share of our 5-sh order.
  - Each fill is valued at its own time and price: sg x (mid(fill + 1 s) - price) x shares. The 5 s markout is reported.
  - Fills in the first 22 ms are excluded: a cancel sent at t cannot stop them.
- No fork and no re-entry leg.
- Statuses, never silent drops: rows with t > window end - 272 ms are "truncated", and windows outside the day or in the sealed /
  forward blocks are excluded.
- Reported lead variants: windows starting at t + 50 ms and t + 100 ms, to see how fast the information decays.

## 4. Inputs, as known at t

| Set | Inputs |
|---|---|
| B | Binance (fresh): returns 25 ms .. 3 s, imbalance and its 250 ms change, book age, flow 250 ms / 1 s / 5 s, trade count 1 s, activity relative to its 30 s mean |
| P | Predict.fun book as known (stale): spread, mid, micro-price deviation, imbalance L1 / L3, L1 sizes and their 250 ms changes, book age, book updates in 1 s, mid changes 250 ms / 1 s / 5 s, pending sizes |
| C | context: time in window, sigma, basis, gap (Binance-implied price minus the book) |
| G | the guard value v (back-of-queue sweep cost, 500 ms) |
| O | our order (variant only): behind, age, shown size at activation / now / smallest |

Model sets:
- G (v only): the baseline (CLAUDE.md rule 9).
- B+C and P+C: which feed carries the danger.
- BPC, BPCG: the primary is BPCG.
- BPCG+O: variant.
- BPCG-S: labels shuffled within day and side (falsifier).

## 5. Stage P1: prediction check (run first; stop if it fails)

- Leave one day out on the 9 consumed replay days (10-23..27, 10-31, 11-01..03), per side.
- LightGBM with the guard's pinned parameters and 300 rounds, regression on D clipped at +-10. Danger score = -prediction.
- Scores per held-out day:
  - activity AUC (any fill in the window);
  - loss AUC among rows with a fill (D < 0 vs D > 0);
  - mean D by danger decile;
  - one-step value at flag shares q = 5 / 10 / 20%: mean D of the flagged rows, and the mean D per resting row once the flagged
    rows are removed.
- Readings, declared before the first run:
  - **R270 signal:** BPCG's loss AUC above G's, and its flagged-row mean D below G's at q = 10%, each on >= 7 of 9 days ->
    DANGER-SIGNAL; else NO-SIGNAL (stop: these inputs do not see the danger in time).
  - **R271 one-step value:** removing BPCG's flagged rows raises the mean D per resting row above G's at the same q, at q = 5, 10
    and 20%, each on >= 7 of 9 days -> VALUE; else NO-VALUE.
  - **R272 feeds:** B+C vs P+C, loss AUC per day (descriptive, days B > P).
  - **R273 falsifier:** BPCG-S must read NO-SIGNAL; otherwise P1 is VOID.
- Gate to P2: R270 DANGER-SIGNAL and R271 VALUE, with R273 valid.

## 6. Stage P2: policy simulation (only after P1 passes)

- Threshold: the q-quantile of the model's danger on its own training days (no held-out information). q = 5 / 10 / 20% are three
  candidates, recorded as such. Primary q = 10%. Realised out-of-book share reported.
- Arms per held-out day at M 0 / 50 / 250:
  - DP-q;
  - DP-S (shuffled);
  - Z-none (never pull);
  - Z-guard (pull while v <= -0.10);
  - J-v3;
  - J-v3+DP (v3 with its entries vetoed while the danger is on: v3 is the only positive arm so far).
- Null:
  - 200 draws of DP-10's own trigger series rotated in time within each window-side by a random offset;
  - this keeps its out-of-book share and run lengths, so it is matched on time out of the book, not on pull count (S4's
    weakness);
  - run at M 0 and M 50.
- Readings, declared before the first run:
  - **R274 decision:** DP-10's $/day at 1 s above Z-none, Z-guard, J-v3 and the rotated p95, at M 0 and M 50, each on >= 7 of 9
    days, and pooled c/sh at 1 s > 0 at M 0 and M 50 -> DECISION-PASS; else DECISION-FAIL.
  - **R275 veto:** J-v3+DP above J-v3 at M 0 and M 50 on >= 7 of 9 days -> VETO-HELPS.
  - **R276 falsifier:** DP-S must read DECISION-FAIL; otherwise P2 is VOID.
- Day-clustered t intervals (8 df).

## 7. Controls

- B21 / S1 / S4 self-tests (the rest lane and its trigger path unchanged).
- C1: with the recording callback, the never-pull path equals Z-none, fill for fill.
- Label reconciliation: on a sample of rows per day, D recomputed from the fill list equals the stored label, and a planted shifted
  window must break it (falsifier).
- Statuses with counts.
- Memory self-guard at 2 GiB; the job goes through run_light, PF_SLOT=claude.

## 8. Cost

- P1 builds without forks: about 1 min per day, then about 25 min of fits (6 model sets x 18).
- P2: about 8 min per day with the rotated null at the primary q only. In total about 2 h if P1 passes.

## 9. Limits

- 9 consumed days, first 60 s of each window. A pass makes a candidate; a forward test on >= 5 untouched days after 11-13 follows
  (freeze first).
- The replay fill model (rule M) is not validated on live orders (B1a). Every number is a simulator number until the D2 live run
  (user's call).
- One order per side, and our order is not in the public book.
- CAP10 / RO60 / MICRO / G4 and the Binance hard guards are not modelled.
- The guard value v on a training day comes from guard models whose fits included the held-out day (indirect).
