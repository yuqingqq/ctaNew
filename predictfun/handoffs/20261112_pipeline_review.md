# Pipeline review: data, time alignment, signals, actions, valuation (2026-11-12)

Session ce68ebe0. USER: "Can we also review the whole logic of the flow, from initial data processing - time alignment, etc to how
do we get signals, take actions, check if there are any existing errors there".

How the review was done:
- Three independent read-only reviewers, one per stage: data and time alignment, features and guard, simulator and valuation.
- Each got the code and FINDINGS.md's known-issue list, not my conclusions. They made no repo edits and launched no jobs. Their
  check scripts are copied to `reviews/pipeline_review_20261112/scripts/`.
- I re-verified every ERROR, and the RISKs that bear on today's results, in the code or data myself, and say so below.
- One follow-up measurement: `research/pf_cs_overlap.py` (committed before its run).

**Disclosure:** while verifying D3 I read the min / max timestamps (no prices) of 11-03's boundary book file, which holds 2.4 s of
forward day 11-04 (00:00:00-00:00:02). Recorded here and in HANDOFF.

## How the flow works (for reference)

1. **Step 1, per day.** REST match records become maker legs (one row per maker per taker order) plus the window registry
   (start / end price, outcome).
   - WS book frames carry a venue stamp `t_ms` and our receipt `recv_ns`.
   - Each leg gets a trade time: the WS pending-increment time when available (~55-65% of shares), else an estimate from the book
     drop (E2).
   - `legs_tk` gives the taker order id.
   - The day's clock shift is the median receipt delay minus 53.1 ms.
2. **Known time.** A frame is known at `max(t_ms + 13.2, recv - shift) + 2 ms`, as a running max. Binance data is known at its
   event time + 3.5 ms.
3. **Features, every 50 ms (`tau`).** 32 inputs from data known at `tau`:
   - Binance: returns over 25 ms-3 s, imbalance, book age, flows over 250 ms / 1 s / 5 s, trade count.
   - Predict.fun: spread, mid, micro-price deviation, imbalances, best sizes, book age, frame count, mid changes over 250 ms /
     1 s / 5 s, pending trade increments.
   - Context: time in window, sigma, basis, and `gap` (the Binance-implied probability minus the Predict mid).
4. **Guard value v, per side.**
   - LightGBM, leave one day out over 8 days.
   - The target, on a 500 ms training grid: v = side x (mid(t_sweep + 1 s) - price) if the known best price is swept through in
     (tau + 22, tau + 522] ms, else 0.
   - So v predicts the cost of an order at the best being run over by a sweep in the next ~0.5 s.
   - The quote line PM90 is the 90th percentile of minute-0 v.
5. **Actions (the replay).**
   - Lane J-v3: one 5-sh order per side at the known best while v > PM90, decided on the 50 ms grid.
   - Live after 40 ms; cancel effective after 22 ms (95 ms if still pending); own fill known after 100 ms.
   - Fills by queue rule M / L / U at trade time minus a shift of 0 / 50 / 250 ms.
6. **Valuation.** Each fill is valued at side x (mid(anchor + 1 s) - price), with mids forward-filled from valid frames.

## Findings, verified

| # | Stage | Severity | Finding | Verified | Affects |
|---|---|---|---|---|---|
| D1 | data | **ERROR** | Tie windows (start = end price; the venue pays 50/50, README) are recorded as "Up" (`pf_step1.py:57-59`, `outcome = resolution.name`) | yes: 14 ties on the 9 days, all "Up" | Settlement-valued numbers only (S4 `st`, B12g settlement, the frozen v3's carry to settlement); 1 s / 5 s markouts unaffected |
| D2 | data | RISK | 10-31: the step-1 and txmap signer indexes disagree (agreement 0.011 against 1.0 on other days), so 13% of first-minute shares lack a taker-order id | yes (txmap.json) | Taker-order analyses on 10-31 (GAP R303, B12 grouping); anchors move for 0.2-0.4% of shares |
| D3 | data | RISK | Step-1 day folders include the adjacent days' boundary hour files: 10-27 and 10-31 include sealed 10-28 / 10-30, 11-03 includes forward 11-04. The loader uses 10-30 frames for the 10-31 00:00 window (ws - 5 s); last-window outcomes come from after-midnight REST hours; the day clock shift uses all files | yes (file stamps) | Numbers negligible; a rule breach (C8 / P4) |
| D4 | data | RISK | Trade anchors are less precise than calibrated. E2 anchors (34-37% of first-minute shares) differ from WS times by IQR about -250 / +100..200 ms. Taking each taker order's minimum time moves 1-1.5% of WS-timed legs > 100 ms earlier | reviewer's check (data) | Every fill time; the 100 ms age split (C1) is within anchor error for about a third of shares |
| D5-D6 | data | NOTE | E2 is used on 10-31 / 11-01 where step 1 adopted E4; the day shift is a whole-day median (hourly +-15 ms) | read | - |
| F1 | features | **ERROR** (S4 only) | `gap` uses the window's start price from tau = ws (`pf_step5_features_rl.py:162,191`). v3 is masked (v = -1 until the price is known, `pf_preds_startknown.py:46-48`); S4's model reads the raw feature (`pf_kp_s4.py:1702`) | yes (code) | S4 KP / KPS (bias in KP's favour; S4's negative result stands) |
| F2 | features | RISK | Binance is known at event time + 3.5 ms (`pf_step5_features_rl.py:81,101`). Binance-side delay spikes are not modelled; this host's tape shows heavy tails | yes (code); tails per the reviewer | Every guard value v; needs a Tokyo busy-hour capture |
| F3 | features | RISK (S4) | S4's folds leak the held-out day through v (the guard models of training days were fit on it) | read | S4 |
| F4-F7 | features | RISK / NOTE | PM90 is in-sample over the 8 read days (documented; quoting share changes <= 1.9 pp leave-one-out) and counts v = -1 placeholders (really the ~89.5th percentile); leave-one-day-out is not walk-forward; Binance hour files rotate 1-3 s early (80 rows of sealed 10-30 in 10-31's features); the guard grid drops NaN-label rows without a status | read / reviewer's checks | Small |
| S1 | simulator | **RISK, framing** | "v3" in all replays is lane J-v3, not the frozen pm_ladder v3. The frozen arm is `MICRO\|G4+G2+G6+G5+G3+G1+PM90+GS+LD2+CAP10+XNA+XMIN1+XV+RO60+S240\|mid\|AM:5:25:95` (`pf_ladder_v3_build.py:20`). J-v3 has v3's PM90 gate only: no MICRO side choice, Binance guards, rung, 25-sh size, CAP10, XMIN1, throttle, RO60 / S240, or event-driven decisions | yes (code) | Every "v3" / "vs v3" number in S4, QR, QR2, AGE, GAP, CS |
| S2 | simulator | RISK | Our own orders are served in time order (the frozen engine: best price first, `pf_step4_ladder_v3.py:171`) and two of them can take shares from one trade leg | reviewer's synthetic test; measured: overlap fills are 1-7% of the CS arms' shares | CS: excluding them, the 10-20 ms gains shrink by up to 14% (best arm 8.10 -> 7.01 $/day) and still beat J-v3 on 8-9 of 9 days |
| S3 | simulator | RISK (S4) | The rest lane asks S4's model only while the order still has shares, not in the 100 ms before an own fill is known | reviewer's synthetic test | S4 KP / KPS / null |
| S4-S7 | simulator | NOTE | S4's null is matched on planned pulls (known); the queue / post-only frame is the last at or before activation (the frozen engine uses strictly before; matters only at M 0 with equal ms); CS's regression control has no falsifier of its own, and its causal cut does not cut the guard series; untimed trades are counted but not printed in the tables | read / synthetic | Small |

## Checked and found correct

- **Known time:**
  - The shift's sign and units are right.
  - Known time is never before t + 15.2 ms, and the decision time is enforced.
  - Receipts are never before stamps; stamps never decrease in receipt order.
  - All times are UTC.
- **Raw REST:**
  - Pages are complete.
  - Leg keys are unique.
  - Maker amounts sum to amountFilled.
  - `legs_tk` is aligned, and shuffle / shift falsifiers fire.
  - The registry is otherwise consistent.
- **Features:**
  - All windows look back only.
  - The known-time formula is identical in features, guard grid and replay.
  - Features and predictions align exactly on (mid, tau) on all 9 days.
  - Side mapping is right.
  - v3's state equals (finite and v > PM90).
  - Leave-one-day-out thresholds come from training rows only.
  - Era purity holds.
- **Simulator:**
  - Event order is right.
  - Latencies match the engine.
  - The known best is on the known clock.
  - Rules M / L / U are as described.
  - Valuation sign, mids, the 60 s mark, settlement formula and $ are correct.
  - The window end is handled.
  - CS's schedules are causal.

## Not checked

- The false-match rate of step 1's leg-timing matcher.
- Netting of pending increments.
- When `recv_ns` is stamped.
- The o3 / c3 prediction pipeline (only a NaN filter here).
- Whether the guard label's sweep rule is right.
- The relay offset.
- The frozen engine `pf_step4_ladder_v3.py` itself.
- S1 / S4 fork controls beyond reading them.
- QR's decomposition beyond its identity.

## What it changes

- **Every "v3" result today is J-v3**, a simplified lane with v3's quote line. The register now says so (FINDINGS C11).
  - The frozen v3 has its own replay engine (`pf_step4_ladder_v3.py`). It needs its own test for any candidate, for example the
    order-path speed.
- The CS speed result survives the own-order overlap defect (S2), at up to 14% smaller.
- Settlement-valued numbers carry the tie error (D1). The markout results do not.
- The age split's 100 ms boundary is blurred by anchor error for E2-timed fills (D4). Report fills by anchor source in future (C1).
- S4's result stands as negative; its KP numbers are biased in KP's favour (F1, F3, S3).

## Fixes (proposals, the user's call)

1. Record ties as "Tie" in the registry with a settlement value of 50, and rerun the settlement-valued reports.
2. Test the order-path candidate on the frozen v3 engine (event-driven birth entries with a 10-20 ms path), not on J-v3.
3. Serve our own orders best price first in the sim2 family (as the frozen engine does), then rerun CS.
4. Drop adjacent-day frames by stamp in the loader, with a status, and filter boundary files in step 1.
5. Split C1's age table by anchor source (WS / E2).
6. Rebuild 10-31's taker ids with a consistent signer index.
7. A Binance receipt-time sensitivity once a Tokyo busy-hour capture exists.
8. S4's defects (F1, F3, S3): record only; S4 is a closed negative.
