# FA: the guard's false alarms and what follows them (results, 2026-11-12)

Session ce68ebe0. USER: "I think the gap are mostly 2 things, one our model detects the signals in time, but it fires quite some false
alarms, two, we don't get the long term good fills measurement" / "Yes, proceed".

Script and run:
- Script `research/pf_false_alarm.py`, driver `research/run_fa.sh` (5dff3d9f, committed before the first run).
- Run 14:08-14:10Z in the claude slot.

Design:
- **Neutral path:** one 5-sh order per side resting at the known best and never pulled. 9 consumed replay days, first 60 s.
- **Rows:** every 500 ms while an order rests, with v3's guard read as v3 sees it.
- **Outcomes, from that order's own fills (1 s markout):**
  - near = (t + 22, t + 522] ms, the guard's own horizon; the windows tile, so each fill counts once;
  - later = (t + 522, t + 5 s].
- **Primary population:** orders >= 100 ms old at t (FINDINGS C1), 547k rows.
- **Classes:** "false alarm" = the guard says stay out and no harm comes within 500 ms. It is defined by the outcome, so sums over
  false alarms are ceilings.

Controls passed: the never-pull lane equals S4's on every window-side; synthetic window cases, two planted errors caught.

Statuses: 1,714,173 rows (77,533 on younger orders); 70,086 with the start price not yet known (v = -1, excluded); 9,741
cancel-pending fills.

## Answer: the alarms are mostly "false" inside 500 ms, but staying through them loses afterwards

Primary population (orders >= 100 ms old):

| | M 0 | M 50 | M 250 |
|---|---|---|---|
| Alarm share of resting moments | 89.8% | 89.7% | 89.5% |
| Alarms with harm within 500 ms / quiet / good | 5.8% / 90.8% / 3.4% | 6.1% / 90.5% / 3.4% | 6.5% / 90.0% / 3.4% |
| Harm moments the guard alarms on (recall) | 97.7% | 97.6% | 96.9% |
| Near value $/day: false-alarm rows (ceiling) / true-alarm rows (harm avoided) | +109 / -392 | +110 / -418 | +109 / -444 |
| Later value per moment (c per order): after false alarms / true alarms / no alarm | -1.70 / -0.10 / -0.80 | -1.89 / -0.11 / -0.87 | |

Readings:
- **R340 MOSTLY-FALSE:** 94% of the alarms see no harm within the guard's own 500 ms. It catches 97-98% of the harm, because it is
  on 90% of the time.
- **R341, declared: NOT-SHOWN, because my instrument failed.** Rows without a guard value (0-6 per day) made the AUC NaN on 5 of 9
  days.
  - Post hoc, with those rows dropped: **ACTIVITY-NOT-HARM on 9 of 9 days** at M 0 and M 50.
  - The guard separates "a fill is coming" (AUC 0.67-0.72) better than "the fill will hurt" (0.60-0.65).
- **R342 CEILING-LARGE (just):** within 500 ms a perfect classifier could keep $110 a day of false-alarm fills, against the $392-444
  a day of harm the alarms avoid. 26-28% of it, just above the declared 25% line.
- **R343 BAD-FLOW-AFTER-FALSE-ALARMS (9 of 9 days at M 0 and M 50):**
  - After a "false" alarm, the resting order loses -1.70 / -1.89 c per moment over the next 4.5 s.
  - That is more than twice what it loses after no-alarm moments (-0.80 / -0.87).
  - After a true alarm the harm has already happened, and later value is about 0.

## What it means for the two points

1. **The guard fires mostly "false" alarms by its own 500 ms yardstick (94%), but most are early, not false.**
   - The loss comes after 500 ms: the target is too short to credit the guard for it.
   - So a classifier tuned to cut those alarms would put us in the book just before the losses arrive.
   - The guard also reacts to activity more than to harm (post hoc 9 of 9 days). Its alarms mark busy moments, which for an order
     at the back of the queue are the expensive ones.
2. **Measuring longer-term fills does not reveal good flow for our order.**
   - Over 5 s a resting order at the back loses in every guard state, and loses most after alarms.
   - The good long-term flow the top makers get (B12g) does not reach an order in our queue position in this replay.
   - So a longer target alone will not find good fills for us. They would need a better queue position (or a different selection
     of moments) first.
3. **This ties back to the queue.** The same moments are worth +0.15 c/sh to the top makers (GAP) and -1.7 to -1.9 c per moment to
   a back-of-queue order of ours.

## Limits

- The replay (rule M, our order mostly at the back), consumed days, first minute only.
- The never-pull path stands in for "staying"; it re-places after its own fills.
- The later windows overlap across rows: they are used as per-moment means only.
- R341's corrected result is post hoc.

Outputs:
- `data/predict_fun/research/fa_20261112/`: fa.json / fa.txt, rows_*.npz, day_*.json, logs.
- Copies (without the npz) in `reviews/fa_20261112/`.
