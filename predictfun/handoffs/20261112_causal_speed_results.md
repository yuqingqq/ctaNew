# CS: the speed test with causal clocks (results, 2026-11-12)

Session ce68ebe0. USER: "Yes" to rerunning the speed test with honest timing. Replaces the VOID fast-entry arms of QR and QR2
(their decision was scheduled before the level existed; FINDINGS.md C6).

Script and run:
- Script `research/pf_causal_speed.py`, driver `research/run_cs.sh` (7e16d03e, committed before the first run).
- Post hoc `research/pf_cs_posthoc.py` (764352cd, committed before its run).
- Run 12:01-12:07Z in the claude slot.
- 9 consumed replay days, first 60 s, one 5-sh order per side. Exploratory.

Design:
- A new best level is born at the venue stamp tb. The decision is made at tb + f, using the birth frame only. The order goes live
  p ms later.
- f: today's own feed delay per frame (median ~53 ms), 30 ms, or 15.2 ms (the smallest measured feed delay plus 2 ms).
- p: 40 ms (today), 20 or 10 ms.
- The birth joins are added to v3; v3's own orders are unchanged.
- Primary metric (FINDINGS C1): $/day without fills on orders under 100 ms old.

Controls passed:
- The v3 lane equals S4's (0 differ).
- The old schedule reproduces QR2's F30 exactly, so the only change is the clock.
- The future-cut check: 400 cuts 5 ms before a birth, 0 differ. The planted acausal schedule differs on 40 of 40, so the check can
  fire.
- Synthetic cases; two planted errors caught.
- Post hoc: holding the pending-cancel window at 95 ms changes nothing. The gain is not an artifact of that assumption.

## Answer: a faster order path adds money; a faster feed alone does not

$/day at 1 s (M 50; v3 = 14.73 in all, 3.70 without young fills, 2.46 B1a-weighted):

| Arm (feed, order path) | Live after birth, p50 | All | Without young fills | B1a-weighted | vs v3 without young (days above, M 0 / M 50) | M 250 without young |
|---|---|---|---|---|---|---|
| today's, 40 ms (event-driven) | 89 ms | 15.72 | 3.80 | 2.61 | NOT-SHOWN (3 / 5) | -4.0 |
| 30 ms, 40 ms | 70 ms | 16.84 | 4.12 | 2.81 | NOT-SHOWN (5 / 6) | -3.9 |
| 15.2 ms, 40 ms | 55 ms | 15.66 | 3.85 | 2.64 | NOT-SHOWN (4 / 5) | -3.7 |
| today's, 20 ms | 69 ms | 21.05 | 5.18 | 3.50 | GAIN (9 / 8) | -3.7 |
| today's, 10 ms | 59 ms | 23.26 | 6.14 | 3.93 | GAIN (9 / 9) | -3.5 |
| 30 ms, 20 ms | 50 ms | 22.49 | 5.89 | 3.80 | GAIN (9 / 9) | -3.4 |
| 30 ms, 10 ms | 40 ms | 27.97 | 8.10 | 4.86 | GAIN (9 / 9) | -2.8 |
| 15.2 ms, 20 ms | 35 ms | 25.00 | 6.45 | 4.26 | GAIN (9 / 9) | -2.8 |
| 15.2 ms, 10 ms | 25 ms | 24.94 | 6.62 | 4.27 | GAIN (9 / 9) | -2.8 |
| random-time joins, 15.2 / 10 ms | | 15.43 | 4.15 | 2.62 | | -4.2 |

(v3 at M 250 without young fills: -4.13.)

Readings:
- **R320 NOT-SHOWN:** event-driven joining at today's speed does not beat v3 (3.80 against 3.70).
- **R321:**
  - Every arm with a 10-20 ms order path gains on 8-9 of 9 days at both clocks: +1.5 to +4.4 $/day at M 50 without young fills,
    +1.5 to +6.0 at M 0.
  - Every arm with today's 40 ms path does not gain, whatever the feed.
  - **SPEED-NEEDED = today's feed with a 20 ms order path** (live a median 69 ms after the birth).
- **R322:** the B1a-weighted figures agree (GAIN for all 10-20 ms arms; 30 ms / 40 ms also GAIN on 7 / 8 days).
- **R323:** at M 250 only the 15.2 ms-feed arms hold above v3 (7-8 of 9 days). Every arm still loses money at M 250.
- **R324 BIRTH-MATTERS:** at the same fastest clock, joining at births beats joining at random times on 9 of 9 days (6.62 against
  4.15).

## What it means

- **It is the order path, not the feed.** Arms that go live at about the same time after the birth differ by their placement leg:
  15.2 ms feed with a 40 ms path (live at ~55 ms) gains nothing, today's feed with a 10 ms path (~59 ms) gains on 9 of 9 days.
  - Why is not established. It is not the pending-cancel window (post hoc).
  - One candidate: with a slow path, switching to the new level leaves us out of the book for p ms after cancelling the old order.
    Not tested.
- **The money is small, and most of it is still young fills.**
  - Without fills on orders under 100 ms old, the best arm is $8 a day against v3's $3.7.
  - About 70% of every arm's headline is still young fills.
  - At M 250 everything loses.
- **The number to measure live is our order path:** decision to live at the venue. Today it is about 40 ms in the replay. B12b's live
  cancel round trip was p50 19 ms, so the live path may already be faster than the replay assumes. The A4b capture measures it.
- Results are non-monotone in the feed: 30 ms beats 15.2 ms at a 10 ms path (8.10 against 6.62). Unexplained.

## Limits

- Consumed days, exploratory.
- Rule M is not validated, and young fills are unvalidated (B1a).
- One 5-sh order per side.
- Exits and v3's own orders keep today's clock.
- The order-path speeds are hypothetical.
- A full-pipeline review (data, signals, simulator) was running when this was written. Its findings may qualify this.

Outputs:
- `data/predict_fun/research/cs_20261112/`: cs.json / cs.txt, day_*.json, posthoc/, logs.
- Copies in `reviews/cs_20261112/`.
