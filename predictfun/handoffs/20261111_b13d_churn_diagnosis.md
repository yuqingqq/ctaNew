# B13d: why our live orders churn, from the engine's code and its own decisions (2026-11-11)

As of 2026-11-11 08:27Z (session ce68ebe0). USER "Can you check the code and do the coding, then update status and
handoffs". Read-only on pmmmrust (tag pfmm-ladder-v3-tokyo-6 = 4d65884, the code of the 11-08 live run; the checkout was
clean and was not changed). Diagnosis scripts `research/pf_b13d_withhold_attrib.py` and `research/pf_b13d_g4_check.py`
(descriptive). Data: the Rust engine's parity decisions of tokyo-5 (`review-exports/continuous-parity-20261108`, 105
windows of 11-07 / 11-08, v3's arm), the 11-08 cohort's live orders and frames. Consumed days only.

## The code path

| Location (pfmm-ladder-v3-tokyo-6) | What it does |
|---|---|
| `crates/pf-strategy/src/ladder.rs:413` | a side is adverse (withheld) whenever its fresh guard value is at or below the minute's pull line (PM90), re-checked at every decision |
| `crates/pf-strategy/src/ladder.rs:278` | AM:5:25:95: size 25 when the value is at or above the PM95 line, else 5, re-checked at every decision |
| `crates/pf-strategy/src/ladder.rs:495` | G4: under MICRO pricing a touch quote that falls behind the best (the lighter side) is withheld |
| `crates/pf-engine/src/ladder.rs:439` | a lane is re-requested whenever its wanted (price, size) differs from the last intent |
| `crates/pf-engine/src/ladder.rs:479-528` | a changed want cancels the current order and places a new one at once ("concurrent replace"), including a size-only change at the same price |

Decisions run on every known book frame (~50 ms), every guard line crossing, Binance guard flip, stale / resume point
and minute boundary. The guard value moves on a 50 ms grid.

## What triggers the cancels

Touch-lane transitions from quoting to not quoting in the first minute, recomputed from the logged decision inputs with
the code's order of reasons (6,866 transitions):

| Reason | Share | Back quoting within 100 ms | Median time to back |
|---|---|---|---|
| pull line: guard value <= PM90 | 83.5% | 53% | 100 ms |
| G4: microprice quote behind the best | 14.7% | 34% | 225 ms |
| inventory and other rules | 1.1% | 72% | 100 ms |
| G3 gap behind the touch | 0.6% | 50% | 71 ms |
| rung at the touch price | 0.1% | 20% | 252 ms |
| Binance guards G1 / G2 | 3 transitions | | |

Same-price size-only changes in the first minute: 5 -> 25 sh 1,383 times, 25 -> 5 sh 1,239 times, each a cancel and a
new order at the back of the queue (the live cancel log of 11-08 has these as 14.4% of 30,572 cancels).

Check of G4 on the live orders of 11-08: at 12,166 touch cancels at the best, our side was the lighter one in only 4.1%;
the bid / ask size balance at the best flips about 1.2 times a second (median window), slower than our ~4 decisions a
second per lane. G4 is the minor cause.

## Correction to B13c

B13c read the research guard 50 ms before each cancel and found it ON for 87% of pulls, and concluded the pulls were not
guard decisions. That read is one grid step before the crossing that triggers the cancel, so it shows the value just
before it fell below the line. The engine's own decisions show the opposite: 83.5% of touch withholds are the pull line.
The churn is the guard value hovering around the line and crossing it back and forth on the 50 ms grid. B13c's other
numbers stand (order life p50 23-38 ms, 82.5% re-placed at the same price within 1 s, ~380 sh ahead at every activation,
flow after the pulls +0.45 c/sh), with "pulls with the research guard ON" read as "pulls on a fresh crossing" and "OFF"
as "pulls after the value had been below the line for a grid step or more": would-be fills after fresh crossings +0.04
c/sh (1,631 sh), after sustained OFF -0.98 (267 sh).

## What this means for the fix

1. "Only act when the model agrees" is not enough: the model itself flickers at the line. A resting touch order needs a
   small hysteresis on the pull: keep it while the value stays within a narrow band below the line, for a short time.
   The earlier pull-buffer study (B3: 0.01 c band, at most 100 ms; mechanics only, `research/pf_review_gate_bands.py`) is
   that rule; it was never put into the live code. Wide bands were not supported (fills inside wide bands were
   adverse), so the band has to stay narrow.
2. Size changes should add or remove orders instead of cancel-and-replace (USER): keep the resting order, add an order
   for the difference when sizing up, cancel the newest add-on first when sizing down. Predict orders are signed and
   cannot be amended; a lane then holds more than one order.
3. G4's flicker (14.7%) could take the same treatment (withhold only a clearly lighter side), as a separate token.
4. Each is a strategy change: a new candidate (v3 is frozen, b261091d), implemented in a new research engine file (not
   in pf_step4_ladder.py, which the scheduled v3 forward run depends on) and in pmmmrust behind tokens that are off by
   default, with research-Rust parity. The replay cannot value queue place reliably (B1a), so the test is a live run with
   the A4b capture.

## Limits

The attribution recomputes the code's reasons from the logged inputs (tokyo-5's replayed decisions); a reason that
needs state not in the log (inventory details) is grouped. The slow venue cancels (B12b) are a separate issue.
