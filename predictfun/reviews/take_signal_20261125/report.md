# Taking on our Binance pressure signal — 2026-11-25

**Answer.** Yes: the signal that makes us withhold asks predicts a Predict move large enough to pay the spread and the 2 % taker fee, but **only if we act within ~50–90 ms of the Binance move**. Every BTC jump that turns our pressure guard ON was valued as an immediate IOC take, over 1,151 markets on consumed Oct 24–27. At the measured Tokyo clock, the taken shares are worth **+2.35 ¢ each after the fee three seconds later**, and +1.36 ¢ even if we also exit by taking. This holds on every day, in both directions, and is larger for stronger signals. Arriving at +150 ms leaves nothing (+0.07 ¢). Random-time takes lose the spread plus fee (-1.20 ¢), so the result is not mechanical.

Maker3 trades on the same signal: 57 % of its takes follow a same-direction trigger within 3.5 s, against 22 % at shifted times. The edge is a **quick in-and-out**, not a reason to hold early inventory: its value at resolution is noisy and depends on the market's outcome.

Consumed days, development evidence only; no replay of our strategy, no live claim.

## The 3789484 opening

Maker3's first fills there are **passive**. It raised its bid to 56 ¢ within about a second of a +2.6 bps BTC burst at 0.44–0.65 s and was filled as sellers arrived; its first take is at 82.8 s. Our guard fired at 550 ms on bn_r250 = +0.32485 bps; this packet's recomputation from the raw tape gives exactly that value.

A take at the first trigger (0.505 s, bn_r250 +0.3124 bps) would have bought the 24.96 shares still offered at 0.50, fee 1.00 ¢:

- +7.5 ¢/share at 1 s, +10.5 at 3 s, +12.5 at 10 s;
- +9.2 if sold by taking at 3 s;
- **-51 ¢ if held to resolution**, because the market resolved Down.

Maker3 instead turned its position over at 58–63 ¢ within ten seconds.

## Method (contract.md, declared before outcomes)

- **Signal:** the guard's own feature, `bn_r250 = 1e4·ln(mid(t)/mid(t−250 ms))` on the Binance bookTicker, recomputed from 259,826 qualifying raw-tape events. UP when ≥ 0.2855 bps (the ask guard ON), DOWN when ≤ −0.4667 bps (the bid guard ON); the frozen guard edges. One trigger per market and direction per 2 s: 31,854 take opportunities.
- **Execution:** the decision uses the Predict book known 63 ms earlier (publication + network); the limit is that best ask or bid, with no chasing. The order arrives 42 ms later and fills only if the price is still executable both in the last frame before arrival and in the first frame after it. Size is at most the touch in both frames (Q = 50 primary); taker fee 200 bps × min(p, 1−p).
- **Values per filled share:** after the entry fee, at the Predict mid 1 / 3 / 10 s later, at the 298 s END mark, at resolution, and as a taker round trip (sell at the opposite touch and pay the fee again).

## Results

| Clock | Triggers | Fill rate | 1 s | 3 s | 10 s | Exit by taking at 3 s | END | Resolution | USD at 3 s, Q 50, 4 days |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Tokyo (arrive +42 ms, book known at −63 ms) | 31,854 | 51% | +2.30 | +2.35 | +2.63 | +1.36 | +3.87 | +3.11 | +12,728 |
| Slow (arrive +86 ms, known at −126 ms) | 31,854 | 38% | +1.64 | +1.71 | +2.03 | +0.77 | +2.10 | +2.15 | +6,476 |
| Very slow (arrive +150 ms, known at −200 ms) | 31,854 | 17% | -0.06 | +0.07 | +0.21 | -0.75 | -0.27 | +0.80 |  |

By signal strength (Tokyo, values in ¢ per filled share; the edge rises with strength):

| abs bn_r250 (bps) | Fills | Markets | Fill rate | 1 s | 3 s | 10 s | Exit by taking at 3 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.29/0.47-0.5 | 7,762 | 1117 | 52% | +2.20 | +2.20 | +2.43 | +1.21 |
| 0.5-1 | 7,734 | 1087 | 51% | +2.11 | +2.20 | +2.41 | +1.21 |
| 1-2 | 770 | 427 | 57% | +3.82 | +3.96 | +4.98 | +2.98 |
| >=2 | 117 | 73 | 65% | +8.72 | +9.13 | +11.73 | +8.08 |

By time in the window: the early window is not special for the edge, but its fills are easier. The last 30 s have few fills because the book is thin or one-sided.

| Seconds into window | Fills | Fill rate | 1 s | 3 s | 10 s | Exit by taking at 3 s | END | Resolution |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 000-030 | 2,171 | 59% | +2.53 | +2.47 | +2.21 | +1.12 | +1.20 | +2.97 |
| 030-060 | 1,885 | 60% | +2.46 | +2.46 | +2.57 | +1.26 | -3.13 | +2.66 |
| 060-180 | 7,794 | 60% | +2.14 | +2.21 | +2.48 | +1.25 | +4.23 | +2.69 |
| 180-240 | 3,131 | 50% | +2.47 | +2.52 | +3.02 | +1.72 | +7.41 | +4.06 |
| 240-270 | 929 | 31% | +2.28 | +2.51 | +3.73 | +1.73 | +7.58 | +4.81 |
| 270-298 | 473 | 18% | +1.98 | +2.12 | +3.76 | +1.16 | +5.33 | +1.74 |

By day (every day positive at every short horizon):

| Day | Fills | Fill rate | 1 s | 3 s | 10 s | Exit by taking at 3 s |
|---|---:|---:|---:|---:|---:|---:|
| 20261024 | 3,385 | 51% | +2.16 | +2.11 | +2.43 | +1.12 |
| 20261025 | 2,600 | 50% | +3.09 | +3.13 | +3.74 | +2.21 |
| 20261026 | 6,194 | 52% | +1.85 | +1.88 | +2.02 | +0.86 |
| 20261027 | 4,204 | 51% | +2.59 | +2.75 | +3.04 | +1.76 |

Near-money books (10–90 ¢) carry the edge: +3.15 ¢ at 3 s on 12,045 fills, against +0.61 ¢ at extreme prices.

## Controls (amendment 1, declared before computing them)

| Tokyo clock, Q 50 | Fill rate | 3 s | Exit by taking at 3 s |
|---|---:|---:|---:|
| Signal triggers (primary) | 51% | +2.35 | +1.36 |
| Random times (same count per market) | 85% | -1.20 | -2.25 |
| Same triggers +7 s | 79% | -1.19 | -2.23 |
| Opposite direction | 85% | -5.31 | -6.37 |

The valuation code first reproduced every primary row exactly (63,708 rows, 0 mismatches). At random times the take loses about the spread plus fee and the stale price is usually still there. After a trigger, half the stale prices are gone before we would arrive. That is the competition and repricing the fill rule is meant to capture.

## Maker3 times its takes on the same signal

| Maker3 takes | Takes | With same-direction trigger | Shifted-time mean | Shifted p95 | Share of 500 shifts ≥ observed | 1 s ¢/sh, with trigger | 1 s, without | Resolution ¢/sh, with trigger | Resolution, without |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adding | 4,432 | 58% | 22% | 25% | 0.000 | +1.54 | +0.72 | +3.39 | -1.77 |
| reducing | 5,130 | 57% | 22% | 24% | 0.000 | +1.09 | -0.02 | -5.98 | -0.99 |

Its trigger-aligned position-adding takes earn +3.39 ¢/share to resolution against -1.77 for the rest. In the first 60 s, 53% of its adding takes are aligned, against 20% at shifted times. Association only; its private trigger is not identified.

## Declared one-split selection

On Oct 24–25 the best cell with ≥ 30 fills in ≥ 10 markets was |bn_r250| ≥ 2 bps in all windows: +11.30 ¢ at 3 s on 57 fills. On Oct 26–27 it earns +6.98 ¢ on 60 fills in 37 markets, positive on both days. That is a thin cell; the broad all-trigger result is the stronger evidence.

## What this means and what it does not

- **Taking is viable only at Tokyo latency.** The edge is +2.3 ¢ at +42 ms, +1.7 ¢ at +86 ms and 0 at +150 ms. This US host (~74 ms to Binance, Predict frames ~126 ms old) is too slow.
- **Take and leave, rather than accumulate.** The short-horizon value is reliable. Holding to resolution adds a large, outcome-driven variance: the per-market sign is a coin flip, as 3789484 shows.
- **Not yet measured:**
  - integration with our passive strategy (inventory caps, the asks we withhold, re-entry);
  - sub-frame competition, because frames are ~50 ms snapshots;
  - our own price impact on later marks;
  - API limits (40/s, 500/min; the triggers average ~6/min);
  - capacity beyond the touch;
  - fee changes.
- **Consumed days only:** this is development evidence, not validation; untouched days need the USER.

## Integrity

- **Sources and stages:** raw bookTicker hours only on the allow-list (no 20261028 stem is opened); stage 1 `bn_events.py` → `take_eval.py` → `summarize.py` → `controls.py` → `m3_align.py` → `check.py` → this file. Every number is generated from saved outputs.
- **Checks:** 8/8 pass ([checks.json](checks.json)):
  - plain raw rescan of bn_r250 on 2,643 triggers;
  - future Binance and Predict perturbations with no effect, plus a positive control;
  - synthetic fill cases;
  - the 2 s rule on all 259,826 events;
  - reconciliation, and pooled readings recomputed.
- **One failed attempt:** the first take_eval receipt refused tuple keys after the outputs were written; the partial outputs were removed and the run repeated with no rule change, as recorded in take_eval.json.

## Next (needs the USER's direction)

1. Add a **SIGNAL_TAKE overlay** to our replay of the 19 complete markets: when the guard turns ON, also send an IOC at the known touch (size ≤ 50), then let the existing passive rules and inventory caps manage the position. Compare full-book value against the current arms.
2. **Measure competition directly:** after each trigger, did the stale level disappear through trades (pending settlements) or cancels, and how fast? That calibrates the fill rule.
3. **Untouched-day confirmation** and any Tokyo shadow run are the USER's call.


## Amendment 2 — how strong must the signal be?

Declared in [amendment_2_strength.md](amendment_2_strength.md) before any of these outcomes. It covers every BTC change with
abs bn_r250 ≥ 0.05 bps (366,251 events), valued with the same parity-checked code: the guard's own rule reproduced all
31,854 primary rows exactly.

**The practical rule** (one trigger per market and direction per 2 s; Tokyo clock; ¢ per filled share). Every threshold is
positive in both halves; a stronger threshold earns more per share but triggers less often:

| Take when abs bn_r250 ≥ (bps) | Triggers | Fill rate | 1 s | 3 s | Exit by taking at 3 s | 3 s Oct 24–25 | 3 s Oct 26–27 | USD at 3 s, Q 50, 4 days |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 37,502 | 55% | +2.02 | +2.06 | +1.06 | +2.16 | +2.00 | +14,317 |
| 0.1 | 36,909 | 54% | +2.05 | +2.09 | +1.09 | +2.19 | +2.02 | +14,108 |
| 0.15 | 36,281 | 54% | +2.08 | +2.12 | +1.12 | +2.23 | +2.05 | +13,875 |
| 0.2 | 35,558 | 53% | +2.11 | +2.15 | +1.16 | +2.29 | +2.07 | +13,665 |
| 0.25 | 34,659 | 53% | +2.16 | +2.20 | +1.21 | +2.39 | +2.09 | +13,427 |
| 0.285 | 34,124 | 53% | +2.19 | +2.23 | +1.24 | +2.43 | +2.11 | +13,300 |
| 0.35 | 32,742 | 52% | +2.25 | +2.31 | +1.32 | +2.52 | +2.19 | +13,015 |
| 0.467 | 29,695 | 50% | +2.41 | +2.46 | +1.47 | +2.70 | +2.33 | +12,068 |
| 0.5 | 28,075 | 49% | +2.45 | +2.53 | +1.54 | +2.75 | +2.41 | +11,499 |
| 0.75 | 19,998 | 44% | +2.98 | +3.12 | +2.14 | +3.43 | +2.96 | +9,159 |
| 1 | 13,578 | 40% | +3.58 | +3.76 | +2.79 | +4.06 | +3.60 | +6,823 |
| 1.5 | 5,938 | 36% | +4.71 | +5.03 | +4.06 | +5.43 | +4.79 | +3,674 |
| 2 | 2,659 | 37% | +5.80 | +6.09 | +5.14 | +6.66 | +5.67 | +2,169 |
| 3 | 799 | 42% | +7.38 | +7.85 | +6.89 | +7.79 | +7.91 | +982 |

**The best strength measure is the implied Predict move in cents**: BTC move (bps) × the contract's sensitivity (cents per bps,
from the Predict price and time to expiry). Below ~0.5 ¢ a take earns about nothing in every part of the window; above it the
value climbs steadily:

| Implied Predict move (¢) | 0-60 s | 60-180 s | 180-270 s | 270-298 s |
|---|---:|---:|---:|---:|
| 0-0.5 | +0.17 | +0.09 | -0.03 | +0.24 |
| 0.5-1 | +0.73 | +0.63 | +0.65 | +1.15 |
| 1-1.5 | +1.16 | +1.06 | +1.29 | +1.98 |
| 1.5-2 | +1.65 | +1.37 | +1.89 | +1.29 |
| 2-3 | +2.00 | +2.03 | +2.41 | +1.50 |
| 3-5 | +3.18 | +3.10 | +4.21 | +2.51 |
| 5-10 | +5.29 | +5.55 | +6.84 | +2.43 |
| >=10 | +9.32 | +10.79 | +15.39 | +16.03 |

**Why even small BTC ticks pay.** A BTC change is usually the start of a burst: BTC moves on average another +0.4–0.5 bps in the
same direction within 250 ms, and Predict follows about 0.3 s later. Near-money books, per event:

| abs bn_r250 (bps) | Events | Fill rate | Implied move (¢) | 3 s | Exit by taking at 3 s |
|---|---:|---:|---:|---:|---:|
| 0.05-0.1 | 11,824 | 61% | +0.54 | +1.59 | +0.44 |
| 0.1-0.2 | 22,998 | 58% | +1.03 | +1.77 | +0.62 |
| 0.2-0.29 | 18,438 | 57% | +1.63 | +1.96 | +0.81 |
| 0.29-0.5 | 48,105 | 54% | +2.62 | +2.17 | +1.02 |
| 0.5-0.75 | 46,797 | 50% | +3.87 | +2.65 | +1.50 |
| 0.75-1 | 31,780 | 46% | +5.22 | +3.53 | +2.38 |
| 1-1.5 | 35,196 | 40% | +6.83 | +4.63 | +3.48 |
| 1.5-2 | 15,085 | 37% | +8.57 | +5.75 | +4.63 |
| 2-3 | 10,733 | 41% | +10.59 | +6.52 | +5.44 |
| >=3 | 8,050 | 44% | +19.29 | +7.77 | +6.70 |

**Trend agreement matters** (3 s value): when the 3 s BTC trend agrees with the jump, the take is worth more than when the jump
reverses an earlier move. This holds in every band and both halves (reading d = True).

| abs bn_r250 (bps) | 3 s trend agrees and is larger | Other | 3 s trend opposite |
|---|---:|---:|---:|
| <0.29 | +1.38 | +1.40 | +1.02 |
| 0.29-1 | +2.13 | +1.95 | +1.29 |
| >=1 | +4.64 | +4.11 | +2.01 |

Readings, computed in `strength.py`:

- (a) every bin is positive in both halves, so no break-even appears down to 0.05 bps at the mid mark (lowest bin passing: 0.05-0.1).
- (b) monotone across strength bins: True.
- (c) the delta-implied edge (implied move − cost) is too pessimistic. Even events with edge < −1 ¢ realise
  +0.27 / +0.65 ¢ because of the burst continuation, so read the empirical curve, not the model's break-even.

**What it means for a take rule.** Size the decision by the implied move:

- ≳ 0.5 ¢ to beat the entry cost at the mid;
- ≳ 1 ¢ if we must also exit by taking;
- the edge per share grows roughly in proportion above that.

Prefer jumps that continue a 3 s trend. All of this is consumed-day development evidence: any threshold chosen from this curve
needs untouched days (rule 11). Capacity, competition below the ~50 ms frame resolution, inventory and our own impact are
still unmeasured.
