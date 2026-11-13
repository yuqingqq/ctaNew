# MOVE: what explains the move after a fill, and what information is missing (results, 2026-11-12)

Session ce68ebe0. USER: "we need to get front position in the queue and the prediction model does not perform well enough to keep/pull,
need to find ways to improve on these 2 key issues" / "finding missing information or explain the move will be more important".

Scripts and runs, all in the claude slot:

| Step | Script / driver | Commit | Run |
|---|---|---|---|
| MOVE | `research/pf_move_info.py` + `run_move.sh` | e62e1fbd, committed before the first run | 17:31-18:36Z |
| Report addition | the same script | 8d8fdd20 | before the report ran |
| AVOID (post hoc) | `research/pf_move_avoid.py` + `run_move_avoid.sh` | 85a17cf4 | 18:49-19:00Z |
| ops (post hoc) | `research/pf_move_avoid.py ops` | 3668f17c, after AVOID's result | 19:00-19:03Z |

The report addition prints Part A for all legs too; no reading changed.

Data:
- **Public tape only:** real maker legs and real book frames. No replay fill model is involved, so B1a's caveat does not apply.
- 9 consumed replay days, first 60 s of each window.
- Other venues' public trades (Coinbase, OKX swap, Bybit perp) on their 8 days. No file dated 11-04 or later was opened.

Controls passed:
- Synthetic target and causality cases with two planted errors, caught on every day.
- A leak assertion on all 591,868 rows.
- A planted feature reads NEW-INFO (+0.060 AUC) and the permuted trade tape reads NONE (+0.000).
- The Part A planted feature reaches R^2 0.35; the permuted tape stays at -0.001.
- No NaN score enters an AUC.
- AVOID reproduces MOVE's 0.8707, and ops reproduces AVOID's 0.8410.

Statuses:
- First-minute legs: 800,186 WS-timed and 773,104 E2-timed. E2 legs are excluded from Part A, because their anchors are uncertain by
  +-250 ms.
- 187,840 WS legs were not at the best.
- 468,596 fill events (taker order x price), 366,307 of them leaving the level standing.
- Decision rows: 591,868; 29,910 without a known guard value; 302 without a best.

## Part A: what moves the price after a fill at the best

Out-of-day R^2 of the 250 ms move after the fill, and the share of the top-5 vs field gap each family removes:

| Family | R^2, all fills | R^2, level left standing | Share of the top-5 gap removed (all fills) |
|---|---|---|---|
| Binance (BN) | 0.34 | 0.20 | 3% |
| Other venues (VEN, 8 days) | 0.38 | 0.22 | 2% |
| Later Predict takes (TAPE) | 0.25 | 0.14 | 12% |
| The book's reaction (BOOK) | 0.64 | 0.89 | 57% |
| All four | 0.75 | 0.89 | 56% |

The move itself, c, share-weighted:

| Population | Mean move at 250 ms | Top 5 | Field |
|---|---|---|---|
| All at-best fills | -0.32 | -0.13 | -0.34 |
| Fills leaving the level standing | -0.02 | -0.001 | +0.005 |

The 1 s figures are the same: -0.13 against -0.33 for all fills.

Readings:
- **R370 EXPLAIN = BOOK (declared). The label is close to circular.** The move is the change of the mid, and the BOOK family describes
  the best prices changing (the level gone, the opposite best moving). The informative part is that BTC explains a third of the move
  (Binance 0.34, the other venues 0.38) and later Predict takes a quarter.
- **R371 GAP, declared on standing fills: NOT INFORMATIVE.** There is no top-5 gap in standing fills (-0.005 c), so the shares
  (739%, 718%...) divide by about zero.
- **On all fills (printed beside the readings):**
  - The top 5's edge (+0.21 c at 250 ms) is not BTC: Binance and the other venues remove 2-3% of it.
  - Most of it, 57%, goes with the book's reaction: the top 5's fills are less often followed by the level being taken.
  - Their edge is not being on the right side of BTC moves; they are less often in the trades that clear the level (cf. the 2026-11-10
    strategy review §34, 8 public days: 84% of their at-best volume is in trades that leave the level standing, against 73% for
    the rest; the same review finds them first in the queue no more often than their volume share, 30.7% of front shares against
    32.0% of volume). Citation corrected 2026-11-13; it previously named B12 §3.1.

## Part B: what predicts the level going, and is any information missing?

Decision rows every 500 ms of the first minute, per side.

Target **DN_250**: the side's best moves worse than the known best within (t + 24, t + 274] ms. The window starts at t + 24 because
decision 2 ms + cancel 22 ms is the earliest a pull can take effect. Base rate 11.8%.

Out-of-day AUC, the guard's own inputs (BASE: its 32 features + v + S4's 4 extras + side) against BASE plus each new source:

| Model | AUC | Difference | Days above |
|---|---|---|---|
| BASE | 0.8707 | | |
| + Predict trade tape (incl. taker reputation) | 0.8706 | -0.0001 | 3 of 9 |
| + book dynamics (cancels / adds, level age, depth 2) | 0.8721 | +0.0014 | 9 of 9 |
| + both | 0.8717 | +0.0010 | 8 of 9 |
| + other venues (8 days, vs BASE on them 0.8679) | 0.8684 | +0.0005 | 8 of 8 |
| + all (8 days) | 0.8695 | +0.0016 | 8 of 8 |

Readings:
- **R372 MISSING-INFO = NONE for every source.** Nothing reaches the +0.003 line.
  - Book dynamics add a consistent but tiny +0.001.
  - The other venues add nothing beyond Binance.
- **R373 TRACKS-HARM (9 of 9 days).** Fitted on the right event, the same inputs predict the level going at 0.87, but a trade arriving
  at that price (activity) only at 0.60.
  - FA's post hoc reading was that the guard's value tracks activity more than harm. That is a property of its target, not of its
    inputs.
- Other targets (BASE / + tape + book):

| Target | BASE | + tape + book |
|---|---|---|
| Cleared by trades (CLR_250) | 0.870 | 0.870 |
| Activity (ACT_250) | 0.600 | 0.600 |
| The level goes within 1 s (DN_1000) | 0.770 | 0.770 |

**AVOID (post hoc): how much of it can a pull avoid?**
- **29.4% of DN_250 is unavoidable at today's speed.** The level was already gone in the last frame stamped <= t + 24, so it moved
  while unseen, before a cancel could land. That part is the speed lever.
- **The avoidable rest is still well predicted.** Rows where the level is still there at t + 24 and goes within the window: 8.7% of
  those rows.
  - AUC 0.841 (0.816-0.859 by day) from the DN fit; 0.841 when fitted on it directly.
- What carries the prediction (share of gain):

| Input | Share of gain |
|---|---|
| The guard's own value v | 53% |
| Binance book-imbalance change over 250 ms | 10% |
| Side | 8% |
| Our best level's size change over 250 ms | 7% |
| Others | 2% or less each |

**ops (post hoc): the false-alarm trade-off for an order at the front.** Rows still present at t + 24:

| Avoidable level losses caught | Moments pulled | Precision | False alarms |
|---|---|---|---|
| 50% | 10.8% | 40.0% | 60.0% |
| 70% | 22.6% | 26.9% | 73.1% |
| 90% | 50.4% | 15.5% | 84.5% |

Among moments when a WS-timed trade hits the level (133,918; 13.8% of them clear it), the model separates "this trade will take the
whole level" from "the level stays" at AUC 0.822 (0.769-0.849 by day).

**feeds (post hoc, 2026-11-13; USER: "so do we take actions based on predict's feed").** The AV fit with the inputs split by the live feed
they read (`pf_move_avoid.py feeds`, 50227d6e, committed before its run; control: the full set reproduces 0.8410):

| Inputs | AUC on AV |
|---|---|
| Binance only (returns, imbalance and its change, flow, activity, volatility) | 0.762 |
| Predict's book only (spread, imbalance, level sizes and their 250 ms change, pending, book age) | 0.767 |
| Both, without v | 0.841 |
| v alone | 0.779 |
| Time and side only | 0.495 |

Each feed adds about +0.08 to the other; v adds nothing once both feeds are in (-0.0003). A pull needs both feeds. By day the
Predict-only fit is steadier (0.747-0.774) than the Binance-only one (0.701-0.804).

These alarm rates are not directly comparable with FA's (94% of alarms false, guard on 90% of moments). FA judged a back-of-queue
order's own fills over 500 ms; this judges the level within 250 ms.

## What it means for the two issues

1. **No missing information was found in any public feed we have.** Predict's trade tape with taker reputation, the book's cancels and
   adds, and Coinbase / OKX / Bybit add nothing on top of the guard's inputs. Whatever else the top makers use (B12c's "other
   information") is not in these feeds, or it is speed.
2. **The prediction exists; the guard aims it at the wrong decision.**
   - The guard's own value v predicts that the level will be taken 24-274 ms ahead at AUC 0.84, for the part a pull can still avoid.
   - Among trades that do arrive, it tells clearing from standing at 0.82.
   - As v3 uses it, it is a high entry line: the guard says stay out ~90% of the time, and we join at the back.
   - For an order already at the front, the same information supports a different rule: stay by default, pull only when the level
     is about to go. Pulling in 11% of moments catches half of the avoidable losses; in 23%, 70%.
3. **This is the top makers' edge in our data.** Their fills are less often in trades that take the level; BTC moves hit them and the
   field alike. Staying at the front and leaving before the level is taken is what a "front order + level-going pull" would do.
4. **Speed matters for the 29% that is gone before a cancel can land.** That is the measurable value of a faster feed and order path
   for pulls.

## Next (proposals, the user's call)

- **FRONT-PULL policy test.** An order that rests at the front by default:
  - entry at level birth, or IM95's improve, or after TB's limits-off base;
  - pull when P(level goes in 250 ms) crosses a line;
  - re-enter by a declared rule.
- **Valuation:** fill value with the age split and the 250 ms clock. Pulling costs the queue place, so value KEEP vs PULL as the other
  session's queue-priority objective does.
- **Feasible without editing the engine:**
  - a predictions file with v = -P for each side;
  - per-minute lines from a lines file (PF_LINES_FILE);
  - the frozen engine with B0 / IM95.
- **Calibration first:** the replay's fill model must be checked against live (A4b), because a front order's value depends on its
  real queue place.

## Limits

- Consumed days, first minute, descriptive. AVOID and ops are post hoc.
- The venue delays are assumptions. Exchange clocks are not synchronised with Predict's (under ~10 ms not resolved).
- Taker reputation uses first-minute trades only.
- The relay (Predict's reference price) is not used: one print per second, ~1.3 s after publication.
- AUC is not money: the pull's cost (losing the front) and the replay's fill model decide the value.

Outputs: `data/predict_fun/research/move_20261112/` (move.json / .txt, avoid.json / .txt, avoid_ops.json / .txt, day_*.npz / .json,
avoid_*.npy, logs); copies (without npz / npy) in `reviews/move_20261112/`.
