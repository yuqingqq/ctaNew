# Research plan — target net Sharpe ≥ 3.5 at $250k–$1M

Goal set 2026-08-07. Capital band **$250k–$1M**. **Not constrained to the existing book.**

## Honest starting position

3.5 net is above every properly-costed number in the literature (best: **2.28**, US equities, 24 years — and
reported with no confidence interval). Our own honest live forecast for the current book is **+0.3 to +0.9**.
So the target is ~4-10× what we have and ~1.5× the published frontier. Pretending otherwise would waste the
next month.

**But the published frontier describes institutional-capacity strategies, and that is the one genuine reason
to think this repo's prior conclusions were too pessimistic for this capital band.**

## The insight that reopens the search

Every cost number in this session used `cost_10k` — modelled slippage for **$10,000 clips**. At $250k–$1M
running a 10% vol target, positions are **$1.8k–$18k**. At the lower end we charged ourselves ~5× the right
cost. That is not a rescale, it is a change in the opportunity set, because this repo **repeatedly found real
edges and then dismissed them for capacity**:

| dismissed finding | recorded reason | binding at $250k–$1M? |
|---|---|---|
| Amihud illiquidity premium, both-era (+0.018/+0.016 @5d) | "capacity-walled: both-era at depth ≥$100k, dies ≥$500k, collapses ≥$2M, lives in thin names, un-harvestable" | **No** |
| the long tail of 176 perps | excluded by a $3M/day liquidity floor and the top-40 ADV filter, adopted because tail cost was 24 bps | **No** |
| 5–15 min order-book lead | "real but sub-cost" | **Recheck at the true clip** |
| positioning / OI signals | "real both-era signal but SUB-COST" | **Recheck at the true clip** |

The big-names restriction — the single biggest lever in the whole cost loop — was adopted **specifically to
solve a cost problem that mostly does not exist at this size.**

## What would have to be true (IR = IC × √breadth)

Current: IC ≈ 0.021–0.030, ~40 names, 4h grid → realised gross ≈ 1.55.
To reach net 3.5 we need gross ≈ 4.0–4.5, i.e. **~3× the current information ratio.** Routes:

| route | multiplier | how |
|---|---|---|
| breadth via universe | ×1.6–2.1 | 40 → 100–176 names, if thin names are tradeable at $2k clips |
| breadth via frequency | ×2 per 4× | 4h → 1h grid, if cost at small clips permits |
| IC | ×N directly | needs genuinely better prediction — the hardest and least likely |
| cost reduction | additive | already measured: taker → passive is worth ~+0.25 |

**Universe × frequency alone is a ×3–4 breadth multiplier = ×1.7–2.0 on IR** — which gets to ~2.6–3.1 gross,
not 4.5. So even the optimistic path needs an IC improvement too. **Stating this now so we can recognise a
dead end early rather than after another 200 configurations.**

## Directions, in priority order

| # | direction | why it is not already closed | first test |
|---|---|---|---|
| C1 | **Re-cost everything at the true clip size** | every prior conclusion used a 5× overcharge | measure the realised cost curve from aggTrades sweeps (`sc_cost_curve.py`) |
| C2 | **Reopen the thin-name universe** | excluded for a cost reason that does not bind here | rerun the incumbent on the full 176 at measured small-clip cost; compare to top-40 |
| C3 | **Reopen the capacity-walled Amihud premium** | dismissed at ≥$500k depth; we trade $2k | both-era + hard-split at the correct clip |
| C4 | **Higher frequency (1h grid)** | 4× breadth; previously blocked by cost | rebuild the panel at 1h, measure IC and net |
| C5 | **Event-driven: new listings** | small-capacity by nature, never tested here | listing-date event study from first-data-date |

## Discipline (inherited, plus two new rules)

All prior rules hold: both eras, day-clustered/block CIs, paired deltas with CI on the delta, chronological
hard split, pre-registered gates and falsifier.

**New rule 1 — every result must be costed at the ACTUAL clip size**, derived from the measured curve, not
from `cost_10k`. Any number quoted at the wrong clip is void.

**New rule 2 — every result must state its capacity in dollars.** "Sharpe 3.5" is meaningless without "up to
$X". A strategy that works at $250k and dies at $1M is still useful here, but only if the wall is stated.

## C1 RESULT — **the premise holds. We over-charged by 3-5× at this capital scale.**
(`live/sc_cost_curve.py`, 31 symbols × 113 sampled days, ~520M taker sweeps)

Realised slippage measured from aggTrades sweeps (consecutive same-side fills within 50 ms = one taker order
walking the book), then combined with each symbol's own measured half-spread from `bx_iter1_markout.py`:

| clip | walk (pooled) | avg levels walked |
|---|---|---|
| $1-2.5k | 0.192 bps | 1.74 |
| $2.5-5k | 0.284 | 2.19 |
| $5-10k | 0.370 | 2.75 |
| $10-25k | 0.478 | 3.74 |
| >$100k | 0.907 | 19.48 |

**True total cost at a $1-2.5k clip = own half-spread + own walk:**

| liquidity quartile | total @$2k | `cost_10k` charged | overcharge |
|---|---|---|---|
| top | **1.20 bps** | 5.77 | **4.8×** |
| middle half | **2.41 bps** | 7.45 | **3.1×** |
| bottom | **3.25 bps** | 9.35 | **2.9×** |

Per name: BTC 0.42, ETH 0.67, BNB 0.86, SOL 1.11, ADA 1.91, ATOM 2.10, HBAR 2.45, LDO 3.23, ICP 3.38,
RUNE 3.75, GMX 4.68.

### Two corrections I had to make to my own measurement
1. **The walk is measured from the first fill (the touch), so it excludes the half-spread.** My first reading
   claimed a 34-46× overcharge; adding each symbol's own half-spread gives the true figure of **~3-5×**.
   The spread, not the walk, is the dominant cost at small clips — and spread is size-independent, so small
   size avoids depth impact but not spread.
2. **Selection bias, stated not buried:** we observe only sweeps that were *sent*. Traders size to available
   liquidity, so each notional bucket contains orders placed when conditions suited. These are lower bounds.

### What it is and is not worth
- **Re-costing the EXISTING book**: cost drag falls from ~0.50 to ~0.13 Sharpe → roughly **+1.26 → +1.40**.
  Real, but nowhere near 3.5. This is not the prize.
- **The prize is what it unlocks.** The top-40 restriction and the $3M/day liquidity floor were adopted
  *specifically* to solve a cost problem that is 3-5× smaller at this size. Removing them is a **×4.4 breadth
  multiplier ≈ ×2.1 on IR** — the largest single lever toward the target, now justified by measurement.
- **Limit on the claim:** all 31 measured symbols have ≥1100 days of aggTrades, i.e. established names. The
  genuinely thin tail below the $3M floor was NOT measured and must not be assumed cheap.

## C2 RESULT — **breadth does NOT convert. The lever C1 was supposed to unlock is not there.**
(`live/sc_c2_universe.py`, measured per-symbol cost, hard split)

| universe | names/bar | HOLDOUT gross | HOLDOUT net | net CI |
|---|---|---|---|---|
| 20 | 20 | +1.23 | +0.97 | [−0.93,+2.92] |
| **40** *(incumbent)* | 40 | +1.55 | **+1.17** | [−0.54,+2.75] |
| 80 | 74 | +0.62 | +0.14 | [−1.76,+1.86] |
| **all** | **139** | +1.57 | **+1.03** | [−0.92,+2.83] |

Paired Δ vs top-40, held out: N=20 −0.19, N=80 −1.03, N=all −0.13 — **every one negative, all spanning zero.**

**Realised IR multiplier from 40 → 139 names: 0.88×. Theory from ×4.4 breadth said 2.1×.**

**Why, and it is the same wall as before.** The fundamental law's √breadth assumes *independent* bets. This
repo already measured that ours are not: realised IR implies ~6,400 effective bets against ~385,000 nominal
name-bars, i.e. ~1.7%. Adding 100 more names adds more of the *same one factor*, not new information. C1's
cost finding is real and correct; it simply does not buy what I hoped, because cost was not what the universe
restriction was ultimately costing us.

### Consequence for the target — stated plainly
The plan's own arithmetic identified two breadth routes: universe (×1.6-2.1) and frequency (×2), together
reaching ~2.6-3.1 gross, still short of the ~4.0-4.5 needed. **Universe has now delivered ×0.88, not ×1.6-2.1.**
And the prior on frequency (C4) is now materially worse: if 4× more *names* does not add independent bets,
4× more *time-steps* is less likely to, since consecutive bars are more correlated than different names.

**The breadth route to 3.5 is substantially closed.** What remains is C5 (new-listing events) — the only
direction left that could raise **IC** through a different mechanism rather than multiply breadth.

## Status
- **C1 DONE** — cost overcharged 3-5× at this scale. Worth ~+0.14 Sharpe on the existing book (+1.26 → ~+1.40).
- **C2 DONE — NULL.** Universe breadth does not convert (0.88× vs 2.1× theory).
- **C3** (Amihud at correct clip) — prior now weak: it is another cross-sectional characteristic on the same
  one factor, and C2 says adding names to that factor does not help.
- **C4** (1h frequency) — prior downgraded by C2's result, as above.
- **C5** (new-listing events) — **now the highest-prior remaining direction**, because it is a different
  mechanism (supply/attention shock) rather than another slice of the same factor, and it is small-capacity
  by nature, which suits this capital band.
