# QUEUE_AND_TYPE — C1 / C2 results

Protocol `queue_type_v1`. Probe `queue_and_type.py` (34 self-test checks,
including controls that make each verdict falsifiable). Research only, not
decision eligible, no forward-day claim.

Source: `clob_v3_1` covered set, 24 windows/coin (C1), 8 windows/coin (C2),
unified Up book, 250 ms knowledge lag, gap-killed. Per coin throughout; nothing
pooled. Two days, so window-clustered intervals only and they understate
uncertainty.

---

## C1 — cancellations do NOT narrow the bracket, and the reason is not the one
## the rule anticipated

**Verdict: `UNIDENTIFIABLE`** — a branch the protocol does not contain. See
revisions R1 and R2 below; I have not re-cut the rules, and the literal reading
of each is recorded before my substitute.

### Reconciliation, reported first as required

| coin | residual / gross turnover | residual / traded volume |
|---|---:|---:|
| BNB | 0.00000 | 0.0000 |
| BTC | 0.00024 | **0.0234** |
| DOGE | 0.00004 | **0.0310** |
| ETH | 0.00011 | **0.0251** |
| HYPE | 0.00006 | **0.0437** |
| SOL | 0.00049 | **0.1207** |
| XRP | 0.00006 | **0.0233** |

Against the protocol's stated denominator — gross level turnover — every coin
passes 1 % by two orders of magnitude. **That pass is close to meaningless**,
because gross level churn is 60–1000× trade volume, so any trade-sized residual
is invisible against it.

Against traded volume, the residual is **2.3 %–12.1 %**: that share of observed
trade volume has **no matching decrease** in the displayed book at that level and
price. On SOL it is one share in eight. Consistent with hidden liquidity, with
sequencing error, or with both; this tape cannot separate them, and I am not
narrating which.

### The measurement degenerated

| coin | cancel saturation p50 | share ≥ 1 | front | back-displayed | cancel-credited |
|---|---:|---:|---:|---:|---:|
| BNB | 3.2 | 0.94 | 0.636 | 0.099 | 0.630 |
| BTC | 10.8 | **0.99** | 0.946 | 0.769 | **0.946** |
| DOGE | 3.0 | 0.95 | 0.714 | 0.096 | 0.705 |
| ETH | 13.2 | **0.99** | 0.848 | 0.556 | **0.848** |
| HYPE | 2.0 | 0.86 | 0.713 | 0.024 | 0.678 |
| SOL | 3.6 | 0.94 | 0.733 | 0.288 | 0.727 |
| XRP | 5.0 | 0.97 | 0.809 | 0.207 | 0.806 |

`cancel saturation` is cancelled-at-level divided by the initial queue ahead.
**86–99 % of actions are saturated at ≥ 1**, so the credit is capped in nearly
every action and the credited bound **collapses onto the optimistic FRONT
bound** — on BTC and ETH to three decimal places.

Bracket width therefore falls by ~97–100 %, which would trip the `MATERIAL`
branch. **That reading would be wrong.** The bound did not tighten; it
degenerated into the other bound.

### What this actually establishes

Cancellation *volume* at a level is abundant — far more than enough to clear any
queue we join. What displayed L2 does not give is cancellation **position**:
whether a departing order was ahead of us or behind us. Crediting all of it
reproduces `FRONT`; crediting none reproduces `BACK_DISPLAYED`. **The interior is
reachable only under an assumption, not a bound.**

So the bracket width *is* the queue-position ambiguity, restated. Cancellation
data does not reduce it, because the missing quantity in both cases is the same
one.

This is close in consequence to the protocol's `IMMATERIAL` branch — fill is not
determinable from data we can collect — but **not for the stated reason**.
`IMMATERIAL` says displayed depth ahead really does trade through. It does not:
it overwhelmingly churns. We simply cannot tell whose.

### R1 — the reconciliation identity as written is a tautology

`size(t+) − size(t−) = new_or_replenished − traded − cancelled` holds **by
construction** whenever `cancelled` is defined as the unexplained residual,
which is the only way it can be computed here. As literally specified the check
cannot fail.

Implemented instead: `traded` is anchored to the **independent**
`last_trade_price` stream, and the residual reported is trade volume the book
never showed being consumed. That can fail, and at 2.3–12.1 % of traded volume
it is doing real work. **The denominator should also be traded volume, not gross
turnover** — against gross churn the check is unfalsifiable.

### R2 — `MATERIAL` cannot distinguish tightening from degeneration

The rule keys on bracket-width narrowing alone. A credited bound that saturates
onto the front bound produces ~100 % narrowing while carrying **less**
information, not more. A saturation guard is required: the credited bound is
only a bound where `cancelled_at_level < queue_ahead`. It is not, in 86–99 % of
actions.

---

## C2 — market self-excitation SURVIVES. Do not delete the layer.

**Verdict: `RETAIN` on eth / sol / xrp / hype / doge / bnb. `CENSORED` on btc.**

Bivariate Hawkes on `{MICRO_002, MARKET}`, 8 windows/coin, baseline operational
time, per coin. `alpha_ij` is the branching ratio from type `j` to type `i`.

| coin | half-life | censored | market←market | market←micro | micro←market | micro←micro | CI(market←market) | n_mkt | n_mic |
|---|---:|---|---:|---:|---:|---:|---|---:|---:|
| BNB | 0.0625 | no | 0.180 | 0.020 | 0.050 | 0.350 | [0.080, 0.250] | 261 | 930 |
| BTC | **0.03** | **YES** | 0.180 | 0.050 | 0.020 | 0.000 | [0.180, 0.180] | 17,507 | 324 |
| DOGE | 0.0625 | no | 0.180 | 0.080 | 0.120 | 0.180 | [0.120, 0.250] | 420 | 793 |
| ETH | 0.25 | no | **0.450** | 0.080 | 0.050 | 0.350 | [0.350, 0.450] | 3,166 | 1,020 |
| HYPE | 0.0625 | no | 0.350 | 0.020 | 0.020 | 0.250 | [0.120, 0.450] | 102 | 806 |
| SOL | 0.25 | no | 0.350 | 0.180 | 0.050 | 0.250 | [0.350, 0.450] | 982 | 479 |
| XRP | 0.125 | no | 0.350 | 0.050 | 0.080 | 0.180 | [0.250, 0.350] | 969 | 1,507 |

### The hypothesis that motivated this test is REFUTED

The suspicion was that the scalar branching of 0.40–0.55 was really
micro↔market cross-excitation, given A1 failed bidirectionally at ~2× within
0.25 s. It is not. **The diagonal dominates the off-diagonal on every coin**:
`market←market` runs 0.18–0.45 while the cross terms sit at 0.02–0.18.

Market self-excitation survives being modelled alongside the micro actor rather
than deleting it. `DELETE_HAWKES_LAYER` does **not** fire; the layer earns its
place.

A1 is not contradicted — the cross terms are non-zero — they are simply smaller
than the self terms. And the micro actor is itself strongly self-exciting
(`micro←micro` 0.18–0.35 on five coins), which is consistent with a clustered
single-participant cadence and is separate from market flow.

### BTC is censored and must not be read

BTC selects the grid **floor** (0.03 operational half-life), so per protocol its
branching value is unresolved. A grid reaching below 0.03 is needed. Note BTC
carries the most market events by far (17,507) and the fewest micro (324, 1.8 %,
matching its 2.0 % share) — so the coin with the most power is the one whose
timescale the grid cannot reach.

### Limitation that bounds every CI above

The intervals are **grid-quantised and conditioned on the selected half-life**.
Bootstrap draws land only on coordinate-descent grid points (0.08, 0.12, 0.18,
0.25, 0.35, 0.45), and BTC's degenerate `[0.180, 0.180]` is proof of it — every
resample returned the identical grid point. These intervals show **fit stability
across window resamples**, not sampling uncertainty, and they understate it.

So `RETAIN` should be read as *the layer is not deletable on this evidence*,
which is what the test was for, rather than as a validated branching estimate. A
continuous optimiser over `alpha` and a joint `beta` are needed before any
branching number is quoted.

---

## Excluded set, reported beside the retained

C1 actions unavailable through collector gaps or in-horizon tick changes are
counted per coin in the JSON receipt (`n_actions_unavailable`), not dropped
silently. HYPE is reported but excluded from the verdict by protocol, at 90 %
single-actor share.

Receipts: `data/pm_5min/derived/queue_c1_cancellation_v1.json`,
`queue_c2_bivariate_v1.json` (both gitignored).
