# QUEUE_AND_TYPE — C1 / C2 results

> **⚠ For current state read [`FLOW_MODEL_STATE.md`](FLOW_MODEL_STATE.md).** This
> document is **provenance** — correct about its own moment, not a statement of
> current belief. Where it conflicts with `FLOW_MODEL_STATE.md`, that page wins.


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

## C2 REFIT 2026-08-22 — instrument floor + continuous optimiser

The original C2 fit was grid-quantised twice (alpha grid and half-life grid) and
reported btc `CENSORED` at the old floor with a degenerate `[0.180, 0.180]`
interval. Both mechanisms from the scalar fit are now ported in:
`hawkes_floor_operational` (ten venue ticks of wall clock, converted per coin
into operational units) and a Nelder-Mead refinement over the four alphas and
the half-life jointly.

**Verdict: `RETAIN` — unchanged, now on far better evidence.**

| coin | m←m | CI95 m←m | m←micro | micro←m | micro←micro | HL op | HL wall | censored |
|---|---:|---|---:|---:|---:|---:|---:|---|
| btc | **0.477** | [0.282, 0.519] | 0.010 | 0.008 | 0.000 | 0.5408 | **75.3 ms** | no |
| hype | 0.400 | [0.267, 0.545] | 0.004 | 0.000 | 0.313 | 0.1462 | 351.8 ms | no |
| eth | 0.389 | [0.339, 0.431] | 0.120 | 0.068 | 0.337 | 0.2257 | 146.3 ms | no |
| xrp | 0.317 | [0.292, 0.350] | 0.067 | 0.108 | 0.203 | 0.1174 | 139.6 ms | no |
| sol | 0.291 | [0.245, 0.338] | 0.167 | 0.067 | 0.218 | 0.1528 | 246.3 ms | no |
| doge | 0.247 | [0.207, 0.289] | 0.082 | 0.096 | 0.273 | 0.0795 | 189.4 ms | no |
| bnb | 0.236 | [0.162, 0.311] | 0.025 | 0.049 | 0.409 | 0.0875 | 218.5 ms | no |

**No coin is censored, btc included.** The floor excluded exactly two grid points
on btc (0.03, 0.0625) — the same two the scalar fit excluded — and the fit then
seeded at 0.50 and refined to 0.5408. Continuous refinement fired on all seven,
so no interval is grid-confined.

**btc moved 0.180 → 0.477**, a 2.65× increase, and its interval went from the
degenerate `[0.180, 0.180]` to `[0.282, 0.519]`. That degenerate interval was the
signature of the defect: bootstrap draws could only land on grid points, so it
reported fit stability rather than sampling uncertainty.

**Independent agreement with the scalar fit.** btc half-life is **75.3 ms**
bivariate against **80.8 ms** scalar — two different estimators, two different
likelihoods, converging on the same timescale. Every coin lands at 75–352 ms,
i.e. **75× to 352× the venue's millisecond tick**, comfortably resolvable.

**The diagonal still dominates the off-diagonal on every coin**, so the finding
C2 was opened for survives: market self-excitation is real once the micro actor
is modelled as a type rather than deleted, and it is not cross-excitation wearing
a self-excitation label. Verdict inputs under the unchanged protocol rule — btc
`[0.282, 0.519]` and eth `[0.339, 0.431]` both exclude zero and both exceed 0.10,
so `DELETE_HAWKES_LAYER` does not fire and `RETAIN` stands.

**Scope, unchanged:** 24 windows/coin, `clob_v3_1`, two days. Window-clustered
bootstrap at n=100 per coin; day-level common factors are not captured, so these
intervals still **understate** uncertainty. `RETAIN` means not-deletable on this
evidence; forward generalization still needs ~10 days in one era.

**Two engineering notes worth keeping.** The first refit attempt built all 168
windows before fitting and was killed silently — no traceback, empty output,
yesterday's file left in place, which is exactly the failure mode that looks like
success. Windows are now built and discarded one coin at a time. And the
refinement initially returned `None` on every coin because acceptance was gated
on scipy's `success` flag: in five dimensions Nelder-Mead routinely hits maxiter
with the improvement already found (measured: `nit=80, success=False` and
`nit=94, success=True` return the identical `+4.1258` gain). Acceptance is now on
the likelihood, which is the real guard.
