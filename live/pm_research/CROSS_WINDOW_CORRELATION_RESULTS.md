# B6 — cross-window same-coin correlation

DA measurement, 2026-08-23, dispatch B6. Probe
`cross_window_correlation.py` (11 selftest checks green), receipt
`data/pm_5min/derived/cross_window_correlation_v1.json`.

**Consumed by DE**, to fill the `SHARED_RISK` edge of the coupling graph and to
decide whether `DE-Allocator` starts as accounting or as a portfolio problem.
It also addresses falsifier #2 in `DA_INVENTORY_STATE_PLAN`.

> **This document contains no decision rule, no threshold and no verdict**, by
> instruction. Where the numbers imply a rule, the rule is surfaced to the
> coordinator in §6 rather than written here or into any plan. Under R-6 the
> correlation is a **Class-C measured** quantity: it is published and adopted,
> not chosen.
>
> **SP set:** `SP_PLANE_PLAN.md` §5, operative (user-ratified 2026-08-23). The
> only parameter binding this measurement is `quote_size_pin = 5 shares`
> (Class B); see §5 for what a change to it would invalidate.

---

## 0. What was measured, and on which population

Three blocks, **three different denominators**. Reading a figure against the
wrong one is the defect this programme has already paid for six times.

| block | source | population | era |
|---|---|---|---|
| concurrency | discovery grid | all markets, all days | era-independent |
| residual (outcome) | settlement facts | every resolved window | era-independent |
| inventory + joint loss | CLOB tape, standard replay | contiguous run per (coin, day) | **`clob_v3_1` only** |

---

## 1. The structural facts — these are exact, not estimated

**Adjacent windows share their settlement reference point exactly.**
`window_end_k == window_start_{k+1}` in **7,080 of 7,080** adjacent pairs, with
7 holes in the grid. Settlement is `S60(T) ≥ S60(t0)`, so

```
outcome_k    = 1{ S60(ws_k + 300) >= S60(ws_k)     }
outcome_k+1  = 1{ S60(ws_k + 600) >= S60(ws_k+300) }
                       ^^^^^^^^^^^^^^^^ the SAME price, entering with OPPOSITE sign
```

This is a coupling channel that exists **independently of any correlation in the
underlying**, and DE should model it as structure rather than as an estimate. It
also carries a testable prediction: `S60` is a 60-second TWAP, i.e. a smoothed
series, so noise in the shared endpoint pushes the two outcomes in opposite
directions and should induce a **negative** lag-1 correlation. §3 reports what
was actually seen.

**Two same-coin windows are open essentially all the time.** Markets are
discovered a median **283.5 s** before their window starts (p10 271.4, p90
296.5), which is almost exactly one window ahead:

| coin | time at 1 open | **time at 2 open** | time at ≥3 |
|---|---:|---:|---:|
| btc | 0.058 | **0.942** | 0.000 |
| eth | 0.059 | **0.941** | 0.000 |
| sol | 0.059 | **0.941** | 0.000 |
| xrp | 0.059 | **0.941** | 0.000 |
| bnb | 0.060 | **0.940** | 0.000 |
| doge | 0.060 | **0.940** | 0.000 |
| hype | 0.061 | **0.939** | 0.000 |

**94 % of quotable time has exactly two same-coin windows open, and three never
happens.** So the `SHARED_RISK` edge is not a corner case to be bounded — it is
the normal operating state, and it is strictly **pairwise**. The Allocator never
has to reason about three concurrent same-coin markets.

*Denominator warning, and it is the reason this table is time-weighted:* counted
as **intervals** instead, the same data reports a median of 1 and a p90 of 2,
because transitions are short and numerous. That reading would have made a
permanent condition look occasional. Both are in the receipt; the time-weighted
one is the decision-relevant one.

*Bound:* the interval used is `[our discovery, window_end]`. The venue creates a
market before we discover it, so 94 % is a **lower bound**.

---

## 2. A scope limit that has to be stated before any inventory number is read

The standard two-sided replay (`inventory_walk.simulate_window`, imported
unmodified) exposes `t ∈ [−60, +300] s` around each window. The real overlap is
~283 s. **So the replay observes 21.2 % of the concurrency that §1 measures**,
and every inventory correlation below is measured on that fifth of the exposure,
not on the whole of it.

This is a property of the harness's exposure convention, not of the venue.
Extending it is a change to a DE-owned file and is **not** made here; it is
raised in §6.

---

## 3. Residual correlation — settlement facts, full population, no replay

Estimand: `corr(outcome_k − 0.5, outcome_{k+lag} − 0.5)`, per coin, pairs
separated by exactly `lag` windows on the 300 s grid. Grid holes are skipped,
never bridged — pairing across a missing window would relabel a 10-minute
separation as 5.

Retained **7,065 windows** (btc/eth/sol/xrp 1,014–1,015 each, bnb/doge/hype
1,002 each); excluded **29** as unresolved. Five day clusters.

**Pooled across coins** (pooling is reported because the day-cluster bootstrap
resamples whole days across all coins together, so contemporaneous coins do not
enter as independent):

| lag | separation | n pairs | r | day-clustered CI95 |
|---|---|---:|---:|---|
| 1 | 5 min | 7,058 | **+0.0223** | [+0.0031, +0.0496] |
| 2 | 10 min | 7,051 | −0.0399 | [−0.0742, +0.0154] |
| 3 | 15 min | 7,044 | +0.0138 | [−0.0355, +0.0646] |
| 6 | 30 min | 7,023 | +0.0369 | [−0.0044, +0.0661] |
| 12 | 60 min | 6,981 | +0.0229 | [+0.0024, +0.0533] |

**Per coin, lag 1:**

| coin | n | r | day-clustered CI95 |
|---|---:|---:|---|
| bnb | 1,001 | +0.0519 | [+0.0281, +0.1021] |
| sol | 1,014 | +0.0457 | [−0.0049, +0.0873] |
| btc | 1,014 | +0.0203 | [−0.0300, +0.0649] |
| hype | 1,001 | +0.0190 | [−0.0202, +0.0748] |
| xrp | 1,014 | +0.0168 | [−0.0625, +0.0724] |
| doge | 1,001 | +0.0044 | [−0.0472, +0.0793] |
| eth | 1,013 | −0.0053 | [−0.0735, +0.0579] |

**Read the effect size, not the interval.** The largest per-coin lag-1
correlation is bnb at +0.052, which is **0.27 % of variance**; the verdict coins
are +0.020 (btc) and −0.005 (eth). Two pooled lags have intervals excluding
zero, but the signs alternate across the lag ladder (+, −, +, +, +) and only one
of seven coins excludes zero at lag 1 — out of 35 per-coin lag tests, where one
or two exclusions are expected at 5 % by chance.

**The mechanism prediction in §1 is not confirmed.** The shared settlement
endpoint plus TWAP smoothing predicts a *negative* lag-1 correlation; the
measured pooled lag-1 is slightly *positive*, and the one clearly negative
figure is at lag 2, where no endpoint is shared. Neither is large enough to
support a mechanism story either way, and none is offered.

---

## 4. Inventory and joint loss — standard replay, era-restricted

Retained **158 windows per coin** (1,106 total) from contiguous runs inside
`clob_v3_1`. Excluded: **3,703** beyond the per-(coin, day) cap of 40, **15**
replayed but unresolved, **0** where the replay returned nothing.

**The mark-ambiguity guard found nothing.** `simulate_window` returns
`terminal_mid == 0.5` for a window that never traded, and a real mid can also be
exactly 0.50, so the two are inseparable from the result. The block is computed
on both arms — and **0 rows of 1,106** hit the ambiguous value, so the arms
coincide exactly. The guard was worth adding and the data did not need it.

**Verdict coins (btc, eth), lag 1, ~153 pairs, 4 day clusters:**

| estimand | btc | eth |
|---|---:|---:|
| `terminal_net` | +0.0443 | +0.0655 |
| `cash_at_risk` | −0.0676 | −0.0088 |
| **simultaneous** — net at k's settlement vs net k+1 already holds | **+0.0227** | **−0.0119** |
| settlement residual | +0.1706 | −0.0435 |
| **JOINT LOSS** — `net × (outcome − mark)` | **−0.0621** | **−0.0197** |

**All seven coins, the two decision-relevant estimands:**

| coin | simultaneous net | JOINT LOSS lag 1 |
|---|---:|---:|
| btc | +0.0227 | −0.0621 |
| eth | −0.0119 | −0.0197 |
| sol | +0.0997 | −0.0558 |
| xrp | +0.0710 | +0.0290 |
| doge | +0.0578 | +0.0444 |
| bnb | −0.0191 | −0.1446 |
| hype | +0.1315 | −0.0749 |

**On the verdict coins nothing here is both positive and material.** The
genuinely simultaneous exposure correlates at +0.023 (btc) and −0.012 (eth), and
joint loss is *negative* on both — consecutive-window losses offset slightly
rather than compounding. Five of seven coins have negative joint loss.

**The one figure that stands out does not survive its own coin pair.** btc's
settlement residual +0.171 sign-flips to −0.044 on eth. It is also computed on a
near-degenerate variable: at the terminal mark the outcome is nearly resolved,
so the residual sits near zero with rare large values, and Pearson on that is
outlier-driven. Recorded, not claimed.

---

## 3a. The day-clustered interval is NOT usable at four clusters

This is a property of the measurement, and it changes how §4 must be read.

Across the **201** reported correlations that carry both intervals, the
day-clustered interval is **narrower** than the pair-level one in **142 (71 %)**
— the wrong direction. Clustering should widen an interval, never tighten it.
Split by cluster count:

| day clusters | n | median width ratio, clustered ÷ pair-level |
|---|---:|---:|
| **4** (era-restricted §4) | 161 | **0.71** |
| **5** (era-independent §3) | 40 | **1.02** |

At five clusters the bootstrap behaves. At four it under-covers by ~30 %,
because resampling four blocks with replacement cannot estimate between-day
variation. **So every interval in §4 is decorative, and the block is point
estimates only.** btc's settlement residual is the clearest case: its clustered
CI [+0.118, +0.184] is five times *narrower* than its own pair-level CI
[+0.025, +0.351]. That narrowness is an artifact, not precision.

The standing instruction is to day-cluster where the data allows it. Measured
answer: **it allows it at five clusters and does not at four.** §4 gets more
clusters only from more days in one collector era, which is the same calendar
the rest of the programme is waiting on.

---

## 5. What a parameter change would invalidate (R-6 Class B)

`quote_size_pin` is the only SP parameter this measurement is conditioned on.

| block | survives a `quote_size_pin` change? |
|---|---|
| §1 concurrency | **yes** — discovery grid only, no position enters it |
| §3 residual (outcome) | **yes** — settlement facts only |
| §4 inventory / joint loss | **no** — fills, and therefore every `net` series, are a function of the pin; these results do not carry over and would need a re-run |

`capital_budget`, `κ_$`, `ScenarioLossLimit`, `gamma_ladder` and `refuse_k` do
not enter this measurement at all, and are recorded in the receipt as
non-binding so that a later reader does not infer a dependency that is not
there.

---

## 6. Surfaced to the coordinator — NOT decided here

**The rule these numbers imply, stated so it can be ruled on rather than
absorbed:** *same-coin cross-window coupling may be modelled as independent, and
`DE-Allocator` may start as accounting rather than as a portfolio optimizer.*
**DA does not adopt that and has written it into no plan.** What the measurement
supports, and what it does not:

*Supports it:*

- The coupling is **strictly pairwise** — three concurrent same-coin windows
  never occur in 7,080 markets. Whatever the Allocator must handle, it is a
  two-market problem per coin.
- On both verdict coins the genuinely simultaneous exposure correlates at
  **+0.023 (btc)** and **−0.012 (eth)**, and joint loss is **negative** on both.
  Five of seven coins have negative joint loss. Nothing measured is both
  positive and material.
- Settlement-residual correlation across the full 7,065-window population is at
  most 0.27 % of variance on any coin.

*Does not support it, and these are not small:*

- The replay observes **21.2 %** of the real overlap (§2). The other 79 % is
  unmeasured, and it is the part where both windows are quoted furthest from
  settlement.
- §4 has **no usable intervals** (§3a). Four day clusters cannot support the
  bootstrap, so those are point estimates with 153 pairs behind them.
- The one interval-excluding result, btc's settlement residual +0.171,
  **sign-flips to −0.044 on eth** and is computed on a near-degenerate variable.

**Three items for the coordinator, none actioned by DA:**

1. **The rule above** — adopt, reject, or hold pending more days. If held, the
   thing it waits on is day clusters inside one collector era, which is the same
   calendar everything else is waiting on.
2. **Falsifier #2 in `DA_INVENTORY_STATE_PLAN`** asks whether correlated
   residuals across overlapping windows *dominate* the single-market residual.
   Nothing measured here dominates anything. Whether that retires the falsifier
   or merely fails to trigger it is a call DA has deliberately not made, because
   the measurement covers a fifth of the exposure.
3. **The replay's exposure convention is DE-owned.** `simulate_window` exposes
   `[−60, +300] s`; the real overlap is ~283 s. Extending it would let the
   simultaneous correlation be measured on the whole overlap instead of its
   last fifth. That is a change to `inventory_walk.py`, which is DE's file, and
   it would invalidate every result conditioned on the current convention —
   flagged, not made.

**One cross-plane note, offered as a question rather than a finding.** The
under-coverage in §3a is a property of *four clusters*, not of this probe. Any
day-clustered interval anywhere in the programme computed over four or five
clusters may have the same problem. DA has not inspected another plane's
estimator and is not asserting that it does — but the diagnostic is cheap
(compare the clustered width against the unclustered one; clustered should never
be narrower) and it is worth running wherever a day-clustered CI is load-bearing.
