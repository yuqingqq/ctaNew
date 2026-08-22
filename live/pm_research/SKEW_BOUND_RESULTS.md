# Bounding the 15× skew claim, and decomposing its fill increase

**Status: DEVELOPMENT · decision eligible: no.** No edge or PnL claim — the edge
estimand is broken (settlement markout measures hold-to-expiry drift) and is
being re-specified separately. This file measures the **inventory process** only.

Probe: `skew_bound.py` (26 self-test checks). Rules pre-registered below before
the run; thresholds are the same bars the published run used.

## Verdict

**`SKEW_ROBUST` on both verdict coins.** The mechanism survives the pessimistic
queue assumption. The published 15× is optimistic on btc but not an artefact.

| coin | JOIN p95 | SKEW_UB | SKEW_LB | red_UB | red_LB | retention | LB half-life | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **btc** | 194.6 | 21.4 | **45.7** | 89.0 % | **76.5 %** | **0.86** | **12.5 s** | `SKEW_ROBUST` |
| **eth** | 92.0 | 20.0 | **17.1** | 78.3 % | **81.4 %** | **1.04** | **10.0 s** | `SKEW_ROBUST` |
| sol | 25.0 | 14.4 | 14.4 | 42.2 % | 42.2 % | 1.00 | 32.1 s | `SKEW_ROBUST` |
| xrp | 32.4 | 15.0 | 15.0 | 53.9 % | 53.9 % | 1.00 | 34.3 s | `SKEW_ROBUST` |
| doge | 15.0 | 10.0 | 10.0 | 33.3 % | 33.3 % | 1.00 | 279.4 s | `SKEW_ROBUST` |
| bnb | 15.4 | 10.0 | 10.0 | 34.9 % | 34.9 % | 1.00 | 177.3 s | `SKEW_ROBUST` |
| hype | 22.4 | 9.9 | 9.9 | 55.6 % | 55.6 % | 1.00 | 328.5 s | `SKEW_BOUND_DEPENDENT` |

Cash at risk, p95: btc **$121.80 → $8.11 (UB) → $9.14 (LB)**; eth
**$44.66 → $2.87 → $3.17**. The 15× survives as roughly **13×** at the bound.

**The re-post idealisation was doing less than feared.** On btc it fires 124
times per window under `SKEW_UB` and 0 under `SKEW_LB`, yet terminal `|net|`
only moves 21.4 → 45.7 and the reduction only 89.0 % → 76.5 %. The honest range
for btc is **76–89 % reduction**; the published figure is the optimistic end of a
narrow band, not a different answer.

**eth's retention exceeds 1.0 (1.04), and that is not noise to wave away.** The
lower bound produced a *smaller* terminal `|net|` than the upper bound (17.1 vs
20.0). Re-joining the back after every lift reduces fills on **both** sides, and
with 27 grants/window on eth the second-order effect on the adding side can
dominate the first-order loss on the reducing side. It is a small-sample
crossing at n=25 and should not be read as the lower bound genuinely beating the
upper one.

**hype is `SKEW_BOUND_DEPENDENT` for a different reason than the rule's name
suggests** — retention is 1.00, but the LB half-life is 328.5 s, outside the
300 s window. The cut is real; it is simply too slow. The `r≈60` decision point
and the dump mechanism both stand there.

## Fill decomposition — the two issues are NOT the same artefact

| coin | fills JOIN | UB | LB | inc_UB | inc_LB | surviving share | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| **btc** | 8,536 | 11,909 | 10,644 | 39.5 % | **24.7 %** | **0.63** | `MIXED` |
| **eth** | 1,719 | 2,668 | 2,519 | 55.2 % | **46.5 %** | **0.84** | `GENUINE` |
| sol | 364 | 600 | 582 | 64.8 % | 59.9 % | 0.92 | `GENUINE` |
| xrp | 323 | 677 | 681 | 109.6 % | 110.8 % | 1.01 | `GENUINE` |
| doge | 59 | 112 | 106 | 89.8 % | 79.7 % | 0.89 | `GENUINE` |
| bnb | 57 | 157 | 154 | 175.4 % | 170.2 % | 0.97 | `GENUINE` |
| hype | 55 | 183 | 183 | 232.7 % | 232.7 % | 1.00 | `GENUINE` |

**The hypothesis that the fill increase and the risk reduction are the same
artefact is refuted.** Between 63 % (btc) and 100 % of the increase survives
removing the re-post. Fronting **genuinely wins fills that the back of the queue
loses** — the advantage is queue position, not the idealisation.

btc is the only coin where a material part (37 %) is attributable to re-posting,
which is consistent with it being the only coin where re-posting fires often
enough to matter (124/window against 0.3–27 elsewhere).

**Candidate (c) is ruled out by construction, not by measurement.** Every arm
calls `reposition(bid, bid_sz)` / `reposition(ask, ask_sz)`, so all arms quote
**at the touch** and differ only in `qahead`. "The fronted side sits closer to
the touch" cannot be the explanation.

## The assumption under test

`PLACEMENT_SKEW_RESULTS.md` labelled its result an upper bound but never
exercised the generous assumption. It lives in
`inventory_walk.RestingSide.consume`:

```python
if self.resting <= 1e-12:                 # fully lifted -> re-post at back
    self.resting = self.size
    self.qahead = 0.0 if self.front else max(0.0, displayed)
```

A fronted side returns to `qahead = 0` **every time it is fully lifted** — it
wins the queue race again, instantly, for free. `JOIN` pays the displayed queue
on every re-post; the fronted side never does.

**Why the published robustness check could not see it.** `SKEW_IDEAL` barely beat
`SKEW` (13.2 vs 21.4), read as evidence the idealisation was not load-bearing.
But `SKEW_IDEAL` toggles the **flip** — teleporting to the front of an existing
queue when the policy changes. The flip was never the generous part. **The
re-post is, and both arms share it**, so their agreement tested nothing.

| arm | front granted by | interpretation |
|---|---|---|
| `SKEW_UB` | `reposition()` **and** every full lift | published behaviour; upper bound |
| `SKEW_LB` | `reposition()` only — genuine level re-formation | re-join behind displayed depth after a lift |

`SKEW_LB` is if anything pessimistic: it re-joins behind the **pre-trade**
displayed depth, part of which was consumed by the very trade that lifted us. The
truth lies between the arms, closer to the upper one than expected.

## Pre-registered decision rule

```
red_ub    = (JOIN_p95 - SKEW_UB_p95) / JOIN_p95
red_lb    = (JOIN_p95 - SKEW_LB_p95) / JOIN_p95
retention = red_lb / red_ub
```

- **SKEW_ROBUST** — `red_lb ≥ 0.20` **and** `retention ≥ 0.50` **and** LB
  half-life inside 300 s.
- **SKEW_BOUND_DEPENDENT** — retention below 0.50, or the cut survives but the
  half-life exceeds the window.
- **SKEW_INEFFECTIVE_AT_BOUND** — `red_lb < 0.20`.
- **UNRESOLVED** — fewer than 20 windows. **Defaults to `SKEW_BOUND_DEPENDENT`.**

Fill decomposition: `share = inc_lb / inc_ub`; below 0.25 artefact, above 0.75
genuine, between mixed.

## Controls

**The `JOIN` arm reproduces `inventory_walk` exactly** — net series, fill counts
and terminal net all identical on every checked window; the run **aborts** if not.
This validates the re-expressed event loop, which exists because
`placement_skew.simulate` constructs `iw.RestingSide` internally and offers no
seam for a different side object.

Independent confirmation: `JOIN` and `SKEW_UB` reproduce
`PLACEMENT_SKEW_RESULTS` **to the digit** — btc 194.6 / 21.4, eth 92.0 / 20.0,
btc fills 8,536 → 11,909 (published 4,249+4,287 → 5,934+5,975).

Self-test controls: `front_on_repost=True` must reproduce `iw.RestingSide`
exactly; with `front=False` the switch must be a **no-op**; a **partial** lift
must trigger no re-post in either arm; and `LB == UB` must be able to return
`SKEW_ROBUST`, else the rule could never confirm anything.

## A caveat on the new provenance mechanism

`fi.provenance()` reports `source_days = [20260819, 20260820, 20260821,
20260822]` — four days — while `PLACEMENT_SKEW_RESULTS` ran before the `DAYS`
fix on three. **The numbers nevertheless reproduce exactly**, because
`iw.select` takes the first `per_coin` slugs in sorted order, which are the
earliest, and the new day appends later timestamps that never displace them.

**So `provenance.source_days` records the days *read*, not the days *sampled*.**
It is an upper bound on a probe's day coverage, not a description of it. For a
probe that subsamples, the two differ, and comparing runs on that field alone
would mislead. Worth tightening if provenance is to be relied on.

## Scope

25 windows/coin, era `clob_v3_1`, window-clustered bootstrap at 400 resamples.
**No day-clustered interval is computable** at this day count, so all intervals
understate uncertainty. Per coin, never pooled. `NEW` is reported for context
only: btc terminal `|net|` p95 1805.1 at $541.76, with 1,455 free re-fronts per
window — the arm most exposed to the idealisation, and not a verdict arm.
