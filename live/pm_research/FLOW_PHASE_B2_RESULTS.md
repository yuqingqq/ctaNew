# `phase × r` interaction, and the B2 demotion check

Probe `flow_phase_b2.py` (18 self-test checks, controls in both directions).
Decision rules pre-registered in the file before either run. Nothing committed.

**Headline: one specification decision is refuted, and the other proposed change
is refuted in the opposite direction from what was expected.**

---

## Item 1 — `phase × r`: `INTERACTION_MATERIAL` on all seven coins

`SPEC_REV2` §2 chose binning option (d) — 15 s bins plus a 4-level phase factor —
on the stated but untested assumption that the unidentified 60 s component enters
**additively** and is **constant in `r`**. It is not.

With 20 bins of 15 s, bin `k` maps bijectively to `(cycle = k // 4, phase = k % 4)`,
so "additive" is exactly the Poisson independence model on a 5 × 4 table with
exposure offset. Effect size is

```
RATIO = rms(interaction residual, log) / rms(phase main effect, log)
```

— the non-separable part relative to the thing option (d) claims to remove.

| coin | n windows | events | phase amp (log) | ratio | ratio CI95 | verdict |
|---|---:|---:|---:|---:|---|---|
| btc | 273 | 588,791 | 0.1576 | **1.685** | [1.379, 2.072] | INTERACTION_MATERIAL |
| eth | 273 | 121,322 | 0.1497 | **1.660** | [1.268, 2.230] | INTERACTION_MATERIAL |
| sol | 273 | 54,654 | 0.1397 | 2.377 | [1.776, 3.524] | INTERACTION_MATERIAL |
| xrp | 273 | 53,789 | 0.1214 | 2.164 | [1.675, 2.939] | INTERACTION_MATERIAL |
| bnb | 273 | 43,490 | 0.1098 | 3.506 | [2.585, 4.937] | INTERACTION_MATERIAL |
| doge | 273 | 35,440 | 0.1446 | 3.519 | [2.730, 4.814] | INTERACTION_MATERIAL |
| hype | 273 | 37,608 | 0.0832 | 4.486 | [3.352, 6.055] | INTERACTION_MATERIAL |

Material bar is 0.50. **Every lower bound clears it by a factor of 2.5 or more**,
and every ratio exceeds **1.0** — the non-separable structure is *larger than the
main effect the factor exists to remove*.

Population: `clob_v3_1` covered slugs, 273 windows/coin. Intervals are
window-clustered and, per the standing caveat, **understate** uncertainty at two
collected days. The parametric p = 0.0025 on every coin is reported as secondary
only: it assumes Poisson within cells and is anti-conservative under the
overdispersion clustering implies. The verdict rests on the effect size.

### Where the interaction lives — diagnostic decomposition

**Post-hoc, and it does not re-open the verdict.** It explains where the failure
comes from, because the source changes what should replace option (d).

Interaction residual RMS per cycle, cycle 0 = window open:

```
             r=300-240  240-180  180-120  120-60   60-0
btc            0.049     0.074    0.086    0.121   0.568
eth            0.094     0.047    0.057    0.094   0.535
```

The terminal minute carries **4.7× (btc) and 5.7× (eth)** the residual of the
next-largest cycle, and within it the pattern is a monotone collapse — btc
`+0.404, +0.165, −0.174, −1.034`, a **≈4.2× decline across the final minute**.

**So the dominant failure is not the 60 s oscillation interacting with `r`. It is
that `f_r` has genuine structure FINER than 60 s in the terminal minute, and a
5-level cycle factor cannot express it.** The additive model is then forced to
attribute the terminal collapse to *phase* — which is worse than a nuisance
factor failing to separate. **A factor labelled `unidentified_60s_component`
would be eating real `f_r` signal.**

`SPEC_REV2` half-anticipated this: it states (d) "does not make the terminal
region interpretable — the phase factor and the terminal fall are still
confounded there". What it did not anticipate is that the terminal contamination
is large enough to dominate the whole-window fit.

### Excluding the terminal minute does not rescue it

Body-only additive fit, cycles 0–3 (`r = 300..60 s`). **Post-hoc diagnostic; any
body-only specification must be pre-registered before use.**

| coin | full ratio | body ratio | body CI95 |
|---|---:|---:|---|
| btc | 1.689 | **0.357** | [0.263, 0.587] |
| eth | 1.615 | **0.369** | [0.267, 0.615] |
| doge | 3.520 | 1.079 | [0.853, 1.442] |
| sol | 2.377 | 1.292 | [1.034, 1.691] |
| xrp | 2.167 | 1.312 | [1.010, 1.801] |
| bnb | 3.530 | 1.774 | [1.360, 2.499] |
| hype | 4.473 | 2.885 | [2.026, 4.638] |

btc and eth improve roughly fivefold, but **both still exceed the 0.25 additive
bar and their intervals reach above 0.50**. Additivity is not supported on any
coin, terminal minute included or excluded.

### What must change in the specification

- **Option (d) is refuted.** `f_r` reported "net of `unidentified_60s_component`"
  is not net of the oscillation, and on the terminal minute the factor would
  absorb real signal. `SPEC_REV2` §2's recommendation, and
  `FLOW_MODEL_PROTOCOL_V4` in so far as it inherits the 15 s + phase grid for
  Tier A, need replacing.
- The replacement is **not** simply option (b). 60 s bins absorb the oscillation
  but put the entire terminal collapse inside one bin — and the terminal
  structure is now measured to be the largest single feature of `f_r`.
- What the evidence points to is a **non-uniform grid: coarse in the body,
  fine in the terminal minute** — the shape `SPEC_REV2` §1.3 already chose for
  Tier B, applied to Tier A for the opposite reason. That is a specification
  decision and is **not made here**.
- The `phase × r` test is now **run**, and `SPEC_REV2` §6.5's listing of it as
  "pre-registerable but unrun" is stale.

---

## Item 2 — B2 demotion: `REFUTED`, and in the unexpected direction

Paired leave-one-window-out, identical windows, 12 per coin, `clob_v3_1`. B3's
gamma is **refit in each arm**, because `fd.fit_baseline` builds B3's offset as
`b1 · exp(b2β · tick_tail)` — B3 is estimated *conditional on* B2, so dropping B2
requires refitting, not skipping a multiplication.

Cumulative NLL per event vs B0 (negative better):

| coin | B1 | B3 with B2 | B3 without B2 | removing B2 helps? |
|---|---:|---:|---:|---|
| btc | −0.0106 | **−0.0233** | −0.0233 | no — identical to 4 dp |
| bnb | −0.0024 | **−0.0147** | −0.0147 | no — identical |
| sol | +0.0006 | **−0.0069** | −0.0069 | no — identical |
| xrp | −0.0127 | **−0.0182** | −0.0182 | no — identical |
| hype | −0.0233 | **−0.1085** | −0.0881 | **no — materially worse without** |
| eth | −0.0013 | +0.0164 | +0.0162 | marginally, but B1 beats both |
| doge | +0.0009 | +0.0028 | +0.0041 | no — B1 beats both |

**Verdict: do not demote B2 from the nesting.** Removing it is neutral on four
coins, marginally better on one where B1 beats both arms anyway, and materially
worse on hype.

**But B2's real defect is different from the one alleged.** It is not harmful —
it is **inert**. On btc, sol and hype the adjacent `B2 − B1` delta is
`0.0000, 0.0000, +0.0002`; the 0.001 tick exists only in the tails, where there
is little activity, so `tick_tail = 0` almost everywhere and `exp(β · 0) = 1`.
B2 is a free parameter that does nothing on most of the book.

### A claim that does not replicate

The premise for this item was that B2 is "actively worse on bnb (+0.0141)" and
produces a non-monotone stack there. At **24 windows** that holds — `B2 − B1 =
+0.0141`, `best_layer = B1`, and the full stack (−0.0086) is worse than B1
(−0.0207). At **12 windows** it does not: bnb's full stack (−0.0147) beats B1
(−0.0024).

**The bnb harm is sample-unstable, which is itself the finding.** A layer whose
contribution flips sign between a 12- and a 24-window sample of the same era is
fitting noise on that coin, not a tick-regime effect. That is a stronger reason
to distrust B2 than "it is harmful", and a different one — it argues for the
existing `HALF_EVENT_FENCE_AND_NOT_PROMOTABLE` rule being load-bearing rather
than decorative.

Recommended change: **retain B2 in the nesting, reclassify it as inert-and-
unstable**, and require that it never be promoted on development evidence. Not
made here.

---

## Contradictions with current claims

1. **`SPEC_REV2` §2, binning option (d) — REFUTED.** Its additivity assumption
   fails on every coin by a factor of 2.5+ on the lower bound. This is a live
   specification decision, and `FLOW_MODEL_PROTOCOL_V4` inherits the grid.
2. **`SPEC_REV2` §6.5 lists the `phase × r` test as unrun.** It is now run.
3. **"B2 is actively worse on bnb" does not replicate at 12 windows.** It is
   sample-unstable rather than settled, and B2's real defect is inertness.
4. **The proposal to demote B2 is refuted** — removing it does not improve B3 on
   any coin and materially worsens hype.

Nothing here contradicts `FLOW_MODEL_STATE.md`, which makes no claim about the
binning or about B2. Its §2 entry that the terminal mechanism is confounded with
a 60 s artefact is **unaffected**: this test says nothing about the *source* of
the oscillation, only that its effect is not separable from `f_r`.

**One caveat on Item 2's scope.** 12 windows per coin, chosen to avoid competing
with a concurrent 24-window run. The paired comparison is valid at any n because
both arms use identical windows, but the absolute NLL figures are not comparable
with the 24-window receipt — and the bnb instability above is precisely why that
matters.
