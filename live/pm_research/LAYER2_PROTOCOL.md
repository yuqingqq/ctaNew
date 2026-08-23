# LAYER2_PROTOCOL — carry-to-resolution, and the reconciliation of the two published numbers

**Status: §3 FROZEN by Ruling R-14 (2026-08-23), with the coordinator's three
amendments landed BEFORE `layer2_v1` runs — the binding sequence. If the
amended bar cannot be satisfied by the data, that IS the answer; it is not
re-cut afterwards.** Drafted blind by the DE session under Ruling R-11; no
settlement markout of the simulated maker population has been computed; every
number below is a previously published figure, cited as the bar's derivation
basis exactly as R-1's `f*` was.

Per R-11 this is **the last unmeasured place a maker edge could still live on
this venue**: the cancellation family is closed 8/8 coin-days; skew solves
inventory, not fill quality; Layer 1 says the passive maker loses at short
horizons. What was never measured is whether **holding to resolution recovers
it** — and the two published numbers pull opposite ways: the settlement census
reads **+0.173 ¢/share** pooled `[−0.251, +0.596]` (ALL real fills,
hold-to-expiry) while Layer 1 reads **−0.53 ¢** on btc at `h=5` (the simulated
`JOIN` maker, marked at mid, CI excluding zero). Different estimands over
different populations; the reconciliation deliberately never narrated. This
protocol measures it.

---

## 1. The estimand — ONE population, BOTH marks

**Population:** the simulated two-sided `JOIN_BBO` 5-share maker's fills — the
`edge_l1_v1` replay, conformance-locked, on the **day-series selection**
(`select_by_day`, 4 era days × 30 windows/coin; 08-19 excluded, pre-era).
This is the *policy-relevant* population: the census's all-fills population
answers a different question.

**Per fill, both marks, signed by maker side (`s` = +1 BUY, −1 SELL):**

```
M_h  =  s · (mid(t_fill + h) − ℓ)          Layer 1, already published at h=5..60
M_T  =  s · (payoff − ℓ)                    hold-to-resolution; payoff = 1{Up wins}
                                            for Up-terms fills, from SETTLEMENT
                                            FACTS (E-M6-verified winners)
bridge(h) = M_T − M_h                       what the remaining window gives back
                                            (or takes) after the h-second mark
```

`M_T` is the census's estimand on the maker's population. `bridge(h)` is the
reconciliation's mechanical core: Layer 1 measured adverse selection at `h`;
Layer 2 asks whether it mean-reverts or continues by resolution.

**Decomposition reported per fill population:** spread capture (vs mid at
fill, the stable leg) + drift to `h` + `bridge(h)` = `M_T`. Sums must close to
the identity exactly (selftest control).

**The reconciliation ledger, pre-committed:** the gap between the two
published numbers is explained as exactly two measured terms —
`population term` = census-`M_T`(all fills) − maker-`M_T`(this population),
and `estimand term` = maker-`M_T` − maker-`M_5`. Both computable; neither is
narrated beyond its number.

## 2. Populations, exclusions, discipline

Per coin × per day, never pooled across days (compare on `days_sampled`).
R-DUAL (with/without the 0.02 micro class). Unresolved windows are named
exclusions; gap/tick-touched fills are `UNAVAILABLE` rows per the Layer-1
convention (tick-touch matters for the mid legs, not the settlement leg —
both variants reported). Settlement facts joined by slug from the
E-M6-verified winners (99.8 % on 1,465; era-independent). Knowledge-time
discipline unchanged. Conformance: fill-for-fill against the reference
engine, abort on divergence. Receipts stamp provenance, the SP set, and the
frozen bars.

**Scope, v1:** the NEVER-CANCEL maker's carry — per-fill hold-to-resolution,
whose sum IS that maker's inventory PnL by linearity. The SKEW policy's
Layer 2 (a different carried residual) is v2, contingent on v1's answer, and
is NOT part of this freeze.

## 3. The bar — three-way per coin — **FROZEN per R-14, amendments 1–3 landed**

Verdict coins btc/eth; others descriptive. Cell = (coin, day), h=5 primary.
**VOID** below 500 fills with valid `M_T` on a (coin, day).

Per cell, on the share-weighted arm with the per-fill arm reported beside
(the census's sign flipped between weightings — both must be visible):

- `POSITIVE` — within-day CI of `M_T` excludes zero from above.
- `NEGATIVE` — excludes zero from below.
- `UNDETERMINED` — spans zero. **This is the expected outcome** (see the
  power declaration below, which turns that expectation into a number); it
  must not be dressed up.

**Coin verdict across the era days (R-14 AMENDMENT 1 — the day rule must
survive a running collector; `DAYS` went stale four times in three days, so
no verdict hardcodes a day count):**

- `CARRY_RESCUES` — **at least 75 % of era days `POSITIVE` and zero
  `NEGATIVE`**, minimum 4 era days.
- `CARRY_FAILS` — symmetrically: **at least 75 % of era days `NEGATIVE` and
  zero `POSITIVE`**, minimum 4 era days. **On both verdict coins this closes
  the last maker-edge hypothesis for the passive JOIN policy on this
  venue** — the symmetric falsifier, stated before the measurement so
  neither direction can be softened after.
- `UNDETERMINED` otherwise — a real outcome: the resolution is then
  calendar (more era days), not re-cutting. The rule keeps one meaning today
  (4 days: 75 % = 3, zero contrary) and at seven days (≥6, zero contrary),
  and cannot silently change character as the tape grows.

**POWER DECLARATION (R-14 AMENDMENT 2 — what it takes to FIRE, stated before
the run; EV_GATES_PLAN §5.1 extended from failing witness to firing
witness).** Derived a priori from the published census dispersion only: the
U10 share-weighted pooled CI half-width was ≈0.42 ¢ at 931 windows
(window-clustered), implying a per-window dispersion σ_w ≈ 6.6 ¢/share; at a
cell's ~30 windows the **minimum detectable |M_T| is ≈ 2.4 ¢/share** (both
verdict coins to order of magnitude; eth wider — fewer fills per window make
noisier window means). Beside it, the effect sizes actually observed in the
census: **+0.173 ¢ share-weighted, −0.211 ¢ per-fill ex-micro** — a **>10×
gap**. **So, said in the protocol before the run: at census-scale effects
this bar cannot fire in either direction, and the honest expectation is
UNDETERMINED on every cell — a pre-registered RESULT, not an excuse.** The
bar CAN fire if the maker-population carry is an order of magnitude larger
than the census's pooled residue (|M_T| ≳ 2.4 ¢ — continuation/reversal at
the scale Layer 1's −0.53 ¢-at-5-s drift would produce if it compounds
rather than reverts). The calendar statement with a number attached:
resolving a 0.2 ¢-scale effect needs ≈ 4,200 windows per cell-equivalent —
≈ 140 era days at 30 windows/coin/day, or ≈ 6 days at full-coverage
680/coin/day, and day-CLUSTERED inference additionally needs ≥ 5 clusters
(B6 §3a); a full-coverage multi-week design is a future protocol, not this
one.

**Sign disagreement between arms (R-14 AMENDMENT 3 — the U10 lesson):**
share-weighted is primary because it answers *does the capital deployed get
paid* — the estimand for an edge question — while per-fill answers *does the
average fill get paid*. But the census's two arms DIVERGED IN SIGN on the
same fills, and a single-weighting spec would have published "+0.165, makers
profitable" while concealing that it rested on one counterparty. **If the
arms disagree in sign on any verdict cell, that is a FIRST-CLASS FINDING
reported in its own right — never resolved silently by the primary.**

**Scope statement, frozen with the bar:** within-day inference only; four day
clusters support no day-clustered interval (B6 §3a measured 4-cluster
under-coverage); day-consistency across cells is the robustness statement,
exactly as in the R-9 day series. No PnL/capacity claim; maker fees are zero
(measured) so `M_T` is gross-and-net for the maker leg, but no
capital/turnover economics are computed under this protocol.

## 4. Controls (must fail if vacuous)

1. Identity control: on a synthetic fill with known mid path and known
   winner, `spread + drift_h + bridge(h) − M_T = 0` exactly.
2. A known-winner fixture: maker BUY Up at 0.40, Up wins ⇒ `M_T = +0.60`;
   Up loses ⇒ `−0.40`. Signs pinned both ways.
3. Settlement-join control: a window whose winner field is missing must
   produce a named exclusion, never a default.
4. Shuffle control: permuting winners across windows within a (coin, day)
   must move `M_T` (guards against the join being vacuous).

## 5. Sequencing

1. This draft goes to the coordinator (D-4). **Freeze before any receipt is
   read**; the probe may be built and run under the R-1 pattern
   (build-allowed / read-forbidden) if the freeze is pending.
2. On freeze: run, read against §3, report per day here — same shape as
   report #17.
3. v2 (skew-policy Layer 2) is scoped only after v1's verdict, as its own
   draft.

---

## 6. ANSWERED — 2026-08-23, `layer2_v1`, receipt `derived/layer2_v1.json`

**Coin verdicts under the frozen §3: btc UNDETERMINED
(U, U, NEGATIVE, NEGATIVE), eth UNDETERMINED (U, U, U, NEGATIVE).** Neither
roll-up threshold was met (btc 2/4 NEGATIVE = 50 % < 75 %; eth 1/4); per the
frozen rule the resolution is CALENDAR — more era days — not re-cutting.

**The descriptive pattern beneath the frozen verdicts, reported exactly:**

- **All 8 share-weighted point estimates NEGATIVE** (btc −0.68/−0.52/−2.37/
  −1.43 ¢; eth −0.71/−1.52/−1.13/−3.14 ¢), **both arms agreeing in sign on
  every cell** — zero amendment-3 findings to report.
- **3 of 8 cells resolve NEGATIVE with the within-day CI excluding zero**
  (btc 08-22, 08-23; eth 08-23); **zero POSITIVE cells anywhere.**
- The amendment-2 power declaration held in an informative way: observed
  effects came in ABOVE census scale (0.5–3.1 ¢ vs 0.17 ¢), which is why
  three cells could resolve despite the ≈2.4 ¢ per-cell MDE.

**The reconciliation ledger, both pre-committed terms measured:**

- **Estimand term ≈ ZERO**: `bridge(5)` is small and mixed-sign per cell
  (btc −0.12/+0.46/−1.46/+0.11; eth +0.24/−0.21/+0.80/−0.28) — **the
  adverse selection done by t+5 s STICKS to resolution**, neither recovering
  nor systematically compounding. `M_T ≈ M_5` on the maker population.
- **Population term = the whole gap**: the census's +0.173 ¢ (all fills)
  against the maker population's −0.5..−3.1 ¢ — the JOIN maker's fills are
  the adversely-selected subset, and marking them at settlement does not
  redeem them. **The +0.173-vs-−0.53 tension is a POPULATION effect, not an
  estimand effect.** Spread capture stayed stable (+0.59–0.85 ¢) and drift
  at 5 s carried the loss on every cell, matching Layer 1's legs.

Scope as frozen: within-day inference, four era days, no PnL/capacity claim.
Exclusions per cell in the receipt (worst: 86 gap/tick-touched M_h legs on
btc 08-20; unresolved-window exclusions zero on all sampled windows).
