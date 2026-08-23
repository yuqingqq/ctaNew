# LAYER2_PROTOCOL — carry-to-resolution, and the reconciliation of the two published numbers

**Status: DRAFT FOR COORDINATOR FREEZE — nothing here is frozen.** Authored by
the DE session 2026-08-23 under Ruling R-11's invitation and the coordinator's
item-3 directive: *scope it and bring me the bar BEFORE you run it.* **Drafted
blind**: no settlement markout of the simulated maker population has been
computed; every number below is a previously published figure, cited as the
bar's derivation basis exactly as R-1's `f*` was.

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

## 3. The bar — three-way per coin, derived a priori, PROPOSED for freeze

Verdict coins btc/eth; others descriptive. Cell = (coin, day), h=5 primary.
**VOID** below 500 fills with valid `M_T` on a (coin, day).

Per cell, on the share-weighted arm with the per-fill arm reported beside
(the census's sign flipped between weightings — both must be visible):

- `POSITIVE` — within-day CI of `M_T` excludes zero from above.
- `NEGATIVE` — excludes zero from below.
- `UNDETERMINED` — spans zero. **This is the expected outcome** given the
  census's pooled CI spanned zero at comparable n; it must not be dressed up.

**Coin verdict across the era days:**

- `CARRY_RESCUES` — ≥ 3 of 4 days `POSITIVE`, 0 `NEGATIVE`.
- `CARRY_FAILS` — ≥ 3 of 4 days `NEGATIVE`, 0 `POSITIVE`. **On both verdict
  coins this closes the last maker-edge hypothesis for the passive JOIN
  policy on this venue** — the symmetric falsifier, stated before the
  measurement so neither direction can be softened after.
- `UNDETERMINED` otherwise — a real outcome: the resolution is then
  calendar (more era days), not re-cutting.

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
