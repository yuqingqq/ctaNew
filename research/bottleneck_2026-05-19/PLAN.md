# Bottleneck Plan — v3 (2026-05-19)

> v2 superseded after Round-2 review (Methodology **STILL-FLAWED** — fatal
> label-as-feature leak; Red-team **PROCEED-WITH-CHANGES** — same leak +
> aggTrades arm redundant with closed Steps 94b/95/98; Profitability
> **ALIGNED-PROCEED**). v3 implements all mandatory fixes.

## Changelog v2→v3 (every mandatory fix, traced)
1. **Leak (fatal):** the "66 superset" is replaced by a **hard by-name
   denylist** (any col containing `target`,`alpha`,`realized`,`basket`,`_fwd`,
   `btc_target`,`demeaned`,`return_pct`,`xs_alpha`) → 39 leak-free candidate
   cols. A **blocking assertion** `max|rankIC(feat,target_A)| < 0.10` runs
   before any fit. Already verified: max = **0.036** (`ema_slope_20_1h`), all
   ≤0.036 → the 39-col safe superset is leak-clean (evidence in
   `results/B_prefeature_ic.txt`).
2. **Redundancy:** the aggTrades-flow arm **A2 is dropped** — the
   flow/microstructure lever was decisively closed 2026-05-18 (linear-line
   Steps 94b LGBM −0.35 / 95 flow LGBM-neg / 98 flow×regime −1.6); cited, not
   re-run. A1 (panel idio-moment/microstructure superset in the **LGBM V3.1
   portable** construction) is NOT covered by those linear-line closures, so
   the one remaining arm is non-redundant.
3. **Gate:** decision is the **paired Δ(A1−A0)** measured through **one
   identical R3c run** (block-bootstrap CI on A1−A0), NOT an absolute +0.5 vs
   the historically-separate −0.33. MDE computed on the pooled per-cycle
   series with the correctly-scaled formula (not R1.metrics()'s √CPY-doubled
   one).
4. **Sized menu:** B3's orthogonal-data row carries a pre-registered EV bracket
   bounded by the on-disk alpha-oracle gap (Sharpe +94.7 vs model-pred −5.0)
   and the already-measured cohort spreads (ethbtc +8.58, xs_ret_disp +7.18,
   below the >11 Glassnode-justification bar).

## Question
Where is the binding constraint — features / model / harness — answered as a
sized, portability-gated lever menu. Single-bottleneck framing is ill-posed
(low IC ceiling + high convexity Sharpe coexist); deliver the menu.

## Reconcile (CITE, do NOT recompute — verified on disk / in records)
| closed result | source |
|---|---|
| alpha-oracle Sharpe **+94.7**; model-pred clean-alpha **−5.0** | `outputs/vBTC_phase1d_oracle_v2/nets_*.npy` (verified) |
| per-cycle IC predictability **R²=0.005** (unpredictable noise) | memory Phase DDI |
| IC-selector S/N **0.32**; ALL-eligible +2.45 > top-15 +2.06 | `diag_ic_selection_*`; memory |
| in-univ +2.23 = ~5-name convexity (H 0.19), rotates VVV→AXS→PENDLE | `R1c_concentration_truth.json` |
| proper portability (full stack, unseen syms) = **−0.33** | `R3c_portability_proper.json` |
| model reframings (RANK/SEG/CAL) rejected (IC ∓0.005) | memory Phase RANK/SEG/CAL |
| flow / micro / interaction features closed-negative | linear-line Steps 94b/95/98 (2026-05-18) |
| **39 leak-free panel cols: max univariate \|rankIC\|=0.036, ~all ≤0.02** | `results/B_prefeature_ic.txt` (this arc) |

## The one test — B★: de-leaked feature-superset, LGBM V3.1 portable
**A0** = WINNER_21 (no `sym_id`). **A1** = WINNER_21 + the 39 leak-free safe
cols (denylist + `max|rankIC|<0.10` asserted, =0.036 ✓). Same harness
(`build_audit_panel.train_fold_restricted`, `phase_ah_sleeve`,
`R1.aggregate_capped`, `R3c_portability_proper` protocol), same folds/embargo/
label-purge/listing-eligibility/`exit_time`, target `target_A` (R0-clean).
Metrics (harness currency, NOT noise-mean-IC):
1. pooled OOS top-K(=3) realized-`alpha_A` spread (bps);
2. in-universe full-V3.1 Sharpe (cap-1/3, equal, flat-4.5);
3. **portable R3c Sharpe**, A0 and A1 run through the *same* group-disjoint /
   no-sym_id / unseen-symbol / beta-neutral / costed protocol; report
   **paired Δ(A1−A0)** with block-bootstrap CI + N_eff + corrected MDE.

**Pre-registered (falsifiable; miss ⇒ rewrite diagnosis not gate):**
- A1 lifts the portable R3c Sharpe over A0 by **Δ ≥ +0.5 with paired
  block-bootstrap CI excluding 0** → **feature engineering is a real fundable
  lever** (sized prize = Δ).
- Δ ≤ +0.2 OR paired CI includes 0 → **feature engineering on free on-disk
  data is exhausted** — earned: 39 extra leak-free features (microstructure,
  idio-moments, name-factor) tested in the portable construction, on top of
  the per-feature IC scan (max 0.036) and the closed flow/sector/redundancy
  phases.
- If paired MDE > +0.5, report as effect-size estimate + "no detectable
  portable feature lever" (never a false "exhausted").

## B3 — Sized portability-gated lever menu (deliverable)
| lever | in-univ Δ | ports? | sized prize | source |
|---|---|---|---|---|
| model/target reframe | ≈0 | n/a | ~0 | closed RANK/DDI |
| harness: drop IC-selector | ≈+0.4 | **no** | 0 portable | closed selector |
| feature engineering | **B★** | **B★** | **B★ Δ** | THIS test + IC scan + closed flow |
| data/scope (orthogonal/on-chain) | unknown | only un-refuted | **bracket: 0 → (94.7−(−5)) ceiling; plausible band anchored to cohort spreads 7–9 (<11 bar)** | reconciled |
Verdict = the lever whose **portable** prize > 0. No pre-written conclusion.

## Process
v3 cleared by a focused final 3-angle check (are the 4 mandatory fixes correctly
implemented + leak-clean) before B★ runs. After B★, 3-agent results review vs
these gates; leak/fudge ⇒ re-initiate.

## Out of scope
Recomputing closed oracle/DDI/selector/flow results; aggTrades-flow arm
(closed 2026-05-18); orthogonal-data acquisition; deployment.
