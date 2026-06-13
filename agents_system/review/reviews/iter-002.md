# Review — iter-002 (fix-round 1)

Reviewer: Review Agent (adversarial). Code under audit:
`research/convexity_portable_2026-05-20/scripts/X120_corr_regime_gate.py`.

## Verdict: **PASS**

No look-ahead leak found. Base arm faithfully reproduces X117. THR is emitted (not
full-sample-selected) so G3 nested-OOS is feasible. Per-cycle parquet emits everything
G4/G5/G6 need. One requirement flagged for Evaluation (G4 placebo must re-derive the
held book, not zero pnl_base rows) — that is an Eval-side construction note, not a bug
in this script.

---

## 1. Look-ahead / leakage audit (the #1 concern)

### corr7d — strictly trailing, excludes current cycle ✔
`avg_pair_corr` (X120:91-102): `lo = max(0, i-window); sub = rt.iloc[lo:i]`. The slice is
`[lo:i]`, i.e. **excludes row i** (the current cycle) — no current/future return enters
the correlation. `min_periods=10` on the corr, `len(sub) < 20` skip guard. Byte-for-byte
identical to `iter002_hl70_dd_anatomy.py::avg_pair_corr` (anatomy:136-146) — verified by
diff of both bodies. `rt = ret4.reindex(idx)` where `ret4` is `np.log(c4/c4.shift(1))`
per symbol (X120:147,150), matching the anatomy's `ret4_map` construction (anatomy:73,75,78).
No future bar. **PASS.**

### pr_t — expanding percentile, NOT full-series quantile ✔
`expanding_pct_rank` (X120:110-121): maintains a growing `hist` list; for each i computes
`(h <= x).mean()` over `h = hist` **then appends x AFTER ranking** (X120:118-120). So the
comparison set is strictly prior (`hist[:t]` excludes `x_t`). No `pd.qcut`, no
`Series.quantile`, no full-series rank anywhere in the file (grep-confirmed). Warmup 100 →
NaN before warmup. Empirically verified: pr_lag ∈ [0,1], the only NaNs are a contiguous
block of 121 rows at the very start (corr warmup + 100-cycle expanding warmup + 1-cycle lag),
no mid-series NaN that would betray a windowed-quantile artifact. **PASS.**

### pr lagged 1 cycle ✔
`pr_lag = pr.shift(1)` (X120:184). The gate at cycle t reads `pr_lag[t] = pr_{t-1}`, i.e.
the corr percentile realized strictly before the decision cycle. Conservative against sleeve
overlap. **PASS.**

### Gate decision uses only info available at t ✔
`heldbook` (X120:205-209): the side-FLAT test reads `pr_lag[t]` only. The held-book overlap
window (X120:216) re-applies **each historical sleeve's OWN flat decision**:
`active = [({} if flat_mask[k] else cyc_w[k]) for k in range(max(0,t-HOLD+1), t+1)]`. Because
`flat_mask` is filled in forward iteration order, `flat_mask[k]` for k≤t is already the PIT
decision made at sleeve k's own cycle — no future decision is consulted. **PASS.**

### THR is TUNED → G3 nested-OOS REQUIRED, and the script does NOT full-sample-select it ✔
The script computes all three THR arms (X120:265-269) and emits `pnl_gated_t060/t070/t080`
+ masks to parquet (X120:307-311). It never runs an `argmax`/best-Calmar pick over the full
sample. `THR_FALLBACK=0.70` is used ONLY for the per-fold and by-year **reporting** blocks
(X120:286,290), explicitly labeled "(fallback)". The fold column is emitted so Evaluation
can pick THR on past folds and apply forward. **No full-sample THR selection. G3 is feasible
and must be run by Evaluation.** PASS.

### Construction features (mom/beta/regime) still trailing/shifted ✔
mom30 = `(c4/c4.shift(180)-1).shift(1)` (X120:144). beta = `cov/var` over
`rolling(180).shift(1)` (X120:146). regime from BTC trailing-30d return `b30 = b4/b4.shift(180)-1`
(X120:152). All identical to X117 (X117:48,50,54). No new look-ahead. **PASS.**

No retrain — uses cached preds untouched (X120:131). **PASS.**

## 2. Correctness / bugs

- **Base arm reproduces X117.** `heldbook(thr=None)` (X120:215-229) takes the
  `active = cyc_w[max(0,t-HOLD+1):t+1]` branch, builds net (`+wt/HOLD`), turnover
  (`sum |net-prev|`), `cyc = Σ net·ret`, `pnl = cyc - turn·0.5·cost` — byte-for-byte X117:81-86.
  Weight construction (X120:159-177) is identical to X117:60-76 (same regime split, `2*K`
  guard, beta-neutral side scaling a/b). Empirically base totPnL = **+10472 bps**, matching
  the handoff's reported +10472 (+1.93 Sharpe / −5674 maxDD). ✔
- **NaN guard explicit** (X120:227): `if not np.isfinite(cyc): cyc = 0.0`. Output parquet
  verified 0 NaN in all pnl columns. The only NaN column is `pr_lag` (121 warmup rows,
  expected and consumed as "trade"). No silent NaN→0 hiding missing labels — the guard is
  on the per-cycle dot product only, matching X117. ✔
- **FLAT mask strictly within the side pool.** Empirically verified: every True in
  `side_flat_t0{60,70,80}` lands on an `is_side==True` cycle. Gate logic (`pr_lag≥THR & finite`)
  reproduced with zero mismatch. Counts 590/446/315 of 1455 side cycles. ✔
- **Cost/turnover correct when a sleeve is flatted.** When thr set, a FLATted sleeve's
  weights are zeroed inside the HOLD overlap (X120:216), so the net book shrinks for HOLD
  subsequent cycles and `turn = Σ|net-prev|` correctly reflects the reduced gross/turnover
  (lower cost on the de-grossed book). This is the intended turnover-reduction mechanism. ✔
- **Per-cycle parquet emits** open_time, fold, regime, pr_lag, is_side, pnl_base,
  pnl_gated_t0XX, side_flat_t0XX (X120:299-312). This covers G4 (is_side pool + matched
  counts), G5 (fold), G6 (paired pnl diff by fold), nested-OOS THR. ✔
- **maxDD/annualization.** `stats` (X120:72-82): eq = cumsum(bps), maxDD = min(eq − eq.cummax)
  on cumulative equity, ann = mean·6·365, Calmar = annr/|mdd| guarded for mdd<0 & finite.
  `ann` = mean/std·√(6·365) matching conventions (4h disjoint). ✔
- RNG seeded `SEED=12345` (X120:339). ✔

## 3. CRITICAL note for Evaluation — G4 placebo construction validity

Flatting a side sleeve does **not** merely remove that cycle's own PnL: because the sleeve
participates in the HOLD-cycle overlap (X120:216), flatting it also changes the **held-book
turnover and gross for the next HOLD−1 cycles**. Therefore a correct count-matched side-pool
placebo MUST **re-derive the held-book PnL under each randomized FLAT mask** (re-run the
`heldbook` overlap+turnover logic with a random side-flat mask of the matched count), NOT
simply zero out `pnl_base` rows at the flatted cycles. Zeroing rows would (a) double-count
by ignoring the held-book decay the real gate gets, and (b) mis-state turnover/cost, producing
an invalid (too-easy or too-hard) placebo distribution.

The emitted parquet supports the correct construction (is_side pool, matched count via
`side_flat_t0XX.sum()`, pr_lag, fold), but the held-book re-derivation needs the engine — the
Implementation handoff offered to expose `heldbook` as a helper. Evaluation should either
import/replicate `heldbook(times, cyc_w, rs, regimes, pr_lag, cost, thr)` with a randomized
mask substituted for the gate decision, or request that helper. Do **not** take the
zero-the-rows shortcut.

## 4. Spec faithfulness
Matches research/handoff.md iter-002 spec: corr7d (trailing 42, exclude current),
expanding pct rank warmup 100, pr lag 1, side-only gate (bull/bear unchanged), THR grid
{0.60,0.70,0.80} emitted for nested-OOS, fallback 0.70, costs {1,3,4.5}, HL70+S44, reuses
X117 engine, no edit to X116/X117/preds. Two declared deviations (S44 corr recomputed from
its own universe; G4 placebo left to Eval) are both correct and necessary. ✔

## Gate readiness summary
- G1 (look-ahead): **PASS** (this review).
- G2/G3/G4/G5/G6/G7/G8: data emitted; Evaluation to adjudicate. G3 nested-OOS is REQUIRED
  (THR is tuned) and feasible. G4 must use the held-book re-derivation (see §3).
