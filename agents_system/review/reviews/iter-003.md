# Review iter-003 (fix-round 1) — X121 structural `side → FLAT`

**Verdict: PASS** → handoff to Evaluation. No FIX-NEEDED items.

Script: `research/convexity_portable_2026-05-20/scripts/X121_flat_side.py`
Audit method: line-by-line diff vs X117/X120, plus independent recomputation from the
emitted parquets and a full X117 reference run.

This is a near-trivial STRUCTURAL change (delete the side trading branch → emit `{}`,
exactly like the existing bear→FLAT). The audit centers on (a) no leak introduced and
(b) measurement correctness — especially the LOFO, which is the decisive result.

---

## 1. Look-ahead / leakage — PASS

| Check | File:line | Finding |
|---|---|---|
| Regime label is the EXISTING PIT signal | X121:112–114 | `b30 = b4/b4.shift(180)-1`, bucketed ±10% → bull/bear/side. **Byte-identical to X117:54–56.** Trailing 30d BTC return; no current/future bar. Same label that already drives bear→FLAT. ✔ |
| mom30 construction trailing/shifted | X121:106 | `(c4/c4.shift(180)-1).shift(1)` — identical to X117:48. ✔ |
| beta construction trailing/shifted | X121:108 | rolling(180) cov/var `.shift(1)` — identical to X117:50. ✔ |
| NO signal/threshold selecting which side cycles | X121:128–129 | `if rg=="side": cyc_w_flat.append({})` **unconditionally** for every side cycle. No percentile, no `pr_lag>=thr`, no tuned fraction. **NOT the iter-002 timing trap** (X120:205–208 gated side cycles on a trailing-corr percentile; X121 has no such construct). Verified: `is_side.sum()=1455` on HL70 = side-regime count → ALL side cycles flatted. ✔ |
| No full-series quantile / model fit | grep | No `qcut`/`quantile`/retrain. Uses cached preds untouched + realized `return_pct` (walk-forward-purged label in the preds file) + the regime label only. ✔ |
| side→FLAT cannot leak | X121:129 | Deletes a branch (emits `{}`). Removing positions adds no future information. ✔ |

## 2. Correctness — PASS

**Base reproduces X117 (verified two ways).**
- Ran X117 reference: Sharpe **+1.93**, maxDD **−5674**, Calmar **+1.68**, totPnL **+10472** @4.5bps.
- Recomputed from `X121_percycle_HL70.parquet` `pnl_base`: Sharpe **+1.93**, maxDD **−5674**,
  Calmar **+1.68**, totPnL **+10472** — exact match. Engine faithful. ✔
- Base-arm weight build (X121:126–147) is line-for-line identical to X117:63–76 (same key
  selection, `<2*K` guard, `sort_values`+tail/head, beta-neutral `a/b`, `w` build).
- Held-book engine (X121:163–177) identical arithmetic to X117:78–87 / X120:200–230
  (sleeve overlap, turnover, `cost*0.5`, NaN guard). X121 precomputes `rs[t]` at build
  time instead of re-zipping inside the loop — equivalent return map per cycle. ✔

**Bull sleeves byte-identical across arms; only side branch differs.**
- Same `w` object appended to both `cyc_w_base` (X121:147) and `cyc_w_flat` (X121:148) on
  bull; bear emits `{}` to both (X121:127); side emits `{}` to flat only (X121:129). ✔
- Append invariant verified by control-flow trace: exactly one append to each arm per
  cycle in all branches (bear / side-traded / side-degenerate / bull / bull-degenerate);
  `assert len(base)==len(times)==len(flat)` at X121:150 confirms. ✔
- Note (expected, not a bug): per-cycle *realized* PnL on bull/bear cycles differs slightly
  between arms (HL70 bull max|Δ|≈5e-2) because flatting a side sleeve changes the HOLD=6
  overlap and turnover at later cycles. The *sleeve weights* are identical; the *measured
  cycle PnL* differs through overlap — this is precisely why G4a must re-derive held-book
  PnL (handoff flags this). ✔

**Cost / turnover.** `pnl = cyc − turn*0.5*cost` (X121:175), cost applied via `heldbook` at
{1,3,4.5} bps; weights cost-independent (built once). annualization √(6·365) (X121:62,73).
maxDD on cumulative bps equity (X121:71–72). ✔

**Per-cycle parquet** (X121:289–305) emits `open_time, fold, regime, is_side,
is_active_base, pnl_base_c{010,030,045}, pnl_flatside_c{010,030,045}, pnl_base,
pnl_flatside`. Verified: HL70 2405 rows, **0 NaN / 0 non-finite** in all 8 pnl columns;
`is_side`=1455, `is_active_base`=1881 (active pool for the matched-active placebo). ✔

**LOFO is CORRECT (decisive result, verified independently).** X121:225–241.
- `fold_arr` (X121:229) built by iterating `times` in the same order the pnl arrays are
  produced (heldbook returns over `range(len(cyc_w))`=`times`) → fold labels are NOT
  mis-mapped to PnL rows. ✔
- `keep = fold_arr != f`; `calmar_of(pnl_base[keep])` (X121:237–238) boolean-masks the
  realized per-cycle PnL and **recomputes Calmar (re-cumsum equity, fresh maxDD) on the
  fold-dropped series** — it does not reuse the full-sample maxDD. This is the correct
  LOFO-on-realized-PnL. ✔
- `.dropna()` inside `calmar_of` is a no-op here (0 NaN), so base/flat stay aligned. ✔
- Independent recomputation from the parquet reproduces the handoff LOFO table exactly,
  including the decisive row: **drop −f5 → lift collapses +3.04 → −0.86** (base Calmar jumps
  to +6.57 as its own worst DD is removed, flat actually loses); every other drop leaves
  lift +3.0…+3.7. Eval can rely on this. ✔

**NaN guards explicit.** X121:174 (`if not np.isfinite(cyc): cyc=0.0`), X121:73/83
(Calmar guarded `mdd<0 and isfinite`). No silent NaN→0 hiding missing data. ✔

**RNG seeded** (`SEED=12345`, X121:326). Placebo construction (≥500 seeds, re-derive
held-book) is left to Evaluation by design — the `heldbook` is mask-agnostic so a random
matched-count active-cycle FLAT mask can be driven directly. ✔

## 3. Faithfulness to spec — PASS
Parameter-free, structural, ALL side cycles flatted, analogous to bear→FLAT. Two arms
(base reproduces X117, flat_side = the change). HL70 + S44, costs {1,3,4.5}. Matches
research/handoff.md and implementation/handoff.md. Only deviation: realized HL70 folds are
2–8 (no fold 0/1/9 in the cached preds), so the research note's "f4 disaster" is realized
as **f5** — a relabeling only; the anatomy and the LOFO finding are unchanged.

## Independent verification run log
- X117 reference @4.5bps: +1.93 / −5674 / +1.68 / +10472.
- X121 parquet `pnl_base` @4.5bps: +1.93 / −5674 / +1.68 / +10472 (exact).
- X121 parquet `pnl_flatside` @4.5bps: +2.75 / −2239 / +4.72 / +11610.
- LOFO independent recompute: matches handoff (−f5 lift = −0.86).
- S44: base +1.84/−4170/+2.10/+25620; flat +1.45/−2778/+2.09/+16942 (transport flagged).
