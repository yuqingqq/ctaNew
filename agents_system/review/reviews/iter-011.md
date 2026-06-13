# Review — iter-011 (REACTIVE risk-control track), fix-round 1

**Script:** `research/convexity_portable_2026-05-20/scripts/X124_reactive_dd_stop.py`
**Verdict: PASS** (reactive-track R1 look-ahead audit clean; correctness verified; supporting R4/R5/R6 logic trustworthy with disclosed scalar-approx caveats).

The decisive check for a reactive equity-DD stop is whether the de-gross trigger peeks at its own
forward (or same-cycle, unrealized) PnL. It does not. Detail below with FILE:LINE.

---

## 1. PIT of the equity/DD trigger — THE DECISIVE CHECK — CLEAN

### Canonical run `run_stop_heldbook` (lines 156–203)
The per-cycle causal ordering is correct:
- **L171–181** — `dd = eq - peak` and the `stopped`/re-entry decision are computed from `eq` and
  `peak` **as they stand entering cycle `t`**, i.e. reflecting realized PnL through cycle `t-1` only.
- **L182–183** — `g` (gross for cycle `t`) is fixed from that decision.
- **L185–198** — `pnl[t]` is realized using `g` (positions scaled BEFORE turnover/cost).
- **L200–202** — `eq += pnl[t]*1e4` and the peak update happen **after** the gross for `t` is fixed.

Walking the indices:
- t=0: `eq=0, peak=0`, `-dd=0 < X` → not stopped → `g=1.0`; `pnl[0]` realized; only then `eq+=pnl[0]`.
  No future peek.
- t=1: `dd = eq - peak` where `eq` reflects only `pnl[0]`. Decision uses through t-1=0.
- general t: the gross for `t` depends solely on `pnl[0..t-1]`. **No same-cycle or forward PnL enters
  the de-gross decision.** This is the textbook-correct equity-curve causal order; there is no
  equity-curve look-ahead.

### HOLD-lag (sleeve overlap) — correctly handled
The concern is whether `eq` at decision-time `t` smuggles in unrealized PnL of sleeves that have not
matured. It does not. `eq` is the cumsum of **realized per-cycle book returns** `pnl[0..t-1]`. Each
`pnl[k]` (L191–197) is the cycle-`k` return of the book actually held at cycle `k` — the
HOLD-sleeve average of `cyc_w[k-HOLD+1..k]` marked with that cycle's `return_pct` (`rs[k]`). That
return is fully known at the close of cycle `k`. The HOLD overlap lives in the *book composition*,
not in the equity accounting; nothing reads a forward sleeve return. The handoff's HOLD-lag claim is
substantiated.

### Running peak — expanding/PIT
`peak` is updated only by `if eq > peak: peak = eq` (L201–202) after each realized cycle → a monotone
running max of realized equity. Expanding, PIT. ✔

### Re-entry — realized-only
L176–181: `trough = min(trough, eq)` (realized equity), `gap = stop_peak - trough`,
`healed = (eq - trough) >= heal*gap`, `timed = (t - stop_t) >= timeout` (bar count). All from
realized-to-date info. The `eq > trough` guard (L180) prevents buying back AT the trough. ✔
(R7 re-entry sanity confirmed; `g_floor=0.40>0` keeps the book healing → no frozen-equity kill.)

### Scalar version `run_stop_on_scalar` (L126–153) — same correct ordering
`dd = eq - peak` → decide → set `gross[t]` → `eq += gross[t]*pb[t]` → update peak (L137–152). Same PIT
causal order. Used only for the cost-cheap R5-LOFO / R6-nested / R4-placebo loops on the already-
cost-netted base scalar (scales cost linearly — disclosed approximation, faithful to ~1%).

### Mechanical, reactive, single parameter
It reacts to a drawdown already underway; no forecast, no future peek. The **only** tuned parameter is
threshold `X` (g_floor/heal/timeout are fixed policy, L57–59) → correctly flagged for R6 nested-OOS,
which is implemented. ✔

**R1 (reactive-track look-ahead) = PASS.**

---

## 2. Correctness

- **Base reproduction.** `heldbook_gross` with gross=1.0 (L91–122) reproduces X123's `heldbook`
  (X123 L217–231) line-for-line: identical active window `cyc_w[max(0,t-HOLD+1):t+1]`, identical
  `wt/HOLD` net build, identical turnover `Σ|net_t - net_{t-1}|`, identical cost `turn*0.5*cost`,
  identical NaN guard (`if not finite: c=0`). So gross=1.0 == X117/X123 base book. The handoff reports
  HL70@4.5bps = X117 EXACT (+1.93 / −5674 / Calmar +1.68 / totPnL +10472); the reproduction path is
  structurally faithful. (Evaluation runs the numbers.)
- **Gross applied before turnover/cost.** `scaled = {s: g*v}` (L190) then turnover and return computed
  from `scaled` (L192–197). So the de-grossing trade itself pays cost — cost is NOT linearly scaled.
  This is the canonical gross-aware engine the contract (R3/G8) wants. ✔
- **maxDD on cumulative equity.** `metrics` (L74–87): `eq=cumsum(pb)`, `dd=eq-maximum.accumulate(eq)`,
  `mdd=dd.min()`. Correct. Calmar = mean·ANN/|mdd|, ANN=6·365 matches √(6·365) convention. ✔
- **R4 `const_degross` (L206–210).** Claim "pnl_const = g·pnl_base is exact for constant gross"
  verified algebraically: constant `g` ⇒ `turn_const = g·turn_base` and `c_const = g·c_base`, so
  `pnl_const = g·pnl_base`. The control matches the stop's *equal average exposure* via
  `ag = gross.mean()` (L300–301). Correct R4 control. ✔
- **R4-placebo (L317–333).** Matched %-time (`n_stop = stop.sum()` random cycles → GFLOOR), matched
  floor, 200 seeds, `rng` seeded (SEED=12345, L223). Real uses canonical held book; placebo uses
  scalar `base*gg`. Minor apples/oranges (placebo skips the marginal de-gross turnover cost) but maxDD
  is dominated by the gross scaling, not marginal cost — conservative, disclosed. ✔
- **R5 episode-LOFO (L373–386).** For each dropped episode, rebuilds `cyc_k`/`rs_k` on the KEPT
  indices, recomputes `base_k` and the stop on the subset (`gross_unit` + `run_stop_heldbook` on the
  subset). **Recomputes on the dropped subset — correct.** Caveat: dropping a middle episode
  concatenates non-adjacent cycles → a small turnover/HOLD seam distortion; standard LOFO
  approximation, conservative. Acceptable.
- **R6 nested-OOS (L395–427).** For fold `i`, `past = folds[:i]`, X chosen to max ddRed under ≤25%
  cost on PAST folds, applied to `fut = folds[i]`. **X selected only from past folds — no future
  leak.** Fallback to deepest X when no X meets budget (L413–414, least intrusive). Correct nested
  structure. ✔
- **NaN guards.** `metrics` drops non-finite (L75–76); held-book sum skips non-finite returns and sets
  `c=0` if the cycle sum is non-finite (L113–114, L195–196) — the same explicit guard X123 uses for
  the missing-symbol-return bug. No silent NaN→0 that hides data loss. ✔
- **RNG seeded** (L223). Deterministic. ✔
- **Reuses canonical engine** — imports X123 `build_universe`, HOLD, preds paths verbatim (L44–48);
  modifies no baseline script or cached pred. ✔
- **Underlying base book is PIT-clean** (X123 `build_universe`): mom30/betas/alt30/b30 all `.shift(1)`
  trailing (X123 L111,113,129,132); preds from a cached walk-forward parquet with `fold`; the stop
  layer adds no features, only reads realized PnL.

---

## 3. Faithfulness to spec / deviations

- Implements the spec exactly: equity-DD stop, X=1600 / g_floor=0.40 / heal=0.50 / timeout=90 bars,
  re-enter on 50%-heal-above-trough OR 90-bar timeout. ✔
- Disclosed deviations (acceptable):
  1. R4-placebo and R6-nested use the scalar approx (scale precomputed base PnL) for the
     resampling/per-fold loops to stay in budget; headline curve + R4-const + R5-episode + LOFO use
     the canonical gross-before-cost held book. The scalar approx is faithful (~1% on the recommended
     config) and is the same family the research probe used.
  2. X grid is {800…3000} (drops 4000); the deep-tail region is the interesting one. Evaluation may
     add 4000 if it wants the full row.

---

## Findings summary

No look-ahead. No correctness bug that would invalidate the backtest. The equity-DD trigger is
strictly PIT (decision for cycle t uses realized equity through t−1 only; HOLD-lag correctly handled;
expanding peak; realized-only re-entry; no forward peek). Base book reproduces X117 by construction.
R4/R5/R6 supporting logic is correct (R4 control is exact for constant gross; R5-LOFO recomputes on
the dropped subset; R6 chooses X only from past folds), with the scalar-approx caveats explicitly
disclosed in the handoff.

**This is a reactive RISK OPTION, not an alpha ADOPT** — consistent with evaluation_contract.md's
reactive track. The honest caveat (R4-PLACEBO FAIL: tail-cap is ~proportional, not skill-selective)
is correctly surfaced by the author and is the expected reactive-track outcome, not a bug.

**VERDICT: PASS → hand to Evaluation.**
