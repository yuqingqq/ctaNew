# Review — iter-001 (X119 de-lever-only vol throttle)
fix-round: 1
verdict: PASS

Code under audit: `research/convexity_portable_2026-05-20/scripts/X119_delever_throttle.py`
Spec: `agents_system/research/handoff.md`  ·  Impl claims: `agents_system/implementation/handoff.md`

## 1. Look-ahead / leakage audit (the #1 concern)

### 1a. Throttle trailing-vol window `rv_t` — PIT, HOLD-lagged (lines 147–162)
At cycle `t` the sizing uses `n_avail = max(0, t-HOLD+1)` (line 153) and the window
`pnl[n_avail-W : n_avail]` (line 155). Verified by index walk:
- The largest PnL index touched is `n_avail-1 = t-HOLD`. At the start of iteration `t` the
  `pnl` list holds indices `0..t-1` (append happens at line 169, end of loop), so the window
  ends at `t-HOLD`, i.e. **HOLD cycles strictly before the cycle being sized**. No same-cycle
  (`pnl[t]`) and no future PnL can enter. Confirmed numerically for t = 6,7,47,48,49,100.
- This is in fact *more conservative than strictly required*: each `pnl[i]` is the held-book's
  per-cycle mark, fully realized at end of cycle `i`, so `pnl[0..t-1]` would already be PIT-legal.
  The implementer applies the HOLD lag unconditionally (the "if any sleeve overlap could leak,
  lag by HOLD" branch of the spec). Conservative, not leaky — accepted.
- Warmup: when `n_avail < W` the `if n_avail >= W` guard (line 154) keeps `s_t = 1.0`, so the
  negative-index slice (`pnl[-42:0]`) that arises early never reaches `.std()`. When the guard
  passes, `n_avail-W >= 0` always → no wraparound. Verified.
- `rv` is the std of the **throttled** arm's own past PnL (self-referential, as the spec intends:
  "the book's own per-cycle net PnL"). Past throttled PnL is fully realized → PIT-safe.

### 1b. Reference `tgt_t` — expanding/PIT median (lines 158–159)
`rv_hist.append(rv)` then `tgt = np.median(rv_hist)`. `rv_hist` accumulates only `rv` values that
were each computed from HOLD-lagged pre-cycle PnL; the median is over the running list, never a
full-series precompute. Including the current `rv_t` is legitimate (known at decision time `t`).
No future data. **PIT-safe.**

### 1c. Hard cap 1.0 (line 160) — never levers up
`np.clip(tgt/rv, FLOOR, 1.0)`. No `min(2.0, …)` anywhere (the X97 lever-up bug is absent — grep
confirms). Parquet check: `scale ∈ [0.30, 1.00]` exactly, **zero** values > 1.0 in both HL70
(n=2405) and S44 (n=6397). Confirmed.

### 1d. Construction features still trailing/shifted (lines 91–107) — no new leak
- `mom30` (line 99): `(c4/c4.shift(180)-1).shift(1)` — trailing 30d momentum, shifted +1 bar.
- `beta` (line 101): `rolling(180).cov/var` then `.shift(1)` — trailing, shifted.
- `btc30`/`regime` (lines 105–107): trailing BTC 30d return; `regime` thresholded from it.
These lines are **byte-for-byte identical** to the validated baseline X117 (X117 lines 42–56,
verified by diff). X119 introduces no construction change; the throttle is the only addition.
Any latent `btc30`-not-shifted question is a pre-existing property of the validated baseline, not
this iteration — and the base arm reproduces X117 to the bp (see 2e), so the engine is faithful.

### 1e. Scale applied to `net` BEFORE turnover/cost (lines 161–169)
`net = {s: v*s_t}` (161) → `turn` computed from scaled `net` (165) → `cyc` from scaled `net`
(167) → cost `turn*0.5*cost` (169). Cost scales with the reduced gross, as the spec requires.

### 1f. W=42 / FLOOR=0.3 are FIXED structural constants — G3 waiver legitimate
Module-level constants (lines 43–44), each used once (line 155 window, line 160 floor). Grep for
sweep/grid/selection over W or FLOOR returns nothing. The only loop is over `COSTS_BPS` (line 198,
required for G8 reporting, not selection). No Calmar-maximizing search. The research spec
pre-registers both as fixed and waives G3 on that basis — **the waiver holds.** (If a future
iteration varies W/FLOOR, G3 must be re-instated with nested-OOS.)

## 2. Correctness / bugs
a. **NaN handling explicit.** Per-symbol non-finite returns skipped (line 167); per-cycle
   non-finite contribution zeroed explicitly (line 168, the `totPnL +nan` guard). Parquet has
   **0 NaNs** in all columns (HL70 & S44). No silent NaN→0 hiding missing data.
b. **maxDD** on cumulative bps equity: `dd=(eq-eq.cummax()); mdd=dd.min()` (lines 72–73). Correct.
c. **Annualization** `√(6·365)` (line 62) — matches the 4h-horizon convention. Calmar
   `annr/|mdd|`, `annr = mean·6·365` (line 74). Correct.
d. **Per-cycle parquet** emits `pnl_base`, `pnl_throttle`, `scale` (= throttle `sc_t`), `fold`
   (lines 236–242), captured at the **4.5 bps production cost** (`keep` set when cost==4.5,
   lines 204–205). This is exactly what G4 (reuse the `scale` multiset, shuffle timing) and G6
   (block-bootstrap paired `pnl_base` vs `pnl_throttle`, block by `fold`) consume. Folds present:
   HL70 {2–8}, S44 {1–8}.
e. **Base arm reproduces X117.** Weight construction (lines 113–130) and held-book loop
   (lines 139–171) match X117 verbatim; with `throttle=False`, `s_t=1.0` ⇒ identical math. The
   handoff's HL70 base (+1.93 Sharpe / −5674 maxDD / +1.68 Calmar) matches X117's reported numbers
   → faithful reuse, not a subtly-different reimplementation.
f. **RNG seeded** (line 265). No randomness is actually exercised here (placebo is Evaluation's
   job) — harmless.
g. **Cost/turnover** correct; **groupby/merge** on `(symbol, open_time)` with tz-normalized
   timestamps (lines 88, 102, 105). No dtype/tz mismatch.

## 3. Faithfulness to spec
De-lever-ONLY (cap 1.0), parameter-free (expanding-median reference, fixed W/FLOOR), keyed on the
book's OWN PnL vol, applied before cost, on HL70 + S44 at {1,3,4.5} bps, emitting the per-cycle
series for G4/G6. All implemented as specified. Only declared deviation: HOLD lag applied
unconditionally — the conservative reading, PIT-safe.

## Verdict
**PASS.** No plausible look-ahead leak (throttle window is HOLD-lagged and provably touches only
realized pre-cycle PnL; reference is expanding/PIT; construction unchanged from validated X117).
No invalidating bug (NaN guard explicit, maxDD/annualization/Calmar correct, parquet complete for
G4/G6, base reproduces X117). W/FLOOR are fixed structural constants → G3 waiver legitimate.
Cleared for Evaluation. Note for Evaluation: the spec's primary gate is G2/Calmar; the implementer's
own sanity run shows HL70 throttle Calmar +0.91 < base +1.68 — that is a *performance* judgment for
Evaluation, not a correctness blocker.
