# Review — iter-012 (REACTIVE track), fix-round 1

**Subject:** `research/convexity_portable_2026-05-20/scripts/X125_volnorm_stop.py` — vol-normalized
(portable) reactive equity-drawdown stop, unitless k (rec k=2.0).

**Verdict: PASS** (clear for Evaluation). No look-ahead found. The held-book engine is the validated
X124/X123 verbatim; only the TRIGGER FORM changed (absolute X bps → unitless k·σ·√win). Base reproduces
X117 exactly. R5/R6 logic is trustworthy.

---

## 1. PIT of the vol-normalized trigger (DECISIVE) — PASS

The trigger loop in `run_volnorm_heldbook` (X125:105-154) is strictly point-in-time. Walked two ways
(static read + synthetic trace + live HL70 run):

- **DD-from-peak uses through t-1.** `dd = eq - peak` (X125:107) reads `eq`/`peak` whose last update was
  at the END of cycle t-1 (X125:151-153). At decision time t they contain realized equity through t-1.
- **σ window is trailing-only and strictly through t-1.** `seg = incr[lo:t]` with `lo = max(0, t-vol_win)`
  (X125:111-112). Upper bound `t` is **exclusive** → reads increments for cycles `lo..t-1` ONLY. No
  full-sample; window is the trailing 180 bars. Confirmed by synthetic trace (seg never includes `incr[t]`).
- **The increment for cycle t is recorded AFTER pnl[t] is realized.** Order in the loop body:
  (a) decide `gross[t]` from t-1 state (X125:118-130) → (b) realize `pnl[t]` (X125:132-144) → (c) write
  `incr[t] = step` (X125:147-150) → (d) advance `eq`, update `peak` (X125:151-153). So `incr[t]` is never
  visible to the σ computation at decision time t, and no same-cycle/future increment can enter σ. The
  numpy `incr` buffer is `np.empty` (uninitialized garbage at index t) but index t is **never read** before
  it is written — only `incr[lo:t]` is sliced, which is all previously-written entries.
- **k is a unitless constant.** `K_GRID=[1.5,2.0,2.5,3.0]`, `REC_K=2.0` (X125:72-73). It is NOT fit in the
  forward loop; the only place k is *selected* is R6 nested-OOS, and that selection uses past folds only
  (see §4). `trig = k*σ*√win` (X125:116) — self-normalizing to each universe's own equity vol.
- **Warmup fires early-firing guard.** `can_fire = (t>=warmup) and isfinite(σ) and (σ>0)` (X125:118).
  WARMUP=60. Live test: `stop[:60].any() == False` — verified no stop before bar 60.
- **HOLD-lag for sleeve overlap.** The realized PnL at t is a book of HOLD=6 sleeves entered t-HOLD+1..t
  (X125:132), multiplied by `rs[t]` = realized `return_pct` at open_time t. The gross only SCALES the
  current net position; nothing indexes a forward `rs` or a forward sleeve. The equity gated on is realized
  through t-1. **No forward peek.**

**Conclusion: the trigger is HOLD-lagged, trailing-only, no forward peek. PIT is clean.** This is the
paramount check and it passes.

## 2. Correctness — PASS

- **Base reproduces X117 EXACTLY.** Live run: HL70 @4.5bps base = Sharpe **+1.93**, maxDD **-5674**,
  Calmar 1.68, totPnL 10472 (X117 target +1.93/-5674). The PnL/turnover/cost block of
  `run_volnorm_heldbook` (X125:132-145) is structurally byte-identical to X124's validated `heldbook_gross`
  (X124:101-117) — only difference is where `g` is sourced (X125 sets it from the stop state at :129;
  X124 reads `gross[t]`). When the stop never fires, `g=1.0` ∀t → exact `gross_unit` → X117. Verified by
  structural diff (identical modulo g sourcing).
- **Gross applied BEFORE turnover/cost.** `scaled = {s: g*v ...}` (X125:137) then `turn` computed on
  `scaled` vs `prev` scaled (X125:138-139), then `pnl[t] = c - turn*0.5*cost` (X125:144). De-grossing pays
  its own turnover cost — correct accounting (the reason X124/X125 exist over the scalar probe).
- **k-sweep (R2/R3):** `run_volnorm_heldbook` per universe × cost × k (X125:247-261), ddRed/totCost
  computed against the matched base; canonical held book. Correct. Live k=2.0 HL70 = +33.1% ddRed,
  matches handoff.
- **R5 per-episode + episode-LOFO:** episode maxDD computed within each window on the same running EXT
  equity (X125:328-340); LOFO rebuilds `gross_unit` + `run_volnorm_heldbook` on the kept indices
  (X125:348-361). Same accepted X124 LOFO pattern. The HOLD-sleeve seam at a dropped episode is a
  measurement approximation (not look-ahead), disclosed.
- **R6 nested-OOS:** k chosen on PAST folds only, applied forward, separately per universe — see §4.
- **R4 constant-de-gross:** `const_degross(base, avg_gross)` = exact for constant gross (scales positions
  AND turnover uniformly). Correct (X124:206-210).
- **NaN guards:** `metrics()` drops non-finite (X124:75-76); held-book skips non-finite returns and sets
  c=0 (X125:142-143); `step` coerced to 0.0 if non-finite (X125:148-149); σ uses only finite seg values,
  requires ≥2 points else σ=0 → no fire (X125:112-113). No NaN can reach the trigger (verified).
- **RNG seeded:** `rng = np.random.default_rng(SEED=12345)` once in main (X125:208); `rng.choice` in the
  R4-placebo loop only. Deterministic. N_PLACEBO=200 ≥ 100.

## 3. Self-normalizing & mechanical — PASS

- Self-normalizing: `trig = k·σ·√win` rescales to each universe's own equity vol; same unitless k means
  "same #σ of equity" everywhere. No per-universe absolute threshold (the X124 portability failure fixed).
- Mechanical (no forecast): the trigger reads only the strategy's own realized equity/σ through t-1.
  Re-entry uses realized-only info (heal from realized trough/peak, timeout = bar count); `eq > trough`
  guard prevents buy-back at trough (X125:127). Live: 15 RT, gross ∈ {0.40, 1.0}, 51.4% time stopped.

## 4. R5/R6 trustworthiness — TRUSTWORTHY

- **R6 (X125:369-408):** the `for name in (HL70, EXT, S44)` loop wraps the whole nested-OOS so k is chosen
  **separately per universe**. Inner `for i in range(1, len(folds))`: `past = folds[:i]` (strictly prior),
  `fut = folds[i]` (next fold only). best_k = argmax ddRed under ≤25% cost budget computed on `base[past]`
  ONLY; applied to `base[fut]`. Forward metrics concatenate only OOS (future) folds. `best_k is None`
  fallback → deepest k (least intrusive) — a legitimate non-peeking default. **k is chosen only from past
  folds and applied forward, per universe. Correct.** Fold slices reset equity/peak to 0 at the slice start
  (scalar-approx) — folds are thousands of cycles so WARMUP=60 is <1% and negligible; the reset is
  conservative (cannot carry a deep DD across the boundary). Same approximation family X124 used.
- **R5 (X125:311-361):** episode-LOFO does NOT vanish dropping any episode (handoff +37–39%); rebuild is
  the validated X124 pattern.

## Notes (non-blocking)

- **N1 (disclosed deviation):** R4-placebo compares the canonical-held-book real maxDD (which pays de-gross
  turnover cost) against scalar `base*gg` random controls (no turnover recompute). This is an apples/oranges
  in the COST dimension but defensible for the maxDD/tail comparison, and is CONSERVATIVE for the
  "is-it-skill" question (the real stop carries extra de-gross cost the placebo doesn't). Handoff discloses
  the scalar approx for R4-placebo / R6. Acceptable.
- **N2:** R4/R4-placebo land p55–p70 (<p95) — the tail-cap is ~proportional to exposure removed, NOT a
  skillful tail-selector. This is the HONEST reactive-track result, correctly framed by Implementation as a
  RISK OPTION, not an alpha ADOPT. R1/R2/R5/R6 are the gates that matter here and they hold.

## Gate summary (reactive track)
- R1 look-ahead: **PASS** (decisive PIT check clean).
- R2 tail reduction: numbers correctly computed (live +33.1% HL70). Evaluation confirms magnitude.
- R3 bounded cost: full curve emitted; correct.
- R4 vs constant-de-gross / placebo: correctly computed; ~proportional (honest, expected).
- R5 cross-episode + LOFO: logic correct.
- R6 nested-OOS: k chosen past-only, applied forward, per universe — correct.
- R7 re-entry: g_floor>0, eq>trough guard, heal-or-timeout — correct.

**PASS → handoff to Evaluation.**
