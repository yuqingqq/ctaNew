> **AUDIT RETRACTION (2026-07-11):** the +2.30 headline and all FULL-STACK per-regime Sharpes referenced here are
> RETRACTED (gate look-ahead + path-coupled pitfall #4). Audited-honest numbers: `V4_LIMITATIONS_DIAGNOSIS.md`
> (book-level rank-IC +0.030/+0.024). The regime-conditional SIGN claims here are especially affected — book-level,
> side/bear are positive in BOTH eras (not opposite-sign); only bull/deep-bull flip.
>
> **CORRECTIONS BANNER (2026-07-07)** — this is a chronological session log; the CANONICAL single-source-of-truth is
> `V4_PERFORMANCE.md` (cleaned). Several verdicts here are superseded — each is now tagged INLINE with `[SUPERSEDED …]`
> at its section (search the file). Summary of superseded items:
> 5. "SIDE-short's period-dependent sign" (core thesis) → side is ONE-LEGGED: side SHORTS stayed profitable, only side
>    LONGS invert (V4_PERFORMANCE §5) — inline tag at the CORE THESIS.
> 6. "GATE DEEP BULL (BULL_DEEP_THR=0.20)" and "NEW BEST STACK w/ SIDE_FLAT_SKIP_THR=0.05" → NON-canonical: measured on
>    non-clean preds + a different stack than KEEPSET4 (which uses dd_stop + bull0); side_flat is a candidate needing a
>    clean-pred dual-window re-test, NOT a validated result — inline tags at those sections.
> Original four corrections:
> 1. "dd_stop REJECT (overfit)" → WRONG (Sharpe-on-negative-mean artifact): dd_stop halves OOS loss/DD,
>    improves every year incl. recent → KEEP (V4_PERFORMANCE §4).
> 2. "Optimized STACK = regime_gate only / bear+gate" → superseded by KEEPSET4 (bear + gate + dd_stop
>    + bull0): +2.23 recent / −0.44 OOS on clean v3 preds (§1).
> 3. bear_ramp/bear_k2/side_beta_neut isolation cells were NO-OPs (env dead code in the vanilla path),
>    not null results; bear refinements later tested properly on the active bear book = era-fit rejects.
> 4. "deep bull: washed-out bounce (long reverts)" in the core thesis → distribution check shows bull
>    longs are a lottery (median negative); the era-robust deep-bull long is the return_1d momentum
>    ranker (V4_PERFORMANCE §6.1), not reversion.
> The core thesis (selector-not-timer), OOS numbers, and regime tables below remain valid.

# v4 investigation — results ledger (2026-07-06)

Question thread: v4 (residual target) vs v3 (return target), regime long/short edges, tip capture,
feature-level validation, regime-tailored features. All numbers OOS (2023-01..2025-09), residual alpha
(`alpha_vs_btc_realized` = return − beta_PIT·btc, beta trailing/PIT), bps. Per-cycle Sharpe on 24h overlap
is ~2.4× inflated (true ≈ /2.4). Regime = BTC 180-bar return (>+10% bull, <−10% bear, else side).

## Books
- v3 OOS: `hl_lean175_oos` (base/short, V0_LEAN, return target), `hl_residrev_oos` (long, +resid_rev).
- v4 OOS: `hl_v4base_oos`, `hl_v4long_oos` — same pipeline, residual target. Gen: `live/gen_oos_v4.py`
  (154 syms, 552k rows, matches v3 OOS exactly).

## Results

### 1. Residual basis verified (task #10)
All leg edges measured on beta-adjusted residual, not raw. `X70:125 alpha = my_fwd − beta·btc_fwd`, beta PIT.

### 2. v4 vs v3 OOS, ungated (task #12) — `deep_v4_vs_v3_oos.py`
Traded-tip L/S at production K=1/2: **v3 +7.1 (Sh +0.72) vs v4 +0.2 (Sh +0.02)**. v4 worse at every raw K.
BUT the entire gap is the **bull-long leg** (v4 −17.8 vs v3 −3.6 @K1) — a leg any gate suppresses.
Shared structural weakness: side-long broken (−20), bull double-broken, edge concentrated in bear (7% of cycles).

### 3. Fair comparison under identical basic gate (task #13) — `deep_v4_tuned.py`
Gate = bear:L/S, side:short-only, bull:flat. Applied identically to both.
| K | v3 gated | v4 gated |
|---|---|---|
| 1/2 | +28.6 / Sh +4.79 | +25.6 / Sh +4.29 |
| 1/3 | +29.1 / Sh +5.50 | +24.8 / Sh +4.60 |
**The gate is the whole value** (both ~+0.5 → ~+4.5 Sharpe). v4 competitive, ~0.5 behind, NOT broken. v4 not rejected.

### 4. K-selection / top-K capture stability (task #14) — `deep_v4_kselect.py`
- **LONG: captured stably at rank-1 only** (+130, both halves +); rank-2 collapses to +4. K_long=1 correct.
- **SHORT: rank-1 unstable** (+7, sign-flips H1/H2); stable alpha is the **rank 2–6 band** (+21..+43). K_short≥2.
  K_short=1 is the *worst* short config. Sharpe rises monotonically to KS=4–5 → wider K_short is a lever (net-check pending, #15).
- v3 ≥ v4 in every (KL,KS) cell by ~0.5–0.8 Sh; no K rescues v4.

### 5. Feature-level tip validation (task #18) — `deep_v4_feat_tip.py`
Capture is feature-grounded, not overfit. LONG tip from **resid_rev_3 (+100/+84)**, vwap_slope, resid_rev_2, vol.
SHORT band from **broad ~11/16-feature consensus** → why it's a stable band not a point.
ret_3d weak at tip (validates dead-weight); autocorr carries short band (why OOS prune failed). return_1d flips
both legs here (it's the bull ranker — value is in the gated-out regime).

### 6. Per-feature tip BY REGIME — calibration map (task #19) — `deep_v4_feat_regime.py`
Pooled feature tip MISLEADS; features reverse by regime. Model confirms gate (long bear-only, short side+bear, flat bull).
- **Bear long**: mean-rev/vol carry it (resid_rev_3 +92, vwap +75, rvol +50); **bars_since_high (−17), ret_3d (−15) reverse**.
- **Side short**: broad robust consensus (all +15..+31) — healthiest leg.
- **Bear short**: NARROW/fragile — only **beta_change (+36)** carries; corr_to_btc (−91), idio_vol_1d (−67), rvol (−43) reverse.
- **Bull**: un-farmable both legs; no feature gives + bull short → sit-out validated.
Headroom check (model vs best single feature): bear-long model +130 > best feature +92 (**pooled already optimal, no headroom**);
bear-short model +27 < beta_change +36 (**~+9 headroom, the one candidate**).

## Verdicts
- v4 ≈ v3 under equal gating; residual target is a defensible wash, no lift. Keep v3 as production; v4 not adopted, not rejected.
- The strategy's value is the **coarse regime gate** (which leg where). Long book is already regime-optimal in bear.
- Only visible Layer-2 (feature-per-regime) headroom is **bear-short** → tested below.

## Open levers
- #15 wider K_short (3–4) net-of-fees — strongest gross signal, discrete.
- #16 resid_rev-heavier long book.
- #19 regime-tailored features — TESTS BELOW.
- #17 net fee+funding validation of the gated config.

## Regime-tailored feature tests (this session) — see V4_REGIME_TAILORED results appended below

## Regime-tailored feature tests (task #21-22) — gen_regime_tailored.py, deep_v4_regimecal[2].py
Built bear-tailored OOS books (residual target, same WF): bear-LONG drops {bars_since_high, bars_since_high_xs_rank,
ret_3d} (bear-long reversers); bear-SHORT drops {corr_to_btc_1d, idio_vol_1d, rvol_7d} (bear-short reversers).
Used ONLY in bear (regime-conditional); side/bull keep pooled v4. Gate = bear:L/S, side:short-only, bull:flat.

Isolation (K=1/3, H1 build / H2 validate; 16 of 33 folds contain bear):
| variant | H2 pooled->tailored | mean Δ/bear-fold | bear-fold wins |
|---|---|---|---|
| bear-LONG-only (drop momentum) | +11.3 -> +15.5 (Sh+1.26->+1.72) | +3.3 | 7/16 |
| bear-SHORT-only (drop reversers)| +11.3 -> +12.2 | -1.3 | 7/16 |
| BOTH | +11.3 -> +16.4 | +1.9 | 7/16 |
Bear-LONG tip H1->H2: pooled +199->+54, tailored +181->+115 (tailored MORE STABLE = less H1 overfit).

VERDICT:
- bear-SHORT tailoring REJECTED — mirage. Single-feature headroom (beta_change +36 vs model +27) did NOT survive a
  real retrain (ALL Δ~0, mean Δ/bear-fold -1.3). Thin bear sample (433 cyc) overfit trap; single-feature tip != book perf.
- bear-LONG tailoring PROMISING NOT CONFIRMED — H2 +4.2 bps/cyc (Sh +0.46 gross ~+0.19 net-of-overlap), positive mean
  per bear-fold (+3.3), sound mechanism (momentum reverses in bear-long). BUT breadth only 7/16 bear-folds (not majority).
- Partial validation of "regimes need different feature combos": YES bear-long (drop momentum), NO bear-short.
NEXT to confirm bear-long: full-history OOS (2022-2026 not just 2023-25 split) + net fee/funding; only adopt if breadth improves.

## Tip-measure -> prune -> re-eval (lean, task #24) — deep_tip_h1_prune.py, gen_tip_pruned.py, deep_tipprune_eval.py
Method (user-preferred, cheap): MEASURE per-feature regime tip on H1 -> prune reversers (H1 tip<=0) -> ONE retrain per
regime-leg -> validate H2. H1-decided prune sets: bear-LONG drop {autocorr, btc_rvol}; bear-SHORT keep 6 {btc_rvol,
obv_z, beta_change, bars_since_high(+xs), return_1d}; side-SHORT drop NONE (all 14 positive H1 = robust broad leg).
H2 validation (gated K=1/3): bear-LONG Δ+0.5 (5/16 folds, meanΔ/fold -1.1); bear-SHORT Δ-0.2 (9/16 folds, +1.1);
BOTH Δ+0.3 (6/16, -0.0). ALL WASHES on H2 (±0.5 bps, inside noise). None clears "H2 Sh up AND fold majority".
KEY: this KILLS the earlier "bear-long tailoring promising" — that decided prune on full-OOS tip (H2-contaminated=circular).
Honest H1-decision -> wash. Decide-H1/validate-H2 caught the false positive.
VERDICT: regime-specific feature pruning does NOT generalize OOS. Pooled 14-feature model is already regime-optimal;
coarse gate captures the regime structure, finer per-regime feature pruning adds nothing surviving honest OOS.
Consistent with all prior prune work (in-sample gains vanish OOS) and the vBTC regime-conditional-feature failures.

## K selection, NET of turnover cost (task #15) — deep_k_net.py
Feature tip STRUCTURE decides K: LONG concentrated (resid_rev_3 pins single name) -> K_long=1; SHORT broad consensus
(rank2-6 band) -> wide K_short. Net-of-fee sweep (cost 4.5 & 9 agree), KL=1, per-cycle Sh (~2.4x overlap-inflated):
  v3 net@4.5: KS=1 +1.23 | KS=2 +4.16 | KS=3 +4.82 (net-mean peak +25.4) | KS=4 +5.26 (Sh peak) | KS=5 +5.09
  v4 net@4.5: KS=1 +1.16 | KS=2 +3.68 | KS=3 +3.93 | KS=4 +4.44 | KS=5 +4.74 (still rising)
KS=1 decisively worst (= unstable rank-1 coin-flip from feature tip). Net Sharpe RISES with K_short even after fees
because broad stable consensus diversifies variance faster than added turnover; net mean flat, Sh climbs.
PICK: K_long=1, K_short=3 (net-mean peak + near-peak Sh + less turnover than 4 + discrete K=3 robustness).
Production K_short=2 leaves stable-band alpha on table -> bump to 3. (KS=4 = max Sh if accept +30% turnover.)

## Multi-agent regime-feature search loop (task #25) — regime-feat-search workflow, 14 configs — DEAD-END CONFIRMED
research->screen->test->review over the fixed regime_feat_harness.py (locked H1-decide/H2-validate PIT protocol,
block-bootstrap CI, per-fold breadth). 4 research angles (minimal subsets, stability-first, cross-regime asymmetry,
aggressive/contrarian) -> 14 unique configs, sequential retrain+eval.
RESULT: 0 PASS, 13 NEUTRAL, 1 FAIL, 0 kept after adversarial review.
Best near-miss a1_bl-stable6-drop-ret3d: h2_diff +0.66, Sh 1.322 vs 1.262, BUT CI [-6.40,+10.24] (~15x point est),
  bear-folds 7/16 = coin-flip noise. FAIL a3_mrpure-long-topmacro-short: h2_diff -14.6, CI hi -0.33.
KEY DEEPER FINDING: most configs are NEGATIVE even in-sample (h1_diff -8 to -13). No overfit gain to chase -> the full
14/16-feature books are already optimal on H1, not just H2. Mechanism: per-symbol RidgeCV L2 SHRINKAGE already
down-weights reversing/noisy features better than a hard binary keep/drop. Can't beat continuous shrinkage with pruning
(same reason earlier feature-prune failed). Root cause IS features, but the model already extracts the edge optimally;
NO per-regime feature lever exists above it. Wins live at CONSTRUCTION layer (gate, K), not feature layer.
DECISIVELY CLOSED: regime-conditional feature tuning, at 14-hypothesis depth across 4 angles.

## VALIDATION of the regime-confinement issue (2026-07-06, deep_validate_regime.py) — REAL, mechanism CORRECTED
Stripped ALL cost + gates, raw gross tip L/S (K=1/2) per year:
2023 +56.1 (Sh+4.08) | 2024 -9.3 (-0.60) | 2025H1 -67.1 (-2.88) | 2025H2 +136.1 (+2.39) | 2026 +109.8 (+3.96).
=> Issue REAL: gross model alpha itself goes NEGATIVE 2024 + 2025H1 (no cost/gate/look-ahead). OOS-negative is genuine alpha.
PRIOR MECHANISMS REFUTED BY DATA: (a) 'collapsed dispersion 2024' FALSE — xs disp ROSE 256->339->383; (b) 'mean-rev sign
flip' FALSE — MR-IC (resid_rev_2 vs fwd) stays POSITIVE every year (+0.009..+0.024). Average signal fine; the TIP inverts.
TRUE MECHANISM (per-regime tip, robust pattern): BEAR + every period (+211/+109/+79/+46 = robust anchor); BULL - every
period (+19/-48/-187/-190 = robust avoid, gate correct); SIDE SWINGS SIGN (+62/-4/-37/+229) = THE source of regime-fragility.
Regime-robustness problem = the SIDE-short's period-dependent sign. When side-MR live (2023,2025H2) strategy prints; when
inverted (2024,2025H1) it bleeds. OPTIMIZATION DIRECTION (validated): detect side-MR sign PIT (live vs inverted); bear is
robustly farmable, bull robustly avoid, side is the swing needing a PIT sign-detector. NOT features/K/target.
> **[SUPERSEDED — side is ONE-LEGGED (V4_PERFORMANCE §5).]** This tip table conflates the two side legs. Canonical:
> side SHORTS stayed profitable through 2024/2025H1 (+13..+43 every feature); only side LONGS invert. So the
> regime-fragility is the SIDE-LONG leg, not "the side-short's sign". The fix is side-long conditioning (open candidate
> §6.2), not a side-short sign-detector. (Side-sign IS confirmed NOT PIT-detectable — that part holds.)

## ============ SESSION STATE / REVISIT-LATER HANDOFF (2026-07-06) ============
Where things stand after the full v4 investigation + tuning validation. Pick up here next session.

### Settled conclusions (do NOT re-litigate)
- Feature layer + regime-conditional feature tuning: CLOSED (14-config multi-agent search 0/14 pass; RidgeCV L2
  shrinkage already optimal). v4 residual target vs v3 return: WASH.
- K: feature-tip decides shape (K_long=1 concentrated, K_short broad); net-of-cost K_short 2->3 is ~NEUTRAL end-to-end.
- ROOT CAUSE VALIDATED (deep_validate_regime.py): regime-break is REAL in the raw gross alpha (2023 +56, 2024 -9,
  2025H1 -67, 2025H2 +136, 2026 +110). Mechanism = traded extremes invert by regime (bear robust +, bull robust -,
  side later corrected to a SIDE-LONG fragility). Prior 'dispersion collapse' + 'mean-rev sign flip' hypotheses
  REFUTED by data.

### Performance record (net = fee4.5 + funding + depth-slippage, ~14.5 bps/fill)
- Backward-OOS 2023-25 (config held-out): full stack -1.68 / vanilla -1.30. LOSES (regime break, gates can't rescue).
- Recent 10mo 2025-10..2026-06 (per-month rolling WF model): full stack +2.70 / vanilla +1.16. BUT October-concentrated:
  vanilla ex-Oct -0.41 (October-luck); full stack ex-Oct +1.65 (gates rescue side + harvest bear -> broadly positive).
- October = side-short month (+15948 side). Side swings +16k(Oct)/-7k(Jan) even within recent window.

### Tuning-logic validation (vanilla-v4 loop, tuning_harness.py; PARTIAL — workflow still running at handoff)
Vanilla baseline: recent +1.263 / OOS -1.301.
| logic | recent Sh | OOS Sh | read |
|---|---|---|---|
| regime_gate | +1.905 | -1.304 (pnl -24099 vs -30790) | KEEP — recent lift + OOS capital-preservation |
| bull_hedge | +2.055 | -1.672 (pnl -34908, WORSE) | OVERFIT — helps recent, hurts OOS |
| bear_ramp | +1.263 (=vanilla) | -1.301 (=vanilla) | NO-OP on vanilla |
| (dd_stop, conc_cap, short_filter, kshort3, bear_k2) | pending | pending | workflow completing |

### RANKED NEXT STEPS (revisit)
1. **[SUPERSEDED / REFRAMED] Side-sign PIT detector** — originally framed as the lever, but canonical analysis
   later found side fragility is one-legged: side SHORTS held up, side LONGS invert. Current open item is side-long
   conditioning, not a generic side-short/sign detector.
2. Finish the tuning-loop re-rank by RECENT-window (ex-October) impact once workflow done (append results below).
3. Apply only validated-robust tuning (regime_gate risk-control kept; bull_hedge/bear_ramp flagged overfit/no-op).
4. If side-long conditioning cannot be validated -> strategy is regime-confined; deploy recent-favorable with reactive gate + kill-switch.

### Key files
Harness: tuning_harness.py (vanilla+env->both windows net). Validation: deep_validate_regime.py, deep_v4_perf.py.
Workflow: tuning_opt_workflow.js (run wf_35e70547-4e1). Books: hl_v4base_oos/hl_v4long_oos (OOS), hl_tgt_res_*(recent).
Trackers: V4_PERFORMANCE.md (perf A-G), V4_REGIME_INVESTIGATION.md (methods).

## TUNING-LOOP COMPLETE (wf_35e70547-4e1, 2026-07-06) — appended to handoff
Vanilla-v4 + 8 production logics validated (check/validate/apply/review) through tuning_harness.py. Vanilla: recent +1.263 / OOS -1.301.
VERDICTS: regime_gate APPLY (only); bull_hedge/dd_stop REJECT (overfit: recent +2.06/+1.59, OOS worse); conc_cap/short_filter/kshort3
REJECT (non-issues); bear_ramp/bear_k2 REJECT (DEAD CODE — vanilla BEAR_MODE=flat, bear never trades so BEAR_K never activates).
> **[SUPERSEDED for dd_stop — canonical KEEPS it, V4_PERFORMANCE §4.]** The "dd_stop REJECT (overfit)" here was a
> Sharpe-on-negative-mean artifact: on a losing OOS book any de-gross lowers Sharpe, but dd_stop HALVES OOS loss & maxDD
> and improves every year incl. recent → KEEP. (regime_gate APPLY and the DEAD-CODE findings for bear_ramp/bear_k2 hold.)
Optimized STACK (regime_gate only): OOS -1.304 (flat), PnL -24099 (22% less loss), maxDD -28126 (13% better), recent +1.905. Win = capital preservation only.
SIDE-SIGN DETECTOR FAILED (later reframed): ss_trailedge OOS -2.005 (worse, every yr neg); ss_trailedge_long = no-op.
Trailing-edge proxy CANNOT predict the side swing. Canonical reframing: side fragility is mostly the side-LONG leg,
not a side-short sign detector problem; generic side-sign timing remains closed.
TWO CARRY-FORWARD FINDINGS:
 1. BEAR anchor UNDER-FARMED: vanilla BEAR_MODE=flat suppresses bear (the ONLY robustly-+ regime, gross +211/+109/+79/+46).
    Untested lever BEAR_MODE=equal to farm it -> NEW TOP REVISIT (targets the robust part, not the fragile side). Opens cost+side exposure to validate.
 2. recent-primary + OOS-guardrail: regime_gate=clean keep; bull_hedge/dd_stop=best-recent but OOS-overfit -> deploy-with-caution only.
REVISED REVISIT PRIORITY (superseded by V4_PERFORMANCE §6): (1) BEAR_MODE=equal bear-anchor farming; (2) side-long
conditioning rather than generic side-sign timing; (3) if both fail -> regime-confined, deploy recent-favorable +
regime_gate + kill-switch. Honest recent ref: bootstrap median +2.68, 90%CI [+0.83,+4.51] IF regime holds.

## ============ CORE THESIS (2026-07-07) — the unifying conclusion ============
The strategy is a cross-sectional MEAN-REVERSION model. It correctly identifies reversion CANDIDATES (short the most
over-extended, long the most washed-out via 14 features), but WHETHER those candidates revert or keep momentum is a
market-REGIME property the model cannot predict PIT. "We pick reversion symbols, but in some regimes they keep momentum."
  - bear + favorable-side: candidates REVERT -> profit (both legs). The money regimes.
  - mild bull: NEITHER reverts (both keep momentum) -> ~zero edge (nothing to farm).
  - deep bull: washed-out bounce (long reverts) BUT over-extended keep pumping (short SQUEEZE).
  - side-inverted (2024/2025H1): washed-out keep falling -> long fails.
Model = good candidate SELECTOR, poor regime TIMER. This single fact explains everything:
  (1) side timing failed and was later reframed to a side-LONG fragility; (2) bear_mode_equal robust (bear reverts every
  period, no timing needed); (3) regime_gate necessary (reactively de-gross when momentum takes over); (4) don't-short-deep-bull
  (the one predictable momentum trap: strong BTC rally -> winners squeeze). Strategy's job is NOT to predict momentum regimes
  but to SURVIVE them: farm the reliable one (bear), gate the predictable trap (deep-bull short), reactively de-gross the rest.
  = the lineage that led to KEEPSET4 (bear_mode_equal + regime_gate + dd_stop + bull0).

## BULL 'MISSING PROPERTY' = MOMENTUM (2026-07-07, deep_bull_missing.py) — confirms parallel session
Hypothesis confirmed: bull LONG alpha is in the RAW/beta/MOMENTUM dimension the residual target strips out.
OOS deep-bull (n=1047) LONG picks res-vs-raw fwd: v4-resid +17/+92, v3-return +31/+106, MOMENTUM-long(recent winners) +55/+128.
=> (1) beta-neutral residual strategy throws away 2-5x of bull long edge (raw>>residual); (2) MOMENTUM-long is BEST deep-bull long
on BOTH raw (+128) AND residual (+55, beats reversion model +17) -> model longs washed-out=WRONG names in momentum regime;
(3) same coin as the short-squeeze: deep bull rewards momentum (long winners win +55/+128, short winners squeeze -44). Model does
reverse of both -> structurally mis-aligned to bull.
CONTRAST: side long alpha PURE residual (raw ~0, res +39 = regime residual model built for); bear both (res+67/raw+130).
MISSING PROPERTY = momentum in (deep) bull. FIX (defensible: deep-bull observable PIT via BTC r30, robust n=1047):
regime-conditional bull long = switch mean-rev -> MOMENTUM (long recent-winner alts, res +55; or long BTC for pure beta) +
reversion-short OFF. Sharper than bot's BULL_LONG_INSTRUMENT=btc (momentum-ALTS capture +55 residual BTC-beta alone doesn't isolate).
NEXT: test deep-bull momentum-long overlay on canonical KEEPSET4 (recent + OOS-guardrail).

## DEEP-BULL report-card cell (2026-07-07, deepbull.py) — GATE DEEP BULL validated
> **[SUPERSEDED / INTERMEDIATE — canonical is stronger.]** BULL_DEEP_THR=0.20 (gate deep bull only) was an intermediate
> step; canonical KEEPSET4 uses **BULL_GROSS_MULT=0 = FLAT BULL** entirely (the model-in-bull failure is established at
> the analysis layer for ALL bull, not just deep — V4_PERFORMANCE §5). The only live bull candidate is a separate
> deep-bull **momentum-LONG-only** overlay on a **1-day** return_1d ranker (§6.1; era-robust as EXPOSURE —
> ranking unproven OOS, Q3 placebo p=0.215) — NOT the mom30 book and
> NOT reversion. Treat the numbers below as a superseded lineage, not the current decision.
On the intermediate stack (bear_mode_equal + regime_gate), treatments for deep bull (btc_r30>0.20, PIT):
| variant | recent Sh | OOS Sh | OOS bull PnL |
|---|---|---|---|
| db_base (mom30 momentum bull) | +1.937 | -1.098 | -8726 |
| db_sitout20 (BULL_DEEP_THR=0.20 = gate deep bull) | +1.958 | -0.696 | +919 (FLIPS +) |
GATE DEEP BULL wins: OOS bull -8726->+919 (flips positive), OOS Sharpe -1.098->-0.696 (+0.40), OOS totPnL -21885->-12240
(44% less loss), recent unchanged (+1.958, few deep-bull cyc recent). Validates report-card cell: deep-bull distribution
change (momentum-short squeeze) is PIT-detectable + gating it works. This is the deep-bull treatment = SIT OUT.
Intermediate stack: bear_mode_equal + regime_gate + BULL_DEEP_THR=0.20 -> recent +1.958 / OOS -0.696 (vanilla was -1.30); superseded by canonical KEEPSET4 bull0.

## DEEP-BULL cell COMPLETE + CONFOUND resolved (2026-07-07)
Three bull constructions on this intermediate stack: gate-deep (BULL_DEEP_THR=0.20) recent+1.958/OOS-0.696/bull+919 [BEST OOS];
mom30-momentum recent+1.937/OOS-1.098/bull-8726; sidealpha-reversion(=v3's bull) recent+2.138/OOS-1.313/bull-14843 [WORST OOS].
=> GATE DEEP BULL was the robust winner inside this intermediate lineage. CONFOUND RESOLVED: v3's sidealpha bull is
RECENT-OVERFIT (best recent +2.138, worst OOS -1.313 = shorting squeeze-winners unhedged); v4's mom30 momentum BETTER
OOS than v3's reversion. INTERMEDIATE STACK: bear_mode_equal + regime_gate + BULL_DEEP_THR=0.20 -> recent +1.958 /
OOS -0.696, later superseded by canonical bull0/KEEPSET4. OOS trajectory in this lineage: vanilla -1.30 ->
+bear+gate -1.10 -> +deepbull-gate -0.70. Report-card framework validated deep-bull as a PIT-detectable trap, but
the canonical construction gates all bull model exposure via BULL_GROSS_MULT=0.

## TIP-SNR diagnostic — systematic regime CEILING (2026-07-07, deep_tip_snr.py)
Does the system work + can a systematic tip-distribution regime beat BTC-r30?
(1) Tip health: RECENT per-cycle SNR +3.61 (21% windows neg = stable signal); OOS SNR -0.07 (50% windows neg = NOISE).
    Tip autocorr(lag6, exit-lagged) ~0 BOTH windows -> tip health does NOT persist -> REGIME_GATE can only be reactive.
(2) PIT predictability (regress fwd tip on {r30,disp,btcvol,trail_tip}):
    SIGNED tip (which way): R2 0.0018 (r30) -> 0.0019 (+disp+vol) = IRREDUCIBLY UNPREDICTABLE (side-sign = noise, the CEILING).
    |tip| MAGNITUDE: R2 0.011 (r30) -> 0.021 (+disp+vol) = dispersion DOUBLES it; corr(disp,|tip|)+0.12 strongest.
CRUX: dispersion predicts tip SCALE (std) not SIGN (mean) -> high disp = bigger |tip| but RANDOM direction -> only
exploitable as inverse-dispersion variance-control = VOL_TARGET, which we already found HURTS. So NOT exploitably.
VERDICT: BTC-r30 + REGIME_GATE + bull de-grossing is at the regime-classification CEILING; the intermediate
BULL_DEEP_THR result was later strengthened to canonical bull0. Systematic tip-SNR regime adds at most
marginal variance-control, CANNOT fix OOS (OOS tip genuinely noise: SNR 0, sign R2 0.2%). Regime-confinement is a tip-SNR fact,
not a classifier artifact. Value = farm where SNR naturally high (bear, fav-side) + defend where it dies
(regime_gate/dd_stop) + gate predictable bull traps.

## MODEL-RELIABILITY ROUTER (2026-07-07, deep_model_reliability.py) — reproduces the intermediate lineage
Per-regime model tip SNR across 3 independent samples (OOS-H1/OOS-H2/RECENT):
  bear +4.96/+0.47/+2.07 = USE-PRED (reliable everywhere) -> bear_mode_equal.
  side +2.28/-1.58/+4.62 = PARTIAL (reliable fav-sign, inverts OOS-H2 = side-sign) -> pred + regime_gate reactive defense.
  bull-mild -0.15/-1.43/-0.01 = UNRELIABLE -> sit-out/de-gross.
  bull-deep -0.12/-2.85 = UNRELIABLE -> gate (deep-bull).
=> user's reliability-routing framework independently reproduced the intermediate stack
(bear_mode_equal+regime_gate+BULL_DEEP_THR), later strengthened by canonical KEEPSET4's bull0 + dd_stop.
KEY: "switch to other signals" for unreliable regimes mostly resolves to SIT-OUT — good alternatives don't exist (bull mom30 alt also loses;
side-inversion has no PIT alt). Only BEAR has a reliable model. Framework validates config; config bounded by no-alt-for-failing-regimes.
Reactivity sweep (REGIME_GATE_W on the intermediate stack): W=30(~5d) OOS-0.772/recent+1.31 WORSE than W=180 (-0.696/+1.958) -> too-fast whipsaws (autocorr~0). Middle windows pending.

## FRAMEWORK-DRIVEN fine-tuning (2026-07-07, sideflat.py + fwroute.py) — side_flat skip = PROMISING CANDIDATE (NOT canonical)
> **[NOT RECONCILED WITH CANONICAL — treat as candidate, not validated result.]** Measured in THIS session's harness
> (tuning_harness.py) on NON-CLEAN preds (hl_v4base_oos, not the `_clean` universe) and on a DIFFERENT stack lineage
> (bear+regime_gate+BULL_DEEP_THR) than canonical KEEPSET4 (bear+regime_gate+**dd_stop**+**bull0**, V4_PERFORMANCE §1).
> So the "-0.696 -> -0.069" and old "NEW BEST STACK" claim are NOT comparable to canonical's clean-pred -0.44, and the
> "both-window = not recent-overfit" claim OVERCLAIMS (2022 holdout still untested; framework calls side_flat FRAGILE/
> noise, t −0.5 = not reliably-negative, not a validated AVOID). NEXT: re-test side_flat skip on CLEAN preds atop KEEPSET4
> with a dual-window paired-CI test before any adoption. Scripts live in live/state/longtail/tune/ (untracked state dir).

Used regime-discovery framework's side_flat=FRAGILE/noise verdict to drive a bot gate (SIDE_FLAT_SKIP_THR: sit out side |btc30|<thr).
On the intermediate stack (bear_mode_equal + regime_gate + BULL_DEEP_THR=0.20):
| config | recent Sh | OOS Sh | OOS side PnL |
|---|---|---|---|
| sf_base (intermediate) | +1.958 | -0.696 | -14786 |
| sf_03 (skip |r30|<3%) | +1.982 | -0.489 | -9928 |
| sf_05 (skip side_flat |r30|<5%) | +2.474 | -0.069 | -3502 |
SESSION-LOCAL RESULT: skipping the side_flat noise band improves both windows in this non-canonical lineage:
OOS -0.696->-0.069, recent +1.958->+2.474, and OOS side loss cut 76% (-14786->-3502). Framework diagnosis
(side_flat=noise) may convert at the bot layer, but this is NOT a canonical stack result yet.
Candidate to retest [2026-07-07 later: retest DONE — V4_PERFORMANCE §6.3: alpha claim REJECTED,
DD lever only; Q6 placebo → mostly exposure-dose, mechanism unresolved. 2022 spent on KEEPSET4
(FAIL) — this lever is permanently uncharacterized there]: KEEPSET4 + SIDE_FLAT_SKIP_THR=0.05 on clean preds, with paired dual-window CI and 2022 holdout
before adoption.
