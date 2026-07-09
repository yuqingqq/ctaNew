# Convexity v4 — performance tracker (CANONICAL, cleaned 2026-07-07)

Single source of truth for the v3/v4 program's validated results. Mechanics reference:
`CONVEXITY_V4_FLOW.md`. Methods log: `V4_REGIME_INVESTIGATION.md`. Review/intervention log:
`ALPHA_TESTING_PLAN_REVIEW.md`. Full pre-cleanup history: git (this file was rewritten 2026-07-07 to
remove superseded/contradictory sections; every number below survived validation; superseded findings
are compressed in the Appendix).

All net numbers: bot cost engine, FEE_BPS_FILL=4.5 (HL base-tier taker, verified) + per-symbol depth
slippage (cost_10k) + funding. Windows: **recent** = 2025-10-04..2026-06 (evaluation-primary per user
doctrine), **OOS** = 2023-01..2025-09 (guardrail); **2022 holdout SPENT 2026-07-07: FAIL — see §1 annotation.** Cleaned universe = excl.
{LITUSDT, VINEUSDT, PUMPUSDT} stale-print symbols (`hl_*_clean` preds).

---

## 1. CANONICAL CONFIG — KEEPSET4 (final validated stack)

**Model preds (two-book: V0_LEAN base → shorts, V0_LEAN+RR → longs) + BEAR_MODE=equal +
REGIME_GATE (W=180, K=2, binary, full univ) + DD-stop (STOP_K_SIGMA=2.0, skip bear) +
BULL_GROSS_MULT=0 + K=1 long / 2 shorts + inv_sqrt_vol + kill-switch.**
Four mechanism-validated levers, no fitted knobs.

| frame | recent Sh / PnL / maxDD | OOS 2023-25 Sh / PnL |
|---|---|---|
| **v3 reference preds (clean)** | **+2.23 / +22,240 / −7,892** | **−0.44 / −6,376** |
| v3 reference preds (pre-clean) | +2.77 / +25,863 / −8,208 | −0.59 / −7,137 |
| v4 candidate preds (pre-clean) | +2.17 / +18,422 | −0.28 / −3,183 (2023 +10.8k) |
| **v4 candidate preds (clean — canonical baseline for §6 paired cells)** | +2.22 / +19,250 | −0.28 / −3,197 |
| *fitted production v3 stack (comparison)* | *+2.68 / +21,393* | *−1.57 / −12,528* |
| *vanilla (no levers, v3 preds)* | *+1.13 / +10,399* | *−1.05 / −26,891* |

KEEPSET4 beats the fitted production stack on BOTH windows with 4 levers instead of ~10 fitted knobs.
Bull-mult dose-response is monotonic (bull0 −15.2k / bull30 −17.2k / bull50 −18.6k OOS) → bull0 is a
mechanism endpoint, not a swept pick. State: `keepset_v3ref_*`, `keepset4_*`, `tune/_cfg/v2_KEEPSET4_m.json`.

> **2022 HOLDOUT (one-shot, 2026-07-07): FAIL — era-fragile.** B=KEEPSET4, 2,190 cycles, 39→48
> syms: Sh −2.83 / PnL −13,339 / maxDD −13,959 cumulative-bps (equity −76%); vanilla A −2.24 /
> −11,678 / −13,429. Rule i FAIL (B < 0.5×A = −5,839, both prongs); Rule ii FAIL (bear net −8,949,
> day-block 95% CI [−17.2k, −0.8k] entirely negative, bot's own labels, seed-stable). All 4
> quarters negative → not cold-start-favored. Funding stress (−309) and 2×-depth (−6,697) only
> worsen. Decomposition: bear net = gross −3,022 (CI [−11.2k, +5.0k], crosses 0) − costs 6,075 + funding +149 —
> **the bear edge did not invert in 2022; it was absent gross and cost-dominated** at full bear
> gross (stop skips bear by design; stop engaged 96.6% of non-bear cycles). B's risk levers still
> beat A where they act (bull0 +4,857 vs A's mom30 book; gate/stop on side +1,618). Survivor
> universe (panel has no LUNA/FTT/UST/SRM/ANC/RAY): all 2022 losses are LOWER bounds. C (deep
> overlay) descriptive: Δ+123 vs B, maxDD guardrail passed. Consequence: §7 blocking gross cap.
> No further 2022 cells ever (one-shot law). Pre-registration + verdict audit:
> RESEARCH_LOOP_20260707.md Iter 4-5. State: `live/state/longtail/holdout2022/`.

## 2. DECISIONS (user, standing)

- **v3 = production REFERENCE** (return target, two-book). All variant results are paired deltas vs
  v3 on the same cycles, never standalone. **v4 (residual target) = parallel forward-test candidate**
  and the ANALYSIS frame (label = tip metric = harvested quantity → clean attribution). Transfer
  rule: v4-frame findings are candidates; one confirmation pass on v3 before production application.
- Promotion (pre-registered): v4 replaces v3 only when the forward ledger repeats the
  bear/dispersion-event advantage (Nov-2025-type months) with paired significance.
- Risk layer (KEEPSET4 levers + kill-switch) is SHARED by both books — not a comparison variable.
- Evaluation frame = recent window primary; OOS as guardrail (2023-25 market differs structurally:
  universe composition, listings/delistings, regimes).

## 3. MODEL VERDICT — v3 vs v4 (settled)

- Full stack, recent: **statistical tie** (+2.98/+3.11 daily-Sharpe frame; ex-Oct +1.84/+1.93; v4
  +6.2% PnL, edge Nov-2025-concentrated, CI crosses 0). OOS gated: v3 ≥ v4 in every (K,K) cell by
  ~0.5 (in noise). Regime texture: v4 better bear (mechanism-consistent), v3 better side; bull
  insulated. Residual label carries trailing-beta estimation noise that offsets its decontamination.
- **rr_both (single model both legs): REJECTED** — reverses under the gate stack (−0.3 to −1.1 vs
  two-book at fair fees); RR degrades side-regime short selection. RR belongs on the LONG book only
  (OOS feature-tip: resid_rev_3 is the strongest long feature, +100/+84 both halves).
- base_both (drop RR from longs): in-sample +3.29 was a WINDOW ARTIFACT (OOS feature-tip refutes);
  reverted. **Two-book split stands.**
- K: **K=1/2 both models** (recent short tip = 2 ranks; the KS=3 preference was an OOS-frame
  property; bot-level wash; single-name-short squeeze risk rules out KS=1). K_long=1 unambiguous
  both eras. Dynamic K (PIT trailing-rank-edge rule): REJECTED both windows — trailing estimators
  lag era structure (6th adaptive-timing failure). v4-specific deep short band (ranks 4-6 recent,
  survives cleaning) = forward-test watch item only.

## 4. LEVER LEDGER (dual-window individual validation + composition; tuning_harness.py)

| lever | recent | OOS | verdict |
|---|---|---|---|
| BEAR_MODE=equal (plain) | +1.26→+1.41 | −1.30→−0.98, bear +5.0k, 2023 flips + | **KEEP — only alpha lever, both windows** |
| REGIME_GATE 180/binary | →+1.90 | loss −22% (era-asymmetric: 2025 helped, 2024 hurt) | keep — reactive risk control |
| DD-stop 2σ (skip bear) | →+1.59 | **loss & maxDD HALVED, every year better** | keep — best protection (early "overfit" verdict was a Sharpe-on-negative-mean artifact) |
| BULL_GROSS_MULT=0 | +2.41 in combo | −15.2k vs −21.9k combo; dose-response monotonic | **keep — see §5 bull anatomy** |
| bull_hedge package (BTC hedge + return_1d shorts + sit-out) | +2.06 | −1.67 (WORSE than vanilla) | overfit — era-fit |
| bear K2 / bear depth-ramp | +1.46/+1.78 | −1.17/−1.26 (worse than plain bear) | reject — refinements are era-fit |
| vol_target | +0.68 (halves the good era) | −26.7k | reject |
| auto_sizer | +1.18 | −22.8k | dominated by dd_stop |
| conc_cap / short_filter / kshort3 / sizing variants | ≈vanilla or worse | ≈vanilla | inert / no lever; inv_sqrt_vol validated |
| side_beta_neut, short_conv_tilt (isolation cells) | bit-identical to vanilla | — | NO-OP (env dead code in vanilla path) — untested, not null results |

Composition: the four keeps act on disjoint failure modes and COMPOSE (§1). Combos re-admitting
era-fit refinements lift recent but give back OOS (greedy-drift, blocked by rule: base combo =
both-window levers only).

## 5. ROOT CAUSES (validated)

- **Core thesis** (parallel session, confirmed): the model is a good reversion-candidate SELECTOR and
  a poor regime TIMER. Avg cross-sectional MR holds in EVERY regime (corr(trail,fwd) −0.02..−0.03
  incl. bull); the failures live at the TRADED EXTREMES, tail-driven, regime-specific.
- **Regime map** (model picks, fwd): side L +19/S −28 ✓revert; bear L +26/S −76 ✓revert (the anchor,
  + every period); mild bull: L −16 (washed-out keep falling); deep bull: S +79 (shorted names keep
  pumping = squeeze). Raw alpha is October-luck in the recent window (vanilla ex-Oct negative).
- **Bull is structurally untradeable for this model class (§5)**: pred-worst decile is the BEST
  decile in bull (+14, sign-inverted tip) at every horizon (pred edge −8@4h/−36@24h — the old
  "+40@4h reverses" claim does NOT replicate); tip noise 1.7×; 26% tail events; crash-vs-squeeze
  discrimination collapses (15.9% vs 14.4%). Bull-long is a lottery (median −73/−107; top-3 cycles
  ≈97-106% of totals). Feature-universal: NO feature (funding incl.) has a positive bull tip →
  missing state variable = positioning/crowding (funding proves it: uncrowded bull shorts earn +46,
  crowded −2 — but the sign flips by regime, so unusable as a global feature).
- **Bot never used the model in bull** (BULL_MODE default=mom → mom30 trend book; user-caught).
  Bot-level bull losses = the mom30 L/S book (half-right per sub-regime, both legs held in both).
  Model-in-bull failure established at the analysis layer. Both fail → bull0 unchanged.
- **"Side swings sign" is ONE-LEGGED**: side SHORTS stayed profitable through 2024/2025H1 (+13..+43
  every feature); only side LONGS invert. Reframes the side-sign problem to the long leg.
- Side-sign / era timing is NOT PIT-detectable (6 failures: trailing gates, dynamic K, meta-gates,
  DDI R²=0.005) — the noise floor (~600bps/cycle vs ~75bps signals) needs 2-3 months to detect;
  only the slow REGIME_GATE + kill-switch survive as reactive defenses.

## 6. OPEN CANDIDATES (ladder: capture paired-CI both windows → bot overlay → forward ledger; 2022 spent 2026-07-07)

1. **Deep-bull LONG overlay — BOT-TESTED 2026-07-07: the only candidate that lifts BOTH windows;
   kept as FORWARD-TEST CANDIDATE (not adopted — CIs cross 0). Mechanism label REVISED by the Q3
   placebo audit (same day): the OOS value is LONG-ALT EXPOSURE in deep bull; the return_1d
   ranking increment is unproven OOS.** Mode wired: `BULL_DEEP_MODE=mom1d_long` (+`BULL_DEEP_THR=0.15`, K=2, gross 0.5,
   shared DD-stop/gate; default flat). Atop KEEPSET4, clean preds, paired day-block CI: recent
   Δ+0.04 Sh / +379 (CI [−0.3,+0.9]); OOS Δ+0.09 Sh / +860, bull book +1,531, maxDD −15.4k→−14.6k,
   side carry −676. Bull-regime attribution positive in BOTH windows — unique among the 4
   bot-tested candidates. **Q3 robustness audit (1000-seed persistence-matched placebo, stateless
   K=2 book on ALL 1,378 OOS deep cycles; the bot entered 749 of them (its PnL-divergence
   footprint is 979 cycles incl. 230 non-deep carry cycles); controls;
   16-cell band; scripts `live/q3_deepbull_placebo.py`):**
   - OOS ranking claim FAILS: signal +62.3k gross vs random-alt placebo median +53.8k, rank p79,
     exact p=0.215 — unproven at K=2 on the stateless book ('unproven', not 'absent': point
     estimate ≈ +6 bps/cycle, within seed noise). Exposure claim STANDS: random alts +53.8k gross vs
     BTC-long +26.9k (committed-script control; results-review independent recomputation
     +25.6k — ≈2:1 either way) vs beta-proxy-ranked +37.2k. OOS top-episode share 41.7%; jackknife
     never negative [+36.3k, +65.7k].
   - Recent (descriptive-only, 6 episodes/47 cycles): ranking p98 (p=0.023) but single-episode-
     load-bearing — drop-one-episode flips negative [−1.9k, +8.2k].
   - The yearly table (+149/+26/+56/+63/+84/+245) had NO random-alt control → it survives only as
     era-robustness of the EXPOSURE, not the ranking. "mom30 died 2026 at −374" establishes mom30
     is harmful, not that mom1d ranking adds value.
   - Band: recent exactly linear in gross (+757/unit ×3, no peaked optimum); OOS tuning-sensitive
     (cells +470..+5,318; gross NON-linear per-unit 1,343/1,720/3,657 = state-path fragility).
     K1_g50 OOS +5,318 is a full-bot path statistic at a K the placebo did not test — logged, NOT
     chased; center (K=2, g=0.5, thr=0.15) kept per pre-registration.
   - Forward counterfactual PRE-REGISTERED: episode-frozen persistence-matched random-pick book,
     K=2, seeds 1-1000, statistic = cumulative signal-minus-placebo-median bps on the forward
     ledger; ranking wording upgrades only if the forward ledger separates, never from these
     windows. 2022 already SPENT on the final stack (C cell descriptive: Δ+123 vs B, maxDD
     guardrail passed); the ranking question has forward-ledger recourse only.
2. **Side long-leg conditioning — TESTED 2026-07-07 via per-leg regime discovery, REJECTED at bot
   level.** Per-leg framework battery (REGIME_DISCOVERY_FRAMEWORK, `--edge-col`) sharpened the
   target: long leg farms ONLY side_down; low-rvol longs lose in ALL periods (t −3.4). But both
   bot implementations fail atop KEEPSET4 (clean preds, paired CI): long rvol-floor 0.33 = OOS
   Δ−0.07 Sh, negative every OOS period; bull_mild short-only book (short_btc_hedge + DEEP_THR
   0.15) = OOS Δ−0.04, bull-regime Δ −1,697 vs base, full-stack Δ era-flipping. Diagnostic buckets with period-t
   up to 4 do not survive costs/hedge/carry at K=1/2. Do not reopen without new data or a cost
   structure change.
3. **Side-flat skip retest — DONE 2026-07-07 (canonical: clean v4 preds, KEEPSET4 base, thr band
   {0.03, 0.05}, paired day-block CI). Verdict: RISK LEVER ONLY, alpha claim REJECTED.**
   Mean lift NOT significant in any cell (all 4 paired CIs cross 0: ins sf05 Δ+1.37 bps/cyc CI
   [−4.6,+7.5]; oos sf05 Δ+0.17 CI [−1.6,+1.8]); threshold-sensitive (sf03 hurts OOS Δ−0.14 Sh,
   sf05 ≈flat Δ+0.04) and era-flipping (ins Δ 2025 −4.5k / 2026 +6.6k; oos 2023 −5.7k / 2024 +6.3k)
   — same tuned-continuous-parameter pathology as K3/decay-weights. The ONE robust effect: **maxDD
   improves in all 4 cells** (ins −11.8k→−3.9k sf05 / −5.4k sf03; oos −15.4k→−8.6k / −9.9k).
   Confirms the framework call (side_flat = FRAGILE/noise): variance removal, mean uncertain. If
   ever adopted, adopt as DD-reduction with zero expected mean lift; do not tune the threshold.
   **Q6 follow-up (2026-07-07, atop the CURRENT stack KEEPSET4+deep-overlay, full-bot dose-matched
   placebo, 50 seeds/window, generator committed `live/q6_gen_skipsets.py` after a first
   under-dosed placebo was voided by results review): the DD lever is real but is mostly an
   EXPOSURE-DOSE effect** — a random contiguous side de-gross of equal dose (581/1,931 suppressed
   entries) replicates ~79/83% of sf05's DD improvement (recent +7,918 vs placebo median +6,230;
   OOS +6,755 vs +5,586). Placement-specificity: recent beats p90 marginally (48/50, exact
   p=0.059; single-episode load-bearing — one Jan-Feb 2026 episode is 61% of baseline DD); OOS
   INCONCLUSIVE (41/50, p=0.196). Joint verdict (pre-registered vocabulary): **DD lever,
   mechanism unresolved.** Threshold 0.05 inherits the §6.3 in-window sweep — the placebo controls
   placement, never selection; only the forward window tests the threshold. sf05 cell numbers atop
   current stack: recent +22,624 / maxDD −3,869; OOS +105 / −7,800 (PnL deltas descriptive only —
   no alpha claim, per pre-registration). Adoption decision DEFERRED until after the §7 gross-cap
   forward confirmation window; 2022 behavior of this lever is permanently uncharacterized.
4. **Positioning/crowding data acquisition** (funding is the free proxy; OI/liquidations/borrow the
   real fix) — the only route to pricing the bull squeeze tail. First OI/LS screen done (positioning
   battery 2026-07-07, REGIME_DISCOVERY_FRAMEWORK): bull AVOIDs merely confirm bull0; side AVOIDs
   retracted (coverage artifact); nothing actionable yet. Construction-layer use only
   (regime-conditional sign makes it unusable as a model feature).
Closed (do not reopen without new data): published-alpha features (4 frames + oracle controls),
regime-conditional feature tuning (14-config, 0 pass; L2 shrinkage beats pruning), feature pruning
(in-sample gains vanish OOS), side-sign PIT detectors, dynamic K, K_short=3, rr_both, base_both,
sizing/gross variants, vol_target.

## 7. PRE-LIVE CHECKLIST

- [ ] **BLOCKING — 2022 holdout FAIL consequence (pre-registered, RESEARCH_LOOP_20260707 Iter 4
      F10): live gross capped at 0.5× on BOTH books (shared risk layer; `GLOBAL_GROSS_MULT=0.5`,
      wired in bot + run_convexity_v4_live.sh)** until the forward ledger independently confirms
      the bear farm: BEAR_MODE=equal bear-regime NET PnL (bot's own labels) over ≥2 calendar
      months of forward data that include ≥1 bear episode, with day-block 95% CI excluding 0 on
      the bear book. Months without bear cycles do not advance the clock. Basis: 2022 bear net
      −8,949, CI [−17,208, −807]; rule i −13,339 vs bar −5,839. (Confirmation criterion pinned
      here, before any forward data exists.)

- [ ] Stale-print eligibility gate in the bot (trailing zero-return-frac >10-20% over 30d →
      ineligible). Currently handled via `hl_*_clean` pred files; LITUSDT-class names otherwise
      pass dvol30 and entered ~10% of picks.
- [x] 2022 window run — SPENT 2026-07-07: FAIL (§1 annotation; consequence = §7 gross cap).
      One-shot law: no repeats.
- [x] **v4 paper wiring DONE 2026-07-07** (`run_convexity_v4_live.sh`): KEEPSET4 + deep-bull
      mom1d_long overlay (§6.1 forward-test), v4 residual-target artifacts
      (`train_v4_artifact.py` → `convexity_v4_{base,residrev}_model.pkl`, matched-cut parity
      1.000 vs research books; `predict_v4_incremental.py`), preds seeded from
      `hl_tgt_res_*_clean`, state bootstrapped via replay (exact reproduction of the validated
      cell: +2.26 / +19,628). State dir `live/state/convexity/v4_live/`. NB canonical KEEPSET4
      runs at bot-default `SIDE_BETA_NEUT=1` — the fitted v3 script's `SIDE_BETA_NEUT=0` is a
      fitted-stack knob, NOT part of KEEPSET4 (flipping it moved the recent replay +2.26→+2.93,
      untested/unvalidated — do not adopt silently). v3 live script NOT yet updated (v3-frame
      deep-bull confirmation passed: recent Δ+0.66 / OOS Δ+0.27, bull-attributed, every OOS
      period ≥ 0 — pending user decision).
- [ ] Launch the v4 forward loop (`tmux new -d -s cvx4 'bash live/run_convexity_v4_live.sh'`) —
      needs a box where the panel refresh pipeline runs (this box: FAPI geo-blocked, funding
      ingest will WARN; maturity_meta built from panel history as fallback).
- [x] Forward monitoring TOOL built + blind-validated 2026-07-08: `live/tip_exceedance_monitor.py`
      (per-leg exceedance-CUSUM, baselines/thresholds calibrated on 2023 side cycles only at ≈1%
      false-alarm/yr, evaluated blind 2024-26). What it can and cannot see: squeeze-rate surges
      DETECTED (5 alarms incl. 2024-06 and 2025-02, ~4-6 week lags into the bad eras); jackpot
      surges DETECTED (2025-10, ~4-month lag into 2025H2); **long-leg inversion NOT detectable**
      (m1/m4 zero alarms — the 2024/2025H1 body shift is beneath any honest-false-alarm detector,
      confirming the §5 noise-floor conclusion at event level). Output feeds de-gross + human
      review only, never auto-switching. Baselines frozen: p0(jackpot L>+500)=0.116,
      p0(squeeze S<−500)=0.054, p0(L>0)=0.441.

## 8. INSTRUMENTS & METHOD (validated this program)

- **Label/feature A/B estimator law (2026-07-08, from the beta-label A/B post-mortem)**: a paired
  replay through the equity-sigma stop + binary regime gate is NOT a valid variant-effect
  estimator — near-identical preds (corr 0.995) bifurcated the stop/gate paths and manufactured
  +6.6k/+5.8k deltas from zero ranking improvement (rank-IC Δ ≤ 0). Primary endpoints for any
  label/feature variant: BOOK-LEVEL per-cycle rank-IC (with t) + top/bot-K selection spread +
  regime/|BTC-move| splits; full-stack replay is secondary and only with overlays frozen or
  disabled. Beta-window family: tested (0.5/0.5 shrunk), KEEP INCUMBENT, closed.
  **Corollary (2026-07-08, feature-window results review)**: paired per-cycle deltas are only
  paired if BOTH arms are scored on the same per-cycle symbol population (incumbent ∩ variant ∩
  fwd) — a 0.7-symbol mean mismatch (variant train-row minimums dropping symbol-folds) flipped
  one verdict and one CI sign. `score_variant_cell.py` fixed; never score arms on per-arm
  populations.
- **Feature-window program (2026-07-08, addenda 6b-7): CLOSED, 0/6 promotions.** C1 ret_36h,
  C2 ret_6d, C3 resid_ret_3d, C5 corr_12h → KEEP incumbent; C4 dd_from_high NO SWAP (power
  outcome — the −2 bps non-inferiority margin was unpassable at the estimator's noise scale;
  parity question unresolved); T1 taker_ls addition rejected (matched OOS Δ −0.0003; univariate
  t=23.6 absorbed by the Ridge). Nearest miss C3: matched OOS Δrank-IC +0.0016, CI excludes 0,
  hit 21/33 — but selection-spread Δ negative in both windows + recent era-flip → non-promotable.
  Approved closing: no promotable variant AMONG THE TESTED CELLS at the pre-registered bars;
  V0_LEAN stays frozen (decision scoped to the candidate set — Pack 6 liquidity unscreened).
  Pre-registration COMPLETED same day: endpoint 3 (|BTC-move| quintile split) computed for all
  cells (2/12 Q4 CIs exclude 0 vs ~0.6 expected — weak; C3's recent lift concentrates in big-move cycles,
  mechanism-consistent, still non-promotable) and the 6b matched control books built for C2/T1
  (honest Δ vs control ≈ 0 all four window-cells — ret_6d and taker_ls confirmed information-free
  over V0_LEAN on the pre-registered instrument). No verdict changed; improvements,
  if any, are <~0.002 Δrank-IC or in endpoints the bars reject ("locally optimal" NOT claimed
  — bounded, not confirmed). Canonical: live/FEATURE_TUNING_RESULTS.md.
- **Window×horizon program (2026-07-08, addenda 8-8d): CLOSED, 0/3 Phase B cells.** Sleeve-
  conditional screen (21 windows × 5 residual horizons, V0-span-orth flag column, horizon-length
  blocks, marginal 24h→h labels) found 3 real screen-level ridges; all die at book level:
  B1 +ret_24h — flag was the residual of a near-duplicate of deployed return_1d (corr 0.995),
  model-inaccessible microstructure, NOT absorption; B2 +dd_3d — noise-floor null; B3
  resid_ret_3d@72h-sleeve — REJECT, selection-spread Δ non-positive BOTH eras (rec entirely
  negative, block-robust), OOS rank-IC lift real but thin and streak-concentrated. New law
  corollaries: check screen flags for near-duplicate parentage before spending a cell; rank-IC
  lift without spread conversion is not tradeable. Canonical: live/WINDOW_HORIZON_RESULTS.md.
- **W1 training-label winsorization (2026-07-09, addenda 9-9c): KEEP INCUMBENT — the most
  informative null on record.** Clipping the training xs_z at ±2 (4.95% of rows = 38.6% of label
  variance; extreme-leverage removal in a linear model) lifts book-level rank-IC by +0.020..+0.024
  (CI-solid, 42/42 folds+months, horizon-robust, not vol-tilt, identity-arm bit-exact) — TEN
  TIMES the program's usual effect scale — and STILL fails promotion: K-spread Δ −19/−9 (CIs
  cross 0), top-of-book pred gap flattens ~40%, tip picks change ~half of cycles. Measured
  proof that book-level rank-IC and production-K tip value are different quantities that can
  move oppositely under a treatment; the tip endpoint stays the only verdict-bearing one.
  Open design question before any follow-up: does anything in the stack consume book-level
  ranks? (If yes, the winsorized model's ordering edge has a consumer worth one cell.)
  Canonical: ledger addendum 9c; books hl_winz2_* kept as diagnostic state.
- **S1 correlation-aware short selection (2026-07-09, addenda 10-10c): KEEP INCUMBENT.**
  Selection-layer cell (no retraining): short-2 = less-correlated of ranks 2-3 to short-1
  (trailing-180c residual corr, PIT). Dose delivered (pair corr −40%) but the short-pair joint
  tail didn't move beyond random de-concentration (OOS 129→115 events vs matched placebo p5 114;
  CI crosses 0; worst-decile flat-to-worse). Durable mechanism finding: **the joint squeeze tail
  is regime-driven, not twin-driven** (rank-3's marginal squeeze rate = rank-2's; always-swap
  ceiling 7/129 events) — within-pool selection cannot de-concentrate it; regime-level (CUSUM
  throttle) and structural (K) levers remain. Canonical: ledger addendum 10c.
- **M1 pooled-LGBM model-class cell (2026-07-09, addenda 11-11c): REJECT; axis closed at fixed
  features.** Pooled LGBM (pinned params, 5 seeds, weights preserved, tripwires clean): REC
  K-spread Δ −89.5 bps/cyc CI entirely negative (long leg −67.5 CI-solid) → REJECT clause; OOS
  rank-IC +0.0073 excl 0 but 2025-concentrated. Decomposition arm attributes everything:
  pooled Ridge lifts rank-IC +0.024/+0.013 (CI-solid, 4th confirmation of the W1 law — never
  converts at tips); the NONLINEARITY increment is negative (LGBM vs pooled Ridge rank-IC
  −0.0176 CI entirely negative REC). Pooling is the lever; trees subtract; neither trades.
  Stage-2 shortlist cell not auto-skipped (formal gate) but premise damaged — escalated.
  Canonical: ledger addendum 11c.
- **Two-stage tip-reliability investigation consolidated in live/TIP_RELIABILITY_RESULTS.md**
  (architecture rationale: why MSE ranker + deterministic stage-2, not end-to-end; rank-IC≠tip
  value 4× confirmed; stage-1 score = W1; stage-2 = SQ1/SK1/wider-pool below).
- **SQ1 crowding→squeeze predictive test (2026-07-09, addenda 14-14c): FIRST POSITIVE — real
  orthogonal signal, monetization open.** Positioning ratios (global + top-trader long/short)
  predict which shortlisted name squeezes, OOS, INCREMENTALLY over the ranker's own pred:
  AUC 0.584→0.605 (Δ +0.020, beats crowding-shuffled permutation p95, 20/31 folds). NOT funding
  (drop −0.001) — it's the L/S positioning ratios (ls_ratio drop −0.022). The free-data crowding
  channel is OPEN. BUT: modest (AUC 0.60, 1.26× precision), fold-noisy, and predictive≠tradeable
  — monetization blocked on reorder (ceiling-dead) and gate (lethal class); escalated. Canonical:
  ledger addendum 14c; build_crowding_panel.py + sq1_crowding_predictive.py.
- **SK1 crowding-skip monetization (2026-07-09, addenda 15-15c): REJECT — signal real, doesn't
  trade.** Discrete skip of predicted-squeeze shorts: BEATS the matched-count placebo OOS (events
  850<p5 872, Sharpe +0.67>p95 +0.63) but FAILS the recent forward holdout (events within band,
  Sharpe −0.08 vs placebo +0.67 — worse than random; skip rate 36% vs OOS 17% = non-stationary
  crowding→squeeze map). Dual-window requirement caught the non-generalization. SQ1's orthogonal
  signal is real OOS but unmonetizable on free data. **Tip axis exhausted (W1/S1/M1/K1/SQ1/SK1);
  levers now = forward ledger, execution, paid positioning-depth data.** Canonical: addendum 15c.
- **Pipeline audit (2026-07-08): NO look-ahead anywhere in the v4 chain** (label β shift, target
  z-scoring, all 14 V0_LEAN features PIT with |IC| ≤ 0.057; CLAUDE.md's historical bug classes have
  no cousins). Fixes applied same day (definitions unchanged — validated artifacts remain valid):
  live loops now pass `--rebuild-days 10` (late-kline backfill + xs-tail repair propagation);
  `predict_v4_incremental` floors its recompute at the artifact fit_cut (no in-sample overwrite of
  the PIT seed / forward ledger); `bars_since_high` truncation guard + thin-cross-section DEFER
  guard for its xs_rank (the two live/backtest parity hazards). Known open (queued for next
  retrain, definition-changing): label β-window A/B (1-day β churn ≈ 5.9% of label variance,
  ~11% in top-decile |BTC-move| bars — test pinned β_5d and shrunk variants through the matched-cut
  harness); `_fill_grid` in the label/beta close path; bars_since_high cap at source. Audit trail:
  RESEARCH_LOOP_20260707 addenda 3-4.

- **Cost engine**: FEE_BPS_FILL added to cost_of (was slippage-only — all pre-2026-07-06 net numbers
  overstate by ~0.3 Sh); accounting identity exact; replay deterministic; funding 0.5×8h/bar correct.
  Tier caveat: cost_10k ≈ $1M AUM; cost_50k ≈ −0.2 Sh further.
- **tip_accuracy_v2.py**: calibrated tip screen (paired-diff block-bootstrap CI, concentration/
  halves/placebo on the diff, production-mirrored selection, regime split). 5-case ledger: zero
  false-PASS; blind spot = construction-layer interactions → verdicts are screens, never adoption.
- **tuning_harness.py**: locked dual-window (recent+OOS) bot harness for env-override cells.
- **Validation ladder** (mandatory): tip screen → full-stack replay at fair fees → OOS (2022
  holdout spent 2026-07-07 — the forward ledger is the remaining out-of-era test). Rules:
  median+concentration+halves next to every mean (three mean-mirages caught); no argmax-of-variants
  on a shared window; bit-identical-to-vanilla cells are NO-OPs not null results; per-cycle Sharpe
  on 24h overlap is ~2.4× inflated (comparative use only); Sharpe-on-negative-mean never judges a
  stop (loss/DD/tail do).

---

## Appendix — superseded findings (compressed; full history in git)

- Gross zero-tuning baseline (rr_both, all-rows): +3.62 at K=1/2 — reproduced exactly, but −25% on
  the eligible universe and rr_both later rejected at the strategy layer. Superseded by §1.
- Gates-off NET tables (pre-fee +2.01/+1.73; fee45 +1.70/+1.42/+1.43/+1.23) — layer-dependence of RR
  mapped (naive book and gate stack disagree by construction). Superseded by full-stack cells.
- Full-stack 2×3 matrix (fee45): ret two-book +2.68 / res two-book +2.68 / ret rr_both +1.59 /
  res rr_both +2.38; res base_both +3.29 (window artifact, reverted). State: `v4_ab_fee45/`.
- Feature work: +alpha191 ΔIC survivors −0.67 net (rejected); prune {ret_3d, autocorr} +0.64
  in-sample → +0.00 OOS (rejected); I1 alpha re-evaluation 0 PASS / 6 NEUTRAL (tip framework);
  oracle controls: I1 pipeline cannot transmit 24h-horizon info (4h-label bottleneck).
- E-section K_short study: KS 2→3 wash end-to-end (+0.07 OOS / −PnL recent). Superseded by §3 K verdict.
- §F stripped-vs-full: fitted gates = 3× capital preservation OOS; bear-ramp/BEAR_K = overfit.
  Superseded by the lever ledger (§4) with correctly wired cells.
- §G October attribution: vanilla ex-Oct negative; gated ex-Oct +1.65 (partly config-fit). Absorbed
  into §5 root causes.
- Old flow-doc rationale "+40bps@4h → −21@24h bull reversal": does not replicate on clean books
  (−8@4h). Corrected in CONVEXITY_V4_FLOW.md.
