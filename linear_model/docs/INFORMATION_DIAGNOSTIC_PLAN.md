# Information Diagnostic Plan (pre-registration)

Created 2026-05-18. Status: **LOCKED, D1 to run.** Supersedes the
momentum-gate line (which reached its contractual terminus at Step 93 — see
`MOMENTUM_GATE_PLAN.md` §10). Production LGBM is unaffected and unchanged.

## Motivation

The 76→93 arc established a real but weak, non-stationary, sub-cost
β-residual convergence pulse. Before proposing any new strategy, the owner
posed three critical questions, which form a **gated decision tree** — each
only matters if the previous one passes:

- **Q1 — Enough information?** Do the current features carry *any*
  net-of-cost tradeable 4h information, even under a best-case (stationary,
  no-memorization) extraction?
- **Q2 — Fully utilized?** If yes, how much of that ceiling do we keep
  out-of-sample (i.e. how much is lost to non-stationarity / extraction)?
- **Q3 — Catch the good ones?** Can we *ex-ante* identify which firings are
  profitable (the meta-label / selection question)?

Existing evidence (do not re-litigate): Q2 failure is strongly evidenced
(in-sample→nested collapse: Step-88 −1.05, Step-90 −2.60, monotone
worsening = non-stationarity). Q3 is strongly evidenced negative (Phase-DDI
per-cycle profitability R²≈0.005; Step-90 nested below random). **The one
decisive measurement never cleanly made on this line is Q1's ceiling.**

## The three diagnostics — D1 → D2 → D3, ordered, pure measurement

No trading strategy is adopted from these. They are measurements. Each
gates the next. All keep the loop-closed discipline (matched placebo where
applicable, honest verdict, no goalpost-moving).

### D1 — Information ceiling (answers Q1). RUN FIRST.

**Why random-shuffled K-fold, not in-sample-fit:** a model fit on the same
rows it predicts is degenerate — it memorizes row-noise, approaches the
oracle, and always looks huge (uninformative). Random 5-fold CV predicts
**held-out** rows (kills memorization) but interleaves folds in time (no
temporal-transfer penalty) ⇒ it measures *information content assuming the
feature→residual map is stationary* — the true best-case ceiling.

- **Panel:** Step-92 `dec` — hl42 (42 syms), OOS folds 1–9, 4h
  NON-overlapping decision grid (BLOCK=48). Non-overlap ⇒ no label-window
  overlap ⇒ random K-fold is leak-free here. Identical universe/grid/cost
  to the locked Step-92 (apples-to-apples with the whole line).
- **Target:** per-symbol-z `tz = clip(alpha_beta / sigma_idio, ±5)`
  (cross-symbol learnable; the production convention). **Positions and
  scoring use realized `alpha_beta`** so NET Sharpe is in Step-92 units.
- **Feature set F_core (primary):** the 21 strict-PIT panel features +
  engineered `s_t` (Step-92 PIT-audited). **Excluded as leak:**
  `return_pct`, `btc_ret_fwd`, `alpha_beta`, `exit_time`, ids.
  **Secondary F_core+OI:** add the 11 PIT OI features on the OI∩hl42
  (~23-sym) subset — context only, NOT gated (does positioning info raise
  the ceiling?).
- **Models:** (a) Ridge = linear ceiling; (b) LGBM (num_leaves 63,
  n_estimators 400, lr 0.03, subsample) = nonlinear ceiling. 5-fold
  shuffled CV, out-of-fold predictions. Ceiling = the better of the two.
- **Scoring:** `pos = sign(oof_pred)` per symbol/cycle, equal-weight; exact
  Step-92 `portfolio()` + VIP-0 cost (s64.COST). Report GROSS, NET,
  NET-Sharpe (ANN=√(365·6)) + block-CI, cost bps/cyc, turnover, %syms
  net+, pooled IC. Pred-weighted & top-k variants reported as context
  (not gated). HL-maker ~1 bps ceiling reported as context (not gated).
- **PIT integrity guard (mandatory, run before scoring):** re-run the
  Step-92 s_t audit (independent strictly-past recompute, corr 1.0);
  per-feature |corr| with forward `alpha_beta` reported, any > 0.10
  flagged (same bar as Step-92); assert forward columns absent from X.
  Random shuffling touches ONLY fold assignment — every feature/target
  stays strictly PIT per row.
- **PRE-REGISTERED DECISION (locked before run):**
  - F_core random-CV **NET Sharpe > +1.5** ⇒ information IS sufficient
    under stationarity ⇒ **proceed to D2**.
  - F_core random-CV NET Sharpe **≤ +1.5** ⇒ the features do **not** carry
    tradeable 4h net-of-cost information even assuming stationarity ⇒ the
    line is **information-bounded; STOP** (a definitive, stronger result
    than the current "sub-cost" finding — it proves no extraction fixes it).
  - **Threshold rationale (pre-stated):** the observed in-sample→nested
    haircut in Steps 87/88 was ~2–3× and sign-destroying; a stationary
    ceiling below +1.5 leaves nothing after the non-stationarity haircut.

### D2 — Utilization gap (answers Q2). Only if D1 > +1.5.

Same features/models, **time-nested walk-forward** (fit folds < k, predict
fold k — the loop-closed honest standard). Report **gap = D1_ceiling −
D2_nested**, plus the in-fold → full-train → nested ladder to localize the
loss. nested ≈ ceiling ⇒ well-utilized, bottleneck is information/new data.
nested ≪ ceiling ⇒ loss is non-stationarity; only D3 can rescue.

### D3 — Signal catchability (answers Q3). Only if D2 leaves a usable gap.

Pure predictability probe, **no trading, non-adoptable.** Nested-OOS
classifier on all PIT conditioning features predicting sign of
(pos·alpha_beta − cost) for Step-92 firings. Report nested AUC + the
cross-sectional IC of predicted-prob vs realized-profit. AUC ≈ 0.5 / IC ≈ 0
⇒ good subset uncatchable ⇒ terminus with full mechanistic closure. AUC
meaningfully > 0.5 nested ⇒ a gate is worth building and the §6
precondition reopens *with evidence*.

## Status / next action

LOCKED 2026-05-18.

**Step 94 (D1 v1, `94_info_ceiling_d1.py`) — INVALID, leaky design,
discarded (not PASS, not FAIL).** Naive random 5-fold shuffle leaks two
ways on this panel: (1) temporal autocorrelation (slowly-moving features +
autocorrelated target ⇒ each test row's t±4h same-symbol neighbor is in
train); (2) contemporaneous cross-sectional (whole timestamps split across
train/test ⇒ model learns market state at t from other symbols at t). Result
was the leak signature: LGBM IC **+0.376** / NET Sharpe **+23.71** (vs the
project's ">0.10 IC = suspicious" rule and honest 4h IC ~0.03); Ridge +2.37
(leak-inflated, CI barely >0). Gate "passed" mechanically but the test does
not measure information. Honest verdict: **D1 must be re-run leak-free
before Q1 can be answered.** (Symmetric-skepticism note: not ridden as a
pass, not over-dismissed — Ridge hints at *some* info; test must be clean.)

**Step 94b (D1 v1.1, gated) — RUN, leak-free, FAIL → Q1 = NO.**
Time-grouped (whole-timestamp) shuffled 5-fold + 1-day embargo. Leak check
confirmed v1 was pure leakage: **LGBM IC collapsed +0.376 → +0.010**, NET
+23.71 → −0.35. Clean ceiling on F_core: **Ridge NET Sharpe +0.62, CI
[−1.68,+3.14] (straddles 0), IC +0.026; LGBM NET −0.35, IC +0.010 (≈
noise).** Best +0.62 ≤ +1.5 ⇒ **FAIL.** F_core+OI both negative (OI
features hurt leak-free). s_t ref +0.29 (= Step-92, harness validated).

## CONCLUSION (2026-05-18) — line is INFORMATION-BOUNDED; D2/D3 moot

The gate (D1 > +1.5) fails ⇒ per the pre-registered tree, **D2 and D3 do
not run** (a sub-cost ceiling cannot be rescued by utilization or
selection). All three questions answered:

- **Q1 — enough information? NO.** Best-case leak-free *stationary*
  extraction is sub-cost (linear +0.62 CI-zero; flexible nonlinear ≈0/neg).
- **Q2 — fully utilized? Effectively YES (moot).** Honest realized
  (Step-92 +0.29; 76–93 arc ≈0) sits just below the +0.62 ceiling — small
  gap, and the ceiling itself is sub-cost. The in-sample→nested "collapses"
  were the non-stationarity penalty on *leak-inflated* numbers.
- **Q3 — catch the good ones? Moot.** No profitable subset exists to select
  from a sub-cost, CI-zero total signal (also: Phase-DDI R²≈0.005).

**Bottleneck = raw information content of free 4h Binance perp features —
not capacity, not extraction, not selection.** This *explains* the entire
76–93 arc. Caveats (symmetric): bounds the *current* feature family
(price/vol/funding/OI/dominance/beta/idio-vol), not literally every free
feature; Ridge +0.62 is a faint real-but-weak pulse, CI-zero & sub-cost.
**Levers with headroom = NEW orthogonal information.** The D1 bound covers
the *current panel family* (perp-OHLCV price/vol/funding/OI/dominance/beta/
idio — its volume features are **perp**, inside the failed ceiling). Two
untested **free** families remain: (1) spot microstructure (spot volume,
spot/perp volume divergence, spot CVD, basis beyond funding-carry) — free
via Binance Vision, not on disk; (2) perp aggTrade order-flow (VPIN/taker-
CVD; machinery in `features_ml/trade_flow.py`, perp aggTrades on disk, not
in this panel). Plus paid on-chain/cohort.

**Step 95 — D1-ext-A (perp aggTrade order-flow) DONE: gate FAIL, but the
cleanest marginal feature signal of the whole arc.** 6 PIT order-flow
features (PIT audit PASS, look-ahead |corr| max 0.014), hl42∩aggTrades = 20
liquid majors, same leak-free CV + same +1.5 gate, F_core vs F_core+oflow on
the same rows. F_core(20-sym) Ridge NET +0.46 → **F_core+oflow +1.09**
(CI[−1.20,+3.56], IC +0.026, 70% syms+, 6/9 folds); LGBM negative. Best
+1.09 ≤ +1.5 ⇒ **FAIL**. **But Δ = +0.63** (IC +0.018→+0.026, syms+
55→70%, folds 5→6/9) — first PIT-clean family to materially move the
leak-free ceiling the right way (IC *and* breadth *and* folds together).
Symmetric: not over-claimed (CI-zero, LGBM neg, sub-gate) nor over-dismissed
(largest clean marginal of the arc ⇒ line is NOT information-empty;
stacking orthogonal microstructure is the productive axis).

**Step 96 — D1-ext-B (spot microstructure, STACKED) DONE: gate FAIL ⇒
DEFINITIVE TERMINUS.** 6 PIT spot features (basis dislocation, spot CVD,
spot/perp volume lead, spot-perp ret lag; Binance-Vision spot klines,
egress OK; builder ms→µs timestamp bug found+fixed+rerun, PIT audit PASS).
Same leak-free CV + same +1.5 gate, same 20-sym rows; harness re-validated
(F_core +0.46, +oflow +1.09, s_t +0.51 = Step-95). Result: F_core +0.46 →
+oflow **+1.09** → +spot **+0.33** (spot marginal ≈ −0.13, slightly hurts)
→ stacked **+0.91 ≤ +1.5 ⇒ FAIL.** Spot adds no marginal info beyond
F_core+order-flow — `funding_rate` already proxies basis carry, perp
volume-z + order-flow already capture flow regime; spot/perp 4h divergence
on liquid majors is dominated by the systematic moves the BTC-β residual
already removes ⇒ spot **redundant, not orthogonal**.

## D1-ext-C (pre-registered 2026-05-18) — perp-vs-spot FLOW divergence + targeted interactions

Owner Q: spot *flow* vs perp *flow*, and more interactions. Motivated: flow
is the one real marginal of the arc (+0.63); spot price/vol/basis ≈0 (96/97)
but perp-vs-spot *aggression divergence* + flow×regime interactions are
untested. LOCKED fixed set (no sweep, one run): `fd_imb` = of_imb_1d −
sp_taker_imb_1d; `fd_absdiff` = |fd_imb|; `fd_prod` = of_imb_1d·
sp_taker_imb_1d; `x_flow_vol` = of_tfi_z1d·vol_zscore_4h_over_7d;
`x_flow_fund` = of_imb_1d·funding_rate_z_7d; `x_fd_st` = fd_imb·s_t. Block
added to F_core+oflow (+1.09 baseline), SAME leak-free CV + SAME +1.5 gate;
also univariate IC + standalone marginal of `fd_imb`/`x_fd_st` (anti
block-masking, Step-97 lesson). Prior guarded (leak-free LGBM already
auto-explores interactions, found none; spot flow here is kline-coarse not
aggTrade) — one clean shot, no iterate-to-pass. Stack >+1.5 ⇒ line reopens;
≤+1.5 ⇒ flow-interaction lever also exhausted (strengthens terminus,
pending aggTrade-granularity spot flow as the only remaining free probe).

**Step 98 — D1-ext-C RESULT: FAIL, block net-destructive.** F_core +0.46
→ F_core+oflow **+1.09** → +FLOWINT **+0.44** (Δ −0.65, variance inflation
erases the order-flow lift). Solo `fd_imb` Δ+0.03, `x_fd_st` Δ−0.07 (core
"leverage-led reverts harder" feature carries nothing — not block-masking).
Univariate IC all ≈0. Perp-vs-spot flow divergence + flow×regime/×deviation
interactions add no ceiling info at 4h (kline-spot granularity). Sole
remaining free probe = aggTrade-granularity spot order-flow (heavy,
guarded-low prior).

## D1-ext-E (pre-registered 2026-05-18, LOCKED) — structural-EVENT paradigm: crowding composites → forced-deleveraging events → return, hedged vs UNHEDGED

Owner thesis: reduced-form feature→return is information-bounded, but a
*structural* feature-composites→event→return paradigm (crowding/liquidation
risk) is untested; and the β-residual may hedge out exactly what these
events produce — so test hedged AND unhedged. LOCKED, one run, no sweep,
multiplicity-controlled.

- **PIT composites (fixed 6, from cached OI panel + panel price/funding,
  all already PIT):** `c_oi_x_ret`=oi_chg_1d·sign(return_1d);
  `c_oi_x_fund`=oi_z_7d·funding_rate_z_7d; `c_oi_own`=oi_z_7d;
  `c_div`=sign(return_1d)·sign(oi_chg_1d) (price/OI quadrant);
  `c_ls`=ls_taker_z_1d; `c_oi_accel`=oi_chg_4h−oi_chg_1d.
- **3 fixed events on the forward 24h window (label; features stay PIT),
  σ_sym = trailing-30-cycle PIT std of 24h returns, k=2 canonical:**
  E1 long-liquidation r_fwd24 ≤ −2σ; E2 short-squeeze r_fwd24 ≥ +2σ;
  E3 vol/deleverage forward-24h range/close ≥ trailing-90pct.
- **Stage 1 predictability:** leak-free grouped+embargo CV classifier
  (logistic + LGBM), nested OOF AUC per event, **Bonferroni ×3**
  (require corrected AUC bootstrap-lower-bound > 0.5, pre-state
  "≥1 event AUC>0.55 corrected-sig").
- **Stage 2 (only for Stage-1 passers), the structural fork — 24h
  non-overlap sampling, ANN=√365:** event-prob signal vs
  **(a) β-residual** fwd-24h (PIT beta_btc_pit·fwd-btc; look-ahead
  audit) — gate +1.5; **(b) raw** fwd-24h — vs **market-exposure-matched
  placebo p95** (NOT zero; raw crypto ~0.7-corr to BTC ⇒ unhedged is
  beta/market-timing, far more false-positive-prone). E3 is
  predictability-only (non-directional).
- **Pre-stated verdicts:** AUC≈0.5 corrected ⇒ crowding/events not
  PIT-predictable from free OI ⇒ closed. Predictable & lifts **residual**
  >+1.5 ⇒ reopens *this* line. Predictable & lifts **raw** > placebo-p95
  but NOT residual ⇒ confirms the β-hedge strips it ⇒ this line closed but
  a real, *separate, directional* (non-market-neutral, full beta risk)
  finding. Predictable & lifts neither ⇒ events real but unmonetizable.
  Designed to be TERMINAL for the event-structural paradigm.

## D1-ext-F (pre-registered 2026-05-18, LOCKED) — MAXIMAL feature set → structural events (closes the "OI-only" loose end)

Owner: Step 100 was OI-anchored; OI alone may not suffice — test the full
feature combination. Honest prior: D1 (94b) already bounded the full F_core
stack for the forward *return*; E1/E2 are 2σ thresholds of that same return
⇒ strong prior AUC≈0.50. But full-features→event-classification is a
distinct untested cell; one decisive maximal test removes the loose end.
LOCKED: feature set = strict SUPERSET of Step 100 = F_core (~24 panel PIT)
+ s_t + OI panel (11) + order-flow panel (6) + the 6 Step-100 OI composites
(~47 PIT feats, all already audited); **events = the 3 LOCKED Step-100
definitions UNCHANGED** (no event redefinition = anti-p-hack); no new
hand-crafted composites (LGBM auto-combines; Step-98 showed hand-built
interactions net-destructive). Same leak-free CV (whole-timestamp 5-fold +
1d embargo), logistic+LGBM, OOF AUC, timestamp-block bootstrap,
Bonferroni×3, same hedged-vs-unhedged Stage-2. One run, no sweep. Verdicts:
AUC≈0.50 ⇒ structural events not predictable from ANY free feature
combination ⇒ airtight comprehensive terminus. AUC>0.55 Bonf-sig ⇒ Stage-2
fork (residual reopens line; raw-only ⇒ separate directional finding).
The event-classification analog of D1; the last distinct free-data test.

## FINAL CONCLUSION (2026-05-18) — free-data line COMPREHENSIVELY & AIRTIGHT CLOSED

**Step 101 (D1-ext-F) — OI-only loose end closed:** maximal 47-PIT-feature
stack (F_core+s_t+OI+order-flow+composites, superset of Step 100) → 3
locked events: E1 AUC 0.508, E2 0.489, E3 0.502 — all ≈0.50, none
Bonferroni-sig. Structural events not predictable from ANY free feature
combination (not OI alone, not the maximal stack incl. order-flow, LGBM
auto-interactions). No distinct free-data test remains.

Closed across **both modeling paradigms**, hedged and unhedged-instrumented:

- **Reduced-form (feature→return), Steps 94b–99:** leak-free stationary
  ceiling sub-cost. order-flow = a real PIT-clean **+0.63** marginal but
  sub-gate (95); spot ≈0 (96–97); flow-interactions net-destructive (98);
  spot-led momentum absent (99). Full free stack ≤ +1.5.
- **Structural (composites→event→return), Step 100 (D1-ext-E):** the
  single best/most-mechanistic remaining hypothesis. 3 forced-deleveraging
  events × 6 PIT crowding composites, leak-free classifier, Bonferroni×3.
  **All AUC ≈ 0.50 (E1 0.481, E2 0.502, E3 0.506) — events not even
  predictable.** Failure is at Stage 1, *before* the hedged-vs-unhedged
  fork ⇒ not "the hedge strips it" — deeper: crowding is the *fuel*, the
  cascade *trigger* is an exogenous shock absent from PIT features
  (AUC≈0.5 is the expected result). Unhedged arm was instrumented; never
  triggered.

- **Q1 = NO (definitive, all free data, both paradigms).** Binding
  constraint = raw information content of *free* 4h crypto data.
- **Q2 / Q3 = moot.**

Honest, mechanistically-understood close. **Carry-forward:** Step-95
order-flow's clean +0.63 is the single genuine orthogonal-information
finding of the whole 76→100 arc — *flow-type* data is where
residual-predictive content lives. **Only remaining levers: paid
on-chain/cohort data (Glassnode, >11 cohort-Sharpe-spread bar) or a
different data domain/horizon — else the line is closed.** No strategy
adopted (D1–D3 + extensions were pure measurements). Production LGBM
unaffected throughout.

**Step 102 (D1-ext-G) — signed-composite consensus, confirmatory:** owner's
"combine all long/short composites, more agreement = more reliable?"
directly falsified — long-consensus hit 50.3% (coin-flip), |V|-monotonicity
ρ=−0.60 (MORE agreement → WORSE), fails matched placebo. Empirically
confirms (not just argued) the composites are correlated reads of one
sub-cost convergence signal. (Disclosed spec flaw: Z=1 mis-scaled for
raw-scale s_t → 4 convergence composites fired ~0%; impact-assessed
non-changing — they are the already-sub-cost Step-92 signal.) Terminus
unchanged.
