# Convexity Mining — Pre-Registered Plan v1 (2026-05-19)

## Origin (validated, not assumed)
Forensic on the production record (`research/convexity_forensic_2026-05-19/`):
- The model `pred` does NOT select the convex winners (corr(pred,PnL)=−0.03;
  hit 50.7%; convex-winner pred-pctile 0.45 vs 0.48). The +2.23 was never
  forecasting.
- **A PIT signature identifies the convex-winner cohort and GENERALIZES
  out-of-universe**: OOS-time AUC 0.71, **OOS-symbol AUC 0.68** (5/5 held-out
  groups 0.63–0.72), label-shuffled placebo 0.55. First result this session
  to pass the out-of-universe gate. Signature loads + on
  atr_pct/idio_vol/name_idio_share/factor_loading, − on trailing
  skew/kurt/max ("primed, not yet popped").

**Identification ≠ profit.** This plan tests whether that portable cohort
signal converts to (a) genuine execution-cost savings and (b) a monetizable
direction net of realistic frictions — or whether it's only a volatility
detector. No pre-written conclusion.

## Hard priors carried in (must not be re-discovered the hard way)
1. **Long-skew is negative-EV a priori** (lottery/MAX premium, Bali/
   Brunnermeier; our own ex-VVV/unseen −0.33). Naive "long the convex names"
   is tested only to *refute*; the literature-positive side is *sell-
   convexity / post-pump reversal + short carry*.
2. **Frictions are the binding question**, worst exactly on convex/low-float
   names (spread, borrow, funding-squeeze). Realistic cost is first-class,
   not an afterthought.
3. **Portability is the gate.** Anything claimed must survive the R3c
   protocol (group-disjoint, no `sym_id`, unseen symbols, beta-neutral,
   costed). In-universe numbers are diagnostics only.
4. **Power-limited** (~5 disjoint-universe replicates / ~0.74y). MDE-in-
   Sharpe pre-computed; a null is "no-detectable", never "exhausted";
   gate must be reachable.
5. No goalpost-moving; a prediction miss rewrites the diagnosis, not the
   gate; v1 superseded files preserved; honest same-day record.

## Locked parameters (pre-registered)
- Panel `outputs/vBTC_features/panel_variants_with_funding.parquet` (R0-clean
  `target_A`). Signature features = the 12 PIT cols from the forensic
  (`atr_pct, idio_vol_1d_vs_bk, idio_vol_to_btc_1d, idio_skew_1d,
  idio_kurt_1d, idio_max_abs_12b, name_idio_share_1d, name_factor_loading_1d,
  funding_rate_z_7d, return_1d, dom_change_288b_vs_bk, corr_to_btc_1d`).
- Leak guard: denylist (`target,alpha,realized,basket,_fwd,btc_target,
  demeaned,return_pct,xs_alpha`); blocking assert
  `max|rankIC(sigfeat,target_A)|<0.10`; prefix-causal recompute; label uses
  realized fwd (necessarily) but every feature strictly prior to entry.
- Eval = R3c portable protocol (group-disjoint 5 groups, seed 20260519,
  no `sym_id`, beta-neutral fwd, full sleeve stack, cost) reused verbatim;
  signature classifier trained OUT-of-fold AND OUT-of-symbol (never on the
  evaluated rows).
- Costs (all reported): flat 4.5 bps; flat 9 bps; realized √ADV; tail-
  stressed (3× √ADV on top-vol-decile legs); **short-borrow/funding-squeeze
  proxy** added to short legs of low-float names (size ∝ idio_vol pctile).
- Stats: moving-block bootstrap (block=11), one-sided LCB, N_eff +
  correctly-scaled MDE-in-Sharpe pre-computed as a BLOCKING step; LOFO
  single-fold sign-flip kill.

## Tests (pre-registered; absolute, falsifiable gates)

### C0 — Specificity & integrity (BLOCKING — decides what the signal *is*)
Build OOS signature for three labels: (i) big-positive leg (current),
(ii) big-negative leg, (iii) big-|contribution| (vol detector). Report
OOS-time + OOS-symbol AUC for each + block-placebo.
- **Pre-registered:** if AUC(pos) ≈ AUC(neg) ≈ AUC(|abs|) (within ±0.04) ⇒
  it is a **volatility/lottery-name detector**, NOT a positive-convexity
  signal ⇒ the only admissible use is cost-concentration (C1); direction
  must come from elsewhere (C2 long arm pre-declared dead). If
  AUC(pos) − AUC(neg) ≥ +0.06 ⇒ positive-convexity-specific (C2 long arm
  is a live hypothesis). Either way C1/C2 proceed with the correct framing.
- Integrity: placebo AUC ≤ 0.55; leak asserts pass; else re-initiate.

### C1 — Cost-concentration (the user's core question, direction-agnostic)
Gate entries to top-Nσ signature names per side, N∈{1,2,3}, vs baseline
production (rolling-IC K=3 and all-eligible K=3). OOS signature, R3c
portable eval, all cost models.
- **Pre-registered gate (absolute):** a "win" iff, at some N, **cost/turnover
  drops ≥ 40%** vs baseline AND **portable net Sharpe ≥ baseline portable
  Sharpe − 0.10** (i.e., we cut cost ~for free by removing ~0-EV legs) AND
  survives tail-stressed cost. Prediction: turnover drops sharply (few
  convex names/cycle); Sharpe roughly preserved or improved (we delete
  coin-flip high-cost legs). A miss (Sharpe drops materially) ⇒ the deleted
  legs carried hidden value ⇒ rewrite diagnosis.

### C2 — Direction layer (monetization, literature-anchored)
On the convex cohort only, BTC-beta-hedged, three pre-registered rules:
`D_short` (sell-convexity / post-pump reversal + short carry — the positive-
EV prior side), `D_long` (naive — pre-declared likely-dead, tested to
refute), `D_trend` (long only under a trailing-trend filter, else short/flat).
Eval = R3c portable, net of realized √ADV + tail-stressed + short-borrow
proxy; block-bootstrap CI; LOFO; MDE pre-checked.
- **Pre-registered gate:** a rule is real iff portable net-of-realistic-cost
  Sharpe ≥ **+0.5** with one-sided LCB > 0 AND no LOFO sign-flip AND ≥ 4/5
  held-out groups positive AND survives the short-borrow + tail-stressed
  cost. Else: cohort is identifiable but **not monetizable net of frictions**
  at 4h/free-data (honest negative; C1 cost-efficiency may still stand
  alone). If MDE_Sharpe > +0.5 ⇒ "no-detectable", never "exhausted".

### C3 — Synthesis & decision (written only after C0–C2; no pre-write)
Sized: cost-efficiency value (C1) · direction premium net frictions (C2) ·
portability. State exactly which gate passed/missed and by how much; rank the
single highest-value next step from observed effect sizes. Fold into the
overall lever menu.

## Process
This plan → **3-agent plan review** (methodology / profitability / red-team)
→ revise to alignment before C0 runs. After C0–C2 → **3-agent results
review** vs these pre-registered gates; any fudge/leak ⇒ re-initiate that
test. Reuse validated machinery (`phase_ah_sleeve`, `R1_baseline_frontier`,
`R3c_portability_proper`) to avoid new bugs.

## Out of scope
Return-direction forecasting (closed, IC≈0.02); the OI/flow arc (separate,
running); paid data; live deployment. No claim closed by doc citation —
only by measurement here.
