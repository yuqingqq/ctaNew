# Orthogonal OI + aggTrade-Flow — Portable Test Plan v1 (2026-05-19)

## Question (fixed)
Does adding **orthogonal positioning data** — Binance `metrics` OI/long-short
and aggTrade order-flow — on the **full 51-symbol** universe lift the
**portable** alpha-residual number (the binding constraint from the bottleneck
arc: R3c unseen-symbol Sharpe), when fed as features to the production LGBM
V3.1 construction **and** a linear model? Deliver a sized, leak-guarded,
portability-gated answer; fold into the bottleneck B3 "orthogonal-data" row.

## Reconcile (CITE, do not re-derive)
- linear-line `PROGRESS.md` §5-INT (2026-05-19): order-flow / OI / spot-perp
  interactions, Tier-B **19-sym**, **in-universe β-residual *linear*** →
  *"add no tradeable signal."* CLOSED for that scope.
- **Incremental novelty this arc adds (the only reason to run it):** (1) **full
  51-sym** coverage (vs 19); (2) **raw OI/flow features** (not interaction
  terms); (3) the **portable R3c gate** — never applied to OI/flow; (4) both
  **LGBM V3.1** and **linear** arms (user observation: flow helps linear).
  If a 3-agent review judges this not materially incremental over §5-INT, the
  arc is dropped before the heavy fetch.
- Bottleneck arc: WINNER_21 feature ceiling ≈ exhausted (univ IC max 0.036;
  B★ superset Δ −0.58 portable); OI/flow are *orthogonal data*, the one
  un-refuted lever — this arc tests exactly that, portably.

## Data scope & fetch (only after plan alignment)
- **OI**: `data_collectors/metrics_loader.py`, all 51 (23 cached + 28 fetch).
  Cheap. Features (oi_panel schema): `oi_chg_{1h,4h,1d}`, `oi_z_{1d,7d}`,
  `oiv_z_1d`, `ls_count_{z_1d,chg_4h}`, `ls_top_z_1d`, `ls_taker_{z_1d,chg_4h}`.
- **aggTrade flow**: 25 cached + **26 missing** (AAVE ASTER AXS BIO ENA ETC GMX
  HBAR HYPE ICP JTO JUP LDO ONDO ORDI PENDLE PENGU PUMP STRK TAO TON TRB
  VIRTUAL VVV WIF ZEC). Stream per-symbol-day: download Vision zip → 5m
  features via `features_ml/trade_flow.py` → discard raw (no 25–50 GB on
  disk). Heavy (~hours, background). Features: `tfi_smooth`,
  `signed_volume_z`, `vpin` (already PIT in trade_flow), `large_trade_*`,
  `kyle_lambda`, `avg_trade_size`, `vwap_dev_bps`.
- Panel range 2025-03-27 → 2026-05-06. Newer syms (PUMP/ASTER/HYPE/VVV/BIO/
  PENGU) may have short/absent history → per-symbol NaN, handled by existing
  listing-eligibility; per-symbol coverage reported.

## Leak guards (mandatory — the recurring failure mode here)
1. Hard denylist: exclude any col containing `target,alpha,realized,basket,
   _fwd,btc_target,demeaned,return_pct,xs_alpha`.
2. Blocking assert before any fit: `max|rankIC(new_feat, target_A)| < 0.10`
   for every admitted OI/flow column (evidence file written).
3. PIT: every OI/flow feature `.shift(1)`; OI/metrics merged to 5m bars by
   **backward `merge_asof`** (publish lag); VPIN uses trade_flow's PIT path.
4. R0-style prefix-causal recompute on the augmented panel at 3 interior cuts
   (max|Δ| ≤ 1e-4·std on a sample of new features) before B-runs.

## Arms (identical disjoint protocol/folds/seed=20260519 as R3c/B★ ⇒ paired Δ)
- **A0** WINNER_21 (no sym_id) — baseline (must reproduce R3c −0.33).
- **A1** A0 + OI features.  **A2** A0 + flow features.  **A3** A0 + OI + flow.
- Each through **(M1) production LGBM** and **(M2) Ridge** (standardized,
  per-fold train stats). Trained in lockstep (same rows/seeds; arms differ
  only in feature columns ⇒ Δ is pure feature effect).

## Metrics & pre-registered gates (decision = portable; miss ⇒ rewrite
## diagnosis, not gate)
- Primary: **portable R3c Sharpe**, paired **Δ(Ai−A0)** per (arm,model), with
  block-bootstrap CI on the per-cycle paired diff (block=11), N_eff, and the
  **correctly-scaled** MDE (not √CPY-doubled). Plus **LOFO** single-fold
  sign-flip kill.
- Secondary: pooled OOS top-K(=3) realized-`alpha_A` spread; in-universe
  V3.1 Sharpe (completeness only).
- Coverage control: report each result **full-51** AND on the
  **OI/flow-covered subset** (separate "no signal" from "coverage dilution").
- **PASS (orthogonal lever real):** some (arm,model) gives portable Δ ≥ **+0.5**
  with paired CI excluding 0 AND no LOFO sign-flip AND top-K spread not
  degraded. Sized prize = Δ.
- **FAIL (earned exhausted):** best portable Δ ≤ **+0.2** or paired CI includes
  0 → free-tier orthogonal data (OI+flow) does not raise the portable ceiling
  even at full-51 across both model classes — extends the closed §5-INT
  in-universe negative to the portable regime; recommendation = paid/alt
  orthogonal data or horizon change (explicit scope decision, not executed).
- If MDE > +0.5: report as effect-size estimate + "no detectable orthogonal
  lever", never a false "exhausted".

## Process (discipline, per user)
1. This plan → **3-agent review** (methodology / profitability / red-team):
   is it materially incremental over closed §5-INT? leak/PIT/coverage design
   sound? fetch cost justified? gate power-appropriate? → revise to alignment.
2. Only then fetch (OI first/cheap; flow heavy/background).
3. Build PIT panel → leak-guard asserts → portable test.
4. **3-agent results review** vs these pre-registered gates; leaky/fudged ⇒
   re-initiate. Honest synthesis → bottleneck `B3` orthogonal-data row.

## Out of scope
Re-running §5-INT interactions; in-universe-only conclusions; paid data
acquisition; deployment. (Bottleneck arc B★b continues independently.)
