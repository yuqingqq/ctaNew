# Status — 2026-05-09

## TL;DR

Two programs. **xyz US-equity v7 shadow harness ready for forward-test.**
**Crypto v6_clean session was closed in the morning of 2026-05-08, but a new validated
finding (PM_M2_b1 entry gate, multi-OOS Sharpe +1.96 alone, +2.75 stacked
with conv_gate, hard-split survives) emerged in the afternoon — see
"v6_clean PM gate addendum" below before re-deciding crypto deployment.**

**2026-05-09 update: v7 architectural audit concluded.** 15 architectural
additions tested across two days (intraday Ridge head, intraday port-blend,
rolling-24mo blend, rolling regime-conditional with 5 indicators, pred_disp
risk-sizing, regime MoE retraining, disp_22d as feature, disp_pctile as
feature, xs_rank replace/add, ts_pct add, xs+ts combo, B_xs_add, F_xs_add,
ABF_xs_add). **All FAIL discipline gates** under full-ensemble validation.
Best candidate (gap-skip overlay) at +0.43 ΔSh fails 2/5 gates. Free-data
signal-level optimization on this universe is now decisively closed.
v7 alone (Sharpe +3.29, 533 cycles 2016-2026) remains production. See
"2026-05-09 architectural audit" section below.

| | crypto v6_clean h=48 | xyz v7 alpha-residual |
|---|---|---|
| Status | session closed AM; **PM gate finding PM** — see addendum | shadow harness deployed, awaiting cron + forward-test cycles |
| Last live activity | 2026-05-03 (3 closed cycles, last 2 negative) | not yet |
| Backtest Sharpe (corrected) | +0.6 active @ 4.5bps; +1.5+ at HYPE-staked tiers; **+2.75 with PM_M2_b1 + conv_gate (validated multi-OOS, NOT YET DEPLOYED)** | +3.11 active @ 3.5bps realistic; +3.25 @ 1.5bps maker |
| Universe | ORIG25 crypto perps (HL) | 11 xyz Tier A+B US-equity perps |
| Cadence | 4h | daily (avg ~7-8 trading days/cycle, gate-dependent) |
| Why closed AM (crypto) | edge thinner than originally claimed at corrected costs; awaiting fee-tier upgrades | n/a |

## xyz v7 status (the active program)

### Spec (locked 2026-05-08)
- **Universe**: tier_ab — 11 names (AAPL, AMZN, GOOGL, META, MSFT, MU, NFLX, NVDA, ORCL, PLTR, TSLA)
- **Position rule**: top-K=4 long, K=4 short, hysteresis exit at K+M=5
- **Cadence**: daily, 1d hold, dispersion gate ≥ 60-pctile of trailing 252d (PIT)
- **Cost assumption**: 3.5 bps/side realistic (slip 2.7 + fee 0.8)
- **Backtest**: walk-forward 11-fold OOS 2016-2026, +3.11 active Sharpe [+1.79, +4.49]
- **Hard-split**: train ≤2019 frozen → +1.67 Sh on 2020-2026 (vs original baseline -0.28)

### Operational layer (built this session)

| component | path | status |
|---|---|---|
| Model artifact | `models/v7_xyz_ensemble.pkl` (15 models, 18 features) | trained 2026-05-08 |
| Trainer | `live/train_v7_xyz_artifact.py` | mirrors crypto pattern; annual retrain mandatory |
| Daily bot | `live/xyz_paper_bot.py` (~930 lines) | end-to-end tested; defensive guards in place |
| Hourly monitor | `live/xyz_hourly_monitor.py` | tested; sends Telegram snapshot with $/bps P&L |
| Cron wrapper | `live/run_xyz.sh` | sources `.env`, auto-detects venv |
| Runbook | `docs/xyz_PAPER_TRADE_RUNBOOK.md` | 250+ lines |
| Telegram integration | `live/telegram.py` (shared) | confirmed working — rebalance + hourly messages |

### Defensive engineering (transactional / fail-closed semantics)

Built into `xyz_paper_bot.py` after multiple audit passes:
- **Atomic state save**: tmp + `os.replace` (POSIX atomic rename)
- **Pending-cycle-row in state**: cycles.csv append + state save are exactly-once via dedup-by-decision_ts
- **L2 book pre-check**: refuses to mutate state if any required position symbol's book is missing
- **`_require_full_fill`**: raises if simulated taker fill is partial (depth insufficient)
- **Same-day re-run guard**: state/P&L unchanged if rerun on same decision_ts
- **Insufficient predictions**: raises (not silent flat) if bar size < 2K+M
- **Feature-order validation**: load_artifact + build_panel verify feat_cols match meta
- **Predictions dedup**: skip duplicate append for same decision_ts
- **Funding checkpoint hold**: hourly_last_tick doesn't advance if any mid/funding fetch failed

### What's NOT yet done (forward-test prerequisites)

- ❌ Cron entries not installed (user task; commands in runbook)
- ❌ Zero forward-test cycles logged
- ❌ HL fee schedule not officially verified (using user-provided 0.8 bps/side)
- ❌ No risk overlay (drawdown brake, kill-switch)
- ❌ No real executor (shadow only; would be a separate file when promoted)

### Tier C names (excluded from tier_ab)

The basis-quality probe (`xyz_data_quality.py`, run 2026-05-08, all 15 candidates) flagged
4 names with concerning behavior:
- AMD: 4.3 min basis half-life, slow mean-revert
- INTC: 247 bps daily basis std, 6 single-1h moves >5% in 90 days
- COST: 19% frozen 3hr runs (illiquid)
- LLY: 14% frozen runs (recent listing)

Backtest at 3.5 bps/side: full15 gives +2.88 Sh; tier_ab (drops Tier C) gives +3.11. Tier C
hurts more than it helps. Re-run probe quarterly to track liquidity build.

## Research dead-ends (this session)

### Intraday redesign (definitively negative)

Goal: increase trade frequency above ~3 cycles/month. Tested:
- 30min cadence cross-sectional residual on Polygon 5m × 2y: composite IC -0.006
- Multi-horizon forward target (30m/1h/2h/4h): naive reversal IC up to +0.022 at 2h
- Hour-of-day filter (closing 30min only): IC +0.048 — real signal but small
- 72-config sweep (cadence × signal × horizon × K × M × cost): **all configs lose at 3.5 bps/side**

Root cause: gross intraday alpha 1-4 bps/cycle, vs minimum realistic round-trip cost 5-7 bps.
Hysteresis cuts turnover 1.44 → 0.20 but per-cycle gross is too small to clear costs.

**Conclusion**: cross-sectional residual on liquid US mega-caps cannot support sub-daily
cadence at xyz taker cost economics. Higher frequency requires different alpha class
(microstructure data, news/events, options flow) or different cost regime (paid maker
rebates), neither feasible currently.

Files (kept for reproducibility): `ml/research/alpha_intraday_*.py` (4 scripts)

## 2026-05-09 architectural audit (definitively negative)

Goal: find any architectural addition that improves v7's deployable Sharpe.
15 variants tested, all fail discipline gates with full-ensemble validation.

### Audit table

| # | Architecture | ΔSh | Verdict |
|---|---|---|---|
| 1 | Intraday Ridge head (signal-level blend) | ≈0 | FAIL — perturbs strong rankings in 11-name universe |
| 2 | Intraday portfolio-level blend | regime-dep | FAIL — H1-trained ΔSh negative on H2 |
| 3 | Rolling-24mo static blend with v7 | +0.17 IS | FAIL — below +0.20 gate, fails OOS |
| 4 | Rolling regime-conditional (5 indicators × hard-switch) | varies | FAIL — overfits 2024Q3 single event |
| 5 | pred_disp_TAB risk-sizing overlay | +0.05 | FAIL Sharpe; max-DD −33% (DD-only useful) |
| 6 | Regime-conditional MoE retraining (3 bins) | **−2.27** | FAIL hard — data fragmentation kills LGBM |
| 7 | disp_22d as model feature (Path A) | −0.13 | FAIL — drift in absolute scale |
| 8 | disp_22d_pctile as feature (Path A') | −0.32 | FAIL — even percentile-stationary doesn't help |
| 9 | xs_rank_replace A features | −0.61 | FAIL — strips info raw scales carry |
| 10 | xs_rank_add A features | +0.05 | FAIL Sharpe; max-DD −36% (DD-only useful) |
| 11 | ts_pct_add (vol features only) | −0.35 | FAIL |
| 12 | xs+ts combo | −0.52 | FAIL — most max-DD reduction (−65%) at biggest Sh cost |
| 13 | B_xs_add (PEAD ranks) | −0.41 | FAIL |
| 14 | F_xs_add (sector ranks) | −0.85 | FAIL |
| 15 | **ABF_xs_add** (1-horizon × 3-seed test) | **+0.49** | **borderline 4/5 — flipped on validation** |
| 15v | **ABF_xs_add** (3-horizon × 5-seed VALIDATION) | **−0.72** | **FAIL 0/5** |
| (★) | gap-skip overlay (skip first cycle after 30d+ gap) | +0.43 | borderline 3/5 — fails G2 (CI), G4 (event-dep) |

### The validation lesson (critical)

ABF_xs_add looked like the audit's strongest result at +0.49 ΔSh / 4/5 gates
on a 3-seed × 1-horizon ensemble. **Validation at full v7 ensemble (5 seeds
× 3 horizons = 15 models) flipped the sign to −0.72.** The +0.49 was
seed/horizon-mix noise from undersized ensembles.

**Discipline going forward**: never trust architectural claims from
sub-production ensemble configs. The v7 production spec (5 seeds × 3
horizons) exists for exactly this reason — smaller ensembles produce
spurious +0.4-0.5 Sh "improvements" that don't survive scaling.

### Cross-cutting structural findings

1. **Drawdown smoothing is buyable; Sharpe lift is not.** Multiple variants
   (5, 10, 12) reduce max-DD by 30-65% with neutral-to-negative Sharpe. The
   tradeoff is roughly 1:1 — adding 30% DD reduction costs ~0.4 Sharpe.
   Only the gate's filter-and-skip mechanism gets DD reduction at zero cost.

2. **Regime info belongs at gate layer, not model layer.** The dispersion
   gate (60th-pctile of 252d) contributes +2.19 Sh to v7 — the dominant
   alpha lever. Encoding the same info as model features (disp_22d as feat,
   disp_pctile, xs_rank, ts_pct) all hurt or wash. Discrete cycle decision
   ≠ continuous prediction input.

3. **Tree-greedy splits don't extract regime info beyond what
   per-name vol features already encode.** LGBM uses xs-rank features at
   ~40% of model importance but the splits don't translate to OOS Sharpe.
   sym_id + raw A_vol_* implicitly does name-conditional normalization
   already.

4. **Regime-stationary feature ≠ regime-stationary label.** Percentile
   transformations stabilize feature distribution but the conditional
   label distribution still varies across regimes; LGBM averages across
   them and gets noise.

5. **Features that bias predictions conservative help drawdowns and
   hurt bull years.** Per-year ΔSh shows the same pattern across tests:
   wins in 2021 (drawdown), loses in 2023/2025/2026 (strong years).
   No indicator distinguishes "good time to be conservative" from "bad
   time to be conservative" reliably.

### Production decision

**v7 alone is the production strategy on Tier A+B.** Spec unchanged from
2026-05-08:
- LGBM ensemble (15 models: 3 horizons × 5 seeds)
- 18 raw features (10 A + 4 B + 3 F + sym_id)
- K=4 long / K=4 short with hysteresis M=1
- Dispersion gate ≥60-pctile of trailing 252d
- 0.8 bps/side taker fee
- Daily rebalance after US close

**Optional drawdown overlay (operational decision):**
- pred_disp_kill gate: max-DD −33% at +0.05 Sh (within noise)
- xs_rank_add features: max-DD −36% at +0.05 Sh (within noise)
- Either deployable IF downstream cares about drawdown smoothing
  (capital allocation rules, leverage caps) — neither adds Sharpe

### Files (kept for reproducibility)

- `ml/research/alpha_v9_xyz_ridge_blend.py` — Test 1
- `ml/research/alpha_v9_xyz_portfolio_blend.py` — Test 2
- `ml/research/alpha_v9_xyz_rolling.py` — Test 3
- `ml/research/alpha_v9_xyz_regime_blend.py` — Test 4
- `ml/research/alpha_v9_xyz_risk_sizing.py` — Test 5
- `ml/research/alpha_v9_xyz_regime_moe.py` — Test 6
- `ml/research/alpha_v9_xyz_disp_feature.py` — Test 7
- `ml/research/alpha_v9_xyz_disp_pctile_feature.py` — Test 8
- `ml/research/alpha_v9_xyz_feature_norm.py` — Tests 9-12
- `ml/research/alpha_v9_xyz_BF_norm.py` — Tests 13-15
- `ml/research/alpha_v9_xyz_ABF_validate.py` — Test 15v (validation)
- `ml/research/alpha_v9_xyz_gap_skip.py` — Gap-skip overlay
- Cached: `data/ml/cache/v9_rolling_24mo_preds.parquet`, `data/ml/cache/v7_regime.parquet`

### Forward-look — what's left for v7

Free-data signal-level optimization is closed. Remaining productive paths:

1. **Wire cron + start gathering forward observations** (operational, blocking).
   Real forward data resolves the +2.2-2.6 long-run vs +5-7 recent-regime gap.
2. **Gap-skip overlay deployment** — only candidate to reach borderline status.
   3/5 gates pass; ΔSh +0.43 (event-dependent on 2021-22). Could deploy at small
   live weight as a probe, but not as production. Honest path: wait for forward
   data to confirm the 2021-22 mechanism replicates.
3. **Different inputs**: paid orthogonal data (L2 microstructure, options flow,
   alternative data). $50-1000/mo + integration weeks. Uncertain payoff.
4. **Different problem framing**: weekly horizon, pair trading, sector rotation.
   Qualitative pivot. Weeks of work.
5. **Different universe**: Russell mid-caps, intl equities — requires xyz to
   list those names (it doesn't currently for most candidates).

## Crypto v6_clean — closed

Last live cycle: 2026-05-03. State archived to `live/state/closed_session_20260508/`.
Three closed cycles in May 2026: net -12.6, -94.5, -139.0 bps. Real positions cleared
on Hyperliquid manually. No cron, no background processes.

To re-deploy: restore from `live/state/crontab.backup.20260503.txt`.

## v6_clean PM gate addendum (2026-05-08, post-closure)

After session closure, a **new turnover-reduction lever validated**:
**PM_M2_b1** (pred-momentum entry gate). It filters NEW entries that
weren't in top-K at the previous cycle, treating one-cycle blip predictions
as noise. Mechanistically distinct from retention/hysteresis (which held
*stale* alpha) — this filters *fresh noise* at entry. K is variable
downward when persistence rejects entries.

### Validated numbers (paired multi-OOS, 10 folds, 1800 cycles)

| variant | Sharpe | net bps/cyc | paired Δsh vs baseline | folds + |
|---|---|---|---|---|
| baseline | +0.33 | +0.39 | — | — |
| conv_p30 (current production) | +1.16 | +1.25 | +1.88 | — |
| **PM_M2_b1** | **+1.96** | +2.33 | +1.98 | 9/10 |
| **conv+PM stacked** | **+2.75** | **+3.01** | **+2.55** | — |

- Bootstrap CI on Δnet for conv+PM: **[+0.37, +4.95]** — first variant to clear lower bound > 0
- Compositionality: 93% additive (gates capture mostly orthogonal signals)
- Hard-split frozen test PM_M2_b1: ΔSharpe **+2.01** (4/5 folds positive) — **survives** structural test that conv_gate did not

### Mechanism: implicit regime detection

In sample-period analysis (1369 active cycles), short-heavy bias correlated
with steep down regimes (S+3+ bucket: 49 cycles, basket avg −29 bps,
strategy net **+24 bps** Sharpe **+7.27**). The gate is doing
persistence-as-conviction: when many same-direction names persist in
top/bot-K, the model has high directional confidence, and that direction
tends to continue. The +2.75 Sharpe includes both pure spread alpha AND
this directional regime alpha; constant-per-name (1/7 fixed) weighting
produces non-market-neutral leg gross when K asymmetry is high.

### Caveats before deploying

1. **Not market-neutral when K_L ≠ K_S.** With current per-name=1/7
   weighting, leg gross varies with K_actual. A constant-gross variant
   (always invest full $10k/leg) would restore strict neutrality but
   probably reduces Sharpe to ~+2.0-2.4 by removing the regime-tilt alpha.
   Decision is policy, not math.
2. **CI lower bound on Δnet is just barely > 0** (+0.37 bps). Strong
   point estimate (+2.62) but not 5σ.
3. **conv+PM hasn't been hard-split-tested directly.** PM alone passed.
4. **Tail behavior unknown.** S+3+ contributed ~33% of total bps from
   3.6% of cycles. No 2008-magnitude tails in training data; flash
   recoveries during S+3+ regime would whipsaw a concentrated short book.
5. **Concentration when K is small.** K_min=1 occurred in 2.6% of active
   cycles. Add capital sizing (50-60% of theoretical max notional).

### Files

- Gate impl: `ml/research/alpha_v9_pred_momentum.py` (function `portfolio_pnl_pred_momentum_bn`)
- Multi-OOS paired: `ml/research/alpha_v9_pred_momentum_multioos.py`
- Hard-split frozen: `ml/research/alpha_v9_pred_momentum_hardsplit.py`
- Stacking with conv_gate (LGBM-only): `ml/research/alpha_v9_pred_momentum_stack.py`
- **Stack on production hybrid (LGBM+Ridge)**: `ml/research/alpha_v9_pm_hybrid_stack.py`
- **Ridge weight sweep with PM active**: `ml/research/alpha_v9_ridge_weight_with_pm.py` (drop Ridge)
- **Hard-split conv+PM frozen**: `ml/research/alpha_v9_pm_stack_hardsplit.py` (SURVIVES Δsh +2.64)
- **Weighting policy test**: `ml/research/alpha_v9_const_gross_pm.py` (keep per-name 1/7)
- Cycle-level CSVs: `outputs/pred_momentum_stack/`, `outputs/pm_hybrid_stack/`, `outputs/ridge_weight_with_pm/`, `outputs/pm_stack_hardsplit/`, `outputs/const_gross_pm/`

### Deploy-blockers RESOLVED 2026-05-08

All four research deploy-blockers passed on this session:

1. **[done]** Tier 1A — conv+PM on production hybrid: Sharpe +2.35 (CI > 0). But LGBM-only +2.75. Ridge head and PM gate partially conflict.
2. **[done]** Ridge weight sweep with PM active: w=0 optimal, w=0.025 marginal +0.11 Sh n.s. **→ drop Ridge for deployment.**
3. **[done]** Hard-split conv+PM frozen: ΔSharpe **+2.64**, Δnet +2.83 bps, CI **[+0.15, +5.69]** (statistically significant). 4/5 test folds positive. Stronger structural pass than the multi-OOS result.
4. **[done]** Weighting policy test: constant per-name (current) wins by Δsh **+1.57** over constant-gross. const_gross wins only 1/10 folds. **→ keep per-name = 1/7 weighting.**

### Validated deployment config

```
LGBM v6_clean ensemble (5 seeds, no Ridge head)
+ conv_p30 (skip cycle if dispersion < trailing 30th-pctile)
+ PM_M2_b1 (filter new entries needing 2-cycle persistence)
+ per-name weighting at 1/7 (variable leg gross)
+ β-neutral execution clip [0.5, 1.5]
+ TAKER fills only

CORRECTED 2026-05-09 — original research evaluator reset positions on
conv-skip and PM-empty cycles, while paper_bot.py holds prior positions.
evaluate_stacked() now supports execution_model="live" (default) which
matches deployment behavior. Re-validated numbers below.

Multi-OOS Sharpe:    +2.47  (was +2.75 research-model)  CI [+0.56, +4.44]
Hard-split frozen:   Δsh +2.92  (was +2.64) — stronger; CI [+0.28, +6.20]
Net per cycle:       +2.94 bps (vs baseline +0.39)
Cost per cycle:      ~3.0 bps  (vs baseline 7.13)
K_avg:               ~3.4 per leg (vs baseline 7.00)
Skip rate:           ~24% conv + entry rejections within active cycles
Production lift:     +1.1 Sh over current LGBM+Ridge+conv production
```

The −0.28 multi-OOS Sharpe shift is from variance addition (held positions
during conv-skipped cycles add real MtM with no expected alpha). The +0.28
hard-split shift is the same mechanism in a regime where the frozen baseline
is broken — held basket-residual exposure happens to be better than 0.

### Remaining steps

1. **[done 2026-05-08]** Wire conv+PM into `live/paper_bot.py`.
2. **[done 2026-05-09]** Production-readiness audit (10 issues fixed across 3 review passes):
   - Backtest/live divergence (Issues 1+2): live-model evaluator (`execution_model="live"` default in `evaluate_stacked`); re-validated +2.47 multi-OOS / +2.92 hard-split.
   - Atomic state with crash recovery (Issues 3 + 1b): two-phase commit, pending-row dedup, schema-compat with hourly_monitor.
   - Partial-tick discipline (Issues 4 + 4b + 3c): defer mark+funding, gate `hourly_last_tick` AND `hourly_pnl.csv` on `tick_complete`.
   - Skip-cycle persistence (Issues 2b + 3b + 2c): conv-skip + PM-empty paths now save cycle row, carry `equity_usd`, accrue funding for held interval.
   - Schema alignment (Issue 1c): `_append_cycle_row` auto-widens cycles.csv schema for new diagnostic fields without misaligning numeric columns.
3. **[done 2026-05-09]** Pred-disp size-overlay test (port from xyz). DD reduction works on v6 but xyz's "zero Sharpe cost" only partially replicates. **Best zero-cost: overlay 0.50-1.00** (Δsh −0.14, 17% DD reduction, Δnet CI crosses 0). **Best DD reduction: 0.30-0.70** (Δsh −0.13, 33% DD reduction = xyz target, Δnet CI < 0). Strong absorption of losing months (Dec 2025 −1.43→−0.51, Apr 2026 −2.30→−0.96). Decision pending on whether to deploy with overlay; baseline still available.
4. Forward-validate live N=15 → N=30 (5 days runtime).

### Ridge/PM conflict (2026-05-08)

Tested in `ml/research/alpha_v9_pm_hybrid_stack.py`. Ridge head's contribution
to Sharpe by gate level (10-fold multi-OOS):

| Gate level | LGBM-only Sh | Hybrid Sh | Ridge Δ |
|---|---|---|---|
| baseline | +0.33 | +0.61 | **+0.28** (Ridge helps) |
| conv_p30 | +1.16 | +1.33 | **+0.17** (Ridge helps) |
| PM_M2_b1 | +1.96 | +1.89 | −0.08 |
| conv+PM | **+2.75** | +2.35 | **−0.40** (Ridge HURTS) |

Mechanism: Ridge head tilts predictions cycle-to-cycle using positioning
features. PM gate filters based on rank persistence. Ridge's small per-cycle
perturbations push names in/out of the top-K threshold → PM gate sees
"unstable rankings" and rejects what would otherwise be persistent LGBM
signals. The two are partially redundant in role (both favor stable,
well-positioned names) but mechanistically antagonistic when stacked.

Composition (hybrid): 83% additive (vs 93% in pure LGBM). Some additivity
preserved but weaker.

**Open production decision**: drop Ridge entirely (deploy LGBM+conv+PM at
+2.75 Sh) vs keep hybrid (deploy hybrid+conv+PM at +2.35 Sh). Quick
follow-up: sweep Ridge blend weight w ∈ {0.0, 0.05, 0.10, 0.15} with PM
gate active to find the new optimum (~10 min test, not yet done).

## Universe expansion exploration (2026-05-09, all paths CLOSED)

Comprehensive 1-day exploration of whether ORIG25 can be expanded.
**Conclusion: ORIG25 stays. Workflow is universe-specific, not generalizable.**

### TL;DR

| Path | Result |
|---|---|
| Bundled additions (FULL39, +DeFi3, +L1_3, +Quality6, +NonMeme10, +MemesOnly4) | All hurt by 2.0-3.3 Sh |
| Per-symbol leave-one-in (14 candidates) | Only LDO (+0.12) and 1000SHIB (+0.35) individually compatible |
| Joint LDO+1000SHIB (27 syms) | **ANTAGONISTIC**: Δsh −2.03, Sharpe drops to +0.72 |
| Forced cluster-bucketed K_c=1 | **Failed**: Sharpe −2.12 (forces weak-cluster picks) |
| Capped per-cluster cap=2/3/4 | **Failed**: Sharpe +0.21 (worse than uncapped FULL +0.88) |
| **Workflow generalization (5 alt 25-name baskets)** | **Mean Sharpe −0.38**, ORIG25 +2.75 is rank 1/6 by 1.96 margin |

### Root cause: rank-competition + workflow overfit

Two compounding mechanisms:

1. **Rank-competition destabilizes PM gate** (per-symbol diagnostic): adding names to universe → more rank churn cycle-to-cycle → PM rejects more entries → K_avg drops 20% → effective deployment shrinks. Per-symbol IC and selection are unchanged; only K_actual collapses.

2. **Workflow is overfit to ORIG25** (workflow generalization test, 5 alternative 25-name baskets): ORIG25 +2.75 is rank 1 of 6 by 1.96 Sh margin. Other 25-name baskets average **−0.38 Sh**. The +2.75 is a property of ORIG25 specifically through the workflow, NOT a property of the workflow itself. The 30+ test audit selected feature set, hyperparameters, K, and gate parameters all on ORIG25 → universe-specific configuration.

### Implication: "v6_clean" isn't a workflow, it's a configuration

The audit produced a configuration that works on ORIG25. Other universes need their own audit (~3-4 weeks per universe). Adding/removing names from ORIG25 breaks the configuration.

### What this means for future planning

- **Path 1 (deploy ORIG25 + 1000SHIB)**: viable single-name expansion, +0.35 Sh expected. ~1 day to deploy.
- **Path 2 (architecture change)**: failed — selection-layer changes can't fix universe expansion under current model.
- **Path 3 (full retrain on FULL39)**: was estimated at +2.0-3.0 Sh. Workflow generalization test downgrades this expectation substantially. Random 25-name baskets retrained with same workflow average −0.38 Sh.
- **Default**: stay at ORIG25. Forward live N=30 to confirm validated +2.75 transfers.

### Files

- `ml/research/alpha_v9_universe_expand.py` — FULL39 + curated subsets (all bundled additions hurt)
- `ml/research/alpha_v9_universe_curated.py` — 6 quality-filtered subsets (all hurt)
- `ml/research/alpha_v9_universe_diag.py` — root-cause diagnostic (rank-competition mechanism)
- `ml/research/alpha_v9_universe_leave_one_in.py` — per-symbol compatibility (only LDO + 1000SHIB pass)
- `ml/research/alpha_v9_universe_27sym.py` — joint LDO+1000SHIB (antagonistic)
- `ml/research/alpha_v9_clustered_backtest.py` — forced cluster-bucketed K_c=1 (failed)
- `ml/research/alpha_v9_capped_backtest.py` — capped cap=2-4 (failed)
- `ml/research/alpha_v9_workflow_generalization.py` — alternative 25-name baskets (workflow doesn't generalize)
- `ml/research/portfolio_clustered.py` — cluster-bucketed and cluster-capped selection backends
- `config/clusters_v1.json` — 6-cluster sector definitions (only 3 are correlation-cohesive)
- Outputs: `outputs/universe_expand/`, `outputs/universe_curated/`, `outputs/universe_diag/`, `outputs/universe_leave_one_in/`, `outputs/universe_27sym/`, `outputs/clustered_backtest/`, `outputs/capped_backtest/`, `outputs/workflow_generalization/`

---

# Status — 2026-05-06 (previous, crypto-focused — kept for history)

## Program

P-2026-001: ML CTA engine for crypto perpetuals. Goal was a deployable signal
layer extracting alpha from kline + aggTrade data on Binance USDM perps.

## ⚠️ Cost-formula correction (2026-05-06)

**Previously published Sharpe values were ~2× too high** because the backtest
cost formula in `ml/research/alpha_v4_xs.py::portfolio_pnl_turnover_aware`
applied `cost_bps_per_leg × 0.5 × Σ|Δw|` instead of `× Σ|Δw|`. The 0.5×
factor was inherited from a "round-trip per leg" cost model; when callers
later substituted the HL VIP-0 one-way taker fee (4.5 bps) directly, fees
were under-counted by exactly 2×. **Fixed in commit on 2026-05-06.**

The corrected economics below are now consistent with `live/paper_bot.py`'s
`HL_TAKER_FEE_BPS × notional_traded / equity` accounting (verified
deterministically against live cycle data).

## Current state (2026-05-06)

**Deployment running but edge is thinner than originally claimed. h=48 K=7
ORIG25 is still the recommended config; needs HL fee discounts (HYPE
staking) to clear cost line comfortably.**

| Config | Sharpe (multi-OOS, **corrected**) | Cost | Status |
|---|---|---|---|
| h=288 K=5 (legacy) | **~+0.5** (was reported +3.30) | 4.5 bps/leg taker, one-way | not running |
| **h=48 K=7 ORIG25** | **~+0.6** (was reported +3.63) | 4.5 bps/leg taker, one-way | running, N=15 forward Sharpe +2.6 |

Forward-test (N=15 cycles, 2.5 days): mean net +2.7 bps/cycle, Sharpe +2.6.
This is within sample noise of the corrected backtest base case but well
below the previously-claimed +3.63. See "Fee sensitivity" below for how
edge scales with HL discount tiers.

## Fee sensitivity (corrected, 2026-05-06)

Per-cycle net at h=48 K=7 ORIG25, both live (N=15 forward, gross+12.56,
turnover 1.64) and backtest (corrected, gross +7.90, turnover 1.72):

| Tier | fee/leg | Live net/cyc | Live Sharpe | BT net/cyc | BT Sharpe |
|---|---:|---:|---:|---:|---:|
| HL VIP-0 taker (current) | 4.50 | +2.78 | +2.69 | +0.16 | +0.13 |
| + Referral (-4%) | 4.32 | +3.08 | +2.98 | +0.47 | +0.39 |
| HYPE Bronze (-5%) | 4.28 | +3.15 | +3.05 | +0.55 | +0.46 |
| Bronze + Referral | 4.10 | +3.44 | +3.33 | +0.85 | +0.71 |
| HYPE Silver (-10%) | 4.05 | +3.52 | +3.41 | +0.93 | +0.78 |
| Silver + Referral | 3.89 | +3.78 | +3.66 | +1.21 | +1.01 |
| HYPE Gold (-15%) | 3.83 | +3.89 | +3.77 | +1.32 | +1.10 |
| Gold + Referral | 3.67 | +4.14 | +4.01 | +1.59 | +1.33 |
| HYPE Platinum (-25%) | 3.38 | +4.63 | +4.48 | +2.10 | +1.75 |
| Platinum + Referral | 3.24 | +4.85 | +4.70 | +2.33 | +1.94 |
| fee = 3.0 bps | 3.00 | +5.24 | +5.08 | +2.74 | +2.29 |
| fee = 2.0 bps | 2.00 | +6.88 | +6.67 | +4.46 | +3.73 |
| HL VIP-0 maker (1.5) | 1.50 | +7.70 | +7.46 | +5.32 | +4.45 |
| Deep-tier maker (~0.75) | 0.75 | +8.95 | +8.66 | +6.61 | +5.53 |

NOTES:
- "Live-data view" extrapolates from observed N=15 gross+turnover, with
  measured slippage 2.52 bps/cycle and funding -0.12 bps/cycle held constant.
- "Backtest view" uses docs' gross +7.90 and corrected turnover 1.72; assumes
  slip = funding = 0 (backtest doesn't model them separately).
- Std assumptions: live std 48.3 bps/cycle (observed), backtest std 56 bps
  (implied from old docs). Std treats fee changes as ~constant per cycle, so
  is approximately tier-invariant.
- N=15 is too small to distinguish live's gross-favorable regime from a
  durable signal advantage. By N=30 the gap to backtest's +7.90 will be
  clearer.

Each 1 bps reduction in per-side fee → ~+1.6 bps/cycle improvement in net
(= turnover × Δfee), which is ~+1.55 annualized Sharpe at h=48 cadence.

## Recommendation

- **Stake 100+ HYPE for Bronze and add a referral link** → gets fees to
  ~4.10 bps. Boosts live-view Sharpe from ~+2.7 to ~+3.3, and pulls
  backtest base case from ~+0.13 (essentially zero) to ~+0.71 (mildly
  positive). Cheapest immediate Sharpe lift.
- **Sustained edge case**: if forward-test still positive at N=30, stake
  to Silver/Gold tier and target backtest Sharpe ~+1.0–+1.5.
- **Maker fills (1.5 bps tier)** would be transformative (Sharpe +4.5),
  but require post-only execution + L2 fill simulation that doesn't exist
  in this codebase yet.

Both at 25-symbol original universe (ORIG25 — original 25 BNF perps,
excludes 14 newer entrants), `REGIME_CUTOFF = 0.50` (label-quality gate),
v6_clean feature set (28 features), 5-seed LGBM ensemble. The h=48 win
comes from reduced volatility (smoother PnL via 6× more rebalances), not
higher absolute return.

### What's in place for h=48 deployment

- `models/v6_clean_h48_ensemble.pkl` + `models/v6_clean_h48_meta.json` —
  artifact trained on h=48 labels, ORIG25 universe, regenerated via
  `HORIZON_BARS=48 UNIVERSE=ORIG25 python -m live.train_v6_clean_artifact`.
  End-to-end pipeline verified: per-bar XS IC = +0.0645 on cal window
  matches multi-OOS expectation +0.0627.
- `live/paper_bot.py`: `HORIZON_BARS` and `TOP_K` env-overridable
  (defaults 288/5 preserve current production). Artifact loader picks
  horizon-suffixed file when env var is set, falls back to legacy.
- `live/train_v6_clean_artifact.py`: `HORIZON_BARS` + `UNIVERSE`
  env-configurable; writes horizon-suffixed `v6_clean_h{N}_ensemble.pkl`.
- `live/cycle_summary.py`: detects horizon (env > legacy meta > suffixed
  meta > 288); annualization tracks chosen horizon.

### To deploy h=48 on a FAPI-accessible server

```bash
# On the new server, after copying repo + state + models/:
HORIZON_BARS=48 TOP_K=7 BINANCE_FAPI_URL=https://fapi.binance.com \
  python -m live.paper_bot --source binance
# Cron: 1 */4 * * *  (every 4h at minute :01)
# Weekly retrain: HORIZON_BARS=48 UNIVERSE=ORIG25 in env for the
# train_v6_clean_artifact cron line.
```

### Research arc 2026-05-04 (aggTrade microstructure path tested)

| Test | Result | Verdict |
|---|---|---|
| 4h-aggregated trade-flow features (signed_volume_4h, tfi_4h, aggr_ratio_4h, buy_count_4h, avg_trade_size_4h) | Univariate IC 0.018-0.026 (6 of 29 passed gates) | Encouraging at feature level |
| Portfolio additive (v6_clean + 5 aggTrade) | +0.04 Sharpe | Capacity dilution |
| **Unified paired test (same panel, same cycles, just feature swap)** | **-0.54 bps/cycle, t=-0.54 (p=0.29)** | **Tested negative — no deployable edge** |

5th independent confirmation of saturation hypothesis. aggTrade features
are information-equivalent to v6_clean's kline-flow proxies for h=48
prediction. See `docs/AGGTRADE_NEGATIVE_RESULT.md`.

### Research arc 2026-05-03 (all 4 paths tested for h=48 lift)

| Test | Result | Verdict |
|---|---|---|
| rc-sweep at h=48 (rc=0.33-1.00) | rc=0.50 still optimal (plateau 0.50-0.70) | **No change needed** |
| Horizon-matched features (replace 24h windows w/ 4h) | Sharpe **-3.67** (catastrophic) | Negative — long-window features ARE the alpha |
| Lean trim (drop 9 perm_drop≈0 features) | Sharpe -1.63 | Negative — interactions matter |
| Stage 2 additive (+dom_z_1d, +realized_vol_4h, +idio_vol_4h) | Sharpe -1.51 to -2.05 | Negative — multi-feature redundancy |

**Mechanism for all 4 negatives**: v6_clean is a tight local optimum at
both h=288 and h=48. The strategy is the same multi-day cross-sectional
reversion alpha sampled at different cadences. Features must span
multiple timescales (1h, 4h, 8h, 24h, 3d, 7d) to identify dislocation;
adding redundant short-window features causes scale collapse, removing
"expendable" features breaks interactions. See
`outputs/h48_features/feature_attribution.json` for per-feature attribution.

## Historical state (2026-04-30 baseline, preserved for context)

**Phase: research complete, deployment plausible at h=288 with VIP-3 + maker
execution. Earlier "blocked on cost" conclusion was inflated by a per-bar
cost-accounting bug.**

The strategy class (LGBM regression on alpha-residual targets, with
cross-sectional ranking across 25 symbols) is fully characterized:

- **Signal exists**: rank IC consistently +0.035 across folds and OOS at h=48,
  +0.038 at h=288.
- **Signal is real**: alpha capture of +2.5 bps (h=48) to +7.4 bps (h=288)
  per rebalance verified in β-neutral execution (which strips market noise).
- **Honest cost picture**: with turnover-aware non-overlapping label
  evaluation, OOS net per cycle is **-7.5 bps** (h=48) / **-8.7 bps**
  (h=288) at retail VIP-0. Was reported as -21 under naive per-bar 24-bps
  accounting (over-charged ~2.5×).
- **Signal-quality plan (Apr 30) lifted Sharpe ~3× over the corrected v4
  baseline** at the deployment-relevant tier. Best config under multi-OOS
  validation: **v6 + K=5 + β-neutral**, Sharpe +1.20 at VIP-3+maker, 95% CI
  [-0.78, +3.30] over 270 OOS cycles (9 expanding-WF folds).
- The single-OOS Sharpe of +3.94 (90 cycles, 2026-01-28 to 2026-04-28) was
  partly regime luck — multi-OOS across 9 windows is the more reliable estimate.
- **v6_clean (Apr 30 PM): permutation-importance audit identified 4 features
  with negative or zero OOS contribution (`beta_short_vs_bk`, `idio_vol_1d_vs_bk`,
  `bars_since_high`, `volume_ma_50_xs_rank`). Dropping them (28 features) lifted
  multi-OOS Sharpe to +2.95 [+0.85, +4.54] at K=5+VIP-3+maker — CI no longer
  crosses zero. Mean rank IC +0.0606 (vs +0.045 v6). All 9 folds positive IC.
  Even at VIP-0 retail: Sharpe +1.62, net +14.7 bps/cycle.**
- **Phase 0 paper-trade prep (May 1):**
  - 0a: All 25 v6_clean symbols available on Hyperliquid via info.meta() ✓
  - 0b: Binance↔HL basis at 1h resolution, full 90-cycle holdout — gross
    Sharpe drop −0.13 (Binance +5.08 → HL +4.95, CI overlap is total).
    Per-symbol return correlations all ≥0.98. Basis is statistically zero
    at portfolio level. ✓
  - Decision: Binance-trained predictions transport to HL execution.
    Proceed to Phase 1 (multi-symbol paper-trade orchestrator).
- **Phase 1 paper-trade orchestrator (May 1):**
  - `live/train_v6_clean_artifact.py` → trains v6_clean ensemble on full
    history, saves to `models/` (regen weekly).
  - `live/paper_bot.py` → daily-cron rebalance: refreshes klines (fapi
    or Binance Vision fallback), builds inference panel, predicts, ranks,
    selects top-5 long / bot-5 short β-neutral, fetches HL mids for
    fill simulation, persists positions + cycle log to `live/state/`.
  - `live/replay_paper_bot.py` → validates live code path against the
    canonical backtest. PASS on holdout fold (Δ spread 0.07 bps, Δ IC
    0.0026, both within tolerance). Best_iter sequence matches audit
    exactly (19, 8, 15, 5, 17).
  - First end-to-end run: long [LTC, LINK, NEAR, DOT, ARB], short [SEI,
    TIA, OP, FIL, ADA], gross 2.0, β-scales [1.11, 0.89], cost 8 bps.
  - Note: Binance fapi REST is geo-blocked (HTTP 451) from this dev
    server; bot auto-falls back to Binance Vision daily archive
    (`--source vision`, 1-day lag). On a non-blocked VPS, `--source
    fapi` gives real-time data.
- **Phase 1.5 (May 1): HL data feed for forward test.**
  - Added `--source hl` to paper_bot.py — pulls 5min klines from
    Hyperliquid info API (15-day max retention, real-time, no
    geo-block).
  - Validated against Binance Vision side-by-side at aligned
    target_time: Spearman rank correlation +0.95, long top-5 overlap
    4/5, short bot-5 overlap 4/5. HL and Binance feeds produce
    near-identical portfolio choices despite HL volume being in coin
    units (much smaller than Binance quote-volume).
  - **Recommended for forward test: `python -m live.paper_bot --source hl`**
    on a 5min cadence cron.
- **Phase 2 (May 1): L2 orderbook + realistic taker fill simulation.**
  - paper_bot now fetches HL `info l2Book` snapshots for each leg at
    entry and exit, walks the book to compute volume-weighted average
    fill price for the target notional, and records per-leg slippage
    in bps (signed: positive = adverse).
  - HL VIP-0 taker fee 4.5 bps per side embedded in cost stack.
  - Cycle log records BOTH cost models:
    - `net_bps`: close-all + reopen-all (conservative, over-charges
       names that carry over between cycles).
    - `tt_net_bps`: turnover-aware (matches the canonical backtest;
       only charges the delta between prev and new portfolios).
  - First-cycle smoke test (10 legs, $10K equity, $1K-2K per name):
    mean entry slippage **+2.1 bps** per leg (~1 bps half-spread + ~1 bps
    depth impact on liquid HL books). Exit slippage similar.
  - L2 maker fill simulation (queue-position tracking) is NOT yet
    implemented — paper trades all execute at taker. For realistic
    maker P&L, place actual passive limit orders on a small live HL
    account via executeEngine HL branch.
- **Phase 2.1 (May 1): turnover-aware execution.** Replaced close-all +
  reopen-all with per-symbol delta trading. When target == prev (no
  rebalance needed), 0 trades, 0 fees, 0 slippage — PnL is pure mid-to-mid
  MtM. Also fixed a 2× fee bug in the prior accounting.
- **Phase 2.2 (May 1): hourly funding accrual.** Each cycle fetches
  HL `info.fundingHistory` for held symbols over the prev → now window,
  accrues per-position payments (long pays positive rate, short receives),
  subtracts from net PnL. Standalone test: $0.94 funding cost over 24h on
  $10K equity ≈ 0.94 bps/day cost in current market state.
- **Phase 2.3 (May 1): hourly monitor + Telegram.** New
  `live/hourly_monitor.py` script: marks open positions to current HL mids,
  fetches funding since last tick, appends to `live/state/hourly_pnl.csv`,
  sends Telegram with per-leg breakdown. `paper_bot` also sends a daily
  decision summary at end of cycle. Telegram opt-in via env vars
  (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`); silent fallback otherwise.
- **To deploy: see `docs/PAPER_TRADE_RUNBOOK.md`** — daily cron at
  00:01 UTC + hourly monitor at :05, monitor via `python -m live.cycle_summary`.
- Phase 3 (aggTrades microstructure features): pulled 10 symbols × 402 days,
  audited 19 features. Only `avg_trade_size` passed gates (OOS |IC| 0.035,
  weak). True microstructure (TFI/VPIN/Kyle's λ) doesn't carry signal at
  h=288 — it's a shorter-horizon phenomenon. v8 not pursued.
- v6 = 32 features: v4 base + cross-asset + 7 kline-flow (obv_z_1d, vwap_*, mfi)
  + 8 cross-sectional pctile-rank features.
- v6_clean = v6 minus 4 confirmed-harmful features (28 features).
- Funding-rate features (Phase 4.1) had strongest single-feature OOS IC (up
  to 0.084) but did NOT improve portfolio Sharpe — already captured implicitly
  by v6's basket-relative features.
- **Leakage audit on both v6 and v6_clean passed all 4 tests** (forward-peek
  shift, sanity control, xs_rank PIT, embargo).
- See `docs/METHODOLOGY_REVIEW.md` "Apr 30 — Signal-quality plan execution"
  for the full progression and per-phase details.

Caveats:
- Non-overlapping label evaluation, not a full equity-curve backtest.
- Single 90-day OOS window — CIs remain wide despite point-estimate gain.
- Deployment-grade still needs funding accrual, real maker fill modelling,
  drawdown limits, queue-position economics, more OOS data.

See `docs/METHODOLOGY_REVIEW.md` "Apr 30 follow-up" section for full tables.

## What works

| Component | Status |
|---|---|
| Binance Vision data loader (klines + aggTrades) | ✅ |
| Feature pipeline (160+ kline + 22 alpha-tailored + cross-asset + cross-sectional) | ✅ |
| Walk-forward CV with embargo + label purging | ✅ |
| Pooled multi-symbol training | ✅ |
| Cross-sectional ranking and portfolio P&L | ✅ |
| Cost model (fee + slip + Roll-spread; per-leg hedged) | ✅ |
| Look-ahead bug detection (Sharpe target shift, VPIN bucket) | ✅ |

## Known issues / debts

1. **Trigger-rate calibration breaks under regime shift** — q=0.95 on cal
   doesn't translate to OOS when prediction distributions widen. SOL
   especially: 5% calibrated → 68% OOS trigger rate. Per-symbol thresholds
   help BTC/ETH but not SOL. Workaround: use rank-based selection (top-K
   per bar) instead of magnitude threshold — built into v4.

2. **`sym_id` underused by LGBM** (0.04% importance in v3) despite per-symbol
   IC sign reversals. Suggests trees don't naturally partition on a
   low-cardinality categorical. Workaround tried: per-symbol heads. Did not
   significantly improve at current sample size.

3. **AggTrades are 16 GB for 3 symbols × 400d** — cross-sectional v4 uses
   kline-only features for the 25-symbol universe. Adding aggTrade features
   (TFI, VPIN, Kyle's λ) for all 25 would be ~130 GB; exceeds local disk.

4. **Hyperparameter selection bias** — LGBM params (num_leaves=63,
   min_data_in_leaf=50, lambda_l2=3.0) and trigger config (q=0.95, h=48)
   were chosen by reviewing all walk-forward folds. Some selection bias
   bakes into the WF results.

## Reproducibility

All results in `docs/METHODOLOGY_REVIEW.md` reproducible from this repo:

1. `python3 -m scripts.pull_xs_klines` (~20 min) — kline data for 25 symbols
2. `FEATURE_SET=v6_clean MULTI_OOS=1 python3 -m ml.research.alpha_v4_xs_1d` (~5 min) — current best config
3. `FEATURE_SET=v6 MULTI_OOS=1 python3 -m ml.research.alpha_v4_xs_1d` — prior baseline for comparison
4. `FEATURE_SET=v6 TRIM_UNIVERSE=1 python3 -m ml.research.alpha_v4_xs_1d` — adds IS-trim
5. `FEATURE_SET=v6 python3 -m ml.research.alpha_v4_edge_diagnostic` — diagnostic sections A–G
6. `FEATURE_SET=v6_clean python3 -m ml.research.alpha_v6_leakage_check` (~3 min) — leakage verification
7. `FEATURE_SET=v6 python3 -m ml.research.alpha_v6_edge_review` — feature ceiling: per-feature IC, oracle, redundancy
8. `FEATURE_SET=v6_clean python3 -m ml.research.alpha_v6_permutation_lean` (~10 min) — model-uses-feature audit
9. `python3 -m ml.research.alpha_v4_flow_audit` (~1 min) — flow feature IC audit
10. `python3 -m ml.research.alpha_v7_funding_audit` (~2 min) — funding feature IC audit

Other feature sets via `FEATURE_SET=v4|v5|v5_lean|v6|v6_clean|v7|v7_lean`. Defaults to v4.

Funding-rate data is auto-downloaded by `data_collectors/funding_rate_loader.py`
on first import (caches per-symbol parquet to `data/ml/cache/funding_*.parquet`).

Caches build to `data/ml/cache/` on first run; subsequent runs are fast.

## Compute footprint

- Disk: ~700 MB for 25-symbol klines, ~16 GB if pulling BTC/ETH/SOL aggTrades
- RAM: peak ~8 GB during cross-sectional panel assembly
- CPU: training a 5-seed LGBM ensemble on 700K rows × 17 features takes
  ~2-5 minutes
