# Handoff — 2026-05-09

## Active program: xyz US-equity v7 alpha-residual

**Where it is**: shadow harness fully built and tested end-to-end. Awaiting cron
installation + forward-test cycles. **Crypto v6_clean program was closed in
the morning of 2026-05-08, but a new validated lever (PM_M2_b1 entry gate,
+2.75 Sharpe stacked with conv_gate) emerged in the afternoon** — see
"v6_clean PM gate addendum" before deciding whether to redeploy crypto.

**2026-05-09 update**: 15-variant architectural audit on v7 completed. All
fail discipline gates under full-ensemble validation. v7 alone remains
production. See STATUS.md "2026-05-09 architectural audit" for the full
table; key finding is that **a 3-seed × 1-horizon test inflated one variant
to +0.49 Sharpe lift but full-ensemble validation flipped it to −0.72** —
discipline lesson for future audits. Free-data signal-level optimization
on this universe is now decisively closed.

### Quick-start (next person)

```bash
# Verify everything still loads
python -m live.xyz_paper_bot --check-state    # should print "(no state)" first run

# Manually fire cycle 1 to seed state + send first telegram
python -m live.xyz_paper_bot                  # ~30 sec including yfinance refresh

# Inspect what got logged
python -m live.xyz_paper_bot --check-state    # full state JSON + cycles.csv tail

# Install cron (one-time)
crontab -e
# add the two entries below — daily rebalance + hourly monitor
```

```cron
# v7 xyz daily rebalance (Mon-Fri 21:30 UTC, 30min after US RTH close)
30 21 * * 1-5 /home/yuqing/ctaNew/live/run_xyz.sh -m live.xyz_paper_bot >> /home/yuqing/ctaNew/live/state/xyz/cron.log 2>&1

# v7 xyz hourly portfolio mark-to-market
5 * * * * /home/yuqing/ctaNew/live/run_xyz.sh -m live.xyz_hourly_monitor >> /home/yuqing/ctaNew/live/state/xyz/cron.log 2>&1
```

### Spec at a glance

| | |
|---|---|
| Universe | tier_ab — 11 xyz US-equity perps (AAPL, AMZN, GOOGL, META, MSFT, MU, NFLX, NVDA, ORCL, PLTR, TSLA) |
| Cadence | daily decision after US close, 1d hold, dispersion-gated |
| Position rule | top-K=4 long, K=4 short, hysteresis exit at K+M=5 |
| Gate | trade only if 22d cross-sectional dispersion ≥ 60-pctile of trailing 252d (PIT) |
| Notional | $10k/leg (configurable via `--notional-usd`) |
| Cost assumption | 0.8 bps taker fee + ~3 bps slippage = 3.5 bps/side realistic |
| Backtest active Sharpe @ realistic cost | +3.11 (CI [+1.79, +4.49]) |
| Hard-split frozen 2020-2026 | +1.67 Sh (vs original baseline -0.28) |
| Recent 12mo OOS | +9.4% ann return, but only 31 firing cycles |
| Realistic live haircut (25-30%) | +2.0 to +2.5 Sharpe expected live |

### What's deployable

- ✅ Model artifact frozen at `models/v7_xyz_ensemble.pkl` (15 LGBMs, 18 features, trained on full S&P 100 2013-2026)
- ✅ Daily bot `live/xyz_paper_bot.py` (~930 lines, transactional state, fail-closed semantics)
- ✅ Hourly monitor `live/xyz_hourly_monitor.py` (telegram-ready)
- ✅ Cron wrapper `live/run_xyz.sh` (auto-detects venv)
- ✅ Runbook `docs/xyz_PAPER_TRADE_RUNBOOK.md`
- ✅ Basis-quality probe `ml/research/xyz_data_quality.py` (re-run quarterly)

### What's settled (don't redo)

- **Universe is `tier_ab` (11 names)**. Tier C names (AMD, INTC, COST, LLY) excluded
  per 2026-05-08 basis probe — they hurt backtest at realistic cost. `full15` backtests
  worse. `tier_a` (8 names) gives +2.78 Sh — slightly worse than tier_ab. Mixing presets
  across cycles makes the realized Sharpe series inconsistent; pick one.
- **K=4, M=1 hysteresis** is right for tier_ab. K=2/M=1 trades too few names; K=2/M=2
  too restrictive — both validated worse on this universe.
- **Gate at 60-pctile** is approximately optimal. Lower gates trade more cycles but
  with worse alpha quality (recent regime: gate=0.30 ann ret +2.2% vs gate=0.60 +9.4%).
  Don't lower the gate unless you specifically want utilization at the cost of Sharpe.
- **Annual retrain mandatory.** More frequent (monthly/quarterly) was tested and hurts.
  Less frequent decays. Set calendar reminder for 2027-05-08.
- **Daily cadence is the natural frequency.** Multiple intraday redesign attempts
  (see Status doc) fail at xyz cost economics. Don't re-litigate this — gross intraday
  alpha is structurally too small to clear 5-7 bps round-trip cost.
- **Free-data signal-level optimization is closed (2026-05-09 audit).** 15
  architectural additions tested and FAILED discipline gates under full-ensemble
  validation. Don't re-litigate any of these without paid orthogonal data:
  intraday Ridge head, intraday port-blend, rolling blend, rolling regime-conditional
  (5 indicators), pred_disp risk-sizing, regime-conditional MoE retraining,
  disp_22d as feature (raw or percentile), xs_rank features (replace or add),
  ts_pct add, xs+ts combo, B_xs_add, F_xs_add, ABF_xs_add. The dispersion gate
  (60-pctile of trailing 252d) is the dominant alpha lever (+2.19 Sh) and the
  per-name vol features already encode regime info LGBM can use. See STATUS.md
  "2026-05-09 architectural audit" for full table and mechanism explanations.
- **Gap-skip overlay** (skip first cycle after 30+ day non-trading gap) is the
  single borderline-positive finding from the audit: ΔSh +0.43, passes 3/5
  discipline gates, fails G2 (Δnet CI crosses zero) and G4 (57% of lift comes
  from 2021-22 drawdown). NOT deployed; revisit only after forward data confirms
  the 2021-22 mechanism replicates. Implementation: `ml/research/alpha_v9_xyz_gap_skip.py`.
- **Don't trust sub-production ensemble configs** for architectural claims.
  ABF_xs_add showed +0.49 ΔSh on 3-seed × 1-horizon (3 models per fold) but
  flipped to −0.72 at full v7 ensemble (5 seeds × 3 horizons = 15 models).
  Discipline: any audit must validate at production ensemble before claiming
  a Sharpe lift.

### What's pending (operational)

1. **Install cron** (commands above). Once installed, hourly telegram fires within 1h,
   daily rebalance fires next 21:30 UTC weekday.
2. **Verify HL xyz taker fee schedule.** I'm using 0.8 bps/side per user-provided info.
   Worth confirming via HL docs / account dashboard.
3. **Risk overlay** (post-N=10 forward cycles). Drawdown brake (e.g., halt if cumulative
   net < -X bps over rolling K cycles), kill-switch on data anomaly.
4. **Real executor** (post-N=30 forward cycles). Currently shadow only — `live/xyz_paper_bot.py`
   simulates fills against L2 books but places no orders. When promoted, fork a separate
   `live/xyz_executor.py` (mirror of `live/hl_executor.py` from crypto, swap symbol
   universe + xyz coin form `xyz:NVDA`).

### What might still help (research, not engineering)

The `tier_ab` strategy is at the architectural ceiling for this alpha class on this
universe. Genuine improvement requires different problem framing:

1. **Daily momentum sleeve** (separate strategy) — different alpha class, fires every
   day without dispersion gate. Untested for this universe. Would compose with v7
   residual sleeve (different alpha → diversification benefit).
2. **Pairs / dispersion-event sleeve** — fire when within-pair spread is wide. Sparse
   but additive. Untested.
3. **Larger universe** — Russell mid-caps would have wider dispersion and more residual
   alpha. Requires xyz to list those names (it doesn't currently).
4. **Different signal class entirely** — sentiment/news/options-flow. Requires paid data.

### Defensive engineering already in place (don't undo)

These were added after multiple audit passes — keep them:
- Atomic state save (tmp + `os.replace`)
- Pending-cycle-row in state, dedup-by-decision_ts on flush (exactly-once)
- L2-book pre-check before any state mutation
- `_require_full_fill` raises on insufficient depth
- Same-day re-run guard
- Predictions.csv dedup on same decision_ts
- Funding checkpoint hold on missing mids
- Feature-column-order validation against frozen meta

---

# Handoff — 2026-05-06 (previous, crypto-era — kept for history)

## v6_clean PM gate addendum (2026-05-08, post-closure)

After morning closure of the crypto session, a new turnover-reduction
lever was validated. **Decision needed before re-deploying crypto.**

### One-line summary

Pred-momentum entry gate (`PM_M2_b1`): only enter a name into top-K if
its prediction was *also* in top-K at the previous cycle. Held names
auto-keep on sharp boundary. K is variable downward when persistence
rejects entries. Filters one-cycle "blip" predictions as noise.

### Why this is significant

This is the first turnover-reduction test in the entire 30+ test audit
that **improves Sharpe**. Prior failures (hysteresis, variable K,
magnitude weighting, DD brake, alpha-gap swap, funding-as-gate) all
held *stale* alpha. PM_M2_b1 instead filters *fresh noise* at entry —
mechanistically distinct, empirically validated.

### Validated numbers (multi-OOS paired, 10 folds, 1800 cycles)

| variant | Sharpe | net bps/cyc | Δsh paired vs baseline | folds + |
|---|---|---|---|---|
| baseline (no gate) | +0.33 | +0.39 | — | — |
| conv_p30 (current production) | +1.16 | +1.25 | +1.88 | — |
| **PM_M2_b1 alone** | **+1.96** | +2.33 | +1.98 | 9/10 |
| **conv_p30 + PM_M2_b1 stacked** | **+2.75** | +3.01 | **+2.55** | — |

- Bootstrap CI on Δnet for stacked: **[+0.37, +4.95]** — first variant in audit history to clear lower bound > 0
- 93% additive composition (gates capture orthogonal signals)
- Hard-split frozen test PM alone: ΔSharpe **+2.01** (survives) — stronger structural test pass than conv_gate

### Discipline gates verdict (PM_M2_b1 alone)

| Gate | Pass? |
|---|---|
| 1. ΔSharpe > +0.20 | ✅ +1.98 paired |
| 2. Bootstrap CI on Δnet > 0 (alone) | ⚠️ −0.07 (just below) |
| 2. Bootstrap CI on Δnet > 0 (stacked) | ✅ +0.37 |
| 3. ≥6/10 folds Sharpe-positive | ✅ 9/10 |
| 4. Hard-split frozen survives | ✅ +2.01 |

### Implicit directional bias (feature, not bug — but needs sizing)

When prediction churn is asymmetric, K_long ≠ K_short → leg gross
asymmetric. With per-name = 1/7 fixed (current implementation), this
produces a directional tilt. Empirically:

- S+3+ bucket (49 cycles, K_S − K_L ≥ 3): basket avg **−29 bps**, strategy net **+24 bps**, Sharpe +7.27
- L+1,2 bucket (457 cycles): basket avg +5.7, strategy net +5.2
- L+3+ bucket (39 cycles, the failure mode): basket avg −9, strategy net −0.3

The gate ends up tilting toward the side where predictions are stable,
which historically correlates with the realized market direction (3 of
4 directional buckets net positive). Mechanism: prediction stability
acts as an implicit regime indicator.

A min-K floor (skip cycles where min(K_L, K_S) < 3) was tested
counterfactually: it would **reduce** Sharpe from +3.15 to +2.95 by
removing the high-conviction cycles. **Don't add the floor.**

### Open question: weighting policy

The current implementation uses **per-name = 1/7 fixed**, so leg gross
varies with K_actual. This produces the directional tilt → +2.75 Sharpe.

Alternative: **constant gross = 1.0/leg**, per-name = 1/K_actual. Always
strictly market-neutral but loses the regime-tilt alpha. Estimated
Sharpe ≈ +2.0 to +2.4. **Not yet tested empirically.**

This is a production-policy choice, not a math problem. If running for
institutional capital with a hard market-neutral mandate → constant gross.
If running prop with directional risk budget → keep current.

### Tier 1A — production-hybrid stack test (DONE 2026-05-08)

The validated +2.75 stacked Sharpe was on **pure LGBM** predictions. The
actual production model is the hybrid `0.9 × z(LGBM) + 0.1 × z(Ridge_pos)`.
Re-ran the stack with hybrid predictions in `alpha_v9_pm_hybrid_stack.py`:

| variant | LGBM-only Sh | Hybrid Sh | Ridge Δ |
|---|---|---|---|
| baseline | +0.33 | +0.61 | **+0.28** (helps) |
| conv_p30 | +1.16 | +1.33 | **+0.17** (helps) |
| PM_M2_b1 | +1.96 | +1.89 | −0.08 |
| **conv+PM** | **+2.75** | **+2.35** | **−0.40** (HURTS) |

Hybrid + conv+PM CI on Δnet: **[+0.34, +4.49]** — still passes deployment gate.

**The Ridge head and PM gate partially conflict.** Ridge tilts rankings
each cycle using positioning info; PM filters unstable rankings. Ridge's
perturbations push names in/out of the top-K threshold → PM gate sees
those as "noise blips" and rejects them, even when LGBM signal would have
been persistent. Composition is 83% additive (vs 93% in LGBM-only).

This is **not a fail** — the stacked gate still lifts production by
~+1.0 Sh — but ~0.4 Sh of further lift is recoverable by reducing or
dropping the Ridge head when PM gate is active.

### All deploy-blockers resolved (2026-05-08)

| Step | Result |
|---|---|
| **Ridge weight sweep w/PM** | w=0 optimal. w=0.025 marginal +0.11 Sh n.s. **→ drop Ridge.** |
| **Hard-split conv+PM frozen** | Δsh **+2.64**, Δnet +2.83 bps, CI **[+0.15, +5.69]** (significant). 4/5 folds positive. **SURVIVES.** |
| **Weighting policy** (per-name vs const-gross) | const_gross loses Δsh **−1.57** and adds extreme concentration risk (max single-name 1.39 vs 0.21). 9/10 folds favor per-name. **→ keep per-name = 1/7.** |

### Validated production config

```
LGBM v6_clean ensemble (5 seeds, no Ridge head)
+ conv_p30 (skip cycle if dispersion < trailing 30th-pctile of 252)
+ PM_M2_b1 (filter new entries needing 2-cycle persistence)
+ per-name weighting at 1/7 (variable leg gross)
+ β-neutral execution clip [0.5, 1.5]
+ TAKER fills only

CORRECTED 2026-05-09 — research evaluator originally reset positions on
conv-skip and PM-empty-leg, diverging from paper_bot.py which holds prior
positions. Fixed evaluate_stacked() with execution_model="live" default,
re-validated. Old "research" model still available for backward compat.

Multi-OOS Sharpe:    +2.47  (was +2.75 research-model)  CI [+0.56, +4.44]
Hard-split frozen:   Δsh +2.92  (was +2.64) — stronger; CI [+0.28, +6.20]
Net per cycle:       +2.94 bps  (vs baseline +0.39, vs Ridge+conv production +1.33)
Cost per cycle:      ~3.0 bps   (~57% reduction from baseline 7.13)
K_avg:               ~3.4 per leg (vs baseline 7.00)
Skip rate:           ~24% conv + entry rejections within active cycles
Production lift:     ~+1.1 Sh over current LGBM+Ridge+conv production
```

Why the shift: live model holds positions through conv-skipped cycles. This
adds real MtM (small, ~0 expected) plus variance, which marginally hurts
multi-OOS Sharpe. In hard-split (frozen model regime), the held positions'
basket-residual MtM happens to be slightly positive, making the gate's
structural pass *stronger* under live model (+2.92 vs +2.64).

### Remaining steps to deploy

1. **[done 2026-05-08]** Wire conv+PM into `live/paper_bot.py`. Combined evaluator
   pattern in `alpha_v9_pred_momentum_stack.py::evaluate_stacked`. PM gate
   per-leg persistence-history buffer persists across restarts via
   `live/state/pm_gate_history.json`.
2. **[done 2026-05-09]** Production-readiness audit (10 issues across 3 review passes):
   - Backtest/live divergence (Issues 1+2): live-model evaluator
     (`execution_model="live"` default) — re-validated +2.47 multi-OOS,
     +2.92 hard-split Δsh.
   - Atomic state with crash recovery (Issue 3 + 1b): two-phase commit,
     dedup-by-decision_time_utc, schema-compat between paper_bot and
     hourly_monitor (preserves pending_cycle_row).
   - Partial-tick discipline (Issues 4 + 4b + 3c): defer mark+funding,
     gate BOTH `hourly_last_tick` AND `hourly_pnl.csv` on `tick_complete`.
   - Skip-cycle persistence (Issues 2b + 3b + 2c): conv-skip and PM-empty
     paths save cycle row, carry `equity_usd` from last cycle, accrue
     funding via `accrue_funding_for_cycle`. Skip rows have real
     `funding_bps` and `net_bps = -funding_bps`.
   - Schema alignment (Issue 1c): `_append_cycle_row` auto-widens cycles.csv
     for new diagnostic fields without misaligning numeric columns.
3. **[done 2026-05-09]** Pred-disp size-overlay test (port from xyz validated lever).
   Sweep over 4 (lo, hi) configurations on top of conv+PM. Findings:
   - **DD reduction works** (17-33% across variants), confirming the
     mechanism transfers from xyz to v6.
   - **xyz's "zero Sharpe cost" doesn't fully replicate** — small Sharpe
     trade (Δsh −0.10 to −0.22) appears across all variants. Reason:
     conv_gate already extracts binary-skip alpha; regime_mult already
     throttles size; 4h cadence concentrates dispersion noise more than
     xyz's daily horizon.
   - **Best zero-cost variant: 0.50-1.00** (Δsh −0.14, 17% DD reduction,
     Δnet CI [−1.39, +0.05] crosses 0).
   - **Best DD-reduction variant: 0.30-0.70** (Δsh −0.13, 33% DD reduction
     matches xyz target, Δnet CI [−2.00, −0.06] significantly negative).
   - **Strong losing-month absorption**: Dec 2025 −1.43 → −0.51 (64%↓),
     Apr 2026 −2.30 → −0.96 (58%↓).
   - **Decision pending**: deploy with overlay (lower DD, slight Sharpe
     cost) vs stay at baseline (max Sharpe). Risk priority dependent.
4. Forward-validate live N=15 → N=30 cycles. Current live (conv-only + Ridge)
   is +2.78 bps/cyc Sh +2.6. Conv+PM (no Ridge) expects ~+4 bps/cyc forward
   under live-model expectations.

### Files (validation)

- `ml/research/alpha_v9_pred_momentum.py` — gate impl + 2-fold sanity
- `ml/research/alpha_v9_pred_momentum_multioos.py` — paired 10-fold (LGBM-only)
- `ml/research/alpha_v9_pred_momentum_hardsplit.py` — frozen ensemble (PM alone)
- `ml/research/alpha_v9_pred_momentum_stack.py` — conv+PM stacking + `disp_overlay_lo/hi` parameter
- `ml/research/alpha_v9_pm_hybrid_stack.py` — conv+PM on production hybrid
- `ml/research/alpha_v9_revalidate_live_model.py` — live vs research execution model comparison
- **`ml/research/alpha_v9_disp_overlay.py` — pred_disp size-overlay sweep (xyz port)**
- Outputs: `outputs/pred_momentum_*`, `outputs/pm_hybrid_stack/`, `outputs/revalidate_live_model/`, `outputs/disp_overlay/`

### Caveats one more time

1. CI lower bound on Δnet barely > 0 for both LGBM-only stacked (+0.37) and hybrid stacked (+0.34). Strong point estimates, not 5σ.
2. Tail risk in extreme regimes (S+3+ accounts for 33% of bps from 3.6% of cycles) is unhedged.
3. conv+PM hasn't been hard-split-tested directly on either LGBM-only or hybrid (PM alone passed).
4. Ridge/PM conflict means production-hybrid lift is +1.0 Sh (not the +1.6 Sh LGBM-only result). Optimal Ridge weight under PM gate untested.
5. Forward live validation N=30 cycles still required before treating as deployed alpha.

---

## Universe expansion exploration (2026-05-09, all paths CLOSED)

**One-day comprehensive exploration. Conclusion: ORIG25 stays. Workflow is universe-specific.**

### Paths tested + outcomes

| Path | Specific test | Result |
|---|---|---|
| **Bundled additions** (FULL39, +DeFi3, +L1_3, +Quality6, +NonMeme10, +MemesOnly4) | `alpha_v9_universe_expand.py`, `alpha_v9_universe_curated.py` | ALL hurt by 2.0-3.3 Sh |
| **Per-symbol leave-one-in** | `alpha_v9_universe_leave_one_in.py` (14 candidates) | Only LDO (+0.12), 1000SHIB (+0.35) individually compatible; 12 reject |
| **Joint LDO+1000SHIB** | `alpha_v9_universe_27sym.py` | **ANTAGONISTIC** Δsh −2.03 (rank-shift non-linear) |
| **Forced cluster-bucketed (K_c=1 per cluster)** | `alpha_v9_clustered_backtest.py` | Sharpe **−2.12** (forces low-conviction picks from weak clusters) |
| **Capped cluster (≤2,3,4 per cluster)** | `alpha_v9_capped_backtest.py` | Sharpe **+0.21** (worse than uncapped FULL +0.88, cap dilutes alpha) |
| **Workflow generalization** (5 alternative 25-name baskets) | `alpha_v9_workflow_generalization.py` | Mean **−0.38** Sh, range [−2.06, +0.79]. ORIG25 +2.75 is rank 1/6 by 1.96 Sh margin |

### Two compounding root causes

1. **Rank-competition destabilizes PM gate** (per-symbol diagnostic in `alpha_v9_universe_diag.py`): adding names → more rank churn cycle-to-cycle → PM rejects more entries → K_avg drops 20% → effective deployment shrinks. Per-symbol IC and selection are unchanged; only K_actual collapses. Diagnostic confirmed: ORIG25-trained IC (+0.054) ≈ FULL39-trained IC on ORIG25 subset (+0.052), and NEW names are picked proportionally (34% vs 36% expected).

2. **Workflow is overfit to ORIG25** (`alpha_v9_workflow_generalization.py` test): 5 alternative 25-name baskets (swap5, swap10, top25_by_vol, 2 random25 seeds) average −0.38 Sh. ORIG25 +2.75 is rank 1/6 by huge margin. The workflow's success isn't generalizable; it's tied to ORIG25's specific cross-sectional structure.

### Why "v6_clean" is a configuration, not a workflow

The 30+ test audit (memory: project_v6_baseline.md) selected:
- 28 v6_clean features (from 50+ candidates) — all on ORIG25
- LGBM hyperparameters (num_leaves=63, min_data=100, λ_l2=3.0) via 27-combo grid — on ORIG25
- K=7 from {3,5,7,10,12} sweep — on ORIG25
- conv_gate p=0.30 from plateau — on ORIG25
- PM_M2_b1 over alternatives — on ORIG25
- per-name=1/7 over const-gross — on ORIG25

Every tunable was selected against ORIG25's loss surface. None of these are guaranteed to transfer to other universes.

### Implications for future planning

| Question | Answer |
|---|---|
| Can we add 5-10 quality names to ORIG25? | No. Bundled additions all hurt; even individually-compatible names (LDO+1000SHIB) fail jointly. |
| Will sector-bucketed selection rescue universe expansion? | No. Forced K_c=1 fails (forces weak-cluster picks). Capped also fails (dilutes alpha). |
| Will full retrain on FULL39 produce comparable Sharpe? | **Unlikely.** Workflow generalization test averaged −0.38 Sh on alt 25-baskets. Forward 39-name retrain expected payoff is much lower than initially estimated (was +2.0-3.0; now downgraded to ~+0 to +1.5 with high variance). |
| What if we need bigger universe later? | Plan for ~3-4 weeks of fresh audit work per new universe. Treat as new program, not extension. |
| For modest expansion now? | Path 1 only: ORIG25 + 1000SHIBUSDT (single-name, +0.35 Sh expected). 1-day deploy. |

### Files (all preserved)

Research scripts:
- `ml/research/alpha_v9_universe_expand.py`
- `ml/research/alpha_v9_universe_curated.py`
- `ml/research/alpha_v9_universe_diag.py`
- `ml/research/alpha_v9_universe_leave_one_in.py`
- `ml/research/alpha_v9_universe_27sym.py`
- `ml/research/alpha_v9_clustered_backtest.py`
- `ml/research/alpha_v9_capped_backtest.py`
- `ml/research/alpha_v9_workflow_generalization.py`
- `ml/research/portfolio_clustered.py`

Configs:
- `config/clusters_v1.json` — 6-cluster sector definitions

Outputs (per-cycle CSVs preserved):
- `outputs/universe_expand/`
- `outputs/universe_curated/`
- `outputs/universe_diag/`
- `outputs/universe_leave_one_in/`
- `outputs/universe_27sym/`
- `outputs/clustered_backtest/`
- `outputs/capped_backtest/`
- `outputs/workflow_generalization/`

---

## ⚠️ Cost-formula correction (2026-05-06)

Previously published Sharpe values (+3.30 / +3.63) assumed `cost_bps_per_leg
× 0.5 × Σ|Δw|` — the 0.5× was a vestigial "round-trip per leg" factor.
With HL one-way fees plugged in, fees were under-counted **2×**. Fixed in
`ml/research/alpha_v4_xs.py`. Live's `paper_bot.py` always used the correct
`fee × notional/equity` formula and was unaffected.

Corrected backtest economics at HL VIP-0 (4.5 bps one-way taker):
- net/cycle ≈ +0.7 bps (was claimed +4.33)
- Sharpe ≈ +0.6 (was claimed +3.63)

The strategy is still positive at corrected costs, but the deployment
thesis ("backtest CI excludes zero") no longer holds at 4.5 bps. Edge
becomes meaningful at HYPE-staked tiers (Silver/Gold) or maker fills.
See "Fee sensitivity" in STATUS.md.

## Summary for the next person

The strategy is deployable but **edge depends on fee tier**:

1. **h=288 K=5 ORIG25** (legacy): corrected multi-OOS Sharpe **~+0.5**.
2. **h=48 K=7 ORIG25** (current, running): corrected multi-OOS Sharpe
   **~+0.6**. Forward-test N=15 cycles: Sharpe +2.6 (above corrected
   backtest, within sample noise).

Forward-test confirms live pipeline matches backtest mechanically (live
fees match exactly, live gross alpha is running ABOVE corrected backtest
gross — possibly favorable regime, possibly real signal in fresh data).

Both use the **same v6_clean features** (28-column set). The only
differences between configs are HORIZON_BARS (288 vs 48), TOP_K (5 vs 7),
and the model artifact's training labels (24h vs 4h forward demeaned
return). Code is parameterized via env vars — no in-place edits needed
to switch.

### What's settled (don't redo)

- **rc=0.50 is optimal** at both h=288 and h=48 (plateau 0.50-0.70).
  Was 0.33; bumped 2026-05-03.
- **Universe is fixed at ORIG25** (the 25 perps that existed at v6_clean
  selection time). Adding the 14 newer perps reliably hurts on multi-OOS.
- **Feature set is fixed at v6_clean (28 cols).** 50+ feature
  modifications have been tested (DVOL, funding, horizon-matched 4h
  variants, lean trims, additive Stage 2 candidates). All hurt Sharpe.
  v6_clean is a tight local optimum.
- **Model architecture is fixed at LGBM ensemble (5 seeds).** Linear
  oracle confirms the model is at the data ceiling, not the architecture
  ceiling. NN/transformer would add complexity without alpha.
- **The alpha is multi-day cross-sectional reversion**, sampled at
  whatever cadence (h=48 or h=288). Long-window features (return_1d,
  dom_z_7d, corr_change_3d) ARE the signal, not stale momentum proxies.

### What's pending (operational)

1. **FAPI server migration**. Current dev server is geo-blocked from
   `fapi.binance.com`; bot runs on `--source hl` fallback. Move to a
   server with FAPI access (US/EU VPS), then deploy h=48 K=7 there.
   Runbook: `live/MIGRATION_FAPI.md`.
2. **Shadow-mode validation** (optional but recommended). Run h=48 K=7
   alongside live h=288 for 1 week, compare cycle-by-cycle PnL on real
   data before cutover.
3. **Stake 100+ HYPE for HL Bronze tier** (10% taker discount). Brings
   per-leg fee from 4.5 → 4.05 bps. Estimated additional Sharpe ~+0.2.

### What might still help (research, not engineering)

The h=48 vs h=288 split delivers the +0.3 Sharpe lift, but the
architectural ceiling is reached. Genuine future improvement requires
**different problem framing**, not more tuning of v6_clean:

1. **Stack h=48 + h=288 as parallel sleeves.** They sample the same
   alpha at different rates → highly correlated, modest diversification.
   Worth trying if dual deployment is cheap.
2. **L2 microstructure features** (order book imbalance, depth ratio,
   spread dynamics). Never tested. Requires Tardis-style L2 data feed.
3. **On-chain features** (whale flows, exchange in/outflows, stablecoin
   supply). Genuinely orthogonal to price-derived signals. Requires
   Glassnode-style data integration.
4. **Per-symbol options data** (BTC/ETH/SOL only — sparse coverage
   breaks the XS architecture). Could spawn a separate options-overlay
   strategy on those 3.

### What does NOT help (proven negative, don't re-test)

- Different horizons other than 48 or 288 (h=24/36/72/96/144 all give
  Sharpe < +1.5 with CIs touching zero — bimodal structure)
- Different K (K=2-10 sweep done at h=48; K=7 marginal best)
- Different universe (full 39, alt 25-curated, drop-one-out — all hurt)
- Different rc (0.33-1.00 sweep; 0.50 is peak at both horizons)
- Replacing 24h-window features with 4h variants (catastrophic -3.67 Sharpe)
- DVOL features, funding-rate features, regime-conditional MoE, ridge
  regression (all proven worse than baseline LGBM ensemble)

The path to a deployable strategy is below. **For deployment context,
read `docs/STATUS.md` first; for the historical research arc, read
`docs/METHODOLOGY_REVIEW.md`.**

---

## Historical: original Apr 30 framing (preserved)

The signal class (4h-horizon alpha-residual prediction from kline + aggTrade
features, traded as long-short cross-sectional or pair-trading) is **fully
characterized**:

- Real, statistically robust (rank IC +0.035 OOS)
- Magnitude ~5–10 bps gross per trade
- Below retail VIP-0 round-trip cost (12 bps naked / 24 bps hedged)

**[Note 2026-05-03: the cost claim was wrong — actual HL VIP-0 taker is
4.5 bps/leg, not 12. At realistic cost, the strategy is well above
breakeven, Sharpe +3.30 (h=288) to +3.63 (h=48). The "fully
characterized" assertion below was based on the over-cost framing.]**

**Don't try to push this signal harder via more features or models.** The
audits in `ml/research/alpha_*_audit.py` already enumerated what's available.
Marginal improvements (more features, deeper trees, ensemble tricks) have
diminishing returns and have been tried.

The path to a deployable strategy is one of three structural pivots, ranked
by ROI to attempt:

## Option A: Different horizon — TESTED, h=288 IS PREFERRED (Apr 30 corrected)

Earlier "1d rejected" verdict was wrong: it compared bps-per-cycle without
normalizing for cycle length. h=48 has 6 cycles/day; h=288 has 1. On a
per-year basis, h=288 wins at every realistic fee tier:

- VIP-0 retail: h=48 -164%/yr vs **h=288 -32%/yr**
- VIP-3 taker: h=48 -54%/yr vs **h=288 -2.3%/yr**
- VIP-3 + maker: h=48 +0.3%/yr vs **h=288 +12.4%/yr (Sharpe 0.42)**
- VIP-9 maker: h=48 +37%/yr (Sharpe 1.15) vs h=288 +22%/yr (only place
  h=48 wins, via more cycles per year — note: returns are arithmetic
  per-cycle bps × cycles/year, not compounded)

Implementation: `alpha_v4_xs_1d.py` at HORIZON=288 with non-overlapping
sampling (rebalance every h bars). β-neutral OOS captures alpha cleanly
(ret_BN = +7.41 bps/cycle vs equal-weight ret +4.50, alpha +9.78). See
`docs/METHODOLOGY_REVIEW.md` Apr 30 follow-up for full fee-sensitivity
tables.

**Why this could work**: at h=288 (1d), the residual signal benefits from:
- Slower signal decay (per-trade alpha 30-50 bps vs 5-10)
- Fewer trades → lower amortized cost
- Different feature regime (1d momentum + reversal patterns)

**What's needed**:
- Change `HORIZON = 48` → `HORIZON = 288` across `alpha_v3.py` / `alpha_v4_xs.py`
- Re-audit features against the 1d alpha target (`alpha_feature_audit.py` with `HORIZON = 288`)
- Adjust cost model: cost is per-trade not per-day, so slow-trading helps
- Walk-forward fold sizes need to grow (50d train won't have many h=288 examples)

**Estimated effort**: 1-2 days. Mostly parameter changes and one round of
audits. Most code already supports it.

## Option B: Lower-fee venue / maker execution — STILL THE DEPLOYMENT LEVER

**Idea**: VIP-3 fee tier (~0.025% taker per leg, ~5 bps RT) and/or
post-only maker orders (~50% fill rate in calm regimes) close the remaining
cost gap.

**With Apr 30 corrected accounting at h=288 OOS β-neutral** (turnover-aware,
total turnover/cycle = 1.34):

| Tier | Fee/leg | Net/cycle | Net/year | Ann. Sharpe |
|---|---|---|---|---|
| VIP-0 retail | 12 bps | -8.69 | -32% | -1.09 |
| VIP-3 taker | 6 bps | -0.64 | -2.3% | -0.08 |
| VIP-3 + maker | 3 bps | +3.38 | +12.4% | +0.42 |
| VIP-9 maker | 1 bps | +6.07 | +22.1% | +0.76 |

Cost saving per tier ≠ per-leg fee saving — saving is `Δfee/leg × turnover_sum`
where turnover_sum is ~1.34 at h=288 (~0.83 at h=48). So a 6 bps/leg fee cut
saves ~8 bps net/cycle at h=288 (not 12).

**Open task**: a real maker-fill simulator (require L2 / Tardis data) to
refine the 50% fill assumption and check queue-position economics. Sharpe
0.42 at VIP-3+maker is marginal; better fill modelling could move it
either direction.

**Bootstrap CI caveat (Apr 30)**: at h=288 OOS, only 90 cycles of data.
Block-bootstrap 95% CI on Sharpe at VIP-3+maker = [-4.9, +4.0] — point
estimate not statistically distinguishable from zero. The 1d > 4h
relative ordering is robust; absolute deployment economics are not.
Require wider OOS sample before deployment.

**Why this works**: alpha is real, just below cost line. Cost reduction is
direct.

**What's needed**:
- Either a higher-volume venue / VIP tier (probably not realistic short-term)
- Or a post-only execution simulation: model fill probability vs queue position,
  measure effective fee given a maker-tilt strategy
- Maker simulation requires L2 orderbook data (not free; Binance Vision is L1)

**Estimated effort**: maker simulation is 1-2 weeks if including data
collection and modeling.

## Apr 30 — Plan-driven signal improvements (v4 → v6)

The "feature ceiling" suggested in earlier sections turned out to be partly
about how features were structured, not whether more existed. After running
a structured plan:

1. **xs_rank features** (per-bar pctile rank within universe): biggest single
   win, +1.4 Sharpe at deployment tier. Address scale heterogeneity across symbols.
2. **Top-K reduction**: K=5 → K=7 was a free +0.7 Sharpe — within-quintile
   rank carries genuine signal that gets diluted at K=5.
3. **Kline-flow features** (obv_z_1d, vwap_*, mfi): +0.3 Sharpe, modest.
4. **Universe trim by IS-IC bottom-quartile**: +0.3 Sharpe, mostly tightens CI.
5. **Funding-rate features**: regressed despite strongest single-feature IC.
   Signal already captured by v6's basket-relative features.

**Best config: v6 (32 features) + K=7 + β-neutral + IS-trim** →
OOS Sharpe +3.94 with 95% CI [+0.37, +6.74] at VIP-3+maker.
That's 9× over the corrected v4 baseline.

The remaining lever in this codebase is true microstructure data (Phase 3
in the signal-quality plan): pulling aggTrades for top-N symbols and
computing TFI / VPIN / Kyle's λ. This is genuinely orthogonal to klines
and is the next experiment.

Validation gap: 90 OOS cycles is the binding constraint on certainty. CI on
the best Sharpe spans [+0.37, +6.74] — point estimate is high but the
window is narrow. Forward testing on data past 2026-04-28 or pulling
additional history is the rigorous next step before deployment.

## Option C: Add orderbook L2 features

**Idea**: microprice, depth imbalance, queue position add 5-10 bps IC
contribution typical in microstructure research. Capture pending aggressor
pressure that price/volume miss.

**Why this could work**: orthogonal information source. The kline+aggTrade
universe is heavily arbed; L2 has lower mining ceiling.

**What's needed**:
- Tardis subscription or own L2 collection (8-12 GB / symbol / month)
- Storage scaling: 25 symbols × 12 months × 10 GB ≈ 3 TB
- New feature module `features_ml/orderbook.py`
- Re-run audits and v4 with orderbook features added

**Estimated effort**: 2-4 weeks counting data infrastructure.

## Other notes for the next person

### Where to start reading
1. `docs/METHODOLOGY_REVIEW.md` — full audit trail, numbered issues, fix attempts, results.
2. `ml/research/alpha_v3.py` — read for the curated 17-feature alpha-tailored model.
3. `ml/research/alpha_v4_xs.py` — read for the cross-sectional pipeline.
4. `features_ml/cross_sectional.py` — basket construction and basket-relative features.

### Look-ahead bugs to watch for
The codebase had two real look-ahead bugs found during this research:
1. **Sharpe target normalization shift** — `rolling.shift(1)` should be
   `rolling.shift(horizon)` because forward returns at horizon h require
   prices h bars ahead. Fixed in `_make_alpha_label` of all `alpha_v*.py`.
2. **VPIN bucket sizing** — used `total_vol.iloc[-1]` (full dataset) to
   size buckets; now uses trailing 7d window per bar. Fixed in
   `features_ml/trade_flow.py::_vpin`.

When adding new features, sanity-check by computing IC on `fwd_ret` *one bar
shifted* — features that have suspicious +0.10 IC vs forward return often
have a hidden lookback that uses bar-t close in computing bar-t feature.

### Fold purging is non-trivial
`ml/cv.py::FoldSpec` + `split_features_by_fold` handles standard cases, but
the alpha-residual labels also have an `exit_time` column that must be
purged from train. The `_expanding_train` helpers in each `alpha_v*.py`
script handle this — copy the pattern, don't reinvent.

### Per-symbol idiosyncrasies
- BTC alpha (vs ETH ref) has highest OOS IC (+0.08 in v3) but worst trade
  P&L because un-hedged BTC eats the alpha via market direction noise.
- ETH alpha (vs BTC ref) is the only consistently profitable single-pair
  setup, but only naked at q=0.99 OOS (+8.49 bps net) — and that's regime-
  dependent, not robust.
- SOL alpha is broken at this horizon. Don't waste time trying to make
  SOL work alone.

### Don't get tricked by walk-forward
WF results are 8-12 bps higher than OOS holdout. The gap is partly
hyperparameter overfitting (q, h, regime cutoff chosen by inspecting all
WF folds), partly genuine distribution shift. Always verify on the 90d
holdout before drawing conclusions.

## Open questions

- Does longer horizon (1d/1w) actually deliver 30+ bps alpha on these symbols?
  (Untested; is Option A above.)
- Does adding funding-rate features help? (Free public data, untried.
  Funding extremes are documented mean-reversion signal in perp basis.)
- Does maker-tilt simulation produce realistic fill rates? (Untested.
  Would require L2 data.)

## Contact

Original research and methodology by yq during the P-2026-001 program.
Code is research-grade; production hardening (live trading integration,
risk limits, monitoring) is out of scope.
