# Convexity v3 — LIVE forward-test deploy runbook (parallel to v1)

v3 = the regime-gate stack (validated backtest **+3.44 Sharpe** on PIT walk-forward preds, `run_convexity_v3_regime_gate.sh`).
This deploys it as a **parallel forward-test** alongside live v1 — it does NOT replace v1.

## Package (built + validated on the research box)
| file | role |
|---|---|
| `live/train_v3_artifact.py` | fit deployable per-symbol Ridge, both books (base `V0_LEAN`, residrev `V0_LEAN+RR`), fit_cut = latest panel − 1d |
| `live/models/convexity_v3_{base,residrev}_model.pkl` | the trained artifact (this build: **fit_cut 06-29**, 175 syms) |
| `live/predict_v3_incremental.py` | each cycle: load artifact → predict trailing window → write `v3_live/{base,long}.parquet` |
| `live/run_convexity_v3_live.sh` | live launcher: frozen v3 env (`REGIME_GATE=1 BULL_DEEP_THR=0.15 …`) + predictor + `bot --cycle` |
| `live/parity_v3.sh` | wiring parity gate (live `--cycle` vs `--replay-all`): **0/359 decision mismatches**, 0.12% seam drift |

## Validated on the research box
- Artifact predicts all 174 syms sanely; parity = identical decisions.
- Warm-start proven: seed preds (PIT backtest through 05-30) → extend via artifact → bootstrap state → `--cycle` runs warm.

## Deploy on the REAL boxes (NOT the research sandbox — its `main` is diverged, can't push safely)

### Training box (synced `main` + Binance feed)
1. Get v3 code onto `main`: `git merge convexity-v3-regime-gate` (or cherry-pick the 5 files above + the committed v3 driver). Keep the off-by-default research levers (`CORR_CEIL`/`DECOUPLE_CEIL`) OUT.
2. Refresh data + **retrain fresh**: `PYTHONPATH=. python3 live/train_v3_artifact.py`  → artifact @ fit_cut ~07-01 (fresher than the 06-29 shipped here).
3. Commit `live/models/convexity_v3_{base,residrev}_model.pkl` + code → `git push origin main`.
4. **Monthly retrain going forward:** add `train_v3_artifact.py` to the monthly cron (mirror `monthly_retrain.sh`; commit+push the v3 models each month).

### Live / execution box (parallel to v1)
5. `git pull` (code + v3 models).
6. **Seed the gate history + bootstrap warm state** (the regime gate needs 180-bar/30d trailing history — do NOT start cold):
   ```bash
   # seed live preds from the validated PIT backtest preds (PIT-clean gate history), then extend:
   mkdir -p live/state/convexity/v3_live/state
   cp live/state/convexity/hl_lean175/v0full_hl60.parquet      live/state/convexity/v3_live/base.parquet
   cp live/state/convexity/hl_residrev_lean/v0full_hl60.parquet live/state/convexity/v3_live/long.parquet
   V3_PREDS_DIR=live/state/convexity/v3_live python3 live/predict_v3_incremental.py
   # bootstrap warm positions.json (equity + gate + sleeve state):
   bash live/run_convexity_v3_regime_gate.sh live/state/convexity/v3_live \
     CONVEXITY_PREDS_PATH=live/state/convexity/v3_live/base.parquet \
     CONVEXITY_PREDS_LONG=live/state/convexity/v3_live/long.parquet
   ```
   (If `hl_lean175` preds aren't on the box, regenerate them once with `gen_v0lean_validate.py`, or let `predict_v3_incremental` self-seed via `PREDICT_SEED_DAYS=300` — slightly look-ahead in the seed, PIT-clean copy preferred.)
7. Launch parallel to v1: `tmux new -d -s cvx3 'bash live/run_convexity_v3_live.sh'`  (v1 stays on `cvx1`).
8. Monitor both; compare v3 vs v1 forward.

## IMPORTANT — honest caveats
- **Forward-test valid period = fit_cut+1 onward (~07-01+).** The frozen artifact predicting bars ≤ fit_cut (the 05-30→06-30 catch-up) is look-ahead — that's *warmup only*, do NOT count its PnL. (The bootstrap replay's headline ~+40k/Sh 5.5 is this look-ahead warmup, NOT performance; the real validated number is **+3.44**.)
- **Regime gate warms over ~30 live days** — its trailing window is artifact-look-ahead at live-start, becoming clean by ~08-01. Expect it near-full-gross initially.
- **Funding source:** the launcher calls `ingest_funding_fapi.py`. If the box is geo-blocked (451), swap to the Vision-monthly + `unpack_funding_export.py` path (funding degrades to stale, not NaN, intra-month — tolerable).
- **Expectation is regime-conditional** (OOS finding): strong when idiosyncratic dispersion is present, negative in trending-bull regimes (2024-type). Size small; this is a *test*, not a promotion.
