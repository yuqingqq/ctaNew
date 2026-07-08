#!/usr/bin/env bash
# Q2 — 2022 HOLDOUT, ONE SHOT (pre-registered in RESEARCH_LOOP_20260707.md BEFORE pred generation).
# Cells (full env pinned below, F12): A=vanilla reference, B=KEEPSET4, C=KEEPSET4+deep-bull overlay.
# Window: cold start 2022-01-01 (same convention as the 2023-01 OOS cells — verified: those replays
# begin scoring at window open with REGIME_GATE_MINHIST warm-up inside the window).
# Preds: hl_2022_res_{base,long} from the frozen generator live/gen_residual_target_2022.py.
# ONE-SHOT ENFORCEMENT: refuses to run if the output dir exists. Repeat allowed ONLY for
# code-defect-class bugs (crash/wrong column), never result-dependent choices. No further 2022
# cells may ever be added (no-argmax at N=1).
set -uo pipefail
R=/home/yuqing/ctaNew; cd "$R"; export PYTHONPATH=$R
OUT=$R/live/state/longtail/holdout2022
[ -e "$OUT" ] && { echo "ONE-SHOT: $OUT exists — refusing"; exit 2; }
mkdir -p "$OUT"
EXEC=(COST_BPS_LEG=9 FEE_BPS_FILL=4.5 CHARGE_FUNDING=1 CONVEXITY_PIT_DVOL=1 XS_LEAN=1
      CONVEXITY_UNIVERSE_META=outputs/vBTC_features/panel_expanded_v0.parquet
      DEPTH_COST_CSV=live/state/v3loop/persym_cost_cal.csv DEPTH_COST_TIER=cost_10k
      CONVEXITY_DVOL_CACHE_PKL=live/state/v3loop/ddloop/_dvol_cache.pkl)
# VANILLA (locked harness base; BULL_MODE=default -> mom30 book in bull, the documented vanilla)
VAN=(STRAT_K=2 STRAT_K_LONG=1 BEAR_K=0 SIDE_MODE=default SIZING_MODE=inv_sqrt_vol
     REGIME_GATE=0 BEAR_DEPTH_RAMP=0 BULL_MODE=default CONC_CAP=0.99 CONC_CAP_SINGLE_EXEMPT=1
     STOP_SKIP_REGIMES=side,bear,bull SHORT_MIN_RET3D=-999 LONG_MAX_RET3D=999)
# KEEPSET4 risk layer (SIDE_BETA_NEUT deliberately unset = bot default 1, the canonical harness value)
K4=(BEAR_MODE=equal REGIME_GATE=1 REGIME_GATE_W=180 REGIME_GATE_FLOOR=0.0 REGIME_GATE_K=2
    REGIME_GATE_MINHIST=60 REGIME_GATE_MODE=binary REGIME_GATE_UNIV=full
    STOP_SKIP_REGIMES=bear STOP_K_SIGMA=2.0 BULL_GROSS_MULT=0.0)
DEEP=(BULL_DEEP_THR=0.15 BULL_DEEP_MODE=mom1d_long)
PB=$R/live/state/convexity/hl_2022_res_base/v0full_hl60.parquet
PL=$R/live/state/convexity/hl_2022_res_long/v0full_hl60.parquet
run(){ local cell=$1; shift
  local sd=$OUT/$cell; mkdir -p "$sd"
  env "${EXEC[@]}" "$@" \
      CONVEXITY_PREDS_PATH=$PB CONVEXITY_PREDS_LONG=$PL CONVEXITY_STATE=$sd \
      python3 -m live.convexity_paper_bot --replay-all > "$sd/run.log" 2>&1
  echo "$cell rc=$? $(grep -o 'Sharpe_ann[^,]*' $sd/run.log | tail -1)"
  env "${EXEC[@]}" "$@" true 2>/dev/null | true; }
env "${EXEC[@]}" "${VAN[@]}" printenv | grep -E "STRAT_|BEAR_|SIDE_|BULL_|REGIME_|STOP_|CONC_|COST_|FEE_|DEPTH_|XS_|CHARGE_|CONVEXITY_PIT" | sort > $OUT/env_A.txt
env "${EXEC[@]}" "${VAN[@]}" "${K4[@]}" printenv | grep -E "STRAT_|BEAR_|SIDE_|BULL_|REGIME_|STOP_|CONC_|COST_|FEE_|DEPTH_|XS_|CHARGE_|CONVEXITY_PIT" | sort > $OUT/env_B.txt
env "${EXEC[@]}" "${VAN[@]}" "${K4[@]}" "${DEEP[@]}" printenv | grep -E "STRAT_|BEAR_|SIDE_|BULL_|REGIME_|STOP_|CONC_|COST_|FEE_|DEPTH_|XS_|CHARGE_|CONVEXITY_PIT" | sort > $OUT/env_C.txt
sha256sum "$PB" "$PL" > $OUT/preds.sha256
run A "${VAN[@]}" &
run B "${VAN[@]}" "${K4[@]}" &
run C "${VAN[@]}" "${K4[@]}" "${DEEP[@]}" &
wait
echo HOLDOUT2022DONE
