#!/usr/bin/env bash
# Two consumed days, one candidate, two timing scenarios; no forward data.
set -euo pipefail
cd /home/yuqing/ctaNew
R=predictfun/research
DEST=data/predict_fun/research/step4/incremental_20261107
# Prevent inherited PF_* switches from silently changing the experiment.
for VAR in ${!PF_@}; do unset "$VAR"; done
$R/run_light.sh incsz_declare -- python3 -m predictfun.research.pf_incremental_replay declare
ARMS=$(cat "$DEST/arms.txt")
COMMON=(PF_RL_DDEC=2 PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100
        PF_TIMING=e2 PF_TXTIME=ws PF_TXKEY=tk PF_PREDS=preds_t60k PF_FILLPRIO=price PF_OWNCROSS=on
        PF_LEGORDER=price PF_REJRESET=on PF_POLICYVIEW=known "PF_ARMS=$ARMS")
env "${COMMON[@]}" PF_TXJIT=frame PF_TXSHIFT=0 PF_QACT= PF_MAX_WINDOWS=8 PF_TAG=incsz_20261107_smokeJ \
  $R/run_light.sh incsz_smoke -- python3 -m predictfun.research.pf_step4_ladder_incremental 20261024 tokyoreal custom \
  > "$DEST/smoke.log" 2>&1
$R/run_light.sh incsz_check_smoke -- python3 -m predictfun.research.pf_incremental_replay smoke > "$DEST/smoke_check.log" 2>&1
for TM in J X; do
  if [ "$TM" = J ]; then EXTRA=(PF_TXJIT=frame PF_TXSHIFT=0 PF_QACT=); else EXTRA=(PF_TXJIT= PF_TXSHIFT=-50 PF_QACT=max); fi
  for DAY in 20261024 20261101; do
    env "${COMMON[@]}" "${EXTRA[@]}" PF_MAX_WINDOWS=0 "PF_TAG=incsz_20261107_$TM" \
      $R/run_light.sh "incsz_${TM}_$DAY" -- python3 -m predictfun.research.pf_step4_ladder_incremental "$DAY" tokyoreal custom \
      > "$DEST/${TM}_$DAY.log" 2>&1
  done
done
$R/run_light.sh incsz_report -- python3 -m predictfun.research.pf_incremental_replay report > "$DEST/report.log" 2>&1
echo "Complete: $DEST/report.json"
