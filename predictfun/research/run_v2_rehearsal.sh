#!/usr/bin/env bash
# pm_ladder_v2 rehearsal on the week-2 read days (in-sample; walk-forward refits from pf_v2_refit.py), then the v1 order-change causes
# and the per-day grid parity. Sequential run_light jobs; stops at the first failure.
set -u
cd /home/yuqing/ctaNew
ARM='MICRO|G4+G2+G6+G5+G3+G1+PM90+GS+LD2+XNA+XMIN1+XV+S60|mid|AM:5:25:95'
LG=data/predict_fun/research/logs; R=predictfun/research/run_light.sh
COMMON="PF_RL_DDEC=2 PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_TIMING=e2 PF_TXTIME=ws PF_TXKEY=tk"
J="PF_TXJIT=frame"; X="PF_TXSHIFT=-50 PF_QACT=max"
die() { echo "FAILED $1"; exit 1; }
# 1. engine parity: v2 engine with v1 lines + v1 values = round i
env $COMMON $J PF_PREDS=preds_t60k PF_ARMS="$ARM" PF_TAG=v2pJ PF_LINES_FILE=/home/yuqing/ctaNew/data/predict_fun/research/step4/minute_lines_t60k.json \
  $R v2_parity_J -- python3 -m predictfun.research.pf_step4_ladder_v2 20261103 tokyoreal custom > $LG/v2_parity_J.log 2>&1 || die parity_run
python3 -m predictfun.research.pf_v2_engine_check lJi v2pJ 20261103 > $LG/v2_parity_check.log 2>&1 || die parity_check
# 1b. forward-enabled v1 engine: default FORWARD on 11-03, PF_FORWARD=20261109 on 11-02; frozen-model apply parity on 11-03
env $COMMON $J PF_PREDS=preds_t60k PF_ARMS="$ARM" PF_TAG=fwdpJ $R fwd_parity_1103 -- python3 -m predictfun.research.pf_step4_ladder_fwd 20261103 tokyoreal custom > $LG/fwd_parity_1103.log 2>&1 || die fwd_run_1103
python3 -m predictfun.research.pf_v2_engine_check lJi fwdpJ 20261103 fwd > $LG/fwd_parity_check_1103.log 2>&1 || die fwd_check_1103
env $COMMON $J PF_PREDS=preds_t60k PF_ARMS="$ARM" PF_TAG=fwdpJ PF_FORWARD=20261109 $R fwd_parity_1102 -- python3 -m predictfun.research.pf_step4_ladder_fwd 20261102 tokyoreal custom > $LG/fwd_parity_1102.log 2>&1 || die fwd_run_1102
python3 -m predictfun.research.pf_v2_engine_check lJi fwdpJ 20261102 fwd > $LG/fwd_parity_check_1102.log 2>&1 || die fwd_check_1102
PF_V1_APPLY_CHECK=1 $R v1_apply_check -- python3 -m predictfun.research.pf_v1_apply > $LG/v1_apply_check.log 2>&1 || die v1_apply_check
# 2. v2 replays, week-2 days, J and X
for TM in J X; do
  EX=$J; [ $TM = X ] && EX=$X
  for D in 20261031 20261101 20261102 20261103; do
    env $COMMON $EX PF_PREDS=preds_v2w7k PF_ARMS="$ARM" PF_TAG=v2r$TM PF_LINES_FILE='/home/yuqing/ctaNew/data/predict_fun/research/step5/{day}/lines_v2w7.json' \
      $R v2r_${TM}_$D -- python3 -m predictfun.research.pf_step4_ladder_v2 $D tokyoreal custom > $LG/v2r_${TM}_$D.log 2>&1 || die v2r_${TM}_$D
  done
done
$R v2_rehearsal_report -- python3 -m predictfun.research.pf_v2_rehearsal_report > $LG/v2_rehearsal_report.log 2>&1 || die v2_report
echo V2 REPLAYS DONE
# 3. v1 order-change causes (18 runs)
predictfun/research/run_causes.sh || die causes
$R causes_report -- python3 -m predictfun.research.pf_causes_report > $LG/causes_report.log 2>&1 || die causes_report
# 4. per-day grid parity
env PF_RL_DDEC=2 PF_V2_GRID_CHECK=1 $R v2_grid_check -- python3 -m predictfun.research.pf_v2_grid_day > $LG/v2_grid_check.log 2>&1 || die grid_check
echo ALL DONE
