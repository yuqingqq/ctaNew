#!/usr/bin/env bash
# D1 inventory mechanisms on pm_ladder_v1 (pf_inv_report.py docstring; declared in STATUS before launch). Sequential run_light jobs:
# never-trigger controls (10-24, 11-01, J) -> pf_inv_check ctrl -> 11 arms x 9 days x J / X -> pf_inv_check main -> pf_inv_report.
set -u
cd /home/yuqing/ctaNew
S4=data/predict_fun/research/step4; LG=data/predict_fun/research/logs; R=predictfun/research
for t in ivc ivJ ivX; do [ -e $S4/ladder_tokyoreal_custom_e2_preds_t60k_$t ] && { echo "REFUSED: run tag $t exists"; exit 1; }; done
[ -f $S4/inv_r60.json ] || { echo "REFUSED: no inv_r60.json"; exit 1; }
{ git rev-parse HEAD; sha256sum $R/pf_step4_ladder_inv.py $R/pf_ladder_inv_build.py $R/pf_inv_r60.py $R/pf_inv_report.py $R/pf_inv_check.py \
  $R/pf_cap_check.py $R/run_inv.sh $S4/inv_r60.json; } > $LG/inv_code_sha256.txt
COMMON="PF_RL_DDEC=2 PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_TIMING=e2 PF_TXTIME=ws PF_TXKEY=tk PF_PREDS=preds_t60k PF_R60_FILE=/home/yuqing/ctaNew/$S4/inv_r60.json"
CTRL=$(python3 -m predictfun.research.pf_inv_report arms ctrl) || exit 1
MAIN=$(python3 -m predictfun.research.pf_inv_report arms main) || exit 1
for D in 20261024 20261101; do
  env $COMMON PF_TXJIT=frame PF_ARMS="$CTRL" PF_TAG=ivc $R/run_light.sh inv_c_$D -- python3 -m predictfun.research.pf_step4_ladder_inv $D tokyoreal custom \
    > $LG/inv_c_$D.log 2>&1 || { echo "FAILED ctrl $D"; exit 1; }
done
$R/run_light.sh inv_check_ctrl -- python3 -m predictfun.research.pf_inv_check ctrl > $LG/inv_check_ctrl.log 2>&1 || { echo "CTRL CHECK FAILED"; exit 1; }
for TM in J X; do
  if [ $TM = J ]; then EXTRA="PF_TXJIT=frame"; else EXTRA="PF_TXSHIFT=-50 PF_QACT=max"; fi
  for D in 20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103; do
    env $COMMON $EXTRA PF_ARMS="$MAIN" PF_TAG=iv$TM $R/run_light.sh inv_${TM}_$D -- python3 -m predictfun.research.pf_step4_ladder_inv $D tokyoreal custom \
      > $LG/inv_${TM}_$D.log 2>&1 || { echo "FAILED $TM $D"; exit 1; }
  done
done
$R/run_light.sh inv_check_main -- python3 -m predictfun.research.pf_inv_check main > $LG/inv_check_main.log 2>&1 || { echo "MAIN CHECK FAILED"; exit 1; }
$R/run_light.sh inv_report -- python3 -m predictfun.research.pf_inv_report > $LG/inv_report.log 2>&1 || { echo "REPORT FAILED"; exit 1; }
echo ALL DONE
