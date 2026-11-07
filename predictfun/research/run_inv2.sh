#!/usr/bin/env bash
# CAP10 + RO60-240 on pm_ladder_v1 (pf_inv2.py docstring; declared in STATUS before launch). Sequential run_light jobs:
# 4 arms x 9 days x J / X (engine pf_step4_ladder_inv.py, same env as run_inv.sh) -> pf_inv2 check -> pf_inv2 report.
set -u
cd /home/yuqing/ctaNew
S4=data/predict_fun/research/step4; LG=data/predict_fun/research/logs; R=predictfun/research
for t in i2J i2X; do [ -e $S4/ladder_tokyoreal_custom_e2_preds_t60k_$t ] && { echo "REFUSED: run tag $t exists"; exit 1; }; done
[ -f $S4/inv_r60.json ] || { echo "REFUSED: no inv_r60.json"; exit 1; }
{ git rev-parse HEAD; sha256sum $R/pf_step4_ladder_inv.py $R/pf_ladder_inv_build.py $R/pf_inv2.py $R/pf_inv_report.py $R/pf_cap_check.py \
  $R/pf_cap10_attribution.py $R/run_inv2.sh $S4/inv_r60.json; } > $LG/inv2_code_sha256.txt
COMMON="PF_RL_DDEC=2 PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_TIMING=e2 PF_TXTIME=ws PF_TXKEY=tk PF_PREDS=preds_t60k PF_R60_FILE=/home/yuqing/ctaNew/$S4/inv_r60.json"
ARMS=$(python3 -m predictfun.research.pf_inv2 arms) || exit 1
for TM in J X; do
  if [ $TM = J ]; then EXTRA="PF_TXJIT=frame"; else EXTRA="PF_TXSHIFT=-50 PF_QACT=max"; fi
  for D in 20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103; do
    env $COMMON $EXTRA PF_ARMS="$ARMS" PF_TAG=i2$TM $R/run_light.sh inv2_${TM}_$D -- python3 -m predictfun.research.pf_step4_ladder_inv $D tokyoreal custom \
      > $LG/inv2_${TM}_$D.log 2>&1 || { echo "FAILED $TM $D"; exit 1; }
  done
done
$R/run_light.sh inv2_check -- python3 -m predictfun.research.pf_inv2 check > $LG/inv2_check.log 2>&1 || { echo "CHECK FAILED"; exit 1; }
$R/run_light.sh inv2_report -- python3 -m predictfun.research.pf_inv2 report > $LG/inv2_report.log 2>&1 || { echo "REPORT FAILED"; exit 1; }
echo ALL DONE
