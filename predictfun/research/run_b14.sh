#!/usr/bin/env bash
# B14 (2026-11-11, USER "run the prediction guard"): the live guard rebuilt with B13a's 13 pressure features, run as the frozen v3
# strategy's guard on the 9 consumed days. Design, readings and controls: predictfun/research/pf_b14_pressure_guard.py (committed
# before this first run). Sequential run_light jobs; stops at the first failure. Writes only new files / dirs (refuses existing ones).
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; LG=data/predict_fun/research/logs/b14; S4=data/predict_fun/research/step4
DAYS="20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103"
run() { local tag=$1; shift; $R "$tag" -- "$@" > "$LG/$tag.log" 2>&1 || { echo "FAILED $tag"; tail -3 "$LG/$tag.log"; exit 1; }; }
for t in b14rJ b14rX; do [ -e $S4/ladder_tokyoreal_custom_e2_preds_t60k_$t ] && { echo "REFUSED: run tag $t exists"; exit 1; }; done
for t in b14eJ b14eX; do [ -e $S4/ladder_tokyoreal_custom_e2_preds_e60km_$t ] && { echo "REFUSED: run tag $t exists"; exit 1; }; done
git diff --quiet HEAD -- predictfun/research/pf_b14_pressure_guard.py predictfun/research/run_b14.sh || { echo "REFUSED: B14 code differs from the commit"; exit 1; }
mkdir -p $LG
git rev-parse HEAD > $LG/code_commit.txt
sha256sum predictfun/research/pf_b14_pressure_guard.py predictfun/research/run_b14.sh predictfun/research/pf_b13a_pressure_model.py \
  predictfun/research/pf_step4_ladder_v3.py predictfun/research/pf_preds_startknown.py predictfun/research/pf_model_spec.py > $LG/code_sha256.txt
M=predictfun.research.pf_b14_pressure_guard
run b14_features python3 -m $M features
run b14_build python3 -m $M build
PF_RL_DDEC=2 run b14_sk python3 -m predictfun.research.pf_preds_startknown preds_e60 $DAYS
run b14_map python3 -m $M map
ARM_3=$(python3 -c "from predictfun.research.pf_ladder_v3_build import ARM; print(ARM)")
unset PF_R60_FILE PF_LINES_FILE PF_TXJIT PF_TXSHIFT PF_QACT PF_ARMS PF_TAG PF_FORWARD PF_PREDS
export PF_RL_DDEC=2 PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_TIMING=e2 PF_TXTIME=ws PF_TXKEY=tk
for G in r e; do
  if [ $G = r ]; then P=preds_t60k; else P=preds_e60km; fi
  for D in $DAYS; do
    PF_PREDS=$P PF_TXJIT=frame PF_ARMS="$ARM_3" PF_TAG=b14${G}J run b14${G}J_$D python3 -m predictfun.research.pf_step4_ladder_v3 $D tokyoreal custom
    PF_PREDS=$P PF_TXSHIFT=-50 PF_QACT=max PF_ARMS="$ARM_3" PF_TAG=b14${G}X run b14${G}X_$D python3 -m predictfun.research.pf_step4_ladder_v3 $D tokyoreal custom
  done
done
run b14_report python3 -m $M report
echo B14_DONE
