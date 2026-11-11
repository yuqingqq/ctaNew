#!/usr/bin/env bash
# B16c (2026-11-11): B16 with the guard trained on the combined 1 s / 5 s / 60 s value of our resting order (pf_b16_queue_label.py
# with an output path -> grid_ql2.parquet), arms qc (32 features) / qd (+13 pressure), the frozen v3 strategy on the 9 consumed days at
# J and X; benchmark = B14's REF replays. Design, readings, controls: pf_b16c_combined_guard.py (committed before this first run).
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; LG=data/predict_fun/research/logs/b16c; S4=data/predict_fun/research/step4
DAYS="20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103"
QL2=$S4/badfill_info_tx_tk/grid_ql2.parquet
run() { local tag=$1; shift; $R "$tag" -- "$@" > "$LG/$tag.log" 2>&1 || { echo "FAILED $tag"; tail -3 "$LG/$tag.log"; exit 1; }; }
for t in b16cJ b16cX; do [ -e $S4/ladder_tokyoreal_custom_e2_preds_qc60km_$t ] && { echo "REFUSED: run tag $t exists"; exit 1; }; done
for t in b16dJ b16dX; do [ -e $S4/ladder_tokyoreal_custom_e2_preds_qd60km_$t ] && { echo "REFUSED: run tag $t exists"; exit 1; }; done
for t in b14rJ b14rX; do [ -e $S4/ladder_tokyoreal_custom_e2_preds_t60k_$t ] || { echo "REFUSED: benchmark run $t missing"; exit 1; }; done
git diff --quiet HEAD -- predictfun/research/pf_b16_queue_label.py predictfun/research/pf_b16c_combined_guard.py predictfun/research/run_b16c.sh \
  predictfun/research/pf_b16_queue_guard.py predictfun/research/pf_b14_pressure_guard.py || { echo "REFUSED: B16c code differs from the commit"; exit 1; }
mkdir -p $LG
git rev-parse HEAD > $LG/code_commit.txt
sha256sum predictfun/research/pf_b16_queue_label.py predictfun/research/pf_b16c_combined_guard.py predictfun/research/run_b16c.sh \
  predictfun/research/pf_b16_queue_guard.py predictfun/research/pf_b14_pressure_guard.py predictfun/research/pf_step4_ladder_v3.py > $LG/code_sha256.txt
M=predictfun.research.pf_b16c_combined_guard
PF_RL_DDEC=2 run b16c_label python3 -m predictfun.research.pf_b16_queue_label $QL2
run b16c_build_qc python3 -m $M build qc
run b16c_build_qd python3 -m $M build qd
PF_RL_DDEC=2 run b16c_sk_qc python3 -m predictfun.research.pf_preds_startknown preds_qc60 $DAYS
PF_RL_DDEC=2 run b16c_sk_qd python3 -m predictfun.research.pf_preds_startknown preds_qd60 $DAYS
run b16c_map_qc python3 -m $M map qc
run b16c_map_qd python3 -m $M map qd
ARM_3=$(python3 -c "from predictfun.research.pf_ladder_v3_build import ARM; print(ARM)")
unset PF_R60_FILE PF_LINES_FILE PF_TXJIT PF_TXSHIFT PF_QACT PF_ARMS PF_TAG PF_FORWARD PF_PREDS
export PF_RL_DDEC=2 PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_TIMING=e2 PF_TXTIME=ws PF_TXKEY=tk
for A in c d; do
  P=preds_q${A}60km
  for D in $DAYS; do
    PF_PREDS=$P PF_TXJIT=frame PF_ARMS="$ARM_3" PF_TAG=b16${A}J run b16${A}J_$D python3 -m predictfun.research.pf_step4_ladder_v3 $D tokyoreal custom
    PF_PREDS=$P PF_TXSHIFT=-50 PF_QACT=max PF_ARMS="$ARM_3" PF_TAG=b16${A}X run b16${A}X_$D python3 -m predictfun.research.pf_step4_ladder_v3 $D tokyoreal custom
  done
done
run b16c_report python3 -m $M report
echo B16C_DONE
