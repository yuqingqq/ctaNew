#!/usr/bin/env bash
# FV3 driver (2026-11-12, session ce68ebe0): the frozen pm_ladder_v3 engine (pf_step4_ladder_v3.py, never edited) on the 9 consumed
# replay days with the v3 forward check's settings minus PF_FORWARD (the engine refuses days >= 11-04 by default), then the report.
# Every engine run through run_light.sh in the claude slot, one at a time; stops at the first failure. Tags must not exist.
set -u
cd /home/yuqing/ctaNew
OUT=data/predict_fun/research/fv3_20261112; mkdir -p $OUT/logs; S4=data/predict_fun/research/step4
DAYS="20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103"; CTRL="20261023 20261103"
for t in fv3c40J fv3c40X fv3p20J fv3p20X fv3p10J fv3p10X fv3x250 fv3x250p10; do
  [ -e $S4/ladder_tokyoreal_custom_e2_preds_t60k_$t ] && { echo "REFUSED: tag $t exists" | tee -a $OUT/logs/driver.log; exit 1; }; done
ARM=$(python3 -c "from predictfun.research.pf_ladder_v3_build import ARM; print(ARM)")
unset PF_R60_FILE PF_LINES_FILE PF_TXJIT PF_TXSHIFT PF_QACT PF_ARMS PF_TAG PF_FORWARD
export PF_RL_DDEC=2 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_TIMING=e2 PF_TXTIME=ws PF_TXKEY=tk PF_PREDS=preds_t60k PF_ARMS="$ARM"
R="env PF_SLOT=claude predictfun/research/run_light.sh"; E="python3 -m predictfun.research.pf_step4_ladder_v3"
job() { local tag=$1 day=$2; shift 2; echo "$(date -u +%H:%M:%SZ) start $tag $day" >> $OUT/logs/driver.log
  env "$@" PF_TAG=$tag $R fv3_${tag}_$day -- $E $day tokyoreal custom > $OUT/logs/${tag}_$day.log 2>&1 || { echo "$(date -u +%H:%M:%SZ) FAILED $tag $day" >> $OUT/logs/driver.log; exit 1; }
  sleep 3; }
for d in $CTRL; do job fv3c40J $d PF_RL_LPLACE=40 PF_TXJIT=frame; job fv3c40X $d PF_RL_LPLACE=40 PF_TXSHIFT=-50 PF_QACT=max; done
for d in $DAYS; do
  job fv3p20J $d PF_RL_LPLACE=20 PF_TXJIT=frame; job fv3p20X $d PF_RL_LPLACE=20 PF_TXSHIFT=-50 PF_QACT=max
  job fv3p10J $d PF_RL_LPLACE=10 PF_TXJIT=frame; job fv3p10X $d PF_RL_LPLACE=10 PF_TXSHIFT=-50 PF_QACT=max
  job fv3x250 $d PF_RL_LPLACE=40 PF_TXSHIFT=-250 PF_QACT=max; job fv3x250p10 $d PF_RL_LPLACE=10 PF_TXSHIFT=-250 PF_QACT=max
done
env PF_SLOT=claude predictfun/research/run_light.sh fv3_report -- python3 predictfun/research/pf_fv3_check.py report > $OUT/logs/report.log 2>&1 || { echo "$(date -u +%H:%M:%SZ) FAILED report" >> $OUT/logs/driver.log; exit 1; }
echo "$(date -u +%H:%M:%SZ) ALL DONE" >> $OUT/logs/driver.log
