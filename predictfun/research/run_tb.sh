#!/usr/bin/env bash
# TB driver (2026-11-12, session ce68ebe0): the frozen pm_ladder_v3 engine (pf_step4_ladder_v3.py, never edited) with the quoting limits
# off (B0) and B0 + IM95, timings J / X / X250, on the 9 consumed replay days; the in-sample arm noCAP_S60 (CTRL) rides along on the two
# control days. Settings as FV3 (run_fv3.sh). Every engine run through run_light.sh in the claude slot, one at a time; waits for a free
# slot (up to 4 h per job); stops at the first other failure. Tags must not exist. Then the report (pf_tb_check.py).
set -u
cd /home/yuqing/ctaNew
OUT=data/predict_fun/research/tb_20261112; mkdir -p $OUT/logs; S4=data/predict_fun/research/step4
DAYS="20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103"
for t in tbJ tbX tb250; do
  [ -e $S4/ladder_tokyoreal_custom_e2_preds_t60k_$t ] && { echo "REFUSED: tag $t exists" | tee -a $OUT/logs/driver.log; exit 1; }; done
B0="MICRO|G4+G2+G5+G3+G1+PM90+GS+LD2+XNA+XMIN1+XV+S60|mid|AM:5:25:95"
IM="MICRO|G4+G2+G5+G3+G1+PM90+GS+LD2+IM95+XNA+XMIN1+XV+S60|mid|AM:5:25:95"
CT="MICRO|G4+G2+G6+G5+G3+G1+PM90+GS+LD2+XNA+XMIN1+XV+S60|mid|AM:5:25:95"
unset PF_R60_FILE PF_LINES_FILE PF_TXJIT PF_TXSHIFT PF_QACT PF_ARMS PF_TAG PF_FORWARD
export PF_RL_DDEC=2 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_TIMING=e2 PF_TXTIME=ws PF_TXKEY=tk PF_PREDS=preds_t60k PF_RL_LPLACE=40
R="env PF_SLOT=claude predictfun/research/run_light.sh"; E="python3 -m predictfun.research.pf_step4_ladder_v3"
job() { local tag=$1 day=$2; shift 2; local n=0 arms="$B0;$IM"
  case $day in 20261023|20261103) [ $tag != tb250 ] && arms="$arms;$CT";; esac
  while :; do echo "$(date -u +%H:%M:%SZ) start $tag $day" >> $OUT/logs/driver.log
    if env "$@" PF_ARMS="$arms" PF_TAG=$tag $R tb_${tag}_$day -- $E $day tokyoreal custom > $OUT/logs/${tag}_$day.log 2>&1; then break; fi
    if grep -q -E "^REFUSED: (research slot|[0-9]+ predictfun research processes)" $OUT/logs/${tag}_$day.log && [ $n -lt 240 ]; then
      n=$((n+1)); echo "$(date -u +%H:%M:%SZ) slot busy ($(head -1 $OUT/logs/${tag}_$day.log)); retry in 60 s" >> $OUT/logs/driver.log; sleep 60; continue; fi
    echo "$(date -u +%H:%M:%SZ) FAILED $tag $day" >> $OUT/logs/driver.log; exit 1
  done; sleep 3; }
for d in $DAYS; do
  job tbJ $d PF_TXJIT=frame; job tbX $d PF_TXSHIFT=-50 PF_QACT=max; job tb250 $d PF_TXSHIFT=-250 PF_QACT=max
done
env PF_SLOT=claude predictfun/research/run_light.sh tb_report -- python3 predictfun/research/pf_tb_check.py report > $OUT/logs/report.log 2>&1 || { echo "$(date -u +%H:%M:%SZ) FAILED report" >> $OUT/logs/driver.log; exit 1; }
echo "$(date -u +%H:%M:%SZ) ALL DONE" >> $OUT/logs/driver.log
