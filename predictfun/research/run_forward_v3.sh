#!/usr/bin/env bash
# Declared forward check of pm_ladder_v3 (v1 + CAP10 + RO60 + S240) on 11-09..11-13, receipt predictfun/freeze/pm_ladder_v3/RECEIPT.md.
# Run by hand after 2026-11-14 02:00Z (step 1 reads hour files up to D+1 01Z); refuses earlier. No timer. Sequential run_light jobs; stops
# at the first failure. Inputs (A) are run_forward.sh's per-day commands without v2. Replays: v1 + control with pf_step4_ladder_fwd.py
# (the race's forward-enabled frozen engine), v3 with pf_step4_ladder_v3.py (parity: pf_v3_engine_check.py), timings J and X.
set -u
cd /home/yuqing/ctaNew
FWD=20261114; DAYS="20261109 20261110 20261111 20261112 20261113"
R=predictfun/research/run_light.sh; LG=data/predict_fun/research/logs/forward_v3; S4=data/predict_fun/research/step4; S5=data/predict_fun/research/step5
run() { local tag=$1; shift; $R "$tag" -- "$@" > "$LG/$tag.log" 2>&1 || { echo "FAILED $tag"; tail -3 "$LG/$tag.log"; exit 1; }; }
now=$(date -u +%Y%m%d%H); [ "$now" -ge ${FWD}02 ] || { echo "REFUSED: run after ${FWD} 02Z (now $now)"; exit 2; }
for t in f3aJ f3aX f3bJ f3bX; do [ -e $S4/ladder_tokyoreal_custom_e2_preds_t60k_$t ] && { echo "REFUSED: run tag $t exists"; exit 1; }; done
mkdir -p $LG
# source coverage (run_c2_day.sh's rule): every hour file from D-1 23Z to D+1 01Z
PF=data/predict_fun/runs/20261021T162823Z_0dd45f2d5a; REF=data/predict_reference/runs/20261022T132901Z_8c043d17b6; BN=data/mm_hf/raw/bookTicker/BTCUSDT
for D in $DAYS; do miss=0
  for h in $(python3 -c "import datetime as t,sys; d=t.datetime.strptime(sys.argv[1],'%Y%m%d'); print(' '.join((d+t.timedelta(hours=k)).strftime('%Y%m%d_%H') for k in range(-1,26)))" "$D"); do
    for f in $PF/predict_ws/$h.jsonl.gz $PF/rest/$h.jsonl.gz $REF/raw_ws/$h.jsonl.gz $REF/reference_prices/$h.jsonl.gz $BN/$h.csv.gz; do [ -s "$f" ] || miss=$((miss+1)); done; done
  [ "$miss" -eq 0 ] || { echo "SOURCE_GAP $D: $miss hour files missing"; exit 3; }
done
python3 -m predictfun.research.pf_ladder_v3_build > $LG/build.log 2>&1 || { echo "FAILED build"; exit 1; }
git diff --quiet HEAD -- predictfun/research/pf_step4_ladder_v3.py predictfun/freeze/pm_ladder_v3/params.json || { echo "REFUSED: regenerated v3 engine / params differ from the commit"; exit 1; }
sha256sum predictfun/research/pf_step4_ladder.py predictfun/research/pf_step4_ladder_fwd.py predictfun/research/pf_step4_ladder_v3.py \
  predictfun/research/pf_ladder_v3_build.py predictfun/research/pf_ladder_inv_build.py predictfun/research/pf_ladder_fwd_build.py \
  predictfun/research/pf_forward_v3_report.py predictfun/research/pf_v1_apply.py predictfun/research/pf_txmap.py \
  predictfun/freeze/pm_ladder_v1/guard_bid.txt predictfun/freeze/pm_ladder_v1/guard_ask.txt predictfun/freeze/pm_ladder_v3/params.json > $LG/code_sha256.txt
git rev-parse HEAD >> $LG/code_sha256.txt
unset PF_R60_FILE PF_LINES_FILE PF_TXJIT PF_TXSHIFT PF_QACT PF_ARMS PF_TAG
export PF_RL_DDEC=2 PF_FORWARD=$FWD
# A. inputs per day (run_forward.sh's commands)
for D in $DAYS; do
  run f3_s1_$D  python3 -m predictfun.research.pf_step1 $D
  run f3_t_$D   python3 -m predictfun.research.pf_step1_timing $D
  run f3_bt_$D  python3 -m predictfun.research.pf_step1_booktime $D
  run f3_f_$D   python3 -m predictfun.research.pf_step5_features_rl $D
  run f3_c3_$D  python3 -m predictfun.research.pf_step5_c3_preds apply $D
  PF_PEND_KNOWN=frame run f3_fpk_$D python3 -m predictfun.research.pf_step5_features_rl $D
  run f3_tx_$D  python3 -m predictfun.research.pf_txmap $D
  run f3_v1_$D  python3 -m predictfun.research.pf_v1_apply $D
done
# the start-known step rewrites one shared report file: keep the one before and this run's
[ -f $S5/startknown_preds_t60_report.json ] && cp -p $S5/startknown_preds_t60_report.json $S5/startknown_preds_t60_report_before_fwd_v3.json
run f3_sk python3 -m predictfun.research.pf_preds_startknown preds_t60 $DAYS
cp -p $S5/startknown_preds_t60_report.json $S5/startknown_preds_t60_report_fwd_v3.json
# B. replays, timings J and X
ARM_3=$(python3 -c "from predictfun.research.pf_ladder_v3_build import ARM; print(ARM)")
ARM_1='MICRO|G4+G2+G6+G5+G3+G1+PM90+GS+LD2+XNA+XMIN1+XV+S60|mid|AM:5:25:95'
ARM_K='MICRO|G4+G2+G6+G5+G3+G1+V0+GS+XNA+XMIN1+XV+S60|mid|AV:5:25:0.06'
export PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_TIMING=e2 PF_TXTIME=ws PF_TXKEY=tk PF_PREDS=preds_t60k
for TM in J X; do
  for D in $DAYS; do
    if [ $TM = J ]; then
      PF_TXJIT=frame PF_ARMS="$ARM_K;$ARM_1" PF_TAG=f3aJ run f3a_J_$D python3 -m predictfun.research.pf_step4_ladder_fwd $D tokyoreal custom
      PF_TXJIT=frame PF_ARMS="$ARM_3" PF_TAG=f3bJ run f3b_J_$D python3 -m predictfun.research.pf_step4_ladder_v3 $D tokyoreal custom
    else
      PF_TXSHIFT=-50 PF_QACT=max PF_ARMS="$ARM_K;$ARM_1" PF_TAG=f3aX run f3a_X_$D python3 -m predictfun.research.pf_step4_ladder_fwd $D tokyoreal custom
      PF_TXSHIFT=-50 PF_QACT=max PF_ARMS="$ARM_3" PF_TAG=f3bX run f3b_X_$D python3 -m predictfun.research.pf_step4_ladder_v3 $D tokyoreal custom
    fi
  done
done
# C. the declared rules
run f3_report python3 -m predictfun.research.pf_forward_v3_report
echo FORWARD_V3_DONE
