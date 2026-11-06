#!/usr/bin/env bash
# Declared forward check of pm_ladder_v1 (fixed) and pm_ladder_v2 (7-day daily refit) on 11-04..11-08, receipt
# predictfun/freeze/forward_20261104_08/RECEIPT.md. Run after 2026-11-09 02:00Z (step 1 reads hour files up to D+1 01Z).
# Sequential run_light jobs; stops at the first failure. Every step is a command that built 11-03 (logs: c2s1/c2t/c2bt, feat_1103,
# c3_1103, featpk, txmap, startknown_t60) except the four checked additions: pf_v1_apply (frozen models, parity on 11-03),
# pf_v2_grid_day (parity on 10-31 / 11-02), pf_v2_refit (fold + line parity), and the generated engines pf_step4_ladder_fwd /
# pf_step4_ladder_v2 (byte-identical to round i at the default FORWARD; FORWARD only bounds admissible input days).
set -u
cd /home/yuqing/ctaNew
FWD=20261109; DAYS="20261104 20261105 20261106 20261107 20261108"
R=predictfun/research/run_light.sh; LG=data/predict_fun/research/logs/forward; S5=data/predict_fun/research/step5; mkdir -p $LG
run() { local tag=$1; shift; $R "$tag" -- "$@" > "$LG/$tag.log" 2>&1 || { echo "FAILED $tag"; tail -3 "$LG/$tag.log"; exit 1; }; }
now=$(date -u +%Y%m%d%H); [ "$now" -ge ${FWD}02 ] || { echo "REFUSED: run after ${FWD} 02Z (now $now)"; exit 2; }
# source coverage (run_c2_day.sh's rule): every hour file from D-1 23Z to D+1 01Z
PF=data/predict_fun/runs/20261021T162823Z_0dd45f2d5a; REF=data/predict_reference/runs/20261022T132901Z_8c043d17b6; BN=data/mm_hf/raw/bookTicker/BTCUSDT
for D in $DAYS; do miss=0
  for h in $(python3 -c "import datetime as t,sys; d=t.datetime.strptime(sys.argv[1],'%Y%m%d'); print(' '.join((d+t.timedelta(hours=k)).strftime('%Y%m%d_%H') for k in range(-1,26)))" "$D"); do
    for f in $PF/predict_ws/$h.jsonl.gz $PF/rest/$h.jsonl.gz $REF/raw_ws/$h.jsonl.gz $REF/reference_prices/$h.jsonl.gz $BN/$h.csv.gz; do [ -s "$f" ] || miss=$((miss+1)); done; done
  [ "$miss" -eq 0 ] || { echo "SOURCE_GAP $D: $miss hour files missing"; exit 3; }
done
sha256sum predictfun/research/pf_step4_ladder.py predictfun/research/pf_step4_ladder_fwd.py predictfun/research/pf_step4_ladder_v2.py \
  predictfun/research/pf_v1_apply.py predictfun/research/pf_v2_grid_day.py predictfun/research/pf_v2_refit.py predictfun/research/pf_txmap.py \
  predictfun/freeze/pm_ladder_v1/guard_bid.txt predictfun/freeze/pm_ladder_v1/guard_ask.txt > $LG/code_sha256.txt
git rev-parse HEAD >> $LG/code_sha256.txt
[ -f $S5/startknown_preds_t60_report.json ] && [ ! -f $S5/startknown_preds_t60_report_round_i.json ] && cp -p $S5/startknown_preds_t60_report.json $S5/startknown_preds_t60_report_round_i.json
export PF_RL_DDEC=2 PF_FORWARD=$FWD
# A. inputs per day
for D in $DAYS; do
  run fw_s1_$D  python3 -m predictfun.research.pf_step1 $D
  run fw_t_$D   python3 -m predictfun.research.pf_step1_timing $D
  run fw_bt_$D  python3 -m predictfun.research.pf_step1_booktime $D
  run fw_f_$D   python3 -m predictfun.research.pf_step5_features_rl $D
  run fw_c3_$D  python3 -m predictfun.research.pf_step5_c3_preds apply $D
  PF_PEND_KNOWN=frame run fw_fpk_$D python3 -m predictfun.research.pf_step5_features_rl $D
  run fw_tx_$D  python3 -m predictfun.research.pf_txmap $D
  run fw_v1_$D  python3 -m predictfun.research.pf_v1_apply $D
done
run fw_sk python3 -m predictfun.research.pf_preds_startknown preds_t60 $DAYS
# B. v2 training rows (11-03 .. 11-07) and the daily refits
run fw_grid python3 -m predictfun.research.pf_v2_grid_day 20261103 20261104 20261105 20261106 20261107
run fw_refit python3 -m predictfun.research.pf_v2_refit $DAYS
# C. replays: v1 candidate + control (forward-enabled frozen engine), v2 candidate (v2 engine), timings J and X
ARM_C='MICRO|G4+G2+G6+G5+G3+G1+PM90+GS+LD2+XNA+XMIN1+XV+S60|mid|AM:5:25:95'
ARM_K='MICRO|G4+G2+G6+G5+G3+G1+V0+GS+XNA+XMIN1+XV+S60|mid|AV:5:25:0.06'
export PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_TIMING=e2 PF_TXTIME=ws PF_TXKEY=tk
for TM in J X; do
  for D in $DAYS; do
    if [ $TM = J ]; then
      PF_TXJIT=frame PF_PREDS=preds_t60k PF_ARMS="$ARM_K;$ARM_C" PF_TAG=fw1J run fw1_J_$D python3 -m predictfun.research.pf_step4_ladder_fwd $D tokyoreal custom
      PF_TXJIT=frame PF_PREDS=preds_v2w7k PF_ARMS="$ARM_C" PF_TAG=fw2J PF_LINES_FILE="/home/yuqing/ctaNew/$S5/{day}/lines_v2w7.json" \
        run fw2_J_$D python3 -m predictfun.research.pf_step4_ladder_v2 $D tokyoreal custom
    else
      PF_TXSHIFT=-50 PF_QACT=max PF_PREDS=preds_t60k PF_ARMS="$ARM_K;$ARM_C" PF_TAG=fw1X run fw1_X_$D python3 -m predictfun.research.pf_step4_ladder_fwd $D tokyoreal custom
      PF_TXSHIFT=-50 PF_QACT=max PF_PREDS=preds_v2w7k PF_ARMS="$ARM_C" PF_TAG=fw2X PF_LINES_FILE="/home/yuqing/ctaNew/$S5/{day}/lines_v2w7.json" \
        run fw2_X_$D python3 -m predictfun.research.pf_step4_ladder_v2 $D tokyoreal custom
    fi
  done
done
# D. the declared rules
run fw_report python3 -m predictfun.research.pf_forward_report
echo FORWARD_DONE
