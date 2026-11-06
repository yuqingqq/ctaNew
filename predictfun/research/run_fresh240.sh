#!/usr/bin/env bash
# run_fresh240.sh - per-market feed refresh of the guard over the WHOLE window (declared in STATUS 2026-11-06; user: "yes we can also
# test that"). Exploratory, IN-SAMPLE (guard fit on the 8 read days; 11-03 beside). Fixed live engine round i, tokyoreal clock as
# run_invwin.sh. Every arm: MICRO + D1 stack + V0 + GS + XNA + XMIN1 + XV + S60 / S120 / S180 / S240, sizes AV:5:25:0.06 and A1:20:100:0.
# stage inputs : every-frame features over [ws, ws + 240 s) -> same guard models (PSTEM tf, k 60) -> start-known -> merged preds_t60kF
#                (refuses unless its first 65 s equal preds_t60kE row for row)
# stage ctrl   : 10-24 / 11-01, J: S60 on preds_t60kF must equal S60 on preds_t60kE (both sizes); S240 on preds_t60kF must differ
#                from S240 on the grid (falsifier)
# stage fullJ / fullX : 9 days on preds_t60kF, central (PF_TXJIT=frame) and bound (PF_TXSHIFT=-50 PF_QACT=max); then the report
set -u
cd /home/yuqing/ctaNew
STAGE=${1:?stage}
D9="20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103"
L=data/predict_fun/research/logs; S5=data/predict_fun/research/step5; S4=data/predict_fun/research/step4
run() { local tag=$1; shift; predictfun/research/run_light.sh "$tag" -- "$@" > $L/$tag.log 2>&1 || { echo "FAILED $tag"; exit 1; }; }
export PF_RL_DDEC=2
P="MICRO|G4+G2+G6+G5+G3+G1+V0+GS+XNA+XMIN1+XV"
ARMS=""; for z in AV:5:25:0.06 A1:20:100:0; do for e in S60 S120 S180 S240; do ARMS="${ARMS:+$ARMS;}$P+$e|mid|$z"; done; done
ENG="PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_TIMING=e2 PF_TXTIME=ws PF_TXKEY=tk"
day_run() {   # $1 J|X, $2 day, $3 preds, $4 tag, $5 arms
  local k=$1 d=$2 pr=$3 tag=$4 arms=$5
  [ -f $S4/live_tokyoreal_custom_e2_${pr}_${tag}/$d/summary.json ] && { echo "skip ${tag}_$d"; return; }
  if [ "$k" = J ]; then
    run ${tag}_$d env $ENG PF_TXJIT=frame PF_PREDS=$pr PF_ARMS="$arms" PF_TAG=$tag python3 -m predictfun.research.pf_step4_live $d tokyoreal custom
  else
    run ${tag}_$d env $ENG PF_TXSHIFT=-50 PF_QACT=max PF_PREDS=$pr PF_ARMS="$arms" PF_TAG=$tag python3 -m predictfun.research.pf_step4_live $d tokyoreal custom
  fi
}
case $STAGE in
  inputs)
    for d in $D9; do
      [ -f $S5/$d/rows_rl_d2_pk_ev240.parquet ] && continue
      run featev240_$d env PF_PEND_KNOWN=frame PF_GRID_EV=1 PF_EV_SPAN_S=240 PF_ROWS_SUFFIX=240 python3 -m predictfun.research.pf_step5_features_rl $d
    done
    run guard_preds_tf60 env PF_GRID=$S4/badfill_info_tx_tk/grid.parquet PF_ROWS=rows_rl_d2_pk_ev240 PF_PSTEM=tf PF_GUARD_KS=60 PF_NOSRC=1 \
      python3 -m predictfun.research.pf_step5_guard_preds 20261103
    run startknown_tf60 python3 -m predictfun.research.pf_preds_startknown preds_tf60 $D9
    run merge_ev240 env PF_EV_PSTEM=tf PF_EV_SPAN_S=240 PF_EV_OUT=preds_t60kF PF_EV_PARITY=preds_t60kE PF_EV_PARITY_S=65 \
      python3 -m predictfun.research.pf_preds_merge_ev $D9 ;;
  ctrl)
    for d in 20261024 20261101; do
      day_run J $d preds_t60kF fwJ "$ARMS"
      day_run J $d preds_t60kE fwEctrl "$P+S60|mid|AV:5:25:0.06;$P+S60|mid|A1:20:100:0"
    done
    run fw_ctrl_cmp python3 -m predictfun.research.pf_fresh240_report ctrl ;;
  fullJ) for d in $D9; do day_run J $d preds_t60kF fwJ "$ARMS"; done ;;
  fullX) for d in $D9; do day_run X $d preds_t60kF fwX "$ARMS"; done; run fw_report python3 -m predictfun.research.pf_fresh240_report ;;
esac
echo "STAGE_DONE $STAGE"
