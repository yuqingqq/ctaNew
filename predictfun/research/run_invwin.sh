#!/usr/bin/env bash
# run_invwin.sh - inventory reduction x quoting window on the FIXED live engine (round i, 2026-11-05; declared in STATUS 17:3xZ before any run).
# Exploratory, IN-SAMPLE (the V0 guard was fit on the same 8 read days; 11-03 beside). Every arm: MICRO + D1 stack + V0 + GS (strict guard)
# + XNA + XMIN1 + XV, tokyoreal clock as run_ladder2.sh. Per size, normal quoting for W s, then: stop (carry all) | CAP<hi> | reduce-only
# W -> W+60 (RO<W> + S<W+60>) | both; W = 60 / 120 / 180, and W = 240 (stop / CAP only: guard predictions end at 239.95 s).
# stage ctrl : never-trigger controls (CAP1000, RO300 on S60) must equal the base S60 arm on 10-24 / 11-01 (summary + per-window equal);
#              falsifier: S60+CAP5 must differ (the comparison can fire)
# stage fullJ / fullX : 9 days, central (PF_TXJIT=frame) and bound (PF_TXSHIFT=-50 PF_QACT=max) timing; then pf_invwin_report
set -u
cd /home/yuqing/ctaNew
STAGE=${1:?stage}
D9="20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103"
L=data/predict_fun/research/logs
run() { local tag=$1; shift; predictfun/research/run_light.sh "$tag" -- "$@" > $L/$tag.log 2>&1 || { echo "FAILED $tag"; exit 1; }; }
export PF_RL_DDEC=2 PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_TIMING=e2 PF_TXTIME=ws PF_TXKEY=tk PF_PREDS=preds_t60k
P="MICRO|G4+G2+G6+G5+G3+G1+V0+GS+XNA+XMIN1+XV"
arms() {   # $1 size, $2 hi, $3 batch a|b
  local z=$1 h=$2 x=""
  if [ "$3" = a ]; then
    for e in S60 S60+CAP$h RO60+S120 RO60+S120+CAP$h S120 S120+CAP$h RO120+S180; do x="${x:+$x;}$P+$e|mid|$z"; done
  else
    for e in RO120+S180+CAP$h S180 S180+CAP$h RO180+S240 RO180+S240+CAP$h S240 S240+CAP$h; do x="${x:+$x;}$P+$e|mid|$z"; done
  fi
  echo "$x"
}
full() {   # $1 J|X
  local k=$1
  if [ "$k" = J ]; then export PF_TXJIT=frame; unset PF_TXSHIFT PF_QACT; else unset PF_TXJIT; export PF_TXSHIFT=-50 PF_QACT=max; fi
  for d in $D9; do
    for zb in "AV:5:25:0.06 25 s5" "A1:20:100:0 100 s20"; do set -- $zb
      for b in a b; do PF_ARMS="$(arms $1 $2 $b)" PF_TAG=iw${k}_$3$b run iw${k}_$3${b}_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom; done
    done
  done
}
case $STAGE in
  ctrl)
    C="$P+S60|mid|AV:5:25:0.06;$P+S60+CAP1000|mid|AV:5:25:0.06;$P+RO300+S60|mid|AV:5:25:0.06;$P+S60+CAP5|mid|AV:5:25:0.06"
    for d in 20261024 20261101; do PF_TXJIT=frame PF_ARMS="$C" PF_TAG=iwctrl run iwctrl_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom; done
    run iwctrl_cmp python3 -m predictfun.research.pf_invwin_report ctrl ;;
  fullJ) full J ;;
  fullX) full X; run iw_report python3 -m predictfun.research.pf_invwin_report ;;
esac
echo "STAGE_DONE $STAGE"
