#!/usr/bin/env bash
# run_ladder2.sh - ladder / improve / per-minute-line / fresh-refresh replays (2026-11-05, exploratory, in-sample on the 8 read days + 11-03 beside),
# after the review fixes of 2026-11-05 (fill priority among our own orders, post-only against our own resting orders, GS strict guard).
# stage fix1  : regression - fixed live engine with PF_FILLPRIO=old PF_OWNCROSS=off reproduces the earlier kJ outputs (10-24, 11-01)
# stage fix2  : fixed live engine (both review rounds), C1-C4 S60, 9 days, central kJh and bound kXh (kJf / kXf = round 1 only) (impact of the fixes on the earlier numbers)
# stage par   : fixed ladder engine vs fixed live engine (10-24, 11-01) + never-trigger controls; own-cross falsifier (OWNCROSS=off)
# stage smoke : all real arms on 40 windows of 10-24, both prediction sets
# stage fullJ / fullX : all real arms, 9 days, both prediction sets, central lJ / bound lX, then the report
# 2026-11-05 15:3xZ: round suffix R (default i = after review-4 R7 / R8 fixes); round 3 = h (kJh ...) and unsuffixed lJ / lX / smk / view_*.
set -u
R=${PF_LAD2_ROUND:-i}
cd /home/yuqing/ctaNew
STAGE=${1:?stage}
D9="20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103"
L=data/predict_fun/research/logs
run() { local tag=$1; shift; predictfun/research/run_light.sh "$tag" -- "$@" > $L/$tag.log 2>&1 || { echo "FAILED $tag"; exit 1; }; }
export PF_RL_DDEC=2 PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_TIMING=e2 PF_TXTIME=ws PF_TXKEY=tk
B=G4+G2+G6+G5+G3+G1; T=XNA+XMIN1+XV+S60
A=""
for sz in A1:5:25:0 A1:20:100:0 AV:20:100:0.06 AV:5:25:0.06; do A="${A:+$A;}MICRO|$B+V0+$T|mid|$sz"; done
PAR="$A;MICRO|$B+PMzero+$T|mid|AV:5:25:0.06;MICRO|$B+V0+$T|mid|AM:5:25:c006;MICRO|$B+V0+LD2+LLinf+$T|mid|AV:5:25:0.06;MICRO|$B+V0+IMinf+$T|mid|AV:5:25:0.06"
REAL="MICRO|$B+V0+$T|mid|AV:5:25:0.06"
REAL="$REAL;MICRO|$B+V0+GS+$T|mid|AV:5:25:0.06"
REAL="$REAL;MICRO|$B+PM90+GS+$T|mid|AM:5:25:95"
REAL="$REAL;MICRO|$B+PM90+GS+LD2+$T|mid|AM:5:25:95"
REAL="$REAL;MICRO|$B+PM90+GS+IM95+$T|mid|AM:5:25:95"
REAL="$REAL;MICRO|$B+PM90+GS+LD2+IM95+$T|mid|AM:5:25:95"
REAL="$REAL;MICRO|$B+V0+GS+LD2+$T|mid|AV:5:25:0.06"
REAL="$REAL;MICRO|$B+PM90+GS+LD2+$T|mid|AM:20:100:95"
REAL="$REAL;MICRO|$B+PM90+GS+LD2+LH+$T|mid|AM:5:25:95"
LIVE=live_tokyoreal_custom_e2_preds_t60k
case $STAGE in
  fix1)
    for d in 20261024 20261101; do PF_PREDS=preds_t60k PF_TXJIT=frame PF_FILLPRIO=old PF_OWNCROSS=off PF_LEGORDER=old PF_REJRESET=off PF_POLICYVIEW=physical PF_ARMS="$A" PF_TAG=kJo$R run kJo${R}_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom; done
    PF_PAR_A=${LIVE}_kJ PF_PAR_B=${LIVE}_kJo$R PF_PAR_CTRL=0 PF_PAR_INV=0 run fix1_cmp$R python3 -m predictfun.research.pf_ladder_parity 20261024 20261101 ;;
  fix2)
    for d in $D9; do PF_PREDS=preds_t60k PF_TXJIT=frame PF_ARMS="$A" PF_TAG=kJ$R run kJ${R}_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom; done
    for d in $D9; do PF_PREDS=preds_t60k PF_TXSHIFT=-50 PF_QACT=max PF_ARMS="$A" PF_TAG=kX$R run kX${R}_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom; done ;;
  par)
    for d in 20261024 20261101; do PF_PREDS=preds_t60k PF_TXJIT=frame PF_ARMS="$PAR" PF_TAG=par$R run ladpar${R}_$d python3 -m predictfun.research.pf_step4_ladder $d tokyoreal custom; done
    PF_PAR_A=${LIVE}_kJ$R PF_PAR_B=ladder_tokyoreal_custom_e2_preds_t60k_par$R run ladpar${R}_cmp python3 -m predictfun.research.pf_ladder_parity 20261024 20261101
    PF_PREDS=preds_t60k PF_TXJIT=frame PF_OWNCROSS=off PF_ARMS="MICRO|$B+PM90+GS+LD2+IM95+$T|mid|AM:5:25:95" PF_TAG=ownoff$R PF_MAX_WINDOWS=100 \
      run ladownoff${R}_20261024 python3 -m predictfun.research.pf_step4_ladder 20261024 tokyoreal custom ;;
  view)
    V="MICRO|$B+V0+GS+LD2+$T|mid|AV:5:25:0.06;MICRO|$B+PM90+GS+LD2+IM95+$T|mid|AM:5:25:95;MICRO|$B+V0+$T|mid|AV:5:25:0.06"
    for pv in physical known; do PF_PREDS=preds_t60k PF_TXJIT=frame PF_POLICYVIEW=$pv PF_ARMS="$V" PF_TAG=view${R}_$pv PF_MAX_WINDOWS=100 \
      run ladview${R}_${pv}_20261024 python3 -m predictfun.research.pf_step4_ladder 20261024 tokyoreal custom; done
    run ladview${R}_cmp python3 -m predictfun.research.pf_view_check ladder_tokyoreal_custom_e2_preds_t60k_view${R}_physical ladder_tokyoreal_custom_e2_preds_t60k_view${R}_known 20261024 ;;
  smoke)
    for p in preds_t60k preds_t60kE; do PF_PREDS=$p PF_TXJIT=frame PF_ARMS="$REAL" PF_TAG=smk$R PF_MAX_WINDOWS=40 run ladsmk${R}_${p}_20261024 python3 -m predictfun.research.pf_step4_ladder 20261024 tokyoreal custom; done ;;
  fullJ)
    for p in preds_t60k preds_t60kE; do for d in $D9; do
      PF_PREDS=$p PF_TXJIT=frame PF_ARMS="$REAL" PF_TAG=lJ$R run ladJ${R}_${p}_$d python3 -m predictfun.research.pf_step4_ladder $d tokyoreal custom; done; done
    PF_LAD2_ROUND=$R run lad2_report_J$R python3 -m predictfun.research.pf_ladder2_report ;;
  fullX)
    for p in preds_t60k preds_t60kE; do for d in $D9; do
      PF_PREDS=$p PF_TXSHIFT=-50 PF_QACT=max PF_ARMS="$REAL" PF_TAG=lX$R run ladX${R}_${p}_$d python3 -m predictfun.research.pf_step4_ladder $d tokyoreal custom; done; done
    PF_LAD2_ROUND=$R run lad2_report_JX$R python3 -m predictfun.research.pf_ladder2_report ;;
esac
echo "STAGE_DONE $STAGE"
