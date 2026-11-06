#!/usr/bin/env bash
# Round O4 (declared in STATUS.yml 2026-11-03): sweep protection on all 8 days at the Tokyo clock, decision 2 ms.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
run() { tag=$1; shift; $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }; }
export PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_RL_DDEC=2 PF_TIMING=e2
ST=G4+G2+G6+G5+G3+G1; X=XNA+XMIN1+XS0+XV+B2W100; A1=A1:5:25:0
P3=$ST+M025+$X+Q025+BKA+S60
AA="MICRO|$P3|mid|$A1"; for v in BKD2 CD250 CD500 CD1000 FC500 FC2000; do AA="$AA;MICRO|$P3+$v|mid|$A1"; done
AA="$AA;MICRO|$P3|mid|5"
BB="MICRO|$ST+M025+$X+Q025+BKA+S30|mid|$A1;MICRO|$ST+M025+$X+Q025+BKA+S45|mid|$A1;MICRO|$ST+M025+$X+Q025+BKA+S90|mid|$A1"
BB="$BB;MICRO|$P3|mid|A1:5:10:0;MICRO|$ST+M015+$X+Q025+BKA+S60|mid|$A1;MICRO|$ST+M025+$X+Q015+BKA+S60|mid|$A1;MICRO|$ST+M025+$X+Q05+BKA+S60|mid|$A1"
for d in 20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102; do
  pre=preds_o3; case $d in 202611*) pre=preds_c3;; esac
  PF_PREDS=$pre PF_ARMS="$AA" PF_TAG=o4a run o4a_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom
  PF_PREDS=$pre PF_ARMS="$BB" PF_TAG=o4b run o4b_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom
done
for pre in preds_o3 preds_c3; do for x in o4a o4b; do
  days="20261023 20261024 20261025 20261026 20261027 20261031"; [ $pre = preds_c3 ] && days="20261101 20261102"
  run chk_${x}_$pre python3 -m predictfun.research.pf_check_fills live_tokyoreal_custom_e2_${pre}_$x $days; done; done
echo ALL_DONE
