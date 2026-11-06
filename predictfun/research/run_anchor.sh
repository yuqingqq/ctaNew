#!/usr/bin/env bash
# Ideal-anchor test (declared in STATUS.yml 2026-11-04 01:10Z): the same arms with the oracle and with the real predictions.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
run() { tag=$1; shift; $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }; }
export PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_RL_DDEC=2 PF_TIMING=e2
B=G4+G2+G6+G5+G3+G1+M025+XNA+XMIN1; A1=A1:5:25:0
A="MICRO|$B+S60|mid|$A1;MICRO|$B+Q025+S60|mid|$A1;MICRO|$B|mid|$A1"
run oracle_preds python3 -m predictfun.research.pf_step5_oracle_preds
for d in 20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102; do
  pre=preds_o3; case $d in 202611*) pre=preds_c3;; esac
  PF_PREDS=preds_oracle PF_ARMS="$A" PF_TAG=anc run anc_or_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom
  PF_PREDS=$pre PF_ARMS="$A" PF_TAG=anc run anc_re_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom
done
echo ALL_DONE
