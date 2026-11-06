#!/usr/bin/env bash
# Round LC (declared in STATUS.yml 2026-11-03): leftover control on base + S60, all 8 days, Tokyo clock, decision 2 ms.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
run() { tag=$1; shift; $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }; }
export PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_RL_DDEC=2 PF_TIMING=e2
B=G4+G2+G6+G5+G3+G1+M025+XNA+XMIN1+XS0+XV+S60; A1=A1:5:25:0
A="MICRO|$B|mid|$A1"; for v in CAP10 CAP25 RO45 RO50 RO45+CAP25; do A="$A;MICRO|$B+$v|mid|$A1"; done
A="$A;MICRO|$B|mid|A1:5:10:0;MICRO|$B|mid|5"
for d in 20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102; do
  pre=preds_o3; case $d in 202611*) pre=preds_c3;; esac
  PF_PREDS=$pre PF_ARMS="$A" PF_TAG=lc run lc_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom
done
run chk_lc_o3 python3 -m predictfun.research.pf_check_fills live_tokyoreal_custom_e2_preds_o3_lc 20261023 20261024 20261025 20261026 20261027 20261031
run chk_lc_c3 python3 -m predictfun.research.pf_check_fills live_tokyoreal_custom_e2_preds_c3_lc 20261101 20261102
echo ALL_DONE
