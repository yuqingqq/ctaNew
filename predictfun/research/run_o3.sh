#!/usr/bin/env bash
# Round O3 (declared in STATUS.yml 2026-11-03): optimisation at the measured Tokyo clock, decision 2 ms. Read days only.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
run() { tag=$1; shift; $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }; }
DAYS="20261023 20261024 20261025 20261026 20261027 20261031"
[ "${SKIP_PREDS:-0}" = 1 ] || run o3preds python3 -m predictfun.research.pf_step5_o3_preds
export PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_RL_DDEC=2 PF_PREDS=preds_o3 PF_TIMING=e2
ST=G4+G2+G6+G5+G3+G1; X=XNA+XMIN1+XS0+XV; A1=A1:5:25:0; B=$ST+M025+$X
AA="MICRO|$B|mid|$A1"
for v in B1T03 B2T03 B2W100 B2W100T03 Q025 Q05 Q10; do AA="$AA;MICRO|$B+$v|mid|$A1"; done
AA="$AA;MICRO|$ST+M015+$X|mid|$A1;MICRO|$ST+M05+$X|mid|$A1"
BB=""
for v in BKA BK60 BKM S60 S120 S180 HY50 HY100 HY250; do BB="$BB;MICRO|$B+$v|mid|$A1"; done
BB="${BB#;};MICRO|$B|mid|A7:5:25:60"
for d in $DAYS; do
  PF_ARMS="$AA" PF_TAG=o3a run o3a_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom
  PF_ARMS="$BB" PF_TAG=o3b run o3b_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom
done
for x in o3a o3b; do run chk_$x python3 -m predictfun.research.pf_check_fills live_tokyoreal_custom_e2_preds_o3_$x $DAYS; done
echo ALL_DONE
