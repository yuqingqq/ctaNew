#!/usr/bin/env bash
# Re-split (user 2026-11-03: validate on the first 2 days): O3's 20 arms on the fresh days 11-01/11-02 (preds_c3), same settings as run_o3.sh.
set -u
cd /home/yuqing/ctaNew
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs
run() { tag=$1; shift; $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }; }
export PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_RL_DDEC=2 PF_PREDS=preds_c3 PF_TIMING=e2
ST=G4+G2+G6+G5+G3+G1; X=XNA+XMIN1+XS0+XV; A1=A1:5:25:0; B=$ST+M025+$X
AA="MICRO|$B|mid|$A1"
for v in B1T03 B2T03 B2W100 B2W100T03 Q025 Q05 Q10; do AA="$AA;MICRO|$B+$v|mid|$A1"; done
AA="$AA;MICRO|$ST+M015+$X|mid|$A1;MICRO|$ST+M05+$X|mid|$A1"
BB=""
for v in BKA BK60 BKM S60 S120 S180 HY50 HY100 HY250; do BB="$BB;MICRO|$B+$v|mid|$A1"; done
BB="${BB#;};MICRO|$B|mid|A7:5:25:60"
for d in 20261101 20261102; do
  PF_ARMS="$AA" PF_TAG=o3a run rso3a_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom
  PF_ARMS="$BB" PF_TAG=o3b run rso3b_$d python3 -m predictfun.research.pf_step4_live $d tokyoreal custom
done
for x in o3a o3b; do run chk_rs_$x python3 -m predictfun.research.pf_check_fills live_tokyoreal_custom_e2_preds_c3_$x 20261101 20261102; done
echo ALL_DONE
