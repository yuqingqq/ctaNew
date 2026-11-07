#!/usr/bin/env bash
# Parity of the generated pm_ladder_v3 engine (pf_v3_engine_check.py docstring): 3 run_light replays on read days, then the check, then the
# forward report's rehearsal on read days (pf_forward_v3_report.py rehearse). Refuses if a check tag exists.
set -u
cd /home/yuqing/ctaNew
S4=data/predict_fun/research/step4; LG=data/predict_fun/research/logs; R=predictfun/research
for t in v3pJ v3pX v3pF; do [ -e $S4/ladder_tokyoreal_custom_e2_preds_t60k_$t ] && { echo "REFUSED: run tag $t exists"; exit 1; }; done
ARM=$(python3 -c "from predictfun.research.pf_ladder_v3_build import ARM; print(ARM)") || exit 1
unset PF_FORWARD PF_R60_FILE PF_TXJIT PF_TXSHIFT PF_QACT PF_ARMS PF_TAG PF_LINES_FILE
export PF_RL_DDEC=2 PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_TIMING=e2 PF_TXTIME=ws PF_TXKEY=tk PF_PREDS=preds_t60k
run() { local tag=$1; shift; $R/run_light.sh "$tag" -- "$@" > "$LG/$tag.log" 2>&1 || { echo "FAILED $tag"; tail -3 "$LG/$tag.log"; exit 1; }; }
PF_TXJIT=frame PF_ARMS="$ARM" PF_TAG=v3pJ run v3p_J_1103 python3 -m predictfun.research.pf_step4_ladder_v3 20261103 tokyoreal custom
PF_TXSHIFT=-50 PF_QACT=max PF_ARMS="$ARM" PF_TAG=v3pX run v3p_X_1103 python3 -m predictfun.research.pf_step4_ladder_v3 20261103 tokyoreal custom
PF_TXJIT=frame PF_FORWARD=20261114 PF_ARMS="$ARM" PF_TAG=v3pF run v3p_F_1102 python3 -m predictfun.research.pf_step4_ladder_v3 20261102 tokyoreal custom
run v3_engine_check python3 -m predictfun.research.pf_v3_engine_check
run v3_report_rehearsal python3 -m predictfun.research.pf_forward_v3_report rehearse
echo ALL DONE
