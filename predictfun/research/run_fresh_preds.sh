#!/usr/bin/env bash
# Fresh-refresh guard predictions (2026-11-05): every-frame rows (first 65 s) -> same guard models (out-of-fold on the 8 read days,
# one fit for 11-03) -> start-known withhold -> merged with the 50 ms grid predictions as preds_t60kE. One light process at a time.
set -u
cd /home/yuqing/ctaNew
D9="20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103"
L=data/predict_fun/research/logs
run() { local tag=$1; shift; predictfun/research/run_light.sh "$tag" -- "$@" > $L/$tag.log 2>&1 || { echo "FAILED $tag"; exit 1; }; }
export PF_RL_DDEC=2
for d in $D9; do
  [ -f data/predict_fun/research/step5/$d/rows_rl_d2_pk_ev.parquet ] && continue
  PF_PEND_KNOWN=frame PF_GRID_EV=1 run featev_$d python3 -m predictfun.research.pf_step5_features_rl $d
done
PF_GRID=data/predict_fun/research/step4/badfill_info_tx_tk/grid.parquet PF_ROWS=rows_rl_d2_pk_ev PF_PSTEM=te PF_GUARD_KS=60 PF_NOSRC=1 \
  run guard_preds_te60 python3 -m predictfun.research.pf_step5_guard_preds 20261103
run startknown_te60 python3 -m predictfun.research.pf_preds_startknown preds_te60 $D9
run merge_ev python3 -m predictfun.research.pf_preds_merge_ev $D9
echo FRESH_PREDS_DONE
