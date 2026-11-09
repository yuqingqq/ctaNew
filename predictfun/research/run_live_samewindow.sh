#!/usr/bin/env bash
# Same-window research replay of the Tokyo live runs' 2026-11-08 windows (pf_live_samewindow.py docstring). Sequential run_light jobs:
# init (declares scope, links the bounded raw hours), then the frozen pipeline steps for 20261108 and the frozen v3 engine at J and X.
# Resume: FROM=<spec> skips init and every step before <spec>. A step refused only because another session holds the research slot
# is retried every 30 s (up to 2 h); any other failure stops the run.
set -euo pipefail
OUT=${1:?fresh output directory}; CAP=${2:?capture root (data/tokyo-live-20261109 checkout)}; ROOT=/home/yuqing/ctaNew
S=$ROOT/predictfun/research/pf_live_samewindow.py; FROM=${FROM:-}
cd "$ROOT"
[ -z "$FROM" ] && predictfun/research/run_light.sh lw_init -- python3 "$S" init --research "$ROOT" --capture "$CAP" --out "$OUT"
mkdir -p "$OUT/logs"
go=$([ -z "$FROM" ] && echo 1 || echo 0)
for SPEC in pf_step1 pf_step1_timing pf_step1_booktime pf_step5_features_rl pf_step5_c3_preds pf_step5_features_rl:pk pf_txmap pf_v1_apply pf_preds_startknown pf_step4_ladder_v3:J pf_step4_ladder_v3:X; do
  [ "$SPEC" = "$FROM" ] && go=1; [ $go = 1 ] || continue
  MODULE=${SPEC%%:*}; MODE=; [[ "$SPEC" == *:* ]] && MODE=${SPEC#*:}
  TAG=lw_${MODULE}_${MODE}
  for try in $(seq 1 240); do
    if predictfun/research/run_light.sh "$TAG" -- python3 "$S" step --out "$OUT" --day 20261108 --module "$MODULE" --mode "$MODE" > "$OUT/logs/$TAG.log" 2>&1; then
      tail -1 "$OUT/logs/$TAG.log"; break; fi
    if grep -q "REFUSED: a predictfun research process is already running" "$OUT/logs/$TAG.log" && [ $try -lt 240 ]; then sleep 30; continue; fi
    echo "FAILED $TAG"; tail -25 "$OUT/logs/$TAG.log"; exit 1
  done
done
echo ALL DONE
