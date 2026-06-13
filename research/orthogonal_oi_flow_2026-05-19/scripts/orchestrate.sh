#!/bin/bash
# Sequencer: avoid CPU thrash. Stage-0 OI/flow test waits for OI-fetch AND
# B*b (bottleneck control) to finish; full test waits for flow-fetch.
set -u
R=/home/yuqing/ctaNew/research
OL=$R/orthogonal_oi_flow_2026-05-19/results
BL=$R/bottleneck_2026-05-19/results
cd /home/yuqing/ctaNew

# 1) wait OI fetch done
until grep -q OI_FETCH_DONE "$OL/fetch_oi.log" 2>/dev/null; do sleep 30; done
echo "[orch] OI fetch done $(date -u +%H:%M)"
# 2) wait B*b done (free the CPU)
until grep -q 'B★b done' "$BL/B_star_b_run.log" 2>/dev/null; do sleep 30; done
echo "[orch] B*b done; launching Stage-0 OI/flow test $(date -u +%H:%M)"
# 3) Stage-0 (OI full + cached flow)
python3 $R/orthogonal_oi_flow_2026-05-19/scripts/oi_flow_test.py --stage0 \
  > "$OL/oi_flow_stage0.log" 2>&1
echo "[orch] Stage-0 done $(date -u +%H:%M)"
# 4) wait flow fetch done
until grep -q FLOW_FETCH_DONE "$OL/fetch_flow.log" 2>/dev/null; do sleep 60; done
echo "[orch] flow fetch done; launching FULL OI/flow test $(date -u +%H:%M)"
# 5) full run (all fetched flow + OI)
python3 $R/orthogonal_oi_flow_2026-05-19/scripts/oi_flow_test.py \
  > "$OL/oi_flow_full.log" 2>&1
echo "[orch] FULL done $(date -u +%H:%M)"
echo ORCH_ALL_DONE
