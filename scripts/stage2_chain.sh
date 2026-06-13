#!/bin/bash
cd /home/yuqing/ctaNew
echo "$(date) stage2_chain start"

# Wait for the V5 3yr pipeline (flow build + X78) to finish
while pgrep -f "v5_3yr_pipeline.sh" >/dev/null 2>&1; do sleep 120; done
# also ensure X78 python itself is done
while pgrep -f "X78_build_v5_3yr.py" >/dev/null 2>&1; do sleep 60; done
sleep 10
echo "$(date) X78 done — running X79"

python3 -u research/convexity_portable_2026-05-20/scripts/X79_beta_neutral.py > /tmp/x79.log 2>&1
echo "$(date) X79 done — running X80"

python3 -u research/convexity_portable_2026-05-20/scripts/X80_hmm_regime.py > /tmp/x80.log 2>&1
echo "$(date) X80 done — running X82"

python3 -u research/convexity_portable_2026-05-20/scripts/X82_matrix_compile_combine.py > /tmp/x82.log 2>&1
echo "$(date) X82 done — STAGE 2 COMPLETE"
