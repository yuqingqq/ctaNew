#!/bin/bash
echo "$(date) clean chain start"

cd /home/yuqing/ctaNew

echo "$(date) X57 start"
timeout 600 python3 -u research/convexity_portable_2026-05-20/scripts/X57_cluster_dropout_v5mv3.py > /tmp/x57.log 2>&1
echo "$(date) X57 done"

echo "$(date) X58 start"
timeout 300 python3 -u research/convexity_portable_2026-05-20/scripts/X58_regime_classify.py > /tmp/x58.log 2>&1
echo "$(date) X58 done"

echo "$(date) X59 start"
timeout 1200 python3 -u research/convexity_portable_2026-05-20/scripts/X59_per_regime_eval.py > /tmp/x59.log 2>&1
echo "$(date) X59 done"

echo "$(date) X64 start"
timeout 1800 python3 -u research/convexity_portable_2026-05-20/scripts/X64_bull_regime_gate.py > /tmp/x64.log 2>&1
echo "$(date) X64 done"

echo "$(date) X65 start"
timeout 1800 python3 -u research/convexity_portable_2026-05-20/scripts/X65_bull_momentum_overlay.py > /tmp/x65.log 2>&1
echo "$(date) X65 done"

echo "$(date) X66 start"
timeout 1800 python3 -u research/convexity_portable_2026-05-20/scripts/X66_regime_conditional_ensemble.py > /tmp/x66.log 2>&1
echo "$(date) X66 done"

echo "$(date) ALL DONE"
