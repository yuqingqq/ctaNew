#!/bin/bash
# Full pipeline for WQ101 + Qlib Alpha158: compute -> betaneut -> screen (raw + betaneut).
# Sequential separate processes = memory released between steps (OOM guard).
set -x
cd /home/yuqing/ctaNew
CACHE=data/ml/cache; OUT=live/state/longtail
mkdir -p $OUT

echo "=== [1/8] compute WQ101 ==="
python3 live/alpha101_lib.py 2>&1 | tail -3
echo "=== [2/8] compute Alpha158 ==="
python3 live/alpha158_lib.py 2>&1 | tail -3

echo "=== [3/8] betaneut WQ101 ==="
FACTORS_PATH=$CACHE/alpha101_factors.parquet OUT_PATH=$CACHE/alpha101_factors_betaneut.parquet PREFIX=wq \
  python3 live/alphaset_betaneut.py 2>&1 | tail -3
echo "=== [4/8] betaneut Alpha158 ==="
FACTORS_PATH=$CACHE/alpha158_factors.parquet OUT_PATH=$CACHE/alpha158_factors_betaneut.parquet PREFIX=q158_ \
  python3 live/alphaset_betaneut.py 2>&1 | tail -3

echo "=== [5/8] screen WQ101 raw ==="
FACTORS_PATH=$CACHE/alpha101_factors.parquet PREFIX=wq OUT_CSV=$OUT/alpha101_screen_raw.csv \
  python3 live/alphaset_screen.py
echo "=== [6/8] screen WQ101 betaneut ==="
FACTORS_PATH=$CACHE/alpha101_factors_betaneut.parquet PREFIX=wq OUT_CSV=$OUT/alpha101_screen_betaneut.csv \
  python3 live/alphaset_screen.py
echo "=== [7/8] screen Alpha158 raw ==="
FACTORS_PATH=$CACHE/alpha158_factors.parquet PREFIX=q158_ OUT_CSV=$OUT/alpha158_screen_raw.csv \
  python3 live/alphaset_screen.py
echo "=== [8/8] screen Alpha158 betaneut ==="
FACTORS_PATH=$CACHE/alpha158_factors_betaneut.parquet PREFIX=q158_ OUT_CSV=$OUT/alpha158_screen_betaneut.csv \
  python3 live/alphaset_screen.py
echo "ALLDONE"
