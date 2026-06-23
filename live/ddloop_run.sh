#!/usr/bin/env bash
# Run ONE convexity-v2 replay experiment for the dd-optimization loop, eval it, append to the ledger.
# usage: ddloop_run.sh <tag> [ENV=val ...]
# Base config = the live v2 stack; extra args override (e.g. CONC_CAP=0.4 BEAR_K=3).
# Idempotent: skips if <tag> already in the ledger. dvol cache pickle shared across runs (speed).
set -uo pipefail
ROOT=/home/yuqing/ctaNew; cd "$ROOT"; PY=python3
LEDGER=$ROOT/live/state/v3loop/ddloop/ledger.jsonl
DVOLPKL=$ROOT/live/state/v3loop/ddloop/_dvol_cache.pkl
mkdir -p "$(dirname "$LEDGER")"
TAG=$1; shift || true
# idempotent
if [ -f "$LEDGER" ] && grep -q "\"tag\": \"$TAG\"" "$LEDGER" 2>/dev/null; then echo "skip $TAG (in ledger)"; exit 0; fi
OUT=$ROOT/live/state/v3loop/dd_$TAG; rm -rf "$OUT"; mkdir -p "$OUT"
env COST_BPS_LEG=4.5 STRAT_K=3 SIDE_MODE=default XS_LEAN=1 CONVEXITY_PIT_DVOL=1 \
    CONVEXITY_UNIVERSE_META=outputs/vBTC_features/panel_expanded_v0.parquet \
    BEAR_MODE=equal STOP_SKIP_REGIMES=bear SIDE_BETA_NEUT=0 BEAR_K=2 SIZING_MODE=inv_vol \
    LONG_MAX_RET3D=0.20 \
    CONVEXITY_DVOL_CACHE_PKL=$DVOLPKL \
    "$@" \
    CONVEXITY_STATE=$OUT/state \
    CONVEXITY_PREDS_PATH=live/state/convexity_v1/base.parquet \
    CONVEXITY_PREDS_LONG=live/state/convexity_v1/long.parquet \
    PYTHONPATH=. $PY -m live.convexity_paper_bot --replay-all > "$OUT/run.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo "FAIL $TAG (rc=$rc) — see $OUT/run.log"; tail -3 "$OUT/run.log"; exit $rc; fi
LINE=$($PY live/ddloop_eval.py "$OUT" "$TAG" 2>>"$OUT/run.log")
if [ -z "$LINE" ]; then echo "EVAL-FAIL $TAG"; tail -3 "$OUT/run.log"; exit 1; fi
# atomic append
( flock 8; echo "$LINE" >> "$LEDGER"; ) 8>"$LEDGER.lock"
echo "OK $TAG"
