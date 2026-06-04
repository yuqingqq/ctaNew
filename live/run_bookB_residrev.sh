#!/usr/bin/env bash
# BOOK-B-ONLY (+ optional resid_rev) paper-test runner — the 2026-06-04 strategy.
# Strategy: drop alpha-barren high-vol book A; trade LOW-VOL book B only.
#   universe : rank eligible syms by trailing-30d rvol_7d, EXCLUDE top-N (default 80), trade the rest (~94)
#   construct: K=3 long/short, beta-neutral, 24h hold / 6 sleeves, regime gate, relative-rank (PIT)
#   resid_rev: OPTIONAL long-ranker overlay (dual-pred). Genuine fast residual-reversion; EXECUTION-LATENCY-GATED
#              (~10-15 min budget after each 4h close). Toggle with USE_RESIDREV=1.
#
# This forward-replays the validated stack as the panel extends (paper test). The LIVE --cycle path
# (real-time Binance feeds) is wired on the EXECUTION SERVER, not here — this box trains + produces preds.
#
# Usage:
#   bash live/run_bookB_residrev.sh                 # baseline book-B-only (latency-robust core, ~+2.55/+2.82)
#   USE_RESIDREV=1 bash live/run_bookB_residrev.sh  # + resid_rev overlay (~+3.4 IF executed promptly)
#   REGEN=1 USE_RESIDREV=1 bash live/run_bookB_residrev.sh  # regenerate preds first (needed after panel update)
#   EXCLUDE_TOPN=80 POLICY=monthly ...              # overrides
set -uo pipefail
ROOT=/home/yuqing/ctaNew; export PYTHONPATH=$ROOT; cd "$ROOT"
PY=python3
: "${EXCLUDE_TOPN:=80}"          # exclude top-N by trailing-30d rvol → book B = the rest
: "${POLICY:=monthly}"           # membership rerank cadence (monthly = production static-at-retrain)
: "${USE_RESIDREV:=1}"           # 1 = enable resid_rev long-ranker overlay (latency-gated) — ON by default (the wired strategy)
: "${REGEN:=0}"                  # 1 = regenerate base + resid_rev preds from current panel first
: "${COST_BPS_LEG:=4.5}"; export COST_BPS_LEG
OUT=$ROOT/live/state/convexity_bookB; mkdir -p "$OUT"; LOG=$OUT/run.log
log(){ echo "[$(date -u '+%F %T')] $*" | tee -a "$LOG"; }

log "== BOOK-B-ONLY runner | exclude top-$EXCLUDE_TOPN rvol | policy=$POLICY | resid_rev=$USE_RESIDREV | cost=${COST_BPS_LEG}bps =="

if [ "$REGEN" = "1" ]; then
  log "regenerating base preds (train_twobook_models)…"; $PY -m live.train_twobook_models >>"$LOG" 2>&1 || log "WARN train failed"
  if [ "$USE_RESIDREV" = "1" ]; then
    log "regenerating resid_rev long-ranker preds (gen_residrev_wf_preds)…"; $PY live/gen_residrev_wf_preds.py >>"$LOG" 2>&1 || log "WARN residrev gen failed"
  fi
fi

# Run book B through the validated split machinery (book A = top-N high-vol is built but IGNORED; we read book B only).
export AB_OUTBASE=live/state/convexity_bookB/run
export AB_SIDEMODE_A=default AB_SIDEMODE_B=default
if [ "$USE_RESIDREV" = "1" ]; then export AB_HLDIR_LONG=live/state/convexity/hl_residrev; else unset AB_HLDIR_LONG; fi
$PY live/ab_split_rerank.py --n "$EXCLUDE_TOPN" --policies "$POLICY" >>"$LOG" 2>&1 || { log "replay FAILED"; exit 1; }

# Report BOOK-B standalone metrics (book-B-only = what we actually trade; book A is dropped).
$PY - <<PY | tee -a "$LOG"
import json
s=json.load(open("$OUT/run/$POLICY/combine/twobook_summary.json"))
tag = "book-B-only + resid_rev" if "$USE_RESIDREV"=="1" else "book-B-only (baseline)"
print(f"\n=== {tag} | exclude top-$EXCLUDE_TOPN rvol | $POLICY ===")
print(f"  Sharpe  {s['sharpe_bookB']:+.3f}")
print(f"  (two-book combined for ref: {s['sharpe_both_active']:+.3f})")
PY
log "done. book-B cycles → $OUT/run/$POLICY/stateB/cycles.csv"
