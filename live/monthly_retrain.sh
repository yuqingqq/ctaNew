#!/usr/bin/env bash
# Monthly retrain on the TRAINING box (this one): refresh data → retrain two-book models on the latest
# data → commit + push to git. The LIVE server then `git pull`s the fresh models. Models are ~44MB (git-able).
set -uo pipefail
ROOT=/home/yuqing/ctaNew; export PYTHONPATH=$ROOT; cd $ROOT; PY=python3
log(){ echo "[$(date -u '+%F %T')] $*"; }
log "monthly retrain start"
# 1. ensure data current (the daily pipeline keeps klines/flow/panel fresh; refresh defensively)
$PY live/ingest_funding.py --months 2 || log "funding ingest warn"   # durable fix for the 2026-04 funding stall
$PY live/ingest_flow_daily.py --workers 4 || log "flow ingest warn"
$PY -m live.refresh_convexity_panel --days-back 10 --skip-rebuild || log "klines warn"
$PY live/incremental_xs_feats.py --workers 6 || log "xs_feats warn"
$PY live/incremental_panel.py --workers 6 || log "panel warn"
# 2. retrain on the latest data (default fit_cut = latest panel - 1d embargo)
$PY live/train_twobook_models.py || { log "train FAILED"; exit 1; }
# 3. commit + push the fresh convexity v1 models (single low-vol book) + frozen universe (static for the month)
git add live/models/convexity_v1_short_model.pkl live/models/convexity_v1_long_model.pkl live/models/convexity_v1_universe.json
if git diff --cached --quiet; then log "models unchanged, nothing to push"; else
  CUT=$($PY -c "import pickle;print(pickle.load(open('live/models/convexity_v1_short_model.pkl','rb'))['meta']['fit_cut'][:10])")
  git commit -q -m "monthly retrain: convexity v1 models @ fit_cut $CUT" && git push -q origin main && log "pushed models @ $CUT"
fi
log "monthly retrain done"
