#!/usr/bin/env bash
# Daily aggTrade flow ingestion for the convexity two-book forward test (BookA dependency).
#
# Binance publishes daily aggTrade archives ~1 day in arrears. This pulls the latest
# available day for the flow-sym universe and rebuilds flow_<SYM>.parquet incrementally,
# so the flow model (BookA) has up-to-date microstructure features.
#
# Run once/day (cron), AFTER ~02:00 UTC (gives Binance time to publish yesterday):
#   0 3 * * *  bash /home/yuqing/ctaNew/live/ingest_flow_daily.sh >> /home/yuqing/ctaNew/live/state/convexity_bookA/ingest.log 2>&1
#
# The flow-sym universe = symbols that have flow_*.parquet today. New symbols are added by
# the quarterly universe refresh, not here.
set -uo pipefail
ROOT=/home/yuqing/ctaNew
export PYTHONPATH=$ROOT
cd $ROOT
log(){ echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] $*"; }

# Flow-sym universe (basenames of existing flow caches; --symbols is space-separated)
SYMS=$(ls data/ml/cache/flow_*.parquet 2>/dev/null | sed 's#.*/flow_##;s#\.parquet##' | paste -sd' ' -)
if [ -z "$SYMS" ]; then log "no flow caches found — run the full fetch first"; exit 1; fi
N=$(echo "$SYMS" | wc -w)
log "rebuilding flow for $N syms (--force: full re-pull through latest published day)"

# NOTE / KNOWN GAP: scripts/build_aggtrade_features.py SKIPS symbols whose flow_<SYM>.parquet
# already exists (resume mode for the bulk fetch). There is no incremental daily-append mode
# yet, so to pick up new days we must --force a full rebuild (re-pulls the whole history per
# sym — heavy: ~100s/sym × N). This is acceptable as a nightly job for a small flow universe
# but should be replaced by an incremental appender before scaling. Tracked in CONVEXITY_BOT_LAUNCH.md.
python3 scripts/build_aggtrade_features.py --force --symbols $SYMS \
  && log "flow ingest OK" || log "flow ingest FAILED (BookA may run on stale flow)"
