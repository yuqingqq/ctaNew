#!/usr/bin/env bash
# Retention/cleanup for raw market data the WS collector + Vision loader accumulate.
#
# The flow pipeline only needs a trailing WARMUP window of raw aggTrades (ingest_flow_daily uses
# WARMUP_FILES=16 days; the computed features live permanently in data/ml/cache/flow_<SYM>.parquet).
# So raw aggTrade daily parquets older than RETENTION_AGG days are safe to delete once the flow caches
# are current. Klines are tiny and feed a longer panel window, so keep more of them.
#
# Cron (daily, AFTER the flow-ingest step so the caches are current first):
#   45 3 * * *  bash /home/yuqing/ctaNew/live/prune_raw_data.sh >> /home/yuqing/ctaNew/live/state/prune.log 2>&1
set -uo pipefail
ROOT=/home/yuqing/ctaNew; cd $ROOT
# Bootstrap-aware retention: raw aggTrades are ~0.5-1 GB/day for the full universe, so we keep as little
# as the pipeline needs. Steady state only needs the un-ingested tail (~2 days); the computed features
# live permanently in data/ml/cache/flow_<SYM>.parquet. The ONE exception is the initial flow-cache build,
# which needs a ~16-day VPIN/Kyle warmup at once — so until the caches exist we keep the longer window.
NFLOW=$(ls data/ml/cache/flow_*.parquet 2>/dev/null | wc -l)
if [ "$NFLOW" -lt 100 ]; then
  RETENTION_AGG=${RETENTION_AGG:-18}   # bootstrap not done — preserve the 16d warmup (+buffer)
else
  RETENTION_AGG=${RETENTION_AGG:-3}    # steady state — need is 2 (yesterday+today; warmup is in cache),
                                       # +1 buffer for a missed ingest. ~2-3 GB total.
fi
RETENTION_KL=${RETENTION_KL:-120}      # days of raw kline day-files to keep (panel uses ~60-90d; tiny)
CUT_AGG=$(date -u -d "-${RETENTION_AGG} days" '+%Y-%m-%d')
CUT_KL=$(date -u -d "-${RETENTION_KL} days" '+%Y-%m-%d')
log(){ echo "[$(date -u '+%F %T')] $*"; }

# Delete by FILENAME date (stem = YYYY-MM-DD), not mtime — robust to re-writes.
prune() {  # $1 = glob root, $2 = cutoff date
  local n=0 bytes=0
  while IFS= read -r f; do
    local stem; stem=$(basename "$f" .parquet)
    if [[ "$stem" < "$2" ]]; then
      bytes=$(( bytes + $(stat -c%s "$f" 2>/dev/null || echo 0) )); rm -f "$f"; n=$((n+1))
    fi
  done < <(find "$1" -name '*.parquet' 2>/dev/null)
  echo "$n $bytes"
}

log "prune aggTrades older than $CUT_AGG ($RETENTION_AGG d)"
read an ab < <(prune data/ml/test/parquet/aggTrades "$CUT_AGG")
log "  removed $an aggTrade day-files ($((ab/1000000)) MB)"

log "prune raw Vision zips (aggTrades) older than $CUT_AGG"
# Vision loader caches raw .zip under data/ml/test/raw/; collector writes none, but backfill/Vision do.
rz=$(find data/ml/test/raw/aggTrades -name '*.zip' 2>/dev/null | while read -r z; do
       d=$(basename "$z" .zip | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}'); [[ -n "$d" && "$d" < "$CUT_AGG" ]] && rm -f "$z" && echo x; done | wc -l)
log "  removed $rz raw aggTrade zips"

log "prune kline day-files older than $CUT_KL ($RETENTION_KL d)"
read kn kb < <(prune data/ml/test/parquet/klines "$CUT_KL")
log "  removed $kn kline day-files ($((kb/1000000)) MB)"

# drop now-empty symbol dirs
find data/ml/test/parquet/aggTrades -type d -empty -delete 2>/dev/null || true
log "disk now: $(df -h /home/yuqing | awk 'NR==2{print $4" free"}')"
