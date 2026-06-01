"""Daily incremental flow ingestion for the convexity two-book forward test (BookA dependency).

For each universe symbol: fetch the newly-published daily aggTrade archives since the symbol's flow
parquet's last bar, recompute flow features over a trailing warmup window (so rolling/VPIN stats are
correct across the boundary), and APPEND the new bars to flow_<SYM>.parquet.

Single-process with modest download parallelism (max_workers=6). Do NOT run many sharded copies —
24+ concurrent downloads throttle Binance Vision and silently drop days (learned 2026-06-01).

Cron (run after ~02:00 UTC so the prior UTC day is published):
  15 3 * * *  cd /home/yuqing/ctaNew && PYTHONPATH=. python3 live/ingest_flow_daily.py >> live/state/convexity_bookA/ingest.log 2>&1
"""
import sys, time
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
sys.path.insert(0, "/home/yuqing/ctaNew")
import logging; logging.basicConfig(level=logging.ERROR)
from data_collectors.binance_vision_loader import LoaderConfig, list_aggtrade_paths
import scripts.build_aggtrade_features as B
from features_ml.trade_flow import aggregate_trades_streaming

REPO = Path("/home/yuqing/ctaNew")
CACHE = B.CACHE_DIR
AGG = B.DATA_DIR / "aggTrades"
OUT = Path("data/ml/test")
cfg = B.TradeFlowConfig(bar_interval="5min", compute_kyle_lambda=True)
WARMUP_FILES = 16   # trailing daily aggTrade files: VPIN needs 7d (2016 bars), +buffer. Validated: 14d
                    # window matches full-history cache to ~1e-12 on recent bars; 6.4s/sym vs 26s at 50d (4×).


TAIL_BARS = 2200   # cached per-bar rows used as rolling/VPIN warmup (vpin lookback 2016 + buffer)

def _update_one(sym):
    """TRUE-incremental: per_bar_features (incl. slow Kyle) on NEW daily files ONLY; reuse the cached
    per-bar columns as the rolling/VPIN warmup tail. Avoids recomputing ~16 days of Kyle every cycle.
    Output matches the full-window recompute (validated). Returns status."""
    from features_ml.trade_flow import per_bar_features, add_rolling_features
    end = (pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=1)).date()
    cache = CACHE / f"flow_{sym}.parquet"
    if not cache.exists():
        return "skip-nocache"
    try:
        ex = pd.read_parquet(cache); ex.index = pd.to_datetime(ex.index, utc=True)
        last = ex.index.max(); last_day = last.normalize().date()
        list_aggtrade_paths(last_day, end, cfg=LoaderConfig(symbol=sym, out_dir=OUT, max_workers=4))
        # NEW daily files only (>= last cached day, to complete the partial last day)
        new_files = [p for p in sorted((AGG / sym).glob("*.parquet")) if p.stem >= str(last_day)]
        if not new_files:
            return "no-paths"
        np_parts = [per_bar_features(pd.read_parquet(p), cfg) for p in new_files]      # Kyle on NEW only
        new_pb = pd.concat(np_parts).sort_index()
        new_pb = new_pb[~new_pb.index.duplicated(keep="last")]
        new_pb.index = pd.to_datetime(new_pb.index, utc=True)
        pb_cols = list(new_pb.columns)
        tail = ex.tail(TAIL_BARS)                                                       # rolling warmup
        combined = pd.concat([tail, new_pb[pb_cols]])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        rolled = add_rolling_features(combined, cfg)                                    # VPIN loop ~tail→new only
        new = rolled[rolled.index > last]
        if len(new) == 0:
            return "current"
        new = new.reindex(columns=ex.columns)
        comb = pd.concat([ex, new]); comb = comb[~comb.index.duplicated(keep="last")].sort_index()
        comb.to_parquet(cache, compression="zstd", row_group_size=5000)
        return f"+{len(new)}"
    except Exception as e:
        return f"FAIL {str(e)[:70]}"


def main():
    import argparse, multiprocessing as mp
    # MEMORY: each worker streams ~14 days of aggTrades (millions of rows) → ~1-2GB peak. Cap at 4 on a
    # 30GB box (~8GB peak). The flow recompute (per-bar VPIN/kyle loops) is the slowest stage; 14-day
    # warmup (validated vs full to ~1e-12) keeps it to ~6s/sym → ~5min @4w for 175.
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    syms = sorted(set(pd.read_parquet(REPO / "outputs/vBTC_features/panel_expanded_v0.parquet",
                                      columns=["symbol"]).symbol.unique()))
    t0 = time.time()
    if a.workers > 1:
        with mp.Pool(a.workers) as pool:
            res = pool.map(_update_one, syms)
    else:
        res = [_update_one(s) for s in syms]
    ok = sum(1 for r in res if str(r).startswith("+"))
    cur = sum(1 for r in res if r == "current")
    fail = sum(1 for r in res if str(r).startswith("FAIL") or r in ("no-paths",))
    for s, r in zip(syms, res):
        if str(r).startswith("FAIL"): print(f"  {s} {r}", flush=True)
    print(f"[ingest_flow] {ok} extended, {cur} current, {fail} fail [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
