"""Reconcile WS partial-day aggTrade files with the settled Vision archive (fixes the 'collector
started mid-day → permanent morning hole' issue). For settled days (yesterday/-2), if the on-disk
aggTrade file looks like a WS partial (first trade well after 00:00) AND Vision has published the
complete archive, replace it with Vision's and force-rebuild that symbol's flow cache.

SAFE: never deletes unless a HEAD check confirms Vision has the day (no data loss if unpublished).
Cheap when there's nothing to reconcile (just HEAD probes). Wired into run_convexity_twobook_live.sh.
"""
import sys, subprocess
from datetime import date, timedelta
from pathlib import Path
import pandas as pd, requests
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
from data_collectors.binance_vision_loader import list_aggtrade_paths, LoaderConfig, _archive_url
AGG = REPO/"data/ml/test/parquet/aggTrades"; OUT = REPO/"data/ml/test"


def _vision_has(sym, d):
    try:
        return requests.head(_archive_url("aggTrades", sym, d, None), timeout=10).status_code == 200
    except Exception:
        return False


def main():
    syms_file = REPO/"live/state/universe174.txt"
    syms = syms_file.read_text().split() if syms_file.exists() else []
    # only flow syms have caches worth rebuilding
    flow = {Path(f).stem.replace("flow_", "") for f in (REPO/"data/ml/cache").glob("flow_*.parquet")}
    today = date.today(); affected = set()
    for d in (today - timedelta(days=2), today - timedelta(days=1)):
        for s in syms:
            p = AGG/s/f"{d}.parquet"
            if not p.exists():
                continue
            try:
                first = pd.to_datetime(pd.read_parquet(p, columns=["transact_time"])["transact_time"].min(), utc=True)
            except Exception:
                continue
            if (first.hour, first.minute) <= (0, 10):      # already starts at day open → complete
                continue
            if not _vision_has(s, d):                       # Vision not published yet → leave WS file intact
                continue
            p.unlink()                                      # safe: Vision confirmed available
            list_aggtrade_paths(d, d, cfg=LoaderConfig(symbol=s, out_dir=OUT, max_workers=4))
            if p.exists() and s in flow:
                affected.add(s)
    if affected:
        subprocess.run([sys.executable, "-m", "scripts.build_aggtrade_features", "--force",
                        "--symbols", *sorted(affected)], cwd=str(REPO))
    print(f"[reconcile] replaced {len(affected)} flow sym(s) with settled Vision archive")


if __name__ == "__main__":
    main()
