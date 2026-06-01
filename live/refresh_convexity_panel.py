"""Refresh the convexity panel for the paper bot.

1) Fetch fresh 5m klines (Binance Vision archive — free; lags ~1d) for the deploy
   universe via data_collectors/binance_vision_loader. Saves to
   data/ml/test/parquet/klines/{sym}/5m/{date}.parquet (X132's input layout).
2) Rebuild panel_expanded_v0.parquet + x132 walk-forward preds by re-running
   X132_build_expanded_panel.py as a subprocess. The model artifact only needs
   re-training quarterly per iter-033 — that's a separate job, not done here.

Usage: python -m live.refresh_convexity_panel [--syms ALL|FROM_PANEL]
"""
from __future__ import annotations
import argparse, subprocess, sys, time
from datetime import date, timedelta
from pathlib import Path
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
KLINES_DIR = REPO/"data/ml/test"


def list_syms_from_panel() -> list[str]:
    """Symbols already in the panel = the trained universe. Always include
    BTCUSDT — it's the regime reference + cross-asset feature source, even though
    excluded from the tradable panel."""
    p = pd.read_parquet(PANEL, columns=["symbol"])
    syms = sorted(p["symbol"].unique().tolist())
    if "BTCUSDT" not in syms: syms = sorted(syms + ["BTCUSDT"])
    return syms


def refresh_klines(syms: list[str], days_back: int = 14):
    """Fetch the last `days_back` days of 5m klines for each sym via Binance Vision."""
    from data_collectors.binance_vision_loader import fetch_klines, LoaderConfig
    today = date.today(); start = today - timedelta(days=days_back)
    t0 = time.time(); ok = 0
    print(f"refresh klines: {len(syms)} syms × {days_back}d ({start}→{today})", flush=True)
    for i, sym in enumerate(syms, 1):
        try:
            df = fetch_klines(start, today, interval="5m",
                              cfg=LoaderConfig(symbol=sym, out_dir=KLINES_DIR, max_workers=4))
            ok += int(len(df) > 0)
            if i % 25 == 0 or i == len(syms):
                print(f"  [{i}/{len(syms)}] {sym:<14} {len(df):>6} rows [{time.time()-t0:.0f}s]", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(syms)}] {sym:<14} ERR {type(e).__name__}: {e}", flush=True)
    print(f"klines refresh: {ok}/{len(syms)} OK [{time.time()-t0:.0f}s]", flush=True)


def invalidate_stale_xs_feats():
    """Delete xs_feats cache entries that are older than their underlying 5m klines.
    Without this, X132 reuses stale per-sym features and the panel caps at the cache age."""
    CACHE = REPO/"data/ml/cache"; KL = REPO/"data/ml/test/parquet/klines"
    n_stale = n_ok = n_missing_klines = 0
    for xs in CACHE.glob("xs_feats_*.parquet"):
        sym = xs.stem.replace("xs_feats_", "")
        kd = KL/sym/"5m"
        if not kd.exists(): n_missing_klines += 1; continue
        files = list(kd.glob("*.parquet"))
        if not files: n_missing_klines += 1; continue
        latest_k = max(f.stat().st_mtime for f in files)
        xs_m = xs.stat().st_mtime
        if latest_k > xs_m + 60:    # 60s grace
            xs.unlink(); n_stale += 1
        else: n_ok += 1
    print(f"xs_feats invalidation: {n_stale} stale deleted, {n_ok} fresh kept, "
          f"{n_missing_klines} no-klines", flush=True)


def rebuild_panel():
    print("invalidate stale xs_feats caches...", flush=True)
    invalidate_stale_xs_feats()
    print("rebuild panel via X132...", flush=True)
    t0 = time.time()
    cmd = ["python3", str(REPO/"research/convexity_portable_2026-05-20/scripts/X132_build_expanded_panel.py")]
    env = {"PYTHONPATH": str(REPO)}
    import os; full_env = {**os.environ, **env}
    res = subprocess.run(cmd, env=full_env, capture_output=True, text=True, timeout=2400)
    print(res.stdout[-2000:] if res.stdout else "(no stdout)", flush=True)
    if res.returncode != 0:
        print(f"X132 FAILED ({res.returncode}):\n{res.stderr[-2000:]}", flush=True); return False
    print(f"panel rebuilt [{time.time()-t0:.0f}s]", flush=True); return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syms", default="FROM_PANEL", choices=["FROM_PANEL"])
    ap.add_argument("--days-back", type=int, default=14)
    ap.add_argument("--skip-fetch", action="store_true",
                    help="skip klines fetch; only rebuild panel")
    ap.add_argument("--skip-rebuild", action="store_true",
                    help="only fetch klines; skip panel rebuild")
    args = ap.parse_args()
    syms = list_syms_from_panel()
    print(f"convexity refresh: {len(syms)} syms (from panel)", flush=True)
    if not args.skip_fetch: refresh_klines(syms, args.days_back)
    if not args.skip_rebuild:
        ok = rebuild_panel()
        if not ok: sys.exit(2)
    print("refresh DONE", flush=True)


if __name__ == "__main__": main()
