"""Lean panel builder — X132 sections B+C (build_sym + target_z + cohort + xs-rank) WITHOUT the
redundant V0 walk-forward pred regen. The two-book forward test uses the separate iter28 preds, so
X132's own pred regeneration (~10min) is wasted work in the live loop. This builds only the panel.

Assumes xs_feats caches are current (run live/incremental_xs_feats.py first). Universe = the symbol set
already in panel_expanded_v0.parquet (the fixed deploy universe; no /tmp/addable.json dependency).
"""
from __future__ import annotations
import time, gc, importlib.util
from pathlib import Path
import pandas as pd, numpy as np
REPO = Path("/home/yuqing/ctaNew")
spec = importlib.util.spec_from_file_location(
    "x70mod", REPO/"research/convexity_portable_2026-05-20/scripts/X70_build_3yr_and_regime_test.py")
X70 = importlib.util.module_from_spec(spec); spec.loader.exec_module(X70)
x6, x6b = X70.x6, X70.x6b
PANEL_OUT = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
KEEP_OBJ = {"symbol", "open_time", "exit_time"}


def _sample_4h(df):
    ot = pd.to_datetime(df["open_time"], utc=True)
    return df[(ot.dt.hour % 4 == 0) & (ot.dt.minute == 0)]


_BTC = None
def _init(btc):
    global _BTC; _BTC = btc

def _build_one(sym):
    try:
        sdf = X70.build_sym(sym, _BTC)
        if sdf is None or len(sdf) == 0: return None
        sdf["open_time"] = pd.to_datetime(sdf["open_time"], utc=True)
        sdf["exit_time"] = pd.to_datetime(sdf["exit_time"], utc=True)
        sdf = sdf.dropna(subset=["alpha_vs_btc_realized"])
        if len(sdf) == 0: return None
        sdf = x6.build_target_z(sdf)
        sdf = _sample_4h(sdf)
        if len(sdf) == 0: return None
        for c in sdf.columns:
            if c not in KEEP_OBJ and pd.api.types.is_float_dtype(sdf[c]):
                sdf[c] = sdf[c].astype("float32")
        return sdf
    except Exception as e:
        print(f"  {sym} ERR {type(e).__name__}: {e}", flush=True); return None


def main():
    import argparse, multiprocessing as mp
    # MEMORY: each build_sym worker decompresses a full-history xs_feats frame (~1-1.5GB). On a 30GB
    # box, >5 workers risks OOM. Default 4 (~6-8GB peak). Do NOT raise without checking free -g.
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    t0 = time.time()
    syms = sorted(set(pd.read_parquet(PANEL_OUT, columns=["symbol"])["symbol"].unique()))
    syms_no_btc = [s for s in syms if s != "BTCUSDT"]
    print(f"=== lean panel build (no preds, {a.workers}w): {len(syms_no_btc)} legs ===", flush=True)
    btc_close = X70.load_closes("BTCUSDT")
    if btc_close is None:
        raise RuntimeError("BTCUSDT closes missing")
    if a.workers > 1:
        with mp.Pool(a.workers, initializer=_init, initargs=(btc_close,)) as pool:
            sdfs = [d for d in pool.map(_build_one, syms_no_btc) if d is not None]
    else:
        _init(btc_close); sdfs = [d for d in (_build_one(s) for s in syms_no_btc) if d is not None]
    n_ok = len(sdfs)
    print(f"  per-sym build: {n_ok} ok [{time.time()-t0:.0f}s]", flush=True)
    panel = pd.concat(sdfs, ignore_index=True); del sdfs; gc.collect()
    panel = x6b.build_cohort_fixed(panel)
    x6.HEAVY_TAIL.discard("rvol_7d"); x6.HEAVY_TAIL.discard("ret_3d"); x6.HEAVY_TAIL.discard("btc_rvol_7d")
    panel["bars_since_high_xs_rank"] = (panel.groupby("open_time")["bars_since_high"]
                                        .rank(pct=True).astype("float32"))
    panel.to_parquet(PANEL_OUT, index=False)
    print(f"  panel: {len(panel):,} rows × {panel['symbol'].nunique()} syms "
          f"{panel['open_time'].min().date()}→{panel['open_time'].max().date()} [{time.time()-t0:.0f}s] DONE", flush=True)


if __name__ == "__main__":
    main()
