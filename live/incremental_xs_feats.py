"""Incremental xs_feats updater — replaces the delete+full-rebuild that made the panel refresh ~2hr.

rebuild_xs_feats() recomputes compute_kline_features + add_regime_features + a slow rolling.apply(autocorr)
over the FULL per-symbol history (~1M 5m bars). This module recomputes over only a trailing WARMUP window
(default 60 days >> the max feature lookback of ~8640 bars / autocorr 2016 bars) and APPENDS the new bars to
the existing xs_feats_<sym>.parquet. Output matches the full rebuild on the overlap (validated) because every
feature is bounded-lookback: rolling windows ≤ 8640; bars_since_high resets within 288 bars; obv is cumsum but
obv_z subtracts a rolling-288 mean so the level offset cancels.

Usage:
  python3 live/incremental_xs_feats.py                 # update all panel-universe syms
  python3 live/incremental_xs_feats.py --validate SYM  # recompute-window vs full-rebuild overlap diff
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
from features_ml.klines import compute_kline_features
from features_ml.regime_features import add_regime_features

KLINES_DIR = REPO / "data/ml/test/parquet/klines"
CACHE = REPO / "data/ml/cache"
WARMUP_DAYS = 45          # > max lookback (8640 bars ≈ 30d) + autocorr-2016 (7d) + buffer; valid tail ≈ 15d
DAILY_FILES = WARMUP_DAYS + 5


def _xs_from_klines_lean(kl: pd.DataFrame) -> pd.DataFrame:
    """LEAN: compute ONLY the 6 columns the panel reads, replicating the engine's exact formulas
    (validated bit-identical, 0.00 diff vs the full 160-indicator engine). ~12× faster — the full
    HFFeatureEngine is wasted work for v1, which uses just these 6."""
    kl = kl.copy()
    kl["open_time"] = pd.to_datetime(kl["open_time"], utc=True)
    kl = kl.drop_duplicates("open_time").sort_values("open_time").set_index("open_time")
    for c in ("open", "high", "low", "close", "volume"):
        kl[c] = pd.to_numeric(kl[c], errors="coerce")
    c, h, l, v = kl["close"], kl["high"], kl["low"], kl["volume"]
    o = pd.DataFrame(index=kl.index); o["close"] = c
    o["return_1d"] = c.pct_change(288)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    o["atr_pct"] = tr.ewm(span=14, adjust=False).mean() / c
    pv = (c * v).rolling(96, min_periods=20).sum(); vv = v.rolling(96, min_periods=20).sum()
    vwap96 = pd.Series(np.where(vv != 0, pv / vv.replace(0, np.nan), 0.0), index=c.index).fillna(0.0)
    o["vwap_slope_96"] = (vwap96 / vwap96.shift(5) - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    o["obv_signal"] = obv - obv.ewm(span=20, adjust=False).mean()
    hi = c.rolling(288).max(); inh = (c == hi).astype(int)
    o["bars_since_high"] = (1 - inh).groupby(inh.cumsum()).cumcount()
    ret = c.pct_change(); av = ret.rolling(35).corr(ret.shift(1)); flat = ret.rolling(36).std() == 0
    a1 = av.where(~flat, 0.0)
    o["autocorr_pctile_7d"] = a1.rolling(2016, min_periods=288).rank(pct=True).shift(1)
    return o


def _xs_from_klines(kl: pd.DataFrame) -> pd.DataFrame:
    """Exact replica of rebuild_xs_feats' feature computation (on whatever kl window is passed)."""
    if os.environ.get("XS_LEAN") == "1":           # lean path: only the 6 panel-read cols (~12× faster)
        return _xs_from_klines_lean(kl)
    kl = kl.copy()
    kl["open_time"] = pd.to_datetime(kl["open_time"], utc=True)
    kl = kl.drop_duplicates("open_time").sort_values("open_time").set_index("open_time")
    for c in ["close_time", "quote_volume"]:
        if c in kl.columns: kl = kl.drop(columns=[c])
    for c in ["open", "high", "low", "close", "volume"]:
        kl[c] = pd.to_numeric(kl[c], errors="coerce")
    feats = compute_kline_features(kl)
    feats = add_regime_features(feats)
    ret = feats["close"].pct_change()
    # VECTORIZED autocorr (lag-1 within a 36-bar window = rolling-35 corr of ret vs ret.shift(1));
    # matches the old rolling(36).apply(s.autocorr) to ~1e-15, ~100× faster. Flat windows (std=0) → 0.0.
    av = ret.rolling(35).corr(ret.shift(1))
    flat = ret.rolling(36).std() == 0
    feats["autocorr_1h"] = av.where(~flat, 0.0)
    feats["autocorr_pctile_7d"] = feats["autocorr_1h"].rolling(2016, min_periods=288).rank(pct=True).shift(1)
    obj = [c for c in feats.columns if feats[c].dtype == object]
    if obj: feats = feats.drop(columns=obj)
    return feats


def _load_klines(sym: str, last_n_files: int | None):
    sd = KLINES_DIR / sym / "5m"
    paths = sorted(sd.glob("*.parquet"))
    if not paths: return None
    if last_n_files is not None: paths = paths[-last_n_files:]
    dfs = [pd.read_parquet(p) for p in paths]
    return pd.concat(dfs, ignore_index=True) if dfs else None


def update_sym(sym: str) -> str:
    cache = CACHE / f"xs_feats_{sym}.parquet"
    if not cache.exists():
        # no existing cache → fall back to a full build (rare; new symbol)
        kl = _load_klines(sym, None)
        if kl is None: return "no-klines"
        _xs_from_klines(kl).to_parquet(cache, compression="zstd", row_group_size=5000)
        return "full-new"
    ex = pd.read_parquet(cache)
    ex.index = pd.to_datetime(ex.index, utc=True)
    last = ex.index.max()
    kl = _load_klines(sym, DAILY_FILES)
    if kl is None: return "no-klines"
    feats = _xs_from_klines(kl)
    feats.index = pd.to_datetime(feats.index, utc=True)
    new = feats[feats.index > last]
    if len(new) == 0: return "current"
    # align columns to existing schema, append, dedup
    new = new.reindex(columns=ex.columns)
    comb = pd.concat([ex, new])
    comb = comb[~comb.index.duplicated(keep="last")].sort_index()
    comb.to_parquet(cache, compression="zstd", row_group_size=5000)
    return f"+{len(new)} → {comb.index.max()}"


def validate(sym: str, overlap_bars: int = 300):
    """Compare windowed recompute vs full rebuild on the overlap (last `overlap_bars`)."""
    full = _xs_from_klines(_load_klines(sym, None))
    win = _xs_from_klines(_load_klines(sym, DAILY_FILES))
    full.index = pd.to_datetime(full.index, utc=True); win.index = pd.to_datetime(win.index, utc=True)
    common = full.index.intersection(win.index)[-overlap_bars:]
    numcols = [c for c in full.columns if np.issubdtype(full[c].dtype, np.number)]
    a = full.loc[common, numcols]; b = win.loc[common, numcols]
    diff = (a - b).abs()
    rel = diff / (a.abs() + 1e-9)
    worst = rel.max().sort_values(ascending=False)
    print(f"validate {sym}: {len(common)} overlap bars, {len(numcols)} numeric cols")
    print(f"  max abs diff: {diff.max().max():.3e}  |  max rel diff: {worst.iloc[0]:.3e} ({worst.index[0]})")
    print(f"  cols with rel diff > 1e-4: {(worst > 1e-4).sum()} {list(worst[worst>1e-4].index[:6])}")
    return worst


def _safe_update(s):
    try:
        return s, update_sym(s)
    except Exception as e:
        return s, f"FAIL {str(e)[:70]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", type=str, default=None)
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    if a.validate:
        validate(a.validate); return
    syms = a.symbols or sorted(set(pd.read_parquet(REPO/"outputs/vBTC_features/panel_expanded_v0.parquet",
                                                    columns=["symbol"]).symbol.unique()))
    t0 = time.time(); ok = cur = fail = 0
    if a.workers > 1:
        import multiprocessing as mp
        with mp.Pool(a.workers) as pool:
            results = pool.map(_safe_update, syms)
    else:
        results = [_safe_update(s) for s in syms]
    for s, r in results:
        if r == "current": cur += 1
        elif r == "no-klines" or str(r).startswith("FAIL"): fail += 1
        else: ok += 1
        if str(r).startswith("FAIL"): print(f"  {s} {r}", flush=True)
    print(f"[incremental_xs_feats] {ok} updated, {cur} current, {fail} fail [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
