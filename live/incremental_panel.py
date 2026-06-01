"""Incremental panel updater — appends ONLY the new 4h bars to panel_expanded_v0.parquet instead of
rebuilding all ~1M rows every cycle. Same output as build_panel_fast/X132 (validated), but ~seconds
and low-memory (never holds full-history per-sym frames → no OOM).

Approach (mirrors the validated windowing pattern):
  - per sym: windowed build_sym over a ~14-day input (reuses X70.btc_cross / target_alpha; replicates the
    obv_z + funding lines), keep only 4h rows after the panel's last open_time.
  - windowed cohort (rvol_7d/ret_3d/btc_rvol_7d) for the new rows from a ~14-day close window.
  - bars_since_high_xs_rank: cross-sectional rank per NEW open_time (self-contained).
  - append; then recompute build_target_z over the full panel (cheap one-column expanding/rolling; correct
    for the expanding mean). target_z isn't used by iter28 but kept correct for the monthly retrain.
Validate: python3 live/incremental_panel.py --validate  (truncates a copy, re-appends, compares to full).
"""
from __future__ import annotations
import argparse, time, importlib.util, multiprocessing as mp
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path("/home/yuqing/ctaNew")
spec = importlib.util.spec_from_file_location(
    "x70mod", REPO/"research/convexity_portable_2026-05-20/scripts/X70_build_3yr_and_regime_test.py")
X70 = importlib.util.module_from_spec(spec); spec.loader.exec_module(X70)
x6, x6b = X70.x6, X70.x6b
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
CACHE = REPO/"data/ml/cache"
KLINES = REPO/"data/ml/test/parquet/klines"
HORIZON = X70.HORIZON
WARMUP_DAYS = 14          # > build_sym max lookback (funding rolling-2016 = 7d) + HORIZON forward + buffer
KEEP_OBJ = {"symbol", "open_time", "exit_time"}
_BTC_FULL = None


def _closes_tail(sym, since):
    """5m closes for `sym` from daily files on/after (since - WARMUP_DAYS)."""
    sd = KLINES/sym/"5m"
    if not sd.exists(): return None
    cutoff = (since - pd.Timedelta(days=WARMUP_DAYS)).date()
    paths = [p for p in sorted(sd.glob("*.parquet")) if p.stem >= str(cutoff)]
    if not paths: return None
    dfs = [pd.read_parquet(p, columns=["open_time", "close"]) for p in paths]
    c = pd.concat(dfs, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
    return c.set_index("open_time")["close"].astype(float)


def _build_sym_window(sym, since):
    """Windowed replica of X70.build_sym: returns new 4h rows (open_time > since) with V0 features."""
    xs_path = CACHE/f"xs_feats_{sym}.parquet"
    if not xs_path.exists(): return None
    cutoff = since - pd.Timedelta(days=WARMUP_DAYS)
    try:                                            # fast row-group-skipping tail read (10-50× less I/O)
        xs = pd.read_parquet(xs_path, filters=[("open_time", ">=", cutoff)])
        if "open_time" in xs.columns: xs = xs.set_index("open_time")
    except Exception:                               # fallback: full read + slice (single-row-group caches)
        xs = pd.read_parquet(xs_path)
    xs.index = pd.DatetimeIndex(xs.index).tz_convert("UTC")
    xs = xs[xs.index >= cutoff]                                      # WINDOW (idempotent w/ filter)
    if len(xs) == 0: return None
    my_close = _closes_tail(sym, since)
    if my_close is None or my_close.empty: return None
    btc = _BTC_FULL[_BTC_FULL.index >= since - pd.Timedelta(days=WARMUP_DAYS)]
    bc = X70.btc_cross(my_close, btc)
    alpha, my_fwd = X70.target_alpha(my_close, btc)
    obv = xs.get("obv_signal")
    if obv is not None:
        obv_z = ((obv - obv.rolling(288, min_periods=72).mean()) /
                 obv.rolling(288, min_periods=72).std().replace(0, np.nan)).shift(1).astype(np.float32)
    else:
        obv_z = pd.Series(np.nan, index=xs.index, dtype=np.float32)
    fr = fr_z = fr_c = pd.Series(np.nan, index=xs.index, dtype=np.float32)
    fp = CACHE/f"funding_{sym}.parquet"
    if fp.exists():
        fund = pd.read_parquet(fp); tc = "calc_time" if "calc_time" in fund.columns else "open_time"
        if tc in fund.columns and "funding_rate" in fund.columns:
            fund[tc] = pd.to_datetime(fund[tc], utc=True); fund = fund.set_index(tc).sort_index()
            fund = fund[~fund.index.duplicated(keep="last")]
            frr = fund["funding_rate"].reindex(xs.index, method="ffill")
            fr_z = ((frr - frr.rolling(288*7, min_periods=288).mean()) /
                    frr.rolling(288*7, min_periods=288).std().replace(0, np.nan)).shift(1).astype(np.float32)
            fr_c = frr.diff(288).shift(1).astype(np.float32); fr = frr.shift(1).astype(np.float32)
    out = pd.DataFrame({
        "symbol": sym, "open_time": xs.index,
        "return_pct": my_fwd.reindex(xs.index), "exit_time": xs.index + pd.Timedelta(minutes=5*HORIZON),
        "alpha_vs_btc_realized": alpha.reindex(xs.index),
        "return_1d": xs.get("return_1d", np.nan).astype(np.float32) if "return_1d" in xs else np.nan,
        "atr_pct": xs.get("atr_pct", np.nan).astype(np.float32) if "atr_pct" in xs else np.nan,
        "vwap_slope_96": xs.get("vwap_slope_96", np.nan).astype(np.float32) if "vwap_slope_96" in xs else np.nan,
        "bars_since_high": xs.get("bars_since_high", np.nan).astype(np.float32) if "bars_since_high" in xs else np.nan,
        "autocorr_pctile_7d": xs.get("autocorr_pctile_7d", np.nan).astype(np.float32) if "autocorr_pctile_7d" in xs else np.nan,
        "obv_z_1d": obv_z, "corr_to_btc_1d": bc["corr_to_btc_1d"].reindex(xs.index),
        "beta_to_btc_change_5d": bc["beta_to_btc_change_5d"].reindex(xs.index),
        "idio_vol_to_btc_1h": bc["idio_vol_to_btc_1h"].reindex(xs.index),
        "idio_vol_to_btc_1d": bc["idio_vol_to_btc_1d"].reindex(xs.index),
        "funding_rate": fr, "funding_rate_z_7d": fr_z, "funding_rate_1d_change": fr_c,
    }).reset_index(drop=True)
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True)
    out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True)
    out = out.dropna(subset=["alpha_vs_btc_realized"])
    m = (out["open_time"].dt.hour % 4 == 0) & (out["open_time"].dt.minute == 0) & (out["open_time"] > since)
    return out[m]


def _cohort_window(new, since):
    """rvol_7d/ret_3d/btc_rvol_7d for the new rows, from ~14-day close windows."""
    syms = sorted(set(new["symbol"].unique()) | {"BTCUSDT"})
    rvol_rows, ret_rows = [], []
    btc_rv = None
    for sym in syms:
        c = _closes_tail(sym, since)
        if c is None or c.empty: continue
        lr = np.log(c / c.shift(1))
        rv = lr.rolling(288*7, min_periods=288).std().shift(1)
        if sym == "BTCUSDT":
            btc_rv = rv.rename("btc_rvol_7d").reset_index(); continue
        rt = c.pct_change(288*3).shift(1)
        dr = rv.rename("rvol_7d").reset_index(); dr["symbol"] = sym; rvol_rows.append(dr)
        dt = rt.rename("ret_3d").reset_index(); dt["symbol"] = sym; ret_rows.append(dt)
    rvol = pd.concat(rvol_rows, ignore_index=True); ret3 = pd.concat(ret_rows, ignore_index=True)
    for df in (rvol, ret3, btc_rv): df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    new = new.merge(rvol, on=["symbol", "open_time"], how="left")
    new = new.merge(ret3, on=["symbol", "open_time"], how="left")
    new = new.merge(btc_rv, on="open_time", how="left")
    return new


def run(panel_path=PANEL, workers=6):
    global _BTC_FULL
    t0 = time.time()
    panel = pd.read_parquet(panel_path)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    since = panel["open_time"].max()
    syms = sorted(s for s in panel["symbol"].unique() if s != "BTCUSDT")
    _BTC_FULL = X70.load_closes("BTCUSDT")
    _BTC_FULL.index = pd.DatetimeIndex(_BTC_FULL.index).tz_convert("UTC")
    print(f"[inc_panel] panel last={since}, {len(syms)} syms, warmup={WARMUP_DAYS}d", flush=True)
    def _w(s):
        try: return _build_sym_window(s, since)
        except Exception as e: print(f"  {s} ERR {str(e)[:60]}", flush=True); return None
    if workers > 1:
        with mp.Pool(workers, initializer=_init_btc, initargs=(_BTC_FULL,)) as pool:
            parts = pool.map(_w_top, [(s, since) for s in syms])
    else:
        parts = [_w(s) for s in syms]
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        print(f"[inc_panel] 0 new bars — panel current [{time.time()-t0:.0f}s]", flush=True); return panel, 0
    new = pd.concat(parts, ignore_index=True)
    new = _cohort_window(new, since)
    new["bars_since_high_xs_rank"] = (new.groupby("open_time")["bars_since_high"].rank(pct=True).astype("float32"))
    new = new.reindex(columns=panel.columns)
    for c in new.columns:
        if c not in KEEP_OBJ and pd.api.types.is_float_dtype(panel[c]):
            new[c] = new[c].astype("float32")
    combined = pd.concat([panel, new], ignore_index=True)
    combined = x6.build_target_z(combined)            # cheap full recompute (expanding mean, 1 col)
    combined = combined.reindex(columns=panel.columns) if set(panel.columns)<=set(combined.columns) else combined
    combined.to_parquet(panel_path, index=False)
    print(f"[inc_panel] appended {len(new)} rows ({new['open_time'].nunique()} new cycles) "
          f"→ {combined['open_time'].max()} [{time.time()-t0:.0f}s]", flush=True)
    return combined, len(new)


# module-level workers (picklable)
def _init_btc(btc):
    global _BTC_FULL; _BTC_FULL = btc
def _w_top(args):
    s, since = args
    try: return _build_sym_window(s, since)
    except Exception as e: print(f"  {s} ERR {str(e)[:60]}", flush=True); return None


def validate(workers=6):
    """Truncate a panel copy to (last-3d), incrementally re-append, compare new rows vs the real panel."""
    full = pd.read_parquet(PANEL); full["open_time"] = pd.to_datetime(full["open_time"], utc=True)
    cut = full["open_time"].max() - pd.Timedelta(days=3)
    trunc = full[full["open_time"] <= cut].copy()
    tmp = REPO/"live/state/convexity/_panel_inc_test.parquet"; trunc.to_parquet(tmp, index=False)
    rebuilt, n = run(panel_path=tmp, workers=workers)
    rebuilt["open_time"] = pd.to_datetime(rebuilt["open_time"], utc=True)
    V0 = x6.BASE + x6.COHORT_EXTRAS
    m = full.merge(rebuilt, on=["symbol", "open_time"], suffixes=("_f", "_i"), how="inner")
    m = m[m["open_time"] > cut]
    worst = 0.0; wc = None
    for c in V0:
        if c+"_f" in m and c+"_i" in m:
            a = m[c+"_f"].astype(float); b = m[c+"_i"].astype(float)
            adiff = (a-b).abs()
            # rel diff only where the value is non-trivial (avoid div-by-near-zero, e.g. flat-coin rvol≈1e-9)
            mask = a.abs() > 1e-6
            rel = (adiff[mask]/a.abs()[mask]).max() if mask.any() else 0.0
            rel = max(rel, adiff.max())   # also flag if even the ABSOLUTE diff is large
            if rel > worst: worst = rel; wc = c
    print(f"[validate] {len(m)} appended rows compared; V0 max (rel|abs) diff {worst:.2e} ({wc}) "
          f"{'MATCH' if worst<1e-4 else 'DIFF'}")
    tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true"); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.validate: validate(a.workers)
    else: run(workers=a.workers)
