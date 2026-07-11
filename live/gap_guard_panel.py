"""Audit #1 remediation — replace fix_panel_labels (which over-masked 175 VALID labels by keying on PANEL
row-spacing) with a principled gap-guard keyed on REAL 5-minute raw-kline gaps.

A rolling feature computed on observed rows is contaminated at any row whose trailing window reaches back
ACROSS a real data gap. This drops exactly those rows:
  - trailing-window guard: W=7.5d (covers the longest BOUNDED feature: funding_z_7d / autocorr_7d ~7d),
  - unbounded guard: bars_since_high is a cumcount (empirical max ~11.9d) -> drop a row if its run reaches
    back across a gap (variable length), covering it without over-dropping short-run rows,
  - BTC gaps apply UNIVERSE-WIDE (cross-features corr/beta/idio/alpha/btc_rvol broadcast off BTC klines),
then RECOMPUTES the cross-sectional rank (bars_since_high_xs_rank, needs the clean universe) and the target
normalization (target_z, expanding) over the surviving rows. Writes panel_expanded_v0_clean.parquet.

Keyed on real gaps (not panel spacing) -> restores the 175 valid labels fix_panel_labels wrongly masked, and
masks FEATURES not just labels. Verified against audit #1.
"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO / "research/convexity_portable_2026-05-20/scripts"))
KD = REPO / "data/ml/test/parquet/klines"
SRC = REPO / "outputs/vBTC_features/panel_expanded_v0.parquet"
OUT = REPO / "outputs/vBTC_features/panel_expanded_v0_clean.parquet"
W = pd.Timedelta(hours=180)        # 7.5d trailing-window guard (bounded features ~7d + margin)
GAP = pd.Timedelta(minutes=6)      # >5min between consecutive 5m bars = missing bar(s)

def gap_intervals(sym):
    fs = sorted(glob.glob(str(KD / sym / "5m" / "*.parquet")))
    if not fs: return []
    t = pd.concat([pd.read_parquet(f, columns=["open_time"]) for f in fs], ignore_index=True)["open_time"]
    t = pd.to_datetime(t, utc=True).drop_duplicates().sort_values().reset_index(drop=True)
    d = t.diff()
    idx = np.where(d > GAP)[0]
    return [(t[i - 1], t[i]) for i in idx]   # (last bar before gap, first bar after gap)

def window_mask(times, gaps):
    m = np.zeros(len(times), dtype=bool)
    for g0, g1 in gaps:   # tz-aware pandas comparisons (numpy strips tz)
        m |= ((times > g0) & (times <= (g1 + W))).to_numpy()
    return m

def main():
    pan = pd.read_parquet(SRC)
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    pan = pan.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    n0 = len(pan)
    print(f"panel: {n0:,} rows, {pan.symbol.nunique()} syms, {pan.open_time.min()}..{pan.open_time.max()}", flush=True)

    drop = np.zeros(n0, dtype=bool)
    # BTC gaps -> universe-wide (cross-features broadcast off BTC)
    btc_gaps = gap_intervals("BTCUSDT")
    print(f"BTC gaps: {len(btc_gaps)} -> {[(str(a)[:16], str(b)[:16]) for a, b in btc_gaps]}", flush=True)
    drop |= window_mask(pan["open_time"], btc_gaps)
    print(f"  after BTC-gap (universe-wide): {drop.sum():,} rows flagged", flush=True)

    # per-symbol gaps: trailing-window + unbounded bars_since_high guard
    nsym_gap = 0
    for sym, g in pan.groupby("symbol", sort=False):
        sg = gap_intervals(sym)
        if not sg: continue
        nsym_gap += 1
        idx = g.index.to_numpy()
        m = window_mask(g["open_time"], sg)
        # unbounded bars_since_high: run reaches back across a gap-end g1 (variable length)
        if "bars_since_high" in g:
            back = g["open_time"] - pd.to_timedelta(g["bars_since_high"].fillna(0).to_numpy() * 5, unit="m")
            for g0, g1 in sg:
                m |= ((g["open_time"] > g1) & (back <= g1)).to_numpy()
        drop[idx] |= m
    n_drop = int(drop.sum())
    print(f"symbols with gaps: {nsym_gap}; TOTAL contaminated rows: {n_drop:,} ({100*n_drop/n0:.2f}%)", flush=True)

    clean = pan.loc[~drop].copy()
    # recompute cross-sectional rank + per-symbol target over CLEAN survivors
    clean["bars_since_high_xs_rank"] = clean.groupby("open_time")["bars_since_high"].rank(pct=True).astype("float32")
    try:
        import importlib.util
        _spec = importlib.util.spec_from_file_location("x6", REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
        x6 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(x6)
        clean = x6.build_target_z(clean)
        print("recomputed target_z over clean survivors", flush=True)
    except Exception as e:
        print(f"WARN: target_z recompute failed ({type(e).__name__}: {e}); keeping existing", flush=True)
    clean = clean.dropna(subset=["alpha_vs_btc_realized"]).reset_index(drop=True)
    clean.to_parquet(OUT)
    print(f"wrote {OUT.name}: {len(clean):,} rows (dropped {n0-len(clean):,} = {100*(n0-len(clean))/n0:.2f}% vs raw)", flush=True)
    print("GAPGUARDDONE")

if __name__ == "__main__":
    main()
