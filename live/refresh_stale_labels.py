"""Audit #1b remediation: refresh STALE forward labels. panel_expanded_v0 was built before some 5m klines
arrived, so ~180 rows (notably all 174 at 2026-06-04 20:00) carry NaN alpha even though the +4h bar EXISTS in
current klines -> the label is recoverable and the (valid, losing) cycle should be TRADED, not silently excluded.
Recompute the GRID-SAFE forward return + residual alpha from CURRENT 5m klines for every NaN-alpha row whose +4h
grid bar now exists; leave genuinely-gap rows (e.g. the 2025-02-28 BTC 22d gap, +4h bar truly missing) NaN.
Writes panel_expanded_v0_relabeled.parquet (gap_guard_panel.py then consumes it via GAP_SRC).
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); KD = REPO / "data/ml/test/parquet/klines"; HORIZON = 48
SRC = REPO / "outputs/vBTC_features/panel_expanded_v0.parquet"
OUT = REPO / "outputs/vBTC_features/panel_expanded_v0_relabeled.parquet"

def load_closes(sym):
    fs = sorted(glob.glob(str(KD / sym / "5m" / "*.parquet")))
    if not fs: return None
    df = pd.concat([pd.read_parquet(f, columns=["open_time", "close"]) for f in fs], ignore_index=True)
    df = df.drop_duplicates("open_time").sort_values("open_time")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df.set_index("open_time")["close"].astype(float)

def target_alpha(my_close, btc_close):   # GRID-SAFE (matches X70 fix): returns+beta+forward on a complete 5m grid
    ci = my_close.index.intersection(btc_close.index)
    if len(ci) == 0: return pd.Series(dtype=float), pd.Series(dtype=float)
    full = pd.date_range(ci.min(), ci.max(), freq="5min", tz="UTC")
    mc = my_close.reindex(full); bc = btc_close.reindex(full)
    my_ret = np.log(mc / mc.shift(1)); btc_ret = np.log(bc / bc.shift(1))
    cov = my_ret.rolling(288, min_periods=72).cov(btc_ret); var = btc_ret.rolling(288, min_periods=72).var()
    beta = (cov / var.replace(0, np.nan)).shift(1)
    my_fwd = (mc.shift(-HORIZON) / mc - 1); btc_fwd = (bc.shift(-HORIZON) / bc - 1)
    alpha = (my_fwd - beta * btc_fwd)
    return alpha.reindex(ci), my_fwd.reindex(ci)

def main():
    pan = pd.read_parquet(SRC); pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    stale = pan[pan["alpha_vs_btc_realized"].isna()]
    print(f"NaN-alpha rows: {len(stale)} across {stale.symbol.nunique()} syms; dates: "
          f"{dict(stale.open_time.dt.date.value_counts().head(4))}", flush=True)
    btc = load_closes("BTCUSDT"); n_fix = 0; n_still = 0
    for sym, g in stale.groupby("symbol"):
        mc = load_closes(sym)
        if mc is None: n_still += len(g); continue
        alpha, fwd = target_alpha(mc, btc)
        a = alpha.reindex(g["open_time"]).to_numpy(); r = fwd.reindex(g["open_time"]).to_numpy()
        idx = g.index.to_numpy(); ok = np.isfinite(a)
        pan.loc[idx[ok], "alpha_vs_btc_realized"] = a[ok].astype(pan["alpha_vs_btc_realized"].dtype)
        pan.loc[idx[ok], "return_pct"] = r[ok].astype(pan["return_pct"].dtype)
        n_fix += int(ok.sum()); n_still += int((~ok).sum())
    print(f"refreshed {n_fix} stale labels from current klines; {n_still} stay NaN (genuine gap / no +4h bar)", flush=True)
    # sanity: the 2026-06-04 20:00 cycle should now be valid
    e = pan[pan.open_time == pd.Timestamp("2026-06-04 20:00", tz="UTC")]
    print(f"  2026-06-04 20:00: {len(e)} rows, valid alpha now {int(e['alpha_vs_btc_realized'].notna().sum())}", flush=True)
    pan.to_parquet(OUT)
    print(f"wrote {OUT.name} ({len(pan):,} rows)")
    print("RELABELDONE")

if __name__ == "__main__":
    main()
