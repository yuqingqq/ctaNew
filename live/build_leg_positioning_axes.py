"""Q5 leg-level positioning features (SCREENING ONLY) — committed per RESEARCH_LOOP_20260707 Iter 6
pre-registration. Implements every blocking fix from the Q5 design review:

F1  TIME-BASED windows on a strict 5-min UTC grid (row-based windows are the retraction's failure
    mode: in-file gaps up to 1,598 days exist). Two-point changes use last-valid-snapshot <= target
    with a 2h staleness cap on BOTH endpoints; trailing means are calendar windows requiring >=80%
    of the 288 expected snapshots.
F2  OI sanity: zeros -> NaN; any 5-min |dlog OI| > 0.5 quarantines the symbol's OI features for the
    following 24h (outage/redenomination). Quarantine counts reported. Threshold fixed, no tuning.
F3  Leg aggregate = NaN if ANY picked name's feature is NaN (K=1/2 -> q_nan bucket downstream).
    Pool statistics require >=70% of pool names valid, else NaN.
F5  Verdict-bearing family = WITHIN-POOL PERCENTILE RANK of the pick (raw values retained as
    columns but not verdict-bearing). 6 axes: {long,short} x {oi_chg_24h, oi_chg_3d, taker_ls_24h}.
F9  Picks mirror live/v4_gate_model_test.py exactly (K_LONG=1 by pred_long, K_SHORT=2 by pred_base,
    clean books, pool = rows with fwd label — the pre-existing mild delisting-conditioning is noted,
    pool ranks computed over the SAME filtered pool). 30d listing-age guard on OI features.
F10 Identity assert: recomputed long_ret1d / short_ret3d leg means must match the stored
    V4_GATE_MODEL_DATASET columns bit-for-bit (hard stop on mismatch) before new columns merge.

Output: live/V4_GATE_DATASET_LEGPOS.parquet (existing dataset + new columns).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/home/yuqing/ctaNew")
CACHE = REPO / "data/ml/cache"
HOLD, K_LONG, K_SHORT = 6, 1, 2
STALE = pd.Timedelta("2h")
QUAR_THR, QUAR_H = 0.5, 288          # |dlog| threshold, quarantine horizon in 5-min slots (24h)
MIN_AGE = pd.Timedelta("30d")
BOOKS = {
    "oos_clean": ("hl_v4base_oos_clean", "hl_v4long_oos_clean"),
    "recent_clean": ("hl_tgt_res_base_clean", "hl_tgt_res_long_clean"),
}

def sym_features(sym, cycle_times):
    """Per-cycle {oi_chg_24h, oi_chg_3d, taker_ls_24h} for one symbol; NaN-safe per F1/F2."""
    f = CACHE / f"metrics_{sym}.parquet"
    if not f.exists(): return None
    m = pd.read_parquet(f, columns=["sum_open_interest", "sum_taker_long_short_vol_ratio"]).sort_index()
    m = m[~m.index.duplicated(keep="last")]
    idx = pd.to_datetime(m.index, utc=True, format="mixed")
    m.index = idx
    grid = pd.date_range(idx.min().ceil("5min"), idx.max(), freq="5min", tz="UTC")
    oi = m["sum_open_interest"].astype(float).replace(0, np.nan).reindex(grid)
    tls = m["sum_taker_long_short_vol_ratio"].astype(float).reindex(grid)
    # F2 quarantine on raw 5-min log steps
    dl = np.log(oi / oi.shift(1)).abs()
    bad = dl > QUAR_THR
    if bad.any():
        qmask = bad.rolling(QUAR_H, min_periods=1).max().astype(bool)
        oi = oi.where(~qmask)
    n_quar = int(bad.sum())
    # F1: staleness-capped snapshot lookup (ffill limited to 2h = 24 slots)
    oi_f = oi.ffill(limit=24)
    tls_cnt = tls.notna().rolling("24h").count()
    tls_24 = tls.rolling("24h").mean().where(tls_cnt >= 0.8 * 288)
    first_valid = oi.first_valid_index()
    if first_valid is None: return None
    age_ok = grid >= (first_valid + MIN_AGE)
    rows = {}
    ct = cycle_times[(cycle_times >= grid[0]) & (cycle_times <= grid[-1])]
    if len(ct) == 0: return None
    pos = grid.searchsorted(ct)          # cycle opens align to the 5-min grid
    pos = np.clip(pos, 0, len(grid) - 1)
    def at(series, offset_slots):
        p = pos - offset_slots
        v = np.full(len(ct), np.nan)
        ok = p >= 0
        v[ok] = series.to_numpy()[p[ok]]
        return v
    now, d24, d72 = at(oi_f, 0), at(oi_f, 288), at(oi_f, 288 * 3)
    out = pd.DataFrame({
        "symbol": sym, "open_time": ct,
        "oi_chg_24h": np.log(now / d24),
        "oi_chg_3d": np.log(now / d72),
        "taker_ls_24h": at(tls_24, 0),
    })
    out.loc[~age_ok[pos], ["oi_chg_24h", "oi_chg_3d", "taker_ls_24h"]] = np.nan
    return out, n_quar

def main():
    ds = pd.read_parquet(REPO / "live/V4_GATE_MODEL_DATASET.parquet")
    ds["open_time"] = pd.to_datetime(ds["open_time"], utc=True)
    cycle_times = pd.DatetimeIndex(sorted(ds["open_time"].unique()))
    # panel + picks (mirror v4_gate_model_test.py)
    pan = pd.read_parquet(REPO / "outputs/vBTC_features/panel_expanded_v0.parquet",
                          columns=["symbol", "open_time", "alpha_vs_btc_realized", "return_1d", "ret_3d"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    pan = pan.sort_values(["symbol", "open_time"])
    pan["fwd"] = pan.groupby("symbol")["alpha_vs_btc_realized"].transform(
        lambda s: s.shift(-1).rolling(HOLD).sum().shift(-(HOLD - 1))) * 1e4
    pan = pan.dropna(subset=["fwd"])
    syms = sorted(pan["symbol"].unique())
    feats, quar = [], {}
    for s in syms:
        r = sym_features(s, cycle_times)
        if r is not None:
            feats.append(r[0]); quar[s] = r[1]
    F = pd.concat(feats, ignore_index=True)
    print(f"features: {len(F)} rows, {F['symbol'].nunique()} syms; "
          f"quarantined 5-min events: total {sum(quar.values())}, worst {sorted(quar.items(), key=lambda x: -x[1])[:5]}")
    FEATS = ["oi_chg_24h", "oi_chg_3d", "taker_ls_24h"]
    rows = []
    for window, (bb, ll) in BOOKS.items():
        base = pd.read_parquet(REPO / f"live/state/convexity/{bb}/v0full_hl60.parquet",
                               columns=["symbol", "open_time", "pred"]).rename(columns={"pred": "pred_base"})
        long = pd.read_parquet(REPO / f"live/state/convexity/{ll}/v0full_hl60.parquet",
                               columns=["symbol", "open_time", "pred"]).rename(columns={"pred": "pred_long"})
        for df in (base, long): df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        d = base.merge(long, on=["symbol", "open_time"]).merge(pan, on=["symbol", "open_time"]) \
                .dropna(subset=["pred_base", "pred_long", "fwd"])
        d = d.merge(F, on=["symbol", "open_time"], how="left")
        for t, g in d.groupby("open_time"):
            if len(g) < K_LONG + K_SHORT: continue
            L = g.nlargest(K_LONG, "pred_long"); S = g.nsmallest(K_SHORT, "pred_base")
            row = {"window": window, "open_time": t,
                   "chk_long_ret1d": float(L["return_1d"].mean()), "chk_short_ret3d": float(S["ret_3d"].mean())}
            for f in FEATS:
                pool = g[f]
                valid_frac = pool.notna().mean()
                rank = pool.rank(pct=True) if valid_frac >= 0.7 else pd.Series(np.nan, index=g.index)
                for leg, sel in (("long", L), ("short", S)):
                    vals, rk = sel[f], rank.loc[sel.index]
                    row[f"{leg}_{f}"] = float(vals.mean()) if vals.notna().all() else np.nan     # F3
                    row[f"{leg}_{f}_rank"] = float(rk.mean()) if rk.notna().all() else np.nan    # F3+F5
            rows.append(row)
    Lp = pd.DataFrame(rows)
    out = ds.merge(Lp, on=["window", "open_time"], how="left")
    # F10 identity assert against stored leg columns
    for chk, ref in (("chk_long_ret1d", "long_ret1d"), ("chk_short_ret3d", "short_ret3d")):
        m = out.dropna(subset=[chk, ref])
        mx = (m[chk] - m[ref]).abs().max()
        print(f"identity {chk} vs {ref}: n={len(m)} max|diff|={mx:.2e}")
        if not (mx < 1e-9):
            raise SystemExit(f"IDENTITY ASSERT FAILED ({chk}): pick logic drifted — hard stop (F10)")
    out = out.drop(columns=["chk_long_ret1d", "chk_short_ret3d"])
    newc = [c for c in out.columns if c not in ds.columns]
    print("new columns:", newc)
    print("coverage:"); print(out[newc].notna().mean().round(3))
    out.to_parquet(REPO / "live/V4_GATE_DATASET_LEGPOS.parquet", index=False)
    print("wrote live/V4_GATE_DATASET_LEGPOS.parquet"); print("LEGPOSDONE")

if __name__ == "__main__":
    main()
