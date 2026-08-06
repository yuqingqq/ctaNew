"""Shared harness for the ORTHOGONAL-DATA loop (charter: ORTHOGONAL_DATA_LOOP.md).

Genuinely-orthogonal (NOT return-derived) Binance futures metrics -> 4h PIT features:
  OI (sum_open_interest_value), top-trader long/short (positions & accounts), global account long/short,
  taker buy/sell volume ratio. Coverage verified BOTH eras (165/176 syms >500 bars each; back to 2021).

Provides:
  build_metrics_features()      per-symbol 5min metrics -> 4h PIT features (merge_asof, strictly backward)
  build_panel_with_metrics()    strategy panel (build_panel) + alpha + vol/reversal controls + metrics feats
  screen(d, cands, controls)    raw IC + orthogonalized (residual-vs-controls) IC, both eras, day-clustered CI
  CANDS / CONTROLS              default feature & control lists

Discipline mirrors the OB/flow loops: both-era (OOS<2025-10-01<=RECENT), day-clustered bootstrap CI, honest
nulls. The definitive test is INCREMENTAL rank-IC through the real per-symbol RidgeCV pipeline (gen from
v0_feature_ablation on V0 + candidate); the screen here is fast triage.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from live.v0_feature_ablation import build_panel, RECENT_CUTS, OOS_CUTS  # noqa: F401

REPO = Path("/home/yuqing/ctaNew")
CACHE = REPO / "data/ml/cache"
FEATS_CACHE = CACHE / "orthogonal_metrics_feats.parquet"
CUT = pd.Timestamp("2025-10-01", tz="UTC")
TGT = "alpha_vs_btc_realized"
rng = np.random.default_rng(11)

CANDS = ["oi_chg_1d", "oi_chg_3d", "tt_pos_ls", "tt_pos_chg_1d", "gl_acc_ls", "smart_dumb",
         "taker_ls", "taker_chg_1d"]
CONTROLS = ["rvol_7d", "atr_pct", "return_1d", "ret_3d"]


def _sym_feats(f):
    m = pd.read_parquet(f).sort_index()
    if not isinstance(m.index, pd.DatetimeIndex):
        return None
    m = m[~m.index.duplicated(keep="last")]
    grid = pd.date_range(m.index.min().ceil("4h"), m.index.max().floor("4h"), freq="4h", tz="UTC")
    if len(grid) < 50:
        return None
    src = m.reset_index().rename(columns={m.index.name or "index": "ct"})
    src["ct"] = pd.to_datetime(src["ct"], utc=True)
    a = pd.merge_asof(pd.DataFrame({"open_time": grid}), src, left_on="open_time", right_on="ct",
                      direction="backward", tolerance=pd.Timedelta("10min")).set_index("open_time")
    oi = a["sum_open_interest_value"].replace(0, np.nan)
    ttp = a["sum_toptrader_long_short_ratio"].replace(0, np.nan)
    gla = a["count_long_short_ratio"].replace(0, np.nan)
    tk = a["sum_taker_long_short_vol_ratio"].replace(0, np.nan)
    out = pd.DataFrame(index=grid)
    out["oi_chg_1d"] = oi.pct_change(6)
    out["oi_chg_3d"] = oi.pct_change(18)
    out["tt_pos_ls"] = np.log(ttp)
    out["tt_pos_chg_1d"] = np.log(ttp).diff(6)
    out["gl_acc_ls"] = np.log(gla)
    out["smart_dumb"] = np.log(ttp) - np.log(gla)
    out["taker_ls"] = np.log(tk)
    out["taker_chg_1d"] = np.log(tk).diff(6)
    # --- expanded (iter 5) ---
    out["tt_pos_chg_3d"] = np.log(ttp).diff(18)
    out["gl_acc_chg_1d"] = np.log(gla).diff(6)
    out["smart_dumb_chg_1d"] = (np.log(ttp) - np.log(gla)).diff(6)
    out["taker_chg_3d"] = np.log(tk).diff(18)
    lo = np.log(oi)
    out["oi_z"] = (lo - lo.rolling(30).mean()) / lo.rolling(30).std()
    out["symbol"] = f.split("/")[-1].replace("metrics_", "").replace(".parquet", "")
    return out.reset_index().rename(columns={"index": "open_time"})


def build_metrics_features(rebuild=False):
    if FEATS_CACHE.exists() and not rebuild:
        return pd.read_parquet(FEATS_CACHE)
    frames = []
    for f in sorted(glob.glob(str(CACHE / "metrics_*.parquet"))):
        try:
            r = _sym_feats(f)
            if r is not None:
                frames.append(r)
        except Exception:
            pass
    df = pd.concat(frames, ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df.to_parquet(FEATS_CACHE)
    return df


def build_panel_with_metrics(rebuild=False):
    from live.build_alpha_beta_decomp import FULL
    PAN = build_panel()
    have = set(PAN.columns)
    need = [c for c in CONTROLS if c not in have]
    if need:
        extra = pd.read_parquet(FULL, columns=["symbol", "open_time"] + need)
        extra["open_time"] = pd.to_datetime(extra["open_time"], utc=True)
        PAN = PAN.merge(extra, on=["symbol", "open_time"], how="left")
    mf = build_metrics_features(rebuild=rebuild)
    return PAN.merge(mf, on=["symbol", "open_time"], how="left")


def _ic_by_era(per):
    per = per.dropna()
    per["day"] = per.index.floor("1D")
    out = {}
    for era, mask in (("OOS", per.index < CUT), ("RECENT", per.index >= CUT)):
        s = per[mask]
        if len(s) < 20:
            out[era] = (np.nan, np.nan, np.nan, 0)
            continue
        gg = [x["ic"].to_numpy() for _, x in s.groupby("day")]
        boot = [np.concatenate([gg[k] for k in rng.integers(0, len(gg), len(gg))]).mean() for _ in range(2000)]
        out[era] = (float(s["ic"].mean()), *np.percentile(boot, [2.5, 97.5]), len(s))
    return out


def screen(d, cands, controls=CONTROLS, tgt=TGT):
    """Per-bar raw IC and orthogonalized (residual-vs-controls) IC of each candidate vs forward alpha."""
    ctrl = [c for c in controls if c in d.columns]
    base = d.dropna(subset=ctrl + [tgt]).copy()
    res = {}
    for c in cands:
        sub = base.dropna(subset=[c])
        raw, orth = [], []
        for t, g in sub.groupby("open_time"):
            if len(g) < 12:
                continue
            y = g[tgt].to_numpy()
            X = np.column_stack([np.ones(len(g))] + [g[k].to_numpy() for k in ctrl])
            xc = g[c].to_numpy()
            beta, *_ = np.linalg.lstsq(X, xc, rcond=None)
            rr = xc - X @ beta
            raw.append((t, spearmanr(xc, y).correlation))
            orth.append((t, spearmanr(rr, y).correlation))
        rw = pd.DataFrame(raw, columns=["t", "ic"]).set_index("t")
        ow = pd.DataFrame(orth, columns=["t", "ic"]).set_index("t")
        res[c] = {"raw": _ic_by_era(rw), "orth": _ic_by_era(ow)}
    return res


def _fmt(v):
    m, lo, hi, n = v
    if not np.isfinite(m):
        return f"{'n/a':>28}"
    star = "*" if (lo > 0 or hi < 0) else " "
    return f"{m:+.4f} [{lo:+.4f},{hi:+.4f}]{star}"


def main():
    d = build_panel_with_metrics()
    cov = d.dropna(subset=CANDS)["symbol"].nunique()
    print(f"panel+metrics: {len(d):,} rows | {d.symbol.nunique()} syms | {cov} syms with all metrics feats\n", flush=True)
    r = screen(d, CANDS)
    print(f"{'candidate':<15}{'RAW OOS':<30}{'RAW RECENT':<30}", flush=True)
    for c in CANDS:
        print(f"  {c:<13}{_fmt(r[c]['raw']['OOS']):<30}{_fmt(r[c]['raw']['RECENT']):<30}", flush=True)
    print(f"\n{'candidate':<15}{'ORTH OOS (vs vol+rev)':<30}{'ORTH RECENT':<30}   (* = CI excludes 0)", flush=True)
    for c in CANDS:
        print(f"  {c:<13}{_fmt(r[c]['orth']['OOS']):<30}{_fmt(r[c]['orth']['RECENT']):<30}", flush=True)
    print("\nORTHSCREENDONE", flush=True)


if __name__ == "__main__":
    main()
