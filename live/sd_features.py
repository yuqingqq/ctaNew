"""Signal-diversity loop — build the literature-backed liquidity characteristics our V0 set lacks.

All computed on the 5m grid from the xs_feats cache (OHLCV), `.shift(1)` there, then sampled to the 4h
decision grid — the exact convention the existing panel uses (`incremental_panel._build_sym_window`), so the
new features are directly comparable to V0_LEAN and cannot see the label.

  turnover_vol_7d  7d std of the DAILY log-change in daily dollar volume        (IRFA 2026 top factor)
  spread_cs_7d     Corwin-Schultz (2012) high-low spread estimator, 7d mean      (IRFA 2026 top factor)
  amihud_7d        log mean of |5m return| / 5m dollar volume over 7d            (illiquidity)
  illiq_vol_7d     7d std of the daily Amihud measure                            (volatility of liquidity)
  log_dvol_7d      log trailing-7d mean daily dollar volume                      (size)

past_alpha_7d is built in the ablation script instead — it comes off the 4h panel's label column and needs a
full 4h-bar shift, not a 5m one.

Writes live/state/cost_loop/sd_chars.parquet (symbol, open_time, <features>).
Run: python3 -u -m live.sd_features [--workers 6]
"""
from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
CACHE = REPO / "data/ml/cache"
OUT = REPO / "live/state/cost_loop/sd_chars.parquet"
D1, D7 = 288, 288 * 7          # 5m bars in a day / week
START = pd.Timestamp("2023-01-01", tz="UTC")
FEATS = ["turnover_vol_7d", "spread_cs_7d", "amihud_7d", "illiq_vol_7d", "log_dvol_7d"]


def corwin_schultz(h: pd.Series, l: pd.Series) -> pd.Series:
    """Corwin & Schultz (2012) two-period high-low spread estimator, per bar-pair."""
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.log(h / l) ** 2 + (np.log(h.shift(-1) / l.shift(-1)) ** 2)
        hi2 = pd.concat([h, h.shift(-1)], axis=1).max(axis=1)
        lo2 = pd.concat([l, l.shift(-1)], axis=1).min(axis=1)
        g = np.log(hi2 / lo2) ** 2
        k = 3 - 2 * np.sqrt(2)
        a = (np.sqrt(2 * b) - np.sqrt(b)) / k - np.sqrt(g / k)
        s = 2 * (np.exp(a) - 1) / (1 + np.exp(a))
    return s.clip(lower=0)          # negative estimates are noise -> floor at 0 (standard practice)


def build_sym(sym: str) -> pd.DataFrame | None:
    fp = CACHE / f"xs_feats_{sym}.parquet"
    if not fp.exists():
        return None
    try:
        x = pd.read_parquet(fp, columns=["high", "low", "close", "volume"])
    except Exception:
        return None
    if x.empty:
        return None
    x.index = pd.DatetimeIndex(x.index).tz_convert("UTC")
    x = x[x.index >= START - pd.Timedelta(days=30)].sort_index()
    if len(x) < D7:
        return None
    close = x["close"].astype(float).replace(0, np.nan)
    dv = (close * x["volume"].astype(float)).replace(0, np.nan)         # 5m dollar volume
    ret = close.pct_change()

    dv_1d = dv.rolling(D1, min_periods=D1 // 2).sum()
    turnover_vol = np.log(dv_1d).diff(D1).rolling(D7, min_periods=D7 // 2).std()

    spread = corwin_schultz(x["high"].astype(float), x["low"].astype(float)) \
        .rolling(D7, min_periods=D7 // 2).mean()

    amihud_5m = (ret.abs() / dv).replace([np.inf, -np.inf], np.nan)
    amihud = np.log(amihud_5m.rolling(D7, min_periods=D7 // 2).mean())
    amihud_1d = amihud_5m.rolling(D1, min_periods=D1 // 2).mean()
    illiq_vol = np.log1p(amihud_1d).rolling(D7, min_periods=D7 // 2).std()

    log_dvol = np.log(dv_1d.rolling(D7, min_periods=D7 // 2).mean())

    F = pd.DataFrame({
        "turnover_vol_7d": turnover_vol, "spread_cs_7d": spread, "amihud_7d": amihud,
        "illiq_vol_7d": illiq_vol, "log_dvol_7d": log_dvol,
    }).shift(1)                                                          # PIT: no same-bar information
    F = F[(F.index.hour % 4 == 0) & (F.index.minute == 0) & (F.index >= START)]
    F = F.replace([np.inf, -np.inf], np.nan)
    if F.dropna(how="all").empty:
        return None
    F = F.reset_index().rename(columns={"index": "open_time"})
    if "open_time" not in F.columns:
        F = F.rename(columns={F.columns[0]: "open_time"})
    F["symbol"] = sym
    return F


def _w(s):
    try:
        return build_sym(s)
    except Exception as e:
        print(f"  {s} ERR {str(e)[:50]}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    syms = sorted(Path(f).stem.replace("xs_feats_", "")
                  for f in glob.glob(str(CACHE / "xs_feats_*.parquet")))
    print(f"building {len(FEATS)} characteristics for {len(syms)} symbols", flush=True)
    with mp.Pool(a.workers) as pool:
        parts = pool.map(_w, syms)
    parts = [p for p in parts if p is not None and len(p)]
    D = pd.concat(parts, ignore_index=True)
    D["open_time"] = pd.to_datetime(D["open_time"], utc=True)
    D = D[["symbol", "open_time"] + FEATS].sort_values(["symbol", "open_time"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    D.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(D):,} rows, {D.symbol.nunique()} syms, "
          f"{D.open_time.min().date()} -> {D.open_time.max().date()}", flush=True)
    print("\ncoverage (non-null share):", flush=True)
    for c in FEATS:
        print(f"  {c:<18}{D[c].notna().mean()*100:.1f}%", flush=True)
    print("SDFEATDONE", flush=True)


if __name__ == "__main__":
    main()
