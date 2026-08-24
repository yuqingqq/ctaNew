"""Shared harness for the cost/turnover loop (live/COST_TURNOVER_LOOP.md).

Everything the iterations need, computed once and cached:
  - get_preds(era)      walk-forward per-symbol RidgeCV preds (the incumbent pipeline) + labels + fwd returns
  - pit_adv()           PIT trailing-30d ADV (shift-1) for the big-names universe
  - cost_map()          per-symbol calibrated round-trip-per-side cost in bps (live/state/v3loop/persym_cost_cal.csv)
  - book()              quintile L/S weights from any signal column
  - net_series()        gross / turnover / market / era-locked-hedged alpha / net at flat or per-symbol cost
  - block_ci(), paired_block_ci()   7d-block bootstrap on Sharpe levels and paired deltas

Conventions match the existing cycle-1..6 scripts (`build_net_result.py`, `build_deployable_stack.py`) so
numbers are comparable with docs/CONSTRUCTION_COST_IMPROVEMENTS_2026-07-29.md:
  W has long leg summing +1 and short leg −1; book return = sum(W*R); turnover = 0.25*sum|dW| per bar
  (1.0 = one full flip of the two-sided book); net = hedged_gross − turnover*cost_bps/1e4.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS   # noqa: E402
from live.build_alpha_beta_decomp import gen_pred, FULL                        # noqa: E402

PYR = 6 * 365.0                       # 4h bars per year
BLOCK = 42                            # 7 days of 4h bars
CACHE = REPO / "live/state/cost_loop"
CACHE.mkdir(parents=True, exist_ok=True)
CUTS = {"RECENT": RECENT_CUTS, "OOS": OOS_CUTS}
ERAS = ("RECENT", "OOS")
_RNG = np.random.default_rng(20260806)


# ---------------------------------------------------------------- predictions
def get_preds(era: str, feats=None, tag: str = "v0lean") -> pd.DataFrame:
    """Walk-forward per-symbol RidgeCV preds for `era`, cached. Columns:
    symbol, open_time, pred, alpha_A (4h BTC-residual label), return_pct (4h raw fwd return)."""
    fp = CACHE / f"preds_{tag}_{era}.parquet"
    if fp.exists():
        d = pd.read_parquet(fp)
        d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
        return d
    PAN = build_panel()
    p = gen_pred(PAN, list(feats or V0), CUTS[era])
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    lab = PAN[["symbol", "open_time", "alpha_vs_btc_realized"]].rename(
        columns={"alpha_vs_btc_realized": "alpha_A"})
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    p = p.merge(lab, on=["symbol", "open_time"], how="inner").merge(
        RP, on=["symbol", "open_time"], how="inner")
    p = p.dropna(subset=["pred", "alpha_A", "return_pct"]).sort_values(["symbol", "open_time"])
    p.to_parquet(fp, index=False)
    return p


# ---------------------------------------------------------------- universe / cost
def pit_adv() -> pd.DataFrame:
    """PIT trailing-30d mean daily dollar volume, shift-1. Columns: symbol, date, tadv."""
    fp = CACHE / "pit_adv.parquet"
    if fp.exists():
        d = pd.read_parquet(fp); d["date"] = pd.to_datetime(d["date"], utc=True); return d
    frames = []
    for f in glob.glob(str(REPO / "data/ml/cache/flow_*.parquet")):
        sym = Path(f).stem.replace("flow_", "")
        try:
            d = pd.read_parquet(f, columns=["total_volume", "vwap"])
            if not isinstance(d.index, pd.DatetimeIndex):
                continue
            dv = (d["total_volume"] * d["vwap"]).sort_index()
            t = dv.resample("1D").sum().rolling(30, min_periods=10).mean().shift(1)
            frames.append(pd.DataFrame({"symbol": sym, "date": t.index, "tadv": t.values}))
        except Exception:
            pass
    A = pd.concat(frames, ignore_index=True)
    A["date"] = pd.to_datetime(A["date"], utc=True)
    A = A.dropna(subset=["tadv"])
    A.to_parquet(fp, index=False)
    return A


def restrict_topn(d: pd.DataFrame, n: int) -> pd.DataFrame:
    """Keep the top-n symbols by PIT trailing ADV at each bar (n>=999 => no restriction)."""
    if n >= 999:
        return d.copy()
    A = pit_adv()
    x = d.copy()
    x["date"] = x["open_time"].dt.floor("1D")
    x = x.merge(A, on=["symbol", "date"], how="left").dropna(subset=["tadv"])
    x["advrank"] = x.groupby("open_time")["tadv"].rank(ascending=False, method="first")
    return x[x["advrank"] <= n].drop(columns=["advrank"])


def cost_map() -> tuple[pd.Series, float]:
    """Per-symbol calibrated cost in bps per fill (depth model, $10k clip) + median fallback."""
    c = pd.read_csv(REPO / "live/state/v3loop/persym_cost_cal.csv").set_index("symbol")["cost_10k"]
    return c, float(c.median())


# ---------------------------------------------------------------- book construction
def book(d: pd.DataFrame, sig: str, q: float = 0.2):
    """Quintile L/S weights from signal `sig`. Returns (W, R, mask) as names x bars frames."""
    x = d.dropna(subset=[sig]).copy()
    x["rk"] = x.groupby("open_time")[sig].rank(pct=True)
    x["pos"] = np.where(x["rk"] >= 1 - q, 1.0, np.where(x["rk"] <= q, -1.0, 0.0))
    R = x.pivot_table(index="symbol", columns="open_time", values="return_pct")
    mask = R.notna().astype(float)
    R = R.fillna(0.0)
    P = x.pivot_table(index="symbol", columns="open_time", values="pos", fill_value=0.0).reindex_like(R)
    pos = P.clip(lower=0); neg = P.clip(upper=0)
    W = pos.div(pos.sum().replace(0, np.nan), axis=1).fillna(0.0) \
        + neg.div(neg.sum().abs().replace(0, np.nan), axis=1).fillna(0.0)
    return W, R, mask


def smooth_ewma(W: pd.DataFrame, mask: pd.DataFrame, lam: float) -> pd.DataFrame:
    """EWMA weight smoothing along time then re-normalize per side (build_turnover_opt convention)."""
    if lam <= 0:
        return W
    S = W.T.ewm(alpha=1 - lam, adjust=False).mean().T * mask
    pos = S.clip(lower=0); neg = S.clip(upper=0)
    return pos.div(pos.sum().replace(0, np.nan), axis=1).fillna(0.0) \
        + neg.div(neg.sum().abs().replace(0, np.nan), axis=1).fillna(0.0)


def net_series(W: pd.DataFrame, R: pd.DataFrame, mask: pd.DataFrame, persym_cost=None) -> pd.DataFrame:
    """Per-bar gross, turnover, market return, and per-symbol-cost charge. First bar dropped (no ΔW)."""
    gross = (W * R).sum(axis=0)
    dW = W.diff(axis=1).abs()
    turn = 0.25 * dW.sum(axis=0)
    mkt = (R * mask).sum(axis=0) / mask.sum(axis=0).replace(0, np.nan)
    out = pd.concat([gross.rename("g"), turn.rename("t"), mkt.rename("m")], axis=1)
    if persym_cost is not None:
        c, med = persym_cost
        cvec = pd.Series([c.get(s, med) for s in W.index], index=W.index)
        out["c_ps"] = 0.25 * dW.mul(cvec, axis=0).sum(axis=0) / 1e4
    return out.iloc[1:].dropna(subset=["g", "t", "m"])


def hedged(j: pd.DataFrame, beta: float) -> pd.Series:
    return j["g"] - beta * j["m"]


def hedge_beta(j: pd.DataFrame) -> float:
    return float(np.polyfit(j["m"], j["g"], 1)[0])


# ---------------------------------------------------------------- statistics
def sharpe(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(x.mean() / x.std() * np.sqrt(PYR)) if len(x) > 2 and x.std() > 0 else np.nan


def maxdd(x) -> float:
    eq = np.cumsum(np.asarray(x, dtype=float))
    return float((eq - np.maximum.accumulate(eq)).min())


def _block_idx(n, block, rng):
    nblk = int(np.ceil(n / block))
    st = rng.integers(0, max(n - block + 1, 1), nblk)
    return np.concatenate([np.arange(s, s + block) for s in st])[:n]


def block_ci(x, block: int = BLOCK, nb: int = 3000, seed: int = 0):
    """7d-block bootstrap CI on the Sharpe of a return series."""
    rng = np.random.default_rng(seed or 20260806)
    a = np.asarray(x, dtype=float); n = len(a)
    d = np.array([sharpe(a[_block_idx(n, block, rng)]) for _ in range(nb)])
    return float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))


def paired_block_ci(a, b, block: int = BLOCK, nb: int = 3000, seed: int = 0):
    """CI on Sharpe(b) − Sharpe(a) with the SAME resampled blocks applied to both (paired)."""
    rng = np.random.default_rng(seed or 20260806)
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    d = np.empty(nb)
    for i in range(nb):
        idx = _block_idx(n, block, rng)
        d[i] = sharpe(b[idx]) - sharpe(a[idx])
    return float(sharpe(b) - sharpe(a)), float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))


def tag_ci(lo, hi):
    return "SIG" if lo > 0 else ("neg" if hi < 0 else "spans0")


def add_slow_signals(p: pd.DataFrame, freeze_days: int = 90) -> pd.DataFrame:
    """Attach the persistent-tilt variants of `pred` (all PIT):
      stat      per-symbol shift-1 expanding mean of own past preds
      stat_ew   same but 30d-halflife EWMA (slow-dynamic control)
      stat_froz tilt frozen after the first `freeze_days` of the era (A1 staleness control)
      dyn       pred − stat
    """
    p = p.sort_values(["symbol", "open_time"]).copy()
    g = p.groupby("symbol")["pred"]
    p["stat"] = g.transform(lambda s: s.shift(1).expanding(min_periods=30).mean())
    p["stat_ew"] = g.transform(lambda s: s.shift(1).ewm(halflife=180, min_periods=30).mean())
    t0 = p["open_time"].min() + pd.Timedelta(days=freeze_days)
    frz = p[p["open_time"] <= t0].groupby("symbol")["pred"].mean()
    p["stat_froz"] = p["symbol"].map(frz).where(p["open_time"] > t0)   # only valid out of its estimation window
    p["dyn"] = p["pred"] - p["stat"]
    return p
