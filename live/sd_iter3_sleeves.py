"""Signal-diversity loop — iteration 3 (S5): how many uncorrelated sleeves do we actually have?

Five iterations have shown no single signal/characteristic/construction/objective moves this book. The
literature route to a 6-Sharpe book is IR = r*sqrt(N) over N uncorrelated sleeves (at the published SOTA
r~2.3, N~7). So the decision-relevant question is empirical: how many sleeves do we have, how correlated are
they, and what does the combination support?

Sleeves (all on the held-out window 2025-01..2026-06, top-40 PIT-ADV universe, quintile L/S on per-name
BTC-residual returns, net of the calibrated per-symbol cost):
  xs_reversal   the incumbent per-symbol-Ridge prediction         (the current book)
  int_momentum  14d trailing return skipping the last 2d           (documented different-root sleeve)
  carry         cross-sectional funding rate, short high / long low
  low_vol       slow trailing-rvol sort                            (negative control: expected ~ sleeve 1)
  ts_trend      30d trend on the equal-weight basket, directional  (diversifier, not cross-sectional)

Gates: G1 >=2 sleeves with standalone net Sharpe CI>0; G2 mean |pairwise corr| of those < 0.3;
G3 equal-RISK combination (weights fixed a priori, NOT optimized) beats the best single sleeve, paired CI>0.
Falsifier: G1 fails -> the sqrt(N) route is not open on owned data.
Run: python3 -u -m live.sd_iter3_sleeves
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from live.cost_loop_harness import (
    CACHE, ERAS, REPO, block_ci, build_panel, get_preds, paired_block_ci, pit_adv, sharpe, tag_ci,
)
from live.build_alpha_beta_decomp import FULL
from live.cl_iter4_capacity import build, cost_tiers
from live.mc_oi_universe import topn, N as NTOP

HO0, HO1 = pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC")
PYR = 6 * 365.0


def load_funding() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(REPO / "data/ml/cache/funding_*.parquet")):
        sym = Path(f).stem.replace("funding_", "")
        try:
            d = pd.read_parquet(f)
        except Exception:
            continue
        tc = "calc_time" if "calc_time" in d.columns else ("open_time" if "open_time" in d.columns else None)
        if tc is None or "funding_rate" not in d.columns:
            continue
        d[tc] = pd.to_datetime(d[tc], utc=True)
        s = d.set_index(tc)["funding_rate"].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        s = s.resample("4h").ffill().shift(2)          # PIT: 2 bars = 8h settled
        rows.append(pd.DataFrame({"symbol": sym, "open_time": s.index, "funding": s.values}))
    F = pd.concat(rows, ignore_index=True).dropna()
    F["open_time"] = pd.to_datetime(F["open_time"], utc=True)
    return F


def sleeve_net(d: pd.DataFrame, sig: str, cost, ctl="band") -> pd.Series:
    v = d.dropna(subset=[sig, "alpha_A"]).copy()
    if "pred" in v.columns and sig != "pred":
        v = v.drop(columns=["pred"])                 # avoid a duplicate 'pred' when renaming the sleeve in
    v = v.rename(columns={sig: "pred"})
    if v.empty:
        return pd.Series(dtype=float)
    W, A = build(v, ctl)
    g = (W * A).sum(axis=0)
    dW = W.diff(axis=1).abs()
    c, med = cost
    kv = pd.Series([c.get(s, med) for s in W.index], index=W.index)
    return (g - 0.25 * dW.mul(kv, axis=0).sum(axis=0) / 1e4).iloc[1:]


def main():
    CT = cost_tiers(); cost = CT["cost_10k"]
    PAN = build_panel()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    P = pd.concat([get_preds(e) for e in ERAS], ignore_index=True).drop_duplicates(
        ["symbol", "open_time"]).sort_values(["symbol", "open_time"])
    lab = PAN[["symbol", "open_time", "alpha_vs_btc_realized", "rvol_7d"]].rename(
        columns={"alpha_vs_btc_realized": "alpha_A"})
    P = P.drop(columns=[c for c in ("alpha_A", "return_pct") if c in P.columns]).merge(
        lab, on=["symbol", "open_time"], how="left").merge(RP, on=["symbol", "open_time"], how="left")
    P = P.merge(load_funding(), on=["symbol", "open_time"], how="left")
    A = pit_adv(); P["date"] = P["open_time"].dt.floor("1D")
    P = P.merge(A, on=["symbol", "date"], how="left").sort_values(["symbol", "open_time"])

    # ---- sleeve signals (all PIT) ----
    r = P.groupby("symbol")["return_pct"]
    P["ret_14d"] = r.transform(lambda s: s.shift(1).rolling(84).sum())
    P["ret_2d"] = r.transform(lambda s: s.shift(1).rolling(12).sum())
    P["int_momentum"] = P["ret_14d"] - P["ret_2d"]                       # 14d skip-recent
    P["xs_reversal"] = P["pred"]
    P["carry"] = -P["funding"]                                           # long low funding / short high
    P["vrank"] = P.groupby("open_time")["rvol_7d"].rank(pct=True)
    P["low_vol"] = -P.groupby("symbol")["vrank"].transform(
        lambda s: s.shift(1).expanding(min_periods=30).mean())

    ho = topn(P[(P.open_time >= HO0) & (P.open_time < HO1)].dropna(subset=["tadv"]), "tadv", NTOP)
    print(f"held-out {ho.open_time.nunique()} bars, {ho.symbol.nunique()} syms\n", flush=True)

    SLEEVES = ["xs_reversal", "int_momentum", "carry", "low_vol"]
    ser = {}
    for s in SLEEVES:
        n = sleeve_net(ho, s, cost)
        if len(n) > 100:
            ser[s] = n

    # ts_trend: directional basket trend, not cross-sectional -> built separately
    mkt = ho.groupby("open_time")["alpha_A"].mean()
    bas = ho.groupby("open_time")["return_pct"].mean()
    trend = np.sign(bas.rolling(180, min_periods=90).sum().shift(1))
    ser["ts_trend"] = (trend * bas).dropna()

    print("============ A2 — standalone sleeve performance (held-out, net@10k) ============", flush=True)
    print(f"  {'sleeve':<15}{'net Sharpe [7d-block CI]':<32}{'bars':<8}", flush=True)
    stats = {}
    for s, n in ser.items():
        lo, hi = block_ci(n.to_numpy())
        stats[s] = (sharpe(n), lo, hi)
        print(f"  {s:<15}{f'{sharpe(n):+.2f} [{lo:+.2f},{hi:+.2f}] {tag_ci(lo, hi)}':<32}{len(n):<8}",
              flush=True)

    print("\n============ G2 — pairwise correlation of sleeve NET returns ============", flush=True)
    D = pd.DataFrame(ser).dropna()
    C = D.corr()
    print(C.round(3).to_string(), flush=True)

    pos = [s for s in ser if stats[s][1] > 0]
    print(f"\n  sleeves with CI>0: {pos if pos else 'NONE'}", flush=True)
    g1 = len(pos) >= 2
    if len(pos) >= 2:
        sub = C.loc[pos, pos].to_numpy()
        iu = np.triu_indices(len(pos), 1)
        mc = float(np.mean(np.abs(sub[iu])))
        print(f"  mean |pairwise corr| among them: {mc:.3f}  -> G2 {'PASS' if mc < 0.3 else 'FAIL'}",
              flush=True)
    else:
        mc = np.nan
        print("  G2 not evaluable (<2 positive sleeves)", flush=True)

    print("\n============ G3 — equal-RISK combination (weights fixed a priori) ============", flush=True)
    use = pos if len(pos) >= 2 else list(ser)
    Dn = D[use]
    wts = (1.0 / Dn.std()).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    wts = wts / wts.sum()
    comb = (Dn * wts).sum(axis=1)
    lo, hi = block_ci(comb.to_numpy())
    best = max(use, key=lambda s: stats[s][0])
    print(f"  combination of {use}", flush=True)
    print(f"  combined net Sharpe {sharpe(comb):+.2f} [{lo:+.2f},{hi:+.2f}] {tag_ci(lo, hi)}", flush=True)
    print(f"  best single sleeve  {best} {stats[best][0]:+.2f}", flush=True)
    dd, dlo, dhi = paired_block_ci(D[best].to_numpy(), comb.to_numpy())
    print(f"  Δ(combo − best single) {dd:+.2f} [{dlo:+.2f},{dhi:+.2f}] {tag_ci(dlo, dhi)}  "
          f"-> G3 {'PASS' if dlo > 0 else 'FAIL'}", flush=True)

    print("\n============ what the sqrt(N) arithmetic implies ============", flush=True)
    for r_ in (1.0, 1.5, 2.0, 2.3, 3.0):
        print(f"  sleeves of net IR {r_:.1f} each, uncorrelated -> need N = {(6.33/r_)**2:5.1f} to reach 6.33",
              flush=True)
    print(f"\n  G1 (>=2 sleeves CI>0): {'PASS' if g1 else 'FAIL'}", flush=True)
    print("\nSDITER3DONE", flush=True)


if __name__ == "__main__":
    main()
