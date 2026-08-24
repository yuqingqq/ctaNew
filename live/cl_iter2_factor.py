"""Cost/turnover loop — iteration 2 (D7/D4).

H2: the incumbent 175-model per-symbol-Ridge stack is reproducible by ONE slow low-vol factor, per-name
beta-neutral, at a fraction of the operational surface.

Runs in falsification order:
  PART A  artifact control (G2/A2/A3) — is the low-vol IC advantage a property of the RESIDUAL LABEL rather
          than of the market? IC(rvol_slow, alpha_A) split by BTC forward-return tercile, both eras, both
          universes; plus IC vs RAW forward return. A beta-estimation artifact flips sign with BTC direction.
  PART B  harness fix (G1/A1) — rebuild the INCUMBENT book on PER-NAME BTC-residual returns (what the strategy
          farms) instead of raw returns + one book-level beta. Must still reproduce OOS/top40 gross +2.2..+2.6.
  PART C  the test (G3/G4) — rvol factor book vs incumbent book on residual returns, net of per-symbol
          calibrated cost, both eras, 7d-block CI on levels and PAIRED CI on the delta.

Run: python3 -u -m live.cl_iter2_factor
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from live.cost_loop_harness import (
    ERAS, CACHE, REPO, add_slow_signals, block_ci, build_panel, cost_map, get_preds,
    maxdd, paired_block_ci, restrict_topn, sharpe, tag_ci,
)

RNG = np.random.default_rng(11)
HORIZON = 48          # 5m bars in the 4h label
SIGS = ["fast", "stat", "rvol_slow"]


def btc_fwd_4h() -> pd.Series:
    """Exact 4h forward return of BTC on the decision grid (same construction as the label's btc_fwd)."""
    spec = importlib.util.spec_from_file_location(
        "x70mod", REPO / "research/convexity_portable_2026-05-20/scripts/X70_build_3yr_and_regime_test.py")
    X70 = importlib.util.module_from_spec(spec); spec.loader.exec_module(X70)
    c = X70.load_closes("BTCUSDT")
    c.index = pd.DatetimeIndex(c.index).tz_convert("UTC")
    full = pd.date_range(c.index.min(), c.index.max(), freq="5min", tz="UTC")
    cf = c.reindex(full).astype(float)
    f = (cf.shift(-HORIZON) / cf - 1)
    f = f[(f.index.hour % 4 == 0) & (f.index.minute == 0)]
    return f.rename("btc_fwd")


def prep(era: str, rvol: pd.DataFrame, bfwd: pd.Series) -> pd.DataFrame:
    p = add_slow_signals(get_preds(era)).merge(rvol, on=["symbol", "open_time"], how="left")
    p["vrank"] = p.groupby("open_time")["rvol_7d"].rank(pct=True)
    p["rvol_slow"] = -p.sort_values(["symbol", "open_time"]).groupby("symbol")["vrank"].transform(
        lambda s: s.shift(1).expanding(min_periods=30).mean())
    p["fast"] = p["pred"]
    return p.merge(bfwd.rename_axis("open_time").reset_index(), on="open_time", how="left")


def ic(d, sig, target):
    return d.groupby("open_time").apply(
        lambda g: spearmanr(g[sig], g[target]).correlation if len(g) >= 10 else np.nan).dropna()


def day_ci(s, nb=3000):
    """Day-clustered CI on the MEAN of a per-bar series."""
    gg = [x.to_numpy() for _, x in s.groupby(pd.to_datetime(s.index, utc=True).floor("1D"))]
    b = [np.concatenate([gg[k] for k in RNG.integers(0, len(gg), len(gg))]).mean() for _ in range(nb)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def resid_book(d: pd.DataFrame, sig: str, q: float = 0.2, cost=None):
    """Quintile L/S on PER-NAME BTC-residual returns (alpha_A) — the quantity the strategy farms.
    Dollar-neutral, equal-weight within leg. Returns per-bar frame with gross / turnover / cost."""
    x = d.dropna(subset=[sig, "alpha_A"]).copy()
    x["rk"] = x.groupby("open_time")[sig].rank(pct=True)
    x["pos"] = np.where(x["rk"] >= 1 - q, 1.0, np.where(x["rk"] <= q, -1.0, 0.0))
    A = x.pivot_table(index="symbol", columns="open_time", values="alpha_A").fillna(0.0)
    P = x.pivot_table(index="symbol", columns="open_time", values="pos", fill_value=0.0).reindex_like(A)
    pos = P.clip(lower=0); neg = P.clip(upper=0)
    W = pos.div(pos.sum().replace(0, np.nan), axis=1).fillna(0.0) \
        + neg.div(neg.sum().abs().replace(0, np.nan), axis=1).fillna(0.0)
    gross = (W * A).sum(axis=0)
    dW = W.diff(axis=1).abs()
    out = pd.DataFrame({"g": gross, "t": 0.25 * dW.sum(axis=0)})
    if cost is not None:
        c, med = cost
        cvec = pd.Series([c.get(s, med) for s in W.index], index=W.index)
        out["c_ps"] = 0.25 * dW.mul(cvec, axis=0).sum(axis=0) / 1e4
    out["names"] = (W.abs() > 0).sum(axis=0)
    return out.iloc[1:].dropna(subset=["g", "t"])


def main():
    pc = cost_map()
    PAN = build_panel()
    rvol = PAN[["symbol", "open_time", "rvol_7d"]]
    bfwd = btc_fwd_4h()
    D = {era: prep(era, rvol, bfwd) for era in ERAS}

    # ---------------------------------------------------------------- PART A
    print("======== PART A — artifact control: IC(signal, residual label) by BTC forward tercile ========",
          flush=True)
    print("   (a beta-estimation artifact flips sign with BTC direction; a real premium does not)\n", flush=True)
    gateA = {}
    for era in ERAS:
        for n, lab in ((40, "top40"), (999, "all")):
            d = restrict_topn(D[era], n).dropna(subset=SIGS + ["btc_fwd"])
            bt = d.groupby("open_time")["btc_fwd"].first()
            terc = pd.qcut(bt, 3, labels=["btc_down", "btc_flat", "btc_up"])
            print(f"----- {era} / {lab} -----", flush=True)
            for sig in ("fast", "rvol_slow"):
                s_all = ic(d, sig, "alpha_A")
                lo, hi = day_ci(s_all)
                cells = []
                signs = []
                for t in ("btc_down", "btc_flat", "btc_up"):
                    bars = terc[terc == t].index
                    ss = s_all[s_all.index.isin(bars)]
                    l2, h2 = day_ci(ss)
                    cells.append(f"{t} {ss.mean():+.4f}[{l2:+.4f},{h2:+.4f}]")
                    signs.append(np.sign(ss.mean()))
                same = len(set(signs)) == 1
                if sig == "rvol_slow":
                    gateA[(era, lab)] = same
                print(f"  {sig:<10} all {s_all.mean():+.4f}[{lo:+.4f},{hi:+.4f}]   " + "  ".join(cells)
                      + f"   {'SAME-SIGN' if same else 'SIGN-FLIP'}", flush=True)
            # A3: raw-return control
            for sig in ("fast", "rvol_slow"):
                sr = ic(d, sig, "return_pct")
                lo, hi = day_ci(sr)
                print(f"  {sig:<10} IC vs RAW fwd return {sr.mean():+.4f}[{lo:+.4f},{hi:+.4f}]", flush=True)
            print("", flush=True)
    g2 = all(gateA.values())
    print(f"  G2 (rvol_slow same-sign in all BTC terciles, all cells): {'PASS' if g2 else 'FAIL'}\n", flush=True)

    # ---------------------------------------------------------------- PART B + C
    print("======== PART B/C — books on PER-NAME RESIDUAL returns (the farmed quantity) ========", flush=True)
    books = {}
    for era in ERAS:
        for n, lab in ((40, "top40"), (999, "all")):
            d = restrict_topn(D[era], n)
            for sig in SIGS:
                books[(era, lab, sig)] = resid_book(d, sig, cost=pc)
    rows = []
    for era in ERAS:
        for lab in ("top40", "all"):
            print(f"\n----- {era} / {lab} -----", flush=True)
            print(f"  {'signal':<11}{'turn':<7}{'names':<7}{'grossSh [CI]':<27}{'net@ps [CI]':<27}"
                  f"{'psCost':<8}{'net@8':<8}{'net@24':<8}{'maxDD':<8}", flush=True)
            for sig in SIGS:
                j = books[(era, lab, sig)]
                glo, ghi = block_ci(j["g"].to_numpy())
                nps = (j["g"] - j["c_ps"]).to_numpy()
                nlo, nhi = block_ci(nps)
                r = dict(era=era, tier=lab, sig=sig, turn=float(j["t"].mean()),
                         names=float(j["names"].mean()), gross=sharpe(j["g"]), g_lo=glo, g_hi=ghi,
                         net_ps=sharpe(nps), n_lo=nlo, n_hi=nhi,
                         cost_bps=float(j["c_ps"].mean() * 1e4),
                         net8=sharpe(j["g"] - j["t"] * 8e-4), net24=sharpe(j["g"] - j["t"] * 24e-4),
                         maxdd=maxdd(j["g"]) * 1e4)
                rows.append(r)
                gcell = f"{r['gross']:+.2f} [{glo:+.2f},{ghi:+.2f}] {tag_ci(glo, ghi)}"
                ncell = f"{r['net_ps']:+.2f} [{nlo:+.2f},{nhi:+.2f}] {tag_ci(nlo, nhi)}"
                print(f"  {sig:<11}{r['turn']:<7.3f}{r['names']:<7.1f}{gcell:<27}{ncell:<27}"
                      f"{r['cost_bps']:<8.2f}{r['net8']:<+8.2f}{r['net24']:<+8.2f}{r['maxdd']:<8.0f}", flush=True)
    T = pd.DataFrame(rows); T.to_csv(CACHE / "iter2_levels.csv", index=False)

    print("\n======== PAIRED Δ vs incumbent (residual books, common bars) ========", flush=True)
    prows = []
    for era in ERAS:
        for lab in ("top40", "all"):
            jf = books[(era, lab, "fast")]
            print(f"\n----- {era} / {lab} -----", flush=True)
            for sig in SIGS[1:]:
                jv = books[(era, lab, sig)]
                idx = jf.index.intersection(jv.index)
                a, b = jf.loc[idx], jv.loc[idx]
                cells, rec = [], dict(era=era, tier=lab, sig=sig)
                for name, fx, vx in (("gross", a["g"], b["g"]),
                                     ("netps", a["g"] - a["c_ps"], b["g"] - b["c_ps"]),
                                     ("net24", a["g"] - a["t"] * 24e-4, b["g"] - b["t"] * 24e-4)):
                    dd, lo, hi = paired_block_ci(fx.to_numpy(), vx.to_numpy())
                    cells.append(f"Δ{name} {dd:+.2f} [{lo:+.2f},{hi:+.2f}] {tag_ci(lo, hi)}")
                    rec.update({f"d_{name}": dd, f"d_{name}_lo": lo, f"d_{name}_hi": hi})
                prows.append(rec)
                print(f"  {sig:<11}" + "   ".join(f"{c:<34}" for c in cells), flush=True)
    P = pd.DataFrame(prows); P.to_csv(CACHE / "iter2_paired.csv", index=False)

    print("\n======== GATE READ ========", flush=True)
    inc = T[(T.sig == "fast") & (T.tier == "top40") & (T.era == "OOS")].iloc[0]
    g1 = 2.0 <= inc.gross <= 3.0
    print(f"  G1 harness validity (OOS/top40 incumbent residual-book gross {inc.gross:+.2f}, want +2.2..+2.6): "
          f"{'PASS' if g1 else 'CHECK'}", flush=True)
    print(f"  G2 artifact control: {'PASS' if g2 else 'FAIL'}", flush=True)
    g3 = all(T[(T.sig == "rvol_slow") & (T.tier == "top40") & (T.era == e)].iloc[0].g_lo > 0 for e in ERAS)
    print(f"  G3 rvol gross CI>0 both eras (top40): {'PASS' if g3 else 'FAIL'}", flush=True)
    sub = P[(P.sig == "rvol_slow") & (P.tier == "top40")]
    g4 = (sub.d_netps_lo > 0).any() and not (sub.d_netps_hi < 0).any()
    print(f"  G4 Δnet@ps CI>0 in >=1 era and not negative in the other: {'PASS' if g4 else 'FAIL'}", flush=True)
    print("\nITER2DONE", flush=True)


if __name__ == "__main__":
    main()
