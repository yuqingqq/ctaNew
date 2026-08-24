"""Follow-through from live/mc_oi_probe.py.

Snapshot finding: log(OI value) correlates +0.896 with log(market cap) — a better MC proxy than trailing ADV
(+0.797) — while OI/ADV is NOT a usable stand-in for OI/MC (corr −0.36, opposite sign). So the crowding ratio
OI/MC needs paid PIT market-cap history, but OI VALUE itself is a free, PIT, full-history size variable that
tracks market cap closely — and this repo has only ever used OI *changes* (oi_chg_1d/3d, oi_z_30d,
oi_price_div), never the level.

Two tests, both on owned data:
  A UNIVERSE. Does a top-N universe defined by OI value (≈ market cap) beat one defined by trailing ADV
    (the loop's incumbent)? Hard-split design from iteration 5: select 2023-06→2024-12, hold out 2025-01→2026-06.
  B SIGNAL. Does the OI-value rank (a size factor) carry rank-IC, and does it ADD to the incumbent prediction?
    Both eras, day-clustered CI on the paired delta.

Data-quality gate first (CLAUDE.md pitfall #3: the metrics cache was once destroyed by an overwrite bug and
recovered) — symbols below 80% daily coverage over the evaluation span are excluded and reported.
Run: python3 -u -m live.mc_oi_universe
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from live.cost_loop_harness import (
    ERAS, CACHE, REPO, block_ci, build_panel, get_preds, paired_block_ci, pit_adv, sharpe, tag_ci,
)
from live.cl_iter4_capacity import build, cost_tiers

SPAN0, SPAN1 = pd.Timestamp("2023-06-01", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC")
SEL0, SEL1 = SPAN0, pd.Timestamp("2025-01-01", tz="UTC")
HO0, HO1 = SEL1, SPAN1
N = 40
RNG = np.random.default_rng(31)


def load_oi() -> tuple[pd.DataFrame, list]:
    """PIT size from OI: 4h grid, 30d trailing mean of sum_open_interest_value, shifted 1 bar."""
    fp = CACHE / "oi_size_4h.parquet"
    if fp.exists():
        d = pd.read_parquet(fp); d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
        return d, []
    rows, thin = [], []
    for f in sorted(glob.glob(str(REPO / "data/ml/cache/metrics_*.parquet"))):
        sym = Path(f).stem.replace("metrics_", "")
        try:
            m = pd.read_parquet(f, columns=["sum_open_interest_value"])
            m = m[(m.index >= SPAN0 - pd.Timedelta(days=40)) & (m.index < SPAN1)]
            if m.empty:
                continue
            v = m["sum_open_interest_value"].astype(float).replace(0, np.nan)
            days = v.resample("1D").count()
            live_days = days[days.index >= max(SPAN0, v.index.min().normalize())]
            cov = float((live_days >= 0.8 * 288).mean()) if len(live_days) else 0.0
            if cov < 0.80:
                thin.append((sym, round(cov, 3), int(len(live_days)))); continue
            g = v.resample("4h", label="left", closed="left").last()
            s = g.rolling(180, min_periods=60).mean().shift(1)      # 30d trailing, PIT
            rows.append(pd.DataFrame({"symbol": sym, "open_time": s.index, "oi_usd": s.values}))
        except Exception as e:
            thin.append((sym, f"ERR {str(e)[:30]}", 0))
    D = pd.concat(rows, ignore_index=True).dropna(subset=["oi_usd"])
    D["open_time"] = pd.to_datetime(D["open_time"], utc=True)
    D.to_parquet(fp, index=False)
    return D, thin


def topn(d: pd.DataFrame, col: str, n: int) -> pd.DataFrame:
    x = d.dropna(subset=[col]).copy()
    x["rk"] = x.groupby("open_time")[col].rank(ascending=False, method="first")
    return x[x["rk"] <= n].drop(columns=["rk"])


def evaluate(d: pd.DataFrame, CT, ctl="band"):
    W, A = build(d, ctl)
    g = (W * A).sum(axis=0)
    dW = W.diff(axis=1).abs()
    c, med = CT["cost_10k"]
    cvec = pd.Series([c.get(s, med) for s in W.index], index=W.index)
    net = (g - 0.25 * dW.mul(cvec, axis=0).sum(axis=0) / 1e4).iloc[1:]
    return g.iloc[1:], net, float((0.25 * dW.sum(axis=0)).iloc[1:].mean())


def perbar_ic(d, sig, tgt="alpha_A"):
    return d.groupby("open_time").apply(
        lambda x: spearmanr(x[sig], x[tgt]).correlation if len(x) >= 10 else np.nan).dropna()


def day_ci_diff(a, b, nb=3000):
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    j["d"] = j["b"] - j["a"]
    gg = [x["d"].to_numpy() for _, x in j.groupby(pd.to_datetime(j.index, utc=True).floor("1D"))]
    boot = [np.concatenate([gg[k] for k in RNG.integers(0, len(gg), len(gg))]).mean() for _ in range(nb)]
    return float(j["d"].mean()), *np.percentile(boot, [2.5, 97.5])


def main():
    CT = cost_tiers()
    OI, thin = load_oi()
    if thin:
        print(f"COVERAGE GATE: excluded {len(thin)} symbols below 80% daily OI coverage: "
              f"{[t[0] for t in thin][:12]}{'...' if len(thin) > 12 else ''}", flush=True)
    print(f"OI size panel: {OI.symbol.nunique()} syms, {OI.open_time.min().date()} -> "
          f"{OI.open_time.max().date()}", flush=True)

    PAN = build_panel()
    lab = PAN[["symbol", "open_time", "alpha_vs_btc_realized"]].rename(
        columns={"alpha_vs_btc_realized": "alpha_A"})
    P = pd.concat([get_preds(e) for e in ERAS], ignore_index=True).drop_duplicates(
        ["symbol", "open_time"]).sort_values(["symbol", "open_time"])
    if "alpha_A" not in P.columns:
        P = P.merge(lab, on=["symbol", "open_time"], how="left")
    P = P.merge(OI, on=["symbol", "open_time"], how="left")
    A = pit_adv(); P["date"] = P["open_time"].dt.floor("1D")
    P = P.merge(A, on=["symbol", "date"], how="left")

    # ---------------------------------------------------------------- A. universe
    print("\n============ A. UNIVERSE — top-40 by OI value (≈mcap) vs by trailing ADV ============",
          flush=True)
    both = P.dropna(subset=["oi_usd", "tadv"])
    ov = both.groupby("open_time").apply(
        lambda g: len(set(g.nlargest(N, "oi_usd").symbol) & set(g.nlargest(N, "tadv").symbol)) / N)
    print(f"  universe overlap (top-{N} OI vs top-{N} ADV): mean {ov.mean()*100:.0f}% "
          f"(p10 {ov.quantile(.1)*100:.0f}%, p90 {ov.quantile(.9)*100:.0f}%)", flush=True)

    res = {}
    for wname, (t0, t1) in (("SELECT", (SEL0, SEL1)), ("HOLDOUT", (HO0, HO1))):
        w = both[(both.open_time >= t0) & (both.open_time < t1)]
        for key, col in (("OI", "oi_usd"), ("ADV", "tadv")):
            g, net, turn = evaluate(topn(w, col, N), CT)
            lo, hi = block_ci(net.to_numpy())
            res[(wname, key)] = (g, net)
            print(f"  {wname:<8} top{N}-by-{key:<4} gross {sharpe(g):+.2f}  net@10k {sharpe(net):+.2f} "
                  f"[{lo:+.2f},{hi:+.2f}] {tag_ci(lo, hi)}  turn {turn:.3f}  bars {len(net)}", flush=True)
    for wname in ("SELECT", "HOLDOUT"):
        a, b = res[(wname, "ADV")][1], res[(wname, "OI")][1]
        idx = a.index.intersection(b.index)
        dd, lo, hi = paired_block_ci(a.loc[idx].to_numpy(), b.loc[idx].to_numpy())
        print(f"  {wname:<8} Δ(OI-universe − ADV-universe) net@10k {dd:+.2f} [{lo:+.2f},{hi:+.2f}] "
              f"{tag_ci(lo, hi)}", flush=True)

    # ---------------------------------------------------------------- B. signal
    print("\n============ B. SIGNAL — is the OI LEVEL (size factor) informative / incremental? ============",
          flush=True)
    P["oi_rank"] = P.groupby("open_time")["oi_usd"].rank(pct=True)
    P["oi_adv_ratio"] = P["oi_usd"] / P["tadv"]
    P["oiadv_rank"] = P.groupby("open_time")["oi_adv_ratio"].rank(pct=True)
    for era, (t0, t1) in (("OOS", (SPAN0, HO0)), ("RECENT", (HO0, SPAN1))):
        d = P[(P.open_time >= t0) & (P.open_time < t1)]
        d40 = topn(d.dropna(subset=["tadv"]), "tadv", N).dropna(subset=["pred", "alpha_A", "oi_rank"])
        print(f"\n----- {era} (top-{N} ADV universe, {d40.open_time.nunique()} bars) -----", flush=True)
        base = perbar_ic(d40, "pred")
        print(f"  incumbent pred        rank-IC {base.mean():+.4f}", flush=True)
        for sig, sgn in (("oi_rank", -1.0), ("oi_rank", +1.0), ("oiadv_rank", -1.0), ("oiadv_rank", +1.0)):
            d40["_s"] = sgn * d40[sig]
            s = perbar_ic(d40, "_s")
            if sgn > 0:
                print(f"  {sig:<21} rank-IC {s.mean():+.4f}  (long HIGH)", flush=True)
            else:
                print(f"  {sig:<21} rank-IC {s.mean():+.4f}  (long LOW)", flush=True)
        # incremental: equal-weight blend of z(pred) and z(signal)
        for sig in ("oi_rank", "oiadv_rank"):
            zp = d40.groupby("open_time")["pred"].transform(lambda x: (x - x.mean()) / (x.std() or 1))
            zs = d40.groupby("open_time")[sig].transform(lambda x: (x - x.mean()) / (x.std() or 1))
            for sgn in (-1.0, +1.0):
                d40["_b"] = zp + sgn * zs
                b = perbar_ic(d40, "_b")
                dd, lo, hi = day_ci_diff(base, b)
                tag = "ADDS" if lo > 0 else ("hurts" if hi < 0 else "within noise")
                print(f"  pred {'−' if sgn < 0 else '+'} {sig:<14} Δ rank-IC {dd:+.4f} "
                      f"[{lo:+.4f},{hi:+.4f}]  {tag}", flush=True)
    print("\nMCOIUNIVDONE", flush=True)


if __name__ == "__main__":
    main()
