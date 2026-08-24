"""Cost/turnover loop — iteration 6 (D5): preprocessing window vs decay half-life.

H5: x6.fit_preproc builds winsor bounds / z-stats / heavy-tail empirical CDFs on the ENTIRE unweighted
per-symbol training history, while RidgeCV weights samples with a 60-day half-life. The scaling therefore
describes a sample the model barely uses. Matching the preproc window to the effective sample should help.

Single axis, no knob-fitting:  full (incumbent)  vs  trail120 (~2 half-lives)  vs  trail240.
Everything else identical to the incumbent pipeline (same features, same HL, same walk-forward cuts, same
exit_time purge + 1d embargo).

G1 book level: paired Δ rank-IC vs incumbent, day-clustered CI excludes 0 in BOTH eras.
G2 portfolio  : held-out (2025-01..2026-06) net@10k on top40/band improves, paired 7d-block CI > 0.
Falsifier: G1 fails -> mismatch immaterial, D5 closes as a null.
Run: python3 -u -m live.cl_iter6_preproc
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

from live.cost_loop_harness import (
    ERAS, CACHE, CUTS, block_ci, build_panel, paired_block_ci, restrict_topn, sharpe, tag_ci,
)
from live.v0_feature_ablation import V0
from live.build_alpha_beta_decomp import x6, FULL
from live.cl_iter4_capacity import build, cost_tiers

EMB = pd.Timedelta(days=1)
HL = 60.0
VARIANTS = {"full": None, "trail120": 120, "trail240": 240}
HO0, HO1 = pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC")
RNG = np.random.default_rng(23)


def gen_pred_pp(PAN, feats, cuts, trail_days):
    """Incumbent walk-forward per-symbol RidgeCV, with the preproc stats fit on the trailing
    `trail_days` of each symbol's training window (None = full history = incumbent)."""
    rec = []
    for i in range(len(cuts) - 1):
        c0, c1 = cuts[i], cuts[i + 1]; fc = c0 - EMB
        tr = PAN[(PAN.exit_time < fc) & PAN["z_res"].notna()]
        te = PAN[(PAN.open_time >= c0) & (PAN.open_time < c1)]
        if tr.empty or te.empty:
            continue
        t_end = tr["open_time"].max()
        for sym, gg in tr.groupby("symbol"):
            if len(gg) < 300:
                continue
            try:
                pp = gg if trail_days is None else gg[gg["open_time"] >= t_end - pd.Timedelta(days=trail_days)]
                if len(pp) < 100:
                    pp = gg                       # fall back rather than fit stats on a stub
                s, h = x6.fit_preproc(pp, feats)
                X = x6.apply_preproc(gg, feats, s, h)
                w = np.exp(-((t_end - gg["open_time"]).dt.total_seconds().to_numpy() / 86400.0) / HL)
                m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, gg["z_res"].to_numpy(), sample_weight=w)
                gte = te[te.symbol == sym]
                if len(gte):
                    rec.append(pd.DataFrame({"symbol": sym, "open_time": gte["open_time"].values,
                                             "pred": m.predict(x6.apply_preproc(gte, feats, s, h))}))
            except Exception:
                pass
    return pd.concat(rec, ignore_index=True) if rec else pd.DataFrame()


def perbar_ic(d):
    return d.groupby("open_time").apply(
        lambda g: spearmanr(g["pred"], g["alpha_A"]).correlation if len(g) >= 10 else np.nan).dropna()


def day_ci_diff(a, b, nb=3000):
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    j["d"] = j["b"] - j["a"]
    gg = [x["d"].to_numpy() for _, x in j.groupby(pd.to_datetime(j.index, utc=True).floor("1D"))]
    boot = [np.concatenate([gg[k] for k in RNG.integers(0, len(gg), len(gg))]).mean() for _ in range(nb)]
    return float(j["d"].mean()), *np.percentile(boot, [2.5, 97.5])


def main():
    CT = cost_tiers()
    PAN = build_panel()
    lab = PAN[["symbol", "open_time", "alpha_vs_btc_realized"]].rename(
        columns={"alpha_vs_btc_realized": "alpha_A"})
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)

    preds = {}
    for name, td in VARIANTS.items():
        for era in ERAS:
            fp = CACHE / f"preds_pp_{name}_{era}.parquet"
            if fp.exists():
                p = pd.read_parquet(fp)
                p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
            else:
                p = gen_pred_pp(PAN, list(V0), CUTS[era], td)
                p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
                p = p.merge(lab, on=["symbol", "open_time"], how="inner").merge(
                    RP, on=["symbol", "open_time"], how="inner").dropna()
                p.to_parquet(fp, index=False)
            preds[(name, era)] = p
            print(f"  [{name}/{era}] {len(p):,} rows", flush=True)

    print("\n============ G1 — book-level rank-IC (day-clustered paired CI vs incumbent) ============",
          flush=True)
    ics = {}
    for era in ERAS:
        print(f"\n----- {era} -----", flush=True)
        for name in VARIANTS:
            ics[(name, era)] = perbar_ic(preds[(name, era)])
            line = f"  {name:<10} rank-IC {ics[(name, era)].mean():+.4f}"
            if name != "full":
                d, lo, hi = day_ci_diff(ics[("full", era)], ics[(name, era)])
                line += f"   Δ vs full {d:+.4f} [{lo:+.4f},{hi:+.4f}] {tag_ci(lo, hi)}"
            print(line, flush=True)
    g1 = {n: all(day_ci_diff(ics[("full", e)], ics[(n, e)])[1] > 0 for e in ERAS)
          for n in VARIANTS if n != "full"}

    print("\n============ G2 — held-out top40/band net@10k (paired 7d-block CI) ============", flush=True)
    series = {}
    for name in VARIANTS:
        P = pd.concat([preds[(name, e)] for e in ERAS], ignore_index=True) \
              .drop_duplicates(["symbol", "open_time"]).sort_values(["symbol", "open_time"])
        w = P[(P.open_time >= HO0) & (P.open_time < HO1)]
        d = restrict_topn(w, 40)
        W, A = build(d, "band")
        g = (W * A).sum(axis=0)
        dW = W.diff(axis=1).abs()
        c, med = CT["cost_10k"]
        cvec = pd.Series([c.get(s, med) for s in W.index], index=W.index)
        net = (g - 0.25 * dW.mul(cvec, axis=0).sum(axis=0) / 1e4).iloc[1:]
        series[name] = net
        lo, hi = block_ci(net.to_numpy())
        print(f"  {name:<10} gross {sharpe(g.iloc[1:]):+.2f}  net@10k {sharpe(net):+.2f} "
              f"[{lo:+.2f},{hi:+.2f}] {tag_ci(lo, hi)}  turn {(0.25*dW.sum(axis=0)).iloc[1:].mean():.3f}",
              flush=True)
    g2 = {}
    for name in VARIANTS:
        if name == "full":
            continue
        idx = series["full"].index.intersection(series[name].index)
        dd, lo, hi = paired_block_ci(series["full"].loc[idx].to_numpy(), series[name].loc[idx].to_numpy())
        g2[name] = lo > 0
        print(f"  Δ({name} − full) net@10k {dd:+.2f} [{lo:+.2f},{hi:+.2f}] {tag_ci(lo, hi)}", flush=True)

    print("\n============ GATE READ ============", flush=True)
    for name in g1:
        print(f"  {name:<10} G1 (Δ rank-IC CI>0 both eras) {'PASS' if g1[name] else 'FAIL'}   "
              f"G2 (held-out Δnet CI>0) {'PASS' if g2.get(name) else 'FAIL'}", flush=True)
    print("\nITER6DONE", flush=True)


if __name__ == "__main__":
    main()
