"""Composite Study — construction test: does cross-sectional rank → top/
bottom-K (the production-style construction) extract a tradeable result
from THE linear alpha model's leak-free predictions, beyond the +0.62
naive-construction baseline?  LOCKED, one run, no tuning.

Model = D1 leak-free F_core Ridge OOF preds. Construction = per cycle rank
by pred, long top-K / short bottom-K, equal 1/K weights, ± V3.1 6-sleeve.
K ∈ {3,5} pre-locked, both reported. Gate: beats matched random-K placebo
p95 AND netSh>+1.5 CI-excl-0 AND ≥6/9 folds AND lift≥+0.5 vs naive +0.62.
Cost sweep {1.0,1.75,2.25}. Prior: strong-guarded. Descriptive/
pre-registered — NOT auto-adopted. Production LGBM unaffected.
"""
from __future__ import annotations
import importlib.util, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(n, r):
    s = importlib.util.spec_from_file_location(n, REPO / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


s94 = _imp("s94", "linear_model/scripts/94_info_ceiling_d1.py")
s94b = _imp("s94b", "linear_model/scripts/94b_info_ceiling_d1_grouped.py")
from ml.research.alpha_v4_xs import block_bootstrap_ci
ANN = np.sqrt(365.0 * 6.0)
OUTD = REPO / "linear_model/composite_study/results"


def sh(x):
    x = np.asarray(x, float)
    return float(x.mean()/x.std(ddof=1)*ANN) if x.std(ddof=1) > 1e-12 else np.nan


def portfolio(df, wcol, cper):
    """df has symbol, open_time, fold, alpha_beta, wcol(weight). net bps/cyc."""
    f = df.sort_values(["symbol", "open_time"]).copy()
    f["dw"] = f.groupby("symbol")[wcol].diff().abs().fillna(f[wcol].abs())
    p = f.groupby(["open_time", "fold"]).apply(
        lambda g: pd.Series({
            "gross": (g[wcol]*g["alpha_beta"]).sum()*1e4,
            "cost": g["dw"].sum()*cper})).reset_index()
    p["net"] = p["gross"] - p["cost"]
    return p.sort_values("open_time")


def kselect(d, K):
    """per cycle: long top-K by pred (w=+1/K), short bottom-K (w=-1/K)."""
    d = d.copy()
    r_hi = d.groupby("open_time")["pred"].rank(ascending=False, method="first")
    r_lo = d.groupby("open_time")["pred"].rank(ascending=True, method="first")
    w = np.where(r_hi <= K, 1.0/K, np.where(r_lo <= K, -1.0/K, 0.0))
    d["w"] = w
    return d


def sleeve(d, wcol="w", dst="ws"):
    f = d.sort_values(["symbol", "open_time"]).copy()
    f[dst] = f.groupby("symbol")[wcol].transform(
        lambda v: v.rolling(6, min_periods=1).mean())
    return f


def main():
    print("=" * 100, flush=True)
    print("  COMPOSITE STUDY — construction: rank → top/bottom-K on the "
          "leak-free alpha model (LOCKED)", flush=True)
    print("=" * 100, flush=True)
    dec, syms, btc, pan = s94.build(universe_oi=False)
    LEAK = s94.LEAK
    FEATS = [c for c in dec.columns if c not in LEAK and
             pd.api.types.is_numeric_dtype(dec[c])]
    if "s_t" not in FEATS:
        FEATS.append("s_t")
    d = dec.dropna(subset=FEATS + ["tz", "alpha_beta"]).reset_index(drop=True)
    rid, _ = s94b.grouped_oof(d, FEATS)
    d["pred"] = rid
    d = d[~d["pred"].isna()].reset_index(drop=True)
    nsym = d.groupby("open_time")["symbol"].nunique()
    print(f"  rows={len(d)} syms={d.symbol.nunique()} cycles={d.open_time.nunique()}"
          f"  median syms/cyc={int(nsym.median())}", flush=True)

    # naive baseline (sign(pred), all symbols, equal-weight) at VIP-0
    base = d.copy()
    base["w"] = np.sign(base["pred"]) / d.groupby("open_time")["symbol"].transform("count")
    bnet = portfolio(base, "w", 2.25)["net"]
    BASE = sh(bnet)
    print(f"  NAIVE baseline (all-sym sign(pred), VIP-0) Sharpe = {BASE:+.2f} "
          f"(the established result to beat by ≥+0.5)\n", flush=True)

    print(f"  {'variant':22s} {'cost':>5s} {'netSh':>6s} {'CI':>15s} "
          f"{'fold+':>5s} {'plcP95':>7s} {'rank':>5s} {'vsBASE':>7s}", flush=True)
    rows = []
    for K in (3, 5):
        dk = kselect(d, K)
        dk = sleeve(dk, "w", "ws")
        for sname, wc in [("noSleeve", "w"), ("V3.1sleeve", "ws")]:
            for cper, clab in [(1.00, "mk"), (1.75, "3.5"), (2.25, "v0")]:
                pr = portfolio(dk, wc, cper)
                net = pr["net"].to_numpy()
                S = sh(net)
                lo, hi = block_bootstrap_ci(
                    net, statistic=lambda z: z.mean()/z.std(ddof=1)*ANN
                    if z.std(ddof=1) > 1e-12 else 0.0, block_size=7,
                    n_boot=800)[1:]
                fp = sum(1 for _, g in pr.groupby("fold") if g["net"].mean() > 0)
                nf = pr["fold"].nunique()
                # matched random-K placebo (only at VIP-0 to bound cost)
                p95 = rk = np.nan
                if cper == 2.25:
                    pl = []
                    rng = np.random.default_rng(0)
                    for _ in range(200):
                        dp = d.copy()
                        gp = dp.groupby("open_time")
                        rr = gp["pred"].transform(
                            lambda s: pd.Series(rng.permutation(s.values),
                                                index=s.index))
                        dp["pred"] = rr
                        dpk = kselect(dp, K)
                        if wc == "ws":
                            dpk = sleeve(dpk, "w", "ws")
                        pl.append(sh(portfolio(dpk, wc, 2.25)["net"]))
                    p95 = float(np.nanpercentile(pl, 95))
                    rk = float((np.array(pl) < S).mean()*100)
                tag = f"K{K}-{sname}"
                print(f"  {tag:22s} {clab:>5s} {S:+6.2f} "
                      f"[{lo:+5.1f},{hi:+5.1f}] {fp:2d}/{nf:<2d} "
                      f"{p95:+7.2f} {rk:5.0f} {S-BASE:+7.2f}", flush=True)
                rows.append(dict(variant=tag, cost=clab, net_sh=round(S, 2),
                    ci_lo=round(lo, 2), ci_hi=round(hi, 2),
                    folds=f"{fp}/{nf}", plc_p95=round(p95, 2) if p95 == p95 else None,
                    plc_rank=rk if rk == rk else None,
                    vs_base=round(S-BASE, 2)))
    R = pd.DataFrame(rows)
    R.to_csv(OUTD/"construction_rankK.csv", index=False)

    # adoption check at VIP-0
    v0 = R[R.cost == "v0"].copy()
    best = v0.loc[v0.net_sh.idxmax()]
    adopt = bool(best.net_sh > 1.5 and best.ci_lo > 0 and
                 (best.plc_rank or 0) >= 95 and
                 (best.vs_base) >= 0.5 and
                 int(best.folds.split("/")[0]) >= 6)
    if adopt:
        verdict = (f"CONSTRUCTION ADOPTED: best {best.variant} VIP-0 netSh "
                   f"{best.net_sh:+.2f} (CI[{best.ci_lo},{best.ci_hi}], "
                   f"folds {best.folds}, placebo rank {best.plc_rank:.0f}, "
                   f"vs naive {best.vs_base:+.2f}) clears all gates — the "
                   f"rank-K construction extracts a tradeable result from the "
                   f"model. Requires strict-walk-forward nested-OOS confirm.")
    else:
        verdict = (f"CONSTRUCTION does NOT rescue: best VIP-0 {best.variant} "
                   f"netSh {best.net_sh:+.2f} CI[{best.ci_lo},{best.ci_hi}] "
                   f"folds {best.folds} placebo-rank {best.plc_rank} vs-naive "
                   f"{best.vs_base:+.2f} — fails the pre-registered gate "
                   f"(>+1.5 & CI-excl-0 & placebo-p95 & ≥6/9 & lift≥+0.5). "
                   f"Cross-sectional rank-K + sleeve does not turn the "
                   f"real-but-sub-cost idiosyncratic signal into a robust "
                   f"edge — confirms extraction is maxed at the construction "
                   f"layer too (consistent with Steps 01–62 architecture-"
                   f"amplification finding, IC-selector value-negative, "
                   f"production universe-overfit). Production LGBM unaffected.")
    print(f"\n  VERDICT: {verdict}", flush=True)
    pd.DataFrame([{"naive_base": BASE, "adopt": adopt,
                   "verdict": verdict}]).to_csv(OUTD/"construction_verdict.csv",
                                                index=False)
    print(f"\nSaved {OUTD}/construction_rankK.csv", flush=True)


if __name__ == "__main__":
    main()
