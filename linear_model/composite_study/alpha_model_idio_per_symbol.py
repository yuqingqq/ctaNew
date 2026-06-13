"""Composite Study — does THE linear alpha model carry idiosyncratic
per-symbol alpha? (LOCKED, descriptive.)

Model = D1's leak-free F_core model (Ridge primary + LGBM), the +0.62-ceiling
model, predicting alpha_beta (which is ALREADY the β-stripped idiosyncratic
residual — pure idio-skill test, no beta confound). Reuses the audited
whole-timestamp+embargo CV (s94b.grouped_oof). Decompose per-symbol:
IC, net-of-cost alpha, hit, per-fold; cross-symbol mean vs noise dispersion;
within-symbol-shuffle placebo. All symbols reported. In-sample-OOF
descriptive — any positive subset needs the loop-closed nested gate (arc
record: such subsets fail). NOT a strategy. Production LGBM unaffected.
"""
from __future__ import annotations
import importlib.util, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(n, r):
    s = importlib.util.spec_from_file_location(n, REPO / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


s94 = _imp("s94", "linear_model/scripts/94_info_ceiling_d1.py")
s94b = _imp("s94b", "linear_model/scripts/94b_info_ceiling_d1_grouped.py")
from ml.research.alpha_v4_xs import block_bootstrap_ci
ANN = np.sqrt(365.0 * 6.0)
COST = s94.COST
OUTD = REPO / "linear_model/composite_study/results"


def main():
    print("=" * 104, flush=True)
    print("  COMPOSITE STUDY — idiosyncratic per-symbol alpha of THE linear "
          "alpha model (D1 leak-free F_core)", flush=True)
    print("=" * 104, flush=True)
    dec, syms, btc, pan = s94.build(universe_oi=False)
    LEAK = s94.LEAK
    FEATS = [c for c in dec.columns if c not in LEAK and
             pd.api.types.is_numeric_dtype(dec[c])]
    if "s_t" not in FEATS:
        FEATS.append("s_t")
    d = dec.dropna(subset=FEATS + ["tz", "alpha_beta"]).reset_index(drop=True)
    print(f"  universe (hl42) = {d.symbol.nunique()} syms, rows={len(d)}, "
          f"F_core={len(FEATS)} feats (the D1 alpha model)", flush=True)

    rid, gbm = s94b.grouped_oof(d, FEATS)              # leak-free OOF preds
    d["pred"] = rid                                    # Ridge = the +0.62 model
    d = d[~d["pred"].isna()].reset_index(drop=True)
    ab = d["alpha_beta"].to_numpy() * 1e4

    # ---- aggregate sanity (reproduce D1) ----
    pos = np.sign(d["pred"].to_numpy())
    f = d.assign(pos=pos).sort_values(["symbol", "open_time"]).copy()
    f["dp"] = f.groupby("symbol")["pos"].diff().abs().fillna(f["pos"].abs())
    g = f["pos"]*f["alpha_beta"]*1e4 - f["dp"]*COST
    aic = float(pd.Series(d["pred"]).corr(d["alpha_beta"], "spearman"))
    ash = float(g.mean()/g.std(ddof=1)*ANN)
    print(f"\n  AGGREGATE (sanity vs D1 +0.62): pooled IC={aic:+.4f}  "
          f"net Sharpe={ash:+.2f}  (D1 Ridge leak-free was ≈+0.62)",
          flush=True)

    # ---- per-symbol idiosyncratic decomposition ----
    print("\n  --- per-symbol: does the model carry idio alpha on each? ---",
          flush=True)
    print(f"  {'sym':12s} {'n':>5s} {'IC':>7s} {'net_bps':>8s} {'hit%':>5s} "
          f"{'netSh':>6s} {'fold+':>6s}", flush=True)
    rows = []
    for s, gg in d.groupby("symbol"):
        gg = gg.sort_values("open_time")
        p = np.sign(gg["pred"].to_numpy())
        dp = np.r_[abs(p[0]), np.abs(np.diff(p))]
        net = p*gg["alpha_beta"].to_numpy()*1e4 - dp*COST
        ic = float(gg["pred"].corr(gg["alpha_beta"], "spearman"))
        hit = float((np.sign(gg["pred"]) == np.sign(gg["alpha_beta"])).mean()*100)
        sh = float(net.mean()/net.std(ddof=1)*ANN) if net.std(ddof=1) > 1e-9 else np.nan
        fp = sum(1 for fl, x in gg.groupby("fold")
                 if (np.sign(x["pred"])*x["alpha_beta"]).mean() > 0)
        nf = gg["fold"].nunique()
        rows.append(dict(sym=s, n=len(gg), ic=ic, net_bps=net.mean(),
                         hit=hit, net_sh=sh, fp=f"{fp}/{nf}", fpn=fp/max(nf, 1)))
    R = pd.DataFrame(rows).sort_values("ic", ascending=False)
    for _, x in R.iterrows():
        print(f"  {x['sym']:12s} {x['n']:5d} {x['ic']:+7.4f} "
              f"{x['net_bps']:+8.2f} {x['hit']:5.1f} {x['net_sh']:+6.2f} "
              f"{x['fp']:>6s}", flush=True)

    ics = R["ic"].to_numpy()
    mu, sd = ics.mean(), ics.std(ddof=1)
    tstat = mu / (sd/np.sqrt(len(ics)))
    fpos = (ics > 0).mean()*100
    # within-symbol shuffle placebo: destroy timing, keep pred distribution
    rng = np.random.default_rng(0)
    pl_mu = []
    for _ in range(300):
        pp = d.copy()
        pp["pred"] = pp.groupby("symbol")["pred"].transform(
            lambda v: rng.permutation(v.values))
        m = pp.groupby("symbol").apply(
            lambda q: q["pred"].corr(q["alpha_beta"], "spearman")).mean()
        pl_mu.append(m)
    p95 = np.nanpercentile(pl_mu, 95)

    print(f"\n  CROSS-SYMBOL: mean per-sym IC = {mu:+.4f}  std = {sd:.4f}  "
          f"(noise/signal = {sd/abs(mu) if mu else float('inf'):.1f}:1)  "
          f"t = {tstat:+.2f}  %syms IC>0 = {fpos:.0f}%", flush=True)
    print(f"  mean per-sym IC vs within-symbol-shuffle placebo p95 = "
          f"{p95:+.4f}  → {'BEATS' if mu > p95 else 'does NOT beat'}",
          flush=True)
    best = R.iloc[0]
    print(f"  best symbol: {best['sym']} IC={best['ic']:+.4f} "
          f"netSh={best['net_sh']:+.2f} folds {best['fp']} "
          f"(1-of-{len(R)} = selection; needs nested gate — arc record: fails)",
          flush=True)
    R.drop(columns="fpn").to_csv(OUTD/"alpha_model_per_symbol.csv", index=False)
    verdict = (
        f"Linear alpha model idiosyncratic per-symbol alpha: mean IC {mu:+.4f} "
        f"(std {sd:.4f}, noise:signal {sd/abs(mu) if mu else 0:.0f}:1, t {tstat:+.2f}), "
        f"{fpos:.0f}% syms IC>0, mean {'<=' if mu<=p95 else '>'} placebo p95 "
        f"{p95:+.4f}. {'No stable idiosyncratic alpha — per-symbol IC is ' 'noise around ~0, dispersion dominates mean; positive symbols are ' 'the selection trap (consistent with D1 +0.62-sub-cost, Phase-CAL ' 'std-7:1, Step-90 nested −2.60). The linear alpha model has no ' 'harvestable per-symbol idiosyncratic edge in free data.' if (mu <= p95 or tstat < 2) else 'mean per-sym IC beats placebo & t>2 — candidate; requires loop-closed nested-OOS before any claim.'} "
        f"Production LGBM unaffected.")
    print(f"\n  VERDICT: {verdict}", flush=True)
    pd.DataFrame([{"verdict": verdict, "mean_ic": mu, "std_ic": sd,
                   "t": tstat, "placebo_p95": p95, "agg_ic": aic,
                   "agg_sh": ash}]).to_csv(OUTD/"alpha_model_verdict.csv",
                                           index=False)
    print(f"\nSaved {OUTD}/alpha_model_per_symbol.csv", flush=True)


if __name__ == "__main__":
    main()
