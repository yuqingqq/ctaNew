"""LONG-PRED iter-002 — Confirm right-tail-event mechanism

CLAIM from iter-001: in H2, the per-sym forward residual is right-skewed
(89.9% syms positive skew). The model's top-K longs underperform the universe
MEAN because the universe mean is being PULLED UP by the right-tail pumpers
(random/unpredictable events), while the model picks "slightly-less-bleeding"
names.

ORTHOGONAL TEST: if this is the operative mechanism, then top-K edge
COMPUTED VS UNIVERSE MEDIAN should be POSITIVE in H2 (because the median
isn't pulled up by the right-tail outliers). If median-based edge is positive,
the model IS picking good names — just not the rare pumpers.

If median-based edge is also negative: the model is actively wrong (picks
that are below median), and the diagnosis is incorrect — we need to look at
per-sym signal decay instead.

Decisive test: produces a clear binary verdict on the hypothesis.
"""
import sys, time
from pathlib import Path
import pandas as pd, numpy as np
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
PREDS = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"

H1 = (pd.Timestamp("2025-10-04",tz="UTC"), pd.Timestamp("2026-01-22",tz="UTC"))
H2 = (pd.Timestamp("2026-01-22",tz="UTC"), pd.Timestamp("2026-05-26",tz="UTC"))

def main():
    t0 = time.time()
    print("=== LONG-PRED iter-002: median-based vs mean-based top-K edge ===\n", flush=True)
    d = pd.read_parquet(PREDS)
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    d = d[(d["open_time"].dt.hour%4==0)&(d["open_time"].dt.minute==0)]

    for label,(s,e) in [("H1",H1),("H2",H2)]:
        sub = d[(d["open_time"]>=s)&(d["open_time"]<e)]
        print(f"=== {label} (n_cycles={sub['open_time'].nunique()}) ===")
        # Per-cycle: top-K edge vs MEAN and vs MEDIAN
        for k in [1, 2, 3, 5]:
            top_edge_mean = []; top_edge_median = []
            bot_edge_mean = []; bot_edge_median = []
            for ot, g in sub.groupby("open_time"):
                if len(g) < 2*k: continue
                g = g.sort_values("pred")
                top_ret = g.tail(k)["return_pct"].mean()
                bot_ret = g.head(k)["return_pct"].mean()
                u_mean = g["return_pct"].mean()
                u_median = g["return_pct"].median()
                top_edge_mean.append(top_ret - u_mean)
                top_edge_median.append(top_ret - u_median)
                bot_edge_mean.append(u_mean - bot_ret)
                bot_edge_median.append(u_median - bot_ret)
            tem = np.mean(top_edge_mean)*1e4
            tmd = np.mean(top_edge_median)*1e4
            bem = np.mean(bot_edge_mean)*1e4
            bmd = np.mean(bot_edge_median)*1e4
            t_mean_pos = 100*(np.array(top_edge_mean)>0).mean()
            t_med_pos  = 100*(np.array(top_edge_median)>0).mean()
            print(f"  K={k}: top vs MEAN {tem:+6.1f} bps ({t_mean_pos:.0f}%>0)  vs MEDIAN {tmd:+6.1f} bps ({t_med_pos:.0f}%>0)  "
                  f"|  bot vs MEAN {bem:+6.1f}  vs MEDIAN {bmd:+6.1f}")

    # Also check: what % of cycles have a "right-tail outlier" (>2 std above mean)?
    print(f"\n=== % cycles with right-tail outliers (max return_pct > mean+2*std) ===")
    for label,(s,e) in [("H1",H1),("H2",H2)]:
        sub = d[(d["open_time"]>=s)&(d["open_time"]<e)]
        cycles_with_outlier = 0; total = 0; outlier_size_bps = []
        for ot, g in sub.groupby("open_time"):
            if len(g) < 15: continue
            total += 1
            m = g["return_pct"].mean(); s_ = g["return_pct"].std()
            mx = g["return_pct"].max()
            if mx > m + 2*s_:
                cycles_with_outlier += 1
                outlier_size_bps.append((mx - m)*1e4)
        print(f"  {label}: {cycles_with_outlier}/{total} = {100*cycles_with_outlier/max(1,total):.1f}% cycles have right-tail outlier")
        if outlier_size_bps:
            print(f"      mean outlier excess over universe mean: {np.mean(outlier_size_bps):.1f} bps")

    # Does the model EVER pick the right-tail outlier? In H2, for cycles with outlier, how often is the outlier in top-5?
    print(f"\n=== Does the model pick the right-tail outliers? (H2 only) ===")
    sub = d[(d["open_time"]>=H2[0])&(d["open_time"]<=H2[1])]
    hits_top5 = 0; hits_top1 = 0; total_outlier_cyc = 0
    for ot, g in sub.groupby("open_time"):
        if len(g) < 15: continue
        m = g["return_pct"].mean(); s_ = g["return_pct"].std()
        mx = g["return_pct"].max()
        if mx < m + 2*s_: continue
        total_outlier_cyc += 1
        outlier_sym = g.loc[g["return_pct"].idxmax(), "symbol"]
        g_sorted = g.sort_values("pred", ascending=False)
        top5 = g_sorted.head(5)["symbol"].tolist()
        top1 = g_sorted.head(1)["symbol"].tolist()
        if outlier_sym in top5: hits_top5 += 1
        if outlier_sym in top1: hits_top1 += 1
    if total_outlier_cyc:
        # baseline: random pick rate = 5/|universe|
        avg_n = sub.groupby("open_time").size().mean()
        baseline_top5 = 5/avg_n
        baseline_top1 = 1/avg_n
        print(f"  H2 cycles with right-tail outlier: {total_outlier_cyc}")
        print(f"  outlier in top-5 (model): {hits_top5}/{total_outlier_cyc} = {100*hits_top5/total_outlier_cyc:.1f}%  (random baseline {100*baseline_top5:.1f}%)")
        print(f"  outlier in top-1 (model): {hits_top1}/{total_outlier_cyc} = {100*hits_top1/total_outlier_cyc:.1f}%  (random baseline {100*baseline_top1:.1f}%)")
        print(f"  → if hit rate ≈ random, model has NO info on outliers (confirms hypothesis)")
        print(f"  → if hit rate > 2× random, model DOES detect outliers (refutes hypothesis)")

    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__ == "__main__": main()
