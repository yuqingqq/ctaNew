"""Add Variant D (cluster-residual target) to the panel_variants.parquet.

Computes data-driven hierarchical clusters on alpha correlation across the
51-name universe, then for each name builds basket_D_fwd = mean of its
cluster mates' forward return (excluding self). target_D = z-scored
alpha_D = my_fwd − basket_D_fwd.

Singletons (clusters of size 1) get basket_D = global basket as fallback,
so they don't have a degenerate target.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants.parquet"
HORIZON = 48
N_CLUSTERS = 8   # from Phase 1.5 EDA: 1 main + ~7 singletons


def main():
    print(f"Loading panel...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    print(f"  {len(panel):,} rows × {panel['symbol'].nunique()} syms", flush=True)

    # === Step 1: hierarchical clustering on full-sample alpha correlation ===
    print(f"Computing alpha correlation matrix...", flush=True)
    t0 = time.time()
    alpha_pivot = panel.pivot_table(index="open_time", columns="symbol",
                                      values="alpha_A", aggfunc="first")
    # Keep only timestamps with most names present
    keep_t = alpha_pivot.notna().sum(axis=1) >= 0.7 * alpha_pivot.shape[1]
    alpha_pivot = alpha_pivot.loc[keep_t]
    corr = alpha_pivot.corr()
    print(f"  corr matrix {corr.shape}, took {time.time()-t0:.0f}s", flush=True)

    print(f"Hierarchical clustering, K={N_CLUSTERS}...", flush=True)
    from scipy.cluster.hierarchy import linkage, fcluster
    dist = (1 - corr).clip(lower=0).to_numpy().copy()
    np.fill_diagonal(dist, 0)
    iu = np.triu_indices(dist.shape[0], k=1)
    Z = linkage(dist[iu], method="average")
    labels = fcluster(Z, t=N_CLUSTERS, criterion="maxclust")
    cluster_map = dict(zip(corr.index, labels))
    # Show sizes
    sizes = pd.Series(labels).value_counts().sort_values(ascending=False)
    print(f"  Cluster sizes: {dict(zip(sizes.index, sizes.values))}", flush=True)
    for c in sizes.index:
        members = [s for s, l in cluster_map.items() if l == c]
        print(f"    cluster {c} ({len(members)}): {sorted(members)[:8]}{' ...' if len(members) > 8 else ''}",
              flush=True)

    # === Step 2: build cluster basket (for each timestamp × cluster, mean my_fwd) ===
    print(f"Computing cluster baskets per (open_time × cluster)...", flush=True)
    t0 = time.time()
    panel["cluster"] = panel["symbol"].map(cluster_map)
    cluster_grp = panel.groupby(["open_time", "cluster"])["return_pct"].agg(["sum", "count"])
    cluster_grp.columns = ["cluster_sum", "cluster_n"]
    cluster_grp = cluster_grp.reset_index()
    panel = panel.merge(cluster_grp, on=["open_time", "cluster"], how="left")
    # basket_D for name i = (cluster_sum - my_fwd) / (cluster_n - 1) — leave-self-out
    panel["basket_D_fwd"] = (panel["cluster_sum"] - panel["return_pct"]) / (panel["cluster_n"] - 1).replace(0, np.nan)
    # For singletons (cluster_n == 1), fallback to overall basket A
    singleton_mask = panel["cluster_n"] == 1
    panel.loc[singleton_mask, "basket_D_fwd"] = panel.loc[singleton_mask, "basket_A_fwd"]
    panel = panel.drop(columns=["cluster_sum", "cluster_n"])
    print(f"  built in {time.time()-t0:.0f}s", flush=True)

    # === Step 3: alpha_D and target_D ===
    print(f"Computing alpha_D + target_D...", flush=True)
    t0 = time.time()
    panel["alpha_D"] = panel["return_pct"] - panel["basket_D_fwd"]
    out = []
    for s, g in panel.groupby("symbol", sort=False):
        a = g["alpha_D"]
        rmean = a.expanding(min_periods=288).mean().shift(HORIZON)
        rstd = a.rolling(288 * 7, min_periods=288).std().shift(HORIZON)
        z = (a - rmean) / rstd.replace(0, np.nan)
        out.append(z.rename("target_D"))
    panel["target_D"] = pd.concat(out).sort_index()
    ar = panel["alpha_D"].dropna()
    tg = panel["target_D"].dropna()
    print(f"  alpha_D: mean={ar.mean():+.6f}, std={ar.std():.6f}", flush=True)
    print(f"  target_D: mean={tg.mean():+.4f}, std={tg.std():.4f}, n={len(tg):,}", flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)

    # === Save ===
    panel.to_parquet(PANEL_PATH, index=False)
    print(f"  saved → {PANEL_PATH}", flush=True)
    print(f"  cluster_map: {cluster_map}", flush=True)
    pd.DataFrame([{"symbol": s, "cluster": c} for s, c in cluster_map.items()]
                 ).to_csv(REPO / "outputs/vBTC_features/cluster_map.csv", index=False)


if __name__ == "__main__":
    main()
