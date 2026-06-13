"""Phase G1: Data-driven clustering of the 111-symbol universe.

Method:
  1. Use return_1d (close.pct_change(288)) for each symbol over time.
  2. Subsample to daily bars (every 288 bars = 1 day) for tractable correlation.
  3. Compute pairwise Pearson correlation on the overlap window where
     both symbols are observed (skipna=True per pair).
  4. Hierarchical clustering (Ward linkage on 1-correlation distance).
  5. Cut dendrogram at multiple K values, report within/between cohesion.
  6. Pick best K, save assignment to config/clusters_data_driven_v1.json.

Output:
  - config/clusters_data_driven_v1.json (best clustering)
  - outputs/vBTC_clusters_dd/cohesion_by_k.csv
  - outputs/vBTC_clusters_dd/cluster_assignment_k{K}.csv
  - outputs/vBTC_clusters_dd/correlation_matrix.parquet
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
PANEL_PATH = REPO / "outputs/vBTC_features_expanded/panel_variants_with_funding.parquet"
OUT_DIR = REPO / "outputs/vBTC_clusters_dd"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_correlation_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """Wide returns matrix → pairwise correlation, skip pairwise NaN."""
    print(f"  panel: {len(panel):,} rows, {panel['symbol'].nunique()} syms", flush=True)
    panel = panel.copy()
    panel["bar_idx"] = panel.groupby("symbol").cumcount()
    panel = panel[panel["bar_idx"] % 48 == 0]
    wide = panel.pivot(index="open_time", columns="symbol", values="return_1d")
    print(f"  4h-spaced returns wide: {wide.shape}", flush=True)
    nn = wide.notna().mean()
    print(f"  non-null % per symbol (min/median/max): "
          f"{nn.min():.0%}/{nn.median():.0%}/{nn.max():.0%}", flush=True)
    keep_syms = nn[nn >= 0.20].index.tolist()
    print(f"  keeping syms with >=20% history: {len(keep_syms)}/{wide.shape[1]}", flush=True)
    wide = wide[keep_syms]
    corr = wide.corr(method="pearson", min_periods=100)
    nan_pairs = corr.isna().sum().sum() // 2
    print(f"  pairwise corr matrix: {corr.shape}, NaN pairs (after diag): {nan_pairs}", flush=True)
    return corr


def cohesion_metrics(corr: pd.DataFrame, assignment: dict[str, str]) -> dict:
    syms = list(corr.index)
    n = len(syms)
    within_sum = 0.0; within_n = 0
    between_sum = 0.0; between_n = 0
    for i in range(n):
        ci = assignment.get(syms[i])
        if ci is None: continue
        for j in range(i + 1, n):
            cj = assignment.get(syms[j])
            if cj is None: continue
            r = corr.iloc[i, j]
            if np.isnan(r): continue
            if ci == cj:
                within_sum += r; within_n += 1
            else:
                between_sum += r; between_n += 1
    within_mean = within_sum / max(within_n, 1)
    between_mean = between_sum / max(between_n, 1)
    return {
        "within_mean": within_mean,
        "between_mean": between_mean,
        "separation": within_mean - between_mean,
        "n_within_pairs": within_n,
        "n_between_pairs": between_n,
    }


def per_cluster_cohesion(corr: pd.DataFrame, assignment: dict[str, str]) -> pd.DataFrame:
    syms = list(corr.index)
    by_c = {}
    for s in syms:
        c = assignment.get(s)
        if c is None: continue
        by_c.setdefault(c, []).append(s)
    rows = []
    for c, members in sorted(by_c.items()):
        n = len(members)
        if n < 2: continue
        within = []
        for i in range(n):
            for j in range(i + 1, n):
                r = corr.loc[members[i], members[j]]
                if not np.isnan(r):
                    within.append(r)
        non_members = [s for s in syms if s not in members]
        between = []
        for m in members:
            for o in non_members:
                r = corr.loc[m, o]
                if not np.isnan(r):
                    between.append(r)
        rows.append({
            "cluster": c, "n_syms": n,
            "within_mean": float(np.mean(within)),
            "between_mean": float(np.mean(between)),
            "separation": float(np.mean(within)) - float(np.mean(between)),
            "members": ",".join(sorted(members)[:5]) + ("..." if n > 5 else ""),
        })
    return pd.DataFrame(rows).sort_values("separation", ascending=False)


def main():
    print("=== Phase G1: Data-driven clustering of 111-symbol universe ===\n", flush=True)

    panel = pd.read_parquet(PANEL_PATH, columns=["open_time", "symbol", "return_1d"])
    corr = compute_correlation_matrix(panel)
    corr.to_parquet(OUT_DIR / "correlation_matrix.parquet")
    print(f"  saved correlation matrix: {corr.shape}", flush=True)

    # Build distance matrix from correlation. Use sqrt(2*(1-corr)) for proper metric.
    syms = list(corr.index)
    n = len(syms)
    print(f"\n  Correlation distribution: min={corr.values[np.triu_indices(n, 1)].min():.3f}, "
          f"median={np.nanmedian(corr.values[np.triu_indices(n, 1)]):.3f}, "
          f"max={corr.values[np.triu_indices(n, 1)].max():.3f}", flush=True)

    # Fill NaN in correlation matrix with overall mean (for hierarchical linkage)
    overall_mean = np.nanmean(corr.values[np.triu_indices(n, 1)])
    corr_filled = corr.fillna(overall_mean).copy()
    corr_arr = corr_filled.values.copy()
    np.fill_diagonal(corr_arr, 1.0)
    dist = np.sqrt(2 * (1 - corr_arr).clip(min=0))
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")

    # Compare against hand-crafted scheme
    handc = json.load(open(REPO / "config/clusters_v1.json"))
    handc_map = {s: c for c, ss in handc.items() for s in ss}
    handc_only_111 = {s: handc_map.get(s) for s in syms if s in handc_map}
    handc_coverage = sum(1 for s in syms if s in handc_map)
    if handc_coverage > 0:
        m_hc = cohesion_metrics(corr, handc_only_111)
        print(f"\n--- Hand-crafted scheme (coverage {handc_coverage}/{n}) ---", flush=True)
        print(f"  within={m_hc['within_mean']:+.4f}  between={m_hc['between_mean']:+.4f}  "
              f"separation={m_hc['separation']:+.4f}", flush=True)

    # Cut dendrogram at various K
    print(f"\n--- Data-driven K-sweep ---", flush=True)
    print(f"  {'K':>3}  {'within':>8}  {'between':>8}  {'separation':>8}  "
          f"{'min_cluster':>10}  {'max_cluster':>10}", flush=True)
    rows = []
    best = None
    for k in [5, 6, 7, 8, 10, 12, 15, 20]:
        labels = fcluster(Z, k, criterion="maxclust")
        assignment = {syms[i]: f"dd_{labels[i]:02d}" for i in range(n)}
        m = cohesion_metrics(corr, assignment)
        from collections import Counter
        szs = Counter(assignment.values())
        m["K"] = k
        m["min_cluster"] = min(szs.values())
        m["max_cluster"] = max(szs.values())
        m["n_clusters_>=3"] = sum(1 for v in szs.values() if v >= 3)
        rows.append(m)
        print(f"  {k:>3}  {m['within_mean']:>+8.4f}  {m['between_mean']:>+8.4f}  "
              f"{m['separation']:>+8.4f}  {m['min_cluster']:>10}  {m['max_cluster']:>10}",
              flush=True)
        if (best is None or m["separation"] > best["separation"]) and m["min_cluster"] >= 3:
            best = {**m, "assignment": assignment, "K": k}

    pd.DataFrame(rows).to_csv(OUT_DIR / "cohesion_by_k.csv", index=False)

    # Use best K to write final clustering
    if best is None:
        print("  no K satisfied min_cluster>=3", flush=True)
        return
    K = best["K"]
    print(f"\n  Best K (max separation, min_cluster>=3): K={K}, sep={best['separation']:+.4f}",
          flush=True)

    # Per-cluster breakdown
    print(f"\n--- Per-cluster cohesion at K={K} ---", flush=True)
    per_c = per_cluster_cohesion(corr, best["assignment"])
    print(per_c.to_string(index=False), flush=True)
    per_c.to_csv(OUT_DIR / f"cluster_assignment_k{K}.csv", index=False)

    # Inverse map → JSON
    inv = {}
    for s, c in best["assignment"].items():
        inv.setdefault(c, []).append(s)
    # Sort each cluster's members
    inv = {c: sorted(members) for c, members in sorted(inv.items())}
    out_json = REPO / "config/clusters_data_driven_v1.json"
    out_json.write_text(json.dumps(inv, indent=2))
    print(f"\n  saved cluster JSON → {out_json}", flush=True)

    # Show each cluster
    print(f"\n--- Cluster members (K={K}) ---", flush=True)
    for c in sorted(inv.keys()):
        members = inv[c]
        print(f"  {c} ({len(members)}): {', '.join(members)}", flush=True)


if __name__ == "__main__":
    main()
