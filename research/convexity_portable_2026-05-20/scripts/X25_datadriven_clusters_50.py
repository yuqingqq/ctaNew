"""X25 — Data-driven clustering on HL-50 panel + universe tests.

Computes hierarchical Ward clustering from 4h-return correlation on HL-50 syms,
explores K=4/5/6/7 cluster counts, picks K=6 for testing.

For K=6 clustering:
- Test each cluster of size >= 7 as a universe
- Test pair combinations
- Test exclusions (drop smallest/weirdest cluster)

Compare to canonical HL-50 +2.01 and X24's hand-crafted clusters.
"""
from __future__ import annotations
import csv, json, sys, time, warnings, importlib.util, gc, resource
from pathlib import Path
import pandas as pd, numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

OUT = REPO / "research/convexity_portable_2026-05-20/results"
CACHE = OUT / "_cache"

spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)
spec_b = importlib.util.spec_from_file_location("x6b",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6b_cohort_fill.py")
x6b = importlib.util.module_from_spec(spec_b); spec_b.loader.exec_module(x6b)


def log_mem(label=""):
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  [MEM {label}] peak_rss={rss_mb:.0f}MB", flush=True)


def compute_data_driven_clusters(panel, n_clusters=6, bar_freq_hours=4):
    """Compute hierarchical Ward clustering from cross-symbol return correlation.

    Returns: dict {cluster_name: [syms]}
    """
    syms = sorted(panel["symbol"].unique())
    print(f"\n=== Computing data-driven clusters for {len(syms)} HL-50 syms ===")

    # Build wide return panel at bar_freq_hours cadence
    panel = panel.copy()
    panel["return_4h_proxy"] = panel["return_pct"]  # 5m return; aggregate later
    # Subsample to 4h-aligned bars (every 48th 5m bar)
    panel = panel[(panel["open_time"].dt.hour % bar_freq_hours == 0)
                  & (panel["open_time"].dt.minute == 0)]
    wide = panel.pivot_table(index="open_time", columns="symbol",
                              values="return_4h_proxy", aggfunc="first")
    # Forward-fill NaN with 0 (treat missing as no return)
    wide = wide.fillna(0)
    print(f"  wide returns: {wide.shape}")

    # Pairwise Pearson correlation
    corr = wide.corr()
    print(f"  correlation matrix: {corr.shape}, mean={corr.values.mean():.3f}, "
          f"diag={corr.values.diagonal().mean():.3f}")

    # Convert to distance, Ward linkage
    dist = 1 - corr.values
    # Make symmetric and zero-diagonal
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    # Squareform for linkage
    from scipy.spatial.distance import squareform
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    clusters = {}
    for sym, lbl in zip(corr.index, labels):
        cname = f"dd_K{n_clusters}_c{lbl}"
        clusters.setdefault(cname, []).append(sym)

    print(f"\n  K={n_clusters} clusters:")
    for k, v in sorted(clusters.items(), key=lambda x: -len(x[1])):
        print(f"    {k}: {len(v):>2} syms — {v[:8]}...")
    return clusters


def get_panel(syms):
    needed = ["symbol", "open_time", "exit_time", "alpha_vs_btc_realized", "return_pct"] + x6.BASE
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                            columns=list(set(needed)))
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[panel["symbol"].isin(syms) & (panel["symbol"] != "BTCUSDT")].copy()
    panel = x6b.build_cohort_fixed(panel)
    panel = x6.build_target_z(panel)
    x6.HEAVY_TAIL.discard("rvol_7d"); x6.HEAVY_TAIL.discard("ret_3d"); x6.HEAVY_TAIL.discard("btc_rvol_7d")
    return panel


def main():
    t0 = time.time()
    print("=== X25 data-driven cluster universes on HL-50 ===\n", flush=True)
    log_mem("start")

    # Load panel to compute clusters
    panel_syms = set(pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                                      columns=["symbol"])["symbol"].unique()) - {"BTCUSDT"}
    HL_50 = sorted(panel_syms)
    full_panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                                  columns=["symbol", "open_time", "return_pct"])
    full_panel["open_time"] = pd.to_datetime(full_panel["open_time"], utc=True)
    full_panel = full_panel[full_panel["symbol"].isin(HL_50)]
    log_mem("after full_panel load")

    # Compute clusterings at K=4, 5, 6
    all_clusterings = {}
    for K in [4, 5, 6]:
        all_clusterings[K] = compute_data_driven_clusters(full_panel, n_clusters=K, bar_freq_hours=4)

    # Save
    out_clusters_path = OUT / "X25_clusters_dd.json"
    with open(out_clusters_path, "w") as f:
        json.dump({f"K{K}": v for K, v in all_clusterings.items()}, f, indent=2)
    print(f"\nSaved clusterings → {out_clusters_path}")

    del full_panel; gc.collect()

    # Build universes from K=6 clustering (most granular)
    K6 = all_clusterings[6]
    print(f"\n=== Building universes from K=6 clusters ===")

    universes = {}
    # Single clusters of size >= 7
    for k, syms in K6.items():
        if len(syms) >= 7:
            universes[f"S_{k}"] = sorted(syms)

    # K=4 clusters (larger, more useful)
    K4 = all_clusterings[4]
    for k, syms in K4.items():
        if len(syms) >= 7:
            universes[f"L_{k}"] = sorted(syms)

    # Exclude smallest cluster from HL-50
    smallest_K6 = min(K6.items(), key=lambda x: len(x[1]))
    universes["EX_smallest_K6"] = sorted(set(HL_50) - set(smallest_K6[1]))

    # Anti-correlated pairs: combine 2 most distant clusters
    # (use largest 2 K6 clusters for diversification)
    K6_sorted = sorted(K6.items(), key=lambda x: -len(x[1]))
    if len(K6_sorted) >= 2:
        universes["P_largest2"] = sorted(set(K6_sorted[0][1]) | set(K6_sorted[1][1]))

    # Pair clusters (2 smaller ones combined for diversification)
    small_clusters = [v for k, v in K6.items() if 3 <= len(v) <= 6]
    if len(small_clusters) >= 2:
        universes["P_small_clusters"] = sorted(set().union(*small_clusters))

    print(f"\n=== {len(universes)} data-driven cluster universes ===")
    for k, v in universes.items():
        print(f"  {k}: {len(v):>3} syms — {v[:5]}...")

    feats = x6.BASE + x6.COHORT_EXTRAS
    results = []
    for u_name, u_syms in universes.items():
        tf = time.time()
        log_mem(f"before {u_name}")
        print(f"\n[{u_name}] {len(u_syms)} syms", flush=True)
        try:
            panel = get_panel(set(u_syms))
            folds = x6.get_folds(panel)
            if len(folds) == 0:
                print(f"  no folds, skipping"); continue
            apd = x6.train_per_sym_ridge(panel, folds, feats, label=u_name)
            pred_path = CACHE / f"x25_{u_name}_preds.parquet"
            apd.to_parquet(pred_path, index=False)
            ic = float(apd["pred"].corr(apd["alpha_A"]))
            print(f"  trained: {len(apd):,} rows, IC={ic:+.4f} [{time.time()-tf:.0f}s]", flush=True)
        except Exception as e:
            print(f"  TRAIN ERR: {e}")
            results.append({"universe": u_name, "n_syms": len(u_syms), "error": str(e)})
            continue
        m = x6.run_sleeve_on_preds(pred_path, f"x25_{u_name}")
        row = {"universe": u_name, "n_syms": len(u_syms), "train_ic": round(ic, 4),
               "syms": ",".join(u_syms), **m}
        results.append(row)
        if "sharpe" in m:
            print(f"  sleeve: Sharpe {m['sharpe']:+.2f} folds {m.get('folds_pos','?')} "
                  f"conc {m.get('concentration','?')} PnL {m.get('totPnL','?')}", flush=True)
        del panel, apd; gc.collect()

    keys = ["universe", "n_syms", "train_ic", "sharpe", "ci_lo", "ci_hi",
            "totPnL", "maxDD", "folds_pos", "concentration", "syms", "error"]
    out_csv = OUT / "X25_datadriven_cluster_universes.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"\nSaved {len(results)} → {out_csv} [{time.time()-t0:.0f}s]")
    print(f"\nReference: HL-50 canonical = +2.01")
    for r in results:
        if "sharpe" in r:
            print(f"  {r['universe']:<26} ({r['n_syms']:>3}) Sharpe={r['sharpe']:+.2f} "
                  f"folds={r.get('folds_pos','?')} conc={r.get('concentration','?')}")


if __name__ == "__main__":
    main()
