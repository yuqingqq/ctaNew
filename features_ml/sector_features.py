"""Sector / cluster momentum features for crypto perp panels.

Adds 3 PIT-correct features to a long-format panel, built from each symbol's
trailing `return_1d` column (which itself is `close.pct_change(288)`, the same
convention used by every other feature in the panel):

  - own_cluster_ret_1d:        mean(return_1d) of own cluster peers (ex-self)
  - relative_to_cluster_1d:    own return_1d minus own_cluster_ret_1d
  - cluster_dispersion_1d:     std(return_1d) of cluster peers (ex-self)

Cluster mapping is loaded from `config/clusters_v1.json`. Symbols not in any
cluster get NaN for these features.

PIT: features at bar t use only `return_1d` values at bar t, which are
themselves computed from close[t] / close[t-288] - 1. Since the model trades
at bar t with target_A spanning [t, t+horizon], no look-ahead.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_cluster_map(clusters_path: str | Path) -> dict[str, str]:
    clusters = json.loads(Path(clusters_path).read_text())
    return {s: c for c, syms in clusters.items() for s in syms}


def add_sector_features(
    panel: pd.DataFrame,
    clusters_path: str | Path,
    src_col: str = "return_1d",
    suffix: str = "1d",
) -> tuple[pd.DataFrame, list[str]]:
    """Add 3 sector features to a long-format panel.

    Args:
        panel: long-format dataframe with at minimum [open_time, symbol, return_1d]
        clusters_path: path to clusters_v1.json
        src_col: name of the trailing per-symbol return column in `panel`
        suffix: suffix used in the produced feature names

    Returns:
        (augmented_panel, [feature_col_names])
    """
    own_col = f"own_cluster_ret_{suffix}"
    rel_col = f"relative_to_cluster_{suffix}"
    disp_col = f"cluster_dispersion_{suffix}"
    feat_cols = [own_col, rel_col, disp_col]

    sym_to_cluster = _load_cluster_map(clusters_path)

    p = panel[["open_time", "symbol", src_col]].copy()
    p["cluster"] = p["symbol"].map(sym_to_cluster)
    p_clustered = p.dropna(subset=["cluster"])

    # Per-(time, cluster) aggregates over peers
    grp = p_clustered.groupby(["open_time", "cluster"])[src_col]
    agg = grp.agg(["sum", "count", lambda s: (s ** 2).sum()]).rename(
        columns={"<lambda_0>": "sum_sq"}
    )
    agg = agg.reset_index()

    merged = p_clustered.merge(agg, on=["open_time", "cluster"], how="left")

    own = merged[src_col].fillna(0.0)
    n_peers = merged["count"] - merged[src_col].notna().astype(int)
    with np.errstate(invalid="ignore", divide="ignore"):
        peer_mean = (merged["sum"] - own) / n_peers.replace(0, np.nan)
        peer_sq_mean = (merged["sum_sq"] - own ** 2) / n_peers.replace(0, np.nan)
        peer_var = peer_sq_mean - peer_mean ** 2
        peer_std = np.sqrt(peer_var.clip(lower=0))

    merged[own_col] = peer_mean
    merged[rel_col] = merged[src_col] - peer_mean
    merged[disp_col] = peer_std

    sector_long = merged[["open_time", "symbol", own_col, rel_col, disp_col]]
    out = panel.merge(sector_long, on=["open_time", "symbol"], how="left")
    return out, feat_cols
