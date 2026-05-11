"""Sector-bucketed portfolio construction for v6_clean.

Phase B of the universe-expansion architecture change. Replaces global
top-K / bot-K with cluster-bucketed top-K_c / bot-K_c per cluster, then
aggregates. Mechanism: bound rank competition within-cluster so PM gate's
persistence test stays meaningful as universe grows.

Public API mirrors `select_portfolio` and `pm_gate_filter` so the rest of
the cycle flow (β-neutral scaling, cost accounting) stays unchanged.
"""
from __future__ import annotations
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CLUSTERS_PATH = REPO / "config" / "clusters_v1.json"


def load_clusters(path: Path = DEFAULT_CLUSTERS_PATH) -> dict[str, list[str]]:
    """Load cluster definitions from JSON config."""
    with open(path) as f:
        return json.load(f)


def get_sym_to_cluster(clusters: dict[str, list[str]]) -> dict[str, str]:
    """Reverse mapping: symbol → cluster name."""
    out = {}
    for c, syms in clusters.items():
        for s in syms:
            out[s] = c
    return out


def select_portfolio_clustered(
    preds: pd.DataFrame,
    clusters: dict[str, list[str]],
    k_per_cluster_long: dict[str, int] | int = 1,
    k_per_cluster_short: dict[str, int] | int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float, dict]:
    """Per-cluster top-K_c long / bot-K_c short selection.

    Selection rule (each cluster, independently):
      - Filter preds to symbols in cluster
      - Take top-K_c_long by pred desc → long picks for this cluster
      - Take bot-K_c_short by pred asc → short picks for this cluster
      - If cluster has < (K_c_long + K_c_short) symbols, both legs select
        from same pool but no symbol can be in both (greedy: longs first by
        pred desc, then shorts by pred asc from remaining)

    Aggregate: union all cluster picks. Per-name weight = scale_L / total_K
    (matches research per-name=1/K_target convention validated in audit).

    Args:
      preds: DataFrame with columns [symbol, pred, beta_short_vs_bk, ...]
      clusters: {cluster_name: [symbols]}
      k_per_cluster_long: int (uniform) or dict {cluster: K} (per-cluster)
      k_per_cluster_short: same

    Returns:
      top_df:   DataFrame of long-leg picks (≤ Σ K_c_long rows, sorted by pred desc)
      bot_df:   DataFrame of short-leg picks (≤ Σ K_c_short rows, sorted by pred asc)
      scale_L:  β-neutral scale factor for long leg (clipped [0.5, 1.5])
      scale_S:  β-neutral scale factor for short leg
      info:     dict with per-cluster counts
    """
    g = preds.dropna(subset=["pred"]).copy()
    sym_to_cluster = get_sym_to_cluster(clusters)

    # Normalize K spec to dict
    if isinstance(k_per_cluster_long, int):
        k_long = {c: k_per_cluster_long for c in clusters}
    else:
        k_long = {c: k_per_cluster_long.get(c, 0) for c in clusters}
    if isinstance(k_per_cluster_short, int):
        k_short = {c: k_per_cluster_short for c in clusters}
    else:
        k_short = {c: k_per_cluster_short.get(c, 0) for c in clusters}

    long_syms, short_syms = set(), set()
    cluster_info = {}
    for c, members in clusters.items():
        in_universe = g[g["symbol"].isin(members)]
        if in_universe.empty:
            cluster_info[c] = {"n_members": 0, "n_long": 0, "n_short": 0}
            continue
        kl, ks = k_long.get(c, 0), k_short.get(c, 0)
        sorted_c = in_universe.sort_values("pred", ascending=False)
        long_picks = sorted_c.head(kl)["symbol"].tolist()
        # Short picks: from remaining names (exclude long picks), tail by pred
        remaining = sorted_c[~sorted_c["symbol"].isin(long_picks)]
        short_picks = remaining.tail(ks)["symbol"].tolist()
        long_syms.update(long_picks)
        short_syms.update(short_picks)
        cluster_info[c] = {
            "n_members": len(in_universe),
            "n_long": len(long_picks),
            "n_short": len(short_picks),
        }

    top_df = g[g["symbol"].isin(long_syms)].sort_values("pred", ascending=False)
    bot_df = g[g["symbol"].isin(short_syms)].sort_values("pred", ascending=True)

    # β-neutral scaling (mirrors research/select_portfolio convention)
    if top_df.empty or bot_df.empty:
        scale_L, scale_S = 1.0, 1.0
    else:
        beta_L = float(top_df["beta_short_vs_bk"].mean())
        beta_S = float(bot_df["beta_short_vs_bk"].mean())
        if beta_L < 0.1 or beta_S < 0.1 or (beta_L + beta_S) < 0.3:
            scale_L, scale_S = 1.0, 1.0
        else:
            denom = beta_L + beta_S
            scale_L = float(np.clip(2.0 * beta_S / denom, 0.5, 1.5))
            scale_S = float(np.clip(2.0 * beta_L / denom, 0.5, 1.5))

    info = {
        "n_long_total": len(long_syms),
        "n_short_total": len(short_syms),
        "k_target_long_total": sum(k_long.values()),
        "k_target_short_total": sum(k_short.values()),
        "per_cluster": cluster_info,
    }
    return top_df, bot_df, scale_L, scale_S, info


def select_portfolio_capped(
    preds: pd.DataFrame,
    clusters: dict[str, list[str]],
    top_k: int = 7,
    max_per_cluster: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float, dict]:
    """Cluster-CAPPED top-K selection.

    Sort all names globally by pred, take top-K respecting cluster caps:
      - Walk down the global rank list
      - Add a name if cluster_count[name's cluster] < max_per_cluster
      - Skip otherwise
      - Continue until K names accepted (or universe exhausted)

    Preserves global ranking discipline while preventing cluster concentration.
    Same logic in reverse for short leg.
    """
    g = preds.dropna(subset=["pred"]).copy()
    sym_to_c = get_sym_to_cluster(clusters)

    # Long leg: greedy top-K with cluster cap
    sorted_desc = g.sort_values("pred", ascending=False)
    long_picks = []
    cluster_count = {c: 0 for c in clusters}
    for _, row in sorted_desc.iterrows():
        if len(long_picks) >= top_k:
            break
        c = sym_to_c.get(row["symbol"])
        if c is None or cluster_count[c] >= max_per_cluster:
            continue
        long_picks.append(row["symbol"])
        cluster_count[c] += 1

    # Short leg: greedy bot-K (sort ascending), exclude long picks, cluster cap
    sorted_asc = g.sort_values("pred", ascending=True)
    short_picks = []
    cluster_count_s = {c: 0 for c in clusters}
    for _, row in sorted_asc.iterrows():
        if len(short_picks) >= top_k:
            break
        if row["symbol"] in long_picks:
            continue
        c = sym_to_c.get(row["symbol"])
        if c is None or cluster_count_s[c] >= max_per_cluster:
            continue
        short_picks.append(row["symbol"])
        cluster_count_s[c] += 1

    top_df = g[g["symbol"].isin(long_picks)].sort_values("pred", ascending=False)
    bot_df = g[g["symbol"].isin(short_picks)].sort_values("pred", ascending=True)

    if top_df.empty or bot_df.empty:
        scale_L, scale_S = 1.0, 1.0
    else:
        beta_L = float(top_df["beta_short_vs_bk"].mean())
        beta_S = float(bot_df["beta_short_vs_bk"].mean())
        if beta_L < 0.1 or beta_S < 0.1 or (beta_L + beta_S) < 0.3:
            scale_L, scale_S = 1.0, 1.0
        else:
            denom = beta_L + beta_S
            scale_L = float(np.clip(2.0 * beta_S / denom, 0.5, 1.5))
            scale_S = float(np.clip(2.0 * beta_L / denom, 0.5, 1.5))

    info = {
        "n_long_total": len(long_picks),
        "n_short_total": len(short_picks),
        "long_cluster_count": cluster_count,
        "short_cluster_count": cluster_count_s,
    }
    return top_df, bot_df, scale_L, scale_S, info


def pm_gate_filter_capped(
    preds: pd.DataFrame,
    clusters: dict[str, list[str]],
    history: list[dict],          # global past band sets {long: [...], short: [...]}
    prev_long_syms: set,
    prev_short_syms: set,
    top_k: int = 7,
    max_per_cluster: int = 2,
    M_cycles: int = 2,
    band_mult: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float, dict]:
    """Cluster-CAPPED PM gate.

    Like global pm_gate_filter but enforces max_per_cluster cap during selection.
    History is GLOBAL (band-K of full universe), not per-cluster, since we're
    using global ranking with caps.
    """
    g = preds.dropna(subset=["pred"]).copy()
    sym_to_c = get_sym_to_cluster(clusters)
    band_k = max(top_k, int(round(band_mult * top_k)))

    if len(g) < 2 * top_k + 1:
        return g.head(0), g.head(0), 1.0, 1.0, {
            "K_long_actual": 0, "K_short_actual": 0,
            "n_rejected_long": 0, "n_rejected_short": 0,
        }

    # Past M-1 cycles' top-band-K and bot-band-K (global)
    past_long_bands = [set(h.get("long", [])) for h in history[-(M_cycles - 1):]] if M_cycles >= 2 else []
    past_short_bands = [set(h.get("short", [])) for h in history[-(M_cycles - 1):]] if M_cycles >= 2 else []
    history_active = len(history) >= M_cycles - 1 and M_cycles >= 2

    # Long leg: walk top-down, accept if (held OR persistent) AND cluster cap not exceeded
    sorted_desc = g.sort_values("pred", ascending=False)
    long_picks = []
    cluster_count = {c: 0 for c in clusters}
    rejected_long = 0
    for _, row in sorted_desc.iterrows():
        if len(long_picks) >= top_k:
            break
        s = row["symbol"]
        c = sym_to_c.get(s)
        if c is None or cluster_count[c] >= max_per_cluster:
            continue
        # Persistence check (only for new entries)
        is_held = s in prev_long_syms
        if is_held:
            long_picks.append(s); cluster_count[c] += 1
            continue
        if history_active:
            persistent = all(s in p for p in past_long_bands)
            if persistent:
                long_picks.append(s); cluster_count[c] += 1
            else:
                rejected_long += 1
                continue
        else:
            long_picks.append(s); cluster_count[c] += 1

    # Short leg: walk bottom-up, similar logic
    sorted_asc = g.sort_values("pred", ascending=True)
    short_picks = []
    cluster_count_s = {c: 0 for c in clusters}
    rejected_short = 0
    for _, row in sorted_asc.iterrows():
        if len(short_picks) >= top_k:
            break
        s = row["symbol"]
        if s in long_picks:
            continue
        c = sym_to_c.get(s)
        if c is None or cluster_count_s[c] >= max_per_cluster:
            continue
        is_held = s in prev_short_syms
        if is_held:
            short_picks.append(s); cluster_count_s[c] += 1
            continue
        if history_active:
            persistent = all(s in p for p in past_short_bands)
            if persistent:
                short_picks.append(s); cluster_count_s[c] += 1
            else:
                rejected_short += 1
                continue
        else:
            short_picks.append(s); cluster_count_s[c] += 1

    top_df = g[g["symbol"].isin(long_picks)].sort_values("pred", ascending=False)
    bot_df = g[g["symbol"].isin(short_picks)].sort_values("pred", ascending=True)

    if top_df.empty or bot_df.empty:
        scale_L, scale_S = 1.0, 1.0
    else:
        beta_L = float(top_df["beta_short_vs_bk"].mean())
        beta_S = float(bot_df["beta_short_vs_bk"].mean())
        if beta_L < 0.1 or beta_S < 0.1 or (beta_L + beta_S) < 0.3:
            scale_L, scale_S = 1.0, 1.0
        else:
            denom = beta_L + beta_S
            scale_L = float(np.clip(2.0 * beta_S / denom, 0.5, 1.5))
            scale_S = float(np.clip(2.0 * beta_L / denom, 0.5, 1.5))

    # Current band-K (global) for history append by caller
    bk = min(band_k, len(g))
    sg = g.sort_values("pred")
    current_top_band = sg.tail(bk)["symbol"].tolist()
    current_bot_band = sg.head(bk)["symbol"].tolist()

    info = {
        "K_long_actual": len(long_picks),
        "K_short_actual": len(short_picks),
        "n_rejected_long": rejected_long,
        "n_rejected_short": rejected_short,
        "long_cluster_count": cluster_count,
        "short_cluster_count": cluster_count_s,
        "current_top_band": current_top_band,
        "current_bot_band": current_bot_band,
    }
    return top_df, bot_df, scale_L, scale_S, info


def _compute_band_per_cluster(preds: pd.DataFrame,
                                clusters: dict[str, list[str]],
                                band_k_per_cluster: dict[str, int] | int) -> dict[str, dict]:
    """For each cluster, compute current cycle's top-band-K and bot-band-K
    name sets. Used to update PM history per-cluster."""
    g = preds.dropna(subset=["pred"]).copy()
    if isinstance(band_k_per_cluster, int):
        bks = {c: band_k_per_cluster for c in clusters}
    else:
        bks = {c: band_k_per_cluster.get(c, 1) for c in clusters}
    out = {}
    for c, members in clusters.items():
        sub = g[g["symbol"].isin(members)]
        if sub.empty:
            out[c] = {"long": [], "short": []}
            continue
        sorted_c = sub.sort_values("pred")
        bk = min(bks.get(c, 1), len(sub))
        out[c] = {
            "long": sorted_c.tail(bk)["symbol"].tolist(),
            "short": sorted_c.head(bk)["symbol"].tolist(),
        }
    return out


def pm_gate_filter_clustered(
    preds: pd.DataFrame,
    clusters: dict[str, list[str]],
    history_per_cluster: dict[str, list[dict]],   # {cluster: [{long, short}, ...]}
    prev_long_per_cluster: dict[str, set],         # {cluster: {syms held last cycle}}
    prev_short_per_cluster: dict[str, set],
    k_per_cluster_long: int = 1,
    k_per_cluster_short: int = 1,
    M_cycles: int = 2,
    band_mult: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float, dict]:
    """Cluster-aware PM_M2_b1 entry gate.

    For each cluster independently:
      - Identify top-K_long candidates (top-1 by pred within cluster)
      - Held names that remain in cluster's top-K auto-keep
      - NEW entries require persistence in past M-1 cycles' band-K of THIS CLUSTER
      - K can shrink within cluster (allow K_actual = 0 → cluster contributes nothing)
    Same logic on short side. Aggregate across clusters.

    History is checked PER CLUSTER, not globally.
    """
    g = preds.dropna(subset=["pred"]).copy()
    band_k_per_cluster = max(k_per_cluster_long, int(round(band_mult * k_per_cluster_long)))
    # band_k uniform for v1; could be per-cluster later

    cluster_results = {}
    long_syms, short_syms = set(), set()
    for c, members in clusters.items():
        sub = g[g["symbol"].isin(members)]
        if sub.empty or len(sub) < k_per_cluster_long + k_per_cluster_short:
            cluster_results[c] = {"K_long": 0, "K_short": 0, "rejected_long": 0, "rejected_short": 0}
            continue

        # Current cluster's top-K_long and bot-K_short
        sorted_c = sub.sort_values("pred")
        cand_long = set(sorted_c.tail(k_per_cluster_long)["symbol"])
        # Short: tail/head logic — exclude long picks
        cand_long_excluded = sorted_c[~sorted_c["symbol"].isin(cand_long)]
        cand_short = set(cand_long_excluded.head(k_per_cluster_short)["symbol"])

        prev_long = prev_long_per_cluster.get(c, set())
        prev_short = prev_short_per_cluster.get(c, set())
        history_c = history_per_cluster.get(c, [])

        # Held names that remain in current cluster's top-K auto-keep
        new_long_c = cand_long & prev_long
        new_short_c = cand_short & prev_short
        rejected_long = 0
        rejected_short = 0

        if len(history_c) >= M_cycles - 1 and M_cycles >= 2:
            past_long = [set(h.get("long", [])) for h in history_c[-(M_cycles - 1):]]
            past_short = [set(h.get("short", [])) for h in history_c[-(M_cycles - 1):]]
            for s in cand_long - prev_long:
                if all(s in p for p in past_long):
                    new_long_c.add(s)
                else:
                    rejected_long += 1
            for s in cand_short - prev_short:
                if all(s in p for p in past_short):
                    new_short_c.add(s)
                else:
                    rejected_short += 1
        else:
            new_long_c |= cand_long
            new_short_c |= cand_short

        # Cap to k_per_cluster_long / short (in case held + persistent > k)
        if len(new_long_c) > k_per_cluster_long:
            new_long_c = set(g[g["symbol"].isin(new_long_c)].nlargest(
                k_per_cluster_long, "pred")["symbol"])
        if len(new_short_c) > k_per_cluster_short:
            new_short_c = set(g[g["symbol"].isin(new_short_c)].nsmallest(
                k_per_cluster_short, "pred")["symbol"])

        long_syms.update(new_long_c)
        short_syms.update(new_short_c)
        cluster_results[c] = {
            "K_long": len(new_long_c),
            "K_short": len(new_short_c),
            "rejected_long": rejected_long,
            "rejected_short": rejected_short,
        }

    # Aggregate
    top_df = g[g["symbol"].isin(long_syms)].sort_values("pred", ascending=False)
    bot_df = g[g["symbol"].isin(short_syms)].sort_values("pred", ascending=True)

    if top_df.empty or bot_df.empty:
        scale_L, scale_S = 1.0, 1.0
    else:
        beta_L = float(top_df["beta_short_vs_bk"].mean())
        beta_S = float(bot_df["beta_short_vs_bk"].mean())
        if beta_L < 0.1 or beta_S < 0.1 or (beta_L + beta_S) < 0.3:
            scale_L, scale_S = 1.0, 1.0
        else:
            denom = beta_L + beta_S
            scale_L = float(np.clip(2.0 * beta_S / denom, 0.5, 1.5))
            scale_S = float(np.clip(2.0 * beta_L / denom, 0.5, 1.5))

    # Compute current band sets per cluster for history append
    current_band_per_cluster = _compute_band_per_cluster(
        preds, clusters, band_k_per_cluster=band_k_per_cluster
    )

    info = {
        "K_long_total": len(long_syms),
        "K_short_total": len(short_syms),
        "K_target_total": k_per_cluster_long * len(clusters),
        "per_cluster": cluster_results,
        "current_band_per_cluster": current_band_per_cluster,
    }
    return top_df, bot_df, scale_L, scale_S, info
