"""Phase 2: Validate + prune the 36-feature v3 candidate pool to 18-24.

Per docs/vBTC_V3_FEATURE_PLAN.md:
  1. Per-feature cross-sectional IC vs alpha_beta (entry-cadence sampled)
  2. Per-feature per-symbol time-series IC distribution
  3. Equal-weight + ridge block composites
  4. Pairwise correlation (100k sample)
  5. NaN rate, per-symbol variance, PIT sanity
  6. Apply pruning rules → save final WINNER_BTC_v3 list
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
PANEL = REPO / "outputs/vBTC_features_btc_v3/panel_v3.parquet"
OUT = REPO / "outputs/vBTC_features_btc_v3"
OUT.mkdir(parents=True, exist_ok=True)

# Per Phase 1: microstructure (block F) dropped because only 25/51 syms have data.
# v3 candidate pool = 36 features across 6 blocks (A,B,C,D,E,G).
BLOCKS = {
    "A_liquidity": ["log_dollar_volume_7d", "log_dollar_volume_30d",
                     "volume_stability_30d", "amihud_illiq_30d",
                     "roll_spread_proxy_30d", "turnover_volatility_30d"],
    "B_btc_relationship": ["beta_btc_30d", "beta_btc_90d", "beta_btc_180d",
                            "beta_btc_instability", "corr_btc_30d",
                            "corr_btc_90d", "corr_breakdown"],
    "C_resid_behavior": ["resid_vol_7d", "resid_vol_30d", "resid_vol_90d",
                          "resid_skew_30d", "resid_kurt_30d",
                          "resid_jump_count_30d", "resid_autocorr_1d",
                          "resid_reversal_score_7d", "resid_trend_score_30d"],
    "D_trend_anchor": ["dist_from_30d_high", "dist_from_90d_high",
                        "dist_from_365d_high", "multi_horizon_trend_score",
                        "volume_confirmed_trend_score"],
    "E_funding": ["funding_mean_7d", "funding_mean_30d", "funding_z_30d",
                   "funding_persistence_7d", "funding_abs_30d",
                   "funding_sign_streak"],
    "G_process_fp": ["idio_skew_1d", "idio_kurt_1d", "idio_max_abs_12b"],
}
ALL_FEATS = [f for blk in BLOCKS.values() for f in blk]
assert len(ALL_FEATS) == 36, f"expected 36, got {len(ALL_FEATS)}"

# Entry-cadence: every 48 bars (4h)
ENTRY_STRIDE = 48
MIN_CYCLE_SYMBOLS = 10  # minimum symbols per cycle for IC


def per_cycle_ic(panel, feat, target="alpha_beta"):
    """Cross-sectional Spearman IC per entry-cadence cycle."""
    samp = panel.dropna(subset=[feat, target])
    if len(samp) == 0: return np.nan, 0
    ics = []
    for t, g in samp.groupby("open_time"):
        if len(g) < MIN_CYCLE_SYMBOLS: continue
        ic = g[feat].rank().corr(g[target].rank())
        if not pd.isna(ic): ics.append(ic)
    if not ics: return np.nan, 0
    return float(np.mean(ics)), len(ics)


def per_symbol_ic(panel, feat, target="alpha_beta"):
    """Time-series Spearman IC per symbol."""
    samp = panel.dropna(subset=[feat, target])
    ics = {}
    for sym, g in samp.groupby("symbol"):
        if len(g) < 50: continue
        ic = g[feat].rank().corr(g[target].rank())
        if not pd.isna(ic): ics[sym] = ic
    return ics


def block_composite_ic(panel, block_feats, target="alpha_beta", train_t=None):
    """Equal-weight block composite, orient signs by training IC."""
    # Use first half as "training" for sign orientation
    if train_t is None:
        all_t = sorted(panel["open_time"].unique())
        train_t = set(all_t[:len(all_t)//2])
    train = panel[panel["open_time"].isin(train_t)]
    # Train-window IC per feature, pick sign
    signs = {}
    for f in block_feats:
        ic, _ = per_cycle_ic(train, f, target)
        if pd.isna(ic) or abs(ic) < 1e-6:
            signs[f] = 0.0
        else:
            signs[f] = np.sign(ic)
    # Build oriented z-score per feature on FULL panel using train-window mu/sigma
    composite = pd.Series(0.0, index=panel.index)
    valid_mask = pd.Series(False, index=panel.index)
    valid_count = pd.Series(0, index=panel.index)
    for f in block_feats:
        if signs[f] == 0.0: continue
        mu = train[f].mean(); sd = train[f].std()
        if not np.isfinite(sd) or sd == 0: continue
        z = (panel[f] - mu) / sd
        oriented = signs[f] * z
        composite = composite.add(oriented.fillna(0), fill_value=0)
        valid_mask = valid_mask | oriented.notna()
        valid_count = valid_count + oriented.notna().astype(int)
    composite = composite / valid_count.replace(0, np.nan)
    composite[~valid_mask] = np.nan
    panel = panel.copy()
    panel["_composite"] = composite
    ic_test, n = per_cycle_ic(panel[~panel["open_time"].isin(train_t)],
                              "_composite", target)
    return ic_test, signs, n


def pairwise_corr(panel, feats, sample_n=100000):
    """Pairwise pearson on a sample."""
    samp = panel[feats].dropna(how="any").sample(min(sample_n, len(panel.dropna(subset=feats))),
                                                  random_state=42) if len(panel) > 0 else None
    if samp is None or len(samp) == 0:
        # Use all valid rows if can't sample
        samp = panel[feats]
    return samp.corr(method="pearson")


def main():
    print("=== Phase 2: Validate + prune v3 features ===\n", flush=True)
    t0 = time.time()
    panel = pd.read_parquet(PANEL)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    print(f"Loaded: {len(panel):,} rows × {panel.shape[1]} cols", flush=True)

    # Subsample to entry cadence
    times = sorted(panel["open_time"].unique())
    keep_t = set(times[::ENTRY_STRIDE])
    samp = panel[panel["open_time"].isin(keep_t)].copy()
    print(f"Entry-cadence sample: {len(samp):,} rows, {len(keep_t)} cycles\n", flush=True)

    # Step 1: per-feature cross-sectional IC
    print("Step 1: cross-sectional IC per feature\n", flush=True)
    ic_rows = []
    for feat in ALL_FEATS:
        if feat not in samp.columns:
            print(f"  WARN: {feat} missing", flush=True)
            continue
        ic, n_cyc = per_cycle_ic(samp, feat)
        # NaN rate
        nan_rate = samp[feat].isna().mean()
        ic_rows.append({"feature": feat, "block": next(b for b, fs in BLOCKS.items() if feat in fs),
                         "mean_ic": ic, "abs_ic": abs(ic) if not pd.isna(ic) else 0,
                         "n_cycles": n_cyc, "nan_rate": nan_rate})
    df_ic = pd.DataFrame(ic_rows).sort_values("abs_ic", ascending=False)
    df_ic.to_csv(OUT / "feature_ic.csv", index=False)
    print(df_ic[["feature","block","mean_ic","n_cycles","nan_rate"]].to_string(
        index=False, float_format=lambda x: f"{x:+.4f}"), flush=True)
    print(f"\n{time.time()-t0:.0f}s elapsed\n", flush=True)

    # Step 2: per-symbol time-series IC
    print("Step 2: per-symbol time-series IC distribution\n", flush=True)
    psym_rows = []
    for feat in ALL_FEATS:
        if feat not in samp.columns: continue
        psics = per_symbol_ic(samp, feat)
        if not psics: continue
        vals = list(psics.values())
        psym_rows.append({
            "feature": feat,
            "n_syms": len(vals),
            "mean_psym_ic": float(np.mean(vals)),
            "median_psym_ic": float(np.median(vals)),
            "frac_pos": float(np.mean(np.array(vals) > 0)),
            "max_psym_ic": float(np.max(vals)),
            "min_psym_ic": float(np.min(vals)),
        })
    df_psym = pd.DataFrame(psym_rows).sort_values("mean_psym_ic", ascending=False, key=abs)
    df_psym.to_csv(OUT / "per_symbol_feature_ic.csv", index=False)
    print(df_psym.to_string(index=False, float_format=lambda x: f"{x:+.4f}"), flush=True)
    print(f"\n{time.time()-t0:.0f}s elapsed\n", flush=True)

    # Step 3: block composites
    print("Step 3: block composite IC (sign-oriented equal-weight, train/test split)\n",
          flush=True)
    comp_rows = []
    sorted_t = sorted(samp["open_time"].unique())
    train_t = set(sorted_t[:len(sorted_t)//2])
    for blk, feats in BLOCKS.items():
        in_panel = [f for f in feats if f in samp.columns]
        if not in_panel: continue
        ic, signs, n_test = block_composite_ic(samp, in_panel, train_t=train_t)
        comp_rows.append({"block": blk, "n_features": len(in_panel),
                          "composite_ic_test": ic, "n_cycles_test": n_test,
                          "signs": ",".join(f"{f}:{int(signs.get(f,0)):+d}" for f in in_panel)})
    df_comp = pd.DataFrame(comp_rows).sort_values("composite_ic_test", ascending=False, key=abs)
    df_comp.to_csv(OUT / "block_composite_ic.csv", index=False)
    print(df_comp[["block","n_features","composite_ic_test","n_cycles_test"]].to_string(
        index=False, float_format=lambda x: f"{x:+.4f}"), flush=True)
    print(f"\n{time.time()-t0:.0f}s elapsed\n", flush=True)

    # Step 4: pairwise correlation
    print("Step 4: pairwise correlation (sampling 100k rows)\n", flush=True)
    avail = [f for f in ALL_FEATS if f in samp.columns]
    corr = pairwise_corr(samp[["open_time"]+avail], avail, sample_n=100000)
    corr.to_csv(OUT / "correlation_matrix.csv")
    # High pairs
    pairs = []
    for i in range(len(avail)):
        for j in range(i+1, len(avail)):
            c = corr.iloc[i, j]
            if pd.notna(c) and abs(c) > 0.85:
                pairs.append({"f1": avail[i], "f2": avail[j], "corr": c})
    df_pairs = pd.DataFrame(pairs).sort_values("corr", key=abs, ascending=False)
    print(f"High-corr pairs (|corr| > 0.85): {len(df_pairs)}", flush=True)
    if len(df_pairs) > 0:
        print(df_pairs.to_string(index=False, float_format=lambda x: f"{x:+.3f}"), flush=True)
    print(f"\n{time.time()-t0:.0f}s elapsed\n", flush=True)

    # Step 5: apply pruning rules
    print("Step 5: apply pruning rules\n", flush=True)
    rules_log = []
    drop_set = set()

    # Rule 4: drop NaN rate > 30%
    high_nan = df_ic[df_ic["nan_rate"] > 0.30]["feature"].tolist()
    for f in high_nan:
        drop_set.add(f); rules_log.append(f"DROP {f}: NaN > 30%")

    # Rule 5: per-symbol variance — drop features with near-zero variance for >25% of syms
    for f in ALL_FEATS:
        if f not in samp.columns: continue
        var_per_sym = samp.groupby("symbol")[f].std()
        # Use 5th percentile of std as threshold for "near-zero"
        threshold = max(1e-8, var_per_sym.median() * 0.01)
        n_near_zero = (var_per_sym < threshold).sum()
        if n_near_zero > 0.25 * len(var_per_sym):
            drop_set.add(f)
            rules_log.append(f"DROP {f}: {n_near_zero}/{len(var_per_sym)} syms have <0.01×median variance")

    # Rule 3: pairwise — for |corr| > 0.85, drop more-derived
    derive_rank = {
        "log_dollar_volume_7d": 1, "log_dollar_volume_30d": 2,
        "beta_btc_30d": 1, "beta_btc_90d": 2, "beta_btc_180d": 3,
        "beta_btc_instability": 4,  # derived diff
        "corr_btc_30d": 1, "corr_btc_90d": 2, "corr_breakdown": 3,
        "resid_vol_7d": 1, "resid_vol_30d": 2, "resid_vol_90d": 3,
        "dist_from_30d_high": 1, "dist_from_90d_high": 2, "dist_from_365d_high": 3,
        "multi_horizon_trend_score": 4,  # derived avg
        "funding_mean_7d": 1, "funding_mean_30d": 2,
        "funding_z_30d": 3, "funding_abs_30d": 3,
    }
    for _, row in df_pairs.iterrows():
        f1, f2, c = row["f1"], row["f2"], row["corr"]
        if f1 in drop_set or f2 in drop_set: continue
        r1, r2 = derive_rank.get(f1, 5), derive_rank.get(f2, 5)
        # Drop the higher-derive (more derived); tie-break by lower |ic|
        if r1 > r2: drop, keep = f1, f2
        elif r1 < r2: drop, keep = f2, f1
        else:
            ic1 = df_ic[df_ic["feature"]==f1]["abs_ic"].iloc[0]
            ic2 = df_ic[df_ic["feature"]==f2]["abs_ic"].iloc[0]
            drop, keep = (f1, f2) if ic1 < ic2 else (f2, f1)
        drop_set.add(drop)
        rules_log.append(f"DROP {drop}: |corr({f1},{f2})|={abs(c):.3f}, keep {keep}")

    # Rule 1+2: weak standalone IC < 0.005, but protect block members
    # ONLY drop low-IC features whose block composite ALSO fails (block composite |IC| < 0.005)
    composite_ic_by_block = {row["block"]: row["composite_ic_test"]
                              for _, row in df_comp.iterrows()}
    for _, row in df_ic.iterrows():
        f, blk, abs_ic = row["feature"], row["block"], row["abs_ic"]
        if f in drop_set: continue
        if abs_ic >= 0.005: continue
        block_ic = composite_ic_by_block.get(blk, 0)
        if pd.isna(block_ic) or abs(block_ic) < 0.005:
            drop_set.add(f)
            rules_log.append(f"DROP {f}: |IC|={abs_ic:.4f} < 0.005 AND block {blk} composite |IC|={abs(block_ic) if not pd.isna(block_ic) else 0:.4f} < 0.005")

    print("Pruning log:", flush=True)
    for line in rules_log: print(f"  {line}", flush=True)

    # Final survivor list
    survivors = [f for f in ALL_FEATS if f not in drop_set and f in samp.columns]
    print(f"\nSurvivors after pruning: {len(survivors)} (target 18-24)", flush=True)

    # If too many survivors, prune lowest IC until <= 24
    if len(survivors) > 24:
        ic_map = {row["feature"]: row["abs_ic"] for _, row in df_ic.iterrows()}
        survivors_sorted = sorted(survivors, key=lambda f: ic_map.get(f, 0), reverse=True)
        # Keep top 24 by absolute IC
        kept = survivors_sorted[:24]
        for f in survivors_sorted[24:]:
            drop_set.add(f)
            rules_log.append(f"DROP {f}: trim-to-24 (rank-{survivors_sorted.index(f)+1} by |IC|)")
        survivors = kept
        print(f"  Trimmed to top 24 by |IC|: {len(survivors)} features", flush=True)

    # Print final list with block grouping
    print("\nFinal WINNER_BTC_v3 feature list:", flush=True)
    for blk, feats in BLOCKS.items():
        keep = [f for f in feats if f in survivors]
        print(f"  {blk} ({len(keep)}/{len(feats)}): {keep}", flush=True)

    out_path = OUT / "winner_btc_v3_features.json"
    json.dump({
        "n_features": len(survivors),
        "features": survivors,
        "by_block": {blk: [f for f in fs if f in survivors] for blk, fs in BLOCKS.items()},
        "drops_with_reason": rules_log,
        "composite_ic_by_block": composite_ic_by_block,
    }, open(out_path, "w"), indent=2, default=str)
    print(f"\nSaved: {out_path}", flush=True)
    print(f"Total: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
