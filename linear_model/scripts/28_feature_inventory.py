"""Step 28: Comprehensive feature inventory + redundancy audit.

Catalog ALL features available in:
  - base panel (panel_variants_with_funding.parquet)
  - btc_only panel
  - v3_5m panel
  - vol_surge_features.parquet

For each feature NOT in current R3, audit:
  1. Per-cycle IC vs alpha_β
  2. Spearman rank correlation with each R3 feature (redundancy)
  3. Shape (monotonic/U/inverted-U/noisy)
  4. NaN coverage

Categorize by concept:
  - momentum (short/medium/long)
  - volatility (multiple windows)
  - volume / liquidity
  - funding
  - cross-sectional context
  - BTC relationship
  - process fingerprint
  - regime (market state)

Then identify GAPS: what concepts are missing entirely from R3?
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice

PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
BTC_PANEL = REPO / "outputs/vBTC_features_btc_only/panel_btc_only_clean.parquet"
V3_5M = REPO / "outputs/vBTC_features_btc_v3/panel_v3_5m.parquet"
VOL_SURGE = REPO / "linear_model/results/vol_surge_features.parquet"
TARGETS = REPO / "linear_model/data/targets.parquet"
OUT = REPO / "linear_model/results"

R3_FEATS = ["return_1d","atr_pct","dom_level_vs_bk","dom_change_288b_vs_bk",
            "bk_ema_slope_4h","corr_change_3d_vs_bk","obv_z_1d","vwap_slope_96",
            "bars_since_high_xs_rank","idio_vol_1d_vs_bk_xs_rank",
            "funding_rate","funding_rate_z_7d","corr_to_btc_1d",
            "idio_vol_to_btc_1h","beta_to_btc_change_5d","funding_rate_1d_change"]


def shape_diag(decile_targets):
    d = decile_targets.values
    if len(d) < 10: return "insufficient"
    rho = stats.spearmanr(range(10), d).statistic
    mid = np.mean(d[3:7])
    tails = np.mean(d[[0,1,2,7,8,9]])
    if rho > 0.7:   return "monotonic_up"
    elif rho < -0.7: return "monotonic_down"
    elif mid > tails + abs(np.std(d)) * 0.5: return "inverted_u"
    elif mid < tails - abs(np.std(d)) * 0.5: return "u_shape"
    else: return "noisy"


def categorize(name):
    n = name.lower()
    if any(k in n for k in ["return_1d", "return_3d", "return_7d", "ema_slope", "vwap_slope"]):
        return "momentum"
    if any(k in n for k in ["atr", "vol_", "_vol", "kurt", "skew", "jump", "max_abs"]):
        return "volatility/distrib"
    if any(k in n for k in ["volume", "quote_vol", "amihud", "surge", "obv", "buy_count", "trade_size"]):
        return "volume/liquidity"
    if "funding" in n:
        return "funding"
    if any(k in n for k in ["xs_alpha", "name_factor", "name_idio", "xs_rank"]):
        return "cross-sectional"
    if any(k in n for k in ["btc", "beta_to", "corr_to_btc", "dom_btc"]):
        return "btc-relationship"
    if any(k in n for k in ["bars_since", "dist_from"]):
        return "anchoring"
    if "vs_bk" in n or n.startswith("bk_"):
        return "basket-relative"
    if any(k in n for k in ["hour_", "cluster", "sym_id"]):
        return "categorical/calendar"
    if any(k in n for k in ["tfi", "aggr", "signed_volume", "price_volume_corr"]):
        return "microstructure"
    return "other"


def main():
    print("=== Step 28: Feature inventory + redundancy audit ===\n", flush=True)
    t0 = time.time()

    # Load all panels + collect feature names
    import pyarrow.parquet as pq
    cols_base = pq.read_schema(PANEL).names
    cols_btc = pq.read_schema(BTC_PANEL).names
    cols_v3 = pq.read_schema(V3_5M).names
    cols_vs = pq.read_schema(VOL_SURGE).names

    all_features = {}  # name -> source
    SKIP = {"open_time","symbol","exit_time","return_pct","__fragment_index",
            "__batch_index","__last_in_fragment","__filename","cluster",
            "basket_fwd","basket_A_fwd","basket_B_fwd","basket_C_fwd","basket_D_fwd",
            "alpha_realized","alpha_A","alpha_B","alpha_C","alpha_D",
            "alpha_vs_btc_realized","demeaned_target","beta_short_vs_bk",
            "alpha_beta","target_A","target_B","target_C","target_D",
            "target_beta_btc","sigma_idio_btc","btc_fwd","btc_target",
            "alpha_beta","target_z","target_bps_raw","sigma_idio_ref","beta_pit",
            "autocorr_pctile_7d","date"}
    for c in cols_base:
        if c in SKIP: continue
        all_features[c] = "base"
    for c in cols_btc:
        if c in SKIP: continue
        if c not in all_features:
            all_features[c] = "btc_only"
    for c in cols_v3:
        if c in SKIP: continue
        if c not in all_features:
            all_features[c] = "v3_5m"
    for c in cols_vs:
        if c in SKIP: continue
        if c not in all_features:
            all_features[c] = "vol_surge"

    # Categorize
    feature_meta = []
    for f, src in all_features.items():
        in_r3 = f in R3_FEATS
        cat = categorize(f)
        feature_meta.append({"feature":f, "source":src, "category":cat,
                              "in_R3": in_r3})
    df_meta = pd.DataFrame(feature_meta)
    print(f"Total features cataloged: {len(df_meta)}", flush=True)
    print(f"R3 features: {df_meta['in_R3'].sum()}", flush=True)

    # Coverage by category
    print(f"\n--- COVERAGE BY CATEGORY ---", flush=True)
    print(f"  {'category':<25} {'total':>6} {'in_R3':>6} {'available':>10}",
          flush=True)
    for cat, g in df_meta.groupby("category"):
        in_r3 = g["in_R3"].sum()
        total = len(g)
        avail = total - in_r3
        print(f"  {cat:<25} {total:>6} {in_r3:>6} {avail:>10}", flush=True)

    # List by category
    print(f"\n--- ALL FEATURES BY CATEGORY ---", flush=True)
    for cat, g in df_meta.groupby("category"):
        print(f"\n  [{cat}]", flush=True)
        for _, r in g.iterrows():
            marker = " ← in R3" if r["in_R3"] else ""
            print(f"    [{r['source']:<10}] {r['feature']:<32}{marker}", flush=True)

    # ===== Audit unused features that aren't already-rejected =====
    # Already-rejected from Step 14 audit (inverted-U):
    REJECTED = {"btc_ret_12b","btc_ret_48b","btc_ret_288b","btc_ema_slope_4h",
                "name_factor_loading_1d","name_idio_share_1d"}
    unused = df_meta[~df_meta["in_R3"] & ~df_meta["feature"].isin(REJECTED)]
    print(f"\n--- UNUSED, NOT-YET-REJECTED FEATURES TO AUDIT ({len(unused)}) ---",
          flush=True)

    # Load actual data for audit
    print(f"\nLoading panels...", flush=True)
    tgt = pd.read_parquet(TARGETS, columns=["symbol","open_time","alpha_beta",
                                              "autocorr_pctile_7d"])
    tgt["open_time"] = pd.to_datetime(tgt["open_time"], utc=True)
    folds = _multi_oos_splits(tgt)
    train0_t = _slice(tgt, folds[0])[0]["open_time"]
    train_lo, train_hi = train0_t.min(), train0_t.max()
    train0 = tgt[(tgt["open_time"] >= train_lo) & (tgt["open_time"] <= train_hi)
                 & (tgt["autocorr_pctile_7d"] >= 0.5)
                 & tgt["alpha_beta"].notna()]
    print(f"  Fold-0 train: {len(train0):,} rows", flush=True)

    # Audit each feature
    print(f"\n--- AUDIT RESULTS ---", flush=True)
    print(f"  {'feature':<32} {'category':<18} {'src':<10} {'nan%':>6} "
          f"{'pearson':>9} {'spearman':>10} {'shape':<14}", flush=True)
    audit_results = []
    for _, r in unused.iterrows():
        f = r["feature"]
        src = r["source"]
        try:
            if src == "base":
                col_data = pd.read_parquet(PANEL, columns=["symbol","open_time",f])
            elif src == "btc_only":
                col_data = pd.read_parquet(BTC_PANEL, columns=["symbol","open_time",f])
            elif src == "v3_5m":
                col_data = pd.read_parquet(V3_5M, columns=["symbol","open_time",f])
            elif src == "vol_surge":
                col_data = pd.read_parquet(VOL_SURGE, columns=["symbol","open_time",f])
            else:
                continue
            col_data["open_time"] = pd.to_datetime(col_data["open_time"], utc=True)
        except Exception:
            continue
        merged = train0.merge(col_data, on=["symbol","open_time"], how="left")
        s = merged[f]
        valid = s.notna() & merged["alpha_beta"].notna()
        if valid.sum() < 1000: continue
        ss = s[valid]
        # winsorize at 0.5/99.5 for cleaner correlations
        ss = ss.clip(ss.quantile(0.005), ss.quantile(0.995))
        yy = merged.loc[valid, "alpha_beta"] * 1e4
        yy = yy.clip(-1000, 1000)
        nan_pct = (1 - valid.mean()) * 100
        try:
            pr = stats.pearsonr(ss, yy).statistic
            sr = stats.spearmanr(ss, yy).statistic
            q = pd.qcut(ss, 10, labels=False, duplicates="drop")
            dec = yy.groupby(q).mean()
            shape = shape_diag(dec)
        except Exception:
            pr = sr = np.nan; shape = "fail"
        audit_results.append({"feature":f, "source":src,
                              "category":r["category"], "nan_pct":nan_pct,
                              "pearson":pr, "spearman":sr,
                              "abs_spearman":abs(sr) if not pd.isna(sr) else 0,
                              "shape":shape})

    df_audit = pd.DataFrame(audit_results).sort_values("abs_spearman", ascending=False)
    df_audit.to_csv(OUT / "feature_inventory_audit.csv", index=False)

    for _, r in df_audit.iterrows():
        print(f"  {r['feature']:<32} {r['category']:<18} {r['source']:<10} "
              f"{r['nan_pct']:>5.1f}% {r['pearson']:>+9.4f} {r['spearman']:>+10.4f} "
              f"{r['shape']:<14}", flush=True)

    # Recommendation logic
    print(f"\n--- RECOMMENDATIONS ---", flush=True)
    rec_keep = df_audit[(df_audit["abs_spearman"] >= 0.01)
                         & (df_audit["nan_pct"] < 30)
                         & ((df_audit["shape"] == "monotonic_up")
                            | (df_audit["shape"] == "monotonic_down")
                            | (df_audit["shape"] == "u_shape"))]
    rec_reject = df_audit[df_audit["shape"] == "inverted_u"]
    print(f"\n  ADD candidates ({len(rec_keep)} — monotonic/u-shape, |sp|≥0.01):", flush=True)
    for _, r in rec_keep.iterrows():
        transform = " + squared" if "u_shape" in r["shape"] else ""
        print(f"    {r['feature']:<32} ({r['category']:<18}) "
              f"|sp|={r['abs_spearman']:.4f}{transform}", flush=True)
    print(f"\n  AVOID ({len(rec_reject)} inverted-U):", flush=True)
    for _, r in rec_reject.iterrows():
        print(f"    {r['feature']:<32} ({r['category']:<18})", flush=True)

    # Categories with NO R3 representation
    print(f"\n--- MISSING CONCEPT COVERAGE ---", flush=True)
    r3_cats = set(df_meta[df_meta["in_R3"]]["category"].unique())
    all_cats = set(df_meta["category"].unique())
    missing_cats = all_cats - r3_cats
    print(f"  Categories with no R3 representation: {missing_cats}", flush=True)
    underused_cats = []
    for cat in sorted(r3_cats):
        cat_total = len(df_meta[df_meta["category"]==cat])
        cat_in_r3 = df_meta[df_meta["category"]==cat]["in_R3"].sum()
        if cat_in_r3 / cat_total < 0.3:
            underused_cats.append((cat, cat_in_r3, cat_total))
    print(f"  Under-used categories (R3 uses <30%):", flush=True)
    for cat, used, total in underused_cats:
        print(f"    {cat:<25} R3 uses {used}/{total}", flush=True)

    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
