"""X18 — Augment the existing panel with aggT_4h features for ALL 51 syms.

DISCOVERY: flow_<SYM>.parquet files exist for all 51 syms with 116k rows each
(complete coverage). The panel was just never merged with these for the 26
syms that initially showed 0% aggT coverage.

This script:
1. Loads existing panel_variants_with_funding.parquet
2. For each of the 26 previously-missing syms, loads flow_<SYM>.parquet
3. Computes 4h-aggregated aggT features (signed_volume_4h, tfi_4h,
   aggr_ratio_4h, buy_count_4h, avg_trade_size_4h)
4. Merges into panel where aggT was previously NaN
5. Saves augmented panel as panel_variants_with_funding_v2.parquet

After this, +aggT cells should re-test with full 50-sym aggT coverage.
"""
from __future__ import annotations
import sys, time, warnings, gc, resource
from pathlib import Path
import pandas as pd, numpy as np

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def log_mem(label=""):
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  [MEM {label}] peak_rss={rss_mb:.0f}MB", flush=True)


def aggregate_4h_flow(flow: pd.DataFrame, w: int = 48) -> pd.DataFrame:
    """4h-aggregated trade-flow features from a flow_<SYM>.parquet cache.
    Matches alpha_v8_h48_audit.aggregate_4h_flow exactly.
    """
    sv = flow["signed_volume"].rolling(w, min_periods=max(2, w // 4)).sum()
    tv = (flow["buy_volume"] + flow["sell_volume"]).rolling(w, min_periods=max(2, w // 4)).sum()
    bc = flow["buy_count"].rolling(w, min_periods=max(2, w // 4)).sum()
    sc = flow["sell_count"].rolling(w, min_periods=max(2, w // 4)).sum()
    out = pd.DataFrame(index=flow.index)
    out["signed_volume_4h"] = sv
    out["tfi_4h"] = sv / tv.replace(0, np.nan)
    out["aggr_ratio_4h"] = (bc - sc) / (bc + sc).replace(0, np.nan)
    out["buy_count_4h"] = bc
    out["avg_trade_size_4h"] = tv / (bc + sc).replace(0, np.nan)
    return out


def main():
    t0 = time.time()
    print("=== X18 augment panel with aggT for previously-missing 26 syms ===\n", flush=True)
    log_mem("start")

    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet")
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    print(f"  panel: {len(panel):,} rows × {panel['symbol'].nunique()} syms", flush=True)
    log_mem("after_load_panel")

    # Identify previously-missing syms (those with 0% aggT coverage)
    syms_with_aggT = panel.groupby("symbol")["aggr_ratio_4h"].apply(
        lambda x: x.notna().mean() > 0.5).pipe(lambda s: s[s].index.tolist())
    syms_missing = sorted(set(panel["symbol"].unique()) - set(syms_with_aggT) - {"BTCUSDT"})
    # BTC has aggT — skip
    print(f"  syms with aggT > 50% coverage: {len(syms_with_aggT)}")
    print(f"  syms missing aggT: {len(syms_missing)}: {syms_missing[:5]}...")
    log_mem("after_coverage_check")

    aggT_cols = ["signed_volume_4h", "tfi_4h", "aggr_ratio_4h",
                 "buy_count_4h", "avg_trade_size_4h"]

    # For each missing sym, load flow + compute 4h features + augment panel
    augment_rows = []
    for i, sym in enumerate(syms_missing, 1):
        flow_path = REPO / f"data/ml/cache/flow_{sym}.parquet"
        if not flow_path.exists():
            print(f"  [{i}/{len(syms_missing)}] {sym}: NO flow file, skipping")
            continue
        flow = pd.read_parquet(flow_path)
        if "signed_volume" not in flow.columns:
            print(f"  [{i}/{len(syms_missing)}] {sym}: flow missing signed_volume cols, skipping")
            continue
        # Ensure datetime index UTC
        if not isinstance(flow.index, pd.DatetimeIndex):
            if "open_time" in flow.columns:
                flow = flow.set_index("open_time")
        if flow.index.tz is None:
            flow.index = flow.index.tz_localize("UTC")
        flow = flow.sort_index()

        agg = aggregate_4h_flow(flow)
        agg["symbol"] = sym
        agg = agg.reset_index().rename(columns={"index": "open_time", flow.index.name or "index": "open_time"})
        if "open_time" not in agg.columns:
            agg["open_time"] = agg.iloc[:, 0]  # fallback
        augment_rows.append(agg[["symbol", "open_time"] + aggT_cols])
        if i % 5 == 0: log_mem(f"after {i}/{len(syms_missing)}")

    augment_df = pd.concat(augment_rows, ignore_index=True)
    augment_df["open_time"] = pd.to_datetime(augment_df["open_time"], utc=True)
    print(f"\n  augment_df: {len(augment_df):,} rows × {augment_df['symbol'].nunique()} syms", flush=True)
    log_mem("after_concat_augment")

    # Merge — update panel's aggT cols where NaN with augment values
    panel_aug = panel.set_index(["symbol", "open_time"]).copy()
    augment_idx = augment_df.set_index(["symbol", "open_time"])
    common = panel_aug.index.intersection(augment_idx.index)
    print(f"  common keys (panel × augment): {len(common):,}", flush=True)
    for c in aggT_cols:
        panel_aug.loc[common, c] = augment_idx.loc[common, c].astype(np.float32)
    panel_aug = panel_aug.reset_index()

    # Coverage check
    print(f"\n=== Coverage AFTER augmentation ===")
    for c in aggT_cols:
        nn = panel_aug[c].notna().mean() * 100
        per_sym = panel_aug.groupby("symbol")[c].apply(lambda x: x.notna().mean() > 0.5).sum()
        print(f"  {c}: {nn:.1f}% overall, {per_sym}/{panel_aug['symbol'].nunique()} syms >50%")

    out_path = REPO / "outputs/vBTC_features/panel_variants_with_funding_v2.parquet"
    panel_aug.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path} [{time.time()-t0:.0f}s]")
    log_mem("end")


if __name__ == "__main__":
    main()
