"""X15 — Diagnose why crossX works in Per-sym but not Pool+symid.

Hypotheses to test:
  H1: Per-symbol crossX IC heterogeneity — some syms have strong signal, others noise.
      In Pool+symid one shared coef averages them. In Per-sym each sym gets its own.
  H2: crossX features collinear with sym_id one-hot — Ridge coefficient gets split,
      both fit noisily.
  H3: crossX features partially correlated cross-symbol at same time (broadcast-like)
      → reduces per-cycle prediction spread → bad K=3 selection.
  H4: Coverage asymmetry — symbols WITH crossX coverage carry the signal; syms WITHOUT
      drag down the shared coefficient in Pool+symid.

Outputs:
  - Per-symbol IC of each crossX feature
  - Cross-correlation of crossX features with sym_id and with each other
  - Per-cycle (time) prediction spread comparison: BASE vs BASE+crossX (Pool+symid)
"""
from __future__ import annotations
import sys, time, warnings, importlib.util
from pathlib import Path
import pandas as pd, numpy as np

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
OUT = REPO / "research/convexity_portable_2026-05-20/results"
CACHE = OUT / "_cache"

spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)

HL_MAP = pd.read_csv(REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv")
HL_SYMS = set(HL_MAP[HL_MAP["on_hl"] == True]["symbol"].tolist())


def main():
    t0 = time.time()
    print("=== X15 crossX diagnostic ===\n", flush=True)

    # Load panel + crossX features
    needed = ["symbol", "open_time", "alpha_vs_btc_realized"]
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                            columns=needed)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel = panel[panel["symbol"].isin(HL_SYMS) & (panel["symbol"] != "BTCUSDT")].copy()

    cross_df = pd.read_parquet(REPO / "data/ml/cache/cross_exchange_features.parquet")
    cross_df["open_time"] = pd.to_datetime(cross_df["open_time"], utc=True)
    z_cols = [c for c in cross_df.columns if c.endswith("_basis_z")]
    panel = panel.merge(cross_df[["symbol", "open_time"] + z_cols],
                        on=["symbol", "open_time"], how="left")
    print(f"  panel: {len(panel):,} rows × {panel['symbol'].nunique()} syms")
    print(f"  crossX features: {z_cols}\n")

    # === H1: Per-symbol IC heterogeneity ===
    print("=== H1: Per-symbol IC of each crossX feature ===")
    target = "alpha_vs_btc_realized"
    per_sym_results = []
    for sym, g in panel.groupby("symbol"):
        row = {"symbol": sym, "n_rows": len(g),
               "n_non_null_crossX": int(g[z_cols[0]].notna().sum())}
        for c in z_cols:
            valid = g[c].notna() & g[target].notna()
            if valid.sum() < 200:
                row[f"{c.replace('_basis_z','')}_ic"] = None
            else:
                row[f"{c.replace('_basis_z','')}_ic"] = float(g.loc[valid, c].corr(g.loc[valid, target]))
        per_sym_results.append(row)
    per_sym_df = pd.DataFrame(per_sym_results)
    # Print top/bottom 5 per feature
    for c in z_cols:
        c_short = c.replace("_basis_z", "")
        col = f"{c_short}_ic"
        if col not in per_sym_df.columns: continue
        valid = per_sym_df[col].notna()
        ic_vals = per_sym_df.loc[valid, col]
        print(f"\n  Feature: {c}")
        print(f"    overall: n_syms_evaluable={valid.sum()}, mean IC={ic_vals.mean():+.4f}, "
              f"median={ic_vals.median():+.4f}, std={ic_vals.std():.4f}")
        print(f"    range: min={ic_vals.min():+.4f}, max={ic_vals.max():+.4f}")
        # Sign distribution
        pos = (ic_vals > 0.005).sum()
        neg = (ic_vals < -0.005).sum()
        flat = ((ic_vals.abs() <= 0.005)).sum()
        print(f"    sign distribution: positive(>0.005)={pos}, negative(<-0.005)={neg}, flat={flat}")
        # Top 5 most negative ICs
        top_neg = per_sym_df.loc[valid].nsmallest(5, col)
        print(f"    top 5 most negative IC symbols:")
        for _, r in top_neg.iterrows():
            print(f"      {r['symbol']:<14} IC={r[col]:+.4f} n_non_null={r['n_non_null_crossX']:,}")

    # === H2: cross-feature correlation ===
    print(f"\n=== H2: Pairwise correlation among crossX features ===")
    z_valid = panel[z_cols].dropna()
    corr = z_valid.corr()
    print(f"  ({len(z_valid):,} rows used)")
    print(corr.round(3).to_string())

    # === H3: Per-cycle cross-symbol homogeneity ===
    print(f"\n=== H3: Per-cycle cross-symbol homogeneity (avg pairwise corr per time) ===")
    # For each time t, compute std of crossX features across symbols (within-cycle)
    print(f"  (lower std = more 'broadcast-like' across symbols)")
    for c in z_cols:
        valid = panel[c].notna()
        within_time_std = panel[valid].groupby("open_time")[c].std()
        print(f"  {c}: median within-time std = {within_time_std.median():.4f}, "
              f"mean = {within_time_std.mean():.4f}")

    # === H4: per-cycle prediction spread comparison ===
    # Compare prediction spread when crossX is included vs excluded
    # Use existing X6 prediction files (Ridge Pool+symid BASE vs +crossX)
    print(f"\n=== H4: Per-cycle prediction spread (Pool+symid) ===")
    base_path = CACHE / "x6_Ridge_pool+symid_BASE_preds.parquet"
    crossx_path = CACHE / "x6_Ridge_pool+symid_pcrossX_preds.parquet"
    if base_path.exists() and crossx_path.exists():
        p_base = pd.read_parquet(base_path)[["symbol", "open_time", "pred"]]
        p_cx = pd.read_parquet(crossx_path)[["symbol", "open_time", "pred"]]
        # Within-cycle std of predictions
        base_within_std = p_base.groupby("open_time")["pred"].std()
        cx_within_std = p_cx.groupby("open_time")["pred"].std()
        print(f"  BASE: median within-time pred std = {base_within_std.median():.4f}")
        print(f"  +crossX: median within-time pred std = {cx_within_std.median():.4f}")
        print(f"  Ratio (crossX/base): {cx_within_std.median()/base_within_std.median():.4f}")
        if cx_within_std.median() < base_within_std.median():
            print(f"  → +crossX REDUCES prediction spread = harder to differentiate top-3 picks")

    # Save summary
    per_sym_df.to_csv(OUT / "X15_per_sym_crossX_IC.csv", index=False)
    print(f"\nSaved per-sym IC table → {OUT / 'X15_per_sym_crossX_IC.csv'}")
    print(f"[total {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
