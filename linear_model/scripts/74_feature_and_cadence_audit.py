"""Step 74: feature-level IC and 4h cadence/anchor audit.

Questions answered:
1. Which current V2 features have any cross-sectional predictive power?
2. Does evaluating only one fixed 4h anchor miss signal that appears at other
   5m offsets inside the 4h block?

The audit is feature-first and model-light:
- Rebuild current drop-BIO+VVV HL>=2M panel/features from Step 67 helpers.
- Measure feature IC vs alpha_beta at the current 4h cadence.
- Measure the same IC/spread across all 48 possible 4h anchors.
- Measure the current model score cadence sensitivity using saved per-symbol
  predictions where available, and a freshly built drop-2 current input.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s64 = _imp("s64", "linear_model/scripts/64_meanrev_v2_backtest.py")
s67 = _imp("s67", "linear_model/scripts/67_persymbol_meanrev.py")
s68 = _imp("s68", "linear_model/scripts/68_persymbol_selfstd.py")
s71 = _imp("s71", "linear_model/scripts/71_battery_alleligible.py")

OUT = REPO / "linear_model/results/step74_feature_cadence"
OUT.mkdir(parents=True, exist_ok=True)
OOS = set(s64.OOS)
BLOCK = s64.BLOCK


def _cycle_ic_rows(df: pd.DataFrame, col: str, offset: int, min_n: int = 10) -> pd.DataFrame:
    d = df[df["fold"].isin(OOS)].copy()
    times = sorted(d["open_time"].unique())[offset::BLOCK]
    d = d[d["open_time"].isin(set(times))]
    rows = []
    for t, g in d.dropna(subset=[col, "alpha_beta"]).groupby("open_time", sort=True):
        if len(g) < min_n or g[col].std() <= 1e-12 or g["alpha_beta"].std() <= 1e-12:
            continue
        rows.append(
            {
                "open_time": t,
                "fold": int(g["fold"].iloc[0]),
                "offset": offset,
                "ic": float(g[col].corr(g["alpha_beta"], method="spearman")),
                "pearson": float(g[col].corr(g["alpha_beta"], method="pearson")),
                "n": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def _top_bottom_rows(df: pd.DataFrame, col: str, offset: int, k: int = 3) -> pd.DataFrame:
    d = df[df["fold"].isin(OOS)].copy()
    times = sorted(d["open_time"].unique())[offset::BLOCK]
    d = d[d["open_time"].isin(set(times))]
    rows = []
    for t, g in d.dropna(subset=[col, "alpha_beta"]).groupby("open_time", sort=True):
        if len(g) < 2 * k:
            continue
        gs = g.sort_values(col, ascending=False)
        top = gs.head(k)["alpha_beta"].mean() * 1e4
        bot = gs.tail(k)["alpha_beta"].mean() * 1e4
        rows.append(
            {
                "open_time": t,
                "fold": int(g["fold"].iloc[0]),
                "offset": offset,
                "spread_bps": float(top - bot),
                "half_weight_bps": float(0.5 * (top - bot)),
            }
        )
    return pd.DataFrame(rows)


def summarize_col(df: pd.DataFrame, col: str, offsets: list[int]) -> dict:
    ic_all = []
    tb_all = []
    for off in offsets:
        ic = _cycle_ic_rows(df, col, off)
        tb = _top_bottom_rows(df, col, off)
        if len(ic):
            ic_all.append(ic)
        if len(tb):
            tb_all.append(tb)
    icd = pd.concat(ic_all, ignore_index=True) if ic_all else pd.DataFrame()
    tbd = pd.concat(tb_all, ignore_index=True) if tb_all else pd.DataFrame()
    if icd.empty:
        return {
            "feature": col,
            "n_cycles": 0,
            "ic_mean": np.nan,
            "ic_median": np.nan,
            "ic_t": np.nan,
            "pearson_mean": np.nan,
            "spread_bps": np.nan,
            "spread_pos_pct": np.nan,
        }
    ic_std = icd["ic"].std(ddof=1)
    return {
        "feature": col,
        "n_cycles": int(len(icd)),
        "ic_mean": float(icd["ic"].mean()),
        "ic_median": float(icd["ic"].median()),
        "ic_t": float(icd["ic"].mean() / (ic_std / np.sqrt(len(icd)))) if ic_std > 0 else np.nan,
        "pearson_mean": float(icd["pearson"].mean()),
        "spread_bps": float(tbd["spread_bps"].mean()) if len(tbd) else np.nan,
        "spread_pos_pct": float((tbd["spread_bps"] > 0).mean()) if len(tbd) else np.nan,
    }


def offset_summary(df: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    rows = []
    for off in range(BLOCK):
        r = summarize_col(df, col, [off])
        r["offset"] = off
        r["label"] = label
        rows.append(r)
    out = pd.DataFrame(rows)
    out = out.sort_values("offset")
    out.to_csv(OUT / f"{label}_{col}_offsets.csv", index=False)
    return out


def print_offset_digest(label: str, off: pd.DataFrame):
    cur = off[off["offset"] == 0].iloc[0]
    best_ic = off.iloc[off["ic_mean"].abs().idxmax()]
    best_sp = off.iloc[off["spread_bps"].idxmax()]
    print(
        f"  {label:18s} offset0 IC={cur.ic_mean:+.4f} spread={cur.spread_bps:+.2f} | "
        f"best |IC| off={int(best_ic.offset):02d} IC={best_ic.ic_mean:+.4f} "
        f"spread={best_ic.spread_bps:+.2f} | "
        f"best spread off={int(best_sp.offset):02d} IC={best_sp.ic_mean:+.4f} "
        f"spread={best_sp.spread_bps:+.2f}",
        flush=True,
    )


def main():
    print("=" * 92, flush=True)
    print("  STEP 74: current feature and fixed-4h cadence audit", flush=True)
    print("=" * 92, flush=True)
    t0 = time.time()

    print("\nBuilding current drop-2 feature matrix...", flush=True)
    panel, px, feat_cols, folds = s67.build_panel(["BIOUSDT", "VVVUSDT"])
    px["open_time"] = pd.to_datetime(px["open_time"], utc=True)
    px = px.merge(panel[["symbol", "open_time", "alpha_beta", "fold"]] if "fold" in panel.columns else panel[["symbol", "open_time", "alpha_beta"]],
                  on=["symbol", "open_time", "alpha_beta"], how="left")
    if "fold" not in px.columns or px["fold"].isna().all():
        # Reconstruct fold labels from Step-67 folds.
        px["fold"] = -1
        for fid, spl in enumerate(folds):
            _, _, te = s67._slice(px, spl)
            px.loc[te.index, "fold"] = fid
    print(f"  rows={len(px):,}, symbols={px['symbol'].nunique()}, features={len(feat_cols)}", flush=True)

    # Feature stats and current-anchor IC.
    feat_rows = []
    for f in feat_cols:
        s = px[f]
        feat_rows.append(
            {
                "feature": f,
                "missing_pct": float(s.isna().mean()),
                "zero_pct": float((s.fillna(0) == 0).mean()),
                "std": float(s.std()),
                "p01": float(s.quantile(0.01)),
                "p99": float(s.quantile(0.99)),
                **summarize_col(px, f, [0]),
            }
        )
    feat_df = pd.DataFrame(feat_rows).sort_values("ic_mean", ascending=False)
    feat_df.to_csv(OUT / "feature_ic_offset0.csv", index=False)

    print("\nTop/bottom feature IC at current fixed 4h anchor (offset 0):", flush=True)
    show_cols = ["feature", "ic_mean", "ic_t", "spread_bps", "spread_pos_pct", "std", "zero_pct"]
    print(feat_df[show_cols].head(8).to_string(index=False), flush=True)
    print("\nWorst feature IC at offset 0:", flush=True)
    print(feat_df[show_cols].tail(8).to_string(index=False), flush=True)

    # Anchor sensitivity for model scores.
    print("\nRebuilding current drop2 model score for cadence test...", flush=True)
    drop2 = s71.build(["BIOUSDT", "VVVUSDT"])[0]
    drop2["open_time"] = pd.to_datetime(drop2["open_time"], utc=True)
    model_off = offset_summary(drop2, "pred_z", "drop2_current")
    print("\n4h anchor sensitivity for current model score:", flush=True)
    print_offset_digest("drop2 pred_z", model_off)
    print(f"  offsets positive IC: {(model_off.ic_mean > 0).sum()}/48; "
          f"positive spread: {(model_off.spread_bps > 0).sum()}/48", flush=True)

    # Anchor sensitivity for strongest absolute feature IC at offset0.
    candidates = feat_df.reindex(feat_df["ic_mean"].abs().sort_values(ascending=False).index)["feature"].head(5).tolist()
    off_rows = []
    print("\n4h anchor sensitivity for strongest offset-0 features:", flush=True)
    for f in candidates:
        off = offset_summary(px, f, f"feature_{f}")
        off_rows.append(off)
        print_offset_digest(f, off)
    pd.concat(off_rows, ignore_index=True).to_csv(OUT / "top_feature_offsets.csv", index=False)

    # Optional exhaustive all-feature/all-anchor pass. It is slow because it
    # recomputes 22 features x 48 anchors over the full panel.
    if os.environ.get("STEP74_ALL_FEATURE_OFFSETS") == "1":
        alloff_rows = [summarize_col(px, f, list(range(BLOCK))) for f in feat_cols]
        alloff = pd.DataFrame(alloff_rows).sort_values("ic_mean", ascending=False)
        alloff.to_csv(OUT / "feature_ic_all_offsets.csv", index=False)
        print("\nBest features averaged over all 48 anchors:", flush=True)
        print(alloff[["feature", "ic_mean", "ic_t", "spread_bps", "spread_pos_pct"]].head(10).to_string(index=False), flush=True)

    print(f"\nSaved outputs under {OUT}", flush=True)
    print(f"Total: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
