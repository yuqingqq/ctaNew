"""Step 9: Diagnose WHY Ridge underperforms.

Hypotheses to test:
  H1. Tail inversion is Ridge-specific (LGBM doesn't have it) — model class issue
  H2. Tail inversion is universal — feature/signal issue (LGBM also has it but
      gets saved by tree clipping that produces smaller-magnitude tail preds)
  H3. Ridge predictions are dominated by sym_id dummies (per-symbol means) and
      its 16 numeric features have low signal
  H4. Specific symbols (high-σ memes) dominate Ridge's tails — universe issue
  H5. Ridge predictions are essentially uncorrelated with LGBM at tails — they're
      seeing different signals

Diagnostics:
  1. Decile analysis side-by-side (Ridge vs LGBM)
  2. Rank correlation Ridge vs LGBM (overall + at top/bottom)
  3. Per-symbol IC for both
  4. Ridge coefficient inspection: what does it lean on?
  5. Pred distribution stats per symbol (which syms dominate tails)
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

RIDGE = REPO / "linear_model/results/predictions.parquet"
LGBM = REPO / "linear_model/results/lgbm_shift49_predictions.parquet"
COEFS = REPO / "linear_model/results/coefficients.csv"
OUT = REPO / "linear_model/results"


def main():
    print("=== Step 9: Ridge failure diagnostic ===\n", flush=True)

    # Load both prediction sets
    r = pd.read_parquet(RIDGE)
    r["open_time"] = pd.to_datetime(r["open_time"], utc=True)
    r = r.rename(columns={"pred_z": "pred_ridge"})
    print(f"Ridge: {len(r):,} rows", flush=True)

    l = pd.read_parquet(LGBM)
    l["open_time"] = pd.to_datetime(l["open_time"], utc=True)
    l = l.rename(columns={"pred": "pred_lgbm"})[
        ["symbol","open_time","pred_lgbm","alpha_A"]]
    print(f"LGBM:  {len(l):,} rows", flush=True)

    # Merge on (symbol, open_time)
    df = r.merge(l, on=["symbol","open_time"], how="inner")
    df["alpha_realized"] = df["alpha_beta"]
    df["alpha_bps"] = df["alpha_realized"] * 1e4
    print(f"Merged: {len(df):,} rows\n", flush=True)

    # ===== H1/H2: Decile analysis side-by-side =====
    print("=" * 90, flush=True)
    print("DECILE ANALYSIS — does LGBM ALSO have inverted tails?", flush=True)
    print("=" * 90, flush=True)

    # 4h entry-cadence sample
    times = sorted(df.open_time.unique())
    keep = set(times[::48])
    samp = df[df.open_time.isin(keep)].copy()

    for model_name in ("ridge", "lgbm"):
        col = f"pred_{model_name}"
        samp[f"dec_{model_name}"] = samp.groupby("open_time")[col].transform(
            lambda x: pd.qcut(x, 10, labels=False, duplicates="drop"))
        print(f"\n  {model_name.upper()} pred deciles vs realized α_β:", flush=True)
        dec = samp.groupby(f"dec_{model_name}")["alpha_bps"].agg(
            ["mean","std","count"]).round(2)
        dec.columns = ["mean_bps","std","n"]
        for i, row in dec.iterrows():
            print(f"    decile {int(i)}: {row['mean_bps']:>+7.2f} bps  "
                  f"(std {row['std']:.4f}, n={int(row['n']):,})", flush=True)
        # Tail spread
        d0 = dec.loc[0,"mean_bps"]; d9 = dec.loc[9,"mean_bps"]
        print(f"    decile 9 - decile 0 = {d9-d0:+.2f} bps "
              f"({'POSITIVE = good' if d9>d0 else 'NEGATIVE = INVERTED'})",
              flush=True)

    # ===== H5: rank correlation Ridge vs LGBM =====
    print("\n" + "=" * 90, flush=True)
    print("RIDGE vs LGBM RANK CORRELATION — same signal?", flush=True)
    print("=" * 90, flush=True)
    # Per-cycle Spearman
    cyc_corr = samp.groupby("open_time").apply(
        lambda g: g["pred_ridge"].rank().corr(g["pred_lgbm"].rank())
        if len(g) >= 5 else np.nan).dropna()
    print(f"\n  Cross-sectional rank correlation Ridge vs LGBM:", flush=True)
    print(f"    Mean: {cyc_corr.mean():+.4f}  Median: {cyc_corr.median():+.4f}  "
          f"Std: {cyc_corr.std():.4f}", flush=True)
    print(f"    p10/p90: [{cyc_corr.quantile(0.1):+.3f}, "
          f"{cyc_corr.quantile(0.9):+.3f}]", flush=True)

    # Agreement at tails: do they pick the same top-3 / bot-3?
    agree_top = 0; agree_bot = 0; n_cycles = 0
    for t, g in samp.groupby("open_time"):
        if len(g) < 7: continue
        r_top3 = set(g.nlargest(3, "pred_ridge")["symbol"])
        l_top3 = set(g.nlargest(3, "pred_lgbm")["symbol"])
        r_bot3 = set(g.nsmallest(3, "pred_ridge")["symbol"])
        l_bot3 = set(g.nsmallest(3, "pred_lgbm")["symbol"])
        agree_top += len(r_top3 & l_top3) / 3.0
        agree_bot += len(r_bot3 & l_bot3) / 3.0
        n_cycles += 1
    if n_cycles > 0:
        print(f"\n  Top-3 pick overlap: {agree_top/n_cycles*100:.1f}% "
              f"(0% = no agreement, 100% = identical)", flush=True)
        print(f"  Bot-3 pick overlap: {agree_bot/n_cycles*100:.1f}%", flush=True)

    # ===== H4: per-symbol pred + IC =====
    print("\n" + "=" * 90, flush=True)
    print("PER-SYMBOL PREDICTION STATS — which syms dominate tails?", flush=True)
    print("=" * 90, flush=True)
    # Frequency each symbol is in Ridge top-3 / bot-3
    r_top_count = {}
    r_bot_count = {}
    l_top_count = {}
    l_bot_count = {}
    for t, g in samp.groupby("open_time"):
        if len(g) < 7: continue
        for s in g.nlargest(3, "pred_ridge")["symbol"]:
            r_top_count[s] = r_top_count.get(s, 0) + 1
        for s in g.nsmallest(3, "pred_ridge")["symbol"]:
            r_bot_count[s] = r_bot_count.get(s, 0) + 1
        for s in g.nlargest(3, "pred_lgbm")["symbol"]:
            l_top_count[s] = l_top_count.get(s, 0) + 1
        for s in g.nsmallest(3, "pred_lgbm")["symbol"]:
            l_bot_count[s] = l_bot_count.get(s, 0) + 1

    print(f"\n  Top-3 PICK FREQUENCY (out of {n_cycles} cycles):", flush=True)
    print(f"    {'symbol':<14} {'Ridge top':>10} {'LGBM top':>10} {'σ_idio bps':>12}",
          flush=True)
    sigma_map = samp.groupby("symbol")["sigma_idio_ref"].first()
    syms_sorted = sorted(samp.symbol.unique(),
                         key=lambda s: -(r_top_count.get(s,0)+r_bot_count.get(s,0)))
    for s in syms_sorted[:15]:
        print(f"    {s:<14} {r_top_count.get(s,0):>10} {l_top_count.get(s,0):>10} "
              f"{sigma_map.loc[s]*1e4:>12.0f}", flush=True)
    print(f"    ... (showing top 15 by Ridge tail-frequency)", flush=True)

    # Per-symbol IC
    print(f"\n  PER-SYMBOL TIME-SERIES IC (Spearman):", flush=True)
    print(f"    {'symbol':<14} {'Ridge IC':>10} {'LGBM IC':>10} "
          f"{'σ_idio bps':>12} {'Ridge tail freq':>17}", flush=True)
    psym_ic_r = samp.groupby("symbol").apply(
        lambda g: g["pred_ridge"].rank().corr(g["alpha_bps"].rank())
        if len(g)>=20 else np.nan)
    psym_ic_l = samp.groupby("symbol").apply(
        lambda g: g["pred_lgbm"].rank().corr(g["alpha_bps"].rank())
        if len(g)>=20 else np.nan)
    psym_df = pd.DataFrame({
        "Ridge_IC": psym_ic_r,
        "LGBM_IC": psym_ic_l,
        "sigma": sigma_map,
        "ridge_tail_freq": pd.Series({s: r_top_count.get(s,0)+r_bot_count.get(s,0)
                                       for s in samp.symbol.unique()}),
    }).sort_values("LGBM_IC", ascending=False)
    psym_df.to_csv(OUT / "per_symbol_ic_ridge_vs_lgbm.csv")
    for s, row in psym_df.iterrows():
        print(f"    {s:<14} {row['Ridge_IC']:+10.4f} {row['LGBM_IC']:+10.4f} "
              f"{row['sigma']*1e4:>12.0f} {int(row['ridge_tail_freq']):>17}",
              flush=True)
    print(f"\n  Mean Ridge IC: {psym_ic_r.mean():+.4f}  "
          f"Mean LGBM IC: {psym_ic_l.mean():+.4f}", flush=True)
    print(f"  Syms where Ridge IC > LGBM IC: "
          f"{(psym_ic_r > psym_ic_l).sum()}/{len(psym_ic_r)}", flush=True)

    # ===== H3: Ridge coefficient inspection =====
    print("\n" + "=" * 90, flush=True)
    print("RIDGE COEFFICIENT INSPECTION — what does Ridge lean on?", flush=True)
    print("=" * 90, flush=True)
    coefs = pd.read_csv(COEFS)
    # Avg coefficient across folds × seeds
    coef_summary = coefs.groupby("feature")["coef"].agg(["mean","std","min","max"])
    coef_summary["abs_mean"] = coef_summary["mean"].abs()
    coef_summary = coef_summary.sort_values("abs_mean", ascending=False)
    print(f"\n  Top 20 features by |mean coefficient| (z-scored features):",
          flush=True)
    print(f"    {'feature':<32} {'mean coef':>10} {'std':>10} "
          f"{'min':>10} {'max':>10}", flush=True)
    for f, row in coef_summary.head(20).iterrows():
        sign = "+" if row['mean'] >= 0 else ""
        print(f"    {f:<32} {sign}{row['mean']:>+9.4f} {row['std']:>10.4f} "
              f"{row['min']:>+10.4f} {row['max']:>+10.4f}", flush=True)

    # Sum of |coef| for sym dummies vs numeric features
    sym_dummies = [f for f in coef_summary.index if f.startswith("sym_")]
    numeric = [f for f in coef_summary.index if not f.startswith("sym_")]
    sum_abs_sym = coef_summary.loc[sym_dummies, "abs_mean"].sum()
    sum_abs_num = coef_summary.loc[numeric, "abs_mean"].sum()
    print(f"\n  Coefficient mass:", flush=True)
    print(f"    sym_id dummies ({len(sym_dummies)}): "
          f"Σ|coef| = {sum_abs_sym:.3f}", flush=True)
    print(f"    numeric features ({len(numeric)}): "
          f"Σ|coef| = {sum_abs_num:.3f}", flush=True)
    print(f"    Ratio sym/numeric = {sum_abs_sym/sum_abs_num:.2f}× "
          f"({'sym dominates' if sum_abs_sym>sum_abs_num else 'numeric dominates'})",
          flush=True)

    # Per-feature OOS IC: what does each Ridge feature individually correlate with?
    # We don't have raw features here, but we can show what coefficients say

    print(f"\n" + "=" * 90, flush=True)
    print("INTERPRETATION SUMMARY", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
