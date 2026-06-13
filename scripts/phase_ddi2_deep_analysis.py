"""Phase DDI-2: deep analysis combining backtest, features, and model.

Goal: find optimization opportunities by combining all three data sources.

  1. Long vs Short attribution: which side drives PnL? Which has better IC?
  2. Best month vs worst month feature comparison (Feb 2026 vs Sep 2025 + Apr 2026).
  3. LGBM feature importance per fold — is feature value stable, or regime-dependent?
  4. Pred distribution moments × IC quality: does pred kurtosis / skew predict IC?
  5. Wrong-pick feature analysis: when a basket member moved AGAINST our bet,
     what features did it have? Is there a flag we missed?
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

OUT = REPO / "outputs/vBTC_ddi2"
OUT.mkdir(parents=True, exist_ok=True)


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt((288 * 365) / 48))


def main():
    print("=== Phase DDI-2: deep analysis ===\n", flush=True)

    # Load all data sources
    apd = pd.read_parquet(REPO / "outputs/vBTC_audit_panel/all_predictions.parquet")
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True).astype("datetime64[ns, UTC]")
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True).astype("datetime64[ns, UTC]")
    print(f"  loaded {len(apd):,} predictions", flush=True)

    sleeves = pd.read_parquet(REPO / "outputs/vBTC_sleeve_horizon/production_sleeves.parquet")
    sleeves["time"] = pd.to_datetime(sleeves["time"], utc=True).astype("datetime64[ns, UTC]")
    print(f"  loaded {len(sleeves):,} sleeves ({sleeves['traded'].sum()} traded)", flush=True)

    # Load full feature panel (only needed cols) to look at features at picked symbols
    print(f"  loading panel features at picked symbols...", flush=True)
    panel = pd.read_parquet(
        REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
        columns=["open_time", "symbol", "target_A", "return_pct",
                  "funding_rate", "atr_pct", "corr_to_btc_1d", "return_1d",
                  "btc_realized_vol_1d", "idio_vol_1d_vs_bk",
                  "funding_rate_z_7d", "beta_to_btc_change_5d"])
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True).astype("datetime64[ns, UTC]")
    print(f"  panel {len(panel):,} rows", flush=True)

    # ---------- 1. Long vs Short attribution per cycle ----------
    print(f"\n=== 1. Long vs Short attribution ===", flush=True)
    rows = []
    for _, rec in sleeves.iterrows():
        if not rec["traded"]: continue
        t = rec["time"]
        # Get realized return_pct for each basket member from panel
        long_syms = list(rec["long_basket"])
        short_syms = list(rec["short_basket"])
        g = panel[panel["open_time"] == t]
        if len(g) == 0: continue
        # Mean return per side (long_ret - short_ret = basket PnL component)
        l_rets = g[g["symbol"].isin(long_syms)]["return_pct"].dropna()
        s_rets = g[g["symbol"].isin(short_syms)]["return_pct"].dropna()
        if len(l_rets) == 0 or len(s_rets) == 0: continue
        rows.append({
            "time": t, "fold": rec["fold"],
            "long_mean_ret": l_rets.mean(),
            "short_mean_ret": s_rets.mean(),
            "long_pnl_bps": l_rets.mean() * 1e4,
            "short_pnl_bps": -s_rets.mean() * 1e4,  # short side wins if return is negative
        })
    ls = pd.DataFrame(rows)
    ls["total_bps"] = ls["long_pnl_bps"] + ls["short_pnl_bps"]
    print(f"  Cycles with valid long+short: {len(ls)}", flush=True)
    print(f"  Long side mean PnL:  {ls['long_pnl_bps'].mean():+.1f} bps", flush=True)
    print(f"  Short side mean PnL: {ls['short_pnl_bps'].mean():+.1f} bps", flush=True)
    print(f"  Total mean:          {ls['total_bps'].mean():+.1f} bps", flush=True)
    print(f"  Long Sharpe:  {_sharpe(ls['long_pnl_bps']):+.2f}", flush=True)
    print(f"  Short Sharpe: {_sharpe(ls['short_pnl_bps']):+.2f}", flush=True)
    print(f"  Long-Short correlation: {ls['long_pnl_bps'].corr(ls['short_pnl_bps']):+.3f}",
          flush=True)
    print(f"  Long % positive cycles:  {(ls['long_pnl_bps'] > 0).mean()*100:.1f}%",
          flush=True)
    print(f"  Short % positive cycles: {(ls['short_pnl_bps'] > 0).mean()*100:.1f}%",
          flush=True)
    ls.to_csv(OUT / "long_short_attribution.csv", index=False)

    # ---------- 2. Best month vs worst months — feature comparison ----------
    print(f"\n=== 2. Best vs worst month feature comparison ===", flush=True)
    cycles_v31 = pd.read_csv(REPO / "outputs/vBTC_sleeve_horizon/per_cycle_robust_equal_6.csv")
    cycles_v31["time"] = pd.to_datetime(cycles_v31["time"], utc=True).astype("datetime64[ns, UTC]")
    cycles_v31["month"] = pd.to_datetime(cycles_v31["time"]).dt.to_period("M").astype(str)
    cycles_v31["traded"] = cycles_v31["gross_exposure"] > 0.01

    best_month = "2026-02"  # +7.24 Sharpe
    worst_months = ["2025-09", "2026-04"]  # -3.42, -3.30

    # For each picked symbol in those months, look at features at entry
    best_cycles = cycles_v31[cycles_v31["month"] == best_month]["time"].tolist()
    worst_cycles = cycles_v31[cycles_v31["month"].isin(worst_months)]["time"].tolist()

    best_picks = []
    worst_picks = []
    for sleeve_rec in sleeves.itertuples():
        if not sleeve_rec.traded: continue
        t = sleeve_rec.time
        if t in best_cycles:
            for s in sleeve_rec.long_basket:
                best_picks.append((t, s, "long"))
            for s in sleeve_rec.short_basket:
                best_picks.append((t, s, "short"))
        elif t in worst_cycles:
            for s in sleeve_rec.long_basket:
                worst_picks.append((t, s, "long"))
            for s in sleeve_rec.short_basket:
                worst_picks.append((t, s, "short"))
    best_picks = pd.DataFrame(best_picks, columns=["time", "symbol", "side"])
    worst_picks = pd.DataFrame(worst_picks, columns=["time", "symbol", "side"])
    print(f"  Feb 2026 picks (best month): {len(best_picks)}", flush=True)
    print(f"  Sep 2025 + Apr 2026 picks (worst months): {len(worst_picks)}", flush=True)

    # Join features
    best_with_feat = best_picks.merge(panel.rename(columns={"open_time": "time"}),
                                            on=["time", "symbol"], how="left")
    worst_with_feat = worst_picks.merge(panel.rename(columns={"open_time": "time"}),
                                              on=["time", "symbol"], how="left")

    features_to_compare = ["target_A", "funding_rate", "atr_pct", "corr_to_btc_1d",
                              "return_1d", "btc_realized_vol_1d", "idio_vol_1d_vs_bk",
                              "funding_rate_z_7d", "beta_to_btc_change_5d"]
    print(f"\n  Feature distribution comparison (best vs worst months):", flush=True)
    print(f"  {'feature':<28}  {'best':>10}  {'worst':>10}  {'diff':>8}", flush=True)
    for f in features_to_compare:
        if f not in best_with_feat.columns: continue
        b = best_with_feat[f].dropna(); w = worst_with_feat[f].dropna()
        if len(b) < 10 or len(w) < 10: continue
        d = b.mean() - w.mean()
        print(f"  {f:<28}  {b.mean():>+10.5f}  {w.mean():>+10.5f}  {d:>+8.5f}",
              flush=True)

    # ---------- 3. Per-fold pred-target signal stability ----------
    print(f"\n=== 3. Per-fold pred → target_A signal strength ===", flush=True)
    apd_oos = apd[apd["fold"].isin(range(1, 10))].dropna(subset=["pred", "alpha_A"])
    print(f"  {'fold':>4}  {'n':>9}  {'IC':>8}  {'pred_std':>10}  {'alpha_std':>10}  {'pred_kurt':>10}",
          flush=True)
    for f, g in apd_oos.groupby("fold"):
        ic = g["pred"].rank().corr(g["alpha_A"].rank())
        kurt = g["pred"].kurtosis()
        print(f"  {int(f):>4}  {len(g):>9d}  {ic:>+8.4f}  "
              f"{g['pred'].std():>10.4f}  {g['alpha_A'].std():>10.4f}  {kurt:>+10.2f}",
              flush=True)

    # ---------- 4. Pred distribution moments × IC ----------
    print(f"\n=== 4. Per-cycle pred-distribution moments × IC ===", flush=True)
    rows = []
    for t, g in apd_oos.groupby("open_time"):
        if len(g) < 10: continue
        ic = g["pred"].rank().corr(g["alpha_A"].rank())
        if pd.isna(ic): continue
        rows.append({
            "open_time": t, "fold": g["fold"].iloc[0],
            "n_syms": len(g),
            "pred_std": g["pred"].std(),
            "pred_iqr": g["pred"].quantile(0.75) - g["pred"].quantile(0.25),
            "pred_range": g["pred"].max() - g["pred"].min(),
            "pred_skew": g["pred"].skew(),
            "pred_kurt": g["pred"].kurtosis(),
            "ic": ic,
        })
    pcic = pd.DataFrame(rows)
    print(f"  Correlations with per-cycle IC:", flush=True)
    for col in ["pred_std", "pred_iqr", "pred_range", "pred_skew", "pred_kurt"]:
        rho = pcic[col].corr(pcic["ic"])
        rho_s = pcic[col].rank().corr(pcic["ic"].rank())
        print(f"    {col:<12}  Pearson = {rho:+.4f}   Spearman = {rho_s:+.4f}",
              flush=True)

    # Bucket by pred_kurt
    pcic["kurt_b"] = pd.qcut(pcic["pred_kurt"], 5, labels=False, duplicates="drop")
    print(f"\n  IC by pred_kurt quintile:", flush=True)
    for b, g in pcic.groupby("kurt_b"):
        print(f"    kurt_q{int(b)}  range=[{g['pred_kurt'].min():+.2f}, {g['pred_kurt'].max():+.2f}]  "
              f"mean_IC={g['ic'].mean():+.4f}  median={g['ic'].median():+.4f}  n={len(g)}",
              flush=True)

    # ---------- 5. Wrong-pick feature analysis ----------
    print(f"\n=== 5. Wrong-pick feature analysis (per-symbol basket members) ===",
          flush=True)
    # For each cycle, look at each basket member: did they go in our predicted direction?
    # "Right pick" = picked long AND alpha > 0, or picked short AND alpha < 0
    # "Wrong pick" = picked long AND alpha < 0, or picked short AND alpha > 0
    rows = []
    for sleeve_rec in sleeves.itertuples():
        if not sleeve_rec.traded: continue
        t = sleeve_rec.time
        g = panel[panel["open_time"] == t][["symbol", "target_A", "return_pct", "atr_pct",
                                                  "funding_rate", "corr_to_btc_1d",
                                                  "idio_vol_1d_vs_bk", "return_1d"]]
        for s in sleeve_rec.long_basket:
            row = g[g["symbol"] == s]
            if len(row) == 0: continue
            alpha = row["target_A"].iloc[0]
            if pd.isna(alpha): continue
            rows.append({**row.iloc[0].to_dict(), "side": "long", "correct": alpha > 0})
        for s in sleeve_rec.short_basket:
            row = g[g["symbol"] == s]
            if len(row) == 0: continue
            alpha = row["target_A"].iloc[0]
            if pd.isna(alpha): continue
            rows.append({**row.iloc[0].to_dict(), "side": "short", "correct": alpha < 0})
    picks = pd.DataFrame(rows)
    print(f"  Total picks analyzed: {len(picks)}", flush=True)
    print(f"  Overall correct rate: {picks['correct'].mean()*100:.1f}%", flush=True)
    print(f"  Long correct rate:    {picks[picks['side']=='long']['correct'].mean()*100:.1f}%",
          flush=True)
    print(f"  Short correct rate:   {picks[picks['side']=='short']['correct'].mean()*100:.1f}%",
          flush=True)

    print(f"\n  Feature comparison: correct vs wrong picks:", flush=True)
    print(f"  {'feature':<22}  {'correct':>10}  {'wrong':>10}  {'diff':>8}", flush=True)
    cmp_features = ["atr_pct", "funding_rate", "corr_to_btc_1d",
                       "idio_vol_1d_vs_bk", "return_1d"]
    for f in cmp_features:
        if f not in picks.columns: continue
        c = picks[picks["correct"]][f].dropna()
        w = picks[~picks["correct"]][f].dropna()
        if len(c) < 10 or len(w) < 10: continue
        print(f"  {f:<22}  {c.mean():>+10.5f}  {w.mean():>+10.5f}  {c.mean()-w.mean():>+8.5f}",
              flush=True)

    # Per-symbol correct rate
    print(f"\n  Per-symbol correct rate (worst 10 + best 10):", flush=True)
    sym_correct = picks.groupby("symbol")["correct"].agg(["mean", "count"])
    sym_correct = sym_correct[sym_correct["count"] >= 50].sort_values("mean")
    print(f"  Bottom 10 (model worst):", flush=True)
    for sym, r in sym_correct.head(10).iterrows():
        print(f"    {sym:<14}  correct_rate = {r['mean']*100:>5.1f}%  n = {int(r['count'])}",
              flush=True)
    print(f"  Top 10 (model best):", flush=True)
    for sym, r in sym_correct.tail(10).iterrows():
        print(f"    {sym:<14}  correct_rate = {r['mean']*100:>5.1f}%  n = {int(r['count'])}",
              flush=True)


if __name__ == "__main__":
    main()
