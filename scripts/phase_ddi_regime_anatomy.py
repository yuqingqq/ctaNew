"""Phase DDI: data-driven insights into regime/noise structure.

Four passes, all on existing data (no new features, no retrain):

  1. 2D regime heatmap: cohort PnL by joint (btc_rvol_7d × btc_ret_3d) bucket.
     Find corners where strategy consistently wins/loses that 1D missed.

  2. Loss anatomy: worst 10% of cycles by V3.1 net PnL. What features do they
     share? Compare feature means/medians vs full sample.

  3. Win anatomy: top 10% of cycles. What features distinguish winners from
     losers? Compute t-stat / effect size per feature.

  4. Time-trend: monthly mean PnL trajectory. Detect regime drift.

  5. Pred-quality stability: is per-cycle IC predictable from observable
     features at entry time? If yes, gate on predicted-IC; if no, IC is noise.

Output: actionable insights about WHEN the strategy works, distinct from
prior single-dimension regime analysis.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

OUT = REPO / "outputs/vBTC_ddi"
OUT.mkdir(parents=True, exist_ok=True)


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt((288 * 365) / 48))


def load_btc():
    bdir = REPO / "data/ml/test/parquet/klines/BTCUSDT/5m"
    files = sorted(bdir.glob("*.parquet"))
    dfs = [pd.read_parquet(f, columns=["open_time", "close", "volume", "high", "low"])
            for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True).astype("datetime64[ns, UTC]")
    df = df.dropna().drop_duplicates("open_time").set_index("open_time").sort_index()
    df["ret_5m"] = np.log(df["close"]).diff()
    out = pd.DataFrame(index=df.index)
    out["btc_ret_24h"] = df["close"].pct_change(288)
    out["btc_ret_3d"] = df["close"].pct_change(288 * 3)
    out["btc_ret_7d"] = df["close"].pct_change(288 * 7)
    out["btc_rvol_24h"] = df["ret_5m"].rolling(288).std() * np.sqrt(288)
    out["btc_rvol_3d"] = df["ret_5m"].rolling(288 * 3).std() * np.sqrt(288)
    out["btc_rvol_7d"] = df["ret_5m"].rolling(288 * 7).std() * np.sqrt(288)
    out["btc_range_4h"] = (df["high"].rolling(48).max() - df["low"].rolling(48).min()) / df["close"]
    out["btc_dvol_24h"] = (df["close"] * df["volume"]).rolling(288).sum()
    return out


def main():
    print("=== Phase DDI: regime / noise anatomy ===\n", flush=True)

    # Load V3.1 cycle PnL + cohort PnL
    cycles = pd.read_csv(REPO / "outputs/vBTC_sleeve_horizon/per_cycle_robust_equal_6.csv")
    cycles["time"] = pd.to_datetime(cycles["time"], utc=True).astype("datetime64[ns, UTC]")
    cycles["traded"] = cycles["gross_exposure"] > 0.01
    cohorts = pd.read_csv(REPO / "outputs/vBTC_iter_loop/iter2_cohort_pnl.csv")
    cohorts["time"] = pd.to_datetime(cohorts["time"], utc=True).astype("datetime64[ns, UTC]")
    print(f"  Loaded {len(cycles):,} V3.1 cycles ({cycles['traded'].sum()} traded)",
          flush=True)
    print(f"  Loaded {len(cohorts):,} cohort 24h PnL records", flush=True)

    # Attach BTC features at cycle entry time
    print(f"  loading BTC features...", flush=True)
    btc = load_btc()
    cycles = cycles.sort_values("time")
    cycles = pd.merge_asof(cycles, btc.reset_index().rename(columns={"open_time": "time"}),
                                on="time", direction="backward",
                                tolerance=pd.Timedelta("5min"))
    # cohorts CSV already has BTC features attached from prior diagnostic
    # only attach btc_ret_7d and btc_range_4h which weren't in prior diagnostic
    missing = [c for c in ["btc_ret_7d", "btc_range_4h"] if c not in cohorts.columns]
    if missing:
        cohorts = cohorts.sort_values("time")
        cohorts = pd.merge_asof(cohorts, btc[missing].reset_index().rename(
            columns={"open_time": "time"}), on="time", direction="backward",
            tolerance=pd.Timedelta("5min"))
    print(f"  BTC features attached", flush=True)

    # Per-cycle IC from prior diagnostic
    pc_ic = pd.read_csv(REPO / "outputs/vBTC_failure_diag/per_cycle_ic.csv")
    pc_ic["open_time"] = pd.to_datetime(pc_ic["open_time"], utc=True).astype("datetime64[ns, UTC]")
    cohorts = cohorts.merge(pc_ic[["open_time", "ic", "pred_std"]].rename(
        columns={"open_time": "time"}), on="time", how="left")

    # ---------- 1. 2D regime heatmap ----------
    print(f"\n=== 1. 2D regime heatmap: btc_rvol_7d × btc_ret_3d ===", flush=True)
    sub = cohorts.dropna(subset=["btc_rvol_7d", "btc_ret_3d", "cohort_pnl_bps_24h"]).copy()
    sub["rvol_b"] = pd.qcut(sub["btc_rvol_7d"], 5, labels=False, duplicates="drop")
    sub["ret_b"] = pd.qcut(sub["btc_ret_3d"], 5, labels=False, duplicates="drop")
    heatmap_mean = sub.pivot_table(values="cohort_pnl_bps_24h",
                                          index="rvol_b", columns="ret_b",
                                          aggfunc="mean")
    heatmap_sharpe = sub.pivot_table(values="cohort_pnl_bps_24h",
                                             index="rvol_b", columns="ret_b",
                                             aggfunc=_sharpe)
    heatmap_n = sub.pivot_table(values="cohort_pnl_bps_24h",
                                       index="rvol_b", columns="ret_b",
                                       aggfunc="count")
    print(f"\n  Cohort PnL mean (bps) by (rvol_7d quintile × ret_3d quintile):",
          flush=True)
    print(f"  rows = btc_rvol_7d quintile (low → high), cols = btc_ret_3d (low → high)",
          flush=True)
    print(heatmap_mean.round(0).to_string(), flush=True)
    print(f"\n  Cohort Sharpe by 2D regime:", flush=True)
    print(heatmap_sharpe.round(2).to_string(), flush=True)
    print(f"\n  n cohorts by 2D regime:", flush=True)
    print(heatmap_n.fillna(0).astype(int).to_string(), flush=True)

    best_cell = heatmap_mean.stack().idxmax()
    worst_cell = heatmap_mean.stack().idxmin()
    print(f"\n  Best regime cell: (rvol_q={best_cell[0]}, ret_q={best_cell[1]})  "
          f"mean PnL = {heatmap_mean.loc[best_cell]:+.0f} bps  "
          f"Sharpe = {heatmap_sharpe.loc[best_cell]:+.2f}  "
          f"n = {int(heatmap_n.loc[best_cell])}", flush=True)
    print(f"  Worst regime cell: (rvol_q={worst_cell[0]}, ret_q={worst_cell[1]})  "
          f"mean PnL = {heatmap_mean.loc[worst_cell]:+.0f} bps  "
          f"Sharpe = {heatmap_sharpe.loc[worst_cell]:+.2f}  "
          f"n = {int(heatmap_n.loc[worst_cell])}", flush=True)

    # ---------- 2. Loss anatomy ----------
    print(f"\n=== 2. Loss anatomy: worst 10% V3.1 cycles ===", flush=True)
    traded = cycles[cycles["traded"]].copy()
    losers = traded[traded["net_pnl_bps"] <= traded["net_pnl_bps"].quantile(0.10)]
    winners = traded[traded["net_pnl_bps"] >= traded["net_pnl_bps"].quantile(0.90)]
    others = traded[(traded["net_pnl_bps"] > traded["net_pnl_bps"].quantile(0.10)) &
                       (traded["net_pnl_bps"] < traded["net_pnl_bps"].quantile(0.90))]
    print(f"  Losers (n={len(losers)}): mean PnL = {losers['net_pnl_bps'].mean():+.1f}",
          flush=True)
    print(f"  Winners (n={len(winners)}): mean PnL = {winners['net_pnl_bps'].mean():+.1f}",
          flush=True)
    print(f"  Middle (n={len(others)}): mean PnL = {others['net_pnl_bps'].mean():+.1f}",
          flush=True)

    features = ["btc_ret_24h", "btc_ret_3d", "btc_ret_7d",
                  "btc_rvol_24h", "btc_rvol_3d", "btc_rvol_7d",
                  "btc_range_4h", "btc_dvol_24h"]
    print(f"\n  Feature comparison (losers vs winners vs middle):", flush=True)
    print(f"  {'feature':<22}  {'losers':>10}  {'winners':>10}  {'middle':>10}  {'L-W diff':>9}  {'t-stat':>7}",
          flush=True)
    for f in features:
        l = losers[f].dropna(); w = winners[f].dropna(); m = others[f].dropna()
        if len(l) < 5 or len(w) < 5: continue
        diff = l.mean() - w.mean()
        # Pooled t-stat
        se = np.sqrt(l.std()**2/len(l) + w.std()**2/len(w))
        t = diff / max(se, 1e-9)
        flag = "***" if abs(t) > 3 else "**" if abs(t) > 2 else "*" if abs(t) > 1 else ""
        print(f"  {f:<22}  {l.mean():>+10.4f}  {w.mean():>+10.4f}  {m.mean():>+10.4f}  "
              f"{diff:>+9.4f}  {t:>+7.2f} {flag}", flush=True)

    # ---------- 3. Win anatomy (extreme tails) ----------
    print(f"\n=== 3. Top 1% vs bottom 1% extreme tails ===", flush=True)
    extreme_losers = traded[traded["net_pnl_bps"] <= traded["net_pnl_bps"].quantile(0.01)]
    extreme_winners = traded[traded["net_pnl_bps"] >= traded["net_pnl_bps"].quantile(0.99)]
    print(f"  Extreme losers (n={len(extreme_losers)}): mean PnL = "
          f"{extreme_losers['net_pnl_bps'].mean():+.1f}", flush=True)
    print(f"  Extreme winners (n={len(extreme_winners)}): mean PnL = "
          f"{extreme_winners['net_pnl_bps'].mean():+.1f}", flush=True)
    print(f"\n  Feature comparison (extreme tails):", flush=True)
    print(f"  {'feature':<22}  {'ext_L':>10}  {'ext_W':>10}  {'diff':>9}", flush=True)
    for f in features:
        l = extreme_losers[f].dropna(); w = extreme_winners[f].dropna()
        if len(l) < 3 or len(w) < 3: continue
        diff = l.mean() - w.mean()
        print(f"  {f:<22}  {l.mean():>+10.4f}  {w.mean():>+10.4f}  {diff:>+9.4f}",
              flush=True)

    # ---------- 4. Time-trend analysis ----------
    print(f"\n=== 4. Monthly trajectory (is the strategy degrading?) ===", flush=True)
    cycles["yyyy_mm"] = pd.to_datetime(cycles["time"]).dt.to_period("M")
    monthly = cycles.groupby("yyyy_mm").agg(
        n=("net_pnl_bps", "size"),
        traded_n=("traded", "sum"),
        mean_net=("net_pnl_bps", "mean"),
        sum_net=("net_pnl_bps", "sum"),
        sharpe=("net_pnl_bps", _sharpe),
    ).reset_index()
    monthly["yyyy_mm"] = monthly["yyyy_mm"].astype(str)
    print(f"  Month-by-month V3.1 performance:", flush=True)
    print(f"  {'month':<10}  {'n':>5}  {'mean':>7}  {'sum':>8}  {'sharpe':>7}",
          flush=True)
    for _, r in monthly.iterrows():
        print(f"  {r['yyyy_mm']:<10}  {int(r['n']):>5d}  {r['mean_net']:>+7.2f}  "
              f"{r['sum_net']:>+8.0f}  {r['sharpe']:>+7.2f}", flush=True)

    # Trend test: does mean PnL change with time?
    monthly["idx"] = range(len(monthly))
    rho = monthly["idx"].corr(monthly["mean_net"])
    print(f"\n  Spearman(month_idx, mean_PnL) = {rho:+.3f}", flush=True)
    print(f"  → if strongly negative, strategy is degrading over time", flush=True)
    print(f"  → if near zero, performance is stable", flush=True)

    # ---------- 5. Is per-cycle IC predictable? ----------
    print(f"\n=== 5. Is per-cycle IC predictable from BTC regime? ===", flush=True)
    sub_ic = cohorts.dropna(subset=["ic", "btc_rvol_7d", "btc_ret_3d"]).copy()
    print(f"  Correlations of per-cycle IC with BTC regime features:", flush=True)
    for f in features:
        if f in sub_ic.columns:
            rho = sub_ic["ic"].corr(sub_ic[f])
            print(f"    Pearson({f}, IC): {rho:+.4f}", flush=True)

    # The full IC standard deviation is 0.17 — what fraction is predictable?
    from sklearn.linear_model import LinearRegression
    X_cols = [c for c in features if c in sub_ic.columns]
    sub_ic = sub_ic.dropna(subset=X_cols + ["ic"])
    X = sub_ic[X_cols].to_numpy()
    y = sub_ic["ic"].to_numpy()
    lr = LinearRegression().fit(X, y)
    y_pred = lr.predict(X)
    r2 = 1 - ((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"\n  Linear regression: per-cycle IC ~ all BTC regime features", flush=True)
    print(f"  In-sample R²: {r2:.4f}", flush=True)
    print(f"  → R² > 0.05 would suggest a real PIT IC-prediction signal", flush=True)
    print(f"  → R² < 0.02 means IC is mostly unpredictable noise", flush=True)


if __name__ == "__main__":
    main()
