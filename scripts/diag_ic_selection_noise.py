"""Diagnose vBTC IC-ranking universe selection: noise vs signal.

For every 90-day refresh boundary in OOS:
  1. Compute per-symbol Spearman IC over the prior 180d (the picker's input)
  2. Compute the next 90d per-symbol IC realised (what the picker is trying to forecast)
  3. Measure rank persistence top-15(past) -> top-15(future)
  4. Standard-error of per-symbol IC estimate vs the rank-15/rank-16 gap
  5. Cross-boundary churn of the top-15 set

This isolates whether the failure is the selector's *input* (IC estimate noise) or
its *signal* (no persistence) or both.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

APD = REPO / "outputs/vBTC_audit_panel/all_predictions.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"

IC_WINDOW_DAYS = 180
IC_UPDATE_DAYS = 90
TOP_N = 15
MIN_OBS = 100
MIN_HIST = 60
OOS_FOLDS = list(range(1, 10))
BAR_MS = 5 * 60 * 1000


def to_ms_int(s):
    ts = pd.to_datetime(s)
    if hasattr(ts.dtype, "tz") and ts.dtype.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return ts.astype("datetime64[ms]").astype("int64").to_numpy()


def get_listings():
    L = {}
    for d in KLINES_DIR.iterdir():
        if not d.is_dir(): continue
        m5 = d / "5m"
        if not m5.exists(): continue
        f = sorted(m5.glob("*.parquet"))
        if not f: continue
        try: L[d.name] = pd.Timestamp(f[0].stem, tz="UTC")
        except Exception: pass
    return L


def per_sym_ic(g):
    if len(g) < MIN_OBS: return np.nan
    return g["pred"].rank().corr(g["alpha_A"].rank())


def bootstrap_ic_se(g, n_boot=200, rng=None):
    """Standard error of Spearman IC for one symbol within a window."""
    if len(g) < MIN_OBS: return np.nan
    rng = rng or np.random.default_rng(0)
    n = len(g)
    p = g["pred"].to_numpy(); a = g["alpha_A"].to_numpy()
    ics = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ics.append(pd.Series(p[idx]).rank().corr(pd.Series(a[idx]).rank()))
    return float(np.nanstd(ics))


def main():
    print("Loading audit panel...", flush=True)
    apd = pd.read_parquet(APD)
    apd = apd.dropna(subset=["alpha_A"]).copy()
    apd["t_int"] = to_ms_int(apd["open_time"])
    apd["exit_t_int"] = to_ms_int(apd["exit_time"]) if "exit_time" in apd.columns else apd["t_int"] + 48*BAR_MS
    listings = get_listings()
    panel_first = apd.groupby("symbol")["open_time"].min()
    for s, t in panel_first.items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t

    # Compute boundaries the same way the picker does
    oos = apd[apd["fold"].isin(OOS_FOLDS)]
    oos_times = sorted(oos["open_time"].unique())
    sampled = oos_times[::48]  # HORIZON
    t0_ms = int(pd.Timestamp(sampled[0]).timestamp() * 1000)
    update_ms = IC_UPDATE_DAYS * 288 * BAR_MS
    window_ms = IC_WINDOW_DAYS * 288 * BAR_MS
    boundaries = sorted({t0_ms + ((int(pd.Timestamp(t).timestamp()*1000)-t0_ms)//update_ms)*update_ms
                         for t in sampled})
    print(f"Boundaries: {len(boundaries)}", flush=True)

    rng = np.random.default_rng(42)
    rows = []
    prev_top = None
    for i, b in enumerate(boundaries):
        ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HIST)
        elig = {s for s in apd["symbol"].unique()
                if listings.get(s) and listings[s] <= cutoff}

        past = apd[(apd["t_int"] >= b - window_ms) &
                   (apd["t_int"] < b) &
                   (apd["exit_t_int"] <= b) &
                   (apd["symbol"].isin(elig))]
        future = apd[(apd["t_int"] >= b) &
                     (apd["t_int"] < b + update_ms) &
                     (apd["symbol"].isin(elig))]

        ics_past = past.groupby("symbol").apply(per_sym_ic).dropna()
        ics_future = future.groupby("symbol").apply(per_sym_ic).dropna()

        common = ics_past.index.intersection(ics_future.index)
        ics_past_c = ics_past.loc[common].sort_values(ascending=False)
        ics_future_c = ics_future.loc[common]

        # Standard error of each past-IC estimate (bootstrap, small subset for speed)
        ses = {}
        for sym in ics_past_c.index:
            g = past[past["symbol"] == sym][["pred", "alpha_A"]]
            ses[sym] = bootstrap_ic_se(g, n_boot=100, rng=rng)
        se_med = float(np.nanmedian(list(ses.values())))

        if len(ics_past_c) < TOP_N + 1:
            continue
        gap_15_16 = float(ics_past_c.iloc[TOP_N-1] - ics_past_c.iloc[TOP_N])
        # Rank-correlation of past vs future
        rho = ics_past_c.rank().corr(ics_future_c.rank())

        top_past = set(ics_past_c.head(TOP_N).index)
        top_future = set(ics_future_c.sort_values(ascending=False).head(TOP_N).index)
        overlap = len(top_past & top_future)
        # Expected random overlap = TOP_N * TOP_N / len(common)
        expected = TOP_N * TOP_N / len(common)

        # Churn vs previous boundary's top-15
        churn = None if prev_top is None else len(top_past.symmetric_difference(prev_top)) / 2
        prev_top = top_past

        # Realised next-90d basket Sharpe-proxy: mean future IC of past-top-15
        mean_fut_ic_top = float(ics_future_c.loc[list(top_past & set(ics_future_c.index))].mean())
        mean_fut_ic_all = float(ics_future_c.mean())

        rows.append({
            "boundary": ts.strftime("%Y-%m-%d"),
            "n_elig": len(common),
            "past_ic_median": float(ics_past_c.median()),
            "past_ic_max": float(ics_past_c.iloc[0]),
            "past_ic_min": float(ics_past_c.iloc[-1]),
            "past_ic_std_xs": float(ics_past_c.std()),
            "se_median_intra_sym": se_med,
            "gap_rank15_rank16": gap_15_16,
            "rho_past_future": float(rho) if not pd.isna(rho) else np.nan,
            "overlap_top15_past_future": overlap,
            "overlap_expected_random": round(expected, 1),
            "churn_top15_vs_prev": churn,
            "mean_fut_ic_picked_top15": mean_fut_ic_top,
            "mean_fut_ic_all": mean_fut_ic_all,
            "lift_picked_minus_all": mean_fut_ic_top - mean_fut_ic_all,
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n=== PER-BOUNDARY DIAGNOSTIC ===", flush=True)
    print(df.to_string(index=False), flush=True)

    print("\n=== AGGREGATES ===", flush=True)
    print(f"  Median past-IC cross-sectional std            : {df['past_ic_std_xs'].median():.4f}", flush=True)
    print(f"  Median intra-symbol IC bootstrap SE           : {df['se_median_intra_sym'].median():.4f}", flush=True)
    print(f"  Median rank15-rank16 gap (signal)             : {df['gap_rank15_rank16'].median():.4f}", flush=True)
    print(f"  Signal/noise (gap / SE)                       : {df['gap_rank15_rank16'].median()/df['se_median_intra_sym'].median():.2f}", flush=True)
    print(f"  Median rho(past_IC_rank, future_IC_rank)      : {df['rho_past_future'].median():+.3f}", flush=True)
    print(f"  Mean overlap top15(past, future)              : {df['overlap_top15_past_future'].mean():.1f}", flush=True)
    print(f"  Random expected overlap                        : {df['overlap_expected_random'].mean():.1f}", flush=True)
    print(f"  Mean churn boundary-over-boundary              : {df['churn_top15_vs_prev'].dropna().mean():.1f}", flush=True)
    print(f"  Picked top15 future-IC lift vs all (mean)     : {df['lift_picked_minus_all'].mean():+.4f}", flush=True)
    print(f"  Picked top15 future-IC lift std               : {df['lift_picked_minus_all'].std():.4f}", flush=True)
    n_lift_pos = (df['lift_picked_minus_all'] > 0).sum()
    print(f"  Boundaries where picked beats panel-avg       : {n_lift_pos}/{len(df)}", flush=True)

    out = REPO / "outputs/vBTC_ic_selection_diag.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
