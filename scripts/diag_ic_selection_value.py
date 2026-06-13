"""How much PnL value does the IC-ranking selector actually add on the 51-panel?

Test: at each boundary, simulate the next 90d K=3 long/short basket spread using
  (a) top-15 by past 180d IC (production)
  (b) top-15 by a random draw of the eligible pool, repeated 50x (placebo)
  (c) ALL eligible symbols (no universe filter)
  (d) bottom-15 by past 180d IC (anti-signal)

If (a) does not materially beat (b)'s median and (c) is competitive, then the
selector is adding little real value — and any universe perturbation just moves
the noisy basket around with proportional Sharpe variance.

Uses the real predictions; no retraining. Compares mean per-cycle long-short spread
(not full V3.1 — this isolates the universe-selection step from sleeve overlay).
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
K = 3
MIN_OBS = 100
MIN_HIST = 60
OOS_FOLDS = list(range(1, 10))
HORIZON = 48
BAR_MS = 5 * 60 * 1000
CYCLES_PER_YEAR = (288 * 365) / HORIZON


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


def cycle_spread(future_df, universe, k=K):
    """Mean per-cycle naked top-K minus bottom-K spread in bps over the period."""
    if not universe: return np.nan, 0
    df = future_df[future_df["symbol"].isin(universe)]
    df = df[df["open_time"].isin(sorted(df["open_time"].unique())[::HORIZON])]
    spreads = []
    for t, g in df.groupby("open_time"):
        if len(g) < 2 * k + 1: continue
        pred = g["pred"].to_numpy(); ret = g["return_pct"].to_numpy()
        top = np.argpartition(-pred, k - 1)[:k]
        bot = np.argpartition(pred, k - 1)[:k]
        spreads.append((ret[top].mean() - ret[bot].mean()) * 1e4)
    return np.array(spreads)


def sharpe(arr):
    if len(arr) < 2 or arr.std() == 0: return 0.0
    return float(arr.mean() / arr.std() * np.sqrt(CYCLES_PER_YEAR))


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

    oos = apd[apd["fold"].isin(OOS_FOLDS)]
    oos_times = sorted(oos["open_time"].unique())
    sampled = oos_times[::HORIZON]
    t0_ms = int(pd.Timestamp(sampled[0]).timestamp() * 1000)
    update_ms = IC_UPDATE_DAYS * 288 * BAR_MS
    window_ms = IC_WINDOW_DAYS * 288 * BAR_MS
    boundaries = sorted({t0_ms + ((int(pd.Timestamp(t).timestamp()*1000)-t0_ms)//update_ms)*update_ms
                         for t in sampled})

    rng = np.random.default_rng(7)
    rows = []
    for b in boundaries:
        ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HIST)
        elig = {s for s in apd["symbol"].unique() if listings.get(s) and listings[s] <= cutoff}
        past = apd[(apd["t_int"] >= b - window_ms) & (apd["t_int"] < b) &
                   (apd["exit_t_int"] <= b) & (apd["symbol"].isin(elig))]
        future = apd[(apd["t_int"] >= b) & (apd["t_int"] < b + update_ms) &
                     (apd["fold"].isin(OOS_FOLDS)) & (apd["symbol"].isin(elig))]
        if len(past) < 1000 or future.empty:
            continue
        ics = past.groupby("symbol").apply(
            lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= MIN_OBS else np.nan
        ).dropna().sort_values(ascending=False)
        if len(ics) < TOP_N: continue

        top = set(ics.head(TOP_N).index)
        bot = set(ics.tail(TOP_N).index)
        all_e = set(ics.index)

        s_top = cycle_spread(future, top)
        s_bot = cycle_spread(future, bot)
        s_all = cycle_spread(future, all_e)
        # placebo: 50 random top-15 picks from the eligible pool
        placebo_sh = []
        elig_list = sorted(all_e)
        for _ in range(50):
            pick = set(rng.choice(elig_list, TOP_N, replace=False))
            s = cycle_spread(future, pick)
            placebo_sh.append(sharpe(s))
        rows.append({
            "boundary": ts.strftime("%Y-%m-%d"),
            "n_elig": len(all_e),
            "sh_top15_ic": round(sharpe(s_top), 2),
            "sh_bot15_ic": round(sharpe(s_bot), 2),
            "sh_all_eligible": round(sharpe(s_all), 2),
            "sh_placebo_median": round(float(np.median(placebo_sh)), 2),
            "sh_placebo_p95": round(float(np.percentile(placebo_sh, 95)), 2),
            "sh_placebo_p05": round(float(np.percentile(placebo_sh, 5)), 2),
            "rank_pctile_top15_in_placebo": round(
                float((np.array(placebo_sh) < sharpe(s_top)).mean()) * 100, 0),
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print("\n=== Naked top-K minus bottom-K spread Sharpe in next 90d ===", flush=True)
    print(df.to_string(index=False), flush=True)

    print("\n=== AGGREGATE ===", flush=True)
    print(f"  Mean Sharpe — top15(IC):      {df['sh_top15_ic'].mean():+.2f}", flush=True)
    print(f"  Mean Sharpe — bot15(IC):      {df['sh_bot15_ic'].mean():+.2f}", flush=True)
    print(f"  Mean Sharpe — all eligible:   {df['sh_all_eligible'].mean():+.2f}", flush=True)
    print(f"  Mean Sharpe — placebo median: {df['sh_placebo_median'].mean():+.2f}", flush=True)
    print(f"  Mean rank-pctile of top15 in placebo: {df['rank_pctile_top15_in_placebo'].mean():.0f}", flush=True)
    out = REPO / "outputs/vBTC_ic_selection_value.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
