"""Root-cause analysis: why do worst cycles happen?

For each of the worst 30 cycles in production:
  1. Picked long/short names and their actual returns
  2. Predicted alpha for each pick vs realized alpha
  3. Dispersion at the time (predictability signal)
  4. Whether universe selection was right (top-K matched realized)
  5. Single-name vs broad failure pattern

Categorizes failures:
  A. Wrong-direction predictions (model error)
  B. High dispersion but wrong selection (model confident-wrong)
  C. Low dispersion (signal weak, conv_gate should have skipped)
  D. Specific-name explosion (universe member behaved wildly)
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice, _train

PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
OUT_DIR = REPO / "outputs/vBTC_dd_root_cause"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
SEEDS = (42, 1337, 7, 19, 2718, 99, 777, 123, 456, 789)
COST_PER_LEG = 4.5
GATE_LOOKBACK = 252
GATE_PCTILE = 0.30
PM_M = 2
PM_BAND = 1.0
MIN_OBS_PER_SYM = 100
TARGET_N = 15
K = 3
ALL_FOLDS = list(range(10))
PROD_FOLDS = [5, 6, 7, 8, 9]

V6_CLEAN_28 = list(XS_FEATURE_COLS_V6_CLEAN)
ALL_DROPS = [
    "return_1d_xs_rank", "bk_ret_48b", "volume_ma_50",
    "ema_slope_20_1h", "ema_slope_20_1h_xs_rank",
    "vwap_zscore_xs_rank", "vwap_zscore",
    "atr_pct_xs_rank", "dom_z_7d_vs_bk", "obv_z_1d_xs_rank",
    "obv_signal", "price_volume_corr_10",
    "hour_cos", "hour_sin",
]
FUNDING_LEAN = ["funding_rate", "funding_rate_z_7d"]
ADD_CROSS_BTC = ["corr_to_btc_1d", "idio_vol_to_btc_1h", "beta_to_btc_change_5d"]
ADD_MORE_FUNDING = ["funding_rate_1d_change", "funding_streak_pos"]
WINNER_21 = [f for f in V6_CLEAN_28 if f not in ALL_DROPS] + FUNDING_LEAN + ADD_CROSS_BTC + ADD_MORE_FUNDING


def train_fold(panel, fold, feat_set):
    train, cal, test = _slice(panel, fold)
    tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
    ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
    if len(tr) < 1000 or len(ca) < 200: return None, None
    Xt = tr[feat_set].to_numpy(np.float32)
    Xc = ca[feat_set].to_numpy(np.float32)
    Xtest = test[feat_set].to_numpy(np.float32)
    yt = tr["target_A"].to_numpy(np.float32)
    yc = ca["target_A"].to_numpy(np.float32)
    mask_t = ~np.isnan(yt); mask_c = ~np.isnan(yc)
    if mask_t.sum() < 1000 or mask_c.sum() < 200: return None, None
    preds = []
    for s in SEEDS:
        m = _train(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=s)
        preds.append(m.predict(Xtest, num_iteration=m.best_iteration))
    return test.copy(), np.mean(preds, axis=0)


def evaluate_with_full_logging(test_df, universe, top_k=K, sample_every=HORIZON):
    """Like evaluate but logs picked names, predictions, returns per cycle."""
    df = test_df.copy()
    times = sorted(df["open_time"].unique())
    if not times: return pd.DataFrame()
    keep_times = set(times[::sample_every])
    df = df[df["open_time"].isin(keep_times)]
    band_k = max(top_k, int(round(PM_BAND * top_k)))
    history = []
    dispersion_history = deque(maxlen=GATE_LOOKBACK)
    cur_long, cur_short = set(), set()
    cycle_logs = []

    for t, g in df.groupby("open_time"):
        g_u = g[g["symbol"].isin(universe)]
        if len(g_u) < 2 * top_k + 1:
            continue
        sym_arr = g_u["symbol"].to_numpy()
        pred_arr = g_u["pred"].to_numpy()
        ret_arr = g_u["return_pct"].to_numpy()
        idx_top = np.argpartition(-pred_arr, top_k - 1)[:top_k]
        idx_bot = np.argpartition(pred_arr, top_k - 1)[:top_k]
        dispersion = float(pred_arr[idx_top].mean() - pred_arr[idx_bot].mean())

        # Conv gate check
        skip = False
        gate_pctile = None
        if len(dispersion_history) >= 30:
            past = list(dispersion_history)
            gate_pctile = (np.array(past) < dispersion).mean()
            thr = float(np.quantile(past, GATE_PCTILE))
            if dispersion < thr: skip = True
        dispersion_history.append(dispersion)

        # PM gate
        bk = min(band_k, len(g_u))
        idx_top_band = np.argpartition(-pred_arr, bk - 1)[:bk] if bk < len(g_u) else np.arange(len(g_u))
        idx_bot_band = np.argpartition(pred_arr, bk - 1)[:bk] if bk < len(g_u) else np.arange(len(g_u))
        history.append({"long": set(sym_arr[idx_top_band]), "short": set(sym_arr[idx_bot_band])})
        if len(history) > PM_M: history = history[-PM_M:]

        if skip:
            # Hold-through
            if cur_long or cur_short:
                long_g = g[g["symbol"].isin(cur_long)]
                short_g = g[g["symbol"].isin(cur_short)]
                long_ret = long_g["return_pct"].mean() if not long_g.empty else 0.0
                short_ret = short_g["return_pct"].mean() if not short_g.empty else 0.0
                spread = (long_ret - short_ret) * 1e4
            else:
                spread = 0.0
            cycle_logs.append({"time": t, "skipped": 1, "dispersion": dispersion,
                                "gate_pctile": gate_pctile, "net_bps": spread,
                                "longs": list(cur_long), "shorts": list(cur_short),
                                "long_returns": "", "short_returns": "",
                                "long_preds": "", "short_preds": ""})
            continue

        # PM gate
        cand_long = set(sym_arr[idx_top]); cand_short = set(sym_arr[idx_bot])
        if len(history) >= PM_M:
            past_long = [h["long"] for h in history[-PM_M:][:PM_M-1]]
            past_short = [h["short"] for h in history[-PM_M:][:PM_M-1]]
            new_long = cur_long & cand_long
            new_short = cur_short & cand_short
            for s in cand_long - cur_long:
                if all(s in p for p in past_long): new_long.add(s)
            for s in cand_short - cur_short:
                if all(s in p for p in past_short): new_short.add(s)
            if len(new_long) > top_k:
                ranked = sorted(new_long, key=lambda s: -pred_arr[sym_arr == s][0])[:top_k]
                new_long = set(ranked)
            if len(new_short) > top_k:
                ranked = sorted(new_short, key=lambda s: pred_arr[sym_arr == s][0])[:top_k]
                new_short = set(ranked)
        else:
            new_long, new_short = cand_long, cand_short

        if not new_long or not new_short:
            cycle_logs.append({"time": t, "skipped": 0, "dispersion": dispersion,
                                "gate_pctile": gate_pctile, "net_bps": 0.0,
                                "longs": [], "shorts": [], "long_returns": "",
                                "short_returns": "", "long_preds": "", "short_preds": ""})
            continue

        # Compute PnL with logging
        long_g = g_u[g_u["symbol"].isin(new_long)]
        short_g = g_u[g_u["symbol"].isin(new_short)]
        long_ret = long_g["return_pct"].mean()
        short_ret = short_g["return_pct"].mean()
        spread = (long_ret - short_ret) * 1e4
        churn_long = len(new_long.symmetric_difference(cur_long)) / max(len(new_long | cur_long), 1)
        churn_short = len(new_short.symmetric_difference(cur_short)) / max(len(new_short | cur_short), 1)
        cost = (churn_long + churn_short) * COST_PER_LEG
        net = spread - cost

        long_data = long_g[["symbol", "pred", "return_pct"]].to_dict("records")
        short_data = short_g[["symbol", "pred", "return_pct"]].to_dict("records")
        cycle_logs.append({
            "time": t, "skipped": 0, "dispersion": dispersion, "gate_pctile": gate_pctile,
            "net_bps": net, "spread_bps": spread, "cost_bps": cost,
            "longs": [d["symbol"] for d in long_data],
            "shorts": [d["symbol"] for d in short_data],
            "long_returns": ",".join(f"{d['symbol']}:{d['return_pct']*1e4:+.0f}" for d in long_data),
            "short_returns": ",".join(f"{d['symbol']}:{d['return_pct']*1e4:+.0f}" for d in short_data),
            "long_preds": ",".join(f"{d['symbol']}:{d['pred']:+.4f}" for d in long_data),
            "short_preds": ",".join(f"{d['symbol']}:{d['pred']:+.4f}" for d in short_data),
        })
        cur_long, cur_short = new_long, new_short

    return pd.DataFrame(cycle_logs)


def main():
    print(f"Loading panel...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    feat_set = [f for f in WINNER_21 if f in panel.columns]
    folds_all = _multi_oos_splits(panel)

    print(f"\n=== Train all 10 folds × 10 seeds ===", flush=True)
    fold_data = {}
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        td, p = train_fold(panel, folds_all[fid], feat_set)
        if td is not None: fold_data[fid] = (td, p)
        print(f"  fold {fid}: ({time.time()-t0:.0f}s)", flush=True)

    rows = []
    for fid, (td, p) in fold_data.items():
        df = td[["symbol", "open_time", "alpha_A", "return_pct"]].copy()
        df["pred"] = p; df["fold"] = fid
        rows.append(df)
    apd = pd.concat(rows, ignore_index=True).sort_values(["open_time", "symbol"])

    # Static universe (calib folds 0-4)
    calib = apd[apd["fold"].isin([0, 1, 2, 3, 4])].dropna(subset=["alpha_A"])
    static_ics = calib.groupby("symbol").apply(
        lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= MIN_OBS_PER_SYM else np.nan
    ).dropna().sort_values(ascending=False)
    static_universe = set(static_ics.head(TARGET_N).index.tolist())

    prod_pred = apd[apd["fold"].isin(PROD_FOLDS)].copy()
    test_data = prod_pred[["symbol", "open_time", "pred", "return_pct", "alpha_A"]].copy()

    print(f"\n=== Evaluating with full logging ===", flush=True)
    df_log = evaluate_with_full_logging(test_data, static_universe, top_k=K)
    print(f"  {len(df_log)} cycles logged", flush=True)

    # Worst 20 cycles
    df_active = df_log[df_log["skipped"] == 0].copy()
    worst = df_active.nsmallest(20, "net_bps")

    print(f"\n=== Worst 20 cycles (root cause analysis) ===", flush=True)
    print(f"  {'time':<22} {'net':>7} {'disp':>7} {'pct':>5}  longs / shorts / returns",
          flush=True)
    for _, r in worst.iterrows():
        t_str = pd.Timestamp(r["time"]).strftime("%Y-%m-%d %H:%M")
        print(f"\n  {t_str:<22} {r['net_bps']:>+7.0f}  disp={r['dispersion']:+.4f}  "
              f"pct={r['gate_pctile'] or 0:.2f}", flush=True)
        print(f"    longs:  {r['long_returns']}", flush=True)
        print(f"    shorts: {r['short_returns']}", flush=True)
        print(f"    L preds: {r['long_preds']}", flush=True)
        print(f"    S preds: {r['short_preds']}", flush=True)

    # Aggregate analysis
    print(f"\n=== Pattern analysis on worst 30 cycles ===", flush=True)
    worst30 = df_active.nsmallest(30, "net_bps")

    # 1. Were they low-dispersion (gate should have skipped)?
    low_disp = (worst30["gate_pctile"] < 0.50).sum()
    print(f"  Low-dispersion (pctile < 50): {low_disp}/30 (gate threshold is 30 pctile)", flush=True)
    print(f"  Mean dispersion percentile of worst-30: {worst30['gate_pctile'].mean():.2f}",
          flush=True)
    print(f"  vs all-cycle mean: {df_active['gate_pctile'].mean():.2f}", flush=True)

    # 2. Single-name explosions: was there a name with extreme adverse return?
    print(f"\n  Single-name extremes in worst-30 cycles:", flush=True)
    extremes = []
    for _, r in worst30.iterrows():
        # Parse long/short return strings
        for side, s in [("long", r["long_returns"]), ("short", r["short_returns"])]:
            for entry in s.split(","):
                if not entry: continue
                sym, ret_str = entry.split(":")
                ret_bps = float(ret_str)
                if side == "long" and ret_bps < -300:   # long that lost > 3%
                    extremes.append({"time": r["time"], "side": side, "symbol": sym,
                                      "ret_bps": ret_bps})
                elif side == "short" and ret_bps > 300:   # short that lost (rallied) > 3%
                    extremes.append({"time": r["time"], "side": side, "symbol": sym,
                                      "ret_bps": ret_bps})
    df_ext = pd.DataFrame(extremes)
    if not df_ext.empty:
        print(f"  Extreme adverse moves in worst-30: {len(df_ext)} occurrences", flush=True)
        sym_counts = df_ext.groupby("symbol").size().sort_values(ascending=False)
        print(f"  Most-frequent culprits:", flush=True)
        for s, c in sym_counts.head(5).items():
            mean_ret = df_ext[df_ext["symbol"] == s]["ret_bps"].mean()
            print(f"    {s}: {c} occurrences, mean adverse move {mean_ret:+.0f} bps", flush=True)

    df_log.to_csv(OUT_DIR / "cycle_logs.csv", index=False)
    worst30.to_csv(OUT_DIR / "worst30_cycles.csv", index=False)
    if not df_ext.empty:
        df_ext.to_csv(OUT_DIR / "extreme_adverse_moves.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
