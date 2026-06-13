"""WINNER_17 + β-residual target on both 51-panel and 111-panel.

Test 1: 51-panel with WINNER_17 — does dropping dead features improve β-hedged baseline?
Test 2: 111-panel with WINNER_17 — does β-residual setup transport to expanded universe?
Analysis: per-cycle IC, per-symbol contribution, universe composition on 111.

WINNER_17 = WINNER_21 minus {mfi, price_volume_corr_20, idio_ret_48b_vs_bk, funding_streak_pos}.
Same model architecture, same hyperparameters, same protocol as Phase 1D.
"""
from __future__ import annotations
import sys, time, importlib.util, warnings
from pathlib import Path
from collections import deque, defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice, _train
spec = importlib.util.spec_from_file_location("psl", REPO / "scripts/phase_ah_sleeve.py")
psl = importlib.util.module_from_spec(spec); spec.loader.exec_module(psl)
from ml.research.alpha_v4_xs import block_bootstrap_ci

PANEL_51 = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
PANEL_111 = REPO / "outputs/vBTC_features_expanded/panel_variants_with_funding.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
OUT_DIR = REPO / "outputs/vBTC_winner17_b_residual"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
SEEDS = (42, 1337, 7, 19, 2718)
ALL_FOLDS = list(range(10))
OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
BETA_WIN_PIT_DAYS = 90
CAPITAL = 100.0

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
WINNER_21 = ([f for f in V6_CLEAN_28 if f not in ALL_DROPS]
             + FUNDING_LEAN + ADD_CROSS_BTC + ADD_MORE_FUNDING)
DEAD_WEIGHT = {"mfi", "price_volume_corr_20", "idio_ret_48b_vs_bk", "funding_streak_pos"}
WINNER_17 = [f for f in WINNER_21 if f not in DEAD_WEIGHT]


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(psl.CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def folds_positive(df_v):
    return sum(1 for _, g in df_v.groupby("fold") if _sharpe(g["net_pnl_bps"]) > 0)


def compute_pit_beta(panel, beta_win_days):
    """Compute PIT rolling β to BTC per (symbol, open_time)."""
    print(f"  Computing PIT β with {beta_win_days}d window...", flush=True)
    t0 = time.time()
    btc_ret = panel[panel.symbol == "BTCUSDT"][["open_time", "return_pct"]].rename(
        columns={"return_pct": "btc_ret"}).drop_duplicates("open_time")
    bar_window = beta_win_days * 288
    out = []
    for sym, g in panel.groupby("symbol"):
        gg = g[["open_time", "return_pct"]].merge(btc_ret, on="open_time", how="left")
        gg = gg.sort_values("open_time").reset_index(drop=True)
        if sym == "BTCUSDT":
            gg["beta_pit"] = 1.0
        else:
            y = gg["return_pct"]; x = gg["btc_ret"]
            cov_xy = y.rolling(bar_window, min_periods=1000).cov(x)
            var_x = x.rolling(bar_window, min_periods=1000).var()
            beta = (cov_xy / var_x.replace(0, np.nan)).shift(1)
            gg["beta_pit"] = beta
        gg["symbol"] = sym
        out.append(gg)
    pit = pd.concat(out, ignore_index=True)[["symbol", "open_time", "beta_pit"]]
    print(f"  PIT β done in {time.time()-t0:.0f}s, "
          f"{pit['beta_pit'].notna().sum():,} rows valid", flush=True)
    return pit


def prepare_panel_with_target(panel_path, label):
    print(f"\n=== Preparing panel: {label} ===", flush=True)
    panel = pd.read_parquet(panel_path)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    print(f"  loaded {len(panel):,} rows × {panel['symbol'].nunique()} symbols", flush=True)

    folds_all = _multi_oos_splits(panel)
    pit_beta = compute_pit_beta(panel, BETA_WIN_PIT_DAYS)
    panel = panel.merge(pit_beta, on=["symbol", "open_time"], how="left")
    btc_ret_map = panel[panel["symbol"] == "BTCUSDT"][["open_time", "return_pct"]].rename(
        columns={"return_pct": "btc_ret_t"}).drop_duplicates("open_time")
    panel = panel.merge(btc_ret_map, on="open_time", how="left")
    panel["alpha_beta"] = panel["return_pct"] - panel["beta_pit"] * panel["btc_ret_t"]

    train0, _, _ = _slice(panel, folds_all[0])
    sigma_idio = train0.groupby("symbol")["alpha_beta"].std().to_dict()
    fallback = panel["alpha_beta"].std()
    panel["sigma_idio_ref"] = panel["symbol"].map(sigma_idio).fillna(fallback).clip(lower=1e-6)
    panel["target_beta"] = panel["alpha_beta"] / panel["sigma_idio_ref"]
    print(f"  target stats: p1={panel['target_beta'].quantile(0.01):.2f}, "
          f"p99={panel['target_beta'].quantile(0.99):.2f}, "
          f"|x|>5: {(panel['target_beta'].abs()>5).sum():,}/{len(panel):,}", flush=True)
    return panel, folds_all


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


def train_fold_local(panel, fold, feat_set, eligible_syms):
    train, cal, test = _slice(panel, fold)
    tr = train[(train["autocorr_pctile_7d"] >= THRESHOLD) & (train["symbol"].isin(eligible_syms))]
    ca = cal[(cal["autocorr_pctile_7d"] >= THRESHOLD) & (cal["symbol"].isin(eligible_syms))]
    test_r = test[test["symbol"].isin(eligible_syms)].copy()
    if len(tr) < 1000 or len(ca) < 200 or len(test_r) < 100: return None, None
    Xt = tr[feat_set].to_numpy(np.float32)
    Xc = ca[feat_set].to_numpy(np.float32)
    Xtest = test_r[feat_set].to_numpy(np.float32)
    yt = tr["target_beta"].to_numpy(np.float32)
    yc = ca["target_beta"].to_numpy(np.float32)
    mt = ~np.isnan(yt); mc = ~np.isnan(yc)
    if mt.sum() < 1000 or mc.sum() < 200: return None, None
    preds = []
    for s in SEEDS:
        m = _train(Xt[mt], yt[mt], Xc[mc], yc[mc], seed=s)
        preds.append(m.predict(Xtest, num_iteration=m.best_iteration))
    return test_r, np.mean(preds, axis=0)


def train_and_predict(panel, folds_all, feat_set, label, listings):
    panel_syms = set(panel["symbol"].unique())
    panel_first = panel.groupby("symbol")["open_time"].min()
    for s, t in panel_first.items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t

    def eligibility_at(timestamp):
        if isinstance(timestamp, (int, np.integer)):
            ts = pd.Timestamp(timestamp, unit="ms", tz="UTC")
        else:
            ts = pd.Timestamp(timestamp)
            if ts.tz is None: ts = ts.tz_localize("UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    print(f"\n  Training {label} ({len(feat_set)} features, 10 folds × 5 seeds)...",
          flush=True)
    all_preds = []
    t_start = time.time()
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        eligible = eligibility_at(folds_all[fid]["cal_start"])
        td, p = train_fold_local(panel, folds_all[fid], feat_set, eligible)
        if td is None: continue
        cols = ["symbol", "open_time", "alpha_beta", "return_pct"]
        if "exit_time" in td.columns: cols.append("exit_time")
        df = td[cols].copy()
        df["pred"] = p; df["fold"] = fid
        df = df.rename(columns={"alpha_beta": "alpha_A"})
        if "exit_time" not in df.columns:
            df["exit_time"] = df["open_time"] + pd.Timedelta(minutes=HORIZON * 5)
        all_preds.append(df)
        print(f"    fold {fid}: n={len(td):,} ({time.time()-t0:.0f}s)", flush=True)
    print(f"    total: {time.time()-t_start:.0f}s", flush=True)
    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    return apd


def aggregate_alpha(records, alpha_wide):
    sleeve_queue = deque(maxlen=psl.N_SLEEVES)
    prev_weights = {}
    rows = []
    for _, rec in records.iterrows():
        t = rec["time"]; fold = rec["fold"]
        if rec["traded"]:
            sleeve_queue.append({"entry_time": t, "longs": list(rec["long_basket"]),
                                  "shorts": list(rec["short_basket"])})
        else:
            sleeve_queue.append({"entry_time": t, "longs": [], "shorts": []})
        target_weights = defaultdict(float)
        sleeve_weight = 1.0 / psl.N_SLEEVES
        for sleeve in sleeve_queue:
            n_long = len(sleeve["longs"]); n_short = len(sleeve["shorts"])
            if n_long == 0 or n_short == 0: continue
            for s in sleeve["longs"]:
                target_weights[s] += sleeve_weight * (1.0 / n_long)
            for s in sleeve["shorts"]:
                target_weights[s] -= sleeve_weight * (1.0 / n_short)
        gross = 0.0
        if t in alpha_wide.index:
            alphas = alpha_wide.loc[t]
            for sym, w in prev_weights.items():
                if sym in alphas.index and not pd.isna(alphas[sym]):
                    gross += w * alphas[sym] * 1e4
        all_syms = set(target_weights.keys()) | set(prev_weights.keys())
        abs_delta = sum(abs(target_weights.get(s, 0.0) - prev_weights.get(s, 0.0)) for s in all_syms)
        cost = abs_delta * psl.COST_PER_UNIT_ABS_DELTA
        rows.append({"time": t, "fold": fold,
                      "gross_pnl_bps": gross, "cost_bps": cost,
                      "net_pnl_bps": gross - cost, "turnover": abs_delta,
                      "gross_exposure": sum(abs(w) for w in target_weights.values()),
                      "n_symbols": len(target_weights)})
        prev_weights = dict(target_weights)
    return pd.DataFrame(rows)


def run_v31_beta_hedged(apd, panel_syms, listings):
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    def elig_pit(b):
        if isinstance(b, pd.Timestamp): ts = b
        else: ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=psl.MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}
    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]
    universe = psl.build_rolling_ic_universe(apd, sampled_t, psl.TOP_N, elig_pit)
    records = psl.run_production_protocol_save_sleeves(apd, universe)
    alpha_wide = apd.pivot_table(index="open_time", columns="symbol",
                                   values="alpha_A", aggfunc="first").sort_index()
    df_v = aggregate_alpha(records, alpha_wide)
    return df_v, records, universe, sampled_t


def main():
    print("=== WINNER_17 + β-residual on 51 vs 111 panels (β-hedged execution) ===\n",
          flush=True)
    t_start = time.time()
    listings = get_listings()

    results = {}
    for label, panel_path in [("51-panel", PANEL_51), ("111-panel", PANEL_111)]:
        panel, folds_all = prepare_panel_with_target(panel_path, label)
        panel_syms = set(panel["symbol"].unique())
        for s, t in panel.groupby("symbol")["open_time"].min().items():
            if s not in listings:
                t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
                listings[s] = t

        # Train WINNER_17 on this panel
        feat_set = [f for f in WINNER_17 if f in panel.columns]
        if len(feat_set) < len(WINNER_17):
            missing = set(WINNER_17) - set(feat_set)
            print(f"  WARNING: features missing on {label}: {missing}", flush=True)

        apd = train_and_predict(panel, folds_all, feat_set, label, listings)
        apd.to_parquet(OUT_DIR / f"{label}_predictions.parquet", index=False)

        cyc_ic = apd.dropna(subset=["alpha_A"]).groupby("open_time").apply(
            lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= 5 else np.nan
        ).dropna()
        per_cycle_ic = float(cyc_ic.mean())
        print(f"  per-cycle IC: {per_cycle_ic:+.4f}", flush=True)

        # Run V3.1 β-hedged
        df_v, records, universe, sampled_t = run_v31_beta_hedged(apd, panel_syms, listings)
        net = df_v["net_pnl_bps"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=1000)
        total_d = net.sum() / 1e4 * CAPITAL
        end_eq = CAPITAL + total_d
        print(f"\n  *** {label} V3.1 β-hedged RESULTS ***", flush=True)
        print(f"    Sharpe          : {sh:+.2f} [{lo:+.2f}, {hi:+.2f}]", flush=True)
        print(f"    end-equity $100 : ${end_eq:.2f} ({total_d/CAPITAL*100:+.1f}%)", flush=True)
        print(f"    totPnL          : {net.sum():+.0f} bps", flush=True)
        print(f"    maxDD           : {_max_dd(net):+.0f} bps", flush=True)
        print(f"    gross/cycle     : {df_v['gross_pnl_bps'].mean():+.2f} bps", flush=True)
        print(f"    cost/cycle      : {df_v['cost_bps'].mean():+.2f} bps", flush=True)
        print(f"    folds positive  : {folds_positive(df_v)}/9", flush=True)
        print(f"    traded cycles   : {records['traded'].sum()}/{len(records)}", flush=True)
        df_v.to_csv(OUT_DIR / f"{label}_v31_hedged.csv", index=False)

        # Universe composition per boundary (for 111 panel diagnostic)
        seen = []
        for t in sampled_t:
            u = universe.get(t, set())
            if not seen or u != seen[-1]["u"]:
                seen.append({"t": t, "u": u})

        results[label] = {
            "sharpe": sh, "totPnL_d": total_d, "end_eq": end_eq,
            "per_cycle_ic": per_cycle_ic,
            "gross": df_v["gross_pnl_bps"].mean(),
            "cost": df_v["cost_bps"].mean(),
            "n_traded": int(records["traded"].sum()),
            "n_cycles": len(records),
            "folds_pos": folds_positive(df_v),
            "universe_boundaries": seen,
        }

    # Summary
    print("\n" + "="*100)
    print("  HEAD-TO-HEAD: WINNER_17 + β-residual + β-hedged V3.1")
    print("="*100)
    print(f"  {'panel':<12} {'Sharpe':>10} {'end-eq':>10} {'pnl%':>8} {'IC':>10} {'gross':>10} {'folds+':>7}",
          flush=True)
    for label in ["51-panel", "111-panel"]:
        r = results[label]
        pct = r['totPnL_d'] / CAPITAL * 100
        print(f"  {label:<12} {r['sharpe']:+10.2f} ${r['end_eq']:>8.2f} "
              f"{pct:+7.1f}% {r['per_cycle_ic']:+10.4f} {r['gross']:>+9.2f} "
              f"{r['folds_pos']:>4}/9", flush=True)

    print("\n  Reference baselines:", flush=True)
    print(f"    51-panel WINNER_21 + β-residual β-hedged: Sharpe +0.57, end-eq $121.18, IC +0.0149",
          flush=True)
    print(f"    111-panel WINNER_21 + basket-residual (Phase UNI-111): Sharpe -1.70 (different target)",
          flush=True)

    # Print 111-panel universe to show what's being picked
    print("\n  111-panel rolling-IC universe at each boundary:", flush=True)
    for b in results["111-panel"]["universe_boundaries"]:
        print(f"    {b['t']}: |U|={len(b['u'])} {sorted(b['u'])}", flush=True)

    print(f"\nTotal runtime: {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
