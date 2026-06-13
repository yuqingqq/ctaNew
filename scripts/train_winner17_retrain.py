"""Retrain production stack with WINNER_17: WINNER_21 minus 4 dead-weight features.

Dropped features (each <0.5% LGBM gain in audit):
  - mfi (0.1%)
  - price_volume_corr_20 (0.1%)
  - idio_ret_48b_vs_bk (0.3%)
  - funding_streak_pos (0.4%)

Same target (target_A = basket-residual z-scored), same hyperparameters, same protocol.
WINNER_17 keeps sym_id and all other WINNER_21 features.

After training:
  - Save predictions parquet
  - Run V3.1 un-hedged (production-style, MTM on raw return) → compare to A's +0.65 baseline
  - Run V3.1 β-hedged (MTM on alpha_A) → compare to β-hedged baseline +0.57
"""
from __future__ import annotations
import sys, time, importlib.util, warnings
from pathlib import Path
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

PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
OUT_DIR = REPO / "outputs/vBTC_audit_panel_winner17"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
SEEDS = (42, 1337, 7, 19, 2718)
ALL_FOLDS = list(range(10))
OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
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

# 4 dead-weight features identified in feature audit
DEAD_WEIGHT = {
    "mfi",
    "price_volume_corr_20",
    "idio_ret_48b_vs_bk",
    "funding_streak_pos",
}
WINNER_17 = [f for f in WINNER_21 if f not in DEAD_WEIGHT]

print(f"WINNER_21 has {len(WINNER_21)} features")
print(f"WINNER_17 has {len(WINNER_17)} features (dropped: {sorted(DEAD_WEIGHT)})")


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(psl.CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def folds_positive(df_v):
    return sum(1 for _, g in df_v.groupby("fold") if _sharpe(g["net_pnl_bps"]) > 0)


def train_fold(panel, fold, feat_set, eligible_syms, target_col):
    train, cal, test = _slice(panel, fold)
    tr = train[(train["autocorr_pctile_7d"] >= THRESHOLD) & (train["symbol"].isin(eligible_syms))]
    ca = cal[(cal["autocorr_pctile_7d"] >= THRESHOLD) & (cal["symbol"].isin(eligible_syms))]
    test_r = test[test["symbol"].isin(eligible_syms)].copy()
    if len(tr) < 1000 or len(ca) < 200 or len(test_r) < 100: return None, None
    Xt = tr[feat_set].to_numpy(np.float32)
    Xc = ca[feat_set].to_numpy(np.float32)
    Xtest = test_r[feat_set].to_numpy(np.float32)
    yt = tr[target_col].to_numpy(np.float32)
    yc = ca[target_col].to_numpy(np.float32)
    mt = ~np.isnan(yt); mc = ~np.isnan(yc)
    if mt.sum() < 1000 or mc.sum() < 200: return None, None
    preds = []
    for s in SEEDS:
        m = _train(Xt[mt], yt[mt], Xc[mc], yc[mc], seed=s)
        preds.append(m.predict(Xtest, num_iteration=m.best_iteration))
    return test_r, np.mean(preds, axis=0)


def main():
    print("=== WINNER_17 retrain on production basket-residual target ===\n", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    print(f"Panel: {len(panel):,} rows × {panel['symbol'].nunique()} symbols", flush=True)

    missing = [f for f in WINNER_17 if f not in panel.columns]
    if missing:
        print(f"ERROR: missing features {missing}", flush=True)
        sys.exit(1)
    print(f"WINNER_17 ({len(WINNER_17)} features): {WINNER_17}\n", flush=True)

    folds_all = _multi_oos_splits(panel)
    listings = psl.get_listings()
    panel_first = panel.groupby("symbol")["open_time"].min()
    for s, t in panel_first.items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t
    panel_syms = set(panel["symbol"].unique())

    def eligibility_at(timestamp):
        if isinstance(timestamp, (int, np.integer)):
            ts = pd.Timestamp(timestamp, unit="ms", tz="UTC")
        else:
            ts = pd.Timestamp(timestamp)
            if ts.tz is None: ts = ts.tz_localize("UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    print("--- Train 10 folds × 5 seeds (target = target_A, production basket-residual) ---",
          flush=True)
    all_preds = []
    t_start = time.time()
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        eligible = eligibility_at(folds_all[fid]["cal_start"])
        td, p = train_fold(panel, folds_all[fid], WINNER_17, eligible, "target_A")
        if td is None: continue
        cols = ["symbol", "open_time", "alpha_A", "return_pct"]
        if "exit_time" in td.columns: cols.append("exit_time")
        df = td[cols].copy()
        df["pred"] = p; df["fold"] = fid
        if "exit_time" not in df.columns:
            df["exit_time"] = df["open_time"] + pd.Timedelta(minutes=HORIZON * 5)
        all_preds.append(df)
        print(f"  fold {fid}: n={len(td):,} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  total train: {time.time()-t_start:.0f}s", flush=True)

    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    apd.to_parquet(OUT_DIR / "all_predictions.parquet", index=False)
    print(f"\nSaved: {OUT_DIR / 'all_predictions.parquet'}", flush=True)

    cyc_ic = apd.dropna(subset=["alpha_A"]).groupby("open_time").apply(
        lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= 5 else np.nan
    ).dropna()
    print(f"Per-cycle IC: mean={cyc_ic.mean():+.4f} median={cyc_ic.median():+.4f}", flush=True)
    print(f"  Reference (WINNER_21 production): +0.0235", flush=True)

    # ============================================================
    # Run V3.1 — un-hedged (production execution)
    # ============================================================
    print("\n--- Run V3.1 with IC ranking on WINNER_17 predictions ---", flush=True)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)

    def elig_pit(b):
        if isinstance(b, pd.Timestamp): ts = b
        else: ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]
    universe = psl.build_rolling_ic_universe(apd, sampled_t, psl.TOP_N, elig_pit)
    records = psl.run_production_protocol_save_sleeves(apd, universe)

    # Build fwd_rets for un-hedged MTM
    print("Loading close prices for un-hedged V3.1 MTM...", flush=True)
    t0 = time.time()
    frames = []
    for sym in sorted(panel_syms):
        sd = KLINES_DIR / sym / "5m"
        if not sd.exists(): continue
        files = sorted(sd.glob("*.parquet"))
        dfs = []
        for f in files:
            try: dfs.append(pd.read_parquet(f, columns=["open_time", "close"]))
            except Exception: pass
        if not dfs: continue
        df = pd.concat(dfs, ignore_index=True)
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
        df = df.dropna(subset=["open_time"]).drop_duplicates("open_time").set_index("open_time")
        df = df.rename(columns={"close": sym})
        frames.append(df)
    close_wide = pd.concat(frames, axis=1).sort_index()
    fwd_rets_4h = (close_wide.shift(-psl.HORIZON_ENTRY) - close_wide) / close_wide
    print(f"  close_wide {close_wide.shape} ({time.time()-t0:.0f}s)", flush=True)

    df_v_unhedged = psl.aggregate_sleeves(records, fwd_rets_4h)
    net_u = df_v_unhedged["net_pnl_bps"].to_numpy()
    sh_u, lo_u, hi_u = block_bootstrap_ci(net_u, statistic=_sharpe, block_size=7, n_boot=1000)

    # Also β-hedged version (MTM on alpha_A)
    alpha_wide = apd.pivot_table(index="open_time", columns="symbol",
                                   values="alpha_A", aggfunc="first").sort_index()
    # Copy aggregate_sleeves but MTM on alpha_wide
    from collections import deque, defaultdict
    sleeve_queue = deque(maxlen=psl.N_SLEEVES)
    prev_weights = {}
    rows_a = []
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
        net_v = gross - cost
        rows_a.append({"time": t, "fold": fold, "gross_pnl_bps": gross, "cost_bps": cost,
                        "net_pnl_bps": net_v, "turnover": abs_delta})
        prev_weights = dict(target_weights)
    df_v_hedged = pd.DataFrame(rows_a)
    net_h = df_v_hedged["net_pnl_bps"].to_numpy()
    sh_h, lo_h, hi_h = block_bootstrap_ci(net_h, statistic=_sharpe, block_size=7, n_boot=1000)

    print("\n" + "="*80)
    print(f"  WINNER_17 RESULTS — capital ${CAPITAL:.0f}, 9-month OOS")
    print("="*80)

    print(f"\n  V3.1 UN-HEDGED (production-style execution, MTM on return_pct):")
    print(f"    Sharpe         : {sh_u:+.2f} [{lo_u:+.2f}, {hi_u:+.2f}]", flush=True)
    print(f"    totPnL         : {net_u.sum():+.0f} bps = ${net_u.sum()/1e4*CAPITAL:+.2f}", flush=True)
    print(f"    end-equity     : ${CAPITAL + net_u.sum()/1e4*CAPITAL:.2f}", flush=True)
    print(f"    maxDD          : {_max_dd(net_u):+.0f} bps", flush=True)
    print(f"    gross/cycle    : {df_v_unhedged['gross_pnl_bps'].mean():+.2f} bps", flush=True)
    print(f"    cost/cycle     : {df_v_unhedged['cost_bps'].mean():+.2f} bps", flush=True)
    print(f"    folds positive : {folds_positive(df_v_unhedged)}/9", flush=True)

    print(f"\n  V3.1 β-HEDGED (MTM on alpha_A):")
    print(f"    Sharpe         : {sh_h:+.2f} [{lo_h:+.2f}, {hi_h:+.2f}]", flush=True)
    print(f"    totPnL         : {net_h.sum():+.0f} bps = ${net_h.sum()/1e4*CAPITAL:+.2f}", flush=True)
    print(f"    end-equity     : ${CAPITAL + net_h.sum()/1e4*CAPITAL:.2f}", flush=True)
    print(f"    maxDD          : {_max_dd(net_h):+.0f} bps", flush=True)
    print(f"    folds positive : {folds_positive(df_v_hedged)}/9", flush=True)

    print(f"\n  Reference baselines (WINNER_21):", flush=True)
    print(f"    un-hedged: Sharpe +0.65, end-equity $124.40", flush=True)
    print(f"    β-hedged:  Sharpe +0.57, end-equity $121.18", flush=True)

    delta_u = sh_u - 0.65
    delta_h = sh_h - 0.57
    print(f"\n  Δ vs WINNER_21:")
    print(f"    un-hedged: {delta_u:+.2f} Sharpe", flush=True)
    print(f"    β-hedged:  {delta_h:+.2f} Sharpe", flush=True)

    df_v_unhedged.to_csv(OUT_DIR / "v31_unhedged.csv", index=False)
    df_v_hedged.to_csv(OUT_DIR / "v31_hedged.csv", index=False)


if __name__ == "__main__":
    main()
