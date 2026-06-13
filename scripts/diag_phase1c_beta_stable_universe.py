"""Phase 1C: BTC-residual target on a beta-stability-selected universe.

Mechanism the user proposed: pick symbols whose relationship to BTC is stable,
so the residual `return - β·BTC_return` is clean alpha, not β-estimation noise.

Pipeline:
  1. For each of the 51 candidate symbols, compute at first-OOS-time:
       - β(s) = OLS slope of 4h returns vs BTC over trailing 365d
       - R²(s) = trailing regression R²
       - σ_β(s) = std of 12 rolling 30d β values over the past year
       - stability_score(s) = |β(s)| × R²(s) / (1 + σ_β(s))
  2. Universe = top-N by stability_score (N=25).
  3. PIT β: for every row, recompute β using trailing 90d 4h-return regression vs BTC.
  4. target_β = (return_pct - β_pit × BTC_return_pct) / σ_idio_per_sym
     where σ_idio is computed from first-fold training-window residuals (PIT, locked).
  5. Retrain WINNER_21 (10 folds × 5 seeds) on this curated universe + target.
  6. Build all_predictions with alpha_A = (return - β_pit × BTC_return) for V3.1's IC ranker.
  7. Run V3.1 with rolling-IC + K=3 + conv_gate + PM_M2 + sleeve.

Comparison points:
  - Production (51-basket): Sharpe +2.23
  - Phase 1B (BTC+ETH residual, full 51): Sharpe +0.40
  - Phase 1A (BTC-only residual, full 51): Sharpe +0.27

If 1C beats Phase 1B materially, the β-stable universe + clean BTC-residual is
extracting universe-portable alpha that the full-panel Phase 1 couldn't.
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
OUT = REPO / "outputs/vBTC_phase1c_beta_stable"
OUT.mkdir(parents=True, exist_ok=True)

HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
SEEDS = (42, 1337, 7, 19, 2718)
ALL_FOLDS = list(range(10))
OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60

N_UNIVERSE = 25
BETA_WIN_LONG_DAYS = 365       # for the stability score (locked once at first-OOS time)
BETA_WIN_PIT_DAYS = 90         # for PIT β at every row used in target
BETA_ROLLING_DAYS = 30         # for σ_β computation
BETA_N_ROLLING = 12            # number of rolling windows used to compute σ_β

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


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(psl.CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def folds_positive(df_v):
    return sum(1 for _, g in df_v.groupby("fold") if _sharpe(g["net_pnl_bps"]) > 0)


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


def compute_stability_scores(panel, t_anchor):
    """At t_anchor (first OOS time), compute per-symbol β-stability score using ONLY
    data strictly before t_anchor. Returns DataFrame with columns:
       symbol, beta, r2, sigma_beta, score
    """
    print(f"  Computing β-stability scores at anchor={t_anchor}", flush=True)
    panel_anch = panel[panel["open_time"] < t_anchor]
    # Pivot to symbol x time of returns
    btc_ret = panel_anch[panel_anch["symbol"] == "BTCUSDT"][["open_time", "return_pct"]].rename(
        columns={"return_pct": "btc_ret"}).set_index("open_time")
    rows = []
    long_window_start = t_anchor - pd.Timedelta(days=BETA_WIN_LONG_DAYS)
    for sym, g in panel_anch.groupby("symbol"):
        if sym == "BTCUSDT": continue
        gg = g[["open_time", "return_pct"]].set_index("open_time")
        gg = gg.join(btc_ret, how="inner")
        gg = gg.dropna()
        gg = gg[gg.index >= long_window_start]
        if len(gg) < 1000:
            continue
        x = gg["btc_ret"].to_numpy(); y = gg["return_pct"].to_numpy()
        # Full-window OLS β
        if np.var(x) == 0: continue
        beta_full = float(np.cov(y, x)[0,1] / np.var(x))
        y_pred = beta_full * x
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
        # Rolling 30d β: 12 windows
        bar_per_30d = 30 * 288  # 5m bars per 30 days
        n = len(gg)
        betas = []
        for i in range(BETA_N_ROLLING):
            end_idx = n - i * bar_per_30d
            start_idx = end_idx - bar_per_30d
            if start_idx < 0: break
            xw = x[start_idx:end_idx]; yw = y[start_idx:end_idx]
            if np.var(xw) == 0: continue
            b = float(np.cov(yw, xw)[0,1] / np.var(xw))
            betas.append(b)
        if len(betas) < 4:
            sigma_b = np.nan
        else:
            sigma_b = float(np.std(betas))
        if np.isnan(sigma_b):
            score = np.nan
        else:
            score = abs(beta_full) * r2 / (1 + sigma_b)
        rows.append({"symbol": sym, "beta": beta_full, "r2": r2,
                     "sigma_beta": sigma_b, "score": score})
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def compute_pit_beta(panel, beta_win_days):
    """For every row, compute trailing β(s, t) of (return_pct vs BTC return_pct) over
    [t - beta_win_days, t). Uses 5m bars throughout.

    Implementation: per symbol s, align with BTC by open_time, then rolling OLS β.
    """
    print(f"  Computing PIT β with {beta_win_days}d window...", flush=True)
    t0 = time.time()
    btc_ret = panel[panel["symbol"] == "BTCUSDT"][["open_time", "return_pct"]].rename(
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
            beta = (cov_xy / var_x.replace(0, np.nan)).shift(1)  # shift to make PIT (t-1 ↓ t)
            gg["beta_pit"] = beta
        gg["symbol"] = sym
        out.append(gg)
    pit = pd.concat(out, ignore_index=True)[["symbol", "open_time", "beta_pit"]]
    print(f"  PIT β done in {time.time()-t0:.0f}s, {pit['beta_pit'].notna().sum():,} rows have valid β",
          flush=True)
    return pit


def train_fold(panel, fold, feat_set, eligible_syms):
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


def main():
    print("=== Phase 1C: β-stable universe + PIT BTC-residual target ===\n", flush=True)
    t_start = time.time()

    panel = pd.read_parquet(PANEL_PATH)
    print(f"Panel: {len(panel):,} rows × {panel['symbol'].nunique()} symbols", flush=True)

    folds_all = _multi_oos_splits(panel)
    first_oos_time = folds_all[1]["test_start"]  # fold 1 is first OOS
    print(f"First OOS time (anchor): {first_oos_time}", flush=True)

    # 1. Stability scores
    scores = compute_stability_scores(panel, first_oos_time)
    print("\nTop-25 by β-stability score:", flush=True)
    print(scores.head(25)[["rank","symbol","beta","r2","sigma_beta","score"]].to_string(index=False),
          flush=True)
    print("\nBottom-10 by β-stability score:", flush=True)
    print(scores.tail(10)[["rank","symbol","beta","r2","sigma_beta","score"]].to_string(index=False),
          flush=True)

    universe = sorted(scores.head(N_UNIVERSE)["symbol"].tolist())
    print(f"\nSelected universe ({len(universe)} symbols): {universe}", flush=True)
    scores.to_csv(OUT / "stability_scores.csv", index=False)

    # 2. PIT β
    pit_beta = compute_pit_beta(panel, BETA_WIN_PIT_DAYS)
    panel = panel.merge(pit_beta, on=["symbol", "open_time"], how="left")
    # Get BTC return per-time
    btc_ret_map = panel[panel["symbol"] == "BTCUSDT"][["open_time", "return_pct"]].rename(
        columns={"return_pct": "btc_ret_t"}).drop_duplicates("open_time")
    panel = panel.merge(btc_ret_map, on="open_time", how="left")
    panel["alpha_beta"] = panel["return_pct"] - panel["beta_pit"] * panel["btc_ret_t"]
    print(f"\nalpha_beta stats: mean={panel['alpha_beta'].mean():+.6f} "
          f"std={panel['alpha_beta'].std():.6f}", flush=True)

    # 3. Per-symbol σ_idio from first-fold training residuals (PIT, locked)
    print("\nComputing per-symbol σ_idio from first-fold training residuals (PIT, locked)...",
          flush=True)
    train_fold0, _, _ = _slice(panel, folds_all[0])
    sigma_idio_per_sym = train_fold0.groupby("symbol")["alpha_beta"].std().to_dict()
    # Fallback: panel-wide std
    fallback = panel["alpha_beta"].std()
    panel["sigma_idio_ref"] = panel["symbol"].map(sigma_idio_per_sym).fillna(fallback)
    panel["sigma_idio_ref"] = panel["sigma_idio_ref"].clip(lower=1e-6)
    panel["target_beta"] = panel["alpha_beta"] / panel["sigma_idio_ref"]
    print(f"Target stats: p1={panel['target_beta'].quantile(0.01):.2f} "
          f"p99={panel['target_beta'].quantile(0.99):.2f} "
          f"min={panel['target_beta'].min():.2f} max={panel['target_beta'].max():.2f}",
          flush=True)
    print(f"|target_beta| > 5: "
          f"{(panel['target_beta'].abs() > 5).sum():,} rows "
          f"({(panel['target_beta'].abs()>5).mean()*100:.2f}%)", flush=True)

    feat_set = [f for f in WINNER_21 if f in panel.columns]
    print(f"\nFeature set: {len(feat_set)} features", flush=True)

    listings = get_listings()
    panel_first = panel.groupby("symbol")["open_time"].min()
    for s, t in panel_first.items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t
    # Eligibility is constrained to the chosen β-stable universe
    universe_set = set(universe)

    def eligibility_at(timestamp):
        if isinstance(timestamp, (int, np.integer)):
            ts = pd.Timestamp(timestamp, unit="ms", tz="UTC")
        else:
            ts = pd.Timestamp(timestamp)
            if ts.tz is None: ts = ts.tz_localize("UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in universe_set if listings.get(s) and listings[s] <= cutoff}

    # 4. Train
    print(f"\n--- Train 10 folds × 5 seeds (target = β-residual, universe={N_UNIVERSE}) ---",
          flush=True)
    all_preds = []
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        eligible = eligibility_at(folds_all[fid]["cal_start"])
        td, p = train_fold(panel, folds_all[fid], feat_set, eligible)
        if td is None: continue
        cols = ["symbol", "open_time", "alpha_beta", "return_pct"]
        if "exit_time" in td.columns: cols.append("exit_time")
        df = td[cols].copy()
        df["pred"] = p; df["fold"] = fid
        df = df.rename(columns={"alpha_beta": "alpha_A"})  # for V3.1 IC ranker
        if "exit_time" not in df.columns:
            df["exit_time"] = df["open_time"] + pd.Timedelta(minutes=HORIZON * 5)
        all_preds.append(df)
        print(f"  fold {fid}: n={len(td):,} ({time.time()-t0:.0f}s)", flush=True)
    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    apd.to_parquet(OUT / "all_predictions.parquet", index=False)
    print(f"  Saved {len(apd):,} prediction rows", flush=True)

    cyc_ic = apd.dropna(subset=["alpha_A"]).groupby("open_time").apply(
        lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= 5 else np.nan
    ).dropna()
    print(f"  Per-cycle IC (vs β-residual): mean={cyc_ic.mean():+.4f} "
          f"median={cyc_ic.median():+.4f}", flush=True)

    # 5. Run V3.1
    print("\n--- Run V3.1 with IC ranking on Phase 1C predictions ---", flush=True)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]
    universe_map = psl.build_rolling_ic_universe(apd, sampled_t, psl.TOP_N, eligibility_at)
    records = psl.run_production_protocol_save_sleeves(apd, universe_map)

    print("  loading close prices...", flush=True)
    t0 = time.time()
    frames = []
    for sym in universe:
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

    df_v = psl.aggregate_sleeves(records, fwd_rets_4h)
    net = df_v["net_pnl_bps"].to_numpy()
    sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=1000)
    df_v.to_csv(OUT / "v31_phase1c.csv", index=False)

    print("\n" + "=" * 80)
    print("  V3.1 RESULTS — Phase 1C (β-stable universe + PIT BTC-residual target)")
    print("=" * 80)
    print(f"  universe size     : {N_UNIVERSE}", flush=True)
    print(f"  per-cycle IC      : {cyc_ic.mean():+.4f}", flush=True)
    print(f"  Sharpe            : {sh:+.2f} [{lo:+.2f}, {hi:+.2f}]", flush=True)
    print(f"  totPnL            : {net.sum():+.0f} bps", flush=True)
    print(f"  maxDD             : {_max_dd(net):+.0f} bps", flush=True)
    print(f"  gross/cycle       : {df_v['gross_pnl_bps'].mean():+.2f} bps", flush=True)
    print(f"  cost/cycle        : {df_v['cost_bps'].mean():+.2f} bps", flush=True)
    print(f"  net/cycle         : {df_v['net_pnl_bps'].mean():+.2f} bps", flush=True)
    print(f"  turnover/cycle    : {df_v['turnover'].mean():.3f}", flush=True)
    print(f"  folds positive    : {folds_positive(df_v)}/9", flush=True)
    print(f"  cycles traded     : {records['traded'].sum()}/{len(records)}", flush=True)

    print(f"\n  Reference points:", flush=True)
    print(f"    Production (51-basket, full 51):       Sharpe +2.23 IC +0.023", flush=True)
    print(f"    Phase 1A (BTC residual, full 51):      Sharpe +0.27 IC +0.006", flush=True)
    print(f"    Phase 1B (BTC+ETH residual, full 51):  Sharpe +0.40 IC +0.011", flush=True)

    print("\n=== PER-FOLD Sharpe ===", flush=True)
    for fid in OOS_FOLDS:
        g = df_v[df_v["fold"] == fid]["net_pnl_bps"].to_numpy()
        print(f"  fold {fid}: {_sharpe(g):+.2f}", flush=True)

    # Universe sets per boundary
    print("\n=== Universe composition at each boundary ===", flush=True)
    seen = []
    for t in sampled_t:
        u = universe_map.get(t, set())
        if not seen or u != seen[-1]["u"]:
            seen.append({"t": t, "u": u})
    for b in seen:
        print(f"  {b['t']}: |U|={len(b['u'])} {sorted(b['u'])}", flush=True)

    print(f"\nTotal Phase 1C runtime: {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
