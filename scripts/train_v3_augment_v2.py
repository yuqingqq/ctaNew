"""v2: apples-to-apples comparison vs WINNER_17 +0.74 baseline.

Use the ORIGINAL panel (panel_variants_with_funding.parquet) as input — the
same one diag_winner17 used. Compute alpha_beta/target_beta_btc at fit time
using the diag_winner17 method (90d × 288 bar β on 4h forward returns).
Merge v3 features from panel_v3.parquet via (symbol, open_time).

Variants tested:
  V0  WINNER_17 (17)                  — reproduce +0.74 reference
  V1  WINNER_17 + v3 augment 8 (25)   — v3 features add value?
  V2  WINNER_17 + v3 augment 4 (21)   — smaller addition to reduce dilution
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

PANEL_BASE = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
PANEL_V3   = REPO / "outputs/vBTC_features_btc_v3/panel_v3_5m.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
OUT_DIR    = REPO / "outputs/vBTC_audit_panel_v3_augment_5m"
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

# WINNER_17
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
WINNER_21 = ([f for f in XS_FEATURE_COLS_V6_CLEAN if f not in ALL_DROPS]
             + FUNDING_LEAN + ADD_CROSS_BTC + ADD_MORE_FUNDING)
DEAD_WEIGHT = {"mfi", "price_volume_corr_20", "idio_ret_48b_vs_bk", "funding_streak_pos"}
WINNER_17 = [f for f in WINNER_21 if f not in DEAD_WEIGHT]

# Full 19-feature v3_5m set (post-dedupe vs WINNER_17)
V3_FULL_19 = [
    # A_liquidity (3)
    "log_dollar_volume_7d", "volume_stability_30d", "amihud_illiq_30d",
    # B_btc (4)
    "beta_btc_30d", "beta_btc_90d", "corr_btc_30d", "corr_breakdown",
    # C_resid (6)
    "resid_vol_30d", "resid_vol_90d",
    "resid_skew_30d", "resid_kurt_30d",
    "resid_jump_count_30d", "resid_trend_score_30d",
    # D_trend (3)
    "dist_from_30d_high", "dist_from_365d_high", "multi_horizon_trend_score",
    # E_funding (1)
    "funding_mean_30d",
    # G_process (2 — passthrough already in base panel)
    "idio_skew_1d", "idio_max_abs_12b",
]

# Smaller variants for sweep
V3_TOP_8 = [
    # Picked by Phase 2 individual IC magnitude, dedupe-filtered
    "idio_max_abs_12b",       # G  -0.040
    "resid_vol_30d",          # C  -0.040 (5m bar window now)
    "corr_btc_30d",           # B  +0.027
    "beta_btc_90d",           # B  -0.024
    "amihud_illiq_30d",       # A  -0.018
    "dist_from_365d_high",    # D  +0.014
    "multi_horizon_trend_score",  # D  -0.014
    "resid_skew_30d",         # C  distributional
]
V3_TOP_4 = [
    "idio_max_abs_12b", "resid_vol_30d",
    "amihud_illiq_30d", "dist_from_365d_high",
]


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


def compute_pit_beta(panel, beta_win_days):
    """diag_winner17 method: rolling β = cov / var over beta_win × 288 bars."""
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
    return pd.concat(out, ignore_index=True)[["symbol", "open_time", "beta_pit"]]


def prepare_panel(panel_base, panel_v3_feats):
    print("Loading base panel...", flush=True)
    panel = pd.read_parquet(panel_base)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    print(f"  base: {len(panel):,} rows × {panel['symbol'].nunique()} syms × {panel.shape[1]} cols",
          flush=True)
    folds_all = _multi_oos_splits(panel)

    # Merge v3 features (skip features already in base panel to avoid _x/_y suffix collision)
    new_v3_feats = [f for f in panel_v3_feats if f not in panel.columns]
    skipped = [f for f in panel_v3_feats if f in panel.columns]
    if skipped:
        print(f"  skip (already in base): {skipped}", flush=True)
    print(f"Merging v3 features ({len(new_v3_feats)} new / {len(panel_v3_feats)} requested)...",
          flush=True)
    p3 = pd.read_parquet(PANEL_V3, columns=["symbol", "open_time"] + new_v3_feats)
    p3["open_time"] = pd.to_datetime(p3["open_time"], utc=True)
    panel = panel.merge(p3, on=["symbol", "open_time"], how="left")
    print(f"  merged: {len(panel):,} rows × {panel.shape[1]} cols", flush=True)

    # β-residual computation (diag_winner17 method)
    print(f"  Computing PIT β with {BETA_WIN_PIT_DAYS}d window...", flush=True)
    t0 = time.time()
    pit_beta = compute_pit_beta(panel, BETA_WIN_PIT_DAYS)
    print(f"    done in {time.time()-t0:.0f}s, "
          f"{pit_beta['beta_pit'].notna().sum():,} rows valid", flush=True)
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


def run_one(label, feat_set, panel, folds_all, listings, panel_syms):
    apd = train_and_predict(panel, folds_all, feat_set, label, listings)
    apd.to_parquet(OUT_DIR / f"{label}_predictions.parquet", index=False)
    cyc_ic = apd.dropna(subset=["alpha_A"]).groupby("open_time").apply(
        lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= 5 else np.nan
    ).dropna()
    per_cycle_ic = float(cyc_ic.mean())
    df_v, records, universe, sampled_t = run_v31_beta_hedged(apd, panel_syms, listings)
    net = df_v["net_pnl_bps"].to_numpy()
    sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=1000)
    total_d = net.sum() / 1e4 * CAPITAL
    end_eq = CAPITAL + total_d
    df_v.to_csv(OUT_DIR / f"{label}_v31_hedged.csv", index=False)
    return {
        "label": label, "n_feats": len(feat_set),
        "sharpe": sh, "sh_lo": lo, "sh_hi": hi,
        "end_eq": end_eq, "totPnL": float(net.sum()),
        "maxDD": _max_dd(net),
        "gross_cycle": float(df_v["gross_pnl_bps"].mean()),
        "cost_cycle": float(df_v["cost_bps"].mean()),
        "per_cycle_ic": per_cycle_ic,
        "folds_pos": folds_positive(df_v),
        "n_traded": int(records["traded"].sum()),
        "n_cycles": len(records),
    }


def main():
    print("=== v3 augment v2 (apples-to-apples vs WINNER_17 +0.74 baseline) ===\n", flush=True)
    t_start = time.time()
    listings = get_listings()

    # Build merged panel with all v3 features needed (union of all variants)
    all_v3_feats = sorted(set(V3_FULL_19 + V3_TOP_8 + V3_TOP_4))
    panel, folds_all = prepare_panel(PANEL_BASE, all_v3_feats)
    panel_syms = set(panel["symbol"].unique())
    for s, t in panel.groupby("symbol")["open_time"].min().items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t

    variants = [
        ("V0_WINNER_17",            WINNER_17),
        ("V1_W17_plus_v3_full19",   WINNER_17 + V3_FULL_19),
        ("V2_W17_plus_v3_top8",     WINNER_17 + V3_TOP_8),
        ("V3_W17_plus_v3_top4",     WINNER_17 + V3_TOP_4),
    ]

    results = []
    for label, feat_set in variants:
        missing = [f for f in feat_set if f not in panel.columns]
        if missing:
            print(f"\n!! {label}: missing features: {missing}", flush=True)
            continue
        r = run_one(label, feat_set, panel, folds_all, listings, panel_syms)
        results.append(r)
        print(f"\n  >>> {label}: Sharpe={r['sharpe']:+.2f} [{r['sh_lo']:+.2f},{r['sh_hi']:+.2f}], "
              f"end-eq=${r['end_eq']:.2f}, IC={r['per_cycle_ic']:+.4f}, "
              f"folds+={r['folds_pos']}/9", flush=True)

    print("\n" + "="*110, flush=True)
    print(f"  HEAD-TO-HEAD COMPARISON (apples-to-apples, β-hedged V3.1)", flush=True)
    print("="*110, flush=True)
    print(f"  {'variant':<32} {'#feats':>7} {'Sharpe':>10} {'CI':>20} {'end-eq':>10} "
          f"{'IC':>10} {'gross':>7} {'fold+':>6}", flush=True)
    for r in results:
        ci = f"[{r['sh_lo']:+.2f},{r['sh_hi']:+.2f}]"
        print(f"  {r['label']:<32} {r['n_feats']:>7} {r['sharpe']:+10.2f} "
              f"{ci:>20} ${r['end_eq']:>8.2f} {r['per_cycle_ic']:+10.4f} "
              f"{r['gross_cycle']:>+7.2f} {r['folds_pos']:>3}/9", flush=True)

    print(f"\n  Reference: WINNER_17 + β-residual β-hedged (docs/v6 source): "
          f"Sharpe +0.74, end-eq $126.96, IC +0.0157", flush=True)
    if len(results) >= 2:
        base = results[0]
        for r in results[1:]:
            delta = r["sharpe"] - base["sharpe"]
            print(f"\n  {r['label']} Δ Sharpe vs V0 baseline: {delta:+.2f}", flush=True)
            if delta >= 0.20 and r["gross_cycle"] > base["gross_cycle"]:
                print(f"    ADOPTION GATE: PASSED ✓", flush=True)
            elif delta >= 0.10:
                print(f"    PARTIAL: Δ ≥ +0.10 — check stability", flush=True)
            else:
                print(f"    NOT MET", flush=True)

    pd.DataFrame(results).to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"\nResults saved to {OUT_DIR}/", flush=True)
    print(f"Total runtime: {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
