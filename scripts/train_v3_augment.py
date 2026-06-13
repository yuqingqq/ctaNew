"""WINNER_17 + v3 unique add-ons (8 features) — test if v3 features lift the
WINNER_17 +0.74 baseline when added as augmentation rather than replacement.

Diagnosed (2026-05-13): pure v3 (24 features, all 1d-granularity forward-filled)
failed catastrophically because LGBM has no within-day variation to learn from.
The 5m-bar features in WINNER_17 (return_1d, atr_pct, obv_z_1d, idio_vol_to_btc_1h, etc.)
match target time-resolution. v3 features should augment, not replace.

Top-8 v3 add-ons selected by:
1. Strong individual |IC| in Phase 2
2. NOT semantically duplicated in WINNER_17
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

PANEL = REPO / "outputs/vBTC_features_btc_v3/panel_v3.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
OUT_DIR = REPO / "outputs/vBTC_audit_panel_v3_augment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RC = 0.50
THRESHOLD = 1 - RC
SEEDS = (42, 1337, 7, 19, 2718)
ALL_FOLDS = list(range(10))
OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
CAPITAL = 100.0

# Reproduce WINNER_17
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

# Top-8 v3 add-ons (selected for strong IC + non-duplication with WINNER_17)
V3_AUGMENT_8 = [
    "idio_max_abs_12b",          # IC -0.040  (G_process_fp)
    "resid_vol_30d",             # IC -0.040  (C_resid_behavior, 30d window not in W17)
    "resid_vol_90d",             # IC -0.038  (C, 90d window not in W17)
    "corr_btc_30d",              # IC +0.027  (B, 30d window vs W17's 1d)
    "beta_btc_90d",              # IC -0.024  (B, level vs W17's Δ-5d)
    "amihud_illiq_30d",          # IC -0.018  (A, new illiquidity)
    "dist_from_365d_high",       # IC +0.014  (D, anchoring)
    "multi_horizon_trend_score", # IC -0.014  (D, composite trend)
]

WINNER_17_PLUS_V3 = WINNER_17 + V3_AUGMENT_8


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


def train_fold_local(panel, fold, feat_set, eligible_syms):
    train, cal, test = _slice(panel, fold)
    tr = train[(train["autocorr_pctile_7d"] >= THRESHOLD) & (train["symbol"].isin(eligible_syms))]
    ca = cal[(cal["autocorr_pctile_7d"] >= THRESHOLD) & (cal["symbol"].isin(eligible_syms))]
    test_r = test[test["symbol"].isin(eligible_syms)].copy()
    if len(tr) < 1000 or len(ca) < 200 or len(test_r) < 100: return None, None
    Xt = tr[feat_set].to_numpy(np.float32)
    Xc = ca[feat_set].to_numpy(np.float32)
    Xtest = test_r[feat_set].to_numpy(np.float32)
    yt = tr["target_beta_btc"].to_numpy(np.float32)
    yc = ca["target_beta_btc"].to_numpy(np.float32)
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
            df["exit_time"] = df["open_time"] + pd.Timedelta(minutes=48 * 5)
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
    print("=== WINNER_17 + v3 augment family ablation (β-hedged V3.1) ===\n", flush=True)
    t_start = time.time()
    listings = get_listings()
    panel = pd.read_parquet(PANEL)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    print(f"Panel: {len(panel):,} rows × {panel['symbol'].nunique()} symbols", flush=True)
    panel_syms = set(panel["symbol"].unique())
    for s, t in panel.groupby("symbol")["open_time"].min().items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t

    folds_all = _multi_oos_splits(panel)

    variants = [
        ("WINNER_17_baseline_reproduce", WINNER_17),
        ("WINNER_17_plus_v3_aug8",       WINNER_17_PLUS_V3),
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
    print(f"  HEAD-TO-HEAD COMPARISON (β-hedged V3.1)", flush=True)
    print("="*110, flush=True)
    print(f"  {'variant':<32} {'#feats':>7} {'Sharpe':>10} {'end-eq':>10} {'IC':>10} "
          f"{'gross':>7} {'cost':>7} {'fold+':>6}", flush=True)
    for r in results:
        print(f"  {r['label']:<32} {r['n_feats']:>7} {r['sharpe']:+10.2f} "
              f"${r['end_eq']:>8.2f} {r['per_cycle_ic']:+10.4f} "
              f"{r['gross_cycle']:>+7.2f} {r['cost_cycle']:>7.2f} "
              f"{r['folds_pos']:>3}/9", flush=True)

    print(f"\n  Reference: WINNER_17 + β-residual β-hedged (prior run): "
          f"Sharpe +0.74, end-eq $126.96, IC +0.0157", flush=True)
    if len(results) == 2:
        base_sh = results[0]["sharpe"]
        aug_sh = results[1]["sharpe"]
        delta = aug_sh - base_sh
        print(f"\n  v3 augment Δ Sharpe vs WINNER_17 (this-run base): {delta:+.2f}", flush=True)
        if delta >= 0.20 and results[1]["gross_cycle"] > results[0]["gross_cycle"]:
            print(f"  ADOPTION GATE: PASSED ✓ (Δ Sharpe ≥ +0.20 AND gross/cycle improves)", flush=True)
        elif delta >= 0.10:
            print(f"  PARTIAL: Δ Sharpe ≥ +0.10 — check stability via Phase 8 drop-5", flush=True)
        else:
            print(f"  ADOPTION GATE: NOT MET — v3 augment does not lift WINNER_17", flush=True)

    pd.DataFrame(results).to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"\nResults saved to {OUT_DIR}/", flush=True)
    print(f"Total runtime: {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
