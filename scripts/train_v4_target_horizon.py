"""Train V4 = WINNER_17 + 7 target-horizon features from base panel.
Full 10-fold × 5-seed training, V3.1 β-hedged, LOFO per-fold breakdown.

If V4 lifts Sharpe with a MORE DISTRIBUTED per-fold pattern than V1's
fold-4-dependent +1.66, then target-horizon features carry real signal.
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
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
OUT_DIR    = REPO / "outputs/vBTC_audit_panel_v4_target"
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

# Target-horizon features from base panel (unused by WINNER_17)
V4_TARGET_HORIZON_7 = [
    "idio_ret_to_btc_48b",     # 4h backward β-residual — matches target horizon
    "idio_ret_to_btc_12b",     # 1h backward residual
    "btc_ret_48b",             # 4h BTC momentum
    "xs_alpha_dispersion_48b", # 4h cross-sectional dispersion
    "xs_alpha_mean_48b",       # 4h cross-sectional mean alpha
    "idio_vol_to_btc_1d",      # 1d idio vol
    "xs_alpha_iqr_12b",        # 1h IQR
]

# Subset by importance from fold-4 diagnostic — top-4 only (more aggressive)
V4_TOP_4 = [
    "xs_alpha_dispersion_48b", # 10.9% gain (#1 in V4)
    "xs_alpha_mean_48b",       # 8.0%
    "btc_ret_48b",             # 6.1%
    "xs_alpha_iqr_12b",        # 5.7%
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
            cov = y.rolling(bar_window, min_periods=1000).cov(x)
            var = x.rolling(bar_window, min_periods=1000).var()
            gg["beta_pit"] = (cov / var.replace(0, np.nan)).shift(1)
        gg["symbol"] = sym
        out.append(gg)
    return pd.concat(out, ignore_index=True)[["symbol", "open_time", "beta_pit"]]


def prepare_panel():
    print("Loading base panel...", flush=True)
    panel = pd.read_parquet(PANEL_BASE)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    folds_all = _multi_oos_splits(panel)
    # All V4 features already in base panel — no merge needed
    missing = [f for f in V4_TARGET_HORIZON_7 if f not in panel.columns]
    if missing:
        print(f"  WARN: V4 features MISSING from base: {missing}", flush=True)
        sys.exit(1)
    print(f"  base: {len(panel):,} rows × {panel.shape[1]} cols", flush=True)
    print(f"  All 7 V4 features present in base panel ✓", flush=True)

    t0 = time.time()
    print(f"  Computing PIT β with {BETA_WIN_PIT_DAYS}d window...", flush=True)
    pit_beta = compute_pit_beta(panel, BETA_WIN_PIT_DAYS)
    panel = panel.merge(pit_beta, on=["symbol","open_time"], how="left")
    btc_ret_map = panel[panel["symbol"]=="BTCUSDT"][["open_time","return_pct"]].rename(
        columns={"return_pct":"btc_ret_t"}).drop_duplicates("open_time")
    panel = panel.merge(btc_ret_map, on="open_time", how="left")
    panel["alpha_beta"] = panel["return_pct"] - panel["beta_pit"] * panel["btc_ret_t"]

    train0, _, _ = _slice(panel, folds_all[0])
    sigma_idio = train0.groupby("symbol")["alpha_beta"].std().to_dict()
    fallback = panel["alpha_beta"].std()
    panel["sigma_idio_ref"] = panel["symbol"].map(sigma_idio).fillna(fallback).clip(lower=1e-6)
    panel["target_beta"] = panel["alpha_beta"] / panel["sigma_idio_ref"]
    print(f"  done in {time.time()-t0:.0f}s", flush=True)
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

    print(f"\n  Training {label} ({len(feat_set)} features, 10 folds × 5 seeds)...", flush=True)
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
    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time","symbol"])
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

    # Per-fold breakdown
    per_fold = []
    for fold in range(1, 10):
        g = df_v[df_v["fold"]==fold]
        per_fold.append({
            "fold": fold,
            "sharpe": _sharpe(g["net_pnl_bps"].to_numpy()),
            "totPnL": float(g["net_pnl_bps"].sum()),
        })

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
        "per_fold": per_fold,
        "df_v": df_v,
    }


def main():
    print("=== V4: WINNER_17 + target-horizon features (4h-1d) ===\n", flush=True)
    t_start = time.time()
    listings = get_listings()
    panel, folds_all = prepare_panel()
    panel_syms = set(panel["symbol"].unique())
    for s, t in panel.groupby("symbol")["open_time"].min().items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t

    variants = [
        ("V0_WINNER_17",          WINNER_17),
        ("V4_W17_plus_target7",   WINNER_17 + V4_TARGET_HORIZON_7),
        ("V4_W17_plus_target4",   WINNER_17 + V4_TOP_4),
    ]

    results = []
    for label, feat_set in variants:
        missing = [f for f in feat_set if f not in panel.columns]
        if missing:
            print(f"\n!! {label}: missing {missing}", flush=True)
            continue
        r = run_one(label, feat_set, panel, folds_all, listings, panel_syms)
        results.append(r)
        print(f"\n  >>> {label}: Sharpe={r['sharpe']:+.2f} [{r['sh_lo']:+.2f},{r['sh_hi']:+.2f}], "
              f"end-eq=${r['end_eq']:.2f}, IC={r['per_cycle_ic']:+.4f}, "
              f"folds+={r['folds_pos']}/9", flush=True)

    print("\n" + "="*110, flush=True)
    print(f"  HEAD-TO-HEAD: WINNER_17 vs target-horizon augments", flush=True)
    print("="*110, flush=True)
    print(f"  {'variant':<32} {'#feats':>7} {'Sharpe':>10} {'CI':>20} {'end-eq':>10} "
          f"{'IC':>10} {'gross':>7} {'fold+':>6}", flush=True)
    for r in results:
        ci = f"[{r['sh_lo']:+.2f},{r['sh_hi']:+.2f}]"
        print(f"  {r['label']:<32} {r['n_feats']:>7} {r['sharpe']:+10.2f} "
              f"{ci:>20} ${r['end_eq']:>8.2f} {r['per_cycle_ic']:+10.4f} "
              f"{r['gross_cycle']:>+7.2f} {r['folds_pos']:>3}/9", flush=True)

    # Per-fold breakdown
    print(f"\n  Per-fold Sharpe:", flush=True)
    print(f"  {'fold':>4}", end="", flush=True)
    for r in results:
        print(f" {r['label']:<28}", end="", flush=True)
    print("", flush=True)
    for fold in range(1, 10):
        print(f"  {fold:>4}", end="", flush=True)
        for r in results:
            pf = next(p for p in r["per_fold"] if p["fold"]==fold)
            print(f" {pf['sharpe']:+7.2f} ({pf['totPnL']:+6.0f} bps)   ", end="", flush=True)
        print("", flush=True)

    # LOFO test if V4 lifts
    if len(results) >= 2:
        v0 = results[0]
        for r in results[1:]:
            delta = r["sharpe"] - v0["sharpe"]
            print(f"\n  {r['label']} Δ Sharpe vs V0 baseline: {delta:+.2f}", flush=True)
            if delta >= 0.20:
                # LOFO
                print(f"    LOFO test:", flush=True)
                v0_pnl = v0["df_v"]
                vR_pnl = r["df_v"]
                for excl in range(1, 10):
                    v0_r = v0_pnl[v0_pnl["fold"]!=excl]["net_pnl_bps"].to_numpy()
                    vR_r = vR_pnl[vR_pnl["fold"]!=excl]["net_pnl_bps"].to_numpy()
                    d = _sharpe(vR_r) - _sharpe(v0_r)
                    flag = "  ← drives lift" if abs(d) < 0.5 * delta else ""
                    print(f"      exclude fold {excl}: Δ = {d:+.2f}{flag}", flush=True)

    pd.DataFrame([{k:v for k,v in r.items() if k not in ("per_fold","df_v")}
                  for r in results]).to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"\nResults saved to {OUT_DIR}/", flush=True)
    print(f"Total runtime: {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
