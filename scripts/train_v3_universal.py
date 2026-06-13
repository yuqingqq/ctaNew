"""Train WINNER_BTC_v3_PURE universal model (24 features) and run V3.1 β-hedged
on the predictions. Combines Phase 3+5 from docs/vBTC_V3_FEATURE_PLAN.md.

Compares vs WINNER_17 baseline (+0.74 Sharpe) using the same architecture
(LGBM 9-fold × 5-seed, expanding-window, target_β = α_β / σ_idio, β-hedged
V3.1 6-sleeve overlay).
"""
from __future__ import annotations
import sys, time, json, importlib.util, warnings
from pathlib import Path
from collections import deque, defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice, _train
spec = importlib.util.spec_from_file_location("psl", REPO / "scripts/phase_ah_sleeve.py")
psl = importlib.util.module_from_spec(spec); spec.loader.exec_module(psl)
from ml.research.alpha_v4_xs import block_bootstrap_ci

PANEL = REPO / "outputs/vBTC_features_btc_v3/panel_v3.parquet"
FEAT_JSON = REPO / "outputs/vBTC_features_btc_v3/winner_btc_v3_features.json"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
OUT_DIR = REPO / "outputs/vBTC_audit_panel_v3_universal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RC = 0.50
THRESHOLD = 1 - RC
SEEDS = (42, 1337, 7, 19, 2718)
ALL_FOLDS = list(range(10))
OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
CAPITAL = 100.0


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


def train_and_predict(panel, folds_all, feat_set, listings):
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

    print(f"\n  Training universal v3 ({len(feat_set)} features, 10 folds × 5 seeds)...",
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


def main():
    print("=== Train WINNER_BTC_v3_PURE universal model + V3.1 β-hedged ===\n", flush=True)
    t_start = time.time()
    feat_set = json.load(open(FEAT_JSON))["features"]
    print(f"Features ({len(feat_set)}):", flush=True)
    for f in feat_set: print(f"  - {f}", flush=True)

    listings = get_listings()
    panel = pd.read_parquet(PANEL)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    print(f"\nPanel: {len(panel):,} rows × {panel['symbol'].nunique()} symbols", flush=True)
    panel_syms = set(panel["symbol"].unique())
    for s, t in panel.groupby("symbol")["open_time"].min().items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t

    # Verify all features present
    missing = [f for f in feat_set if f not in panel.columns]
    if missing:
        print(f"ERROR: features missing from panel: {missing}", flush=True)
        sys.exit(1)

    folds_all = _multi_oos_splits(panel)
    apd = train_and_predict(panel, folds_all, feat_set, listings)
    apd.to_parquet(OUT_DIR / "all_predictions.parquet", index=False)
    print(f"\nPredictions saved: {OUT_DIR / 'all_predictions.parquet'}", flush=True)

    cyc_ic = apd.dropna(subset=["alpha_A"]).groupby("open_time").apply(
        lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= 5 else np.nan
    ).dropna()
    per_cycle_ic = float(cyc_ic.mean())
    print(f"\nPer-cycle IC (universal v3, all folds): {per_cycle_ic:+.4f}", flush=True)

    # Run V3.1 β-hedged
    print(f"\nRunning V3.1 β-hedged (OOS folds {OOS_FOLDS})...", flush=True)
    df_v, records, universe, sampled_t = run_v31_beta_hedged(apd, panel_syms, listings)
    net = df_v["net_pnl_bps"].to_numpy()
    sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=1000)
    total_d = net.sum() / 1e4 * CAPITAL
    end_eq = CAPITAL + total_d

    print(f"\n{'='*70}", flush=True)
    print(f"  WINNER_BTC_v3_PURE ({len(feat_set)} features) V3.1 β-hedged", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Sharpe          : {sh:+.2f} [{lo:+.2f}, {hi:+.2f}]", flush=True)
    print(f"  end-equity $100 : ${end_eq:.2f} ({total_d/CAPITAL*100:+.1f}%)", flush=True)
    print(f"  totPnL          : {net.sum():+.0f} bps", flush=True)
    print(f"  maxDD           : {_max_dd(net):+.0f} bps", flush=True)
    print(f"  gross/cycle     : {df_v['gross_pnl_bps'].mean():+.2f} bps", flush=True)
    print(f"  cost/cycle      : {df_v['cost_bps'].mean():+.2f} bps", flush=True)
    print(f"  per-cycle IC    : {per_cycle_ic:+.4f}", flush=True)
    print(f"  folds positive  : {folds_positive(df_v)}/9", flush=True)
    print(f"  traded cycles   : {records['traded'].sum()}/{len(records)}", flush=True)
    df_v.to_csv(OUT_DIR / "v31_hedged.csv", index=False)

    print(f"\n  Reference baselines:", flush=True)
    print(f"    WINNER_21 + β-residual β-hedged: Sharpe +0.57, end-eq $121.18, IC +0.0149", flush=True)
    print(f"    WINNER_17 + β-residual β-hedged: Sharpe +0.74, end-eq $126.96, IC +0.0157", flush=True)
    print(f"    Adoption gate (+0.20 over WINNER_17): Sharpe ≥ +0.94", flush=True)
    delta_sh = sh - 0.74
    print(f"\n  Δ Sharpe vs WINNER_17 baseline: {delta_sh:+.2f}", flush=True)
    if sh >= 0.94 and df_v['gross_pnl_bps'].mean() > 1.90:
        print(f"  ADOPTION GATE: PASSED ✓ (Sharpe ≥ +0.94 AND gross/cycle improves)", flush=True)
    elif sh >= 0.94:
        print(f"  ADOPTION GATE: PARTIAL — Sharpe ≥ +0.94 but gross/cycle < WINNER_17's +1.90", flush=True)
    else:
        print(f"  ADOPTION GATE: NOT MET", flush=True)
    print(f"\nTotal runtime: {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
