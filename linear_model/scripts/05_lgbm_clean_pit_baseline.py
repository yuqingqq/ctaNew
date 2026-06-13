"""Step 5a: Run LGBM WINNER_17 with clean-PIT (shift 49) to test if Ridge's
collapse is leak-dependent or model-class-dependent.

If LGBM also drops to Sharpe near 0 or negative with shift(49), then the
+0.74 baseline was leveraging the look-ahead leak — Ridge's clean-PIT result
is the honest baseline.

If LGBM stays near +0.74 with shift(49), Ridge has a model-specific issue.
"""
from __future__ import annotations
import sys, time, warnings, importlib.util
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
OUT = REPO / "linear_model/results"
OUT.mkdir(parents=True, exist_ok=True)

# Production hyperparams matching diag_winner17_beta_residual_51_vs_111.py
RC = 0.50
THRESHOLD = 1 - RC
SEEDS = (42, 1337, 7, 19, 2718)
ALL_FOLDS = list(range(10))
OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
BETA_WIN_DAYS = 90
BAR_PER_DAY = 288
HORIZON = 48
CAPITAL = 100.0
COST_PER_UNIT_ABS_DELTA = psl.COST_PER_UNIT_ABS_DELTA
TOP_N = psl.TOP_N
HORIZON_ENTRY = psl.HORIZON_ENTRY

# WINNER_17
ALL_DROPS = ["return_1d_xs_rank", "bk_ret_48b", "volume_ma_50",
             "ema_slope_20_1h", "ema_slope_20_1h_xs_rank",
             "vwap_zscore_xs_rank", "vwap_zscore",
             "atr_pct_xs_rank", "dom_z_7d_vs_bk", "obv_z_1d_xs_rank",
             "obv_signal", "price_volume_corr_10",
             "hour_cos", "hour_sin"]
FUNDING_LEAN = ["funding_rate", "funding_rate_z_7d"]
ADD_CROSS_BTC = ["corr_to_btc_1d", "idio_vol_to_btc_1h", "beta_to_btc_change_5d"]
ADD_MORE_FUNDING = ["funding_rate_1d_change", "funding_streak_pos"]
WINNER_21 = ([f for f in XS_FEATURE_COLS_V6_CLEAN if f not in ALL_DROPS]
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


def compute_pit_beta_clean(panel, win_days=90, shift_bars=49,
                            min_warmup=1000):
    """CLEAN-PIT version: shift 49 bars (= HORIZON+1) to ensure β at time t
    uses ONLY past close prices, no overlap with target window."""
    btc_ret = panel[panel.symbol == "BTCUSDT"][["open_time", "return_pct"]].rename(
        columns={"return_pct": "btc_ret"}).drop_duplicates("open_time")
    bar_window = win_days * BAR_PER_DAY
    out = []
    for sym, g in panel.groupby("symbol"):
        gg = g[["open_time", "return_pct"]].merge(btc_ret, on="open_time", how="left")
        gg = gg.sort_values("open_time").reset_index(drop=True)
        if sym == "BTCUSDT":
            gg["beta_pit"] = 1.0
        else:
            y = gg["return_pct"]; x = gg["btc_ret"]
            cov = y.rolling(bar_window, min_periods=min_warmup).cov(x)
            var = x.rolling(bar_window, min_periods=min_warmup).var()
            gg["beta_pit"] = (cov / var.replace(0, np.nan)).shift(shift_bars)
        gg["symbol"] = sym
        out.append(gg)
    return pd.concat(out, ignore_index=True)[["symbol", "open_time", "beta_pit"]]


def prepare(shift_bars):
    print(f"\n=== Preparing panel with shift({shift_bars}) ===", flush=True)
    panel = pd.read_parquet(PANEL_BASE)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    folds_all = _multi_oos_splits(panel)
    pit_beta = compute_pit_beta_clean(panel, win_days=BETA_WIN_DAYS,
                                       shift_bars=shift_bars)
    panel = panel.merge(pit_beta, on=["symbol", "open_time"], how="left")
    btc_t = panel[panel.symbol=="BTCUSDT"][["open_time","return_pct"]].rename(
        columns={"return_pct":"btc_ret_t"}).drop_duplicates("open_time")
    panel = panel.merge(btc_t, on="open_time", how="left")
    panel["alpha_beta"] = panel["return_pct"] - panel["beta_pit"]*panel["btc_ret_t"]
    train0, _, _ = _slice(panel, folds_all[0])
    sigma = train0.groupby("symbol")["alpha_beta"].std().to_dict()
    panel["sigma_idio_ref"] = panel["symbol"].map(sigma).fillna(
        panel["alpha_beta"].std()).clip(lower=1e-6)
    panel["target_beta"] = panel["alpha_beta"] / panel["sigma_idio_ref"]
    return panel, folds_all


def train_fold(panel, fold, feat_set, eligible):
    train, cal, test = _slice(panel, fold)
    tr = train[(train["autocorr_pctile_7d"]>=THRESHOLD) & (train.symbol.isin(eligible))]
    ca = cal[(cal["autocorr_pctile_7d"]>=THRESHOLD) & (cal.symbol.isin(eligible))]
    te = test[test.symbol.isin(eligible)].copy()
    if len(tr) < 1000 or len(ca) < 200 or len(te) < 100: return None, None
    Xt = tr[feat_set].to_numpy(np.float32); Xc = ca[feat_set].to_numpy(np.float32)
    Xte = te[feat_set].to_numpy(np.float32)
    yt = tr["target_beta"].to_numpy(np.float32); yc = ca["target_beta"].to_numpy(np.float32)
    mt = ~np.isnan(yt); mc = ~np.isnan(yc)
    if mt.sum()<1000 or mc.sum()<200: return None, None
    preds = []
    for s in SEEDS:
        m = _train(Xt[mt], yt[mt], Xc[mc], yc[mc], seed=s)
        preds.append(m.predict(Xte, num_iteration=m.best_iteration))
    return te, np.mean(preds, axis=0)


def train_all(panel, folds_all, feat_set, listings):
    panel_syms = set(panel.symbol.unique())
    panel_first = panel.groupby("symbol")["open_time"].min()
    for s, t in panel_first.items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t
    def elig_at(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC") if isinstance(b,(int,np.integer)) \
             else (pd.Timestamp(b) if pd.Timestamp(b).tz else pd.Timestamp(b).tz_localize("UTC"))
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s]<=cutoff}
    all_preds = []
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        eligible = elig_at(folds_all[fid]["cal_start"])
        td, p = train_fold(panel, folds_all[fid], feat_set, eligible)
        if td is None: continue
        cols = ["symbol","open_time","alpha_beta","return_pct"]
        if "exit_time" in td.columns: cols.append("exit_time")
        df = td[cols].copy()
        df["pred"] = p; df["fold"] = fid
        df = df.rename(columns={"alpha_beta":"alpha_A"})
        if "exit_time" not in df.columns:
            df["exit_time"] = df["open_time"] + pd.Timedelta(minutes=HORIZON*5)
        all_preds.append(df)
        print(f"  fold {fid}: n={len(td):,} ({time.time()-t0:.0f}s)", flush=True)
    return pd.concat(all_preds, ignore_index=True).sort_values(["open_time","symbol"])


def aggregate_alpha(records, alpha_wide):
    sleeve_queue = deque(maxlen=psl.N_SLEEVES)
    prev_weights = {}
    rows = []
    for _, rec in records.iterrows():
        t = rec["time"]; fold = rec["fold"]
        if rec["traded"]:
            sleeve_queue.append({"entry_time":t,"longs":list(rec["long_basket"]),
                                  "shorts":list(rec["short_basket"])})
        else:
            sleeve_queue.append({"entry_time":t,"longs":[],"shorts":[]})
        target_weights = defaultdict(float)
        sw = 1.0 / psl.N_SLEEVES
        for sl in sleeve_queue:
            nL, nS = len(sl["longs"]), len(sl["shorts"])
            if nL==0 or nS==0: continue
            for s in sl["longs"]: target_weights[s] += sw*(1.0/nL)
            for s in sl["shorts"]: target_weights[s] -= sw*(1.0/nS)
        gross = 0.0
        if t in alpha_wide.index:
            a = alpha_wide.loc[t]
            for sym, w in prev_weights.items():
                if sym in a.index and not pd.isna(a[sym]): gross += w*a[sym]*1e4
        syms = set(target_weights.keys()) | set(prev_weights.keys())
        abs_d = sum(abs(target_weights.get(s,0)-prev_weights.get(s,0)) for s in syms)
        cost = abs_d * COST_PER_UNIT_ABS_DELTA
        rows.append({"time":t,"fold":fold,"gross_pnl_bps":gross,"cost_bps":cost,
                     "net_pnl_bps":gross-cost,"turnover":abs_d})
        prev_weights = dict(target_weights)
    return pd.DataFrame(rows)


def run_v31(apd, panel_syms, listings):
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    def elig_pit(b):
        ts = b if isinstance(b, pd.Timestamp) else pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=psl.MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s]<=cutoff}
    target_t = sorted(apd[apd.fold.isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::HORIZON_ENTRY]
    universe = psl.build_rolling_ic_universe(apd, sampled_t, TOP_N, elig_pit)
    records = psl.run_production_protocol_save_sleeves(apd, universe)
    alpha_wide = apd.pivot_table(index="open_time", columns="symbol",
                                   values="alpha_A", aggfunc="first").sort_index()
    df_v = aggregate_alpha(records, alpha_wide)
    return df_v, records


def main():
    print("=== Test LGBM WINNER_17 with shift(49) clean-PIT vs shift(1) leaky ===\n",
          flush=True)
    t0 = time.time()
    listings = get_listings()
    results = {}
    for shift_bars in (1, 49):
        panel, folds_all = prepare(shift_bars)
        panel_syms = set(panel.symbol.unique())
        for s, t in panel.groupby("symbol")["open_time"].min().items():
            if s not in listings:
                t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
                listings[s] = t
        feat = [f for f in WINNER_17 if f in panel.columns]
        apd = train_all(panel, folds_all, feat, listings)
        apd.to_parquet(OUT / f"lgbm_shift{shift_bars}_predictions.parquet", index=False)
        cyc_ic = apd.dropna(subset=["alpha_A"]).groupby("open_time").apply(
            lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g)>=5 else np.nan
        ).dropna()
        per_cycle_ic = float(cyc_ic.mean())
        df_v, records = run_v31(apd, panel_syms, listings)
        net = df_v["net_pnl_bps"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=1000)
        results[shift_bars] = {
            "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
            "end_eq": CAPITAL + net.sum()/1e4*CAPITAL,
            "per_cycle_ic": per_cycle_ic,
            "gross_cycle": float(df_v["gross_pnl_bps"].mean()),
            "folds_pos": folds_positive(df_v),
            "n_traded": int(records["traded"].sum()),
        }
        df_v.to_csv(OUT / f"lgbm_shift{shift_bars}_v31.csv", index=False)
        print(f"\n  shift({shift_bars}): Sharpe={sh:+.2f} [{lo:+.2f},{hi:+.2f}], "
              f"IC={per_cycle_ic:+.4f}, folds+={results[shift_bars]['folds_pos']}/9, "
              f"end-eq=${results[shift_bars]['end_eq']:.2f}", flush=True)

    print("\n" + "="*90, flush=True)
    print("  LEAK QUANTIFICATION", flush=True)
    print("="*90, flush=True)
    r1 = results[1]; r49 = results[49]
    print(f"  shift(1)  [leaky, original baseline]: Sharpe {r1['sharpe']:+.2f}, "
          f"IC {r1['per_cycle_ic']:+.4f}, end-eq ${r1['end_eq']:.2f}", flush=True)
    print(f"  shift(49) [clean-PIT]:                Sharpe {r49['sharpe']:+.2f}, "
          f"IC {r49['per_cycle_ic']:+.4f}, end-eq ${r49['end_eq']:.2f}", flush=True)
    print(f"\n  Δ Sharpe from leak fix: {r49['sharpe']-r1['sharpe']:+.2f}", flush=True)
    print(f"  Δ IC from leak fix:     {r49['per_cycle_ic']-r1['per_cycle_ic']:+.4f}",
          flush=True)
    print(f"\n  Total time: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
