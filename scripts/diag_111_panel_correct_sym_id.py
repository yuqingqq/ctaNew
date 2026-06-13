"""Train WINNER_21 on the FULL 111-symbol panel with correct (alphabetical) sym_id,
build audit panel, run V3.1 with IC ranking, compare to 51-panel production.

The 111-panel target_A is clipped at ±5 (a known issue from Phase E5a/E5b — see
[[project_vBTC_status]]). We keep it as-is so this test is reproducible against
existing pipeline. Separate question is whether unclipping would change anything.

Steps:
  1. Add sym_id (alphabetical over 111 symbols) to panel.
  2. Train 10 folds × 5 seeds, expanding window, PIT eligibility (same as prod).
  3. Save predictions → outputs/vBTC_audit_panel_111/all_predictions.parquet
  4. Run V3.1 with rolling-IC top-15 selector + K=3 + conv_gate + PM_M2 + sleeve.
  5. Head-to-head Sharpe / per-cycle IC / universe composition vs 51-panel.
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

PANEL_111 = REPO / "outputs/vBTC_features_expanded/panel_variants_with_funding.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
OUT_DIR = REPO / "outputs/vBTC_audit_panel_111_correct"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
SEEDS = (42, 1337, 7, 19, 2718)
ALL_FOLDS = list(range(10))
OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60

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


def train_fold(panel, fold, feat_set, eligible_syms):
    train, cal, test = _slice(panel, fold)
    tr = train[(train["autocorr_pctile_7d"] >= THRESHOLD) & (train["symbol"].isin(eligible_syms))]
    ca = cal[(cal["autocorr_pctile_7d"] >= THRESHOLD) & (cal["symbol"].isin(eligible_syms))]
    test_r = test[test["symbol"].isin(eligible_syms)].copy()
    if len(tr) < 1000 or len(ca) < 200 or len(test_r) < 100: return None, None
    Xt = tr[feat_set].to_numpy(np.float32)
    Xc = ca[feat_set].to_numpy(np.float32)
    Xtest = test_r[feat_set].to_numpy(np.float32)
    yt = tr["target_A"].to_numpy(np.float32)
    yc = ca["target_A"].to_numpy(np.float32)
    mt = ~np.isnan(yt); mc = ~np.isnan(yc)
    if mt.sum() < 1000 or mc.sum() < 200: return None, None
    preds = []
    for s in SEEDS:
        m = _train(Xt[mt], yt[mt], Xc[mc], yc[mc], seed=s)
        preds.append(m.predict(Xtest, num_iteration=m.best_iteration))
    return test_r, np.mean(preds, axis=0)


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(psl.CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def main():
    print("=== 111-panel retrain with correct alphabetical sym_id ===\n", flush=True)
    panel = pd.read_parquet(PANEL_111)
    print(f"Panel: {len(panel):,} rows × {panel['symbol'].nunique()} symbols", flush=True)

    # Add sym_id (alphabetical over 111)
    syms = sorted(panel["symbol"].unique())
    sym_to_id = {s: i for i, s in enumerate(syms)}
    panel["sym_id"] = panel["symbol"].map(sym_to_id).astype("int64")
    print(f"sym_id added: range [0..{len(syms)-1}], dtype int64", flush=True)

    # Check overlap with 51-panel encoding for inspection only
    syms_51 = sorted(pd.read_parquet(
        REPO/"outputs/vBTC_features/panel_variants_with_funding.parquet",
        columns=["symbol"])["symbol"].unique())
    sym_to_id_51 = {s: i for i, s in enumerate(syms_51)}
    same_id = sum(1 for s in syms_51 if sym_to_id_51[s] == sym_to_id[s])
    print(f"\nsym_id encoding agreement with 51-panel: {same_id}/{len(syms_51)} symbols "
          f"keep same id; {len(syms_51)-same_id} have shifted", flush=True)
    # Show a few shifts
    shifts = [(s, sym_to_id_51[s], sym_to_id[s])
              for s in syms_51 if sym_to_id_51[s] != sym_to_id[s]][:8]
    if shifts:
        print("  examples of shifted ids:", flush=True)
        for s, a, b in shifts: print(f"    {s}: {a} → {b}", flush=True)

    feat_set = [f for f in WINNER_21 if f in panel.columns]
    print(f"\nFeature set: {len(feat_set)} features", flush=True)

    folds_all = _multi_oos_splits(panel)
    listings = get_listings()
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

    print("\n--- Train 10 folds × 5 seeds on 111-panel ---", flush=True)
    all_preds = []
    t_total = time.time()
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        eligible = eligibility_at(folds_all[fid]["cal_start"])
        td, p = train_fold(panel, folds_all[fid], feat_set, eligible)
        if td is None: continue
        cols = ["symbol", "open_time", "alpha_A", "return_pct"]
        if "exit_time" in td.columns: cols.append("exit_time")
        df = td[cols].copy()
        df["pred"] = p; df["fold"] = fid
        if "exit_time" not in df.columns:
            df["exit_time"] = df["open_time"] + pd.Timedelta(minutes=HORIZON * 5)
        all_preds.append(df)
        print(f"  fold {fid}: n_test={len(td):,} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  total train: {time.time()-t_total:.0f}s", flush=True)

    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    apd.to_parquet(OUT_DIR / "all_predictions.parquet", index=False)
    print(f"\nSaved predictions: {len(apd):,} rows", flush=True)

    # Prediction sanity diagnostics
    pred_std_per_sym = apd.groupby("symbol")["pred"].std().describe()
    cyc_ic = apd.dropna(subset=["alpha_A"]).groupby("open_time").apply(
        lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= 5 else np.nan
    ).dropna()
    print(f"\n  Per-symbol pred std: median={pred_std_per_sym['50%']:.4f}, "
          f"min={pred_std_per_sym['min']:.4f}, max={pred_std_per_sym['max']:.4f}",
          flush=True)
    print(f"  Per-cycle IC: mean={cyc_ic.mean():+.4f} median={cyc_ic.median():+.4f} "
          f"std={cyc_ic.std():.4f}", flush=True)

    # === RUN V3.1 with IC ranking ===
    print("\n--- Run V3.1 with IC ranking on 111-panel ---", flush=True)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)

    def elig111(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]
    t0 = time.time()
    universe = psl.build_rolling_ic_universe(apd, sampled_t, psl.TOP_N, elig111)
    print(f"  built universe ({time.time()-t0:.0f}s)", flush=True)

    # Universe size stats
    u_sizes = [len(u) for u in universe.values()]
    print(f"  universe sizes (top-15 from 111 eligible): "
          f"min/median/max = {min(u_sizes)}/{int(np.median(u_sizes))}/{max(u_sizes)}",
          flush=True)
    # Print universes at each boundary
    seen = []
    for t in sampled_t:
        u = universe.get(t, set())
        if not seen or u != seen[-1]["u"]:
            seen.append({"t": t, "u": u})
    syms_51_set = set(syms_51)
    for b in seen:
        n_from_51 = len(b["u"] & syms_51_set)
        new_picks = sorted(b["u"] - syms_51_set)
        print(f"  {b['t']}: |U|={len(b['u'])}, from-51 count={n_from_51}, "
              f"new picks={new_picks}", flush=True)

    records = psl.run_production_protocol_save_sleeves(apd, universe)
    print(f"  traded {records['traded'].sum()}/{len(records)}", flush=True)

    # Load fwd_rets for all 111 symbols
    print("  loading close prices for 111 symbols...", flush=True)
    t0 = time.time()
    frames = []
    for sym in syms:
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
    folds_pos = sum(1 for _, g in df_v.groupby("fold") if _sharpe(g["net_pnl_bps"]) > 0)

    print("\n" + "=" * 70)
    print("  V3.1 RESULTS ON 111-PANEL (with correct sym_id)")
    print("=" * 70)
    print(f"  per-cycle IC mean : {cyc_ic.mean():+.4f}", flush=True)
    print(f"  Sharpe            : {sh:+.2f} [{lo:+.2f}, {hi:+.2f}]", flush=True)
    print(f"  totPnL            : {net.sum():+.0f} bps", flush=True)
    print(f"  maxDD             : {_max_dd(net):+.0f} bps", flush=True)
    print(f"  gross/cycle       : {df_v['gross_pnl_bps'].mean():+.2f} bps", flush=True)
    print(f"  cost/cycle        : {df_v['cost_bps'].mean():+.2f} bps", flush=True)
    print(f"  net/cycle         : {df_v['net_pnl_bps'].mean():+.2f} bps", flush=True)
    print(f"  turnover/cycle    : {df_v['turnover'].mean():.3f}", flush=True)
    print(f"  folds positive    : {folds_pos}/9", flush=True)
    print(f"  cycles traded     : {records['traded'].sum()}/{len(records)}", flush=True)

    print("\n=== PER-FOLD Sharpe ===", flush=True)
    for fid in OOS_FOLDS:
        g = df_v[df_v["fold"] == fid]["net_pnl_bps"].to_numpy()
        print(f"  fold {fid}: {_sharpe(g):+.2f}", flush=True)

    print("\n=== Reference: 51-panel production V3.1 = +2.23 ===", flush=True)

    df_v.to_csv(OUT_DIR / "v31_111_correct.csv", index=False)


if __name__ == "__main__":
    main()
