"""Phase 1: rebuild target_A against a FIXED reference basket (BTC-only or BTC+ETH).

Replaces the current basket-residual target (which moves whenever the universe
changes) with a target defined against a constant reference. The model now learns
"alpha vs BTC (and optionally ETH)" — a universe-invariant concept.

For each variant:
  1. Build new alpha_ref = return_pct - reference_basket_return (PIT, no look-ahead)
  2. Compute per-symbol std of alpha_ref using only the FIRST FOLD's training window
     (truly PIT: this std value never sees test-set data; reused across folds).
  3. target_ref = alpha_ref / std_per_sym
  4. Train WINNER_21 (same hyperparams, same procedure) on target_ref
  5. Save predictions with alpha_A column = alpha_ref (so V3.1's IC ranker uses the
     coherent realization target)
  6. Run V3.1 with rolling-IC + K=3 + conv_gate + PM_M2 + sleeve overlay

Pass criteria (per user):
  - 51-panel V3.1 Sharpe > +1.0
  - random-drop K=5 std materially falls (target < 0.4 vs current 0.70)

Usage:
  python3 scripts/diag_phase1_fixed_ref_target.py btc
  python3 scripts/diag_phase1_fixed_ref_target.py btc_eth
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


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(psl.CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def build_reference_residual(panel, variant):
    """Add columns alpha_ref and basket_fwd_ref to panel.

    variant='btc'     : reference = BTCUSDT.return_pct
    variant='btc_eth' : reference = 0.5*(BTCUSDT + ETHUSDT).return_pct
    """
    print(f"  Building reference basket: {variant}", flush=True)
    btc = panel[panel.symbol == "BTCUSDT"][["open_time", "return_pct"]].rename(
        columns={"return_pct": "btc_ret"})
    eth = panel[panel.symbol == "ETHUSDT"][["open_time", "return_pct"]].rename(
        columns={"return_pct": "eth_ret"})
    ref = btc.merge(eth, on="open_time", how="outer")
    if variant == "btc":
        ref["basket_fwd_ref"] = ref["btc_ret"]
    elif variant == "btc_eth":
        ref["basket_fwd_ref"] = 0.5 * (ref["btc_ret"] + ref["eth_ret"])
    else:
        raise ValueError(variant)
    ref = ref[["open_time", "basket_fwd_ref"]]
    panel = panel.merge(ref, on="open_time", how="left")
    panel["alpha_ref"] = panel["return_pct"] - panel["basket_fwd_ref"]
    print(f"  alpha_ref stats: mean={panel['alpha_ref'].mean():+.6f} "
          f"std={panel['alpha_ref'].std():.6f}", flush=True)
    return panel


def compute_per_sym_std_first_fold(panel, fold0):
    """PIT per-symbol std of alpha_ref using ONLY the first fold's training window.

    This std is fixed across all folds — no look-ahead, no per-fold leakage. Same
    structural pattern as the production target_A normalization (single value per
    symbol, fixed at panel build time)."""
    train, _, _ = _slice(panel, fold0)
    per_sym_std = train.groupby("symbol")["alpha_ref"].std().to_dict()
    return per_sym_std


def train_fold(panel, fold, feat_set, eligible_syms):
    train, cal, test = _slice(panel, fold)
    tr = train[(train["autocorr_pctile_7d"] >= THRESHOLD) & (train["symbol"].isin(eligible_syms))]
    ca = cal[(cal["autocorr_pctile_7d"] >= THRESHOLD) & (cal["symbol"].isin(eligible_syms))]
    test_r = test[test["symbol"].isin(eligible_syms)].copy()
    if len(tr) < 1000 or len(ca) < 200 or len(test_r) < 100: return None, None
    Xt = tr[feat_set].to_numpy(np.float32)
    Xc = ca[feat_set].to_numpy(np.float32)
    Xtest = test_r[feat_set].to_numpy(np.float32)
    yt = tr["target_ref"].to_numpy(np.float32)
    yc = ca["target_ref"].to_numpy(np.float32)
    mt = ~np.isnan(yt); mc = ~np.isnan(yc)
    if mt.sum() < 1000 or mc.sum() < 200: return None, None
    preds = []
    for s in SEEDS:
        m = _train(Xt[mt], yt[mt], Xc[mc], yc[mc], seed=s)
        preds.append(m.predict(Xtest, num_iteration=m.best_iteration))
    return test_r, np.mean(preds, axis=0)


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


def folds_positive(df_v):
    return sum(1 for _, g in df_v.groupby("fold") if _sharpe(g["net_pnl_bps"]) > 0)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("btc", "btc_eth"):
        print("Usage: python3 scripts/diag_phase1_fixed_ref_target.py {btc|btc_eth}",
              flush=True)
        sys.exit(1)
    variant = sys.argv[1]
    OUT = REPO / f"outputs/vBTC_phase1_ref_{variant}"
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 90}", flush=True)
    print(f"  Phase 1{'A' if variant=='btc' else 'B'}: fixed reference basket = {variant.upper()}",
          flush=True)
    print(f"{'=' * 90}\n", flush=True)
    t_start = time.time()

    panel = pd.read_parquet(PANEL_PATH)
    print(f"Loaded panel: {len(panel):,} rows × {panel['symbol'].nunique()} symbols",
          flush=True)

    # 1. Build new residual target
    panel = build_reference_residual(panel, variant)
    # 2. Per-symbol std from first-fold training window only (PIT)
    folds_all = _multi_oos_splits(panel)
    per_sym_std = compute_per_sym_std_first_fold(panel, folds_all[0])
    panel["per_sym_std_ref"] = panel["symbol"].map(per_sym_std)
    # Guard against zero/nan std
    panel["per_sym_std_ref"] = panel["per_sym_std_ref"].fillna(panel["alpha_ref"].std())
    panel["per_sym_std_ref"] = panel["per_sym_std_ref"].clip(lower=1e-6)
    panel["target_ref"] = panel["alpha_ref"] / panel["per_sym_std_ref"]
    print(f"\nTarget stats: min={panel['target_ref'].min():.2f} "
          f"p1={panel['target_ref'].quantile(0.01):.2f} "
          f"p99={panel['target_ref'].quantile(0.99):.2f} "
          f"max={panel['target_ref'].max():.2f}", flush=True)
    print(f"|target_ref| > 5: "
          f"{(panel['target_ref'].abs() > 5).sum():,} rows ({(panel['target_ref'].abs()>5).mean()*100:.2f}%)",
          flush=True)
    print(f"per_sym_std_ref values (5 syms): "
          f"{dict(list(per_sym_std.items())[:5])}", flush=True)

    feat_set = [f for f in WINNER_21 if f in panel.columns]
    print(f"\nFeature set: {len(feat_set)} features (incl sym_id)", flush=True)

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

    # 3. Train all folds
    print(f"\n--- Train 10 folds × 5 seeds (target = alpha_{variant} normalized) ---",
          flush=True)
    all_preds = []
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        eligible = eligibility_at(folds_all[fid]["cal_start"])
        td, p = train_fold(panel, folds_all[fid], feat_set, eligible)
        if td is None: continue
        cols = ["symbol", "open_time", "alpha_ref", "return_pct"]
        if "exit_time" in td.columns: cols.append("exit_time")
        df = td[cols].copy()
        df["pred"] = p; df["fold"] = fid
        df = df.rename(columns={"alpha_ref": "alpha_A"})  # V3.1 expects this column
        if "exit_time" not in df.columns:
            df["exit_time"] = df["open_time"] + pd.Timedelta(minutes=HORIZON * 5)
        all_preds.append(df)
        print(f"  fold {fid}: n={len(td):,} ({time.time()-t0:.0f}s)", flush=True)
    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    apd.to_parquet(OUT / "all_predictions.parquet", index=False)
    print(f"  Saved predictions: {len(apd):,} rows", flush=True)

    # Pred quality diagnostics
    cyc_ic = apd.dropna(subset=["alpha_A"]).groupby("open_time").apply(
        lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= 5 else np.nan
    ).dropna()
    print(f"  Per-cycle IC (vs new alpha_{variant}): mean={cyc_ic.mean():+.4f} "
          f"median={cyc_ic.median():+.4f}", flush=True)

    # 4. Run V3.1 with IC ranking
    print(f"\n--- Run V3.1 with IC ranking on Phase 1{('A' if variant=='btc' else 'B')} predictions ---",
          flush=True)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)

    def elig(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]
    universe = psl.build_rolling_ic_universe(apd, sampled_t, psl.TOP_N, elig)
    records = psl.run_production_protocol_save_sleeves(apd, universe)

    # Load fwd_rets
    print("  loading close prices...", flush=True)
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

    df_v = psl.aggregate_sleeves(records, fwd_rets_4h)
    net = df_v["net_pnl_bps"].to_numpy()
    sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=1000)
    df_v.to_csv(OUT / f"v31_phase1_{variant}.csv", index=False)

    print("\n" + "=" * 70)
    print(f"  V3.1 RESULTS — Phase 1{'A' if variant=='btc' else 'B'} ({variant.upper()})")
    print("=" * 70)
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
    print(f"\n  Reference Sharpe 51-panel production = +2.23", flush=True)
    print(f"  Pass threshold (Phase 1):                 Sharpe > +1.0", flush=True)
    verdict = "PASS" if sh > 1.0 else "FAIL"
    print(f"  Verdict on Sharpe: {verdict}", flush=True)

    print("\n=== PER-FOLD Sharpe ===", flush=True)
    for fid in OOS_FOLDS:
        g = df_v[df_v["fold"] == fid]["net_pnl_bps"].to_numpy()
        print(f"  fold {fid}: {_sharpe(g):+.2f}", flush=True)

    # Universe set composition
    print("\n=== Universe composition at each boundary ===", flush=True)
    seen = []
    for t in sampled_t:
        u = universe.get(t, set())
        if not seen or u != seen[-1]["u"]:
            seen.append({"t": t, "u": u})
    for b in seen:
        print(f"  {b['t']}: |U|={len(b['u'])} {sorted(b['u'])}", flush=True)

    print(f"\nTotal Phase 1{'A' if variant=='btc' else 'B'} runtime: {time.time()-t_start:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
