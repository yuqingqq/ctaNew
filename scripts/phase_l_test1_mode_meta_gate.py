"""Phase L / Test 1: mode meta-gate — predict at each cycle which of
{production, sparse, flat} maximizes per-cycle net PnL using PIT regime features.

Approach:
  1. Load production per-cycle PnL (margin=0 from K3 = WINNER_21 baseline).
  2. Load sparse per-cycle PnL (margin=0.25, K_min=1 — K3 best in-sample).
  3. Compute PIT meta-features per cycle from all_predictions + panel:
     - pred_disp (top-K mean - bot-K mean within universe)
     - topK_overlap_prev (jaccard of top-K vs prev cycle's top-K)
     - rank_churn (1 - topK_overlap_prev for combined L+S)
     - rolling_IC_universe (trailing 30d avg per-symbol IC in universe)
     - BTC_30d_drawdown (BTC close drawdown from rolling 30d high)
     - BTC_realized_vol (1d realized vol on BTC)
     - xs_alpha_dispersion (xs_alpha_dispersion_48b mean across universe)
     - funding_dispersion (cross-symbol std of funding_rate within universe)
  4. Build labeled dataset: label = argmax(prod_pnl, sparse_pnl, 0).
  5. Per fold f (>=3): train LGBM classifier on folds < f, predict for fold f.
     Use uniform-prior default for folds 1-2.
  6. Stitch per-cycle PnL using predicted mode → nested OOS curve.
  7. Compare to production: per-fold Sharpe + matched mode-timing placebo.

Pass: nested Sharpe > production +0.3, ≥6/9 folds improve, beats placebo p95.

Output: outputs/vBTC_meta_gate/{meta_features.csv, per_cycle_nested.csv, ...}
"""
from __future__ import annotations
import sys, warnings, time
from pathlib import Path
from collections import deque, defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

OUT = REPO / "outputs/vBTC_meta_gate"
OUT.mkdir(parents=True, exist_ok=True)
APD_PATH = REPO / "outputs/vBTC_audit_panel/all_predictions.parquet"
PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
PROD_CSV = REPO / "outputs/vBTC_swap_rule/k2_robustness/per_cycle_m0.0.csv"
SPARSE_CSV = REPO / "outputs/vBTC_swap_rule/k2_robustness/per_cycle_m4.5.csv"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"

HORIZON = 48
CYCLES_PER_YEAR = (288 * 365) / HORIZON
COST_PER_LEG = 4.5
OOS_FOLDS = list(range(1, 10))
N_PLACEBO_SEEDS = 100
K = 4
TOP_N = 15
IC_WINDOW_DAYS = 180
IC_UPDATE_DAYS = 90
MIN_HISTORY_DAYS = 60
MIN_OBS_PER_SYM = 100


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR))


def to_ms_int(s):
    ts = pd.to_datetime(s)
    if hasattr(ts.dtype, "tz") and ts.dtype.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return ts.astype("datetime64[ms]").astype("int64").to_numpy()


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


def build_rolling_ic_universe(apd, target_times, top_n, eligibility_at_t):
    bar_ms = 5 * 60 * 1000
    window_ms = IC_WINDOW_DAYS * 288 * bar_ms
    update_ms = IC_UPDATE_DAYS * 288 * bar_ms
    df = apd.copy()
    df["t_int"] = to_ms_int(df["open_time"])
    df["exit_t_int"] = to_ms_int(df["exit_time"])
    df_clean = df.dropna(subset=["alpha_A"])
    t0_ms = int(pd.Timestamp(target_times[0]).timestamp() * 1000)
    boundaries = []
    for t in target_times:
        t_ms = int(pd.Timestamp(t).timestamp() * 1000)
        n = (t_ms - t0_ms) // update_ms
        b = t0_ms + n * update_ms
        boundaries.append((t, b))
    unique_b = sorted(set(b for _, b in boundaries))
    b2u = {}
    for b in unique_b:
        elig = eligibility_at_t(b)
        past = df_clean[(df_clean["t_int"] >= b - window_ms) &
                          (df_clean["t_int"] < b) &
                          (df_clean["exit_t_int"] <= b) &
                          (df_clean["symbol"].isin(elig))]
        if len(past) < 1000:
            b2u[b] = set(); continue
        ics = past.groupby("symbol").apply(
            lambda g: g["pred"].rank().corr(g["alpha_A"].rank())
            if len(g) >= MIN_OBS_PER_SYM else np.nan
        )
        ics_sorted = ics.dropna().sort_values(ascending=False)
        b2u[b] = set(ics_sorted.head(top_n).index.tolist()) if top_n else set(ics_sorted.index)
    return {t: b2u[b] for t, b in boundaries}


def compute_meta_features(apd, panel, universe):
    """For each cycle, compute PIT meta-features."""
    print(f"  Computing meta features...", flush=True)
    panel_short = panel[["open_time", "symbol", "xs_alpha_dispersion_48b",
                          "funding_rate", "btc_ret_288b", "btc_realized_vol_1d"]].copy()
    panel_short["open_time"] = pd.to_datetime(panel_short["open_time"], utc=True)
    # Pre-group by open_time for fast cycle-level lookup
    panel_by_time = {t: g for t, g in panel_short.groupby("open_time")}
    btc_vol_by_time = panel_short[panel_short["symbol"] == "BTCUSDT"].set_index("open_time")["btc_realized_vol_1d"].to_dict()

    # Subsample apd to HORIZON cycles
    apd_oos = apd[apd["fold"].isin(OOS_FOLDS)].copy()
    times = sorted(apd_oos["open_time"].unique())
    keep_t = set(times[::HORIZON])
    df_sub = apd_oos[apd_oos["open_time"].isin(keep_t)].copy()
    sampled_t = sorted(df_sub["open_time"].unique())

    # Trailing IC per universe: precompute per-symbol trailing IC at each boundary
    # Already have universe → keep
    prev_top_set_long = set(); prev_top_set_short = set()
    rows = []

    # For BTC dd: load BTC close directly from kline cache
    btc_klines = []
    btc_dir = KLINES_DIR / "BTCUSDT" / "5m"
    if btc_dir.exists():
        for f in sorted(btc_dir.glob("*.parquet")):
            try:
                df_kline = pd.read_parquet(f, columns=["open_time", "close"])
                btc_klines.append(df_kline)
            except Exception:
                pass
    if btc_klines:
        btc_panel = pd.concat(btc_klines, ignore_index=True)
        btc_panel["open_time"] = pd.to_datetime(btc_panel["open_time"], utc=True)
        btc_panel = btc_panel.sort_values("open_time").drop_duplicates("open_time").set_index("open_time")
        btc_panel["roll30d_max"] = btc_panel["close"].rolling(8640, min_periods=288).max()
        btc_panel["dd_30d"] = btc_panel["close"] / btc_panel["roll30d_max"] - 1.0
        btc_lookup = btc_panel["dd_30d"].to_dict()
    else:
        btc_lookup = {}

    for t in sampled_t:
        g = df_sub[df_sub["open_time"] == t]
        u = universe.get(t, set())
        g_u = g[g["symbol"].isin(u)]
        if len(g_u) < 2 * K + 1:
            rows.append({"time": t, "fold": int(g["fold"].iloc[0]) if len(g) else 0,
                          "valid": False})
            continue
        sym_arr = g_u["symbol"].to_numpy()
        pred_arr = g_u["pred"].to_numpy()
        idx_t = np.argpartition(-pred_arr, K - 1)[:K]
        idx_b = np.argpartition(pred_arr, K - 1)[:K]
        top_set = set(sym_arr[idx_t])
        bot_set = set(sym_arr[idx_b])
        pred_disp = float(pred_arr[idx_t].mean() - pred_arr[idx_b].mean())
        # Top-K overlap with prev
        if prev_top_set_long and prev_top_set_short:
            overlap_l = len(top_set & prev_top_set_long) / K
            overlap_s = len(bot_set & prev_top_set_short) / K
            topK_overlap = (overlap_l + overlap_s) / 2
        else:
            topK_overlap = 1.0
        rank_churn = 1.0 - topK_overlap

        # Rolling IC of universe (trailing 30d per-symbol IC averaged)
        # Use apd predictions and alphas in window
        t_ms = pd.Timestamp(t).timestamp() * 1000
        window_ms_30d = 30 * 288 * 5 * 60 * 1000
        apd_past = apd[(apd["open_time"] < t) &
                          (apd["symbol"].isin(u)) &
                          (apd["open_time"] >= pd.Timestamp(t_ms - window_ms_30d, unit="ms", tz="UTC"))]
        if len(apd_past) > 100:
            per_sym_ic = apd_past.groupby("symbol").apply(
                lambda gg: gg["pred"].rank().corr(gg["alpha_A"].rank())
                if len(gg) > 20 else np.nan
            )
            rolling_ic = float(per_sym_ic.dropna().mean()) if len(per_sym_ic.dropna()) > 0 else 0.0
        else:
            rolling_ic = 0.0

        # BTC dd & vol — use kline-derived rolling drawdown
        btc_dd = btc_lookup.get(t, np.nan)

        # BTC realized vol (1d)
        btc_vol = btc_vol_by_time.get(t, np.nan)

        # xs_alpha_dispersion + funding dispersion: pre-grouped fast lookup
        p_at_t = panel_by_time.get(t)
        if p_at_t is not None:
            p_u = p_at_t[p_at_t["symbol"].isin(u)]
            xs_disp = float(p_u["xs_alpha_dispersion_48b"].mean()) if len(p_u) > 0 else np.nan
            funding_disp = float(p_u["funding_rate"].std()) if len(p_u) > 0 else np.nan
        else:
            xs_disp = np.nan
            funding_disp = np.nan

        # avg pairwise corr (cheap proxy): just use 1 - xs_alpha_dispersion (inverse proxy)
        # Or compute from trailing returns — skip for now
        avg_pairwise_corr = np.nan

        rows.append({
            "time": t, "fold": int(g_u["fold"].iloc[0]), "valid": True,
            "pred_disp": pred_disp, "topK_overlap_prev": topK_overlap,
            "rank_churn": rank_churn, "rolling_IC_universe": rolling_ic,
            "BTC_30d_drawdown": btc_dd, "BTC_realized_vol": btc_vol,
            "xs_alpha_dispersion": xs_disp, "funding_dispersion": funding_disp,
            "avg_pairwise_corr": avg_pairwise_corr,
        })
        prev_top_set_long = top_set
        prev_top_set_short = bot_set
    return pd.DataFrame(rows)


def main():
    print("=== Phase L / Test 1: mode meta-gate ===\n", flush=True)

    # Load data
    apd = pd.read_parquet(APD_PATH)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    panel = pd.read_parquet(PANEL_PATH,
                              columns=["open_time", "symbol", "xs_alpha_dispersion_48b",
                                       "funding_rate", "btc_ret_288b",
                                       "btc_realized_vol_1d"])
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)

    prod = pd.read_csv(PROD_CSV)
    prod["time"] = pd.to_datetime(prod["time"], utc=True)
    sparse = pd.read_csv(SPARSE_CSV)
    sparse["time"] = pd.to_datetime(sparse["time"], utc=True)
    print(f"  prod: {len(prod):,} cycles; sparse: {len(sparse):,} cycles", flush=True)

    # Merge by time
    merged = prod[["time", "fold", "net_bps"]].rename(columns={"net_bps": "prod_pnl"})
    merged = merged.merge(sparse[["time", "net_bps"]].rename(columns={"net_bps": "sparse_pnl"}),
                            on="time", how="inner")
    merged["flat_pnl"] = 0.0
    # Label: argmax of (prod, sparse, flat)
    merged["label"] = merged[["prod_pnl", "sparse_pnl", "flat_pnl"]].idxmax(axis=1)
    merged["label_int"] = merged["label"].map({"prod_pnl": 0, "sparse_pnl": 1, "flat_pnl": 2})
    print(f"  Label distribution:")
    print(merged["label"].value_counts().to_string(), flush=True)

    # Build universe
    listings = get_listings()
    panel_syms = set(panel["symbol"].unique())
    def eligibility_at(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}
    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::HORIZON]
    print(f"\n  Building universe...", flush=True)
    universe = build_rolling_ic_universe(apd, sampled_t, TOP_N, eligibility_at)

    # Compute meta-features per cycle
    meta = compute_meta_features(apd, panel, universe)
    meta = meta[meta["valid"]].drop(columns=["valid"])
    meta.to_csv(OUT / "meta_features.csv", index=False)
    print(f"\n  meta-features: {len(meta)} valid cycles, {meta.shape[1]} cols", flush=True)
    print(f"  Coverage: " + ", ".join(
        f"{c}={meta[c].notna().mean():.0%}"
        for c in ["pred_disp", "topK_overlap_prev", "rolling_IC_universe",
                  "BTC_30d_drawdown", "BTC_realized_vol", "xs_alpha_dispersion",
                  "funding_dispersion"]), flush=True)

    # Join meta + labels by time
    df = meta.merge(merged[["time", "fold", "prod_pnl", "sparse_pnl",
                              "flat_pnl", "label_int"]],
                      left_on="time", right_on="time", how="inner",
                      suffixes=("_meta", "_merged"))
    df = df.dropna(subset=["pred_disp", "rolling_IC_universe", "BTC_30d_drawdown",
                             "xs_alpha_dispersion", "funding_dispersion"])
    print(f"\n  Training set: {len(df):,} cycles after dropna", flush=True)

    feat_cols = ["pred_disp", "topK_overlap_prev", "rank_churn",
                  "rolling_IC_universe", "BTC_30d_drawdown", "BTC_realized_vol",
                  "xs_alpha_dispersion", "funding_dispersion"]
    feat_cols = [c for c in feat_cols if c in df.columns and df[c].notna().mean() > 0.5]
    print(f"  Using features: {feat_cols}", flush=True)

    # Nested per-fold training
    import lightgbm as lgb
    nested_pred_modes = []
    chose_log = []
    fold_col = "fold_meta" if "fold_meta" in df.columns else "fold"
    for f in OOS_FOLDS:
        past = df[df[fold_col] < f]
        cur = df[df[fold_col] == f]
        if len(past) < 50:
            # Default: production
            cur = cur.copy()
            cur["pred_mode"] = 0
            chose_log.append({"fold": f, "n_train": len(past), "method": "default_prod"})
        else:
            Xt = past[feat_cols].fillna(past[feat_cols].median()).to_numpy(np.float32)
            yt = past["label_int"].to_numpy(np.int32)
            Xc = cur[feat_cols].fillna(past[feat_cols].median()).to_numpy(np.float32)
            try:
                model = lgb.LGBMClassifier(
                    objective="multiclass", num_class=3,
                    n_estimators=200, learning_rate=0.05,
                    max_depth=4, num_leaves=15,
                    min_child_samples=20, reg_alpha=1.0, reg_lambda=1.0,
                    random_state=42, verbose=-1,
                )
                model.fit(Xt, yt)
                pred = model.predict(Xc)
                cur = cur.copy()
                cur["pred_mode"] = pred
                # Importance
                imp = dict(zip(feat_cols, model.feature_importances_))
                chose_log.append({"fold": f, "n_train": len(past),
                                    "method": "lgbm", **imp})
            except Exception as e:
                cur = cur.copy()
                cur["pred_mode"] = 0
                chose_log.append({"fold": f, "n_train": len(past),
                                    "method": f"failed: {e}"})
        nested_pred_modes.append(cur)
    pd.DataFrame(chose_log).to_csv(OUT / "per_fold_training.csv", index=False)
    nested_df = pd.concat(nested_pred_modes, ignore_index=True)

    # Apply predicted mode to extract net_bps
    nested_df["nested_net_bps"] = nested_df.apply(
        lambda row: row["prod_pnl"] if row["pred_mode"] == 0
        else row["sparse_pnl"] if row["pred_mode"] == 1
        else 0.0, axis=1
    )
    nested_df.to_csv(OUT / "per_cycle_nested.csv", index=False)

    # Compute results
    print(f"\n--- Nested mode-gate result ---", flush=True)
    print(f"  Mode distribution: prod={int((nested_df['pred_mode']==0).sum())} "
          f"sparse={int((nested_df['pred_mode']==1).sum())} "
          f"flat={int((nested_df['pred_mode']==2).sum())}", flush=True)
    sh_nested = _sharpe(nested_df["nested_net_bps"].to_numpy())
    sh_prod_baseline = _sharpe(nested_df["prod_pnl"].to_numpy())
    sh_sparse_baseline = _sharpe(nested_df["sparse_pnl"].to_numpy())
    print(f"  Production baseline (always prod):     Sharpe={sh_prod_baseline:+.2f}, "
          f"PnL={nested_df['prod_pnl'].sum():+.0f}", flush=True)
    print(f"  Sparse baseline (always sparse=0.25):   Sharpe={sh_sparse_baseline:+.2f}, "
          f"PnL={nested_df['sparse_pnl'].sum():+.0f}", flush=True)
    print(f"  Meta-gate (nested):                     Sharpe={sh_nested:+.2f}, "
          f"PnL={nested_df['nested_net_bps'].sum():+.0f}", flush=True)
    lift_vs_prod = sh_nested - sh_prod_baseline
    print(f"  Lift vs production:                     {lift_vs_prod:+.2f}", flush=True)

    # Per-fold
    print(f"\n  Per-fold nested Sharpe:")
    n_fold_pos = 0
    for f in OOS_FOLDS:
        fd = nested_df[nested_df[fold_col] == f]
        if len(fd) < 3: continue
        sh_nf = _sharpe(fd["nested_net_bps"].to_numpy())
        sh_pf = _sharpe(fd["prod_pnl"].to_numpy())
        lift = sh_nf - sh_pf
        if lift > 0: n_fold_pos += 1
        print(f"    fold {f}: meta={sh_nf:+.2f}  prod={sh_pf:+.2f}  lift={lift:+.2f}", flush=True)
    print(f"  Folds where meta beats production: {n_fold_pos}/9", flush=True)

    # Matched mode-timing placebo
    print(f"\n--- Matched mode-timing placebo ({N_PLACEBO_SEEDS} seeds) ---", flush=True)
    real_modes = nested_df["pred_mode"].to_numpy()
    # Distribution of modes used
    mode_dist = np.bincount(real_modes, minlength=3) / len(real_modes)
    print(f"  Real mode rates: prod={mode_dist[0]:.2%}, sparse={mode_dist[1]:.2%}, "
          f"flat={mode_dist[2]:.2%}", flush=True)
    placebo_sh = []
    for seed in range(N_PLACEBO_SEEDS):
        rng = np.random.RandomState(seed)
        rand_modes = rng.choice(3, size=len(nested_df), p=mode_dist)
        plac_net = np.where(rand_modes == 0, nested_df["prod_pnl"].to_numpy(),
                              np.where(rand_modes == 1, nested_df["sparse_pnl"].to_numpy(), 0.0))
        placebo_sh.append(_sharpe(plac_net))
    p_sh = np.array(placebo_sh)
    p95 = float(np.percentile(p_sh, 95))
    rank = float((p_sh < sh_nested).mean() * 100)
    print(f"  Placebo: mean={p_sh.mean():+.2f}, p50={np.percentile(p_sh,50):+.2f}, "
          f"p95={p95:+.2f}, max={p_sh.max():+.2f}", flush=True)
    print(f"  Meta-gate ranks p{rank:.0f} vs mode-timing placebo  "
          f"beats_p95={'PASS' if sh_nested > p95 else 'FAIL'}", flush=True)
    pd.DataFrame({"seed": range(N_PLACEBO_SEEDS), "sharpe": p_sh}).to_csv(
        OUT / "mode_timing_placebo.csv", index=False)

    # Final verdict
    print(f"\n=== Test 1 verdict ===", flush=True)
    pass_lift = lift_vs_prod > 0.30
    pass_folds = n_fold_pos >= 6
    pass_placebo = sh_nested > p95
    print(f"  Lift vs prod >+0.30:       {'PASS' if pass_lift else 'FAIL'} ({lift_vs_prod:+.2f})",
          flush=True)
    print(f"  ≥6/9 folds improve:        {'PASS' if pass_folds else 'FAIL'} ({n_fold_pos}/9)",
          flush=True)
    print(f"  Beats placebo p95:         {'PASS' if pass_placebo else 'FAIL'}",
          flush=True)
    if pass_lift and pass_folds and pass_placebo:
        print(f"  → ADOPT meta-gate", flush=True)
    else:
        print(f"  → NOT ADOPTED", flush=True)
    print(f"\n  saved → {OUT}", flush=True)


if __name__ == "__main__":
    main()
