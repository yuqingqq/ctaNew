"""Step 38: 24h target + 24h-ALIGNED features + V3.1 sleeve.

Same as Step 37 but with horizon-aligned feature set:
  Drop 5 short-horizon: atr_pct, vwap_slope_96, idio_vol_to_btc_1h,
                          return_8h_orth, vol_zscore_4h_over_7d
  Add  5 longer-horizon: return_3d, return_5d (compounded from return_1d),
                          btc_realized_vol_1d, btc_ret_288b,
                          log_dollar_volume_7d

Total: 22 features (net zero from V2).

Training target: 24h forward β-residual.
Execution: V3.1 6-sleeve × 24h hold @ 4h cadence (production).
"""
from __future__ import annotations
import sys, time, importlib.util, warnings
from pathlib import Path
from collections import deque, defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LinearRegression

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("psl", REPO / "scripts/phase_ah_sleeve.py")
psl = importlib.util.module_from_spec(spec); spec.loader.exec_module(psl)
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice
from ml.research.alpha_v4_xs import block_bootstrap_ci

TARGETS_4H = REPO / "linear_model/data/targets.parquet"
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
PANEL_BTC = REPO / "outputs/vBTC_features_btc_only/panel_btc_only_clean.parquet"
V3_PANEL = REPO / "outputs/vBTC_features_btc_v3/panel_v3_5m.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
OUT = REPO / "linear_model/results/step38_24h_aligned"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- 22 horizon-aligned features ----------
# Kept R3_BTC base (1d-7d aligned, 12 features):
FRAME_NEUTRAL_KEEP = ["return_1d", "obv_z_1d", "bars_since_high_xs_rank",
                       "funding_rate", "funding_rate_z_7d",
                       "funding_rate_1d_change", "corr_to_btc_1d",
                       "beta_to_btc_change_5d"]
BTC_KEEP = ["dom_btc_z_1d", "dom_btc_change_288b", "corr_to_btc_change_3d",
            "idio_vol_to_btc_1d"]

# New longer-horizon features (5):
NEW_BUILD = ["return_3d", "return_5d"]  # compounded from return_1d
NEW_FROM_BASE = ["btc_realized_vol_1d", "btc_ret_288b"]
NEW_FROM_V3 = ["log_dollar_volume_7d"]

# Squared U-shape (5):
KEEP_USHAPE_R3 = ["beta_to_btc_change_5d", "corr_to_btc_1d", "return_1d"]
BTC_USHAPE = ["dom_btc_change_288b", "corr_to_btc_change_3d"]

# Preprocessing buckets (only one feature is per-symbol now; funding rate is funding-only)
# Note: none of the new features are in HEAVY_TAIL (verified empirically: kurt < 16)
HEAVY_TAIL = {"funding_rate", "funding_rate_1d_change", "funding_rate_z_7d"}
PER_SYMBOL_Z = {"funding_rate", "funding_rate_z_7d", "funding_rate_1d_change"}

ALPHAS = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]
SEEDS = (42, 1337, 7, 19, 2718)
AUTO_THRESH = 0.5
ALL_FOLDS = list(range(10))
OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
TRAILING_IC_DAYS = 90
HOLD_BARS = 288
H24_STEPS = 6
WINSORIZE_SIGMA = 5.0


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(psl.CYCLES_PER_YEAR))


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


# ---------- Target ----------

def build_alpha_24h(panel):
    panel = panel.sort_values(["symbol", "open_time"]).reset_index(drop=True).copy()
    alpha_24h = np.full(len(panel), np.nan, dtype=np.float64)
    for sym, g in panel.groupby("symbol"):
        idx = g.index.to_numpy()
        a = g["alpha_beta"].to_numpy()
        if len(a) < H24_STEPS + 1:
            continue
        roll = np.full(len(a), np.nan)
        for i in range(len(a) - H24_STEPS + 1):
            window = a[i:i + H24_STEPS]
            if not np.isnan(window).any():
                roll[i] = window.sum()
        alpha_24h[idx] = roll
    panel["alpha_beta_24h"] = alpha_24h
    return panel


def build_target_z_24h(panel, fold0_train_idx):
    train_panel = panel.loc[fold0_train_idx]
    sigma_per_sym = train_panel.groupby("symbol")["alpha_beta_24h"].std()
    median_sigma = float(sigma_per_sym.dropna().median())
    sigma_per_sym = sigma_per_sym.fillna(median_sigma)
    panel = panel.copy()
    panel["sigma_idio_24h"] = panel["symbol"].map(sigma_per_sym).astype("float32")
    panel.loc[panel["sigma_idio_24h"].isna(), "sigma_idio_24h"] = median_sigma
    raw_z = panel["alpha_beta_24h"] / panel["sigma_idio_24h"]
    panel["target_z_24h"] = raw_z.clip(lower=-WINSORIZE_SIGMA,
                                          upper=WINSORIZE_SIGMA).astype("float32")
    return panel, float(median_sigma)


# ---------- Feature engineering ----------

def build_return_3d_5d(panel):
    """Compound 3 / 5 consecutive trailing 24h returns from return_1d at 4h panel cadence.

    At 4h cadence, 1 day = 6 4h-cycles, so:
      return_3d(t) = (1+r1d(t)) * (1+r1d(t-6)) * (1+r1d(t-12)) - 1
      return_5d(t) = ... 5 windows
    """
    panel = panel.sort_values(["symbol", "open_time"]).reset_index(drop=True).copy()
    out_3d = np.full(len(panel), np.nan, dtype=np.float64)
    out_5d = np.full(len(panel), np.nan, dtype=np.float64)
    for sym, g in panel.groupby("symbol"):
        idx = g.index.to_numpy()
        r1d = g["return_1d"].to_numpy()
        # Sliding 3d window: indices [i-12, i-6, i] in the symbol's sorted rows
        for i in range(12, len(r1d)):
            window = np.array([r1d[i], r1d[i-6], r1d[i-12]])
            if not np.isnan(window).any():
                out_3d[idx[i]] = float(np.prod(1.0 + window) - 1.0)
        for i in range(24, len(r1d)):
            window = np.array([r1d[i], r1d[i-6], r1d[i-12], r1d[i-18], r1d[i-24]])
            if not np.isnan(window).any():
                out_5d[idx[i]] = float(np.prod(1.0 + window) - 1.0)
    panel["return_3d"] = out_3d
    panel["return_5d"] = out_5d
    return panel


# ---------- Preprocessing (same as Step 34) ----------

def winsorize_zscore(s, train_s, p_lo=0.01, p_hi=0.99):
    s_train = train_s.dropna()
    lo, hi = s_train.quantile(p_lo), s_train.quantile(p_hi)
    s_w = s_train.clip(lower=lo, upper=hi)
    mu, sd = s_w.mean(), s_w.std()
    if sd < 1e-8: sd = 1.0
    return ((s.clip(lower=lo, upper=hi) - mu) / sd).astype("float32")


def rank_transform(panel_full, train_s):
    train_vals = train_s.dropna().sort_values().values
    n_train = len(train_vals)
    if n_train < 100:
        return pd.Series(np.zeros(len(panel_full), dtype=np.float32),
                         index=panel_full.index)
    raw = panel_full.values
    out = np.zeros(len(raw), dtype=np.float32)
    mask = np.isfinite(raw)
    if mask.any():
        ranks = np.searchsorted(train_vals, raw[mask]) / n_train - 0.5
        out[mask] = ranks.astype(np.float32)
    return pd.Series(out, index=panel_full.index)


def per_symbol_rank(panel, feat_name, train_mask):
    out = np.zeros(len(panel), dtype=np.float32)
    idx_pos = {idx: pos for pos, idx in enumerate(panel.index)}
    train_all = panel.loc[train_mask, feat_name].dropna().sort_values().values
    n_all = len(train_all)
    for sym, g in panel.groupby("symbol"):
        idx = g.index
        sym_mask = train_mask & (panel["symbol"] == sym)
        train_vals = panel.loc[sym_mask, feat_name].dropna().sort_values().values
        n_train = len(train_vals)
        if n_train < 50:
            if n_all < 100:
                continue
            train_vals = train_all
            n_train = n_all
        vals = panel.loc[idx, feat_name].values
        mask_v = np.isfinite(vals)
        if not mask_v.any():
            continue
        ranks = np.searchsorted(train_vals, vals[mask_v]) / n_train - 0.5
        idx_list = list(idx)
        finite_positions = np.where(mask_v)[0]
        for i, fp in enumerate(finite_positions):
            out[idx_pos[idx_list[fp]]] = ranks[i]
    return pd.Series(out, index=panel.index, dtype="float32")


def restandardize(col, train_mask):
    train_vals = col[train_mask].replace([np.inf, -np.inf], np.nan).dropna()
    if len(train_vals) < 100:
        return col.astype("float32")
    mu, sd = train_vals.mean(), train_vals.std()
    if sd < 1e-8: sd = 1.0
    return ((col - mu) / sd).astype("float32")


def build_features(panel, train_mask):
    train_panel = panel[train_mask]
    X = pd.DataFrame({"symbol": panel["symbol"], "open_time": panel["open_time"],
                      "alpha_beta_24h": panel["alpha_beta_24h"],
                      "target_z_24h": panel["target_z_24h"],
                      "autocorr_pctile_7d": panel["autocorr_pctile_7d"]})
    all_base = (FRAME_NEUTRAL_KEEP + BTC_KEEP
                + NEW_BUILD + NEW_FROM_BASE + NEW_FROM_V3)
    for f in all_base:
        if f not in panel.columns:
            print(f"  WARNING: feature {f} not in panel, skipping")
            continue
        if f in PER_SYMBOL_Z:
            ranked = per_symbol_rank(panel, f, train_mask)
            X[f] = restandardize(ranked, train_mask)
        elif f in HEAVY_TAIL:
            ranked = rank_transform(panel[f], train_panel[f])
            X[f] = restandardize(ranked, train_mask)
        else:
            X[f] = winsorize_zscore(panel[f], train_panel[f])

    for f in KEEP_USHAPE_R3:
        base = winsorize_zscore(panel[f], train_panel[f])
        X[f + "_sq"] = (base ** 2).astype("float32")
    for f in BTC_USHAPE:
        base = winsorize_zscore(panel[f], train_panel[f])
        X[f + "_sq"] = (base ** 2).astype("float32")

    feat_cols = [c for c in X.columns if c not in
                 ("symbol","open_time","alpha_beta_24h","target_z_24h","autocorr_pctile_7d")]
    X[feat_cols] = X[feat_cols].fillna(0)
    return X, feat_cols


# ---------- Training + sleeve aggregation ----------

def train_ridge_24h(panel_x, folds_all, feat_cols):
    all_preds = []
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t_fold = time.time()
        train_, cal, test = _slice(panel_x, folds_all[fid])
        tr = train_[train_["autocorr_pctile_7d"] >= AUTO_THRESH].dropna(subset=["target_z_24h"])
        te = test.dropna(subset=["target_z_24h"]).copy()
        if len(tr) < 1000 or len(te) < 100: continue
        Xt = tr[feat_cols].to_numpy(np.float32)
        Xte = te[feat_cols].to_numpy(np.float32)
        yt = tr["target_z_24h"].to_numpy(np.float32)
        mt = ~np.isnan(yt)
        fold_preds = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, mt.sum(), size=mt.sum())
            m = RidgeCV(alphas=ALPHAS, scoring="r2", cv=None, fit_intercept=True)
            m.fit(Xt[mt][idx], yt[mt][idx])
            fold_preds.append(m.predict(Xte).astype(np.float32))
        pred = np.mean(fold_preds, axis=0)
        df_pred = te[["symbol","open_time","alpha_beta_24h"]].copy()
        df_pred["pred_z"] = pred
        df_pred["fold"] = fid
        all_preds.append(df_pred)
        cyc_ic = df_pred.dropna(subset=["alpha_beta_24h"]).groupby("open_time").apply(
            lambda g: g["pred_z"].rank().corr(g["alpha_beta_24h"].rank())
            if len(g) >= 5 else np.nan).dropna()
        print(f"    fold {fid}: IC_24h={cyc_ic.mean():+.4f}, {time.time()-t_fold:.0f}s",
              flush=True)
    return pd.concat(all_preds, ignore_index=True).sort_values(["open_time","symbol"])


def compute_trailing_ic_24h(apd, sampled_t, win_days=90):
    apd_s = apd[apd["open_time"].isin(set(sampled_t))].sort_values(
        ["symbol","open_time"]).reset_index(drop=True)
    win_cycles = win_days * 6
    rows = []
    for sym, g in apd_s.groupby("symbol"):
        g = g.sort_values("open_time").reset_index(drop=True)
        pred = g["pred_z"].to_numpy(); alpha = g["alpha_beta_24h"].to_numpy()
        n = len(g)
        ics = np.full(n, np.nan)
        for i in range(50, n):
            lo = max(0, i - win_cycles)
            p, a = pred[lo:i], alpha[lo:i]
            mask = ~np.isnan(p) & ~np.isnan(a)
            if mask.sum() < 50: continue
            pr = pd.Series(p[mask]).rank().to_numpy()
            ar = pd.Series(a[mask]).rank().to_numpy()
            if pr.std() < 1e-6 or ar.std() < 1e-6: continue
            ics[i] = np.corrcoef(pr, ar)[0,1]
        for j, t in enumerate(g["open_time"]):
            rows.append({"symbol":sym, "open_time":t, "trail_ic":ics[j]})
    return pd.DataFrame(rows).fillna(0)


def aggregate_hold_through(records, alpha_wide_4h):
    sleeve_queue = deque(maxlen=psl.N_SLEEVES)
    prev_weights = {}
    bar_freq = pd.Timedelta(minutes=5)
    rows = []
    for _, rec in records.iterrows():
        t = rec["time"]; fold = rec["fold"]
        if rec["traded"]:
            sleeve_queue.append({"entry_time":t, "longs":list(rec["long_basket"]),
                                  "shorts":list(rec["short_basket"])})
        max_age = HOLD_BARS * bar_freq
        sleeve_queue = deque(
            [s for s in sleeve_queue if (t - s["entry_time"]) < max_age],
            maxlen=psl.N_SLEEVES)
        tw = defaultdict(float)
        sw = 1.0 / psl.N_SLEEVES
        for sl in sleeve_queue:
            nL, nS = len(sl["longs"]), len(sl["shorts"])
            if nL == 0 or nS == 0: continue
            for s in sl["longs"]: tw[s] += sw * (1.0/nL)
            for s in sl["shorts"]: tw[s] -= sw * (1.0/nS)
        gross = 0.0
        if t in alpha_wide_4h.index:
            a = alpha_wide_4h.loc[t]
            for sym, w in prev_weights.items():
                if sym in a.index and not pd.isna(a[sym]):
                    gross += w * a[sym] * 1e4
        syms = set(tw.keys()) | set(prev_weights.keys())
        abs_d = sum(abs(tw.get(s,0)-prev_weights.get(s,0)) for s in syms)
        cost = abs_d * psl.COST_PER_UNIT_ABS_DELTA
        rows.append({"time":t,"fold":fold,"gross_pnl_bps":gross,"cost_bps":cost,
                     "net_pnl_bps":gross-cost,"turnover":abs_d})
        prev_weights = dict(tw)
    return pd.DataFrame(rows)


# ---------- Main ----------

def main():
    print("=" * 100, flush=True)
    print("  STEP 38: SLEEVE + 24h TARGET + 24h-ALIGNED FEATURES", flush=True)
    print("=" * 100, flush=True)
    print("  Features (22): 12 R3_BTC base (1d-7d), 5 new long, 5 squared", flush=True)
    print("  Dropped from V2: atr_pct, vwap_slope_96, idio_vol_to_btc_1h,", flush=True)
    print("                   return_8h_orth, vol_zscore_4h_over_7d", flush=True)
    print("  Added: return_3d, return_5d (compound from return_1d),", flush=True)
    print("         btc_realized_vol_1d, btc_ret_288b, log_dollar_volume_7d", flush=True)
    print()
    t0 = time.time()
    listings = get_listings()

    # Load panel + new features
    print("Loading panels...", flush=True)
    tgt_ref = pd.read_parquet(TARGETS_4H,
                                columns=["symbol","open_time","alpha_beta",
                                          "autocorr_pctile_7d"])
    tgt_ref["open_time"] = pd.to_datetime(tgt_ref["open_time"], utc=True)
    base_cols = list(set(FRAME_NEUTRAL_KEEP + KEEP_USHAPE_R3 + NEW_FROM_BASE))
    base = pd.read_parquet(PANEL, columns=["symbol","open_time"] + base_cols)
    base["open_time"] = pd.to_datetime(base["open_time"], utc=True)
    btc_panel = pd.read_parquet(PANEL_BTC, columns=["symbol","open_time"] + BTC_KEEP)
    btc_panel["open_time"] = pd.to_datetime(btc_panel["open_time"], utc=True)
    v3_panel = pd.read_parquet(V3_PANEL,
                                  columns=["symbol","open_time"] + NEW_FROM_V3)
    v3_panel["open_time"] = pd.to_datetime(v3_panel["open_time"], utc=True)
    panel = tgt_ref.merge(base, on=["symbol","open_time"], how="left")
    panel = panel.merge(btc_panel, on=["symbol","open_time"], how="left")
    panel = panel.merge(v3_panel, on=["symbol","open_time"], how="left")
    print(f"Panel: {len(panel):,} rows", flush=True)

    # Build α_β_24h
    print("Building α_β_24h...", flush=True)
    panel = build_alpha_24h(panel)
    print(f"  α_β_24h coverage: {panel['alpha_beta_24h'].notna().sum():,}", flush=True)

    folds_all = _multi_oos_splits(panel)
    fold0_train_idx = _slice(panel, folds_all[0])[0].index
    panel, median_sigma_24h = build_target_z_24h(panel, fold0_train_idx)
    print(f"Built target_z_24h (median σ_idio_24h = {median_sigma_24h:.4f})", flush=True)

    # Build return_3d, return_5d
    print("Building return_3d, return_5d via compounding...", flush=True)
    panel = build_return_3d_5d(panel)
    print(f"  return_3d coverage: {panel['return_3d'].notna().sum():,}", flush=True)
    print(f"  return_5d coverage: {panel['return_5d'].notna().sum():,}", flush=True)
    print(f"  return_3d std: {panel['return_3d'].std():.4f}  "
          f"vs return_1d std: {panel['return_1d'].std():.4f}", flush=True)

    train_mask = panel["open_time"].between(
        _slice(panel, folds_all[0])[0].open_time.min(),
        _slice(panel, folds_all[0])[0].open_time.max())

    panel_syms = set(panel["symbol"].unique())
    for s, t in panel.groupby("symbol")["open_time"].min().items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t

    print("\nBuilding feature matrix...", flush=True)
    X, feat_cols = build_features(panel, train_mask)
    print(f"  Features: {len(feat_cols)}", flush=True)
    for f in feat_cols:
        std = X[f].std()
        print(f"    {f:35s} std={std:.3f}")

    panel_x = panel[["symbol","open_time","alpha_beta_24h","target_z_24h",
                       "autocorr_pctile_7d"]].merge(
        X.drop(columns=["alpha_beta_24h","target_z_24h","autocorr_pctile_7d"]),
        on=["symbol","open_time"], how="left")

    print("\nTraining Ridge on 24h target...", flush=True)
    apd = train_ridge_24h(panel_x, folds_all, feat_cols)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)

    # Merge 4h α_β for MTM
    a4h = tgt_ref[["symbol","open_time","alpha_beta"]].copy()
    apd = apd.merge(a4h, on=["symbol","open_time"], how="left")
    apd["alpha_A"] = apd["alpha_beta"]
    extra = pd.read_parquet(PANEL,
                              columns=["symbol","open_time","return_pct","exit_time"])
    extra["open_time"] = pd.to_datetime(extra["open_time"], utc=True)
    extra["exit_time"] = pd.to_datetime(extra["exit_time"], utc=True)
    apd = apd.merge(extra, on=["symbol","open_time"], how="left")

    cyc_ic = apd.dropna(subset=["alpha_beta_24h"]).groupby("open_time").apply(
        lambda g: g["pred_z"].rank().corr(g["alpha_beta_24h"].rank())
        if len(g) >= 5 else np.nan).dropna()
    overall_ic = float(cyc_ic.mean())
    print(f"\nOverall IC (pred_z vs α_β_24h): {overall_ic:+.4f}", flush=True)

    def elig_pit(b):
        ts = b if isinstance(b, pd.Timestamp) else pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}
    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]
    df_ic = compute_trailing_ic_24h(apd, sampled_t, TRAILING_IC_DAYS)
    apd_full = apd.merge(df_ic, on=["symbol","open_time"], how="left")
    apd_full["trail_ic"] = apd_full["trail_ic"].fillna(0)
    apd_full["pred_B"] = apd_full["pred_z"] * apd_full["trail_ic"]
    apd_full["pred"] = apd_full["pred_z"]
    universe = psl.build_rolling_ic_universe(apd_full, sampled_t, psl.TOP_N, elig_pit)
    alpha_wide_4h = apd_full.pivot_table(index="open_time", columns="symbol",
                                          values="alpha_A", aggfunc="first").sort_index()

    apd_full.to_parquet(OUT / "predictions.parquet", index=False)

    print(f"\n{'='*100}", flush=True)
    print(f"  V3.1 sleeve overlay on 24h-aligned + 24h-target Ridge", flush=True)
    print(f"{'='*100}", flush=True)
    results = []
    for sub, col in [("A","pred_z"), ("B","pred_B")]:
        apd_v = apd_full.copy(); apd_v["pred"] = apd_v[col]
        records = psl.run_production_protocol_save_sleeves(apd_v, universe)
        df_v = aggregate_hold_through(records, alpha_wide_4h)
        net = df_v["net_pnl_bps"].to_numpy()
        sh = _sharpe(net)
        sh_lo, sh_hi = block_bootstrap_ci(net, statistic=_sharpe,
                                            block_size=7, n_boot=1000)[1:]
        n_traded = (df_v["gross_pnl_bps"] != 0).sum()
        df_v.to_csv(OUT / f"per_cycle_{sub}.csv", index=False)
        sub_label = "baseline (pred_z)" if sub == "A" else "IC_signed (pred_B)"
        print(f"  {sub} {sub_label}: Sharpe={sh:+.2f} [{sh_lo:+.2f},{sh_hi:+.2f}]  "
              f"folds+={folds_positive(df_v)}/9  gross={df_v['gross_pnl_bps'].mean():+.2f}  "
              f"traded={n_traded}/{len(df_v)}", flush=True)
        results.append({"sub":sub, "sharpe":sh, "sh_lo":sh_lo, "sh_hi":sh_hi,
                         "folds_pos":folds_positive(df_v),
                         "overall_ic":overall_ic, "n_traded":n_traded})

    sh_B = results[1]["sharpe"]
    df_v_B = pd.read_csv(OUT / "per_cycle_B.csv")
    print(f"\n  LOFO on B (Sharpe = {sh_B:+.2f}):", flush=True)
    lofo_rows = []
    for excl in range(1, 10):
        rem = df_v_B[df_v_B["fold"] != excl]["net_pnl_bps"].to_numpy()
        sh_rem = _sharpe(rem)
        d = sh_rem - sh_B
        flag = "  ← drives" if d < -0.4 else ""
        print(f"    excl {excl}: {sh_rem:+.2f} (Δ {d:+.2f}){flag}", flush=True)
        lofo_rows.append({"excl":excl, "sharpe":sh_rem, "delta":d})
    pd.DataFrame(lofo_rows).to_csv(OUT / "lofo.csv", index=False)
    pd.DataFrame(results).to_csv(OUT / "summary.csv", index=False)

    print(f"\n{'='*100}", flush=True)
    print(f"  COMPARISON LADDER", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"  4h target + V2 (22 ft) + V3.1 sleeve   (Step 34/35): Sharpe +2.19", flush=True)
    print(f"  24h target + V2 (22 ft) + V3.1 sleeve  (Step 37):    Sharpe +1.50", flush=True)
    print(f"  24h target + 24h-aligned (22 ft) + sleeve (THIS):    Sharpe {sh_B:+.2f}", flush=True)
    print(f"  4h target + V2 (22 ft) raw 4h cycle    (Step 36):    Sharpe -7.45", flush=True)
    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
