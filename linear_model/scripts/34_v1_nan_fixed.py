"""Step 34: V1 with NaN-safe rank transforms + re-standardization.

Reviewer findings fixed:
  1. NaN bug — np.searchsorted maps NaN to +0.5 (max rank). Mask finite
     values before ranking; NaN positions get 0 (median rank).
  2. Scale mismatch — rank features have std ~0.29 vs z-score std ~1.0.
     Re-z-score rank-transformed columns using fold-0 train stats so
     Ridge regularization pressure is uniform across feature families.
  3. Artifacts — save per-cycle PnL, predictions, fold records, LOFO
     tables to results/step34_v1_fixed/ for independent audit.

Re-runs V0 (standard), V1 (proper + fixed + re-std), V2 (V1 + short_h).
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

TARGETS = REPO / "linear_model/data/targets.parquet"
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
PANEL_BTC = REPO / "outputs/vBTC_features_btc_only/panel_btc_only_clean.parquet"
STEP29 = REPO / "linear_model/results/step29_features.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
OUT_DIR = REPO / "linear_model/results/step34_v1_fixed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEAVY_TAIL = {"funding_rate", "funding_rate_1d_change", "vwap_slope_96",
              "idio_max_abs_12b", "idio_vol_to_btc_1h", "funding_rate_z_7d"}
PER_SYMBOL_Z = {"funding_rate", "funding_rate_z_7d", "funding_rate_1d_change"}

FRAME_NEUTRAL = ["return_1d","atr_pct","obv_z_1d","vwap_slope_96",
                 "bars_since_high_xs_rank","funding_rate","funding_rate_z_7d",
                 "funding_rate_1d_change",
                 "corr_to_btc_1d", "idio_vol_to_btc_1h", "beta_to_btc_change_5d"]
KEEP_USHAPE_R3 = ["beta_to_btc_change_5d", "corr_to_btc_1d", "return_1d"]
BTC_KEEP = ["dom_btc_z_1d", "dom_btc_change_288b", "corr_to_btc_change_3d",
            "idio_vol_to_btc_1d"]
BTC_USHAPE = ["dom_btc_change_288b", "corr_to_btc_change_3d"]
SHORT_HORIZON = ["return_8h", "vol_zscore_4h_over_7d"]

ALPHAS = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]
SEEDS = (42, 1337, 7, 19, 2718)
AUTO_THRESH = 0.5
ALL_FOLDS = list(range(10))
OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
CAPITAL = 100.0
TRAILING_IC_DAYS = 90
HOLD_BARS = 288


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


def winsorize_zscore(s, train_s, p_lo=0.01, p_hi=0.99):
    s_train = train_s.dropna()
    lo, hi = s_train.quantile(p_lo), s_train.quantile(p_hi)
    s_w = s_train.clip(lower=lo, upper=hi)
    mu, sd = s_w.mean(), s_w.std()
    if sd < 1e-8: sd = 1.0
    return ((s.clip(lower=lo, upper=hi) - mu) / sd).astype("float32")


# ---------- FIXED rank transforms (NaN-safe) ----------

def rank_transform(panel_full, train_s, feat_name):
    """NaN-SAFE PIT percentile rank using fold-0 training distribution.
    NaN inputs → 0 (median rank) explicitly. Finite values get ranked."""
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
    """NaN-SAFE per-symbol percentile rank using fold-0 train distribution."""
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
        # Place ranked values; NaN positions stay 0
        finite_positions = np.where(mask_v)[0]
        for i, fp in enumerate(finite_positions):
            out[idx_pos[idx_list[fp]]] = ranks[i]
    return pd.Series(out, index=panel.index, dtype="float32")


def restandardize(col, train_mask):
    """Z-score using fold-0 train stats. Used to give rank features unit std."""
    train_vals = col[train_mask].replace([np.inf, -np.inf], np.nan).dropna()
    if len(train_vals) < 100:
        return col.astype("float32")
    mu, sd = train_vals.mean(), train_vals.std()
    if sd < 1e-8: sd = 1.0
    return ((col - mu) / sd).astype("float32")


def orthogonalize(target, predictor, train_mask):
    train_idx = train_mask & target.notna() & predictor.notna()
    if train_idx.sum() < 1000:
        return target
    lr = LinearRegression()
    lr.fit(predictor[train_idx].values.reshape(-1,1), target[train_idx].values)
    beta = lr.coef_[0]
    return (target - beta * predictor).astype("float32")


def preprocess_features(panel, train_mask, mode="standard"):
    """mode=standard: winsorize+z everything
       mode=proper:   rank-transform heavy-tail (+ re-std), per-sym-rank funding (+ re-std)"""
    train_panel = panel[train_mask]
    X = pd.DataFrame({"symbol": panel["symbol"], "open_time": panel["open_time"],
                      "alpha_beta": panel["alpha_beta"],
                      "target_z": panel["target_z"],
                      "autocorr_pctile_7d": panel["autocorr_pctile_7d"]})

    all_features = list(set(FRAME_NEUTRAL + BTC_KEEP))
    for f in all_features:
        if f not in panel.columns: continue
        if mode == "standard":
            X[f] = winsorize_zscore(panel[f], train_panel[f])
        elif mode == "proper":
            if f in PER_SYMBOL_Z:
                ranked = per_symbol_rank(panel, f, train_mask)
                # Re-standardize the rank column to unit variance
                X[f] = restandardize(ranked, train_mask)
            elif f in HEAVY_TAIL:
                ranked = rank_transform(panel[f], train_panel[f], f)
                X[f] = restandardize(ranked, train_mask)
            else:
                X[f] = winsorize_zscore(panel[f], train_panel[f])

    for f in KEEP_USHAPE_R3:
        if mode == "proper" and f in HEAVY_TAIL:
            base = winsorize_zscore(panel[f], train_panel[f])
            X[f + "_sq"] = (base ** 2).astype("float32")
        else:
            X[f + "_sq"] = (X[f] ** 2).astype("float32")
    for f in BTC_USHAPE:
        if mode == "proper" and f in HEAVY_TAIL:
            base = winsorize_zscore(panel[f], train_panel[f])
            X[f + "_sq"] = (base ** 2).astype("float32")
        else:
            X[f + "_sq"] = (X[f] ** 2).astype("float32")

    feat_cols = [c for c in X.columns if c not in
                 ("symbol","open_time","alpha_beta","target_z","autocorr_pctile_7d")]
    X[feat_cols] = X[feat_cols].fillna(0)
    return X, feat_cols


def add_short_horizon(X, panel, train_mask):
    train_panel = panel[train_mask]
    if "return_8h" in panel.columns and "return_1d" in panel.columns:
        r8h_orth = orthogonalize(panel["return_8h"], panel["return_1d"], train_mask)
        X["return_8h_orth"] = winsorize_zscore(r8h_orth, r8h_orth[train_mask])
    if "vol_zscore_4h_over_7d" in panel.columns:
        X["vol_zscore_4h_over_7d"] = winsorize_zscore(
            panel["vol_zscore_4h_over_7d"],
            train_panel["vol_zscore_4h_over_7d"] if train_mask.any() else panel["vol_zscore_4h_over_7d"])
    new_cols = ["return_8h_orth", "vol_zscore_4h_over_7d"]
    for c in new_cols:
        if c in X.columns:
            X[c] = X[c].fillna(0)
    return X, [c for c in new_cols if c in X.columns]


def compute_trailing_ic(apd, sampled_t, win_days=90):
    apd_s = apd[apd["open_time"].isin(set(sampled_t))].sort_values(
        ["symbol","open_time"]).reset_index(drop=True)
    win_cycles = win_days * 6
    rows = []
    for sym, g in apd_s.groupby("symbol"):
        g = g.sort_values("open_time").reset_index(drop=True)
        pred = g["pred_z"].to_numpy(); alpha = g["alpha_beta"].to_numpy()
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


def aggregate_hold_through(records, alpha_wide):
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
        if t in alpha_wide.index:
            a = alpha_wide.loc[t]
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


def train(panel_x, folds_all, feat_cols):
    all_preds = []
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t_fold = time.time()
        train_, cal, test = _slice(panel_x, folds_all[fid])
        tr = train_[train_["autocorr_pctile_7d"] >= AUTO_THRESH].dropna(subset=["target_z"])
        te = test.dropna(subset=["target_z"]).copy()
        if len(tr) < 1000 or len(te) < 100: continue
        Xt = tr[feat_cols].to_numpy(np.float32)
        Xte = te[feat_cols].to_numpy(np.float32)
        yt = tr["target_z"].to_numpy(np.float32)
        mt = ~np.isnan(yt)
        fold_preds = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, mt.sum(), size=mt.sum())
            m = RidgeCV(alphas=ALPHAS, scoring="r2", cv=None, fit_intercept=True)
            m.fit(Xt[mt][idx], yt[mt][idx])
            fold_preds.append(m.predict(Xte).astype(np.float32))
        pred = np.mean(fold_preds, axis=0)
        df_pred = te[["symbol","open_time","alpha_beta"]].copy()
        df_pred["pred_z"] = pred
        df_pred["fold"] = fid
        all_preds.append(df_pred)
        cyc_ic = df_pred.dropna(subset=["alpha_beta"]).groupby("open_time").apply(
            lambda g: g["pred_z"].rank().corr(g["alpha_beta"].rank())
            if len(g) >= 5 else np.nan).dropna()
        print(f"    fold {fid}: IC={cyc_ic.mean():+.4f}, {time.time()-t_fold:.0f}s",
              flush=True)
    return pd.concat(all_preds, ignore_index=True).sort_values(["open_time","symbol"])


def run_variant(label, slug, panel, folds_all, feat_cols, listings, panel_syms):
    print(f"\n{'='*100}", flush=True)
    print(f"  {label}: {len(feat_cols)} features", flush=True)
    print(f"{'='*100}", flush=True)
    apd = train(panel, folds_all, feat_cols)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["alpha_A"] = apd["alpha_beta"]
    base = pd.read_parquet(PANEL, columns=["symbol","open_time","return_pct","exit_time"])
    base["open_time"] = pd.to_datetime(base["open_time"], utc=True)
    base["exit_time"] = pd.to_datetime(base["exit_time"], utc=True)
    apd = apd.merge(base, on=["symbol","open_time"], how="left")

    cyc_ic = apd.dropna(subset=["alpha_beta"]).groupby("open_time").apply(
        lambda g: g["pred_z"].rank().corr(g["alpha_beta"].rank())
        if len(g) >= 5 else np.nan).dropna()
    overall_ic = cyc_ic.mean()
    print(f"  Overall IC: {overall_ic:+.4f}", flush=True)

    def elig_pit(b):
        ts = b if isinstance(b, pd.Timestamp) else pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}
    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]
    df_ic = compute_trailing_ic(apd, sampled_t, TRAILING_IC_DAYS)
    apd_full = apd.merge(df_ic, on=["symbol","open_time"], how="left")
    apd_full["trail_ic"] = apd_full["trail_ic"].fillna(0)
    apd_full["pred_B"] = apd_full["pred_z"] * apd_full["trail_ic"]
    apd_full["pred"] = apd_full["pred_z"]
    universe = psl.build_rolling_ic_universe(apd_full, sampled_t, psl.TOP_N, elig_pit)
    alpha_wide = apd_full.pivot_table(index="open_time", columns="symbol",
                                        values="alpha_A", aggfunc="first").sort_index()

    # Save predictions + trail_ic
    apd_full.to_parquet(OUT_DIR / f"{slug}_predictions.parquet", index=False)

    out_results = {}
    for sub, col in [("A","pred_z"), ("B","pred_B")]:
        apd_v = apd_full.copy(); apd_v["pred"] = apd_v[col]
        records = psl.run_production_protocol_save_sleeves(apd_v, universe)
        df_v = aggregate_hold_through(records, alpha_wide)
        net = df_v["net_pnl_bps"].to_numpy()
        sh = _sharpe(net)
        out_results[sub] = {"sharpe":sh, "gross":df_v["gross_pnl_bps"].mean(),
                             "folds_pos":folds_positive(df_v),
                             "end_eq":CAPITAL+net.sum()/1e4*CAPITAL,
                             "df_v":df_v, "overall_ic":overall_ic}
        # Save per-cycle df
        df_v.to_parquet(OUT_DIR / f"{slug}_{sub}_per_cycle.parquet", index=False)
        print(f"  {label} {sub}_{'baseline' if sub=='A' else 'IC_signed'}: "
              f"Sharpe={sh:+.2f}, gross={df_v['gross_pnl_bps'].mean():+.2f}, "
              f"folds+={folds_positive(df_v)}/9, end-eq=${out_results[sub]['end_eq']:.2f}",
              flush=True)

    # LOFO on B
    sh_B = out_results["B"]["sharpe"]; df_v_B = out_results["B"]["df_v"]
    lofo_rows = []
    print(f"  LOFO on B (Sharpe = {sh_B:+.2f}):", flush=True)
    for excl in range(1, 10):
        rem = df_v_B[df_v_B["fold"] != excl]["net_pnl_bps"].to_numpy()
        sh_rem = _sharpe(rem)
        d = sh_rem - sh_B
        flag = "  ← drives" if d < -0.4 else ""
        lofo_rows.append({"excl_fold":excl, "sharpe":sh_rem, "delta":d})
        print(f"    excl {excl}: {sh_rem:+.2f} (Δ {d:+.2f}){flag}", flush=True)
    pd.DataFrame(lofo_rows).to_csv(OUT_DIR / f"{slug}_lofo.csv", index=False)
    return out_results


def main():
    print("=== Step 34: V1 with NaN fix + re-standardization ===\n", flush=True)
    t0 = time.time()
    listings = get_listings()

    tgt = pd.read_parquet(TARGETS)
    tgt["open_time"] = pd.to_datetime(tgt["open_time"], utc=True)
    cols_base = list(set(FRAME_NEUTRAL + KEEP_USHAPE_R3))
    base = pd.read_parquet(PANEL, columns=["symbol","open_time"] + cols_base)
    base["open_time"] = pd.to_datetime(base["open_time"], utc=True)
    btc_panel = pd.read_parquet(PANEL_BTC, columns=["symbol","open_time"] + BTC_KEEP)
    btc_panel["open_time"] = pd.to_datetime(btc_panel["open_time"], utc=True)
    step29 = pd.read_parquet(STEP29, columns=["symbol","open_time"] + SHORT_HORIZON)
    step29["open_time"] = pd.to_datetime(step29["open_time"], utc=True)
    panel = tgt.merge(base, on=["symbol","open_time"], how="left")
    panel = panel.merge(btc_panel, on=["symbol","open_time"], how="left")
    panel = panel.merge(step29, on=["symbol","open_time"], how="left")
    print(f"Panel: {len(panel):,} rows", flush=True)

    # Pre-fix diagnostic: count NaNs in heavy-tail features before fix
    print("\nNaN counts in heavy-tail features:")
    for f in HEAVY_TAIL:
        if f in panel.columns:
            n_nan = panel[f].isna().sum()
            print(f"  {f}: {n_nan:,} NaNs ({n_nan/len(panel)*100:.1f}%)")

    folds_all = _multi_oos_splits(panel)
    train_mask = panel["open_time"].between(
        _slice(panel, folds_all[0])[0].open_time.min(),
        _slice(panel, folds_all[0])[0].open_time.max())

    panel_syms = set(panel["symbol"].unique())
    for s, t in panel.groupby("symbol")["open_time"].min().items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t

    # V0
    X0, feat_cols_0 = preprocess_features(panel, train_mask, mode="standard")
    panel_x0 = panel[["symbol","open_time","alpha_beta","target_z","autocorr_pctile_7d"]].merge(
        X0.drop(columns=["alpha_beta","target_z","autocorr_pctile_7d"]),
        on=["symbol","open_time"], how="left")
    r0 = run_variant("V0 standard", "v0_standard", panel_x0, folds_all, feat_cols_0,
                      listings, panel_syms)

    # V1 (fixed)
    X1, feat_cols_1 = preprocess_features(panel, train_mask, mode="proper")
    # Sanity: print feature std stats so we can verify uniform scale
    print("\nV1 feature std (post fix + re-std):")
    for f in feat_cols_1:
        std = X1[f].std()
        print(f"  {f:35s} std={std:.3f}")
    panel_x1 = panel[["symbol","open_time","alpha_beta","target_z","autocorr_pctile_7d"]].merge(
        X1.drop(columns=["alpha_beta","target_z","autocorr_pctile_7d"]),
        on=["symbol","open_time"], how="left")
    r1 = run_variant("V1 proper (fixed)", "v1_fixed", panel_x1, folds_all, feat_cols_1,
                      listings, panel_syms)

    # V2
    X2, new_cols = add_short_horizon(X1.copy(), panel, train_mask)
    feat_cols_2 = feat_cols_1 + new_cols
    panel_x2 = panel[["symbol","open_time","alpha_beta","target_z","autocorr_pctile_7d"]].merge(
        X2.drop(columns=["alpha_beta","target_z","autocorr_pctile_7d"]),
        on=["symbol","open_time"], how="left")
    r2 = run_variant("V2 proper + short", "v2_fixed", panel_x2, folds_all, feat_cols_2,
                      listings, panel_syms)

    # Summary
    summary = []
    print(f"\n{'='*100}", flush=True)
    print(f"  STEP 34 SUMMARY (NaN-fixed + re-standardized)", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"  Variant                              A_baseline    B_IC_signed    folds+ (B)    overall_IC",
          flush=True)
    for label, slug, r in [("V0 standard", "v0_standard", r0),
                      ("V1 proper (fixed)", "v1_fixed", r1),
                      ("V2 proper + short", "v2_fixed", r2)]:
        print(f"  {label:<38} {r['A']['sharpe']:+10.2f}    {r['B']['sharpe']:+10.2f}     "
              f"{r['B']['folds_pos']}/9       {r['B']['overall_ic']:+.4f}", flush=True)
        summary.append({"variant":label,
                        "A_sharpe":r["A"]["sharpe"], "B_sharpe":r["B"]["sharpe"],
                        "B_folds_pos":r["B"]["folds_pos"], "overall_ic":r["B"]["overall_ic"],
                        "B_end_eq":r["B"]["end_eq"]})
    pd.DataFrame(summary).to_csv(OUT_DIR / "summary.csv", index=False)

    print(f"\n  Artifacts in: {OUT_DIR}", flush=True)
    print(f"  Total: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
