"""X6 — Controlled experiment: Linear vs LGBM × Pooled vs Per-symbol × Feature variants.

ALL trained from scratch with IDENTICAL conditions:
  - Universe: HL-50 (51-panel symbols minus BTC, all HL-tradeable, 50 syms)
  - Sample: 51-panel time range (2025-04 → 2026-05)
  - Target: alpha_vs_btc_realized → per-symbol PIT z (no clip)
  - Folds: 9 walk-forward expanding, 1-day embargo, label purge via exit_time
  - Preprocessing: heavy-tail features rank-transformed fold-0, others winsor p1/p99 + z fold-0
  - Eval: V3.1 6-sleeve, K=3, conv_gate + flat_real, 9 bps RT

Vary (16 + 2 control cells):
  - Model class: LGBM vs Ridge
  - Architecture: Pool+symid / Pool-nosym (LGBM control only) / Per-symbol
  - Feature set: BASE / +aggTrades / +cohort / +v3

Outputs CSV table + each cell's predictions parquet.
"""
from __future__ import annotations
import csv, sys, time, warnings, io, contextlib, json
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.linear_model import RidgeCV
import lightgbm as lgb

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

OUT = REPO / "research/convexity_portable_2026-05-20/results"
CACHE = OUT / "_cache"; CACHE.mkdir(parents=True, exist_ok=True)
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
KLINES = REPO / "data/ml/test/parquet/klines"

HORIZON = 48          # 4h target
TRAIL_BARS = 288 * 7
MIN_TRAIL = 288
N_FOLDS = 9
EMBARGO_DAYS = 1
SEED = 20260520

# Feature sets (all exist in 51-panel)
BASE = [
    "return_1d", "atr_pct", "obv_z_1d", "vwap_slope_96",
    "bars_since_high", "bars_since_high_xs_rank", "autocorr_pctile_7d",
    "corr_to_btc_1d", "beta_to_btc_change_5d",
    "idio_vol_to_btc_1h", "idio_vol_to_btc_1d",
    "funding_rate", "funding_rate_z_7d", "funding_rate_1d_change",
]

AGGT_EXTRAS = [
    "aggr_ratio_4h", "tfi_4h", "buy_count_4h",
    "signed_volume_4h", "avg_trade_size_4h",
]

# Cohort features — need to compute from close prices
COHORT_EXTRAS = ["rvol_7d", "ret_3d", "btc_rvol_7d"]

# v3-augment subset that exists in 51-panel
V3_EXTRAS = ["idio_max_abs_12b", "idio_skew_1d", "idio_kurt_1d", "name_idio_share_1d"]

# Heavy-tail features (use rank transform)
HEAVY_TAIL = set([
    "funding_rate", "funding_rate_z_7d", "funding_rate_1d_change",
    "idio_vol_to_btc_1h", "idio_vol_to_btc_1d",
    "vwap_slope_96",
    "aggr_ratio_4h", "tfi_4h", "buy_count_4h", "signed_volume_4h", "avg_trade_size_4h",
    "idio_max_abs_12b", "idio_skew_1d", "idio_kurt_1d",
])

# V3.1 LGBM params (pinned from production)
LGB_PARAMS_POOLED = dict(
    objective="regression", metric="rmse", learning_rate=0.03,
    num_leaves=31, max_depth=6, min_data_in_leaf=300,
    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
    reg_alpha=0.1, reg_lambda=0.1, verbose=-1, n_estimators=400,
)
LGB_PARAMS_PERSYM = dict(
    objective="regression", metric="rmse", learning_rate=0.05,
    num_leaves=15, max_depth=4, min_data_in_leaf=30,
    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
    reg_alpha=0.1, reg_lambda=0.1, verbose=-1, n_estimators=200,
)
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def build_cohort_features(panel):
    """Compute rvol_7d, ret_3d (per-symbol from close) + btc_rvol_7d (broadcast)."""
    closes = {}
    for sym in panel["symbol"].unique():
        sd = KLINES / sym / "5m"
        if not sd.exists(): continue
        dfs = []
        for f in sorted(sd.glob("*.parquet")):
            try:
                df = pd.read_parquet(f, columns=["open_time", "close"])
                dfs.append(df)
            except Exception: pass
        if not dfs: continue
        c = pd.concat(dfs, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
        c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
        c = c.set_index("open_time")["close"]
        closes[sym] = c
    print(f"  loaded closes for {len(closes)} syms")
    # build feature long-form
    rows_rvol, rows_ret = [], []
    for sym, c in closes.items():
        logret = np.log(c / c.shift(1))
        rv = logret.rolling(288*7, min_periods=288).std().shift(1)
        rt = c.pct_change(288*3).shift(1)
        df_r = rv.rename("rvol_7d").reset_index(); df_r["symbol"] = sym
        df_t = rt.rename("ret_3d").reset_index(); df_t["symbol"] = sym
        rows_rvol.append(df_r); rows_ret.append(df_t)
    rvol = pd.concat(rows_rvol, ignore_index=True)
    ret3 = pd.concat(rows_ret, ignore_index=True)
    btc = closes.get("BTCUSDT")
    if btc is not None:
        btc_rvol_series = np.log(btc / btc.shift(1)).rolling(288*7, min_periods=288).std().shift(1)
        btc_rvol = btc_rvol_series.rename("btc_rvol_7d").reset_index()
        panel = panel.merge(rvol, on=["symbol", "open_time"], how="left")
        panel = panel.merge(ret3, on=["symbol", "open_time"], how="left")
        panel = panel.merge(btc_rvol, on="open_time", how="left")
    return panel


def build_target_z(panel):
    """Per-symbol PIT z of alpha_vs_btc_realized. No clip."""
    panel = panel.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    panel["rmean"] = panel.groupby("symbol")["alpha_vs_btc_realized"].transform(
        lambda s: s.expanding(min_periods=MIN_TRAIL).mean().shift(HORIZON))
    panel["rstd"] = panel.groupby("symbol")["alpha_vs_btc_realized"].transform(
        lambda s: s.rolling(TRAIL_BARS, min_periods=MIN_TRAIL).std().shift(HORIZON))
    panel["target_z"] = ((panel["alpha_vs_btc_realized"] - panel["rmean"]) /
                         panel["rstd"].replace(0, np.nan)).clip(-50, 50)
    return panel


def fit_preproc(train_df, feat_cols):
    """Build preprocessing stats on fold-0 train rows."""
    sstats, hstats = {}, {}
    for c in feat_cols:
        if c not in train_df.columns: continue
        v = train_df[c].dropna()
        if len(v) < 50:
            (hstats if c in HEAVY_TAIL else sstats)[c] = (
                {"vals": np.array([0., 1.]), "mu": 0., "sd": 1.} if c in HEAVY_TAIL
                else {"lo": 0., "hi": 1., "mu": 0., "sd": 1.})
            continue
        if c in HEAVY_TAIL:
            sv = np.sort(v.to_numpy())
            ranks = np.searchsorted(sv, v, side="left") / max(1, len(sv)-1)
            hstats[c] = {"vals": sv, "mu": float(ranks.mean()),
                         "sd": float(ranks.std()) or 1.0}
        else:
            lo = float(v.quantile(0.01)); hi = float(v.quantile(0.99))
            vc = v.clip(lo, hi)
            sstats[c] = {"lo": lo, "hi": hi,
                         "mu": float(vc.mean()), "sd": float(vc.std()) or 1.0}
    return sstats, hstats


def apply_preproc(df, feat_cols, sstats, hstats):
    X = np.zeros((len(df), len(feat_cols)), dtype=np.float32)
    for i, c in enumerate(feat_cols):
        v = df[c].to_numpy() if c in df.columns else np.zeros(len(df))
        if c in hstats:
            h = hstats[c]
            with np.errstate(invalid="ignore"):
                r = np.searchsorted(h["vals"], v, side="left") / max(1, len(h["vals"])-1)
            X[:, i] = (r - h["mu"]) / h["sd"]
        elif c in sstats:
            s = sstats[c]
            X[:, i] = (np.clip(v, s["lo"], s["hi"]) - s["mu"]) / s["sd"]
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def get_folds(panel):
    """9-fold walk-forward expanding."""
    times = sorted(panel["open_time"].unique())
    n = len(times); fs = n // N_FOLDS
    folds = []
    for f in range(N_FOLDS):
        i0 = f * fs
        i1 = min((f+1)*fs, n-1) if f < N_FOLDS-1 else n
        oos_start = pd.Timestamp(times[i0])
        oos_end = pd.Timestamp(times[i1-1])
        embargo_cut = oos_start - pd.Timedelta(days=EMBARGO_DAYS)
        folds.append((f, oos_start, oos_end, embargo_cut))
    return folds


def train_pooled_lgbm(panel, folds, feat_cols, with_symid=True, label="lgbm_pool"):
    """Train one pooled LGBM (with or without sym_id) across all folds."""
    all_preds = []
    feats = feat_cols + (["sym_id"] if with_symid else [])
    syms_sorted = sorted(panel["symbol"].unique())
    sym_map = {s: i for i, s in enumerate(syms_sorted)}
    panel = panel.copy()
    panel["sym_id"] = panel["symbol"].map(sym_map).astype("int32")
    for f, ts, te, ec in folds:
        train = panel[(panel["exit_time"] < ec) & panel["target_z"].notna()]
        test = panel[(panel["open_time"] >= ts) & (panel["open_time"] <= te)]
        if len(train) < 5000 or len(test) < 1000: continue
        Xtr = train[feats]; Xte = test[feats]
        ytr = train["target_z"].to_numpy()
        cat = ["sym_id"] if with_symid else []
        m = lgb.LGBMRegressor(random_state=SEED, **LGB_PARAMS_POOLED)
        m.fit(Xtr, ytr, categorical_feature=cat)
        pred = m.predict(Xte)
        out = test[["symbol", "open_time", "alpha_vs_btc_realized",
                    "return_pct", "exit_time"]].copy()
        out.columns = ["symbol", "open_time", "alpha_A", "return_pct", "exit_time"]
        out["pred"] = pred; out["fold"] = f
        all_preds.append(out)
    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    return apd


def train_per_sym_lgbm(panel, folds, feat_cols, label="lgbm_persym"):
    """Train an independent LGBM per symbol per fold."""
    all_preds = []
    for f, ts, te, ec in folds:
        train_all = panel[(panel["exit_time"] < ec) & panel["target_z"].notna()]
        test_all = panel[(panel["open_time"] >= ts) & (panel["open_time"] <= te)]
        out_frames = []
        for sym, gtr in train_all.groupby("symbol"):
            if len(gtr) < 200: continue
            gte = test_all[test_all["symbol"] == sym]
            if len(gte) < 30: continue
            m = lgb.LGBMRegressor(random_state=SEED, **LGB_PARAMS_PERSYM)
            m.fit(gtr[feat_cols], gtr["target_z"].to_numpy())
            pred = m.predict(gte[feat_cols])
            o = gte[["symbol", "open_time", "alpha_vs_btc_realized",
                     "return_pct", "exit_time"]].copy()
            o.columns = ["symbol", "open_time", "alpha_A", "return_pct", "exit_time"]
            o["pred"] = pred; o["fold"] = f
            out_frames.append(o)
        if out_frames:
            all_preds.append(pd.concat(out_frames, ignore_index=True))
    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    return apd


def train_pooled_ridge(panel, folds, feat_cols, with_symid=True, label="ridge_pool"):
    """Train pooled Ridge (one-hot sym_id intercepts if with_symid)."""
    syms_sorted = sorted(panel["symbol"].unique())
    panel = panel.copy()
    if with_symid:
        sym_dum = pd.get_dummies(panel["symbol"], prefix="sym", drop_first=True).astype(np.float32)
        sym_dum.index = panel.index
    all_preds = []
    sstats_fold0, hstats_fold0 = None, None
    for f, ts, te, ec in folds:
        train = panel[(panel["exit_time"] < ec) & panel["target_z"].notna()]
        test = panel[(panel["open_time"] >= ts) & (panel["open_time"] <= te)]
        if len(train) < 5000 or len(test) < 1000: continue
        if f == 0 or sstats_fold0 is None:
            sstats_fold0, hstats_fold0 = fit_preproc(train, feat_cols)
        Xtr = apply_preproc(train, feat_cols, sstats_fold0, hstats_fold0)
        Xte = apply_preproc(test, feat_cols, sstats_fold0, hstats_fold0)
        if with_symid:
            Xtr_sym = sym_dum.loc[train.index].to_numpy()
            Xte_sym = sym_dum.loc[test.index].to_numpy()
            Xtr = np.hstack([Xtr, Xtr_sym]); Xte = np.hstack([Xte, Xte_sym])
        ytr = train["target_z"].to_numpy()
        m = RidgeCV(alphas=RIDGE_ALPHAS).fit(Xtr, ytr)
        pred = m.predict(Xte)
        out = test[["symbol", "open_time", "alpha_vs_btc_realized",
                    "return_pct", "exit_time"]].copy()
        out.columns = ["symbol", "open_time", "alpha_A", "return_pct", "exit_time"]
        out["pred"] = pred; out["fold"] = f
        all_preds.append(out)
    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    return apd


def train_per_sym_ridge(panel, folds, feat_cols, label="ridge_persym"):
    """Train Ridge independently per symbol per fold."""
    all_preds = []
    for f, ts, te, ec in folds:
        train_all = panel[(panel["exit_time"] < ec) & panel["target_z"].notna()]
        test_all = panel[(panel["open_time"] >= ts) & (panel["open_time"] <= te)]
        out_frames = []
        for sym, gtr in train_all.groupby("symbol"):
            if len(gtr) < 300: continue
            gte = test_all[test_all["symbol"] == sym]
            if len(gte) < 30: continue
            sstats, hstats = fit_preproc(gtr, feat_cols)
            Xtr = apply_preproc(gtr, feat_cols, sstats, hstats)
            Xte = apply_preproc(gte, feat_cols, sstats, hstats)
            try:
                m = RidgeCV(alphas=RIDGE_ALPHAS).fit(Xtr, gtr["target_z"].to_numpy())
                pred = m.predict(Xte)
            except Exception:
                continue
            o = gte[["symbol", "open_time", "alpha_vs_btc_realized",
                     "return_pct", "exit_time"]].copy()
            o.columns = ["symbol", "open_time", "alpha_A", "return_pct", "exit_time"]
            o["pred"] = pred; o["fold"] = f
            out_frames.append(o)
        if out_frames:
            all_preds.append(pd.concat(out_frames, ignore_index=True))
    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    return apd


def run_sleeve_on_preds(pred_path, label):
    """Run V3.1 sleeve on a predictions parquet, return key metrics."""
    import phase_ah_sleeve as P
    P.APD_PATH = pred_path
    P.OUT = OUT / f"_x6_sleeve_{label}"
    P.OUT.mkdir(parents=True, exist_ok=True)
    P.N_PLACEBO_SEEDS = 0
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            P.main()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    txt = buf.getvalue()
    metrics = {}
    for ln in txt.split("\n"):
        s = ln.strip()
        if s.startswith("Sharpe:") and "[" in s:
            try:
                metrics["sharpe"] = round(float(s.split("Sharpe:")[1].split("[")[0].strip().lstrip("+")), 3)
                metrics["ci_lo"] = round(float(s.split("[")[1].split(",")[0]), 3)
                metrics["ci_hi"] = round(float(s.split(",")[-1].split("]")[0]), 3)
            except Exception: pass
        elif s.startswith("totPnL:"):
            try: metrics["totPnL"] = int(s.split("totPnL:")[1].split("bps")[0].strip().lstrip("+"))
            except: pass
        elif s.startswith("maxDD:"):
            try: metrics["maxDD"] = int(s.split("maxDD:")[1].split("bps")[0].strip())
            except: pass
        elif "Folds positive:" in s:
            metrics["folds_pos"] = s.split("Folds positive:")[1].strip()
        elif "Concentration:" in s:
            metrics["concentration"] = s.split("Concentration:")[1].strip()
        elif s.startswith("net_avg:"):
            try: metrics["net_bps_cycle"] = round(float(s.split("net_avg:")[1].split("bps")[0].strip().lstrip("+")), 3)
            except: pass
    return metrics


def main():
    t0 = time.time()
    print("=== X6 controlled comparison matrix ===\n", flush=True)

    # Load 51-panel, restrict to HL-50
    HL_MAP = pd.read_csv(REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv")
    HL_SYMS = set(HL_MAP[HL_MAP["on_hl"] == True]["symbol"].tolist())
    needed_cols = (["symbol", "open_time", "exit_time", "alpha_vs_btc_realized", "return_pct"]
                   + BASE + AGGT_EXTRAS + V3_EXTRAS)
    needed_cols = list(set(needed_cols))
    p = pd.read_parquet(PANEL, columns=needed_cols)
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    p["exit_time"] = pd.to_datetime(p["exit_time"], utc=True)
    p = p[p["symbol"].isin(HL_SYMS)].copy()
    p = p[p["symbol"] != "BTCUSDT"]   # HL-50 = 51-panel minus BTC
    print(f"  HL-50 panel: {len(p):,} rows × {p['symbol'].nunique()} syms", flush=True)

    # Compute cohort features
    print("  computing cohort features (rvol_7d, ret_3d, btc_rvol_7d)...", flush=True)
    p = build_cohort_features(p)
    for c in COHORT_EXTRAS:
        if c in p.columns:
            print(f"    {c} non-null: {p[c].notna().mean()*100:.1f}%", flush=True)

    # Build target
    print("  building per-sym PIT-z target...", flush=True)
    p = build_target_z(p)
    print(f"    target_z non-null: {p['target_z'].notna().mean()*100:.1f}%, "
          f"std: {p['target_z'].std():.2f}", flush=True)

    folds = get_folds(p)
    print(f"  9 folds: {folds[0][1].date()} → {folds[-1][2].date()}", flush=True)

    # CELLS to train
    feature_sets = {
        "BASE": BASE,
        "+aggT": BASE + AGGT_EXTRAS,
        "+cohort": BASE + COHORT_EXTRAS,
        "+v3": BASE + V3_EXTRAS,
    }

    archs = [
        ("LGBM", "pool+symid", lambda panel, folds, feats:
            train_pooled_lgbm(panel, folds, feats, with_symid=True)),
        ("LGBM", "pool-nosym", lambda panel, folds, feats:
            train_pooled_lgbm(panel, folds, feats, with_symid=False)),
        ("LGBM", "per-sym", lambda panel, folds, feats:
            train_per_sym_lgbm(panel, folds, feats)),
        ("Ridge", "pool+symid", lambda panel, folds, feats:
            train_pooled_ridge(panel, folds, feats, with_symid=True)),
        ("Ridge", "pool-nosym", lambda panel, folds, feats:
            train_pooled_ridge(panel, folds, feats, with_symid=False)),
        ("Ridge", "per-sym", lambda panel, folds, feats:
            train_per_sym_ridge(panel, folds, feats)),
    ]

    rows = []
    cell_id = 0
    for model, arch, train_fn in archs:
        for fs_label, feats in feature_sets.items():
            cell_id += 1
            cell_label = f"{model}_{arch}_{fs_label}".replace("+", "p")
            print(f"\n[{cell_id}/{len(archs)*len(feature_sets)}] {model} | {arch} | {fs_label} "
                  f"(features={len(feats)})", flush=True)
            tf = time.time()
            try:
                apd = train_fn(p, folds, feats)
                pred_path = CACHE / f"x6_{cell_label}_preds.parquet"
                apd.to_parquet(pred_path, index=False)
                # quick IC
                ic = float(apd["pred"].corr(apd["alpha_A"]))
                print(f"    trained: {len(apd):,} rows, IC={ic:+.4f} [{time.time()-tf:.0f}s]",
                      flush=True)
            except Exception as e:
                print(f"    TRAIN ERROR: {type(e).__name__}: {e}", flush=True)
                rows.append({"cell": cell_label, "model": model, "arch": arch,
                              "feature_set": fs_label, "n_feats": len(feats),
                              "error": f"train: {e}"})
                continue
            # Sleeve eval
            m = run_sleeve_on_preds(pred_path, cell_label)
            row = {"cell": cell_label, "model": model, "arch": arch,
                   "feature_set": fs_label, "n_feats": len(feats),
                   "train_ic": round(ic, 4),
                   "train_time_s": round(time.time()-tf, 0),
                   **m}
            rows.append(row)
            if "sharpe" in m:
                print(f"    sleeve: Sharpe {m['sharpe']:+.2f} folds {m.get('folds_pos','?')} "
                      f"conc {m.get('concentration','?')} PnL {m.get('totPnL','?')}",
                      flush=True)
            else:
                print(f"    sleeve ERR: {m.get('error','?')}", flush=True)

    # Write CSV
    keys = ["cell", "model", "arch", "feature_set", "n_feats",
            "train_ic", "sharpe", "ci_lo", "ci_hi", "totPnL", "maxDD",
            "folds_pos", "concentration", "net_bps_cycle",
            "train_time_s", "error"]
    out_csv = OUT / "X6_controlled_matrix.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nSaved {len(rows)} cells to {out_csv} [{time.time()-t0:.0f}s]")

    # printout
    print(f"\n=== APPLES-TO-APPLES MATRIX (HL-50 universe, identical conditions) ===")
    print(f"{'model':<6} {'arch':<13} {'features':<10} {'n':>4} {'IC':>7} {'Sharpe':>8} {'folds':>7} {'conc':>6} {'PnL':>7}")
    print("-" * 95)
    for r in rows:
        sh = f"{r.get('sharpe', '?'):+.2f}" if "sharpe" in r else "ERR"
        print(f"{r['model']:<6} {r['arch']:<13} {r['feature_set']:<10} {r['n_feats']:>4} "
              f"{r.get('train_ic','?'):>+7} {sh:>8} "
              f"{str(r.get('folds_pos','?')):>7} {str(r.get('concentration','?')):>6} "
              f"{str(r.get('totPnL','?')):>7}")


if __name__ == "__main__":
    main()
