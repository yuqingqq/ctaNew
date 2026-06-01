"""X19 — Hyperparameter sweep beyond α: preprocessing × fold × target horizon.

Applied to the BEST cell: Ridge Per-sym +cohort (baseline +2.01).

Variants:
  P1: Feature preprocessing = standard winsor p1/p99 + z (baseline)
  P2: Feature preprocessing = rank-transform full XS rank + z
  P3: Feature preprocessing = robust median/MAD scaling
  E1: Embargo = 1 day (baseline)
  E2: Embargo = 3 days
  E3: Embargo = 7 days

Plus a coverage indicator test: add `has_aggT` binary flag as feature.
"""
from __future__ import annotations
import csv, sys, time, warnings, importlib.util, gc, resource
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

OUT = REPO / "research/convexity_portable_2026-05-20/results"
CACHE = OUT / "_cache"

spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)
spec_b = importlib.util.spec_from_file_location("x6b",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6b_cohort_fill.py")
x6b = importlib.util.module_from_spec(spec_b); spec_b.loader.exec_module(x6b)

HL_MAP = pd.read_csv(REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv")
HL_SYMS = set(HL_MAP[HL_MAP["on_hl"] == True]["symbol"].tolist())


def log_mem(label=""):
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  [MEM {label}] peak_rss={rss_mb:.0f}MB", flush=True)


def load_panel():
    needed = ["symbol", "open_time", "exit_time", "alpha_vs_btc_realized", "return_pct"] + x6.BASE
    p = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                        columns=list(set(needed)))
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    p["exit_time"] = pd.to_datetime(p["exit_time"], utc=True)
    p = p[p["symbol"].isin(HL_SYMS) & (p["symbol"] != "BTCUSDT")].copy()
    p = x6b.build_cohort_fixed(p)
    p = x6.build_target_z(p)
    for c in p.columns:
        if p[c].dtype in ("float64",): p[c] = p[c].astype("float32")
    for c in x6.COHORT_EXTRAS: x6.HEAVY_TAIL.add(c)
    return p


def preproc_winsor(train, test, feats):
    """Standard winsor p1/p99 + z (baseline)."""
    sstats, hstats = x6.fit_preproc(train, feats)
    Xtr = x6.apply_preproc(train, feats, sstats, hstats).astype(np.float32)
    Xte = x6.apply_preproc(test, feats, sstats, hstats).astype(np.float32)
    return Xtr, Xte


def preproc_rank(train, test, feats):
    """Rank-transform features (full distribution-rank, robust to outliers)."""
    Xtr = np.zeros((len(train), len(feats)), dtype=np.float32)
    Xte = np.zeros((len(test), len(feats)), dtype=np.float32)
    for j, f in enumerate(feats):
        vals = train[f].to_numpy()
        sorted_vals = np.sort(vals[~np.isnan(vals)])
        if len(sorted_vals) == 0:
            continue
        # Rank in train
        tr_ranks = np.searchsorted(sorted_vals, train[f].fillna(sorted_vals[len(sorted_vals)//2])) / len(sorted_vals)
        te_ranks = np.searchsorted(sorted_vals, test[f].fillna(sorted_vals[len(sorted_vals)//2])) / len(sorted_vals)
        # Center on 0.5 and scale
        Xtr[:, j] = ((tr_ranks - 0.5) * 2).astype(np.float32)
        Xte[:, j] = ((te_ranks - 0.5) * 2).astype(np.float32)
    return Xtr, Xte


def preproc_robust(train, test, feats):
    """Robust scaling: (x - median) / MAD."""
    Xtr = np.zeros((len(train), len(feats)), dtype=np.float32)
    Xte = np.zeros((len(test), len(feats)), dtype=np.float32)
    for j, f in enumerate(feats):
        vals = train[f].to_numpy()
        med = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals - med))
        if mad == 0:
            mad = 1.0
        Xtr[:, j] = ((train[f].fillna(med) - med) / (mad * 1.4826)).astype(np.float32)
        Xte[:, j] = ((test[f].fillna(med) - med) / (mad * 1.4826)).astype(np.float32)
    # Clip to [-5, 5] to handle extreme outliers
    Xtr = np.clip(Xtr, -5, 5)
    Xte = np.clip(Xte, -5, 5)
    return Xtr, Xte


def train_persym_ridge(panel, folds, feats, preproc_fn):
    all_preds = []
    for f, ts, te, ec in folds:
        train_all = panel[(panel["exit_time"] < ec) & panel["target_z"].notna()]
        test_all = panel[(panel["open_time"] >= ts) & (panel["open_time"] <= te)]
        out_frames = []
        for sym, gtr in train_all.groupby("symbol"):
            if len(gtr) < 300: continue
            gte = test_all[test_all["symbol"] == sym]
            if len(gte) < 30: continue
            try:
                Xtr, Xte = preproc_fn(gtr, gte, feats)
                ytr = gtr["target_z"].to_numpy(np.float32)
                m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(Xtr, ytr)
                pred = m.predict(Xte).astype(np.float32)
            except Exception:
                continue
            o = gte[["symbol", "open_time", "alpha_vs_btc_realized",
                     "return_pct", "exit_time"]].copy()
            o.columns = ["symbol", "open_time", "alpha_A", "return_pct", "exit_time"]
            o["pred"] = pred; o["fold"] = f
            out_frames.append(o)
            del Xtr, Xte, ytr, m
        if out_frames: all_preds.append(pd.concat(out_frames, ignore_index=True))
        gc.collect()
    return pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])


def main():
    t0 = time.time()
    print("=== X19 hyperparameter sweep on best cell (Ridge Per-sym +cohort) ===\n", flush=True)
    log_mem("start")
    panel = load_panel()
    folds = x6.get_folds(panel)
    print(f"  panel: {len(panel):,} rows", flush=True)
    log_mem("after_panel")

    feats = x6.BASE + x6.COHORT_EXTRAS

    variants = [
        ("P1_winsor_baseline", "Standard winsor p1/p99 + z (baseline)", preproc_winsor),
        ("P2_rank_transform",  "Rank-transform [-1, 1]", preproc_rank),
        ("P3_robust_mad",      "Robust median/MAD scaling, clip ±5", preproc_robust),
    ]

    results = []
    for v_name, desc, preproc_fn in variants:
        tf = time.time()
        log_mem(f"before {v_name}")
        print(f"\n[{v_name}] {desc}", flush=True)
        try:
            apd = train_persym_ridge(panel, folds, feats, preproc_fn)
            pred_path = CACHE / f"x19_{v_name}_preds.parquet"
            apd.to_parquet(pred_path, index=False)
            ic = float(apd["pred"].corr(apd["alpha_A"]))
            print(f"  trained: {len(apd):,} rows, IC={ic:+.4f} [{time.time()-tf:.0f}s]", flush=True)
        except Exception as e:
            print(f"  TRAIN ERR: {e}"); results.append({"variant": v_name, "error": str(e)}); continue
        m = x6.run_sleeve_on_preds(pred_path, f"x19_{v_name}")
        row = {"variant": v_name, "desc": desc, "train_ic": round(ic, 4),
               "train_time_s": round(time.time()-tf, 0), **m}
        results.append(row)
        if "sharpe" in m:
            print(f"  sleeve: Sharpe {m['sharpe']:+.2f} folds {m.get('folds_pos','?')} "
                  f"conc {m.get('concentration','?')} PnL {m.get('totPnL','?')}", flush=True)
        del apd; gc.collect()
        log_mem(f"after {v_name}")

    keys = ["variant", "desc", "train_ic", "sharpe", "ci_lo", "ci_hi",
            "totPnL", "maxDD", "folds_pos", "concentration", "train_time_s", "error"]
    out_csv = OUT / "X19_preproc_sweep.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"\nSaved {len(results)} → {out_csv} [{time.time()-t0:.0f}s]")
    print(f"\nReference: Ridge Per-sym +cohort (winsor baseline) = +2.01")


if __name__ == "__main__":
    main()
