"""X21a — Verify float32 vs float64 is the framework drift bug.

Train Ridge Per-sym +cohort with float32 (X19 mode) and float64 (X6b mode)
on the EXACT SAME panel and folds. If float64 reproduces +2.01 and float32
gives +0.19, the bug is confirmed and fixable.
"""
from __future__ import annotations
import sys, time, warnings, importlib.util, gc
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


def load_panel(downcast_f32):
    needed = ["symbol", "open_time", "exit_time", "alpha_vs_btc_realized", "return_pct"] + x6.BASE
    p = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                        columns=list(set(needed)))
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    p["exit_time"] = pd.to_datetime(p["exit_time"], utc=True)
    p = p[p["symbol"].isin(HL_SYMS) & (p["symbol"] != "BTCUSDT")].copy()
    p = x6b.build_cohort_fixed(p)
    p = x6.build_target_z(p)
    if downcast_f32:
        for c in p.columns:
            if p[c].dtype == "float64": p[c] = p[c].astype("float32")
    for c in x6.COHORT_EXTRAS: x6.HEAVY_TAIL.add(c)
    return p


def train_persym(panel, folds, feats, dtype):
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
                sstats, hstats = x6.fit_preproc(gtr, feats)
                Xtr = x6.apply_preproc(gtr, feats, sstats, hstats).astype(dtype)
                Xte = x6.apply_preproc(gte, feats, sstats, hstats).astype(dtype)
                ytr = gtr["target_z"].to_numpy(dtype)
                m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(Xtr, ytr)
                pred = m.predict(Xte).astype(np.float32)
            except Exception: continue
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
    print("=== X21a: float32 vs float64 verification on Ridge Per-sym +cohort ===\n", flush=True)
    feats = x6.BASE + x6.COHORT_EXTRAS

    for label, downcast_f32, dtype in [
        ("F64_canonical (X6b mode)", False, np.float64),
        ("F32_drifted (X19 mode)", True, np.float32),
    ]:
        tf = time.time()
        print(f"\n[{label}] downcast_panel={downcast_f32}, train_dtype={dtype.__name__}")
        panel = load_panel(downcast_f32)
        folds = x6.get_folds(panel)
        apd = train_persym(panel, folds, feats, dtype)
        ic = float(apd["pred"].corr(apd["alpha_A"]))
        pred_path = CACHE / f"x21a_{label.split()[0]}_preds.parquet"
        apd.to_parquet(pred_path, index=False)
        m = x6.run_sleeve_on_preds(pred_path, f"x21a_{label.split()[0]}")
        print(f"  trained: IC={ic:+.4f}, Sharpe={m.get('sharpe', '?'):+.2f} "
              f"folds={m.get('folds_pos','?')} conc={m.get('concentration','?')} "
              f"[{time.time()-tf:.0f}s]", flush=True)
        del panel, apd; gc.collect()

    print(f"\n=== Reference: X6b canonical Ridge Per-sym +cohort = +2.01 ===")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
