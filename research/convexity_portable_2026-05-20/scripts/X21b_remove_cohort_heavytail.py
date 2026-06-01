"""X21b — Test fix: do NOT add COHORT_EXTRAS to HEAVY_TAIL.

X19/X20/X21a added COHORT_EXTRAS to HEAVY_TAIL set, switching them from standard
winsor+z to rank-transform+z. This was the framework drift bug.

X6b canonical setup uses standard preproc for cohort features → Sharpe +2.01.
This script verifies removing the HEAVY_TAIL addition restores the canonical Sharpe.
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


def load_panel(add_cohort_to_heavytail):
    needed = ["symbol", "open_time", "exit_time", "alpha_vs_btc_realized", "return_pct"] + x6.BASE
    p = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                        columns=list(set(needed)))
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    p["exit_time"] = pd.to_datetime(p["exit_time"], utc=True)
    p = p[p["symbol"].isin(HL_SYMS) & (p["symbol"] != "BTCUSDT")].copy()
    p = x6b.build_cohort_fixed(p)
    p = x6.build_target_z(p)
    # Reset HEAVY_TAIL to canonical default first (in case mutated by prior runs)
    x6.HEAVY_TAIL.discard("rvol_7d")
    x6.HEAVY_TAIL.discard("ret_3d")
    x6.HEAVY_TAIL.discard("btc_rvol_7d")
    if add_cohort_to_heavytail:
        for c in x6.COHORT_EXTRAS: x6.HEAVY_TAIL.add(c)
    print(f"  HEAVY_TAIL contains COHORT_EXTRAS = {add_cohort_to_heavytail}")
    return p


def train_persym(panel, folds, feats):
    """Match X6b's x6.train_per_sym_ridge exactly."""
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
                Xtr = x6.apply_preproc(gtr, feats, sstats, hstats)
                Xte = x6.apply_preproc(gte, feats, sstats, hstats)
                m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(Xtr, gtr["target_z"].to_numpy())
                pred = m.predict(Xte)
            except Exception: continue
            o = gte[["symbol", "open_time", "alpha_vs_btc_realized",
                     "return_pct", "exit_time"]].copy()
            o.columns = ["symbol", "open_time", "alpha_A", "return_pct", "exit_time"]
            o["pred"] = pred; o["fold"] = f
            out_frames.append(o)
            del Xtr, Xte, m
        if out_frames: all_preds.append(pd.concat(out_frames, ignore_index=True))
        gc.collect()
    return pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])


def main():
    t0 = time.time()
    print("=== X21b: cohort features in HEAVY_TAIL vs not ===\n", flush=True)
    feats = x6.BASE + x6.COHORT_EXTRAS

    for label, add_ht in [
        ("CANONICAL (cohort NOT in HEAVY_TAIL)", False),
        ("DRIFTED (cohort IN HEAVY_TAIL)", True),
    ]:
        tf = time.time()
        print(f"\n[{label}]")
        panel = load_panel(add_ht)
        folds = x6.get_folds(panel)
        apd = train_persym(panel, folds, feats)
        ic = float(apd["pred"].corr(apd["alpha_A"]))
        pred_path = CACHE / f"x21b_{label.split()[0]}_preds.parquet"
        apd.to_parquet(pred_path, index=False)
        m = x6.run_sleeve_on_preds(pred_path, f"x21b_{label.split()[0]}")
        print(f"  trained: IC={ic:+.4f}, Sharpe={m.get('sharpe', '?'):+.2f} "
              f"folds={m.get('folds_pos','?')} conc={m.get('concentration','?')} "
              f"[{time.time()-tf:.0f}s]", flush=True)
        del panel, apd; gc.collect()

    print(f"\nReference: X6b canonical Ridge Per-sym +cohort = +2.01")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
