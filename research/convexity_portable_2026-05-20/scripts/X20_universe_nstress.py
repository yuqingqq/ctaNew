"""X20 — Universe N-stress + tier subsets on best cell (Ridge Per-sym +cohort).

Beyond X11's 3 universes (drop top-10-vol, $5M+ filter, with BTC), test:
  N1: HL-30 (30 smallest in HL-50)
  N2: HL-40 (40 random or by mid-tier)
  N3: HL-50 baseline (+2.01)
  N4: HL-60 (HL-50 + top 10 from outside HL)
  N5: HL-70 (HL-50 + top 20 from outside HL)
  N6: Coverage-intersect: syms with both aggT and crossX > 80% coverage

  T1: Top-10-vol only
  T2: Top-25-vol only
  T3: Bottom-25-vol only
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


def log_mem(label=""):
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  [MEM {label}] peak_rss={rss_mb:.0f}MB", flush=True)


def get_panel(syms):
    needed = ["symbol", "open_time", "exit_time", "alpha_vs_btc_realized", "return_pct"] + x6.BASE
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                            columns=list(set(needed)))
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[panel["symbol"].isin(syms) & (panel["symbol"] != "BTCUSDT")].copy()
    panel = x6b.build_cohort_fixed(panel)
    panel = x6.build_target_z(panel)
    for c in panel.columns:
        if panel[c].dtype in ("float64",): panel[c] = panel[c].astype("float32")
    for c in x6.COHORT_EXTRAS: x6.HEAVY_TAIL.add(c)
    return panel


def train_persym_ridge(panel, folds, feats):
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
                Xtr = x6.apply_preproc(gtr, feats, sstats, hstats).astype(np.float32)
                Xte = x6.apply_preproc(gte, feats, sstats, hstats).astype(np.float32)
                ytr = gtr["target_z"].to_numpy(np.float32)
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
    print("=== X20 universe N-stress + tier subsets ===\n", flush=True)
    log_mem("start")
    HL_MAP = pd.read_csv(REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv")
    hl_syms_df = HL_MAP[HL_MAP.on_hl].sort_values("hl_day_vol_usd", ascending=False)
    p51_syms = set(pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                                    columns=["symbol"])["symbol"].unique())
    hl_50 = sorted(set(hl_syms_df["symbol"].tolist()) & (p51_syms - {"BTCUSDT"}))
    # rank by vol
    vol_ranked = [s for s in hl_syms_df["symbol"].tolist() if s in hl_50]

    # All 70-ish HL syms (incl those not in 51-panel)
    hl_all = hl_syms_df["symbol"].tolist()
    # Syms in 51-panel but maybe missing from HL: keep hl_50 for HL-only universes

    universes = {
        "N1_HL30":      vol_ranked[:30],
        "N2_HL40":      vol_ranked[:40],
        "N3_HL50":      vol_ranked,
        "T1_top10_vol":  vol_ranked[:10],
        "T2_top25_vol":  vol_ranked[:25],
        "T3_bottom25_vol": vol_ranked[25:],
    }

    print(f"\n=== Universe definitions ===")
    for name, syms in universes.items():
        print(f"  {name}: {len(syms)} syms — {syms[:5]}...")

    feats = x6.BASE + x6.COHORT_EXTRAS
    results = []
    for u_name, u_syms in universes.items():
        tf = time.time()
        log_mem(f"before {u_name}")
        print(f"\n[{u_name}] {len(u_syms)} syms", flush=True)
        try:
            panel = get_panel(set(u_syms))
            folds = x6.get_folds(panel)
            print(f"  panel: {len(panel):,} rows", flush=True)
            apd = train_persym_ridge(panel, folds, feats)
            pred_path = CACHE / f"x20_{u_name}_preds.parquet"
            apd.to_parquet(pred_path, index=False)
            ic = float(apd["pred"].corr(apd["alpha_A"]))
            print(f"  trained: {len(apd):,} rows, IC={ic:+.4f} [{time.time()-tf:.0f}s]", flush=True)
        except Exception as e:
            print(f"  TRAIN ERR: {e}"); import traceback; traceback.print_exc()
            results.append({"universe": u_name, "n_syms": len(u_syms), "error": str(e)})
            continue
        m = x6.run_sleeve_on_preds(pred_path, f"x20_{u_name}")
        row = {"universe": u_name, "n_syms": len(u_syms), "train_ic": round(ic, 4), **m}
        results.append(row)
        if "sharpe" in m:
            print(f"  sleeve: Sharpe {m['sharpe']:+.2f} folds {m.get('folds_pos','?')} "
                  f"conc {m.get('concentration','?')} PnL {m.get('totPnL','?')}", flush=True)
        del panel, apd; gc.collect()
        log_mem(f"after {u_name}")

    keys = ["universe", "n_syms", "train_ic", "sharpe", "ci_lo", "ci_hi",
            "totPnL", "maxDD", "folds_pos", "concentration", "error"]
    out_csv = OUT / "X20_universe_nstress.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"\nSaved {len(results)} universes → {out_csv} [{time.time()-t0:.0f}s]")
    print(f"\nBaselines (Ridge Per-sym +cohort):")
    print(f"  HL-50 (X6b): +2.01")
    print(f"  HL-50 drop top-10-vol (X11): +0.61")
    print(f"  HL-50 $5M+ filter (X11): +0.92")
    for r in results:
        if "sharpe" in r:
            print(f"  {r['universe']:<22} ({r['n_syms']:>3} syms) Sharpe={r['sharpe']:+.2f}")


if __name__ == "__main__":
    main()
