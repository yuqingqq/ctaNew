"""X11 — Universe stress test. Take BEST cell config and run on universe variants.

Best cell so far: Ridge Per-sym +cohort = +2.01 on HL-50.

Test 3 universe variants:
  U1: HL-50 minus top-10-vol syms (40-sym drop-top-vol)
  U2: $5M+ HL daily vol filter (subset of HL-50)
  U3: 51-panel including BTC (V3.1 native universe)

Goal: assess strategy robustness when universe composition shifts.
"""
from __future__ import annotations
import csv, sys, time, warnings, importlib.util, gc, resource
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.linear_model import RidgeCV


def log_mem(label=""):
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  [MEM {label}] peak_rss={rss_mb:.0f}MB", flush=True)
    return rss_mb

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


def get_panel(universe_syms):
    needed = (["symbol", "open_time", "exit_time", "alpha_vs_btc_realized", "return_pct"]
              + x6.BASE)
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                            columns=list(set(needed)))
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[panel["symbol"].isin(universe_syms)].copy()
    panel = x6b.build_cohort_fixed(panel)
    panel = x6.build_target_z(panel)
    for c in panel.columns:
        if panel[c].dtype in ("float64",): panel[c] = panel[c].astype("float32")
    for c in x6.COHORT_EXTRAS:
        x6.HEAVY_TAIL.add(c)
    return panel


def main():
    t0 = time.time()
    print("=== X11 universe stress test (best cell: Ridge Per-sym +cohort) ===\n", flush=True)

    HL_MAP = pd.read_csv(REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv")
    hl_syms = set(HL_MAP[HL_MAP["on_hl"] == True]["symbol"].tolist())
    p51 = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                          columns=["symbol"])
    syms51 = set(p51["symbol"].unique())
    hl50 = sorted(syms51 & hl_syms)  # 50 syms

    # U1: HL-50 minus top-10-vol (drop highest-volume)
    hl_with_vol = HL_MAP[HL_MAP.on_hl & HL_MAP.symbol.isin(hl50)].sort_values("hl_day_vol_usd",
                                                                                ascending=False)
    drop_top10 = set(hl_with_vol.head(10)["symbol"].tolist())
    u1 = sorted(set(hl50) - drop_top10)
    print(f"  U1: HL-50 minus top-10-vol = {len(u1)} syms")
    print(f"      dropped: {sorted(drop_top10)}")

    # U2: $5M+ HL vol
    u2 = sorted(set(hl_with_vol[hl_with_vol.hl_day_vol_usd >= 5e6]["symbol"].tolist()))
    print(f"  U2: $5M+ HL vol filter = {len(u2)} syms")

    # U3: 51-panel (incl BTC)
    u3 = sorted(syms51)
    print(f"  U3: full 51-panel (incl BTC) = {len(u3)} syms")

    universes = [("U1_drop_top10_vol", u1), ("U2_vol_5M_filter", u2), ("U3_51panel_with_btc", u3)]
    results = []
    panel = None  # explicit cleanup between universes
    for u_name, u_syms in universes:
        tf = time.time()
        log_mem(f"before {u_name}")
        if panel is not None:
            del panel
            gc.collect()
        print(f"\n[{u_name}] {len(u_syms)} syms", flush=True)
        try:
            panel = get_panel(set(u_syms))
            folds = x6.get_folds(panel)
            print(f"  panel: {len(panel):,} rows", flush=True)
            # Ridge Per-sym + cohort (best cell config)
            feats = x6.BASE + x6.COHORT_EXTRAS
            from research.convexity_portable_2026_05_20.scripts.X9b_ridge_all_memlite import train_persym_ridge_memlite as train_persym
        except ImportError:
            # inline the function
            def train_persym(panel, folds, feats):
                all_preds = []
                for f, ts, te, ec in folds:
                    train_all = panel[(panel["exit_time"] < ec) & panel["target_z"].notna()]
                    test_all = panel[(panel["open_time"] >= ts) & (panel["open_time"] <= te)]
                    out_frames = []
                    for sym, gtr in train_all.groupby("symbol"):
                        if len(gtr) < 300: continue
                        gte = test_all[test_all["symbol"] == sym]
                        if len(gte) < 30: continue
                        sstats, hstats = x6.fit_preproc(gtr, feats)
                        Xtr = x6.apply_preproc(gtr, feats, sstats, hstats).astype(np.float32)
                        Xte = x6.apply_preproc(gte, feats, sstats, hstats).astype(np.float32)
                        ytr = gtr["target_z"].to_numpy(np.float32)
                        try:
                            m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(Xtr, ytr)
                            pred = m.predict(Xte).astype(np.float32)
                        except Exception: continue
                        o = gte[["symbol", "open_time", "alpha_vs_btc_realized",
                                 "return_pct", "exit_time"]].copy()
                        o.columns = ["symbol", "open_time", "alpha_A", "return_pct", "exit_time"]
                        o["pred"] = pred; o["fold"] = f
                        out_frames.append(o)
                        del Xtr, Xte, ytr, m
                    if out_frames:
                        all_preds.append(pd.concat(out_frames, ignore_index=True))
                    gc.collect()
                return pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])

        try:
            apd = train_persym(panel, folds, feats)
            pred_path = CACHE / f"x11_{u_name}_preds.parquet"
            apd.to_parquet(pred_path, index=False)
            ic = float(apd["pred"].corr(apd["alpha_A"]))
            print(f"  trained: {len(apd):,} rows, IC={ic:+.4f} [{time.time()-tf:.0f}s]", flush=True)
        except Exception as e:
            print(f"  TRAIN ERR: {e}"); import traceback; traceback.print_exc()
            results.append({"universe": u_name, "n_syms": len(u_syms), "error": str(e)}); continue
        m = x6.run_sleeve_on_preds(pred_path, f"x11_{u_name}")
        row = {"universe": u_name, "n_syms": len(u_syms), "train_ic": round(ic, 4), **m}
        results.append(row)
        if "sharpe" in m:
            print(f"  sleeve: Sharpe {m['sharpe']:+.2f} folds {m.get('folds_pos','?')} "
                  f"conc {m.get('concentration','?')} PnL {m.get('totPnL','?')}", flush=True)
        del apd
        gc.collect()
        log_mem(f"after {u_name}")

    keys = ["universe", "n_syms", "train_ic", "sharpe", "ci_lo", "ci_hi",
            "totPnL", "maxDD", "folds_pos", "concentration", "error"]
    out_csv = OUT / "X11_universe_stress.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"\nSaved {len(results)} universes → {out_csv} [{time.time()-t0:.0f}s]")
    print(f"\nReference: HL-50 baseline Ridge Per-sym +cohort = +2.01 Sharpe")
    for r in results:
        if "sharpe" in r:
            print(f"  {r['universe']:<25} ({r['n_syms']} syms) Sharpe={r['sharpe']:+.2f}")


if __name__ == "__main__":
    main()
