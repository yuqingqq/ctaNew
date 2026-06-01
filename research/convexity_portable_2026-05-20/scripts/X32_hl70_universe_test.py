"""X32 — Universe test on HL-70 panel: extends X23/X24 to 70 syms.

Tests Ridge Per-sym + cohort (canonical baseline) on:
  - Full HL-70 (70 syms) — main comparison vs HL-50 +2.01
  - HL-50 baseline (sanity check, should reproduce +2.01)
  - HL-70 minus low-vol-20 (= HL-50 effectively)
  - HL-70 top-15-vol-only, top-30-vol-only
  - HL-70 bottom-20-vol-only (the 20 new syms)
  - HL-70 minus-AI-cluster (test AI dependency on bigger universe)

All use canonical Ridge Per-sym + cohort + X21 framework fix.
"""
from __future__ import annotations
import csv, sys, time, warnings, importlib.util, gc, resource
from pathlib import Path
import pandas as pd, numpy as np

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
    """Load HL-70 panel and filter to given syms. Rebuild cohort fresh.
    Compute bars_since_high_xs_rank on the fly (not in panel_hl70)."""
    base_minus_xsrank = [c for c in x6.BASE if c != "bars_since_high_xs_rank"]
    needed = ["symbol", "open_time", "exit_time", "alpha_vs_btc_realized", "return_pct"] + base_minus_xsrank
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_hl70.parquet",
                            columns=list(set(needed)))
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[panel["symbol"].isin(syms) & (panel["symbol"] != "BTCUSDT")].copy()
    # Compute bars_since_high_xs_rank: cross-sectional rank at each time
    panel["bars_since_high_xs_rank"] = (panel.groupby("open_time")["bars_since_high"]
                                        .rank(pct=True).astype("float32"))
    panel = x6b.build_cohort_fixed(panel)
    panel = x6.build_target_z(panel)
    # X21 fix
    x6.HEAVY_TAIL.discard("rvol_7d"); x6.HEAVY_TAIL.discard("ret_3d"); x6.HEAVY_TAIL.discard("btc_rvol_7d")
    return panel


def main():
    t0 = time.time()
    print("=== X32 HL-70 universe test ===\n", flush=True)
    log_mem("start")

    HL_MAP = pd.read_csv(REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv")
    hl70 = HL_MAP[HL_MAP.on_hl].sort_values("hl_day_vol_usd", ascending=False)
    HL_SYMS = [s for s in hl70["symbol"].tolist()]
    print(f"  HL-70 (excl BTC): {len(HL_SYMS)}", flush=True)

    # Identify subsets
    panel_hl70 = pd.read_parquet(REPO / "outputs/vBTC_features/panel_hl70.parquet",
                                   columns=["symbol"])
    syms_in_panel = set(panel_hl70["symbol"].unique())
    HL_70 = [s for s in HL_SYMS if s in syms_in_panel]
    print(f"  HL-70 in panel: {len(HL_70)}", flush=True)
    # First 50 by vol = HL-50 (top-vol)
    HL_50 = HL_70[:50]
    HL_30 = HL_70[:30]

    # Identify AI cluster (from clusters_v1.json)
    import json
    with open(REPO / "config/clusters_v1.json") as f:
        clusters = json.load(f)
    AI_SYMS = set(clusters.get("ai", []))

    universes = {
        "HL70_full":       HL_70,
        "HL50_sanity":     HL_50,
        "HL70_top30":      HL_70[:30],
        "HL70_top15":      HL_70[:15],
        "HL70_bot20":      HL_70[-20:],
        "HL70_no_AI":      [s for s in HL_70 if s not in AI_SYMS],
        "HL70_minus_top10":HL_70[10:],  # like X11 U1 but on 70
    }

    print(f"\n=== {len(universes)} universe variants on HL-70 panel ===")
    for k, v in universes.items():
        print(f"  {k}: {len(v):>2} syms — {v[:3]}...{v[-1] if v else ''}")

    feats = x6.BASE + x6.COHORT_EXTRAS
    results = []
    for u_name, u_syms in universes.items():
        tf = time.time()
        log_mem(f"before {u_name}")
        print(f"\n[{u_name}] {len(u_syms)} syms", flush=True)
        try:
            panel = get_panel(set(u_syms))
            folds = x6.get_folds(panel)
            print(f"  panel: {len(panel):,} rows × {panel['symbol'].nunique()} syms, {len(folds)} folds")
            if len(folds) == 0:
                print(f"  no folds, skipping"); continue
            apd = x6.train_per_sym_ridge(panel, folds, feats, label=u_name)
            pred_path = CACHE / f"x32_{u_name}_preds.parquet"
            apd.to_parquet(pred_path, index=False)
            ic = float(apd["pred"].corr(apd["alpha_A"]))
            print(f"  trained: {len(apd):,} rows, IC={ic:+.4f} [{time.time()-tf:.0f}s]", flush=True)
        except Exception as e:
            print(f"  TRAIN ERR: {e}"); import traceback; traceback.print_exc()
            results.append({"universe": u_name, "n_syms": len(u_syms), "error": str(e)})
            continue
        m = x6.run_sleeve_on_preds(pred_path, f"x32_{u_name}")
        row = {"universe": u_name, "n_syms": len(u_syms), "train_ic": round(ic, 4), **m}
        results.append(row)
        if "sharpe" in m:
            print(f"  sleeve: Sharpe {m['sharpe']:+.2f} folds {m.get('folds_pos','?')} "
                  f"conc {m.get('concentration','?')} PnL {m.get('totPnL','?')}", flush=True)
        del panel, apd; gc.collect()

    keys = ["universe", "n_syms", "train_ic", "sharpe", "ci_lo", "ci_hi",
            "totPnL", "maxDD", "folds_pos", "concentration", "error"]
    out_csv = OUT / "X32_hl70_universe.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"\nSaved {len(results)} → {out_csv} [{time.time()-t0:.0f}s]")
    print(f"\nReference: HL-50 canonical = +2.01 (X23 reproduces exactly)")
    print(f"\n=== X32 results ===")
    for r in results:
        if "sharpe" in r:
            print(f"  {r['universe']:<22} ({r['n_syms']:>2}) Sharpe={r['sharpe']:+.2f} "
                  f"folds={r.get('folds_pos','?')} conc={r.get('concentration','?')}")


if __name__ == "__main__":
    main()
