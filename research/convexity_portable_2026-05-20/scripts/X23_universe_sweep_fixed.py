"""X23 — Expanded universe sweep with framework fix.

Uses canonical setup: x6.train_per_sym_ridge directly (no HEAVY_TAIL drift),
so absolute Sharpe is comparable to X6b's +2.01 baseline.

Universe variants:
  N-sweep (sym count): HL-20, HL-25, HL-30, HL-35, HL-40, HL-45, HL-50
  Top-N-vol:           top-5, top-10, top-15, top-20, top-25, top-30
  Bottom-N-vol:        bottom-15, bottom-20, bottom-25
  Coverage subset:     syms with both aggT_v2 + crossX (~22-25 intersection)

Best cell: Ridge Per-sym + cohort (canonical baseline +2.01).
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
    needed = ["symbol", "open_time", "exit_time", "alpha_vs_btc_realized", "return_pct"] + x6.BASE
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                            columns=list(set(needed)))
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[panel["symbol"].isin(syms) & (panel["symbol"] != "BTCUSDT")].copy()
    panel = x6b.build_cohort_fixed(panel)
    panel = x6.build_target_z(panel)
    # CRITICAL: do NOT add COHORT_EXTRAS to HEAVY_TAIL (X21 fix)
    # also reset in case any prior call mutated
    x6.HEAVY_TAIL.discard("rvol_7d"); x6.HEAVY_TAIL.discard("ret_3d"); x6.HEAVY_TAIL.discard("btc_rvol_7d")
    return panel


def main():
    t0 = time.time()
    print("=== X23 expanded universe sweep with framework fix ===\n", flush=True)
    log_mem("start")

    HL_MAP = pd.read_csv(REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv")
    hl_syms_df = HL_MAP[HL_MAP.on_hl].sort_values("hl_day_vol_usd", ascending=False)
    p51_syms = set(pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                                    columns=["symbol"])["symbol"].unique())
    hl_50 = [s for s in hl_syms_df["symbol"].tolist() if s in p51_syms - {"BTCUSDT"}]
    print(f"  HL-50 size: {len(hl_50)}", flush=True)

    universes = {
        # N-sweep
        "N_HL20": hl_50[:20],
        "N_HL25": hl_50[:25],
        "N_HL30": hl_50[:30],
        "N_HL35": hl_50[:35],
        "N_HL40": hl_50[:40],
        "N_HL45": hl_50[:45],
        "N_HL50": hl_50[:50],
        # Top-N-vol
        "T_top5":   hl_50[:5],
        "T_top10":  hl_50[:10],
        "T_top15":  hl_50[:15],
        "T_top20":  hl_50[:20],
        "T_top25":  hl_50[:25],
        "T_top30":  hl_50[:30],
        # Bottom-N-vol
        "B_bot15": hl_50[-15:],
        "B_bot20": hl_50[-20:],
        "B_bot25": hl_50[-25:],
    }

    print(f"\n=== {len(universes)} universe variants ===")
    for name, syms in universes.items():
        print(f"  {name}: {len(syms)} syms — {syms[:3]}...{syms[-1] if syms else ''}")

    feats = x6.BASE + x6.COHORT_EXTRAS
    results = []
    for u_name, u_syms in universes.items():
        tf = time.time()
        log_mem(f"before {u_name}")
        print(f"\n[{u_name}] {len(u_syms)} syms", flush=True)
        try:
            panel = get_panel(set(u_syms))
            folds = x6.get_folds(panel)
            if len(folds) == 0:
                print(f"  no folds, skipping"); continue
            apd = x6.train_per_sym_ridge(panel, folds, feats, label=u_name)
            pred_path = CACHE / f"x23_{u_name}_preds.parquet"
            apd.to_parquet(pred_path, index=False)
            ic = float(apd["pred"].corr(apd["alpha_A"]))
            print(f"  trained: {len(apd):,} rows, IC={ic:+.4f} [{time.time()-tf:.0f}s]", flush=True)
        except Exception as e:
            print(f"  TRAIN ERR: {e}"); import traceback; traceback.print_exc()
            results.append({"universe": u_name, "n_syms": len(u_syms), "error": str(e)})
            continue
        m = x6.run_sleeve_on_preds(pred_path, f"x23_{u_name}")
        row = {"universe": u_name, "n_syms": len(u_syms), "train_ic": round(ic, 4), **m}
        results.append(row)
        if "sharpe" in m:
            print(f"  sleeve: Sharpe {m['sharpe']:+.2f} folds {m.get('folds_pos','?')} "
                  f"conc {m.get('concentration','?')} PnL {m.get('totPnL','?')}", flush=True)
        del panel, apd; gc.collect()
        log_mem(f"after {u_name}")

    keys = ["universe", "n_syms", "train_ic", "sharpe", "ci_lo", "ci_hi",
            "totPnL", "maxDD", "folds_pos", "concentration", "error"]
    out_csv = OUT / "X23_universe_sweep_fixed.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"\nSaved {len(results)} → {out_csv} [{time.time()-t0:.0f}s]")
    print(f"\nReference: Ridge Per-sym +cohort canonical = +2.01 on HL-50")
    for r in results:
        if "sharpe" in r:
            print(f"  {r['universe']:<12} ({r['n_syms']:>3} syms) Sharpe={r['sharpe']:+.2f} "
                  f"folds={r.get('folds_pos','?')} conc={r.get('concentration','?')}")


if __name__ == "__main__":
    main()
