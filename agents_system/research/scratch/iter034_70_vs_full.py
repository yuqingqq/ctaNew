"""iter-034 — REPORTING: clean 70 vs 156 head-to-head for the deployment champion.

Holds MODEL + CONSTRUCTION + WINDOW constant, varies ONLY the universe, using the SAME
expanded x132 V0 preds so the comparison is apples-to-apples (within-V0). Reuses the iter-032
fast engine VERBATIM (verified == iter-031 slow engine to 1e-13, which itself reuses X117
held-book regime-hybrid + X125 iter-012 vol-norm stop k=2.0).

Universes (all restricted to the x132 preds):
  - 70  = the original HL70 symbols present in x132 (intersect = 68 names; ASTER/TST absent)
  - 156 = full x132 (all tradable, ex-BTC)
  - hist-gated wide = full set above ~30d (180 4h-bar) per-cycle trailing-history floor (iter-032 deploy)

Windows:
  - FULL  = all 8 folds (2021-26 multi-episode transport view)
  - HL70-era = folds 7+8 (2025-02 -> 2026-05, the production-relevant recent period; HL70 preds
    span 2025-03 -> 2026-05 which lives inside these two folds)

Each cell: base held-book AND +iter-012 stop. Metrics: Sharpe, maxDD(bps), Calmar, totPnL, %positive,
folds_positive.
"""
from __future__ import annotations
import time, importlib.util as _ilu
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
SCR = REPO/"agents_system/research/scratch"
RC = REPO/"research/convexity_portable_2026-05-20/results/_cache"
OUT = REPO/"outputs/iter034"; OUT.mkdir(parents=True, exist_ok=True)

X132_PREDS = RC/"x132_expanded_v0_preds.parquet"
HL70_PREDS = RC/"x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet"
ANN = 6*365
HL70_ERA_FOLDS = {7, 8}   # 2025-02-09 -> 2026-05-06; HL70 preds (2025-03..2026-05) live inside these

# reuse the iter-032 fast engine (which imports iter-031 verbatim)
_s = _ilu.spec_from_file_location("i32", SCR/"iter032_expanded_universe.py")
i32 = _ilu.module_from_spec(_s); _s.loader.exec_module(i32)
i31 = i32.i31
precompute_cycles = i32.precompute_cycles
fast_cyc_weights = i32.fast_cyc_weights
PRIMARY_COST = i31.PRIMARY_COST


def metrics_window(pnl, folds, fold_mask):
    """Metrics over the cycles whose fold is in fold_mask (set), plus folds_positive within it."""
    sel = np.array([f in fold_mask for f in folds])
    p = pd.Series(pnl[sel]).dropna()
    m = i31.metrics(pnl[sel])
    # folds_positive within the window
    fp = 0; nf = 0
    for f in sorted(fold_mask):
        seg = pnl[folds == f]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= 3:
            nf += 1
            sh = i31.ann(pd.Series(seg)/1e4)
            if np.isfinite(sh) and sh > 0:
                fp += 1
    m["folds_positive"] = fp; m["n_folds"] = nf
    return m


def run_cell(PC, subset_ids, with_stop, hist_ok, folds_arr):
    cyc_w, rs = fast_cyc_weights(PC, subset_ids, hist_ok=hist_ok)
    pnl = i31.volnorm_stop(cyc_w, rs, PRIMARY_COST) if with_stop else i31.heldbook(cyc_w, rs, PRIMARY_COST)*1e4
    all_folds = set(int(f) for f in np.unique(folds_arr) if f >= 0)
    out = {}
    out["FULL"] = metrics_window(pnl, folds_arr, all_folds)
    out["HL70era"] = metrics_window(pnl, folds_arr, HL70_ERA_FOLDS)
    return out


def main():
    t0 = time.time()
    print("="*100)
    print("iter-034 — 70 vs 156 head-to-head (within-V0 x132 preds, model+construction+window held constant)")
    print("="*100, flush=True)

    PX = i32.cached_panel(X132_PREDS, "X132 expanded (156)", "x132") if hasattr(i32, "cached_panel") else None
    if PX is None:
        # cached_panel is defined inside i32.main(); replicate the cache load here.
        import pickle
        cp = REPO/"outputs/iter032/panel_x132.pkl"
        PX = pickle.loads(cp.read_bytes())
        print("  loaded cached x132 panel", flush=True)
    PCX = precompute_cycles(PX)

    # symbol sets
    full156 = [s for s in PX["syms"] if s != "BTCUSDT"]
    hl70 = sorted(pd.read_parquet(HL70_PREDS, columns=["symbol"])["symbol"].unique())
    hl_in_x132 = [s for s in hl70 if s in set(PX["syms"]) and s != "BTCUSDT"]
    missing = sorted(set(hl70) - set(PX["syms"]))
    print(f"  full-156 tradable legs: {len(full156)}")
    print(f"  HL70 intersect x132: {len(hl_in_x132)} (missing from x132: {missing})", flush=True)

    def ids(names):
        return [PCX["sidx"][s] for s in names if s in PCX["sidx"]]

    # per-cycle history gate (rebuilt exactly as in iter-032 block 4b)
    draw = pd.read_parquet(X132_PREDS, columns=["symbol", "open_time", "pred"])
    draw["open_time"] = pd.to_datetime(draw["open_time"], utc=True)
    draw = draw[(draw["open_time"].dt.hour % 4 == 0) & (draw["open_time"].dt.minute == 0)].copy()
    draw = draw.sort_values(["symbol", "open_time"])
    draw["hist_bars"] = draw.groupby("symbol").cumcount()
    hb = draw.set_index(["symbol", "open_time"])["hist_bars"].to_dict()
    times = PCX["times"]; syms_by_id = PCX["syms"]
    hist_arr = []
    for ti, ot in enumerate(times):
        idz = PCX["cyc"][ti][0]
        hist_arr.append(np.array([hb.get((syms_by_id[j], ot), 0) for j in idz]))

    def hist_ok_180(ti, idz):
        return hist_arr[ti] >= 180

    fbt = PCX["fold_by_time"]
    folds_arr = np.array([fbt.get(t, -1) for t in times])

    # one-time fast==slow verification on the 70-subset (record)
    _, mF = run_cell(PCX, ids(hl_in_x132), True, None, folds_arr)["FULL"], None
    cyc_w, rs = i31.build_cyc_weights(PX, hl_in_x132)
    pnl_slow = i31.volnorm_stop(cyc_w, rs, PRIMARY_COST)
    sh_slow = i31.metrics(pnl_slow)["Sharpe"]
    cyc_w_f, rs_f = fast_cyc_weights(PCX, ids(hl_in_x132), hist_ok=None)
    pnl_fast = i31.volnorm_stop(cyc_w_f, rs_f, PRIMARY_COST)
    sh_fast = i31.metrics(pnl_fast)["Sharpe"]
    print(f"  VERIFY fast==slow on 70-subset: slow {sh_slow:+.6f} vs fast {sh_fast:+.6f} "
          f"(maxabs diff {np.nanmax(np.abs(pnl_slow-pnl_fast)):.2e})", flush=True)

    universes = [
        ("70 (HL68 in x132)", ids(hl_in_x132), None),
        ("156 (full)",        ids(full156),    None),
        ("hist-gated wide",   ids(full156),    hist_ok_180),
    ]

    rows = []
    print("\n" + "="*100)
    print("RESULTS  (Sharpe / maxDD / Calmar / totPnL / %pos / folds_pos)")
    print("="*100, flush=True)
    for uname, sids, hg in universes:
        for kind, ws in (("base", False), ("stop", True)):
            res = run_cell(PCX, sids, ws, hg, folds_arr)
            for win in ("FULL", "HL70era"):
                m = res[win]
                rows.append(dict(universe=uname, window=win, variant=kind,
                                 Sharpe=m["Sharpe"], maxDD=m["maxDD"], Calmar=m["Calmar"],
                                 totPnL=m["tot"], pct_pos=m["pct_pos"],
                                 folds_positive=m["folds_positive"], n_folds=m["n_folds"]))
                print(f"  {uname:<20} {win:<8} {kind:<4} "
                      f"Sh {m['Sharpe']:>+6.2f}  mDD {m['maxDD']:>+8.0f}  Cal {m['Calmar']:>+6.2f}  "
                      f"tot {m['tot']:>+9.0f}  %pos {m['pct_pos']:>5.1f}  fp {m['folds_positive']}/{m['n_folds']}",
                      flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT/"iter034_70_vs_full.csv", index=False)
    print(f"\nDone [{time.time()-t0:.0f}s]  -> {OUT/'iter034_70_vs_full.csv'}", flush=True)
    return df


if __name__ == "__main__":
    main()
