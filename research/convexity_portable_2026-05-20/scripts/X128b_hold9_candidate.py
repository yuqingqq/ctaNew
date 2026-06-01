"""X128b — iter-014 follow-up: isolate the ONE robust lever the grid surfaced — HOLD (sleeve count) —
as a single discrete change K=5,HOLD=9 vs baseline K=5,HOLD=6, and test it honestly.

The X128 grid showed HOLD is the dominant cross-universe lever (longer hold cuts maxDD + raises Calmar
on ALL three universes via cost amortization, same mechanism as the vBTC sleeve-overlap finding), while
K is universe-dependent / noisy. The cross-universe-best K=2,HOLD=9 fails honest tests (nested-OOS churns
on HL70/S44; G6 CI crosses 0 on HL70). This script tests the SINGLE cleanest discrete change — bump only
the hold from 6 to 9 sleeves (24h -> 36h), keep K=5 — which is the structural choice most likely to
generalize because it is monotone and same-signed on every universe.

Tests: G6 paired block-bootstrap CI (HOLD=9 vs HOLD=6) per universe; nested-OOS choose-HOLD-on-past-folds;
per-fold Calmar win-count; cost sensitivity. Also a quick HOLD-only nested-OOS (choose hold in {3,6,9,12}).
Reuses X128 build_panel / pnl_for / metrics / paired_ci verbatim.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import importlib.util as _ilu
from pathlib import Path

REPO = Path("/home/yuqing/ctaNew")
SCRIPTS = REPO/"research/convexity_portable_2026-05-20/scripts"
_spec = _ilu.spec_from_file_location("x128", SCRIPTS/"X128_struct_K_hold_sweep.py")
x128 = _ilu.module_from_spec(_spec); _spec.loader.exec_module(x128)
build_panel, pnl_for, metrics, paired_ci = x128.build_panel, x128.pnl_for, x128.metrics, x128.paired_ci
HL70_PREDS, EXT_PREDS, S44_PREDS = x128.HL70_PREDS, x128.EXT_PREDS, x128.S44_PREDS
PRIMARY_COST = x128.PRIMARY_COST
COSTS_BPS = x128.COSTS_BPS
SEED = x128.SEED

K_FIX = 5
HOLD_GRID = [3, 6, 9, 12]
HOLD_BASE, HOLD_CAND = 6, 9


def main():
    t0 = time.time()
    print("="*110, flush=True)
    print("X128b — single discrete change: K=5 HOLD 6->9 (36h hold). Honest tests vs baseline.", flush=True)
    print("="*110, flush=True)
    panels = {nm: build_panel(pp, nm) for nm, pp in
              (("HL70", HL70_PREDS), ("EXT", EXT_PREDS), ("S44", S44_PREDS))}
    rng = np.random.default_rng(SEED)

    # precompute pnl for all holds at K=5
    full = {nm: {h: pnl_for(panels[nm], K_FIX, h, PRIMARY_COST) for h in HOLD_GRID} for nm in panels}

    print("\n--- G6 paired CI + G5 fold-wins: K5 HOLD9 vs HOLD6, @4.5bps ---", flush=True)
    for nm in ("HL70", "EXT", "S44"):
        panel = panels[nm]
        fold_arr = np.array([panel["fold_by_time"].get(t, -1) for t in panel["times"]])
        base = full[nm][HOLD_BASE]; cand = full[nm][HOLD_CAND]
        folds = sorted(f for f in pd.unique(fold_arr) if f >= 0)
        wins = nf = 0
        for f in folds:
            m = fold_arr == f
            if m.sum() < 3: continue
            nf += 1
            if metrics(cand[m])["Calmar"] >= metrics(base[m])["Calmar"]: wins += 1
        mean, lo, hi = paired_ci(cand-base, fold_arr, rng=rng)
        bm = metrics(base); cm = metrics(cand)
        print(f"  {nm}: H6 Sh{bm['Sharpe']:+.2f}/Cal{bm['Calmar']:+.2f}/DD{bm['maxDD']:+.0f}  ->  "
              f"H9 Sh{cm['Sharpe']:+.2f}/Cal{cm['Calmar']:+.2f}/DD{cm['maxDD']:+.0f} "
              f"(ddRed {(1-cm['maxDD']/bm['maxDD'])*100:+.0f}%)", flush=True)
        print(f"    G5 Calmar-wins {wins}/{nf}; G6 paired diff {mean:+.2f}bps/cyc CI[{lo:+.2f},{hi:+.2f}] "
              f"{'CLEARS 0' if lo > 0 else 'crosses 0'}", flush=True)

    print("\n--- nested-OOS: choose HOLD in {3,6,9,12} (K=5) on past folds, apply forward ---", flush=True)
    for nm in ("HL70", "EXT", "S44"):
        panel = panels[nm]
        fold_arr = np.array([panel["fold_by_time"].get(t, -1) for t in panel["times"]])
        folds = sorted(f for f in pd.unique(fold_arr) if f >= 0)
        oos_b, oos_c, chosen = [], [], []
        for i in range(1, len(folds)):
            past = np.isin(fold_arr, folds[:i]); fut = fold_arr == folds[i]
            best_h, best_sh = HOLD_BASE, -1e18
            for h in HOLD_GRID:
                sh = metrics(full[nm][h][past])["Sharpe"]
                if np.isfinite(sh) and sh > best_sh: best_sh, best_h = sh, h
            oos_c.append(full[nm][best_h][fut]); oos_b.append(full[nm][HOLD_BASE][fut]); chosen.append(best_h)
        ob = metrics(np.concatenate(oos_b)); oc = metrics(np.concatenate(oos_c))
        print(f"  {nm}: chosen HOLD per fold {chosen}", flush=True)
        print(f"    OOS H6-base Sh{ob['Sharpe']:+.2f}/Cal{ob['Calmar']:+.2f} -> nested-HOLD "
              f"Sh{oc['Sharpe']:+.2f}/Cal{oc['Calmar']:+.2f}  Δcal {oc['Calmar']-ob['Calmar']:+.2f} "
              f"{'GENERALIZES' if oc['Calmar'] >= ob['Calmar'] else 'churns'}", flush=True)
        # also the FIXED H9 forward (no choosing): does the static bump beat base OOS?
        print(f"    static H9 (no selection) forward over same folds: "
              f"Sh{metrics(np.concatenate([full[nm][9][fold_arr==folds[i]] for i in range(1,len(folds))]))['Sharpe']:+.2f}",
              flush=True)

    print("\n--- G8 cost sensitivity: K5 H9 vs H6 Calmar @ {1,3,4.5}bps ---", flush=True)
    for nm in ("HL70", "EXT", "S44"):
        row = f"  {nm}: "
        for cb in COSTS_BPS:
            b = metrics(pnl_for(panels[nm], K_FIX, HOLD_BASE, cb*1e-4))["Calmar"]
            c = metrics(pnl_for(panels[nm], K_FIX, HOLD_CAND, cb*1e-4))["Calmar"]
            row += f"@{cb}bps H6 {b:+.2f}/H9 {c:+.2f} (Δ{c-b:+.2f})   "
        print(row, flush=True)

    print(f"\nDone [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
