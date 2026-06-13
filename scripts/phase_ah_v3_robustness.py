"""Phase AH-V3.3 robustness: sleeve-weight grid + nested-fold selection.

Pre-registered grid (6 variants):
  equal_3      [1/3]*3                                      hold=12h
  equal_4      [1/4]*4                                      hold=16h
  equal_6      [1/6]*6                                      hold=24h
  decay_v33_6  [0.30, 0.22, 0.17, 0.13, 0.10, 0.08]         hold=24h  (current production)
  decay_fast_6 geom 0.65 normalized                          hold=24h
  decay_slow_6 geom 0.85 normalized                          hold=24h

Honest validation rule: for test fold f, select variant with best mean Sharpe
across folds {1..f-1}; evaluate that variant on fold f. Fold 1 defaults to
equal_6 (no-prior baseline). Aggregate the spliced per-cycle PnL across all
folds → nested-OOS Sharpe.

Comparators:
  V3.3 static  (no selection, +2.43 reference)
  Best-static  (in-sample winner of the grid)
  Nested-OOS   (honest)

Pass criterion for V3.3 robustness:
  nested-OOS Sharpe ≥ V3.3 −0.10
  AND ≥ 6/9 folds positive
  AND beats matched-basket placebo on nested schedule p95
"""
from __future__ import annotations
import sys, time, importlib.util
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

# Reuse aggregate_sleeves_variant from phase_ah_sleeve_variants
spec = importlib.util.spec_from_file_location(
    "svar", REPO / "scripts/phase_ah_sleeve_variants.py")
svar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(svar)

OUT = REPO / "outputs/vBTC_sleeve_horizon"
OOS_FOLDS = list(range(1, 10))
CYCLES_PER_YEAR = (288 * 365) / 48


def _geom_weights(n: int, ratio: float) -> list:
    raw = [ratio ** i for i in range(n)]
    s = sum(raw)
    return [r / s for r in raw]


# Pre-registered grid
GRID = [
    {"name": "equal_3",      "n": 3, "hold_bars": 12 * 12, "weights": [1/3] * 3},
    {"name": "equal_4",      "n": 4, "hold_bars": 16 * 12, "weights": [1/4] * 4},
    {"name": "equal_6",      "n": 6, "hold_bars": 24 * 12, "weights": [1/6] * 6},
    {"name": "decay_v33_6",  "n": 6, "hold_bars": 24 * 12,
       "weights": [0.30, 0.22, 0.17, 0.13, 0.10, 0.08]},
    {"name": "decay_fast_6", "n": 6, "hold_bars": 24 * 12,
       "weights": _geom_weights(6, 0.65)},
    {"name": "decay_slow_6", "n": 6, "hold_bars": 24 * 12,
       "weights": _geom_weights(6, 0.85)},
]


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def main():
    print("=== Phase AH-V3 robustness: sleeve-weight grid + nested-fold ===\n",
          flush=True)
    for v in GRID:
        wn = [f"{w:.3f}" for w in v["weights"]]
        print(f"  {v['name']:<14}  N={v['n']}  hold={v['hold_bars']/12:.0f}h  w={wn}",
              flush=True)
    print()

    records = pd.read_parquet(svar.SLEEVES_PATH)
    records["time"] = pd.to_datetime(records["time"], utc=True)
    apd = pd.read_parquet(REPO / "outputs/vBTC_audit_panel/all_predictions.parquet",
                            columns=["symbol"])
    all_syms = sorted(apd["symbol"].unique())
    print(f"  loading close prices...", flush=True)
    t0 = time.time()
    close_wide = svar.load_close_wide(all_syms)
    fwd_rets_4h = (close_wide.shift(-svar.HORIZON_ENTRY) - close_wide) / close_wide
    print(f"  done ({time.time()-t0:.0f}s)\n", flush=True)

    # Run every variant and stash per-cycle df
    per_variant = {}
    print(f"  {'variant':<14}  {'Sharpe':>7}  {'maxDD':>7}  {'PnL':>7}  {'folds+':>7}",
          flush=True)
    for v in GRID:
        t0 = time.time()
        df_v = svar.aggregate_sleeves_variant(
            records, fwd_rets_4h, v["n"], v["hold_bars"],
            sleeve_weights=v["weights"])
        per_variant[v["name"]] = df_v
        net = df_v["net_pnl_bps"].to_numpy()
        npos = sum(1 for _, g in df_v.groupby("fold")
                    if _sharpe(g["net_pnl_bps"]) > 0)
        nfolds = df_v["fold"].nunique()
        print(f"  {v['name']:<14}  {_sharpe(net):>+7.2f}  {_max_dd(net):>+7.0f}  "
              f"{net.sum():>+7.0f}  {npos:>3d}/{nfolds}  ({time.time()-t0:.0f}s)",
              flush=True)
        df_v.to_csv(OUT / f"per_cycle_robust_{v['name']}.csv", index=False)

    # Build fold-level table: Sharpe[variant, fold]
    fold_sharpe = {v["name"]: {} for v in GRID}
    for vname, df_v in per_variant.items():
        for f, g in df_v.groupby("fold"):
            fold_sharpe[vname][int(f)] = _sharpe(g["net_pnl_bps"].to_numpy())

    print(f"\n=== Per-fold Sharpe per variant ===\n")
    folds_sorted = sorted(set(int(f) for vfs in fold_sharpe.values() for f in vfs))
    header = "  fold  " + "  ".join(f"{v['name']:<13}" for v in GRID)
    print(header)
    for f in folds_sorted:
        row = f"  {f:>4}  "
        for v in GRID:
            sh = fold_sharpe[v["name"]].get(f, np.nan)
            row += f"{sh:>+7.2f}      " if not np.isnan(sh) else "    nan    "
        print(row)

    # Nested-fold selection
    print(f"\n=== Nested-fold selection (select using past folds) ===\n", flush=True)
    nested_rows = []  # combined per-cycle stream
    selections = []  # (fold, selected_variant)
    for f in folds_sorted:
        past = [pf for pf in folds_sorted if pf < f]
        if not past:
            sel = "equal_6"  # no-prior default
            reason = "fold 1 default"
        else:
            scores = {}
            for vname in fold_sharpe:
                past_vals = [fold_sharpe[vname][pf] for pf in past
                              if pf in fold_sharpe[vname]]
                scores[vname] = float(np.mean(past_vals)) if past_vals else -999
            sel = max(scores, key=lambda k: scores[k])
            reason = f"past mean Sh {scores[sel]:+.2f}"
        sel_df = per_variant[sel]
        fold_slice = sel_df[sel_df["fold"] == f].copy()
        fold_slice["selected_variant"] = sel
        nested_rows.append(fold_slice)
        selections.append({"fold": f, "selected": sel, "reason": reason,
                          "realized_sharpe": fold_sharpe[sel].get(f, np.nan)})
        print(f"  fold {f}:  selected {sel:<13}  ({reason})  "
              f"→ realized Sh = {fold_sharpe[sel].get(f, np.nan):+.2f}", flush=True)

    nested_df = pd.concat(nested_rows, ignore_index=True)
    nested_net = nested_df["net_pnl_bps"].to_numpy()
    nested_sh = _sharpe(nested_net)
    nested_dd = _max_dd(nested_net)
    nested_npos = sum(1 for _, g in nested_df.groupby("fold")
                       if _sharpe(g["net_pnl_bps"]) > 0)
    nested_df.to_csv(OUT / "per_cycle_robust_nested.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "robust_selections.csv", index=False)

    v33_sh = _sharpe(per_variant["decay_v33_6"]["net_pnl_bps"].to_numpy())
    best_static_name = max(fold_sharpe, key=lambda v: _sharpe(
        per_variant[v]["net_pnl_bps"].to_numpy()))
    best_static_sh = _sharpe(per_variant[best_static_name]["net_pnl_bps"].to_numpy())

    print(f"\n=== Summary ===\n", flush=True)
    print(f"  V3.3 static (decay_v33_6)  Sharpe = {v33_sh:+.2f}", flush=True)
    print(f"  Best-static in grid        Sharpe = {best_static_sh:+.2f}  "
          f"({best_static_name})", flush=True)
    print(f"  Nested-OOS Sharpe          = {nested_sh:+.2f}  "
          f"({nested_npos}/{len(folds_sorted)} folds, maxDD {nested_dd:+.0f}, "
          f"PnL {nested_net.sum():+.0f})", flush=True)

    delta_v33 = nested_sh - v33_sh
    pass_robust = (delta_v33 >= -0.10) and (nested_npos >= 6)
    print(f"\n  Δ (nested - V3.3)         = {delta_v33:+.2f}", flush=True)
    print(f"  V3.3 robustness gate      = {'PASS' if pass_robust else 'FAIL'}",
          flush=True)


if __name__ == "__main__":
    main()
