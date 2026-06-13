"""Phase AH-V3 robustness v2: V3.1 vs V3.3 decisive validation.

Four tests:
  1. Paired per-cycle bootstrap V3.1 vs V3.3 (does decay weighting add real PnL?)
  2. Restricted 2-variant nested-OOS {equal_6, decay_v33_6} (removes 6-variant noise)
  3. Per-variant per-fold block-bootstrap Sharpe CIs (where does V3.3's edge survive CI?)
  4. V3.1 equal_6 matched-basket placebo (100 seeds, equal weights through placebo)

Reuses per_cycle_robust_*.csv saved by phase_ah_v3_robustness.py.
"""
from __future__ import annotations
import sys, time, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location(
    "svar", REPO / "scripts/phase_ah_sleeve_variants.py")
svar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(svar)

spec2 = importlib.util.spec_from_file_location(
    "psl", REPO / "scripts/phase_ah_sleeve.py")
psl = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(psl)

OUT = REPO / "outputs/vBTC_sleeve_horizon"
CYCLES_PER_YEAR = (288 * 365) / 48


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def block_bootstrap_ci(x, stat=_sharpe, block_size=7, n_boot=2000, alpha=0.05, seed=0):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < block_size + 2: return stat(x), stat(x), stat(x)
    rng = np.random.RandomState(seed)
    n = len(x)
    nb = n // block_size + 1
    boots = []
    for _ in range(n_boot):
        starts = rng.randint(0, n - block_size + 1, size=nb)
        blocks = np.concatenate([x[s:s+block_size] for s in starts])[:n]
        boots.append(stat(blocks))
    boots = np.array(boots)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(stat(x)), lo, hi


def main():
    print("=== Phase AH-V3 robustness v2 ===\n", flush=True)

    # Load all 6 per-cycle CSVs
    variants = ["equal_3", "equal_4", "equal_6", "decay_v33_6",
                 "decay_fast_6", "decay_slow_6"]
    per = {}
    for v in variants:
        df = pd.read_csv(OUT / f"per_cycle_robust_{v}.csv")
        df["time"] = pd.to_datetime(df["time"], utc=True)
        per[v] = df

    # ---------- Test 1: Paired per-cycle V3.1 vs V3.3 ----------
    print("--- Test 1: Paired per-cycle bootstrap V3.1 (equal_6) vs V3.3 (decay_v33_6) ---\n",
          flush=True)
    a = per["equal_6"][["time", "fold", "net_pnl_bps"]].rename(
        columns={"net_pnl_bps": "v31"})
    b = per["decay_v33_6"][["time", "fold", "net_pnl_bps"]].rename(
        columns={"net_pnl_bps": "v33"})
    paired = a.merge(b, on=["time", "fold"], how="inner")
    paired["diff"] = paired["v33"] - paired["v31"]
    diff = paired["diff"].to_numpy()
    print(f"  cycles aligned: {len(paired)}", flush=True)
    print(f"  mean V3.1 = {paired['v31'].mean():+.3f} bps/cycle  "
          f"(Sharpe {_sharpe(paired['v31']):+.2f})", flush=True)
    print(f"  mean V3.3 = {paired['v33'].mean():+.3f} bps/cycle  "
          f"(Sharpe {_sharpe(paired['v33']):+.2f})", flush=True)
    print(f"  mean diff = {diff.mean():+.3f} bps/cycle  "
          f"(total +{diff.sum():.0f} bps)", flush=True)

    # Bootstrap mean diff
    def _mean(x): return float(np.mean(x))
    mu, lo, hi = block_bootstrap_ci(diff, stat=_mean, block_size=7, n_boot=2000)
    print(f"  bootstrap mean diff CI: {mu:+.3f}  [{lo:+.3f}, {hi:+.3f}]", flush=True)
    diff_significant = (lo > 0) or (hi < 0)
    print(f"  diff significantly nonzero (95% CI excludes 0): "
          f"{'YES' if diff_significant else 'NO'}", flush=True)

    # Bootstrap Sharpe diff
    def _sharpe_diff(d): return _sharpe(d)
    # Sharpe of diff series — a strict paired test
    sh_diff, sh_lo, sh_hi = block_bootstrap_ci(diff, stat=_sharpe,
                                                  block_size=7, n_boot=2000)
    print(f"  bootstrap Sharpe-of-diff: {sh_diff:+.2f}  [{sh_lo:+.2f}, {sh_hi:+.2f}]",
          flush=True)
    print(f"  Sharpe-of-diff significantly nonzero: "
          f"{'YES' if (sh_lo > 0 or sh_hi < 0) else 'NO'}", flush=True)

    # Per-fold diff means
    print(f"\n  Per-fold mean diff:", flush=True)
    for f, g in paired.groupby("fold"):
        d = g["diff"].to_numpy()
        m, l, h = block_bootstrap_ci(d, stat=_mean, block_size=5, n_boot=500)
        sign = "+" if m > 0 else "-"
        sig = "*" if (l > 0 or h < 0) else " "
        print(f"    fold {int(f)}:  mean diff = {m:+.3f} [{l:+.3f}, {h:+.3f}]  {sig}",
              flush=True)

    # ---------- Test 2: Restricted 2-variant nested ----------
    print(f"\n--- Test 2: Restricted 2-variant nested {{equal_6, decay_v33_6}} ---\n",
          flush=True)
    fold_sharpe_2 = {"equal_6": {}, "decay_v33_6": {}}
    for v in fold_sharpe_2:
        for f, g in per[v].groupby("fold"):
            fold_sharpe_2[v][int(f)] = _sharpe(g["net_pnl_bps"])
    folds_sorted = sorted(set(int(f) for d in fold_sharpe_2.values() for f in d))

    nested_rows = []
    selections = []
    for f in folds_sorted:
        past = [pf for pf in folds_sorted if pf < f]
        if not past:
            sel = "equal_6"; reason = "fold 1 default"
        else:
            scores = {v: float(np.mean([fold_sharpe_2[v][pf] for pf in past
                                        if pf in fold_sharpe_2[v]]))
                       for v in fold_sharpe_2}
            sel = max(scores, key=lambda k: scores[k])
            reason = f"past mean Sh {scores[sel]:+.2f}"
        sel_df = per[sel]
        fold_slice = sel_df[sel_df["fold"] == f].copy()
        fold_slice["selected_variant"] = sel
        nested_rows.append(fold_slice)
        realized = fold_sharpe_2[sel].get(f, np.nan)
        selections.append({"fold": f, "selected": sel, "reason": reason,
                            "realized_sharpe": realized})
        print(f"  fold {f}:  selected {sel:<14}  ({reason})  → realized Sh = {realized:+.2f}",
              flush=True)

    nested_df = pd.concat(nested_rows, ignore_index=True)
    nested_net = nested_df["net_pnl_bps"].to_numpy()
    nested_sh = _sharpe(nested_net)
    nested_dd = _max_dd(nested_net)
    nested_npos = sum(1 for _, g in nested_df.groupby("fold")
                       if _sharpe(g["net_pnl_bps"]) > 0)

    v33_sh = _sharpe(per["decay_v33_6"]["net_pnl_bps"])
    v31_sh = _sharpe(per["equal_6"]["net_pnl_bps"])
    print(f"\n  V3.1 static (equal_6)        Sharpe = {v31_sh:+.2f}", flush=True)
    print(f"  V3.3 static (decay_v33_6)    Sharpe = {v33_sh:+.2f}", flush=True)
    print(f"  Restricted-nested Sharpe     = {nested_sh:+.2f}  "
          f"({nested_npos}/{len(folds_sorted)} folds, maxDD {nested_dd:+.0f}, "
          f"PnL {nested_net.sum():+.0f})", flush=True)
    print(f"  Δ (nested - V3.3)            = {nested_sh - v33_sh:+.2f}", flush=True)
    print(f"  Δ (nested - V3.1)            = {nested_sh - v31_sh:+.2f}", flush=True)

    # ---------- Test 3: Per-variant per-fold Sharpe CIs ----------
    print(f"\n--- Test 3: Per-variant per-fold Sharpe with 95% block-bootstrap CI ---\n",
          flush=True)
    print(f"  {'fold':>4}", end="")
    for v in ["equal_6", "decay_v33_6"]:
        print(f"  {v:<25}", end="")
    print()
    fold_compare = []
    for f in folds_sorted:
        row = f"  {f:>4}"
        fold_row = {"fold": f}
        for v in ["equal_6", "decay_v33_6"]:
            x = per[v][per[v]["fold"] == f]["net_pnl_bps"].to_numpy()
            sh, lo, hi = block_bootstrap_ci(x, stat=_sharpe, block_size=5, n_boot=1000)
            row += f"  {sh:+5.2f}[{lo:+5.2f},{hi:+5.2f}] "
            fold_row[f"{v}_sh"] = sh; fold_row[f"{v}_lo"] = lo; fold_row[f"{v}_hi"] = hi
        fold_compare.append(fold_row)
        print(row, flush=True)

    # Count folds where V3.3 CI lies strictly above V3.1 CI (non-overlapping)
    n_v33_clearly_wins = sum(1 for r in fold_compare
                              if r["decay_v33_6_lo"] > r["equal_6_hi"])
    n_v31_clearly_wins = sum(1 for r in fold_compare
                              if r["equal_6_lo"] > r["decay_v33_6_hi"])
    print(f"\n  Folds where V3.3 CI > V3.1 CI (non-overlapping): {n_v33_clearly_wins}/9",
          flush=True)
    print(f"  Folds where V3.1 CI > V3.3 CI (non-overlapping): {n_v31_clearly_wins}/9",
          flush=True)

    # ---------- Test 4: V3.1 matched-basket placebo (equal weights) ----------
    print(f"\n--- Test 4: V3.1 equal_6 matched-basket placebo (100 seeds, equal weights) ---\n",
          flush=True)
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

    apd_full = pd.read_parquet(psl.APD_PATH)
    apd_full["open_time"] = pd.to_datetime(apd_full["open_time"], utc=True)
    apd_full["exit_time"] = pd.to_datetime(apd_full["exit_time"], utc=True)
    listings = psl.get_listings()
    def elig_at(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=psl.MIN_HISTORY_DAYS)
        return {s for s in all_syms if listings.get(s) and listings[s] <= cutoff}
    tgt = sorted(apd_full[apd_full["fold"].isin(svar.OOS_FOLDS)]["open_time"].unique())
    sampled = tgt[::svar.HORIZON_ENTRY]
    universe = psl.build_rolling_ic_universe(apd_full, sampled, psl.TOP_N, elig_at)

    N_SEEDS = 100
    eq6 = [1/6] * 6
    df_real = svar.aggregate_sleeves_variant(
        records, fwd_rets_4h, 6, 288, sleeve_weights=eq6,
        placebo_universe=None, placebo_seed=None)
    real_sh = _sharpe(df_real["net_pnl_bps"])
    print(f"  V3.1 real Sharpe = {real_sh:+.3f}", flush=True)

    placebo_rows = []
    t0 = time.time()
    for seed in range(N_SEEDS):
        df_p = svar.aggregate_sleeves_variant(
            records, fwd_rets_4h, 6, 288, sleeve_weights=eq6,
            placebo_universe=universe, placebo_seed=seed)
        placebo_rows.append({"seed": seed,
                              "sharpe": _sharpe(df_p["net_pnl_bps"])})
        if (seed + 1) % 25 == 0:
            print(f"  ... {seed+1}/{N_SEEDS}  ({time.time()-t0:.0f}s)", flush=True)
    pdf = pd.DataFrame(placebo_rows)
    pdf.to_csv(OUT / "matched_placebo_V3.1.csv", index=False)
    p_sh = pdf["sharpe"].to_numpy()
    p95 = float(np.percentile(p_sh, 95))
    rank = float((p_sh < real_sh).mean() * 100)
    print(f"\n  Placebo: mean={p_sh.mean():+.2f}  p50={np.median(p_sh):+.2f}  "
          f"p95={p95:+.2f}  max={p_sh.max():+.2f}", flush=True)
    print(f"  V3.1 ({real_sh:+.2f}) ranks p{rank:.0f}  "
          f"beats_p95={'PASS' if real_sh > p95 else 'FAIL'}", flush=True)
    print(f"  V3.1 beats {(p_sh < real_sh).sum()} / {N_SEEDS} placebos", flush=True)

    # ---------- Verdict ----------
    print(f"\n=== Verdict ===\n", flush=True)
    print(f"  V3.1 equal_6 :  Sharpe +{v31_sh:.2f} (no tunable weights, structurally clean)",
          flush=True)
    print(f"                  paired placebo rank p{rank:.0f}  beats_p95={'PASS' if real_sh > p95 else 'FAIL'}",
          flush=True)
    print(f"  V3.3 decay_v33_6:  Sharpe +{v33_sh:.2f} (decay weights, in-sample tuned)",
          flush=True)
    print(f"  Paired V3.3-V3.1 mean diff CI: [{lo:+.3f}, {hi:+.3f}]  "
          f"{'real' if (lo > 0 or hi < 0) else 'in noise'}", flush=True)
    print(f"  Restricted 2-variant nested: {nested_sh:+.2f}  "
          f"(Δ vs V3.1 {nested_sh - v31_sh:+.2f})", flush=True)
    print(f"\n  Decisive evidence summary:", flush=True)
    print(f"    Mean diff V3.3-V3.1   : statistically {'SIG' if (lo > 0 or hi < 0) else 'NOT-SIG'}",
          flush=True)
    print(f"    Folds V3.3 CI > V3.1  : {n_v33_clearly_wins}/9", flush=True)
    print(f"    2-var nested vs V3.3  : Δ {nested_sh - v33_sh:+.2f}", flush=True)
    print(f"    2-var nested vs V3.1  : Δ {nested_sh - v31_sh:+.2f}", flush=True)


if __name__ == "__main__":
    main()
