"""Phase UNI: universe stress test — drop random symbols and re-run V3.1.

Tests if V3.1 performance is universe-overfit by dropping K random symbols from
the 51-panel at each cycle, recomputing rolling-IC top-15 on the reduced panel,
and re-running the production protocol + V3.1 aggregation.

For each K_drop ∈ {5, 10, 15, 20}, run N_DRAWS = 30 random drops.
Compare Sharpe distribution to baseline +2.23.

Decision rule:
  - If perturbed Sharpe mean ≈ +2.0 with std < 0.3: strategy is universe-robust
  - If mean drops materially or std > 0.5: universe-sensitive (overfit)

No model retraining — uses existing predictions, just filters the universe.
"""
from __future__ import annotations
import sys, time, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location(
    "psl", REPO / "scripts/phase_ah_sleeve.py")
psl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(psl)
spec2 = importlib.util.spec_from_file_location(
    "svar", REPO / "scripts/phase_ah_sleeve_variants.py")
svar = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(svar)

OUT = REPO / "outputs/vBTC_universe_stress"
OUT.mkdir(parents=True, exist_ok=True)

HORIZON = 48
HOLD_BARS = 288
N_SLEEVES = 6
OOS_FOLDS = list(range(1, 10))
N_DRAWS = 30
CYCLES_PER_YEAR = (288 * 365) / 48


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def run_one_universe(apd, fwd_rets_4h, drop_syms, listings):
    """Drop given symbols, rebuild universe, run V3.1, return Sharpe."""
    apd_f = apd[~apd["symbol"].isin(drop_syms)].copy()

    def elig_at(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=psl.MIN_HISTORY_DAYS)
        return {s for s in apd_f["symbol"].unique()
                  if listings.get(s) and listings[s] <= cutoff}
    tgt = sorted(apd_f[apd_f["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled = tgt[::HORIZON]
    universe = psl.build_rolling_ic_universe(apd_f, sampled, psl.TOP_N, elig_at)
    sleeves = psl.run_production_protocol_save_sleeves(apd_f, universe)
    sleeves["time"] = pd.to_datetime(sleeves["time"], utc=True)
    df_v = svar.aggregate_sleeves_variant(sleeves, fwd_rets_4h, N_SLEEVES,
                                                HOLD_BARS, sleeve_weights=[1/6]*6)
    net = df_v["net_pnl_bps"].to_numpy()
    sh = _sharpe(net)
    dd = _max_dd(net)
    npos = sum(1 for _, g in df_v.groupby("fold") if _sharpe(g["net_pnl_bps"]) > 0)
    return sh, dd, net.sum(), npos, int(sleeves["traded"].sum())


def main():
    print("=== Phase UNI: universe stress test (drop random symbols) ===\n", flush=True)

    apd = pd.read_parquet(REPO / "outputs/vBTC_audit_panel/all_predictions.parquet")
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    all_syms = sorted(apd["symbol"].unique())
    print(f"  loaded {len(apd):,} predictions, {len(all_syms)} symbols", flush=True)

    listings = psl.get_listings()
    panel_first_obs = apd.groupby("symbol")["open_time"].min()
    for sym, t in panel_first_obs.items():
        if sym not in listings:
            listings[sym] = t

    print(f"  loading close prices...", flush=True)
    t0 = time.time()
    close_wide = svar.load_close_wide(all_syms)
    fwd_rets_4h = (close_wide.shift(-HORIZON) - close_wide) / close_wide
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    # Baseline (no drop)
    print(f"\n  Baseline (no drop):", flush=True)
    t0 = time.time()
    base_sh, base_dd, base_pnl, base_pos, base_tr = run_one_universe(
        apd, fwd_rets_4h, drop_syms=set(), listings=listings)
    print(f"  baseline Sharpe={base_sh:+.3f}  maxDD={base_dd:+.0f}  "
          f"PnL={base_pnl:+.0f}  folds+={base_pos}/9  traded={base_tr}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    # Stress test
    rng = np.random.RandomState(42)
    results = []
    drop_counts = [5, 10, 15, 20]

    for K in drop_counts:
        print(f"\n  === Dropping K={K} random symbols ({N_DRAWS} draws) ===",
              flush=True)
        sharpes = []; pnls = []; dds = []; npos_list = []
        for draw in range(N_DRAWS):
            t0 = time.time()
            dropped = set(rng.choice(all_syms, size=K, replace=False).tolist())
            try:
                sh, dd, pnl, npos, ntr = run_one_universe(
                    apd, fwd_rets_4h, drop_syms=dropped, listings=listings)
                sharpes.append(sh); pnls.append(pnl); dds.append(dd); npos_list.append(npos)
                results.append({"K_drop": K, "draw": draw,
                                  "sharpe": sh, "maxDD": dd, "pnl": pnl,
                                  "folds_pos": npos, "n_traded": ntr,
                                  "dropped": sorted(dropped)})
                if (draw + 1) % 5 == 0 or draw == 0:
                    print(f"    draw {draw+1:>3d}/{N_DRAWS}  "
                          f"Sharpe={sh:+.2f}  PnL={pnl:+.0f}  "
                          f"({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                print(f"    draw {draw+1}: ERROR {e}", flush=True)
                continue

        sharpes = np.array(sharpes)
        pnls = np.array(pnls)
        print(f"\n    Summary for K={K}:", flush=True)
        print(f"      Sharpe:   mean={sharpes.mean():+.3f}  std={sharpes.std():.3f}  "
              f"min={sharpes.min():+.3f}  max={sharpes.max():+.3f}", flush=True)
        print(f"      PnL:      mean={pnls.mean():+.0f}  std={pnls.std():.0f}",
              flush=True)
        print(f"      Δ from baseline: mean={sharpes.mean() - base_sh:+.3f}",
              flush=True)
        print(f"      % runs ≥ baseline: {(sharpes >= base_sh).mean()*100:.0f}%",
              flush=True)
        print(f"      % runs ≥ +1.50: {(sharpes >= 1.50).mean()*100:.0f}%",
              flush=True)

    # Save full results
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUT / "universe_stress_results.csv", index=False)

    # Final summary
    print(f"\n=== Final summary ===\n", flush=True)
    print(f"  Baseline V3.1 Sharpe: {base_sh:+.3f}\n", flush=True)
    print(f"  {'K_drop':>6}  {'mean Sh':>+8}  {'std':>5}  {'min':>+6}  {'max':>+6}  "
          f"{'% ≥ base':>9}  {'% ≥ +1.5':>9}", flush=True)
    for K in drop_counts:
        sub = df_results[df_results["K_drop"] == K]["sharpe"].to_numpy()
        if len(sub) == 0: continue
        pct_base = (sub >= base_sh).mean() * 100
        pct_15 = (sub >= 1.50).mean() * 100
        print(f"  {K:>6d}  {sub.mean():>+8.2f}  {sub.std():>5.2f}  "
              f"{sub.min():>+6.2f}  {sub.max():>+6.2f}  "
              f"{pct_base:>8.0f}%  {pct_15:>8.0f}%", flush=True)

    # Verdict
    print(f"\n=== Verdict ===\n", flush=True)
    for K in drop_counts:
        sub = df_results[df_results["K_drop"] == K]["sharpe"].to_numpy()
        if len(sub) == 0: continue
        mean_drop = base_sh - sub.mean()
        if mean_drop < 0.3 and sub.std() < 0.3:
            verdict = "ROBUST"
        elif mean_drop < 0.5 and sub.std() < 0.5:
            verdict = "MODERATELY robust"
        else:
            verdict = "OVERFIT"
        print(f"  K={K:>2d}: Δ from baseline = {-mean_drop:+.2f}, std = {sub.std():.2f}  "
              f"→ {verdict}", flush=True)


if __name__ == "__main__":
    main()
