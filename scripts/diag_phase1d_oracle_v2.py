"""Oracle ceiling for Phase 1D — corrected to measure α-PnL (β-hedged execution).

Prior version (v1) was incorrect: it picked by realized α_β but measured PnL on
raw return, which carried implicit β-bias and gave negative oracle Sharpe.

For an α-residual strategy, the right ceiling is α-PnL: assume β is hedged at
execution, and we earn the realized α-spread. Compare three measurements:

  A. α-PnL oracle on full 51, K=3 picks, no gates
     (theoretical max for a β-hedged execution of this target)
  B. α-PnL noisy model (Phase 1D's pred picks, α-PnL accounting)
     (what Phase 1D would have earned with β-hedged execution)
  C. raw-return oracle on full 51, K=3 picks, no gates
     (absolute ceiling for any return-based strategy with K=3 long-short structure)

Cost: 2K × COST_PER_LEG bps per cycle (full turnover assumed; conservative).

Sample every HORIZON=48 bars (4h), aggregate per-cycle PnL into a Sharpe series.
This bypasses V3.1 sleeve overlay complexity to give a clean per-cycle ceiling.
"""
from __future__ import annotations
import sys, time, importlib.util, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("psl", REPO / "scripts/phase_ah_sleeve.py")
psl = importlib.util.module_from_spec(spec); spec.loader.exec_module(psl)
from ml.research.alpha_v4_xs import block_bootstrap_ci

APD_PATH = REPO / "outputs/vBTC_phase1d_rolling_beta/all_predictions.parquet"
OUT = REPO / "outputs/vBTC_phase1d_oracle_v2"
OUT.mkdir(parents=True, exist_ok=True)

OOS_FOLDS = list(range(1, 10))
K = 3
COST_PER_LEG_BPS = 4.5


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(psl.CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def per_cycle_oracle(apd, sort_key, pnl_key, cycle_cost_bps):
    """At each entry cycle (every HORIZON bars), pick top-K and bot-K by `sort_key`
    (realized α_β or pred), then compute per-cycle PnL using `pnl_key` (α_β or
    return). Returns array of per-cycle net PnL in bps.
    """
    df = apd.sort_values(["open_time", "symbol"]).copy()
    df = df[df["fold"].isin(OOS_FOLDS)]
    times = sorted(df["open_time"].unique())
    keep_t = set(times[::psl.HORIZON_ENTRY])
    df = df[df["open_time"].isin(keep_t)]
    df = df.dropna(subset=[sort_key, pnl_key])

    nets = []
    for t, g in df.groupby("open_time"):
        if len(g) < 2 * K + 1: continue
        scores = g[sort_key].to_numpy()
        pnls = g[pnl_key].to_numpy()
        idx_t = np.argpartition(-scores, K - 1)[:K]
        idx_b = np.argpartition(scores, K - 1)[:K]
        # gross PnL: long top, short bot
        gross_bps = (pnls[idx_t].mean() - pnls[idx_b].mean()) * 1e4
        net_bps = gross_bps - cycle_cost_bps
        nets.append(net_bps)
    return np.array(nets)


def per_cycle_diagnostic(apd):
    """Return per-cycle dispersion stats so we understand what the ceiling means."""
    df = apd.sort_values(["open_time", "symbol"]).copy()
    df = df[df["fold"].isin(OOS_FOLDS)]
    times = sorted(df["open_time"].unique())
    keep_t = set(times[::psl.HORIZON_ENTRY])
    df = df[df["open_time"].isin(keep_t)]
    df = df.dropna(subset=["alpha_A", "return_pct"])

    rows = []
    for t, g in df.groupby("open_time"):
        if len(g) < 2 * K + 1: continue
        a = g["alpha_A"].to_numpy()
        r = g["return_pct"].to_numpy()
        idx_t_a = np.argpartition(-a, K - 1)[:K]
        idx_b_a = np.argpartition(a, K - 1)[:K]
        idx_t_r = np.argpartition(-r, K - 1)[:K]
        idx_b_r = np.argpartition(r, K - 1)[:K]
        rows.append({
            "time": t,
            "alpha_spread_bps": (a[idx_t_a].mean() - a[idx_b_a].mean()) * 1e4,
            "return_spread_bps_oracle_alpha": (r[idx_t_a].mean() - r[idx_b_a].mean()) * 1e4,
            "return_spread_bps_oracle_return": (r[idx_t_r].mean() - r[idx_b_r].mean()) * 1e4,
            "alpha_spread_bps_oracle_return": (a[idx_t_r].mean() - a[idx_b_r].mean()) * 1e4,
        })
    return pd.DataFrame(rows)


def main():
    print("=== Phase 1D oracle ceiling v2 (corrected for β-hedged execution) ===\n",
          flush=True)
    apd = pd.read_parquet(APD_PATH)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    print(f"Loaded {len(apd):,} predictions, {apd['symbol'].nunique()} symbols\n", flush=True)

    cycle_cost = 2 * K * COST_PER_LEG_BPS  # full turnover: K longs + K shorts = 6 legs
    print(f"Cost model: full turnover assumed → {cycle_cost} bps/cycle "
          f"(equivalent to V3.1 sleeve overlay's amortized cost when entirely new picks)\n",
          flush=True)

    # === Diagnostic: per-cycle dispersion ===
    print("=" * 80)
    print("DIAGNOSTIC — per-cycle spread between oracle-K=3 long and short baskets")
    print("=" * 80)
    diag = per_cycle_diagnostic(apd)
    for col in ["alpha_spread_bps", "return_spread_bps_oracle_alpha",
                 "return_spread_bps_oracle_return", "alpha_spread_bps_oracle_return"]:
        s = diag[col]
        print(f"\n  {col}", flush=True)
        print(f"    mean={s.mean():+.1f} bps  median={s.median():+.1f} bps  "
              f"p25={s.quantile(0.25):+.1f}  p75={s.quantile(0.75):+.1f}  "
              f"std={s.std():.1f}  min={s.min():+.1f}  max={s.max():+.1f}",
              flush=True)

    # === A. α-PnL oracle on full 51 ===
    print("\n" + "=" * 80)
    print("A. α-PnL oracle on full 51 (β-hedged execution assumed)")
    print("=" * 80)
    print("    pick top-K/bot-K by realized α_β; PnL = realized α-spread − cost", flush=True)
    nets_A = per_cycle_oracle(apd, sort_key="alpha_A", pnl_key="alpha_A",
                                cycle_cost_bps=cycle_cost)
    sh_A, lo_A, hi_A = block_bootstrap_ci(nets_A, statistic=_sharpe, block_size=7, n_boot=1000)
    print(f"\n  n cycles            : {len(nets_A)}", flush=True)
    print(f"  net mean/cycle      : {nets_A.mean():+.1f} bps", flush=True)
    print(f"  gross mean/cycle    : {nets_A.mean() + cycle_cost:+.1f} bps", flush=True)
    print(f"  Sharpe              : {sh_A:+.2f} [{lo_A:+.2f}, {hi_A:+.2f}]", flush=True)
    print(f"  totPnL              : {nets_A.sum():+.0f} bps", flush=True)
    print(f"  maxDD               : {_max_dd(nets_A):+.0f} bps", flush=True)

    # === B. α-PnL using Phase 1D's actual model preds ===
    print("\n" + "=" * 80)
    print("B. α-PnL with Phase 1D's noisy model picks (β-hedged execution assumed)")
    print("=" * 80)
    print("    pick top-K/bot-K by model pred; PnL = realized α-spread − cost", flush=True)
    nets_B = per_cycle_oracle(apd, sort_key="pred", pnl_key="alpha_A",
                                cycle_cost_bps=cycle_cost)
    sh_B, lo_B, hi_B = block_bootstrap_ci(nets_B, statistic=_sharpe, block_size=7, n_boot=1000)
    print(f"\n  n cycles            : {len(nets_B)}", flush=True)
    print(f"  net mean/cycle      : {nets_B.mean():+.1f} bps", flush=True)
    print(f"  gross mean/cycle    : {nets_B.mean() + cycle_cost:+.1f} bps", flush=True)
    print(f"  Sharpe              : {sh_B:+.2f} [{lo_B:+.2f}, {hi_B:+.2f}]", flush=True)
    print(f"  totPnL              : {nets_B.sum():+.0f} bps", flush=True)

    # === C. raw-return oracle (absolute ceiling for any return-based strategy) ===
    print("\n" + "=" * 80)
    print("C. raw-return oracle on full 51 (absolute ceiling, no β-hedge needed)")
    print("=" * 80)
    print("    pick top-K/bot-K by realized return; PnL = realized return-spread − cost", flush=True)
    nets_C = per_cycle_oracle(apd, sort_key="return_pct", pnl_key="return_pct",
                                cycle_cost_bps=cycle_cost)
    sh_C, lo_C, hi_C = block_bootstrap_ci(nets_C, statistic=_sharpe, block_size=7, n_boot=1000)
    print(f"\n  n cycles            : {len(nets_C)}", flush=True)
    print(f"  net mean/cycle      : {nets_C.mean():+.1f} bps", flush=True)
    print(f"  gross mean/cycle    : {nets_C.mean() + cycle_cost:+.1f} bps", flush=True)
    print(f"  Sharpe              : {sh_C:+.2f} [{lo_C:+.2f}, {hi_C:+.2f}]", flush=True)
    print(f"  totPnL              : {nets_C.sum():+.0f} bps", flush=True)

    # === D. return-PnL with Phase 1D's pred picks (matches Phase 1D actual roughly) ===
    print("\n" + "=" * 80)
    print("D. return-PnL with Phase 1D's pred picks (no β-hedge; ≈ Phase 1D actual)")
    print("=" * 80)
    nets_D = per_cycle_oracle(apd, sort_key="pred", pnl_key="return_pct",
                                cycle_cost_bps=cycle_cost)
    sh_D, lo_D, hi_D = block_bootstrap_ci(nets_D, statistic=_sharpe, block_size=7, n_boot=1000)
    print(f"\n  n cycles            : {len(nets_D)}", flush=True)
    print(f"  net mean/cycle      : {nets_D.mean():+.1f} bps", flush=True)
    print(f"  gross mean/cycle    : {nets_D.mean() + cycle_cost:+.1f} bps", flush=True)
    print(f"  Sharpe              : {sh_D:+.2f} [{lo_D:+.2f}, {hi_D:+.2f}]", flush=True)

    print("\n" + "=" * 80)
    print("  SUMMARY — ceilings on Phase 1D's β-residual setup, K=3 long-short")
    print("=" * 80)
    print(f"  Phase 1D actual (full V3.1 stack, return-MTM):  Sharpe +0.65", flush=True)
    print()
    print(f"  A. α-PnL oracle on α_β (β-hedged exec.):        Sharpe {sh_A:+.2f}", flush=True)
    print(f"  B. α-PnL with noisy preds (β-hedged exec.):     Sharpe {sh_B:+.2f}", flush=True)
    print(f"  C. raw-return oracle (absolute ceiling):        Sharpe {sh_C:+.2f}", flush=True)
    print(f"  D. return-PnL with noisy preds (no β-hedge):    Sharpe {sh_D:+.2f}", flush=True)
    print()
    print(f"  Headroom for β-hedged + improved model = A − B  = {sh_A - sh_B:+.2f} Sharpe", flush=True)
    print(f"  β-hedge value with current model       = B − D  = {sh_B - sh_D:+.2f} Sharpe", flush=True)
    print(f"  Absolute ceiling (perfect return pick) = C       = {sh_C:+.2f} Sharpe", flush=True)

    diag.to_csv(OUT / "per_cycle_diagnostic.csv", index=False)
    np.save(OUT / "nets_A_alpha_oracle.npy", nets_A)
    np.save(OUT / "nets_B_pred_alpha.npy", nets_B)
    np.save(OUT / "nets_C_return_oracle.npy", nets_C)
    np.save(OUT / "nets_D_pred_return.npy", nets_D)
    print(f"\nSaved to {OUT}/", flush=True)


if __name__ == "__main__":
    main()
