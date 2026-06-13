"""V3.3-specific 100-seed matched-basket placebo with decay weights.

Re-runs the matched placebo using V3.3's decay weights [0.30, 0.22, 0.17, 0.13,
0.10, 0.08] and saves to matched_placebo_V3.3.csv. The existing
matched_placebo.csv is V3.1 equal-weight era and does not support the doc's
V3.3 rank claim.
"""
from __future__ import annotations
import sys, time, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

# Import V3.3 aggregation from sleeve_variants
spec = importlib.util.spec_from_file_location(
    "svar", REPO / "scripts/phase_ah_sleeve_variants.py")
svar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(svar)

# Import universe/listings helpers from phase_ah_sleeve
spec2 = importlib.util.spec_from_file_location(
    "psl", REPO / "scripts/phase_ah_sleeve.py")
psl = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(psl)

OUT = REPO / "outputs/vBTC_sleeve_horizon"
N_SEEDS = 100
DECAY_WEIGHTS = [0.30, 0.22, 0.17, 0.13, 0.10, 0.08]
HOLD_BARS = 288  # 24h
N_SLEEVES = 6


def main():
    print("=== V3.3 decay6 matched-basket placebo (100 seeds) ===\n", flush=True)
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

    # Confirm V3.3 real Sharpe first
    df_real = svar.aggregate_sleeves_variant(
        records, fwd_rets_4h, N_SLEEVES, HOLD_BARS,
        sleeve_weights=DECAY_WEIGHTS, placebo_universe=None, placebo_seed=None)
    real_sh = svar._sharpe(df_real["net_pnl_bps"].to_numpy())
    print(f"  V3.3 real Sharpe = {real_sh:+.3f}\n", flush=True)

    rows = []
    t0 = time.time()
    for seed in range(N_SEEDS):
        df_p = svar.aggregate_sleeves_variant(
            records, fwd_rets_4h, N_SLEEVES, HOLD_BARS,
            sleeve_weights=DECAY_WEIGHTS,
            placebo_universe=universe, placebo_seed=seed)
        rows.append({"seed": seed,
                      "sharpe": svar._sharpe(df_p["net_pnl_bps"].to_numpy())})
        if (seed + 1) % 25 == 0:
            print(f"  ... {seed+1}/{N_SEEDS}  ({time.time()-t0:.0f}s)", flush=True)

    pdf = pd.DataFrame(rows)
    pdf.to_csv(OUT / "matched_placebo_V3.3.csv", index=False)

    p_sh = pdf["sharpe"].to_numpy()
    p95 = float(np.percentile(p_sh, 95))
    p99 = float(np.percentile(p_sh, 99))
    rank = float((p_sh < real_sh).mean() * 100)
    print(f"\n  Placebo: mean={p_sh.mean():+.2f}  p50={np.median(p_sh):+.2f}  "
          f"p95={p95:+.2f}  p99={p99:+.2f}  max={p_sh.max():+.2f}", flush=True)
    print(f"  V3.3 ({real_sh:+.2f}) ranks p{rank:.0f}  "
          f"beats_p95={'PASS' if real_sh > p95 else 'FAIL'}", flush=True)
    print(f"  V3.3 beats {(p_sh < real_sh).sum()} / {N_SEEDS} placebos", flush=True)
    print(f"\n  Saved: {OUT / 'matched_placebo_V3.3.csv'}", flush=True)


if __name__ == "__main__":
    main()
