"""Step 17: Random-pick placebo — is the architecture alone the source of alpha?

For each cycle, instead of ranking by pred, RANDOMLY pick K=3 long + K=3 short
from the 15-symbol rolling-IC universe. Same gates, same V3.1 sleeve, β-hedged
MTM. Run 100 seeds. If mean Sharpe ≈ +0.74, the model contributes essentially
zero alpha — everything comes from architecture.

Uses psl.run_production_protocol_save_sleeves with placebo_seed parameter
(already implemented for matched-basket placebo testing).
"""
from __future__ import annotations
import sys, time, importlib.util, warnings
from pathlib import Path
from collections import deque, defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("psl", REPO / "scripts/phase_ah_sleeve.py")
psl = importlib.util.module_from_spec(spec); spec.loader.exec_module(psl)
from ml.research.alpha_v4_xs import block_bootstrap_ci

R7_PREDS = REPO / "linear_model/results/ridge_r7_preds.parquet"
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
OUT = REPO / "linear_model/results"

OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
CAPITAL = 100.0
N_SEEDS = 100


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(psl.CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def get_listings():
    L = {}
    for d in KLINES_DIR.iterdir():
        if not d.is_dir(): continue
        m5 = d / "5m"
        if not m5.exists(): continue
        f = sorted(m5.glob("*.parquet"))
        if not f: continue
        try: L[d.name] = pd.Timestamp(f[0].stem, tz="UTC")
        except Exception: pass
    return L


def aggregate(records, alpha_wide):
    sleeve_queue = deque(maxlen=psl.N_SLEEVES)
    prev_weights = {}
    rows = []
    for _, rec in records.iterrows():
        t = rec["time"]; fold = rec["fold"]
        if rec["traded"]:
            sleeve_queue.append({"longs":list(rec["long_basket"]),
                                  "shorts":list(rec["short_basket"])})
        else:
            sleeve_queue.append({"longs":[],"shorts":[]})
        tw = defaultdict(float)
        sw = 1.0 / psl.N_SLEEVES
        for sl in sleeve_queue:
            nL, nS = len(sl["longs"]), len(sl["shorts"])
            if nL == 0 or nS == 0: continue
            for s in sl["longs"]: tw[s] += sw * (1.0/nL)
            for s in sl["shorts"]: tw[s] -= sw * (1.0/nS)
        gross = 0.0
        if t in alpha_wide.index:
            a = alpha_wide.loc[t]
            for sym, w in prev_weights.items():
                if sym in a.index and not pd.isna(a[sym]):
                    gross += w * a[sym] * 1e4
        syms = set(tw.keys()) | set(prev_weights.keys())
        abs_d = sum(abs(tw.get(s,0)-prev_weights.get(s,0)) for s in syms)
        cost = abs_d * psl.COST_PER_UNIT_ABS_DELTA
        rows.append({"time":t,"fold":fold,"gross_pnl_bps":gross,"cost_bps":cost,
                     "net_pnl_bps":gross-cost,"turnover":abs_d})
        prev_weights = dict(tw)
    return pd.DataFrame(rows)


def main():
    print("=== Step 17: Random-pick placebo (no model) ===\n", flush=True)
    t0 = time.time()
    listings = get_listings()

    # Load R7 predictions — needed only for universe construction (uses pred_z)
    apd = pd.read_parquet(R7_PREDS)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    apd["alpha_A"] = apd["alpha_beta"]
    base = pd.read_parquet(PANEL, columns=["symbol","open_time","return_pct"])
    base["open_time"] = pd.to_datetime(base["open_time"], utc=True)
    if "return_pct" not in apd.columns:
        apd = apd.merge(base, on=["symbol","open_time"], how="left")
    apd["pred"] = apd["pred_z"]
    print(f"R7 preds: {len(apd):,} rows", flush=True)

    panel_syms = set(apd["symbol"].unique())
    for s, t in apd.groupby("symbol")["open_time"].min().items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t

    def elig_pit(b):
        ts = b if isinstance(b, pd.Timestamp) else pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]

    # Build universe ONCE using R7 pred_z
    universe = psl.build_rolling_ic_universe(apd, sampled_t, psl.TOP_N, elig_pit)
    alpha_wide = apd.pivot_table(index="open_time", columns="symbol",
                                   values="alpha_A", aggfunc="first").sort_index()
    print(f"Universe built: {len(universe)} time points (top-15 by rolling-IC of R7 pred_z)",
          flush=True)

    # ===== Real run (R7 pred_z baseline rank) =====
    print(f"\n--- Real R7 baseline (rank by pred_z, no IC weighting) ---", flush=True)
    records_real = psl.run_production_protocol_save_sleeves(apd, universe,
                                                              placebo_seed=None)
    df_v_real = aggregate(records_real, alpha_wide)
    sh_real = _sharpe(df_v_real["net_pnl_bps"].to_numpy())
    print(f"  R7 baseline Sharpe: {sh_real:+.2f}", flush=True)

    # ===== Random pick placebo =====
    print(f"\n--- Random-pick placebo ({N_SEEDS} seeds, random K=3 long+short from universe) ---",
          flush=True)
    placebo_sharpes = []
    for seed in range(N_SEEDS):
        records_p = psl.run_production_protocol_save_sleeves(apd, universe,
                                                               placebo_seed=seed)
        df_v_p = aggregate(records_p, alpha_wide)
        sh_p = _sharpe(df_v_p["net_pnl_bps"].to_numpy())
        placebo_sharpes.append(sh_p)
        if (seed + 1) % 20 == 0:
            print(f"  seed {seed+1}/{N_SEEDS}: cumulative mean={np.mean(placebo_sharpes):+.3f}",
                  flush=True)

    placebo_sharpes = np.array(placebo_sharpes)
    p5, p25, p50, p75, p95 = np.percentile(placebo_sharpes, [5, 25, 50, 75, 95])
    print(f"\nRandom-pick placebo distribution (100 seeds):", flush=True)
    print(f"  mean: {placebo_sharpes.mean():+.3f}  std: {placebo_sharpes.std():.3f}",
          flush=True)
    print(f"  p5/p25/p50/p75/p95: {p5:+.2f} / {p25:+.2f} / {p50:+.2f} / "
          f"{p75:+.2f} / {p95:+.2f}", flush=True)
    print(f"  min/max: {placebo_sharpes.min():+.2f} / {placebo_sharpes.max():+.2f}",
          flush=True)

    print(f"\n--- COMPARISONS ---", flush=True)
    print(f"  Random pick mean (this test):              {placebo_sharpes.mean():+.2f}",
          flush=True)
    print(f"  R7 baseline rank (this test):              {sh_real:+.2f}", flush=True)
    print(f"  Earlier trail_ic-shuffle placebo mean:     +0.75", flush=True)
    print(f"  R3 + IC-signed (validated linear best):    +0.15", flush=True)
    print(f"  R7 + IC-signed (failed p95 test):          +1.60", flush=True)
    print(f"  LGBM production:                            +0.74", flush=True)

    # Interpretation
    arch_floor = placebo_sharpes.mean()
    print(f"\n--- INTERPRETATION ---", flush=True)
    print(f"  Architecture floor (random pick):  {arch_floor:+.2f}", flush=True)
    if arch_floor > 0.5:
        print(f"  → The pipeline produces ~{arch_floor:+.2f} Sharpe with NO model at all",
              flush=True)
        print(f"  → Models add: R7-baseline={sh_real-arch_floor:+.2f}, "
              f"R3+IC-signed={0.15-arch_floor:+.2f}, LGBM={0.74-arch_floor:+.2f}",
              flush=True)
    pd.DataFrame({"placebo_sharpe": placebo_sharpes}).to_csv(
        OUT / "random_pick_placebo.csv", index=False)
    print(f"\nSaved: {OUT / 'random_pick_placebo.csv'}", flush=True)
    print(f"Total: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
