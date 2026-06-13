"""Step 40: V2 universe stress test.

Drop K random symbols from 51-panel, rebuild rolling-IC universe on the
remaining (51-K) symbols, run V3.1 sleeve, compute Sharpe.
30 random draws per K ∈ {5, 10, 15, 20}.

Replicates vBTC Phase UNI methodology for Ridge V2 fixed.

LGBM Phase UNI reference (from memory):
  K=5:  mean +1.82  std 0.70  worst +0.21
  K=10: mean +1.44  std 0.86  worst -0.18
  K=15: mean +1.22  std 1.04  worst -0.35
  K=20: mean +0.95  std 1.16  worst -1.40

LGBM baseline +2.23. V2 Ridge baseline +2.19.
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

PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
PREDS_DIR = REPO / "linear_model/results/step34_v1_fixed"
OUT = REPO / "linear_model/results"

OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
HOLD_BARS = 288
K_DROPS = [5, 10, 15, 20]
N_DRAWS = 30
SEED_BASE = 42


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(psl.CYCLES_PER_YEAR))


def folds_positive(df_v):
    return sum(1 for _, g in df_v.groupby("fold") if _sharpe(g["net_pnl_bps"]) > 0)


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


def aggregate_hold_through(records, alpha_wide):
    sleeve_queue = deque(maxlen=psl.N_SLEEVES)
    prev_weights = {}
    bar_freq = pd.Timedelta(minutes=5)
    rows = []
    for _, rec in records.iterrows():
        t = rec["time"]; fold = rec["fold"]
        if rec["traded"]:
            sleeve_queue.append({"entry_time":t, "longs":list(rec["long_basket"]),
                                  "shorts":list(rec["short_basket"])})
        max_age = HOLD_BARS * bar_freq
        sleeve_queue = deque(
            [s for s in sleeve_queue if (t - s["entry_time"]) < max_age],
            maxlen=psl.N_SLEEVES)
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


def run_subset(apd_full, drop_syms, listings):
    """Drop drop_syms from universe, rebuild universe + alpha_wide, run sleeve."""
    apd_sub = apd_full[~apd_full["symbol"].isin(drop_syms)].copy()
    panel_syms = set(apd_sub["symbol"].unique())
    for s, t in apd_sub.groupby("symbol")["open_time"].min().items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t
    def elig_pit(b):
        ts = b if isinstance(b, pd.Timestamp) else pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}
    target_t = sorted(apd_sub[apd_sub["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]
    apd_sub["pred"] = apd_sub["pred_z"] * apd_sub["trail_ic"]  # B_IC_signed
    universe = psl.build_rolling_ic_universe(apd_sub, sampled_t, psl.TOP_N, elig_pit)
    alpha_wide = apd_sub.pivot_table(index="open_time", columns="symbol",
                                       values="alpha_A", aggfunc="first").sort_index()
    records = psl.run_production_protocol_save_sleeves(apd_sub, universe)
    df_v = aggregate_hold_through(records, alpha_wide)
    net = df_v["net_pnl_bps"].to_numpy()
    return _sharpe(net), folds_positive(df_v)


def main():
    print("=" * 100, flush=True)
    print("  STEP 40: V2 fixed UNIVERSE STRESS TEST", flush=True)
    print("=" * 100, flush=True)
    print(f"  K_drop ∈ {K_DROPS}, {N_DRAWS} random draws per K", flush=True)
    print(f"  Baseline: V2 fixed full 51-panel + V3.1 sleeve = +2.19", flush=True)
    print()
    t0 = time.time()
    listings = get_listings()

    apd_full = pd.read_parquet(PREDS_DIR / "v2_fixed_predictions.parquet")
    apd_full["open_time"] = pd.to_datetime(apd_full["open_time"], utc=True)
    apd_full["alpha_A"] = apd_full["alpha_beta"]
    extra = pd.read_parquet(PANEL,
                              columns=["symbol","open_time","exit_time","return_pct"])
    extra["open_time"] = pd.to_datetime(extra["open_time"], utc=True)
    extra["exit_time"] = pd.to_datetime(extra["exit_time"], utc=True)
    if "exit_time" not in apd_full.columns:
        apd_full = apd_full.merge(extra, on=["symbol","open_time"], how="left")

    all_syms = sorted(apd_full["symbol"].unique())
    n_syms = len(all_syms)
    print(f"Panel has {n_syms} symbols", flush=True)

    results = []
    for K_drop in K_DROPS:
        if n_syms - K_drop < psl.TOP_N + K_drop:
            print(f"  Warning: K_drop={K_drop} leaves only {n_syms-K_drop} syms, "
                  f"may have universe issues", flush=True)
        print(f"\n--- K_drop = {K_drop} (universe size = {n_syms - K_drop}) ---",
              flush=True)
        shs = []
        folds_pos_list = []
        for draw in range(N_DRAWS):
            rng = np.random.default_rng(SEED_BASE + K_drop * 1000 + draw)
            drop_syms = set(rng.choice(all_syms, size=K_drop, replace=False))
            sh, fp = run_subset(apd_full, drop_syms, listings.copy())
            shs.append(sh)
            folds_pos_list.append(fp)
            if (draw + 1) % 10 == 0:
                print(f"  draw {draw+1}: mean={np.mean(shs):+.3f}", flush=True)
        shs = np.array(shs)
        mean_sh = float(shs.mean())
        std_sh = float(shs.std())
        p5, p50, p95 = np.percentile(shs, [5, 50, 95])
        min_sh = float(shs.min())
        max_sh = float(shs.max())
        n_pos = (shs > 0).sum()
        n_beat_baseline = (shs > 2.19).sum()
        print(f"  K={K_drop}: mean={mean_sh:+.2f}  std={std_sh:.2f}  "
              f"p5/p50/p95={p5:+.2f}/{p50:+.2f}/{p95:+.2f}  "
              f"min={min_sh:+.2f}  max={max_sh:+.2f}", flush=True)
        print(f"           {n_pos}/{N_DRAWS} positive, "
              f"{n_beat_baseline}/{N_DRAWS} beat baseline +2.19", flush=True)
        results.append({"K_drop":K_drop, "mean":mean_sh, "std":std_sh,
                         "p5":p5, "p50":p50, "p95":p95, "min":min_sh, "max":max_sh,
                         "n_positive":n_pos, "n_beat_baseline":n_beat_baseline,
                         "all_sharpes":shs.tolist()})

    print(f"\n{'='*100}", flush=True)
    print(f"  STEP 40 SUMMARY (Ridge V2 vs LGBM Phase UNI)", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"  {'K_drop':>8} | {'Ridge V2 mean':>14} | {'std':>6} | {'worst':>8} | "
          f"{'LGBM mean':>10} | {'LGBM worst':>10}", flush=True)
    lgbm_ref = {5:(1.82, 0.21), 10:(1.44, -0.18), 15:(1.22, -0.35), 20:(0.95, -1.40)}
    for r in results:
        lgbm = lgbm_ref.get(r["K_drop"], (None, None))
        print(f"  {r['K_drop']:>8} | {r['mean']:>14.2f} | {r['std']:>6.2f} | "
              f"{r['min']:>8.2f} | {lgbm[0]:>10.2f} | {lgbm[1]:>10.2f}", flush=True)

    pd.DataFrame(results).drop(columns=["all_sharpes"]).to_csv(
        OUT / "step40_universe_stress_summary.csv", index=False)
    print(f"\n  Total: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
