"""Step 43: V2 universe stress test on 110-panel.

Analogous to Step 40 (which was on 51-panel). Drop K random symbols from
110-panel, rebuild rolling-IC universe on remaining (110−K), run V3.1 sleeve.
30 random draws per K ∈ {10, 20, 30, 40}.

Tests whether V2's +2.03 on 110-panel is robust to symbol composition.

Step 40 51-panel reference:
  K=5  mean +1.16  std 0.74  worst -0.99
  K=10 mean +1.28  std 0.83  worst -1.09
  K=15 mean +0.78  std 1.07  worst -1.35
  K=20 mean +0.69  std 1.01  worst -2.07
51-panel baseline: +2.19

110-panel baseline: +2.03 (Step 41 rerun)
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

PANEL_PIT = REPO / "outputs/vBTC_features_btc_only_111_pit/panel_btc_only_111.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
PREDS_DIR = REPO / "linear_model/results/step41_111panel_pit"
OUT = REPO / "linear_model/results"

OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
HOLD_BARS = 288
K_DROPS = [10, 20, 30, 40]
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
    apd_sub["pred"] = apd_sub["pred_z"] * apd_sub["trail_ic"]
    universe = psl.build_rolling_ic_universe(apd_sub, sampled_t, psl.TOP_N, elig_pit)
    alpha_wide = apd_sub.pivot_table(index="open_time", columns="symbol",
                                       values="alpha_A", aggfunc="first").sort_index()
    records = psl.run_production_protocol_save_sleeves(apd_sub, universe)
    df_v = aggregate_hold_through(records, alpha_wide)
    net = df_v["net_pnl_bps"].to_numpy()
    return _sharpe(net), folds_positive(df_v)


def main():
    print("=" * 100, flush=True)
    print("  STEP 43: V2 110-panel UNIVERSE STRESS TEST", flush=True)
    print("=" * 100, flush=True)
    print(f"  K_drop ∈ {K_DROPS}, {N_DRAWS} random draws per K", flush=True)
    print(f"  Baseline: V2 on 110-panel + V3.1 sleeve = +2.03", flush=True)
    t0 = time.time()
    listings = get_listings()

    apd_full = pd.read_parquet(PREDS_DIR / "predictions.parquet")
    apd_full["open_time"] = pd.to_datetime(apd_full["open_time"], utc=True)
    apd_full["alpha_A"] = apd_full["alpha_beta"]
    if "exit_time" not in apd_full.columns:
        extra = pd.read_parquet(PANEL_PIT,
                                  columns=["symbol","open_time","exit_time","return_pct"])
        extra["open_time"] = pd.to_datetime(extra["open_time"], utc=True)
        extra["exit_time"] = pd.to_datetime(extra["exit_time"], utc=True)
        apd_full = apd_full.merge(extra, on=["symbol","open_time"], how="left")

    all_syms = sorted(apd_full["symbol"].unique())
    n_syms = len(all_syms)
    print(f"\nPanel has {n_syms} symbols (BTC excluded upstream)", flush=True)

    results = []
    for K_drop in K_DROPS:
        if n_syms - K_drop < psl.TOP_N + 2:
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
        n_pos = int((shs > 0).sum())
        n_beat_baseline = int((shs > 2.03).sum())
        print(f"  K={K_drop}: mean={mean_sh:+.2f}  std={std_sh:.2f}  "
              f"p5/p50/p95={p5:+.2f}/{p50:+.2f}/{p95:+.2f}  "
              f"min={min_sh:+.2f}  max={max_sh:+.2f}", flush=True)
        print(f"           {n_pos}/{N_DRAWS} positive, "
              f"{n_beat_baseline}/{N_DRAWS} beat baseline +2.03", flush=True)
        results.append({"K_drop": K_drop, "mean": mean_sh, "std": std_sh,
                         "p5": p5, "p50": p50, "p95": p95,
                         "min": min_sh, "max": max_sh,
                         "n_positive": n_pos, "n_beat_baseline": n_beat_baseline})

    print(f"\n{'='*100}", flush=True)
    print(f"  STEP 43 SUMMARY (V2 110-panel vs Step 40 V2 51-panel)", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"  {'K_drop':>8} | {'110 mean':>10} | {'110 std':>8} | {'110 worst':>10} | "
          f"{'51 mean (ref)':>14} | {'51 worst (ref)':>14}", flush=True)
    step40 = {5: (1.16, -0.99), 10: (1.28, -1.09), 15: (0.78, -1.35), 20: (0.69, -2.07)}
    for r in results:
        ref = step40.get(r["K_drop"], (None, None))
        ref_mean = f"{ref[0]:+.2f}" if ref[0] is not None else "—"
        ref_worst = f"{ref[1]:+.2f}" if ref[1] is not None else "—"
        print(f"  {r['K_drop']:>8} | {r['mean']:+10.2f} | {r['std']:>8.2f} | "
              f"{r['min']:+10.2f} | {ref_mean:>14} | {ref_worst:>14}", flush=True)

    pd.DataFrame(results).to_csv(OUT / "step43_110_universe_stress.csv", index=False)
    print(f"\n  Total: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
