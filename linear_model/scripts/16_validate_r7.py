"""Step 16: Validate R7 + IC-signed Sharpe +1.60 finding.

Three tests:
  1. Per-fold LOFO: exclude each fold and check if Sharpe collapses
  2. Matched placebo: shuffle trail_ic randomly across (sym, t) and rerank
     (100 seeds). If random trail_ic produces similar Sharpe, the IC mechanism
     is selection-via-noise, not signal.
  3. Time-shuffled IC placebo: take trail_ic and shift it forward in time so
     each cycle uses a future cycle's IC. If this also produces +1.60 Sharpe,
     trail_ic is not the active mechanism.
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
TRAILING_IC_DAYS = 90
N_PLACEBO_SEEDS = 100


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(psl.CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


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


def compute_trailing_ic(apd, sampled_t, win_days=90):
    apd_s = apd[apd["open_time"].isin(set(sampled_t))].sort_values(
        ["symbol","open_time"]).reset_index(drop=True)
    cycles_per_day = 6
    win_cycles = win_days * cycles_per_day
    ic_records = []
    for sym, g in apd_s.groupby("symbol"):
        g = g.sort_values("open_time").reset_index(drop=True)
        pred = g["pred_z"].to_numpy(); alpha = g["alpha_beta"].to_numpy()
        n = len(g)
        ics = np.full(n, np.nan)
        for i in range(50, n):
            lo = max(0, i - win_cycles)
            p, a = pred[lo:i], alpha[lo:i]
            mask = ~np.isnan(p) & ~np.isnan(a)
            if mask.sum() < 50: continue
            pr = pd.Series(p[mask]).rank().to_numpy()
            ar = pd.Series(a[mask]).rank().to_numpy()
            if pr.std() < 1e-6 or ar.std() < 1e-6: continue
            ics[i] = np.corrcoef(pr, ar)[0,1]
        for j, t in enumerate(g["open_time"]):
            ic_records.append({"symbol":sym, "open_time":t, "trail_ic": ics[j]})
    return pd.DataFrame(ic_records).fillna(0)


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


def run_one(apd_full, sampled_t, universe, alpha_wide, ranking_col):
    apd_v = apd_full.copy()
    apd_v["pred"] = apd_v[ranking_col]
    records = psl.run_production_protocol_save_sleeves(apd_v, universe)
    return aggregate(records, alpha_wide), records


def main():
    print("=== Step 16: Validate R7 + IC-signed ===\n", flush=True)
    t0 = time.time()
    listings = get_listings()

    apd = pd.read_parquet(R7_PREDS)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    apd["alpha_A"] = apd["alpha_beta"]
    # return_pct needed for production protocol
    base = pd.read_parquet(PANEL, columns=["symbol","open_time","return_pct"])
    base["open_time"] = pd.to_datetime(base["open_time"], utc=True)
    if "return_pct" not in apd.columns:
        apd = apd.merge(base, on=["symbol","open_time"], how="left")
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
    print(f"Sampled cycles: {len(sampled_t)}", flush=True)

    print("\nComputing trailing IC...", flush=True)
    df_ic = compute_trailing_ic(apd, sampled_t, TRAILING_IC_DAYS)
    print(f"  trail_ic mean={df_ic['trail_ic'].mean():+.4f}, "
          f"std={df_ic['trail_ic'].std():.4f}", flush=True)

    apd_full = apd.merge(df_ic, on=["symbol","open_time"], how="left")
    apd_full["trail_ic"] = apd_full["trail_ic"].fillna(0)
    apd_full["pred_A"] = apd_full["pred_z"]
    apd_full["pred_B"] = apd_full["pred_z"] * apd_full["trail_ic"]

    apd_full["pred"] = apd_full["pred_z"]
    universe = psl.build_rolling_ic_universe(apd_full, sampled_t, psl.TOP_N, elig_pit)
    alpha_wide = apd_full.pivot_table(index="open_time", columns="symbol",
                                        values="alpha_A", aggfunc="first").sort_index()

    # ===== Test 1: Reproduce headline result =====
    print("\n--- Test 1: Reproduce R7 + IC-signed Sharpe ---", flush=True)
    df_v_B, _ = run_one(apd_full, sampled_t, universe, alpha_wide, "pred_B")
    net = df_v_B["net_pnl_bps"].to_numpy()
    sh_B, lo_B, hi_B = block_bootstrap_ci(net, statistic=_sharpe,
                                            block_size=7, n_boot=1000)
    print(f"  Sharpe={sh_B:+.2f} [{lo_B:+.2f},{hi_B:+.2f}], "
          f"end-eq=${CAPITAL+net.sum()/1e4*CAPITAL:.2f}, "
          f"folds+={folds_positive(df_v_B)}/9", flush=True)

    # ===== Test 2: LOFO (Leave One Fold Out) =====
    print("\n--- Test 2: LOFO — exclude each fold and check delta ---", flush=True)
    sh_all = _sharpe(net)
    print(f"  Reference R7+IC-signed all 9 folds: Sharpe = {sh_all:+.2f}", flush=True)
    print(f"  {'exclude':>9}  {'Sharpe-rem':>10}  {'Δ':>7}  {'fold pnl':>10}",
          flush=True)
    for excl in range(1, 10):
        rem = df_v_B[df_v_B["fold"] != excl]["net_pnl_bps"].to_numpy()
        fold_pnl = df_v_B[df_v_B["fold"] == excl]["net_pnl_bps"].sum()
        sh_rem = _sharpe(rem)
        delta = sh_rem - sh_all
        flag = "  ← lift would collapse" if delta < -0.5 else ""
        print(f"  {excl:>9}  {sh_rem:>+10.2f}  {delta:>+7.2f}  {fold_pnl:>+10.0f}{flag}",
              flush=True)

    # ===== Test 3: Matched placebo — shuffle trail_ic =====
    print(f"\n--- Test 3: Matched placebo (shuffle trail_ic across syms × time) ---",
          flush=True)
    print(f"  Running {N_PLACEBO_SEEDS} placebo seeds...", flush=True)
    placebo_sharpes = []
    rng_master = np.random.default_rng(42)
    apd_p = apd_full.copy()
    real_trail_ic = apd_p["trail_ic"].values.copy()
    for seed in range(N_PLACEBO_SEEDS):
        # Shuffle trail_ic randomly (breaks the sym-pred relationship)
        rng = np.random.default_rng(rng_master.integers(0, 2**31))
        shuffled_ic = rng.permutation(real_trail_ic)
        apd_p["trail_ic_p"] = shuffled_ic
        apd_p["pred_placebo"] = apd_p["pred_z"] * apd_p["trail_ic_p"]
        df_v_p, _ = run_one(apd_p, sampled_t, universe, alpha_wide, "pred_placebo")
        sh_p = _sharpe(df_v_p["net_pnl_bps"].to_numpy())
        placebo_sharpes.append(sh_p)
        if (seed + 1) % 20 == 0:
            print(f"    seed {seed+1}/{N_PLACEBO_SEEDS}: "
                  f"current mean = {np.mean(placebo_sharpes):+.3f}", flush=True)

    placebo_sharpes = np.array(placebo_sharpes)
    p25, p50, p75, p95, p99 = np.percentile(placebo_sharpes, [25, 50, 75, 95, 99])
    rank_pct = (placebo_sharpes < sh_B).mean() * 100
    print(f"\n  Placebo distribution:", flush=True)
    print(f"    mean: {placebo_sharpes.mean():+.3f}  std: {placebo_sharpes.std():.3f}",
          flush=True)
    print(f"    p25/p50/p75: {p25:+.2f} / {p50:+.2f} / {p75:+.2f}", flush=True)
    print(f"    p95/p99: {p95:+.2f} / {p99:+.2f}", flush=True)
    print(f"    max: {placebo_sharpes.max():+.2f}", flush=True)
    print(f"\n  R7+IC-signed real Sharpe: {sh_B:+.2f}", flush=True)
    print(f"  Real rank vs placebo: {rank_pct:.1f}% "
          f"({'PASS p95' if rank_pct >= 95 else 'FAIL p95 — selection-via-noise likely'})",
          flush=True)
    print(f"  Edge over placebo p95: {sh_B - p95:+.2f}", flush=True)

    # Save placebo distribution
    pd.DataFrame({"placebo_sharpe": placebo_sharpes}).to_csv(
        OUT / "r7_ic_signed_placebo.csv", index=False)
    print(f"\n  Saved placebo distribution: {OUT / 'r7_ic_signed_placebo.csv'}",
          flush=True)
    print(f"  Total: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
