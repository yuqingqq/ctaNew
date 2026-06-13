"""Step 20: Validate R3_BTC apparent +1.92 Sharpe.

Three tests:
  1. Reproduce R3_BTC + IC-signed Sharpe
  2. Per-fold LOFO: does any single fold drive it?
  3. Random-pick placebo using R3_BTC's universe (100 seeds)
     If random pick mean ≈ +1.92, R3_BTC's predictions don't matter.
     If random pick mean is much lower, R3_BTC has real signal.
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

R3_BTC_PREDS = REPO / "linear_model/results/ridge_r3_btc_preds.parquet"
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
    rows = []
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
            rows.append({"symbol":sym, "open_time":t, "trail_ic":ics[j]})
    return pd.DataFrame(rows).fillna(0)


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
    print("=== Step 20: Validate R3_BTC ===\n", flush=True)
    t0 = time.time()
    listings = get_listings()

    apd = pd.read_parquet(R3_BTC_PREDS)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    apd["alpha_A"] = apd["alpha_beta"]
    if "return_pct" not in apd.columns:
        base = pd.read_parquet(PANEL, columns=["symbol","open_time","return_pct"])
        base["open_time"] = pd.to_datetime(base["open_time"], utc=True)
        apd = apd.merge(base, on=["symbol","open_time"], how="left")
    apd["pred"] = apd["pred_z"]
    print(f"R3_BTC preds: {len(apd):,} rows", flush=True)

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

    universe = psl.build_rolling_ic_universe(apd, sampled_t, psl.TOP_N, elig_pit)
    alpha_wide = apd.pivot_table(index="open_time", columns="symbol",
                                   values="alpha_A", aggfunc="first").sort_index()
    print(f"R3_BTC universe built: {len(universe)} cycles", flush=True)

    # ===== Test 1: Reproduce =====
    print(f"\n--- Test 1: Reproduce R3_BTC results ---", flush=True)
    # Compute trail_ic for pred_B
    df_ic = compute_trailing_ic(apd, sampled_t, TRAILING_IC_DAYS)
    apd_full = apd.merge(df_ic, on=["symbol","open_time"], how="left")
    apd_full["trail_ic"] = apd_full["trail_ic"].fillna(0)
    apd_full["pred_A"] = apd_full["pred_z"]
    apd_full["pred_B"] = apd_full["pred_z"] * apd_full["trail_ic"]

    for label, col in [("A_baseline","pred_A"), ("B_IC_signed","pred_B")]:
        apd_v = apd_full.copy(); apd_v["pred"] = apd_v[col]
        records = psl.run_production_protocol_save_sleeves(apd_v, universe)
        df_v = aggregate(records, alpha_wide)
        net = df_v["net_pnl_bps"].to_numpy()
        sh = _sharpe(net)
        print(f"  R3_BTC {label}: Sharpe={sh:+.2f}, folds+={folds_positive(df_v)}/9",
              flush=True)
        if label == "B_IC_signed":
            df_v_B = df_v
            sh_B = sh

    # ===== Test 2: LOFO =====
    print(f"\n--- Test 2: LOFO on R3_BTC + IC-signed ---", flush=True)
    print(f"  All folds: Sharpe = {sh_B:+.2f}", flush=True)
    print(f"  {'exclude':>9}  {'Sharpe-rem':>10}  {'Δ':>7}  {'fold pnl':>10}",
          flush=True)
    for excl in range(1, 10):
        rem = df_v_B[df_v_B["fold"] != excl]["net_pnl_bps"].to_numpy()
        fold_pnl = df_v_B[df_v_B["fold"] == excl]["net_pnl_bps"].sum()
        sh_rem = _sharpe(rem)
        delta = sh_rem - sh_B
        flag = "  ← drives lift" if delta < -0.5 else ""
        print(f"  {excl:>9}  {sh_rem:>+10.2f}  {delta:>+7.2f}  {fold_pnl:>+10.0f}{flag}",
              flush=True)

    # ===== Test 3: Random-pick placebo using R3_BTC's universe =====
    print(f"\n--- Test 3: Random-pick placebo on R3_BTC's universe ({N_PLACEBO_SEEDS} seeds) ---",
          flush=True)
    placebo_sharpes = []
    apd_pl = apd_full.copy()
    apd_pl["pred"] = apd_pl["pred_z"]  # baseline pred for universe (already used)
    for seed in range(N_PLACEBO_SEEDS):
        records_p = psl.run_production_protocol_save_sleeves(
            apd_pl, universe, placebo_seed=seed)
        df_v_p = aggregate(records_p, alpha_wide)
        sh_p = _sharpe(df_v_p["net_pnl_bps"].to_numpy())
        placebo_sharpes.append(sh_p)
        if (seed + 1) % 20 == 0:
            print(f"  seed {seed+1}/{N_PLACEBO_SEEDS}: mean={np.mean(placebo_sharpes):+.3f}",
                  flush=True)

    placebo_sharpes = np.array(placebo_sharpes)
    p5, p25, p50, p75, p95 = np.percentile(placebo_sharpes, [5, 25, 50, 75, 95])
    rank_pct = (placebo_sharpes < sh_B).mean() * 100
    print(f"\n  Random-pick placebo distribution (R3_BTC universe):", flush=True)
    print(f"    mean: {placebo_sharpes.mean():+.3f}  std: {placebo_sharpes.std():.3f}",
          flush=True)
    print(f"    p5/p25/p50/p75/p95: {p5:+.2f}/{p25:+.2f}/{p50:+.2f}/"
          f"{p75:+.2f}/{p95:+.2f}", flush=True)
    print(f"    min/max: {placebo_sharpes.min():+.2f}/{placebo_sharpes.max():+.2f}",
          flush=True)
    print(f"\n  R3_BTC + IC-signed: {sh_B:+.2f}", flush=True)
    print(f"  Rank in placebo: {rank_pct:.1f}% "
          f"({'PASS p95' if rank_pct >= 95 else 'FAIL p95 — architecture-noise'})",
          flush=True)
    print(f"  Edge over placebo p95: {sh_B - p95:+.2f}", flush=True)
    print(f"  Edge over placebo mean: {sh_B - placebo_sharpes.mean():+.2f}", flush=True)

    pd.DataFrame({"placebo_sharpe": placebo_sharpes}).to_csv(
        OUT / "r3_btc_random_pick_placebo.csv", index=False)
    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
