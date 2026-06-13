"""Step 25: Validate corrected R3 + IC-signed (+0.86) per reviewer's protocol.

Three explicit placebos (reviewer's recommendation):
  P1. Fixed liquidity universe (top-30 by 90d dollar volume, no model) + random picks
      → "pure architecture" floor
  P2. Model-selected universe (R3 rolling-IC top-15) + random picks
      → "architecture given model-filtered universe" floor
  P3. Same traded cycles (R3-determined) + shuffled ranks within cycle
      → tests whether ranks (vs which cycles trade) carry the signal

Plus LOFO and per-fold breakdown.
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

R3_PREDS = REPO / "linear_model/results/ridge_r3_corrected_preds.parquet"
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
OUT = REPO / "linear_model/results"

OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
CAPITAL = 100.0
TRAILING_IC_DAYS = 90
N_PLACEBO = 100
HOLD_BARS = 288


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(psl.CYCLES_PER_YEAR))


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


def build_liquidity_universe(sampled_t, n_top=30):
    """Universe = top-30 by trailing 90d dollar volume from raw klines."""
    print(f"  Loading klines to compute dollar volume...", flush=True)
    # Aggregate kline quote_volume to daily per symbol
    daily_dv = {}
    for sym_dir in KLINES_DIR.iterdir():
        if not sym_dir.is_dir(): continue
        m5 = sym_dir / "5m"
        if not m5.exists(): continue
        files = sorted(m5.glob("*.parquet"))
        if not files: continue
        sym = sym_dir.name
        if sym == "BTCUSDT": continue
        df = pd.concat([pd.read_parquet(f, columns=["open_time","quote_volume"])
                          for f in files], ignore_index=True)
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df["date"] = df["open_time"].dt.floor("1D")
        daily = df.groupby("date")["quote_volume"].sum()
        daily_dv[sym] = daily
    # Combine into wide
    dv_wide = pd.DataFrame(daily_dv).sort_index()
    universe = {}
    for t in sampled_t:
        cutoff_lo = t - pd.Timedelta(days=90)
        win = dv_wide[(dv_wide.index >= cutoff_lo) & (dv_wide.index < t)]
        if len(win) < 10: continue
        avg_dv = win.mean(axis=0).sort_values(ascending=False)
        universe[t] = set(avg_dv.head(n_top).index)
    return universe


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


def main():
    print("=== Step 25: Validate corrected R3 + IC-signed ===\n", flush=True)
    t0 = time.time()
    listings = get_listings()

    apd = pd.read_parquet(R3_PREDS)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    apd["alpha_A"] = apd["alpha_beta"]
    if "return_pct" not in apd.columns:
        base = pd.read_parquet(PANEL, columns=["symbol","open_time","return_pct"])
        base["open_time"] = pd.to_datetime(base["open_time"], utc=True)
        apd = apd.merge(base, on=["symbol","open_time"], how="left")
    apd["pred"] = apd["pred_z"]
    print(f"R3 corrected preds: {len(apd):,} rows", flush=True)

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

    df_ic = compute_trailing_ic(apd, sampled_t, TRAILING_IC_DAYS)
    apd_full = apd.merge(df_ic, on=["symbol","open_time"], how="left")
    apd_full["trail_ic"] = apd_full["trail_ic"].fillna(0)
    apd_full["pred_B"] = apd_full["pred_z"] * apd_full["trail_ic"]
    apd_full["pred"] = apd_full["pred_z"]

    # Universe from R3 pred_z (matches step 24's setup)
    universe_R3 = psl.build_rolling_ic_universe(apd_full, sampled_t, psl.TOP_N, elig_pit)
    alpha_wide = apd_full.pivot_table(index="open_time", columns="symbol",
                                        values="alpha_A", aggfunc="first").sort_index()

    # Build liquidity universe (P1) — top-30 by 90d dollar volume from klines
    print("Building liquidity universe (top-30 by 90d dollar volume)...",
          flush=True)
    universe_liq = build_liquidity_universe(sampled_t, n_top=30)
    print(f"  liquidity universe size: top-30 by 90d volume", flush=True)

    # ===== Test 1: Reproduce R3 + IC-signed =====
    print(f"\n--- Test 1: Reproduce R3 + IC-signed ---", flush=True)
    apd_v = apd_full.copy(); apd_v["pred"] = apd_v["pred_B"]
    records_real = psl.run_production_protocol_save_sleeves(apd_v, universe_R3)
    df_v_real = aggregate_hold_through(records_real, alpha_wide)
    net_real = df_v_real["net_pnl_bps"].to_numpy()
    sh_real = _sharpe(net_real)
    sh_lo, sh_hi = block_bootstrap_ci(net_real, statistic=_sharpe,
                                        block_size=7, n_boot=1000)[1:]
    print(f"  R3 + IC-signed (prod aggregator): Sharpe = {sh_real:+.2f} "
          f"[{sh_lo:+.2f},{sh_hi:+.2f}]", flush=True)

    # ===== Test 2: LOFO =====
    print(f"\n--- Test 2: LOFO ---", flush=True)
    print(f"  All folds: Sharpe = {sh_real:+.2f}", flush=True)
    print(f"  {'exclude':>9}  {'Sharpe-rem':>10}  {'Δ':>7}  {'fold pnl':>10}",
          flush=True)
    for excl in range(1, 10):
        rem = df_v_real[df_v_real["fold"] != excl]["net_pnl_bps"].to_numpy()
        fold_pnl = df_v_real[df_v_real["fold"] == excl]["net_pnl_bps"].sum()
        sh_rem = _sharpe(rem)
        delta = sh_rem - sh_real
        flag = "  ← drives lift" if delta < -0.4 else ""
        print(f"  {excl:>9}  {sh_rem:>+10.2f}  {delta:>+7.2f}  {fold_pnl:>+10.0f}{flag}",
              flush=True)

    # ===== P1: Liquidity universe + random picks =====
    print(f"\n--- P1: Liquidity universe + random picks ({N_PLACEBO} seeds) ---",
          flush=True)
    p1 = []
    for seed in range(N_PLACEBO):
        records_p = psl.run_production_protocol_save_sleeves(
            apd_full, universe_liq, placebo_seed=seed)
        df_v_p = aggregate_hold_through(records_p, alpha_wide)
        p1.append(_sharpe(df_v_p["net_pnl_bps"].to_numpy()))
        if (seed+1) % 25 == 0:
            print(f"  seed {seed+1}/{N_PLACEBO}: mean={np.mean(p1):+.3f}", flush=True)
    p1 = np.array(p1)
    print(f"  P1 mean: {p1.mean():+.3f}  std: {p1.std():.3f}", flush=True)
    print(f"  P1 p5/p25/p50/p75/p95: {np.percentile(p1,5):+.2f}/"
          f"{np.percentile(p1,25):+.2f}/{np.percentile(p1,50):+.2f}/"
          f"{np.percentile(p1,75):+.2f}/{np.percentile(p1,95):+.2f}", flush=True)
    print(f"  R3+IC-signed rank in P1: {(p1 < sh_real).mean()*100:.1f}%", flush=True)

    # ===== P2: Model universe + random picks =====
    print(f"\n--- P2: R3 model universe + random picks ({N_PLACEBO} seeds) ---",
          flush=True)
    p2 = []
    for seed in range(N_PLACEBO):
        records_p = psl.run_production_protocol_save_sleeves(
            apd_full, universe_R3, placebo_seed=seed)
        df_v_p = aggregate_hold_through(records_p, alpha_wide)
        p2.append(_sharpe(df_v_p["net_pnl_bps"].to_numpy()))
        if (seed+1) % 25 == 0:
            print(f"  seed {seed+1}/{N_PLACEBO}: mean={np.mean(p2):+.3f}", flush=True)
    p2 = np.array(p2)
    print(f"  P2 mean: {p2.mean():+.3f}  std: {p2.std():.3f}", flush=True)
    print(f"  P2 p5/p25/p50/p75/p95: {np.percentile(p2,5):+.2f}/"
          f"{np.percentile(p2,25):+.2f}/{np.percentile(p2,50):+.2f}/"
          f"{np.percentile(p2,75):+.2f}/{np.percentile(p2,95):+.2f}", flush=True)
    print(f"  R3+IC-signed rank in P2: {(p2 < sh_real).mean()*100:.1f}%", flush=True)

    # ===== P3: Same traded cycles + shuffled ranks =====
    print(f"\n--- P3: Same traded cycles + shuffled ranks within cycle ({N_PLACEBO} seeds) ---",
          flush=True)
    # Build apd with shuffled pred_B per cycle
    p3 = []
    apd_p = apd_full.copy()
    rng_master = np.random.default_rng(42)
    for seed in range(N_PLACEBO):
        rng = np.random.default_rng(rng_master.integers(0, 2**31))
        # Shuffle pred_B within each open_time (cycle)
        def shuffle_within(g):
            v = g["pred_B"].to_numpy().copy()
            rng.shuffle(v)
            g = g.copy(); g["pred_shuffled"] = v
            return g
        apd_p2 = apd_p.groupby("open_time", group_keys=False).apply(shuffle_within)
        apd_p2["pred"] = apd_p2["pred_shuffled"]
        records_p = psl.run_production_protocol_save_sleeves(apd_p2, universe_R3)
        df_v_p = aggregate_hold_through(records_p, alpha_wide)
        p3.append(_sharpe(df_v_p["net_pnl_bps"].to_numpy()))
        if (seed+1) % 25 == 0:
            print(f"  seed {seed+1}/{N_PLACEBO}: mean={np.mean(p3):+.3f}", flush=True)
    p3 = np.array(p3)
    print(f"  P3 mean: {p3.mean():+.3f}  std: {p3.std():.3f}", flush=True)
    print(f"  P3 p5/p25/p50/p75/p95: {np.percentile(p3,5):+.2f}/"
          f"{np.percentile(p3,25):+.2f}/{np.percentile(p3,50):+.2f}/"
          f"{np.percentile(p3,75):+.2f}/{np.percentile(p3,95):+.2f}", flush=True)
    print(f"  R3+IC-signed rank in P3: {(p3 < sh_real).mean()*100:.1f}%", flush=True)

    # Summary
    print(f"\n{'='*100}", flush=True)
    print(f"  CORRECTED R3 + IC-SIGNED VALIDATION SUMMARY", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"  Real Sharpe:               {sh_real:+.2f} [{sh_lo:+.2f},{sh_hi:+.2f}]",
          flush=True)
    print(f"  P1 (liquidity univ random):  mean {p1.mean():+.2f}, p95 {np.percentile(p1,95):+.2f}, "
          f"R3 rank {(p1 < sh_real).mean()*100:.0f}%", flush=True)
    print(f"  P2 (R3 univ + random pick):  mean {p2.mean():+.2f}, p95 {np.percentile(p2,95):+.2f}, "
          f"R3 rank {(p2 < sh_real).mean()*100:.0f}%", flush=True)
    print(f"  P3 (R3 univ + shuffled rank): mean {p3.mean():+.2f}, p95 {np.percentile(p3,95):+.2f}, "
          f"R3 rank {(p3 < sh_real).mean()*100:.0f}%", flush=True)

    pd.DataFrame({"P1":p1, "P2":p2, "P3":p3}).to_csv(
        OUT / "r3_corrected_placebos.csv", index=False)
    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
