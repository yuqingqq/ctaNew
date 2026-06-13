"""Step 50: Causal-aligned PnL aggregator + rerun Step 47/48 on full-PIT V2 110-panel.

Reviewer-identified bug in sleeve aggregator (`aggregate_hold_through`):
  Current (lagged):   gross[t] = prev_weights × alpha[t]
  Causal-immediate:   gross[t] = tw × alpha[t]

Under "decision at close of bar t" convention:
  - alpha[t] = close[t+48]/close[t] - 1 is the forward 4h return starting at decision time t
  - tw[t] are the new weights established at decision time t
  - These weights tw[t] earn alpha[t] over cycle [t, t+48]
  - Old prev_weights should have closed at t (no future return)

The lagged version attributes alpha[t] to OLD weights, effectively letting closed
positions earn one extra cycle of alpha. Reviewer recompute showed +0.22 Sharpe inflation.

This script:
  1. Loads Step 47 predictions (full-PIT V2 110-panel)
  2. Reruns the strategy with the causal-immediate aggregator
  3. Reports causal Sharpe + LOFO
  4. Runs P1+P2 placebos under same causal aggregator
  5. Saves all artifacts to results/step53_predB_universe/
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

PANEL_FULL_PIT = REPO / "outputs/vBTC_features_btc_only_111_full_pit/panel_btc_only_111.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
PREDS_DIR = REPO / "linear_model/results/step47_110_full_pit"
OUT = REPO / "linear_model/results/step53_predB_universe"
OUT.mkdir(parents=True, exist_ok=True)

OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
HOLD_BARS = 288
N_PLACEBO = 100


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


def aggregate_hold_through_causal(records, alpha_wide):
    """CAUSAL-IMMEDIATE aggregator: gross[t] = tw × alpha[t].

    The new weights established at cycle t earn the forward 4h alpha[t]
    over cycle [t, t+48]. Cost charged at t for transitioning prev → tw.
    Closed weights at t earn nothing forward (they earned during their hold).
    """
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
        # CAUSAL: gross uses NEW weights × alpha[t]
        gross = 0.0
        if t in alpha_wide.index:
            a = alpha_wide.loc[t]
            for sym, w in tw.items():
                if sym in a.index and not pd.isna(a[sym]):
                    gross += w * a[sym] * 1e4
        syms = set(tw.keys()) | set(prev_weights.keys())
        abs_d = sum(abs(tw.get(s,0)-prev_weights.get(s,0)) for s in syms)
        cost = abs_d * psl.COST_PER_UNIT_ABS_DELTA
        rows.append({"time":t,"fold":fold,"gross_pnl_bps":gross,"cost_bps":cost,
                     "net_pnl_bps":gross-cost,"turnover":abs_d})
        prev_weights = dict(tw)
    return pd.DataFrame(rows)


def build_liquidity_universe(sampled_t, panel_syms, n_top=30):
    print(f"  Loading kline volumes for {len(panel_syms)} symbols...", flush=True)
    daily_dv = {}
    for sym in panel_syms:
        sym_dir = KLINES_DIR / sym
        if not sym_dir.exists(): continue
        m5 = sym_dir / "5m"
        if not m5.exists(): continue
        files = sorted(m5.glob("*.parquet"))
        if not files: continue
        df = pd.concat([pd.read_parquet(f, columns=["open_time","quote_volume"])
                          for f in files], ignore_index=True)
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df["date"] = df["open_time"].dt.floor("1D")
        daily = df.groupby("date")["quote_volume"].sum()
        daily_dv[sym] = daily
    dv_wide = pd.DataFrame(daily_dv).sort_index()
    universe = {}
    for t in sampled_t:
        cutoff_lo = t - pd.Timedelta(days=90)
        win = dv_wide[(dv_wide.index >= cutoff_lo) & (dv_wide.index < t)]
        if len(win) < 10: continue
        avg_dv = win.mean(axis=0).sort_values(ascending=False)
        universe[t] = set(avg_dv.head(n_top).index)
    return universe


def main():
    print("=" * 100, flush=True)
    print("  STEP 53: pred_B-universe variant (vs Step 50 pred_z-universe +3.11)", flush=True)
    print("=" * 100, flush=True)
    print("  Causal-immediate: gross[t] = tw × alpha[t] (new weights earn new cycle)",
          flush=True)
    print("  vs lagged (Step 47): gross[t] = prev_weights × alpha[t]", flush=True)
    print("  Reviewer recompute: lagged +3.35 → causal ~+3.11–3.13", flush=True)
    t0 = time.time()
    listings = get_listings()

    apd_full = pd.read_parquet(PREDS_DIR / "predictions.parquet")
    apd_full["open_time"] = pd.to_datetime(apd_full["open_time"], utc=True)
    apd_full["alpha_A"] = apd_full["alpha_beta"]
    if "exit_time" not in apd_full.columns or "return_pct" not in apd_full.columns:
        extra = pd.read_parquet(PANEL_FULL_PIT,
                                  columns=["symbol","open_time","exit_time","return_pct"])
        extra["open_time"] = pd.to_datetime(extra["open_time"], utc=True)
        extra["exit_time"] = pd.to_datetime(extra["exit_time"], utc=True)
        apd_full = apd_full.merge(extra, on=["symbol","open_time"], how="left")

    panel_syms = sorted(apd_full["symbol"].unique())
    print(f"\nPanel symbols: {len(panel_syms)} (BTC excluded)", flush=True)

    for s, t in apd_full.groupby("symbol")["open_time"].min().items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t
    panel_syms_set = set(panel_syms)
    def elig_pit(b):
        ts = b if isinstance(b, pd.Timestamp) else pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms_set if listings.get(s) and listings[s] <= cutoff}
    target_t = sorted(apd_full[apd_full["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]

    apd_full["pred_B"] = apd_full["pred_z"] * apd_full["trail_ic"]
    # VARIANT B: build rolling-IC universe from pred_B (NOT pred_z).
    # Compare to Step 50 which built it from pred_z (= +3.11).
    apd_full["pred"] = apd_full["pred_B"]
    universe_V2 = psl.build_rolling_ic_universe(apd_full, sampled_t, psl.TOP_N, elig_pit)
    alpha_wide = apd_full.pivot_table(index="open_time", columns="symbol",
                                        values="alpha_A", aggfunc="first").sort_index()

    # ===== REAL V2 under CAUSAL aggregator =====
    print(f"\n--- Real V2 B_IC_signed under CAUSAL aggregator ---", flush=True)
    apd_v = apd_full.copy(); apd_v["pred"] = apd_v["pred_B"]
    records_real = psl.run_production_protocol_save_sleeves(apd_v, universe_V2)
    df_v_real = aggregate_hold_through_causal(records_real, alpha_wide)
    net_real = df_v_real["net_pnl_bps"].to_numpy()
    sh_real_causal = _sharpe(net_real)
    sh_lo, sh_hi = block_bootstrap_ci(net_real, statistic=_sharpe,
                                        block_size=7, n_boot=1000)[1:]
    fp_real = folds_positive(df_v_real)
    n_traded = (df_v_real["gross_pnl_bps"] != 0).sum()
    gross_avg = df_v_real["gross_pnl_bps"].mean()
    print(f"  V2 110-panel CAUSAL: Sharpe = {sh_real_causal:+.2f} "
          f"[{sh_lo:+.2f}, {sh_hi:+.2f}], folds+={fp_real}/9, "
          f"gross={gross_avg:+.2f}, traded={n_traded}/{len(df_v_real)}", flush=True)
    print(f"  (For reference: lagged Step 47 = +3.35; reviewer recompute = +3.11-3.13)",
          flush=True)
    df_v_real.to_csv(OUT / "per_cycle_real_causal.csv", index=False)

    # LOFO
    print(f"\n  LOFO on B (causal, Sharpe = {sh_real_causal:+.2f}):", flush=True)
    lofo_rows = []
    for excl in range(1, 10):
        rem = df_v_real[df_v_real["fold"] != excl]["net_pnl_bps"].to_numpy()
        sh_rem = _sharpe(rem)
        d = sh_rem - sh_real_causal
        flag = "  ← drives" if d < -0.4 else ""
        print(f"    excl {excl}: {sh_rem:+.2f} (Δ {d:+.2f}){flag}", flush=True)
        lofo_rows.append({"excl_fold": excl, "sharpe": sh_rem, "delta": d})
    pd.DataFrame(lofo_rows).to_csv(OUT / "lofo_causal.csv", index=False)

    # ===== Placebos under CAUSAL aggregator =====
    universe_liq = build_liquidity_universe(sampled_t, panel_syms, n_top=30)

    print(f"\n--- P1 placebo (liq univ random) × {N_PLACEBO} (CAUSAL) ---", flush=True)
    p1 = []
    for seed in range(N_PLACEBO):
        records_p = psl.run_production_protocol_save_sleeves(
            apd_v, universe_liq, placebo_seed=seed)
        df_v_p = aggregate_hold_through_causal(records_p, alpha_wide)
        p1.append(_sharpe(df_v_p["net_pnl_bps"].to_numpy()))
        if (seed + 1) % 25 == 0:
            print(f"  seed {seed+1}: mean={np.mean(p1):+.3f}", flush=True)
    p1 = np.array(p1)
    p1_95 = float(np.percentile(p1, 95))
    p1_pcts = np.percentile(p1, [5, 25, 50, 75, 95])
    print(f"  P1 mean: {p1.mean():+.3f}  std: {p1.std():.3f}", flush=True)
    print(f"  P1 p5/p25/p50/p75/p95: " + "/".join(f"{x:+.2f}" for x in p1_pcts), flush=True)
    print(f"  V2 rank in P1: {(p1 < sh_real_causal).mean()*100:.1f}%", flush=True)
    print(f"  Edge over P1 p95: {sh_real_causal - p1_95:+.2f}", flush=True)

    print(f"\n--- P2 placebo (V2 univ random) × {N_PLACEBO} (CAUSAL) ---", flush=True)
    p2 = []
    for seed in range(N_PLACEBO):
        records_p = psl.run_production_protocol_save_sleeves(
            apd_v, universe_V2, placebo_seed=seed)
        df_v_p = aggregate_hold_through_causal(records_p, alpha_wide)
        p2.append(_sharpe(df_v_p["net_pnl_bps"].to_numpy()))
        if (seed + 1) % 25 == 0:
            print(f"  seed {seed+1}: mean={np.mean(p2):+.3f}", flush=True)
    p2 = np.array(p2)
    p2_95 = float(np.percentile(p2, 95))
    p2_pcts = np.percentile(p2, [5, 25, 50, 75, 95])
    print(f"  P2 mean: {p2.mean():+.3f}  std: {p2.std():.3f}", flush=True)
    print(f"  P2 p5/p25/p50/p75/p95: " + "/".join(f"{x:+.2f}" for x in p2_pcts), flush=True)
    print(f"  V2 rank in P2: {(p2 < sh_real_causal).mean()*100:.1f}%", flush=True)
    print(f"  Edge over P2 p95: {sh_real_causal - p2_95:+.2f}", flush=True)

    pd.DataFrame({"P1": p1, "P2": p2}).to_csv(OUT / "placebos_causal.csv", index=False)

    print(f"\n{'='*100}", flush=True)
    print(f"  V2 110-PANEL FULL-PIT + CAUSAL AGGREGATOR — FINAL VERDICT", flush=True)
    print(f"{'='*100}", flush=True)
    p1_pass = sh_real_causal > p1_95
    p2_pass = sh_real_causal > p2_95
    print(f"  V2 Sharpe (causal): {sh_real_causal:+.2f}  CI=[{sh_lo:+.2f},{sh_hi:+.2f}]",
          flush=True)
    print(f"  P1 (liq univ random) p95: {p1_95:+.2f}  →  "
          f"{'PASS' if p1_pass else 'FAIL'} (edge {sh_real_causal - p1_95:+.2f})",
          flush=True)
    print(f"  P2 (V2 univ random)  p95: {p2_95:+.2f}  →  "
          f"{'PASS' if p2_pass else 'FAIL'} (edge {sh_real_causal - p2_95:+.2f})",
          flush=True)
    print(f"\n  References (lagged aggregator):", flush=True)
    print(f"    Step 47 reported (lagged): Sharpe +3.35", flush=True)
    print(f"    Step 48 placebos (lagged): P1 p100 (+1.69), P2 p100 (+1.81)", flush=True)
    print(f"    V2 51-panel (lagged):      Sharpe +2.19, P1 p99 (+1.04), P2 p97 (+0.58)",
          flush=True)

    verdict = {"sharpe_causal": sh_real_causal, "sh_lo": sh_lo, "sh_hi": sh_hi,
                "folds_positive": fp_real, "p1_95": p1_95, "p2_95": p2_95,
                "p1_pass": p1_pass, "p2_pass": p2_pass,
                "p1_edge": sh_real_causal - p1_95, "p2_edge": sh_real_causal - p2_95}
    pd.DataFrame([verdict]).to_csv(OUT / "verdict.csv", index=False)
    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
