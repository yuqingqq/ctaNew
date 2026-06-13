"""Step 42: P1+P2 placebos on V2 110-panel (Step 41 rerun predictions).

Tests whether V2's Sharpe +2.03 on the 110-panel beats matched-universe
random p95. Gate-consistent methodology (real and placebo both gate on pred_B).
Reuses Step 41 rerun predictions — no retraining needed.

References:
  V2 51-panel + sleeve (Step 34/35): Sharpe +2.19, P1 p99, P2 p97
  V2 110-panel + sleeve (Step 41 rerun): Sharpe +2.03, P1/P2 PENDING (this script)
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

PANEL_111_PIT = REPO / "outputs/vBTC_features_btc_only_111_pit/panel_btc_only_111.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
PREDS_DIR = REPO / "linear_model/results/step41_111panel_pit"
OUT = REPO / "linear_model/results"

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


def build_liquidity_universe(sampled_t, panel_syms, n_top=30):
    """Top n_top symbols by trailing 90d kline dollar volume — restricted to
    symbols present in the 110-panel (panel_syms)."""
    print(f"  Loading kline volumes for {len(panel_syms)} 110-panel symbols...", flush=True)
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


def run_placebos(apd_v, sampled_t, universe_model, universe_liq, alpha_wide,
                 sh_real, n_seeds=N_PLACEBO):
    print(f"\n--- P1 placebo (liq univ random, gate on pred_B) × {n_seeds} ---",
          flush=True)
    p1 = []
    for seed in range(n_seeds):
        records_p = psl.run_production_protocol_save_sleeves(
            apd_v, universe_liq, placebo_seed=seed)
        df_v_p = aggregate_hold_through(records_p, alpha_wide)
        p1.append(_sharpe(df_v_p["net_pnl_bps"].to_numpy()))
        if (seed+1) % 25 == 0:
            print(f"  seed {seed+1}: mean={np.mean(p1):+.3f}", flush=True)
    p1 = np.array(p1)
    p1_95 = float(np.percentile(p1, 95))
    p1_pcts = np.percentile(p1, [5,25,50,75,95])
    print(f"  P1 mean: {p1.mean():+.3f}  std: {p1.std():.3f}", flush=True)
    print(f"  P1 p5/p25/p50/p75/p95: " + "/".join(f"{x:+.2f}" for x in p1_pcts),
          flush=True)
    print(f"  V2 rank in P1: {(p1 < sh_real).mean()*100:.1f}%", flush=True)
    print(f"  Edge over P1 p95: {sh_real - p1_95:+.2f}", flush=True)

    print(f"\n--- P2 placebo (V2 model univ random, gate on pred_B) × {n_seeds} ---",
          flush=True)
    p2 = []
    for seed in range(n_seeds):
        records_p = psl.run_production_protocol_save_sleeves(
            apd_v, universe_model, placebo_seed=seed)
        df_v_p = aggregate_hold_through(records_p, alpha_wide)
        p2.append(_sharpe(df_v_p["net_pnl_bps"].to_numpy()))
        if (seed+1) % 25 == 0:
            print(f"  seed {seed+1}: mean={np.mean(p2):+.3f}", flush=True)
    p2 = np.array(p2)
    p2_95 = float(np.percentile(p2, 95))
    p2_pcts = np.percentile(p2, [5,25,50,75,95])
    print(f"  P2 mean: {p2.mean():+.3f}  std: {p2.std():.3f}", flush=True)
    print(f"  P2 p5/p25/p50/p75/p95: " + "/".join(f"{x:+.2f}" for x in p2_pcts),
          flush=True)
    print(f"  V2 rank in P2: {(p2 < sh_real).mean()*100:.1f}%", flush=True)
    print(f"  Edge over P2 p95: {sh_real - p2_95:+.2f}", flush=True)

    pd.DataFrame({"P1":p1, "P2":p2}).to_csv(OUT / "step42_placebos_v2_110.csv",
                                              index=False)
    return p1, p2, p1_95, p2_95


def main():
    print("=" * 100, flush=True)
    print("  STEP 42: P1+P2 placebos on V2 110-panel (Step 41 rerun)", flush=True)
    print("=" * 100, flush=True)
    print("  Reference: V2 on 110-panel + sleeve = Sharpe +2.03 (CI [-0.22, +4.14])",
          flush=True)
    print("  Real and placebo BOTH gate on pred_B (gate-consistent)", flush=True)
    t0 = time.time()
    listings = get_listings()

    apd_full = pd.read_parquet(PREDS_DIR / "predictions.parquet")
    apd_full["open_time"] = pd.to_datetime(apd_full["open_time"], utc=True)
    apd_full["alpha_A"] = apd_full["alpha_beta"]
    if "return_pct" not in apd_full.columns or "exit_time" not in apd_full.columns:
        extra = pd.read_parquet(PANEL_111_PIT,
                                  columns=["symbol","open_time","return_pct","exit_time"])
        extra["open_time"] = pd.to_datetime(extra["open_time"], utc=True)
        extra["exit_time"] = pd.to_datetime(extra["exit_time"], utc=True)
        apd_full = apd_full.merge(extra, on=["symbol","open_time"], how="left")

    panel_syms = sorted(apd_full["symbol"].unique())
    print(f"\nPanel symbols (BTC already excluded): {len(panel_syms)}", flush=True)
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
    print(f"OOS cycles: {len(sampled_t)}", flush=True)

    apd_full["pred_B"] = apd_full["pred_z"] * apd_full["trail_ic"]
    apd_full["pred"] = apd_full["pred_z"]
    universe_V2 = psl.build_rolling_ic_universe(apd_full, sampled_t, psl.TOP_N, elig_pit)
    universe_liq = build_liquidity_universe(sampled_t, panel_syms, n_top=30)
    print(f"  V2 model universe built ({len(universe_V2)} cycles), "
          f"liq universe ({len(universe_liq)} cycles)", flush=True)

    alpha_wide = apd_full.pivot_table(index="open_time", columns="symbol",
                                        values="alpha_A", aggfunc="first").sort_index()

    # Real V2 B Sharpe (gate on pred_B)
    apd_v = apd_full.copy(); apd_v["pred"] = apd_v["pred_B"]
    records_real = psl.run_production_protocol_save_sleeves(apd_v, universe_V2)
    df_v_real = aggregate_hold_through(records_real, alpha_wide)
    net = df_v_real["net_pnl_bps"].to_numpy()
    sh_real = _sharpe(net)
    sh_lo, sh_hi = block_bootstrap_ci(net, statistic=_sharpe,
                                        block_size=7, n_boot=1000)[1:]
    print(f"\nV2 110-panel B_IC_signed: Sharpe = {sh_real:+.2f} [{sh_lo:+.2f}, {sh_hi:+.2f}], "
          f"folds+={folds_positive(df_v_real)}/9", flush=True)

    p1, p2, p1_95, p2_95 = run_placebos(apd_v, sampled_t, universe_V2, universe_liq,
                                          alpha_wide, sh_real)

    print(f"\n{'='*100}", flush=True)
    print(f"  V2 110-PANEL PLACEBO VERDICT", flush=True)
    print(f"{'='*100}", flush=True)
    p1_pass = sh_real > p1_95
    p2_pass = sh_real > p2_95
    print(f"  V2 110-panel Sharpe: {sh_real:+.2f}  CI=[{sh_lo:+.2f},{sh_hi:+.2f}]",
          flush=True)
    print(f"  P1 (liq univ random, gate on pred_B) p95: {p1_95:+.2f}  →  "
          f"{'PASS' if p1_pass else 'FAIL'} (edge {sh_real - p1_95:+.2f})",
          flush=True)
    print(f"  P2 (V2 univ random, gate on pred_B)  p95: {p2_95:+.2f}  →  "
          f"{'PASS' if p2_pass else 'FAIL'} (edge {sh_real - p2_95:+.2f})",
          flush=True)
    print(f"\n  V2 51-panel reference (Step 35):  P1 PASS p99 (+1.04), P2 PASS p97 (+0.58)",
          flush=True)

    verdict = {"variant": "V2_110_panel", "sharpe": sh_real,
                "sh_lo": sh_lo, "sh_hi": sh_hi,
                "p1_95": p1_95, "p2_95": p2_95,
                "p1_pass": p1_pass, "p2_pass": p2_pass,
                "p1_edge": sh_real - p1_95, "p2_edge": sh_real - p2_95}
    pd.DataFrame([verdict]).to_csv(OUT / "step42_verdict.csv", index=False)
    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
