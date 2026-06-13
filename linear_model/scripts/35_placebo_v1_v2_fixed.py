"""Step 35: P1+P2 placebos on Step 34 fixed V1 and V2.

Loads predictions from results/step34_v1_fixed/ (no retraining), runs:
  P1 — random pick from broad liquidity universe (top-30 by 90d $vol)
  P2 — random pick from model's own rolling-IC top-15 universe

100 seeds each. Block-bootstrap CI on real Sharpe. Verdict per variant.
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
TRAILING_IC_DAYS = 90
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


def build_liquidity_universe(sampled_t, n_top=30):
    print(f"  Loading kline volumes for liq universe...", flush=True)
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


def run_placebos(variant_slug, apd_full, sampled_t, universe_model, universe_liq,
                 alpha_wide, sh_real, n_seeds=N_PLACEBO):
    print(f"\n--- P1 placebo (liq univ random) × {n_seeds} ---", flush=True)
    p1 = []
    for seed in range(n_seeds):
        records_p = psl.run_production_protocol_save_sleeves(
            apd_full, universe_liq, placebo_seed=seed)
        df_v_p = aggregate_hold_through(records_p, alpha_wide)
        p1.append(_sharpe(df_v_p["net_pnl_bps"].to_numpy()))
        if (seed+1) % 25 == 0:
            print(f"  seed {seed+1}: mean={np.mean(p1):+.3f}", flush=True)
    p1 = np.array(p1)
    p1_95 = float(np.percentile(p1, 95))
    p1_pcts = np.percentile(p1, [5,25,50,75,95])
    print(f"  P1 mean: {p1.mean():+.3f}  std: {p1.std():.3f}", flush=True)
    print(f"  P1 p5/p25/p50/p75/p95: "
          + "/".join(f"{x:+.2f}" for x in p1_pcts), flush=True)
    print(f"  {variant_slug} rank in P1: {(p1 < sh_real).mean()*100:.1f}%", flush=True)
    print(f"  Edge over P1 p95: {sh_real - p1_95:+.2f}", flush=True)

    print(f"\n--- P2 placebo (model univ random) × {n_seeds} ---", flush=True)
    p2 = []
    for seed in range(n_seeds):
        records_p = psl.run_production_protocol_save_sleeves(
            apd_full, universe_model, placebo_seed=seed)
        df_v_p = aggregate_hold_through(records_p, alpha_wide)
        p2.append(_sharpe(df_v_p["net_pnl_bps"].to_numpy()))
        if (seed+1) % 25 == 0:
            print(f"  seed {seed+1}: mean={np.mean(p2):+.3f}", flush=True)
    p2 = np.array(p2)
    p2_95 = float(np.percentile(p2, 95))
    p2_pcts = np.percentile(p2, [5,25,50,75,95])
    print(f"  P2 mean: {p2.mean():+.3f}  std: {p2.std():.3f}", flush=True)
    print(f"  P2 p5/p25/p50/p75/p95: "
          + "/".join(f"{x:+.2f}" for x in p2_pcts), flush=True)
    print(f"  {variant_slug} rank in P2: {(p2 < sh_real).mean()*100:.1f}%", flush=True)
    print(f"  Edge over P2 p95: {sh_real - p2_95:+.2f}", flush=True)

    pd.DataFrame({"P1":p1, "P2":p2}).to_csv(
        OUT / f"step35_placebos_{variant_slug}.csv", index=False)
    return p1, p2, p1_95, p2_95


def run_variant(variant_slug, listings, universe_liq):
    """Load preds for variant, recompute sleeve universe, run placebos."""
    print(f"\n{'='*100}", flush=True)
    print(f"  {variant_slug}", flush=True)
    print(f"{'='*100}", flush=True)
    apd_full = pd.read_parquet(PREDS_DIR / f"{variant_slug}_predictions.parquet")
    apd_full["open_time"] = pd.to_datetime(apd_full["open_time"], utc=True)
    apd_full["alpha_A"] = apd_full["alpha_beta"]
    extra = pd.read_parquet(PANEL,
                              columns=["symbol","open_time","exit_time","return_pct"])
    extra["open_time"] = pd.to_datetime(extra["open_time"], utc=True)
    extra["exit_time"] = pd.to_datetime(extra["exit_time"], utc=True)
    if "exit_time" not in apd_full.columns:
        apd_full = apd_full.merge(extra, on=["symbol","open_time"], how="left")

    # Rebuild universe
    panel_syms = set(apd_full["symbol"].unique())
    for s, t in apd_full.groupby("symbol")["open_time"].min().items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t
    def elig_pit(b):
        ts = b if isinstance(b, pd.Timestamp) else pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}
    target_t = sorted(apd_full[apd_full["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]

    apd_full["pred"] = apd_full["pred_z"]  # ensure column exists
    apd_full["pred_B"] = apd_full["pred_z"] * apd_full["trail_ic"]
    universe_model = psl.build_rolling_ic_universe(apd_full, sampled_t,
                                                     psl.TOP_N, elig_pit)
    alpha_wide = apd_full.pivot_table(index="open_time", columns="symbol",
                                        values="alpha_A", aggfunc="first").sort_index()

    # Reproduce real Sharpe (B_IC_signed)
    apd_v = apd_full.copy(); apd_v["pred"] = apd_v["pred_B"]
    records_real = psl.run_production_protocol_save_sleeves(apd_v, universe_model)
    df_v_real = aggregate_hold_through(records_real, alpha_wide)
    net = df_v_real["net_pnl_bps"].to_numpy()
    sh_real = _sharpe(net)
    sh_lo, sh_hi = block_bootstrap_ci(net, statistic=_sharpe,
                                        block_size=7, n_boot=1000)[1:]
    print(f"  {variant_slug} B_IC_signed Sharpe = {sh_real:+.2f} "
          f"[{sh_lo:+.2f},{sh_hi:+.2f}], folds+={folds_positive(df_v_real)}/9",
          flush=True)

    p1, p2, p1_95, p2_95 = run_placebos(variant_slug, apd_v, sampled_t,
                                          universe_model, universe_liq, alpha_wide,
                                          sh_real)

    return {"slug":variant_slug, "sharpe":sh_real, "sh_lo":sh_lo, "sh_hi":sh_hi,
            "p1_95":p1_95, "p2_95":p2_95,
            "p1_pass":sh_real > p1_95, "p2_pass":sh_real > p2_95}


def main():
    print("=== Step 35: P1+P2 placebos on Step 34 fixed V1 and V2 ===\n", flush=True)
    t0 = time.time()
    listings = get_listings()

    # Build liquidity universe once (re-used across variants)
    apd0 = pd.read_parquet(PREDS_DIR / "v1_fixed_predictions.parquet",
                            columns=["open_time", "fold"])
    apd0["open_time"] = pd.to_datetime(apd0["open_time"], utc=True)
    target_t = sorted(apd0[apd0["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]
    universe_liq = build_liquidity_universe(sampled_t, n_top=30)
    print(f"  liq universe built for {len(universe_liq)} timestamps", flush=True)

    results = []
    for slug in ["v1_fixed", "v2_fixed"]:
        r = run_variant(slug, listings, universe_liq)
        results.append(r)

    # Verdict
    print(f"\n{'='*100}", flush=True)
    print(f"  STEP 35 VERDICT", flush=True)
    print(f"{'='*100}", flush=True)
    for r in results:
        p1_str = f"PASS p95={r['p1_95']:+.2f}" if r["p1_pass"] else f"FAIL p95={r['p1_95']:+.2f}"
        p2_str = f"PASS p95={r['p2_95']:+.2f}" if r["p2_pass"] else f"FAIL p95={r['p2_95']:+.2f}"
        print(f"  {r['slug']:<12}  Sharpe={r['sharpe']:+.2f}  CI=[{r['sh_lo']:+.2f},{r['sh_hi']:+.2f}]",
              flush=True)
        print(f"               P1: {p1_str}    P2: {p2_str}", flush=True)
    pd.DataFrame(results).to_csv(OUT / "step35_verdict.csv", index=False)
    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
