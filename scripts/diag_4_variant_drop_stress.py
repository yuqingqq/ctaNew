"""Random-symbol-drop stress test for the 4 variants.

The portability hypothesis: variants C (BTC-only features) and D (+ LIQ universe)
should have LOWER drop-5 Sharpe std than A/B, even if absolute Sharpe is lower.

Drop K=5 random symbols, 20 draws per variant, same seed for fair comparison.
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
sys.path.insert(0, str(REPO / "scripts"))
import diag_4_variant_comparison as vc

OUT = REPO / "outputs/vBTC_4variant_stress"
OUT.mkdir(parents=True, exist_ok=True)

N_DRAWS = 20
K_DROP = 5
OOS_FOLDS = list(range(1, 10))
CAPITAL = 100.0


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(psl.CYCLES_PER_YEAR))


def run_one_drop(apd, alpha_wide_all, drop_syms, listings, panel_syms, universe_builder_fn):
    """Drop given symbols, rebuild universe, run V3.1 β-hedged with gates."""
    apd_f = apd[~apd["symbol"].isin(drop_syms)].copy()
    syms_remain = sorted(set(apd_f["symbol"].unique()))
    def elig(b):
        if isinstance(b, pd.Timestamp): ts = b
        else: ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=psl.MIN_HISTORY_DAYS)
        return {s for s in syms_remain if listings.get(s) and listings[s] <= cutoff}
    target_t = sorted(apd_f[apd_f["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]
    fold_lookup = apd_f[apd_f["open_time"].isin(set(sampled_t))].groupby("open_time")["fold"].first().to_dict()
    universe = universe_builder_fn(apd_f, sampled_t, elig, syms_remain)
    # Restrict alpha_wide to remaining symbols
    alpha_wide = alpha_wide_all.copy()
    drop_in_alpha = [s for s in drop_syms if s in alpha_wide.columns]
    if drop_in_alpha:
        alpha_wide = alpha_wide.drop(columns=drop_in_alpha)
    records = vc.run_protocol(apd_f, universe, fold_lookup,
                                use_conv_gate=True, use_pm=True, use_filter_refill=True)
    df_v = vc.aggregate_alpha(records, alpha_wide)
    net = df_v["net_pnl_bps"].to_numpy()
    return _sharpe(net), float(net.sum())


def universe_ic(apd, sampled_t, elig_fn, syms_remain):
    return psl.build_rolling_ic_universe(apd, sampled_t, vc.TOP_N_IC, elig_fn)


def universe_liq_factory(panel_btc):
    def fn(apd, sampled_t, elig_fn, syms_remain):
        return vc.build_liquidity_universe(panel_btc, sampled_t, vc.TOP_N_LIQ,
                                              vc.LIQ_REFRESH_DAYS, syms_remain,
                                              listings_get_listings(), min_age_days=180)
    return fn


def main():
    print("=== 4-variant drop-5 stress test ===\n", flush=True)
    t_start = time.time()
    # Load
    apd_old = pd.read_parquet(vc.APD_OLD)
    apd_old["open_time"] = pd.to_datetime(apd_old["open_time"], utc=True)
    apd_old["exit_time"] = pd.to_datetime(apd_old["exit_time"], utc=True)
    apd_new = pd.read_parquet(vc.APD_NEW)
    apd_new["open_time"] = pd.to_datetime(apd_new["open_time"], utc=True)
    apd_new["exit_time"] = pd.to_datetime(apd_new["exit_time"], utc=True)
    panel_syms = sorted(apd_old["symbol"].unique())
    listings = psl.get_listings()
    panel_btc = pd.read_parquet(vc.PANEL_BTC, columns=["open_time","symbol","log_quote_volume_90d"])
    panel_btc["open_time"] = pd.to_datetime(panel_btc["open_time"], utc=True)

    alpha_wide_old = apd_old.pivot_table(index="open_time", columns="symbol",
                                          values="alpha_A", aggfunc="first").sort_index()
    alpha_wide_new = apd_new.pivot_table(index="open_time", columns="symbol",
                                          values="alpha_A", aggfunc="first").sort_index()

    def universe_liq_builder(apd, sampled_t, elig_fn, syms_remain):
        return vc.build_liquidity_universe(panel_btc, sampled_t, vc.TOP_N_LIQ,
                                              vc.LIQ_REFRESH_DAYS, syms_remain,
                                              listings, min_age_days=180)

    variants = [
        ("A_old_features_IC_universe",  apd_old, alpha_wide_old, universe_ic),
        ("B_old_features_LIQ_universe", apd_old, alpha_wide_old, universe_liq_builder),
        ("C_BTC_features_IC_universe",  apd_new, alpha_wide_new, universe_ic),
        ("D_BTC_features_LIQ_universe", apd_new, alpha_wide_new, universe_liq_builder),
    ]

    summary = []
    rng_seed_base = 2026
    for label, apd, alpha_wide, univ_fn in variants:
        print(f"\n--- {label} ---", flush=True)
        # Baseline (no drop)
        sh_base, pnl_base = run_one_drop(apd, alpha_wide, set(), listings, panel_syms, univ_fn)
        print(f"  baseline: Sharpe = {sh_base:+.2f}, totPnL = {pnl_base:+.0f} bps", flush=True)
        rng = np.random.default_rng(rng_seed_base)
        results = []
        for i in range(N_DRAWS):
            drop = set(rng.choice(panel_syms, K_DROP, replace=False))
            sh, pnl = run_one_drop(apd, alpha_wide, drop, listings, panel_syms, univ_fn)
            results.append({"draw": i, "drop": sorted(drop), "sharpe": sh, "pnl_bps": pnl})
            if (i+1) % 5 == 0:
                print(f"  draw {i+1}/{N_DRAWS} done", flush=True)
        df = pd.DataFrame(results)
        shs = df["sharpe"].values
        summary.append({
            "variant": label,
            "baseline_sharpe": sh_base,
            "drop5_mean": shs.mean(),
            "drop5_std": shs.std(),
            "drop5_min": shs.min(),
            "drop5_max": shs.max(),
            "drop5_p25": float(np.percentile(shs, 25)),
            "drop5_p75": float(np.percentile(shs, 75)),
        })
        df.to_csv(OUT / f"stress_{label}.csv", index=False)

    print("\n" + "="*100)
    print(f"  STRESS TEST SUMMARY — drop {K_DROP} random symbols, {N_DRAWS} draws (same seed)")
    print("="*100)
    s = pd.DataFrame(summary).set_index("variant")
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:+.3f}")
    print(s.T.to_string(), flush=True)
    s.to_csv(OUT / "summary.csv")

    print("\nPortability check: lower std = more universe-portable", flush=True)
    print(f"  Baseline std targets: A & B should be HIGH (universe-bound), C & D should be LOW", flush=True)
    print(f"\n  A drop5_std: {summary[0]['drop5_std']:.3f}", flush=True)
    print(f"  B drop5_std: {summary[1]['drop5_std']:.3f}", flush=True)
    print(f"  C drop5_std: {summary[2]['drop5_std']:.3f}", flush=True)
    print(f"  D drop5_std: {summary[3]['drop5_std']:.3f}", flush=True)
    print(f"\nTotal runtime: {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
