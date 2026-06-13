"""Phase 1 stability check: drop-5 random symbols stress test on each variant.

For each variant (production / Phase 1A BTC / Phase 1B BTC+ETH):
  - Take the existing predictions (no retrain)
  - For each of N_DRAWS=20 random 5-symbol drops:
    - Filter audit panel
    - Rebuild rolling-IC universe with reduced eligibility
    - Run V3.1 with same sleeve overlay
    - Record Sharpe, totPnL, maxDD
  - Report mean / std / range across draws

Pass criterion (per user spec):
  - Phase 1: random-drop K=5 std materially falls, ideally < 0.4

Note: this is a no-retrain stress test, which has the confound we discussed
(target_A definition, sym_id encoding, xs_rank). But it's the same protocol
applied to all three variants, so the COMPARATIVE std is meaningful even if
the absolute std overstates fragility for any single variant.
"""
from __future__ import annotations
import sys, time, importlib.util, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("psl", REPO / "scripts/phase_ah_sleeve.py")
psl = importlib.util.module_from_spec(spec); spec.loader.exec_module(psl)

OUT = REPO / "outputs/vBTC_phase1_stress"
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = {
    "production_basket_51":   REPO / "outputs/vBTC_audit_panel/all_predictions.parquet",
    "phase1A_btc_residual":   REPO / "outputs/vBTC_phase1_ref_btc/all_predictions.parquet",
    "phase1B_btc_eth_resid":  REPO / "outputs/vBTC_phase1_ref_btc_eth/all_predictions.parquet",
}
N_DRAWS = 20
K_DROP = 5
OOS_FOLDS = list(range(1, 10))


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(psl.CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def run_one_universe(apd, drop_syms, listings, panel_syms, fwd_rets_4h):
    apd_f = apd[~apd["symbol"].isin(drop_syms)].copy()
    syms_remain = set(apd_f["symbol"].unique())
    def elig(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=psl.MIN_HISTORY_DAYS)
        return {s for s in syms_remain if listings.get(s) and listings[s] <= cutoff}
    target_t = sorted(apd_f[apd_f["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]
    universe = psl.build_rolling_ic_universe(apd_f, sampled_t, psl.TOP_N, elig)
    records = psl.run_production_protocol_save_sleeves(apd_f, universe)
    df_v = psl.aggregate_sleeves(records, fwd_rets_4h)
    net = df_v["net_pnl_bps"].to_numpy()
    return _sharpe(net), _max_dd(net), float(net.sum()), int(records["traded"].sum())


def run_variant(label, apd_path, fwd_rets_4h, listings, panel_syms, rng):
    print(f"\n=== Variant: {label} ===", flush=True)
    apd = pd.read_parquet(apd_path)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    syms_avail = sorted(set(apd["symbol"].unique()))
    print(f"  predictions cover {len(syms_avail)} symbols", flush=True)

    # Baseline (no drop)
    sh0, dd0, pnl0, n0 = run_one_universe(apd, set(), listings, panel_syms, fwd_rets_4h)
    print(f"  Baseline (no drop): Sharpe={sh0:+.2f} totPnL={pnl0:+.0f} maxDD={dd0:+.0f} traded={n0}",
          flush=True)

    results = []
    for i in range(N_DRAWS):
        drop = set(rng.choice(syms_avail, K_DROP, replace=False))
        sh, dd, pnl, n = run_one_universe(apd, drop, listings, panel_syms, fwd_rets_4h)
        results.append({"draw": i, "drop": sorted(drop), "sharpe": sh, "maxDD": dd,
                        "totPnL": pnl, "n_traded": n})
        if (i+1) % 5 == 0:
            print(f"  draw {i+1}/{N_DRAWS} done", flush=True)
    df = pd.DataFrame(results)
    return sh0, df


def main():
    print(f"=== Phase 1 stability: drop {K_DROP} random symbols, {N_DRAWS} draws per variant ===\n",
          flush=True)
    # Load fwd_rets once (same 51-symbol close prices used by all variants)
    apd = pd.read_parquet(VARIANTS["production_basket_51"])
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    panel_syms = sorted(apd["symbol"].unique())
    listings = psl.get_listings()
    print(f"Loading close prices for 51 symbols...", flush=True)
    t0 = time.time()
    frames = []
    for sym in panel_syms:
        sd = psl.KLINES_DIR / sym / "5m"
        if not sd.exists(): continue
        files = sorted(sd.glob("*.parquet"))
        dfs = []
        for f in files:
            try: dfs.append(pd.read_parquet(f, columns=["open_time", "close"]))
            except Exception: pass
        if not dfs: continue
        df = pd.concat(dfs, ignore_index=True)
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
        df = df.dropna(subset=["open_time"]).drop_duplicates("open_time").set_index("open_time")
        df = df.rename(columns={"close": sym})
        frames.append(df)
    close_wide = pd.concat(frames, axis=1).sort_index()
    fwd_rets_4h = (close_wide.shift(-psl.HORIZON_ENTRY) - close_wide) / close_wide
    print(f"  ready ({time.time()-t0:.0f}s)", flush=True)

    # Same seed for all variants so the SAME drops are tested across variants
    summary = []
    for label, path in VARIANTS.items():
        rng = np.random.default_rng(2026)
        sh0, df = run_variant(label, path, fwd_rets_4h, listings, panel_syms, rng)
        df.to_csv(OUT / f"stress_{label}.csv", index=False)
        sh_vals = df["sharpe"].values
        summary.append({
            "variant": label,
            "baseline_sharpe": sh0,
            "drop5_mean_sh": sh_vals.mean(),
            "drop5_std_sh": sh_vals.std(),
            "drop5_min_sh": sh_vals.min(),
            "drop5_max_sh": sh_vals.max(),
            "drop5_p25": float(np.percentile(sh_vals, 25)),
            "drop5_p75": float(np.percentile(sh_vals, 75)),
        })

    print("\n" + "=" * 100)
    print("  SUMMARY: drop-5 random symbols, 20 draws per variant (same seed)")
    print("=" * 100)
    s = pd.DataFrame(summary).set_index("variant")
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:+.3f}")
    print(s.T.to_string(), flush=True)

    print("\n=== Pass criteria check ===", flush=True)
    for r in summary:
        ok_std = r["drop5_std_sh"] < 0.4
        ok_sh = r["baseline_sharpe"] > 1.0
        print(f"  {r['variant']:<28}: "
              f"baseline_sh={r['baseline_sharpe']:+.2f} ({'PASS' if ok_sh else 'FAIL'} >1.0)  "
              f"drop5_std={r['drop5_std_sh']:.2f} ({'PASS' if ok_std else 'FAIL'} <0.4)",
              flush=True)


if __name__ == "__main__":
    main()
