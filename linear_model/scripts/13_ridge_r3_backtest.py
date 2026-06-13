"""Step 13: Run R3 (Ridge + 6 squared terms) through V3.1 β-hedged backtest.

Variants:
  A. Pure rank by pred_z (production-style)
  B. IC-weighted: rank by pred × trailing_90d_per_symbol_IC (magnitude × sign)
  C. IC-magnitude-weighted: rank by pred × |trailing_IC|  (no sign flip)
  D. IC-shrunk: rank by pred × shrunk_IC (tau-shrunk toward 0)

For each variant: rolling-IC universe top-15, K=3 each side, full production
gates (conv_gate + PM_M + filter_refill), V3.1 6-sleeve.

Compare to:
  - R3 baseline rank Sharpe (Variant A)
  - LGBM production +0.74
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

R3_PREDS = REPO / "linear_model/results/ridge_opt_v3_ridge_preds.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
OUT = REPO / "linear_model/results"

OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
CAPITAL = 100.0
TRAILING_IC_DAYS = 90      # for IC weighting
IC_SHRINK_TAU = 50         # rows of prior for shrinkage


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


def compute_trailing_ic_per_symbol(apd, sampled_t, win_days=90):
    """For each (symbol, t in sampled_t), compute Spearman IC over trailing
    win_days of past (pred, alpha_beta) pairs for that symbol.

    PIT: uses only rows with open_time strictly < t (excludes current cycle).
    """
    print(f"  Computing trailing {win_days}d per-symbol IC...", flush=True)
    t0 = time.time()
    # Sample apd at 4h cadence for efficiency (ranks are computed across syms at t)
    apd_s = apd[apd["open_time"].isin(set(sampled_t))].copy()
    apd_s = apd_s.sort_values(["symbol","open_time"]).reset_index(drop=True)

    # Per-symbol rolling spearman over 90d window of 4h samples
    # Approx: 90d × 6 cycles/day = 540 samples
    cycles_per_day = 6  # 24h / 4h
    win_cycles = win_days * cycles_per_day

    ic_records = []
    for sym, g in apd_s.groupby("symbol"):
        g = g.sort_values("open_time").reset_index(drop=True)
        # Compute trailing IC using past N cycles only (exclude current)
        pred = g["pred_z"].to_numpy()
        alpha = g["alpha_beta"].to_numpy()
        n = len(g)
        ics = np.full(n, np.nan)
        for i in range(50, n):  # need at least 50 past obs
            lo = max(0, i - win_cycles)
            p = pred[lo:i]  # past preds (exclusive of i)
            a = alpha[lo:i]  # past realized
            mask = ~np.isnan(p) & ~np.isnan(a)
            if mask.sum() < 50: continue
            p_r = pd.Series(p[mask]).rank().to_numpy()
            a_r = pd.Series(a[mask]).rank().to_numpy()
            if p_r.std() < 1e-6 or a_r.std() < 1e-6: continue
            ics[i] = np.corrcoef(p_r, a_r)[0, 1]
        for j, t in enumerate(g["open_time"]):
            ic_records.append({"symbol": sym, "open_time": t, "trail_ic": ics[j]})
    df_ic = pd.DataFrame(ic_records)
    df_ic["trail_ic"] = df_ic["trail_ic"].fillna(0)
    # Shrunk version
    # Treat trail_ic as estimate with n = win_cycles, shrink toward 0
    df_ic["trail_ic_shrunk"] = df_ic["trail_ic"] * win_cycles / (win_cycles + IC_SHRINK_TAU)
    print(f"    done {time.time()-t0:.0f}s, {len(df_ic):,} (sym, time) pairs", flush=True)
    print(f"    trail_ic stats: mean={df_ic['trail_ic'].mean():+.4f}  "
          f"std={df_ic['trail_ic'].std():.4f}  "
          f"p25/p75=[{df_ic['trail_ic'].quantile(0.25):+.3f},"
          f"{df_ic['trail_ic'].quantile(0.75):+.3f}]", flush=True)
    return df_ic


def build_universe_and_records(apd, sampled_t, listings, panel_syms, ranking_col,
                                 universe):
    """Universe is pre-computed (uses pred_z for IC). Ranking column is variant-specific."""
    apd2 = apd.copy()
    apd2["pred"] = apd2[ranking_col]  # set 'pred' to variant's ranking column
    records = psl.run_production_protocol_save_sleeves(apd2, universe)
    return records


def aggregate_alpha(records, alpha_wide):
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
        target_weights = defaultdict(float)
        sw = 1.0 / psl.N_SLEEVES
        for sl in sleeve_queue:
            nL, nS = len(sl["longs"]), len(sl["shorts"])
            if nL == 0 or nS == 0: continue
            for s in sl["longs"]: target_weights[s] += sw * (1.0/nL)
            for s in sl["shorts"]: target_weights[s] -= sw * (1.0/nS)
        gross = 0.0
        if t in alpha_wide.index:
            a = alpha_wide.loc[t]
            for sym, w in prev_weights.items():
                if sym in a.index and not pd.isna(a[sym]):
                    gross += w * a[sym] * 1e4
        syms = set(target_weights.keys()) | set(prev_weights.keys())
        abs_d = sum(abs(target_weights.get(s,0)-prev_weights.get(s,0)) for s in syms)
        cost = abs_d * psl.COST_PER_UNIT_ABS_DELTA
        rows.append({"time":t,"fold":fold,"gross_pnl_bps":gross,"cost_bps":cost,
                     "net_pnl_bps":gross-cost,"turnover":abs_d})
        prev_weights = dict(target_weights)
    return pd.DataFrame(rows)


def main():
    print("=== Step 13: R3 backtest + IC-weighted ranking variants ===\n", flush=True)
    t0 = time.time()
    listings = get_listings()

    apd = pd.read_parquet(R3_PREDS)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    # Also need return_pct for production protocol's picks_hist accounting
    base = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                           columns=["symbol","open_time","return_pct"])
    base["open_time"] = pd.to_datetime(base["open_time"], utc=True)
    apd = apd.merge(base, on=["symbol","open_time"], how="left")
    apd["alpha_A"] = apd["alpha_beta"]
    apd["pred_bps"] = apd["pred_z"] * apd["sigma_idio_ref"] * 1e4
    print(f"R3 predictions: {len(apd):,} rows", flush=True)

    panel_syms = set(apd["symbol"].unique())
    for s, t in apd.groupby("symbol")["open_time"].min().items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t

    # Universe + sampled times
    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]
    print(f"Sampled cycles: {len(sampled_t)}", flush=True)

    # Compute per-symbol trailing IC
    df_ic = compute_trailing_ic_per_symbol(apd, sampled_t,
                                            win_days=TRAILING_IC_DAYS)
    apd_s = apd[apd["open_time"].isin(set(sampled_t))].copy()
    apd_s = apd_s.merge(df_ic, on=["symbol","open_time"], how="left")
    apd_s["trail_ic"] = apd_s["trail_ic"].fillna(0)
    apd_s["trail_ic_shrunk"] = apd_s["trail_ic_shrunk"].fillna(0)

    # Build ranking variants
    apd_s["pred_A"] = apd_s["pred_z"]                                 # baseline
    apd_s["pred_B"] = apd_s["pred_z"] * apd_s["trail_ic"]              # IC weighted (signed)
    apd_s["pred_C"] = apd_s["pred_z"] * apd_s["trail_ic"].abs()        # |IC| weighted (no flip)
    apd_s["pred_D"] = apd_s["pred_z"] * apd_s["trail_ic_shrunk"]       # shrunk IC

    # Need to also have these columns for ALL apd rows (production protocol needs
    # them at every 5m bar). Re-merge.
    apd_full = apd.merge(apd_s[["symbol","open_time","trail_ic","trail_ic_shrunk",
                                  "pred_A","pred_B","pred_C","pred_D"]],
                          on=["symbol","open_time"], how="left")
    # NaN for non-sampled bars: just fill with pred_z (won't be used at trade time anyway)
    for c in ("pred_A","pred_B","pred_C","pred_D"):
        apd_full[c] = apd_full[c].fillna(apd_full["pred_z"])

    alpha_wide = apd_full.pivot_table(index="open_time", columns="symbol",
                                        values="alpha_A", aggfunc="first").sort_index()

    # Build universe ONCE using base pred_z (consistent across variants)
    apd_for_universe = apd_full.copy()
    apd_for_universe["pred"] = apd_for_universe["pred_z"]
    def elig_pit(b):
        ts = b if isinstance(b, pd.Timestamp) else pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}
    universe = psl.build_rolling_ic_universe(apd_for_universe, sampled_t,
                                              psl.TOP_N, elig_pit)
    print(f"\n  Universe built (top {psl.TOP_N} by rolling-IC of pred_z)", flush=True)

    results = []
    for label, col in [("A_baseline", "pred_A"),
                       ("B_IC_signed", "pred_B"),
                       ("C_IC_magnitude", "pred_C"),
                       ("D_IC_shrunk", "pred_D")]:
        print(f"\n  ===== {label} (rank by {col}) =====", flush=True)
        records = build_universe_and_records(apd_full, sampled_t, listings,
                                              panel_syms, col, universe)
        df_v = aggregate_alpha(records, alpha_wide)
        net = df_v["net_pnl_bps"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe,
                                         block_size=7, n_boot=1000)
        total_d = net.sum() / 1e4 * CAPITAL
        end_eq = CAPITAL + total_d
        # Per-cycle IC of the ranking column vs realized
        cyc_ic = apd_full.dropna(subset=["alpha_beta"]).groupby("open_time").apply(
            lambda g: g[col].rank().corr(g["alpha_beta"].rank())
            if len(g) >= 5 else np.nan).dropna()
        per_cycle_ic = float(cyc_ic.mean())
        r = {"label": label, "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
             "end_eq": end_eq, "ic": per_cycle_ic, "folds_pos": folds_positive(df_v),
             "gross": float(df_v["gross_pnl_bps"].mean()),
             "cost": float(df_v["cost_bps"].mean()),
             "maxDD": _max_dd(net), "n_traded": int(records["traded"].sum())}
        results.append(r)
        print(f"    Sharpe={sh:+.2f} [{lo:+.2f},{hi:+.2f}], IC={per_cycle_ic:+.4f}, "
              f"end-eq=${end_eq:.2f}, folds+={r['folds_pos']}/9, "
              f"gross={r['gross']:+.2f}, cost={r['cost']:+.2f}", flush=True)
        df_v.to_csv(OUT / f"r3_backtest_{label}.csv", index=False)

    print("\n" + "="*100, flush=True)
    print("  R3 RANKING SCHEME COMPARISON (V3.1 β-hedged, all gates)", flush=True)
    print("="*100, flush=True)
    print(f"  {'variant':<22} {'Sharpe':>8} {'CI':>20} {'IC':>10} "
          f"{'end-eq':>10} {'gross':>7} {'cost':>7} {'fold+':>6}", flush=True)
    for r in results:
        ci = f"[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}]"
        print(f"  {r['label']:<22} {r['sharpe']:+8.2f} {ci:>20} {r['ic']:+10.4f} "
              f"${r['end_eq']:>8.2f} {r['gross']:>+7.2f} {r['cost']:>7.2f} "
              f"{r['folds_pos']:>3}/9", flush=True)

    print(f"\n  References:", flush=True)
    print(f"    LGBM production (shift 1, all gates):    Sharpe +0.74", flush=True)
    print(f"    LGBM clean-PIT (shift 49, all gates):    Sharpe +0.68", flush=True)
    print(f"    Original Ridge (with sym dummies, R1):   Sharpe -1.62", flush=True)
    pd.DataFrame(results).to_csv(OUT / "r3_backtest_summary.csv", index=False)
    print(f"\n  Total: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
