"""Oracle ceiling for Phase 1D (β-neutral residual target on full 51).

Measures the theoretical edge available in the β-residual setup by substituting
perfect-foresight picks at the K=3 step. Universe / gates / V3.1 sleeve overlay
all unchanged.

Three measurements:
  A. Oracle K=3 on Phase 1D universe (rolling-IC top-15) — ceiling with current selector
  B. Oracle K=3 on full 51 (no universe filter) — ceiling with strategy structure only
  C. Cross-sectional dispersion: per-cycle realized α_β spread between top-3 and bot-3 ranks

Compares against Phase 1D actual (+0.65 Sharpe). Reveals:
  - Model-side improvement headroom (A − actual)
  - Universe-filter cost (B − A)
  - Cycle-level opportunity magnitude (C → ultimate cap)
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

APD_PATH = REPO / "outputs/vBTC_phase1d_rolling_beta/all_predictions.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
OUT = REPO / "outputs/vBTC_phase1d_oracle"
OUT.mkdir(parents=True, exist_ok=True)

OOS_FOLDS = list(range(1, 10))


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(psl.CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def folds_positive(df_v):
    return sum(1 for _, g in df_v.groupby("fold") if _sharpe(g["net_pnl_bps"]) > 0)


def run_oracle_protocol(apd, universe_fn, K, sort_key="alpha_A", apply_conv_gate=True,
                         apply_pm_persistence=True, apply_filter_refill=True):
    """Run the production K=3 protocol but pick by `sort_key` instead of pred.

    sort_key='alpha_A' = oracle (perfect foresight of realized residual).
    sort_key='pred'    = production (model prediction).

    Returns sleeve records DataFrame compatible with psl.aggregate_sleeves.
    """
    df = apd.sort_values(["open_time", "symbol"]).copy()
    df = df[df["fold"].isin(OOS_FOLDS)]
    times = sorted(df["open_time"].unique())
    keep_t = set(times[::psl.HORIZON_ENTRY])
    df = df[df["open_time"].isin(keep_t)]
    times = sorted(df["open_time"].unique())
    fold_lookup = df.groupby("open_time")["fold"].first().to_dict()
    hist_disp = deque(maxlen=psl.GATE_LOOKBACK)
    hist_basket = []
    cur_long, cur_short = set(), set()
    is_flat = False
    picks_hist = defaultdict(list)
    by_t = {t: g for t, g in df.groupby("open_time")}
    records = []
    for t in times:
        g = by_t.get(t)
        if g is None: continue
        u = universe_fn(t)
        g_u = g[g["symbol"].isin(u)] if u else g.iloc[0:0]
        if len(g_u) < 2 * K + 1:
            records.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "long_basket": [], "short_basket": [], "traded": False})
            continue
        sym_arr = g_u["symbol"].to_numpy()
        score_arr = g_u[sort_key].to_numpy()  # ← key change: oracle uses alpha_A
        exit_l = dict(zip(sym_arr, g_u["exit_time"].to_numpy()))
        ret_l = dict(zip(sym_arr, g_u["return_pct"].to_numpy()))

        # NaN guard for oracle (some symbols may have missing alpha)
        valid = ~np.isnan(score_arr)
        if valid.sum() < 2 * K + 1:
            records.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "long_basket": [], "short_basket": [], "traded": False})
            continue
        sym_arr = sym_arr[valid]; score_arr = score_arr[valid]

        idx_t = np.argpartition(-score_arr, K - 1)[:K]
        idx_b = np.argpartition(score_arr, K - 1)[:K]
        disp = float(score_arr[idx_t].mean() - score_arr[idx_b].mean())
        skip = False
        if apply_conv_gate and len(hist_disp) >= 30:
            thr = float(np.quantile(list(hist_disp), psl.GATE_PCTILE))
            if disp < thr: skip = True
        hist_disp.append(disp)
        if skip:
            records.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "long_basket": [], "short_basket": [], "traded": False})
            if not is_flat and (cur_long or cur_short):
                is_flat = True; cur_long, cur_short = set(), set()
            continue

        order_d = np.argsort(-score_arr); order_a = np.argsort(score_arr)
        long_r = [sym_arr[i] for i in order_d]
        short_r = [sym_arr[i] for i in order_a]
        if apply_filter_refill:
            cand_l, _ = psl.select_refill(long_r, "long", K, picks_hist, 90, t)
            cand_s, _ = psl.select_refill(short_r, "short", K, picks_hist, 90, t)
        else:
            cand_l = long_r[:K]; cand_s = short_r[:K]
        c_ls = set(cand_l); c_ss = set(cand_s)
        hist_basket.append({"long": c_ls, "short": c_ss})
        if len(hist_basket) > psl.PM_M:
            hist_basket = hist_basket[-psl.PM_M:]
        if apply_pm_persistence and len(hist_basket) >= psl.PM_M:
            p_l = [h["long"] for h in hist_basket[-psl.PM_M:][:psl.PM_M - 1]]
            p_s = [h["short"] for h in hist_basket[-psl.PM_M:][:psl.PM_M - 1]]
            nl = cur_long & c_ls; ns = cur_short & c_ss
            for s_ in c_ls - cur_long:
                if all(s_ in p for p in p_l): nl.add(s_)
            for s_ in c_ss - cur_short:
                if all(s_ in p for p in p_s): ns.add(s_)
            if len(nl) > K:
                nl = set(sorted(nl, key=lambda s_: -score_arr[np.where(sym_arr == s_)[0][0]])[:K])
            if len(ns) > K:
                ns = set(sorted(ns, key=lambda s_: score_arr[np.where(sym_arr == s_)[0][0]])[:K])
        else:
            nl, ns = c_ls, c_ss
        if not nl or not ns:
            records.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "long_basket": [], "short_basket": [], "traded": False})
            if not is_flat and (cur_long or cur_short):
                is_flat = True; cur_long, cur_short = set(), set()
            continue
        # Update picks_hist using realized PnL (only when not in oracle mode? — we still need
        # the filter_refill mechanism to behave; use realized PnL same as production)
        if apply_filter_refill:
            for s_ in nl:
                picks_hist[(s_, "long")].append((t, exit_l[s_], ret_l[s_] * 1e4 / len(nl)))
            for s_ in ns:
                picks_hist[(s_, "short")].append((t, exit_l[s_], -ret_l[s_] * 1e4 / len(ns)))
        records.append({"time": t, "fold": fold_lookup.get(t, 0),
                          "long_basket": sorted(list(nl)),
                          "short_basket": sorted(list(ns)),
                          "traded": True})
        cur_long, cur_short = nl, ns
        is_flat = False
    return pd.DataFrame(records)


def cross_section_dispersion(apd, K=3):
    """Per-cycle: top-K mean realized α_β minus bot-K mean realized α_β.
    Independent of strategy mechanics — pure cross-sectional opportunity."""
    rows = []
    for t, g in apd.groupby("open_time"):
        g = g.dropna(subset=["alpha_A"])
        if len(g) < 2 * K + 1: continue
        a = g["alpha_A"].to_numpy()
        idx_t = np.argpartition(-a, K - 1)[:K]
        idx_b = np.argpartition(a, K - 1)[:K]
        spread = float(a[idx_t].mean() - a[idx_b].mean())
        rows.append({"time": t, "spread_alpha_bps": spread * 1e4, "n_elig": len(g)})
    return pd.DataFrame(rows)


def main():
    print("=== Oracle ceiling for Phase 1D (β-neutral residual target) ===\n", flush=True)
    apd = pd.read_parquet(APD_PATH)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    panel_syms = sorted(apd["symbol"].unique())
    listings = psl.get_listings()
    print(f"Loaded {len(apd):,} predictions, {len(panel_syms)} symbols", flush=True)

    def elig_pit(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=psl.MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]
    universe_real = psl.build_rolling_ic_universe(apd, sampled_t, psl.TOP_N, elig_pit)
    print(f"Rolling-IC universe built (top-{psl.TOP_N} per refresh boundary)\n", flush=True)

    # Load fwd_rets once (for MTM)
    print("Loading close prices for 51 symbols...", flush=True)
    t0 = time.time()
    frames = []
    for sym in panel_syms:
        sd = KLINES_DIR / sym / "5m"
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
    print(f"  ready ({time.time()-t0:.0f}s)\n", flush=True)

    # ====================================================================
    # C. Cross-sectional dispersion diagnostic (no strategy mechanics)
    # ====================================================================
    print("=" * 70)
    print("C. Cross-sectional dispersion of realized α_β (full 51, no filter)")
    print("=" * 70)
    disp_full = cross_section_dispersion(apd, K=3)
    disp_univ_rows = []
    for t in sampled_t:
        u = universe_real.get(t, set())
        g = apd[(apd["open_time"] == t) & (apd["symbol"].isin(u))]
        g = g.dropna(subset=["alpha_A"])
        if len(g) < 7: continue
        a = g["alpha_A"].to_numpy()
        idx_t = np.argpartition(-a, 2)[:3]
        idx_b = np.argpartition(a, 2)[:3]
        disp_univ_rows.append({
            "time": t,
            "spread_alpha_bps": (a[idx_t].mean() - a[idx_b].mean()) * 1e4,
            "n_elig": len(g),
        })
    disp_univ = pd.DataFrame(disp_univ_rows)

    def disp_summary(df, label):
        s = df["spread_alpha_bps"]
        print(f"\n  {label}", flush=True)
        print(f"    n_cycles      : {len(df)}", flush=True)
        print(f"    spread mean   : {s.mean():+.1f} bps", flush=True)
        print(f"    spread median : {s.median():+.1f} bps", flush=True)
        print(f"    spread p25    : {s.quantile(0.25):+.1f} bps", flush=True)
        print(f"    spread p75    : {s.quantile(0.75):+.1f} bps", flush=True)
        print(f"    spread std    : {s.std():.1f} bps", flush=True)
    disp_summary(disp_full, "Full 51 universe, top-3 vs bot-3 (4h fwd)")
    disp_summary(disp_univ, "Rolling-IC top-15 universe, top-3 vs bot-3 (4h fwd, sampled at entry cadence)")

    # ====================================================================
    # A. Oracle K=3 on Phase 1D universe (rolling-IC top-15)
    # ====================================================================
    print("\n" + "=" * 70)
    print("A. Oracle K=3 on Phase 1D's rolling-IC top-15 universe")
    print("=" * 70)
    def universe_fn_real(t):
        return universe_real.get(t, set())
    records_A = run_oracle_protocol(apd, universe_fn_real, K=psl.K, sort_key="alpha_A")
    df_A = psl.aggregate_sleeves(records_A, fwd_rets_4h)
    net_A = df_A["net_pnl_bps"].to_numpy()
    sh_A, lo_A, hi_A = block_bootstrap_ci(net_A, statistic=_sharpe, block_size=7, n_boot=1000)
    print(f"\n  Oracle Sharpe       : {sh_A:+.2f} [{lo_A:+.2f}, {hi_A:+.2f}]", flush=True)
    print(f"  Oracle totPnL       : {net_A.sum():+.0f} bps", flush=True)
    print(f"  Oracle maxDD        : {_max_dd(net_A):+.0f} bps", flush=True)
    print(f"  Oracle gross/cycle  : {df_A['gross_pnl_bps'].mean():+.2f} bps", flush=True)
    print(f"  Oracle cost/cycle   : {df_A['cost_bps'].mean():+.2f} bps", flush=True)
    print(f"  Oracle traded       : {records_A['traded'].sum()}/{len(records_A)}", flush=True)
    print(f"  Oracle folds_pos    : {folds_positive(df_A)}/9", flush=True)
    print(f"\n  Reference Phase 1D actual: Sharpe +0.65", flush=True)

    # ====================================================================
    # B. Oracle K=3 on full 51 (no universe filter)
    # ====================================================================
    print("\n" + "=" * 70)
    print("B. Oracle K=3 on full 51 (no universe filter; K=3 from ~50 candidates)")
    print("=" * 70)
    def universe_fn_full(t):
        return elig_pit(int(pd.Timestamp(t).timestamp() * 1000))
    records_B = run_oracle_protocol(apd, universe_fn_full, K=psl.K, sort_key="alpha_A")
    df_B = psl.aggregate_sleeves(records_B, fwd_rets_4h)
    net_B = df_B["net_pnl_bps"].to_numpy()
    sh_B, lo_B, hi_B = block_bootstrap_ci(net_B, statistic=_sharpe, block_size=7, n_boot=1000)
    print(f"\n  Oracle Sharpe       : {sh_B:+.2f} [{lo_B:+.2f}, {hi_B:+.2f}]", flush=True)
    print(f"  Oracle totPnL       : {net_B.sum():+.0f} bps", flush=True)
    print(f"  Oracle maxDD        : {_max_dd(net_B):+.0f} bps", flush=True)
    print(f"  Oracle gross/cycle  : {df_B['gross_pnl_bps'].mean():+.2f} bps", flush=True)
    print(f"  Oracle cost/cycle   : {df_B['cost_bps'].mean():+.2f} bps", flush=True)
    print(f"  Oracle traded       : {records_B['traded'].sum()}/{len(records_B)}", flush=True)
    print(f"  Oracle folds_pos    : {folds_positive(df_B)}/9", flush=True)

    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY — theoretical ceilings vs Phase 1D actual")
    print("=" * 70)
    print(f"  Phase 1D actual (model pred):                Sharpe +0.65", flush=True)
    print(f"  A. Oracle on Phase 1D universe (top-15):     Sharpe {sh_A:+.2f}", flush=True)
    print(f"  B. Oracle on full 51 (no filter):            Sharpe {sh_B:+.2f}", flush=True)
    print(f"\n  Model-side headroom (A − actual):           {sh_A - 0.65:+.2f} Sharpe", flush=True)
    print(f"  Universe-filter cost (B − A):               {sh_B - sh_A:+.2f} Sharpe", flush=True)
    print(f"  Total achievable (B − actual):              {sh_B - 0.65:+.2f} Sharpe", flush=True)

    disp_univ.to_csv(OUT / "disp_univ.csv", index=False)
    disp_full.to_csv(OUT / "disp_full.csv", index=False)
    df_A.to_csv(OUT / "oracle_A_phase1d_universe.csv", index=False)
    df_B.to_csv(OUT / "oracle_B_full_51.csv", index=False)
    print(f"\nSaved CSVs to {OUT}/", flush=True)


if __name__ == "__main__":
    main()
