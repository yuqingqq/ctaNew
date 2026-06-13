"""Step 52: WRAPPER-SYMMETRIC placebo on full-PIT V2 110-panel.

The cleanest "does the model add value?" test. Unlike Steps 35/42/45/48/50
(gate-consistent but NOT wrapper-symmetric — placebo bypassed select_refill
and picks_hist), here the ONLY difference between real and placebo is the
`pred` values:

  real:    pred = pred_z × trail_ic (the model)
  placebo: pred = deterministic random scores per (seed, cycle)

EVERYTHING else — conv_gate, select_refill, picks_hist feedback, PM_M,
sleeve overlay, causal aggregator — runs IDENTICALLY on whatever pred is
given. This isolates the model's prediction contribution from all the
architecture machinery.

Reference (gate-consistent, NOT wrapper-symmetric):
  Step 50 causal: Sharpe +3.11, P1 p100 (+1.45), P2 p100 (+1.61)
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
OUT = REPO / "linear_model/results/step52_wrapper_symmetric"
OUT.mkdir(parents=True, exist_ok=True)

OOS_FOLDS = psl.OOS_FOLDS
MIN_HISTORY_DAYS = 60
HOLD_BARS = 288
N_PLACEBO = 100
K = psl.K
PM_M = psl.PM_M
TOP_N = psl.TOP_N
GATE_PCTILE = psl.GATE_PCTILE
GATE_LOOKBACK = psl.GATE_LOOKBACK
HORIZON_ENTRY = psl.HORIZON_ENTRY


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


def run_protocol_wrapper_symmetric(apd, universe, placebo_seed=None):
    """IDENTICAL to psl.run_production_protocol_save_sleeves EXCEPT:
      - placebo: pred_arr replaced by deterministic random scores per (seed, cycle)
      - select_refill + picks_hist update run for BOTH real and placebo
    Only the pred VALUES differ between real and placebo; all machinery is shared.
    """
    df = apd.sort_values(["open_time", "symbol"]).copy()
    df = df[df["fold"].isin(OOS_FOLDS)]
    times = sorted(df["open_time"].unique())
    keep_t = set(times[::HORIZON_ENTRY])
    df = df[df["open_time"].isin(keep_t)]
    times = sorted(df["open_time"].unique())
    fold_lookup = df.groupby("open_time")["fold"].first().to_dict()
    hist_disp = deque(maxlen=GATE_LOOKBACK)
    hist_basket = []
    cur_long, cur_short = set(), set()
    is_flat = False
    picks_hist = defaultdict(list)
    by_t = {t: g for t, g in df.groupby("open_time")}
    records = []
    for ti, t in enumerate(times):
        g = by_t.get(t)
        if g is None: continue
        u = universe.get(t, set())
        g_u = g[g["symbol"].isin(u)] if u else g.iloc[0:0]
        if len(g_u) < 2 * K + 1:
            records.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "long_basket": [], "short_basket": [], "traded": False})
            continue
        sym_arr = g_u["symbol"].to_numpy()
        if placebo_seed is not None:
            # Deterministic random pred per (seed, cycle index) — replaces model pred
            rng_t = np.random.RandomState((placebo_seed * 100003 + ti) % (2**32))
            pred_arr = rng_t.permutation(len(sym_arr)).astype(float)
        else:
            pred_arr = g_u["pred"].to_numpy()
        exit_l = dict(zip(sym_arr, g_u["exit_time"].to_numpy()))
        ret_l = dict(zip(sym_arr, g_u["return_pct"].to_numpy()))
        idx_t = np.argpartition(-pred_arr, K - 1)[:K]
        idx_b = np.argpartition(pred_arr, K - 1)[:K]
        disp = float(pred_arr[idx_t].mean() - pred_arr[idx_b].mean())
        skip = False
        if len(hist_disp) >= 30:
            thr = float(np.quantile(list(hist_disp), GATE_PCTILE))
            if disp < thr: skip = True
        hist_disp.append(disp)
        if skip:
            records.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "long_basket": [], "short_basket": [], "traded": False})
            if not is_flat and (cur_long or cur_short):
                is_flat = True; cur_long, cur_short = set(), set()
            continue
        # SAME path for real & placebo: select_refill on pred-sorted lists
        order_d = np.argsort(-pred_arr); order_a = np.argsort(pred_arr)
        long_r = [sym_arr[i] for i in order_d]
        short_r = [sym_arr[i] for i in order_a]
        cand_l, _ = psl.select_refill(long_r, "long", K, picks_hist, 90, t)
        cand_s, _ = psl.select_refill(short_r, "short", K, picks_hist, 90, t)
        c_ls = set(cand_l); c_ss = set(cand_s)
        hist_basket.append({"long": c_ls, "short": c_ss})
        if len(hist_basket) > PM_M:
            hist_basket = hist_basket[-PM_M:]
        if len(hist_basket) >= PM_M:
            p_l = [h["long"] for h in hist_basket[-PM_M:][:PM_M - 1]]
            p_s = [h["short"] for h in hist_basket[-PM_M:][:PM_M - 1]]
            nl = cur_long & c_ls; ns = cur_short & c_ss
            for s_ in c_ls - cur_long:
                if all(s_ in p for p in p_l): nl.add(s_)
            for s_ in c_ss - cur_short:
                if all(s_ in p for p in p_s): ns.add(s_)
            if len(nl) > K:
                nl = set(sorted(nl, key=lambda s_: -pred_arr[np.where(sym_arr == s_)[0][0]])[:K])
            if len(ns) > K:
                ns = set(sorted(ns, key=lambda s_: pred_arr[np.where(sym_arr == s_)[0][0]])[:K])
        else:
            nl, ns = c_ls, c_ss
        if not nl or not ns:
            records.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "long_basket": [], "short_basket": [], "traded": False})
            if not is_flat and (cur_long or cur_short):
                is_flat = True; cur_long, cur_short = set(), set()
            continue
        # picks_hist updated for BOTH real and placebo (wrapper-symmetric)
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


def aggregate_causal(records, alpha_wide):
    """Causal-immediate aggregator (Step 50 convention): gross = tw × alpha[t]."""
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


def main():
    print("=" * 100, flush=True)
    print("  STEP 52: WRAPPER-SYMMETRIC placebo (full-PIT V2 110-panel, causal)", flush=True)
    print("=" * 100, flush=True)
    print("  Only difference real vs placebo: pred = model pred_B vs random scores", flush=True)
    print("  select_refill + picks_hist + conv_gate + PM all run IDENTICALLY", flush=True)
    print("  Reference (gate-consistent only, Step 50): +3.11, P1 +1.45, P2 +1.61", flush=True)
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
    for s, tt in apd_full.groupby("symbol")["open_time"].min().items():
        if s not in listings:
            tt = tt.tz_convert("UTC") if tt.tz is not None else tt.tz_localize("UTC")
            listings[s] = tt
    panel_syms_set = set(panel_syms)
    def elig_pit(b):
        ts = b if isinstance(b, pd.Timestamp) else pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms_set if listings.get(s) and listings[s] <= cutoff}
    target_t = sorted(apd_full[apd_full["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::HORIZON_ENTRY]

    apd_full["pred_B"] = apd_full["pred_z"] * apd_full["trail_ic"]
    # Build rolling-IC universe from pred_z (production convention, matches Step 47/50);
    # within-universe ranking uses pred_B / random in the protocol below.
    apd_full["pred"] = apd_full["pred_z"]
    universe_V2 = psl.build_rolling_ic_universe(apd_full, sampled_t, TOP_N, elig_pit)
    apd_full["pred"] = apd_full["pred_B"]
    alpha_wide = apd_full.pivot_table(index="open_time", columns="symbol",
                                        values="alpha_A", aggfunc="first").sort_index()

    # REAL (model pred through symmetric wrapper)
    print(f"\n--- Real V2 (model pred) through wrapper-symmetric protocol ---", flush=True)
    records_real = run_protocol_wrapper_symmetric(apd_full, universe_V2, placebo_seed=None)
    df_real = aggregate_causal(records_real, alpha_wide)
    net_real = df_real["net_pnl_bps"].to_numpy()
    sh_real = _sharpe(net_real)
    sh_lo, sh_hi = block_bootstrap_ci(net_real, statistic=_sharpe,
                                        block_size=7, n_boot=1000)[1:]
    fp = folds_positive(df_real)
    print(f"  Sharpe = {sh_real:+.2f} [{sh_lo:+.2f},{sh_hi:+.2f}], folds+={fp}/9, "
          f"gross={df_real['gross_pnl_bps'].mean():+.2f}", flush=True)
    df_real.to_csv(OUT / "per_cycle_real.csv", index=False)

    # WRAPPER-SYMMETRIC placebos (random pred through SAME machinery)
    print(f"\n--- Wrapper-symmetric placebos × {N_PLACEBO} (random pred, same wrapper) ---",
          flush=True)
    ps = []
    for seed in range(N_PLACEBO):
        records_p = run_protocol_wrapper_symmetric(apd_full, universe_V2,
                                                     placebo_seed=seed)
        df_p = aggregate_causal(records_p, alpha_wide)
        ps.append(_sharpe(df_p["net_pnl_bps"].to_numpy()))
        if (seed + 1) % 25 == 0:
            print(f"  seed {seed+1}: mean={np.mean(ps):+.3f}", flush=True)
    ps = np.array(ps)
    p95 = float(np.percentile(ps, 95))
    p99 = float(np.percentile(ps, 99))
    pcts = np.percentile(ps, [5, 25, 50, 75, 95, 99])
    print(f"  Placebo mean: {ps.mean():+.3f}  std: {ps.std():.3f}", flush=True)
    print(f"  p5/p25/p50/p75/p95/p99: " + "/".join(f"{x:+.2f}" for x in pcts), flush=True)
    print(f"  max placebo: {ps.max():+.2f}", flush=True)
    print(f"  Real rank: {(ps < sh_real).mean()*100:.1f}%", flush=True)
    print(f"  Edge over p95: {sh_real - p95:+.2f}", flush=True)
    print(f"  Edge over p99: {sh_real - p99:+.2f}", flush=True)

    pd.DataFrame({"placebo_sharpe": ps}).to_csv(OUT / "placebos.csv", index=False)

    print(f"\n{'='*100}", flush=True)
    print(f"  WRAPPER-SYMMETRIC VERDICT", flush=True)
    print(f"{'='*100}", flush=True)
    pass95 = sh_real > p95
    print(f"  Real Sharpe (model pred):  {sh_real:+.2f}  CI=[{sh_lo:+.2f},{sh_hi:+.2f}]",
          flush=True)
    print(f"  Wrapper-symmetric placebo p95: {p95:+.2f}  →  "
          f"{'PASS' if pass95 else 'FAIL'} (edge {sh_real - p95:+.2f})", flush=True)
    print(f"  vs Step 50 (gate-consistent only): real +3.11, P1 +1.45, P2 +1.61", flush=True)
    print(f"\n  Interpretation: if real still beats this placebo, the model's", flush=True)
    print(f"  prediction adds value beyond the wrapper machinery (refill/picks_hist/", flush=True)
    print(f"  gate) — the cleanest model-vs-random test.", flush=True)
    pd.DataFrame([{"sharpe_real": sh_real, "sh_lo": sh_lo, "sh_hi": sh_hi,
                    "folds_pos": fp, "placebo_p95": p95, "placebo_p99": p99,
                    "placebo_max": float(ps.max()),
                    "edge_p95": sh_real - p95, "pass": pass95}]).to_csv(
        OUT / "verdict.csv", index=False)
    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
