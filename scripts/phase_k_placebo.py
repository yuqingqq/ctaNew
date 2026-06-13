"""Phase K placebo: matched-basket-size falsification on K4_cost_margin.

Tests whether K4's +1.88 Sharpe comes from genuine name-selection alpha or
just from exposure reduction (avg L/S = 0.8/0.8 vs production 1.7/1.8).

Procedure:
  - Load K4's per-cycle CSV; record (n_long, n_short) per cycle.
  - For each placebo seed: at each cycle, randomly select that exact number
    of names from the rolling-IC N=15 universe (uniform random, no pred-based
    selection). Compute spread + cost.
  - 100 seeds → Sharpe distribution of matched-exposure placebos.
  - If K4 Sharpe > p95 of placebos: K4's name selection is real signal.
  - If K4 Sharpe ≤ p95: K4's lift is the exposure-reduction Sharpe artifact.
"""
from __future__ import annotations
import sys, warnings, time
from pathlib import Path
from collections import deque, defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
from ml.research.alpha_v4_xs import block_bootstrap_ci

OUT = REPO / "outputs/vBTC_swap_rule"
APD_PATH = REPO / "outputs/vBTC_audit_panel/all_predictions.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
K4_PER_CYCLE = OUT / "per_cycle_K4_cost_margin.csv"

HORIZON = 48
CYCLES_PER_YEAR = (288 * 365) / HORIZON
COST_PER_LEG = 4.5
OOS_FOLDS = list(range(1, 10))
N_PLACEBO_SEEDS = 100
K = 4
MIN_HISTORY_DAYS = 60
IC_WINDOW_DAYS = 180
IC_UPDATE_DAYS = 90
MIN_OBS_PER_SYM = 100
TOP_N = 15
PROD_SKIP_PCTILE = 0.30


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def to_ms_int(s):
    ts = pd.to_datetime(s)
    if hasattr(ts.dtype, "tz") and ts.dtype.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return ts.astype("datetime64[ms]").astype("int64").to_numpy()


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


def build_rolling_ic_universe(all_pred_df, target_times, top_n, eligibility_at_t):
    bar_ms = 5 * 60 * 1000
    window_ms = IC_WINDOW_DAYS * 288 * bar_ms
    update_ms = IC_UPDATE_DAYS * 288 * bar_ms
    df = all_pred_df.copy()
    df["t_int"] = to_ms_int(df["open_time"])
    df["exit_t_int"] = to_ms_int(df["exit_time"])
    df_clean = df.dropna(subset=["alpha_A"])
    t0_ms = int(pd.Timestamp(target_times[0]).timestamp() * 1000)
    boundaries = []
    for t in target_times:
        t_ms = int(pd.Timestamp(t).timestamp() * 1000)
        n = (t_ms - t0_ms) // update_ms
        b = t0_ms + n * update_ms
        boundaries.append((t, b))
    unique_b = sorted(set(b for _, b in boundaries))
    b2u = {}
    for b in unique_b:
        elig = eligibility_at_t(b)
        past = df_clean[(df_clean["t_int"] >= b - window_ms) &
                          (df_clean["t_int"] < b) &
                          (df_clean["exit_t_int"] <= b) &
                          (df_clean["symbol"].isin(elig))]
        if len(past) < 1000:
            b2u[b] = set(); continue
        ics = past.groupby("symbol").apply(
            lambda g: g["pred"].rank().corr(g["alpha_A"].rank())
            if len(g) >= MIN_OBS_PER_SYM else np.nan
        )
        ics_sorted = ics.dropna().sort_values(ascending=False)
        b2u[b] = set(ics_sorted.head(top_n).index.tolist()) if top_n else set(ics_sorted.index)
    return {t: b2u[b] for t, b in boundaries}


def evaluate_random_matched(apd, universe, target_n_l, target_n_s, seed):
    """Replay K4's basket sizes but with RANDOM name selection from universe."""
    rng = np.random.RandomState(seed)
    df = apd.sort_values(["open_time", "symbol"]).copy()
    df = df[df["fold"].isin(OOS_FOLDS)]
    times = sorted(df["open_time"].unique())
    keep_t = set(times[::HORIZON])
    df = df[df["open_time"].isin(keep_t)]
    times = sorted(df["open_time"].unique())
    fold_lookup = df.groupby("open_time")["fold"].first().to_dict()

    hist_disp = deque(maxlen=252)
    cur_long, cur_short = set(), set()
    is_flat = False
    by_t = {t: g for t, g in df.groupby("open_time")}
    rows = []

    cycle_id = 0
    for t in times:
        g = by_t.get(t)
        if g is None: continue
        u = universe.get(t, set())
        g_u = g[g["symbol"].isin(u)] if u else g.iloc[0:0]
        sym_arr = g_u["symbol"].to_numpy()
        ret_l_map = dict(zip(sym_arr, g_u["return_pct"].to_numpy()))

        # Production conv_gate
        if len(g_u) >= 2 * K + 1:
            pred_arr = g_u["pred"].to_numpy()
            idx_t = np.argpartition(-pred_arr, K - 1)[:K]
            idx_b = np.argpartition(pred_arr, K - 1)[:K]
            disp = float(pred_arr[idx_t].mean() - pred_arr[idx_b].mean())
            skip = False
            if len(hist_disp) >= 30:
                thr = float(np.quantile(list(hist_disp), PROD_SKIP_PCTILE))
                if disp < thr: skip = True
            hist_disp.append(disp)
        else:
            skip = True

        if skip:
            if not is_flat and (cur_long or cur_short):
                rows.append({"time": t, "net_bps": -2 * COST_PER_LEG,
                              "n_long": 0, "n_short": 0})
                is_flat = True; cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "net_bps": 0,
                              "n_long": 0, "n_short": 0})
            cycle_id += 1
            continue

        # Random selection to match K4 basket size
        target_l = target_n_l[cycle_id] if cycle_id < len(target_n_l) else 0
        target_s = target_n_s[cycle_id] if cycle_id < len(target_n_s) else 0

        if target_l == 0 or target_s == 0 or len(sym_arr) < target_l + target_s:
            if not is_flat and (cur_long or cur_short):
                rows.append({"time": t, "net_bps": -2 * COST_PER_LEG,
                              "n_long": 0, "n_short": 0})
                is_flat = True; cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "net_bps": 0,
                              "n_long": 0, "n_short": 0})
            cycle_id += 1
            continue

        # Random partition of universe into longs/shorts
        shuffled = rng.permutation(len(sym_arr))
        nl_idx = shuffled[:target_l]
        ns_idx = shuffled[target_l:target_l + target_s]
        nl = set(sym_arr[nl_idx])
        ns = set(sym_arr[ns_idx])

        lr = [ret_l_map[s] for s in nl]
        sr = [ret_l_map[s] for s in ns]
        spread = (np.mean(lr) - np.mean(sr)) * 1e4
        if is_flat:
            cost = 2 * COST_PER_LEG; is_flat = False
        else:
            cl = len(nl.symmetric_difference(cur_long)) / max(len(nl | cur_long), 1)
            cs = len(ns.symmetric_difference(cur_short)) / max(len(ns | cur_short), 1)
            cost = (cl + cs) * COST_PER_LEG
        net = spread - cost
        rows.append({"time": t, "net_bps": net, "n_long": len(nl), "n_short": len(ns)})
        cur_long, cur_short = nl, ns
        cycle_id += 1
    return pd.DataFrame(rows)


def main():
    print("=== Phase K placebo: matched-basket-size falsification on K4 ===\n",
          flush=True)
    apd = pd.read_parquet(APD_PATH)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)

    listings = get_listings()
    panel_syms = set(apd["symbol"].unique())

    def eligibility_at(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::HORIZON]
    print(f"  Building universe...", flush=True)
    universe = build_rolling_ic_universe(apd, sampled_t, TOP_N, eligibility_at)

    # Load K4's per-cycle basket sizes
    k4 = pd.read_csv(K4_PER_CYCLE)
    target_n_l = k4["n_long"].tolist()
    target_n_s = k4["n_short"].tolist()
    k4_sharpe = _sharpe(k4["net_bps"].to_numpy())
    print(f"  K4 per-cycle: {len(k4)} cycles, real Sharpe={k4_sharpe:+.2f}",
          flush=True)
    print(f"  K4 basket size distribution: long mean={np.mean(target_n_l):.2f}, "
          f"short mean={np.mean(target_n_s):.2f}\n", flush=True)

    print(f"  Running {N_PLACEBO_SEEDS}-seed matched-basket-size placebo...",
          flush=True)
    t0 = time.time()
    placebo_sharpes = []
    placebo_pnls = []
    for seed in range(N_PLACEBO_SEEDS):
        df_v = evaluate_random_matched(apd, universe, target_n_l, target_n_s, seed)
        sh = _sharpe(df_v["net_bps"].to_numpy())
        placebo_sharpes.append(sh)
        placebo_pnls.append(float(df_v["net_bps"].sum()))
        if (seed + 1) % 25 == 0:
            print(f"  ... {seed+1}/{N_PLACEBO_SEEDS}  ({time.time()-t0:.0f}s)",
                  flush=True)

    p_sh = np.array(placebo_sharpes)
    p_pnl = np.array(placebo_pnls)
    p95 = np.percentile(p_sh, 95)
    rank = (p_sh < k4_sharpe).mean() * 100
    print(f"\n  Placebo Sharpe: mean={p_sh.mean():+.2f}, p50={np.percentile(p_sh,50):+.2f}, "
          f"p95={p95:+.2f}, max={p_sh.max():+.2f}", flush=True)
    print(f"  Placebo PnL: mean={p_pnl.mean():+.0f}, "
          f"p95={np.percentile(p_pnl,95):+.0f}", flush=True)
    print(f"\n  K4_cost_margin  Sharpe={k4_sharpe:+.2f}  totPnL={k4['net_bps'].sum():+.0f}",
          flush=True)
    print(f"  Rank vs placebo: {rank:.0f}%   "
          f"beats_p95: {'PASS — name selection adds real alpha' if k4_sharpe > p95 else 'FAIL — lift is exposure reduction artifact'}",
          flush=True)

    pd.DataFrame({"seed": range(N_PLACEBO_SEEDS),
                    "sharpe": p_sh, "total_pnl": p_pnl}).to_csv(
        OUT / "matched_basket_placebo.csv", index=False)
    print(f"\n  saved → {OUT}/matched_basket_placebo.csv", flush=True)


if __name__ == "__main__":
    main()
