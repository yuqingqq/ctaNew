"""Phase T: middle-zone calibration gate on 51-panel K=3.

Motivation: decile analysis shows inverted-U pattern in pred_disp vs realized
spread. Deciles 0-2 (low pred): flat-to-small positive. Deciles 3-5 (moderate
pred): strongly positive (+9 to +24 bps). Deciles 6-9 (high pred): NEGATIVE
(-3 to -16 bps). Production conv_gate only skips decile 0; trades right
through the loss zone (deciles 6-9).

Variants (all with K=3, filter_refill, PM, on 51-panel):
  V0 production:          skip if pred_disp < 30th pctile (current)
  V1 middle_40:           skip if pred_disp NOT in [30th, 70th] pctile
  V2 middle_60:           skip if pred_disp NOT in [20th, 80th] pctile
  V3 skip_top_only:       skip if pred_disp > 70th pctile

Matched skip-placebo on best non-V0 variant.
Pass: lift ≥ +0.3, ≥6/9 folds, beats placebo p95, concentration ≤ 40%.
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

OUT = REPO / "outputs/vBTC_middle_zone_gate"
OUT.mkdir(parents=True, exist_ok=True)
APD_PATH = REPO / "outputs/vBTC_audit_panel/all_predictions.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"

HORIZON = 48
CYCLES_PER_YEAR = (288 * 365) / HORIZON
COST_PER_LEG = 4.5
GATE_LOOKBACK = 252
PM_M = 2
K = 3
MIN_PICKS_FOR_FILTER = 30
MIN_OBS_PER_SYM = 100
IC_WINDOW_DAYS = 180
IC_UPDATE_DAYS = 90
OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
TOP_N = 15
N_PLACEBO_SEEDS = 100


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


def build_rolling_ic_universe(apd, target_times, top_n, eligibility_at_t):
    bar_ms = 5 * 60 * 1000
    window_ms = IC_WINDOW_DAYS * 288 * bar_ms
    update_ms = IC_UPDATE_DAYS * 288 * bar_ms
    df = apd.copy()
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


def filter_decision(hist, window_days, t):
    if not hist: return True
    cutoff = t - pd.Timedelta(days=window_days)
    vals = [c for (tt, et, c) in hist if et <= t and tt >= cutoff]
    if len(vals) < MIN_PICKS_FOR_FILTER: return True
    return float(np.mean(vals)) >= 0


def select_refill(ranked, side, k, hist, window, t):
    kept = []; n_excl = 0
    for s in ranked:
        if len(kept) >= k: break
        if filter_decision(hist.get((s, side), []), window, t):
            kept.append(s)
        else: n_excl += 1
    return kept, n_excl


def gate_decision(disp, hist_disp, gate_kind):
    """Return skip:bool given current dispersion + history of past dispersions."""
    if len(hist_disp) < 30:
        return False
    h = np.array(hist_disp)
    if gate_kind == "production":
        thr_lo = float(np.quantile(h, 0.30))
        return disp < thr_lo
    if gate_kind == "middle_40":
        thr_lo = float(np.quantile(h, 0.30))
        thr_hi = float(np.quantile(h, 0.70))
        return (disp < thr_lo) or (disp > thr_hi)
    if gate_kind == "middle_60":
        thr_lo = float(np.quantile(h, 0.20))
        thr_hi = float(np.quantile(h, 0.80))
        return (disp < thr_lo) or (disp > thr_hi)
    if gate_kind == "skip_top_only":
        thr_hi = float(np.quantile(h, 0.70))
        return disp > thr_hi
    if gate_kind == "no_gate":
        return False
    return False


def evaluate(apd, universe, gate_kind, K_target=K, placebo_seed=None,
              matched_skip_rate=None):
    df = apd.sort_values(["open_time", "symbol"]).copy()
    df = df[df["fold"].isin(OOS_FOLDS)]
    times = sorted(df["open_time"].unique())
    keep_t = set(times[::HORIZON])
    df = df[df["open_time"].isin(keep_t)]
    times = sorted(df["open_time"].unique())
    fold_lookup = df.groupby("open_time")["fold"].first().to_dict()
    hist_disp = deque(maxlen=GATE_LOOKBACK)
    hist_basket = []
    cur_long, cur_short = set(), set()
    is_flat = False
    picks_hist = defaultdict(list)
    by_t = {t: g for t, g in df.groupby("open_time")}
    rng = np.random.RandomState(placebo_seed if placebo_seed is not None else 0)
    rows = []
    for t in times:
        g = by_t.get(t)
        if g is None: continue
        u = universe.get(t, set())
        g_u = g[g["symbol"].isin(u)] if u else g.iloc[0:0]
        if len(g_u) < 2 * K_target + 1:
            rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                          "net_bps": 0, "n_long": 0, "n_short": 0})
            continue
        sym_arr = g_u["symbol"].to_numpy()
        pred_arr = g_u["pred"].to_numpy()
        ret_l = dict(zip(sym_arr, g_u["return_pct"].to_numpy()))
        exit_l = dict(zip(sym_arr, g_u["exit_time"].to_numpy()))
        idx_t = np.argpartition(-pred_arr, K_target - 1)[:K_target]
        idx_b = np.argpartition(pred_arr, K_target - 1)[:K_target]
        disp = float(pred_arr[idx_t].mean() - pred_arr[idx_b].mean())

        if matched_skip_rate is not None:
            skip = rng.random() < matched_skip_rate
            hist_disp.append(disp)
        else:
            skip = gate_decision(disp, hist_disp, gate_kind)
            hist_disp.append(disp)

        if skip:
            if not is_flat and (cur_long or cur_short):
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                              "net_bps": -2 * COST_PER_LEG, "n_long": 0, "n_short": 0})
                is_flat = True; cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                              "net_bps": 0, "n_long": 0, "n_short": 0})
            continue
        order_d = np.argsort(-pred_arr); order_a = np.argsort(pred_arr)
        long_r = [sym_arr[i] for i in order_d]
        short_r = [sym_arr[i] for i in order_a]
        cand_l, _ = select_refill(long_r, "long", K_target, picks_hist, 90, t)
        cand_s, _ = select_refill(short_r, "short", K_target, picks_hist, 90, t)
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
            if len(nl) > K_target:
                nl = set(sorted(nl, key=lambda s_: -pred_arr[np.where(sym_arr == s_)[0][0]])[:K_target])
            if len(ns) > K_target:
                ns = set(sorted(ns, key=lambda s_: pred_arr[np.where(sym_arr == s_)[0][0]])[:K_target])
        else:
            nl, ns = c_ls, c_ss
        if not nl or not ns:
            if not is_flat and (cur_long or cur_short):
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                              "net_bps": -2 * COST_PER_LEG, "n_long": 0, "n_short": 0})
                is_flat = True; cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 0,
                              "net_bps": 0, "n_long": 0, "n_short": 0})
            continue
        lr = [ret_l[s_] for s_ in nl]; sr = [ret_l[s_] for s_ in ns]
        spread = (np.mean(lr) - np.mean(sr)) * 1e4
        if is_flat:
            cost = 2 * COST_PER_LEG; is_flat = False
        else:
            cl = len(nl.symmetric_difference(cur_long)) / max(len(nl | cur_long), 1)
            cs = len(ns.symmetric_difference(cur_short)) / max(len(ns | cur_short), 1)
            cost = (cl + cs) * COST_PER_LEG
        net = spread - cost
        for s_ in nl:
            picks_hist[(s_, "long")].append((t, exit_l[s_], ret_l[s_] * 1e4 / len(nl)))
        for s_ in ns:
            picks_hist[(s_, "short")].append((t, exit_l[s_], -ret_l[s_] * 1e4 / len(ns)))
        rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 0,
                      "net_bps": net, "n_long": len(nl), "n_short": len(ns)})
        cur_long, cur_short = nl, ns
    return pd.DataFrame(rows)


def fold_concentration(df_v):
    fold_pnls = df_v.groupby("fold")["net_bps"].sum()
    pos = fold_pnls[fold_pnls > 0]
    total_pos = pos.sum() if len(pos) > 0 else 0
    if total_pos <= 0: return 0.0
    return float(pos.max() / total_pos)


def main():
    print("=== Phase T: middle-zone calibration gate ===\n", flush=True)
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

    variants = ["production", "middle_40", "middle_60", "skip_top_only", "no_gate"]
    print(f"\n  {'variant':<20}  {'Sharpe':>7}  {'CI':>17}  {'skip%':>6}  "
          f"{'maxDD':>7}  {'totPnL':>8}  {'pos_folds':>9}  {'conc':>5}", flush=True)
    results = {}
    for v in variants:
        t0 = time.time()
        df_v = evaluate(apd, universe, v)
        net = df_v["net_bps"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        skip_rate = float((df_v["skipped"] == 1).mean())
        n_pos = 0
        for f in OOS_FOLDS:
            d = df_v[df_v["fold"] == f]["net_bps"].to_numpy()
            if len(d) >= 3 and _sharpe(d) > 0: n_pos += 1
        conc = fold_concentration(df_v)
        results[v] = {"sharpe": sh, "ci_lo": lo, "ci_hi": hi,
                        "max_dd": _max_dd(net), "total_pnl": net.sum(),
                        "skip_rate": skip_rate, "n_folds_positive": n_pos,
                        "concentration": conc, "df": df_v}
        df_v.to_csv(OUT / f"per_cycle_{v}.csv", index=False)
        print(f"  {v:<20}  {sh:>+7.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  "
              f"{skip_rate*100:>5.1f}%  {_max_dd(net):>+7.0f}  {net.sum():>+8.0f}  "
              f"{n_pos:>5d}/9  {conc*100:>3.0f}%  ({time.time()-t0:.0f}s)", flush=True)

    prod = results["production"]
    cands = {k: v for k, v in results.items() if k != "production" and k != "no_gate"}
    best_name = max(cands, key=lambda k: cands[k]["sharpe"])
    best = cands[best_name]
    print(f"\n  Production: Sharpe={prod['sharpe']:+.2f}, folds={prod['n_folds_positive']}/9",
          flush=True)
    print(f"  Best alternative ({best_name}): Sharpe={best['sharpe']:+.2f}, "
          f"folds={best['n_folds_positive']}/9, conc={best['concentration']*100:.0f}%",
          flush=True)
    lift = best['sharpe'] - prod['sharpe']
    print(f"  Lift: {lift:+.2f}", flush=True)

    # Matched skip-placebo on best alternative
    print(f"\n--- Matched skip-placebo on {best_name} ({N_PLACEBO_SEEDS} seeds, "
          f"skip {best['skip_rate']*100:.1f}%) ---", flush=True)
    t0 = time.time()
    placebo_sh = []
    for seed in range(N_PLACEBO_SEEDS):
        df_p = evaluate(apd, universe, "no_gate", K, placebo_seed=seed,
                          matched_skip_rate=best["skip_rate"])
        placebo_sh.append(_sharpe(df_p["net_bps"].to_numpy()))
        if (seed + 1) % 25 == 0:
            print(f"  ... {seed+1}/{N_PLACEBO_SEEDS}  ({time.time()-t0:.0f}s)",
                  flush=True)
    p_sh = np.array(placebo_sh)
    p95 = float(np.percentile(p_sh, 95))
    rank = float((p_sh < best["sharpe"]).mean() * 100)
    print(f"\n  Placebo: mean={p_sh.mean():+.2f}, p50={np.percentile(p_sh,50):+.2f}, "
          f"p95={p95:+.2f}, max={p_sh.max():+.2f}", flush=True)
    print(f"  {best_name} ranks p{rank:.0f}  "
          f"beats_p95={'PASS' if best['sharpe'] > p95 else 'FAIL'}", flush=True)
    pd.DataFrame({"seed": range(N_PLACEBO_SEEDS), "sharpe": p_sh}).to_csv(
        OUT / f"matched_placebo_{best_name}.csv", index=False)

    print(f"\n=== Phase T verdict ===", flush=True)
    pass_lift = lift >= 0.3
    pass_folds = best['n_folds_positive'] >= 6
    pass_conc = best['concentration'] <= 0.40
    pass_placebo = best['sharpe'] > p95
    print(f"  Lift ≥ +0.30:           {'PASS' if pass_lift else 'FAIL'} ({lift:+.2f})",
          flush=True)
    print(f"  ≥6/9 folds positive:    {'PASS' if pass_folds else 'FAIL'} "
          f"({best['n_folds_positive']}/9)", flush=True)
    print(f"  Concentration ≤ 40%:    {'PASS' if pass_conc else 'FAIL'} "
          f"({best['concentration']*100:.0f}%)", flush=True)
    print(f"  Beats placebo p95:      {'PASS' if pass_placebo else 'FAIL'}",
          flush=True)
    if pass_lift and pass_folds and pass_conc and pass_placebo:
        print(f"  → ADOPT {best_name}", flush=True)
    else:
        print(f"  → NOT ADOPTED", flush=True)


if __name__ == "__main__":
    main()
