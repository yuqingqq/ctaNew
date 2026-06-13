"""Phase N: combinations of individually-failing-but-best-of-failures variants.

Each individual test in Phase L ranked > p50 of its matched placebo. None
cleared p95. If two are independent, their combination might pass.

Combinations:
  C1 = shrinkage_IC (λ=20) universe + composite gate (pred_disp × xs_dispersion)
  C2 = shrinkage_IC (λ=20) universe + rank_stability hard_skip pct30

Each combination keeps WINNER_21 + filter_refill + PM + flat_real. Only the
universe-selection rule and the gate signal change.

Matched skip-placebo on best. Pass: lift ≥ +0.3, ≥6/9 folds, beats p95.
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

OUT = REPO / "outputs/vBTC_combinations"
OUT.mkdir(parents=True, exist_ok=True)
APD_PATH = REPO / "outputs/vBTC_audit_panel/all_predictions.parquet"
PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"

HORIZON = 48
CYCLES_PER_YEAR = (288 * 365) / HORIZON
COST_PER_LEG = 4.5
GATE_LOOKBACK = 252
GATE_PCTILE = 0.30
PM_M = 2
K = 4
MIN_PICKS_FOR_FILTER = 30
MIN_OBS_PER_SYM = 100
IC_WINDOW_DAYS = 180
IC_UPDATE_DAYS = 90
OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
TOP_N = 15
N_PLACEBO_SEEDS = 100
SHRINK_LAMBDA = 20.0


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


def build_universe(apd, target_times, top_n, eligibility_at_t, shrink_lambda):
    """Universe = top-N by (IC - shrink_lambda * SE_IC). λ=0 reproduces production."""
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
        def compute_ic_se(gg):
            if len(gg) < MIN_OBS_PER_SYM: return pd.Series([np.nan, np.nan])
            ic = gg["pred"].rank().corr(gg["alpha_A"].rank())
            if pd.isna(ic): return pd.Series([np.nan, np.nan])
            n = len(gg)
            se = np.sqrt((1 - ic ** 2) / max(n - 2, 1))
            return pd.Series([ic, se])
        ic_se = past.groupby("symbol").apply(compute_ic_se)
        ic_se.columns = ["ic", "se"]
        ic_se = ic_se.dropna()
        ic_se["score"] = ic_se["ic"] - shrink_lambda * ic_se["se"]
        b2u[b] = set(ic_se["score"].sort_values(ascending=False).head(top_n).index.tolist())
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


def evaluate(apd, universe, gate_mode, panel_by_time, rng_seed=0,
              matched_skip_rate=None):
    """Run protocol with chosen gate.

    gate_mode: "production" | "composite_xs" | "rank_stab_hard_skip" | "placebo"
    """
    df = apd.sort_values(["open_time", "symbol"]).copy()
    df = df[df["fold"].isin(OOS_FOLDS)]
    times = sorted(df["open_time"].unique())
    keep_t = set(times[::HORIZON])
    df = df[df["open_time"].isin(keep_t)]
    times = sorted(df["open_time"].unique())
    fold_lookup = df.groupby("open_time")["fold"].first().to_dict()
    hist_score = deque(maxlen=GATE_LOOKBACK)
    hist_basket = []
    cur_long, cur_short = set(), set()
    is_flat = False
    picks_hist = defaultdict(list)
    by_t = {t: g for t, g in df.groupby("open_time")}
    prev_top_long, prev_top_short = set(), set()
    rng = np.random.RandomState(rng_seed)
    rows = []

    for t in times:
        g = by_t.get(t)
        if g is None: continue
        u = universe.get(t, set())
        g_u = g[g["symbol"].isin(u)] if u else g.iloc[0:0]
        if len(g_u) < 2 * K + 1:
            rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                          "net_bps": 0, "n_long": 0, "n_short": 0})
            continue
        sym_arr = g_u["symbol"].to_numpy()
        pred_arr = g_u["pred"].to_numpy()
        ret_l = dict(zip(sym_arr, g_u["return_pct"].to_numpy()))
        exit_l = dict(zip(sym_arr, g_u["exit_time"].to_numpy()))
        idx_t = np.argpartition(-pred_arr, K - 1)[:K]
        idx_b = np.argpartition(pred_arr, K - 1)[:K]
        top_set = set(sym_arr[idx_t]); bot_set = set(sym_arr[idx_b])
        pred_disp = float(pred_arr[idx_t].mean() - pred_arr[idx_b].mean())

        # Rank stability
        if prev_top_long:
            overlap_l = len(top_set & prev_top_long) / K
            overlap_s = len(bot_set & prev_top_short) / K
            stability = (overlap_l + overlap_s) / 2
        else:
            stability = 1.0
        prev_top_long = top_set; prev_top_short = bot_set

        # xs_dispersion within universe
        xs_disp = 1.0
        p_at_t = panel_by_time.get(t) if panel_by_time else None
        if p_at_t is not None:
            p_u = p_at_t[p_at_t["symbol"].isin(u)]
            if len(p_u) > 0:
                v = p_u["xs_alpha_dispersion_48b"].mean()
                if not pd.isna(v): xs_disp = v

        # Compute score
        if gate_mode == "production":
            score = pred_disp
        elif gate_mode == "composite_xs":
            score = pred_disp * xs_disp
        elif gate_mode == "rank_stab_hard_skip":
            score = stability  # higher = more stable; skip low
        elif gate_mode == "placebo":
            score = pred_disp
        else:
            score = pred_disp

        # Skip decision
        if matched_skip_rate is not None:
            skip = rng.random() < matched_skip_rate
        else:
            skip = False
            if len(hist_score) >= 30:
                thr = float(np.quantile(list(hist_score), GATE_PCTILE))
                if score < thr: skip = True
            hist_score.append(score)

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
        cand_l, n_el = select_refill(long_r, "long", K, picks_hist, 90, t)
        cand_s, n_es = select_refill(short_r, "short", K, picks_hist, 90, t)
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


def main():
    print("=== Phase N: combinations of best-but-failing variants ===\n", flush=True)
    apd = pd.read_parquet(APD_PATH)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    panel = pd.read_parquet(PANEL_PATH,
                              columns=["open_time", "symbol", "xs_alpha_dispersion_48b"])
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel_by_time = {t: g for t, g in panel.groupby("open_time")}

    listings = get_listings()
    panel_syms = set(apd["symbol"].unique())
    def eligibility_at(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}
    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::HORIZON]

    print(f"  Building production universe (λ=0)...", flush=True)
    universe_prod = build_universe(apd, sampled_t, TOP_N, eligibility_at, 0.0)
    print(f"  Building shrinkage universe (λ={SHRINK_LAMBDA})...", flush=True)
    universe_shr = build_universe(apd, sampled_t, TOP_N, eligibility_at, SHRINK_LAMBDA)

    variants = [
        ("V0_production", universe_prod, "production"),
        ("V1_shrinkIC_only", universe_shr, "production"),
        ("V2_composite_only", universe_prod, "composite_xs"),
        ("V3_rank_stab_only", universe_prod, "rank_stab_hard_skip"),
        ("C1_shrinkIC_+_composite", universe_shr, "composite_xs"),
        ("C2_shrinkIC_+_rank_stab", universe_shr, "rank_stab_hard_skip"),
    ]

    print(f"\n  {'variant':<30}  {'Sharpe':>7}  {'CI':>17}  {'skip%':>6}  "
          f"{'maxDD':>7}  {'totPnL':>8}  {'pos_folds':>9}  per-fold", flush=True)
    results = []
    skip_rates = {}
    for label, u, gate_mode in variants:
        t0 = time.time()
        df_v = evaluate(apd, u, gate_mode, panel_by_time)
        net = df_v["net_bps"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        n_pos = 0
        per_fold = []
        for f in OOS_FOLDS:
            d = df_v[df_v["fold"] == f]["net_bps"].to_numpy()
            if len(d) >= 3:
                sh_f = _sharpe(d)
                per_fold.append(sh_f)
                if sh_f > 0: n_pos += 1
        pf_str = " ".join(f"{x:+.1f}" for x in per_fold)
        skip_rate = float((df_v["skipped"] == 1).mean())
        skip_rates[label] = skip_rate
        results.append({"variant": label, "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
                          "skip_rate": skip_rate, "max_dd": _max_dd(net),
                          "total_pnl": net.sum(), "n_folds_positive": n_pos})
        df_v.to_csv(OUT / f"per_cycle_{label}.csv", index=False)
        print(f"  {label:<30}  {sh:>+7.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  "
              f"{skip_rate*100:>5.1f}%  {_max_dd(net):>+7.0f}  {net.sum():>+8.0f}  "
              f"{n_pos:>5d}/9   {pf_str}  ({time.time()-t0:.0f}s)", flush=True)

    pd.DataFrame(results).to_csv(OUT / "results.csv", index=False)

    # Best combination (C1 or C2)
    combos = [r for r in results if r["variant"].startswith("C")]
    best = max(combos, key=lambda r: r["sharpe"])
    prod = next(r for r in results if r["variant"] == "V0_production")
    print(f"\n  Production: Sharpe={prod['sharpe']:+.2f}, "
          f"folds={prod['n_folds_positive']}/9", flush=True)
    print(f"  Best combination ({best['variant']}): Sharpe={best['sharpe']:+.2f}, "
          f"folds={best['n_folds_positive']}/9", flush=True)
    lift = best['sharpe'] - prod['sharpe']
    print(f"  Lift: {lift:+.2f}", flush=True)

    # Matched skip-placebo on best combination
    print(f"\n--- Matched placebo (skip rate {best['skip_rate']*100:.1f}%, "
          f"shrinkage universe, {N_PLACEBO_SEEDS} seeds) ---", flush=True)
    universe_used = universe_shr if "shrink" in best["variant"] else universe_prod
    t0 = time.time()
    placebo_sh = []
    for seed in range(N_PLACEBO_SEEDS):
        df_p = evaluate(apd, universe_used, "placebo", panel_by_time,
                          rng_seed=seed, matched_skip_rate=best["skip_rate"])
        placebo_sh.append(_sharpe(df_p["net_bps"].to_numpy()))
        if (seed + 1) % 25 == 0:
            print(f"  ... {seed+1}/{N_PLACEBO_SEEDS}  ({time.time()-t0:.0f}s)",
                  flush=True)
    p_sh = np.array(placebo_sh)
    p95 = float(np.percentile(p_sh, 95))
    rank = float((p_sh < best['sharpe']).mean() * 100)
    print(f"\n  Placebo: mean={p_sh.mean():+.2f}, p50={np.percentile(p_sh,50):+.2f}, "
          f"p95={p95:+.2f}, max={p_sh.max():+.2f}", flush=True)
    print(f"  Best ranks p{rank:.0f}  beats_p95="
          f"{'PASS' if best['sharpe'] > p95 else 'FAIL'}", flush=True)

    print(f"\n=== Phase N verdict ===", flush=True)
    pass_prod = lift >= 0.3
    pass_folds = best['n_folds_positive'] >= 6
    pass_placebo = best['sharpe'] > p95
    print(f"  Beats production (Δ ≥ +0.3):  {'PASS' if pass_prod else 'FAIL'} ({lift:+.2f})",
          flush=True)
    print(f"  ≥6/9 folds positive:           {'PASS' if pass_folds else 'FAIL'} "
          f"({best['n_folds_positive']}/9)", flush=True)
    print(f"  Beats placebo p95:             {'PASS' if pass_placebo else 'FAIL'}",
          flush=True)
    if pass_prod and pass_folds and pass_placebo:
        print(f"  → ADOPT {best['variant']}", flush=True)
    else:
        print(f"  → NOT ADOPTED", flush=True)
    print(f"\n  saved → {OUT}", flush=True)


if __name__ == "__main__":
    main()
