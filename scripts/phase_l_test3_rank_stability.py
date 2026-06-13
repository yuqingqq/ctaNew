"""Phase L / Test 3: rank-stability sizing gate.

Tests rank-stability signals at SOFTER sizing (0.5×) instead of hard skip.
The hard-skip version (V4 rank_instability) was rejected in Phase J at
Sharpe −0.79. This variant tests whether soft sizing rescues the signal.

Stability metric: topK_overlap_prev = jaccard(current top-K, prev top-K).
Higher = more stable rankings.

Variants (all with production conv_gate + filter_refill + PM):
  V0 production                  — baseline
  V1 size_0.5_low_stab_20pct     — size 0.5 on bottom-20% stability cycles
  V2 size_0.5_low_stab_30pct     — size 0.5 on bottom-30%
  V3 size_0.5_low_stab_40pct     — size 0.5 on bottom-40%
  V4 hard_skip_30pct             — hard skip bottom-30% (reference for J's V4)

Compare to matched-skip placebo (random skips at same rates as variants).

Pass: beats production AND beats matched placebo p95 AND ≥6/9 folds improve.
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

OUT = REPO / "outputs/vBTC_rank_stability"
OUT.mkdir(parents=True, exist_ok=True)
APD_PATH = REPO / "outputs/vBTC_audit_panel/all_predictions.parquet"
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


def evaluate(apd, universe, variant):
    """Run protocol with rank-stability variant.

    variant: dict with keys:
      kind: "production" | "size_low_stab" | "hard_skip_low_stab" | "matched_placebo"
      pctile: float — fraction of cycles considered "low stability"
      size: float — multiplier for low-stab cycles (0.5 default; 0 = skip)
      placebo_seed: int — for matched placebo
      placebo_skip_pctile: float — fraction to randomly downsize (matches variant rate)
    """
    df = apd.sort_values(["open_time", "symbol"]).copy()
    df = df[df["fold"].isin(OOS_FOLDS)]
    times = sorted(df["open_time"].unique())
    keep_t = set(times[::HORIZON])
    df = df[df["open_time"].isin(keep_t)]
    times = sorted(df["open_time"].unique())
    fold_lookup = df.groupby("open_time")["fold"].first().to_dict()
    hist_disp = deque(maxlen=GATE_LOOKBACK)
    hist_stab = deque(maxlen=GATE_LOOKBACK)
    hist_basket = []
    cur_long, cur_short = set(), set()
    is_flat = False
    picks_hist = defaultdict(list)
    by_t = {t: g for t, g in df.groupby("open_time")}
    rng = np.random.RandomState(variant.get("placebo_seed", 0))
    prev_top_long, prev_top_short = set(), set()
    rows = []

    kind = variant["kind"]
    pctile = variant.get("pctile", 0.30)
    size_mult = variant.get("size", 0.5)

    for t in times:
        g = by_t.get(t)
        if g is None: continue
        u = universe.get(t, set())
        g_u = g[g["symbol"].isin(u)] if u else g.iloc[0:0]
        if len(g_u) < 2 * K + 1:
            rows.append({"time": t, "fold": fold_lookup.get(t, 0), "spread_bps": 0,
                          "cost_bps": 0, "net_bps": 0, "n_long": 0, "n_short": 0,
                          "size_mult": 0})
            continue
        sym_arr = g_u["symbol"].to_numpy()
        pred_arr = g_u["pred"].to_numpy()
        ret_l = dict(zip(sym_arr, g_u["return_pct"].to_numpy()))
        exit_l = dict(zip(sym_arr, g_u["exit_time"].to_numpy()))
        idx_t = np.argpartition(-pred_arr, K - 1)[:K]
        idx_b = np.argpartition(pred_arr, K - 1)[:K]
        top_set = set(sym_arr[idx_t])
        bot_set = set(sym_arr[idx_b])
        # Stability metric (PIT — uses prev cycle's top sets)
        if prev_top_long:
            overlap_l = len(top_set & prev_top_long) / K
            overlap_s = len(bot_set & prev_top_short) / K
            stability = (overlap_l + overlap_s) / 2
        else:
            stability = 1.0
        prev_top_long = top_set; prev_top_short = bot_set
        disp = float(pred_arr[idx_t].mean() - pred_arr[idx_b].mean())
        # Conv gate
        skip_conv = False
        if len(hist_disp) >= 30:
            thr = float(np.quantile(list(hist_disp), GATE_PCTILE))
            if disp < thr: skip_conv = True
        hist_disp.append(disp)

        # Determine size multiplier
        this_size_mult = 1.0
        if kind == "production":
            this_size_mult = 1.0
        elif kind in ("size_low_stab", "hard_skip_low_stab"):
            if len(hist_stab) >= 30:
                thr_stab = float(np.quantile(list(hist_stab), pctile))
                if stability < thr_stab:
                    if kind == "size_low_stab":
                        this_size_mult = size_mult
                    else:  # hard_skip
                        this_size_mult = 0.0
            hist_stab.append(stability)
        elif kind == "matched_placebo":
            hist_stab.append(stability)
            target_rate = variant.get("placebo_skip_pctile", 0.30)
            if rng.random() < target_rate:
                this_size_mult = size_mult

        # If conv gate fires OR size_mult is 0, skip
        skip = skip_conv or (this_size_mult <= 0)
        if skip:
            if not is_flat and (cur_long or cur_short):
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "spread_bps": 0,
                              "cost_bps": 2 * COST_PER_LEG, "net_bps": -2 * COST_PER_LEG,
                              "n_long": 0, "n_short": 0, "size_mult": this_size_mult})
                is_flat = True; cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "spread_bps": 0,
                              "cost_bps": 0, "net_bps": 0, "n_long": 0, "n_short": 0,
                              "size_mult": this_size_mult})
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
            nl = cur_long & c_ls
            ns = cur_short & c_ss
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
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "spread_bps": 0,
                              "cost_bps": 2 * COST_PER_LEG, "net_bps": -2 * COST_PER_LEG,
                              "n_long": 0, "n_short": 0, "size_mult": this_size_mult})
                is_flat = True; cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "spread_bps": 0,
                              "cost_bps": 0, "net_bps": 0, "n_long": 0, "n_short": 0,
                              "size_mult": this_size_mult})
            continue
        lr = [ret_l[s_] for s_ in nl]; sr = [ret_l[s_] for s_ in ns]
        spread = (np.mean(lr) - np.mean(sr)) * 1e4 * this_size_mult
        if is_flat:
            cost = 2 * COST_PER_LEG; is_flat = False
        else:
            cl = len(nl.symmetric_difference(cur_long)) / max(len(nl | cur_long), 1)
            cs = len(ns.symmetric_difference(cur_short)) / max(len(ns | cur_short), 1)
            cost = (cl + cs) * COST_PER_LEG * this_size_mult
        net = spread - cost
        for s_ in nl:
            picks_hist[(s_, "long")].append((t, exit_l[s_], ret_l[s_] * 1e4 / len(nl)))
        for s_ in ns:
            picks_hist[(s_, "short")].append((t, exit_l[s_], -ret_l[s_] * 1e4 / len(ns)))
        rows.append({"time": t, "fold": fold_lookup.get(t, 0), "spread_bps": spread,
                      "cost_bps": cost, "net_bps": net,
                      "n_long": len(nl), "n_short": len(ns), "size_mult": this_size_mult})
        cur_long, cur_short = nl, ns
    return pd.DataFrame(rows)


def main():
    print("=== Phase L / Test 3: rank-stability sizing gate ===\n", flush=True)
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

    variants = [
        ("V0_production", {"kind": "production"}),
        ("V1_size0.5_pct20", {"kind": "size_low_stab", "pctile": 0.20, "size": 0.5}),
        ("V2_size0.5_pct30", {"kind": "size_low_stab", "pctile": 0.30, "size": 0.5}),
        ("V3_size0.5_pct40", {"kind": "size_low_stab", "pctile": 0.40, "size": 0.5}),
        ("V4_hard_skip_pct30", {"kind": "hard_skip_low_stab", "pctile": 0.30}),
    ]

    print(f"\n  {'variant':<22}  {'Sharpe':>7}  {'CI':>17}  {'maxDD':>7}  {'totPnL':>8}  "
          f"{'pos_folds':>9}  per-fold", flush=True)
    results = []
    for label, v_spec in variants:
        t0 = time.time()
        df_v = evaluate(apd, universe, v_spec)
        net = df_v["net_bps"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        per_fold = []
        n_pos = 0
        for f in OOS_FOLDS:
            d = df_v[df_v["fold"] == f]["net_bps"].to_numpy()
            if len(d) >= 3:
                sh_f = _sharpe(d)
                per_fold.append(sh_f)
                if sh_f > 0: n_pos += 1
        pf_str = " ".join(f"{x:+.1f}" for x in per_fold)
        results.append({"variant": label, "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
                          "max_dd": _max_dd(net), "total_pnl": net.sum(),
                          "n_folds_positive": n_pos,
                          "per_fold": pf_str})
        df_v.to_csv(OUT / f"per_cycle_{label}.csv", index=False)
        print(f"  {label:<22}  {sh:>+7.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  "
              f"{_max_dd(net):>+7.0f}  {net.sum():>+8.0f}  {n_pos:>5d}/9   {pf_str}",
              flush=True)

    pd.DataFrame(results).to_csv(OUT / "results.csv", index=False)

    # Best variant vs production
    prod = results[0]
    best = max(results[1:], key=lambda r: r["sharpe"])
    print(f"\n  Production: Sharpe={prod['sharpe']:+.2f}, "
          f"folds_pos={prod['n_folds_positive']}/9", flush=True)
    print(f"  Best variant ({best['variant']}): Sharpe={best['sharpe']:+.2f}, "
          f"folds_pos={best['n_folds_positive']}/9", flush=True)
    lift = best['sharpe'] - prod['sharpe']
    print(f"  Lift: {lift:+.2f}", flush=True)

    # Matched skip-placebo (random downsize at same rate)
    best_pctile = next(v["pctile"] for label, v in variants if label == best["variant"]
                        and "pctile" in v)
    best_kind = next(v["kind"] for label, v in variants if label == best["variant"])
    print(f"\n--- Matched placebo ({N_PLACEBO_SEEDS} seeds): random downsize at "
          f"{best_pctile*100:.0f}% rate ---", flush=True)
    if best_kind == "size_low_stab":
        placebo_size = 0.5
    else:
        placebo_size = 0.0
    t0 = time.time()
    placebo_sh = []
    for seed in range(N_PLACEBO_SEEDS):
        v_plac = {"kind": "matched_placebo", "placebo_seed": seed,
                   "placebo_skip_pctile": best_pctile, "size": placebo_size}
        df_p = evaluate(apd, universe, v_plac)
        placebo_sh.append(_sharpe(df_p["net_bps"].to_numpy()))
        if (seed + 1) % 25 == 0:
            print(f"  ... {seed+1}/{N_PLACEBO_SEEDS}  ({time.time()-t0:.0f}s)",
                  flush=True)
    p_sh = np.array(placebo_sh)
    p95 = float(np.percentile(p_sh, 95))
    rank = float((p_sh < best['sharpe']).mean() * 100)
    print(f"\n  Placebo Sharpe: mean={p_sh.mean():+.2f}, p50={np.percentile(p_sh,50):+.2f}, "
          f"p95={p95:+.2f}, max={p_sh.max():+.2f}", flush=True)
    print(f"  Best variant ranks p{rank:.0f}  beats_p95={'PASS' if best['sharpe'] > p95 else 'FAIL'}",
          flush=True)
    pd.DataFrame({"seed": range(N_PLACEBO_SEEDS), "sharpe": p_sh}).to_csv(
        OUT / "matched_placebo.csv", index=False)

    # Final verdict
    print(f"\n=== Test 3 verdict ===", flush=True)
    pass_prod = best['sharpe'] > prod['sharpe']
    pass_folds = best['n_folds_positive'] >= 6
    pass_placebo = best['sharpe'] > p95
    print(f"  Beats production:   {'PASS' if pass_prod else 'FAIL'} ({lift:+.2f})", flush=True)
    print(f"  ≥6/9 folds:         {'PASS' if pass_folds else 'FAIL'} ({best['n_folds_positive']}/9)",
          flush=True)
    print(f"  Beats placebo p95:  {'PASS' if pass_placebo else 'FAIL'}", flush=True)
    if pass_prod and pass_folds and pass_placebo:
        print(f"  → ADOPT", flush=True)
    else:
        print(f"  → NOT ADOPTED", flush=True)


if __name__ == "__main__":
    main()
