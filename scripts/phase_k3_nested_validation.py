"""Phase K3: nested-fold validation of cost-aware swap rule.

Replaces the unstable bps→pred-unit calibration with FIXED pred-unit margins.
Selects margin via nested folds: best margin from folds < f applied to fold f.
Adds K_min activity floor to prevent sparse-lottery (avg basket < 1).

Margins tested (fixed pred-unit thresholds, no slope calibration):
  {0.00, 0.15, 0.25, 0.40, 0.60, 0.80, 1.00}

Activity floor:
  K_min ∈ {1, 2} — if cost-margin rule produces basket < K_min names, fall back
  to top-K_min by pred (no cost gate). Prevents sparseness collapse.

Procedure:
  1. Run protocol for each (margin, K_min) combination → per-cycle CSVs.
  2. Nested-fold selection: for each fold f, pick margin/K_min with best cumulative
     Sharpe on folds < f. Use default (margin=0.40, K_min=2) for folds 0-2.
     Stitch fold f's per-cycle from the selected variant.
  3. Compute nested-OOS aggregate Sharpe + per-fold breakdown.
  4. Matched-basket-size placebo (100 seeds) on the nested-OOS result.
  5. Adopt if: ≥6/9 folds positive AND nested-Sharpe > placebo p95.

Output: outputs/vBTC_swap_rule/k3_nested/
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

OUT = REPO / "outputs/vBTC_swap_rule/k3_nested"
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
N_PLACEBO_SEEDS = 100
MIN_HISTORY_DAYS = 60
TOP_N = 15

MARGINS_PRED_UNIT = [0.00, 0.15, 0.25, 0.40, 0.60, 0.80, 1.00]
KMIN_VALUES = [1, 2]


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


def build_basket(sym_arr, pred_arr, side, cur_basket, picks_hist, t,
                  K_target, cost_margin_pred, K_min):
    """Build basket with fixed pred-unit cost margin + K_min activity floor."""
    if side == "long":
        order = np.argsort(-pred_arr)
    else:
        order = np.argsort(pred_arr)
    ranked = [sym_arr[i] for i in order]
    ranked_passing, n_excl = select_refill(ranked, side, K_target, picks_hist, 90, t)
    sym_to_pred = dict(zip(sym_arr, pred_arr))

    if not cur_basket:
        return set(ranked_passing[:K_target]), n_excl

    extended_top = set(ranked_passing)
    kept = cur_basket & extended_top
    if len(kept) >= K_target:
        kept_sorted = sorted(kept, key=lambda s: -sym_to_pred[s] if side == "long" else sym_to_pred[s])
        return set(kept_sorted[:K_target]), n_excl

    candidates_to_add = [s for s in ranked_passing if s not in kept]
    if cost_margin_pred > 0 and kept:
        if side == "long":
            weakest = min(sym_to_pred[s] for s in kept)
        else:
            weakest = max(sym_to_pred[s] for s in kept)
        filtered = []
        for s in candidates_to_add:
            p = sym_to_pred[s]
            lift = (p - weakest) if side == "long" else (weakest - p)
            if lift > cost_margin_pred:
                filtered.append(s)
        candidates_to_add = filtered

    n_to_add = K_target - len(kept)
    proposed = kept | set(candidates_to_add[:n_to_add])

    # K_min activity floor: if proposed basket would have < K_min names,
    # fall back to top-K_min by pred (no cost gate)
    if len(proposed) < K_min and len(ranked_passing) >= K_min:
        return set(ranked_passing[:K_min]), n_excl

    return proposed, n_excl


def evaluate(apd, universe, cost_margin_pred, K_min):
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
    rows = []
    for t in times:
        g = by_t.get(t)
        if g is None: continue
        u = universe.get(t, set())
        g_u = g[g["symbol"].isin(u)] if u else g.iloc[0:0]
        if len(g_u) < 2 * K + 1:
            rows.append({"time": t, "fold": fold_lookup.get(t, 0), "spread_bps": 0,
                          "cost_bps": 0, "net_bps": 0, "n_long": 0, "n_short": 0})
            continue
        sym_arr = g_u["symbol"].to_numpy()
        pred_arr = g_u["pred"].to_numpy()
        ret_l = dict(zip(sym_arr, g_u["return_pct"].to_numpy()))
        exit_l = dict(zip(sym_arr, g_u["exit_time"].to_numpy()))
        idx_t = np.argpartition(-pred_arr, K - 1)[:K]
        idx_b = np.argpartition(pred_arr, K - 1)[:K]
        disp = float(pred_arr[idx_t].mean() - pred_arr[idx_b].mean())
        skip = False
        if len(hist_disp) >= 30:
            thr = float(np.quantile(list(hist_disp), GATE_PCTILE))
            if disp < thr: skip = True
        hist_disp.append(disp)
        if skip:
            if not is_flat and (cur_long or cur_short):
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "spread_bps": 0,
                              "cost_bps": 2 * COST_PER_LEG, "net_bps": -2 * COST_PER_LEG,
                              "n_long": 0, "n_short": 0})
                is_flat = True; cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "spread_bps": 0,
                              "cost_bps": 0, "net_bps": 0, "n_long": 0, "n_short": 0})
            continue
        nl, n_el = build_basket(sym_arr, pred_arr, "long", cur_long, picks_hist, t,
                                   K, cost_margin_pred, K_min)
        ns, n_es = build_basket(sym_arr, pred_arr, "short", cur_short, picks_hist, t,
                                   K, cost_margin_pred, K_min)
        c_ls = nl; c_ss = ns
        hist_basket.append({"long": c_ls, "short": c_ss})
        if len(hist_basket) > PM_M:
            hist_basket = hist_basket[-PM_M:]
        if len(hist_basket) >= PM_M:
            p_l = [h["long"] for h in hist_basket[-PM_M:][:PM_M - 1]]
            p_s = [h["short"] for h in hist_basket[-PM_M:][:PM_M - 1]]
            nl_pm = cur_long & c_ls
            ns_pm = cur_short & c_ss
            for s_ in c_ls - cur_long:
                if all(s_ in p for p in p_l): nl_pm.add(s_)
            for s_ in c_ss - cur_short:
                if all(s_ in p for p in p_s): ns_pm.add(s_)
            if len(nl_pm) > K:
                nl_pm = set(sorted(nl_pm, key=lambda s_: -pred_arr[np.where(sym_arr == s_)[0][0]])[:K])
            if len(ns_pm) > K:
                ns_pm = set(sorted(ns_pm, key=lambda s_: pred_arr[np.where(sym_arr == s_)[0][0]])[:K])
            nl, ns = nl_pm, ns_pm
        if not nl or not ns:
            if not is_flat and (cur_long or cur_short):
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "spread_bps": 0,
                              "cost_bps": 2 * COST_PER_LEG, "net_bps": -2 * COST_PER_LEG,
                              "n_long": 0, "n_short": 0})
                is_flat = True; cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "spread_bps": 0,
                              "cost_bps": 0, "net_bps": 0, "n_long": 0, "n_short": 0})
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
        rows.append({"time": t, "fold": fold_lookup.get(t, 0), "spread_bps": spread,
                      "cost_bps": cost, "net_bps": net,
                      "n_long": len(nl), "n_short": len(ns)})
        cur_long, cur_short = nl, ns
    return pd.DataFrame(rows)


def evaluate_matched_placebo(apd, universe, target_n_l, target_n_s, seed):
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
        if len(g_u) >= 2 * K + 1:
            pred_arr = g_u["pred"].to_numpy()
            idx_t = np.argpartition(-pred_arr, K - 1)[:K]
            idx_b = np.argpartition(pred_arr, K - 1)[:K]
            disp = float(pred_arr[idx_t].mean() - pred_arr[idx_b].mean())
            skip = False
            if len(hist_disp) >= 30:
                thr = float(np.quantile(list(hist_disp), GATE_PCTILE))
                if disp < thr: skip = True
            hist_disp.append(disp)
        else:
            skip = True
        if skip:
            if not is_flat and (cur_long or cur_short):
                rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "net_bps": -2 * COST_PER_LEG, "n_long": 0, "n_short": 0})
                is_flat = True; cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "net_bps": 0, "n_long": 0, "n_short": 0})
            cycle_id += 1
            continue
        tl = target_n_l[cycle_id] if cycle_id < len(target_n_l) else 0
        ts = target_n_s[cycle_id] if cycle_id < len(target_n_s) else 0
        if tl == 0 or ts == 0 or len(sym_arr) < tl + ts:
            if not is_flat and (cur_long or cur_short):
                rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "net_bps": -2 * COST_PER_LEG, "n_long": 0, "n_short": 0})
                is_flat = True; cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "net_bps": 0, "n_long": 0, "n_short": 0})
            cycle_id += 1
            continue
        shuffled = rng.permutation(len(sym_arr))
        nl = set(sym_arr[shuffled[:tl]])
        ns = set(sym_arr[shuffled[tl:tl + ts]])
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
        rows.append({"time": t, "fold": fold_lookup.get(t, 0), "net_bps": net,
                      "n_long": len(nl), "n_short": len(ns)})
        cur_long, cur_short = nl, ns
        cycle_id += 1
    return pd.DataFrame(rows)


def main():
    print("=== Phase K3: nested-fold pred-unit margin sweep ===\n", flush=True)
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

    # (1) Run all (margin, K_min) variants
    print(f"\n--- (1) (margin, K_min) sweep ---", flush=True)
    print(f"  {'margin':>7}  {'K_min':>5}  {'Sharpe':>7}  {'totPnL':>8}  "
          f"{'maxDD':>7}  {'avg_L/S':>8}  per-fold Sharpe", flush=True)
    variants_data = {}  # (margin, K_min) -> DataFrame
    for margin in MARGINS_PRED_UNIT:
        for K_min in KMIN_VALUES:
            df_v = evaluate(apd, universe, margin, K_min)
            variants_data[(margin, K_min)] = df_v
            net = df_v["net_bps"].to_numpy()
            sh = _sharpe(net)
            per_fold_sh = []
            for f in OOS_FOLDS:
                d = df_v[df_v["fold"] == f]["net_bps"].to_numpy()
                if len(d) >= 3:
                    per_fold_sh.append(_sharpe(d))
                else:
                    per_fold_sh.append(np.nan)
            pf_str = " ".join(f"{x:+.1f}" for x in per_fold_sh)
            print(f"  {margin:>7.2f}  {K_min:>5d}  {sh:>+7.2f}  {net.sum():>+8.0f}  "
                  f"{_max_dd(net):>+7.0f}  {df_v['n_long'].mean():.2f}/{df_v['n_short'].mean():.2f}  "
                  f"{pf_str}", flush=True)

    # Save summary
    summary = []
    for (margin, K_min), df_v in variants_data.items():
        net = df_v["net_bps"].to_numpy()
        per_fold = {f: _sharpe(df_v[df_v["fold"] == f]["net_bps"].to_numpy())
                    for f in OOS_FOLDS if len(df_v[df_v["fold"] == f]) >= 3}
        n_positive = sum(1 for v in per_fold.values() if v > 0)
        summary.append({
            "margin": margin, "K_min": K_min,
            "sharpe": _sharpe(net), "total_pnl": net.sum(),
            "max_dd": _max_dd(net),
            "avg_L": float(df_v["n_long"].mean()),
            "avg_S": float(df_v["n_short"].mean()),
            "n_folds_positive": n_positive,
            **{f"sh_f{f}": v for f, v in per_fold.items()},
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT / "sweep_summary.csv", index=False)

    # (2) Nested-fold selection: for each fold f, pick best (margin, K_min) from folds < f
    print(f"\n--- (2) Nested-fold selection ---", flush=True)
    default = (0.40, 2)  # fallback for early folds
    nested_per_cycle = []
    selected_per_fold = {}
    for f in OOS_FOLDS:
        # Past folds available: 1..f-1
        past_folds = [pf for pf in OOS_FOLDS if pf < f]
        if len(past_folds) < 2:
            # Default for early folds (we don't have enough history)
            chosen = default
        else:
            # Pick variant with best cumulative Sharpe on past folds
            scores = {}
            for (margin, K_min), df_v in variants_data.items():
                d = df_v[df_v["fold"].isin(past_folds)]["net_bps"].to_numpy()
                if len(d) < 10: continue
                scores[(margin, K_min)] = _sharpe(d)
            if not scores:
                chosen = default
            else:
                chosen = max(scores, key=scores.get)
        selected_per_fold[f] = chosen
        # Append fold f's per-cycle from chosen variant
        df_chosen = variants_data[chosen]
        nested_per_cycle.append(df_chosen[df_chosen["fold"] == f].copy())
        print(f"  fold {f}: chosen (margin={chosen[0]:.2f}, K_min={chosen[1]})", flush=True)
    nested_df = pd.concat(nested_per_cycle, ignore_index=True)
    nested_df.to_csv(OUT / "nested_per_cycle.csv", index=False)

    # (3) Compute nested-OOS aggregate
    nested_net = nested_df["net_bps"].to_numpy()
    sh_nested = _sharpe(nested_net)
    sh_nested_ci_lo, sh_nested_ci_hi = block_bootstrap_ci(
        nested_net, statistic=_sharpe, block_size=7, n_boot=2000)[1:]
    print(f"\n--- (3) Nested-OOS aggregate ---", flush=True)
    print(f"  Nested Sharpe: {sh_nested:+.2f} [{sh_nested_ci_lo:+.2f}, {sh_nested_ci_hi:+.2f}]",
          flush=True)
    print(f"  totPnL: {nested_net.sum():+.0f}", flush=True)
    print(f"  maxDD:  {_max_dd(nested_net):+.0f}", flush=True)
    print(f"  avg L/S: {nested_df['n_long'].mean():.2f}/{nested_df['n_short'].mean():.2f}",
          flush=True)

    # Per-fold nested Sharpe
    print(f"\n  Per-fold nested Sharpe:", flush=True)
    n_pos_nested = 0
    for f in OOS_FOLDS:
        d = nested_df[nested_df["fold"] == f]["net_bps"].to_numpy()
        if len(d) < 3: continue
        sh_f = _sharpe(d)
        if sh_f > 0: n_pos_nested += 1
        margin, K_min = selected_per_fold[f]
        print(f"    fold {f} (chose margin={margin:.2f}, K_min={K_min}): "
              f"Sharpe={sh_f:+.2f}  pnl={d.sum():+.0f}",
              flush=True)
    print(f"\n  Folds positive: {n_pos_nested}/9", flush=True)

    # (4) Matched-basket-size placebo on nested-OOS
    print(f"\n--- (4) Matched-basket-size placebo ({N_PLACEBO_SEEDS} seeds) ---",
          flush=True)
    target_l = nested_df["n_long"].tolist()
    target_s = nested_df["n_short"].tolist()
    t0 = time.time()
    placebo_sh = []
    for seed in range(N_PLACEBO_SEEDS):
        df_p = evaluate_matched_placebo(apd, universe, target_l, target_s, seed)
        placebo_sh.append(_sharpe(df_p["net_bps"].to_numpy()))
        if (seed + 1) % 25 == 0:
            print(f"  ... {seed+1}/{N_PLACEBO_SEEDS}  ({time.time()-t0:.0f}s)",
                  flush=True)
    p_sh = np.array(placebo_sh)
    p95 = float(np.percentile(p_sh, 95))
    rank = float((p_sh < sh_nested).mean() * 100)
    print(f"\n  Placebo: mean={p_sh.mean():+.2f}, p50={np.percentile(p_sh,50):+.2f}, "
          f"p95={p95:+.2f}, max={p_sh.max():+.2f}", flush=True)
    print(f"  Nested Sharpe={sh_nested:+.2f}  rank={rank:.0f}%  "
          f"beats_p95={'PASS' if sh_nested > p95 else 'FAIL'}", flush=True)
    pd.DataFrame({"seed": range(N_PLACEBO_SEEDS), "sharpe": p_sh}).to_csv(
        OUT / "nested_matched_placebo.csv", index=False)

    # (5) Final verdict
    prod = next(s for s in summary if s["margin"] == 0.0 and s["K_min"] == 1)
    print(f"\n=== Phase K3 verdict ===", flush=True)
    print(f"  Production (margin=0, K_min=1): Sharpe={prod['sharpe']:+.2f}, "
          f"folds_pos={prod['n_folds_positive']}/9", flush=True)
    print(f"  Nested-OOS:                      Sharpe={sh_nested:+.2f}, "
          f"folds_pos={n_pos_nested}/9", flush=True)
    print(f"  Lift vs production:              {sh_nested - prod['sharpe']:+.2f}", flush=True)
    print(f"  Beats placebo p95:               {'YES' if sh_nested > p95 else 'NO'}",
          flush=True)
    if n_pos_nested >= 6 and sh_nested > p95 and sh_nested > prod['sharpe'] + 0.2:
        verdict = "ADOPT — robust nested-OOS lift beats placebo and lifts ≥6/9 folds"
    elif sh_nested > p95 and n_pos_nested >= 5:
        verdict = "PROMISING — lift modest but real; do live forward test"
    else:
        verdict = "NOT ADOPTED — nested validation fails 6/9 or p95 criterion"
    print(f"\n  {verdict}\n  saved → {OUT}", flush=True)


if __name__ == "__main__":
    main()
