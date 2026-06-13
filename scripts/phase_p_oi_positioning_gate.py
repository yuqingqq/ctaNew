"""Phase P: OI / positioning as a tradability gate.

Tests whether cached Binance futures metrics (OI, top-trader L/S, taker L/S)
can improve the current filter_refill stack as a cycle-level gate.

Protocol:
  - Reuse outputs/vBTC_audit_panel/all_predictions.parquet.
  - Reuse Phase M's K=3/K=4 filter_refill + conv_gate + PM simulation.
  - Build PIT metrics features for the candidate basket at decision time.
  - Train a nested Ridge meta-gate on prior folds only to predict cycle net_bps.
  - Skip the bottom 30% predicted tradability cycles.
  - Compare with matched skip-placebo at the same per-fold skip count.

This is intentionally a sidecar research script. It does not edit production
paths and it does not declare OI/positioning "closed" because local metrics
coverage is only the cached 25-symbol set.
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

from ml.research.alpha_v4_xs import block_bootstrap_ci  # noqa: E402
from ml.research.alpha_vBTC_metrics_feature_probe import (  # noqa: E402
    RAW_METRIC_FEATURES,
    _build_symbol_metrics_features,
)
from scripts.phase_m_k_sweep import (  # noqa: E402
    APD_PATH,
    COST_PER_LEG,
    GATE_LOOKBACK,
    GATE_PCTILE,
    HORIZON,
    KLINES_DIR,
    MIN_HISTORY_DAYS,
    OOS_FOLDS,
    PM_M,
    build_rolling_ic_universe,
    filter_decision,
    get_listings,
    select_refill,
)

OUT = REPO / "outputs/vBTC_oi_positioning_gate"
OUT.mkdir(parents=True, exist_ok=True)

CYCLES_PER_YEAR = (288 * 365) / HORIZON
TOP_N = 15
K_VALUES = [3, 4]
SKIP_Q = 0.30
N_PLACEBO = 100


def _sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0:
        return 0.0
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR))


def _max_dd(net: np.ndarray) -> float:
    cum = np.cumsum(np.asarray(net, dtype=float))
    return float((cum - np.maximum.accumulate(cum)).min())


def _summarize(df: pd.DataFrame, label: str) -> dict:
    net = df["net_bps"].to_numpy(float)
    sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
    return {
        "variant": label,
        "sharpe": sh,
        "ci_lo": lo,
        "ci_hi": hi,
        "max_dd": _max_dd(net),
        "total_pnl": float(np.nansum(net)),
        "mean_bps": float(np.nanmean(net)),
        "active_cycles": int(((df["n_long"] > 0) & (df["n_short"] > 0)).sum()),
        "skip_rate": float((df["n_long"].eq(0) | df["n_short"].eq(0)).mean()),
        "avg_L": float(df["n_long"].mean()),
        "avg_S": float(df["n_short"].mean()),
        "folds_positive": int(sum(_sharpe(g["net_bps"].to_numpy(float)) > 0 for _, g in df.groupby("fold"))),
    }


def _metric_feature_table(symbols: list[str], times: list[pd.Timestamp]) -> dict[tuple[pd.Timestamp, str], dict]:
    """Build PIT metric features for cached symbols at sampled decision times."""
    out = {}
    loaded = 0
    for sym in symbols:
        g = pd.DataFrame({"open_time": times})
        feats = _build_symbol_metrics_features(sym, g)
        if feats.notna().any().any():
            loaded += 1
        for t, row in zip(times, feats.to_dict("records")):
            vals = {k: float(v) for k, v in row.items() if pd.notna(v)}
            if vals:
                out[(t, sym)] = vals
    print(f"  metrics coverage: {loaded}/{len(symbols)} symbols with cached metrics", flush=True)
    return out


def _basket_metric_features(t, longs, shorts, metric_lookup) -> dict:
    feats: dict[str, float] = {}

    def vals_for(syms, col):
        vals = [metric_lookup.get((t, s), {}).get(col, np.nan) for s in syms]
        arr = np.asarray(vals, dtype=float)
        return arr[~np.isnan(arr)]

    n_l = max(len(longs), 1)
    n_s = max(len(shorts), 1)
    cov_l = np.mean([bool(metric_lookup.get((t, s))) for s in longs]) if longs else 0.0
    cov_s = np.mean([bool(metric_lookup.get((t, s))) for s in shorts]) if shorts else 0.0
    feats["metrics_cov_l"] = float(cov_l)
    feats["metrics_cov_s"] = float(cov_s)
    feats["metrics_cov_all"] = float((cov_l * n_l + cov_s * n_s) / (n_l + n_s))

    for col in RAW_METRIC_FEATURES:
        lv = vals_for(longs, col)
        sv = vals_for(shorts, col)
        allv = np.concatenate([lv, sv]) if len(lv) or len(sv) else np.array([])
        feats[f"{col}_l_mean"] = float(lv.mean()) if len(lv) else np.nan
        feats[f"{col}_s_mean"] = float(sv.mean()) if len(sv) else np.nan
        feats[f"{col}_ls_spread"] = (
            float(lv.mean() - sv.mean()) if len(lv) and len(sv) else np.nan
        )
        feats[f"{col}_abs_mean"] = float(np.abs(allv).mean()) if len(allv) else np.nan
        feats[f"{col}_std"] = float(allv.std()) if len(allv) > 1 else np.nan
    return feats


def evaluate(apd: pd.DataFrame, universe: dict, k: int, metric_lookup: dict,
             gate_skip_times: set[pd.Timestamp] | None = None,
             collect_features: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = apd.sort_values(["open_time", "symbol"]).copy()
    df = df[df["fold"].isin(OOS_FOLDS)]
    times = sorted(df["open_time"].unique())
    keep_t = set(times[::HORIZON])
    df = df[df["open_time"].isin(keep_t)]
    times = sorted(df["open_time"].unique())
    fold_lookup = df.groupby("open_time")["fold"].first().to_dict()
    by_t = {t: g for t, g in df.groupby("open_time")}

    hist_disp = deque(maxlen=GATE_LOOKBACK)
    hist_basket = []
    cur_long, cur_short = set(), set()
    is_flat = False
    picks_hist = defaultdict(list)
    rows = []
    feat_rows = []

    for t in times:
        g = by_t.get(t)
        if g is None:
            continue
        fold = int(fold_lookup.get(t, 0))
        u = universe.get(t, set())
        g_u = g[g["symbol"].isin(u)] if u else g.iloc[0:0]
        if len(g_u) < 2 * k + 1:
            rows.append({"time": t, "fold": fold, "net_bps": 0.0, "n_long": 0, "n_short": 0})
            continue

        sym_arr = g_u["symbol"].to_numpy()
        pred_arr = g_u["pred"].to_numpy()
        ret_l = dict(zip(sym_arr, g_u["return_pct"].to_numpy()))
        exit_l = dict(zip(sym_arr, g_u["exit_time"].to_numpy()))

        idx_t = np.argpartition(-pred_arr, k - 1)[:k]
        idx_b = np.argpartition(pred_arr, k - 1)[:k]
        pred_disp = float(pred_arr[idx_t].mean() - pred_arr[idx_b].mean())
        skip = False
        if len(hist_disp) >= 30:
            thr = float(np.quantile(list(hist_disp), GATE_PCTILE))
            if pred_disp < thr:
                skip = True
        hist_disp.append(pred_disp)

        if skip:
            if not is_flat and (cur_long or cur_short):
                rows.append({"time": t, "fold": fold, "net_bps": -2 * COST_PER_LEG,
                             "n_long": 0, "n_short": 0})
                is_flat = True
                cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold, "net_bps": 0.0, "n_long": 0, "n_short": 0})
            continue

        order_d = np.argsort(-pred_arr)
        order_a = np.argsort(pred_arr)
        long_r = [sym_arr[i] for i in order_d]
        short_r = [sym_arr[i] for i in order_a]
        cand_l, n_el = select_refill(long_r, "long", k, picks_hist, 90, t)
        cand_s, n_es = select_refill(short_r, "short", k, picks_hist, 90, t)
        c_ls, c_ss = set(cand_l), set(cand_s)

        hist_basket.append({"long": c_ls, "short": c_ss})
        if len(hist_basket) > PM_M:
            hist_basket = hist_basket[-PM_M:]
        if len(hist_basket) >= PM_M:
            p_l = [h["long"] for h in hist_basket[-PM_M:][:PM_M - 1]]
            p_s = [h["short"] for h in hist_basket[-PM_M:][:PM_M - 1]]
            nl = cur_long & c_ls
            ns = cur_short & c_ss
            for s_ in c_ls - cur_long:
                if all(s_ in p for p in p_l):
                    nl.add(s_)
            for s_ in c_ss - cur_short:
                if all(s_ in p for p in p_s):
                    ns.add(s_)
            if len(nl) > k:
                nl = set(sorted(nl, key=lambda s_: -pred_arr[np.where(sym_arr == s_)[0][0]])[:k])
            if len(ns) > k:
                ns = set(sorted(ns, key=lambda s_: pred_arr[np.where(sym_arr == s_)[0][0]])[:k])
        else:
            nl, ns = c_ls, c_ss

        if not nl or not ns:
            if not is_flat and (cur_long or cur_short):
                rows.append({"time": t, "fold": fold, "net_bps": -2 * COST_PER_LEG,
                             "n_long": 0, "n_short": 0})
                is_flat = True
                cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold, "net_bps": 0.0, "n_long": 0, "n_short": 0})
            continue

        if collect_features:
            fr = {
                "time": t,
                "fold": fold,
                "pred_disp": pred_disp,
                "n_long": len(nl),
                "n_short": len(ns),
                "n_excl_long": n_el,
                "n_excl_short": n_es,
            }
            fr.update(_basket_metric_features(t, sorted(nl), sorted(ns), metric_lookup))
            feat_rows.append(fr)

        extra_skip = gate_skip_times is not None and t in gate_skip_times
        if extra_skip:
            if not is_flat and (cur_long or cur_short):
                rows.append({"time": t, "fold": fold, "net_bps": -2 * COST_PER_LEG,
                             "n_long": 0, "n_short": 0})
                is_flat = True
                cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold, "net_bps": 0.0, "n_long": 0, "n_short": 0})
            continue

        lr = [ret_l[s_] for s_ in nl]
        sr = [ret_l[s_] for s_ in ns]
        spread = (np.mean(lr) - np.mean(sr)) * 1e4
        if is_flat:
            cost = 2 * COST_PER_LEG
            is_flat = False
        else:
            cl = len(nl.symmetric_difference(cur_long)) / max(len(nl | cur_long), 1)
            cs = len(ns.symmetric_difference(cur_short)) / max(len(ns | cur_short), 1)
            cost = (cl + cs) * COST_PER_LEG
        net = spread - cost
        for s_ in nl:
            picks_hist[(s_, "long")].append((t, exit_l[s_], ret_l[s_] * 1e4 / len(nl)))
        for s_ in ns:
            picks_hist[(s_, "short")].append((t, exit_l[s_], -ret_l[s_] * 1e4 / len(ns)))
        rows.append({"time": t, "fold": fold, "net_bps": net, "n_long": len(nl), "n_short": len(ns)})
        cur_long, cur_short = nl, ns

    return pd.DataFrame(rows), pd.DataFrame(feat_rows)


def _ridge_scores(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    xtr = train[feature_cols].to_numpy(float)
    xte = test[feature_cols].to_numpy(float)
    med = np.nanmedian(xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    xtr = np.where(np.isnan(xtr), med, xtr)
    xte = np.where(np.isnan(xte), med, xte)
    mu = xtr.mean(axis=0)
    sd = xtr.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    xtr = (xtr - mu) / sd
    xte = (xte - mu) / sd
    xtr = np.column_stack([np.ones(len(xtr)), xtr])
    xte = np.column_stack([np.ones(len(xte)), xte])
    y = train["net_bps"].to_numpy(float)
    lam = 10.0
    eye = np.eye(xtr.shape[1])
    eye[0, 0] = 0.0
    beta = np.linalg.solve(xtr.T @ xtr + lam * eye, xtr.T @ y)
    return xtr @ beta, xte @ beta


def nested_gate_times(features: pd.DataFrame, feature_cols: list[str], q: float) -> tuple[set, pd.DataFrame]:
    decisions = []
    skip_times = set()
    for f in OOS_FOLDS:
        train = features[(features["fold"] < f) & (features["fold"] >= 1)].copy()
        test = features[features["fold"] == f].copy()
        if len(train) < 100 or len(test) == 0:
            continue
        train_scores, test_scores = _ridge_scores(train, test, feature_cols)
        thr = float(np.quantile(train_scores, q))
        test = test.copy()
        test["score"] = test_scores
        test["threshold"] = thr
        test["gate_skip"] = test["score"] < thr
        skip_times.update(test.loc[test["gate_skip"], "time"].tolist())
        decisions.append(test[["time", "fold", "score", "threshold", "gate_skip", "net_bps"]])
    return skip_times, pd.concat(decisions, ignore_index=True) if decisions else pd.DataFrame()


def matched_placebo_times(features: pd.DataFrame, real_decisions: pd.DataFrame, seed: int) -> set:
    rng = np.random.RandomState(seed)
    out = set()
    counts = real_decisions.groupby("fold")["gate_skip"].sum().to_dict()
    for f, n in counts.items():
        pool = features[features["fold"] == f]["time"].tolist()
        n = int(min(n, len(pool)))
        if n > 0:
            out.update(rng.choice(pool, size=n, replace=False).tolist())
    return out


def main():
    print("=== Phase P: OI / positioning tradability gate ===\n", flush=True)
    apd = pd.read_parquet(APD_PATH)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    all_times = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_times = all_times[::HORIZON]

    listings = get_listings()
    panel_syms = sorted(apd["symbol"].unique())

    def eligibility_at(t):
        ts = pd.Timestamp(t, unit="ms", tz="UTC") if isinstance(t, (int, np.integer)) else pd.Timestamp(t)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    metric_lookup = _metric_feature_table(panel_syms, sampled_times)
    results = []
    details = {"coverage_symbols": len({s for _, s in metric_lookup.keys()}), "k": {}}

    for k in K_VALUES:
        print(f"\n--- K={k} ---", flush=True)
        universe = build_rolling_ic_universe(apd, sampled_times, TOP_N, eligibility_at)
        base_df, feat_df = evaluate(apd, universe, k, metric_lookup, collect_features=True)
        feat_df = feat_df.merge(base_df[["time", "net_bps"]], on="time", how="left")
        base_df.to_csv(OUT / f"per_cycle_K{k}_baseline.csv", index=False)
        feat_df.to_csv(OUT / f"meta_features_K{k}.csv", index=False)

        base_sum = _summarize(base_df, f"K{k}_baseline")
        results.append(base_sum)
        print(f"  baseline Sharpe {base_sum['sharpe']:+.2f} PnL {base_sum['total_pnl']:+.0f}", flush=True)

        metric_cols = [c for c in feat_df.columns if c.startswith("metrics_")]
        metric_cols = [c for c in metric_cols if feat_df[c].notna().any()]
        state_cols = ["pred_disp", "n_long", "n_short", "n_excl_long", "n_excl_short"]

        for label, cols in [
            ("metrics_only", metric_cols),
            ("metrics_plus_state", metric_cols + state_cols),
        ]:
            skip_times, dec = nested_gate_times(feat_df, cols, SKIP_Q)
            dec.to_csv(OUT / f"gate_decisions_K{k}_{label}.csv", index=False)
            gated_df, _ = evaluate(apd, universe, k, metric_lookup, gate_skip_times=skip_times)
            gated_df.to_csv(OUT / f"per_cycle_K{k}_{label}.csv", index=False)
            summ = _summarize(gated_df, f"K{k}_{label}")
            summ["extra_skip_rate_active"] = float(dec["gate_skip"].mean()) if len(dec) else np.nan
            results.append(summ)
            print(f"  {label}: Sharpe {summ['sharpe']:+.2f} "
                  f"Δ {summ['sharpe'] - base_sum['sharpe']:+.2f} "
                  f"skip_active {summ['extra_skip_rate_active']:.1%}", flush=True)

            placebo_sh = []
            for seed in range(N_PLACEBO):
                p_times = matched_placebo_times(feat_df, dec, seed)
                p_df, _ = evaluate(apd, universe, k, metric_lookup, gate_skip_times=p_times)
                placebo_sh.append(_sharpe(p_df["net_bps"].to_numpy(float)))
            p = np.asarray(placebo_sh)
            details["k"][f"{k}_{label}"] = {
                "placebo_mean": float(p.mean()),
                "placebo_p50": float(np.percentile(p, 50)),
                "placebo_p95": float(np.percentile(p, 95)),
                "placebo_max": float(p.max()),
                "rank_vs_placebo": float((p < summ["sharpe"]).mean() * 100),
                "beats_p95": bool(summ["sharpe"] > np.percentile(p, 95)),
            }
            print(f"    placebo mean {p.mean():+.2f} p95 {np.percentile(p,95):+.2f} "
                  f"rank p{(p < summ['sharpe']).mean()*100:.0f}", flush=True)

    pd.DataFrame(results).to_csv(OUT / "results.csv", index=False)
    with open(OUT / "summary.json", "w") as f:
        json.dump(details, f, indent=2)
    print(f"\n  saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
