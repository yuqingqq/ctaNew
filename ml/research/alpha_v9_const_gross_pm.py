"""Constant-gross weighting variant — Tier 2 deployment policy test.

Compares two weighting schemes under conv+PM gate:

A. Constant per-name (current implementation):
     per-name weight = 1/top_k = 1/7 (FIXED)
     leg gross = K_actual / 7  (varies cycle-to-cycle)
     Effect: directional tilt when K_L ≠ K_S; +2.75 Sh validated

B. Constant gross (proposed alternative):
     per-name weight = 1/K_actual
     leg gross = 1.0 (constant)
     Effect: strict market-neutral every cycle; concentration when K small
     Expected: lower Sharpe by loss of regime-tilt alpha

Output: Sharpe trade-off + market-neutrality metrics for the policy choice.
"""
from __future__ import annotations
import json, sys, time, warnings
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import (
    ENSEMBLE_SEEDS, _multi_oos_splits, _slice, _train,
)
from ml.research.alpha_v4_xs import block_bootstrap_ci
from ml.research.alpha_v8_h48_audit import build_wide_panel

HORIZON = 48
TOP_K = 7
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
GATE_LOOKBACK = 252
GATE_PCTILE = 0.30
PM_M = 2
PM_BAND = 1.0
OUT_DIR = REPO / "outputs/const_gross_pm"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _bn_scale(beta_L, beta_S):
    if beta_L < 0.1 or beta_S < 0.1 or (beta_L + beta_S) < 0.3:
        return 1.0, 1.0
    denom = beta_L + beta_S
    return (float(np.clip(2.0 * beta_S / denom, 0.5, 1.5)),
            float(np.clip(2.0 * beta_L / denom, 0.5, 1.5)))


def evaluate_with_weighting(
    test: pd.DataFrame, yt: np.ndarray, *, weighting: str,
    use_conv_gate: bool = True, use_pm_gate: bool = True,
    top_k: int = TOP_K, cost_bps_per_leg: float = COST_PER_LEG,
    sample_every: int = HORIZON, gate_pctile: float = GATE_PCTILE,
    gate_lookback: int = GATE_LOOKBACK,
    pm_m: int = PM_M, pm_band: float = PM_BAND,
) -> pd.DataFrame:
    """Stacked gate evaluator with selectable weighting policy.
    weighting='per_name': per-name = 1/top_k, leg gross varies.
    weighting='const_gross': per-name = 1/K_actual, leg gross = 1.0."""
    assert weighting in ("per_name", "const_gross")
    cols = ["open_time", "symbol", "return_pct", "alpha_realized",
            "basket_fwd", "beta_short_vs_bk"]
    df = test[cols].copy()
    df["pred"] = yt
    times = sorted(df["open_time"].unique())
    if not times:
        return pd.DataFrame()
    if sample_every > 1:
        keep_times = set(times[::sample_every])
        df = df[df["open_time"].isin(keep_times)]

    band_k = max(top_k, int(round(pm_band * top_k)))
    history: list[dict] = []
    dispersion_history: deque = deque(maxlen=gate_lookback)

    bars = []
    cur_long, cur_short = set(), set()
    prev_long_w, prev_short_w = {}, {}

    for t, g in df.groupby("open_time"):
        n = len(g)
        if n < 2 * top_k + 1:
            continue
        sym_arr = g["symbol"].to_numpy()
        pred_arr = g["pred"].to_numpy()
        idx_top_k = np.argpartition(-pred_arr, top_k - 1)[:top_k]
        idx_bot_k = np.argpartition(pred_arr, top_k - 1)[:top_k]

        # Conv gate
        dispersion = float(pred_arr[idx_top_k].mean() - pred_arr[idx_bot_k].mean())
        skip = False
        if use_conv_gate and len(dispersion_history) >= 30:
            thr = float(np.quantile(list(dispersion_history), gate_pctile))
            if dispersion < thr:
                skip = True
        dispersion_history.append(dispersion)

        # Update PM history
        bk = min(band_k, n)
        idx_top_band = np.argpartition(-pred_arr, bk - 1)[:bk] if bk < n else np.arange(n)
        idx_bot_band = np.argpartition(pred_arr, bk - 1)[:bk] if bk < n else np.arange(n)
        history.append({
            "long": set(sym_arr[idx_top_band]),
            "short": set(sym_arr[idx_bot_band]),
        })
        if len(history) > pm_m:
            history = history[-pm_m:]

        if skip:
            bars.append({
                "time": t, "spread_ret_bps": 0.0,
                "long_turnover": 0.0, "short_turnover": 0.0,
                "cost_bps": 0.0, "net_bps": 0.0,
                "n_long": 0, "n_short": 0, "skipped": 1,
                "gross_L": 0.0, "gross_S": 0.0,
            })
            cur_long, cur_short = set(), set()
            prev_long_w = {}; prev_short_w = {}
            continue

        cand_long = set(sym_arr[idx_top_k])
        cand_short = set(sym_arr[idx_bot_k])

        if use_pm_gate:
            new_long = cur_long & cand_long
            new_short = cur_short & cand_short
            if len(history) >= pm_m:
                past_long = [h["long"] for h in history[-pm_m:][:pm_m - 1]]
                past_short = [h["short"] for h in history[-pm_m:][:pm_m - 1]]
                for s in cand_long - cur_long:
                    if all(s in p for p in past_long):
                        new_long.add(s)
                for s in cand_short - cur_short:
                    if all(s in p for p in past_short):
                        new_short.add(s)
            else:
                new_long |= cand_long
                new_short |= cand_short
            if len(new_long) > top_k:
                ranked = sorted(new_long, key=lambda s: -pred_arr[sym_arr == s][0])[:top_k]
                new_long = set(ranked)
            if len(new_short) > top_k:
                ranked = sorted(new_short, key=lambda s: pred_arr[sym_arr == s][0])[:top_k]
                new_short = set(ranked)
        else:
            new_long = cand_long
            new_short = cand_short

        if not new_long or not new_short:
            cur_long, cur_short = new_long, new_short
            prev_long_w = {s: 1.0 / max(len(new_long), 1) for s in new_long} if weighting == "const_gross" else {s: 1.0 / top_k for s in new_long}
            prev_short_w = {s: 1.0 / max(len(new_short), 1) for s in new_short} if weighting == "const_gross" else {s: 1.0 / top_k for s in new_short}
            bars.append({
                "time": t, "spread_ret_bps": 0.0,
                "long_turnover": 0.0, "short_turnover": 0.0,
                "cost_bps": 0.0, "net_bps": 0.0,
                "n_long": len(new_long), "n_short": len(new_short),
                "skipped": 0, "gross_L": 0.0, "gross_S": 0.0,
            })
            continue

        long_g = g[g["symbol"].isin(new_long)]
        short_g = g[g["symbol"].isin(new_short)]
        scale_L, scale_S = _bn_scale(long_g["beta_short_vs_bk"].mean(),
                                      short_g["beta_short_vs_bk"].mean())

        n_L, n_S = len(new_long), len(new_short)
        if weighting == "per_name":
            per_name_L = 1.0 / top_k
            per_name_S = 1.0 / top_k
            gross_L_unscaled = n_L / top_k
            gross_S_unscaled = n_S / top_k
        else:  # const_gross
            per_name_L = 1.0 / n_L
            per_name_S = 1.0 / n_S
            gross_L_unscaled = 1.0
            gross_S_unscaled = 1.0

        long_w = {s: scale_L * per_name_L for s in new_long}
        short_w = {s: scale_S * per_name_S for s in new_short}
        gross_L = scale_L * gross_L_unscaled
        gross_S = scale_S * gross_S_unscaled

        long_ret = gross_L * long_g["return_pct"].mean()
        short_ret = gross_S * short_g["return_pct"].mean()
        spread_ret = long_ret - short_ret

        if not prev_long_w:
            long_to = sum(long_w.values())
            short_to = sum(short_w.values())
        else:
            all_l = set(long_w) | set(prev_long_w)
            long_to = sum(abs(long_w.get(s, 0) - prev_long_w.get(s, 0)) for s in all_l)
            all_s = set(short_w) | set(prev_short_w)
            short_to = sum(abs(short_w.get(s, 0) - prev_short_w.get(s, 0)) for s in all_s)
        cost_bps = cost_bps_per_leg * (long_to + short_to)

        bars.append({
            "time": t, "spread_ret_bps": spread_ret * 1e4,
            "long_turnover": long_to, "short_turnover": short_to,
            "cost_bps": cost_bps, "net_bps": spread_ret * 1e4 - cost_bps,
            "n_long": n_L, "n_short": n_S,
            "skipped": 0, "gross_L": gross_L, "gross_S": gross_S,
        })
        cur_long, cur_short = new_long, new_short
        prev_long_w, prev_short_w = long_w, short_w

    return pd.DataFrame(bars)


def _sharpe(x: np.ndarray) -> float:
    if len(x) == 0 or x.std() == 0:
        return 0.0
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR))


def main():
    panel = build_wide_panel()
    folds = _multi_oos_splits(panel)
    print(f"Multi-OOS folds: {len(folds)}")

    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    avail_feats = [c for c in v6_clean if c in panel.columns]

    cycles = {"per_name": [], "const_gross": []}
    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200: continue
        Xt = tr[avail_feats].to_numpy(dtype=np.float32)
        yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)
        Xc = ca[avail_feats].to_numpy(dtype=np.float32)
        yc_ = ca["demeaned_target"].to_numpy(dtype=np.float32)
        models = [_train(Xt, yt_, Xc, yc_, seed=s) for s in ENSEMBLE_SEEDS]
        Xtest = test[avail_feats].to_numpy(dtype=np.float32)
        pred_test = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                              for m in models], axis=0)

        line = f"  fold {fold['fid']:>2}: "
        for w_mode in ["per_name", "const_gross"]:
            df_eval = evaluate_with_weighting(test, pred_test,
                                               weighting=w_mode,
                                               use_conv_gate=True, use_pm_gate=True)
            for _, row in df_eval.iterrows():
                cycles[w_mode].append({
                    "fold": fold["fid"], "time": row["time"],
                    "net": row["net_bps"], "spread": row["spread_ret_bps"],
                    "cost": row["cost_bps"], "skipped": row["skipped"],
                    "n_long": row["n_long"], "n_short": row["n_short"],
                    "gross_L": row["gross_L"], "gross_S": row["gross_S"],
                })
            net_arr = df_eval["net_bps"].to_numpy()
            line += f"{w_mode[:6]}={net_arr.mean():+.2f}({_sharpe(net_arr):+.1f})  "
        print(line + f"({time.time()-t0:.0f}s)")

    print("\n" + "=" * 110)
    print(f"WEIGHTING POLICY COMPARISON  (conv+PM active, h={HORIZON} K={TOP_K} 4.5 bps/leg β-neutral)")
    print("=" * 110)

    rows = []
    nets = {}
    for name in ["per_name", "const_gross"]:
        df_v = pd.DataFrame(cycles[name])
        if df_v.empty: continue
        net = df_v["net"].to_numpy()
        nets[name] = net
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        # Market-neutrality metric: |gross_L - gross_S| on active cycles
        active = df_v[df_v["skipped"] == 0]
        gross_imbalance = (active["gross_L"] - active["gross_S"]).abs().mean()
        max_gross = (active[["gross_L", "gross_S"]].max(axis=1)).max()
        max_per_name = active.apply(lambda r: r["gross_L"]/max(r["n_long"],1) if r["n_long"]>0 else 0, axis=1).max()
        rows.append({
            "weighting": name,
            "n": len(net),
            "net_bps": net.mean(),
            "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
            "gross_L_avg": active["gross_L"].mean(),
            "gross_S_avg": active["gross_S"].mean(),
            "imbalance_abs": gross_imbalance,
            "max_gross": max_gross,
            "max_per_name_w": max_per_name,
        })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False, float_format="%+.3f"))

    if "per_name" in nets and "const_gross" in nets:
        n_min = min(len(nets["per_name"]), len(nets["const_gross"]))
        delta = nets["const_gross"][:n_min] - nets["per_name"][:n_min]
        rng = np.random.default_rng(42)
        n_boot = 2000; block = 7
        n_blocks = int(np.ceil(len(delta) / block))
        boot_means = np.empty(n_boot)
        for i in range(n_boot):
            starts = rng.integers(0, len(delta) - block + 1, size=n_blocks)
            idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:len(delta)]
            boot_means[i] = delta[idx].mean()
        lo_d, hi_d = np.percentile(boot_means, [2.5, 97.5])
        d_sh = _sharpe(nets["const_gross"][:n_min]) - _sharpe(nets["per_name"][:n_min])
        print(f"\n  Paired Δ (const_gross − per_name):")
        print(f"    Δnet={delta.mean():+.3f} bps  CI=[{lo_d:+.3f}, {hi_d:+.3f}]  Δsh={d_sh:+.2f}")

    # Per-fold breakdown
    print("\n  Per-fold Δsh (const_gross − per_name):")
    for fid in range(10):
        per_n = [r["net"] for r in cycles["per_name"] if r["fold"] == fid]
        const_n = [r["net"] for r in cycles["const_gross"] if r["fold"] == fid]
        if not per_n or not const_n: continue
        d_sh = _sharpe(np.array(const_n)) - _sharpe(np.array(per_n))
        print(f"    fold {fid:>2}: per_name Sh={_sharpe(np.array(per_n)):+.2f}  "
              f"const_gross Sh={_sharpe(np.array(const_n)):+.2f}  Δ={d_sh:+.2f}")

    summary.to_csv(OUT_DIR / "weighting_summary.csv", index=False)
    for name in ["per_name", "const_gross"]:
        if cycles[name]:
            pd.DataFrame(cycles[name]).to_csv(OUT_DIR / f"{name}_cycles.csv", index=False)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
