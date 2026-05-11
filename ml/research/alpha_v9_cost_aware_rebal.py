"""Cost-aware rebalancing variants — quick 2-fold sanity test.

Tests two rules from STRATEGY_OPTIMIZATION_PLAN.md §3:

A. Alpha-gap swap rule (discrete K=7 with retention bonus on held names).
   Plan §3 prescribes margin = round-trip cost (~9 bps). Sweep margin ∈
   {0, 0.5, 1.0, 2.0} × round-trip cost.

B. Continuous-weight LP optimizer with turnover penalty:
     max Σ pred_i w_i - λ Σ |w_i - w_prev_i|
     s.t. Σ w_i = 1, 0 ≤ w_i ≤ w_cap = 2/K  (per leg)
   λ in pred-bps units, swept ∈ {0.5, 1.0, 2.0} × one-way cost.

Both run on the same predictions per fold (LGBM v6_clean ensemble, β-neutral
execution, 4.5 bps/leg HL VIP-0). Reference: M=2 rank-band hysteresis.

If any variant shows positive direction in the 2-fold sanity (Δnet_bps > 0),
escalate to full multi-OOS validation.
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import (
    ENSEMBLE_SEEDS, _multi_oos_splits, _slice, _train,
)
from ml.research.alpha_v4_xs import portfolio_pnl_turnover_aware
from ml.research.alpha_v8_h48_audit import build_wide_panel
from ml.research.alpha_v8_hysteresis_multioos import portfolio_pnl_hysteresis_bn

HORIZON = 48
TOP_K = 7
TOP_FRAC = TOP_K / 25.0
COST_PER_LEG = 4.5  # HL VIP-0 taker
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
N_FOLDS = 2
OUT_DIR = REPO / "outputs/cost_aware_rebal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RTC = 2 * COST_PER_LEG  # 9 bps round-trip
A_MARGINS_BPS = [0.0, 0.5 * RTC, 1.0 * RTC, 2.0 * RTC]
B_LAMBDAS_BPS = [0.5 * COST_PER_LEG, 1.0 * COST_PER_LEG, 2.0 * COST_PER_LEG]


def _calib_pred_to_bps(pred_cal: np.ndarray, alpha_cal: np.ndarray) -> float:
    """OLS slope of alpha_realized_bps ~ pred. Used to translate margin/λ
    in bps-space to pred-units. Falls back to 1.0 if degenerate."""
    pred = np.asarray(pred_cal, dtype=float)
    y = np.asarray(alpha_cal, dtype=float) * 1e4
    if pred.std() < 1e-9 or len(pred) < 50:
        return 1.0
    slope = float(np.cov(pred, y, ddof=0)[0, 1] / np.var(pred))
    return max(slope, 1e-6)


def _beta_neutral_scale(beta_L: float, beta_S: float):
    if beta_L < 0.1 or beta_S < 0.1 or (beta_L + beta_S) < 0.3:
        return 1.0, 1.0, True
    denom = beta_L + beta_S
    sL = float(np.clip(2.0 * beta_S / denom, 0.5, 1.5))
    sS = float(np.clip(2.0 * beta_L / denom, 0.5, 1.5))
    return sL, sS, False


def portfolio_swap_rule_bn(
    test: pd.DataFrame, yt: np.ndarray, *, top_k: int, margin_bps: float,
    slope: float, cost_bps_per_leg: float, sample_every: int,
) -> dict:
    """Variant A. Held names get +bonus = margin_bps / slope in pred units.
    A non-held candidate displaces a held name iff its pred exceeds held's
    by > bonus (i.e. > margin_bps in alpha-bps space)."""
    cols = ["open_time", "symbol", "return_pct", "alpha_realized",
            "basket_fwd", "beta_short_vs_bk"]
    df = test[cols].copy()
    df["pred"] = yt
    times = sorted(df["open_time"].unique())
    if not times:
        return {"n_bars": 0}
    if sample_every > 1:
        keep_times = set(times[::sample_every])
        df = df[df["open_time"].isin(keep_times)]

    bonus_pred = margin_bps / slope
    bars = []
    cur_long, cur_short = set(), set()
    prev_long_w, prev_short_w = {}, {}

    for t, g in df.groupby("open_time"):
        n = len(g)
        if n < 2 * top_k:
            continue
        sym_arr = g["symbol"].to_numpy()
        pred_arr = g["pred"].to_numpy()
        is_held_long = np.array([s in cur_long for s in sym_arr])
        is_held_short = np.array([s in cur_short for s in sym_arr])
        score_L = pred_arr + bonus_pred * is_held_long.astype(float)
        score_S = -pred_arr + bonus_pred * is_held_short.astype(float)

        idx_L = np.argpartition(-score_L, top_k)[:top_k]
        idx_S = np.argpartition(-score_S, top_k)[:top_k]
        new_long = set(sym_arr[idx_L])
        new_short = set(sym_arr[idx_S])

        # Resolve any overlap (rare): higher pred wins for long, drops from short
        overlap = new_long & new_short
        if overlap:
            for s in overlap:
                p = pred_arr[sym_arr == s][0]
                if p > 0:
                    new_short.discard(s)
                else:
                    new_long.discard(s)
            # Backfill missing slots (unlikely path; pred-rank fallback)
            while len(new_long) < top_k:
                avail = [s for s in sym_arr if s not in new_long and s not in new_short]
                cand = max(avail, key=lambda s: pred_arr[sym_arr == s][0])
                new_long.add(cand)
            while len(new_short) < top_k:
                avail = [s for s in sym_arr if s not in new_long and s not in new_short]
                cand = min(avail, key=lambda s: pred_arr[sym_arr == s][0])
                new_short.add(cand)

        long_g = g[g["symbol"].isin(new_long)]
        short_g = g[g["symbol"].isin(new_short)]
        scale_L, scale_S, degen = _beta_neutral_scale(
            long_g["beta_short_vs_bk"].mean(), short_g["beta_short_vs_bk"].mean()
        )

        long_w = {s: scale_L / top_k for s in new_long}
        short_w = {s: scale_S / top_k for s in new_short}
        if not prev_long_w:
            long_to, short_to = scale_L, scale_S
        else:
            all_l = set(long_w) | set(prev_long_w)
            long_to = sum(abs(long_w.get(s, 0) - prev_long_w.get(s, 0)) for s in all_l)
            all_s = set(short_w) | set(prev_short_w)
            short_to = sum(abs(short_w.get(s, 0) - prev_short_w.get(s, 0)) for s in all_s)

        spread_ret = scale_L * long_g["return_pct"].mean() - scale_S * short_g["return_pct"].mean()
        spread_alpha = scale_L * long_g["alpha_realized"].mean() - scale_S * short_g["alpha_realized"].mean()
        cost_bps = cost_bps_per_leg * (long_to + short_to)
        bars.append({
            "time": t, "spread_ret_bps": spread_ret * 1e4,
            "spread_alpha_bps": spread_alpha * 1e4,
            "long_turnover": long_to, "short_turnover": short_to,
            "cost_bps": cost_bps, "net_bps": spread_ret * 1e4 - cost_bps,
            "degen_beta": int(degen),
        })
        cur_long, cur_short = new_long, new_short
        prev_long_w, prev_short_w = long_w, short_w

    bdf = pd.DataFrame(bars)
    if bdf.empty:
        return {"n_bars": 0}
    return {
        "n_bars": len(bdf),
        "spread_ret_bps_mean": float(bdf["spread_ret_bps"].mean()),
        "cost_bps_mean": float(bdf["cost_bps"].mean()),
        "net_bps_mean": float(bdf["net_bps"].mean()),
        "long_turnover_mean": float(bdf["long_turnover"].mean()),
        "short_turnover_mean": float(bdf["short_turnover"].mean()),
        "df": bdf,
    }


def _solve_leg_lp(pred: np.ndarray, w_prev: np.ndarray, *,
                   lam_pred: float, eff_cap: np.ndarray) -> np.ndarray:
    """Per-leg LP:
        max Σ pred_i w_i - λ Σ u_i
        s.t. Σ w_i = 1, 0 ≤ w_i ≤ eff_cap_i, u_i ≥ |w_i - w_prev_i|
    eff_cap_i = 0 means symbol unavailable. Falls back to uniform top-K
    if infeasible or solver fails.
    """
    n = len(pred)
    avail = eff_cap > 1e-9
    n_avail = int(avail.sum())
    if n_avail == 0 or eff_cap.sum() < 0.999:
        # Fallback: uniform on top-K available
        w = np.zeros(n)
        if n_avail > 0:
            target_k = min(7, n_avail)
            idx_av = np.where(avail)[0]
            top = idx_av[np.argsort(-pred[idx_av])[:target_k]]
            w[top] = 1.0 / target_k
        return w

    # Variables [w_1..n, u_1..n], minimize -pred·w + λ·1·u
    c = np.concatenate([-pred, lam_pred * np.ones(n)])
    A_eq = np.zeros((1, 2 * n))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])
    # u_i ≥ w_i - w_prev_i  →   w_i - u_i ≤ w_prev_i
    # u_i ≥ -(w_i - w_prev_i) → -w_i - u_i ≤ -w_prev_i
    A_ub = np.vstack([
        np.hstack([np.eye(n), -np.eye(n)]),
        np.hstack([-np.eye(n), -np.eye(n)]),
    ])
    b_ub = np.concatenate([w_prev, -w_prev])
    bounds = [(0.0, float(eff_cap[i])) for i in range(n)] + [(0.0, None)] * n
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        w = np.zeros(n)
        idx_av = np.where(avail)[0]
        top = idx_av[np.argsort(-pred[idx_av])[:min(7, n_avail)]]
        w[top] = 1.0 / len(top)
        return w
    return res.x[:n]


def portfolio_lp_optimizer_bn(
    test: pd.DataFrame, yt: np.ndarray, *, top_k_target: int, lam_bps: float,
    slope: float, cost_bps_per_leg: float, sample_every: int,
) -> dict:
    """Variant B. Continuous-weight LP per leg. w_cap = 2/K (≥ K/2 names
    must hold weight), turnover penalty λ in pred-bps units."""
    cols = ["open_time", "symbol", "return_pct", "alpha_realized",
            "basket_fwd", "beta_short_vs_bk"]
    df = test[cols].copy()
    df["pred"] = yt
    times = sorted(df["open_time"].unique())
    if not times:
        return {"n_bars": 0}
    if sample_every > 1:
        keep_times = set(times[::sample_every])
        df = df[df["open_time"].isin(keep_times)]

    w_cap = 2.0 / top_k_target
    lam_pred = lam_bps / slope

    universe = sorted(df["symbol"].unique())
    sym_idx = {s: i for i, s in enumerate(universe)}
    n_full = len(universe)

    bars = []
    prev_long_w = np.zeros(n_full)
    prev_short_w = np.zeros(n_full)
    is_first = True

    for t, g in df.groupby("open_time"):
        pred_arr = np.zeros(n_full)
        avail = np.zeros(n_full, dtype=bool)
        ret_arr = np.zeros(n_full)
        alpha_arr = np.zeros(n_full)
        beta_arr = np.zeros(n_full)
        for _, row in g.iterrows():
            i = sym_idx[row["symbol"]]
            pred_arr[i] = row["pred"]
            ret_arr[i] = row["return_pct"]
            alpha_arr[i] = row["alpha_realized"]
            beta_arr[i] = row["beta_short_vs_bk"]
            avail[i] = True
        eff_cap = np.where(avail, w_cap, 0.0)
        if eff_cap.sum() < 0.999:
            continue

        w_long = _solve_leg_lp(pred_arr, prev_long_w if not is_first else np.zeros(n_full),
                                lam_pred=lam_pred, eff_cap=eff_cap)
        w_short = _solve_leg_lp(-pred_arr, prev_short_w if not is_first else np.zeros(n_full),
                                 lam_pred=lam_pred, eff_cap=eff_cap)

        # Beta-neutral scale on weighted leg betas
        beta_L = float((w_long * beta_arr).sum())
        beta_S = float((w_short * beta_arr).sum())
        scale_L, scale_S, degen = _beta_neutral_scale(beta_L, beta_S)

        long_ret = float((w_long * ret_arr).sum())
        short_ret = float((w_short * ret_arr).sum())
        long_alpha = float((w_long * alpha_arr).sum())
        short_alpha = float((w_short * alpha_arr).sum())
        spread_ret = scale_L * long_ret - scale_S * short_ret
        spread_alpha = scale_L * long_alpha - scale_S * short_alpha

        scaled_long_w = scale_L * w_long
        scaled_short_w = scale_S * w_short
        if is_first:
            long_to = float(np.abs(scaled_long_w).sum())
            short_to = float(np.abs(scaled_short_w).sum())
        else:
            long_to = float(np.abs(scaled_long_w - prev_long_w).sum())
            short_to = float(np.abs(scaled_short_w - prev_short_w).sum())

        cost_bps = cost_bps_per_leg * (long_to + short_to)
        bars.append({
            "time": t, "spread_ret_bps": spread_ret * 1e4,
            "spread_alpha_bps": spread_alpha * 1e4,
            "long_turnover": long_to, "short_turnover": short_to,
            "cost_bps": cost_bps, "net_bps": spread_ret * 1e4 - cost_bps,
            "n_long_active": int((w_long > 1e-9).sum()),
            "n_short_active": int((w_short > 1e-9).sum()),
            "degen_beta": int(degen),
        })
        prev_long_w = scaled_long_w
        prev_short_w = scaled_short_w
        is_first = False

    bdf = pd.DataFrame(bars)
    if bdf.empty:
        return {"n_bars": 0}
    return {
        "n_bars": len(bdf),
        "spread_ret_bps_mean": float(bdf["spread_ret_bps"].mean()),
        "cost_bps_mean": float(bdf["cost_bps"].mean()),
        "net_bps_mean": float(bdf["net_bps"].mean()),
        "long_turnover_mean": float(bdf["long_turnover"].mean()),
        "short_turnover_mean": float(bdf["short_turnover"].mean()),
        "n_long_active_mean": float(bdf["n_long_active"].mean()),
        "n_short_active_mean": float(bdf["n_short_active"].mean()),
        "df": bdf,
    }


def main():
    panel = build_wide_panel()
    folds = _multi_oos_splits(panel)
    print(f"Total folds: {len(folds)}; running first {N_FOLDS}")

    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    avail_feats = [c for c in v6_clean if c in panel.columns]

    rows = []
    for fold in folds[:N_FOLDS]:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200:
            print(f"  fold {fold['fid']}: skipped (insufficient data)"); continue
        Xt = tr[avail_feats].to_numpy(dtype=np.float32)
        yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)
        Xc = ca[avail_feats].to_numpy(dtype=np.float32)
        yc_ = ca["demeaned_target"].to_numpy(dtype=np.float32)
        models = [_train(Xt, yt_, Xc, yc_, seed=s) for s in ENSEMBLE_SEEDS]
        Xtest = test[avail_feats].to_numpy(dtype=np.float32)
        Xcal = ca[avail_feats].to_numpy(dtype=np.float32)
        pred_test = np.mean([m.predict(Xtest, num_iteration=m.best_iteration) for m in models], axis=0)
        pred_cal = np.mean([m.predict(Xcal, num_iteration=m.best_iteration) for m in models], axis=0)

        slope = _calib_pred_to_bps(pred_cal, ca["alpha_realized"].to_numpy())
        print(f"  fold {fold['fid']}: pred→bps slope = {slope:.2f}  (train {time.time()-t0:.0f}s)")

        results = {}
        results["baseline"] = portfolio_pnl_turnover_aware(
            test, pred_test, top_frac=TOP_FRAC,
            cost_bps_per_leg=COST_PER_LEG, sample_every=HORIZON, beta_neutral=True,
        )
        results["M=2_hyst"] = portfolio_pnl_hysteresis_bn(
            test, pred_test, top_k=TOP_K, exit_buffer=2,
            cost_bps_per_leg=COST_PER_LEG, sample_every=HORIZON, beta_neutral=True,
        )
        for m_bps in A_MARGINS_BPS:
            results[f"A_m{m_bps:.1f}"] = portfolio_swap_rule_bn(
                test, pred_test, top_k=TOP_K, margin_bps=m_bps, slope=slope,
                cost_bps_per_leg=COST_PER_LEG, sample_every=HORIZON,
            )
        for lam_bps in B_LAMBDAS_BPS:
            results[f"B_lam{lam_bps:.1f}"] = portfolio_lp_optimizer_bn(
                test, pred_test, top_k_target=TOP_K, lam_bps=lam_bps, slope=slope,
                cost_bps_per_leg=COST_PER_LEG, sample_every=HORIZON,
            )

        for label, r in results.items():
            if r.get("n_bars", 0) == 0:
                continue
            df = r["df"]
            sh = df["net_bps"].mean() / df["net_bps"].std() * np.sqrt(CYCLES_PER_YEAR) if df["net_bps"].std() > 0 else 0
            rows.append({
                "fold": fold["fid"], "variant": label, "n": r["n_bars"],
                "net": df["net_bps"].mean(), "spread": df["spread_ret_bps"].mean(),
                "cost": df["cost_bps"].mean(),
                "L_to": df["long_turnover"].mean(), "S_to": df["short_turnover"].mean(),
                "sh_fold": float(sh),
            })
        print(f"  fold {fold['fid']}: total {time.time()-t0:.0f}s")

    summary_df = pd.DataFrame(rows)
    by_var = summary_df.groupby("variant").agg(
        net=("net", "mean"), spread=("spread", "mean"), cost=("cost", "mean"),
        L_to=("L_to", "mean"), S_to=("S_to", "mean"),
        sh_avg=("sh_fold", "mean"),
    ).reset_index()
    order = (["baseline", "M=2_hyst"]
             + [f"A_m{m:.1f}" for m in A_MARGINS_BPS]
             + [f"B_lam{l:.1f}" for l in B_LAMBDAS_BPS])
    by_var["sort"] = by_var["variant"].apply(lambda v: order.index(v) if v in order else 999)
    by_var = by_var.sort_values("sort").drop(columns=["sort"]).reset_index(drop=True)

    print("\n" + "=" * 90)
    print(f"VARIANT COMPARISON  (h={HORIZON}, K={TOP_K}, {N_FOLDS} folds, 4.5 bps/leg, β-neutral)")
    print("=" * 90)
    print(by_var.to_string(index=False, float_format="%+.2f"))
    base_row = by_var[by_var["variant"] == "baseline"].iloc[0]
    print("\nΔ vs baseline (net_bps gain):")
    for _, r in by_var.iterrows():
        if r["variant"] == "baseline":
            continue
        print(f"  {r['variant']:<12}  Δnet={r['net']-base_row['net']:+.2f} bps   "
              f"Δcost={r['cost']-base_row['cost']:+.2f}   Δsh={r['sh_avg']-base_row['sh_avg']:+.2f}")

    summary_df.to_csv(OUT_DIR / "fold_rows.csv", index=False)
    by_var.to_csv(OUT_DIR / "by_variant.csv", index=False)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
