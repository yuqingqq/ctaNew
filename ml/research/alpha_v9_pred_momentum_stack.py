"""PM_M2_b1 × conv_gate p=0.30 stacking — composability test.

Tests whether the pred-momentum entry gate composes with the existing
production conviction gate, or whether they overlap (capture same signal).

If they compose: PM lift adds on top of conv_gate's +1.85 ΔSharpe.
If they overlap: stacked lift ≈ max(individual lifts), gate redundant.

Variants:
  baseline           — sharp K=7
  conv_p30           — conv_gate p=0.30 only (memory-validated production)
  PM_M2_b1           — pred-momentum strict gate only
  conv_p30 + PM      — both gates active (cycle-skip + entry-filter)
"""
from __future__ import annotations
import json, sys, time, warnings
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import (
    ENSEMBLE_SEEDS, _multi_oos_splits, _slice, _train,
)
from ml.research.alpha_v4_xs import portfolio_pnl_turnover_aware, block_bootstrap_ci
from ml.research.alpha_v8_h48_audit import build_wide_panel

HORIZON = 48
TOP_K = 7
TOP_FRAC = TOP_K / 25.0
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
GATE_LOOKBACK = 252
GATE_PCTILE = 0.30
PM_M = 2
PM_BAND = 1.0
OUT_DIR = REPO / "outputs/pred_momentum_stack"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _bn_scale(beta_L, beta_S):
    if beta_L < 0.1 or beta_S < 0.1 or (beta_L + beta_S) < 0.3:
        return 1.0, 1.0, True
    denom = beta_L + beta_S
    return (float(np.clip(2.0 * beta_S / denom, 0.5, 1.5)),
            float(np.clip(2.0 * beta_L / denom, 0.5, 1.5)),
            False)


def evaluate_stacked(
    test: pd.DataFrame, yt: np.ndarray, *,
    use_conv_gate: bool, use_pm_gate: bool,
    top_k: int = TOP_K, cost_bps_per_leg: float = COST_PER_LEG,
    sample_every: int = HORIZON, gate_pctile: float = GATE_PCTILE,
    gate_lookback: int = GATE_LOOKBACK,
    pm_m: int = PM_M, pm_band: float = PM_BAND,
    execution_model: str = "live",
    disp_overlay_lo: float | None = None,
    disp_overlay_hi: float | None = None,
) -> pd.DataFrame:
    """Combined gate evaluator. Conv gate runs first (skip cycle), then
    PM gate (filter NEW entries). Held names auto-keep on sharp K=7.

    execution_model:
      "live"     — hold prior positions through skip/empty-leg cycles, compute
                    real MtM on held names (matches paper_bot.py behavior).
                    Validated 2026-05-09 as the correct model for forward.
      "research" — reset positions on skip/empty-leg, record 0 PnL, pay full
                    re-entry cost at next trade. Original (pessimistic) behavior.

    disp_overlay_lo, disp_overlay_hi (None = disabled):
      Continuous pred_disp size-overlay (port from xyz validated lever).
      For each non-skipped cycle, compute the current dispersion's percentile
      rank within trailing dispersion_history. Linear-clip into [lo, hi] →
      multiplier on scale_L, scale_S. e.g., (0.5, 1.0) means scale at 50%
      when current dispersion is at the 30th-pctile (just above conv_gate
      threshold), full size at 100th-pctile.
    """
    assert execution_model in ("live", "research")
    overlay_active = disp_overlay_lo is not None and disp_overlay_hi is not None
    if overlay_active:
        assert 0.0 <= disp_overlay_lo <= disp_overlay_hi <= 1.5
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
    cur_long: set = set()
    cur_short: set = set()
    prev_long_w: dict[str, float] = {}
    prev_short_w: dict[str, float] = {}
    # Track last trade-cycle's β-neutral scales so live-model can compute
    # MtM on held positions during conv-skip / PM-empty cycles.
    prev_scale_L: float = 1.0
    prev_scale_S: float = 1.0

    for t, g in df.groupby("open_time"):
        n = len(g)
        if n < 2 * top_k + 1:
            continue
        sym_arr = g["symbol"].to_numpy()
        pred_arr = g["pred"].to_numpy()
        idx_top_k = np.argpartition(-pred_arr, top_k - 1)[:top_k]
        idx_bot_k = np.argpartition(pred_arr, top_k - 1)[:top_k]

        # ---- Conv gate: skip cycle if dispersion below trailing pctile
        top_pred_mean = pred_arr[idx_top_k].mean()
        bot_pred_mean = pred_arr[idx_bot_k].mean()
        dispersion = float(top_pred_mean - bot_pred_mean)
        skip = False
        if use_conv_gate and len(dispersion_history) >= 30:
            thr = float(np.quantile(list(dispersion_history), gate_pctile))
            if dispersion < thr:
                skip = True
        dispersion_history.append(dispersion)

        # Always update PM history regardless of skip
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
            if execution_model == "live" and cur_long and cur_short:
                # Hold-through MtM on prior positions. No trades, no cost.
                long_g_h = g[g["symbol"].isin(cur_long)]
                short_g_h = g[g["symbol"].isin(cur_short)]
                if not long_g_h.empty and not short_g_h.empty:
                    gross_L_h = prev_scale_L * len(long_g_h) / top_k
                    gross_S_h = prev_scale_S * len(short_g_h) / top_k
                    long_ret_h = gross_L_h * long_g_h["return_pct"].mean()
                    short_ret_h = gross_S_h * short_g_h["return_pct"].mean()
                    long_alpha_h = gross_L_h * long_g_h["alpha_realized"].mean()
                    short_alpha_h = gross_S_h * short_g_h["alpha_realized"].mean()
                    spread_ret_h = long_ret_h - short_ret_h
                    spread_alpha_h = long_alpha_h - short_alpha_h
                    bars.append({
                        "time": t,
                        "spread_ret_bps": spread_ret_h * 1e4,
                        "spread_alpha_bps": spread_alpha_h * 1e4,
                        "long_turnover": 0.0, "short_turnover": 0.0,
                        "cost_bps": 0.0, "net_bps": spread_ret_h * 1e4,
                        "n_long": len(long_g_h), "n_short": len(short_g_h),
                        "skipped": 1,
                        "gross_L": gross_L_h, "gross_S": gross_S_h,
                    })
                    # Don't reset: keep cur_long/short and prev_long_w/short_w
                    continue
            # Cold start (no held positions) OR research model: record 0 bar.
            bars.append({
                "time": t, "spread_ret_bps": 0.0, "spread_alpha_bps": 0.0,
                "long_turnover": 0.0, "short_turnover": 0.0,
                "cost_bps": 0.0, "net_bps": 0.0,
                "n_long": 0, "n_short": 0, "skipped": 1,
                "gross_L": 0.0, "gross_S": 0.0,
            })
            if execution_model == "research":
                # Skipping closes positions; reset (original research behavior)
                cur_long, cur_short = set(), set()
                prev_long_w = {}; prev_short_w = {}
            continue

        cand_long = set(sym_arr[idx_top_k])
        cand_short = set(sym_arr[idx_bot_k])

        if use_pm_gate:
            new_long = cur_long & cand_long
            new_short = cur_short & cand_short
            if len(history) >= pm_m:
                past_long = [h["long"] for h in history[-(pm_m):][:pm_m - 1]]
                past_short = [h["short"] for h in history[-(pm_m):][:pm_m - 1]]
                # past_long is the M-1 cycles BEFORE current. We just appended
                # current, so history[-(pm_m):][:pm_m-1] = history[-pm_m:-1] i.e.
                # excludes the newly-added current cycle.
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
            if execution_model == "live" and cur_long and cur_short:
                # Hold-through MtM on prior positions (matches paper_bot
                # behavior when PM gate empties a leg — no rebalance).
                long_g_h = g[g["symbol"].isin(cur_long)]
                short_g_h = g[g["symbol"].isin(cur_short)]
                if not long_g_h.empty and not short_g_h.empty:
                    gross_L_h = prev_scale_L * len(long_g_h) / top_k
                    gross_S_h = prev_scale_S * len(short_g_h) / top_k
                    long_ret_h = gross_L_h * long_g_h["return_pct"].mean()
                    short_ret_h = gross_S_h * short_g_h["return_pct"].mean()
                    long_alpha_h = gross_L_h * long_g_h["alpha_realized"].mean()
                    short_alpha_h = gross_S_h * short_g_h["alpha_realized"].mean()
                    spread_ret_h = long_ret_h - short_ret_h
                    spread_alpha_h = long_alpha_h - short_alpha_h
                    bars.append({
                        "time": t,
                        "spread_ret_bps": spread_ret_h * 1e4,
                        "spread_alpha_bps": spread_alpha_h * 1e4,
                        "long_turnover": 0.0, "short_turnover": 0.0,
                        "cost_bps": 0.0, "net_bps": spread_ret_h * 1e4,
                        "n_long": len(long_g_h), "n_short": len(short_g_h),
                        "skipped": 0,  # not conv-skipped, just PM-empty
                        "gross_L": gross_L_h, "gross_S": gross_S_h,
                    })
                    # Don't reset; hold prev positions through PM-empty cycle
                    continue
            # Cold-start OR research model: record 0 bar, reset (research) or hold zero (cold)
            cur_long, cur_short = new_long, new_short
            prev_long_w = {s: 1.0 / top_k for s in new_long}
            prev_short_w = {s: 1.0 / top_k for s in new_short}
            bars.append({
                "time": t, "spread_ret_bps": 0.0, "spread_alpha_bps": 0.0,
                "long_turnover": 0.0, "short_turnover": 0.0,
                "cost_bps": 0.0, "net_bps": 0.0,
                "n_long": len(new_long), "n_short": len(new_short),
                "skipped": 0, "gross_L": 0.0, "gross_S": 0.0,
            })
            continue

        long_g = g[g["symbol"].isin(new_long)]
        short_g = g[g["symbol"].isin(new_short)]
        scale_L, scale_S, _ = _bn_scale(long_g["beta_short_vs_bk"].mean(),
                                        short_g["beta_short_vs_bk"].mean())

        # Pred-disp size-overlay (port from xyz). Scales scale_L/scale_S
        # by the current cycle's dispersion-percentile within trailing
        # dispersion_history. Conv-gate has already filtered very-low-disp
        # cycles; this overlay further throttles size at moderate-low dispersion.
        # Note: dispersion_history was just appended above with current cycle's
        # value; we want to rank against PRIOR cycles, so use [:-1].
        if overlay_active and len(dispersion_history) > 30:
            prior_disp = list(dispersion_history)[:-1]  # exclude current
            cur_disp = float(dispersion_history[-1])
            disp_pctile = float(np.mean(np.array(prior_disp) <= cur_disp))
            disp_mult = max(disp_overlay_lo, min(disp_overlay_hi, disp_pctile))
            scale_L *= disp_mult
            scale_S *= disp_mult

        long_w = {s: scale_L / top_k for s in new_long}
        short_w = {s: scale_S / top_k for s in new_short}
        gross_L = scale_L * len(new_long) / top_k
        gross_S = scale_S * len(new_short) / top_k

        long_ret = gross_L * long_g["return_pct"].mean()
        short_ret = gross_S * short_g["return_pct"].mean()
        long_alpha = gross_L * long_g["alpha_realized"].mean()
        short_alpha = gross_S * short_g["alpha_realized"].mean()
        spread_ret = long_ret - short_ret
        spread_alpha = long_alpha - short_alpha

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
            "time": t,
            "spread_ret_bps": spread_ret * 1e4,
            "spread_alpha_bps": spread_alpha * 1e4,
            "long_turnover": long_to, "short_turnover": short_to,
            "cost_bps": cost_bps, "net_bps": spread_ret * 1e4 - cost_bps,
            "n_long": len(new_long), "n_short": len(new_short),
            "skipped": 0, "gross_L": gross_L, "gross_S": gross_S,
        })
        cur_long, cur_short = new_long, new_short
        prev_long_w, prev_short_w = long_w, short_w
        prev_scale_L, prev_scale_S = scale_L, scale_S  # for live-model hold-through

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

    variants = [
        ("baseline", False, False),
        ("conv_p30", True,  False),
        ("PM_M2_b1", False, True),
        ("conv+PM",  True,  True),
    ]
    cycles_by_var: dict[str, list] = {v[0]: [] for v in variants}

    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200:
            print(f"  fold {fold['fid']:>2}: skipped"); continue
        Xt = tr[avail_feats].to_numpy(dtype=np.float32)
        yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)
        Xc = ca[avail_feats].to_numpy(dtype=np.float32)
        yc_ = ca["demeaned_target"].to_numpy(dtype=np.float32)
        models = [_train(Xt, yt_, Xc, yc_, seed=s) for s in ENSEMBLE_SEEDS]
        Xtest = test[avail_feats].to_numpy(dtype=np.float32)
        pred_test = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                              for m in models], axis=0)

        line = f"  fold {fold['fid']:>2}: "
        for name, use_conv, use_pm in variants:
            df_eval = evaluate_stacked(test, pred_test,
                                        use_conv_gate=use_conv, use_pm_gate=use_pm)
            if df_eval.empty:
                continue
            for _, row in df_eval.iterrows():
                cycles_by_var[name].append({
                    "fold": fold["fid"], "time": row["time"],
                    "spread": row["spread_ret_bps"], "cost": row["cost_bps"],
                    "net": row["net_bps"], "skipped": row["skipped"],
                    "n_long": row["n_long"], "n_short": row["n_short"],
                    "long_to": row["long_turnover"], "short_to": row["short_turnover"],
                })
            net_arr = df_eval["net_bps"].to_numpy()
            line += f"{name}={net_arr.mean():+.2f}({_sharpe(net_arr):+.1f})  "
        print(line + f"({time.time()-t0:.0f}s)")

    print("\n" + "=" * 110)
    print(f"STACKING TEST  (h={HORIZON} K={TOP_K}, β-neutral, 4.5 bps/leg, conv_p={GATE_PCTILE} M={PM_M} band={PM_BAND})")
    print("=" * 110)

    rows = []
    nets = {}
    for name, _, _ in variants:
        df_v = pd.DataFrame(cycles_by_var[name])
        if df_v.empty:
            continue
        net = df_v["net"].to_numpy()
        nets[name] = net
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        rows.append({
            "variant": name, "n": len(net),
            "net_bps": net.mean(),
            "spread_bps": df_v["spread"].mean(),
            "cost_bps": df_v["cost"].mean(),
            "skip_pct": df_v["skipped"].mean() * 100,
            "K_avg": (df_v["n_long"].mean() + df_v["n_short"].mean()) / 2,
            "L_to": df_v["long_to"].mean(),
            "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
        })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False, float_format="%+.2f"))

    print("\n  Paired Δ vs baseline (per-cycle, aligned by time):")
    base_net = nets["baseline"]
    for name in ["conv_p30", "PM_M2_b1", "conv+PM"]:
        if name not in nets:
            continue
        n_min = min(len(base_net), len(nets[name]))
        delta = nets[name][:n_min] - base_net[:n_min]
        delta_sh = _sharpe(delta)
        t_stat = delta.mean() / (delta.std() / np.sqrt(len(delta))) if delta.std() > 0 else 0
        # Bootstrap CI on Δnet mean
        rng = np.random.default_rng(42)
        n_boot = 2000
        block = 7
        n_blocks = int(np.ceil(len(delta) / block))
        boot_means = np.empty(n_boot)
        for i in range(n_boot):
            starts = rng.integers(0, len(delta) - block + 1, size=n_blocks)
            idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:len(delta)]
            boot_means[i] = delta[idx].mean()
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        d_sh_to_base = _sharpe(nets[name][:n_min]) - _sharpe(base_net[:n_min])
        print(f"    {name:<10}  Δnet={delta.mean():+.3f} bps  CI95=[{lo:+.3f}, {hi:+.3f}]  "
              f"Δsh={d_sh_to_base:+.2f}  t={t_stat:+.2f}  paired Δsh={delta_sh:+.2f}")

    # Compositionality test: is conv+PM ≈ conv_p30 + PM_M2_b1, or do they overlap?
    if all(k in nets for k in ["baseline", "conv_p30", "PM_M2_b1", "conv+PM"]):
        n_min = min(len(nets[k]) for k in nets)
        d_conv = nets["conv_p30"][:n_min] - nets["baseline"][:n_min]
        d_pm = nets["PM_M2_b1"][:n_min] - nets["baseline"][:n_min]
        d_stk = nets["conv+PM"][:n_min] - nets["baseline"][:n_min]
        additive = d_conv.mean() + d_pm.mean()
        actual = d_stk.mean()
        print(f"\n  Compositionality:")
        print(f"    Δnet conv alone:  {d_conv.mean():+.3f} bps")
        print(f"    Δnet PM alone:    {d_pm.mean():+.3f} bps")
        print(f"    Sum if additive:  {additive:+.3f} bps")
        print(f"    Δnet conv+PM:     {actual:+.3f} bps  ({100*actual/additive:+.0f}% of additive)")

    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    for name, _, _ in variants:
        if cycles_by_var[name]:
            pd.DataFrame(cycles_by_var[name]).to_csv(OUT_DIR / f"{name}_cycles.csv", index=False)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
