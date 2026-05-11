"""BTC-hedge variants vs cross-sectional baseline.

Evaluates whether replacing one side of the cross-sectional book with a
single BTC hedge (and optionally picking only the side with stronger
conviction each cycle) preserves enough alpha to justify the cost saving
(7+1=8 trades per rebalance vs 7+7=14).

Variants:
  baseline       — current production: long top-K alts, short bot-K alts
  long_only_btc  — long top-K alts, short BTC (forgo short alpha)
  short_only_btc — short bot-K alts, long BTC (forgo long alpha)
  pick_stronger  — each cycle, pick |mean(top-K pred)| vs |mean(bot-K pred)|;
                   trade winner-side alts, opposite BTC

All variants run conv+PM gates on top of selection. β-neutral: BTC is sized
so dollar-β of BTC leg matches dollar-β of alt leg (with BTC's β=1, this
means BTC notional = alt_leg_notional × mean(β_alts)).

Cost model: uniform 4.5 bps/leg (research convention). Real BTC slippage
is much lower than alt slippage, so this UNDERSTATES the BTC-hedge benefit.
"""
from __future__ import annotations
import sys, time, warnings
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
from ml.research.alpha_v9_pred_momentum_stack import evaluate_stacked

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
BTC_SYMBOL = "BTCUSDT"
OUT_DIR = REPO / "outputs/btc_hedge"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _bn_scale(beta_L, beta_S):
    if beta_L < 0.1 or beta_S < 0.1 or (beta_L + beta_S) < 0.3:
        return 1.0, 1.0
    denom = beta_L + beta_S
    return (float(np.clip(2.0 * beta_S / denom, 0.5, 1.5)),
            float(np.clip(2.0 * beta_L / denom, 0.5, 1.5)))


def evaluate_btc_hedge(
    test: pd.DataFrame, yt: np.ndarray, *,
    side_mode: str,  # "long_only" | "short_only" | "pick_stronger"
    use_conv_gate: bool = True, use_pm_gate: bool = True,
    top_k: int = TOP_K, cost_bps_per_leg: float = COST_PER_LEG,
    sample_every: int = HORIZON,
) -> pd.DataFrame:
    """BTC-hedge evaluator. Selects K alt names on the active side; uses
    BTC as single-name hedge on the other side. β-neutral via BTC scaling."""
    assert side_mode in ("long_only", "short_only", "pick_stronger")
    cols = ["open_time", "symbol", "return_pct", "alpha_realized",
            "basket_fwd", "beta_short_vs_bk"]
    df = test[cols].copy()
    df["pred"] = yt
    times = sorted(df["open_time"].unique())
    if not times: return pd.DataFrame()
    if sample_every > 1:
        keep_times = set(times[::sample_every])
        df = df[df["open_time"].isin(keep_times)]

    band_k = max(top_k, int(round(PM_BAND * top_k)))
    history = []
    dispersion_history = deque(maxlen=GATE_LOOKBACK)
    cur_long, cur_short = set(), set()
    prev_long_w, prev_short_w = {}, {}
    prev_scale_L, prev_scale_S = 1.0, 1.0

    bars = []
    for t, g in df.groupby("open_time"):
        if len(g) < 2 * top_k + 1: continue
        # Need BTC in the cycle to use as hedge
        btc_row = g[g["symbol"] == BTC_SYMBOL]
        if btc_row.empty: continue

        sym_arr = g["symbol"].to_numpy()
        pred_arr = g["pred"].to_numpy()
        idx_top_k = np.argpartition(-pred_arr, top_k - 1)[:top_k]
        idx_bot_k = np.argpartition(pred_arr, top_k - 1)[:top_k]

        # Conv gate (same as evaluate_stacked)
        top_pred_mean = float(pred_arr[idx_top_k].mean())
        bot_pred_mean = float(pred_arr[idx_bot_k].mean())
        dispersion = top_pred_mean - bot_pred_mean
        skip = False
        if use_conv_gate and len(dispersion_history) >= 30:
            thr = float(np.quantile(list(dispersion_history), GATE_PCTILE))
            if dispersion < thr: skip = True
        dispersion_history.append(dispersion)

        # PM history (band-K)
        bk = min(band_k, len(g))
        idx_top_band = np.argpartition(-pred_arr, bk - 1)[:bk] if bk < len(g) else np.arange(len(g))
        idx_bot_band = np.argpartition(pred_arr, bk - 1)[:bk] if bk < len(g) else np.arange(len(g))
        history.append({"long": set(sym_arr[idx_top_band]), "short": set(sym_arr[idx_bot_band])})
        if len(history) > PM_M: history = history[-PM_M:]

        if skip:
            # Hold-through MtM (live-model)
            if cur_long and cur_short:
                # Position structure for skip: cur_long is alt set or {BTC},
                # cur_short is the opposite. Compute MtM on whatever's held.
                long_g_h = g[g["symbol"].isin(cur_long)]
                short_g_h = g[g["symbol"].isin(cur_short)]
                if not long_g_h.empty and not short_g_h.empty:
                    n_long_h = len(long_g_h); n_short_h = len(short_g_h)
                    gross_L_h = prev_scale_L * n_long_h / max(n_long_h, top_k) if n_long_h > 1 else prev_scale_L
                    gross_S_h = prev_scale_S * n_short_h / max(n_short_h, top_k) if n_short_h > 1 else prev_scale_S
                    long_ret_h = gross_L_h * long_g_h["return_pct"].mean()
                    short_ret_h = gross_S_h * short_g_h["return_pct"].mean()
                    spread_ret_h = long_ret_h - short_ret_h
                    bars.append({
                        "time": t, "spread_ret_bps": spread_ret_h * 1e4,
                        "long_turnover": 0.0, "short_turnover": 0.0,
                        "cost_bps": 0.0, "net_bps": spread_ret_h * 1e4,
                        "n_long": n_long_h, "n_short": n_short_h, "skipped": 1,
                        "side": "skip_hold",
                    })
                    continue
            bars.append({
                "time": t, "spread_ret_bps": 0.0,
                "long_turnover": 0.0, "short_turnover": 0.0,
                "cost_bps": 0.0, "net_bps": 0.0,
                "n_long": 0, "n_short": 0, "skipped": 1,
                "side": "skip_no_pos",
            })
            continue

        cand_top = set(sym_arr[idx_top_k])
        cand_bot = set(sym_arr[idx_bot_k])

        # Apply PM gate (per-name persistence)
        if use_pm_gate:
            new_long = cur_long & cand_top
            new_short = cur_short & cand_bot
            if len(history) >= PM_M:
                past_long = [h["long"] for h in history[-PM_M:][:PM_M-1]]
                past_short = [h["short"] for h in history[-PM_M:][:PM_M-1]]
                for s in cand_top - cur_long:
                    if all(s in p for p in past_long): new_long.add(s)
                for s in cand_bot - cur_short:
                    if all(s in p for p in past_short): new_short.add(s)
            else:
                new_long |= cand_top
                new_short |= cand_bot
            if len(new_long) > top_k:
                ranked = sorted(new_long, key=lambda s: -pred_arr[sym_arr == s][0])[:top_k]
                new_long = set(ranked)
            if len(new_short) > top_k:
                ranked = sorted(new_short, key=lambda s: pred_arr[sym_arr == s][0])[:top_k]
                new_short = set(ranked)
        else:
            new_long, new_short = cand_top, cand_bot

        if not new_long or not new_short:
            # Treat as skip (similar to evaluate_stacked's empty-leg path)
            if cur_long and cur_short:
                long_g_h = g[g["symbol"].isin(cur_long)]
                short_g_h = g[g["symbol"].isin(cur_short)]
                if not long_g_h.empty and not short_g_h.empty:
                    n_long_h = len(long_g_h); n_short_h = len(short_g_h)
                    gross_L_h = prev_scale_L * n_long_h / max(n_long_h, top_k) if n_long_h > 1 else prev_scale_L
                    gross_S_h = prev_scale_S * n_short_h / max(n_short_h, top_k) if n_short_h > 1 else prev_scale_S
                    long_ret_h = gross_L_h * long_g_h["return_pct"].mean()
                    short_ret_h = gross_S_h * short_g_h["return_pct"].mean()
                    spread_ret_h = long_ret_h - short_ret_h
                    bars.append({
                        "time": t, "spread_ret_bps": spread_ret_h * 1e4,
                        "long_turnover": 0.0, "short_turnover": 0.0,
                        "cost_bps": 0.0, "net_bps": spread_ret_h * 1e4,
                        "n_long": n_long_h, "n_short": n_short_h, "skipped": 0,
                        "side": "pm_empty_hold",
                    })
                    continue
            bars.append({"time": t, "spread_ret_bps": 0.0,
                         "long_turnover": 0.0, "short_turnover": 0.0,
                         "cost_bps": 0.0, "net_bps": 0.0,
                         "n_long": 0, "n_short": 0, "skipped": 0,
                         "side": "pm_empty_no_pos"})
            continue

        # === Pick which side based on side_mode ===
        if side_mode == "long_only":
            active_alts = new_long; active_side = "L"
        elif side_mode == "short_only":
            active_alts = new_short; active_side = "S"
        else:  # pick_stronger
            # Compare |top mean pred| vs |bot mean pred| as conviction proxy
            top_conv = abs(pred_arr[idx_top_k].mean())
            bot_conv = abs(pred_arr[idx_bot_k].mean())
            if top_conv >= bot_conv:
                active_alts = new_long; active_side = "L"
            else:
                active_alts = new_short; active_side = "S"

        # Build alt leg + BTC hedge leg
        alt_g = g[g["symbol"].isin(active_alts)]
        if alt_g.empty: continue
        # BTC is the hedge — exclude from active_alts if BTC happened to be picked
        if BTC_SYMBOL in active_alts:
            # BTC is in the alt leg; use a different name as hedge (next best alt)
            # Or simpler: drop BTC from alt and use as hedge. K_actual drops by 1.
            active_alts = active_alts - {BTC_SYMBOL}
            alt_g = g[g["symbol"].isin(active_alts)]
            if alt_g.empty: continue

        # β-neutral sizing: scale BTC notional so dollar-β of BTC = dollar-β of alts
        # Convention: β_short_vs_bk is a feature representing each name's β to basket.
        # For BTC hedge (single name), use scale_BTC = scale_alts × mean(β_alts) / β_BTC
        beta_alts = float(alt_g["beta_short_vs_bk"].mean())
        beta_btc = float(btc_row["beta_short_vs_bk"].iloc[0])
        if beta_alts < 0.1 or beta_btc < 0.1:
            scale_alts, scale_btc = 1.0, 1.0
        else:
            # Want: scale_alts * beta_alts = scale_btc * beta_btc (same dollar β)
            # And total gross ≈ 2 (split equally is not required, target similar)
            scale_alts, scale_btc = _bn_scale(beta_alts, beta_btc)
            # Adjust BTC for β ratio
            scale_btc = scale_btc * (beta_alts / beta_btc) if beta_btc > 0 else scale_btc
            scale_btc = float(np.clip(scale_btc, 0.5, 2.0))

        # Per-name weights: alts at scale_alts/top_k each, BTC alone at scale_btc
        if active_side == "L":
            long_w = {s: scale_alts / top_k for s in active_alts}
            short_w = {BTC_SYMBOL: scale_btc}
            cur_long_new = set(active_alts)
            cur_short_new = {BTC_SYMBOL}
        else:
            short_w = {s: scale_alts / top_k for s in active_alts}
            long_w = {BTC_SYMBOL: scale_btc}
            cur_long_new = {BTC_SYMBOL}
            cur_short_new = set(active_alts)

        # Compute spread
        if active_side == "L":
            long_ret = sum(long_w[s] * alt_g.set_index("symbol").loc[s, "return_pct"] for s in active_alts)
            short_ret = scale_btc * float(btc_row["return_pct"].iloc[0])
        else:
            short_ret = sum(short_w[s] * alt_g.set_index("symbol").loc[s, "return_pct"] for s in active_alts)
            long_ret = scale_btc * float(btc_row["return_pct"].iloc[0])
        spread_ret = long_ret - short_ret

        # Turnover (compare to prev)
        if not prev_long_w:
            long_to = sum(abs(v) for v in long_w.values())
            short_to = sum(abs(v) for v in short_w.values())
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
            "n_long": len(long_w), "n_short": len(short_w),
            "skipped": 0, "side": active_side,
        })
        cur_long, cur_short = cur_long_new, cur_short_new
        prev_long_w, prev_short_w = long_w, short_w
        prev_scale_L = scale_alts if active_side == "L" else scale_btc
        prev_scale_S = scale_btc if active_side == "L" else scale_alts

    return pd.DataFrame(bars)


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def _max_dd(net):
    cum = np.cumsum(net)
    peak = np.maximum.accumulate(cum)
    return float((cum - peak).min())


def main():
    panel = build_wide_panel()
    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    avail_feats = [c for c in v6_clean if c in panel.columns]

    cycles = {
        "baseline_xs":     [],     # current production (cross-sectional)
        "long_only_btc":   [],
        "short_only_btc":  [],
        "pick_stronger":   [],
    }
    side_pick_log = []  # for pick_stronger: log which side picked each cycle

    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200: continue
        Xt = tr[avail_feats].to_numpy(np.float32); yt_ = tr["demeaned_target"].to_numpy(np.float32)
        Xc = ca[avail_feats].to_numpy(np.float32); yc_ = ca["demeaned_target"].to_numpy(np.float32)
        models = [_train(Xt, yt_, Xc, yc_, seed=s) for s in ENSEMBLE_SEEDS]
        Xtest = test[avail_feats].to_numpy(np.float32)
        pred_test = np.mean([m.predict(Xtest, num_iteration=m.best_iteration) for m in models], axis=0)

        # Baseline (current production)
        df_xs = evaluate_stacked(test, pred_test, use_conv_gate=True, use_pm_gate=True)
        for _, r in df_xs.iterrows():
            cycles["baseline_xs"].append({
                "fold": fold["fid"], "time": r["time"],
                "net": r["net_bps"], "cost": r["cost_bps"],
                "skipped": r["skipped"],
            })

        # BTC-hedge variants
        for label, mode in [
            ("long_only_btc", "long_only"),
            ("short_only_btc", "short_only"),
            ("pick_stronger", "pick_stronger"),
        ]:
            df_v = evaluate_btc_hedge(test, pred_test, side_mode=mode,
                                       use_conv_gate=True, use_pm_gate=True)
            for _, r in df_v.iterrows():
                cycles[label].append({
                    "fold": fold["fid"], "time": r["time"],
                    "net": r["net_bps"], "cost": r["cost_bps"],
                    "skipped": r["skipped"],
                    "side": r.get("side", "?"),
                })

        line = f"  fold {fold['fid']:>2}: "
        for label in cycles:
            df_t = pd.DataFrame(cycles[label])
            df_t = df_t[df_t["fold"] == fold["fid"]] if "fold" in df_t.columns else df_t
            n = df_t["net"].to_numpy() if not df_t.empty else np.array([0.0])
            line += f"{label[:13]}={n.mean():+.2f}({_sharpe(n):+.1f}) "
        print(line + f"({time.time()-t0:.0f}s)")

    # ===== Headline =====
    print("\n" + "=" * 110)
    print("BTC-HEDGE VARIANTS  (multi-OOS, conv+PM stack, live-model, 4.5 bps/leg uniform)")
    print("=" * 110)
    print(f"  {'variant':<20}  {'n':>4}  {'mean_net':>9}  {'mean_cost':>10}  "
          f"{'Sharpe':>7}  {'CI_lo':>7}  {'CI_hi':>7}  {'max_DD':>8}")
    rows = []
    for label in cycles:
        df_v = pd.DataFrame(cycles[label])
        if df_v.empty: continue
        net = df_v["net"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        max_dd = _max_dd(net)
        cost_avg = df_v["cost"].mean()
        rows.append({"variant": label, "n": len(net),
                     "mean_net": net.mean(), "mean_cost": cost_avg,
                     "sharpe": sh, "ci_lo": lo, "ci_hi": hi, "max_dd": max_dd})
        print(f"  {label:<20}  {len(net):>4}  {net.mean():>+9.2f}  {cost_avg:>+10.2f}  "
              f"{sh:>+7.2f}  {lo:>+7.2f}  {hi:>+7.2f}  {max_dd:>+8.0f}")

    # Pick-stronger side breakdown
    print("\n  pick_stronger side selection (active cycles only):")
    df_ps = pd.DataFrame(cycles["pick_stronger"])
    active_ps = df_ps[df_ps["skipped"] == 0]
    if not active_ps.empty:
        side_counts = active_ps["side"].value_counts()
        print(f"    {side_counts.to_dict()}")
        for side in ["L", "S"]:
            sub = active_ps[active_ps["side"] == side]
            if len(sub) > 0:
                print(f"    side={side}: n={len(sub)}, mean_net={sub['net'].mean():+.2f}, "
                      f"sharpe={_sharpe(sub['net'].to_numpy()):+.2f}")

    pd.DataFrame(rows).to_csv(OUT_DIR / "summary.csv", index=False)
    for label, c in cycles.items():
        if c: pd.DataFrame(c).to_csv(OUT_DIR / f"{label}_cycles.csv", index=False)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
