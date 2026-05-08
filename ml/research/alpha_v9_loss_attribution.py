"""Diagnostic: why is the strategy losing money in April 2026?

For the bad recent fold (fold 9: 2026-04-01 → 2026-04-30, Sharpe -2.83),
decompose the loss into:

  (1) Per-symbol prediction quality (rank IC by symbol)
  (2) Per-symbol position-level P&L contribution
  (3) Cycle-level P&L trajectory through the period
  (4) Market regime characteristics vs other folds (basket vol, dispersion,
      BTC dominance, autocorr regime)
  (5) Long-leg vs short-leg performance
  (6) Which symbols repeatedly lost money on long vs short side

Also compares with prior bad folds (3 & 6) to see if same mechanism or different.
"""
from __future__ import annotations
import sys
import time
import warnings
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
from ml.research.alpha_v8_h48_audit import build_wide_panel

HORIZON = 48
TOP_K = 7
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
GATE_PCTILE = 0.30
GATE_LOOKBACK = 252


def _bn_scales(top_g, bot_g):
    beta_L = top_g["beta_short_vs_bk"].mean()
    beta_S = bot_g["beta_short_vs_bk"].mean()
    if beta_L < 0.1 or beta_S < 0.1 or (beta_L + beta_S) < 0.3:
        return 1.0, 1.0
    denom = beta_L + beta_S
    return (float(np.clip(2.0 * beta_S / denom, 0.5, 1.5)),
            float(np.clip(2.0 * beta_L / denom, 0.5, 1.5)))


def evaluate_fold_with_attribution(test, yt_pred, top_k=TOP_K):
    """Run conv_gate p=0.30 portfolio and return:
       - cycle-level P&L (gross, cost, net)
       - per-symbol per-cycle: was-long, was-short, realized return, alpha
    """
    cols = ["open_time", "symbol", "return_pct", "alpha_realized",
            "basket_fwd", "beta_short_vs_bk"]
    df = test[cols].copy()
    df["pred"] = yt_pred
    times = sorted(df["open_time"].unique())
    keep_times = set(times[::HORIZON])
    df = df[df["open_time"].isin(keep_times)]

    cycle_records = []
    position_records = []
    prev_long_w: dict = {}
    prev_short_w: dict = {}
    conv_history = deque(maxlen=GATE_LOOKBACK)

    for t, g in df.groupby("open_time"):
        if len(g) < 2 * top_k + 1:
            continue
        sorted_g = g.sort_values("pred")
        bot = sorted_g.head(top_k)
        top = sorted_g.tail(top_k)
        dispersion = top["pred"].mean() - bot["pred"].mean()
        skip = False
        if len(conv_history) >= 30:
            thr = np.quantile(list(conv_history), GATE_PCTILE)
            if dispersion < thr:
                skip = True
        conv_history.append(dispersion)

        rank_ic = sorted_g["pred"].rank().corr(sorted_g["alpha_realized"].rank())

        if skip:
            cycle_records.append({"time": t, "skipped": 1, "spread_ret_bps": 0,
                                    "cost_bps": 0, "net_bps": 0, "rank_ic": rank_ic,
                                    "dispersion": dispersion})
            continue

        scale_L, scale_S = _bn_scales(top, bot)
        n_l, n_s = len(top), len(bot)
        long_w = {s: scale_L / n_l for s in top["symbol"]}
        short_w = {s: scale_S / n_s for s in bot["symbol"]}
        long_ret = scale_L * top["return_pct"].mean()
        short_ret = scale_S * bot["return_pct"].mean()
        spread_ret = long_ret - short_ret

        # Per-symbol contribution
        for _, r in top.iterrows():
            position_records.append({
                "time": t, "symbol": r["symbol"], "side": "long",
                "pred": r["pred"], "return_pct": r["return_pct"],
                "alpha_realized": r["alpha_realized"],
                "weight": scale_L / n_l,
                "contribution_bps": (scale_L / n_l) * r["return_pct"] * 1e4,
            })
        for _, r in bot.iterrows():
            position_records.append({
                "time": t, "symbol": r["symbol"], "side": "short",
                "pred": r["pred"], "return_pct": r["return_pct"],
                "alpha_realized": r["alpha_realized"],
                "weight": -(scale_S / n_s),  # short is negative
                "contribution_bps": -(scale_S / n_s) * r["return_pct"] * 1e4,
            })

        # Turnover-aware cost
        if not prev_long_w:
            long_to, short_to = scale_L, scale_S
        else:
            all_l = set(long_w) | set(prev_long_w)
            long_to = sum(abs(long_w.get(s, 0) - prev_long_w.get(s, 0)) for s in all_l)
            all_s = set(short_w) | set(prev_short_w)
            short_to = sum(abs(short_w.get(s, 0) - prev_short_w.get(s, 0)) for s in all_s)
        cost_bps = COST_PER_LEG * (long_to + short_to)
        net_bps = (spread_ret * 1e4) - cost_bps

        cycle_records.append({
            "time": t, "skipped": 0, "spread_ret_bps": spread_ret * 1e4,
            "long_ret_bps": long_ret * 1e4, "short_ret_bps": short_ret * 1e4,
            "cost_bps": cost_bps, "net_bps": net_bps, "rank_ic": rank_ic,
            "dispersion": dispersion, "scale_L": scale_L, "scale_S": scale_S,
        })
        prev_long_w, prev_short_w = long_w, short_w

    return pd.DataFrame(cycle_records), pd.DataFrame(position_records)


def regime_chars(panel: pd.DataFrame, fold_test_dates):
    """Compute regime characteristics for a fold's test period."""
    test_start, test_end = fold_test_dates
    sub = panel[(panel["open_time"] >= test_start) & (panel["open_time"] < test_end)]
    if sub.empty:
        return None
    # Per bar: average across universe of various features
    bar_grp = sub.groupby("open_time")
    autocorr = bar_grp["autocorr_pctile_7d"].mean()
    return {
        "test_start": test_start,
        "test_end": test_end,
        "n_bars": sub["open_time"].nunique(),
        # Universe-mean autocorr regime (from feature)
        "autocorr_mean": float(autocorr.mean()),
        "autocorr_p25": float(autocorr.quantile(0.25)),
    }


def main():
    panel = build_wide_panel()
    folds = _multi_oos_splits(panel)
    print(f"Multi-OOS folds: {len(folds)}")

    target_folds = [3, 6, 9]  # the bad folds
    good_folds = [2, 4, 7]    # for comparison

    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    fold_data = {}
    for fold in folds:
        if fold["fid"] not in target_folds + good_folds:
            continue
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200:
            continue
        avail = [c for c in v6_clean if c in panel.columns]
        Xt = tr[avail].to_numpy(dtype=np.float32)
        yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)
        Xc = ca[avail].to_numpy(dtype=np.float32)
        yc_ = ca["demeaned_target"].to_numpy(dtype=np.float32)
        Xtest = test[avail].to_numpy(dtype=np.float32)
        models = [_train(Xt, yt_, Xc, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
        yt_pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                            for m in models], axis=0)
        cycles_df, positions_df = evaluate_fold_with_attribution(test, yt_pred)
        fold_data[fold["fid"]] = {
            "cycles": cycles_df, "positions": positions_df,
            "test_start": fold["test_start"], "test_end": fold["test_end"],
        }
        print(f"  fold {fold['fid']} ({fold['test_start'].date()}): "
              f"{time.time()-t0:.0f}s, net {cycles_df['net_bps'].mean():+.2f} bps")

    # ===================================================================
    # ANALYSIS
    # ===================================================================
    print("\n" + "=" * 110)
    print("LOSS ATTRIBUTION ACROSS BAD vs GOOD FOLDS")
    print("=" * 110)

    print(f"\n--- (1) PER-FOLD HIGH-LEVEL DECOMPOSITION ---")
    print(f"  {'fold':>5} {'period':>30} {'rank_IC':>9} {'spread':>8} {'long_ret':>9} "
          f"{'short_ret':>10} {'cost':>6} {'net':>7}")
    for fid in sorted(fold_data.keys()):
        d = fold_data[fid]
        c = d["cycles"]
        traded = c[c["skipped"] == 0]
        period = f"{d['test_start'].date()} → {d['test_end'].date()}"
        ic = c["rank_ic"].mean()
        print(f"  {fid:>5d} {period:>30} {ic:>+8.4f}  "
              f"{traded['spread_ret_bps'].mean() if len(traded) > 0 else 0:>+7.2f}  "
              f"{traded['long_ret_bps'].mean() if len(traded) > 0 else 0:>+8.2f}  "
              f"{traded['short_ret_bps'].mean() if len(traded) > 0 else 0:>+9.2f}  "
              f"{traded['cost_bps'].mean() if len(traded) > 0 else 0:>5.2f}  "
              f"{c['net_bps'].mean():>+6.2f}")

    print(f"\n--- (2) PER-SYMBOL P&L CONTRIBUTION (bad fold 9 = April 2026) ---")
    if 9 in fold_data:
        pos9 = fold_data[9]["positions"]
        sym9 = pos9.groupby("symbol").agg(
            n_long=("side", lambda x: (x == "long").sum()),
            n_short=("side", lambda x: (x == "short").sum()),
            mean_pred=("pred", "mean"),
            mean_realized=("return_pct", lambda x: x.mean() * 1e4),
            mean_alpha=("alpha_realized", lambda x: x.mean() * 1e4),
            total_contrib_bps=("contribution_bps", "sum"),
            mean_contrib_bps=("contribution_bps", "mean"),
        ).sort_values("total_contrib_bps")
        print(f"  {'symbol':>10} {'n_L':>4} {'n_S':>4} {'pred_avg':>9} "
              f"{'realized':>9} {'alpha_b':>9} {'total_bps':>10} {'mean_bps':>10}")
        for s in sym9.index:
            r = sym9.loc[s]
            print(f"  {s:>10} {int(r['n_long']):>4d} {int(r['n_short']):>4d} "
                  f"{r['mean_pred']:>+8.4f} {r['mean_realized']:>+8.2f} "
                  f"{r['mean_alpha']:>+8.2f} {r['total_contrib_bps']:>+9.1f} "
                  f"{r['mean_contrib_bps']:>+9.2f}")
        print(f"\n  Worst long contributors (bot 5):")
        long9 = pos9[pos9["side"] == "long"]
        long_by_sym = long9.groupby("symbol")["contribution_bps"].sum().sort_values()
        for s, v in long_by_sym.head(5).items():
            n = (pos9["symbol"] == s).sum()
            print(f"    {s}: {v:+.1f} bps total over {(long9['symbol']==s).sum()} cycles long")
        print(f"\n  Worst short contributors (bot 5, more negative = bigger loss):")
        short9 = pos9[pos9["side"] == "short"]
        short_by_sym = short9.groupby("symbol")["contribution_bps"].sum().sort_values()
        for s, v in short_by_sym.head(5).items():
            print(f"    {s}: {v:+.1f} bps total over {(short9['symbol']==s).sum()} cycles short")

    print(f"\n--- (3) CYCLE-LEVEL P&L TRAJECTORY (fold 9) ---")
    if 9 in fold_data:
        c9 = fold_data[9]["cycles"].copy()
        c9["time"] = pd.to_datetime(c9["time"])
        c9["cum_net"] = c9["net_bps"].cumsum()
        c9 = c9.sort_values("time").reset_index(drop=True)
        # Sample evenly through the fold for display
        n = len(c9)
        for i in range(0, n, max(1, n // 20)):
            r = c9.iloc[i]
            print(f"    {r['time']}   net {r['net_bps']:+8.2f}   cum {r['cum_net']:+8.0f}   "
                  f"IC {r['rank_ic']:+.3f}   skipped={int(r['skipped'])}")

    print(f"\n--- (4) PER-SYMBOL RANK IC (fold 9 vs good fold 7) ---")
    print(f"  How well did the model predict each symbol's alpha-rank in fold 9?")
    if 9 in fold_data and 7 in fold_data:
        for fid_label, fid in [("BAD fold 9 (April 2026)", 9), ("GOOD fold 7 (Feb 2026)", 7)]:
            pos = fold_data[fid]["positions"]
            # IC: per symbol, correlation between pred and alpha_realized
            sym_ic = pos.groupby("symbol").apply(
                lambda g: g["pred"].corr(g["alpha_realized"]) if len(g) >= 5 else np.nan
            )
            sym_ic = sym_ic.sort_values()
            print(f"\n  {fid_label}:")
            print(f"    worst-IC symbols: {dict([(s, f'{v:+.3f}') for s, v in sym_ic.head(5).items()])}")
            print(f"    best-IC symbols:  {dict([(s, f'{v:+.3f}') for s, v in sym_ic.tail(5).items()])}")
            print(f"    mean per-symbol IC: {sym_ic.mean():+.4f}")

    print(f"\n--- (5) REGIME CHARACTERISTICS (bad vs good folds) ---")
    print(f"  {'fold':>5} {'period':>30} {'autocorr_mean':>14} {'autocorr_p25':>13}")
    for fid in sorted(fold_data.keys()):
        d = fold_data[fid]
        rg = regime_chars(panel, (d["test_start"], d["test_end"]))
        if rg:
            period = f"{d['test_start'].date()} → {d['test_end'].date()}"
            label = "BAD" if fid in target_folds else "GOOD"
            print(f"  {fid:>3d}{label:>2} {period:>30} {rg['autocorr_mean']:>+13.4f}  "
                  f"{rg['autocorr_p25']:>+12.4f}")

    print(f"\n--- (6) LONG vs SHORT LEG BREAKDOWN ---")
    for fid in [9, 7, 3, 6]:
        if fid not in fold_data:
            continue
        c = fold_data[fid]["cycles"]
        traded = c[c["skipped"] == 0]
        if len(traded) == 0:
            continue
        long_pnl = traded["long_ret_bps"].mean()
        short_leg_return = traded["short_ret_bps"].mean()  # mean return of bot-K names
        # Long leg P&L = +long_ret; short leg P&L = -short_leg_return (we shorted)
        # spread = long_ret - short_ret. spread > 0 means long > short ⇒ profitable.
        period = f"{fold_data[fid]['test_start'].date()} → {fold_data[fid]['test_end'].date()}"
        period_label = " [BAD]" if fid in target_folds else " [GOOD]"
        print(f"  fold {fid:>2d}{period_label:>7} ({period}): "
              f"long_ret +{long_pnl:>+6.2f} bps, short_leg_ret {short_leg_return:>+6.2f}, "
              f"spread {long_pnl - short_leg_return:>+6.2f} bps")

    out = Path("/home/yuqing/ctaNew/outputs/h48_loss_attribution")
    out.mkdir(parents=True, exist_ok=True)
    if 9 in fold_data:
        fold_data[9]["cycles"].to_csv(out / "fold9_cycles.csv", index=False)
        fold_data[9]["positions"].to_csv(out / "fold9_positions.csv", index=False)
    print(f"\n  saved → {out}")


if __name__ == "__main__":
    main()
