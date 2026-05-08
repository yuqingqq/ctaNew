"""Run extended multi-OOS backtest on panel including fresh data through 2026-05-06.

Production setup: h=48 K=7 ORIG25 + conv_gate p=0.30 + LGBM (63, 100, 3.0)
+ β-neutral. Same framework that produced +1.47 Sharpe on the 9-fold base.

Compares per-fold performance, with extra emphasis on the new fold(s)
that include data after 2026-03-30 (previously OOS-untested).
"""
from __future__ import annotations
import sys
import time
import warnings
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
from ml.research.alpha_v4_xs import block_bootstrap_ci
from ml.research.alpha_v8_h48_audit import build_wide_panel
from ml.research.alpha_v9_conviction_v2 import evaluate_portfolio

HORIZON = 48
TOP_K = 7
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
GATE_PCTILE = 0.30
sharpe_est = lambda x: x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0


def main():
    panel = build_wide_panel()
    print(f"Panel range: {panel['open_time'].min().date()} → {panel['open_time'].max().date()}")
    folds = _multi_oos_splits(panel)
    print(f"Multi-OOS folds: {len(folds)}")
    for f in folds:
        print(f"  fold {f['fid']}: test {f['test_start'].date()} → {f['test_end'].date()}")

    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    cycles = []
    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200:
            print(f"  fold {fold['fid']}: SKIPPED (insufficient data)")
            continue
        avail = [c for c in v6_clean if c in panel.columns]
        Xt = tr[avail].to_numpy(dtype=np.float32)
        yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)
        Xc = ca[avail].to_numpy(dtype=np.float32)
        yc_ = ca["demeaned_target"].to_numpy(dtype=np.float32)
        models = [_train(Xt, yt_, Xc, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
        Xtest = test[avail].to_numpy(dtype=np.float32)
        yt_pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                            for m in models], axis=0)
        df = evaluate_portfolio(test, yt_pred, use_gate=True, gate_pctile=GATE_PCTILE,
                                 use_magweight=False, top_k=TOP_K)
        for _, r in df.iterrows():
            cycles.append({
                "fold": fold["fid"], "time": r["time"],
                "gross": r["spread_ret_bps"], "cost": r["cost_bps"],
                "net": r["net_bps"], "skipped": r["skipped"],
                "long_turn": r["long_turnover"],
            })
        print(f"  fold {fold['fid']}: {time.time()-t0:.0f}s, "
              f"net {df['net_bps'].mean():+.2f} bps, "
              f"trades {(df['skipped']==0).sum()}")

    df = pd.DataFrame(cycles)
    df['time'] = pd.to_datetime(df['time'])

    print("\n" + "=" * 100)
    print("EXTENDED MULTI-OOS PERFORMANCE (h=48 K=7 ORIG25 + conv_gate p=0.30, post-fix cost)")
    print("=" * 100)
    print(f"{'fold':>5} {'period':>30} {'cycles':>7} {'%trade':>7} {'gross':>7} "
          f"{'cost':>6} {'net':>7} {'L_turn':>7} {'Sharpe':>7}")
    for fid in sorted(df['fold'].unique()):
        f = df[df['fold'] == fid]
        traded = f[f['skipped'] == 0]
        period = f"{f['time'].min().date()} → {f['time'].max().date()}"
        sh = sharpe_est(f['net'].to_numpy())
        print(f"{fid:>5d} {period:>30} {len(f):>7d} "
              f"{100*len(traded)/len(f):>6.1f}% "
              f"{traded['gross'].mean() if len(traded) > 0 else 0:>+6.2f}  "
              f"{traded['cost'].mean() if len(traded) > 0 else 0:>5.2f}  "
              f"{f['net'].mean():>+6.2f}  "
              f"{traded['long_turn'].mean() if len(traded) > 0 else 0:>6.0%}  "
              f"{sh:>+6.2f}")

    # Aggregate stats
    print(f"\n{'TOTAL ALL FOLDS':>35} {len(df):>7d} "
          f"{100*(df['skipped']==0).sum()/len(df):>6.1f}% "
          f"{df[df['skipped']==0]['gross'].mean():>+6.2f}  "
          f"{df[df['skipped']==0]['cost'].mean():>5.2f}  "
          f"{df['net'].mean():>+6.2f}  "
          f"{df[df['skipped']==0]['long_turn'].mean():>6.0%}  "
          f"{sharpe_est(df['net'].to_numpy()):>+6.2f}")

    # CIs and recent
    sh_full, lo_full, hi_full = block_bootstrap_ci(df['net'].values, statistic=sharpe_est,
                                                      block_size=7, n_boot=2000)
    print(f"  95% CI: [{lo_full:+.2f}, {hi_full:+.2f}]")

    # Recent (last 60 days)
    cutoff = df['time'].max() - pd.Timedelta(days=60)
    recent = df[df['time'] >= cutoff]
    if not recent.empty:
        recent_traded = recent[recent['skipped'] == 0]
        sh_r = sharpe_est(recent['net'].to_numpy())
        print(f"\n{'LAST 60 DAYS':>35} {len(recent):>7d} "
              f"{100*len(recent_traded)/len(recent):>6.1f}% "
              f"{recent_traded['gross'].mean() if len(recent_traded) > 0 else 0:>+6.2f}  "
              f"{recent_traded['cost'].mean() if len(recent_traded) > 0 else 0:>5.2f}  "
              f"{recent['net'].mean():>+6.2f}  "
              f"{recent_traded['long_turn'].mean() if len(recent_traded) > 0 else 0:>6.0%}  "
              f"{sh_r:>+6.2f}")
        print(f"  recent period: {recent['time'].min().date()} → {recent['time'].max().date()}")

    # Cumulative PnL chronological
    df_chrono = df.sort_values('time').reset_index(drop=True)
    df_chrono['cum_net'] = df_chrono['net'].cumsum()
    days_span = (df_chrono['time'].max() - df_chrono['time'].min()).total_seconds() / 86400
    print(f"\n=== CUMULATIVE NET P&L (chronological) ===")
    print(f"  total cycles: {len(df_chrono)}, span {days_span:.0f} days")
    print(f"  cumulative net: {df_chrono['cum_net'].iloc[-1]:+.0f} bps "
          f"({df_chrono['cum_net'].iloc[-1]/100:+.1f}%)")
    print(f"  annualized return: {df_chrono['cum_net'].iloc[-1] / days_span * 365 / 100:.1f}%")

    # Drawdown
    rmax = df_chrono['cum_net'].cummax()
    dd = df_chrono['cum_net'] - rmax
    print(f"  max drawdown: {dd.min():+.0f} bps ({dd.min()/100:+.1f}%)")
    worst = dd.idxmin()
    print(f"  worst DD bottom: {df_chrono.loc[worst, 'time'].date()}")

    # save the data
    out = Path("/home/yuqing/ctaNew/outputs/h48_recent_backtest")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "extended_cycles.csv", index=False)
    print(f"\n  saved → {out}")


if __name__ == "__main__":
    main()
