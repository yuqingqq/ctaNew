"""Sharpe ratios instead of P&L for our chase entries and exits (USER 2026-11-27: "We check sharpe ratio instead of pure pnl"). Reads the
per-action P&L of EXIT_FOLLOW, EXIT_FOLLOW2 and EXIT_HALF (no new simulation; the shared policies are equal across them: a check). Four
consumed days; the selections, exits and their parameters were chosen on these days (post hoc for the later two); declared before the run.
Unit: the 5-minute market window, every window of the registry on the four days (no action = 0), P&L summed over the window's actions
(positions entered in a window settle or exit in it). Per cell (selection x entry book x entry limit x exit policy): total $, per-window
mean and standard deviation, Sharpe per window = mean / sd, a day equivalent = x sqrt(windows per day) (assumes independent windows: the
lag-1 autocorrelation of the window P&L in time order is reported), the daily Sharpe from the four daily totals (G = 4: shown, not to be
relied on), the maximum drawdown of the cumulative P&L in time order, the share of active windows. Rankings by Sharpe and by P&L per
selection x entry. Point estimates; no intervals (G = 4 < 5).
Output (write-once): sharpe_exits.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/sharpe_exits.py [selftest]
"""
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import exit_follow as E  # noqa: E402
import exit_follow2 as E2  # noqa: E402
import exit_half as EH  # noqa: E402

C, TR, MM = E.C, E.TR, E.MM
DAYS = E.DAYS


def window_series(pnl_usd, win_key, all_keys):
    """P&L per window over every window (0 where no action), in all_keys' order."""
    s = pd.Series(pnl_usd).groupby(win_key).sum()
    return s.reindex(all_keys, fill_value=0.).to_numpy(float)


def stats(w, day_of, per_day):
    mu, sd = float(w.mean()), float(w.std(ddof=1))
    cum = np.cumsum(w)
    dd = float(np.max(np.maximum.accumulate(np.r_[0., cum])[1:] - cum)) if len(w) else 0.
    daily = np.array([w[day_of == d].sum() for d in DAYS])
    ac = float(np.corrcoef(w[:-1], w[1:])[0, 1]) if sd > 0 else None
    return dict(total_usd=float(w.sum()), window_mean_usd=mu, window_sd_usd=sd, sharpe_window=mu / sd if sd > 0 else None,
                sharpe_day_equiv=mu / sd * np.sqrt(per_day) if sd > 0 else None, lag1_autocorr=ac,
                sharpe_daily_4days=float(daily.mean() / daily.std(ddof=1)) if daily.std(ddof=1) > 0 else None, daily_usd=[float(x) for x in daily],
                max_drawdown_usd=dd, active_windows=float(np.mean(w != 0.)))


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (600, 601))
    fn = H / 'sharpe_exits.json'
    C.require(not fn.exists(), 'sharpe_exits.json exists (write-once)')
    keys, dayk = [], []
    for d in DAYS:
        reg = MM.registry(d).sort_values('ws')
        keys += [f'{d}:{int(m)}' for m in reg.mid]
        dayk += [d] * len(reg)
    keys, dayk = np.array(keys), np.array(dayk)
    per_day = len(keys) / len(DAYS)
    srcs = {'v1': (E.OUT, E.POLICIES), 'v2': (E2.OUT, E2.POLICIES), 'half': (EH.OUT, EH.POLICIES)}
    T = {nm: pd.concat([pd.read_parquet(o / f'{d}.parquet') for d in DAYS], ignore_index=True) for nm, (o, _) in srcs.items()}
    for nm in ('v2', 'half'):
        C.require((T[nm].t.to_numpy() == T['v1'].t.to_numpy()).all(), f'{nm}: actions differ from EXIT_FOLLOW')
    for b, s in E.COMBOS:
        for pol in ('HOLD', 'TB1S', 'FAV_ONLY'):
            k = f'{b}_s{int(s)}_{pol}_pnl'
            for nm in ('v2', 'half'):
                C.require(np.allclose(T[nm][k].to_numpy(float), T['v1'][k].to_numpy(float), rtol=0, atol=1e-9), f'{nm} {k} differs from EXIT_FOLLOW')
    X = T['v1']
    wk = (X.day.astype(str) + ':' + X.mid.astype(np.int64).astype(str)).to_numpy()
    C.require(set(wk) <= set(keys), 'an action outside the registry windows')
    pols = {}
    for nm, (_, ps) in srcs.items():
        for p in ps:
            pols.setdefault(p, nm)
    out = dict(asof='2026-11-27', windows=int(len(keys)), windows_per_day=per_day, policies={p: nm for p, nm in pols.items()}, cells={}, rankings={})
    for sel in E.SELS:
        m = X[sel].to_numpy(bool)
        for b, s in E.COMBOS:
            pre = f'{b}_s{int(s)}'
            for p, nm in pols.items():
                pn = T[nm][f'{pre}_{p}_pnl'].to_numpy(float) / 100.
                w = window_series(pn[m], wk[m], keys)
                out['cells'][f'{sel}|{pre}|{p}'] = stats(w, dayk, per_day)
            rows = [(p, out['cells'][f'{sel}|{pre}|{p}']) for p in pols]
            by_sh = sorted(rows, key=lambda x: -(x[1]['sharpe_window'] if x[1]['sharpe_window'] is not None else -9))
            by_pnl = sorted(rows, key=lambda x: -x[1]['total_usd'])
            out['rankings'][f'{sel}|{pre}'] = dict(by_sharpe=[p for p, _ in by_sh], by_pnl=[p for p, _ in by_pnl])
    strat = [(k, c) for k, c in out['cells'].items() if not k.startswith('CONTROL') and '|AB_' in k and c['sharpe_window'] is not None]
    top = sorted(strat, key=lambda x: -x[1]['sharpe_window'])[:15]
    out['top_conservative_by_sharpe'] = [dict(cell=k, sharpe_day_equiv=c['sharpe_day_equiv'], sharpe_window=c['sharpe_window'], total_usd=c['total_usd'],
                                              max_drawdown_usd=c['max_drawdown_usd'], daily_usd=c['daily_usd'], lag1_autocorr=c['lag1_autocorr']) for k, c in top]
    TR.dump(fn, out)
    for r in out['top_conservative_by_sharpe']:
        print(r['cell'], 'SR/day', round(r['sharpe_day_equiv'], 2), 'total', round(r['total_usd']), 'maxDD', round(r['max_drawdown_usd']), 'daily', [round(x) for x in r['daily_usd']],
              'ac1', round(r['lag1_autocorr'], 3))


def selftest():
    keys = np.array(['a:1', 'a:2', 'b:1', 'b:2'])
    w = window_series(np.array([1., 2., -1.]), np.array(['a:1', 'a:1', 'b:2']), keys)
    assert w.tolist() == [3., 0., 0., -1.], w
    st = stats(np.array([1., -2., 3., 0.]), np.array(['20261023', '20261023', '20261031', '20261031']), 2.)
    assert abs(st['total_usd'] - 2.) < 1e-12 and abs(st['max_drawdown_usd'] - 2.) < 1e-12, st          # cum 1, -1, 2, 2: peak 1 -> -1
    assert abs(st['sharpe_window'] - .5 / np.std([1., -2., 3., 0.], ddof=1)) < 1e-12
    assert abs(st['sharpe_day_equiv'] - st['sharpe_window'] * np.sqrt(2.)) < 1e-12
    print('sharpe_exits selftest PASS')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'selftest':
        selftest()
    else:
        main()
