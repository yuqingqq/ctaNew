"""Cheap sides recover within 30 s (follow-up to TAKE_MARKOUT_EXIT, USER 2026-11-27: "1. We can get slippage to fill the taker order, 2. The
training target is the mark out price"): the 30 s markout model's paying picks are sides priced near 0 late in the window, and over all
moments a take of a side under 10c taken back 30 s later pays. Readout over TAKE_MARKOUT's and TAKE_MARKOUT_EXIT's labels; no new data,
no model. Found after looking (TAKE_MARKOUT_EXIT's price table): development evidence on four consumed days, not a test. Declared before
the run, every value per intended share of a 50-share order at the known best (TAKE_MODEL's walk at 172 ms, misses 0):
1 by the direction bought (Up = YES, Down = NO) x the price of the side bought (its mid at the decision; 0-10, 10-20, 20-35, 35-65, 65-80,
  80-90, 90-100 c) x exit (take back after 5 s / 30 s through the speed bump, TAKE_MARKOUT_EXIT's rule; hold to settlement; hold with
  the tie markets left out): moments, markets, value pooled and by day, the markets' share positive and the ten largest markets' share of
  the total.
2 calibration of the side bought: its entry price (filled moments, share-weighted) against how often it won (payout 1 = 100c, a tie 50c),
  over moments and over markets (each market once per side and bucket: did the side that sat in the bucket win).
3 sides under 10c by time left (0-60, 60-120, 120-200, 200-300 s): 30 s take back and hold.
Output (write-once): take_markout_longshot.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/take_markout_longshot.py [<dir>]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import take_markout as TK  # noqa: E402
import take_markout_exit as TE  # noqa: E402
import take_model_insample as TI  # noqa: E402

C, TR, Q, DAYS = TK.C, TK.TR, TK.Q, TK.DAYS
BINS = ((0, 10), (10, 20), (20, 35), (35, 65), (65, 80), (80, 90), (90, 100.01))
TAU = ((0, 60), (60, 120), (120, 200), (200, 300.01))


def lab(lo, hi, cap, unit=''):
    return f'{lo}-{min(hi, cap):g}{unit}'


def group(v, m, day, mid):
    """Per intended share over the moments m: value, by day, markets, markets positive, the ten largest markets' share of the total."""
    if not m.any():
        return dict(n=0)
    s = pd.Series(v[m] / Q).groupby(mid[m]).sum()
    tot = float(s.sum())
    big = s.reindex(s.abs().sort_values(ascending=False).index).iloc[:10].sum()
    return dict(n=int(m.sum()), markets=int(s.size), value_c=float(v[m].mean() / Q),
                by_day={dh: (float(v[m & (day == dh)].mean() / Q) if (m & (day == dh)).any() else None) for dh in DAYS},
                days_positive=int(sum(1 for dh in DAYS if (m & (day == dh)).any() and v[m & (day == dh)].mean() > 0)),
                markets_positive_share=float((s > 0).mean()), top10_markets_share_of_total=float(big / tot) if tot != 0 else None)


def main():
    fn = (Path(sys.argv[1]) if len(sys.argv) > 1 else H) / 'take_markout_longshot.json'
    C.require(not fn.exists(), 'take_markout_longshot.json exists (write-once)')
    X, _ = TI.frame()
    K = pd.concat([pd.read_parquet(TK.OUT / f'{d}_labels.parquet') for d in DAYS], ignore_index=True)
    E = pd.concat([pd.read_parquet(TE.OUT / f'{d}_labels.parquet') for d in DAYS], ignore_index=True)
    C.require(len(K) == len(E) == len(X), 'labels do not cover the population')
    X = X.merge(K.drop(columns=['day', 'mid', 't', 'd']), on='row', how='left', validate='1:1').merge(E.drop(columns=['day', 'mid', 'd']), on='row', how='left', validate='1:1')
    d, lim, pay = X.d.to_numpy(float), X.lim.to_numpy(float), X.payout.to_numpy(float)
    day, mid, ip, tau = X.day.to_numpy(), X.mid.to_numpy(), X.pf_its_price.to_numpy(float), X.p_tau.to_numpy(float)
    C.require(np.isfinite(pay).all() and np.isfinite(ip).all() and np.isfinite(tau).all(), 'a moment without its payout, price or time left')
    PA = X[[f'pA172_{k}' for k in range(5)]].to_numpy(float); SA = X[[f'sA172_{k}' for k in range(5)]].to_numpy(float)
    fl, co, fe = TK.vwalk(PA, SA, d, lim)
    tie = pay == .5
    ex = {}
    for hs in ('5s', '30s'):
        ok = np.isfinite(X[f'm{hs}_172'].to_numpy(float))
        eP = X[[f'eP{hs}_{k}' for k in range(5)]].to_numpy(float); eS = X[[f'eS{hs}_{k}' for k in range(5)]].to_numpy(float)
        back, hold, _ = TE.exit_values(fl, co, fe, d, eP, eS, pay)
        ex[hs] = (ok, back, hold)
    out = dict(asof='2026-11-27', rows=int(len(X)), tie_markets=int(np.unique(mid[tie]).size), markets=int(np.unique(mid).size), by_direction={}, calibration={}, by_time_left={})
    for dn, dm in (('up', d > 0), ('down', d < 0)):
        out['by_direction'][dn] = {}
        for lo, hi in BINS:
            b = dm & (ip >= lo) & (ip < hi)
            r = {}
            for hs, (ok, back, hold) in ex.items():
                r[f'take_back_{hs}'] = group(back, b & ok, day, mid)
            ok30, _, hold = ex['30s']
            r['hold_to_settlement'] = group(hold, b, day, mid)
            r['hold_without_ties'] = group(hold, b & ~tie, day, mid)
            out['by_direction'][dn][lab(lo, hi, 100)] = r
    # calibration of the side bought (both directions): price paid vs how often it won
    win = np.where(d > 0, pay, 1. - pay)                     # the side bought's payout share
    paid = np.where(fl > 0, np.where(d > 0, co / np.where(fl > 0, fl, 1.), 100. - co / np.where(fl > 0, fl, 1.)), np.nan)   # its own price
    for lo, hi in BINS:
        b = (ip >= lo) & (ip < hi)
        f_ = b & (fl > 0)
        mk = pd.DataFrame(dict(mid=mid[b], d=d[b], win=win[b])).drop_duplicates(['mid', 'd'])
        out['calibration'][lab(lo, hi, 100)] = dict(moments=int(b.sum()), filled=int(f_.sum()), its_price_paid_c=float(np.average(paid[f_], weights=fl[f_])) if f_.any() else None,
                                                won_c_moments=float(100. * np.average(win[f_], weights=fl[f_])) if f_.any() else None,
                                                market_sides=int(len(mk)), won_c_market_sides=float(100. * mk.win.mean()) if len(mk) else None)
    ok30, back30, hold = ex['30s']
    for lo, hi in TAU:
        b = (ip < 10) & (tau >= lo) & (tau < hi)
        out['by_time_left'][lab(lo, hi, 300, 's')] = dict(take_back_30s=group(back30, b & ok30, day, mid), hold_to_settlement=group(hold, b, day, mid))
    # predicates
    cheap = [out['by_direction'][dn]['0-10'][f'take_back_30s'] for dn in ('up', 'down')]
    out['predicates'] = dict(cheap_30s_take_back_positive_both_directions=bool(all(c['value_c'] > 0 for c in cheap)),
                             cheap_30s_take_back_days_positive=[c['days_positive'] for c in cheap],
                             cheap_won_more_than_paid=bool(out['calibration']['0-10']['won_c_moments'] > out['calibration']['0-10']['its_price_paid_c']),
                             dear_won_less_than_paid=bool(out['calibration']['90-100']['won_c_moments'] < out['calibration']['90-100']['its_price_paid_c']))
    TR.dump(fn, out)
    print(json.dumps(dict(calibration=out['calibration'], predicates=out['predicates']), indent=1))
    for dn in ('up', 'down'):
        for k, r in out['by_direction'][dn].items():
            print(dn, k, {e: (g.get('n'), g.get('markets'), round(g.get('value_c', np.nan), 2), g.get('days_positive')) for e, g in r.items()})


if __name__ == '__main__':
    main()
