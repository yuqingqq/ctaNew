"""Review diagnostic D-A (2026-11-10, strategy review). Descriptive only.

Question: the touch trades where our live order rested at the trade price but was not filled ("queue" status, 2,189 trades on
the consumed 2026-11-08 cohort, cut 23:00Z) show a favourable maker markout (+0.71 c/sh at 1 s, +1.74 at 5 s, from the trade
price). Did our gate have that information BEFORE the trade, or was our order placed about when / after the trade matched
(the time of a trade here is the live receipt of the WS pending message - 50 ms, which lags the match by an unknown amount)?

Input: derived cohort outputs only (queue_misses.parquet, queue_misses.json, touch_trade_reasons.parquet/.json). No raw capture,
no protected day. One partial consumed UTC day -> point estimates, no interval, no inferential claim (G < 5).

Reading, DECLARED BEFORE THE TABLE WAS SEEN (age = our current order's age at pending receipt - 50 ms; weights = trade shares):
  young = age < 200 ms, old = age >= 500 ms, base = all 0-60 s touch flow at 1 s (recomputed below from the reasons file).
  PRE-TRADE  : mo1(old) > +0.20 c/sh AND mo1(old) >= 0.5 * mo1(young)  -> the gate picks favourable trades ahead of time;
               what is missing is queue priority.
  REACTIVE   : mo1(old) <= +0.20 c/sh                                  -> the favourable flow belongs to orders placed around
               the trade; resting longer under the same gate would not earn it.
  MIXED      : anything else.
The script evaluates the predicate; it prints no verdict of its own choosing.

Self-test (falsifier): the parquet must reproduce queue_misses.json (count, share-weighted mo5) and the reasons file's queue
row (mo1, mo5) to 0.01; a copy with 10% of rows dropped must FAIL the same check.
Usage: predictfun/research/run_light.sh review_queue_miss_age -- python3 predictfun/research/pf_review_queue_miss_age.py <out.json>
"""
import json
import sys

import numpy as np
import pandas as pd

OUT = '/home/yuqing/pmmm-data/review-exports/live-samewindow-20261108/live_vs_replay'
AGE_BINS = [-np.inf, 50, 100, 200, 500, 1000, 2000, np.inf]
AGE_LAB = ['<50', '50-100', '100-200', '200-500', '500-1000', '1000-2000', '>=2000']


def wmean(d, col, w='trade_shares'):
    ok = d[col].notna()
    return float((d.loc[ok, w] * d.loc[ok, col]).sum() / d.loc[ok, w].sum()) if ok.any() and d.loc[ok, w].sum() > 0 else float('nan')


def reconcile(X, ref_q, ref_r):
    """Returns the list of mismatches (empty = pass)."""
    bad = []
    if len(X) != ref_q['queue_misses']:
        bad.append(f"count {len(X)} != {ref_q['queue_misses']}")
    if abs(round(wmean(X, 'mo5'), 2) - ref_q['mo5_c_sh']) > 0.011:
        bad.append(f"mo5 {wmean(X, 'mo5'):.3f} != {ref_q['mo5_c_sh']}")
    if abs(round(wmean(X, 'mo1'), 2) - ref_r['mo1_c']) > 0.011:
        bad.append(f"mo1 {wmean(X, 'mo1'):.3f} != {ref_r['mo1_c']}")
    return bad


def table(X, key, labels=None):
    rows = []
    tot = X.trade_shares.sum()
    for k, d in X.groupby(key, observed=True):
        rows.append(dict(bucket=str(k), trades=len(d), shares=round(float(d.trade_shares.sum()), 0), share_pct=round(100 * d.trade_shares.sum() / tot, 1),
                         mo1_c=round(wmean(d, 'mo1'), 2), mo5_c=round(wmean(d, 'mo5'), 2), settle_c=round(wmean(d, 'settle'), 2),
                         mo1_na=int(d.mo1.isna().sum()), mo5_na=int(d.mo5.isna().sum()),
                         reached_pct=round(100 * float(d.reached.mean()), 1), q_ahead_p50=round(float(d.q_ahead.median()), 0),
                         traded_since_p50=round(float(d.traded_since.median()), 0), windows=int(d.mid.nunique())))
    return pd.DataFrame(rows)


def main():
    X = pd.read_parquet(f'{OUT}/queue_misses.parquet')
    ref_q = json.load(open(f'{OUT}/queue_misses.json'))
    RJ = json.load(open(f'{OUT}/touch_trade_reasons.json'))
    ref_r = RJ['by_reason']['resting_not_filled (queue)']
    print('columns:', list(X.columns)); print('rows:', len(X), 'shares:', round(float(X.trade_shares.sum()), 1), 'windows:', X.mid.nunique())
    # --- self-test: positive control and known-bad input
    bad = reconcile(X, ref_q, ref_r)
    Xbad = X.sort_values('mo5').iloc[: int(len(X) * 0.9)]
    bad2 = reconcile(Xbad, ref_q, ref_r)
    print('reconcile(real):', bad or 'PASS'); print('reconcile(10% dropped) must fail:', 'FAILS as required' if bad2 else 'DID NOT FAIL')
    if bad or not bad2:
        print('SELFTEST FAILED -> no table'); return 1
    print('status counts: mo1 NaN', int(X.mo1.isna().sum()), '| mo5 NaN', int(X.mo5.isna().sum()), '| settle NaN', int(X.settle.isna().sum()),
          '| age NaN', int(X.age_ms.isna().sum()), '| age < 0', int((X.age_ms < 0).sum()))
    X['age_b'] = pd.cut(X.age_ms, AGE_BINS, labels=AGE_LAB, right=False)
    X['run_b'] = pd.cut(X.run_age_ms, AGE_BINS, labels=AGE_LAB, right=False)
    X['kind'] = np.where(X.lane % 4 < 2, 'touch lane', 'rung lane')
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print('\n== by our current order age at the trade (ms) =='); print(table(X, 'age_b').to_string(index=False))
    print('\n== by age of our unbroken run at the price (ms) =='); print(table(X, 'run_b').to_string(index=False))
    X['age3'] = pd.cut(X.age_ms, [-np.inf, 200, 500, np.inf], labels=['young <200', 'mid 200-500', 'old >=500'], right=False)
    X['run3'] = pd.cut(X.run_age_ms, [-np.inf, 200, 500, np.inf], labels=['run <200', 'run 200-500', 'run >=500'], right=False)
    print('\n== age x run age =='); print(table(X, ['age3', 'run3']).to_string(index=False))
    print('\n== by lane kind =='); print(table(X, 'kind').to_string(index=False))
    print('\n== by run ==');  print(table(X, 'run').to_string(index=False))
    # --- baseline: all 0-60 s touch flow, recomputed from the reasons file
    R = pd.read_parquet(f'{OUT}/touch_trade_reasons.parquet'); V = R[R.status != 'untimed']
    b60 = V[V.t_in < 60]; base1 = wmean(b60, 'mo1', 'shares'); base5 = wmean(b60, 'mo5', 'shares')
    qm = V[V.status == 'resting_not_filled (queue)']
    print(f"\nbaseline all 0-60 s touch flow: mo1 {base1:+.3f} mo5 {base5:+.3f} c/sh (shares {b60.shares.sum():.0f}); "
          f"queue rows in reasons file {len(qm)}, of which t_in < 60: {int((qm.t_in < 60).sum())}")
    # --- declared predicate
    y = X[X.age_ms < 200]; o = X[X.age_ms >= 500]
    my, mo_ = wmean(y, 'mo1'), wmean(o, 'mo1'); my5, mo5_ = wmean(y, 'mo5'), wmean(o, 'mo5')
    pre = (mo_ > 0.20) and (mo_ >= 0.5 * my); rea = mo_ <= 0.20
    verdict = 'PRE-TRADE' if pre else ('REACTIVE' if rea else 'MIXED')
    print(f"\npredicate: young(<200 ms) n={len(y)} sh={y.trade_shares.sum():.0f} mo1={my:+.3f} mo5={my5:+.3f} | "
          f"old(>=500 ms) n={len(o)} sh={o.trade_shares.sum():.0f} mo1={mo_:+.3f} mo5={mo5_:+.3f} | old/young mo1 = {mo_ / my if my else float('nan'):.2f}")
    print(f"predicate old>0.20: {mo_ > 0.20}; old>=0.5*young: {mo_ >= 0.5 * my}; reading = {verdict}")
    # how concentrated is the old bucket (day/window clustering): top windows' share of old shares and leave-top-5-windows-out mo1
    if len(o):
        ws = o.groupby('mid').trade_shares.sum().sort_values(ascending=False)
        top5 = set(ws.index[:5])
        print(f"old bucket: {o.mid.nunique()} windows; top-5 windows hold {100 * ws.iloc[:5].sum() / ws.sum():.1f}% of its shares; "
              f"mo1 without them {wmean(o[~o.mid.isin(top5)], 'mo1'):+.3f}; unweighted mean mo1 {o.mo1.mean():+.3f}, median {o.mo1.median():+.3f}; "
              f"share of trades with mo1 > 0: {100 * (o.mo1 > 0).mean():.1f}%")
    if len(y):
        print(f"young bucket: {y.mid.nunique()} windows; unweighted mean mo1 {y.mo1.mean():+.3f}, median {y.mo1.median():+.3f}; "
              f"share of trades with mo1 > 0: {100 * (y.mo1 > 0).mean():.1f}%")
    json.dump(dict(young=dict(n=len(y), shares=float(y.trade_shares.sum()), mo1=my, mo5=my5), old=dict(n=len(o), shares=float(o.trade_shares.sum()), mo1=mo_, mo5=mo5_),
                   base_0_60=dict(mo1=base1, mo5=base5), reading=verdict,
                   by_age=table(X, 'age_b').to_dict('records'), by_run_age=table(X, 'run_b').to_dict('records')),
              open(sys.argv[1], 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
