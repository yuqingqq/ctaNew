"""Reliability review report (declared in STATUS.yml 2026-11-04 13:41Z). Candidates at pull 60%, S60, V0 withhold: C1 A1:5:25:0,
C2 A1:20:100:0, C3 AV:20:100:0.06. Variants: orig (gz / gzd runs, LODO guard preds_g60), start (preds_g60k: both sides withheld until
the start price is known live), start+XV (+ the venue request throttle), wf (preds_w60k: walk-forward guard + start + XV; 10-27,
10-31, 11-01, 11-02). Money = mark@stop + rebate (UNVERIFIED) and settle + rebate, $ / day; +1 s c/sh without rebate. Predicates
R_START, R_WF, R_DAYS, R_TOP5, R_TAIL as restated 14:02Z: money WITHOUT the rebate (the rebate is a separate line); final variant = tx
(one time per taker transaction, PF_TXTIME=ws; + start + XV); txL = tx with cancel 34 ms / fills known 287 ms; txm = PF_TXTIME=min
(top 5's +1 s c/sh in 0-60 s from report_size_stop.json). Requests per quoting minute from
the requests files (places + cancels inside [ws, ws + 60 s]).
Usage: python3 -m predictfun.research.pf_review_report
"""
import json
import numpy as np, pandas as pd
from . import pf_io as io

D8 = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']; D4 = ['20261027', '20261031', '20261101', '20261102']
S4 = f'{io.OUT}/step4'; B = 'MICRO+G4+G2+G6+G5+G3+G1'
CAND = dict(C1='A1:5:25:0', C2='A1:20:100:0', C3='AV:20:100:0.06')
stem = lambda n: n.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')
arm = lambda sz, xv='': f'{B}+V0+XNA+XMIN1{xv}+S60/mid@{sz}'


def src(var, c):
    sz = CAND[c]
    if var == 'orig':
        return (f"{S4}/live_tokyoreal_custom_e2_preds_g60_{'gzd' if c == 'C3' else 'gz'}", arm(sz))
    if var == 'start':
        return (f'{S4}/live_tokyoreal_custom_e2_preds_g60k_rv', arm(sz))
    if var == 'start+XV':
        return (f'{S4}/live_tokyoreal_custom_e2_preds_g60k_rv', arm(sz, '+XV'))
    if var in ('tx', 'txL', 'txm', 'txQ'):
        return (f"{S4}/live_tokyoreal_custom_e2_preds_g60k_rv{var}", arm(sz, '+XV'))
    return (f'{S4}/live_tokyoreal_custom_e2_preds_w60k_rv', arm(sz, '+XV'))


def load(var, c, days):
    base, a = src(var, c); mn, sn, md, sd, q, mo1, bad_q, badv, goodv, rq, thr = [], [], [], [], 0.0, 0.0, 0.0, 0.0, 0.0, [], 0
    for d in days:
        s = json.load(open(f'{base}/{d}/summary.json'))['arms'][a]; rb = s['rebate_usd_unverified']
        md.append(s['mtm_stop_usd'] + rb); sd.append(s['gross_usd'] + rb); mn.append(s['mtm_stop_usd']); sn.append(s['gross_usd']); thr += int(s['live'].get('counters', {}).get('throttled', 0) or 0)
        f = pd.read_parquet(f'{base}/{d}/fills/{stem(a)}.parquet', columns=['q', 'mo1'])
        cc = 100 * f.mo1 / f.q; q += f.q.sum(); mo1 += f.mo1.sum(); bad_q += f.q[cc <= -2].sum(); badv += -f.mo1[cc <= -2].sum(); goodv += f.mo1[cc >= 0].sum()
        R = pd.read_parquet(f'{base}/{d}/requests/{stem(a)}.parquet', columns=['t_ms', 'mid', 'kind'])
        W = pd.read_parquet(f'{base}/{d}/win/{stem(a)}.parquet', columns=['mid', 'ws'])
        R = R.merge(W, on='mid', how='left'); R = R[(R.t_ms >= R.ws) & (R.t_ms < R.ws + 60_000)]
        n = R.groupby('mid').size().reindex(W.mid.values, fill_value=0).values; rq.append(n)
    rq = np.concatenate(rq)
    return dict(days=days, mark_day=mn, settle_day=sn, mark_reb_day=md, settle_reb_day=sd, mark_reb=float(np.mean(md)), shares_day=q / len(days), mo1_c=100 * mo1 / q, bad=bad_q / q, ratio=badv / goodv,
                mark=float(np.mean(mn)), settle=float(np.mean(sn)), req_min_mean=float(rq.mean()), req_min_p99=float(np.quantile(rq, 0.99)), throttled=thr)


def ci(dd):
    dd = np.asarray(dd, float); se = dd.std(ddof=1) / np.sqrt(len(dd))
    return dict(mean=float(dd.mean()), lo=float(dd.mean() - 2.365 * se), hi=float(dd.mean() + 2.365 * se), days_better=int((dd > 0).sum()), n=len(dd))


top5 = json.load(open(f'{S4}/report_size_stop.json'))['market']['top 5|0']['mo1_c']
out, pred = {}, {}
print(f'pull 60%, S60, V0 withhold; $ / day WITHOUT rebate: mark@stop | settle (rebate, UNVERIFIED: + column); top 5 +1 s in 0-60 s: {top5:+.3f} c/sh')
print(f"{'':22s} {'sh/day':>7s} {'+1s c/sh':>8s} {'bad':>6s} {'ratio':>6s} {'mark $/d':>9s} {'settle $/d':>10s} {'+reb mark':>9s} {'days>0':>6s} {'req/min':>8s} {'p99':>5s} {'throttled':>9s}")
for c in CAND:
    for var, days in (('orig', D8), ('start', D8), ('start+XV', D8), ('tx', D8), ('txL', D8), ('txm', D8), ('txQ', D8),
                      ('orig', ['20261103']), ('start+XV', ['20261103']), ('tx', ['20261103']), ('txL', ['20261103'])):
        r = load(var, c, days); out[f'{c}|{var}|{len(days)}'] = r
        lab = f"{c} {var}{' 11-03' if len(days) == 1 else ''}"
        print(f"{lab:22s} {r['shares_day']:7.0f} {r['mo1_c']:+8.3f} {r['bad']:6.1%} {r['ratio']:6.0%} {r['mark']:9.1f} {r['settle']:10.1f} {r['mark_reb']:9.1f} "
              f"{sum(x > 0 for x in r['mark_day'])}/{len(days):<4d} {r['req_min_mean']:8.1f} {r['req_min_p99']:5.0f} {r['throttled']:9d}")
    o, s_, x_, f_, fl = out[f'{c}|orig|8'], out[f'{c}|start|8'], out[f'{c}|start+XV|8'], out[f'{c}|tx|8'], out[f'{c}|txL|8']
    pred[f'{c}_R_START'] = bool(sum(s_['mark_day']) >= 0.7 * sum(o['mark_day']))
    js = [np.mean([x for j, x in enumerate(f_['mark_day']) if j != i]) for i in range(8)]
    pred[f'{c}_R_DAYS'] = bool(sum(x > 0 for x in f_['mark_day']) >= 6 and min(js) > 0)
    pred[f'{c}_R_TOP5'] = bool(f_['mo1_c'] >= top5)
    pred[f'{c}_R_TAIL'] = bool(sum(fl['mark_day']) > 0)
    pred[f'{c}_R_QUEUE'] = bool(sum(out[f'{c}|txQ|8']['mark_day']) > 0)
    w = load('wf', c, D4); g = load('start+XV', c, D4); out[f'{c}|wf|4'] = w; out[f'{c}|start+XV|4'] = g
    pred[f'{c}_R_WF'] = bool(sum(w['mark_day']) >= 0.5 * sum(g['mark_day']))
    print(f"  start withhold vs orig: mark {s_['mark'] - o['mark']:+.1f} $/d ({ci(np.array(s_['mark_day']) - o['mark_day'])['days_better']}/8 better), kept "
          f"{sum(s_['mark_day']) / sum(o['mark_day']):.0%}; XV vs none: mark {x_['mark'] - s_['mark']:+.1f} $/d; tx timing vs per-leg: mark {f_['mark'] - x_['mark']:+.1f} $/d "
          f"(kept {sum(f_['mark_day']) / sum(x_['mark_day']):.0%}), +1 s {f_['mo1_c']:+.3f} vs {x_['mo1_c']:+.3f} c/sh; latency tails: mark {fl['mark']:+.1f} $/d")
    print(f"  walk-forward vs LODO (start+XV, 4 days): mark {w['mark']:.1f} vs {g['mark']:.1f} $/d (kept {sum(w['mark_day']) / sum(g['mark_day']):.0%}), "
          f"+1 s {w['mo1_c']:+.3f} vs {g['mo1_c']:+.3f} c/sh, bad {w['bad']:.1%} vs {g['bad']:.1%}; per day {[round(x) for x in w['mark_day']]} vs {[round(x) for x in g['mark_day']]}")
    print(f"  final (tx + start + XV) per day {[round(x) for x in f_['mark_day']]}; leave-one-day-out means min {min(js):.1f} max {max(js):.1f}")
    print(f"  predicates: R_START {pred[f'{c}_R_START']}, R_WF {pred[f'{c}_R_WF']}, R_DAYS {pred[f'{c}_R_DAYS']}, R_TOP5 {pred[f'{c}_R_TOP5']}, R_TAIL {pred[f'{c}_R_TAIL']}, R_QUEUE {pred[f'{c}_R_QUEUE']}")
for c in CAND:
    pred[f'{c}_ALL'] = all(pred[f'{c}_{k}'] for k in ('R_START', 'R_WF', 'R_DAYS', 'R_TOP5', 'R_TAIL', 'R_QUEUE'))
print('\nall six predicates pass:', {c: pred[f'{c}_ALL'] for c in CAND})
sk = json.load(open(f'{io.OUT}/step5/startknown_preds_g60_report.json'))
print('start price known (s after ws), by day p50 / p90 / max; windows from the WS patch / REST fallback / price mismatches:')
for d, v in sk.items():
    print(f"  {d} {v['known_lag_s']['p50']:.2f} / {v['known_lag_s']['p90']:.2f} / {v['known_lag_s']['max']:.1f}  {v['from_ws_patch']} / {v['from_rest_fallback']} / {v['price_mismatch']}  missing hours {v['missing_hours']}")
io.write_json(f'{S4}/report_review.json', dict(results=out, predicates=pred, top5_mo1_c=top5))
