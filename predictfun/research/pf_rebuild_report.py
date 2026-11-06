"""Rebuild report (declared in STATUS.yml 2026-11-04 14:45Z / 14:50Z). Guard refit on the corrected grid (preds_h60k: pending increments
known per frame, transaction-timed labels, start withhold); replays with one time per taker transaction + XV, S60, V0. Candidates
C1 A1:5:25:0, C2 A1:20:100:0, C3 AV:20:100:0.06, C4 AV:5:25:0.06; variants rbtx, rbtxQ (pessimistic queue at activation), rbtxS
(matches 25 ms before the frame). Money = mark@stop WITHOUT rebate ($ / day); rebate a separate column. Predicates per candidate:
P_DAYS = rbtx mark > 0 on >= 6/8 read days AND every leave-one-day-out mean > 0; P_QUEUE = rbtxQ 8-day sum > 0; P_SHIFT = rbtxS 8-day
sum > 0; P_TOP5 = rbtx +1 s c/sh >= the top 5's in 0-60 s. Beside: the pre-rebuild guard at the same settings (rvtx, C1-C3).
Usage: python3 -m predictfun.research.pf_rebuild_report
"""
import json
import numpy as np, pandas as pd
from . import pf_io as io

D8 = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']
S4 = f'{io.OUT}/step4'; B = 'MICRO+G4+G2+G6+G5+G3+G1'
CAND = dict(C1='A1:5:25:0', C2='A1:20:100:0', C3='AV:20:100:0.06', C4='AV:5:25:0.06')
VAR = dict(rbtx='preds_h60k_rbtx', rbtxQ='preds_h60k_rbtxQ', rbtxS='preds_h60k_rbtxS', pre='preds_g60k_rvtx')
stem = lambda n: n.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')
arm = lambda sz: f'{B}+V0+XNA+XMIN1+XV+S60/mid@{sz}'


def load(var, c, days):
    base = f'{S4}/live_tokyoreal_custom_e2_{VAR[var]}'; a = arm(CAND[c]); mk, se, rb, rq, thr = [], [], [], [], 0
    q = mo1 = bq = bv = gv = 0.0
    for d in days:
        s = json.load(open(f'{base}/{d}/summary.json'))['arms'][a]
        mk.append(s['mtm_stop_usd']); se.append(s['gross_usd']); rb.append(s['rebate_usd_unverified']); thr += int(s['live'].get('counters', {}).get('throttled', 0) or 0)
        f = pd.read_parquet(f'{base}/{d}/fills/{stem(a)}.parquet', columns=['q', 'mo1']); cc = 100 * f.mo1 / f.q
        q += f.q.sum(); mo1 += f.mo1.sum(); bq += f.q[cc <= -2].sum(); bv += -f.mo1[cc <= -2].sum(); gv += f.mo1[cc >= 0].sum()
        R = pd.read_parquet(f'{base}/{d}/requests/{stem(a)}.parquet', columns=['t_ms', 'mid']).merge(
            pd.read_parquet(f'{base}/{d}/win/{stem(a)}.parquet', columns=['mid', 'ws']), on='mid')
        rq.append(R[(R.t_ms >= R.ws) & (R.t_ms < R.ws + 60_000)].groupby('mid').size().values)
    rq = np.concatenate(rq) if rq else np.zeros(1)
    return dict(mark_day=mk, settle_day=se, rebate_day=rb, shares_day=q / len(days), mo1_c=100 * mo1 / q, bad=bq / q, ratio=bv / gv,
                mark=float(np.mean(mk)), settle=float(np.mean(se)), rebate=float(np.mean(rb)), req_p99=float(np.quantile(rq, .99)), req_mean=float(rq.mean()), throttled=thr)


top5 = json.load(open(f'{S4}/report_size_stop.json'))['market']['top 5|0']['mo1_c']
out, pred = {}, {}
print(f'$ / day WITHOUT rebate (rebate UNVERIFIED, own column); 8 read days unless marked; top 5 +1 s in 0-60 s {top5:+.3f} c/sh')
print(f"{'':20s} {'sh/day':>7s} {'+1s c/sh':>8s} {'bad':>6s} {'ratio':>6s} {'mark':>7s} {'settle':>7s} {'rebate':>7s} {'days>0':>6s} {'LOO min':>8s} {'req/min':>7s} {'p99':>5s} {'thr/day':>7s}")
for c in CAND:
    for var in ('pre', 'rbtx', 'rbtxQ', 'rbtxS'):
        if var == 'pre' and c == 'C4':
            continue
        for days in (D8, ['20261103']):
            r = load(var, c, days); out[f'{c}|{var}|{len(days)}'] = r
            loo = min(np.mean([x for j, x in enumerate(r['mark_day']) if j != i]) for i in range(len(days))) if len(days) > 1 else float('nan')
            r['loo_min'] = float(loo)
            lab = f"{c} {var}{' 11-03' if len(days) == 1 else ''}"
            print(f"{lab:20s} {r['shares_day']:7.0f} {r['mo1_c']:+8.3f} {r['bad']:6.1%} {r['ratio']:6.0%} {r['mark']:7.1f} {r['settle']:7.1f} {r['rebate']:7.1f} "
                  f"{sum(x > 0 for x in r['mark_day'])}/{len(days):<4d} {loo:8.1f} {r['req_mean']:7.0f} {r['req_p99']:5.0f} {r['throttled'] / len(days):7.0f}")
    t, tq, ts = out[f'{c}|rbtx|8'], out[f'{c}|rbtxQ|8'], out[f'{c}|rbtxS|8']
    pred[c] = dict(P_DAYS=bool(sum(x > 0 for x in t['mark_day']) >= 6 and t['loo_min'] > 0), P_QUEUE=bool(sum(tq['mark_day']) > 0),
                   P_SHIFT=bool(sum(ts['mark_day']) > 0), P_TOP5=bool(t['mo1_c'] >= top5))
    pred[c]['ALL'] = all(pred[c].values())
    print(f"  {c} per day (rbtx) {[round(x) for x in t['mark_day']]}; predicates {pred[c]}")
io.write_json(f'{S4}/report_rebuild.json', dict(results=out, predicates=pred, top5_mo1_c=top5))
