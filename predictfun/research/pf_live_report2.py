"""Report for live-deviation round 2 (run_live_dev2.sh; exec-port request 2026-11-02): live-like C2-P = MICRO + D1 stack +
M025 + A1 5..25, no amend, min 1 USD withheld. A: sequential removal lock XS0/200/500/1000 vs concurrent (XL200A here;
500/1000 ms from round 1, which lacked the 1 USD minimum, worth ~+100$). B: exec-port throttle XT1 / XT2 at XS0 / XS500.
Per arm: 6-day gross, by day, vs live-like, shares, +1 s mark, window-end inventory p95, max |net| inside a window,
requests per quoting minute, throttle counters (place intents immediate / delayed / never placed, delay p50 / p90).
Usage: python3 -m predictfun.research.pf_live_report2
"""
import json
import numpy as np, pandas as pd
from . import pf_io as io

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031']
D = f'{io.OUT}/step4/live_tokyo20_custom_e2_preds_lodo6_livedev2'; D1R = f'{io.OUT}/step4/live_tokyo20_custom_e2_preds_lodo6_livedev'
PL = 'MICRO+G4+G2+G6+G5+G3+G1+M025+XNA+XMIN1'; A1 = '/mid@A1:5:25:0'
XS = ['', '+XS0', '+XS200', '+XS500', '+XS1000', '+XL200A', '+XS0+XT1', '+XS500+XT1', '+XS0+XT2', '+XS500+XT2']
SUM = {d: json.load(open(f'{D}/{d}/summary.json'))['arms'] for d in DAYS}
R1 = {d: json.load(open(f'{D1R}/{d}/summary.json'))['arms'] for d in DAYS}


def stem(n):
    return n.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')


def stats(base, S, name):
    v = [S[d][name] for d in DAYS]; sh = sum(x['shares'] for x in v); g = [x['gross_usd'] for x in v]
    out = dict(by_day=g, sum=round(sum(g), 2), pos=sum(x > 0 for x in g), shares=round(sh),
               mo1_c_sh=round(100 * sum(x['mo1_usd'] for x in v) / max(sh, 1e-9), 3), inv_end_p95_max=max(x['abs_net_end_p95'] for x in v))
    mx = 0.0
    for d in DAYS:
        f = pd.read_parquet(f'{base}/{d}/fills/{stem(name)}.parquet', columns=['mid', 't', 'side', 'q']).sort_values(['mid', 't'], kind='stable')
        if len(f):
            mx = max(mx, float(pd.Series(np.where(f.side.values == 0, 1.0, -1.0) * f.q.values).groupby(f.mid.values).cumsum().abs().max()))
    out['max_abs_net_in_window'] = round(mx, 1)
    q = pd.concat([pd.read_parquet(f'{base}/{d}/requests/{stem(name)}.parquet') for d in DAYS], ignore_index=True)
    W = pd.concat([pd.read_parquet(f'{base}/{d}/win/{stem(name)}.parquet') for d in DAYS], ignore_index=True)
    qm = np.concatenate([np.arange(int(w // 60000), int(w // 60000) + 4) for w in W.ws.values])
    cm = pd.Series((q.t_ms // 60000).astype(np.int64)).value_counts().reindex(qm, fill_value=0).values
    pm = pd.Series((q.t_ms[q.kind == 'place'] // 60000).astype(np.int64)).value_counts().reindex(qm, fill_value=0).values
    out['requests_per_min'] = dict(mean=round(float(cm.mean()), 1), p99=float(np.percentile(cm, 99)), max=int(cm.max()), share_gt240=round(float((cm > 240).mean()), 3),
                                   places_mean=round(float(pm.mean()), 1), places_max=int(pm.max()))
    c = {}
    for x in v:
        for k, n in x['live'].get('counters', {}).items():
            c[k] = c.get(k, 0) + n
    out['counters'] = c
    dl = [x['live'].get('place_delay_ms') for x in v]
    if any(dl):
        out['place_delay_ms_by_day'] = [dict(p50=x['p50'], p90=x['p90']) if x else None for x in dl]
    if 'place_immediate' in c:
        tot = c.get('place_immediate', 0) + c.get('place_delayed', 0) + c.get('place_never', 0)
        out['place_intents'] = dict(total=tot, immediate_share=round(c.get('place_immediate', 0) / tot, 3), delayed_share=round(c.get('place_delayed', 0) / tot, 3),
                                    never_share=round(c.get('place_never', 0) / tot, 3))
    return out


out = dict(days=DAYS, base='live-like C2-P: ' + PL + A1, arms={})
ref = None
for x in XS:
    r = stats(D, SUM, PL + x + A1)
    if x == '':
        ref = r
    r['vs_live_like'] = dict(by_day=[round(a - b, 2) for a, b in zip(r['by_day'], ref['by_day'])], sum=round(r['sum'] - ref['sum'], 2))
    out['arms'][x or 'live-like (lock 0, no throttle)'] = r
P = 'MICRO+G4+G2+G6+G5+G3+G1+M025'
for x in ('+XNA+XL500A', '+XNA+XL1000A'):
    r = stats(D1R, R1, P + x + A1); out['arms'][f'{x} (round 1, no 1 USD minimum)'] = r
io.write_json(f'{D}/report_live2.json', out)
print(f"{'arm':38s} {'6-day $':>10s} {'vs base':>9s} {'shares':>8s} {'mo1':>6s} {'inv95':>6s} {'max|net|':>8s} {'req/min':>7s} {'>240':>5s}  place intents")
for k, r in out['arms'].items():
    pi = r.get('place_intents'); vs = r.get('vs_live_like', {}).get('sum', float('nan'))
    print(f"{k:38s} {r['sum']:10.2f} {vs:9.2f} {r['shares']:8d} {r['mo1_c_sh']:6.2f} {r['inv_end_p95_max']:6.1f} {r['max_abs_net_in_window']:8.1f} "
          f"{r['requests_per_min']['mean']:7.1f} {r['requests_per_min']['share_gt240']:5.2f}  "
          + (f"imm {pi['immediate_share']} delayed {pi['delayed_share']} never {pi['never_share']} of {pi['total']}" if pi else ''))
    if 'place_delay_ms_by_day' in r:
        print('      delay p50/p90 by day', [(x['p50'], round(x['p90'])) if x else None for x in r['place_delay_ms_by_day']], 'by day $', r['by_day'])
