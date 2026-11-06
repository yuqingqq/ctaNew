"""Aggregate step-4 arms across days; deltas vs the same-policy base at the same timing.
Usage: pf_step4_report <scenario>_<armset> <day>..."""
import sys, json, re
from . import pf_io as io
tag = sys.argv[1]; days = sys.argv[2:]
A = {}
for d in days:
    s = json.load(open(f'{io.OUT}/step4/{tag}/{d}/summary.json'))
    for k, v in s['arms'].items():
        a = A.setdefault(k, dict(by_day=[], shares=0.0, fills=0, mo1=0.0, rebate=0.0, pos=0, windows=0, withheld={}))
        a['by_day'].append(v['gross_usd']); a['shares'] += v['shares']; a['fills'] += v['fills']; a['mo1'] += v['mo1_usd']
        a['rebate'] += v['rebate_usd_unverified']; a['pos'] += v['windows_positive']; a['windows'] += v['windows']
        for w, n in v['withheld_decisions'].items(): a['withheld'][w] = a['withheld'].get(w, 0) + n
out = {}
for k, a in A.items():
    pol, rest = k.split('/')[0], k.split('/')[1]
    base = A.get(f"{pol.split('+')[0]}/{rest}")
    tot = sum(a['by_day'])
    d = [x - y for x, y in zip(a['by_day'], base['by_day'])] if base else None
    out[k] = dict(gross_usd=round(tot, 2), by_day=a['by_day'], days_positive=sum(x > 0 for x in a['by_day']),
                  delta_vs_base=round(sum(d), 2) if d else None, delta_by_day=[round(x, 2) for x in d] if d else None,
                  delta_days_positive=sum(x > 0 for x in d) if d else None, fills=a['fills'], shares=round(a['shares']),
                  c_per_share=round(100 * tot / max(a['shares'], 1e-9), 3), mo1_c_per_share=round(100 * a['mo1'] / max(a['shares'], 1e-9), 3),
                  windows_positive=f"{a['pos']}/{a['windows']}", rebate_unverified=round(a['rebate'], 2), withheld=a['withheld'])
io.write_json(f'{io.OUT}/step4/report_{tag}_{"_".join(days)}.json', out)
for k in sorted(out, key=lambda x: (x.split('/')[1], x)):
    v = out[k]
    print(f"{k:34s} gross {v['gross_usd']:10.2f} by_day {v['by_day']} | d_vs_base {v['delta_vs_base']} {v['delta_by_day']} | sh {v['shares']:8d} c/sh {v['c_per_share']:7.3f} mo1 {v['mo1_c_per_share']:7.3f}")
