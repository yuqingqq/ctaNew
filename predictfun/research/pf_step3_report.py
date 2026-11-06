"""Aggregate step-3 baseline summaries across days. Usage: pf_step3_report <suffix> <day>..."""
import sys, json
from . import pf_io as io
suf = sys.argv[1] if sys.argv[1] != '-' else ''; days = sys.argv[2:]
arms = {}
for d in days:
    s = json.load(open(f'{io.OUT}/step3/{d}{suf}/summary.json'))
    for k, v in s['arms'].items():
        a = arms.setdefault(k, dict(gross_by_day=[], fills=0, shares=0.0, mo1=0.0, mo60=0.0, rebate=0.0, windows=0, pos=0, kinds={}))
        a['gross_by_day'].append(v['gross_usd']); a['fills'] += v['fills']; a['shares'] += v['shares']; a['windows'] += v['windows']
        a['pos'] += v['windows_positive']; a['mo1'] += v['markout_usd']['mo1']; a['mo60'] += v['markout_usd']['mo60']; a['rebate'] += v['rebate_usd_unverified']
        for kk, b in v['by_kind'].items():
            c = a['kinds'].setdefault(kk, dict(shares=0.0, mo1=0.0, settle=0.0)); c['shares'] += b['shares']; c['mo1'] += b['mo1']; c['settle'] += b['settle']
        a['delay_ms'] = round(s['delay_ms'])
out = {}
for k, a in arms.items():
    tot = sum(a['gross_by_day'])
    out[k] = dict(delay_ms=a['delay_ms'], gross_usd=round(tot, 2), gross_by_day=a['gross_by_day'], days_positive=sum(1 for x in a['gross_by_day'] if x > 0),
                  per_window=round(tot / a['windows'], 3), windows_positive=f"{a['pos']}/{a['windows']}", fills=a['fills'], shares=round(a['shares']),
                  gross_c_per_share=round(100 * tot / a['shares'], 3), mo1_c_per_share=round(100 * a['mo1'] / a['shares'], 3),
                  rebate_usd_unverified=round(a['rebate'], 2),
                  by_kind={kk: dict(share_of_volume=round(c['shares'] / a['shares'], 3), mo1_c_per_sh=round(100 * c['mo1'] / max(c['shares'], 1e-9), 3),
                                    settle_c_per_sh=round(100 * c['settle'] / max(c['shares'], 1e-9), 3)) for kk, c in sorted(a['kinds'].items())})
io.write_json(f'{io.OUT}/step3/report{suf or "_d260"}_{"_".join(days)}.json', out)
for k, v in out.items():
    print(k, {x: v[x] for x in ('delay_ms', 'gross_usd', 'gross_by_day', 'per_window', 'windows_positive', 'shares', 'gross_c_per_share', 'mo1_c_per_share', 'rebate_usd_unverified')})
    if k.endswith('/mid'):
        for kk, b in v['by_kind'].items(): print('     ', kk, b)
