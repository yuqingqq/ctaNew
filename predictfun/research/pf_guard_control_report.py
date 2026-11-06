"""Stage 3 control (declared in STATUS.yml 2026-11-04 05:10Z; rule 7): the current guard at stricter thresholds (M015 / M010 /
M005) vs the new per-side guard, matched on shares filled. For each new-guard arm the control curve (current guard: M025 and the
stricter arms) is interpolated linearly in shares; the new guard's gain = its value minus the control's at the same shares, for
+1 s c/sh, bad share and bad $ / good $ (8 read days; 11-03 shown separately).
Usage: python3 -m predictfun.research.pf_guard_control_report
"""
import json
import numpy as np, pandas as pd
from . import pf_io as io

D8 = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']; S4 = f'{io.OUT}/step4'; B = 'MICRO+G4+G2+G6+G5+G3+G1'
stem = lambda n: n.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')
arm = lambda tok: f'{B}+{tok}+XNA+XMIN1+S60/mid@A1:5:25:0'


def stats(days, k, n, tag):
    pre = f'preds_{k}' if isinstance(k, str) else f'preds_g{k}'
    S = {d: json.load(open(f'{S4}/live_tokyoreal_custom_e2_{pre}_{tag}/{d}/summary.json'))['arms'][n] for d in days}
    f = pd.concat([pd.read_parquet(f'{S4}/live_tokyoreal_custom_e2_{pre}_{tag}/{d}/fills/{stem(n)}.parquet', columns=['q', 'mo1']) for d in days])
    c = 100 * f.mo1 / f.q; q = f.q.sum(); bad = c <= -2; good = c >= 0
    return dict(mtm=float(sum(S[d]['mtm_stop_usd'] for d in days)), shares=float(q), mo1=float(100 * f.mo1.sum() / q), bad=float(f.q[bad].sum() / q),
                ratio=float(-f.mo1[bad].sum() / f.mo1[good].sum()))


out = {}
for lab, days in (('8 read days', D8), ('11-03', ['20261103'])):
    ctl = [('M025 (current)', stats(days, 20, arm('M025'), 'gd'))] + [(t, stats(days, 20, arm(t), 'gdc')) for t in ('M015', 'M010', 'M005')]
    new = [(f'{w} {k}%', stats(days, k, arm('V0' if w == 'new instead' else 'M025+V0'), 'gd')) for w in ('new instead', 'new added') for k in (10, 20, 30, 40, 50, 60, 70)]
    print(f'\n{lab}: {"arm":16s} {"shares":>8s} {"mark@stop":>9s} {"+1s c/sh":>8s} {"bad %":>6s} {"bad$/good$":>10s} | vs current guard at the same shares: +1 s, bad, ratio')
    cs = np.array([r['shares'] for _, r in ctl]); o = np.argsort(cs)
    for nm, r in ctl:
        print(f'  {nm:16s} {r["shares"]:8.0f} {r["mtm"]:9.0f} {r["mo1"]:8.3f} {r["bad"]:6.1%} {r["ratio"]:10.0%}')
    for nm, r in new:
        m = {x: float(np.interp(r['shares'], cs[o], np.array([c[x] for _, c in ctl])[o])) for x in ('mo1', 'bad', 'ratio')}
        inside = cs.min() <= r['shares'] <= cs.max()
        d = dict(mo1=r['mo1'] - m['mo1'], bad=r['bad'] - m['bad'], ratio=r['ratio'] - m['ratio'], inside_control_range=bool(inside))
        out[f'{lab}|{nm}'] = dict(arm=r, matched_control=m, gain=d)
        print(f'  {nm:16s} {r["shares"]:8.0f} {r["mtm"]:9.0f} {r["mo1"]:8.3f} {r["bad"]:6.1%} {r["ratio"]:10.0%} | {d["mo1"]:+.3f}  {d["bad"]:+.1%}  {d["ratio"]:+.0%}'
              + ('' if inside else '  (outside the control range: extrapolated, not interpolated)'))
    print(f'  matched pull rate (current guard input ranked through the same V0 mechanism, preds_b<k>): new minus base-ranked')
    for w, tok in (('instead', 'V0'), ('added', 'M025+V0')):
        for k in (10, 20, 30, 40, 50, 60, 70):
            nw = stats(days, k, arm(tok), 'gd'); bs = stats(days, f'b{k}', arm(tok), 'gdb')
            out[f'{lab}|matched|{w}|{k}'] = dict(new=nw, base_ranked=bs)
            print(f"    {w:8s} {k}%: base-ranked shares {bs['shares']:7.0f} +1 s {bs['mo1']:+.3f} bad {bs['bad']:.1%} ratio {bs['ratio']:.0%} mark@stop {bs['mtm']:6.0f} | "
                  f"new shares {nw['shares']:7.0f} +1 s {nw['mo1']:+.3f} bad {nw['bad']:.1%} ratio {nw['ratio']:.0%} mark@stop {nw['mtm']:6.0f} | gain +1 s {nw['mo1'] - bs['mo1']:+.3f}, ratio {nw['ratio'] - bs['ratio']:+.0%}")
io.write_json(f'{S4}/report_guard_control.json', out)
