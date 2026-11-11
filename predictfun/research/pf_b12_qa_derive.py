"""B12 Q&A addendum (2026-11-11, post hoc, descriptive): arithmetic on the committed b12.json / b12_posthoc.json only (no
new run). Our first-minute $ by guard state at the best, the ON-state gap to the top 5, and the at-best share of trades
that go through the level (sweep-known), for ours / top 5 / ranks >= 6.
Usage: python3 -m predictfun.research.pf_b12_qa_derive predictfun/reviews/b12_20261110
"""
import json, sys
R = sys.argv[1]
d = json.load(open(f'{R}/b12.json')); p = json.load(open(f'{R}/b12_posthoc.json'))
days = p['days']
P3 = p['P3']; T3 = d['T3']['50']; T1 = d['T1']
usd = lambda x: x['sh'] / days * x['net1'] / 100
on, off = P3['ours']['ON'], P3['ours']['OFF']
tot = T1['ours']['usd_day1']; at_best = usd(on) + usd(off)
# identity: ON + OFF (+ warm-up, 0 legs for ours) = at-best $/day from P2
assert P3['ours']['warm-up']['legs'] == 0
ab = p['P2']['ours']['at best']; assert abs(at_best - usd(ab)) < 1e-9, (at_best, usd(ab))
print(f"ours first minute: {T1['ours']['legs']} legs, ${tot:+.2f}/day at 1 s")
print(f"ours at best, guard OFF (lag 50): {off['legs']} legs, {off['sh']:.1f} sh, {off['sh']/days:.0f} sh/day, {off['net1']:+.2f} c/sh, ${usd(off):+.2f}/day = {usd(off)/tot:.0%} of the first-minute $")
print(f"ours at best, guard ON: {on['legs']} legs, {on['sh']/days:.0f} sh/day, {on['net1']:+.2f}; top 5 ON {P3['top 5']['ON']['net1']:+.2f}; "
      f"gap at our ON volume ${on['sh']/days*(P3['top 5']['ON']['net1']-on['net1'])/100:+.2f}/day")
for g in ('ours', 'top 5', 'O (ranks >= 6)'):
    c = T3[g]
    sw = sum(v['share'] for k, v in c.items() if isinstance(v, dict) and k.endswith(', swept'))
    unk = sum(v['share'] for k, v in c.items() if isinstance(v, dict) and k.endswith('sweep unknown'))
    print(f"{g}: at-best share from trades through the level {sw:.1%} of all, {sw/(1-unk):.1%} of sweep-known (unknown {unk:.1%}); "
          f"ON not swept net {c['ON, not swept']['net1']:+.2f}")
