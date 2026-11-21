"""Compare the v1.1 and v1.2 outputs: every CSV byte for byte, checks.json without its asof -> exit 0 iff identical."""
import json, sys
from pathlib import Path
E = Path('/home/yuqing/ctaNew/predictfun/reviews/r2f_production_20261120/checks_equivalence')
A, B = (sys.argv[1:3] if len(sys.argv) == 3 else ('v11', 'v12'))
names = sorted({p.name for v in (A, B) for p in (E / v).glob('*.csv')}); bad = []
want = ['arms.csv', 'cost.csv', 'coverage.csv', 'paths.csv', 'population.csv', 'weights.csv', 'y_summary.csv']
if names != want:
    bad.append(f'csv set {names}')
for n in names:
    a, b = E / A / n, E / B / n
    if not (a.exists() and b.exists() and a.read_bytes() == b.read_bytes()):
        bad.append(n)
ja, jb = (json.loads((E / v / 'checks.json').read_text()) for v in (A, B))
for j in (ja, jb):
    j.pop('asof')
if ja != jb:
    bad.append('checks.json: ' + str([k for k in set(ja) | set(jb) if ja.get(k) != jb.get(k)]))
n_ok = sum(r['n_ok'] for r in ja['readings']['y_summary'] if r['day'] == 'both' and r['head'] == 'both')
print(json.dumps(dict(identical=not bad, differences=bad, files=names, ok_pairs=n_ok, controls=len(ja['readings']['controls']))))
sys.exit(1 if bad else 0)
