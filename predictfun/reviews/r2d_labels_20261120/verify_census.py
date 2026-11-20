"""Verify a finished R2-D census independently of its CLI. Its candidates file must load against the manifest's sha256 (load_candidates
refuses otherwise), hold exactly candidate_rows rows, reproduce the manifest's candidate counts, and carry no outcome field. Every selected
candidate must be ok. Added 2026-11-20 because the census CLI's final print still referenced the removed manifest field 'candidates' and
exited 1 AFTER writing its outputs (that line is fixed next round; the module is unchanged here so that its suite's certification holds).
Usage: verify_census.py <build directory>; exit 0 only when every check holds."""
import json
import sys
from pathlib import Path

sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import pf_r2_labels as L                                  # noqa: E402

d = Path(sys.argv[1]); man = json.loads((d / 'manifest.json').read_text()); rows = L.load_candidates(d); cnt = {}
for r in rows:
    k = (r['day'], int(r['side']), r['kind'], r['source'], r['status']); cnt[k] = cnt.get(k, 0) + 1
want = {(c['day'], c['side'], c['kind'], c['source'], c['status']): c['n'] for c in man['candidate_counts']}
ok = dict(rows=len(rows) == man['candidate_rows'], counts=cnt == want, outcome_free=not any(set(r) & set(L.OUTCOMES) for r in rows),
          selected_ok=all(r['status'] == 'ok' for r in man['selected']), selected=len(man['selected']))
print(json.dumps(dict(ok, candidate_rows=man['candidate_rows'], manifest_sha256=L.C.sha(d / 'manifest.json'))))
sys.exit(0 if all(v for k, v in ok.items() if k != 'selected') else 1)
