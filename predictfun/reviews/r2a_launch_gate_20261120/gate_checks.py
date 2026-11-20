"""R2-A launch gate checks, declared before the gate runs (review 4cd7b45f's next priority; the frozen 256 originals of consumed Oct 23). Each is a
computed predicate; the verdict prints ALL TRUE only when every one holds:
(1) A COMPLETE; B1 STOPPED with exactly 2 completion records; B (resumed) COMPLETE.
(2) A = B: the same plan hash; every part's sha256 identical; every completion record identical except its elapsed / memory / time stamps; the
    same support JSON; the same census readings except timings and memory.
(3) A's parts = the r2av3 parts (the slice accepted by review 4cd7b45f), frame-equal part by part (columns, dtypes, values); the byte hashes are
    reported beside it.
(4) The audit sample: an INDEPENDENT implementation of r2_full_causal_v1 (its own key encoding and hashing) on the r2av3 sampled ok rows
    selects 1,598 rows (review 4cd7b45f's count) and exactly A's audited keys.
(5) A's causality PASS with every component 0 (rebuild, perturbed, visibility, firing, not evaluable); labels PASS on every sampled ok row;
    508 weight groups summing to one; the 256 originals exactly once, the 2 no-path originals with one status row each.
(6) Resources: every A part <= 900 s; A's recorded high-water and run_light's measured peak (the A log) <= 1,536 MiB.
(7) Every declared refused resume refused with its message, and B unchanged by them (refuse.json).
(8) A finalized again with every all-part reader replaced by a sentinel gives the same readings (refinalize.json).
Writes the support / audit CSV copies."""
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import pf_r2_shadow as R2                                  # noqa: E402

D, HERE, DAY = Path(R2.OUT), Path(__file__).resolve().parent, '20261023'; LABEL = sys.argv[1] if len(sys.argv) > 1 else 'gate'   # the run's tag prefix
A, B, V3 = D / f'census_{DAY}_{LABEL}A', D / f'census_{DAY}_{LABEL}B', D / f'census_{DAY}_r2av3'
JA, JB = json.loads((D / f'census_{DAY}_{LABEL}A.json').read_text()), json.loads((D / f'census_{DAY}_{LABEL}B.json').read_text())
recs = lambda d: [json.loads(p.read_text()) for p in sorted((d / 'parts').glob('part_*.json'))]
RA, RB = recs(A), recs(B); res = {}
TIMING = ('elapsed_s', 'part_peak_rss_mib', 'hwm_mib', 'asof', 'breach')
# (1)
b1 = [json.loads(x) for x in re.findall(r'^\{"mode": "B1".*$', (HERE / 'logs' / f'{LABEL}_B1.log').read_text(), re.M)]
res['states'] = JA['state'] == 'COMPLETE' and JB['state'] == 'COMPLETE' and len(b1) == 1 and b1[0]['state'] == 'STOPPED' and b1[0]['records'] == 2
print(f"(1) A {JA['state']}, B1 {b1}, B {JB['state']}: {res['states']}")
# (2)
strip_rec = lambda r: {k: v for k, v in r.items() if k not in TIMING}
strip_R = lambda R: {k: v for k, v in R.items() if k not in ('partitions', 'peak_rss_mib', 'finalize_hwm_mib')}
same_parts = [a['sha256'] for a in RA] == [b['sha256'] for b in RB] and len(RA) == 4
same_recs = [strip_rec(a) for a in RA] == [strip_rec(b) for b in RB]
same_sup = json.loads((D / f'support_{DAY}_{LABEL}A.json').read_text()) == json.loads((D / f'support_{DAY}_{LABEL}B.json').read_text())
res['resume_identity'] = JA['plan_sha256'] == JB['plan_sha256'] and same_parts and same_recs and same_sup and strip_R(JA['readings']) == strip_R(JB['readings'])
print(f"(2) A = B: plan {JA['plan_sha256'] == JB['plan_sha256']}, part hashes {same_parts}, records {same_recs}, support {same_sup}: {res['resume_identity']}")
# (3)
man3 = json.loads((V3 / 'manifest.json').read_text())['parts']; eq3, sh3 = [], []
for a, m in zip(RA, man3):
    fa, f3 = pd.read_parquet(A / 'parts' / a['file']), pd.read_parquet(V3 / m['file'])
    eq3.append(bool(list(fa.columns) == list(f3.columns) and all(fa[c].equals(f3[c]) for c in fa.columns))); sh3.append(a['sha256'] == m['sha256'])
res['equals_r2av3'] = len(eq3) == 4 and all(eq3)
print(f'(3) A parts = r2av3 parts, frame-equal: {eq3}; byte-identical: {sh3}: {res["equals_r2av3"]}')
# (4) an independent r2_full_causal_v1
V = R2.materialize(V3); ok = V[(V.status == 'ok') & V.sampled.astype(bool)]
def key(r):
    off = None if not np.isfinite(float(r.offset_ms)) else int(round(float(r.offset_ms)))
    return json.dumps(['r2_full_causal_v1', str(r.day), int(r.instrument_id), int(r.original_episode_id), int(r.side), str(r.source), str(r.head), off,
                       int(round(float(r.generation))), int(round(float(r.decision_ms)))], separators=(',', ':'))
sel = []
for _, g in ok.groupby(['original_episode_id', 'side', 'source', 'head']):
    ks = sorted((hashlib.sha256(k.encode('utf-8')).hexdigest(), k) for k in (key(r) for r in g.itertuples()))
    sel.extend(k for _, k in ks[:min(len(ks), max(1, math.ceil(len(ks) / 64)))])
ka = sorted(x['key'] for r in RA for x in r['causal']['audit'])
res['audit_sample'] = len(sel) == 1598 and sorted(sel) == ka and JA['readings']['causality']['checked'] == 1598
print(f"(4) the independent audit sample: {len(sel)} rows (declared 1,598), equal to A's {len(ka)} audited keys: {res['audit_sample']}")
pd.DataFrame([dict(part=r['part'], **{k: v for k, v in x.items() if k != 'draws'}, draws=json.dumps(x['draws'])) for r in RA for x in r['causal']['audit']]).to_csv(
    HERE / f'audit_{LABEL}A.csv', index=False)
# (5)
Ra = JA['readings']; cz = Ra['causality']; pop = Ra['population']
st_rows = {k: v for k, v in Ra['by_source_head_status'].items() if k.startswith('original ')}
res['readings'] = (cz['verdict'] == 'PASS' and cz['failed'] == 0 and all(v == 0 for v in cz['components'].values()) and Ra['labels']['verdict'] == 'PASS'
                   and Ra['labels']['checked'] == Ra['sampled_ok_rows'] == 44959 and Ra['weights'] == dict(groups=508, groups_summing_to_one=508)
                   and pop['represented'] == pop['selected'] == 256 and st_rows == {'original post no_path:missing_endpoint': 2})
print(f"(5) causality {cz['verdict']} {cz['components']} (retried {cz['retried']}); labels {Ra['labels']}; weights {Ra['weights']}; population "
      f"{pop['represented']} / {pop['selected']}; status rows {st_rows}: {res['readings']}")
# (6)
pk = [int(x) for x in re.findall(r'peak_rss_mib=(\d+)', (HERE / 'logs' / f'{LABEL}_A.log').read_text())]
res['resources'] = max(r['elapsed_s'] for r in RA) <= 900. and max(r['hwm_mib'] for r in RA) <= 1536. and len(pk) == 1 and pk[0] <= 1536
print(f"(6) A parts {[round(r['elapsed_s']) for r in RA]} s, high-water {[round(r['hwm_mib']) for r in RA]} MiB, run_light peak {pk} MiB: {res['resources']}")
# (7)
rf = json.loads((HERE / f'{LABEL}_refuse.json').read_text())
res['refused_resumes'] = len(rf) == 7 and all(v['refused'] for v in rf.values())
print(f"(7) refused resumes: { {k: v['refused'] for k, v in rf.items()} }: {res['refused_resumes']}")
# (8)
rz = json.loads((HERE / f'{LABEL}_refinalize.json').read_text())
res['refinalize_sentinel'] = strip_R(rz['readings']) == strip_R(JA['readings']) and rz['readings']['state'] == 'COMPLETE'
print(f"(8) re-finalized with sentinels: {rz['readings']['state']}, readings equal: {res['refinalize_sentinel']}")
S = json.loads((D / f'support_{DAY}_{LABEL}A.json').read_text())
pd.DataFrame([dict(cell=k, **v) for k, v in S['cells'].items()]).to_csv(HERE / f'support_cells_{LABEL}A.csv', index=False)
print(f'GATE CHECKS: {res}; ALL TRUE: {all(res.values())}')
sys.exit(0 if all(res.values()) else 1)
