"""R2-A v2 receipt checks, declared before the partitioned run; each a computed predicate on the saved outputs. Writes the support CSV copies.
(1) manifest: parts consecutive; each at most 64 originals; their union = the census JSON's selected IDs exactly, no original twice; every
    part's SHA-256 = the manifest = the census JSON's outputs; the manifest's ID hash = the census JSON's.
(2) population: every selected original has episode rows or exactly one `original` status row, never both; status rows by status.
(3) weights: 1 per original x side x head over its sampled ok rows (atol 1e-12), 0 elsewhere.
(4) inputs: the 91 + 16 in the export's order (list hash), float32, non-null context on every sampled ok row.
(5) refusals: the census reading per source = the parquet's refused POST sampled ok rows; retain rows never carry the flag.
(6) causality: PASS, every component 0, checked = sum over originals of ceil(sampled ok rows / 17) (the every-17th rule).
(7) labels: reconciliation PASS on every sampled ok row.
(8) resources: run_light's measured peak RSS <= 1,536 MiB (review d10d3698's 1.5 GiB target; the guard is 2,048) and every partition
    <= 900 s (its 15-minute stop rule).
(9) support: 28 declared cells, a fallback (WAIT post / KEEP retain) exactly on the insufficient ones; presence for 4 side-heads x 6 dimensions;
    per side / head ranges of the 6 numeric fields; the terminal other-side state and the delayed-entry / later-generation coverage reported."""
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

D = Path('/home/yuqing/ctaNew/data/predict_fun/research/r2_shadow_20261119'); RC = Path(__file__).resolve().parent; TAG = sys.argv[1] if len(sys.argv) > 1 else 'r2av2'
J = json.loads((D / f'census_20261023_{TAG}.json').read_text()); Rd = J['readings']; pdir = D / f'census_20261023_{TAG}'
man = json.loads((pdir / 'manifest.json').read_text()); sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
sel = {tuple(x) for x in Rd['population']['selected_ids']}; res = {}
# (1)
seen, dup, size_ok, sha_ok = set(), False, True, True
for i, p in enumerate(man['parts']):
    o_ = {tuple(x) for x in p['originals']}; dup |= bool(o_ & seen); seen |= o_; size_ok &= (p['part'] == i and len(o_) <= 64)
    sha_ok &= (sha(pdir / p['file']) == p['sha256'] == J['outputs'][f"census_20261023_{TAG}/{p['file']}"])
res['manifest'] = size_ok and sha_ok and not dup and seen == sel and man['selected_ids_sha256'] == Rd['population']['selected_ids_sha256'] \
    and J['outputs'][f'census_20261023_{TAG}/manifest.json'] == sha(pdir / 'manifest.json')
print(f"(1) manifest: {len(man['parts'])} parts, sizes {[len(p['originals']) for p in man['parts']]}, union = selected {seen == sel}, duplicate {dup}, "
      f"hashes {sha_ok}: {res['manifest']}")
P = R2.read_parts(pdir); ok = (P.status == 'ok') & P.sampled.astype(bool)
# (2)
k_ = list(zip(P.side.astype(int), P.original_episode_id.astype(int))); P['k'] = k_
orig = P[P.source == 'original']; epi = P[P.source != 'original']
n_orig = orig.groupby('k').size(); both = set(orig.k) & set(epi.k)
res['population'] = set(P.k) == sel and (n_orig == 1).all() and not both
print(f"(2) population: {len(set(P.k))} represented of {len(sel)}; status rows {orig.status.value_counts().to_dict()}; an original with both kinds: "
      f"{len(both)}: {res['population']}")
# (3)
g = P[ok].groupby(['original_episode_id', 'side', 'head']).weight.sum()
res['weights'] = bool(len(g) and np.allclose(g.values, 1., rtol=0, atol=1e-12) and (P.weight[~ok] == 0).all())
print(f"(3) weights: {len(g)} original x side x head groups sum to 1, 0 elsewhere: {res['weights']}")
# (4)
cols = R2.INPUTS + R2.CTX
res['inputs'] = J['inputs_list_sha256'] == hashlib.sha256(json.dumps(cols).encode()).hexdigest() and len(R2.INPUTS) == 91 and len(R2.CTX) == 16 \
    and all(str(P[c].dtype) == 'float32' for c in cols) and bool(P.loc[ok, R2.CTX].notna().all().all())
print(f"(4) inputs: 91 + 16 in the export's order, float32, context on every sampled ok row: {res['inputs']}")
# (5)
pp = P[ok & (P['head'] == 'post')]; cnt = {s_: int((g_.refused == True).sum()) for s_, g_ in pp.groupby('source')}  # noqa: E712
res['refusals'] = cnt == {s_: v for s_, v in Rd['refusals_by_source'].items()} and bool(P.refused[P['head'] == 'retain'].isna().all())
print(f"(5) refusals by source {cnt} = the census reading {Rd['refusals_by_source']}; retain rows carry none: {res['refusals']}")
# (6)
exp = int(sum(math.ceil(n / 17) for n in P[ok].groupby('k').size().values)); cz = Rd['causality']
res['causality'] = cz['verdict'] == 'PASS' and cz['failed'] == 0 and all(v == 0 for v in cz['components'].values()) and cz['checked'] == exp and cz['every'] == 17
print(f"(6) causality: {cz['verdict']}, checked {cz['checked']} (every 17th: {exp}), components {cz['components']}, draws retried {cz.get('retried')}: {res['causality']}")
# (7)
res['labels'] = Rd['labels']['verdict'] == 'PASS' and Rd['labels']['checked'] == int(ok.sum()) == Rd['sampled_ok_rows']
print(f"(7) labels: {Rd['labels']} on {int(ok.sum())} sampled ok rows: {res['labels']}")
# (8)
lg = (RC / 'logs' / f'pr2b_census_{TAG}.log').read_text(); pk = [int(x) for x in re.findall(r'peak_rss_mib=(\d+)', lg)]
res['resources'] = len(pk) == 1 and pk[0] <= 1536 and max(Rd['partitions']['seconds']) <= 900.
print(f"(8) resources: measured peak {pk} MiB (<= 1,536), partition seconds {[round(x) for x in Rd['partitions']['seconds']]} (<= 900): {res['resources']}")
# (9)
S = json.loads((D / f'support_20261023_{TAG}.json').read_text()); cl = S['cells']
fb = all((v['fallback'] is None) == (v['support'] == 'SUPPORTED') and (v['fallback'] in (None, {'post': 'WAIT', 'retain': 'KEEP'}[k.split()[1]])) for k, v in cl.items())
res['support'] = len(cl) == 28 and fb and set(S['presence']) == set(R2.SIDE_HEADS) and all(set(S['presence'][h]) == set(R2.DIMS) for h in R2.SIDE_HEADS) \
    and set(S['ranges']) == set(R2.SIDE_HEADS) and all(set(S['ranges'][h]) == set(R2.NUMERIC) for h in R2.SIDE_HEADS)
term = {h: S['presence'][h]['c_other']['terminal'] for h in R2.SIDE_HEADS}
print(f"(9) support: {len(cl)} cells ({sum(v['support'] == 'SUPPORTED' for v in cl.values())} supported), fallbacks on the insufficient ones {fb}, "
      f"presence and per side / head ranges complete: {res['support']}; terminal other-side originals per side / head {term}")
pd.DataFrame([dict(cell=k, **v) for k, v in cl.items()]).to_csv(RC / f'support_cells_20261023_{TAG}.csv', index=False)
for nm in ('categories', 'cross_cells', 'coverage'):
    pd.DataFrame(S[nm]).to_csv(RC / f'support_{nm}_20261023_{TAG}.csv', index=False)
pd.DataFrame([dict(side_head=h, field=c, lo=v[0], hi=v[1]) for h, r in S['ranges'].items() for c, v in r.items()]).to_csv(RC / f'support_ranges_20261023_{TAG}.csv', index=False)
pd.DataFrame([dict(side_head=h, dim=d, value=v, originals=n) for h, r in S['presence'].items() for d, x in r.items() for v, n in x.items()]).to_csv(
    RC / f'support_presence_20261023_{TAG}.csv', index=False)
cv = pd.DataFrame(S['coverage']); print('coverage (rows / distinct originals / nonzero originals):'); print(cv.to_string(index=False))
print(f'RECEIPT CHECKS: {res}; all: {all(res.values())}')
sys.exit(0 if all(res.values()) else 1)
