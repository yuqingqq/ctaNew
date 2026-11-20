"""R2-A v2 (review d10d3698 closed): the partitioned census of the same frozen 256 originals (4 partitions of 64) against census v3.
Declared before the partitioned run (amended before it, to add the per-column diagnostic and the joint retain / refusal-count predicates):
the concatenated parts are in the canonical order and equal v3 on every column, exactly (dtypes included; float32 inputs), except `refused`,
which may change only on joint POST sampled ok rows, from missing to the engine's boolean outcome (P3); every other row's `refused` identical
(joint retain rows stay missing); the coarse support cells (rows, distinct originals, nonzero originals per side x head x horizon bin)
identical; the selected-ID hash identical; the population reconciliation EXACT. A failing column is followed by a diagnostic line (dtypes;
values equal as objects), which is not part of the verdict. Prints one predicate per check and the verdict."""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import pf_r2_shadow as R2                                  # noqa: E402

D = '/home/yuqing/ctaNew/data/predict_fun/research/r2_shadow_20261119'; TAG = sys.argv[1] if len(sys.argv) > 1 else 'r2av2'   # the census tag
v3 = pd.read_parquet(f'{D}/census_20261023_v3.parquet'); pt = R2.read_parts(f'{D}/census_20261023_{TAG}')
order = pt[[k for k in R2.SORT_KEY if k in pt]].equals(R2.canonical(pt)[[k for k in R2.SORT_KEY if k in pt]])
same_cols = set(v3.columns) == set(pt.columns) and len(v3) == len(pt)
cols_v3 = [c for c in v3.columns if c != 'refused']
bad_cols = [c for c in cols_v3 if not (c in pt and len(v3) == len(pt) and v3[c].equals(pt[c]))]
for c in bad_cols:
    if c in pt and len(v3) == len(pt):
        a_, b_ = v3[c].astype(object).values, pt[c].astype(object).values
        eq = all((pd.isna(x) and pd.isna(y)) or x == y for x, y in zip(a_, b_))
        print(f'  diagnostic: {c}: dtype v3 {v3[c].dtype} vs parts {pt[c].dtype}; values equal as objects: {eq}')
same_other = same_cols and not bad_cols
jp = (pt.source.str.startswith('joint') & (pt['head'] == 'post') & (pt.status == 'ok') & pt.sampled.astype(bool)).values
ra, rb = v3.refused.values, pt.refused.values
same_ref = same_cols and all((pd.isna(x) and pd.isna(y)) or x == y for x, y, j in zip(ra, rb, jp) if not j)
joint_ok = same_cols and all(pd.isna(x) and isinstance(y, (bool, np.bool_)) for x, y, j in zip(ra, rb, jp) if j)
jr = (pt.source.str.startswith('joint') & (pt['head'] == 'retain')).values
joint_retain_missing = bool(pd.isna(pt.refused[jr]).all())
s3, sp = json.load(open(f'{D}/support_20261023_v3.json'))['cells'], json.load(open(f'{D}/support_20261023_{TAG}.json'))['cells']
key = lambda k: k.replace('.0,', ',').replace('.0]', ']')
cells = {key(k): (v['rows'], v['episodes'], v['nonzero_episodes']) for k, v in s3.items()} == {key(k): (v['rows'], v['episodes'], v['nonzero_episodes']) for k, v in sp.items()}
j3, jn = json.load(open(f'{D}/census_20261023_v3.json'))['readings'], json.load(open(f'{D}/census_20261023_{TAG}.json'))['readings']
pop = j3['population']['selected_ids_sha256'] == jn['population']['selected_ids_sha256'] and jn['population']['reconciliation'] == 'EXACT'
n_joint_ref = {s_: int(sum(1 for y, j, src in zip(rb, jp, pt.source.values) if j and src == s_ and y is True)) for s_ in ('joint_post_keep', 'joint_chase')}
ref_counts = all(jn['refusals_by_source'].get(s_) == n_joint_ref[s_] for s_ in n_joint_ref)
print(f'canonical order: {order}')
print(f'every column but refused exact, dtypes included ({len(pt)} rows, {len(pt.columns)} columns; v3 {len(v3)} x {len(v3.columns)}): {same_other}'
      + (f' (failing: {bad_cols})' if bad_cols else ''))
print(f'refused unchanged off the joint POST sampled ok rows: {same_ref}; joint retain rows carry none: {joint_retain_missing}')
print(f'joint POST sampled ok rows ({int(jp.sum())}) now carry the boolean outcome: {joint_ok}; refused among them {n_joint_ref}, equal to the census '
      f'reading: {ref_counts}')
print(f'coarse support cells identical: {cells}; selected-ID hash identical and reconciliation EXACT: {pop}')
ok = order and same_other and same_ref and joint_retain_missing and joint_ok and ref_counts and cells and pop
print(f'R2-A v2 PARTITIONED ({TAG}) = v3 (+ the joint POST refusal outcome): {ok}')
sys.exit(0 if ok else 1)
