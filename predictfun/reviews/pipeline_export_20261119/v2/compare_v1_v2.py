"""Declared before it ran: export v2 changes only the pre-activation (POST) order state. Per day and part: every RETAIN row identical to v1 in
every column; POST rows identical outside the queue inputs (queue_* / o_q* / fp_queue); prints the predicate and the POST columns that changed."""
import json
import sys
import numpy as np
import pandas as pd
R = sys.argv[1] if len(sys.argv) > 1 else 'data/predict_fun/research'
QUEUE = {'o_q_rel', 'o_q0_rel', 'o_q_frac', 'o_qunk', 'fp_queue'}
ok_all = True; changed = set(); n_post_changed = 0
for d in ('20261023', '20261024', '20261025', '20261026', '20261027'):
    r1 = json.load(open(f'{R}/pipeline_export_20261119/{d}/receipt.json')); r2 = json.load(open(f'{R}/pipeline_export_v2_20261119/{d}/receipt.json'))
    for x in sorted(r1['outputs']):
        a = pd.read_parquet(f'{R}/pipeline_export_20261119/{d}/{x}'); b = pd.read_parquet(f'{R}/pipeline_export_v2_20261119/{d}/{x}')
        same_shape = list(a.columns) == list(b.columns) and len(a) == len(b) and a[['oid', 'kind', 'j']].equals(b[['oid', 'kind', 'j']])
        ret = (a.kind == 'retain').values; diff = [c for c in a.columns if not a.loc[ret, c].equals(b.loc[ret, c])]
        post = ~ret; pdiff = [c for c in a.columns if not a.loc[post, c].equals(b.loc[post, c])]
        if pdiff:
            n_post_changed += int((~np.isclose(a.loc[post, 'fp_queue'].astype(float), b.loc[post, 'fp_queue'].astype(float), equal_nan=True)).sum())
        changed |= set(pdiff); ok = same_shape and not diff and set(pdiff) <= QUEUE; ok_all &= ok
        if not ok:
            print(d, x, 'shape', same_shape, 'RETAIN columns changed', diff[:6], 'POST columns changed outside the queue inputs', sorted(set(pdiff) - QUEUE)[:6])
print(f'RETAIN ROWS IDENTICAL AND POST CHANGES ONLY IN THE QUEUE INPUTS: {ok_all}; POST columns that changed {sorted(changed)}; POST rows with a changed fp_queue {n_post_changed:,}')
