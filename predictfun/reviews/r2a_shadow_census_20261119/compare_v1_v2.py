"""The R2-A census v1 (the first run: before the review 7d59bde7 amendments and the memory reduction) against v2 (the amended module), Oct 23.
Declared before this comparison read either output: the rows in the same order with identical identities, sources, heads, decisions, statuses
and sampling flags; identical labels (q, value, qm, severe; exact); the 91 inputs and 14 context fields equal to float32 rounding on the rows
where v2 keeps them (v2 stores them as float32 and drops the context on rows that are not sampled ok). Prints ALL TRUE at the end."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import pf_r2_shadow as R2                                  # noqa: E402

D = '/home/yuqing/ctaNew/data/predict_fun/research/r2_shadow_20261119'
a, b = pd.read_parquet(f'{D}/census_20261023.parquet'), pd.read_parquet(f'{D}/census_20261023_v2.parquet')
keys = ['instrument_id', 'original_episode_id', 'side', 'source', 'head', 'offset_ms', 'decision_ms', 'status', 'sampled']
same_keys = len(a) == len(b) and all(a[k].astype(str).equals(b[k].astype(str)) for k in keys)
lab = ['q', 'value', 'qm', 'severe']; ok = (b.status == 'ok') & b.sampled.astype(bool)
same_lab = all(np.array_equal(a[c].values[ok.values], b[c].values[ok.values], equal_nan=True) for c in lab)
cols = [c for c in R2.INPUTS + R2.CTX if c in a and c in b]
f32 = all(np.array_equal(a[c].values[ok.values].astype(np.float32), b[c].values[ok.values].astype(np.float32), equal_nan=True) for c in cols)
print(f'rows {len(a)} / {len(b)}; identities, statuses and sampling identical: {same_keys}; labels on the {int(ok.sum())} sampled ok rows identical: {same_lab}; '
      f'{len(cols)} inputs / context fields equal at float32: {f32}')
ok_all = same_keys and same_lab and f32
print(f'CENSUS v2 = v1 (identities, statuses, labels exact; inputs at float32): {ok_all}')
sys.exit(0 if ok_all else 1)
