"""Diagnostic (after compare_v1_v2.py FAILED as declared): where the v1 and v2 census rows sit, and whether their contents agree once both are
sorted by the full identity key. Not a pass criterion - an explanation of the declared failure."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import pf_r2_shadow as R2                                  # noqa: E402

D = '/home/yuqing/ctaNew/data/predict_fun/research/r2_shadow_20261119'
a, b = pd.read_parquet(f'{D}/census_20261023.parquet'), pd.read_parquet(f'{D}/census_20261023_v2.parquet')
print('positions of the original (no-path) rows: v1', a.index[a.source == 'original'].tolist(), '| v2', b.index[b.source == 'original'].tolist())
key = ['instrument_id', 'original_episode_id', 'side', 'source', 'head', 'offset_ms', 'decision_ms', 'status']
srt = lambda x: x.assign(_o=x.offset_ms.fillna(-1.), _d=x.decision_ms.fillna(-1.)).sort_values(['instrument_id', 'original_episode_id', 'side', 'source', 'head', '_o', '_d', 'status'], kind='stable').reset_index(drop=True)
a2, b2 = srt(a), srt(b)
same_keys = all(a2[k].astype(str).equals(b2[k].astype(str)) for k in key + ['sampled'])
ok = (b2.status == 'ok') & b2.sampled.astype(bool)
same_lab = all(np.array_equal(a2[c].values[ok.values], b2[c].values[ok.values], equal_nan=True) for c in ('q', 'value', 'qm', 'severe'))
cols = [c for c in R2.INPUTS + R2.CTX if c in a2 and c in b2]
bad_cols = [c for c in cols if not np.array_equal(a2[c].values[ok.values].astype(np.float32), b2[c].values[ok.values].astype(np.float32), equal_nan=True)]
print(f'after sorting by the identity key: identities / statuses / sampling identical {same_keys}; labels identical {same_lab}; inputs / context unequal at float32: {bad_cols}')
