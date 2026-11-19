"""Receipt checks on the committed census v3 (Oct 23): the weights sum to one per original episode x side x head over the sampled ok rows and
are zero elsewhere; every row carries its absolute decision / END time and the mark source; the sampled ok rows carry the 91 inputs' names in
the export's order and the 14 context fields; the support tables as CSV. Prints one predicate per check and ALL TRUE at the end."""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import pf_r2_shadow as R2                                  # noqa: E402

D = '/home/yuqing/ctaNew/data/predict_fun/research/r2_shadow_20261119'; RV = '/home/yuqing/ctaNew/predictfun/reviews/r2a_shadow_census_20261119'
c = pd.read_parquet(f'{D}/census_20261023_v3.parquet'); ok = (c.status == 'ok') & c.sampled.astype(bool)
gw = c[ok].groupby(['original_episode_id', 'side', 'head']).weight.sum()
w_ok = bool(np.allclose(gw.values, 1., rtol=0, atol=1e-12) and (c.weight[~ok] == 0).all())
ep = c[c.source != 'original']; t_ok = bool((ep.end_abs_ms - ep.t0 - R2.END).abs().max() <= 1e-9 and (ep.decision_abs_ms - ep.t0 - ep.decision_ms).abs().max() <= 1e-9
                                            and ep.mark_source.notna().all())
cols_ok = all(x in c.columns for x in R2.INPUTS + R2.CTX) and list(c.columns[[list(c.columns).index(x) for x in R2.INPUTS]]) == R2.INPUTS
ctx_ok = bool(c.loc[ok, R2.CTX].notna().all().all())
S = json.load(open(f'{D}/support_20261023_v3.json'))
pd.DataFrame([dict(cell=k, **v) for k, v in S['cells'].items()]).to_csv(f'{RV}/support_cells_20261023_v3.csv', index=False)
pd.DataFrame([dict(side_head=k, category=cat, value=val, rows=n) for k, v in S['categories'].items() for cat, d in v.items() for val, n in d.items()]).to_csv(
    f'{RV}/support_categories_20261023_v3.csv', index=False)
print(f'weights one per episode x side x head ({len(gw)} groups), zero elsewhere: {w_ok}; absolute times and mark source on every episode row: {t_ok}; '
      f'the 91 inputs (export order) and 14 context columns present: {cols_ok}; context on every sampled ok row: {ctx_ok}; '
      f"support cells {len(S['cells'])}, SUPPORTED {sum(v['support'] == 'SUPPORTED' for v in S['cells'].values())}")
ok_all = w_ok and t_ok and cols_ok and ctx_ok; print(f'RECEIPT CHECKS ALL TRUE: {ok_all}')
sys.exit(0 if ok_all else 1)
