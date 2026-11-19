"""The R2-A census v2 against v3 (the canonical manifest order, c68a072c), Oct 23. Declared before v3 ran: v3 is in the canonical order and
equals v2 put in that order, on every column, exactly (the inputs / context included, both float32). Prints ALL TRUE at the end."""
import sys

import pandas as pd

sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import pf_r2_shadow as R2                                  # noqa: E402

D = '/home/yuqing/ctaNew/data/predict_fun/research/r2_shadow_20261119'
b, c = pd.read_parquet(f'{D}/census_20261023_v2.parquet'), pd.read_parquet(f'{D}/census_20261023_v3.parquet')
in_order = c[[k for k in R2.SORT_KEY if k in c]].equals(R2.canonical(c)[[k for k in R2.SORT_KEY if k in c]])
b2 = R2.canonical(b.drop(columns=['weight'])); b2['weight'] = R2.weights(b2)
same = list(b2.columns) == list(c.columns) and b2.equals(c)
print(f'v3 in the canonical order: {in_order}; v3 = v2 in that order, every column exact: {same} ({len(c)} rows, {len(c.columns)} columns)')
ok = in_order and same; print(f'CENSUS v3 = CANONICAL(v2): {ok}')
sys.exit(0 if ok else 1)
