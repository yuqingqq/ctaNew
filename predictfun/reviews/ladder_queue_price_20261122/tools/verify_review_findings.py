"""Verification of review 72a36de9's numerical claims (findings 1-2) from hash-bound inputs, before any new economics (session aea5bfb9, 2026-11-22).
Prints counts only: the front states' current displayed size by price state (finding 1) and the all-state vs common filled support (finding 2).
Inputs: the committed ARRIVAL-QUEUE receipt copies (hash = its audit.json) and the four original feature parquets (hash = the committed day records)."""
import gzip
import hashlib
import io
import json
from pathlib import Path

import pandas as pd

REPO = Path('/home/yuqing/ctaNew'); AQ = REPO / 'predictfun/reviews/ladder_arrival_queue_20261122/outputs'; DR = REPO / 'predictfun/reviews/ladder_rest_20261121/outputs/days'
FD = REPO / 'data/predict_fun/research/ladder_rest_v1_20261121/days'; DAYS = ('20261024', '20261025', '20261026', '20261027'); K = ['day', 'm', 'gid']
sha = lambda b: hashlib.sha256(b).hexdigest()
audit = json.loads((AQ / 'audit.json').read_text()); tab = {}
for n in ('states.csv', 'intervals.csv'):
    raw = gzip.decompress((AQ / f'{n}.gz').read_bytes()); assert sha(raw) == audit['outputs'][n], n; tab[n] = pd.read_csv(io.BytesIO(raw), dtype={'day': str}, float_precision='round_trip')
S, I_ = tab['states.csv'], tab['intervals.csv']; F = []
for d in DAYS:
    rec = json.loads((DR / f'{d}.json').read_text()); assert (FD / f'{d}.json').read_bytes() == (DR / f'{d}.json').read_bytes(); b = (FD / f'{d}_features.parquet').read_bytes()
    assert sha(b) == rec['sha256']['feat'], d; f = pd.read_parquet(io.BytesIO(b)); f['day'] = f.day.astype(str); F.append(f[f.kind == 'live'])
F = pd.concat(F, ignore_index=True); print('parquet live rows', len(F), 'columns with fp_/t_to_end/rung:', [c for c in F.columns if c.startswith('fp_') or c in ('t_to_end', 'rung', 'price', 'q')])
J = S.merge(F[K + ['t', 'side', 'fp_queue', 'fp_shown', 'fp_remaining', 'o_depth', 't_to_end']], on=K + ['t', 'side'], how='left', validate='one_to_one', suffixes=('', '_f'))
print('joined', len(J), 'missing', int(J.fp_shown.isna().sum() if 'fp_shown' in J else -1), 'Q equal', bool(((J.Q == J.fp_queue) | (J.Q.isna() & J.fp_queue.isna())).all()),
      'R equal', bool((J.R == J.fp_remaining_f if 'fp_remaining_f' in J else J.R == J.fp_remaining).all()), 'depth equal', bool((J.o_depth == J.o_depth_f).all()))
fr = J[J.queue_band == 'front']; ps = lambda d: 'inside_best' if d < 0 else ('at_touch' if d == 0 else 'behind_best')
J['price_state'] = J.o_depth.map(ps); J['display'] = J.fp_shown.map(lambda v: 'shown_unknown' if pd.isna(v) else ('shown_zero' if v == 0 else 'shown_positive'))
print('front by price x display'); print(pd.crosstab(J[J.queue_band == 'front'].price_state, J[J.queue_band == 'front'].display, margins=True).to_string())
print('all by price x display'); print(pd.crosstab(J.price_state, J.display, margins=True).to_string())
print('at_touch & shown_positive by queue band'); print(J[(J.price_state == 'at_touch') & (J.display == 'shown_positive')].queue_band.value_counts().to_string())
U = I_.groupby(K).agg(fills=('fills', 'sum'), shares=('shares', 'sum'), common=('common', 'all')).reset_index().merge(S[K + ['queue_band']], on=K)
print('finding 2 support (union):'); print(U.groupby('queue_band').apply(lambda g: pd.Series(dict(all_filled=int((g.fills > 0).sum()), common_filled=int(((g.fills > 0) & g.common).sum()),
                                                                                     all_shares=round(float(g.shares.sum()), 2), common_shares=round(float(g.shares[g.common].sum()), 2)))).to_string())
print('t_to_end < 30 s states', int((J.t_to_end < 30000).sum()), 'median t_to_end', float(J.t_to_end.median()))
