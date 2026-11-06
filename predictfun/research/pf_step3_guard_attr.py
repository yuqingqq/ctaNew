"""Descriptive: where do baseline fills sit relative to D1's Binance guard condition?
Joins step-3 fills (mode 'mid', given delay suffix) to the step-2 grid at the fill time
(grid = receipt-time, fill = venue time; ~0.13 s apart on this host - descriptive only).
Against = BUY side with Binance 1 s move <= -0.5 bps, or SELL side with >= +0.5 bps (D1's rule);
favourable = the opposite; 0.25 s leg likewise.
Usage: pf_step3_guard_attr <suffix|-> <day>..."""
import sys
import numpy as np, pandas as pd
from . import pf_io as io

suf = sys.argv[1] if sys.argv[1] != '-' else ''; days = sys.argv[2:]
rows = []
for d in days:
    F = pd.read_parquet(f'{io.OUT}/step3/{d}{suf}/fills.parquet')
    F = F[F['mode'] == 'mid']
    G = pd.read_parquet(f'{io.OUT}/step2/{d}/grid.parquet', columns=['mid', 't', 'r1', 'rq'])
    G['k'] = np.round(G.t * 10).astype(int)
    F = F.assign(k=np.floor(F.t * 10).astype(int).clip(0, 2999))
    F = F.merge(G[['mid', 'k', 'r1', 'rq']], on=['mid', 'k'], how='left')
    F['day'] = d
    rows.append(F)
F = pd.concat(rows, ignore_index=True)
sg = np.where(F.side == 0, 1.0, -1.0)
out = {}
for pol, f in F.groupby('policy'):
    s = np.where(f.side == 0, 1.0, -1.0)
    res = {}
    for leg in ('r1', 'rq'):
        x = f[leg].values * s
        cls = np.select([np.isnan(x), x <= -0.5, x >= 0.5], ['unavailable', 'against', 'favourable'], 'neutral')
        tot_settle = f.settle.sum()
        res[leg] = {c: dict(share_of_volume=round(float(f.q[cls == c].sum() / f.q.sum()), 3),
                            mo1_c_per_sh=round(float(100 * f.mo1[cls == c].sum() / max(f.q[cls == c].sum(), 1e-9)), 3),
                            settle_c_per_sh=round(float(100 * f.settle[cls == c].sum() / max(f.q[cls == c].sum(), 1e-9)), 3),
                            settle_usd=round(float(f.settle[cls == c].sum()), 2),
                            share_of_settle_loss=round(float(f.settle[cls == c].sum() / tot_settle), 3),
                            days_negative=int(sum(1 for _, g in f[cls == c].groupby('day') if g.settle.sum() < 0)))
                    for c in ('against', 'neutral', 'favourable', 'unavailable')}
    out[pol] = res
io.write_json(f'{io.OUT}/step3/guard_attr{suf or "_d260"}_{"_".join(days)}.json', out)
for pol, res in out.items():
    for leg, r in res.items():
        print(pol, leg); [print('   ', c, v) for c, v in r.items()]
