"""Trade-to-book mapping check for pf_hold_check (2 days): depth of timed legs incl. negative (inside the spread = mapping/timing
miss); at-best legs: does the size at the leg's price drop in the next frame, vs a placebo price one tick deeper.
Usage: python3 -m predictfun.research.pf_hold_align_check"""
import os, sys

import numpy as np, pandas as pd
from predictfun.research import pf_io as io
COLS = ['mid', 't_ms'] + [f'{s}{i}_{k}' for s in 'ba' for i in range(1, 6) for k in 'ps']
acc = dict(vol=0.0, neg=0.0, best=0.0, drop_any=0.0, drop90=0.0, plac_any=0.0, plac_n=0.0)
for d in ['20261023', '20261101']:
    S1 = f'{io.OUT}/step1/{d}'
    bk = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=COLS) for f in sorted(os.listdir(f'{S1}/books'))], ignore_index=True).sort_values(['mid', 't_ms'], kind='stable')
    lt = pd.read_parquet(f'{S1}/legs_timed2.parquet', columns=['mid', 'side', 'ptick', 'shares', 't_ms', 'how', 't_e2', 't_rel'])
    lt = lt[(lt.t_rel >= 0) & (lt.t_rel < 240) & ((lt.how > 0) | np.isfinite(lt.t_e2))].copy()
    lt['ta'] = np.where(lt.how.values > 0, lt.t_ms.values, lt.t_e2.values)
    B = {m: g for m, g in bk.groupby('mid', sort=False)}
    for (mid, side), g in lt.groupby(['mid', 'side'], sort=False):
        b = B.get(mid)
        if b is None: continue
        tf = b.t_ms.values.astype(float); pre = 'b' if side == 0 else 'a'
        P = b[[f'{pre}{i}_p' for i in range(1, 6)]].values; S = b[[f'{pre}{i}_s' for i in range(1, 6)]].values
        ta = g.ta.values.astype(float); p = g.ptick.values; q = g.shares.values
        j0 = np.searchsorted(tf, ta, 'left') - 1; jn = np.searchsorted(tf, ta, 'right'); ok = (j0 >= 0) & (jn < len(tf))
        j0, jn, p, q = j0[ok], jn[ok], p[ok], q[ok]; b0 = P[j0, 0]; ok2 = b0 >= 0; j0, jn, p, q, b0 = j0[ok2], jn[ok2], p[ok2], q[ok2], b0[ok2]
        dep = (b0 - p) if side == 0 else (p - b0)
        acc['vol'] += q.sum(); acc['neg'] += q[dep < 0].sum(); at = dep == 0; acc['best'] += q[at].sum()
        G = pd.DataFrame(dict(p=p[at], j0=j0[at], jn=jn[at], q=q[at])).groupby(['p', 'j0', 'jn']).q.sum().reset_index()
        if not len(G): continue
        sz = lambda J, pp: np.where(P[J] == pp[:, None], S[J], 0.0).sum(1)
        s0 = sz(G.j0.values, G.p.values); sn = sz(G.jn.values, G.p.values)
        acc['drop_any'] += G.q.values[(sn < s0 - 1e-9) | (sn <= 1e-9)].sum(); acc['drop90'] += G.q.values[(s0 - sn >= 0.9 * G.q.values - 1e-9) | (sn <= 1e-9)].sum()
        pp = G.p.values - 1 if side == 0 else G.p.values + 1; s0p = sz(G.j0.values, pp); snp = sz(G.jn.values, pp); has = s0p > 0
        acc['plac_any'] += G.q.values[has & (snp < s0p - 1e-9)].sum(); acc['plac_n'] += G.q.values[has].sum()
    del bk, B; io.mem_check(d)
v = acc['vol']; print(f"timed live legs {v / 1e6:.1f}M sh: inside the spread (negative depth) {acc['neg'] / v:.1%}; at best {acc['best'] / v:.1%}")
print(f"at-best legs: size at the price drops in the next frame {acc['drop_any'] / acc['best']:.1%} (>= 90% of the leg or level gone {acc['drop90'] / acc['best']:.1%}); "
      f"placebo one tick deeper drops {acc['plac_any'] / acc['plac_n']:.1%}")
