"""Diagnostic (not a receipt): for the slice's fill mismatches, the engine's per-trade fills vs the keep/pull run's fills, raw and summed per timestamp."""
import sys
sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import numpy as np, pandas as pd
import pf_branch_engine as BE, pf_keep_pull as KP, pf_queue_place_test as QP
from pathlib import Path
day = '20261024'; D = pd.read_csv(Path(BE.OUT) / f'slice_{day}.csv'); bad = D[~D.keep_fills_equal.astype(bool) | ~D.pull_fills_equal.astype(bool)]
kpd = Path(KP.OUT) / day; O = pd.read_parquet(kpd / 'orders.parquet', columns=['oid', 'm', 'side', 't', 'p']).set_index('oid')
F = pd.read_parquet(kpd / 'fills.parquet'); F = F[(F.clock == 50.) & F.oid.isin(bad.oid)]
src = KP.Tape(); off = src.offset(day); ms = sorted(set(bad.m.astype(int))); data = src.load(day, ms, off); Ws = {int(W[0]['m']): W for W in src.pairs(data)}
agg_ok = {'keep': 0, 'pull': 0}; agg_bad = {'keep': [], 'pull': []}
for r in bad.itertuples():
    o = O.loc[r.oid]; path, _ = QP.build_path(Ws[int(o.m)], float(o.t), float(o.t) + BE.END)
    kf = F[F.oid == r.oid].sort_values('t'); kt, kq = kf.t.values, kf.q.values
    for mode in ('keep', 'pull'):
        if getattr(r, f'{mode}_fills_equal'):
            continue
        L = BE.finish(BE.run(path, BE.side_policy(int(o.side), mode))); ef = [(f[0], f[3]) for f in L['fills'] if f[1] == 0]
        m_ = (kt < float(BE.KP.landing(BE.KP.J0))) if mode == 'pull' else np.ones(len(kt), bool)
        ea = pd.Series([q for _, q in ef]).groupby([t for t, _ in ef]).sum() if ef else pd.Series(dtype=float)
        ka = pd.Series(kq[m_]).groupby(kt[m_]).sum() if m_.any() else pd.Series(dtype=float)
        same = len(ea) == len(ka) and np.allclose(ea.index.values, ka.index.values, rtol=0, atol=1e-9) and np.allclose(ea.values, ka.values, rtol=0, atol=1e-9)
        agg_ok[mode] += same
        if not same:
            agg_bad[mode].append((int(r.oid), list(zip(ea.index.round(3), ea.values)), list(zip(ka.index.round(3), ka.values))))
print('mismatching orders (keep / pull):', int((~D.keep_fills_equal.astype(bool)).sum()), int((~D.pull_fills_equal.astype(bool)).sum()))
print('equal once summed per timestamp (keep / pull):', agg_ok)
for mode in agg_bad:
    for x in agg_bad[mode][:3]:
        print(mode, 'oid', x[0], 'engine', x[1][:6], 'kp', x[2][:6])
