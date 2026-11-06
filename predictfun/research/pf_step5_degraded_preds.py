"""Degraded-anchor predictions for the quality sweep (declared in STATUS.yml 2026-11-04 01:57Z).
pred = beta (y + e): y = the realised move (y1000 / y250 from rows_rl_d2), e = a 1 s moving average (20 rows of 50 ms) of white
noise per market, scaled so that corr(pred, y)^2 = R2; beta = var(y) / (var(y) + var(e)) (the calibrated shrink), var over the
live rows (t_rel < 240, known mid) of the day. Same rows as preds_o3 / preds_c3 (asserted). Upper-bound family, never a candidate.
Usage: PF_DEG_R2=0.15,0.3,0.5,0.7,0.9 python3 -m predictfun.research.pf_step5_degraded_preds
"""
import os
import numpy as np, pandas as pd
from . import pf_io as io

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']
LEVELS = [float(x) for x in os.environ.get('PF_DEG_R2', '0.15,0.3,0.5,0.7,0.9').split(',')]


def smooth_noise(mids, rng, n=20):
    """Per market: moving average over n rows of iid N(0,1), rescaled to unit variance (rows are in market order)."""
    z = rng.standard_normal(len(mids)); out = np.empty(len(mids))
    starts = np.r_[0, np.flatnonzero(mids[1:] != mids[:-1]) + 1, len(mids)]
    for a, b in zip(starts[:-1], starts[1:]):
        c = np.cumsum(np.r_[np.zeros(n), z[a:b]]); out[a:b] = (c[n:] - c[:-n]) / n
    return out / out.std()


rep = {}
for d in DAYS:
    r = pd.read_parquet(f'{io.OUT}/step5/{d}/rows_rl_d2.parquet', columns=['mid', 'tau', 't_rel', 'pf_mid', 'y250', 'y1000'])
    a = pd.read_parquet(f"{io.OUT}/step5/{d}/{'preds_c3' if d >= '20261101' else 'preds_o3'}.parquet", columns=['mid', 'tau'])
    assert len(a) == len(r) and (a.mid.values == r.mid.values).all() and np.allclose(a.tau.values, r.tau.values), f'{d}: row alignment'
    live = ((r.t_rel < 240) & r.pf_mid.notna()).values
    for R2 in LEVELS:
        rng = np.random.default_rng(int(d) * 100 + int(round(R2 * 100)))
        o = pd.DataFrame(dict(mid=r.mid.values, tau=r.tau.values))
        for col, dst in (('y1000', 'pred_1000'), ('y250', 'pred_250')):
            y = r[col].values.astype(float); vy = np.nanvar(y[live]); ve = vy * (1.0 / R2 - 1.0)
            e = smooth_noise(r.mid.values, rng) * np.sqrt(ve); beta = vy / (vy + ve)
            o[dst] = (beta * (y + e)).astype(np.float32)
        ok = live & np.isfinite(o.pred_1000.values) & np.isfinite(r.y1000.values)
        yy = r.y1000.values[ok]; pp = o.pred_1000.values[ok]
        rep[f'{d}|{R2}'] = dict(r2=round(float(1 - np.sum((yy - pp) ** 2) / np.sum((yy - yy.mean()) ** 2)), 3), corr2=round(float(np.corrcoef(pp, yy)[0, 1] ** 2), 3))
        o.to_parquet(f'{io.OUT}/step5/{d}/preds_deg{int(round(R2 * 100)):02d}.parquet', index=False)
    print(d, {k.split('|')[1]: v for k, v in rep.items() if k.startswith(d)}, flush=True)
    del r, a
io.write_json(f'{io.OUT}/step5/degraded_preds_report.json', rep)
