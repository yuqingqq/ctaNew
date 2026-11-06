"""Confirmation C3 predictions (2026-11-03): models fit on the six read days only, applied unchanged to fresh days.
  pred_1000 = model_c2_y1000.txt (C2's fit: tokyo20 rows of the six read days, y1000 = model A's spec), checked by sha256
  pred_250  = a y250 model fit on the six read days' Tokyo-clock rows with 2 ms decision (rows_rl_d2), same spec otherwise
Both applied to the fresh day's rows_rl_d2 (Tokyo clock, decision 2 ms).
  fit:   python3 -m predictfun.research.pf_step5_c3_preds fit      -> step5/model_c3_y250.txt + step5/model_c3.json (sha256)
  apply: python3 -m predictfun.research.pf_step5_c3_preds apply <d> -> step5/<d>/preds_c3.parquet + apply_c3.json
"""
import sys, json, hashlib
import numpy as np, pandas as pd, lightgbm as lgb
from . import pf_io as io
from .pf_model_spec import SETS, PARAMS, ROUNDS

READ = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031']
FE = SETS['ALL']; TC = ['t_rel', 'pf_mid', 'y250', 'y1000']
M1000 = f'{io.OUT}/step5/model_c2_y1000.txt'; M250 = f'{io.OUT}/step5/model_c3_y250.txt'; META = f'{io.OUT}/step5/model_c3.json'


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def mask(r):
    return ((r.t_rel < 240) & r.pf_mid.notna() & r.y250.notna() & r.y1000.notna()).values


def fit():
    n = {}
    for d in READ:
        r = pd.read_parquet(f'{io.OUT}/step5/{d}/rows_rl_d2.parquet', columns=TC); n[d] = len(np.flatnonzero(mask(r))[::2]); del r
    X = np.empty((sum(n.values()), len(FE)), np.float32); y = np.empty(len(X), np.float32); i = 0
    for d in READ:
        r = pd.read_parquet(f'{io.OUT}/step5/{d}/rows_rl_d2.parquet', columns=list(dict.fromkeys(FE + TC)))
        k = np.flatnonzero(mask(r))[::2]
        X[i:i + len(k)] = r[FE].values[k].astype(np.float32); y[i:i + len(k)] = r.y250.values[k].clip(-5, 5); i += len(k); del r, k
        io.mem_check(f'fill {d}')
    ds = lgb.Dataset(X, y, params=PARAMS, free_raw_data=True).construct(); del X, y
    m = lgb.train(PARAMS, ds, ROUNDS); del ds
    m.save_model(M250)
    meta = dict(train=READ, rows=n, y250_model=M250, y250_sha256=sha(M250), y1000_model=M1000, y1000_sha256=sha(M1000),
                y1000_expected_sha256='be6e916e04e7852f9201c18a7a750e086fde01b8272e4bc133cb505a59455b87')
    if meta['y1000_sha256'] != meta['y1000_expected_sha256']:
        sys.exit('REFUSED: model_c2_y1000.txt is not the C2 model')
    io.write_json(META, meta); print(json.dumps({k: meta[k] for k in ('y250_sha256', 'y1000_sha256')}), flush=True)


def apply(d):
    if d in READ or d < '20261101':
        sys.exit(f'REFUSED: {d} is not a fresh day')
    meta = json.load(open(META))
    if sha(M250) != meta['y250_sha256'] or sha(M1000) != meta['y1000_expected_sha256']:
        sys.exit('REFUSED: a model file differs from the one recorded at fit time')
    r = pd.read_parquet(f'{io.OUT}/step5/{d}/rows_rl_d2.parquet', columns=['mid', 'tau'] + list(dict.fromkeys(FE + TC)))
    X = r[FE].values.astype(np.float32); ok = ((r.t_rel < 240) & r.pf_mid.notna()).values
    p1 = lgb.Booster(model_file=M1000).predict(X, num_threads=1).astype(np.float32)
    p2 = lgb.Booster(model_file=M250).predict(X, num_threads=1).astype(np.float32)
    out = r[['mid', 'tau']].copy(); out['pred_250'] = np.where(ok, p2, np.nan).astype(np.float32); out['pred_1000'] = np.where(ok, p1, np.nan).astype(np.float32)
    out.to_parquet(f'{io.OUT}/step5/{d}/preds_c3.parquet', index=False)
    rep = dict(day=d, y250_sha256=meta['y250_sha256'], y1000_sha256=meta['y1000_expected_sha256'])
    for c, p in (('y1000', p1), ('y250', p2)):
        ev = ok & r[c].notna().values; yt = r[c].values[ev]; pp = p[ev]
        rep[c] = dict(n=int(ev.sum()), r2=round(float(1 - np.sum((yt - pp) ** 2) / np.sum(yt ** 2)), 4), corr=round(float(np.corrcoef(pp, yt)[0, 1]), 4))
    io.write_json(f'{io.OUT}/step5/{d}/apply_c3.json', rep); print(rep, flush=True)


if __name__ == '__main__':
    fit() if sys.argv[1] == 'fit' else apply(sys.argv[2]) if sys.argv[1] == 'apply' else sys.exit('usage: fit | apply <day>')
