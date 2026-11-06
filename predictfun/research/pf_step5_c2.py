"""Step 5e (confirmation C2, 2026-11-02): ONE model fit on all six read days, applied unchanged to C2's fresh days.

Spec = the model the C2 candidates were chosen with (preds_lodo6, pf_step5_lodo6.py): ALL features, target y1000, pinned
PARAMS / ROUNDS from pf_model_spec; train rows t < 240 s with a valid known mid and valid y250 and y1000, every 2nd row,
target clipped to +-5 c, float32. Only y1000 is fitted (C2's arms use pred_1000 only; pred_250 = NaN).
Memory: the training matrix is preallocated and filled one day at a time (no per-day copies + concatenate).
  fit:   python3 -m predictfun.research.pf_step5_c2 fit 20261023 20261024 20261025 20261026 20261027 20261031
         -> step5/model_c2_y1000.txt + step5/model_c2.json (train days, row counts, model sha256)
  apply: python3 -m predictfun.research.pf_step5_c2 apply <day>
         -> step5/<day>/preds_c2.parquet (mid, tau, pred_250 NaN, pred_1000) + step5/<day>/apply_c2.json (R2, corr)
         refuses unless the model file's sha256 equals the one recorded at fit time.
"""
import sys, json, hashlib
import numpy as np, pandas as pd, pyarrow.parquet as pq, lightgbm as lgb
from . import pf_io as io
from .pf_model_spec import SETS, PARAMS, ROUNDS

FE = SETS['ALL']; MODEL = f'{io.OUT}/step5/model_c2_y1000.txt'; META = f'{io.OUT}/step5/model_c2.json'
READ_DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031']


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def train_mask(r):
    return ((r.t_rel < 240) & r.pf_mid.notna() & r.y250.notna() & r.y1000.notna()).values


def fit(days):
    if sorted(days) != READ_DAYS:
        sys.exit(f'REFUSED: C2 fits on exactly the six read days {READ_DAYS}, got {days}')
    n = {}
    for d in days:                                  # pass 1: training-row counts per day (4 small columns)
        r = pd.read_parquet(f'{io.OUT}/step5/{d}/rows.parquet', columns=['t_rel', 'pf_mid', 'y250', 'y1000'])
        n[d] = int(len(np.flatnonzero(train_mask(r))[::2])); del r
    X = np.empty((sum(n.values()), len(FE)), dtype=np.float32); y = np.empty(len(X), dtype=np.float32); i = 0
    for d in days:                                  # pass 2: fill in place
        r = pd.read_parquet(f'{io.OUT}/step5/{d}/rows.parquet', columns=list(dict.fromkeys(FE + ['t_rel', 'pf_mid', 'y250', 'y1000'])))
        k = np.flatnonzero(train_mask(r))[::2]
        assert len(k) == n[d], (d, len(k), n[d])
        X[i:i + len(k)] = r[FE].values[k].astype(np.float32); y[i:i + len(k)] = r.y1000.values[k].clip(-5, 5); i += len(k)
        del r, k
        print(d, 'rows', n[d], '[mem]', io.rss_mib(), flush=True); io.mem_check(f'fill {d}')
    ds = lgb.Dataset(X, y, params=PARAMS, free_raw_data=True).construct(); del X, y
    io.mem_check('dataset')
    m = lgb.train(PARAMS, ds, ROUNDS); del ds
    m.save_model(MODEL)
    meta = dict(train=days, rows=n, rows_total=sum(n.values()), features=FE, target='y1000 clip +-5', subsample='every 2nd valid row',
                params=PARAMS, rounds=ROUNDS, model=MODEL, model_sha256=sha(MODEL), mem_peak_mib_selfreport=io.rss_mib())
    io.write_json(META, meta); print(json.dumps({k: meta[k] for k in ('rows_total', 'model_sha256')}), flush=True)


def apply(day):
    if day in READ_DAYS or day < '20261101':
        sys.exit(f'REFUSED: {day} is not a C2 day (C2 days are complete UTC days from 20261101 on)')
    meta = json.load(open(META))
    if sha(MODEL) != meta['model_sha256']:
        sys.exit('REFUSED: model file sha256 differs from the one recorded at fit time')
    m = lgb.Booster(model_file=MODEL)
    r = pd.read_parquet(f'{io.OUT}/step5/{day}/rows.parquet', columns=['mid', 'tau'] + list(dict.fromkeys(FE + ['t_rel', 'pf_mid', 'y250', 'y1000'])))
    p = m.predict(r[FE].values.astype(np.float32), num_threads=1).astype(np.float32)
    ok = ((r.t_rel < 240) & r.pf_mid.notna()).values          # same prediction support as preds_lodo6
    pr = r[['mid', 'tau']].copy(); pr['pred_250'] = np.float32(np.nan); pr['pred_1000'] = np.where(ok, p, np.nan).astype(np.float32)
    pr.to_parquet(f'{io.OUT}/step5/{day}/preds_c2.parquet', index=False)
    ev = ok & r.y1000.notna().values; yt = r.y1000.values[ev]; pp = p[ev]
    rep = dict(day=day, model_sha256=meta['model_sha256'], n_pred=int(ok.sum()), n_eval=int(ev.sum()),
               r2=round(float(1 - np.sum((yt - pp) ** 2) / np.sum(yt ** 2)), 4), corr=round(float(np.corrcoef(pp, yt)[0, 1]), 4),
               share_flagged_025=round(float((np.abs(pp) >= 0.25).mean()), 4), mem_peak_mib_selfreport=io.rss_mib())
    io.write_json(f'{io.OUT}/step5/{day}/apply_c2.json', rep); print(rep, flush=True)


if __name__ == '__main__':
    mode, args = sys.argv[1], sys.argv[2:]
    fit(args) if mode == 'fit' else apply(args[0]) if mode == 'apply' else sys.exit(f'unknown mode {mode}')
