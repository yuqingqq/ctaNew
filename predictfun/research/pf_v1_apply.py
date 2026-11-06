"""pm_ladder_v1 guard values for a forward day from the FROZEN models (predictfun/freeze/pm_ladder_v1: guard_bid.txt / guard_ask.txt,
params.json thresholds), no refit. Same output as pf_step5_guard_preds.py's fresh-day path with PF_GRID=step4/badfill_info_tx_tk/grid.parquet
PF_ROWS=rows_rl_d2_pk PF_PSTEM=t PF_GUARD_KS=60 (the freeze builder proved its 8-day fit = the frozen models): step5/<D>/preds_t60.parquet =
mid, tau, pred_1000, pred_250 (from preds_c3), v_side = model_side(features) - threshold_c60[side] as float32 on live rows (t_rel < 240 and a
known Predict mid), NaN elsewhere. pf_preds_startknown then writes preds_t60k.
Refuses: model / params hashes differ from the receipt; D is a read day, sealed, or before 11-03. Parity (PF_V1_APPLY_CHECK=1): applied to
20261103 into step5/v1_apply_check/, every column equals step5/20261103/preds_t60.parquet (falsifier: threshold + 1e-6 must differ).
Usage: python3 -m predictfun.research.pf_v1_apply <D> ...   |   PF_V1_APPLY_CHECK=1 python3 -m predictfun.research.pf_v1_apply
"""
import hashlib, json, os, sys
import numpy as np, pandas as pd, lightgbm as lgb
from . import pf_io as io
from .pf_model_spec import SETS

FZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'freeze', 'pm_ladder_v1')
MODEL_SHA = {'bid': '0dc09d84bb28e0c0cafac99ee0fa8ff1140cb97ca70571526469f3b42f345e39', 'ask': '6d2d7149e6c556c6a2a63dad89e68ce4ea6113c76400edf9bd4f7a345ea9fade'}
READ = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']; SEALED = ('20261028', '20261029', '20261030')
FE = SETS['ALL']


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def load():
    P = json.load(open(f'{FZ}/params.json'))
    for s in ('bid', 'ask'):
        if sha(f'{FZ}/guard_{s}.txt') != MODEL_SHA[s]:
            sys.exit(f'REFUSED: guard_{s}.txt is not the frozen model')
    if P['guard']['features'] != FE:
        sys.exit('REFUSED: feature order differs from params.json')
    return {s: lgb.Booster(model_file=f'{FZ}/guard_{s}.txt') for s in ('bid', 'ask')}, P['guard']['threshold_c60']


def apply(d, ms, cs):
    r = pd.read_parquet(f'{io.OUT}/step5/{d}/rows_rl_d2_pk.parquet', columns=list(dict.fromkeys(['mid', 'tau', 't_rel', 'pf_mid'] + FE)))
    pr = pd.read_parquet(f'{io.OUT}/step5/{d}/preds_c3.parquet', columns=['mid', 'tau', 'pred_1000', 'pred_250'])
    assert len(pr) == len(r) and (pr.mid.values == r.mid.values).all() and np.allclose(pr.tau.values, r.tau.values), f'{d}: row alignment'
    live = ((r.t_rel < 240) & r.pf_mid.notna()).values; X = r[FE].values.astype(np.float32)
    vh = {s: np.where(live, ms[s].predict(X, num_threads=1), np.nan) for s in ('bid', 'ask')}
    o = pr.copy(); o['v_bid'] = (vh['bid'] - cs['bid']).astype(np.float32); o['v_ask'] = (vh['ask'] - cs['ask']).astype(np.float32)
    rep = dict(rows=int(len(r)), live=int(live.sum()), pulled_share={s: float(np.mean(vh[s][live] <= cs[s])) for s in ('bid', 'ask')})
    return o, rep


def equal(a, b):
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    for c in a.columns:
        x, y = a[c].values, b[c].values
        if x.dtype.kind == 'f':
            if not (np.array_equal(np.isnan(x), np.isnan(y)) and np.array_equal(x[~np.isnan(x)], y[~np.isnan(y)])):
                return False
        elif not np.array_equal(x, y):
            return False
    return True


def main():
    ms, cs = load()
    if os.environ.get('PF_V1_APPLY_CHECK') == '1':
        d = '20261103'; od = f'{io.OUT}/step5/v1_apply_check'; os.makedirs(od, exist_ok=True)
        o, rep = apply(d, ms, cs); o.to_parquet(f'{od}/preds_t60_{d}.parquet', index=False)
        ref = pd.read_parquet(f'{io.OUT}/step5/{d}/preds_t60.parquet')
        if not equal(pd.read_parquet(f'{od}/preds_t60_{d}.parquet'), ref):
            sys.exit(f'REFUSED: the frozen-model apply does not reproduce {d} preds_t60')
        f, _ = apply(d, ms, {'bid': cs['bid'] + 1e-6, 'ask': cs['ask']})
        if equal(f, ref):
            sys.exit('FALSIFIER DID NOT FIRE: a moved threshold reproduced preds_t60')
        io.write_json(f'{od}/check.json', dict(day=d, equal=True, falsifier_fired=True, **rep)); print('apply parity PASS', rep, flush=True)
        return 0
    for d in sys.argv[1:]:
        if d in READ or d in SEALED or d < '20261103':
            sys.exit(f'REFUSED: {d} is a read / sealed / earlier day')
        o, rep = apply(d, ms, cs); o.to_parquet(f'{io.OUT}/step5/{d}/preds_t60.parquet', index=False)
        io.write_json(f'{io.OUT}/step5/{d}/apply_v1.json', rep); print(d, rep, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
