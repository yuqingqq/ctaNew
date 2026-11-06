"""Freeze builder for the trained per-side guard (2026-11-04; user: "freeze a version and tell implementation session to implement the
new strategy"). Fits M1 exactly as pf_step5_guard_preds does for a fresh day (LightGBM, pf_model_spec PARAMS / ROUNDS, v per side
clipped +-10, on the 32 features of pf_model_spec ALL, every row of step4/badfill_info_tx/grid.parquet = the 8 read days: features
with pending increments known per frame, labels on one time per taker transaction - the 2026-11-04 rebuild), saves each
side's model as LightGBM text, and the pull thresholds c(k) = k-quantile of the model's in-sample predictions per side.
Live rule: v_side = model_side(x) - c_side(k); V0 withholds the side when v_side <= 0.
Selftest (refuses on failure): the saved text models, reloaded, reproduce step5/20261103/preds_h<k>.parquet (fit on the same 8 days)
on every live row to float32 equality. Refuses unless this file and the pipeline files are committed and clean at HEAD.
Writes predictfun/frozen/<name>/{model_v_bid.txt, model_v_ask.txt, guard.json}.
Usage: python3 -m predictfun.research.pf_guard_freeze <name> <k>
"""
import sys, os, json, hashlib, subprocess
import numpy as np, pandas as pd, lightgbm as lgb
from . import pf_io as io
from .pf_model_spec import SETS, PARAMS, ROUNDS

NAME, K = sys.argv[1], int(sys.argv[2])
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(f'{HERE}/../..'); OUTD = f'{REPO}/predictfun/frozen/{NAME}'
PIPE = ['pf_guard_freeze.py', 'pf_model_spec.py', 'pf_step5_guard_preds.py', 'pf_badfill_grid_tx.py', 'pf_step5_features_rl.py', 'pf_clock_rl.py',
        'pf_preds_startknown.py', 'pf_step4_live.py', 'pf_io.py', 'pf_step1.py', 'pf_step1_timing.py', 'pf_step1_booktime.py', 'pf_rebuild_report.py']
GRID = f'{io.OUT}/step4/badfill_info_tx/grid.parquet'
git = lambda *a: subprocess.run(['git', '-C', REPO, *a], capture_output=True, text=True, check=True).stdout.strip()
dirty = git('status', '--porcelain', '--', *[f'predictfun/research/{f}' for f in PIPE])
if dirty:
    sys.exit(f'REFUSED: pipeline files not committed / not clean at HEAD:\n{dirty}')
head = git('rev-parse', 'HEAD')
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
if os.path.exists(OUTD):
    sys.exit(f'REFUSED: {OUTD} exists (a frozen artifact is never rebuilt in place; use a new name)')
FE = SETS['ALL']; READ = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']
G = pd.read_parquet(GRID, columns=['day', 'v_bid', 'v_ask'] + FE)
assert sorted(G.day.unique()) == READ, f'grid days {sorted(G.day.unique())}'
os.makedirs(OUTD)
X = G[FE].values.astype(np.float32); thr, files = {}, {}
for side in ('bid', 'ask'):
    ds = lgb.Dataset(X, G[f'v_{side}'].clip(-10, 10).values, params=PARAMS, free_raw_data=True).construct()
    m = lgb.train(PARAMS, ds, ROUNDS); del ds
    thr[side] = float(np.quantile(m.predict(X, num_threads=1), K / 100.0))
    p = f'{OUTD}/model_v_{side}.txt'; m.save_model(p); files[side] = p; del m
del X, G
# selftest: reloaded text models reproduce the 11-03 prediction file
d = '20261103'
r = pd.read_parquet(f'{io.OUT}/step5/{d}/rows_rl_d2_pk.parquet', columns=list(dict.fromkeys(['mid', 'tau', 't_rel', 'pf_mid'] + FE)))
pr = pd.read_parquet(f'{io.OUT}/step5/{d}/preds_h{K}.parquet', columns=['mid', 'tau', 'v_bid', 'v_ask'])
assert len(pr) == len(r) and (pr.mid.values == r.mid.values).all(), 'row alignment'
live = ((r.t_rel < 240) & r.pf_mid.notna()).values; Xl = r.loc[live, FE].values.astype(np.float32); st = {}
for side in ('bid', 'ask'):
    v = (lgb.Booster(model_file=files[side]).predict(Xl, num_threads=1) - thr[side]).astype(np.float32)
    ref = pr[f'v_{side}'].values[live]; st[side] = dict(rows=int(live.sum()), equal=int(np.sum(v == ref)), max_abs_diff=float(np.max(np.abs(v - ref))))
if any(s['equal'] != s['rows'] for s in st.values()):
    sys.exit(f'REFUSED: selftest failed {st}')
rec = dict(name=NAME, built_at_commit=head, pull_k=K, thresholds_c=thr, features=FE, params=PARAMS, rounds=ROUNDS, train_days=READ,
           label='v per side: value of keeping an order at the known best through the next exposure (pf_badfill_info grid), clipped +-10',
           live_rule='v_side = model_side(x) - c_side; V0 withholds the side when v_side <= 0; predictions only for t_rel < 240 s with a valid known mid; '
                     'prediction rounded to float32; stale after 100 ms',
           model_sha256={s: sha(files[s]) for s in files}, pipeline_sha256={f: sha(f'{HERE}/{f}') for f in PIPE},
           grid_sha256=sha(GRID), selftest_20261103=st)
io.write_json(f'{OUTD}/guard.json', rec)
print(json.dumps(rec, indent=1)[:3000])
