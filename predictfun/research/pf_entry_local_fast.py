"""Exact single-row LightGBM replay acceleration; no model/policy changes.

Uses the official LGBM_BoosterPredictForMatSingleRowFast API, normal prediction,
all trees, float32, one thread. Reference: LightGBM C API documentation.
Frozen v3 model/data artifacts are symlinked read-only by convention into a new
output namespace. All stages verify hashes through the frozen driver.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import sys
from unittest import mock
import unittest

import lightgbm as lgb
from lightgbm.basic import _LIB, _safe_call
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import pf_entry_local as P

DATA=Path(P.K.IO)/'entry_local_20261115_v3'
OUT=Path(P.K.IO)/'entry_local_20261115_v4'
ORIGINAL=lgb.Booster


class Fast:
    def __init__(self,booster):
        self.booster=booster; self.reference=booster.predict; self.n=booster.num_feature()
        if booster.num_model_per_iteration()!=1: raise ValueError('single-output regression required')
        self.handle=C.c_void_p(); self.length=C.c_int64(); self.buffer=np.empty(1,np.float64); self.calls=0
        _safe_call(_LIB.LGBM_BoosterPredictForMatSingleRowFastInit(booster._handle,C.c_int(0),C.c_int(0),C.c_int(-1),
            C.c_int(0),C.c_int32(self.n),C.c_char_p(b'num_threads=1'),C.byref(self.handle)))

    def predict(self,data,*args,**kwargs):
        a=np.asarray(data)
        if args or kwargs or a.ndim!=2 or a.shape[0]!=1:
            return self.reference(data,*args,**kwargs)
        if a.shape[1]!=self.n: raise ValueError('feature shape mismatch')
        if not self.handle: raise ValueError('closed predictor')
        a=np.ascontiguousarray(a,dtype=np.float32)
        _safe_call(_LIB.LGBM_BoosterPredictForMatSingleRowFast(self.handle,C.c_void_p(a.ctypes.data),
            C.byref(self.length),C.c_void_p(self.buffer.ctypes.data)))
        if self.length.value!=1: raise ValueError('wrong prediction length')
        self.calls+=1
        return self.buffer.copy()

    def close(self):
        if self.handle:
            _safe_call(_LIB.LGBM_FastConfigFree(self.handle)); self.handle=C.c_void_p()


def equal(a,b):
    np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
    if not np.array_equal(np.asarray(a)>=0,np.asarray(b)>=0): raise ValueError('entry decision sign changed')


class Checks(unittest.TestCase):
    def test_predictor_controls(self):
        rng=np.random.default_rng(7); x=rng.normal(size=(400,3)).astype('float32'); x[::9,1]=np.nan
        y=x[:,0]*2+np.nan_to_num(x[:,1])
        model=lgb.train(dict(objective='regression',num_threads=1,num_leaves=4,min_data_in_leaf=10,verbose=-1),lgb.Dataset(x,y),12)
        fast=Fast(model)
        try:
            expected=model.predict(x); got=np.array([fast.predict(row[None,:])[0] for row in x]); equal(got,expected)
            with self.assertRaises(AssertionError): equal(got+1.,expected)
            with self.assertRaisesRegex(ValueError,'shape'): fast.predict(x[:1,:2])
        finally: fast.close()
        with self.assertRaisesRegex(ValueError,'closed'): fast.predict(x[:1])


def links(out):
    out.mkdir(parents=True,exist_ok=True)
    for name in [*['rows_'+d for d in P.DAYS],*['fit_'+d for d in P.TESTS]]:
        target=DATA/name; link=out/name
        if link.is_symlink():
            if link.resolve()!=target.resolve(): raise ValueError('wrong artifact link')
        elif link.exists(): raise ValueError('refusing non-link artifact reuse')
        else: link.symlink_to(target,target_is_directory=True)


def accelerated(day,out,limit=None,fingerprints=None):
    predictors=[]; original_run=P.run
    def factory(*args,**kwargs):
        b=ORIGINAL(*args,**kwargs); f=Fast(b); predictors.append(f); b.predict=f.predict; return b
    def run(*args,**kwargs):
        r=original_run(*args,**kwargs)
        if fingerprints is not None: fingerprints.append(fingerprint(r))
        return r
    try:
        with mock.patch.object(lgb,'Booster',factory),mock.patch.object(P,'run',run):
            P.replay(day,out,limit)
        P.write(out/f'runtime_{day}.json',dict(asof=P.now(),mode='native_single_row_fast',prediction_calls=sum(x.calls for x in predictors),models=len(predictors)))
    finally:
        for f in predictors: f.booster.predict=f.reference; f.close()


def fingerprint(r):
    return hashlib.sha256(repr((r['fl'],r['policy_decisions'],r['orders'],r['stat'],r['endpos'])).encode()).hexdigest()


def parity(out):
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if not result.wasSuccessful(): raise ValueError('synthetic fast-path checks failed')
    checked=0; model_hashes={}
    # Every saved chronological evaluation row, all 30 saved heads.
    for day in P.TESTS:
        f=P.load(day,DATA); receipt=json.loads((DATA/('fit_'+day)/'receipt.json').read_text())
        for side in (0,1):
            x=f[f.side==side][P.FEATURES].to_numpy(np.float32)
            for target in ('fill_fraction','drift_value','local_value'):
                path=DATA/('fit_'+day)/f'{side}_{target}.txt'
                if P.digest(path)!=receipt['models'][path.name]: raise ValueError('changed model')
                b=ORIGINAL(model_file=str(path));fast=Fast(b)
                try: equal(np.array([fast.predict(row[None,:])[0] for row in x]),b.predict(x))
                finally: fast.close()
                checked+=len(x);model_hashes[str(path)]=P.digest(path)
        P.guard()
    normal=out/'normal';quick=out/'fast';links(normal);links(quick)
    normal_f=[];quick_f=[]; original=P.run
    def capture(*a,**kw):
        r=original(*a,**kw);normal_f.append(fingerprint(r));return r
    with mock.patch.object(P,'run',capture): P.replay(P.TESTS[0],normal,2)
    accelerated(P.TESTS[0],quick,2,quick_f)
    if normal_f!=quick_f or len(normal_f)!=24: raise ValueError('whole-replay action/fill mismatch')
    a=(normal/('replay_'+P.TESTS[0])/'windows.csv').read_bytes()
    b=(quick/('replay_'+P.TESTS[0])/'windows.csv').read_bytes()
    if a!=b: raise ValueError('whole-replay accounting mismatch')
    P.write(out/'parity.json',dict(asof=P.now(),prediction_pairs_checked=checked,model_hashes=model_hashes,
        replay_cases=len(normal_f),replay_fingerprints=normal_f,exact_replay=True,
        maximum_allowed_prediction_error=1e-12,synthetic_controls=True,
        interpretation='Engineering equivalence only; no new model or statistical selection.'))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['selftest','parity','replay','report']);p.add_argument('day',nargs='?');p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    if a.command=='selftest':
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
        if not result.wasSuccessful(): raise SystemExit(1)
    elif a.command=='parity': a.out.mkdir();parity(a.out)
    elif a.command=='replay': links(a.out);accelerated(a.day,a.out)
    else: P.report(a.out)


if __name__=='__main__': main()
