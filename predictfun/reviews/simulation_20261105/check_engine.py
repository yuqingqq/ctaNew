"""Synthetic review: execute repository functions without importing the data-reading module.

A successful run confirms the documented defects still reproduce; it does not
certify correct simulation. No market data or simulation output is read.
"""
import ast, bisect, collections, heapq, itertools, math, json, hashlib, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

ROOT = Path('/home/yuqing/ctaNew/predictfun/research')

def load(name):
    p = ROOT / name
    src = p.read_text()
    tree = ast.parse(src)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    ns = dict(np=np, pd=pd, bisect=bisect, collections=collections, heapq=heapq,
              itertools=itertools, math=math, SIZE=5., V2=True, SPLIT=True,
              L_O=22., L_FILL=100., L_PL=40., L_CA=22., L_ANS=44.,
              LEGACY_INTENT=False, QACT=False, G2V={}, PRED={}, PRED250={}, PREDVB={}, PREDVA={},
              CK=SimpleNamespace(L_PENDCANCEL=95.), LINES={'zero':[0.]*4,'inf':[float('inf')]*4})
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(p), 'exec'), ns)
    ns['reg'] = pd.DataFrame([dict(mid=1, ws=0., outcome='Up')]).set_index('mid')
    return ns, hashlib.sha256(src.encode()).hexdigest()

def frame(t, bid, ask=52):
    f = dict(t_ms=float(t))
    for s in 'ba':
        for i in range(1,6):
            f[f'{s}{i}_p'] = -1
            f[f'{s}{i}_s'] = 0.
    f.update(b1_p=bid, b1_s=0., a1_p=ask, a1_s=0.)
    return f

report = {}
for file in ('pf_step4_live.py','pf_step4_ladder.py'):
    ns, sha = load(file)
    def run(bids, trade_t=60., trade_p=48):
        ns['LT_BY'] = {1:pd.DataFrame([dict(t_ms=trade_t, side=0, ptick=trade_p, shares=5., how=1, exec_s=0)])}
        return ns['run_window'](1, [frame(-1,bids[0]),frame(10,bids[1])],
                                [0.,10.], {'XNA','G5','S1'}, 'JOIN', 'mid', 5.)
    up = run([49,50]); down = run([50,49]); stable = run([50,50])
    assert stable['fills'][0][3] == 50, 'positive control: unchanged 50 bid must fill at 50'
    assert down['fills'][0][3] == 50, 'positive control: older better-priced bid must trade first'
    assert up['fills'][0][3] == 49, 'counterexample no longer reproduces (inspect new bytes)'
    report[file] = dict(sha256=sha, older_worse_bid_first=up['fills'],
                        older_better_bid_control=down['fills'], unchanged_bid_control=stable['fills'])
    # Missing guard: a V0 arm globally loaded with predictions for a different market.
    ns['PREDVB']={2:(np.array([0.]),np.array([-1.]))}
    ns['PREDVA']={2:(np.array([0.]),np.array([-1.]))}
    ns['LT_BY']={1:pd.DataFrame([dict(t_ms=60.,side=0,ptick=48,shares=5.,how=1,exec_s=0)])}
    missing = ns['run_window'](1,[frame(-1,50),frame(10,50)],[0.,10.],{'XNA','G5','S1','V0'},'JOIN','mid',5.)
    assert missing['n_fills'] == 1, 'missing-market guard counterexample no longer reproduces'
    ns['PREDVB']={1:(np.array([0.,10.]),np.array([-1.,-1.]))}
    ns['PREDVA']={1:(np.array([0.,10.]),np.array([-1.,-1.]))}
    guarded = ns['run_window'](1,[frame(-1,50),frame(10,50)],[0.,10.],{'XNA','G5','S1','V0'},'JOIN','mid',5.)
    assert guarded['n_fills'] == 0, 'positive control: negative V0 must prevent fills'
    report[file].update(missing_guard_fills=missing['n_fills'],negative_guard_control_fills=guarded['n_fills'])
    ns['LT_BY']={1:pd.DataFrame([dict(t_ms=160.,side=0,ptick=48,shares=5.,how=1,exec_s=0)])}
    stale=ns['run_window'](1,[frame(-1,50),frame(10,50)],[0.,10.],{'XNA','G5','S1','V0'},'JOIN','mid',5.)
    assert stale['n_fills'] == 1, 'stale-guard counterexample no longer reproduces'
    report[file]['stale_negative_guard_fills'] = stale['n_fills']
    try:
        ns['run_window'](1,[frame(-1,50)],[0.],{'G5','S1'},'JOIN','mid',5.)
    except SystemExit:
        report[file]['known_bad_split_without_XNA_refused'] = True
    else:
        raise AssertionError('known-bad unsupported split/amend configuration was accepted')
    if file == 'pf_step4_ladder.py':
        ns['LINES']['test'] = [0.]*4
        ns['PREDVB']={1:(np.array([0.,10.]),np.array([1.,1.]))}
        ns['PREDVA']={1:(np.array([0.,10.]),np.array([1.,1.]))}
        ns['LT_BY']={}
        witness=[]
        def trace(f, event, arg):
            if f.f_code.co_name == 'run_window' and event == 'line':
                l=f.f_locals
                if all(k in l for k in ('t','orders','zombies')):
                    t=l['t']; oo=[o for o in l['orders'].values() if o is not None]
                    oo += [o for zs in l['zombies'].values() for o in zs if o['t_end']>t and o['rem']>0]
                    bids=[o['pt'] for o in oo if o['side']==0]; asks=[o['pt'] for o in oo if o['side']==1]
                    if bids and asks and max(bids)>=min(asks):
                        witness.append((t,max(bids),min(asks)))
            return trace
        sys.settrace(trace)
        crossed=ns['run_window'](1,[frame(-1,49,51),frame(10,50,52)],[0.,10.],{'XNA','G5','S1','IMtest'},'JOIN','mid',5.)
        sys.settrace(None)
        assert witness, 'own-cross witness no longer reproduces'
        assert crossed['live_ctr'].get('own_cross',0)==0, 'invariant now detects the counterexample'
        report[file].update(actual_cross_witness=list(dict.fromkeys(witness)),reported_own_cross=crossed['live_ctr'].get('own_cross',0))

Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
