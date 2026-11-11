"""Equivalent causal feature cache; v3 changes runtime, not the experiment design."""
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path('/home/yuqing/ctaNew')
OUT=ROOT/'data/predict_fun/research/default_pull_20261111_v3'
GOLD=ROOT/'data/predict_fun/research/default_pull_20261111_v2'


def install():
    import numpy as np
    from .pf_default_pull_v2 import configure
    from .pf_own_order_warmup_v3 import CausalWindow
    from .pf_model_spec import SETS
    from . import pf_order_policy_hooks as hooks
    S=configure();S.OUT=OUT;S.REVIEW=HERE.parent/'reviews/default_pull_20261111_v3'

    class FastWindow(CausalWindow):
        def __init__(self,*args,**kw):
            super().__init__(*args,**kw);self._depth={};self._minimum={}

        def depth(self,side,p):
            key=(side,p)
            if key not in self._depth:
                pp,qq=(self.bp,self.bs) if side==0 else (self.ap,self.ass)
                match=pp==p;has=match.any(axis=1);ix=match.argmax(axis=1)
                a=np.where(has,qq[np.arange(len(pp)),ix],0.).astype(float)
                outside=(pp[:,-1]>=0)&((p<pp[:,-1]) if side==0 else (p>pp[:,-1]))
                # Original shown() returns a matching level first, then applies the 5-level bound.
                a[outside&~has]=np.nan
                self._depth[key]=a
            return self._depth[key]

        def minimum(self,side,p,birth,i,depth):
            key=(side,p,birth);old=self._minimum.get(key)
            if old is not None and i>=old[0]:
                end,value=old;extra=depth[end+1:i+1];extra=extra[np.isfinite(extra)]
                if len(extra):value=min(value,float(extra.min()))
            else:
                a=depth[birth:i+1];a=a[np.isfinite(a)]
                value=float(a.min()) if len(a) else np.nan
            if old is None or i>=old[0]:self._minimum[key]=(i,value)
            return value

        def features(self,t,o,net,other,v,line):
            j=self.row(t);i=self.index(t);birth=self.index(o['t0'])
            if j<0 or i<0 or birth<0 or birth>i:return None
            depth=self.depth(o['side'],o['pt']);now=float(depth[i]);initial=float(depth[birth])
            if not np.isfinite(now) or not np.isfinite(initial):return None
            minimum=initial if birth==i else self.minimum(o['side'],o['pt'],birth,i,depth)
            if not np.isfinite(minimum):return None
            x=dict(zip(SETS['ALL'],map(float,self.x[j])))
            x.update(side=float(o['side']),order_price=float(o['pt']),remaining=float(o['size']-o.get('kq',0)),inventory=float(net),
                order_age_ms=t-o['t0'],other_reserved=float(other),guard_v=float(v),guard_margin=float(v-line),seed=float(o.get('_seed',False)),
                calm=float(self.calms[j]),activity_ratio=float(self.ratio[j]),d_bnimb250=float(self.db[j]),d_pfimb250=float(self.dp[j]),
                queue_initial=initial,queue_now=now,queue_min=minimum,queue_progress=initial-minimum)
            return x

    hooks.Window=FastWindow
    return S,FastWindow


def declare():
    import datetime as dt
    import json
    import subprocess
    from . import pf_default_pull_spec as S
    old=GOLD/'declaration.json';d=json.loads(old.read_text())
    if OUT.exists():raise ValueError('v3 output exists')
    golden=GOLD/'runs/build'/S.DAYS[0];g=json.loads((golden/'checks.json').read_text())
    if g['status']!='PASS' or not g['reference_exact']:raise ValueError('slow golden day not verified')
    golden_hashes={}
    for r in g['variants']:
        for kind,h in r['hashes'].items():
            p=golden/kind/S.stem(r['mode'])
            if S.sha(p)!=h:raise ValueError('golden artifact changed')
            golden_hashes[str(p.relative_to(golden))]=h
        p=golden/'generations'/f"{r['mode']}.parquet"
        if S.sha(p)!=r['generation_sha']:raise ValueError('golden generations changed')
        golden_hashes[str(p.relative_to(golden))]=r['generation_sha']
    for kind in ('independent_windows.parquet','checks.json','feature_check.json'):
        golden_hashes[kind]=S.sha(golden/kind)
    d.update(version=3,as_of=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        runtime_golden_sha=golden_hashes,
        runtime_supersedes=dict(declaration_sha=S.sha(old),path=str(old)),
        runtime_change='Cache displayed depth at each price, and a causal observed minimum per order birth. All slices end at the decision-known frame; backward label queries recompute only their own past slice. No feature definition, action, label, model, threshold, null or replay timing changes. Full288-window Oct23 output must match the completed slow v2 build exactly (generation rows, independent windows, engine windows/fills/requests) before proceeding to more days or fitting.',
        runtime_controls='Compare every feature against the original CausalWindow on monotone and backward queries, changed/absent levels and unknown depth; future-mutation positive and deliberately global-minimum falsifier. Original13 tests and v2 late-fill reservation controls retained. Direct cached-feature identity audits retained for all runs.')
    for p in (HERE/'pf_default_pull_fast.py',HERE/'run_default_pull_v3.sh'):
        d['source_sha'][str(p.relative_to(HERE.parents[1]))]=S.sha(p)
    S.write(OUT/'declaration.json',d);S.write(HERE.parent/'reviews/default_pull_20261111_v3/declaration.json',d)
    print('V3_DECLARED',flush=True)


def controls(FastWindow):
    import numpy as np
    from .pf_own_order_warmup_v3 import CausalWindow
    from .pf_order_policy_inputs import FEATURES
    from .tests.test_order_policy import fixture,order
    from .pf_default_pull_v2 import controls as old_controls
    old_controls()
    inputs,frames,times=fixture()
    for k,f in enumerate(frames):
        f['b1_s']=float(50+k%17);f['a2_s']=float(70-k%23)
        if k%13==0:f['b1_p']=50;f['b2_p']=49
        if k%19==0:f['b5_p']=-1
        if k%29==0:f['a1_s']=np.nan
    a=CausalWindow(inputs,1,frames,times,0.);b=FastWindow(inputs,1,frames,times,0.)
    rng=np.random.default_rng(31);n=0
    for side in (0,1):
        for p in range(43,58):
            for birth in (0.,500.,5000.):
                o=order(side=side,pt=p,t0=birth)
                queries=list(np.arange(max(birth,500),15000,250.))+list(rng.uniform(max(birth,500),14999,50))
                for t in queries:
                    x=a.features(t,o,1,3,.1,.05);y=b.features(t,o,1,3,.1,.05)
                    if (x is None)!=(y is None):raise ValueError('fast feature missing-status mismatch')
                    if x is not None:np.testing.assert_allclose([x[k] for k in FEATURES],[y[k] for k in FEATURES],rtol=0,atol=0,equal_nan=True)
                    n+=1
    inputs,frames,times=fixture();o=order()
    before=FastWindow(inputs,1,frames,times,0.).features(6500,o,0,0,.1,.05)
    for f in frames[131:]:f['b1_s']=1.
    changed=FastWindow(inputs,1,frames,times,0.);after=changed.features(6500,o,0,0,.1,.05)
    np.testing.assert_allclose([before[k] for k in FEATURES],[after[k] for k in FEATURES],rtol=0,atol=0,equal_nan=True)
    if np.nanmin(changed.depth(0,49))==after['queue_min']:raise ValueError('global future-minimum falsifier did not fire')
    print('FAST_FEATURE_CONTROLS_PASS',n,'queries',flush=True)


def parity(S):
    import datetime as dt
    import json
    import pandas as pd
    from .pf_order_policy_verify import exact
    day=S.DAYS[0];old=GOLD/'runs/build'/day;new=S.folder(day,'build')
    c=json.loads((old/'checks.json').read_text());n=json.loads((new/'checks.json').read_text())
    declaration=S.check()
    for p,h in declaration['runtime_golden_sha'].items():
        if S.sha(old/p)!=h:raise ValueError('frozen golden identity changed')
    if c['status']!='PASS' or n['status']!='PASS' or not c['reference_exact'] or not n['reference_exact']:raise ValueError('unverified parity source')
    matches=[]
    for mode in S.GROUPS['build']:
        for kind in ('win','fills','requests','generations'):
            rel=Path(kind)/(f'{mode}.parquet' if kind=='generations' else S.stem(mode))
            exact(pd.read_parquet(new/rel),pd.read_parquet(old/rel));matches.append(dict(path=str(rel),old_sha=S.sha(old/rel),new_sha=S.sha(new/rel)))
    exact(pd.read_parquet(new/'independent_windows.parquet'),pd.read_parquet(old/'independent_windows.parquet'))
    # The same comparator must reject a changed artifact, not just accept equality.
    good=pd.DataFrame({'value':[1.]});bad=pd.DataFrame({'value':[2.]})
    try:exact(good,bad)
    except AssertionError:pass
    else:raise ValueError('parity falsifier failed')
    r=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),status='PASS',day=day,windows=288,arms=2,matches=matches,
        old_independent_sha=S.sha(old/'independent_windows.parquet'),new_independent_sha=S.sha(new/'independent_windows.parquet'),
        old_checks_sha=S.sha(old/'checks.json'),new_checks_sha=S.sha(new/'checks.json'),declaration_sha=S.sha(S.OUT/'declaration.json'),
        comparator='Exact DataFrame identities, all columns and rows; changed-value artifact is refused.')
    S.write(S.OUT/'runtime_parity.json',r);S.write(S.REVIEW/'runtime_parity.json',r)
    print('FULL_DAY_RUNTIME_PARITY_PASS',flush=True)


def main():
    if sys.argv[1]=='declare':declare();return
    S,FastWindow=install();S.check();command=sys.argv[1]
    if command=='controls':controls(FastWindow)
    elif command=='run':
        import gc
        from .pf_default_pull_engine import run
        from .pf_default_pull_v2 import feature_check
        day,group=sys.argv[2:]
        if group!='smoke' and not (group=='build' and day==S.DAYS[0]) and not (S.OUT/'runtime_parity.json').exists():raise ValueError('runtime parity gate incomplete')
        run(day,group);gc.collect();feature_check(day,group,S)
        if group=='build' and day==S.DAYS[0]:parity(S)
    elif command=='fit':
        from .pf_default_pull_fit import run
        run()
    elif command=='report':
        from .pf_default_pull_report import run
        run(sys.argv[2])
    else:raise ValueError('unknown fast command')


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1] in ('progress','slot'):
        import importlib.util
        spec=importlib.util.spec_from_file_location('progress',HERE/'pf_default_pull_progress.py')
        P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P);P.OUT=OUT
        if sys.argv[1]=='slot':
            import json
            b=P.busy();print(json.dumps(b));sys.exit(bool(b))
        P.update(*sys.argv[2:])
    else:main()
