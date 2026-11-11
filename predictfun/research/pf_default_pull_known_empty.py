"""v4: distinguish an observed empty price level from unavailable history."""
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path('/home/yuqing/ctaNew')
OUT=ROOT/'data/predict_fun/research/default_pull_20261111_v4'
PRIOR=ROOT/'data/predict_fun/research/default_pull_20261111_v3'


def install():
    import numpy as np
    from .pf_default_pull_fast import install as cached_install
    from . import pf_default_pull_hooks as H
    S,Window=cached_install();S.OUT=OUT;S.REVIEW=HERE.parent/'reviews/default_pull_20261111_v4'
    original=H.thin_ratio
    def known_empty(w,t,side,p):
        i=w.index(t);j=w.index(t-250)
        if i>=0 and j>=0:
            now=w.shown(i,side,p);old=w.shown(j,side,p)
            if np.isfinite(now) and now>=0 and np.isfinite(old) and old==0:
                # No historical volume existed to thin. 1 is the neutral thinning indicator,
                # not a claim that current volume equals historical volume (division by0).
                return 1.
        return original(w,t,side,p)
    H.thin_ratio=known_empty
    return S,Window,original


def declare():
    import datetime as dt
    import json
    import subprocess
    from . import pf_default_pull_spec as S
    old=PRIOR/'declaration.json';d=json.loads(old.read_text())
    if OUT.exists():raise ValueError('v4 output exists')
    parity=PRIOR/'runtime_parity.json';p=json.loads(parity.read_text())
    if p['status']!='PASS':raise ValueError('cached feature implementation not verified')
    d.update(version=4,as_of=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        semantic_supersedes=dict(path=str(old),declaration_sha=S.sha(old),reason='Observed historical depth0 was treated as unavailable; this unintentionally delayed entry at newly formed levels and truncated local continuation labels.'),
        known_empty_correction='If past and current same-price depth are observed, nonnegative and the past depth is0, the thinning indicator is neutral1 (no volume existed to thin). This is a categorical no-thinning sentinel, not a mathematical volume ratio. Missing/stale books, out-of-depth observations, negative/nonfinite sizes remain unknown and still withdraw. Same-price thinning from positive depth is unchanged. All eight arms retain their original rules; only this false unknown status is corrected before any model fit.',
        cache_equivalence='The v3 cached feature implementation is unchanged and has prior exact full-day parity with the original implementation. Its receipt is inherited as implementation evidence only; v4 policy outputs intentionally differ due to the known-empty correction.',
        inherited_runtime_parity=dict(path=str(parity),sha=S.sha(parity)),
        additional_controls='A known-new best price with historical observed depth0 must permit a default quote immediately. Planting the old thin_ratio must suppress that same quote. Known0-to0 is not thinning; positive-to-zero still triggers; missing old history and outside-depth inputs remain unknown. Original controls, reservation controls,9180 feature-cache equivalences and direct per-run feature audits retained.')
    d['multiplicity'].update(learned_policies_tested_before_known_empty_fix=0,legacy_neutral_policy_versions_observed=1)
    for name in ['pf_default_pull_known_empty.py','run_default_pull_v4.sh','pf_default_pull_completion.py',
                 'pf_default_pull_completion_v4.py','pf_default_pull_plot_v4.py']:
        q=HERE/name;d['source_sha'][str(q.relative_to(HERE.parents[1]))]=S.sha(q)
    S.write(OUT/'declaration.json',d);S.write(HERE.parent/'reviews/default_pull_20261111_v4/declaration.json',d)
    inherited=dict(as_of=d['as_of'],status='PASS',kind='inherited_cache_implementation_equivalence',
        inherited_path=str(parity),inherited_sha=S.sha(parity),policy_output_equality_claim=False,
        scope='Original versus cached observed features, verified before v4. This does not claim equal v3/v4 policies after the known-empty fix.')
    S.write(OUT/'runtime_parity.json',inherited);S.write(HERE.parent/'reviews/default_pull_20261111_v4/runtime_parity.json',inherited)
    print('V4_DECLARED',flush=True)


def controls(S,Window,original):
    import datetime as dt
    import types
    import numpy as np
    from .pf_default_pull_fast import controls as cached_controls
    from .tests.test_order_policy import fixture
    from . import pf_default_pull_hooks as H
    cached_controls(Window)
    inputs,frames,times=fixture();frames[130]['b1_p']=50.;frames[130]['b1_s']=10.;frames[130]['b2_p']=49.
    parent=types.SimpleNamespace(inputs=inputs,day='20261023',timing='X',phase='build',models={},windows=[],save=lambda m,r:None)
    h=H.Hook(parent,'default',1,0.,frames,times,[],[],lambda t:(False,False),lambda t,s:.01,lambda t:.05)
    known={i:None for i in range(4)};fl={i:[] for i in known}
    fixed=h.targets(6500.,[],known,fl,[],0.)[0]
    if fixed!=(50.,5.):raise ValueError('known new level not quotable')
    if H.thin_ratio(h.w,6500.,0,50.)!=1.:raise ValueError('known empty thinning status')
    if H.thin_ratio(h.w,6400.,0,50.)!=1.:raise ValueError('known0-to0 should not thin')
    if not np.isnan(H.thin_ratio(h.w,100.,0,49.)):raise ValueError('missing history accepted')
    if not np.isnan(H.thin_ratio(h.w,6500.,0,40.)):raise ValueError('outside depth accepted')
    corrected=H.thin_ratio
    try:
        H.thin_ratio=original;bad=h.targets(6500.,[],known,fl,[],0.)[0]
        if bad!=(None,0.):raise ValueError('known-bad admission-delay falsifier failed')
    finally:H.thin_ratio=corrected
    # Existing positive-size thinning semantics are unchanged, including a disappearance.
    inputs,f,t=fixture();f[130]['b1_s']=0.;w=Window(inputs,1,f,t,0.)
    if H.thin_ratio(w,6500.,0,49.)!=0.:raise ValueError('actual thinning no longer detected')
    S.write(S.OUT/'known_empty_controls.json',dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),status='PASS',
        declaration_sha=S.sha(S.OUT/'declaration.json'),source_sha=S.sha(__file__),
        new_level_quote=list(fixed),planted_old_behavior=list(bad),missing_history_refused=True,outside_depth_refused=True,
        zero_to_zero_neutral=True,positive_to_zero_thinning=True,prior_controls='Original13, late-fill reservation regressions and9180 exact cached-feature queries pass.'))
    print('KNOWN_EMPTY_CONTROLS_PASS',flush=True)


def main():
    if sys.argv[1]=='declare':declare();return
    S,Window,original=install();S.check();command=sys.argv[1]
    if command=='controls':controls(S,Window,original)
    elif command=='run':
        import gc
        from .pf_default_pull_engine import run
        from .pf_default_pull_v2 import feature_check
        if not (S.OUT/'known_empty_controls.json').exists():raise ValueError('known-empty control gate incomplete')
        day,group=sys.argv[2:];run(day,group);gc.collect();feature_check(day,group,S)
    elif command=='fit':
        from .pf_default_pull_fit import run
        run()
    elif command=='report':
        from .pf_default_pull_report import run
        run(sys.argv[2])
    elif command=='audit':
        from .pf_default_pull_completion_v4 import run
        run()
    elif command=='plot':
        from .pf_default_pull_plot_v4 import run
        run()
    else:raise ValueError('unknown v4 command')


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
