"""Honor engine event priority when trade and decision clocks are equal."""
import datetime as dt
import hashlib
import json
import subprocess
import sys
from . import pf_continuation_spec as S
from . import pf_continuation_v2 as V2

PRIOR=S.ROOT/'data/predict_fun/research/continuation_20261112_v2'
OUT=S.ROOT/'data/predict_fun/research/continuation_20261112_v3'
_progress=S.progress


def configure():
    S.OUT=OUT;S.REL=S.REL.with_name('continuation_20261112_v3')
    def progress(stage,state,detail):
        _progress(stage,state,detail)
        for base in (S.WORK,S.ROOT):
            for rel,begin,end in [('predictfun/HANDOFF.md','<!-- CONTINUATION_20261112 -->','<!-- /CONTINUATION_20261112 -->'),
                                  ('predictfun/STATUS.yml','# CONTINUATION_20261112','# /CONTINUATION_20261112')]:
                p=base/rel;s=p.read_text();i=s.index(begin);j=s.index(end,i)
                s=s[:i]+s[i:j].replace('reviews/continuation_20261112/','reviews/continuation_20261112_v3/')+s[j:]
                S.atomic(p,s)
    S.progress=progress


def builder_source():
    from .pf_order_policy_engine import replace
    text=(S.WORK/'predictfun/research/pf_continuation_build.py').read_text()
    text=replace(text,"actual=[f for f in rawfills if t<=f[0]<t+h]", "actual=[f for f in rawfills if t<f[0]<t+h]")
    text=replace(text,"a=self.trades[np.searchsorted(self.tt,t):np.searchsorted(self.tt,t+h)]", "a=self.trades[np.searchsorted(self.tt,t,side='right'):np.searchsorted(self.tt,t+h)]")
    return text


def install():
    V2.install();configure()
    from . import pf_continuation_build as B
    text=builder_source();exec(compile(text,'continuation_builder_v3','exec'),B.__dict__)
    B.generated_source_sha=hashlib.sha256(text.encode()).hexdigest()


def declare():
    configure();d=json.loads((PRIOR/'declaration.json').read_text());commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S.WORK,text=True).strip()
    for name in ('pf_continuation_v3.py','continuation_driver_v3.py','tests/test_continuation_v3.py'):
        rel='predictfun/research/'+name
        if subprocess.check_output(['git','show',commit+':'+rel],cwd=S.WORK)!=(S.WORK/rel).read_bytes():raise ValueError('commit correction before declaration')
        d['source_sha'][rel]=S.sha(S.WORK/rel)
    d.update(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=commit,version=3,
        supersedes=dict(path=str(PRIOR/'declaration.json'),sha=S.sha(PRIOR/'declaration.json'),status='SMOKE_PASSED_FULL_BUILD_STOPPED_FOR_CLOCK_REVIEW',smoke_sha=S.sha(PRIOR/'smoke_20261023.json')),
        correction_v3='At equal timestamps all tape trades (priority1) precede decisions (priority2). Fork fills already start after that event prefix. Align the neutral-future comparator and incoming-flow label by using trades strictly after the decision time, instead of including equal-time predecision trades. No policy/model/horizon/population change. v2 full build was stopped before completion/fit; no result selected.',
        generated_builder_sha=hashlib.sha256(builder_source().encode()).hexdigest(),
        additional_controls_v3='A planted trade exactly at the probe timestamp is already in the engine prefix: it must be excluded from future own fills and forward incoming volume. The uncorrected comparator must reject that same case. Re-heapify, alias, cancellation, re-entry, shared outcome and known-bad controls retained.')
    for p,h in d['source_sha'].items():
        if S.sha(S.WORK/p)!=h:raise ValueError('changed source')
    S.write('declaration.json',d);S.progress('declaration_v3','complete','v2 smoke passed; v3 excludes predecision trades sharing the decision clock from future flow and reconciliation. Failed/stopped artifacts preserved.')


if __name__=='__main__':
    if sys.argv[1]=='declare':declare()
    else:
        install();d=S.check()
        from . import pf_continuation_build as B
        if B.generated_source_sha!=d['generated_builder_sha']:raise ValueError('generated builder identity')
        if sys.argv[1]=='build':B.run(sys.argv[2],len(sys.argv)>3 and sys.argv[3]=='smoke')
        elif sys.argv[1]=='fit':
            from .pf_continuation_model import run
            run()
        else:raise ValueError('unknown v3 command')
