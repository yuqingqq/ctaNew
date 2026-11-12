"""Supersede failed smoke: filtering a heap requires rebuilding its ordering."""
import datetime as dt
import json
import subprocess
import sys
from . import pf_continuation_spec as S

OLD_OUT=S.ROOT/'data/predict_fun/research/continuation_20261112'
NEW_OUT=S.ROOT/'data/predict_fun/research/continuation_20261112_v2'
_progress=S.progress


def configure():
    S.OUT=NEW_OUT;S.REL=S.REL.with_name('continuation_20261112_v2')
    def progress(stage,state,detail):
        _progress(stage,state,detail)
        for base in (S.WORK,S.ROOT):
            for rel,begin,end in [('predictfun/HANDOFF.md','<!-- CONTINUATION_20261112 -->','<!-- /CONTINUATION_20261112 -->'),
                                  ('predictfun/STATUS.yml','# CONTINUATION_20261112','# /CONTINUATION_20261112')]:
                p=base/rel;s=p.read_text();i=s.index(begin);j=s.index(end,i)
                s=s[:i]+s[i:j].replace('reviews/continuation_20261112/','reviews/continuation_20261112_v2/')+s[j:]
                S.atomic(p,s)
    S.progress=progress


def install():
    configure()
    from . import pf_continuation_engine as E
    from .pf_order_policy_engine import replace
    original=E.source
    def corrected(queue='M'):
        return replace(original(queue),"        heapq.heappush(ev, (start, 2, -1, 'D', None))\n",
                       "        heapq.heapify(ev)  # filtered heap arrays do not retain heap order\n        heapq.heappush(ev, (start, 2, -1, 'D', None))\n")
    E.source=corrected
    from . import pf_continuation_build as B
    compare=B.equal_fills
    def diagnostic(actual,expected):
        try:compare(actual,expected)
        except ValueError as exc:
            raise ValueError(f'{exc}; actual_n={len(actual)} expected_n={len(expected)}; actual_first={actual[:3]}; expected_first={expected[:3]}') from exc
    B.equal_fills=diagnostic


def declare():
    configure();d=json.loads((OLD_OUT/'declaration.json').read_text())
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S.WORK,text=True).strip()
    for name in ('pf_continuation_v2.py','continuation_driver_v2.py','tests/test_continuation_v2.py'):
        rel='predictfun/research/'+name
        if subprocess.check_output(['git','show',commit+':'+rel],cwd=S.WORK)!=(S.WORK/rel).read_bytes():raise ValueError('commit correction before declaration')
        d['source_sha'][rel]=S.sha(S.WORK/rel)
    d.update(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=commit,version=2,
        supersedes=dict(path=str(OLD_OUT/'declaration.json'),sha=S.sha(OLD_OUT/'declaration.json'),status='FAILED_KEEP_FORK_PARITY',failure_log_sha=S.sha(OLD_OUT/'logs/smoke_20261023.log')),
        correction='After filtering the copied event heap to the fork horizon, heapify it before injecting the fork decision. Filtering an array representation does not preserve parent/child heap order; otherwise an out-of-order event or prematurely stopped drain corrupts KEEP labels. No population, model, feature, horizon, threshold or policy change. Original failed artifacts and source are preserved.',
        additional_controls='Known valid heap with interleaved pruned subtrees becomes invalid under filtering; original behavior must fail sorted pop order, corrected heap must restore it. Original synthetic engine-fork and moment controls rerun under corrected source; full real-data neutral and every KEEP fork parity remain mandatory.')
    for p,h in d['source_sha'].items():
        if S.sha(S.WORK/p)!=h:raise ValueError('changed frozen source')
    S.write('declaration.json',d);S.progress('declaration_v2','complete','v1 smoke failed KEEP parity; v2 re-heapifies filtered fork events. Design unchanged, failed run preserved.')


if __name__=='__main__':
    if sys.argv[1]=='declare':declare()
    else:
        install();S.check()
        if sys.argv[1]=='build':
            from .pf_continuation_build import run
            run(sys.argv[2],len(sys.argv)>3 and sys.argv[3]=='smoke')
        elif sys.argv[1]=='fit':
            from .pf_continuation_model import run
            run()
        else:raise ValueError('unknown v2 command')
