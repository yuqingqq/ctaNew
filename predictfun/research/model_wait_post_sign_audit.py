"""Post hoc interpretation of saved score bins and exact PNL sign controls.

No fit, replay, inverted policy, new null or skill claim. All measured cohorts
come from the already frozen five-day report. Controls execute the source AST
of the actual terminal accounting function with scalar search/finite shims.
"""
import ast
import bisect
import datetime
import hashlib
import json
import math
from pathlib import Path
import types

HERE=Path(__file__).resolve().parent
REVIEW=HERE.parent/'reviews/wait_post_diagnosis_20261114/v3'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def controls():
    path=HERE/'pf_wait_post_v2_lane.py'
    fn=next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='portfolio_terminal')
    ns={'np':types.SimpleNamespace(searchsorted=lambda a,v,side='right':bisect.bisect_right(a,v) if side=='right' else bisect.bisect_left(a,v),isfinite=math.isfinite)}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
    cases=[];refused=0
    for side,mark,expected in ((0,55,1.5),(0,45,-1.5),(1,45,1.5),(1,55,-1.5)):
        r=dict(terminal_time=100.,fl=[(side,0,50.,50.,30.)],order_states={(side,0):[dict(ref=False,rem=0.,tr=100.)]},endpos=30. if side==0 else -30.)
        value=ns['portfolio_terminal'](r,[0.,100.],[mark,mark])['pnl']
        if not math.isclose(value,expected,abs_tol=1e-12):raise ValueError('wrong economic sign')
        cases.append(dict(side=side,fill_price=50.,mark=mark,quantity=30.,pnl=value,expected=expected))
        r['endpos']+=1
        try:ns['portfolio_terminal'](r,[0.,100.],[mark,mark])
        except ValueError as exc:
            if 'position differs' not in str(exc):raise
            refused+=1
        else:raise ValueError('corrupt position not refused')
    return dict(source_sha256=digest(path),cases=cases,wrong_position_refusals=refused)


def main():
    raw=REVIEW/'raw_report.json';completion=json.loads((REVIEW/'completion.json').read_text())
    if digest(raw)!=completion['files']['raw_report.json']:raise ValueError('changed source report')
    r=json.loads(raw.read_text());a=r['saved_audit'];cohorts=[];correlations=[]
    for side in (0,1):
        for group in ('negative_forecast','nonnegative_forecast'):
            rows=[x for x in a['score_bins'] if x['split']=='test' and x['side']==side and ((x['bin']<3) if group=='negative_forecast' else (x['bin']>=3))]
            n=sum(x['n'] for x in rows)
            cohorts.append(dict(side=side,group=group,n=n,predicted=sum(x['n']*x['predicted'] for x in rows)/n,actual=sum(x['n']*x['actual'] for x in rows)/n))
        for head in ('post','delta'):
            rows=[dict(day=x['day'],n=x['n_test'],rank_ic=x['heads'][head]['test']['rank_ic']) for x in a['fits'] if x['side']==side]
            correlations.append(dict(side=side,head=head,days=rows,negative_days=sum(x['rank_ic']<0 for x in rows)))
    result=dict(asof=datetime.datetime.now(datetime.timezone.utc).isoformat(),measurements_asof=r['asof'],
                status='complete_descriptive_followup',raw_report_sha256=digest(raw),source_sha256=digest(Path(__file__)),
                n_complete_days=r['n_complete_days'],n_probes=sum(x['n'] for x in cohorts),cohorts=cohorts,rank_correlations=correlations,
                sign_controls=controls(),forward_validation=False,new_candidates=0,
                interpretation='No consistent inverse ranking. Both forecast-sign cohorts lose on average. Positive selected forecasts can reflect optimistic estimation errors when true values are mostly negative. Training fit and later failure support overfitting; noisy targets, missing state and state shift remain plausible contributors, not isolated causal effects.',
                limitations=['Forecast-sign cohorts include all labeled probes; they differ from the selected POST subset, which also requires POST >= WAIT.',
                             'Nonnegative bin includes a zero forecast; no row-level rethresholding is performed.',
                             'No inverted trading policy was fitted or evaluated, and inversion is not established as a remedy.',
                             'Four synthetic accounting sign cases do not exclude every possible feature, label or simulator bug.'])
    with (REVIEW/'sign_followup.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('Saved sign-interpretation followup; four economic sign controls and four bad-position refusals passed')


if __name__=='__main__':main()
