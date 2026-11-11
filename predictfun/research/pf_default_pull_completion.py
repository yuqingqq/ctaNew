"""Final artifact audit, independent score reconstruction and true resting-time union.

Run only after the serial v2 driver finishes. Does not change a policy or receipt.
"""
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from .pf_default_pull_v2 import configure
from .pf_order_policy_inputs import FEATURES,MARKET


def need_hash(path,wanted):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    if h.hexdigest()!=wanted:raise ValueError('artifact hash mismatch: '+str(path))


def scores(actual,wanted):
    actual=np.asarray(actual,float);wanted=np.asarray(wanted,float)
    if actual.shape!=wanted.shape or not np.isfinite(actual).all() or not np.isfinite(wanted).all():
        raise ValueError('score shape/finite identity')
    if not np.allclose(actual,wanted,rtol=0,atol=1e-10):raise ValueError('saved action score mismatch')


def union_ms(intervals):
    end=-np.inf;total=0.
    for a,b in sorted(intervals):
        if not np.isfinite(a+b) or b<a:raise ValueError('invalid order lifetime interval')
        total+=max(0.,b-max(a,end));end=max(end,b)
    return total


def controls():
    scores([1.,2.],[1.,2.])
    try:scores([1.,2.],[1.,3.])
    except ValueError:pass
    else:raise ValueError('bad-score falsifier failed')
    if union_ms([(0.,10.),(5.,15.),(15.,17.)])!=17:raise ValueError('presence positive control')
    if union_ms([(0.,10.),(5.,15.)])==20:raise ValueError('double-counted presence falsifier failed')
    try:union_ms([(2.,1.)])
    except ValueError:pass
    else:raise ValueError('invalid lifetime refusal failed')
    need_hash(__file__,hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:need_hash(__file__,'0'*64)
    except ValueError:pass
    else:raise ValueError('changed artifact refusal failed')


def run():
    S=configure();declaration=S.check();controls()
    report=json.loads((S.OUT/'comparison_final.json').read_text())
    fit=json.loads((S.OUT/'fit.json').read_text())
    if report['stage']!='final' or report['days']!=S.TEST or report['validation'] is not False:
        raise ValueError('wrong final cohort/status')
    need_hash(S.OUT/'declaration.json',report['declaration_sha'])
    expected={('X',m) for m in S.MODES}|{('stress',m) for m in S.MODES}
    if {(r['scenario'],r['mode']) for r in report['table']}!=expected or len(report['table'])!=16:
        raise ValueError('missing full portfolio arm/scenario')
    for r in report['table']:
        if r['windows']!=1728 or r['marked_windows']+r['marked_missing']!=1728:raise ValueError('window/exclusion population')
        if not np.isclose(r['marked']+r['carry_common'],r['settlement_common'],rtol=0,atol=1e-8):raise ValueError('carry decomposition')
    from .pf_default_pull_engine import source
    from .pf_default_pull_verify import actions
    checks=[];score_checks=[];presence=[];window_rows={};fit_counts={};feature_decisions=0
    for group,days in [('smoke',[S.DAYS[0]]),('build',S.DAYS),('eval',S.TEST),('stress',S.TEST)]:
        for day in days:
            out=S.folder(day,group);c=json.loads((out/'checks.json').read_text());f=json.loads((out/'feature_check.json').read_text())
            generated=json.loads((out/'source.json').read_text());timing='stress' if group=='stress' else 'X'
            wanted=hashlib.sha256(source(S.TIMINGS[timing]['queue']).encode()).hexdigest()
            if generated['generated_sha']!=wanted or generated['timing']!=S.TIMINGS[timing]:raise ValueError('generated engine identity')
            if c['status']!='PASS' or f['status']!='PASS':raise ValueError('failed checks')
            if group in ('build','smoke') and not c['reference_exact']:raise ValueError('missing exact reference parity')
            need_hash(out/'checks.json',f['checks_sha']);need_hash(out/'independent_windows.parquet',c['independent_sha'])
            cache=S.ROOT/f'data/predict_fun/research/step5/{day}/rows_rl_d2_pk.parquet';need_hash(cache,f['cache_sha'])
            if {r['mode'] for r in c['variants']}!=set(S.GROUPS[group]):raise ValueError('missing checked arm')
            ind=pd.read_parquet(out/'independent_windows.parquet')
            for mode in S.GROUPS[group]:
                receipt=next(r for r in c['variants'] if r['mode']==mode)
                for kind,h in receipt['hashes'].items():need_hash(out/kind/S.stem(mode),h)
                p=out/'generations'/f'{mode}.parquet';need_hash(p,receipt['generation_sha']);g=pd.read_parquet(p);actions(g)
                fr=next(r for r in f['rows'] if r['mode']==mode)
                if fr['generation_sha']!=receipt['generation_sha'] or fr['decisions']!=int(g.order_age_ms.notna().sum()):raise ValueError('feature population identity')
                feature_decisions+=fr['decisions']
                if group=='build' and mode=='default':fit_counts[day]=int((g.status.eq('eligible')&g.label.notna()).sum())
                w=ind[ind['mode']==mode]
                if mode!='reference' and w.max_inventory.max()>10+1e-6:raise ValueError('inventory cap')
                if group!='smoke' and day in S.TEST:window_rows.setdefault((timing,mode),[]).append(w)
                if mode in ('market','queue','queue_cost'):
                    which='market' if mode=='market' else 'queue';cols=MARKET if which=='market' else FEATURES
                    fold=next(r for r in fit['folds'] if r['day']==day);model=S.OUT/'models'/f'{day}_{which}.txt';need_hash(model,fold['models_sha'][which])
                    m=lgb.Booster(model_file=str(model));a=g[g.score.notna()]
                    if len(a):
                        scores(m.predict(a[cols],num_threads=1),a.score.to_numpy())
                        threshold=-.01*a.remaining if mode=='queue_cost' else 0.
                        if not np.array_equal(a.pull.eq(1).to_numpy(),(a.score<threshold).to_numpy()):raise ValueError('policy threshold/action mismatch')
                        pulls=a[a.pull.eq(1)]
                        if not np.allclose(pulls.first_cancel,pulls.decision+S.TIMINGS[timing]['cancel'],rtol=0,atol=1e-7):raise ValueError('pull request/effectiveness identity')
                    score_checks.append(dict(day=day,scenario=timing,mode=mode,scored_generations=len(a),pulls=int(a.pull.sum()),generation_sha=receipt['generation_sha'],model_sha=fold['models_sha'][which]))
                if group!='smoke' and day in S.TEST:
                    total=0.
                    for (mid,side),gg in g[g.life_ms.gt(0)].groupby(['mid','side'],sort=False):
                        total+=union_ms(zip(gg.t_act.to_numpy(),(gg.t_act+gg.life_ms).to_numpy()))
                    frac=total/(len(w)*240100*2)
                    if not -1e-12<=frac<=1+1e-12:raise ValueError('presence outside range')
                    presence.append(dict(day=day,scenario=timing,mode=mode,windows=len(w),resting_union_ms=total,side_presence_fraction=frac,
                        aggregate_order_ms=float(g.life_ms.sum())))
            checks.append(dict(group=group,day=day,checks_sha=S.sha(out/'checks.json'),feature_sha=S.sha(out/'feature_check.json'),source_sha=S.sha(out/'source.json')))
    for fold in fit['folds']:
        if fold['train_days']!=[d for d in S.DAYS if d<fold['day']]:raise ValueError('training chronology')
        if fold['train_n']!=sum(fit_counts[d] for d in fold['train_days']) or fold['test_n']!=fit_counts[fold['day']]:raise ValueError('model action population')
    paired=[]
    for scenario in ('X','stress'):
        for mode in S.MODES:
            for baseline in ('reference','matched_guard','default'):
                if mode==baseline:continue
                a=pd.concat(window_rows[scenario,mode],ignore_index=True);b=pd.concat(window_rows[scenario,baseline],ignore_index=True)
                z=a.merge(b,on=['day','mid'],suffixes=('_a','_b'),validate='one_to_one');z=z[z.marked_strict_a.notna()&z.marked_strict_b.notna()]
                counts=z.groupby('day').size().reindex(S.TEST,fill_value=0)
                if not counts.gt(0).all():raise ValueError('day interval includes a day with no paired marks')
                key=f'{scenario}:{mode}-{baseline}';r=report['comparisons'][key]
                if len(z)!=r['common_mark_windows'] or not np.isclose((z.marked_strict_a-z.marked_strict_b).sum(),r['marked_delta'],atol=1e-8,rtol=0):raise ValueError('paired report identity')
                paired.append(dict(comparison=key,common_marks_by_day=counts.to_dict()))
    for k,r in report['diagnostics'].items():
        if r['draws']!=500 or r['n']!=sum(fit_counts[d] for d in S.TEST):raise ValueError('null population/draw identity')
    for p,h in declaration['input_sha'].items():need_hash(p,h)
    stages=['controls','smoke']+[f'build_{d}' for d in S.DAYS]+['fit']+[f'eval_{d}' for d in S.TEST]+['primary_report']+[f'stress_{d}' for d in S.TEST]+['final_report']
    for stage in stages:
        done=S.OUT/'logs'/f'{stage}.done';line=done.read_text().strip().split(maxsplit=1)
        if Path(line[1])!=S.OUT/'logs'/f'{stage}.log':raise ValueError('stage log identity')
        need_hash(line[1],line[0])
    result=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),status='PASS',source_sha=S.sha(__file__),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S.HERE,text=True).strip(),
        comparison_sha=S.sha(S.OUT/'comparison_final.json'),declaration_sha=S.sha(S.OUT/'declaration.json'),
        stages=len(stages),checked_runs=len(checks),feature_decisions=feature_decisions,checks=checks,score_checks=score_checks,
        presence=presence,paired_population_checks=paired,inputs_rehashed=len(declaration['input_sha']),
        note='Original report rest_side_fraction is aggregate order-time divided by side-time and can overlap during cancellation. Use side_presence_fraction here for union of simulated live order intervals; neither measures actual live venue uptime.',
        controls='Correct scores/hash/presence pass; altered score/hash, overlap double-counting and reversed interval are detected.')
    S.write(S.OUT/'completion_audit.json',result);S.write(S.REVIEW/'completion_audit.json',result)
    print('COMPLETION_AUDIT_PASS',len(checks),'runs',sum(x['scored_generations'] for x in score_checks),'policy scores',flush=True)


if __name__=='__main__':run()
