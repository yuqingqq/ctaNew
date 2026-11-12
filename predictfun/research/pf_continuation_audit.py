"""Independent publication audit of frozen continuation artifacts; no new fitting."""
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
from . import pf_continuation_spec as S
from .pf_continuation_v3 import configure
configure()


def require_hash(path,expected):
    if S.sha(path)!=expected:raise ValueError('artifact hash mismatch: '+str(path))


def declare():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S.WORK,text=True).strip()
    rel='predictfun/research/pf_continuation_audit.py'
    if subprocess.check_output(['git','show',commit+':'+rel],cwd=S.WORK)!=(S.WORK/rel).read_bytes():raise ValueError('commit audit source first')
    S.write('audit_declaration.json',dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=commit,source_sha=S.sha(__file__),
        experiment_declaration_sha=S.sha(S.OUT/'declaration.json'),design='Artifact audit only: complete action/status and deterministic sampling identities; independently reconstruct every saved baseline and model forecast; compare report metric values to predictions; validate source/input/output hashes, chronology and shared receipts. Synthetic valid hash and known-bad hash refusal. No fits, new candidates, thresholds or changed population.'))


def run():
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    from . import pf_continuation_model as M
    from .pf_order_policy_inputs import MARKET,FEATURES
    from .pf_continuation_report import FAMILIES
    d=S.check();ad=json.loads((S.OUT/'audit_declaration.json').read_text())
    require_hash(__file__,ad['source_sha']);require_hash(S.OUT/'declaration.json',ad['experiment_declaration_sha'])
    require_hash(S.OUT/'declaration.json',S.sha(S.OUT/'declaration.json'))
    try:require_hash(S.OUT/'declaration.json','0'*64)
    except ValueError:pass
    else:raise ValueError('hash refusal failed')
    for p,h in d['input_sha'].items():require_hash(p,h)
    parts=[];populations=[]
    for day in S.DAYS:
        r=json.loads((S.OUT/f'build_{day}.json').read_text());folder=S.OUT/'runs'/day
        if r['status']!='PASS' or not r['neutral_exact'] or r['windows']!=288:raise ValueError('incomplete day')
        for name,h in r['hashes'].items():
            if name in ('win','fills','requests'):
                from .pf_default_pull_spec import stem
                path=folder/name/stem('default')
            else:path=folder/name
            require_hash(path,h)
        a=pd.read_parquet(folder/'generations/default.parquet');p=pd.read_parquet(folder/'paired.parquet');ledger=pd.read_parquet(folder/'population.parquet')
        eligible=a.decision.notna();sample=a.id.map(S.selected);full=a.decision+30000<=a.ws+240000
        expected=set(a.loc[eligible&sample&full,'id'])
        if expected!=set(p.id) or set(a.id)!=set(ledger.id) or p.id.duplicated().any() or ledger.id.duplicated().any():raise ValueError('action sampling identity')
        if len(p)!=r['keep_forks_reconciled'] or len(a)!=r['generations']:raise ValueError('reconciled count')
        if not p[[f'keep_reconciled_{h}' for h in S.HORIZONS]].all().all():raise ValueError('unreconciled action')
        good=p[p.status.eq('eligible')].copy();M.tensor(good);parts.append(good)
        populations.append(dict(day=day,generations=len(a),sampled_actions=len(p),usable=len(good),statuses=ledger.status.value_counts().to_dict(),paired_statuses=p.status.value_counts().to_dict()))
    a=pd.concat(parts,ignore_index=True);fit=json.loads((S.OUT/'fit.json').read_text());r=json.loads((S.OUT/'result.json').read_text())
    require_hash(S.OUT/'fit.json',r['fit_sha']);require_hash(S.OUT/'predictions.parquet',r['predictions_sha'])
    pred=pd.read_parquet(S.OUT/'predictions.parquet');rows=0
    for fold in fit['folds']:
        day=fold['day'];calday=fold['calibration_day'];core=a[a.day<calday];cal=a[a.day.eq(calday)];test=a[a.day.eq(day)]
        if max(core.day)>=calday or calday>=day or [len(core),len(cal),len(test)]!=[fold['core_n'],fold['calibration_n'],fold['test_n']]:raise ValueError('fold chronology/population')
        saved=pred[pred.day.eq(day)].set_index('id').loc[test.id].reset_index()
        np.testing.assert_allclose(test[FEATURES],saved[FEATURES],atol=0,rtol=0,equal_nan=True)
        _,_,prior=M.tensor(core)
        forecasts={'baseline':M.flatten(np.repeat(prior[None],len(test),axis=0))}
        for family,features in [('market',MARKET),('queue',FEATURES)]:
            info=fold['models'][family];path=S.OUT/'models'/f'{day}_{family}.txt';tablepath=S.OUT/'models'/f'{day}_{family}.npz'
            require_hash(path,info['model_sha']);require_hash(tablepath,info['table_sha'])
            with np.load(tablepath) as table:
                np.testing.assert_allclose(table['prior'],prior,atol=1e-12,rtol=0)
                raw=M.predict(lgb.Booster(model_file=str(path)),features,test,table['table'],table['prior'])
            forecasts[family+'_raw']=M.flatten(raw);forecasts[family+'_cal']=M.flatten(M.reweight(raw,info['calibrator']['weights']))
        for family,values in forecasts.items():
            for key,value in values.items():np.testing.assert_allclose(value,saved[family+'_'+key],atol=1e-8,rtol=0)
            rows+=len(test)
    if rows!=len(pred)*len(FAMILIES) or len(pred)!=r['n'] or pred.id.duplicated().any():raise ValueError('forecast action identity')
    for metric in r['metrics']:
        target=metric['target'];family=metric['family'];y=pred[target].to_numpy();x=pred[family+'_'+target].to_numpy()
        np.testing.assert_allclose([metric['actual_mean'],metric['predicted_mean'],metric['mse']],
                                   [y.mean(),x.mean(),((x-y)**2).mean()],atol=1e-8,rtol=1e-12)
    copies=0
    for path in (S.WORK/S.REL).rglob('*.json'):
        rel=path.relative_to(S.WORK/S.REL)
        if path.read_bytes()!=(S.OUT/rel).read_bytes() or path.read_bytes()!=(S.ROOT/S.REL/rel).read_bytes():raise ValueError('shared receipt mismatch')
        copies+=1
    S.write('audit.json',dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),status='PASS',audit_declaration_sha=S.sha(S.OUT/'audit_declaration.json'),
        source_sha=S.sha(__file__),result_sha=S.sha(S.OUT/'result.json'),input_hashes_verified=len(d['input_sha']),shared_receipts_verified=copies,
        forecast_rows_reconstructed=rows,actions=len(pred),models=sum(len(f['models']) for f in fit['folds']),metrics_recomputed=len(r['metrics']),populations=populations,
        scope='Exact saved artifact reconstruction and population/metric audit; not untouched or live validation. No new fitted model or policy variant.'))
    S.progress('artifact_audit','complete','Full source/input/output, population, forecast reconstruction and metric audit passed; interpretation still requires human review.')

if __name__=='__main__':
    declare() if sys.argv[1]=='declare' else run()
