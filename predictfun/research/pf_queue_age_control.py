"""Pre-result age-matched baseline: isolate queue estimates from remaining order lifetime.

Addendum to the fixed-entry experiment; original artifacts stay frozen. Reuse its
queue candidate and all generation labels; fit only a third, age-matched baseline.
The original market-only model excludes age, so its comparison identifies the
joint benefit of age and queue estimates. This comparison removes that ambiguity.
"""
import datetime as dt
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from . import pf_queue_model_spec as S
from . import pf_queue_model_fit as F
from .pf_queue_model_sim import MARKET, QUEUE
from .pf_queue_model_verify import verify_selection,verify_window
from .pf_io import mem_check

OUT=S.ROOT/'data/predict_fun/research/queue_age_control_20261111'
REVIEW=Path(__file__).resolve().parents[1]/'reviews/queue_age_control_20261111'
COLS=MARKET+['order_age_ms']


def check_columns(columns):
    if set(columns)&set(QUEUE)!= {'order_age_ms'}:raise ValueError('age baseline feature contamination')


def declare():
    original=S.check()
    if OUT.exists():raise ValueError('refuse overwrite')
    check_columns(COLS)
    for bad in (MARKET,COLS+['queue_progress']):
        try:check_columns(bad)
        except ValueError:pass
        else:raise ValueError('feature-checker falsifier failed')
    hashes={}
    for day in S.DAYS:
        if json.loads((S.OUT/f'{day}_build.json').read_text())['missing_registry_windows']:raise ValueError('incomplete UTC day')
        for suffix in ('.parquet','_fills.parquet','_build.json'):
            p=S.OUT/f'{day}{suffix}';hashes[p.name]=S.sha(p)
    d=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
           source_sha=S.sha(__file__),original_declaration_sha=S.sha(S.OUT/'declaration.json'),input_sha=hashes,
           design='Add order_age_ms to the market-only baseline before inspecting original PnL results. Same labels, chronological folds, pinned parameters, portfolio, null500, day intervals, and timing scenarios. Reuse the unchanged original queue candidate; no candidate selection or new tuning.',
           purpose='Queue versus market_age isolates incremental queue size/progress information from age and remaining lease. Original queue versus market is joint order-state value.',
           multiplicity=dict(feature_sets_total=3,candidate_policies=1,comparators=2,scenarios=3),
           limits=original['protections'],original_source_commit=original['source_commit'],columns=COLS)
    S.write(OUT/'declaration.json',d);S.write(REVIEW/'declaration.json',d)
    print('AGE_CONTROL_DECLARED',flush=True)


def run():
    d=json.loads((OUT/'declaration.json').read_text());S.check()
    if S.sha(__file__)!=d['source_sha'] or S.sha(S.OUT/'declaration.json')!=d['original_declaration_sha']:raise ValueError('source changed')
    for name,h in d['input_sha'].items():
        if S.sha(S.OUT/name)!=h:raise ValueError('generation artifact changed')
    verification=json.loads((S.OUT/'verification.json').read_text())
    if verification['status']!='PASS':raise ValueError('original verifier did not pass')
    original=json.loads((S.OUT/'report.json').read_text())
    for key,name in [('prediction_sha','predictions.parquet'),('portfolio_sha','portfolio_windows.parquet')]:
        if S.sha(S.OUT/name)!=original[key]:raise ValueError('original result changed')
    a=pd.concat([pd.read_parquet(S.OUT/f'{day}.parquet') for day in S.DAYS],ignore_index=True)
    p=pd.read_parquet(S.OUT/'predictions.parquet');base=pd.read_parquet(S.OUT/'portfolio_windows.parquet')
    results=[];predictions=[];folds=[];checks=0;selected_hashes={}
    (OUT/'models').mkdir(exist_ok=False)
    for day in S.DAYS[3:]:
        mem_check('queue age control')
        tr=a[(a.day<day)&(a.config=='L50')&(a.status=='eligible')&a.reference_admitted]
        models=F.fit(tr,day,COLS)
        for action,m in zip(('keep','rejoin'),models):m.save_model(str(OUT/f'models/{day}_{action}.txt'))
        folds.append(dict(day=day,train_days=sorted(tr.day.unique()),train_n=len(tr)))
        fills=pd.read_parquet(S.OUT/f'{day}_fills.parquet')
        for config in S.CONFIGS:
            rows=a[(a.day==day)&(a.config==config)]
            neutral=p[(p.day==day)&(p.config==config)].copy()
            acts,scores=F.choices(neutral,models,COLS)
            neutral['original_market_action']=neutral.market_action
            neutral['market_action']=acts;neutral['market_keep_score']=scores[:,0];neutral['market_rejoin_score']=scores[:,2]
            predictions.append(neutral)
            rr,chosen=F.portfolio(rows,fills[fills.id.isin(set(rows.id))],'market',models,COLS)
            rr['arm']='market_age';results.append(rr)
            verify_selection(chosen,fills);identity=rows.drop_duplicates('mid').set_index('mid')
            groups={m:g for m,g in chosen.groupby('mid',sort=False)}
            for r in rr.to_dict('records'):
                ident=identity.loc[r['mid']]
                verify_window(r,groups.get(r['mid'],chosen.iloc[:0]),ident.stop_mid,ident.outcome);checks+=1
            path=OUT/f'{day}_{config}_selected.parquet';chosen.to_parquet(path,index=False);selected_hashes[path.name]=S.sha(path)
            print('AGE_CONTROL',day,config,'train',len(tr),'test',len(neutral),flush=True)
    r=pd.concat(results,ignore_index=True);pred=pd.concat(predictions,ignore_index=True)
    r.to_parquet(OUT/'portfolio_windows.parquet',index=False);pred.to_parquet(OUT/'predictions.parquet',index=False)
    null=F.null_test(pred);summary=[];deltas=[];neutral_summary=[]
    for config,g in r.groupby('config',sort=False):
        summary.append(dict(config=config,arm='market_age',windows=len(g),days=int(g.day.nunique()),
            stop_value_usd=float(g.stop_value_c.sum()/100),settle_value_usd=float(g.settle_value_c.sum()/100),
            missing_stop=int(g.stop_value_c.isna().sum()),missing_settle=int(g.settle_value_c.isna().sum()),
            settlement_carry_usd=float((g.settle_value_c-g.stop_value_c).sum()/100),
            inventory_square_s=float(g.inventory_square_s.sum()),terminal_abs_mean=float(g.terminal_abs.mean()),
            shares=float(g.shares.sum()),orders=int(g.orders.sum()),fills=int(g.fills.sum()),
            spread_usd=float(g.spread_c.sum()/100),mo1_usd=float(g.mo1_c.sum()/100),mo5_usd=float(g.mo5_c.sum()/100),mo60_usd=float(g.mo60_c.sum()/100),
            missing_mark_groups=int(g.missing_marks.sum())))
        pair=base[(base.config==config)&(base.arm=='queue')].merge(g,on=['day','mid'],suffixes=('_q','_b'),validate='one_to_one')
        for metric in ('stop_value_c','settle_value_c','inventory_square_s','terminal_abs'):
            if pair[[metric+'_q',metric+'_b']].isna().any().any():
                deltas.append(dict(config=config,metric=metric,status='missing_mark'));continue
            x=(pair[metric+'_q']-pair[metric+'_b']).groupby(pair.day).sum();scale=100 if metric.endswith('_c') else 1
            deltas.append(dict(config=config,metric=metric,sum=float(x.sum()/scale),days_positive=int((x>0).sum()),
                per_day={str(k):float(v/scale) for k,v in x.items()},interval=F.bootstrap_day(x/scale)))
        npop=pred[pred.config==config];y=npop[['keep_utility','cancel_utility','rejoin_utility']].to_numpy()
        for arm in ('queue','market'):
            act=npop[arm+'_action'].to_numpy(int)
            neutral_summary.append(dict(config=config,arm='market_age' if arm=='market' else arm,n=len(npop),
                utility_c_mean=float(y[np.arange(len(y)),act].mean()),keep=int((act==0).sum()),cancel=int((act==1).sum()),rejoin=int((act==2).sum())))
    report=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),declaration=d,folds=folds,summary=summary,deltas=deltas,
        neutral_summary=neutral_summary,null=null,verified_windows=checks,selected_sha=selected_hashes,
        portfolio_sha=S.sha(OUT/'portfolio_windows.parquet'),prediction_sha=S.sha(OUT/'predictions.parquet'),
        limits='Consumed dates, uncalibrated simulated queues, fixed-entry continuation only. Missing marks are counts; partial sums are not complete PnL. No deployment claim.')
    S.write(OUT/'report.json',report);S.write(REVIEW/'report.json',report)
    print('AGE_CONTROL_VERIFIED',checks,'windows',flush=True)


if __name__=='__main__':
    import sys
    if sys.argv[1]=='declare':declare()
    elif sys.argv[1]=='run':run()
    else:raise ValueError('declare or run')
