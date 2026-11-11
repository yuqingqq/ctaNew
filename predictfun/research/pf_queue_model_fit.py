"""Chronological models, matched-action null and sequential controlled portfolios."""
import datetime as dt
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from .pf_model_spec import PARAMS, ROUNDS
from .pf_queue_model_spec import OUT,REVIEW,DAYS,CONFIGS,ACTIONS,STARTS,SIZE,RISK_C,L90,check,sha,write
from .pf_queue_model_sim import MARKET,QUEUE
from .pf_queue_model_data import admit
from .pf_io import mem_check

ARMS=['no_quote','always_keep','always_cancel','legacy_score','market','queue']


def fit(train,day,columns):
    if train.empty or train.day.max()>=day:raise ValueError('noncausal training split')
    models=[]
    for action in ('keep','rejoin'):
        target=train[action+'_utility'].to_numpy()-train.cancel_utility.to_numpy()
        if not np.isfinite(target).all():raise ValueError('missing training target')
        ds=lgb.Dataset(train[columns],target,params=PARAMS,free_raw_data=True)
        models.append(lgb.train(PARAMS,ds,ROUNDS))
    return models


def choices(rows,models,columns):
    score=np.column_stack([models[0].predict(rows[columns],num_threads=1),np.zeros(len(rows)),models[1].predict(rows[columns],num_threads=1)])
    # Counterfactual rejoin reserves both the still-uncertain old size and new size.
    illegal=~rejoin_allowed(rows)
    score[illegal,2]=-np.inf
    return np.argmax(score,axis=1),score


def rejoin_allowed(rows):
    sg=np.where(rows.side.to_numpy()==0,1.,-1.)
    worst=sg*rows.inventory.to_numpy()+2*rows.known_remaining.to_numpy()
    limit=np.where(rows.episode.to_numpy()>=180,0.,25.)
    return worst<=limit+1e-9


def mask_admission(net,start,side,row):
    return np.isfinite(row.get('keep_q',np.nan)) and admit(net,side,start)


def portfolio(frame,fillframe,arm,models=None,columns=None):
    """Same fixed scheduled admissions; only continuation policy differs. No overlapping episodes."""
    scenario=frame.config.iloc[0];c=CONFIGS[scenario]
    fillmap={k:g.sort_values('t',kind='stable').to_dict('records') for k,g in fillframe.groupby(['id','action'],sort=False)}
    states={int(m):dict(net=0.,cash=0.,square=0.,last=0.,max_net=0.,fills=0,shares=0.,orders=0.,spread=0.,
                       mo1=0.,mo5=0.,mo60=0.,missing_marks=0,through_q=0.,actions=[0,0,0],admitted=0)
            for m in frame.mid.unique()}
    look={(int(r['mid']),int(r['episode']),int(r['side'])):r for r in frame.to_dict('records')}
    selected=[]
    def fs(row,a):return fillmap.get((row['id'],ACTIONS[a]),[])
    for start in STARTS:
        pairs=[]
        for mid,state in states.items():
            rows=[look[(mid,start,s)] for s in (0,1)]
            active=[arm!='no_quote' and mask_admission(state['net'],start,s,r) for s,r in enumerate(rows)]
            pairs.append(dict(mid=mid,rows=rows,active=active,chosen=[0,0],initial=state['net']))
        # Decisions in age order. Each side can use only notices already received from the other side's chosen path.
        for age in (250.,1000.,2000.):
            queries=[];refs=[]
            for pair in pairs:
                for side,row in enumerate(pair['rows']):
                    if not pair['active'][side] or row.get('order_age_ms')!=age or row['decision_status']!='eligible':continue
                    q=dict(row);now=row['decision'];known=pair['initial'];other_reserved=0.
                    for s,other in enumerate(pair['rows']):
                        if not pair['active'][s]:continue
                        notified=sum(f['q'] for f in fs(other,pair['chosen'][s]) if f['t']+c['notice']<=now)
                        known+=(1 if s==0 else -1)*notified
                        if s!=side:other_reserved=max(0.,SIZE-notified)
                    q['inventory']=known;q['other_reserved']=other_reserved
                    queries.append(q);refs.append((pair,side))
            if not queries:continue
            q=pd.DataFrame(queries)
            if arm in ('market','queue'):acts,_=choices(q,models,columns)
            elif arm=='always_cancel':acts=np.ones(len(q),int)
            elif arm=='legacy_score':
                line=np.array([L90[min(3,int(t//60000))] for t in q.decision])
                acts=np.where(q.legacy_v.to_numpy()>line,0,1)
            else:acts=np.zeros(len(q),int)
            for (pair,side),act in zip(refs,acts):pair['chosen'][side]=int(act)
        for pair in pairs:
            state=states[pair['mid']];events=[]
            for side,row in enumerate(pair['rows']):
                if not pair['active'][side]:continue
                action=pair['chosen'][side];name=ACTIONS[action];state['admitted']+=1
                state['actions'][action]+=int(row['decision_status']=='eligible')
                state['orders']+=row[name+'_orders']
                for key in ('spread','mo1','mo5','mo60'):
                    value=row[name+'_'+key]
                    if np.isfinite(value):state[key]+=value
                    else:state['missing_marks']+=1
                state['through_q']+=row[name+'_through_q']
                for f in fs(row,action):
                    events.append((f['t'],side,f['q'],f['price']))
                    selected.append({**f,'mid':pair['mid'],'episode':start,'side':side,'action':name})
            for t,side,q,p in sorted(events):
                if t<state['last']-1e-8:raise ValueError('overlapping episode clock')
                state['square']+=(t-state['last'])/1000*state['net']**2;state['last']=t
                sg=1 if side==0 else -1;state['net']+=sg*q;state['cash']-=sg*q*p
                state['max_net']=max(state['max_net'],abs(state['net']));state['shares']+=q;state['fills']+=1
                if abs(state['net'])>25+1e-7:raise ValueError('portfolio capacity violated')
    results=[]
    for mid,s in states.items():
        row=look[(mid,STARTS[0],0)];s['square']+=(240000-s['last'])/1000*s['net']**2
        results.append(dict(mid=mid,arm=arm,config=scenario,day=row['day'],net=s['net'],cash_c=s['cash'],
            stop_value_c=s['cash']+s['net']*row['stop_mid'] if s['net'] else s['cash'],
            settle_value_c=s['cash']+s['net']*100*row['outcome'] if s['net'] else s['cash'],
            inventory_square_s=s['square'],max_inventory=s['max_net'],terminal_abs=abs(s['net']),
            fills=s['fills'],shares=s['shares'],orders=s['orders'],admitted=s['admitted'],
            mo1_c=s['mo1'],mo5_c=s['mo5'],mo60_c=s['mo60'],spread_c=s['spread'],missing_marks=s['missing_marks'],
            through_q=s['through_q'],keep_actions=s['actions'][0],cancel_actions=s['actions'][1],rejoin_actions=s['actions'][2]))
    return pd.DataFrame(results),pd.DataFrame(selected,columns=['id','action','fill_index','t','q','through','generation','anchor','price','side','mid','episode'])


def bootstrap_day(x):
    x=np.asarray(x,float)
    if len(x)<5 or not np.isfinite(x).all():return None
    rng=np.random.default_rng(20261111);b=x[rng.integers(0,len(x),(1000,len(x)))].mean(axis=1)
    return dict(n_days=len(x),mean=float(x.mean()),low=float(np.quantile(b,.025)),high=float(np.quantile(b,.975)))


def null_test(pred,draws=500):
    if draws<200:raise ValueError('undersampled null')
    rng=np.random.default_rng(20261111);groups=[];observed={}
    for config,g in pred.groupby('config',sort=False):
        y=g[['keep_utility','cancel_utility','rejoin_utility']].to_numpy();a=g.queue_action.to_numpy(dtype=int);b=g.market_action.to_numpy(dtype=int)
        if not np.isfinite(y).all() or ((a<0)|(a>2)|(b<0)|(b>2)).any():raise ValueError('invalid null population')
        if (((a==2)|(b==2))&~rejoin_allowed(g)).any():raise ValueError('illegal null action')
        ref=y[np.arange(len(y)),b].sum();observed[config]=float((y[np.arange(len(y)),a].sum()-ref)/100)
        strata=g.assign(minute=(g.decision//60000).astype(int),hour=(g.ws%86400//3600).astype(int),inventory_sign=np.sign(g.inventory),rejoin_allowed=rejoin_allowed(g)).groupby(['day','side','hour','minute','inventory_sign','rejoin_allowed'],sort=False).indices
        groups.append((config,y,a,ref,list(strata.values())))
    maxima=[]
    for _ in range(draws):
        values=[]
        for _,y,a,ref,parts in groups:
            perm=a.copy()
            for idx in parts:perm[idx]=rng.permutation(perm[idx])
            values.append(float((y[np.arange(len(y)),perm].sum()-ref)/100))
        maxima.append(max(values))
    return dict(draws=draws,seed=20261111,family=len(groups),observed_utility_delta_usd=observed,
        family_p={k:float((1+np.sum(np.asarray(maxima)>=v))/(draws+1)) for k,v in observed.items()},
        null_max_p95=float(np.quantile(maxima,.95)))


def run():
    declaration=check();allrows=[];allfills=[];builds=[]
    for d in DAYS:
        b=json.loads((OUT/f'{d}_build.json').read_text())
        if sha(OUT/f'{d}.parquet')!=b['data_sha'] or sha(OUT/f'{d}_fills.parquet')!=b['fills_sha']:raise ValueError('data changed')
        builds.append(b);allrows.append(pd.read_parquet(OUT/f'{d}.parquet'));allfills.append(pd.read_parquet(OUT/f'{d}_fills.parquet'))
    a=pd.concat(allrows,ignore_index=True);f=pd.concat(allfills,ignore_index=True);del allrows,allfills
    mem_check('queue training inputs')
    if a.id.duplicated().any():raise ValueError('duplicate generation')
    outputs=[];predictions=[];folds=[];importance=[]
    (OUT/'models').mkdir(exist_ok=False)
    for day in DAYS[3:]:
        mem_check('queue fold')
        tr=a[(a.day<day)&(a.config=='L50')&(a.status=='eligible')&a.reference_admitted]
        if len(tr)<500:raise ValueError('insufficient training generations')
        models={}
        for name,columns in [('market',MARKET),('queue',MARKET+QUEUE)]:
            models[name]=fit(tr,day,columns)
            for action,m in zip(('keep','rejoin'),models[name]):
                m.save_model(str(OUT/f'models/{day}_{name}_{action}.txt'))
                importance.append(dict(day=day,model=name,action=action,
                    gain={k:float(v) for k,v in zip(columns,m.feature_importance('gain'))}))
        folds.append(dict(test_day=day,train_days=sorted(tr.day.unique()),train_n=len(tr)))
        for config in CONFIGS:
            rows=a[(a.day==day)&(a.config==config)].copy()
            ids=set(rows.id);ff=f[f.id.isin(ids)]
            neutral=rows[(rows.status=='eligible')&rows.reference_admitted].copy()
            for name,cols in [('market',MARKET),('queue',MARKET+QUEUE)]:
                act,score=choices(neutral,models[name],cols);neutral[name+'_action']=act
                neutral[name+'_keep_score']=score[:,0];neutral[name+'_rejoin_score']=score[:,2]
            predictions.append(neutral)
            for arm in ARMS:
                cols=(MARKET+QUEUE) if arm=='queue' else MARKET
                res,selected=portfolio(rows,ff,arm,models.get(arm),cols)
                outputs.append(res)
                selected.to_parquet(OUT/f'{day}_{config}_{arm}_selected.parquet',index=False)
            print('EVAL',day,config,'train',len(tr),'test',len(neutral),flush=True)
    p=pd.concat(predictions,ignore_index=True);r=pd.concat(outputs,ignore_index=True)
    p.to_parquet(OUT/'predictions.parquet',index=False);r.to_parquet(OUT/'portfolio_windows.parquet',index=False)
    null=null_test(p);summary=[];deltas=[];neutral_summary=[]
    for config,g in p.groupby('config',sort=False):
        y=g[['keep_utility','cancel_utility','rejoin_utility']].to_numpy()
        keep=np.zeros(len(g),int);cancel=np.ones(len(g),int)
        for name,action in [('always_keep',keep),('always_cancel',cancel),('market',g.market_action.to_numpy(int)),('queue',g.queue_action.to_numpy(int))]:
            neutral_summary.append(dict(config=config,arm=name,n=len(g),days=int(g.day.nunique()),
                mean_action_utility_c=float(y[np.arange(len(g)),action].mean()),
                keep=int((action==0).sum()),cancel=int((action==1).sum()),rejoin=int((action==2).sum()),
                keep_filled_generations=int((g.keep_q>0).sum()),unfilled_keep_generations=int((g.keep_q==0).sum())))
    for (config,arm),g in r.groupby(['config','arm'],sort=False):
        summary.append(dict(config=config,arm=arm,days=int(g.day.nunique()),windows=len(g),
            stop_value_usd=float(g.stop_value_c.sum()/100),settle_value_usd=float(g.settle_value_c.sum()/100),
            missing_stop=int(g.stop_value_c.isna().sum()),missing_settlement=int(g.settle_value_c.isna().sum()),
            settlement_carry_usd=float((g.settle_value_c-g.stop_value_c).sum()/100),missing_fill_mark_groups=int(g.missing_marks.sum()),
            mo1_usd=float(g.mo1_c.sum()/100),mo5_usd=float(g.mo5_c.sum()/100),mo60_usd=float(g.mo60_c.sum()/100),
            spread_usd=float(g.spread_c.sum()/100),shares=float(g.shares.sum()),fills=int(g.fills.sum()),orders=int(g.orders.sum()),
            inventory_square_s=float(g.inventory_square_s.sum()),terminal_abs_mean=float(g.terminal_abs.mean()),max_inventory=float(g.max_inventory.max()),
            keep_actions=int(g.keep_actions.sum()),cancel_actions=int(g.cancel_actions.sum()),rejoin_actions=int(g.rejoin_actions.sum())))
    for config in CONFIGS:
        sub=r[r.config==config]
        for ref in ('market','always_keep','always_cancel','legacy_score'):
            paired=sub[sub.arm=='queue'].merge(sub[sub.arm==ref],on=['day','mid'],suffixes=('_q','_b'),validate='one_to_one')
            for metric in ('stop_value_c','settle_value_c','inventory_square_s','terminal_abs'):
                if paired[[metric+'_q',metric+'_b']].isna().any().any():
                    deltas.append(dict(config=config,ref=ref,metric=metric,status='missing_mark'));continue
                x=(paired[metric+'_q']-paired[metric+'_b']).groupby(paired.day).sum()
                scale=100 if metric.endswith('_c') else 1
                deltas.append(dict(config=config,ref=ref,metric=metric,sum=float(x.sum()/scale),days_positive=int((x>0).sum()),
                                   per_day={str(d):float(v/scale) for d,v in x.items()},interval=bootstrap_day(x/scale)))
    report=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),declaration=declaration,builds=builds,folds=folds,
        summary=summary,deltas=deltas,neutral_summary=neutral_summary,feature_importance=importance,
        null=null,prediction_rows=len(p),generation_rows=len(a),
        prediction_sha=sha(OUT/'predictions.parquet'),portfolio_sha=sha(OUT/'portfolio_windows.parquet'),
        limits=['Fixed scheduled5share continuation experiment, not full v3 or live implementation.',
                'Initial placement is held fixed; no learned admission model. Queue estimates are uncertain and fill timing uncalibrated.',
                'Legacy-score benchmark uses prior LODO scores and is not a chronological model validation.',
                'Consumed dates only; model forecasts and bootstrap intervals are exploratory. No promotion.'])
    write(OUT/'report.json',report);write(REVIEW/'report.json',report)
    print('DONE',len(p),'held-out-in-time generation rows',flush=True)


if __name__=='__main__':run()
