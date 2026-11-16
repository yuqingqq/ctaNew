"""First chronological fold readout; frozen predictions, no model/policy changes."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pf_flow_price_common as C
import pf_flow_price_model as M
import pf_flow_price_report as R
import pf_flow_price_check as CHECK
import lightgbm as lgb
np,pd=C.np,C.I.pd


def consistency_counts(qty,flow,value,severe,fill,price,bid):
    qty,flow,value,severe,fill,price=[np.asarray(x,float) for x in [qty,flow,value,severe,fill,price]]
    C.require(all(np.isfinite(x).all() for x in [qty,flow,value,severe,fill,price]),'nonfinite response audit input')
    C.require(all(x.shape==qty.shape for x in [flow,value,severe,fill,price]),'response audit shape')
    low=-np.where(bid,price,100.-price)*qty;high=np.where(bid,100.-price,price)*qty
    value_excess=np.maximum.reduce([low-value,value-high,np.zeros(len(qty))])
    probability_excess=np.maximum(severe-fill,0.);quantity_excess=np.maximum(qty-flow,0.)
    return dict(n=len(qty),value_outside_quantity_implied_bounds=int(((value<low-1e-6)|(value>high+1e-6)).sum()),
        severe_probability_above_own_fill_probability=int((severe>fill+1e-6).sum()),
        expected_own_quantity_above_public_quantity=int((qty>flow+1e-6).sum()),
        value_bound_excess_cents=dict(mean=float(value_excess.mean()),max=float(value_excess.max())),
        severe_probability_excess=dict(mean=float(probability_excess.mean()),max=float(probability_excess.max())),
        quantity_bound_excess_shares=dict(mean=float(quantity_excess.mean()),max=float(quantity_excess.max())),
        interpretation='separately fitted conditional response heads can be incoherent; diagnostic only, no post-hoc clipping')


def consistency_controls():
    good=consistency_counts([1.],[2.],[0.],[.1],[.2],[50.],[True])
    bad=consistency_counts([3.],[2.],[1000.],[.8],[.2],[50.],[True])
    names=['value_outside_quantity_implied_bounds','severe_probability_above_own_fill_probability','expected_own_quantity_above_public_quantity']
    C.require(all(good[k]==0 and bad[k]==1 for k in names),'response inconsistency control did not fire')
    caught=False
    try:consistency_counts([1.],[2.],[np.nan],[.1],[.2],[50.],[True])
    except ValueError:caught=True
    C.require(caught,'nonfinite response refusal failed')
    return dict(valid_fixture=True,three_planted_inconsistencies=True,nonfinite_refused=True)


def plot(result,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models=result['pooled']['models'];colors={'joint':'#238b77','independent':'#cb7b28','direct_value':'#5e6fa1','mechanical':'#888888','empirical':'#bbbbbb'}
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    names=['empirical','mechanical','independent','joint']
    ax[0,0].bar(names,[models[n]['joint_logloss'] for n in names],color=[colors[n] for n in names])
    ax[0,0].set(title='Joint volume / price prediction',ylabel='Log loss (lower is better)')
    categories=['Flow reaches price','Adverse price move','Own fill','Severe own loss']
    for i,name in enumerate(['joint','independent']):
        m=models[name];v=[m['flow_any_auc'],m['adverse_move_auc'],m['own_fill']['auc'],m['severe']['auc']]
        ax[0,1].bar(np.arange(4)+(i-.5)*.34,v,width=.34,label=name,color=colors[name])
    ax[0,1].axhline(.5,color='#777777',ls=':',lw=1)
    ax[0,1].set(xticks=np.arange(4),xticklabels=categories,ylim=(0,1),ylabel='AUC',title='Ranking quality');ax[0,1].tick_params(axis='x',labelsize=8);ax[0,1].legend()
    cal=pd.DataFrame(result['calibration']);points=[]
    for name,col in [('joint',colors['joint']),('direct',colors['direct_value'])]:
        g=cal[(cal.model==name)&(cal.target=='value_cents')]
        ax[1,0].plot(g.predicted,g.actual,'o-',label=name,color=col);points.extend(g.predicted);points.extend(g.actual)
    lo,hi=min(points),max(points)
    ax[1,0].plot([lo,hi],[lo,hi],':',color='#777777',label='Calibrated')
    ax[1,0].axhline(0,color='#bbbbbb',lw=.7);ax[1,0].axvline(0,color='#bbbbbb',lw=.7)
    ax[1,0].set(xlabel='Predicted mean (cents / order)',ylabel='Realized mean (cents / order)',title='Local-value calibration by score decile');ax[1,0].legend()
    for name in ['joint','independent','direct_value']:
        ys=[]
        for phase in range(1,6):
            m=result['by_phase'][phase]['models'][name]
            ys.append(np.sqrt(m['mse'] if name=='direct_value' else m['value_cents']['mse']))
        ax[1,1].plot(range(1,6),ys,'o-',label=name,color=colors[name])
    ax[1,1].set(xlabel='Contract minute',ylabel='RMSE (cents / order)',title='Local-value error by contract phase',xticks=range(1,6));ax[1,1].legend()
    for a in ax.flat:a.spines[['top','right']].set_visible(False)
    fig.suptitle(f"First chronological prediction check: {result['test_day']} · {result['pooled']['n']:,} neutral actions\nConsumed development data; simulated local values, not strategy P&L",fontsize=13)
    fig.savefig(dest/'predictions.png',dpi=160);plt.close(fig)


def summary():
    C.verify();controls=CHECK.selftest();response_controls=consistency_controls();day=C.TESTS[0]
    fold=C.OUT/('fit_'+day);fit=json.loads((fold/'receipt.json').read_text())
    C.require(tuple(fit['training_days'])==tuple(C.FOLDS[day][0]) and fit['calibration_day']==C.FOLDS[day][1], 'first fold identity')
    dataset_reports={}
    for d in (*fit['training_days'],fit['calibration_day'],day):
        p=C.OUT/d;receipt=json.loads((p/'receipt.json').read_text())
        C.require(C.sha(p/'receipt.json')==fit['data_receipts'][d], 'data receipt identity')
        C.require(C.sha(p/'actions.parquet')==receipt['output_sha256'], 'data identity')
        C.require(receipt['source_manifest']==C.sha(C.REVIEW/'freeze_v1.json'), 'data source contract')
        for path,h in receipt['input_hashes'].items():C.require(C.sha(path)==h,'changed input '+path)
        dataset_reports[d]={k:receipt[k] for k in ['asof','selected_markets','rows','eligible','statuses','by_phase']}
    for name,h in fit['outputs'].items():C.require(C.sha(fold/name)==h,'fit output '+name)
    for spec in fit['artifacts'].values():C.require(C.sha(spec['path'])==spec['sha256'],'model hash')
    p=pd.read_parquet(fold/'test.parquet');raw=M.read(day)
    keys=['day','m','side','rung','generation','decision_ms']
    C.require(p[keys].equals(raw[keys].reset_index(drop=True)), 'prediction row identities')
    C.require(not p.duplicated(keys[:-1]).any(), 'duplicate action')
    for target in C.TARGETS+['joint_class','flow_class','move_class','move_cents']:
        C.require(np.array_equal(p[target].to_numpy(),raw[target].to_numpy()),'prediction target '+target)
    # Independently reload probabilities and the scenario mixture on spread-out rows.
    ix=np.linspace(0,len(raw)-1,min(512,len(raw)),dtype=int)
    x=raw.iloc[ix][C.FEATURES].to_numpy(np.float32)
    prob={}
    for name,features in [('joint',C.FEATURES),('mechanical',C.MECHANICAL),('flow',C.FEATURES),('move',C.FEATURES)]:
        b=lgb.Booster(model_file=str(fold/(name+'.txt')))
        xx=raw.iloc[ix][features].to_numpy(np.float32)
        prob[name]=M.temperature(b.predict(xx,num_threads=1),fit['temperature'][name]['value'])
    prob['independent']=(prob['flow'][:,:,None]*prob['move'][:,None,:]).reshape(-1,15)
    for name in ['joint','mechanical','independent']:
        saved=p.iloc[ix][[f'{name}_p{k}' for k in range(15)]].to_numpy()
        C.require(np.allclose(prob[name],saved,rtol=0,atol=1e-10),'probability reload '+name)
    b=lgb.Booster(model_file=str(fold/'conditional_value_cents.txt'));response=np.empty((len(ix),15))
    for k in range(15):
        xx=np.column_stack([x,np.full(len(ix),k//5),np.full(len(ix),k%5)]).astype(np.float32)
        response[:,k]=b.predict(xx,num_threads=1) if k//5 else 0.
    for name in ['joint','independent']:
        C.require(np.allclose((prob[name]*response).sum(1),p.iloc[ix][f'{name}_value_cents'],rtol=0,atol=1e-8),'mixture reload '+name)
    b=lgb.Booster(model_file=str(fold/'direct_value.txt'))
    C.require(np.allclose(b.predict(x,num_threads=1),p.iloc[ix].direct_value,rtol=0,atol=1e-10),'direct reload')
    corrupted=prob['joint'].copy();corrupted[0,0]+=.01
    C.require(not np.allclose(corrupted,p.iloc[ix][[f'joint_p{k}' for k in range(15)]].to_numpy(),rtol=0,atol=1e-10),'corruption falsifier did not fire')
    metrics=R.metrics(p);calibration=[];selection={};consistency={}
    for name in ['joint','independent','empirical','direct']:
        value_col='direct_value' if name=='direct' else f'{name}_value_cents'
        targets=[('value_cents',value_col)] if name=='direct' else [(t,f'{name}_{t}') for t in C.TARGETS]
        for target,column in targets:
            bins=pd.qcut(p[column].rank(method='first'),10,labels=False)
            for bucket,g in p.groupby(bins):
                calibration.append(dict(model=name,target=target,decile=int(bucket)+1,n=len(g),predicted=float(g[column].mean()),actual=float(g[target].mean())))
        mask=p[value_col]>0;g=p.loc[mask]
        selection[name]=dict(n=int(mask.sum()),fraction=float(mask.mean()),predicted_mean=float(g[value_col].mean()) if len(g) else None,
                            actual_mean=float(g.value_cents.mean()) if len(g) else None,
                            filled_actions=int(g.own_fill.sum()),severe_actions=int(g.severe.sum()),
                            caveat='score calibration diagnostic; not a replay or a validated trading rule')
        if name!='direct':
            consistency[name]=consistency_counts(p[f'{name}_own_quantity'],p[f'{name}_flow_quantity'],p[value_col],
                p[f'{name}_severe'],p[f'{name}_own_fill'],raw.x_price,raw.side.to_numpy()==0)
    train_counts=np.zeros(15,int);train_value_sum=0.;train_n=0
    for d in fit['training_days']:
        f=pd.read_parquet(C.OUT/d/'actions.parquet',columns=['status','joint_class','value_cents'])
        f=f[f.status.isin(['eligible','eligible_estimated_timing'])]
        train_counts+=np.bincount(f.joint_class.astype(int),minlength=15)
        train_value_sum+=float(f.value_cents.sum());train_n+=len(f)
    prior_mean=train_value_sum/train_n;y=p.value_cents.to_numpy()
    value_baselines=dict(zero=dict(predicted_mean=0.,mse=float(np.mean(y*y))),
        training_mean=dict(training_n=train_n,predicted_mean=prior_mean,mse=float(np.mean((y-prior_mean)**2))))
    result=dict(asof=C.stamp(),test_day=day,training_days=fit['training_days'],calibration_day=fit['calibration_day'],
        evidence='one consumed chronological development day; point estimates only, no interval or strategy-profit claim',
        pooled=metrics,by_phase={int(k):R.metrics(g) for k,g in p.groupby('phase')},by_side={int(k):R.metrics(g) for k,g in p.groupby('side')},
        calibration=calibration,positive_score_diagnostic=selection,response_consistency=consistency,value_baselines=value_baselines,
        class_support=dict(training=train_counts,test=np.bincount(p.joint_class.astype(int),minlength=15)),
        within_null=R.within_null(p),coverage=dataset_reports,
        audit=dict(controls=controls,response_controls=response_controls,model_reload_rows=len(ix),probability_and_value_reload=True,corrupted_probability_refused=True,
                   input_data_model_prediction_hashes=True,prediction_identity=True,fit_receipt_sha256=C.sha(fold/'receipt.json')))
    q=metrics['models']
    result['comparisons']=dict(joint_logloss_beats_independent=q['joint']['joint_logloss']<q['independent']['joint_logloss'],
        joint_logloss_beats_mechanical=q['joint']['joint_logloss']<q['mechanical']['joint_logloss'],
        joint_value_mse_beats_independent=q['joint']['value_cents']['mse']<q['independent']['value_cents']['mse'],
        joint_value_mse_beats_direct=q['joint']['value_cents']['mse']<q['direct_value']['mse'],
        joint_value_mse_beats_zero=q['joint']['value_cents']['mse']<value_baselines['zero']['mse'],
        joint_value_mse_beats_training_mean=q['joint']['value_cents']['mse']<value_baselines['training_mean']['mse'])
    dest=C.REVIEW/'first_day_v1';C.require(not dest.exists(),'first-day output exists');dest.mkdir(parents=True)
    C.dump(dest/'report.json',result);pd.DataFrame(calibration).to_csv(dest/'calibration.csv',index=False);plot(result,dest)
    C.dump(dest/'receipt.json',dict(asof=C.stamp(),script_sha256=C.sha(__file__),source_manifest=C.sha(C.REVIEW/'freeze_v1.json'),
        report_sha256=C.sha(dest/'report.json'),calibration_sha256=C.sha(dest/'calibration.csv'),plot_sha256=C.sha(dest/'predictions.png'),n=len(p),test_day=day))
    print(json.dumps(C.clean(dict(asof=result['asof'],pooled=metrics,comparisons=result['comparisons'],positive_score=selection))),flush=True)


if __name__=='__main__':summary()
