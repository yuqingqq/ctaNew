"""Prediction quality only. No replay or inferred trading profitability."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pf_flow_price_common as C
from pf_flow_price_model import logloss
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
np,pd=C.np,C.I.pd


def correlation(a,b):
    return float(spearmanr(a,b).statistic) if len(a)>2 and np.std(a)>0 and np.std(b)>0 else None


def auc(y,p):
    return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None


def metrics(df):
    out=dict(n=len(df),markets=int(df.m.nunique()),own_fill_rate=float(df.own_fill.mean()),
             severe_rate=float(df.severe.mean()),actual_value_sum_cents=float(df.value_cents.sum()),models={})
    y=df.joint_class.to_numpy(int);yf=df.flow_class.to_numpy(int);ym=df.move_class.to_numpy(int)
    for name in ['joint','mechanical','independent','empirical']:
        p=df[[f'{name}_p{k}' for k in range(15)]].to_numpy()
        pf=p.reshape(-1,3,5).sum(2);pm=p.reshape(-1,3,5).sum(1)
        v=dict(joint_logloss=logloss(y,p),flow_logloss=logloss(yf,pf),move_logloss=logloss(ym,pm),
               flow_any_auc=auc(yf>0,1-pf[:,0]),adverse_move_auc=auc(ym<2,pm[:,:2].sum(1)),
               high_flow_adverse_brier=float(np.mean((p[:,10:12].sum(1)-((yf==2)&(ym<2)))**2)))
        if name!='mechanical':
            for target in C.TARGETS:
                pred=df[f'{name}_{target}'].to_numpy(float);actual=df[target].to_numpy(float)
                v[target]=dict(actual_mean=float(actual.mean()),predicted_mean=float(pred.mean()),
                    mse=float(np.mean((pred-actual)**2)),mae=float(np.mean(np.abs(pred-actual))),
                    rank_correlation=correlation(actual,pred))
                if target in ['severe','own_fill']:v[target]['auc']=auc(actual,pred)
            for mask,label in [(df.own_fill.to_numpy()>0,'filled_only_diagnostic'),
                               (df.severe.to_numpy()>0,'severe_only_diagnostic')]:
                if mask.any():
                    v[label]=dict(n=int(mask.sum()),actual_value_mean=float(df.value_cents.to_numpy()[mask].mean()),
                        predicted_value_mean=float(df[f'{name}_value_cents'].to_numpy()[mask].mean()))
        out['models'][name]=v
    pred=df.direct_value.to_numpy();actual=df.value_cents.to_numpy()
    out['models']['direct_value']=dict(mse=float(np.mean((pred-actual)**2)),mae=float(np.mean(np.abs(pred-actual))),
        actual_mean=float(actual.mean()),predicted_mean=float(pred.mean()),rank_correlation=correlation(actual,pred))
    return out


def within_null(df,draws=200):
    rng=np.random.default_rng(20261116)
    # Rank scores within each market/side/rung; this removes between-group variation.
    groups=list(df.groupby(['day','m','side','rung']).indices.values())
    y=df.value_cents.to_numpy();p=df.joint_value_cents.to_numpy();ranks=pd.Series(p).rank().to_numpy()
    centered=ranks.copy()
    for ix in groups:centered[ix]-=centered[ix].mean()
    observed=float(np.dot(centered,y));null=[]
    for _ in range(draws):
        yp=y.copy()
        for ix in groups:yp[ix]=rng.permutation(yp[ix])
        null.append(float(np.dot(centered,yp)))
    return dict(draws=draws,statistic=observed,p_one_sided=(1+sum(x>=observed for x in null))/(draws+1),
                null_mean=float(np.mean(null)),interpretation='within-group association on consumed data; not strategy gain or forward validation')


def report():
    C.verify();frames=[];fits=[];coverage={}
    for day in C.DAYS:
        r=json.loads((C.OUT/day/'receipt.json').read_text());coverage[day]={k:r[k] for k in ['asof','rows','eligible','statuses','by_phase','selected_markets','registry_exclusions']}
    for day in C.TESTS:
        p=C.OUT/('fit_'+day);r=json.loads((p/'receipt.json').read_text())
        C.require(C.sha(p/'test.parquet')==r['outputs']['test.parquet'],'prediction hash')
        frames.append(pd.read_parquet(p/'test.parquet'));fits.append(r)
    df=pd.concat(frames,ignore_index=True)
    C.require(not df.duplicated(['day','m','side','rung','generation']).any(),'duplicate out-of-fold prediction')
    result=dict(asof=C.stamp(),scope='chronological development predictions on consumed dates; no trading policy or profitability claim',
        pooled=metrics(df),by_day={d:metrics(g) for d,g in df.groupby('day')},
        by_side={int(s):metrics(g) for s,g in df.groupby('side')},by_phase={int(s):metrics(g) for s,g in df.groupby('phase')},
        coverage=coverage,within_null=within_null(df),fits=[{k:r[k] for k in ['test_day','training_days','calibration_day','n_train','n_calibration','n_test','temperature']} for r in fits])
    q=result['pooled']['models'];days=result['by_day']
    result['comparisons']=dict(joint_logloss_below_independent=q['joint']['joint_logloss']<q['independent']['joint_logloss'],
        joint_logloss_below_mechanical=q['joint']['joint_logloss']<q['mechanical']['joint_logloss'],
        joint_value_mse_below_independent=q['joint']['value_cents']['mse']<q['independent']['value_cents']['mse'],
        joint_value_mse_below_direct=q['joint']['value_cents']['mse']<q['direct_value']['mse'],
        days_joint_logloss_below_independent=sum(r['models']['joint']['joint_logloss']<r['models']['independent']['joint_logloss'] for r in days.values()),
        days_joint_value_mse_below_direct=sum(r['models']['joint']['value_cents']['mse']<r['models']['direct_value']['mse'] for r in days.values()))
    C.dump(C.REVIEW/'results_v1/report.json',result)
    rows=[]
    for (day,phase,side),g in df.groupby(['day','phase','side']):
        for name,x in metrics(g)['models'].items():
            row=dict(day=day,phase=phase,side=side,model=name,n=len(g))
            row.update({k:v for k,v in x.items() if not isinstance(v,dict)})
            if 'value_cents' in x:row.update({'value_'+k:v for k,v in x['value_cents'].items()})
            rows.append(row)
    pd.DataFrame(rows).to_csv(C.REVIEW/'results_v1/by_day_phase_side.csv',index=False)
    calibration=[]
    for name in ['joint','independent','empirical']:
        for target in ['value_cents','severe','own_quantity','flow_quantity']:
            p=df[f'{name}_{target}'];bins=pd.qcut(p.rank(method='first'),10,labels=False)
            for k,g in df.groupby(bins):
                calibration.append(dict(model=name,target=target,decile=int(k)+1,n=len(g),predicted=float(g[f'{name}_{target}'].mean()),actual=float(g[target].mean())))
    pd.DataFrame(calibration).to_csv(C.REVIEW/'results_v1/calibration.csv',index=False)
    print(json.dumps(C.clean(dict(asof=result['asof'],n=len(df),pooled=result['pooled'],comparisons=result['comparisons']))),flush=True)


if __name__=='__main__':report()
