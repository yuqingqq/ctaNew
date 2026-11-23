"""Bounded pinned suites and independent saved-table recount. No tape/context rebuild."""
import os, sys, json, hashlib, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path('/home/yuqing/ctaNew'); PIN=Path('/tmp/pf-posting-review-ed7d1eba')
BASE=Path('predictfun/reviews/maker_trace_20261122'); H=Path(__file__).resolve().parent
TOOLS=PIN/BASE/'tools'; TARGET='ed7d1ebaf34aa79cadec467ef41a894ea395423e'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def file(sub,name):
    a=PIN/BASE/'v2/chains'/sub/name
    return a if a.exists() else ROOT/BASE/'v2/chains'/sub/name
def csv(sub,name): return pd.read_csv(file(sub,name),float_precision='round_trip')
def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [clean(v) for v in x]
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x): return str(x)
    return x
E={'target':TARGET,'suites':[],'hashes':[]}
for name in ['posting_opportunity_selftest','queue_front_v2_selftest','chains_v3_selftest']:
    if '--reuse-suites' not in sys.argv:
        with (H/(name+'.log')).open('w') as f:
            r=subprocess.run([sys.executable,str(TOOLS/(name+'.py'))],env={**os.environ,'MT_RUN':'v2','PYTHONDONTWRITEBYTECODE':'1'},stdout=f,stderr=subprocess.STDOUT)
        rc=r.returncode
    else: rc=0 if '"failed": 0' in (H/(name+'.log')).read_text().splitlines()[-1] else 1
    E['suites'].append({'name':name,'rc':rc,'last':(H/(name+'.log')).read_text().splitlines()[-2:]})
for sub in ['posting_opportunity','queue_front_v2','chains_v3']:
    rec=json.loads(file(sub,'receipt.json').read_text())
    for name,meta in rec['outputs'].items():
        p=file(sub,name); want=meta['sha256'] if isinstance(meta,dict) else meta
        E['hashes'].append({'sub':sub,'kind':'output','name':name,'ok':sha(p)==want})
    for name,meta in rec['sources'].items():
        p=PIN/name; want=meta['sha256'] if isinstance(meta,dict) else meta
        E['hashes'].append({'sub':sub,'kind':'source','name':name,'ok':p.exists() and sha(p)==want})
p=pd.read_parquet(file('posting_opportunity','rows.parquet')); ev=csv('posting_opportunity','events.csv')
p['day']=p.day.astype(str); ev['day']=ev.day.astype(str)
r=csv('posting_opportunity','ratios.csv'); c=csv('posting_opportunity','consistency.csv')
E['columns']=list(p.columns); E['event_columns']=list(ev.columns)
E['population']={'rows':len(p),'keys_unique':not p.duplicated(['day','mid','side','k']).any(),'per_market_side_counts':p.groupby(['day','mid','side']).size().value_counts().to_dict(),'status':p.status_D0.value_counts().to_dict(),'event_statuses':ev.status.value_counts().to_dict()}
E['improve_exclusion']={'rows_eligible':int(p.improve_population.sum()),'event_seconds':int(p.E_improve.sum()),'event_seconds_excluded':int(p.loc[~p.improve_population,'E_improve'].sum()),'creations':int(p.N_improve.sum()),'creations_excluded':int(p.loc[~p.improve_population,'N_improve'].sum())}
E['uncertain_rows']={ty:{'event_seconds':int(p['E_'+ty].sum()),'straddle_without_event':int(((p.n_straddle>0)&(p['E_'+ty]==0)).sum())} for ty in ['any','improve','re_establish','deep']}
types=['any','improve','re_establish','deep','at_empty_touch']
idx=pd.MultiIndex.from_frame(p[['day','mid','side','k']]); good=ev[ev.status=='event']
E['outcome_recounts']=[]
for ty in types:
    g=good if ty=='any' else good[good['is_'+ty]]
    tab=g.groupby(['day','mid','side','row_k']).size()
    n=tab.reindex(idx,fill_value=0).to_numpy()
    E['outcome_recounts'].append({'type':ty,'ok':bool(np.array_equal(n,p['N_'+ty]) and np.array_equal(n>0,p['E_'+ty]))})
ms,keys=pd.factorize(pd.MultiIndex.from_frame(p[['day','mid','side']]),sort=False)
mi,mkeys=pd.factorize(pd.MultiIndex.from_frame(p[['day','mid']]),sort=False)
nsms=len(keys);nm=len(mkeys);n=len(p)
E['ratios']=[]; E['nulls']=[]; E['consistency_failures']=[]
chosen={('any','spread'),('any','depth_own'),('re_establish','fair_gap'),('re_establish','btc_against'),('deep','time_left')}
shifted={}
for ty in set(x[0] for x in chosen):
    mat=np.zeros((500,n),dtype=np.float64)
    for j,key in enumerate(keys):
        ix=np.flatnonzero(ms==j);day,mid,side=key
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261123,int(day),int(mid),0 if side=='bid' else 1])))
        off=rng.integers(0,len(ix),size=500)
        mat[:,ix]=p['E_'+ty].to_numpy()[ix][(np.arange(len(ix))[None,:]-off[:,None])%len(ix)]
    shifted[ty]=mat
for row in r.itertuples():
    v,ty,cn=row.variant,row.type,row.contrast
    status=p['status_'+v].to_numpy()=='ok';sp=p['spread_ticks_'+v].to_numpy();fg=p['fair_gap_c_'+v].to_numpy();bc=p['btc_5s_aligned_'+v].to_numpy();dp=p['depth_touch_own_'+v].to_numpy();fw=p['flow_5s_sh_'+v].to_numpy();tl=p['time_left_s_'+v].to_numpy()
    tb=np.select([tl>300,tl>240,tl>120,tl>60,tl>30,tl>10],[0,1,2,3,4,5],6)
    if ty=='improve': status &= p.improve_population.to_numpy()
    if cn=='spread':
        # Tick categories use half-tick boundaries; raw floating values need not equal an integer.
        sb=p['spread_bin_'+v].to_numpy()
        grp=np.select([np.isin(sb,['3-4','>=5']),sb=='2'],[1,0],-1) if ty=='improve' else np.select([np.isin(sb,['2','3-4','>=5']),sb=='1'],[1,0],-1)
    elif cn=='fair_gap':grp=np.select([fg>1,fg<=0],[1,0],-1)
    elif cn=='btc_with':grp=np.select([bc=='with',bc=='flat'],[1,0],-1)
    elif cn=='btc_against':grp=np.select([bc=='against',bc=='flat'],[1,0],-1)
    elif cn=='depth_own':grp=np.select([(dp>=0)&(dp<=50),dp>200],[1,0],-1)
    elif cn=='flow':grp=np.select([fw>0,fw==0],[1,0],-1)
    else:grp=np.select([tl<=30,(tl>60)&(tl<=240)],[1,0],-1)
    grp=np.where(status,grp,-1);st=ms if cn=='time_left' else ms*7+tb;ns=nsms if cn=='time_left' else nsms*7
    keep=grp>=0;counts=np.bincount(st[keep]*2+grp[keep],minlength=ns*2).reshape(ns,2)
    events=p['E_'+ty].to_numpy();obs=np.bincount(st[keep]*2+grp[keep],weights=events[keep],minlength=ns*2).reshape(ns,2)
    valid=(counts>0).all(1);total=np.maximum(counts.sum(1),1)
    wn=np.zeros(n);wd=np.zeros(n);a=keep&(grp==1);b=keep&(grp==0)
    wn[a]=np.where(valid[st[a]],counts[st[a],0]/total[st[a]],0);wd[b]=np.where(valid[st[b]],counts[st[b],1]/total[st[b]],0)
    num=events@wn;den=events@wd;value=num/den if den else (np.inf if num else np.nan)
    E['ratios'].append({'key':[v,ty,cn],'computed':value,'saved':row.RR_MH,'ok':bool(np.isclose(value,row.RR_MH,equal_nan=True,rtol=1e-11,atol=1e-12))})
    for j,key in enumerate(mkeys):
        m=mi==j;num=events[m]@wn[m];den=events[m]@wd[m];vr=num/den if den else (np.inf if num else np.nan)
        a=int(events[m&(grp==1)].sum());b=int(events[m&(grp==0)].sum());t1=int((m&(grp==1)).sum());t0=int((m&(grp==0)).sum());sup=(a+b>=1 and t1>=30 and t0>=30)
        z=c[(c.row=='market')&(c.variant==v)&(c.type==ty)&(c.contrast==cn)&(c.day.astype(str)==str(key[0]))&(c.mid==key[1])].iloc[0]
        if not(np.isclose(vr,z.rr_mh,equal_nan=True,rtol=1e-11,atol=1e-12) and sup==z.supported):E['consistency_failures'].append([v,ty,cn,list(key),vr,z.rr_mh,sup,z.supported])
    if v=='D0' and (ty,cn) in chosen:
        num=shifted[ty]@wn;den=shifted[ty]@wd
        with np.errstate(divide='ignore',invalid='ignore'): vals=num/den
        tol=1e-12*max(abs(value),1e-300);ph=(1+np.count_nonzero(vals>=value-tol)+np.count_nonzero(np.isnan(vals)))/501;pl=(1+np.count_nonzero(vals<=value+tol)+np.count_nonzero(np.isnan(vals)))/501
        E['nulls'].append({'key':[v,ty,cn],'p_hi':ph,'p_lo':pl,'saved_hi':row.p_hi,'saved_lo':row.p_lo,'ok':bool(np.allclose([ph,pl],[row.p_hi,row.p_lo],atol=1e-12,rtol=1e-12))})
E['identity']=good.identity.value_counts().to_dict()
cl=csv('chains_v3','chain_links.csv'); E['chain_link_columns']=list(cl.columns)
E['chain_links']=cl.to_dict('records')
(H/'evidence.json').write_text(json.dumps(clean(E),indent=2)+'\n')
print(json.dumps(clean({k:v for k,v in E.items() if k not in ['columns','event_columns','chain_links','hashes','ratios']}),indent=2))
print(json.dumps({'hash_checks':len(E['hashes']),'bad_hashes':[x for x in E['hashes'] if not x['ok']],'mh_checks':len(E['ratios']),'bad_ratios':[x for x in E['ratios'] if not x['ok']]},indent=2))
