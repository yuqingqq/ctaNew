"""Bounded review of saved, hash-bound outputs; no tape load, fit or replay."""
import os, sys, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path('/home/yuqing/ctaNew'); PIN=Path('/tmp/pf-touch-review-f81b5b0d')
BASE=Path('predictfun/reviews/maker_trace_20261122'); HERE=Path(__file__).resolve().parent
TARGET='f81b5b0d61ca14eb6039231e9daaeaa954b647ee'
os.environ['MT_RUN']='v2'
sys.path.insert(0,str(PIN/BASE/'tools'))
import touch_transition as TT

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def file(sub,name):
    p=PIN/BASE/'v2/chains'/sub/name
    return p if p.exists() else ROOT/BASE/'v2/chains'/sub/name
def csv(sub,name): return pd.read_csv(file(sub,name),float_precision='round_trip')
def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [clean(v) for v in x]
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x): return str(x)
    return x
E={'target':TARGET,'hashes':[]}
for sub in ['touch_transition','chains_v4']:
    rec=json.loads(file(sub,'receipt.json').read_text())
    for name,meta in rec['outputs'].items():
        want=meta['sha256'] if isinstance(meta,dict) else meta
        E['hashes'].append(dict(sub=sub,kind='output',name=name,ok=sha(file(sub,name))==want))
    for name,meta in rec.get('sources',{}).items():
        want=meta['sha256'] if isinstance(meta,dict) else meta
        E['hashes'].append(dict(sub=sub,kind='source',name=name,ok=sha(PIN/name)==want))
assert all(x['ok'] for x in E['hashes'])
ep=pd.read_parquet(file('touch_transition','episodes.parquet'))
rc=csv('touch_transition','reconcile.csv'); cn=csv('touch_transition','continuations.csv')
rr=csv('touch_transition','ratios.csv'); cons=csv('touch_transition','consistency.csv')
E['columns']={'episodes':list(ep.columns),'continuations':list(cn.columns)}
assert not ep.duplicated(['day','mid','episode_id']).any()
assert (ep.exposure_ms==ep.t_end_ms-ep.t_start_ms).all()
assert (ep.state_ms_D0<=ep.t_prev_ms).all()
E['population']={'episodes':len(ep),'markets':ep.mid.nunique(),'days':ep.day.nunique(),
 'end_reasons':ep.end_reason.value_counts().to_dict(), 'response_type':ep.loc[ep.end_type=='response','resp_type'].value_counts().to_dict()}
overlap=0
for _,g in ep.groupby(['day','mid']):
    g=g.sort_values('t_start_ms'); overlap+=int((g.t_end_ms.to_numpy()[:-1]>g.t_start_ms.to_numpy()[1:]).sum())
E['population']['episode_overlaps']=overlap
eg=ep.groupby(['day','mid']).exposure_ms.sum()/1000
E['exposure']={'total_seconds':float(eg.sum()),'market_min_seconds':float(eg.min()),'market_max_seconds':float(eg.max()),'by_market_seconds':eg.to_dict()}
res=ep[ep.end_type=='response'].copy()
E['timing']={'n_response':len(res),'next_frame':int(res.no_frame_between.sum()),'endpoint_median_ms':float(res.exposure_ms.median()),
 'physical_lower_median_ms':float(np.maximum(0,res.resp_lo_ms-res.t_start_ms).median()),
 'physical_upper_median_ms':float((res.resp_hi_ms-res.t_prev_ms).median())}
im=rc[rc.is_improve & (rc.po_status=='event')]; ex=im[im.anchor_excluded]
E['reconciliation']={'improvements':len(im),'improvement_classes':im.cls.value_counts().to_dict(),
 'excluded':len(ex),'excluded_classes':ex.cls.value_counts().to_dict(), 'keys_unique':not rc.key.duplicated().any()}
assert len(im)==11804 and len(ex)==8821
E['ratios']=[]
for r in rr.itertuples():
    v,co=r.variant,r.contrast; ok=ep['state_status_'+v].to_numpy()=='ok'
    bc=ep['btc_5s_aligned_'+v].astype(str).to_numpy();fg=ep['fair_gap_c_'+v].to_numpy();d=ep['depth_own_'+v].to_numpy();tl=ep['time_left_s_'+v].to_numpy()
    if co=='btc_with': grp=np.select([bc=='with',bc=='flat'],[1,0],-1)
    elif co=='btc_against': grp=np.select([bc=='against',bc=='flat'],[1,0],-1)
    elif co=='fair_gap': grp=np.select([fg>1,fg<=0],[1,0],-1)
    elif co=='depth_own': grp=np.select([d<=50,d>200],[1,0],-1)
    else: grp=np.select([tl<=30,(tl>60)&(tl<=240)],[1,0],-1)
    grp=np.where(ok,grp,-1)
    cols=['day','mid','side','delta_bin']+([] if co=='time_left' else ['time_left_bin_'+v])
    st,_=pd.factorize(pd.MultiIndex.from_frame(ep[cols])); n=int(st.max()+1)
    ev=(ep.end_type=='response').to_numpy().astype(float); t=ep.exposure_ms.to_numpy()/1000
    a=np.bincount(st[grp==1],weights=ev[grp==1],minlength=n);b=np.bincount(st[grp==0],weights=ev[grp==0],minlength=n)
    t1=np.bincount(st[grp==1],weights=t[grp==1],minlength=n);t0=np.bincount(st[grp==0],weights=t[grp==0],minlength=n)
    k=(t1>0)&(t0>0); num=(a[k]*t0[k]/(t1[k]+t0[k])).sum();den=(b[k]*t1[k]/(t1[k]+t0[k])).sum()
    rat=num/den if den else np.nan
    same=bool(np.isclose(rat,r.RR_MH,atol=1e-12,rtol=1e-12))
    E['ratios'].append({'variant':v,'contrast':co,'computed':rat,'stored':r.RR_MH,'equal':same,'verdict':r.verdict})
assert all(x['equal'] for x in E['ratios'])
assert set(rr.verdict)=={'insufficient_support'}
# A fully known synthetic future take outside END should not belong to a to-END continuation.
# Real artifacts are checked first; the direct predicate is recorded without changing production.
rec=json.loads(file('touch_transition','receipt.json').read_text())
tkpath=ROOT/BASE/'v2/chains/take_logic_v2/events.parquet'
assert sha(tkpath)==rec['inputs'][str(tkpath.relative_to(ROOT))]
tk=pd.read_parquet(tkpath)
E['take_columns']=list(tk.columns)
O=cn[cn.table=='order'].copy(); takes=[]
for r in O.itertuples():
    if r.takes_status!='evaluated': continue
    T=tk[(tk.mid==int(r.mid))&(tk.addr.str.lower()==str(r.cand_addr).lower())]
    # Market END is recovered exactly from the pre-state clock and saved time-left.
    g=ep[ep.mid==int(r.mid)]; ends=g.state_ms_D0.to_numpy()+np.rint(g.time_left_s_D0.to_numpy()*1000).astype(np.int64)
    assert len(set(ends))==1; end=int(ends[0])
    selected=T[T.x_lo_hard_ms>int(r.creation_hi_ms)]
    assert len(selected)==int(r.takes_end)
    definite_after=selected.x_lo_hard_ms>=end
    straddle=(selected.x_lo_hard_ms<end)&(selected.hi_ms>end)
    takes.append({'mid':int(r.mid),'hash':r.cand_hash,'selected':len(selected),'end_ms':end,
                  'definite_after_end':int(definite_after.sum()),'straddle_end':int(straddle.sum()),
                  'post_end_sids':selected.loc[definite_after,'sid'].tolist()})
E['continuation_takes']=takes
sample=O[O.takes_status=='evaluated'].iloc[0]
S=ep[(ep.mid==int(sample.mid))&(ep.cand_hash==sample.cand_hash)]
T=tk[(tk.mid==int(sample.mid))&(tk.addr.str.lower()==str(sample.cand_addr).lower())].copy()
T['day']=T.day.astype(str);T['addr_l']=T.addr.str.lower()
end=takes[0]['end_ms']
bad=T.iloc[[0]].copy();bad['sid']='review_synthetic_after_END';bad['x_lo_hard_ms']=end+1;bad['hi_ms']=end+2
empty=pd.DataFrame(columns=list(TT.FVV_READ)+['mk_hash','shares','hi_ms','addr_l'])
good_out,_=TT.continuations(S,empty,{'PF':None,'TK':T,'FV':empty})
bad_out,_=TT.continuations(S,empty,{'PF':None,'TK':pd.concat([T,bad],ignore_index=True),'FV':empty})
E['synthetic_end_predicate']={'good_count':int(good_out.takes_end.iloc[0]),'with_post_END_count':int(bad_out.takes_end.iloc[0]),
 'production_function_accepts_post_END':int(bad_out.takes_end.iloc[0])==int(good_out.takes_end.iloc[0])+1}
assert E['synthetic_end_predicate']['production_function_accepts_post_END']
E['chain3_response_price']=ep[(ep.mid==3702960)&(ep.episode_id==3)][['resp_tick','resp_best','resp_lo_ms','resp_hi_ms']].to_dict('records')
# Selected hash rows only, from the saved economic artifact; no book/tape work.
fv=ROOT/'data/predict_fun/research/maker_trace_v2_20261123/econ/fills_valued.parquet'
E['economic_input_hash_ok']=sha(fv)==rec['inputs'][str(fv.relative_to(ROOT))]
assert E['economic_input_hash_ok']
F=pq.read_table(fv,columns=['mid','hash','role','status','self_trade','st_10s','st_60s','st_end','q'],
 filters=[('mid','in',list(map(int,O.mid.unique()))),('hash','in',O.cand_hash.tolist()),('role','==','maker')]).to_pandas()
keys=set(zip(O.mid.astype(int),O.cand_hash)); F=F[[ (int(m),h) in keys for m,h in zip(F.mid,F['hash'])]]
E['continuation_quality']={'rows':len(F),'status':F.status.astype(str).value_counts().to_dict(),
 'self_trade_rows':int(F.self_trade.sum()),'invalid_rows':int((F.status.astype(str)!='ok').sum())}
# Positive and known-bad controls for the independent recount instrument.
assert np.isclose(E['ratios'][0]['computed'],E['ratios'][0]['stored'])
assert not np.isclose(E['ratios'][0]['computed'],E['ratios'][0]['stored']+1)
E['controls']={'intact_ratio_pass':True,'changed_ratio_refused':True,'synthetic_END_violation_exposed':True}
(HERE/'evidence.json').write_text(json.dumps(clean(E),indent=2)+'\n')
print(json.dumps(clean({k:E[k] for k in ['population','exposure','timing','reconciliation','continuation_takes','continuation_quality','controls']}),indent=2))
