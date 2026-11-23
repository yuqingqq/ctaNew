"""Independent saved-data review. No raw tape reconstruction, fitting or strategy run."""
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT=Path('/home/yuqing/ctaNew'); PIN=Path('/tmp/pf-competing-review-4eab1efe')
BASE=Path('predictfun/reviews/maker_trace_20261122'); HERE=Path(__file__).resolve().parent
TARGET='4eab1efe2aea01d8b7e8c7a530604537d1a87ee8'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def f(sub,name):
    p=PIN/BASE/'v2/chains'/sub/name
    return p if p.exists() else ROOT/BASE/'v2/chains'/sub/name
def csv(sub,name): return pd.read_csv(f(sub,name),float_precision='round_trip')
def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [clean(v) for v in x]
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x): return str(x)
    return x
E={'target':TARGET,'hashes':[]}; receipts={}
for sub in ['competing_transition','continuation_econ']:
    rec=json.loads(f(sub,'receipt.json').read_text()); receipts[sub]=rec
    for name,m in rec['outputs'].items():
        E['hashes'].append({'kind':'output','name':sub+'/'+name,'ok':sha(f(sub,name))==m['sha256']})
    for name,m in rec.get('sources',{}).items():
        E['hashes'].append({'kind':'source','name':name,'ok':sha(PIN/name)==(m['sha256'] if isinstance(m,dict) else m)})
assert all(x['ok'] for x in E['hashes'])
ep=pd.read_parquet(f('touch_transition','episodes.parquet')); rd=csv('competing_transition','rd.csv'); sp=csv('competing_transition','support.csv')
saved=csv('competing_transition','outcomes.csv')
recon=csv('touch_transition','reconcile.csv'); submap={}
for r in recon[recon.cls=='two_side'].itertuples():
    key=(str(r.day),r.mid,r.frame); submap.setdefault(key,set()).add(r.sublabel)
assert all(len(v)==1 for v in submap.values())
labels=[]
for r in ep.itertuples():
    if r.end_type=='response': lab='same_'+r.resp_type
    elif r.end_reason=='other_side': lab='other_improve' if r.end_class.startswith('improve_') else 'other_retreat'
    elif r.end_reason=='two_side': lab='two_side_'+next(iter(submap.get((str(r.day),r.mid,r.end_frame),{'sub_unknown'})))
    elif r.end_reason=='same_side_retreat':lab='same_retreat_further'
    elif r.end_type=='no_response_1s':lab='horizon_no_change'
    else: lab='censored_'+r.end_reason
    labels.append(lab)
E['saved_columns']=list(saved.columns)
E['outcomes']={'count':len(ep),'unique':not ep.duplicated(['day','mid','episode_id']).any(),'labels':pd.Series(labels).value_counts().to_dict()}
key=['day','mid','episode_id']; ep['day']=ep.day.astype(str); saved['day']=saved.day.astype(str)
labcol='outcome' if 'outcome' in saved else 'next_outcome'
lookup=dict(zip(map(tuple,ep[key].to_numpy()),labels))
assert all(lookup[tuple(k)]==lab for k,lab in zip(saved[key].to_numpy(),saved[labcol]))
E['outcomes']['saved_equal']=True
resp=(ep.end_type=='response').to_numpy();cen=(ep.end_type=='censored_gap').to_numpy();comp=(ep.end_type=='censored_other').to_numpy()
allre=[]
for v in ['D0','LAG']:
    for cn in ['btc_with','btc_against','fair_gap','depth_own','time_left']:
        bc=ep['btc_5s_aligned_'+v].astype(str).to_numpy(); fg=ep['fair_gap_c_'+v].to_numpy();d=ep['depth_own_'+v].to_numpy();tl=ep['time_left_s_'+v].to_numpy()
        if cn=='btc_with':g=np.select([bc=='with',bc=='flat'],[1,0],-1)
        elif cn=='btc_against':g=np.select([bc=='against',bc=='flat'],[1,0],-1)
        elif cn=='fair_gap':g=np.select([fg>1,fg<=0],[1,0],-1)
        elif cn=='depth_own':g=np.select([d<=50,d>200],[1,0],-1)
        else:g=np.select([tl<=30,(tl>60)&(tl<=240)],[1,0],-1)
        g=np.where(ep['state_status_'+v].to_numpy()=='ok',g,-1)
        cols=['day','mid','side','delta_bin']+([] if cn=='time_left' else ['time_left_bin_'+v])
        t=ep[cols].copy();t['g']=g;t['r']=resp.astype(int);t['c']=cen.astype(int);t['k']=comp.astype(int);t['n']=1
        a=t[t.g==1].groupby(cols)[['n','r','c','k']].sum();b=t[t.g==0].groupby(cols)[['n','r','c','k']].sum()
        z=a.join(b,how='inner',lsuffix='1',rsuffix='0').reset_index()
        z['w']=z.n1*z.n0/(z.n1+z.n0);z['d']=z.r1/z.n1-z.r0/z.n0
        z['lo']=z.r1/z.n1-(z.r0+z.c0)/z.n0;z['hi']=(z.r1+z.c1)/z.n1-z.r0/z.n0
        def val(q,col='d'): return float((q.w*q[col]).sum()/q.w.sum()) if len(q) else np.nan
        stored=rd[(rd.variant==v)&(rd.contrast==cn)]
        for r in stored.itertuples():
            if r.scope=='pooled':q=z
            elif r.scope=='day':q=z[z.day==str(r.key)]
            elif r.scope=='side':q=z[z.side==r.key]
            elif r.scope=='market':
                day,mid=str(r.key).split(':');q=z[(z.day==day)&(z.mid==int(mid))]
            else:raise AssertionError(r.scope)
            for fld,col in [('rd','d'),('rd_censor_lo','lo'),('rd_censor_hi','hi')]:
                assert np.isclose(val(q,col),getattr(r,fld),equal_nan=True,atol=1e-12), (cn,v,r.scope,r.key,fld)
        mk=z.groupby(['day','mid'])[['n1','n0']].sum();eligible=int(((mk.n1>=5)&(mk.n0>=5)).sum())
        floor=min(z.n1.sum(),z.n0.sum())>=30 and min(z.r1.sum(),z.r0.sum())>=10 and min(z.k1.sum(),z.k0.sum())>=10 and eligible>=5
        s=sp[(sp.variant==v)&(sp.contrast==cn)].iloc[0];assert bool(floor)==bool(s.adequate_clock)
        for fld,col in [('n_exposed_mixed','n1'),('n_reference_mixed','n0'),('responses_exposed_mixed','r1'),('responses_reference_mixed','r0'),('competing_exposed_mixed','k1'),('competing_reference_mixed','k0')]: assert int(z[col].sum())==int(s[fld])
        days={day:val(q) for day,q in z.groupby('day')};contrib=z.assign(wd=z.w*z.d).groupby(['day','mid'])[['w','wd']].sum()
        contribution=contrib.wd/z.w.sum();top=contribution.abs().idxmax()
        allre.append({'variant':v,'contrast':cn,'rd':val(z),'day_values':days,'adequate':bool(floor),'markets_with_5_each':eligible,
          'all_exposed':int((g==1).sum()),'all_reference':int((g==0).sum()),'mixed_exposed':int(z.n1.sum()),'mixed_reference':int(z.n0.sum()),
          'raw_rd':float(resp[g==1].mean()-resp[g==0].mean()),'largest_abs_market':top,'contribution':float(contribution.loc[top]),
          'net_share':float(contribution.loc[top]/val(z)),'abs_share':float(abs(contribution.loc[top])/contribution.abs().sum())})
E['matched_comparisons']=allre;E['rd_rows_verified']=len(rd)
readings={}
for cn in ['btc_with','btc_against','fair_gap','depth_own','time_left']:
    a,b=[x for x in allre if x['contrast']==cn];sg=np.sign(a['rd'])
    consistent=sg!=0 and np.sign(b['rd'])==sg and all(sum(np.sign(y)==sg for y in x['day_values'].values())>=3 for x in [a,b])
    r='thin' if not(a['adequate'] and b['adequate']) else ('adequate_direction_consistent' if consistent else 'adequate_inconsistent')
    assert set(sp.loc[sp.contrast==cn,'reading'])=={r};readings[cn]=r
E['readings']=readings
# Exact source artifacts used for the independent order/close accounting recount.
rec=receipts['continuation_econ'];paths={}
for name in ['fills_valued','closes','events','episodes']:
    m=rec['inputs'][name];p=ROOT/m['path'];assert sha(p)==m['sha256'];paths[name]=p
O=csv('continuation_econ','orders_econ.csv');O['day']=O.day.astype(str)
cols=['day','mid','sid','mk_idx','role','hash','addr','status','self_trade','settle_ms','q','s','p','mq','rq','m_end','V','fee_fill']
F=pq.read_table(paths['fills_valued'],columns=cols,filters=[('mid','in',O.mid.unique().tolist()),('hash','in',O.order_hash.tolist()),('role','==','maker')]).to_pandas()
F['day']=F.day.astype(str);F['hash']=F['hash'].str.lower();F=F[(F.status.astype(str)=='ok')&~F.self_trade]
keys=['day','mid','sid','mk_idx','role'];assert not F.duplicated(keys).any()
xc=['day','mid','addr','entry_sid','entry_mk_idx','entry_role','close_sid','close_mk_idx','close_role','sh','pnl','net','fee_fill_entry','fee_fill_close','close_settle_ms','hold_ms','dust']
X=pq.read_table(paths['closes'],columns=xc,filters=[('mid','in',O.mid.unique().tolist())]).to_pandas();X['day']=X.day.astype(str)
X=X[X.addr.str.lower().isin(O.addr.str.lower())]
ck=['day','mid','addr','entry_sid','entry_mk_idx','entry_role','close_sid','close_mk_idx','close_role'];assert not X.duplicated(ck).any()
checks=[]
for r in O.itertuples():
    G=F[(F.day==r.day)&(F.mid==r.mid)&(F['hash']==r.order_hash)]
    assert len(G)==r.n_clean
    resid=(G.s*G.rq*(G.m_end-G.p)).sum();assert np.isclose(resid,r.resid_val_end,atol=1e-9)
    if r.exact_link!='available':continue
    Q=X[(X.day==r.day)&(X.mid==r.mid)&(X.addr.str.lower()==r.addr.lower())]
    lk=set(zip(G.sid,G.mk_idx,G.role.astype(str)))
    ent=Q.loc[np.array([k in lk for k in zip(Q.entry_sid,Q.entry_mk_idx,Q.entry_role.astype(str))],dtype=bool)]
    old=Q.loc[np.array([k in lk for k in zip(Q.close_sid,Q.close_mk_idx,Q.close_role.astype(str))],dtype=bool)]
    for win,mask in [('le10s',(ent.close_settle_ms<=r.end_ms)&(ent.hold_ms<=10000)),('le60s',(ent.close_settle_ms<=r.end_ms)&(ent.hold_ms<=60000)),('leEND',ent.close_settle_ms<=r.end_ms),('afterEND',ent.close_settle_ms>r.end_ms)]:
        q=ent[mask]
        for field in ['sh','pnl','net']:assert np.isclose(q[field].sum(),getattr(r,'close_'+win+'_'+field),atol=1e-9)
        assert int((~q.dust).sum())==getattr(r,'close_'+win+'_n')
    assert np.isclose(old.net.sum(),r.earlier_net,atol=1e-9)
    assert np.isclose((ent.fee_fill_entry.sum()+old.fee_fill_close.sum()+(G.fee_fill*G.rq/G.q).sum()),G.fee_fill.sum(),atol=1e-9)
    checks.append(r.order_hash)
R=O[O.n_response_links>0];K=R[R.exact_link=='available']
types=ep[(ep.end_type=='response') & ep.cand_hash.notna() & (ep.cand_hash!='')][['day','mid','cand_hash','resp_type']].rename(columns={'cand_hash':'order_hash'})
J=K.merge(types,on=['day','mid','order_hash'],how='left',validate='one_to_one')
E['economics']={'orders':len(O),'response_orders':len(R),'exact_response_orders':len(K),'exact_all_recounted':len(checks),
 'unavailable_response_orders':int((R.exact_link!='available').sum()),'clean_legs':len(F),'after_END_closes':int(O.close_afterEND_n.sum()),
 'le10s':{'closes':int(K.close_le10s_n.sum()),'shares':float(K.close_le10s_sh.sum()),'net':float(K.close_le10s_net.sum())},
 'leEND':{'closes':int(K.close_leEND_n.sum()),'shares':float(K.close_leEND_sh.sum()),'net':float(K.close_leEND_net.sum())},
 'by_response_type':J.groupby('resp_type').agg(orders=('order_hash','count'),entry_net=('close_leEND_net','sum'),earlier_net=('earlier_net','sum'),residual=('resid_val_end','sum')).to_dict('index'),
 'available_residual_END':float(K.resid_val_end.sum()),'earlier_net':float(K.earlier_net.sum()),'available_fee':float(K.fee_fill.sum()),
 'top3_entry_net':float(K.loc[K.top3,'close_leEND_net'].sum()),'unavailable_residual_END':float(R.loc[R.exact_link!='available','resid_val_end'].sum())}
E['instrument_controls']={'changed_RD_refused':not np.isclose(allre[0]['rd'],allre[0]['rd']+1),'duplicate_close_detectable':bool(pd.concat([X.iloc[:1],X.iloc[:1]]).duplicated(ck).any())}
assert all(E['instrument_controls'].values())
(HERE/'evidence.json').write_text(json.dumps(clean(E),indent=2)+'\n')
print(json.dumps(clean({k:E[k] for k in ['outcomes','readings','economics','rd_rows_verified','instrument_controls']}),indent=2))
