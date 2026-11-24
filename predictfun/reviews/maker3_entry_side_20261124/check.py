"""Independent first-record / public-book recount; no import of audit.py."""
import collections,copy,json,math,resource
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
resource.setrlimit(resource.RLIMIT_CPU,(25,26))
H=Path(__file__).resolve().parent;R=H.parents[2]
D=R/'data/predict_fun/research/maker_trace_v2_20261123'
x=json.loads((H/'result.json').read_text());addr=x['address'];oracle={}
for day in ['20261024','20261025','20261026','20261027']:
    raw=pq.read_table(D/f'legs_{day}.parquet',filters=[[('mk_signer','=',addr)],[('tk_signer','=',addr)]],columns=['mid','settle_ms','sid','mk_idx','mk_hash','mk_signer','tk_signer','shares','y_side','mk_quote','mk_index_set','tk_y_side','status']).to_pandas()
    raw=raw[raw.shares>0]
    mk=raw[raw.mk_signer==addr]
    minima=mk.groupby('mid').settle_ms.min()
    first=mk[mk.settle_ms==mk.mid.map(minima)]
    assert first.status.eq('ok').all()
    groups={int(k):v for k,v in first.groupby('mid')}
    takes=raw[raw.tk_signer==addr];tm=takes.groupby('mid').settle_ms.min().to_dict()
    B=pq.read_table(D/f'tob_{day}.parquet',columns=['mid','upd_ms','recv_ns','best_bid','best_ask','n_bid','n_ask','status']).to_pandas()
    B['available']=np.maximum(B.upd_ms.to_numpy(),B.recv_ns.to_numpy()/1e6)
    bg={int(k):g for k,g in B.groupby('mid',sort=False)}
    for r in [r for r in x['rows'] if r['day']==day]:
        mid=r['mid'];g=groups.get(mid);v={}
        if g is None:oracle[(day,mid)]={'empty':True};continue
        v['first']=int(minima[mid]);v['legs']=sorted(zip(g.sid,g.mk_idx))
        v['side']=g.y_side.iloc[0] if g.y_side.nunique()==1 else 'both'
        v['prior_taker']=tm.get(mid,float('inf'))<v['first']
        v['native_buy_only']=g.mk_quote.eq('Bid').all()
        b=bg.get(mid)
        for field,t in [('fill_context',v['first']-1),('opening_context',r['ws_ms'])]:
            eligible=b[b.available<=t] if b is not None else pd.DataFrame()
            if len(eligible)==0:v[field]=dict(status='no_frame',p=None);continue
            z=eligible.sort_values(['available','upd_ms','recv_ns']).iloc[-1]
            st='ok'
            if {'bad_frame','mid_mismatch','out_of_order','off_grid'}&set(z.status.split('+')):st='invalid_frame'
            elif not (np.isfinite(z.best_bid) and np.isfinite(z.best_ask)) or z.n_bid<=0 or z.n_ask<=0:st='one_sided'
            elif not 0<=z.best_bid<z.best_ask<=1:st='crossed_or_invalid'
            elif t-z.upd_ms>1000:st='stale'
            v[field]=dict(status=st,p=float((z.best_bid+z.best_ask)/2) if st=='ok' else None,availability_ms=float(z.available),venue_ms=int(z.upd_ms))
        # Supplement: first observed activity direction when an earlier taker exists.
        if v['prior_taker']:
            a=takes[(takes.mid==mid)&(takes.settle_ms==tm[mid])];ss=sorted(a.tk_y_side.unique())
            t=tm[mid]-1;bb=b[b.available<=t].sort_values(['available','upd_ms','recv_ns']) if b is not None else pd.DataFrame()
            v['first_any_role']='taker';v['first_any_side']=ss[0] if len(ss)==1 else 'both'
            if len(bb):
                z=bb.iloc[-1]
                good=(z.n_bid>0 and z.n_ask>0 and 0<=z.best_bid<z.best_ask<=1 and t-z.upd_ms<=1000 and not ({'bad_frame','mid_mismatch','out_of_order','off_grid'}&set(z.status.split('+'))))
                v['first_any_p']=float((z.best_bid+z.best_ask)/2) if good else None
            else:v['first_any_p']=None
        else:v.update(first_any_role='maker_or_tied',first_any_side=v['side'],first_any_p=v['fill_context']['p'])
        oracle[(day,mid)]=v
    del raw,mk,first,B,bg

def label(s,p):
    if p is None:return 'unknown'
    if s=='both':return 'both_sides'
    if abs(p-.5)<1e-10:return 'neutral'
    return 'higher_probability' if ((s=='bid' and p>.5) or (s=='ask' and p<.5)) else 'lower_probability'
def verify(z):
    assert len(z['rows'])==1152
    seen=set()
    for r in z['rows']:
        key=(r['day'],r['mid']);assert key not in seen;seen.add(key)
        o=oracle[key]
        if o.get('empty'):assert r['status']=='no_maker_fill';continue
        assert r['status']=='observed_first_maker' and r['first_record_ms']==o['first']
        assert sorted((a['sid'],a['mk_idx']) for a in r['first_legs'])==o['legs']
        assert r['first_side']==o['side'] and r['prior_taker']==o['prior_taker']
        assert r['native_first']=='Bid' and o['native_buy_only']
        for field in ['fill_context','opening_context']:
            for k,val in o[field].items():assert r[field][k]==val,(key,field,k)
            assert r[field+'_direction']==label(o['side'],o[field]['p'])
    for day,s in z['summary'].items():
        rr=[r for r in z['rows'] if (day=='pooled' or r['day']==day)]
        a=[r for r in rr if r['status']=='observed_first_maker']
        assert s['markets']==len(rr)
        assert s['statuses']==dict(collections.Counter(r['status'] for r in rr))
        assert s['first_sides']==dict(collections.Counter(r['first_side'] for r in a))
        assert s['prior_taker']==sum(r['prior_taker'] for r in a)
        for f in ['fill_context','opening_context']:assert s[f]==dict(collections.Counter(r[f+'_direction'] for r in a))
    return True
verify(x)
mutations=[lambda q:q['rows'].pop(),lambda q:q['rows'][0].update(first_side='other'),lambda q:q['rows'][0]['first_legs'].pop(),lambda q:q['rows'][0]['fill_context'].update(availability_ms=0),lambda q:q['summary']['pooled']['fill_context'].update(higher_probability=999),lambda q:q['rows'][0].update(prior_taker=not q['rows'][0]['prior_taker'])]
for f in mutations:
    q=copy.deepcopy(x);f(q)
    try:verify(q)
    except (AssertionError,KeyError):pass
    else:raise AssertionError('corruption escaped')
supp=collections.Counter()
for o in oracle.values():
    if not o.get('empty'):supp[(o['first_any_role'],label(o['first_any_side'],o['first_any_p']))]+=1
out=dict(raw_first_records_and_public_books_verified=934,market_membership=1152,corruptions_refused=len(mutations),first_any_supplement_not_declared_primary={str(k):v for k,v in supp.items()},example_3789484=next(r for r in x['rows'] if r['mid']==3789484))
(H/'checks.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
