"""Descriptive maker3 first recorded fill and bounded posting-side audit.
See declaration.md; no private decision or entry-skill inference.
"""
import collections,csv,gzip,hashlib,json,math,resource,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
H=Path(__file__).resolve().parent;R=H.parents[2]
D=R/'data/predict_fun/research/maker_trace_v2_20261123'
ADDR='0x57FbC891E89184DA071AE9D97E745531cd7F718a'
DAYS=['20261024','20261025','20261026','20261027']
BAD={'bad_frame','mid_mismatch','out_of_order','off_grid'}
def normalize(quote,ix):
    assert quote in ('Bid','Ask') and ix in (1,2)
    return 'bid' if (quote=='Bid')==(ix==1) else 'ask'
def direction(side,p):
    if p is None:return 'unknown'
    if abs(p-.5)<1e-10:return 'neutral'
    return 'higher_probability' if (side=='bid')==(p>.5) else 'lower_probability'
def book_at(b,t):
    if b is None:return dict(status='no_frame',p=None)
    j=int(np.searchsorted(b['av'],t,side='right'))-1
    if j<0:return dict(status='no_frame',p=None)
    z={k:v[j].item() if hasattr(v[j],'item') else v[j] for k,v in b.items()}
    st='ok'
    if BAD.intersection(str(z['status']).split('+')):st='invalid_frame'
    elif not (math.isfinite(z['best_bid']) and math.isfinite(z['best_ask'])) or z['n_bid']<=0 or z['n_ask']<=0:st='one_sided'
    elif not (0<=z['best_bid']<z['best_ask']<=1):st='crossed_or_invalid'
    elif t-z['upd_ms']>1000:st='stale'
    return dict(status=st,p=(z['best_bid']+z['best_ask'])/2 if st=='ok' else None,availability_ms=z['av'],venue_ms=z['upd_ms'],age_ms=t-z['upd_ms'],bid=z['best_bid'],ask=z['best_ask'])
def window(b,lo,hi,side):
    if b is None:return dict(status='no_frame')
    times=np.r_[lo,b['av'][(b['av']>lo)&(b['av']<hi)],hi]
    states=[book_at(b,float(t)) for t in times]
    # Include just-before-next-frame age, so a hidden stale gap cannot pass.
    gap=any(book_at(b,float(t)-.001)['status']!='ok' for t in times[1:])
    if gap or any(x['status']!='ok' for x in states):return dict(status='incomplete_context',n=len(states))
    cats=sorted({direction(side,x['p']) for x in states})
    return dict(status='ok',n=len(states),p_min=min(x['p'] for x in states),p_max=max(x['p'] for x in states),directions=cats,stable_direction=cats[0] if len(cats)==1 else 'changes_or_neutral')
def postrow(f,posts,b,ws):
    p=posts.get(f['mk_hash'])
    if p is None:return dict(status='no_post_estimate')
    assert int(p['mid'])==int(f['mid']) and p['side']==f['y_side'] and int(p['tick'])==int(f['y_ptick'])
    out=dict(hash=f['mk_hash'],side=f['y_side'],grade=p['grade'],status='refused',first_record_s=(f['settle_ms']-ws)/1000)
    if p['estimate_enabled']=='True':
        lo,hi=float(p['effect_lo_ms']),float(p['effect_hi_ms']);assert lo<=hi<=f['settle_ms']
        out.update(status='enabled',post_lo_s=(lo-ws)/1000,post_hi_s=(hi-ws)/1000,context=window(b,lo,hi,f['y_side']))
    return out
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not math.isfinite(x):return None
    return x
def summarize(rows):
    out={}
    for day in DAYS+['pooled']:
        rr=[r for r in rows if day=='pooled' or r['day']==day]
        s=dict(markets=len(rr),statuses=dict(collections.Counter(r['status'] for r in rr)))
        z=[r for r in rr if r['status']=='observed_first_maker']
        s['first_sides']=dict(collections.Counter(r['first_side'] for r in z))
        s['preopen']=sum(r['first_record_s']<0 for r in z)
        s['prior_taker']=sum(r['prior_taker'] for r in z)
        s['first_record_s_quantiles']=dict(zip(['min','p25','median','p75','max'],np.quantile([r['first_record_s'] for r in z],[0,.25,.5,.75,1]))) if z else {}
        for ref in ['fill_context','opening_context']:
            s[ref]=dict(collections.Counter(r[ref+'_direction'] for r in z))
        s['native_first']=dict(collections.Counter(r['native_first'] for r in z))
        s['native_buy_probability']=dict(collections.Counter(r['native_buy_probability'] for r in z))
        s['post_first_context']=dict(collections.Counter(p.get('context',{}).get('stable_direction',p.get('context',{}).get('status',p['status'])) for r in z for p in r['first_posts']))
        s['bid_ask_post_relation']=dict(collections.Counter(r.get('bid_ask_post_relation','no_post_subset') for r in z))
        out[day]=s
    return clean(out)
def selftest():
    assert [normalize(q,i) for q,i in [('Bid',1),('Ask',1),('Bid',2),('Ask',2)]]==['bid','ask','ask','bid']
    assert direction('bid',.6)==direction('ask',.4)=='higher_probability'
    assert direction('bid',.4)==direction('ask',.6)=='lower_probability'
    assert direction('bid',.5)=='neutral' and direction('ask',None)=='unknown'
    b={k:np.array(v) for k,v in dict(av=[10.,20.],upd_ms=[10,20],best_bid=[.4,.7],best_ask=[.5,.8],n_bid=[1,1],n_ask=[1,1],status=['ok','ok']).items()}
    assert book_at(b,19)['p']==.45 and book_at(b,20)['p']==.75
    assert book_at(b,9)['status']=='no_frame' and book_at(b,1021)['status']=='stale'
    assert window(b,10,20,'bid')['stable_direction']=='changes_or_neutral'
    bad={k:v.copy() for k,v in b.items()};bad['av'][1]=2000;bad['upd_ms'][1]=2000
    assert window(bad,10,2000,'bid')['status']=='incomplete_context'
    # Known-bad future-asof and reversed-side implementations are discriminated.
    assert book_at(b,19)['p']!=book_at(b,20)['p']
    assert direction('bid',.6)!=direction('ask',.6)
    print('selftest: normalization, neutral, future frame, stale gap, changing direction pass')
def run():
    resource.setrlimit(resource.RLIMIT_CPU,(50,51))
    posts_path=R/'predictfun/reviews/maker_timing_estimates_20261124/outputs/post_estimates.csv.gz'
    with gzip.open(posts_path,'rt') as f:posts={r['order_hash']:r for r in csv.DictReader(f) if r['maker'].lower()==ADDR.lower()}
    rows=[];input_hashes={};counts={}
    def bind(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for s in iter(lambda:f.read(1<<20),b''):h.update(s)
        input_hashes[str(p.relative_to(R))]=h.hexdigest()
    bind(posts_path)
    for day in DAYS:
        t0=time.process_time()
        mp=D/f'markets_{day}.parquet';lp=D/f'legs_{day}.parquet';bp=D/f'tob_{day}.parquet'
        for p in [mp,lp,bp]:bind(p)
        m=pq.read_table(mp,columns=['mid','ws','status','n_poll_incomplete']).to_pandas()
        assert len(m)==288 and m.mid.is_unique
        L=pq.read_table(lp,filters=[[('mk_signer','=',ADDR)],[('tk_signer','=',ADDR)]],columns=['mid','sid','mk_idx','settle_ms','mk_hash','mk_signer','tk_signer','mk_quote','mk_index_set','mk_price','shares','y_side','y_price','y_ptick','status']).to_pandas()
        L=L[L.shares.gt(0)];L=L.sort_values(['mid','settle_ms','sid','mk_idx'])
        assert not L.duplicated(['mid','sid','mk_idx']).any()
        F=L[L.mk_signer.eq(ADDR)]
        for r in F[['mk_quote','mk_index_set','y_side']].drop_duplicates().to_dict('records'):assert normalize(r['mk_quote'],r['mk_index_set'])==r['y_side']
        normalized=np.where(F.mk_index_set.eq(1),F.mk_price,1-F.mk_price)
        assert np.allclose(normalized,F.y_price,atol=1e-6,rtol=0)
        fg={int(k):v for k,v in F.groupby('mid',sort=False)}
        tk=L[L.tk_signer.eq(ADDR)].groupby('mid').settle_ms.min().to_dict()
        max_t={int(r.mid):int(r.ws*1000) for r in m.itertuples()}
        for k,g in fg.items():
            max_t[k]=max(max_t[k],int(g.iloc[0].settle_ms))
            if any(p['day']==day and int(p['mid'])==k for p in posts.values()):
                for _,v in g.groupby('y_side'):
                    h=v.iloc[0];max_t[k]=max(max_t[k],int(h.settle_ms))
        B=pq.read_table(bp,columns=['mid','upd_ms','recv_ns','best_bid','best_ask','n_bid','n_ask','status']).to_pandas()
        B['av']=np.maximum(B.upd_ms.to_numpy(),B.recv_ns.to_numpy()/1e6)
        B=B[B.av<=B.mid.map(max_t)]
        B=B.sort_values(['mid','av','upd_ms','recv_ns'])
        books={int(k):{c:g[c].to_numpy() for c in ['av','upd_ms','best_bid','best_ask','n_bid','n_ask','status']} for k,g in B.groupby('mid',sort=False)}
        for mm in m.to_dict('records'):
            mid=int(mm['mid']);ws=int(mm['ws']*1000);g=fg.get(mid)
            r=dict(day=day,mid=mid,ws_ms=ws,market_status=mm['status'],n_poll_incomplete=int(mm['n_poll_incomplete']),status='no_maker_fill')
            if g is None:rows.append(r);continue
            ft=int(g.iloc[0].settle_ms);first=g[g.settle_ms.eq(ft)];sides=sorted(first.y_side.unique())
            side=sides[0] if len(sides)==1 else 'both'
            b=books.get(mid);fc=book_at(b,ft-1);oc=book_at(b,ws)
            src_ok=first.status.eq('ok').all();raw=first.to_dict('records')
            r.update(status='observed_first_maker',first_record_ms=ft,first_record_s=(ft-ws)/1000,first_side=side,first_source_ok=bool(src_ok),first_legs=raw,prior_taker=tk.get(mid,float('inf'))<ft,first_taker_record_s=(tk[mid]-ws)/1000 if mid in tk else None,fill_context=fc,opening_context=oc,first_posts=[postrow(x,posts,b,ws) for x in first.drop_duplicates('mk_hash').to_dict('records')])
            for name,c in [('fill_context',fc),('opening_context',oc)]:
                r[name+'_direction']='flagged_source' if not src_ok else 'preopen_not_comparable' if name=='opening_context' and ft<=ws else 'both_sides' if side=='both' else direction(side,c['p'])
            quotes=sorted(first.mk_quote.unique());r['native_first']=quotes[0] if len(quotes)==1 else 'mixed'
            if not src_ok:r['native_buy_probability']='flagged_source'
            elif r['native_first']!='Bid':r['native_buy_probability']='not_native_buy_only'
            else:r['native_buy_probability']='both_outcomes' if len(sides)>1 else direction(side,fc['p'])
            if any(p['status']!='no_post_estimate' for p in r['first_posts']):
                pp={s:postrow(v.iloc[0].to_dict(),posts,b,ws) for s,v in g.groupby('y_side')}
                r['first_filled_side_orders']=pp
                if set(pp)!=set(['bid','ask']):rel='only_one_filled_side'
                elif any(p['status']!='enabled' for p in pp.values()):rel='refused_or_unknown'
                elif pp['bid']['post_hi_s']<pp['ask']['post_lo_s']:rel='bid_before_ask_conditional'
                elif pp['ask']['post_hi_s']<pp['bid']['post_lo_s']:rel='ask_before_bid_conditional'
                else:rel='overlapping_windows'
                r['bid_ask_post_relation']=rel
            rows.append(r)
        counts[day]=dict(maker_rows=len(F),book_rows_retained=len(B),cpu_s=time.process_time()-t0)
        print(day,counts[day],flush=True)
        del L,F,B,books,fg
    result=clean(dict(asof='2026-11-24',address=ADDR,rows=rows,summary=summarize(rows),inputs=input_hashes,counts=counts))
    (H/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (H/'summary.json').write_text(json.dumps(result['summary'],indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2))
if __name__=='__main__':
    selftest()
    if '--run' in sys.argv:run()
