"""Independent saved-event recount, population and all-cell/contrast validation."""
from collections import defaultdict
import json
import math
import numpy as np
import pandas as pd
import build as B

def eq(a,b,message):
    B.need((pd.isna(a) and pd.isna(b)) or math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12), message)

def validate_map(m):
    B.need(not m.loc[m.layer!='identified_maker_take','creator_identified'].any(), 'maker attribution')
    for r in m.itertuples():
        B.need(0<=r.e1<=r.n1 and 0<=r.e0<=r.n0, 'event count')
        eq(r.exposed_rate,r.e1/r.n1 if r.n1 else np.nan,'rate')
        eq(r.reference_rate,r.e0/r.n0 if r.n0 else np.nan,'rate')

def events(p,e):
    n=len(p); anytake=np.zeros(n,int)
    raw=e[e.in_raw_schedule.astype(bool)]
    for gi in raw.gi.astype(int): anytake[gi]=1
    B.need(np.array_equal(anytake,p.O3),'raw any-take schedule')
    for v,col in [('D0','q0'),('D1000','q1'),('pend','qp')]:
        opp=np.zeros(n,int); same=np.zeros(n,int)
        for r in raw.itertuples():
            gi=int(r.gi); q=float(p.iloc[gi][col]); s=float(r.s)
            # Unknown-side status was declared by the accepted raw-side builder.
            direction=getattr(r,'dir_'+v)
            if direction=='side_unknown': continue
            if abs(q)<=1e-6: continue
            if not np.isfinite(s): continue
            if s*np.sign(q)<0: opp[gi]=1
            elif s*np.sign(q)>0: same[gi]=1
        B.need(np.array_equal(opp,p['O1_'+v]),'raw opposing schedule '+v)
        B.need(np.array_equal(same,p['O2_'+v]),'raw same-direction schedule '+v)

def recount_cells(x,saved):
    base=x.to_dict('records')
    for scope in ['day_rank','rank','pooled']:
        c={}
        for r in base:
            key=(str(r['day']) if scope=='day_rank' else 'all',r['rank'] if scope!='pooled' else 0,*[r[k] for k in B.STATES])
            if key not in c: c[key]=[0,set(),0,0,0,0]
            z=c[key]; z[0]+=1; z[1].add(r['mid'])
            for i,o in enumerate(B.OUTCOMES): z[2+i]+=r[o]
        ss=saved[saved.scope==scope]
        B.need(len(ss)==len(c),'cell membership')
        for r in ss.to_dict('records'):
            key=(str(r['day']),r['rank'],*[r[k] for k in B.STATES]); z=c.pop(key)
            B.need(r['seconds']==z[0] and r['markets']==len(z[1]),'cell denominator')
            B.need(r['thin']==(z[0]<30 or z[4]<5),'thin flag')
            for i,o in enumerate(B.OUTCOMES):
                B.need(r[o]==z[2+i],'cell outcome')
                eq(r[o+'_rate'],z[2+i]/z[0],'cell rate')
        B.need(not c,'omitted cell')

def recount_matched(x,saved):
    data=defaultdict(lambda:[[],[]])
    eligible=[]
    for r in x.to_dict('records'):
        if not r['q_nonzero'] or r['prior_fills']=='1-2' or r['btc']=='missing': continue
        e=int(r['prior_fills']=='>=3'); eligible.append((r,e))
        key=(str(r['day']),r['mid'],r['rank'],r['inventory'],r['btc'],r['time_left'])
        data[key][e].append(r)
    for r in saved.to_dict('records'):
        o=r['outcome']; day=str(r['day']); a=b=ne=nr=0; w=tot=0.; k=0; markets=set()
        for key, (zero,one) in data.items():
            if day!='all' and key[0]!=day: continue
            if not zero or not one: continue
            wt=len(zero)*len(one)/(len(zero)+len(one)); ae=sum(v[o] for v in one); be=sum(v[o] for v in zero)
            a+=ae; b+=be; ne+=len(one); nr+=len(zero); w+=wt; tot+=wt*(ae/len(one)-be/len(zero));k+=1;markets.add(key[1])
        B.need((r['exposed_events'],r['reference_events'],r['exposed_common'],r['reference_common'],r['strata'],r['markets'])==(a,b,ne,nr,k,len(markets)),'matched count')
        eq(r['weight'],w,'matched weight');eq(r['rd'],tot/w if w else np.nan,'matched RD')
        for flag,col in [(1,'exposed_all'),(0,'reference_all')]:
            B.need(r[col]==sum(e==flag and (day=='all' or str(v['day'])==day) for v,e in eligible),'eligible denominator')

def run():
    B.bound();out=B.HERE/'outputs'
    for prefix,folder in [('take','take_logic_v2'),('post','posting_opportunity'),('ct','competing_transition')]:
        receipt=json.loads(B.INPUTS[prefix+'_receipt'].read_text())
        for key,path in B.INPUTS.items():
            if path.parent.name==folder and path.name!='receipt.json':
                expected=receipt['outputs'][path.name]['sha256']
                B.need(B.digest(path)==expected,'original receipt binding '+key)
    rec=json.loads((out/'receipt.json').read_text())
    for name,sha in rec['hashes'].items(): B.need(B.digest(out/name)==sha,'output hash')
    p=pd.read_parquet(B.INPUTS['take_panel']);q=pd.read_parquet(B.INPUTS['post_panel']);e=pd.read_parquet(B.INPUTS['take_events'])
    B.validate_population(p,17100,['day','mid','rank']);B.validate_population(q,11400,['day','mid','side']);events(p,e)
    cells=B.csv(out/'joint_states.csv.gz');seq=B.csv(out/'sequence_contrast.csv');m=B.csv(out/'evidence_map.csv');validate_map(m)
    B.need(len(m)==224,'all frozen comparisons')
    B.need(len(m[m.layer=='identified_maker_take'])==144,'take comparison membership')
    B.need(len(m[m.layer=='anonymous_market_creation'])==70,'post comparison membership')
    for v in B.TL.VARIANTS:
        for pn in ['full','clean']:
            x=B.state_frame(p,v,pn)
            recount_cells(x,cells[(cells.variant==v)&(cells.panel==pn)])
            recount_matched(x,seq[(seq.variant==v)&(seq.panel==pn)])
    # Frozen selection/ratio fields are exact copies of the hash-bound tables.
    pd.testing.assert_frame_equal(m,B.make_map(),check_dtype=False,rtol=1e-14,atol=1e-14)
    failures={}
    for name,fn in [
        ('missing_opportunity',lambda:B.validate_population(p.iloc[1:],17100,['day','mid','rank'])),
        ('false_creator',lambda:validate_map(m.assign(creator_identified=True))),
        ('event_count',lambda:validate_map(m.assign(e1=m.n1+1))),
        ('changed_raw_action',lambda:events(p.assign(O3=1-p.O3),e))]:
        try: fn()
        except ValueError as exc: failures[name]=str(exc)
        else: raise AssertionError('falsifier escaped: '+name)
    result=dict(pass_all=True,maker_seconds=len(p),market_side_seconds=len(q),map_rows=len(m),joint_cells=len(cells),sequence_rows=len(seq),falsifiers=failures)
    (B.HERE/'checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__': run()
