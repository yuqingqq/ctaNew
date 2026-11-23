"""Pinned synthetic controls and saved-output review; no full build or replay."""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
import pandas as pd

ROOT=Path('/home/yuqing/ctaNew')
PIN=Path('/tmp/pf-queue-review-0b2b0b13')
BASE=Path('predictfun/reviews/maker_trace_20261122')
HERE=Path(__file__).resolve().parent
TOOLS=PIN/BASE/'tools'
sys.path.insert(0,str(TOOLS))
sys.dont_write_bytecode=True
E={'target':'0b2b0b13eb19f9f8c3df40d149a31c0b832905f8'}
start=time.process_time()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x): return None
    return x
def file(sub,name):
    p=PIN/BASE/'v2/chains'/sub/name
    return p if p.exists() else ROOT/BASE/'v2/chains'/sub/name
def csv(sub,name): return pd.read_csv(file(sub,name),float_precision='round_trip')
E['suites']=[]
for name in ('take_logic_v2_selftest','queue_front_selftest'):
    if '--reuse-suites' not in sys.argv:
        with (HERE/f'{name}.log').open('w') as out:
            r=subprocess.run([sys.executable,str(TOOLS/f'{name}.py')],env={**os.environ,'MT_RUN':'v2','PYTHONDONTWRITEBYTECODE':'1'},stdout=out,stderr=subprocess.STDOUT)
        rc=r.returncode
    else:
        rc=0 if '"failed": 0' in (HERE/f'{name}.log').read_text().splitlines()[-1] else 1
    E['suites'].append({'name':name,'rc':rc,'last_lines':(HERE/f'{name}.log').read_text().splitlines()[-2:]})
E['hashes']=[]
for sub in ('take_logic_v2','queue_front'):
    rec=json.loads(file(sub,'receipt.json').read_text())
    for name,meta in rec['outputs'].items():
        p=file(sub,name); E['hashes'].append({'sub':sub,'kind':'output','name':name,'ok':sha(p)==meta['sha256']})
    for name,meta in rec['sources'].items():
        p=PIN/name; expect=meta.get('sha256') if isinstance(meta,dict) else meta
        E['hashes'].append({'sub':sub,'kind':'source','name':name,'ok':p.exists() and sha(p)==expect})

# Hand-built, observationally indistinguishable hidden histories. The records and every public frame
# are identical. History A: A30 fills at3300, B20 at4100. History B: a behind-order cancels30 at3300,
# B20 fills4100, A30 fills4500 and a new30 replenishes before4600. FIFO allows B then A in history B.
import queue_front_selftest as ST
m=ST.MK(ST.DAY,919001,1)
m.setc(2000,3400,'bid',45,100).setc(3400,4200,'bid',45,70).setc(4200,30001,'bid',45,50)
a=m.fill(ST.A1,'bid',45,30,5000)
b=m.fill(ST.OTHER,'bid',45,20,5100)
part=m.part(); row=ST.lrow(part,'bid',45,2000,m)
histA=[(1900,100),(3300,-30),(4100,-20)]
histB=[(1900,100),(3300,-30),(4100,-20),(4500,-30),(4501,30)]
def observed(hist): return [sum(q for ts,q in hist if ts<=t-m.ws) for t in sorted(m.frames)]
book=[bk.get(('bid',45),0.) for _,bk in sorted(m.frames.items())]
assert observed(histA)==observed(histB)==book
assert a['lo']<=m.ws+4500<=a['hi'] and b['lo']<=m.ws+4100<=b['hi']
E['masked_timing_counterexample']={'code_result':ST._brief(row),'history_A_front':ST.A1L,'history_B_front':ST.OTHER,
    'public_frames_equal':True,'both_actual_match_times_inside_hard_windows':True,
    'wrongly_identified_in_history_B':row['status']=='front_identified' and row['holder']==ST.A1L,
    'history_A':histA,'history_B':histB,'relative_ms':True}

# An untouched 100-share order can partially fill twice while keeping the queue front.
n=ST.MK(ST.DAY,919002,2)
n.setc(2000,4600,'bid',45,100).setc(4600,8800,'bid',45,70).setc(8800,30001,'bid',45,50)
x=n.fill(ST.A1,'bid',45,30,5000); y=n.fill(ST.A1,'bid',45,20,9000); y['mk_hash']=x['mk_hash']
nm,nfv=n.build()
nm.orders=nm.orders.drop_duplicates(['scenario','mk_hash'])
nr=ST.QF.market_pass(n.day,n.mid,nm,nfv,None)['fronts']
E['same_order_partial_fill_counterexample']=nr[['sid','mk_hash','role','q']].to_dict('records')

lv=csv('queue_front','levels.csv'); fr=csv('queue_front','fronts.csv'); sh=csv('queue_front','shares.csv'); ec=csv('queue_front','economics.csv')
E['level_status']=lv.groupby(['unit','status']).size().reset_index(name='n').to_dict('records')
E['front_timing']=lv[lv.status=='front_identified'].front_timing.value_counts().to_dict()
keys=['day','mid','level_id']
first=fr[fr.role=='front'].groupby(keys).mk_hash.agg(set)
behind=fr[fr.role=='behind'].copy()
same=np.array([r.mk_hash in first.get((r.day,r.mid,r.level_id),set()) for r in behind.itertuples()])
E['real_later_same_hash']={'behind_rows':len(behind),'same_initial_front_hash':int(same.sum()),'same_hash_shares':float(behind.loc[same,'q'].sum()),'same_hash_cohort':int(behind.loc[same,'cohort'].sum())}
E['queue_economics_columns']=list(ec.columns)
E['queue_economics_pooled']=ec[ec.scope=='pooled'].to_dict('records')
E['shares_pooled']=sh[sh.scope=='pooled'].to_dict('records')
E['economics_recount']=[]
for role in ['front','behind']:
    z=fr[(fr.role==role)&fr.cohort]
    E['economics_recount'].append({'role':role,'all_rows':int((fr.role==role).sum()),'valid_rows':len(z),'valid_shares':float(z.q_c.sum()),'v_10s_c_sh':100*float(z.v_10s.sum())/float(z.q_c.sum())})

# Independently recompute the primary size rate from the saved panel, preserving raw event membership.
p=pd.read_parquet(file('take_logic_v2','panel.parquet')); ev=pd.read_parquet(file('take_logic_v2','events.parquet'))
rr=csv('take_logic_v2','ratios.csv')
rt=ev[ev.in_raw_schedule].copy(); gi=rt.gi.to_numpy(int); side=rt.s.to_numpy(int)
okside=~rt.status.fillna('').str.split('+').apply(lambda x:'side_mixed' in x).to_numpy()
E['take_outcomes']=[]; E['take_size_recount']=[]
for v,qkey,d in [('D0','q0','0'),('D1000','q1','1'),('pend','qp','0')]:
    q=p[qkey].to_numpy(float); sign=np.sign(np.where(abs(q)<=1e-6,0,q)); a5=sign*p['r_5s'+d].to_numpy(float)
    cat=np.select([~np.isfinite(a5),a5>=1,a5<=-1],[3,0,1],2)
    tl=p['tl'+d].to_numpy(float); tlb=np.select([tl>300,tl>240,tl>120,tl>60,tl>30,tl>10],[0,1,2,3,4,5],6)
    group=np.select([abs(q)>200,(abs(q)>1e-6)&(abs(q)<=50)],[1,0],-1)
    stratum=(p['rank'].to_numpy(int)-1)*28+cat*7+tlb
    for out,polarity in [('O1',-1),('O2',1)]:
        event=np.zeros(len(p),bool); event[gi[okside&(side*sign[gi]==polarity)]]=True
        E['take_outcomes'].append({'variant':v,'outcome':out,'event_seconds':int(event.sum()),'matches_saved':bool(np.array_equal(event,p[out+'_'+v].to_numpy(bool)))})
        for panel in ['full','clean']:
            for scope in ['rank1','pooled']:
                keep=(group>=0)&((p['rank'].to_numpy(int)==1) if scope=='rank1' else True)&(p.clean.to_numpy(bool) if panel=='clean' else True)
                tk=np.bincount(2*stratum[keep]+group[keep],minlength=168).reshape(-1,2)
                nk=np.bincount(2*stratum[keep]+group[keep],weights=event[keep],minlength=168).reshape(-1,2)
                informative=(tk>0).all(1); tk=tk[informative];nk=nk[informative]
                num=(nk[:,1]*tk[:,0]/tk.sum(1)).sum();den=(nk[:,0]*tk[:,1]/tk.sum(1)).sum();value=num/den
                z=rr[(rr.variant==v)&(rr.panel==panel)&(rr.scope==scope)&(rr.outcome==out)&(rr.ratio=='RR_size')].iloc[0]
                E['take_size_recount'].append({'key':[v,panel,scope,out],'computed':float(value),'saved':float(z.RR_MH),'ok':bool(np.isclose(value,z.RR_MH,rtol=1e-12,atol=1e-12))})
E['take_key_ratios']=rr[(rr.scope=='pooled')&rr.ratio.isin(['RR_size','RR_end','RR_burst'])&rr.outcome.isin(['O1','O2'])].to_dict('records')
E['cpu_s']=time.process_time()-start+resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
(HERE/'evidence.json').write_text(json.dumps(clean(E),indent=2)+'\n')
print(json.dumps(clean({'suites':E['suites'],'hash_checks':len(E['hashes']),'bad_hashes':[x for x in E['hashes'] if not x['ok']],
    'masked_timing_counterexample':E['masked_timing_counterexample'],'same_order_counterexample':E['same_order_partial_fill_counterexample'],
    'real_later_same_hash':E['real_later_same_hash'],'economics':E['economics_recount'],
    'bad_take_outcomes':[x for x in E['take_outcomes'] if not x['matches_saved']],'bad_take_recounts':[x for x in E['take_size_recount'] if not x['ok']], 'cpu_s':E['cpu_s']}),indent=2))
