"""Independent bounded review: pinned synthetic suites and saved-output arithmetic, no measurement build."""
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

ROOT = Path('/home/yuqing/ctaNew')
PIN = Path('/tmp/pf-takelogic-review-cccac366')
BASE = Path('predictfun/reviews/maker_trace_20261122')
HERE = Path(__file__).resolve().parent
E = {'target': 'cccac3665b967fc4858b2335add811779918b87a'}
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [clean(v) for v in x]
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x): return None
    return x
def read(name, sub='take_logic'):
    return pd.read_csv(PIN/BASE/'v2/chains'/sub/name, float_precision='round_trip')

start = time.process_time()
E['suites'] = []
for name in ('chain_correct_post_selftest','chain_correct_pull_take_selftest','take_logic_selftest'):
    p = PIN/BASE/'tools'/f'{name}.py'
    with (HERE/f'{name}.log').open('w') as out:
        r = subprocess.run([sys.executable,str(p)], env={**os.environ,'MT_RUN':'v2','PYTHONDONTWRITEBYTECODE':'1'}, stdout=out,stderr=subprocess.STDOUT)
    log=(HERE/f'{name}.log').read_text()
    E['suites'].append({'name':name,'rc':r.returncode,'tail':log.splitlines()[-2:]})

# Verify small receipt-named output/source bytes. Large upstream inputs remain hash-bound in receipts,
# and are not scanned just to repeat the earlier acceptance.
E['artifact_checks'] = []
for sub, recname in [('take_logic','receipt.json'),('correction_v1','receipt_post.json'),('correction_v1','receipt_pull_take.json')]:
    rec=json.loads((PIN/BASE/'v2/chains'/sub/recname).read_text())
    for name, meta in rec.get('outputs',{}).items():
        if isinstance(meta,dict): expect=meta.get('sha256')
        else: expect=meta
        if not expect: continue
        p=PIN/BASE/'v2/chains'/sub/name
        if not p.exists(): p=ROOT/BASE/'v2/chains'/sub/name
        E['artifact_checks'].append({'receipt':recname,'kind':'output','path':str(p),'ok':p.exists() and sha(p)==expect})
    for name, expect in rec.get('sources',{}).items():
        p=PIN/name
        if isinstance(expect,dict): expect=expect.get('sha256')
        E['artifact_checks'].append({'receipt':recname,'kind':'source','path':name,'ok':p.exists() and sha(p)==expect})

h=read('hazard_cells.csv'); rr=read('ratios.csv'); co=read('consistency.csv'); pat=read('patterns.csv')
E['qbin_counts']=h[(h.table=='marginal')&(h.factor=='qbin')].to_dict('records')
E['uncertainty_counts']=h[(h.table=='marginal')&h.factor.isin(['inv_timing_sensitive','inv_ledger_conditional'])].to_dict('records')
E['patterns_status']=pat[pat.pattern.str.startswith('status:')].to_dict('records')
E['inventory']=read('inventory_at_take.csv').to_dict('records')
E['end_rank1']=read('end_of_market.csv').query("kind == 'pair' and rank == 1").to_dict('records')

# Independently reduce joint qbin/sign/BTC/time-left cells back to MH strata.
# Post-hoc ANY/timing-sensitive contrasts are diagnostic, not a newly declared candidate.
c=h[(h.table=='cell')&(h.scope!='pooled')].copy()
c[['qb','sign','btc','tl']]=c.level.str.split('|',expand=True)
def mh(scope, ratio, typ):
    x=c if scope=='pooled' else c[c.scope==scope]
    x=x[x.qb!='0'].copy()
    if ratio=='RR_size':
        x=x[x.qb.isin(['(0,50]','(200,500]','>500'])].copy()
        x['g']=(x.qb!='(0,50]').astype(int); keys=['scope','btc','tl']
    elif ratio=='RR_end':
        x=x[x.tl.isin(['[0,10]','(10,30]','(60,120]','(120,240]'])].copy()
        x['g']=x.tl.isin(['[0,10]','(10,30]']).astype(int); keys=['scope','qb','btc']
    else:
        expose='with' if ratio=='RR_with_5s' else 'against'
        x=x[x.btc.isin([expose,'flat'])].copy(); x['g']=(x.btc==expose).astype(int); keys=['scope','qb','tl']
    agg=x.groupby(keys+['g'])[['maker_seconds','event_seconds_'+typ]].sum()
    a=b=0.
    for _, group in agg.groupby(level=list(range(len(keys)))):
        z=group.reset_index()
        if len(z)!=2: continue
        t0,t1=[float(z.loc[z.g==g,'maker_seconds'].iloc[0]) for g in (0,1)]
        e0,e1=[float(z.loc[z.g==g,'event_seconds_'+typ].iloc[0]) for g in (0,1)]
        a+=e1*t0/(t0+t1); b+=e0*t1/(t0+t1)
    return a/b if b else None
E['independent_MH']=[]
for sc in ['rank1','pooled']:
    for ratio in ['RR_size','RR_end','RR_with_5s','RR_against_5s']:
        for ty in ['reduce','add']:
            v=mh(sc,ratio,ty)
            stored=float(rr[(rr.variant=='D0')&(rr.scope==sc)&(rr.type==ty)&(rr.ratio==ratio)].RR_MH.iloc[0])
            E['independent_MH'].append({'scope':sc,'ratio':ratio,'type':ty,'computed':v,'stored':stored,'ok':np.isclose(v,stored,atol=1e-12,rtol=1e-12)})
E['posthoc_size_by_event_type']=[{'scope':sc,'type':ty,'RR_MH':mh(sc,'RR_size',ty)} for sc in ['rank1','pooled'] for ty in ['reduce','add','timing_sensitive','any']]
E['consistency_recount']=[]
for (v,s,t,r), g in co[co.row=='market'].groupby(['variant','scope','type','ratio']):
    sup=(g.a+g.b>=1)&(g.T1>=30)&(g.T0>=30)
    greater=sup & ((g.a*g.T0)>(g.b*g.T1))
    saved=rr[(rr.variant==v)&(rr.scope==s)&(rr.type==t)&(rr.ratio==r)].iloc[0]
    E['consistency_recount'].append({'key':[v,s,t,r],'ok':int(sup.sum())==saved.n_markets_supported and int(greater.sum())==saved.n_markets_rr_gt1})
E['tail_count_check']=bool(np.allclose(rr.p_hi,(rr.c_hi+0.)/501, equal_nan=True))
z=read('census_graded.csv','correction_v1')
E['census_columns']=list(z.columns)
E['census_pooled']=z[(z['scope'].astype(str)=='pooled')&(z['day'].astype(str)=='all')].to_dict('records')

E['process_cpu_s']=time.process_time()-start
E['child_cpu_s']=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
(HERE/'evidence.json').write_text(json.dumps(clean(E),indent=2)+'\n')
print(json.dumps({'suites':E['suites'],'artifact_checks':len(E['artifact_checks']),'artifact_bad':[x for x in E['artifact_checks'] if not x['ok']], 'MH_bad':[x for x in E['independent_MH'] if not x['ok']], 'consistency_bad':[x for x in E['consistency_recount'] if not x['ok']], 'posthoc_size':E['posthoc_size_by_event_type'],'cpu_s':E['process_cpu_s']+E['child_cpu_s']},indent=2))
