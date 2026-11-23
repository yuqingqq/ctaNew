"""Post-hoc checks of the report's economic interpretation, not a new candidate or validation.
All saved rows, Oct24--27 only; exact cohort for fee and horizon comparisons.
"""
from pathlib import Path
import json,time,hashlib,signal
import numpy as np
import pandas as pd
PIN=Path('/tmp/pf-mt-review-1af85c56');ROOT=Path('/home/yuqing/ctaNew')
P=PIN/'predictfun/reviews/maker_trace_20261122';OUT=ROOT/'data/predict_fun/research/maker_trace_v1_20261122'
HERE=Path(__file__).resolve().parent; start=time.process_time();signal.setitimer(signal.ITIMER_PROF,60)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
rd=json.loads((P/'outputs/readings.json').read_text())
hashes={k:sha(OUT/k)==v['sha256'] for k,v in rd['inputs'].items()}
assert all(hashes.values())
claims=json.loads((P/'claims.json').read_text())
top={r['addr'] for r in json.loads((OUT/'econ/identities.json').read_text())['ex_ante']['top10_shares']}
consistent={r['addr'] for r in claims['addresses.exante_sh']['positive_every_day']}
md=pd.read_parquet(OUT/'econ/maker_day.parquet',columns=['day','addr','sh_mk','population'])
R={'target':'1af85c5691d5ca4c2f9565d94a13cd9680688d3c','input_hashes':hashes,'rows':[],'inventory':[]}
cols=['day','mid','addr','role','s','q','p','settle_ms','v_1s','net_res','st_res','st_1s','status','fee_dy','fee_dc','dpt']
for day in ['20261024','20261025','20261026','20261027']:
    makers=set(md[(md.day==day)&(md.population=='FILL')&(md.sh_mk>0)].addr)
    f=pd.read_parquet(OUT/'econ/fills_valued.parquet',columns=cols,filters=[('day','=',day),('status','=','ok')])
    for group,mask in [('in',f.addr.isin(top)),('field',f.addr.isin(makers-top))]:
        for role in ['maker_d0','taker']:
            m=mask & (f.st_1s=='ok')
            m&=((f.role=='maker')&(f.dpt=='d0')) if role=='maker_d0' else f.role=='taker'
            z=f[m];q=float(z.q.sum());mid=z.p+z.s*z.v_1s/z.q
            net1=z.v_1s+z.fee_dc+z.fee_dy*mid
            R['rows'].append(dict(day=day,group=group,role=role,n=len(z),q=q,gross_1s=float(z.v_1s.sum()),net_1s=float(net1.sum()),
                                  resolution_on_same_rows=float(z.net_res.sum()),gross_c_1s=100*z.v_1s.sum()/q if q else None,net_c_1s=100*net1.sum()/q if q else None))
    # Recorded matched trades only; excludes deposits/withdrawals and unobserved holdings. This is enough to refute 'flat' as an observed path.
    z=f[f.addr.isin(consistent)].sort_values(['addr','mid','settle_ms'],kind='stable').copy()
    z['dq']=z.s*z.q; z['pos']=z.groupby(['addr','mid'],sort=False).dq.cumsum()
    for addr,g in z.groupby('addr',sort=False):
        R['inventory'].append(dict(day=day,addr=addr,peak_abs_recorded_position=float(g.pos.abs().max()),
                                  max_abs_terminal_position=float(g.groupby('mid').pos.last().abs().max())))
    del f,z
table=pd.DataFrame(R['rows']);R['pooled']=[]
for (grp,role),g in table.groupby(['group','role']):
    q=g.q.sum();R['pooled'].append(dict(group=grp,role=role,q=q,net_c_1s=100*g.net_1s.sum()/q,gross_c_1s=100*g.gross_1s.sum()/q,
                                      resolution_c_same_rows=100*g.resolution_on_same_rows.sum()/q))
R['cpu_s']=time.process_time()-start
(HERE/'economic_probe.json').write_text(json.dumps(R,indent=2,default=str)+'\n')
print(json.dumps({k:R[k] for k in ['pooled','inventory','cpu_s']},indent=2))
