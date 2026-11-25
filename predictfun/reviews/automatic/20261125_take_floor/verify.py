"""Bounded independent recount on the five already-consumed days at 23282c33b.
Read all paired ledgers and the archived category responses for one complete market/day.
Rebuild settlement cash flows; compare the venue's buy-side minimum at requested size
for native complementary buys and a static fee-in-tokens sensitivity on fixed fills.
"""
from collections import Counter, defaultdict
from pathlib import Path
import gzip, hashlib, json, math, resource, sys
import pandas as pd
R=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(R/'predictfun/research'))
import pf_io
D=R/'data/predict_fun/research'
F=D/'take_exit_floor_20261125'
B=D/'take_exit_replay_v3_20261125'
resource.setrlimit(resource.RLIMIT_CPU,(60,61))
def native(s,p): return p if s==0 else 100-p
def small(s,p,q): return native(s,p)*q < 100-1e-7
assert small(1,99,50) and not small(1,98,50) and not small(0,25,4)
def check(x,y): assert math.isclose(x,y,abs_tol=1e-5),(x,y)
sm=json.loads((F/'summary.json').read_text())
v3=json.loads((B/'summary.json').read_text())
base={(str(r['day']),r['mid']):r['sha256'] for r in v3['rows'] if r['setting']=='STRICT58'}
rows=sm['rows']
assert len(base)==len(rows)==len({(str(r['day']),r['mid']) for r in rows})==408
out=defaultdict(Counter)
days=defaultdict(Counter)
meta={};ties=[]
for day in sorted({str(r['day']) for r in rows}):
    reg=pd.read_parquet(D/'step1'/day/'registry.parquet').set_index('mid')
    markets=[r for r in rows if str(r['day'])==day]
    example=markets[0]
    for _,__,___,pl in pf_io.rest_rows(f"{day}_{int(example['hour']):02d}",'/v1/categories/'):
        objs=[m for m in (pl.get('data') or {}).get('markets',[]) if m.get('id')==example['mid']]
        if objs:
            obj=objs[0]
            meta[day]=dict(mid=example['mid'],fields={k:v for k,v in obj.items() if any(w in k.lower() for w in ['delay','bump','lock'])},has_speedBump='speedBump' in obj)
            break
    for r in markets:
        mid=r['mid']
        u=reg.loc[mid]
        assert u.up_index==1 and u.outcome in ('Up','Down')
        y=0.5 if u.start_price==u.end_price else float(u.outcome=='Up')
        if y!=r['yup']: ties.append(dict(day=day,mid=mid,yup=r['yup'],canonical=y))
        check(float(u.outcome=='Up'),r['yup'])
        for arm,folder,fn,h in [('base',B,'STRICT58',base[day,mid]),('floor',F,'FLOOR',r['sha256'])]:
            raw=(folder/f'{mid}_{fn}.json.gz').read_bytes()
            assert hashlib.sha256(raw).hexdigest()==h
            rec=json.loads(gzip.decompress(raw))
            c=Counter(); pos=cash=token_fees=0.
            iocs=rec['iocs']; orders={o['gid']:o for o in rec['orders']}
            for o in orders.values():
                is_take=str(o['gid']) in iocs
                kind='ioc' if is_take else 'exit'
                c[kind+'_orders']+=1
                if small(o['side'],o['price'],o['size']): c[kind+'_small_orders']+=1
            for t,gid,s,q,p,it in rec['fills']:
                sign=1-2*s
                kind='ioc' if it<0 else 'exit'
                cost=q*.02*min(p,100-p) if it<0 else 0.
                cf=sign*q*(y*100-p)-cost
                c[kind+'_direct_usd']+=cf/100
                if it<0:
                    io=iocs[str(gid)]
                    check(io['act']-io['submit'],58)
                    assert (p<=io['limit'] if s==0 else p>=io['limit'])
                    token_fees+=cost/native(s,p)*(y if s==0 else 1-y)*100
                elif abs(pos+sign*q)>abs(pos)+1e-8:
                    c['exit_fills_increasing_abs_position']+=1
                o=orders[gid]
                if small(s,o['price'],o['size']):
                    c[kind+'_smallorder_filled_shares']+=q
                    c[kind+'_smallorder_direct_usd']+=cf/100
                pos+=sign*q;cash-=sign*q*p;c['fee_c']+=cost
            check(pos,rec['position']);check(cash,rec['cash']);check(c['fee_c'],rec['fees_c'])
            net=(cash+100*pos*y-c['fee_c'])/100
            c['tie_correction_usd']=pos*(y-r['yup']);check(net-c['tie_correction_usd'],r[arm]['settle_net_c']/100)
            c['settlement_usd']=net;c['native_fee_adjustment_fixed_fills_usd']=(c['fee_c']-token_fees)/100
            out[arm].update(c);days[day][arm]+=net
print(json.dumps(dict(sha='23282c33b600a34c7ca5f4d7210f5608d60be64c',markets=len(rows),census=dict(Counter(x['status'] for x in sm['census'])),category_samples=meta,ties=ties,arms=out,days=days),indent=2))
