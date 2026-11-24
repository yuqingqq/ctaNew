"""User-selected first-minute, four 15s windows in market 3789484.
Descriptive accounting on the verified display projection, not a new candidate
or validation. All identity roles retained. FIFO runs from zero at market start
across window boundaries; no reset, no inferred causal order pairing. Amounts
inherit the payload's six-decimal rounding. Exact match order is bounded for
recorded makers, so FIFO is settlement-order accounting only.
"""
import collections,copy,gzip,json,math
from pathlib import Path
H=Path(__file__).resolve().parent
D=json.loads(gzip.decompress((H/'data.json.gz').read_bytes()))
M=next(x for x in D['cases'] if x['id']==3789484)
def fifo(fs):
 lots=collections.deque(); closes=[];cash=pos=fees=0
 for f in fs:
  side=1 if f['side']==0 else -1;q=f['q'];p=f['price'];fee=f.get('fee',0)*100
  cash-=side*q*p;pos+=side*q;fees+=fee
  while q>1e-9 and lots and lots[0]['side']!=side:
   l=lots[0];n=min(q,l['q']);v=l['side']*n*(p-l['p'])
   closes.append(dict(entry=l['f'],exit=f,q=n,gross_c=v,entry_side=l['side'],hold_s=f['t']-l['f']['t'],hard_ordered=l['f']['t']<(f.get('lo') if f.get('lo') is not None else f['t'])))
   q-=n;l['q']-=n
   if l['q']<1e-9:lots.popleft()
  if q>1e-9:lots.append(dict(side=side,q=q,p=p,f=f))
  # Exact accounting identity at arbitrary fixed marks; independent of the
  # arbitrary FIFO choice of which opposite fills to pair.
  for mark in [0,50,100]:
   residual=sum(l['side']*l['q']*(mark-l['p']) for l in lots)
   assert abs(cash+pos*mark-(sum(c['gross_c'] for c in closes)+residual))<1e-5
 return closes,list(lots),cash,pos,fees
def selftest():
 def f(t,side,q,p):return dict(t=t,side=side,q=q,price=p)
 cases=[([f(0,0,10,40),f(20,1,4,45)],20,6),([f(0,1,10,60),f(20,0,4,55)],20,-6),([f(0,0,10,40),f(20,1,15,45),f(30,0,5,43)],60,0)]
 for fs,pnl,pos in cases:
  c,l,_,i,_=fifo(fs);assert sum(x['gross_c'] for x in c)==pnl and i==pos
 fs=[f(0,0,10,60),f(20,1,10,40)];assert sum(x['gross_c'] for x in fifo(fs)[0])==-200
selftest()
out={}
for name,fs in [('maker3',[x for x in M['makers'] if x['rank']==3]),('ours',M['ours']['CURRENT_ASYNC_FAST10']['fills'])]:
 fs=[dict(x,role=x.get('role','maker')) for x in fs if 0<=x['t']<60]
 cs,lots,cash,pos,fees=fifo(fs);rows=[]
 for start in [0,15,30,45]:
  group=[f for f in fs if start<=f['t']<start+15]
  row=dict(start_s=start,end_s=start+15,legs=len(group),taker_legs=sum(f['role']=='taker' for f in group),self_legs=sum(f.get('self_trade',False) for f in group))
  for side,lab in [(0,'buy'),(1,'sell')]:
   g=[f for f in group if f['side']==side];q=sum(f['q'] for f in g)
   key='1s' if name=='maker3' else '1000';valued=[f for f in g if f['markouts'][key] is not None];vq=sum(f['q'] for f in valued)
   row[lab]=dict(shares=q,vwap_c=sum(f['q']*f['price'] for f in g)/q if q else None,min_c=min((f['price'] for f in g),default=None),max_c=max((f['price'] for f in g),default=None),markout_1s_c_sh=sum(f['q']*f['markouts'][key] for f in valued)/vq if vq else None,markout_valid_sh=vq)
  c=[x for x in cs if start<=x['exit']['t']<start+15];q=sum(x['q'] for x in c)
  row['fifo_closes']=dict(shares=q,gross_usd=sum(x['gross_c'] for x in c)/100,cents_per_matched_share=sum(x['gross_c'] for x in c)/q if q else None,buy_then_sell_shares=sum(x['q'] for x in c if x['entry_side']==1),sell_then_buy_shares=sum(x['q'] for x in c if x['entry_side']==-1),hard_ordered_shares=sum(x['q'] for x in c if x['hard_ordered']))
  row['cum_net_flow_shares']=sum((1-2*f['side'])*f['q'] for f in fs if f['t']<start+15)
  rows.append(row)
 book=next(r for r in M['book'] if r[0]==60);mark=(book[1]+book[2])/2
 out[name]=dict(windows=rows,total=dict(fifo_gross_usd=sum(x['gross_c'] for x in cs)/100,matched_shares=sum(x['q'] for x in cs),net_flow_shares=pos,fees_usd=fees/100,mark60_c=mark,residual_usd=sum(l['side']*l['q']*(mark-l['p']) for l in lots)/100,total_marked_usd=(cash+pos*mark-fees)/100),fills=fs,closes=cs)
print(json.dumps({n:{k:v for k,v in x.items() if k not in ['fills','closes']} for n,x in out.items()},indent=2))
(H/'first_minute_turnover.json').write_text(json.dumps(out,indent=2)+'\n')
