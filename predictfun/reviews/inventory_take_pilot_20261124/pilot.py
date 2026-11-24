"""QUOTE-ACCUMULATE-REDUCE-v1. See contract.md; no frozen source edited."""
import collections,copy,gzip,hashlib,json,math,resource,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent;R=H.parents[2]
sys.path.insert(0,str(R/'predictfun/reviews/opening_entry_pilot_20261124'))
import opening as O
X=O.X;AF=X.AF;P=X.P;M=X.M;LB=X.LB;BE=X.BE
OUT=X.DATA/'inventory_take_pilot_20261124'
ARMS={'PASSIVE_100':(100,False),'TAKE_100':(100,True),'PASSIVE_220':(220,False),'TAKE_220':(220,True)}
EPS=1e-9
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not math.isfinite(x):return None
    return x
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')
def sha(p):return X.sha(p)
def requested_qty(I,B,A,Q,S,R_):
    if abs(I)<S-EPS:return None,0.
    side=1 if I>0 else 0
    room=(2*Q+I-A) if side==1 else (2*Q-I-B)
    return side,max(0.,float(math.floor(min(Q,abs(I)-R_,room)+EPS)))
def fee_c(price,qty,bps):return bps/10000.*min(price,100-price)*qty
def view_levels(frame):
    return {(s,float(frame[2+s*10+i])):float(frame[7+s*10+i]) for s in (0,1) for i in range(5) if 0<=frame[2+s*10+i]<=100}
class IOCEngine(AF.ProfileEngine):
    def __init__(self,path,end,bps):
        super().__init__(path,end,False,AF.PROFILES['ASYNC_FAST10'])
        self.pending=[];self.iocs={};self.fee_bps=float(bps);self.fees=[];self.depth={};self.available={};self.lj=-1;self.exec_log=[]
    def submit_ioc(self,g,s,q,limit):
        gid=len(self.o);act=float(g+10.)
        # Passive kernel never matches this order: IOC fills are inserted only
        # by execute_ioc at its actual effect timestamp.
        self.o.append(dict(gid=gid,side=s,rung=-1.,price=float(limit),size=float(q),rem=0.,ahead=0.,act=act,remove=act,
            submit=float(g),new=False,ci=0,refused=False,req=None,why='IOC'))
        self.iocs[gid]=dict(gid=gid,side=s,q=float(q),limit=float(limit),submit=float(g),act=act,ack=act+BE.FILL_NOTICE,done=False)
        self.pending.append(gid);return gid
    def liquidity(self,t,s,price):
        j=int(np.searchsorted(self.f[:,0],t,side='right'))-1
        if j<0:return 0.,-1,0.
        while self.lj<j:
            self.lj+=1;new=view_levels(self.f[self.lj])
            for k in self.depth.keys()|new.keys():
                shown=new.get(k,0.);old=self.depth.get(k,0.)
                self.available[k]=max(0.,min(shown,self.available.get(k,0.)+shown-old))
            self.depth=new
        lo=int(np.searchsorted(self.tr[:,0],self.f[j,0],side='right'));hi=int(np.searchsorted(self.tr[:,0],t,side='right'))
        tr=self.tr[lo:hi];debit=float(tr[(tr[:,1]==s)&((tr[:,2]<=price) if s==0 else (tr[:,2]>=price)),3].sum())
        return max(0.,self.available.get((s,price),0.)-debit),j,debit
    def execute_ioc(self,gid):
        a=self.iocs[gid];t=a['act'];s=a['side'];j=int(np.searchsorted(self.f[:,0],t,side='right'))-1
        row=dict(a,filled=0.,price=None,fee_c=0.,frame=j,reason='empty',available_before=None,public_debit=None)
        pos=sum((1-2*f[2])*f[3] for f in self.fills);row['venue_before']=pos
        if j<0:row['reason']='no_venue_book'
        else:
            f=self.f[j];opp=1-s;p=float(f[2+opp*10]);row['venue_frame_ms']=float(f[0]);row['venue_received_ms']=float(f[1])
            if not (0<=f[2]<f[12]<=100) or t-f[0]>1000:row['reason']='invalid_or_stale'
            elif not 1<=p<=99:row['reason']='price_boundary'
            elif (p>a['limit'] if s==0 else p<a['limit']):row['reason']='limit_moved'
            elif any(o['gid'] not in self.iocs and o['side']==opp and o['rem']>EPS and o['act']<=t<o['remove'] and
                     (o['price']<=a['limit'] if s==0 else o['price']>=a['limit']) for o in self.o):row['reason']='own_cross'
            else:
                available,j,debit=self.liquidity(t,opp,p);q=min(a['q'],available)
                row.update(available_before=available,public_debit=debit,frame=j,price=p)
                if q>EPS:
                    self.available[(opp,p)]-=q
                    self.fills.append((t,gid,s,q,p,-1));fee=fee_c(p,q,self.fee_bps);self.fees.append((t,gid,fee))
                    row.update(filled=q,fee_c=fee,reason='filled' if q>=a['q']-EPS else 'partial')
        row['venue_after']=pos+(1-2*s)*row['filled'];row['absolute_risk_reduced']=abs(row['venue_after'])<abs(pos)-EPS
        self.exec_log.append(row);a['done']=True;self.pending.remove(gid)
    def advance(self,grid):
        limit=min(grid,self.end)
        for gid in sorted([i for i in self.pending if self.iocs[i]['act']<=limit],key=lambda i:(self.iocs[i]['act'],i)):
            super().advance(self.iocs[gid]['act']);self.execute_ioc(gid)
        super().advance(grid)
class TakeBranch(X.FixedBranch):
    def __init__(self,path,end,ident,q,bps,enabled):
        super().__init__(path,end,ident,False,'ASYNC_FAST10')
        assert not self.E.o
        self.E=IOCEngine(path,end,bps);self.take_enabled=enabled;self.take_open=[];self.take_decisions=[];self.last_take_state=None
        self.q0=float(q);self.cap=float(2*q);self.inv=dict(self.inv,Qside=2*q,S=q,R=q/2)
        self.settings=dict(self.settings,q0=self.q0,cap=self.cap,inventory=self.inv,inventory_take=enabled,taker_fee_bps=bps,
            taker_liquidity='net_depth_budget_v1',ioc_notice_ms=100.)
    def reserve(self,g):
        I,B,A=super().reserve(g);res=[B,A]
        for gid in self.take_open:
            a=self.E.iocs[gid];res[a['side']]+=a['q']-self.known_filled(gid,g)
        return I,res[0],res[1]
    def notices(self,g):
        super().notices(g)
        self.take_open=[i for i in self.take_open if self.E.iocs[i]['ack']>g]
    def side_view(self,g,s):
        v=super().side_view(g,s)
        v['opp_prices'] += [self.E.iocs[i]['limit'] for i in self.take_open if self.E.iocs[i]['side']==1-s]
        return v
    def take_step(self,g):
        if not self.take_enabled:return
        I,B,A=self.reserve(g);side,q=requested_qty(I,B,A,self.q0,self.inv['S'],self.inv['R'])
        if q<1:return
        row=dict(t=float(g),known_I=I,B=B,A=A,side=side,q=q,status='candidate',gid=None)
        if self.take_open:row['status']='pending_ioc'
        elif g+10>=self.end:row['status']='near_end'
        else:
            v=self.side_view(g,side);limit=v['ask'] if side==0 else v['bid'];row['limit']=limit;row['book_received_ms']=v['received']
            signature=(I,side,limit,*self.E.f[self.E.known,2:].tolist())
            if not v['book_ok'] or g-v['received']>1000:row['status']='no_fresh_book'
            elif any(limit>=p if side==0 else limit<=p for p in v['opp_prices']):row['status']='own_cross'
            elif signature==self.last_take_state:row['status']='unchanged_state'
            else:
                gid=self.E.submit_ioc(g,side,q,limit);self.take_open.append(gid);self.last_take_state=signature;row.update(gid=gid,status='sent')
        self.take_decisions.append(row);self.check_position(g)
def run_one(path,cache,old,edges,q,bps,enabled):
    B=TakeBranch(path,old['end'],old['ident'],q,bps,enabled);pol=O.OpeningGate(B,cache,edges)
    grid=[]
    for k in range(LB.n_ticks(B)):
        LB.step(B,k,pol);g=k*BE.STEP;B.take_step(g)
        cash,pos,_=B.E.value(B.mark);I,RB,RA=B.reserve(g)
        live=[sum(o['side']==s and o['gid'] not in B.E.iocs and not o['refused'] and o['act']<=g<o['remove'] and o['rem']>EPS for o in B.E.o) for s in (0,1)]
        grid.append([g,I,RB,RA,pos,cash,*live])
    lg=LB.finish(B);rec=M.ledger_record(B,lg)
    rec.update(commands=clean(B.cmd_log),guard_log=clean(pol.log),scope_log=pol.scope_log,opening_batch_ms=O.initial_tick([o for o in B.E.o if o['gid'] not in B.E.iocs]),
        iocs=clean(B.E.iocs),ioc_decisions=B.take_decisions,ioc_results=B.E.exec_log,fees_c=sum(f[2] for f in B.E.fees),grid=grid)
    rec['net_value_c']=rec['value']-rec['fees_c'];rec['fee2_net_value_c']=rec['value']-2*rec['fees_c']
    metric=X.fifo(rec['fills'],rec['mark']);g=np.asarray(grid)
    metric.update(net_c=rec['net_value_c'],fee2_net_c=rec['fee2_net_value_c'],fees_c=rec['fees_c'],maker_shares=sum(f[3] for f in rec['fills'] if f[5]>=0),
        taker_shares=sum(f[3] for f in rec['fills'] if f[5]<0),take_requests=len(rec['iocs']),takes_filled=sum(x['filled']>EPS for x in rec['ioc_results']),
        risk_reducing_takes=sum(x['absolute_risk_reduced'] for x in rec['ioc_results']),ioc_status=dict(collections.Counter(x['reason'] for x in rec['ioc_results'])),
        peak_inventory=float(np.max(abs(g[:,4]))),mean_abs_inventory=float(np.mean(abs(g[:,4]))),both_grid_fraction=float(np.mean((g[:,6]>0)&(g[:,7]>0))),
        passive_generations=len(B.E.o)-len(B.E.iocs),passive_pulls=sum(c[7]=='PULL' for c in B.cmd_log))
    rec['metrics']=metric;return clean(rec)
def files():
    old=json.loads((R/'predictfun/reviews/opening_entry_pilot_20261124/manifest.json').read_text())
    paths={R/p for p in old['sources']}
    paths|={H/'pilot.py',H/'contract.md',H/'selftest.py',H/'check.py',R/'predictfun/reviews/opening_entry_pilot_20261124/opening.py'}
    for _,m,_ in X.CASES:paths.add(O.OUT/f'{m}_OPENING_ONLY.json.gz')
    return sorted(paths)
def prepare():
    assert not (H/'manifest.json').exists()
    fees={}
    for day,m,_ in X.CASES:
        p=X.DATA/'step1'/day/'registry.parquet';reg=pd.read_parquet(p)
        key='m' if 'm' in reg else 'mid';row=reg[reg[key]==m];assert len(row)==1 and 'fee_bps' in row
        bps=float(row.fee_bps.iloc[0]);assert math.isfinite(bps) and 0<=bps<=10000;fees[str(m)]=bps
    dump(H/'manifest.json',dict(arms=ARMS,cases=X.CASES,fees_bps=fees,sources={str(p.relative_to(R)):sha(p) for p in files()},
        new_candidate_count=3,asof='2026-11-24',untouched_days=False,validation=False))
def run():
    resource.setrlimit(resource.RLIMIT_CPU,(240,241));start=time.process_time()
    man=json.loads((H/'manifest.json').read_text())
    for p,h in man['sources'].items():assert sha(R/p)==h,p
    import subprocess
    for name in ['pilot.py','selftest.py','check.py','contract.md','manifest.json']:
        p=H/name;assert subprocess.check_output(['git','show','HEAD:'+str(p.relative_to(R))],cwd=R)==p.read_bytes()
    OUT.mkdir(exist_ok=True);assert not (OUT/'summary.json').exists();rows=[];parities=[];edges=P.frozen_edges()
    for day,m,part in X.CASES:
        old=X.records(day,part)[m]['CURRENT_ASYNC_FAST10'];path,cache=X.get_path(day,m,part,old)
        for arm,(q,enabled) in ARMS.items():
            p=OUT/f'{m}_{arm}.json.gz';assert not p.exists()
            rec=run_one(path,cache,old,edges,q,man['fees_bps'][str(m)],enabled)
            if arm=='PASSIVE_100':
                reference=json.loads(gzip.decompress((O.OUT/f'{m}_OPENING_ONLY.json.gz').read_bytes()))
                fields=['fills','orders','requests','value','cash','position','commands','guard_log','closed','mode_log']
                parity={k:rec[k]==reference[k] for k in fields};assert all(parity.values()),(m,parity);parities.append(dict(m=m,fields=parity))
            rec.update(arm=arm,manifest_sha256=sha(H/'manifest.json'));p.write_bytes(gzip.compress(json.dumps(rec,allow_nan=False).encode(),mtime=0))
            rows.append(dict(m=m,day=day,arm=arm,sha256=sha(p),metrics=rec['metrics']))
            dump(OUT/'progress.json',dict(rows=rows,parities=parities,cpu_s=time.process_time()-start));print(m,arm,round(time.process_time()-start,2),flush=True)
    result=dict(rows=rows,parities=parities,cpu_s=time.process_time()-start,status='COMPLETE');dump(OUT/'summary.json',result);dump(H/'summary.json',result)
if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
