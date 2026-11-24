"""Independent result ledger checks. No runner execution for check/falsifier."""
import collections,copy,gzip,json,math,resource
from pathlib import Path
import numpy as np
import pilot as T
H=Path(__file__).resolve().parent;resource.setrlimit(resource.RLIMIT_CPU,(40,41))
def verify(r,q,bps):
    fills=r['fills'];ioc={int(k):v for k,v in r['iocs'].items()};results={x['gid']:x for x in r['ioc_results']}
    assert set(results)==set(ioc) and len(results)==len(r['ioc_results'])
    cash=pos=0.;fees=0.;known={};orderfills=collections.defaultdict(float)
    for t,g,s,n,p,it in fills:
        assert 0<=t<=r['end'] and n>0 and s in [0,1] and 0<=p<=100
        cash-=(1-2*s)*n*p;pos+=(1-2*s)*n;orderfills[g]+=n
        if it<0:
            a=ioc[g];assert it==-1 and t==a['submit']+10 and s==a['side']
            assert (p<=a['limit'] if s==0 else p>=a['limit'])
            fees+=bps/10000*min(p,100-p)*n
        else:assert g not in ioc
    assert abs(cash-r['cash'])<1e-5 and abs(pos-r['position'])<1e-7
    assert abs(cash+pos*r['mark']-fees-r['net_value_c'])<1e-5 and abs(fees-r['fees_c'])<1e-7
    assert abs(cash+pos*r['mark']-2*fees-r['fee2_net_value_c'])<1e-5
    lots=collections.deque();matched=0.
    for t,g,s,n,p,it in fills:
        sg=1-2*s
        while n>1e-9 and lots and lots[0][0]!=sg:
            a,b,c=lots[0];v=min(b,n);matched+=a*v*(p-c);b-=v;n-=v
            if b<1e-9:lots.popleft()
            else:lots[0]=(a,b,c)
        if n>1e-9:lots.append((sg,n,p))
    residual=sum(s*n*(r['mark']-p) for s,n,p in lots)
    assert abs(matched-r['metrics']['completed_c'])<1e-5 and abs(residual-r['metrics']['residual_c'])<1e-5
    assert abs(matched+residual-fees-r['net_value_c'])<1e-5
    sent={x['gid']:x for x in r['ioc_decisions'] if x['status']=='sent'};assert set(sent)==set(ioc)
    for gid,a in ioc.items():
        x=sent[gid];z=results[gid];assert a['act']==a['submit']+10 and a['ack']==a['act']+100
        assert abs(x['known_I'])>=q-1e-9 and a['side']==(1 if x['known_I']>0 else 0)
        room=2*q+x['known_I']-x['A'] if a['side']==1 else 2*q-x['known_I']-x['B']
        expected=max(0,math.floor(min(q,abs(x['known_I'])-q/2,room)+1e-9));assert a['q']==expected
        assert x['book_received_ms']<=x['t'] and x['limit']==a['limit']
        assert z['filled']<=a['q']+1e-9 and abs(z['filled']-orderfills[gid])<1e-8
        if z['filled']>0:
            assert z['available_before']>=z['filled']-1e-9 and z['venue_frame_ms']<=a['act']
            before=sum((1-2*f[2])*f[3] for f in fills if f[0]<a['act'] or (f[0]==a['act'] and f[1]!=gid))
            assert abs(before-z['venue_before'])<1e-6
            assert z['absolute_risk_reduced']==(abs(before+(1-2*a['side'])*z['filled'])<abs(before)-1e-9)
    grid=np.array(r['grid']);assert np.max(abs(grid[:,4]))<=2*q+1e-6
    assert np.max(grid[:,1]+grid[:,2])<=2*q+1e-6 and np.max(-grid[:,1]+grid[:,3])<=2*q+1e-6
    assert np.all(grid[:,4]>=grid[:,1]-grid[:,3]-1e-6) and np.all(grid[:,4]<=grid[:,1]+grid[:,2]+1e-6)
    # Reconstruct client-known fills (notice delay) from every ledger fill.
    ts=np.array([f[0] for f in fills]);dq=np.array([(1-2*f[2])*f[3] for f in fills]);cum=np.r_[0,np.cumsum(dq)]
    assert np.all(np.diff(ts)>=0)
    expect=cum[np.searchsorted(ts,grid[:,0]-100,side='right')]
    assert np.allclose(expect,grid[:,1],atol=1e-6,rtol=0)
    return True
def run():
    man=json.loads((H/'manifest.json').read_text());summary=json.loads((T.OUT/'summary.json').read_text());records=[]
    for row in summary['rows']:
        p=T.OUT/f"{row['m']}_{row['arm']}.json.gz";assert T.sha(p)==row['sha256'];r=json.loads(gzip.decompress(p.read_bytes()))
        q,active=T.ARMS[row['arm']];verify(r,q,man['fees_bps'][str(row['m'])]);records.append((r,q,man['fees_bps'][str(row['m'])]))
    assert len(records)==12 and all(all(p['fields'].values()) for p in summary['parities'])
    r,q,bps=next(x for x in records if x[0]['metrics']['takes_filled']>0);gid=next(x['gid'] for x in r['ioc_results'] if x['filled']>0)
    funcs=[lambda x:x.__setitem__('net_value_c',x['net_value_c']+1),lambda x:x.__setitem__('fees_c',0),
        lambda x:x['iocs'][str(gid)].__setitem__('act',x['iocs'][str(gid)]['act']-10),
        lambda x:x['grid'][0].__setitem__(1,999),lambda x:x['ioc_results'].append(x['ioc_results'][0]),
        lambda x:x['ioc_results'][next(i for i,z in enumerate(x['ioc_results']) if z['gid']==gid)].__setitem__('available_before',0)]
    n=0
    for fn in funcs:
        bad=copy.deepcopy(r);fn(bad)
        try:verify(bad,q,bps)
        except AssertionError:n+=1
    assert n==len(funcs)
    T.dump(H/'checks.json',dict(pass_all=True,records=len(records),parity_markets=3,corruptions_refused=n))
    print('PASS',len(records),'ledgers and',n,'corruptions')
if __name__=='__main__':run()
