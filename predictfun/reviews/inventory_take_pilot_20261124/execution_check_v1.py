"""Post-build independent execution reconstruction from frozen tapes.
No policy or reading change. Mirrors the declared assumptions, using the raw
frame/trade matrices rather than trusting the run's availability fields.
"""
import collections,copy,gzip,json,resource
import numpy as np
import pilot as T
resource.setrlimit(resource.RLIMIT_CPU,(60,61))
def verify(rec,frames,trades):
    orders=rec['orders'];results=sorted(rec['ioc_results'],key=lambda r:(r['act'],r['gid']));last=-1;prev={};budget={};n=0
    for z in results:
        t=z['act'];s=z['side'];opp=1-s;idx=np.where(frames[:,0]<=t)[0];j=int(idx[-1]) if len(idx) else -1
        reason='empty';q=0.;price=None;available=None;debit=None
        if j<0:reason='no_venue_book'
        else:
            f=frames[j];p=float(f[2+10*opp])
            rem=lambda o:o['size']-sum(v[3] for v in rec['fills'] if v[1]==o['gid'] and v[0]<=t)
            if not (0<=f[2]<f[12]<=100) or t-f[0]>1000:reason='invalid_or_stale'
            elif not 1<=p<=99:reason='price_boundary'
            elif (p>z['limit'] if s==0 else p<z['limit']):reason='limit_moved'
            elif any(str(o['gid']) not in rec['iocs'] and o['side']==opp and o['act']<=t<o['remove'] and rem(o)>1e-9 and
                     (o['price']<=z['limit'] if s==0 else o['price']>=z['limit']) for o in orders):reason='own_cross'
            else:
                for k in range(last+1,j+1):
                    cur={}
                    for side in [0,1]:
                        for col in range(5):
                            px=frames[k,2+10*side+col]
                            if 0<=px<=100:cur[(side,px)]=frames[k,7+10*side+col]
                    for key in set(prev)|set(cur):budget[key]=max(0.,min(cur.get(key,0.),budget.get(key,0.)+cur.get(key,0.)-prev.get(key,0.)))
                    prev=cur
                last=j
                debit=sum(v for tt,ss,pp,v in trades if frames[j,0]<tt<=t and ss==opp and (pp<=p if opp==0 else pp>=p))
                available=max(0.,budget.get((opp,p),0.)-debit);q=min(z['q'],available);price=p
                if q>1e-9:
                    budget[(opp,p)]-=q;reason='filled' if q>=z['q']-1e-9 else 'partial'
        assert z['reason']==reason and abs(z['filled']-q)<1e-7,(z['gid'],reason,z['reason'],q,z['filled'])
        if available is not None:
            assert abs(z['available_before']-available)<1e-7 and abs(z['public_debit']-debit)<1e-7 and z['price']==price
        n+=1
    return n
man=json.loads((T.H/'manifest.json').read_text());total=0;corrupt=0
for day,m,part in T.X.CASES:
    old=T.X.records(day,part)[m]['CURRENT_ASYNC_FAST10'];path,_=T.X.get_path(day,m,part,old)
    for arm in ['TAKE_100','TAKE_220']:
        rec=json.loads(gzip.decompress((T.OUT/f'{m}_{arm}.json.gz').read_bytes()))
        total+=verify(rec,path['frames'],path['trades'])
        if any(x['filled']>0 for x in rec['ioc_results']):
            bad=copy.deepcopy(rec);next(x for x in bad['ioc_results'] if x['filled']>0)['available_before']+=1
            try:verify(bad,path['frames'],path['trades'])
            except AssertionError:corrupt+=1
            else:raise AssertionError('availability corruption escaped')
T.dump(T.H/'execution_checks.json',dict(pass_all=True,raw_tape_markets=3,ioc_requests_reconstructed=total,corruptions_refused=corrupt));print(total,'requests independently reconciled;',corrupt,'corruptions refused')
