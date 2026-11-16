"""Training-only paired empirical paths and their common scenario weights."""
from collections import Counter, defaultdict
import hashlib
import json
import pickle
import time
from pathlib import Path
import pf_flow_control_common as F
import pf_flow_price_model as M
import pf_flow_price_data as D0
C,np,pd,I=F.C,F.np,F.pd,F.I


def keyhash(day,m,t):
    return hashlib.sha256(f'{day}:{m}:{t:.0f}:flow-control-v1'.encode()).hexdigest()


def extract(W,t,day):
    w=W[0];j=int(np.searchsorted(w['kf'],t,side='right')-1)
    if j<0:return None,'no_known_book'
    bid,ask=float(W[0]['P5'][j,0]),float(W[1]['P5'][j,0])
    if not 0<=bid<ask<=100:return None,'invalid_known_book'
    if t+2000.>w['ws']+300000.:return None,'expiry_censored'
    end=int(np.searchsorted(w['tf'],t+2000.,side='right'))
    if end<=j or w['tf'][-1]<t+2000.:return None,'missing_future_book'
    if np.max(np.diff(np.r_[w['tf'][j:end],t+2000.]))>1000.:return None,'stale_future_book'
    frames=np.column_stack([np.asarray(w['tf'])[j:end]-t,np.asarray(w['kf'])[j:end]-t,
        W[0]['P5'][j:end],W[0]['S5'][j:end],W[1]['P5'][j:end],W[1]['S5'][j:end]]).astype(float)
    if not (np.diff(frames[:,0])>=0).all():raise ValueError('path book chronology')
    tape=[]
    for side in (0,1):
        z=W[side];tt=np.asarray(z['anc'])-50.-t
        ix=np.flatnonzero((tt>0)&(tt<=2000.))
        tape.extend((float(tt[i]),side,float(z['ptl'][i]),float(z['shl'][i]),int(i)) for i in ix)
    tape.sort(key=lambda x:(x[0],x[1],x[4]))
    trades=np.asarray([x[:4] for x in tape],float).reshape(-1,4)
    marks=[]
    for h in (1000.,2000.):
        mt,mm=C.AC.valid_marks(w,0);ix=np.searchsorted(mt,t+h,side='right')-1
        if ix<0 or t+h-mt[ix]>1000.:return None,'missing_endpoint'
        marks.append(float(mm[ix]))
    classes=[]
    for side,price in ((0,bid),(1,ask)):
        z=D0.public_target(W[side],t,price,(bid+ask)/2)
        classes.append(int(C.classes(z['flow_quantity'],z['move_cents'])[2]))
    allp=np.r_[frames[:,2:7].ravel(),frames[:,12:17].ravel(),trades[:,2],marks]
    allp=allp[allp>=0]
    path=dict(day=day,m=int(w['m']),t=float(t),uid=keyhash(day,w['m'],t),
        phase=min(5,int((t-w['ws'])//60000)+1),bid=bid,spread=ask-bid,classes=classes,
        frames=frames,trades=trades,marks=marks,min_price=float(allp.min()),max_price=float(allp.max()))
    return path,'eligible'


def build():
    F.verify();out=F.OUT/'bank';C.require(not out.exists(),'bank exists');out.mkdir(parents=True)
    start=time.time();cells=defaultdict(list);counts=Counter();statuses=Counter();sources={};eligible=chosen=0
    for day in F.TRAIN:
        receipt=json.loads((C.OUT/day/'receipt.json').read_text())
        for p,h in receipt['inputs'].items() if 'inputs' in receipt else receipt['input_hashes'].items():
            C.require(C.sha(p)==h,'training source changed '+p);sources[p]=h
        frame=M.read(day)
        mask=(frame.side==0)&(frame.rung==0)&(frame.x_depth==0)
        frame=frame.loc[mask].drop_duplicates(['m','decision_ms']).copy();eligible+=len(frame)
        frame['hash']=[keyhash(day,m,t) for m,t in frame[['m','decision_ms']].itertuples(index=False,name=None)]
        frame=frame.sort_values('hash').groupby('m',sort=False).head(F.BANK_PER_MARKET)
        selected=sorted(map(int,frame.m.unique()));chosen+=len(frame)
        offset=json.loads((Path(C.V.E.OUTD)/f'day_{day}_imp.json').read_text())['untimed']['imp_off_ms']
        for i in range(0,len(selected),24):
            batch=selected[i:i+24];data=I.load_day(day,set(batch),offset);seen=set()
            for W in I.pairs(data):
                m=int(W[0]['m']);seen.add(m)
                for t in frame.loc[frame.m==m,'decision_ms']:
                    p,status=extract(W,float(t),day);statuses[status]+=1
                    if p is None:continue
                    cell=(p['phase'],p['spread'],*p['classes']);counts[cell]+=1
                    cells[cell].append(p);cells[cell].sort(key=lambda z:z['uid'])
                    del cells[cell][F.CELL_TEMPLATES:]
            statuses['missing_market_probes']+=int(frame[frame.m.isin(set(batch)-seen)].shape[0])
            del data;C.guard('bank batch')
            print(json.dumps(dict(stage='bank',day=day,markets=min(i+24,len(selected)),seconds=round(time.time()-start))),flush=True)
    paths=[]
    for cell,items in sorted(cells.items()):
        for p in items:p['base_weight']=counts[cell]/len(items);paths.append(p)
    C.require(paths,'empty bank')
    path=out/'paths.pkl';path.write_bytes(pickle.dumps(paths,protocol=5))
    C.dump(out/'receipt.json',dict(asof=C.stamp(),source_manifest=C.sha(F.FREEZE),
        training_days=F.TRAIN,eligible_candidate_times=eligible,selected_times=chosen,statuses=dict(statuses),
        paths=len(paths),cells=len(cells),path_sha256=C.sha(path),inputs=sources,seconds=time.time()-start))


def load():
    out=F.OUT/'bank';r=json.loads((out/'receipt.json').read_text())
    C.require(r['training_days']==list(F.TRAIN) and r['path_sha256']==C.sha(out/'paths.pkl'),'bank identity')
    return pickle.loads((out/'paths.pkl').read_bytes())


def select(paths,phase,bid,spread):
    selected=[p for p in paths if p['phase']==phase and p['spread']==spread and
              p['min_price']+bid-p['bid']>=0 and p['max_price']+bid-p['bid']<=100]
    return selected


def weights(paths,prob=None):
    base=np.asarray([p['base_weight'] for p in paths],float);base/=base.sum()
    if prob is None:return base,dict(unsupported_mass=0.,marginal_l1=0.,effective_paths=float(1/(base@base)))
    p=np.asarray(prob,float)
    C.require(p.shape==(2,15) and np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1.),'invalid joint forecast')
    labels=np.asarray([z['classes'] for z in paths],int)
    support=np.array([np.bincount(labels[:,s],minlength=15)>0 for s in (0,1)])
    missing=np.sum(p*~support,axis=1);targets=p*support
    C.require((targets.sum(1)>0).all(),'no supported mass')
    targets/=targets.sum(1,keepdims=True);w=base.copy()
    for _ in range(F.IPF_ITERATIONS):
        for s in (0,1):
            have=np.bincount(labels[:,s],weights=w,minlength=15)
            # Damp IPF when two independently forecast marginals are incompatible.
            ratio=np.divide(targets[s],have,out=np.ones(15),where=have>1e-30)
            w*=np.sqrt(np.maximum(ratio[labels[:,s]],1e-30));w/=w.sum()
    achieved=np.array([np.bincount(labels[:,s],weights=w,minlength=15) for s in (0,1)])
    return w,dict(unsupported_mass=float(missing.mean()),max_unsupported_mass=float(missing.max()),
        marginal_l1=float(np.abs(achieved-p).sum(1).mean()),effective_paths=float(1/(w@w)))
