"""Expand representative coverage using training-state geometry, never test P&L."""
from collections import Counter,defaultdict
import gc
import json
import pickle
import time
from pathlib import Path
import pf_conditional_map as B
F,C,np,pd,I=B.F,B.C,B.np,B.pd,B.I


def same_path(a,b):
    for k in ('day','m','t','uid','phase','bid','spread','classes','marks','min_price','max_price'):
        C.require(a[k]==b[k],'legacy path identity '+k)
    for k in ('frames','trades'):C.require(np.array_equal(a[k],b[k]),'legacy raw path changed '+k)


def representatives(z,seed,limit):
    chosen=list(seed);C.require(chosen and len(set(chosen))==len(chosen),'invalid representative seeds')
    nearest=np.min(np.stack([B.distance(z,z[i]) for i in chosen]),axis=0)
    while len(chosen)<min(limit,len(z)):
        nearest[chosen]=-1.;i=int(np.argmax(nearest));chosen.append(i)
        nearest=np.minimum(nearest,B.distance(z,z[i]))
    distance=np.stack([B.distance(z,z[i]) for i in chosen],axis=1)
    ties=np.abs(distance-distance.min(1,keepdims=True))<=1e-12
    count=(ties/ties.sum(1,keepdims=True)).sum(0)
    C.require((count>0).all() and np.isclose(count.sum(),len(z),rtol=0,atol=1e-8),'representative population lost')
    return chosen,count


def build():
    B.verify();start=time.time();out=B.OUT/'bank';C.require(not out.exists(),'bank exists');out.mkdir(parents=True)
    legacy=B.PATH.load();old={p['uid']:p for p in legacy};paths=[];statuses=Counter();source={};eligible=selected_n=0
    for day in F.TRAIN:
        src=C.OUT/day;r=json.loads((src/'receipt.json').read_text())
        C.require(C.sha(src/'actions.parquet')==r['output_sha256'],'training actions changed')
        C.require(r['source_manifest']==C.sha(C.REVIEW/'freeze_v1.json'),'training contract changed')
        source[str(src/'receipt.json')]=C.sha(src/'receipt.json')
        for p,h in r['input_hashes'].items():C.require(C.sha(p)==h,'raw training input changed '+p)
        columns=['day','m','side','rung','decision_ms','status','x_depth','fp_scale','pf_mid']+list(B.DYNAMICS)
        a=pd.read_parquet(src/'actions.parquet',columns=columns)
        a=a[a.status.isin(['eligible','eligible_estimated_timing'])&(a.side==0)&(a.rung==0)&(a.x_depth==0)].drop_duplicates(['m','decision_ms']).copy()
        eligible+=len(a);a['uid']=[B.PATH.keyhash(day,m,t) for m,t in a[['m','decision_ms']].itertuples(index=False,name=None)]
        a=a.sort_values('uid').groupby('m',sort=False).head(F.BANK_PER_MARKET);selected_n+=len(a)
        markets=sorted(a.m.astype(int).unique());by_market={int(m):g.to_dict('records') for m,g in a.groupby('m')}
        for begin in range(0,len(markets),24):
            batch=markets[begin:begin+24];data=I.load_day(day,set(batch),r['imputation_offset_ms']);seen=set()
            for W in I.pairs(data):
                m=int(W[0]['m']);seen.add(m)
                for row in by_market[m]:
                    p,status=B.PATH.extract(W,float(row['decision_ms']),day);statuses[status]+=1
                    if p is None:continue
                    C.require(p['uid']==row['uid'] and np.isclose(row['pf_mid'],p['bid']+p['spread']/2.),'training state join mismatch')
                    f=p['frames'][0];p['geometry']=B.geometry([f[2:7],f[12:17]],[f[7:12],f[17:22]],row['fp_scale'],[row[k] for k in B.DYNAMICS])
                    if p['uid'] in old:same_path(p,old[p['uid']])
                    paths.append(p)
            statuses['missing_market_probes']+=sum(len(by_market[m]) for m in set(batch)-seen)
            del data;gc.collect();C.guard('conditional bank batch')
            print(json.dumps(dict(stage='bank',day=day,markets=min(begin+24,len(markets)),seconds=round(time.time()-start),rss_mib=round(I.K.rss_mib()))),flush=True)
    oldr=json.loads((F.OUT/'bank/receipt.json').read_text())
    C.require(eligible==oldr['eligible_candidate_times'] and selected_n==oldr['selected_times'] and dict(statuses)==oldr['statuses'],'training population changed')
    C.require(set(old)<=set(p['uid'] for p in paths),'legacy scenario missing')
    raw=np.stack([p['geometry'] for p in paths]);median=np.nanmedian(raw,axis=0);iqr=np.nanpercentile(raw,75,axis=0)-np.nanpercentile(raw,25,axis=0)
    median=np.where(np.isfinite(median),median,0.);iqr=np.where(np.isfinite(iqr)&(iqr>1e-9),iqr,1.)
    normalizer=dict(median=median,iqr=iqr,features=B.FEATURES,missing_counts=np.isnan(raw).sum(0))
    z=B.normalize(raw,normalizer);cells=defaultdict(list)
    for i,p in enumerate(paths):p['state_z']=z[i];cells[p['phase'],p['spread'],*p['classes']].append(i)
    selected=[];cell_counts={}
    for key,indices in sorted(cells.items()):
        indices=sorted(indices,key=lambda i:paths[i]['uid']);local=z[indices]
        seeds=[j for j,i in enumerate(indices) if paths[i]['uid'] in old]
        chosen,count=representatives(local,seeds,B.MAX_CELL)
        for j,n in zip(chosen,count):p=paths[indices[j]];p['base_weight']=float(n);selected.append(p)
        cell_counts[str(key)]=dict(population=len(indices),representatives=len(chosen),weight_sum=float(count.sum()))
    C.require(set(old)<=set(p['uid'] for p in selected),'retained anchor paths lost')
    obj=dict(paths=selected,normalizer=normalizer);file=out/'bank.pkl';file.write_bytes(pickle.dumps(obj,protocol=5))
    C.dump(out/'receipt.json',dict(asof=C.stamp(),source_manifest=C.sha(B.FREEZE),training_days=F.TRAIN,eligible_times=eligible,selected_times=selected_n,
        statuses=dict(statuses),retained_legacy=len(old),representatives=len(selected),cells=cell_counts,normalizer=normalizer,
        inputs=source,old_bank_receipt_sha256=C.sha(F.OUT/'bank/receipt.json'),output_sha256=C.sha(file),seconds=time.time()-start))


def load():
    out=B.OUT/'bank';r=json.loads((out/'receipt.json').read_text())
    C.require(r['source_manifest']==C.sha(B.FREEZE) and r['output_sha256']==C.sha(out/'bank.pkl'),'conditional bank changed')
    return pickle.loads((out/'bank.pkl').read_bytes())
