"""Memory bounded, causal response training and query-time distributions."""
import gc
import json
import time
import lightgbm as lgb
import pf_response_bridge_v2 as B
F,C,np,pd,I,FM=B.F,B.C,B.np,B.pd,B.I,B.FM


def read(day):
    C.require(day in (*F.TRAIN,F.CAL,F.TEST),'undeclared date')
    src=C.OUT/day;r=json.loads((src/'receipt.json').read_text())
    C.require(C.sha(src/'actions.parquet')==r['output_sha256'],'training dataset changed')
    C.require(r['source_manifest']==C.sha(C.REVIEW/'freeze_v1.json'),'data source contract')
    columns=list(dict.fromkeys(['day','m','side','rung','generation','decision_ms','phase','status','x_depth','x_price','fp_scale']+C.MARKET+[h[0] for h in B.HEADS.values()]))
    a=pd.read_parquet(src/'actions.parquet',columns=columns)
    a=a[a.status.isin(['eligible','eligible_estimated_timing'])].copy()
    C.require(not a.duplicated(['day','m','side','rung','generation']).any(),'duplicate training action')
    a['response_depth']=a.x_depth
    a['response_edge']=(1-2*a.side)*(a.pf_mid-a.x_price)
    a['response_scale']=a.fp_scale
    C.require(np.isfinite(a.response_scale).all() and (a.response_scale>0).all(),'invalid training liquidity scale')
    for c in B.FEATURES:a[c]=a[c].astype(np.float32)
    C.require(not np.isinf(a[B.FEATURES].to_numpy()).any(),'infinite response input')
    return a,r


def population(a,name):
    return a[(a.rung==0)&(a.x_depth==0)].copy() if name=='move' else a


def strata(a):
    depth=np.where(a.response_depth<0,-1,np.minimum(a.response_depth,6)).astype(int)
    return list(zip(a.side.astype(int),a.phase.astype(int),depth))


def empirical_fit(a,y,n):
    prior=(np.bincount(y,minlength=n)+1.);prior/=prior.sum();tab={}
    groups={}
    for i,key in enumerate(strata(a)):groups.setdefault(key,[]).append(i)
    for key,ix in groups.items():tab[','.join(map(str,key))]=((np.bincount(y[ix],minlength=n)+30.*prior)/(len(ix)+30.)).tolist()
    return dict(prior=prior.tolist(),groups=tab)


def empirical(a,obj):
    return np.array([obj['groups'].get(','.join(map(str,key)),obj['prior']) for key in strata(a)])


def scores(y,p,base):
    one=np.eye(p.shape[1])[y]
    return dict(n=len(y),logloss=FM.logloss(y,p),empirical_logloss=FM.logloss(y,base),
        brier=float(np.mean(np.sum((p-one)**2,axis=1))),empirical_brier=float(np.mean(np.sum((base-one)**2,axis=1))),
        counts=np.bincount(y,minlength=p.shape[1]),mean_probability=p.mean(0),empirical_mean=base.mean(0))


def fit():
    B.verify();out=B.OUT/'fit';C.require(not out.exists(),'fit exists');out.mkdir(parents=True)
    start=time.time();frames=[];receipts={}
    for d in F.TRAIN:
        a,r=read(d);frames.append(a);receipts[d]=C.sha(C.OUT/d/'receipt.json')
    train=pd.concat(frames,ignore_index=True);del frames,a;gc.collect();cal,cr=read(F.CAL)
    receipts[F.CAL]=C.sha(C.OUT/F.CAL/'receipt.json');C.guard('response data')
    artifacts={};temperatures={};priors={};diagnostics={};importance={}
    for name,(label,n) in B.HEADS.items():
        a=population(train,name);b=population(cal,name);y=B.category(name,a[label]);yc=B.category(name,b[label])
        X=a[B.FEATURES].to_numpy(np.float32);Xc=b[B.FEATURES].to_numpy(np.float32)
        path=out/(name+'.txt');model=FM.fit(X,y,B.FEATURES,path,'multiclass',n)
        raw=model.predict(Xc,num_threads=1);t,loss=FM.choose_temperature(raw,yc);p=FM.temperature(raw,t)
        prior=empirical_fit(a,y,n);priors[name]=prior
        diagnostics[name]=dict(training_rows=len(a),calibration=scores(yc,p,empirical(b,prior)))
        temperatures[name]=dict(value=t,grid_logloss=loss)
        importance[name]=dict(zip(B.FEATURES,map(float,model.feature_importance(importance_type='gain'))))
        artifacts[name]=dict(path=str(path),sha256=C.sha(path))
        del a,b,X,Xc,model,raw,p;gc.collect();C.guard('response '+name)
        print(json.dumps(dict(stage='fit',head=name,seconds=round(time.time()-start),rss_mib=round(I.K.rss_mib()))),flush=True)
    C.dump(out/'empirical.json',priors)
    C.dump(out/'receipt.json',dict(asof=C.stamp(),source_manifest=C.sha(B.FREEZE),train=F.TRAIN,calibration=F.CAL,
        features=B.FEATURES,artifacts=artifacts,temperatures=temperatures,diagnostics=diagnostics,importance=importance,
        data_receipts=receipts,empirical_sha256=C.sha(out/'empirical.json'),seconds=time.time()-start,
        scale_coverage=dict(training_rows=len(train),training_min=float(train.response_scale.min()),training_max=float(train.response_scale.max()),
            calibration_rows=len(cal),calibration_outside=int(((cal.response_scale<train.response_scale.min())|(cal.response_scale>train.response_scale.max())).sum()))))


def load():
    r=json.loads((B.OUT/'fit/receipt.json').read_text());C.require(r['source_manifest']==C.sha(B.FREEZE),'wrong model source')
    C.require(r['train']==list(F.TRAIN) and r['calibration']==F.CAL and r['features']==B.FEATURES,'wrong response model contract')
    for a in r['artifacts'].values():C.require(C.sha(a['path'])==a['sha256'],'response model changed')
    path=B.OUT/'fit/empirical.json';C.require(C.sha(path)==r['empirical_sha256'],'empirical response changed')
    return r,json.loads(path.read_text())


def forecasts(W,X):
    r,priors=load();rows=[];keys=[]
    for s in (0,1):
        for k,t in enumerate(W[0]['tg']):
            ix=X[s][0][k];j=int(W[0]['jk'][k])
            if ix<0 or j<0:continue
            bid=float(W[0]['P5'][j,0]);ask=float(W[1]['P5'][j,0])
            if not 0<=bid<ask<=100:continue
            if not np.isfinite(X[s][2][ix]) or X[s][2][ix]<=0:continue  # explicit no_response_row at policy query
            x=X[s][1][ix];mid=X[s][3][ix]
            C.require(np.isclose(mid,(bid+ask)/2.),'response book/feature alignment')
            phase=min(5,int((t-W[0]['ws'])//60000)+1)
            for d in B.DEPTHS:
                price=(bid if s==0 else ask)-(1-2*s)*d
                if not 0<=price<=100:continue
                rows.append(B.input_row(x,d,(1-2*s)*(mid-price),X[s][2][ix]))
                keys.append((s,k,d,phase))
    matrix=np.asarray(rows,np.float32);keys=np.asarray(keys,int);learned={};baseline={}
    for name,(_,n) in B.HEADS.items():
        mask=np.ones(len(keys),bool) if name=='q1000' else keys[:,2]==0
        local=keys[mask];inp=matrix[mask];model=lgb.Booster(model_file=r['artifacts'][name]['path'])
        p=FM.temperature(model.predict(inp,num_threads=1),r['temperatures'][name]['value'])
        check=np.linspace(0,len(inp)-1,min(3,len(inp))).astype(int)
        for i in check:C.require(np.allclose(p[i],FM.temperature(model.predict(inp[i:i+1],num_threads=1),r['temperatures'][name]['value'])[0],atol=1e-10),'response batch identity')
        a=pd.DataFrame(dict(side=local[:,0],phase=local[:,3],response_depth=local[:,2]));emp=empirical(a,priors[name])
        for d in (B.DEPTHS if name=='q1000' else (0,)):
            v=np.full((2,W[0]['n'],n),np.nan);e=v.copy();take=local[:,2]==d
            v[local[take,0],local[take,1]]=p[take];e[local[take,0],local[take,1]]=emp[take]
            # Outside-payoff query levels have structurally zero reachable flow.
            if name=='q1000' and d>0:
                for s in (0,1):
                    for k in np.flatnonzero(np.isfinite(learned['q1000',0][s,:,0])):
                        j=int(W[0]['jk'][k]);price=float(W[s]['P5'][j,0])-(1-2*s)*d
                        if not 0<=price<=100:v[s,k]=np.eye(n)[0];e[s,k]=np.eye(n)[0]
            learned[name,d]=v;baseline[name,d]=e
        del model;C.guard('response forecasts')
    return learned,baseline


def report():
    B.verify();r,priors=load();a,raw=read(F.TEST);result={}
    for name,(label,n) in B.HEADS.items():
        b=population(a,name);y=B.category(name,b[label]);model=lgb.Booster(model_file=r['artifacts'][name]['path'])
        p=FM.temperature(model.predict(b[B.FEATURES].to_numpy(np.float32),num_threads=1),r['temperatures'][name]['value'])
        base=empirical(b,priors[name]);result[name]=scores(y,p,base)
        result[name]['by_side']={str(s):scores(y[b.side.to_numpy()==s],p[b.side.to_numpy()==s],base[b.side.to_numpy()==s]) for s in (0,1)}
        del model;C.guard('response prediction report')
    C.dump(B.OUT/'prediction_report.json',dict(asof=C.stamp(),day=F.TEST,metrics=result,source_statuses=raw['statuses'],
        scale_coverage=dict(n=len(a),minimum=float(a.response_scale.min()),maximum=float(a.response_scale.max()),
            outside_training_range=int(((a.response_scale<r['scale_coverage']['training_min'])|(a.response_scale>r['scale_coverage']['training_max'])).sum())),
        fit_receipt_sha256=C.sha(B.OUT/'fit/receipt.json'),data_receipt_sha256=C.sha(C.OUT/F.TEST/'receipt.json'),limits=B.verify()['limits']))
