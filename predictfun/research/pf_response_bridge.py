"""Richer public-response distributions, projected onto conserved paired paths."""
from collections import OrderedDict
import json
import subprocess
from pathlib import Path
import pf_value_bridge as V
import pf_flow_price_model as FM
F,C,np,pd,I,P,K,PATH=V.F,V.C,V.np,V.pd,V.I,V.P,V.K,V.PATH
OUT=Path(I.K.IO)/'response_bridge_20261116_v1'
REVIEW=F.HERE.parent/'reviews/response_bridge_20261116'
FREEZE=REVIEW/'freeze_v1.json'
FEATURES=C.MARKET+['response_depth','response_edge']
Q_EDGES=np.array([10.,30.,60.,120.,300.])
M_EDGES=np.array([-8.,-4.,-2.,-.5,.5,2.,4.,8.])
HEADS={'q100':('flow_q_100ms',7),'q250':('flow_q_250ms',7),'q1000':('flow_quantity',7),'move':('move_cents',9)}
DEPTHS=(0,1,3,5)
CASES=('SOFT_JOINT','RICH_JOINT','RICH_EMPIRICAL')
SOURCES=('pf_response_bridge.py','pf_response_fit.py','pf_response_eval.py','pf_response_checks.py','run_response_bridge.sh')


def category(head,value):
    a=np.asarray(value,float)
    C.require(np.isfinite(a).all(),'nonfinite response label')
    if head=='move':return np.searchsorted(M_EDGES,a,side='right').astype(int)
    C.require((a>=0).all(),'negative public volume')
    return np.where(a<=1e-10,0,1+np.searchsorted(Q_EDGES,a,side='left')).astype(int)


def signature(path,spread):
    """Future labels of TRAINING donors, aligned to current prices, never inputs."""
    tape=path['trades'];out=[]
    for s in (0,1):
        sg=1-2*s;best=path['bid']+s*spread
        for name,d in specs():
            if name=='joint':out.append(V.labels(path,spread)[s]);continue
            if name=='move':value=sg*(path['marks'][0]-path['bid']-spread/2.)
            else:
                price=best-sg*d;h=int(name[1:])
                reach=tape[:,2]<=price if s==0 else tape[:,2]>=price
                ok=(tape[:,1]==s)&(tape[:,0]>=37.)&(tape[:,0]<=h)&reach
                value=float(tape[ok,3].sum())
            out.append(int(category(name,value)))
    return np.asarray(out,int)


def specs():
    return [('joint',0),('move',0),('q100',0),('q250',0)]+[('q1000',d) for d in DEPTHS]


def project(paths,labels,targets):
    """Generalize the inherited damped IPF; retain all shared raw trajectories."""
    w=np.asarray([p['base_weight'] for p in paths],float);w/=w.sum()
    C.require(len(targets)==2*len(specs()) and labels.shape==(len(paths),len(targets)),'signature alignment')
    supported=[];missing=[]
    for j,p in enumerate(targets):
        p=np.asarray(p,float)
        C.require(np.isfinite(p).all() and (p>=0).all() and np.isclose(p.sum(),1.),'invalid response probability')
        support=np.bincount(labels[:,j],minlength=len(p))>0
        missing.append(float(p[~support].sum()));target=p*support
        C.require(target.sum()>0,'no supported response mass');supported.append(target/target.sum())
    for _ in range(F.IPF_ITERATIONS):
        for j,target in enumerate(supported):
            have=np.bincount(labels[:,j],weights=w,minlength=len(target))
            ratio=np.divide(target,have,out=np.ones(len(target)),where=have>1e-30)
            w*=np.sqrt(np.maximum(ratio[labels[:,j]],1e-30));w/=w.sum()
    residual=[float(np.abs(np.bincount(labels[:,j],weights=w,minlength=len(p))-p).sum()) for j,p in enumerate(targets)]
    return w,dict(unsupported_mass=float(np.mean(missing)),max_unsupported_mass=max(missing),
        marginal_l1=float(np.mean(residual)),effective_paths=float(1/(w@w)),
        coarse_marginal_l1=float(np.mean([residual[0],residual[len(specs())]])),
        rich_marginal_l1=float(np.mean([x for j,x in enumerate(residual) if j%len(specs())!=0])))


class Controller(V.Controller):
    def __init__(self,W,paths,prob,response,case='RICH_JOINT'):
        C.require(case in CASES,'unknown response arm')
        super().__init__(W,paths,prob,'SOFT_JOINT');self.case=case;self.response=response;self.labels=OrderedDict()

    def context(self,k,t):
        ctx,status=super().context(k,t)
        if ctx is None or self.case=='SOFT_JOINT':return ctx,status
        j=int(self.W[0]['jk'][k]);bid=float(self.W[0]['P5'][j,0]);spread=float(self.W[1]['P5'][j,0]-bid)
        phase=min(5,int((t-self.W[0]['ws'])//60000)+1);key=(phase,bid,spread)
        pp=self.cache[key][0]
        if key not in self.labels:
            self.labels[key]=np.stack([signature(p,spread) for p in pp])
            if len(self.labels)>16:self.labels.popitem(last=False)
        self.labels.move_to_end(key)
        targets=[]
        for s in (0,1):
            for name,d in specs():
                p=self.prob[s,k] if name=='joint' else self.response[name,d][s,k]
                if not np.isfinite(p).all():return None,'no_response_row'
                targets.append(p)
        w,diag=project(pp,self.labels[key],targets)
        return (ctx[0],w,ctx[2],dict(ctx[3],**diag)),'eligible'


def freeze():
    V.verify();C.require(not FREEZE.exists(),'response freeze exists')
    C.dump(FREEZE,dict(asof=C.stamp(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=F.ROOT,text=True).strip(),
        sources={str((F.HERE/n).relative_to(F.ROOT)):C.sha(F.HERE/n) for n in SOURCES},inherited_bridge_sha256=C.sha(V.FREEZE),
        train=F.TRAIN,calibration=F.CAL,test=F.TEST,features=FEATURES,parameters=C.V.PARAMS,rounds=200,
        population='All eligible existing neutral action rows, one probe per generation; no-fill retained. Move head uses neutral at-best rung0 rows to avoid duplicating a common endpoint across depth. Data provenance/statuses inherited and hashed.',
        targets=dict(quantity='7 categories: zero,(0,10],(10,30],(30,60],(60,120],(120,300],>300 shares',
            timing='Public at/through quantity in [37,100], [37,250], [37,1000] ms; inherited actual M50 anchors. Labels at focal price; query 1s volume at current depths 0,1,3,5 ticks, early volume at depth0.',
            move='9 side-signed endpoint move categories, edges '+str(M_EDGES.tolist())+' cents, current client midpoint to t+1s'),
        estimator='Four pooled LGBM distributions with fside and causal market features, price-depth and quoted edge only. No own queue/age/fill or future labels in inputs. Fixed inherited params/200 rounds; independent temperature calibration on Oct26.',
        projection='Same soft-spread raw paired bank, current book, 24 damped IPF cycles, now constraints include original coarse joint plus fine response distributions. Order of constraints fixed in specs. Incompatible/unsupported mass and posterior effective paths reported; never independent per-order volume.',
        baseline='SOFT_JOINT unchanged. RICH_EMPIRICAL adds side/phase/depth training-frequency response constraints to the SAME learned coarse joint, removing richer market-feature learning only.',
        calibration_recipe=dict(temperature_grid=C.TEMPERATURES,criterion='categorical logloss',empirical_shrinkage=30),
        evaluation='Exact archived action identities on Oct26/27, cached canonical labels by full identity/hash; rederive causal neutral states. 200 market/side/kind-count matched nulls. Matched12-market Oct27 full5min pilot versus archived SOFT_JOINT, same controller/risk/latency.',
        multiplicity=dict(new_response_candidates=1,response_heads=4,empirical_ablation=1,depth_sets=1,bin_sets=1,parameter_search=0,new_policy_thresholds=0),
        limits=['Consumed development dates, no independent validation or confidence intervals/promotion',
            'Four marginal response distributions plus coarse joint do not fully identify event paths; coherence enforced only on finite donor support',
            'Same sparse bank, approximate donor transport, estimated tape timing, client queue uncertainty and zero 1s continuation',
            'Fine bins still do not identify exact within-bin volume or tail severity; early flow starts37ms rather than58ms post activation',
            'No live profitability or external simulator validation; calibration-day comparisons are in-sample for temperature only'],output=str(OUT)))


def verify():
    V.verify();f=json.loads(FREEZE.read_text())
    C.require(f['inherited_bridge_sha256']==C.sha(V.FREEZE),'inherited bridge changed')
    for p,h in f['sources'].items():C.require(C.sha(F.ROOT/p)==h,'response source changed '+p)
    return f
