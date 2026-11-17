"""Causal state-conditioned responses with exactly preserved coarse event mass."""
from collections import OrderedDict
import json
from pathlib import Path
import subprocess
import pf_value_bridge as V
F,C,np,pd,I,P,K,PATH=V.F,V.C,V.np,V.pd,V.I,V.P,V.K,V.PATH
OUT=Path(I.K.IO)/'conditional_map_20261117_v1'
REVIEW=F.HERE.parent/'reviews/conditional_map_20261117'
FREEZE=REVIEW/'freeze_v1.json'
CASES=('SOFT_JOINT','EXPANDED_JOINT','STATE_CONDITIONAL')
DYNAMICS=('own_mv250','opp_mv250','adv_ofi250','adv_ofi1000','adv_bn_r100','adv_bn_r1000')
FEATURES=[f'{s}_cum_{d}' for s in ('bid','ask') for d in (0,1,3,5)]+['liquidity_scale','spread','mid']+list(DYNAMICS)
MAX_CELL=8
SOURCES=('pf_conditional_map.py','pf_conditional_map_bank.py','pf_conditional_map_eval.py','pf_conditional_map_checks.py','run_conditional_map.sh')


def geometry(prices,sizes,scale,dynamics):
    C.require(np.isfinite(scale) and scale>0,'invalid causal liquidity scale')
    bid,ask=float(prices[0][0]),float(prices[1][0]);C.require(0<=bid<ask<=100,'invalid geometry book')
    out=[]
    for side in (0,1):
        p=np.asarray(prices[side],float);q=np.asarray(sizes[side],float)
        C.require(p.shape==q.shape==(5,) and np.isfinite(q).all() and (q>=0).all(),'invalid displayed sizes')
        distance=(1-2*side)*(p[0]-p)
        for d in (0,1,3,5):out.append(np.log1p(q[(p>=0)&(distance>=0)&(distance<=d)].sum()/30.))
    return np.asarray(out+[np.log1p(scale/30.),np.log1p(ask-bid),(bid+ask)/200.]+list(dynamics),float)


def normalize(x,normalizer):
    x=np.asarray(x,float);median=np.asarray(normalizer['median']);width=np.asarray(normalizer['iqr'])
    C.require(x.shape[-1]==len(FEATURES) and np.isfinite(width).all() and (width>0).all(),'geometry schema')
    missing=~np.isfinite(x);z=(np.where(missing,median,x)-median)/width
    return np.concatenate([z,missing.astype(float)],axis=-1)


def distance(z,query):
    n=len(FEATURES);d=np.abs(z[...,:n]-query[...,:n]);h=np.where(d<=3.,.5*d*d,3.*d-4.5)
    return np.mean(h+(z[...,n:]-query[...,n:])**2,axis=-1)


def pairs(paths):
    return np.array([15*p['classes'][0]+p['classes'][1] for p in paths],int)


def anchored(log_prior,groups,mass):
    """Only refine within an event pair. Never relax its forecast probability."""
    C.require(np.isfinite(log_prior).all() and mass.shape==(225,) and (mass>=0).all() and np.isclose(mass.sum(),1.),'invalid conditional mass')
    w=np.zeros(len(groups));support=np.bincount(groups,minlength=225)>0
    C.require(not (mass[~support]>0).any(),'anchor event unsupported')
    for g in np.flatnonzero(mass>0):
        ix=np.flatnonzero(groups==g);v=log_prior[ix];v=np.exp(v-v.max());w[ix]=mass[g]*v/v.sum()
    achieved=np.bincount(groups,weights=w,minlength=225)
    C.require(np.allclose(achieved,mass,rtol=0,atol=1e-12),'coarse pair probability erased')
    return w,float(np.abs(achieved-mass).sum())


class Controller(V.Controller):
    def __init__(self,W,legacy,bank,prob,X,case='STATE_CONDITIONAL'):
        C.require(case in CASES,'unknown map');super().__init__(W,legacy,prob,'SOFT_JOINT')
        self.case=case;self.bank=bank;self.X=X;self.expanded=OrderedDict()

    def context(self,k,t,known_pair=None):
        # known_pair is used ONLY by the separately labelled retrospective test.
        ctx,status=super().context(k,t)
        if ctx is None:return ctx,status
        j=int(self.W[0]['jk'][k]);bid=float(self.W[0]['P5'][j,0]);spread=float(self.W[1]['P5'][j,0]-bid)
        phase=min(5,int((t-self.W[0]['ws'])//60000)+1);key=(phase,bid,spread)
        old=self.cache[key][0];old_groups=pairs(old)
        mass=np.bincount(old_groups,weights=ctx[1],minlength=225)
        if known_pair is not None:
            if not (old_groups==known_pair).any():return None,'known_coarse_pair_unsupported'
            mass=np.eye(225)[known_pair]
        if self.case=='SOFT_JOINT':
            if known_pair is None:return ctx,status
            w,err=anchored(np.log([p['base_weight'] for p in old]),old_groups,mass)
            return (ctx[0],w,ctx[2],dict(ctx[3],unsupported_mass=0.,marginal_l1=0.,effective_paths=float(1/(w@w)),anchor_pair_l1=err)),'eligible'
        if key not in self.expanded:
            pp=V.pool(self.bank['paths'],*key)
            C.require(pp,'expanded bank lost baseline support')
            self.expanded[key]=(pp,K.pack(pp,bid,1000.),pairs(pp),np.stack([p['state_z'] for p in pp]))
            if len(self.expanded)>16:self.expanded.popitem(last=False)
        self.expanded.move_to_end(key);pp,packed,groups,z=self.expanded[key]
        log_prior=np.log([p['base_weight'] for p in pp]);mean_distance=0.
        if self.case=='STATE_CONDITIONAL':
            ix=self.X[0][0][k]
            if ix<0:return None,'missing_state_features'
            scale=self.X[0][2][ix]
            if not np.isfinite(scale) or scale<=0:return None,'missing_liquidity_scale'
            x=self.X[0][1][ix];dyn=[x[C.MARKET.index(n)] for n in DYNAMICS]
            query=geometry([self.W[s]['P5'][j] for s in (0,1)],[self.W[s]['S5'][j] for s in (0,1)],scale,dyn)
            d=distance(z,normalize(query,self.bank['normalizer']));log_prior-=.5*d
        w,err=anchored(log_prior,groups,mass)
        if self.case=='STATE_CONDITIONAL':mean_distance=float(w@d)
        K.current_book(packed,[self.W[s]['P5'][j] for s in (0,1)],[self.W[s]['S5'][j] for s in (0,1)])
        diag=dict(ctx[3],paths=len(pp),effective_paths=float(1/(w@w)),anchor_pair_l1=err,state_distance=mean_distance)
        if known_pair is not None:diag.update(unsupported_mass=0.,marginal_l1=0.)
        return (packed,w,ctx[2],diag),'eligible'


def freeze():
    V.verify();C.require(not FREEZE.exists(),'map freeze exists')
    C.dump(FREEZE,dict(asof=C.stamp(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=F.ROOT,text=True).strip(),
        sources={str((F.HERE/n).relative_to(F.ROOT)):C.sha(F.HERE/n) for n in SOURCES},inherited_bridge_sha256=C.sha(V.FREEZE),
        train=F.TRAIN,calibration=F.CAL,test=F.TEST,features=FEATURES,
        bank='Reconstruct the SAME 19198 hash-selected neutral bid/rung0/at-best training exposures. Retain every legacy representative, add farthest causal-state representatives up to8 per phase/spread/native joint-pair cell. Assign each training path to nearest representatives, dividing ties equally; weights sum to complete cell population. Raw paired futures unchanged.',
        state='Causal current known depth cumulated at0/1/3/5 ticks on both sides, log1p(shares/30); W30 absolute scale log1p(scale/30), log1p(spread), midpoint/100; six inherited causal book/flow/Binance dynamics. Median/IQR from training states only, missingness indicators; Huber distance transition3 IQR, mean across17 coordinates. Kernel exp(-0.5*distance), one fixed width.',
        anchor='Compute frozen SOFT_JOINT weights on original bank. Preserve its ENTIRE bid/ask coarse class-pair mass exactly, not only marginals. Expanded responses reweight only inside each pair, with inherited soft-spread prior. All legacy paths retained, so anchor support must not disappear.',
        cases=CASES,unchanged='No forecast retraining, fine-marginal fusion, kernel volume/price editing, controller changes, thresholds or cooldown. Same size30/nr6/cap180/risk1/360,1s endpoint,58/37/100/50ms clocks.',
        tests='Exact archived Oct26/27 neutral actions and labels; baseline reproduction, same current client states, no-fill retained. On Oct27, additionally supply actual coarse pair ONLY in labelled retrospective branch to isolate conditional response. Gate before continuous policy replay.',
        null=dict(draws=200,match='market/side/kind count of positive predicted actions',metric='actual incremental value'),
        replay_gate='Run the fixed12-market Oct27 full5min candidate only if STATE_CONDITIONAL true-pair RMSE is no worse than SOFT_JOINT for BOTH posting and cancellation on identical support, and strictly better for at least one. Gate is a development compute screen, never validation/promotion. If failed, record skipped_by_frozen_gate.',
        multiplicity=dict(new_response_mapping=1,expanded_unconditional_ablation=1,widths=1,representative_counts=1,new_predictors=0,new_controllers=0),
        limits=['Consumed days; no validation, intervals, promotion or live profitability claim','Raw donor futures remain approximate, despite causal state conditioning','Same estimated trade clocks, client queue uncertainty and zero continuation beyond1s','Keeping event probabilities coherent does not guarantee accurate conditional responses or solve churn'],output=str(OUT)))


def verify():
    V.verify();f=json.loads(FREEZE.read_text());C.require(f['inherited_bridge_sha256']==C.sha(V.FREEZE),'baseline changed')
    for p,h in f['sources'].items():C.require(C.sha(F.ROOT/p)==h,'map source changed '+p)
    return f
