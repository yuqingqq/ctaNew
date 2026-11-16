"""One frozen forecast-to-economic-value mapping experiment; research only."""
from collections import OrderedDict
import json
from pathlib import Path
import subprocess
import pf_flow_control_common as F
import pf_flow_control_paths as PATH
import pf_flow_control_policy as P
import pf_flow_control_kernel as K
C,np,pd,I=F.C,F.np,F.pd,F.I
OUT=Path(I.K.IO)/'value_bridge_20261116_v1'
REVIEW=F.HERE.parent/'reviews/value_bridge_20261116'
FREEZE=REVIEW/'freeze_v1.json'
WIDTH=1.0  # one price tick, design choice frozen before candidate outcomes
CASES=('EXACT_JOINT','SOFT_JOINT','SOFT_EMPIRICAL')
SOURCES=('pf_value_bridge.py','pf_value_bridge_actions.py','pf_value_bridge_run.py','pf_value_bridge_checks.py','run_value_bridge.sh')


def labels(path,spread):
    """Classify translated raw futures against the CURRENT best bid and ask.

    Bid anchoring cancels algebraically; differences in donor/current spread
    change ask reach and the reference midpoint. No future data is a model input.
    These are labels of training scenarios, not labels of the evaluated market.
    """
    t=path['trades'];mid0=path['bid']+spread/2.;mark=path['marks'][0];out=[]
    for side,price in ((0,path['bid']),(1,path['bid']+spread)):
        reach=t[:,2]<=price if side==0 else t[:,2]>=price
        ok=(t[:,0]>=37.)&(t[:,0]<=1000.)&(t[:,1]==side)&reach
        quantity=float(t[ok,3].sum());move=(1-2*side)*(mark-mid0)
        out.append(int(C.classes(quantity,move)[2]))
    return out


def pool(paths,phase,bid,spread):
    C.require(0<=bid<bid+spread<=100 and WIDTH>0,'invalid scenario query')
    ans=[]
    for p in paths:
        if p['phase']!=phase or p['min_price']+bid-p['bid']<0 or p['max_price']+bid-p['bid']>100:continue
        z=dict(p);z['donor_classes']=p['classes'];z['classes']=labels(p,spread)
        z['base_weight']=p['base_weight']*float(np.exp(-abs(p['spread']-spread)/WIDTH))
        ans.append(z)
    return ans


class Controller(P.Controller):
    def __init__(self,W,paths,prob,case='SOFT_JOINT'):
        C.require(case in CASES,'unknown bridge case');super().__init__(W,paths,prob,case)
        self.cache=OrderedDict()

    def context(self,k,t):
        w=self.W[0];j=int(w['jk'][k]);phase=min(5,int((t-w['ws'])//60000)+1)
        if j<0:return None,'no_current_book'
        bid=float(w['P5'][j,0]);ask=float(self.W[1]['P5'][j,0])
        if not 0<=bid<ask<=100:return None,'invalid_current_book'
        if w['ws']+300000.-t<2000.:return None,'horizon_near_expiry'
        key=(phase,bid,ask-bid)
        if key not in self.cache:
            pp=PATH.select(self.paths,phase,bid,ask-bid) if self.case=='EXACT_JOINT' else pool(self.paths,phase,bid,ask-bid)
            self.cache[key]=(pp,K.pack(pp,bid,1000.) if pp else None)
            if len(self.cache)>16:self.cache.popitem(last=False)
        self.cache.move_to_end(key);pp,packed=self.cache[key]
        if not pp:return None,'no_scenario_support'
        K.current_book(packed,[self.W[s]['P5'][j] for s in (0,1)],[self.W[s]['S5'][j] for s in (0,1)])
        probability=None if self.case=='SOFT_EMPIRICAL' else self.prob[:,k,:]
        if probability is not None and not np.isfinite(probability).all():return None,'no_model_row'
        weight,diagnostic=PATH.weights(pp,probability)
        diagnostic['paths']=len(pp)
        diagnostic['spread_distance']=float(sum(w*abs(p['spread']-(ask-bid)) for w,p in zip(weight,pp)))
        diagnostic['relabeled_fraction']=float(sum(w*(p['classes']!=p.get('donor_classes',p['classes'])) for w,p in zip(weight,pp)))
        return (packed,weight,(bid+ask)/2,diagnostic),'eligible'


def freeze():
    F.verify();C.require(not FREEZE.exists(),'bridge freeze exists')
    C.dump(FREEZE,dict(asof=C.stamp(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=F.ROOT,text=True).strip(),
        sources={str((F.HERE/n).relative_to(F.ROOT)):C.sha(F.HERE/n) for n in SOURCES},
        inherited_controller_freeze_sha256=C.sha(F.FREEZE),bank_receipt_sha256=C.sha(F.OUT/'bank/receipt.json'),
        training=F.TRAIN,forecast_model='Same frozen Oct27 classifier, trained Oct23-25, temperature calibrated Oct26; unchanged on both diagnostic dates',
        candidate='Pool all same-phase payoff-valid training paths; prior weight count*exp(-abs(donor spread-current spread)/1 tick). Translate prices by current bid, overwrite initial book, relabel public reach and signed movement at current best bid/ask before common IPF projection.',
        unchanged='Classifier, raw path data, 24 IPF iterations, execution kernel/lane, same greedy policy, risk cap180/lambda1/360, size30 six rungs, 58/37/100/50ms clocks,1s horizon, no retry or new entry/cancel threshold',
        comparison=list(CASES),
        actions='Neutral capped reference, one hash-assigned resting probe per generation plus one opportunity at each reference submission. All statuses retained, no-fill actions retained. Select two per market/side/minute/action kind by identity hash. Exact canonical post/wait or keep/cancel branches with identical historical/unavoidable prefixes, no subsequent posts, common endpoint wealth minus risk.',
        dates={'20261026':'12 first markets in two-hour blocks; earlier value-calibration diagnostic, no parameter selection',
               '20261027':'same12 consumed pilot markets; value diagnostic and SOFT_JOINT full5min replay compared with archived JOINT on identical markets'},
        reuse='Oct27 cancellation labels may be reused only by exact generation/time identity from audited bounded_probe_v1; source receipt/hash pinned, other labels freshly reconciled',
        null=dict(draws=200,seed=20261116,match='market/side/action-kind count of positive predicted actions',metric='actual incremental economic value'),
        multiplicity=dict(new_value_mappings=1,new_predictors=0,bandwidths=1,new_thresholds=0,new_controllers=0,empirical_ablation=1),
        uncertainty='Report weighted scenario-advantage standard deviation and effective path count as dispersion diagnostics, not confidence intervals or a new decision threshold',
        limits=['Consumed development dates; no confidence intervals, promotion or live profitability claim',
            'Donor futures remain approximate; pooling does not prove conditional severity or timing calibrated',
            'Exact phase and payoff bounds remain support cuts; relabelled categories can change at tick thresholds',
            'One-second zero continuation and unreceived-fill uncertainty are unchanged',
            'Calibration-day classifier temperature used that day; no OOS claim on Oct26'],output=str(OUT)))


def verify():
    F.verify();f=json.loads(FREEZE.read_text())
    for p,h in f['sources'].items():C.require(C.sha(F.ROOT/p)==h,'bridge source changed '+p)
    C.require(f['inherited_controller_freeze_sha256']==C.sha(F.FREEZE),'controller freeze changed')
    C.require(f['bank_receipt_sha256']==C.sha(F.OUT/'bank/receipt.json'),'training bank changed')
    return f
