"""Descriptive follow-up: observed price direction versus empirical-bank changes.

Declared before readout: signed midpoint change groups on all first-review
pairs; exact bank identity/spread/phase switches on the fixed384-pair sample;
freeze old probabilities, old market, or old target queue individually while
holding all other current inputs fixed. No new selection, model, or policy.
"""
from collections import Counter
import json
from pathlib import Path
import pickle
import subprocess
import pf_flow_churn_diagnostic as D
C,np,pd,F=D.C,D.np,D.pd,D.F


def run():
    F.verify();receipt=json.loads((D.OUT/'receipt.json').read_text())
    for name,h in receipt['outputs'].items():C.require(C.sha(D.OUT/name)==h,'changed diagnostic artifact')
    rows=pd.read_parquet(D.OUT/'pairs.parquet');a=pd.read_parquet(D.OUT/'attribution.parquet').set_index('uid')
    pairs=pickle.loads((D.OUT/'states.pkl').read_bytes());paths=D.PATH.load();extra=[]
    for p in pairs:
        e,r=p['entry'],p['review'];s=r['state'];i=r['index'];target=s['orders'][i];side=int(target[0])
        current=D.context(paths,r['spec']);oldp=D.context(paths,r['spec'],e['spec']['prob']);oldm=D.context(paths,e['spec'],r['spec']['prob'])
        oldids={z['uid'] for z in oldm[3]};newids={z['uid'] for z in current[3]}
        keep=lambda orders,ctx:D.advantage(orders,r['prefix'],s['position'],i,ctx)['value']
        actual=keep(s['orders'],current)
        C.require(np.isclose(actual,a.loc[p['uid'],'current_keep'],atol=1e-7),'actual value not reproduced')
        queue=s['orders'].copy();queue[i,4]=D.queue_shown(e['spec'],side,target[2])[0]
        extra.append(dict(uid=p['uid'],m=p['m'],side=side,cancel=r['cancel'],current_keep=actual,
            old_forecast_keep=keep(s['orders'],oldp),old_market_keep=keep(s['orders'],oldm),old_queue_keep=keep(queue,current),
            bank_changed=oldids!=newids,bank_jaccard=len(oldids&newids)/len(oldids|newids),
            phase_changed=e['spec']['phase']!=r['spec']['phase'],spread_changed=e['spec']['spread']!=r['spec']['spread'],
            bid_change=r['spec']['bid']-e['spec']['bid'],
            signed_best_change=(1-2*side)*(r['spec']['prices'][side,0]-e['spec']['prices'][side,0]),
            signed_mid_change=(1-2*side)*(r['spec']['bid']+r['spec']['spread']/2-e['spec']['bid']-e['spec']['spread']/2),
            market_shapley=a.loc[p['uid'],'shapley_market_paths_book']))
    x=pd.DataFrame(extra);detail={}
    rows['signed_mid_change']=(1-2*rows.side)*rows.mid_delta
    for name,cancel in [('cancel',True),('keep',False)]:
        g=rows[rows.cancel==cancel];z=x[x.cancel==cancel];b=a[a.cancel==cancel]
        detail[name]=dict(n=len(g),adverse_mid=int((g.signed_mid_change<0).sum()),unchanged_mid=int((g.signed_mid_change==0).sum()),
            favorable_mid=int((g.signed_mid_change>0).sum()),mean_signed_mid_cents=float(g.signed_mid_change.mean()),
            entry_advantage_quantiles_cents={str(q):float(g.entry_advantage.quantile(q)) for q in (0,.25,.5,.75,.9,.99)},
            cancel_advantage_quantiles_cents={str(q):float(g.current_cancel_advantage.quantile(q)) for q in (.25,.5,.75,.9)} if cancel else {},
            sample_n=len(z),bank_changed=int(z.bank_changed.sum()),spread_changed=int(z.spread_changed.sum()),phase_changed=int(z.phase_changed.sum()),
            cancel_with_old_forecast=int((z.old_forecast_keep<-1e-9).sum()),cancel_with_old_market=int((z.old_market_keep<-1e-9).sum()),
            cancel_with_old_queue=int((z.old_queue_keep<-1e-9).sum()),
            original_endpoint_still_cancel_and_h2_still_cancel=int(((b.fixed_original_endpoint_keep<-1e-9)&(b.h2_keep<-1e-9)).sum()),
            by_bank_change={str(flag):dict(n=len(h),mean_market_shapley_cents=float(h.market_shapley.mean()),
                mean_signed_mid_cents=float(h.signed_mid_change.mean()),adverse_mid=int((h.signed_mid_change<0).sum())) for flag,h in z.groupby('bank_changed')})
    D.REVIEW.mkdir(exist_ok=True)
    C.require(not (D.OUT/'market_readout_v1.json').exists(),'readout exists')
    x.to_parquet(D.OUT/'market_sensitivity_v1.parquet',index=False)
    result=dict(asof=C.stamp(),groups=detail,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=F.ROOT,text=True).strip(),
        source_sha256=C.sha(Path(__file__)),input_receipt_sha256=C.sha(D.OUT/'receipt.json'),
        sensitivity_sha256=C.sha(D.OUT/'market_sensitivity_v1.parquet'),
        controls=dict(all_actual_values_reproduced=True,corrupt_actual_refused=D.CHECK.refused(lambda:C.require(np.isclose(actual,actual+1),'corrupt actual'),'corrupt actual')),
        limits=['Descriptive same consumed pilot, no new trading policy or profit claim','Old-market intervention combines observed book and empirical future paths; bank-change grouping does not separate their causal contributions'])
    C.dump(D.OUT/'market_readout_v1.json',result);print(json.dumps(C.clean(result)),flush=True)


if __name__=='__main__':run()
