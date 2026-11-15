"""Audit saved timing-certificate labels and source counts, without reading raw data."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import resource

import numpy as np
import pandas as pd

DAYS=('20261023','20261024','20261025','20261026','20261027','20261031','20261101','20261102','20261103')
MODES=('certified_ws','certified_estimated')
LABELS=('value_cents','value_per_known_remaining_share','filled_quantity','past_quantity',
        'common_prefix_quantity','actual_remaining_after_prefix','at_price_value_cents','through_price_value_cents')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(ok,msg):
    if not bool(ok):raise ValueError(msg)


def check(df,proofs,estimated):
    require(not df.duplicated(['day','m','side','rung','generation']).any(),'duplicate action')
    ok=df.status.str.startswith('eligible');a=df[ok]
    require(df.loc[~ok,list(LABELS)].isna().all().all(),'invalid label has value')
    require(np.isfinite(a[list(LABELS)].values).all(),'nonfinite label')
    require((a.x_known_remaining>0).all(),'invalid known quantity')
    require((a.filled_quantity>=0).all() and (a.filled_quantity<=a.actual_remaining_after_prefix+1e-7).all(),'quantity overused')
    require(np.allclose(a.actual_remaining_after_prefix,30.-a.past_quantity-a.common_prefix_quantity,rtol=0,atol=1e-7),'remainder differs')
    require(np.allclose(a.value_cents,a.at_price_value_cents+a.through_price_value_cents,rtol=0,atol=1e-7),'components differ')
    require(np.allclose(a.value_cents,a.value_per_known_remaining_share*a.x_known_remaining,rtol=0,atol=1e-7),'normalization differs')
    for market,g in df.groupby('m',sort=False):
        p=proofs[int(market)]
        require(p['raw_legs']==p['timed_legs']+len(p['missing']),'window source count differs')
        require((g.source_raw_legs==p['raw_legs']).all() and (g.source_missing_legs==len(p['missing'])).all(),'action source count differs')
        for side,h in g.groupby('side',sort=False):
            times=np.asarray([e['earliest'] for e in p['events'] if e['side']==side and e['earliest'] is not None and (not estimated or e['kind']=='missing')])
            # first_reach is bounded by the replay terminal time already.
            n=np.searchsorted(np.sort(times),h.decision_ms.values+1000.,side='right')
            prior=np.searchsorted(np.sort(times),h.decision_ms.values,side='right')
            require(np.array_equal(n,h.unresolved_events.values) and np.array_equal(prior,h.prior_state_events.values),'certificate count differs')
            require((n[h.status.str.startswith('eligible').values]==0).all(),'uncertified eligible action')
    return dict(n=len(df),eligible=int(ok.sum()),statuses={k:int(v) for k,v in df.status.value_counts().items()})


def run(out,receipt):
    report=json.loads((out/'report.json').read_text());summaries={};controls={}
    for day in DAYS:
        d=out/day;r=json.loads((d/'receipt.json').read_text())
        require(sha(d/'receipt.json')==report['days_receipt_sha256'][day],'day receipt changed')
        require(sha(d/'window_proofs.jsonl.gz')==r['proofs_sha256'],'proof file changed')
        with gzip.open(d/'window_proofs.jsonl.gz','rt') as f:proofs={x['market']:x for x in map(json.loads,f)}
        summaries[day]={}
        for mode in MODES:
            p=d/(mode+'.parquet');require(sha(p)==r['artifacts'][p.name],'action file changed')
            df=pd.read_parquet(p);x=check(df,proofs,mode=='certified_estimated');summaries[day][mode]=x
            for s,n in x['statuses'].items():require(r['counts'][mode+':'+s]==n,'receipt status count differs')
            if not controls:
                good=df.index[df.status.str.startswith('eligible')][0];bad=df.index[~df.status.str.startswith('eligible')][0]
                cases={'duplicate':pd.concat([df,df.iloc[:1]],ignore_index=True)}
                for name,col,value,ix in [('quantity','filled_quantity',10000.,good),('certificate','unresolved_events',1,good),('source','source_missing_legs',-1,good),('invalid_label','value_cents',0.,bad)]:
                    b=df.copy();b.loc[ix,col]=value;cases[name]=b
                for name,b in cases.items():
                    try:check(b,proofs,False)
                    except ValueError:controls[name]=True
                    else:controls[name]=False
                require(all(controls.values()),'corruption not refused')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2048*1024,'RSS cap')
    def table(days):
        counts=Counter()
        for d in days:counts.update(json.loads((out/d/'receipt.json').read_text())['counts'])
        result={}
        for mode in MODES:
            eligible=sum(summaries[d][mode]['eligible'] for d in days)
            base=sum(n for k,n in counts.items() if k.startswith('baseline_'+mode+':eligible'))
            result[mode]=dict(probes=sum(summaries[d][mode]['n'] for d in days),baseline_eligible=base,certified_eligible=eligible,gain=eligible-base,multiple=eligible/base)
        result['future_only_eligible_without_certificate']=counts['future_only_eligible_without_certificate']
        return result
    final=dict(asof=datetime.now(timezone.utc).isoformat(),status='saved_labels_and_certificates_pass',
        checker_sha256=sha(__file__),source_report_sha256=sha(out/'report.json'),source_asof=report['asof'],
        all_days=table(DAYS),development=table(DAYS[:4]),days=summaries,controls=controls,
        overlap_parity_pairs=report['overlap_parity_pairs'],interpretation='Certified coverage within the declared replay/timing assumptions, not physical timing truth or model performance.')
    receipt.parent.mkdir(parents=True,exist_ok=True)
    with receipt.open('x') as f:json.dump(final,f,indent=2);f.write('\n')
    print(json.dumps({k:final[k] for k in ('asof','status','all_days','development','controls','overlap_parity_pairs')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();run(a.out,a.receipt)
