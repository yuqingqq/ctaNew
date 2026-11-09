"""Supplemental audit: reducing-at-send fills that open reverse inventory.

Includes fills from flat after an earlier fill reached zero. Same-time fill
notifications are bounded over all prefix states, rather than assuming a
favorable event-priority order. This is a post-hoc mechanism diagnostic.
"""
import datetime as dt
import json
import numpy as np
import pandas as pd
from .pf_inflight_study import HERE, DAYS, ARMS, DEST, sha, write
from .pf_inflight_run import location, stem


def audit(f, requests, ws):
    if f.empty:
        return dict(opening_shares_lower=0.,opening_shares_upper=0.,tranches_lower=0,tranches_upper=0,tied_tranches=0,examples=[])
    t=f.t.to_numpy(float);q=f.q.to_numpy(float);s=f.side.to_numpy(int)
    if not np.isfinite(t).all() or not np.isfinite(q).all() or (q<=0).any() or (np.diff(t)<0).any() or not np.isin(s,[0,1]).all():
        raise ValueError('invalid fill chronology')
    sg=np.where(s==0,1.,-1.);prefix=np.r_[0.,np.cumsum(sg*q)]
    send=np.rint((ws+t*1000-f.age_ms.to_numpy()-40)*4096).astype(np.int64)
    notifications=np.rint((ws+t*1000+100)*4096).astype(np.int64)
    places=requests[requests.kind=='place']
    keys=set(zip(np.rint(places.t_ms.to_numpy()*4096).astype(np.int64),places.side,places.lane))
    if any((int(ts),int(side),int(lane)) not in keys for ts,side,lane in zip(send,s,f.lane)):
        raise ValueError('fill does not match a sent generation')
    left=np.searchsorted(notifications,send,'left');right=np.searchsorted(notifications,send,'right')
    opening=np.maximum(0.,q-np.where(prefix[:-1]*sg<0,np.minimum(np.abs(prefix[:-1]),q),0.))
    lower=[];upper=[];examples=[]
    for i in range(len(f)):
        states=prefix[left[i]:right[i]+1]*sg[i]
        definite=bool(states.max() < -1e-7);possible=bool(states.min() < -1e-7)
        lower.append(opening[i] if definite else 0.);upper.append(opening[i] if possible else 0.)
        if possible and opening[i]>1e-7 and len(examples)<3:
            examples.append(dict(t=float(t[i]),send_rel=float((send[i]/4096-ws)/1000),side=int(s[i]),
                q=float(q[i]),known_signed_min=float(states.min()),known_signed_max=float(states.max()),
                actual_before=float(prefix[i]),actual_after=float(prefix[i+1]),opening=float(opening[i]),definite=definite))
    return dict(opening_shares_lower=float(sum(lower)),opening_shares_upper=float(sum(upper)),
        tranches_lower=int((np.array(lower)>1e-7).sum()),tranches_upper=int((np.array(upper)>1e-7).sum()),
        tied_tranches=int((left!=right).sum()),examples=examples)


def controls():
    f=pd.DataFrame(dict(t=[.05,.30,.31],q=[10.,10.,5.],side=[0,1,1],lane=[0,1,1],age_ms=[10.,60.,70.]))
    r=pd.DataFrame(dict(t_ms=[0.,200.],side=[0,1],lane=[0,1],kind=['place','place']))
    a=audit(f,r,0.);assert a['opening_shares_lower']==a['opening_shares_upper']==5.
    assert a['tranches_lower']==1 and a['examples'][0]['actual_before']==0.
    tied=pd.DataFrame(dict(t=[.1,.3],q=[10.,14.],side=[0,1],lane=[0,1],age_ms=[60.,60.]))
    a=audit(tied,r,0.);assert a['opening_shares_lower']==0 and a['opening_shares_upper']==4 and a['tied_tranches']==1
    for bad in [r.iloc[:1],r.assign(t_ms=r.t_ms+1)]:
        try:audit(f,bad,0.)
        except ValueError:continue
        raise AssertionError('missing or shifted generation accepted')
    return dict(flat_then_reverse_positive=True,event_priority_bounds_positive=True,missing_or_shifted_event_refused=True)


def main():
    checked=controls();main=json.loads((DEST/'report.json').read_text());rows=[];examples=[]
    for tm in ['J','X']:
        for label,spec in ARMS.items():
            for day in DAYS:
                loc=location(label,day,tm);f,r,w=(pd.read_parquet(loc/k/stem(spec)) for k in ['fills','requests','win'])
                if len(w)!=288 or w.mid.duplicated().any():raise ValueError('market population')
                for k in ['fills','requests','win']:
                    ref=main['artifacts'][f'{tm}/{day}/{label}'][k]
                    if sha(loc/k/stem(spec))!=ref['sha']:raise ValueError('run artifact changed')
                fg={m:g for m,g in f.groupby('mid',sort=False)};rg={m:g for m,g in r.groupby('mid',sort=False)}
                stats=[]
                for market in w.itertuples():
                    a=audit(fg.get(market.mid,f.iloc[:0]),rg.get(market.mid,r.iloc[:0]),market.ws)
                    ex=a.pop('examples');stats.append(a)
                    examples.extend(dict(day=day,timing=tm,variant=label,mid=int(market.mid),**x) for x in ex)
                rows.append(dict(day=day,timing=tm,variant=label,markets=len(w),excluded=0,
                    **{k:sum(a[k] for a in stats) for k in stats[0]},
                    windows_lower=sum(a['tranches_lower']>0 for a in stats),windows_upper=sum(a['tranches_upper']>0 for a in stats)))
    totals=[]
    for tm in ['J','X']:
        for label in ARMS:
            selected=[r for r in rows if r['timing']==tm and r['variant']==label]
            totals.append(dict(timing=tm,variant=label,**{k:sum(r[k] for r in selected) for k in selected[0] if k not in ['day','timing','variant']}))
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),kind='post_hoc_mechanism_audit',
        definition='quantity opening inventory from orders whose direction reduced known inventory at send; includes starting from actual flat',
        ambiguity='bounds over all prefixes of exact same-time fill notifications; generation matched by epoch tick, side and lane',
        threshold_shares=1e-7,main_receipt_sha=sha(DEST/'report.json'),source_sha=sha(__file__),
        controls=checked,totals=totals,per_day=rows,examples=examples,
        interpretation='all-market diagnostic, no new inference or policy selection; direct sign-crossing count in main receipt is narrower')
    write(DEST/'overshoot_diagnostic.json',out);write(HERE.parent/'reviews/inflight_budget_20261109/overshoot_diagnostic.json',out)
    print(json.dumps(totals,indent=2))


if __name__=='__main__':main()
