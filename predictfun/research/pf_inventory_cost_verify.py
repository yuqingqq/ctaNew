"""Independent saved-artifact checks; no policy tuning or new market inputs."""
import copy
import datetime as dt
import json
import math
import subprocess
import numpy as np
import pandas as pd
from .pf_inventory_cost_study import DEST,HERE,DAYS,ARMS,sha,write
from .pf_inventory_cost_run import folder,stem


def close(a,b,label):
    if not math.isfinite(float(a)) or not math.isfinite(float(b)) or not math.isclose(float(a),float(b),abs_tol=1e-7,rel_tol=1e-10):
        raise ValueError('artifact mismatch: '+label)


def path_metrics(t,s,q):
    t=np.asarray(t,float);s=np.asarray(s);q=np.asarray(q,float)
    if not np.isfinite(t).all() or not np.isfinite(q).all() or (q<=0).any() or not np.isin(s,[0,1]).all():
        raise ValueError('invalid fill population')
    if (np.diff(t)<0).any() or (t<0).any() or (t>300).any():raise ValueError('invalid fill time')
    net=np.cumsum(q*np.where(s==0,1.,-1.));intervals=np.diff(np.r_[0.,t,300.]);states=np.r_[0.,net]
    return dict(net_end=float(states[-1]),inventory_square=float(np.dot(intervals,states**2)/300),
        mean_abs_inventory=float(np.dot(intervals,abs(states))/300),peak_inventory=float(np.max(abs(states))),
        seconds_above10=float(intervals[abs(states)>10+1e-7].sum()))


def check_comparison(comparison,idx):
    a=idx[comparison['timing'],comparison['candidate']];b=idx[comparison['timing'],comparison['reference']]
    for k,value in comparison['delta'].items():close(value,a[k]-b[k],k)
    objective=a['plus1']>=b['plus1'] and a['mark_stop']>=b['mark_stop'] and a['mean_abs_net_end']<b['mean_abs_net_end']
    if comparison['value_and_inventory_objective']!=objective:raise ValueError('false objective predicate')


def controls():
    r=path_metrics([10,20],[0,1],[5,5])
    close(r['net_end'],0,'positive net');close(r['inventory_square'],25/30,'positive integral')
    bad=0
    for args in [([20,10],[0,1],[5,5]),([10],[2],[5]),([10],[0],[float('nan')])]:
        try:path_metrics(*args)
        except ValueError:bad+=1
    if bad!=3:raise ValueError('bad fill control did not fire')
    idx={('J','a'):dict(plus1=2.,mark_stop=2.,mean_abs_net_end=1.),('J','b'):dict(plus1=1.,mark_stop=1.,mean_abs_net_end=2.)}
    c=dict(timing='J',candidate='a',reference='b',delta=dict(plus1=1.),value_and_inventory_objective=True)
    check_comparison(c,idx)
    for key in ('value','predicate'):
        corrupt=copy.deepcopy(c)
        if key=='value':corrupt['delta']['plus1']=0.
        else:corrupt['value_and_inventory_objective']=False
        try:check_comparison(corrupt,idx)
        except ValueError:bad+=1
    if bad!=5:raise ValueError('bad table control did not fire')
    return dict(positive_payoff_and_inventory=True,known_bad_refusals=bad)


def main():
    checks=controls();path=DEST/'report.json';r=json.loads(path.read_text());dec=r['declaration']
    if dec['days']!=DAYS or dec['arms']!=ARMS:raise ValueError('wrong declared population')
    for name,h in dec['source_sha'].items():
        if sha(HERE.parents[1]/name)!=h:raise ValueError('declared source mismatch')
    for name,h in r['artifacts'].items():
        if sha(name)!=h:raise ValueError('checks artifact mismatch')
    idx={(x['timing'],x['variant']):x for x in r['results']}
    if len(idx)!=10:raise ValueError('incomplete comparisons')
    verified=0
    for (tm,label),total in idx.items():
        if [d['day'] for d in total['per_day']]!=DAYS:raise ValueError('day order/population')
        for d in total['per_day']:
            out=folder(d['day'],tm);spec=ARMS[label]
            saved=json.loads((out/'checks.json').read_text())
            for kind in ('win','fills','requests'):
                p=out/kind/stem(spec)
                if sha(p)!=saved['artifacts'][str(p.relative_to(out))]:raise ValueError('changed result file')
            w=pd.read_parquet(out/'win'/stem(spec));f=pd.read_parquet(out/'fills'/stem(spec))
            if len(w)!=288 or w.mid.duplicated().any():raise ValueError('bad market count')
            close(d['settle'],w.pnl.sum(),'settle');close(d['plus1'],f.mo1.sum(),'plus1');close(d['shares'],f.q.sum(),'shares')
            groups={m:g for m,g in f.groupby('mid',sort=False)};ps=[]
            for row in w.itertuples():
                g=groups.get(row.mid,f.iloc[:0]);p=path_metrics(g.t,g.side,g.q);ps.append(p)
                close(p['net_end'],row.net_end,'terminal inventory')
            for metric in ('inventory_square','mean_abs_inventory','seconds_above10'):
                close(d[metric],np.mean([p[metric] for p in ps]),metric)
            close(d['mean_abs_net_end'],np.mean([abs(p['net_end']) for p in ps]),'absolute terminal inventory')
            verified+=1
        for k in ('settle','plus1','shares','concession','committed_concession','spread','adverse_1s','mark_stop','carry'):
            close(total[k],sum(d[k] for d in total['per_day']),k+' total')
        for k in ('inventory_square','mean_abs_net_end','mean_abs_inventory'):
            close(total[k],np.mean([d[k] for d in total['per_day']]),k+' mean')
    for c in r['comparisons']:check_comparison(c,idx)
    if len(r['inference'])!=42 or len(r['stress'])!=5:raise ValueError('inference/stress coverage')
    for x in r['inference'].values():
        if not 0<=x['p_two_sided']<=1 or not 0<=x['p_max_t_family']<=1:raise ValueError('invalid p value')
    dest=DEST/'verification.json'
    if dest.exists():raise ValueError('refuse verification overwrite')
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        verifier_sha=sha(__file__),report_sha=sha(path),controls=checks,arm_day_artifacts_verified=verified,
        independent_inventory_integrals=True,comparison_arithmetic_and_predicates=True,
        limitation='artifact verification does not validate historical fill assumptions or independent strategy performance')
    write(dest,out);write(HERE.parent/'reviews/inventory_cost_20261110/verification_v2.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':main()
