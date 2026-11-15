"""Post-result attribution/export of the frozen experiment; no fits or selection.

All setups exported. Example cases are explicitly hindsight-selected losses;
they explain errors and are never used to define or validate a policy.
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import sys

import numpy as np
import pandas as pd

src=Path(sys.argv[1]);out=Path(sys.argv[2]);r=json.loads((src/'report.json').read_text())
assert hashlib.sha256(Path(__file__).parents[2].joinpath('research/pf_stop_ev_economic_layer.py').read_bytes()).hexdigest()==r['source_sha256']
shutil.copyfile(src/'report.json',out/'report_v1.json')
shutil.copyfile(src/'states_pooled.json',out/'states_pooled_v1.json')
panels=[]
for day in r['folds']:
    for stem in ['calibrator','states','worst_raw_positive_cases']:
        ext='csv' if stem.endswith('cases') else 'json'
        shutil.copyfile(src/day/(stem+'.'+ext),out/(stem+'_'+day+'.'+ext))
    panels.append(pd.read_parquet(src/day/'actions.parquet'))
d=pd.concat(panels,ignore_index=True);y=d.value_cents.to_numpy();raw=d.raw.to_numpy()>=0;order=d.order.to_numpy()>=0
rows=[]
for name,z in r['pooled']['scores'].items():
    for margin,p in z['policies'].items():
        daily=[q['scores'][name]['policies'][margin] for q in r['folds'].values()]
        row=dict(score=name,margin=float(margin),kept_n=p['kept']['n'],filled_n=p['kept']['filled'],
            good_value_dollars=p['kept']['good_value_cents']/100,gross_loss_dollars=p['kept']['gross_loss_cents']/100,
            net_value_dollars=p['kept']['value_cents']/100,delta_vs_raw_dollars=p['delta_vs_raw_cents']/100,
            matched_raw_value_dollars=p['matched_raw_value_cents']/100,selection_vs_raw_dollars=p['selection_vs_raw_cents']/100,
            selection_vs_order_dollars=p['selection_vs_order_cents']/100,positive_days=sum(q['kept']['value_cents']>0 for q in daily),
            delta_vs_raw_positive_days=sum(q['delta_vs_raw_cents']>0 for q in daily),
            matched_raw_positive_days=sum(q['selection_vs_raw_cents']>0 for q in daily),
            matched_order_positive_days=sum(q['selection_vs_order_cents']>0 for q in daily))
        rows.append(row)
with (out/'all_policies_v1.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
base=r['pooled']['raw_nonnegative'];err=r['pooled']['raw_kept_error']
z=r['pooled']['scores']['order']['policies']['0.0']
group=[]
for key,g in d.groupby(['side','queue','edge'],sort=True):
    a=g[g.raw>=0];b=g[g.order>=0]
    group.append(dict(side=int(key[0]),queue=int(key[1]),edge=int(key[2]),n=len(g),raw_kept_n=len(a),
        raw_kept_value_dollars=float(a.value_cents.sum()/100),raw_predicted_kept_dollars=float(a.raw.sum()/100),
        order_kept_n=len(b),order_kept_value_dollars=float(b.value_cents.sum()/100)))
tail=y/d.x_known_remaining.to_numpy()<=d.tail_cut.to_numpy()
remainder=raw&order&(y<0)
cases=[]
for day,g in d[d.raw>=0].groupby('day'):
    for kind,mask in [('avoided_loss',(g.order<0)&(g.value_cents<0)),('missed_loss',(g.order>=0)&(g.value_cents<0)),('sacrificed_gain',(g.order<0)&(g.value_cents>0))]:
        a=g[mask].sort_values('value_cents',ascending=kind!='sacrificed_gain').head(3).copy();a['case_kind']=kind;cases.append(a)
pd.concat(cases,ignore_index=True).to_csv(out/'attribution_cases_v1.csv',index=False)
v=dict(asof=r['asof'],source_report_sha256=hashlib.sha256((src/'report.json').read_bytes()).hexdigest(),population=r['pooled']['population'],
    raw_nonnegative=base,all_policies=rows,screen=r['screen'],
    raw_kept_value_error_fraction=err['value_term_cents']/err['total_cents'],
    raw_mean_good_fill_cents=base['good_value_cents']/base['good'],raw_mean_bad_fill_cost_cents=base['gross_loss_cents']/base['bad'],
    order_zero_cutoff_local_loss_reduction_fraction=1-z['kept']['value_cents']/base['value_cents'],
    order_zero_cutoff_good_value_retained_fraction=z['kept']['good_value_cents']/base['good_value_cents'],
    order_zero_cutoff_gross_loss_reduction_fraction=1-z['kept']['gross_loss_cents']/base['gross_loss_cents'],
    order_zero_cutoff_action_retention_fraction=z['kept']['n']/r['pooled']['population']['n'],
    order_extra_pull=z['extra_pulled_vs_raw'],order_extra_keep=z['extra_kept_vs_raw'],
    raw_tail=dict(n=int((raw&tail).sum()),loss_dollars=float(-y[raw&tail].sum()/100)),
    raw_tail_removed_by_order=dict(n=int((raw&tail&~order).sum()),loss_dollars=float(-y[raw&tail&~order].sum()/100)),
    raw_tail_remaining_after_order=dict(n=int((raw&tail&order).sum()),loss_dollars=float(-y[raw&tail&order].sum()/100)),
    original_bad_kept_by_both=dict(n=int(remainder.sum()),loss_dollars=float(-y[remainder].sum()/100)),
    order_states=group,score_corrections_are_new_learned_calibrators=True,lgbm_models_unchanged=True,
    raw_collector_rebuilt=False,portfolio_replay=False,queue_cost_estimated=False,all_variants_remain_negative=all(x['net_value_dollars']<0 for x in rows),
    interpretation='Exploratory consumed-data evidence; the order-only zero-cutoff arm is highlighted after seeing all 15 variants, not promoted. Count-matched controls isolate selection descriptively; no significance claim.')
(out/'interpretation_v1.json').write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
print(json.dumps(v,indent=2,allow_nan=False))

# Reconcile the already-completed activity-only replay, separately from this
# neutral local-action experiment. These portfolio numbers cannot be added to
# or substituted for the local-label sums above.
prior=Path(__file__).parents[1]/'stop_ev_play_20261115'/'stop_ev_play.json'
j=json.loads(prior.read_text());days=list(j['days'].values())
prior_asof=prior.with_suffix('.txt').read_text().splitlines()[0].split('as of ')[1]
replay=dict(source=str(prior),sha256=hashlib.sha256(prior.read_bytes()).hexdigest(),asof=prior_asof,
    days=list(j['days']),windows=sum(x['stat']['windows'] for x in days),controls=j['readings'],arms={})
for arm in ['QR','LDG','MD1000','KA']:
    z={k:float(sum(x['value']['M50'][arm][k] for x in days)) for k in ['pnl','pnl_old','age_lt100','age_100_500','age_ge500','pulls','shares','shares_young']}
    assert np.isclose(z['pnl']-z['pnl_old'],z['age_lt100'],rtol=0,atol=1e-7)
    assert np.isclose(z['pnl'],sum(z[k] for k in ['age_lt100','age_100_500','age_ge500']),rtol=0,atol=1e-7)
    replay['arms'][arm]=dict(total=z,mean_daily={k:v/len(days) for k,v in z.items()})
replay['young_order_loss_fraction']=replay['arms']['QR']['total']['age_lt100']/replay['arms']['QR']['total']['pnl']
replay['interpretation']='Prior activity-only replay, not a replay of our new calibration. Young-order loss is an age attribution, not causal measurement of churn cost. v2 falsifier pending, timing assumptions remain.'
(out/'prior_replay_attribution_v1.json').write_text(json.dumps(replay,indent=2)+'\n')
