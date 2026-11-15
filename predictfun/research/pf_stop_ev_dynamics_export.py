"""Export frozen audit artifacts and arithmetic summaries; no new data/tests."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil


def export(src,out):
    r=json.loads((src/'report.json').read_text())
    ps={day:json.loads((src/f'profile_{day}.json').read_text()) for day in r['build']}
    out.mkdir(exist_ok=True,parents=True)
    shutil.copyfile(src/'report.json',out/'report_v1.json')
    with (out/'feature_profiles_v1.csv').open('w') as f:
        keys=['asof','day','feature','n','finite','missing','inf','zero','min','p001','p01','p50','p99','p999','max',
              'abs_p50','abs_p99','abs_p999','abs_max','max_over_abs_p99','mean','std']
        w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader()
        for day,p in ps.items():
            for c,z in p.items():
                v={k:z[k] for k in ['n','finite','missing','inf','zero','max_over_abs_p99','mean','std'] if k in z}
                v.update(z.get('quantiles',{}));v.update({'abs_'+k:v for k,v in z.get('abs_quantiles',{}).items()})
                w.writerow(dict(asof=r['asof'],day=day,feature=c,**v))
    with (out/'discriminators_v1.csv').open('w') as f:
        keys=['asof','scope','feature','n','severe','raw_auc','matched_auc','matched_n','pairs','groups']
        w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader()
        for c,z in r['pooled'].items():
            for scope,a in z.items():w.writerow(dict(asof=r['asof'],scope=scope,feature=c,**a))
        for day,z in r['folds'].items():
            for c,a in z['discriminators'].items():w.writerow(dict(asof=r['asof'],scope=day,feature=c,**a))
    h=r['health'];byday={day:dict(actions=z['population']['n'],filled_actions=z['population']['filled'],
        **z['test_error_tail_mass']['0.01']) for day,z in r['folds'].items()}
    rr=dict(asof=r['asof'],original_actions=sum(z['original_actions'] for z in r['build'].values()),
        primary_actions_nine_dates=sum(z['certified_actions'] for z in r['build'].values()),
        population=r['population'],strict_population=r['strict_population'],
        current_feature_infinite_fields=sum(p[c]['inf'] for p in ps.values() for c in r['features']),
        raw_kept_severe=dict(n=h['raw_kept_severe']['n'],outside_training_full_ranges=h['raw_kept_severe']['outside_range'],
            outside_training_tail_ranges=h['raw_kept_severe']['any_extreme'],fair_missing=h['raw_kept_severe']['fair_missing'],
            queue_missing=h['raw_kept_severe']['queue_missing']),
        test_error_tail_mass_by_day=byday,
        squared_ev_error_top1_daily_range=[min(z['ev_squared_error_share'] for z in byday.values()),max(z['ev_squared_error_share'] for z in byday.values())],
        squared_value_head_error_top1_daily_range=[min(z['filled_value_head_squared_error_share'] for z in byday.values()),max(z['filled_value_head_squared_error_share'] for z in byday.values())],
        label_top1_training_squared_magnitude_range=[min(z['label_tail_mass']['train']['0.01']['squared_value_share'] for z in r['folds'].values()),max(z['label_tail_mass']['train']['0.01']['squared_value_share'] for z in r['folds'].values())],
        daily_median_w30={day:math.exp(p['log_w30']['quantiles']['p50']) for day,p in ps.items()},
        daily_bn_flow250_abs_p99={day:p['adv_bn_flow250']['abs_quantiles']['p99'] for day,p in ps.items()},
        proposed_net_flow_ratio=dict(max_absolute=max(p['bn_flow_relative30']['abs_quantiles']['max'] for p in ps.values()),
            missing_test_actions=sum(ps[day]['bn_flow_relative30']['missing'] for day in r['folds']),test_actions=r['population']['n'],
            status='rejected_as_numerically_ill_conditioned_not_in_current_model'),
        level_depletion_ratio_daily_auc_below_half=all(z['discriminators']['level_depletion_over_order_and_shown']['matched_auc']<.5 for z in r['folds'].values()),
        health=r['health'],history_1s_complete_actions=sum(r['build'][day]['history_1000_grid_complete'] for day in r['folds']),
        history_fixed_price_1s_complete_actions=sum(r['build'][day]['fixed_price_visible_history1000'] for day in r['folds']),
        scope='Consumed-data description. No model trained, feature promoted, causal loss attribution, or untouched validation.',
        report_sha256=hashlib.sha256((src/'report.json').read_bytes()).hexdigest())
    (out/'readings_v1.json').write_text(json.dumps(rr,indent=2,allow_nan=False)+'\n')
    print(json.dumps(rr,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--src',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();export(a.src,a.out)
