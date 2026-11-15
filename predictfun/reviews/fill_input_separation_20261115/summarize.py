"""Export all frozen audit readings; no fitted rule or model selection."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import sys

src=Path(sys.argv[1]);out=Path(sys.argv[2]);r=json.loads((src/'report.json').read_text())
core={k:v for k,v in r.items() if k not in ['pooled','dev']}
core['full_report_sha256']=hashlib.sha256((src/'report.json').read_bytes()).hexdigest()
core['full_report_path']=str(src/'report.json');core['pooled_parts']=[]
names=list(r['pooled']);rows=[]
for start in range(0,len(names),12):
    fn=f'features_{start//12+1:02d}.json';core['pooled_parts'].append(fn)
    (out/fn).write_text(json.dumps({c:r['pooled'][c] for c in names[start:start+12]},indent=2)+'\n')
(out/'report_core_v1.json').write_text(json.dumps(core,indent=2)+'\n')
(out/'dev_v1.json').write_text(json.dumps(r['dev'],indent=2)+'\n')
for p in src.glob('day_*.json'):shutil.copyfile(p,out/p.name)
shutil.copyfile(src/'cases.csv',out/'cases.csv')
days={p.stem[4:]:json.loads(p.read_text()) for p in src.glob('day_*.json')}
def orient(x,s):return x if x is None or s==1 else 1-x
for c,z in r['pooled'].items():
    s=z['dev_direction']
    rows.append(dict(feature=c,dev_auc=r['dev'][c]['auc_bad_good'],dev_direction=s,raw_auc=z['auc_bad_good'],
        oriented_auc=z['oriented_auc'],bad_good_n=z['bad_good_n'],fill_auc=z['auc_fill'],
        material_auc=z['strong_auc'],material_n=z['strong_n'],
        matched_auc=z['matched']['auc'],oriented_matched_auc=orient(z['matched']['auc'],s),
        matched_actions=z['matched']['good']+z['matched']['bad'],matched_pairs=z['matched']['pairs'],
        matched_hour_auc=z['matched_hour']['auc'],matched_hour_actions=z['matched_hour']['good']+z['matched_hour']['bad'],
        positive_days=z['oriented_positive_days'],bid_auc=z['by_side']['0']['auc_bad_good'],ask_auc=z['by_side']['1']['auc_bad_good'],
        strict_auc=z['strict']['auc_bad_good'],strict_n=z['strict']['bad_good_n'],
        missing_good=z['missing_by_class']['good'],missing_bad=z['missing_by_class']['bad'],
        **{d:q['features'][c]['auc_bad_good'] for d,q in days.items()}))
with (out/'all_features.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
best=sorted(rows,key=lambda z:z['oriented_auc'] or 0,reverse=True)
print('POPULATION',json.dumps(r['population']))
print('LINEAGE',sum(x['actions'] for x in r['lineage'].values()),'ASOF',r['asof'])
for z in best[:18]:print('TOP',json.dumps({k:z[k] for k in ['feature','raw_auc','oriented_auc','oriented_matched_auc','matched_actions','positive_days','bid_auc','ask_auc','strict_auc','strict_n']}))
for c in ['fair_deficit','fair_shock_over_edge','bn50_shock_over_edge','bn250_shock_over_edge','micro_deficit','dfv250','gap_adv','adv_bn_r50','adv_ofi250','adv_pf_imb1','own_mv250','o_edge','o_q_frac']:
    z=r['pooled'][c];print('PHYSICAL',c,json.dumps({k:z[k] for k in ['auc_bad_good','auc_fill','strong_auc','bad_good_n','matched','matched_hour','median_good','median_bad']}))
print('RULES',json.dumps(r['rules']))
summary=dict(asof=r['asof'],n_actions=r['population']['test']['n'],controls=r['controls'],
    lineage_actions=sum(x['actions'] for x in r['lineage'].values()),lineage_fields=83,
    lineage_mismatches=sum(x['mismatches'] for x in r['lineage'].values()),
    source_sha256=core['full_report_sha256'],all_features=rows,rules=r['rules'],
    best_univariate_chosen_after_inspection_for_description_only=best[0],
    raw_collector_rebuilt=False,new_model_fits=0,interpretation='All features exported; no selected feature set or rule promoted; consumed dates only.')
(out/'interpretation_v1.json').write_text(json.dumps(summary,indent=2)+'\n')
print('DAY RULES',json.dumps({d:{k:dict(n=z['selected']['n'],value=z['selected']['value_cents'],increment=z['extra_loss_avoidance_vs_matched_cents']) for k,z in q['rules'].items()} for d,q in days.items()}))
focus=['o_q_rel','o_q_frac','adv_bnd_ofi250','adv_ofi250','adv_bn_imb','dfv250','fair_deficit','adv_bn_r50']
readings=dict(asof=r['asof'],population=r['population'],features={},rules={},statuses=r['statuses'])
for c in focus:
    z=r['pooled'][c]
    readings['features'][c]={k:z[k] for k in ['auc_bad_good','bad_good_n','matched','matched_hour','strong_auc','strong_n','strict','missing_by_class','oriented_positive_days']}
    readings['features'][c]['days']={d:dict(auc=q['features'][c]['auc_bad_good'],n=q['features'][c]['bad_good_n'],matched=q['features'][c]['matched']['auc']) for d,q in sorted(days.items())}
    readings['features'][c]['both_side_auc_above_half']=all(z['by_side'][s]['auc_bad_good'] is not None and z['by_side'][s]['auc_bad_good']>.5 for s in ['0','1'])
for k,z in r['rules'].items():
    s=z['selected'];p=r['population']['test'];daily={d:q['rules'][k]['extra_loss_avoidance_vs_matched_cents'] for d,q in sorted(days.items())}
    readings['rules'][k]=dict(selected_n=s['n'],selected_share=s['n']/p['n'],
        loss_capture_share=s['bad_value_cents']/p['bad_value_cents'],good_value_sacrifice_share=s['good_value_cents']/p['good_value_cents'],
        selected_value_dollars=s['value_cents']/100,gross_loss_dollars=-s['bad_value_cents']/100,
        gross_gain_dollars=s['good_value_cents']/100,extra_loss_avoidance_vs_matched_dollars=z['extra_loss_avoidance_vs_matched_cents']/100,
        positive_days=sum(v>0 for v in daily.values()),daily_extra_cents=daily,
        remaining_local_value_negative=z['unselected']['value_cents']<0)
readings['all_rules_leave_negative_remaining_local_value']=all(z['remaining_local_value_negative'] for z in readings['rules'].values())
readings['lineage_field_comparisons']=sum(x['actions']*x['fields'] for x in r['lineage'].values())
(out/'readings_v1.json').write_text(json.dumps(readings,indent=2)+'\n')
