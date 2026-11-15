"""Mechanical interpretation of the frozen model-choice report; no selection."""
from pathlib import Path
import json
import hashlib
import sys

p=Path(sys.argv[1]);r=json.loads(p.read_text());f=r['folds'];names=['hurdle','direct','components','extra_trees']
summary=dict(asof=r['asof'],source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),n=r['n_unique_test_actions'],
    controls=r['controls'],pooled=r['pooled'],days={},comparisons={},side_totals={})
for d,z in f.items():
    summary['days'][d]=dict(n=z['test_n'],excluded=z['excluded_test'],train_n=z['train_n'],
        diagnostics=z['diagnostics'],offsets=z['calibration_offsets'],
        models={n:{k:z['variants'][n][k] for k in ['test','calibrated_test']} for n in names})
for n in names:
    summary['comparisons'][n]=dict(
        mse_better_than_hurdle_days=sum(z['variants'][n]['test']['mse']<z['variants']['hurdle']['test']['mse'] for z in f.values()),
        ic_better_than_hurdle_days=sum(z['variants'][n]['test']['ic']>z['variants']['hurdle']['test']['ic'] for z in f.values()),
        same_count_better_than_hurdle_days=sum(z['variants'][n]['test']['same_count_value_cents']>z['variants']['hurdle']['test']['same_count_value_cents'] for z in f.values()),
        same_count_better_than_b2_days=sum(z['variants'][n]['test']['same_count_value_cents']>z['b2']['same_count_value_cents'] for z in f.values()),
        raw_positive_set_losing_days=sum(z['variants'][n]['test']['kept_value_cents']<0 for z in f.values()),
        calibrated_positive_set_losing_days=sum(z['variants'][n]['calibrated_test']['kept_value_cents']<0 for z in f.values()),
        calibration_improves_mean_bias_days=sum(abs(z['variants'][n]['calibrated_test']['optimism'])<abs(z['variants'][n]['test']['optimism']) for z in f.values()))
    summary['side_totals'][n]={s:dict(n=sum(z['variants'][n]['side'][s]['n'] for z in f.values()),
        kept_n=sum(z['variants'][n]['side'][s]['kept_n'] for z in f.values()),
        kept_value_cents=sum(z['variants'][n]['side'][s]['kept_value_cents'] for z in f.values())) for s in ['0','1']}
summary['fill_auc_range']=[min(z['diagnostics']['fill_auc'] for z in f.values()),max(z['diagnostics']['fill_auc'] for z in f.values())]
summary['filled_r2_range']=[min(z['diagnostics']['filled_value_r2'] for z in f.values()),max(z['diagnostics']['filled_value_r2'] for z in f.values())]
summary['all_positive_sets_lose_pooled']=all(r['pooled'][n]['kept_value_cents']<0 for n in names)
summary['all_calibrated_positive_sets_lose_pooled']=all(r['pooled'][n+'_calibrated']['kept_value_cents']<0 for n in names)
summary['all_same_count_selections_lose_pooled']=all(r['pooled'][n]['same_count_value_cents']<0 for n in names)
Path(sys.argv[2]).write_text(json.dumps(summary,indent=2)+'\n')
for n in names+['b2']:
    z=r['pooled'][n]
    print(n,{k:z[k] for k in ['n','ic','mse','optimism','top_pred','top_value','kept_n','kept_value_cents','same_count_n','same_count_value_cents','same_count_increment_cents','tail_auc','risk_tail_capture']})
for n in names:
    z=r['pooled'][n+'_calibrated'];print(n+'_calibrated',{k:z[k] for k in ['optimism','top_pred','top_value','kept_n','kept_value_cents']})
print(json.dumps({k:summary[k] for k in ['asof','controls','comparisons','side_totals','fill_auc_range','filled_r2_range','all_positive_sets_lose_pooled','all_calibrated_positive_sets_lose_pooled']},indent=2))
