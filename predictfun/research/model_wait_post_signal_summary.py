"""Presentation of completed distribution/processing/sampling experiments."""
import collections
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
REVIEW=HERE.parent/'reviews/wait_post_signal_data_20261114'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def pool_original(rows,head):
    sse=sum(x['n']*x[head]['rmse']**2 for x in rows)
    den=sum(x['n']*x[head]['rmse']**2/(1-x[head]['r2_vs_train_mean']) for x in rows if x[head]['r2_vs_train_mean'] is not None)
    return dict(n=sum(x['n'] for x in rows),r2_vs_fold_prior_mean=1-sse/den if den else None)


def main():
    audit=json.loads((REVIEW/'phase1/audit.json').read_text());first=json.loads((REVIEW/'phase1/probes.json').read_text());weighted=json.loads((REVIEW/'sampling/result.json').read_text())
    n=sum(x['n_test'] for x in audit['cells']);strata=collections.defaultdict(list)
    for c in audit['cells']:
        for s in c['strata']:strata[s['stratum']].append(s)
    support=[]
    for name,rows in strata.items():
        selected=sum(x['selection']['n'] for x in rows)
        support.append(dict(stratum=name,n=sum(x['n'] for x in rows),selected=selected,
            actual_selected_post=sum(x['selection']['n']*x['selection']['actual_post'] for x in rows if x['selection']['n'])/selected if selected else None,
            post=pool_original(rows,'post'),delta=pool_original(rows,'delta')))
    distance={k:dict(n=n,outside=sum(c[k]['outside'] for c in audit['cells']),outside_fraction=sum(c[k]['outside'] for c in audit['cells'])/n) for k in ('market_distance','state_distance')}
    rt=audit['runtime'];posts=sum(x['posts'] for x in rt);generations=sum(x['generations'] for x in rt)
    runtime=dict(posts=posts,generations=generations,unseen_combinations=sum(x['post_unseen'] for x in rt),
                 outside_five_marginal_ranges=sum(x['post_outside_marginal'] for x in rt),
                 marginal_variables=['known_position','own_exposure','other_exposure','idle_age_ms','side_price_change_while_empty'],
                 initial_generation_fraction=sum(x['initial_generation_fraction']*x['generations'] for x in rt)/generations)
    features=collections.defaultdict(list)
    for c in audit['cells']:
        for f in c['feature_shift']:
            if 'ks_distance' in f:features[f['feature']].append(f)
    shifts=[dict(feature=k,mean_cell_ks=sum(x['ks_distance'] for x in v)/len(v),outside_train_range=sum(x['outside_train_range'] for x in v),
                 test_n=sum(x['test_n'] for x in v),test_missing=sum(x['test_missing'] for x in v)) for k,v in features.items()]
    shifts.sort(key=lambda x:-x['mean_cell_ks'])
    first_arms=[]
    for arm in sorted(set(x['arm'] for x in first['cells'])):
        rows=[x for x in first['cells'] if x['arm']==arm]
        first_arms.append(dict(arm=arm,post=pool_original(rows,'post'),delta=pool_original(rows,'delta'),
             post_cells_beating_prior=sum(x['post']['r2_vs_train_mean']>0 for x in rows),delta_cells_beating_prior=sum(x['delta']['r2_vs_train_mean']>0 for x in rows)))
    weighted_arms=[]
    for arm in sorted(set(x['arm'] for x in weighted['cells'])):
        rows=[x for x in weighted['cells'] if x['arm']==arm];heads={}
        for h in ('post','delta'):
            sse=sum(x[h]['sse'] for x in rows);den=sum(x[h]['baseline_sse'] for x in rows)
            heads[h]=dict(r2_vs_fold_prior_mean=1-sse/den,cells_beating_prior=sum(x[h]['sse']<x[h]['baseline_sse'] for x in rows),
                          cells_beating_price_rung=sum(x[h]['sse']<x['baseline_'+h]['sse'] for x in rows))
        weighted_arms.append(dict(arm=arm,**heads))
    tr=[x for x in weighted['support'] if x['split']=='train'];source=list(audit['source_exclusions'].values())
    result=dict(asof=weighted['asof'],status='complete_development_study',forward_validation=False,cumulative_candidates=5,
        scope=dict(dates=sorted(audit['source_exclusions']),complete_days=5,heldout_probes=n,market_windows=5*288,
                   exclusions={k:sum(x[k] for x in source) for k in ('untimed_legs_0_60','untimed_shares_0_60','grid_steps_without_row')}),
        distance_audit=distance,support_conditioned_errors=support,runtime=runtime,largest_feature_distribution_changes=shifts[:12],
        sampling=dict(first_probability=.5,later_probability=1/32,relative_inclusion=16,first_weight=2,later_weight=32,
            training_initial_fraction_range=[min(x['initial_fraction'] for x in tr),max(x['initial_fraction'] for x in tr)],
            weighted_training_initial_fraction_range=[min(x['weighted_initial_fraction'] for x in tr),max(x['weighted_initial_fraction'] for x in tr)],
            training_effective_n_range=[min(x['effective_n'] for x in tr),max(x['effective_n'] for x in tr)]),
        unweighted_probes=first_arms,unweighted_matched_tests=first['matched_tests'],weighted_probes=weighted_arms,weighted_matched_tests=weighted['matched_tests'],
        computed_counts=dict(unweighted_tests_passed=sum(x['passes'] for x in first['matched_tests']),unweighted_tests=len(first['matched_tests']),
                             weighted_tests_passed=sum(x['passes'] for x in weighted['matched_tests']),weighted_tests=len(weighted['matched_tests'])),
        implementation=['Train-only economic normalization, missing/extreme flags, constant/duplicate feature removal, fixed regularized baseline.',
                        'Direct timing-advantage target with separate positive absolute-entry value gate.',
                        'Known hash-inclusion weights in fitting and evaluation; matched controls preserve generation type and weighted action mass.'],
        limitations=['Distance checks are descriptive support proxies, not proof that distributions or conditional relationships are unchanged.',
                     'Only five listed order-state marginal ranges were audited over every runtime entry; no claim all runtime market inputs are in range.',
                     'Sampling weights correct known inclusion probabilities, not absent joint waiting/ladder configurations or unreached probes.',
                     'Neutral-reference and runtime generation populations differ; weights are not fitted to force the runtime34% proportion.',
                     'All labels still use the frozen neutral KEEP continuation. No combined keeper/entry replay or portfolio acceptance.',
                     'All five dates were already consumed; all five candidate variants reported, no forward validation.'],
        sources={n:sha(REVIEW/n) for n in ('phase1/audit.json','phase1/probes.json','sampling/result.json')})
    with (REVIEW/'summary.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('Distribution and processing summary written')


if __name__=='__main__':main()
