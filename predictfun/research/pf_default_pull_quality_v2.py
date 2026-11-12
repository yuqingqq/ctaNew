"""Frozen, descriptive pull-quality audit of saved neutral actions; no new strategy."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys

WORK = Path(__file__).resolve().parents[2]
ROOT = Path('/home/yuqing/ctaNew')
PRIOR = ROOT / 'data/predict_fun/research/default_pull_20261111_v4'
OUT = ROOT / 'data/predict_fun/research/default_pull_quality_20261112_v2'
REL = Path('predictfun/reviews/default_pull_quality_20261112_v2')
TEST = ['20261026', '20261027', '20261031', '20261101', '20261102', '20261103']
KEYS = ['day', 'side', 'hour', 'age_bin', 'inv_sign']
EPS = 1e-9


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(name, obj):
    def scalar(value):
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    text = json.dumps(obj, indent=2, allow_nan=False, default=scalar) + '\n'
    for base in (OUT, WORK / REL, ROOT / REL):
        base.mkdir(parents=True, exist_ok=True)
        with (base / name).open('x') as f:
            f.write(text)


def source():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK, text=True).strip()
    rel = str(Path(__file__).relative_to(WORK))
    if subprocess.check_output(['git', 'show', commit + ':' + rel], cwd=WORK) != Path(__file__).read_bytes():
        raise ValueError('commit checker source before declaration')
    return commit, rel


def declare():
    commit, rel = source()
    files = ['predictions.parquet', 'fit.json', 'comparison_final.json', 'completion_audit.json',
             'completion_runtime_v4.json', 'declaration.json']
    inputs = {str(PRIOR / f): sha(PRIOR / f) for f in files}
    for group in ('eval', 'stress'):
        for day in TEST:
            folder = PRIOR / 'runs' / group / day
            checks = json.loads((folder / 'checks.json').read_text())
            if checks['status'] != 'PASS':
                raise ValueError('unverified actual-policy run')
            inputs[str(folder / 'checks.json')] = sha(folder / 'checks.json')
            for mode in ('market', 'queue', 'queue_cost'):
                p = folder / 'generations' / (mode + '.parquet')
                v = next(x for x in checks['variants'] if x['mode'] == mode)
                if sha(p) != v['generation_sha']:
                    raise ValueError('actual generation hash mismatch')
                inputs[str(p)] = v['generation_sha']
    fit = json.loads((PRIOR / 'fit.json').read_text())
    if inputs[str(PRIOR / 'predictions.parquet')] != fit['predictions_sha']:
        raise ValueError('prediction identity')
    for fold in fit['folds']:
        for model, h in fold['models_sha'].items():
            p = PRIOR / 'models' / f"{fold['day']}_{model}.txt"
            if sha(p) != h:
                raise ValueError('saved model identity')
            inputs[str(p)] = h
    d = dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(), source_commit=commit,
             source_path=rel, source_sha=sha(__file__), input_sha=inputs, days=TEST,
             validation=False, candidates_added=0, model_fits=0,
             supersedes='default_pull_quality_20261112',
             correction='Only convert NumPy scalars when writing JSON. Prior run finished calculations but failed serialization; no prior numerical result was published. Original source, declaration and failed log retained.',
             design='Diagnostic only on saved six-day neutral test actions, one hash-selected probe per order; all unfilled/zero actions retained. Existing labels price effective cancellation delay and a same-price reset after 1s with lost queue. Preserve all upstream statuses. Actual-policy generations used only for population/score diagnostics, never as labeled counterfactuals.',
             policies='Existing market score<0, queue score<0, queue score<-0.01*remaining; all-pull/no-pull controls. No threshold search. Pull beneficial if RESET-KEEP>1e-9 c, harmful if <-1e-9 c; otherwise indifferent. Outcome-sign subsets descriptive only.',
             comparisons='Within day/side/UTC-hour/order-age-bucket/inventory-sign, match every model pull count using lowest guard_margin or lowest market_score, breaking ties by action id. guard_margin is only the saved value-line component, not the complete PM90/G1/G2 gate. Also split at guard_margin>0 descriptively, not as a deployable entry policy.',
             null=dict(draws=500, seed=20261111, keys=KEYS,
                       design='Reproduce original matched-count shuffle and report exact stratum expectation; intervals bootstrap six UTC-day summed differences 1000 times. No action-level significance or fresh validation.'),
             bins=[None, -5, -1, -.1, 0, .1, 1, 5, None],
             metrics='Selected beneficial/harmful/indifferent counts, avoided relative harm and sacrificed relative value, missed beneficial pulls, KEEP/RESET markout and inventory penalty decomposition, score calibration by fixed bins, daily and age-stratified values, matched guard/market incremental selection; actual generation score/pull distributions and label availability.',
             limitations='Local overlapping counterfactual values in cents, not portfolio P&L. Primary delay/queue only for labels; actual stress has no paired labels. Conditional LODO guard chronology. No zero-delay counterfactual. No extra raw/protected data or new portfolio replay.',
             controls='Synthetic oracle/anti-oracle/no-op, matched-count ranking, duplicate/missing/incorrect-label refusals, exact mark/risk decomposition, saved-null identity, no-op label assertion and no silently missing comparisons.')
    write('declaration.json', d)
    print('DECLARED', commit, len(inputs), flush=True)


def check_actions(a):
    import numpy as np
    cols = ['label', 'keep_value', 'reset_value', 'label_q', 'reset_q', 'remaining',
            'market_score', 'queue_score', 'guard_margin']
    if a.id.duplicated().any() or not a.status.eq('eligible').all():
        raise ValueError('action population identity')
    if not np.isfinite(a[cols].to_numpy()).all():
        raise ValueError('missing action values')
    if not np.allclose(a.label, a.keep_value - a.reset_value, atol=1e-9, rtol=0):
        raise ValueError('label identity')


def summarize(a, mask):
    import numpy as np
    b = a.loc[mask]; benefit = -b.label
    good = benefit > EPS; bad = benefit < -EPS
    result = dict(n=len(b), beneficial=int(good.sum()), harmful=int(bad.sum()),
                  indifferent=int((~good & ~bad).sum()),
                  both_unfilled=int((b.label_q.eq(0) & b.reset_q.eq(0)).sum()),
                  benefit_c=float(benefit.sum()), good_benefit_c=float(benefit[good].sum()),
                  bad_cost_c=float(-benefit[bad].sum()),
                  predicted_benefit_c=float(-b['selected_score'].sum()),
                  keep_value_c=float(b.keep_value.sum()), reset_value_c=float(b.reset_value.sum()),
                  keep_mark_c=float(b.keep_mark.sum()), reset_mark_c=float(b.reset_mark.sum()),
                  mark_benefit_c=float((b.reset_mark - b.keep_mark).sum()),
                  risk_benefit_c=float((b.keep_risk - b.reset_risk).sum()))
    if not np.isclose(result['benefit_c'], result['good_benefit_c'] - result['bad_cost_c'], atol=1e-7):
        raise ValueError('sign decomposition')
    if not np.isclose(result['benefit_c'], result['mark_benefit_c'] + result['risk_benefit_c'], atol=1e-7):
        raise ValueError('risk decomposition')
    return result


def matched(a, pull, score):
    import numpy as np
    z = np.zeros(len(a), dtype=bool)
    for ix in a.groupby(KEYS, dropna=False, sort=True).indices.values():
        n = int(pull[ix].sum())
        order = np.lexsort((a.id.to_numpy()[ix], a[score].to_numpy()[ix]))
        z[ix[order[:n]]] = True
        if int(z[ix].sum()) != n:
            raise ValueError('matched stratum count')
    return z


def controls():
    import numpy as np
    import pandas as pd
    a = pd.DataFrame(dict(id=['a', 'b', 'c'], status=['eligible'] * 3,
                          label=[-2., 3., 0.], keep_value=[-2., 3., 0.], reset_value=[0.] * 3,
                          label_q=[1., 1., 0.], reset_q=[0.] * 3, remaining=[5.] * 3,
                          market_score=[-2., 3., 0.], queue_score=[-2., 3., 0.], guard_margin=[3., -2., 0.]))
    check_actions(a)
    for key in KEYS:
        a[key] = 0
    a['selected_score'] = a.queue_score
    a['keep_mark'] = a.keep_value; a['reset_mark'] = 0.; a['keep_risk'] = 0.; a['reset_risk'] = 0.
    if summarize(a, a.label.lt(0))['benefit_c'] != 2 or summarize(a, a.label.gt(0))['benefit_c'] != -3:
        raise ValueError('oracle/anti-oracle control')
    if summarize(a, a.label.eq(0))['benefit_c'] != 0:
        raise ValueError('no-op control')
    if not np.array_equal(matched(a, np.array([True, False, False]), 'guard_margin'), [False, True, False]):
        raise ValueError('matched ranking control')
    for bad in (pd.concat([a, a.iloc[[0]]], ignore_index=True), a.assign(label=1.), a.assign(queue_score=np.nan)):
        try:
            check_actions(bad)
        except ValueError:
            pass
        else:
            raise ValueError('known-bad input accepted')


def run():
    import numpy as np
    import pandas as pd
    from predictfun.research.pf_order_policy_report import null, bootstrap
    d = json.loads((OUT / 'declaration.json').read_text())
    if sha(__file__) != d['source_sha']:
        raise ValueError('changed checker')
    for p, h in d['input_sha'].items():
        if sha(p) != h:
            raise ValueError('changed input: ' + p)
    # The two imported instruments must still match the original numerical freeze.
    prior_d = json.loads((PRIOR / 'declaration.json').read_text())
    for p, h in prior_d['source_sha'].items():
        if sha(WORK / p) != h:
            raise ValueError('changed frozen numerical source')
    controls()
    a = pd.read_parquet(PRIOR / 'predictions.parquet'); check_actions(a)
    fit = json.loads((PRIOR / 'fit.json').read_text())
    prior = json.loads((PRIOR / 'comparison_final.json').read_text())
    if sorted(a.day.unique()) != TEST or len(a) != sum(f['test_n'] for f in fit['folds']):
        raise ValueError('test population changed')
    a['hour'] = (a.decision // 3600000).astype(int)
    a['age_bin'] = pd.cut(a.order_age_ms, [0,100,250,500,1000,2000,5000,np.inf], include_lowest=True).astype(str)
    a['inv_sign'] = np.sign(a.inventory)
    a['keep_mark'] = .5 * (a.label_value1 + a.label_value5)
    a['reset_mark'] = .5 * (a.reset_value1 + a.reset_value5)
    for name, qty in [('keep', 'label_q'), ('reset', 'reset_q')]:
        dq = np.where(a.side.eq(0), 1, -1) * a[qty]
        a[name + '_risk'] = .01 * ((a.inventory + dq) ** 2 - a.inventory ** 2)
        np.testing.assert_allclose(a[name + '_value'], a[name + '_mark'] - a[name + '_risk'], atol=1e-8, rtol=0)
    if not np.isfinite(a[['keep_mark','reset_mark','keep_risk','reset_risk']].to_numpy()).all():
        raise ValueError('missing decomposition')
    zero = a.label_q.eq(0) & a.reset_q.eq(0)
    if not np.all(np.abs(a.loc[zero, 'label']) <= EPS):
        raise ValueError('zero-fill action has value')
    policies = {}; daily = []; calibration = []; subgroups = []
    a['selected_score'] = a.queue_score
    all_actions = summarize(a, np.ones(len(a), dtype=bool))
    for mode in ('market', 'queue', 'queue_cost'):
        a['selected_score'] = a.market_score if mode == 'market' else a.queue_score
        pull = (a.selected_score < (-.01 * a.remaining if mode == 'queue_cost' else 0.)).to_numpy()
        selected = summarize(a, pull); kept = summarize(a, ~pull)
        n = null(a, pull, -a.label, KEYS)
        for k in ('n','kept','groups','shuffleable','observed_c','null_mean_c','null_p95_c','p_ge','draws'):
            if not np.isclose(n[k], prior['diagnostics'][mode][k], atol=1e-7, rtol=0):
                raise ValueError('saved null identity: ' + k)
        masks = {key: matched(a, pull, key) for key in ('guard_margin', 'market_score')}
        expected = np.zeros(len(a))
        for ix in a.groupby(KEYS, dropna=False, sort=True).indices.values():
            expected[ix] = pull[ix].mean() * (-a.label.to_numpy()[ix])
        ds = []
        for day in TEST:
            ix = a.day.eq(day).to_numpy(); obs = float((-a.label.to_numpy()[ix & pull]).sum())
            row = dict(mode=mode, day=day, n=int(ix.sum()), selected=summarize(a, ix & pull),
                       kept=summarize(a, ix & ~pull), random_expected_c=float(expected[ix].sum()))
            row['vs_random_c'] = obs - row['random_expected_c']
            for key, mask in masks.items():
                row[key + '_matched'] = summarize(a, ix & mask)
                row['vs_' + key + '_c'] = obs - row[key + '_matched']['benefit_c']
            daily.append(row); ds.append(row)
        comparisons = {}
        for key in ('random', 'guard_margin', 'market_score'):
            values = [x['vs_' + key + '_c'] for x in ds]
            comparisons[key] = dict(delta_c=float(sum(values)), day_interval_c=bootstrap(values),
                                    positive_days=int(sum(v > EPS for v in values)), days=len(values))
        policies[mode] = dict(selected=selected, kept=kept, null=n, comparisons=comparisons,
                              matched={k:summarize(a,v) for k,v in masks.items()},
                              selected_action_ids_sha=hashlib.sha256('\n'.join(a.loc[pull,'id']).encode()).hexdigest())
        bins = pd.cut(a.selected_score, [-np.inf,-5,-1,-.1,0,.1,1,5,np.inf], right=False)
        for name, idx in a.groupby(bins, observed=True).groups.items():
            mask = a.index.isin(idx)
            calibration.append(dict(mode=mode, score_bin=str(name), **summarize(a,mask)))
        for field, groups in [('age_bin', a.age_bin), ('guard_value_component', np.where(a.guard_margin>0,'positive','nonpositive'))]:
            for name in sorted(set(groups)):
                ix = np.asarray(groups == name)
                subgroups.append(dict(mode=mode,field=field,group=name,n=int(ix.sum()),
                                      selected=summarize(a,ix & pull),kept=summarize(a,ix & ~pull)))
        print('POLICY_DONE', mode, flush=True)
    actual = []
    cols = ['id','day','status','score_count','score','pull','label','order_age_ms','life_ms']
    for group in ('eval','stress'):
        for mode in ('market','queue','queue_cost'):
            parts = [pd.read_parquet(PRIOR/'runs'/group/day/'generations'/f'{mode}.parquet',columns=cols) for day in TEST]
            g = pd.concat(parts,ignore_index=True)
            if g.id.duplicated().any(): raise ValueError('duplicate actual generation')
            scored = g.score_count.gt(0); pulled = g.pull.eq(1)
            old = next(x for x in prior['table'] if x['mode']==mode and x['scenario']==('X' if group=='eval' else 'stress'))
            if len(g)!=old['generations'] or int(pulled.sum())!=old['model_pulls'] or int(g.score_count.sum())!=old['model_scores']:
                raise ValueError('actual policy population')
            actual.append(dict(scenario=group,mode=mode,n=len(g),scored_generations=int(scored.sum()),
                               model_pulls=int(pulled.sum()),labeled_generations=int(g.label.notna().sum()),
                               stored_score_semantics='First keep score, overwritten by eventual pull score; not every rescore.',
                               stored_score_quantiles={str(q):float(g.loc[scored,'score'].quantile(q)) for q in (.1,.5,.9)},
                               scored_age_median_ms=float(g.loc[scored,'order_age_ms'].median()),
                               pulled_age_median_ms=float(g.loc[pulled,'order_age_ms'].median()),
                               order_life_median_ms=float(g.life_ms.median()),statuses={str(k):int(v) for k,v in g.status.value_counts().items()}))
    r = dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(), prior_as_of=prior['as_of'],
             declaration_sha=sha(OUT/'declaration.json'), source_commit=d['source_commit'],
             validation=False, status='PASS', days=TEST, n=len(a),
             upstream_statuses=fit['statuses'], all_actions=all_actions, policies=policies,
             daily=daily, calibration=calibration, subgroups=subgroups, actual_policy=actual,
             controls='Synthetic oracle, anti-oracle, no-op, matched ranking and duplicate/nonfinite/wrong-label refusal passed; real label decomposition, original 500-draw null reproduction, source/input hashes and actual policy populations verified.',
             predicates=dict(all_models_incremental_to_random=all(v['comparisons']['random']['delta_c']>0 for v in policies.values()),
                             queue_incremental_to_matched_market=policies['queue']['comparisons']['market_score']['delta_c']>0,
                             queue_incremental_to_matched_guard_value=policies['queue']['comparisons']['guard_margin']['delta_c']>0,
                             all_pull_exceeds_queue_local_benefit=all_actions['benefit_c']>policies['queue']['selected']['benefit_c'],
                             actual_policy_counterfactual_labels_available=all(x['labeled_generations']>0 for x in actual)),
             limits=d['limitations'])
    write('result.json',r)
    print('PULL_QUALITY_PASS',OUT,flush=True)


if __name__ == '__main__':
    {'declare':declare,'run':run}[sys.argv[1]]()
