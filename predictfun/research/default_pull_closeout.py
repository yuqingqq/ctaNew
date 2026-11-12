"""Final metadata integrity checks after the numerical and human reviews.

No new models, tuning, portfolio valuations or validation populations.
"""
import copy
import datetime as dt
import json
from pathlib import Path
import subprocess
from default_pull_checkpoint import WORK, ROOT, OUT, REL, sha, need, copy_exact

REVIEW = Path('predictfun/handoffs/20261112_default_pull_review.md')
TEST = ['20261026', '20261027', '20261031', '20261101', '20261102', '20261103']
MODES = {'reference', 'matched_guard', 'default', 'pressure', 'floor', 'market', 'queue', 'queue_cost'}


def shape(r):
    if r['stage'] != 'final' or r['validation'] is not False or r['days'] != TEST:
        raise ValueError('wrong final cohort')
    expected = {(s, m) for s in ('X', 'stress') for m in MODES}
    if {(x['scenario'], x['mode']) for x in r['table']} != expected or len(r['table']) != 16:
        raise ValueError('missing portfolio')
    for x in r['table']:
        if x['windows'] != 1728 or x['marked_windows'] + x['marked_missing'] != 1728:
            raise ValueError('incorrect population')
    for m, baselines in r['predicates'].items():
        for b, p in baselines.items():
            rows = [r['comparisons'][f'{s}:{m}-{b}'] for s in ('X', 'stress')]
            computed = all(x['marked_delta'] > 0 and x['exposure_delta'] <= 0 for x in rows)
            if p['all_scenarios_trading_up_exposure_down'] != computed or p['stress_complete'] is not True:
                raise ValueError('incorrect criterion')
    for x in r['comparisons'].values():
        if x['windows'] != 1728 or x['common_mark_windows'] + x['missing_either_mark'] != 1728:
            raise ValueError('incorrect paired population')


def run():
    report = json.loads((OUT / 'comparison_final.json').read_text())
    shape(report)
    bad = copy.deepcopy(report); bad['table'][0]['windows'] = 1727
    try:
        shape(bad)
    except ValueError:
        pass
    else:
        raise ValueError('bad population refusal failed')
    need(__file__, sha(__file__))
    try:
        need(__file__, '0' * 64)
    except ValueError:
        pass
    else:
        raise ValueError('changed hash refusal failed')
    d = json.loads((OUT / 'declaration.json').read_text())
    for p, h in d['source_sha'].items():
        need(WORK / p, h)
    audit = json.loads((OUT / 'completion_audit.json').read_text())
    runtime = json.loads((OUT / 'completion_runtime_v4.json').read_text())
    figure = json.loads((WORK / REL / 'tradeoff_receipt.json').read_text())
    if audit['status'] != 'PASS' or runtime['status'] != 'PASS' or audit['checked_runs'] != 22:
        raise ValueError('incomplete numerical audit')
    for name, key in [('comparison_final.json', 'comparison_sha'), ('completion_audit.json', 'completion_audit_sha'),
                      ('runtime_parity.json', 'runtime_parity_sha'), ('known_empty_controls.json', 'known_empty_controls_sha'),
                      ('declaration.json', 'declaration_sha')]:
        need(OUT / name, runtime[key])
    need(OUT / 'comparison_final.json', figure['comparison_sha'])
    need(OUT / 'completion_runtime_v4.json', figure['audit_sha'])
    for name, h in figure['artifacts'].items():
        need(WORK / REL / name, h); need(ROOT / REL / name, h)
    artifacts = 0
    for entry in audit['checks']:
        folder = OUT / 'runs' / entry['group'] / entry['day']
        for name, key in [('checks.json', 'checks_sha'), ('feature_check.json', 'feature_sha'), ('source.json', 'source_sha')]:
            need(folder / name, entry[key]); artifacts += 1
        c = json.loads((folder / 'checks.json').read_text())
        summary = json.loads((folder / 'summary.json').read_text())
        need(folder / 'independent_windows.parquet', c['independent_sha']); artifacts += 1
        stems = {}
        for arm in summary['env']['PF_ARMS'].split(';'):
            p, rules, m, size = arm.split('|')
            modes = [k for k, v in d['modes'].items() if v and v in rules.split('+')]
            if len(modes) > 1:
                raise ValueError('ambiguous arm')
            mode = modes[0] if modes else 'reference'
            stems[mode] = (f'{p}+{rules}/{m}@{size}'.replace('/', '__').replace('+', '_')
                           .replace('@', '_sz').replace(':', '-') + '.parquet')
        for v in c['variants']:
            need(folder / 'generations' / (v['mode'] + '.parquet'), v['generation_sha']); artifacts += 1
            for kind, h in v['hashes'].items():
                need(folder / kind / stems[v['mode']], h); artifacts += 1
    fit = json.loads((OUT / 'fit.json').read_text())
    need(OUT / 'predictions.parquet', fit['predictions_sha'])
    model_count = 0
    for fold in fit['folds']:
        for model, h in fold['models_sha'].items():
            need(OUT / 'models' / f'{fold["day"]}_{model}.txt', h); model_count += 1
    stages = (['controls', 'smoke'] + [f'build_{day}' for day in d['days']] + ['fit'] +
              [f'eval_{day}' for day in TEST] + ['primary_report'] +
              [f'stress_{day}' for day in TEST] + ['final_report', 'completion_audit', 'comparison_figure'])
    for stage in stages:
        digest, target = (OUT / 'logs' / (stage + '.done')).read_text().strip().split(maxsplit=1)
        log = OUT / 'logs' / (stage + '.log')
        if Path(target) != log or ' rc=0 ' not in log.read_text().splitlines()[-1]:
            raise ValueError('unsuccessful stage')
        need(log, digest)
    prior = json.loads((WORK / REL / 'prior_result_preserved.json').read_text())
    for entry in prior['artifacts']:
        original = subprocess.check_output(['git', 'show', 'd475eccb:' + entry['path']], cwd=WORK)
        for base in (ROOT, WORK):
            if (base / entry['path']).read_bytes() != original:
                raise ValueError('prior completed result changed')
    for name in ('comparison_final.json', 'completion_audit.json', 'completion_runtime_v4.json'):
        for base in (ROOT, WORK):
            need(base / REL / name, sha(OUT / name))
    need(ROOT / REVIEW, sha(WORK / REVIEW))
    paths = subprocess.check_output(['git', 'diff', '--name-only', 'd475eccb..HEAD'], cwd=WORK, text=True).splitlines()
    if any(p not in ('predictfun/HANDOFF.md', 'predictfun/STATUS.yml') and not p.startswith(
            ('predictfun/research/', 'predictfun/reviews/', 'predictfun/handoffs/')) for p in paths):
        raise ValueError('change outside research/documentation scope')
    authorization = json.loads((OUT / 'execution_two_slots_20261112.json').read_text())
    need(OUT / 'declaration.json', authorization['supersedes_operations_only_of'])
    for p, h in authorization['execution_sources'].items():
        need(WORK / p, h)
    candidates = [m for m, bs in report['predicates'].items()
                  if all(p['all_scenarios_trading_up_exposure_down'] for p in bs.values())]
    result = dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(), status='FINAL_REVIEW_INTEGRITY_PASS',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK, text=True).strip(), source_sha=sha(__file__),
        comparison_sha=sha(OUT / 'comparison_final.json'), numerical_audit_sha=sha(OUT / 'completion_audit.json'),
        runtime_audit_sha=sha(OUT / 'completion_runtime_v4.json'), human_review_sha=sha(WORK / REVIEW),
        figure_receipt_sha=sha(WORK / REL / 'tradeoff_receipt.json'), authorization_sha=sha(OUT / 'execution_two_slots_20261112.json'),
        stage_logs_verified=len(stages), run_artifact_hashes_verified=artifacts, frozen_sources_verified=len(d['source_sha']),
        saved_models_verified=model_count, checked_runs=audit['checked_runs'], feature_decisions=audit['feature_decisions'],
        saved_policy_score_checks=sum(x['scored_generations'] for x in audit['score_checks']), qualifying_candidates=candidates,
        prior_completed_results_preserved=True, research_only_diff=True, changed_paths=len(paths),
        controls='Actual final population/hash accepted; planted wrong population and changed hash refused; criteria recomputed.',
        scope='Current artifact integrity after the completed numerical audit and human source/report/PNG review. Numerical reconciliation, causal-feature equality, saved-model/action reconstruction and input rehashing are the recorded independent numerical audit, not new results from this metadata script. No fresh validation or live promotion.')
    target = OUT / 'final_review_integrity.json'
    with target.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    for base in (WORK, ROOT):
        copy_exact(target, base / REL / target.name)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    run()
