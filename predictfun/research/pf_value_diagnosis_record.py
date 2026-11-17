"""Publish or reverify immutable diagnostic evidence; no new numerical study."""
import argparse
import json
import subprocess
from pathlib import Path
import pf_value_diagnosis as D

C = D.C
PUBLICATION = D.REVIEW / 'publication_v1.json'
REPORT = D.F.ROOT / 'predictfun/handoffs/20261117_value_diagnosis_results.md'


def read(path):
    return json.loads(Path(path).read_text())


def check_hash(path, digest):
    C.require(C.sha(path) == digest, 'publication artifact changed: ' + str(path))


def controls():
    check_hash(__file__, C.sha(__file__))
    refused = D.FIT.CHECK.refused(lambda: check_hash(__file__, '0' * 64), 'changed')
    return dict(matching_hash_positive=True, corrupt_digest_refused=refused)


def evidence():
    freeze = D.verify()
    index = read(D.REVIEW / 'index.json')
    check_hash(D.FREEZE, index['source_manifest'])
    check_hash(D.REVIEW / index['audit_file'], index['audit_sha256'])
    revision_path = D.REVIEW / 'audit_revision_v2.json'
    check_hash(revision_path, index['audit_revision_sha256'])
    revision = read(revision_path)
    check_hash(D.F.HERE / 'pf_value_diagnosis_audit_v2.py', revision['source_sha256'])
    check_hash(D.FREEZE, revision['parent_freeze_sha256'])
    check_hash(D.OUT / 'logs/audit.log', revision['supersedes']['failure_log_sha256'])
    for part, digest in revision['unchanged_inputs'].items():
        check_hash(D.OUT / part / 'receipt.json', digest)
    for name, digest in index['outputs'].items():
        check_hash(D.REVIEW / name, digest)
        check_hash(D.OUT / name, digest)
    root = D.checked_receipt(D.OUT / 'calibration')
    for day, digest in root['inputs'].items():
        check_hash(D.J.OUT / day / 'receipt.json', digest)
        D.checked_receipt(D.J.OUT / day)
    for history, digest in root['children'].items():
        dest = D.OUT / 'calibration' / history
        check_hash(dest / 'receipt.json', digest)
        receipt = D.checked_receipt(dest)
        for artifact in receipt['artifacts'].values():
            check_hash(artifact['path'], artifact['sha256'])
    D.checked_receipt(D.OUT / 'controller')
    files = [REPORT, *sorted(D.REVIEW.rglob('*.json'))]
    files = [p for p in files if p != PUBLICATION]
    return dict(source_manifest=C.sha(D.FREEZE), source_commit=freeze['source_commit'],
                audit_source_commit=revision['source_commit'],
                files={str(p.relative_to(D.F.ROOT)): C.sha(p) for p in files},
                logs={str(p): C.sha(p) for p in sorted((D.OUT / 'logs').glob('*.log'))})


def predicates():
    result = {}
    for day in (D.F.CAL, D.F.TEST):
        reports = {h: read(D.REVIEW / 'calibration' / h / (day + '_report.json'))
                   for h in D.HISTORY}
        for kind in ('cancel', 'post'):
            scores = {h: r['economic']['metrics']['full:' + kind]['rmse_cents']
                      for h, r in reports.items()}
            g = reports['H3']['calibration']['groups'][f"kind:('{kind}',)"]
            bins = reports['H3']['economic']['value_bins']['full:' + kind]
            positive = {k: v for k, v in bins.items()
                        if float(k.split(',')[0][1:]) >= 0}
            result[day + ':' + kind] = dict(
                asof=reports['H3']['asof'], n=g['full']['n'],
                history_rmse_cents=scores,
                history_monotone_improvement=scores['H1'] > scores['H2'] > scores['H3'],
                matched_history_improves_h1=scores['H3_MATCHED_N1'] < scores['H1'],
                fill_auc_improves=g['full']['fill']['auc'] > g['mechanical']['fill']['auc'],
                fill_brier_improves=g['full']['fill']['brier'] < g['mechanical']['fill']['brier'],
                fill_logloss_improves=g['full']['fill']['logloss'] < g['mechanical']['fill']['logloss'],
                positive_band_n={k: v['n'] for k, v in positive.items()},
                all_positive_bands_actual_negative=bool(positive) and all(
                    v['actual_mean'] < 0 for v in positive.values()))
    ctrl = read(D.REVIEW / 'controller/report.json')
    groups = ctrl['sensitivity'].values()
    result['controller'] = dict(
        asof=ctrl['asof'], n=ctrl['first_review_pairs'],
        sample_n=sum(g['n'] for g in groups),
        mirrored_preference_changed=sum(g['mirrored_preference_changed'] for g in groups),
        sample_cancel_n=ctrl['groups']['cancel']['sample_n'],
        cancellation_sensitivities=ctrl['groups']['cancel']['counts'])
    return result


def run(mode):
    checks = controls()
    verified = evidence()
    summary = predicates()
    if mode == 'publish':
        C.require(not PUBLICATION.exists(), 'publication already exists')
        C.dump(PUBLICATION, dict(asof=C.stamp(), **verified, predicates=summary,
               recorder_source_sha256=C.sha(__file__), controls=checks,
               recorder_source_commit=subprocess.check_output(
                   ['git', 'rev-parse', 'HEAD'], cwd=D.F.ROOT, text=True).strip(),
               controller_rollout=False, promotion=False))
    else:
        saved = read(PUBLICATION)
        check_hash(__file__, saved['recorder_source_sha256'])
        for key, value in verified.items():
            C.require(value == saved[key], 'publication evidence mismatch: ' + key)
        C.require(summary == saved['predicates'], 'publication predicate mismatch')
        C.require(checks == saved['controls'], 'publication control mismatch')
    print(json.dumps(dict(mode=mode, asof=C.stamp(), controls=checks,
                          publication_sha256=C.sha(PUBLICATION), predicates=summary)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['publish', 'verify'])
    run(parser.parse_args().mode)
