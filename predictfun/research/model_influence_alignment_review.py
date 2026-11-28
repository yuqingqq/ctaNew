"""Read-only source/receipt audit; no historical data arrays or models loaded.

Prints an audit receipt. Run from the isolated model-upgrade worktree.
This verifies artifact identity and receipt totals, not numerical correctness.
"""
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/home/yuqing/ctaNew')
REV = '54408bcd'


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def require_hash(blob, expected):
    if digest(blob) != expected:
        raise ValueError('artifact digest mismatch')


def main():
    probe = b'influence-review-positive-control'
    require_hash(probe, digest(probe))
    try:
        require_hash(probe + b'corrupt', digest(probe))
    except ValueError:
        refused = True
    else:
        raise AssertionError('known-bad hash accepted')
    result = dict(asof=dt.datetime.now(dt.timezone.utc).isoformat(),
                  kind='source_and_receipt_review_not_replay', source_revision=REV,
                  controls=dict(valid_digest_passed=True, corrupted_digest_refused=refused),
                  artifacts={}, sources={})
    for study in ('influence', 'next_price'):
        base = ROOT / f'predictfun/reviews/{study}_20261114'
        for row in (base / 'SHA256SUMS').read_text().splitlines():
            expected, name = row.split()
            blob = (base / name).read_bytes()
            require_hash(blob, expected)
            runtime = ROOT / f'data/predict_fun/research/{study}_20261114' / name
            if runtime.read_bytes() != blob:
                raise ValueError('runtime/copy mismatch')
            result['artifacts'][str((base / name).relative_to(ROOT))] = expected
    for name in ('pf_future_best.py', 'pf_influence.py', 'pf_stop_depth.py',
                 'pf_pull_depth.py', 'pf_target_test.py'):
        rel = f'predictfun/research/{name}'
        blob = (ROOT / rel).read_bytes()
        committed = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{REV}:{rel}'])
        if blob != committed:
            raise ValueError('source differs from reviewed revision')
        result['sources'][rel] = digest(blob)
    base = ROOT / 'predictfun/reviews/influence_20261114'
    rows = [json.loads(p.read_text()) for p in sorted(base.glob('rows_*.json'))]
    if len(rows) != 9:
        raise ValueError('incomplete onset receipt population')
    result['onsets'] = {k: sum(r[k] for r in rows) for k in
                       ('onsets', 'onset_rows', 'onset_no_pbn', 'onset_invalid', 'fv_events')}
    t = result['onsets']
    if t['onsets'] != t['onset_rows'] + t['onset_no_pbn'] + t['onset_invalid']:
        raise ValueError('onset statuses fail reconciliation')
    base = ROOT / 'data/predict_fun/research/target_test_20261114'
    scores = [json.loads(p.read_text()) for p in base.glob('score_*.json')]
    result['target_test_snapshot'] = dict(
        scored_days=len(scores), evaluated_days=len(list(base.glob('day_*.json'))),
        report_exists=(base / 'target_test.json').exists(),
        missing_onset_predictions=sum(x['onsets_unmatched'] for x in scores),
        max_onset_prediction_difference=max(x['max_diff_at_onsets'] for x in scores))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
