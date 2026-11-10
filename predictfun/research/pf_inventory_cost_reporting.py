"""Superseding report adapter: verify existing checks without recreating them.

The original exclusive writer correctly refuses replacement, but the original
aggregator calls the runner's write-producing verifier a second time. Keep all
declared simulation/report calculations unchanged; compare the newly computed
check payload to the saved payload, ignoring only the observation timestamp.
"""
import copy
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from . import pf_inventory_cost_study as study
from . import pf_inventory_cost_run as runner
from . import pf_inventory_cost_report as report

DECLARATION = study.DEST / 'declaration_reporting_v3.json'


def compare_saved(path, computed):
    path = Path(path)
    if path.name != 'checks.json' or not path.is_file():
        raise ValueError('expected an existing checks artifact')
    saved = json.loads(path.read_text())
    if not isinstance(saved, dict) or 'as_of' not in saved or 'as_of' not in computed:
        raise ValueError('missing checks provenance')
    strip = lambda d: {k: v for k, v in d.items() if k != 'as_of'}
    if strip(saved) != strip(computed):
        raise ValueError('recomputed checks disagree with immutable artifact')


def controls():
    payload = dict(as_of='first observation', markets=288,
                   artifacts={'fills/example.parquet': 'original-hash'})
    refused = 0
    with tempfile.TemporaryDirectory(prefix='inventory-report-control-') as tmp:
        path = Path(tmp) / 'checks.json'
        study.write(path, payload)
        original = path.read_bytes()
        later = dict(payload, as_of='later observation')
        compare_saved(path, later)
        for kind in ('market_count', 'artifact_hash', 'missing_provenance'):
            bad = copy.deepcopy(later)
            if kind == 'market_count':
                bad['markets'] = 287
            elif kind == 'artifact_hash':
                bad['artifacts']['fills/example.parquet'] = 'changed-hash'
            else:
                del bad['as_of']
            try:
                compare_saved(path, bad)
            except ValueError:
                refused += 1
        try:
            compare_saved(Path(tmp) / 'missing' / 'checks.json', later)
        except ValueError:
            refused += 1
        if path.read_bytes() != original or refused != 4:
            raise ValueError('reporting controls failed')
    return dict(equal_payload_accepted=True, timestamp_only_change_accepted=True,
                original_bytes_preserved=True, known_bad_refusals=refused)


def declare():
    original = study.check()
    if (study.DEST / 'report.json').exists():
        raise ValueError('reporting correction must precede aggregate outcomes')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=study.HERE, text=True).strip()
    extra = [Path(__file__), study.HERE / 'pf_inventory_cost_verify.py']
    d = copy.deepcopy(original)
    d['source_sha'].update({str(p.relative_to(study.HERE.parents[1])): study.sha(p) for p in extra})
    d['reporting_correction'] = dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(), source_commit=commit,
        supersedes=str(study.DEST / 'declaration.json'), superseded_sha=study.sha(study.DEST / 'declaration.json'),
        reason='Original report attempts an exclusive second write of existing checks.json; compare the recomputed payload read-only, ignoring only as_of. Policies, populations, calculations and original output artifacts unchanged.',
        source_commit_is_original_policy_freeze=True)
    study.write(DECLARATION, d)
    study.write(study.REVIEW / 'declaration_reporting_v3.json', d)
    print('DECLARED read-only report correction', commit)


def check():
    study.check()
    d = json.loads(DECLARATION.read_text())
    for rel, digest in d['source_sha'].items():
        if study.sha(study.HERE.parents[1] / rel) != digest:
            raise ValueError('reporting source changed: ' + rel)
    if study.sha(study.DEST / 'declaration.json') != d['reporting_correction']['superseded_sha']:
        raise ValueError('original declaration changed')
    return d


def main():
    check()
    if (study.DEST / 'report.json').exists() or (study.REVIEW / 'receipt_v3.json').exists():
        raise ValueError('refuse existing report before verification')
    control_result = controls()
    original_runner_write, original_report_write, original_check = runner.write, report.write, report.check
    def save(path, payload):
        path = Path(path)
        if path.name in ('report.json', 'receipt_v2.json'):
            payload = dict(payload, reporting_controls=control_result,
                           reporting_declaration_sha=study.sha(DECLARATION))
        if path == study.REVIEW / 'receipt_v2.json':
            path = study.REVIEW / 'receipt_v3.json'
        original_report_write(path, payload)
    runner.write, report.write, report.check = compare_saved, save, check
    try:
        report.main()
    finally:
        runner.write, report.write, report.check = original_runner_write, original_report_write, original_check


if __name__ == '__main__':
    if sys.argv[1:] == ['--declare']:
        declare()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit('usage: pf_inventory_cost_reporting [--declare]')
