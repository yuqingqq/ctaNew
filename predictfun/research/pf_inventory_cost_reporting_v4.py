"""Supersede reporting v3: select the audit flags column explicitly in pandas."""
import copy
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from . import pf_inventory_cost_reporting as v3
from . import pf_inventory_cost_study as study
from . import pf_inventory_cost_run as runner
from . import pf_inventory_cost_report as report

DECLARATION = study.DEST / 'declaration_reporting_v4.json'


def select_score_only(active, blockers):
    if 'flags' not in active.columns:
        raise ValueError('missing audit flags column')
    def eligible(flags):
        if not isinstance(flags, (list, tuple, np.ndarray)) or any(not isinstance(x, str) for x in flags):
            raise ValueError('invalid audit flags')
        return 'score_line' in flags and not blockers.intersection(flags)
    return active[active['flags'].map(eligible)]


def summarizer():
    source_path = study.HERE / 'pf_inventory_cost_report.py'
    source = source_path.read_text()
    old = "score_only=active[active.flags.map(lambda flags:'score_line' in flags and not blockers.intersection(flags))]"
    if source.count(old) != 1:
        raise ValueError('wrong frozen reporting source')
    source = source.replace(old, 'score_only=select_score_only(active,blockers)')
    ns = dict(__name__='inventory_report_generated', __package__=__package__,
              __file__=str(source_path), select_score_only=select_score_only)
    exec(compile(source, str(source_path), 'exec'), ns)
    return ns['summarize']


def controls():
    result = v3.controls()
    frame = pd.DataFrame({'flags': [['score_line'], ['score_line', 'minimum_order'], [], ['score_line', 'reserved_room']]})
    if select_score_only(frame, {'minimum_order'}).index.tolist() != [0, 3]:
        raise ValueError('audit-column positive control failed')
    refused = 0
    for bad in (pd.DataFrame({'missing': [1]}), pd.DataFrame({'flags': ['score_line']}), pd.DataFrame({'flags': [None]})):
        try:
            select_score_only(bad, set())
        except ValueError:
            refused += 1
    try:
        frame.flags.map(lambda x: True)
    except AttributeError:
        refused += 1
    if refused != 4:
        raise ValueError('audit-column bad controls failed')
    summarizer()
    return dict(result, audit_column_selection=True, audit_column_known_bad_refusals=refused)


def declare():
    original = study.check()
    if (study.DEST / 'report.json').exists():
        raise ValueError('reporting correction must precede aggregate outcomes')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=study.HERE, text=True).strip()
    extra = [Path(__file__), Path(v3.__file__), study.HERE / 'pf_inventory_cost_verify.py']
    d = copy.deepcopy(original)
    d['source_sha'].update({str(p.relative_to(study.HERE.parents[1])): study.sha(p) for p in extra})
    d['reporting_correction'] = dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(), source_commit=commit,
        supersedes=str(v3.DECLARATION), superseded_sha=study.sha(v3.DECLARATION),
        original_declaration_sha=study.sha(study.DEST / 'declaration.json'),
        reason='Retain v3 read-only check comparison; replace pandas active.flags metadata access with explicit flags-column selection. No strategy, population, or economic calculation changes; original failed logs and declarations retained.',
        source_commit_is_original_policy_freeze=True)
    study.write(DECLARATION, d)
    study.write(study.REVIEW / 'declaration_reporting_v4.json', d)
    print('DECLARED explicit audit-column correction', commit)


def check():
    study.check()
    d = json.loads(DECLARATION.read_text())
    for rel, digest in d['source_sha'].items():
        if study.sha(study.HERE.parents[1] / rel) != digest:
            raise ValueError('reporting source changed: ' + rel)
    if study.sha(v3.DECLARATION) != d['reporting_correction']['superseded_sha']:
        raise ValueError('superseded declaration changed')
    if study.sha(study.DEST / 'declaration.json') != d['reporting_correction']['original_declaration_sha']:
        raise ValueError('original declaration changed')
    return d


def main():
    check()
    if (study.DEST / 'report.json').exists() or (study.REVIEW / 'receipt_v4.json').exists():
        raise ValueError('refuse existing report before verification')
    control_result = controls()
    original = runner.write, report.write, report.check, report.summarize
    def save(path, payload):
        path = Path(path)
        if path.name in ('report.json', 'receipt_v2.json'):
            payload = dict(payload, reporting_controls=control_result, reporting_declaration_sha=study.sha(DECLARATION))
        if path == study.REVIEW / 'receipt_v2.json':
            path = study.REVIEW / 'receipt_v4.json'
        original[1](path, payload)
    runner.write, report.write, report.check, report.summarize = v3.compare_saved, save, check, summarizer()
    try:
        report.main()
    finally:
        runner.write, report.write, report.check, report.summarize = original


if __name__ == '__main__':
    if sys.argv[1:] == ['--declare']:
        declare()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit('usage: pf_inventory_cost_reporting_v4 [--declare]')
