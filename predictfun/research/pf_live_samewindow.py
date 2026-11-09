#!/usr/bin/env python3
"""Run the unchanged research pipeline (frozen pm_ladder_v3 engine) on the windows the Tokyo live runs traded on 2026-11-08
(user 2026-11-09: "do the local replay here, check what's the main discrepancy"). Adapted from pmmmrust
review/pfmm-same-window-local-20261108:scripts/pf_local_comparison.py (Tokyo-5 cohort) with the cohort, hours and runs changed:
  runs: tokyo-6 1794108465172 (pm_ladder_v3, complete) and basis-tokyo-1 1794147344737 (pm_ladder_v3_basis_v1, prefix), data branch
        data/tokyo-live-20261109 e41bb5c; cohort = every complete window of either run ending by 2026-11-08 23:00Z (122 + 98);
  hours: 2026-11-08 02Z..23Z only, so no file of the 11-09..11-13 v3 forward block is opened.
The source tree is read-only. All outputs and bounded raw-hour symlinks live outside Git (io paths redirected into --out; REST /
WS rows filtered to cohort markets). Same-window diagnostic, not a forward validation. Each `step` runs under run_light.sh.
"""
import argparse
import datetime as dt
import hashlib
import gzip
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

ENGINE_SHA = 'd678e7c3c4e47c6d60ebc031edf8d7fc7317210d5e58ff301d92d7ae0ece5f17'
CUTOFF = 1794178800000   # 2026-11-08 23:00Z: windows ending by then
RUNS = (('v6', '1794108465172', 122), ('basis', '1794147344737', 98))
DATA_COMMIT = 'e41bb5c5a97cbda2b19e89f8658276aa1beed0ae'
ARM = 'MICRO|G4+G2+G6+G5+G3+G1+PM90+GS+LD2+CAP10+XNA+XMIN1+XV+RO60+S240|mid|AM:5:25:95'
MODULES = ('pf_step1', 'pf_step1_timing', 'pf_step1_booktime',
           'pf_step5_features_rl', 'pf_step5_c3_preds', 'pf_txmap',
           'pf_v1_apply', 'pf_preds_startknown', 'pf_step4_ladder_v3')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def verify_artifacts(repo, capture):
    result={}
    for r, run, _ in RUNS:
        with gzip.open(capture/f'{r}/run-{run}/inputs.jsonl.gz','rt') as f:
            header=json.loads(next(f))
        if header['protocol']!='PFMM_LADDER_JOURNAL_V3' or header['ladder']['version'] not in ('pm_ladder_v3', 'pm_ladder_v3_basis_v1'):
            raise ValueError('wrong capture protocol or version')
        if header['ladder']['rules'] != ARM.split('|')[1].split('+') or header['ladder']['size'] != ARM.split('|')[3]:
            raise ValueError(f'{r}: live rules differ from the frozen v3 arm')
        for name in ('params','guard_bid','guard_ask'):   # each run's artifacts at its own build commit
            ref=header['ladder'][name]
            blob=subprocess.run(['git','-C',str(repo),'show',f"{header['build']}:{ref['path']}"],capture_output=True,check=True).stdout
            got=hashlib.sha256(blob).hexdigest()
            if got!=ref['sha256']: raise ValueError(f'capture artifact mismatch: {r} {name}')
            result[f'{r} {name}']=got
        result[f'{r} version']=header['ladder']['version']; result[f'{r} build']=header.get('build')
    return result


def cohort(capture):
    out = []
    for r, run, n in RUNS:
        L = json.load(open(capture / f'{r}/stageA/ledger.json'))
        w = [dict(x, run=r) for x in L['windows'] if 'complete' in x['status'] and (x['window_start_s'] + 300) * 1000 <= CUTOFF]
        if len(w) != n:
            raise ValueError(f'{r}: {len(w)} windows, expected {n}')
        out += w
    if len({r['market'] for r in out}) != len(out):
        raise ValueError('duplicate market across runs')
    return sorted(out, key=lambda r: (r['window_start_s'], r['market']))


def initialize(a):
    root, out, capture = map(Path, (a.research, a.out, a.capture))
    if out.exists():
        raise ValueError('output already exists; refuse to overwrite a diagnostic')
    engine = root / 'predictfun/research/pf_step4_ladder_v3.py'
    if sha(engine) != ENGINE_SHA:
        raise ValueError('frozen engine hash differs')
    windows = cohort(capture)
    hours = [f'20261108_{h:02}' for h in range(2, 24)]
    pf = 'data/predict_fun/runs/20261021T162823Z_0dd45f2d5a'
    ref = 'data/predict_reference/runs/20261022T132901Z_8c043d17b6'
    groups = [(pf + '/' + d, '.jsonl.gz') for d in ('rest', 'predict_ws', 'events')]
    groups += [(ref + '/reference_prices', '.jsonl.gz')]
    groups += [('data/mm_hf/raw/' + d + '/BTCUSDT', '.csv.gz') for d in ('bookTicker', 'trade')]
    raw = []
    for folder, ext in groups:
        for h in hours:
            p = root / folder / (h + ext)
            if not p.is_file() or not p.stat().st_size:
                raise ValueError(f'missing source dependency {p}')
            raw.append(p)
    out.mkdir(parents=True)
    # Declare the target, metrics and limitations before any new result.
    scope = dict(schema=1, declared_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                 runs=[r[:2] for r in RUNS], capture=str(capture), research=str(root), cutoff_ms=CUTOFF,
                 data_branch_commit=DATA_COMMIT,
                 windows=[dict(market=w['market'], ws=w['window_start_s'], run=w['run']) for w in windows],
                 hours=hours, days_consumed=['20261108'],
                 untouched_forward_days=['20261109', '20261110', '20261111', '20261112', '20261113'],
                 interpretation='descriptive same-window diagnostic; no null test, CI, tuning or profitability inference',
                 metrics=['windows and statuses', 'fill legs/shares', 'settlement PnL excluding rebates',
                          'paired cost', 'markout', 'requests', 'inventory and spread/carry decomposition'],
                 engine_sha256=ENGINE_SHA, arm=ARM,
                 timings=dict(J='WS transaction anchor minus seeded within-frame jitter',
                              X='WS transaction anchor minus 50ms; activation queue max before/after'),
                 budget='live $150 unchanged; frozen research arm has no account-level XCAP; report difference',
                 limitations=['one partial day (20 h); no statistical robustness inference',
                              'basis run used continuous_v1 basis; research features are the frozen legacy hourly ones',
                              'research-host tape shifted to frozen Tokyo clock, not actual Tokyo receipt clock',
                              'unresolved/non-binary outcomes remain explicit exclusions',
                              'no live fills attached to counterfactual orders'])
    write(out / 'scope.json', scope)
    sources = {}
    for p in raw:
        dest = out / 'sources' / p.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(p)
        sources[str(p)] = dict(bytes=p.stat().st_size, sha256=sha(p))
    for p in (root / 'predictfun/research').glob('*.py'):
        sources[str(p)] = dict(bytes=p.stat().st_size, sha256=sha(p))
    for rel in ('step5/model_c2_y1000.txt', 'step5/model_c3_y250.txt', 'step5/model_c3.json',
                'step4/minute_lines_t60k.json'):
        src = root / 'data/predict_fun/research' / rel
        dst = out / 'research' / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        sources[str(src)] = dict(bytes=src.stat().st_size, sha256=sha(src))
    for p in (root / 'predictfun/freeze/pm_ladder_v1').glob('*'):
        if p.is_file():
            sources[str(p)] = dict(bytes=p.stat().st_size, sha256=sha(p))
    write(out / 'sources.json', sources)
    print(json.dumps(dict(out=str(out), windows=len(windows), raw_files=len(raw), raw_bytes=sum(p.stat().st_size for p in raw))))


def filtered_payload(path, payload, mids):
    d = payload.get('data')
    if path.startswith('/v1/categories/'):
        if not isinstance(d, dict):
            return None
        markets = [m for m in d.get('markets', []) if m['id'] in mids]
        return {**payload, 'data': {**d, 'markets': markets}} if markets else None
    if path.startswith('/v1/orders/matches'):
        rows = [r for r in d or [] if (r.get('market') or {}).get('id') in mids]
        return {**payload, 'data': rows} if rows else None
    raise ValueError(f'unapproved endpoint {path}')


def step(a):
    out = Path(a.out)
    scope = json.load(open(out / 'scope.json'))
    root = Path(scope['research'])
    if a.day not in scope['days_consumed'] or a.module not in MODULES:
        raise ValueError('day or module outside declared scope')
    code = root / 'predictfun/research' / (a.module + '.py')
    hashes = json.load(open(out / 'sources.json'))
    # All research sources are fixed for this run, including imported helpers.
    for path, info in hashes.items():
        if '/predictfun/research/' in path and sha(path) != info['sha256']:
            raise ValueError(f'changed research dependency: {path}')
    if sha(root / 'predictfun/research/pf_step4_ladder_v3.py') != ENGINE_SHA:
        raise ValueError('frozen engine changed')
    sys.path.insert(0, str(root))
    from predictfun.research import pf_io as io
    original_hours, original_rest, original_books = io.hours_of, io.rest_rows, io.ws_books
    mids = {w['market'] for w in scope['windows']}
    allowed = set(scope['hours'])
    io.hours_of = lambda day, extra_after=0, extra_before=0: [
        h for h in original_hours(day, extra_after, extra_before) if h in allowed]
    io.ROOT = str(out / 'sources')
    io.OUT = str(out / 'research')
    io.PF_RUN = str(out / 'sources' / Path(io.PF_RUN).relative_to(root))
    io.REF_RUN = str(out / 'sources' / Path(io.REF_RUN).relative_to(root))
    io.BN_DIR = str(out / 'sources' / Path(io.BN_DIR).relative_to(root))

    def rest(hour, prefixes):
        if hour not in allowed:
            raise ValueError('REST hour outside scope')
        for recv, params, path, payload in original_rest(hour, prefixes):
            pl = filtered_payload(path, payload, mids)
            if pl is not None:
                yield recv, params, path, pl

    def books(hour):
        if hour not in allowed:
            raise ValueError('WS hour outside scope')
        for recv, d in original_books(hour):
            if d.get('marketId') in mids:
                yield recv, d
    io.rest_rows, io.ws_books = rest, books
    # No inherited experiment override may silently change the frozen run.
    for k in list(os.environ):
        if k.startswith('PF_'):
            del os.environ[k]
    os.environ.update(PF_RL_DDEC='2', PF_FORWARD='20261109')
    if a.module == 'pf_step5_features_rl' and a.mode == 'pk':
        os.environ['PF_PEND_KNOWN'] = 'frame'
    args = [a.day]
    if a.module == 'pf_step5_c3_preds':
        args = ['apply', a.day]
    if a.module == 'pf_preds_startknown':
        args = ['preds_t60', a.day]
    if a.module == 'pf_step4_ladder_v3':
        if a.mode not in ('J', 'X'):
            raise ValueError('simulation requires J or X')
        os.environ.update(PF_RL_LPLACE='40', PF_RL_LCANCEL='22', PF_RL_LPENDCANCEL='95',
                          PF_RL_LFILL='100', PF_TIMING='e2', PF_TXTIME='ws', PF_TXKEY='tk',
                          PF_PREDS='preds_t60k', PF_ARMS=ARM, PF_TAG='lw' + a.mode)
        if a.mode == 'J':
            os.environ['PF_TXJIT'] = 'frame'
        else:
            os.environ.update(PF_TXSHIFT='-50', PF_QACT='max')
        args = [a.day, 'tokyoreal', 'custom']
    sys.argv = [str(code), *args]
    runpy.run_module('predictfun.research.' + a.module, run_name='__main__')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    i = sub.add_parser('init')
    i.add_argument('--research', required=True)
    i.add_argument('--capture', required=True)
    i.add_argument('--out', required=True)
    s = sub.add_parser('step')
    s.add_argument('--out', required=True)
    s.add_argument('--day', required=True)
    s.add_argument('--module', required=True, choices=MODULES)
    s.add_argument('--mode', default='')
    v=sub.add_parser('verify')
    v.add_argument('--repo',required=True)
    v.add_argument('--capture',required=True)
    a = p.parse_args()
    if a.command=='verify':
        print(json.dumps(verify_artifacts(Path(a.repo),Path(a.capture))))
    elif a.command=='init': initialize(a)
    else: step(a)


if __name__ == '__main__':
    main()
