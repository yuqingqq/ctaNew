"""Declare, bound and replay the original native inputs before soft-pull research."""
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from . import pf_fill_clock as a

NATIVE=Path('/tmp/pf-native-guard-20261109')
NATIVE_COMMIT='366d4192af219eef568ae17c530e4303e6f4ca45'
DEST=a.ROOT/'data/predict_fun/research/native_guard_20261109'
REVIEW=a.HERE.parent/'reviews/native_guard_20261109'
CAP=a.ROOT.parent/'pmmm-data/tokyo-live-data-20261109/tokyo-live-20261109'
RUNS={'v6':'1794108465172','basis':'1794147344737'}
TARGET=DEST/'target'
CLOCK=re.compile(rb'"(?:known_ms|now_ms)":([0-9]+(?:\.[0-9]+)?)')
OUT_CLOCK=re.compile(rb'"t":([0-9]+(?:\.[0-9]+)?)')
SOURCES=['pf_native_guard.py','native_guard_evidence.rs','tests/test_native_guard.py']


def prefix_time(line,kind):
    if kind=='inputs' and line.startswith(b'{"OpenWindow":'):return None
    m=(CLOCK if kind=='inputs' else OUT_CLOCK).search(line[:512])
    if not m:raise ValueError('unknown timestamp envelope')
    return float(m[1])


def bound(src,dst,kind,cutoff=a.END_MS):
    h=hashlib.sha256();n=0;last=None;stopped=False
    with gzip.open(src,'rb') as reader,gzip.open(dst,'xb',compresslevel=1) as writer:
        if kind=='inputs':
            header=next(reader);json.loads(header);writer.write(header);h.update(header)
        for line in reader:
            t=prefix_time(line,kind)
            if t is not None and t>=cutoff:stopped=True;break
            if t is not None:last=t
            writer.write(line);h.update(line);n+=1
    return dict(prefix_sha256=h.hexdigest(),records=n,last_time=last,stopped_before_future_payload=stopped,
        bounded_gzip_sha256=a.sha(dst),cutoff_ms=cutoff)


def check():
    d=json.loads((DEST/'declaration.json').read_text())
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=NATIVE,text=True).strip()!=NATIVE_COMMIT:raise ValueError('native commit differs')
    if subprocess.check_output(['git','diff','--name-only'],cwd=NATIVE,text=True).strip():raise ValueError('native tracked source changed')
    for p,h in d['sources'].items():
        if a.sha(a.HERE/p)!=h:raise ValueError('instrument source changed')
    for p,h in d['inputs'].items():
        if a.sha(p)!=h:raise ValueError('declared artifact changed')
    return d


def declare():
    if DEST.exists():raise ValueError('already exists')
    (DEST/'logs').mkdir(parents=True)
    scope=json.loads((a.RP/'scope.json').read_text())
    if scope['days_consumed']!=['20261108'] or len(scope['windows'])!=220:raise ValueError('wrong cohort')
    inputs=[a.RP/'scope.json',a.ROOT/'data/predict_fun/research/quote_capture_20261109/decisions.parquet']
    headers={}
    for run,rid in RUNS.items():
        with gzip.open(CAP/run/f'run-{rid}'/'inputs.jsonl.gz','rb') as f:h=json.loads(next(f))
        if h['protocol']!='PFMM_LADDER_JOURNAL_V3':raise ValueError('protocol differs')
        for name in ['params','guard_bid','guard_ask']:
            x=h['ladder'][name];p=(NATIVE/x['path']).resolve();p.relative_to(NATIVE)
            if a.sha(p)!=x['sha256']:raise ValueError('capture model/params digest differs')
            inputs.append(p)
        headers[run]=h
    d=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),
        scope='exact factual native input replay on220 consumed Nov8 windows; export decision states0-65s and books-1..65s; compare prior279059 first-minute decisions',
        native_commit=NATIVE_COMMIT,headers=headers,cutoff_ms=a.END_MS,
        boundary='copy bounded decompressed prefixes; examine timestamp envelope before future payload decode; permit only unprocessed scheduled-output tail after last input and all cohort endpoints',
        metrics=['all applied-input outputs exact','conditional native quote parity','all scope decision identities/values','exact soft-line intervention eligibility at fixed native state'],
        counterfactual='none in this stage; relaxed pure quotes differ only by pull/rung lines0.005/0.01/0.02/0.04 cents, holding size line and all other native inputs/config fixed; no changed OMS gets captured fills',
        null='deterministic replay identity check; no profit, predictive skill, selection or interval',
        controls='exact output positive/corrupt size refusal; basis/version mismatch refusal; bounded reader future/malformed/unknown timestamp controls',
        no_protected_dates_read=True,no_live_change=True,no_fill_clock_bypass=True,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.HERE,text=True).strip(),
        sources={p:a.sha(a.HERE/p) for p in SOURCES},inputs={str(p):a.sha(p) for p in inputs})
    a.write(DEST/'declaration.json',d);a.write(REVIEW/'declaration.json',d)
    print('DECLARED native identity replay before soft-exit evaluation')


def build():
    check();target=NATIVE/'crates/pf-engine/examples/native_guard_evidence.rs'
    if target.exists():
        if a.sha(target)!=a.sha(a.HERE/'native_guard_evidence.rs'):raise ValueError('different instrument already installed')
    else:shutil.copyfile(a.HERE/'native_guard_evidence.rs',target)
    common=['cargo','test','--locked','--offline','-p','pf-engine','--example','native_guard_evidence','--target-dir',str(TARGET),'-j','1']
    subprocess.run(common,cwd=NATIVE,check=True)
    common[1]='build';common.append('--release');subprocess.run(common,cwd=NATIVE,check=True)
    p=TARGET/'release/examples/native_guard_evidence'
    a.write(DEST/'build.json',dict(native_commit=NATIVE_COMMIT,instrument_sha256=a.sha(target),binary_sha256=a.sha(p)))


def prepare(run):
    check();path=DEST/run
    if path.exists():raise ValueError('prefix directory exists')
    path.mkdir();reads={}
    for kind in ['inputs','outputs']:
        reads[kind]=bound(CAP/run/f'run-{RUNS[run]}'/f'{kind}.jsonl.gz',path/f'{kind}.jsonl.gz',kind)
    a.write(path/'prefixes.json',reads);print(json.dumps(reads,indent=2))


def replay(run):
    check();b=json.loads((DEST/'build.json').read_text());binary=TARGET/'release/examples/native_guard_evidence'
    if a.sha(binary)!=b['binary_sha256']:raise ValueError('binary changed')
    for kind,row in json.loads((DEST/run/'prefixes.json').read_text()).items():
        if a.sha(DEST/run/f'{kind}.jsonl.gz')!=row['bounded_gzip_sha256']:raise ValueError('bounded input changed')
    subprocess.run([str(binary),str(NATIVE),str(DEST/run),str(DEST/run/'evidence'),str(a.RP/'scope.json'),run],check=True)


if __name__=='__main__':
    mode=sys.argv[1]
    if mode in ['declare','build']:{'declare':declare,'build':build}[mode]()
    else:{'prepare':prepare,'replay':replay}[mode](sys.argv[2])
