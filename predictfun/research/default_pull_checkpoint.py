"""Stdlib-only integrity checkpoints for completed default-pull replay days."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys

WORK=Path(__file__).resolve().parents[2]
ROOT=Path('/home/yuqing/ctaNew')
OUT=ROOT/'data/predict_fun/research/default_pull_20261111_v4'
REL=Path('predictfun/reviews/default_pull_20261111_v4')
DAYS=('20261026','20261027','20261031','20261101','20261102','20261103')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def need(path,digest):
    if sha(path)!=digest:raise ValueError('artifact changed: '+str(path))


def scope(group,day):
    if group not in ('eval','stress') or day not in DAYS:raise ValueError('checkpoint outside declared replay cohort')


def controls():
    need(__file__,sha(__file__))
    try:need(__file__,'0'*64)
    except ValueError:pass
    else:raise ValueError('changed-artifact refusal failed')
    scope('eval',DAYS[0])
    try:scope('eval','20261104')
    except ValueError:pass
    else:raise ValueError('protected-cohort refusal failed')


def copy_exact(source,target):
    data=source.read_bytes()
    if len(data)>1000000:raise ValueError('refuse large checkpoint file')
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if target.read_bytes()!=data:raise ValueError('refuse checkpoint overwrite: '+str(target))
    else:
        with target.open('xb') as f:f.write(data)


def run(group,day):
    controls();scope(group,day)
    declaration=json.loads((OUT/'declaration.json').read_text())
    q=OUT/'runs'/group/day
    c=json.loads((q/'checks.json').read_text());features=json.loads((q/'feature_check.json').read_text())
    summary=json.loads((q/'summary.json').read_text())
    if c['status']!='PASS' or features['status']!='PASS' or c['day']!=day or c['group']!=group:raise ValueError('failed/wrong run')
    need(q/'checks.json',features['checks_sha']);need(q/'independent_windows.parquet',c['independent_sha'])
    done=OUT/'logs'/f'{group}_{day}.done';digest,path=done.read_text().strip().split(maxsplit=1)
    log=OUT/'logs'/f'{group}_{day}.log'
    if Path(path)!=log:raise ValueError('wrong stage log')
    need(log,digest)
    if 'rc=0' not in log.read_text():raise ValueError('stage did not exit successfully')
    expected=set(declaration['modes'])-({'reference','default'} if group=='eval' else set())
    variants={x['mode']:x for x in c['variants']};rows={x['mode']:x for x in features['rows']}
    if set(variants)!=expected or len(variants)!=len(c['variants']) or set(rows)!=expected or len(rows)!=len(features['rows']):raise ValueError('incomplete/duplicated arm population')
    stems={}
    for arm in summary['env']['PF_ARMS'].split(';'):
        p,r,m,s=arm.split('|');modes=[k for k,v in declaration['modes'].items() if v and v in r.split('+')]
        if len(modes)>1:raise ValueError('ambiguous arm identity')
        mode=modes[0] if modes else 'reference'
        if mode in stems:raise ValueError('duplicate engine arm')
        stems[mode]=(f'{p}+{r}/{m}@{s}'.replace('/','__').replace('+','_').replace('@','_sz').replace(':','-')+'.parquet')
    if set(stems)!=expected:raise ValueError('engine/check arm mismatch')
    for mode,v in variants.items():
        if v['windows']!=288:raise ValueError('incomplete window population')
        need(q/'generations'/f'{mode}.parquet',v['generation_sha'])
        if rows[mode]['generation_sha']!=v['generation_sha']:raise ValueError('feature/action identity')
        for kind,h in v['hashes'].items():need(q/kind/stems[mode],h)
    rel=REL/'runs'/group/day
    checkpoint=WORK/rel/'checkpoint.json';created=not checkpoint.exists()
    identities={name:sha(q/name) for name in ('checks.json','feature_check.json','source.json','summary.json')}
    if created:
        result=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),status='COMPLETED_REPLAY_INTEGRITY_PASS',
            group=group,day=day,windows_per_arm=288,arms=sorted(expected),
            feature_decisions=sum(x['decisions'] for x in rows.values()),artifacts=identities,log_sha=digest,
            source_sha=sha(__file__),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip(),
            scope='Integrity of completed per-run checks and actual win/fill/request/generation artifacts. Numeric reconciliation was performed by the frozen run checker; complete comparisons and final independent model/action audit remain required.',
            controls='Correct artifact/cohort accepted; planted changed digest and protected date refused.')
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        with checkpoint.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    else:
        result=json.loads(checkpoint.read_text())
        if result['artifacts']!=identities or result['log_sha']!=digest or result['source_sha']!=sha(__file__):raise ValueError('existing checkpoint identity changed')
    for base in (WORK,ROOT):
        for name in identities:copy_exact(q/name,base/rel/name)
        if base!=WORK:copy_exact(checkpoint,base/rel/'checkpoint.json')
    print(json.dumps(dict(created=created,day=day,group=group,feature_decisions=result['feature_decisions'],relative_path=str(rel))),flush=True)


if __name__=='__main__':run(*sys.argv[1:])
