"""Metadata-only watcher: publish completed replays; never start numeric research.

Invoke by file path (not -m) so the numeric-slot observer can distinguish it.
"""
import argparse
import datetime as dt
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from default_pull_checkpoint import WORK,ROOT,OUT,REL,DAYS,sha,need,copy_exact,run as checkpoint,controls as checkpoint_controls

BRANCH='research/predict-default-pull-20261111'
COMMON=['predictfun/HANDOFF.md','predictfun/STATUS.yml','predictfun/handoffs/20261111_default_pull_progress.md']
MODES={'reference','matched_guard','default','pressure','floor','market','queue','queue_cost'}


def driver_args(args):
    return args in (
        ['bash','predictfun/research/run_default_pull_v4.sh'],
        ['bash','predictfun/research/run_default_pull_two_slots.sh'],
    )


def report_shape(r,stage):
    scenarios={'X'} if stage=='primary' else {'X','stress'}
    expected={(s,m) for s in scenarios for m in MODES}
    if r['stage']!=stage or r['validation'] is not False or r['days']!=list(DAYS):raise ValueError('wrong report cohort')
    if {(x['scenario'],x['mode']) for x in r['table']}!=expected or len(r['table'])!=len(expected):raise ValueError('wrong report arms')
    if any(x['windows']!=1728 or x['marked_windows']+x['marked_missing']!=1728 for x in r['table']):raise ValueError('wrong report window population')


def controls():
    checkpoint_controls()
    if not driver_args(['bash','predictfun/research/run_default_pull_two_slots.sh']):raise ValueError('two-slot driver refused')
    if driver_args(['bash','unrelated.sh']):raise ValueError('unrelated driver accepted')
    r=dict(stage='primary',validation=False,days=list(DAYS),table=[dict(scenario='X',mode=m,windows=1728,marked_windows=1700,marked_missing=28) for m in sorted(MODES)])
    report_shape(r,'primary');r['table'][0]['windows']=1727
    try:report_shape(r,'primary')
    except ValueError:pass
    else:raise ValueError('bad report population refusal failed')


def git(*args):return subprocess.check_output(['git',*args],cwd=WORK,text=True).strip()


def publish(paths,message):
    if git('branch','--show-current')!=BRANCH:raise ValueError('wrong publication branch')
    paths=list(dict.fromkeys(COMMON+list(paths)))
    staged=git('diff','--cached','--name-only').splitlines()
    if any(not any(p==q or p.startswith(q.rstrip('/')+'/') for q in paths) for p in staged):raise ValueError('unrelated staged changes; refuse publication')
    if git('status','--porcelain','--',*paths):
        subprocess.run(['git','add','--',*paths],cwd=WORK,check=True)
        subprocess.run(['git','commit','-m',message],cwd=WORK,check=True)
    if git('rev-list','--count','@{u}..HEAD')!='0':subprocess.run(['git','push','origin',BRANCH],cwd=WORK,check=True)
    event=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),message=message,commit=git('rev-parse','HEAD'),watcher_sha=sha(__file__))
    with (OUT/'publication_log.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
    print('PUBLISHED',json.dumps(event),flush=True)


def done(stage):
    p=OUT/'logs'/f'{stage}.done'
    if not p.exists():return False
    h,target=p.read_text().strip().split(maxsplit=1)
    if Path(target)!=OUT/'logs'/f'{stage}.log':raise ValueError('wrong completed log')
    need(target,h);return True


def report(stage):
    p=OUT/f'comparison_{stage}.json';r=json.loads(p.read_text());report_shape(r,stage)
    need(OUT/'declaration.json',r['declaration_sha'])
    rel=REL/p.name;human=Path(f'predictfun/handoffs/20261111_default_pull_{stage}.md')
    for base in (WORK,ROOT):copy_exact(p,base/rel)
    copy_exact(WORK/human,ROOT/human)
    return [str(rel),str(human)]


def final_artifacts():
    if not done('completion_audit') or not done('comparison_figure'):raise ValueError('final verification stages incomplete')
    paths=report('final');r=json.loads((OUT/'completion_runtime_v4.json').read_text())
    if r['status']!='PASS':raise ValueError('final audit failed')
    for filename,key in [('comparison_final.json','comparison_sha'),('completion_audit.json','completion_audit_sha'),('runtime_parity.json','runtime_parity_sha'),('known_empty_controls.json','known_empty_controls_sha'),('declaration.json','declaration_sha')]:need(OUT/filename,r[key])
    figure=WORK/REL/'tradeoff_receipt.json';f=json.loads(figure.read_text())
    need(OUT/'completion_runtime_v4.json',f['audit_sha']);need(OUT/'comparison_final.json',f['comparison_sha'])
    for name,h in f['artifacts'].items():need(WORK/REL/name,h)
    for name in ('completion_audit.json','completion_runtime_v4.json'):
        for base in (WORK,ROOT):copy_exact(OUT/name,base/REL/name)
        paths.append(str(REL/name))
    for name in ('tradeoff_receipt.json','tradeoff.png','tradeoff.pdf'):
        copy_exact(WORK/REL/name,ROOT/REL/name);paths.append(str(REL/name))
    return paths


def live(pid):
    try:return driver_args(Path(f'/proc/{pid}/cmdline').read_bytes().decode().strip('\0').split('\0'))
    except (FileNotFoundError,ProcessLookupError):return False


def jobs():
    out=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            a=p.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
            if len(a)>2 and a[1:3]==['-m','predictfun.research.pf_default_pull_known_empty']:
                out.append(subprocess.check_output(['ps','-p',p.name,'-o','pid=,stat=,etime=,rss='],text=True).strip())
        except (FileNotFoundError,PermissionError,ProcessLookupError,UnicodeDecodeError,subprocess.CalledProcessError):pass
    return out


def run(pid):
    controls()
    with (OUT/'publication_watcher.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        handled=set();primary=False;last=0.
        while True:
            for group in ('eval','stress'):
                for day in DAYS:
                    key=group,day
                    if key in handled or not done(f'{group}_{day}'):continue
                    rel=REL/'runs'/group/day
                    if not (WORK/rel/'checkpoint.json').exists():checkpoint(group,day)
                    publish([str(rel)],f'Record verified {group} replay {day}');handled.add(key)
            if not primary and done('primary_report'):
                publish(report('primary'),'Record complete primary comparisons; stress results pending');primary=True
            p=json.loads((OUT/'progress.json').read_text());alive=live(pid)
            if p['stage']=='driver' and p['state']=='complete':
                if not done('final_report'):raise ValueError('driver complete without final report')
                publish(final_artifacts(),'Publish verified final replay artifacts; human review pending')
                print('WATCHER_COMPLETE_HUMAN_REVIEW_REQUIRED',flush=True);return
            if not alive or p['state']=='failed':
                publish([],f'Track stopped research stage {p["stage"]}')
                raise RuntimeError('research driver stopped before complete: '+json.dumps(p))
            if time.monotonic()-last>=45:
                print('WATCH',json.dumps(dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),stage=p['stage'],state=p['state'],driver_pid=pid,live_jobs=jobs(),replay_checkpoints=len(handled))),flush=True)
                last=time.monotonic()
            time.sleep(5)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--driver-pid',type=int,default=199624);parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.check_only:controls();print('WATCHER_CONTROLS_PASS',flush=True)
    else:run(args.driver_pid)
