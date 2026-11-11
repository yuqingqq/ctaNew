"""Small stdlib-only durable progress recorder and numeric-slot observer."""
import datetime as dt
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT=Path('/home/yuqing/ctaNew')
WORK=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/predict_fun/research/default_pull_20261111'
TITLE='Quote-default / learned-pull research loop'
MARKER='<!-- DEFAULT_PULL_LOOP -->'
END='<!-- /DEFAULT_PULL_LOOP -->'

def atomic(p,text):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=p.parent,prefix='.'+p.name+'.')
    with os.fdopen(fd,'w') as f:f.write(text);f.flush();os.fsync(f.fileno())
    os.replace(name,p)

def busy():
    result=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:
            args=p.joinpath('cmdline').read_bytes().decode().split('\0')
            exe=Path(args[0]).name
            if not exe.startswith('python'):continue
            command=' '.join(args)
            if 'pf_default_pull_progress.py' in command:continue
            if 'predictfun.research.' in command or '/predictfun/research/pf_' in command or 'predictfun/research/pf_' in command:
                result.append(dict(pid=int(p.name),command=command[:240]))
        except (FileNotFoundError,PermissionError,ProcessLookupError,UnicodeDecodeError):continue
    return result

def update(stage,state,detail=''):
    now=dt.datetime.now(dt.timezone.utc).isoformat();path=OUT/'progress.json'
    previous=json.loads(path.read_text()) if path.exists() else dict(completed=[],stages={})
    event=dict(as_of=now,stage=stage,state=state,detail=detail,driver_pid=os.environ.get('DEFAULT_PULL_DRIVER_PID'))
    previous.update(event);previous['stages'][stage]=event
    if state=='complete' and stage not in previous['completed']:previous['completed'].append(stage)
    with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps(event)+'\n');f.flush();os.fsync(f.fileno())
    atomic(path,json.dumps(previous,indent=2)+'\n')
    lines=[f'# {TITLE}',f'As of {now}. **{state}: {stage}**. {detail}', '',
        'Completed prior test: own-order/calm study d475eccb, canonical comparison_v4.json; no promoted variant.',
        'Current branch: research/predict-default-pull-20261111. One numeric process; consumed dates only; no live changes.',
        'Plan: controls →8-window parity smoke →9 neutral build days →6 expanding model folds →6 primary replay days →primary report →6 stress replay days →final report.',
        '', '| Stage | Status | As of | Detail |', '|---|---|---|---|']
    for name,e in previous['stages'].items():lines.append(f"| {name} | {e['state']} | {e['as_of']} | {e['detail'].replace('|','/')} |")
    text='\n'.join(lines)+'\n'
    for base in (WORK,ROOT):
        atomic(base/'predictfun/handoffs/20261111_default_pull_progress.md',text)
        hand=base/'predictfun/HANDOFF.md';old=hand.read_text()
        block=f'{MARKER}\n\n## {now} — {TITLE}: {state}\n\n- Stage `{stage}`. {detail}\n- {len(previous["completed"])} stages complete. Tracked plan, receipts and failures: handoffs/20261111_default_pull_progress.md.\n- Prior own-order/calm test remains COMPLETE, no promotion; comparison_v4.json / d475eccb.\n- Branch research/predict-default-pull-20261111; one light numeric process; no live changes/protected-day reads.\n\n{END}\n'
        if MARKER in old:
            a=old.index(MARKER);b=old.index(END,a)+len(END);old=old[:a]+block.rstrip()+old[b:]
        else:
            first=old.find('\n');old=old[:first+1]+'\n'+block+old[first+1:]
        atomic(hand,old)
        status=base/'predictfun/STATUS.yml';old=status.read_text();a='# DEFAULT_PULL_LOOP';b='# /DEFAULT_PULL_LOOP'
        block=f'{a}\ndefault_pull_loop:\n  as_of: {json.dumps(now)}\n  state: {json.dumps(state)}\n  stage: {json.dumps(stage)}\n  completed_stages: {len(previous["completed"])}\n  detail: {json.dumps(detail)}\n  progress: handoffs/20261111_default_pull_progress.md\n  branch: research/predict-default-pull-20261111\n{b}\n'
        if a in old:
            x=old.index(a);y=old.index(b,x)+len(b);old=old[:x]+block.rstrip()+old[y:]
        else:old=old.rstrip()+'\n\n'+block
        atomic(status,old)
    print(json.dumps(event),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='slot':
        b=busy();print(json.dumps(b));sys.exit(bool(b))
    update(*sys.argv[1:])
