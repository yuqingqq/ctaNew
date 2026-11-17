#!/usr/bin/env python3
"""Queue one review in an existing Codex thread when relevant commits arrive.

No agent or research job is started by this script itself. `codex queue --resume`
requests a turn in the user's existing review thread. A durable pending record
prevents duplicates until that review acknowledges its exact commit and report.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile
import uuid

DEFAULT_STATE = Path.home()/'.local/state/predictfun-commit-review/state.json'
REVIEW_TRAILER = 'Predictfun-Review-Only: true'
STATE_ONLY = {'predictfun/HANDOFF.md', 'predictfun/FINDINGS.md',
              'predictfun/STATUS.yml', 'predictfun/research/run_light.log'}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def save(path, state):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix='.review-state-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as out:
            json.dump(state, out, indent=2); out.write('\n'); out.flush(); os.fsync(out.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


@contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def relevant(repo, commit):
    message = git(repo, 'show', '-s', '--format=%B', commit)
    if REVIEW_TRAILER in message.splitlines():
        return False
    parents = git(repo, 'rev-list', '--parents', '-n', '1', commit).split()[1:]
    names = (git(repo, 'diff', '--name-only', parents[0], commit) if parents else
             git(repo, 'show', '--format=', '--name-only', commit)).splitlines()
    return any(p.startswith('predictfun/') and p not in STATE_ONLY
               and not p.startswith(('predictfun/reviews/automatic/', 'predictfun/automation/'))
               for p in names)


def initialize(path, repo, branch, thread, codex, baseline):
    repo = str(Path(repo).resolve()); codex = str(Path(codex).resolve())
    config = dict(repo=repo, branch=branch, thread=thread, codex=codex)
    with locked(path):
        if path.exists():
            state = json.loads(path.read_text())
            if state['config'] != config: raise ValueError('existing watcher configuration differs')
            return dict(status='already_initialized', cursor=state['cursor'])
        base = git(repo, 'rev-parse', '--verify', baseline+'^{commit}')
        tip = git(repo, 'rev-parse', '--verify', 'refs/heads/'+branch)
        subprocess.run(['git','-C',repo,'merge-base','--is-ancestor',base,tip],check=True)
        state = dict(version=1, config=config, created_at=stamp(), cursor=base,
                     last_reviewed=None, pending=None, completed=[])
        save(path,state)
        return dict(status='initialized', cursor=base)


def prompt(state, request, state_path):
    c=state['config']; ids='\n'.join(request['commits'])
    return f'''Scheduled Predict.fun review, explicitly authorized by the user. Another session owns fixes and development.
Request ID: {request['id']}
Repository: {c['repo']}; branch: {c['branch']}
Review range: {request['base']}..{request['target']}
Pinned target commit: {request['target']}
Relevant new commits:
{ids}

Review-only task:
1. Read AGENTS.md, CLAUDE.md reliability rules, predictfun/HANDOFF.md, STATUS.yml and FINDINGS.md, plus the latest prior review. Review all relevant commits in this range, not just their messages. Treat this request as queued work if another user task is active.
2. Check whether prior findings were actually fixed, inspect new issues, and verify result artifacts and labels/populations/timestamps/accounting. Distinguish synthetic tests, consumed development days and untouched validation. Never infer profitability from a passing implementation test.
3. Review the pinned committed source. Use git show or an isolated checkout/worktree when development files are dirty or HEAD advances; preserve all other-session changes and artifacts. Run only necessary bounded checks, through PF_SLOT=codex predictfun/research/run_light.sh with absolute script paths. If that slot is busy, defer/retry the check; do not use the other session's slot. Do not launch full training or tape experiments as part of a routine review.
4. Do not implement fixes or change strategy, models, simulator, controller or development tests. Independent reproduction scripts and review evidence may be added under predictfun/reviews/automatic/. Write a dated report there containing the FULL target SHA, findings ranked by severity with file/line evidence, tests and limitations, resolved versus open prior findings, and ONE concrete next priority for the development session. If clean, say so and identify the next experiment/gate.
5. Update only review-owned portions of predictfun/HANDOFF.md, STATUS.yml and FINDINGS.md, preserving existing unstaged changes. Commit only review artifacts/owned state hunks and add the exact git trailer `{REVIEW_TRAILER}` so the watcher does not review its own work. Never push, merge development branches, or edit another session's implementation. If a safe review commit is unavailable, retain the report and explain that limitation.
6. After the report and evidence are complete, acknowledge this exact request using:
/usr/bin/python3 {c['repo']}/predictfun/automation/commit_review_watch.py --state {state_path} ack --request-id {request['id']} --report /absolute/path/to/the/report.md
The report must be inside predictfun/reviews/automatic/ and contain {request['target']}. Do not acknowledge before finishing the review. This advances only to the reviewed target; newer commits will be picked up next time.
7. Reply in this review thread with the result, report link, and next priority. The development session will read the repository handoff and handle fixes. No external messaging.
'''


def dispatch(config, message):
    return subprocess.run([config['codex'],'queue','--thread',config['thread'],'--resume',
                           '--message',message],text=True,capture_output=True,timeout=45)


def check(path, send=dispatch):
    with locked(path):
        s=json.loads(path.read_text()); c=s['config']
        if s['pending']:
            return dict(status='pending',request_id=s['pending']['id'],delivery=s['pending']['delivery'])
        target=git(c['repo'],'rev-parse','--verify','refs/heads/'+c['branch'])
        ancestor=subprocess.run(['git','-C',c['repo'],'merge-base','--is-ancestor',s['cursor'],target],capture_output=True)
        if ancestor.returncode:
            raise ValueError('branch no longer descends from checkpoint; manual reconciliation required')
        commits=git(c['repo'],'rev-list','--reverse','--first-parent',s['cursor']+'..'+target).splitlines()
        selected=[x for x in commits if relevant(c['repo'],x)]
        if not selected:
            if s['cursor']!=target:
                s['cursor']=target; s['last_scan_at']=stamp(); save(path,s)
            return dict(status='no_new_relevant_commits',cursor=target)
        request=dict(id=str(uuid.uuid4()),base=s['cursor'],target=target,commits=selected,
                     queued_at=stamp(),delivery='in_progress')
        s['pending']=request
        # Persist BEFORE sending: a crash/timeout must not silently queue duplicates.
        save(path,s)
        message=prompt(s,request,path)
        try:
            result=send(c,message)
            request['queue_stdout']=result.stdout.strip()
            request['queue_stderr']=result.stderr.strip()
            request['delivery']='queued' if result.returncode==0 and not result.stderr.strip() else 'needs_attention'
            request['returncode']=result.returncode
        except Exception as e:
            request['delivery']='uncertain';request['error']=str(e)
        save(path,s)
        return dict(status=request['delivery'],request_id=request['id'],target=target,commits=len(selected))


def acknowledge(path, request_id, report):
    with locked(path):
        s=json.loads(path.read_text()); p=s['pending']
        if not p:
            if any(x['id']==request_id for x in s['completed']): return dict(status='already_acknowledged')
            raise ValueError('no matching pending review')
        if p['id']!=request_id: raise ValueError('pending review ID differs')
        report=Path(report).resolve(); allowed=Path(s['config']['repo'])/'predictfun/reviews/automatic'
        if not report.is_relative_to(allowed.resolve()): raise ValueError('report is outside the review directory')
        if p['target'] not in report.read_text(): raise ValueError('report does not identify the full reviewed target SHA')
        s['cursor']=p['target'];s['last_reviewed']=p['target']
        s['completed']=(s['completed']+[dict(id=p['id'],target=p['target'],report=str(report),completed_at=stamp())])[-100:]
        s['pending']=None;save(path,s)
        return dict(status='acknowledged',target=s['cursor'],report=str(report))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--state',type=Path,default=DEFAULT_STATE)
    sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('init');p.add_argument('--repo',required=True);p.add_argument('--branch',required=True)
    p.add_argument('--thread',required=True);p.add_argument('--codex',required=True);p.add_argument('--baseline',required=True)
    sub.add_parser('check');sub.add_parser('status')
    p=sub.add_parser('ack');p.add_argument('--request-id',required=True);p.add_argument('--report',required=True)
    a=ap.parse_args();path=a.state.expanduser().resolve()
    if a.cmd=='init':r=initialize(path,a.repo,a.branch,a.thread,a.codex,a.baseline)
    elif a.cmd=='check':r=check(path)
    elif a.cmd=='ack':r=acknowledge(path,a.request_id,a.report)
    else:r=json.loads(path.read_text())
    print(json.dumps(r,indent=2))
    if r.get('status') in ('needs_attention','uncertain'):raise SystemExit(1)


if __name__=='__main__':main()
