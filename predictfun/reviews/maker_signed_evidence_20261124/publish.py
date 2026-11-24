"""Stage only this packet and additive owned state; preserve other-session edits."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path('/home/yuqing/ctaNew');HERE=Path(__file__).resolve().parent
def git(*args,data=None):return subprocess.check_output(['git','-C',str(ROOT),*args],input=data)
def sha(b):return hashlib.sha256(b).hexdigest()
assert not git('diff','--cached','--name-only').strip(), 'staged work exists'
blocks=json.loads((HERE/'state_blocks.json').read_text())
scope={'parent':git('rev-parse','HEAD').decode().strip(),'preserved_dirty':{},'owned_state':{}}
for name in git('diff','--name-only','-z','HEAD').decode().split('\0'):
    if not name or name in blocks:continue
    p=ROOT/name;scope['preserved_dirty'][name]=sha(p.read_bytes()) if p.exists() else None
for name,block in blocks.items():
    p=ROOT/name;work=p.read_bytes();base=git('show','HEAD:'+name);b=block.encode()
    assert b'SIGNED_EVIDENCE_20261124_CODEX' not in work and b'signed_order_bridge_20261124_codex:' not in work
    def add(x):
        if name.endswith('HANDOFF.md'):
            anchor=b'<!-- /STANDING_BATCH_RULE_20261120_USER -->\n';assert x.count(anchor)==1
            return x.replace(anchor,anchor+b,1)
        if name.endswith('STATUS.yml'):
            first,rest=x.split(b'\n',1);return first+b'\n'+b+rest
        return b+x
    p.write_bytes(add(work));staged=add(base)
    blob=git('hash-object','-w','--stdin',data=staged).decode().strip()
    git('update-index','--add','--cacheinfo','100644,'+blob+','+name)
    scope['owned_state'][name]={'preexisting_work_sha':sha(work),'staged_sha':sha(staged),'post_work_sha':sha(p.read_bytes())}
for name,h in scope['preserved_dirty'].items():
    p=ROOT/name;assert (sha(p.read_bytes()) if p.exists() else None)==h, name
(HERE/'commit_scope.json').write_text(json.dumps(scope,indent=2)+'\n')
files=[]
for p in sorted(HERE.rglob('*')):
    if p.is_file() and '__pycache__' not in p.parts and p.name!='SHA256SUMS':
        assert p.stat().st_size<1024*1024, p
        files.append(p)
(HERE/'SHA256SUMS').write_text(''.join(sha(p.read_bytes())+'  '+str(p.relative_to(HERE))+'\n' for p in files))
files.append(HERE/'SHA256SUMS')
git('add','--',*[str(p.relative_to(ROOT)) for p in files])
print(git('diff','--cached','--stat').decode())
