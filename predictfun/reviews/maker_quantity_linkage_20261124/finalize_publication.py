"""Check isolated staging, all evidence hashes and measured resource; refresh receipt manifest."""
import hashlib,json,re,subprocess,importlib.util
from pathlib import Path
ROOT=Path('/home/yuqing/ctaNew');H=Path(__file__).resolve().parent
scope=json.loads((H/'commit_scope.json').read_text());blocks=json.loads((H/'state_blocks.json').read_text())
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,h in scope['preserved_dirty'].items():
 p=ROOT/name
 if name=='predictfun/research/run_light.log':
  lines=p.read_bytes().splitlines(keepends=True)
  assert any(hashlib.sha256(b''.join(lines[:len(lines)-i])).hexdigest()==h for i in range(min(20,len(lines)))), 'run_light log prefix changed'
 else:assert (sha(p) if p.exists() else None)==h,name
for name,s in scope['owned_state'].items():
 assert sha(ROOT/name)==s['post_work_sha'];assert hashlib.sha256(git('show',':'+name)).hexdigest()==s['staged_sha']
assert all(p in blocks or p.startswith(str(H.relative_to(ROOT))+'/') for p in git('diff','--cached','--name-only').decode().splitlines())
for fn in ('sources.json','fast_sources.json','fast_sources_v3.json','recovery_sources.json'):
 for path,h in json.loads((H/fn).read_text()).items():assert sha(ROOT/path)==h,path
r=json.loads((H/'outputs/result.json').read_text())
for name,h in r['hashes'].items():assert sha(H/'outputs'/name)==h,name
for x in json.loads((H/'outputs/normalization.json').read_text()):assert sha(ROOT/x['normalized_file'])==x['normalized_sha256']
assert json.loads((H/'checks.json').read_text())['pass_all']
spec=importlib.util.spec_from_file_location('ledger',ROOT/'predictfun/reviews/maker_trace_20261122/tools/cpu_ledger.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);rr=m.ledger();packet=sum(r['cpu_s'] for r in rr if r['tag'].startswith('mt_qty_'));task=sum(r['cpu_s'] for r in rr)/3600+m.EXTRA_H
resource=dict(packet_charged_cpu_s=packet,packet_cap_coreh=.022,task_charged_coreh=task,task_cap=2.,line_max=m.LINE_BEFORE+task,line_cap=5.30,standing_reserve=.05,finalizer_extra_reserved_cpu_s=1.0)
assert packet+1<=.022*3600 and task+.05+1/3600<=2 and m.LINE_BEFORE+task+.05+1/3600<=5.3
(H/'resource_final.json').write_text(json.dumps(resource,indent=2)+'\n')
files=[p for p in sorted(H.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='SHA256SUMS'];assert all(p.stat().st_size<1024*1024 for p in files)
(H/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(H))+'\n' for p in files))
git('add','--',str((H/'resource_final.json').relative_to(ROOT)),str((H/'SHA256SUMS').relative_to(ROOT)))
print(json.dumps(dict(preserved_changes=True,only_owned_staging=True,source_and_artifact_hashes=True,resources=resource),indent=2))
