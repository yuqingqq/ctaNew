"""Preserve unrelated work; verify hashes, staging and charged resources."""
import hashlib,json,subprocess,importlib.util
from pathlib import Path
R=Path('/home/yuqing/ctaNew');H=Path(__file__).resolve().parent
def git(*args):return subprocess.check_output(['git','-C',str(R),*args])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
s=json.loads((H/'commit_scope.json').read_text());blocks=json.loads((H/'state_blocks.json').read_text());assert git('rev-parse','HEAD').decode().strip()==s['parent']
for name,h in s['preserved_dirty'].items():
 p=R/name
 if name=='predictfun/research/run_light.log':
  lines=p.read_bytes().splitlines(keepends=True);assert any(hashlib.sha256(b''.join(lines[:len(lines)-i])).hexdigest()==h for i in range(min(20,len(lines))))
 else:assert (sha(p) if p.exists() else None)==h,name
for name,z in s['owned_state'].items():assert sha(R/name)==z['post_work_sha'] and hashlib.sha256(git('show',':'+name)).hexdigest()==z['staged_sha']
for p,h in json.loads((H/'manifest.json').read_text())['files'].items():assert sha(R/p)==h,p
for p,h in json.loads((H/'reference_bindings.json').read_text()).items():assert sha(R/p)==h,p
for p,h in json.loads((H/'census.json').read_text())['hashes'].items():
 assert sha(R/p)==h,p
 if '/data/' in str(R/p):assert sha(H/'evidence'/Path(p).name)==h
assert json.loads((H/'checks.json').read_text())['pass_all'] and json.loads((H/'reference_checks.json').read_text())['pass_all']
mr=json.loads((H/'map_result.json').read_text());assert sha(H/'outputs/orders.csv')==mr['physical_sha256'];assert sha(H/'outputs/contexts.csv')==mr['contexts_sha256'];assert sha(H/'outputs/state_cells.csv')==mr['cells_sha256']
spec=importlib.util.spec_from_file_location('cpu',R/'predictfun/reviews/maker_trace_20261122/tools/cpu_ledger.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);rr=m.ledger();packet=sum(x['cpu_s'] for x in rr if x['tag'].startswith('mt_actpop_'));task=sum(x['cpu_s'] for x in rr)/3600+m.EXTRA_H
resource=dict(packet_cpu_s_before_finalizer=packet,packet_cap_coreh=.004,task_before_finalizer=task,line_before_finalizer=m.LINE_BEFORE+task,task_cap=2.,line_cap=5.30,standing_reserve=.05,finalizer_reserved_cpu_s=1.)
assert packet+1<=.004*3600 and task+.05+1/3600<=2. and m.LINE_BEFORE+task+.05+1/3600<=5.30
(H/'resource_final.json').write_text(json.dumps(resource,indent=2)+'\n');files=[p for p in sorted(H.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='SHA256SUMS'];assert all(p.stat().st_size<1024*1024 for p in files)
(H/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(H))+'\n' for p in files));files.append(H/'SHA256SUMS');git('add','--',*[str(p.relative_to(R)) for p in files]);assert all(p in blocks or p.startswith(str(H.relative_to(R))+'/') for p in git('diff','--cached','--name-only').decode().splitlines())
for p in files:assert git('show',':'+str(p.relative_to(R)))==p.read_bytes()
print(json.dumps(dict(preserved_work=True,only_owned_staging=True,frozen_inputs_and_sources=True,checks=True,resource=resource),indent=2))
