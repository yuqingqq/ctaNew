"""Close resource/artifact receipt after the publisher log closes."""
import hashlib,importlib.util,json,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2]
def git(*args):return subprocess.check_output(['git','-C',str(R),*args])
def sha(b):return hashlib.sha256(b).hexdigest()
spec=importlib.util.spec_from_file_location('cpu',R/'predictfun/reviews/maker_trace_20261122/tools/cpu_ledger.py');cpu=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpu);allrows=cpu.ledger();rows=[r for r in allrows if r['tag'].startswith('mt_choices_')];used=sum(r['cpu_s'] for r in rows);task=sum(r['cpu_s'] for r in allrows)/3600+cpu.EXTRA_H;allowance=.1
assert used+allowance<=5.04 and task+allowance/3600+.05<=2 and cpu.LINE_BEFORE+task+allowance/3600+.05<=5.3
scope=json.loads((H/'commit_scope.json').read_text());blocks=json.loads((H/'state_blocks.json').read_text());assert git('rev-parse','HEAD').decode().strip()==scope['parent']
for name,v in scope['owned_state'].items():
 work=(R/name).read_bytes();staged=git('show',':'+name);assert sha(work.replace(blocks[name].encode(),b'',1))==v['preexisting_work_sha'];assert sha(staged)==v['staged_sha']
for name,digest in scope['preserved_dirty'].items():
 if name=='predictfun/research/run_light.log':continue
 f=R/name;assert (sha(f.read_bytes()) if f.exists() else None)==digest,name
for name,digest in json.loads((H/'manifest.json').read_text())['files'].items():assert sha((R/name).read_bytes())==digest,name
assert (H/'check_v2.py').read_bytes()==git('show','HEAD:'+str((H/'check_v2.py').relative_to(R)))
(H/'resources.json').write_text(json.dumps(dict(runs=rows,measured_before_finalizer_seconds=used,finalizer_conservative_seconds=allowance,packet_upper_coreh=(used+allowance)/3600,packet_cap_coreh=.0014,task_upper_coreh=task+allowance/3600,line_upper_coreh=cpu.LINE_BEFORE+task+allowance/3600,standing_reserve_coreh=.05,task_cap_coreh=2,line_cap_coreh=5.3,within_caps=True),indent=2)+'\n')
paths=[f for f in sorted(H.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.name not in ['SHA256SUMS','finalize.log']]
assert all(f.stat().st_size<1024*1024 for f in paths)
(H/'SHA256SUMS').write_text(''.join(sha(f.read_bytes())+'  '+str(f.relative_to(H))+'\n' for f in paths))
git('add','--',*[str(f.relative_to(R)) for f in paths],str((H/'SHA256SUMS').relative_to(R)));print('resources and evidence verified; packet staged, no push')
print(git('diff','--cached','--stat').decode())
