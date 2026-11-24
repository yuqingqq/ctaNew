"""Verify preserved work, bound evidence, lossless archive and final resource/staging."""
import gzip,hashlib,io,json,subprocess,importlib.util
from pathlib import Path
R=Path('/home/yuqing/ctaNew');H=Path(__file__).resolve().parent
def git(*args):return subprocess.check_output(['git','-C',str(R),*args])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
scope=json.loads((H/'commit_scope.json').read_text());blocks=json.loads((H/'state_blocks.json').read_text());assert git('rev-parse','HEAD').decode().strip()==scope['parent']
for name,h in scope['preserved_dirty'].items():
 p=R/name
 if name=='predictfun/research/run_light.log':
  lines=p.read_bytes().splitlines(keepends=True);assert any(hashlib.sha256(b''.join(lines[:len(lines)-i])).hexdigest()==h for i in range(min(20,len(lines)))),'run_light prefix changed'
 else:assert (sha(p) if p.exists() else None)==h,name
for name,s in scope['owned_state'].items():
 assert sha(R/name)==s['post_work_sha'];assert hashlib.sha256(git('show',':'+name)).hexdigest()==s['staged_sha']
for fn in ('sources.json','quote_bindings.json'):
 for path,h in json.loads((H/fn).read_text()).items():assert sha(R/path)==h,path
for path,h in json.loads((H/'inputs.json').read_text())['files'].items():assert sha(R/path)==h,path
for name,h in json.loads((H/'outputs/result.json').read_text())['hashes'].items():assert sha(H/'outputs'/name)==h,name
assert json.loads((H/'checks.json').read_text())['pass_all'] and json.loads((H/'quote_checks.json').read_text())['pass_all']
assert sha(H/'outputs/quote_relative.csv')==json.loads((H/'quote_result.json').read_text())['sha256']
archive=json.loads((H/'archive.json').read_text());parts=[];header=None
for part in archive['parts']:
 p=H/'outputs'/part['file'];assert sha(p)==part['sha256'];lines=gzip.decompress(p.read_bytes()).splitlines(keepends=True)
 if header is None:header=lines[0];parts.append(header)
 assert lines[0]==header and len(lines)-1==part['rows'];parts.extend(lines[1:])
raw=b''.join(parts);assert hashlib.sha256(raw).hexdigest()==archive['uncompressed_sha256'];buf=io.BytesIO()
with gzip.GzipFile(filename='',fileobj=buf,mode='wb',mtime=0,compresslevel=1) as z:z.write(raw)
assert hashlib.sha256(buf.getvalue()).hexdigest()==archive['original_sha256']
spec=importlib.util.spec_from_file_location('cpu',R/'predictfun/reviews/maker_trace_20261122/tools/cpu_ledger.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);ledger=m.ledger();packet=sum(x['cpu_s'] for x in ledger if x['tag'].startswith('mt_chain_'));task=sum(x['cpu_s'] for x in ledger)/3600+m.EXTRA_H
resource=dict(packet_charged_cpu_s_before_finalizer=packet,packet_cap_coreh=.006,task_charged_coreh_before_finalizer=task,task_cap=2.,line_max_before_finalizer=m.LINE_BEFORE+task,line_cap=5.30,standing_reserve=.05,finalizer_reserved_cpu_s=1.0)
assert packet+1<=.006*3600 and task+.05+1/3600<=2 and m.LINE_BEFORE+task+.05+1/3600<=5.30
(H/'resource_final.json').write_text(json.dumps(resource,indent=2)+'\n')
files=[p for p in sorted(H.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in ('SHA256SUMS','competing_events.csv.gz')];assert all(p.stat().st_size<1024*1024 for p in files)
(H/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(H))+'\n' for p in files));files.append(H/'SHA256SUMS')
git('add','--',*[str(p.relative_to(R)) for p in files]);assert all(p in blocks or p.startswith(str(H.relative_to(R))+'/') for p in git('diff','--cached','--name-only').decode().splitlines())
for p in files:assert git('show',':'+str(p.relative_to(R)))==p.read_bytes(),p
print(json.dumps(dict(preserved_work=True,only_owned_staging=True,frozen_hashes_and_inputs=True,independent_checks=True,lossless_archive=True,resources=resource),indent=2))
