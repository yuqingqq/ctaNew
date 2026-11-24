"""Close resource/artifact receipt after the publisher log closes."""
import hashlib,importlib.util,json,subprocess,csv,gzip,statistics,collections
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2]
def git(*args):return subprocess.check_output(['git','-C',str(R),*args])
def sha(b):return hashlib.sha256(b).hexdigest()
spec=importlib.util.spec_from_file_location('cpu',R/'predictfun/reviews/maker_trace_20261122/tools/cpu_ledger.py');cpu=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpu);allrows=cpu.ledger();rows=[r for r in allrows if r['tag'].startswith('mt_age_')];used=sum(r['cpu_s'] for r in rows);task=sum(r['cpu_s'] for r in allrows)/3600+cpu.EXTRA_H;allowance=.25
assert used+allowance<=2.88 and task+allowance/3600+.05<=2 and cpu.LINE_BEFORE+task+allowance/3600+.05<=5.3
scope=json.loads((H/'commit_scope.json').read_text());blocks=json.loads((H/'state_blocks.json').read_text());assert git('rev-parse','HEAD').decode().strip()==scope['parent']
for name,v in scope['owned_state'].items():
 work=(R/name).read_bytes();staged=git('show',':'+name);assert sha(work.replace(blocks[name].encode(),b'',1))==v['preexisting_work_sha'];assert sha(staged)==v['staged_sha']
for name,digest in scope['preserved_dirty'].items():
 if name=='predictfun/research/run_light.log':continue
 f=R/name;assert (sha(f.read_bytes()) if f.exists() else None)==digest,name
for name,digest in json.loads((H/'manifest.json').read_text())['files'].items():assert sha((R/name).read_bytes())==digest,name
def read(p):
 with (gzip.open if str(p).endswith('.gz') else open)(p,'rt') as f:return list(csv.DictReader(f))
op=read(H/'orders.csv.gz');fp=read(H/'fills.csv.gz');out=json.loads((H/'readings.json').read_text());good=[r for r in fp if r['good_fill']=='True' and r['timing_status']=='ok'];examples=read(H/'good_fill_intervals.csv');assert len(good)==len(examples)==out['good_fills']==19
key=lambda r:(r['day'],r['mid'],r['sid'],r['mk_idx'],r['order_hash'])
assert {key(r) for r in examples}=={key(r) for r in good}
lookup={key(r):r for r in good}
for r in examples:
 p=lookup[key(r)];assert r['age_min_s']==p['age_min_s'] and r['age_max_s']==p['age_max_s'] and r['grade']==p['grade']
 for h in (10,60):assert abs(float(r[f'gross{h}_c_share'])-100*float(p[f'gross{h}_usd'])/float(p['q_valid']))<1e-9
assert out['good_days']==dict(collections.Counter(r['day'] for r in good));assert out['good_orders']==len({r['order_hash'] for r in good});assert out['all_good_min_lt5']==all(float(r['age_min_s'])<5 for r in good)
assert out['good_min_gt0']==sum(float(r['age_min_s'])>0 for r in good)==1
for t in (1,5):assert out[f'upper_le{t}']==sum(float(r['age_max_s'])<=t for r in good)
assert out['all_comparisons_insufficient']==all(r['verdict']=='insufficient' for r in json.loads((H/'matched.json').read_text()))
for r in out['order_table']:
 g=[x for x in op if x['grade']==r['grade'] and x['timing_status']=='ok'];e=[x for x in g if x['economics_status']=='counted_fills_available'];pos=[x for x in e if x['good_order']=='True'];ages=[float(x['first_age_max_s']) for x in pos];assert (len(g),len(e),len(pos))==(r['orders'],r['valued_orders'],r['good_aggregate_orders']);assert statistics.median(ages)==r['median_age_upper_s'];assert r['first_age_lower_all_zero']==all(float(x['first_age_min_s'])==0 for x in pos)
 for t in (1,5,10,30):assert r[f'upper_le{t}']==sum(x<=t for x in ages)
 assert r['upper_gt30']==sum(x>30 for x in ages)
(H/'resources.json').write_text(json.dumps(dict(runs=rows,measured_before_finalizer_seconds=used,finalizer_conservative_seconds=allowance,packet_upper_coreh=(used+allowance)/3600,packet_cap_coreh=.0008,task_upper_coreh=task+allowance/3600,line_upper_coreh=cpu.LINE_BEFORE+task+allowance/3600,standing_reserve_coreh=.05,task_cap_coreh=2,line_cap_coreh=5.3,within_caps=True),indent=2)+'\n')
paths=[f for f in sorted(H.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.name not in ['SHA256SUMS','finalize.log']]
assert all(f.stat().st_size<1024*1024 for f in paths)
(H/'SHA256SUMS').write_text(''.join(sha(f.read_bytes())+'  '+str(f.relative_to(H))+'\n' for f in paths))
git('add','--',*[str(f.relative_to(R)) for f in paths],str((H/'SHA256SUMS').relative_to(R)));print('resources and evidence verified; packet staged, no push')
print(git('diff','--cached','--stat').decode())
