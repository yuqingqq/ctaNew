import gzip,json,resource,subprocess,sys,time
import launch
E=launch.E;H=E.H
def prepare():
 assert not (H/'resume_manifest.json').exists();m=E.bound();assert m['cases'][0][1]==3697902
 paths=sorted(E.OUT.glob('*.json.gz'));assert {p.name for p in paths}=={f'3697902_{a}.json.gz' for a in E.ARMS}
 E.dump(H/'resume_manifest.json',dict(original_manifest_sha256=E.sha(H/'manifest.json'),projection_sha256=E.sha(H/'projection.json'),
  sources={p:E.sha(H/p) for p in ['resume.py','resource_amendment.md']},parts={p.name:E.sha(p) for p in paths},remaining_cpu_cap=570))
def run():
 resource.setrlimit(resource.RLIMIT_CPU,(570,571));m=E.bound();rm=json.loads((H/'resume_manifest.json').read_text())
 for p,h in rm['sources'].items():assert E.sha(H/p)==h
 for p in ['resume.py','resource_amendment.md','resume_manifest.json']:
  assert subprocess.check_output(['git','show','HEAD:'+str((H/p).relative_to(E.R))],cwd=E.R)==(H/p).read_bytes()
 assert rm['original_manifest_sha256']==E.sha(H/'manifest.json') and rm['projection_sha256']==E.sha(H/'projection.json')
 assert {p.name for p in E.OUT.glob('*.json.gz')}==set(rm['parts'])
 for p,h in rm['parts'].items():assert E.sha(E.OUT/p)==h
 assert not (E.OUT/'summary.json').exists();start=time.process_time();edges=E.T.P.frozen_edges();rows=[];parity=[]
 for ix,(day,mid,part) in enumerate(m['cases']):
  old=E.X.records(day,part)[mid]['CURRENT_ASYNC_FAST10'];path,cache=E.X.get_path(day,mid,part,old)
  for arm in E.ARMS:
   fn=E.OUT/f'{mid}_{arm}.json.gz'
   if fn.name in rm['parts']:
    assert E.sha(fn)==rm['parts'][fn.name];rec=json.loads(gzip.decompress(fn.read_bytes()))
    assert rec['arm']==arm and rec['mid']==mid and rec['day']==day and rec['manifest_sha256']==E.sha(H/'manifest.json')
    assert rec['tape_id']==old['tape_id'] and rec['end']==old['end'] and rec['mark']==old['mark']
   else:
    assert not fn.exists();rec=E.run_one(path,cache,old,edges,arm,m['fees_bps']);rec.update(day=day,mid=mid,manifest_sha256=E.sha(H/'manifest.json'))
    fn.write_bytes(gzip.compress(json.dumps(rec,allow_nan=False).encode(),mtime=0))
   if mid in [v[1] for v in E.X.CASES] and arm in ['BASE','WIDE']:
    prev=json.loads(gzip.decompress((E.CAP.OUT/f"{mid}_{'BASE' if arm=='BASE' else 'WIDE_PASSIVE'}.json.gz").read_bytes()))
    fs=['fills','orders','requests','value','cash','position','commands','guard_log','grid','closed','mode_log']
    same={f:rec[f]==prev[f] for f in fs};assert all(same.values()),(mid,arm,same);parity.append(dict(mid=mid,arm=arm,same=same))
   assert {z['command_index'] for z in rec['exit_log']}=={i for i,c in enumerate(rec['commands']) if c[8] in ['pressure_pull','pressure_withhold','exit_override']}
   assert len({z['command_index'] for z in rec['exit_log']})==len(rec['exit_log'])
   rows.append(dict(day=day,mid=mid,arm=arm,metrics=rec['metrics'],sha256=E.sha(fn)))
  E.dump(E.OUT/'progress.json',dict(markets=ix+1,rows=rows,cpu_s=time.process_time()-start));print(day,mid,'done',round(time.process_time()-start,2),flush=True)
  assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1536*1024
 E.dump(E.OUT/'summary.json',dict(rows=rows,parity=parity,cpu_s=time.process_time()-start,status='COMPLETE',resume_manifest_sha256=E.sha(H/'resume_manifest.json'),pressure_membership='exact_all_commands'))
if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
