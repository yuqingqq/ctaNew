"""Recover normalized cells from the stopped run; recheck every cell independently.
No result from the stopped full run is accepted. The evidence cache is content
bound here before recovery, and every per-market inference is recomputed.
"""
import json,sys,time,resource,subprocess
import numpy as np
import pandas as pd
import build as B
import fast_v3 as F
H=B.HERE;DATA=F.DATA

def main():
 start=time.process_time();resource.setrlimit(resource.RLIMIT_CPU,(8,10));B.bind();cache=json.loads((H/'recovery_sources.json').read_text())
 for p,h in cache.items():B.need(B.sha(B.ROOT/p)==h,'recovery identity')
 F.install();normal=B.normalized_book;write=B.write;meta=json.loads((H/'inputs.json').read_text());rawframes={(x['day'],x['mid']):x['rows'] for x in meta['parquets'] if x['kind']=='frames'};cachepaths={p for p in cache if p.endswith('_normalized.csv.gz')}
 def cached_book(day,mid,ws,we,ticks):
  path=DATA/f'{day}_{mid}_normalized.csv.gz'
  if str(path.relative_to(B.ROOT)) not in cachepaths:return normal(day,mid,ws,we,ticks)
  df=pd.read_csv(path,float_precision='round_trip');B.need(set(zip(df.side,df.tick))==set(ticks),'cached cells');cells={}
  for (s,k),g in df.groupby(['side','tick'],sort=False):
   B.need(not g.fid.duplicated().any(),'cached frame duplicate');g=g.sort_values(['t_ms','fid']);cells[(s,int(k))]={dst:g[src].to_numpy(dtype) for dst,src,dtype in [('t','t_ms',np.int64),('fid','fid',np.int64),('d','displayed_shares',float),('own_usd','pending_own_usd',float),('opp_usd','pending_other_usd',float),('own','pending_own_shares',float),('opp','pending_other_shares',float),('cap','capacity_upper_shares',float),('pair','coherent_previous',bool),('recv_ns','recv_ns',np.int64)]}
  nf=int(df.fid.nunique());return cells,range(len(df)),dict(frames=nf,raw_frames=rawframes[(day,mid)],excluded_frame_count=rawframes[(day,mid)]-nf,exclusion_reason_census='not_rebuilt_from_cache',display_semantics='bracketed_not_identified',pending_allocation='aggregate_unattributed',recovered_from='stopped_mt_qty_full_normalized_cells')
 def reuse_write(path,rs,fields=None):
  if str(path.relative_to(B.ROOT)) in cachepaths:B.need(B.sha(path)==cache[str(path.relative_to(B.ROOT))],'cache changed');return
  write(path,rs,fields)
 B.normalized_book=cached_book;B.write=reuse_write;inputs=B.input_tables();parts=[json.loads((DATA/'pilot.json').read_text())];partdir=DATA/'recovered_parts';partdir.mkdir(exist_ok=False)
 for day,mid in meta['pairs'][1:]:
  p=B.clean(B.market(day,mid,inputs,DATA));B.dump(partdir/f'{day}_{mid}.json',p);parts.append(p);B.guard()
 out=H/'outputs';B.need(not out.exists(),'no prior full outputs');out.mkdir();names={'ledger':'order_ledger.csv','orders':'order_summary.csv','bounds':'chain_bounds.csv','relations':'relation_edges.csv.gz','book':'book_edges.csv.gz','joint':'joint_conflicts.csv'}
 for field,name in names.items():write(out/name,[r for p in parts for r in p[field]],['day','mid','fid','t_ms','side','tick','required_future_shares','capacity_upper','order_hashes'] if field=='joint' else None)
 B.dump(out/'normalization.json',[p['normalization'] for p in parts]);result=B.summarize(parts);result.update(as_of_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=B.ROOT,text=True).strip(),cpu_seconds=time.process_time()-start,hashes={p.name:B.sha(p) for p in sorted(out.iterdir())});B.dump(out/'result.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
