"""Read frozen physical labels, then annotate full feasible state intervals."""
import json,time,math,collections
import numpy as np,pandas as pd,pyarrow as pa
import common as B
pa.set_cpu_count(1);pa.set_io_thread_count(1)
def category(xs,fn):return '|'.join(sorted({fn(float(x)) for x in xs})) if len(xs) and np.isfinite(xs).all() else 'unknown'
def indices(a,z,ws,we,valid):
 return list(range((a-ws)//1000,(z-ws)//1000+1)) if valid and ws<=a<=z<we else []
def run():
 start=time.process_time();m=json.loads((B.H/'manifest.json').read_text());res=json.loads((B.H/'census.json').read_text());path=B.H/'outputs/orders.csv';before=B.sha(path);assert before==res['hashes'][str(path.relative_to(B.R))];assert not (B.H/'outputs/contexts.csv').exists()
 for p in (B.C/'take_logic_v2/panel.parquet',B.C/'posting_opportunity/rows.parquet'):assert B.sha(p)==m['files'][str(p.relative_to(B.R))]
 p=pd.read_parquet(B.C/'posting_opportunity/rows.parquet');i=pd.read_parquet(B.C/'take_logic_v2/panel.parquet');pg={(str(d),int(mm),s):g.sort_values('k').set_index('k') for (d,mm,s),g in p.groupby(['day','mid','side'])};ig={(str(d),int(mm),int(rank)):g.sort_values('k').set_index('k') for (d,mm,rank),g in i.groupby(['day','mid','rank'])};rr=B.read(path);cx=[]
 for r in rr:
  for stage,lo,hi in [('post','post_lo_ms','post_hi_ms'),('ending','end_lo_ms','end_hi_ms'),('successor','successor_post_lo_ms','successor_post_hi_ms')]:
   for lag,sf,iv in ((0,'D0','0'),(1000,'D1000','1')):
    valid=B.flag(r['posting_census']) and B.flag(r['complete_recorded_history']) and r['budget_status'] in ('exhausted','positive_remainder_not_a_pull') and bool(r[lo]) and bool(r[hi])
    valid=valid and (r['post_grade'] in ('G2_bound_constrained','G4_scenario_only') if stage=='post' else (r['end_kind']!='unknown' if stage=='ending' else r['successor_post_grade'] in ('G2_bound_constrained','G4_scenario_only')))
    ix=indices(int(r[lo]),int(r[hi]),int(r['ws_ms']),int(r['we_ms']),valid) if valid else [];btc=inv=book=spread=depth=tl='unknown';qlo=qhi='';clean=False
    if ix:
     pp=pg[(r['day'],int(r['mid']),r['side'])].loc[ix];ii=ig[(r['day'],int(r['mid']),int(r['rank']))].loc[ix];sg=1 if r['side']=='bid' else -1
     assert pp.index.is_unique and ii.index.is_unique;assert (pp.t_ms.to_numpy()==int(r['ws_ms'])+np.array(ix)*1000).all() and (ii.t_ms.to_numpy()==pp.t_ms.to_numpy()).all()
     z=pp['btc_r_5s_bp_'+sf].to_numpy()*sg;q=ii['q'+iv].to_numpy()*sg;btc=category(z,lambda v:'with' if v>=1 else ('against' if v<=-1 else 'flat'));inv=category(q,lambda v:'adding' if v>1e-6 else ('offsetting' if v< -1e-6 else 'flat'));book='|'.join(sorted(set(pp['status_'+sf])));spread=category(pp['spread_ticks_'+sf].to_numpy(),lambda v:'wide' if v>=2 else 'tight') if book=='ok' else 'unknown';depth=category(pp['depth_touch_own_'+sf].to_numpy(),lambda v:'thin' if v<=50 else ('deep' if v>200 else 'middle')) if book=='ok' else 'unknown';tl=category(pp['time_left_s_'+sf].to_numpy(),lambda v:'last30s' if v<=30 else ('last60s' if v<=60 else 'earlier'));qlo=float(q.min()) if np.isfinite(q).all() else '';qhi=float(q.max()) if np.isfinite(q).all() else '';clean=bool(ii.clean.all())
    cx.append(dict(order_hash=r['order_hash'],old88=r['old88'],day=r['day'],mid=r['mid'],rank=r['rank'],budget_status=r['budget_status'],end_kind=r['end_kind'],successor_relation=r['successor_relation'],successor_kind=r['successor_kind'],stage=stage,lag_ms=lag,window_lo_ms=r[lo],window_hi_ms=r[hi],coverage='covered' if ix else 'unknown_or_refused',sampled_states=len(ix),btc_side=btc,inventory_side=inv,inventory_signed_min=qlo,inventory_signed_max=qhi,inventory_ledger_clean=clean,book_status=book,spread=spread,own_depth=depth,time_left=tl,trigger_identified=False))
 B.write(B.H/'outputs/contexts.csv',cx);assert B.sha(path)==before,'physical labels changed'
 groups=collections.defaultdict(collections.Counter)
 for r in cx:groups[(r['old88'],r['budget_status'],r['successor_relation'],r['stage'],r['lag_ms'])][(r['coverage'],r['btc_side'],r['inventory_side'],r['spread'],r['own_depth'])]+=1
 cells=[dict(old88=k[0],budget_status=k[1],successor_relation=k[2],stage=k[3],lag_ms=k[4],coverage=v[0],btc_side=v[1],inventory_side=v[2],spread=v[3],own_depth=v[4],orders=n) for k,g in sorted(groups.items()) for v,n in sorted(g.items())];B.write(B.H/'outputs/state_cells.csv',cells);B.dump(B.H/'map_result.json',dict(rows=len(cx),physical_sha256=before,contexts_sha256=B.sha(B.H/'outputs/contexts.csv'),cells_sha256=B.sha(B.H/'outputs/state_cells.csv'),new_descriptive_only=True,matched_posting_test_run=False));B.guard(start,2);print(json.dumps(dict(rows=len(cx),cells=len(cells),physical_labels_unchanged=True)))
if __name__=='__main__':run()
