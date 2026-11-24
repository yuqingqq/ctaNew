"""Independent raw-log, feed-membership, time-window and state recount."""
import collections,copy,json,math,time
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.compute as pc,pyarrow.parquet as pq
import common as B
pa.set_cpu_count(1);pa.set_io_thread_count(1)
def equal(rows,expected,keys):
 actual={tuple(str(r[k]) for k in keys):r for r in rows};assert len(rows)==len(actual)==len(expected) and set(actual)==set(expected),'population'
 for k,e in expected.items():
  for name,v in e.items():
   x=actual[k][name]
   if isinstance(v,bool):ok=B.flag(x)==v
   elif isinstance(v,(float,np.floating)):ok=math.isclose(float(x),float(v),abs_tol=1e-6,rel_tol=1e-12)
   else:ok=str(x)==str(v)
   assert ok,(k,name,x,v)
def main():
 start=time.process_time();m=B.bound();rr=B.read(B.H/'outputs/orders.csv');raw=B.read(B.D/'decoded_events.csv');saved=json.loads((B.D/'decoded_orders.json').read_text());rsp=B.responses();ev={};signed={};ids=json.loads((B.R/'data/predict_fun/research/maker_trace_v1_20261122/econ/identities.json').read_text());ranks={r['addr'].lower():int(r['rank']) for r in ids['ex_ante']['top10_shares'] if int(r['rank'])<=3}
 for (tx,kind),rec in sorted(rsp.items()):
  if kind!='receipt':continue
  inp=rsp[(tx,'input')];assert rec['transactionHash'].lower()==inp['hash'].lower()==tx and rec['blockHash']==inp['blockHash'] and rec['status']=='0x1';os={o['hash']:o for o in B.S.decode(inp['input'])}
  for lg in rec['logs']:
   if lg['address'].lower()!=B.S.EX or not lg['topics'] or lg['topics'][0].lower()!=B.S.FILLED:continue
   h=lg['topics'][1].lower();v=os[h]['values'];addr='0x'+v[1].to_bytes(20,'big').hex()
   if addr not in ranks:continue
   assert B.S.order_hash(v)==h and int(lg['topics'][2],16)==v[1];d=bytes.fromhex(lg['data'][2:]);assert len(d)==160;aa,bb,mf,tf,fee=[int.from_bytes(d[k:k+32],'big') for k in range(0,160,32)];assert (aa,bb)==((0,v[4]) if v[10]==0 else (v[4],0));role='taker' if int(lg['topics'][3],16)==int(B.S.EX,16) else 'maker';assert role==os[h]['role'];key=(tx,str(int(lg['logIndex'],16)));assert key not in ev;ev[key]=dict(order_hash=h,tx=tx,log_index=key[1],role=role,making_wei=str(mf),shares_wei=str(tf if v[10]==0 else mf));record=dict(values=v,maker=addr,rank=ranks[addr]);assert h not in signed or signed[h]==record;signed[h]=record
 equal(raw,ev,('tx','log_index'));assert signed==saved
 mem=[];keys=pa.array(list(signed));cols=['day','mid','sid','tx','mk_idx','mk_hash','tk_hash','mk_signer','shares','status']
 for day in ('20261024','20261025','20261026','20261027'):
  for block in pq.ParquetFile(B.R/f'data/predict_fun/research/maker_trace_v1_20261122/legs_{day}.parquet').iter_batches(columns=cols,batch_size=65536):
   mask=pc.or_(pc.is_in(block.column('mk_hash'),value_set=keys),pc.is_in(block.column('tk_hash'),value_set=keys));mem.extend(block.filter(mask).to_pylist())
  B.guard(start,5)
 assert mem==json.loads((B.D/'members.json').read_text()),'membership source'
 by=collections.defaultdict(list);ee=collections.defaultdict(list)
 for x in mem:
  for role in ('mk_hash','tk_hash'):
   if x[role] in signed:by[x[role]].append((role,x))
 for e in ev.values():ee[e['order_hash']].append(e)
 ps={x['mk_hash']:x for x in B.read(B.C/'post_fill/orders.csv')};ns={x['order_hash']:x for x in B.read(B.T/'outputs/post_estimates.csv.gz')};old={x['order_hash']:x for x in B.read(B.OLD/'outputs/chains.csv')};life=collections.defaultdict(list)
 for day,mid in m['pairs']:
  z=pq.read_table(B.V/'life'/f'{day}_{mid}_fills.parquet',columns=['mk_hash','sid','mk_idx','shares','x_lo_hard_ms','hi_ms','status']);z=z.filter(pc.is_in(z['mk_hash'],value_set=keys))
  for x in z.to_pylist():life[x['mk_hash']].append(x)
 assert dict(life)==json.loads((B.D/'life.json').read_text())
 expected={}
 for h,b in signed.items():
  v=b['values'];ex=ee[h];le=by[h];roles={e['role'] for e in ex};pure=roles=={'maker'} and all(k=='mk_hash' for k,x in le);rt={e['tx'] for e in ex};lt={x['tx'].lower() for k,x in le};making=sum(int(e['making_wei']) for e in ex);shares=sum(int(e['shares_wei']) for e in ex);assert 0<making<=v[5];complete=bool(le) and rt==lt
  if complete and pure:complete=len(ex)==len(le) and abs(shares/1e18-sum(x['shares'] for k,x in le))<=1e-6
  status='incomplete_recorded_history' if not complete else ('mixed_or_taker_role' if not pure else ('exhausted' if making==v[5] else 'positive_remainder_not_a_pull'));p=ps.get(h);n=ns.get(h);r=old.get(h);ff=life[h];end='unknown';el=eh=''
  if status=='exhausted' and ff and all(0<=x['x_lo_hard_ms']<=x['hi_ms'] for x in ff):end='exhausted';el=max(int(x['x_lo_hard_ms']) for x in ff);eh=max(int(x['hi_ms']) for x in ff)
  elif status=='positive_remainder_not_a_pull' and r and r['terminal']=='conditional_unavailable':end='inherited_conditional_unavailability';el=int(r['end_lo_ms']);eh=int(r['end_hi_ms'])
  e=dict(order_hash=h,maker=b['maker'],rank=b['rank'],old88=h in old,original_shares=(v[6] if v[10]==0 else v[5])/1e18,original_making_wei=str(v[5]),recorded_making_wei=str(making),remaining_upper_wei=str(v[5]-making),recorded_shares=shares/1e18,raw_events=len(ex),raw_transactions=len(rt),feed_transactions=len(lt),complete_recorded_history=complete,roles='|'.join(sorted(roles)),budget_status=status,posting_census=h in ps,day=p['day'] if p else '',mid=p['mid'] if p else '',side=p['y_side'] if p else '',tick=int(p['tick']) if p else '',ws_ms=int(p['ws_ms']) if p else '',we_ms=int(p['we_ms']) if p else '',post_grade=n['grade'] if n else 'missing',post_lo_ms=int(n['effect_lo_ms']) if n and n['effect_lo_ms'] else '',post_hi_ms=int(n['effect_hi_ms']) if n and n['effect_hi_ms'] else '',end_kind=end,end_lo_ms=el,end_hi_ms=eh,successor_hash=p['next_hash'] if p else '',successor_relation='no_recorded_successor',successor_kind='unknown',successor_post_grade='missing',successor_post_lo_ms='',successor_post_hi_ms='',successor_original_shares='',private_pull_identified=False)
  if p:
   assert p['addr'].lower()==b['maker'];assert all(str(x['day'])==p['day'] and str(x['mid'])==p['mid'] for k,x in le)
   if pure and complete:assert len(ex)==len(ff)==int(p['n_fills']) and abs(sum(x['shares'] for x in ff)-shares/1e18)<=1e-6
  nh=e['successor_hash']
  if nh:
   nxt=ns.get(nh);valid=nxt and B.flag(nxt['estimate_enabled']) and nxt['grade'] in ('G2_bound_constrained','G4_scenario_only') and end!='unknown'
   if nxt:
    assert (nxt['maker'].lower(),nxt['day'],nxt['mid'],nxt['side'])==(b['maker'],e['day'],e['mid'],e['side']);a=int(nxt['effect_lo_ms']) if nxt['effect_lo_ms'] else '';z=int(nxt['effect_hi_ms']) if nxt['effect_hi_ms'] else '';e.update(successor_kind='refill' if int(nxt['tick'])==e['tick'] else 'reprice',successor_post_grade=nxt['grade'],successor_post_lo_ms=a,successor_post_hi_ms=z)
   e['successor_relation']='unavailable' if not valid else ('sequential_after_under_bounds' if a>eh else ('successor_before_ending_possible_overlap' if z<el else 'ordering_ambiguous'))
   if nh in signed:nv=signed[nh]['values'];e['successor_original_shares']=(nv[6] if nv[10]==0 else nv[5])/1e18
  expected[(h,)]=e
 equal(rr,expected,('order_hash',));B.guard(start,5)
 # Separate scalar context reconstruction; whole interval, never a selected midpoint.
 ip=pd.read_parquet(B.C/'take_logic_v2/panel.parquet');pp=pd.read_parquet(B.C/'posting_opportunity/rows.parquet');ig={(str(d),int(mid),int(rank)):g.set_index('k').to_dict('index') for (d,mid,rank),g in ip.groupby(['day','mid','rank'])};pg={(str(d),int(mid),side):g.set_index('k').to_dict('index') for (d,mid,side),g in pp.groupby(['day','mid','side'])};xc={}
 def cats(vals,f):return '|'.join(sorted({f(v) for v in vals})) if vals and all(math.isfinite(v) for v in vals) else 'unknown'
 for r in rr:
  for stage,lo,hi in [('post','post_lo_ms','post_hi_ms'),('ending','end_lo_ms','end_hi_ms'),('successor','successor_post_lo_ms','successor_post_hi_ms')]:
   for lag,z in ((0,'0'),(1000,'1')):
    valid=B.flag(r['posting_census']) and B.flag(r['complete_recorded_history']) and r['budget_status'] in ('exhausted','positive_remainder_not_a_pull') and bool(r[lo]) and bool(r[hi]);valid=valid and (r['post_grade'] in ('G2_bound_constrained','G4_scenario_only') if stage=='post' else (r['end_kind']!='unknown' if stage=='ending' else r['successor_post_grade'] in ('G2_bound_constrained','G4_scenario_only')));ix=[]
    if valid and int(r['ws_ms'])<=int(r[lo])<=int(r[hi])<int(r['we_ms']):ix=list(range((int(r[lo])-int(r['ws_ms']))//1000,(int(r[hi])-int(r['ws_ms']))//1000+1))
    e={k:r[k] for k in ('order_hash','old88','day','mid','rank','budget_status','end_kind','successor_relation','successor_kind')};e.update(stage=stage,lag_ms=lag,window_lo_ms=r[lo],window_hi_ms=r[hi],coverage='covered' if ix else 'unknown_or_refused',sampled_states=len(ix),btc_side='unknown',inventory_side='unknown',inventory_signed_min='',inventory_signed_max='',inventory_ledger_clean=False,book_status='unknown',spread='unknown',own_depth='unknown',time_left='unknown',trigger_identified=False)
    if ix:
     pk=pg[(r['day'],int(r['mid']),r['side'])];ik=ig[(r['day'],int(r['mid']),int(r['rank']))];s=1 if r['side']=='bid' else -1;sf='D'+str(lag);px=[pk[k] for k in ix];qx=[ik[k] for k in ix];assert all(p['t_ms']==i['t_ms']==int(r['ws_ms'])+k*1000 for k,p,i in zip(ix,px,qx));b=[p['btc_r_5s_bp_'+sf]*s for p in px];q=[i['q'+z]*s for i in qx];bk='|'.join(sorted({p['status_'+sf] for p in px}));e.update(btc_side=cats(b,lambda v:'with' if v>=1 else ('against' if v<=-1 else 'flat')),inventory_side=cats(q,lambda v:'adding' if v>1e-6 else ('offsetting' if v< -1e-6 else 'flat')),inventory_signed_min=min(q) if all(math.isfinite(x) for x in q) else '',inventory_signed_max=max(q) if all(math.isfinite(x) for x in q) else '',inventory_ledger_clean=all(i['clean'] for i in qx),book_status=bk,spread=cats([p['spread_ticks_'+sf] for p in px],lambda v:'wide' if v>=2 else 'tight') if bk=='ok' else 'unknown',own_depth=cats([p['depth_touch_own_'+sf] for p in px],lambda v:'thin' if v<=50 else ('deep' if v>200 else 'middle')) if bk=='ok' else 'unknown',time_left=cats([p['time_left_s_'+sf] for p in px],lambda v:'last30s' if v<=30 else ('last60s' if v<=60 else 'earlier')))
    xc[(r['order_hash'],stage,str(lag))]=e
 cx=B.read(B.H/'outputs/contexts.csv');equal(cx,xc,('order_hash','stage','lag_ms'));cells=collections.Counter(tuple(str(r[k]) for k in ('old88','budget_status','successor_relation','stage','lag_ms','coverage','btc_side','inventory_side','spread','own_depth')) for r in cx);actual=B.read(B.H/'outputs/state_cells.csv');assert len(actual)==len(cells)
 for r in actual:assert int(r['orders'])==cells[tuple(r[k] for k in ('old88','budget_status','successor_relation','stage','lag_ms','coverage','btc_side','inventory_side','spread','own_depth'))]
 res=json.loads((B.H/'census.json').read_text());assert res['orders']==len(rr) and res['new_orders']==sum(not B.flag(r['old88']) for r in rr);new=sum(not B.flag(r['old88']) and B.flag(r['complete_recorded_history']) and r['budget_status'] in ('exhausted','positive_remainder_not_a_pull') and r['post_grade'] in ('G2_bound_constrained','G4_scenario_only') for r in rr);assert res['new_usable_posting_chains']==new and res['annotation_gate']==(new>0) and res['nonposting_controls_observed']==0 and res['posting_propensity_gate'] is False
 for field,key in (('budget_counts','budget_status'),('end_counts','end_kind'),('successors','successor_relation')):assert res[field]==dict(collections.Counter(r[key] for r in rr))
 assert res['new_successors']==dict(collections.Counter(r['successor_relation'] for r in rr if not B.flag(r['old88'])))
 for p,h in res['hashes'].items():assert B.sha(B.R/p)==h
 mr=json.loads((B.H/'map_result.json').read_text());assert mr['physical_sha256']==B.sha(B.H/'outputs/orders.csv') and mr['rows']==len(cx)
 # Positive inputs are checked again after each known-bad copy.
 falsifiers={};oi=next(i for i,r in enumerate(rr) if r['budget_status']=='exhausted');ci=next(i for i,r in enumerate(cx) if r['coverage']=='covered')
 for name,table,ex,keys,index,field,val in [('budget',rr,expected,('order_hash',),oi,'recorded_making_wei','0'),('false_pull',rr,expected,('order_hash',),oi,'private_pull_identified','True'),('history',rr,expected,('order_hash',),oi,'complete_recorded_history','False'),('actor',rr,expected,('order_hash',),oi,'maker','wrong'),('successor',rr,expected,('order_hash',),oi,'successor_relation','invented'),('future_state',cx,xc,('order_hash','stage','lag_ms'),ci,'window_lo_ms','0'),('trigger',cx,xc,('order_hash','stage','lag_ms'),ci,'trigger_identified','True'),('btc',cx,xc,('order_hash','stage','lag_ms'),ci,'btc_side','invented')]:
  z=copy.deepcopy(table);z[index][field]=val
  try:equal(z,ex,keys);raise RuntimeError('falsifier escaped '+name)
  except AssertionError:falsifiers[name]='refused'
  equal(table,ex,keys)
 try:equal(rr[:-1],expected,('order_hash',));raise RuntimeError('omission escaped')
 except AssertionError:falsifiers['omitted_order']='refused'
 B.guard(start,5);B.dump(B.H/'checks.json',dict(pass_all=True,orders=len(rr),raw_events=len(ev),matched_feed_legs=len(mem),contexts=len(cx),falsifiers=falsifiers,cpu_seconds=time.process_time()-start));print(json.dumps(dict(pass_all=True,orders=len(rr),contexts=len(cx),falsifiers=len(falsifiers),cpu_seconds=time.process_time()-start)))
if __name__=='__main__':main()
