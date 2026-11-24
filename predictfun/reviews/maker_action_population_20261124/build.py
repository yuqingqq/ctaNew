"""Physical action census, no market-feature reads."""
import collections,json,math,time
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import common as B
pa.set_cpu_count(1);pa.set_io_thread_count(1)
def run():
 start=time.process_time();m=B.bound();assert not (B.H/'outputs/orders.csv').exists();rsp=B.responses();ids=json.loads((B.R/'data/predict_fun/research/maker_trace_v1_20261122/econ/identities.json').read_text());ranks={r['addr'].lower():int(r['rank']) for r in ids['ex_ante']['top10_shares'] if int(r['rank'])<=3};raw=[];seen=set();book={}
 for tx,kind in sorted(rsp):
  if kind!='input':continue
  inp,rec=rsp[(tx,'input')],rsp[(tx,'receipt')];assert inp['blockHash']==rec['blockHash'] and inp['blockNumber']==rec['blockNumber'] and rec['status']=='0x1';os={o['hash']:o for o in B.S.decode(inp['input'])}
  for lg in rec['logs']:
   if lg['address'].lower()!=B.S.EX or not lg['topics'] or lg['topics'][0].lower()!=B.S.FILLED:continue
   h=lg['topics'][1].lower();o=os[h];addr='0x'+o['values'][1].to_bytes(20,'big').hex()
   if addr not in ranks:continue
   key=(tx,int(lg['logIndex'],16));assert key not in seen;seen.add(key);mf,tf,fee=B.S.bind_event(o,lg);v=o['values'];q=tf if v[10]==0 else mf
   if h in book:assert book[h]['values']==v
   else:book[h]=dict(values=v,maker=addr,rank=ranks[addr])
   raw.append(dict(order_hash=h,tx=tx,log_index=key[1],role=o['role'],making_wei=str(mf),shares_wei=str(q)))
 B.dump(B.D/'decoded_orders.json',book);B.write(B.D/'decoded_events.csv',raw);B.guard(start)
 keys=pa.array(list(book));members=[]
 cols=['day','mid','sid','tx','mk_idx','mk_hash','tk_hash','mk_signer','shares','status']
 for day in ('20261024','20261025','20261026','20261027'):
  path=B.R/f'data/predict_fun/research/maker_trace_v1_20261122/legs_{day}.parquet'
  for batch in pq.ParquetFile(path).iter_batches(columns=cols,batch_size=65536):
   filt=pc.or_(pc.is_in(batch.column('mk_hash'),value_set=keys),pc.is_in(batch.column('tk_hash'),value_set=keys));members.extend(batch.filter(filt).to_pylist())
  B.guard(start)
 B.dump(B.D/'members.json',members)
 ps={r['mk_hash']:r for r in B.read(B.C/'post_fill/orders.csv')};ns={r['order_hash']:r for r in B.read(B.T/'outputs/post_estimates.csv.gz')};old={r['order_hash']:r for r in B.read(B.OLD/'outputs/chains.csv')};legs=collections.defaultdict(list);events=collections.defaultdict(list)
 for r in members:
  for name in ('mk_hash','tk_hash'):
   if r[name] in book:legs[r[name]].append((name,r))
 for r in raw:events[r['order_hash']].append(r)
 life={}
 for day,mid in m['pairs']:
  path=B.V/'life'/f'{day}_{mid}_fills.parquet';dd=pq.read_table(path,columns=['mk_hash','sid','mk_idx','shares','x_lo_hard_ms','hi_ms','status']);dd=dd.filter(pc.is_in(dd['mk_hash'],value_set=keys))
  for r in dd.to_pylist():life.setdefault(r['mk_hash'],[]).append(r)
 B.dump(B.D/'life.json',life);out=[]
 for h,b in sorted(book.items()):
  ee=events[h];ll=legs[h];v=b['values'];making=sum(int(e['making_wei']) for e in ee);shares=sum(int(e['shares_wei']) for e in ee);txs={r['tx'].lower() for _,r in ll};rawtx={e['tx'] for e in ee};roles={e['role'] for e in ee};pure=roles=={'maker'} and all(n=='mk_hash' for n,r in ll);complete=bool(ll) and txs==rawtx
  if pure and complete:complete=len(ee)==len(ll) and abs(shares/1e18-sum(float(r['shares']) for _,r in ll))<=1e-6
  for name,r in ll:
   if name=='mk_hash':assert r['mk_signer'].lower()==b['maker']
  assert 0<making<=v[5],'raw signed budget'
  status=B.budget(v[5],making,complete,roles if pure else {'mixed_or_taker'});p=ps.get(h);n=ns.get(h);rr=old.get(h);ff=life.get(h,[]);a=dict(order_hash=h,maker=b['maker'],rank=b['rank'],old88=h in old,original_shares=(v[6] if v[10]==0 else v[5])/1e18,original_making_wei=str(v[5]),recorded_making_wei=str(making),remaining_upper_wei=str(v[5]-making),recorded_shares=shares/1e18,raw_events=len(ee),raw_transactions=len(rawtx),feed_transactions=len(txs),complete_recorded_history=complete,roles='|'.join(sorted(roles)),budget_status=status,posting_census=h in ps,day=p['day'] if p else '',mid=p['mid'] if p else '',side=p['y_side'] if p else '',tick=int(p['tick']) if p else '',ws_ms=int(p['ws_ms']) if p else '',we_ms=int(p['we_ms']) if p else '',post_grade=n['grade'] if n else 'missing',post_lo_ms=int(n['effect_lo_ms']) if n and n['effect_lo_ms'] else '',post_hi_ms=int(n['effect_hi_ms']) if n and n['effect_hi_ms'] else '',end_kind='unknown',end_lo_ms='',end_hi_ms='',successor_hash=p['next_hash'] if p else '',successor_relation='no_recorded_successor',successor_kind='unknown',successor_post_grade='missing',successor_post_lo_ms='',successor_post_hi_ms='',successor_original_shares='',private_pull_identified=False)
  if p:
   assert b['maker']==p['addr'].lower();assert all(str(r['mid'])==p['mid'] and str(r['day'])==p['day'] for _,r in ll),'market identity'
   if pure and complete:assert len(ff)==int(p['n_fills'])==len(ee) and abs(sum(r['shares'] for r in ff)-shares/1e18)<=1e-6,'lifecycle count'
  if status=='exhausted' and ff and all(0<=int(r['x_lo_hard_ms'])<=int(r['hi_ms']) for r in ff):
   a.update(end_kind='exhausted',end_lo_ms=max(int(r['x_lo_hard_ms']) for r in ff),end_hi_ms=max(int(r['hi_ms']) for r in ff))
  elif status=='positive_remainder_not_a_pull' and rr and rr['terminal']=='conditional_unavailable':a.update(end_kind='inherited_conditional_unavailability',end_lo_ms=int(rr['end_lo_ms']),end_hi_ms=int(rr['end_hi_ms']))
  nh=a['successor_hash']
  if nh:
   np_=ns.get(nh);valid=np_ is not None and B.flag(np_['estimate_enabled']) and np_['grade'] in ('G2_bound_constrained','G4_scenario_only') and a['end_kind']!='unknown'
   if np_:
    assert (np_['maker'].lower(),np_['day'],np_['mid'],np_['side'])==(b['maker'],a['day'],a['mid'],a['side']),'successor identity';a.update(successor_kind='refill' if int(np_['tick'])==a['tick'] else 'reprice',successor_post_grade=np_['grade'],successor_post_lo_ms=int(np_['effect_lo_ms']) if np_['effect_lo_ms'] else '',successor_post_hi_ms=int(np_['effect_hi_ms']) if np_['effect_hi_ms'] else '')
   a['successor_relation']=B.relation(a['end_lo_ms'],a['end_hi_ms'],a['successor_post_lo_ms'],a['successor_post_hi_ms'],valid)
   if nh in book:a['successor_original_shares']=(book[nh]['values'][6] if book[nh]['values'][10]==0 else book[nh]['values'][5])/1e18
  out.append(a)
 B.write(B.H/'outputs/orders.csv',out);B.guard(start)
 newusable=sum(not r['old88'] and r['complete_recorded_history'] and r['budget_status'] in ('exhausted','positive_remainder_not_a_pull') and r['post_grade'] in ('G2_bound_constrained','G4_scenario_only') for r in out)
 res=dict(as_of_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),orders=len(out),new_orders=sum(not r['old88'] for r in out),new_usable_posting_chains=newusable,annotation_gate=newusable>0,nonposting_controls_observed=0,posting_propensity_gate=False,budget_counts=dict(collections.Counter(r['budget_status'] for r in out)),end_counts=dict(collections.Counter(r['end_kind'] for r in out)),successors=dict(collections.Counter(r['successor_relation'] for r in out)),new_successors=dict(collections.Counter(r['successor_relation'] for r in out if not r['old88'])),hashes={str(p.relative_to(B.R)):B.sha(p) for p in (B.D/'decoded_orders.json',B.D/'decoded_events.csv',B.D/'members.json',B.D/'life.json',B.H/'outputs/orders.csv')},cpu_seconds=time.process_time()-start)
 B.dump(B.H/'census.json',res);print(json.dumps(res,indent=2))
if __name__=='__main__':run()
