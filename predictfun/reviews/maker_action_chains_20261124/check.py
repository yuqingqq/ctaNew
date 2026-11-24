"""Independent raw-event, window, successor and context verification; no builder import."""
import csv,gzip,json,hashlib,math,resource,time,copy
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np,pandas as pd,pyarrow as pa
pa.set_cpu_count(1);pa.set_io_thread_count(1)
H=Path(__file__).resolve().parent;R=H.parents[2];O=H/'outputs';Q=R/'predictfun/reviews/maker_quantity_linkage_20261124';T=R/'predictfun/reviews/maker_timing_estimates_20261124';S=R/'predictfun/reviews/maker_terminal_chain_20261124';V=R/'data/predict_fun/research/maker_trace_v2_20261123';CH=R/'predictfun/reviews/maker_trace_20261122/v2/chains'
def need(x,m):
 if not x:raise ValueError(m)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):
 with (gzip.open if str(p).endswith('.gz') else open)(p,'rt') as f:return list(csv.DictReader(f))
def flag(v):return v is True or v=='True'
def same(r,e,where):
 for k,v in e.items():
  z=r[k]
  if v is None:ok=z in ('',None)
  elif isinstance(v,(bool,np.bool_)):ok=flag(z)==bool(v)
  elif isinstance(v,(int,np.integer)):ok=z not in ('',None) and int(z)==int(v)
  elif isinstance(v,(float,np.floating)):ok=z not in ('',None) and math.isclose(float(z),float(v),rel_tol=1e-12,abs_tol=1e-7)
  else:ok=z==v
  need(ok,where+' '+k)
def check_rows(actual,expected,key,where):
 d={key(x):x for x in actual};need(len(d)==len(actual) and set(d)==set(expected),where+' population')
 for k,e in expected.items():same(d[k],e,where)
def cat(v,fn):
 return '|'.join(sorted({fn(float(x)) for x in v})) if len(v) and np.isfinite(v).all() else 'unknown'
def main():
 start=time.process_time();resource.setrlimit(resource.RLIMIT_CPU,(8,10))
 for file,key in [('inputs.json','files'),('sources.json',None)]:
  d=json.loads((H/file).read_text());d=d[key] if key else d
  for p,h in d.items():need(sha(R/p)==h,'input/source hash')
 result=json.loads((O/'result.json').read_text())
 for p,h in result['hashes'].items():need(sha(O/p)==h,'output hash')
 rows=read(O/'chains.csv');ops=read(O/'competing_events.csv.gz');cx=read(O/'contexts.csv.gz');base={r['order_hash']:r for r in read(Q/'outputs/chain_bounds.csv')};orders={r['order_hash']:r for r in read(Q/'outputs/order_summary.csv')};old={r['order_hash']:r for r in read(T/'outputs/signed_chain_estimates.csv')};posts={r['order_hash']:r for r in read(T/'outputs/post_estimates.csv.gz')};signed={r['order_hash']:r for r in read(S/'orders.csv')};by={r['order_hash']:r for r in rows};need(len(rows)==len(by)==88 and set(by)==set(orders),'chain population');expected_ops={};own={};expected_chains={};records={}
 pairs=json.loads((H/'inputs.json').read_text())['pairs']
 for d in sorted({d for d,m in pairs}):
  rr=pd.read_parquet(V/f'records_{d}.parquet',columns=['mid','sid','tk_signer','tk_y_side','status','n_makers'],filters=[('mid','in',[m for dd,m in pairs if dd==d])])
  records[d]={x['sid']:x for x in rr.to_dict('records')}
 for n in json.loads((Q/'outputs/normalization.json').read_text()):
  d,m=str(n['day']),int(n['mid']);life=pd.read_parquet(V/'life'/f'{d}_{m}_fills.parquet');norm=pd.read_csv(R/n['normalized_file'],usecols=['side','tick','t_ms','pending_own_shares','pending_other_shares','coherent_previous'],float_precision='round_trip');meta=json.loads((V/'life'/f'{d}_{m}.json').read_text());cells={(s,int(k)):g for (s,k),g in norm.groupby(['side','tick'],sort=False)};gs=defaultdict(list)
  for f in life.to_dict('records'):gs[f['sid']].append(f)
  for h,o in orders.items():
   if (o['day'],int(o['mid']))!=(d,m):continue
   b=base[h];actor=signed[h]['maker'].lower();side=o['side'];tick=int(o['tick']);ofs=life[life.mk_hash==h];olo=ofs.x_lo_hard_ms.to_numpy(np.int64);ohi=ofs.hi_ms.to_numpy(np.int64);oq=ofs.shares.to_numpy();own[h]=(olo,ohi,oq);ng=cells[(side,tick)];nt=ng.t_ms.to_numpy(np.int64);npending=ng.pending_own_shares.to_numpy()+ng.pending_other_shares.to_numpy();co=ng.coherent_previous.to_numpy(bool)
   for sid,fs in gs.items():
    prices=[int(f['y_ptick']) for f in fs if f['y_side']==side]
    if not prices or not any(p<=tick if side=='bid' else p>=tick for p in prices):continue
    lo=min(int(f['x_lo_hard_ms']) for f in fs);hi=max(int(f['hi_ms']) for f in fs)
    if hi<max(meta['ws_ms'],int(b['post_A_lo'])) or lo>meta['we_ms']:continue
    raw=records[d][sid];flags=[];sides={f['y_side'] for f in fs};s=next(iter(sides)) if len(sides)==1 else 'unknown'
    if len(fs)!=int(raw['n_makers']):flags.append('incomplete_record')
    if len(sides)!=1 or not sides<={'bid','ask'}:flags.append('mixed_side')
    if raw['tk_y_side']!=('ask' if s=='bid' else 'bid'):flags.append('taker_side')
    if raw['status']!='ok' or any(any(v in f['status'] for v in ('unplaceable','mismatch','malformed','side_mixed')) for f in fs):flags.append('source_flags')
    if lo<0 or lo>hi:flags.append('unbounded_clock')
    if any(not 0<int(f['y_ptick'])<100 for f in fs):flags.append('price_invalid')
    record_status='|'.join(sorted(flags)) if flags else 'ok';taker=raw['tk_signer'].lower();rel='through' if any(p<tick if side=='bid' else p>tick for p in prices) else 'same_price'
    ph='unbounded' if lo<0 or (olo<0).any() else ('pre_first' if hi<olo.min() else ('after_last' if lo>ohi.max() else ('between_fills' if (ohi<lo).any() and (olo>hi).any() else 'overlap_or_tie')))
    required=float(oq.sum()) if ph=='pre_first' else (float(oq[olo>hi].sum()) if ph=='between_fills' else (float(o['remaining_quote_shares_upper']) if ph=='after_last' else 0.));ps='not_checked';mx=None
    if flags:st='record_refused'
    elif h in {f['mk_hash'] for f in fs}:st='own_execution_record'
    elif taker==actor:st='self_trade_prevention_possible'
    elif ph in ('unbounded','overlap_or_tie'):st='clock_overlap_or_unknown'
    elif ph=='pre_first' and rel=='same_price':st='no_initial_time_priority'
    elif required<=.01:st='no_remaining_required'
    else:
     aa=np.flatnonzero(nt<=lo);bb=np.flatnonzero(nt>=hi)
     if not len(aa) or not len(bb):ps='unknown'
     else:
      a=int(aa[-1]);z=int(bb[0]);vals=npending[a:z+1]
      if lo-nt[a]>1000 or nt[z]-hi>1000 or not np.isfinite(vals).all() or not co[a+1:z+1].all():ps='unknown'
      else:mx=float(max(vals));ps='visible_insufficient' if required>mx+.01 else 'could_cover'
     st='conditional_available_priority_violation' if ps=='visible_insufficient' else ('reservation_can_explain' if ps=='could_cover' else 'pending_unknown')
    expected_ops[(h,sid)]=dict(day=d,mid=m,order_hash=h,sid=sid,side=side,tick=tick,relation=rel,phase=ph,lo_ms=lo,hi_ms=hi,taker=taker,record_status=record_status,required_shares=required,pending_status=ps,pending_max_shares=mx,witness_status=st,private_action_observed=False)
   relevant=[e for (hh,sid),e in expected_ops.items() if hh==h];ok=[e for e in relevant if e['witness_status']=='conditional_available_priority_violation'];eligible=b['bound_status']=='conditional_only' and flag(o['eligible_conditional_bounds']);pl=int(b['post_A_lo']);eh=int(b['end_hi_conditional']);el=int(b['old_end_lo']);pw=ew='';expiry=int(float(signed[h]['expiration_s'])*1000)
   for e in relevant:
    if e['witness_status']!='conditional_available_priority_violation' or not eligible:continue
    if e['phase']=='pre_first' and e['relation']=='through' and e['lo_ms']>pl:pl=e['lo_ms'];pw=e['sid']
    if e['phase']=='after_last' and o['group']=='conditional_withdrawal' and e['hi_ms']<min(eh,expiry):eh=e['hi_ms'];ew=e['sid']
   conflict=pl>int(b['post_hi_unchanged']) or eh<el
   if conflict:pl=int(b['post_A_lo']);eh=int(b['end_hi_conditional']);pw=ew=''
   if o['group']=='exhausted':el=int(olo.max());eh=int(ohi.max());ew=''
   terminal='exhausted' if o['group']=='exhausted' else ('conditional_unavailable' if o['group']=='conditional_withdrawal' and eligible else 'unresolved')
   expected_chains[h]=dict(day=d,mid=m,maker=actor,rank=int(o['rank']),side=side,tick=tick,ws_ms=int(meta['ws_ms']),we_ms=int(meta['we_ms']),original_shares=float(o['original_shares']),observed_shares=float(oq.sum()),remaining_upper_shares=float(o['remaining_quote_shares_upper']),maker_fill_records=len(ofs),first_fill_lo_ms=int(olo.min()),first_fill_hi_ms=int(ohi.min()),last_fill_lo_ms=int(olo.max()),last_fill_hi_ms=int(ohi.max()),group=o['group'],prior_post_grade=posts[h]['grade'],accepted_bounds=eligible,post_lo_ms=pl,post_hi_ms=int(b['post_hi_unchanged']),end_lo_ms=el,end_hi_ms=eh,post_witness=pw,end_witness=ew,bounds_conflict=conflict,interruption_records=len({e['sid'] for e in ok if e['phase']=='between_fills'}),old_post_lo_ms=int(b['post_A_lo']),old_end_hi_ms=int(b['end_hi_conditional']),terminal=terminal,expiry_ms=expiry,expiry_relation='after_interval' if expiry>eh else ('inside_interval' if expiry>=el else 'before_interval'),candidate_records=len(relevant),candidate_status_counts=json.dumps(dict(sorted(Counter(e['witness_status'] for e in relevant).items())),sort_keys=True),private_pull_observed=False)
 check_rows(ops,expected_ops,lambda r:(r['order_hash'],r['sid']),'events')
 # Successors use the exact predeclared identity and full intervals, never first-fill timing alone.
 for h,r in expected_chains.items():
  nh=old[h]['successor_hash'];n=posts.get(nh);usable=n is not None and (n['maker'].lower(),n['day'],int(n['mid']),n['side'])==(r['maker'],r['day'],r['mid'],r['side']);a=z=None;grade='missing';ns=None
  if nh and n is not None:need(usable,'successor identity')
  if usable:
   a=int(n['effect_lo_ms']) if n['effect_lo_ms'] else None;z=int(n['effect_hi_ms']) if n['effect_hi_ms'] else None;grade=n['grade']
   if nh in expected_chains and expected_chains[nh]['accepted_bounds']:a=expected_chains[nh]['post_lo_ms'];z=expected_chains[nh]['post_hi_ms']
   if nh in orders:ns=float(orders[nh]['original_shares'])
  valid=usable and flag(n['estimate_enabled']) and a is not None and z is not None and r['terminal']!='unresolved' and 0<=r['end_lo_ms']<=r['end_hi_ms'] and 0<=a<=z
  rel='no_recorded_successor' if not nh else ('refused_or_missing_window' if not valid else ('sequential_after_under_bounds' if a>r['end_hi_ms'] else ('posted_before_prior_end' if z<r['end_lo_ms'] else 'ordering_ambiguous')))
  kind=('refill' if int(n['tick'])==r['tick'] else 'reprice') if usable else 'unknown';direc=('same_price' if kind=='refill' else ('toward_spread' if (int(n['tick'])-r['tick'])*(1 if r['side']=='bid' else -1)>0 else 'away_from_spread')) if usable else 'unknown'
  cls=('exhaustion_then_'+kind if r['terminal']=='exhausted' else 'conditional_ending_then_'+kind) if rel=='sequential_after_under_bounds' else ('successor_already_posted_candidate_overlap' if rel=='posted_before_prior_end' else ('successor_ordering_unresolved' if rel=='ordering_ambiguous' else r['terminal']+'_no_timed_successor'))
  r.update(successor_hash=nh,successor_kind=kind,successor_direction=direc,successor_post_lo_ms=a,successor_post_hi_ms=z,successor_post_grade=grade,successor_original_shares=ns,successor_relation=rel,replacement_intent_identified=False,chain_class=cls)
 take=pd.read_parquet(CH/'take_logic_v2/events.parquet');ip=pd.read_parquet(CH/'take_logic_v2/panel.parquet');pp=pd.read_parquet(CH/'posting_opportunity/rows.parquet');pg={(str(d),int(m),s):g.sort_values('k') for (d,m,s),g in pp.groupby(['day','mid','side'],sort=False)};ig={(str(d),int(m),int(rank)):g.sort_values('k') for (d,m,rank),g in ip.groupby(['day','mid','rank'],sort=False)}
 expected_cx={}
 for h,r in expected_chains.items():
  ev=take[(take.day.astype(str)==r['day'])&(take.mid==r['mid'])&(take.addr.str.lower()==r['maker'])&(take.x_lo_hard_ms>r['last_fill_hi_ms'])&(take.hi_ms<=r['we_ms'])].sort_values(['hi_ms','sid']);e=ev.iloc[0] if len(ev) else None
  r.update(later_take_sid=str(e.sid) if e is not None else '',later_take_lo_ms=int(e.x_lo_hard_ms) if e is not None else None,later_take_hi_ms=int(e.hi_ms) if e is not None else None,later_take_inventory_status=str(e.o4_status) if e is not None else 'none_observed',later_take_direction_D0=str(e.dir_D0) if e is not None else 'none_observed',take_is_this_orders_exit=False)
  p=pg[(r['day'],r['mid'],r['side'])];i=ig[(r['day'],r['mid'],r['rank'])];sgn=1 if r['side']=='bid' else -1
  for stage,a,z in [('post',r['post_lo_ms'],r['post_hi_ms']),('ending',r['end_lo_ms'],r['end_hi_ms']),('successor_post',r['successor_post_lo_ms'],r['successor_post_hi_ms'])]:
   for lag in (0,1000):
    admissible=r['accepted_bounds'] if stage=='post' else (r['terminal']!='unresolved' if stage=='ending' else r['successor_post_grade'] in ('G2_bound_constrained','G4_scenario_only'));valid=admissible and a is not None and z is not None and r['ws_ms']<=a<=z<r['we_ms'];idx=list(range((a-r['ws_ms'])//1000,(z-r['ws_ms'])//1000+1)) if valid else [];sf='D'+str(lag);isf='0' if lag==0 else '1';qlo=qhi=None
    if idx:
     zq=i['q'+isf].to_numpy()[idx]*sgn;btc=p['btc_r_5s_bp_'+sf].to_numpy()[idx]*sgn;book='|'.join(sorted(set(map(str,p['status_'+sf].to_numpy()[idx]))));bc=cat(btc,lambda v:'with' if v>=1 else ('against' if v<=-1 else 'flat'));ic=cat(zq,lambda v:'adding' if v>1e-6 else ('offsetting' if v< -1e-6 else 'flat'));qlo=float(min(zq)) if np.isfinite(zq).all() else None;qhi=float(max(zq)) if np.isfinite(zq).all() else None;sp=cat(p['spread_ticks_'+sf].to_numpy()[idx],lambda v:'wide' if v>=2 else 'tight') if book=='ok' else 'unknown';dep=cat(p['depth_touch_own_'+sf].to_numpy()[idx],lambda v:'thin' if v<=50 else ('deep' if v>200 else 'middle')) if book=='ok' else 'unknown';tl=cat(p['time_left_s_'+sf].to_numpy()[idx],lambda v:'last30s' if v<=30 else ('last60s' if v<=60 else 'earlier'));il='clean' if i.clean.to_numpy()[idx].all() else 'conditional_or_unavailable'
    else:bc=ic=book=sp=dep=tl='unknown';il='conditional_or_unavailable'
    expected_cx[(h,stage,str(lag))]=dict(day=r['day'],mid=r['mid'],rank=r['rank'],order_hash=h,chain_class=r['chain_class'],stage=stage,lag_ms=lag,window_lo_ms=a,window_hi_ms=z,coverage='grid_interval_covered' if valid else 'outside_grid_missing_or_refused',asof_seconds=len(idx),btc_side=bc,inventory_side=ic,inventory_signed_min=qlo,inventory_signed_max=qhi,inventory_ledger_status=il,book_status=book,spread=sp,own_depth=dep,time_left=tl,client_inventory_observed=False,context_identifies_trigger=False)
 check_rows(rows,expected_chains,lambda x:x['order_hash'],'chains');check_rows(cx,expected_cx,lambda x:(x['order_hash'],x['stage'],x['lag_ms']),'contexts')
 cell=Counter((x['rank'],x['chain_class'],x['stage'],x['lag_ms'],x['coverage'],x['btc_side'],x['inventory_side'],x['time_left']) for x in cx);actual_cells=read(O/'condition_chains.csv');need(len(cell)==len(actual_cells),'cell membership')
 for x in actual_cells:need(int(x['orders'])==cell[tuple(x[k] for k in ('rank','chain_class','stage','lag_ms','coverage','btc_side','inventory_side','time_left'))],'cell count')
 es=list(expected_chains.values());count=lambda rs,f:dict(sorted(Counter(x[f] for x in rs).items()));expected_result=dict(orders=88,groups=count(es,'group'),chain_classes=count(es,'chain_class'),successors=count(es,'successor_relation'),candidate_records=len(ops),candidate_status=count(list(expected_ops.values()),'witness_status'),new_post_bounds=sum(x['post_lo_ms']>x['old_post_lo_ms'] for x in es),new_end_bounds=sum(bool(x['end_witness']) for x in es),orders_with_between_fill_interruption=sum(x['interruption_records']>0 for x in es),bound_conflicts=sum(x['bounds_conflict'] for x in es),exhausted_terminal_pulls=sum(x['terminal']!='exhausted' for x in es if x['group']=='exhausted'),private_pulls_identified=0,later_take_references=sum(bool(x['later_take_sid']) for x in es),distinct_later_takes=len({(x['day'],x['mid'],x['maker'],x['later_take_sid']) for x in es if x['later_take_sid']}),contexts=len(cx))
 for k,v in expected_result.items():need(result[k]==v,'summary '+k)
 falsifiers={}
 def bad(name,data,expected,key,field,value):
  x=copy.deepcopy(data);x[0][field]=value
  try:check_rows(x,expected,key,name)
  except ValueError as e:falsifiers[name]=str(e);return
  raise AssertionError('falsifier escaped')
 for name,field,val in [('false_pull','private_pull_observed','True'),('false_repost','successor_relation','sequential_automatic'),('false_time','end_hi_ms','0'),('wrong_size','original_shares','999999'),('wrong_actor','maker','wrong')]:bad(name,rows,expected_chains,lambda x:x['order_hash'],field,val)
 for name,field,val in [('wrong_pending','pending_max_shares','-1'),('wrong_phase','phase','invented')]:bad(name,ops,expected_ops,lambda x:(x['order_hash'],x['sid']),field,val)
 bad('future_context',cx,expected_cx,lambda x:(x['order_hash'],x['stage'],x['lag_ms']),'window_lo_ms','0');bad('invented_trigger',cx,expected_cx,lambda x:(x['order_hash'],x['stage'],x['lag_ms']),'context_identifies_trigger','True')
 try:check_rows(rows[1:],expected_chains,lambda x:x['order_hash'],'omitted order')
 except ValueError as e:falsifiers['omitted_order']=str(e)
 else:raise AssertionError('membership control escaped')
 out=dict(pass_all=True,orders=88,competing_records=len(ops),contexts=len(cx),falsifiers=falsifiers,cpu_seconds=time.process_time()-start);need(not (H/'checks.json').exists(),'write once');(H/'checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
