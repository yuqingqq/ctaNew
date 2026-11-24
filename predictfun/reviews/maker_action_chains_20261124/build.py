"""Frozen conditional availability, successor chains and independent context join."""
import csv,gzip,hashlib,io,json,math,resource,subprocess,sys,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import pandas as pd
import pyarrow as pa
pa.set_cpu_count(1);pa.set_io_thread_count(1)
H=Path(__file__).resolve().parent;R=H.parents[2];Q=R/'predictfun/reviews/maker_quantity_linkage_20261124';T=R/'predictfun/reviews/maker_timing_estimates_20261124';S=R/'predictfun/reviews/maker_terminal_chain_20261124';V=R/'data/predict_fun/research/maker_trace_v2_20261123';CH=R/'predictfun/reviews/maker_trace_20261122/v2/chains';DATA=R/'data/predict_fun/research/maker_action_chains_20261124'
def need(x,m):
 if not x:raise ValueError(m)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):
 with (gzip.open if str(p).endswith('.gz') else open)(p,'rt') as f:return list(csv.DictReader(f))
def flag(x):return x is True or x=='True'
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def write(p,rs,fields=None):
 b=io.StringIO();w=csv.DictWriter(b,fieldnames=list(rs[0]) if rs else fields,lineterminator='\n');w.writeheader();w.writerows(rs)
 if str(p).endswith('.gz'):
  with p.open('wb') as f:
   with gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0,compresslevel=1) as z:z.write(b.getvalue().encode())
 else:p.write_text(b.getvalue())
def bind():
 for fn,key in [('inputs.json','files'),('sources.json',None)]:
  d=json.loads((H/fn).read_text());d=d[key] if key else d
  for p,h in d.items():need(sha(R/p)==h,'identity '+p)
def freeze():
 need(not (H/'sources.json').exists(),'write once');dump(H/'sources.json',{str(p.relative_to(R)):sha(p) for p in sorted(list(H.glob('*.py'))+[H/'contract.md',H/'inputs.json'])})
def phase(lo,hi,own):
 if lo<0 or any(x['lo']<0 for x in own):return 'unbounded'
 if hi<min(x['lo'] for x in own):return 'pre_first'
 if lo>max(x['hi'] for x in own):return 'after_last'
 if any(x['hi']<lo for x in own) and any(x['lo']>hi for x in own):return 'between_fills'
 return 'overlap_or_tie'
def pending(t,p,coherent,lo,hi,required):
 a=int(np.searchsorted(t,lo,'right'))-1;b=int(np.searchsorted(t,hi,'left'))
 if a<0 or b>=len(t) or lo-t[a]>1000 or t[b]-hi>1000 or not np.isfinite(p[a:b+1]).all() or not coherent[a+1:b+1].all():return 'unknown',None
 mx=float(p[a:b+1].max())
 return ('visible_insufficient' if required>mx+.01 else 'could_cover'),mx
def successor_order(lo,hi,nlo,nhi,valid):
 if not valid:return 'refused_or_missing_window'
 if nlo>hi:return 'sequential_after_under_bounds'
 if nhi<lo:return 'posted_before_prior_end'
 return 'ordering_ambiguous'
def refine(base,ops,group,eligible,expiry):
 pl=int(base['post_A_lo']);el=int(base['old_end_lo']);eh=int(base['end_hi_conditional']);pw=ew='';interrupt=set()
 for x in ops:
  if x['witness_status']!='conditional_available_priority_violation':continue
  if x['phase']=='between_fills':interrupt.add(x['sid'])
  if not eligible:continue
  if x['phase']=='pre_first' and x['relation']=='through' and x['lo_ms']>pl:pl=x['lo_ms'];pw=x['sid']
  if x['phase']=='after_last' and group=='conditional_withdrawal' and x['hi_ms']<eh and x['hi_ms']<expiry:eh=x['hi_ms'];ew=x['sid']
 conflict=pl>int(base['post_hi_unchanged']) or eh<el
 if conflict:pl=int(base['post_A_lo']);eh=int(base['end_hi_conditional']);pw=ew=''
 return dict(post_lo_ms=pl,post_hi_ms=int(base['post_hi_unchanged']),end_lo_ms=el,end_hi_ms=eh,post_witness=pw,end_witness=ew,bounds_conflict=conflict,interruption_records=len(interrupt))
def record_groups(life,records):
 gs=defaultdict(list)
 for f in life.to_dict('records'):gs[f['sid']].append(f)
 out=[]
 for sid,fs in gs.items():
  need(sid in records,'record identity');r=records[sid];sides={f['y_side'] for f in fs};status=[]
  if len(fs)!=int(r['n_makers']):status.append('incomplete_record')
  if len(sides)!=1 or not sides<={'bid','ask'}:status.append('mixed_side')
  side=next(iter(sides)) if len(sides)==1 else 'unknown'
  if r['tk_y_side']!=('ask' if side=='bid' else 'bid'):status.append('taker_side')
  if r['status']!='ok' or any(any(s in f['status'] for s in ('unplaceable','mismatch','malformed','side_mixed')) for f in fs):status.append('source_flags')
  lo=min(int(f['x_lo_hard_ms']) for f in fs);hi=max(int(f['hi_ms']) for f in fs)
  if lo<0 or lo>hi:status.append('unbounded_clock')
  if any(not 0<int(f['y_ptick'])<100 for f in fs):status.append('price_invalid')
  out.append(dict(sid=sid,lo=lo,hi=hi,taker=r['tk_signer'].lower(),side=side,prices={int(f['y_ptick']) for f in fs},prices_by_side={s:{int(f['y_ptick']) for f in fs if f['y_side']==s} for s in sides},hashes={f['mk_hash'] for f in fs},status='|'.join(sorted(status)) if status else 'ok'))
 return out

def physical_market(day,mid,orders,bounds,posts,signed,normal,records):
 life=pd.read_parquet(V/'life'/f'{day}_{mid}_fills.parquet');meta=json.loads((V/'life'/f'{day}_{mid}.json').read_text());lo,hi=meta['ws_ms'],meta['we_ms'];groups=record_groups(life,records)
 path=R/normal['normalized_file'];need(sha(path)==normal['normalized_sha256'],'normalized identity');df=pd.read_csv(path,usecols=['side','tick','t_ms','pending_own_shares','pending_other_shares','coherent_previous'],float_precision='round_trip')
 cells={(s,int(k)):(g.t_ms.to_numpy(np.int64),g.pending_own_shares.to_numpy()+g.pending_other_shares.to_numpy(),g.coherent_previous.to_numpy(bool)) for (s,k),g in df.groupby(['side','tick'],sort=False)}
 out=[];allops=[]
 for o in orders:
  h=o['order_hash'];side=o['side'];tick=int(o['tick']);b=bounds[h];p=posts[h];own=[dict(lo=int(f.x_lo_hard_ms),hi=int(f.hi_ms),q=float(f.shares)) for f in life[life.mk_hash==h].itertuples()];need(own,'own fills');ss=signed[h];maker=ss['maker'].lower();expiry=float(ss['expiration_s'])*1000;op=[];census=Counter()
  for g in groups:
   if side not in g['prices_by_side']:continue
   prices=g['prices_by_side'][side];reach=any(k<=tick if side=='bid' else k>=tick for k in prices)
   if not reach or g['hi']<max(lo,int(b['post_A_lo'])) or g['lo']>hi:continue
   rel='through' if any(k<tick if side=='bid' else k>tick for k in prices) else 'same_price';ph=phase(g['lo'],g['hi'],own)
   required=sum(x['q'] for x in own) if ph=='pre_first' else (sum(x['q'] for x in own if x['lo']>g['hi']) if ph=='between_fills' else (float(o['remaining_quote_shares_upper']) if ph=='after_last' else 0.))
   st=g['status'];ps='not_checked';pm=None
   if st!='ok':status='record_refused'
   elif h in g['hashes']:status='own_execution_record'
   elif g['taker']==maker:status='self_trade_prevention_possible'
   elif ph in ('unbounded','overlap_or_tie'):status='clock_overlap_or_unknown'
   elif ph=='pre_first' and rel=='same_price':status='no_initial_time_priority'
   elif required<=.01:status='no_remaining_required'
   else:
    ps,pm=pending(*cells[(side,tick)],g['lo'],g['hi'],required)
    status='conditional_available_priority_violation' if ps=='visible_insufficient' else ('reservation_can_explain' if ps=='could_cover' else 'pending_unknown')
   census[status]+=1;op.append(dict(day=day,mid=mid,order_hash=h,sid=g['sid'],side=side,tick=tick,relation=rel,phase=ph,lo_ms=g['lo'],hi_ms=g['hi'],taker=g['taker'],record_status=st,required_shares=required,pending_status=ps,pending_max_shares=pm,witness_status=status,private_action_observed=False))
  eligible=b['bound_status']=='conditional_only' and flag(o['eligible_conditional_bounds']);rr=refine(b,op,o['group'],eligible,expiry)
  if o['group']=='exhausted':rr['end_lo_ms']=max(x['lo'] for x in own);rr['end_hi_ms']=max(x['hi'] for x in own);rr['end_witness']=''
  terminal='exhausted' if o['group']=='exhausted' else ('conditional_unavailable' if o['group']=='conditional_withdrawal' and eligible else 'unresolved')
  out.append(dict(day=day,mid=mid,order_hash=h,rank=int(o['rank']),maker=maker,side=side,tick=tick,ws_ms=lo,we_ms=hi,original_shares=float(o['original_shares']),observed_shares=sum(x['q'] for x in own),remaining_upper_shares=float(o['remaining_quote_shares_upper']),maker_fill_records=len(own),first_fill_lo_ms=min(x['lo'] for x in own),first_fill_hi_ms=min(x['hi'] for x in own),last_fill_lo_ms=max(x['lo'] for x in own),last_fill_hi_ms=max(x['hi'] for x in own),group=o['group'],prior_post_grade=p['grade'],accepted_bounds=eligible,**rr,old_post_lo_ms=int(b['post_A_lo']),old_end_hi_ms=int(b['end_hi_conditional']),terminal=terminal,expiry_ms=int(expiry),expiry_relation='after_interval' if expiry>rr['end_hi_ms'] else ('inside_interval' if expiry>=rr['end_lo_ms'] else 'before_interval'),candidate_records=len(op),candidate_status_counts=json.dumps(dict(sorted(census.items())),sort_keys=True),private_pull_observed=False))
  allops+=op
 return out,allops

def link_successors(chains,posts,old,qs):
 lookup={x['order_hash']:x for x in chains};res=[]
 for x in chains:
  r=dict(x);s=old[x['order_hash']];h=s['successor_hash'];n=posts.get(h);usable=n is not None and n['maker'].lower()==x['maker'] and str(n['day'])==x['day'] and int(n['mid'])==x['mid'] and n['side']==x['side']
  if h and n is not None:need(usable,'successor exact identity')
  a=z=None;grade='missing';ns=None
  if usable:
   a=int(n['effect_lo_ms']) if n['effect_lo_ms'] else None;z=int(n['effect_hi_ms']) if n['effect_hi_ms'] else None;grade=n['grade']
   if h in lookup and lookup[h]['accepted_bounds']:a=lookup[h]['post_lo_ms'];z=lookup[h]['post_hi_ms']
   if h in qs:ns=float(qs[h]['original_shares'])
  valid=usable and flag(n['estimate_enabled']) and a is not None and z is not None and x['terminal']!='unresolved' and 0<=x['end_lo_ms']<=x['end_hi_ms'] and 0<=a<=z
  relation=successor_order(x['end_lo_ms'],x['end_hi_ms'],a,z,valid) if h else 'no_recorded_successor'
  kind=('refill' if int(n['tick'])==x['tick'] else 'reprice') if usable else 'unknown'
  direction=('same_price' if kind=='refill' else ('toward_spread' if (int(n['tick'])-x['tick'])*(1 if x['side']=='bid' else -1)>0 else 'away_from_spread')) if usable else 'unknown'
  r.update(successor_hash=h,successor_kind=kind,successor_direction=direction,successor_post_lo_ms=a,successor_post_hi_ms=z,successor_post_grade=grade,successor_original_shares=ns,successor_relation=relation,replacement_intent_identified=False)
  r['chain_class']=('exhaustion_then_'+kind if x['terminal']=='exhausted' else 'conditional_ending_then_'+kind) if relation=='sequential_after_under_bounds' else ('successor_already_posted_candidate_overlap' if relation=='posted_before_prior_end' else ('successor_ordering_unresolved' if relation=='ordering_ambiguous' else (x['terminal']+'_no_timed_successor')))
  res.append(r)
 return res

def categories(values,fn):
 known=[float(x) for x in values if math.isfinite(float(x))];ss=sorted({fn(x) for x in known});return '|'.join(ss) if len(known)==len(values) and known else 'unknown'
def annotate(chains,postpanel,takepanel,takes):
 P={(str(d),int(m),s):g.sort_values('k') for (d,m,s),g in postpanel.groupby(['day','mid','side'],sort=False)};I={(str(d),int(m),int(rank)):g.sort_values('k') for (d,m,rank),g in takepanel.groupby(['day','mid','rank'],sort=False)};events={(str(d),int(m),str(a).lower()):g for (d,m,a),g in takes.groupby(['day','mid','addr'],sort=False)};contexts=[]
 for r in chains:
  p=P[(r['day'],r['mid'],r['side'])];iv=I[(r['day'],r['mid'],r['rank'])];need(len(p)==len(iv)==300 and np.array_equal(p.t_ms,iv.t_ms),'context grids');sgn=1 if r['side']=='bid' else -1
  for stage,a,z in [('post',r['post_lo_ms'],r['post_hi_ms']),('ending',r['end_lo_ms'],r['end_hi_ms']),('successor_post',r['successor_post_lo_ms'],r['successor_post_hi_ms'])]:
   for lag in (0,1000):
    admissible=(r['accepted_bounds'] if stage=='post' else (r['terminal']!='unresolved' if stage=='ending' else r['successor_post_grade'] in ('G2_bound_constrained','G4_scenario_only')))
    valid=admissible and a is not None and z is not None and r['ws_ms']<=a<=z<r['we_ms'];ix=np.arange(max(0,int((a-r['ws_ms'])//1000)),min(299,int((z-r['ws_ms'])//1000))+1) if valid else np.array([],int);suffix='D'+str(lag);isuf='0' if lag==0 else '1'
    if valid and len(ix):
     btc=p['btc_r_5s_bp_'+suffix].to_numpy()[ix]*sgn;q=iv['q'+isuf].to_numpy()[ix]*sgn;book=p['status_'+suffix].to_numpy()[ix];sp=p['spread_ticks_'+suffix].to_numpy()[ix];dep=p['depth_touch_own_'+suffix].to_numpy()[ix];tl=p['time_left_s_'+suffix].to_numpy()[ix]
     bc=categories(btc,lambda v:'with' if v>=1 else ('against' if v<=-1 else 'flat'));ic=categories(q,lambda v:'adding' if v>1e-6 else ('offsetting' if v< -1e-6 else 'flat'));qlo=float(np.min(q)) if np.isfinite(q).all() else None;qhi=float(np.max(q)) if np.isfinite(q).all() else None
     bs='|'.join(sorted(set(map(str,book))));spc=categories(sp,lambda v:'wide' if v>=2 else 'tight');dc=categories(dep,lambda v:'thin' if v<=50 else ('deep' if v>200 else 'middle'));tc=categories(tl,lambda v:'last30s' if v<=30 else ('last60s' if v<=60 else 'earlier'))
    else:bc=ic=bs=spc=dc=tc='unknown';qlo=qhi=None
    contexts.append(dict(day=r['day'],mid=r['mid'],rank=r['rank'],order_hash=r['order_hash'],chain_class=r['chain_class'],stage=stage,lag_ms=lag,window_lo_ms=a,window_hi_ms=z,coverage='grid_interval_covered' if valid else 'outside_grid_missing_or_refused',asof_seconds=len(ix),btc_side=bc,inventory_side=ic,inventory_signed_min=qlo,inventory_signed_max=qhi,inventory_ledger_status=('clean' if valid and iv.clean.to_numpy()[ix].all() else 'conditional_or_unavailable'),book_status=bs,spread=spc if bs=='ok' else 'unknown',own_depth=dc if bs=='ok' else 'unknown',time_left=tc,client_inventory_observed=False,context_identifies_trigger=False))
  ev=events.get((r['day'],r['mid'],r['maker']));chosen=None
  if ev is not None:
   eligible=ev[(ev.x_lo_hard_ms>r['last_fill_hi_ms'])&(ev.hi_ms<=r['we_ms'])].sort_values(['hi_ms','sid'])
   if len(eligible):chosen=eligible.iloc[0]
  r.update(later_take_sid=str(chosen.sid) if chosen is not None else '',later_take_lo_ms=int(chosen.x_lo_hard_ms) if chosen is not None else None,later_take_hi_ms=int(chosen.hi_ms) if chosen is not None else None,later_take_inventory_status=str(chosen.o4_status) if chosen is not None else 'none_observed',later_take_direction_D0=str(chosen.dir_D0) if chosen is not None else 'none_observed',take_is_this_orders_exit=False)
 return contexts

def summary(chains,ops,cx):
 count=lambda rs,f:dict(sorted(Counter(x[f] for x in rs).items()))
 return dict(orders=len(chains),groups=count(chains,'group'),chain_classes=count(chains,'chain_class'),successors=count(chains,'successor_relation'),candidate_records=len(ops),candidate_status=count(ops,'witness_status'),new_post_bounds=sum(x['post_lo_ms']>x['old_post_lo_ms'] for x in chains),new_end_bounds=sum(x['end_witness']!='' for x in chains),orders_with_between_fill_interruption=sum(x['interruption_records']>0 for x in chains),bound_conflicts=sum(x['bounds_conflict'] for x in chains),exhausted_terminal_pulls=sum(x['terminal']!='exhausted' for x in chains if x['group']=='exhausted'),private_pulls_identified=sum(x['private_pull_observed'] for x in chains),later_take_references=sum(bool(x['later_take_sid']) for x in chains),distinct_later_takes=len({(x['day'],x['mid'],x['maker'],x['later_take_sid']) for x in chains if x['later_take_sid']}),contexts=len(cx))

def run():
 start=time.process_time();resource.setrlimit(resource.RLIMIT_CPU,(8,10));bind();need(not (H/'outputs').exists(),'write once');DATA.mkdir(exist_ok=False)
 orders=read(Q/'outputs/order_summary.csv');bounds={x['order_hash']:x for x in read(Q/'outputs/chain_bounds.csv')};posts={x['order_hash']:x for x in read(T/'outputs/post_estimates.csv.gz')};old={x['order_hash']:x for x in read(T/'outputs/signed_chain_estimates.csv')};signed={x['order_hash']:x for x in read(S/'orders.csv')};norm={(str(x['day']),int(x['mid'])):x for x in json.loads((Q/'outputs/normalization.json').read_text())};pairs=json.loads((H/'inputs.json').read_text())['pairs'];records={}
 for d in sorted({d for d,m in pairs}):
  ms=[m for dd,m in pairs if dd==d];rr=pd.read_parquet(V/f'records_{d}.parquet',columns=['mid','sid','tk_signer','tk_y_side','status','n_makers'],filters=[('mid','in',ms)])
  for m,g in rr.groupby('mid'):records[(d,int(m))]={x['sid']:x for x in g.to_dict('records')}
 chains=[];ops=[]
 for d,m in pairs:
  oo=[x for x in orders if x['day']==d and int(x['mid'])==m];a,z=physical_market(d,m,oo,bounds,posts,signed,norm[(d,m)],records[(d,m)]);chains+=a;ops+=z;write(DATA/f'{d}_{m}_physical.csv',a);write(DATA/f'{d}_{m}_evidence.csv.gz',z,fields=['day','mid','order_hash','sid'])
  need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1450*1024,'RSS');print('physical checkpoint',d,m,round(time.process_time()-start,3),flush=True)
 chains=link_successors(chains,posts,old,{x['order_hash']:x for x in orders});need(len(chains)==88 and len({x['order_hash'] for x in chains})==88,'population')
 pp=pd.read_parquet(CH/'posting_opportunity/rows.parquet');ip=pd.read_parquet(CH/'take_logic_v2/panel.parquet');ev=pd.read_parquet(CH/'take_logic_v2/events.parquet');cx=annotate(chains,pp,ip,ev)
 out=H/'outputs';out.mkdir();write(out/'chains.csv',chains);write(out/'competing_events.csv.gz',ops);write(out/'contexts.csv.gz',cx)
 cells=Counter((x['rank'],x['chain_class'],x['stage'],x['lag_ms'],x['coverage'],x['btc_side'],x['inventory_side'],x['time_left']) for x in cx)
 write(out/'condition_chains.csv',[dict(rank=k[0],chain_class=k[1],stage=k[2],lag_ms=k[3],coverage=k[4],btc_side=k[5],inventory_side=k[6],time_left=k[7],orders=n) for k,n in sorted(cells.items())])
 result=summary(chains,ops,cx);result.update(as_of_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),cpu_seconds=time.process_time()-start,hashes={p.name:sha(p) for p in sorted(out.iterdir())});dump(out/'result.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':{'freeze':freeze,'run':run}[sys.argv[1]]()
