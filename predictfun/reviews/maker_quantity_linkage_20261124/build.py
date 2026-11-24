"""Frozen, bounded normalization and indirect-link checks. No behavioural inputs."""
import csv,gzip,hashlib,importlib.util,io,json,math,os,resource,subprocess,sys,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
pa.set_cpu_count(1);pa.set_io_thread_count(1)
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];DATA=ROOT/'data/predict_fun/research/maker_quantity_linkage_20261124'
T=ROOT/'predictfun/reviews/maker_terminal_chain_20261124';RAW=ROOT/'data/predict_fun/research/maker_terminal_chain_20261124'
TI=ROOT/'predictfun/reviews/maker_timing_estimates_20261124';DEPTH=ROOT/'data/predict_fun/research/maker_trace_v1_20261122/depth';LIFE=ROOT/'data/predict_fun/research/maker_trace_v2_20261123/life'
sys.path.insert(0,str(ROOT/'predictfun/research'));import pf_mt_lifecycle as LC
E18=10**18

def need(x,s):
 if not x:raise ValueError(s)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def rows(p):
 with (gzip.open if str(p).endswith('.gz') else open)(p,'rt',newline='') as f:return list(csv.DictReader(f))
def write(p,rs,fields=None):
 buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(rs[0]) if rs else fields,lineterminator='\n');need(w.fieldnames is not None,'empty schema');w.writeheader();w.writerows(rs)
 if str(p).endswith('.gz'):
  with p.open('wb') as f:
   with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as g:g.write(buf.getvalue().encode())
 else:p.write_text(buf.getvalue())
def truth(x):return x is True or x=='True'
def num(x):
 try:return float(x)
 except (TypeError,ValueError):return math.nan
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 if isinstance(x,np.generic):x=x.item()
 if isinstance(x,float) and not math.isfinite(x):return None
 return x

def guard():need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1450*1024,'RSS cap')
def bind():
 for p,h in json.loads((HERE/'inputs.json').read_text())['files'].items():need(sha(ROOT/p)==h,'input hash '+p)
 for p,h in json.loads((HERE/'sources.json').read_text()).items():need(sha(ROOT/p)==h,'source hash '+p)
def freeze():
 need(not (HERE/'sources.json').exists(),'write once');ps=list(HERE.glob('*.py'))+[HERE/'contract.md',HERE/'inputs.json',ROOT/'predictfun/research/pf_mt_lifecycle.py',ROOT/'predictfun/research/pf_mt_common.py']
 ps += [ROOT/'predictfun/reviews/maker_reversal_hypothesis_20261124/outputs/chronology.csv']
 dump(HERE/'sources.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(ps)});print('Source closure bound before pilot.')
def native(v,y_side,tick):
 need(v[10] in (0,1) and v[5]>0 and v[6]>0,'native budget')
 price=v[5]/v[6] if v[10]==0 else v[6]/v[5];q=(v[6] if v[10]==0 else v[5])/E18
 index=1 if (v[10]==0 and y_side=='bid') or (v[10]==1 and y_side=='ask') else 2
 yp=price if index==1 else 1-price
 need(0<price<1 and abs(100*yp-tick)<1e-6,'native YES price')
 return q,price,index

def ledger(es,order):
 need(len(es)==len({(e['tx'],e['log_index']) for e in es}),'duplicate event')
 es=sorted(es,key=lambda e:(e['block_number'],e['transaction_index'],e['log_index']));v=es[0]['values'];cum=0;out=[]
 for e in es:
  need(e['values']==v,'terms changed');before=v[5]-cum;cum+=e['making'];need(0<e['making'] and cum<=v[5],'budget exceeded')
  out.append(dict(day=order['day'],mid=int(order['market']),order_hash=e['order_hash'],tx=e['tx'],log_index=e['log_index'],role=e['role'],block_number=e['block_number'],transaction_index=e['transaction_index'],
   native_side=v[10],token_id=str(v[4]),original_making_wei=str(v[5]),making_wei=str(e['making']),cumulative_making_wei=str(cum),remaining_before_wei=str(before),remaining_after_wei=str(v[5]-cum),fill_shares_wei=str(e['shares']),
   timeline='settlement_order_not_match_order'))
 need(cum==int(order['observed_making_wei']) and v[5]==int(order['original_making_wei']),'source budget')
 return out

def reserved(original,settled,pending):
 need(0<=settled<=original and 0<=pending<=original-settled,'reservation budget');return original-settled-pending

def capacity_post(t,cap,first_lo,oldlo,oldhi,required):
 # All points before earliest possible FIRST match: no premature disappearance interpreted as late posting.
 mask=(t>=oldlo)&(t<min(first_lo,oldhi))&np.isfinite(cap)&(cap+.01<required)
 if not mask.any():return oldlo,None
 i=np.flatnonzero(mask)[-1];return max(oldlo,int(t[i])),int(i)

def capacity_end(t,cap,last_hi,oldlo,oldhi,remainder,exhausted):
 if exhausted or remainder<=.01:return oldhi,None
 a=np.flatnonzero((t>=last_hi)&(t>=oldlo)&(t<=oldhi)&np.isfinite(cap)&(cap+.01<remainder))
 if not len(a):return oldhi,None
 i=int(a[0]);return min(oldhi,int(t[i])),i

def input_tables():
 orders=rows(T/'orders.csv');byhash={r['order_hash']:r for r in orders};need(len(byhash)==88,'88 keys')
 es=json.loads((RAW/'events.json').read_text());need(len(es)==len({(e['tx'],e['log_index']) for e in es}),'global event dedup')
 per=defaultdict(list)
 for e in es:need(e['order_hash'] in byhash,'unexpected hash');per[e['order_hash']].append(e)
 posts={r['order_hash']:r for r in rows(TI/'outputs/post_estimates.csv.gz')};un={r['order_hash']:r for r in rows(TI/'outputs/union_windows.csv.gz')};signed={r['order_hash']:r for r in rows(TI/'outputs/signed_chain_estimates.csv')}
 manifest=json.loads((T/'manifest.json').read_text());rest=manifest['legs']
 return orders,per,posts,un,signed,rest

def normalized_book(day,mid,ws,we,ticks):
 f=pd.read_parquet(DEPTH/f'{day}_{mid}_frames.parquet');lv=pd.read_parquet(DEPTH/f'{day}_{mid}_levels.parquet')
 b=LC._book(f,lv,mid,ws,we);t=b['t'];fid=b['fid'];fm=f.set_index('fid').loc[fid]
 bad=np.zeros(len(t),bool)
 for i,r in enumerate(fm.itertuples()):
  for field in ('pend_bids','pend_asks'):
   pp=LC._pend(getattr(r,field))
   if pp is None or any(abs(p*100-round(p*100))>1e-6 for p,u in pp):bad[i]=True
 conn=fm.conn.astype(str).to_numpy();pos={int(v):i for i,v in enumerate(f.fid)};rawpos=np.array([pos[int(v)] for v in fid])
 pair=np.zeros(len(t),bool);pair[1:]=(np.diff(t)>0)&(np.diff(t)<=1000)&(conn[1:]==conn[:-1])&(np.diff(rawpos)==1)&~bad[1:]&~bad[:-1]
 evidence=[];cell={}
 for side,k in sorted(set(ticks)):
  sc=0 if side=='bid' else 1;p=k/100;need(0<p<1,'cell price')
  d=b['shown'][:,sc,k].copy();own=b['pend'][:,sc,k].copy();opp=b['pend'][:,1-sc,k].copy();d[bad]=np.nan;own[bad]=np.nan;opp[bad]=np.nan
  po=own/p;pa=opp/p;cap=d+po+pa;validpair=pair.copy();validpair[1:] &= np.isfinite(d[1:]+d[:-1]+po[1:]+po[:-1]+pa[1:]+pa[:-1])
  cell[(side,k)]=dict(t=t,fid=fid,d=d,own_usd=own,opp_usd=opp,own=po,opp=pa,cap=cap,pair=validpair,recv_ns=fm.recv_ns.to_numpy(np.int64))
  for i in range(len(t)):
   evidence.append(dict(day=day,mid=mid,fid=int(fid[i]),t_ms=int(t[i]),recv_ns=int(fm.recv_ns.iat[i]),side=side,tick=k,displayed_shares=clean(d[i]),pending_own_usd=clean(own[i]),pending_other_usd=clean(opp[i]),pending_own_shares=clean(po[i]),pending_other_shares=clean(pa[i]),capacity_upper_shares=clean(cap[i]),coherent_previous=bool(validpair[i])))
 return cell,evidence,dict(frames=b['F'],raw_frames=len(f),excluded=b['excluded'],pending_offgrid_frames=int(bad.sum()),display_semantics='bracketed_not_identified',pending_allocation='aggregate_unattributed')

def market(day,mid,inputs,output):
 start=time.process_time();orders,per,posts,un,signed,rest=inputs;orders=[r for r in orders if (r['day'],int(r['market']))==(day,mid)]
 life=pd.read_parquet(LIFE/f'{day}_{mid}_fills.parquet');meta=json.loads((LIFE/f'{day}_{mid}.json').read_text());ws=meta['ws_ms'];we=meta['we_ms']
 ticks=[(posts[r['order_hash']]['side'],int(posts[r['order_hash']]['tick'])) for r in orders];cells,ev,norm=normalized_book(day,mid,ws,we,ticks)
 outfile=output/f'{day}_{mid}_normalized.csv.gz';write(outfile,ev);guard()
 ll=[];summ=[];bounds=[];edges=[];be=[];joint=[];prepared={};raw_rest={(x['sid'],int(x['mk_idx'])):x for x in rest if (x['day'],int(x['mid']))==(day,mid)}
 for o in orders:
  h=o['order_hash'];p=posts[h];s=p['side'];k=int(p['tick']);q,native_price,index=native(per[h][0]['values'],s,k);fl=life[life.mk_hash==h].sort_values(['hi_ms','sid','mk_idx']);need(len(fl)>0,'missing lifecycle fills')
  need((fl.y_side==s).all() and (fl.y_ptick==k).all(),'fill cell');es=per[h];l=ledger(es,o);ll+=l
  rr=[x for x in rest if x['mk_hash']==h];need(len(fl)==len(rr),'REST/lifecycle membership')
  rt=defaultdict(float)
  for r in fl.to_dict('records'):
   x=raw_rest[(r['sid'],int(r['mk_idx']))];need(x['mk_hash']==h and abs(float(x['shares'])-r['shares'])<1e-6,'REST/life quantity');rt[x['tx']]+=r['shares']
  ct=defaultdict(int)
  for e in es:
   if e['role']=='maker':ct[e['tx']]+=e['shares']
  need(set(ct)==set(rt) and all(abs(rt[x]-ct[x]/E18)<1e-6 for x in rt),'REST/raw same execution')
  remwei=int(o['remaining_upper_wei']);v=es[0]['values'];remq=remwei*(v[6]/v[5] if v[10]==0 else 1)/E18
  exhausted=remwei==0;group='exhausted' if exhausted else ('conditional_withdrawal' if truth(signed[h]['conditional_withdrawal_all_views']) else 'unresolved')
  eligible=truth(p['estimate_enabled']) and all(e['role']=='maker' for e in es) and (fl.x_lo_hard_ms>=0).all() and un[h]['union_status']=='three_view_union'
  t=cells[(s,k)]['t'];req=np.zeros(len(t));firsthi=int(fl.hi_ms.min());firstlo=int(fl.x_lo_hard_ms.min());lasthi=int(fl.hi_ms.max());lastlo=int(fl.x_lo_hard_ms.max())
  for f in fl.itertuples():req += np.where(t<int(f.x_lo_hard_ms),float(f.shares),0.)
  prepared[h]=dict(o=o,p=p,s=s,k=k,fl=fl,q=q,remq=remq,exhausted=exhausted,group=group,eligible=eligible,req=req,firsthi=firsthi,firstlo=firstlo,lasthi=lasthi,lastlo=lastlo)
  summ.append(dict(day=day,mid=mid,order_hash=h,rank=int(o['rank']),side=s,tick=k,token_id=str(v[4]),native_side=v[10],native_price=native_price,YES_price=k/100,accepted_outcome_mapping=index,
   original_shares=q,native_making_budget=str(v[5]),observed_making=str(v[5]-remwei),remaining_budget_upper=str(remwei),remaining_quote_shares_upper=remq,raw_events=len(es),maker_fills=len(fl),raw_transactions=len({e['tx'] for e in es}),group=group,eligible_conditional_bounds=bool(eligible),first_hard_lo=firstlo,first_hi=firsthi,last_hi=lasthi,
   original_size_immutable=True,live_remainder_observed=False,pending_per_order='unknown_aggregate_only'))
 # Joint future demand under continuous rest, with unknown other orders left unallocated.
 conflicts=set()
 for cell,z in cells.items():
  hs=[h for h,a in prepared.items() if (a['s'],a['k'])==cell];dem=np.zeros(len(z['t']));parts={}
  for h in hs:
   a=prepared[h];parts[h]=np.where((z['t']>=a['firsthi'])&(z['t']<a['lastlo']),a['req'],0.);dem+=parts[h]
  bad=np.isfinite(z['cap'])&(dem>z['cap']+.01*max(1,len(hs)))
  for i in np.flatnonzero(bad):
   members=[h for h in hs if parts[h][i]>0];conflicts.update(members);joint.append(dict(day=day,mid=mid,fid=int(z['fid'][i]),t_ms=int(z['t'][i]),side=cell[0],tick=cell[1],required_future_shares=float(dem[i]),capacity_upper=float(z['cap'][i]),order_hashes='|'.join(members)))
 # Shared record groups at every selected cell, including nonselected maker legs in that group.
 groups={}
 for (sid,s,k),g in life.groupby(['sid','y_side','y_ptick'],sort=False):
  if (s,int(k)) in cells:groups[(sid,s,int(k))]=dict(q=float(g.shares.sum()),lo=int(g.x_lo_hard_ms.min()),hi=int(g.hi_ms.max()),hashes=set(g.mk_hash))
 candidate_keys=set();add_keys=set()
 for h,a in prepared.items():
  z=cells[(a['s'],a['k'])];t=z['t'];cap=z['cap'];p=a['p'];oldlo=int(p['effect_lo_ms']);oldhi=int(p['effect_hi_ms']);u=un[h];elo=int(u['union_lo_ms']);ehi=int(u['union_hi_ms'])
  alo,ai=capacity_post(t,cap,a['firstlo'],oldlo,oldhi,a['req']);blo,bi=capacity_post(t,cap,a['firstlo'],oldlo,oldhi,a['q']);eup,ei=capacity_end(t,cap,a['lasthi'],elo,ehi,a['remq'],a['exhausted'])
  validold=0<=oldlo<=oldhi and 0<=elo<=ehi
  can=a['eligible'] and h not in conflicts and validold
  postA=alo if can else oldlo;postB=blo if can else oldlo;end=eup if can and a['group']=='conditional_withdrawal' else ehi
  if validold:need(oldlo<=postA<=oldhi and oldlo<=postB<=oldhi,'post bound consistency');need(elo<=end<=ehi,'end bound consistency')
  changed=(postA>oldlo or postB>oldlo or end<ehi)
  bounds.append(dict(day=day,mid=mid,order_hash=h,group=a['group'],prior_post_grade=p['grade'],bound_status='joint_capacity_conflict' if h in conflicts else ('conditional_only' if can else 'prior_grade_or_clock_or_role_refused'),
   old_post_lo=oldlo,old_post_hi=oldhi,post_A_lo=postA,post_B_lo=postB,post_hi_unchanged=oldhi,post_A_candidate=alo,post_B_candidate=blo,
   post_A_witness_fid=int(z['fid'][ai]) if ai is not None else None,post_B_witness_fid=int(z['fid'][bi]) if bi is not None else None,
   old_end_lo=elo,old_end_hi=ehi,end_hi_conditional=end,end_candidate=eup,end_witness_fid=int(z['fid'][ei]) if ei is not None else None,
   narrowed=bool(changed),terminal_pull_certified=False,exhausted_terminal_pull=False,post_lower_open=bool(postA>oldlo or postB>oldlo),full_signed_coverage_assumed=bool(postB>oldlo or end<ehi)))
  counts={};match_e=0;owned=[]
  for gkey in sorted({(f.sid,a['s'],a['k']) for f in a['fl'].itertuples()}):
   g=groups[gkey];candidate=[]
   for j in np.flatnonzero(z['pair']&(t>=g['lo'])&(np.r_[t[0],t[:-1]]<=g['hi'])):
    if j==0:continue
    dq=z['d'][j-1]-z['d'][j];own=z['own_usd'][j]-z['own_usd'][j-1];opp=z['opp_usd'][j]-z['opp_usd'][j-1];v=g['q']*a['k']/100
    pend=max(own,opp)>=max(0.,v-max(.011*v,.01)) and max(own,opp)>1e-9
    drop=dq>=max(.01,g['q']-max(.10*g['q'],.5))
    if not (pend or drop):continue
    candidate.append(j);ek=(gkey,int(z['fid'][j-1]),int(z['fid'][j]));match_e+=1
    if ek not in candidate_keys:
     candidate_keys.add(ek);be.append(dict(day=day,mid=mid,kind='match_reflection_candidate',side=a['s'],tick=a['k'],sid=gkey[0],fid_before=int(z['fid'][j-1]),fid_after=int(z['fid'][j]),lo_ms=int(t[j-1]),hi_ms=int(t[j]),
      order_hashes='|'.join(sorted(g['hashes']&set(prepared))),quantity_shares=g['q'],drop_shares=float(dq),pending_gain_usd=float(max(own,opp)),evidence='both' if pend and drop else ('pending' if pend else 'drop'),attribution='scenario_possible_shared',competing_records=0))
   counts[gkey[0]]=len(candidate)
  # Addition and unexplained-drop indicators are per order window, not attributed commands.
  add=0;unexpl=0
  for j in np.flatnonzero(z['pair']&(t>=oldlo)&(np.r_[t[0],t[:-1]]<=oldhi)):
   if j and z['d'][j]>z['d'][j-1]+.01:
    add+=1;ek=(a['s'],a['k'],int(z['fid'][j-1]),int(z['fid'][j]))
    if ek not in add_keys:
     add_keys.add(ek);be.append(dict(day=day,mid=mid,kind='unattributed_add_or_restore',side=a['s'],tick=a['k'],sid='',fid_before=int(z['fid'][j-1]),fid_after=int(z['fid'][j]),lo_ms=int(t[j-1]),hi_ms=int(t[j]),order_hashes='',quantity_shares=float(z['d'][j]-z['d'][j-1]),drop_shares=float(z['d'][j-1]-z['d'][j]),pending_gain_usd=float(max(z['own_usd'][j]-z['own_usd'][j-1],z['opp_usd'][j]-z['opp_usd'][j-1])),evidence='zero_to_positive' if z['d'][j-1]==0 else 'net_addition',attribution='anonymous_not_post_certificate',competing_records=0))
  fg=life[(life.y_side==a['s'])&(life.y_ptick==a['k'])];glo=fg.x_lo_hard_ms.to_numpy();ghi=fg.hi_ms.to_numpy();gq=fg.shares.to_numpy()
  for j in np.flatnonzero(z['pair']&(t>=elo)&(np.r_[t[0],t[:-1]]<=ehi)):
   if j==0:continue
   drop=z['d'][j-1]-z['d'][j];possible=float(gq[(glo<=t[j])&(ghi>=t[j-1])].sum())
   if drop>possible+max(.5,.10*possible):
    unexpl+=1;certain=float(gq[(glo>=t[j-1])&(ghi<=t[j])].sum())
    be.append(dict(day=day,mid=mid,kind='unexplained_drop_candidate',side=a['s'],tick=a['k'],sid='',fid_before=int(z['fid'][j-1]),fid_after=int(z['fid'][j]),lo_ms=int(t[j-1]),hi_ms=int(t[j]),order_hashes=h,quantity_shares=possible,drop_shares=float(drop),pending_gain_usd=float(max(z['own_usd'][j]-z['own_usd'][j-1],z['opp_usd'][j]-z['opp_usd'][j-1])),evidence='recorded_execution_shortfall_not_cancel_proof',attribution='anonymous_not_pull_certificate',competing_records=0,certain_recorded_shares=certain,possible_recorded_shares=possible))
  relations=[('R01','observed_budget_ledger',len(per[h])),('R02','exhausted' if a['exhausted'] else 'incomplete_budget_not_live',1),('R03','candidate_match_reflection' if match_e else 'no_visible_match_candidate',match_e),('R04','unknown_failure_stream',0),('R05','first_fill_execution_bound',1),('R06','repeated_execution_not_continuity' if len(a['fl'])>1 else 'single_recorded_fill',len(a['fl'])),('R07','unattributed_add_restore_candidates' if add else 'no_visible_creation_candidate',add),('R08','conditional_bound_narrowed' if changed else ('refused' if not can else 'no_bound_gain'),int(changed)),('R09','joint_capacity_conflict' if h in conflicts else 'no_selected_joint_conflict_no_unique_allocation',sum(counts.values())),('R10','unexplained_drop_candidates' if unexpl else 'no_unexplained_drop',unexpl)]
  for rid,status,n in relations:edges.append(dict(day=day,mid=mid,order_hash=h,relation=rid,status=status,evidence_count=int(n),private_action_identified=False))
 # Count records competing for one shared book pair; amounts are not allocated to several orders independently.
 shares=defaultdict(set)
 for x in be:
  if x['kind']=='match_reflection_candidate':shares[(x['side'],x['tick'],x['fid_before'],x['fid_after'])].add(x['sid'])
 for x in be:
  if x['kind']=='match_reflection_candidate':x['competing_records']=len(shares[(x['side'],x['tick'],x['fid_before'],x['fid_after'])])
  x.setdefault('certain_recorded_shares',None);x.setdefault('possible_recorded_shares',None)
 norm.update(day=day,mid=mid,orders=len(orders),normalized_file=str(outfile.relative_to(ROOT)),normalized_sha256=sha(outfile),normalized_rows=len(ev),cpu_seconds=time.process_time()-start)
 return dict(ledger=ll,orders=summ,bounds=bounds,relations=edges,book=be,joint=joint,normalization=norm)

def summarize(parts):
 orders=[r for p in parts for r in p['orders']];bounds=[r for p in parts for r in p['bounds']];rel=[r for p in parts for r in p['relations']]
 need(len(orders)==88 and len({x['order_hash'] for x in orders})==88,'full membership');need(len(rel)==880,'10 families')
 return dict(orders=len(orders),markets=len(parts),groups=dict(Counter(x['group'] for x in orders)),raw_events=sum(x['raw_events'] for x in orders),
  refinements=dict(post_A=sum(x['post_A_lo']>x['old_post_lo'] for x in bounds),post_B=sum(x['post_B_lo']>x['old_post_lo'] for x in bounds),end=sum(x['end_hi_conditional']<x['old_end_hi'] for x in bounds),any=sum(x['narrowed'] for x in bounds)),
  bound_status=dict(Counter(x['bound_status'] for x in bounds)),relations={f'R{k:02}':dict(Counter(x['status'] for x in rel if x['relation']==f'R{k:02}')) for k in range(1,11)},
  exhausted_false_pulls=sum(x['exhausted_terminal_pull'] for x in bounds if x['group']=='exhausted'),private_actions_identified=sum(x['terminal_pull_certified'] for x in bounds),pending_display_convention='bracketed_not_identified',
  joint_conflict_frames=sum(len(p['joint']) for p in parts),same_book_pair_competing_records=sum(x['competing_records']>1 for p in parts for x in p['book'] if x['kind']=='match_reflection_candidate'))

def run(mode):
 start=time.process_time();resource.setrlimit(resource.RLIMIT_CPU,(16,18));bind();inputs=input_tables();pairs=json.loads((HERE/'inputs.json').read_text())['pairs'];DATA.mkdir(exist_ok=True)
 if mode=='pilot':
  need(not (DATA/'pilot.json').exists(),'write once pilot');p=market(*pairs[0],inputs,DATA);dump(DATA/'pilot.json',clean(p))
  frames={}
  for x in json.loads((HERE/'inputs.json').read_text())['parquets']:
   if x['kind']=='frames':frames[(x['day'],x['mid'])]=x['rows']
  projection=(time.process_time()-start)+p['normalization']['cpu_seconds']*sum(frames.values())/max(1,frames[tuple(pairs[0])])*2+8
  gate=dict(projected_remaining_all_in_cpu_s=projection,cap_cpu_s=54,run_full=projection<45,pilot_sha=sha(DATA/'pilot.json'),cpu_seconds=time.process_time()-start)
  dump(HERE/'pilot_gate.json',gate);print(json.dumps(gate,indent=2));return
 gate=json.loads((HERE/'pilot_gate.json').read_text());need(gate['run_full'] and sha(DATA/'pilot.json')==gate['pilot_sha'],'pilot gate');out=HERE/'outputs';need(not out.exists(),'write once outputs')
 parts=[json.loads((DATA/'pilot.json').read_text())]
 for d,m in pairs[1:]:parts.append(clean(market(d,m,inputs,DATA)));guard()
 out.mkdir();names={'ledger':'order_ledger.csv','orders':'order_summary.csv','bounds':'chain_bounds.csv','relations':'relation_edges.csv.gz','book':'book_edges.csv.gz','joint':'joint_conflicts.csv'}
 for field,name in names.items():write(out/name,[r for p in parts for r in p[field]],['day','mid','fid','t_ms','side','tick','required_future_shares','capacity_upper','order_hashes'] if field=='joint' else None)
 norm=[p['normalization'] for p in parts];dump(out/'normalization.json',norm);result=summarize(parts)
 result.update(as_of_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),cpu_seconds=time.process_time()-start,hashes={p.name:sha(p) for p in sorted(out.iterdir())})
 dump(out/'result.json',clean(result));print(json.dumps(result,indent=2))
if __name__=='__main__':
 if sys.argv[1]=='freeze':freeze()
 else:run(sys.argv[1])
