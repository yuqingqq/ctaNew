"""Independent raw ledger, cell and conditional-window recount. Does not import builder."""
import collections,copy,csv,gzip,hashlib,importlib.util,json,math,resource,time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
pa.set_cpu_count(1);pa.set_io_thread_count(1)
H=Path(__file__).resolve().parent;ROOT=H.parents[2];O=H/'outputs';T=ROOT/'predictfun/reviews/maker_terminal_chain_20261124';RAW=ROOT/'data/predict_fun/research/maker_terminal_chain_20261124';TI=ROOT/'predictfun/reviews/maker_timing_estimates_20261124/outputs';L=ROOT/'data/predict_fun/research/maker_trace_v2_20261123/life';D=ROOT/'data/predict_fun/research/maker_trace_v1_20261122/depth'
def need(x,m):
 if not x:raise ValueError(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 with (gzip.open if str(p).endswith('.gz') else open)(p,'rt') as f:return list(csv.DictReader(f))
def f(x):return float(x) if x not in ('',None) else math.nan
def eq(a,b):return (math.isnan(a) and math.isnan(b)) or math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-6)
def flag(x):return x is True or x=='True'
def ledger_check(got,es):
 expected={};per=collections.defaultdict(list)
 for e in es:per[e['order_hash']].append(e)
 for h,events in per.items():
  cum=0
  for e in sorted(events,key=lambda x:(x['block_number'],x['transaction_index'],x['log_index'])):
   orig=e['values'][5];before=orig-cum;cum+=e['making'];need(cum<=orig,'overfill');expected[(e['tx'],e['log_index'])]=(h,orig,e['making'],cum,before,orig-cum,e['shares'])
 need(len(got)==len(expected) and len({(x['tx'],int(x['log_index'])) for x in got})==len(got),'ledger membership')
 for x in got:
  actual=(x['order_hash'],*[int(x[k]) for k in ('original_making_wei','making_wei','cumulative_making_wei','remaining_before_wei','remaining_after_wei','fill_shares_wei')]);need(actual==expected[(x['tx'],int(x['log_index']))],'ledger amounts')
  need(x['timeline']=='settlement_order_not_match_order','ledger clock')
def bounds_check(b,expected):
 need(len(b)==len(expected) and len({x['order_hash'] for x in b})==len(expected),'bounds population')
 for r in b:
  e=expected[r['order_hash']]
  for k,v in e.items():
   if isinstance(v,bool):need(flag(r[k])==v,'bounds '+k)
   elif isinstance(v,(int,float)):need(eq(f(r[k]),float(v)),'bounds '+k)
   elif v is None:need(r[k] in ('',None),'bounds '+k)
   else:need(r[k]==v,'bounds '+k)
  need(not flag(r['terminal_pull_certified']) and not flag(r['exhausted_terminal_pull']),'unsupported pull')

def main():
 start=time.process_time();resource.setrlimit(resource.RLIMIT_CPU,(16,18))
 for p,h in json.loads((H/'inputs.json').read_text())['files'].items():need(sha(ROOT/p)==h,'input hash')
 for p,h in json.loads((H/'sources.json').read_text()).items():need(sha(ROOT/p)==h,'source hash')
 result=json.loads((O/'result.json').read_text())
 for p,h in result['hashes'].items():need(sha(O/p)==h,'output hash')
 spec=importlib.util.spec_from_file_location('terminal_check',T/'check.py');tc=importlib.util.module_from_spec(spec);spec.loader.exec_module(tc)
 orig=read(T/'orders.csv');events=json.loads((RAW/'events.json').read_text());manifest=json.loads((T/'manifest.json').read_text());responses={}
 for p in sorted(RAW.glob('batch_*.json')):
  for r in json.loads(p.read_text()):need(r['id'] not in responses,'duplicate response');responses[r['id']]=r['result']
 raw=tc.verify(orig,events,manifest,responses)
 led=read(O/'order_ledger.csv');ledger_check(led,events)
 orders=read(O/'order_summary.csv');bounds=read(O/'chain_bounds.csv');rel=read(O/'relation_edges.csv.gz');books=read(O/'book_edges.csv.gz');joint=read(O/'joint_conflicts.csv')
 need(len(orders)==88 and {r['order_hash'] for r in orders}=={r['order_hash'] for r in orig},'orders membership');by={r['order_hash']:r for r in orders};old={r['order_hash']:r for r in orig};pe=collections.defaultdict(list)
 for e in events:pe[e['order_hash']].append(e)
 posts={r['order_hash']:r for r in read(TI/'post_estimates.csv.gz')};un={r['order_hash']:r for r in read(TI/'union_windows.csv.gz')};signed={r['order_hash']:r for r in read(TI/'signed_chain_estimates.csv')}
 expected={};expected_rel={};expected_joint=[];expected_book={};normrows=0;renamed={};cell_checks=0
 for n in json.loads((O/'normalization.json').read_text()):
  d,m=str(n['day']),int(n['mid']);path=ROOT/n['normalized_file'];need(sha(path)==n['normalized_sha256'],'normalized hash');nr=pd.read_csv(path,float_precision='round_trip');normrows+=len(nr);need(len(nr)==n['normalized_rows'],'normalized rows')
  ff=pd.read_parquet(D/f'{d}_{m}_frames.parquet');lv=pd.read_parquet(D/f'{d}_{m}_levels.parquet');fl=pd.read_parquet(L/f'{d}_{m}_fills.parquet');fm=ff.set_index('fid');positions={int(fid):i for i,fid in enumerate(ff.fid)}
  # Independently convert raw pending units and raw displayed rows for each saved point.
  pend={};bad={}
  for r in ff.itertuples():
   ps=[json.loads(r.pend_bids or '[]'),json.loads(r.pend_asks or '[]')];bad[int(r.fid)]=any(not (isinstance(p,(int,float)) and isinstance(u,(int,float)) and 0<p<1 and math.isfinite(u) and u>=0 and abs(p*100-round(p*100))<=1e-6) for a in ps for p,u in a);pend[int(r.fid)]=ps
  levels={}
  for g in lv.itertuples():
   key=(g.fid,g.side,g.ptick);valid=key not in levels and g.status=='ok' and getattr(g,'x_status','ok')=='ok';levels[key]=float(g.shares) if valid else math.nan
  cells={};members=[h for h,r in by.items() if (r['day'],int(r['mid']))==(d,m)];prep={}
  for (s,k),z in nr.groupby(['side','tick'],sort=False):
   z=z.sort_values(['t_ms','fid']);a=z.to_dict('records');prev=None
   for r in a:
    fid=int(r['fid']);fr=fm.loc[fid];need(int(r['t_ms'])==int(fr.upd_ms) and int(r['recv_ns'])==int(fr.recv_ns),'book clocks')
    own=0 if s=='bid' else 1;us=[sum(float(u) for p,u in pend[fid][j] if round(p*100)==k) for j in (own,1-own)];display=levels.get((fid,s,k),0.)
    if bad[fid]:display=math.nan;us=[math.nan,math.nan]
    values=(display,*us,us[0]/(k/100),us[1]/(k/100),display+sum(us)/(k/100))
    for key,v in zip(('displayed_shares','pending_own_usd','pending_other_usd','pending_own_shares','pending_other_shares','capacity_upper_shares'),values):need(eq(float(r[key]),v),'normalization '+key)
    coherent=bool(prev is not None and 0<r['t_ms']-prev['t_ms']<=1000 and str(fr.conn)==str(fm.loc[int(prev['fid'])].conn) and positions[fid]-positions[int(prev['fid'])]==1 and not bad[fid] and not bad[int(prev['fid'])] and all(math.isfinite(float(v)) for v in (r['displayed_shares'],r['pending_own_shares'],r['pending_other_shares'],prev['displayed_shares'],prev['pending_own_shares'],prev['pending_other_shares'])))
    need(flag(r['coherent_previous'])==coherent,'coherent pair');prev=r;cell_checks+=1
   cells[(s,int(k))]=a
  for h in members:
   r=by[h];p=posts[h];fs=fl[fl.mk_hash==h];v=pe[h][0]['values'];origq=(v[6] if v[10]==0 else v[5])/10**18;rem=int(old[h]['remaining_upper_wei']);remq=rem*(v[6]/v[5] if v[10]==0 else 1)/10**18
   group='exhausted' if rem==0 else ('conditional_withdrawal' if flag(signed[h]['conditional_withdrawal_all_views']) else 'unresolved')
   need(r['group']==group and eq(float(r['original_shares']),origq) and eq(float(r['remaining_quote_shares_upper']),remq),'order quantities')
   need(int(r['maker_fills'])==len(fs) and int(r['raw_events'])==len(pe[h]) and r['native_making_budget']==str(v[5]) and r['remaining_budget_upper']==str(rem),'order accounting')
   need(not flag(r['live_remainder_observed']) and r['pending_per_order']=='unknown_aggregate_only','order interpretation')
   eligible=flag(p['estimate_enabled']) and all(e['role']=='maker' for e in pe[h]) and (fs.x_lo_hard_ms>=0).all() and un[h]['union_status']=='three_view_union';need(flag(r['eligible_conditional_bounds'])==eligible,'order eligibility')
   prep[h]=(r,p,fs,origq,remq,group,eligible)
  conflict=set();future={}
  for cell,pts in cells.items():
   hs=[h for h in members if (by[h]['side'],int(by[h]['tick']))==cell]
   tm=np.array([p['t_ms'] for p in pts]);cap=np.array([p['capacity_upper_shares'] for p in pts]);active={}
   for h in hs:
    fs=prep[h][2];future[h]=sum((np.where(tm<int(f.x_lo_hard_ms),float(f.shares),0.) for f in fs.itertuples()),start=np.zeros(len(tm)))
    active[h]=np.where((tm>=fs.hi_ms.min())&(tm<fs.x_lo_hard_ms.max()),future[h],0.)
   demand=sum(active.values(),start=np.zeros(len(tm)))
   for i in np.flatnonzero(np.isfinite(cap)&(demand>cap+.01*max(1,len(hs)))):
    ah=[h for h in hs if active[h][i]>0];conflict.update(ah);expected_joint.append((d,m,int(pts[i]['fid']),cell[0],cell[1],round(float(demand[i]),6),round(float(cap[i]),6),tuple(sorted(ah))))
  groups={(sid,s,int(k)):g for (sid,s,k),g in fl.groupby(['sid','y_side','y_ptick'],sort=False)}
  for h,(r,p,fs,q,rq,group,eligible) in prep.items():
   pts=cells[(r['side'],int(r['tick']))];olo,ohi=int(p['effect_lo_ms']),int(p['effect_hi_ms']);elo,ehi=int(un[h]['union_lo_ms']),int(un[h]['union_hi_ms']);pa,pb=olo,olo;wa=wb=None;en=ehi;we=None;lo=int(fs.x_lo_hard_ms.min());hi=int(fs.hi_ms.max())
   for i,point in enumerate(pts):
    tm=point['t_ms'];c=point['capacity_upper_shares'];required=future[h][i]
    if olo<=tm<min(lo,ohi) and math.isfinite(c):
     if c+.01<required:pa=int(tm);wa=int(point['fid'])
     if c+.01<q:pb=int(tm);wb=int(point['fid'])
    if group!='exhausted' and rq>.01 and we is None and max(hi,elo)<=tm<=ehi and math.isfinite(c) and c+.01<rq:en=int(tm);we=int(point['fid'])
   can=eligible and h not in conflict and 0<=olo<=ohi and 0<=elo<=ehi;aa=pa if can else olo;bb=pb if can else olo;ee=en if can and group=='conditional_withdrawal' else ehi;changed=aa>olo or bb>olo or ee<ehi
   expected[h]=dict(group=group,old_post_lo=olo,old_post_hi=ohi,post_A_lo=aa,post_B_lo=bb,post_hi_unchanged=ohi,post_A_candidate=pa,post_B_candidate=pb,post_A_witness_fid=wa,post_B_witness_fid=wb,old_end_lo=elo,old_end_hi=ehi,end_hi_conditional=ee,end_candidate=en,end_witness_fid=we,narrowed=changed,bound_status='joint_capacity_conflict' if h in conflict else ('conditional_only' if can else 'prior_grade_or_clock_or_role_refused'),post_lower_open=aa>olo or bb>olo,full_signed_coverage_assumed=bb>olo or ee<ehi)
   match=add=unexpl=0
   for sid in set(fs.sid):
    g=groups[(sid,r['side'],int(r['tick']))];gq=float(g.shares.sum());gl=int(g.x_lo_hard_ms.min());gh=int(g.hi_ms.max())
    for before,after in zip(pts,pts[1:]):
     if not flag(after['coherent_previous']) or after['t_ms']<gl or before['t_ms']>gh:continue
     drop=before['displayed_shares']-after['displayed_shares'];gain=max(after['pending_own_usd']-before['pending_own_usd'],after['pending_other_usd']-before['pending_other_usd']);value=gq*int(r['tick'])/100;pend=gain>1e-9 and gain>=max(0,value-max(.011*value,.01));dep=drop>=max(.01,gq-max(.1*gq,.5))
     if pend or dep:
      match+=1;key=(d,m,'match_reflection_candidate',r['side'],int(r['tick']),sid,int(before['fid']),int(after['fid']),'');expected_book[key]=(gq,drop,gain,'both' if pend and dep else ('pending' if pend else 'drop'))
   fg=fl[(fl.y_side==r['side'])&(fl.y_ptick==int(r['tick']))];glo=fg.x_lo_hard_ms.to_numpy();ghi=fg.hi_ms.to_numpy();gq=fg.shares.to_numpy()
   for before,after in zip(pts,pts[1:]):
    if not flag(after['coherent_previous']):continue
    drop=before['displayed_shares']-after['displayed_shares'];gain=max(after['pending_own_usd']-before['pending_own_usd'],after['pending_other_usd']-before['pending_other_usd'])
    if after['t_ms']>=olo and before['t_ms']<=ohi and drop<-.01:
     add+=1;key=(d,m,'unattributed_add_or_restore',r['side'],int(r['tick']),'',int(before['fid']),int(after['fid']),'');expected_book[key]=(-drop,drop,gain,'zero_to_positive' if before['displayed_shares']==0 else 'net_addition')
    if after['t_ms']>=elo and before['t_ms']<=ehi:
     possible=float(gq[(glo<=after['t_ms'])&(ghi>=before['t_ms'])].sum())
     if drop>possible+max(.5,.1*possible):
      unexpl+=1;key=(d,m,'unexplained_drop_candidate',r['side'],int(r['tick']),'',int(before['fid']),int(after['fid']),h);expected_book[key]=(possible,drop,gain,'recorded_execution_shortfall_not_cancel_proof')
   vals=[('observed_budget_ledger',len(pe[h])),('exhausted' if group=='exhausted' else 'incomplete_budget_not_live',1),('candidate_match_reflection' if match else 'no_visible_match_candidate',match),('unknown_failure_stream',0),('first_fill_execution_bound',1),('repeated_execution_not_continuity' if len(fs)>1 else 'single_recorded_fill',len(fs)),('unattributed_add_restore_candidates' if add else 'no_visible_creation_candidate',add),('conditional_bound_narrowed' if changed else ('refused' if not can else 'no_bound_gain'),int(changed)),('joint_capacity_conflict' if h in conflict else 'no_selected_joint_conflict_no_unique_allocation',match),('unexplained_drop_candidates' if unexpl else 'no_unexplained_drop',unexpl)]
   for i,v in enumerate(vals):expected_rel[(h,f'R{i+1:02}')]=v
  need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1450*1024,'memory')
 bounds_check(bounds,expected)
 need(len(rel)==880 and len({(r['order_hash'],r['relation']) for r in rel})==880,'relations membership')
 for r in rel:need((r['status'],int(r['evidence_count']))==expected_rel[(r['order_hash'],r['relation'])] and not flag(r['private_action_identified']),'relation recount')
 gotjoint=[(r['day'],int(r['mid']),int(r['fid']),r['side'],int(r['tick']),round(float(r['required_future_shares']),6),round(float(r['capacity_upper']),6),tuple(sorted(r['order_hashes'].split('|')))) for r in joint];need(sorted(gotjoint)==sorted(expected_joint),'joint conflicts')
 need(len(books)==len(expected_book),'book edge count');seen=set();comp=collections.defaultdict(set)
 for r in books:
  key=(r['day'],int(r['mid']),r['kind'],r['side'],int(r['tick']),r['sid'],int(r['fid_before']),int(r['fid_after']),r['order_hashes'] if r['kind']=='unexplained_drop_candidate' else '')
  need(key not in seen and key in expected_book,'book edge key');seen.add(key);v=expected_book[key]
  need(all(eq(float(r[c]),v[i]) for i,c in enumerate(('quantity_shares','drop_shares','pending_gain_usd'))) and r['evidence']==v[3],'book edge amount')
  if r['kind']=='match_reflection_candidate':comp[(r['day'],r['mid'],r['side'],r['tick'],r['fid_before'],r['fid_after'])].add(r['sid'])
 for r in books:
  if r['kind']=='match_reflection_candidate':need(int(r['competing_records'])==len(comp[(r['day'],r['mid'],r['side'],r['tick'],r['fid_before'],r['fid_after'])]),'competing book records')
 need(result['groups']==dict(collections.Counter(r['group'] for r in orders)),'group summary')
 need(result['refinements']==dict(post_A=sum(x['post_A_lo']>x['old_post_lo'] for x in expected.values()),post_B=sum(x['post_B_lo']>x['old_post_lo'] for x in expected.values()),end=sum(x['end_hi_conditional']<x['old_end_hi'] for x in expected.values()),any=sum(x['narrowed'] for x in expected.values())),'refinement summary')
 need(result['relations']=={f'R{k:02}':dict(collections.Counter(v[0] for (h,relid),v in expected_rel.items() if relid==f'R{k:02}')) for k in range(1,11)},'relation summary')
 need(result['joint_conflict_frames']==len(joint) and result['private_actions_identified']==0 and result['exhausted_false_pulls']==0,'claims')
 falsifiers={}
 def bad(name,fn):
  try:fn()
  except ValueError as e:falsifiers[name]=str(e);return
  raise AssertionError('falsifier escaped '+name)
 for name,key in [('original_reset','original_making_wei'),('double_count','cumulative_making_wei'),('wrong_remaining','remaining_after_wei')]:
  c=copy.deepcopy(led);c[0][key]=str(int(c[0][key])+1);bad(name,lambda:ledger_check(c,events))
 bad('duplicate_event',lambda:ledger_check(led+led[:1],events));bad('omitted_order',lambda:bounds_check(bounds[1:],expected))
 for name,key,val in [('invented_post','post_A_lo','1'),('invented_pull','terminal_pull_certified','True'),('exhausted_pull','exhausted_terminal_pull','True'),('wrong_clock','post_hi_unchanged','1'),('wrong_grade','bound_status','certified')]:
  c=copy.deepcopy(bounds);c[0][key]=val;bad(name,lambda:bounds_check(c,expected))
 out=dict(pass_all=True,raw=raw,ledger_events=len(led),orders=len(orders),relations=len(rel),normalized_points=cell_checks,book_edges=len(books),joint_conflicts=len(joint),falsifiers=falsifiers,cpu_seconds=time.process_time()-start)
 need(not (H/'checks.json').exists(),'write once');(H/'checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
