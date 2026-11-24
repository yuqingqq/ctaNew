"""Saved-trace projection only. Does not import or run a strategy/simulator."""
import base64,collections,csv,gzip,hashlib,io,json,math,sys,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
H=Path(__file__).resolve().parent;R=H.parents[2];D=R/'data/predict_fun/research';MT=D/'maker_trace_v2_20261123';L=D/'ladder_async_fast_v1_20261122';MR=R/'predictfun/reviews/maker_trace_20261122';AR=R/'predictfun/reviews/ladder_async_fast_20261122';CASES=[('20261024',3702960,'rest','Good fills with short inferred post windows'),('20261026',3789484,'pilot','Bid repricing during an adverse move'),('20261027',3877472,'pilot','Repeated good fills on the same ask')];ARMS=['CURRENT_ASYNC_FAST10','FRONT_ASYNC_FAST10'];REASONS=['post','keep','inventory_pull','restricted','zero_desired','exposure_full','no_room','cap','near_end','own_cross_blocked','unavailable_price','fallback','pressure_pull','pressure_withhold'];ACTIONS=['POST','WAIT','KEEP','PULL']
def read(p):
 with (gzip.open if str(p).endswith('.gz') else open)(p,'rt') as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(tuple,list,np.ndarray)):return [clean(v) for v in x]
 if isinstance(x,(float,np.floating)):return round(float(x),6) if math.isfinite(x) else None
 if isinstance(x,np.integer):return int(x)
 if isinstance(x,np.bool_):return bool(x)
 return x
def dump(p,x):p.write_text(json.dumps(clean(x),separators=(',',':'),allow_nan=False)+'\n')
def paths():
 ps=[H/'extract.py',H/'check.py',H/'contract.md',H/'approval.md',AR/'manifest.json',MR/'v2/matched_gap_r1/per_market.csv',MT/'econ/fills_valued.parquet',MT/'econ/identities.json',R/'predictfun/reviews/maker_timing_estimates_20261124/outputs/post_estimates.csv.gz',R/'predictfun/reviews/maker_action_population_20261124/outputs/orders.csv',MR/'v2/chains/post_fill/orders.csv']
 for day,mid,part,_ in CASES:
  ps += [L/part/(day+x) for x in ['.json','.jsonl.gz','_arrays.npz']]+[MT/'depth'/f'{day}_{mid}_{x}.parquet' for x in ['frames','levels']]+[MT/'life'/f'{day}_{mid}_fills.parquet',MT/'btc'/f'btc_{day}.parquet']
 return list(dict.fromkeys(ps))
def presence(lo,hi,intervals):
 if lo is None or hi is None or hi<lo:return 'unknown'
 spans=sorted((a,b) for a,b in intervals if b>lo and a<=hi and b>a)
 if not spans:return 'none'
 cursor=lo
 for a,b in spans:
  if a>cursor:return 'partial'
  cursor=max(cursor,b)
  if cursor>hi:return 'throughout'
 return 'partial'
def selftest():
 assert presence(1,2,[(0,3)])=='throughout';assert presence(1,2,[(0,2)])=='partial';assert presence(1,2,[(0,1)])=='none';assert presence(1,2,[(0,1.5),(1.5,3)])=='throughout';assert presence(None,2,[])=='unknown';assert presence(1,2,[(0,1.3),(1.4,3)])=='partial'
 # A sell at60 cents is profitable at a50-cent mark; no cents/USD confusion.
 assert (-1)*10*(50-60)==100;print('interval/tie/missing/price controls pass')
def freeze():
 selftest();assert not (H/'manifest.json').exists();dump(H/'manifest.json',dict(version='market_microscope_v1',cases=CASES,cap_cpu_seconds=180,files={str(p.relative_to(R)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in paths()}));print('source/input manifest written')
def load_records():
 out={};manifest=json.loads((AR/'manifest.json').read_text());mh=sha(AR/'manifest.json')
 for day,mid,part,_ in CASES:
  hdr=json.loads((L/part/(day+'.json')).read_text());assert hdr['manifest_sha256']==mh and hdr['status']=='DAY_DONE';assert hdr['sha256']['gz']==sha(L/part/(day+'.jsonl.gz'));assert hdr['sha256']['npz']==sha(L/part/(day+'_arrays.npz'))
  with gzip.open(L/part/(day+'.jsonl.gz'),'rt') as f:
   for line in f:
    row=json.loads(line)
    if row['m']==mid:out[mid]=row
 assert len(out)==3;return out
def top_fills():
 p=pq.ParquetFile(MT/'econ/fills_valued.parquet');cols=['day','mid','sid','mk_idx','role','addr','hash','y_side','s','q','p','self_trade','settle_ms','fee_fill','status','st_1s','v_1s','st_10s','v_10s','st_60s','v_60s'];names=p.schema_arrow.names;j=names.index('mid');mids=[x[1] for x in CASES];orders=read(MR/'v2/chains/post_fill/orders.csv');rank={r['addr'].lower():int(r['rank']) for r in orders};ids=json.loads((MT/'econ/identities.json').read_text());assert set(rank)=={r['addr'].lower() for r in ids['ex_ante']['top10_shares'][:3]};parts=[]
 for i in range(p.metadata.num_row_groups):
  st=p.metadata.row_group(i).column(j).statistics
  if st and not any(st.min<=m<=st.max for m in mids):continue
  t=p.read_row_group(i,columns=cols);mask=pc.and_(pc.is_in(t['mid'],value_set=pa.array(mids,pa.int64())),pc.is_in(pc.utf8_lower(t['addr']),value_set=pa.array(list(rank))));parts+=t.filter(mask).to_pylist()
 return parts,rank
def book(day,mid,ws,end):
 fr=pq.read_table(MT/'depth'/f'{day}_{mid}_frames.parquet').to_pandas();lv=pq.read_table(MT/'depth'/f'{day}_{mid}_levels.parquet',columns=['fid','side','ptick','shares']).to_pandas();assert fr.fid.is_unique
 fr=fr.sort_values(['upd_ms','recv_ns','fid']).reset_index(drop=True);lookup={int(fid):i for i,fid in enumerate(fr.fid)};fi=lv.fid.map(lookup).to_numpy(int);s=(lv.side.to_numpy()=='ask').astype(int);ticks=lv.ptick.to_numpy(int);valid=(ticks>=0)&(ticks<=100);mat=np.zeros((len(fr),2,101),np.float32);mat[fi[valid],s[valid],ticks[valid]]=lv.shares.to_numpy(float)[valid]
 bp=np.max(np.where(mat[:,0,:]>0,np.arange(101),-1),axis=1);ap=np.min(np.where(mat[:,1,:]>0,np.arange(101),101),axis=1);t=(fr.upd_ms.to_numpy(float)-ws)/1000;bad=fr.status.str.contains('bad_frame|mid_mismatch|out_of_order|off_grid',regex=True).to_numpy();good=(bp>=0)&(ap<=100)&(ap>=bp)&~bad
 grid=np.arange(0,end+1e-6,.1);ix=np.searchsorted(t,grid,'right')-1;bars=[]
 for g,i in zip(grid,ix):
  ok=i>=0 and good[i] and g-t[i]<=1.;bars.append([g,float(bp[i]) if ok else None,float(ap[i]) if ok else None,float(mat[i,0,bp[i]]) if ok else None,float(mat[i,1,ap[i]]) if ok else None,((fr.recv_ns.iloc[i]/1e6-ws)/1000) if i>=0 else None,t[i] if i>=0 else None])
 depth=[]
 for g in np.arange(0,end+1e-6,.5):
  i=np.searchsorted(t,g,'right')-1
  if i<0 or not good[i] or g-t[i]>1:depth.append([g,None,None]);continue
  bids=[(p,float(mat[i,0,p])) for p in range(bp[i],max(-1,bp[i]-7),-1) if mat[i,0,p]>0];asks=[(p,float(mat[i,1,p])) for p in range(ap[i],min(101,ap[i]+7)) if mat[i,1,p]>0];depth.append([g,bids,asks])
 btc=pq.read_table(MT/'btc'/f'btc_{day}.parquet',filters=[('g_ms','>=',int(ws-5000)),('g_ms','<=',int(ws+end*1000))]).to_pandas();btc=btc.sort_values('g_ms');bt=btc.g_ms.to_numpy(float);px=btc.mid.to_numpy(float);status=btc.status.to_numpy();bi=np.searchsorted(bt,ws+grid*1000,'right')-1;base=px[max(0,np.searchsorted(bt,ws,'right')-1)];bs=[]
 for g,i in zip(grid,bi):
  valid=i>=0 and status[i]=='ok' and math.isfinite(px[i]) and px[i]>0;bs.append([g,10000*(px[i]/base-1) if valid else None,px[i] if valid else None])
 return bars,depth,bs
def own_arm(rec,z,mid,arm,ws):
 start=(rec['ident']['start_abs_ms']-ws)/1000;end=(rec['ident']['end_abs_ms']-ws)/1000;fills=rec['fills'];cash=pos=0.;by=collections.defaultdict(list)
 for i,f in enumerate(fills):
  t,g,s,q,p,tr=f;sg=1 if s==0 else -1;cash-=sg*q*p;pos+=sg*q;by[int(g)].append(i)
 assert abs(cash-rec['cash'])<1e-6 and abs(pos-rec['position'])<1e-6 and abs(cash+pos*rec['mark']-rec['value'])<1e-6
 prefix=f'{mid}__{arm}__';cm=z[prefix+'commands'];commands=[[start+c[0]/1000,int(c[1]),int(c[3]),int(c[4]),None if not math.isfinite(c[5]) else float(c[5]),None if not math.isfinite(c[6]) else float(c[6]),ACTIONS[int(c[7])],REASONS[int(c[8])]] for c in cm];reasons={c[2]:c[7] for c in commands if c[6]=='PULL'};orders=[];qv=rec['queue_venue']
 for o in rec['orders']:
  g=o['gid'];cum=0.;full=end
  for i in by[g]:
   f=fills[i];cum+=f[3]
   if cum>=o['size']-1e-8:full=min(full,start+f[0]/1000);break
  stop=min(end,start+o['remove']/1000,full);reason=reasons.get(g,o['why'] or ('filled' if cum>=o['size']-1e-8 else 'END'));qs={k:([start+v[0]/1000,v[1],v[2]] if (v:=qv[k].get(str(g))) is not None else None) for k in ('ahead0','first','pull')}
  orders.append(dict(gid=g,side=o['side'],rung=o['rung'],price=o['price'],size=o['size'],remaining=o['rem'],submit=start+o['submit']/1000,act=start+o['act']/1000,stop=stop,remove=min(end+1,start+o['remove']/1000),request=None if o['req'] is None else start+o['req']/1000,ack=rec['closed'].get(str(g)),refused=o['refused'],reason=reason,queue=qs,fill_ids=by[g],end_value=sum((1 if fills[i][2]==0 else -1)*fills[i][3]*(rec['mark']-fills[i][4]) for i in by[g])/100))
 ff=[]
 for i,(f,mm) in enumerate(zip(fills,rec['fill_mids'])):
  t,g,s,q,p,tr=f;sg=1 if s==0 else -1;markouts={str(h):(sg*(mm[2*j]-p) if mm[2*j] is not None else None) for j,h in enumerate([0,50,100,1000,5000,30000])};ff.append(dict(id=i,t=start+t/1000,gid=g,side=s,q=q,price=p,trade=tr,markouts=markouts,end_value=sg*q*(rec['mark']-p)/100,after_request=orders[g]['request'] is not None and orders[g]['request']<start+t/1000<orders[g]['remove']))
 pose=z[prefix+'pose'];posc=z[prefix+'posc'];cc=z[prefix+'cash'];mm=z[prefix+'mid'];mode=z[prefix+'mode'];nva=z[prefix+'nva'];curve=[[start+i*.05,float(pose[i]),float(posc[i]),float((cc[i]+pose[i]*mm[i])/100) if math.isfinite(mm[i]) else None,int(mode[i]),int(nva[i,0]),int(nva[i,1])] for i in range(0,len(pose),4)];curve.append([end,pos,None,rec['value']/100,None,None,None])
 # Compress identical contiguous slot reasons; keep POST/PULL commands exact.
 waits=collections.defaultdict(list)
 for c in commands:
  if c[6]=='WAIT':waits[(c[1],c[3],c[7])].append(c[0])
 wi=[]
 for (s,r,why),ts in waits.items():
  a=b=ts[0]
  for t in ts[1:]:
   if t-b>.050001:wi.append([a,b+.05,s,r,why]);a=t
   b=t
  wi.append([a,min(end,b+.05),s,r,why])
 return dict(label=arm,orders=orders,fills=ff,commands=[c for c in commands if c[6] in ('POST','PULL')],waits=wi,curve=curve,guards=rec['guard_log'],summary=dict(pnl_usd=rec['value']/100,position=pos,filled_shares=sum(f[3] for f in fills),generations=len(orders),accepted=sum(not o['refused'] for o in orders),cancels=sum(o['request'] is not None for o in orders),cancel_reasons=dict(collections.Counter(o['reason'] for o in orders if o['request'] is not None)),both_live_fraction=float(np.mean((nva[:,0]>0)&(nva[:,1]>0))),front_treatment=arm.startswith('FRONT')))
def main():
 assert not (H/'data.json.gz').exists();manifest=json.loads((H/'manifest.json').read_text())
 for name,v in manifest['files'].items():assert sha(R/name)==v['sha256'],name
 recs=load_records();vals,ranks=top_fills();posts={r['order_hash']:r for r in read(R/'predictfun/reviews/maker_timing_estimates_20261124/outputs/post_estimates.csv.gz')};signed={r['order_hash']:r for r in read(R/'predictfun/reviews/maker_action_population_20261124/outputs/orders.csv')};accepted=read(MR/'v2/matched_gap_r1/per_market.csv');allm=[]
 for day,mid,part,title in CASES:
  rec=recs[mid];curr=rec[ARMS[0]];ws=curr['ident']['opening_ms'];end=(curr['ident']['end_abs_ms']-ws)/1000;start=(curr['ident']['start_abs_ms']-ws)/1000;z=np.load(L/part/(day+'_arrays.npz'));ours={arm:own_arm(rec[arm],z,mid,arm,ws) for arm in ARMS};z.close();bars,dep,btc=book(day,mid,ws,end)
  life=pq.read_table(MT/'life'/f'{day}_{mid}_fills.parquet').to_pylist();li={(r['sid'],int(r['mk_idx']),r['mk_hash']):r for r in life};theirs=[];count=collections.Counter()
  for v in vals:
   if int(v['mid'])!=mid:continue
   assert str(v['day'])==day;count['source']+=1;t=(v['settle_ms']-ws)/1000
   if not start<=t<=end:count['outside_interval']+=1;continue
   if v['status']!='ok' or v['self_trade']:count['excluded_'+('self_trade' if v['self_trade'] else v['status'])]+=1;continue
   rank=ranks[v['addr'].lower()];p=posts.get(v['hash']);l=li.get((v['sid'],int(v['mk_idx']),v['hash'])) if v['role']=='maker' else None;flo=(l['x_lo_hard_ms']-ws)/1000 if l and l['x_lo_hard_ms']>=0 else None;side=0 if v['y_side']=='bid' else 1;price=v['p']*100;pp=[(float(p[k])-ws)/1000 for k in ['effect_lo_ms','effect_hi_ms']] if p and p['estimate_enabled']=='True' else None
   if l:assert abs(l['shares']-v['q'])<1e-7 and l['y_side']==v['y_side'] and l['hi_ms']==v['settle_ms']
   g=signed.get(v['hash']);f=dict(id=len(theirs),t=t,lo=flo,rank=rank,role=v['role'],hash=v['hash'],sid=v['sid'],mk_idx=int(v['mk_idx']),side=side,price=price,q=v['q'],fee=v['fee_fill'],post=pp,post_grade=p['grade'] if p else 'missing',original_size=float(g['original_shares']) if g and g['original_shares'] else None,markouts={h:100*v['v_'+h]/v['q'] if v['st_'+h]=='ok' and v['q']>0 else None for h in ['1s','10s','60s']},end_value=v['s']*v['q']*(curr['mark']/100-v['p'])-v['fee_fill'],presence={})
   for arm,oa in ours.items():
    eligible=[o for o in oa['orders'] if not o['refused'] and o['side']==side];same=[(o['act'],o['stop']) for o in eligible if abs(o['price']-price)<1e-7];better=[(o['act'],o['stop']) for o in eligible if (o['price']>=price-1e-7 if side==0 else o['price']<=price+1e-7)];f['presence'][arm]=dict(same=presence(flo,t,same),equal_or_better=presence(flo,t,better))
   theirs.append(f)
  theirs.sort(key=lambda f:(f['t'],f['sid'],f['mk_idx'],f['role']))
  for i,f in enumerate(theirs):f['id']=i
  totals={str(rank):sum(f['end_value'] for f in theirs if f['rank']==rank) for rank in [1,2,3]}
  for rank in [1,2,3]:
   a=next(x for x in accepted if int(x['m'])==mid and x['party']==f'rank{rank}' and x['view']=='zero');assert abs(totals[str(rank)]-float(a['total']))<1e-5
  plot=[]
  for rank in [1,2,3]:
   fs=[f for f in theirs if f['rank']==rank];j=0;cash=pos=0.;curve=[]
   for t,b,a,*_ in bars[::2]:
    while j<len(fs) and fs[j]['t']<=t+1e-7:
     f=fs[j];sg=1 if f['side']==0 else -1;pos+=sg*f['q'];cash-=sg*f['q']*f['price']/100+f['fee'];j+=1
    curve.append([t,pos,(cash+pos*(a+b)/200) if a is not None and b is not None else None])
   curve.append([end,sum((1 if f['side']==0 else -1)*f['q'] for f in fs),totals[str(rank)]]);plot.append(dict(rank=rank,curve=curve))
  allm.append(dict(id=mid,day=day,title=title,ws=ws,start=start,end=end,mark=curr['mark'],book=bars,depth=dep,btc=btc,ours=ours,makers=theirs,maker_curves=plot,maker_totals=totals,source_counts=dict(count),profiles=curr['profile']))
 data=dict(version='market_microscope_v1',asof=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),cases=allm,notes={'clock':'Our assumed execution time vs maker settlement upper bound; whiskers show hard maker match window.','queue':'Initial ahead is activation-book seeded; no continuous maker queue rank is identified.','value':'END compares zero-start numerical intervals using the same mark; maker recorded fees, ours zero maker fee.','population':'Three previously discussed consumed markets, selected illustrations.'});payload=json.dumps(clean(data),separators=(',',':'),allow_nan=False).encode();compressed=gzip.compress(payload,compresslevel=6,mtime=0);(H/'data.json.gz').write_bytes(compressed);(H/'data.js').write_text('window.MM_DATA_GZIP="'+base64.b64encode(compressed).decode()+'";\n');dump(H/'extract_summary.json',dict(markets=[dict(id=m['id'],our=m['ours'][ARMS[0]]['summary'],front=m['ours'][ARMS[1]]['summary'],makers=m['maker_totals'],counts=m['source_counts']) for m in allm],payload_bytes=len(payload),compressed_bytes=len(compressed)));print((H/'extract_summary.json').read_text())
if __name__=='__main__':freeze() if '--freeze' in sys.argv else main()
