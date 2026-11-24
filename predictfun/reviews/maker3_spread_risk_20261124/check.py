import collections,copy,gzip,json,math,resource
import numpy as np
import pandas as pd
import core as C
import build as B
import risk as K
def verify_case(c,ref):
 fs=c['fills'];ids={f['id']:f for f in fs};assert len(ids)==len(fs)
 cash=sum(-f['s']*f['q']*f['price'] for f in fs);pos=sum(f['s']*f['q'] for f in fs);fees=sum(f['fee_c'] for f in fs);v=c['totals']
 assert abs(cash-v['cash_c'])<1e-5 and abs(pos-v['position'])<1e-6 and abs(fees-v['fee_c'])<1e-7
 assert abs(cash+pos*c['mark']-fees-v['net_c'])<1e-5
 used=collections.Counter();net=gross=fee=sh=0.
 for z in c['closes']:
  a,b=ids[z['entry']],ids[z['exit']];q=z['q'];assert a['s']==-b['s'] and a['t']<=b['t'] and q>0
  gg=a['s']*q*(b['price']-a['price']);ff=q*(a['fee_c']/a['q']+b['fee_c']/b['q'])
  assert abs(gg-z['gross_c'])<1e-6 and abs(ff-z['fee_c'])<1e-6 and abs(gg-ff-z['net_c'])<1e-6
  used[a['id']]+=q;used[b['id']]+=q;gross+=gg;fee+=ff;net+=gg-ff;sh+=q
  for clock in C.CLOCKS:
   zc=z['by_clock'][clock];am=a['book'][clock]['mid'];bm=b['book'][clock]['mid']
   if am is None or bm is None:assert zc['kind']=='unknown' and zc['drift_c'] is None;continue
   aa=a['book'][clock]['spread'];bb=b['book'][clock]['spread'];typ='both_wide' if aa>4+1e-7 and bb>4+1e-7 else 'one_wide' if aa>4+1e-7 or bb>4+1e-7 else 'neither_wide'
   assert typ==zc['kind'];vals=[a['s']*q*(am-a['price']),a['s']*q*(b['price']-bm),a['s']*q*(bm-am)]
   assert all(abs(x-zc[k])<1e-6 for x,k in zip(vals,['entry_edge_c','exit_edge_c','drift_c'])) and abs(sum(vals)-gg)<1e-6
 residual=uf=0.
 for f in fs:
  rem=f['q']-used[f['id']];assert rem>=-1e-7
  residual+=max(0,rem)*f['s']*(c['mark']-f['price']);uf+=max(0,rem)/f['q']*f['fee_c']
 assert abs(net-v['completed_c'])<1e-5 and abs(residual-v['residual_c'])<1e-5 and abs(uf-v['unclosed_fee_c'])<1e-6
 if ref.status=='ok':assert abs(net-ref.close_net*100)<1e-4 and abs(v['net_c']-ref.total*100)<1e-4
 for occ in c['occupancy']:
  expected={'all':c['end']-(ref.start_ms-c['ws'])/1000,'early':180-(ref.start_ms-c['ws'])/1000,'late':c['end']-180,'last60':c['end']-240}[occ['phase']]
  assert abs(sum(occ['seconds'].values())-expected)<1e-6
 return True
def run():
 resource.setrlimit(resource.RLIMIT_CPU,(120,121));B.bound();meta=json.loads((C.H/'build.json').read_text())
 for fn,h in meta['hashes'].items():assert C.sha(C.OUT/fn)==h
 data=json.loads(gzip.decompress((C.OUT/'data.json.gz').read_bytes()));pop=C.population();refs={int(r.m):r for r in pop.itertuples()};assert {x['mid'] for x in data['cases']}==set(refs)
 source=C.load_fills(pop.m);accepted=source[(source.status=='ok')&~source.self_trade&(source.q>0)];nf=0
 for c in data['cases']:
  ref=refs[c['mid']];verify_case(c,ref);raw=accepted[(accepted.mid==c['mid'])&(accepted.settle_ms>=ref.start_ms)&(accepted.settle_ms<=ref.end_ms)]
  rmap={(r.role,r.sid,int(r.mk_idx)):r for r in raw.itertuples()};assert len(rmap)==len(c['fills'])
  tob=pd.read_parquet(C.V/f"tob_{c['day']}.parquet",filters=[('mid','=',c['mid'])]).sort_values(['upd_ms','recv_ns']);ts=tob.upd_ms.to_numpy()
  for f in c['fills']:
   r=rmap[(f['role'],f['sid'],f['mk_idx'])];assert abs(f['q']-r.q)<1e-9 and abs(f['price']-r.p*100)<1e-9 and abs(f['fee_c']-r.fee_fill*100)<1e-9 and f['hash']==r.hash
   for clock,t in [('pre',f['lo_ms']-1 if f['lo_ms'] is not None else None),('lag1',f['lo_ms']-1000 if f['lo_ms'] is not None else None),('upper',f['hi_ms'])]:
    z=f['book'][clock]
    if t is None:assert z['status']=='unknown_time';continue
    i=int(np.searchsorted(ts,t,'right'))-1
    if i<0:assert z['status']=='no_book';continue
    assert z['frame']==int(ts[i]) and z['frame']<=t and abs(z['age']-(t-ts[i]))<1e-6
    if z['status']=='ok':assert z['age']<=1000 and abs(z['spread']-(tob.best_ask.iloc[i]-tob.best_bid.iloc[i])*100)<1e-7
   nf+=1
 # Reaggregate every stored spread cell from fills/close identities.
 keys={};
 for row in data['spread_cells']:keys[(row['family'],row['clock'],row['phase'],row['band'],row['role'])]=row
 for key,row in keys.items():
  family,clock,phase,band,role=key;chosen=[]
  for c in data['cases']:
   for z in c['fills'] if family=='fills' else c['closes']:
    t=z['t'] if family=='fills' else z['exit_t'];phaseok=phase=='all' or phase=='early' and t<180 or phase=='late' and t>=180 or phase=='last60' and t>=240
    b=C.band(z['book'][clock]['spread']) if family=='fills' else z['by_clock'][clock]['kind'];r=z['role'] if family=='fills' else z['entry_role']+'->'+z['exit_role']
    if phaseok and band==b and role==r:chosen.append(z)
  assert abs(sum(x['q'] for x in chosen)-row['shares'])<1e-6
  if family=='closes':
   assert abs(sum(x['net_c'] for x in chosen)-row['net_c'])<1e-5 and abs(sum(max(0,x['net_c']) for x in chosen)-row['positive_c'])<1e-5
 # Known-bad identity, money, component, band, source time and quantity.
 base=next(x for x in data['cases'] if x['closes']);ref=refs[base['mid']];caught=0
 mods=[lambda x:x['totals'].__setitem__('net_c',x['totals']['net_c']+1),lambda x:x['fills'][0].__setitem__('q',x['fills'][0]['q']+1),lambda x:x['closes'][0].__setitem__('net_c',999999),lambda x:x['closes'][0]['by_clock']['pre'].__setitem__('kind','false')]
 for fn in mods:
  bad=copy.deepcopy(base);fn(bad)
  try:verify_case(bad,ref)
  except (AssertionError,TypeError):caught+=1
 assert caught==len(mods)
 risk=json.loads((C.H/'risk.json').read_text());p=pd.read_parquet(C.OUT/'risk_states.parquet');assert len(p)==5700 and len(risk['rows'])==10
 events=pd.read_parquet(C.C/'take_logic_v2/events.parquet',filters=[('rank','=',3)]);inds,raw=K.schedule(events,p)
 for row in risk['rows']:
  d='0' if row['clock']=='D0' else '1';q=p['q'+d].to_numpy();y=K.outcome(inds,raw,q);assert np.array_equal(y,p['O1_'+row['clock']].to_numpy(bool))
  feat={'adding_burst_100':(p['add_5_'+d]-p['reduce_5_'+d]).to_numpy()>=100,'no_passive_reduction_5s':p['reduce_5_'+d].to_numpy()<=1e-9,'btc_against_1bp':np.sign(q)*p['r_5s'+d].to_numpy()<=-1,'profitable_exit_1c':np.where(np.isfinite(p['exit_edge_'+d]),p['exit_edge_'+d]>1,np.nan),'wide_book_4c':(p['ask'+d]-p['bid'+d]).to_numpy()*100>4+1e-7}[row['feature']]
  valid=p.clean.to_numpy(bool)&(abs(q)>1e-6)&p['kok'+d].to_numpy(bool)&p['bok'+d].to_numpy(bool)&np.isfinite(feat);code,n=K.encode(p,q);num=den=0.;n1=n0=e1=e0=0
  for c in set(code[valid]):
   a=valid&(code==c)&(feat==1);b=valid&(code==c)&(feat==0);aa=int(a.sum());bb=int(b.sum())
   if not aa or not bb:continue
   w=aa*bb/(aa+bb);num+=w*(y[a].mean()-y[b].mean());den+=w;n1+=aa;n0+=bb;e1+=int(y[a].sum());e0+=int(y[b].sum())
  expected=num/den if den else None;assert expected is None and row['matched_RD'] is None or expected is not None and abs(expected-row['matched_RD'])<1e-12
  assert (n1,n0,e1,e0)==(row['on_seconds'],row['off_seconds'],row['on_events'],row['off_events'])
 C.dump(C.H/'checks.json',dict(pass_all=True,markets=19,raw_fills_checked=nf,cells=len(keys),risk_screens=10,corruptions_refused=caught));print('PASS',nf,'raw fills;',len(keys),'cells;10 risk screens;',caught,'corruptions')
if __name__=='__main__':run()
