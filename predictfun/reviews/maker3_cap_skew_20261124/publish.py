"""Post-run presentation/reconciliation only. No new candidate or selection."""
import collections,copy,gzip,json,math
import numpy as np
import pandas as pd
import experiment as E
H,T=E.H,E.T
assert json.loads((H/'checks_v2.json').read_text())['pass_all']
src=json.loads((H/'replay.json').read_text());totals={};equal=[]
for arm in E.ARMS:
 rows=[x for x in src['rows'] if x['arm']==arm];assert len(rows)==3
 totals[arm]={k:sum(r['metrics'][k] for r in rows) for k in ['net_c','completed_c','residual_c','maker_shares','taker_shares','take_requests','passive_generations','passive_pulls']}
 for key in ['mean_abs_inventory','both_grid_fraction']:totals[arm][key]=float(np.mean([r['metrics'][key] for r in rows]))
 totals[arm]['peak_inventory']=max(r['metrics']['peak_inventory'] for r in rows)
 assert abs(totals[arm]['net_c']-totals[arm]['completed_c']-totals[arm]['residual_c'])<1e-5
fields=['fills','orders','requests','commands','guard_log','mode_log','closed','grid','cash','position','value']
for _,mid,_ in E.X.CASES:
 rr={a:json.loads(gzip.decompress((E.OUT/f'{mid}_{a}.json.gz').read_bytes())) for a in E.ARMS}
 eq={k:rr['BASE'][k]==rr['CAP_ONLY'][k] for k in fields};assert all(eq.values())
 wt={k:rr['WIDE_PASSIVE'][k]==rr['WIDE_TAKE'][k] for k in fields};assert all(wt.values())
 bad=copy.deepcopy(rr['CAP_ONLY']);bad['fills'][0][3]+=1;assert not all(rr['BASE'][k]==bad[k] for k in fields)
 equal.append(dict(mid=mid,cap_only=eq,wide_take_no_effect=wt))
# Positive callback control: wider taking code actually fires above its threshold.
f=[0,0,40,-1,-1,-1,-1,100,0,0,0,0,42,-1,-1,-1,-1,100,0,0,0,0]
b=E.Branch(dict(frames=np.array([f],float),trades=np.empty((0,4)),mark=41.),1000,{},200,E.ARMS['WIDE_TAKE'])
gid=b.E.add9([0,0,40,550,0,-100,-50,-110,0]);b.E.o[gid]['rem']=0;b.E.fills.append((-100.,gid,0,550.,40.,-2));b.pre_tick(0)
b.take_step(0);assert len(b.E.iocs)==1 and b.take_decisions[-1]['status']=='sent'
b.pre_tick(10);assert b.E.exec_log[-1]['filled']==100 and b.E.exec_log[-1]['venue_after']==450
assert b.reserve(10)==(550.,0.,100.);b.pre_tick(110);assert b.reserve(110)==(450.,0.,0.)
orders=pd.read_csv(H.parent/'maker_action_population_20261124/outputs/orders.csv',float_precision='round_trip');orders=orders[orders['rank']==3]
sizes=pd.read_csv(E.OUT/'sizes.csv',float_precision='round_trip');ev=pd.read_csv(E.OUT/'take_events.csv',float_precision='round_trip')
roles=orders.roles.value_counts().to_dict();assert roles=={'maker':33,'taker':1}
quote=orders[orders.roles=='maker'];assert len(quote)==33 and set(quote.original_shares)=={10.,220.}
named=ev[ev.mid==3789484][['sid','k_anchor','t_anchor_ms','x_lo_hard_ms','hi_ms','s','q_all','q0','q1','qp','reduce_5_0','reduce_30_0','add_5_0','add_30_0','take_class','dir_D0','dir_D1000','dir_pend']]
named.to_csv(H/'named_takes.csv',index=False)
facts=dict(totals=totals,exact_equalities=equal,identity_falsifiers=3,wide_take_positive_control=True,
 signed_roles=roles,signed_quote_sizes=quote.original_shares.value_counts().to_dict(),maker_windows_covered=int(sizes[sizes.lag_ms==0].status.eq('sampled_window').sum()),
 wider_net_delta_c=totals['WIDE_PASSIVE']['net_c']-totals['BASE']['net_c'],wider_pooled_positive=totals['WIDE_PASSIVE']['net_c']>0,
 every_market_better=all(next(x for x in src['rows'] if x['mid']==m and x['arm']=='WIDE_PASSIVE')['metrics']['net_c']>next(x for x in src['rows'] if x['mid']==m and x['arm']=='BASE')['metrics']['net_c'] for _,m,_ in E.X.CASES),
 automatic_take_activated=totals['WIDE_TAKE']['take_requests']>0,validation=False)
T.dump(H/'claims.json',facts)
rows=['| Market | Current | Cap only | Wider inventory band | Wider + take |','|---|---:|---:|---:|---:|']
for _,m,_ in E.X.CASES:
 v={x['arm']:x['metrics']['net_c']/100 for x in src['rows'] if x['mid']==m};rows.append('| '+str(m)+' | '+' | '.join(f'{v[a]:+.2f}' for a in E.ARMS)+' |')
rows.append('| Total | '+' | '.join(f"{totals[a]['net_c']/100:+.2f}" for a in E.ARMS)+' |')
(H/'table.md').write_text('\n'.join(rows)+'\n')
print('\n'.join(rows));print(json.dumps(T.clean(facts),allow_nan=False));print(named.to_string(index=False))
