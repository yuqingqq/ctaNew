"""Post-run presentation of frozen observations; no fitted/tuned rule."""
import collections,gzip,json
import numpy as np
import pandas as pd
import core as C
data=json.loads(gzip.decompress((C.OUT/'data.json.gz').read_bytes()));summary={};wide=[];activity=[];late=[];locations=[]
for clock in C.CLOCKS:
 cells=[r for r in data['spread_cells'] if r['family']=='closes' and r['phase']=='all' and r['clock']==clock];den=sum(x['positive_c'] for x in cells)
 for label in ['both_wide','one_wide','neither_wide','unknown']:
  a=[r for r in cells if r['band']==label];wide.append(dict(clock=clock,kind=label,matched_shares=sum(r['shares'] for r in a),net_usd=sum(r['net_c'] for r in a)/100,positive_usd=sum(r['positive_c'] for r in a)/100,positive_fraction=sum(r['positive_c'] for r in a)/den))
for scope,cc in [('all19',data['cases']),('3789484',[c for c in data['cases'] if c['mid']==3789484])]:
 for clock in C.CLOCKS:
  for phase in ['all','early','late','last60']:
   ff=[f for c in cc for f in c['fills'] if phase=='all' or phase=='early' and f['t']<180 or phase=='late' and f['t']>=180 or phase=='last60' and f['t']>=240]
   cl=[z for c in cc for z in c['closes'] if phase=='all' or phase=='early' and z['exit_t']<180 or phase=='late' and z['exit_t']>=180 or phase=='last60' and z['exit_t']>=240]
   total_sec=sum(sum(x['seconds'].values()) for c in cc for x in c['occupancy'] if x['phase']==phase)
   for band in ['<=1','1-2','2-4','>4','unknown']:
    v=[f for f in ff if C.band(f['book'][clock]['spread'])==band];seconds=sum(x['seconds'].get(band,0) for c in cc for x in c['occupancy'] if x['phase']==phase)
    activity.append(dict(scope=scope,clock=clock,phase=phase,band=band,seconds=seconds,time_fraction=seconds/total_sec,legs=len(v),shares=sum(f['q'] for f in v),passive_shares=sum(f['q'] for f in v if f['role']=='maker')))
   valid=[z for z in cl if z['by_clock'][clock]['kind']!='unknown'];late.append(dict(scope=scope,clock=clock,phase=phase,legs=len(ff),shares=sum(f['q'] for f in ff),maker_shares=sum(f['q'] for f in ff if f['role']=='maker'),taker_shares=sum(f['q'] for f in ff if f['role']=='taker'),matched_shares=sum(z['q'] for z in cl),net_turnover_usd=sum(z['net_c'] for z in cl)/100,component_valid_shares=sum(z['q'] for z in valid),edge_usd=sum(z['by_clock'][clock]['entry_edge_c']+z['by_clock'][clock]['exit_edge_c'] for z in valid)/100,drift_usd=sum(z['by_clock'][clock]['drift_c'] for z in valid)/100,valid_fees_usd=sum(z['fee_c'] for z in valid)/100))
  for phase in ['all','late']:
   g=collections.defaultdict(lambda:collections.Counter())
   for c in cc:
    for f in c['fills']:
     if f['role']!='maker' or phase=='late' and f['t']<180:continue
     b=f['book'][clock];band=C.band(b['spread']);dist=b['bid']-f['price'] if b['status']=='ok' and f['s']==1 else f['price']-b['ask'] if b['status']=='ok' else None
     loc='unknown' if dist is None else 'at_touch' if abs(dist)<1e-7 else 'behind_touch' if dist>0 else 'inside_touch_or_clock_move'
     g[(band,loc)]['legs']+=1;g[(band,loc)]['shares']+=f['q']
   locations.extend(dict(scope=scope,clock=clock,phase=phase,band=k[0],location=k[1],**v) for k,v in sorted(g.items()))
trend=data['trend'];m=next(x for x in data['cases'] if x['mid']==trend['mid']);lo,hi=trend['start_s'],trend['end_s'];tf=[f for f in m['fills'] if lo<=f['t']<=hi];tc=[z for z in m['closes'] if lo<=z['exit_t']<=hi]
def position(c,t):return sum(f['s']*f['q'] for f in c['fills'] if f['t']<=t)
trend_info=dict(**trend,legs=len(tf),maker_legs=sum(f['role']=='maker' for f in tf),taker_legs=sum(f['role']=='taker' for f in tf),buy_sh=sum(f['q'] for f in tf if f['s']==1),sell_sh=sum(f['q'] for f in tf if f['s']==-1),inventory_start=position(m,lo),inventory_end=position(m,hi),close_net_usd=sum(z['net_c'] for z in tc)/100,
 phases=[dict(lo=a,hi=b,buy_sh=sum(f['q'] for f in tf if a<=f['t']<b and f['s']==1),sell_sh=sum(f['q'] for f in tf if a<=f['t']<b and f['s']==-1),takes=sum(f['role']=='taker' for f in tf if a<=f['t']<b)) for a,b in [(lo,lo+10),(lo+10,lo+20),(lo+20,hi+.001)]])
summary=dict(wide=wide,late=late,trend=trend_info,totals={key:sum(c['totals'][key] for c in data['cases']) for key in ['net_c','completed_c','residual_c','unclosed_fee_c','matched_shares','fee_c']})
fr={r['clock']:r['positive_fraction'] for r in wide if r['kind']=='both_wide'};summary['mainly_wide_supported']=fr['pre']>.5 and fr['upper']>.5;summary['clock_sensitive_mainly_wide']=(fr['pre']>.5)!=(fr['upper']>.5)
for name,rows in [('activity.csv',activity),('fill_locations.csv',locations),('late.csv',late),('wide.csv',wide)]:pd.DataFrame(rows).to_csv(C.H/name,index=False)
C.dump(C.H/'readings.json',summary);C.dump(C.H/'trend_fills.json',dict(market=trend_info,fills=tf))
print(json.dumps(C.clean(summary),allow_nan=False))
