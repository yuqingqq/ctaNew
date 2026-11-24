"""Independent descriptive checks plus per-day size/flow context (not a cap fit)."""
import collections,csv,gzip,json,math,resource
from pathlib import Path
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent;R=H.parents[2];D=R/'data/predict_fun/research'
resource.setrlimit(resource.RLIMIT_CPU,(15,16))
s=json.loads((H/'summary.json').read_text());f=pd.read_parquet(D/'maker3_late_activity_20261124/actor_legs.parquet')
z=f[f.mid.eq(3789484)].sort_values('settle_ms');p=0.;mx=mn=0
for _,gg in z.groupby('settle_ms'):
    p+=sum(q if side=='bid' else -q for side,q in zip(gg.y_side,gg.q));mx=max(mx,p);mn=min(mn,p)
e=s['flow']['example'][0];assert abs(mx-e['max'])<1e-6 and abs(mn-e['min'])<1e-6 and abs(p-e['last'])<1e-6
# Tick-independent union checker: evaluate each elementary interval of exact
# starts/stops from previously verified chart orders, then add lengths.
with gzip.open(R/'predictfun/reviews/opening_chart_20261124/data.json.gz','rt') as h:chart=json.load(h)
def verify(row):
    c=next(c for c in chart['cases'] if c['id']==row['mid']);orders=c['arms'][row['arm']]['orders'];lo=240 if row['window']=='last_minute' else 0;hi=c['end']
    oo=[o for o in orders if not o['refused'] and o['act']<hi and o['stop']>lo]
    cuts=sorted({lo,hi}|{max(lo,o['act']) for o in oo}|{min(hi,o['stop']) for o in oo});duration=[0.,0.];either=both=0.
    for a,b in zip(cuts,cuts[1:]):
        mid=(a+b)/2;live=[any(o['side']==side and o['act']<=mid<o['stop'] for o in oo) for side in [0,1]]
        for side in [0,1]:duration[side]+=(b-a)*live[side]
        either+=(b-a)*any(live);both+=(b-a)*all(live)
    for key,v in zip(['bid_live_s','ask_live_s','either_s','both_s'],[*duration,either,both]):assert abs(v-row[key])<1e-6
    ff=[q for q in c['arms'][row['arm']]['fills'] if lo<=q['t']<=hi];assert abs(sum(q['q'] for q in ff)-row['filled_shares'])<1e-5
for r in s['own']:
    if r['arm']!='ORIGINAL':verify(r)
bad=dict(next(r for r in s['own'] if r['arm']=='OPENING_ONLY'));bad['either_s']+=1
try:verify(bad)
except AssertionError:pass
else:raise AssertionError('uptime corruption escaped')
signed=pd.read_csv(R/'predictfun/reviews/maker_action_population_20261124/outputs/orders.csv');signed=signed[signed['rank']==3]
o=f[(f.role=='maker')&(f.t_rel_s>=0)&(f.t_rel_s<300)].groupby(['day','mid','hash']).q.sum().reset_index()
ex=pd.read_csv(H/'maker3_market_flow_extremes.csv');out=[]
for day in sorted(o.day.unique()):
    a=o[o.day==day];b=signed[signed.day==int(day)];e=ex[ex.day==int(day)]
    out.append(dict(day=day,filled_orders=len(a),recorded_exactly_10=int(np.isclose(a.q,10,atol=1e-6,rtol=0).sum()),recorded_at_least_219=int(a.q.ge(219).sum()),
        signed_size_counts=b.original_shares.value_counts().sort_index().to_dict(),market_peak_abs_median=float(e.max_abs.median()),market_peak_abs_max=float(e.max_abs.max()),
        markets_peak_at_least_2990=int(e.max_abs.ge(2990).sum()),markets=len(e)))
(H/'context.json').write_text(json.dumps(out,indent=2)+'\n')
(H/'checks.json').write_text(json.dumps(dict(pass_all=True,independent_occupancy_arms=6,windows=12,flow_example=True,known_bad_uptime_refused=True),indent=2)+'\n')
print(json.dumps(out,indent=2))
