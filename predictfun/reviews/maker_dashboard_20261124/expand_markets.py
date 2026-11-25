"""Extend the dashboard from saved market paths and public books."""
import base64
import collections
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import extract_v2 as source
from extract_v2 import ARMS, H, L, MR, MT, R, book, own_arm, presence, read


def choose_cases():
    panel = pq.read_table(MR / 'v2/chains/take_logic_v2/panel.parquet',
                          columns=['mid','k','rank','bid0','ask0','kok0'],
                          filters=[('rank','=',3)]).to_pandas()
    moves = []
    for mid, group in panel.groupby('mid'):
        if mid in {case[1] for case in source.CASES}:
            continue
        group = group.sort_values('k')
        assert list(group.k) == list(range(300))
        prices = ((group.bid0 + group.ask0) * 50).where(group.kok0.astype(bool)).to_numpy()
        for start in range(0, 269, 5):
            window = prices[start:start+31]
            if np.isfinite(window).all():
                delta = window[-1] - window[0]
                distance = abs(np.diff(window)).sum()
                if distance and abs(delta) >= 10 and abs(delta)/distance >= .6:
                    moves.append((float(delta), int(mid), start))
    chosen = {}
    # Four distinct up markets and three distinct down markets, alternating
    # to give each direction a chance at markets with two strong swings.
    for i in range(7):
        sign = 1 if i % 2 == 0 else -1
        delta, mid, start = min((v for v in moves if v[1] not in chosen and v[0]*sign>0),
                                key=lambda v:(-v[0]*sign,v[1],v[2]))
        chosen[mid] = dict(start=start, end=start+30, move=delta)
    manifest = json.loads((source.AR/'manifest.json').read_text())
    cases = [(day,mid,part,'Inspect the 30-second YES midpoint move') for part, days in manifest['parts'].items()
             for day, ids in days.items() for mid in ids if mid in chosen]
    assert len(cases)==len(chosen)==7
    return sorted(cases), chosen


def load(cases):
    records = {}
    for day, part in sorted({(d,p) for d,m,p,title in cases}):
        root = L/part
        receipt = json.loads((root/(day+'.json')).read_text())
        assert receipt['manifest_sha256']==source.sha(source.AR/'manifest.json')
        assert receipt['status']=='DAY_DONE'
        for suffix, field in [('.jsonl.gz','gz'),('_arrays.npz','npz')]:
            assert receipt['sha256'][field]==source.sha(root/(day+suffix))
        selected = {m for d,m,p,title in cases if (d,p)==(day,part)}
        with gzip.open(root/(day+'.jsonl.gz'),'rt') as stream:
            for row in map(json.loads, stream):
                if row['m'] in selected:
                    records[row['m']]=row
    assert len(records)==len(cases)
    return records


def extract(cases,recs):
 # The projection below follows the original dashboard's fill-clock, lifecycle
 # and fee conventions. Its per-rank totals reconcile against matched_gap_r1.
 vals,ranks=source.top_fills();posts={r['order_hash']:r for r in read(R/'predictfun/reviews/maker_timing_estimates_20261124/outputs/post_estimates.csv.gz')};signed={r['order_hash']:r for r in read(R/'predictfun/reviews/maker_action_population_20261124/outputs/orders.csv')};accepted=read(MR/'v2/matched_gap_r1/per_market.csv');allm=[]
 for day,mid,part,title in cases:
  rec=recs[mid];curr=rec[ARMS[0]];ws=curr['ident']['opening_ms'];end=(curr['ident']['end_abs_ms']-ws)/1000;start=(curr['ident']['start_abs_ms']-ws)/1000;z=np.load(L/part/(day+'_arrays.npz'));ours={arm:own_arm(rec[arm],z,mid,arm,ws) for arm in ARMS};z.close();bars,dep,btc=book(day,mid,ws,end)
  life=pq.read_table(MT/'life'/f'{day}_{mid}_fills.parquet').to_pylist();li={(r['sid'],int(r['mk_idx']),r['mk_hash']):r for r in life};theirs=[];count=collections.Counter()
  for v in vals:
   if int(v['mid'])!=mid:continue
   assert str(v['day'])==day;count['source']+=1;t=(v['settle_ms']-ws)/1000
   if not start<=t<=end:count['outside_interval']+=1;continue
   if v['status']!='ok':count['excluded_'+v['status']]+=1;continue
   rank=ranks[v['addr'].lower()];p=posts.get(v['hash']);l=li.get((v['sid'],int(v['mk_idx']),v['hash'])) if v['role']=='maker' else None;flo=(l['x_lo_hard_ms']-ws)/1000 if l and l['x_lo_hard_ms']>=0 else None;side=0 if v['y_side']=='bid' else 1;price=v['p']*100;pp=[(float(p[k])-ws)/1000 for k in ['effect_lo_ms','effect_hi_ms']] if p and p['estimate_enabled']=='True' else None
   if l:assert abs(l['shares']-v['q'])<1e-7 and l['y_side']==v['y_side'] and l['hi_ms']==v['settle_ms']
   g=signed.get(v['hash']);f=dict(id=len(theirs),t=t,lo=flo,rank=rank,role=v['role'],hash=v['hash'],sid=v['sid'],mk_idx=int(v['mk_idx']),side=side,price=price,q=v['q'],fee=v['fee_fill'],self_trade=v['self_trade'],post=pp,post_grade=p['grade'] if p else 'missing',original_size=float(g['original_shares']) if g and g['original_shares'] else None,markouts={h:100*v['v_'+h]/v['q'] if v['st_'+h]=='ok' and v['q']>0 else None for h in ['1s','10s','60s']},end_value=v['s']*v['q']*(curr['mark']/100-v['p'])-v['fee_fill'],presence={})
   for arm,oa in ours.items():
    eligible=[o for o in oa['orders'] if not o['refused'] and o['side']==side];same=[(o['act'],o['stop']) for o in eligible if abs(o['price']-price)<1e-7];better=[(o['act'],o['stop']) for o in eligible if (o['price']>=price-1e-7 if side==0 else o['price']<=price+1e-7)];f['presence'][arm]=dict(same=presence(flo,t,same),equal_or_better=presence(flo,t,better))
   theirs.append(f)
  theirs.sort(key=lambda f:(f['t'],f['sid'],f['mk_idx'],f['role']))
  for i,f in enumerate(theirs):f['id']=i
  totals={str(rank):sum(f['end_value'] for f in theirs if f['rank']==rank) for rank in [1,2,3]}
  for rank in [1,2,3]:
   a=next(x for x in accepted if int(x['m'])==mid and x['party']==f'rank{rank}' and x['view']=='zero');assert abs(totals[str(rank)]-(float(a['total']) if a['total'] else 0))<1e-5;assert a['total'] or not any(f['rank']==rank for f in theirs)
  plot=[]
  for rank in [1,2,3]:
   fs=[f for f in theirs if f['rank']==rank];j=0;cash=pos=0.;curve=[]
   for t,b,a,*_ in bars[::2]:
    while j<len(fs) and fs[j]['t']<=t+1e-7:
     f=fs[j];sg=1 if f['side']==0 else -1;pos+=sg*f['q'];cash-=sg*f['q']*f['price']/100+f['fee'];j+=1
    curve.append([t,pos,(cash+pos*(a+b)/200) if a is not None and b is not None else None])
   curve.append([end,sum((1 if f['side']==0 else -1)*f['q'] for f in fs),totals[str(rank)]]);plot.append(dict(rank=rank,curve=curve))
  allm.append(dict(id=mid,day=day,title=title,ws=ws,start=start,end=end,mark=curr['mark'],book=bars,depth=dep,btc=btc,ours=ours,makers=theirs,maker_curves=plot,maker_totals=totals,source_counts=dict(count),profiles=curr['profile']))
 return allm, signed


def encoded(data):
    raw=json.dumps(source.clean(data),separators=(',',':'),allow_nan=False).encode()
    return base64.b64encode(gzip.compress(raw,compresslevel=6,mtime=0)).decode()


def build():
    cases, trends = choose_cases()
    records = load(cases)
    original = source.CASES
    try:
        source.CASES = cases
        markets, orders = extract(cases,records)
    finally:
        source.CASES = original
    scripts=[]
    for market in markets:
        mid=market['id']
        market['trend']=trends[mid]
        extra={arm:{'structural':records[mid][arm]['struct_log']} for arm in ARMS}
        signed={f['hash']:orders[f['hash']] for f in market['makers'] if f['hash'] in orders}
        path=f'markets/{mid}.js'
        (H/path).write_text('window.MM_MARKETS.push("'+encoded([market,extra,signed])+'");\n')
        scripts.append(path)
    (H/'markets/selection.json').write_text(json.dumps(trends,indent=2)+'\n')
    print(scripts)


if __name__=='__main__':
    build()
