"""Existing-data participation census; absence of fills is not absence of orders.

No economics, fitted selection rule or new market days. Fixed Oct 23 rank-3
address. Primary population: the extractor's Oct 24-27 BTC 5m registry,
with REST-envelope censoring/missing hours reported separately. Any recorded
maker/taker leg in the saved market envelope establishes observed activity;
live-window counts use settlement upper timestamps, not posting timestamps.
"""
import csv, hashlib, json, resource
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CPU, (11, 12))
import pandas as pd
import pyarrow.parquet as pq

R=Path('/home/yuqing/ctaNew')
D=R/'data/predict_fun/research/maker_trace_v2_20261123'
O=Path(__file__).resolve().parent
days=['20261024','20261025','20261026','20261027']
ids=json.loads((D/'econ/identities.json').read_text())
who=ids['ex_ante']['top10_shares'][2]
assert who['rank']==3
addr=who['addr']
assert addr.lower()=='0x57fbc891e89184da071ae9d97e745531cd7f718a'
allrows=[]; summaries=[]; inputs={}
def bind(p):
    inputs[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest()
bind(D/'econ/identities.json')
for day in days:
    mp=D/f'markets_{day}.parquet'; ep=D/f'extract_{day}.json'; lp=D/f'legs_{day}.parquet'
    bind(mp);bind(ep)
    ex=json.loads(ep.read_text())
    m=pq.read_table(mp, columns=['mid','ws','we','status','n_records','n_poll_incomplete','poll_incomplete_reasons']).to_pandas()
    assert m.mid.is_unique and len(m)==ex['population']['n_markets']==288
    assert (m.we-m.ws).eq(300).all()
    assert not ex['rest']['missing'] and not ex['rest']['clipped']
    # Projected columns only; case-normalized matching on both roles.
    l=pq.read_table(lp,columns=['mid','mk_signer','tk_signer','shares','status','phase','t_rel_s']).to_pandas()
    assert set(l.mid)<=set(m.mid)
    masks={k:l[col].str.lower().eq(addr.lower()) for k,col in [('maker','mk_signer'),('taker','tk_signer')]}
    rows=[]
    for x in m.to_dict('records'):
        mid=int(x['mid']);r=dict(day=day,**x)
        r['coverage_ok']=not any(z in x['status'].split('+') for z in ['boundary_censored','hour_missing','no_records'])
        for k,mask in masks.items():
            a=l.loc[mask & l.mid.eq(mid)]
            r[k+'_legs']=len(a)
            r[k+'_ok_legs']=int(a.status.eq('ok').sum())
            r[k+'_live_legs']=int(a.phase.eq('live').sum())
            r[k+'_shares']=float(a.shares.sum())
        r['any_fill']=bool(r['maker_legs']+r['taker_legs'])
        r['any_live_fill']=bool(r['maker_live_legs']+r['taker_live_legs'])
        rows.append(r)
    f=pd.DataFrame(rows).sort_values('ws')
    stats={}
    for name,sel in [('all',f),('coverage_ok',f[f.coverage_ok])]:
        stats[name]=dict(markets=len(sel),maker=int(sel.maker_legs.gt(0).sum()),taker=int(sel.taker_legs.gt(0).sum()),any_fill=int(sel.any_fill.sum()),no_fill=int((~sel.any_fill).sum()),live_maker=int(sel.maker_live_legs.gt(0).sum()),live_any=int(sel.any_live_fill.sum()),only_taker=int((sel.maker_legs.eq(0)&sel.taker_legs.gt(0)).sum()))
    # Descriptive gap duration, with censored rows breaking a run.
    gaps=[];start=None;last=None
    for x in f.to_dict('records'):
        absent=x['coverage_ok'] and not x['any_fill']
        if absent and (start is None or x['ws']!=last):
            if start is not None:gaps.append([start,last])
            start=x['ws']
        elif not absent and start is not None:
            gaps.append([start,last]);start=None
        last=x['we'] if absent else None
    if start is not None:gaps.append([start,last])
    stats['no_fill_runs']=gaps
    stats['market_statuses']=f.status.value_counts().to_dict()
    stats['incomplete_poll_markets']=int(f.n_poll_incomplete.gt(0).sum())
    summaries.append(dict(day=day,**stats));allrows.extend(rows)
    del l

f=pd.DataFrame(allrows)
f.to_csv(O/'markets.csv',index=False)
mm=pq.read_table(D/'econ/maker_market.parquet',columns=['day','mid','addr','n_mk','n_tk','status'],filters=[('addr','=',addr)]).to_pandas()
mm=mm[mm.day.isin(days)]
econ=dict(markets_with_accepted_maker=int(mm.n_mk.gt(0).sum()),markets_with_accepted_any=int((mm.n_mk+mm.n_tk).gt(0).sum()),excluded_only=int((mm.n_mk+mm.n_tk).eq(0).sum()))
assert set(mm.loc[(mm.n_mk+mm.n_tk).gt(0),'mid'])<=set(f.loc[f.any_fill,'mid'])
out=dict(asof='2026-11-24',address=addr,identity_day='20261023',population='BTC five-minute registry markets Oct 24-27; per-market REST envelope',days=summaries,econ_crosscheck=econ,inputs=inputs,limitations=['No fills does not establish no posted orders or deliberate selection.','One address; no attribution of related wallets.','Settlement-based live classification, not posting time.','Coverage_ok removes boundary_censored/hour_missing/no_records; all other statuses retained.'])
(O/'evidence.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
