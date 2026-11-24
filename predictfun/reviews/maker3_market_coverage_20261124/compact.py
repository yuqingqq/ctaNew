"""Participation in accepted canonical fill aggregates, not inferred posting.
Uses the existing checked maker_market artifact; no raw tape or economics run.
"""
import csv,json,resource
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CPU,(2,3))
import pyarrow.parquet as pq
D=Path('/home/yuqing/ctaNew/data/predict_fun/research/maker_trace_v2_20261123')
O=Path(__file__).resolve().parent
addr=json.loads((D/'econ/identities.json').read_text())['ex_ante']['top10_shares'][2]['addr']
assert addr.lower()=='0x57fbc891e89184da071ae9d97e745531cd7f718a'
days=['20261024','20261025','20261026','20261027']
a=pq.read_table(D/'econ/maker_market.parquet',columns=['day','mid','n_mk','n_tk','n_excl','status'],filters=[('addr','=',addr)]).to_pylist()
lookup={(r['day'],r['mid']):r for r in a}
assert len(lookup)==len(a)
rows=[];summaries=[]
for day in days:
    ms=pq.read_table(D/f'markets_{day}.parquet',columns=['mid','ws','we','status','n_records','n_poll_incomplete']).to_pylist()
    ex=json.loads((D/f'extract_{day}.json').read_text())
    assert len(ms)==ex['population']['n_markets']==288
    assert len({m['mid'] for m in ms})==288
    rr=[]
    for m in ms:
        x=lookup.get((day,m['mid']),{})
        r=dict(day=day,**m,accepted_maker=x.get('n_mk',0)>0,accepted_taker=x.get('n_tk',0)>0,recorded=x!={},excluded_only=bool(x) and x['n_mk']+x['n_tk']==0,econ_status=x.get('status','no_address_record'))
        r['coverage_ok']=not any(z in m['status'].split('+') for z in ['boundary_censored','hour_missing','no_records'])
        r['accepted_any']=r['accepted_maker'] or r['accepted_taker']
        rr.append(r)
    s=dict(day=day,missing_hours=ex['rest']['missing'],clipped_hours=ex['rest']['clipped'])
    for name,rs in [('all',rr),('coverage_ok',[r for r in rr if r['coverage_ok']])]:
        s[name]={'markets':len(rs)}
        for k in ['accepted_maker','accepted_taker','accepted_any','recorded','excluded_only']:
            s[name][k]=sum(r[k] for r in rs)
        s[name]['no_address_record']=sum(not r['recorded'] for r in rs)
        s[name]['poll_incomplete']=sum(r['n_poll_incomplete']>0 for r in rs)
    # Consecutive no-address-record intervals within a fully read envelope.
    runs=[];start=None;end=None
    for r in sorted(rr,key=lambda x:x['ws']):
        if r['coverage_ok'] and not r['recorded']:
            if start is None:start=r['ws']
            elif r['ws']!=end:runs.append([start,end]);start=r['ws']
            end=r['we']
        elif start is not None:runs.append([start,end]);start=end=None
    if start is not None:runs.append([start,end])
    s['no_record_runs']=runs
    summaries.append(s);rows.extend(rr)
with (O/'markets.csv').open('w') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
out=dict(asof='2026-11-24',address=addr,identity_day='20261023',days=summaries,sources=['econ/maker_market.parquet','econ/identities.json','markets_20261024..20261027.parquet','extract_20261024..20261027.json'],source_root=str(D),limitations=['Recorded includes excluded-only rows; accepted is the existing economics validity screen.','No address record is no recorded fill, not proof of no order.','One address, four consumed days, BTC 5m universe only.'])
(O/'evidence.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
