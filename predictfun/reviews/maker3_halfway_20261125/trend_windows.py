"""Count published rank-3 chart markers for the seven saved directional windows."""
import base64
import csv
import gzip
import json
from pathlib import Path

from pandas import read_parquet

here = Path(__file__).resolve().parent
charts = here.parent / 'maker_dashboard_20261124/markets'
actor = read_parquet(here.parents[2] / 'data/predict_fun/research/maker3_late_activity_20261124/actor_legs.parquet')
rows = []
for mid, window in json.loads((charts/'selection.json').read_text()).items():
    encoded = (charts/f'{mid}.js').read_text().split('"')[1]
    market = json.loads(gzip.decompress(base64.b64decode(encoded)))[0]
    assert market['id'] == int(mid) and market['trend'] == window
    fills = [f for f in market['makers'] if f['rank'] == 3 and not f['self_trade']]
    raw = actor[(actor.mid == int(mid)) & actor.t_rel_s.between(market['start'], market['end'])]
    def keys(items):
        return sorted((f['sid'], int(f['mk_idx']), f['role'], round(f['t'],5)) for f in items)
    assert keys(fills) == keys(raw.rename(columns={'t_rel_s':'t'}).to_dict('records'))
    t0,t1=window['start'],window['end']
    row=dict(mid=mid,start=t0,end=t1,move=window['move'])
    for role in ('maker','taker'):
        fs=[f for f in fills if f['role']==role]
        for phase,group in [('before',[f for f in fs if f['t']<t0]),('during',[f for f in fs if t0<=f['t']<=t1]),('after',[f for f in fs if f['t']>t1]),('final_minute',[f for f in fs if f['t']>=240])]:
            row[role+'_'+phase]=len(group)
        assert sum(row[role+'_'+phase] for phase in ['before','during','after'])==len(fs)
    pre = raw[raw.t_rel_s<t0]
    row['position_before_move']=round(pre.dq.sum(),2)
    row['last_passive_s']=max(f['t'] for f in fills if f['role']=='maker')
    rows.append(row)
with (here/'trend_windows.csv').open('w') as out:
    writer=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
