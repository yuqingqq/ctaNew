"""User-requested attribution of the 61c disappearance in market 3789484.
Saved evidence only; enumerate all 61c maker legs whose hard windows overlap
the native 8.539--8.592s disappearance, plus nearby records for context.
No new trading rule or exact execution time is inferred from proximity.
"""
import json
from pathlib import Path
import pyarrow.parquet as pq
H=Path(__file__).resolve().parent
MT=Path('/home/yuqing/ctaNew/data/predict_fun/research/maker_trace_v2_20261123')
WS=1792977000000;MID=3789484
life=pq.read_table(MT/'life/20261026_3789484_fills.parquet').to_pandas()
target=life[(life.y_side=='ask')&(life.y_ptick==61)&(life.hi_ms>=WS+7539)&(life.hi_ms<=WS+11592)]
ls=pq.read_table(MT/'legs_20261026.parquet',filters=[('mid','=',MID),('sid','in',target.sid.unique().tolist())]).to_pandas()
rs=pq.read_table(MT/'records_20261026.parquet',filters=[('mid','=',MID),('sid','in',target.sid.unique().tolist())]).to_pandas()
out=[]
for f in target.sort_values(['hi_ms','sid','mk_idx']).to_dict('records'):
 raw=ls[(ls.sid==f['sid'])&(ls.mk_idx==f['mk_idx'])].iloc[0].to_dict()
 rec=rs[rs.sid==f['sid']].iloc[0].to_dict()
 out.append(dict(sid=f['sid'],hash=f['mk_hash'],q=f['shares'],hi_s=(f['hi_ms']-WS)/1000,hard_lo_s=(f['x_lo_hard_ms']-WS)/1000,
     method=f['t_method'],status=f['status'],maker=raw['mk_signer'],native_price=raw['mk_price'],native_side=raw['mk_quote'],
     native_index=raw['mk_index_set'],take_hash=rec['tk_hash'],take_q=rec['amount_filled'],take_price=rec['price_executed'],
     hard_window_overlaps=bool(f['hi_ms']>=WS+8539 and f['x_lo_hard_ms']<=WS+8592),
     all_legs=ls[ls.sid==f['sid']][['mk_idx','mk_hash','y_side','y_price','shares']].to_dict('records')))
e=json.loads((H/'fill_clock_evidence.json').read_text())
frames=[r for r in e['frames'] if 8.3<=r['venue_s']<=8.698]
data=dict(market=MID,opening_ms=WS,frames=frames,nearby_61c_fills=out)
(H/'ask61_evidence.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data,indent=2))
