"""Independently query the complete canonical market/price slice.
Known-bad controls: move the prior fill into the drop window; add a delayed
fill; make the book drop unknown. These must reject a nontrade-removal reading.
"""
import copy,json
from pathlib import Path
import pyarrow.parquet as pq
H=Path(__file__).resolve().parent
MT=Path('/home/yuqing/ctaNew/data/predict_fun/research/maker_trace_v2_20261123')
WS=1792977000000;mid=3789484;a=WS+8539;b=WS+8592
raw=pq.read_table(MT/'legs_20261026.parquet',filters=[('mid','=',mid),('y_side','=','ask'),('y_ptick','=',61)]).to_pylist()
frames=pq.read_table(MT/'depth/20261026_3789484_frames.parquet',filters=[('fid','in',[746,747])]).to_pylist()
levels=pq.read_table(MT/'depth/20261026_3789484_levels.parquet',filters=[('fid','in',[746,747]),('side','=','ask'),('ptick','=',61)]).to_pylist()
def test(raw,frames,levels):
 assert len(frames)==2 and all(x['status']=='ok' for x in frames)
 assert sorted(x['upd_ms'] for x in frames)==[a,b]
 assert all(x['status']=='ok' for x in levels)
 before=sum(x['shares'] for x in levels if x['fid']==746)
 after=sum(x['shares'] for x in levels if x['fid']==747)
 assert abs(before-205.4098360656)<1e-8 and after==0
 # Use canonical records independently of lifecycle row exclusion. Receipt-
 # derived timestamps would not justify a finite lower bound; refuse those.
 assert all(x['settle_ms']>0 and x['status']=='ok' for x in raw)
 overlapping=[x for x in raw if x['settle_ms']>=a and x['settle_ms']-2000<=b]
 assert not overlapping
 prior=[x for x in raw if x['settle_ms']==WS+8537]
 assert len(prior)==2 and abs(sum(x['shares'] for x in prior)-24.59016393442623)<1e-8
 return {'all_market_61c_legs':len(raw),'overlapping_61c_legs':len(overlapping),'prior_fill_shares':sum(x['shares'] for x in prior),'prior_upper_s':8.537,'book_before_s':8.539,'book_after_s':8.592,'removed_display_shares':before}
result=test(raw,frames,levels)
for kind in ['same_window_fill','delayed_fill','unknown_book']:
 r,f,l=copy.deepcopy((raw,frames,levels))
 if kind=='unknown_book':f[0]['status']='bad_frame'
 else:
  x=next(x for x in r if x['settle_ms']==WS+8537)
  x['settle_ms']=WS+(8560 if kind=='same_window_fill' else 10000)
 try:test(r,f,l)
 except AssertionError:pass
 else:raise AssertionError('corruption escaped '+kind)
result.update(checks_pass=True,corruptions_refused=3,
 inference='nontrade removal under the existing complete-record and <=2s settlement-lag premises; private cancel command/owner not observed',
 canonical_61c_legs=[{k:x[k] for k in ['sid','settle_ms','mk_hash','shares','status']} for x in raw])
(H/'ask61_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='canonical_61c_legs'}))
