"""Independent saved-interval joins and probability ranges for posting subset."""
import csv,gzip,json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
H=Path(__file__).resolve().parent;R=H.parents[2];D=R/'data/predict_fun/research/maker_trace_v2_20261123'
x=json.loads((H/'result.json').read_text())
with gzip.open(R/'predictfun/reviews/maker_timing_estimates_20261124/outputs/post_estimates.csv.gz','rt') as f:p={r['order_hash']:r for r in csv.DictReader(f)}
cohort=[r for r in x['rows'] if 'first_filled_side_orders' in r]
n=0
for day in sorted({r['day'] for r in cohort}):
    rr=[r for r in cohort if r['day']==day];mids=[r['mid'] for r in rr]
    B=pq.read_table(D/f'tob_{day}.parquet',filters=[('mid','in',mids)],columns=['mid','upd_ms','recv_ns','best_bid','best_ask','n_bid','n_ask','status']).to_pandas()
    B['av']=np.maximum(B.upd_ms,B.recv_ns/1e6)
    L=pq.read_table(D/f'legs_{day}.parquet',filters=[('mid','in',mids),('mk_signer','=',x['address'])],columns=['mid','settle_ms','mk_hash','shares','y_side']).to_pandas()
    L=L[L.shares>0]
    for r in rr:
        b=B[B.mid==r['mid']].sort_values(['av','upd_ms','recv_ns'])
        for side,q in r['first_filled_side_orders'].items():
            assert q['first_record_s']==(L[(L.mid==r['mid'])&(L.y_side==side)].settle_ms.min()-r['ws_ms'])/1000
        for q in r['first_posts']+list(r['first_filled_side_orders'].values()):
            if q['status']=='no_post_estimate':continue
            z=p[q['hash']];assert q['grade']==z['grade']
            assert (q['status']=='enabled')==(z['estimate_enabled']=='True')
            if q['status']!='enabled':continue
            lo=float(z['effect_lo_ms']);hi=float(z['effect_hi_ms'])
            assert (q['post_lo_s'],q['post_hi_s'])==((lo-r['ws_ms'])/1000,(hi-r['ws_ms'])/1000)
            j=b.av.searchsorted(lo,side='right')-1
            frames=b.iloc[max(0,j):];frames=frames[frames.av<=hi]
            if q['context']['status']=='ok':
                assert j>=0 and len(frames)>0
                assert (frames.best_bid<frames.best_ask).all()
                assert all(not {'bad_frame','mid_mismatch','out_of_order','off_grid'}&set(s.split('+')) for s in frames.status)
                a=frames.av.to_numpy();t=frames.upd_ms.to_numpy()
                assert lo-t[0]<=1000 and hi-t[-1]<=1000
                assert len(a)<2 or np.max(a[1:]-t[:-1])<=1000.001
                prob=(frames.best_bid+frames.best_ask)/2
                assert q['context']['p_min']==prob.min() and q['context']['p_max']==prob.max()
                labs=set('neutral' if abs(v-.5)<1e-10 else 'higher_probability' if (q['side']=='bid')==(v>.5) else 'lower_probability' for v in prob)
                assert sorted(labs)==q['context']['directions']
            n+=1
        pp=r['first_filled_side_orders']
        if len(pp)<2:expected='only_one_filled_side'
        elif any(z['status']!='enabled' for z in pp.values()):expected='refused_or_unknown'
        elif pp['bid']['post_hi_s']<pp['ask']['post_lo_s']:expected='bid_before_ask_conditional'
        elif pp['ask']['post_hi_s']<pp['bid']['post_lo_s']:expected='ask_before_bid_conditional'
        else:expected='overlapping_windows'
        assert expected==r['bid_ask_post_relation']
out=dict(markets=len(cohort),enabled_interval_views_checked=n,conditional_not_private_post_order=True)
(H/'post_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
