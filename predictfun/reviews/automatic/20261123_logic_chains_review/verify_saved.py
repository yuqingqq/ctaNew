"""Bounded saved-artifact review. No market context rebuild, tape replay or fitted rule."""
import hashlib
import json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path('/home/yuqing/ctaNew')
PIN=Path('/tmp/pf-chains-review-8d68808d')
BASE=ROOT/'predictfun/reviews/maker_trace_20261122'
PB=PIN/'predictfun/reviews/maker_trace_20261122'
H=Path(__file__).resolve().parent
out={'target':'8d68808d4c5232f910d92808bef14871eb6ab1b5','checks':{},'post':{},'pull':{},'take':{}}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return pd.read_csv(p,float_precision='round_trip',dtype={'day':str})
def clean(v):
    if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [clean(x) for x in v]
    if hasattr(v,'item'): return clean(v.item())
    if isinstance(v,float) and not pd.notna(v): return None
    return v
for name in ('post_fill','pull','take'):
    p=BASE/'v2/chains'/name
    rec=json.loads((PB/'v2/chains'/name/'receipt.json').read_text())
    n=0
    for rel,x in rec.get('sources',{}).items():
        if rel.startswith('/home/yuqing/ctaNew/'):
            rel=rel[len('/home/yuqing/ctaNew/'):]
        pp=PIN/rel
        if pp.is_file():
            assert sha(pp)==x,(name,rel)
            n+=1
    outs=rec.get('outputs',{})
    good=[]
    for rel,x in outs.items():
        if isinstance(x,dict) and 'sha256' in x:
            assert sha(p/rel)==x['sha256'],(name,rel)
            good.append(rel)
    out['checks'][name]={'pinned_sources':n,'receipt_output_hashes_verified':good,
                         'saved_summary_equals_committed':sha(p/'summary.csv')==sha(PB/'v2/chains'/name/'summary.csv')}
    assert out['checks'][name]['saved_summary_equals_committed']
P=read(BASE/'v2/chains/post_fill/orders.csv')
life=ROOT/'data/predict_fun/research/maker_trace_v2_20261123/life'
L=[]
for (d,m),g in P.groupby(['day','mid']):
    f=life/f'{d}_{m}_orders.parquet'
    x=pq.read_table(f,columns=['mk_hash','scenario','post_class','status']).to_pandas()
    x=x[x.scenario.eq('S_NOPRIO')].copy()
    x['day'],x['mid']=d,m
    L.append(x)
L=pd.concat(L).rename(columns={'post_class':'life_post_class','status':'life_status'})
Q=P.merge(L[['day','mid','mk_hash','life_post_class','life_status']],on=['day','mid','mk_hash'],how='left',validate='one_to_one')
assert len(Q)==4149 and Q.life_post_class.notna().all()
out['post']['count']=len(Q)
out['post']['classes']=Q.post_class.value_counts().to_dict()
out['post']['class_by_hard_grade']=pd.crosstab(Q.post_class,Q.life_post_class).to_dict()
short=Q.post_class.isin(['late_post','short_rest'])
out['post']['scenario_short_count']=int(short.sum())
out['post']['short_hard_constrained_count']=int((short & Q.life_post_class.eq('constrained')).sum())
out['post']['short_statuses']=Q.loc[short,'status'].value_counts().to_dict()
out['post']['next_kind_by_link_status']=pd.crosstab(Q.next_kind,Q.next_link_status).to_dict()
out['post']['examples_false_upgrade']=Q[short & ~Q.life_post_class.eq('constrained')][['day','mid','mk_hash','post_class','window_width_ms','life_post_class','status']].head(8).to_dict('records')
S=read(BASE/'v2/chains/pull/summary.csv')
mask=S['view'].eq('S_FIFO') & S['scope'].eq('pooled') & S['day'].eq('all') & S['D'].astype(str).eq('250')
out['pull']['comparison_rows']=S[mask & S['feature'].isin(['btc_against_1s_some','btc_against_5s_some','inv_built_5s_some','behind_touch_some','tau_start'])].to_dict('records')
# Preserve exact names too, so a changed spelling cannot silently make a clean empty table.
out['pull']['available_features']=sorted(S.feature.unique().tolist())
out['pull']['all_primary_rows']=S[mask & S['group'].eq('cancel_possible')].to_dict('records')
C=read(BASE/'v2/chains/pull/controls.csv')
c=C[C['view'].eq('S_FIFO') & C.cls.eq('cancel_possible')]
out['pull']['control_counts']=c.groupby(['control','draw_status']).size().to_dict()
V=read(BASE/'v2/chains/pull/pulls.csv')
v=V[V['view'].eq('S_FIFO')]
out['pull']['columns']=list(V.columns)
out['pull']['identified_any']=int(v.pull_identified.eq(True).sum())
out['pull']['end_histories']=v.end_histories.value_counts().to_dict()
T=read(BASE/'v2/chains/take/takes.csv')
TS=read(BASE/'v2/chains/take/summary.csv')
out['take']['summary_rank1_all']=TS[TS.scope.eq('rank1') & TS.day.eq('all')].to_dict('records')
out['take']['take_count']=len(T)
out['take']['columns']=list(T.columns)
out['take']['excluded_before']=int(T.n_excl_before.gt(0).sum())
out['take']['excluded_before_claimed']=int((T.n_excl_before.gt(0) & T.claim.eq(True)).sum())
out['take']['status']=T.status.value_counts().to_dict()
tr=json.loads((PB/'v2/chains/take/receipt.json').read_text())
out['take']['pools']=tr.get('pools')
out['take']['direction_fixture']={'inventory':100,'take_side':-1,'btc_return_bps':2,
                                'against_take':True,'against_inventory':False,
                                'checks':'A sell reducing a long position with rising BTC is against the take but with inventory.'}
assert out['post']['scenario_short_count']>0 and len(out['pull']['all_primary_rows'])>0
assert len(out['take']['summary_rank1_all'])==1
(H/'evidence.json').write_text(json.dumps(clean(out),indent=2)+'\n')
print(json.dumps(clean({'checks':out['checks'],'post':out['post'],
                       'pull_controls':out['pull']['control_counts'],'pull_identified_any':out['pull']['identified_any'],
                       'take_pools':out['take']['pools'],'excluded_before':out['take']['excluded_before']}),indent=2))
