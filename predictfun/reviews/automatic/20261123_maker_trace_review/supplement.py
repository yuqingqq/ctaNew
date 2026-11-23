"""Bounded saved-evidence crosschecks; pinned source, no development mutation."""
from pathlib import Path
import sys, json, hashlib, time, importlib.util, gc, signal
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
PIN=Path('/tmp/pf-mt-review-1af85c56'); ROOT=Path('/home/yuqing/ctaNew')
OUT=ROOT/'data/predict_fun/research/maker_trace_v1_20261122'; HERE=Path(__file__).resolve().parent
P=PIN/'predictfun/reviews/maker_trace_20261122'
sys.path.insert(0,str(PIN/'predictfun/research'))
import pf_mt_truth as TR
import pf_mt_common as C
start=time.process_time(); signal.setitimer(signal.ITIMER_PROF,90)
R={'target':'1af85c5691d5ca4c2f9565d94a13cd9680688d3c'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((P/'manifest.json').read_text()); amended=json.loads((P/'amend2_manifest.json').read_text())
sources=manifest['sources'] | amended['changed']
amend1=json.loads((P/'amend1_manifest.json').read_text())
sources.update({'predictfun/reviews/maker_trace_20261122/'+n:v for n,v in amend1['amended'].items()})
R['sources']={n:sha(PIN/n)==v for n,v in sources.items()}
R['source_disagreements']={n:{'actual':sha(PIN/n),'declared':v} for n,v in sources.items() if sha(PIN/n)!=v}
print('source_disagreements',json.dumps(R['source_disagreements']),flush=True)
assert not R['source_disagreements']
readings=json.loads((P/'outputs/readings.json').read_text())
R['readings_input_schema']=list(readings['inputs'].items())[:2]
# All 71 saved lifecycles: distinguish the trivial nonnegative queue bound from actual queue information.
queue_diff=queue_trivial=queue_rounding=0; tob={}; nlife=0; q_examples=[]
for op in sorted((OUT/'life').glob('*_orders.parquet')):
    o=pd.read_parquet(op,columns=['mk_hash','scenario','ahead_lo','ahead_hi','queue_class'])
    for h,g in o.groupby('mk_hash',sort=False):
        if g.queue_class.iloc[0]=='unresolved' and (g.ahead_lo.nunique(dropna=False)>1 or g.ahead_hi.nunique(dropna=False)>1):
            queue_diff+=1
            queue_trivial+=int(g.ahead_hi.isna().all() and g.ahead_lo.fillna(0).eq(0).all())
            queue_rounding+=int(g.ahead_hi.isna().all() and g.ahead_lo.fillna(0).le(.01).all())
            if not(g.ahead_hi.isna().all() and g.ahead_lo.fillna(0).eq(0).all()) and len(q_examples)<3:
                q_examples.append(g.to_dict('records'))
    nlife+=1
R['queue']={'markets':nlife,'unresolved_differing_hashes':queue_diff,'only_0_vs_nan_lo_and_nan_hi':queue_trivial,
            'only_display_tolerance_vs_nan':queue_rounding,'nonzero_examples':q_examples}
print('queue',json.dumps(R['queue']),flush=True)
assert queue_diff==queue_rounding
# Four examples include both sides and all remaining failure types; mismatch is empty side, not displayed liquidity disagreement.
for day,m in [('20261024',3697902),('20261025',3734864),('20261108',4353706),('20261108',4334454)]:
    fr=pd.read_parquet(OUT/'depth'/f'{day}_{m}_frames.parquet'); lv=pd.read_parquet(OUT/'depth'/f'{day}_{m}_levels.parquet')
    tb=pd.read_parquet(OUT/f'tob_{day}.parquet',filters=[('mid','=',m)])
    x=fr.merge(tb,on=['upd_ms','recv_ns','conn'],suffixes=('','_t')); count={}
    for side,how in [('bid','max'),('ask','min')]:
        v=lv[(lv.side==side)&(lv.status=='ok')]; b=getattr(v.groupby('fid').ptick,how)(); sh=v.set_index(['fid','ptick']).shares
        expect=np.array([sh.get((f,int(p)),np.nan) if np.isfinite(p) else np.nan for f,p in zip(x.fid,x.fid.map(b))])
        got=x['best_'+side+'_sh'].to_numpy(); bad=~np.isclose(got,expect,rtol=0,atol=1e-9,equal_nan=True)
        count[side]={'n':int(bad.sum()),'all_empty_side_zero_vs_nan':bool(((got[bad]==0)&np.isnan(expect[bad])).all())}
    tob[f'{day}_{m}']=count
R['tob']=tob
assert all(z['all_empty_side_zero_vs_nan'] for k in tob.values() for z in k.values())
# Original R2 function rerun on the frozen tables (no write through its CLI).
truth=TR.read_truth(TR.TRUTH_DIR/'own_orders.parquet')
G=TR.gt_lifecycle_inputs(OUT,C.GT_DAY)
orders=pd.concat([pd.read_parquet(OUT/'life'/f'{C.GT_DAY}_{m}_orders.parquet') for m in G['required']],ignore_index=True)
fills=pd.concat([pd.read_parquet(OUT/'life'/f'{C.GT_DAY}_{m}_fills.parquet') for m in G['required']],ignore_index=True)
pub=set(pd.read_parquet(OUT/f'legs_{C.GT_DAY}.parquet',columns=['mk_hash']).mk_hash)
scored=TR.score(orders,fills,truth,pub,TR.gt_markets(),censored_mids=G['censored'])
saved=json.loads((TR.TRUTH_DIR/'scores.json').read_text())
R['R2']={'population_equal':scored['population']==saved['population'],'defects_equal':scored['defects_total']==saved['defects_total'],
         'population':scored['population'],'defects':scored['defects_total'],
         'H2_flagged_order_is_own':bool(truth.mk_hash.eq('0x7f538ba598412e0521677e9392ad4285f949a86256d6bc034230552995570993').any())}
assert R['R2']['population_equal'] and R['R2']['defects_equal']
del orders,fills,truth,pub,scored;gc.collect()
# Independent address/day conservation and pooled top-ten sum. No rebuilding of rows.
md=pd.read_parquet(OUT/'econ/maker_day.parquet')
R['maker_day_columns']=list(md.columns)
ident=json.loads((OUT/'econ/identities.json').read_text()); top={r['addr'] for r in ident['ex_ante']['top10_shares']}
x=md[(md.population=='FILL') & md.addr.isin(top)]
cols=['gross_res_mk','gross_res_tk','fee_res','net_res','rt_pnl_res','resid_res']
R['top10_independent_sums']={c:float(x[c].sum()) for c in cols}
R['top10_byday']=x.groupby('day').net_res.sum().to_dict()
allx=md[md.population=='FILL']
R['all_addresses_net_plus_fees']=float((allx.net_res+allx.fee_res).sum())
claims=json.loads((P/'claims.json').read_text()); claim=claims['exante_sh.in']
for col,k in [('net_res','net'),('fee_res','fees'),('rt_pnl_res','rt'),('resid_res','resid')]: assert abs(R['top10_independent_sums'][col]-claim[k])<0.011
assert abs(R['all_addresses_net_plus_fees'])<1e-5
# A gross markout is not net of taker fees. Record the available denominators rather than silently subtract pooled fees from a different cohort.
seg=pd.read_csv(P/'outputs/econ_segments.csv',dtype={'day':str},float_precision='round_trip')
R['economic_columns']=list(seg.columns)
R['taker_all']=seg[(seg.day=='pooled')&(seg.id_set=='exante_sh')&(seg.role=='taker')&(seg.dim=='all')&(seg.grp.isin(['in','field']))][['grp','sh','sh_1s','v_1s','fee_fill','fee_res','net_res']].to_dict('records')
R['R4_claim_reproduction']='independent maker_day sums equal reported rounded top-ten totals; all-address conservation holds'
R['cpu_s']=time.process_time()-start;R['peak_mib']=TR.vmhwm_mib()
(HERE/'supplement.json').write_text(json.dumps(R,indent=2,default=str)+'\n')
print(json.dumps({k:R[k] for k in ['queue','tob','top10_independent_sums','all_addresses_net_plus_fees','cpu_s','peak_mib']},indent=2))
