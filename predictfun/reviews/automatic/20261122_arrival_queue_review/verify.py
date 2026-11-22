"""Pinned saved-evidence review; no model, labels or replay. Extra probes target
cohort denominators, reported queue quality, and price-occupancy interpretation.
All corruptions are in-memory copies. No development artifact is modified.
"""
from pathlib import Path
import hashlib, importlib.util, json, gzip, sys
import numpy as np
import pandas as pd
ROOT=Path('/home/yuqing/ctaNew'); PIN=Path('/tmp/pf-aq-review-75e9531d')
O=Path(__file__).resolve().parent; REL='predictfun/reviews/ladder_arrival_queue_20261122'
R=PIN/REL; DATA=ROOT/'data/predict_fun/research/ladder_arrival_queue_v1_20261122'
def mod(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def eq(a,b):return bool(np.allclose(np.asarray(a,float),np.asarray(b,float),rtol=1e-9,atol=1e-6,equal_nan=True))
def csv(p):return pd.read_csv(p,dtype={'day':str,'band':str,'side':str},float_precision='round_trip')
C=mod('review_aq_checks',R/'aq_checks.py'); C.REPO=PIN;C.R=R;C.STATES=PIN/C.STATES.relative_to(ROOT)
E=C.load(DATA,R/'manifest.json'); audit=E['audit']; bound={}
for d,expect in E['gm']['parent_day_records'].items():
 p=C.PD_/'days'/f'{d}.json';assert sha(p)==expect
 rec=json.loads(p.read_text());raw=p.with_suffix('.jsonl.gz');assert sha(raw)==rec['sha256']['gz'];bound[str(raw)]=sha(raw)
for n in ('states.csv','intervals.csv','cells.csv'):
 raw=DATA/n; assert sha(raw)==audit['outputs'][n]
 assert gzip.decompress((R/'outputs'/f'{n}.gz').read_bytes())==raw.read_bytes()
assert (R/'outputs/audit.json').read_bytes()==(DATA/'audit.json').read_bytes()
assert sha(R/'manifest.json')==audit['manifest_sha256']
for n,v in E['gm']['sources'].items():assert sha(PIN/'predictfun/research'/n)==v
E['mine']=C.my_rows(E);E['mine_u']=C.my_union(E['mine'][1]);base=C.run_checks(E);fz=C.falsify(E,base)
assert base['all_ok'] and fz['falsifier_ok'];print('Pinned checker and 12 falsifiers PASS',flush=True)
S=E['states'];I=E['intervals'];T=E['cells'];keys=['day','m','gid']
raws=E['S'].set_index(keys+['kind']);bad=[];base_fill_counts=[]
for z in S.to_dict('records'):
 k=(z['day'],int(z['m']),int(z['gid']));L=E['led'][k[:2]];o=next(o for o in L['orders'] if int(o['gid'])==k[2]);f=raws.loc[k+('first',)]
 live=raws.loc[k+('live',)];entry=raws.loc[k+('entry',)];v=L['queue_venue']['first'][str(k[2])]
 expect=dict(Q=live.fp_queue,R=live.fp_remaining,Q_over_R=live.fp_queue/live.fp_remaining,o_q_rel=live.o_q_rel,known_filled=o['size']-live.fp_remaining,
             size=o['size'],price=o['price'],age_ms=live.ctx_generation_age_ms,age_since_activation_ms=live.t-o['act'],o_depth=live.o_depth,pf_age=live.pf_age,
             venue_ahead=v[1],venue_rem=v[2],client_minus_venue=f.fp_queue-v[1],known_minus_venue_rem=f.fp_remaining-v[2],
             Q_first=f.fp_queue,Q_entry=entry.fp_queue,dt_first=live.t-f.t,dt_entry=live.t-entry.t)
 if z['recent_status']=='available':expect['dQ_250']=live.fp_queue-f.fp_queue
 for n,a in expect.items():
  if not eq(z[n],a):bad.append([list(k),n,z[n],a])
 assert len(L['fills'])==len(L['fill_mids'])
assert not bad,bad[:4]
for h in C.H:
 assert eq(T[f'h{h}_per_share'],T[f'h{h}_net']/T[f'h{h}_valid_shares'].where(T[f'h{h}_valid_shares']>0))
 assert eq(T[f'h{h}_per_opportunity'],T[f'h{h}_net']/T[f'h{h}_valid'].where(T[f'h{h}_valid']>0))
 assert eq(T[f'h{h}_state_negative'],T[f'h{h}_net']-T[f'h{h}_state_positive'])
assert eq(T.any_fill_rate,T.any_fill/T.at_risk.where(T.at_risk>0))
# Extra accounting fields omitted from C: filled-out and all status counts.
U=E['mine_u']; X=pd.concat([I,U],ignore_index=True).merge(S[keys+['side']+list(C.VIEWS)],on=keys,validate='many_to_one')
for row in T.itertuples():
 g=X[(X[row.view].astype(str)==str(row.band))&(X.interval==row.interval)]
 if row.day!='all':g=g[g.day==row.day]
 if row.side!='all':g=g[g.side==int(row.side)]
 assert int(g.filled_out.sum())==row.filled_out
 for status in ('not_at_risk','at_risk_full','removed_in_interval','filled_out_in_interval','end_censored'):
  assert int((g.follow_up==status).sum())==getattr(row,'n_'+status)
# In-memory known-bad examples the committed checker does not cover.
mut=dict(E);mut['states']=S.assign(client_minus_venue=0.)
accepted_venue=all(C.fam_F(mut)[0].values()) and all(C.fam_S(mut)[0].values())
mut2=dict(E);mut2['cells']=T.copy();mut2['cells']['h1000_per_share']=-T.h1000_per_share
accepted_ratio=all(C.fam_C(mut2)[0].values())
# Recompute the existing reading artifacts from the pinned reader, without its writers.
RD=mod('review_readings',R/'tools/read_arrival_queue.py')
aa,p1=RD.r1(S,I,T);pd.testing.assert_frame_equal(aa,csv(R/'outputs/r1_decomposition.csv'),check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-8)
tables={}
for n,view in [('r2_queue','queue'),('r3_queue_pressure','queue_pressure'),('r3b_queue_pressure_flow','queue_pressure_flow'),('r4_progress','progress'),('r4b_pressure_flow','pressure_flow')]:
 a=RD.cells(T,view);pd.testing.assert_frame_equal(a,csv(R/'outputs'/f'{n}.csv'),check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-8);tables[n]=a
contr=RD.contrasts(T);pd.testing.assert_frame_equal(contr,csv(R/'outputs/r3_contrasts.csv'),check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-8)
rd=json.loads((R/'outputs/readings.json').read_text());assert p1==rd['r1'];assert RD.venue(S)==rd['r5_venue']
orders={iv:{str(h):RD.order_of(tables['r2_queue'],iv,h,['front','within_R','beyond_R']) for h in (1000,30000)} for iv in RD.IVS}
assert orders==rd['r2_queue_order'];assert rd['r3_evaluable']==int((contr.status=='evaluable').sum())
# Independent support counts on the population actually used by the markout.
sup=[]
for view in ('queue','queue_pressure','progress'):
 for (b,iv),g in X.groupby([view,'interval']):
  gc=g[g.common.astype(bool)]
  sup.append(dict(view=view,band=b,interval=iv,total_states=len(g),total_filled=int((g.fills>0).sum()),common_states=len(gc),common_filled=int((gc.fills>0).sum()),
                  total_shares=float(g.shares.sum()),common_shares=float(gc.shares.sum()),sparse_total=bool((g.fills>0).sum()<20),sparse_common=bool((gc.fills>0).sum()<20)))
support=pd.DataFrame(sup);support.to_csv(O/'support.csv',index=False)
# R5 and price-state statements use saved features, never tape lookup or a new economics split.
feats=[];feature_bind={}
for d in E['gm']['days']:
 p=ROOT/'data/predict_fun/research/ladder_rest_v1_20261121/days'/f'{d}.json'
 rec=json.loads((PIN/'predictfun/reviews/ladder_rest_20261121/outputs/days'/f'{d}.json').read_text()); assert p.read_bytes()==(PIN/'predictfun/reviews/ladder_rest_20261121/outputs/days'/f'{d}.json').read_bytes()
 fp=p.with_name(d+'_features.parquet');assert sha(fp)==rec['sha256']['feat'];feature_bind[str(fp)]=sha(fp)
 f=pd.read_parquet(fp,columns=['day','m','gid','kind','t','side','fp_queue','fp_shown','fp_remaining','o_depth']);f['day']=f.day.astype(str);feats.append(f[f.kind=='live'])
F=pd.concat(feats,ignore_index=True);J=S.merge(F,on=keys+['t','side'],suffixes=('','_f'),validate='one_to_one');assert len(J)==len(S)
assert eq(J.Q,J.fp_queue) and eq(J.R,J.fp_remaining) and eq(J.o_depth,J.o_depth_f)
front=J[J.queue_band=='front'];occupied=front[front.fp_shown>0]
occ=dict(front=len(front),inside=int((front.o_depth<0).sum()),touch=int((front.o_depth==0).sum()),behind=int((front.o_depth>0).sum()),
         currently_displayed_positive=int((front.fp_shown>0).sum()),currently_empty=int((front.fp_shown==0).sum()),unknown_shown=int(front.fp_shown.isna().sum()),
         behind_and_displayed_positive=int(((front.o_depth>0)&(front.fp_shown>0)).sum()))
occupied[keys+['t','Q','fp_shown','o_depth']].head(12).to_csv(O/'occupancy_examples.csv',index=False)
# Rejudge saved per-mutant evidence against the pinned harness/source/selftest hashes.
sys.path.insert(0,str(PIN/'predictfun/research'));import pf_mutant_summary as MS;import pf_mutant_one as MO;import pf_ladder_arrival_queue_mutants as MH
logs=R/'logs/mutants_final';recs={'control':MS.last_record(logs/'laqf_control.log'),**{i:MS.last_record(logs/f'laqf_{i}.log') for i in range(len(MH.MUTANTS))}}
mut_ok,mut_bad=MS.judge(len(MH.MUTANTS),recs,MO.sources(MH),[m[0] for m in MH.MUTANTS]);assert mut_ok,mut_bad
assert not MS.judge(len(MH.MUTANTS),{**recs,0:recs[1]},MO.sources(MH),[m[0] for m in MH.MUTANTS])[0]
out=dict(target='75e9531d21610a3b16f5597153299bc92c210a3a',checks=base,falsifier=fz,raw_ledger_hashes=bound,feature_hashes=feature_bind,
         extra_state_value_mismatches=bad,extra_cell_fields_match=True,reader_outputs_reproduced=True,mutants_rejudged=len(MH.MUTANTS),
         checker_escapes=dict(zeroed_client_minus_venue=accepted_venue,negated_h1000_per_share=accepted_ratio),
         support_flips=support[support.sparse_total!=support.sparse_common].to_dict('records'),r1_denominators=aa[['interval','shares','common_shares','common_states']].to_dict('records'),
         front_occupancy=occ,venue_error=dict(within_one=int((S.client_minus_venue.abs()<1).sum()),below_one=int((S.client_minus_venue<=-1).sum()),above_one=int((S.client_minus_venue>=1).sum())))
(O/'evidence.json').write_text(json.dumps(out,indent=2,default=str)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('checks','falsifier','raw_ledger_hashes','feature_hashes')},indent=2,default=str))
