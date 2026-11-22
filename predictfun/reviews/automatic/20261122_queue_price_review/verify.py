"""Bounded pinned-source review. Saved evidence only; no strategy/label replay.
Run through PF_SLOT=codex run_light.sh. All writes stay in this review directory.
"""
from pathlib import Path
import copy, gzip, hashlib, importlib.util, json, sys
import numpy as np
import pandas as pd

PIN = Path('/tmp/pf-qp-review-012db7bb')
ROOT = Path('/home/yuqing/ctaNew')
OUT = Path(__file__).resolve().parent
REL = 'predictfun/reviews/ladder_queue_price_20261122'
sys.path.insert(0, str(PIN / 'predictfun/research'))

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

Q = module('review_checker', PIN / REL / 'qp_checks.py')
R = module('review_reader', PIN / REL / 'tools/read_queue_price.py')
Q.REPO = PIN; Q.R = PIN / REL
Q.D = ROOT / 'data/predict_fun/research/ladder_queue_price_v1_20261122'
Q.FEAT = ROOT / 'data/predict_fun/research/ladder_rest_v1_20261121/days'
Q.LED = ROOT / 'data/predict_fun/research/ladder_pressure_v1_20261121/days'
E = Q.load()
E['mine_s'] = Q.my_states(E)
E['mine_c'] = Q.my_cells(E['mine_s'], E['aq_intervals.csv'])
E['mine_r'] = Q.my_readings(E['mine_s'], E['mine_c'])
base = Q.run_checks(E)
assert base['all_ok'], base

# Preserve the frozen falsifier logic, but put its temporary resource fixture
# under the review directory rather than the development output directory.
name, fam, target, old = Q.CASES[-1]
def review_fake(e):
    out = e['out']; e['out'] = OUT
    try: old(e)
    finally: e['out'] = out
Q.CASES[-1] = (name, fam, target, review_fake)
fz = Q.falsify(E, base); assert fz['falsifier_ok'], fz

# Check fields omitted by frozen family P, using independently rebuilt cells.
C, S, rd = E['mine_c'], E['mine_s'], E['readings']
def ord_at(iv, h, day='all', side='all', dep=None):
    view = 'price_display_queue' if dep is None else 'behind_depth_queue'
    prefix = Q.PR if dep is None else dep
    vals = {b: Q.val(Q.cell(C, view, f'{prefix}|{b}', iv, day, side), h) for b in Q.BANDS}
    return Q.vorder(vals), vals
extra = {}
for iv in Q.IVS + ('union',):
    for h in (1000, 30000):
        got = rd['q2_principal'][iv][str(h)]
        assert got['days'] == {d: ord_at(iv,h,day=d)[0] for d in Q.DAYS}
        assert got['sides'] == {s: ord_at(iv,h,side=s)[0] for s in ('0','1')}
        g = C[(C.view == 'price_display_queue') & (C.interval == iv) & (C.day == 'all') & (C.side == 'all') & (C.cohort == 'common')]
        for b in Q.BANDS:
            x = g[g.band.str.endswith('|'+b)]; q = x[f'h{h}_valid_shares'].sum()
            want = dict(per_share=x[f'h{h}_net'].sum()/q if q else np.nan, valid_filled=int(x[f'h{h}_valid_filled'].sum()))
            assert Q.same(list(want.values()), list(rd['q3_pooled'][iv][str(h)]['bands'][b].values()))
for dep, hs in rd['q5_behind_depth'].items():
    for h, got in hs.items():
        _, vals = ord_at('union',int(h),dep=dep)
        assert Q.same([v[0] for v in vals.values()], [got['per_share'][b] for b in Q.BANDS])
        assert got['evaluable'] == {b:v[1] for b,v in vals.items()}
assert rd['states'] == len(S) == 4840
assert rd['q1_front']['states'] == int((S.queue_band=='front').sum())
assert rd['asof_outputs'] == E['audit']['asof']
for b, got in rd['q7_principal_censoring'].items():
    assert got['states'] == int(Q.cell(C,'price_display_queue',Q.PR+'|'+b,'union').all_opportunities)
extra['all_detailed_reading_fields'] = True

specs = {
 't_principal': ('price_display_queue',[f'{Q.PR}|{b}' for b in Q.BANDS],R.IVS),
 't_price_display': ('price_display',None,R.IVS),
 't_front_composition': ('price_display_queue',sorted(b for b in C.band.unique() if str(b).endswith('|front')),R.IVS),
 't_behind_depth': ('behind_depth_queue',sorted(b for b in C.band.unique() if str(b).startswith('depth=')),R.IVS),
 't_pressure_census': ('principal_pressure',None,('union',)),
}
tables = {}
for name,(view,bands,ivs) in specs.items():
    expected = R.table(C,view,bands,ivs).sort_values(['band','interval']).reset_index(drop=True)
    saved = pd.read_csv(Q.R/'outputs'/f'{name}.csv',float_precision='round_trip').sort_values(['band','interval']).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected,saved,check_dtype=False,atol=1e-6,rtol=1e-9)
    tables[name] = len(saved)
extra['five_reading_tables'] = tables

# Demonstrate a genuine remaining checker blind spot, without touching files.
escapes = {}
for name, mutate in (
 ('wrong_per_day_order', lambda d: d['q2_principal']['union']['1000']['days'].__setitem__('20261024','front<within_R<beyond_R')),
 ('wrong_pooled_numeric_value', lambda d: d['q3_pooled']['union']['1000']['bands']['front'].__setitem__('per_share',999.)),
):
    changed=copy.deepcopy(rd); mutate(changed)
    escapes[name]=Q.fam_P(dict(E,readings=changed))[0]['P_readings']
assert all(escapes.values())

# Q8's numeric difference is from the first review. Determine whether its
# price/display grouping is ALSO from that time, or from the later live row.
fs=[]
for d in Q.DAYS:
    f=pd.read_parquet(Q.FEAT/f'{d}_features.parquet',columns=['day','m','gid','kind','t','o_depth','fp_shown'])
    f['day']=f.day.astype(str); fs.append(f[f.kind=='first'])
F=pd.concat(fs,ignore_index=True)
assert not F.duplicated(Q.K).any()
J=S.merge(F.drop(columns='kind'),on=Q.K,suffixes=('','_first'),validate='one_to_one')
assert len(J)==len(S)
J['price_display_first']=np.where(J.o_depth_first<0,'inside_best',np.where(J.o_depth_first==0,'at_touch','behind_best'))+'|'+np.where(J.fp_shown_first>0,'shown_positive','shown_zero')
J['group_changed']=J.price_display!=J.price_display_first
clock=dict(states=len(J),later_than_first=int((J.t>J.t_first).sum()),group_changed=int(J.group_changed.sum()),
           current_touch_changed=int(((J.price_display==Q.PR)&J.group_changed).sum()))
clock['actual_first_touch']=dict(n=int((J.price_display_first==Q.PR).sum()), within_1=int(((J.price_display_first==Q.PR)&(J.client_minus_venue.abs()<1)).sum()))
J[J.group_changed][Q.K+['t','t_first','price_display','price_display_first','client_minus_venue']].head(12).to_csv(OUT/'group_clock_examples.csv',index=False)

# Identity of all manifest inputs and outputs, beyond family A's subset.
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
gm=E['gm']; audit=E['audit']
assert audit['inputs']==gm['inputs'] and audit['sources']==gm['sources']
assert audit['manifest_sha256']==sha(Q.R/'manifest.json')
assert all(sha(PIN/'predictfun/research'/n)==v for n,v in gm['sources'].items())
for n in ('states.csv','intervals.csv'):
    assert gm['inputs'][n]==hashlib.sha256(gzip.decompress(Q.blob(f'{Q.AQ}/outputs/{n}.gz'))).hexdigest()
for d in Q.DAYS: assert gm['inputs'][f'feat_{d}']==sha(Q.FEAT/f'{d}_features.parquet')
assert (Q.R/'outputs/audit.json').read_bytes()==(Q.D/'audit.json').read_bytes()

import pf_mutant_summary as MS, pf_mutant_one as MO, pf_ladder_queue_price_mutants as MH
logs=Q.R/'logs/mutants_try2'; recs={'control':MS.last_record(logs/'lqp2_control.log')}
recs.update({i:MS.last_record(logs/f'lqp2_{i}.log') for i in range(len(MH.MUTANTS))})
mut=MS.judge(len(MH.MUTANTS),recs,MO.sources(MH),[m[0] for m in MH.MUTANTS]); assert mut[0], mut

result=dict(target='012db7bbcdd861bbd19a984286935d1553c88db3',checks=base,falsifiers=fz,extra=extra,
            unchecked_reading_corruptions_pass=escapes,queue_error_group_clock=clock,saved_mutants=len(MH.MUTANTS),
            note='Detailed readings and all five tables independently verified correct now; frozen family P has narrower coverage.')
(OUT/'evidence.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
print(json.dumps(dict(checks=base['all_ok'],falsifiers=fz['falsifier_ok'],extra=extra,escapes=escapes,clock=clock,mutants=len(MH.MUTANTS)),indent=2))
