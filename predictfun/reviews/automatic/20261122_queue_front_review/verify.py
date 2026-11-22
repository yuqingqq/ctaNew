"""Bounded review of pinned source and saved records; no historical replay.
Run with PF_SLOT=codex run_light.sh. Writes only review-owned evidence.
"""
from pathlib import Path
import collections, copy, gzip, hashlib, importlib.util, json, math, sys
import numpy as np
import pandas as pd

PIN = Path('/tmp/pf-qf-review-c7404f21')
ROOT = Path('/home/yuqing/ctaNew')
OUT = Path(__file__).resolve().parent
REL = 'predictfun/reviews/ladder_queue_front_20261122'
sys.path.insert(0, str(PIN / 'predictfun/research'))

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

import pf_ladder_queue_front as X
import pf_ladder_queue_front_selftest as ST
ST.main(X)
Q = module('review_front_checks', PIN / REL / 'fq_checks_v2.py')
Q.REPO = PIN; Q.R = PIN / REL
E = Q.load()
E['mine_t'] = Q.my_totals(E); E['mine_p'] = Q.my_predicates(E['mine_t'], E)
base = Q.run_checks(E); assert base['all_ok'], base
name, fam, target, old_fake = Q.CASES[-1]
def fake_in_review(e):
    out = e['out']; e['out'] = OUT
    try: old_fake(e)
    finally: e['out'] = out
Q.CASES[-1] = (name, fam, target, fake_in_review)
fz = Q.falsify(E, base); assert fz['falsifier_ok'], fz

sha = lambda b: hashlib.sha256(b).hexdigest()
gm = E['gm']
assert X.sources() == gm['sources']
assert X.treatment_sha() == gm['treatment_sha256']
assert E['audit']['sources'] == gm['sources']
assert E['audit']['treatment_sha256'] == gm['treatment_sha256']
assert E['audit']['control_all'] and E['audit']['markets'] == 19
for (d,p), r in E['recs'].items():
    assert r['sources'] == gm['sources']
    assert r['manifest_sha256'] == sha((Q.R/'manifest.json').read_bytes())
    assert r['members'] == gm['parts'][p][d]
    assert r['status']=='DAY_DONE' and r['day']==d and r['part']==p
    assert r['parent_day_record_sha256'] == gm['parent']['day_records'][d]
for n in ('arms','control','lost_added','classes','treatment'):
    assert gzip.decompress((Q.R/'outputs'/f'{n}.csv.gz').read_bytes()) == (Q.D/'measure'/f'{n}.csv').read_bytes()
assert (Q.R/'outputs/audit.json').read_bytes() == (Q.D/'measure/audit.json').read_bytes()

PC = module('review_pressure_checks', PIN/'predictfun/reviews/ladder_pressure_20261121/pressure_checks.py')
# Its import finds the already loaded pinned module, never the active checkout.
assert Path(PC.P.__file__).resolve().is_relative_to(PIN)

def same(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b), (set(a)-set(b),set(b)-set(a))
        for k in a: same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): same(x,y)
    elif isinstance(a,(str,bool)) or a is None:
        assert a==b,(a,b)
    else:
        assert (pd.isna(a) and pd.isna(b)) or math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-6),(a,b)

verified_metrics = 0; reconstructed=[]; class_rows=[]; movements=[]; trades_checked=0
for (d,m), lst in sorted(E['runs'].items()):
    r=lst[0]; end=r['path_info']['end_rel']; part=r['part']
    trade_meta={}
    for arm in ('BASELINE',)+Q.ARMS:
        rec=E['parent'][(d,m)]['disabled'] if arm=='BASELINE' else r[arm]
        assert len(rec['fills'])==len(rec['fill_mids'])
        z=E['pz'][d] if arm=='BASELINE' else E['z'][(d,part)]
        prefix=f'{m}__'+('disabled' if arm=='BASELINE' else arm)+'__'
        arr={k[len(prefix):]:z[k] for k in z.files if k.startswith(prefix)}
        saved=E['arms'].query('day==@d and m==@m and arm==@arm')
        assert len(saved)==1
        sr=saved.iloc[0].to_dict()
        # Independent fill/accounting reconstruction from the accepted checker.
        vals=PC.recompute(rec,end)
        for k in PC.RECON: same(vals[k],sr[k]); verified_metrics+=1
        cash=sum(-(1 if f[2]==0 else -1)*f[3]*f[4] for f in rec['fills'])
        pos=sum((1 if f[2]==0 else -1)*f[3] for f in rec['fills'])
        same([cash,pos,cash+pos*rec['mark']],[rec['cash'],rec['position'],rec['value']])
        same(vals['all_id_mo1s']+vals['all_id_move']+vals['all_id_late'],rec['value'])
        assert rec['status']=='ok' and rec['end']==end
        assert rec['tape_id']==r[Q.C_]['tape_id'] and rec['settings']==r[Q.C_]['settings']
        assert rec['mark']==r[Q.C_]['mark']
        # Reproduce all exported arm fields, including unchanged helper metrics.
        rebuilt=dict(X.LG.arm_row(d,m,arm,rec,arr,end),part=part)
        same(rebuilt,sr); reconstructed.append(rebuilt)
        movements.append(dict(day=d,m=m,arm=arm,mo1s=vals['all_id_mo1s'],move_to_end=vals['all_id_move'],late=vals['all_id_late'],end=rec['value']))
        od={int(o['gid']):o for o in rec['orders']}; by_gid=collections.Counter()
        for f,mm in zip(rec['fills'],rec['fill_mids']):
            t,gid,s,q,p,it=f; o=od[int(gid)]
            assert 0<=t<=end and o['act']<=t<o['remove'] and q>0 and p==o['price'] and s==o['side']
            by_gid[gid]+=q
            for h,j in ((0,0),(50,2),(100,4),(1000,6),(5000,8),(30000,10)):
                if t+h>end: assert mm[j] is None
                if mm[j] is not None: assert np.isfinite(mm[j]) and 0<=mm[j]<=100
        for gid,q in by_gid.items(): assert q<=od[gid]['size']+1e-7
        if arm=='BASELINE': continue
        # Check internal event consistency across both arms, including trade time/side.
        for f,c in zip(rec['fills'],rec['fill_class']):
            it=int(f[5]); meta=(f[0],int(f[2]),c[1],c[2])
            if it in trade_meta: assert meta==trade_meta[it]
            trade_meta[it]=meta; trades_checked+=1
        assert len(rec['queue_venue']['ahead0'])==len(od)
        accepted=[o for o in od.values() if not o['refused']]
        assert len(accepted)==rec['treatment']['accepted_new']
        if arm==Q.F_:
            assert all(o['ahead_final']==0 for o in accepted)
        # Independently verify all class fields, including native 5/30s censoring.
        for cls in ('same_price','through'):
            k=collections.Counter()
            counts=collections.Counter(int(f[5]) for f in rec['fills'])
            for f,mm,c in zip(rec['fills'],rec['fill_mids'],rec['fill_class']):
                if c[0]!=cls: continue
                t,_,s,q,p,it=f; sg=1 if s==0 else -1
                k['fills']+=1;k['q']+=q;k['end']+=sg*q*(rec['mark']-p)
                k['shared_trade_fills']+=counts[int(it)]>1
                if t<=end-1000 and mm[6] is not None:
                    v=sg*q*(mm[6]-p); k['el_q']+=q;k['mo1s']+=v;k['pos1s']+=max(v,0);k['neg1s']+=min(v,0)
                for h,j in ((5,8),(30,10)):
                    if mm[j] is None:k[f'cens{h}']+=1
                    else:k[f'el{h}_q']+=q;k[f'mo{h}s']+=sg*q*(mm[j]-p)
            row=E['classes'].query('day==@d and m==@m and arm==@arm and cls==@cls')
            assert len(row)==1
            for name in row.columns.difference(['day','m','arm','cls']):same(k[name],row.iloc[0][name])
            class_rows.append(dict(day=d,m=m,arm=arm,cls=cls,**{name:k[name] for name in row.columns.difference(['day','m','arm','cls'])}))
    # All lost/added fields, not just counts and the stored identity_gap.
    lr=E['lost_added'].query('day==@d and m==@m').iloc[0]
    expected={}
    for nm,a,b in (('lost',r[Q.C_],r[Q.F_]),('added',r[Q.F_],r[Q.C_])):
        mask=PC.matched_split(a['fills'],b['fills']); k=collections.Counter()
        for f,mm,on in zip(a['fills'],a['fill_mids'],mask):
            if not on:continue
            t,_,s,q,p,_=f; sg=1 if s==0 else -1; en=sg*q*(a['mark']-p)
            k['n']+=1;k['q']+=q;k['end']+=en;k['end_pos']+=max(en,0);k['end_neg']+=min(en,0)
            if t<=end-1000 and mm[6] is not None:
                v=sg*q*(mm[6]-p);k['mo1s_pos']+=max(v,0);k['mo1s_neg']+=min(v,0)
        expected.update({f'{nm}_{n}':k[n] for n in ('n','q','end','end_pos','end_neg','mo1s_pos','mo1s_neg')})
    expected['delta_value']=r[Q.F_]['value']-r[Q.C_]['value']
    expected['identity_gap']=expected['delta_value']-(expected['added_end']-expected['lost_end'])
    for k,v in expected.items():same(v,lr[k])

RR=pd.DataFrame(reconstructed); KK=pd.DataFrame(class_rows)
pr=X.predicates(RR,KK); same(pr,E['audit']['predicates'])
# Independently rebuilt totals and declared answers passed above; helper reproduction
# fills the missing full JSON coverage without claiming independent policy replay.
escapes={}
zz=E['classes'].copy();zz.loc[0,'mo30s']+=99999
escapes['wrong_30s_class_value']=Q.fam_M(dict(E,classes=zz))[0]['M_measurement']
zz=E['lost_added'].copy();zz.loc[0,'added_end']+=99999
escapes['wrong_added_end_with_stored_gap_unchanged']=Q.fam_M(dict(E,lost_added=zz))[0]['M_measurement']
zz=copy.deepcopy(E['audit']);zz['predicates']['end_negative']['by_day']['20261024'][Q.F_]=False
escapes['wrong_day_absolute_sign']=Q.fam_P(dict(E,audit=zz))[0]['P_readings']
assert all(escapes.values()),escapes
# The independent reconstruction above detects each same change.
assert not math.isclose(KK.iloc[0].mo30s,E['classes'].iloc[0].mo30s+99999,abs_tol=1e-6)

import pf_mutant_summary as MS
import pf_mutant_one as MO
import pf_ladder_queue_front_mutants as MH
ld=Q.R/'logs/mutants_try1'
records={'control':MS.last_record(ld/'lqf_control.log')}
records.update({i:MS.last_record(ld/f'lqf_{i}.log') for i in range(len(MH.MUTANTS))})
mut_ok,mut_bad=MS.judge(len(MH.MUTANTS),records,MO.sources(MH),[m[0] for m in MH.MUTANTS]); assert mut_ok,mut_bad

M=pd.DataFrame(movements)
summary=M.groupby('arm')[['mo1s','move_to_end','late','end']].sum().to_dict('index')
report=dict(target='c7404f212700ebf1c6977e1806a28c23b5961f02',selftest_pass=True,
            frozen_checks=base,falsifiers=fz,mutants_rejudged=len(MH.MUTANTS),
            markets=len(E['runs']),arms_reconstructed=len(RR),class_rows_reconstructed=len(KK),
            independently_recomputed_arm_fields=verified_metrics,fill_event_consistency_rows=trades_checked,
            full_predicates_verified=pr,identity_sources_and_receipt_copies=True,
            checker_escapes=escapes,arm_decomposition=summary,
            limitations=['No raw market tape reloaded; class metadata mapping verified in pinned source and synthetic controls, and cross-arm event consistency.',
                         'Full arm helper reproduction complements independent economic checks; it is not an independent simulator.'])
(OUT/'evidence.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
M.to_csv(OUT/'end_decomposition.csv',index=False)
print(json.dumps({k:report[k] for k in ('markets','arms_reconstructed','class_rows_reconstructed','mutants_rejudged','checker_escapes','arm_decomposition')},indent=2))
print('PINNED SAVED-EVIDENCE REVIEW: PASS; no historical replay, training or new day.')
