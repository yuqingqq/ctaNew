"""Pinned-source, saved-artifact review. No historical replay or tape load.
Run only through PF_SLOT=codex run_light.sh; outputs stay in this review.
"""
from pathlib import Path
import collections, copy, gzip, hashlib, importlib.util, json, math, sys
import numpy as np
import pandas as pd

PIN = Path('/tmp/pf-af-review-85127b3d')
ROOT = Path('/home/yuqing/ctaNew')
OUT = Path(__file__).resolve().parent
REL = 'predictfun/reviews/ladder_async_fast_20261122'
sys.path.insert(0, str(PIN / 'predictfun/research'))
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
import pf_ladder_async_fast as X
import pf_ladder_async_fast_selftest as ST
old_data = X.A.DATA
try:
    X.A.DATA = OUT
    ST.main(X)
finally:
    X.A.DATA = old_data
Q = module('review_async_checks', PIN / REL / 'fa_checks.py')
Q.REPO = PIN; Q.R = PIN / REL
E = Q.load(); E['mine'] = Q.mine(E)
base = Q.run_checks(E); assert base['all_ok'], base
name, fam, target, old_fake = Q.CASES[-1]
def fake_in_review(e):
    old = e['out']; e['out'] = OUT
    try: old_fake(e)
    finally: e['out'] = old
Q.CASES[-1] = (name, fam, target, fake_in_review)
fz = Q.falsify(E, base); assert fz['falsifier_ok'], fz
sha = lambda b: hashlib.sha256(b).hexdigest()
gm = E['gm']
assert X.sources() == gm['sources'] == E['audit']['sources']
assert X.profile_sha() == gm['profile_sha256'] == E['audit']['profile_sha256']
assert X.QF.treatment_sha() == gm['treatment_sha256']
for d in gm['days']:
    for part in ('pilot','rest'):
        if not gm['parts'][part][d]: continue
        r = json.loads((Q.D/part/f'{d}.json').read_text())
        assert r['sources'] == gm['sources']
        assert r['manifest_sha256'] == sha((Q.R/'manifest.json').read_bytes())
        assert r['members'] == gm['parts'][part][d]
        assert r['status']=='DAY_DONE' and r['day']==d and r['part']==part
        assert r['arms']==list(X.RUN_ARMS) if part=='pilot' else r['arms']==list(X.RUN_ARMS)[1:]
for n in ('arms','parity','classes','treatment','blocks'):
    assert gzip.decompress((Q.R/'outputs'/f'{n}.csv.gz').read_bytes()) == (Q.D/'measure'/f'{n}.csv').read_bytes()
assert (Q.R/'outputs/audit.json').read_bytes() == (Q.D/'measure/audit.json').read_bytes()
PC = module('review_pressure_checks', PIN/'predictfun/reviews/ladder_pressure_20261121/pressure_checks.py')
assert Path(PC.P.__file__).resolve().is_relative_to(PIN)
def same(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b), (set(a)-set(b),set(b)-set(a))
        for k in a: same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): same(x,y)
    elif isinstance(a,(str,bool)) or a is None: assert a==b,(a,b)
    else: assert (pd.isna(a) and pd.isna(b)) or math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-6),(a,b)
verified_metrics=0; rebuilt_rows=[]; class_rows=[]; moves=[]; events=0; near=[]
for (d,m),lst in sorted(E['runs'].items()):
    rr=lst[0]; end=rr['path_info']['end_rel']; part=rr['part']; tm={}
    qa=np.load(Q.QD/part/f'{d}_arrays.npz')
    ar=dict(BASELINE=E['parent'][(d,m)]['disabled'],**Q.arm_recs(E,(d,m)))
    for arm,rec in ar.items():
        assert len(rec['fills'])==len(rec['fill_mids'])
        if arm=='BASELINE': z=E['pz'][d]; src='disabled'
        elif arm in Q.REUSE: z=qa; src=Q.REUSE[arm]
        else: z=E['z'][(d,part)]; src=arm
        pref=f'{m}__{src}__'; arr={k[len(pref):]:z[k] for k in z.files if k.startswith(pref)}
        row=E['arms'].query('day==@d and m==@m and arm==@arm'); assert len(row)==1; row=row.iloc[0].to_dict()
        v=PC.recompute(rec,end)
        for k in PC.RECON: same(v[k],row[k]); verified_metrics+=1
        cash=sum(-(1 if f[2]==0 else -1)*f[3]*f[4] for f in rec['fills'])
        pos=sum((1 if f[2]==0 else -1)*f[3] for f in rec['fills'])
        same([cash,pos,cash+pos*rec['mark']],[rec['cash'],rec['position'],rec['value']])
        same(v['all_id_mo1s']+v['all_id_move']+v['all_id_late'],rec['value'])
        assert rec['status']=='ok' and rec['end']==end
        assert rec['mark']==ar['CURRENT_BASELINE']['mark'] and rec['tape_id']==ar['CURRENT_BASELINE']['tape_id']
        new=dict(X.LG.arm_row(d,m,arm,rec,arr,end),part=part); same(new,row); rebuilt_rows.append(new)
        moves.append(dict(day=d,m=m,arm=arm,mo1s=v['all_id_mo1s'],move_to_end=v['all_id_move'],late=v['all_id_late'],end=rec['value']))
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
        # Full fingerprint/log/command parity is covered by Q.fam_K above.
        # Independently cross-check tape-event metadata across saved arms.
        for f,c in zip(rec['fills'],rec['fill_class']):
            it=int(f[5]); meta=(f[0],int(f[2]),c[1],c[2])
            if it in tm: assert tm[it]==meta
            tm[it]=meta; events+=1
        accepted=[o for o in od.values() if not o['refused']]
        assert len(accepted)==rec['treatment']['accepted_new']
        if arm.startswith('FRONT'): assert all(o['ahead_final']==0 for o in accepted)
        tr=E['treatment'].query('day==@d and m==@m and arm==@arm'); assert len(tr)==1
        for k,value in rec['treatment'].items(): same(value,tr.iloc[0][k])
        same(len(accepted),tr.iloc[0]['accepted'])
        same(sum(float(v[2])>0 and float(v[1])>0 for v in rec['queue_venue']['ahead0'].values()),tr.iloc[0]['accepted_with_ahead'])
        # Independent complete class reconstruction, including deferred 5/30s fields.
        for cls in ('same_price','through'):
            k=collections.Counter(); cnt=collections.Counter(int(f[5]) for f in rec['fills'])
            for f,mm,c in zip(rec['fills'],rec['fill_mids'],rec['fill_class']):
                if c[0]!=cls: continue
                t,_,s,q,p,it=f; sg=1 if s==0 else -1
                k['fills']+=1;k['q']+=q;k['end']+=sg*q*(rec['mark']-p);k['shared_trade_fills']+=cnt[int(it)]>1
                if t<=end-1000 and mm[6] is not None:
                    value=sg*q*(mm[6]-p);k['el_q']+=q;k['mo1s']+=value;k['pos1s']+=max(value,0);k['neg1s']+=min(value,0)
                for h,j in ((5,8),(30,10)):
                    if mm[j] is None:k[f'cens{h}']+=1
                    else:k[f'el{h}_q']+=q;k[f'mo{h}s']+=sg*q*(mm[j]-p)
            cr=E['classes'].query('day==@d and m==@m and arm==@arm and cls==@cls'); assert len(cr)==1
            names=cr.columns.difference(['day','m','arm','cls'])
            for key in names: same(k[key],cr.iloc[0][key])
            class_rows.append(dict(day=d,m=m,arm=arm,cls=cls,**{key:k[key] for key in names}))
        place=10 if arm in Q.NEW else 58
        gl=rec['guard_log']; old=sum(e[10]=='pressure_withhold' and e[0]+58>=end for e in gl)
        actual=sum(e[10]=='pressure_withhold' and e[0]+place>=end for e in gl)
        near.append(dict(day=d,m=m,arm=arm,frozen_window=old,profile_window=actual))
    qa.close()
RR=pd.DataFrame(rebuilt_rows); KK=pd.DataFrame(class_rows); MM=pd.DataFrame(moves)
same(X.predicates(RR),E['audit']['predicates'])
# Demonstrate exactly the inherited uncovered fields, with actual exports checked above.
escapes={}
bad=E['classes'].copy();bad.loc[0,'mo30s']+=99999
escapes['wrong_class_mo30s']=Q.fam_M(dict(E,classes=bad))[0]['M_measurement']
bad=copy.deepcopy(E['audit']);bad['predicates']['end_negative']['by_day']['20261024']['FRONT_ASYNC_FAST10']=False
escapes['wrong_day_absolute_sign']=Q.fam_P(dict(E,audit=bad))[0]['P_readings']
bad=E['arms'].copy();bad.loc[0,'both_exact']+=0.1
escapes['wrong_uptime']=Q.fam_M(dict(E,arms=bad))[0]['M_measurement']
assert all(escapes.values()),escapes
import pf_mutant_summary as MS
import pf_mutant_one as MO
import pf_ladder_async_fast_mutants as MH
ld=Q.R/'logs/mutants_try1'; records={'control':MS.last_record(ld/'lfa_control.log')}
records.update({i:MS.last_record(ld/f'lfa_{i}.log') for i in range(len(MH.MUTANTS))})
mut_ok,mut_bad=MS.judge(len(MH.MUTANTS),records,MO.sources(MH),[m[0] for m in MH.MUTANTS]);assert mut_ok,mut_bad
proj=json.loads((Q.R/'logs/projection.json').read_text())
same(proj['batch_spent']+(proj['projected_rest_cpu_s']+proj['tail_s'])/3600,proj['batch_projected'])
same(proj['line_now']+(proj['projected_rest_cpu_s']+proj['tail_s'])/3600+.03,proj['line_projected'])
assert proj['run_rest'] and proj['batch_projected']<=.25 and proj['line_projected']<=2.5
summary=MM.groupby('arm')[['mo1s','move_to_end','late','end']].sum().to_dict('index')
evidence=dict(target='85127b3dfbdfb8689159fe8b9e76b447324b7cfc',selftest_pass=True,checks=base,falsifier=fz,
    mutants_rejudged=len(MH.MUTANTS),markets=len(E['runs']),arms_reconstructed=len(RR),class_rows_reconstructed=len(KK),
    independently_recomputed_arm_fields=verified_metrics,event_consistency_rows=events,source_and_receipt_identity=True,
    full_predicates_verified=True,projection_verified=True,checker_escapes=escapes,arm_decomposition=summary,
    near_end_label=pd.DataFrame(near).groupby('arm')[['frozen_window','profile_window']].sum().to_dict('index'),
    limits=['No market tape reloaded, no real replay. Trade metadata checked across arms, pinned source and synthetic controls.',
            'Full arm helper reproduction supplements independent economic checks, not an independent simulator.',
            'Consumed Oct24-27, 19 complete markets, fixed external flow, no profitability/validation claim.'])
(OUT/'evidence.json').write_text(json.dumps(evidence,indent=2,default=str)+'\n')
MM.to_csv(OUT/'end_decomposition.csv',index=False)
print(json.dumps({k:evidence[k] for k in ('markets','arms_reconstructed','class_rows_reconstructed','mutants_rejudged','checker_escapes','arm_decomposition','near_end_label')},indent=2))
print('PINNED ASYNC REVIEW PASS; saved evidence and synthetic controls only.')
