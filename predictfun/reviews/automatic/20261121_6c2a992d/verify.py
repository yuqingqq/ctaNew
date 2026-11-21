"""Read-only pinned admission review; saved artifacts, frozen inference, synthetic controls.

Run only through PF_SLOT=codex run_light.sh. No fits or real tape simulations.
"""
from pathlib import Path
import collections, copy, datetime, gzip, hashlib, io, json, os, subprocess, sys
ROOT=Path('/home/yuqing/ctaNew'); PIN=Path('/tmp/pf-review-6c2a992d')
RES=PIN/'predictfun/research'; RC=PIN/'predictfun/reviews/pi0_admission_20261121'
OUT=ROOT/'predictfun/reviews/automatic/20261121_6c2a992d'
sys.path.insert(0,str(RES))
import numpy as np
import pandas as pd
import pf_pi0_admission as M
import pf_pi0_admission_selftest as ST
sys.path.insert(0,str(RC))
import admission_checks as X
sys.path.insert(0,str(RES)); X.RES=RES; X.LOGS=RC/'logs'
assert Path(M.__file__).parent==RES
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*a): return subprocess.check_output(['git','-C',str(PIN),*a])
def rows(): return [r for _,_,b in X.parts(M.OUT) for r in M.LS.gz_rows(b)]
def metadata():
    checked=[]
    for line in (RC/'outputs/SHA256SUMS').read_text().splitlines():
        h,p=line.split(); assert sha(RC/'outputs'/p)==h,p
        if p in ('stages.csv.gz','generations.csv.gz'):
            assert gzip.decompress((RC/'outputs'/p).read_bytes())==(M.OUT/p[:-3]).read_bytes(),p
        else: assert (RC/'outputs'/p).read_bytes()==(M.OUT/p).read_bytes(),p
        checked.append(p)
    for n in ('pf_pi0_admission.py','pf_pi0_admission_selftest.py','pf_pi0_admission_mutants.py'):
        assert git('show','6b14953f:predictfun/research/'+n)==(RES/n).read_bytes(),n
    for n in ('admission_checks.py','manifest.json','declaration.md'):
        assert git('show','6b14953f:predictfun/reviews/pi0_admission_20261121/'+n)==(RC/n).read_bytes(),n
    assert (RC/'manifest.json').read_bytes()==(M.OUT/'manifest.json').read_bytes()
    _,_,freeze=M.L.teacher_assets(); R={}; assert X.check_H(R)
    assert all(Path(z.__file__).is_relative_to(RES) for n,z in sys.modules.items() if n.startswith('pf_') and getattr(z,'__file__',None))
    # Cross-tab source tables are also bound to the prior reviewed receipts.
    prior=[]
    for rel,base in [('entry_exposure_20261121/outputs',M.EX.OUT/'v2'),('entry_retention_20261121/outputs',M.RT.OUT)]:
        p=PIN/'predictfun/reviews'/rel
        name='forks.csv' if 'exposure' in rel else 'stages.csv'
        q=p/name
        if not q.exists(): q=p/(name+'.gz')
        assert q.exists(),q
        b=gzip.decompress(q.read_bytes()) if q.suffix=='.gz' else q.read_bytes()
        assert b==(base/name).read_bytes(),q
        prior.append(str(q.relative_to(PIN)))
    return dict(receipt_hashes=len(checked),teacher_sources=len(freeze['sources']),final_mutants=R,prior_tables=prior,freeze_unchanged=True)

def data():
    man=read(M.OUT/'manifest.json'); gens,worlds=M.census(M.r4e_parts()); selected,cells=M.select(gens)
    assert selected==man['selected'] and cells==man['cells'] and worlds==man['worlds']
    want={X.GKEY(g):g for g in selected}; rr=rows(); assert len(rr)==len(want)==329
    tr=X.traces(); count=collections.Counter(); staged=[]; used=set()
    for k,rec,b in X.parts(M.OUT):
        assert sha(M.OUT/f'parts/part_{k:03d}.jsonl.gz')==rec['sha256']
        assert rec==read(RC/f'outputs/parts/part_{k:03d}.json')
        assert rec['part']==k and rec['sources']==M.sources() and rec['manifest_sha256']==sha(M.OUT/'manifest.json')
        assert len(M.LS.gz_rows(b))==rec['rows']
    for r in rr:
        g=want[X.GKEY(r)]; used.add(X.WKEY(r)); count[r['status']]+=1
        for z in ('weight','cell','n_cell'): assert r[z]==g[z],(X.GKEY(r),z)
        assert r['status']=='ok' and all(v is not False for v in r['checks'].values())
        t=tr[X.WKEY(r)]; assert t['settings']==M.BE.SETTINGS
        assert not X.b_row(r,t) and not X.c_row(r,t),X.GKEY(r)
        ka=r['keep_arm']; assert M.L._h(ka['fill_list'])==ka['fills_sha256'] and M.L._h(ka['decision_list'])==ka['decisions_sha256']
        assert r['held']['mark']==ka['mark']==t['mark'] and r['held']['act']==r['decision_ms']+M.BE.PLACE
        assert all(0<=f[0]<=M.END for f in ka['fill_list'])
        for nm,x in r['stages'].items():
            count[nm+':'+x['status']]+=1
            if x['status']!='ok': continue
            assert x['hidden_same'] and x['dec_status']=='model' and x['moments']['ok']
            assert x['state']==('absent' if nm=='entry' else 'live')
            assert x['head']==('post' if nm=='entry' else 'retain')
            assert x['side']==r['side'] and x['t']==x['view']['t']
            if nm=='entry': assert x['t']==r['decision_ms'] and x['action']=='POST'
            else: assert x['gid']==r['gid'] and x['effect']==x['t']+M.BE.CANCEL
            a=np.asarray(x['inputs'],np.float32); assert len(a)==107 and hashlib.sha256(a.tobytes()).hexdigest()==x['inputs_sha256']
            mo=x['moments']; assert all(mo[z]==x['logged'][z] for z in ('P','V'))
            assert abs(mo['edge']*mo['Eq']+mo['Eqm']-mo['V'])<1e-8
            staged.append((r['side'],x))
        rv,pp=r['stages']['first_review'],r['stages']['first_pull']
        if pp.get('same_as_review'):
            assert {k:v for k,v in pp.items() if k not in ('stage','same_as_review')}=={k:v for k,v in rv.items() if k not in ('stage','same_as_review')}
        assert r['overrides']==len(r['shadow'])
        ncall=ka['teacher']['calls']; assert ka['teacher']['binding_checked']==ncall
        assert ka['causal']['checked']==(ncall+34)//35
        count['first_review_hour_changes']+=int(rv.get('status')=='ok' and M.cell_of(r['t_ms'],rv['t'],r['side'])!=tuple(r['cell']))
    Mo,_,_=M.L.teacher_assets()
    for s in (0,1):
        for h in ('post','retain'):
            xs=[x for side,x in staged if side==s and x['head']==h]
            df=pd.DataFrame([dict(zip(M.RT.FEATS,x['inputs']),price=x['price']) for x in xs]); pr=M.T.predict_rows(Mo[(s,h)],df,s)
            for i,x in enumerate(xs):
                for z in ('P','Eq','Eqm','V','ok'): assert pr[z][i]==x['moments'][z],(s,h,i,z)
    S=M.stage_rows(rr); diag=[]
    for nm in M.RT.STAGES:
        E=S[(S.stage==nm)&(S.dec_status=='model')]
        for (day,side),g in [(('all','all'),E)]+list(E.groupby(['day','side'])):
            w=g.weight.to_numpy(float); wm=lambda z:float(np.average(np.asarray(z,float),weights=w))
            ed=g.edge*g.q; dm=g.qm; pred_ed=g.mech; pred_dm=g.Eqm
            assert np.allclose(ed+dm,g.value,atol=1e-7)
            diag.append(dict(stage=nm,day=str(day),side=str(side),n=len(g),w_sum=float(w.sum()),fill_n=int((g.q>0).sum()),
                bias_V=wm(g.V-g.value),bias_edge_quantity=wm(pred_ed-ed),bias_qm=wm(pred_dm-dm),
                mean_pred_qm=wm(pred_dm),mean_actual_qm=wm(dm),mse_qm=wm((pred_dm-dm)**2),mse_qm_zero=wm(dm**2),
                mse_V=wm((g.V-g.value)**2),mse_zero=wm(g.value**2),mse_mech=wm((g.mech-g.value)**2),
                mean_P=wm(g.P),fill_rate=wm(g.q>0)))
    pd.DataFrame(diag).to_csv(OUT/'error_components.csv',index=False)
    R={}; checks={name:fn(M.OUT,R) for name,fn in X.CHECKS if name!='D_forecasts'}; assert all(checks.values()),R
    # Inspect concentration on the SAME weighted estimand, not a new optimized policy.
    dd=np.asarray([r['D_EXPOSURE'] for r in rr]); ww=np.asarray([r['weight'] for r in rr]); aw=np.sort(abs(dd*ww))[::-1]
    return dict(population=dict(census=len(gens),selected=len(rr),worlds=len(used),weights=sum(ww),cells=len(cells)),counts=dict(count),
                stored_vectors_repredicted=len(staged),checks=checks,details=R,weighted_top5_abs_D_share=float(aw[:5].sum()/aw.sum()),
                errors=[d for d in diag if d['day']=='all'])

def controls():
    ST.run(M); X.selftest(); X.HERE=OUT; X.falsify()
    j=read(OUT/'checks_falsifier.json'); assert j['falsifier_ok']
    return dict(synthetic_suites_passed=True,falsifiers=len(j['cases']),falsifier_ok=True)

def interpretation():
    rr=rows(); S=M.stage_rows(rr); E=S[(S.stage=='entry')&(S.dec_status=='model')]
    decomposition=[]
    for nm in M.RT.STAGES:
        g=S[(S.stage==nm)&(S.dec_status=='model')]; w=g.weight.to_numpy(float); avg=lambda x:float(np.average(x,weights=w))
        z=(g.q>0).astype(float)
        prob=(g.P-z)*g.qm_fill; cond=z*(g.qm_fill-g.qm)
        assert np.allclose(prob+cond,g.Eqm-g.qm,atol=1e-8)
        decomposition.append(dict(stage=nm,probability_contribution_c=avg(prob),conditional_movement_contribution_c=avg(cond),total_qm_bias_c=avg(g.Eqm-g.qm)))
    bn=M.bins(S); b=[r for r in bn if r['stage']=='entry' and r['n']]
    strictly_monotone=all(a['wfill_rate']<=c['wfill_rate'] for a,c in zip(b,b[1:]))
    # The age comes from BT. A fresh quote with unchanged trades can trigger new_bn_age.
    source=(RES/'pf_stop_model_review.py').read_text()
    assert "k, m, imb = BT; i = np.searchsorted" in source and "out['bn_age'] = np.where(ok, t - k[ic], np.nan)" in source
    r=copy.deepcopy(rr[0]); en=r['stages']['entry']; rv=r['stages']['first_review']
    if rv.get('status')=='ok':
        a0=list(en['inputs']); a1=list(a0); dt=rv['t']-en['t']
        for ag in M.AGES: a1[M.RT.FEATS.index(ag)]=a0[M.RT.FEATS.index(ag)]+dt
        a1[M.RT.FEATS.index('bn_age')]=0.
        en['inputs']=a0; rv['inputs']=a1; rv['view']['received']=en['view']['received']
        out=M.stage_rows([r]); x=out[out.stage=='first_review'].iloc[0]
        assert x.new_bn_age and not x.new_pf_age and not x.new_bnd_age
    # Check-A coverage control: content-authenticated row weights/cells are not compared to the manifest.
    oldparts=X.parts; partlist=list(oldparts(M.OUT)); k,rec,raw=partlist[0]; altered=M.LS.gz_rows(raw)
    altered[0]['weight']*=2; nb=M.LS.gz_bytes(altered); partlist[0]=(k,dict(rec,sha256=hashlib.sha256(nb).hexdigest()),nb)
    X.parts=lambda out:iter(partlist)
    try: escaped=X.check_A(M.OUT,{})
    finally: X.parts=oldparts
    assert escaped
    # This run's actual weights/cells WERE independently matched in data().
    for d in ('20261026','20261027'):
        group=E[E.day==d]; assert len(group)>0
    pj=read(RC/'logs/pilot_projection.json'); assert pj['projected_core_hours']<1 and pj['run_rest']
    proj=(pj['pilot_cpu_s']*pj['selected_generations']/pj['pilot_generations'])/3600+pj['so_far_core_hours']
    assert abs(proj-pj['projected_core_hours'])<1e-12
    return dict(moment_bias_decomposition=decomposition,entry_fill_bins_strictly_monotone=strictly_monotone,
                bn_age_source='Binance bookTicker; does not identify trade arrival',
                check_A_rehashed_wrong_weight_accepted=escaped,actual_row_weights_and_cells_independently_exact=True,
                pilot_projection_core_hours=proj,scope='post-hoc diagnostic arithmetic, no fitted corrections or new policy evaluation')

if __name__=='__main__':
    mode=sys.argv[1]; result=globals()[mode](); result['target']='6c2a992d8fc0e3e52b073c694687e674aa562f34'
    result['asof']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (OUT/(mode+'.json')).write_text(json.dumps(result,indent=2,sort_keys=True,default=str)+'\n')
    print(json.dumps(result,sort_keys=True,default=str))
