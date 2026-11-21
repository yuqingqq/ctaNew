"""Pinned movement-shrink review. Run only through PF_SLOT=codex run_light.sh.

Saved-parameter/evaluation reproduction and synthetic controls only; no new
candidate fit, tape lookup, real branch replay or untouched-day access.
"""
from pathlib import Path
import copy, gzip, hashlib, json, subprocess, sys, tempfile
ROOT=Path('/home/yuqing/ctaNew'); PIN=Path('/tmp/pf-review-530eac3b')
RES=PIN/'predictfun/research'; RC=PIN/'predictfun/reviews/movement_shrink_20261121'
OUT=ROOT/'predictfun/reviews/automatic/20261121_530eac3b'
TARGET='530eac3b230659f1d90c630aac9ecd0f80f139e5'
sys.path.insert(0,str(RES))
import numpy as np
import pandas as pd
import pf_movement_shrink as M
import pf_movement_shrink_selftest as ST
sys.path.insert(0,str(RC))
import shrink_checks as X
sys.path.insert(0,str(RES)); X.RES=RES; X.REPO=PIN; X.LOGS=RC/'logs'
X.REAL=dict(X.REAL,rec=RC,logs=X.LOGS)

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*a): return subprocess.check_output(['git','-C',str(PIN),*a])
def avg(x,w): return float(np.average(np.asarray(x,float),weights=np.asarray(w,float)))
def same(a,b): assert np.isclose(a,b,rtol=1e-12,atol=1e-9),(a,b)

def metadata():
    assert git('rev-parse','HEAD').decode().strip()==TARGET
    names=[]
    for line in (RC/'outputs/SHA256SUMS').read_text().splitlines():
        h,n=line.split(); p=RC/'outputs'/n; assert sha(p)==h,n
        b=gzip.decompress(p.read_bytes()) if n.endswith('.csv.gz') else p.read_bytes()
        assert b==(M.OUT/(n[:-3] if n.endswith('.csv.gz') else n)).read_bytes(),n
        names.append(n)
    frozen=[]
    for rel in ['predictfun/research/'+n for n in ('pf_movement_shrink.py','pf_movement_shrink_selftest.py','pf_movement_shrink_mutants.py')]+[
        M.RECEIPT+'/'+n for n in ('manifest.json','declaration.md','shrink_checks.py','launch_shrink.sh')]:
        assert git('show','97208136:'+rel)==(PIN/rel).read_bytes(),rel
        frozen.append(rel)
    assert git('show','bd5e2f01:'+M.RECEIPT+'/fit.json')==(RC/'fit.json').read_bytes()
    for commit,rel in [('97208136',M.RECEIPT+'/fit.json'),('bd5e2f01',M.RECEIPT+'/outputs/audit.json')]:
        p=subprocess.run(['git','-C',str(PIN),'cat-file','-e',commit+':'+rel],capture_output=True)
        assert p.returncode!=0,(commit,rel)
    assert M.frozen() and M.inputs_bound()
    man=read(RC/'manifest.json'); fit=read(RC/'fit.json')
    assert fit['sources']==M.sources() and fit['manifest_sha256']==sha(RC/'manifest.json')
    for h,kk in man['train'].items():
        assert M.keys_sha(kk)==man['train_keys_sha256'][h]==fit['train_keys_sha256'][h]
        assert not set(kk)&set(man['eval'][h])
        assert all(json.loads(k)[0]=='20261026' for k in kk)
        assert all(json.loads(k)[0]=='20261027' for k in man['eval'][h])
    su=X.suites(X.LOGS/'mutants_shr'); assert su['ok'],su
    assert all(Path(z.__file__).is_relative_to(RES) for n,z in sys.modules.items() if n.startswith('pf_') and getattr(z,'__file__',None))
    return dict(target=TARGET,artifacts_byte_equal=names,freeze_files_unchanged=frozen,fit_commit_bound=True,
                freeze_without_fit=True,fit_commit_without_scores=True,mutants=su,source_imports_pinned=True)

def data():
    ck,det=X.run_checks(X.REAL); assert all(ck.values()),(ck,det)
    ar,msha,O=M.load(); man=read(RC/'manifest.json'); fit=read(RC/'fit.json'); M.check_manifest(man,O,msha)
    assert M.populations(O)['statuses']==man['statuses']
    assert len(O)==987 and len(ar)==329
    # Re-evaluate the frozen scalar parameters; no training call or new candidate.
    with tempfile.TemporaryDirectory(dir=OUT) as td:
        q=Path(td); js=M.score_all(O,man,fit['params'],q); audit=read(M.OUT/'audit.json')
        for k,v in js.items(): assert v==audit[k],k
        names=['eval_rows','scores','calibration','actions','null_draws']
        for n in names: assert (q/(n+'.csv')).read_bytes()==(M.OUT/(n+'.csv')).read_bytes(),n
    D=X.load(X.REAL); groups=[]; actions=[]; bias=[]
    for stage in M.HEADS:
        for day in (M.TRAIN_DAY,M.EVAL_DAY):
            T=O[(O.stage==stage)&(O.day==day)&(O.status=='ok')]
            for side,g in [('all',T)]+[(str(s),T[T.side==s]) for s in (0,1)]:
                bias.append(dict(stage=stage,day=day,side=side,n=len(g),fills=int(g.z.sum()),w_sum=float(g.weight.sum()),
                                 original_movement_bias=avg(g.Eqm-g.M,g.weight),original_value_bias=avg(g.V-g.value,g.weight),
                                 neutral_hold_value=avg(g.value,g.weight)))
        T=O[(O.stage==stage)&(O.day==M.TRAIN_DAY)&(O.status=='ok')]
        for s in (0,1):
            g=T[T.side==s]; w=g.weight.to_numpy(float); mu=float((w*g.M).sum()/(w*g.z).sum())
            lo=g.Eq*(-g.mid if s==0 else g.mid-100); hi=g.Eq*(100-g.mid if s==0 else g.mid)
            a=np.clip(g.P*mu,lo,hi).to_numpy(); d=a-g.Eqm.to_numpy(); y=g.M.to_numpy()-g.Eqm.to_numpy()
            lam=float(np.linalg.lstsq((np.sqrt(w)*d)[:,None],np.sqrt(w)*y,rcond=None)[0][0])
            saved=fit['params'][f'{stage}|{s}']; same(mu,saved['mu']); same(lam,saved['lam_raw'])
            assert saved['lam']==1. and lam>1.; groups.append(dict(head=stage,side=s,mu=mu,lam_raw=lam,lam=saved['lam']))
        E=D['R'][D['R']['head']==stage].reset_index(drop=True)
        for x in ('Eqm','V'): assert np.array_equal(E[x+'_candidate'],E[x+'_prior'])
        orig=E.mask_original.to_numpy(bool); cand=E.mask_candidate.to_numpy(bool)
        switches=dict(both_chosen=int((orig&cand).sum()),removed=int((orig&~cand).sum()),added=int((~orig&cand).sum()),both_not_chosen=int((~orig&~cand).sum()))
        ac=D['AC'][(D['AC']['head']==stage)&(D['AC'].side=='all')].set_index('rule')
        for rule in M.RULES:
            m=E['mask_'+rule].to_numpy(bool); same(float((E.weight*m*E.value).sum()/E.weight.sum()),ac.loc[rule,'J'])
        selected=E[cand]
        actions.append(dict(head=stage,n=len(E),switches=switches,candidate_selected=len(selected),candidate_filled=int(selected.z.sum()),
                            candidate_positive=int((selected.value>1e-9).sum()),candidate_negative=int((selected.value<-1e-9).sum()),
                            candidate_zero=int((selected.value.abs()<=1e-9).sum()),J={k:float(ac.loc[k,'J']) for k in M.RULES},
                            null=read(M.OUT/'audit.json')['null'][stage]))
    pd.DataFrame(bias).to_csv(OUT/'day_comparison.csv',index=False)
    return dict(target=TARGET,checks=ck,details=det,admitted_rows=329,all_stage_rows=len(O),tables_byte_reproduced=names,
                parameters_verified_by_linear_solve=groups,action_sets=actions)

def controls():
    ST.run(M); X.selftest()
    with tempfile.TemporaryDirectory(dir=OUT) as td:
        cases,ok=X.falsify_on(X.REAL,td,True)
    assert ok,cases
    return dict(target=TARGET,synthetic_selftests=True,falsifier_ok=ok,cases=cases)

def instrument():
    with tempfile.TemporaryDirectory(dir=OUT) as td:
        Q=X.copy_tree(X.REAL,Path(td)/'copy'); clean,_=X.run_checks(Q); assert all(clean.values())
        p=Path(Q['out'])/'eval_rows.csv'; R=pd.read_csv(p,float_precision='round_trip',dtype=X.STR)
        # A decision-time metadata field later consumers could use; numeric scores unchanged.
        R.loc[0,'mid']=float(R.loc[0,'mid'])+1.; R.to_csv(p,index=False)
        changed,_=X.run_checks(Q); assert all(changed.values()),changed
        assert p.read_bytes()!=(M.OUT/'eval_rows.csv').read_bytes()
    return dict(target=TARGET,clean=clean,changed_mid=changed,all_checks_pass_after_mid_change=True,
                current_full_row_bytes_independently_reproduced=True,limitation='A-J do not bind every saved decision-context column; reuse exact committed rows or add field comparisons')

if __name__=='__main__':
    mode=sys.argv[1]; result=globals()[mode]()
    (OUT/(mode+'.json')).write_text(json.dumps(result,indent=2,sort_keys=True,default=str)+'\n')
    print(json.dumps(dict(mode=mode,passed=True)))
