"""Pinned ENTRY-RETENTION review: saved artifacts, inference and synthetic controls only."""
from pathlib import Path
import collections, copy, gzip, hashlib, inspect, io, json, os, subprocess, sys
ROOT=Path('/home/yuqing/ctaNew'); PIN=Path('/tmp/pf-review-a44f7dd8')
OUT=ROOT/'predictfun/reviews/automatic/20261121_a44f7dd8'
RES=PIN/'predictfun/research'; RC=PIN/'predictfun/reviews/entry_retention_20261121'
sys.path.insert(0,str(RES))
import numpy as np
import pandas as pd
import pf_entry_retention as M
import pf_entry_retention_selftest
sys.path.insert(0,str(RC))
import retention_checks as X
import retention_checks_v2 as V2
sys.path.insert(0,str(RES)); X.RES=RES; X.LOGS=RC/'logs'
assert Path(M.__file__).parent==RES
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(['git','-C',str(PIN),*a])
def equal(a,b):
    a=pd.read_csv(io.StringIO(a.to_csv(index=False)),float_precision='round_trip')
    b=pd.read_csv(io.StringIO(b.to_csv(index=False)),float_precision='round_trip')
    pd.testing.assert_frame_equal(a,b,check_dtype=False,rtol=1e-12,atol=1e-10)
def metadata():
    hh=[]
    for line in (RC/'outputs/SHA256SUMS').read_text().splitlines():
        h,p=line.split();assert sha(RC/'outputs'/p)==h,p;hh.append(p)
        data=M.OUT/p
        if p=='stages.csv.gz':assert gzip.decompress((RC/'outputs'/p).read_bytes())==(M.OUT/'stages.csv').read_bytes()
        else:assert (RC/'outputs'/p).read_bytes()==data.read_bytes(),p
    for n in ('pf_entry_retention.py','pf_entry_retention_selftest.py','pf_entry_retention_mutants.py'):
        assert git('show','2ac6e084:predictfun/research/'+n)==(RES/n).read_bytes()
    assert git('show','2ac6e084:predictfun/reviews/entry_retention_20261121/retention_checks.py')==(RC/'retention_checks.py').read_bytes()
    a=inspect.getsource(X.falsify).replace('def falsify(', 'def falsify_v2(').replace('    global LOGS\n','')
    a=a.replace('args_c, args_x = (d, clean_e) if on == \'out\' else (clean, d)', "args = (d, clean_e) if on == 'out' else (clean, d)")
    # Compare the cases and declarations directly; v2 also qualifies frozen module globals.
    assert len(read(RC/'checks_falsifier_v2.json')['cases'])==11
    old=read(RC/'logs/pilot_projection.json'); frozen=json.loads(git('show','0f89182f:predictfun/reviews/entry_retention_20261121/logs/pilot_projection.json'))
    assert {k:v for k,v in old.items() if k!='note'}=={k:v for k,v in frozen.items() if k!='note'}
    _,_,fz=M.L.teacher_assets(); R={};assert X.check_H(R)
    assert all(Path(z.__file__).is_relative_to(RES) for n,z in sys.modules.items() if n.startswith('pf_') and getattr(z,'__file__',None))
    return dict(receipt_hashes=len(hh),frozen_sources_unchanged=True,teacher_sources=len(fz['sources']),mutants=R,pilot_restoration_matches_committed_copy=True)

def data():
    man=read(M.EXPO/'manifest.json'); n=collections.Counter(); rows=[]; exall={}; staged=[]; schema={}
    Mo,sup,fz=M.L.teacher_assets()
    for k,pt in enumerate(man['parts']):
        rec=read(M.OUT/f'parts/part_{k:03d}.json'); b=(M.OUT/f'parts/part_{k:03d}.jsonl.gz').read_bytes()
        assert hashlib.sha256(b).hexdigest()==rec['sha256'] and rec['sources']==M.sources()
        assert rec['manifest_sha256']==sha(M.EXPO/'manifest.json')
        assert rec['part']==k and rec['worlds']==pt['worlds'] and rec['day']==pt['day'] and rec['set']==pt['set']
        ep=M.EXPO/f'parts/part_{k:03d}.jsonl.gz';er=read(ep.with_name(f'part_{k:03d}.json'))
        assert sha(ep)==er['sha256']==rec['exposure_part_sha256']
        assert er==read(PIN/f'predictfun/reviews/entry_exposure_20261121/outputs/parts/part_{k:03d}.json')
        es=X.expo_rows(M.EXPO,k); rr=M.LS.gz_rows(b)
        assert [X.KEY(r) for r in rr]==[(str(w['day']),w['m'],w['t_ms'],s) for w in pt['worlds'] for s in (0,1)]
        assert rec['rows']==len(rr)
        for r in rr:
            key=X.KEY(r);e=es[key];exall[key]=e;rows.append(r);n[r['status']]+=1
            if r['status']!='ok':assert r['status']==e['status'];continue
            assert all(r['checks'][a] for a in ('horizon_ok','prefix_same','callbacks_ok','equivalent','causal_ok','moments_ok'))
            assert r['checks']['pull_reconciled'] in (True,None)
            assert not X.b_row(r,e) and not X.c_row(r,e),key
            assert all(a['settings']==M.BE.SETTINGS for a in e['arms'].values())
            g0=r['gid0'];o=e['arms']['POST_PI0']['orders'][g0]
            assert o['side']==r['side'] and o['act']==r['held']['act'] and o['submit']==0.
            for nm,x in r['stages'].items():
                n[nm+':'+x['status']]+=1
                if x['status']!='ok':continue
                assert x['hidden_same'] and x['side']==r['side'] and x['t']==x['view']['t']
                assert x['state']==('absent' if nm=='entry' else 'live')
                assert x['head']==('post' if nm=='entry' else 'retain')
                if nm!='entry':assert x['gid']==g0 and x['effect']==x['t']+M.BE.CANCEL and x['venue']['act']==o['act']
                if x.get('inputs'):
                    a=np.asarray(x['inputs'],np.float32);assert len(a)==107 and hashlib.sha256(a.tobytes()).hexdigest()==x['inputs_sha256']
                    assert x['price']==o['price']; mo=x['moments'];assert mo['reproduced'] and mo['v_identity']
                    assert all(x['logged'].get(z) is None or x['logged'][z]==mo[z] for z in ('P','V'))
                    assert x['dec_status']!='model' or mo['ok']
                    staged.append((r['side'],x))
                n['decision:'+nm+':'+x['dec_status']]+=1
            rv,pp=r['stages']['first_review'],r['stages']['first_pull']
            if pp.get('same_as_review'):
                assert {k:v for k,v in pp.items() if k not in ('stage','same_as_review')}=={k:v for k,v in rv.items() if k not in ('stage','same_as_review')}
            ff=r['held']['fills'];fte=r['fills'];assert len(ff)==len(fte)
            for f,t in zip(ff,fte):
                assert (f[0],f[3],f[4])==(t['t'],t['q'],t['price'])
                assert abs(t['end_value']-(1 if r['side']==0 else -1)*f[3]*(r['held']['mark']-f[4]))<1e-9
    # Batch fresh inference from every saved vector, including same-as-review duplicates and refused forecasts.
    for side in (0,1):
        for head in ('post','retain'):
            xs=[x for s,x in staged if s==side and x['head']==head]
            df=pd.DataFrame([dict(zip(M.FEATS,x['inputs']),price=x['price']) for x in xs])
            pr=M.T.predict_rows(Mo[(side,head)],df,side)
            for i,x in enumerate(xs):
                for z in ('P','Eq','Eqm','V','ok'):
                    assert pr[z][i]==x['moments'][z],(side,head,i,z)
    S=M.stage_table(rows)
    equal(S,pd.read_csv(M.OUT/'stages.csv',float_precision='round_trip'))
    for name,fn in (('calibration',M.calibration),('bins',M.bins),('decisions',M.decisions),('changes',M.changes)):
        equal(pd.DataFrame(fn(S)),pd.read_csv(M.OUT/f'{name}.csv',float_precision='round_trip'))
    assert json.loads(json.dumps(M.predicates(S),default=str))==read(M.OUT/'retention.json')['predicates']
    # Independent additional diagnostics, not new frozen gates or new candidate selection.
    scores=[]
    for nm in M.STAGES:
        E=M.eligible(S,nm)
        for (day,side),g in [(('all','all'),E)]+list(E.groupby(['day','side'])):
            y=g.value.to_numpy(float); v=g.V.to_numpy(float); mech=g.mech.to_numpy(float)
            scores.append(dict(stage=nm,day=str(day),side=str(side),n=len(g),mse_V=float(np.mean((v-y)**2)),mse_zero=float(np.mean(y*y)),
                mse_mech=float(np.mean((mech-y)**2)),mse_in_sample_mean=float(np.var(y)),mae_V=float(np.mean(abs(v-y))),mae_zero=float(np.mean(abs(y))),
                bias=float(np.mean(v-y)),mean_value=float(y.mean()),mse_movement=float(np.mean((g.Eqm-g.qm)**2)),mse_qm_zero=float(np.mean(g.qm**2))))
    (OUT/'squared_errors.csv').write_text(pd.DataFrame(scores).to_csv(index=False))
    # Features by actual change, separately observed without attributing causality.
    changes=collections.Counter(); stage_pair_counts=collections.Counter()
    for r in rows:
        if r['status']!='ok':continue
        a,b=r['stages']['entry'],r['stages']['first_review']
        if a.get('inputs') and b.get('inputs'):
            stage_pair_counts[(a['dec_status'],b['dec_status'])]+=1
            aa=np.array(a['inputs'],np.float32);bb=np.array(b['inputs'],np.float32)
            changes.update(n for n,c in zip(M.FEATS,aa.view(np.uint32)!=bb.view(np.uint32)) if c)
    return dict(counts=dict(n),all_fields_and_summaries_reproduced=True,prediction_vectors_recomputed=len(staged),
        scores=[x for x in scores if x['day']=='all'],top_changed_inputs=changes.most_common(20),stage_status_pairs={str(k):v for k,v in stage_pair_counts.items()},
        limits='Reference mids for fill-time edges not re-extracted from raw tape; no real branch reconstruction in review.')

def controls():
    os.environ['PF_TEACHER_SYN']=str(ROOT/'data/predict_fun/research/r2_teacher_selftest_world')
    M.selftest();X.selftest()
    R={};checks={name:f(M.OUT,M.EXPO,R) for name,f in X.CHECKS};checks['H']=X.check_H(R);assert all(checks.values()),checks
    V2.HERE=OUT;V2.falsify_v2();fals=read(OUT/'checks_falsifier_v2.json');assert fals['falsifier_ok']
    # Exact conditional mean can lose MAE to zero with a zero-inflated payoff.
    rr=[]
    for st in ('entry','first_review'):
        for i in range(10):
            y=-100. if i<3 else 0.
            rr.append(dict(stage=st,day='synthetic',m=i,t_ms=i,hour=0,side=0,fork_status='ok',stage_status='ok',dec_status='model',
                action='WAIT' if st=='entry' else 'PULL',P=.3,Eq=9.,Eqm=-30.,V=-30.,edge=0.,mech=0.,q_fill=30.,qm_fill=-100.,q=30. if y else 0.,value=y,qm=y))
    S=pd.DataFrame(rr);pred=M.predicates(S);cal=M.calibration(S)
    assert pred['readings']['forecasts_fail']
    assert cal[0]['mae_V']==42. and cal[0]['mae_zero']==30.
    y=np.array([-100.]*3+[0.]*7);assert np.mean((-30.-y)**2)==2100. and np.mean(y*y)==3000.
    return dict(checks=checks,readings=R,falsifier=fals,mean_forecast_counterexample=dict(true_mean=-30.,forecast=-30.,mae=42.,mae_zero=30.,mse=2100.,mse_zero=3000.,declared_forecasts_fail=True))

def interpretation():
    import ast, re
    from datetime import datetime,timedelta
    a=inspect.getsource(X.falsify).replace('def falsify(', 'def falsify_v2(').replace('    global LOGS\n','')
    a=a.replace('args_c, args_x =', 'args =').replace("safe(chk, *args_c, {}) if on == 'out' else safe(chk, *args_x, {})",'safe(chk, *args, {})')
    a=a.replace('checks_falsifier.json','checks_falsifier_v2.json')
    b=inspect.getsource(V2.falsify_v2)
    b=re.sub(r'X\.(check_[A-H]|LOGS)',r'\1',b)
    assert ast.dump(ast.parse(a),include_attributes=False)==ast.dump(ast.parse(b),include_attributes=False)
    # Recover the pilot calculation from completed job logs before the first non-pilot part.
    jobs=[]
    for p in (RC/'logs').rglob('*.log'):
        for line in p.read_text().splitlines():
            m=X.RL.match(line)
            if m:
                start=datetime.fromisoformat(m[1].replace('Z','+00:00'))
                jobs.append(dict(tag=m[2],start=start,end=start+timedelta(seconds=float(m[5])),cpu=float(m[6])))
    rest=min(j['start'] for j in jobs if j['tag'].startswith('retention_part_') and j['tag']!='retention_part_000')
    cpu=sum(j['cpu'] for j in jobs if j['end']<=rest)
    pj=read(RC/'logs/pilot_projection.json')
    assert abs(cpu/3600.-pj['so_far_core_hours'])<1e-10
    assert abs((pj['pilot_cpu_s']*12+cpu)/3600.-pj['projected_core_hours'])<1e-10
    # A changing age and time-to-expiry sensitivity are not new market events.
    import pf_influence as IN
    dd=dict(k=np.array([0.]),imb5=np.array([.1]),imb20=np.array([.2]),ce=np.array([0.]),cd=np.array([100.]))
    import pf_stop_model_x5 as D5
    f=D5.depth_feats(dd,np.array([50.,100.]));assert f['bnd_age'][1]!=f['bnd_age'][0] and f['bnd_imb5'][1]==f['bnd_imb5'][0]
    sa,sb=IN.sens_of(50.,.01,10.),IN.sens_of(50.,.01,9.9)
    assert sa!=sb and M.group_of('sens')=='fair' and M.group_of('bnd_age')=='market_flow'
    S=pd.read_csv(M.OUT/'stages.csv',float_precision='round_trip')
    R=S[S.stage=='first_review'];E=S[S.stage=='entry'].set_index(['day','m','t_ms','side'])
    paired=R.set_index(['day','m','t_ms','side']).join(E[['dec_status']],rsuffix='_entry')
    both=paired[(paired.dec_status=='model')&(paired.dec_status_entry=='model')&(paired.stage_status=='ok')]
    # Size the prospective admission cohort using existing traces only; do NOT read D_EXPOSURE by admission here.
    root=M.EX.A.DATA/'adv2_entry_smooth_v1_20261121/replay_v1'; counts=collections.Counter(); gen=collections.Counter()
    for k in range(12):
        rec=read(root/f'parts/part_{k:03d}.json');p=root/f'parts/part_{k:03d}.jsonl.gz'
        assert sha(p)==rec['sha256'] and rec==read(PIN/f'predictfun/reviews/r4e_entry_smooth_20261121/replay/part_{k:03d}.json')
        for r in M.LS.gz_rows(p.read_bytes()):
            counts[str(r['day'])+':'+r['status']]+=1
            if r['status']!='ok':continue
            ds=r['episodes']['PI0']['trace']['decision_list']
            posts=[d for d in ds if d[2]=='absent' and d[3]=='POST' and d[4] is not None]
            assert len(posts)==len({d[4] for d in posts})
            for d in posts:
                absolute_ms=float(r['t_ms'])+float(d[0])
                action_day=datetime.fromtimestamp(absolute_ms/1000.,tz=__import__('datetime').timezone.utc).strftime('%Y%m%d')
                assert action_day==str(r['day'])
                gen[(action_day,int(d[1]),int(absolute_ms//3600000)%24)]+=1
    cells=[dict(day=k[0],side=k[1],hour=k[2],generations=v,sample=min(4,v),weight=v/min(4,v)) for k,v in sorted(gen.items())]
    return dict(falsifier_amendment_only_declared_plumbing=True,pilot_cpu_before_rest=cpu,pilot_projection_reconstructed=pj['projected_core_hours'],
        clock_only_control=dict(bnd_age=list(f['bnd_age']),fair_sensitivity=[sa,sb],no_new_snapshot=True),
        both_modeled_transition=dict(n=len(both),flips=int(both.flip.sum()),share=float(both.flip.mean())),
        admission_population=dict(world_statuses=dict(counts),total_POST_generations=sum(gen.values()),nonempty_cells=len(cells),max_four_per_cell_sample=sum(c['sample'] for c in cells),cells=cells))

if __name__=='__main__':
    out=globals()[sys.argv[1]]();(OUT/(sys.argv[1]+'.json')).write_text(json.dumps(out,indent=2,sort_keys=True,default=str)+'\n');print(json.dumps(out,sort_keys=True,default=str))
