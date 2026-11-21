"""Independent R2-F artifact review; no fitting, feature building, or tape replay.
Run under PF_SLOT=codex run_light.sh. Pinned imports; one partition at a time.
"""
import ast
import collections as co
import copy
import csv
import gzip
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path('/home/yuqing/ctaNew'); PIN=Path('/tmp/pf-review-f68fd805')
OUT=ROOT/'predictfun/reviews/automatic/20261121_f68fd805'
RC=PIN/'predictfun/reviews/r2f_production_20261120'
DATA=ROOT/'data/predict_fun/research/r2_label_sample_v2_20261120'
P=DATA/'production_v1'; PAR=DATA/'parity_pilot_e35'; OLD=DATA/'attempt4_pilot_A'
TARGET='f68fd805ca6ba52c411b7d570aaa791f10d17feb'
FREEZE='d3d024fd2c5ccb1ca7e4d33e9b1162b91b50abd5'
sys.path.insert(0,str(PIN/'predictfun/research'))
import numpy as np
import pandas as pd
import pf_r2_label_sample as S

def read(p): return json.loads(Path(p).read_text())
def digest(b): return hashlib.sha256(b).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4<<20),b''): h.update(b)
    return h.hexdigest()
def canon(x): return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True)
def rows(d,p):
    rp,fp,qp=S.part_paths(d,p['part']); b,q=fp.read_bytes(),qp.read_bytes()
    return read(rp),S.gz_rows(b),S.gz_rows(q),(b,q)
def metadata():
    plan=read(P/'plan.json'); pp=read(PAR/'plan.json'); old=read(OLD/'plan.json')
    for name,d in [('production',P),('parity',PAR)]:
        assert gzip.decompress((RC/f'plan/{name}_plan.json.gz').read_bytes())==(d/'plan.json').read_bytes()
        assert sha(d/'plan.json')==(RC/f'plan/{name}_plan.sha256').read_text().split()[0]
    assert plan['sample']==pp['sample']==old['sample'] and plan['audit_every']==pp['audit_every']==35
    assert len(plan['partitions'])==96 and all(len(p['members'])==32 for p in plan['partitions'])
    assert plan['sources']==S.sources_now()==pp['sources'] and plan['settings']==S.BE.SETTINGS
    assert plan['feature_schema']==dict(n=107,sha256=digest(canon(S.T.FEATS).encode()))
    for n,h in plan['sources'].items():
        assert digest(subprocess.check_output(['git','show',f'{FREEZE}:predictfun/research/{n}'],cwd=ROOT))==h
    fz=S.T.frozen(); S.L.libraries_enforced(fz); assert sha(S.T.REAL['out']/'freeze.json')==plan['freeze_sha256']
    for n,h in fz['models'].items(): assert sha(S.T.REAL['out']/'models'/(n+'.txt'))==h
    files={}
    for day,sp in plan['sample'].items():
        inp=sp['inputs']; kp=Path(plan['kp_dir'])/day
        files[str(kp/'receipt.json')]=inp['kp_receipt']; files.update({str(kp/n):h for n,h in inp['kp_outputs'].items()})
        files.update(inp['files']);files.update(inp['tracked'])
        assert S.KP.Tape().offset(day)==inp['offset'] and S.X.I.X5.depth_hours(day)==inp['hours']
    for p,h in files.items(): assert sha(p)==h,p
    assert read(P/'accepted.json')==read(RC/'checkpoint/accepted.json')
    nh=0
    for ln in (RC/'checkpoint/checkpoint.sha256').read_text().splitlines():
        h,p=ln.split(); assert sha(P/p)==h,p;nh+=1
    import pf_r2_label_sample_mutants as H
    import pf_mutant_one as O
    import pf_mutant_summary as M
    recs={}
    for i in ['control']+list(range(len(H.MUTANTS))):
        lines=(RC/f'logs/mutants_v2/mut_v2_{i}.log').read_text().splitlines()
        rr=[json.loads(x) for x in lines if x.startswith('{') and '"cls"' in x];assert len(rr)==1
        recs[i]=rr[0]
    assert len(H.MUTANTS)==113 and M.judge(113,recs,O.sources(H),[x[0] for x in H.MUTANTS])[0]
    M.selftest()
    return dict(plan_sha256=sha(P/'plan.json'),data_sha256=read(P/'accepted.json')['dataset_sha256'],sources=len(plan['sources']),
                models=len(fz['models']),hashed_inputs=len(files),input_bytes=sum(Path(p).stat().st_size for p in files),checkpoint_hashes=nh,
                mutants=113,params=S.T.PARAMS,rounds=S.T.ROUNDS,features=S.T.FEATS,context=S.T.R2.CTX)

def audit():
    plan=read(P/'plan.json'); psha=sha(P/'plan.json'); acc=read(P/'accepted.json'); receipt=read(RC/'checks.json')
    assert all(receipt['checks'].values()) and receipt['plan_sha256']==psha
    records=[]; slim=[]; status=co.Counter(); worldst={}; unique=set(); seenparts=[]; counters=co.Counter(); controls={}; nonok=[]; zeroarms=[]
    nearzero=[]; groups=co.defaultdict(list); paths=co.Counter(); feature_nan=co.Counter(); fmin=np.full(107,np.inf);fmax=-fmin.copy()
    for part in plan['partitions']:
        rec,rr,qq,raw=rows(P,part); assert not S.validate(plan,psha,part,rr,qq,rec,raw)
        records.append(rec); seenparts.append(dict(part=part['part'],day=part['day'],sha256=rec['sha256'],pools_sha256=rec['pools_sha256'],rows=len(rr)))
        assert rec['elapsed_s']<=600+120*len(part['members']) and rec['max_pair_s']<=120 and rec['hwm_mib']<=1536
        for r in rr:
            key=(r['day'],r['m'],r['t_ms'],r['side'],r['head'],r['pick']);assert key not in unique;unique.add(key)
            status[f"{r['day']} {r['head']} {r['status']}"]+=1
            if r['status'].startswith('world:'):worldst[key[:3]]=r['status']
            if r['status'] not in S.PICKED:continue
            res=r['result']; assert res['status']=='ok'
            arms=res['arms']; ns=S.NAMES[r['head']]; a,b=[arms[n] for n in ns];g=res['k']*S.L.STEP
            assert res['end']==5000 and res['ident']['end_abs_ms']-res['ident']['start_abs_ms']==5000
            assert res['fork_first_side']==int(res['k'])%2
            assert abs(res['Y']-(a['value']-b['value']))<=1e-6
            for n in ns:
                arm=arms[n]; fl=arm['fill_list']; cash=pos=0.;gen=co.Counter()
                for t,gid,side,q,price,ti in fl:
                    assert 0<=t<=5000 and q>=0 and 0<=price<=100 and all(math.isfinite(v) for v in (t,q,price))
                    sign=1 if side==0 else -1;pos+=sign*q;cash-=sign*q*price;gen[gid]+=q
                assert abs(pos-arm['position'])<=1e-9 and abs(cash-arm['cash'])<=1e-6
                assert arm['mark']==res['mark'] and abs(cash+pos*res['mark']-arm['value'])<=1e-6
                assert all(q<=arm['sizes'][gid]+1e-9 for gid,q in gen.items())
                assert S.L._h(arm['decision_list'])==arm['decisions_sha256']
                # Independent callback denominator: the decision log only records eligible states.
                eligible=[x for x in arm['decision_list'] if x[0]>g or (x[0]==g and x[1]!=r['side'] and res['fork_first_side']==r['side'])]
                fb={(x[0],x[1]) for x in arm['fallback_list']}
                expected=sum((x[0],x[1]) not in fb for x in eligible)
                assert all(x[2] in ('absent','live') for x in arm['decision_list'])
                assert arm['teacher']['calls']==arm['teacher']['binding_checked']==expected and arm['teacher']['binding_bad']==0
                assert sum(arm['teacher']['by_action'].values())==expected==sum(arm['teacher']['by_status'].values())
                if not expected:zeroarms.append(dict(key=key,arm=n,decision_ms=g,Y=res['Y']))
            for n,c in zip(ns,r['causal']):
                assert c['checked']==math.ceil(arms[n]['teacher']['calls']/35) and c['mismatch']==c['parity_mismatch']==0
            if r['status']!='ok':
                nonok.append(dict(key=key,status=r['status'],fork=r['fork'],Y=res['Y'],decision_ms=g));continue
            x=np.asarray(r['fork']['inputs'],np.float32);assert digest(x.tobytes())==r['fork']['inputs_sha256']
            assert not np.isinf(x).any(); good=np.isfinite(x); fmin[good]=np.minimum(fmin[good],x[good]);fmax[good]=np.maximum(fmax[good],x[good])
            feature_nan.update({S.T.FEATS[i]:1 for i in np.flatnonzero(~good)})
            counters['binding_checked']+=a['teacher']['calls']+b['teacher']['calls']
            counters['zero_callback_arms']+=int(a['teacher']['calls']==0)+int(b['teacher']['calls']==0)
            for c in r['causal']:
                for k in ('checked','mismatch','parity_checked','parity_mismatch'):counters[k]+=c[k]
            y=res['Y']; row=dict(key=key,day=r['day'],side=r['side'],head=r['head'],source=r['source'],Y=y,w=r['w'],remaining_ms=5000-g,
                                support=r['fork']['support'],partial=res['fork_known_filled']>0)
            slim.append(row);groups[f"{r['day']} {r['side']} {r['head']}"].append(row)
            same=a['fill_list']==b['fill_list'];cls='identical' if same else 'economic' if y!=0 else 'path_only' if a['cash']==b['cash'] and a['position']==b['position'] else 'offsetting'
            paths[cls]+=1
            if 0<abs(y)<=1e-6:nearzero.append(row)
        if part['part'] in (0,48):
            base=next(i for i,r in enumerate(rr) if r['status']=='ok'); name=S.NAMES[rr[base]['head']][0]
            corrupt={'END':lambda r:r['result'].__setitem__('end',4000),'Y':lambda r:r['result'].__setitem__('Y',r['result']['Y']+1),
                     'hidden_failed_pair':lambda r:(r.__setitem__('status','feature_unavailable'),r['fork'].__setitem__('status','feature_unavailable'),r['result'].__setitem__('reconciled',False)),
                     'false_pair_status':lambda r:r.__setitem__('status','pair_not_ok'),
                     'skipped_callback':lambda r:r['result']['arms'][name]['teacher'].__setitem__('binding_checked',0)}
            for n,fn in corrupt.items():
                cr=copy.deepcopy(rr);fn(cr[base]);bad=S.validate(plan,psha,part,cr,qq);assert bad
                controls[f"{part['part']} {n}"]=bad[:1]
    assert acc['rows']==len(unique) and acc['statuses']==dict(status)
    assert acc['parts']==seenparts and acc['dataset_sha256']==digest(canon(seenparts).encode())
    assert counters['binding_checked']==receipt['readings']['causal']['binding_checked']
    assert all(counters[k]==receipt['readings']['causal'][k] for k in ('checked','parity_checked','zero_callback_arms','mismatch','parity_mismatch'))
    ys=list(csv.DictReader((RC/'y_summary.csv').open()));ws=list(csv.DictReader((RC/'weights.csv').open()))
    for z in ys:
        subset=[r for r in slim if (z['day']=='both' or r['day']==z['day']) and (z['head']=='both' or r['head']==z['head'])]
        y=np.array([r['Y'] for r in subset]);nz=[r for r in subset if r['Y']!=0]
        assert len(y)==int(z['n_ok']) and (y==0).sum()==int(z['zero']) and (y>0).sum()==int(z['positive']) and (y<0).sum()==int(z['negative'])
        assert len({r['key'][:4] for r in nz})==int(z['originals_nonzero_union']) and len({r['key'][:3] for r in nz})==int(z['worlds_nonzero'])
        assert abs(y.mean()-float(z['mean_c']))<1e-10
    for z in ws:
        sub=[r for r in slim if (z['day']=='both' or r['day']==z['day']) and r['head']==z['head']];w=np.array([r['w'] for r in sub]);y=np.array([r['Y'] for r in sub])
        assert abs(w.sum()-float(z['ok_weight_total']))<1e-7 and abs((w*y).sum()/w.sum()-float(z['weighted_mean_Y_c']))<1e-10
    stats={}
    for key,rs in groups.items():
        w=np.array([r['w'] for r in rs]);y=np.array([r['Y'] for r in rs]);nw=[r for r in rs if abs(r['Y'])>1e-6]
        stats[key]=dict(rows=len(rs),originals=len({r['key'][:4] for r in rs}),positive=int((y>0).sum()),negative=int((y<0).sum()),
                       nonzero_originals=len({r['key'][:4] for r in rs if r['Y']!=0}),material_nonzero_originals=len({r['key'][:4] for r in nw}),
                       weighted_mean=float(w@y/w.sum()),weighted_MSE_zero=float(w@(y*y)/w.sum()),
                       events_by_source=dict(co.Counter(r['source'] for r in nw)),rows_by_source=dict(co.Counter(r['source'] for r in rs)))
    return dict(parts=96,rows=len(unique),ok=len(slim),status=dict(status),no_path_worlds=len(worldst),non_ok_picked=nonok,counts=dict(counters),
                zero_callback_arms=zeroarms,near_zero_nonzero=nearzero,paths=dict(paths),group_support=stats,controls=controls,
                feature_missing=dict(feature_nan),constant_features=[n for n,a,b in zip(S.T.FEATS,fmin,fmax) if a==b],all_checks_pass=True)

def parity():
    po=read(OLD/'plan.json');pn=read(PAR/'plan.json'); assert po['sample']==pn['sample'] and po['partitions']==pn['partitions']
    def normalized(r):
        r=copy.deepcopy(r);r.pop('causal',None)
        for a in r['result']['arms'].values():a.pop('decision_list',None);a.pop('fallback_list',None)
        return canon(r)
    nr=nq=0
    for part in pn['partitions']:
        rec,rr,qq,raw=rows(PAR,part);_,oldr,oldq,_=rows(OLD,part)
        assert not S.validate(pn,sha(PAR/'plan.json'),part,rr,qq,rec,raw)
        assert len(rr)==len(oldr) and qq==oldq
        for a,b in zip(rr,oldr):assert normalized(a)==normalized(b)
        changed=copy.deepcopy(rr[0]);changed['result']['Y']+=1;assert normalized(changed)!=normalized(oldr[0])
        neutral=copy.deepcopy(rr[0]);neutral['causal'][0]['checked']+=1;assert normalized(neutral)==normalized(oldr[0])
        nr+=len(rr);nq+=len(qq)
    assert nr==288 and nq==192
    return dict(rows=nr,pools=nq,byte_equal_payload=True,positive_and_bad_controls=True)

def receipts():
    a=RC/'checks_equivalence/v11';b=RC/'checks_equivalence/v12'
    fs=sorted(p.name for p in a.glob('*.csv'));assert len(fs)==8
    assert all((a/f).read_bytes()==(b/f).read_bytes() for f in fs)
    ja,jb=read(a/'checks.json'),read(b/'checks.json');ja.pop('asof');jb.pop('asof');assert ja==jb
    changed=copy.deepcopy(jb);changed['checks']['A_plan']=not changed['checks']['A_plan'];assert changed!=ja
    rg=re.compile(r'tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
    jobs=[]
    for k in range(96):
        tag=f'r2f_run_{k}';ls=(RC/f'logs/{tag}.log').read_text().splitlines()
        found=[m for ln in ls for m in [rg.search(ln)] if m and m[1]==tag];assert len(found)==1
        m=found[0];assert int(m[2])==0 and int(m[3])<=1536
        jobs.append(dict(part=k,peak_mib=int(m[3]),wall=float(m[4]),cpu=float(m[5])))
    cost={r['item']:float(r['value']) for r in csv.DictReader((RC/'cost.csv').open())}
    assert abs(sum(x['cpu'] for x in jobs)/3600-cost['partition_cpu_core_hours'])<1e-9
    assert abs(sum(x['wall'] for x in jobs)/3600-cost['partition_wall_hours_sum'])<1e-9
    assert max(x['peak_mib'] for x in jobs)<=cost['peak_mib_jobs']
    assert not (P/'stops.jsonl').exists() or not (P/'stops.jsonl').read_text().strip()
    return dict(equal_csvs=fs,checks_equal_except_asof=True,comparison_bad_control=True,production_jobs=len(jobs),
                cpu_core_hours=sum(x['cpu'] for x in jobs)/3600,wall_hours_sum=sum(x['wall'] for x in jobs)/3600,peak_mib=max(x['peak_mib'] for x in jobs),
                zero_stops=True)

if __name__=='__main__':
    mode=sys.argv[1];result=globals()[mode]();result.update(target=TARGET,asof=datetime.now(timezone.utc).isoformat(),script_sha256=sha(__file__),mode=mode)
    (OUT/(mode+'.json')).write_text(json.dumps(result,indent=2,default=str)+'\n')
    print(json.dumps(dict(mode=mode,result='PASS',output=str(OUT/(mode+'.json')))))
