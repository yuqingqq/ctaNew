"""Independent R3 review; pinned imports, saved artifacts and synthetic checks only.
Run under PF_SLOT=codex run_light.sh; never fits or builds real labels.
"""
import collections as co
import copy
import gzip
import hashlib
import importlib
import json
import math
import subprocess
import sys
from pathlib import Path
ROOT=Path('/home/yuqing/ctaNew'); PIN=Path('/tmp/pf-review-76be4960')
OUT=ROOT/'predictfun/reviews/automatic/20261121_76be4960'
RC=PIN/'predictfun/reviews/r3_adv2_20261121'
sys.path.insert(0,str(PIN/'predictfun/research'))
import numpy as np
import pandas as pd
import pf_adv2 as A
import pf_adv2_eval as E
import pf_adv2_replay as R
import pf_adv2_replay_summary as RS
S=A.LS
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4<<20),b''): h.update(b)
    return h.hexdigest()
def git(*args): return subprocess.check_output(['git','-C',str(PIN),*args])
def same_frame(a,b):
    pd.testing.assert_frame_equal(a.reset_index(drop=True),b.reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-10)
def metadata():
    n=0
    for ln in (RC/'checkpoint/outputs.sha256').read_text().splitlines():
        h,p=ln.split(); assert sha(A.OUT/p)==h,p;n+=1
    plan=read(E.EVAL_DIR/'plan.json'); assert sha(E.EVAL_DIR/'plan.json')==E.EVAL_PLAN_SHA
    assert plan['sources']==S.sources_now()
    assert all('20261028' not in p for d in plan['sample'].values() for p in list(d['inputs']['files'])+list(d['inputs']['tracked']))
    files={}
    for day,d in plan['sample'].items():
        i=d['inputs']; kp=Path(plan['kp_dir'])/day
        files[str(kp/'receipt.json')]=i['kp_receipt'];files.update({str(kp/k):v for k,v in i['kp_outputs'].items()});files.update(i['files']);files.update(i['tracked'])
    for p,h in files.items(): assert sha(p)==h,p
    A.check_inputs();fz,heads,sup=A.frozen_fit()
    assert (RC/'fit/fit.json').read_bytes()==(A.OUT/'fit.json').read_bytes()
    assert git('show','2bec465d2d63de193795a5907626b5a948b0c835:predictfun/reviews/r3_adv2_20261121/fit/fit.json')==(A.OUT/'fit.json').read_bytes()
    for name in ('pf_adv2.py','pf_adv2_eval.py','pf_adv2_replay.py'):
        assert git('show',f'c8655f75251c71dcdd38eb4f78d3a536d4dec7a3:predictfun/research/{name}')==(PIN/'predictfun/research'/name).read_bytes()
    ex=read(A.OUT/'train/extract.json');Et=A.load_train();assert A.manifest_of(Et,ex['statuses'])['heads']==ex['heads']
    assert A.support_of(Et)==read(A.OUT/'train/support.json')
    import pf_mutant_one as MO
    import pf_mutant_summary as MS
    counts={}
    for tag,mod in [('sample','pf_adv2_sample'),('adv2','pf_adv2'),('eval','pf_adv2_eval'),('replay','pf_adv2_replay'),('rsummary','pf_adv2_replay_summary')]:
        H=importlib.import_module(mod+'_mutants');rr={}
        for k in ['control']+list(range(len(H.MUTANTS))):
            p=RC/f'logs/mutants_{tag}/mut_{tag}_{k}.log'
            lines=[json.loads(ln) for ln in p.read_text().splitlines() if ln.startswith('{') and '"cls"' in ln]
            assert len(lines)==1;rr[k]=lines[0]
        assert MS.judge(len(H.MUTANTS),rr,MO.sources(H),[m[0] for m in H.MUTANTS])[0]
        bad=copy.deepcopy(rr);bad[1]=bad[0];assert not MS.judge(len(H.MUTANTS),bad,MO.sources(H),[m[0] for m in H.MUTANTS])[0]
        counts[tag]=len(H.MUTANTS)
    return dict(checkpoint_hashes=n,source_files=len(plan['sources']),input_files=len(files),input_bytes=sum(Path(p).stat().st_size for p in files),model_hashes=len(fz['models']),mutants=counts,plan_sha256=sha(E.EVAL_DIR/'plan.json'))
def synthetic():
    for M in (E,R,RS): M.selftest()
    return dict(passed=['eval','replay','replay_summary'],model_fits=0,real_tape_runs=0)
def data():
    plan=read(E.EVAL_DIR/'plan.json');acc=read(E.EVAL_DIR/'accepted.json');psha=sha(E.EVAL_DIR/'plan.json');st=co.Counter();cnt=co.Counter();allrows=[]
    for part in plan['partitions']:
        rp,fp,qp=S.part_paths(E.EVAL_DIR,part['part']);raw=(fp.read_bytes(),qp.read_bytes());rr,qq=map(S.gz_rows,raw);rec=read(rp)
        assert not S.validate(plan,psha,part,rr,qq,rec,raw)
        for r in rr:
            st[f"{r['day']} {r['head']} {r['status']}"]+=1
            if r['status']=='ok':
                res=r['result'];av=[]
                for arm in res['arms'].values():
                    pos=sum((1 if f[2]==0 else -1)*f[3] for f in arm['fill_list']);cash=-sum((1 if f[2]==0 else -1)*f[3]*f[4] for f in arm['fill_list'])
                    assert all(0<=f[0]<=5000 for f in arm['fill_list'])
                    assert abs(pos-arm['position'])<1e-9 and abs(cash-arm['cash'])<1e-6
                    assert abs(cash+pos*res['mark']-arm['value'])<1e-6
                n1,n0=S.NAMES[r['head']];assert abs(res['Y']-(res['arms'][n1]['value']-res['arms'][n0]['value']))<1e-6
                for c in r['causal']: cnt['causal']+=c['checked'];cnt['parity']+=c.get('parity_checked',0)
            # Keep only the extraction fields to bound memory.
            allrows.append({k:r[k] for k in ('day','status','t_ms','side','head','m','oid','pick','source','pi','hour','row_key','w','fork','result') if k in r} if r['status']!='ok' else
                           {**{k:r[k] for k in ('day','status','t_ms','side','head','m','oid','pick','source','pi','hour','row_key','w','fork')},'result':{k:r['result'][k] for k in ('Y','status','kind','side')}})
    assert dict(st)==acc['statuses'];ex,st2=A.extract(allrows,('20261026','20261027'));allrows=None
    fz,heads,sup=A.frozen_fit();df=E.frame(ex,heads,sup,fz)
    saved=pd.read_csv(A.OUT/'eval/predictions.csv.gz',float_precision='round_trip',dtype={'day':str})
    same_frame(df[list(saved.columns)],saved)
    rd,sc=E.readings(df)
    for name,vals in [('readings',rd),('screens',sc),('calibration',E.calibration(df)),('coverage',E.coverage(df,'RICH'))]:
        fresh=pd.read_csv(__import__('io').StringIO(pd.DataFrame(vals).to_csv(index=False)))
        same_frame(fresh,pd.read_csv(A.OUT/f'eval/{name}.csv'))
    # Independent all-row error and action value, without the evaluator's formulas.
    readings=[]
    for (day,s,h),g in df.groupby(['day','side','head']):
        w=g.w.to_numpy();y=g.Y.to_numpy();p=g.pred_RICH.to_numpy();d=g.d_RICH.to_numpy();W=w.sum()
        readings.append(dict(day=day,side=int(s),head=h,n=len(g),mse_ratio=float(np.dot(w,(p-y)**2)/np.dot(w,y*y)),value=float(np.dot(w,d*y)/W),always1=float(np.dot(w,y)/W)))
    ctl=[]
    for c in A.CANDS:ctl+=E.control(df,c,['day','head'])+E.control(df,c,['day'])
    same_frame(pd.read_csv(__import__('io').StringIO(pd.DataFrame(ctl).to_csv(index=False))),pd.read_csv(A.OUT/'eval/control.csv'))
    # Independently re-extract the previously accepted training rows and compare all hashes.
    tr,ts=A.extract(A.dataset_rows(A.TRAIN['dir'],A.TRAIN['dataset_sha256']),A.TRAIN['days']);assert A.manifest_of(tr,ts)['heads']==read(A.OUT/'train/extract.json')['heads']
    return dict(parts=len(plan['partitions']),statuses=dict(st),rows=len(df),counters=dict(cnt),readings=readings,all_predictions_reloaded=True,training_extraction_exact=True,control_draws_recomputed=500)
def replay():
    plan=read(E.EVAL_DIR/'plan.json');parts=R.population(plan);worlds=co.Counter();ctr=co.Counter();reached=[];seen=set()
    for k,(day,ws) in enumerate(parts):
        rec=read(R.OUT/f'parts/part_{k:03d}.json');b=(R.OUT/f'parts/part_{k:03d}.jsonl.gz').read_bytes();rr=S.gz_rows(b)
        assert hashlib.sha256(b).hexdigest()==rec['sha256'] and rec['part']==k and rec['members']==[S.wkey(w) for w in ws]
        assert rec['fit_sha256']==sha(A.OUT/'fit.json') and rec['eval_plan_sha256']==E.EVAL_PLAN_SHA
        for name,h in rec['sources'].items():assert sha(PIN/'predictfun/research'/name)==h
        assert [(r['day'],r['m'],r['t_ms']) for r in rr]==[(day,w['m'],w['t_ms']) for w in ws]
        for r in rr:
            key=(r['day'],r['m'],r['t_ms']);assert key not in seen;seen.add(key);worlds[r['status']]+=1
            if r['status']!='ok':continue
            assert set(r['episodes'])==set(A.POLICIES)
            for name,e in r['episodes'].items():
                assert e['ledger_ok'] and e['cap_ok'] and e['binding_bad']==0 and e['binding_checked']==e['calls']
                assert math.isclose(e['value'],e['value_bid']+e['value_ask'],abs_tol=1e-6) and math.isclose(e['value'],e['mend'],abs_tol=1e-6)
                assert len(e['churn'])==e['reviewed_generations'] and sum(c['action']=='PULL' for c in e['churn'])==e['first_review_pulls']
                assert 0<=e['quote_ms_both']<=min(e['quote_ms_bid'],e['quote_ms_ask'])+1e-9<=e['quote_ms_either']+1e-9<=5000+1e-9
                ca=e['causal'];assert ca['mismatch']==ca['parity_mismatch']==0
                for kk in ('checked','parity_checked'):ctr[kk]+=ca.get(kk,0)
                ctr['binding_checked']+=e['binding_checked']
            assert sorted((x['key']['side'],x['key']['head']) for x in r['reached'])==sorted(A.HEADS)
            for x in r['reached']:
                if x['status']!='ok':assert x['status']=='no_eligible_decision';continue
                assert x['reconciled'] and x['binding_bad']==0 and x['rank']==R.rank_of(x['key']) and x['eligible']>0
                assert math.isclose(x['A'],x['pred']['RICH'],abs_tol=1e-12)
                if x['binding_checked']==0: assert x['key']['decision_ms']==4950 and x['key']['side']==0;ctr['zero_callback_pairs']+=1
                reached.append(dict(day=day,**x))
    e,r,st,ch,exp=RS.frames(R.OUT,len(parts))
    for name,df in zip(RS.CSVS,(e,r,ch,exp)):assert df.to_csv(index=False).encode()==(R.OUT/f'{name}.csv').read_bytes()
    js=RS.summary(e,r,st);old=read(R.OUT/'replay.json')
    for k,v in js.items():assert v==old[k],k
    for folder,src in [('eval',A.OUT/'eval'),('replay',R.OUT)]:
        for p in (RC/folder).glob('*'):
            if p.name.startswith('part_'):continue
            target=src/p.name
            if target.exists(): assert p.read_bytes()==target.read_bytes()
            elif p.suffix=='.gz': assert gzip.decompress(p.read_bytes())==(src/p.stem).read_bytes()
    # The existing summary mixes sampling-weighted cell means using row counts. Quantify its effect separately.
    cal=pd.read_csv(A.OUT/'eval/calibration.csv');pred=pd.read_csv(A.OUT/'eval/predictions.csv.gz');correct=[]
    for h in ('post','keep'):
        d=pred[(pred['head']==h)&(pred.support=='supported')].copy();d['bin']=d.pred_RICH.map(E.cbin)
        for bn,g in d.groupby('bin'):
            z=cal[(cal['head']==h)&(cal.candidate=='RICH')&(cal.bin==bn)]
            wrong=float(np.sum(z.rows*z.wmean_y)/z.rows.sum());right=float(np.sum(g.w*g.Y)/g.w.sum())
            correct.append(dict(head=h,bin=bn,receipt_y=wrong,pooled_weighted_y=right,diff=wrong-right))
    return dict(worlds=dict(worlds),episode_policies=len(e),counters=dict(ctr),diffs=[v for v in js['diffs'] if v['versus']=='PI0'],
                rich_churn=dict(reviewed=int(len(ch[ch.policy=='RICH'])),pulled=int(((ch.policy=='RICH')&(ch.action=='PULL')).sum())),calibration_pooling=correct)
if __name__=='__main__':
    mode=sys.argv[1];result=globals()[mode]();(OUT/(mode+'.json')).write_text(json.dumps(result,indent=2,sort_keys=True,default=str)+'\n');print(json.dumps(result,sort_keys=True,default=str))
