"""Pinned saved-evidence review. No tape, replay, fitting or development writes."""
from pathlib import Path
import collections, copy, gzip, hashlib, importlib.util, json, math, subprocess, sys
import numpy as np
import pandas as pd

ROOT = Path('/home/yuqing/ctaNew')
PIN = Path('/tmp/pf-ta-review-51ed592d')
OUT = Path(__file__).resolve().parent
REC = 'predictfun/reviews/ladder_turnover_attribution_20261122'
LCREC = 'predictfun/reviews/ladder_long_continuation_20261122'
TARGET = '51ed592d9ddf9e2968d12ffb9fc966763c0a8ce9'
sys.path.insert(0, str(PIN/'predictfun/research'))
import pf_ladder_turnover_attribution as X
import pf_ladder_turnover_attribution_selftest as ST
import pf_ladder_turnover_attribution_mutants as MH
import pf_mutant_summary as MS
import pf_mutant_one as MO
spec = importlib.util.spec_from_file_location('ta_check_pinned', PIN/REC/'ta_checks.py')
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
C.REPO = PIN; C.R = PIN/REC; C.LCR = PIN/LCREC
X.LC_OUT = ROOT/'data/predict_fun/research/ladder_long_continuation_v1_20261122'
sha = lambda b: hashlib.sha256(b).hexdigest()
git = lambda *a: subprocess.check_output(['git','-C',str(PIN),*a])
assert git('rev-parse','HEAD').decode().strip() == TARGET
for rel in C.COMMITTED:
    assert git('show','db044afb:'+rel) == (PIN/rel).read_bytes(), rel
ST.main(X)
logs = PIN/REC/'logs/mutants_try2'
recs = {'control': MS.last_record(logs/'lta2_control.log')}
recs.update({i: MS.last_record(logs/f'lta2_{i}.log') for i in range(len(MH.MUTANTS))})
assert MS.judge(len(MH.MUTANTS),recs,MO.sources(MH),[m[0] for m in MH.MUTANTS])[0]
E=C.load()
assert sha((C.LC/'measure/audit.json').read_bytes()) == E['gm']['lc_audit_sha256']
for rel in E['gm']['lc_records']:
    part, day = rel.split('/')
    assert (C.LC/part/(day+'.json')).read_bytes() == (PIN/LCREC/'outputs'/(part+'_'+day+'.json')).read_bytes()
for n in C.OUTS:
    assert gzip.decompress((PIN/REC/'outputs'/f'{n}.csv.gz').read_bytes()) == (C.D/f'{n}.csv').read_bytes()
assert (PIN/REC/'outputs/audit.json').read_bytes() == (C.D/'audit.json').read_bytes()
checks=C.run_checks(E)
assert checks['all_ok'], checks
# The resource falsifier creates a temporary directory; keep it review-owned.
oldcase=C.CASES[-1]
def fake_in_review(e):
    savedD=C.D; C.D=OUT
    try: oldcase[3](e)
    finally: C.D=savedD
C.CASES[-1]=(*oldcase[:3],fake_in_review)
falsifier=C.falsify(E,checks)
assert falsifier['falsifier_ok'], falsifier

def close(a,b):
    if isinstance(a,(str,bool)) or isinstance(b,(str,bool)): return a==b
    return (pd.isna(a) and pd.isna(b)) or abs(float(a)-float(b)) <= 1e-7*max(1,abs(float(a)),abs(float(b)))

def my_fifo(a):
    """Independent interval matching: mutable deque with actual acquisition metadata."""
    book=collections.deque(); closes=[]
    for phase in ('pre_fills','post_raw'):
        if phase=='post_raw':
            for lot in book: lot['basis']=float(a['M_fork']); lot['carried']=True
        for t,g,s,q,p,idx in sorted(a[phase],key=lambda f:(f[0],f[5])):
            sign=1 if int(s)==0 else -1; left=float(q)
            while book and book[0]['sign'] != sign and left>1e-12:
                lot=book[0]; amount=min(left,lot['left'])
                if phase=='post_raw':
                    closes.append(dict(q=amount,sign=lot['sign'],basis=lot['basis'],open_price=lot['price'],open_t=lot['t'],open_gid=lot['gid'],carried=lot['carried'],close_price=p,close_t=t,close_gid=g,profit=amount*lot['sign']*(p-lot['basis'])))
                lot['left']-=amount; left-=amount
                if lot['left']<=1e-12: book.popleft()
            if left>1e-12: book.append(dict(sign=sign,left=left,basis=p,price=p,t=t,gid=g,carried=False))
    lin=set(a['lineage']); mq=sum(c['q'] for c in closes)
    profit=sum(c['profit'] for c in closes)
    val=lambda lot: lot['left']*lot['sign']*(a['mark']-lot['basis'])
    inc=sum(c['profit'] for c in closes if c['open_gid'] in lin or c['close_gid'] in lin)
    d=dict(matched=profit,residual=sum(map(val,book)),matched_q=mq,profit_per_share=profit/mq if mq else np.nan,
           hold_ms=sum(c['q']*(c['close_t']-c['open_t']) for c in closes)/mq if mq else np.nan,
           matched_carried=sum(c['profit'] for c in closes if c['carried']),matched_new=sum(c['profit'] for c in closes if not c['carried']),
           matched_q_carried=sum(c['q'] for c in closes if c['carried']),carried_q=sum(c['q'] for c in closes if c['carried'])+sum(l['left'] for l in book if l['carried']),
           residual_carried=sum(val(l) for l in book if l['carried']),residual_new=sum(val(l) for l in book if not l['carried']),
           remaining_q=sum(l['left']*l['sign'] for l in book),remaining_q_carried=sum(l['left']*l['sign'] for l in book if l['carried']),
           matched_lineage=inc,matched_other=profit-inc)
    return d,closes

arms=E['arms'].set_index(['key','arm']); pairs=E['pairs'].set_index('key')
lp=E['lp'].set_index('key'); ld=E['delays'].set_index(['key','arm'])
assert len(set(x['key'] for x in E['led']))==64
assert set(pairs.index)==set(lp.index)==set(x['key'] for x in E['led'])
alllots=[]; derived=[]; maxgap=0; newprice_stats=[]
for x in E['led']:
    assert x['retain']['pre_fills']==x['alternative']['pre_fills']
    assert x['retain']['M_fork']==x['alternative']['M_fork'] and x['retain']['mark']==x['alternative']['mark']
    vals={}
    for name in ('retain','alternative'):
        a=x[name]; d,cl=my_fifo(a); vals[name]=d; r=arms.loc[(x['key'],name)]
        assert all(close(r[k],v) for k,v in d.items()),(x['key'],name)
        assert close(d['matched']+d['residual'],a['Q'])
        maxgap=max(maxgap,abs(d['matched']+d['residual']-a['Q']))
        for c in cl: alllots.append(dict(key=x['key'],arm=name,**c))
        rows=E['lr'][(E['lr'].key==x['key']) & (E['lr'].arm==name)].to_dict('records')
        att,acc=C.my_delay(rows,x['kind'],name,float(lp.loc[x['key'],'t']))
        for field,val in [('first_attempt_delay',att),('first_accepted_delay',acc)]:
            assert close(r[field],val) and close(ld.loc[(x['key'],name),field],val)
        newprice_stats.append(dict(kind=x['kind'],arm=name,matched=d['matched'],q=d['matched_q'],carried=d['matched_carried'],qc=d['matched_q_carried'],new=d['matched_new'],qnew=d['matched_q']-d['matched_q_carried']))
    ev=X.event_diff(x['retain'],x['alternative']); p=pairs.loc[x['key']]
    for k,v in ev.items(): assert close(p[k],v),(k,p[k],v)
    for field,k in [('dMatched_lineage','matched_lineage'),('dMatched_other','matched_other')]: assert close(p[field],vals['retain'][k]-vals['alternative'][k])
    derived.append(dict(key=x['key'],kind=x['kind'],day=x['day'],side=str(lp.loc[x['key'],'side']),Y=float(p.Y),**{k:float(p[k]) for k in ['dMatched','dResidual','dMatched_lineage','dMatched_other','Y_events']}))
# Exact lot membership and all emitted fields, including carried basis and real ages.
mine=pd.DataFrame(alllots); saved=E['lots']; assert list(mine.columns)==list(saved.columns) and len(mine)==len(saved)
for col in mine:
    if mine[col].dtype.kind in 'bOUS': assert mine[col].tolist()==saved[col].tolist(),col
    else: assert np.allclose(mine[col],saved[col],rtol=1e-10,atol=1e-7,equal_nan=True),col
# Every summary column and every expected group, not merely the checked subset.
S=X.levels(pd.DataFrame(derived),X.srow).set_index(['kind','day','side']).sort_index()
T=E['summary'].set_index(['kind','day','side']).sort_index(); assert S.index.equals(T.index)
for c in S: assert np.allclose(S[c],T[c],rtol=1e-9,atol=1e-7,equal_nan=True),c
assert X.day_signs(pd.DataFrame(derived))==E['audit']['predicates']['day_signs']

# Demonstrate ancillary checker limitations without changing any saved artifact.
escapes={}
for table,col,fam in [('arms','hold_ms','D'),('arms','matched_lineage','D'),('lots','basis','D'),('summary','top5_share_abs_Y','S')]:
    e=dict(E); e[table]=E[table].copy(); e[table].loc[e[table].index[0],col]+=10000
    escapes[table+'.'+col]=all(C.FAMS[fam](e)[0].values())

agg=[]
for (kind,arm), g in pd.DataFrame(newprice_stats).groupby(['kind','arm']):
    agg.append(dict(kind=kind,arm=arm,matched_c=float(g.matched.sum()),matched_shares=float(g.q.sum()),
                    ratio_of_sums=float(g.matched.sum()/g.q.sum()),mean_arm_ratio=float((g.matched/g.q).mean()),
                    carried_c=float(g.carried.sum()),carried_shares=float(g.qc.sum()),
                    new_only_ratio=float(g.new.sum()/g.qnew.sum())))
result=dict(target=TARGET,checks_all_pass=checks['all_ok'],falsifiers=10,final_source_mutants=22,
            pairs=64,arms=128,lots=len(mine),summary_rows=len(S),max_Q_identity_gap_c=maxgap,
            ancillary_falsifiers_escape=escapes,profit_denominators=agg,
            per_day=E['pairs'].groupby(['kind','day'])[['Y','dMatched','dResidual']].sum().reset_index().to_dict('records'),
            audit_predicates=E['audit']['predicates'],limitations='Saved inputs only; no economic test, raw tape reload, fit or replay.')
for n,obj in [('checks.json',checks),('falsifier.json',falsifier),('evidence.json',result)]:
    (OUT/n).write_text(json.dumps(obj,indent=2,default=str)+'\n')
print(json.dumps(result,indent=2,default=str))
