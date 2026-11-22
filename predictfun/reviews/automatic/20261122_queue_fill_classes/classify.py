"""Independent saved-data classification; see frozen declaration.md. No replay.
prepare hashes inputs without parsing fills. run refuses uncommitted instruments.
"""
from pathlib import Path
import collections, gzip, hashlib, io, json, math, subprocess, sys
import numpy as np
import pandas as pd

ROOT=Path('/home/yuqing/ctaNew'); HERE=Path(__file__).resolve().parent
SRC='85127b3dfbdfb8689159fe8b9e76b447324b7cfc'
REL='predictfun/reviews/ladder_async_fast_20261122'
DATA=ROOT/'data/predict_fun/research/ladder_async_fast_v1_20261122'
ARMS=('CURRENT_ASYNC_FAST10','FRONT_ASYNC_FAST10'); TOL=1e-6
def sha(b): return hashlib.sha256(b).hexdigest()
def gitblob(ref,path): return subprocess.check_output(['git','-C',str(ROOT),'show',ref+':'+path])
def prepare():
    p=HERE/'input_manifest.json'; assert not p.exists()
    mb=gitblob(SRC,REL+'/manifest.json'); gm=json.loads(mb)
    files={}
    for d in gm['days']:
        for part in ('pilot','rest'):
            if not gm['parts'][part][d]:continue
            for suf in ('.json','.jsonl.gz','_arrays.npz'):
                path=DATA/part/(d+suf);files[str(path.relative_to(ROOT))]=sha(path.read_bytes())
    p.write_text(json.dumps(dict(source=SRC,source_manifest_sha=sha(mb),files=files,
        reference_arms_sha=sha(gitblob(SRC,REL+'/outputs/arms.csv.gz'))),indent=2)+'\n')
    print('Prepared input byte hashes only; no fill/outcome parsed.')

def category(g,l,n,missing,has_action=True):
    if not has_action:return 'NO_OPPORTUNITY'
    if missing:return 'INCOMPLETE'
    if not n:return 'NO_SUFFIX_FILLS'
    if l>TOL and g<=TOL:return 'PURE_BAD'
    if g>TOL and l<=TOL:return 'GOOD_ONLY'
    if g>TOL and l>TOL:
        return 'MIXED_COMPENSATED' if g-l>TOL else 'MIXED_UNCOMPENSATED' if l-g>TOL else 'MIXED_TIE'
    return 'NEAR_ZERO'

def sequence(suffix,keeps):
    if not suffix:return 'NO_SUFFIX',False
    if any(v is None for _,_,v in suffix):return 'INCOMPLETE',False
    pos=[t for t,_,v in suffix if v>0];neg=[t for t,_,v in suffix if v<0]
    if not pos:return ('BAD_ONLY' if neg else 'ZERO_ONLY'),False
    if not neg:return 'GOOD_ONLY',False
    if max(neg)<min(pos):return 'BAD_THEN_GOOD',False
    if max(pos)<min(neg):return 'GOOD_THEN_BAD',any(max(pos)<t+10<=min(neg) for t in keeps)
    return 'INTERLEAVED_OR_SIMULTANEOUS',False

def sums(fl):
    vs=[v for _,_,v in fl if v is not None]
    return sum(v for v in vs if v>0),-sum(v for v in vs if v<0),sum(v is None for _,_,v in fl)

def selftest():
    cases=[(0,0,0,0,False,'NO_OPPORTUNITY'),(5,0,2,1,True,'INCOMPLETE'),(0,0,0,0,True,'NO_SUFFIX_FILLS'),
      (0,3,1,0,True,'PURE_BAD'),(3,0,1,0,True,'GOOD_ONLY'),(8,2,2,0,True,'MIXED_COMPENSATED'),
      (2,8,2,0,True,'MIXED_UNCOMPENSATED'),(3,3,2,0,True,'MIXED_TIE'),(0,0,1,0,True,'NEAR_ZERO')]
    for g,l,n,mis,act,want in cases: assert category(g,l,n,mis,act)==want
    assert category(0,3,2,1)=='INCOMPLETE' # missing later good cannot become PURE_BAD
    assert sequence([(20,1,-2),(100,1,8)],[0,50])[0]=='BAD_THEN_GOOD'
    assert sequence([(20,1,8),(100,1,-2)],[0,50])==('GOOD_THEN_BAD',True)
    assert sequence([(20,1,8),(30,1,-2)],[0,50])==('GOOD_THEN_BAD',False)
    assert sequence([(20,1,8),(20,1,-2)],[0])[0]=='INTERLEAVED_OR_SIMULTANEOUS'
    fl=[(50,1,1),(59,1,-2),(60,1,-3),(70,1,4)]
    assert sums([x for x in fl if x[0]>=60])[:2]==(4,3)
    print('Synthetic classes / timing / missing marks / sequences: PASS')

def run():
    for name in ('classify.py','declaration.md','input_manifest.json'):
        p=HERE/name;assert gitblob('HEAD',str(p.relative_to(ROOT)))==p.read_bytes(),name
    out=HERE/'outputs';assert not out.exists();selftest()
    man=json.loads((HERE/'input_manifest.json').read_text())
    for rel,h in man['files'].items():assert sha((ROOT/rel).read_bytes())==h,rel
    mb=gitblob(SRC,REL+'/manifest.json');assert sha(mb)==man['source_manifest_sha'];gm=json.loads(mb)
    rb=gitblob(SRC,REL+'/outputs/arms.csv.gz');assert sha(rb)==man['reference_arms_sha']
    ref=pd.read_csv(io.BytesIO(gzip.decompress(rb)),dtype={'day':str},float_precision='round_trip')
    rows=[];seen=set();recon=[]
    for d in gm['days']:
      for part in ('pilot','rest'):
        if not gm['parts'][part][d]:continue
        rr=json.loads((DATA/part/f'{d}.json').read_text());gz=(DATA/part/f'{d}.jsonl.gz').read_bytes()
        assert rr['sources']==gm['sources'] and rr['manifest_sha256']==sha(mb)
        assert rr['sha256']['gz']==sha(gz) and rr['sha256']['npz']==sha((DATA/part/f'{d}_arrays.npz').read_bytes())
        assert rr['status']=='DAY_DONE' and rr['members']==gm['parts'][part][d]
        worlds=[json.loads(x) for x in gzip.decompress(gz).splitlines() if x]
        assert [int(w['m']) for w in worlds]==rr['members']
        with np.load(DATA/part/f'{d}_arrays.npz') as zz:
          for w in worlds:
            m=int(w['m']);assert (d,m) not in seen;seen.add((d,m))
            end=float(w['path_info']['end_rel'])
            for arm in ARMS:
              rec=w[arm];assert rec['settings']['cancel_effect_ms']==10
              cm=zz[f'{m}__{arm}__commands'];kt=collections.defaultdict(list)
              for t,s,kind,gid,rung,price,q,act,reason in cm:
                if kind==0 and act==2:kt[int(gid)].append(float(t))
              by=collections.defaultdict(list);vend=collections.Counter();fcounts=collections.Counter()
              assert len(rec['fills'])==len(rec['fill_mids'])==len(rec['fill_class'])
              for f,mm in zip(rec['fills'],rec['fill_mids']):
                t,gid,s,q,p,it=f;sg=1 if s==0 else -1
                v=sg*q*(mm[6]-p) if t+1000<=end and mm[6] is not None and math.isfinite(mm[6]) else None
                by[int(gid)].append((float(t),float(q),v));vend[int(gid)]+=sg*q*(rec['mark']-p);fcounts[int(gid)]+=1
              od={int(o['gid']):o for o in rec['orders']};assert set(by)<=set(od) and set(kt)<=set(od)
              market_rows=[]
              for gid,o in od.items():
                fl=by[gid];keeps=sorted(kt[gid]);d0=keeps[0] if keeps else None;e=d0+10 if keeps else math.inf
                if keeps:assert o['act']<=d0 and (o['req'] is None or d0<=o['req'])
                sf=[x for x in fl if x[0]>=e];prefix=[x for x in fl if x[0]<e]
                g,l,mis=sums(sf);pg,pl,pm=sums(prefix);wg,wl,wm=sums(fl)
                assert abs(pg+g-wg)<1e-6 and abs(pl+l-wl)<1e-6 and pm+mis==wm
                seq,sep=sequence(sf,keeps)
                q0=rec['queue_venue']['ahead0'].get(str(gid));q1=rec['queue_venue']['first'].get(str(gid))
                closed=rec['closed'].get(str(gid));endcause=closed[0] if closed else 'open_or_unnotified_at_END'
                row=dict(day=d,m=m,arm=arm,gid=gid,side=int(o['side']),refused=bool(o['refused']),size=o['size'],
                  submit=o['submit'],activation=o['act'],anchor=d0,effect=e if keeps else None,
                  category=category(g,l,len(sf),mis,bool(keeps)),sequence=seq,separable_good_then_bad=sep,
                  good=g,loss=l,net=g-l,suffix_fills=len(sf),suffix_shares=sum(q for _,q,_ in sf),missing_suffix=mis,
                  prefix_good=pg,prefix_loss=pl,whole_good=wg,whole_loss=wl,whole_1s=wg-wl,whole_end=vend[gid],
                  whole_fills=len(fl),missing_whole=wm,known_end_cause=endcause,recorded_pull=o['req'] is not None,
                  activation_ahead=q0[1] if q0 else None,first_review_ahead=q1[1] if q1 else None,
                  first_review_time=q1[0] if q1 else None,
                  exact_zero_suffix=sum(v==0 for _,_,v in sf if v is not None))
                rows.append(row);market_rows.append(row)
              r=ref[(ref.day==d)&(ref.m==m)&(ref.arm==arm)];assert len(r)==1
              totals=[sum(x['whole_1s'] for x in market_rows),sum(x['whole_end'] for x in market_rows)]
              assert np.allclose(totals,[r.iloc[0].all_id_mo1s,r.iloc[0].value],rtol=1e-9,atol=1e-6)
              assert sum(x['whole_fills'] for x in market_rows)==len(rec['fills'])
              recon.append(dict(day=d,m=m,arm=arm,generations=len(od),fills=len(rec['fills']),value_1s=totals[0],end=totals[1]))
    assert seen=={(d,int(m)) for d in gm['days'] for m in gm['complete'][d]}
    R=pd.DataFrame(rows);assert not R.duplicated(['day','m','arm','gid']).any()
    # Verify classification independently as a partition, including known-bad refusal.
    def independently_ok(row):
      a=row['anchor'];g=row['good'];l=row['loss'];n=row['suffix_fills'];mis=row['missing_suffix'];c=row['category']
      if a is None or pd.isna(a):return c=='NO_OPPORTUNITY'
      if mis:return c=='INCOMPLETE'
      if n==0:return c=='NO_SUFFIX_FILLS'
      allowed={'PURE_BAD':g<=TOL<l,'GOOD_ONLY':l<=TOL<g,'NEAR_ZERO':max(g,l)<=TOL,
        'MIXED_COMPENSATED':min(g,l)>TOL and g-l>TOL,
        'MIXED_UNCOMPENSATED':min(g,l)>TOL and l-g>TOL,
        'MIXED_TIE':min(g,l)>TOL and abs(g-l)<=TOL}
      return allowed.get(c,False)
    assert all(independently_ok(x) for x in rows)
    assert not independently_ok(dict(anchor=50,good=10,loss=2,suffix_fills=2,missing_suffix=0,category='PURE_BAD'))
    assert not independently_ok(dict(anchor=50,good=0,loss=2,suffix_fills=2,missing_suffix=1,category='PURE_BAD'))
    summaries=[]
    for keys in (['arm'],['arm','day'],['arm','side']):
      ss=R.groupby(keys+['category'],dropna=False).agg(generations=('gid','size'),suffix_good=('good','sum'),suffix_loss=('loss','sum'),
          suffix_net=('net','sum'),suffix_shares=('suffix_shares','sum'),prefix_loss=('prefix_loss','sum'),whole_1s=('whole_1s','sum')).reset_index()
      ss['grouping']='|'.join(keys);summaries.append(ss)
    S=pd.concat(summaries,ignore_index=True)
    X=R.groupby(['arm','category','sequence']).agg(generations=('gid','size'),separable=('separable_good_then_bad','sum'),good=('good','sum'),loss=('loss','sum')).reset_index()
    assert S[S.grouping=='arm'].generations.sum()==len(R)
    rank=lambda x:sha(f'{x["arm"]}|{x["day"]}|{x["m"]}|{x["gid"]}'.encode())
    ex=[]
    for _,grp in R.groupby(['arm','category']):ex+=sorted(grp.to_dict('records'),key=rank)[:2]
    out.mkdir();R.to_csv(out/'generations.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    S.to_csv(out/'classes.csv',index=False);X.to_csv(out/'sequences.csv',index=False)
    pd.DataFrame(recon).to_csv(out/'reconciliation.csv',index=False);pd.DataFrame(ex).to_csv(out/'examples.csv',index=False)
    chk=dict(synthetic=True,input_hashes=True,all_market_membership=True,one_row_per_generation=True,first_keep_effect=True,
      per_market_value_reproduction=True,fill_count_reconciled=True,prefix_suffix_reconciled=True,independent_classes=True,
      corrupt_category_refused=True,missing_mark_pure_bad_refused=True,class_count_reconciled=True)
    (out/'checks.json').write_text(json.dumps(chk,indent=2)+'\n')
    result=dict(source=SRC,freeze=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD']).decode().strip(),
      markets=len(seen),generations=len(R),anchor='first_recorded_live_KEEP_plus_10ms',horizon='fill_plus_1s',
      classes=S[S.grouping=='arm'].fillna('all').to_dict('records'),checks=chk,
      limits=['Consumed days, descriptive hindsight; no predictive or queue-necessity claim.',
       'Recorded suffix ends at the original policy termination; future retained/reposted fills not observed.',
       'First-anchor economics, not the best future cancel; prefix losses are separate.'])
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(S[S.grouping=='arm'][['arm','category','generations','suffix_good','suffix_loss','suffix_net']].to_string(index=False))
    print('ALL CHECKS PASS; rows',len(R),'markets',len(seen))

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='run':run()
    else:raise SystemExit('prepare | run')
