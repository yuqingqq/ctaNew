"""Frozen coverage study and versioned labels for timing certificates.

Consumed days only. No fitting or value-based selection. Compare original
whole-window exclusions with certified side-prefix exclusions on exactly the
same neutral probes. The newer future-only rule is a coverage comparator, not
a correctness control. E2 +/-250 ms remains an assumption in both variants.
"""
import argparse
import ast
import copy
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess

import numpy as np
import pandas as pd

import pf_stop_ev as E
import pf_stop_ev_accounting as A
import pf_stop_ev_timing_certificate as C

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
DAYS=('20261023','20261024','20261025','20261026','20261027','20261031','20261101','20261102','20261103')
MODES=('certified_ws','certified_estimated')
LAB=tuple(E.A_LAB)


def now(): return datetime.now(timezone.utc).isoformat()


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def sources():
    todo=['pf_stop_ev_timing_build','pf_stop_ev_timing_certificate','pf_stop_ev_timing_check']; out={}
    while todo:
        n=todo.pop(); p=HERE/(n+'.py')
        if str(p.relative_to(ROOT)) in out or not p.is_file():continue
        out[str(p.relative_to(ROOT))]=sha(p)
        for x in ast.walk(ast.parse(p.read_text())):
            names=[a.name.split('.')[0] for a in x.names] if isinstance(x,ast.Import) else [x.module.split('.')[0]] if isinstance(x,ast.ImportFrom) and x.module else []
            todo.extend(n for n in names if (HERE/(n+'.py')).is_file())
    out['predictfun/research/run_stop_ev_timing.sh']=sha(HERE/'run_stop_ev_timing.sh')
    out['predictfun/research/run_light.sh']=sha(HERE/'run_light.sh')
    return out


def freeze(path):
    src=sources(); commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for p,h in src.items():
        b=subprocess.check_output(['git','show',commit+':'+p],cwd=ROOT)
        if hashlib.sha256(b).hexdigest()!=h:raise ValueError('uncommitted source '+p)
    obj=dict(asof=now(),commit=commit,sources=src,days=DAYS,modes=MODES,contract=C.CONTRACT,
             size=30.,shift_ms=50.,nr=6,cap=None,queue='M',unknown_queue='zero',ages_ms=E.AGES,
             book_gap_ms=500.,e2_uncertainty_ms=250.,unknown_interval='unbounded',
             learned_candidates=0,selection='none; all declared days, no value-based choices',
             readings=['same-action coverage/statuses by day and side','baseline overlap value/feature parity',
                       'future-only admission without prefix certificate is unproved, not an observed error rate'],
             null='No model, significance or profitability test; coverage instrument only.')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(obj,f,indent=2);f.write('\n')


def manifest(day):
    if day not in DAYS:raise ValueError('not a consumed day')
    if E.K.PREROLL_MS!=30000. or not E.K.SKIP_FOREIGN_PARTS:raise ValueError('loader boundary/preroll contract changed')
    p=Path(E.K.IO)/'step1'/day; q=Path(E.K.IO)/'step5'/day
    files=[p/n for n in ('registry.parquet','clock_rl_d2.json','legs_timed2.parquet','legs_tk.parquet')]
    files += [q/n for n in ('preds_t60k.parquet','rows_rl_d2_pk.parquet')]
    files += [f for f in sorted((p/'books').glob('part_*.parquet')) if E.K.part_allowed(str(f),day)]
    return {str(p):sha(p) for p in files}


def guard():
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=2048*1024:raise MemoryError('2 GiB RSS guard')


def build(day,out,freeze_path):
    if day not in DAYS:raise ValueError('not a consumed day')
    f=json.loads(freeze_path.read_text()); src=sources()
    if src!=f['sources']:raise ValueError('changed frozen sources')
    dest=out/day;dest.mkdir(parents=True,exist_ok=False); inputs=manifest(day)
    D=E.K.load_day(day); RW,_=E.raw_windows(day,D); raw=pd.read_parquet(Path(E.K.IO)/'step1'/day/'legs_timed2.parquet',columns=['side','ptick','shares'])
    tk=pd.read_parquet(Path(E.K.IO)/'step1'/day/'legs_tk.parquet',columns=['tk']).tk.values
    LT=D['LT']; ix=LT.index.values; groups=np.where(tk[ix]>=0,tk[ix],-1-ix)
    ws_src=pd.Series(E.TS.ws_groups(LT.how.values,groups),index=LT.index)
    counts=Counter(); rows={m:[] for m in MODES}; start=now(); parity=0
    with gzip.open(dest/'window_proofs.jsonl.gz','wt') as proofs:
        for W in E.P1.window_pairs(D):
            if W[0]['ws']<E.M.boundary_cut(day):counts['boundary_windows']+=1;continue
            counts['windows']+=1; market=int(W[0]['m']); ws=float(W[0]['ws']); info={}
            for side in (0,1):
                w=W[side]; loc=D['LI'].get((market,side),[]); l=LT.iloc[loc]
                order=np.lexsort(((-l.ptick.values if side==0 else l.ptick.values),l.anc.values)); ids=l.index.values[order]
                if not (np.array_equal(l.loc[ids,'anc'].values,w['anc']) and np.array_equal(l.loc[ids,'ptick'].values,w['ptl'])
                        and np.array_equal(l.loc[ids,'shares'].values,w['shl'])):raise ValueError('raw leg identity/alignment')
                for i,rid in enumerate(ids):
                    exact=bool(ws_src.loc[rid]);info[side,i]=A.LegInfo(str(int(rid)),'ws' if exact else 'e2',0. if exact else 250.)
            rw=RW.get(market,dict(expected=0,missing=[],unt={0:[],1:[]})); tf=np.asarray(W[0]['tf'],float)
            gaps=np.flatnonzero(np.diff(tf)>500.)
            cov=A.Coverage(ws-2050.,ws+60950.,float(tf[-1]),int(rw['expected']),float(tf[0]),float(tf[-1]),
                tuple(A.MissingLeg(*g) for g in rw['missing']),
                tuple(A.MissingLeg(f'book:{i}',float(tf[i]),float(tf[i+1]),'feed_gap') for i in gaps))
            missing=[C.MissingInfo(str(g[0]),int(raw.iloc[int(g[0])].side),float(raw.iloc[int(g[0])].ptick),float(raw.iloc[int(g[0])].shares)) for g in rw['missing']]
            result=A.replay_neutral(W,size=30.,shift_ms=50.,nr=6,cap=None,mode='none')
            led=A.Ledger(W,result,submitted_quantity=30.,shift_ms=50.,leg_info=info,coverage=cov); cert=C.Certificate(led,missing)
            events=[{k:(v if not isinstance(v,float) or math.isfinite(v) else None) for k,v in e.items()} for e in cert.events]
            proofs.write(json.dumps(dict(market=market,raw_legs=cov.expected_legs,timed_legs=len(led.legs),missing=[g.__dict__ for g in missing],events=events))+'\n')
            for key,t in A.generation_probes(led,day=day,ages_ms=E.AGES):
                side,rung,generation=key
                for mode in MODES:
                    estimated=mode=='certified_estimated'; old=A.label_action(led,key,t,day=day,settings=A.Settings(timing='all_anchors_sensitivity' if estimated else 'ws_strict'))
                    row=cert.label(key,t,day=day,estimated=estimated); good=row['status'].startswith('eligible')
                    old_good=old['status'].startswith('eligible'); counts[mode+':'+row['status']]+=1;counts['baseline_'+mode+':'+old['status']]+=1
                    if old_good:
                        if not good or row['label']!=old['label']:raise ValueError('baseline overlap changed')
                        np.testing.assert_array_equal([row['features'][c] for c in A.ORDER_FEATURES], [old['features'][c] for c in A.ORDER_FEATURES])
                        parity+=1
                    future=None
                    if estimated:
                        # Exact reproduction of the future-only coverage rule.
                        # Obtain the nominal label eligibility without interpreting
                        # the proof; all other data validity checks remain intact.
                        lens=copy.copy(led);lens.coverage=replace(cov,missing=tuple(g for g in cov.missing if g.reason!='untimed'))
                        base=A.label_action(lens,key,t,day=day,settings=A.Settings(timing='all_anchors_sensitivity'))
                        future=base['status']
                        if future.startswith('eligible'):
                            price=base['features']['price'];end=min(t+1000.,led.end)
                            if any(u[0]<=end and u[1]>=t and (u[2]<=price if side==0 else u[2]>=price) for u in rw['unt'][side]):future='untimed_reach'
                        counts['future_only:'+future]+=1
                        if future.startswith('eligible') and not good:counts['future_only_eligible_without_certificate']+=1
                    y=row['label'] or {};x=row['features'] or {};proof=row['timing_certificate']
                    item=dict(day=day,m=market,side=side,rung=rung,generation=generation,decision_ms=t,
                        status=row['status'],baseline_status=old['status'],future_only_status=future,
                        unresolved_events=len(proof['unresolved_ids']),prior_state_events=len(proof['prior_state_ids']),
                        source_raw_legs=cov.expected_legs,source_missing_legs=len(cov.missing))
                    item.update({c:y.get(c,math.nan) for c in LAB});item.update({'x_'+c:x.get(c,math.nan) for c in A.ORDER_FEATURES})
                    rows[mode].append(item)
            if counts['windows']%40==0:guard();print(day,counts['windows'],'windows',flush=True)
    guard(); artifacts={}
    for mode,rr in rows.items():
        df=pd.DataFrame(rr)
        if df.duplicated(['day','m','side','rung','generation']).any():raise ValueError('duplicate action')
        path=dest/(mode+'.parquet');df.to_parquet(path,index=False);artifacts[path.name]=sha(path)
    if sources()!=src or manifest(day)!=inputs:raise ValueError('input/source changed during build')
    rec=dict(asof=now(),started=start,day=day,freeze_sha256=sha(freeze_path),contract=C.CONTRACT,
        counts=dict(counts),overlap_parity_pairs=parity,artifacts=artifacts,
        proofs_sha256=sha(dest/'window_proofs.jsonl.gz'),source_hashes=src,input_hashes=inputs,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
        interpretation='Coverage only; noncertification is not proof a label is wrong. E2 intervals and replay assumptions remain qualified.')
    with (dest/'receipt.json').open('x') as stream:json.dump(rec,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(day=day,asof=rec['asof'],counts=counts,overlap_parity_pairs=parity)),flush=True)


def report(out,freeze_path):
    rs=[json.loads((out/d/'receipt.json').read_text()) for d in DAYS]
    if any(r['freeze_sha256']!=sha(freeze_path) for r in rs):raise ValueError('mixed study freezes')
    total=Counter();dev=Counter()
    for r in rs:
        total.update(r['counts'])
        if r['day'] in DAYS[:4]:dev.update(r['counts'])
    result=dict(asof=now(),days=list(DAYS),totals=dict(total),dev=dict(dev),
        overlap_parity_pairs=sum(r['overlap_parity_pairs'] for r in rs),freeze_sha256=sha(freeze_path),
        days_receipt_sha256={d:sha(out/d/'receipt.json') for d in DAYS},
        qualification='Coverage and parity only, consumed data; neither timing truth nor performance validated.')
    with (out/'report.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','day','report']);p.add_argument('--day');p.add_argument('--out',type=Path,required=True);p.add_argument('--freeze',type=Path,required=True)
    a=p.parse_args()
    if a.command=='freeze':freeze(a.freeze)
    elif a.command=='day':build(a.day,a.out,a.freeze)
    else:report(a.out,a.freeze)
