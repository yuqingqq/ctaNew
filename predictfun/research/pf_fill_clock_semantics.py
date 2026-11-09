"""Post-hoc timestamp diagnosis of the frozen Nov8 exact-identity clock audit.

No fitted clocks, strategy replay, or new dates. Preserve status-specific timestamps
and report whether their seconds overlap necessary lifecycle bounds.
"""
import collections
import datetime as dt
import json
import math
import subprocess
import sys
import numpy as np
import pandas as pd
from . import pf_fill_clock as c

OUT = c.DEST
FILES = ['report.json', 'clock_cases.parquet', 'fill_events.parquet',
         'engine_anchors.parquet', 'own_rest_legs.parquet']
SOURCE = ['pf_fill_clock_semantics.py', 'tests/test_fill_clock_semantics.py']


def overlap(stamp, lo, hi, rounded=False):
    if not all(math.isfinite(x) for x in [stamp, lo, hi]) or lo > hi:
        raise ValueError('invalid timestamp interval')
    left = stamp - (500 if rounded else 0)
    return left <= hi and left + 1000 > lo


def status_stamps(events, order_hash, trade_id):
    if not len(events) or set(events.order_hash) != {order_hash} or set(events.trade_id) != {trade_id}:
        raise ValueError('wrong trade identity')
    if events.q.nunique() != 1 or events.price.nunique() != 1:
        raise ValueError('changed fill quantity/price')
    result = {}
    for status in ['Matched', 'Confirmed', 'Failed']:
        g = events[events.status == status].sort_values('at', kind='stable')
        result[status] = dict(n=len(g), first_receipt=float(g['at'].iloc[0]) if len(g) else None,
                             stamps=sorted(map(float, g.reported_match_ms.unique())))
    return result


def constant_interval(q):
    if not len(q) or not np.isfinite(q[['Z', 'lower', 'upper']].to_numpy()).all():
        raise ValueError('missing clock bounds')
    lo = float((q.Z - q.upper).max()); hi = float((q.Z - q.lower).min())
    return dict(n_legs=len(q), n_orders=int(q.order_hash.nunique()), lower_ms=lo, upper_ms=hi,
                exists=lo <= hi, exists_at10ms=lo - 10 <= hi + 10)


def declare():
    c.new_paths('semantics_declaration.json')
    d = dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),
        scope='post-hoc diagnosis after original audit and its two extreme witnesses; same consumed Nov8 cohort',
        known_before_declaration='one trade changes reported match_time by 10s between Matched and Confirmed; other witness Z precedes placement by975ms',
        population='all99 exact confirmed REST/kernel legs, including6 ambiguous engine anchors; clock comparisons on original93 eligible legs',
        design=['compare exact REST executedAt with all per-status reported timestamps, retain unavailable/nonunique statuses',
                'test floor-second and rounded-second overlap, not equality with actual matching time',
                'intersect constant-shift feasibility separately per run, no offset selection',
                'trace two previously reported witnesses through exact original candidate rows and transaction groups'],
        multiplicity='descriptive forensic follow-up, no profit ranking, significance claim, calibration or new model; original500 null draws unchanged',
        data_boundary='read only original frozen audit artifacts and its declared22 Nov8 REST hours',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.HERE,text=True).strip(),
        source_sha={f:c.sha(c.HERE/f) for f in SOURCE},
        input_sha={f:c.sha(OUT/f) for f in FILES})
    c.write(OUT/'semantics_declaration.json', d)
    c.write(c.HERE.parent/'reviews/fill_clock_20261109/semantics_declaration.json',d)
    print('DECLARED post-hoc timestamp diagnosis')


def run():
    c.new_paths('semantics_report.json','status_clock_cases.parquet')
    d=json.loads((OUT/'semantics_declaration.json').read_text())
    for f,h in d['source_sha'].items():
        if c.sha(c.HERE/f)!=h:raise ValueError('supplement source changed')
    for f,h in d['input_sha'].items():
        if c.sha(OUT/f)!=h:raise ValueError('supplement input changed')
    original=json.loads((OUT/'report.json').read_text())
    for f,h in original['declaration']['source_sha'].items():
        if c.sha(c.HERE.parents[1]/f)!=h:raise ValueError('original source changed')
    for f,h in original['declaration']['input_sha'].items():
        if c.sha(f)!=h:raise ValueError('original input changed')
    for f,h in original['artifacts'].items():
        if c.sha(OUT/f)!=h:raise ValueError('original result changed')
    if c.sha(OUT/'report.json')!=c.sha(c.HERE.parent/'reviews/fill_clock_20261109/receipt.json'):
        raise ValueError('receipt copy differs')
    control=(OUT/'logs/fill_clock_semantics_controls.log').read_text()
    if '\nOK\n' not in control or 'rc=0 ' not in control:raise ValueError('missing controls')
    C=pd.read_parquet(OUT/'clock_cases.parquet');E=pd.read_parquet(OUT/'fill_events.parquet')
    groups={k:g for k,g in E.groupby(['order_hash','trade_id'],sort=False)}
    from . import pf_io as io
    scope=json.loads((c.RP/'scope.json').read_text())
    wanted=set(zip(C.tx,C.order_hash));raw=collections.defaultdict(set)
    for hour in scope['hours']:
        if not hour.startswith(c.DAY+'_'):raise ValueError('out-of-scope hour')
        for recv,params,path,payload in io.rest_rows(hour,('/v1/orders/matches',)):
            for trade in payload.get('data') or []:
                for maker in trade['makers']:
                    k=(trade['transactionHash'].lower(),maker['hash'].lower())
                    if k in wanted:raw[k].add(trade['executedAt'])
    rows=[];histories={}
    for x in C.to_dict('records'):
        k=(x['order_hash'],x['trade_id']);history=status_stamps(groups[k],*k);histories[k]=history
        rs=raw[(x['tx'],x['order_hash'])]
        if len(rs)!=1:raise ValueError('REST raw timestamp missing or conflicting')
        rawtext=next(iter(rs));rawms=io.iso_s(rawtext)*1000
        if int(rawms/1000)!=x['exec_s']:raise ValueError('raw REST second differs')
        mt=history['Matched']['stamps'];ct=history['Confirmed']['stamps']
        if not ct:raise ValueError('confirmed REST trade lacks confirmed kernel notice')
        y=dict(run=x['run'],cid=x['cid'],order_hash=x['order_hash'],trade_id=x['trade_id'],
               exec_s=x['exec_s'],rest_raw=rawtext,rest_fraction_ms=rawms-x['exec_s']*1000,
               matched_status='unique' if len(mt)==1 else 'missing' if not mt else 'conflicting',
               confirmed_status='unique' if len(ct)==1 else 'conflicting',
               rest_equals_any_confirmed_second=x['exec_s']*1000 in ct,
               rest_equals_any_matched_second=x['exec_s']*1000 in mt,
               rest_floor_overlaps=overlap(x['exec_s']*1000,x['lower'],x['upper']),
               rest_rounded_overlaps=overlap(x['exec_s']*1000,x['lower'],x['upper'],True),
               engine_status=x['status'])
        if len(mt)==1:
            y.update(matched_reported_ms=mt[0],rest_minus_matched_ms=x['exec_s']*1000-mt[0],
                     matched_floor_overlaps=overlap(mt[0],x['lower'],x['upper']),
                     matched_rounded_overlaps=overlap(mt[0],x['lower'],x['upper'],True))
        if len(mt)==1 and len(ct)==1:y['confirmed_minus_matched_ms']=ct[0]-mt[0]
        rows.append(y)
    R=pd.DataFrame(rows);R.to_parquet(OUT/'status_clock_cases.parquet',index=False)
    eligible=C[C.status=='eligible'];matched=R[R.matched_status=='unique']
    L=pd.read_parquet(c.DATA/'step1'/c.DAY/'legs_timed2.parquet')
    T=pd.read_parquet(c.DATA/'step1'/c.DAY/'legs_tk.parquet')
    A=pd.read_parquet(OUT/'engine_anchors.parquet')
    if not len(L)==len(T)==len(A):raise ValueError('engine table alignment differs')
    tkg=np.where(T.tk>=0,T.tk,-1-np.arange(len(T)))
    witnesses={}
    for name in ['lower_witness','upper_witness']:
        h=original['constant_shift_interval_ms'][name]['order_hash']
        x=eligible[eligible.order_hash==h].iloc[0];match=np.ones(len(L),dtype=bool)
        for col in c.C:match &= (L[col].to_numpy()==x[col])
        ix=np.flatnonzero(match);a,status=c.unique_anchor(ix.tolist(),A)
        if a is None or any(a[k]!=x[k] for k in ['Z','J','X']):raise ValueError('witness clock identity differs')
        evidence=[]
        for i in ix:
            same=np.flatnonzero(tkg==tkg[i]);g=L.iloc[same]
            ws=g[g.how>0];fallback=g[(g.how==0)&g.t_e2.notna()]
            source=ws if len(ws) else fallback
            field='t_ms' if len(ws) else 't_e2'
            stamp=float(source[field].min()) if len(source) else float(x.exec_s*1000)
            if stamp!=a['Z']:raise ValueError('witness transaction anchor origin differs')
            origins=source[source[field]==stamp]
            evidence.append(dict(row=int(i),original_how=int(L.iloc[i]['how']),tk_status=int(T.iloc[i].tk_status),
                transaction_group_rows=len(g),anchor_origin='pending_increment' if len(ws) else 'book_drop' if len(fallback) else 'REST_second',
                origin_rows=[int(j) for j in origins.index],anchor_ms=stamp,
                origin_leg_how=[int(z) for z in origins.how],
                original_leg_ms=None if pd.isna(L.iloc[i].t_ms) else float(L.iloc[i].t_ms)))
        witnesses[name]=dict(order_hash=h,trade_id=x.trade_id,tx=x.tx,run=x.run,cid=int(x.cid),
            place_ms=float(x.lower),last_possible_ms=float(x.upper),Z_ms=float(x.Z),
            rest_raw=next(iter(raw[(x.tx,h)])),notice_history=histories[(h,x.trade_id)],engine_origin=evidence)
    hist={str(int(k)):int(v) for k,v in matched.rest_minus_matched_ms.value_counts().sort_index().items()}
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),declaration=d,
        declaration_sha=c.sha(OUT/'semantics_declaration.json'),original_report_sha=c.sha(OUT/'report.json'),
        controls=control,population=dict(confirmed_legs=len(R),confirmed_orders=int(R.order_hash.nunique()),
            matched_status=R.matched_status.value_counts().to_dict(),confirmed_status=R.confirmed_status.value_counts().to_dict(),
            engine_status=R.engine_status.value_counts().to_dict()),
        timestamp_checks=dict(rest_matches_confirmed=int(R.rest_equals_any_confirmed_second.sum()),
            rest_matches_matched=int(R.rest_equals_any_matched_second.sum()),
            raw_rest_with_fraction_ms=int((R.rest_fraction_ms!=0).sum()),rest_minus_matched_ms_histogram=hist,
            rest_floor_nonoverlap=int((~R.rest_floor_overlaps).sum()),rest_rounded_nonoverlap=int((~R.rest_rounded_overlaps).sum()),
            unique_matched_legs=len(matched),matched_floor_nonoverlap=int((matched.matched_floor_overlaps==False).sum()),
            matched_rounded_nonoverlap=int((matched.matched_rounded_overlaps==False).sum())),
        constant_shift_by_run={run:constant_interval(g) for run,g in eligible.groupby('run')},
        witnesses=witnesses,artifact_sha=c.sha(OUT/'status_clock_cases.parquet'),
        predicates=dict(all_rest_seconds_equal_confirmation=bool(R.rest_equals_any_confirmed_second.all()),
            any_changed_timestamp=bool((matched.rest_minus_matched_ms!=0).any()),
            constant_offset_fails_each_run=all(not constant_interval(g)['exists_at10ms'] for _,g in eligible.groupby('run'))),
        limitations=['post-hoc forensic follow-up on one consumed partial day; no interval or model validation',
            'a Matched status timestamp is not an exchange matching-engine timestamp; overlap does not prove precise time',
            'no strategy PnL rerun; clipping fitted clocks to own lifecycle would manufacture a pass',
            'public pending increments carry no signed-order identity; nearby volume/price association cannot establish exact trade timing'])
    c.write(OUT/'semantics_report.json',out)
    c.write(c.HERE.parent/'reviews/fill_clock_20261109/semantics_receipt.json',out)
    print(json.dumps({k:out[k] for k in ['population','timestamp_checks','constant_shift_by_run','predicates']},indent=2))


if __name__=='__main__':{'declare':declare,'run':run}[sys.argv[1]]()
