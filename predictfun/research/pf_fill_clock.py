"""Exact-order clock conformance on the consumed Nov8 cohort; no PnL fitting."""
import collections
import datetime as dt
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
import pandas as pd
from .pf_sizing_study import ROOT,HERE,sha,write

RP=ROOT.parent/'pmmm-data/review-exports/live-samewindow-20261108'
KX=ROOT.parent/'pmmm-data/review-exports/live-samewindow-20261108-kernel'
OLD=RP/'live_vs_replay'
DEST=ROOT/'data/predict_fun/research/fill_clock_20261109'
DATA=DEST/'research'
DAY='20261108'
END_MS=1794178920000 # Nov8 23:02; complete cohort ends by23:00, before protected Nov9.
RUNS=['v6','basis']
SHIFTS=[0,50,100,200,400,800]
ENGINE_SHA='d678e7c3c4e47c6d60ebc031edf8d7fc7317210d5e58ff301d92d7ae0ece5f17'
C=['mid','exec_s','first_recv_ns','side','ptick','shares','price_executed','taker_index','taker_quote']
PREFIX=re.compile(rb'^\{"frame":\{"at":([0-9]+),')


def new_paths(*names):
    if any((DEST/n).exists() for n in names):raise ValueError('refuse overwriting audit artifacts')


def verify_clock_stats(actual,expected):
    if json.loads(json.dumps(actual,allow_nan=False))!=expected:raise ValueError('original engine clock diagnostics differ')


def amendment():
    old=json.loads((DEST/'declaration.json').read_text())
    revised=dict(old,version=2,as_of=dt.datetime.now(dt.timezone.utc).isoformat(),
        original_as_of=old['as_of'],supersedes_declaration_sha=sha(DEST/'declaration.json'),
        correction='canonicalize integer dictionary keys through JSON before exact clock diagnostics comparison; add no-overwrite guards and corresponding controls; no timing/data/selection rule change',
        failed_anchor_log_sha=sha(DEST/'logs/fill_clock_anchors.log'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        source_sha={p:sha(HERE.parents[1]/p) for p in old['source_sha']},
        inherited_extraction_sha={f:sha(DEST/f) for f in ['orders.parquet','fill_events.parquet','own_rest_legs.parquet','kernel_prefixes.json']})
    write(DEST/'declaration_v2.json',revised)
    print('DECLARED v2 instrument correction; no clock comparison result read',flush=True)


def coordinate_amendment():
    old=json.loads((DEST/'declaration_v2.json').read_text())
    revised=dict(old,version=3,as_of=dt.datetime.now(dt.timezone.utc).isoformat(),
        supersedes_declaration_sha=sha(DEST/'declaration_v2.json'),previous_correction=old['correction'],
        correction='kernel Control price is Up-space on BOTH sides; derive token price as p for BuyUp and 1-p for BuyDown/SellUp; override deprecated derived yes_tick in inherited orders without rewriting it',
        failed_identity_log_sha=sha(DEST/'logs/fill_clock_report.log'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        source_sha={p:sha(HERE.parents[1]/p) for p in old['source_sha']},
        inherited_extraction_sha={**old['inherited_extraction_sha'],'engine_anchors.parquet':sha(DEST/'engine_anchors.parquet')})
    write(DEST/'declaration_v3.json',revised)
    print('DECLARED v3 coordinate correction; raw identities and engine clocks retained',flush=True)


def order_coordinates(up,up_tick,up_index):
    if up_index not in (1,2) or not math.isfinite(up_tick) or not 0<up_tick<100:raise ValueError('invalid order coordinate')
    side=0 if bool(up)==(up_index==1) else 1
    return side,up_tick if up_index==1 else 100-up_tick,(up_tick if up else 100-up_tick)/100


def verify_leg_order(order,row,up_index):
    side,tick,token=order_coordinates(order['up'],order['price'],up_index)
    if order['market']!=row['mid'] or side!=row['side'] or abs(tick-row['ptick'])>1e-6:
        raise ValueError('exact REST order identity differs')
    return token


def hex_id(value):
    if not isinstance(value,list) or len(value)!=32 or any(type(x)!=int or not 0<=x<=255 for x in value):
        raise ValueError('invalid signed order hash')
    return '0x'+bytes(value).hex()


def bounded_steps(path,stats):
    h=hashlib.sha256();last=0;lines=0
    with Path(path).open('rb') as f:
        for line in f:
            m=PREFIX.match(line)
            if m:
                ns=int(m[1])
                if ns>=END_MS*1_000_000:
                    stats['stopped_before_cutoff_payload']=True;break
                if ns<last:raise ValueError('kernel UTC clock regressed')
                last=ns
            elif b'"kind":"step"' in line[-100:]:
                raise ValueError('unknown timestamp envelope')
            h.update(line);lines+=1
            if m:
                x=json.loads(line)['frame']
                if x['at']!=last:raise ValueError('timestamp envelope identity')
                yield x
    stats.update(prefix_sha256=h.hexdigest(),lines=lines,last_at_ns=last,cutoff_ms=END_MS)


def causal_bounds(place,cancel,notice):
    if not math.isfinite(place) or not math.isfinite(notice):raise ValueError('missing causal bound')
    upper=min(notice,cancel) if math.isfinite(cancel) else notice
    if upper<place:raise ValueError('reversed causal bounds')
    return place,upper


def classify(t,lo,hi,tol=0.):
    if not all(math.isfinite(x) for x in [t,lo,hi,tol]) or lo>hi or tol<0:raise ValueError('invalid clock comparison')
    return 'before_place' if t<lo-tol else 'after_terminal_or_notice' if t>hi+tol else 'compatible'


def key(row):
    return tuple(row[c] for c in C)


def unique_anchor(indices,anchors):
    if not indices:return None,'leg_unmapped'
    values=anchors.loc[indices,['Z','J','X']].to_numpy()
    if not np.isfinite(values).all():return None,'nonfinite_anchor'
    if not (values==values[0]).all():return None,'ambiguous_leg_clock'
    return dict(zip(['Z','J','X'],map(float,values[0]))),'unique_leg' if len(indices)==1 else 'identical_clock_candidates'


def source_prefix():
    p=HERE/'pf_step4_ladder_v3.py'
    if sha(p)!=ENGINE_SHA:raise ValueError('wrong engine source')
    s=p.read_text();marker='# ---- Binance futures: knowledge time'
    if s.count(marker)!=1:raise ValueError('wrong prefix boundary')
    return s.split(marker)[0]


def setup():
    if DEST.exists():raise ValueError('refuse overwrite')
    scope=json.loads((RP/'scope.json').read_text())
    if scope['days_consumed']!=[DAY] or any(not h.startswith(DAY+'_') for h in scope['hours']):raise ValueError('unbounded cohort')
    if len(scope['windows'])!=220 or any((w['ws']+300)*1000>END_MS-120000 for w in scope['windows']):raise ValueError('cohort identity')
    (DEST/'logs').mkdir(parents=True);(DATA/'step4').mkdir(parents=True)
    for stage in ['step1','step5']:
        (DATA/stage).mkdir();(DATA/stage/DAY).symlink_to(RP/'research'/stage/DAY,target_is_directory=True)
    inputs=[RP/'scope.json',OLD/'matches_20261108.parquet']
    inputs += [OLD/f'{kind}_{run}.parquet' for run in RUNS for kind in ['kernel_orders','core_orders']]
    inputs += [RP/'research/step1'/DAY/f for f in ['registry.parquet','legs_timed2.parquet','legs_tk.parquet','clock_rl_d2.json']]
    inputs += sorted((RP/'research/step1'/DAY/'books').glob('*.parquet'))
    inputs += [RP/'research/step5'/DAY/'preds_t60k.parquet']
    inputs += [RP/'research/step4'/f'ladder_tokyoreal_custom_e2_preds_t60k_lw{t}'/DAY/'summary.json' for t in ['J','X']]
    from . import pf_io as io
    raw=[Path(io.PF_RUN)/'rest'/f'{h}.jsonl.gz' for h in scope['hours']]
    sources=[HERE/'pf_fill_clock.py',HERE/'tests/test_fill_clock.py',HERE/'run_fill_clock.sh',HERE/'pf_step4_ladder_v3.py',HERE/'pf_io.py',HERE/'pf_clock_rl.py',HERE/'pf_sizing_study.py']
    declaration=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),scope='clock conformance, one consumed partial day,220 markets; no strategy/PnL estimate',
        population='all cohort kernel/core order actions; clock evidence available only for exact confirmed maker legs; unavailable identities retained as statuses',
        identity='Signed order_id -> cid -> native market and signed side/price/size; REST full maker hash + transaction/settlement identity; never nearest price/time',
        bounds='necessary interval [kernel Control Place, min(first positive fill notice, confirmed cancel)]; Accepted is an upper bound on activation, not a hard lower match bound',
        timestamp_test='REST executedAt second interval overlap is an assumption check, not assumed actual matching time; wallet-derived match_time is rounded by pf-exec',
        engine_clock='run exact pinned engine prefix through transaction jitter; restore full REST leg identity through all nine original content keys; disagreeing candidate clocks remain ambiguous',
        comparisons=dict(anchors=['J','X'],fixed_earlier_ms=SHIFTS,clock_tolerance_ms=[0,10]),
        null=dict(draws=500,seed=20261109,design='permute first-confirmed-leg anchor among order actions within run/placement-UTC-hour/side; fixed action count; instrument diagnostic only, no CI or inference from one day'),
        multiplicity='8 timing specifications x2 tolerances, no best-delay selection or strategy tuning',
        controls=['exact hash positive and corrupted hash refusal','before/inside/after lifecycle','unknown or conflicting leg identity','future payload not decoded','source mismatch','duplicate fill notices collapse by trade identity'],
        promotion='compatibility is necessary, not sufficient; failing clocks cannot be called calibrated. Full live-order fill-rate validation and later independent days still required.',
        no_strategy_replay_until_clock_issue_resolved=True,forward_data_read=False,live_changed=False,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        source_sha={str(p.relative_to(HERE.parents[1])):sha(p) for p in sources},input_sha={str(p):sha(p) for p in inputs+raw},
        kernel_read='bounded step prefix through Nov8 23:02; no unbounded ledger read; raw prefix hashes captured at extraction',kernel_paths=[str(KX/f'{r}.kernel.jsonl') for r in RUNS])
    write(DEST/'declaration.json',declaration);print('DECLARED clock conformance',flush=True)


def extract():
    new_paths('orders.parquet','fill_events.parquet','own_rest_legs.parquet','kernel_prefixes.json')
    scope=json.loads((RP/'scope.json').read_text());reg=pd.read_parquet(RP/'research/step1'/DAY/'registry.parquet').set_index('mid')
    orders=[];events=[];reads={}
    for run in RUNS:
        k=pd.read_parquet(OLD/f'kernel_orders_{run}.parquet');c=pd.read_parquet(OLD/f'core_orders_{run}.parquet')
        mids={x['market'] for x in scope['windows'] if x['run']==run}
        if k.cid.duplicated().any() or c.cid.duplicated().any() or set(k.cid)!=set(c.cid) or not set(k.market)<=mids:raise ValueError('order population identity')
        c=c.set_index('cid');order={int(x.cid):dict(x._asdict(),run=run) for x in k.itertuples(index=False)}
        oid={};stats={};ixmap={}
        for f in bounded_steps(KX/f'{run}.kernel.jsonl',stats):
            inp=f.get('input');at=f['at']/1e6
            if not isinstance(inp,dict):continue
            p=inp.get('Control',{}).get('Place') if isinstance(inp.get('Control'),dict) else None
            if p and p['cid'] in order:
                o=order[p['cid']];native=int(o['market'])
                if p['market'] in ixmap and ixmap[p['market']]!=native:raise ValueError('kernel market index collision')
                ixmap[p['market']]=native
                if abs(p['size']/1e6-o['size'])>1e-7 or abs(p['price']/1e4-o['price'])>1e-7 or (p['side']=='BuyUp')!=o['up'] or abs(at-o['t_place'])>1e-3:
                    raise ValueError('raw Control identity differs')
                cr=c.loc[p['cid']]
                if cr.market!=native or cr.lane%2!=(0 if o['up'] else 1) or abs(cr.tick-o['price'])>1e-7 or abs(cr['size']-o['size'])>1e-7:
                    raise ValueError('core/kernel action identity differs')
                o['raw_control_verified']=True
            e=inp.get('Exec')
            if not isinstance(e,dict):continue
            if 'Signed' in e and e['Signed']['cid'] in order:
                a=e['Signed'];h=hex_id(a['order_id']);o=order[a['cid']]
                if h in oid and oid[h]!=a['cid']:raise ValueError('signed hash reused')
                if 'order_hash' in o and o['order_hash']!=h:raise ValueError('cid signed twice')
                oid[h]=a['cid'];o['order_hash']=h;o['signed_at']=at
            if 'Fill' in e:
                a=e['Fill'];h=hex_id(a['order_id'])
                if h in oid:
                    events.append(dict(run=run,cid=oid[h],order_hash=h,trade_id=a['trade_id'],at=at,
                        reported_match_ms=a['match_time']/1e6,q=a['size']/1e6,price=a['price']/1e6,status=a['status'],role=a['role']))
        reads[run]=stats
        for cid,o in order.items():
            if not o.get('raw_control_verified'):raise ValueError('cohort action absent from bounded kernel prefix')
            o.update(core_id=int(c.loc[cid,'id']),core_filled=float(c.loc[cid,'filled']))
            o['yes_side'],o['yes_tick'],o['token_price']=order_coordinates(o['up'],o['price'],reg.loc[o['market'],'up_index'])
            o['action_status']='accepted' if math.isfinite(o.get('t_accepted',float('nan'))) else 'signed_not_accepted' if 'order_hash' in o else 'not_signed'
        orders.extend(order.values())
    O=pd.DataFrame(orders);E=pd.DataFrame(events)
    O.to_parquet(DEST/'orders.parquet',index=False);E.to_parquet(DEST/'fill_events.parquet',index=False);write(DEST/'kernel_prefixes.json',reads)
    from . import pf_io as io
    wanted=set(O.order_hash.dropna());allowed=set(O.market);legs={}
    for hour in scope['hours']:
        for recv,params,path,payload in io.rest_rows(hour,('/v1/orders/matches',)):
            for x in payload.get('data') or []:
                mid=(x.get('market') or {}).get('id')
                if mid not in allowed:continue
                for maker in x['makers']:
                    h=maker['hash'].lower()
                    if h not in wanted:continue
                    tx=x['transactionHash'].lower();s,py=io.yes_side(maker);t=x['taker'];uid=(tx,h)
                    row=dict(mid=mid,exec_s=int(io.iso_s(x['executedAt'])),first_recv_ns=int(recv),side=0 if s=='bid' else 1,
                        ptick=int(round(py*100)),shares=int(maker['amount'])/io.E18,price_executed=int(x['priceExecuted'])/io.E18,
                        taker_index=t['outcome']['indexSet'],taker_quote=0 if t['quoteType']=='Bid' else 1,
                        order_hash=h,tx=tx,trade_id=x.get('settlementId') or tx,maker=maker['signer'].lower())
                    if uid not in legs:legs[uid]=row
                    else:
                        old=legs[uid]
                        if old['trade_id']==tx:old['trade_id']=row['trade_id']
                        if row['trade_id']==tx:row['trade_id']=old['trade_id']
                        if any(old[k]!=row[k] for k in row if k!='first_recv_ns'):raise ValueError('REST identity changed across pages')
                        old['first_recv_ns']=min(old['first_recv_ns'],row['first_recv_ns'])
    pd.DataFrame(legs.values()).to_parquet(DEST/'own_rest_legs.parquet',index=False)
    print('EXTRACTED bounded identity artifacts',flush=True)


def anchors():
    new_paths('engine_anchors.parquet')
    from . import pf_io as io
    io.OUT=str(DATA);oldargv=sys.argv;oldenv=dict(os.environ);out={};src=source_prefix()
    try:
        for tm in ['Z','J','X']:
            meta=json.loads((RP/'research/step4'/f'ladder_tokyoreal_custom_e2_preds_t60k_lw{"X" if tm=="X" else "J"}'/DAY/'summary.json').read_text())
            for k in list(os.environ):
                if k.startswith('PF_'):del os.environ[k]
            os.environ.update(meta['env']);os.environ['PF_TAG']='clock_audit_'+tm
            if tm=='Z':os.environ['PF_TXSHIFT']='0';os.environ['PF_TXJIT']=''
            sys.argv=['clock_audit',DAY,'tokyoreal','custom']
            ns=dict(__name__='clock_audit',__package__=__package__,__file__=str(HERE/'pf_step4_ladder_v3.py'))
            exec(compile(src,'pinned_clock_prefix','exec'),ns)
            if tm in ['J','X']:verify_clock_stats(ns['TXSTAT'],meta['tx_timing'])
            f=ns['LT'];out[tm]=np.where(f.how.to_numpy()>0,f.t_ms.to_numpy(),f.exec_s.to_numpy()*1000.).copy()
            del ns,f;gc.collect()
    finally:
        os.environ.clear();os.environ.update(oldenv);sys.argv=oldargv
    pd.DataFrame(out).to_parquet(DEST/'engine_anchors.parquet',index=False)
    print('PINNED ENGINE CLOCKS REPRODUCED',flush=True)


def dedup_events(e):
    rows=[]
    for (run,h,tr),g in e.groupby(['run','order_hash','trade_id'],sort=False):
        g=g.sort_values('at',kind='stable')
        if g.q.nunique()!=1 or g.price.nunique()!=1 or g.cid.nunique()!=1:raise ValueError('fill identity changed')
        positive=g[g.status.isin(['Matched','Confirmed'])]
        rows.append(dict(run=run,order_hash=h,trade_id=tr,cid=int(g.cid.iloc[0]),q=float(g.q.iloc[0]),price=float(g.price.iloc[0]),
            final_status=g.status.iloc[-1],first_positive_notice=float(positive['at'].min()) if len(positive) else float('nan'),
            notice_count=len(g),reported_match_ms=float(g.reported_match_ms.iloc[-1])))
    return pd.DataFrame(rows)


def report():
    new_paths('clock_cases.parquet','confirmed_without_rest.parquet','report.json')
    declaration_path=next(DEST/n for n in ['declaration_v3.json','declaration_v2.json','declaration.json'] if (DEST/n).exists())
    declaration=json.loads(declaration_path.read_text())
    for p,h in declaration['source_sha'].items():
        if sha(HERE.parents[1]/p)!=h:raise ValueError('declared source changed')
    for p,h in declaration['input_sha'].items():
        if sha(p)!=h:raise ValueError('declared input changed')
    for p,h in declaration.get('inherited_extraction_sha',{}).items():
        if sha(DEST/p)!=h:raise ValueError('inherited extraction changed')
    version=declaration.get('version',1)
    control_path=DEST/'logs'/('fill_clock_controls.log' if version==1 else f'fill_clock_controls_v{version}.log')
    controls=control_path.read_text()
    if '\nOK\n' not in controls or 'rc=0 ' not in controls:raise ValueError('missing controls')
    O=pd.read_parquet(DEST/'orders.parquet');E=dedup_events(pd.read_parquet(DEST/'fill_events.parquet'))
    M=pd.read_parquet(DEST/'own_rest_legs.parquet');A=pd.read_parquet(DEST/'engine_anchors.parquet')
    reg=pd.read_parquet(DATA/'step1'/DAY/'registry.parquet').set_index('mid')
    L=pd.read_parquet(DATA/'step1'/DAY/'legs_timed2.parquet',columns=C)
    if len(A)!=len(L):raise ValueError('engine anchor row count')
    want={key(x) for x in M.to_dict('records')};matches=collections.defaultdict(list)
    for i,row in enumerate(L.itertuples(index=False,name=None)):
        if row in want:matches[row].append(i)
    if O.order_hash.dropna().duplicated().any():raise ValueError('order identity collision across runs')
    oi=O.dropna(subset=['order_hash']).set_index('order_hash');ei=E.set_index(['order_hash','trade_id'])
    records=[]
    for x in M.to_dict('records'):
        o=oi.loc[x['order_hash']];y=dict(x,run=o.run,cid=int(o.cid),action_status=o.action_status,place=float(o.t_place),
            accepted=float(o.t_accepted),cancel=float(o.t_CancelPendingAudit),status='eligible')
        token_price=verify_leg_order(o,x,reg.loc[x['mid'],'up_index'])
        ix=(x['order_hash'],x['trade_id'])
        if ix not in ei.index:y['status']='no_exact_kernel_trade'
        else:
            ev=ei.loc[ix]
            if isinstance(ev,pd.DataFrame):raise ValueError('duplicate fill identity')
            if abs(ev.q-x['shares'])>1e-6 or abs(ev.price-token_price)>1e-6:raise ValueError('exact fill quantity/price differs')
            y.update(final_status=ev.final_status,first_positive_notice=ev.first_positive_notice)
            if ev.final_status!='Confirmed':y['status']='not_confirmed_in_kernel_prefix'
            elif not math.isfinite(ev.first_positive_notice):y['status']='no_positive_notice'
            else:
                lo,hi=causal_bounds(y['place'],y['cancel'],ev.first_positive_notice);y.update(lower=lo,upper=hi,
                    exec_second_overlaps=(x['exec_s']*1000<=hi and (x['exec_s']+1)*1000>lo),bounds_width_ms=hi-lo)
        anc,st=unique_anchor(matches[key(x)],A);y['leg_clock_status']=st;y['candidate_legs']=len(matches[key(x)])
        if anc is not None:y.update(anc)
        if y['status']=='eligible' and anc is None:y['status']=st
        records.append(y)
    R=pd.DataFrame(records);R.to_parquet(DEST/'clock_cases.parquet',index=False)
    q=R[R.status=='eligible'].copy()
    if not len(q):raise ValueError('no exact eligible clock evidence')
    specs={'J':q.J,'X':q.X,**{f'earlier_{d}ms':q.Z-d for d in SHIFTS}};checks={}
    for name,times in specs.items():
        for tol in [0,10]:
            lab=pd.Series([classify(t,l,u,tol) for t,l,u in zip(times,q.lower,q.upper)],index=q.index)
            bad=lab!='compatible';key_=f'{name}/tol{tol}'
            checks[key_]=dict(legs=len(q),orders=int(q.order_hash.nunique()),leg_status=lab.value_counts().to_dict(),
                violating_orders=int(q.loc[bad,'order_hash'].nunique()),violating_shares=float(q.loc[bad,'shares'].sum()),all_compatible=bool(not bad.any()))
            q['status_'+key_]=lab
    # An earlier constant delay d must lie in every [anchor-upper, anchor-lower] interval.
    lower=float((q.Z-q.upper).max());upper=float((q.Z-q.lower).min())
    def witness(i):
        a=q.loc[i]
        return dict(run=a.run,cid=int(a.cid),order_hash=a.order_hash,tx=a.tx,Z=float(a.Z),lower=float(a.lower),upper=float(a.upper))
    per_order=q.sort_values(['exec_s','tx']).drop_duplicates('order_hash').copy()
    per_order['hour']=(per_order.place//3600000).astype('int64');per_order.reset_index(drop=True,inplace=True)
    groups=list(per_order.groupby(['run','hour','side'],sort=False).indices.values());rng=np.random.default_rng(20261109);null=[]
    z=per_order.Z.to_numpy();l=per_order.lower.to_numpy();u=per_order.upper.to_numpy()
    for _ in range(500):
        a=z.copy()
        for ix in groups:a[ix]=rng.permutation(a[ix])
        null.append(float(((a>=l)&(a<=u)).mean()))
    booked=M.groupby('order_hash').shares.sum();known=oi.core_filled
    dif=(known-booked.reindex(known.index,fill_value=0.)).abs()
    rest_keys=set(zip(M.order_hash,M.trade_id))
    missing=E[(E.final_status=='Confirmed')&~pd.Series(list(zip(E.order_hash,E.trade_id))).isin(rest_keys)]
    missing.to_parquet(DEST/'confirmed_without_rest.parquet',index=False)
    null_out=dict(draws=500,order_actions=len(per_order),permutable_actions=sum(len(ix) for ix in groups if len(ix)>1),
        observed_compatible=float(((z>=l)&(z<=u)).mean()),null_mean=float(np.mean(null)),null_max=float(max(null)),
        interpretation='matched action diagnostic; no day-level CI or profitability inference')
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),declaration=declaration,declaration_sha=sha(declaration_path),source_controls=controls,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        population=dict(day=DAY,markets=220,order_actions=len(O),by_action_status=O.action_status.value_counts().to_dict(),
            raw_fill_notices=len(pd.read_parquet(DEST/'fill_events.parquet')),unique_kernel_trades=len(E),rest_maker_legs=len(R),
            rest_filled_orders=int(R.order_hash.nunique()),eligible_clock_legs=len(q),eligible_clock_orders=int(q.order_hash.nunique()),
            case_status=R.status.value_counts().to_dict(),leg_clock_status=R.leg_clock_status.value_counts().to_dict(),
            kernel_final_trade_status=E.final_status.value_counts().to_dict(),confirmed_kernel_without_rest=len(missing)),
        live_quantity_reconciliation=dict(signed_orders=len(oi),orders_differing_over_microshare=int((dif>1e-6).sum()),
            max_difference=float(dif.max()),core_shares=float(known.sum()),rest_shares=float(M.shares.sum())),
        checks=checks,constant_shift_interval_ms=dict(lower=lower,upper=upper,exists=lower<=upper,exists_earlier=max(0.,lower)<=upper,
            exists_with_10ms_tolerance=lower-10<=upper+10,
            lower_witness=witness((q.Z-q.upper).idxmax()),upper_witness=witness((q.Z-q.lower).idxmin())),
        exec_second_nonoverlap_legs=int((~q.exec_second_overlaps).sum()),causal_width_ms=q.bounds_width_ms.quantile([0,.25,.5,.75,1]).to_dict(),
        null_diagnostic=null_out,kernel_prefixes=json.loads((DEST/'kernel_prefixes.json').read_text()),
        artifacts={f.name:sha(f) for f in DEST.glob('*.parquet')},
        predicates=dict(any_fixed_clock_conforms_at10ms=any(x['all_compatible'] for k,x in checks.items() if k.endswith('/tol10')),
            original_JX_conform=all(checks[f'{t}/tol10']['all_compatible'] for t in ['J','X']),
            quantity_exact=bool((dif<=1e-6).all()),exact_clock_population_complete=bool((R.status=='eligible').all())),
        limitations=['one consumed partial day','lifecycle bounds constrain matching time but do not identify it',
            'compatible filled-order clocks do not prove correct unfilled-order queue/fill predictions',
            'no strategy ranking, live calibration claim, deployment or future-day read'])
    write(DEST/'report.json',out);write(HERE.parent/'reviews/fill_clock_20261109/receipt.json',out)
    print(json.dumps({k:out[k] for k in ['population','live_quantity_reconciliation','checks','constant_shift_interval_ms','exec_second_nonoverlap_legs','null_diagnostic','predicates']},indent=2))


if __name__=='__main__':
    {'declare':setup,'amend':amendment,'amend3':coordinate_amendment,'extract':extract,'anchors':anchors,'report':report}[sys.argv[1]]()
