"""Rebuild the consumed Nov8 cohort with exact identities and honest clock precision.

declare -> controls -> build. No fitted parameters, PnL, or forward data.
"""
import collections
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd
from . import pf_fill_clock as audit
from . import pf_trade_tape as tape
from . import pf_io as io

DEST=audit.ROOT/'data/predict_fun/research/trade_timing_fix_20261109'
REVIEW=audit.HERE.parent/'reviews/trade_timing_fix_20261109'
SOURCES=['pf_trade_tape.py','pf_trade_tape_build.py','pf_step4_ladder_identity.py','tests/test_trade_tape.py',
         'pf_step1.py','pf_step1_timing.py','pf_step1_booktime.py','pf_txmap.py','pf_fill_clock.py','pf_io.py',
         'pf_clock_preflight.py','run_forward.sh','run_forward_v3.sh','run_c2_day.sh']


def declare():
    if DEST.exists():raise ValueError('refuse overwriting a repair run')
    DEST.mkdir(parents=True);(DEST/'logs').mkdir()
    original=json.loads((audit.DEST/'report.json').read_text())
    scope=json.loads((audit.RP/'scope.json').read_text())
    if scope['days_consumed']!=[audit.DAY] or len(scope['windows'])!=220:raise ValueError('scope differs')
    if any(not h.startswith(audit.DAY+'_') for h in scope['hours']):raise ValueError('forward hour in scope')
    inputs={str(audit.DEST/p):audit.sha(audit.DEST/p) for p in ['report.json','orders.parquet','own_rest_legs.parquet','kernel_prefixes.json']}
    inputs[str(audit.RP/'scope.json')]=audit.sha(audit.RP/'scope.json')
    for h in scope['hours']:
        path=str(Path(io.PF_RUN)/'rest'/f'{h}.jsonl.gz')
        inputs[path]=original['declaration']['input_sha'][path]
    d=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),
        design='engineering correction on220 already-consumed Nov8 windows; exact identity, separate clocks, no timestamp fitting or PnL test',
        population='all REST maker legs in the declared markets/hours, including pre/post-confirmation phases; all42064 live order actions',
        clocks='kernel status timestamps kept as floor-second intervals; REST confirmation never supplies a matching time; no lifecycle clipping or other-leg propagation',
        controls='full-hash collision, duplicate/conflicting economics, timestamp drift, wrong-identity refusal, coarse-clock refusal, source/hash/population corruption, entrypoint refusal',
        null='no statistical hypothesis, model selection, fitted delay or profit result; prior500-draw audit unchanged',
        multiplicity='one deterministic repair; no strategy arms',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=audit.HERE,text=True).strip(),
        sources={p:audit.sha(audit.HERE/p) for p in SOURCES},inputs=inputs,
        protected_dates_read=False,live_changed=False)
    audit.write(DEST/'declaration.json',d);audit.write(REVIEW/'declaration.json',d)
    print('DECLARED deterministic clock repair')


def build():
    if (DEST/'legs.parquet').exists() or (DEST/'receipt.json').exists():raise ValueError('refuse overwriting repair artifacts')
    declaration_path=DEST/('declaration_v2.json' if (DEST/'declaration_v2.json').exists() else 'declaration.json')
    d=json.loads(declaration_path.read_text())
    for p,h in d['sources'].items():
        if audit.sha(audit.HERE/p)!=h:raise ValueError('declared source changed: '+p)
    for p,h in d['inputs'].items():
        if audit.sha(p)!=h:raise ValueError('declared input changed: '+p)
    controls=(DEST/'logs'/('controls_v2.log' if d.get('version')==2 else 'controls.log')).read_text()
    if '\nOK\n' not in controls or 'rc=0 ' not in controls:raise ValueError('missing passing controls')
    scope=json.loads((audit.RP/'scope.json').read_text());mids={w['market'] for w in scope['windows']}
    O=pd.read_parquet(audit.DEST/'orders.parquet');own=set(O.order_hash.dropna());cids=set(map(int,O.cid))
    event=[];prefixes={};places={};cancels={}
    for run in audit.RUNS:
        stats={}
        for f in audit.bounded_steps(audit.KX/f'{run}.kernel.jsonl',stats):
            inp=f.get('input')
            if not isinstance(inp,dict):continue
            control=inp.get('Control');ex=inp.get('Exec')
            if isinstance(control,dict) and 'Place' in control and control['Place']['cid'] in cids:
                places[control['Place']['cid']]=f['at']
            if not isinstance(ex,dict):continue
            if 'CancelPendingAudit' in ex:
                h=audit.hex_id(ex['CancelPendingAudit']['order_id'])
                if h in own:cancels[h]=min(cancels.get(h,f['at']),f['at'])
            if 'Fill' in ex:
                fill=ex['Fill'];h=audit.hex_id(fill['order_id'])
                if h in own:
                    row=tape.status_event(h,fill['trade_id'],fill['status'],fill['match_time'],f['at'],fill['size'],fill['price'])
                    event.append(dict(row,run=run))
        prefixes[run]=stats
    oldprefix=json.loads((audit.DEST/'kernel_prefixes.json').read_text())
    if prefixes!=oldprefix:raise ValueError('bounded kernel prefix changed')
    if set(places)!=cids:raise ValueError('action population differs from kernel')
    legs={}
    for hour in scope['hours']:
        for recv,params,path,payload in io.rest_rows(hour,('/v1/orders/matches',)):
            for trade in payload.get('data') or []:
                if (trade.get('market') or {}).get('id') not in mids:continue
                for maker in trade['makers']:
                    row=tape.rest_leg(trade,maker,int(recv));k=row['leg_id']
                    if k in legs:tape.merge_leg(legs[k],row)
                    else:legs[k]=row
        io.mem_check(hour)
        print('READ',hour,'distinct full-identity legs',len(legs),flush=True)
    rows,used=tape.attach_status(legs.values(),event);del legs
    L=pd.DataFrame(rows)
    for col in ['match_lo_ns','match_hi_ns','matched_notice_ns','confirmed_notice_ns','status_changed_ns']:
        L[col]=pd.array([r[col] for r in rows],dtype='Int64')
    del rows
    if set(map(int,L.mid.unique()))!=mids:raise ValueError('REST market population differs from scope')
    L['transaction_group']=tape.transaction_groups(L)
    L.to_parquet(DEST/'legs.parquet',index=False)
    pd.DataFrame(event).to_parquet(DEST/'status_events.parquet',index=False)
    owned=L[L.maker_order_hash.isin(own)].copy();reference=pd.read_parquet(audit.DEST/'own_rest_legs.parquet')
    expected={(x.tx,x.order_hash,x.trade_id) for x in reference.itertuples()}
    actual={(x.transaction_hash,x.maker_order_hash,x.settlement_id or x.transaction_hash) for x in owned.itertuples()}
    if actual!=expected:raise ValueError('confirmed own fill identities differ from original exact audit')
    sums=owned.groupby('maker_order_hash').shares.sum()
    O['tape_filled']=O.order_hash.map(sums).fillna(0.)
    O['quantity_difference']=(O.core_filled-O.tape_filled).abs()
    if (O.quantity_difference>1e-6).any():raise ValueError('core quantity reconciliation differs')
    O.to_parquet(DEST/'action_reconciliation.parquet',index=False)
    oi=O.dropna(subset=['order_hash']).set_index('order_hash');cases=[]
    for x in owned.itertuples():
        o=oi.loc[x.maker_order_hash]
        positive=[e['receipt_ns'] for e in event if e['maker_order_hash']==x.maker_order_hash and e['trade_id']==(x.settlement_id or x.transaction_hash) and e['status'] in ('Matched','Confirmed')]
        lo=places[int(o.cid)];hi=min(min(positive),cancels.get(x.maker_order_hash,min(positive)))
        if hi<lo:raise ValueError('reversed lifecycle')
        overlap=(int(x.match_lo_ns)<=hi and int(x.match_hi_ns)>lo) if pd.notna(x.match_lo_ns) else False
        cases.append(dict(leg_id=x.leg_id,order_hash=x.maker_order_hash,cid=int(o.cid),run=o.run,
            lifecycle_lo_ns=lo,lifecycle_hi_ns=hi,match_lo_ns=x.match_lo_ns,match_hi_ns=x.match_hi_ns,
            clock_status=x.clock_status,interval_overlaps=overlap))
    C=pd.DataFrame(cases);C.to_parquet(DEST/'own_clock_cases.parquet',index=False)
    unmatched=[]
    for key,g in pd.DataFrame(event).groupby(['maker_order_hash','trade_id']):
        if key not in used:
            state=tape.resolve_status(g.to_dict('records'))
            unmatched.append(dict(maker_order_hash=key[0],trade_id=key[1],**state))
    pd.DataFrame(unmatched).to_parquet(DEST/'unmatched_status_events.parquet',index=False)
    manifest=dict(schema=tape.SCHEMA,day=audit.DAY,legs=len(L),markets=sorted(mids),
        legs_sha256=audit.sha(DEST/'legs.parquet'),scope_sha256=audit.sha(audit.RP/'scope.json'),
        clock_status=L.clock_status.value_counts().to_dict(),exact_point_events=int((L.clock_status=='matching_engine_event').sum()),
        no_economic_or_timing_population_filter=True)
    audit.write(DEST/'manifest.json',manifest)
    refusal=None
    try:tape.load_replay_tape(DEST,audit.DAY)
    except ValueError as e:refusal=str(e)
    if refusal is None:raise ValueError('coarse/unresolved tape was accepted as point clocks')
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),declaration=d,declaration_sha256=audit.sha(declaration_path),
        population=dict(markets=len(mids),rest_legs=len(L),order_actions=len(O),own_confirmed_legs=len(owned),
            own_filled_orders=int(owned.maker_order_hash.nunique()),raw_status_notices=len(event),
            clock_status=L.clock_status.value_counts().to_dict(),unmatched_status=collections.Counter(x['clock_status'] for x in unmatched)),
        price_grid_status=L.price_grid_status.value_counts().to_dict(),
        reconciliation=dict(own_exact_identities_equal=actual==expected,ambiguous_own_identities=int(owned.leg_id.duplicated().sum()),
            quantity_mismatches=int((O.quantity_difference>1e-6).sum()),max_quantity_difference=float(O.quantity_difference.max()),
            own_interval_overlaps=int(C.interval_overlaps.sum()),own_interval_nonoverlap=int((~C.interval_overlaps).sum()),
            exact_subsecond_match_times=int((L.clock_status=='matching_engine_event').sum())),
        replay_refusal=refusal,source_controls=controls,kernel_prefixes=prefixes,
        artifacts={p.name:audit.sha(p) for p in DEST.glob('*.parquet')},manifest_sha256=audit.sha(DEST/'manifest.json'),
        predicates=dict(point_replay_refused=refusal is not None,full_population_retained=len(O)==42064 and actual==expected,
            own_intervals_consistent=bool(C.interval_overlaps.all()),strategy_replay_ready=bool((L.clock_status=='matching_engine_event').all())),
        limitations=['matching-status intervals are not exact matching-engine timestamps',
            'lifecycle bounds are used only for independent checks, never clipping or timestamp fitting',
            'unknown public trade clocks remain rows; no dropping or propagating another leg clock',
            'one consumed partial day, deterministic engineering regression, no PnL estimate or statistical validation'])
    audit.write(DEST/'receipt.json',out);audit.write(REVIEW/'receipt.json',out)
    print(json.dumps({k:out[k] for k in ['population','reconciliation','replay_refusal','predicates']},indent=2))


def amend():
    old=json.loads((DEST/'declaration.json').read_text())
    d=dict(old,version=2,as_of=dt.datetime.now(dt.timezone.utc).isoformat(),
        supersedes_sha256=audit.sha(DEST/'declaration.json'),
        correction='retain exact off-cent-grid REST prices with explicit status and nullable ptick; refuse these prices in point replay instead of rounding or dropping trades',
        failed_build_log_sha256=audit.sha(DEST/'logs/build.log'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=audit.HERE,text=True).strip(),
        sources={p:audit.sha(audit.HERE/p) for p in SOURCES})
    audit.write(DEST/'declaration_v2.json',d);audit.write(REVIEW/'declaration_v2.json',d)
    print('DECLARED v2 price-status preservation; no result-bearing v1 receipt')


if __name__=='__main__':{'declare':declare,'build':build,'amend':amend}[sys.argv[1]]()
