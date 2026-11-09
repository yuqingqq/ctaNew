"""Stream native states to flat artifacts and reconcile exact prior decision IDs."""
import collections
import datetime as dt
import gzip
import json
import math
import subprocess
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from . import pf_native_guard as n

DELTAS=['0.005','0.01','0.02','0.04']


def same_decision(row,ref):
    if row['t']!=ref[0]:raise ValueError('decision timestamp differs')
    for lane,(tick,size) in enumerate(row['lanes']):
        if (-1 if tick is None else tick)!=ref[1+lane*2] or size!=ref[2+lane*2]:raise ValueError('decision target differs')


def flatten(d,run,ordinal):
    book=d['book'];bid=book['bids'][0][0] if book else -1;ask=book['asks'][0][0] if book else -1
    x=dict(run=run,mid=int(d['market']),t=float(d['t']),ordinal=ordinal,input_index=int(d['input_index']),
        rel_s=float(d['rel_s']),v_bid=d['v'][0],v_ask=d['v'][1],binance_bid=d['binance'][0],binance_ask=d['binance'][1],
        net=float(d['net']),live_bid=float(d['live'][0]),live_ask=float(d['live'][1]),held=d['held'],
        bid=int(bid),ask=int(ask),midpoint=(bid+ask)/2 if 0<bid<ask<100 else None)
    for prefix,lanes in [('raw',d['lanes'])]+[(f'd{v}',d['relaxed'][v]) for v in DELTAS]:
        for lane,(tick,size) in enumerate(lanes):x[f'{prefix}_tick{lane}']=-1 if tick is None else tick;x[f'{prefix}_size{lane}']=float(size)
    return x


def state_schema():
    types={'run':pa.string(),'mid':pa.int64(),'t':pa.float64(),'ordinal':pa.int64(),'input_index':pa.int64(),
        'rel_s':pa.float64(),'v_bid':pa.float64(),'v_ask':pa.float64(),'binance_bid':pa.bool_(),'binance_ask':pa.bool_(),
        'net':pa.float64(),'live_bid':pa.float64(),'live_ask':pa.float64(),'held':pa.bool_(),'bid':pa.int64(),'ask':pa.int64(),'midpoint':pa.float64()}
    for prefix in ['raw']+[f'd{v}' for v in DELTAS]:
        for lane in range(4):types[f'{prefix}_tick{lane}']=pa.int64();types[f'{prefix}_size{lane}']=pa.float64()
    return pa.schema(types)


def flatten_file(src,dst,run,check_row=None):
    schema=state_schema();batch=[];total=0
    with pq.ParquetWriter(dst,schema) as writer,gzip.open(src,'rt') as reader:
        for i,line in enumerate(reader):
            d=json.loads(line)
            if check_row:check_row(d)
            batch.append(flatten(d,run,i));total+=1
            if len(batch)>=5000:writer.write_table(pa.Table.from_pylist(batch,schema=schema));batch=[]
        if batch:writer.write_table(pa.Table.from_pylist(batch,schema=schema))
    return total


def main():
    a=n.a;n.check()
    if (n.DEST/'verification.json').exists():raise ValueError('verification exists')
    control=dict(t=1.,lanes=[[50,5.],[None,0.],[None,0.],[None,0.]])
    ref=[1.,50,5.,-1,0.,-1,0.,-1,0.];same_decision(control,ref);bad=ref.copy();bad[2]=25.;refused=False
    try:same_decision(control,bad)
    except ValueError:refused=True
    if not refused:raise ValueError('changed-size control accepted')
    old=pd.read_parquet(a.ROOT/'data/predict_fun/research/quote_capture_20261109/decisions.parquet')
    by={(r,m):g[['t']+[v for lane in range(4) for v in (f'tick{lane}',f'size{lane}')]].to_numpy() for (r,m),g in old.groupby(['run','mid'])}
    scope=json.loads((a.RP/'scope.json').read_text());wanted={(w['run'],w['market']):w['ws']*1000 for w in scope['windows']}
    seen=collections.Counter();reports={};diag={}
    for run in n.RUNS:
        native=n.DEST/run/'evidence';r=json.loads((native/'replay.json').read_text())
        if not r['outputs_exact_for_applied_inputs'] or not r['pure_quote_function_exact']:raise ValueError('native replay failed')
        params=json.loads((n.NATIVE/json.loads((n.DEST/'declaration.json').read_text())['headers'][run]['ladder']['params']['path']).read_text())
        line=params['minute_lines']['pull_and_ladder_line_90'][0];counts=collections.Counter()
        def check(d):
            key=(run,int(d['market']));ws=wanted[key]
            if not ws<=d['t']<ws+65000:raise ValueError('export outside scope')
            if d['t']>=ws+60000:return
            j=seen[key];arr=by[key]
            if j>=len(arr):raise ValueError('extra native decision')
            same_decision(d,arr[j]);seen[key]+=1
            for lane,(tick,size) in enumerate(d['lanes']):
                if tick is not None and size>0:
                    counts['wanted_lane_updates']+=1;v=d['v'][lane%2]
                    if v is None or not math.isfinite(v) or v<=line or d['binance'][lane%2] or d['held']:
                        raise ValueError('native wanted lane violates exact guard')
                    for delta in DELTAS:
                        # Looser lines may add a rung at the touch, changing touch de-duplication;
                        # therefore relaxed targets are not blindly assumed a superset.
                        counts[f'original_wanted_removed_by_relaxed_{delta}']+=int(d['relaxed'][delta][lane][0] is None)
                elif tick is None:
                    for delta in DELTAS:counts[f'empty_lane_opened_by_relaxed_{delta}']+=int(d['relaxed'][delta][lane][0] is not None)
        total=flatten_file(native/'states.jsonl.gz',n.DEST/f'states_{run}.parquet',run,check)
        if total!=r['exported_decisions']:raise ValueError('native state count differs')
        reports[run]=r;diag[run]=dict(counts)
    if set(seen)!=set(by) or any(seen[k]!=len(v) for k,v in by.items()):raise ValueError('missing first-minute decisions')
    # Verify the stored target artifacts, not only the streaming comparison.
    for run in n.RUNS:
        q=pd.read_parquet(n.DEST/f'states_{run}.parquet');q=q[q.rel_s<60].sort_values(['mid','t','ordinal'],kind='stable')
        p=old[old.run==run].sort_values(['mid','t','sequence'],kind='stable')
        for lane in range(4):
            if not (q[f'raw_tick{lane}'].to_numpy()==p[f'tick{lane}'].to_numpy()).all() or not (q[f'raw_size{lane}'].to_numpy()==p[f'size{lane}'].to_numpy()).all():raise ValueError('saved targets differ')
        if not (q.t.to_numpy()==p.t.to_numpy()).all() or not (q.mid.to_numpy()==p.mid.to_numpy()).all():raise ValueError('saved identities differ')
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.HERE,text=True).strip(),
        verifier_sha256=a.sha(a.HERE/'pf_native_guard_verify.py'),population=dict(windows=len(seen),first_minute_decisions=sum(seen.values()),excluded_windows=0),
        controls=dict(positive=True,changed_size_refused=refused),exact_guard_diagnostics=diag,
        native_reports=reports,artifact_sha256={p.name:a.sha(p) for p in n.DEST.glob('states_*.parquet')},
        native_evidence_sha256={f'{run}/{p}':a.sha(n.DEST/run/'evidence'/p) for run in n.RUNS for p in ['states.jsonl.gz','books.jsonl.gz','replay.json']},
        limitation='exact factual decision evidence; relaxed targets conditional on recorded inventory/order state; no profitability or hypothetical fill validation')
    a.write(n.DEST/'verification.json',out);a.write(n.REVIEW/'verification.json',out)
    print(json.dumps({k:v for k,v in out.items() if k not in ['native_reports','native_evidence_sha256']},indent=2))


if __name__=='__main__':main()
