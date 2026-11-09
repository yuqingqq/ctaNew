"""Pinned native A2 execution experiment on the already consumed intent path.

This is not a full counterfactual live-core replay: no hypothetical fills or
recomputed predictions/inventory, and venue feedback is explicitly assumed.
"""
import datetime as dt
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from . import pf_native_guard as native
from . import pf_native_inventory_events as inventory

a=native.a
ROOT=Path('/tmp/pf-a2-parity-20261109')
PIN='f20fdb1'
DEST=a.ROOT/'data/predict_fun/research/optimization_loop_20261109/native_a2'
REVIEW=a.HERE.parent/'reviews/optimization_loop_20261109/native_a2'
TARGET=native.TARGET
NAME='native_a2_continuity'
REL='crates/pf-engine/src/ladder.rs'
SOURCES=['pf_native_a2_continuity.py','native_a2_continuity.rs','tests/test_native_a2_continuity.py']
ARMS=['reference','all100','update_first50']
OFF=[[None,0.] for _ in range(4)]
LIMITS=['Consumed partial Nov8 day only; no CI or independent validation.',
    'Factual quote targets, guards and inventory held fixed; no counterfactual fills or recomputed policy.',
    'Native f20fdb1 A2 mapping, request throttle and retry code used directly; not a full B4 generated-engine integration.',
    'Venue assumed create/answer40ms, accepted-cancel22ms, pending-create cancel>=95ms, external reservation grace1000ms.',
    'External capacity150 shares per side at hypothetical net0; this is not the dollar loss budget. Factual net only conditions exposure measures.',
    'No post-only/rejection model, wallet/kernel I/O or shared cross-market throttling. Native XV is per-window as coded.',
    'Equal-clock update priority is a proposed scheduling contract, not demonstrated live processing latency.',
    'Historical fill timestamps remain uncalibrated; quote coverage and exposure are not PnL or inventory variance.']

def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(p):return a.sha(p)
def command(cmd,**kw):return subprocess.check_output(cmd,text=True,**kw).strip()

def instrument(original):
    start='        let wants = self.order_wants(&d.lanes);\n'
    end='        self.last_decision = Some((t, d));'
    if original.count(start)!=1 or original.count(end)!=1:raise ValueError('native request-loop anchor changed')
    body=original.split(start)[1].split(end)[0]
    block=start.replace('&d.lanes','lanes')+body
    methods='''    // Research-only access to unchanged A2 methods; generated in an isolated worktree.
    pub fn diagnostic_clear_events(&mut self) { self.triggers.clear(); }
    pub fn diagnostic_current(&self, lane: usize) -> Option<u64> { self.cur(lane) }
    pub fn diagnostic_wants(&mut self, t: f64, lanes: &[Want; LANES], out: &mut Vec<Request>) {
'''+block+'''    }
    pub fn diagnostic_pull(&mut self, lane: usize, t: f64, out: &mut Vec<Request>) {
        assert!(lane < LANES);
        self.pair_total[lane] = None;
        self.pair_filled[lane] = 0.0;
        for member in [lane, lane + LANES] {
            self.intent[member] = Some((None, 0.0));
            self.request(t, member, (None, 0.0), out);
        }
    }

'''
    anchor='    pub fn ws_ms(&self) -> f64 {'
    if original.count(anchor)!=1:raise ValueError('native accessor anchor changed')
    changed=original.replace(anchor,methods+anchor)
    if changed.replace(methods,'',1)!=original:raise ValueError('observer changes native implementation')
    return changed

def build():
    DEST.mkdir(parents=True,exist_ok=True)
    commit=command(['git','rev-parse','HEAD'],cwd=ROOT)
    if not commit.startswith(PIN):raise ValueError('wrong native version')
    original=command(['git','show',f'HEAD:{REL}'],cwd=ROOT)+'\n'
    path=ROOT/REL
    if path.read_text() not in [original,instrument(original)]:raise ValueError('unexpected native edit')
    path.write_text(instrument(original))
    copy=ROOT/f'crates/pf-engine/examples/{NAME}.rs';copy.parent.mkdir(exist_ok=True);shutil.copyfile(a.HERE/f'{NAME}.rs',copy)
    base=['cargo','test','--locked','--offline','-p','pf-engine','--example',NAME,'--target-dir',str(TARGET),'-j','1']
    subprocess.run(base,cwd=ROOT,check=True)
    # Also exercise the existing native paired-order tests, not only the adapter.
    subprocess.run(['cargo','test','--locked','--offline','-p','pf-engine','--lib','layer','--target-dir',str(TARGET),'-j','1'],cwd=ROOT,check=True)
    base[1]='build';base.append('--release');subprocess.run(base,cwd=ROOT,check=True)
    write(DEST/'build.json',dict(as_of=now(),native_commit=commit,original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        instrumented_sha256=sha(path),binary_sha256=sha(TARGET/f'release/examples/{NAME}'),example_sha256=sha(copy),
        native_request_loop_copied_exactly=True,accessor_removal_restores_original=True))

def sources():return {p:sha(a.HERE/p) for p in SOURCES}

def inputs():
    files=[a.RP/'scope.json',DEST/'build.json']
    for run in native.RUNS:
        files.extend([native.DEST/f'states_{run}.parquet',native.DEST/f'grid/probes_{run}.parquet',inventory.DEST/f'inventory_{run}.parquet'])
    files.extend([a.ROOT/'data/predict_fun/research/optimization_loop_20261109/findings_v2.json'])
    return {str(p):sha(p) for p in files}

def declare():
    d=dict(as_of=now(),source_commit=command(['git','rev-parse','HEAD'],cwd=a.HERE),sources=sources(),inputs=inputs(),
        build=json.loads((DEST/'build.json').read_text()),arms=ARMS,primary='update_first50',
        population='all220 pre-existing consumed Nov8 windows, first60s, all279059 decisions and13077 neutral touch-pull actions per arm; exclusions as statuses',
        hypothesis='Does updates-first50ms retain fewer requests and greater original-price coverage under current native A2 throttle/retry semantics? Report full costs and all arms; do not tune.',
        null='deterministic paired execution mechanics; no statistical null, profit/skill/variance claim or confidence interval',
        policy='same .01-cent band, accepted small touch<=5, hard/native relaxed-price-size checks, no rung duplicate, no new order while guard off; accepted small member may occupy either physical pair member',
        timing='all100 deadline before same-clock data; update_first50 after all same-clock source events; feedback before native retries; original decisions alone drive new wants; probes may cancel held pairs only',
        controls='native split/partial-fill/front rotation/retry/rejection; equal-clock positive and later-event refusal; hard probe; reservation boundary; malformed input, instrument anchor and corrupted artifact refusal; independently rebuild coverage from saved order intervals',
        multiplicity=dict(new_policy_parameters=0,existing_policies_retested=3,cumulative_distinct_policies=21,new_execution_model=1,latency_scenarios=1),
        limitations=LIMITS,no_live_change=True,no_protected_dates_read=True,no_fill_clock_bypass=True)
    write(DEST/'declaration.json',d);write(REVIEW/'declaration.json',d)

def check():
    d=json.loads((DEST/'declaration.json').read_text())
    if sources()!=d['sources']:raise ValueError('declared sources changed')
    for p,h in d['inputs'].items():
        if sha(p)!=h:raise ValueError(f'declared input changed {p}')
    b=d['build']
    if command(['git','rev-parse','HEAD'],cwd=ROOT)!=b['native_commit'] or sha(ROOT/REL)!=b['instrumented_sha256']:raise ValueError('native implementation changed')
    if command(['git','diff','--name-only'],cwd=ROOT)!=REL:raise ValueError('unexpected native source changes')
    if sha(TARGET/f'release/examples/{NAME}')!=b['binary_sha256']:raise ValueError('native executable changed')
    return d

def quote(row,prefix):
    return [[None if pd.isna(row[f'{prefix}_tick{l}']) else int(row[f'{prefix}_tick{l}']),float(row[f'{prefix}_size{l}'])] for l in range(4)]

def prepare():
    check();scope=json.loads((a.RP/'scope.json').read_text())
    if scope['days_consumed']!=['20261108'] or len(scope['windows'])!=220:raise ValueError('wrong cohort')
    cols=['mid','t','input_index','ordinal','v_bid','v_ask','binance_bid','binance_ask','held','net']
    cols += [f'{p}_{field}{l}' for p in ['raw','d0.01'] for field in ['tick','size'] for l in range(4)]
    statuses=[]
    with gzip.open(DEST/'input.jsonl.gz','xt',compresslevel=1) as f:
        for run in native.RUNS:
            data=[]
            for kind,p in enumerate([native.DEST/f'states_{run}.parquet',native.DEST/f'grid/probes_{run}.parquet',inventory.DEST/f'inventory_{run}.parquet']):
                z=pd.read_parquet(p,columns=cols);z['kind']=kind;data.append(z)
            grouped=pd.concat(data,ignore_index=True).groupby('mid',sort=False)
            for w in scope['windows']:
                if w['run']!=run:continue
                start=w['ws']*1000.;mid=w['market'];z=grouped.get_group(mid)
                z=z[(z.t>=start)&(z.t<start+60000)].sort_values(['t','input_index','kind','ordinal'],kind='stable')
                events=[];previous=OFF;actions=decisions=0
                for r in z.to_dict('records'):
                    e=dict(t=r['t']-start,kind=int(r['kind']),input_index=int(r['input_index']),ordinal=int(r['ordinal']),
                        raw=quote(r,'raw'),relaxed=quote(r,'d0.01'),v=[None if pd.isna(r[k]) else float(r[k]) for k in ['v_bid','v_ask']],
                        binance=[bool(r[k]) for k in ['binance_bid','binance_ask']],held=bool(r['held']),net=float(r['net']))
                    if e['kind']==0:
                        decisions+=1;actions+=sum(previous[l][0] is not None and e['raw'][l][0] is None for l in [0,1]);previous=e['raw']
                    events.append(e)
                if not decisions:raise ValueError('missing window')
                f.write(json.dumps(dict(run=run,mid=mid,start=start,events=events),allow_nan=False,separators=(',',':'))+'\n')
                statuses.append(dict(run=run,mid=mid,status='included',events=len(events),decisions=decisions,actions=actions))
            del grouped,data
    assert sum(s['decisions'] for s in statuses)==279059
    assert sum(s['actions'] for s in statuses)==13077
    write(DEST/'prepared.json',dict(as_of=now(),input_sha256=sha(DEST/'input.jsonl.gz'),windows=statuses))

def run():
    check();p=json.loads((DEST/'prepared.json').read_text())
    if sha(DEST/'input.jsonl.gz')!=p['input_sha256']:raise ValueError('prepared input changed')
    subprocess.run([str(TARGET/f'release/examples/{NAME}'),str(DEST/'input.jsonl.gz'),str(DEST/'output.jsonl.gz')],check=True)
    write(DEST/'execution.json',dict(as_of=now(),input_sha256=p['input_sha256'],output_sha256=sha(DEST/'output.jsonl.gz'),binary_sha256=sha(TARGET/f'release/examples/{NAME}')))

def measure(window,arm):
    """Independent interval accounting, using only saved wants/order lifecycles."""
    es=window['events'];orders=arm['orders'];ds=[e for e in es if e['kind']==0]
    cuts={0.,60000.};cuts.update(e['t'] for e in ds)
    old=None
    for e in es:
        if e['net']!=old:cuts.add(e['t']);old=e['net']
    for o in orders:
        for k in ['sent','accepted','end','release']:
            if o[k] is not None and 0<=o[k]<=60000:cuts.add(o[k])
    ts=np.array(sorted(cuts));dtv=np.diff(ts);ts=ts[:-1];n=len(ts)
    idx=np.searchsorted([e['t'] for e in ds],ts,side='right')-1
    raw=np.array([[-1 if p is None else p for p,z in e['raw']] for e in ds]);want=np.full((n,4),-1);want[idx>=0]=raw[idx[idx>=0]]
    ni=np.searchsorted([e['t'] for e in es],ts,side='right')-1
    known=ni>=0;net=np.zeros(n);net[known]=np.array([e['net'] for e in es])[ni[known]]
    aligned=np.zeros((n,4));off=np.zeros((n,4));active=np.zeros((n,2));reserved=np.zeros((n,2));other=np.zeros(n);reducing=np.zeros(n)
    for o in orders:
        l=o['lane']%4;side=l%2
        live=(ts>=o['accepted']) & (ts<(o['end'] if o['end'] is not None else 60000.))
        held=(ts>=o['sent']) & (ts<(o['release'] if o['release'] is not None else 60000.))
        match=want[:,l]==o['tick'];aligned[:,l]+=live&match;off[:,l]+=live&~match
        active[:,side]+=live*o['size'];reserved[:,side]+=held*o['size']
        is_reduce=net*(1 if side==0 else -1)<0
        reducing+=live*~match*known*is_reduce*o['size'];other+=live*~match*known*~is_reduce*o['size']
    def total(x):return float(np.dot(x,dtv))
    def extra(q):return np.maximum(0,np.maximum(np.abs(net+q[:,0]),np.abs(net-q[:,1]))-np.abs(net))
    m=dict(aligned_lane_ms=total((aligned>0).sum(axis=1)),elsewhere_lane_ms=total((off>0).sum(axis=1)),
        desired_lane_ms=total((want>=0).sum(axis=1)),active_share_ms=total(active.sum(axis=1)),reserved_share_ms=total(reserved.sum(axis=1)),
        active_range_extra_share_ms=total(extra(active)*known),reserved_range_extra_share_ms=total(extra(reserved)*known),
        off_reducing_share_ms=total(reducing),off_other_share_ms=total(other),inventory_unobserved_ms=total(~known),
        peak_reserved_shares=float(reserved.max()),actions=len(arm['actions']),retained_actions=sum(x['status']=='retained' for x in arm['actions']),
        same_price_reopens=sum(x['reason']=='same_price_reopen' for x in arm['actions']),
        native_requests=len(arm['commands']),venue_requests=sum(c['sent'] for c in arm['commands']),retry_times=arm['retry_times'],**arm['counts'])
    m.pop('withheld',None)
    return m

def verify_arm(w,r):
    """Check source identity and lifecycle accounting; refuse silent exclusions."""
    expected=[];previous=OFF
    for i,e in enumerate(w['events']):
        if e['kind']!=0:continue
        for l in [0,1]:
            if previous[l][0] is not None and e['raw'][l][0] is None:expected.append((i,e['input_index'],e['ordinal'],l,e['t'],previous[l][0]))
        previous=e['raw']
    actual=[(q['index'],q['input_index'],q['ordinal'],q['lane'],q['start'],q['old_tick']) for q in r['actions']]
    if actual!=expected:raise ValueError('neutral action identities differ')
    orders={o['id']:o for o in r['orders']}
    if len(orders)!=len(r['orders']):raise ValueError('duplicate order')
    for c in r['commands']:
        if c['kind']=='place' and c['sent']:
            o=orders[c['id']]
            if [o[k] for k in ['sent','tick','size','lane']]!=[c[k] for k in ['t','tick','size','lane']]:raise ValueError('order/place mismatch')
        if c['kind']=='cancel' and c['sent']:
            o=orders[c['id']]
            end=max(c['t']+22,o['sent']+95) if c['t']<o['accepted'] else c['t']+22
            if o['cancel_sent']!=c['t'] or o['end']!=end or o['release']!=end+1000:raise ValueError('cancel lifecycle mismatch')
    for o in orders.values():
        if o['accepted']!=o['sent']+40:raise ValueError('accept lifecycle mismatch')
        reserved=sum(x['size'] for x in orders.values() if x['lane']%2==o['lane']%2 and (x['sent'],x['id'])<=(o['sent'],o['id']) and (x['release'] is None or x['release']>o['sent']))
        if reserved>150+1e-9:raise ValueError('external capacity exceeded')
    for q in r['actions']:
        if q['end'] is None or q['end']<q['start']:raise ValueError('unclosed action')
        if q['status']=='retained':
            e=w['events'][q['index']];o=orders[q['order_id']];l=q['lane'];v=e['v'][l]
            if r['arm']=='reference' or o['accepted']>q['start'] or o['size']>5 or o['tick']!=q['old_tick']:raise ValueError('invalid retained ticket')
            if e['held'] or e['binance'][l] or v is None or not .0492-.01<v<=.0492:raise ValueError('invalid retained guard')
            if e['relaxed'][l][0]!=o['tick'] or e['relaxed'][l][1]<o['size'] or e['raw'][l+2][0]==o['tick']:raise ValueError('invalid retained quote')
            duration=100 if r['arm']=='all100' else 50
            if q['deadline']!=q['start']+duration or q['end']>q['deadline']:raise ValueError('grace overrun')
            if q['reason']=='deadline' and (q['end']!=q['deadline'] or o['cancel_sent']!=q['deadline']):raise ValueError('late deadline cancellation')
    counts=r['counts'];cmd=r['commands']
    if counts['places']!=sum(c['kind']=='place' for c in cmd) or counts['cancels']!=sum(c['kind']=='cancel' for c in cmd):raise ValueError('request counts differ')
    if counts['throttled_locally']!=sum(c['kind']=='place' and not c['sent'] for c in cmd):raise ValueError('refusal counts differ')

def verify():
    d=check();execution=json.loads((DEST/'execution.json').read_text());prepared=json.loads((DEST/'prepared.json').read_text())
    for k in ['input','output']:
        if sha(DEST/f'{k}.jsonl.gz')!=execution[f'{k}_sha256']:raise ValueError('execution artifact changed')
    rows=[];seen=[]
    with gzip.open(DEST/'input.jsonl.gz','rt') as fi,gzip.open(DEST/'output.jsonl.gz','rt') as fo:
        for li in fi:
            w=json.loads(li);o=json.loads(next(fo));key=(w['run'],w['mid'])
            if key!=(o['run'],o['mid']) or w['start']!=o['start'] or len(w['events'])!=o['events']:raise ValueError('window identity mismatch')
            if [r['arm'] for r in o['arms']]!=ARMS:raise ValueError('arm population differs')
            seen.append(key)
            for r in o['arms']:
                verify_arm(w,r);rows.append(dict(run=key[0],mid=key[1],arm=r['arm'],**measure(w,r)))
        if next(fo,None) is not None:raise ValueError('extra output windows')
    if seen!=[(s['run'],s['mid']) for s in prepared['windows']]:raise ValueError('excluded/reordered window')
    frame=pd.DataFrame(rows);frame.to_parquet(DEST/'window_metrics.parquet',index=False)
    totals=frame.groupby('arm').sum(numeric_only=True).drop(columns=['mid','peak_reserved_shares']).to_dict('index')
    for r in totals.values():
        r['coverage_pct']=100*r['aligned_lane_ms']/r['desired_lane_ms']
    ref=totals['reference'];delta={arm:{k:100*(v[k]/ref[k]-1) if ref[k] else None for k in ['native_requests','venue_requests','aligned_lane_ms','elsewhere_lane_ms','active_share_ms','reserved_share_ms','active_range_extra_share_ms','reserved_range_extra_share_ms']} for arm,v in totals.items()}
    predicate={arm:totals[arm]['venue_requests']<ref['venue_requests'] and totals[arm]['aligned_lane_ms']>ref['aligned_lane_ms'] for arm in ARMS[1:]}
    receipt=dict(as_of=now(),declaration_sha256=sha(DEST/'declaration.json'),source_commit=d['source_commit'],build=d['build'],
        artifacts={str(p):sha(p) for p in [DEST/'input.jsonl.gz',DEST/'output.jsonl.gz',DEST/'prepared.json',DEST/'execution.json',DEST/'window_metrics.parquet']},
        population=dict(windows=len(seen),decision_rows=sum(s['decisions'] for s in prepared['windows']),neutral_actions_per_arm=sum(s['actions'] for s in prepared['windows']),excluded_windows=0,window_arm_rows=len(rows)),
        totals=totals,change_pct=delta,predicate_fewer_venue_requests_more_aligned=predicate,limitations=LIMITS)
    write(DEST/'receipt.json',receipt);write(REVIEW/'receipt.json',receipt)
    write(DEST/'verification.json',dict(as_of=now(),receipt_sha256=sha(DEST/'receipt.json'),window_arm_rows=len(rows),
        neutral_action_rows=sum(int(r['actions']) for r in rows),action_identity_exact=True,lifecycle_and_capacity_exact=True,coverage_rebuilt_from_saved_lifecycles=True))
    shutil.copyfile(DEST/'verification.json',REVIEW/'verification.json')
    print(json.dumps(dict(population=receipt['population'],totals=totals,change_pct=delta,predicates=predicate),indent=2))

if __name__=='__main__':
    {'build':build,'declare':declare,'prepare':prepare,'run':run,'verify':verify}[sys.argv[1]]()
