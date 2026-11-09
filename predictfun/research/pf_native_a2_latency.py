"""Declared latency sensitivity of the pinned A2 quote-continuity experiment.

Reuses the bounded cohort and frozen policy. Only assumed venue feedback clocks
change. Base output must be byte-identical before either stress case can run.
"""
import copy
import gzip
import hashlib
import json
import re
import subprocess
import sys
from . import pf_native_a2_continuity as s

DEST=s.DEST.parent/'native_latency'
REVIEW=s.REVIEW.parent/'native_latency'
NAME='native_a2_latency'
LATENCIES=dict(base=[40.,22.,95.],faster=[20.,10.,50.],slower=[80.,50.,160.])
SOURCES=['pf_native_a2_latency.py','tests/test_native_a2_latency.py']

LATENCY_SOURCE='''
#[derive(Clone, Copy, Debug, PartialEq)]
struct Latency { place:f64, cancel:f64, pending:f64 }
fn parse_latency(name:&str) -> Result<Latency, &'static str> {
    let (place,cancel,pending)=match name {
        "base" => (40.,22.,95.), "faster" => (20.,10.,50.), "slower" => (80.,50.,160.),
        _ => return Err("unknown latency profile"),
    };
    Ok(Latency{place,cancel,pending})
}
'''
LATENCY_TESTS='''
    #[test] fn latency_profiles_change_both_pending_and_accepted_feedback() {
        for name in ["base","faster","slower"] {
            let l=parse_latency(name).unwrap();
            for pull in [5.,l.place+20.] {
                let out=Sim::with_latency("reference",7,l).run(&[event(0.,true),event(pull,false)]);
                let order=&out["orders"][0];
                let end=if pull<l.place {(pull+l.cancel).max(l.pending)} else {pull+l.cancel};
                assert_eq!(order["accepted"],l.place);assert_eq!(order["end"],end);assert_eq!(order["release"],end+1000.);
            }
        }
    }
    #[test] fn unknown_latency_refused_and_later_reopen_stays_late() {
        assert!(parse_latency("unknown").is_err());
        for name in ["base","faster","slower"] {
            let out=Sim::with_latency("update_first50",7,parse_latency(name).unwrap()).run(&[event(0.,true),event(100.,false),event(151.,true)]);
            assert_eq!(out["actions"][0]["reason"],"deadline");assert_eq!(out["actions"][0]["end"],150.);
        }
    }
'''

def generate(source):
    """Mechanical extension of the frozen driver; refuse drifted anchors."""
    changes=[
        ('const END: f64 = 60000.;','const END: f64 = 60000.;\n'+LATENCY_SOURCE),
        ("w: LadderWindow, arm: &'static str, previous: [Want; 4],","w: LadderWindow, arm: &'static str, latency: Latency, previous: [Want; 4],"),
        ("    fn new(arm: &'static str, mid: u64) -> Self {","    #[cfg(test)]\n    fn new(arm: &'static str, mid: u64) -> Self {Self::with_latency(arm,mid,parse_latency(\"base\").unwrap())}\n    fn with_latency(arm: &'static str, mid: u64, latency: Latency) -> Self {"),
        ('Self { w, arm, previous: OFF,','Self { w, arm, latency, previous: OFF,'),
        ('accepted:t+40.,','accepted:t+self.latency.place,'),
        ('Feedback { t:t+40., id, kind:0 }','Feedback { t:t+self.latency.place, id, kind:0 }'),
        ('(t+22.).max(o.sent+95.) } else { t+22. }','(t+self.latency.cancel).max(o.sent+self.latency.pending) } else { t+self.latency.cancel }'),
        ('assert_eq!(args.len(),3);','assert_eq!(args.len(),4);let latency=parse_latency(&args[3]).unwrap();'),
        ('Sim::new(arm,w.mid).run(&w.events)','Sim::with_latency(arm,w.mid,latency).run(&w.events)'),
        ('    #[test] fn whole_first_and_incremental_keep() {',LATENCY_TESTS+'    #[test] fn whole_first_and_incremental_keep() {')]
    result=source
    for before,after in changes:
        if result.count(before)!=1:raise ValueError('native driver anchor changed: '+before)
        result=result.replace(before,after,1)
    return result

def build():
    s.check();DEST.mkdir(exist_ok=True)
    source=generate((s.a.HERE/'native_a2_continuity.rs').read_text())
    path=s.ROOT/f'crates/pf-engine/examples/{NAME}.rs';path.write_text(source)
    command=['cargo','test','--locked','--offline','-p','pf-engine','--example',NAME,'--target-dir',str(s.TARGET),'-j','1']
    subprocess.run(command,cwd=s.ROOT,check=True)
    command[1]='build';command.append('--release');subprocess.run(command,cwd=s.ROOT,check=True)
    s.write(DEST/'build.json',dict(as_of=s.now(),native_build=s.sha(s.DEST/'build.json'),
        generated_sha256=s.sha(path),binary_sha256=s.sha(s.TARGET/f'release/examples/{NAME}')))

def declare():
    s.check();prior=json.loads((s.DEST/'receipt.json').read_text())
    files=[s.DEST/k for k in ['declaration.json','build.json','input.jsonl.gz','output.jsonl.gz','receipt.json','verification.json','window_metrics.parquet','prepared.json']]
    files += [DEST/'build.json',s.REVIEW.parent/'findings_v3.json']
    d=dict(as_of=s.now(),source_commit=s.command(['git','rev-parse','HEAD'],cwd=s.a.HERE),
        sources={p:s.sha(s.a.HERE/p) for p in SOURCES+s.SOURCES},inputs={str(p):s.sha(p) for p in files},
        population=prior['population'],arms=s.ARMS,latencies=LATENCIES,
        hypothesis='The existing50ms updates-first policy reduces venue requests and increases original-price coverage, while adding less off-target time than100ms, under both earlier declared latency stress cases. Report exposure costs regardless of predicate.',
        design='Each latency has its own reference, all100 and update_first50. No parameter selection. Base output must reproduce native_a2_v2 byte-for-byte before other profiles run.',
        null='Deterministic paired mechanics; no statistical null or draw-based claim, no independent validation, PnL, variance or confidence intervals.',
        timing='Only create/answer, accepted-cancel and pending-create-cancel assumptions change. Native retries, tokens, external cap150shares and1000ms reservation grace unchanged. Events/guards/targets remain factual and fixed.',
        controls='All prior adapter tests plus each profile pending/accepted lifecycle, unknown profile refusal, late reopen refusal; generated-anchor refusal; lifecycle corruption and action identity refusal; byte-exact base output and metric parity.',
        multiplicity=dict(new_policy_parameters=0,cumulative_distinct_policies=21,existing_policies=3,latency_profiles=3,new_latency_profiles_for_native_model=2),
        limitations=[z for z in s.LIMITS if not z.startswith('Venue assumed')]+['Latency profiles are assumed stresses, not estimated venue quantiles or independent days.'],
        no_protected_dates_read=True,no_live_change=True,no_fill_clock_bypass=True)
    s.write(DEST/'declaration.json',d);s.write(REVIEW/'declaration.json',d)

def check():
    s.check();d=json.loads((DEST/'declaration.json').read_text())
    for p,h in d['sources'].items():
        if s.sha(s.a.HERE/p)!=h:raise ValueError('latency source changed')
    for p,h in d['inputs'].items():
        if s.sha(p)!=h:raise ValueError('latency input changed')
    b=json.loads((DEST/'build.json').read_text())
    if s.sha(s.TARGET/f'release/examples/{NAME}')!=b['binary_sha256']:raise ValueError('latency binary changed')
    generated=generate((s.a.HERE/'native_a2_continuity.rs').read_text())
    if hashlib.sha256(generated.encode()).hexdigest()!=b['generated_sha256']:raise ValueError('generated driver mismatch')
    return d

def exact_base(candidate,reference):
    if s.sha(candidate)!=s.sha(reference):raise ValueError('base output is not byte-identical')

def run():
    check();records={}
    for profile in LATENCIES:
        dest=DEST/f'{profile}.jsonl.gz'
        subprocess.run([str(s.TARGET/f'release/examples/{NAME}'),str(s.DEST/'input.jsonl.gz'),str(dest),profile],check=True)
        if profile=='base':exact_base(dest,s.DEST/'output.jsonl.gz')
        records[profile]=dict(output_sha256=s.sha(dest),base_byte_exact=(profile=='base'))
        print('finished',profile,flush=True)
    s.write(DEST/'execution.json',dict(as_of=s.now(),profiles=records))

def verify_arm(w,r,profile):
    place,cancel,pending=LATENCIES[profile]
    expected=[];previous=s.OFF
    for i,e in enumerate(w['events']):
        if e['kind']!=0:continue
        for l in [0,1]:
            if previous[l][0] is not None and e['raw'][l][0] is None:expected.append((i,e['input_index'],e['ordinal'],l,e['t'],previous[l][0]))
        previous=e['raw']
    actual=[(q['index'],q['input_index'],q['ordinal'],q['lane'],q['start'],q['old_tick']) for q in r['actions']]
    if actual!=expected:raise ValueError('neutral action identities differ')
    orders={o['id']:o for o in r['orders']};commands=r['commands']
    if len(orders)!=len(r['orders']):raise ValueError('duplicate order')
    places=[c for c in commands if c['kind']=='place' and c['sent']]
    cancels=[c for c in commands if c['kind']=='cancel' and c['sent']]
    if set(orders)!={c['id'] for c in places} or len(orders)!=len(places):raise ValueError('order/place population mismatch')
    if len({c['id'] for c in cancels})!=len(cancels):raise ValueError('duplicate cancel')
    if {o['id'] for o in orders.values() if o['end'] is not None}!={c['id'] for c in cancels}:raise ValueError('order/cancel population mismatch')
    for c in places:
        o=orders[c['id']]
        if [o[k] for k in ['sent','tick','size','lane']]!=[c[k] for k in ['t','tick','size','lane']]:raise ValueError('order/place mismatch')
    for c in cancels:
        o=orders[c['id']];end=max(c['t']+cancel,o['sent']+pending) if c['t']<o['accepted'] else c['t']+cancel
        if o['cancel_sent']!=c['t'] or o['end']!=end or o['release']!=end+1000:raise ValueError('cancel lifecycle mismatch')
    for o in orders.values():
        if o['accepted']!=o['sent']+place:raise ValueError('accept lifecycle mismatch')
        reserved=sum(x['size'] for x in orders.values() if x['lane']%2==o['lane']%2 and (x['sent'],x['id'])<=(o['sent'],o['id']) and (x['release'] is None or x['release']>o['sent']))
        if reserved>150+1e-9:raise ValueError('external capacity exceeded')
    for q in r['actions']:
        if q['end'] is None or q['end']<q['start']:raise ValueError('unclosed action')
        if q['status']!='retained':continue
        e=w['events'][q['index']];o=orders[q['order_id']];l=q['lane'];v=e['v'][l]
        if r['arm']=='reference' or o['accepted']>q['start'] or o['size']>5 or o['tick']!=q['old_tick']:raise ValueError('invalid retained ticket')
        if e['held'] or e['binance'][l] or v is None or not .0492-.01<v<=.0492:raise ValueError('invalid retained guard')
        if e['relaxed'][l][0]!=o['tick'] or e['relaxed'][l][1]<o['size'] or e['raw'][l+2][0]==o['tick']:raise ValueError('invalid retained quote')
        duration=100 if r['arm']=='all100' else 50
        if q['deadline']!=q['start']+duration or q['end']>q['deadline']:raise ValueError('grace overrun')
        if q['reason']=='deadline' and (q['end']!=q['deadline'] or o['cancel_sent']!=q['deadline']):raise ValueError('late deadline cancellation')
    counts=r['counts']
    if counts['places']!=sum(c['kind']=='place' for c in commands) or counts['cancels']!=sum(c['kind']=='cancel' for c in commands):raise ValueError('request counts differ')
    if counts['throttled_locally']!=sum(c['kind']=='place' and not c['sent'] for c in commands):raise ValueError('refusal counts differ')

def summarize(frame):
    totals=frame.groupby('arm').sum(numeric_only=True).drop(columns=['mid','peak_reserved_shares']).to_dict('index')
    ref=totals['reference'];deltas={}
    keys=['venue_requests','native_requests','aligned_lane_ms','elsewhere_lane_ms','active_share_ms','reserved_share_ms','active_range_extra_share_ms']
    for arm,q in totals.items():
        q['coverage_pct']=100*q['aligned_lane_ms']/q['desired_lane_ms']
        deltas[arm]={k:100*(q[k]/ref[k]-1) if ref[k] else None for k in keys}
    short=totals['update_first50'];long=totals['all100']
    return dict(totals=totals,change_pct=deltas,
        fewer_requests_more_aligned_less_off_than100=short['venue_requests']<ref['venue_requests'] and short['aligned_lane_ms']>ref['aligned_lane_ms'] and short['elsewhere_lane_ms']<long['elsewhere_lane_ms'],
        active_exposure_lower_than_reference=short['active_share_ms']<ref['active_share_ms'])

def verify():
    d=check();execution=json.loads((DEST/'execution.json').read_text());prepared=json.loads((s.DEST/'prepared.json').read_text())
    exact_base(DEST/'base.jsonl.gz',s.DEST/'output.jsonl.gz');rows=[]
    for profile in LATENCIES:
        if s.sha(DEST/f'{profile}.jsonl.gz')!=execution['profiles'][profile]['output_sha256']:raise ValueError('execution changed')
        seen=[]
        with gzip.open(s.DEST/'input.jsonl.gz','rt') as fi,gzip.open(DEST/f'{profile}.jsonl.gz','rt') as fo:
            for li in fi:
                w=json.loads(li);o=json.loads(next(fo));key=(w['run'],w['mid'])
                if key!=(o['run'],o['mid']) or w['start']!=o['start'] or len(w['events'])!=o['events']:raise ValueError('window identity mismatch')
                if [r['arm'] for r in o['arms']]!=s.ARMS:raise ValueError('arm population differs')
                seen.append(key)
                for r in o['arms']:
                    verify_arm(w,r,profile);rows.append(dict(profile=profile,run=key[0],mid=key[1],arm=r['arm'],**s.measure(w,r)))
            if next(fo,None) is not None:raise ValueError('extra output windows')
        if seen!=[(q['run'],q['mid']) for q in prepared['windows']]:raise ValueError('excluded/reordered windows')
        print('verified',profile,flush=True)
    frame=s.pd.DataFrame(rows);base=frame[frame.profile=='base'].drop(columns='profile').reset_index(drop=True)
    old=s.pd.read_parquet(s.DEST/'window_metrics.parquet');s.pd.testing.assert_frame_equal(base,old,check_exact=True)
    frame.to_parquet(DEST/'window_metrics.parquet',index=False)
    summary={profile:summarize(g.drop(columns='profile')) for profile,g in frame.groupby('profile')}
    by_run={f'{p}/{run}':summarize(g.drop(columns='profile')) for (p,run),g in frame.groupby(['profile','run'])}
    controls=(DEST/'logs/controls.log').read_text();build=(DEST/'logs/build.log').read_text()
    if 'Ran 4 tests' not in controls or '\nOK\n' not in controls:raise ValueError('Python controls absent')
    if not re.search(r'test result: ok\. 9 passed; 0 failed',build):raise ValueError('native adapter controls absent')
    receipt=dict(as_of=s.now(),declaration_sha256=s.sha(DEST/'declaration.json'),source_commit=d['source_commit'],
        population=dict(windows_per_profile=220,decisions_per_profile=279059,neutral_actions_per_arm_profile=13077,excluded_windows=0,window_arm_profile_rows=len(rows)),
        summary=summary,by_run=by_run,base_output_byte_identical=True,base_metrics_exact=True,
        all_profiles_primary_predicate=all(q['fewer_requests_more_aligned_less_off_than100'] for q in summary.values()),
        all_run_profiles_primary_predicate=all(q['fewer_requests_more_aligned_less_off_than100'] for q in by_run.values()),
        controls=dict(native_adapter=9,python=4),limitations=d['limitations'],
        artifacts={str(p):s.sha(p) for p in [DEST/f'{p}.jsonl.gz' for p in LATENCIES]+[DEST/'window_metrics.parquet',DEST/'execution.json',DEST/'build.json',DEST/'logs/controls.log',DEST/'logs/build.log']})
    s.write(DEST/'receipt.json',receipt);s.write(REVIEW/'receipt.json',receipt)
    verification=dict(as_of=s.now(),receipt_sha256=s.sha(DEST/'receipt.json'),window_arm_profile_rows=len(rows),action_arm_profile_rows=sum(int(r['actions']) for r in rows),
        base_output_byte_identical=True,base_metrics_exact=True,action_lifecycle_capacity_verified=True,coverage_rebuilt_from_saved_lifecycles=True)
    s.write(DEST/'verification.json',verification);s.write(REVIEW/'verification.json',verification)
    print(json.dumps(dict(summary={p:q['change_pct']['update_first50'] for p,q in summary.items()},all_profiles_primary_predicate=receipt['all_profiles_primary_predicate']),indent=2))

if __name__=='__main__':
    {'build':build,'declare':declare,'run':run,'verify':verify}[sys.argv[1]]()
