"""Synthetic-only B4 freeze and native differential validation pipeline."""
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from . import pf_native_a2_continuity as s
from . import pf_ladder_pairs_build as builder

NATIVE=Path('/tmp/pf-b4-parity-20261109')
DEST=s.DEST.parent/'paired_engine'
REVIEW=s.REVIEW.parent/'paired_engine'
NAME='native_pairs_oracle'
SOURCES=['pf_ladder_pairs.py','pf_ladder_pairs_build.py','pf_step4_ladder_pairs.py','tests/test_ladder_pairs.py',
         'native_pairs_oracle.rs','pf_pairs_validation.py','tests/test_simulation_causality.py','pf_native_a2_continuity.py']
ACCESSORS='''    pub fn diagnostic_pairs(&self) -> ([Option<(i32,f64)>;LANES],[f64;LANES]) { (self.pair_total,self.pair_filled) }
    pub fn diagnostic_retry_only(&mut self) { self.triggers.retain(|Reverse((_,trig))| matches!(trig,Trig::Retry{..})); }
    pub fn diagnostic_fill_decide(&self, t:f64) -> bool { self.triggers.iter().any(|Reverse((k,trig))| k.t==t && matches!(trig,Trig::Decide)) }
'''

def build():
    builder.main();DEST.mkdir(exist_ok=True)
    native_commit=s.command(['git','rev-parse','HEAD'],cwd=NATIVE)
    if not native_commit.startswith('f20fdb1'):raise ValueError('wrong native version')
    original=s.command(['git','show',f'HEAD:{s.REL}'],cwd=NATIVE)+'\n'
    instrumented=s.instrument(original)
    anchor='    pub fn ws_ms(&self) -> f64 {'
    if instrumented.count(anchor)!=1:raise ValueError('native accessor anchor changed')
    instrumented=instrumented.replace(anchor,ACCESSORS+anchor)
    path=NATIVE/s.REL
    if path.read_text() not in [original,instrumented]:raise ValueError('unrelated native edits')
    path.write_text(instrumented)
    example=NATIVE/f'crates/pf-engine/examples/{NAME}.rs';example.parent.mkdir(exist_ok=True);shutil.copyfile(s.a.HERE/f'{NAME}.rs',example)
    command=['cargo','test','--locked','--offline','-p','pf-engine','--example',NAME,'--target-dir',str(s.TARGET),'-j','1']
    subprocess.run(command,cwd=NATIVE,check=True)
    command[1]='build';command.append('--release');subprocess.run(command,cwd=NATIVE,check=True)
    s.write(DEST/'build.json',dict(as_of=s.now(),native_commit=native_commit,native_instrument_sha256=s.sha(path),
        native_original_sha256=hashlib.sha256(original.encode()).hexdigest(),oracle_source_sha256=s.sha(example),
        binary_sha256=s.sha(s.TARGET/f'release/examples/{NAME}'),frozen_source_sha256=s.sha(builder.SRC),generated_engine_sha256=s.sha(builder.DST)))

def declare():
    d=dict(as_of=s.now(),source_commit=s.command(['git','rev-parse','HEAD'],cwd=s.a.HERE),sources={p:s.sha(s.a.HERE/p) for p in SOURCES},
        build=json.loads((DEST/'build.json').read_text()),frozen_v3=builder.FROZEN,
        design='Implementation validation only: generated LZ copies incremental_v1 logical pairs. No-token full returned outputs byte-exact to frozen v3; fixed-size paired/nonpaired full outputs exact; variable-v3 resize positive and differing outputs; synthetic native differential traces compare each physical request, pair totals/fills, and full-fill decision obligations.',
        population='Predetermined32 explicit differential scenarios plus200 seeded synthetic scenarios, all generated without reading market data; unit test matrix separately. No exclusions permitted.',
        seed=20261109,random_scenarios=200,
        controls='No-token and fixed-size positive parity; variable resize must fire and differ; partial/full/late fill, front rotation, pending retry, minimum-value/rounding, concurrent cancel/create, chronological paired fill order, unsupported modes, read-only tracing, corrupt native request and pair-state refusal.',
        null='Deterministic software parity, not a trading experiment or statistical null. No calibrated fills, profit, inventory variance or skill inference.',
        scope='Pair mapping, logical rung identity, known fill accounting and the research split request handler. Synthetic market trades only exercise lifecycle. Native oracle consumes generated quote decisions/feedback; does not independently recompute model predictions or prove live scheduling/venue behavior.',
        limitations=['Current historical fill model remains uncalibrated. No LZ economic promotion.',
            'No-token parity is checked on a declared synthetic matrix, not a fresh historical/protected replay.',
            'Native kernel/local refusal, wallet, venue throttles across markets and reservation1s deployment integration are outside this mapper validation.',
            'Frozen simulator fill/venue assumptions remain unless required for paired FIFO; their simulated PnL is never reported as evidence.'],
        no_market_data_read=True,no_protected_dates_read=True,no_live_change=True,no_fill_clock_bypass=True,
        multiplicity=dict(policy_settings_added=0,implementation_token='LZ',cumulative_strategy_settings=21))
    s.write(DEST/'declaration.json',d);s.write(REVIEW/'declaration.json',d)

def check():
    d=json.loads((DEST/'declaration.json').read_text())
    for p,h in d['sources'].items():
        if s.sha(s.a.HERE/p)!=h:raise ValueError('paired source changed: '+p)
    if builder.generate()!=builder.DST.read_text():raise ValueError('generated engine differs')
    if s.sha(builder.SRC)!=builder.FROZEN:raise ValueError('frozen engine changed')
    if s.sha(NATIVE/s.REL)!=d['build']['native_instrument_sha256'] or s.sha(s.TARGET/f'release/examples/{NAME}')!=d['build']['binary_sha256']:raise ValueError('native oracle changed')
    if s.command(['git','diff','--name-only'],cwd=NATIVE)!=s.REL:raise ValueError('unrelated native changes')
    return d

def cases():
    from .tests.test_ladder_pairs import scenario,frame,leg
    out=[]
    # Explicit paired paths on both sides, repeated with/without a rung.
    designs=[('grow_shrink',((0,5),(100,25),(200,5)),()),
             ('whole_shrink',((0,25),(100,5)),()),
             ('pending',((0,5),(10,25),(20,5),(30,25)),()),
             ('partial',((0,5),(100,25)),((150,2),(180,3))),
             ('full',((0,5),(100,25)),((150,25),)),
             ('late_cancelled',((0,5),(100,25),(150,0),(170,25)),((160,25),)),
             ('front_rotate',((0,5),(100,25),(400,5),(500,25)),((150,5),)),
             ('min_remainder',((0,5),(200,25),(300,5)),((60,1),))]
    for side in [0,1]:
        for ladder in [False,True]:
            for name,changes,trades in designs:
                trace=[];extra={'XV','XMIN1'}|({'LD2'} if ladder else set())
                scenario(side,changes,tuple(leg(t,side,48 if side==0 else 54,q) for t,q in trades),extra,trace=trace)
                out.append(dict(name=f'{name}/{side}/{ladder}',events=trace,minimum=1.,rpm=500.))
    rng=random.Random(20261109)
    for i in range(200):
        side=i%2;ladder=bool(i%3);changes=[(float(t),rng.choice([0,5,5,25,25])) for t in range(0,1000,25)]
        changes[0]=(0.,rng.choice([5,25]))
        ft=list(range(0,1000,155));frames=[]
        for t in ft:
            bid=50+rng.choice([-2,-1,0,1,2]);f=frame(t-1,bid,bid+2,rng.choice([0.,5.,20.]),rng.choice([0.,5.,20.]))
            f.update(b2_p=bid-1,a2_p=bid+3,b2_s=0.,a2_s=0.);frames.append(f)
        trades=[leg(t,side,45 if side==0 else 57,rng.choice([1,2,5,10,20,25,50])) for t in range(65,1000,90)]
        trace=[];extra={'XV','XMIN1'}|({'LD2'} if ladder else set())
        scenario(side,changes,trades,extra,frames=frames,known_times=ft,trace=trace,delay=rng.choice([20.,100.,250.]))
        out.append(dict(name=f'random/{i}',events=trace,minimum=1.,rpm=500.))
    if len(out)!=232:raise ValueError('synthetic population changed')
    return out

def traces():
    check()
    with (DEST/'synthetic.jsonl').open('x') as f:
        for case in cases():f.write(json.dumps(case,allow_nan=False,separators=(',',':'))+'\n')
    s.write(DEST/'synthetic.json',dict(as_of=s.now(),cases=232,sha256=s.sha(DEST/'synthetic.jsonl'),market_data_read=False))

def oracle():
    check();declared=json.loads((DEST/'synthetic.json').read_text())
    if s.sha(DEST/'synthetic.jsonl')!=declared['sha256']:raise ValueError('synthetic input changed')
    subprocess.run([str(s.TARGET/f'release/examples/{NAME}'),str(DEST/'synthetic.jsonl'),str(DEST/'native.jsonl')],check=True)

if __name__=='__main__':
    {'build':build,'declare':declare,'traces':traces,'oracle':oracle}[sys.argv[1]]()
