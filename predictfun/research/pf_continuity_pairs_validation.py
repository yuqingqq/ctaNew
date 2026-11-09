"""Declared synthetic-only lifecycle integration of LC50 and LZ."""
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

from . import pf_pairs_validation as parent
from . import pf_ladder_continuity_build as builder

s = parent.s
DEST = parent.DEST.parent/'paired_continuity'
REVIEW = parent.REVIEW.parent/'paired_continuity'
NATIVE = Path('/tmp/pf-continuity-parity-20261109')
NAME = 'native_continuity_pairs_oracle'
SOURCES = ['pf_ladder_continuity_build.py', 'pf_step4_ladder_continuity.py',
           'tests/test_ladder_continuity.py', 'native_continuity_pairs_oracle.rs', 'pf_continuity_pairs_validation.py']


def build():
    parent.check(); builder.main(); DEST.mkdir(exist_ok=True)
    commit = s.command(['git', 'rev-parse', 'HEAD'], cwd=NATIVE)
    if not commit.startswith('f20fdb1'): raise ValueError('wrong native version')
    original = s.command(['git', 'show', f'HEAD:{s.REL}'], cwd=NATIVE)+'\n'
    instrumented = s.instrument(original)
    anchor = '    pub fn ws_ms(&self) -> f64 {'
    if instrumented.count(anchor) != 1: raise ValueError('native accessor anchor changed')
    instrumented = instrumented.replace(anchor, parent.ACCESSORS+anchor)
    path = NATIVE/s.REL
    if path.read_text() not in (original, instrumented): raise ValueError('unrelated native edit')
    path.write_text(instrumented)
    example = NATIVE/f'crates/pf-engine/examples/{NAME}.rs'
    shutil.copyfile(s.a.HERE/f'{NAME}.rs', example)
    command = ['cargo', 'test', '--locked', '--offline', '-p', 'pf-engine', '--example', NAME, '--target-dir', str(s.TARGET), '-j', '1']
    subprocess.run(command, cwd=NATIVE, check=True)
    command[1] = 'build'; command.append('--release')
    subprocess.run(command, cwd=NATIVE, check=True)
    s.write(DEST/'build.json', dict(as_of=s.now(), native_commit=commit, native_instrument_sha256=s.sha(path),
                                  binary_sha256=s.sha(s.TARGET/f'release/examples/{NAME}'), oracle_source_sha256=s.sha(example)))


def declare():
    parent.check()
    declaration = dict(as_of=s.now(), source_commit=s.command(['git', 'rev-parse', 'HEAD'], cwd=s.a.HERE),
        sources={p:s.sha(s.a.HERE/p) for p in SOURCES}, build=json.loads((DEST/'build.json').read_text()),
        parent_declaration_sha256=s.sha(parent.DEST/'declaration.json'), paired_parent_sha256=builder.PARENT,
        design='LC50: retain one accepted sent-size<=5 touch member for at most50ms on a PM90-only soft exit, score>line-0.01. Read all same-clock decisions before expiry. No new orders or replenishment while the guard is off. Preserve unchanged fill budget; if a larger logical pair shrinks, target only retained known remainder. Prediction probes and fill notices can cancel, never initiate quotes. Hard guard, changed price/size, full member fill, RO/stop cancel immediately. No timer extension or hold after RO.',
        population=dict(explicit_synthetic_traces=28, seeded_synthetic_traces=200, excluded_traces=0, python_checks=33),
        seed=2026110950, null='Deterministic engineering assertions and differential replay only; no statistical or economic null.',
        controls=['Exact no-LC parent outputs and native traces', 'Same-clock150ms recovery retains;151ms recovery cancels150ms',
                  'Partial/full fills and no off-guard topups', 'CAP10 from cancelled-order fill, RO60/stop, lower-line probe and price-change preemption',
                  'Stale timer must not end a later hold', 'Unsupported rules refused', 'Native corrupt request and pair-state refused',
                  'Saved lifecycle, count and identity corruptions refused'],
        scope='Generated research simulation with synthetic trades. Native oracle checks effective desired quotes, cancellation-only pulls and feedback lifecycle; it does not independently implement LC50 policy or desired guards.',
        limitations=['No historical/forward data or PnL/variance result.', 'Venue/fill/queue model remains uncalibrated.',
                     'All equal-clock events are available in this synthetic scheduler; live timer behavior remains unverified.',
                     'Only the declared MICRO/AM paired rule family is supported. Deployment kernel/wallet/cross-market limits are outside this study.'],
        multiplicity=dict(policy_settings_added=0,cumulative_strategy_settings=21,engineering_token='LC50'),
        no_market_data_read=True,no_protected_dates_read=True,no_live_change=True)
    s.write(DEST/'declaration.json',declaration); s.write(REVIEW/'declaration.json',declaration)


def check():
    parent.check()
    d = json.loads((DEST/'declaration.json').read_text())
    for p,h in d['sources'].items():
        if s.sha(s.a.HERE/p) != h: raise ValueError('source changed: '+p)
    if builder.generate() != builder.DST.read_text(): raise ValueError('generated source differs')
    if s.sha(NATIVE/s.REL) != d['build']['native_instrument_sha256'] or s.sha(s.TARGET/f'release/examples/{NAME}') != d['build']['binary_sha256']:
        raise ValueError('native instrument changed')
    return d


def controls():
    check()
    subprocess.run([sys.executable,'-m','pytest','-q',str(s.a.HERE/'tests/test_ladder_continuity.py'),'--tb=short'],check=True)


def cases():
    from .tests.test_ladder_continuity import case, late_cap_case, frame
    designs = [
        ('reopen140',dict(changes=((0.,.055),(100.,.045),(140.,.055)))),
        ('reopen150',dict()),
        ('reopen151',dict(changes=((0.,.055),(100.,.045),(151.,.055)))),
        ('deadline',dict(changes=((0.,.055),(100.,.045)))),
        ('partial',dict(trades=((60.,45,2.),(120.,45,1.)))),
        ('pair_shrink',dict(changes=((0.,.055),(50.,.1),(100.,.045),(151.,.055)),trades=((80.,45,2.),))),
        ('full',dict(changes=((0.,.055),(100.,.045),(200.,.055)),trades=((110.,45,5.),))),
        ('lower_line',dict(changes=((0.,.055),(100.,.045),(125.,.02),(200.,.055)))),
        ('pending',dict(changes=((0.,.055),(20.,.045)))),
        ('ro',dict(changes=((0.,-1.),(59900.,.055),(59975.,.045)))),
        ('stop',dict(changes=((0.,-1.),(900.,.055),(975.,.045)),stop=1)),
        ('second_hold',dict(changes=((0.,.055),(100.,.045),(120.,.055),(130.,.045),(200.,.055)))),
    ]
    frames=[]
    for t,bid in [(-1.,50),(124.,49)]:
        f=frame(t,bid,bid+2,100.,100.);f.update(b2_p=bid-1,a2_p=bid+3,b2_s=100.,a2_s=100.);frames.append(f)
    designs.append(('reprice',dict(frames=frames,known_times=[0.,125.])))
    for side in (0,1):
        for name,args in designs:
            yield f'{name}/{side}',case(side,**args)
        yield f'late_cap/{side}',late_cap_case(side)
    rng=random.Random(2026110950)
    for index in range(200):
        changes=[(float(t),rng.choice([.055,.055,.1,.045,.045,.02,-1.])) for t in range(0,1000,25)]
        changes[0]=(0.,.055)
        trades=[(float(t),45,rng.choice([1.,2.,5.,10.,25.])) for t in range(65,1000,90)]
        yield f'random/{index}',case(index%2,changes=changes,trades=trades,delay=rng.choice([20.,100.,250.]))


def run_cases():
    check()
    identities=[]
    with (DEST/'synthetic.jsonl').open('x') as traces,(DEST/'lifecycle.jsonl').open('x') as lifecycle:
        for name,(result,events,holds) in cases():
            identities.append(name)
            traces.write(json.dumps(dict(name=name,events=events,minimum=1.,rpm=500.),separators=(',',':'))+'\n')
            lifecycle.write(json.dumps(dict(name=name,holds=holds,output_sha256=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()),separators=(',',':'))+'\n')
    if len(identities)!=228:raise ValueError('wrong synthetic population')
    s.write(DEST/'manifest.json',dict(as_of=s.now(),identities=identities,trace_sha256=s.sha(DEST/'synthetic.jsonl'),lifecycle_sha256=s.sha(DEST/'lifecycle.jsonl')))
    subprocess.run([str(s.TARGET/f'release/examples/{NAME}'),str(DEST/'synthetic.jsonl'),str(DEST/'native.jsonl')],check=True)


if __name__=='__main__':
    {'build':build,'declare':declare,'controls':controls,'run':run_cases}[sys.argv[1]]()
