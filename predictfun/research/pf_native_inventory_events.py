"""Supplement the native observer at exact Filled knowledge events."""
import datetime as dt
import json
import subprocess
import sys
from . import pf_native_grid as g
from .pf_native_guard_verify import flatten_file
from .pf_native_grid_verify import check_scope

n=g.n;a=n.a
DEST=a.ROOT/'data/predict_fun/research/optimization_loop_20261109/native_inventory'
REVIEW=a.HERE.parent/'reviews/optimization_loop_20261109/native_inventory'
NAME='native_inventory_events'
SOURCES=['pf_native_inventory_events.py','pf_native_grid.py','native_guard_evidence.rs','pf_native_guard_verify.py']


def instrument():
    x=g.instrument((a.HERE/'native_guard_evidence.rs').read_text())
    x=g.once(x,'let mut probes=output(&out.join("probes.jsonl.gz"));let mut nprobe=0u64;let mut advances=0u64;',
        'let mut probes=output(&out.join("probes.jsonl.gz"));let mut nprobe=0u64;let mut advances=0u64;\n    let mut inventory=output(&out.join("inventory.jsonl.gz"));let mut ninventory=0u64;')
    x=g.once(x,'        if !synthetic {break;}',
        '''        if let CoreInput::Filled{market,known_ms,..}=&i {
            if let Some(ws)=windows.get(market) {
                let only=BTreeMap::from([(*market,*ws)]);
                ninventory+=probe(&core,&cfg,*known_ms,&only,&mut inventory,inputs);
            }
        }
        if !synthetic {break;}''')
    x=g.once(x,'probes.finish().unwrap();','probes.finish().unwrap();inventory.finish().unwrap();')
    x=g.once(x,'"synthetic_advances":advances,','"synthetic_advances":advances,"inventory_events":ninventory,')
    return x


def declare():
    if DEST.exists():raise ValueError('inventory observer exists')
    (DEST/'logs').mkdir(parents=True)
    b=json.loads((g.DEST/'build.json').read_text())
    if a.sha(n.NATIVE/g.REL)!=b['accessor_source_sha256']:raise ValueError('native accessor differs')
    d=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.HERE,text=True).strip(),
        sources={p:a.sha(a.HERE/p) for p in SOURCES},native_commit=n.NATIVE_COMMIT,native_accessor_sha256=b['accessor_source_sha256'],
        scope='same bounded Nov8 factual inputs; export read-only state after each Filled knowledge input at0-65s',
        motivation='partial fills change known position without necessarily causing a decision; inventory-dependent exits need this event clock',
        design='unchanged core and existing grid instrumentation; require exact original outputs and byte-identical states/books/grid-probes; extra observations only, no changed OMS feedback',
        null='deterministic identity comparison; positive exact prior stream, corrupted source anchor/input hashes refused',
        no_live_change=True,no_protected_dates_read=True,no_fill_clock_bypass=True)
    a.write(DEST/'declaration.json',d);a.write(REVIEW/'declaration.json',d)


def check():
    d=json.loads((DEST/'declaration.json').read_text())
    for p,h in d['sources'].items():
        if a.sha(a.HERE/p)!=h:raise ValueError('source differs')
    if a.sha(n.NATIVE/g.REL)!=d['native_accessor_sha256']:raise ValueError('native source differs')


def build():
    check();path=n.NATIVE/f'crates/pf-engine/examples/{NAME}.rs'
    if path.exists():raise ValueError('example exists')
    try:g.once('known','missing','new')
    except ValueError:refused=True
    else:raise ValueError('bad anchor accepted')
    path.write_text(instrument())
    cmd=['cargo','test','--locked','--offline','-p','pf-engine','--example',NAME,'--target-dir',str(n.TARGET),'-j','1']
    subprocess.run(cmd,cwd=n.NATIVE,check=True);cmd[1]='build';cmd.append('--release');subprocess.run(cmd,cwd=n.NATIVE,check=True)
    a.write(DEST/'build.json',dict(example_sha256=a.sha(path),binary_sha256=a.sha(n.TARGET/f'release/examples/{NAME}'),bad_anchor_refused=refused))


def replay(run):
    check();b=json.loads((DEST/'build.json').read_text());binary=n.TARGET/f'release/examples/{NAME}'
    if a.sha(binary)!=b['binary_sha256']:raise ValueError('binary differs')
    for kind,row in json.loads((n.DEST/run/'prefixes.json').read_text()).items():
        if a.sha(n.DEST/run/f'{kind}.jsonl.gz')!=row['bounded_gzip_sha256']:raise ValueError('prefix differs')
    subprocess.run([str(binary),str(n.NATIVE),str(n.DEST/run),str(DEST/run),str(a.RP/'scope.json'),run],check=True)
    for name in ['states.jsonl.gz','books.jsonl.gz','probes.jsonl.gz']:
        if a.sha(DEST/run/name)!=a.sha(g.DEST/run/name):raise ValueError('original observer artifact changed')
    rep=json.loads((DEST/run/'replay.json').read_text());prior=json.loads((g.DEST/run/'replay.json').read_text())
    for k in prior:
        if rep[k]!=prior[k]:raise ValueError(f'original replay differs: {k}')
    scope=json.loads((a.RP/'scope.json').read_text());starts={w['market']:w['ws']*1000 for w in scope['windows'] if w['run']==run}
    count=flatten_file(DEST/run/'inventory.jsonl.gz',DEST/f'inventory_{run}.parquet',run,lambda r:check_scope(r,starts))
    if count!=rep['inventory_events'] or count<=0:raise ValueError('inventory event population differs')
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),run=run,events=count,original_artifacts_exact=True,
        inventory_sha256=a.sha(DEST/f'inventory_{run}.parquet'),replay_sha256=a.sha(DEST/run/'replay.json'),
        scope='post-Filled native known position; not match time and not hypothetical fill feedback')
    a.write(DEST/run/'verification.json',out);a.write(REVIEW/f'verification_{run}.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':
    if sys.argv[1]=='replay':replay(sys.argv[2])
    else:{'declare':declare,'build':build}[sys.argv[1]]()
