"""Observe each due native event without changing the factual output stream.

The isolated native tree gets a read-only input accessor copied from decide().
Advance calls strictly before the next captured input split work the original
core would do anyway; every resulting output still must match the capture.
"""
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
from . import pf_native_guard as n

DEST=n.DEST/'grid'
REVIEW=n.REVIEW/'grid'
SOURCE=['pf_native_grid.py','native_guard_evidence.rs']
REL='crates/pf-engine/src/ladder.rs'

PROBE=r'''
fn probe(core: &LadderCore, cfg: &LadderConfig, t: f64, windows: &BTreeMap<u64,f64>, w: &mut impl Write, index: u64) -> u64 {
    let mut count=0;
    for (&market,&ws) in windows {
        if t<ws || t>=ws+65000. {continue;}
        let Some(window)=core.window(market) else {continue;};
        let x=window.diagnostic_inputs(t);let held=window.held();
        let mut pure=desired_ladder(cfg,&x).lanes;if held{pure=[(None,0.);4];}
        let mut relaxed=BTreeMap::new();
        for delta in [0.005,0.01,0.02,0.04] {
            let mut alt=cfg.clone();alt.pull=cfg.pull.map(|v|MinuteLine(v.0.map(|z|z-delta)));
            alt.rung_line=MinuteLine(cfg.rung_line.0.map(|z|z-delta));
            let mut q=desired_ladder(&alt,&x).lanes;if held{q=[(None,0.);4];}
            relaxed.insert(format!("{delta}"),q);
        }
        let book=x.book.map(|b|json!({"bids":b.bids.map(|z|(z.tick,z.size)),"asks":b.asks.map(|z|(z.tick,z.size))}));
        emit(w,&json!({"market":market,"t":t,"input_index":index,"rel_s":x.rel_s,"v":x.v,
            "binance":x.binance,"net":x.net,"live":x.live,"rung_cur":x.rung_cur,"held":held,"book":book,
            "lanes":pure,"relaxed":relaxed}));count+=1;
    }
    count
}
'''


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('source anchor identity differs')
    return text.replace(old,new)


def accessor(original):
    anchor='    fn decide(&mut self, t: f64, out: &mut Vec<Request>) -> LadderDecided {\n'
    if original.count(anchor)!=1:raise ValueError('decide identity differs')
    body=original.split(anchor)[1].split('        let mut d = desired_ladder(&self.cfg, &x);')[0]
    if not body.endswith('        };\n') or 'let x = LadderInputs {' not in body:raise ValueError('input extraction boundary differs')
    method='    /// Read-only diagnostic; exact input construction from decide().\n    pub fn diagnostic_inputs(&self, t: f64) -> LadderInputs {\n'+body+'        x\n    }\n\n'
    result=once(original,anchor,method+anchor)
    if result.replace(method,'',1)!=original:raise ValueError('native behavior changed')
    return result,method


def instrument(original):
    x=once(original,'fn main() {',PROBE+'\nfn main() {')
    x=once(x,'let mut books=output(&out.join("books.jsonl.gz"));let mut held=BTreeMap::new();',
        'let mut books=output(&out.join("books.jsonl.gz"));let mut held=BTreeMap::new();\n    let mut probes=output(&out.join("probes.jsonl.gz"));let mut nprobe=0u64;let mut advances=0u64;')
    x=once(x,'        let step=core.apply(&i).unwrap();',
        '''        let original=i;
        loop {
            let earlier=original.time().and_then(|limit|core.next_due().filter(|t|*t<limit));
            let synthetic=earlier.is_some();
            let i=earlier.map_or_else(||original.clone(),|t|CoreInput::Advance{now_ms:t});
            if synthetic {advances+=1;}
            let step=core.apply(&i).unwrap();''')
    x=once(x,'        if let CoreInput::Hold{market,held:h,..}=i{held.insert(market,h);}',
        '''        if let CoreInput::Hold{market,held:h,..}=&i{held.insert(*market,*h);}
        if let CoreInput::Advance{now_ms}=&i {
            nprobe+=probe(&core,&cfg,*now_ms,&windows,&mut probes,inputs);
        }
        if !synthetic {break;}
        }''')
    x=once(x,'states.finish().unwrap();books.finish().unwrap();','states.finish().unwrap();books.finish().unwrap();probes.finish().unwrap();')
    x=once(x,'"book_frames":nbook,"market_counts":per_market','"book_frames":nbook,"grid_probes":nprobe,"synthetic_advances":advances,"market_counts":per_market')
    return x


def declare():
    n.check()
    if DEST.exists():raise ValueError('grid declaration exists')
    (DEST/'logs').mkdir(parents=True)
    original=(n.NATIVE/REL).read_text();patched,method=accessor(original)
    a=n.a
    d=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),native_commit=n.NATIVE_COMMIT,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.HERE,text=True).strip(),
        sources={p:a.sha(a.HERE/p) for p in SOURCE},original_native_sha256=a.sha(n.NATIVE/REL),
        method=method,scope='same bounded factual inputs; observe all due events to catch lower-line crossings absent from old decisions',
        design='read-only input accessor; split run_until with inclusive Advance at each next_due strictly before the next captured input; compare every output exactly, preserve original states/books byte-for-byte',
        no_changed_order_feedback=True,no_protected_dates_read=True,no_live_change=True,
        null='deterministic observer parity and corrupted anchor refusal; no strategy result')
    a.write(DEST/'declaration.json',d);a.write(REVIEW/'declaration.json',d)


def build():
    a=n.a;d=json.loads((DEST/'declaration.json').read_text())
    for p,h in d['sources'].items():
        if a.sha(a.HERE/p)!=h:raise ValueError('observer source changed')
    original=(n.NATIVE/REL).read_text()
    if a.sha(n.NATIVE/REL)!=d['original_native_sha256']:raise ValueError('native source changed before accessor installation')
    patched,method=accessor(original)
    if method!=d['method']:raise ValueError('accessor changed')
    try:once('known','missing','new')
    except ValueError:refused=True
    else:refused=False
    if not refused:raise ValueError('bad source anchor accepted')
    (n.NATIVE/REL).write_text(patched)
    example=n.NATIVE/'crates/pf-engine/examples/native_guard_grid.rs'
    if example.exists():raise ValueError('grid example exists')
    example.write_text(instrument((a.HERE/'native_guard_evidence.rs').read_text()))
    cmd=['cargo','test','--locked','--offline','-p','pf-engine','--example','native_guard_grid','--target-dir',str(n.TARGET),'-j','1']
    subprocess.run(cmd,cwd=n.NATIVE,check=True)
    cmd[1]='build';cmd.append('--release');subprocess.run(cmd,cwd=n.NATIVE,check=True)
    a.write(DEST/'build.json',dict(native_commit=n.NATIVE_COMMIT,accessor_source_sha256=a.sha(n.NATIVE/REL),
        generated_example_sha256=a.sha(example),binary_sha256=a.sha(n.TARGET/'release/examples/native_guard_grid'),
        controls=dict(original_method_unchanged=patched.replace(method,'',1)==original,bad_anchor_refused=refused)))


def replay(run):
    a=n.a;d=json.loads((DEST/'declaration.json').read_text());b=json.loads((DEST/'build.json').read_text())
    for p,h in d['sources'].items():
        if a.sha(a.HERE/p)!=h:raise ValueError('observer source changed')
    if a.sha(n.NATIVE/REL)!=b['accessor_source_sha256'] or a.sha(n.TARGET/'release/examples/native_guard_grid')!=b['binary_sha256']:raise ValueError('observer build changed')
    for kind,row in json.loads((n.DEST/run/'prefixes.json').read_text()).items():
        if a.sha(n.DEST/run/f'{kind}.jsonl.gz')!=row['bounded_gzip_sha256']:raise ValueError('bounded capture changed')
    subprocess.run([str(n.TARGET/'release/examples/native_guard_grid'),str(n.NATIVE),str(n.DEST/run),str(DEST/run),str(a.RP/'scope.json'),run],check=True)
    old=json.loads((n.DEST/run/'evidence/replay.json').read_text());new=json.loads((DEST/run/'replay.json').read_text())
    if new['grid_probes']<=0 or new['synthetic_advances']<=0:raise ValueError('observer failed to sample due events')
    for k in ['inputs','outputs_matched','exported_decisions','book_frames','market_counts','basis_mode','unapplied_boundary_tail_outputs']:
        if new[k]!=old[k]:raise ValueError('factual replay differs')
    for p in ['states.jsonl.gz','books.jsonl.gz']:
        if a.sha(DEST/run/p)!=a.sha(n.DEST/run/'evidence'/p):raise ValueError('observer changed factual artifact')
    a.write(DEST/run/'parity.json',dict(run=run,old_replay_sha256=a.sha(n.DEST/run/'evidence/replay.json'),
        replay_sha256=a.sha(DEST/run/'replay.json'),states_exact=True,books_exact=True,
        probes_sha256=a.sha(DEST/run/'probes.jsonl.gz')))


if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='replay':replay(sys.argv[2])
    else:{'declare':declare,'build':build}[mode]()
