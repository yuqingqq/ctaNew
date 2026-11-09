"""Validate and flatten the already declared read-only native event observer."""
import datetime as dt
import json
import subprocess
from . import pf_native_grid as g
from .pf_native_guard_verify import flatten_file


def check_scope(row, starts):
    ws=starts[int(row['market'])]
    if not ws<=row['t']<ws+65000:raise ValueError('observer outside scope')


def main():
    n=g.n;a=n.a
    if (g.DEST/'verification.json').exists():raise ValueError('verification exists')
    scope=json.loads((a.RP/'scope.json').read_text());counts={}
    check_scope(dict(market=1,t=100),{1:100})
    try:check_scope(dict(market=1,t=65100),{1:100})
    except ValueError:refused=True
    else:raise ValueError('boundary control accepted')
    for run in n.RUNS:
        p=g.DEST/run;par=json.loads((p/'parity.json').read_text());rep=json.loads((p/'replay.json').read_text())
        for name in ['states.jsonl.gz','books.jsonl.gz']:
            if a.sha(p/name)!=a.sha(n.DEST/run/'evidence'/name):raise ValueError('factual artifact differs')
        if a.sha(p/'probes.jsonl.gz')!=par['probes_sha256']:raise ValueError('probes changed')
        starts={w['market']:w['ws']*1000 for w in scope['windows'] if w['run']==run}
        counts[run]=flatten_file(p/'probes.jsonl.gz',g.DEST/f'probes_{run}.parquet',run,lambda d:check_scope(d,starts))
        if counts[run]!=rep['grid_probes'] or counts[run]<=0:raise ValueError('probe count differs')
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),counts=counts,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.HERE,text=True).strip(),
        source_sha256=a.sha(a.HERE/'pf_native_grid_verify.py'),
        artifacts={p.name:a.sha(p) for p in g.DEST.glob('probes_*.parquet')},
        controls=dict(positive=True,outside_scope_refused=refused),factual_states_and_books_exact=True)
    a.write(g.DEST/'verification.json',out);a.write(g.REVIEW/'verification.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':main()
