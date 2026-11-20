"""Run-log disposition for the R2-B acceptance, check (6), kept apart so it can be falsified.
Every run under run_light ends its log with one footer '... tag=<t> rc=<n> peak_rss_mib=<m> ...'; a resumed log carries one per run.
disposition(logdir, declared) holds only when every declared log is present and every log in logdir except the acceptance's own has at
least one footer, each with rc 0 and peak <= CAP MiB. The acceptance's own logs (accept*.log) are excluded by name: they are written while
the acceptance runs, so they have no footer yet, and the acceptance's own disposition is its exit code.
v2, 2026-11-20 10:03 UTC: v1 globbed every log in logs/, including the acceptance's own unfinished log (the production run redirected into
logs/; the dry run had written elsewhere), so check (6) failed on the instrument, not on the build.
cases() runs the falsifiers on synthetic logs and returns (case, expected, got) for each:
a clean set passes, and so do a second clean footer and peak == CAP;
rc 1, rc 10, peak CAP + 1, a missing declared log, a footerless build log and a failing second footer each fail;
a footerless and a failing acceptance log are both excluded.
Usage: accept_logs.py selftest"""
import re
import sys
import tempfile
from pathlib import Path

CAP = 1536
FOOTER = re.compile(r' rc=(\d+) peak_rss_mib=(\d+)')
OWN = 'accept'


def disposition(logdir, declared):
    assert not any(n.startswith(OWN) for n in declared), 'a declared build log cannot carry the acceptance prefix'
    logdir = Path(logdir); every = sorted(logdir.glob('*.log'))
    seen = {p.name: FOOTER.findall(p.read_text()) for p in every if not p.name.startswith(OWN)}
    missing = sorted(set(declared) - set(seen))
    bad = sorted(k for k, v in seen.items() if not (len(v) >= 1 and all(int(a) == 0 and int(b) <= CAP for a, b in v)))
    return dict(ok=not missing and not bad, missing=missing, bad=bad, logs=seen, excluded=sorted(p.name for p in every if p.name.startswith(OWN)))


def foot(rc=0, peak=100):
    return f'2026-11-20T00:00:00Z tag=x rc={rc} peak_rss_mib={peak} wall_s=1.0 cpu_user_s=1.0 memavail_start_gib=23 slot=claude\n'


DECL = ['plan.log', 'run_0.log']
CASES = [('clean', {'plan.log': 'x\n' + foot(), 'run_0.log': foot()}, True),
         ('a second clean footer (resume)', {'plan.log': foot(), 'run_0.log': foot() + foot(peak=900)}, True),
         ('peak 1536 = the cap', {'plan.log': foot(peak=1536), 'run_0.log': foot()}, True),
         ('rc 1', {'plan.log': foot(rc=1), 'run_0.log': foot()}, False),
         ('rc 10', {'plan.log': foot(), 'run_0.log': foot(rc=10)}, False),
         ('peak 1537', {'plan.log': foot(), 'run_0.log': foot(peak=1537)}, False),
         ('a declared log missing', {'plan.log': foot()}, False),
         ('a footerless build log', {'plan.log': foot(), 'run_0.log': foot(), 'gate.log': 'running\n'}, False),
         ('a failing second footer', {'plan.log': foot() + foot(rc=1), 'run_0.log': foot()}, False),
         ('a footerless acceptance log is excluded', {'plan.log': foot(), 'run_0.log': foot(), 'accept_full.log': 'running\n'}, True),
         ('a failing acceptance log is excluded', {'plan.log': foot(), 'run_0.log': foot(), 'accept_full_v1.log': foot(rc=1)}, True)]


def cases():
    out = []
    for name, files, want in CASES:
        with tempfile.TemporaryDirectory() as t:
            for k, v in files.items():
                (Path(t) / k).write_text(v)
            out.append((name, want, disposition(t, DECL)['ok']))
    return out


if __name__ == '__main__' and sys.argv[1:] == ['selftest']:
    r = cases(); wrong = [c for c in r if c[1] != c[2]]
    for c in r:
        print(f'  {c[0]}: expected {c[1]}, got {c[2]}')
    print(f'accept_logs selftest: {len(r) - len(wrong)} / {len(r)} cases as expected' + ('' if not wrong else f'; WRONG {wrong}'))
    sys.exit(0 if not wrong else 1)
