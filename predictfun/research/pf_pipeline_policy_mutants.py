"""MUTATION CHECK of the pipeline policy test's selftest (2026-11-19, session aea5bfb9): a checker that cannot fail proves nothing. Each entry plants one
code error in a COPY of pf_pipeline_policy.py (text replacement, executed in a subprocess; the repo file is never touched) and runs the copy's
selftest (synthetic tape only). Required: the unchanged copy passes, every mutant is caught BY A CHECK (the selftest's own failure message; a
crash is unexpected), and every replacement target occurs exactly as often as declared (a stale target is a failure). No market data is read.
Usage (run_light, PF_SLOT=claude): pf_pipeline_policy_mutants.py [index | targets]
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / 'pf_pipeline_policy.py'
# (name, text in the source, replacement[, occurrences]) - entry 0 is the unchanged copy
MUT = [
    ('none (must pass)', None, None),
    ('P0a never refuses', "        if ref is None or abs(v['E'] - float(ref)) > 1e-6 * max(1., abs(float(ref))):", "        if False:"),
    ('P0b never refuses', "    if not (int(round(D['n'][i])) == b['pulls'] and abs(D['imp'][i] - b['score']) <= 1e-6 * max(1., abs(D['imp'][i]))):", "    if False:"),
    ('P0c never refuses', "    if not (abs(m['E'] - e_) < 1e-6 and abs(m['score'] - s_) < 1e-6 and pf['pulls'] == m['pulls']):", "    if False:"),
    ('P0d never refuses', "        if len({(res[r][nm]['KEEP'], res[r][nm]['PULLALL'], res[r][nm]['fills'], res[r][nm]['eligible']) for r in res}) != 1:", "        if False:"),
    ('P0e never refuses', "    if got != exp_rows:", "    if False:"),
    ('P0f never refuses', "    if not (comp > refd and agree == comp):", "    if False:"),
    ('the rule pulls at the last predicted loss', ".groupby(['day', 'oid']).j.min()   # v2", ".groupby(['day', 'oid']).j.max()   # v2"),
    ('an order without a predicted loss pulled at once', ".fillna(KP.NSTEP + 1)", ".fillna(KP.J0)"),
    ('the differences against A+0 reversed', "d[nm] = dict(dV=a['V'] - b['V'], dE=a['E'] - b['E'], dV_ny=a['V_ny'] - b['V_ny'])", "d[nm] = dict(dV=b['V'] - a['V'], dE=a['E'] - b['E'], dV_ny=a['V_ny'] - b['V_ny'])"),
    ('the PULL-ALL value column holds the improvement', "PULLALL_value=res[r][nm]['KEEP'] + res[r][nm]['PULLALL']", "PULLALL_value=res[r][nm]['PULLALL']"),
    # --- v2 (review 89caa74c P2): one accepted-forecast path
    ('the rule acts on refused forecasts', "hit = p[acts if RAW_TRIGGERS else (acts & p.J_ok.astype(bool))]", "hit = p[acts]"),
    ('P0g never refuses', "    if off:\n        bad.append(f'P0g", "    if False:\n        bad.append(f'P0g"),
    ('the parity ignores the contract verdict', "agree += int(coherent == bool(r.J_ok) and action == rule)", "agree += int(action == rule)"),
    # --- v3 (review 1adf6a33 P2): the parity on every row, with the moments' round trip
    ('the parity compares actions only (no round trip)', "        coherent = accepted and rt", "        coherent = accepted"),
    ('the round-trip tolerance widened', "tol = lambda x: 1e-9 * max(1., abs(x))", "tol = lambda x: 1e3 * max(1., abs(x))"),
    ('the parity skips zero-quantity rows', "    for r in p.itertuples():", "    for r in p[p.J_Eq > 0].itertuples():"),
    ('no row compared by the parity', "    p = pred[pred.kind == 'retain']; comp = ref = agree = rtbad = tiny = 0", "    p = pred[pred.kind == 'retain'].iloc[:0]; comp = ref = agree = rtbad = tiny = 0"),
    ('near-zero values skip the verdict', "            tiny += 1; agree += int(coherent == bool(r.J_ok))", "            tiny += 1; agree += 1"),
    # --- v3.1: the rebuilt quantity at the remaining-size clip
    ('the rebuilt quantity not set back to the remaining', "            qf = rem if rem < qf <= rem + tol(rem) else qf ", "            qf = qf "),
]
STALE = 97                                                # exit code of a replacement target that does not occur as declared


def source(i):
    name, a, b = MUT[i][:3]; n = (MUT[i] + (1,))[3]; s = SRC.read_text()
    if a is not None:
        if s.count(a) != n:
            print(f'mutation target occurs {s.count(a)} times, declared {n}: {name}', file=sys.stderr); sys.exit(STALE)
        s = s.replace(a, b)
    return s


def one(i):
    g = dict(__name__='pf_pipeline_policy_mutant', __file__=str(SRC)); exec(compile(source(i), str(SRC), 'exec'), g); g['selftest']()


def main():
    bad = 0
    for i, (name, *_) in enumerate(MUT):
        r = subprocess.run([sys.executable, __file__, str(i)], capture_output=True, text=True)
        tail = r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or ['']
        caught = r.returncode == 1 and 'PIPELINE POLICY SELFTEST FAILED' in r.stderr   # caught BY A CHECK (the selftest's own exit message); a crash is unexpected
        ok = (r.returncode == 0) if i == 0 else caught
        bad += not ok
        first = next((x for x in r.stderr.splitlines() if 'SELFTEST FAILED' in x), None)
        how = 'passed' if r.returncode == 0 else ('caught: ' + first.strip()[:200]) if caught and first else ' / '.join(x.strip()[:200] for x in tail[-2:])
        print(f'{i:2d} {name}: rc {r.returncode} -> {how}{"" if ok else "   <<< UNEXPECTED"}', flush=True)
    if bad:
        sys.exit(f'PIPELINE POLICY MUTATION CHECK FAILED: {bad} unexpected outcomes of {len(MUT)}')
    print(f'pipeline policy mutation check passed: the unchanged copy passes its selftest, {len(MUT) - 1} / {len(MUT) - 1} planted code errors are caught by a check', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['targets']:
        [source(i) for i in range(len(MUT))]; print(f'all {len(MUT) - 1} mutation targets occur as declared')
    elif len(sys.argv) > 1:
        one(int(sys.argv[1]))
    else:
        main()
