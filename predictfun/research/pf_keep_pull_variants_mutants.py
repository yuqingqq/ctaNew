"""MUTATION CHECK of the keep / pull variants comparison's selftest (2026-11-19, session aea5bfb9): a checker that cannot fail proves nothing.
Each entry plants one code error in a COPY of pf_keep_pull_variants.py (text replacement, executed in a subprocess; the repo file is never
touched) and runs the copy's selftest (synthetic tape only). Required: the unchanged copy passes, every mutant is caught BY A CHECK (the
selftest's own failure message; a crash is unexpected), and every replacement target occurs exactly as often as declared (a stale target is a
failure, never a silent pass). The policy parameter of pf_keep_pull_scenarios.outcome is mutated in that module's own suite. No market data is read.
Usage (run_light, PF_SLOT=claude): pf_keep_pull_variants_mutants.py [index | targets]
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / 'pf_keep_pull_variants.py'
# (name, text in the source, replacement[, occurrences]) - entry 0 is the unchanged copy
MUT = [
    ('none (must pass)', None, None),
    # --- the controls
    ('V0a never refuses', "            if dd:\n                bad.append(f'V0a {nm}", "            if False:\n                bad.append(f'V0a {nm}"),
    ('V0b never refuses', "        if not (int(round(n)) == b['pulls'] and abs(imp - b['score']) <= 1e-6 * max(1., abs(imp))):", "        if False:"),
    ('V0b compares the pulls only', " and abs(imp - b['score']) <= 1e-6 * max(1., abs(imp))):", "):"),
    ('V0b reads the next day of the committed run', "ix = [KP.DAYS.index(d) for d in days]", "ix = [KP.DAYS.index(d) + 1 for d in days]"),
    ('V0c never refuses', "        if not (abs(b['E'] - e_) < 1e-6 and abs(b['score'] - s_) < 1e-6 and pf['pulls'] == b['pulls']):", "        if False:"),
    ('V0c ignores E', "        if not (abs(b['E'] - e_) < 1e-6 and abs(b['score'] - s_) < 1e-6", "        if not (abs(b['score'] - s_) < 1e-6"),
    ('V0c values every policy with the baseline\'s triggers', "pf = A.per_fill(T[p], cell, endj, fl)", "pf = A.per_fill(T[BASE], cell, endj, fl)"),
    ('V0d ignores KEEP', "{(res[p][nm]['KEEP'], res[p][nm]['PULLALL'],", "{(res[p][nm]['PULLALL'],"),
    ('V0d never checks KEEP against its fills', "        if k:\n            bad.append(f'V0d {p}", "        if False:\n            bad.append(f'V0d {p}"),
    ('V0e: scenario outputs not checked against their receipt',
     "C.require(all(C.sha(sc_out / d / x) == h for x, h in r['outputs'].items()), f'V0e {d}: scenario outputs changed since their receipt')", "pass"),
    ('V0e: keep / pull outputs not checked against their receipt',
     "C.require(all(C.sha(kp / d / x) == h for x, h in rk['outputs'].items()), f'V0e {d}: keep-pull outputs changed since their receipt')", "pass"),
    ('V0e: the scenario days\' keep / pull inputs not checked', "C.require(r['keep_pull_outputs'] == rk['outputs'], f'V0e {d}: scenario day built on other keep-pull outputs')", "pass"),
    ('output written before a refusal', "    if bad:\n        sys.exit(f'{MARK} FAILED -> no output: '",
     "    if bad:\n        out.mkdir(parents=True, exist_ok=True); (out / f'{stem}.txt').write_text('x')\n        sys.exit(f'{MARK} FAILED -> no output: '"),
    # --- the values
    ('every policy valued as the baseline', "SC.outcome(O, Os, Fs, sc, policy=p)", "SC.outcome(O, Os, Fs, sc)"),
    ('M0 / M250 read the baseline clock\'s fills', "Fs = Fl[Fl.clock == float(sc['tl'][1:])][['day', 'oid', 't', 'q', 'src']]", "Fs = Fl[Fl.clock == SC.SHIFT][['day', 'oid', 't', 'q', 'src']]"),
    ('V without KEEP', "V=v['KEEP'] + v['score'], score=v['score'], vs_pullall", "V=v['score'], score=v['score'], vs_pullall"),
    ('V - V(PULL-ALL) with the wrong sign', "vs_pullall=v['score'] - v['PULLALL']", "vs_pullall=v['score'] + v['PULLALL']"),
    ('E without young fills keeps them', "E_ny=ny['E'],", "E_ny=v['E'],"),
    ('V without young fills keeps them', "V_ny=ny['fills_value'] + ny['score'])", "V_ny=v['KEEP'] + v['score'])"),
    ('dV with the wrong sign', "out[nm] = dict(dV=a['V'] - b['V'],", "out[nm] = dict(dV=b['V'] - a['V'],"),
    ('the candidate compared with itself', "a, b = numbers(res[c][nm]), numbers(res[BASE][nm])", "a, b = numbers(res[c][nm]), numbers(res[c][nm])"),
    ('the CSV E column holds the improvement', "scenario=nm, family=fam[nm], E=v['E'], score=v['score']", "scenario=nm, family=fam[nm], E=v['score'], score=v['score']"),
    # --- the readings
    ('above: zero counts as above', "if lo > 0 else f'BELOW", "if lo >= 0 else f'BELOW"),
    ('above: zero counts as below', "if hi < 0 else 'NOT-STABLE-IN-TESTED-SCENARIOS'", "if hi <= 0 else 'NOT-STABLE-IN-TESTED-SCENARIOS'"),
    ('P4 V - KEEP read on V', "above(lo['score'], hi['score'], 'KEEP')", "above(lo['V'], hi['V'], 'KEEP')"),
    ('C2 E compares B with itself', "SC.at_m250(x[B_NAME]['E'], x[M250_NAME]['E'])", "SC.at_m250(x[B_NAME]['E'], x[B_NAME]['E'])"),
    ('C1 B1a sign of the M250 E', "SC.b1a_sign(res[p][B_NAME]['E'], res[p][B_NAME]['b1a_E'])", "SC.b1a_sign(res[p][M250_NAME]['E'], res[p][B_NAME]['b1a_E'])"),
    ('SURVIVES-TIMING read from P3', "if R[p][P_KEYS[0]] == 'STABLE-POSITIVE-IN-TESTED-SCENARIOS'", "if R[p][P_KEYS[1]] == 'STABLE-POSITIVE-IN-TESTED-SCENARIOS'"),
    ('the summary of D1 read from D2', "if R[c][D_KEYS[0]] == 'ABOVE-BASELINE-IN-TESTED-SCENARIOS'", "if R[c][D_KEYS[1]] == 'ABOVE-BASELINE-IN-TESTED-SCENARIOS'"),
    ('D1-NY read on dV', "above(lo['dV_ny'], hi['dV_ny'], 'BASELINE')", "above(lo['dV'], hi['dV'], 'BASELINE')"),
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
    g = dict(__name__='pf_keep_pull_variants_mutant', __file__=str(SRC)); exec(compile(source(i), str(SRC), 'exec'), g); g['selftest']()


def main():
    bad = 0
    for i, (name, *_) in enumerate(MUT):
        r = subprocess.run([sys.executable, __file__, str(i)], capture_output=True, text=True)
        tail = r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or ['']
        caught = r.returncode == 1 and 'KEEP-PULL VARIANTS SELFTEST FAILED' in r.stderr   # caught BY A CHECK (the selftest's own exit message); a crash is unexpected
        ok = (r.returncode == 0) if i == 0 else caught
        bad += not ok
        first = next((x for x in r.stderr.splitlines() if 'SELFTEST FAILED' in x), None)
        how = 'passed' if r.returncode == 0 else ('caught: ' + first.strip()[:200]) if caught and first else ' / '.join(x.strip()[:200] for x in tail[-2:])
        print(f'{i:2d} {name}: rc {r.returncode} -> {how}{"" if ok else "   <<< UNEXPECTED"}', flush=True)
    if bad:
        sys.exit(f'KEEP-PULL VARIANTS MUTATION CHECK FAILED: {bad} unexpected outcomes of {len(MUT)}')
    print(f'keep-pull variants mutation check passed: the unchanged copy passes its selftest, {len(MUT) - 1} / {len(MUT) - 1} planted code errors are caught by a check', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['targets']:
        [source(i) for i in range(len(MUT))]; print(f'all {len(MUT) - 1} mutation targets occur as declared')
    elif len(sys.argv) > 1:
        one(int(sys.argv[1]))
    else:
        main()
