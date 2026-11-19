"""MUTATION CHECK of the pipeline benchmark's selftest (2026-11-19, session aea5bfb9): a checker that cannot fail proves nothing. Each entry plants one
code error in a COPY of pf_pipeline_bench.py (text replacement, executed in a subprocess; the repo file is never touched) and runs the copy's
selftest (synthetic tape only). Required: the unchanged copy passes, every mutant is caught BY A CHECK (the selftest's own failure message; a
crash is unexpected), and every replacement target occurs exactly as often as declared (a stale target is a failure). No market data is read.
Usage (run_light, PF_SLOT=claude): pf_pipeline_bench_mutants.py [index | targets]
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / 'pf_pipeline_bench.py'
# (name, text in the source, replacement[, occurrences]) - entry 0 is the unchanged copy
MUT = [
    ('none (must pass)', None, None),
    # --- the controls
    ('B1 export parts never checked against their receipt', "            C.require(C.sha(exp / d / x) == h, f'B1 {d}: export part {x} changed since its receipt')", "            pass"),
    ('B1 chronology never refuses', "    bad += [f'B1: training day {d} is not before every evaluation day' for d in TRAIN if not all(d < e for e in EVAL)]", "    bad += []"),
    ('B2 never refuses', "    bad = [f'B2 {c}: label or identity column {x} among the inputs' for c, fs in SETS.items() for x in fs if x.startswith(LABELS) or x in FORBIDDEN]", "    bad = []"),
    ('B3 never refuses', "                    if not np.allclose(p_['V'], edge * P_ * qf_ + P_ * m_['qm'].predict(Xc, num_threads=1), rtol=0, atol=1e-9):", "                    if False:"),
    ('B4 never refuses', "                        if not np.allclose(b.predict(", "                        if False and not np.allclose(b.predict("),
    ('B5 never refuses', "                    if not ((p_['P'] >= 0).all() and (p_['P'] <= 1).all() and (p_['Eq'] >= -1e-12).all() and (p_['Eq'] <= e.fp_remaining.values + 1e-9).all()):", "                    if False:"),
    ('a refused benchmark leaves its partial output', "        shutil.rmtree(part); sys.exit(f'{MARK} FAILED -> no output: '", "        sys.exit(f'{MARK} FAILED -> no output: '"),
    # --- the model and the mapping
    ('the mapping without the edge term', "    out = dict(P=P, Eq=Eq, Eqm=Eqm, V=edge * Eq + Eqm)", "    out = dict(P=P, Eq=Eq, Eqm=Eqm, V=Eq + Eqm)"),
    ('E[q] without the fill probability', "    Eq = P * qf; Eqm = P * models['qm'].predict(X, num_threads=1)", "    Eq = qf; Eqm = P * models['qm'].predict(X, num_threads=1)"),
    ('every row counted as filled', "filled = q > 1e-12", "filled = q >= 0."),
    # --- the readings
    ('Q1 band widened to [0.5, 2]', "if .9 <= ratio <= 1.1 else 'MISCALIBRATED-IN-AMOUNT'", "if .5 <= ratio <= 2. else 'MISCALIBRATED-IN-AMOUNT'"),
    ('Q2 ignores the rank', "'MOVE-TERM-INFORMATIVE' if rho_m > 0 and .5 <= rq <= 2. else", "'MOVE-TERM-INFORMATIVE' if .5 <= rq <= 2. else"),
    ('Q2 ignores the amount', "'MOVE-TERM-INFORMATIVE' if rho_m > 0 and .5 <= rq <= 2. else", "'MOVE-TERM-INFORMATIVE' if rho_m > 0 else"),
    ('Q3 a tie counts as beyond', "'RANKS-BEYOND-FAIR-EDGE' if rho > rho_f else", "'RANKS-BEYOND-FAIR-EDGE' if rho >= rho_f else"),
    ('Q4 a zero decile counts as losing', "'LOSING-DECILE-IDENTIFIED' if float(tab.rv.iloc[0]) < 0 else", "'LOSING-DECILE-IDENTIFIED' if float(tab.rv.iloc[0]) <= 0 else"),
    ('the gate ignores the amount calibration', "    return {kind: 'PASS' if all(R[kind][s][d]['J']['Q1 AMOUNT'] == 'CALIBRATED-IN-AMOUNT' and R[kind][s][d]['J']['Q3 VALUE RANKING'] == 'RANKS-BEYOND-FAIR-EDGE'",
     "    return {kind: 'PASS' if all(R[kind][s][d]['J']['Q3 VALUE RANKING'] == 'RANKS-BEYOND-FAIR-EDGE'"),
    # --- v2 (review 89caa74c): moment coherence and the economic gate
    ('the value-level band widened', "def ratio_in(a, b, lo=.8, hi=1.25):", "def ratio_in(a, b, lo=.0, hi=1e9):"),
    ('the value-level sign ignored', "    return bool(b != 0 and np.sign(a) == np.sign(b) and lo <= a / b <= hi)", "    return bool(b != 0 and lo <= abs(a / b) <= hi)"),
    ('E2 ignores the kept side', "'PASS' if float(v[loss].mean()) < 0 <= float(v[gain].mean()) else 'FAIL')", "'PASS' if float(v[loss].mean()) < 0 else 'FAIL')"),
    ('incoherent forecasts act', "    act = ok & ((V < 0) if kind == 'retain' else (V > 0));", "    act = ((V < 0) if kind == 'retain' else (V > 0));"),
    ('coherence ignores the endpoint bound', "(end >= -1e-9) & (end <= 100. + 1e-9)", "(end >= -1e9) & (end <= 1e9)"),
    ('the gate read on the post-hoc day', "E[kind][s][GATE_DAY][k] == 'PASS'", "E[kind][s][EVAL[-1]][k] == 'PASS'"),
    ('the gate ignores E2', "for k in ('E1 VALUE LEVEL', 'E2 BOUNDARY SIGN', 'E3 LOSS AMOUNT'))", "for k in ('E1 VALUE LEVEL', 'E3 LOSS AMOUNT'))"),
    ('the gate ignores the ranking', "    return {kind: 'PASS' if all(R[kind][s][d]['J']['Q1 AMOUNT'] == 'CALIBRATED-IN-AMOUNT' and R[kind][s][d]['J']['Q3 VALUE RANKING'] == 'RANKS-BEYOND-FAIR-EDGE'",
     "    return {kind: 'PASS' if all(R[kind][s][d]['J']['Q1 AMOUNT'] == 'CALIBRATED-IN-AMOUNT'"),
    # --- v3 (review 1adf6a33 P2): zero-quantity moments
    ('zero-quantity moments accepted (v2 coherence, no round trip)', "    if not V2_COHERENCE:\n        out['ok'] = out['ok'] & rt", "    if False:\n        out['ok'] = out['ok'] & rt"),
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
    g = dict(__name__='pf_pipeline_bench_mutant', __file__=str(SRC)); exec(compile(source(i), str(SRC), 'exec'), g); g['selftest']()


def main():
    bad = 0
    for i, (name, *_) in enumerate(MUT):
        r = subprocess.run([sys.executable, __file__, str(i)], capture_output=True, text=True)
        tail = r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or ['']
        caught = r.returncode == 1 and 'PIPELINE BENCH SELFTEST FAILED' in r.stderr   # caught BY A CHECK (the selftest's own exit message); a crash is unexpected
        ok = (r.returncode == 0) if i == 0 else caught
        bad += not ok
        first = next((x for x in r.stderr.splitlines() if 'SELFTEST FAILED' in x), None)
        how = 'passed' if r.returncode == 0 else ('caught: ' + first.strip()[:200]) if caught and first else ' / '.join(x.strip()[:200] for x in tail[-2:])
        print(f'{i:2d} {name}: rc {r.returncode} -> {how}{"" if ok else "   <<< UNEXPECTED"}', flush=True)
    if bad:
        sys.exit(f'PIPELINE BENCH MUTATION CHECK FAILED: {bad} unexpected outcomes of {len(MUT)}')
    print(f'pipeline bench mutation check passed: the unchanged copy passes its selftest, {len(MUT) - 1} / {len(MUT) - 1} planted code errors are caught by a check', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['targets']:
        [source(i) for i in range(len(MUT))]; print(f'all {len(MUT) - 1} mutation targets occur as declared')
    elif len(sys.argv) > 1:
        one(int(sys.argv[1]))
    else:
        main()
