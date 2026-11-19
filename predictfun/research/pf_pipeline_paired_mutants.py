"""MUTATION CHECK of the paired action-value candidate's selftest (2026-11-19, session aea5bfb9): a checker that cannot fail proves nothing. Each entry plants one code
error in a COPY of pf_pipeline_paired.py (text replacement, executed in a subprocess; the repo file is never touched) and runs the copy's selftest
(synthetic tape only). Required: the unchanged copy passes, every mutant is caught BY A CHECK (the selftest's own failure message; a crash is
unexpected), and every replacement target occurs exactly as often as declared (a stale target is a failure). No market data is read.
Usage (run_light, PF_SLOT=claude): pf_pipeline_paired_mutants.py [index | targets]
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / 'pf_pipeline_paired.py'
# (name, text in the source, replacement[, occurrences]) - entry 0 is the unchanged copy
MUT = [
    ('none (must pass)', None, None),
    # --- the controls
    ('K1 never refuses (warm-up order)', "    bad += [f'K1: the warm-up day {w} is not before {d}' for w in WARM for d in LABEL + EVAL if not w < d]", "    bad += []"),
    ('K1 training rows outside the label days accepted', "            C.require(set(tr[k].day.astype(str)) <= set(LABEL), f'K1: {k} training rows outside {LABEL}')", "            pass"),
    ('candidates trained on every labelled day', "            if d in LABEL or (TRAIN_EVAL and d == EVAL[0]):\n                trn['retain']", "            if True:\n                trn['retain']"),
    ('K2 never refuses', "if x.startswith(BX.LABELS) or x.startswith('pair_') or x in BX.FORBIDDEN]", "if False]"),
    ('K3 export parts never checked', "    BX.verify_exports(exp, WARM + LABEL + EVAL); part", "    part"),
    ('K4 never refuses', "        if off_r or off_p or off_t:", "        if False:"),
    ('K4 ignores the RETAIN windows', "w = window_sums(f, r.oid.values, lo, hi); off_r = int(", "w = window_sums(f, r.oid.values, lo, hi); off_r = 0 * int("),
    ('K4 ignores the POST windows', "hp)); off_p = int(", "hp)); off_p = 0 * int("),
    ('K5 never refuses', "        if got != want or len(got) != len(r) + len(p):", "        if False:"),
    ('K6 never refuses', "                    if not np.allclose(b.predict(xx, num_threads=1), live.predict(xx, num_threads=1), rtol=0, atol=1e-10):", "                    if False:"),
    ('a refused run leaves its partial output', "        if part.exists():\n            shutil.rmtree(part)\n        raise", "        raise"),
    # --- the labels
    ('T* from the current step', "    nxt = inc if TSTAR_SAME else inc.groupby(d.oid.values).shift(-1)", "    nxt = inc"),
    ('the POST label at pi0\'s last pull', "first = d.groupby('oid').s.min()", "first = d.groupby('oid').s.max()"),
    ('the RETAIN label without pi0\'s later pull', "        out[f'pair_{x}'] = rv if NEUTRAL_LABEL else rv - later", "        out[f'pair_{x}'] = rv"),
    ('the POST label without pi0\'s pull', "dp[col].values.astype(float) - (0. if NEUTRAL_POST else later))", "dp[col].values.astype(float))"),
    # --- the gate, the reached rows and the bins
    ('the gate ignores the realized sign', "ok = (m > 0 if want_pos else m < 0) and ratio_ok(", "ok = True and ratio_ok("),
    ('the gate ratio band widened', "    return bool(r != 0 and np.sign(p) == np.sign(r) and .8 <= p / r <= 1.25)", "    return bool(r != 0 and np.sign(p) == np.sign(r) and .1 <= p / r <= 10.)"),
    ('an empty group passes', "    if not len(real) or float(np.sum(real)) == 0.:\n        return 'NOT-EVALUABLE'", "    if not len(real) or float(np.sum(real)) == 0.:\n        return 'PASS'"),
    ('reached rows past the pull', "    return d.j.values <= first", "    return d.j.values <= first + 1"),
    ('zero joins the negative bin', "    v = np.asarray(v, float); return np.where(v == 0., -1, np.digitize(v, EDGES, right=True))", "    v = np.asarray(v, float); return np.digitize(v, EDGES, right=True)"),
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
    g = dict(__name__='pf_pipeline_paired_mutant', __file__=str(SRC)); exec(compile(source(i), str(SRC), 'exec'), g); g['selftest']()


def main():
    bad = 0
    for i, (name, *_) in enumerate(MUT):
        r = subprocess.run([sys.executable, __file__, str(i)], capture_output=True, text=True)
        tail = r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or ['']
        caught = r.returncode == 1 and 'PIPELINE PAIRED SELFTEST FAILED' in r.stderr   # caught BY A CHECK (the selftest's own exit message); a crash is unexpected
        ok = (r.returncode == 0) if i == 0 else caught
        bad += not ok
        first = next((x for x in r.stderr.splitlines() if 'SELFTEST FAILED' in x), None)
        how = 'passed' if r.returncode == 0 else ('caught: ' + first.strip()[:200]) if caught and first else ' / '.join(x.strip()[:200] for x in tail[-2:])
        print(f'{i:2d} {name}: rc {r.returncode} -> {how}{"" if ok else "   <<< UNEXPECTED"}', flush=True)
    if bad:
        sys.exit(f'PIPELINE PAIRED MUTATION CHECK FAILED: {bad} unexpected outcomes of {len(MUT)}')
    print(f'pipeline paired mutation check passed: the unchanged copy passes its selftest, {len(MUT) - 1} / {len(MUT) - 1} planted code errors are caught by a check', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['targets']:
        [source(i) for i in range(len(MUT))]; print(f'all {len(MUT) - 1} mutation targets occur as declared')
    elif len(sys.argv) > 1:
        one(int(sys.argv[1]))
    else:
        main()
