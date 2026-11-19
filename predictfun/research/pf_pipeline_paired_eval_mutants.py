"""MUTATION CHECK of the paired candidate evaluation's selftest (2026-11-19, session aea5bfb9): a checker that cannot fail proves nothing. Each entry plants one code
error in a COPY of pf_pipeline_paired_eval.py (text replacement, executed in a subprocess; the repo file is never touched) and runs the copy's selftest
(synthetic tape only). Required: the unchanged copy passes, every mutant is caught BY A CHECK (the selftest's own failure message; a crash is
unexpected), and every replacement target occurs exactly as often as declared (a stale target is a failure). No market data is read.
Usage (run_light, PF_SLOT=claude): pf_pipeline_paired_eval_mutants.py [index | targets]
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / 'pf_pipeline_paired_eval.py'
# (name, text in the source, replacement[, occurrences]) - entry 0 is the unchanged copy
MUT = [
    ('none (must pass)', None, None),
    # --- the controls
    ('R0a never refuses', "    if got != want or len(got) != len(pred) or pred.J_V.isna().any():", "    if False:"),
    ('R0b never refuses (values)', "                if not E.close(b[c], D[c].values[m].sum()):", "                if False:"),
    ('R0c never refuses', "            if not E.close(b0['MODEL-J'], lab):", "            if False:"),
    ('R0d never refuses', "            if not all(np.allclose(Ds[r][f'MODEL-J_w{w}'].values, base, rtol=0, atol=1e-9) and", "            if False and not all(np.allclose(Ds[r][f'MODEL-J_w{w}'].values, base, rtol=0, atol=1e-9) and"),
    ('R0e never refuses', "            if not (len(adm[ent]) == b['eligible'] + n_ref and int(m.sum()) == b['eligible']):", "            if False:"),
    ('R0f never refuses', "        if a2 != adm[r] or h2 != hd:", "        if False:"),
    # --- the rules
    ('admissions without the coherence flag', "    return np.isfinite(x) & (df[ok].values.astype(bool) if ok else True)", "    return np.isfinite(x)"),
    ('A posts on a predicted loss', "    m = usable(post, rule) & (post[RULES[rule][0]].values.astype(float) > 0)", "    m = usable(post, rule) & (post[RULES[rule][0]].values.astype(float) < 0)"),
    ('the matched null over all opportunities', "keys['sel'] = [(d_, int(o)) in adm['A'] for d_, o in zip(keys.day, keys.oid)]", "keys['sel'] = True"),
    # --- the readings
    ('the matched control against zero', "'PASS' if v['A']['both'] > null_b else 'FAIL'", "'PASS' if v['A']['both'] > 0 else 'FAIL'"),
    ('the economic gate\'s pi0 predicate against A-PA', "v['A'][sd] > v['A-PA'][sd], v['A'][sd] > v['PI0'][sd]]", "v['A'][sd] > v['A-PA'][sd], v['A'][sd] > v['A-PA'][sd]]"),
    ('the economic gate ignores pi0', "v['A'][sd] > v['A-PA'][sd], v['A'][sd] > v['PI0'][sd]]", "v['A'][sd] > v['A-PA'][sd], True]"),
    ('the practical gate ignores the economic gate', "if not support_ok else ('PASS' if all(prac) and econ == 'PASS' else 'FAIL')", "if not support_ok else ('PASS' if all(prac) else 'FAIL')"),
    ('the practical gate ignores the support floor', "    return 'INSUFFICIENT-SUPPORT' if not support_ok else", "    return 'INSUFFICIENT-SUPPORT' if False else"),
    ('the practical predicates reversed (lifetime)', "lt(a['both']) is not None and lt(b['both']) is not None and lt(a['both']) > lt(b['both'])", "lt(a['both']) is not None and lt(b['both']) is not None and lt(a['both']) < lt(b['both'])"),
    ('first-review reversals over every admitted order', "        live0 = ms & (D.endj.values > KP.J0); pol_c", "        live0 = ms.copy(); pol_c"),
    ('the reached-state shift ignores admissions', "reach = reach_all & kix.isin(pd.MultiIndex.from_tuples(sorted(adm['A']))) if adm['A'] else np.zeros(len(ra), bool)", "reach = reach_all"),
    ('the scenario reading against PI0 without PI0', "        x = {nm: val('A', 'both', nm) - val(ref, 'both', nm) for nm in nul}", "        x = {nm: val('A', 'both', nm) - (0. if ref == 'PI0' else val(ref, 'both', nm)) for nm in nul}"),
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
    g = dict(__name__='pf_pipeline_paired_eval_mutant', __file__=str(SRC)); exec(compile(source(i), str(SRC), 'exec'), g); g['selftest']()


def main():
    bad = 0
    for i, (name, *_) in enumerate(MUT):
        r = subprocess.run([sys.executable, __file__, str(i)], capture_output=True, text=True)
        tail = r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or ['']
        caught = r.returncode == 1 and 'PIPELINE PAIRED EVAL SELFTEST FAILED' in r.stderr   # caught BY A CHECK (the selftest's own exit message); a crash is unexpected
        ok = (r.returncode == 0) if i == 0 else caught
        bad += not ok
        first = next((x for x in r.stderr.splitlines() if 'SELFTEST FAILED' in x), None)
        how = 'passed' if r.returncode == 0 else ('caught: ' + first.strip()[:200]) if caught and first else ' / '.join(x.strip()[:200] for x in tail[-2:])
        print(f'{i:2d} {name}: rc {r.returncode} -> {how}{"" if ok else "   <<< UNEXPECTED"}', flush=True)
    if bad:
        sys.exit(f'PIPELINE PAIRED EVAL MUTATION CHECK FAILED: {bad} unexpected outcomes of {len(MUT)}')
    print(f'pipeline paired eval mutation check passed: the unchanged copy passes its selftest, {len(MUT) - 1} / {len(MUT) - 1} planted code errors are caught by a check', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['targets']:
        [source(i) for i in range(len(MUT))]; print(f'all {len(MUT) - 1} mutation targets occur as declared')
    elif len(sys.argv) > 1:
        one(int(sys.argv[1]))
    else:
        main()
