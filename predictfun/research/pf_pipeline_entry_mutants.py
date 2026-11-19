"""MUTATION CHECK of the entry test's selftest (2026-11-19, session aea5bfb9): a checker that cannot fail proves nothing. Each entry plants one code
error in a COPY of pf_pipeline_entry.py (text replacement, executed in a subprocess; the repo file is never touched) and runs the copy's selftest
(synthetic tape only). Required: the unchanged copy passes, every mutant is caught BY A CHECK (the selftest's own failure message; a crash is
unexpected), and every replacement target occurs exactly as often as declared (a stale target is a failure). No market data is read.
Usage (run_light, PF_SLOT=claude): pf_pipeline_entry_mutants.py [index | targets]
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / 'pf_pipeline_entry.py'
# (name, text in the source, replacement[, occurrences]) - entry 0 is the unchanged copy
MUT = [
    ('none (must pass)', None, None),
    # --- the controls
    ('Q0a never refuses', "            if not close(res['POST-ALL'][nm][c], ref_[c]):", "            if False:"),
    ('Q0a ignores the prediction file', "    if pj['inputs']['predictions'] != (C.sha(pf_) if pf_.exists() else None):", "    if False:"),
    ('Q0b never refuses', "                if not close(res[r][nm][c], D[c].values[m].sum()):", "                if False:"),
    ('Q0c never refuses', "    if not (close(sum(gsum), dif.sum()) and gsum[0] == 0. and all(close(sum(wsum[c]), Db[c].sum()) for c in CANCEL)):", "    if False:"),
    ('Q0d never refuses', "    if not (par['rows'] == len(pp) and par['rows'] > par['refused'] and par['verdict_disagreements'] == 0 and exec_cmp == par['posts']):", "    if False:"),
    ('Q0d ignores the round trip', "        coherent = accepted and rt; ", "        coherent = accepted; "),
    ('Q0d compares the verdicts only', " and exec_cmp == par['posts']):", "):"),
    ('Q0e never refuses (the POST rows)', "    if not (got == want == with_path and len(pp) == len(got)):", "    if False:"),
    ('Q0e never refuses (the labels)', "    if not (np.allclose(Db.KEEP.values, Db.label.values, rtol=0, atol=1e-6) and np.allclose(", "    if False and not (np.allclose(Db.KEEP.values, Db.label.values, rtol=0, atol=1e-6) and np.allclose("),
    # --- the entry rule, the valuation and the null
    ('the rule posts on refused forecasts', "    s = p[acts if RAW_ENTRY else (acts & p.J_ok.astype(bool))]", "    s = p[acts]"),
    ('the matched null is global', "    if NULL_GLOBAL:", "    if True:"),
    ('the null weight over all opportunities', "    return (k / n)[inv]", "    return (k / n.sum())[inv]"),
    ('MODEL-J removal from the decision, not its landing', "    rj = KP.landing(np.clip(T[ri], 1, KP.NSTEP), lf) + LAND_SHIFT", "    rj = KP.STEP * np.clip(T[ri], 1, KP.NSTEP) + LAND_SHIFT"),
    ('PULL-ALL from the second live step', "j0 = KP.landing(np.full(len(ri), KP.J0), lf)", "j0 = KP.landing(np.full(len(ri), KP.J0 + 1), lf)"),
    ('MODEL-J pulls that do not execute counted', " & ((T < endj) | ACT_ANY)", ""),
    ('Q0b pull count never refuses', "            if int(D.act.values[m].sum()) != res[r][nm]['pulls']:", "            if False:"),
    ('the first window ends at step 1', "win = np.where(t < KP.STEP * KP.J0, 0,", "win = np.where(t < KP.STEP, 0,"),
    ('the age from submission, not activation', "age = np.digitize(t - float(sc['A']), SC.AGE_EDGES)", "age = np.digitize(t, SC.AGE_EDGES)"),
    # --- the readings and tables
    ('N1 read on POST-ALL', "(('N1 V(MODEL-POST, MODEL-J) vs NO-ENTRY', V_('MODEL-POST', 'MODEL-J'), 'NO-ENTRY')", "(('N1 V(MODEL-POST, MODEL-J) vs NO-ENTRY', V_('POST-ALL', 'MODEL-J'), 'NO-ENTRY')"),
    ('N2 read under KEEP', "('N2 V(MODEL-POST, PULL-ALL) vs NO-ENTRY', V_('MODEL-POST', 'PULL-ALL'), 'NO-ENTRY')", "('N2 V(MODEL-POST, PULL-ALL) vs NO-ENTRY', V_('MODEL-POST', 'KEEP'), 'NO-ENTRY')"),
    ('N5 without the reference', "{nm: res['MODEL-POST'][nm]['MODEL-J'] - res['POST-ALL'][nm]['MODEL-J'] for nm in nul}", "{nm: res['MODEL-POST'][nm]['MODEL-J'] for nm in nul}"),
    ('E_entry against zero instead of the null', "Ent = lambda c: {nm: res['MODEL-POST'][nm][c] - nul[nm][c] for nm in nul}", "Ent = lambda c: {nm: res['MODEL-POST'][nm][c] for nm in nul}"),
    ('the error split without the spread', "error_edge_term=float((e_ * (g.J_Eq.values - g.q_m50.values)).sum())", "error_edge_term=float((g.J_Eq.values - g.q_m50.values).sum())"),
    # --- G1 (plan 0c3d2523) and the v3.1 parity
    ('Q0f never refuses', "            if not all(np.allclose(D[f'KEEP_w{w}'].values, D[f'{c}_w{w}'].values, rtol=0, atol=1e-9) for c in CANCEL):", "            if False:"),
    ('Q0g never refuses', "            if not (len(adm[r]) == res[r][nm]['eligible'] + n_ref and int(m.sum()) == res[r][nm]['eligible'] and not D.refused_s.values[m].any()):", "            if False:"),
    ('the parity rebuilds E[q | fill] without the snap', "            qf = rem if rem < qf <= rem + tol(rem) else qf ", "            qf = qf "),
    ('the live end ignores full fills', "full = np.where((tf < rc[c]) & (tf < ld) & (tf <= KP.H), tf, np.inf)", "full = np.full(n, np.inf)"),
    ('shares kept ignore the cancellation', "qk = Fs.q.values * k; pm", "qk = Fs.q.values * 1.; pm"),
    ('availability from submission, not activation', "live = np.maximum(Db[f'end_{c}'].values - A_, 0.)", "live = np.maximum(Db[f'end_{c}'].values, 0.)"),
    ('first-review reversals over every order', "live_j0 = ms & (Db.endj.values > KP.J0); first =", "live_j0 = ms; first ="),
    ('attempts counted from the activated orders', "att = adm_n[r][sd]; acc", "att = int(ms.sum()); acc"),
    ('the tail without the refused episodes', "ep = np.r_[vals, np.zeros(att - acc)] if att >= acc else vals", "ep = vals"),
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
    g = dict(__name__='pf_pipeline_entry_mutant', __file__=str(SRC)); exec(compile(source(i), str(SRC), 'exec'), g); g['selftest']()


def main():
    bad = 0
    for i, (name, *_) in enumerate(MUT):
        r = subprocess.run([sys.executable, __file__, str(i)], capture_output=True, text=True)
        tail = r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or ['']
        caught = r.returncode == 1 and 'PIPELINE ENTRY SELFTEST FAILED' in r.stderr   # caught BY A CHECK (the selftest's own exit message); a crash is unexpected
        ok = (r.returncode == 0) if i == 0 else caught
        bad += not ok
        first = next((x for x in r.stderr.splitlines() if 'SELFTEST FAILED' in x), None)
        how = 'passed' if r.returncode == 0 else ('caught: ' + first.strip()[:200]) if caught and first else ' / '.join(x.strip()[:200] for x in tail[-2:])
        print(f'{i:2d} {name}: rc {r.returncode} -> {how}{"" if ok else "   <<< UNEXPECTED"}', flush=True)
    if bad:
        sys.exit(f'PIPELINE ENTRY MUTATION CHECK FAILED: {bad} unexpected outcomes of {len(MUT)}')
    print(f'pipeline entry mutation check passed: the unchanged copy passes its selftest, {len(MUT) - 1} / {len(MUT) - 1} planted code errors are caught by a check', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['targets']:
        [source(i) for i in range(len(MUT))]; print(f'all {len(MUT) - 1} mutation targets occur as declared')
    elif len(sys.argv) > 1:
        one(int(sys.argv[1]))
    else:
        main()
