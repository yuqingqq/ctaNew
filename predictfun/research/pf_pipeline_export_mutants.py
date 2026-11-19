"""MUTATION CHECK of the pipeline export's selftest (2026-11-19, session aea5bfb9): a checker that cannot fail proves nothing. Each entry plants one
code error in a COPY of pf_pipeline_export.py (text replacement, executed in a subprocess; the repo file is never touched) and runs the copy's
selftest (synthetic tape only). Required: the unchanged copy passes, every mutant is caught BY A CHECK (the selftest's own failure message; a
crash is unexpected), and every replacement target occurs exactly as often as declared (a stale target is a failure). No market data is read.
Usage (run_light, PF_SLOT=claude): pf_pipeline_export_mutants.py [index | targets]
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / 'pf_pipeline_export.py'
# (name, text in the source, replacement[, occurrences]) - entry 0 is the unchanged copy
MUT = [
    ('none (must pass)', None, None),
    # --- the controls
    ('X1 never refuses', "    if not (st['post_rows'] == len(O) and st['retain_rows'] == want and st['ids_unique']):", "    if False:"),
    ('X2 never refuses', "        if not (abs(q_[0] - float(getattr(o, qc))) < 1e-7 and abs(v_[0] - float(getattr(o, vc))) < 1e-7):", "        if False:"),
    ('X3 never refuses', "            if not (n == n_ and abs(-v - imp) <= 1e-6 * max(1., abs(imp))):", "            if False:"),
    ('X3 compares the pull counts only', " and abs(-v - imp) <= 1e-6 * max(1., abs(imp))):", "):"),
    ('X4 never refuses', "                if not (np.array_equal(tt[mine], fl[0]) and np.array_equal(fq[mine], fl[1])):", "                if False:"),
    ('X5 causality never refuses', "                    if not (same and fired):", "                    if False:"),
    ('X5 positive control always fires', "    fired = not np.array_equal(feat_fn(fire(W, s, k), fair, depth, chosen)[s][k][0], before[0], equal_nan=True)", "    fired = True"),
    ('X5 future perturbation not applied', "    mut, fair2, dep2 = perturb(W, fair, depth, k, t, rng); after = feat_fn(mut, fair2, dep2, chosen)[s][k]", "    after = before"),
    ('X5 known-state parity never refuses', "        if not all(eq(float(ref[a]), float(vec[a][i])) for a in KNOWN):", "        if False:"),
    ('X6 never refuses', "    bad = [f'X6 hour {h} is sealed or forward' for h in hrs if not hour_ok(h)]", "    bad = []"),
    ('the hour rule lets sealed hours through', "    return not (I.SEAL_MS[0] <= h0 < I.SEAL_MS[1]) and h0 < I.FWD_MS", "    return h0 < I.FWD_MS"),
    ('a sampled check that never ran is accepted', "    if not (st['kernel_checked'] and st['known_checked'] and st['causal_checked']) and len(E):", "    if False:"),
    ('a refused day leaves its partial output', "        shutil.rmtree(part); sys.exit(f'{MARK} FAILED -> no output: '", "        sys.exit(f'{MARK} FAILED -> no output: '"),
    # --- the labels
    ('fill value with the wrong sign', "val = fq * sg * (float(o.mark) - p); q_, v_ = affected(ft, fq, val, eff)", "val = fq * sg * (p - float(o.mark)); q_, v_ = affected(ft, fq, val, eff)"),
    ('qm from the END mark instead of the causal mid', "        B[f'qm_{c}'] = B[f'value_{c}'] - sg * (ref - p) * B[f'q_{c}']", "        B[f'qm_{c}'] = B[f'value_{c}'] - sg * (B['mark'] - p) * B[f'q_{c}']"),
    ('severe excess without its threshold', "np.maximum(-v_ - KP.SEVERE, 0.)", "np.maximum(-v_, 0.)"),
    ('live flags of every clock from M50', "        endc = int(A.endj_of(np.array([float(getattr(o, tc))]), np.array([o.g_struct]))[0])",
     "        endc = int(A.endj_of(np.array([float(o.t_full)]), np.array([o.g_struct]))[0])"),
    ('RETAIN landing without the cancel latency', "KP.landing(js) + LANDING_OFFSET", "KP.STEP * js + LANDING_OFFSET"),
    ('a fill at the landing instant counted as unremovable', "i = np.searchsorted(ft, eff, side='left')", "i = np.searchsorted(ft, eff, side='right')"),
    # --- the inputs
    ('the fair edge from the posting fair price at every step', "getattr(o, f'f{x:02d}')", "getattr(o, 'f00')"),
    ('time to END from the absolute clock', "             t_to_end=END - (d - t))", "             t_to_end=END - d)"),
    ('known fills without the notice delay', "if a + notice <= x) for x in d], float)", "if a <= x) for x in d], float)"),
    ('the old 1 s exposure as the horizon', "SETTINGS = replace(FC.SETTINGS, exposure_ms=END)", "SETTINGS = FC.SETTINGS"),
    ('the queue ahead not min-updated', "    run = fin if NO_MINUPDATE else (np.minimum.accumulate(fin) if len(fin) else fin)", "    run = fin"),
    ('order inputs given the mid as the scale', "oi = FC.V.order_inputs(pd.DataFrame({f'x_{a}': kn[a] for a in KNOWN}), scale, ref)",
     "oi = FC.V.order_inputs(pd.DataFrame({f'x_{a}': kn[a] for a in KNOWN}), ref, ref)"),
    ('a refused post labelled eligible', "        st[0] = 'refused_post_only'", "        pass"),
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
    g = dict(__name__='pf_pipeline_export_mutant', __file__=str(SRC)); exec(compile(source(i), str(SRC), 'exec'), g); g['selftest']()


def main():
    bad = 0
    for i, (name, *_) in enumerate(MUT):
        r = subprocess.run([sys.executable, __file__, str(i)], capture_output=True, text=True)
        tail = r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or ['']
        caught = r.returncode == 1 and 'PIPELINE EXPORT SELFTEST FAILED' in r.stderr   # caught BY A CHECK (the selftest's own exit message); a crash is unexpected
        ok = (r.returncode == 0) if i == 0 else caught
        bad += not ok
        first = next((x for x in r.stderr.splitlines() if 'SELFTEST FAILED' in x), None)
        how = 'passed' if r.returncode == 0 else ('caught: ' + first.strip()[:200]) if caught and first else ' / '.join(x.strip()[:200] for x in tail[-2:])
        print(f'{i:2d} {name}: rc {r.returncode} -> {how}{"" if ok else "   <<< UNEXPECTED"}', flush=True)
    if bad:
        sys.exit(f'PIPELINE EXPORT MUTATION CHECK FAILED: {bad} unexpected outcomes of {len(MUT)}')
    print(f'pipeline export mutation check passed: the unchanged copy passes its selftest, {len(MUT) - 1} / {len(MUT) - 1} planted code errors are caught by a check', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['targets']:
        [source(i) for i in range(len(MUT))]; print(f'all {len(MUT) - 1} mutation targets occur as declared')
    elif len(sys.argv) > 1:
        one(int(sys.argv[1]))
    else:
        main()
