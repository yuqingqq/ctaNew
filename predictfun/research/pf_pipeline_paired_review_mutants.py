"""MUTATION CHECK of the four-part review's selftest (2026-11-19, session aea5bfb9): a checker that cannot fail proves nothing. Each entry plants one code
error in a COPY of pf_pipeline_paired_review.py (text replacement, executed in a subprocess; the repo file is never touched) and runs the copy's selftest
(synthetic tape only). Required: the unchanged copy passes, every mutant is caught BY A CHECK (the selftest's own failure message; a crash is
unexpected), and every replacement target occurs exactly as often as declared (a stale target is a failure). No market data is read.
Usage (run_light, PF_SLOT=claude): pf_pipeline_paired_review_mutants.py [index | targets]
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / 'pf_pipeline_paired_review.py'
# (name, text in the source, replacement[, occurrences]) - entry 0 is the unchanged copy
MUT = [
    ('none (must pass)', None, None),
    # --- the validity predicates
    ('V1 never fails', "mismatches=nbad, verdict='PASS' if (nbad == 0 and chk > 0) else 'FAIL')", "mismatches=nbad, verdict='PASS')"),
    ('V1 checks the RETAIN value only', "bad += not (abs(float(f.v[m].sum()) - r.pair_v) <= 1e-6 and abs(float(f.q[m].sum()) - r.pair_q) <= 1e-6)", "bad += not (abs(float(f.v[m].sum()) - r.pair_v) <= 1e-6)"),
    ('V1 lets NaN pass', "bad += not (abs(float(f.v[m].sum()) - r.pair_v) <= 1e-6 and abs(float(f.q[m].sum()) - r.pair_q) <= 1e-6)", "bad += abs(float(f.v[m].sum()) - r.pair_v) > 1e-6 or abs(float(f.q[m].sum()) - r.pair_q) > 1e-6"),
    ('V2 lets NaN pass', "(~(np.abs(lab.pair_v.values - (sg * (lab.pf_mid.values.astype(float) - lab.p.values.astype(float)) * lab.pair_q.values + lab.pair_qm.values)) <= 1e-6)).sum()",
     "(np.abs(lab.pair_v.values - (sg * (lab.pf_mid.values.astype(float) - lab.p.values.astype(float)) * lab.pair_q.values + lab.pair_qm.values)) > 1e-6).sum()"),
    ('V2 never fails', "failures=n2, verdict='PASS' if n2 == 0 else 'FAIL')", "failures=n2, verdict='PASS')"),
    ('V3 ignores the effective time', " & np.isclose(ex.effective_ms.values - t, eff, atol=1e-6)", ""),
    ('V4 ignores reached rows of unposted orders', " and pop['A_reached_orders_not_admitted'] == 0)", ")"),
    ('V4 drops the two-source original count', "v4 = (two_orig and two_path and labp", "v4 = (two_path and labp"),
    ('V4 drops the two-source with-path count', "v4 = (two_orig and two_path and labp", "v4 = (two_orig and labp"),
    ('V5 never fails', "verdict='PASS' if (nacc == 0 and len(acc) > 40) else 'FAIL'", "verdict='PASS'"),
    ('V5 at a relative tolerance', "acc[f'{p_} {sd}'] = (a, b, abs(a - b) <= 1e-6)", "acc[f'{p_} {sd}'] = (a, b, abs(a - b) <= 1e-6 * max(1., abs(a), abs(b)))"),
    ('V5 ignores the evaluation inputs', "acc[f'evaluation input {k_}'] = (ins.get(k_, 'missing'), w_, ins.get(k_, 'missing') == w_)", "acc[f'evaluation input {k_}'] = (ins.get(k_, 'missing'), w_, True)"),
    ('V5 ignores the null mean', "abs(nullr['both']['null_mean'] - nA) <= 1e-6)", "True)"),
    ('V5 ignores the evaluation source', "ev.get('source_sha256') == es)", "True)"),
    # --- the matched null, nesting, first review, FINDINGS C1 / C3 / C12
    ('the null variance without the finite-population factor', "var = float((k * (n - k) / (n * (n - 1)).where(n > 1, 1.) * ss).where(n > 1, 0.).sum())", "var = float((k / n * ss).where(n > 1, 0.).sum())"),
    ('the null mean over every opportunity', "return float((k * mc).sum()), var", "return float((n * mc).sum()), var"),
    ('the draws choose one too many per cell', "m = (pos - start[ci]) < kc[ci]", "m = (pos - start[ci]) <= kc[ci]"),
    ('the draws ignore the cells', "pos[np.lexsort((rng.random(n), ci))] = np.arange(n)", "pos[np.argsort(rng.random(n))] = np.arange(n)"),
    ('the draws-vs-moments check never fails', "return bool(abs(float(d.mean()) - e0) <= 4. * math.sqrt(v0 / len(d)) and .8 <= float(d.var(ddof=1)) / v0 <= 1.25)", "return True"),
    ('the spread verdict at 0.5', "return 'PASS' if p_draws <= .05 else 'FAIL'", "return 'PASS' if p_draws <= .5 else 'FAIL'"),
    ('the first-review share counts every pull', "return float((m & D.act.values & (D['T'].values == KP.J0)).sum() / m.sum()) if m.any() else None", "return float((m & D.act.values).sum() / m.sum()) if m.any() else None"),
    ('nesting ignores the quantities', "(DA.q.values[m] >= DP.q.values[m] - 1e-9).all() and ", ""),
    ('the nesting status always NOT-NESTED', "'IMPLIED-BY-NESTING (not evidence)' if nest else 'NOT-NESTED'", "'NOT-NESTED'"),
    ('C1 without the 100-500 ms bin instead', "without_young=V_ - va[0]", "without_young=V_ - va[1]"),
    ('C3 share over the net value', "share_of_gross=float(np.abs(D.pend_v).sum() / D.gross.sum())", "share_of_gross=float(np.abs(D.pend_v).sum() / abs(D.V.sum()))"),
    ('the ledger age from the post time', "age = np.digitize(t - float(sc['A']), SC.AGE_EDGES)", "age = np.digitize(t, SC.AGE_EDGES)"),
    ('the ledger pending after the landing', "pm = kept & (t >= req[ri]) & (t < land[ri])", "pm = kept & (t >= land[ri])"),
    ('the ledger sources shifted', "D[f'v_s{c_}'] = np.bincount(ri, vv * (src == c_), n)", "D[f'v_s{c_}'] = np.bincount(ri, vv * (src == (c_ + 1) % len(SC.SRC_LABELS)), n)"),
    # --- the ledger
    ('the ledger ignores the cancellation', "kept = a[ri] & el[ri] & ~(act[ri] & (t >= land[ri]))", "kept = a[ri] & el[ri]"),
    ('the ledger ignores the admissions', "kept = a[ri] & el[ri] & ~(act[ri] & (t >= land[ri]))", "kept = el[ri] & ~(act[ri] & (t >= land[ri]))"),
    ('the first window ends at step 1', "win = np.where(t < KP.STEP * KP.J0, 0,", "win = np.where(t < KP.STEP, 0,"),
    ('the spread component from the END mark', "edge = sg * (E.pf_mid_post.values.astype(float) - E.p.values.astype(float))", "edge = sg * (E.mark.values - E.p.values.astype(float))"),
    # --- the prediction groups and screens
    ('the episode-balanced mean is the raw mean', "return float(pd.Series(np.asarray(x, float)).groupby(np.asarray(oid)).mean().mean()) if len(x) else float('nan')", "return float(np.mean(x)) if len(x) else float('nan')"),
    ('the screen without its support floor', "    if n_ep < 200 or nz < 20:", "    if False:"),
    ('the screen ignores the label sign', "ok = (ml > 0 if want_pos else ml < 0) and abs(mp - ml) <= .05 + .20 * abs(ml)", "ok = abs(mp - ml) <= .05 + .20 * abs(ml)"),
    ('NO_ENTRY counts the ties', "'NO_ENTRY (pred < 0)': (po, pp, pl, np.isfinite(pp) & (pp < 0), False)", "'NO_ENTRY (pred < 0)': (po, pp, pl, np.isfinite(pp) & (pp <= 0), False)"),
    # --- the tails
    ('the severe excess without its threshold', "S = np.maximum(-V_ - KP.SEVERE, 0.); n = len(V_)", "S = np.maximum(-V_, 0.); n = len(V_)"),
    ('the worst 1% share over the top 10%', "k = max(int(np.ceil(.01 * n)), 1)", "k = max(int(np.ceil(.1 * n)), 1)"),
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
    g = dict(__name__='pf_pipeline_paired_review_mutant', __file__=str(SRC)); exec(compile(source(i), str(SRC), 'exec'), g); g['selftest']()


def main():
    bad = 0
    for i, (name, *_) in enumerate(MUT):
        r = subprocess.run([sys.executable, __file__, str(i)], capture_output=True, text=True)
        tail = r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or ['']
        caught = r.returncode == 1 and 'PIPELINE PAIRED REVIEW SELFTEST FAILED' in r.stderr   # caught BY A CHECK (the selftest's own exit message); a crash is unexpected
        ok = (r.returncode == 0) if i == 0 else caught
        bad += not ok
        first = next((x for x in r.stderr.splitlines() if 'SELFTEST FAILED' in x), None)
        how = 'passed' if r.returncode == 0 else ('caught: ' + first.strip()[:200]) if caught and first else ' / '.join(x.strip()[:200] for x in tail[-2:])
        print(f'{i:2d} {name}: rc {r.returncode} -> {how}{"" if ok else "   <<< UNEXPECTED"}', flush=True)
    if bad:
        sys.exit(f'PIPELINE PAIRED REVIEW MUTATION CHECK FAILED: {bad} unexpected outcomes of {len(MUT)}')
    print(f'pipeline paired review mutation check passed: the unchanged copy passes its selftest, {len(MUT) - 1} / {len(MUT) - 1} planted code errors are caught by a check', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['targets']:
        [source(i) for i in range(len(MUT))]; print(f'all {len(MUT) - 1} mutation targets occur as declared')
    elif len(sys.argv) > 1:
        one(int(sys.argv[1]))
    else:
        main()
