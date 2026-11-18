"""MUTATION CHECK of the keep / pull scenario replay's selftest (2026-11-18, session aea5bfb9): a checker that cannot fail proves nothing.
Each entry plants one code error in a COPY of pf_keep_pull_scenarios.py (text replacement, executed in a subprocess; the repo file is never
touched) and runs the copy's selftest (synthetic tape only). Required: the unchanged copy passes, every mutant fails, and every replacement
target occurs exactly as often as declared (a stale target is a failure, never a silent pass). No market data is read.
Usage (run_light, PF_SLOT=claude): pf_keep_pull_scenarios_mutants.py [index | targets]
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / 'pf_keep_pull_scenarios.py'
# (name, text in the source, replacement[, occurrences]) - entry 0 is the unchanged copy
MUT = [
    ('none (must pass)', None, None),
    # --- timelines
    ('a contradictory event moved to its (empty) window', "tau[k + 'lo'] = np.where(hold[e], m50, lo[e])", "tau[k + 'lo'] = lo[e]"),
    ('window end replaced by its start', "tau[k + 'hi'] = np.where(hold[e], m50, hi[e])", "tau[k + 'hi'] = np.where(hold[e], m50, lo[e])"),
    ('a contradictory event held at M0 instead of M50', "m50 = L.anc.values - SHIFT; tau = {'M50': m50}", "m50 = L.anc.values - 0.; tau = {'M50': L.anc.values - SHIFT}"),
    ('the evidence timeline at the window start', "tau['EVhi'] = EVT.EV_hi.values[e]", "tau['EVhi'] = EVT.S1_lo.values[e]"),
    ('every scenario re-uses the baseline path', "pk, _ = QP.build_path(Wk, t, t + KP.H, shift=0.)", "pk = base"),
    ('timing alignment check removed', "C.require(np.array_equal(legs[s].anc.values, np.asarray(W[s]['anc'], float)), 'timing legs not aligned with the window tape')", "pass"),
    # --- activation, cancels, notice
    ('activation ignored (always +58 ms)', "QP.order(side, 0, price, act=act)", "QP.order(side, 0, price, act=KP.PLACE)"),
    ('pending landing fixed at 95 ms', "fills_g(pk, s, p, a_, l_, a_ + l_ + PEND_OFFSET)", "fills_g(pk, s, p, a_, l_, KP.PENDING)"),
    ('kernel structural cancels at 37 ms in every scenario', "kernel_eval(p, o[a:a + 72], KP.H, cancel, pending)", "kernel_eval(p, o[a:a + 72], KP.H, KP.CANCEL, pending)"),
    ('structural landing recorded at the baseline latency', "lnd = eff_landing(g, a_, l_) + MIRROR_OFFSET", "lnd = float(KP.landing(g / KP.STEP)) + MIRROR_OFFSET"),
    ('pending landing from the baseline activation', "np.maximum(d + lat, act + lat + PEND_OFFSET)", "np.maximum(d + lat, KP.PLACE + lat + PEND_OFFSET)"),
    ('notice ignored in eligibility', "lim = np.fmin(g_struct, t_full + notice)", "lim = np.fmin(g_struct, t_full + KP.NOTICE)"),
    ('zero-latency tie order ignored (a cancel stops the trade at its own instant)', "    return x if x > g else float(np.nextafter(g, np.inf))", "    return x"),
    # --- the outcome
    ('the score lands at the baseline latency in every scenario', "r = KP.landing(np.clip(T[ri], 1, KP.NSTEP), sc['L'])", "r = KP.landing(np.clip(T[ri], 1, KP.NSTEP))"),
    ('the uniform shifts read the baseline fills', "Fs = Fl[Fl.clock == float(sc['tl'][1:])][['day', 'oid', 't', 'q']]", "Fs = Fl[Fl.clock == SHIFT][['day', 'oid', 't', 'q']]"),
    ('notice scenarios re-use the baseline eligibility', "endj = endj_n(E.t_full_s.values, E.g_struct.values, sc['N'])", "endj = endj_n(E.t_full_s.values, E.g_struct.values, KP.NOTICE)"),
    # --- the controls
    ('baseline control removed', "C.require(same, f'the baseline scenario differs from the committed replay (order {int(o.oid)})')",
     "C.require(True, f'the baseline scenario differs from the committed replay (order {int(o.oid)})')"),
    ('a fill at / after the structural landing accepted', "C.require(not (fq[tt < a_] > 1e-12).any() and not (np.isfinite(lnd) and (fq[tt >= lnd] > 1e-12).any()),",
     "C.require(True or not (fq[tt < a_] > 1e-12).any() and not (np.isfinite(lnd) and (fq[tt >= lnd] > 1e-12).any()),"),
    ('B checked against the clock audit removed', "    if not ok_h:\n", "    if False:\n"),
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
    g = dict(__name__='pf_keep_pull_scenarios_mutant', __file__=str(SRC)); exec(compile(source(i), str(SRC), 'exec'), g); g['selftest']()


def main():
    bad = 0
    for i, (name, *_) in enumerate(MUT):
        r = subprocess.run([sys.executable, __file__, str(i)], capture_output=True, text=True)
        tail = r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or ['']
        ok = (r.returncode == 0) if i == 0 else (r.returncode not in (0, STALE))
        bad += not ok
        print(f'{i:2d} {name}: rc {r.returncode} -> {"passed" if r.returncode == 0 else " / ".join(x.strip()[:200] for x in tail[-2:])}{"" if ok else "   <<< UNEXPECTED"}', flush=True)
    if bad:
        sys.exit(f'KEEP-PULL SCENARIO MUTATION CHECK FAILED: {bad} unexpected outcomes of {len(MUT)}')
    print(f'keep-pull scenario mutation check passed: the unchanged copy passes its selftest, {len(MUT) - 1} / {len(MUT) - 1} planted code errors fail it', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['targets']:
        [source(i) for i in range(len(MUT))]; print(f'all {len(MUT) - 1} mutation targets occur as declared')
    elif len(sys.argv) > 1:
        one(int(sys.argv[1]))
    else:
        main()
