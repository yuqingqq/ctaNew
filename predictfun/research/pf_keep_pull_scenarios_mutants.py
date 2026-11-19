"""MUTATION CHECK of the keep / pull scenario replay's selftest (2026-11-18, session aea5bfb9; v2: the review of 1fd77004): a checker that cannot fail proves nothing.
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
    ('the score lands at the baseline latency in every scenario', "r = KP.landing(np.clip(T[ri], 1, KP.NSTEP), lf)", "r = KP.landing(np.clip(T[ri], 1, KP.NSTEP))"),
    ('the uniform shifts read the baseline fills', "Fs = Fl[Fl.clock == float(sc['tl'][1:])][['day', 'oid', 't', 'q', 'src']]", "Fs = Fl[Fl.clock == SHIFT][['day', 'oid', 't', 'q', 'src']]"),
    ('notice scenarios re-use the baseline eligibility', "endj = endj_n(E.t_full_s.values, E.g_struct.values, sc['N'])", "endj = endj_n(E.t_full_s.values, E.g_struct.values, KP.NOTICE)"),
    # --- the controls
    ('baseline control removed', "C.require(same, f'the baseline scenario differs from the committed replay (order {int(o.oid)})')",
     "C.require(True, f'the baseline scenario differs from the committed replay (order {int(o.oid)})')"),
    ('a fill at / after the structural landing accepted', "C.require(not (fq[tt < a_] > 1e-12).any() and not (np.isfinite(lnd) and (fq[tt >= lnd] > 1e-12).any()),",
     "C.require(True or not (fq[tt < a_] > 1e-12).any() and not (np.isfinite(lnd) and (fq[tt >= lnd] > 1e-12).any()),"),
    ('B checked against the clock audit removed', "    if not ok_h:\n", "    if False:\n"),
    # --- the review of 1fd77004: S3 timelines, held fills and denominators, families
    ('S3 timelines read the S1 windows', "hold = EVT[f'{k}_st'].values >= 2; lo, hi = EVT[f'{k}_lo'].values, EVT[f'{k}_hi'].values; held[k] = int(hold.sum())",
     "hold = EVT[f'{k}_st'].values >= 2; lo, hi = EVT['S1_lo'].values, EVT['S1_hi'].values; held[k] = int(hold.sum())"),
    ('S3 legs joined by position, not by row id', "L['S3_llo'], L['S3_lhi'] = sem.S3_llo.reindex(L.row_id.values).values, sem.S3_lhi.reindex(L.row_id.values).values",
     "L['S3_llo'], L['S3_lhi'] = sem.S3_llo.values[:len(L)], sem.S3_lhi.values[:len(L)]"),
    ('held fills not marked', "mine = fq > 1e-12; hc = held_code(tt[mine] + t, hm[sc['tl']][s])", "mine = fq > 1e-12; hc = np.full(int(mine.sum()), 2, np.int8)"),
    ('held legs left out of the hold map', "hl[k + 'lo'] = hl[k + 'hi'] = hold[e]", "hl[k + 'lo'] = hl[k + 'hi'] = np.zeros(len(L), bool)"),
    ('held value counted over every fill', "hv = np.isin(Fs.held.values, (1, 3)) if 'held' in Fs else np.zeros(len(Fs), bool)", "hv = np.ones(len(Fs), bool)"),
    ('the event denominator not counted', "st['events'] += nev", "st['events'] += 0"),
    ('held events of a family not counted', "st[f'held_events_{k}'] += held[k]", "st[f'held_events_{k}'] += 0"),
    ('a trade-clock shift labelled a cancel latency', "return 'trade-clock shift', f\"every trade stamp", "return 'cancel latency', f\"every trade stamp"),
    # --- settlement times: ST timelines, the cancel-latency mixture
    ('ST timelines read the S3 windows', "L['ST_llo'], L['ST_lhi'], _ = X.st_windows(tset.t_set_ms.reindex(L.row_id.values).values, L.S3_llo.values, L.S3_lhi.values, lag)",
     "L['ST_llo'], L['ST_lhi'] = L.S3_llo.values, L.S3_lhi.values"),
    ('the mixture ignored in the kernel', "a_, l_ = sc['A'], lat_of(sc, int(o.oid));", "a_, l_ = sc['A'], sc['L'];"),
    ('the mixture ignored in the outcome', "lf = np.where(slow_order(E.oid.values[ri], sc['X'][0]), sc['X'][1], sc['L']) if sc.get('X') else np.full(len(ri), float(sc['L']))",
     "lf = np.full(len(ri), float(sc['L']))"),
    ('every order slow', "    return ((np.asarray(oid, np.int64) * 2654435761) % 2 ** 32) / 2. ** 32 < share", "    return np.ones(np.shape(oid), bool) | (share < 0)"),
    # --- check of 9a114cc1: the production (upper-only) default path
    ('the production day ignores the corrected known receipt', "        st_lag, lag_in = X.known_lag()", "        st_lag, lag_in = 259.034, {}"),
    ('an upper-only ST window replayed as STlo', "del tau['STlo'], hl['STlo']", "pass"),
    # --- v5: the C1 age split and the C3 cancel-pending share
    ('the age taken from posting, not activation', "yb = np.digitize(t - float(sc['A']), AGE_EDGES)", "yb = np.digitize(t, AGE_EDGES)"),
    ('removed fills counted as cancel-pending', "pend = act[ri] & (t >= KP.STEP * T[ri]) & (t < r)", "pend = act[ri] & (t >= KP.STEP * T[ri])"),
    ('the headline without young fills keeps them', "no_young=dict(E=e - by_age['lt100']['E'], score=score - by_age['lt100']['score'],", "no_young=dict(E=e, score=score,"),
    ('control (i) never fires', "    return [nm for nm, v in res.items() if abs(v['KEEP'] - v['fills_value']) > 1e-6 * max(1., abs(v['KEEP']))]", "    return []"),
    # --- v6: C12 trade-anchor sources, control (j), the C2 reading
    ('every fill labelled mixed', "sx.append(cs.pop() if len(cs) == 1 else 3)", "sx.append(3)"),
    ('control (j) never compares the sources', "and np.array_equal(np.asarray(sx, np.int64), cs_))", "and True)"),
    ('the age-by-source table not a partition', "E=con((yb == b) & (sc_ == c)), fills=int(((yb == b) & (sc_ == c)).sum())", "E=con((yb == b) & (sc_ >= c)), fills=int(((yb == b) & (sc_ >= c)).sum())"),
    ('the M250 reading always holds', "    return 'HOLDS-AT-M250' if np.sign(v50) == np.sign(v250) and v50 != 0 else 'SIGN-DIFFERS-AT-M250'", "    return 'HOLDS-AT-M250'"),
    # --- v6: C1 B1a-weighted headline
    ('B1a weights applied in reverse bin order', "    return {k: float(sum(by_age[lab]['E'] * w for lab, w in zip(AGE_LABELS, wt))) for k, wt in FV.B1A.items()}",
     "    return {k: float(sum(by_age[lab]['E'] * w for lab, w in zip(AGE_LABELS, wt[::-1]))) for k, wt in FV.B1A.items()}"),
    ('the report not given the B1a values', "                b1a_E=b1a_of(by_age),", "                b1a_E=dict(M0=0., M50=0.),"),
    ('the B1a reading ignores a sign change', "    return 'HOLDS-SIGN' if e != 0 and all(np.sign(x) == np.sign(e) for x in w.values()) and len(w) > 0 else 'SIGN-DIFFERS'",
     "    return 'HOLDS-SIGN' if e != 0 and len(w) > 0 else 'SIGN-DIFFERS'"),
    # --- v6.1: the score policy as a parameter (pf_keep_pull_variants.py)
    ('the policy parameter ignored (every policy valued as A+0)', "E.side.values, E.p.values)[pol]", "E.side.values, E.p.values)[KP.DECLARED]"),
    ('an undeclared policy accepted', "C.require(pol in KP.POLICIES, f'undeclared score policy {pol}')", "pass"),
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
        caught = r.returncode == 1 and 'KEEP-PULL SCENARIOS SELFTEST FAILED' in r.stderr   # caught BY A CHECK (the selftest's own exit message); a crash is unexpected
        ok = (r.returncode == 0) if i == 0 else caught
        bad += not ok
        first = next((x for x in r.stderr.splitlines() if 'SELFTEST FAILED' in x), None)
        how = 'passed' if r.returncode == 0 else ('caught: ' + first.strip()[:200]) if caught and first else ' / '.join(x.strip()[:200] for x in tail[-2:])
        print(f'{i:2d} {name}: rc {r.returncode} -> {how}{"" if ok else "   <<< UNEXPECTED"}', flush=True)
    if bad:
        sys.exit(f'KEEP-PULL SCENARIO MUTATION CHECK FAILED: {bad} unexpected outcomes of {len(MUT)}')
    print(f'keep-pull scenario mutation check passed: the unchanged copy passes its selftest, {len(MUT) - 1} / {len(MUT) - 1} planted code errors are caught by a check', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['targets']:
        [source(i) for i in range(len(MUT))]; print(f'all {len(MUT) - 1} mutation targets occur as declared')
    elif len(sys.argv) > 1:
        one(int(sys.argv[1]))
    else:
        main()
