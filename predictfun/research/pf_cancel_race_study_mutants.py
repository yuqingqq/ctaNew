"""MUTATION CHECK of the per-cancel race study's selftest (2026-11-18, session aea5bfb9): a checker that cannot fail proves nothing. Each entry
plants one code error in a COPY of pf_cancel_race_study.py (text replacement, executed in a subprocess; the repo file is never touched) and runs
the copy's selftest (synthetic only). Required: the unchanged copy passes; every mutant is caught BY A CHECK (rc 1 and 'SELFTEST FAIL' printed -
a crash is an unexpected outcome, not a kill); every replacement target occurs exactly as often as declared. No market data is read.
Usage (run_light, PF_SLOT=claude): pf_cancel_race_study_mutants.py [index | targets]
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / 'pf_cancel_race_study.py'
MUT = [
    ('none (must pass)', None, None),
    ('the cutoff not applied', "            if at >= cutoff_ms * 1e6:\n                stopped = True; break", "            if False:\n                stopped = True; break"),
    ('a regressing clock accepted', "C.require(at >= last, 'kernel UTC clock regressed')", "C.require(True, 'kernel UTC clock regressed')"),
    ("a fill keyed by its trade alone (a sibling order's fill counted)", "            c = o2c.get(tuple(x['order_id']))\n            if c is None:\n                continue",
     "            c = o2c.get(tuple(x['order_id']), next(iter(o2c.values())) if o2c else None)\n            if c is None:\n                continue"),
    ('a request of an unplaced cid not counted', "                st['request_of_a_cid_never_placed'] += 1", "                pass"),
    ('the Matched second taken from a Confirmed notice', "m2 = min(m_, x['match_time'] / 1e9) if x['status'] == 'Matched' else m_", "m2 = min(m_, x['match_time'] / 1e9)"),
    ('the lower status bound from K7.lo', "lb = m_ * 1000. - k7['hi'] if mf else np.nan", "lb = m_ * 1000. - k7['lo'] if mf else np.nan"),
    ('the upper status bound from K7.hi', "ub = (m_ + 1.) * 1000. - k7['lo'] if mf else np.nan", "ub = (m_ + 1.) * 1000. - k7['hi'] if mf else np.nan"),
    ('the notice bound ignored', "    ub = min(ub_n, ub_s) if np.isfinite(ub_s) else ub_n", "    ub = ub_s if np.isfinite(ub_s) else np.inf"),
    ('every before-request bound typed EVIDENCE', "return 'BEFORE-REQUEST', ('EVIDENCE' if ub_n < t_req else 'CONDITIONAL')", "return 'BEFORE-REQUEST', 'EVIDENCE'"),
    ('after-answer tested against the request', "if np.isfinite(t_ans) and t_ans >= t_req and np.isfinite(lb_s) and lb_s > t_ans:",
     "if np.isfinite(t_ans) and t_ans >= t_req and np.isfinite(lb_s) and lb_s > t_req:"),
    ('after-request never tested', "    if np.isfinite(lb_s) and lb_s > t_req:\n        return 'AFTER-REQUEST', 'CONDITIONAL'", "    if False:\n        return 'AFTER-REQUEST', 'CONDITIONAL'"),
    ('the order class from its first fill', "ocls = next((c_ for c_ in ORDER_RANK if c_ in fl), None) or", "ocls = (fl[0] if fl else None) or"),
    ('the priority reversed', "ORDER_RANK = ('AFTER-ANSWER', 'AFTER-REQUEST', 'UNORDERED', 'BEFORE-REQUEST')", "ORDER_RANK = ('BEFORE-REQUEST', 'UNORDERED', 'AFTER-REQUEST', 'AFTER-ANSWER')"),
    ('censoring ignored', "('NO-FILL' if h >= H_MIN else 'CENSORED')", "'NO-FILL'"),
    ('the first confirmation from the last answer', "cls = TCA.answer_class(conf[0][2], conf[0][1] == 'pending_audit') if conf else 'NO-ANSWER'; t_ans = conf[0][0] if conf else np.nan",
     "cls = TCA.answer_class(conf[-1][2], conf[-1][1] == 'pending_audit') if conf else 'NO-ANSWER'; t_ans = conf[-1][0] if conf else np.nan"),
    ('the pending audit not a confirmation', "+ [(t, 'pending_audit', None) for t in o['pa']], key=lambda z: z[0])", ", key=lambda z: z[0])"),
    ('the request from the last retry', "        t_req = o['reqs'][0]\n", "        t_req = o['reqs'][-1]\n"),
    ('K9 contradiction never flagged', "contra = bool(kk) and fc in ('AFTER-ANSWER', 'AFTER-REQUEST')", "contra = False"),
    ('C3 always NO', "                                     if aa.any() else 'NO')", "                                     if False else 'NO')"),
    ('C5 never compares the answer receipt', "('answer_at', eq(float(r.answer_at), float(r.t_ans)))", "('answer_at', True)"),
    ('an empty K7 accepted', "and k7['lo'] <= k7['hi'], 'K7 missing or empty", "and True, 'K7 missing or empty"),
    # --- v2: outbox sends, the first Canceled confirmation (C3b), C5 against the case table's requests
    ('outbox sends never read', "                if last and b'\"type\":\"outbox\"' in line[-60:] and b'\"cancels\":[]' not in line:", "                if False:"),
    ('C3b against the first confirmation of any class', "cc = [t for t, kd, a in conf if TCA.answer_class(a, kd == 'pending_audit') == 'CANCELED' and t >= t_req]",
     "cc = [t for t, kd, a in conf if t >= t_req]"),
    ('C3b always NO', "                                                       if ac.any() else f\"NO (CONDITIONAL on K7)", "                                                       if False else f\"NO (CONDITIONAL on K7)"),
    ("C5 ignores the case table's requests", "        asked = (r.run, int(r.cid)) in requested", "        asked = True"),
]
STALE = 97


def source(i):
    name, a, b = MUT[i][:3]; n = (MUT[i] + (1,))[3]; s = SRC.read_text()
    if a is not None:
        if s.count(a) != n:
            print(f'mutation target occurs {s.count(a)} times, declared {n}: {name}', file=sys.stderr); sys.exit(STALE)
        s = s.replace(a, b)
    return s


def one(i):
    g = dict(__name__='pf_cancel_race_study_mutant', __file__=str(SRC)); exec(compile(source(i), str(SRC), 'exec'), g)
    sys.exit(0 if g['selftest']() else 1)


def main():
    bad = 0
    for i, (name, *_) in enumerate(MUT):
        r = subprocess.run([sys.executable, __file__, str(i)], capture_output=True, text=True)
        tail = r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or ['']
        ok = (r.returncode == 0) if i == 0 else (r.returncode == 1 and 'SELFTEST FAIL' in r.stdout)
        bad += not ok
        how = 'passed' if r.returncode == 0 else ('caught: ' + r.stdout.strip().splitlines()[-1][:300]) if 'SELFTEST FAIL' in r.stdout else ' / '.join(x.strip()[:200] for x in tail[-2:])
        print(f'{i:2d} {name}: rc {r.returncode} -> {how}{"" if ok else "   <<< UNEXPECTED"}', flush=True)
    if bad:
        sys.exit(f'CANCEL RACE STUDY MUTATION CHECK FAILED: {bad} unexpected outcomes of {len(MUT)}')
    print(f'cancel race study mutation check passed: the unchanged copy passes its selftest, {len(MUT) - 1} / {len(MUT) - 1} planted code errors are caught by a check', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['targets']:
        [source(i) for i in range(len(MUT))]; print(f'all {len(MUT) - 1} mutation targets occur as declared')
    elif len(sys.argv) > 1:
        one(int(sys.argv[1]))
    else:
        main()
