"""MUTATION CHECK of the timing case audit's selftest (2026-11-18, session aea5bfb9; the review of 3bf8f3a9; v2: the input-journal key; v3 / v4: the census reuse and its lifecycle fixes): a checker that cannot fail proves
nothing. Each entry plants one code error in a COPY of pf_timing_case_audit.py (text replacement, executed in a subprocess; the repo file is never
touched) and runs the copy's selftest (synthetic only). Required: the unchanged copy passes; every mutant is caught BY A CHECK (rc 1 and
'SELFTEST FAIL' printed - a crash is an unexpected outcome, not a kill); every replacement target occurs exactly as often as declared (a stale
target is a failure, never a silent pass). No market data is read.
Usage (run_light, PF_SLOT=claude): pf_timing_case_audit_mutants.py [index | targets]
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / 'pf_timing_case_audit.py'
# (name, text in the source, replacement[, occurrences]) - entry 0 is the unchanged copy
MUT = [
    ('none (must pass)', None, None),
    # --- the book-clock bound and the identity rule
    ('B is the largest pair (every pair must be right)', "B=float(x[int(np.ceil(n / 2.)) - 1])", "B=float(x[0])"),
    ('identity: several research twins accepted', "            if len(c) > 1:\n                st['several_research'] += 1; continue",
     "            if False:\n                st['several_research'] += 1; continue"),
    ('identity: several bot twins accepted', "            if len(back) != 1:", "            if False:"),
    ('content key ignores pending amounts',
     "tuple(sorted((int(a), int(b), round(float(v), DP)) for (a, b), v in pend.items() if round(float(v), DP) != 0.))", "()"),
    ('content key rounds sizes to whole shares', "W_LOCAL, W_ID, N_MIN, DP, CLOCK_STEP_MS = 10_000., 5_000., 20, 4, 1.", "W_LOCAL, W_ID, N_MIN, DP, CLOCK_STEP_MS = 10_000., 5_000., 20, 0, 1."),
    # --- the verdict
    ('verdict ignores the clock domain', "    if not a3_ok or domain != 'V_book':", "    if not a3_ok:"),
    ('verdict ignores a stepped bot clock', "    if step_ms is None or not np.isfinite(step_ms) or step_ms > CLOCK_STEP_MS:", "    if step_ms is None:"),
    ('verdict needs B strictly above the critical offset', "    elif not b['B'] >= crit:", "    elif not b['B'] > crit:"),
    ('verdict accepts an ambiguous settlement', "    if a2 != 'SAME-SETTLEMENT':", "    if a2 == 'DIFFERENT':"),
    ('verdict ignores the pair count', "    if b['n'] < N_MIN:", "    if b['n'] < 1:"),
    # --- the readers
    ('kernel reader does not stop at the cutoff', "                if at >= CUTOFF_MS * 1e6:\n                    stopped = True; break",
     "                if False:\n                    stopped = True; break"),
    ('input reader does not stop at the cutoff', "            if m and float(m[1]) >= CUTOFF_MS:", "            if False:"),
    ('an hour outside Nov 8 accepted', "    C.require(all(x.startswith(day + '_') for x in h), f'an hour outside {day}: {h}')", "    pass"),
    ('kernel outbox not joined by its step seq', "                if s in kept_seq and kept_seq[s][0] not in hit:", "                if False:"),
    ('input reader keeps another order id', "                if mk in line and (line.rstrip().endswith(s + '}') or (s[:-1] + ',') in line):", "                if True:"),
    ('input reader ignores the market of an order id (v1 defect)', "                if mk in line and (line.rstrip().endswith(s + '}') or (s[:-1] + ',') in line):",
     "                if (line.rstrip().endswith(s + '}') or (s[:-1] + ',') in line):"),
    ('kernel clock regression accepted', "                C.require(at >= last, 'kernel UTC clock regressed'); last = at; n += 1", "                last = at; n += 1"),
    # --- the binding endpoint and the cancel answer
    ('binding increment without the previous frame', "                d = v - prev.get(k, 0.)", "                d = v"),
    ('binding endpoint always on the book clock', "    dom = 'V_book' if ws.any() else 'V_status'", "    dom = 'V_book'"),
    ('any NotCanceled read as matched', "        return 'MATCHED-OR-GONE' if (\"can't be found\" in t and 'matched' in t) else 'NO-EFFECT'", "        return 'MATCHED-OR-GONE'"),
    ('a pending audit read as no answer', "        return 'CANCELED' if pending_audit else 'NO-ANSWER'", "        return 'NO-ANSWER'"),
    # --- the per-leg evaluation
    ('pairs outside the local window counted', "    loc = P_[np.abs(P_.t_bot.values.astype(float) - center) <= W_LOCAL] if len(P_) else P_", "    loc = P_"),
    ('A0 ignores the bot-clock identity of the two journals', "one_bot_clock=len(acc_in) == 1 and np.isfinite(k_acc) and abs(acc_in[0] - k_acc) <= BOT_CLOCK_TOL_MS,", "one_bot_clock=True,"),
    ('A2 reads any settlement as the same', "          else 'SAME-SETTLEMENT' if ids == set(own_ids) else 'DIFFERENT')", "          else 'SAME-SETTLEMENT')"),
    ('the bot clock step never measured', "float((ck[:, 0] - ck[:, 1]).max() - (ck[:, 0] - ck[:, 1]).min()) / 1e6", "0."),
    # --- v3 (the census reuse): the generalized cancel-confirmation field and the reader speed-ups
    ('a leg without a cancel confirmation fails A0', "if pa else not np.isfinite(float(r.ack)))   # the case table's ack", "if pa else False)   # the case table's ack"),
    ("a sibling order's fills in the same settlement counted", "and x['Exec']['Fill']['trade_id'] == r.trade_id and x['Exec']['Fill']['order_id'] == oid]", "and x['Exec']['Fill']['trade_id'] == r.trade_id]"),
    ('the confirmed cancel compared with a refused first answer', "cancel_confirmation=near(min(t for t, _ in pa), r.ack) if pa", "cancel_confirmation=near(a1_at, r.ack) if pa"),
    ('one bot clock read to 1 us', "BOT_CLOCK_TOL_MS = 0.1 ", "BOT_CLOCK_TOL_MS = 0.001 "),
    ('kernel needle gate never opens', "                for k, nd in (need.items() if rx is not None and rx.search(line) else ()):", "                for k, nd in ():"),
    ('clock windows found from their start only', "                for j in range(bisect.bisect_left(wlo, t_ - span), bisect.bisect_right(wlo, t_)):",
     "                for j in range(bisect.bisect_left(wlo, t_), bisect.bisect_right(wlo, t_)):"),
    ('input reader never parses a market', "            mid = int(line[19:j]) if j > 19 and line[19:j].isdigit() else None", "            mid = None"),
    # --- A6
    ('A6: an EV CANCEL_FIRST accepted', "            C.require(not (ev == 0).any(), 'EV (evidence only) certified a CANCEL_FIRST')", "            pass"),
    ('A6: the cross-tab reproduction not checked', "                ok &= int(m.sum()) == int(x.get('fills', 0)) and", "                ok &= True or"),
    ('A6: a clipped WS leg read as WS_DELAY', "s3dom = np.where(lg.S3_code.isin([0, 1]).values & (lg.S3_llo.values > lo0),", "s3dom = np.where(lg.S3_code.isin([0, 1]).values,"),
    ('A6: every certified fill read as evidence', "                evd = m & (ev == o)", "                evd = m & (ev >= 0)"),
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
    g = dict(__name__='pf_timing_case_audit_mutant', __file__=str(SRC)); exec(compile(source(i), str(SRC), 'exec'), g)
    sys.exit(0 if g['selftest']() else 1)


def main():
    bad = 0
    for i, (name, *_) in enumerate(MUT):
        r = subprocess.run([sys.executable, __file__, str(i)], capture_output=True, text=True)
        tail = r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or ['']
        ok = (r.returncode == 0) if i == 0 else (r.returncode == 1 and 'SELFTEST FAIL' in r.stdout)
        bad += not ok
        how = 'passed' if r.returncode == 0 else ('caught: ' + ' / '.join(x.strip()[:160] for x in r.stdout.strip().splitlines()[1:3])) if 'SELFTEST FAIL' in r.stdout else ' / '.join(x.strip()[:200] for x in tail[-2:])
        print(f'{i:2d} {name}: rc {r.returncode} -> {how}{"" if ok else "   <<< UNEXPECTED"}', flush=True)
    if bad:
        sys.exit(f'TIMING CASE AUDIT MUTATION CHECK FAILED: {bad} unexpected outcomes of {len(MUT)}')
    print(f'timing case audit mutation check passed: the unchanged copy passes its selftest, {len(MUT) - 1} / {len(MUT) - 1} planted code errors are caught by a check', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['targets']:
        [source(i) for i in range(len(MUT))]; print(f'all {len(MUT) - 1} mutation targets occur as declared')
    elif len(sys.argv) > 1:
        one(int(sys.argv[1]))
    else:
        main()
