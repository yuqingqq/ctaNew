"""MUTATION CHECK of the branch engine's selftest (plan v2 R1; 2026-11-19, session aea5bfb9): a checker that cannot fail proves nothing. Each
entry plants one code error in a COPY of pf_branch_engine.py (text replacement, executed in a subprocess; the repo file is never touched) and runs
the copy's selftest (synthetic tapes only). Required: the unchanged copy passes, every mutant is caught BY A CHECK (the selftest's own failure
message; a crash is unexpected), and every replacement target occurs exactly as often as declared (a stale target is a failure). No market data
is read. Usage (run_light, PF_SLOT=claude): pf_branch_engine_mutants.py [index | targets]
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / 'pf_branch_engine.py'
# (name, text in the source, replacement[, occurrences]) - entry 0 is the unchanged copy
MUT = [
    ('none (must pass)', None, None),
    # --- the engine (the kernel's rules)
    ('the queue ahead never shrinks to later shown sizes (no rule M)', "                        o['ahead'] = min(o['ahead'], a)", "                        o['ahead'] = o['ahead']"),
    ('a trade at the price skips the queue ahead', "use = min(left, o['ahead']); o['ahead'] -= use; q = min(left - use, o['rem'])", "use = 0.; q = min(left, o['rem'])"),
    ('a trade through the price fills nothing', "                    q = min(left, o['rem']); o['ahead'] = 0.", "                    q = 0.; o['ahead'] = 0."),
    ('the removal is inclusive (a match at the effect fills)', "o['act'] > t or t >= o['remove']:", "o['act'] > t or t > o['remove']:"),
    ('the activation is exclusive', "o['act'] > t or t >= o['remove']:", "o['act'] >= t or t >= o['remove']:"),
    ('no post-only refusal at activation', "            if (0 <= opp <= price) if o['side'] == 0 else (opp >= price):", "            if False:"),
    ('the queue ahead from the first frame, not the one at activation', "            while j + 1 < self.nf and self.f[j + 1, 0] <= act:", "            while False:"),
    ('the structural rule before the submission (no guard)', " or (not KERNEL_MODE and grid <= o['submit'])", ""),
    ('the kernel-path structural rule never fires', "cross = o['price'] >= opp if o['side'] == 0 else o['price'] <= opp\n            if best < 0 or behind > HOLD + o['rung'] or cross:",
     "cross = o['price'] >= opp if o['side'] == 0 else o['price'] <= opp\n            if False:"),
    ('the client structural rule never fires', "cross = o['price'] >= opp if s == 0 else o['price'] <= opp\n            if best < 0 or behind > HOLD + o['rung'] or cross:",
     "cross = o['price'] >= opp if s == 0 else o['price'] <= opp\n            if False:"),
    ('the client structural rule skips orders the engine has filled (043ca403 P1)', "            if o['req'] is not None or g <= o['submit']:",
     "            if o['req'] is not None or g <= o['submit'] or o['rem'] <= EPS:"),
    ('the client structural rule on the engine frame, not the received one', "        fk = self.E.f[self.E.known]\n        for s in (0, 1):",
     "        fk = self.E.f[max(0, int(np.searchsorted(self.E.f[:, 0], g, side='right')) - 1)]\n        for s in (0, 1):"),
    ('a removal without the cancel latency', "o = self.o[gid]; arr = max(g + CANCEL, o['submit'] + PENDING) if g < o['act'] else g + CANCEL", "o = self.o[gid]; arr = max(g + CANCEL, o['submit'] + PENDING) if g < o['act'] else g"),
    ('no pending-cancel clock before activation', "o = self.o[gid]; arr = max(g + CANCEL, o['submit'] + PENDING) if g < o['act'] else g + CANCEL", "o = self.o[gid]; arr = g + CANCEL"),
    ('price priority reversed between own orders', "-self.o[i]['price'] if self.o[i]['side'] == 0 else self.o[i]['price']", "self.o[i]['price'] if self.o[i]['side'] == 0 else -self.o[i]['price']"),
    # --- the client
    ('a post known before its activation', "POST_NOTICE = 0. ", "POST_NOTICE = -58. "),
    ('the reservation inequality never checked', "        C.require(Ik + Bk <= CAP + 1e-9 and -Ik + Ak <= CAP + 1e-9,", "        C.require(True,"),
    ('a saved fork state of another tick accepted', "C.require(Bv.k == k and Bv.episode == episode, 'a saved fork state of another tick / episode');", "C.require(True, '');"),
    ('cap-blocked opportunities counted as feasible', "self.tally(s, 'absent cap-blocked' if v['room'] < -1e-9 else", "self.tally(s, 'absent feasible, POST' if v['room'] < -1e-9 else"),
    ('a fill known at once (no notice delay)', "return sum(q for t, i, _, q, _, _ in self.E.fills if i == gid and t + FILL_NOTICE <= g)", "return sum(q for t, i, _, q, _, _ in self.E.fills if i == gid and t <= g)"),
    ('a removal acknowledged at its effect', "ack = o['remove'] + CANCEL_ACK if o['req'] is not None else np.inf", "ack = o['remove'] if o['req'] is not None else np.inf"),
    ('a pulled generation closed at its request (switch)', "REENTER_EARLY = False ", "REENTER_EARLY = True "),
    ('WAIT never posts later (switch)', "WAIT_IS_NO_ENTRY = False ", "WAIT_IS_NO_ENTRY = True "),
    ('the second side sees a stale view (switch)', "STALE_VIEW = False ", "STALE_VIEW = True "),
    ('the sides always in the same order', "order = ((B.episode + k) % 2, 1 - (B.episode + k) % 2)", "order = (0, 1)"),
    ('the cap ignored', "            elif v['room'] < -1e-9:", "            elif False:"),
    ('a post at or after END - 58 ms allowed', "            elif g + PLACE >= self.end:", "            elif False:"),
    ('a JOIN at the opposite best', "self.open[s] = self.E.add(s, v['bid'] if s == 0 else v['ask'], g)", "self.open[s] = self.E.add(s, v['ask'] if s == 0 else v['bid'], g)"),
    ('the view reads the last frame of the tape', "fk = self.E.f[self.E.known]; st = self.status(s, g)", "fk = self.E.f[-1]; st = self.status(s, g)"),
    ('the position check never fails', "C.require(Ik - Ak - 1e-9 <= P <= Ik + Bk + 1e-9,", "C.require(True,"),
    ('a fallback not counted', "action = 'WAIT' if st == 'absent' else 'KEEP'; self.fallbacks.append((float(g), s, st, 'feature_unavailable'))", "action = 'WAIT' if st == 'absent' else 'KEEP'"),
    ('the policy horizon not checked', "    C.require(getattr(policy, 'horizon', None) == END, 'the policy does not declare the common END horizon')\n    B = B or", "    B = B or"),
    ('a branch END other than 5,000 ms accepted', "        C.require(float(end) == END, 'a branch END other than the contract\\'s common END')\n", ""),
    ('a stale generation accepted', "C.require(v['gid'] is not None and v['gid'] == self.open[s], 'a PULL for a generation that is not the open one')", "C.require(True, '')"),
    # --- the contract
    ('a missing mark valued at 0', "    if B.mark is None or not np.isfinite(B.mark):\n        return dict(status='missing_mark', fills=list(B.E.fills))", "    if B.mark is None or not np.isfinite(B.mark):\n        B.mark = 0."),
    ('the pair difference reversed', "Y=arms[na]['value'] - arms[nb]['value']", "Y=arms[nb]['value'] - arms[na]['value']"),
    ('the pre-fork comparison always true', "    return pre(fa) == pre(fb)", "    return True"),
    ('the END value from the start mid', "return cash, pos, cash + pos * mark", "return cash, pos, cash + pos * 50."),
    ('the slice comparison ignores the quantities', "    return sorted(ea) == sorted(ka) and all(abs(ea[t_] - ka[t_]) <= 1e-9 for t_ in ea)", "    return sorted(ea) == sorted(ka)"),
    ('the slice comparison ignores the times', "    return sorted(ea) == sorted(ka) and all(abs(ea[t_] - ka[t_]) <= 1e-9 for t_ in ea)", "    return abs(sum(ea.values()) - sum(ka.values())) <= 1e-9"),
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
    g = dict(__name__='pf_branch_engine_mutant', __file__=str(SRC)); exec(compile(source(i), str(SRC), 'exec'), g); g['selftest']()


def main():
    bad = 0
    for i, (name, *_) in enumerate(MUT):
        r = subprocess.run([sys.executable, __file__, str(i)], capture_output=True, text=True)
        tail = r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or ['']
        caught = r.returncode == 1 and 'BRANCH ENGINE SELFTEST FAILED' in r.stderr          # caught BY A CHECK (the selftest's own exit message); a crash is unexpected
        ok = (r.returncode == 0) if i == 0 else caught
        bad += not ok
        first = next((x for x in r.stderr.splitlines() if 'SELFTEST FAILED' in x), None)
        how = 'passed' if r.returncode == 0 else ('caught: ' + first.strip()[:220]) if caught and first else ' / '.join(x.strip()[:220] for x in tail[-2:])
        print(f'{i:2d} {name}: rc {r.returncode} -> {how}{"" if ok else "   <<< UNEXPECTED"}', flush=True)
    if bad:
        sys.exit(f'BRANCH ENGINE MUTATION CHECK FAILED: {bad} unexpected outcomes of {len(MUT)}')
    print(f'branch engine mutation check passed: the unchanged copy passes its selftest, {len(MUT) - 1} / {len(MUT) - 1} planted code errors are caught by a check', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['targets']:
        [source(i) for i in range(len(MUT))]; print(f'all {len(MUT) - 1} mutation targets occur as declared')
    elif len(sys.argv) > 1:
        one(int(sys.argv[1]))
    else:
        main()
