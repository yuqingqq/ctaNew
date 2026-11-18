"""MUTATION CHECK of the settlement-time extraction's selftest (2026-11-18, session aea5bfb9): a checker that cannot fail proves nothing.
Each entry plants one code error in a COPY of pf_settlement_times.py (text replacement, executed in a subprocess; the repo file is never
touched) and runs the copy's selftest (synthetic only). Required: the unchanged copy passes, every mutant is caught BY A CHECK (rc 1 and
'SETTLEMENT SELFTEST FAILED' in its exit message - v2: a crash was counted as a kill before), every target occurs as declared (a stale
target is a failure). No market data is read.
Usage (run_light, PF_SLOT=claude): pf_settlement_times_mutants.py [index | targets]
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / 'pf_settlement_times.py'
MUT = [
    ('none (must pass)', None, None),
    ('the UUID time from 44 bits', "return float(int(sid[:8] + sid[9:13], 16))", "return float(int(sid[:8] + sid[9:12], 16))"),
    ('the version nibble not checked', "if not isinstance(sid, str) or len(sid) != 36 or sid[14] != '7':", "if not isinstance(sid, str) or len(sid) != 36:"),
    ('the latest receipt kept, not the first', "if recv < old[2]:", "if recv > old[2]:"),
    ('equal-content legs with different ids not marked', "st[(st == 0) & (J._nid.values > 1)] = 3", "st[(st == 0) & (J._nid.values > 9)] = 3"),
    ('a count mismatch not marked', "st[(st == 0) & (J._cnt.values != J._cnt_m.values)] = 2", "st[(st == 0) & False] = 2"),
    ('C1 threshold ignored', "ok = dict(C1=v1 <= MAX_VIOLATION and v1s >= MIN_SHUFFLED,", "ok = dict(C1=v1s >= MIN_SHUFFLED,"),
    ('C1 shuffled control ignored', "ok = dict(C1=v1 <= MAX_VIOLATION and v1s >= MIN_SHUFFLED,", "ok = dict(C1=v1 <= MAX_VIOLATION,"),
    ('C3 ignored', "C3=capok >= MIN_CAP,", "C3=True,"),
    ('the shuffle not applied', "return share(s), share(rng.permutation(s)), int(ok.sum())", "return share(s), share(s), int(ok.sum())"),
    ('a refused day still writes', "    if not all(ok.values()):\n        sys.exit(", "    if False:\n        sys.exit("),
    # --- v2: sealed / forward hours never opened, boundary legs marked
    ('the page reader opens a sealed hour', "        C.require(hour_allowed(h), f'sealed or forward hour {h}: never opened')", "        pass"),
    ('day_run keeps the sealed hours in its padded list', "excluded = [h for h in hours if not hour_allowed(h)]; hours = [h for h in hours if hour_allowed(h)]",
     "excluded = [h for h in hours if not hour_allowed(h)]"),
    ('boundary legs keep status absent', "        st = np.where((st == 1) & near, 5, st)", "        st = st"),
    ('the forward boundary ignored', "    return not (I.SEAL_MS[0] <= h0 < I.SEAL_MS[1]) and h0 < I.FWD_MS", "    return not (I.SEAL_MS[0] <= h0 < I.SEAL_MS[1])"),
    ('the boundary window 4 h wide', "near |= (ex >= h0 - EXTRA_AFTER_H * 3600000.)", "near |= (ex >= h0 - 4 * 3600000.)"),
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
    g = dict(__name__='pf_settlement_times_mutant', __file__=str(SRC)); exec(compile(source(i), str(SRC), 'exec'), g); g['selftest']()


def main():
    bad = 0
    for i, (name, *_) in enumerate(MUT):
        r = subprocess.run([sys.executable, __file__, str(i)], capture_output=True, text=True)
        tail = r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or ['']
        ok = (r.returncode == 0) if i == 0 else (r.returncode == 1 and 'SETTLEMENT SELFTEST FAILED' in r.stderr)
        bad += not ok
        print(f'{i:2d} {name}: rc {r.returncode} -> {"passed" if r.returncode == 0 else " / ".join(x.strip()[:200] for x in tail[-2:])}{"" if ok else "   <<< UNEXPECTED"}', flush=True)
    if bad:
        sys.exit(f'SETTLEMENT MUTATION CHECK FAILED: {bad} unexpected outcomes of {len(MUT)}')
    print(f'settlement mutation check passed: the unchanged copy passes its selftest, {len(MUT) - 1} / {len(MUT) - 1} planted code errors are caught by a check', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['targets']:
        [source(i) for i in range(len(MUT))]; print(f'all {len(MUT) - 1} mutation targets occur as declared')
    elif len(sys.argv) > 1:
        one(int(sys.argv[1]))
    else:
        main()
