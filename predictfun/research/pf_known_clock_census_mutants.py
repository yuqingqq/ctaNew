"""MUTATION CHECK of the known-leg clock census's selftest (2026-11-18, session aea5bfb9): a checker that cannot fail proves nothing. Each entry
plants one code error in a COPY of pf_known_clock_census.py (text replacement, executed in a subprocess; the repo file is never touched) and runs
the copy's selftest (synthetic only). Required: the unchanged copy passes; every mutant is caught BY A CHECK (rc 1 and 'SELFTEST FAIL' printed -
a crash is an unexpected outcome, not a kill); every replacement target occurs exactly as often as declared. No market data is read.
Usage (run_light, PF_SLOT=claude): pf_known_clock_census_mutants.py [index | targets]
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / 'pf_known_clock_census.py'
MUT = [
    ('none (must pass)', None, None),
    ('bound ignores the one-bot-clock check', "    if not one_bot_clock:", "    if False:"),
    ('bound ignores a stepped bot clock', "    if step is None or not np.isfinite(step) or step > TCA.CLOCK_STEP_MS:", "    if step is None:"),
    ('bound ignores the pair count', "    if b['n'] < TCA.N_MIN:", "    if b['n'] < 1:"),
    ('the local window centred on the first notice', "center = float(r.ack) if np.isfinite(float(r.ack)) else float(r.upper)", "center = float(r.upper)"),
    ('alerts of other orders counted', "for t, k, line in L_['rec'] if k == 'outbox' and r.order_hash.lower() in line for a", "for t, k, line in L_['rec'] if k == 'outbox' for a"),
    ('every order read as filled in full', "full=bool(float(x['filled']) >= float(x['size']) - 1e-9)", "full=True"),
    ('every snapshot read as understating', "under = [s for s in snaps if s['size_matched'] < round(float(x['filled']) * 1e6)]", "under = snaps"),
    ('understating snapshots all read as after the notice', "understating_after_notice=sum(1 for s in under if np.isfinite(fn) and s['at'] >= fn)",
     "understating_after_notice=len(under)"),
    ("the first notice taken as the latest leg's", "fn = float(np.nanmin(fn)) if np.isfinite(fn).any() else np.nan", "fn = float(np.nanmax(fn)) if np.isfinite(fn).any() else np.nan"),
    ('any alert read as a late-fill alert', "alerted=any(any(k in a for k in ALERT_KEYS) for a in alerts)", "alerted=bool(alerts)"),
    ('a partial fill after Canceled counted as a race', "race = int(((O.cls == 'CANCELED') & O.full).sum())", "race = int((O.cls == 'CANCELED').sum())"),
    ('the case-audit reproduction never checked', "    read['N0 REPRODUCES-THE-CASE-AUDIT'] = 'YES' if same else 'NO'", "    read['N0 REPRODUCES-THE-CASE-AUDIT'] = 'YES'"),
    ('every order read as acknowledged (v2 descriptive)', "acknowledged=bool(g.acknowledged.any())", "acknowledged=True"),
    ('legs without a cancel confirmation counted in N2', "can = np.isfinite(L.ack.values.astype(float)); idf = can & (L.bound_status.values == 'IDENTIFIED')",
     "can = np.ones(len(L), bool); idf = can & (L.bound_status.values == 'IDENTIFIED')"),
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
    g = dict(__name__='pf_known_clock_census_mutant', __file__=str(SRC)); exec(compile(source(i), str(SRC), 'exec'), g)
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
        sys.exit(f'KNOWN CLOCK CENSUS MUTATION CHECK FAILED: {bad} unexpected outcomes of {len(MUT)}')
    print(f'known clock census mutation check passed: the unchanged copy passes its selftest, {len(MUT) - 1} / {len(MUT) - 1} planted code errors are caught by a check', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['targets']:
        [source(i) for i in range(len(MUT))]; print(f'all {len(MUT) - 1} mutation targets occur as declared')
    elif len(sys.argv) > 1:
        one(int(sys.argv[1]))
    else:
        main()
