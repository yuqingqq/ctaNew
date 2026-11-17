"""MUTATION CHECK of the keep / pull test's selftest (2026-11-17, session aea5bfb9): a checker that cannot fail proves nothing. Each entry
plants one code error in a COPY of pf_keep_pull.py (text replacement, executed in a subprocess; the repo file is never touched) and runs
the copy's selftest on the synthetic tape. Required: the unchanged copy passes, every mutant fails, and every replacement target occurs
exactly as often as declared (a stale target is a failure, never a silent pass). No market data is read.
Usage (run_light, PF_SLOT=claude): pf_keep_pull_mutants.py [index | targets]
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / 'pf_keep_pull.py'
# (name, text in the source, replacement[, occurrences]) - entry 0 is the unchanged copy
MUT = [
    ('none (must pass)', None, None),
    # --- clocks and the per-trade valuation
    ('pending-cancel rule dropped', "np.where(d < PLACE, np.maximum(d + lat, PENDING), d + lat)", "d + lat"),
    ('cancel latency 25 ms', "PLACE, CANCEL, PENDING, NOTICE = 58., 37., 95., 100.", "PLACE, CANCEL, PENDING, NOTICE = 58., 25., 95., 100."),
    ('cancel latency 50 ms', "PLACE, CANCEL, PENDING, NOTICE = 58., 37., 95., 100.", "PLACE, CANCEL, PENDING, NOTICE = 58., 50., 95., 100."),
    ('own fills known at once', "PLACE, CANCEL, PENDING, NOTICE = 58., 37., 95., 100.", "PLACE, CANCEL, PENDING, NOTICE = 58., 37., 95., 0."),
    ('young window 50 ms', "SIZE, SEVERE, YOUNG = 30., 30., 100.", "SIZE, SEVERE, YOUNG = 30., 30., 50."),
    ('severe threshold 20 c', "SIZE, SEVERE, YOUNG = 30., 30., 100.", "SIZE, SEVERE, YOUNG = 30., 20., 100."),
    ('structural drift limit 2 ticks', "HOLD = 3.", "HOLD = 2."),
    ('cuts at the trade times (each trade left out of its own branch)', "cut = np.r_[(tt[:-1] + tt[1:]) / 2., tt[-1:] + 1.]", "cut = tt.copy()"),
    ('a fill at a landing time counted as before it', "jmin = np.searchsorted(landing(np.arange(1, NSTEP + 1), lat), f.t.values, side='right')",
     "jmin = np.searchsorted(landing(np.arange(1, NSTEP + 1), lat), f.t.values, side='left')"),
    ('variant ignores its latency', "jmin = np.searchsorted(landing(np.arange(1, NSTEP + 1), lat), f.t.values, side='right')",
     "jmin = np.searchsorted(landing(np.arange(1, NSTEP + 1), CANCEL), f.t.values, side='right')"),
    ('young filter inverted', "ok &= f.t.values >= PLACE + YOUNG", "ok &= f.t.values < PLACE + YOUNG"),
    ('outcome reads the next step', "jj = P[ii] - 1", "jj = np.minimum(P[ii], NSTEP - 1)"),
    ('decisions continue at the step the client learns the order is gone', "np.searchsorted(d, lim, side='left') + 1", "np.searchsorted(d, lim, side='right') + 1"),
    ('mixed trade source labelled WS', "cs.pop() if len(cs) == 1 else 3", "cs.pop() if len(cs) == 1 else 0"),
    ('post-only refusal ignored', "rec.update(status='refused_post_only' if rej else 'eligible',", "rec.update(status='eligible',"),
    # --- scores and nulls
    ('trigger step off by one', "return np.where(mask.any(1), mask.argmax(1) + 1, NSTEP + 1)", "return np.where(mask.any(1), mask.argmax(1), NSTEP + 1)"),
    ('score A not strict', "out[f'A{th:+g}'] = first((a < th) & live)", "out[f'A{th:+g}'] = first((a <= th) & live)"),
    ('score triggers before the order is live', "out[f'A{th:+g}'] = first((a < th) & live)", "out[f'A{th:+g}'] = first(a < th)"),
    ('a missing fair input triggers a pull', "out[f'A{th:+g}'] = first((a < th) & live)", "out[f'A{th:+g}'] = first(~(a >= th) & live)"),
    ('score B baseline one step late', "b = sg * (x - f[:, :1])", "b = sg * (x - f[:, 1:2])"),
    ('score reads the posting-time fair price at every step', "sg = np.where(np.asarray(side) == 0, 1., -1.)[:, None]; x = f[:, 1:]",
     "sg = np.where(np.asarray(side) == 0, 1., -1.)[:, None]; x = np.repeat(f[:, :1], NSTEP, 1)"),
    ('an order with any missing fair input is dropped from the score (whole-future selection)', "sg = np.where(np.asarray(side) == 0, 1., -1.)[:, None]; x = f[:, 1:]",
     "sg = np.where(np.asarray(side) == 0, 1., -1.)[:, None]; x = np.where(np.isnan(f).any(1, keepdims=True), np.nan, f)[:, 1:]"),
    ('live decisions start at the pending step', "J0 = int(np.ceil(PLACE / STEP))", "J0 = 1"),
    ('outcome without the live lower bound', "ii = np.flatnonzero((P >= j0) & (P <= NSTEP) & (P < V['endj']))", "ii = np.flatnonzero((P <= NSTEP) & (P < V['endj']))"),
    ('oracle without the live lower bound', "M = np.where((cols >= j0) & (cols < V['endj'][:, None]), V['Mv'], -np.inf)", "M = np.where(cols < V['endj'][:, None], V['Mv'], -np.inf)"),
    ('null risk set ignores eligibility', "elig = (Es[src] > j) & ~picked[src]", "elig = (Ms[src] >= 0) & ~picked[src]"),
    ('null does not match the executed counts', "sel = elig & (np.cumsum(elig, 1) <= cnt[r, j][:, None])", "sel = elig"),
    ('unmatched null pulls not counted', "um += int((cnt[r, j] - sel.sum(1)).sum())", "um += 0"),
    # (re-choosing the 37 ms side of the latency pair, outcome(V, oracle(V)) for outcome(V, O0), is an EQUIVALENT mutant by design: with one price and
    #  one END mark the 37 ms oracle's own pulls are worth exactly the zero-latency decisions valued at 37 ms - so it is not listed)
    ('zero-latency side of the latency pair taken from the 37 ms oracle', "o0 = np.array(R[K0]['oracle_by_day']['imp']); o0at37 = np.array(p['oracle_0ms_decisions']['imp'])",
     "o0 = np.array(p['oracle_by_day']['imp']); o0at37 = np.array(p['oracle_0ms_decisions']['imp'])"),
    ('null strata without the market', "cell = E.groupby(['day', 'm', 'side', 'phase'], sort=True).ngroup().values", "cell = E.groupby(['day', 'side', 'phase'], sort=True).ngroup().values"),
    # --- reading predicates
    ('K1 at 40%', "return bool(orc.sum() / max(loss.sum(), 1e-9) >= .5 and", "return bool(orc.sum() / max(loss.sum(), 1e-9) >= .4 and"),
    ('K2 day rule 3 of 5', "return bool(res['imp'].sum() > np.percentile(a, 95) and days >= 4)", "return bool(res['imp'].sum() > np.percentile(a, 95) and days >= 3)"),
    ('K2 against the null median', "return bool(res['imp'].sum() > np.percentile(a, 95) and days >= 4)", "return bool(res['imp'].sum() > np.percentile(a, 50) and days >= 4)"),
    ('K3 against the null median', "r < np.nanpercentile(a, 5) and days >= 4)", "r < np.nanpercentile(a, 50) and days >= 4)"),
    # --- the controls themselves
    ('conservation check removed', "                C.require(abs(fq.sum() - qk) <= 1e-9 and abs(val.sum() - vk) <= 1e-7 and (fq >= -1e-12).all(), 'per-trade fills do not sum to the keep branch')\n", ""),
    ('structural landing check removed', "                C.require(not (fq[tt >= land] > 1e-12).any(), 'a fill at or after the structural cancel landed')\n", ""),
    ('invariance check removed', "        C.require(st['invariance_changed'] == 0, f\"a decision changed when only later fair inputs changed ({st['invariance_changed']})\")\n", ""),
    ('null stratum check removed', "            C.require(bool((sid[who] == np.repeat(r, sel.sum(1))).all()), 'a null pull left its stratum')\n", ""),
    ('v1 value comparison removed', "    C.require(same('v_1', 'v') and same('q_1', 'q') and same('v_m250_1', 'v_m250'), 'KEEP differs from v1 (value, shares or M250 value)')\n", ""),
    ('v1 hash check removed', "    C.require(C.sha(d / 'rows.parquet') == r['outputs']['rows.parquet'], 'v1 rows changed since its receipt')\n", ""),
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
    g = dict(__name__='pf_keep_pull_mutant', __file__=str(SRC)); exec(compile(source(i), str(SRC), 'exec'), g); g['selftest']()


def main():
    bad = 0
    for i, (name, *_) in enumerate(MUT):
        r = subprocess.run([sys.executable, __file__, str(i)], capture_output=True, text=True)
        tail = r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or ['']
        ok = (r.returncode == 0) if i == 0 else (r.returncode not in (0, STALE))
        bad += not ok
        print(f'{i:2d} {name}: rc {r.returncode} -> {"passed" if r.returncode == 0 else " / ".join(x.strip()[:200] for x in tail[-2:])}{"" if ok else "   <<< UNEXPECTED"}', flush=True)
    if bad:
        sys.exit(f'KEEP-PULL MUTATION CHECK FAILED: {bad} unexpected outcomes of {len(MUT)}')
    print(f'keep-pull mutation check passed: the unchanged copy passes its selftest, {len(MUT) - 1} / {len(MUT) - 1} planted code errors fail it', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['targets']:
        [source(i) for i in range(len(MUT))]; print(f'all {len(MUT) - 1} mutation targets occur as declared')
    elif len(sys.argv) > 1:
        one(int(sys.argv[1]))
    else:
        main()
