"""MUTATION CHECK of the queue place ledger's selftest (2026-11-17, session aea5bfb9): a checker that cannot fail proves nothing.
Each entry plants one code error in a COPY of pf_queue_place_ledger.py (text replacement, executed in a subprocess; the repo file is
never touched) and runs the copy's selftest on the synthetic tape. Required: the unchanged copy passes, every mutant fails, and every
replacement target occurs exactly as often as declared (a stale target is a failure, never a silent pass). No market data is read.
Usage (run_light, PF_SLOT=claude): pf_queue_place_ledger_mutants.py [index]
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / 'pf_queue_place_ledger.py'
# (name, text in the source, replacement[, occurrences]) - entry 0 is the unchanged copy
MUT = [
    ('none (must pass)', None, None),
    # --- the walk against v1 (caught by the reconciliation and / or the conservation predicates)
    ('IMPROVE allowed only at spread >= 3', "not spread >= 2 else", "not spread >= 3 else"),
    ('2 s status copies the 5 s path status', "else pstat['' if tag == '_m250' else tag]", "else pstat['']"),
    ('post-only mirror at 0 ms, not 58 ms', "PLACE_MS = 58.", "PLACE_MS = 0."),
    ('LATE row dropped for gone_before_first', "skip('FIRST', 'gone_before_first', t=t_f); skip('LATE', 'gone_before_first');", "skip('FIRST', 'gone_before_first', t=t_f);"),
    ('one-sided book called locked', "else 'one_sided_book' if best < 0 or oppb < 0", "else 'locked_or_crossed' if best < 0 or oppb < 0"),
    ('sampled / not_sampled codes swapped', "np.where(np.isin(bj, sorted(chosen)), 3, 2)", "np.where(np.isin(bj, sorted(chosen)), 2, 3)"),
    ('value comparison skipped in the reconciliation', "            x, y = x.values.astype(float), y.values.astype(float); eq = (x == y) | (np.isnan(x) & np.isnan(y))",
     "            eq = np.ones(len(x), bool)"),
    ('state with a failed 5 s path keeps no 2 s value', "                if pth is None or not live:\n                    continue\n                vals[tag] = valued(pth, s, o, [c[2] for c in live], H)",
     "                if pth is None or not live or pstat.get('', 'eligible') != 'eligible':\n                    continue\n                vals[tag] = valued(pth, s, o, [c[2] for c in live], H)"),
    ('EARLY allowed up to 5 ticks behind', "'early_too_far' if dist > 3 else None", "'early_too_far' if dist > 5 else None"),
    ('start margin of the grid 0 s, not 2 s', "'before_start_margin' if t < ws + 2000. else", "'before_start_margin' if t < ws + 0. else"),
    ('births sampled 39 per side', "[:PER_SIDE]; chosen", "[:PER_SIDE - 1]; chosen"),
    # --- columns v1 never saved (only the conservation predicates and the planted expectations can catch these)
    ('time to END from the 2 s end', "ttl=b + H_MAIN - t_a", "ttl=b + H_SHORT - t_a"),
    ('LATE rung not floored at 0', "                    rung = max(rung, 0)\n", "                    rung = rung\n"),
    ('grid row of an excluded step keeps no book', "sink['grid'].append(dict(g, cand_status=why, best=best, opp=oppb, spread=spread)); continue", "sink['grid'].append(dict(g, cand_status=why)); continue"),
    ('birth status: end margin 299 s (the sample filter unchanged)', "np.where(~(bb + H_MAIN <= ws + 298000.), 1,", "np.where(~(bb + H_MAIN <= ws + 299000.), 1,"),
    ('birth status: start margin 2.0 s (the sample filter unchanged)', "code = np.where(~(bb >= ws + 2500.), 0,", "code = np.where(~(bb >= ws + 2000.), 0,"),
    ('window start in the markets table 1 s late', "ws=data['W0'].get(m, np.nan))", "ws=data['W0'].get(m, np.nan) + 1000.)", 2),
    ('FIRST rung 1', "arms = [('FIRST', t_f, 0),", "arms = [('FIRST', t_f, 1),"),
    ('REST1 rung 0', "('REST1', 1, best - sgn)", "('REST1', 0, best - sgn)"),
    ('LATE place: inside reported as at_or_behind', "'inside' if rung < 0 else 'at_or_behind'", "'at_or_behind'"),
    ('grid spread saved with the wrong sign', "cand_status='counted', best=best, opp=oppb, spread=spread, phase=phase", "cand_status='counted', best=best, opp=oppb, spread=-spread, phase=phase"),
    ('state arm path status from the other valuation', "else pstat['' if tag == '_m250' else tag]", "else pstat['_2s' if tag == '' else '' if tag == '_m250' else tag]"),
    # --- L5 (corrected after the ledger / keep-pull review): the tipping point and the worst case at legal payoff bounds
    ('L5a tipping point from the saved mean, not the saved sum', "need = -float(inc.sum()) / n_exc", "need = -float(inc.mean())"),
    ('L5b worst case without payoff bounds', "return bool(inc_sum + hi.sum() < 0) if inc_sum < 0 else bool(inc_sum + lo.sum() > 0) if inc_sum > 0 else False",
     "return bool(inc_sum < 0) if inc_sum < 0 else bool(inc_sum > 0) if inc_sum > 0 else False"),
    ('L5b payoff bounds of bid and ask swapped', "return np.where(s == 0, -30. * p, -30. * (100. - p)), np.where(s == 0, 30. * (100. - p), 30. * p)",
     "return np.where(s == 1, -30. * p, -30. * (100. - p)), np.where(s == 1, 30. * (100. - p), 30. * p)"),
    ('L5b pair bounds ignore the JOIN arm', "    return lx - hy, hx - ly\n", "    return lx, hx\n"),
    # --- the controls themselves
    ('conservation: the status predicates are skipped', "    # --- every status equals its declared predicate, recomputed from the saved columns\n", "    return bad\n"),
    ('planted silent skip is not planted', "            if MUT_SILENT_SKIP and pstat[''] != 'eligible':\n                continue", "            if False:\n                continue"),
    ('refused orders are not checked for a fill', "    C.require(((q == 0) & (v == 0))[rj > 0].all(), 'post-only mirror: a refused order filled')\n", ""),
]
STALE = 97                                                # exit code of a replacement target that does not occur as declared


def one(i):
    name, a, b = MUT[i][:3]; n = (MUT[i] + (1,))[3]; s = SRC.read_text()
    if a is not None:
        if s.count(a) != n:
            print(f'mutation target occurs {s.count(a)} times, declared {n}: {name}', file=sys.stderr); sys.exit(STALE)
        s = s.replace(a, b)
    g = dict(__name__='pf_queue_place_ledger_mutant', __file__=str(SRC)); exec(compile(s, str(SRC), 'exec'), g); g['selftest']()


def main():
    bad = 0
    for i, (name, *_) in enumerate(MUT):
        r = subprocess.run([sys.executable, __file__, str(i)], capture_output=True, text=True)
        tail = r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or ['']
        ok = (r.returncode == 0) if i == 0 else (r.returncode not in (0, STALE))
        bad += not ok
        print(f'{i:2d} {name}: rc {r.returncode} -> {"passed" if r.returncode == 0 else " / ".join(x.strip()[:200] for x in tail[-2:])}{"" if ok else "   <<< UNEXPECTED"}', flush=True)
    if bad:
        sys.exit(f'QUEUE-PLACE LEDGER MUTATION CHECK FAILED: {bad} unexpected outcomes of {len(MUT)}')
    print(f'queue-place ledger mutation check passed: the unchanged copy passes its selftest, {len(MUT) - 1} / {len(MUT) - 1} planted code errors fail it', flush=True)


if __name__ == '__main__':
    one(int(sys.argv[1])) if len(sys.argv) > 1 else main()
