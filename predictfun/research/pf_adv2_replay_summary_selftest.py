"""SELFTEST of pf_adv2_replay_summary (the R3 replay summary v2; 2026-11-21, session aea5bfb9). run(M) checks the module object M (the
file, or a mutated copy under pf_mutant_one.py); any failed check exits with FAIL. Synthetic only (two parts in a temporary directory):
 1. a valid summary completes (a positive control: the frozen function's crash on the int count key must not recur) and writes replay.json
    with sorted keys;
 2. its content against hand values: the statuses (an excluded world counted), the per day / policy totals (value, not the bid value),
    the worlds, the paired differences against PI0 and ENTRY_PA (sums, worlds above / below, a tie counted in neither), the reached status
    counts under the key 'n', the CSV hashes;
 3. refusals: a CSV that is not byte-identical to the frozen function's, and a part whose bytes differ from its record."""
import json
import sys
import tempfile
from pathlib import Path

FAIL = 'ADV2 REPLAY SUMMARY SELFTEST FAILED'


def _refused(fn, msg):
    try:
        fn()
    except ValueError as e:
        return msg in str(e)
    except Exception:                                                                       # noqa: BLE001  (a crash is not the check)
        return False
    return False


def _ep(v, vb):
    return dict(status='ok', value=v, value_bid=vb, value_ask=v - vb, fills=1, filled_shares=30., generations=2, loss=max(-v, 0.), severe=0,
                ledger_ok=True, binding_bad=0, causal=dict(checked=1), churn=[], cap_ok=True)


def run(M):  # noqa: C901
    bad = []; LS, pd = M.LS, M.pd
    V0 = dict(N0=0., PI0=1., RICH=3., FAIR=-2., MECH=.5, ENTRY_ONLY=2., RETAIN_ONLY=1., ENTRY_PA=-1.)
    V1 = dict(N0=0., PI0=2., RICH=1., FAIR=0., MECH=0., ENTRY_ONLY=0., RETAIN_ONLY=0., ENTRY_PA=0.)
    def world(day, m, t, vals, reached):
        eps = {p: _ep(v, v / 2. + .25) for p, v in vals.items()}
        eps['RICH']['churn'] = [dict(gid=0, side=0, t_post=0., t_review=50., dt=50., action='KEEP')]; eps['RICH']['exposure'] = {'absent|model': 3, 'live|model': 1}
        return dict(day=day, m=m, t_ms=t, hour=0, status='ok', episodes=eps, reached=reached)
    rq = lambda s, h, st, **k: dict(key=dict(side=s, head=h), status=st, eligible=k.get('eligible', 2), Y=k.get('Y'), A=.1, action='POST', pred=dict(RICH=.1, FAIR=0., MECH=0.))
    p0 = [world('20261026', 1, 1000, V0, [rq(0, 'post', 'ok', Y=1.), rq(0, 'keep', 'ok', Y=0.), rq(1, 'post', 'no_eligible_decision', eligible=0), rq(1, 'keep', 'missing_mark')]),
          dict(day='20261026', m=2, t_ms=2000, hour=0, status='world:no_path:x')]
    p1 = [world('20261027', 3, 3000, V1, [rq(s, h, 'ok', Y=0.) for s, h in ((0, 'post'), (0, 'keep'), (1, 'post'), (1, 'keep'))])]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td); (out / 'parts').mkdir()

        def put(k, rows):
            b = LS.gz_bytes(rows); (out / 'parts' / f'part_{k:03d}.jsonl.gz').write_bytes(b)
            (out / 'parts' / f'part_{k:03d}.json').write_text(json.dumps(dict(part=k, sha256=M.hashlib.sha256(b).hexdigest())))
        put(0, p0); put(1, p1)
        E, R, st, Ch, Ex = M.frames(out, 2)                                                 # the frozen function's CSV writes, reproduced
        E.to_csv(out / 'episodes.csv', index=False); R.to_csv(out / 'reached.csv', index=False); Ch.to_csv(out / 'churn.csv', index=False); Ex.to_csv(out / 'exposure.csv', index=False)
        try:                                                                                # a positive control: a valid summary completes
            M.summarize(out, 2); js = json.loads((out / 'replay.json').read_text())
        except Exception as e:                                                              # noqa: BLE001
            js = None; bad.append(f'summarize: a valid summary stopped ({e!r})'[:200])
        if js is not None:
            if js['statuses'] != {'ok': 2, 'world:no_path:x': 1} or js['worlds'] != 2:
                bad.append(f"statuses / worlds: {js['statuses']} / {js['worlds']}")
            tot = {(t['day'], t['policy']): t for t in js['totals']}
            if not (tot[('20261026', 'RICH')]['value'] == 3. and tot[('20261026', 'RICH')]['value_bid'] == 1.75 and tot[('20261027', 'PI0')]['value'] == 2.
                    and tot[('20261026', 'FAIR')]['worlds'] == 1):
                bad.append('totals: not the per day / policy sums of the episode values')
            D = {(d['day'], d['policy'], d['versus']): d for d in js['diffs']}
            want = {('20261026', 'RICH', 'PI0'): (2., 1, 0), ('20261026', 'RICH', 'ENTRY_PA'): (4., 1, 0), ('20261026', 'FAIR', 'PI0'): (-3., 0, 1),
                    ('20261027', 'RICH', 'PI0'): (-1., 0, 1), ('20261027', 'MECH', 'ENTRY_PA'): (0., 0, 0), ('20261027', 'N0', 'PI0'): (-2., 0, 1)}
            for k, (s_, p_, n_) in want.items():
                d = D.get(k)
                if d is None or abs(d['diff_sum'] - s_) > 1e-12 or d['positive'] != p_ or d['negative'] != n_ or d['n'] != 1:
                    bad.append(f'diffs: {k} = {d}, not {(s_, p_, n_)}')
            if len(js['diffs']) != 2 * (2 * len(V0) - 2):                                     # per day: 7 policies against PI0, 7 against ENTRY_PA
                bad.append(f"diffs: {len(js['diffs'])} rows, not every policy against PI0 and ENTRY_PA per day")
            rc = {(r_['day'], r_['head'], r_['status']): r_ for r_ in js['reached']}
            if not (all(set(r_) == {'day', 'head', 'status', 'n'} for r_ in js['reached']) and rc[('20261026', 'post', 'ok')]['n'] == 1
                    and rc[('20261027', 'keep', 'ok')]['n'] == 2 and rc[('20261026', 'keep', 'missing_mark')]['n'] == 1):
                bad.append(f"reached: not the status counts under 'n' ({js['reached']})")
            if js['csv_sha256'] != {n: M.C.sha(out / f'{n}.csv') for n in M.CSVS}:
                bad.append('csv hashes: not the CSVs on disk')
        (out / 'churn.csv').write_bytes((out / 'churn.csv').read_bytes() + b'\n')
        if not _refused(lambda: M.summarize(out, 2), "differs from the frozen summarize's"):
            bad.append('a CSV that is not the frozen one was not refused')
        Ch.to_csv(out / 'churn.csv', index=False)
        b = (out / 'parts' / 'part_001.jsonl.gz').read_bytes(); (out / 'parts' / 'part_001.jsonl.gz').write_bytes(b[:-1] + bytes([b[-1] ^ 1]))
        if not _refused(lambda: M.summarize(out, 2), 'bytes differ from its record'):
            bad.append('a part whose bytes differ from its record was not refused')
    if bad:
        sys.exit(FAIL + ': ' + '; '.join(bad))
    print('adv2 replay summary selftest passed: a valid summary completes with sorted keys; statuses, totals, worlds, paired differences '
          "(sums, above / below, a tie), reached counts under 'n', CSV hashes; refusals of a changed CSV and a changed part")
