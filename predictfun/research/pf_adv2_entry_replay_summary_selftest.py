"""SELFTEST of pf_adv2_entry_replay_summary (the R4E replay summary v2; 2026-11-21, session aea5bfb9). run(M) checks the module object M; any
failed check exits with FAIL. Synthetic only (two parts of this round's episode schema in a temporary directory):
 1. a valid summary completes (the positive control the frozen code lacked) and writes replay.json with sorted keys;
 2. hand values: statuses (an excluded world counted), per day / policy totals, the binding failures summed from the traces (a trace without
    the counter counts as a failure), the paired differences against ENTRY_ONLY and PI0 (a tie counted in neither), reached counts under 'n';
 3. refusals: a CSV that is not byte-identical to the frozen function's, and a part whose bytes differ from its record."""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

FAIL = 'ADV2 ENTRY REPLAY SUMMARY SELFTEST FAILED'


def _refused(fn, msg):
    try:
        fn()
    except ValueError as e:
        return msg in str(e)
    except Exception:                                                                       # noqa: BLE001  (a crash is not the check)
        return False
    return False


def run(M):  # noqa: C901
    bad = []; NR, LS = M.NR, M.LS
    def ep(v, bb=0, has=True):
        t = dict(teacher=dict(calls=3, binding_checked=3, **({'binding_bad': bb} if has else {})))
        return dict(status='ok', value=v, value_bid=v / 2., value_ask=v / 2., fills=1, filled_shares=30., generations=2, loss=max(-v, 0.), severe=0, ledger_ok=True,
                    cap_ok=True, trace=t, churn=[], statuses={}, actions={})
    def world(day, m, t, vals, extra=None):
        eps = {p: ep(v) for p, v in vals.items()}; eps['SMOOTH']['exposure'] = {'absent|model': 2}
        if extra:
            eps.update(extra)
        rs = [dict(side=0, status='ok', eligible=3, result=dict(Y=1.), A=.2, action='POST', inputs_equal_recorded=True, pred=dict(SMOOTH=.2, RICH=.5)),
              dict(side=1, status='no_eligible_decision', eligible=0)]
        return dict(day=day, m=m, t_ms=t, hour=0, status='ok', episodes=eps, reached=rs)
    V0 = dict(N0=0., PI0=1., ENTRY_ONLY=-2., SMOOTH=3.); V1 = dict(N0=0., PI0=2., ENTRY_ONLY=1., SMOOTH=1.)
    p0 = [world('20261026', 1, 1000, V0), dict(day='20261026', m=2, t_ms=2000, hour=0, status='world:no_path:x')]
    p1 = [world('20261027', 3, 3000, V1, extra={'PI0': ep(2., has=False)})]
    parts = [('20261026', [dict(day='20261026', m=1, t_ms=1000), dict(day='20261026', m=2, t_ms=2000)]), ('20261027', [dict(day='20261027', m=3, t_ms=3000)])]
    fz = 'f' * 64; saved = NR.fit_sha
    with tempfile.TemporaryDirectory() as td:
        out = Path(td); (out / 'parts').mkdir()
        try:
            NR.fit_sha = lambda: fz
            for k, rows in enumerate((p0, p1)):
                b = LS.gz_bytes(rows); (out / 'parts' / f'part_{k:03d}.jsonl.gz').write_bytes(b)
                (out / 'parts' / f'part_{k:03d}.json').write_text(json.dumps(dict(members=[LS.wkey(x) for x in parts[k][1]], sources=NR.sources(), fit_sha256=fz,
                                                                                  eval_plan_sha256=NR.AE.EVAL_PLAN_SHA, sha256=hashlib.sha256(b).hexdigest())))
            E, R, Ch, Ex, st = NR.frames(out, parts)
            for n, df in zip(M.CSVS, (E, R, Ch, Ex)):
                df.to_csv(out / f'{n}.csv', index=False)
            try:                                                                            # a positive control: a valid summary completes
                M.summarize(out, parts); js = json.loads((out / 'replay.json').read_text())
            except Exception as e:                                                          # noqa: BLE001
                js = None; bad.append(f'summarize: a valid summary stopped ({e!r})'[:200])
            if js is not None:
                if js['statuses'] != {'ok': 2, 'world:no_path:x': 1} or js['worlds'] != 2:
                    bad.append(f"statuses / worlds: {js['statuses']} / {js['worlds']}")
                tot = {(t['day'], t['policy']): t for t in js['totals']}
                if not (tot[('20261026', 'SMOOTH')]['value'] == 3. and tot[('20261027', 'PI0')]['binding_bad'] == 1 and tot[('20261026', 'PI0')]['binding_bad'] == 0):
                    bad.append('totals: values or the bindings summed from the traces (a missing counter is a failure)')
                D = {(d['day'], d['policy'], d['versus']): d for d in js['diffs']}
                want = {('20261026', 'SMOOTH', 'ENTRY_ONLY'): (5., 1, 0), ('20261026', 'SMOOTH', 'PI0'): (2., 1, 0), ('20261027', 'SMOOTH', 'ENTRY_ONLY'): (0., 0, 0),
                        ('20261027', 'ENTRY_ONLY', 'PI0'): (-1., 0, 1)}
                for k, (s_, p_, n_) in want.items():
                    d = D.get(k)
                    if d is None or abs(d['diff_sum'] - s_) > 1e-12 or d['positive'] != p_ or d['negative'] != n_:
                        bad.append(f'diffs: {k} = {d}')
                rc = {(r_['day'], r_['status']): r_ for r_ in js['reached']}
                if not (all(set(r_) == {'day', 'status', 'n'} for r_ in js['reached']) and rc[('20261026', 'ok')]['n'] == 1 and rc[('20261027', 'no_eligible_decision')]['n'] == 1):
                    bad.append(f"reached: not the status counts under 'n' ({js['reached']})")
            (out / 'churn.csv').write_bytes((out / 'churn.csv').read_bytes() + b'\n')
            if not _refused(lambda: M.summarize(out, parts), "differs from the frozen summarize's"):
                bad.append('a CSV that is not the frozen one was not refused')
            Ch.to_csv(out / 'churn.csv', index=False)
            b = (out / 'parts' / 'part_001.jsonl.gz').read_bytes(); (out / 'parts' / 'part_001.jsonl.gz').write_bytes(b[:-1] + bytes([b[-1] ^ 1]))
            if not _refused(lambda: M.summarize(out, parts), "not this round's identity"):
                bad.append('a part whose bytes differ from its record was not refused')
        finally:
            NR.fit_sha = saved
    if bad:
        sys.exit(FAIL + ': ' + '; '.join(bad))
    print('adv2 entry replay summary selftest passed: a valid summary completes; statuses, totals, trace-summed bindings (a missing counter '
          "fails), paired differences (sums, above / below, a tie), reached counts under 'n'; refusals of a changed CSV and a changed part")
