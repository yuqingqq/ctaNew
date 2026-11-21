"""SELFTEST of pf_entry_exposure_summary (ENTRY-EXPOSURE-v1 summary v2; 2026-11-21, session aea5bfb9). run(V) checks the module object V (the
file, or a mutated copy under pf_mutant_one.py); any failed check exits with FAIL. Hand rows only:
 1. flat_v2: a counter above the log reads the log (the counter kept as overrides_counter_v1); no logged override with equal KEEP / POST_PI0
    ledgers is baseline-equal and ok; a different decisions hash, fills hash or value is keep_baseline_mismatch; a world row is unchanged;
 2. the frozen summaries on v2 rows count overrides_forks from the log;
 3. summarize_v2 end to end in a temporary directory: the four outputs, the manifest's byte copy, the linked partitions, what it supersedes,
    the changed-fork count; a second run is refused (immutable); a partition whose bytes differ from its record is refused."""
import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path

FAIL = 'ENTRY EXPOSURE SUMMARY V2 SELFTEST FAILED'


def run(V):  # noqa: C901
    bad = []; M, pd, LS = V.M, V.pd, V.LS

    def row(i, counter, shadow, eq=True, diff=None, status='ok', side=0):
        arms = {n: dict(value=1.5, position=0., fills_sha256='F', decisions_sha256='D') for n in ('WAIT_PI0', 'POST_PI0', 'POST_KEEP_INITIAL')}
        if diff:
            arms['POST_KEEP_INITIAL'][diff] = 2.5 if diff == 'value' else 'X'
        return dict(day='20261024', m=i, t_ms=i, hour=0, side=side, oid=1, status=status, reproduced=True, keep_baseline_equal=True, prefix_ok=True, keep_reconciled=True,
                    initial_gid_equal=True, components_ok=True, identity_ok=True, overrides=counter, Y_PI0=0., Y_stored=0., Y_KEEP_INITIAL=1., D_EXPOSURE=1., D_initial=1., D_rest=0.,
                    D_focal_rest=0., D_opposite=0., ahead_at_submit=3., shadow=[dict(t=100. + 50 * j, gid=0, shadow_action='PULL') for j in range(shadow)],
                    init_PI0=dict(why='policy', filled=0., active_ms=79., v_initial=0., unfilled_at_end=0., fills=[]),
                    init_KEEP=dict(why='end', filled=0., active_ms=4942., v_initial=1., unfilled_at_end=30., fills=[]), arms=arms)
    rows = [row(0, 5, 3), row(1, 2, 0), row(2, 2, 0, diff='decisions_sha256'), row(3, 2, 0, diff='fills_sha256'), row(4, 2, 0, diff='value'),
            dict(day='20261024', m=5, t_ms=5, hour=0, side=0, oid=None, status='world:x')]
    try:
        f = [V.flat_v2(copy.deepcopy(r)) for r in rows]
        want = [(3, 5, True, 'ok'), (0, 2, True, 'ok'), (0, 2, False, 'keep_baseline_mismatch'), (0, 2, False, 'keep_baseline_mismatch'), (0, 2, False, 'keep_baseline_mismatch')]
        got = [(x['overrides'], x['overrides_counter_v1'], x['keep_baseline_equal'], x['status']) for x in f[:5]]
        if got != want:
            bad.append(f'flat_v2: {got} not {want}')
        if f[5]['status'] != 'world:x' or f[5].get('overrides') is not None or f[5]['overrides_counter_v1'] is not None:
            bad.append('flat_v2: a world row was changed')
        if V.logged_overrides(rows[0]) != 3 or V.logged_overrides(dict(rows[0], shadow=None)) != 0:
            bad.append('logged_overrides: not the shadow log length')
        S = M.summaries(pd.DataFrame(f))
        if len(S) != 1 or S[0]['forks'] != 2 or S[0]['overrides_forks'] != 1:
            bad.append(f'summaries on v2 rows: {[(s_["forks"], s_["overrides_forks"]) for s_ in S]}')
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'flat_v2 / summaries: stopped ({e!r})'[:200])
    try:
        with tempfile.TemporaryDirectory(dir=str(M.R2.I.K.IO)) as tmp:
            out = Path(tmp) / 'out'; (out / 'parts').mkdir(parents=True)
            parts_rows = [row(0, 5, 3), row(0, 4, 4, side=1), row(1, 2, 0), row(1, 2, 0, diff='value', side=1)]
            man = dict(parts=[dict(set='train', day='20261024', worlds=[dict(day='20261024', m=0, t_ms=0), dict(day='20261024', m=1, t_ms=1)])])
            (out / 'manifest.json').write_text(json.dumps(man)); b = LS.gz_bytes(parts_rows); (out / 'parts' / 'part_000.jsonl.gz').write_bytes(b)
            (out / 'parts' / 'part_000.json').write_text(json.dumps(dict(sha256=hashlib.sha256(b).hexdigest(), manifest_sha256=M.C.sha(out / 'manifest.json'), sources=M.sources())))
            (out / 'exposure.json').write_text('{"frozen": true}\n'); o2 = Path(tmp) / 'v2'
            V.summarize_v2(out, o2); js = json.loads((o2 / 'exposure.json').read_text()); F2 = pd.read_csv(o2 / 'forks.csv'); S2 = pd.read_csv(o2 / 'summary.csv')
            if not (list(F2.overrides) == [3, 4, 0, 0] and list(F2.overrides_counter_v1) == [5, 4, 2, 2] and list(F2.status) == ['ok', 'ok', 'ok', 'keep_baseline_mismatch']
                    and js['overrides_changed_forks'] == 3 and js['supersedes'] == V.SUPERSEDES and js['superseded_sha256'] == M.C.sha(out / 'exposure.json')
                    and (o2 / 'manifest.json').read_bytes() == (out / 'manifest.json').read_bytes() and (o2 / 'parts' / 'part_000.json').exists()
                    and (o2 / 'reasons.csv').exists() and int(S2.overrides_forks.sum()) == 2 and js['sources'] == M.sources() and js['version'] == V.VERSION):
                bad.append(f"summarize_v2: the written outputs ({list(F2.overrides)}, {list(F2.status)}, {js.get('overrides_changed_forks')})")
            try:
                V.summarize_v2(out, o2); bad.append('summarize_v2: an existing v2 summary was overwritten')
            except Exception:                                                               # noqa: BLE001  (the refusal)
                pass
            (out / 'parts' / 'part_000.jsonl.gz').write_bytes(LS.gz_bytes(parts_rows[::-1]))
            try:
                V.summarize_v2(out, Path(tmp) / 'v2b'); bad.append('summarize_v2: a partition whose bytes differ from its record was read')
            except Exception:                                                               # noqa: BLE001  (the refusal)
                pass
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'summarize_v2: stopped ({e!r})'[:200])
    if bad:
        sys.exit(FAIL + ': ' + '; '.join(bad))
    print('entry exposure summary v2 selftest passed: flat_v2 (log count, counter kept, baseline equality from the ledgers, world rows unchanged); '
          'summaries on v2 rows; summarize_v2 end to end (outputs, manifest copy, linked partitions, supersedes; immutable; partition hash refused)')
