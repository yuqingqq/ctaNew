"""SELFTEST of pf_adv2_sample (the evaluation label sample's day guard; 2026-11-21, session aea5bfb9). run(M) checks the module object M (the
file, or a mutated copy under pf_mutant_one.py); a failure exits with FAIL. Synthetic roots except check 7 (consumed Oct 24).
 1. guarded_files on synthetic roots (parts from day-1 22h to day+1 01h): a recorder of every hashed path shows no part outside hours
    -1..23 opened; exactly those 25 parts listed; the unguarded construction (the frozen rule) opens the after-day part (the positive
    control); guarded = unguarded minus exactly the foreign parts.
 2. A sealed date (a complete synthetic Oct 28) is refused before any file is hashed (the recorder stays empty).
 3. A loader rule that admitted an Oct 28 part is still refused by the identity's own name check.
 4. A loader that stopped skipping foreign parts is refused (the guard would no longer mirror it).
 5. plan_ok: a conforming plan passes; a wrong day list, audit rate, limit, a not-allowed book part (day-1 21h), a sealed name in the
    tracked hours, a selected world and a partition member of another day are each flagged.
 6. plan / run / finalize pass guarded_files and the declared days to the frozen runner (stubs record the call); a plan failing its
    post-condition is refused by make.
 7. Consumed Oct 24: guarded_files = pf_r2_label_sample.real_files minus exactly the after-day 00h part (consumed Oct 25)."""
import sys
import tempfile
from pathlib import Path

FAIL = 'ADV2 SAMPLE SELFTEST FAILED'
DAY = '20261027'
HOURS = ['20261026_22', '20261026_23'] + [f'{DAY}_{h:02d}' for h in range(24)] + ['20261028_00', '20261028_01']


def _tree(td, day, hours):
    s1, s5 = td / 'io' / 'step1' / day, td / 'io' / 'step5' / day; (s1 / 'books').mkdir(parents=True); s5.mkdir(parents=True)
    for f in ('registry.parquet', 'clock_rl_d2.json', 'legs_timed2.parquet', 'legs_tk.parquet'):
        (s1 / f).write_text('x')
    for f in ('preds_t60k.parquet', 'rows_rl_d2_pk.parquet'):
        (s5 / f).write_text('x')
    (td / 'imp').mkdir(exist_ok=True); (td / 'imp' / f'day_{day}_imp.json').write_text('{}')
    (td / 'fp' / day).mkdir(parents=True); (td / 'fp' / day / 'receipt.json').write_text('{}')
    for h in hours:
        (s1 / 'books' / f'part_{h}.parquet').write_text(h)
    return s1


def _refused(fn):
    try:
        fn()
    except ValueError as e:
        return str(e)
    return None


def _plan(M):
    LS = M.LS; days = list(M.EVAL_DAYS)
    files = lambda d: {f'/x/step1/{d}/books/part_{d}_{h:02d}.parquet': 'h' for h in range(24)}
    return dict(version=LS.VERSION, domain=LS.DOMAIN, days=days, per_hour=LS.PER_HOUR, run_set='full', audit_every=LS.AUDIT_EVERY, part_size=LS.PART_WORLDS,
                strict_day=True, freeze_sha256=LS.L.FREEZE_SHA, limits=dict(LS.LIMITS, peak_mib=LS.PEAK, guard_mib=LS.GUARD),
                sample={d: dict(inputs=dict(files=files(d), tracked={f'/bn/{d}_05.csv.gz': 'h'}), selected=[dict(day=d)]) for d in days},
                partitions=[dict(part=i, day=d, members=[dict(day=d)]) for i, d in enumerate(days)])


def run(M):
    bad = []
    if tuple(M.EVAL_DAYS) != ('20261026', '20261027') or '20261028' not in M.FORBIDDEN:
        bad.append(f'the declared days changed: {M.EVAL_DAYS} / {M.FORBIDDEN}')
    with tempfile.TemporaryDirectory() as td:
        td = Path(td); s1 = _tree(td, DAY, HOURS); roots = (td / 'io', td / 'imp', td / 'fp'); opened = []
        rec = lambda p: opened.append(Path(p).name) or 'h'
        try:                                                                                                # 1
            g = M.guarded_files(DAY, sha=rec, roots=roots)
        except ValueError as e:                                                                             # the guard refused its own valid day
            g = {}; bad.append(f'the guard refused a valid day ({e})')
        if any(n in opened for n in ('part_20261026_22.parquet', 'part_20261028_00.parquet', 'part_20261028_01.parquet')):
            bad.append('the guard opened a foreign part')
        if {Path(p).name for p in g if '/books/' in p} != {f'part_{h}.parquet' for h in HOURS[1:26]}:
            bad.append('the guarded book parts are not exactly hours -1..23')
        opened.clear(); u = M._files(DAY, *roots, lambda p, d: True, rec)
        if 'part_20261028_00.parquet' not in opened:
            bad.append('positive control: the unguarded rule did not open the after-day part')
        if set(u) - set(g) != {str(s1 / 'books' / f'part_{h}.parquet') for h in ('20261026_22', '20261028_00', '20261028_01')} or set(g) - set(u):
            bad.append('guarded != unguarded minus exactly the foreign parts')
        _tree(td, '20261028', [f'20261028_{h:02d}' for h in range(24)]); opened.clear()                    # 2
        why = _refused(lambda: M.guarded_files('20261028', sha=rec, roots=roots))
        if why is None or opened:
            bad.append(f'a sealed date was not refused before any hash ({why}, {len(opened)} hashed)')
        orig = M.LOADER.part_allowed                                                                        # 3
        try:
            M.LOADER.part_allowed = lambda p, d: True
            if _refused(lambda: M.guarded_files(DAY, sha=rec, roots=roots)) is None:
                bad.append('an Oct 28 part admitted by the loader rule was not refused by the name check')
        finally:
            M.LOADER.part_allowed = orig
        flag = M.LOADER.SKIP_FOREIGN_PARTS                                                                  # 4
        try:
            M.LOADER.SKIP_FOREIGN_PARTS = False
            if _refused(lambda: M.guarded_files(DAY, sha=rec, roots=roots)) is None:
                bad.append('a loader that reads foreign parts was not refused')
        finally:
            M.LOADER.SKIP_FOREIGN_PARTS = flag
    P = _plan(M)                                                                                            # 5
    if M.plan_ok(P):
        bad.append(f'a conforming plan was flagged: {M.plan_ok(P)}')
    d0, d1 = M.EVAL_DAYS
    corrupt = {
        'days': lambda p: p.update(days=['20261024', '20261025']),
        'audit rate': lambda p: p.update(audit_every=7),
        'limits': lambda p: p.update(limits=dict(p['limits'], pair_s=999.)),
        'book part': lambda p: p['sample'][d1]['inputs']['files'].update({f'/x/step1/{d1}/books/part_20261026_21.parquet': 'h'}),
        'sealed tracked': lambda p: p['sample'][d1]['inputs']['tracked'].update({'/bn/20261028_00.csv.gz': 'h'}),
        'selected day': lambda p: p['sample'][d0]['selected'].append(dict(day=d1)),
        'partition day': lambda p: p['partitions'][0]['members'].append(dict(day=d1))}
    for name, f in corrupt.items():
        Q = _plan(M); f(Q)
        if not M.plan_ok(Q):
            bad.append(f'plan_ok missed: {name}')
    calls = {}                                                                                              # 6
    LS = M.LS; saved = (LS.make_plan, LS.run, LS.finalize)

    class _Stop(Exception):
        pass

    def stub(name):
        def f(*a, **k):
            calls[name] = (a, k); raise _Stop()
        return f
    try:
        LS.make_plan, LS.run, LS.finalize = stub('plan'), stub('run'), stub('finalize')
        with tempfile.TemporaryDirectory() as td:
            for name, fn in (('plan', lambda: M.make(str(Path(td) / 'o'))), ('run', lambda: M.run_part(td, 3)), ('finalize', lambda: M.finalize_out(td))):
                try:
                    fn()
                except _Stop:
                    pass
    finally:
        LS.make_plan, LS.run, LS.finalize = saved
    for name in ('plan', 'run', 'finalize'):
        a, k = calls.get(name, ((), {}))
        if k.get('files_fn') is not M.guarded_files:
            bad.append(f'{name}: the frozen runner was not given guarded_files')
    if tuple(calls.get('plan', ((), {}))[1].get('days', ())) != tuple(M.EVAL_DAYS):
        bad.append('plan: not the declared days')
    if calls.get('run', ((None, None), {}))[0][1:2] != (3,):
        bad.append('run: not the requested partition')
    import json                                                                                             # 6b: the post-condition
    try:
        def bad_plan(out, **k):
            Q = _plan(M); Q['days'] = ['20261024', '20261025']; Path(out).mkdir(parents=True, exist_ok=True)
            (Path(out) / 'plan.json').write_text(json.dumps(Q)); return Q
        LS.make_plan = bad_plan
        with tempfile.TemporaryDirectory() as td:
            try:
                M.make(str(Path(td) / 'o')); bad.append('a plan failing its post-condition was accepted')
            except ValueError as e:
                if 'post-condition' not in str(e):
                    bad.append(f'a failing plan was not refused by the post-condition ({e})')
            except Exception as e:                                                                          # it went past the check
                bad.append(f'a plan failing its post-condition was not refused by it ({e!r})')
    finally:
        LS.make_plan = saved[0]
    fr, gd = M.LS.real_files('20261024'), M.guarded_files('20261024')                                      # 7
    extra = {Path(p).name for p in set(fr) - set(gd)}
    if extra != {'part_20261025_00.parquet'} or any(fr[p] != gd[p] for p in gd) or set(gd) - set(fr):
        bad.append(f'Oct 24: guarded != real_files minus the after-day part ({sorted(extra)})')
    if bad:
        sys.exit(FAIL + ': ' + '; '.join(bad))
    print(f'adv2 sample selftest passed: the guard never opens a part outside hours -1..23 (recorder), the frozen rule does (positive control); '
          f'a sealed date is refused before any hash; the name check and the loader flag hold; plan_ok flags 7 corruptions; plan / run / finalize '
          f'get the guard; consumed Oct 24: guarded = real_files minus the after-day part ({len(gd)} files)')
