"""SELFTEST of pf_source_extend (SOURCE-EXTEND-v1; 2026-11-21, session aea5bfb9). run(M) checks the module object M; any failed check exits with
FAIL. No real builder runs (recorders stand in for the frozen day_run functions):
 1. allowed: the new days and the control day pass; an undeclared day fails; a declared day in the sealed block or ending after the forward
    boundary fails (declared for the test only);
 2. declared: inside, the three day lists hold exactly the day and v1 writes under the given root; after it, and after an error inside it,
    everything is restored;
 3. build: v1 runs first with the swapped globals, keep / pull second with v1 = root/queue_place and out = root/keep_pull;
 4. frames_equal / compare: equal frames and receipts pass; a changed value, dtype or column order, a changed stat / status / reconciliation /
    fair_missing, v1 markets or a changed frozen-source hash fail;
 5. control: write-once; refused when the wrapper is not committed; run_day: refused for an undeclared day, without a passed control and when
    the wrapper is not committed."""
import json
import sys
import tempfile
from pathlib import Path

FAIL = 'SOURCE EXTEND SELFTEST FAILED'


def run(M):  # noqa: C901
    bad = []; pd = M.pd
    # ---- 1 allowed
    try:
        if not (all(M.allowed(d) for d in M.NEW_DAYS) and M.allowed(M.CONTROL_DAY) and not M.allowed('20261023') and not M.allowed('20261105')):
            bad.append('allowed: the declared / undeclared days')
        keep = M.NEW_DAYS
        try:
            M.NEW_DAYS = keep + ('20261029', '20261030', '20261028', '20261104', '20261105')
            if any(M.allowed(d) for d in ('20261028', '20261029', '20261030')):
                bad.append('allowed: a sealed day passed')
            if any(M.allowed(d) for d in ('20261104', '20261105')):
                bad.append('allowed: a forward day passed')
            if not M.allowed('20261031') or not M.allowed('20261103'):
                bad.append('allowed: a boundary-adjacent declared day failed')
        finally:
            M.NEW_DAYS = keep
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'allowed: stopped ({e!r})'[:200])
    # ---- 2 declared, 3 build
    try:
        before = (M.QP.DAYS, M.QL.DAYS, M.KPm.DAYS, M.QP.OUT)
        with M.declared('20261031', Path('/tmp/x_qp')):
            inside = (M.QP.DAYS, M.QL.DAYS, M.KPm.DAYS, M.QP.OUT)
        if inside != (('20261031',), ('20261031',), ('20261031',), Path('/tmp/x_qp')) or (M.QP.DAYS, M.QL.DAYS, M.KPm.DAYS, M.QP.OUT) != before:
            bad.append(f'declared: the swap or the restore ({inside})')
        try:
            with M.declared('20261101', Path('/tmp/x_qp2')):
                raise RuntimeError('inside')
        except RuntimeError:
            pass
        if (M.QP.DAYS, M.QL.DAYS, M.KPm.DAYS, M.QP.OUT) != before:
            bad.append('declared: not restored after an error')
        try:
            with M.declared('20261023', Path('/tmp/x')):
                pass
            bad.append('declared: an undeclared day was swapped in')
        except Exception:                                                                   # noqa: BLE001
            pass
        calls = []; real = (M.QP.day_run, M.KPm.day_run)
        try:
            M.QP.day_run = lambda day, limit=None: calls.append(('qp', day, M.QP.OUT, M.QP.DAYS, M.KPm.DAYS))
            M.KPm.day_run = lambda day, limit=None, src=None, v1=None, out=None: (calls.append(('kp', day, Path(v1), Path(out), M.KPm.DAYS, M.QL.DAYS)), Path(out) / day)[1]
            qd, kd = M.build('20261102', Path('/tmp/root'))
        finally:
            M.QP.day_run, M.KPm.day_run = real
        want = [('qp', '20261102', Path('/tmp/root/queue_place'), ('20261102',), ('20261102',)),
                ('kp', '20261102', Path('/tmp/root/queue_place'), Path('/tmp/root/keep_pull'), ('20261102',), ('20261102',))]
        if calls != want or qd != Path('/tmp/root/queue_place/20261102') or kd != Path('/tmp/root/keep_pull/20261102') or (M.QP.DAYS, M.QP.OUT) != before[0::3]:
            bad.append(f'build: the call order / roots / globals ({calls})')
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'declared / build: stopped ({e!r})'[:220])
    # ---- 4 frames_equal / compare
    try:
        with tempfile.TemporaryDirectory(dir=str(M.KPm.I.K.IO)) as tmp:
            tmp = Path(tmp); base = pd.DataFrame(dict(a=[1, 2, 3], b=[.5, .25, .125], c=['x', 'y', 'z']))
            base.to_parquet(tmp / 'a.parquet', index=False); base.to_parquet(tmp / 'b.parquet', index=False)
            base.assign(b=[.5, .25, .126]).to_parquet(tmp / 'v.parquet', index=False); base.assign(a=base.a.astype(float)).to_parquet(tmp / 'd.parquet', index=False)
            base[['b', 'a', 'c']].to_parquet(tmp / 'o.parquet', index=False)
            if not (M.frames_equal(tmp / 'a.parquet', tmp / 'b.parquet') and not M.frames_equal(tmp / 'a.parquet', tmp / 'v.parquet')
                    and not M.frames_equal(tmp / 'a.parquet', tmp / 'd.parquet') and not M.frames_equal(tmp / 'a.parquet', tmp / 'o.parquet')):
                bad.append('frames_equal: a changed value, dtype or column order passed')

            def day_dirs(name, qrec, krec):
                q, k = tmp / name / 'q', tmp / name / 'k'; q.mkdir(parents=True); k.mkdir(parents=True)
                base.to_parquet(q / 'rows.parquet', index=False); base.to_parquet(k / 'orders.parquet', index=False); base.to_parquet(k / 'fills.parquet', index=False)
                (q / 'receipt.json').write_text(json.dumps(qrec)); (k / 'receipt.json').write_text(json.dumps(krec)); return q, k
            qr = dict(markets=288, stat=dict(states=10), asof='a', seconds=1.)
            kr = dict(stat=dict(orders=10), status=dict(eligible=9), reconciliation=dict(ok=True), fair_missing=dict(steps=0), provenance=dict(commit='c1', source=dict(x='h')),
                      asof='a', seconds=1.)
            q0, k0 = day_dirs('old', qr, kr); q1, k1 = day_dirs('new', dict(qr, asof='b', seconds=2.), dict(kr, asof='b', seconds=2., provenance=dict(commit='c2', source=dict(x='h'))))
            if not all(M.compare(q1, k1, q0, k0).values()):
                bad.append(f'compare: equal outputs failed ({M.compare(q1, k1, q0, k0)})')
            cases = [('stat', dict(qr), dict(kr, stat=dict(orders=11)), 'kp_receipt'), ('status', dict(qr), dict(kr, status=dict(eligible=8)), 'kp_receipt'),
                     ('reconciliation', dict(qr), dict(kr, reconciliation=dict(ok=False)), 'kp_receipt'), ('fair_missing', dict(qr), dict(kr, fair_missing=dict(steps=1)), 'kp_receipt'),
                     ('markets', dict(qr, markets=287), dict(kr), 'qp_receipt'), ('source', dict(qr), dict(kr, provenance=dict(commit='c1', source=dict(x='h2'))), 'kp_frozen_sources')]
            for i, (nm, q_, k_, key) in enumerate(cases):
                qx, kx = day_dirs(f'c{i}', q_, k_)
                if M.compare(qx, kx, q0, k0)[key]:
                    bad.append(f'compare: a changed {nm} passed')
            # ---- 5 control / run_day guards
            real_c, real_b = M.committed, M.build
            try:
                M.build = lambda day, root: (_ for _ in ()).throw(RuntimeError('build must not run'))
                M.committed = lambda: False
                for f, why, msg in ((lambda: M.control(tmp / 'r1'), 'control', 'not committed'), (lambda: M.run_day('20261031', tmp / 'r1'), 'run_day', 'positive control')):
                    try:
                        f(); bad.append(f'{why}: ran unguarded')
                    except Exception as e:                                                  # noqa: BLE001
                        if msg not in str(e):
                            bad.append(f'{why}: refused for another reason ({e!r})'[:200])
                (tmp / 'r2').mkdir(); (tmp / 'r2' / 'control.json').write_text(json.dumps(dict(all_ok=True)))
                for f, why, msg in ((lambda: M.control(tmp / 'r2'), 'control', 'write-once'), (lambda: M.run_day('20261031', tmp / 'r2'), 'run_day', 'not committed'),
                                    (lambda: M.run_day('20261023', tmp / 'r2'), 'run_day undeclared', 'not a new day')):
                    try:
                        f(); bad.append(f'{why}: ran unguarded')
                    except Exception as e:                                                  # noqa: BLE001
                        if msg not in str(e):
                            bad.append(f'{why}: refused for another reason ({e!r})'[:200])
                (tmp / 'r3').mkdir(); (tmp / 'r3' / 'control.json').write_text(json.dumps(dict(all_ok=False))); M.committed = lambda: True
                try:
                    M.run_day('20261031', tmp / 'r3'); bad.append('run_day: ran after a failed control')
                except Exception as e:                                                      # noqa: BLE001
                    if 'positive control' not in str(e):
                        bad.append(f'run_day: refused a failed control for another reason ({e!r})'[:200])
            finally:
                M.committed, M.build = real_c, real_b
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'frames / compare / guards: stopped ({e!r})'[:220])
    if bad:
        sys.exit(FAIL + ': ' + '; '.join(bad))
    print('source extend selftest passed: allowed (declared, sealed, forward); declared swap and restore (also on error); build order and roots; '
          'frames_equal and compare (value, dtype, column order, receipts, frozen-source hashes); control write-once / committed; run_day guards')
