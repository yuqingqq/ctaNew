"""PRE-FREEZE REHEARSAL of the headroom run and ch_checks.py (FEASIBLE-CANCEL-HEADROOM-v1; session aea5bfb9, 2026-11-22). A synthetic two-day
ASYNC-like record set (every end cause, fallback / keep / pull / slot commands, unvalued and censored marks, queue records) goes through the module's
run() and then every checker family and falsifier case, with the paths pointed at a temporary directory. No saved data is read. Required: every
family passes except A_committed (the files are not committed before the freeze); every falsifier case is refused. Usage: rehearse_checks.py"""
import gzip
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('/home/yuqing/ctaNew'); HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / 'predictfun/research'))
import pf_ladder_cancel_headroom as X                     # noqa: E402

spec = importlib.util.spec_from_file_location('ch_checks', HERE.parent / 'ch_checks.py'); CH = importlib.util.module_from_spec(spec); spec.loader.exec_module(CH)
ARMS = X.ARMS


def market(rs, day, m):
    """One synthetic market line: both arms, orders of every end cause, fills, marks, closures, queue records, commands."""
    line = dict(day=day, m=m, path_info=dict(end_rel=10000.)); arrays = {}
    for a_i, arm in enumerate(ARMS):
        orders, fills, mids, cls_, closed, opn, cmds = [], [], [], [], {}, {}, []; ahead0, first, pull, known = {}, {}, {}, {}; idx = 0
        for g in range(14):
            side = g % 2; price = 50. if side == 0 else 51.; submit = 200. * g; act = submit + 10.; kind = g % 7
            req = remove = None; refused = kind == 6; nf = 0 if kind in (5, 6) else 1 + rs.randint(4)
            if kind in (0, 1):
                req = act + 1500. + 100. * rs.randint(10); remove = req + 10.
            if g == 10:
                req, remove = 9900., 9910.                                             # removed at the venue, unacknowledged at END
            end_t = remove if remove is not None else 10000.; t_f = sorted(act + rs.uniform(0., end_t - act, nf)) if not refused else []
            if g == 12:
                t_f = [9500.]                                                          # an unvalued (last-second) fill
            q_tot = 0.
            for t in t_f:
                q = float(rs.randint(1, 20)); q_tot += q; mid = price + rs.choice([-1.5, -0.5, 0.5, 1.5])
                fills.append([float(round(t)), g, side, q, price, idx]); idx += 1
                mids.append([price, 1., price, 1., price, 1., None if t > 9000. else mid, 1., None if t > 5000. else mid + 0.5, 1., None if t > 1000. else mid, 1.])
                cls_.append(['through' if rs.rand() < .5 else 'same_price', price, q])
            size = q_tot if kind == 2 else q_tot + 20.
            rem = 0. if (refused or kind == 2) else size - q_tot
            if refused:
                closed[str(g)] = ['refused', act + 100.]
            elif kind == 2:
                closed[str(g)] = ['filled', end_t]
            elif req is not None and g != 10:
                closed[str(g)] = ['removed', remove + 100.]
            elif req is None:
                opn[str(g)] = rem
            o = dict(gid=g, side=side, rung=0., price=price, size=size, rem=rem, act=act, remove=1e18 if remove is None else remove, submit=submit, refused=refused, req=req,
                     why=None if req is None else 'policy', ahead_final=0.)
            orders.append(o)
            if not refused:
                ahead0[str(g)] = [submit, float(rs.randint(0, 50)) * (a_i == 0), size]
                live_end = min(req if req is not None else 10000., 10000.); ts = np.arange(act + 40., live_end, 50.)
                if kind != 4 and len(ts):
                    first[str(g)] = [float(ts[0]), 3., size]; known[str(g)] = dict(first=None if g % 3 == 0 else 2., n=len(ts), front=0, unknown=0, last=1., last_t=float(ts[-1]),
                                                                                  first_best_t=None, first_best_queue=None)
                    for j, t in enumerate(ts):
                        cmds.append([t, side, 0, g, 0., price, np.nan, 2, 11 if j % 9 == 4 else 1])
                    if req is not None and g != 10:
                        cmds.append([req, side, 0, g, 0., price, np.nan, 3, 12]); pull[str(g)] = [req, 1., size]
            cmds.append([submit, side, 1, -1, 0., price, np.nan, 0, 0])
        mark = 52.; sg = lambda s: 1. if s == 0 else -1.
        value = sum(sg(f[2]) * f[3] * (mark - f[4]) for f in fills); mo1 = sum(sg(f[2]) * f[3] * (mm[6] - f[4]) for f, mm in zip(fills, mids) if mm[6] is not None and f[0] <= 9000.)
        line[arm] = dict(orders=orders, fills=fills, fill_mids=mids, fill_class=cls_, closed=closed, open_at_end_gid=opn, mark=mark, profile=dict(cancel=10., fill_notice=100.),
                         queue_venue=dict(ahead0=ahead0, first=first, pull=pull), queue_known=known, _value=value, _mo1=mo1)
        arrays[f'{m}__{arm}__commands'] = np.array(sorted(cmds, key=lambda c: c[0]), float)
    return line, arrays


def main():
    tmp = Path(tempfile.mkdtemp(dir=str(X.OUT.parent))); rs = np.random.RandomState(7); saved = dict(AF_OUT=X.AF_OUT, arms_ref=X.arms_ref, frozen=X.frozen)
    try:
        af = tmp / 'af' / 'pilot'; af.mkdir(parents=True); days = ['20990101', '20990102']; complete, arm_rows, recs = {}, [], {}
        for day in days:
            lines, arrs = [], {}
            for m in (1, 2):
                ln, ar = market(rs, day, m); lines.append(ln); arrs.update(ar)
                for arm in ARMS:
                    arm_rows.append(dict(day=day, m=m, arm=arm, all_id_mo1s=ln[arm].pop('_mo1'), value=ln[arm].pop('_value')))
            complete[day] = [1, 2]; gz = gzip.compress('\n'.join(json.dumps(x) for x in lines).encode()); (af / f'{day}.jsonl.gz').write_bytes(gz)
            np.savez(af / f'{day}_arrays.npz', **arrs); nb = (af / f'{day}_arrays.npz').read_bytes()
            (af / f'{day}.json').write_text(json.dumps(dict(sha256=dict(gz=hashlib.sha256(gz).hexdigest(), npz=hashlib.sha256(nb).hexdigest()))))
            recs[f'pilot/{day}'] = hashlib.sha256((af / f'{day}.json').read_bytes()).hexdigest()
        A = pd.DataFrame(arm_rows); raw = A.to_csv(index=False).encode(); audit = json.dumps(dict(outputs={'arms.csv': hashlib.sha256(raw).hexdigest()})).encode()
        am = json.loads((REPO / X.AF_RECEIPT / 'manifest.json').read_text())
        rc = tmp / 'receipt'; (rc / 'logs').mkdir(parents=True)
        man = dict(af_records=recs, days=days, complete=complete, sources=X.sources(), caps=dict(round_core_hours=X.CAP_ROUND, review_reserve=X.REVIEW_RESERVE, line_ceiling=X.LINE_CEILING,
                   unlogged_inspection_s=X.UNLOGGED_S, tail_s=X.TAIL_S), line_receipts=am['line_receipts'] + ['ladder_async_fast_20261122'],
                   line_reviews=am['line_reviews'] + ['20261122_async_fast_review', '20261122_queue_fill_tradeoff_amendment'])
        (rc / 'manifest.json').write_text(json.dumps(man)); (rc / 'logs' / 'projection.json').write_text(json.dumps(dict(fits=True)))
        X.AF_OUT = tmp / 'af'; X.arms_ref = lambda: pd.read_csv(io.BytesIO(raw), dtype={'day': str}); X.frozen = lambda: True
        X.run(out=tmp / 'out', require_frozen=False, man_path=rc / 'manifest.json', proj_path=rc / 'logs' / 'projection.json')
        real_blob = CH.blob
        CH.R, CH.D, CH.AF, CH.EXPECT_MARKETS = rc, tmp / 'out', tmp / 'af', 4
        CH.blob = lambda rel: audit if rel.endswith('ladder_async_fast_20261122/outputs/audit.json') else (gzip.compress(raw) if rel.endswith('ladder_async_fast_20261122/outputs/arms.csv.gz')
                                                                                                     else real_blob(rel))
        E = CH.load(); E['mine'], E['pan_mine'] = CH.mine(E)
        g = E['gen']; nm = dict(no_feasible_callback='NO_OPPORTUNITY', incomplete_valuation='INCOMPLETE', pure_observed_loss='PURE_BAD', compensated_mixed='MIXED_COMPENSATED',
                                uncompensated_mixed='MIXED_UNCOMPENSATED', mixed_tie='MIXED_TIE')
        sub = np.where(g.a_suf_n == 0, 'NO_SUFFIX_FILLS', np.where((g.a_suf_G > 1e-6) & (g.a_suf_L <= 1e-6), 'GOOD_ONLY', 'NEAR_ZERO'))
        E['codex'] = pd.DataFrame(dict(day=g.day, m=g.m, arm=g.arm, gid=g.gid, category=np.where(g.a_cls == 'good_only_or_no_fill', sub, g.a_cls.map(nm)), good=g.a_suf_G, loss=g.a_suf_L,
                                       prefix_good=g.a_pre_G, prefix_loss=g.a_pre_L, whole_1s=g.v_keep, whole_end=g.end_keep, anchor=g.a_cut,
                                       known_end_cause=np.where(g.end_cause.eq('end_open') | g.unacked_at_end, 'open_or_unnotified_at_END',
                                                                g.end_cause.map(dict(recorded_cancellation='removed', filled_out='filled', refused='refused')))))
        E['src']['codex'] = True; base = CH.run_checks(E)  # a self-consistent stand-in for the Codex table: exercises the X code path only
        fz = CH.falsify(dict(E), dict(base, all_ok=all(v for k, v in base['checks'].items() if k != 'A_committed')))
        G = pd.read_csv(tmp / 'out' / 'generations.csv'); cover = dict(a_classes=sorted(G.a_cls.unique()), b_classes=sorted(G.b_cls.unique()), end_causes=sorted(G.end_cause.unique()),
                                                                     btg=int(G.seq_bad_then_good.sum()), gtb=int(G.seq_good_then_bad.sum()), fallback_keeps=int(G.n_keeps_fallback.sum()))
        failed = [k for k, v in base['checks'].items() if not v and k != 'A_committed']; bad_cases = [k for k, v in fz['cases'].items() if not v['refused']]
        res = dict(checks=base['checks'], failed_except_committed=failed, falsifier_unrefused=bad_cases, coverage=cover, details=base['details'],
                   rehearsal_ok=bool(not failed and not bad_cases))
        print(json.dumps(res, indent=1, default=str)[:5000])
        if not res['rehearsal_ok']:
            raise SystemExit('REHEARSAL FAILED')
    finally:
        X.AF_OUT, X.arms_ref, X.frozen = saved['AF_OUT'], saved['arms_ref'], saved['frozen']; shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
