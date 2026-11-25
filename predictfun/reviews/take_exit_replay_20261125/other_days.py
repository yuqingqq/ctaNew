"""TAKE-EXIT-REPLAY v3 (amendment_v3_other_days.md): the frozen take + maker-exit rule on 20261023 / 20261031 / 20261101-03 under the
LENIENT40 (frozen v2 simulator) and STRICT58 (review bound) settings. Reuses bn_events.py (allow-list swapped only), exit_replay.py,
more_markets.py and the review's strict_rerun.py unchanged.
Usage: run_light.sh txr3_<stage> -- python3 predictfun/reviews/take_exit_replay_20261125/other_days.py bn|run|check|report
"""
import collections
import gzip
import json
import math
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
R = H.parents[2]
for d in (H, H.parent / 'take_exit_review_20261125', H.parent / 'take_signal_20261125'):
    sys.path.insert(0, str(d))
import more_markets as MM  # noqa: E402
import strict_rerun as SR  # noqa: E402
import bn_events as BEV  # noqa: E402

ER, TR, X, LB, BE, P = MM.ER, MM.TR, MM.X, MM.LB, MM.BE, MM.P
OUT3 = X.DATA / 'take_exit_replay_v3_20261125'
RCPT = H / 'v3_bn'
DAYS3 = ('20261023', '20261031', '20261101', '20261102', '20261103')
STEMS = (tuple(f'20261023_{h:02d}' for h in range(24)) + ('20261024_00',)
         + tuple(f'{d}_{h:02d}' for d in ('20261031', '20261101', '20261102', '20261103') for h in range(24)))
FWD_MS = 1793750400000                                  # 2026-11-04 00:00Z, the loader's forward boundary
SEG_BOUNDS = {'A': (1792713600000, 1792803600000), 'B': (1793404800000, FWD_MS)}   # Oct 23 00:00 .. Oct 24 01:00; Oct 31 00:00 .. Nov 4 00:00
SEG_OF = {'20261023': 'A', '20261031': 'B', '20261101': 'B', '20261102': 'B', '20261103': 'B'}
SETTINGS = ('LENIENT40', 'STRICT58')
CPU_CAP = 1100.
FILES = ['amendment_v3_other_days.md', 'other_days.py']
EPS = 1e-9
_bn = {}


def build_changes(stems, seg_start, out, rcpt):
    """bn_events.main's loop (same parse, dedupe, carry and emission; the change points from bn_events.changes) with two additions:
    rows before a segment's start are dropped right after parsing (no row of a sealed or unlisted day is kept), and the carry is
    discarded at a segment start (no cross-gap continuity)."""
    BEV.ALLOWED = tuple(stems)
    out.parent.mkdir(parents=True, exist_ok=True)
    assert not out.exists(), 'write-once'
    rec = dict(files={}, rows=0, dropped_before_segment=0, dup_u=0, invalid=0)
    buf, ek, em, emit_lo, last_mid = None, [], [], -np.inf, None
    for stem in stems:
        p = BEV.path_of(stem)
        raw = p.read_bytes()
        rec['files'][stem] = dict(sha256=__import__('hashlib').sha256(raw).hexdigest(), bytes=len(raw))
        cols = dict(header=None, names=['recv', 'E', 'T', 'u', 'bid', 'bq', 'ask', 'aq'], usecols=['E', 'u', 'bid', 'ask'], on_bad_lines='skip')
        try:
            df = pd.read_csv(p, dtype={'E': 'float64', 'u': 'float64', 'bid': 'float64', 'ask': 'float64'}, **cols)
        except ValueError:
            df = pd.read_csv(p, dtype=str, **cols).apply(pd.to_numeric, errors='coerce')
        n0 = len(df)
        df = df.dropna()
        okr = (df.bid > 0) & (df.ask > df.bid)
        rec['invalid'] += int(n0 - okr.sum())
        df = df[okr]
        nd = len(df)
        df = df.drop_duplicates('u', keep='first')
        rec['dup_u'] += int(nd - len(df))
        k, u, m = df.E.to_numpy(np.int64), df.u.to_numpy(np.int64), (df.bid.to_numpy() + df.ask.to_numpy()) / 2
        if stem in seg_start:
            keep = k >= seg_start[stem]
            rec['dropped_before_segment'] += int((~keep).sum())
            k, u, m = k[keep], u[keep], m[keep]
            buf, last_mid = None, None
        rec['rows'] += len(k)
        if buf is not None:
            k, u, m = np.r_[buf[0], k], np.r_[buf[1], u], np.r_[buf[2], m]
            uq, first = np.unique(u, return_index=True)
            rec['dup_u'] += int(len(u) - len(uq))
            k, u, m = k[first], u[first], m[first]
        ck, cm = BEV.changes(k, u, m)
        hi = k.max() - BEV.CARRY_MS if stem != stems[-1] else np.inf
        sel = (ck > emit_lo) & (ck <= hi)
        if last_mid is not None and sel.any():
            i0 = int(np.argmax(sel))
            if cm[i0] == last_mid:
                sel[i0] = False
        ek.append(ck[sel])
        em.append(cm[sel])
        if sel.any():
            last_mid = cm[sel][-1]
        if np.isfinite(hi):
            emit_lo = hi
        keep = k > k.max() - 2 * BEV.CARRY_MS
        buf = (k[keep], u[keep], m[keep])
    K, M = np.concatenate(ek), np.concatenate(em)
    assert np.all(np.diff(K) > 0)
    pd.DataFrame(dict(k_ms=K, mid=M)).to_parquet(out, index=False)
    rec.update(changes=int(len(K)), out_sha256=TR.sha(out))
    rcpt.parent.mkdir(parents=True, exist_ok=True)
    rcpt.write_text(json.dumps(rec, indent=1) + '\n')
    return K, M


def bn_build():
    assert all(s[:8] in DAYS3 + ('20261024',) for s in STEMS)
    OUT3.mkdir(parents=True, exist_ok=True)
    build_changes(STEMS, {'20261023_00': SEG_BOUNDS['A'][0], '20261031_00': SEG_BOUNDS['B'][0]}, OUT3 / 'bn_changes.parquet', RCPT / 'bn_events.json')


def bn_validate():
    """The same builder on the consumed Oct 24 stems must reproduce the original bn_changes exactly inside Oct 24 00:00:30 .. 23:59:30."""
    stems = ('20261023_23',) + tuple(f'20261024_{h:02d}' for h in range(24))
    tmp = OUT3 / 'validation_oct24_bn_changes.parquet'
    K, M = build_changes(stems, {}, tmp, RCPT / 'validation_oct24.json')
    o = pd.read_parquet(X.DATA / 'take_signal_20261125/bn_changes.parquet')
    lo, hi = 1792800030000, 1792886370000
    a = (K >= lo) & (K < hi)
    b = (o.k_ms.to_numpy() >= lo) & (o.k_ms.to_numpy() < hi)
    same = np.array_equal(K[a], o.k_ms.to_numpy()[b]) and np.array_equal(M[a], o.mid.to_numpy()[b])
    print(json.dumps(dict(changes_compared=int(a.sum()), identical=bool(same))))
    assert same, 'the v3 builder does not reproduce the original Oct 24 change points'


def seg_arrays(seg):
    if seg not in _bn:
        d = pd.read_parquet(OUT3 / 'bn_changes.parquet')
        k, m = d.k_ms.to_numpy(np.int64), d.mid.to_numpy(float)
        lo, hi = SEG_BOUNDS[seg]
        sel = (k >= lo) & (k < hi)
        _bn[seg] = (k[sel], m[sel])
    return _bn[seg]


def make_cache3(t0, end, seg):
    bk, bm = seg_arrays(seg)
    nt = int(np.ceil(end / BE.STEP))
    d = t0 + BE.STEP * np.arange(nt)
    t = bk + MM.B_IN
    j0 = np.searchsorted(t, d, 'right') - 1
    j1 = np.searchsorted(t, d - 250, 'right') - 1
    ok = (j0 >= 0) & (j1 >= 0)
    r = np.where(ok, 1e4 * np.log(bm[np.maximum(j0, 0)] / bm[np.maximum(j1, 0)]), np.nan)
    x = {1: r.astype(np.float32), 0: (-r).astype(np.float32)}
    st = {s: np.where(np.isfinite(x[s]), 0, P.ST_CODES.index('nan')).astype(np.int8) for s in (0, 1)}
    return dict(x=x, age={s: np.full(nt, np.nan, np.float32) for s in (0, 1)}, status=st)


def bound():
    for f in FILES:
        assert subprocess.check_output(['git', 'show', 'HEAD:' + str((H / f).relative_to(R))], cwd=R) == (H / f).read_bytes(), f
    for p in [H / 'exit_replay.py', H / 'more_markets.py', H.parent / 'take_exit_review_20261125/strict_rerun.py', Path(BEV.__file__),
              H.parent / 'signal_take_replay_20261125/take_replay.py']:
        assert subprocess.check_output(['git', 'show', 'HEAD:' + str(p.relative_to(R))], cwd=R) == p.read_bytes(), p
    rc = json.loads((RCPT / 'bn_events.json').read_text())
    assert tuple(rc['files'].keys()) == STEMS, 'Binance stems read differ from the allow-list'
    assert json.loads((RCPT / 'validation_oct24.json').read_text())['changes'] > 0, 'run bn_validate first'


def one(p, cache, old, edges, bps, setting, take=True):
    TR.TAKE_DELAY = 40. if setting == 'LENIENT40' else 58.
    try:
        if setting == 'LENIENT40':
            rec = ER.run_one(p, cache, old, edges, bps, 'PASSIVE_EXIT', take=take)
        else:
            rec = SR.run_one(p, cache, old, edges, bps, strict=True) if take else None
        if rec is not None:
            assert all(abs(v['act'] - v['submit'] - TR.TAKE_DELAY) < 1e-9 for v in rec['iocs'].values())
        return rec
    finally:
        TR.TAKE_DELAY = 40.


def run():
    resource.setrlimit(resource.RLIMIT_CPU, (int(CPU_CAP) + 60, int(CPU_CAP) + 61))
    bound()
    assert not (OUT3 / 'summary.json').exists()
    edges = TR.T.P.frozen_edges()
    src, rows, census = MM.KP.Tape(), [], []
    t0 = time.process_time()
    for di, day in enumerate(DAYS3):
        reg = MM.registry(day).set_index('mid')
        off = src.offset(day)
        for h, g in reg.groupby('hour'):
            for m, st, p, info in MM.worlds(day, [int(x) for x in g.index], src, off, reg):
                ws = float(reg.loc[m, 'ws']) * 1000.
                if ws + 300000. >= FWD_MS:
                    st = 'boundary_censored_forward_tail'
                row = dict(day=day, mid=m, hour=int(h), status=st)
                census.append(row)
                if st != 'complete':
                    continue
                bps = float(reg.loc[m, 'fee_bps'])
                assert bps == 200.
                cache = make_cache3(float(info['t_start']), float(info['end_rel']), SEG_OF[day])
                old = dict(end=float(info['end_rel']), ident=MM.ident_of(day, m, ws, info, p))
                for setting in SETTINGS:
                    rec = one(p, cache, old, edges, bps, setting)
                    rec.update(day=day, mid=m, hour=int(h), setting=setting)
                    fn = OUT3 / f'{m}_{setting}.json.gz'
                    assert not fn.exists()
                    fn.write_bytes(gzip.compress(json.dumps(rec, allow_nan=False).encode(), mtime=0))
                    rows.append(dict(day=day, mid=m, hour=int(h), setting=setting, metrics=rec['metrics'], sha256=TR.sha(fn)))
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2500 * 1024
        el = time.process_time() - t0
        if di == 0:
            proj = el * len(DAYS3) * 1.2
            TR.dump(H / 'v3_projection.json', dict(first_day=day, cpu_s=el, projection_cpu_s=proj, cap=CPU_CAP, proceed=proj <= CPU_CAP))
            assert proj <= CPU_CAP, 'projection stop'
        TR.dump(OUT3 / 'progress.json', dict(day=day, rows=len(rows), cpu_s=el))
        print(day, 'done', len(rows), round(el, 1), flush=True)
    TR.dump(OUT3 / 'summary.json', dict(rows=rows, census=census, cpu_s=time.process_time() - t0, status='COMPLETE'))


def check():
    sm = json.loads((OUT3 / 'summary.json').read_text())
    res = {}

    def ok(name, cond, detail=''):
        res[name] = dict(ok=bool(cond), detail=detail)
        print(('PASS ' if cond else 'FAIL ') + name, detail)
    rc = json.loads((RCPT / 'bn_events.json').read_text())
    ok('V1_binance_stems_equal_allow_list', tuple(rc['files'].keys()) == STEMS, f"{len(rc['files'])} stems")
    edges = TR.T.P.frozen_edges()
    b = collections.Counter()
    for r in sm['rows']:
        fn = OUT3 / f"{r['mid']}_{r['setting']}.json.gz"
        assert TR.sha(fn) == r['sha256']
        rec = json.loads(gzip.decompress(fn.read_bytes()))
        dl = 40. if r['setting'] == 'LENIENT40' else 58.
        b['delay'] += not all(abs(v['act'] - v['submit'] - dl) < 1e-9 for v in rec['iocs'].values())
        io = {int(k): v for k, v in rec['iocs'].items()}
        fills = {(int(x[1]), round(float(x[3]), 9), float(x[4])) for x in rec['fills']}
        for x in rec['ioc_results']:
            if r['setting'] == 'STRICT58' and x.get('strict_cut'):
                b['cap'] += not (x['filled'] <= x['strict_cap'] + EPS)
            if r['setting'] == 'LENIENT40':
                b['lenient_has_cap'] += 'strict_cap' in x
            if x['filled'] <= EPS:
                continue
            b['n_ioc'] += 1
            a = io[int(x['gid'])]
            inside = x['price'] <= a['limit'] + EPS if a['side'] == 0 else x['price'] >= a['limit'] - EPS
            fee = 200. / 1e4 * min(x['price'], 100 - x['price']) * x['filled']
            b['ioc'] += not (inside and x['filled'] <= a['q'] + EPS and abs(fee - x['fee_c']) < 1e-9 and (int(x['gid']), round(float(x['filled']), 9), float(x['price'])) in fills)
        grid = np.array(rec['grid'])
        for c in rec['commands']:
            if c[7] == 'POST':
                b['n_post'] += 1
                Ik = grid[int(round(c[0] / BE.STEP)), 1]
                need = 1 if Ik > EPS else 0 if Ik < -EPS else None
                b['post'] += not (c[4] == 0 and c[1] == need and c[6] <= abs(Ik) + EPS)
        mt = rec['metrics']
        b['ident'] += not (abs(rec['value'] - (rec['cash'] + rec['position'] * rec['mark'])) < 1e-6 and abs(mt['completed_c'] + mt['residual_c'] - mt['total_c']) < 1e-6
                           and abs(mt['total_c'] - rec['value']) < 1e-6)
        idn = rec['ident']
        cache = make_cache3(float(idn['start_abs_ms']), float(idn['end_rel']), SEG_OF[r['day']])
        trig = TR.triggers(TR.on_arrays(cache, edges, len(rec['grid'])))
        want = {(int(k), s) for s in (0, 1) for k in np.where(trig[s])[0]}
        b['trig'] += {(x['k'], x['guard_side']) for x in rec['signal_log']} != want
    ok('V2_delays_40_58', b['delay'] == 0, f"{len(sm['rows'])} ledgers")
    ok('V3_strict_cap_respected', b['cap'] == 0 and b['lenient_has_cap'] == 0, f"cap violations {b['cap']}, lenient rows with a cap {b['lenient_has_cap']}")
    ok('V4_ioc_rederived', b['n_ioc'] > 0 and b['ioc'] == 0, f"{b['n_ioc']} filled IOCs, {b['ioc']} mismatches")
    ok('V5_exit_posts_closing_rung0', b['n_post'] > 0 and b['post'] == 0, f"{b['n_post']} posts, {b['post']} violations")
    ok('V6_ledger_identities', b['ident'] == 0, f"{len(sm['rows'])} ledgers")
    ok('V7_triggers_recomputed', b['trig'] == 0, f"{len(sm['rows'])} ledgers")
    first = next(c for c in sm['census'] if c['status'] == 'complete')
    reg = MM.registry(first['day']).set_index('mid')
    src = MM.KP.Tape()
    for m, st, p, info in MM.worlds(first['day'], [first['mid']], src, src.offset(first['day']), reg):
        cache = make_cache3(float(info['t_start']), float(info['end_rel']), SEG_OF[first['day']])
        old = dict(end=float(info['end_rel']), ident=MM.ident_of(first['day'], m, float(reg.loc[m, 'ws']) * 1000., info, p))
        z = one(p, cache, old, edges, 200., 'LENIENT40', take=False)
        ok('V8_no_trigger_falsifier', len(z['fills']) == 0 and len(z['orders']) == 0 and not z['iocs'], f"market {m}, takes disabled")
    res['all_pass'] = all(v['ok'] for v in res.values() if isinstance(v, dict))
    TR.dump(H / 'v3_checks.json', res)
    print('ALL PASS' if res['all_pass'] else 'SOME CHECK FAILED')
    sys.exit(0 if res['all_pass'] else 1)


def report():
    sm = json.loads((OUT3 / 'summary.json').read_text())
    cen = collections.Counter((c['day'], c['status']) for c in sm['census'])
    df = pd.DataFrame([dict(day=r['day'], mid=r['mid'], hour=r['hour'], setting=r['setting'], **{k: v for k, v in r['metrics'].items() if not isinstance(v, dict)})
                       for r in sm['rows']])
    df.to_csv(H / 'v3_per_market.csv', index=False)
    out = dict(census={d: {s: n for (dd, s), n in cen.items() if dd == d} for d in DAYS3}, in_sample=dict(LENIENT40_per_market=14.00, STRICT58_per_market=5.76))
    for s in SETTINGS:
        x = df[df.setting == s]
        dd = x.groupby('day').net_c.sum()
        absm = x.net_c.abs().sort_values(ascending=False)
        out[s] = dict(markets=int(len(x)), net_usd=float(x.net_c.sum() / 100), per_market_usd=float(x.net_c.mean() / 100),
                      day_net_usd={k: float(v / 100) for k, v in dd.items()}, day_markets={k: int(v) for k, v in x.groupby('day').size().items()},
                      day_per_market_usd={k: float(v / 100) for k, v in x.groupby('day').net_c.mean().items()}, days_positive=int((dd > 0).sum()),
                      markets_positive=int((x.net_c > 0).sum()), top10_abs_share=float(absm.iloc[:10].sum() / absm.sum()) if absm.sum() > 0 else None,
                      takes_filled=int(x.takes_filled.sum()), fees_usd=float(x.fees_c.sum() / 100), completed_usd=float(x.completed_c.sum() / 100),
                      net_without_young_exit_value_usd=float((x.net_c.sum() - x.young_exit_value_c.sum()) / 100), mean_abs_inventory=float(x.mean_abs_inventory.mean()))
    st = out['STRICT58']
    ratio = st['per_market_usd'] / 5.76
    out['readings'] = dict(holds_out_of_sample=bool(st['net_usd'] > 0 and st['days_positive'] >= 4), strict58_days_positive=st['days_positive'],
                           strict58_per_market_ratio_to_in_sample=ratio, similar_size=bool(0.5 <= ratio <= 2.0),
                           lenient40_per_market_ratio_to_in_sample=out['LENIENT40']['per_market_usd'] / 14.00)
    TR.dump(H / 'v3_readings.json', out)
    print(json.dumps(out, indent=1, default=float))


def resume():
    """Resource-only amendment R1: reuse the 20261023 ledgers (verified against a fresh census), run 20261031..20261103."""
    resource.setrlimit(resource.RLIMIT_CPU, (1160, 1161))
    bound()
    assert not (OUT3 / 'summary.json').exists()
    spent = 0.
    for line in open(R / 'predictfun/research/run_light.log'):
        if line.startswith('2026-11-25T') and 'slot=claude' in line and (' tag=txr3_run ' in line or ' tag=txr3_resume ' in line):
            spent += float(line.split('cpu_user_s=')[1].split()[0])
    assert spent > 0
    cap_left = 1300. - spent
    edges = TR.T.P.frozen_edges()
    src, rows, census = MM.KP.Tape(), [], []
    t0 = time.process_time()
    day = DAYS3[0]
    reg = MM.registry(day).set_index('mid')
    off = src.offset(day)
    comp = set()
    for h, g in reg.groupby('hour'):
        for m, st, p, info in MM.worlds(day, [int(x) for x in g.index], src, off, reg):
            census.append(dict(day=day, mid=m, hour=int(h), status=st))
            if st == 'complete':
                comp.add((m, int(h)))
    saved = {}
    for fn in sorted(OUT3.glob('*_LENIENT40.json.gz')) + sorted(OUT3.glob('*_STRICT58.json.gz')):
        rec = json.loads(gzip.decompress(fn.read_bytes()))
        if rec.get('day') != day:
            continue
        saved.setdefault((int(rec['mid']), int(rec['hour'])), {})[rec['setting']] = (rec['metrics'], TR.sha(fn))
        del rec
    assert set(saved) == comp and all(set(v) == set(SETTINGS) for v in saved.values()), 'saved 20261023 ledgers differ from the fresh census'
    for (m, h), v in sorted(saved.items()):
        for setting in SETTINGS:
            met, hsh = v[setting]
            rows.append(dict(day=day, mid=m, hour=h, setting=setting, metrics=met, sha256=hsh, reused=True))
    redo = 0
    for di, day in enumerate(DAYS3[1:]):
        reg = MM.registry(day).set_index('mid')
        off = src.offset(day)
        for h, g in reg.groupby('hour'):
            for m, st, p, info in MM.worlds(day, [int(x) for x in g.index], src, off, reg):
                ws = float(reg.loc[m, 'ws']) * 1000.
                if ws + 300000. >= FWD_MS:
                    st = 'boundary_censored_forward_tail'
                census.append(dict(day=day, mid=m, hour=int(h), status=st))
                if st != 'complete':
                    continue
                bps = float(reg.loc[m, 'fee_bps'])
                assert bps == 200.
                cache = make_cache3(float(info['t_start']), float(info['end_rel']), SEG_OF[day])
                old = dict(end=float(info['end_rel']), ident=MM.ident_of(day, m, ws, info, p))
                for setting in SETTINGS:
                    rec = one(p, cache, old, edges, bps, setting)
                    rec.update(day=day, mid=m, hour=int(h), setting=setting)
                    fn = OUT3 / f'{m}_{setting}.json.gz'
                    blob = gzip.compress(json.dumps(rec, allow_nan=False).encode(), mtime=0)
                    if fn.exists():                                  # written by the stopped resume attempt: must be byte-identical
                        assert fn.read_bytes() == blob, ('non-deterministic ledger', m, setting)
                        redo += 1
                    else:
                        fn.write_bytes(blob)
                    rows.append(dict(day=day, mid=m, hour=int(h), setting=setting, metrics=rec['metrics'], sha256=TR.sha(fn)))
                    del rec, blob
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2500 * 1024
        el = time.process_time() - t0
        if di == 0:
            proj = el * 3 * 1.1
            TR.dump(H / 'v3_projection_r1.json', dict(first_resumed_day=day, cpu_s=el, spent_before=spent, projection_cpu_s=spent + proj, cap=1300., proceed=proj <= cap_left))
            assert proj <= cap_left, 'projection stop (R1)'
        print(day, 'done', len(rows), round(el, 1), flush=True)
    TR.dump(OUT3 / 'summary.json', dict(rows=rows, census=census, cpu_s=spent + time.process_time() - t0, status='COMPLETE', amendment='R1',
                                        resume_byte_identical_rewrites=redo))


if __name__ == '__main__':
    {'bn': bn_build, 'bn_validate': bn_validate, 'run': run, 'resume': resume, 'check': check, 'report': report}[sys.argv[1]]()
