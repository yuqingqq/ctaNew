"""TAKE-EXIT-REPLAY v2 (amendment_v2_more_markets.md): census of every Oct 24-27 market with the unchanged loader / path builder, then the
frozen v1 arms PASSIVE_EXIT and HOLD (exit_replay.py, ee0f16ee6) on the complete paths, with the builder-exact signal cache generated
from the raw-tape change points. Gates: cache bits and ledgers must equal v1 on its 19 markets.
Usage: run_light.sh txr2_<stage> -- python3 predictfun/reviews/take_exit_replay_20261125/more_markets.py census|run|check|report
"""
import collections
import gzip
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
R = H.parents[2]
sys.path.insert(0, str(H))
import exit_replay as ER  # noqa: E402

TR, X, LB, BE, P = ER.TR, ER.X, ER.LB, ER.BE, ER.P
KP = X.AF.M.KP
OUT2 = X.DATA / 'take_exit_replay_v2_20261125'
BN = X.DATA / 'take_signal_20261125/bn_changes.parquet'
S1 = X.DATA / 'step1'
DAYS = ('20261024', '20261025', '20261026', '20261027')
ARMS2 = ('PASSIVE_EXIT', 'HOLD')
B_IN = 4
CPU_CAP = 900.
ARM_CPU_S, BATCH_CPU_S = 0.92, 1.2                      # v1 measured: (72.5 s - 19 x 1.07 s loading) / 57 arm runs; one hour batch load
PAR_FIELDS = ['fills', 'orders', 'requests', 'value', 'cash', 'position', 'commands', 'grid', 'signal_log', 'ioc_results']
FILES = ['amendment_v2_more_markets.md', 'more_markets.py']
EPS = 1e-9
_bn = None


def bn():
    global _bn
    if _bn is None:
        d = pd.read_parquet(BN)
        _bn = (d.k_ms.to_numpy(np.int64), d.mid.to_numpy(float))
    return _bn


def make_cache(t0, end):
    bk, bm = bn()
    nt = int(np.ceil(end / BE.STEP))
    d = t0 + BE.STEP * np.arange(nt)
    t = bk + B_IN
    j0 = np.searchsorted(t, d, 'right') - 1
    j1 = np.searchsorted(t, d - 250, 'right') - 1
    ok = (j0 >= 0) & (j1 >= 0)
    r = np.where(ok, 1e4 * np.log(bm[np.maximum(j0, 0)] / bm[np.maximum(j1, 0)]), np.nan)
    x = {1: r.astype(np.float32), 0: (-r).astype(np.float32)}
    st = {s: np.where(np.isfinite(x[s]), 0, P.ST_CODES.index('nan')).astype(np.int8) for s in (0, 1)}
    return dict(x=x, age={s: np.full(nt, np.nan, np.float32) for s in (0, 1)}, status=st)


def ident_of(day, m, ws, info, p):
    return dict(config='LADDER_REST_INVENTORY_v1', day=day, end_abs_ms=float(info['t_start'] + info['end_rel']), end_rel=float(info['end_rel']),
                instrument_id=int(m), mark_source='QP.build_path supported-end mark', opening_ms=float(ws), start_abs_ms=float(info['t_start']),
                tape_id=BE.tape_id(p['frames'], p['trades'], p['mark']))


def registry(day):
    r = pd.read_parquet(S1 / day / 'registry.parquet', columns=['mid', 'ws', 'decimal_precision', 'fee_bps'])
    r['hour'] = ((r.ws.astype(np.int64) % 86400) // 3600).astype(int)
    return r


def worlds(day, mids, src, off, reg):
    """Yield (mid, status, path, info) for one opening-hour batch through the unchanged loader and path builder."""
    data = src.load(day, list(mids), off)
    seen = set()
    for W in src.pairs(data):
        m = int(W[0]['m'])
        seen.add(m)
        if int(reg.loc[m, 'decimal_precision']) != 2:
            yield m, 'precision_not_2', None, None
            continue
        p, st, info = X.M.market_path(W, float(reg.loc[m, 'ws']) * 1000.)
        yield m, st, p, info
    for m in mids:
        if m not in seen:
            yield int(m), 'loader_excluded', None, None


def census():
    out = OUT2 / 'census.json'
    OUT2.mkdir(parents=True, exist_ok=True)
    assert not out.exists(), 'write-once'
    resource.setrlimit(resource.RLIMIT_CPU, (400, 401))
    t0 = time.process_time()
    src, rows = KP.Tape(), []
    for day in DAYS:
        reg = registry(day).set_index('mid')
        off = src.offset(day)
        for h, g in reg.groupby('hour'):
            for m, st, p, info in worlds(day, [int(x) for x in g.index], src, off, reg):
                row = dict(day=day, mid=m, hour=int(h), status=st)
                if st == 'complete':
                    row.update(t_start=float(info['t_start']), end_rel=float(info['end_rel']), tape_id=BE.tape_id(p['frames'], p['trades'], p['mark']))
                rows.append(row)
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2500 * 1024, 'VmHWM guard'
    st = collections.Counter(r['status'] for r in rows)
    comp = [r for r in rows if r['status'] == 'complete']
    proj = len(comp) * len(ARMS2) * ARM_CPU_S * 1.2 + 24 * len(DAYS) * BATCH_CPU_S
    if proj <= CPU_CAP:
        sel = comp
        rule = 'all complete markets'
    else:
        k = int((CPU_CAP - 24 * len(DAYS) * BATCH_CPU_S) / (len(ARMS2) * ARM_CPU_S * 1.2))
        byday = {d: sorted([r for r in comp if r['day'] == d], key=lambda r: hashlib.sha256(f"txr_v2|{r['day']}|{r['mid']}".encode()).hexdigest()) for d in DAYS}
        sel, i = [], 0
        while len(sel) < k and any(i < len(v) for v in byday.values()):
            for d in DAYS:
                if i < len(byday[d]) and len(sel) < k:
                    sel.append(byday[d][i])
            i += 1
        rule = f'hash-ordered round-robin prefix of {k} complete markets'
    TR.dump(out, dict(rows=rows, status_counts=dict(st), by_day={d: dict(collections.Counter(r['status'] for r in rows if r['day'] == d)) for d in DAYS},
                      complete=len(comp), projection_cpu_s=proj, cap_cpu_s=CPU_CAP, selection_rule=rule, selected=[(r['day'], r['mid']) for r in sel],
                      cpu_s=time.process_time() - t0))
    print(json.dumps(dict(status_counts=dict(st), complete=len(comp), projection=proj, rule=rule, selected=len(sel), cpu_s=round(time.process_time() - t0, 2))))


def bound():
    for f in FILES + ['contract.md', 'exit_replay.py', 'manifest.json']:
        assert subprocess.check_output(['git', 'show', 'HEAD:' + str((H / f).relative_to(R))], cwd=R) == (H / f).read_bytes(), f


def gl(log):
    return [list(r[:6]) + list(r[7:]) for r in log]


def run():
    resource.setrlimit(resource.RLIMIT_CPU, (int(CPU_CAP) + 100, int(CPU_CAP) + 101))
    bound()
    cen = json.loads((OUT2 / 'census.json').read_text())
    assert not (OUT2 / 'summary.json').exists()
    sel = {(d, int(m)) for d, m in cen['selected']}
    comp = {(r['day'], r['mid']): r for r in cen['rows'] if r['status'] == 'complete'}
    v1 = {(d, int(m)) for d, m, _ in json.loads((H / 'manifest.json').read_text())['cases']}
    assert v1 <= set(comp), 'every v1 market must be complete in the census'
    sel |= v1                                               # the parity gate always runs on the 19 v1 markets
    edges = TR.T.P.frozen_edges()
    src, rows, gates = KP.Tape(), [], []
    t0 = time.process_time()
    for day in DAYS:
        reg = registry(day).set_index('mid')
        off = src.offset(day)
        for h, g in reg.groupby('hour'):
            want = [int(x) for x in g.index if (day, int(x)) in sel]
            if not want:
                continue
            for m, st, p, info in worlds(day, want, src, off, reg):
                assert st == 'complete' and comp[(day, m)]['tape_id'] == BE.tape_id(p['frames'], p['trades'], p['mark']), (day, m, st)
                ws = float(reg.loc[m, 'ws']) * 1000.
                cache = make_cache(float(info['t_start']), float(info['end_rel']))
                old = dict(end=float(info['end_rel']), ident=ident_of(day, m, ws, info, p))
                bps = float(reg.loc[m, 'fee_bps'])
                assert bps == 200.
                if (day, m) in v1:
                    case = [c for c in json.loads((H / 'manifest.json').read_text())['cases'] if c[1] == m][0]
                    saved = X.records(day, case[2])[m]['CURRENT_ASYNC_FAST10']
                    pp = 'pilot' if case[2] == 'pilot' else 'days'
                    sc = X.AF.LG.cache_of(np.load(X.P.OUT / pp / f'{day}_arrays.npz'), m)
                    same_cache = all(np.array_equal(sc['x'][s].view(np.uint32), cache['x'][s].view(np.uint32)) and np.array_equal(sc['status'][s], cache['status'][s])
                                     for s in (0, 1))
                    same_ident = {k: v for k, v in saved['ident'].items()} == old['ident']
                    assert same_cache and same_ident, (m, same_cache, same_ident)
                for arm in ARMS2:
                    rec = ER.run_one(p, cache, old, edges, bps, arm)
                    rec.update(day=day, mid=m, hour=int(h))
                    if (day, m) in v1:
                        prev = json.loads(gzip.decompress((ER.OUT / f'{m}_{arm}.json.gz').read_bytes()))
                        r2 = json.loads(json.dumps(rec, allow_nan=False))
                        same = {f: r2[f] == prev[f] for f in PAR_FIELDS}
                        same['guard_log_without_age'] = gl(r2['guard_log']) == gl(prev['guard_log'])
                        gates.append(dict(day=day, mid=m, arm=arm, same=same))
                        assert all(same.values()), (m, arm, same)
                    fn = OUT2 / f'{m}_{arm}.json.gz'
                    assert not fn.exists()
                    fn.write_bytes(gzip.compress(json.dumps(rec, allow_nan=False).encode(), mtime=0))
                    rows.append(dict(day=day, mid=m, hour=int(h), arm=arm, v1=(day, m) in v1, metrics=rec['metrics'], sha256=TR.sha(fn)))
            TR.dump(OUT2 / 'progress.json', dict(rows=len(rows), gates=len(gates), cpu_s=time.process_time() - t0, at=f'{day} {h}'))
        print(day, 'done', len(rows), round(time.process_time() - t0, 1), flush=True)
    TR.dump(OUT2 / 'summary.json', dict(rows=rows, gates=gates, cpu_s=time.process_time() - t0, status='COMPLETE'))


def check():
    sm = json.loads((OUT2 / 'summary.json').read_text())
    res = {}

    def ok(name, cond, detail=''):
        res[name] = dict(ok=bool(cond), detail=detail)
        print(('PASS ' if cond else 'FAIL ') + name, detail)
    ok('G1_v1_parity_gates', len(sm['gates']) == 38 and all(all(g['same'].values()) for g in sm['gates']), f"{len(sm['gates'])} gated ledgers")
    bad2 = n2 = bad3 = n3 = bad4 = bad5 = 0
    edges = TR.T.P.frozen_edges()
    for r in sm['rows']:
        fn = OUT2 / f"{r['mid']}_{r['arm']}.json.gz"
        assert TR.sha(fn) == r['sha256']
        rec = json.loads(gzip.decompress(fn.read_bytes()))
        io = {int(k): v for k, v in rec['iocs'].items()}
        grid = np.array(rec['grid'])
        for c in rec['commands']:
            if c[7] == 'POST':
                n2 += 1
                Ik = grid[int(round(c[0] / BE.STEP)), 1]
                need = 1 if Ik > EPS else 0 if Ik < -EPS else None
                bad2 += not (c[4] == 0 and c[1] == need and c[6] <= abs(Ik) + EPS and r['arm'] == 'PASSIVE_EXIT')
        fills = {(int(x[1]), round(float(x[3]), 9), float(x[4])) for x in rec['fills']}
        for x in rec['ioc_results']:
            if x['filled'] <= EPS:
                continue
            n3 += 1
            a = io[int(x['gid'])]
            inside = x['price'] <= a['limit'] + EPS if a['side'] == 0 else x['price'] >= a['limit'] - EPS
            fee = 200. / 1e4 * min(x['price'], 100 - x['price']) * x['filled']
            bad3 += not (inside and x['filled'] <= a['q'] + EPS and abs(fee - x['fee_c']) < 1e-9 and (int(x['gid']), round(float(x['filled']), 9), float(x['price'])) in fills)
        mt = rec['metrics']
        bad4 += not (abs(rec['value'] - (rec['cash'] + rec['position'] * rec['mark'])) < 1e-6 and abs(mt['completed_c'] + mt['residual_c'] - mt['total_c']) < 1e-6
                     and abs(mt['total_c'] - rec['value']) < 1e-6)
        ident = rec['ident']
        cache = make_cache(float(ident['start_abs_ms']), float(ident['end_rel']))
        n = len(rec['grid'])
        trig = TR.triggers(TR.on_arrays(cache, edges, n))
        want = {(int(k), s) for s in (0, 1) for k in np.where(trig[s])[0]}
        bad5 += {(x['k'], x['guard_side']) for x in rec['signal_log']} != want
    ok('G2_exit_orders_closing_side_rung0', n2 > 0 and bad2 == 0, f'{n2} posts, {bad2} violations (HOLD posts none)')
    ok('G3_ioc_results_rederived', n3 > 0 and bad3 == 0, f'{n3} filled IOCs, {bad3} mismatches')
    ok('G4_ledger_identities', bad4 == 0, f"{len(sm['rows'])} ledgers")
    ok('G5_triggers_recomputed', bad5 == 0, f"{len(sm['rows'])} ledgers")
    res['all_pass'] = all(v['ok'] for v in res.values() if isinstance(v, dict))
    TR.dump(H / 'v2_checks.json', res)
    print('ALL PASS' if res['all_pass'] else 'SOME CHECK FAILED')
    sys.exit(0 if res['all_pass'] else 1)


def report():
    sm = json.loads((OUT2 / 'summary.json').read_text())
    cen = json.loads((OUT2 / 'census.json').read_text())
    df = pd.DataFrame([dict(day=r['day'], mid=r['mid'], hour=r['hour'], arm=r['arm'], v1=r['v1'], **{k: v for k, v in r['metrics'].items() if not isinstance(v, dict)})
                       for r in sm['rows']])
    df.to_csv(H / 'v2_per_market.csv', index=False)
    out = dict(census=dict(status_counts=cen['status_counts'], by_day=cen['by_day'], complete=cen['complete'], selection_rule=cen['selection_rule'],
                           selected=len(cen['selected'])))
    for arm in ARMS2:
        x = df[df.arm == arm]
        dd = x.groupby('day').net_c.sum()
        absm = x.net_c.abs().sort_values(ascending=False)
        out[arm] = dict(markets=int(len(x)), net_usd=float(x.net_c.sum() / 100), net_per_market_usd=float(x.net_c.mean() / 100),
                        completed_usd=float(x.completed_c.sum() / 100), residual_usd=float(x.residual_c.sum() / 100), fees_usd=float(x.fees_c.sum() / 100),
                        day_net_usd={k: float(v / 100) for k, v in dd.items()}, days_positive=int((dd > 0).sum()), markets_positive=int((x.net_c > 0).sum()),
                        top10_abs_share=float(absm.iloc[:10].sum() / absm.sum()) if absm.sum() > 0 else None,
                        takes_filled=int(x.takes_filled.sum()), taker_shares=float(x.taker_shares.sum()), exit_shares=float(x.exit_shares.sum()),
                        young_exit_shares=float(x.young_exit_shares.sum()), net_usd_without_young_exit_value=float((x.net_c.sum() - x.young_exit_value_c.sum()) / 100),
                        mean_abs_inventory=float(x.mean_abs_inventory.mean()),
                        v1_markets_net_usd=float(x[x.v1].net_c.sum() / 100), new_markets_net_usd=float(x[~x.v1].net_c.sum() / 100),
                        new_markets=int((~x.v1).sum()), new_markets_per_market_usd=float(x[~x.v1].net_c.mean() / 100) if (~x.v1).any() else None,
                        by_hour_net_usd={int(k): float(v / 100) for k, v in x.groupby('hour').net_c.sum().items()})
    pe, ho = out['PASSIVE_EXIT'], out['HOLD']
    out['readings'] = dict(passive_exit_positive=pe['net_usd'] > 0, passive_exit_days_positive=pe['days_positive'], robust_3_of_4=pe['days_positive'] >= 3,
                           passive_exit_minus_hold_usd=pe['net_usd'] - ho['net_usd'], passive_exit_beats_hold=pe['net_usd'] > ho['net_usd'],
                           positive_without_young_exit_value=pe['net_usd_without_young_exit_value'] > 0,
                           new_markets_positive=bool(pe['new_markets_net_usd'] > 0))
    TR.dump(H / 'v2_readings.json', out)
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    {'census': census, 'run': run, 'check': check, 'report': report}[sys.argv[1]]()
