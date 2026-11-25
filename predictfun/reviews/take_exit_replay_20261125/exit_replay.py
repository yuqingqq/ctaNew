"""TAKE-EXIT-REPLAY v1 (contract.md): signal takes + passive exits only, on the 19 complete consumed markets.
Reuses the frozen SIGNAL-TAKE-REPLAY branch / triggers (signal_take_replay_20261125/take_replay.py, commit 7e37888c8) and the frozen
pressure guard; the only new pieces are the exit-only inner policy below and the standalone configuration (hard cap 50, no modes).
Usage: run_light.sh txr_<stage> -- python3 predictfun/reviews/take_exit_replay_20261125/exit_replay.py prepare|selftest|run|check|report
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
sys.path.insert(0, str(H.parent / 'signal_take_replay_20261125'))
import take_replay as TR  # noqa: E402

T, X, O, LB, BE, P = TR.T, TR.X, TR.O, TR.LB, TR.BE, TR.P
OUT = X.DATA / 'take_exit_replay_20261125'
CAP_POS = 50.
TR.HOSTS['EXIT50'] = (CAP_POS, 1e6, 5e5)          # standalone: one +-50 position; the inventory modes are never reached
ARMS = {'PASSIVE_EXIT': dict(exit=True, guard=True), 'PASSIVE_EXIT_NOGUARD': dict(exit=True, guard=False), 'HOLD': dict(exit=False, guard=False)}
YOUNG_MS = 100.
CPU_CAP = 400.
EPS = 1e-9
FILES = ['contract.md', 'exit_replay.py']


class ExitInner(LB.Policy):
    """Exit-only: one rung-0 post-only order on the closing side sized to the uncovered position; nothing on the adding side."""

    def __init__(self, B, exit_on):
        super().__init__('take_exit_inner', B.end)
        self.exit_on, self.q0 = bool(exit_on), B.q0

    @staticmethod
    def closing(v):
        I = float(v['position'])
        return (v['side'] == 1 and I > EPS) or (v['side'] == 0 and I < -EPS)

    def live(self, v, o):
        return ('KEEP', 'exit_keep') if (self.exit_on and self.closing(v)) else ('PULL', 'exit_not_needed')

    def slot(self, v, r, price):
        if not self.exit_on:
            return 'WAIT', None, 'hold'
        if r != 0:
            return 'WAIT', None, 'exit_rung0_only'
        if not self.closing(v):
            return 'WAIT', None, 'no_exit_side'
        res = v['A'] if v['side'] == 1 else v['B']
        q = float(math.floor(min(self.q0, abs(float(v['position'])) - res) + EPS))
        if q < LB.QMIN:
            return 'WAIT', q, 'exit_covered'
        return 'POST', q, 'exit_post'


class ExitPolicy(P.PressureGuard):
    """The frozen pressure guard around the exit-only inner policy (enabled=False: no guard)."""

    def __init__(self, B, cache, edges, exit_on, guard):
        super().__init__(B, cache, edges, enabled=guard)
        self.inner = ExitInner(B, exit_on)


def run_one(path, cache, old, edges, bps, arm, take=True):
    a = ARMS[arm]
    B = TR.SignalBranch(path, old['end'], old['ident'], bps, 'EXIT50', take)
    pol = ExitPolicy(B, cache, edges, a['exit'], a['guard'])
    n = LB.n_ticks(B)
    if take:
        B.trig = TR.triggers(TR.on_arrays(cache, edges, n))
    grid = []
    for k in range(n):
        LB.step(B, k, pol)
        g = k * BE.STEP
        B.signal_step(k, g)
        cash, pos, _ = B.E.value(B.mark)
        I, RB, RA = B.reserve(g)
        grid.append([g, I, RB, RA, pos, cash])
    lg = LB.finish(B)
    rec = T.M.ledger_record(B, lg)
    io = set(B.E.iocs)
    passive = [f for f in B.E.fills if f[1] not in io]
    ages = [float(f[0] - B.E.o[f[1]]['act']) for f in passive]
    exit_val = [((B.mark - f[4]) if f[2] == 0 else (f[4] - B.mark)) * f[3] for f in passive]   # exit vs holding to the END mark, cents
    rec.update(commands=TR.clean(B.cmd_log), guard_log=TR.clean(pol.log), iocs=TR.clean(B.E.iocs), ioc_results=B.E.exec_log, signal_log=B.sig_log,
               fees_c=sum(x[2] for x in B.E.fees), grid=grid, arm=arm, take=bool(take), take_marks=TR.take_marks(B) if take else [],
               passive_ages_ms=ages, passive_exit_value_c=exit_val)
    rec['net_value_c'] = rec['value'] - rec['fees_c']
    met = X.fifo(rec['fills'], rec['mark'])
    gg = np.array(grid)
    young = [i for i, x in enumerate(ages) if x < YOUNG_MS]
    met.update(net_c=rec['net_value_c'], fees_c=rec['fees_c'], taker_shares=sum(f[3] for f in B.E.fills if f[1] in io),
               exit_shares=sum(f[3] for f in passive), exit_fills=len(passive), young_exit_shares=sum(passive[i][3] for i in young),
               young_exit_value_c=sum(exit_val[i] for i in young), exit_value_vs_hold_c=sum(exit_val),
               triggers=len(B.sig_log), trigger_status=dict(collections.Counter(r['status'] for r in B.sig_log)), takes_sent=len(B.E.iocs),
               takes_filled=sum(x['filled'] > EPS for x in B.E.exec_log), ioc_status=dict(collections.Counter(x['reason'] for x in B.E.exec_log)),
               exit_orders=len(B.E.o) - len(io), peak_inventory=float(max(abs(gg[:, 4]))), mean_abs_inventory=float(np.mean(abs(gg[:, 4]))),
               time_in_position=float(np.mean(abs(gg[:, 4]) > EPS)))
    rec['metrics'] = met
    return TR.clean(rec)


def prepare():
    assert not (H / 'manifest.json').exists()
    cs, bps = TR.cases()
    src = {str((H / f).relative_to(R)): TR.sha(H / f) for f in FILES}
    for p in [TR.H / 'take_replay.py', TR.H / 'manifest.json', TR.CAP.H / 'experiment.py', TR.T.H / 'pilot.py', TR.O.H / 'opening.py']:
        src[str(p.relative_to(R))] = TR.sha(p)
    TR.dump(H / 'manifest.json', dict(asof='2026-11-25', cases=cs, fees_bps=bps, arms=ARMS, cap=CAP_POS, take_q=TR.TAKE_Q, take_delay_ms=TR.TAKE_DELAY,
                                      quiet_ticks=TR.QUIET_TICKS, young_ms=YOUNG_MS, sources=src, new_candidates=2, validation=False))


def bound():
    m = json.loads((H / 'manifest.json').read_text())
    for p, h in m['sources'].items():
        assert TR.sha(R / p) == h, p
    for f in FILES + ['manifest.json']:
        assert subprocess.check_output(['git', 'show', 'HEAD:' + str((H / f).relative_to(R))], cwd=R) == (H / f).read_bytes(), f
    return m


class _B:
    q0 = 100.
    end = 298000.


def selftest():
    e = ExitInner(_B(), True)
    long_ask = dict(side=1, position=50., A=0., B=0.)
    long_bid = dict(side=0, position=50., A=0., B=0.)
    short_bid = dict(side=0, position=-30., A=0., B=0.)
    flat = dict(side=1, position=0., A=0., B=0.)
    covered = dict(side=1, position=50., A=50., B=0.)
    assert e.slot(long_ask, 0, 60.) == ('POST', 50., 'exit_post')
    assert e.slot(long_ask, 1, 61.)[0] == 'WAIT' and e.slot(long_bid, 0, 59.)[0] == 'WAIT'
    assert e.slot(short_bid, 0, 40.) == ('POST', 30., 'exit_post') and e.slot(flat, 0, 50.)[0] == 'WAIT' and e.slot(covered, 0, 60.)[0] == 'WAIT'
    assert e.live(long_ask, {}) == ('KEEP', 'exit_keep') and e.live(flat, {}) == ('PULL', 'exit_not_needed')
    h = ExitInner(_B(), False)
    assert h.slot(long_ask, 0, 60.)[0] == 'WAIT' and h.live(long_ask, {})[0] == 'PULL'
    print('selftest ok')


def run():
    resource.setrlimit(resource.RLIMIT_CPU, (int(CPU_CAP), int(CPU_CAP) + 1))
    m = bound()
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'summary.json').exists()
    edges = TR.T.P.frozen_edges()
    start = time.process_time()
    rows = []
    for ix, (day, mid, part) in enumerate(m['cases']):
        old = X.records(day, part)[mid]['CURRENT_ASYNC_FAST10']
        path, cache = X.get_path(day, mid, part, old)
        for arm in ARMS:
            rec = run_one(path, cache, old, edges, m['fees_bps'], arm)
            rec.update(day=day, mid=mid, manifest_sha256=TR.sha(H / 'manifest.json'))
            fn = OUT / f'{mid}_{arm}.json.gz'
            assert not fn.exists()
            fn.write_bytes(gzip.compress(json.dumps(rec, allow_nan=False).encode(), mtime=0))
            rows.append(dict(day=day, mid=mid, arm=arm, metrics=rec['metrics'], sha256=TR.sha(fn)))
        el = time.process_time() - start
        if ix == 0:
            proj = el * len(m['cases']) * 1.2
            TR.dump(H / 'projection.json', dict(first_mid=mid, cpu_s=el, projection_cpu_s=proj, limit=CPU_CAP, proceed=proj <= CPU_CAP))
            assert proj <= CPU_CAP, 'projection stop'
        TR.dump(OUT / 'progress.json', dict(markets=ix + 1, rows=rows, cpu_s=el))
        print(day, mid, 'done', round(el, 2), flush=True)
    TR.dump(OUT / 'summary.json', dict(rows=rows, cpu_s=time.process_time() - start, status='COMPLETE'))


def load(mid, arm):
    return json.loads(gzip.decompress((OUT / f'{mid}_{arm}.json.gz').read_bytes()))


def check():
    m = json.loads((H / 'manifest.json').read_text())
    sm = json.loads((OUT / 'summary.json').read_text())
    res = {}

    def ok(name, cond, detail=''):
        res[name] = dict(ok=bool(cond), detail=detail)
        print(('PASS ' if cond else 'FAIL ') + name, detail)
    edges = TR.T.P.frozen_edges()
    day, mid, part = m['cases'][0]
    old = X.records(day, part)[mid]['CURRENT_ASYNC_FAST10']
    path, cache = X.get_path(day, mid, part, old)
    z = [run_one(path, cache, old, edges, m['fees_bps'], arm, take=False) for arm in ARMS]
    ok('E1_no_trigger_no_trading', all(len(r['fills']) == 0 and len(r['orders']) == 0 and not r['iocs'] for r in z), f'{len(z)} arms with takes disabled')
    bad2 = n2 = bad3 = n3 = bad4 = bad5 = 0
    for r in sm['rows']:
        rec = load(r['mid'], r['arm'])
        assert TR.sha(OUT / f"{r['mid']}_{r['arm']}.json.gz") == r['sha256']
        io = {int(k): v for k, v in rec['iocs'].items()}
        grid = np.array(rec['grid'])
        for c in rec['commands']:
            if c[7] == 'POST':
                n2 += 1
                g, s, rung, q = c[0], c[1], c[4], c[6]
                Ik = grid[int(round(g / BE.STEP)), 1]
                need_side = 1 if Ik > EPS else 0 if Ik < -EPS else None
                bad2 += not (rung == 0 and s == need_side and q <= abs(Ik) + EPS and ARMS[r['arm']]['exit'])
        fills = {(int(x[1]), round(float(x[3]), 9), float(x[4])) for x in rec['fills']}
        for x in rec['ioc_results']:
            if x['filled'] <= EPS:
                continue
            n3 += 1
            a = io[int(x['gid'])]
            inside = x['price'] <= a['limit'] + EPS if a['side'] == 0 else x['price'] >= a['limit'] - EPS
            fee = m['fees_bps'] / 1e4 * min(x['price'], 100 - x['price']) * x['filled']
            bad3 += not (inside and x['filled'] <= a['q'] + EPS and abs(fee - x['fee_c']) < 1e-9 and (int(x['gid']), round(float(x['filled']), 9), float(x['price'])) in fills)
        mt = rec['metrics']
        bad4 += not (abs(rec['value'] - (rec['cash'] + rec['position'] * rec['mark'])) < 1e-6 and abs(mt['completed_c'] + mt['residual_c'] - mt['total_c']) < 1e-6
                     and abs(mt['total_c'] - rec['value']) < 1e-6 and abs(rec['net_value_c'] - (rec['value'] - rec['fees_c'])) < 1e-9)
        st = TR.EXIT_OUT  # trigger set vs the SIGNAL-TAKE ledgers of the same market
        ref = json.loads(gzip.decompress((TR.OUT / f"{r['mid']}_WIDE_TAKE.json.gz").read_bytes()))
        bad5 += {(x['k'], x['guard_side']) for x in rec['signal_log']} != {(x['k'], x['guard_side']) for x in ref['signal_log']}
    ok('E2_exit_orders_closing_side_rung0', n2 > 0 and bad2 == 0, f'{n2} posts, {bad2} violations (HOLD must post none)')
    ok('E3_ioc_results_rederived', n3 > 0 and bad3 == 0, f'{n3} filled IOCs, {bad3} mismatches')
    ok('E4_ledger_identities', bad4 == 0, f"{len(sm['rows'])} ledgers")
    ok('E5_trigger_set_equals_signal_take', bad5 == 0, f"{len(sm['rows'])} ledgers vs WIDE_TAKE")
    res['all_pass'] = all(v['ok'] for v in res.values() if isinstance(v, dict))
    TR.dump(H / 'checks.json', res)
    print('ALL PASS' if res['all_pass'] else 'SOME CHECK FAILED')
    sys.exit(0 if res['all_pass'] else 1)


def report():
    m = json.loads((H / 'manifest.json').read_text())
    rows = []
    for d, mid, _ in m['cases']:
        for arm in ARMS:
            mt = load(mid, arm)['metrics']
            rows.append(dict(day=d, mid=mid, arm=arm, **{k: v for k, v in mt.items() if not isinstance(v, dict)}))
    df = pd.DataFrame(rows)
    df.to_csv(H / 'per_market.csv', index=False)
    out = {}
    for arm in ARMS:
        x = df[df.arm == arm]
        dd = x.groupby('day').net_c.sum()
        out[arm] = dict(net_usd=float(x.net_c.sum() / 100), completed_usd=float(x.completed_c.sum() / 100), residual_usd=float(x.residual_c.sum() / 100),
                        fees_usd=float(x.fees_c.sum() / 100), day_net_usd={k: float(v / 100) for k, v in dd.items()}, days_positive=int((dd > 0).sum()),
                        markets_positive=int((x.net_c > 0).sum()), takes_sent=int(x.takes_sent.sum()), takes_filled=int(x.takes_filled.sum()),
                        taker_shares=float(x.taker_shares.sum()), exit_shares=float(x.exit_shares.sum()), young_exit_shares=float(x.young_exit_shares.sum()),
                        young_exit_value_usd=float(x.young_exit_value_c.sum() / 100), exit_value_vs_hold_usd=float(x.exit_value_vs_hold_c.sum() / 100),
                        net_usd_without_young_exit_value=float((x.net_c.sum() - x.young_exit_value_c.sum()) / 100),
                        mean_abs_inventory=float(x.mean_abs_inventory.mean()), time_in_position=float(x.time_in_position.mean()))
    pe, ho, ng = out['PASSIVE_EXIT'], out['HOLD'], out['PASSIVE_EXIT_NOGUARD']
    out['readings'] = dict(passive_exit_positive=pe['net_usd'] > 0, passive_exit_3_of_4_days=pe['days_positive'] >= 3,
                           passive_exit_minus_hold_usd=pe['net_usd'] - ho['net_usd'], passive_exit_beats_hold=pe['net_usd'] > ho['net_usd'],
                           passive_exit_minus_noguard_usd=pe['net_usd'] - ng['net_usd'],
                           sign_without_young_exit_value=bool(pe['net_usd_without_young_exit_value'] > 0))
    TR.dump(H / 'readings.json', out)
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    {'prepare': prepare, 'selftest': selftest, 'run': run, 'check': check, 'report': report}[sys.argv[1]]()
