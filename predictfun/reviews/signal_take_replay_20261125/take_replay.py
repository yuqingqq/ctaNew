"""SIGNAL-TAKE-REPLAY v1 (contract.md). Host arms = the saved BASE / WIDE ledgers of inventory_exit_exception_20261125 (hash-bound; my
take-disabled branch must reproduce them exactly on the declared parity markets). Candidate arms add SIGNAL_TAKE: an IOC in the
direction of the pressure guard's own trigger (OFF->ON after 2 s quiet), 50 shares within the hard cap, effective 40 ms after the
decision tick. No frozen module is edited; every engine class is subclassed here.
Usage: run_light.sh str_<stage> -- python3 predictfun/reviews/signal_take_replay_20261125/take_replay.py prepare|selftest|run
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

H = Path(__file__).resolve().parent
R = H.parents[2]
sys.path.insert(0, str(H.parent / 'maker3_cap_skew_20261124'))
import experiment as CAP  # noqa: E402  (the frozen cap experiment: Branch, IOC engine via the inventory pilot)

T, X, O, LB, BE = CAP.T, CAP.X, CAP.O, CAP.LB, CAP.BE
P = T.P
EXIT_H = H.parent / 'inventory_exit_exception_20261125'
EXIT_OUT = X.DATA / 'inventory_exit_exception_20261125'
OUT = X.DATA / 'signal_take_replay_20261125'
TAKE_Q, TAKE_DELAY, QUIET_TICKS = 50., 40., 40
HOSTS = {'BASE': (200., 100., 50.), 'WIDE': (1000., 500., 250.)}
PARITY_MIDS = (3697902, 3789484, 3899224)
PARITY_FIELDS = ['fills', 'orders', 'requests', 'value', 'cash', 'position', 'commands', 'guard_log', 'grid', 'closed', 'mode_log']
CPU_CAP = 450.
EPS = 1e-9
clean, dump, sha = T.clean, T.dump, T.sha


class SignalIOCEngine(T.IOCEngine):
    """The pilot's IOC engine with the take's own effect delay (TAKE_DELAY instead of 10 ms); everything else inherited."""

    def submit_ioc(self, g, s, q, limit):
        gid = len(self.o)
        act = float(g + TAKE_DELAY)
        self.o.append(dict(gid=gid, side=s, rung=-1., price=float(limit), size=float(q), rem=0., ahead=0., act=act, remove=act,
                           submit=float(g), new=False, ci=0, refused=False, req=None, why='IOC'))
        self.iocs[gid] = dict(gid=gid, side=s, q=float(q), limit=float(limit), submit=float(g), act=act, ack=act + BE.FILL_NOTICE, done=False)
        self.pending.append(gid)
        return gid


def on_arrays(cache, edges, n):
    """The guard's own ON state per side and tick (PressureGuard.signal with enabled=True)."""
    out = {}
    for s in (0, 1):
        xs, st = cache['x'][s], cache['status'][s]
        out[s] = np.array([bool(k < len(xs) and int(st[k]) == 0 and P.on_value(float(xs[k]), edges[s])) for k in range(n)], bool)
    return out


def triggers(on, quiet=QUIET_TICKS):
    """OFF->ON after `quiet` ticks without ON on that side."""
    tr = {}
    for s, a in on.items():
        c = np.r_[0, np.cumsum(a.astype(int))]
        k = np.arange(len(a))
        prior = c[k] - c[np.maximum(0, k - quiet)]
        tr[s] = a & (prior == 0)
    return tr


class SignalBranch(CAP.Branch):
    def __init__(self, path, end, ident, bps, host, take):
        cap, S, R_ = HOSTS[host]
        super().__init__(path, end, ident, bps, (cap, S, R_, False))
        assert not self.E.o
        self.E = SignalIOCEngine(path, end, bps)
        self.signal_take, self.trig, self.sig_log = bool(take), None, []
        if self.signal_take:
            self.settings = dict(self.settings, signal_take=True, take_q=TAKE_Q, take_delay_ms=TAKE_DELAY, quiet_ticks=QUIET_TICKS)

    def signal_step(self, k, g):
        if not self.signal_take:
            return
        for gs in (1, 0):                                   # ask-side ON = BTC up -> buy YES; bid-side ON = BTC down -> sell YES
            if not self.trig[gs][k]:
                continue
            s = 0 if gs == 1 else 1
            I, B, A = self.reserve(g)
            room = self.cap - I - B if s == 0 else self.cap + I - A
            q = float(math.floor(min(TAKE_Q, room) + EPS))
            row = dict(t=float(g), k=int(k), guard_side=gs, side=s, known_I=I, B=B, A=A, room=room, q=q, status='candidate', gid=None)
            if q < 1:
                row['status'] = 'no_room'
            elif self.take_open:
                row['status'] = 'pending_ioc'
            elif g + TAKE_DELAY >= self.end:
                row['status'] = 'near_end'
            else:
                v = self.side_view(g, s)
                limit = v['ask'] if s == 0 else v['bid']
                row.update(limit=limit, book_received_ms=v['received'], known_frame=int(self.E.known))
                if not v['book_ok'] or g - v['received'] > 1000:
                    row['status'] = 'no_fresh_book'
                elif any(limit >= p if s == 0 else limit <= p for p in v['opp_prices']):
                    row['status'] = 'own_cross'
                else:
                    gid = self.E.submit_ioc(g, s, q, limit)
                    self.take_open.append(gid)
                    row.update(gid=gid, status='sent')
            self.sig_log.append(row)
        self.check_position(g)


def mid_ok(f, t):
    j = int(np.searchsorted(f[:, 0], t, side='right')) - 1
    if j < 0:
        return None
    b, a = f[j, 2], f[j, 12]
    return float((b + a) / 2) if (0 <= b < a <= 100 and t - f[j, 0] <= 1000) else None


def take_marks(B):
    out = []
    for x in B.E.exec_log:
        if x['filled'] <= EPS:
            continue
        sg = 1 if x['side'] == 0 else -1
        row = dict(gid=x['gid'], t=x['act'], side=x['side'], q=x['filled'], price=x['price'], fee_c=x['fee_c'])
        for h in (1000, 3000, 10000):
            m = mid_ok(B.E.f, x['act'] + h) if x['act'] + h <= B.end else None
            row[f'm{h // 1000}s_c'] = None if m is None else sg * (m - x['price']) - x['fee_c'] / x['filled']
        row['end_c'] = sg * (B.mark - x['price']) - x['fee_c'] / x['filled']
        out.append(row)
    return out


def run_one(path, cache, old, edges, bps, host, take):
    B = SignalBranch(path, old['end'], old['ident'], bps, host, take)
    pol = O.OpeningGate(B, cache, edges)
    n = LB.n_ticks(B)
    if take:
        B.trig = triggers(on_arrays(cache, edges, n))
    grid = []
    for k in range(n):
        LB.step(B, k, pol)
        g = k * BE.STEP
        B.signal_step(k, g)
        cash, pos, _ = B.E.value(B.mark)
        I, RB, RA = B.reserve(g)
        live = [sum(o['side'] == s and o['gid'] not in B.E.iocs and not o['refused'] and o['act'] <= g < o['remove'] and o['rem'] > EPS for o in B.E.o)
                for s in (0, 1)]
        grid.append([g, I, RB, RA, pos, cash, *live])
    lg = LB.finish(B)
    rec = T.M.ledger_record(B, lg)
    rec.update(commands=clean(B.cmd_log), guard_log=clean(pol.log), scope_log=pol.scope_log, iocs=clean(B.E.iocs), ioc_results=B.E.exec_log,
               signal_log=B.sig_log, fees_c=sum(x[2] for x in B.E.fees), grid=grid, host=host, take=bool(take),
               take_marks=take_marks(B) if take else [], n_triggers={str(s): int(B.trig[s].sum()) for s in (0, 1)} if take else {})
    rec['net_value_c'] = rec['value'] - rec['fees_c']
    met = X.fifo(rec['fills'], rec['mark'])
    gg = np.array(grid)
    st = collections.Counter(r['status'] for r in B.sig_log)
    met.update(net_c=rec['net_value_c'], fees_c=rec['fees_c'], maker_shares=sum(f[3] for f in rec['fills'] if f[5] >= 0),
               taker_shares=sum(f[3] for f in rec['fills'] if f[5] < 0), triggers=len(B.sig_log), trigger_status=dict(st),
               takes_sent=len(rec['iocs']), takes_filled=sum(x['filled'] > EPS for x in B.E.exec_log),
               ioc_status=dict(collections.Counter(x['reason'] for x in B.E.exec_log)), peak_inventory=float(max(abs(gg[:, 4]))),
               mean_abs_inventory=float(np.mean(abs(gg[:, 4]))), both_grid_fraction=float(np.mean((gg[:, 6] > 0) & (gg[:, 7] > 0))),
               passive_generations=len(B.E.o) - len(B.E.iocs), passive_pulls=sum(c[7] == 'PULL' for c in B.cmd_log))
    rec['metrics'] = met
    return clean(rec)


def cases():
    m = json.loads((EXIT_H / 'manifest.json').read_text())
    assert len(m['cases']) == 19
    return [(d, int(mid), part) for d, mid, part in m['cases']], float(m['fees_bps'])


def host_ledgers():
    s = json.loads((EXIT_H / 'summary.json').read_text())
    assert s['status'] == 'COMPLETE'
    return {(int(r['mid']), r['arm']): r['sha256'] for r in s['rows'] if r['arm'] in HOSTS}


FILES = ['contract.md', 'take_replay.py', 'check.py', 'report.py']


def prepare():
    assert not (H / 'manifest.json').exists()
    cs, bps = cases()
    hl = host_ledgers()
    assert len(hl) == 38
    src = {str((H / f).relative_to(R)): sha(H / f) for f in FILES}
    for p in [CAP.H / 'experiment.py', T.H / 'pilot.py', O.H / 'opening.py', EXIT_H / 'manifest.json', EXIT_H / 'summary.json']:
        src[str(p.relative_to(R))] = sha(p)
    for (mid, arm), h in hl.items():
        p = EXIT_OUT / f'{mid}_{arm}.json.gz'
        assert sha(p) == h, p
        src[str(p.relative_to(R))] = h
    dump(H / 'manifest.json', dict(asof='2026-11-25', cases=cs, fees_bps=bps, hosts=HOSTS, take_q=TAKE_Q, take_delay_ms=TAKE_DELAY, quiet_ticks=QUIET_TICKS,
                                   parity_mids=PARITY_MIDS, sources=src, new_candidates=2, validation=False))


def bound():
    m = json.loads((H / 'manifest.json').read_text())
    for p, h in m['sources'].items():
        assert sha(R / p) == h, p
    for f in FILES + ['manifest.json']:
        assert subprocess.check_output(['git', 'show', 'HEAD:' + str((H / f).relative_to(R))], cwd=R) == (H / f).read_bytes(), f
    return m


def selftest():
    on = {0: np.array([0, 1, 1, 0, 0, 0, 1, 0, 1], bool), 1: np.zeros(9, bool)}
    tr = triggers(on, quiet=3)
    assert tr[0].tolist() == [False, True, False, False, False, False, True, False, False], tr[0].tolist()
    assert not tr[1].any()
    on2 = {0: np.array([1, 0, 0, 0, 1], bool), 1: np.array([0] * 5, bool)}
    assert triggers(on2, quiet=3)[0].tolist() == [True, False, False, False, True]
    print('selftest ok')


def run():
    resource.setrlimit(resource.RLIMIT_CPU, (int(CPU_CAP), int(CPU_CAP) + 1))
    m = bound()
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'summary.json').exists()
    edges = T.P.frozen_edges()
    start = time.process_time()
    rows, parity, runs = [], [], 0
    total_runs = 2 * len(m['cases']) + 2 * len(m['parity_mids'])
    for ix, (day, mid, part) in enumerate(m['cases']):
        old = X.records(day, part)[mid]['CURRENT_ASYNC_FAST10']
        path, cache = X.get_path(day, mid, part, old)
        if mid in m['parity_mids']:
            for host in HOSTS:
                rec = json.loads(json.dumps(run_one(path, cache, old, edges, m['fees_bps'], host, False), allow_nan=False))
                runs += 1
                prev = json.loads(gzip.decompress((EXIT_OUT / f'{mid}_{host}.json.gz').read_bytes()))
                same = {f: rec[f] == prev[f] for f in PARITY_FIELDS}
                parity.append(dict(mid=mid, host=host, same=same))
                assert all(same.values()), (mid, host, same)
        for host in HOSTS:
            arm = f'{host}_TAKE'
            rec = run_one(path, cache, old, edges, m['fees_bps'], host, True)
            runs += 1
            rec.update(day=day, mid=mid, arm=arm, manifest_sha256=sha(H / 'manifest.json'))
            fn = OUT / f'{mid}_{arm}.json.gz'
            assert not fn.exists()
            fn.write_bytes(gzip.compress(json.dumps(rec, allow_nan=False).encode(), mtime=0))
            rows.append(dict(day=day, mid=mid, arm=arm, metrics=rec['metrics'], sha256=sha(fn)))
        elapsed = time.process_time() - start
        if ix == 0:
            proj = elapsed / runs * total_runs * 1.2
            dump(H / 'projection.json', dict(first_mid=mid, runs=runs, cpu_s=elapsed, projection_cpu_s=proj, limit=CPU_CAP, proceed=proj <= CPU_CAP))
            assert proj <= CPU_CAP, 'projection stop'
        dump(OUT / 'progress.json', dict(markets=ix + 1, rows=rows, parity=parity, cpu_s=elapsed))
        print(day, mid, 'done', round(elapsed, 2), flush=True)
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 1536 * 1024
    dump(OUT / 'summary.json', dict(rows=rows, parity=parity, cpu_s=time.process_time() - start, status='COMPLETE'))


if __name__ == '__main__':
    {'prepare': prepare, 'selftest': selftest, 'run': run}[sys.argv[1]]()
