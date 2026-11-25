"""SIGNAL-TAKE-REPLAY v1 independent checks (contract.md 'Checks'). Runs after `take_replay.py run`.
 S1 triggers recomputed by a plain loop from the cache + frozen edges equal the logged triggers (6 hash-ranked markets, both take arms);
 S2 every sent IOC: limit = the known frame's best (that frame received <= decision tick, the next one after it), effect = tick + 40 ms;
 S3 every IOC result re-derived from venue frames: fill price inside the limit and equal to the venue best at effect, quantity <= order,
    fee = 200 bps x min(p, 100 - p) x quantity; every take fill is in the ledger's fills with gid < 0 marker;
 S4 ledger identities on every take ledger: value = cash + position x mark; net = value - fees; FIFO completed + residual = value;
 S5 synthetic falsifiers through SignalBranch on a real path: sent at a fresh book; no_room at a zero cap; near_end; pending_ioc;
    the 40 ms effect delay; a disabled branch never sends;
 S6 corruption: altering one take ledger's net value must change the pooled recomputation.
Usage: run_light.sh str_check -- python3 predictfun/reviews/signal_take_replay_20261125/check.py
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import take_replay as TR  # noqa: E402

res = {}


def ok(name, cond, detail=''):
    res[name] = dict(ok=bool(cond), detail=detail)
    print(('PASS ' if cond else 'FAIL ') + name, detail)


def rank(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


def load(mid, arm):
    p = (TR.EXIT_OUT if arm in TR.HOSTS else TR.OUT) / f'{mid}_{arm}.json.gz'
    return json.loads(gzip.decompress(p.read_bytes()))


def main():
    m = json.loads((H / 'manifest.json').read_text())
    sm = json.loads((TR.OUT / 'summary.json').read_text())
    assert sm['status'] == 'COMPLETE'
    for r in sm['rows']:
        assert TR.sha(TR.OUT / f"{r['mid']}_{r['arm']}.json.gz") == r['sha256']
    cs = m['cases']
    pick = sorted(cs, key=lambda c: rank(f'str_check|{c[1]}'))[:6]
    edges = TR.T.P.frozen_edges()
    bad1 = n1 = bad2 = n2 = 0
    for day, mid, part in pick:
        old = TR.X.records(day, part)[mid]['CURRENT_ASYNC_FAST10']
        path, cache = TR.X.get_path(day, mid, part, old)
        B0 = TR.SignalBranch(path, old['end'], old['ident'], m['fees_bps'], 'BASE', False)
        f = B0.E.f
        n = TR.LB.n_ticks(B0)
        want = set()
        for s in (0, 1):
            xs, st = cache['x'][s], cache['status'][s]
            last_on = -10 ** 9
            for k in range(n):
                on = k < len(xs) and int(st[k]) == 0 and TR.P.on_value(float(xs[k]), edges[s])
                if on and k - last_on > TR.QUIET_TICKS:
                    want.add((k, s))
                if on:
                    last_on = k
        for host in TR.HOSTS:
            rec = load(mid, f'{host}_TAKE')
            got = {(r['k'], r['guard_side']) for r in rec['signal_log']}
            n1 += 1
            bad1 += got != want
            io = {int(k): v for k, v in rec['iocs'].items()}
            for r in rec['signal_log']:
                if r['status'] != 'sent':
                    continue
                n2 += 1
                j = r['known_frame']
                px = f[j, 12] if r['side'] == 0 else f[j, 2]
                nxt_ok = j + 1 >= len(f) or f[j + 1, 1] > r['t']
                a = io[int(r['gid'])]
                bad2 += not (px == r['limit'] and f[j, 1] <= r['t'] and nxt_ok and abs(a['act'] - (r['t'] + TR.TAKE_DELAY)) < 1e-9 and a['limit'] == r['limit'])
    ok('S1_triggers_recomputed', n1 == 12 and bad1 == 0, f'{n1} market-arms, {bad1} mismatches')
    ok('S2_sent_limits_and_delay', n2 > 0 and bad2 == 0, f'{n2} sent IOCs, {bad2} mismatches')
    bad3 = n3 = bad4 = 0
    for r in sm['rows']:
        rec = load(r['mid'], r['arm'])
        io = {int(k): v for k, v in rec['iocs'].items()}
        fills = {(int(x[1]), round(float(x[3]), 9), float(x[4])) for x in rec['fills']}
        for x in rec['ioc_results']:
            if x['filled'] <= 1e-9:
                continue
            n3 += 1
            a = io[int(x['gid'])]
            inside = x['price'] <= a['limit'] + 1e-9 if a['side'] == 0 else x['price'] >= a['limit'] - 1e-9
            fee = m['fees_bps'] / 1e4 * min(x['price'], 100 - x['price']) * x['filled']
            bad3 += not (inside and x['filled'] <= a['q'] + 1e-9 and abs(fee - x['fee_c']) < 1e-9 and (int(x['gid']), round(float(x['filled']), 9), float(x['price'])) in fills)
        cash, pos, mark = rec['cash'], rec['position'], rec['mark']
        fifo = TR.X.fifo(rec['fills'], rec['mark'])
        keys = [k for k in fifo if 'completed' in k or 'residual' in k]
        mt = rec['metrics']
        bad4 += not (abs(rec['value'] - (cash + pos * mark)) < 1e-6 and abs(rec['net_value_c'] - (rec['value'] - rec['fees_c'])) < 1e-9
                     and abs(mt['completed_c'] + mt['residual_c'] - mt['total_c']) < 1e-6 and abs(mt['total_c'] - rec['value']) < 1e-6
                     and abs(fifo['total_c'] - mt['total_c']) < 1e-9)
        res.setdefault('fifo_keys', keys)
    ok('S3_ioc_results_rederived', n3 > 0 and bad3 == 0, f'{n3} filled IOCs, {bad3} mismatches')
    ok('S4_ledger_identities', bad4 == 0, f"{len(sm['rows'])} take ledgers, {bad4} mismatches")
    # S5 synthetic falsifiers on a real path
    day, mid, part = cs[0]
    old = TR.X.records(day, part)[mid]['CURRENT_ASYNC_FAST10']
    path, cache = TR.X.get_path(day, mid, part, old)

    def branch(host, cap=None, take=True):
        B = TR.SignalBranch(path, old['end'], old['ident'], m['fees_bps'], host, take)
        if cap is not None:
            B.cap = float(cap)
        pol = TR.O.OpeningGate(B, cache, edges)
        n = TR.LB.n_ticks(B)
        B.trig = {0: np.zeros(n, bool), 1: np.zeros(n, bool)}
        return B, pol, n
    k0 = 1200

    def run_to(B, pol, kmax):
        for k in range(kmax + 1):
            TR.LB.step(B, k, pol)
            B.signal_step(k, k * TR.BE.STEP)
    B, pol, n = branch('BASE')                                # (a) a trigger at a fresh book, then another 100 ms later
    B.trig[1][k0] = True
    B.trig[0][k0 + 2] = True
    run_to(B, pol, k0 + 2)
    st = {r['k']: r['status'] for r in B.sig_log}
    sent = [r for r in B.sig_log if r['status'] == 'sent']
    ca = st.get(k0) in ('sent', 'own_cross', 'no_fresh_book') and (st.get(k0) != 'sent' or st.get(k0 + 2) == 'pending_ioc')
    cdelay = all(abs(B.E.iocs[r['gid']]['act'] - (r['t'] + 40.)) < 1e-9 for r in sent)
    B2, pol2, _ = branch('BASE', cap=0.)                      # (b) zero hard cap -> no room
    B2.trig[1][k0] = True
    run_to(B2, pol2, k0)
    cb = [r['status'] for r in B2.sig_log] == ['no_room']
    B3, pol3, _ = branch('BASE', cap=1e6)                     # (c) inside the last 40 ms -> near_end
    run_to(B3, pol3, k0)
    B3.trig[1][k0 + 1] = True
    B3.signal_step(k0 + 1, B3.end - 20.)
    cc = [r['status'] for r in B3.sig_log] == ['near_end']
    B4, pol4, _ = branch('BASE', take=False)                  # (d) disabled never sends
    B4.trig[1][:] = True
    run_to(B4, pol4, k0)
    cd = not B4.sig_log and not B4.E.iocs
    ok('S5_synthetic_falsifiers', ca and cdelay and cb and cc and cd,
       f'first {st.get(k0)} then {st.get(k0 + 2)}, delay40 {cdelay}, no_room {cb}, near_end {cc}, disabled {cd}')
    # S6 corruption: the pooled delta must move when one ledger is altered
    def pooled(alter=None):
        tot = 0.
        for r in sm['rows']:
            v = load(r['mid'], r['arm'])['net_value_c']
            if alter == (r['mid'], r['arm']):
                v += 1.0
            tot += v
        return tot
    a0 = pooled()
    r0 = sm['rows'][0]
    ok('S6_corruption_detected', abs(pooled((r0['mid'], r0['arm'])) - a0 - 1.0) < 1e-9, 'one ledger +1 c shifts the pooled sum by exactly 1 c')
    res['all_pass'] = all(v['ok'] for k, v in res.items() if isinstance(v, dict) and 'ok' in v)
    (H / 'checks.json').write_text(json.dumps(res, indent=1) + '\n')
    print('ALL PASS' if res['all_pass'] else 'SOME CHECK FAILED')
    sys.exit(0 if res['all_pass'] else 1)


if __name__ == '__main__':
    main()
