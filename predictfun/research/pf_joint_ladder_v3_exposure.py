"""JOINT-LADDER v3 POST HOC control (2026-11-14, session ce68ebe0): TRUE EXPOSURE under the hard caps, measured.

Why: v3's printed header calls its cap statistic "max true exposure", but the lane records the largest |true position| only (stat maxpos, and
its 'cap' violation compares the position with the cap). The hard cap's claim is about exposure - the position plus every share of ours that
can still fill - which is bounded by the client's reserved exposure by construction but was never measured. This control measures it from each
run's own record, on the v3 replay's 9 days and its client-state runs (same code path: pf_joint_ladder_v3.sim_joint, unchanged).

Definition, side s (0 = bid, 1 = ask), at time t:
  E_s(t) = sum over our ACCEPTED side-s orders submitted by t of (size if t < its end else its total fills)
           - (the other side's fills before t)
  an order counts from its SUBMISSION (ts = ta - L_PLACE); its end is when its cancel lands or it is fully filled (the lane's
  min(tr, tfill)); a refused order can never fill and counts 0. Own-side fills leave E_s unchanged (position up, remaining down); E_s rises
  only at a submission, so its maximum over the window is read at each accepted submission (all orders submitted at that instant included).
  Ties are read conservatively: an order whose end equals t still counts as open; an other-side fill at t is not subtracted.

Readings (predicates evaluated in the code):
  HARD-EXPOSURE-OK  max E <= K x size on every run of hard2 / hard5 / hard10 (all arms, clocks M 0 / 50 / 250, 9 days);
                    else HARD-EXPOSURE-EXCEEDED with the count of runs.
  Positive control  soft2 (a delayed trigger on the known position, known to overshoot) must exceed 2 x size on some run of EVERY day, else the
                    instrument is VOID (no reading).
Falsifier (selftest, synthetic run records): two 30-sh orders open together read 60 (> a 30-sh cap: flagged); a fully filled order replaced
by a new one reads position + the new order; a refused order reads 0; an other-side fill lowers it; a cancelled order keeps only its fills;
the planted position-only statistic (what v3's header reported) must fail these cases.
Usage (run_light, PF_SLOT=claude): pf_joint_ladder_v3_exposure.py selftest | day <day> | report; driver run_joint_ladder_v3_exposure.sh
"""
import bisect
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_kp_s4 as K                                      # noqa: E402
import pf_pull_depth as PD                                # noqa: E402
import pf_joint_ladder_v3 as J3                           # noqa: E402

OUTD = J3.OUTD
SIZE = J3.SIZE
RUN_CAPS = (('hard', 2), ('hard', 5), ('hard', 10), ('soft', 2))
TOL = 1e-6
MUT_POS_ONLY = 0                                          # planted error for the selftest's falsifier
TOL_T = 0.01                                              # ms: matching a fill to its order by activation time


def max_exposure(r, size=SIZE):
    """Max true exposure per side over the window, from a sim_joint result (orders, fills, fl): {0: e_bid, 1: e_ask, 'n_sub': submissions}.
    Event sweep: +size at each accepted submission, -(size - its fills) just after its end, -q just after each other-side fill; the maximum is
    read after each submission (submissions sort before releases at the same time: the conservative tie)."""
    O = {}
    for (s, d), lst in r['orders'].items():
        O[(s, d)] = [[ta - K.L_PLACE, end, ta, 0.0, ref] for (ta, end, a0, ref) in lst]
    seq = {}
    for (s, d, t, pt, qq) in r['fl']:
        seq.setdefault((s, d), []).append((t, qq))
    for (s, d), lst in r['fills'].items():
        tq = seq.get((s, d), [])
        if len(tq) != len(lst):
            raise RuntimeError(f'fills / fl mismatch on {(s, d)}: {len(lst)} vs {len(tq)}')
        tas = [o[2] for o in O[(s, d)]]
        for (f, (t, qq)) in zip(lst, tq):
            ta_f = t - f[4]; j = bisect.bisect_right(tas, ta_f + TOL_T) - 1
            if j < 0 or abs(tas[j] - ta_f) >= TOL_T or O[(s, d)][j][4] or abs(f[2] - qq) > 1e-9:
                raise RuntimeError(f'fill on {(s, d)} at {t} does not match an accepted order')
            O[(s, d)][j][3] += qq
    out = {'n_sub': 0}
    for s in (0, 1):
        ev = []
        for (s_, d), lst in O.items():
            if s_ != s:
                continue
            for (ts, end, ta, fq, ref) in lst:
                if not ref:
                    if MUT_POS_ONLY:                      # planted error: the position-only statistic (v3's mislabelled header)
                        ev.append((ts, 0, fq)); ev.append((end, 1, 0.0))
                    else:
                        ev.append((ts, 0, size)); ev.append((end, 1, -(size - fq)))
        ev += [(t, 1, -qq) for (s_, d, t, pt, qq) in r['fl'] if s_ != s]
        ev.sort(key=lambda x: (x[0], x[1]))
        e = 0.0; best = 0.0
        for (t, kind, dv) in ev:
            e += dv
            if kind == 0:
                best = max(best, e); out['n_sub'] += 1
        out[s] = best
    return out


def _rec(orders, fills):
    """Synthetic run record: orders {(s, d): [(ta, end, ref)]}, fills [(s, d, t, qq, ta)] -> the sim_joint fields max_exposure reads."""
    r = dict(orders={}, fills={}, fl=[])
    for (s, d), lst in orders.items():
        r['orders'][(s, d)] = [(ta, end, np.nan, ref) for (ta, end, ref) in lst]
        r['fills'][(s, d)] = []
    for (s, d, t, qq, ta) in sorted(fills, key=lambda x: x[2]):
        r['fills'].setdefault((s, d), []).append((0, 50, qq, False, t - ta, 'hold', np.nan))
        r['fl'].append((s, d, t, 50, qq))
    return r


def selftest():
    bad = []
    e = max_exposure(_rec({(0, 0): [(40.0, 1000.0, False)], (0, 1): [(40.0, 1000.0, False)]}, []))
    if abs(e[0] - 60.0) > 1e-9 or not e[0] > 30.0:
        bad.append(f'two open orders must read 60 (flagged under a 30-sh cap): {e}')
    e = max_exposure(_rec({(0, 0): [(40.0, 500.0, False)], (0, 1): [(740.0, 1000.0, False)]}, [(0, 0, 500.0, 30.0, 40.0)]))
    if abs(e[0] - 60.0) > 1e-9:
        bad.append(f'filled order + replacement must read 60: {e}')
    e = max_exposure(_rec({(0, 0): [(40.0, 40.0, True)], (0, 1): [(40.0, 1000.0, False)]}, []))
    if abs(e[0] - 30.0) > 1e-9:
        bad.append(f'a refused order counts 0: {e}')
    e = max_exposure(_rec({(0, 0): [(40.0, 1000.0, False)], (1, 0): [(40.0, 300.0, False)], (0, 1): [(240.0, 1000.0, False)]},
                          [(1, 0, 100.0, 30.0, 40.0)]))
    if abs(e[0] - 30.0) > 1e-9 or abs(e[1] - 30.0) > 1e-9:
        bad.append(f'an ask fill lowers the bid exposure (30 + 30 - 30): {e}')
    e = max_exposure(_rec({(0, 0): [(40.0, 300.0, False)], (0, 1): [(440.0, 1000.0, False)]}, [(0, 0, 200.0, 10.0, 40.0)]))
    if abs(e[0] - 40.0) > 1e-9:
        bad.append(f'a cancelled order keeps only its fills (10 + 30): {e}')
    return bad


def run_selftest():
    global MUT_POS_ONLY
    b = selftest()
    MUT_POS_ONLY = 1
    try:
        caught = bool(selftest())
    finally:
        MUT_POS_ONLY = 0
    if b or not caught:
        sys.exit(f'JOINT-LADDER v3 EXPOSURE SELFTEST FAILED -> no output: {b}; planted position-only statistic caught: {caught}')
    print('exposure selftests passed (5 synthetic cases incl. the flagged two-order case); planted position-only statistic caught', flush=True)


def day_run(day):
    run_selftest(); J3.all_selftests()
    T0 = time.time(); D = K.load_day(day)
    PO = pd.read_parquet(f'{PD.OUTD}/pred_{day}.parquet'); POI = PO.groupby(['m', 'side']).indices
    st = dict(windows=0, runs=0, submissions=0, maxexp={}, maxpos={}, over={}, over_runs={}); pend = {}; rows = []
    for w in K.window_sides(D):
        pend.setdefault(w['m'], {})[w['side']] = w
        if len(pend[w['m']]) < 2:
            continue
        W = pend.pop(w['m']); st['windows'] += 1; n = W[0]['n']; ctx = J3.window_ctx(W, PO, POI, n)
        for q, shf in J3.CFGS:
            cfg = f'{q}{int(shf)}'
            for arm in J3.ARMS:
                dec, idle = J3.arm_dec(arm, W, ctx)
                for mode, ck in RUN_CAPS:
                    capv = ck * SIZE
                    r = J3.sim_joint(W, q, shf, capv, {0: ctx[0]['vg'], 1: ctx[1]['vg']}, decide=dec, idle_only=idle, mode=mode)
                    e = max_exposure(r); cn = J3.cap_name(mode, ck); mx = max(e[0], e[1])
                    st['runs'] += 1; st['submissions'] += e['n_sub']
                    st['maxexp'][cn] = max(st['maxexp'].get(cn, 0.0), mx); st['maxpos'][cn] = max(st['maxpos'].get(cn, 0.0), r['stat']['maxpos'])
                    ov = int(mx > capv + TOL)
                    st['over'][cn] = st['over'].get(cn, 0) + ov; st['over_runs'][cn] = st['over_runs'].get(cn, 0) + 1
                    rows.append((int(day), cfg, arm, cn, int(W[0]['m']), e[0], e[1], r['stat']['maxpos']))
        for s in (0, 1):
            W[s].pop('cache', None)
        if st['windows'] % 72 == 0:
            r2 = K.rss_mib(); print(f'{day}: {st["windows"]} windows, {time.time() - T0:.0f} s, rss {r2:.0f} MiB', flush=True)
            if r2 > J3.MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r2:.0f} MiB > {J3.MEM_CAP_MIB:.0f} MiB')
    os.makedirs(OUTD, exist_ok=True)
    pd.DataFrame(rows, columns=['day', 'cfg', 'arm', 'capk', 'm', 'exp_bid', 'exp_ask', 'maxpos']).to_parquet(f'{OUTD}/exposure_{day}.parquet',
                                                                                                           index=False)
    json.dump(st, open(f'{OUTD}/exposure_{day}.json', 'w'), indent=1, default=float)
    print(f'{day}: windows {st["windows"]}, runs {st["runs"]}, max exposure {st["maxexp"]}, runs over the cap {st["over"]} in '
          f'{time.time() - T0:.0f} s', flush=True)


def report():
    run_selftest()
    S = {d: json.load(open(f'{OUTD}/exposure_{d}.json')) for d in K.DAYS9 if os.path.exists(f'{OUTD}/exposure_{d}.json')}
    if len(S) != len(K.DAYS9):
        sys.exit(f'REFUSED: exposure files for {len(S)} of {len(K.DAYS9)} days')
    X = pd.concat([pd.read_parquet(f'{OUTD}/exposure_{d}.parquet') for d in K.DAYS9], ignore_index=True)
    X['exp'] = X[['exp_bid', 'exp_ask']].max(axis=1)
    pc_days = sum(1 for d in K.DAYS9 if S[d]['maxexp'].get('soft2', 0.0) > 2 * SIZE + TOL)
    hard = [J3.cap_name('hard', k) for k in (2, 5, 10)]
    over = {cn: sum(S[d]['over'].get(cn, 0) for d in K.DAYS9) for cn in hard + ['soft2']}
    runs = {cn: sum(S[d]['over_runs'].get(cn, 0) for d in K.DAYS9) for cn in hard + ['soft2']}
    if pc_days < len(K.DAYS9):
        verdict = f'VOID (positive control soft2 exceeded 2 x size on {pc_days} of {len(K.DAYS9)} days only)'
    elif any(over[cn] for cn in hard):
        verdict = 'HARD-EXPOSURE-EXCEEDED (' + ', '.join(f'{cn} {over[cn]} of {runs[cn]} runs' for cn in hard if over[cn]) + ')'
    else:
        verdict = 'HARD-EXPOSURE-OK'
    L = [f'JOINT-LADDER v3 POST HOC control: true exposure (position + every share of ours that can still fill, from submission), 9 days, '
         f'client-state runs; as of {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}',
         f'runs {sum(S[d]["runs"] for d in K.DAYS9)}, accepted submissions read {sum(S[d]["submissions"] for d in K.DAYS9)}',
         '', 'cap      | cap sh | max exposure sh | runs over the cap | max |position| sh | window-side exposure p50 / p95 / p99 sh']
    for cn, k in (('hard2', 2), ('hard5', 5), ('hard10', 10), ('soft2', 2)):
        Y = X[X.capk == cn]
        L.append(f'{cn:8s} | {k * SIZE:6.0f} | {max(S[d]["maxexp"][cn] for d in K.DAYS9):15.1f} | {over[cn]:7d} of {runs[cn]:6d} | '
                 f'{max(S[d]["maxpos"][cn] for d in K.DAYS9):17.1f} | {Y.exp.quantile(0.5):6.1f} / {Y.exp.quantile(0.95):6.1f} / '
                 f'{Y.exp.quantile(0.99):6.1f}')
    L += ['', f'positive control: soft2 exceeded 2 x size on {pc_days} of {len(K.DAYS9)} days', f'READING: {verdict}']
    txt = '\n'.join(L)
    open(f'{OUTD}/exposure.txt', 'w').write(txt + '\n')
    json.dump(dict(verdict=verdict, over=over, runs=runs, pc_days=pc_days), open(f'{OUTD}/exposure.json', 'w'), indent=1)
    print(txt, flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        run_selftest()
    elif cmd == 'day':
        day_run(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit(__doc__)
