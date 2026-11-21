"""SELFTEST of pf_ladder_pressure (BINANCE-PRESSURE-GUARD-v1; 2026-11-21, session aea5bfb9). main(P) checks the module; a failed check raises
SystemExit with FAIL. Each group names only what its code runs (review 2c5dbd53 finding 3).
 1 disabled = baseline: the disabled guard equals pf_branch_ladder.LadderInventory on 8 random tapes with random signal caches (fingerprint,
   fills, value); the enabled guard differs on at least one (the cache is used);
 2 the signal: on_value at each side's float32 boundary under the frozen edges (the bid edge is not a float32 value: the next float32 above
   it ON, the one below OFF; the ask edge is one: itself ON, the next below OFF); NaN / 0 / -0 / negative OFF; a value ON under the ask
   edge only stays OFF for a bid, in on_value and in the policy; a non-ok status OFF even with a finite value, and counted; the disabled
   guard OFF; the inventory rule's PULL / WAIT keep their reasons on both sides while the signal is ON;
 3 a hand tape with a synthetic cache: bids ON over ticks 1-39 -> the bids posted at 0 are pulled at their first live review (t = 100), one
   request each; bid slots withheld until clearance; bids re-posted at t = 2,000 (no timer); asks untouched; both sides ON over ticks
   20-29 -> all four pulled at t = 1,000, both sides re-posted at 1,500; every intervention ON with its logged x; the queue observer records
   ahead after submission, at the first live review and at every pressure pull;
 4 the cache helpers with injected builders: tick -> grid lookup and the off_grid / no_market_row / nan statuses, float32 values and ages;
   snapshot_parity (equal, mismatched, unavailable; -0.0 against 0.0 by bits); side_sign; causal_one: a causal builder passes (same,
   same_signal, fired, cached_equal), a builder reading the next step row fails same, a builder ignoring the step row fails fired;
 5 the selection control: a worse selected set is SELECTION_BEYOND_EXPOSURE, a random one is not, a thin one INCONCLUSIVE; the draws
   reproduce with the seed; a censored high-quantity selected state changes neither valid term; matching on side and on the B >= 200
   indicator each removes a confounded difference; a fully degenerate population (varied outcomes) draws exactly the selected set; per-side figures;
 6 measurement on hand records: fill contributions, the END identity (eligible +1 s + movement + late = END = value), the shared cutoff,
   valid quantities and censor counts per horizon, the exact inventory integrals (time at |I| = 100 counted), lifetimes, lost / added
   fills (a price difference is not a match; guard - baseline END = added - lost), the arm counts (pulls, withholds, reducing-side, near
   END), the first-review pull rate, the queue re-entry, the readings (IMPROVES / WORSENS / MIXED; better_selection / quoting_less / mixed);
 7 guards, each isolated with every earlier guard passing and its refusal message asserted: run_day before the freeze, under another rule,
   under another source, the full run without (or with a refusing) projection, an existing output; with every guard passing run_day
   reaches the data load (a sentinel); run_selection and measure refuse to overwrite; measure refuses without the selection control."""
import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

FAIL = 'PRESSURE SELFTEST FAILED'


class _Sentinel(Exception):
    pass


def run(mod):
    main(mod)


def _cache(nt, x0, x1, st0=None, st1=None):
    z = lambda v: np.asarray(v, np.float32)
    return dict(x={0: z(x0), 1: z(x1)}, age={0: np.full(nt, 5., np.float32), 1: np.full(nt, 5., np.float32)},
                status={0: np.zeros(nt, np.int8) if st0 is None else np.asarray(st0, np.int8), 1: np.zeros(nt, np.int8) if st1 is None else np.asarray(st1, np.int8)},
                kk={0: np.arange(nt), 1: np.arange(nt)}, nt=nt, grids_equal=True)


def main(P):  # noqa: C901
    M, LB, BE, pd = P.M, P.LB, P.BE, P.pd; bad = []
    ok = lambda name, cond: None if cond else bad.append(name)
    import pf_ladder_rest_selftest as V1
    fr = V1.fr; E = P.frozen_edges(); ON = {0: 1.0, 1: 1.0}
    # ---- 1 disabled = baseline
    try:
        rng = np.random.default_rng(41); nb = nd = 0
        for i in range(8):
            p = BE.rand_tape(rng, 300, 500, end=30000.); nt = 600
            vals = np.array([-0.5, 0., 0.24, 0.47, 1.0], np.float32); cz = _cache(nt, rng.choice(vals, nt), rng.choice(vals, nt), rng.choice([0, 0, 0, 1, 3], nt), rng.choice([0, 0, 0, 2], nt))
            B1 = LB.LadderBranch(p, LB.LADDER, end=30000., episode=i % 2); LB.run_episode(B1, LB.LadderInventory(B1)); L1 = LB.finish(B1)
            B2 = LB.LadderBranch(p, LB.LADDER, end=30000., episode=i % 2); LB.run_episode(B2, P.PressureGuard(B2, cz, E, enabled=False)); L2 = LB.finish(B2)
            B3 = LB.LadderBranch(p, LB.LADDER, end=30000., episode=i % 2); LB.run_episode(B3, P.PressureGuard(B3, cz, E, enabled=True)); L3 = LB.finish(B3)
            nb += int(not (LB.fingerprint(B1) == LB.fingerprint(B2) and L1['fills'] == L2['fills'] and L1['value'] == L2['value']))
            nd += int(LB.fingerprint(B1) != LB.fingerprint(B3))
        ok(f'1 the disabled guard equals the baseline policy ({nb} of 8 differ)', nb == 0)
        ok(f'1 the enabled guard uses the cache ({nd} of 8 differ)', nd >= 1)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'1 disabled: stopped ({e!r})'[:200])
    # ---- 2 the signal and precedence
    try:
        hi0 = np.float32(P.LAST_EDGES[0]); hi0 = hi0 if float(hi0) >= P.LAST_EDGES[0] else np.nextafter(hi0, np.float32(np.inf)); lo0 = np.nextafter(hi0, np.float32(-np.inf))
        hi1 = np.float32(P.LAST_EDGES[1]); lo1 = np.nextafter(hi1, np.float32(-np.inf))
        ok('2 the bid boundary (the edge is not a float32 value)', float(np.float32(P.LAST_EDGES[0])) != P.LAST_EDGES[0] and P.on_value(float(hi0), E[0]) and not P.on_value(float(lo0), E[0]))
        ok('2 the ask boundary (the edge is a float32 value)', float(hi1) == P.LAST_EDGES[1] and P.on_value(float(hi1), E[1]) and not P.on_value(float(lo1), E[1]))
        ok('2 NaN / 0 / -0 / negative OFF', not any(P.on_value(v, E[s]) for v in (np.nan, 0., -0., -1., -5.) for s in (0, 1)))
        ok('2 a value ON under the ask edge only stays OFF for a bid', P.on_value(0.3, E[1]) and not P.on_value(0.3, E[0]))
        Bh = LB.LadderBranch(BE.tape([fr(0, 0, [(50, 0)], [(52, 0)])], [], 50.), LB.LADDER, end=3000.)
        cz = _cache(60, [1.] * 60, [1.] * 60, [0] * 10 + [1] * 50, [0] * 10 + [3] * 50)
        g_on, g_off = P.PressureGuard(Bh, cz, E), P.PressureGuard(Bh, cz, E, enabled=False)
        v = lambda t, s, **k: dict(dict(t=float(t), side=s, position=0., B=0., A=0., mode='UNRESTRICTED', avail=200.), **k)
        ok('2 ON at an ok tick, OFF at a non-ok tick with a finite value (and counted), OFF when disabled',
           g_on.signal(v(100, 0))[0] and g_on.signal(v(100, 1))[0] and not g_on.signal(v(600, 0))[0] and not g_on.signal(v(600, 1))[0]
           and g_on.missing[(0, 'off_grid')] == 1 and g_on.missing[(1, 'nan')] == 1 and not g_off.signal(v(100, 0))[0])
        g3 = P.PressureGuard(Bh, _cache(60, [0.3] * 60, [0.3] * 60), E)
        ok('2 the policy reads its own side\'s edge (0.3: OFF for a bid, ON for an ask)', not g3.signal(v(100, 0))[0] and g3.signal(v(100, 1))[0])
        o = lambda s: dict(gid=3, price=50. if s == 0 else 52., rung=0., size=100., state='live', known_filled=0., act=58., submit=0.)
        ok('2 the inventory rule keeps precedence and its reasons while ON',
           g_on.live(v(100, 1, mode='SHORT_RESTRICTED', position=-120.), o(1)) == ('PULL', 'inventory_pull') and g_on.slot(v(100, 1, mode='SHORT_RESTRICTED', position=-120.), 0, 52.)[2] == 'restricted'
           and g_on.live(v(100, 0, mode='LONG_RESTRICTED', position=120.), o(0)) == ('PULL', 'inventory_pull') and g_on.slot(v(100, 0, mode='LONG_RESTRICTED', position=120.), 0, 50.)[2] == 'restricted'
           and g_on.live(v(100, 0), o(0)) == ('PULL', 'pressure_pull') and g_on.slot(v(100, 1), 0, 52.) == ('WAIT', None, 'pressure_withhold'))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'2 signal: stopped ({e!r})'[:200])
    # ---- 3 the hand tape
    try:
        frames = [fr(0, 0, [(50, 0), (49, 0)], [(52, 0), (53, 0)])]; p = BE.tape(frames, [], 50.); nt = 80
        xa = np.zeros(nt, np.float32); xa[1:40] = 1.; Ba = LB.LadderBranch(p, LB.LADDER, end=4000., episode=0); ga = P.PressureGuard(Ba, _cache(nt, xa, np.zeros(nt)), E)
        oa = P.QueueObs(Ba, ga); LB.run_episode(Ba, ga, observe=oa)
        b0 = [o for o in Ba.E.o if o['side'] == 0 and o['submit'] == 0.]; a0 = [o for o in Ba.E.o if o['side'] == 1]; reqs = [e for e in Ba.E.events if e[1] == 'policy_request']
        ok(f'3 the bids posted at 0 are pulled at their first live review, once each ({[(o["req"], o["why"]) for o in b0]})',
           len(b0) == 2 and all(o['req'] == 100. and o['why'] == 'policy' for o in b0) and all(sum(1 for e in reqs if e[2] == o['gid']) == 1 for o in b0))
        later = [o['submit'] for o in Ba.E.o if o['side'] == 0 and o['submit'] > 0.]
        ok(f'3 bid slots withheld until clearance, re-posted at 2,000 (no timer): {later}', later and min(later) == 2000.
           and any(z[10] == 'pressure_withhold' and z[1] == 0 and 200. <= z[0] < 2000. for z in ga.log) and not any(z[0] >= 2000. for z in ga.log))
        ok('3 asks untouched while only the bids are ON', all(o['req'] is None for o in a0) and not any(z[1] == 1 for z in ga.log) and len(a0) == 2)
        ok('3 every intervention ON with its logged x', all(P.on_value(z[5], E[z[1]]) and z[5] == 1. for z in ga.log))
        ok(f'3 the queue observer: ahead after submission for every order, first review and pulls ({sorted(oa.pull)})',
           len(oa.ahead0) == len(Ba.E.o) and sorted(oa.pull) == sorted(o['gid'] for o in b0) and all(oa.first[o['gid']][0] == 100. and oa.pull[o['gid']][0] == 100. for o in b0))
        xb = np.zeros(nt, np.float32); xb[20:30] = 1.; Bb = LB.LadderBranch(p, LB.LADDER, end=4000., episode=1); gb = P.PressureGuard(Bb, _cache(nt, xb, xb), E); LB.run_episode(Bb, gb)
        f0 = [o for o in Bb.E.o if o['submit'] == 0.]; nx = {s: min([o['submit'] for o in Bb.E.o if o['side'] == s and o['submit'] > 0.] or [None]) for s in (0, 1)}
        ok(f'3 both sides ON: all four pulled at 1,000, both re-posted at 1,500 ({[o["req"] for o in f0]}, {nx})',
           len(f0) == 4 and all(o['req'] == 1000. for o in f0) and nx == {0: 1500., 1: 1500.})
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'3 hand tape: stopped ({e!r})'[:220])
    # ---- 4 the cache helpers and causal_one with injected builders
    try:
        n, t0 = 30, 1000.; tg0 = t0 + 50. * np.array([i for i in range(n) if i != 5], float); tg1 = t0 + 50. * np.arange(n, dtype=float); NM = len(M.FC.MARKET)
        W = {0: dict(tg=tg0), 1: dict(tg=tg1)}

        def fb(W_, fair, depth, ch):
            out = {}
            for s in (0, 1):
                out[s] = {}
                for j in ch[s]:
                    if s == 0 and j == 7:
                        continue
                    row = np.zeros(NM, np.float32); row[P.FI] = np.nan if (s == 0 and j == 9) else (0.25 * (j % 4) * (1 if s == 1 else -1)); row[P.AGE_I] = float(j); out[s][j] = (row, 1.)
            return out
        cz = P.pressure_cache(W, None, None, t0, 50. * n, fb); st0 = [P.ST_CODES[int(c)] for c in cz['status'][0]]
        ok(f'4 statuses: off_grid at tick 5, no_market_row at tick 8 (grid 7), nan at tick 10 (grid 9) ({st0[:12]})',
           st0[5] == 'off_grid' and st0[8] == 'no_market_row' and st0[10] == 'nan' and st0.count('ok') == n - 3 and all(c == 0 for c in cz['status'][1]) and not cz['grids_equal'])
        ok('4 values and ages at the grid steps', cz['x'][0][6] == np.float32(-0.25 * (5 % 4)) and cz['age'][0][6] == 5. and cz['x'][1][6] == np.float32(0.25 * (6 % 4)) and cz['x'][0][4] == np.float32(-0.))
        pr_ = P.snapshot_parity(cz, [(0, 300., float(cz['x'][0][6])), (0, 250., 0.), (0, 100., 0.5), (1, 0., -0.), (1, 0., 0.)])
        ok(f'4 snapshot_parity by bits ({pr_})', pr_['equal'] == 2 and pr_['unavailable'] == 1 and pr_['mismatched'] == 2)
        Wq = {0: dict(tg=tg1), 1: dict(tg=tg1)}

        def fq(W_, fair, depth, ch):
            o = fb(Wq, fair, depth, ch)
            for j in list(o[0]):
                o[0][j][0][P.FI] = -0.25 * (j % 4)
            if 3 in o[1]:
                o[1][3][0][P.FI] = 9.
            return o
        sg = P.side_sign(P.pressure_cache(Wq, None, None, t0, 50. * n, fq))
        ok(f'4 side_sign ({sg})', sg['ticks'] == n - 1 and sg['mismatched'] == 1)
        X37 = 37; nn = 40; tgc = t0 + 50. * np.arange(nn, dtype=float); Xs = {s: np.zeros((nn, X37), float) for s in (0, 1)}
        Xs[0][10, P.BN_COL] = -1.; Xs[1][12, P.BN_COL] = 1.
        Wc = {s: dict(tg=tgc, X=Xs[s], jk=np.arange(nn)) for s in (0, 1)}

        def mk(read):
            def f(W_, fair, depth, ch):
                out = {}
                for s in (0, 1):
                    out[s] = {}
                    for j in ch[s]:
                        row = np.zeros(NM, np.float32); row[P.FI] = read(W_[s]['X'], j) * (1. if s == 1 else -1.); out[s][j] = (row, 1.)
                return out
            return f

        def perturb(W_, fair, depth, j, t, rng):
            mut = {s: dict(w, X=np.array(w['X'], copy=True)) for s, w in W_.items()}
            for s in mut:
                mut[s]['X'][j + 1:] += rng.normal(0., 3., mut[s]['X'][j + 1:].shape)
            return mut, fair, depth
        causal, leaky, deaf = mk(lambda X_, j: X_[j, P.BN_COL]), mk(lambda X_, j: X_[min(j + 1, len(X_) - 1), P.BN_COL]), mk(lambda X_, j: 1.)
        cc = P.pressure_cache(Wc, None, None, t0, 50. * nn, causal); r1 = P.causal_one(Wc, None, None, cc, 0, 10, E[0], 'ab' * 32, causal, perturb)
        r2 = P.causal_one(Wc, None, None, cc, 1, 3, E[1], 'cd' * 32, causal, perturb)
        ok(f'4 a causal builder passes (ON pulled to OFF, OFF pushed to ON): {r1["on"]}, {r2["on"]}', r1['on'] and not r2['on'] and all(r[k] for r in (r1, r2) for k in ('same', 'same_signal', 'fired', 'cached_equal')))
        r3 = P.causal_one(Wc, None, None, P.pressure_cache(Wc, None, None, t0, 50. * nn, leaky), 0, 9, E[0], 'ef' * 32, leaky, perturb)
        r4 = P.causal_one(Wc, None, None, P.pressure_cache(Wc, None, None, t0, 50. * nn, deaf), 0, 10, E[0], '01' * 32, deaf, perturb)
        ok(f'4 a builder reading the next step row fails same ({r3["same"]}); one ignoring the step row fails fired ({r4["fired"]})', not r3['same'] and not r4['fired'])
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'4 cache: stopped ({e!r})'[:220])
    # ---- 5 the selection control
    try:
        rng = np.random.default_rng(5)

        def states(fn, n_m=4, n_i=40, days=('20261024', '20261025')):
            rr = []
            for day in days:
                for m in range(n_m):
                    for s in (0, 1):
                        for i in range(n_i):
                            sel, y, b = fn(s, i)
                            rr.append(dict(day=day, m=m, kind='live', side=s, gid=i, t=0., win_n=1, win_q=10., win_mo1s=y, censored_1s=False,
                                           adv_bn_r250=1.0 if sel else 0.0, ctx_bid_reserved_shares=b, fp_remaining=float(10 + 40 * (i % 2)), ctx_generation_age_ms=float(100 + 3000 * (i % 3)), inv_known=0.))
            return pd.DataFrame(rr)
        S1 = states(lambda s, i: (i < 12, -20. if i < 12 else 0., 150.)); r1 = P.selection(S1, E, draws=200); r1b = P.selection(S1, E, draws=200)
        ok(f'5 a worse selected set is SELECTION_BEYOND_EXPOSURE ({r1["answer"]})', r1['answer'] == 'SELECTION_BEYOND_EXPOSURE' and r1['selected'] == 192)
        ok('5 the draws reproduce with the seed', r1['null_mean'] == r1b['null_mean'] and r1['null_p05'] == r1b['null_p05'])
        S2 = states(lambda s, i: (i < 12, 0., 150.)); r2 = P.selection(S2, E, draws=200)
        ok(f'5 a random selection is not ({r2["answer"]})', r2['answer'] == 'NO_SELECTION_BEYOND_EXPOSURE')
        r3 = P.selection(S1[(S1.m == 0) & (S1.day == '20261024') & (S1.side == 0)].head(14), E, draws=50)
        ok(f'5 a thin control is INCONCLUSIVE ({r3["answer"]})', r3['answer'] == 'INCONCLUSIVE')
        S4 = pd.concat([S1, pd.DataFrame([dict(S1.iloc[0].to_dict(), gid=999, win_q=5000., win_mo1s=np.nan, censored_1s=True)])], ignore_index=True); r4 = P.selection(S4, E, draws=200)
        ok('5 a censored high-quantity selected state changes neither valid term', r4['J'] == r1['J'] and r4['J_per_valid_share'] == r1['J_per_valid_share']
           and r4['valid_window_q'] == r1['valid_window_q'] and r4['excluded_window_q'] == 5000. and r4['censored'] == 1)
        S5 = states(lambda s, i: (s == 0 and i < 30, -20. if s == 0 else 0., 150.)); r5 = P.selection(S5, E, draws=200)
        ok(f'5 matching on side removes a side-confounded difference ({r5["answer"]})', r5['answer'] == 'NO_SELECTION_BEYOND_EXPOSURE')
        S6 = states(lambda s, i: ((i % 4 == 0) or (i < 20 and i % 2 == 0), -20. if i < 20 else 0., 200. if i < 20 else 150.)); r6 = P.selection(S6, E, draws=200)
        ok(f'5 matching on the B >= 200 indicator removes a confounded difference ({r6["answer"]})', r6['answer'] == 'NO_SELECTION_BEYOND_EXPOSURE')
        S7 = S1.copy(); S7['adv_bn_r250'] = 1.; r7 = P.selection(S7, E, draws=20)
        ok('5 a fully degenerate population draws the selected set', abs(r7['null_mean'] - r7['J']) < 1e-12 and r7['movable_strata'] == 0)
        ok('5 per-side figures', set(r1['per_side']) == {'0', '1'} and r1['per_side']['0']['selected'] == 96 and r1['per_side']['1']['J'] is not None)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'5 selection: stopped ({e!r})'[:220])
    # ---- 6 measurement and readings
    try:
        mm = lambda m0, m1: [m0, 0., m0, 0., m0, 0., m1, 0., None, None, None, None]
        orders = [dict(gid=0, side=0, rung=0., req=None, why=None, act=58., remove=4000., size=100., refused=False, submit=0., price=49.),
                  dict(gid=1, side=1, rung=1., req=20., why='structural', act=58., remove=1500., size=10., refused=False, submit=0., price=51.),
                  dict(gid=2, side=1, rung=0., req=None, why=None, act=1558., remove=4000., size=100., refused=False, submit=1500., price=51.)]
        fills = [[1000., 0, 0, 100., 49., 1], [1200., 1, 1, 10., 51., 2], [2500., 2, 1, 5., 51., 3], [3500., 2, 1, 5., 51., 4]]
        mark = 50.; pos = 100. - 20.; cash = -100. * 49. + 20. * 51.; rec = dict(mark=mark, value=cash + pos * mark, cash=cash, position=pos, end=4000., fills=fills,
                                                                                  fill_mids=[mm(50., 49.5), mm(50., 50.5), mm(50., None), mm(50., 52.)], orders=orders,
                                                                                  guard_log=[[0., 1, 'slot', -1, 0., 1., 5., 100., 5., 'UNRESTRICTED', 'pressure_withhold'],
                                                                                             [50., 0, 'live', 0, 0., 1., 5., 100., -1., 'UNRESTRICTED', 'pressure_pull'],
                                                                                             [3990., 1, 'slot', -1, 1., 1., 5., 50., 7., 'UNRESTRICTED', 'pressure_withhold']],
                                                                                  queue_venue=dict(ahead0={'0': [0., 30., 100.], '1': [0., 5., 10.], '2': [1500., 80., 100.]},
                                                                                                   first={'0': [100., 20., 100.]}, pull={'1': [1000., 4., 10.]}),
                                                                                  queue_known={'0': dict(first=25.), '1': dict(first=None)}, missing={'0|off_grid': 2, '1|nan': 1})
        cmds = np.array([[100., 0, M.CMD_KINDS.index('live'), 0, 0., 49., np.nan, M.CMD_ACTIONS.index('KEEP'), P.REASONS.index('keep')],
                         [150., 0, M.CMD_KINDS.index('live'), 0, 0., 49., np.nan, M.CMD_ACTIONS.index('PULL'), P.REASONS.index('pressure_pull')],
                         [100., 1, M.CMD_KINDS.index('live'), 1, 1., 51., np.nan, M.CMD_ACTIONS.index('PULL'), P.REASONS.index('inventory_pull')]], float)
        x = P.arm_metrics(rec, dict(commands=cmds), 4000.)
        ident = x['all_id_mo1s'] + x['all_id_move'] + x['all_id_late']
        ok(f'6 the END identity ({ident:.6f} = {x["all_end"]:.6f} = {rec["value"]:.6f})', abs(ident - x['all_end']) < 1e-9 and abs(x['all_end'] - rec['value']) < 1e-9
           and abs(x['ask_id_mo1s'] + x['ask_id_move'] + x['ask_id_late'] - x['ask_end']) < 1e-9 and x['all_id_late'] == x['ask_end'] - (-10. * (50.5 - 51.)) - (-10. * (50. - 50.5)))
        ok(f'6 the shared cutoff, valid quantities and censor counts ({x["all_cut_q"]}, {x["ask_cut_censored"]}, {x["all_cut_valid_q"]})',
           x['all_cut_q'] == 115. and x['ask_cut_censored'] == 1 and x['all_cut_valid_q'] == 110. and x['all_edge_valid_q'] == 120. and x['all_mo50_censored'] == 0
           and x['bid_cut_mo1s'] == 100. * (49.5 - 49.) and x['ask_good_mo1s'] == 5. and x['bid_bad_mo1s'] == 0.)
        ok(f'6 arm counts ({x["pulls"]}, {x["withholds"]}, {x["reducing_suppressed"]}, {x["withholds_near_end"]})', x['pulls'] == 1 and x['withholds'] == 2 and x['bid_reducing_suppressed'] == 1
           and x['ask_reducing_suppressed'] == 2 and x['withholds_near_end'] == 1 and x['bid_req_pressure_pull'] == 1 and x['ask_req_inventory_pull'] == 1)
        ok('6 the first-review pull rate', x['reviewed'] == 2 and x['first_review_pulls'] == 1 and x['first_review_pressure'] == 0)
        ok(f'6 lifetimes ({x["bid_life_mean_ms"]}, {x["ask_life_mean_ms"]})', x['bid_life_mean_ms'] == 1000. - 58. and x['ask_life_mean_ms'] == ((1200. - 58.) + (4000. - 1558.)) / 2.)
        ok(f'6 the queue re-entry ({x["q_reentry_gap_mean"]}, {x["q_surrendered_mean"]})', x['q_pulls'] == 1 and x['q_reentries'] == 1 and x['q_reentry_gap_mean'] == 500.
           and x['q_surrendered_mean'] == 76. and x['q_reentry_same_price'] == 1. and x['q_first_known_mean'] == 25. and x['missing_bid'] == 2 and x['missing_ask'] == 1)
        iv = P.inventory_exact(dict(fills=[[1000., 0, 0, 100., 49., 0], [3000., 0, 1, 40., 51., 0]], position=60.), 4000.)
        ok(f'6 the exact inventory integrals ({iv})', abs(iv['mean_abs_inv'] - (100. * 2000. + 60. * 1000.) / 4000.) < 1e-12 and iv['share_abs_ge_S'] == .5 and iv['peak_abs_inv'] == 100.)
        rg = copy.deepcopy(rec); rg['fills'] = [fills[0], [1200., 1, 1, 10., 52., 2], fills[3]]; rg['fill_mids'] = [mm(50., 49.5), mm(50., 50.5), mm(50., 52.)]
        rg['position'] = 100. - 15.; rg['cash'] = -100. * 49. + 10. * 52. + 5. * 51.; rg['value'] = rg['cash'] + rg['position'] * mark; la = P.lost_added(rec, rg, 4000.)
        ok(f'6 lost / added (a price difference is not a match; identity gap {la["identity_gap"]:.2e})', la['lost_n'] == 2 and la['added_n'] == 1 and abs(la['identity_gap']) < 1e-9)
        R_ = pd.DataFrame([dict(day=d, m=m, arm=a, value=v, all_q=q, all_cut_mo1s=c, all_cut_valid_q=10., bid_cut_mo1s=c, bid_cut_valid_q=5., ask_cut_mo1s=c, ask_cut_valid_q=5., bid_end=v / 2,
                                ask_end=v / 2, both_exact=.4, bid_exact=.5, ask_exact=.5, pulls=0, withholds=0, reducing_suppressed=0, mean_abs_inv=1., share_abs_ge_S=0., terminal_inv=0.)
                           for d in ('20261024', '20261025') for m in (1, 2) for a, v, q, c in (('disabled', 0., 100., -10.), ('pressure', 5., 80., -5.))])
        p1 = P.predicates(R_, 'SELECTION_BEYOND_EXPOSURE'); p2 = P.predicates(R_, 'NO_SELECTION_BEYOND_EXPOSURE')
        R2 = R_.copy(); R2.loc[R2.arm == 'pressure', 'all_cut_mo1s'] = -20.; p3 = P.predicates(R2, 'NO_SELECTION_BEYOND_EXPOSURE')
        R3 = R_.copy(); R3.loc[(R3.arm == 'pressure') & (R3.day == '20261025'), 'value'] = -9.; p4 = P.predicates(R3, 'SELECTION_BEYOND_EXPOSURE')
        ok(f'6 readings ({p1["interpretation"]}, {p2["interpretation"]}, {p3["interpretation"]}, {p4["interpretation"]})',
           p1['END_paired']['answer'] == 'IMPROVES_EVERY_DAY' and p1['interpretation'] == 'better_selection' and p2['interpretation'] == 'mixed'
           and p3['interpretation'] == 'quoting_less' and p3['mo1s_per_share']['all']['answer'] == 'WORSENS_EVERY_DAY' and p4['END_paired']['answer'] == 'MIXED' and p4['interpretation'] == 'mixed')
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'6 measurement: stopped ({e!r})'[:220])
    # ---- 7 guards
    saved = dict(frozen=P.frozen, rule_sha=P.rule_sha, sources=P.sources); ld = M.load_day; tmp = Path(tempfile.mkdtemp(dir=str(P.A.DATA)))
    try:
        day = '20261024'; man = dict(rule_sha256=P.rule_sha(), sources=P.sources(), pilot={day: [3697902]}, complete={day: [3697902]}, days=[day])
        mp = tmp / 'manifest.json'; mp.write_text(json.dumps(man)); pp = tmp / 'projection.json'

        def refuses(word, **kw):
            try:
                P.run_day(day, man_path=mp, proj_path=pp, out=tmp / 'out', **kw); return False
            except ValueError as e:
                return word in str(e)
            except _Sentinel:
                return False

        def sentinel(*a, **k):
            raise _Sentinel()
        M.load_day = sentinel                                                              # a guard that lets a run through stops here, never at a real load
        P.frozen = lambda: False; ok('7 run_day refuses before the freeze', refuses('committed', pilot=True))
        P.frozen = lambda: True; P.rule_sha = lambda: 'x' * 64; ok('7 run_day refuses another rule', refuses('rule', pilot=True)); P.rule_sha = saved['rule_sha']
        P.sources = lambda: dict(saved['sources'](), **{'pf_stop_model.py': '0' * 64}); ok('7 run_day refuses another source', refuses('source', pilot=True)); P.sources = saved['sources']
        ok('7 the full run refuses without the projection', refuses('projection', pilot=False))
        pp.write_text(json.dumps(dict(run_rest=False))); ok('7 the full run refuses a refusing projection', refuses('projection', pilot=False))
        Pd = P.day_paths(day, True, tmp / 'out'); Pd['rec'].parent.mkdir(parents=True, exist_ok=True); Pd['rec'].write_text('{}'); ok('7 run_day refuses an existing output', refuses('write-once', pilot=True))
        Pd['rec'].unlink()
        try:
            P.run_day(day, pilot=True, man_path=mp, proj_path=pp, out=tmp / 'out'); bad.append('7 the all-pass run_day did not reach the load')
        except _Sentinel:
            pass
        except Exception as e:                                                             # noqa: BLE001
            bad.append(f'7 the all-pass run_day stopped before the load ({e!r})'[:200])
        M.load_day = ld
        so = tmp / 'sel'; so.mkdir(); (so / 'selection.json').write_text('{}')
        try:
            P.run_selection(out=so, require_frozen=False); bad.append('7 run_selection overwrote')
        except ValueError as e:
            ok('7 run_selection refuses to overwrite', 'write-once' in str(e))
        mo = tmp / 'meas'; (mo / 'measure').mkdir(parents=True)
        try:
            P.measure(out=mo, man_path=mp); bad.append('7 measure ran without the selection control')
        except ValueError as e:
            ok('7 measure refuses without the selection control', 'selection' in str(e))
        (mo / 'measure' / 'audit.json').write_text('{}')
        try:
            P.measure(out=mo, man_path=mp); bad.append('7 measure overwrote')
        except ValueError as e:
            ok('7 measure refuses to overwrite', 'write-once' in str(e))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'7 guards: stopped ({e!r})'[:200])
    finally:
        P.frozen, P.rule_sha, P.sources = saved['frozen'], saved['rule_sha'], saved['sources']; M.load_day = ld; shutil.rmtree(tmp, ignore_errors=True)
    if bad:
        raise SystemExit(FAIL + ': ' + ' | '.join(bad))
    print('PRESSURE SELFTEST PASSED: disabled = baseline, the signal and precedence, the hand tape (pulls, withholds, re-entry, queue), the cache helpers and '
          'causal plumbing, the selection control, measurement and readings, guards')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pf_ladder_pressure as P_
    main(P_)
