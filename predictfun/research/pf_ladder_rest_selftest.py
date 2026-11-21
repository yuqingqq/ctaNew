"""SELFTEST of pf_branch_ladder and pf_ladder_rest (LADDER_REST_INVENTORY_v1; 2026-11-21, session aea5bfb9). main(M) checks the runner module M
(and its client M.LB); any failed check prints it and exits with FAIL. The plan's necessary checks (plan.md section 5) on synthetic tapes:
 1 legacy reproduction: LadderBranch LEGACY + FixedPostKeep = pf_branch_engine.Branch + POST_KEEP on random tapes (both episode parities);
 2 the long partial-fill / queue case: one 100-share bid that fills only after 5 s in three partial events while later same-price arrivals
   grow the shown size (the queue ahead never grows), then its replacement joins behind the shown queue; a RESETTING engine (queue ahead
   reset to the shown size) and a TRUNCATED one (END 5 s) are rejected by the same check;
 3 ladder fixtures: two bid levels at I = 0 get 100 + 100 shares; one sweep fills both in price priority within the trade's size; a partial
   fill of one gid leaves the other unchanged; I = +40 gives two 60-share bids; the cap holds both levels and a cancel-pending order until
   its acknowledgement; one structural pull (cross) leaves the other order resting with its subreason logged; a pending price is never
   duplicated; a quote crossing our own unclosed opposite order is own_cross_blocked;
 4 hidden notices: an unnotified fill changes no client view, mode or command before its notice (+100 ms); the delivered control changes;
 5 inventory mode boundaries (next_mode table, reversal) and a run through LONG_RESTRICTED (the live bid pulled, its reservation released
   only at the acknowledgement, new bids blocked, resumption at |I| <= 50);
 6 cancel timing: a pull at g lands at g + 37: a trade at g + 36 fills, one at g + 37 does not;
 7 resume identity: a saved / resumed run equals the continuous one; another configuration, END, tape or a changed state is refused;
 8 the observer and the reconstruction: observer states equal gid_state, running with / without it gives the same fingerprint, every
   snapshot reconstructs from the saved events and a corrupted snapshot does not;
 9 features (PF_TEACHER_SYN): known_steps_q = pf_pipeline_export.known_steps at size 30, rung 0, exposure 5 s; legacy_check on a synthetic
   world: engine parity and the 107 inputs equal to pf_r2_teacher.Teacher.inputs;
10 measurement on a synthetic market record: generation / fill values sum to cash + inventory x mark; minute increments telescope; markout =
   edge + movement; downtime precedence; summaries and predicates; population (select_markets, build_manifest write-once); guards (run_day
   before the freeze, measure write-once, load_outputs refuses a changed output)."""
import copy
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

FAIL = 'LADDER SELFTEST FAILED'


def fr(t, recv, bids, asks):
    """A 22-column frame with explicit levels [(price, size)] (empty levels -1 / 0)."""
    bp = [p for p, _ in bids] + [-1.] * (5 - len(bids)); bs = [z for _, z in bids] + [0.] * (5 - len(bids))
    ap = [p for p, _ in asks] + [-1.] * (5 - len(asks)); as_ = [z for _, z in asks] + [0.] * (5 - len(asks))
    return [float(t), float(recv)] + bp + bs + ap + as_


def run(mod):
    """The mutation harness entry: mod is the runner, or the client (then the runner is imported over it)."""
    import importlib
    main(mod if hasattr(mod, 'run_ladder') else importlib.import_module('pf_ladder_rest'))


def main(M):  # noqa: C901
    LB, BE, C = M.LB, M.BE, M.C; bad = []
    ok = lambda name, cond: None if cond else bad.append(name)

    class Scripted(LB.Policy):
        """A test policy: KEEP live unless (t, gid) is in pulls; POST q at the listed (t, side, rung) slots, else WAIT."""

        def __init__(self, B, posts, pulls=(), q=None):
            super().__init__('scripted', B.end); self.posts, self.pulls, self.q = set(posts), set(pulls), q or B.q0

        def live(self, v, o):
            return ('PULL', 'test') if (int(v['t']), o['gid']) in self.pulls else ('KEEP', 'keep')

        def slot(self, v, r, price):
            return ('POST', self.q, 'post') if (int(v['t']), v['side'], r) in self.posts or ('any', v['side'], r) in self.posts else ('WAIT', None, 'test')

    # ---- 1 legacy reproduction (random tapes, plus one-sided books for the fallback path)
    try:
        rng = np.random.default_rng(11); nbad = 0; tapes = [BE.rand_tape(rng, 40, 80) for _ in range(40)]
        for i in range(4):
            p = BE.rand_tape(rng, 40, 80); f = np.array(p['frames'], float, copy=True); sl = slice(10 + 5 * i, 20 + 5 * i); f[sl, 2 + 10 * (i % 2)] = -1.
            f[sl, 7 + 10 * (i % 2)] = 0.; tapes.append(dict(p, frames=f))
        nfb = 0
        for i, p in enumerate(tapes):
            B1 = BE.run(p, BE.POST_KEEP, 0, 0, BE.Branch(p, i % 2)); L1 = BE.finish(B1)
            B2 = LB.LadderBranch(p, LB.LEGACY, episode=i % 2); LB.run_episode(B2, LB.FixedPostKeep(B2)); L2 = LB.finish(B2); nfb += len(L1['fallbacks'])
            nbad += int(not (L1['fills'] == L2['fills'] and L1['decisions'] == L2['decisions'] and L1['requests'] == L2['requests'] and B1.E.events == B2.E.events
                             and B1.closed == B2.closed and L1['constraints'] == L2['constraints'] and L1['fallbacks'] == L2['fallbacks'] and L1['value'] == L2['value']
                             and L1['open_at_end'] == L2['open_at_end']))
        ok(f'1 legacy reproduction: {nbad} of {len(tapes)} tapes differ ({nfb} fallbacks exercised)', nbad == 0 and nfb > 0)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'1 legacy reproduction: stopped ({e!r})'[:200])

    # ---- 2 the long partial-fill / queue case
    def long_case(engine_cls=None, config=LB.LADDER, end=60000.):
        frames = [fr(0, 0, [(50, 40)], [(55, 100)]), fr(6000, 6000, [(50, 200)], [(55, 100)]), fr(59000, 59000, [(50, 200)], [(55, 100)])]
        trades = [(7000, 0, 50, 30), (8000, 0, 50, 30), (9000, 0, 50, 50), (20000, 0, 50, 40), (30000, 0, 50, 100)]
        p = BE.tape(frames, trades, 50.); B = LB.LadderBranch(p, config, end=None if config == LB.LEGACY else end)
        if engine_cls is not None:
            E2 = engine_cls(p, B.end); B.E = E2
        LB.run_episode(B, Scripted(B, {('any', 0, 0)}, q=30. if config == LB.LEGACY else 100.)); L_ = LB.finish(B)
        return B, L_

    def long_ok(B, L_):
        f1 = [(f[0], f[3]) for f in L_['fills'] if f[1] == 0]; g2 = [o for o in B.E.o if o['gid'] == 1]
        return bool(f1 == [(8000., 20.), (9000., 50.), (20000., 30.)] and B.closed.get(0, (None,))[0] == 'filled' and len(g2) == 1
                    and g2[0]['submit'] == 20100. and [f for f in L_['fills'] if f[1] == 1] == [] and abs(g2[0]['ahead'] - 100.) < 1e-9)

    class ResetEngine(LB.LadderEngine):
        """A planted wrong queue: each live order's queue ahead reset to the latest shown size before every grid's trades."""

        def advance(self, grid):
            j = int(np.searchsorted(self.f[:, 0], grid, side='right')) - 1
            for o in self.o:
                a = self.shown(max(j, 0), o['side'], o['price'])
                if o['rem'] > 0 and o['act'] <= grid and np.isfinite(a):
                    o['ahead'] = a
            super().advance(grid)
    try:
        ok('2 long case: the partial fills after 5 s, priority kept, replacement behind the queue', long_ok(*long_case()))
        ok('2 long case: a resetting queue implementation is NOT rejected', not long_ok(*long_case(ResetEngine)))
        ok('2 long case: a truncated (5 s) implementation is NOT rejected', not long_ok(*long_case(config=LB.LEGACY)))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'2 long case: stopped ({e!r})'[:200])

    # ---- 3 ladder fixtures
    try:
        frames = [fr(0, 0, [(50, 20), (49, 30)], [(52, 100), (53, 100)]), fr(1500, 1500, [(49, 10)], [(52, 100), (53, 100)])]
        p = BE.tape(frames, [(1000, 0, 49, 160)], 50.); B = LB.LadderBranch(p, LB.LADDER, end=3000.); pol = LB.LadderInventory(B)
        LB.run_episode(B, pol, until=1); bids = [B.E.o[g] for g in B.unclosed[0]]
        ok('3 two bid levels at I = 0: 100 + 100 at 50 / 49 with birth rungs 0 / 1',
           sorted((o['price'], o['size'], o['rung']) for o in bids) == [(49., 100., 1.), (50., 100., 0.)] and len(B.unclosed[1]) == 2)
        LB.run_episode(B, pol, k0=1, until=21); fl = [(f[0], B.E.o[f[1]]['price'], f[3]) for f in B.E.fills if f[2] == 0]
        ok(f'3 one sweep fills both levels in price priority within the trade size ({fl})', fl == [(1000., 50., 100.), (1000., 49., 30.)] and sum(x[2] for x in fl) <= 160.)
        g49 = [g for g in range(len(B.E.o)) if B.E.o[g]['price'] == 49. and B.E.o[g]['side'] == 0][0]
        ok('3 the partial fill of one gid leaves the other open', g49 in B.unclosed[0] and B.closed.get(0, (None,))[0] in (None, 'filled'))
        v = {'side': 0, 'position': 40., 'B': 0., 'A': 0., 'mode': 'UNRESTRICTED'}; f_, D, Bs, room, q1 = pol.sizing(v); v2 = dict(v, B=q1); q2 = pol.sizing(v2)[4]
        ok(f'3 I = +40 gives two 60-share bids ({q1}, {q2})', (q1, q2) == (60., 60.))
        ok('3 sizes round down to whole shares (I = 33.3: 66, never above the budget)', pol.sizing(dict(v, position=33.3))[4] == 66.)
        ok('3 reservations above the desired exposure add nothing (exposure_full)', pol.slot(dict(v, B=130., A=0., position=40., side=0, mode='UNRESTRICTED'), 0, 50.)[2] == 'exposure_full')
        # the cap holds both levels and a cancel-pending order until its acknowledgement
        p = BE.tape([fr(0, 0, [(50, 0), (49, 0)], [(52, 0), (53, 0)])], [], 50.); B = LB.LadderBranch(p, LB.LADDER, end=3000.)
        LB.run_episode(B, Scripted(B, {(0, 0, 0), (0, 0, 1), (150, 0, 0), (150, 0, 1), (200, 0, 0), (200, 0, 1), (250, 0, 0), (250, 0, 1), (300, 0, 0), (300, 0, 1)},
                                   pulls={(100, 0)}), until=7)
        ok('3 a cancel-pending order holds its slot and reservation until its acknowledgement (no post at 150-200)',
           [o['submit'] for o in B.E.o if o['side'] == 0] == [0., 0., 250.] and B.closed.get(0, (None,))[0] == 'removed' and B.closed[0][1] == 250.)
        # one structural pull (cross) leaves the other order resting, with its subreason
        frames = [fr(0, 0, [(50, 0), (49, 0)], [(52, 0), (53, 0)]), fr(200, 200, [(48, 0)], [(50, 0), (51, 0)])]
        p = BE.tape(frames, [], 50.); B = LB.LadderBranch(p, LB.LADDER, end=3000.); LB.run_episode(B, Scripted(B, {(0, 0, 0), (0, 0, 1)}), until=6)
        sl = B.struct_log
        ok(f'3 one structural (cross) pull leaves the other bid resting ({sl})', len(sl) == 1 and sl[0]['primary'] == 'cross' and sl[0]['price'] == 50. and sl[0]['cross']
           and not sl[0]['no_book'] and any(B.E.o[g]['price'] == 49. and B.E.o[g]['req'] is None for g in B.unclosed[0]))
        # a pending price is never duplicated
        p = BE.tape([fr(0, 0, [(50, 0), (49, 0)], [(52, 0), (53, 0)])], [], 50.); B = LB.LadderBranch(p, LB.LADDER, end=3000.)
        LB.run_episode(B, Scripted(B, {(0, 0, 0), ('any', 0, 0), ('any', 0, 1)}), until=3)
        ok('3 a held price is never duplicated (the next tick fills the other level)', sorted(o['price'] for o in B.E.o if o['side'] == 0) == [49., 50.])
        # own cross: a new bid at or above our unclosed ask is refused
        frames = [fr(0, 0, [(50, 0)], [(52, 0)]), fr(200, 200, [(52, 0)], [(53, 0)])]
        p = BE.tape(frames, [], 50.); B = LB.LadderBranch(p, LB.LADDER, end=3000.)
        LB.run_episode(B, Scripted(B, {(0, 1, 0), ('any', 0, 0)}), until=8)
        ack = B.closed.get(1, (None, None))[1]
        ok('3 a bid crossing our own unclosed (cancel-pending) ask is own_cross_blocked until its acknowledgement',
           any(c[2] == 'own_cross_blocked' and c[0] < (ack or 0) for c in B.constraints) and ack is not None
           and all(o['submit'] >= ack for o in B.E.o if o['side'] == 0 and o['price'] >= 52.) and any(o['side'] == 0 and o['price'] == 52. for o in B.E.o))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'3 ladder fixtures: stopped ({e!r})'[:200])

    # ---- 4 hidden notices
    try:
        p = BE.tape([fr(0, 0, [(50, 0), (49, 0)], [(52, 0), (53, 0)])], [(1000, 0, 50, 40)], 50.); B = LB.LadderBranch(p, LB.LADDER, end=3000.)
        pol = LB.LadderInventory(B); LB.run_episode(B, pol, until=20); v19 = B.side_view(950., 0); Bc = copy.deepcopy(B)
        LB.run_episode(B, pol, k0=20, until=21); v20 = B.side_view(1000., 0)
        LB.run_episode(B, pol, k0=21, until=22); v21 = B.side_view(1050., 0); LB.run_episode(B, pol, k0=22, until=23); v22 = B.side_view(1100., 0)
        same = lambda a, b: all(a[k] == b[k] for k in ('position', 'B', 'A', 'mode')) and [o['known_filled'] for o in a['orders']] == [o['known_filled'] for o in b['orders']]
        ok('4 an unnotified fill changes no client view before its notice', same(v19, v20) and same(v19, v21) and not same(v19, v22) and v22['position'] == 40.)
        B2 = copy.deepcopy(Bc); B2.E.tr = np.array(B2.E.tr, float, copy=True); B2.E.tr[:, 3] = 10.
        LB.run_episode(Bc, pol, k0=20, until=22); LB.run_episode(B2, pol, k0=20, until=22)
        ok('4 a different unnotified fill gives the same commands before its notice', Bc.cmd_log == B2.cmd_log and Bc.decisions == B2.decisions)
        B3 = copy.deepcopy(B2); B3.E.f = np.array(B3.E.f, float, copy=True); B3.E.f[B3.E.known, 2] = 51.
        ok('4 the delivered control (the received frame changed) changes the view', B3.side_view(1050., 0)['bid'] != B2.side_view(1050., 0)['bid'])
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'4 hidden notices: stopped ({e!r})'[:200])

    # ---- 5 inventory mode
    try:
        tab = [('UNRESTRICTED', 0., 'UNRESTRICTED'), ('UNRESTRICTED', 99.999, 'UNRESTRICTED'), ('UNRESTRICTED', 100., 'LONG_RESTRICTED'), ('LONG_RESTRICTED', 60., 'LONG_RESTRICTED'),
               ('LONG_RESTRICTED', 50.000001, 'LONG_RESTRICTED'), ('LONG_RESTRICTED', 50., 'UNRESTRICTED'), ('LONG_RESTRICTED', -100., 'SHORT_RESTRICTED'),
               ('SHORT_RESTRICTED', -51., 'SHORT_RESTRICTED'), ('SHORT_RESTRICTED', -50., 'UNRESTRICTED'), ('SHORT_RESTRICTED', 100., 'LONG_RESTRICTED')]
        ok('5 next_mode boundaries and reversal', all(LB.next_mode(a, I, 100., 50.) == b for a, I, b in tab))
        frames = [fr(0, 0, [(50, 0), (49, 0)], [(52, 0), (53, 0)])]
        trades = [(1000, 0, 50, 100), (3000, 1, 52, 30), (3500, 1, 52, 70)]
        p = BE.tape(frames, trades, 50.); B = LB.LadderBranch(p, LB.LADDER, end=6000.); pol = LB.LadderInventory(B); LB.run_episode(B, pol)
        ml = B.mode_log; g49 = [o for o in B.E.o if o['side'] == 0 and o['price'] == 49.][0]
        ok(f'5 LONG at the notice of I = 100, recovery at I <= 50 ({ml})', ml[:1] == [(1100., 100., 'UNRESTRICTED', 'LONG_RESTRICTED')]
           and any(x[3] == 'UNRESTRICTED' and abs(x[1]) <= 50 for x in ml[1:]))
        ok('5 the live bid pulled at the LONG tick, effect +37, released at the acknowledgement', g49['req'] == 1100. and g49['why'] == 'policy' and g49['remove'] == 1137.
           and B.closed[g49['gid']] == ('removed', 1250.))
        t_rec = [x[0] for x in ml if x[3] == 'UNRESTRICTED'][0]
        ok('5 no new bid while restricted, bids resume after recovery', not any(o['side'] == 0 and 1100. <= o['submit'] < t_rec for o in B.E.o)
           and any(o['side'] == 0 and o['submit'] >= t_rec for o in B.E.o))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'5 inventory mode: stopped ({e!r})'[:200])

    # ---- 6 cancel timing
    try:
        res = []
        for tt in (1036., 1037.):
            p = BE.tape([fr(0, 0, [(50, 0)], [(52, 0)])], [(tt, 0, 50, 10)], 50.); B = LB.LadderBranch(p, LB.LADDER, end=3000.)
            LB.run_episode(B, Scripted(B, {(0, 0, 0)}, pulls={(1000, 0)})); res.append(sum(f[3] for f in B.E.fills))
        ok(f'6 a pull at 1000 lands at 1037: fills at 1036 / 1037 = {res}', res == [10., 0.])
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'6 cancel timing: stopped ({e!r})'[:200])

    # ---- 7 resume identity
    try:
        p = BE.rand_tape(np.random.default_rng(5), 300, 500, end=30000.); B = LB.LadderBranch(p, LB.LADDER, end=30000.); LB.run_episode(B, LB.LadderInventory(B)); Lc = LB.finish(B)
        Bh = LB.LadderBranch(p, LB.LADDER, end=30000.); LB.run_episode(Bh, LB.LadderInventory(Bh), until=300); sv = LB.save_state(Bh, 300)
        Br, k = LB.resume(sv, LB.LADDER, p, 30000.); LB.run_episode(Br, LB.LadderInventory(Br), k0=k); Lr = LB.finish(Br)
        ok('7 a resumed run equals the continuous one', LB.fingerprint(Br) == LB.fingerprint(B) and Lr['fills'] == Lc['fills'] and Br.mode_log == B.mode_log and Lr['value'] == Lc['value'])
        refused = 0
        for args in ((sv, LB.LEGACY, p, 5000.), (sv, LB.LADDER, p, 29000.), (sv, LB.LADDER, dict(p, mark=51.), 30000.)):
            try:
                LB.resume(*args)
            except ValueError:
                refused += 1
        sv2 = dict(sv); Bx = LB.pickle.loads(sv['blob']); Bx.mode_log = Bx.mode_log + [(1., 0., 'UNRESTRICTED', 'UNRESTRICTED')]; sv2['blob'] = LB.pickle.dumps(Bx)
        try:
            LB.resume(sv2, LB.LADDER, p, 30000.)
        except ValueError:
            refused += 1
        ok(f'7 another configuration / END / tape / a changed state refused ({refused} / 4)', refused == 4)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'7 resume: stopped ({e!r})'[:200])

    # ---- 8 observer and reconstruction
    try:
        p = BE.rand_tape(np.random.default_rng(9), 400, 700, end=30000.); ident = dict(day='20261024', instrument_id=1, end_rel=30000.)
        Bo, rec, pol, obs = M.run_ladder(p, dict(day='20261024', m=1), ident); Bn = LB.LadderBranch(p, LB.LADDER, end=30000., ident=ident); LB.run_episode(Bn, LB.LadderInventory(Bn))
        LB.finish(Bn)
        ok('8 the observer does not change the run', LB.fingerprint(Bo) == LB.fingerprint(Bn))
        arr = dict(obs.arrays(), **M.command_arrays(Bo)); sn = M.all_snaps(pol, rec)
        ok('8 every snapshot reconstructs from the saved events', len(sn) > 50 and all(M.reconstruct_ok(rec, arr['commands'], s_) for s_ in sn))
        live = [s_ for s_ in sn if s_['kind'] == 'live' and s_['position'] != 0.]
        ok('8 a corrupted snapshot does not reconstruct', bool(live) and not M.reconstruct_ok(rec, arr['commands'], dict(live[0], position=live[0]['position'] + 1.)))
        Bq = LB.LadderBranch(p, LB.LADDER, end=30000.); st_bad = 0; o2 = M.Observer(Bq); pq = LB.LadderInventory(Bq)
        for k in range(LB.n_ticks(Bq)):
            LB.step(Bq, k, pq); o2(Bq, k); g = k * M.STEP
            for s in (0, 1):
                n_ = {z: sum(1 for gid in Bq.unclosed[s] if Bq.gid_state(gid, g) == z) for z in ('post_pending', 'live', 'cancel_pending')}
                st_bad += int((o2.npp[k, s], o2.nlive[k, s], o2.ncp[k, s]) != (n_['post_pending'], n_['live'], n_['cancel_pending']))
        ok(f'8 observer counts equal the client states ({st_bad})', st_bad == 0)
        ok('8 a venue-active order is client-live or cancel-pending', bool((o2.nva <= o2.nlive + o2.ncp).all()))
        cm = arr['commands']; lv = {}
        for c in cm:
            if int(c[2]) == M.CMD_KINDS.index('live') and int(c[3]) >= 0:
                lv.setdefault((int(c[1]), int(c[3])), []).append(int(round(c[0])))
        want = {kk: min(ts, key=lambda t_: M.rank('live', dict(day='20261024', m=1, side=kk[0], gid=kk[1], t=t_))) for kk, ts in lv.items()}
        got = {(s_['side'], s_['gid']): int(round(s_['t'])) for s_ in sn if s_['kind'] == 'live'}
        ok('8 each live snapshot is its generation\'s minimum-rank live decision, with its eligible count', got == want
           and all(s_['n_eligible'] == len(lv[(s_['side'], s_['gid'])]) for s_ in sn if s_['kind'] == 'live'))
        Bz, recz, polz, obsz = M.run_ladder(BE.tape([fr(0, 0, [(50, 0)], [(52, 0)])], [], 50.), dict(day='20261024', m=2), dict(end_rel=50.))
        ok('8 a POST that cannot activate before END records no entry', [o for o in recz['orders']] == [] and polz.entries == {}
           and any(c[2] == 'near_end' for c in Bz.constraints))
        ent = [s_ for s_ in sn if s_['kind'] == 'entry']
        ok('8 every generation has exactly one entry snapshot', sorted(s_['gid'] for s_ in ent if s_['gid'] is not None) == sorted(o['gid'] for o in rec['orders'])
           and all(s_['gid'] is not None for s_ in ent))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'8 observer: stopped ({e!r})'[:200])

    # ---- 9 features on the synthetic world
    try:
        import pf_r2_labels_selftest  # noqa: F401  (the synthetic world's helpers are importable)
        wj = json.loads((Path(os.environ['PF_TEACHER_SYN']) / 'world.json').read_text()); day = '20261024'; L, R2, X = M.L, M.R2, M.X
        W = L.worlds_of_day(day, Path(wj['kp']), 3); _, prs, fair, depth, ff = L.world_inputs(day, W, R2.KP.Syn(), X.syn_inputs, X.syn_features()); checked = 0; nks = 0
        for w in W:
            wb, st = L.world_bind(day, dict(w, t_ms=int(round(w['t']))), prs, fair, depth, ff)
            if wb is None:
                continue
            lc = M.legacy_check(day, dict(w, t_ms=int(round(w['t']))), wb, ff); checked += 1
            ok(f'9 legacy_check engine parity on a synthetic world ({lc["engine_equal"]})', lc['engine_all'])
            ok(f'9 the 107 inputs equal Teacher.inputs ({lc["features_equal"]} equal, {lc["features_bad"]} bad: {lc["bad_cols"]})', lc['features_bad'] == 0 and lc['features_equal'] > 0
               and lc['points_unmatched'] == 0)
            for s in (0, 1):
                o = w['o'][s]; t0 = float(o.t); ws_ = wb['W'][s]; d = t0 + np.array([100., 1000., 3000.])
                a = X.known_steps(ws_, s, t0 + 58., float(o.p), (np.array([t0 + 500.]), np.array([5.])), d, t0 + 5000.)
                b = M.known_steps_q(ws_, s, t0 + 58., float(o.p), (np.array([t0 + 500.]), np.array([5.])), d, t0 + 5000., 30., 5000., rung=0.)
                nks += int(all(np.array_equal(np.asarray(a[k_], float), np.asarray(b[k_], float), equal_nan=True) for k_ in a))
        ok(f'9 synthetic worlds checked ({checked}); known_steps_q = known_steps ({nks})', checked >= 1 and nks == 2 * checked)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'9 features: stopped ({e!r})'[:200])

    # ---- 10 measurement, population, guards
    try:
        p = BE.rand_tape(np.random.default_rng(21), 500, 900, end=60000.); ident = dict(day='20261024', instrument_id=7, end_rel=60000.)
        B, rec, pol, obs = M.run_ladder(p, dict(day='20261024', m=7), ident); arr = dict(obs.arrays(), **M.command_arrays(B))
        rec['queue'] = {}; bm = {str(j): dict(t_rel=j * 12000., mark=50. + j, status='eligible') for j in range(1, 5)}
        r = dict(day='20261024', m=7, hour=0, ws=0., status='complete', ledger=rec, path_info=dict(t_start=0., ws=0., end_rel=60000., boundary_marks=bm))
        arrs = {f"20261024|7__{k}": v for k, v in arr.items()}
        G = M.gen_rows(r, arrs); F = M.fill_rows(r); Mi, sk = M.minute_rows(r)
        val = float(rec['value']); cashpos = float(rec['cash']) + float(rec['position']) * float(rec['mark'])
        ok('10 generation and fill values sum to cash + inventory x mark', abs(sum(g['value'] for g in G) - val) < 1e-6 and abs(sum(f['value'] for f in F) - val) < 1e-6
           and abs(val - cashpos) < 1e-6)
        ok('10 minute increments telescope and split into carry + new fills', abs(sum(x['dv'] for x in Mi) - val) < 1e-6 and all(abs(x['dv'] - x['carry'] - x['new']) < 1e-6 for x in Mi)
           and len(Mi) == 5 and sk == [])
        mv = [f for f in F if f['edge'] is not None and f['mo1000'] is not None]
        ok('10 markout = edge + movement on the valid fills', bool(mv) and all(abs(f['mo1000'] - f['edge'] - f['mv1000']) < 1e-9 for f in mv))
        ok('10 markouts never past END', all(f['mo30000'] is None for f in F if f['t'] + 30000. > 60000.))
        fa = np.asarray(p['frames'], float)
        ok('10 the fill-time mid is the last frame at or before the fill', all((f['mid0'] is None and M.AR.mid_at(fa, f['t']) is None) or f['mid0'] == M.AR.mid_at(fa, f['t']) for f in F))
        hand = dict(day='20261024', m=8, hour=0, ws=0., status='complete', path_info=dict(t_start=0., ws=0., end_rel=1000., boundary_marks={}),
                    ledger=dict(mark=52., end=1000., value=0., fills=[[136., 0, 0, 1., 50., 0], [137., 0, 0, 1., 50., 1]], fill_mids=[[51., 0.] + [None] * 10] * 2,
                                orders=[dict(gid=0, side=0, rung=0., price=50., size=10., act=58., remove=1e30, submit=0., refused=False, req=None, why=None)],
                                closed={}, requests=[], struct_log=[], ahead0={'0': 3.}, queue={}))
        gh = M.gen_rows(hand, {'20261024|8__commands': np.zeros((0, 9))})[0]
        ok(f'10 the earliest cancel effect is the first tick at or after activation + 37: locked {gh["value_locked"]}, later {gh["value_later"]}',
           gh['earliest_cancel_effect'] == 137. and gh['value_locked'] == 2. and gh['value_later'] == 2.)
        ok('10 downtime precedence', M.downtime_of({'restricted', 'activation_latency'}) == 'restricted' and M.downtime_of({'awaiting_notice', 'cap'}) == 'cap'
           and M.downtime_of({'invalid_book', 'restricted'}) == 'invalid_book' and M.downtime_of(set()) == 'other')
        ws_ = 1792800000000.; tg = ws_ + 50. * np.arange(10)
        Wm = {0: dict(tg=tg, jk=np.array([-1, -1, 0, 0, 1, 1, 1, 1, 1, 1]), P5=np.array([[50.] + [-1.] * 4, [51.] + [-1.] * 4]), kf=np.array([ws_ + 90., ws_ + 180.])),
              1: dict(tg=tg, jk=np.array([-1, -1, 0, 0, 1, 1, 1, 1, 1, 1]), P5=np.array([[49.] + [-1.] * 4, [53.] + [-1.] * 4]), kf=np.array([ws_ + 90., ws_ + 180.]))}
        st0 = M.market_start(Wm, ws_)
        ok(f'10 the opening waits for a received valid book ({st0})', st0 == (ws_ + 200., 2, 2))
        Wf = {0: dict(tg=tg, kf=np.array([ws_ - 10.]), P5=np.array([[50., 49., -1., -1., -1.]]), S5=np.array([[20., 30., 0., 0., 0.]])),
              1: dict(tg=tg, kf=np.array([ws_ - 10.]), P5=np.array([[52., 53., -1., -1., -1.]]), S5=np.array([[20., 30., 0., 0., 0.]]))}
        xm = np.zeros(len(M.FC.MARKET), np.float32); xm[M.FC.MARKET.index('pf_mid')] = 51.
        ctx = M.ladder_ctx(LB.LadderBranch(BE.tape([fr(0, 0, [(50, 0)], [(52, 0)])], [], 50.), LB.LADDER, end=1000.),
                           dict(side=0, t=100., position=0., B=0., A=0.), None)
        tgl = ws_ + 50. * np.arange(4000); Wl = {s_: dict(Wf[s_], tg=tgl) for s_ in (0, 1)}
        row, st_ = M.features_at(Wl, 0, ws_, dict(t=130000., act=130058., price=49., q=37., rung=1., ctx=ctx), 298000., ws_, {2600: (xm, 10.)}, [])
        ok(f'10 full-market features: fair_edge unavailable, t_to_end, the posted size, the phase at the decision ({st_})', row is not None and np.isnan(row['fair_edge'])
           and row['t_to_end'] == 168000. and row['fp_size'] == 37. and row['fp_phase'] == 3. and row['fp_queue'] == 30.)
        why, flags = M.tick_reasons(r, arrs); P_ = M.part_rows(r, arrs); I_ = M.inventory_row(r, arrs); E_ = M.equity_row(r, arrs)
        ok('10 downtime reasons are declared and quoting ticks have none', set(why.ravel()) <= set(M.DOWNTIME) | {'quoting'}
           and all(why[k, s] == 'quoting' for k in range(why.shape[0]) for s in (0, 1) if arr['nva'][k, s] > 0))
        pd = M.pd; Mk = pd.DataFrame([dict(day='20261024', m=7, status='complete')])
        Sm = M.summaries(Mk, pd.DataFrame(G), pd.DataFrame(F), pd.DataFrame([I_]), pd.DataFrame(P_), pd.DataFrame([E_]), pd.DataFrame(M.replacement_rows(r, G)))
        row = Sm[(Sm.day == 'all') & (Sm.side == 'both') & (Sm.rung.astype(str) == 'all')].iloc[0]
        sides = Sm[(Sm.day == 'all') & (Sm.side.isin(['bid', 'ask'])) & (Sm.rung.astype(str) == 'all')]
        ok('10 the summary value equals the market value and the sides add up', abs(float(row['value']) - val) < 1e-6 and abs(float(sides.value.sum()) - val) < 1e-6)
        rr = Sm[(Sm.day == 'all') & (Sm.side == 'both') & (Sm.rung.astype(str).isin(['0', '1']))]
        ok('10 the birth-rung rows add up', abs(float(rr.value.sum()) - val) < 1e-6)
        pr = M.predicates(Sm, pd.DataFrame(Mi))
        ok('10 predicates on one day are INCOMPLETE (four days declared)', pr['strategy_value']['answer'] == 'INCOMPLETE' and pr['complete_days'] is False)
        d0 = 1792800000000.
        cells, sel = M.select_markets({'20261024': [3, 1, 2], '20261025': [4]}, {'20261024': {1: (d0 + 3600e3, 2), 2: (d0 + 3600e3 + 1, 2), 3: (d0 + 7200e3, 2)},
                                                                                  '20261025': {4: (d0 + 86400e3, 2)}, '20261026': {}, '20261027': {}})
        h1 = sorted([1, 2], key=lambda m: (M.rank('', dict(day='20261024', m=m)), M.LS.ckey(dict(day='20261024', m=m))))[0]
        ok('10 select_markets: one per day x opening hour by rank', [(x['day'], x['m']) for x in sel] == sorted([('20261024', h1), ('20261024', 3), ('20261025', 4)])
           and [c['candidates'] for c in cells] == [2, 1, 1])
        with tempfile.TemporaryDirectory(dir=str(M.KP.I.K.IO)) as tmp:
            par = dict(sets=dict(train=dict(forks=[dict(day='20261024', m=1, t_ms=10, side=0), dict(day='20261024', m=1, t_ms=20, side=1),
                                                   dict(day='20261024', m=2, t_ms=40, side=0), dict(day='20261024', m=3, t_ms=50, side=0)]
                                         + [dict(day=d, m=10 + i, t_ms=30, side=0) for i, d in enumerate(M.DAYS[1:])])))
            pp = Path(tmp) / 'par.json'; pp.write_text(json.dumps(par)); mp = Path(tmp) / 'man.json'
            plans = dict(train=dict(kp_dir='x', sample={d: dict(inputs=dict(day=d)) for d in M.DAYS}))
            regf = lambda d: ({1: (d0, 2), 2: (d0 + 3600e3, 2), 3: (d0 + 7200e3, 2)} if d == '20261024' else {10 + M.DAYS.index(d) - 1: (d0 + 86400e3 * M.DAYS.index(d), 2)}, 'rsha', 'msha')
            man = M.build_manifest(mp, pp, regf, plans, identity=False)
            ok('10 build_manifest: the frozen population', [x['m'] for x in man['selected']] == [1, 2, 3, 10, 11, 12] and man['selected'][0]['parent_worlds'] == [10, 20]
               and man['prefix'] == M.PREFIX and man['pilot']['20261024'] == [1, 2] and man['pilot']['20261025'] == [10])
            try:
                M.build_manifest(mp, pp, regf, plans, identity=False); bad.append('10 build_manifest overwrote its output')
            except ValueError:
                pass
            def raiser():
                raise RuntimeError('the binder was reached: a guard did not refuse')
            try:
                M.run_day('20261024', pilot=True, binder=raiser, man_path=mp, out=Path(tmp)); bad.append('10 run_day ran before the freeze')
            except ValueError as e:
                ok(f'10 run_day refuses before the freeze for the freeze reason ({e})', 'freeze' in str(e))
            fz = M.frozen; M.frozen = lambda: True
            try:
                M.run_day('20261024', pilot=False, binder=raiser, man_path=mp, out=Path(tmp)); bad.append('10 the full run ran without the projection')
            except ValueError as e:
                ok(f'10 the full run refuses without the pilot projection ({e})', 'projection' in str(e))
            finally:
                M.frozen = fz
            md = Path(tmp) / 'measure'; md.mkdir(); (md / 'audit.json').write_text('{}')
            try:
                M.measure(out=Path(tmp)); bad.append('10 measure overwrote an existing measurement')
            except ValueError as e:
                ok(f'10 measure refuses an existing measurement ({e})', 'write-once' in str(e))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'10 measurement: stopped ({e!r})'[:220])
    if bad:
        raise SystemExit(FAIL + ': ' + ' | '.join(bad))
    print('LADDER SELFTEST PASSED: legacy reproduction, the long queue case and its two planted implementations, ladder fixtures, hidden notices, '
          'inventory mode, cancel timing, resume, observer / reconstruction, features, measurement, population and guards')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pf_ladder_rest as M_
    main(M_)
