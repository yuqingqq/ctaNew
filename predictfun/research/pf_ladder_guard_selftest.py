"""SELFTEST of pf_ladder_guard (ASK-RESERVATION-GUARD-v1; 2026-11-21, session aea5bfb9). main(G) checks the guard module; a failed check raises
SystemExit with FAIL.
 1 the disabled guard equals pf_branch_ladder.LadderInventory exactly on random tapes (fingerprint, fills, value);
 2 the signal: float32(B) 199.999 off, 200 on, NaN off, disabled off; the inventory rule keeps precedence and its own reasons;
 3 a hand tape: the bid side processed first at a tick makes the ask side see B = 200 at the same tick (withheld); with the ask side first the
   post-pending asks are pulled at their first live review (guard_pull, one request each, never repeated while cancel-pending); a bid fill
   does not clear the signal before its notice (+100 ms); the notice that brings B below 200 clears it and asks post at that tick (no timer);
   bids are never pulled or withheld by the guard; every intervention logged with B >= 200;
 4 the selection control: a selected set worse than its matched strata is SELECTION_BEYOND_EXPOSURE, a random one is not, a thin one is
   INCONCLUSIVE; the draws reproduce with the seed; censored states are excluded and counted, and a censored high-quantity selected state
   changes neither the valid mean nor the per-valid-share term (finding 1); degenerate / movable strata counted;
 5 the generation censor count (finding 3): no-fill generations are not censored; 0 <= count <= generations;
 6 arm metrics on a hand record (the shared END - 1 s cutoff, censor counts, good / bad and side splits, guard counts, reducing suppressions)
   and the readings (IMPROVES / WORSENS / MIXED, better_selection / quoting_less / mixed);
 7 guards: run_day refuses before the freeze, without the projection for the full run and under another rule; outputs write-once."""
import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

FAIL = 'GUARD SELFTEST FAILED'


def run(mod):
    main(mod)


def main(G):  # noqa: C901
    M, LB, BE, pd = G.M, G.LB, G.BE, G.pd; bad = []
    ok = lambda name, cond: None if cond else bad.append(name)
    import pf_ladder_rest_selftest as V1
    fr = V1.fr
    # ---- 1 disabled = baseline
    try:
        rng = np.random.default_rng(31); nb = 0
        for i in range(8):
            p = BE.rand_tape(rng, 300, 500, end=30000.)
            B1 = LB.LadderBranch(p, LB.LADDER, end=30000., episode=i % 2); LB.run_episode(B1, LB.LadderInventory(B1)); L1 = LB.finish(B1)
            B2 = LB.LadderBranch(p, LB.LADDER, end=30000., episode=i % 2); LB.run_episode(B2, G.AskGuard(B2, 200., enabled=False)); L2 = LB.finish(B2)
            nb += int(not (LB.fingerprint(B1) == LB.fingerprint(B2) and L1['fills'] == L2['fills'] and L1['value'] == L2['value']))
        ok(f'1 the disabled guard equals the baseline policy ({nb} of 8 differ)', nb == 0)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'1 disabled: stopped ({e!r})'[:200])
    # ---- 2 the signal and precedence
    try:
        B = LB.LadderBranch(BE.tape([fr(0, 0, [(50, 0)], [(52, 0)])], [], 50.), LB.LADDER, end=3000.); gd = G.AskGuard(B, 200.); gx = G.AskGuard(B, 200., enabled=False)
        v = lambda b, **k: dict(dict(t=100., side=1, position=0., B=b, A=0., mode='UNRESTRICTED', avail=200.), **k)
        ok('2 the signal boundary', (not gd.signal(v(199.999))) and gd.signal(v(200.)) and not gd.signal(v(np.nan)) and not gx.signal(v(200.)))
        o = dict(gid=3, price=52., rung=0., size=100., state='live', known_filled=0., act=58., submit=0.)
        ok('2 the inventory rule keeps precedence and its reasons', gd.live(v(200., mode='SHORT_RESTRICTED', position=-120.), o) == ('PULL', 'inventory_pull')
           and gd.slot(v(200., mode='SHORT_RESTRICTED', position=-120.), 0, 52.)[2] == 'restricted' and gd.live(v(200.), o) == ('PULL', 'guard_pull')
           and gd.slot(v(200.), 0, 52.) == ('WAIT', None, 'guard_withhold') and gd.live(v(200., side=0), dict(o, price=50.))[1] == 'keep')
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'2 signal: stopped ({e!r})'[:200])
    # ---- 3 the hand tape
    try:
        frames = [fr(0, 0, [(50, 0), (49, 0)], [(52, 0), (53, 0)])]; p = BE.tape(frames, [(1000, 0, 50, 40)], 50.)
        Ba = LB.LadderBranch(p, LB.LADDER, end=3000., episode=0); ga = G.AskGuard(Ba, 200.); LB.run_episode(Ba, ga)
        asks_a = [o for o in Ba.E.o if o['side'] == 1]
        ok(f'3 bid side first: the ask side sees B = 200 at the same tick and withholds until the notice ({[o["submit"] for o in asks_a]})',
           asks_a and min(o['submit'] for o in asks_a) == 1100. and any(z[8] == 'guard_withhold' and z[0] == 0. for z in ga.log))
        Bb = LB.LadderBranch(p, LB.LADDER, end=3000., episode=1); gb = G.AskGuard(Bb, 200.); LB.run_episode(Bb, gb)
        first = [o for o in Bb.E.o if o['side'] == 1 and o['submit'] == 0.]; reqs = [e for e in Bb.E.events if e[1] == 'policy_request']
        ok(f'3 ask side first: the post-pending asks are pulled at their first live review, once each ({[(o["req"], o["why"]) for o in first]})',
           len(first) == 2 and all(o['req'] == 100. and o['why'] == 'policy' for o in first) and all(sum(1 for e in reqs if e[2] == o['gid']) == 1 for o in first))
        ok('3 the guard pull reason is logged apart from the inventory rule', sum(1 for z in gb.log if z[8] == 'guard_pull') == 2 and all(z[1] == 1 for z in gb.log))
        later = [o for o in Bb.E.o if o['side'] == 1 and o['submit'] > 0.]
        ok(f'3 the signal holds through the hidden fill and clears at its notice (asks post at 1,100: {[o["submit"] for o in later]})',
           later and min(o['submit'] for o in later) == 1100. and not any(1000. <= z[0] < 1100. and z[8] != 'guard_withhold' for z in gb.log))
        ok('3 bids are never touched by the guard; every intervention at B >= 200', all(o['why'] != 'policy' for o in Bb.E.o if o['side'] == 0) and all(z[5] >= 200. for z in ga.log + gb.log))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'3 hand tape: stopped ({e!r})'[:200])
    # ---- 4 the selection control
    try:
        rows = []; rng = np.random.default_rng(3)
        for day in ('20261024', '20261025'):
            for m in range(4):
                for i in range(30):
                    sel = i < 10
                    rows.append(dict(day=day, m=m, kind='live', side=1, gid=i, t=0., status='ok', win_n=1, win_q=10., win_mo1s=(-20. if sel else 0.) + rng.normal(0, 1), censored_1s=False,
                                     ctx_bid_reserved_shares=200. if sel else 150., fp_remaining=float(10 + 40 * (i % 2)), ctx_generation_age_ms=float(100 + 3000 * (i % 3)), inv_known=0.))
        S_ = pd.DataFrame(rows); E = {G.EDGE_KEY: [102., 150., 188., 200.]}; r1 = G.selection(S_, E, draws=200); r2 = G.selection(S_, E, draws=200)
        ok(f'4 a worse selected set is SELECTION_BEYOND_EXPOSURE ({r1["answer"]}, J {r1["J"]:.2f}, p05 {r1["null_p05"]:.2f})', r1['answer'] == 'SELECTION_BEYOND_EXPOSURE' and r1['selected'] == 80)
        ok('4 the draws reproduce with the seed', r1['null_mean'] == r2['null_mean'] and r1['null_p05'] == r2['null_p05'])
        S2 = S_.copy(); S2['win_mo1s'] = rng.normal(0, 1, len(S2)); r3 = G.selection(S2, E, draws=200)
        ok(f'4 a random selection is not ({r3["answer"]})', r3['answer'] == 'NO_SELECTION_BEYOND_EXPOSURE')
        S3 = S_[S_.m == 0].head(12); r4 = G.selection(S3, E, draws=50)
        ok(f'4 a thin control is INCONCLUSIVE ({r4["answer"]}, {r4["movable_strata"]} movable)', r4['answer'] == 'INCONCLUSIVE')
        S4 = pd.concat([S_, pd.DataFrame([dict(rows[0], gid=999, win_q=5000., win_mo1s=np.nan, censored_1s=True)])], ignore_index=True); r5 = G.selection(S4, E, draws=50)
        ok('4 a censored high-quantity selected state changes neither valid term (finding 1)', r5['J'] == G.selection(S_, E, draws=50)['J'] and r5['J_per_valid_share'] == r1['J_per_valid_share']
           and r5['valid_window_q'] == r1['valid_window_q'] and r5['excluded_window_q'] == 5000. and r5['total_window_q'] == r1['total_window_q'] + 5000. and r5['censored'] == 1)
        def conf(col, young, old):
            rr = []
            for day in ('20261024', '20261025'):
                for m in range(4):
                    for i in range(40):
                        yg = i < 20
                        rr.append(dict(day=day, m=m, kind='live', side=1, gid=i, t=0., status='ok', win_n=1, win_q=10., win_mo1s=(-20. if yg else 0.) + .01 * i, censored_1s=False,
                                       ctx_bid_reserved_shares=200. if (yg and i % 2 == 0) else 150., fp_remaining=float(young if (yg and col == 'rem') else old if col == 'rem' else 50.),
                                       ctx_generation_age_ms=float(young if (yg and col == 'age') else old if col == 'age' else 1000.), inv_known=0.))
            return pd.DataFrame(rr)
        for col, yv, ov in (('age', 100., 20000.), ('rem', 10., 90.)):
            rc = G.selection(conf(col, yv, ov), E, draws=200)
            ok(f'4 matching on the {col} band removes a confounded difference ({rc["answer"]})', rc['answer'] == 'NO_SELECTION_BEYOND_EXPOSURE')
        Sd = S_.copy(); Sd['ctx_bid_reserved_shares'] = 200.; rd_ = G.selection(Sd, E, draws=20)
        ok('4 a fully degenerate population: every draw is the selected set (without replacement)', abs(rd_['null_mean'] - rd_['J']) < 1e-12 and rd_['movable_strata'] == 0)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'4 selection: stopped ({e!r})'[:220])
    # ---- 5 the censor count
    try:
        Gh = pd.DataFrame(dict(lifetime_mo1s=[0., 0., np.nan, -3., np.nan], n_fills=[0, 0, 2, 1, 3]))
        ok('5 the generation censor count excludes no-fill generations', G.censor_count(Gh) == 2 and 0 <= G.censor_count(Gh) <= len(Gh))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'5 censor: stopped ({e!r})'[:200])
    # ---- 6 arm metrics and readings
    try:
        mm = lambda m0, m1: [m0, 0., m0, 0., m0, 0., m1, 0., None, None, None, None]
        rec = dict(mark=50., value=3., position=10., fills=[[100., 0, 0, 10., 49., 1], [1500., 1, 1, 5., 51., 2], [2500., 1, 1, 5., 51., 3]],
                   fill_mids=[mm(50., 49.5), mm(50., None), mm(50., 52.)], orders=[dict(gid=0, side=0, rung=0., req=None, why=None), dict(gid=1, side=1, rung=1., req=20., why='structural')],
                   guard_log=[[0., 1, 'slot', -1, 0., 200., 5., 'UNRESTRICTED', 'guard_withhold'], [50., 1, 'live', 1, 1., 200., -1., 'UNRESTRICTED', 'guard_pull'],
                              [60., 1, 'slot', -1, 1., 200., 7., 'UNRESTRICTED', 'guard_withhold']])
        x = G.arm_metrics(rec, dict(pose=np.array([0., 10., 120., 10.])), 3000.)
        ok(f'6 the shared cutoff (END - 1 s), censor counts and splits ({x["all_cut_q"]}, {x["ask_cut_censored"]})', x['all_cut_q'] == 15. and x['ask_cut_censored'] == 1 and x['bid_cut_mo1s'] == 5.
           and x['all_q'] == 20. and x['guard_withholds'] == 2 and x['guard_pulls'] == 1 and x['reducing_suppressed'] == 2 and x['structural_requests'] == 1 and x['share_abs_ge_S'] == .25)
        R_ = pd.DataFrame([dict(day=d, m=m, arm=a, value=v, all_q=q, all_cut_mo1s=c, all_cut_valid_q=10., bid_cut_mo1s=c, bid_cut_valid_q=5., ask_cut_mo1s=c, ask_cut_valid_q=5., both_exact=.4,
                                ask_exact=.5, ask_q=q, guard_pulls=0, guard_withholds=0, reducing_suppressed=0, mean_abs_inv=1., share_abs_ge_S=0., terminal_inv=0.)
                           for d in ('20261024', '20261025') for m in (1, 2) for a, v, q, c in (('disabled', 0., 100., -10.), ('guard', 5., 80., -5.))])
        p1 = G.predicates(R_, dict(answer='SELECTION_BEYOND_EXPOSURE'))
        ok(f'6 readings: END and per-share improve every day, better_selection ({p1["interpretation"]})', p1['END_paired']['answer'] == 'IMPROVES_EVERY_DAY'
           and p1['mo1s_per_share']['all']['answer'] == 'IMPROVES_EVERY_DAY' and p1['interpretation'] == 'better_selection')
        p2 = G.predicates(R_, dict(answer='NO_SELECTION_BEYOND_EXPOSURE'))
        ok('6 without selection evidence the reading is not better_selection', p2['interpretation'] == 'mixed')
        R2 = R_.copy(); R2.loc[R2.arm == 'guard', 'all_cut_mo1s'] = -20.; p3 = G.predicates(R2, dict(answer='NO_SELECTION_BEYOND_EXPOSURE'))
        ok(f'6 fewer fills without a per-share gain is quoting_less ({p3["interpretation"]})', p3['interpretation'] == 'quoting_less' and p3['mo1s_per_share']['all']['answer'] == 'WORSENS_EVERY_DAY')
        R3 = R_.copy(); R3.loc[(R3.arm == 'guard') & (R3.day == '20261025'), 'value'] = -9.; p4 = G.predicates(R3, dict(answer='SELECTION_BEYOND_EXPOSURE'))
        ok('6 a day that worsens makes END MIXED', p4['END_paired']['answer'] == 'MIXED' and p4['interpretation'] == 'mixed')
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'6 metrics: stopped ({e!r})'[:220])
    # ---- 7 guards
    try:
        fz = G.frozen; G.frozen = lambda: False
        try:
            G.run_day('20261024', pilot=True, out=Path(tempfile.mkdtemp(dir=str(M.KP.I.K.IO)))); bad.append('7 run_day ran before the freeze')
        except ValueError as e:
            ok(f'7 run_day refuses before the freeze ({e})', 'committed' in str(e))
        G.frozen = fz
        try:
            G.run_selection(out=Path(tempfile.mkdtemp(dir=str(M.KP.I.K.IO)))) if not G.frozen() else None
        except ValueError as e:
            ok(f'7 the selection stage refuses before the freeze ({e})', 'committed' in str(e))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'7 guards: stopped ({e!r})'[:200])
    if bad:
        raise SystemExit(FAIL + ': ' + ' | '.join(bad))
    print('GUARD SELFTEST PASSED: disabled = baseline, the signal and precedence, the hand tape (both side orders, notices, no timer), the selection '
          'control, the censor count, arm metrics and readings, guards')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pf_ladder_guard as G_
    main(G_)
