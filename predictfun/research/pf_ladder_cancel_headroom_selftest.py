"""SELFTEST of pf_ladder_cancel_headroom (FEASIBLE-CANCEL-HEADROOM-v1; 2026-11-22, session aea5bfb9). main(X) checks the module on synthetic records only
(no data is read); a failed check raises SystemExit with FAIL. Groups (review 3e84ca6d's list and the USER amendment 6d200317's hand cases):
 1 the cut clock: a fill 0.1 ms before the effect kept, at the effect deleted; the request tick kept; before the first review never deleted;
   partial / same-time fills indivisible;
 2 the oracle: good-then-bad cuts the bad suffix; bad-then-good with a net loss cuts both, with a net gain keeps; KEEP wins exact and
   within-tolerance ties; the earliest of equal (within-tolerance) cuts; unvalued fills count 0;
 3 the suffix classes and their precedence / tolerance edges, scalar and vectorized (panel) identical;
 4 generation(): end causes (refused, filled out, recorded cancellation, END-open, unacknowledged) and a closure disagreement refused; the
   queue fields (unknown -> NaN); a same-price NEW generation after a request; no feasible callback; the anchors' parts reconcile;
 5 sequences: bad-then-good with net gain / net loss / no feasible cancel before the bad fill; good-then-bad with and without a feasible
   intervening effect; simultaneous good / bad; interleaved; a missing later mark (incomplete, never pure);
 6 fills_of (+1 s eligibility at END - 1 s, censored 5 s / 30 s, END value, two gids at one price kept apart), keeps_of (live KEEP only,
   fallback flags), panel_rows = the scalar suffix per callback, the timeline;
 7 tables: reproduce and reconcile refuse a mismatch; summarize totals, class loss shares, concentration and readings; sequence counts;
   the class table; the casebook's outcome-independent key;
 8 guards: run refuses before the freeze, under another source, without a fitting projection and over an existing audit; manifest and
   projection are write-once."""
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

FAIL = 'CANCEL HEADROOM SELFTEST FAILED'
NAN = float('nan')


def run(mod):
    main(mod)


def F_(rows):
    """Synthetic fills [(t, v1, q=10, thr=True, v5=v1, v30=v1, vend=v1)] -> the module's per-gid arrays (sorted by t, stable)."""
    rows = sorted([tuple(r) + (10., True, None, None, None)[len(r) - 2:] for r in rows], key=lambda r: r[0])
    g = lambda i, d: np.array([(r[i] if r[i] is not None else (r[1] if d else NAN)) for r in rows], float)
    return dict(t=g(0, 0), v1=g(1, 0), q=g(2, 0), thr=np.array([bool(r[3]) for r in rows], bool), v5=g(4, 1), v30=g(5, 1),
                vend=np.array([r[6] if r[6] is not None else (0. if r[1] is None or math.isnan(r[1]) else r[1]) for r in rows], float))


def O_(gid, side=0, price=50., size=100., act=10., submit=0., req=None, remove=1e18, rem=100., refused=False, why=None, rung=0.):
    return dict(gid=gid, side=side, price=price, size=size, act=act, submit=submit, req=req, remove=remove, rem=rem, refused=refused, why=why, rung=rung, ahead_final=0.)


def REC_(orders, closed=None, open_gid=None, ahead0=None, first=None, pull=None, known=None):
    return dict(orders=orders, closed=closed or {}, open_at_end_gid=open_gid or {}, queue_venue=dict(ahead0=ahead0 or {}, first=first or {}, pull=pull or {}), queue_known=known or {})


def main(X):  # noqa: C901
    import pandas as pd
    bad = []
    ok = lambda name, cond: None if cond else bad.append(name)
    K = lambda *ts: np.array(ts, float)

    def refuses(word, fn, *a, **kw):
        try:
            fn(*a, **kw); return False
        except ValueError as e2:
            return word in str(e2)
    # ---- 1 the cut clock
    try:
        f1 = F_([(500., 2.), (1000., -1.), (1009.9, -1.), (1010., -5.), (1020., -4.)])
        s1 = X.part(f1, X.suffix_mask(f1, 1000.)); p1 = X.part(f1, ~X.suffix_mask(f1, 1000.))
        ok(f'1 the effect boundary ({s1["n"]}, {s1["L"]}, {p1["G"] - p1["L"]})', s1['n'] == 2 and s1['L'] == 9. and s1['G'] == 0. and p1['n'] == 3 and p1['G'] - p1['L'] == 0.)
        ok(f'1 the oracle cut at the request tick ({X.oracle(f1, K(1000.))})', X.oracle(f1, K(1000.)) == 1000.)
        f2 = F_([(300., -7.), (1500., -3.)]); ok('1 a fill before the first review is never deleted', X.oracle(f2, K(1000.)) == 1000.
                                                     and X.anchor(f2, O_(0), 1000., None)['v'] == -7.)
        f3 = F_([(1100., 1., 5.), (1100., -2., 5.), (1200., -2., 5.)]); c3 = X.oracle(f3, K(1050., 1150.))
        ok(f'1 same-time partial fills are indivisible ({c3})', c3 == 1050. and X.part(f3, X.suffix_mask(f3, 1090.))['n'] == 3 and X.part(f3, X.suffix_mask(f3, 1095.))['n'] == 1)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'1 clock: stopped ({e!r})'[:220])
    # ---- 2 the oracle
    try:
        ok('2 good-then-bad cuts the bad suffix', X.oracle(F_([(1100., 5.), (2100., -8.)]), K(1000., 2000.)) == 2000.)
        ok('2 bad-then-good with a net loss cuts both', X.oracle(F_([(1100., -8.), (2100., 5.)]), K(1000., 2000.)) == 1000.)
        ok('2 bad-then-good with a net gain keeps', X.oracle(F_([(1100., -3.), (2100., 5.)]), K(1000., 2000.)) is None)
        ok('2 KEEP wins an exact tie', X.oracle(F_([(1100., 0.), (2100., 0.)]), K(1000., 2000.)) is None)
        ok('2 KEEP wins within the tolerance', X.oracle(F_([(1100., -5e-7)]), K(1000.)) is None and X.oracle(F_([(1100., -2e-6)]), K(1000.)) == 1000.)
        ok('2 the earliest equal cut', X.oracle(F_([(1100., -3.)]), K(1000., 1050.)) == 1000.)
        ok('2 the earliest cut within the tolerance of the best', X.oracle(F_([(1055., 4e-7), (1100., -3.)]), K(1000., 1050.)) == 1000.)
        ok('2 the later strictly better cut', X.oracle(F_([(1055., 2.), (1100., -3.)]), K(1000., 1050.)) == 1050.)
        ok('2 unvalued fills count 0', X.oracle(F_([(1100., NAN), (1200., 2.)]), K(1000.)) is None and X.oracle(F_([(1100., NAN), (1200., .5)]), K(1000.)) is None
           and X.oracle(F_([(1100., NAN), (1200., -.5)]), K(1000.)) == 1000.)
        ok('2 no callback', X.oracle(F_([(1100., -3.)]), K()) is None)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'2 oracle: stopped ({e!r})'[:220])
    # ---- 3 the classes
    try:
        P = lambda G, L, U=0: dict(G=G, L=L, n_unv=U)
        cases = [(P(0., 5.), 'pure_observed_loss'), (P(5e-7, 5.), 'pure_observed_loss'), (P(5., 3.), 'compensated_mixed'), (P(3., 5.), 'uncompensated_mixed'),
                 (P(3., 3. + 5e-7), 'mixed_tie'), (P(5., 0.), 'good_only_or_no_fill'), (P(0., 5e-7), 'good_only_or_no_fill'), (P(0., 0.), 'good_only_or_no_fill'),
                 (P(0., 5., 1), 'incomplete_valuation'), (P(5., 3., 2), 'incomplete_valuation')]
        got = [X.suffix_class(p) for p, _ in cases]; ok(f'3 suffix classes ({got})', got == [c for _, c in cases])
        vec = list(X.panel_class(np.array([p['G'] for p, _ in cases]), np.array([p['L'] for p, _ in cases]), np.array([p['n_unv'] for p, _ in cases])))
        ok(f'3 the panel classes = the scalar ones ({vec})', vec == [c for _, c in cases])
        ok('3 the declared class lists', X.CLASSES == ('incomplete_valuation', 'pure_observed_loss', 'compensated_mixed', 'uncompensated_mixed', 'mixed_tie', 'good_only_or_no_fill')
           and X.B_CLASSES[-2:] == ('oracle_keep', 'no_feasible_callback') and X.TOL == 1e-6)
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'3 classes: stopped ({e!r})'[:220])
    # ---- 4 generation()
    try:
        orders = [O_(0, req=2500., remove=2510., rem=40., why='policy'), O_(1, rem=0.), O_(2, refused=True, rem=0.), O_(3, rem=30.), O_(4, req=9900., remove=9910., rem=10., why='structural'),
                  O_(5, submit=2500., act=2510.), O_(6, side=1, price=51., submit=2500.)]
        rec = REC_(orders, closed={'0': ['removed', 2610.], '1': ['filled', 3000.], '2': ['refused', 110.], '5': ['filled', 5000.], '6': ['filled', 5000.]}, open_gid={'3': 30.},
                   ahead0={'0': [0., 12., 100.], '5': [2500., 33., 100.]}, first={'0': [1000., 7., 100.]}, pull={'0': [2500., 3., 60.]},
                   known={'0': dict(first=None, n=30, front=0, unknown=30, last=None, last_t=2450., first_best_t=None, first_best_queue=None)})
        causes = [X.end_cause(o, rec['closed'], rec['open_at_end_gid'], 10000.) for o in orders[:5]]
        ok(f'4 end causes ({causes})', causes == [('recorded_cancellation', False), ('filled_out', False), ('refused', False), ('end_open', False), ('recorded_cancellation', True)])
        ok('4 a closure disagreement is refused', refuses('disagrees', X.end_cause, orders[1], {'1': ['removed', 3000.]}, {}, 10000.)
           and refuses('disagrees', X.end_cause, orders[3], {}, {}, 10000.))
        f0 = F_([(1100., 5.), (2100., -8.)]); g0 = X.generation('d', 1, 'A', orders[0], f0, (K(1000., 2000.), np.array([False, True])), rec, 10000.)
        ok(f'4 the anchors ({g0["a_cls"]}, {g0["b_cls"]}, {g0["b_cut"]}, {g0["a_v"]}, {g0["b_v"]})', g0['a_cls'] == 'uncompensated_mixed' and g0['b_cls'] == 'pure_observed_loss'
           and g0['b_cut'] == 2000. and g0['a_v'] == 0. and g0['b_v'] == 5. and g0['v_keep'] == -3. and g0['b_fallback'] is True and g0['a_net'] == -3. and g0['b_net'] == -8.)
        ok(f'4 the anchor clocks ({g0["a_age"]}, {g0["a_rem_known"]}, {g0["b_rem_known"]}, {g0["b_rem_venue"]})', g0['a_age'] == 990. and g0['a_rem_known'] == 100.
           and g0['b_rem_known'] == 90. and g0['b_rem_venue'] == 90. and g0['a_rem_venue'] == 100.)
        ok('4 the queue fields (unknown -> NaN)', g0['q_ahead_act'] == 12. and g0['q_first_t'] == 1000. and g0['q_first_venue_ahead'] == 7. and math.isnan(g0['q_first_known_ahead'])
           and g0['q_known_n'] == 30. and g0['q_pull_venue_ahead'] == 3. and g0['q_pull_venue_rem'] == 60.)
        ok(f'4 the same-price NEW generation ({g0["repl_gid"]}, {g0["repl_gap"]}, {g0["repl_ahead_act"]})', g0['repl_gid'] == 5 and g0['repl_gap'] == 0. and g0['repl_ahead_act'] == 33.)
        ok('4 the known remaining waits for the fill notice', X.anchor(F_([(950., 1.)]), O_(0), 1000., None)['rem_known'] == 100.
           and X.anchor(F_([(900., 1.)]), O_(0), 1000., None)['rem_known'] == 90.)
        g3 = X.generation('d', 1, 'A', orders[3], F_([(500., -2.)]), X.NO_KEEPS, rec, 10000.)
        ok(f'4 no feasible callback ({g3["status"]}, {g3["a_cls"]}, {g3["b_cls"]})', g3['status'] == 'no_feasible_callback' and g3['a_cls'] == g3['b_cls'] == 'no_feasible_callback'
           and g3['a_v'] == g3['b_v'] == -2. and g3['a_suf_n'] == 0 and math.isnan(g3['q_ahead_act']) and g3['repl_gid'] == -1 and g3['end_cause'] == 'end_open')
        g2 = X.generation('d', 1, 'A', orders[2], X.EMPTY, X.NO_KEEPS, rec, 10000.)
        ok('4 a refused generation stays', g2['end_cause'] == 'refused' and g2['all_n'] == 0 and g2['status'] == 'no_feasible_callback' and g2['b_cls'] == 'no_feasible_callback')
        g1 = X.generation('d', 1, 'A', orders[1], X.EMPTY, (K(1000.), np.array([False])), rec, 10000.)
        ok(f'4 no fill with a callback ({g1["a_cls"]}, {g1["b_cls"]})', g1['a_cls'] == 'good_only_or_no_fill' and g1['b_cls'] == 'oracle_keep' and g1['status'] == 'feasible')
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'4 generation: stopped ({e!r})'[:220])
    # ---- 5 sequences (the USER amendment's hand cases)
    try:
        q1 = X.sequences(F_([(1100., -3.), (2100., 5.)]), K(1000., 2000.)); q2 = X.sequences(F_([(1100., -8.), (2100., 5.)]), K(1000., 2000.))
        ok(f'5 bad-then-good, net gain ({q1["seq_btg_cut"]}, {q1["seq_btg_class"]})', q1['seq_bad_then_good'] and q1['seq_btg_feasible'] and q1['seq_btg_cut'] == 1000.
           and q1['seq_btg_G'] == 5. and q1['seq_btg_L'] == 3. and q1['seq_btg_class'] == 'compensated_mixed' and not q1['seq_good_then_bad'])
        ok(f'5 bad-then-good, net loss ({q2["seq_btg_class"]})', q2['seq_bad_then_good'] and q2['seq_btg_class'] == 'uncompensated_mixed' and q2['seq_btg_L'] == 8.)
        q3 = X.sequences(F_([(900., -3.), (2100., 5.)]), K(1000., 2000.))
        ok('5 bad-then-good with no feasible cancel before the bad fill', q3['seq_bad_then_good'] and not q3['seq_btg_feasible'] and math.isnan(q3['seq_btg_cut']) and q3['seq_btg_class'] == '')
        q4 = X.sequences(F_([(1100., 5.), (2100., -8.)]), K(1000., 2000.)); q5 = X.sequences(F_([(1100., 5.), (1105., -8.)]), K(1000., 1200.))
        ok(f'5 good-then-bad with a feasible intervening effect ({q4["seq_gtb_cut"]}, {q4["seq_gtb_class"]})', q4['seq_good_then_bad'] and q4['seq_gtb_feasible'] and q4['seq_gtb_cut'] == 2000.
           and q4['seq_gtb_class'] == 'pure_observed_loss' and q4['seq_gtb_t_good'] == 1100. and q4['seq_gtb_t_bad'] == 2100. and not q4['seq_bad_then_good'])
        ok('5 good-then-bad without a feasible intervening effect', q5['seq_good_then_bad'] and not q5['seq_gtb_feasible'] and math.isnan(q5['seq_gtb_cut']))
        ok('5 the latest feasible cut before the bad fill', X.sequences(F_([(1100., -3.), (2100., 5.)]), K(500., 1000., 2000.))['seq_btg_cut'] == 1000.)
        ok('5 an effect at the good fill is not after it', not X.sequences(F_([(1100., 5.), (2100., -8.)]), K(1090.))['seq_gtb_feasible']
           and X.sequences(F_([(1100., 5.), (2100., -8.)]), K(1091.))['seq_gtb_cut'] == 1091.)
        q6 = X.sequences(F_([(1100., 2.), (1100., -3.)]), K(1000.)); q7 = X.sequences(F_([(1100., 1.), (1200., -1.), (1300., 1.)]), K(1000.)); q8 = X.sequences(F_([(1100., 1.), (1200., -1.)]), K())
        ok(f'5 simultaneous / interleaved ({q6["seq_simultaneous_good_bad"]}, {q7["seq_interleaved"]}, {q8["seq_interleaved"]})', q6['seq_simultaneous_good_bad'] and q6['seq_interleaved']
           and q7['seq_interleaved'] and not q7['seq_simultaneous_good_bad'] and not q8['seq_interleaved'] and q8['seq_good_then_bad'] and not q8['seq_gtb_feasible'])
        q9 = X.sequences(F_([(1100., -5.), (2100., NAN)]), K(1000.)); a9 = X.anchor(F_([(1100., -5.), (2100., NAN)]), O_(0), 1000., None)
        ok(f'5 a missing later mark is incomplete, never pure ({a9["cls"]}, {q9["gen_class"]})', a9['cls'] == 'incomplete_valuation' and q9['gen_class'] == 'incomplete_valuation'
           and a9['suf_L'] == 5. and a9['suf_n_unv'] == 1)
        ok('5 the whole-generation class', X.sequences(F_([(1100., -5.)]), K())['gen_class'] == 'pure_observed_loss' and X.sequences(X.EMPTY, K())['gen_class'] == 'good_only_or_no_fill')
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'5 sequences: stopped ({e!r})'[:220])
    # ---- 6 fills_of, keeps_of, panel_rows, timeline
    try:
        mm = lambda m1, m5=None, m30=None: [50., 1., 50., 1., 50., 1., m1, 1., m5, 1., m30, 1.]
        rr = dict(mark=52., fills=[[9000., 7, 0, 10., 50., 3], [9001., 7, 0, 5., 50., 4], [100., 8, 1, 4., 50., 1], [100., 7, 0, 2., 50., 2]],
                  fill_mids=[mm(51., 52.), mm(51.), mm(49., 48., 47.), mm(53., None, 54.)], fill_class=[['same_price'], ['through'], ['through'], ['same_price']])
        fo = X.fills_of(rr, 10000.); a7, a8 = fo[7], fo[8]
        ok(f'6 fills_of ({a7["t"].tolist()}, {a7["v1"].tolist()})', a7['t'].tolist() == [100., 9000., 9001.] and a7['v1'][0] == 6. and a7['v1'][1] == 10. and math.isnan(a7['v1'][2])
           and a7['v5'][1] == 20. and math.isnan(a7['v5'][0]) and a7['v30'][0] == 8. and a7['vend'].tolist() == [4., 20., 10.] and a7['thr'].tolist() == [False, False, True])
        ok('6 two gids at one price stay apart', a8['t'].tolist() == [100.] and a8['v1'][0] == 4. and a8['v5'][0] == 8. and a8['vend'][0] == -8. and set(fo) == {7, 8})
        ok('6 fills_of refuses an unknown class', refuses('fill class', X.fills_of, dict(rr, fill_class=[['x']] * 4), 10000.))
        cm = np.array([[1000., 0, 0, 7, 0, 50, np.nan, 2, 1], [1500., 0, 0, 7, 0, 50, np.nan, 3, 12], [1200., 0, 1, 7, 0, 50, np.nan, 0, 0], [1300., 0, 0, 8, 0, 50, np.nan, 2, 11],
                       [900., 0, 0, 7, 0, 50, np.nan, 2, 11]], float)
        kp = X.keeps_of(cm)
        ok(f'6 keeps_of ({ {k: (v[0].tolist(), v[1].tolist()) for k, v in kp.items()} })', kp[7][0].tolist() == [900., 1000.] and kp[7][1].tolist() == [True, False]
           and kp[8][0].tolist() == [1300.] and kp[8][1].tolist() == [True] and set(kp) == {7, 8})
        fp = F_([(1005., 2.), (1100., -3.), (1100., -1.), (1500., NAN), (2000., 0.)]); kk = K(1000., 1090., 1400., 1600., 2100.)
        pr = X.panel_rows(fp, O_(0, act=10.), kk, np.array([False, False, True, False, False])); okp = True
        for row in pr:
            s = X.part(fp, X.suffix_mask(fp, row[0])); okp &= (row[5], row[6], row[7], row[8], row[9], row[10]) == (s['G'], s['L'], s['n_pos'], s['n_neg'], s['n_zero'], s['n_unv'])
            okp &= row[4] == 100. - float(fp['q'][~X.suffix_mask(fp, row[0])].sum()) and row[3] == 100. - float(fp['q'][fp['t'] + 100. <= row[0]].sum())
        ok(f'6 panel_rows = the scalar suffix per callback ({pr[:, 5:].tolist()})', okp and len(pr) == 5 and pr[2, 1] == 1. and pr[0, 2] == 990.)
        tl = X.timeline(F_([(1100., 5.), (2100., -8.)]), K(1000., 2000., 3000.))
        ok(f'6 the timeline ({tl})', tl == '|d=1000|1100:10.00:+5.000:T |d=2000|2100:10.00:-8.000:T |after_last=1|')
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'6 inputs: stopped ({e!r})'[:220])
    # ---- 7 tables
    try:
        recT = REC_([O_(i) for i in range(6)], open_gid={str(i): 100. for i in range(6)})
        specs = [('d1', 1, 'A', 0, [(1100., 5.), (2100., -8.)], (1000., 2000.)), ('d1', 1, 'A', 1, [(1100., -8.), (2100., 5.)], (1000., 2000.)),
                 ('d1', 2, 'A', 2, [(1100., -3.), (2100., 5.)], (1000., 2000.)), ('d2', 3, 'A', 3, [(1100., -6.)], (1000.,)), ('d2', 3, 'A', 4, [(500., -2.)], ()),
                 ('d1', 1, 'B', 0, [(1100., 1.)], (1000.,)), ('d2', 4, 'A', 5, [(1100., -9.), (2100., 1.)], (1000., 2000.))]
        rows, lines = [], {}
        for day, m, arm, g, fl, ks in specs:
            o = dict(recT['orders'][g], rem=100.); F = F_(fl); kp_ = (K(*ks), np.zeros(len(ks), bool))
            rows.append(X.generation(day, m, arm, o, F, kp_, recT, 10000.)); lines[(arm, day, m, g)] = X.timeline(F, kp_[0])
        G_ = pd.DataFrame(rows); ref = G_.groupby(['day', 'm', 'arm']).agg(all_id_mo1s=('v_keep', 'sum'), value=('end_keep', 'sum')).reset_index()
        ok('7 reproduce accepts', X.reproduce(G_, ref) == 5)
        ok('7 reproduce refuses a mismatched market', refuses('does not reproduce', X.reproduce, G_, ref.assign(value=ref.value + np.where(ref.index == 0, 1., 0.))))
        rc = X.reconcile(G_); ok(f'7 reconcile accepts ({rc})', all(rc.values()))
        ok('7 reconcile refuses a broken prefix', refuses('reconciliation', X.reconcile, G_.assign(a_pre_G=G_.a_pre_G + np.where(G_.index == 0, 1., 0.))))
        ok('7 reconcile refuses an impossible oracle class', refuses('reconciliation', X.reconcile, G_.assign(b_cls=np.where(G_.index == 0, 'compensated_mixed', G_.b_cls))))
        T, rd = X.summarize(G_); a = T[(T.arm == 'A') & (T.day == 'all') & (T.side == 'all')].iloc[0]; a1 = T[(T.arm == 'A') & (T.day == 'd1') & (T.side == 'all')].iloc[0]
        ok(f'7 summary totals ({a.v_keep}, {a.b_v}, {a.a_v}, {a.oracle_cuts})', a.v_keep == -3. - 3. + 2. - 6. - 2. - 8. and a.b_v == 5. + 0. + 2. + 0. - 2. + 0.
           and a.a_v == -2. and a.oracle_cuts == 4 and a.generations == 6 and a.feasible == 5 and a.no_feasible_callback == 1 and a.headroom_oracle == a.b_v - a.v_keep)
        ok(f'7 class loss shares ({a.a_uncompensated_mixed_L}, {a.a_pure_observed_loss_Lshare})', a.all_L == 8. + 8. + 3. + 6. + 2. + 9. and a.a_uncompensated_mixed_L == 8. + 8. + 9.
           and a.a_compensated_mixed_L == 3. and abs(a.a_pure_observed_loss_Lshare - 6. / 36.) < 1e-12 and a.b_pure_observed_loss_n == 2 and a.b_uncompensated_mixed_n == 2
           and a.b_oracle_keep_n == 1 and a.b_no_feasible_callback_n == 1)
        ok(f'7 concentration ({a.b_removed_top3_market_share}, {a1.b_removed_top3_market_share})', abs(a.b_removed_top3_market_share - 1.) < 1e-12
           and abs(a1.b_removed_top3_market_share - 1.) < 1e-12)
        ok(f'7 readings ({rd})', rd['A'] == dict(by_day={'d1': 'local_cancel_headroom', 'd2': 'no_local_cancel_rescue'}, pooled='local_cancel_headroom', days_no_rescue=1)
           and rd['B']['pooled'] == 'local_cancel_headroom')
        S_ = X.sequence_table(G_); s_ = lambda f, c: S_[(S_.arm == 'A') & (S_.day == 'all') & (S_.flag == f)].iloc[0][c]
        ok(f'7 sequence counts ({s_("seq_bad_then_good", "generations")}, {s_("seq_good_then_bad", "feasible")})', s_('seq_bad_then_good', 'generations') == 3
           and s_('seq_bad_then_good', 'compensated') == 1 and s_('seq_bad_then_good', 'uncompensated') == 2 and s_('seq_good_then_bad', 'feasible') == 1
           and s_('gen_pure_observed_loss', 'generations') == 2)
        C_ = X.class_table(G_); cr = C_[(C_.arm == 'A') & (C_.day == 'all') & (C_.side == 'all') & (C_.anchor == 'b')]
        ok(f'7 the class table ({dict(zip(cr.cls, cr.generations))})', dict(zip(cr.cls, cr.generations)) == {'no_feasible_callback': 1, 'oracle_keep': 1, 'pure_observed_loss': 2,
                                                                                                         'uncompensated_mixed': 2})
        cb = X.casebook(G_, lines); pk = cb[cb.category == 'a_uncompensated_mixed']
        ok(f'7 the casebook key ({len(cb)})', len(pk) == 3 and list(pk.key) == sorted(pk.key) and pk.key.iloc[0] == min(X.case_key('A', 'd1', 1, 0), X.case_key('A', 'd1', 1, 1),
                                                                                                                     X.case_key('A', 'd2', 4, 5))
           and all(len(g) <= 3 for _, g in cb.groupby(['category', 'arm'])))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'7 tables: stopped ({e!r})'[:220])
    # ---- 8 guards
    saved = dict(frozen=X.frozen, sources=X.sources); tmp = Path(tempfile.mkdtemp())
    try:
        mp = tmp / 'man.json'; mp.write_text(json.dumps(dict(sources=X.sources()))); pj = tmp / 'proj.json'; pj.write_text(json.dumps(dict(fits=True))); pn = tmp / 'no.json'
        pn.write_text(json.dumps(dict(fits=False)))
        X.frozen = lambda: False; ok('8 run refuses before the freeze', refuses('committed', X.run, man_path=mp, proj_path=pj, out=tmp / 'out'))
        X.frozen = lambda: True; X.sources = lambda: dict(saved['sources'](), **{'pf_ladder_cancel_headroom.py': '0' * 64})
        ok('8 run refuses another source', refuses('source', X.run, man_path=mp, proj_path=pj, out=tmp / 'out')); X.sources = saved['sources']
        ok('8 run refuses a projection that does not fit', refuses('projection', X.run, man_path=mp, proj_path=pn, out=tmp / 'out')
           and refuses('projection', X.run, man_path=mp, proj_path=tmp / 'absent.json', out=tmp / 'out'))
        (tmp / 'out').mkdir(); (tmp / 'out' / 'audit.json').write_text('{}'); ok('8 run refuses an existing audit', refuses('write-once', X.run, man_path=mp, proj_path=pj, out=tmp / 'out'))
        ok('8 the manifest and the projection are write-once', refuses('write-once', X.build_manifest, mp) and refuses('write-once', X.project, pj))
    except Exception as e:                                                                 # noqa: BLE001
        bad.append(f'8 guards: stopped ({e!r})'[:200])
    finally:
        X.frozen, X.sources = saved['frozen'], saved['sources']; shutil.rmtree(tmp, ignore_errors=True)
    if bad:
        raise SystemExit(FAIL + ': ' + ' | '.join(bad))
    print('CANCEL HEADROOM SELFTEST PASSED: cut clock, oracle, classes, generations, sequences, inputs and panel, tables, guards')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pf_ladder_cancel_headroom as X_
    main(X_)
