"""SELFTEST of pf_ladder_arrival_queue (ARRIVAL-QUEUE-DECOMPOSITION-v1; 2026-11-22, session aea5bfb9). main(X) checks the module; a failed check raises
SystemExit with FAIL. Each group names only what its code runs (the controls of the queue plan 13a70491 and the horizon report ed7c3db5).
 1 arrival boundaries: a fill exactly at the effect is I0, at +250 I1, at +1,000 I2, at +5,000 I3, just before each boundary the earlier one,
   before the effect none; the follow-up statuses not_at_risk / at_risk_full / removed_in_interval / filled_out_in_interval / end_censored
   with their at-risk ms, a completing fill exactly at an interval start inside that interval, a removal there not at risk;
 2 bands: queue zero is front, a missing or negative Q unknown (never front), Q = R within_R, Q just above R beyond_R, R = 0 / missing
   invalid_R; Binance -0 non_adverse, the smallest positive below q5, the adjacent float32 values around the bid edge, the float32-exact ask
   edge q5, NaN missing; Predict flow > 0 adverse;
 3 state_record on a hand ledger: fills split by interval, the pre-effect and before-decision fills counted outside the window, a missing
   +5 s mark missing (never zero) and its interval out of the common cohort, a horizon past END missing, END kept, the fill-level parts
   apart, the generation end and its cause (full fill, removal, END, the tie order), the follow-up of the hand state, the progress view
   (advanced / unchanged / increased / its statuses); refusals: another
   order's snapshot, another effect, a window fill after the generation end;
 4 reconcile: the hand state reproduces its audit window; a fill moved before the effect, a changed fill quantity, a changed pre-effect
   count, a duplicated state and a missing state each refuse;
 5 progress and the venue join: an earlier first review gives dQ, the same tick first_review_is_decision, a later one future_state_refused,
   none no_first_review_saved, a changed price or gid order_mismatch, a decision before activation not_yet_born, an unknown queue
   unknown_queue, the entry dQ_from_entry; recent 250 ms only at exactly d - 250; a future-only perturbation (a later saved state, a later
   fill's size, a fill inside the notice delay) leaves every decision input unchanged and a delivered change (the earlier first-review
   queue, a fill notified by d) moves it; the venue record joins only at the first review's exact tick, the notice gap agrees and a wrong
   venue remaining disagrees;
 6 cell_table and union_rows on the hand rows: opportunities, at-risk, any-fill rate, fills, shares, partial / filled-out / multi-fill,
   native and common values per opportunity and per share, state-level against fill-level positive / negative parts, the union's status
   (the end cause, also for an end exactly on the I2 / I3 boundary);
 7 guards: run refuses before the freeze, under another source and over an existing output; build_manifest refuses an existing manifest."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

FAIL = 'ARRIVAL QUEUE SELFTEST FAILED'
EDGES = {0: 0.4666730940341952, 1: 0.285456120967865}


def run(mod):
    main(mod)


def fixture():
    """The hand ledger: one bid of 100 at 49 (submit 0, activation 58), a decision at 1,000 (effect 1,037), END 60,000, mark 51."""
    mm = lambda m0, m5=None, miss5=False: [m0, 1., m0, 1., m0, 1., m0, 1., None if miss5 else (m0 if m5 is None else m5), 1., m0, 1.]
    o = dict(gid=3, side=0, rung=0., price=49., size=100., submit=0., act=58., remove=20000., refused=False)
    fills = [[900., 3, 0, 5., 49., 1], [1010., 3, 0, 3., 49., 2], [1037., 3, 0, 10., 49., 3], [1300., 3, 0, 20., 49., 4], [2500., 3, 0, 30., 49., 5], [3000., 3, 0, 2., 49., 6],
             [7000., 3, 0, 30., 49., 7]]
    mids = [mm(50.), mm(50.), mm(50.), mm(50., miss5=True), mm(50.), mm(48.), mm(50.)]
    L_ = dict(end=60000., mark=51., orders=[o], fills=fills, fill_mids=mids, queue_venue=dict(first={}))
    z = dict(day='20261024', m=7, side=0, gid=3, rung=0, size=100., t=1000., effect=1037., fp_queue=20., fp_remaining=95., o_q_rel=0.4, ctx_generation_age_ms=1000., o_depth=0.,
             adv_bn_r250=0., adv_ofi250=0.2, own_d250=-1., inv_signed=0., pf_age=12.)
    first = dict(gid=3, side=0, rung=0, size=100., t=500., fp_queue=60., fp_remaining=100.)
    return o, fills, mids, L_, z, first


def main(X):  # noqa: C901
    pd, bad = X.pd, []
    ok = lambda name, cond: None if cond else bad.append(name)

    def refuses(word, fn, *a, **kw):
        try:
            fn(*a, **kw); return False
        except ValueError as e2:
            return word in str(e2)
    # ---- 1 arrival boundaries and follow-up
    try:
        e = 1000.; got = [X.interval_of(t, e) for t in (1000., 1249.999, 1250., 1999.999, 2000., 5999.999, 6000., 1e7, 999.999)]
        ok(f'1 arrival intervals ({got})', got == ['I0', 'I0', 'I1', 'I1', 'I2', 'I2', 'I3', 'I3', None])
        fr = X.follow_up(1000., 1600., 'removal'); fe = X.follow_up(1000., 9000., 'end'); fn = X.follow_up(1000., 1000., 'removal')
        ff = X.follow_up(1000., 1250., 'full_fill'); fz = X.follow_up(1000., 1250., 'removal')
        ok(f'1 follow-up statuses ({fr}, {fe["I3"]}, {ff["I1"]}, {fz["I1"]})', fr['I0'] == ('at_risk_full', 250.) and fr['I1'] == ('removed_in_interval', 350.) and fr['I2'] == ('not_at_risk', 0.)
           and fe['I3'] == ('end_censored', 3000.) and fe['I2'] == ('at_risk_full', 4000.) and all(v[0] == 'not_at_risk' for v in fn.values())
           and ff['I0'] == ('at_risk_full', 250.) and ff['I1'] == ('filled_out_in_interval', 0.) and fz['I1'] == ('not_at_risk', 0.))
    except Exception as e_:                                                                # noqa: BLE001
        bad.append(f'1 arrival: stopped ({e_!r})'[:200])
    # ---- 2 bands
    try:
        qb = [X.queue_band(q, r) for q, r in ((0., 10.), (np.nan, 10.), (-1., 10.), (10., 10.), (10.0001, 10.), (5., 0.), (5., np.nan), (np.nan, np.nan), (3., 10.), (None, 10.))]
        ok(f'2 queue bands ({qb})', qb == ['front', 'unknown', 'unknown', 'within_R', 'beyond_R', 'invalid_R', 'invalid_R', 'invalid_R', 'within_R', 'unknown'])
        tiny = float(np.nextafter(np.float32(0.), np.float32(1.)))
        bb = [X.bn_band(x, EDGES[0]) for x in (-0., tiny, 0.4666731059551239, 0.4666730761528015, np.nan, -0.3, None)]
        ok(f'2 Binance bands ({bb})', bb == ['non_adverse', 'positive_below_q5', 'q5', 'positive_below_q5', 'missing', 'non_adverse', 'missing'])
        ok('2 q5 at the float32-exact ask edge', X.bn_band(0.285456120967865, EDGES[1]) == 'q5' and X.bn_band(0.2854560911655426, EDGES[1]) == 'positive_below_q5')
        ok('2 the float32 cast before the comparison (the frozen rule\'s)', X.bn_band(0.466673092, EDGES[0]) == 'q5' and X.bn_band(0.466673090, EDGES[0]) == 'positive_below_q5')
        ok('2 Predict flow bands', [X.pf_band(x) for x in (0.1, 0., -0.2, np.nan)] == ['adverse', 'non_adverse', 'non_adverse', 'missing'])
    except Exception as e_:                                                                # noqa: BLE001
        bad.append(f'2 bands: stopped ({e_!r})'[:200])
    # ---- 3 state_record on a hand ledger
    try:
        o, fills, mids, L_, z, first = fixture()
        rec, rows = X.state_record(z, o, fills, mids, L_, first, None, None, EDGES); by = {r['interval']: r for r in rows}
        ok(f'3 fills by interval ({[(r["interval"], r["fills"]) for r in rows]}, pre {rec["pre_effect_fills"]}, before {rec["before_decision_fills"]})',
           [by[n]['fills'] for n in ('I0', 'I1', 'I2', 'I3')] == [1, 1, 2, 1] and rec['pre_effect_fills'] == 1 and rec['before_decision_fills'] == 1)
        ok('3 a missing +5 s mark stays missing, never zero; the interval leaves the common cohort', np.isnan(by['I1']['v5000']) and not by['I1']['common'] and by['I0']['common']
           and by['I1']['v1000'] == 20. and by['I1']['vEND'] == 40. and by['I2']['common'])
        ok(f'3 the fill-level parts apart ({by["I2"]["v1000"]}, {by["I2"]["fpos1000"]}, {by["I2"]["fneg1000"]})', by['I2']['v1000'] == 28. and by['I2']['fpos1000'] == 30. and by['I2']['fneg1000'] == -2.)
        fv = X.fill_values([50000., 3, 0, 1., 49., 9], [50.] * 12, 51., 60000.)
        ok('3 a horizon past END stays missing', fv[30000] is None and fv[5000] == 1. and fv['END'] == 2.)
        ok(f'3 the generation end and cause ({rec["gen_end"]}, {rec["gen_end_cause"]})', (rec['gen_end'], rec['gen_end_cause']) == (7000., 'full_fill') and by['I3']['filled_out']
           and not by['I2']['filled_out'] and X.generation_end(dict(o, remove=1500.), fills[:3], 60000.) == (1500., 'removal')
           and X.generation_end(dict(o, remove=1e9), fills[:2], 60000.) == (60000., 'end') and X.generation_end(dict(o, remove=7000.), fills, 60000.) == (7000., 'full_fill')
           and X.generation_end(dict(o, remove=60000.), fills[:2], 60000.) == (60000., 'end'))
        ok(f'3 the hand follow-up ({[(r["interval"], r["follow_up"], r["at_risk_ms"]) for r in rows]})', [(r['follow_up'], r['at_risk_ms']) for r in rows]
           == [('at_risk_full', 250.), ('at_risk_full', 750.), ('at_risk_full', 4000.), ('filled_out_in_interval', 7000. - 6037.)])
        pv = [X.state_record(z, o, fills, mids, L_, f_, None, None, EDGES)[0]['progress'] for f_ in (first, dict(first, fp_queue=20.), dict(first, fp_queue=10.), dict(first, t=1000.), None)]
        ok(f'3 the progress view ({pv})', pv == ['advanced', 'unchanged', 'increased', 'first_review_is_decision', 'no_first_review_saved'])
        ok('3 the hand queue / condition fields', rec['queue_band'] == 'within_R' and rec['pressure_flow'] == 'non_adverse|PF_adverse' and rec['Q_over_R'] == 20. / 95.
           and rec['known_filled'] == 5. and rec['age_since_activation_ms'] == 942. and rec['R_recomputed'] == 95. and rec['R_agrees'] and rec['population'] == 'BASELINE_neutral_live')
        ok('3 another order\'s snapshot refuses', refuses('ledger order', X.state_record, dict(z, side=1), o, fills, mids, L_, first, None, None, EDGES)
           and refuses('ledger order', X.state_record, dict(z, rung=1), o, fills, mids, L_, first, None, None, EDGES))
        ok('3 another effect refuses', refuses('effect', X.state_record, dict(z, effect=1036.), o, fills, mids, L_, first, None, None, EDGES))
        ok('3 a window fill after the generation end refuses', refuses('generation end', X.state_record, z, dict(o, remove=2000.), fills, mids, L_, first, None, None, EDGES))
    except Exception as e_:                                                                # noqa: BLE001
        bad.append(f'3 state_record: stopped ({e_!r})'[:220])
    # ---- 4 reconcile
    try:
        S1 = pd.DataFrame([rec]); I1 = pd.DataFrame(rows)
        win = pd.DataFrame([dict(day='20261024', m=7, gid=3, win_n=5, win_q=92., win_end=184., win_mo1s=88., win_mo50=88., win_mo100=88., pre_n=1)])
        rc = X.reconcile(S1, I1, win); ok(f'4 the hand state reconciles ({rc})', rc['window_fills'] == 5 and rc['interval_rows'] == 4)
        f2 = [list(f) for f in fills]; f2[2][0] = 1036.; r2, rows2 = X.state_record(z, o, f2, mids, L_, first, None, None, EDGES)
        ok('4 a fill moved before the effect refuses', refuses('window', X.reconcile, pd.DataFrame([r2]), pd.DataFrame(rows2), win))
        f3 = [list(f) for f in fills]; f3[4][3] = 31.; r3, rows3 = X.state_record(z, o, f3, mids, L_, first, None, None, EDGES)
        ok('4 a changed fill quantity refuses', refuses('window', X.reconcile, pd.DataFrame([r3]), pd.DataFrame(rows3), win))
        ok('4 a changed pre-effect count refuses', refuses('window', X.reconcile, S1, I1, win.assign(pre_n=0)))
        ok('4 a duplicated state refuses', refuses('exactly once', X.reconcile, pd.concat([S1, S1]), pd.concat([I1, I1]), win))
        ok('4 a missing state refuses', refuses('exactly once', X.reconcile, S1.iloc[:0], I1.iloc[:0], win))
        ok('4 a missing interval row refuses', refuses('interval row', X.reconcile, S1, I1.iloc[:3], win))
    except Exception as e_:                                                                # noqa: BLE001
        bad.append(f'4 reconcile: stopped ({e_!r})'[:220])
    # ---- 5 progress, recent, the decision inputs and the venue join
    try:
        o, fills, mids, L_, z, first = fixture(); idn = lambda r: X.ident_of(r, o)
        pg = lambda d, Q, pr, ident=None, kind='first_review': X.progress(d, Q, pr, o, ident if ident is not None else (idn(pr) if pr is not None else None), kind)
        ok('5 progress statuses', pg(1000., 20., first)[0:3:2] == ('ok', -40.) and pg(1000., 20., first)[3] == 500. and pg(1000., 20., dict(first, t=1000.))[0] == 'first_review_is_decision'
           and pg(1000., 20., dict(first, t=1500.))[0] == 'future_state_refused' and pg(1000., 20., None)[0] == 'no_first_review_saved'
           and pg(1000., 20., first, dict(idn(first), price=48.))[0] == 'order_mismatch' and pg(1000., 20., first, dict(idn(first), gid=4))[0] == 'order_mismatch'
           and pg(50., 20., first)[0] == 'not_yet_born' and pg(1000., np.nan, first)[0] == 'unknown_queue'
           and pg(1000., 20., dict(first, t=0., fp_queue=80.), kind='entry')[0:3:2] == ('ok', -60.) and pg(1000., 20., None, kind='entry')[0] == 'no_entry_saved')
        ok('5 recent 250 ms only at exactly d - 250', X.recent(1000., 20., dict(first, t=750.), o) == ('available', -40.) and X.recent(1000., 20., first, o)[0] == 'unavailable_no_saved_state'
           and X.recent(200., 20., dict(first, t=-50.), o)[0] == 'not_yet_born_at_t_minus_250')
        keys = ('Q', 'R', 'Q_over_R', 'queue_band', 'bn_band', 'pf_band', 'first_status', 'dQ_first', 'entry_status', 'dQ_from_entry', 'recent_status', 'dQ_250', 'R_recomputed', 'known_filled')
        dec = lambda r: tuple('nan' if isinstance(r[k], float) and np.isnan(r[k]) else r[k] for k in keys)
        entry = dict(gid=3, side=0, rung=0, size=100., t=0., fp_queue=80., fp_remaining=100.)
        b0 = dec(X.state_record(z, o, fills, mids, L_, first, entry, None, EDGES)[0])
        later = [list(f) for f in fills]; later[6][3] = 29.; later.append([7100., 3, 0, 1., 49., 8]); mids_l = mids + [mids[-1]]
        nd = [list(f) for f in fills]; nd[0][0] = 905.                                     # 905 + 100 > 1,000: not yet notified at d
        fut = [dec(X.state_record(z, o, fills, mids, L_, first, entry, None, EDGES)[0]), dec(X.state_record(z, o, later, mids_l, L_, first, entry, None, EDGES)[0]),
               dec(X.state_record(z, o, nd, mids, L_, first, entry, None, EDGES)[0])]
        ok(f'5 a future-only perturbation leaves every decision input unchanged ({[f == b0 for f in fut]})', all(f == b0 for f in fut[:2]) and fut[2][keys.index('R_recomputed')] == 100.
           and fut[2][:keys.index('R_recomputed')] == b0[:keys.index('R_recomputed')])
        dl = dec(X.state_record(z, o, fills, mids, L_, dict(first, fp_queue=75.), entry, None, EDGES)[0]); nf = [list(f) for f in fills]; nf[0][3] = 6.
        dn = dec(X.state_record(z, o, nf, mids, L_, first, entry, None, EDGES)[0])
        lf = [dec(X.state_record(z, o, fills, mids, L_, dict(first, t=1500., fp_queue=q_), entry, None, EDGES)[0]) for q_ in (1., 999.)]
        ok(f'5 a delivered change moves the decision input ({dl[keys.index("dQ_first")]}, {dn[keys.index("R_recomputed")]})', dl[keys.index('dQ_first')] == -55. and dl != b0
           and dn[keys.index('R_recomputed')] == 94.)
        ok('5 a later saved state\'s queue leaves the decision inputs unchanged', lf[0] == lf[1] and lf[0][keys.index('first_status')] == 'future_state_refused')
        ok('5 R known only after the notice', fut[2][keys.index('R_recomputed')] == 100. and not X.state_record(z, o, nd, mids, L_, first, entry, None, EDGES)[0]['R_agrees'])
        vf = [[400., 3, 0, 2., 49., 0], [450., 3, 0, 5., 49., 1]]; fq = dict(first, fp_remaining=98.)                # 400 + 100 <= 500: known at the review
        va = X.venue_quality(fq, [500., 58., 93.], vf, o); vb = X.venue_quality(fq, [450., 58., 93.], vf, o); vc = X.venue_quality(fq, [500., 58., 88.], vf, o)
        ok(f'5 the venue joins only at the first review\'s exact tick ({va["venue_status"]}, {vb["venue_status"]})', va['venue_status'] == 'available' and va['client_minus_venue'] == 2.
           and va['unnotified_q'] == 5. and va['notice_gap_agrees'] is True and vb['venue_status'] == 'time_mismatch' and np.isnan(vb['client_minus_venue'])
           and vc['notice_gap_agrees'] is False and X.venue_quality(first, None, vf, o)['venue_status'] == 'no_venue_entry'
           and X.venue_quality(None, [500., 58., 93.], vf, o)['venue_status'] == 'no_first_review_saved' and X.venue_quality(dict(fq, gid=4), [500., 58., 93.], vf, o)['venue_status'] == 'order_mismatch')
    except Exception as e_:                                                                # noqa: BLE001
        bad.append(f'5 progress: stopped ({e_!r})'[:220])
    # ---- 6 cell_table and union_rows
    try:
        T = X.cell_table(S1, I1); sel = lambda iv, coh: T[(T.view == 'queue') & (T.interval == iv) & (T.day == 'all') & (T.side == 'all') & (T.cohort == coh)].iloc[0]
        g = sel('I0', 'native'); g2 = sel('I2', 'native'); g1 = sel('I1', 'native'); u = sel('union', 'common'); un = sel('union', 'native')
        ok(f'6 an interval cell ({g.opportunities}, {g.at_risk}, {g.any_fill_rate}, {g.h1000_net}, {g.h1000_per_share})', g.opportunities == 1 and g.at_risk == 1 and g.any_fill_rate == 1.
           and g.fills == 1 and g.shares == 10. and g.h1000_net == 10. and g.h1000_per_share == 1. and g.h1000_per_opportunity == 10. and g.h0_net == 10.)
        ok(f'6 state-level and fill-level parts ({g2.h1000_state_positive}, {g2.h1000_fill_positive}, {g2.h1000_fill_negative})', g2.h1000_state_positive == 28. and g2.h1000_state_negative == 0.
           and g2.h1000_fill_positive == 30. and g2.h1000_fill_negative == -2. and g2.multi_fill == 1 and g2.partial == 1 and g1.h5000_valid == 0 and g1.h1000_valid == 1)
        ok(f'6 the union cell ({u.h1000_valid}, {un.h1000_net}, {un.h5000_valid})', u.h1000_valid == 0 and u.fills == 5 and u.shares == 92. and u.filled_out == 1 and u.partial == 0
           and u.n_filled_out_in_interval == 1 and un.h1000_net == 88. and un.h5000_valid == 0 and un.end_net == 184.)
        U = X.union_rows(I1.merge(S1[['day', 'm', 'gid']], on=['day', 'm', 'gid']))
        r5, rows5 = X.state_record(z, dict(o, remove=6037.), fills[:6], mids[:6], L_, first, None, None, EDGES); U2 = X.union_rows(pd.DataFrame(rows5))   # G = e + 5,000 exactly
        I3 = I1.assign(follow_up='not_at_risk'); U3 = X.union_rows(I3)
        o2, z2 = dict(o, gid=4, remove=1200.), dict(z, gid=4); r4, rows4 = X.state_record(z2, o2, [], [], L_, None, None, None, EDGES)
        T2 = X.cell_table(pd.concat([S1, pd.DataFrame([r4])], ignore_index=True), pd.concat([I1, pd.DataFrame(rows4)], ignore_index=True))
        c2 = T2[(T2.view == 'queue') & (T2.interval == 'I1') & (T2.day == 'all') & (T2.side == 'all') & (T2.cohort == 'native')].iloc[0]
        ok(f'6 the any-fill rate over the at-risk states ({c2.opportunities}, {c2.at_risk}, {c2.any_fill_rate})', c2.opportunities == 2 and c2.at_risk == 1 and c2.any_fill_rate == 1.
           and c2.n_not_at_risk == 1 and rows4[0]['follow_up'] == 'removed_in_interval')
        ok(f'6 the union status is the end cause\'s ({U.follow_up.iloc[0]}, {U2.follow_up.iloc[0]}, {[r["follow_up"] for r in rows5][2:]}, {U3.follow_up.iloc[0]})',
           U.follow_up.iloc[0] == 'filled_out_in_interval' and U2.follow_up.iloc[0] == 'removed_in_interval' and [r['follow_up'] for r in rows5][2:] == ['at_risk_full', 'not_at_risk']
           and U3.follow_up.iloc[0] == 'not_at_risk')
    except Exception as e_:                                                                # noqa: BLE001
        bad.append(f'6 cells: stopped ({e_!r})'[:220])
    # ---- 7 guards
    saved = dict(frozen=X.frozen, sources=X.sources); tmp = Path(tempfile.mkdtemp(dir=str(X.A.DATA)))
    try:
        mp = tmp / 'man.json'; mp.write_text(json.dumps(dict(sources=X.sources())))
        X.frozen = lambda: False; ok('7 run refuses before the freeze', refuses('committed', X.run, man_path=mp, out=tmp / 'out'))
        X.frozen = lambda: True; X.sources = lambda: dict(saved['sources'](), **{'pf_ladder_rest.py': '0' * 64}); ok('7 run refuses another source', refuses('source', X.run, man_path=mp, out=tmp / 'out'))
        X.sources = saved['sources']; (tmp / 'out').mkdir(); (tmp / 'out' / 'audit.json').write_text('{}'); ok('7 run refuses an existing output', refuses('write-once', X.run, man_path=mp, out=tmp / 'out'))
        ok('7 build_manifest refuses an existing manifest', refuses('write-once', X.build_manifest, mp))
    except Exception as e_:                                                                # noqa: BLE001
        bad.append(f'7 guards: stopped ({e_!r})'[:200])
    finally:
        X.frozen, X.sources = saved['frozen'], saved['sources']; shutil.rmtree(tmp, ignore_errors=True)
    if bad:
        raise SystemExit(FAIL + ': ' + ' | '.join(bad))
    print('ARRIVAL QUEUE SELFTEST PASSED: arrival boundaries and follow-up, bands, state_record, reconcile, progress / decision inputs / venue join, cells and union, guards')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pf_ladder_arrival_queue as X_
    main(X_)
