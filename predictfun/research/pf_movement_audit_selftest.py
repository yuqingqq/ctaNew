"""SELFTEST of pf_movement_audit (PI0-MOVEMENT-CALIBRATION-AUDIT-v1; 2026-11-21, session aea5bfb9). run(M) checks the module object M; any failed
check exits with FAIL. Hand rows only (no world, no tape):
 1. accounting: the exact calibrated mixture (zero bias, both movement terms zero); a biased probability with an exact conditional head (the
    probability term carries it); an exact probability with a biased head (the conditional term carries it); a changed value or a V in other
    units refused (accounting_mismatch); zero_P, movement_missing and no_forecast_or_target;
 2. fills: an equal total movement split into pre- and post-fill paths; a fill before the effect or of another gid not counted; the delay
    classes; a missing mid (reference_missing), no record (reference_unavailable), a mid stamped after the fill (reference_out_of_time); the
    ask side's signs; retained forced mids; a moved fill time or a moved END mark refused (fill_mismatch);
 3. strata: every bin boundary; the decision-time queue reads the client's fp_queue, the venue's ahead only the explanatory stratum;
 4. sources: a quote-only update (bookTicker new, PF and depth aging) and a clock-only transition (no new observation);
 5. binding (review 5a8ee6f5 item 4): a doubled row weight, a wrong cell or a wrong part is weight_mismatch; a substituted partition is
    refused; incomplete membership is refused;
 6. tables: the moments views (forced overlap and admission), severe sums, the pooled Brier reference, ESS; prepost per share; weighted /
    unweighted concentration and signed contributions; the retained control's hour from the review's own decision time;
 7. predicates on hand tables: broad / concentrated / inconclusive, decision_state / fill_timing / both / inconclusive, material / similar /
    inconclusive, populated / thin, and the recommendation map; mid_lookup's convention; evidence rows;
 8. the writers: measure on hand cohorts (every output; immutable; incomplete membership refused); run_part refuses before the freeze;
    frozen / inputs_frozen refuse uncommitted copies."""
import datetime
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

FAIL = 'MOVEMENT AUDIT SELFTEST FAILED'


def run(M):  # noqa: C901
    bad = []; pd, LS, C, RT, PA = M.pd, M.LS, M.C, M.RT, M.PA; F_ = RT.FEATS
    vec = lambda **kw: [float(kw.get(n, 0.)) for n in F_]
    mo = lambda P, qh, mh, edge=.5: dict(P=P, Eq=P * qh, Eqm=P * mh, V=edge * P * qh + P * mh, edge=edge)

    def stage(P, qh, mh, q, M_, edge=.5):
        return dict(moments=mo(P, qh, mh, edge), target=dict(q=q, value=edge * q + M_, qm=M_ if q > 0 else 0.))
    # ---- 1 accounting
    try:
        rows = [M.accounting(stage(.3, 30., -100., 30., -100.)) for _ in range(3)] + [M.accounting(stage(.3, 30., -100., 0., 0.)) for _ in range(7)]
        D = pd.DataFrame(rows)
        if not (all(r['acc_status'] == 'ok' for r in rows) and abs(D.err_m.mean()) < 1e-9 and abs(D.mov_prob.mean()) < 1e-9 and abs(D.mov_cond.mean()) < 1e-9
                and abs(D.err_V.mean()) < 1e-9 and D.identity_residual.max() < 1e-9 and abs(D.mech.iloc[0] - .5 * 9.) < 1e-12):
            bad.append('accounting: the exact calibrated mixture')
        Dp = pd.DataFrame([M.accounting(stage(.5, 30., -100., 30., -100.)) for _ in range(3)] + [M.accounting(stage(.5, 30., -100., 0., 0.)) for _ in range(7)])
        Dh = pd.DataFrame([M.accounting(stage(.3, 30., -50., 30., -100.)) for _ in range(3)] + [M.accounting(stage(.3, 30., -50., 0., 0.)) for _ in range(7)])
        if not (abs(Dp.mov_prob.mean() - (-20.)) < 1e-9 and abs(Dp.mov_cond.mean()) < 1e-9 and abs(Dh.mov_prob.mean()) < 1e-9 and abs(Dh.mov_cond.mean() - 15.) < 1e-9
                and abs(Dh.err_m.mean() - 15.) < 1e-9 and abs(Dh.c_m.dropna().mean() - 50.) < 1e-9 and abs(Dp.q_prob.mean() - 6.) < 1e-9 and abs(Dp.q_cond.mean()) < 1e-9):
            bad.append('accounting: the probability / conditional-head attribution')
        s0 = stage(.3, 30., -100., 30., -100.); s0['target']['value'] += 1.
        s1 = stage(.3, 30., -100., 30., -100.); s1['moments']['V'] *= 100.
        if M.accounting(s0)['acc_status'] != 'accounting_mismatch' or M.accounting(s1)['acc_status'] != 'accounting_mismatch':
            bad.append('accounting: a changed value or a V in other units was accepted')
        if not (M.accounting(stage(0., 30., -100., 0., 0.))['acc_status'] == 'zero_P'
                and M.accounting(dict(moments=mo(.3, 30., -100.), target=dict(q=30., value=1., qm=None)))['acc_status'] == 'movement_missing'
                and M.accounting(dict(moments=None, target=dict(q=0., value=0., qm=0.)))['acc_status'] == 'no_forecast_or_target'):
            bad.append('accounting: zero_P / movement_missing / no_forecast_or_target')
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'accounting: stopped ({e!r})'[:200])
    # ---- 2 fills
    try:
        r = dict(side=0, weight=2., day='20261026', held=dict(gid=0, act=58., mark=52., fills=[[110., 0, 0, 10., 50., 0], [600., 0, 0, 20., 50., 1], [40., 0, 0, 1., 50., 2],
                                                                                             [700., 3, 0, 5., 50., 3]]))
        x = dict(effect=137., inputs=vec(pf_mid=50.5))
        ok_ref = dict(status='ok', mids={0: dict(mid=51., frame_t=100., recv_t=101.), 1: dict(mid=51.5, frame_t=590., recv_t=592.)})
        fl = M.fill_rows(r, x, 'first_review', ok_ref, 'admitted', 'k'); fe = M.fill_rows(r, dict(x), 'entry', ok_ref, 'admitted', 'k')
        if not (len(fl) == 1 and fl[0]['t'] == 600. and abs(fl[0]['pre'] + fl[0]['post'] - fl[0]['total']) < 1e-12 and abs(fl[0]['pre'] - 20. * 1.) < 1e-12
                and abs(fl[0]['post'] - 20. * .5) < 1e-12 and abs(fl[0]['total'] - 20. * 1.5) < 1e-12 and fl[0]['delay_class'] == '(100,500]' and fl[0]['age'] == 10.
                and len(fe) == 2 and fe[0]['delay_class'] == '[0,100]' and fe[0]['ref_status'] == 'ok'):
            bad.append(f'fills: the pre / post split, the effect / gid cut or the delay classes ({[(f["t"], f["pre"], f["post"], f["delay_class"]) for f in fl + fe]})')
        if [f['ref_status'] for f in M.fill_rows(r, x, 'entry', dict(status='ok', mids={0: ok_ref['mids'][0]}), 'admitted', 'k')] != ['ok', 'reference_missing']:
            bad.append('fills: a missing mid is not reference_missing')
        if [f['ref_status'] for f in M.fill_rows(r, x, 'entry', dict(status='no_record'), 'admitted', 'k')] != ['reference_unavailable:no_record'] * 2:
            bad.append('fills: a missing record is not reference_unavailable')
        if M.fill_rows(r, x, 'first_review', dict(status='ok', mids={1: dict(mid=51.5, frame_t=700., recv_t=701.)}), 'admitted', 'k')[0]['ref_status'] != 'reference_out_of_time':
            bad.append('fills: a mid stamped after the fill was accepted')
        fa = M.fill_rows(dict(r, side=1), x, 'first_review', ok_ref, 'admitted', 'k')
        if not (abs(fa[0]['total'] - (-1.) * 20. * (52. - 50.5)) < 1e-12 and abs(fa[0]['pre'] - (-1.) * 20. * 1.) < 1e-12 and abs(fa[0]['pre'] + fa[0]['post'] - fa[0]['total']) < 1e-12):
            bad.append('fills: the ask side signs')
        fm = M.forced_mids(dict(side=1, fills=[dict(q=10., price=52., fill_time_edge=-1. * (51. - 52.) * 10.), dict(q=5., price=52., fill_time_edge=None)]))
        if not (fm['status'] == 'ok' and abs(fm['mids'][0]['mid'] - 51.) < 1e-12 and fm['mids'][0]['frame_t'] is None and 1 not in fm['mids']):
            bad.append('forced_mids: the mid from a retained fill-time edge')
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'fills: stopped ({e!r})'[:200])
    # ---- 3 strata
    try:
        ok = (M.hbin(100.) == '(0,100]' and M.hbin(100.5) == '(100,500]' and M.hbin(500.) == '(100,500]' and M.hbin(2000.) == '(500,2000]' and M.hbin(5000.) == '(2000,5000]'
              and M.hbin(0.) == 'outside' and M.hbin(5001.) == 'outside'
              and M.qbin(0., 30.) == '0' and M.qbin(30., 30.) == '(0,1]' and M.qbin(31., 30.) == '(1,3]' and M.qbin(90., 30.) == '(1,3]' and M.qbin(91., 30.) == '>3'
              and M.qbin(5., 0.) == 'missing_or_invalid' and M.qbin(-1., 30.) == 'missing_or_invalid' and M.qbin(np.nan, 30.) == 'missing_or_invalid'
              and M.obin(1.) == '1' and M.obin(2.) == '2' and M.obin(3.) == '>=3' and M.obin(np.nan) == 'missing_or_invalid'
              and M.dbin(0.) == '[0,100]' and M.dbin(100.) == '[0,100]' and M.dbin(100.1) == '(100,500]' and M.dbin(500.) == '(100,500]' and M.dbin(2000.) == '(500,2000]'
              and M.dbin(2000.1) == '>2000' and M.dbin(-1.) == 'negative'
              and M.mbin(-10.) == '(-inf,-10.0]' and M.mbin(-3.) == '(-10.0,-3.0]' and M.mbin(-1.) == '(-3.0,-1.0]' and M.mbin(0.) == '(-1.0,0.0]' and M.mbin(.1) == '(0.0,inf]'
              and M.vbin(-3.) == '(-10.0,-3.0]' and M.vbin(0.) == '(-3.0,0.0]' and M.vbin(3.) == '(0.0,3.0]' and M.vbin(3.1) == '(3.0,inf]'
              and M.eqbin(1.) == '[0,1]' and M.eqbin(3.) == '(1,3]' and M.eqbin(10.) == '(3,10]' and M.eqbin(10.1) == '>10' and M.eqbin(-.1) == 'negative'
              and M.pbin(.005) == '[0.0,0.01)' and M.pbin(1.) == '[0.4,1.0)' and M.THIN_N == 30 and M.THIN_FILLS == 10 and M.RATIO == 1.5 and M.SEVERE == 30.)
        if not ok:
            bad.append('strata: a bin boundary or a declared constant')
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'strata: stopped ({e!r})'[:200])
    t0 = int(datetime.datetime(2026, 10, 26, 0, 59, 59, tzinfo=datetime.timezone.utc).timestamp() * 1000)

    def hrow(m, side=0, P=.3, qh=30., mh=-10., q=0., M_=0., fills=(), venue=500., review=None, day='20261026', t_ms=None, weight=1., act='POST', ract='PULL', mark=52.):
        t_ms = t_ms or t0 + 1000 * m; edge = .5
        ent = dict(status='ok', dec_status='model', action=act, t=0., inputs=vec(pf_mid=50.5, fp_queue=10., fp_remaining=30., t_to_end=5000., ctx_generation_ordinal=1.),
                   moments=dict(mo(P, qh, mh, edge), mech=edge * P * qh, q_fill=qh, qm_fill=mh), target=dict(q=q, value=edge * q + M_, qm=M_ if q > 0 else 0.), view=dict(received=1.))
        st = dict(entry=ent, first_pull=dict(status='no_pull'))
        st['first_review'] = dict(ent, t=float(review), action=ract, venue=dict(ahead=venue), effect=float(review) + 37.) if review is not None else dict(status='no_review')
        return dict(day=day, m=m, t_ms=t_ms, side=side, gid=0, decision_ms=0, weight=weight, n_cell=1, cell=list(PA.cell_of(t_ms, 0, side)), status='ok',
                    held=dict(gid=0, act=58., mark=mark, fills=[list(f) for f in fills]), fills=[], stages=st)
    # ---- the fill / mark refusals and the strata reads through tables_of
    try:
        good = hrow(1, q=10., M_=10. * (52. - 50.5), fills=[(100., 0, 0, 10., 50., 0)])
        moved_t = hrow(2, q=10., M_=10. * (52. - 50.5), fills=[(40., 0, 0, 10., 50., 0)])
        moved_mark = hrow(3, q=10., M_=10. * (52. - 50.5), fills=[(100., 0, 0, 10., 50., 0)], mark=53.)
        O, Fx = M.tables_of([good, moved_t, moved_mark], 'admitted', lambda r_: dict(status='ok', mids={}))
        e = O[O.stage == 'entry']
        if list(e.acc_status) != ['ok', 'fill_mismatch', 'fill_mismatch']:
            bad.append(f'tables_of: a moved fill time or END mark was accepted ({list(e.acc_status)})')
        g0 = e.iloc[0]
        if not (g0.qbin == '(0,1]' and g0.hbin == '(2000,5000]' and g0.obin == '1' and g0.delay_class == '[0,100]' and g0.split_status == 'reference_incomplete'
                and g0.venue_qbin == 'missing_or_invalid'):
            bad.append(f'tables_of: the client queue stratum ({g0.qbin}), the horizon / ordinal or the split status')
        nm_row = hrow(5); nm_row['stages']['entry']['dec_status'] = 'fallback'
        O3, _ = M.tables_of([nm_row], 'admitted', lambda r_: dict(status='no_record'))
        if O3[O3.stage == 'entry'].acc_status.iloc[0] != 'not_eligible':
            bad.append('tables_of: a non-model decision was eligible')
        if not (M.admission_of(dict(stages=dict(entry=dict(status='ok', dec_status='incoherent_forecast', admission='would_WAIT'))), 'forced') == 'fallback'
                and M.admission_of(dict(stages=dict(entry=dict(status='not_absent'))), 'forced') == 'entry:not_absent'
                and M.admission_of(dict(stages=dict(entry=dict(status='ok', dec_status='model', admission='would_POST'))), 'forced') == 'would_POST'
                and M.admission_of(dict(stages={}), 'admitted') == 'admitted'):
            bad.append('admission_of: the shadow admission labels')
        O2, _ = M.tables_of([hrow(4, review=100., venue=500.)], 'admitted', lambda r_: dict(status='no_record'))
        rv = O2[O2.stage == 'first_review'].iloc[0]
        if not (rv.qbin == '(0,1]' and rv.venue_qbin == '>3' and rv.delay_class == 'no_fill' and rv.split_status == 'no_fill' and rv.pre == 0.):
            bad.append(f'tables_of: the venue queue leaked into the decision stratum or the no-fill split ({rv.qbin}, {rv.venue_qbin})')
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'tables_of: stopped ({e!r})'[:200])
    # ---- 4 sources
    try:
        def trow(a0, a1, rcv0, rcv1):
            return dict(day='20261026', side=0, weight=1., status='ok', stages=dict(entry=dict(dec_status='model', t=0., inputs=vec(**a0), view=dict(received=rcv0)),
                                                                                     first_review=dict(dec_status='model', t=100., inputs=vec(**a1), view=dict(received=rcv1))))
        quote = trow(dict(pf_age=10., bn_age=40., bnd_age=5., pf_mid=50.), dict(pf_age=110., bn_age=3., bnd_age=105., pf_mid=49.), 1., 1.)
        clock = trow(dict(pf_age=10., bn_age=40., bnd_age=5., pf_mid=50.), dict(pf_age=110., bn_age=140., bnd_age=105., pf_mid=50.), 1., 1.)
        summ, S = M.sources_table([quote, clock], 'admitted')
        if not (bool(S.iloc[0].new_bookticker) and not bool(S.iloc[0].new_pf_frame) and not bool(S.iloc[0].new_depth) and not bool(S.iloc[0].new_pf_receipt)
                and bool(S.iloc[0].any_new) and not bool(S.iloc[1].any_new) and S.iloc[0].move_class == 'against' and S.iloc[1].move_class == 'unchanged'
                and summ[0]['new_bookticker'] == .5 and summ[-1]['n'] == 1):
            bad.append('sources_table: the quote-only update or the clock-only transition')
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'sources: stopped ({e!r})'[:200])
    # ---- 5 binding, 8 writers
    try:
        with tempfile.TemporaryDirectory(dir=str(PA.R2.I.K.IO)) as tmp:
            tmp = Path(tmp); adm = tmp / 'adm'; (adm / 'parts').mkdir(parents=True)
            base = [hrow(i, q=10. if i == 0 else 0., M_=15. if i == 0 else 0., fills=[(100., 0, 0, 10., 50., 0)] if i == 0 else [], review=100.) for i in range(4)]
            sel = [dict({k: b_[k] for k in ('day', 'm', 't_ms', 'side', 'gid', 'decision_ms')}, weight=1.5, n_cell=3, cell=b_['cell'], part=0) for b_ in base]
            (adm / 'manifest.json').write_text(json.dumps(dict(selected=sel, parts=[0])))
            prow = [dict(base[0], weight=1.5, n_cell=3), dict(base[1], weight=3., n_cell=3), dict(base[2], weight=1.5, n_cell=3, cell=[base[2]['cell'][0], 0, 23]),
                    dict(base[3], weight=1.5, n_cell=3)]
            b = LS.gz_bytes(prow); (adm / 'parts' / 'part_000.jsonl.gz').write_bytes(b)
            (adm / 'parts' / 'part_000.json').write_text(json.dumps(dict(sha256=hashlib.sha256(b).hexdigest(), manifest_sha256=C.sha(adm / 'manifest.json'))))
            ar, _ = M.adm_rows(adm)
            if [r_['status'] for r_ in ar] != ['ok', 'weight_mismatch', 'weight_mismatch', 'ok'] or ar[1]['weight'] != 1.5:
                bad.append(f"adm_rows: the manifest binding ({[r_['status'] for r_ in ar]})")
            sel_p = [dict(g, part=1) if i == 3 else g for i, g in enumerate(sel)]; adm_p = tmp / 'adm_p'; (adm_p / 'parts').mkdir(parents=True)
            (adm_p / 'manifest.json').write_text(json.dumps(dict(selected=sel_p, parts=[0]))); (adm_p / 'parts' / 'part_000.jsonl.gz').write_bytes(b)
            (adm_p / 'parts' / 'part_000.json').write_text(json.dumps(dict(sha256=hashlib.sha256(b).hexdigest(), manifest_sha256=C.sha(adm_p / 'manifest.json'))))
            if M.adm_rows(adm_p)[0][3]['status'] != 'weight_mismatch':
                bad.append('adm_rows: a row in another part than its manifest row was accepted')
            sub = tmp / 'sub'; (sub / 'parts').mkdir(parents=True); (sub / 'manifest.json').write_text((adm / 'manifest.json').read_text())
            (sub / 'parts' / 'part_000.jsonl.gz').write_bytes(LS.gz_bytes(prow[:3])); (sub / 'parts' / 'part_000.json').write_text((adm / 'parts' / 'part_000.json').read_text())
            try:
                M.adm_rows(sub); bad.append('adm_rows: a substituted partition was accepted')
            except Exception:                                                               # noqa: BLE001
                pass
            ret = tmp / 'ret'; (ret / 'parts').mkdir(parents=True); fman = tmp / 'fman.json'
            fr_rows = [hrow(10 + i, side=s, day=d, review=100., q=10. if i == 0 else 0., M_=10. if i == 0 else 0., fills=[(100., 0, 0, 10., 50., 0)] if i == 0 else [])
                       for i, (d, s) in enumerate([('20261024', 0), ('20261024', 1), ('20261026', 0), ('20261026', 1)])]
            for i, r_ in enumerate(fr_rows):
                r_['stages']['entry']['admission'] = 'would_POST' if i % 2 else 'would_WAIT'
                r_['t_ms'] = t0 + 10 ** 6 * (i // 2); r_['m'] = 10 + i // 2
            fr_rows[0]['fills'] = [dict(t=100., q=10., price=50., fill_time_edge=10. * (51. - 50.))]
            fman.write_text(json.dumps(dict(parts=[dict(worlds=[dict(day=r_['day'], m=r_['m'], t_ms=r_['t_ms']) for r_ in fr_rows[::2]])])))
            fb = LS.gz_bytes(fr_rows); (ret / 'parts' / 'part_000.jsonl.gz').write_bytes(fb)
            (ret / 'parts' / 'part_000.json').write_text(json.dumps(dict(sha256=hashlib.sha256(fb).hexdigest(), manifest_sha256=C.sha(fman))))
            mem = M.membership(ar, M.forced_rows(ret, fman), adm, fman)
            if not (mem['admitted_equal'] and mem['forced_equal'] and mem['weight_mismatch'] == 2):
                bad.append(f'membership: the complete hand cohorts ({mem})')
            fman2 = tmp / 'fman2.json'; fman2.write_text(json.dumps(dict(parts=[dict(worlds=[dict(day='20261024', m=99, t_ms=1)] + json.loads(fman.read_text())['parts'][0]['worlds'])])))
            if M.membership(ar, M.forced_rows(ret, fman), adm, fman2)['forced_equal']:
                bad.append('membership: a missing fork was accepted')
            out = tmp / 'out'; M.measure(out, adm, ret, fman, tmp / 'nomids')
            js = json.loads((out / 'audit.json').read_text())
            names = ('moments', 'strata', 'prepost', 'concentration', 'contributions', 'sources', 'evidence', 'opportunities', 'fills', 'transitions')
            if not (all((out / f'{n}.csv').exists() for n in names) and [p['stage'] for p in js['predicates']] == list(M.STAGES) and js['parts_without_mids'] == [0]
                    and js['membership']['forced_equal']):
                bad.append('measure: the written outputs')
            MO = pd.read_csv(out / 'moments.csv'); fo = MO[(MO.cohort == 'forced') & (MO.stage == 'entry')]
            if not (set(fo.view) >= {'all', 'day', 'side', 'day_side', 'overlap', 'admission', 'overlap_admission'}
                    and int(fo[fo.view == 'overlap'].n.iloc[0]) == 2 and int(fo[(fo.view == 'admission') & (fo.admission == 'would_POST')].n.iloc[0]) == 2
                    and 'overlap' not in set(MO[MO.cohort == 'admitted'].view)):
                bad.append('moments: the forced views')
            FI = pd.read_csv(out / 'fills.csv')
            if not (len(FI) == 2 and set(FI[FI.cohort == 'forced'].ref_status) == {'ok'} and set(FI[FI.cohort == 'admitted'].ref_status) == {'reference_unavailable:no_record'}
                    and abs(float(FI[FI.cohort == 'forced'].fill_mid.iloc[0]) - 51.) < 1e-12):
                bad.append('measure: the fill references of the hand cohorts')
            md = tmp / 'mids'; md.mkdir(); k0 = json.dumps(list(M.GKEY(prow[0]))); mids = {k0: dict(status='ok', mids={'0': dict(mid=51., frame_t=90., recv_t=95.)})}
            rec = dict(mids=mids, sha256=hashlib.sha256(json.dumps(mids, sort_keys=True).encode()).hexdigest(), admission_manifest_sha256=C.sha(adm / 'manifest.json'), sources=M.sources())
            (md / 'mids_000.json').write_text(json.dumps(rec)); M.measure(tmp / 'out3', adm, ret, fman, md); F3 = pd.read_csv(tmp / 'out3' / 'fills.csv'); fa3 = F3[F3.cohort == 'admitted']
            if not (len(fa3) == 1 and fa3.ref_status.iloc[0] == 'ok' and float(fa3.fill_mid.iloc[0]) == 51. and float(fa3.age.iloc[0]) == 10.
                    and json.loads((tmp / 'out3' / 'audit.json').read_text())['parts_without_mids'] == []):
                bad.append('measure: a valid mids record did not bind the admitted fill mid')
            rec['mids'][k0]['mids']['0']['mid'] = 49.; (md / 'mids_000.json').write_text(json.dumps(rec))
            try:
                M.measure(tmp / 'out4', adm, ret, fman, md); bad.append('measure: a tampered mids record was accepted')
            except Exception:                                                               # noqa: BLE001
                pass
            try:
                M.measure(out, adm, ret, fman, tmp / 'nomids'); bad.append('measure: an existing measurement was overwritten')
            except Exception:                                                               # noqa: BLE001
                pass
            ret2 = tmp / 'ret2'; (ret2 / 'parts').mkdir(parents=True); (ret2 / 'parts' / 'part_000.jsonl.gz').write_bytes(fb)
            (ret2 / 'parts' / 'part_000.json').write_text(json.dumps(dict(sha256=hashlib.sha256(fb).hexdigest(), manifest_sha256=C.sha(fman2))))
            try:
                M.measure(tmp / 'out2', adm, ret2, fman2, tmp / 'nomids'); bad.append('measure: incomplete membership was accepted')
            except Exception as e:                                                          # noqa: BLE001
                if 'incomplete membership' not in str(e):
                    bad.append(f'measure: refused incomplete membership for another reason ({e!r})'[:200])
            real_ce = M.committed_equal
            try:
                M.committed_equal = lambda *a_: False
                try:
                    M.run_part(0, out=tmp / 'rp'); bad.append('run_part: an uncommitted module was used')
                except Exception as e:                                                      # noqa: BLE001
                    if 'not committed' not in str(e):
                        bad.append(f'run_part: refused for another reason ({e!r})'[:200])
                if M.frozen() or M.inputs_frozen():
                    bad.append('frozen / inputs_frozen: true without committed copies')
            finally:
                M.committed_equal = real_ce
            if M.inputs_frozen(adm / 'manifest.json', fman) or not M.committed_equal('predictfun/research/pf_pi0_admission.py', M.HERE / 'pf_pi0_admission.py'):
                bad.append('inputs_frozen accepted hand manifests, or committed_equal refused a committed file')
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'binding / writers: stopped ({e!r})'[:220])
    # ---- 6 tables
    try:
        g = pd.DataFrame([dict(cohort='forced', stage='entry', acc_status='ok', day=d, m=i, t_ms=i, side=i % 2, weight=1., admission='would_WAIT', same_as_review=False,
                               P=.4, z=float(i < 2), Eq=1., q=float(i < 2), Eqm=-1., M=-40. if i == 0 else 0., V=0., value=-40. if i == 0 else 1., err_V=40. if i == 0 else -1.,
                               mech=0., err_q=0., err_m=39. if i == 0 else -1., qty_term=0., mov_term=0., mov_prob=0., mov_cond=0., q_prob=0., q_cond=0.,
                               c_m=1. if i < 2 else np.nan, c_q=0. if i < 2 else np.nan, mhat=-2.) for i, d in enumerate(['20261024', '20261024', '20261026', '20261027'])])
        mm = pd.DataFrame(M.moments(g)); a_ = mm[mm.view == 'all'].iloc[0]
        if not (a_.severe_n == 1 and a_.severe_sum_value == -40. and abs(a_.brier_pooled - .25) < 1e-12 and abs(a_.brier - .26) < 1e-12 and a_.n == 4 and a_.fills == 2 and abs(a_.ess - 4.) < 1e-12
                and int(mm[mm.view == 'overlap'].n.iloc[0]) == 2 and abs(a_.sum_err_m - 36.) < 1e-12):
            bad.append('moments: severe sums, the pooled Brier reference, ESS or the overlap view')
        Gt = pd.DataFrame(dict(status=['ok'] * 6, weight=[1., 1., 1., 1., 1., 10.], D_EXPOSURE=[1., 1., 1., 1., 1., 1.], day=['d'] * 6, m=range(6), t_ms=[0] * 6, side=[0] * 6))
        Ot = pd.DataFrame(dict(cohort=['admitted'] * 6, stage=['entry'] * 6, acc_status=['ok'] * 6, err_m=[1.] * 6, weight=Gt.weight, day=['d'] * 6, m=range(6), t_ms=[0] * 6,
                               side=[0] * 6, err_V=[1.] * 6))
        cc = {(c['quantity'], c['weighting']): c for c in M.concentration(Gt, Ot)}
        if not (abs(cc[('D_EXPOSURE', 'unweighted')]['top5_generation_share'] - 5. / 6.) < 1e-12 and abs(cc[('D_EXPOSURE', 'weighted')]['top5_generation_share'] - 14. / 15.) < 1e-12
                and abs(cc[('movement_error_entry', 'weighted')]['top5_world_share'] - 14. / 15.) < 1e-12):
            bad.append('concentration: the weighted / unweighted shares')
        ct = M.contributions(Gt, Ot)
        if not (len(ct) == 1 and ct[0]['weighted_sum_D'] == 15. and ct[0]['share_D'] == 1. and ct[0]['weighted_sum_entry_u'] == 15.):
            bad.append('contributions: the signed day x side sums')
        Ft = pd.DataFrame([dict(cohort='admitted', stage='entry', day='d', side=0, weight=2., q=10., total=15., pre=10., post=5., ref_status='ok', delay_class='[0,100]', age=3.),
                           dict(cohort='admitted', stage='entry', day='d', side=0, weight=1., q=20., total=-6., pre=np.nan, post=np.nan, ref_status='reference_missing',
                                delay_class='>2000', age=None)])
        pp = {(p['view'], p['cell']): p for p in M.prepost(Ft)}; al = pp[('all', 'all')]
        if not (al['fills'] == 2 and al['with_mid'] == 1 and al['missing'] == 1 and al['sum_total'] == 24. and al['sum_pre'] == 20. and al['pre_per_share'] == 1.
                and al['post_per_share'] == .5 and al['q'] == 30. and al['median_age'] == 3.):
            bad.append(f'prepost: per-share units or the missing count ({al})')
        crows = [hrow(0, review=1500., t_ms=t0, ract='PULL'), hrow(1, review=100., t_ms=t0 + 1, ract='KEEP')]
        ctl = M.control(crows)
        if not (ctl['reviews_with_time'] == 2 and ctl['cell_equal_admission'] == 1 and ctl['n'] == 2 and ctl['n_sel'] == 1 and ctl['strata'] == 2):
            bad.append(f"control: the review hour from its own decision time ({ {k: ctl.get(k) for k in ('reviews_with_time', 'cell_equal_admission', 'n', 'n_sel', 'strata')} })")
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'tables: stopped ({e!r})'[:220])
    # ---- 7 predicates
    try:
        def O_(cells_u, c_adm=-5., c_forced=-5., nf=12):
            rows_ = []
            for (d, s), u in cells_u.items():
                for i in range(40):
                    z = 1. if i < nf else 0.
                    rows_.append(dict(cohort='admitted', stage='entry', acc_status='ok', day=d, m=hash((d, s, i)) % 10 ** 6, t_ms=i, side=s, weight=1., err_m=u, z=z, c_m=c_adm if z else np.nan))
            for i in range(40):
                z = 1. if i < nf else 0.
                rows_.append(dict(cohort='forced', stage='entry', acc_status='ok', day='20261026', m=10 ** 7 + i, t_ms=i, side=0, weight=1., err_m=0., z=z, c_m=c_forced if z else np.nan))
                rows_.append(dict(cohort='forced', stage='entry', acc_status='ok', day='20261024', m=2 * 10 ** 7 + i, t_ms=i, side=0, weight=1., err_m=0., z=z, c_m=50. if z else np.nan))
            return pd.DataFrame(rows_)
        ST_ = lambda dec, dly, n=40, fills=12: ([dict(cohort='admitted', subset='all', stage='entry', dim=d, cell=str(j), n=n, fills=fills, c_m=float(v)) for d, vs in dec.items()
                                                  for j, v in enumerate(vs)]
                                                 + [dict(cohort='admitted', subset='all', stage='entry', dim='delay_class', cell=str(j), n=n, fills=fills, c_m=float(v)) for j, v in enumerate(dly)]
                                                 + [dict(cohort='admitted', subset='all', stage='entry', dim='delay_class', cell='no_fill', n=n, fills=fills, c_m=999.),
                                                    dict(cohort='admitted', subset='overlap', stage='entry', dim='hbin', cell='x', n=n, fills=fills, c_m=999.)])
        CO = lambda s: [dict(quantity='movement_error_entry', weighting='weighted', top5_world_share=s), dict(quantity='movement_error_entry', weighting='unweighted', top5_world_share=.9)]
        cellsB = {('20261026', 0): 5., ('20261026', 1): 5., ('20261027', 0): 5., ('20261027', 1): -1.}
        P1 = M.predicates(O_(cellsB), ST_(dict(hbin=[0, 10], qbin=[0, 2], obin=[1, 3]), [0, 4]), CO(.2))
        P2 = M.predicates(O_(cellsB, c_adm=-5., c_forced=5.), ST_(dict(hbin=[0, 1], qbin=[0, 1], obin=[0, 1]), [0, 9]), CO(.7))
        P3 = M.predicates(O_(cellsB, c_adm=-5., c_forced=-4.8, nf=5), ST_(dict(hbin=[0, 1], qbin=[0], obin=[1]), [], fills=12), CO(.2))
        P4 = M.predicates(O_({('20261026', 0): 5., ('20261026', 1): 5., ('20261027', 0): -1., ('20261027', 1): -1.}), ST_(dict(hbin=[0, 1], qbin=[0, 1], obin=[0, 1]), [0, 1]), CO(.2))
        P5 = M.predicates(O_(cellsB, c_adm=-5., c_forced=-2.4), ST_(dict(hbin=[0, 1], qbin=[0, 1], obin=[0, 1]), [0, 1.2]), CO(.2))
        P6 = M.predicates(O_(cellsB), ST_(dict(hbin=[0, 1], qbin=[0, 1], obin=[0, 1]), [0, 1], fills=5), CO(.2))
        P7 = M.predicates(O_(cellsB), ST_(dict(hbin=[0, 1], qbin=[0, 1], obin=[0, 1]), [0, 9]), CO(.2))
        chk = [(P1, dict(i='broad', ii='decision_state', iii='similar', iv='populated', recommendation='calibration_candidate_defensible')),
               (P2, dict(i='concentrated', ii='fill_timing', iii='material', recommendation='representation_or_data_needed')),
               (P3, dict(iii='inconclusive', iv='thin', recommendation='representation_or_data_needed')),
               (P4, dict(i='inconclusive', ii='both', iv='populated', recommendation='inconclusive')),
               (P5, dict(i='broad', ii='both', iii='material', recommendation='calibration_candidate_defensible')),
               (P6, dict(ii='inconclusive', iv='thin', recommendation='representation_or_data_needed')),
               (P7, dict(i='broad', ii='fill_timing', iv='populated', recommendation='representation_or_data_needed'))]
        for j, (P, want) in enumerate(chk, 1):
            got = {k: (P[k] if k == 'recommendation' else P[k]['answer']) for k in want}
            if got != want:
                bad.append(f'predicates P{j}: {got} != {want}')
        if len(M.evidence_rows(P1)) != 5 or M.evidence_rows(P1)[-1]['answer'] != 'calibration_candidate_defensible':
            bad.append('evidence_rows')
        fr = np.asarray([[0., 0., 50., 0, 0, 0, 0, 100, 0, 0, 0, 0, 52.] + [0.] * 9, [100., 105., 51., 0, 0, 0, 0, 100, 0, 0, 0, 0, 53.] + [0.] * 9,
                         [200., 201., -1., 0, 0, 0, 0, 0, 0, 0, 0, 0, 53.] + [0.] * 9])
        if not (M.mid_lookup(fr, 150.) == dict(mid=52., frame_t=100., recv_t=105.) and M.mid_lookup(fr, 100.) == dict(mid=52., frame_t=100., recv_t=105.)
                and M.mid_lookup(fr, 250.) is None and M.mid_lookup(fr, -1.) is None):
            bad.append('mid_lookup: the convention')
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'predicates: stopped ({e!r})'[:220])
    if bad:
        sys.exit(FAIL + ': ' + '; '.join(bad))
    print('movement audit selftest passed: accounting (calibrated mixture, probability vs head attribution, refusals); fills (pre / post split, effect and gid cut, '
          'delay classes, missing / unavailable / late references, ask signs, retained mids, moved fill time / END mark refused); strata boundaries, the client queue '
          'and the explanatory venue queue; quote-only / clock-only sources; the manifest binding, substituted partitions and membership; moments views, severe sums, '
          'Brier reference, prepost units, concentration, contributions, the control hour; the evidence predicates and recommendation; mid_lookup; writers and the freeze guards')
