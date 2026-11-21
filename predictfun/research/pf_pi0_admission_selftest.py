"""SELFTEST of pf_pi0_admission (PI0-ADMISSION-EXPOSURE-v1; 2026-11-21, session aea5bfb9). run(M) checks the module object M; any failed check
exits with FAIL. Synthetic only (plus two read-only reconciliations against the frozen consumed outputs):
 1. census / select on hand traces: POST generations only, the decision's UTC cell, the four lowest declared ranks, all if fewer, the weights;
 2. the factual episode on the cached synthetic world (stub models) equals an independent pf_branch_engine.step run of pf_adv2.Teacher0 in
    every decision and fill; world_rows on its own POST generations (a first one, nonzero / delayed gids with inventory and opposite orders,
    the latest one near END): every row ok with every check true, the held fills equal to the KEEP arm's, the stages captured; refusals: a
    point PI0 did not POST (not_admitted), a wrong gid (gid_mismatch), a different saved decisions hash (factual_not_reproduced) and END
    (horizon_mismatch);
 3. the measurements on hand rows: the perfect-mean mixture (MSE of the exact mean below zero's while its MAE is above), the weighted
    economics, the weighted matched null with its free strata, the new-observation classification, the predicates and readings; the
    cross-tab and RETENTION v2 against the frozen consumed outputs (768 joined, the review's MSE values);
 4. the writers: build_manifest (immutable) and measure (immutable, a changed partition refused) on hand partitions; run_part refuses an
    uncommitted manifest."""
import copy
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

FAIL = 'PI0 ADMISSION SELFTEST FAILED'


def run(M):  # noqa: C901
    bad = []; BE, L, LS, R2, T, pd, C, RT = M.BE, M.L, M.LS, M.R2, M.T, M.pd, M.C, M.RT
    import pf_r2_labels_selftest as LT
    # ---- 1 census / select
    try:
        t0 = 1793000000000
        tr = [[0., 0, 'absent', 'POST', 0], [0., 1, 'absent', 'WAIT', None], [100., 0, 'live', 'PULL', 0], [250., 0, 'absent', 'POST', 1], [300., 1, 'absent', 'POST', 2]]
        rows = [dict(day='20261026', m=7, t_ms=t0 + 3600000 * h, hour=h, status='ok', episodes=dict(PI0=dict(trace=dict(decision_list=tr)))) for h in range(2)]
        rows += [dict(day='20261026', m=8 + j, t_ms=t0 + 60000 * j, hour=0, status='ok', episodes=dict(PI0=dict(trace=dict(decision_list=tr[:1])))) for j in range(4)]
        rows.append(dict(day='20261026', m=99, t_ms=t0, hour=0, status='world:x'))
        gens, worlds = M.census([(0, {}, rows)]); sel, cells = M.select(gens)
        c0 = tuple(M.cell_of(t0, 0., 0)); g0 = [g for g in gens if tuple(g['cell']) == c0]
        want0 = sorted(g0, key=lambda g: (hashlib.sha256(('pi0_admission_exposure_v1|' + LS.ckey(dict(day=g['day'], m=g['m'], t_ms=g['t_ms'], side=g['side'], gid=g['gid'],
                                                                                                            decision_ms=g['decision_ms']))).encode()).hexdigest(), M.key_of(g)))[:4]
        s0 = [g for g in sel if tuple(g['cell']) == c0]
        if not (len(gens) == 2 * 3 + 4 and len(worlds) == 7 and len(g0) == 6 and [M.key_of(g) for g in s0] == [M.key_of(g) for g in want0]
                and all(abs(g['weight'] - 6. / 4.) < 1e-12 for g in s0) and abs(sum(g['weight'] for g in sel) - len(gens)) < 1e-9 and M.PER_CELL == 4
                and M.DOMAIN == 'pi0_admission_exposure_v1|' and M.cell_of(t0 + 1599950, 100., 1)[2] == (M.cell_of(t0, 0., 1)[2] + 1) % 24 and M.cell_of(t0 + 1599950, 0., 1)[2] == M.cell_of(t0, 0., 1)[2]):
            bad.append(f'census / select: {len(gens)} generations, {len(s0)} in the first cell')
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'census / select: stopped ({e!r})'[:220])
    # ---- 2 the factual episode and world_rows on the synthetic world
    try:
        wj = json.loads((Path(os.environ['PF_TEACHER_SYN']) / 'world.json').read_text()); day = '20261024'
        W = L.worlds_of_day(day, Path(wj['kp']), 2); _, prs, fair, depth, ff = L.world_inputs(day, W, R2.KP.Syn(), R2.X.syn_inputs, R2.X.syn_features())
        Mst, sst = LT.stubs(-12.), LT.OpenSupport(); done = False
        for w in W:
            wb = L.world_bind(day, w, prs, fair, depth, ff)[0]
            if wb is None or done:
                continue
            probe, cst = T.causal_probe(35); pol = M.A.Teacher0(Mst, sst); pol.causal = probe; B = BE.Branch(wb['path'], 0, ident=R2.episode_ident(day, w['o'][0])); pol.bind(B, wb)
            for k in range(M.NT):
                BE.step(B, k, pol)
            Lx = BE.finish(B); posts = [d for d in M.RT.jl(B.decisions) if d[2] == 'absent' and d[3] == 'POST']
            if len(posts) < 3:
                continue
            saved = dict(fills_sha256=L._h(Lx['fills']), decisions_sha256=L._h(Lx['decisions']), value=float(Lx['value']), teacher=dict(calls=len(pol.log)), end=float(B.end),
                         settings=dict(B.settings), mark=float(B.mark), decision_list=M.RT.jl(B.decisions))
            w_row = dict(episodes=dict(PI0=dict(trace=saved))); tms = int(round(float(w['t'])))
            mk = lambda d: dict(day=day, m=int(w['m']), t_ms=tms, side=int(d[1]), gid=int(d[4]), decision_ms=int(round(float(d[0]))), weight=1.5, n_cell=6,
                                cell=list(M.cell_of(tms, float(d[0]), int(d[1]))))
            pick = [posts[0]] + [d for d in posts[1:] if int(d[4]) > 0 and float(d[0]) > 0][:2] + [posts[-1]]
            sel = [mk(d) for d in {(float(d[0]), int(d[1])): d for d in pick}.values()]
            Bf, Lf, pf, cf, sn, ps = M.factual(wb, day, w, Mst, sst, {(int(g['decision_ms']), int(g['side'])) for g in sel})
            if not (L._h(Lf['fills']) == saved['fills_sha256'] and L._h(Lf['decisions']) == saved['decisions_sha256'] and len(pf.log) == saved['teacher']['calls']):
                bad.append('factual: the snapshotting loop differs from pf_branch_engine.step')
            rr, _ = M.world_rows(day, w_row, wb, w, Mst, sst, sel)
            for r in rr:
                ck = r.get('checks', {})
                if not (r['status'] == 'ok' and all(v_ is not False for v_ in ck.values()) and r['stages']['entry'].get('status') == 'ok'
                        and r['stages']['entry'].get('admission') == 'would_POST' and r['held']['fills'] == [x for x in r['keep_arm']['fill_list'] if int(x[1]) == r['gid']]
                        and abs(r['D_EXPOSURE'] - (r['D_selected'] + r['D_focal_rest'] + r['D_opposite'])) <= 1e-6):
                    bad.append(f"world_rows: generation gid {r['gid']} at {r['decision_ms']} ({r['status']}, {ck})"[:300])
            for r in rr:
                fr_ = r['stages']['first_review']
                if fr_.get('status') == 'ok' and not (fr_['target']['eff'] == fr_['effect'] and r['stages']['entry']['target']['eff'] == r['held']['act']):
                    bad.append('world_rows: a stage target that does not start at its effect / activation')
            tested = False
            for g_ in sel:
                key0 = (int(g_['decision_ms']), int(g_['side'])); Bk, Lk_, pk, ck_, d0_, g0_ = M.keep_arm(sn[key0], wb, Mst, sst)
                if not M.keep_prefix_ok(Bk, Bf, pk, None):
                    bad.append('keep_prefix_ok: a valid KEEP arm refused')
                ov_ = [d for d in pk.log if d.get('status') == 'keep_selected_override']
                if ov_:
                    tested = True; break
            if tested:
                Bk2 = copy.deepcopy(Bk); Bk2.E.fills.insert(0, (float(ov_[0]['t']) - 1., g0_, int(g_['side']), 1., 50., 0))
                if M.keep_prefix_ok(Bk2, Bf, pk, None):
                    bad.append('keep_prefix_ok: a fill before the override that the factual lacks was accepted')
            else:
                bad.append('keep_arm: no selected generation has an override to test the prefix with')
            real_hold = M.R2.hold_from
            def fake_hold(B_, k_, s_):
                gid_, L_, gen_ = real_hold(B_, k_, s_); L_ = dict(L_, fills=list(L_['fills']) + [(4999., gid_, s_, 1., 50., 0)]); return gid_, L_, gen_
            M.R2.hold_from = fake_hold
            try:
                gotq = M.world_rows(day, w_row, wb, w, Mst, sst, sel[:1])[0][0]['status']
            finally:
                M.R2.hold_from = real_hold
            if gotq != 'target_not_equivalent':
                bad.append(f'world_rows: a held fill the KEEP arm lacks gave {gotq}')
            hf = [[10., 3, 0, 5., 50., 0], [200., 3, 0, 5., 50., 1]]
            if not (M.reconcile_pull(4., 4., hf[:1], hf, 137., 137.) and not M.reconcile_pull(4., 5., hf[:1], hf, 137., 137.) and not M.reconcile_pull(4., 4., hf, hf, 137., 137.)
                    and not M.reconcile_pull(4., 4., hf[:1], hf, 137., 150.)):
                bad.append('reconcile_pull: the value, the pre-effect fills or the removal time')
            if not any(r['gid'] > 0 and r['decision_ms'] > 0 for r in rr):
                bad.append('world_rows: no nonzero / delayed generation exercised')
            live = next((d for d in M.RT.jl(B.decisions) if d[2] == 'live'), None)
            wrong = [dict(sel[0], decision_ms=int(round(float(live[0]))), side=int(live[1]))] if live else []
            for lab, sel_, sv, want in (('a point PI0 did not POST', wrong, saved, 'not_admitted'), ('a wrong gid', [dict(sel[0], gid=sel[0]['gid'] + 50)], saved, 'gid_mismatch'),
                                        ('a different decisions hash', sel[:1], dict(saved, decisions_sha256='0' * 64), 'factual_not_reproduced'),
                                        ('a different END', sel[:1], dict(saved, end=6000.), 'horizon_mismatch')):
                if not sel_:
                    bad.append(f'{lab}: no case built'); continue
                got = M.world_rows(day, dict(episodes=dict(PI0=dict(trace=sv))), wb, w, Mst, sst, sel_)[0][0]['status']
                if got != want:
                    bad.append(f'{lab}: {got}, not {want}')
            done = True
        if not done:
            bad.append('no synthetic world with three PI0 POST generations')
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'world_rows: stopped ({e!r})'[:220])
    # ---- 3 the measurements on hand rows
    try:
        rng = np.random.default_rng(5); n = 100; vals = np.where(np.arange(n) < 30, -100., 0.)
        S = pd.DataFrame(dict(day='20261026', m=np.arange(n), t_ms=0, side=0, gid=0, weight=1., hour=np.arange(n) % 4, stage='entry', fork_status='ok', stage_status='ok', dec_status='model',
                              action='POST', t=0., P=.3, Eq=9., Eqm=-30., V=-30., edge=0., mech=0., q_fill=30., qm_fill=-100., q=np.where(vals < 0, 30., 0.), value=vals, qm=vals))
        fm = {f['stage']: f for f in M.forecast_metrics(S) if f['day'] == 'all'}['entry']
        if not (abs(fm['mse_V'] - 2100.) < 1e-9 and abs(fm['mse_zero'] - 3000.) < 1e-9 and abs(fm['mae_V'] - 42.) < 1e-9 and abs(fm['mae_zero'] - 30.) < 1e-9 and abs(fm['mean_error']) < 1e-9):
            bad.append(f'the perfect-mean mixture: {fm}')
        Gh = pd.DataFrame([dict(day=d, m=i, t_ms=0, side=0, status='ok', weight=wt, D_EXPOSURE=dv, D_selected=dv, D_focal_rest=0., D_opposite=0., overrides=1,
                                **{f'{a}_{k}': 0. for a in ('factual', 'keep') for k in ('filled', 'lifetime_ms', 'end_value')})
                           for d in ('20261026', '20261027') for i, (wt, dv) in enumerate([(1., 2.), (3., -1.)] * 20)])
        Gh['keep_filled'] = 10.; Gh['factual_filled'] = 0.
        Gh = pd.concat([Gh, pd.DataFrame([dict(Gh.iloc[0].to_dict(), status='world:x', D_EXPOSURE=999.)])], ignore_index=True)
        ec = {(e['day'], e['side']): e for e in M.economics(Gh)}[('all', 'all')]
        if not (abs(ec['wmean_D'] - (20 * 2. - 60 * 1.) * 2 / (80 * 2)) < 1e-12 and ec['n'] == 80 and ec['n_pos'] == 40):
            bad.append(f"economics: weighted mean {ec['wmean_D']}")
        Sr = pd.DataFrame([dict(day='20261026', side=0, hour=h, stage='first_review', fork_status='ok', stage_status='ok', dec_status='model', V=-1., weight=1. + (j % 2),
                                value=float(j) - 20., action='PULL' if j < 10 else 'KEEP') for h in range(4) for j in range(20)])
        se = M.selection(Sr)
        if not (se['percentile'] is not None and se['percentile'] <= .01 and se['free_strata'] == 4 and se['free_rows'] == 80 and not se['thin'] and se['n_sel'] == 40):
            bad.append(f'selection: a planted protective selection ({se})')
        se2 = M.selection(Sr.assign(action='PULL'))
        if not (se2['free_strata'] == 0 and se2['thin']):
            bad.append('selection: an all-PULL population is not degenerate / thin')
        F_ = M.RT.FEATS; x0 = [0.] * 107; x1 = list(x0); x1[F_.index('pf_age')] = 100.; x1[F_.index('bn_age')] = 3.; x1[F_.index('bnd_age')] = 100.
        mo = lambda V: dict(P=.5, Eq=5., Eqm=0., V=V, edge=0., mech=0., q_fill=10., qm_fill=0.)
        fork = dict(day='20261026', m=1, t_ms=1, side=0, gid=0, weight=2., cell=['20261026', 0, 3], status='ok', stages=dict(
            entry=dict(status='ok', dec_status='model', action='POST', t=0., inputs=x0, view=dict(received=10.), moments=mo(2.), target=dict(q=0., value=0., qm=0.)),
            first_review=dict(status='ok', dec_status='model', action='PULL', t=100., inputs=x1, view=dict(received=10.), moments=mo(-2.), target=dict(q=0., value=0., qm=0.)),
            first_pull=dict(status='no_pull')))
        x2 = list(x0); x2[F_.index('pf_age')] = 100.; x2[F_.index('bn_age')] = 100.; x2[F_.index('bnd_age')] = 100.
        fork2 = copy.deepcopy(fork); fork2['m'] = 2; fork2['stages']['first_review'].update(inputs=x2, view=dict(received=60.))
        S2x = M.stage_rows([fork2]); r2 = S2x[S2x.stage == 'first_review'].iloc[0]
        if not (r2.new_pf_receipt == True and r2.any_new == True and r2.new_pf_age == False and r2.new_bn_age == False):  # noqa: E712
            bad.append('stage_rows: a new PF receipt with aging-only ages')
        Sx = M.stage_rows([fork]); rv = Sx[Sx.stage == 'first_review'].iloc[0]
        if not (rv.new_pf_age == False and rv.new_bn_age == True and rv.new_bnd_age == False and rv.new_pf_receipt == False and rv.any_new == True and rv.dV == -4.):  # noqa: E712
            bad.append('stage_rows: the new-observation classification (ages in ms grow by the elapsed time unless a new observation arrived)')
        Gp = Gh.copy(); P1 = M.predicates(Gp, S, [dict(stage='entry', day='all', mse_V=1., mse_zero=2.), dict(stage='first_review', day='all', mse_V=3., mse_zero=2.)])
        Gn = Gh.assign(D_EXPOSURE=-Gh.D_EXPOSURE.abs()); P2 = M.predicates(Gn, S, [dict(stage='entry', day='all', mse_V=1., mse_zero=2.), dict(stage='first_review', day='all', mse_V=3., mse_zero=2.)])
        Gt = Gn.assign(keep_filled=0.); P3 = M.predicates(Gt, S, [dict(stage='entry', day='all', mse_V=1., mse_zero=2.), dict(stage='first_review', day='all', mse_V=3., mse_zero=2.)])
        Gm = Gh.copy(); Gm.loc[Gm.day == '20261027', 'D_EXPOSURE'] = 5.; P4 = M.predicates(Gm, S, [dict(stage='entry', day='all', mse_V=1., mse_zero=2.), dict(stage='first_review', day='all', mse_V=3., mse_zero=2.)])
        if not (P2['readings']['admitted_longer_harmful'] and not P2['readings']['not_identified'] and P3['readings']['not_identified'] and P3['thin']
                and P4['readings']['not_identified'] and P1['retention_v2'] == dict(entry_mse_beats_zero=True, review_mse_beats_zero=False) and P1['mse_beats_zero']['entry'] and M.THIN == 30):
            bad.append('predicates: harmful / thin / days disagreeing / RETENTION v2')
        Gb = Gh.assign(D_EXPOSURE=Gh.D_EXPOSURE.abs()); P5 = M.predicates(Gb, S, [dict(stage='entry', day='all', mse_V=1., mse_zero=2.), dict(stage='first_review', day='all', mse_V=3., mse_zero=2.)])
        P6 = M.predicates(Gb.assign(keep_filled=0.), S, [dict(stage='entry', day='all', mse_V=1., mse_zero=2.), dict(stage='first_review', day='all', mse_V=3., mse_zero=2.)])
        if not P5['readings']['admitted_longer_beneficial'] or P6['readings']['admitted_longer_beneficial'] or not P6['readings']['not_identified']:
            bad.append('predicates: both days positive is not beneficial, or a thin positive sample is')
        ct, nj, nmiss = M.crosstab(); cls = {c_['cls']: c_['n'] for c_ in ct if c_['day'] == 'all'}
        if not (nj == 768 and nmiss == 0 and cls.get('excluded') == 12 and cls.get('would_POST') == 43 and cls.get('fallback') == 1 and cls.get('would_WAIT') == 712):
            bad.append(f'crosstab: {nj} joined, {nmiss} missing, classes {cls}')
        rv2, rf = M.retention_v2(); e2 = {(r['stage'], r['day']): r for r in rv2}
        if not (abs(e2[('entry', 'all')]['mse_V'] - 4690.44) < .01 and abs(e2[('entry', 'all')]['mse_zero'] - 4785.28) < .01 and abs(e2[('entry', 'all')]['mse_mech'] - 4985.39) < .01
                and abs(e2[('first_review', 'all')]['mse_V'] - 4794.91) < .01 and rf['both_modeled'] == 717 and rf['flips'] == 34):
            bad.append(f"retention_v2: {e2[('entry', 'all')]['mse_V']:.2f} / {rf}")
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'measurements: stopped ({e!r})'[:220])
    # ---- 4 the writers
    try:
        mo4 = dict(P=.5, Eq=5., Eqm=0., V=1., edge=0., mech=0., q_fill=10., qm_fill=0.)
        fork = dict(status='ok', stages=dict(entry=dict(status='ok', dec_status='model', action='POST', t=0., inputs=[0.] * 107, view=dict(received=1.), moments=mo4,
                                                        target=dict(q=0., value=0., qm=0.)), first_review=dict(status='no_review'), first_pull=dict(status='no_pull')))
        with tempfile.TemporaryDirectory(dir=str(R2.I.K.IO)) as tmp:
            tmp = Path(tmp); out = tmp / 'out'
            M.build_manifest(out, parts=[(0, {}, rows)])
            try:
                M.build_manifest(out, parts=[(0, {}, rows)]); bad.append('build_manifest: an existing manifest was overwritten')
            except Exception:                                                               # noqa: BLE001
                pass
            man = json.loads((out / 'manifest.json').read_text()); (out / 'parts').mkdir(exist_ok=True)
            prow = [dict(fork, day=g['day'], m=g['m'], t_ms=g['t_ms'], gid=g['gid'], side=g['side'], decision_ms=g['decision_ms'], weight=g['weight'], n_cell=g['n_cell'], cell=g['cell'], D_EXPOSURE=1.,
                         D_selected=1., D_focal_rest=0., D_opposite=0., R_factual=0., R_keep=1., overrides=0, gen_factual=dict(filled=0.), gen_keep=dict(filled=0.)) for g in man['selected']]
            b = LS.gz_bytes(prow); (out / 'parts' / 'part_000.jsonl.gz').write_bytes(b)
            (out / 'parts' / 'part_000.json').write_text(json.dumps(dict(sha256=hashlib.sha256(b).hexdigest(), sources=M.sources(), manifest_sha256=C.sha(out / 'manifest.json'))))
            M.measure(out); js = json.loads((out / 'admission.json').read_text())
            if not (js['generations'] == len(man['selected']) and all((out / f'{n_}.csv').exists() for n_ in ('generations', 'stages', 'economics', 'forecasts', 'bins', 'transitions',
                                                                                                                 'crosstab', 'retention_v2'))):
                bad.append('measure: the written outputs')
            try:
                M.measure(out); bad.append('measure: an existing measurement was overwritten')
            except Exception:                                                               # noqa: BLE001
                pass
            (out / 'admission.json').unlink(); (out / 'parts' / 'part_000.jsonl.gz').write_bytes(LS.gz_bytes(prow[:-1] + prow[:1]))
            try:
                M.measure(out); bad.append('measure: a partition whose bytes differ from its record was read')
            except Exception:                                                               # noqa: BLE001
                pass
            try:
                M.run_part(0, out=out); bad.append('run_part: an uncommitted manifest was used')
            except Exception as e:                                                          # noqa: BLE001
                if 'manifest is not the committed one' not in str(e):
                    bad.append(f'run_part: refused for another reason ({e!r})'[:200])
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'writers: stopped ({e!r})'[:220])
    if bad:
        sys.exit(FAIL + ': ' + '; '.join(bad))
    print('pi0 admission selftest passed: census / select; the factual loop equals pf_branch_engine.step; world_rows on first, nonzero / delayed and '
          'late generations with four refusals; the perfect-mean mixture, weighted economics, the weighted null, new observations, predicates, the '
          'cross-tab and RETENTION v2 against the frozen outputs; writers (manifest and measure immutable, a changed partition refused, run_part refused)')
