"""SELFTEST of pf_coverage (PI0-COVERAGE-v1; 2026-11-21, session aea5bfb9). run(M) checks the module object M; any failed check exits with FAIL.
 1. population: sample_worlds (per hour the lowest ranks, ties by key, pi_world, all worlds of a thin hour), pilot_of, partitions (every world once,
    blocks of BLOCK_HOURS), build_manifest (hand worlds / identities, write-once);
 2. categorize: every branch; sample_queries: at most PER_CELL per cell by rank, the cell by the world's stratum hour, pi_cell, weights, no
    uncategorized row; ledger_ns / reconcile_arm; recompute / reconcile (fills at or after eff; a changed value refused);
 3. the synthetic world (PF_TEACHER_SYN, stub forecasts): roll() equals pf_branch_engine.step decision for decision; its categories match the
    executed decisions; labels() on queries of every category passes every check; falsifiers - other census hashes (reroll_mismatch), a hold
    ledger with an unbooked fill (ledger_ok), a hold with another gid (gid), a POST query relabelled WAIT (factual);
 4. measurement on hand rows: frame / eligibility, weighted forecasts against independent arithmetic, coverage, actions (degenerate within a
    factual source, varying on the combined populations), the count-preserving null, predicates (covered / lacking, concentration,
    POSITIVE / NOT_POSITIVE / INCONCLUSIVE);
 5. guards: run_part / run_pilot / measure refuse before the freeze, run_part without an allowing projection, run_pilot write-once.
PF_TEACHER_SYN: the cached synthetic world (pf_r2_teacher_selftest.build_world)."""
import copy
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

FAIL = 'COVERAGE SELFTEST FAILED'


def run(M):  # noqa: C901
    bad = []; pd, LS, L, BE, R2, T = M.pd, M.LS, M.L, M.BE, M.R2, M.T
    # ---- 1 population
    try:
        o = lambda oid: types.SimpleNamespace(oid=oid)
        Wd = [dict(day='20261024', m=m, t=float(1792800000000 + h * 3600000 + m), t_ms=1792800000000 + h * 3600000 + m, hour=h, o={0: o(2 * m), 1: o(2 * m + 1)})
              for h in (0, 1, 2, 5) for m in range(12 if h != 5 else 3)]
        strata, sel = M.sample_worlds(Wd)
        by = {}
        for w in Wd:
            by.setdefault(w['hour'], []).append(w)
        want = sorted(LS.ckey(LS.wkey(w)) for h, ws in by.items() for w in sorted(ws, key=lambda w: (M.rank('world', LS.wkey(w)), LS.ckey(LS.wkey(w))))[:M.PER_HOUR])
        if not (sorted(LS.ckey(LS.wkey(x)) for x in sel) == want and {s['hour']: (s['N'], s['n']) for s in strata} == {0: (12, 8), 1: (12, 8), 2: (12, 8), 5: (3, 3)}
                and all(abs(x['pi_world'] - (8 / 12 if x['hour'] != 5 else 1.)) < 1e-12 for x in sel) and sel == sorted(sel, key=lambda x: (x['day'], x['m'], x['t_ms']))):
            bad.append('sample_worlds: the per-hour ranks / pi / thin hour')
        pil = M.pilot_of(sel, '20261024')
        if [LS.ckey(LS.wkey(x)) for x in pil] != [LS.ckey(LS.wkey(x)) for x in sorted(sel, key=lambda x: (x['rank'], LS.ckey(LS.wkey(x))))[:M.PILOT_PER_DAY]]:
            bad.append('pilot_of: not the lowest-ranked worlds of the day')
        parts = M.partitions(sel); mem = [LS.ckey(k) for p in parts for k in p['members']]
        if not (len(mem) == len(set(mem)) == len(sel) and all(p['hours'][1] - p['hours'][0] == M.BLOCK_HOURS - 1 for p in parts)
                and all(all(p['hours'][0] <= next(x['hour'] for x in sel if LS.wkey(x) == k) <= p['hours'][1] for k in p['members']) for p in parts)):
            bad.append('partitions: a world twice / missing / outside its block')
        with tempfile.TemporaryDirectory(dir=str(M.KP.I.K.IO)) as tmp:
            mp = Path(tmp) / 'man.json'; man = M.build_manifest(mp, worlds_fn=lambda d: [dict(w, day=d) for w in Wd], identity_fn=lambda d: dict(day=d))
            if not (set(man['sample']) == set(M.DAYS) and man['sample']['20261031']['inputs'] == dict(day='20261031') and len(man['partitions']) == 6 * len(parts)
                    and man['per_hour'] == M.PER_HOUR and man['per_cell'] == M.PER_CELL and man['domain'] == M.DOMAIN):
                bad.append('build_manifest: the frozen population')
            try:
                M.build_manifest(mp, worlds_fn=lambda d: [], identity_fn=lambda d: {}); bad.append('build_manifest: overwritten')
            except Exception:                                                               # noqa: BLE001
                pass
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'population: stopped ({e!r})'[:220])
    # ---- 2 categorize, sample_queries, ledgers
    try:
        mdl = lambda a: dict(status='model', action=a)
        cases = [(('post_pending', None, None, False), (None, 'post_pending')), (('live', mdl('KEEP'), [0, 0, 'live', 'KEEP', 1], False), (None, 'live_later')),
                 (('absent', None, [0, 0, 'absent', 'WAIT', None], False), (None, 'engine_fallback')), (('absent', dict(status='cap_blocked', action='WAIT'), [0, 0, 'absent', 'WAIT', None], False), (None, 'cap_blocked')),
                 (('absent', mdl('POST'), [0, 0, 'absent', 'WAIT', None], False), (None, 'constrained')), (('absent', mdl('POST'), [0, 0, 'absent', 'POST', 3], False), ('absent_POST', 'model')),
                 (('absent', mdl('WAIT'), [0, 0, 'absent', 'WAIT', None], False), ('absent_WAIT', 'model')), (('live', mdl('KEEP'), [0, 0, 'live', 'KEEP', 1], True), ('review_KEEP', 'model')),
                 (('live', mdl('PULL'), [0, 0, 'live', 'PULL', 1], True), ('review_PULL', 'model')), (('live', None, [0, 0, 'live', 'KEEP', 1], True), (None, 'engine_fallback'))]
        for args, want in cases:
            if M.categorize(*args) != want:
                bad.append(f'categorize{args[:2]}: {M.categorize(*args)} != {want}'[:200])
        selk = {('20261024', 1, 10): dict(hour=3, pi_world=.5), ('20261024', 2, 20): dict(hour=3, pi_world=.25)}
        cr = [dict(day='20261024', m=m, t_ms=t0, side=0, t=t, cat='absent_WAIT', gid=None, created_at=None) for (m, t0) in ((1, 10), (2, 20)) for t in range(0, 300, 50)]
        cr += [dict(day='20261024', m=1, t_ms=10, side=1, t=t, cat=None, gid=None, created_at=None) for t in (0, 50)] + [dict(day='20261024', m=2, t_ms=20, side=0, t=500, cat='review_PULL', gid=4, created_at=[100, 0])]
        qs, cells = M.sample_queries(cr, selk); cw = [c for c in cells if c['cat'] == 'absent_WAIT'][0]; qw = [q for q in qs if q['cat'] == 'absent_WAIT']
        key = lambda r: dict(day=r['day'], m=int(r['m']), t_ms=int(r['t_ms']), side=int(r['side']), t=int(r['t']), cat=r['cat'])
        top = sorted([r for r in cr if r['cat'] == 'absent_WAIT'], key=lambda r: (M.rank('query', key(r)), LS.ckey(key(r))))[:M.PER_CELL]
        if not (cw['N'] == 12 and cw['n'] == M.PER_CELL and cw['hour'] == 3 and sorted(LS.ckey(key(q)) for q in qw) == sorted(LS.ckey(key(r)) for r in top)
                and all(abs(q['weight'] - 1. / (q['pi_world'] * (M.PER_CELL / 12.))) < 1e-12 for q in qw) and len(cells) == 2
                and [q for q in qs if q['cat'] == 'review_PULL'][0]['created_at'] == [100, 0] and all(q['cat'] is not None for q in qs)):
            bad.append(f'sample_queries: the cell sample ({cw}, {len(qs)})')
        Lh = dict(fills=[(100., 0, 0, 10., 50., 0), (300., 1, 1, 5., 52., 1)], cash=-(10. * 50.) + 5. * 52., position=10. - 5., orders9=[[0, 0, 50., 30.], [1, 0, 52., 30.]])
        Lh['value'] = Lh['cash'] + Lh['position'] * 51.
        if not (L.reconcile_arm(Lh, M.ledger_ns(Lh, 51.)) and not L.reconcile_arm(dict(Lh, fills=Lh['fills'] + [(400., 0, 0, 1., 50., 2)]), M.ledger_ns(Lh, 51.))):
            bad.append('ledger_ns: reconcile_arm on the hold ledger')
        f = [[100., 3, 0, 10., 50., 0], [137., 3, 0, 5., 50., 1], [200., 4, 0, 5., 49., 2]]; tg = M.RT.stage_target(f, 3, 0, 52., 137., 50.5)
        r_ = M.recompute(f, 3, 0, 52., 137., 50.5)
        if not (M.reconcile(tg, f, 3, 0, 52., 137., 50.5) and r_ == dict(q=5., value=10., qm=7.5) and not M.reconcile(dict(tg, value=tg['value'] + 1.), f, 3, 0, 52., 137., 50.5)):
            bad.append(f'recompute / reconcile: the inclusive effect or a refusal ({r_})')
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'categorize / queries / ledgers: stopped ({e!r})'[:220])
    # ---- 3 the synthetic world
    try:
        import pf_r2_labels_selftest as LT
        wj = json.loads((Path(os.environ['PF_TEACHER_SYN']) / 'world.json').read_text()); day = '20261024'
        W = L.worlds_of_day(day, Path(wj['kp']), 3); _, prs, fair, depth, ff = L.world_inputs(day, W, R2.KP.Syn(), R2.X.syn_inputs, R2.X.syn_features())
        found = None
        for vr in (-12., -30., 0.):
            Mst, sst = LT.stubs(vr), LT.OpenSupport()
            for w in W:
                wb = L.world_bind(day, w, prs, fair, depth, ff)[0]
                if wb is None:
                    continue
                Bc, Lc, pc, cc, rows = M.roll(wb, day, w, Mst, sst); cats = {r['cat'] for r in rows}
                if {'absent_POST', 'review_KEEP'} <= cats or {'absent_POST', 'review_PULL'} <= cats:
                    found = (vr, w, wb, Mst, sst, Bc, Lc, pc, rows)
                    if {'absent_POST', 'absent_WAIT', 'review_KEEP', 'review_PULL'} <= cats:
                        break
            if found and {'absent_POST', 'absent_WAIT', 'review_KEEP', 'review_PULL'} <= {r['cat'] for r in found[8]}:
                break
        if found is None:
            bad.append('synthetic: no world with a POST and a first review')
        else:
            vr, w, wb, Mst, sst, Bc, Lc, pc, rows = found; cats = {r['cat'] for r in rows}
            probe, cst = T.causal_probe(M.AUDIT_EVERY); pol = M.A.Teacher0(Mst, sst); pol.causal = probe; B = BE.Branch(wb['path'], 0, ident=R2.episode_ident(day, w['o'][0])); pol.bind(B, wb)
            for k in range(M.NT):
                BE.step(B, k, pol)
            Lr = BE.finish(B)
            if not (L._h(Lr['fills']) == L._h(Lc['fills']) and L._h(Lr['decisions']) == L._h(Lc['decisions']) and len(pol.log) == len(pc.log)):
                bad.append('roll: the census loop differs from pf_branch_engine.step')
            ex = {(int(round(d[0])), int(d[1])): d for d in M.RT.jl(Lc['decisions'])}
            for r in rows:
                d = ex.get((r['t'], r['side']))
                if r['cat'] == 'absent_POST' and not (d and d[3] == 'POST' and r['executed'] == 'POST'):
                    bad.append('roll: an absent_POST row without an executed POST'); break
                if r['cat'] in ('review_KEEP', 'review_PULL') and not (r['first_review'] and r['created_at'] is not None and d and d[3] == r['cat'].split('_')[1]):
                    bad.append('roll: a review row without its first-review / creation / executed action'); break
            if sum(1 for r in rows if r['first_review']) != len({r['gid'] for r in rows if r['state'] == 'live'}):
                bad.append('roll: first reviews are not once per generation')
            tms = int(round(float(w['t'])))
            qs = []
            for c in ('absent_POST', 'absent_WAIT', 'review_KEEP', 'review_PULL'):
                for r in [r for r in rows if r['cat'] == c][:2]:
                    qs.append(dict(day=day, m=int(w['m']), t_ms=tms, side=int(r['side']), t=int(r['t']), cat=c, hour=0, N_cell=10, n_cell=2, pi_cell=.2, pi_world=.5,
                                   weight=10., rank='x', gid=r['gid'], created_at=r['created_at']))
            hs = [L._h(Lc['fills']), L._h(Lc['decisions'])]; lr, info = M.labels(wb, day, w, Mst, sst, qs, hs)
            for r in lr:
                if not (r['status'] == 'ok' and all(v is not False for v in r['checks'].values())):
                    bad.append(f"labels: {r['cat']} at {r['t']} is {r['status']} {r.get('checks')}"[:240]); break
            for r in lr:
                if r['status'] == 'ok' and r['cat'].startswith('absent') and r['target']['eff'] != r['hold']['act']:
                    bad.append('labels: an absent target not from the hold activation'); break
                if r['status'] == 'ok' and r['cat'].startswith('review') and r['target']['eff'] != r['stage']['effect']:
                    bad.append('labels: a review target not from the modeled effect'); break
            if not {q['cat'] for q in qs} >= {'absent_POST'} or not any(q['cat'].startswith('review') for q in qs):
                bad.append(f"labels: the synthetic world lacks a category ({sorted({q['cat'] for q in qs})})")
            if any(r['status'] != 'reroll_mismatch' for r in M.labels(wb, day, w, Mst, sst, qs[:2], ['x', 'y'])[0]):
                bad.append('labels: other census hashes were accepted')
            real_hold = M.R2.hold_from
            try:
                def unbooked(B_, k_, s_):
                    gid_, L_, gen_ = real_hold(B_, k_, s_); return gid_, dict(L_, fills=list(L_['fills']) + [(4999., gid_, s_, 1., 50., 0)]), gen_
                M.R2.hold_from = unbooked; got = M.labels(wb, day, w, Mst, sst, qs[:1], hs)[0][0]
                if got['status'] != 'ledger_ok_failed':
                    bad.append(f"labels: an unbooked hold fill gave {got['status']}")
                def regid(B_, k_, s_):
                    gid_, L_, gen_ = real_hold(B_, k_, s_); return gid_ + 7, L_, gen_
                M.R2.hold_from = regid; gp = [q for q in qs if q['cat'] == 'absent_POST'][:1]; got = M.labels(wb, day, w, Mst, sst, gp, hs)[0][0]
                if got['status'] not in ('gid_failed',):
                    bad.append(f"labels: a hold of another gid gave {got['status']}")
            finally:
                M.R2.hold_from = real_hold
            with tempfile.TemporaryDirectory(dir=str(M.KP.I.K.IO)) as tmp:
                wk_ = lambda w_: dict(day=day, m=int(w_['m']), t_ms=int(round(float(w_['t']))))
                selx = [dict(wk_(w_), hour=0, pi_world=.5) for w_ in W]; manx = dict(sample={day: dict(selected=selx)})
                binder = lambda: ({LS.ckey(wk_(w_)): w_ for w_ in W}, prs, fair, depth, ff)
                st_ = M.run_members(day, [wk_(w_) for w_ in W], manx, Path(tmp) / 'p', 'selftest', 'msha', binder=binder, assets=lambda: (Mst, sst))
                rec_ = json.loads((Path(tmp) / 'p.json').read_text()); rws = LS.gz_rows((Path(tmp) / 'p_rows.jsonl.gz').read_bytes())
                cns = LS.gz_rows((Path(tmp) / 'p_census.jsonl.gz').read_bytes())
                if not (st_ == 'DONE' and rec_['rows'] == len(rws) > 0 and rec_['census_rows'] == len(cns) and all(isinstance(k_, str) for k_ in rec_['worlds'])
                        and rec_['manifest_sha256'] == 'msha' and all(r_['status'] == 'ok' for r_ in rws) and {r_['cat'] for r_ in rws} <= set(M.CATS)
                        and all(abs(r_['weight'] - 1. / (.5 * r_['pi_cell'])) < 1e-12 for r_ in rws) and len(rec_['timing']['census']) == sum(1 for v_ in rec_['worlds'].values() if v_ == 'ok') and sum(rec_['timing']['label_queries']) == rec_['queries'] == len(rws)):
                    bad.append(f"run_members: the synthetic end-to-end run ({st_}, {rec_.get('rows')}, {sorted({r_['status'] for r_ in rws})})"[:240])
            fk = [dict(q, cat='absent_WAIT') for q in qs if q['cat'] == 'absent_POST'][:1]
            if fk and M.labels(wb, day, w, Mst, sst, fk, hs)[0][0]['status'] != 'factual_failed':
                bad.append('labels: a POST query relabelled WAIT passed')
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'synthetic world: stopped ({e!r})'[:240])
    # ---- 4 measurement
    try:
        rng = np.random.default_rng(3); rows = []
        for i in range(240):
            cat = M.CATS[(i // 2) % 4]; z = float(i % 3 == 0); V = (1. + i % 5) if cat in ('absent_POST', 'review_KEEP') else -(1. + i % 5); P = .3; Eq = 3.; edge = .5; Eqm = V - edge * Eq
            q = 10. * z; Mv = (-20. + i % 7 - (50. if i % 33 == 0 else 0.)) * z; val = edge * q + Mv
            rows.append(dict(day=M.DAYS[i % 2], m=i // 4, t_ms=i, side=i % 2, t=0, cat=cat, hour=(i // 16) % 6, weight=1. + (i % 3), N_cell=8, n_cell=4, status='ok' if i != 7 else 'world:x',
                             gid=i, stage=dict(dec_status='model' if i != 11 else 'fallback', moments=dict(P=P, Eq=Eq, Eqm=Eqm, V=V, edge=edge), inputs=[100. + i] * len(M.RT.FEATS)),
                             target=dict(q=q, value=val, qm=Mv if z else 5., severe=max(-val - 30., 0.))))
        D = M.frame(rows)
        if not (int(D.eligible.sum()) == 238 and not bool(D.eligible.iloc[7]) and not bool(D.eligible.iloc[11]) and (D.M[D.z == 0] == 0).all()):
            bad.append('frame: eligibility / M on no fill')
        FC = M.forecasts(D); f0 = [r for r in FC if r['cat'] == 'absent_POST' and r['day'] == 'all' and r['side'] == 'all'][0]; E = D[D.eligible & (D.cat == 'absent_POST')]
        dsh = (E.weight * (E.V - E.value)).groupby(E.day).sum().abs(); sv_ = (E.weight * E.value.where(E.value < -M.SEVERE, 0.)).groupby(E.day).sum().abs()
        if not ((E.value < -M.SEVERE).any() and f0['max_date_share_severe'] is not None and abs(f0['max_date_share_severe'] - float(sv_.max() / sv_.sum())) < 1e-12):
            bad.append('forecasts: the severe date share')
        if not (abs(f0['max_date_share_err'] - float(dsh.max() / dsh.sum())) < 1e-12 and set(D.qbin) <= {'0', '(0,1]', '(1,3]', '>3', 'missing_or_invalid', 'none'}
                and {c['dim'] for c in M.calibration(D)} == {'P', 'horizon', 'queue', 'ordinal'}):
            bad.append('forecasts / calibration: the date share or the diagnostic bins')
        rf, rsha = M.reference(); rcat = {r['cat'] for r in rf}
        if not (rcat == {'ref_admitted_entry', 'ref_admitted_review'} and len(rsha) == 64 and next(r for r in rf if r['cat'] == 'ref_admitted_entry' and r['day'] == 'all' and r['side'] == 'all')['n'] == 329):
            bad.append(f'reference: the admitted cohort ({sorted(rcat)})')
        if not (abs(f0['mse_V'] - np.average(((E.V - E.value) ** 2).to_numpy(), weights=E.weight.to_numpy())) < 1e-9 and f0['n'] == len(E)
                and abs(f0['brier'] - np.average(((E.P - E.z) ** 2).to_numpy(), weights=E.weight.to_numpy())) < 1e-12):
            bad.append('forecasts: the weighted MSE / Brier')
        AC, NU, DR = M.actions(D); a = {(r['population'], r['rule']): r for r in AC}
        if not (a[('absent_POST', 'pi0_rule')]['degenerate'] and a[('absent_WAIT', 'pi0_rule')]['degenerate'] and not a[('entry_combined', 'pi0_rule')]['degenerate']
                and set(NU) == {'entry_combined', 'review_combined', 'absent_POST', 'absent_WAIT', 'review_KEEP', 'review_PULL'} and len(DR) == 6 * M.NULL_DRAWS
                and NU['absent_POST']['free_strata'] == 0 and NU['entry_combined']['free_strata'] > 0):
            bad.append('actions: degeneracy within sources / the combined populations / the null')
        g = D[D.eligible & D.cat.isin(['absent_POST', 'absent_WAIT'])].reset_index(drop=True); mk = np.array([bool(M.rule(c, x)) for c, x in zip(g.cat, g.V)])
        nu, J = M.null(g, mk)
        if not (nu['chosen'] == int(mk.sum()) and abs(nu['J'] - float((g.weight * mk * g.value).sum() / g.weight.sum())) < 1e-12 and len(J) == M.NULL_DRAWS):
            bad.append('null: the chosen count / J')
        if not (M.qbin(0., 30.) == '0' and M.qbin(30., 30.) == '(0,1]' and M.qbin(31., 30.) == '(1,3]' and M.qbin(91., 30.) == '>3' and M.qbin(1., 0.) == 'missing_or_invalid'
                and M.obin(1.) == '1' and M.obin(2.) == '2' and M.obin(3.) == '>=3' and M.hbin(100.) == '(0,100]' and M.hbin(100.5) == '(100,500]' and M.hbin(5000.) == '(2000,5000]'):
            bad.append('qbin / obin / hbin: a boundary')
        if not (M.rule('absent_WAIT', 0.) is False and bool(M.rule('absent_POST', 1e-9)) and bool(M.rule('review_KEEP', 0.)) and not bool(M.rule('review_PULL', -1e-9))):
            bad.append('rule: the POST / KEEP ties')
        gc = g.assign(value=10., weight=1.); nc, Jc = M.null(gc, mk)
        if not np.allclose(Jc, nc['J']):
            bad.append('null: draws that do not preserve the chosen count')
        P = M.predicates(FC, M.coverage(D), AC, NU)
        if not (set(P['selects_positive_exposure']) == {'entry_combined', 'review_combined'} and isinstance(P['covered'], list) and isinstance(P['lacking_support'], list)
                and all(x['answer'] in ('POSITIVE', 'NOT_POSITIVE', 'INCONCLUSIVE') for x in P['selects_positive_exposure'].values())):
            bad.append('predicates: the readings')
        ACp = [dict(population='entry_combined', rule='pi0_rule', J=2.), dict(population='entry_combined', rule='all', J=1.), dict(population='entry_combined', rule='none', J=0.)]
        NUp = dict(entry_combined=dict(free_strata=6, free_rows=40, null_p95=1.5))
        if M.predicates([], [], ACp, NUp)['selects_positive_exposure']['entry_combined']['answer'] != 'POSITIVE':
            bad.append('predicates: a positive selection above all / p95 is not POSITIVE')
        if M.predicates([], [], ACp, dict(entry_combined=dict(NUp['entry_combined'], null_p95=2.5)))['selects_positive_exposure']['entry_combined']['answer'] != 'NOT_POSITIVE':
            bad.append('predicates: a selection below the null p95 is not NOT_POSITIVE')
        ACa = [dict(r, J=2.5) if r['rule'] == 'all' else r for r in ACp]
        if M.predicates([], [], ACa, NUp)['selects_positive_exposure']['entry_combined']['answer'] != 'NOT_POSITIVE':
            bad.append('predicates: a selection below choosing all is not NOT_POSITIVE')
        if M.predicates([], [], ACp, dict(entry_combined=dict(NUp['entry_combined'], free_rows=29)))['selects_positive_exposure']['entry_combined']['answer'] != 'INCONCLUSIVE':
            bad.append('predicates: thin movable rows are not INCONCLUSIVE')
        if M.predicates([], [], ACp, dict(entry_combined=dict(NUp['entry_combined'], free_strata=4)))['selects_positive_exposure']['entry_combined']['answer'] != 'INCONCLUSIVE':
            bad.append('predicates: thin movable support is not INCONCLUSIVE')
        CVp = [dict(cat='absent_POST', side=0, hbin='(2000,5000]', rows=30, fills=10), dict(cat='absent_POST', side=0, hbin='(0,100]', rows=29, fills=20)]
        Pp = M.predicates([], CVp, [], {})
        if Pp['covered'] != ['absent_POST|0|(2000,5000]'] or Pp['lacking_support'] != ['absent_POST|0|(0,100]']:
            bad.append(f"predicates: covered / lacking ({Pp['covered']}, {Pp['lacking_support']})")
        FCp = [dict(cat='absent_POST', day='all', side='all', top5_world_err=.5, top5_world_severe=.49, max_date_share_err=.5, max_date_share_severe=.49)]
        cc = M.predicates(FCp, [], [], {})['concentration']['absent_POST']
        if not (cc['value_error_concentrated'] and not cc['severe_concentrated'] and cc['value_error_one_date'] and not cc['severe_one_date']):
            bad.append('predicates: the concentration threshold')
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'measurement: stopped ({e!r})'[:240])
    # ---- 5 guards
    try:
        real = M.frozen; M.frozen = lambda: False
        try:
            with tempfile.TemporaryDirectory(dir=str(M.KP.I.K.IO)) as tmp:
                for f_, why in ((lambda: M.run_part(0, Path(tmp)), 'run_part'), (lambda: M.run_pilot('20261024', Path(tmp)), 'run_pilot')):
                    try:
                        f_(); bad.append(f'{why}: ran before the freeze')
                    except Exception as e:                                                  # noqa: BLE001
                        if 'not frozen' not in str(e):
                            bad.append(f'{why}: refused for another reason ({e!r})'[:200])
                (Path(tmp) / 'audit.json').write_text('{}')
                try:
                    M.measure(Path(tmp)); bad.append('measure: overwrote a measurement')
                except Exception as e:                                                      # noqa: BLE001
                    if 'immutable' not in str(e):
                        bad.append(f'measure: refused for another reason ({e!r})'[:200])
        finally:
            M.frozen = real
        real_f, real_rm, real_rec = M.frozen, M.run_members, M.RECEIPT
        try:
            with tempfile.TemporaryDirectory(dir=str(M.KP.I.K.IO)) as tmp:
                tmp = Path(tmp); rel = str(tmp.relative_to(M.REPO)) if str(tmp).startswith(str(M.REPO)) else None
                rec_dir = tmp / 'rec'; (rec_dir / 'logs').mkdir(parents=True); (rec_dir / 'manifest.json').write_text(json.dumps(dict(partitions=[dict(part=0, day='20261024', members=[])],
                                                                                                                     days=['20261024'], sample={'20261024': dict(pilot=[])})))
                M.frozen = lambda: True; M.run_members = lambda *a_: 'RAN'; M.REPO, real_repo = tmp, M.REPO; M.RECEIPT = 'rec'
                try:
                    try:
                        M.run_part(0, tmp / 'out'); bad.append('run_part: ran without an allowing projection')
                    except Exception as e:                                                  # noqa: BLE001
                        if 'projection' not in str(e):
                            bad.append(f'run_part: refused for another reason ({e!r})'[:200])
                    (rec_dir / 'logs' / 'pilot_projection.json').write_text(json.dumps(dict(run_rest=False)))
                    try:
                        M.run_part(0, tmp / 'out'); bad.append('run_part: ran against a refusing projection')
                    except Exception as e:                                                  # noqa: BLE001
                        if 'projection' not in str(e):
                            bad.append(f'run_part: refused a refusing projection for another reason ({e!r})'[:200])
                    (rec_dir / 'logs' / 'pilot_projection.json').write_text(json.dumps(dict(run_rest=True)))
                    if M.run_part(0, tmp / 'out') != 'RAN':
                        bad.append('run_part: refused an allowing projection')
                    (tmp / 'out' / 'pilot').mkdir(parents=True); (tmp / 'out' / 'pilot' / 'pilot_20261024.json').write_text('{}')
                    try:
                        M.run_pilot('20261024', tmp / 'out'); bad.append('run_pilot: overwrote a pilot')
                    except Exception as e:                                                  # noqa: BLE001
                        if 'write-once' not in str(e):
                            bad.append(f'run_pilot: refused for another reason ({e!r})'[:200])
                finally:
                    M.REPO = real_repo
        finally:
            M.frozen, M.run_members, M.RECEIPT = real_f, real_rm, real_rec
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'guards: stopped ({e!r})'[:200])
    if bad:
        sys.exit(FAIL + ': ' + '; '.join(bad))
    print('coverage selftest passed: population (ranks, pi, thin hours, pilot, partitions, manifest write-once); categorize; the cell sample and weights; '
          'hold ledgers and the inclusive target; the synthetic world (roll = engine step, categories, labels of every category, four falsifiers); '
          'measurement (eligibility, weighted metrics, degeneracy, null, readings); the freeze guards')
