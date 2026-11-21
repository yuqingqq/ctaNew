"""SELFTEST of pf_entry_exposure (ENTRY-EXPOSURE-v1; 2026-11-21, session aea5bfb9). run(M) checks the module object M (the file, or a mutated
copy under pf_mutant_one.py); any failed check exits with FAIL. Synthetic only:
 1. crafted tapes on the real engine with a stub inner policy (absent WAIT, live PULL, like PI0's early pull), the review's controls:
    a queue that needs more resting time (KEEP fills, PI0 does not; D > 0), a harmful late fill avoided by PI0 (D < 0), a structural removal
    despite KEEP (reason structural, D = 0), a post-only refusal despite KEEP, a partial fill then closure at END, later generations and the
    opposite side untouched (overrides only on the live focal gid0), wrong / missing gid0 (no override), a stale branch (binding failures),
    baseline equivalence when the inner policy already keeps (KEEP = POST_PI0 exactly);
 2. prefix_ok on crafted arms: true for an override arm and a no-override arm, false after a decision before the override, a fill before it or
    the override itself is altered;
 3. fork_package end to end on the cached synthetic world (the frozen fork / pair machinery, the stub teacher): a stored pair is reproduced
    (status ok, every check true, the three branches, the identities); a stored Y, decisions hash, requests hash or fingerprint that differs is
    reproduction_failed / fork_mismatch; a pair that does not reconcile is reproduction_failed, a KEEP arm that does not is not_reconciled;
 4. select_worlds: the four lowest sha256 ranks per (day, hour) under the declared prefix, all if fewer, ties by the canonical key; values on
    hand numbers (the contrasts, D = D_initial + D_rest);
 5. flat / summaries / reasons on hand rows: equal-world means, fill probabilities, the D sums and components, the termination reason counts;
 6. every writer end to end in a temporary directory: build_manifest on two hand plans and hand dataset rows (the four-per-hour selection, one
    fork per world and side, every status: ok, world:x, no_mandatory_row, mandatory_pair_not_ok, mandatory_row_not_unique; the stored hashes;
    the parts; an existing manifest is refused); summarize on hand partitions (forks.csv, summary.csv, reasons.csv, exposure.json; a
    partition whose bytes differ is refused); run_part refuses a manifest that is not the committed one."""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

FAIL = 'ENTRY EXPOSURE SELFTEST FAILED'


def run(M):  # noqa: C901
    bad = []; BE, L, LS, R2, T, pd = M.BE, M.L, M.LS, M.R2, M.T, M.pd; C_sha = M.C.sha
    import pf_r2_labels_selftest as LT
    import pf_r2_teacher_selftest as TS

    class Stub(L.ArmTeacher):
        def __init__(self, absent='WAIT', live='PULL'):
            super().__init__(None, None); self.a, self.l = absent, live
        def decide(self, v, B):
            return dict(t=float(v['t']), side=int(v['side']), state=v['state'], gid=v['gid'], status='stub', action=self.a if v['state'] == 'absent' else self.l)

    def fac(absent='WAIT', live='PULL'):
        made = []
        def make(B):
            p = Stub(absent, live); p.bind(B, None); made.append((p, {})); return p
        make.made = made
        return make
    ident = lambda s: dict(day='20261024', instrument_id=1, original_episode_id=1, side=s, start_abs_ms=0., end_abs_ms=5000., mark_source='x')
    def three(path, side=0, absent='WAIT', live='PULL'):
        ctx = BE.fork_state(path, BE.NEVER, 0, side, 0, None, ident(side)); f = fac(absent, live); res = L.pair_bound(ctx, f, 'post')
        kf = M.keep_factory(lambda B: (lambda p: (p.bind(B, None), p)[1])(Stub(absent, live)), side); Bk, Lk, pk = M.run_arm(ctx, kf, keep=True)
        return ctx, res, f, Bk, Lk, pk
    fr = lambda *a, **k: BE.frame(*a, **k)
    cases = {}
    try:
        p_a = BE.tape([fr(0, 0, 50, 52, 0, 100)], [(600, 0, 50, 30)], 51.)
        cases['more resting time'] = (three(p_a), dict(D=30., keep_filled=30., pi0_filled=0.))
        cases['harmful late fill'] = (three(BE.tape([fr(0, 0, 50, 52, 0, 100)], [(600, 0, 50, 30)], 49.)), dict(D=-30., keep_filled=30., pi0_filled=0.))
        cases['structural removal'] = (three(BE.tape([fr(0, 0, 50, 52, 0, 100), fr(300, 300, 54, 56, 100, 100)], [(600, 0, 50, 30)], 51.)), dict(D=0., keep_filled=0., keep_why='structural'))
        cases['post-only refusal'] = (three(BE.tape([fr(0, 0, 50, 52, 0, 100), fr(30, 30, 50, 50, 0, 100)], [(600, 0, 50, 30)], 51.)), dict(D=0., keep_filled=0., keep_why='refused'))
        cases['partial fill'] = (three(BE.tape([fr(0, 0, 50, 52, 0, 100)], [(600, 0, 50, 12)], 51.)), dict(D=12., keep_filled=12., keep_why='end', unfilled=18.))
        cases['already keeps'] = (three(p_a, live='KEEP'), dict(D=0., overrides=0))
        cases['ask side, other side first'] = (three(BE.tape([fr(0, 0, 50, 52, 100, 0)], [(600, 1, 52, 30)], 51.), side=1), dict(D=30., keep_filled=30., pi0_filled=0.))
        cases['late fill'] = (three(BE.tape([fr(0, 0, 50, 52, 0, 100)], [(4500, 0, 50, 30)], 51.)), dict(D=30., keep_filled=30.))
    except Exception as e:                                                                  # noqa: BLE001  (a valid crafted case must run)
        bad.append(f'crafted: a case stopped ({e!r})'[:200])
    for name, ((ctx, res, f, Bk, Lk, pk), want) in cases.items():
        if res.get('status') != 'ok' or Lk.get('status') != 'ok':
            bad.append(f'{name}: pair / arm status {res.get("status")} / {Lk.get("status")}'); continue
        Bp = f.made[0][0].B; s = int(ctx['side']); gp = [d for d in Bp.decisions if float(d[0]) == 0. and int(d[1]) == s and d[3] == 'POST'][0][4]
        if pk.gid0 is None or pk.gid0 != gp:
            bad.append(f'{name}: the KEEP arm\'s gid0 {pk.gid0} is not the fork POST\'s {gp}'); continue
        ip, ik = M.initial(Bp, int(gp), s), M.initial(Bk, int(pk.gid0), s); D = float(Lk['value']) - float(res['arms']['POST']['value'])
        got = dict(D=D, keep_filled=ik['filled'], pi0_filled=ip['filled'], keep_why=ik['why'], unfilled=ik['unfilled_at_end'], overrides=pk.overrides)
        for k, v in want.items():
            if (abs(got[k] - v) > 1e-9) if isinstance(v, float) else got[k] != v:
                bad.append(f'{name}: {k} = {got[k]}, not {v}')
        if name == 'late fill' and not (ik['fills'] and ik['fills'][0]['m1000'] is None and ik['fills'][0]['m100'] is not None):
            bad.append('late fill: a markout past END was not missing, or one inside it was')
        if name == 'already keeps' and not (L._h(Lk['fills']) == res['arms']['POST']['fills_sha256'] and L._h(Lk['decisions']) == res['arms']['POST']['decisions_sha256']):
            bad.append('already keeps: KEEP_INITIAL is not POST_PI0 exactly')
        if pk.bind_bad:
            bad.append(f'{name}: binding failures in the KEEP arm')
        if not M.prefix_ok(Bk, Bp, pk):
            bad.append(f'{name}: prefix_ok false on a valid arm')
        cp, ck = M.components(Bp, gp, s), M.components(Bk, pk.gid0, s); vv = M.values(0., float(res['arms']['POST']['value']), float(Lk['value']), cp, ck)
        if not (vv['components_ok'] and abs(cp['v_initial'] - ip['v_initial']) <= 1e-9 and abs(ck['v_initial'] - ik['v_initial']) <= 1e-9):
            bad.append(f'{name}: the components do not reconcile ({vv})')
        if name in ('more resting time', 'harmful late fill', 'partial fill', 'ask side, other side first') and not (
                abs(vv['D_initial'] - D) <= 1e-9 and abs(vv['D_rest']) <= 1e-9 and vv['D_opposite'] == 0. and vv['D_focal_rest'] == 0.):
            bad.append(f'{name}: D is not carried by the initial generation ({vv})')
    # ---- 2 prefix_ok on altered arms
    try:
        import copy as _c
        (ctx, res, f, Bk, Lk, pk), _ = cases['more resting time']; Bp = f.made[0][0].B; i0 = next(j for j, (a, b) in enumerate(zip(Bk.decisions, Bp.decisions)) if a != b)
        alt = []
        if i0 > 0:
            B1 = _c.deepcopy(Bk); d0 = list(B1.decisions[0]); d0[2] = 'tampered'; B1.decisions[0] = tuple(d0); alt.append(('a decision before the override', B1))
        B2 = _c.deepcopy(Bk); B2.E.fills.insert(0, (0., 0, 0, 1., 50., 0)); alt.append(('a fill before the override', B2))
        B3 = _c.deepcopy(Bk); d3 = list(B3.decisions[i0]); d3[3] = 'WAIT'; B3.decisions[i0] = tuple(d3); alt.append(('the override altered', B3))
        if i0 == 0:
            bad.append('prefix_ok: the crafted override arm has no decision before its override')
        for lab, Bx in alt:
            if M.prefix_ok(Bx, Bp, pk):
                bad.append(f'prefix_ok: true after {lab}')
        (ctx2, res2, f2, Bk2, Lk2, pk2), _ = cases['already keeps']; B4 = _c.deepcopy(Bk2); B4.E.fills.append((4000., 0, 0, 1., 50., 0))
        if not M.prefix_ok(Bk2, f2.made[0][0].B, pk2) or M.prefix_ok(B4, f2.made[0][0].B, pk2):
            bad.append('prefix_ok: the no-override arm is not compared in full')
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'prefix_ok: the case stopped ({e!r})'[:200])
    # later generations and the opposite side untouched
    try:
        ctx, res, f, Bk, Lk, pk = three(BE.tape([fr(0, 0, 50, 52, 0, 0)], [(300, 0, 50, 30), (2000, 0, 50, 30), (2500, 1, 52, 30)], 51.), absent='POST')
        ov = [d for d in pk.log if d.get('status') == 'keep_initial_override']; later = [d for d in pk.log if d.get('state') == 'live' and d.get('action') == 'PULL']
        if not ov or any(d['gid'] != pk.gid0 or d['side'] != 0 for d in ov) or not any(d['gid'] != pk.gid0 for d in later):
            bad.append('untouched: an override outside the live focal gid0, or no later / opposite PULL left to PI0')
        Bp = f.made[0][0].B; vv = M.values(0., float(res['arms']['POST']['value']), float(Lk['value']), M.components(Bp, pk.gid0, 0), M.components(Bk, pk.gid0, 0))
        if not vv['components_ok']:
            bad.append(f'untouched: the two-sided components do not reconcile ({vv})')
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'untouched: the case stopped ({e!r})'[:200])
    # wrong / missing gid0 and a stale branch
    try:
        ctx = BE.fork_state(p_a, BE.NEVER, 0, 0, 0, None, ident(0)); base = L.pair_bound(ctx, fac(), 'post')['arms']['POST']
        for g0 in (None, 99):
            kf = M.keep_factory(lambda B: (lambda p: (p.bind(B, None), p)[1])(Stub()), 0); Bx, Lx, px = M.run_arm(ctx, kf, keep=False); px_g = g0
            kf2 = M.keep_factory(lambda B: (lambda p: (p.bind(B, None), p)[1])(Stub()), 0)
            def make_wrong(B, kf2=kf2, g0=g0):
                p = kf2(B); p.gid0 = g0; return p
            make_wrong.made = kf2.made; Bw, Lw, pw = M.run_arm(ctx, make_wrong, keep=False)
            if pw.overrides != 0 or L._h(Lw['fills']) != base['fills_sha256']:
                bad.append(f'wrong gid0 {g0}: an override happened')
        stale = BE.Branch(p_a, 0, ident=ident(0)); kf3 = M.keep_factory(lambda B: (lambda p: (p.bind(stale, None), p)[1])(Stub()), 0)
        def make_stale(B, kf3=kf3):
            p = kf3(B); p.bind(stale, None); return p
        make_stale.made = kf3.made; Bs, Ls, ps = M.run_arm(ctx, make_stale, keep=True)
        if ps.bind_bad == 0:
            bad.append('stale branch: no binding failure')
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'wrong gid / stale: the case stopped ({e!r})'[:200])
    # ---- 3 fork_package end to end on the cached synthetic world
    wroot = Path(os.environ['PF_TEACHER_SYN']) if os.environ.get('PF_TEACHER_SYN') else None
    with tempfile.TemporaryDirectory(dir=str(R2.I.K.IO)) as tmp:
        if wroot is None:
            wroot = TS.build_world(Path(tmp) / 'world')
        wj = json.loads((wroot / 'world.json').read_text()); day = '20261024'; kp = Path(wj['kp']); src = R2.KP.Syn(); ff = R2.X.syn_features()
        W = L.worlds_of_day(day, kp, 2); _, prs, fair, depth, ff = L.world_inputs(day, W, src, R2.X.syn_inputs, ff)
        Mst, sst = LT.stubs(-12.), LT.OpenSupport(); done = False
        for w in W:
            wb = L.world_bind(day, w, prs, fair, depth, ff)[0]
            if wb is None:
                continue
            sel = dict(day=day, side=0, decision_ms=0, source='neutral', shadow_source='entry', offset_ms=0, kind='post')
            try:
                ctx = L.fork_for(sel, wb, Mst, sst); st = L.pair_bound(ctx, L.teacher_factory(Mst, sst, wb, every=M.AUDIT_EVERY), 'post')
                f = dict(day=day, m=int(w['m']), t_ms=int(round(float(w['t']))), hour=0, side=0, oid=1, status='ok', sel=sel, fork_fingerprint=ctx['fingerprint'], Y=st['Y'],
                         arms={n: dict(value=a['value'], fills_sha256=a['fills_sha256'], decisions_sha256=a['decisions_sha256'], requests_sha256=a['requests_sha256'])
                               for n, a in st['arms'].items()})
                r = M.fork_package(f, wb, Mst, sst)
                if not (r['status'] == 'ok' and r['reproduced'] and r['identity_ok'] and r['prefix_ok'] and r['keep_reconciled'] and r['initial_gid_equal']
                        and r['keep_baseline_equal'] and set(r['arms']) == {'WAIT_PI0', 'POST_PI0', 'POST_KEEP_INITIAL'}
                        and abs(r['Y_KEEP_INITIAL'] - (r['Y_PI0'] + r['D_EXPOSURE'])) <= 1e-9 and r['Y_PI0'] == st['Y']
                        and abs(r['D_initial'] - (r['init_KEEP']['v_initial'] - r['init_PI0']['v_initial'])) <= 1e-9 and abs(r['D_initial'] + r['D_rest'] - r['D_EXPOSURE']) <= 1e-9
                        and abs(r['arms']['POST_KEEP_INITIAL']['value'] - r['arms']['POST_PI0']['value'] - r['D_EXPOSURE']) <= 1e-9):
                    bad.append(f"fork_package: a stored pair was not reproduced ({r.get('status')})")
                if M.fork_package(dict(f, Y=f['Y'] + 1.), wb, Mst, sst)['status'] != 'reproduction_failed':
                    bad.append('fork_package: a different stored Y was not reproduction_failed')
                for arm_, h_ in (('WAIT', 'decisions_sha256'), ('POST', 'requests_sha256')):
                    fa = json.loads(json.dumps(f['arms'])); fa[arm_][h_] = '0' * 64
                    if M.fork_package(dict(f, arms=fa), wb, Mst, sst)['status'] != 'reproduction_failed':
                        bad.append(f'fork_package: a different stored {arm_} {h_} was not reproduction_failed')
                real = L.reconcile_arm
                for fail_at, want_st in ((1, 'reproduction_failed'), (3, 'not_reconciled')):
                    n_ = [0]
                    def rec_(Lx, Bx, fail_at=fail_at, n_=n_):
                        n_[0] += 1; return False if n_[0] == fail_at else real(Lx, Bx)
                    L.reconcile_arm = rec_
                    try:
                        got_st = M.fork_package(f, wb, Mst, sst)['status']
                    finally:
                        L.reconcile_arm = real
                    if got_st != want_st:
                        bad.append(f'fork_package: reconcile call {fail_at} failing gave {got_st}, not {want_st}')
                if M.fork_package(dict(f, fork_fingerprint='0' * 64), wb, Mst, sst)['status'] != 'fork_mismatch':
                    bad.append('fork_package: a different fingerprint was not fork_mismatch')
                done = True; break
            except Exception as e:                                                          # noqa: BLE001
                bad.append(f'fork_package: a valid synthetic fork stopped ({e!r})'[:200]); break
        if not done and not any('fork_package' in b for b in bad):
            bad.append('fork_package: no synthetic world ran')
    # ---- 4 select_worlds and values
    sel = [dict(m=100 * h + i, t_ms=1000 * i + h, hour=h, rank='x') for h in range(3) for i in range(7 if h != 1 else 3)]
    plan = dict(days=['20261024'], sample={'20261024': dict(selected=sel)}); got, pop = M.select_worlds(plan)
    want = []
    for h in range(3):
        import hashlib
        rk_ = lambda w: hashlib.sha256(('entry_exposure_v1|' + LS.ckey(dict(day=w['day'], m=w['m'], t_ms=w['t_ms']))).encode()).hexdigest()     # the declared rank, independently
        ws = sorted([dict(w, day='20261024') for w in sel if w['hour'] == h], key=lambda w: (rk_(w), LS.ckey(dict(day=w['day'], m=w['m'], t_ms=w['t_ms']))))[:4]
        want += [(w['m'], w['t_ms']) for w in ws]
    if [(w['m'], w['t_ms']) for w in got] != want or [p_['selected'] for p_ in pop] != [4, 3, 4] or [p_['population'] for p_ in pop] != [7, 3, 7] or M.PER_HOUR != 4 \
            or M.PREFIX != 'entry_exposure_v1|':
        bad.append('select_worlds: not the four lowest declared ranks per hour (all if fewer)')
    cp_, ck_ = dict(v_initial=2., v_focal_rest=.5, v_opposite=.5), dict(v_initial=4.5, v_focal_rest=1., v_opposite=.5)
    v_ = M.values(1., 3., 6., cp_, ck_)
    if not (v_['Y_PI0'] == 2. and v_['Y_KEEP_INITIAL'] == 5. and v_['D_EXPOSURE'] == 3. and v_['D_initial'] == 2.5 and v_['D_rest'] == .5 and v_['D_focal_rest'] == .5
            and v_['D_opposite'] == 0. and v_['identity_ok'] and v_['components_ok']):
        bad.append(f'values: the declared contrasts ({v_})')
    if M.values(1., 3., 6., cp_, dict(ck_, v_opposite=1.5))['components_ok']:
        bad.append('values: components that do not sum to the arm value were accepted')
    import types
    FB = types.SimpleNamespace(mark=51., E=types.SimpleNamespace(fills=[(0., 0, 0, 10., 50., 0), (0., 1, 0, 5., 50., 0), (0., 2, 1, 3., 52., 0)]))
    if M.components(FB, 0, 0) != dict(v_initial=10., v_focal_rest=5., v_opposite=3.) or M.components(FB, 2, 1) != dict(v_initial=3., v_focal_rest=0., v_opposite=15.):
        bad.append('components: the hand fills are not split into initial / focal rest / opposite')
    # ---- 5 flat / summaries / reasons
    rows = []
    for i, (d, f1, f2) in enumerate(((10., 0., 30.), (-5., 0., 0.), (0., 0., 0.))):
        rows.append(dict(day='d', m=i, t_ms=i, hour=0, side=0, status='ok', reproduced=True, keep_baseline_equal=True, overrides=1, Y_PI0=0., Y_stored=0., Y_KEEP_INITIAL=d,
                         D_EXPOSURE=d, D_initial=d / 2., D_rest=d / 2., D_focal_rest=d / 4., D_opposite=d / 4., identity_ok=True, components_ok=True,
                         init_PI0=dict(why='policy', filled=f1, active_ms=50., v_initial=0., unfilled_at_end=0., fills=[]),
                         init_KEEP=dict(why='end' if f2 == 0. else 'full_fill', filled=f2, first_fill=600. if f2 else None, active_ms=500., v_initial=d, unfilled_at_end=0.,
                                        fills=[dict(q=30., m50=1., m100=2., m1000=None)] if f2 else []),
                         arms=dict(WAIT_PI0=dict(value=0., position=0.), POST_PI0=dict(value=0., position=0.), POST_KEEP_INITIAL=dict(value=d, position=f2))))
    rows.append(dict(day='d', m=9, t_ms=9, hour=0, side=0, status='world:x'))
    S = M.summaries(pd.DataFrame([M.flat(r) for r in rows]))[0]
    if not (S['forks'] == 3 and abs(S['mean_D'] - 5. / 3.) < 1e-12 and S['sum_D_pos'] == 10. and S['sum_D_neg'] == -5. and abs(S['fill_prob_KEEP'] - 1. / 3.) < 1e-12
            and S['fill_prob_PI0'] == 0. and S['n_D_pos'] == 1 and S['n_D_neg'] == 1 and S['n_D_zero'] == 1 and abs(S['mean_D_initial'] - 5. / 6.) < 1e-12
            and S['mean_first_fill_KEEP'] == 600. and S['mean_first_fill_PI0'] is None and S['sum_m100_KEEP'] == 2. and S['m1000_fills_KEEP'] == 0
            and S['fills_KEEP'] == 1 and abs(S['open_inv_KEEP'] - 10.) < 1e-12 and S['sum_D'] == 5. and S['m100_q_KEEP'] == 30. and S['m1000_q_KEEP'] == 0.
            and S['addl_fill_forks'] == 1 and S['mean_D_addl'] == 10. and abs(S['mean_D_opposite'] - 5. / 12.) < 1e-12 and S['q_filled_KEEP'] == 30.
            and S['n_meaningful_Y_KEEP'] == 2):
        bad.append(f'summaries: hand values ({S})')
    W = {(x['branch'], x['why']): x['n'] for x in M.reasons(pd.DataFrame([M.flat(r) for r in rows]))}
    if W != {('PI0', 'policy'): 3, ('KEEP', 'full_fill'): 1, ('KEEP', 'end'): 2}:
        bad.append(f'reasons: hand counts ({W})')
    # predicates on hand rows: four days, the signs, the thin floor, the execution / component / harm readings
    def prow(day, d, di, fp, fk):
        return dict(day=day, status='ok', D_EXPOSURE=d, D_initial=di, D_rest=d - di, PI0_filled=fp, KEEP_filled=fk)
    PF = pd.DataFrame([prow(d_, 2., 1.5, 0., 30.) for d_ in M.DAYS for _ in range(8)] + [prow(d_, -1., -.5, 0., 0.) for d_ in M.DAYS for _ in range(8)]
                      + [dict(prow('20261024', -99., -99., 0., 0.), status='world:x')])
    P_ = M.predicates(PF)
    if not (P_['diagnosis'] == 'keep_better' and all(P_['days'][d_]['sign'] == 1 and P_['days'][d_]['forks'] == 16 for d_ in M.DAYS) and P_['fill_prob_KEEP'] == .5
            and P_['fill_prob_PI0'] == 0. and P_['more_execution'] is True and P_['initial_carries'] is True and abs(P_['sum_D_initial'] - 32.) < 1e-12 and P_['forks'] == 64
            and P_['additional_fill_forks'] == 32 and P_['extra_fills_harmful'] is False and P_['mean_D_initial_additional'] == 1.5):
        bad.append(f'predicates: keep_better hand rows ({P_})')
    P2 = M.predicates(PF.assign(D_EXPOSURE=-PF.D_EXPOSURE, D_initial=-PF.D_initial, D_rest=-PF.D_rest))
    if not (P2['diagnosis'] == 'pi0_better' and P2['more_execution'] is True and P2['extra_fills_harmful'] is True and P2['share_D_initial_neg_additional'] == 1.
            and P2['initial_carries'] is True and P2['sum_D_initial'] == -32.):
        bad.append(f'predicates: pi0_better hand rows ({P2})')
    sm = pd.DataFrame([prow(d_, 1., 1., 0., 30. if (d_ == '20261024' and i < 3) else 0.) for d_ in M.DAYS for i in range(25)]); Ps = M.predicates(sm)
    if not (abs(Ps['fill_prob_KEEP'] - .03) < 1e-12 and Ps['more_execution'] is False):
        bad.append(f"predicates: a 0.03 fill-probability gain counted as more execution ({Ps['fill_prob_KEEP']}, {Ps['more_execution']})")
    th = PF.copy(); idx = th.index[th.KEEP_filled > 0]; th.loc[idx[29:], 'KEEP_filled'] = 0.; Pt = M.predicates(th)
    th2 = PF.copy(); th2.loc[th2.index[th2.KEEP_filled > 0][30:], 'KEEP_filled'] = 0.
    if Pt['diagnosis'] != 'not_identified_thin' or Pt['additional_fill_forks'] != 29 or M.predicates(th2)['diagnosis'] != 'keep_better' or M.THIN != 30:
        bad.append(f"predicates: the thin floor ({Pt['diagnosis']}, {Pt['additional_fill_forks']})")
    PF3 = PF.copy(); PF3.loc[(PF3.day == '20261026') & (PF3.status == 'ok'), 'D_EXPOSURE'] = -1.; P3 = M.predicates(PF3)
    PF4 = PF.copy(); ok4 = PF4.index[(PF4.day == '20261024') & (PF4.status == 'ok')]; PF4.loc[ok4, 'D_EXPOSURE'] = [1., -1.] * (len(ok4) // 2); P4 = M.predicates(PF4)
    if P3['diagnosis'] != 'not_identified_days_disagree' or P4['days']['20261024']['sign'] != 0 or P4['diagnosis'] != 'not_identified_days_disagree' \
            or M.predicates(PF[PF.day != '20261027'])['diagnosis'] != 'incomplete' or M.MORE_EXECUTION != .05 or M.DAYS != ('20261024', '20261025', '20261026', '20261027'):
        bad.append('predicates: days disagreeing / zero sign / incomplete')
    # ---- 6 every writer end to end
    try:
        import hashlib
        with tempfile.TemporaryDirectory(dir=str(R2.I.K.IO)) as tmp:
            tmp = Path(tmp); plans = {}
            for kind, day in (('train', '20261024'), ('eval', '20261026')):
                (tmp / kind).mkdir(); sel_ = [dict(day=day, m=(1 + i) if h == 0 else 10 + i, t_ms=100 * i, hour=h) for h in range(2) for i in range(4 if h == 0 else 6)]
                (tmp / kind / 'plan.json').write_text(json.dumps(dict(days=[day], sample={day: dict(selected=sel_)})))
                plans[kind] = dict(dir=tmp / kind, dataset=kind * 8, files='real_files')
            def fake_rows(d, sha):
                pl = json.loads((Path(d) / 'plan.json').read_text()); day = pl['days'][0]
                for w in pl['sample'][day]['selected']:
                    for s in (0, 1):
                        rk = dict(day=day, side=s, source='neutral', shadow_source='entry', decision_ms=0, offset_ms=0, kind='post')
                        base = dict(day=day, m=w['m'], t_ms=w['t_ms'], side=s, head='post', oid=7, row_key=rk, fork_fingerprint='f' * 64,
                                    result=dict(Y=1.5, arms={n: dict(value=v, fills_sha256=n + 'F', decisions_sha256=n + 'D', requests_sha256=n + 'R') for n, v in (('POST', 2.), ('WAIT', .5))}))
                        if w['m'] == 1:
                            yield dict(base, status='world:x', pick='only'); continue
                        if w['m'] == 2 and s == 1:
                            yield dict(base, status='ok', pick='only'); continue
                        if w['m'] == 3 and s == 1:
                            yield dict(base, status='pair_not_ok', pick='mandatory'); continue
                        if w['m'] == 4 and s == 0:
                            yield dict(base, status='ok', pick='mandatory')
                        yield dict(base, status='ok', pick='mandatory'); yield dict(base, status='ok', pick='extra', row_key=dict(rk, decision_ms=500))
                    yield dict(day=day, m=w['m'], t_ms=w['t_ms'], side=0, head='keep', pick='keep', status='ok')
            oldP, oldR = M.PLANS, M.A.dataset_rows; M.PLANS, M.A.dataset_rows = plans, fake_rows
            try:
                out = tmp / 'out'; M.build_manifest(out); man = json.loads((out / 'manifest.json').read_text())
                try:
                    M.build_manifest(out); bad.append('build_manifest: an existing manifest was overwritten')
                except Exception:                                                           # noqa: BLE001  (the refusal)
                    pass
            finally:
                M.PLANS, M.A.dataset_rows = oldP, oldR
            for kind in ('train', 'eval'):
                S_ = man['sets'][kind]; pl = json.loads((tmp / kind / 'plan.json').read_text()); want_w, _ = M.select_worlds(pl)
                st_ = {(f['m'], f['side']): f['status'] for f in S_['forks']}
                exp = {}
                for w in want_w:
                    for s in (0, 1):
                        exp[(w['m'], s)] = ('world:x' if w['m'] == 1 else 'no_mandatory_row' if (w['m'] == 2 and s == 1) else 'mandatory_pair_not_ok' if (w['m'] == 3 and s == 1)
                                            else 'mandatory_row_not_unique' if (w['m'] == 4 and s == 0) else 'ok')
                okf = [f for f in S_['forks'] if f['status'] == 'ok']
                if st_ != exp or len(S_['forks']) != 2 * len(want_w) or [p_['selected'] for p_ in S_['population']] != [4, 4] or [p_['population'] for p_ in S_['population']] != [4, 6] \
                        or set(st_.values()) != {'ok', 'world:x', 'no_mandatory_row', 'mandatory_pair_not_ok', 'mandatory_row_not_unique'} or S_['plan_sha256'] != C_sha(tmp / kind / 'plan.json') \
                        or not okf or any(f['Y'] != 1.5 or f['arms']['POST']['requests_sha256'] != 'POSTR' or f['sel']['decision_ms'] != 0 or f['fork_fingerprint'] != 'f' * 64 for f in okf) \
                        or sum(len(pt['worlds']) for pt in S_['parts']) != len(want_w):
                    bad.append(f'build_manifest {kind}: statuses / selection / stored fields ({sorted(st_.items())[:6]})')
            if [(p_['set'], p_['day']) for p_ in man['parts']] != [('train', '20261024'), ('eval', '20261026')]:
                bad.append('build_manifest: the partitions are not train then eval')
            # summarize on hand partitions (2 worlds, 4 forks: the hand rows of section 5)
            out2 = tmp / 'out2'; (out2 / 'parts').mkdir(parents=True); m2 = dict(parts=[dict(set='train', day='d', worlds=[dict(day='d', m=0, t_ms=0), dict(day='d', m=1, t_ms=1)])])
            (out2 / 'manifest.json').write_text(json.dumps(m2)); b = LS.gz_bytes(rows)
            (out2 / 'parts' / 'part_000.jsonl.gz').write_bytes(b)
            (out2 / 'parts' / 'part_000.json').write_text(json.dumps(dict(sha256=hashlib.sha256(b).hexdigest(), manifest_sha256=C_sha(out2 / 'manifest.json'), sources=M.sources())))
            M.summarize(out2); js = json.loads((out2 / 'exposure.json').read_text()); F2 = pd.read_csv(out2 / 'forks.csv'); S2 = pd.read_csv(out2 / 'summary.csv')
            W2 = pd.read_csv(out2 / 'reasons.csv')
            if not (js['forks'] == 4 and len(F2) == 4 and js['predicates']['diagnosis'] == 'incomplete' and js['predicates']['forks'] == 3 and abs(float(S2.mean_D.iloc[0]) - 5. / 3.) < 1e-12 and int(W2.n.sum()) == 6 and js['reproduced_all'] is True
                    and sorted((x['status'], x['n']) for x in js['statuses']) == [('ok', 3), ('world:x', 1)]):
                bad.append(f"summarize: the written outputs ({js.get('forks')}, {js.get('statuses')})")
            (out2 / 'parts' / 'part_000.jsonl.gz').write_bytes(LS.gz_bytes(rows[:2] + rows[:2]))
            try:
                M.summarize(out2); bad.append('summarize: a partition whose bytes differ from its record was read')
            except Exception:                                                               # noqa: BLE001  (the refusal)
                pass
            try:
                M.run_part(0, out=out); bad.append('run_part: an uncommitted manifest was simulated')
            except Exception as e:                                                          # noqa: BLE001  (the refusal)
                if 'committed' not in str(e):
                    bad.append(f'run_part: refused for another reason ({e!r})'[:200])
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'writers: the case stopped ({e!r})'[:200])
    if bad:
        sys.exit(FAIL + ': ' + '; '.join(bad))
    print('entry exposure selftest passed: crafted tapes (more resting time, harmful late fill, structural removal, refusal, partial fill, already '
          'keeps, ask side, late fill; later / opposite untouched; wrong gid0; stale branch); prefix_ok (valid, altered); fork_package end to end '
          '(reproduced; different Y / decisions / requests / fingerprint refused; reconcile failures); select_worlds; values; summaries; reasons; '
          'writers end to end (build_manifest statuses and refusal; summarize outputs and refusal; run_part refuses an uncommitted manifest)')
