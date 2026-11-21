"""SELFTEST of pf_entry_retention (ENTRY-RETENTION-DISCRIMINATION-v1; 2026-11-21, session aea5bfb9). run(M) checks the module object M (the file,
or a mutated copy under pf_mutant_one.py); any failed check exits with FAIL. Synthetic only:
 1. crafted tapes on the real engine with stub teachers (RecTeacher subclasses deciding from the view): a first review that keeps and a later
    distinct pull (stop after the pull tick; effect 37 ms later); a first review that pulls (same_as_review); a refusal at activation and a
    structural removal before any review (no review captured, the closure as the stop reason); the other side never captured; the prefix equal
    to the saved pair_bound POST arm, refused after one changed decision or one dropped callback; the neutral hold's initial fills equal to the
    KEEP arm's; entry / review targets split at the effect (a loss before the effect is unavoidable at review, a later one avoidable); the
    first-pull reconciliation with D_initial; a decision that reads the branch object itself (a hidden-state leak) flagged by the hidden copy;
 2. fork_stages end to end on the cached synthetic world (stub models; the exposure fork_package as the saved row): ok with every check true,
    107 inputs and their hash, moments reproduced; a changed D_initial, KEEP initial fill, END or POST_PI0 decision gives pull_not_reconciled /
    target_not_equivalent / horizon_mismatch / prefix_mismatch;
 3. the measurements on hand rows: the input groups and an unchanged-market transition, calibration / Brier, bins, decision tables, the matched
    null (a planted protective selection, a random one, a degenerate one), predicates and readings;
 4. the writers end to end: measure on hand partitions (every output; immutable; a changed partition refused); run_part refuses an uncommitted
    exposure manifest before any input is read."""
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

FAIL = 'ENTRY RETENTION SELFTEST FAILED'


def run(M):  # noqa: C901
    bad = []; BE, L, LS, R2, T, pd, EX, C = M.BE, M.L, M.LS, M.R2, M.T, M.pd, M.EX, M.C
    import pf_r2_labels_selftest as LT

    def stub_cls(focal, pull_at=None, leak=False):
        class Stub(M.RecTeacher):
            def __init__(self, Mo, sup, causal=None):
                super().__init__(None, None, causal=causal)

            def decide(self, v, B):
                s_, st = int(v['side']), v['state']
                if st == 'absent':
                    a = 'POST' if s_ == focal else 'WAIT'
                elif leak:
                    a = 'PULL' if B is self.B else 'KEEP'
                else:
                    a = 'PULL' if (s_ == focal and pull_at is not None and float(v['t']) >= pull_at) else 'KEEP'
                return dict(t=float(v['t']), side=s_, state=st, head='post' if st == 'absent' else 'retain', gid=v['gid'], status='model', action=a)
        return Stub
    ident = lambda s: dict(day='20261024', instrument_id=1, original_episode_id=1, side=s, start_abs_ms=0., end_abs_ms=5000., mark_source='x')
    fr = BE.frame
    ctx_of = lambda path, s: BE.fork_state(path, BE.NEVER, 0, s, 0, None, ident(s))

    def saved_for(ctx, focal, pull_at):
        Cls = stub_cls(focal, pull_at); made = []

        def make(B):
            p = Cls(None, None); p.bind(B, None); made.append((p, {})); return p
        make.made = made; res = L.pair_bound(ctx, make, 'post'); Bp = made[0][0].B
        kf = EX.keep_factory(lambda B: (lambda p: (p.bind(B, None), p)[1])(Cls(None, None)), focal); Bk, Lk, pk = EX.run_arm(ctx, kf, keep=True)
        gid = [d for d in Bp.decisions if float(d[0]) == 0. and int(d[1]) == focal and d[3] == 'POST'][0][4]
        lp = EX.ledger(Bp, res['arms']['POST'], made[0][0], {}); lk = EX.ledger(Bk, dict(Lk, fills_sha256=L._h(Lk['fills']), decisions_sha256=L._h(Lk['decisions'])), pk, {})
        ip, ik = EX.initial(Bp, int(gid), focal), EX.initial(Bk, int(pk.gid0), focal)
        return json.loads(json.dumps(dict(status='ok', arms=dict(POST_PI0=lp, POST_KEEP_INITIAL=lk), init_PI0=ip, init_KEEP=ik, D_initial=ik['v_initial'] - ip['v_initial']), default=str))
    # ---- 1 crafted tapes
    try:
        p_a = BE.tape([fr(0, 0, 50, 52, 0, 100)], [(600, 0, 50, 30)], 51.)
        ctx = ctx_of(p_a, 0); sv = saved_for(ctx, 0, 200.)
        Bx, pol, cst, gid0, stop, why, nd0, nf0 = M.prefix(ctx, None, None, None, 7, cls=stub_cls(0, 200.)); st = pol.stage
        if not (st.get('first_review', {}).get('t') == 100. and st['first_review']['action'] == 'KEEP' and st.get('first_pull', {}).get('t') == 200.
                and st['first_pull']['same_as_review'] is False and st['first_pull']['effect'] == 237. and stop == 200. and why == 'pull'):
            bad.append(f"distinct pull: stages {[(k_, v_.get('t'), v_.get('action')) for k_, v_ in st.items()]}, stop {stop}, {why}")
        if any(v_['side'] != 0 for v_ in st.values()):
            bad.append('a stage captured on the other side')
        pc = M.prefix_check(Bx, pol, cst, stop, sv['arms']['POST_PI0'], nd0, nf0)
        if not (pc['prefix_same'] and pc['callbacks_ok']):
            bad.append(f'prefix_check: a valid prefix refused ({pc})')
        sv2 = copy.deepcopy(sv['arms']['POST_PI0']); i = next(j for j, d in enumerate(sv2['decision_list']) if 0. < float(d[0]) <= stop); sv2['decision_list'][i][3] = 'X'
        if M.prefix_check(Bx, pol, cst, stop, sv2, nd0, nf0)['prefix_same']:
            bad.append('prefix_check: a changed saved decision was accepted')

        class Short:
            log = pol.log[:-1]; bind_bad = 0; bind_checked = len(pol.log) - 1
        if M.prefix_check(Bx, Short, cst, stop, sv['arms']['POST_PI0'], nd0, nf0)['callbacks_ok']:
            bad.append('prefix_check: a dropped callback was accepted')
        cf = ctx_of(BE.tape([fr(0, 0, 50, 52, 0, 100)], [(110, 0, 50, 10), (600, 0, 50, 20)], 51.), 0); svf = saved_for(cf, 0, 200.)
        Bf, polf, cstf, _, stopf, _, nd0f, nf0f = M.prefix(cf, None, None, None, 7, cls=stub_cls(0, 200.)); sa = svf['arms']['POST_PI0']
        if not M.prefix_check(Bf, polf, cstf, stopf, sa, nd0f, nf0f)['prefix_same']:
            bad.append('prefix_check: a valid prefix with a fill and a request refused')
        for lab, fn in (('fill', lambda x: x['fill_list'][0].__setitem__(4, float(x['fill_list'][0][4]) + 1.)),
                        ('request', lambda x: x['requests'][0][3].__setitem__('effect', 999.))):
            x = copy.deepcopy(sa)
            if not x['fill_list' if lab == 'fill' else 'requests']:
                bad.append(f'prefix_check: no saved {lab} to change'); continue
            fn(x)
            if M.prefix_check(Bf, polf, cstf, stopf, x, nd0f, nf0f)['prefix_same']:
                bad.append(f'prefix_check: a changed saved {lab} was accepted')
        import types
        nsB = types.SimpleNamespace(decisions=[(0., 1, 'absent', 'WAIT', None), (0., 0, 'absent', 'POST', 0), (50., 1, 'absent', 'WAIT', None), (50., 0, 'post_pending', 'WAIT', 0)],
                                    fallbacks=[(50., 0, 'x', 'feature_unavailable')], E=types.SimpleNamespace(fills=[], events=[]))
        nsP = types.SimpleNamespace(log=[1], bind_bad=0, bind_checked=1); good = dict(checked=1, mismatch=0, parity_mismatch=0)        # 2 after the fork - the POST - 1 fallback
        if not (M.prefix_check(nsB, nsP, good, 50., dict(decision_list=M.jl(nsB.decisions), fill_list=[], requests=[]), 1, 0)['callbacks_ok']
                and not M.prefix_check(nsB, nsP, dict(good, checked=2), 50., dict(decision_list=M.jl(nsB.decisions), fill_list=[], requests=[]), 1, 0)['callbacks_ok']):
            bad.append('prefix_check: the callback denominator (forced POST and engine fallbacks excluded) or the probe count')
        if M.unavoidable_of(dict(value=-5.), dict(value=-2.)) != -3.:
            bad.append('unavoidable_of: the entry value less the stage value')
        if not (M.effect_of(dict(act=58., submit=0.), 30.) == 95. and M.effect_of(dict(act=58., submit=0.), 100.) == 137.):
            bad.append('effect_of: the engine\'s request rule')
        gid_h, Lh, gen = R2.hold_from(ctx['state'], 0, 0); fh = [x for x in M.jl(Lh['fills']) if int(x[1]) == gid0]
        fk = [x for x in sv['arms']['POST_KEEP_INITIAL']['fill_list'] if int(x[1]) == gid0]
        if not (gid_h == gid0 and fh == fk and len(fh) == 1):
            bad.append('the neutral hold: its initial fills are not the KEEP arm\'s')
        tp = M.stage_target(fh, gid0, 0, 51., st['first_pull']['effect'], 51.)
        if not (abs(tp['value'] - sv['D_initial']) <= 1e-9 and tp['value'] == 30. and abs(tp['q'] - (sv['init_KEEP']['filled'] - sv['init_PI0']['filled'])) <= 1e-9
                and float(Bx.E.o[gid0]['remove']) == st['first_pull']['effect']):
            bad.append(f"the first-pull reconciliation ({tp}, D_initial {sv['D_initial']})")
        Bb, polb, _, _, stopb, whyb, _, _ = M.prefix(ctx, None, None, None, 7, cls=stub_cls(0, 0.))
        if not (polb.stage.get('first_pull', {}).get('same_as_review') is True and polb.stage['first_review']['action'] == 'PULL' and stopb == 100. and whyb == 'pull'):
            bad.append('a first review that pulls: not same_as_review')
        for lab, path, want in (('refused', BE.tape([fr(0, 0, 50, 52, 0, 100), fr(30, 30, 49, 50, 100, 100)], [(600, 0, 50, 30)], 51.), 'closed:refused'),
                                ('structural', BE.tape([fr(0, 0, 50, 52, 0, 100), fr(60, 60, 55, 57, 100, 100)], [(600, 0, 50, 30)], 51.), 'closed:removed')):
            _, pz, _, _, _, wz, _, _ = M.prefix(ctx_of(path, 0), None, None, None, 7, cls=stub_cls(0, 200.))
            if pz.stage or wz != want:
                bad.append(f'{lab}: a stage captured or the stop reason {wz} is not {want}')
        for lab, tr, want_e, want_r in (('a loss before the review effect', [(110, 0, 50, 30)], -30., 0.), ('a loss after it', [(600, 0, 50, 30)], -30., -30.)):
            cz = ctx_of(BE.tape([fr(0, 0, 50, 52, 0, 100)], tr, 49.), 0); gz, Lz, gz_gen = R2.hold_from(cz['state'], 0, 0); fz = M.jl(Lz['fills'])
            e_ = M.stage_target(fz, gz, 0, 49., gz_gen['act'], 51.); r_ = M.stage_target(fz, gz, 0, 49., 137., 51.)
            if not (e_['value'] == want_e and r_['value'] == want_r and abs(r_['qm'] - (r_['value'] - (51. - 50.) * r_['q'])) <= 1e-12):
                bad.append(f'{lab}: entry {e_["value"]} / review {r_["value"]}, not {want_e} / {want_r}')
        _, pl, _, _, _, _, _, _ = M.prefix(ctx, None, None, None, 7, cls=stub_cls(0, leak=True))
        if pl.stage.get('first_review', {}).get('hidden_same') is not False:
            bad.append('a decision reading the branch object was not flagged by the hidden copy')
        if st['first_review'].get('hidden_same') is not True:
            bad.append('a view-only decision was flagged by the hidden copy')
    except Exception as e:                                                                  # noqa: BLE001  (a valid crafted case must run)
        bad.append(f'crafted: a case stopped ({e!r})'[:220])
    # ---- 2 fork_stages end to end on the cached synthetic world
    try:
        wroot = Path(os.environ['PF_TEACHER_SYN']); wj = json.loads((wroot / 'world.json').read_text()); day = '20261024'
        W = L.worlds_of_day(day, Path(wj['kp']), 2); _, prs, fair, depth, ff = L.world_inputs(day, W, R2.KP.Syn(), R2.X.syn_inputs, R2.X.syn_features())
        Mst, sst = LT.stubs(-12.), LT.OpenSupport(); done = False
        for w in W:
            wb = L.world_bind(day, w, prs, fair, depth, ff)[0]
            if wb is None or done:
                continue
            for s in (0, 1):
                sel = dict(day=day, side=s, decision_ms=0, source='neutral', shadow_source='entry', offset_ms=0, kind='post')
                ctx = L.fork_for(sel, wb, Mst, sst); stp = L.pair_bound(ctx, L.teacher_factory(Mst, sst, wb, every=M.AUDIT_EVERY), 'post')
                if stp.get('status') != 'ok':
                    continue
                f = dict(day=day, m=int(w['m']), t_ms=int(round(float(w['t']))), hour=0, side=s, oid=1, status='ok', sel=sel, fork_fingerprint=ctx['fingerprint'], Y=stp['Y'],
                         arms={n: {h: a[h] for h in ('value', 'fills_sha256', 'decisions_sha256', 'requests_sha256')} for n, a in stp['arms'].items()})
                sv = json.loads(json.dumps(EX.fork_package(f, wb, Mst, sst), default=str)); r = M.fork_stages(f, wb, Mst, sst, sv); ck = r.get('checks', {})
                e_, rv = r.get('stages', {}).get('entry', {}), r.get('stages', {}).get('first_review', {})
                if not (r['status'] == 'ok' and all(ck.get(k_) is True for k_ in ('horizon_ok', 'prefix_same', 'callbacks_ok', 'equivalent', 'causal_ok', 'moments_ok'))
                        and ck.get('pull_reconciled') in (True, None) and len(e_.get('inputs') or []) == 107 and len(rv.get('inputs') or []) == 107 and e_.get('inputs_sha256')
                        and (e_.get('moments') or {}).get('reproduced') and e_.get('admission') in ('would_POST', 'would_WAIT')):
                    bad.append(f"fork_stages: a valid synthetic fork ({r['status']}, {ck})"[:300]); done = True; break
                fp_ = r['stages'].get('first_pull', {})
                if not (e_['target']['eff'] == r['held']['act'] and rv['target']['eff'] == rv['effect'] and abs(rv['unavoidable'] - (e_['target']['value'] - rv['target']['value'])) <= 1e-12
                        and (e_['admission'] == 'would_POST') == (e_['action'] == 'POST') and fp_.get('status') == 'ok'):
                    bad.append('fork_stages: the stage target times, the unavoidable value, the admission or the first pull')
                row_ = {c_: np.float32(x_) for c_, x_ in zip(M.FEATS, rv['inputs'])}; row_['price'] = rv['price']; mm = rv['moments']
                if M.moments(Mst, dict(row=row_, head='retain', V=mm['V'], P=mm['P']), s)['reproduced'] is not True or \
                        M.moments(Mst, dict(row=row_, head='retain', V=mm['V'] + 1., P=mm['P']), s)['reproduced'] is not False:
                    bad.append('moments: a logged V that differs was reproduced, or the true one was not')
                tampers = [('D_initial', lambda x: x.update(D_initial=x['D_initial'] + 1.), 'pull_not_reconciled'),
                           ('a PI0 initial fill before the effect', lambda x: x['arms']['POST_PI0']['fill_list'].append([r['stop_ms'] + 10., int(x['init_PI0']['gid0']), s, 1., 50., 0]),
                            'pull_not_reconciled'),
                           ('END', lambda x: x['arms']['POST_PI0'].update(end=6000.), 'horizon_mismatch'),
                           ('a decision', lambda x: x['arms']['POST_PI0']['decision_list'][next(j for j, d in enumerate(x['arms']['POST_PI0']['decision_list'])
                                                                                                  if 0. < float(d[0]) <= r['stop_ms'])].__setitem__(3, 'X'), 'prefix_mismatch'),
                           ('a KEEP initial fill', lambda x: x['arms']['POST_KEEP_INITIAL']['fill_list'].append([4999., int(x['init_KEEP']['gid0']), s, 1., 50., 0]), 'target_not_equivalent')]
                if ck.get('pull_reconciled') is None:
                    tampers = tampers[2:]; bad.append('fork_stages: the synthetic fork has no first pull to reconcile')
                for lab, fn, want in tampers:
                    x = copy.deepcopy(sv); fn(x); got = M.fork_stages(f, wb, Mst, sst, x)['status']
                    if got != want:
                        bad.append(f'fork_stages: {lab} changed gave {got}, not {want}')
                done = True; break
        if not done:
            bad.append('fork_stages: no synthetic fork ran')
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'fork_stages: a valid synthetic fork stopped ({e!r})'[:220])
    # ---- 3 the measurements on hand rows
    try:
        F = M.FEATS
        if not (M.group_of('t_rel') == 'time' and M.group_of('ctx_generation_age_ms') == 'time' and M.group_of('fair_edge') == 'fair' and M.group_of('o_edge') == 'order_state'
                and M.group_of('ctx_inventory_shares') == 'order_state' and M.group_of('pf_mid') == 'market_flow' and M.group_of('adv_ofi250') == 'market_flow'):
            bad.append('group_of: the declared input groups')
        x0 = [float(i) for i in range(107)]; x1 = list(x0); x1[F.index('o_age')] += 50.; x1[F.index('t_rel')] += .01
        mo = lambda V: dict(P=.5, Eq=5., Eqm=V - 1., V=V, edge=.2, mech=1., q_fill=10., qm_fill=2. * (V - 1.), S=None, ok=True)
        tg = lambda v: dict(q=10. if v else 0., value=v, qm=v - 2., severe=max(-v - 30., 0.))
        fork = dict(day='20261024', m=1, t_ms=1, hour=0, side=0, status='ok', stages=dict(
            entry=dict(status='ok', dec_status='model', action='POST', admission='would_POST', inputs=x0, moments=mo(2.), target=tg(-5.)),
            first_review=dict(status='ok', dec_status='model', action='PULL', inputs=x1, moments=mo(-2.), target=tg(-5.), unavoidable=0., effect=137., t=100.),
            first_pull=dict(status='ok', dec_status='model', action='PULL', inputs=x1, moments=mo(-2.), target=tg(-5.), unavoidable=0., same_as_review=True, effect=137., t=100.)))
        S1 = M.stage_table([fork]); rv = S1[S1.stage == 'first_review'].iloc[0]
        if not (rv.chg_market_flow == 0 and rv.chg_fair == 0 and rv.chg_time == 2 and rv.chg_order_state == 0 and rv.dV == -4. and bool(rv.flip) and len(S1) == 3):
            bad.append('stage_table: the unchanged-market transition')
        rng = np.random.default_rng(3); rows = []
        for st_ in ('entry', 'first_review'):
            for h in range(4):
                vals = np.sort(rng.normal(0, 10, 30))
                for j, v in enumerate(vals):
                    act = (('WAIT' if j < 15 else 'POST') if st_ == 'entry' else ('PULL' if rng.uniform() < .5 else 'KEEP'))
                    rows.append(dict(day='20261024', m=100 * h + j, t_ms=j, hour=h, side=0, stage=st_, fork_status='ok', stage_status='ok', dec_status='model', action=act,
                                     admission='would_WAIT' if (st_ == 'entry' and j < 15) else 'would_POST', P=.5, Eq=5., Eqm=0., V=float(v) if st_ == 'entry' else 0.,
                                     edge=0., mech=0., q_fill=10., qm_fill=0., q=10. if j % 2 else 0., value=float(v), qm=0.))
        rows += [dict(rows[0], m=9000 + j, dec_status='feature_unavailable') for j in range(4)]
        S2 = pd.DataFrame(rows); sel = M.selections(S2); P2 = M.predicates(S2)
        S4 = S2.copy(); e4 = S4.stage == 'entry'; S4.loc[e4, 'action'] = 'POST'; S4.loc[e4 & (S4.t_ms < 2), 'action'] = 'WAIT'; P4 = M.predicates(S4)
        if not (P4['selection']['entry']['thin'] and P4['selection']['entry']['percentile'] <= .05 and P4['discriminates']['entry'] is False):
            bad.append('predicates: a thin planted selection discriminates')
        if not (sel['entry']['percentile'] is not None and sel['entry']['percentile'] <= .01 and not sel['entry']['thin'] and sel['entry']['n_sel'] == 60
                and .05 < sel['first_review']['percentile'] < .95 and P2['readings']['entry_identifiable'] and not P2['readings']['review_protective']
                and P2['value_fails']['first_review'] and not P2['value_fails']['entry'] and P2['readings']['forecasts_fail'] and not P2['readings']['not_distinguishable']):
            bad.append(f"selection / predicates on hand rows ({sel['entry'].get('percentile')}, {sel['first_review'].get('percentile')}, {P2['readings']})")
        S3 = S2[S2.stage == 'entry'].assign(action='WAIT'); s3 = M.selection_null(S3.reset_index(drop=True), S3.reset_index(drop=True).action == 'WAIT')
        if not (s3['degenerate_strata'] == s3['strata'] == 4 and s3['percentile'] == 1. and s3['thin']):
            bad.append('selection_null: an all-selected population is not degenerate / thin')
        cal = {(c['stage'], c['day']): c for c in M.calibration(S2)}; c = cal[('entry', 'all')]
        if not (c['n'] == 120 and c['brier'] == .25 and c['fill_rate'] == .5 and c['mae_V'] == 0. and abs(c['mae_zero'] - float(np.abs(S2[(S2.stage == 'entry') & (S2.dec_status == 'model')].value).mean())) < 1e-12):
            bad.append(f'calibration: hand values ({c})')
        bn = [b for b in M.bins(S2) if b['stage'] == 'entry' and b['n']]
        if not (len(bn) == 1 and bn[0]['p_lo'] == .4 and bn[0]['n'] == 120) or M.PBINS != (0., .01, .02, .05, .1, .2, .4, 1.):
            bad.append('bins: the declared probability bins')
        dd = {(d['stage'], d['vsign'], d['admission']): d for d in M.decisions(S2)}; a_ = dd[('entry', 'all', 'all')]
        ev = S2[S2.stage == 'entry'].value
        if not (a_['n'] == 124 and a_['n_benefit'] == int((ev > 1e-6).sum()) and abs(a_['sum_loss'] - float(ev[ev < -1e-6].sum())) < 1e-9 and a_['worlds'] == 124):
            bad.append('decisions: hand counts')
        if not (M.NULL_DRAWS == 500 and M.THIN == 30 and M.PCT == .05 and M.SEED == 20261121 and M.MEANINGFUL == 1e-6 and M.SEVERE == 30.):
            bad.append('the declared constants')
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'measurements: stopped ({e!r})'[:220])
    # ---- 4 the writers
    try:
        with tempfile.TemporaryDirectory(dir=str(R2.I.K.IO)) as tmp:
            tmp = Path(tmp); out = tmp / 'out'; (out / 'parts').mkdir(parents=True); mp = tmp / 'manifest.json'
            mp.write_text(json.dumps(dict(parts=[dict(set='train', day='20261024', worlds=[dict(day='20261024', m=1, t_ms=1)])])))
            b = LS.gz_bytes([fork, dict(fork, side=1)]); (out / 'parts' / 'part_000.jsonl.gz').write_bytes(b)
            import hashlib
            (out / 'parts' / 'part_000.json').write_text(json.dumps(dict(sha256=hashlib.sha256(b).hexdigest(), sources=M.sources(), manifest_sha256=C.sha(mp))))
            M.measure(out, mp); js = json.loads((out / 'retention.json').read_text())
            if not (js['forks'] == 2 and all((out / f'{n}.csv').exists() for n in ('stages', 'calibration', 'bins', 'decisions', 'changes')) and 'readings' in js['predicates']):
                bad.append('measure: the written outputs')
            try:
                M.measure(out, mp); bad.append('measure: an existing measurement was overwritten')
            except Exception:                                                               # noqa: BLE001  (the refusal)
                pass
            out2 = tmp / 'out2'; shutil_copy = __import__('shutil').copytree; shutil_copy(out / 'parts', out2 / 'parts')
            (out2 / 'parts' / 'part_000.jsonl.gz').write_bytes(LS.gz_bytes([fork, fork]))
            try:
                M.measure(out2, mp); bad.append('measure: a partition whose bytes differ from its record was read')
            except Exception:                                                               # noqa: BLE001  (the refusal)
                pass
            fake = tmp / 'fake_expo'; fake.mkdir(); (fake / 'manifest.json').write_text('{}'); old = M.EXPO; M.EXPO = fake
            try:
                M.run_part(0, out=tmp / 'rp'); bad.append('run_part: an uncommitted exposure manifest was used')
            except Exception as e:                                                          # noqa: BLE001  (the refusal)
                if 'exposure manifest' not in str(e):
                    bad.append(f'run_part: refused for another reason ({e!r})'[:200])
            finally:
                M.EXPO = old
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'writers: stopped ({e!r})'[:220])
    if bad:
        sys.exit(FAIL + ': ' + '; '.join(bad))
    print('entry retention selftest passed: crafted tapes (distinct pull, same-as-review pull, refusal, structural removal, other side, prefix and callback '
          'refusals, neutral hold, target split, first-pull reconciliation, hidden-state leak); fork_stages end to end with four refusals; groups, '
          'transition, calibration, bins, decisions, matched nulls, predicates; writers (measure immutable / refused; run_part refused)')
