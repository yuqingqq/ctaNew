"""SELFTEST of pf_r2_labels (synthetic only; no market data). run(L) checks the module object L (the file or a mutated copy) and exits with
'R2 LABELS SELFTEST FAILED' on any failure. Each block is a computed predicate:
 (A) stand-in parity: pair_bound with a stateless policy equals pf_branch_engine.pair exactly (status, Y, every arm's value / cash /
     position and fill / decision hashes). Run on random engine tapes for 5 policies x 8 ticks x 2 sides x 2 kinds;
 (B) arm binding on the synthetic world with stub forecasts (orders go live): every arm's binding check passes, with callbacks made.
     Falsifiers: one teacher bound to the fork copy for both arms, and the first arm's teacher reused in the second; each must report
     binding failures;
 (C) reversal: executing the arms in reverse order leaves every labeled output unchanged;
 (D) a single branch: the binding-checked teacher reproduces the plain frozen teacher's decisions and ledger;
 (E) fork identity: bound_fork = pf_branch_engine.fork_state for a stateless roll-in; the pi0 roll-in = the census rollout's state
     (fingerprint);
 (F) reconciliation falsifiers: a ledger with cash + 1, and one with a fill above its generation's size, fail;
 (G) selection: the declared round robin over the sources, ok candidates only, the cap, deterministic;
 (H) end to end (synthetic world, stub forecasts): census -> manifest (immutable, content-only) -> pairs -> finalize.
     - An interrupted + resumed build equals the uninterrupted one (every record's result byte for byte), and the resume leaves the
       finished records untouched.
     - A tampered fork fingerprint is refused.
     - A STOP_HARD_RSS status holds the build until a disposition of its identity and this manifest.
 (I) the frozen teacher's pins: a wrong freeze sha and a library mismatch are refused;
 COVERAGE (computed, required): post and keep forks on both sides; a delayed post; a later generation in an arm; a partial known fill at a
 keep fork; a fill after a cancel request; the remaining side deciding at the fork tick. All of these occur among the checked pairs.
PF_TEACHER_SYN: the cached synthetic world (pf_r2_teacher_selftest.build_world)."""
import contextlib
import copy
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_branch_engine as BE                             # noqa: E402
import pf_r2_shadow as R2                                 # noqa: E402
import pf_r2_teacher as T                                 # noqa: E402
import pf_r2_teacher_selftest as TS                       # noqa: E402

C, pd, I, KP, X, QP = R2.C, R2.pd, R2.I, R2.KP, R2.X, R2.QP
FAIL = 'R2 LABELS SELFTEST FAILED'


class OpenSupport:
    def check(self, s, head, row):
        return None


def stubs(v_retain=0.):
    """Stub forecasts: V > 0 at the JOIN (POST when absent); live orders keep or pull with the sign of edge E[q] + E[q move]."""
    return {g: dict(p=TS.stub([.5]), q=TS.stub([10.]), qm=TS.stub([v_retain]), severe=TS.stub([0.])) for g in T.GROUPS}


def must(fn, what):
    try:
        return fn()
    except ValueError as e:
        raise SystemExit(f'{FAIL}: {what} refused valid input ({str(e)[:200]})')


def pb(L, bad, what, *a, **kw):
    """pair_bound on a valid fork: the adapter's own refusal (ValueError) is a check failure, recorded, never a crash."""
    try:
        return L.pair_bound(*a, **kw)
    except ValueError as e:
        bad.append(f'{what}: pair_bound refused a valid fork ({str(e)[:80]})'); return dict(status='refused')


def cover(res, kind, k, side, cv):
    if res.get('status') != 'ok':
        return
    cv.add(f"{kind} {'bid' if side == 0 else 'ask'}")
    if kind == 'post' and k > 0:
        cv.add('delayed post')
    if (res['k'] + 0) % 2 == side:
        cv.add('remaining side at the fork tick')
    if kind == 'keep' and 0 < res['fork_known_filled'] < 30.:
        cv.add('partial known fill at a keep fork')
    for a in res['arms'].values():
        if max(a['gens_bid'], a['gens_ask']) >= 2:
            cv.add('later generation')
        if a['fills_after_request'] > 0:
            cv.add('fill after a cancel request')


def run(L):  # noqa: C901
    bad = []; cv = set()
    # ---- (A) stand-in parity on random engine tapes
    pols = [BE.POST_KEEP, BE.CHASE, BE.reactive(1.), BE.NEVER, BE.ONCE_PULL_FIRST]; n_par = 0; n_ok = 0
    for seed in (3, 11):
        path = BE.rand_tape(np.random.default_rng(seed), n_frames=60, n_trades=160)
        for pol in pols:
            for k in (0, 2, 5, 9, 17, 33, 61, 90):
                for side in (0, 1):
                    ctx = BE.fork_state(path, pol, k, side)
                    for kind in L.KINDS:
                        r1 = BE.pair(path, pol, k, side, kind, ctx=ctx); n_par += 1
                        try:
                            r2 = L.pair_bound(ctx, L.standin_factory(pol), kind)
                        except ValueError as e:
                            bad.append(f'parity: pair_bound refused a valid fork ({str(e)[:80]})'); break
                        same = r1['status'] == r2['status']
                        if same and r1['status'] == 'ok':
                            n_ok += 1; cover(r2, kind, k, side, cv)
                            same = abs(r1['Y'] - r2['Y']) <= 1e-12 and all(
                                abs(r1['arms'][n]['value'] - r2['arms'][n]['value']) <= 1e-12 and r1['arms'][n]['cash'] == r2['arms'][n]['cash']
                                and r1['arms'][n]['position'] == r2['arms'][n]['position'] and L._h(r1['arms'][n]['fills']) == r2['arms'][n]['fills_sha256']
                                and L._h(r1['arms'][n]['decisions']) == r2['arms'][n]['decisions_sha256'] and r2['arms'][n]['reconciled'] for n in r1['arms']) and r2['reconciled']
                        if not same:
                            bad.append(f'parity: pair_bound differs from pf_branch_engine.pair ({pol.name} k {k} side {side} {kind}: {r1.get("status")} / {r2.get("status")})')
                            break
    # ---- the synthetic world
    wroot = Path(os.environ['PF_TEACHER_SYN']) if os.environ.get('PF_TEACHER_SYN') else None
    with tempfile.TemporaryDirectory(dir=str(I.K.IO)) as tmp:
        tmp = Path(tmp)
        if wroot is None:
            wroot = TS.build_world(tmp / 'world')
        wj = json.loads((wroot / 'world.json').read_text()); day = '20261024'; kp = Path(wj['kp']); src = KP.Syn(); ff = X.syn_features()
        W = L.worlds_of_day(day, kp, 3); src_, prs, fair, depth, ff = L.world_inputs(day, W, src, X.syn_inputs, ff)
        worlds = [L.world_bind(day, w, prs, fair, depth, ff)[0] for w in W]; worlds = [(w, wb) for w, wb in zip(W, worlds) if wb is not None]
        if not worlds:
            raise SystemExit(f'{FAIL}: no synthetic world with an episode')
        M, sup = stubs(-12.), OpenSupport(); n_b = 0
        # ---- (B) binding, (C) reversal, coverage with the teacher
        for w, wb in worlds:
            for k in (0, 3, 10, 21, 40):
                for side in (0, 1):
                    ident = R2.episode_ident(day, w['o'][side]); ctx = L.bound_fork(wb['path'], L.teacher_factory(M, sup, wb), k, side, ident)
                    for kind in L.KINDS:
                        fac = L.teacher_factory(M, sup, wb, every=3)
                        try:
                            r = L.pair_bound(ctx, fac, kind)
                        except ValueError as e:
                            bad.append(f'binding: pair_bound refused a valid fork ({str(e)[:80]})'); break
                        if r.get('status') != 'ok':
                            continue
                        n_b += 1; cover(r, kind, k, side, cv)
                        tch = [a['teacher'] for a in r['arms'].values()]
                        cz = [st for _, st in fac.made]
                        if not (all(t_['binding_bad'] == 0 and t_['binding_checked'] > 0 for t_ in tch) and r['reconciled']
                                and all(c_['mismatch'] == 0 and c_['parity_mismatch'] == 0 for c_ in cz) and sum(c_['checked'] for c_ in cz) > 0):
                            bad.append(f'binding: an arm-bound pair failed its binding / reconciliation / causal checks ({tch}, {cz[:1]})'); break
                        rr = pb(L, bad, 'reversal', ctx, L.teacher_factory(M, sup, wb, every=10 ** 9), kind, reverse=True)
                        r0 = pb(L, bad, 'reversal', ctx, L.teacher_factory(M, sup, wb, every=10 ** 9), kind)
                        if not L.same_pair(r0, rr):
                            bad.append(f'reversal: executing the arms in reverse order changed the outputs (k {k} side {side} {kind})'); break
        if n_b == 0:
            bad.append('binding: no teacher-bound pair was checked')
        w, wb = worlds[0]; ident = R2.episode_ident(day, w['o'][0]); ctx = None
        for kf in (12, 15, 20, 25, 30, 8, 5, 3, 40, 50):                                   # the first fork with a feasible pair (declared search)
            c_ = L.bound_fork(wb['path'], L.teacher_factory(M, sup, wb), kf, 0, ident); kd = 'keep' if copy.deepcopy(c_['state']).view(kf * L.STEP, 0)['state'] == 'live' else 'post'
            if pb(L, bad, 'stale search', c_, L.teacher_factory(M, sup, wb, every=10 ** 9), kd).get('status') == 'ok':
                ctx, kind_s = c_, kd; break
        if ctx is None:
            raise SystemExit(f'{FAIL}: no feasible fork for the stale-binding falsifier')
        stale = L.ArmTeacher(M, sup); stale.bind(copy.deepcopy(ctx['state']), wb)
        rs = pb(L, bad, 'stale', ctx, lambda B: stale, kind_s)
        reuse = {}
        def first_arm(B):
            if 't' not in reuse:
                reuse['t'] = L.ArmTeacher(M, sup); reuse['t'].bind(B, wb)
            return reuse['t']
        ru = pb(L, bad, 'reused arm', ctx, first_arm, kind_s)
        if not (rs.get('status') == 'ok' and stale.bind_bad > 0 and ru.get('status') == 'ok' and reuse['t'].bind_bad > 0):
            bad.append(f'binding: a stale-bound teacher ({stale.bind_bad}) or a reused first-arm teacher ({reuse.get("t") and reuse["t"].bind_bad}) is not caught')
        # ---- (D) a single branch
        for w, wb in worlds[:2]:
            out_ = []
            for cls in (T.Teacher, L.ArmTeacher):
                t_ = cls(M, sup); B = BE.Branch(wb['path'], 0, ident=R2.episode_ident(day, w['o'][0])); t_.bind(B, wb)
                for k in range(L.NT):
                    BE.step(B, k, t_)
                Lg = BE.finish(B); out_.append(([(d_['t'], d_['side'], d_['status'], d_['action']) for d_ in t_.log], L._h(Lg['fills']), Lg.get('value')))
            if out_[0] != out_[1]:
                bad.append('single branch: the binding-checked teacher does not reproduce the frozen teacher')
        # ---- (E) fork identity
        w, wb = worlds[0]
        for k, side in ((0, 0), (7, 1), (30, 0)):
            ident = R2.episode_ident(day, w['o'][side])
            a = L.bound_fork(wb['path'], L.standin_factory(BE.CHASE), k, side, ident); b = BE.fork_state(wb['path'], BE.CHASE, k, side, 0, None, ident)
            if a['fingerprint'] != b['fingerprint']:
                bad.append(f'fork: bound_fork differs from pf_branch_engine.fork_state (k {k} side {side})')
        rows_w, _ = L.census_world(day, 0, w, wb, M, sup); okp = [r for r in rows_w if r['source'] == 'pi0' and r['status'] == 'ok']
        ph = {(s_, hd): R2.phase(day, int(w['o'][s_].m), int(w['o'][s_].oid), s_, hd) for s_ in (0, 1) for hd in ('post', 'retain')}
        kk_ = lambda r: int(round(r['decision_ms'] / L.STEP))
        samp_ok = all((kk_(r) == 0 or kk_(r) % 5 == ph[(r['side'], 'post')]) if r['kind'] == 'post' else kk_(r) % 5 == ph[(r['side'], 'retain')] for r in okp)
        has_uns = any(r['source'] == 'pi0' and r['status'] == 'unsampled' for r in rows_w)
        naming = all((r['source'] == 'neutral') == (float(r['offset_ms']) == 0.) for r in rows_w if r['shadow_source'] in ('entry', 'retain'))
        no_out = not any(set(r) & set(L.OUTCOMES) for r in rows_w)
        if not (okp and samp_ok and has_uns and naming and no_out):
            bad.append(f'census: pi0 sampling {samp_ok} / unsampled present {has_uns}, neutral / delayed naming {naming}, outcome-free rows {no_out}')
        for r in okp[:6]:
            c_ = L.bound_fork(wb['path'], L.teacher_factory(M, sup, wb, every=10 ** 9), int(round(r['decision_ms'] / L.STEP)), int(r['side']),
                              R2.episode_ident(day, w['o'][int(r['side'])]))
            if c_['fingerprint'] != r['fingerprint']:
                bad.append(f"fork: the pi0 roll-in differs from the census rollout's state (k {r['decision_ms']} side {r['side']})"); break
        if not okp:
            bad.append('fork: no pi0 candidate to compare')
        # ---- (F) reconciliation falsifiers
        ctx = L.bound_fork(wb['path'], L.standin_factory(BE.POST_KEEP), 20, 0, R2.episode_ident(day, w['o'][0]))
        Bx = copy.deepcopy(ctx['state'])
        for kk in range(21, L.NT):
            BE.step(Bx, kk, BE.POST_KEEP)
        Lg = BE.finish(Bx); f0 = Lg['fills'][0] if Lg['fills'] else None
        clean_ok = L.reconcile_arm(Lg, Bx); mk = Bx.mark
        bad_cash = L.reconcile_arm(dict(Lg, cash=Lg['cash'] + 1., value=Lg['value'] + 1.), Bx)                       # only the cash-from-fills guard
        bad_pos = L.reconcile_arm(dict(Lg, position=Lg['position'] + 1., value=Lg['value'] + mk), Bx)               # only the position-from-fills guard
        bad_val = L.reconcile_arm(dict(Lg, value=Lg['value'] + 1.), Bx)                                            # only the value identity
        if f0:
            sg0 = 1. if f0[2] == 0 else -1.
            Lz = dict(Lg, fills=Lg['fills'] + [(f0[0], f0[1], f0[2], 31., f0[4], f0[5])], cash=Lg['cash'] - sg0 * 31. * f0[4], position=Lg['position'] + sg0 * 31.)
            Lz['value'] = Lz['cash'] + Lz['position'] * mk; bad_size = L.reconcile_arm(Lz, Bx)                     # only the size guard
        else:
            bad_size = None
        if not (clean_ok and bad_cash is False and bad_pos is False and bad_val is False and bad_size is False):
            bad.append(f'reconciliation: clean {clean_ok}; cash {bad_cash}, position {bad_pos}, value {bad_val}, a fill above size {bad_size} must each fail alone')
        # ---- (G) selection
        cands = []
        for i, src_n in enumerate(L.SOURCES):
            for j in range(5):
                cands.append(dict(day='d', side=0, kind='post', source=src_n, world=j, decision_ms=50. * j, offset_ms=np.nan, generation=1., status='ok'))
        cands.append(dict(day='d', side=0, kind='post', source='neutral', world=0, decision_ms=-1., offset_ms=np.nan, generation=1., status='unsampled'))
        sel = L.select(cands, cap=7); got = [(r['source'], r['world']) for r in sel]
        want = [(s_, 0) for s_ in L.SOURCES] + [('neutral', 1), ('delayed', 1)]
        if got != want or [r['order'] for r in sel] != list(range(7)) or json.dumps(L.select(cands, cap=7), sort_keys=True, default=str) != json.dumps(sel, sort_keys=True, default=str):
            bad.append(f'selection: {got} is not the declared round robin {want} (ok only, the cap, deterministic)')
        # ---- (H) end to end
        kw = dict(cfg=None, src=src, inputs=X.syn_inputs, feat_fn=ff, kp=kp, assets=(M, sup, None))
        o1, o2 = tmp / 'L1', tmp / 'L2'
        with contextlib.redirect_stdout(io.StringIO()):
            m1 = must(lambda: L.census(o1, days=(day,), prefix=3, cap=2, freeze_sha='synthetic', **kw), 'census 1')
            m2 = must(lambda: L.census(o2, days=(day,), prefix=3, cap=2, freeze_sha='synthetic', **kw), 'census 2')
            try:
                L.census(o1, days=(day,), prefix=3, cap=2, freeze_sha='synthetic', **kw); bad.append('manifest: an existing manifest was rewritten')
            except ValueError:
                pass
            s1 = must(lambda: L.pairs(o1, **{k_: v_ for k_, v_ in kw.items()}), 'pairs 1')
            s2a = must(lambda: L.pairs(o2, stop_after=2, **kw), 'pairs 2 (interrupted)')
            first2 = {p.name: p.read_bytes() for p in (o2 / 'pairs').glob('*.json')}
            s2b = must(lambda: L.pairs(o2, **kw), 'pairs 2 (resumed)')
        cd1 = L.load_candidates(o1); cnt1 = {}
        for r in cd1:
            kk = (r['day'], int(r['side']), r['kind'], r['source'], r['status']); cnt1[kk] = cnt1.get(kk, 0) + 1
        want_cnt = {(c_['day'], c_['side'], c_['kind'], c_['source'], c_['status']): c_['n'] for c_ in m1['candidate_counts']}
        if not ((o1 / 'candidates.jsonl.gz').read_bytes() == (o2 / 'candidates.jsonl.gz').read_bytes() and len(cd1) == m1['candidate_rows'] > 0
                and cnt1 == want_cnt and not any(set(r) & set(L.OUTCOMES) for r in cd1) and {r['source'] for r in cd1} == set(L.SOURCES)):
            bad.append(f"candidates: the streamed file differs between builds, or its rows ({len(cd1)}) / counts differ from the manifest ({m1['candidate_rows']}), "
                       f"or carry an outcome field, or miss a source")
        oc = tmp / 'Lc'; shutil.copytree(o1, oc); import gzip as _gz                       # a VALID gzip with one row changed: only the hash can refuse
        rows_t = [dict(r) for r in cd1]; rows_t[0]['status'] = 'tampered'
        with _gz.GzipFile(filename='', mode='wb', fileobj=open(oc / 'candidates.jsonl.gz', 'wb'), mtime=0) as g_:
            g_.write(''.join(json.dumps(r, sort_keys=True, default=str) + '\n' for r in rows_t).encode())
        try:
            L.load_candidates(oc); bad.append('candidates: a tampered candidates file was not refused')
        except ValueError:
            pass
        strip = lambda p: {k_: v_ for k_, v_ in json.loads(p.read_text()).items() if k_ not in ('asof', 'elapsed_s', 'hwm_mib', 'breach')}
        r1 = {p.name: strip(p) for p in (o1 / 'pairs').glob('*.json')}; r2 = {p.name: strip(p) for p in (o2 / 'pairs').glob('*.json')}
        if not ((o1 / 'manifest.json').read_bytes() == (o2 / 'manifest.json').read_bytes() and s1 == 'PAIRS_DONE' and s2a == 'STOPPED' and s2b == 'PAIRS_DONE'
                and len(first2) == 2 and r1 == r2 and len(r1) == len(m1['selected']) > 0 and all((o2 / 'pairs' / n).read_bytes() == b for n, b in first2.items())):
            bad.append(f'resume: the interrupted + resumed build differs from the uninterrupted one, or the resume rewrote a record ({s1}, {s2a}, {s2b}, {len(first2)}, {len(r1)}, {len(r2)})')
        cz1 = [json.loads(p.read_text()).get('causal') for p in (o1 / 'pairs').glob('*.json')]
        okc = [json.loads(p.read_text()) for p in (o1 / 'pairs').glob('*.json')]
        if not (all(isinstance(c_, list) for c_ in cz1) and all(len(rc['causal']) == 2 for rc in okc if rc['result'].get('status') == 'ok')
                and sum(a_['checked'] for rc in okc for a_ in rc['causal']) > 0 and all(a_['mismatch'] == 0 and a_['parity_mismatch'] == 0 for rc in okc for a_ in rc['causal'])):
            bad.append('records: the arm causal / parity probes are not recorded per arm, did not run, or mismatch')
        Dl = must(lambda: L.finalize(o1), 'finalize')
        if not (len(Dl) == len(m1['selected']) and Dl.status.eq('ok').any()):
            bad.append('finalize: the labels do not cover the selected pairs')
        for r in m1['selected']:
            res = json.loads((o1 / 'pairs' / L.rname(r)).read_text())['result']; cover(res, r['kind'], int(round(r['decision_ms'] / L.STEP)), int(r['side']), cv)
        o3 = tmp / 'L3'; shutil.copytree(o1, o3); mf = json.loads((o3 / 'manifest.json').read_text()); mf['selected'][0]['fork_fingerprint'] = '0' * 64
        (o3 / 'manifest.json').write_text(json.dumps(mf)); shutil.rmtree(o3 / 'pairs')
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                L.pairs(o3, **kw)
            bad.append('identity: a manifest fork fingerprint that the rebuilt fork does not match was not refused')
        except ValueError:
            pass
        o4 = tmp / 'L4'; shutil.copytree(o1, o4); shutil.rmtree(o4 / 'pairs'); (o4 / 'label_status.json').write_text(json.dumps(dict(state='STOP_HARD_RSS', rss_mib=2100.)))
        with contextlib.redirect_stdout(io.StringIO()):
            h1 = L.pairs(o4, **kw); ev = [json.loads(x) for x in (o4 / 'stops.jsonl').read_text().splitlines()] if (o4 / 'stops.jsonl').exists() else []
            if ev:
                (o4 / 'disposition.json').write_text(json.dumps(dict(plan_sha256=C.sha(o4 / 'manifest.json'), entries=[dict(id=ev[0]['id'], name='label_status', reason='selftest')])))
            h2 = L.pairs(o4, **kw)
        if not (h1 == 'HELD' and len(ev) == 1 and h2 == 'PAIRS_DONE'):
            bad.append(f'stops: a STOP_HARD_RSS status not held until its identity-bound disposition ({h1}, {len(ev)}, {h2})')
    # ---- (I) the frozen teacher's pins
    try:
        L.teacher_assets(T.REAL, freeze_sha='0' * 64); bad.append('pins: a wrong freeze sha was not refused')
    except ValueError:
        pass
    fz = json.loads((T.REAL['out'] / 'freeze.json').read_text())
    try:
        L.libraries_enforced(dict(fz, libraries=dict(fz['libraries'], lightgbm='0.0'))); bad.append('pins: a library mismatch was not refused')
    except ValueError:
        pass
    need = {'post bid', 'post ask', 'keep bid', 'keep ask', 'delayed post', 'later generation', 'partial known fill at a keep fork', 'fill after a cancel request',
            'remaining side at the fork tick'}
    if not need <= cv:
        bad.append(f'coverage: states never checked {sorted(need - cv)}')
    if bad:
        raise SystemExit(f'{FAIL}: ' + ' | '.join(bad))
    print(f'r2 labels selftests passed (synthetic only): parity with pf_branch_engine.pair on {n_par} forks ({n_ok} ok pairs), {n_b} teacher-bound pairs '
          f'(binding, causal, reversal), stale-bound and reused-arm teachers caught, single branch = frozen teacher, fork identity, reconciliation '
          f'falsifiers, selection, end to end (resume = uninterrupted, tampered fork refused, stop held), pins; coverage {sorted(cv)}')


if __name__ == '__main__':
    import pf_r2_labels as L_
    run(L_)
