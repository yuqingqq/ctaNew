"""SELFTEST of pf_adv2 (the ADV2 candidate, its extraction, fits, support and controller; 2026-11-21, session aea5bfb9). run(M) checks the
module object M (the file, or a mutated copy under pf_mutant_one.py); any failed check exits with FAIL. Synthetic only (no market data):
 1. inputs: the three declared lists pass; a list with an identity field, a wrong count or a changed hash is refused;
 2. the dataset reader: a synthetic accepted dataset reads back every row; a changed part byte, a parts list that does not hash to the dataset
    and another dataset hash are refused;
 3. extraction: per side / head rows, labels and raw weights exactly as written; every non-ok status counted, never trained; a crossing
    episode is the status crosses_day_boundary; a row whose inputs differ from their stored hash, a row of an undeclared day and a result of
    another head are refused; the manifest is deterministic and changes with one label;
 4. weights: one constant per set (mean 1, the relative weights unchanged);
 5. fits: deterministic (two fits, equal model text); each candidate's booster has exactly its declared inputs in order; fit_all saves,
    reloads bit-equal and refuses to overwrite a model; the training mean / zero / in-sample errors are the weighted formulas;
 6. support: the manifest from training rows; a supported row passes; a horizon outside the observed bins, fewer than 200 originals, an
    unseen category, a NUMERIC value outside its range and a NaN are each their own status;
 7. the controller on stub heads and stub inputs: POST iff > 0 (a tie and a negative WAIT), PULL iff < 0 (a tie and a positive KEEP); each
    side / head uses its own head; cap / near-END / feature-unavailable / unsupported / nonfinite fall back with their own status;
 8. the policies: N0 never posts; ENTRY_PA pulls every live review; ENTRY_ONLY / RETAIN_ONLY dispatch entry and retention to the declared
    sub-policies; an undeclared name is refused;
 9. end to end on the real engine and the cached synthetic world (stub heads, stub teacher): every policy's episode reconciles (cash /
    position / value from the fills, every generation within its size) with zero binding failures; N0 has no order and value 0; RICH with a
    negative POST head equals N0; RICH with positive POST / positive KEEP heads equals the engine's POST_KEEP; RICH with positive POST /
    negative KEEP equals ENTRY_PA (POST then PULL at the first review), ledger for ledger.
PF_TEACHER_SYN: the cached synthetic world (pf_r2_teacher_selftest.build_world)."""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_branch_engine as BE                             # noqa: E402
import pf_pipeline_bench as BX                            # noqa: E402
import pf_r2_label_sample as LS                           # noqa: E402
import pf_r2_labels as L                                  # noqa: E402
import pf_r2_labels_selftest as LT                        # noqa: E402
import pf_r2_shadow as R2                                 # noqa: E402
import pf_r2_teacher as T                                 # noqa: E402
import pf_r2_teacher_selftest as TS                       # noqa: E402

C, pd, I, KP, X = R2.C, R2.pd, R2.I, R2.KP, R2.X
FAIL = 'ADV2 SELFTEST FAILED'
BX_PARAMS = dict(BX.PARAMS)                                # the pinned dictionary, read independently of the module under test


def _refused(fn, msg=None):
    """True iff fn raises the module's own refusal (ValueError, with msg if given); another exception is not a refusal."""
    try:
        fn()
    except ValueError as e:
        return msg is None or msg in str(e)
    except Exception:                                                                       # noqa: BLE001  (a crash is not the check)
        return False
    return False


def _row(rng, day, m, t_ms, side, head, status='ok', y=None, w=None, dec=0, pick=None):
    x = rng.normal(size=107).astype(np.float32); x[T.FEATS.index('ctx_generation_ordinal')] = 1.
    base = dict(day=day, m=m, t_ms=t_ms, hour=int(t_ms // 3600000 % 24), side=side, oid=10 * m + side, head=head, pick=pick or ('mandatory' if head == 'post' else 'keep'),
                N=5, q=.2, w=float(w if w is not None else rng.uniform(.5, 3.)), pi=.1, world_rank='r', status=status)
    if status not in ('ok', 'feature_unavailable', 'pair_not_ok'):
        return dict(base, w=None, N=None, q=None)
    rk = dict(day=day, decision_ms=dec, generation=1, instrument_id=m, kind=head, offset_ms=0, original_episode_id=10 * m + side, shadow_source='entry', side=side, source='neutral')
    res = dict(status='ok', Y=float(y if y is not None else rng.choice([0., 0., 0., 5., -7.5])), kind=head, side=side, k=int(dec // 50), arms={})
    return dict(base, source='neutral', row_key=rk, row_rank='r', fork_fingerprint='f', causal=[],
                fork=dict(status='ok', inputs=[float(v) for v in x], inputs_sha256=hashlib.sha256(x.tobytes()).hexdigest(), price=40., support='supported'),
                result=res)


def _dataset(d, rows_by_part):
    d = Path(d); (d / 'parts').mkdir(parents=True, exist_ok=True); parts = []
    for k, rows in enumerate(rows_by_part):
        b = LS.gz_bytes(rows); LS.part_paths(d, k)[1].write_bytes(b)
        parts.append(dict(part=k, day=rows[0]['day'], sha256=hashlib.sha256(b).hexdigest(), pools_sha256='p', rows=len(rows)))
    sha = hashlib.sha256(LS.ckey(parts).encode()).hexdigest()
    (d / 'accepted.json').write_text(json.dumps(dict(accepted=True, parts=parts, dataset_sha256=sha)))
    return sha


class _Head:
    def __init__(self, v):
        self.v = v

    def predict(self, X, num_threads=1):
        return np.full(len(X), self.v, float)


def _view(state, side=0, t=100., room=0., gid=None):
    return dict(t=t, side=side, state=state, gid=gid, room=room)


class _B:
    end = 5000.


def run(M):  # noqa: C901
    bad = []
    rng = np.random.default_rng(7)
    # ---- 1 inputs
    try:
        M.check_inputs()
    except ValueError as e:
        bad.append(f'inputs: the declared lists were refused ({e})')
    for name, f in (('identity field', lambda l: l + ['day']), ('count', lambda l: l[:-1]), ('order (hash)', lambda l: l[::-1])):
        saved = M.INPUTS['FAIR']
        try:
            M.INPUTS['FAIR'] = f(list(saved))
            if not _refused(M.check_inputs):
                bad.append(f'inputs: a FAIR list with a changed {name} was accepted')
        finally:
            M.INPUTS['FAIR'] = saved
    if not (M.INPUTS['RICH'] == list(T.FEATS) and set(M.INPUTS['FAIR']) <= set(T.FEATS) and set(M.INPUTS['MECH']) <= set(T.FEATS)
            and len(M.INPUTS['FAIR']) == 18 and len(M.INPUTS['MECH']) == 40 and 'fair_edge' in M.INPUTS['FAIR'] and 't_to_end' in M.INPUTS['MECH']):
        bad.append('inputs: not the declared RICH 107 / FAIR 18 / MECH 40')
    with tempfile.TemporaryDirectory() as td:
        td = Path(td); day = '20261024'
        # ---- 2 / 3 the reader and the extraction
        P0 = [_row(rng, day, m, 3600000 * (m % 20) + 1000 * m, s, h, dec=50 * (m % 3)) for m in range(1, 41) for s in (0, 1) for h in ('post', 'keep')]
        P0 += [_row(rng, day, 99, 7200000, 0, 'keep', status='feature_unavailable'), _row(rng, day, 98, 7200000, 1, 'post', status='empty_pool'),
               _row(rng, day, 97, 7200000, 0, 'post', status='world:no_path:missing_endpoint')]
        P1 = [_row(rng, '20261025', m, 1000 * m, s, 'post') for m in range(41, 61) for s in (0, 1)]
        cross = _row(rng, '20261025', 200, 86400000 - 1000, 1, 'post')                         # t0 + END after midnight
        cross['t_ms'] = int(pd.Timestamp('2026-10-25', tz='UTC').value // 10 ** 6) + 86400000 - 1000; P1.append(cross)
        for r in P0 + P1:
            if r['day'] == '20261024' and r is not cross:
                r['t_ms'] = int(pd.Timestamp('2026-10-24', tz='UTC').value // 10 ** 6) + int(r['t_ms'])
            elif r is not cross:
                r['t_ms'] = int(pd.Timestamp('2026-10-25', tz='UTC').value // 10 ** 6) + int(r['t_ms'])
        sha = _dataset(td / 'd', [P0, P1]); rows = list(M.dataset_rows(td / 'd', sha))
        if len(rows) != len(P0) + len(P1):
            bad.append('reader: not every row read back')
        if not _refused(lambda: list(M.dataset_rows(td / 'd', 'x' * 64))):
            bad.append('reader: another dataset hash was accepted')
        b = LS.part_paths(td / 'd', 1)[1].read_bytes(); LS.part_paths(td / 'd', 1)[1].write_bytes(b[:-1] + bytes([b[-1] ^ 1]))
        if not _refused(lambda: list(M.dataset_rows(td / 'd', sha))):
            bad.append('reader: a changed part byte was accepted')
        LS.part_paths(td / 'd', 1)[1].write_bytes(b)
        acc = json.loads((td / 'd' / 'accepted.json').read_text()); acc2 = dict(acc, parts=acc['parts'][:1]); (td / 'd' / 'accepted.json').write_text(json.dumps(acc2))
        if not _refused(lambda: list(M.dataset_rows(td / 'd', sha))):
            bad.append('reader: a parts list that does not hash to the dataset was accepted')
        (td / 'd' / 'accepted.json').write_text(json.dumps(acc))
        try:
            E, st = M.extract(iter(rows), ('20261024', '20261025'))
        except ValueError as e:
            E, st = None, {}; bad.append(f'extract: refused valid rows ({e})')
        if E is not None:
            want = {}
            for r in P0 + P1:
                if r['status'] == 'ok' and r is not cross:
                    want.setdefault((int(r['side']), r['head']), []).append(r)
            for h, rr in want.items():
                e = E[h]
                if not (len(e['y']) == len(rr) and np.array_equal(e['y'], [x_['result']['Y'] for x_ in rr]) and np.array_equal(e['w'], [x_['w'] for x_ in rr])
                        and np.array_equal(e['X'], np.asarray([x_['fork']['inputs'] for x_ in rr], np.float32))):
                    bad.append(f'extract: {h} rows / labels / weights / inputs not exactly the written ones')
            mt_ok = all(m_['generation'] == 1. and m_['decision_ms'] == float(m_['row_key']['decision_ms']) for h in E for m_ in E[h]['meta'])
            if not mt_ok or not any(m_['decision_ms'] > 0 for h in E for m_ in E[h]['meta']):
                bad.append('extract: the metadata generation / decision ms are not the stored ones')
            if st.get('20261025 post crosses_day_boundary') != 1 or st.get('20261024 keep feature_unavailable') != 1 or st.get('20261024 post empty_pool') != 1 \
                    or st.get('20261024 post world:no_path:missing_endpoint') != 1:
                bad.append(f'extract: the statuses are not counted ({st})')
            m1 = M.manifest_of(E, st); m2 = M.manifest_of(E, st); E2 = {k: dict(v, y=v['y'].copy()) for k, v in E.items()}; E2[(0, 'post')]['y'][0] += 1.
            if m1 != m2 or M.manifest_of(E2, st)['heads']['0 post'] == m1['heads']['0 post']:
                bad.append('extract: the manifest is not deterministic or misses a changed label')
        r_bad = json.loads(json.dumps(P0[0])); r_bad['fork']['inputs'][3] += 1.
        if not _refused(lambda: M.extract(iter([r_bad]), ('20261024',)), 'stored hash'):
            bad.append('extract: a row whose inputs differ from their hash was accepted')
        if not _refused(lambda: M.extract(iter(P0[:2]), ('20261026', '20261027')), 'outside the declared days'):
            bad.append('extract: a training-day row was accepted as an evaluation row (split leakage)')
        r_h = json.loads(json.dumps(P0[0])); r_h['result']['kind'] = 'keep' if r_h['head'] == 'post' else 'post'
        if not _refused(lambda: M.extract(iter([r_h]), ('20261024',))):
            bad.append('extract: a result of another head was accepted')
        # ---- 4 weights
        w = np.array([1., 2., 5.]); wn = M.norm(w)
        if not (abs(wn.mean() - 1.) < 1e-12 and np.allclose(wn / w, wn[0] / w[0])):
            bad.append('weights: not one constant n / sum(w)')
        # ---- 5 fits
        Xs = rng.normal(size=(600, 107)).astype(np.float32); Xs[rng.random(Xs.shape) < .05] = np.nan; ys = 3 * Xs[:, T.FEATS.index('fair_edge')].astype(float) + rng.normal(size=600)
        ys[np.isnan(ys)] = 0.; ws = rng.uniform(.5, 2., 600)
        for cand in M.CANDS:
            try:                                                                            # a positive control: valid inputs must fit
                b1, b2 = M.fit_head(Xs, ys, ws, cand, rounds=5), M.fit_head(Xs, ys, ws, cand, rounds=5)
            except Exception as e:                                                          # noqa: BLE001
                bad.append(f'fit: {cand} fit_head failed on valid inputs ({e!r})'[:200]); continue
            if b1.model_to_string() != b2.model_to_string():
                bad.append(f'fit: {cand} is not deterministic')
            if b1.feature_name() != M.INPUTS[cand]:
                bad.append(f'fit: {cand} does not use exactly its declared inputs in order')
            cols = [T.FEATS.index(c_) for c_ in M.INPUTS[cand]]; lg_ = M.lgb()
            bm = lg_.train(dict(BX_PARAMS, objective='regression'), lg_.Dataset(np.ascontiguousarray(Xs[:, cols]), label=ys, weight=ws * (len(ws) / ws.sum()),
                                                                                feature_name=list(M.INPUTS[cand])), num_boost_round=5)
            tr = lambda b_: b_.model_to_string()[b_.model_to_string().index('Tree=0'):b_.model_to_string().index('end of trees')]
            if tr(b1) != tr(bm):
                bad.append(f'fit: {cand} is not the declared construction (pinned parameters, normalized weights, its columns)')
        Ef = {h: dict(X=Xs, y=ys, w=ws, meta=[]) for h in M.HEADS}
        try:
            man = M.fit_all(Ef, td / 'fit', rounds=5)
        except ValueError as e:
            man = {}; bad.append(f'fit_all refused ({e})')
        if len(man) != 12:
            bad.append(f'fit_all: {len(man)} heads, not 12')
        lg = M.lgb()
        for k_, v in man.items():
            cand, s, h = k_.split(); bb = lg.Booster(model_file=str(M.model_path(td / 'fit', cand, s, h)))
            if C.sha(M.model_path(td / 'fit', cand, s, h)) != v['sha256'] or bb.num_trees() != 5:
                bad.append(f'fit_all: {k_} model file / hash / trees')
            wn = M.norm(ws); ym = float(np.sum(wn * ys) / np.sum(wn)); p_ = M.predict(bb, cand, Xs)
            if not (abs(v['train_weighted_mean_y'] - ym) < 1e-9 and abs(v['insample_wmse_zero'] - np.sum(wn * ys ** 2) / np.sum(wn)) < 1e-9
                    and abs(v['insample_wmse'] - np.sum(wn * (p_ - ys) ** 2) / np.sum(wn)) < 1e-9):
                bad.append(f'fit_all: {k_} in-sample readings are not the weighted formulas')
        if not _refused(lambda: M.fit_all(Ef, td / 'fit', rounds=5), 'never overwritten'):
            bad.append('fit_all: an existing model was overwritten')
        # ---- 6 support
        if E is not None:
            S = M.support_of(E); sup = M.Support(S); e0 = E[(0, 'post')]; mt = e0['meta'][0]
            row = {c: float(e0['X'][0, i]) for i, c in enumerate(T.FEATS)}; row.update(price=mt['price'], generation=mt['generation'], decision_ms=mt['decision_ms'])
            S200 = json.loads(json.dumps(S)); S200['0 post']['originals'] = 250; sup2 = M.Support(S200)
            if sup2.check(0, 'post', row) is not None:
                bad.append(f'support: a training row is not supported ({sup2.check(0, "post", row)})')
            if sup.check(0, 'post', row) != 'insufficient_candidate_support':
                bad.append('support: fewer than 200 originals passed')
            cases = {'unsupported_horizon': dict(decision_ms=4990.), 'unsupported_state_category': dict(price=99.5),
                     'outside_training_range': dict(ctx_inventory_shares=1e6)}
            for want_st, ch in cases.items():
                if sup2.check(0, 'post', dict(row, **ch)) != want_st:
                    bad.append(f'support: {want_st} not reported ({sup2.check(0, "post", dict(row, **ch))})')
            if sup2.check(0, 'post', dict(row, t_to_end=float('nan'))) != 'outside_training_range':
                bad.append('support: a NaN numeric context passed')
        # ---- 7 the controller
        okrow = dict(row) if E is not None else None
        heads = {(0, 'post'): _Head(1.), (1, 'post'): _Head(-1.), (0, 'keep'): _Head(-1.), (1, 'keep'): _Head(1.)}
        OS = type('OS', (), {'check': lambda self, s, h, r: None})()
        c = M.Candidate(heads, OS, 'RICH'); c.inputs = lambda v, B, s, g: dict(okrow, decision_ms=g)
        got = {(s, st): c.decide(_view(st, s, gid=None if st == 'absent' else 0), _B())['action'] for s in (0, 1) for st in ('absent', 'live')}
        if got != {(0, 'absent'): 'POST', (1, 'absent'): 'WAIT', (0, 'live'): 'PULL', (1, 'live'): 'KEEP'}:
            bad.append(f'controller: the side / head mapping or signs are wrong ({got})')
        for val, wa, wl in ((0., 'WAIT', 'KEEP'), (2., 'POST', 'KEEP'), (-2., 'WAIT', 'PULL')):
            c.H = {h: _Head(val) for h in M.HEADS}
            if (c.decide(_view('absent'), _B())['action'], c.decide(_view('live', gid=0), _B())['action']) != (wa, wl):
                bad.append(f'controller: prediction {val} does not give {wa} / {wl}')
        c.H = {h: _Head(5.) for h in M.HEADS}
        d_ = c.decide(_view('absent', room=-1.), _B())
        if (d_['status'], d_['action']) != ('cap_blocked', 'WAIT'):
            bad.append('controller: a cap-blocked POST was not WAIT / cap_blocked')
        d_ = c.decide(_view('absent', t=4950.), _B())
        if (d_['status'], d_['action']) != ('near_end', 'WAIT'):
            bad.append('controller: a near-END POST was not WAIT / near_end')
        c.inputs = lambda v, B, s, g: None
        if [c.decide(_view(st, gid=0), _B())['action'] for st in ('absent', 'live')] != ['WAIT', 'KEEP'] or c.decide(_view('absent'), _B())['status'] != 'feature_unavailable':
            bad.append('controller: feature_unavailable did not fall back to WAIT / KEEP')
        c.inputs = lambda v, B, s, g: dict(okrow, decision_ms=g); c.csup = type('NS', (), {'check': lambda self, s, h, r: 'outside_training_range'})()
        if [(x_['status'], x_['action']) for x_ in (c.decide(_view('absent'), _B()), c.decide(_view('live', gid=0), _B()))] != [('outside_training_range', 'WAIT'), ('outside_training_range', 'KEEP')]:
            bad.append('controller: an unsupported state did not fall back with its own status')
        c.csup = OS; c.H = {h: _Head(float('nan')) for h in M.HEADS}
        if [(x_['status'], x_['action']) for x_ in (c.decide(_view('absent'), _B()), c.decide(_view('live', gid=0), _B()))] != [('nonfinite_prediction', 'WAIT'), ('nonfinite_prediction', 'KEEP')]:
            bad.append('controller: a nonfinite prediction did not fall back')
        # ---- 8 the policies
        Mst, sst = LT.stubs(-12.), LT.OpenSupport(); hh = {cnd: {h: _Head(1.) for h in M.HEADS} for cnd in M.CANDS}
        n0 = M.make_policy('N0', hh, OS, Mst, sst)
        if n0.decide(_view('absent'), _B())['action'] != 'WAIT' or not _refused(lambda: n0.decide(_view('live', gid=0), _B())):
            bad.append('policies: N0 posts or acts in a live state')
        pa = M.make_policy('ENTRY_PA', hh, OS, Mst, sst); pa.entry.inputs = lambda v, B, s, g: dict(okrow, decision_ms=g)
        try:                                                                                # valid states: neither decision may refuse
            pa_ok = pa.decide(_view('live', gid=0), _B())['action'] == 'PULL' and pa.decide(_view('absent'), _B())['action'] == 'POST'
        except Exception as e:                                                              # noqa: BLE001
            pa_ok = False; bad.append(f'policies: ENTRY_PA refused a valid state ({e!r})'[:200])
        if not pa_ok:
            bad.append('policies: ENTRY_PA does not POST by RICH and PULL at the review')
        eo, ro = M.make_policy('ENTRY_ONLY', hh, OS, Mst, sst), M.make_policy('RETAIN_ONLY', hh, OS, Mst, sst)
        if not (isinstance(eo.entry, M.Candidate) and eo.entry.cand == 'RICH' and eo.retain.name == 'PI0' and ro.entry.name == 'PI0'
                and isinstance(ro.retain, M.Candidate) and ro.retain.cand == 'RICH'):
            bad.append('policies: the hybrids do not dispatch to the declared sub-policies')
        for hy in (eo, ro):
            hy.entry.decide = lambda v, B, n_=hy.entry.name: dict(policy=n_, action='WAIT'); hy.retain.decide = lambda v, B, n_=hy.retain.name: dict(policy=n_, action='KEEP')
        if (eo.decide(_view('absent'), _B())['policy'], eo.decide(_view('live', gid=0), _B())['policy'], ro.decide(_view('absent'), _B())['policy'],
                ro.decide(_view('live', gid=0), _B())['policy']) != ('RICH', 'PI0', 'PI0', 'RICH'):
            bad.append('policies: a hybrid routes absent / live states to the wrong sub-policy')
        if not _refused(lambda: M.make_policy('NEU2', hh, OS, Mst, sst)):
            bad.append('policies: an undeclared policy was made')
        # ---- 9 end to end on the real engine
        wroot = Path(os.environ['PF_TEACHER_SYN']) if os.environ.get('PF_TEACHER_SYN') else None
        with tempfile.TemporaryDirectory(dir=str(I.K.IO)) as tmp:
            if wroot is None:
                wroot = TS.build_world(Path(tmp) / 'world')
            wj = json.loads((wroot / 'world.json').read_text()); sday = '20261024'; kp = Path(wj['kp']); src = KP.Syn(); ff = X.syn_features()
            W = L.worlds_of_day(sday, kp, 3); _, prs, fair, depth, ff = L.world_inputs(sday, W, src, X.syn_inputs, ff)
            worlds = [(w, L.world_bind(sday, w, prs, fair, depth, ff)[0]) for w in W]; worlds = [(w, wb) for w, wb in worlds if wb is not None]
            if not worlds:
                bad.append('end to end: no synthetic world')

            def episode(pol, wb, w):
                B = BE.Branch(wb['path'], 0, ident=R2.episode_ident(sday, w['o'][0]))
                if hasattr(pol, 'bind'):
                    pol.bind(B, wb)
                for k in range(L.NT):
                    BE.step(B, k, pol)
                return B, BE.finish(B)
            led = lambda Lx: (Lx['status'], [tuple(f) for f in Lx['fills']], Lx.get('generations'), round(Lx.get('value', 0.), 9), [tuple(d) for d in Lx.get('decisions', [])])
            for w, wb in worlds:
                for name in M.POLICIES:
                    pol = M.make_policy(name, hh, OS, Mst, sst)
                    try:                                                                    # every declared policy's episode must complete
                        B, Lx = episode(pol, wb, w)
                    except Exception as e:                                                  # noqa: BLE001
                        bad.append(f'end to end: the {name} episode stopped ({e!r})'[:200]); continue
                    if Lx['status'] != 'ok' or not L.reconcile_arm(Lx, B) or pol.bind_bad != 0 or pol.bind_checked == 0:
                        bad.append(f'end to end: {name} episode not reconciled / bound ({Lx["status"]}, bad {pol.bind_bad}, checked {pol.bind_checked})')
                    if name == 'N0' and (Lx.get('generations') != 0 or abs(Lx.get('value', 1.)) > 0):
                        bad.append('end to end: N0 entered or has value')
                try:                                                                        # the fixed-sign comparisons: declared policies must complete
                    neg = M.make_policy('RICH', {cnd: {h: _Head(-1. if h[1] == 'post' else 1.) for h in M.HEADS} for cnd in M.CANDS}, OS, Mst, sst)
                    _, Ln = episode(neg, wb, w); _, L0 = episode(M.make_policy('N0', hh, OS, Mst, sst), wb, w)
                    if led(Ln)[:4] != led(L0)[:4]:
                        bad.append('end to end: RICH with a negative POST head differs from N0')
                    pk = M.make_policy('RICH', {cnd: {h: _Head(1.) for h in M.HEADS} for cnd in M.CANDS}, OS, Mst, sst)
                    _, Lp = episode(pk, wb, w); _, Le = episode(BE.POST_KEEP, wb, w)
                    if led(Lp) != led(Le):
                        bad.append('end to end: RICH (POST > 0, KEEP > 0) differs from the engine\'s POST_KEEP')
                    pp = M.make_policy('RICH', {cnd: {h: _Head(1. if h[1] == 'post' else -1.) for h in M.HEADS} for cnd in M.CANDS}, OS, Mst, sst)
                    _, Lq = episode(pp, wb, w); _, La = episode(M.make_policy('ENTRY_PA', {cnd: {h: _Head(1.) for h in M.HEADS} for cnd in M.CANDS}, OS, Mst, sst), wb, w)
                    if led(Lq) != led(La) or not any(d[3] == 'PULL' for d in Lq.get('decisions', [])):
                        bad.append('end to end: RICH (POST > 0, KEEP < 0) differs from ENTRY_PA or never pulls')
                except Exception as e:                                                      # noqa: BLE001
                    bad.append(f'end to end: a fixed-sign comparison episode stopped ({e!r})'[:200])
    if bad:
        sys.exit(FAIL + ': ' + '; '.join(bad))
    print('adv2 selftest passed: inputs; reader (3 refusals); extraction (exact rows, statuses, crossing purge, 3 refusals, manifest); weights; '
          'fits (deterministic, declared inputs, reload bit-equal, no overwrite, weighted readings); support (5 statuses); controller (signs, '
          'ties, own heads, 5 fallbacks); policies (N0, ENTRY_PA, hybrids, undeclared); end to end on the engine (every policy reconciled and '
          'bound; N0 = 0; RICH = N0 / POST_KEEP / ENTRY_PA at fixed signs)')
