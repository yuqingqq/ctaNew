"""SELFTEST of pf_adv2_entry (ADV2-ENTRY-SMOOTH-v1: the declared parameter delta, the two POST fits, R3's training identity, the policies;
2026-11-21, session aea5bfb9). run(M) checks the module object M (the file, or a mutated copy under pf_mutant_one.py); any failed check exits
with FAIL. Synthetic only (no market data):
 1. the declared delta: check_params passes; another changed parameter, a different min_data_in_leaf and 200 rounds are each refused;
 2. fits: deterministic; the booster's inputs are RICH's 107 in order; a fit equals the explicit construction (the pinned pf_pipeline_bench
    dictionary, objective regression, min_data_in_leaf 1,000, n / sum(w) weights, the rounds given); with fewer than 2 x 1,000 rows no split
    is possible (every prediction equal: the smoothing parameter is in effect); fit_all saves two heads, reloads bit-equal, records hashes /
    weighted readings and refuses to overwrite a model;
 3. R3's training identity: r3_train returns the POST heads of a written extraction when its manifest is the committed one; a manifest that
    is not the committed one and a training file that differs from its manifest are refused;
 4. the policies: SMOOTH = the candidate's entry (its own heads, RICH's columns, the name SMOOTH) with PI0 retention; ENTRY_ONLY / PI0 / N0 are
    pf_adv2's; an undeclared name is refused; the entry controller POSTs iff its prediction > 0 (a tie WAITs) and asks PI0 in a live state;
 5. end to end on the real engine and the cached synthetic world: every declared policy's episode reconciles with zero binding failures; N0
    has no order; SMOOTH with a negative head equals N0 ledger for ledger.
PF_TEACHER_SYN: the cached synthetic world."""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_adv2_selftest as AT                             # noqa: E402  R3's helpers (stub heads, views)
import pf_branch_engine as BE                             # noqa: E402
import pf_pipeline_bench as BX                            # noqa: E402
import pf_r2_labels as L                                  # noqa: E402
import pf_r2_labels_selftest as LT                        # noqa: E402
import pf_r2_shadow as R2                                 # noqa: E402
import pf_r2_teacher as T                                 # noqa: E402
import pf_r2_teacher_selftest as TS                       # noqa: E402

C, pd, I, KP, X = R2.C, R2.pd, R2.I, R2.KP, R2.X
FAIL = 'ADV2 ENTRY SELFTEST FAILED'
BX_PARAMS = dict(BX.PARAMS)                                # the pinned dictionary, read independently of the module under test


def _refused(fn, msg=None):
    try:
        fn()
    except ValueError as e:
        return msg is None or msg in str(e)
    except Exception:                                                                       # noqa: BLE001  (a crash is not the check)
        return False
    return False


def run(M):  # noqa: C901
    bad = []; A = M.A; rng = np.random.default_rng(11)
    # ---- 1 the declared delta
    try:
        M.check_params()
    except ValueError as e:
        bad.append(f'delta: the declared parameters were refused ({e})')
    rest = lambda d: {k: v for k, v in d.items() if k != 'min_data_in_leaf'}
    if not (M.PARAMS.get('min_data_in_leaf') == 1000 and M.ROUNDS == 50 and rest(M.PARAMS) == rest(dict(BX_PARAMS, objective='regression'))):
        bad.append('delta: the parameters are not the pinned dictionary with min_data_in_leaf 1,000 and 50 rounds')
    saved = dict(M.PARAMS), M.ROUNDS
    for ch, rounds in ((dict(num_leaves=63), 50), (dict(min_data_in_leaf=500), 50), ({}, 200)):
        try:
            M.PARAMS.clear(); M.PARAMS.update(dict(saved[0], **ch)); M.ROUNDS = rounds
            if not _refused(M.check_params):
                bad.append(f'delta: {ch} / {rounds} rounds was not refused')
        finally:
            M.PARAMS.clear(); M.PARAMS.update(saved[0]); M.ROUNDS = saved[1]
    sa = dict(A.PARAMS)
    try:                                                                                    # the base changed with the candidate: only the hash sees it
        A.PARAMS['lambda_l2'] = 5.; M.PARAMS['lambda_l2'] = 5.
        if not _refused(M.check_params, 'parameters changed'):
            bad.append('delta: a base parameter changed together with the candidate was not refused (the frozen hash)')
    finally:
        A.PARAMS.clear(); A.PARAMS.update(sa); M.PARAMS.clear(); M.PARAMS.update(saved[0])
    if M.HEADS != ((0, 'post'), (1, 'post')) or M.POLICIES != ('N0', 'PI0', 'ENTRY_ONLY', 'SMOOTH') or M.NAME != 'SMOOTH':
        bad.append('constants: the heads / policies / name changed')
    # ---- 2 fits
    n = 3000; Xs = rng.normal(size=(n, 107)).astype(np.float32); Xs[rng.random(Xs.shape) < .05] = np.nan
    ys = 3 * np.nan_to_num(Xs[:, T.FEATS.index('fair_edge')].astype(float)) + rng.normal(size=n); ws = rng.uniform(.5, 2., n)
    try:                                                                                    # a positive control: valid inputs must fit
        b1, b2 = M.fit_head(Xs, ys, ws, rounds=5), M.fit_head(Xs, ys, ws, rounds=5)
        if b1.model_to_string() != b2.model_to_string() or b1.feature_name() != list(A.INPUTS['RICH']):
            bad.append('fit: not deterministic, or not RICH\'s 107 inputs in order')
        lg = A.lgb(); bm = lg.train(dict(BX_PARAMS, objective='regression', min_data_in_leaf=1000), lg.Dataset(np.ascontiguousarray(Xs), label=ys, weight=ws * (len(ws) / ws.sum()),
                                                                                                         feature_name=list(A.INPUTS['RICH'])), num_boost_round=5)
        tr = lambda b_: b_.model_to_string()[b_.model_to_string().index('Tree=0'):b_.model_to_string().index('end of trees')]
        if tr(b1) != tr(bm):
            bad.append('fit: not the declared construction (pinned parameters with min_data_in_leaf 1,000, normalized weights, RICH\'s columns)')
        bs = M.fit_head(Xs[:1500], ys[:1500], ws[:1500], rounds=5); ps = M.predict(bs, Xs[:50])
        if not np.all(ps == ps[0]):
            bad.append('fit: a split under 2 x 1,000 rows: min_data_in_leaf 1,000 is not in effect')
    except Exception as e:                                                                  # noqa: BLE001
        bad.append(f'fit: fit_head failed on valid inputs ({e!r})'[:200])
    with tempfile.TemporaryDirectory() as td:
        td = Path(td); Ef = {h: dict(X=Xs, y=ys, w=ws, meta=[]) for h in M.HEADS}
        try:
            man = M.fit_all(Ef, td / 'fit', rounds=5)
        except Exception as e:                                                              # noqa: BLE001
            man = {}; bad.append(f'fit_all failed ({e!r})'[:200])
        if sorted(man) != ['SMOOTH 0 post', 'SMOOTH 1 post']:
            bad.append(f'fit_all: heads {sorted(man)}')
        for k_, v in man.items():
            s = int(k_.split()[1]); p = M.model_path(td / 'fit', s); bb = A.lgb().Booster(model_file=str(p)); wn = A.norm(ws); p_ = M.predict(bb, Xs)
            if not (C.sha(p) == v['sha256'] and bb.num_trees() == 5 and abs(v['train_weighted_mean_y'] - float(np.sum(wn * ys) / np.sum(wn))) < 1e-9
                    and abs(v['insample_wmse'] - float(np.sum(wn * (p_ - ys) ** 2) / np.sum(wn))) < 1e-9):
                bad.append(f'fit_all: {k_} file / trees / readings')
        if not _refused(lambda: M.fit_all(Ef, td / 'fit', rounds=5), 'never overwritten'):
            bad.append('fit_all: an existing model was overwritten')
        # ---- 3 R3's training identity
        out = td / 'r3'; (out / 'train').mkdir(parents=True); E = {}
        for (s, h) in A.HEADS:
            k = 40 if h == 'post' else 20; x = rng.normal(size=(k, 107)).astype(np.float32)
            meta = [dict(day='20261024', m=i, t_ms=1000 * i, side=s, head=h, pick='mandatory', row_key=dict(i=i)) for i in range(k)]
            E[(s, h)] = dict(X=x, y=rng.normal(size=k), w=rng.uniform(.5, 2., k), meta=meta)
            np.savez(out / 'train' / f'{s}_{h}.npz', X=E[(s, h)]['X'], y=E[(s, h)]['y'], w=E[(s, h)]['w'])
            (out / 'train' / f'{s}_{h}.meta.json').write_text(json.dumps(meta, sort_keys=True) + '\n')
        st = {'20261024 post ok': 80}; (out / 'train' / 'extract.json').write_text(json.dumps(dict(A.manifest_of(E, st)), indent=1, sort_keys=True) + '\n')
        sc = M.committed_equal
        try:
            M.committed_equal = lambda rel, local: True
            got = M.r3_train(out, Path('x'))
            if sorted(got) != sorted(M.HEADS) or not all(np.array_equal(got[h]['y'], E[h]['y']) for h in M.HEADS):
                bad.append('r3_train: not the POST heads of the written extraction')
            yb = E[(1, 'post')]['y'].copy(); yb[0] += 1.; np.savez(out / 'train' / '1_post.npz', X=E[(1, 'post')]['X'], y=yb, w=E[(1, 'post')]['w'])
            if not _refused(lambda: M.r3_train(out, Path('x')), 'differ from their extraction manifest'):
                bad.append('r3_train: a training file that differs from its manifest was not refused')
            M.committed_equal = lambda rel, local: False
            if not _refused(lambda: M.r3_train(out, Path('x')), 'not the committed one'):
                bad.append('r3_train: an extraction manifest that is not the committed one was not refused')
        finally:
            M.committed_equal = sc
    # ---- 4 the policies
    Mst, sst = LT.stubs(-12.), LT.OpenSupport(); OS = type('OS', (), {'check': lambda self, s, h, r: None})()
    sm = {h: AT._Head(1.) for h in M.HEADS}; rich = {'RICH': {h: AT._Head(-1.) for h in A.HEADS}}
    p = M.make_policy('SMOOTH', sm, rich, OS, Mst, sst)
    if not (isinstance(p, A.Hybrid) and isinstance(p.entry, A.Candidate) and p.entry.name == 'SMOOTH' and p.entry.cand == 'RICH' and p.entry.H is sm
            and p.retain.name == 'PI0' and p.name == 'SMOOTH'):
        bad.append('policies: SMOOTH is not the candidate entry (its heads, RICH columns) with PI0 retention')
    eo = M.make_policy('ENTRY_ONLY', sm, rich, OS, Mst, sst)
    if not (isinstance(eo, A.Hybrid) and eo.entry.H is rich['RICH'] and eo.retain.name == 'PI0') or M.make_policy('N0', sm, rich, OS, Mst, sst).name != 'N0':
        bad.append("policies: ENTRY_ONLY / N0 are not pf_adv2's")
    if not _refused(lambda: M.make_policy('RICH', sm, rich, OS, Mst, sst)):
        bad.append('policies: an undeclared policy was made')
    okrow = {c: 0. for c in T.FEATS}; okrow.update(price=40., generation=1., ctx_generation_ordinal=1.)
    for val, want in ((2., 'POST'), (0., 'WAIT'), (-1., 'WAIT')):
        q = M.make_policy('SMOOTH', {h: AT._Head(val) for h in M.HEADS}, rich, OS, Mst, sst); q.entry.inputs = lambda v, B, s, g: dict(okrow, decision_ms=g)
        got = q.decide(AT._view('absent'), AT._B())
        if got['action'] != want or got.get('policy') != 'SMOOTH':
            bad.append(f'controller: SMOOTH entry {val} -> {got["action"]}, not {want}')
    q.retain.decide = lambda v, B: dict(policy='PI0', action='KEEP')
    if q.decide(AT._view('live', gid=0), AT._B()).get('policy') != 'PI0':
        bad.append('controller: a live SMOOTH state is not asked to PI0')
    # ---- 5 end to end on the real engine
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
            B = BE.Branch(wb['path'], 0, ident=R2.episode_ident(sday, w['o'][0])); pol.bind(B, wb)
            for k in range(L.NT):
                BE.step(B, k, pol)
            return B, BE.finish(B)
        led = lambda Lx: (Lx['status'], [tuple(f) for f in Lx['fills']], Lx.get('generations'), round(Lx.get('value', 0.), 9))
        alt = {h: type('H', (), {'predict': lambda self, X_, num_threads=1: np.asarray([(float(x[2]) % 4.) - 1.5 for x in X_], float)})() for h in M.HEADS}
        posted = False
        for w, wb in worlds:
            for name in M.POLICIES:
                pol = M.make_policy(name, alt, {'RICH': {h: alt.get(h, AT._Head(1.)) for h in A.HEADS}}, OS, Mst, sst)
                try:
                    B, Lx = episode(pol, wb, w)
                except Exception as e:                                                      # noqa: BLE001
                    bad.append(f'end to end: the {name} episode stopped ({e!r})'[:200]); continue
                if Lx['status'] != 'ok' or not L.reconcile_arm(Lx, B) or pol.bind_bad != 0 or pol.bind_checked == 0:
                    bad.append(f'end to end: {name} not reconciled / bound')
                if name == 'N0' and Lx.get('generations') != 0:
                    bad.append('end to end: N0 entered')
                posted |= name == 'SMOOTH' and Lx.get('generations', 0) > 0
            try:
                _, Ln = episode(M.make_policy('SMOOTH', {h: AT._Head(-1.) for h in M.HEADS}, rich, OS, Mst, sst), wb, w); _, L0 = episode(M.make_policy('N0', sm, rich, OS, Mst, sst), wb, w)
                if led(Ln) != led(L0):
                    bad.append('end to end: SMOOTH with a negative head differs from N0')
            except Exception as e:                                                          # noqa: BLE001
                bad.append(f'end to end: a comparison episode stopped ({e!r})'[:200])
        if not posted:
            bad.append('end to end: the synthetic run never covered a SMOOTH post')
    if bad:
        sys.exit(FAIL + ': ' + '; '.join(bad))
    print('adv2 entry selftest passed: the declared delta (3 refusals); fits (deterministic, RICH inputs, the explicit construction, no split '
          'under 2 x 1,000 rows, fit_all hashes / readings / no overwrite); R3 training identity (2 refusals); policies and the entry controller; '
          'end to end (every policy reconciled and bound, N0 no order, a negative SMOOTH = N0)')
