"""GAIN / LOSS DECOMPOSITION OF ACTION VALUE (2026-11-17, session aea5bfb9; USER: "Next priority: predict expected loss amount, especially
severe losses, and test its calibration before changing the controller ... Keep the current economic target, inputs, latency and controller
fixed; predict expected gains and expected losses separately, explicitly retaining severe-loss amounts; train on every eligible decision,
including no-fill outcomes; require better net-value calibration and actual value at matched participation - not merely better risk AUC or
fewer fills"). A prediction screen declared before any run; no controller change; consumed days; exploratory.

TARGET (unchanged): the other session's joint-moment labels (pf_joint_moment: same 1 s paired action value, clocks 58 / 37 / 100 / 50 ms,
M50, queue M) - the focal order's oriented resting value W = y_w (post: value of posting; cancel: value of keeping), and on the audited
861 actions the full action value = SOFT_JOINT's other-order and risk terms + orientation x W (exactly as its DIRECT_WEALTH screen).
INPUTS (unchanged): pf_joint_moment.FEATURES (71 causal market + 16 order / depth / portfolio inputs); MECHANICAL as the control set.
DECOMPOSITION: W = G - L1 - L2 with G = max(W, 0), L = max(-W, 0), L1 = min(L, 30 c), L2 = max(L - 30 c, 0): severe amounts kept whole in
L2, no winsorization. Each component is a hurdle: P(component > 0) (pinned LightGBM binary, 200 rounds, on EVERY eligible training decision,
no-fill included) x E[component | > 0] (pinned LightGBM regression on the decisions where it is > 0); W_hat = G_hat - L1_hat - L2_hat, held to
the same finite-payoff bounds as DIRECT_WEALTH. Training: Oct23-25, all 27,236 eligible joint-moment actions.
COMPARATORS: DIRECT = the other session's saved direct_w regression (reproduced exactly); MECH = the same decomposition on MECHANICAL inputs;
ZERO; on the 861 actions also SOFT_JOINT.
EVALUATION: (L) every market of Oct26 and Oct27 with the training selection rule (pf_joint_moment_data.select_training: 2 per market x side x
phase x kind by identity hash, statuses retained, labels by pf_joint_moment.fast_label with its sampled canonical parity) - focal W; cells =
day x kind; (S) the audited 861 actions - full action value, descriptive.
READINGS (set L; computed by report()):
 G1 NET-CALIBRATION: GL's RMSE(W) below DIRECT's in >= 3 of 4 cells AND GL's quintile calibration error (bins of predicted W; mean over bins
    of |mean predicted - mean realized|, weighted by count) below DIRECT's in >= 3 of 4 -> GL-CALIBRATES-BETTER, else NOT-SHOWN.
 G2 MATCHED-PARTICIPATION: rank each cell by predicted W, take the top 10% / 25% / 50%: GL's realized mean W >= DIRECT's in >= 3 of 4 cells at
    >= 2 of the 3 rates -> GL-SELECTS-BETTER, else NOT-SHOWN (a 200-draw random selection of the same count per cell is printed).
 PASS = G1 and G2 (the USER's acceptance); else NOT-SHOWN.
 G3 LOSS-GENERALIZES (diagnostic): in every cell predicted total loss (L1 + L2) within 0.75-1.33x realized AND predicted severe excess (L2)
    within 0.5-2x realized -> LOSS-GENERALIZES, else NOT-GENERALIZED (next: state representation and training coverage, USER).
 Printed: component calibration by predicted quintile, Spearman, AUC of P(L > 0) and P(L > 30 c), MECH beside GL, ZERO, by side.
CONTROLS (any failure -> no output): training = every eligible row of the joint-moment data (count = its fit receipt, statuses eligible,
 no-fill rows present); G - L1 - L2 = W on every row; DIRECT reproduces the other session's DIRECT_WEALTH predictions on the 861 actions;
 set L and the 861 audited actions carry identical labels (y_w, y_value) on every shared action id (the audited actions were chosen after
 the client-availability check, set L before it, as in the joint-moment training data; the overlap is counted); sampled canonical label
 parity on set L; feature contract
 (no label columns); saved heads reload to the same predictions. Planted errors (selftest): fills-only training, a truncated severe tail,
 a label column among the inputs - each refused.
LIMITS: consumed days; simulated labels (rule M, queue M, unknown depth 0, E2 / imputed timing); the neutral decision population, not the
 controller's own; 1 s value with zero continuation (unchanged target).
Usage (run_light, PF_SLOT=claude): pf_gain_loss.py selftest | label <day> | fit | evaluate | report; driver run_gain_loss.sh
"""
import argparse
from collections import Counter
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_joint_moment as J                               # noqa: E402
import pf_joint_moment_data as JD                         # noqa: E402
import pf_joint_moment_fit as JF                          # noqa: E402
import pf_flow_control_run as RUN                         # noqa: E402

F, C, pd, I, A, V, D = J.F, J.C, J.pd, J.I, J.A, J.V, J.D
OUT = Path(I.K.IO) / 'gain_loss_20261117'
EVAL_DAYS = (F.CAL, F.TEST)
SEVERE = 30.
SCALE = 30.
RATES = (0.10, 0.25, 0.50)
COMP = ('g', 'l1', 'l2')
MUT_FILLS_ONLY = MUT_TRUNCATE_SEVERE = MUT_LEAK = 0       # planted errors (selftest only)


def components(w):
    w = np.asarray(w, float); g = np.maximum(w, 0.); loss = np.maximum(-w, 0.)
    l1 = np.minimum(loss, SEVERE); l2 = np.maximum(loss - SEVERE, 0.)
    if MUT_TRUNCATE_SEVERE:
        l2 = np.zeros_like(l2)
    return g, l1, l2


def identity_ok(w):
    g, l1, l2 = components(w)
    return bool(np.allclose(g - l1 - l2, w, rtol=0, atol=1e-9))


def training_rows(frame, expected_n):
    """Every eligible decision, no-fill included; refuses an outcome-selected subset."""
    z = frame[frame.w.abs() > 1e-9] if MUT_FILLS_ONLY else frame
    C.require(len(z) == expected_n and (z.w.abs() <= 1e-9).any(), 'training population is not every eligible decision')
    return z


def calibration_error(pred, actual, bins=5):
    s = pd.DataFrame(dict(p=np.asarray(pred, float), a=np.asarray(actual, float)))
    s['b'] = pd.qcut(s.p.rank(method='first'), bins, labels=False)
    g = s.groupby('b').agg(p=('p', 'mean'), a=('a', 'mean'), n=('p', 'size'))
    return float((g.n * (g.p - g.a).abs()).sum() / g.n.sum())


def top_mean(pred, actual, rate, tie):
    s = pd.DataFrame(dict(p=np.asarray(pred, float), a=np.asarray(actual, float), t=tie)).sort_values(['p', 't'], ascending=[False, True])
    n = max(1, int(math.ceil(rate * len(s))))
    return float(s.a.iloc[:n].mean()), n


def selftest():
    bad = []
    w = np.array([0., 5., -10., -45., 120., -300.])
    g, l1, l2 = components(w)
    if not (np.array_equal(g, [0, 5, 0, 0, 120, 0]) and np.array_equal(l1, [0, 0, 10, 30, 0, 30]) and np.array_equal(l2, [0, 0, 0, 15, 0, 270])):
        bad.append('components')
    if not identity_ok(w):
        bad.append('identity')
    fr = pd.DataFrame(dict(w=[0., 0., 3., -40.]))
    try:
        training_rows(fr, 4)
    except ValueError:
        bad.append('valid training refused')
    if abs(calibration_error([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) - 0.) > 1e-12 or abs(calibration_error([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]) - 1.) > 1e-12:
        bad.append('calibration error')
    if top_mean([3, 1, 2, 0], [10, 0, 5, -5], 0.5, [0, 1, 2, 3]) != (7.5, 2):
        bad.append('matched participation')
    caught = {}
    globals()['MUT_FILLS_ONLY'] = 1
    try:
        training_rows(fr, 4); caught['MUT_FILLS_ONLY'] = False
    except ValueError:
        caught['MUT_FILLS_ONLY'] = True
    finally:
        globals()['MUT_FILLS_ONLY'] = 0
    globals()['MUT_TRUNCATE_SEVERE'] = 1
    try:
        caught['MUT_TRUNCATE_SEVERE'] = not identity_ok(w)
    finally:
        globals()['MUT_TRUNCATE_SEVERE'] = 0
    globals()['MUT_LEAK'] = 1
    try:
        JF.feature_contract(feature_list('full')); caught['MUT_LEAK'] = False
    except ValueError:
        caught['MUT_LEAK'] = True
    finally:
        globals()['MUT_LEAK'] = 0
    if bad or not all(caught.values()):
        sys.exit(f'GAIN-LOSS SELFTEST FAILED -> no output: {bad}, planted errors caught {caught}')
    print(f'gain-loss selftests passed; planted errors caught: {caught}', flush=True)
    return caught


def feature_list(variant):
    f = list(J.FEATURES if variant == 'full' else J.MECHANICAL)
    return f + ['y_w'] if MUT_LEAK else f


def label(day):
    """Set L: the training selection rule on every market of an evaluation day (pf_joint_moment_data.run's training branch)."""
    C.require(day in EVAL_DAYS, 'undeclared evaluation day'); J.verify(); selftest(); start = time.time()
    out = OUT / f'label_{day}'; C.require(not out.exists(), 'label output exists'); out.mkdir(parents=True)
    raw, reg = JD.metadata(day, True); markets = reg.mid.astype(int).tolist()
    fair, _ = I.fair_fix(day, {int(r.mid): float(r.ws) * 1000 for r in reg.itertuples()})
    clock = json.loads((Path(I.K.IO) / 'step1' / day / 'clock_rl_d2.json').read_text())
    depth, _ = I.X5.depth_arrays(I.X5.depth_hours(day), float(clock['bn_in']) + float(clock['d_dec']))
    output = []; mismatches = []; statuses = Counter(); checked = 0; positive = 0; seen = set()
    for begin in range(0, len(markets), 24):
        batch = markets[begin:begin + 24]; data = I.load_day(day, set(batch), raw['imputation_offset_ms'])
        for W in I.pairs(data):
            m = int(W[0]['m']); seen.add(m)
            reference, all_rows = JD.select_training(W, data, day); chosen = all_rows[all_rows.selected].copy()
            statuses.update({str(k): int(v) for k, v in all_rows.groupby(['kind', 'status']).size().items()})
            bytime = {float(t): g for t, g in chosen.groupby('decision_ms')}
            ks = [int(np.searchsorted(W[0]['tg'], t)) for t in bytime]
            sparse = J.DATA.feature_rows(W, fair[m], depth, {0: ks, 1: ks})
            check_ids = set(chosen.sort_values('hash').groupby(['kind', 'side'], sort=False).head(1).uid) if (m % 17 == 0 or m == markets[0]) else set()

            def observe(k, t, ST, kpos):
                nonlocal checked, positive
                if t not in bytime:
                    return
                state = V.P.snapshot(W, k, t, ST, kpos); j = int(W[0]['jk'][k]); mid = (W[0]['P5'][j, 0] + W[1]['P5'][j, 0]) / 2.
                for row in bytime[t].to_dict('records'):
                    idx = [i for i, x in enumerate(state['info']) if x['kind'] == row['kind'] and x['side'] == row['side'] and x['rung'] == row['rung']
                           and (row['kind'] == 'post' or abs(x['activation'] - (t - row['x_age_ms'])) < .001)]
                    base = {n_: row[n_] for n_ in ('uid', 'day', 'm', 'kind', 'side', 'rung', 'generation', 'phase', 'decision_ms', 'x_age_ms', 'hash')}
                    status = 'eligible'
                    if len(idx) != 1:
                        status = 'not_available_in_client_state'
                    else:
                        index = idx[0]; action = state['baseline'].copy(); action[index] = 1. if row['kind'] == 'post' else 2.
                        if row['kind'] == 'post' and not V.P.feasible(state, action):
                            status = 'inventory_reservation_limit'
                    if status != 'eligible':
                        output.append(dict(base, status=status)); continue
                    price = state['orders'][index, 2]; base.update(status=status, mid=mid, price=price, known_remaining=state['orders'][index, 3])
                    lab = J.fast_label(W, k, t, ST, kpos, state, index, row['kind'], day)
                    if 'status' in lab:
                        output.append(dict(base, **lab)); continue
                    if row['uid'] in check_ids:
                        chk = J.canonical_label(W, row, reference, mid, price, lab['y_remaining']); checked += 1; positive += int(chk['y_q'] > 1e-9)
                        diff = {key: float(lab[key] - chk[key]) for key in ('y_q', 'y_m', 'y_w', 'y_value', 'y_risk', 'y_other') if not np.isclose(lab[key], chk[key], rtol=0, atol=1e-7)}
                        if diff:
                            mismatches.append(dict(uid=row['uid'], difference=diff))
                    market, scale = sparse[int(row['side'])][k]; x = J.features(W, k, t, state, index, market, scale)
                    base.update(zip(J.FEATURES, map(float, x))); output.append(dict(base, **lab))
            replay = RUN.run_lane(W, cap=F.CAP, observer=observe)
            for key in ('fl', 'stat', 'orders', 'cancels'):
                np.testing.assert_equal(replay[key], reference[key])
            print(json.dumps(dict(stage='label', day=day, markets=len(seen), rows=len(output), seconds=round(time.time() - start), mismatches=len(mismatches))), flush=True)
            del reference, replay, sparse; gc.collect(); C.guard('gain-loss label')
        del data; gc.collect()
    C.require(seen == set(markets), 'missing markets')
    if mismatches or not checked:
        sys.exit(f'GAIN-LOSS CONTROL FAILED on {day}: canonical label parity ({checked} checked, {len(mismatches)} mismatches) -> no output')
    frame = pd.DataFrame(output); C.require(not frame.uid.duplicated().any(), 'duplicate action')
    frame.to_parquet(out / 'actions.parquet', index=False)
    C.dump(out / 'receipt.json', dict(asof=C.stamp(), day=day, markets=len(markets), statuses=frame.groupby(['kind', 'status']).size().to_dict(),
                                      candidate_statuses=dict(statuses), canonical_checked=checked, canonical_positive=positive,
                                      outputs={'actions.parquet': C.sha(out / 'actions.parquet')}, seconds=time.time() - start))


def fit():
    J.verify(); selftest(); start = time.time(); out = OUT / 'fit'; C.require(not out.exists(), 'fit exists'); out.mkdir(parents=True)
    fr = json.loads((J.OUT / 'fit' / 'receipt.json').read_text())
    train = pd.concat([JF.read(d)[0] for d in F.TRAIN], ignore_index=True); train['w'] = train.y_w.astype(float)
    train = training_rows(train, int(fr['n'])); C.require(identity_ok(train.w.values), 'decomposition identity on training rows')
    g, l1, l2 = components(train.w.values); comp = dict(g=g, l1=l1, l2=l2); arts = {}
    for variant in ('full', 'mechanical'):
        feats = feature_list(variant); JF.feature_contract(feats); X = train[feats].to_numpy(np.float32)
        for c in COMP:
            y = comp[c]; pos = y > 1e-9
            for head, Xh, yh, obj in ((f'{variant}_{c}_p', X, pos.astype(float), 'binary'), (f'{variant}_{c}_amt', X[pos], y[pos] / SCALE, 'regression')):
                path = out / f'{head}.txt'; J.FM.fit(Xh, yh, feats, path, obj)
                arts[head] = dict(path=str(path), sha256=C.sha(path), n=int(len(yh)), positives=int(pos.sum()), objective=obj, features=variant)
                print(json.dumps(dict(stage='fit', head=head, n=int(len(yh)), seconds=round(time.time() - start))), flush=True)
        del X; gc.collect(); C.guard('gain-loss fit')
    C.dump(out / 'receipt.json', dict(asof=C.stamp(), training_days=list(F.TRAIN), n=len(train), no_fill_rows=int((train.w.abs() <= 1e-9).sum()),
                                      gains=int((g > 1e-9).sum()), losses=int(((l1 + l2) > 1e-9).sum()), severe=int((l2 > 1e-9).sum()),
                                      joint_moment_fit_receipt_sha256=C.sha(J.OUT / 'fit' / 'receipt.json'), artifacts=arts, seconds=time.time() - start))


def predict(frame, variant, arts):
    import lightgbm as lgb
    feats = feature_list(variant); X = frame[feats].to_numpy(np.float32); out = {}
    for c in COMP:
        p = lgb.Booster(model_file=arts[f'{variant}_{c}_p']['path']).predict(X, num_threads=1)
        amt = np.maximum(lgb.Booster(model_file=arts[f'{variant}_{c}_amt']['path']).predict(X, num_threads=1), 0.) * SCALE
        if c == 'l1':
            amt = np.minimum(amt, SEVERE)
        out[c] = p * amt; out[c + '_p'] = p
    side = frame.side.to_numpy(); sign = 1 - 2 * side; price = frame.price.to_numpy(float); rem = frame.known_remaining.to_numpy(float)
    low = np.minimum(-sign * price, sign * (100. - price)) * rem; high = np.maximum(-sign * price, sign * (100. - price)) * rem
    out['w'] = np.clip(out['g'] - out['l1'] - out['l2'], low, high); out['low'] = low; out['high'] = high
    return out


def direct(frame, low, high):
    import lightgbm as lgb
    art = json.loads((J.OUT / 'fit' / 'receipt.json').read_text())['artifacts']['direct_w']
    C.require(C.sha(art['path']) == art['sha256'], 'direct_w changed')
    return np.clip(lgb.Booster(model_file=art['path']).predict(frame[art['features']].to_numpy(np.float32), num_threads=1) * art['scale'], low, high)


def evaluate():
    J.verify(); selftest(); start = time.time(); out = OUT / 'eval'; C.require(not out.exists(), 'evaluation exists'); out.mkdir(parents=True)
    arts = json.loads((OUT / 'fit' / 'receipt.json').read_text())['artifacts']
    for a_ in arts.values():
        C.require(C.sha(a_['path']) == a_['sha256'], 'gain-loss head changed')
    frames = []; overlap = {}; audited = {}
    for day in EVAL_DAYS:
        lr = json.loads((OUT / f'label_{day}' / 'receipt.json').read_text())
        C.require(C.sha(OUT / f'label_{day}' / 'actions.parquet') == lr['outputs']['actions.parquet'], 'set L changed')
        L_ = pd.read_parquet(OUT / f'label_{day}' / 'actions.parquet'); L_ = L_[L_.status == 'eligible'].copy()
        S_, _ = JF.read(day)                               # the audited actions, the other session's labels
        both = S_.set_index('uid').join(L_.set_index('uid')[['y_w', 'y_value']], rsuffix='_L', how='inner')
        C.require(len(both) > 0 and np.allclose(both.y_w, both.y_w_L, rtol=0, atol=1e-9) and np.allclose(both.y_value, both.y_value_L, rtol=0, atol=1e-9),
                  f'set L and the audited actions disagree on shared labels on {day}')
        overlap[day] = dict(audited=len(S_), shared=len(both))
        L_['audited'] = L_.uid.isin(set(S_.uid)); frames.append(L_); audited[day] = S_
    E = pd.concat(frames, ignore_index=True); E['w'] = E.y_w.astype(float)
    C.require(identity_ok(E.w.values), 'decomposition identity on evaluation rows')
    full = predict(E, 'full', arts); mech = predict(E, 'mechanical', arts)
    E['GL'] = full['w']; E['MECH'] = mech['w']; E['DIRECT'] = direct(E, full['low'], full['high']); E['ZERO'] = 0.
    for c in COMP:
        E['GL_' + c] = full[c]; E['MECH_' + c] = mech[c]; E['GL_' + c + '_p'] = full[c + '_p']
    g, l1, l2 = components(E.w.values); E['g'] = g; E['l1'] = l1; E['l2'] = l2
    # the audited 861: full action value with SOFT_JOINT's other-order and risk terms (the DIRECT_WEALTH screen's own construction)
    S = []
    for day in EVAL_DAYS:
        jv = pd.read_parquet(J.OUT / f'eval_{day}' / 'values.parquet'); jv = jv[jv.case.isin(['SOFT_JOINT', 'DIRECT_WEALTH'])].pivot_table(index='uid', columns='case', values='predicted')
        src = D.OUT / day; base = pd.read_parquet(src / 'values.parquet'); base = base[(base.case == 'SOFT_JOINT') & (base.channel == 'ordinary')].set_index('uid')
        e = audited[day].set_index('uid'); base = base.loc[e.index]
        C.require(np.allclose(e.y_value.values, base.actual.values, rtol=0, atol=1e-7), f'audited target identity on {day}')
        fs = predict(e, 'full', arts); ms = predict(e, 'mechanical', arts); ds = direct(e, fs['low'], fs['high'])
        orient = np.where(e.kind == 'post', 1., -1.); rest = base.predicted.to_numpy() - (base.pred_focal_edge + base.pred_focal_motion).to_numpy()
        z = pd.DataFrame(dict(day=day, kind=e.kind.values, side=e.side.values, actual=base.actual.to_numpy(), SOFT_JOINT=base.predicted.to_numpy(),
                              DIRECT=rest + orient * ds, GL=rest + orient * fs['w'], MECH=rest + orient * ms['w'], ZERO=0.), index=e.index)
        C.require(np.allclose(z.DIRECT.values, jv.loc[e.index, 'DIRECT_WEALTH'].values, rtol=0, atol=1e-7), f'DIRECT does not reproduce DIRECT_WEALTH on {day}')
        C.require(np.allclose(z.SOFT_JOINT.values, jv.loc[e.index, 'SOFT_JOINT'].values, rtol=0, atol=1e-9), f'SOFT_JOINT archive mismatch on {day}')
        S.append(z.reset_index())
    E.to_parquet(out / 'set_L.parquet', index=False); pd.concat(S, ignore_index=True).to_parquet(out / 'set_S.parquet', index=False)
    C.dump(out / 'receipt.json', dict(asof=C.stamp(), set_L_n=len(E), set_S_n=int(sum(len(s) for s in S)), direct_reproduced=True, overlap=overlap,
                                      outputs={p.name: C.sha(p) for p in out.glob('*.parquet')}, seconds=time.time() - start))


def report():
    selftest(); E = pd.read_parquet(OUT / 'eval' / 'set_L.parquet'); S = pd.read_parquet(OUT / 'eval' / 'set_S.parquet')
    E['day'] = E.day.astype(str); models = ('GL', 'DIRECT', 'MECH', 'ZERO')
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()); fr = json.loads((OUT / 'fit' / 'receipt.json').read_text())
    Lr = [f'GAIN / LOSS DECOMPOSITION OF ACTION VALUE (prediction screen; target, inputs, latency, controller unchanged); as of {asof}',
          f"training Oct23-25: {fr['n']:,} eligible decisions ({fr['no_fill_rows']:,} no-fill, {fr['gains']:,} gains, {fr['losses']:,} losses, {fr['severe']:,} beyond 30 c)",
          f'set L (every market, Oct26-27): {len(E):,} eligible actions ({int(E.audited.sum())} shared with the audited set, identical labels); set S (audited): {len(S)}', '',
          'PART 1 (set L, focal W c/action): n | realized mean | RMSE  GL / DIRECT / MECH / ZERO | quintile calibration error  GL / DIRECT / MECH | Spearman GL / DIRECT']
    cell = {}
    for (day, kind), g in E.groupby(['day', 'kind']):
        r = {m_: float(np.sqrt(np.mean((g[m_] - g.w) ** 2))) for m_ in models}
        ce = {m_: calibration_error(g[m_], g.w) for m_ in ('GL', 'DIRECT', 'MECH')}
        cell[(day, kind)] = dict(rmse=r, ce=ce, g=g)
        Lr.append(f'  {day} {kind:<6} n {len(g):5,} | {g.w.mean():+6.2f} | ' + ' / '.join(f'{r[m_]:6.2f}' for m_ in models) + ' | ' +
                  ' / '.join(f'{ce[m_]:5.2f}' for m_ in ('GL', 'DIRECT', 'MECH')) + f' | {g.GL.corr(g.w, method="spearman"):+.3f} / {g.DIRECT.corr(g.w, method="spearman"):+.3f}')
    g1r = sum(cell[k]['rmse']['GL'] < cell[k]['rmse']['DIRECT'] for k in cell); g1c = sum(cell[k]['ce']['GL'] < cell[k]['ce']['DIRECT'] for k in cell)
    g1 = g1r >= 3 and g1c >= 3
    Lr += ['', 'PART 2 (set L): matched participation - realized mean W of the top-ranked share of each cell: GL / DIRECT / MECH | random (200 draws, mean / p95)']
    rng = np.random.default_rng(20261117); wins = {}
    for rate in RATES:
        w_ = 0
        for k, d in cell.items():
            g = d['g']; tie = [J.V.PATH.keyhash(k[0], int(m_), float(t_)) for m_, t_ in zip(g.m, g.decision_ms)]
            vals = {m_: top_mean(g[m_], g.w, rate, tie) for m_ in ('GL', 'DIRECT', 'MECH')}; n = vals['GL'][1]
            null = [float(g.w.values[rng.permutation(len(g))[:n]].mean()) for _ in range(200)]
            w_ += int(vals['GL'][0] >= vals['DIRECT'][0])
            Lr.append(f'  top {rate:.0%} {k[0]} {k[1]:<6} n {n:4d}: ' + ' / '.join(f'{vals[m_][0]:+6.2f}' for m_ in ('GL', 'DIRECT', 'MECH')) +
                      f' | random {np.mean(null):+6.2f} / {np.quantile(null, .95):+6.2f}')
        wins[rate] = w_
    g2 = sum(v >= 3 for v in wins.values()) >= 2
    Lr += ['', 'PART 3 (set L): components - predicted / realized totals (c), AUC of P(> 0)']
    g3 = True
    for k, d in cell.items():
        g = d['g']; pl = (g.GL_l1 + g.GL_l2).sum(); al = (g.l1 + g.l2).sum(); p2 = g.GL_l2.sum(); a2 = g.l2.sum()
        ok = (0.75 <= pl / max(al, 1e-9) <= 1.33) and (0.5 <= p2 / max(a2, 1e-9) <= 2.)
        g3 &= ok
        aucs = {c: JF.binary_metrics((g[c] > 1e-9).astype(float), np.clip(g['GL_' + c + '_p'], 0, 1))['auc'] for c in COMP}
        Lr.append(f'  {k[0]} {k[1]:<6}: gain {g.GL_g.sum():9.0f} / {g.g.sum():9.0f} | loss {pl:9.0f} / {al:9.0f} (x{pl / max(al, 1e-9):.2f}) | severe excess {p2:9.0f} / {a2:9.0f} '
                  f'(x{p2 / max(a2, 1e-9):.2f}) | AUC gain {aucs["g"]:.3f} loss {aucs["l1"]:.3f} severe {aucs["l2"]:.3f} | MECH loss x{(g.MECH_l1 + g.MECH_l2).sum() / max(al, 1e-9):.2f}')
    Lr += ['', 'PART 4 (set L): by predicted-W quintile, pooled days, GL: n | predicted W / realized W | predicted loss / realized loss | predicted severe / realized severe (c/action)']
    for kind, g in E.groupby('kind'):
        b = pd.qcut(g.GL.rank(method='first'), 5, labels=False)
        for q in range(5):
            z = g[b == q]
            Lr.append(f'  {kind:<6} Q{q + 1} n {len(z):5,} | {z.GL.mean():+7.2f} / {z.w.mean():+7.2f} | {(z.GL_l1 + z.GL_l2).mean():6.2f} / {(z.l1 + z.l2).mean():6.2f} | '
                      f'{z.GL_l2.mean():6.2f} / {z.l2.mean():6.2f}')
    Lr += ['', 'PART 5 (set S, the audited 861, full action value c, descriptive): RMSE GL / DIRECT / MECH / SOFT_JOINT / ZERO | calibration error GL / DIRECT / SOFT']
    for (day, kind), g in S.groupby(['day', 'kind']):
        Lr.append(f'  {day} {kind:<6} n {len(g):3d} | ' + ' / '.join(f'{np.sqrt(np.mean((g[m_] - g.actual) ** 2)):6.2f}' for m_ in ('GL', 'DIRECT', 'MECH', 'SOFT_JOINT', 'ZERO')) +
                  ' | ' + ' / '.join(f'{calibration_error(g[m_], g.actual):5.2f}' for m_ in ('GL', 'DIRECT', 'SOFT_JOINT')))
    verdict = 'PASS' if (g1 and g2) else 'NOT-SHOWN'
    Lr += ['', f'G1 NET-CALIBRATION: GL RMSE below DIRECT in {g1r} / 4 cells, calibration error below in {g1c} / 4 -> {"GL-CALIBRATES-BETTER" if g1 else "NOT-SHOWN"}',
           f'G2 MATCHED-PARTICIPATION: GL >= DIRECT in ' + ', '.join(f'{wins[r_]} / 4 at {r_:.0%}' for r_ in RATES) + f' -> {"GL-SELECTS-BETTER" if g2 else "NOT-SHOWN"}',
           f'PASS (G1 and G2): {verdict}', f'G3 LOSS-GENERALIZES: {"LOSS-GENERALIZES" if g3 else "NOT-GENERALIZED"}', '',
           'LIMITS: consumed days; simulated labels (rule M, queue M, unknown depth 0, E2 / imputed timing); the neutral decision population; 1 s target, zero continuation']
    (OUT / 'gain_loss.txt').write_text('\n'.join(Lr) + '\n')
    C.dump(OUT / 'gain_loss.json', dict(asof=asof, readings=dict(G1='GL-CALIBRATES-BETTER' if g1 else 'NOT-SHOWN', G2='GL-SELECTS-BETTER' if g2 else 'NOT-SHOWN',
                                                                  PASS=verdict, G3='LOSS-GENERALIZES' if g3 else 'NOT-GENERALIZED'),
                                         g1=dict(rmse_wins=g1r, ce_wins=g1c), g2=wins))
    print('\n'.join(Lr), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['selftest', 'label', 'fit', 'evaluate', 'report']); ap.add_argument('day', nargs='?')
    a_ = ap.parse_args()
    if a_.mode == 'selftest':
        selftest()
    elif a_.mode == 'label':
        label(a_.day)
    elif a_.mode == 'fit':
        fit()
    elif a_.mode == 'evaluate':
        evaluate()
    else:
        report()
