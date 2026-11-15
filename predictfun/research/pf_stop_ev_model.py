"""STOP-EV STAGE 2 - MODEL (2026-11-15, session aea5bfb9; USER: "Yes model now", on the recommendations after stage 1: train on the labels
that allow E2 point times; an untimed leg refuses only the actions it can reach). Design: handoffs/20261115_stop_ev_design.md section 8
(stage 2 as revised by the Codex accounting contract). The pipeline (keep rule in the lane, margins, portfolio) is stage 3, not here.

DATA: stage 1's actions_<day>_all_anchors_sensitivity (pf_stop_ev.py; contract stop-ev-chronological-order-value-v1), statuses eligible /
  eligible_estimated_timing: one outcome-independent probe per neutral order generation (no grid copies, weight 1); every other status
  counted per day.
INPUTS, all known at the decision t: v6's 71 market inputs at the action's own decision step (the STOP-MODEL row (m, side, k), built by
  pf_stop_model.day_matrix with v6's W30 scale) + ORDER inputs from the contract's client-known state (known_state: received frames and
  acknowledged fills only): rung, depth, price, age (capped 60 s), known remainder / submitted size, known queue ahead / shown size /
  initial queue relative to the W30 liquidity scale, queue ahead / shown size (NaN when nothing is shown or unknown), unknown queue at
  activation, edge = sg (pf_mid - price), effective horizon / 1 s. Never an outcome column (asserted).
TARGET: value_cents - the focal order's own tranches in [t + 22 ms, min(t + 1 s, end)] in the corrected neutral ladder, each at fill time
  + 1 s (cents per order; 30 sh).
MODEL (one candidate; declared by stage 1's loss rule HURDLE + ROBUST): P(fill) = LightGBM binary on y = [filled quantity > 0] (all
  training actions) x E[value | fill] = LightGBM Huber regression on the filled training actions, delta = their median |value|; the
  prediction = P x E. v6's pinned parameters (PD.LGB_PARAMS + lambda_l2 10), 200 rounds, no early stopping; chronological folds as v6
  (test <- training days before its calibration day). Calibration-day predictions are saved for stage 3's margin, unused here.
BASELINES (training-only comparators, not candidates): B1 = the training days' mean value by (rung, side); B2 = the same hurdle model on
  the ORDER inputs only (price / edge, level, queue, size, age, side: the mechanical explanation the market inputs must beat, rule 9).
READINGS DECLARED BEFORE THE FIRST FIT (test days = the 5 fold test days; day rule >= 4 of 5; eligible actions; c per order):
 R900 RANKS: the realized mean value rises from the bottom to the top quintile of the prediction (pooled: top > bottom) and top - bottom >
      0 on >= 4 days -> RANKS, else NO-RANK.
 R901 SKILL: Spearman IC of prediction vs value > 0 pooled and on >= 4 days -> IC+; the model's IC above B2's on >= 4 days ->
      BEATS-B2-IC; MSE below B1's on >= 4 days -> BEATS-B1-MSE (heavy tails: printed, the IC decides).
 R902 SIDES: on each side separately IC > 0 and top - bottom quintile > 0 (pooled) -> BOTH-SIDES, else the failing side is named.
 R903 LOCAL DECISION (keep iff prediction > 0; keep on ties, no margin): local gain vs keep-all = - sum of the pulled actions' values >
      0 pooled and on >= 4 days, AND the matched null (pf_stop_ev_accounting.action_metrics: the same kept count within day / side / hour
      / rung / known remainder, 500 draws) p <= 0.05 pooled -> LOCAL-GAIN, else NOT-SHOWN. Pull share printed.
 R904 VS B2: the model's local gain above B2's (same rule) on >= 4 days -> BEATS-B2-DECISION, else NOT-SHOWN.
 VERDICT: MODEL-READY iff R900 RANKS, R901 IC+ and BEATS-B2-IC, R902 BOTH-SIDES and R903 LOCAL-GAIN; else NOT-READY naming the parts.
 Printed, not decided on: calibration by decile, MSE, per-rung IC, the strict-WS subset of the test days, top inputs.
TRAINING CHECKS (any failure -> the report refuses): T1 chronology (asserted); E10 every eligible action joins exactly one STOP-MODEL
  row at the same t and its depth = sg (best - price) (mismatch 0; control: the join one step later must break it); T5 permuted-label
  falsifier: the training labels permuted jointly over the training set, 10 draws: test |IC| mean <= 0.01 and max <= 0.03; T6 IC train
  / test; T7 top inputs by gain.
CONTROLS (synthetic, any failure -> no output): the order-input derivation (units, NaN for an unknown / empty queue, the ask edge sign);
  the join and its depth identity; the hurdle on a known mixture; the B1 table; the matched-null adapter; planted errors each caught: the
  join one step late, an outcome column among the inputs, the value head fitted on every action (not the filled ones), the permutation
  falsifier without the permutation.
LIMITS: consumed days (exploratory, USER); replay rule M (B1a); E2 point estimates (+-250 ms, C12); a local 1 s value under a neutral
  continuation, not portfolio P&L (stage 3); the untimed rule's limit (stage 1 docstring).
Usage (run_light, PF_SLOT=claude): pf_stop_ev_model.py selftest | prep <day> | fit <test day> | report; driver run_stop_ev_model.sh.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_stop_ev as E                                    # noqa: E402  (stage 1: actions, the accounting contract)

M, K, AC = E.M, E.K, E.AC
PD = M.PD
OUTD = f'{K.IO}/stop_ev_model_20261115'
TIMING = 'all_anchors_sensitivity'
ELIG = ('eligible', 'eligible_estimated_timing')
FOLDS = M.FOLDS; TESTS = M.TESTS; DAYS9 = E.DAYS9
F71 = json.load(open(f'{M.OUTD}/fit_{TESTS[0]}_v6.json'))['features']
ORDER_IN = ['o_rung', 'o_depth', 'o_price', 'o_age', 'o_rem', 'o_q_rel', 'o_shown_rel', 'o_q0_rel', 'o_q_frac', 'o_qunk', 'o_edge', 'o_horizon']
B2_IN = ORDER_IN + ['fside']
FULL_IN = list(F71) + ORDER_IN
OUTCOME = ('value_cents', 'value_per_known_remaining_share', 'filled_quantity', 'past_quantity', 'common_prefix_quantity',
           'actual_remaining_after_prefix', 'at_price_value_cents', 'through_price_value_cents', 'n_tranches', 'status')
PARAMS = {**PD.LGB_PARAMS, 'lambda_l2': 10.0}; ROUNDS = 200
T5_DRAWS = 10; T5_MEAN = 0.01; T5_MAX = 0.03; NULL_DRAWS = 500; P_MAX = 0.05
MEM_CAP_MIB = 2048.0; NAN = float('nan')
MUT_JOIN_LATE = MUT_OUTCOME_IN = MUT_HURDLE_ALL = MUT_NO_PERM = 0   # planted errors (selftest only)
MUTS = ('MUT_JOIN_LATE', 'MUT_OUTCOME_IN', 'MUT_HURDLE_ALL', 'MUT_NO_PERM')


# ---------------------------------------------------------------- inputs
def order_inputs(A, scale, pf_mid):
    """The ORDER inputs from the contract's client-known columns (x_*), the W30 scale and pf_mid at the decision step."""
    sg = np.where(A.x_side.values == 0, 1.0, -1.0); sc = np.where(np.asarray(scale, float) > 0, np.asarray(scale, float), np.nan)
    qa = A.x_queue_ahead_known.values.astype(float); sh = A.x_shown_known.values.astype(float); q0 = A.x_queue_initial_known.values.astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = dict(o_rung=A.x_rung.values.astype(float), o_depth=A.x_depth.values.astype(float), o_price=A.x_price.values.astype(float),
                   o_age=np.minimum(A.x_age_ms.values.astype(float), 60000.0),
                   o_rem=A.x_known_remaining.values.astype(float) / A.x_submitted_quantity.values.astype(float),
                   o_q_rel=qa / sc, o_shown_rel=sh / sc, o_q0_rel=q0 / sc,
                   o_q_frac=np.where(np.isfinite(qa) & np.isfinite(sh) & (sh > 0), qa / np.where(sh > 0, sh, 1.0), np.nan),
                   o_qunk=A.x_queue_initial_unknown.values.astype(float), o_edge=sg * (np.asarray(pf_mid, float) - A.x_price.values.astype(float)),
                   o_horizon=A.x_effective_horizon_ms.values.astype(float) / 1000.0)
    for c in ('o_q_rel', 'o_shown_rel', 'o_q0_rel'):
        out[c] = np.where(np.isfinite(out[c]), out[c], np.nan)
    return out


def join_rows(A, R):
    """Row positions in the STOP-MODEL rows (R: m, side, k, t, best in file order) of the actions (m, side, k); E10 statistics."""
    pos = pd.DataFrame({'m': R.m.values, 'side': R.side.values, 'k': R.k.values + (1 if MUT_JOIN_LATE else 0), 'pos': np.arange(len(R))})
    J = A[['m', 'side', 'k']].reset_index(drop=True).merge(pos, on=['m', 'side', 'k'], how='left', validate='many_to_one')
    p = J.pos.values; ok = np.isfinite(p.astype(float)); pi = np.where(ok, p, 0).astype('int64')
    t_ok = ok & (R.t.values[pi] == A.decision_ms.values)
    sg = np.where(A.x_side.values == 0, 1.0, -1.0)
    d_ok = t_ok & (A.x_depth.values.astype(float) == sg * (R.best.values[pi].astype(float) - A.x_price.values.astype(float)))
    return pi, dict(actions=int(len(A)), joined=int(ok.sum()), t_mismatch=int((ok & ~t_ok).sum()), depth_mismatch=int((t_ok & ~d_ok).sum())), ok & d_ok


def prep_day(day):
    """One day's model data: eligible actions with the 71 market + ORDER inputs; E10 (join identity) and its control."""
    all_selftests()
    for t in TESTS:                                       # one input list for every fold (v6's decisions were taken on the dev days)
        if json.load(open(f'{M.OUTD}/fit_{t}_v6.json'))['features'] != F71:
            sys.exit(f'REFUSED: v6 fold {t} has a different input list')
    A = pd.read_parquet(f'{E.OUTD}/actions_{day}{E.ZTAG}_{TIMING}.parquet')
    st = dict(day=day, probes=int(len(A)), status={k: int(v) for k, v in A.status.value_counts().items()})
    A = A[A.status.isin(ELIG).values].reset_index(drop=True)
    R = pd.read_parquet(f'{M.OUTD}/rows_{day}.parquet', columns=['m', 'side', 'k', 't', 'best', 'scale_w30'])
    pi, e10, good = join_rows(A, R)
    global MUT_JOIN_LATE
    keep_ = MUT_JOIN_LATE; MUT_JOIN_LATE = 1
    _, ctl, _ = join_rows(A, R)
    MUT_JOIN_LATE = keep_
    e10['control_breaks'] = bool(ctl['depth_mismatch'] + ctl['t_mismatch'] + (ctl['actions'] - ctl['joined']) > 0)
    e10['pass'] = bool(e10['joined'] == e10['actions'] and e10['t_mismatch'] == 0 and e10['depth_mismatch'] == 0 and e10['control_breaks'])
    st['E10'] = e10
    if not e10['pass']:
        sys.exit(f'REFUSED (no output for {day}): E10 {e10}')
    X = M.day_matrix(day, 'W30', F71, sel=pi)
    O = order_inputs(A, R.scale_w30.values[pi], X[:, F71.index('pf_mid')])
    D = pd.DataFrame(X, columns=F71)
    for c in ORDER_IN:
        D[c] = O[c].astype(np.float32)
    keys = A[['day', 'm', 'side', 'rung', 'generation', 'decision_ms', 'k', 'hour', 'status', 'x_known_remaining', 'value_cents', 'filled_quantity',
              'at_price_value_cents', 'through_price_value_cents']].reset_index(drop=True)
    D = pd.concat([keys, D], axis=1)
    os.makedirs(OUTD, exist_ok=True)
    D.to_parquet(f'{OUTD}/ev_x_{day}.parquet', index=False)
    st['rows'] = int(len(D)); st['missing_share'] = {c: float(np.isnan(D[c].values.astype(float)).mean()) for c in ORDER_IN}
    json.dump(st, open(f'{OUTD}/ev_x_{day}.json', 'w'), indent=1, default=str)
    print(f'{day}: {len(D)} eligible actions of {st["probes"]} probes; E10 {e10}', flush=True)


# ---------------------------------------------------------------- model
def inputs_of(cols):
    bad = [c for c in cols if c in OUTCOME]
    if bad:
        raise ValueError(f'outcome columns among the inputs: {bad}')
    return cols


def fit_hurdle(X, y, fill, F, rounds=ROUNDS):
    """P(fill) on every action x E[value | fill] (Huber, delta = median |value| of the filled ones) on the filled actions."""
    import lightgbm as lgb
    fill = np.asarray(fill, bool); y = np.asarray(y, float)
    bp = lgb.train(PARAMS, lgb.Dataset(X, fill.astype(float), feature_name=F), rounds)
    sel = np.ones(len(y), bool) if MUT_HURDLE_ALL else fill
    delta = float(np.median(np.abs(y[sel]))) if sel.any() else 1.0
    delta = delta if delta > 0 else 1.0
    pv = {**PARAMS, 'objective': 'huber', 'alpha': delta}
    bv = lgb.train(pv, lgb.Dataset(X[sel], y[sel], feature_name=F), rounds)
    return dict(p=bp, v=bv, delta=delta)


def predict_hurdle(H, X):
    return H['p'].predict(X) * H['v'].predict(X), H['p'].predict(X), H['v'].predict(X)


def spearman(a, b):
    a = pd.Series(np.asarray(a, float)); b = pd.Series(np.asarray(b, float)); ok = a.notna() & b.notna()
    return float(a[ok].rank().corr(b[ok].rank())) if ok.sum() > 2 else NAN


def b1_table(D):
    return D.groupby(['rung', 'side']).value_cents.mean().to_dict()


def b1_predict(tab, D, default):
    return np.asarray([tab.get((int(r), int(s)), default) for r, s in zip(D.rung.values, D.side.values)], float)


def load_days(days):
    return pd.concat([pd.read_parquet(f'{OUTD}/ev_x_{d}.parquet') for d in days], ignore_index=True)


def permute_labels(y, fill, rng):
    if MUT_NO_PERM:
        return y, fill
    ix = rng.permutation(len(y)); return np.asarray(y)[ix], np.asarray(fill)[ix]


def fit_fold(test):
    all_selftests()
    if test not in FOLDS:
        sys.exit(f'REFUSED: {test} is not a fold test day')
    tr, cal = FOLDS[test]
    assert max(tr) < cal < test, 'T1 chronology'                     # T1
    T0 = time.time(); Dtr = load_days(list(tr)); F = inputs_of(list(FULL_IN)); FB = inputs_of(list(B2_IN))
    Xtr = Dtr[F].to_numpy(np.float32); ytr = Dtr.value_cents.values.astype(float); ftr = Dtr.filled_quantity.values > 0
    H = fit_hurdle(Xtr, ytr, ftr, F); HB = fit_hurdle(Dtr[FB].to_numpy(np.float32), ytr, ftr, FB)
    tab = b1_table(Dtr); dflt = float(ytr.mean())
    info = dict(test=test, cal=cal, train_days=list(tr), n_train=int(len(Dtr)), filled_train=int(ftr.sum()), delta=H['delta'], delta_b2=HB['delta'],
                params=PARAMS, rounds=ROUNDS, features=F, b2_features=FB, b1=[[int(k[0]), int(k[1]), float(v)] for k, v in tab.items()])
    info['T6_ic_train'] = spearman(predict_hurdle(H, Xtr)[0], ytr)
    imp = pd.Series(H['v'].feature_importance('gain'), index=F).sort_values(ascending=False)
    imp_p = pd.Series(H['p'].feature_importance('gain'), index=F).sort_values(ascending=False)
    info['T7_top_value'] = [[c, float(v)] for c, v in imp.head(12).items()]; info['T7_top_fill'] = [[c, float(v)] for c, v in imp_p.head(12).items()]
    Dte = load_days([test]); Xte = Dte[F].to_numpy(np.float32); yte = Dte.value_cents.values.astype(float)
    rng = np.random.default_rng(int(test)); t5 = []
    for _ in range(T5_DRAWS):                                         # T5: permuted training labels -> no test skill
        yp, fp = permute_labels(ytr, ftr, rng)
        t5.append(spearman(predict_hurdle(fit_hurdle(Xtr, yp, fp, F), Xte)[0], yte))
    info['T5_ic'] = t5; info['T5_pass'] = bool(np.mean(np.abs(t5)) <= T5_MEAN and np.max(np.abs(t5)) <= T5_MAX)
    del Dtr, Xtr
    for d in (cal, test):
        Dd = Dte if d == test else load_days([d]); Xd = Dd[F].to_numpy(np.float32)
        pr, pp, pv = predict_hurdle(H, Xd); pb = predict_hurdle(HB, Dd[FB].to_numpy(np.float32))[0]
        P_ = Dd[['day', 'm', 'side', 'rung', 'generation', 'decision_ms', 'hour', 'status', 'x_known_remaining', 'value_cents', 'filled_quantity']].copy()
        P_['pred'] = pr; P_['p_fill'] = pp; P_['v_fill'] = pv; P_['b2'] = pb; P_['b1'] = b1_predict(tab, Dd, dflt)
        P_.to_parquet(f'{OUTD}/pred_{test}_{d}.parquet', index=False)
    info['seconds'] = round(time.time() - T0, 1)
    json.dump(info, open(f'{OUTD}/fit_{test}.json', 'w'), indent=1, default=float)
    print(f'fold {test}: train {info["n_train"]} ({info["filled_train"]} filled), delta {H["delta"]:.2f} c, IC train {info["T6_ic_train"]:+.3f}, '
          f'T5 {"PASS" if info["T5_pass"] else "FAIL"} (mean |IC| {np.mean(np.abs(t5)):.4f}, max {np.max(np.abs(t5)):.4f}), {info["seconds"]} s', flush=True)


# ---------------------------------------------------------------- report
def metric_rows(P, keep):
    """pf_stop_ev_accounting.action_metrics adapter: the flattened eligible actions as the contract's row dicts."""
    ref = dict(contract=AC.REFERENCE); spec = dict(timing=TIMING)
    rows = [dict(action_id=(str(int(d)), int(m), int(s), int(r), int(g)), timing=TIMING, reference=ref, label_spec=spec, status=st,
                 label=dict(value_cents=float(v)), hour=int(h), features=dict(known_remaining=float(kr)))
            for d, m, s, r, g, st, v, h, kr in zip(P.day.values, P.m.values, P.side.values, P.rung.values, P.generation.values, P.status.values,
                                                    P.value_cents.values, P.hour.values, P.x_known_remaining.values)]
    return AC.action_metrics(rows, [bool(x) for x in keep], asof=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), draws=NULL_DRAWS)


def quint(pred, y):
    q = pd.qcut(pd.Series(pred).rank(method='first'), 5, labels=False)
    return [float(np.mean(np.asarray(y)[q.values == i])) for i in range(5)]


def report():
    fits = {t: json.load(open(f'{OUTD}/fit_{t}.json')) for t in TESTS if os.path.exists(f'{OUTD}/fit_{t}.json')}
    if len(fits) < len(TESTS):
        sys.exit(f'REFUSED: folds missing {[t for t in TESTS if t not in fits]}')
    if not all(f['T5_pass'] for f in fits.values()):
        sys.exit(f'REFUSED: T5 failed on {[t for t, f in fits.items() if not f["T5_pass"]]}')
    P = pd.concat([pd.read_parquet(f'{OUTD}/pred_{t}_{t}.parquet') for t in TESTS], ignore_index=True)
    xs = {d: json.load(open(f'{OUTD}/ev_x_{d}.json')) for d in DAYS9}
    T = [f'STOP-EV STAGE 2 - MODEL (hurdle P(fill) x E[value | fill], Huber; {TIMING}; untimed rule {E.UNTIMED_RULE}); 5 chronological test days; '
         f'as of {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}', '']
    T.append('DATA per day: eligible actions / probes (statuses) | E10 join')
    for d in DAYS9:
        x = xs[d]; T.append(f"  {d}: {x['rows']} / {x['probes']} ({', '.join(f'{k} {v}' for k, v in sorted(x['status'].items(), key=lambda z: -z[1]))}) | "
                            f"E10 {'PASS' if x['E10']['pass'] else 'FAIL'}")
    T.append('TRAINING per fold: n train (filled) | Huber delta | IC train | T5 permuted IC mean |.| / max |.| | top value / fill inputs')
    for t, f in fits.items():
        T.append(f"  {t}: {f['n_train']} ({f['filled_train']}) | {f['delta']:.2f} c | {f['T6_ic_train']:+.3f} | {np.mean(np.abs(f['T5_ic'])):.4f} / "
                 f"{np.max(np.abs(f['T5_ic'])):.4f} | {', '.join(c for c, _ in f['T7_top_value'][:5])} / {', '.join(c for c, _ in f['T7_top_fill'][:5])}")
    days = [int(t) for t in TESTS]; R = {}
    per = {}
    for d in days:
        x = P[P.day.values == d]; y = x.value_cents.values.astype(float)
        q = quint(x.pred.values, y); per[d] = dict(n=len(x), ic=spearman(x.pred.values, y), ic_b2=spearman(x.b2.values, y), ic_b1=spearman(x.b1.values, y),
                                                   mse=float(np.mean((x.pred.values - y) ** 2)), mse_b1=float(np.mean((x.b1.values - y) ** 2)),
                                                   mse_b2=float(np.mean((x.b2.values - y) ** 2)), q=q, tb=q[-1] - q[0],
                                                   gain=float(-y[~(x.pred.values > 0)].sum()), gain_b2=float(-y[~(x.b2.values > 0)].sum()),
                                                   pull=float(np.mean(~(x.pred.values > 0))), pull_b2=float(np.mean(~(x.b2.values > 0))), mean=float(y.mean()))
    y = P.value_cents.values.astype(float); qp = quint(P.pred.values, y)
    pool = dict(n=len(P), ic=spearman(P.pred.values, y), ic_b2=spearman(P.b2.values, y), ic_b1=spearman(P.b1.values, y), q=qp, tb=qp[-1] - qp[0],
                gain=float(-y[~(P.pred.values > 0)].sum()), gain_b2=float(-y[~(P.b2.values > 0)].sum()), pull=float(np.mean(~(P.pred.values > 0))))
    nd = lambda f: sum(int(f(per[d])) for d in days)
    r900 = 'RANKS' if (pool['tb'] > 0 and nd(lambda z: z['tb'] > 0) >= 4) else 'NO-RANK'
    ic_pos = pool['ic'] > 0 and nd(lambda z: z['ic'] > 0) >= 4
    beats_b2 = nd(lambda z: z['ic'] > z['ic_b2']) >= 4; beats_b1 = nd(lambda z: z['mse'] < z['mse_b1']) >= 4
    sides = {}
    for s in (0, 1):
        x = P[P.side.values == s]; yy = x.value_cents.values.astype(float); qq = quint(x.pred.values, yy)
        sides[s] = dict(n=len(x), ic=spearman(x.pred.values, yy), tb=qq[-1] - qq[0])
    r902 = 'BOTH-SIDES' if all(sides[s]['ic'] > 0 and sides[s]['tb'] > 0 for s in (0, 1)) else 'FAILS-' + '/'.join(
        ('BID', 'ASK')[s] for s in (0, 1) if not (sides[s]['ic'] > 0 and sides[s]['tb'] > 0))
    mm = metric_rows(P, P.pred.values > 0); mb = metric_rows(P, P.b2.values > 0)
    r903 = 'LOCAL-GAIN' if (pool['gain'] > 0 and nd(lambda z: z['gain'] > 0) >= 4 and mm['p_value'] is not None and mm['p_value'] <= P_MAX) else 'NOT-SHOWN'
    r904 = 'BEATS-B2-DECISION' if nd(lambda z: z['gain'] > z['gain_b2']) >= 4 else 'NOT-SHOWN'
    T += ['', f"TEST DAYS ({len(P)} eligible actions; value per action c): day | n | mean | IC model / B2 / B1 | quintiles of the prediction (realized "
              f"mean, low -> high) | MSE model / B1 / B2 | keep iff pred > 0: pull share, local gain $ (model / B2)"]
    for d in days:
        z = per[d]
        T.append(f"  {d}: {z['n']} | {z['mean']:+.2f} | {z['ic']:+.3f} / {z['ic_b2']:+.3f} / {z['ic_b1']:+.3f} | " + ' '.join(f'{v:+.2f}' for v in z['q'])
                 + f" | {z['mse']:.0f} / {z['mse_b1']:.0f} / {z['mse_b2']:.0f} | {z['pull']:.1%}, {z['gain'] / 100:+.1f} / {z['gain_b2'] / 100:+.1f}")
    T.append(f"  pooled: IC {pool['ic']:+.3f} / {pool['ic_b2']:+.3f} / {pool['ic_b1']:+.3f}; quintiles " + ' '.join(f'{v:+.2f}' for v in qp)
             + f"; pull {pool['pull']:.1%}; local gain ${pool['gain'] / 100:+.1f} (B2 ${pool['gain_b2'] / 100:+.1f}); matched null p {mm['p_value']} "
               f"(null mean kept value ${mm['null_mean'] / 100:+.1f} vs kept ${mm['local_kept_value_cents'] / 100:+.1f}; B2 p {mb['p_value']})")
    T.append('  by side (n | IC | top - bottom quintile): ' + '; '.join(f"{('bid', 'ask')[s]} {sides[s]['n']} | {sides[s]['ic']:+.3f} | {sides[s]['tb']:+.2f}" for s in (0, 1)))
    T.append('  by rung (n | IC | mean value | pull share): ' + '; '.join(
        f"r{r} {int((P.rung.values == r).sum())} | {spearman(P.pred.values[P.rung.values == r], y[P.rung.values == r]):+.3f} | "
        f"{y[P.rung.values == r].mean():+.2f} | {np.mean(~(P.pred.values[P.rung.values == r] > 0)):.0%}" for r in range(E.NR)))
    dec = pd.qcut(pd.Series(P.pred.values).rank(method='first'), 10, labels=False).values
    T.append('  calibration by decile (mean prediction -> realized, c): ' + ' | '.join(f'{P.pred.values[dec == i].mean():+.2f} -> {y[dec == i].mean():+.2f}' for i in range(10)))
    S = P[P.status.values == 'eligible']
    T.append(f"  strict-WS subset (descriptive): {len(S)} actions, IC {spearman(S.pred.values, S.value_cents.values):+.3f}, mean value {S.value_cents.mean():+.2f}")
    T += ['', f"R900 {r900}; R901 {'IC+' if ic_pos else 'IC-NOT-SHOWN'}, {'BEATS-B2-IC' if beats_b2 else 'NOT-ABOVE-B2-IC'}, "
              f"{'BEATS-B1-MSE' if beats_b1 else 'NOT-BELOW-B1-MSE'}; R902 {r902}; R903 {r903}; R904 {r904}"]
    parts = [n for n, ok in (('R900', r900 == 'RANKS'), ('R901 IC+', ic_pos), ('R901 B2', beats_b2), ('R902', r902 == 'BOTH-SIDES'), ('R903', r903 == 'LOCAL-GAIN')) if not ok]
    verdict = 'MODEL-READY' if not parts else 'NOT-READY (' + ', '.join(parts) + ')'
    T.append(f'VERDICT: {verdict} (development only: consumed days; replay rule M; E2 point estimates; a local 1 s value, not portfolio P&L)')
    R.update(per_day=per, pooled=pool, sides=sides, metrics=mm, metrics_b2=mb, readings=dict(R900=r900, R901=[ic_pos, beats_b2, beats_b1], R902=r902,
             R903=r903, R904=r904), verdict=verdict)
    open(f'{OUTD}/stop_ev_model.txt', 'w').write('\n'.join(T) + '\n')
    json.dump(R, open(f'{OUTD}/stop_ev_model.json', 'w'), indent=1, default=float)
    print('\n'.join(T), flush=True)


# ---------------------------------------------------------------- self-tests (synthetic)
def selftest():
    bad = []
    # (1) order inputs: units, NaN for unknown / empty queue, the ask edge sign
    A = pd.DataFrame(dict(x_side=[0, 1, 0], x_rung=[0, 1, 2], x_depth=[0, 1, 2], x_price=[50.0, 53.0, 48.0], x_age_ms=[100.0, 70000.0, 0.0],
                          x_known_remaining=[30.0, 15.0, 30.0], x_submitted_quantity=[30.0, 30.0, 30.0], x_queue_ahead_known=[40.0, np.nan, 0.0],
                          x_shown_known=[80.0, 10.0, 0.0], x_queue_initial_known=[100.0, np.nan, 0.0], x_queue_initial_unknown=[0.0, 1.0, 0.0],
                          x_effective_horizon_ms=[1000.0, 500.0, 1000.0]))
    O = order_inputs(A, [200.0, 200.0, 0.0], [51.0, 51.0, 51.0])
    exp = dict(o_q_rel=[0.2, NAN, NAN], o_q_frac=[0.5, NAN, NAN], o_edge=[1.0, 2.0, 3.0], o_age=[100.0, 60000.0, 0.0], o_rem=[1.0, 0.5, 1.0], o_horizon=[1.0, 0.5, 1.0])
    for c, v in exp.items():
        a = np.asarray(O[c], float); b = np.asarray(v, float)
        if not np.all((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) < 1e-12)):
            bad.append(f'order input {c}: {a.tolist()} vs {b.tolist()}')
    # (2) the join and its depth identity
    R = pd.DataFrame(dict(m=[7, 7, 7, 7], side=[0, 0, 1, 1], k=[0, 1, 0, 1], t=[1000.0, 1050.0, 1000.0, 1050.0], best=[50, 49, 52, 53]))
    Aj = pd.DataFrame(dict(m=[7, 7], side=[0, 1], k=[1, 0], decision_ms=[1050.0, 1000.0], x_side=[0, 1], x_depth=[1.0, 1.0], x_price=[48.0, 53.0]))
    pi, e10, good = join_rows(Aj, R)
    if not (list(pi) == [1, 2] and e10['joined'] == 2 and e10['t_mismatch'] == 0 and e10['depth_mismatch'] == 0 and good.all()):
        bad.append(f'join {list(pi)} {e10}')
    # (3) outcome columns are never inputs
    try:
        inputs_of(FULL_IN + (['filled_quantity'] if MUT_OUTCOME_IN else []))
    except ValueError:
        bad.append('an outcome column among the inputs')
    # (4) the hurdle on a known mixture: P(fill) 0.3, value | fill = +2 or -8 by an input -> E = 0.3 x (+2 / -8)
    rng = np.random.default_rng(3); n = 6000
    x1 = rng.integers(0, 2, n).astype(float); fill = rng.random(n) < 0.3; y = np.where(fill, np.where(x1 > 0, 2.0, -8.0) + rng.normal(0, 0.1, n), 0.0)
    X = np.stack([x1, rng.normal(0, 1, n)], 1).astype(np.float32)
    H = fit_hurdle(X, y, fill, ['x1', 'z'], rounds=60)
    pr = predict_hurdle(H, np.asarray([[1.0, 0.0], [0.0, 0.0]], np.float32))[0]
    if abs(pr[0] - 0.6) > 0.15 or abs(pr[1] + 2.4) > 0.4:
        bad.append(f'hurdle on a known mixture: {pr.tolist()} vs [+0.6, -2.4]')
    # (5) T5 on synthetic data with signal: the permuted fit has no test skill
    Xt = np.stack([rng.integers(0, 2, 3000).astype(float), rng.normal(0, 1, 3000)], 1).astype(np.float32); ft = rng.random(3000) < 0.3
    yt = np.where(ft, np.where(Xt[:, 0] > 0, 2.0, -8.0), 0.0)
    yp, fp = permute_labels(y, fill, rng)
    ic_perm = spearman(predict_hurdle(fit_hurdle(X, yp, fp, ['x1', 'z'], rounds=60), Xt)[0], yt)
    if abs(ic_perm) > 0.1:
        bad.append(f'permutation falsifier: permuted-label IC {ic_perm:+.3f}')
    # (6) B1 table and the matched-null adapter (kept = the positive half must beat the null on a separable toy)
    Dd = pd.DataFrame(dict(rung=[0, 0, 1], side=[0, 0, 1], value_cents=[2.0, 4.0, -1.0]))
    tb = b1_table(Dd)
    if tb.get((0, 0)) != 3.0 or tb.get((1, 1)) != -1.0:
        bad.append(f'B1 table {tb}')
    nn = 400; yy = np.where(np.arange(nn) % 2 == 0, 5.0, -5.0)
    Pm = pd.DataFrame(dict(day=[20261027] * nn, m=np.arange(nn), side=[0] * nn, rung=[0] * nn, generation=[0] * nn, status=['eligible'] * nn, value_cents=yy,
                           hour=[1] * nn, x_known_remaining=[30.0] * nn))
    mt = metric_rows(Pm, yy > 0)
    if not (mt['p_value'] is not None and mt['p_value'] <= 0.01 and abs(mt['local_gain_vs_keep_all_cents'] - 5.0 * nn / 2) < 1e-6):
        bad.append(f'matched null adapter {mt}')
    return bad


def mutation_check():
    out = {}
    for nm in MUTS:
        globals()[nm] = 1
        try:
            out[nm] = bool(selftest())
        finally:
            globals()[nm] = 0
    return out


def all_selftests():
    bad = selftest()
    if bad:
        sys.exit(f'STOP-EV MODEL SELFTEST FAILED: {bad}')
    m = mutation_check()
    if not all(m.values()):
        sys.exit(f'STOP-EV MODEL PLANTED ERROR NOT CAUGHT: {m}')
    print(f'stop-ev model selftests passed; planted errors caught: {m}', flush=True)
    return m


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        all_selftests()
    elif cmd == 'prep':
        prep_day(sys.argv[2])
    elif cmd == 'fit':
        fit_fold(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit(__doc__)
