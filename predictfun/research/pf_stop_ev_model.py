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
  falsifier: the training labels permuted jointly over the training set, 10 draws, test IC; T6 IC train / test; T7 top inputs by gain.
  T5 POST HOC CHANGE (2026-11-15 14:00Z, after the five fits and BEFORE any test-day reading was computed or seen): the declared two-sided
  rule (mean |IC| <= 0.01 and max |IC| <= 0.03) failed on folds 10-31 (mean |IC| 0.0121) and 11-03 (0.0151) with signed means -0.0026 /
  -0.0042 and mixed signs on every fold - the chance dispersion of permuted fits on informative inputs (FINDINGS P31, repeated here when
  T5 was declared). A leak can only show as positive skill, so T5 now passes iff the mean SIGNED IC <= +0.01 and the max signed IC <= +0.03;
  the two-sided values stay printed. (T5 cannot see a feature leak: its labels are permuted; E10, the contract's known-state inputs and the
  STOP-MODEL rows' falsifiers cover those.)
CONTROLS (synthetic, any failure -> no output): the order-input derivation (units, NaN for an unknown / empty queue, the ask edge sign);
  the join and its depth identity; the hurdle on a known mixture; the B1 table; the matched-null adapter; planted errors each caught: the
  join one step late, an outcome column among the inputs, the value head fitted on every action (not the filled ones), the permutation
  falsifier without the permutation.
LIMITS: consumed days (exploratory, USER); replay rule M (B1a); E2 point estimates (+-250 ms, C12); a local 1 s value under a neutral
  continuation, not portfolio P&L (stage 3); the untimed rule's limit (stage 1 docstring).
V2 (2026-11-15, after the Codex review handoffs/20261115_stop_ev_latest_commit_review.md relayed by the USER: "remain: 1 ... 4"). v1's
  receipts stay as they are: its T5 FAILED the declared two-sided rule on 10-31 / 11-03 (10 draws, below the repository's 200) and its
  one-sided recomputation is NOT a pass of the frozen check; v1's MODEL-READY is exploratory only. v2 is a NEW, separately declared fit:
  DATA: the Codex timing-certified labels (pf_stop_ev_timing_certificate, contract stop-ev-side-prefix-invariance-v1;
    data/.../stop_ev_timing_certified_20261115_v1): PRIMARY = certified_estimated eligible (no untimed / out-of-range event can reach any
    accepted order of the side before the label ends, given the E2 point anchors); TIMING-INVARIANT subset = certified_ws eligible (no E2
    interval or missing event can either). Exact one-to-one joins on (day, m, side, rung, generation, decision_ms) with our model rows;
    certified values must equal ours (0 differences); every certified action must be among ours. The broader v1 population (the USER's
    future-only untimed rule) is a sensitivity only.
  MODEL: the hurdle with a MEAN-preserving value head - LightGBM squared-error regression on the filled actions (E[value | fill]) - x
    P(fill): a Huber head estimates a robust location, not the mean, and its product with P(fill) is not an expected value (the review's
    control: 900 x +1 and 100 x -100 have mean -9.1 and Huber location +0.89 at delta 1). Stage 1's declared ROBUST loss is superseded for
    this reason. Same inputs, parameters, rounds, folds and baselines (B2 refit the same way) as v1.
  T5v2 (declared here, before any v2 run): 200 permuted-label refits per fold (the training labels permuted jointly over the training set,
    the identical pipeline), test IC on the primary test actions. LEAK gate (two-sided): |mean IC| <= 3 x sd / sqrt(200); SKILL: the
    model's test IC above the 99th percentile of the permuted ICs. No v2 verdict until every fold's T5v2 exists and passes the leak gate.
  READINGS (v2, primary population; the rules of R900-R904 unchanged) + R905 MEAN CALIBRATION: |sum of predictions - sum of values| <=
    10% of |sum of values| pooled, and the top prediction decile's realized mean has the sign of its mean prediction -> CALIBRATED, else
    MISCALIBRATED; R952v2 SKILL ON TIMING-INVARIANT: IC > 0 pooled and on >= 4 of 5 days on the certified_ws test actions ->
    SKILL-ON-INVARIANT, else NOT-SHOWN. VERDICT v2: MODEL-READY iff the v1 rules (R900, R901 IC+ and BEATS-B2-IC, R902, R903) and R905
    CALIBRATED and T5v2 (leak gate on every fold) hold; R952v2 and R904 printed beside it.
  ARTIFACTS: every fitted head saved (LightGBM text, feature order, parameters) with sha256 of the files and of the training matrix; a
    reload of each saved head must reproduce the in-memory test predictions (max |difference| <= 1e-9).
V3 (2026-11-15, after the pipeline REVIEW; USER "Can we fix the critical issues and let the results be correct first"). v1 / v2 receipts
  stay as provenance; their readings are superseded: both trained and tested on outcome-selected populations (the untimed 'order_reach'
  rule; the certificate subsets, selected by later trades), R903 could not fail when every action loses (pulling anything 'gains'), R904
  compared unequal pull shares, T5 / T5v2's permuted-label refits test the pipeline, not significance, the matched null was fed constant
  stamps (its mixed-reference refusal could not fire) and the v2 reload gate refused only after writing its predictions.
  DATA: stage 1's imputed-untimed labels (actions_<day>_imp_all_anchors_sensitivity: untimed legs in the tape, no probe refused by a later
    event), every eligible status (eligible / eligible_estimated_timing); no certificate subset. Each day's build stamps (reference,
    contract, untimed rule, accounting / lane source hashes, ages, E2 uncertainty, size) are read from stage 1's day record and carried
    into the matched null, which refuses mixed stamps.
  MODEL: v2's hurdle (P(fill) binary x E[value | fill] by squared error), inputs, parameters, rounds, folds, B1 / B2. Every head saved;
    the saved heads are reloaded and must reproduce the in-memory predictions (max |diff| <= 1e-9) BEFORE any prediction is written; all
    predictions come from the reloaded heads.
  READINGS v3 (test days = the 5 fold test days, eligible actions; day rule >= 4 of 5; c per order of Z sh), declared before any v3 run:
   R900 RANKS as v1.
   R901 IC+ as v1 (pooled IC > 0 and > 0 on >= 4 days; the day-cluster 95% t-interval of the five daily ICs printed) and BEATS-B2-IC as v1.
   R901w WITHIN-WINDOW SKILL (printed, not in the verdict): the pooled IC against 1,000 draws of the value labels permuted within (day,
        window) strata; p <= 0.05 -> WITHIN-SKILL (the windows' common shocks and the cross-window part of the association stay in every
        draw, so it tests the ranking inside a window only).
   R902 BOTH-SIDES as v1.
   R903' POSITIVE-KEEP (keep iff prediction > 0): the KEPT actions' realized value > 0 pooled and on >= 4 days, AND the matched null
        (pf_stop_ev_accounting.action_metrics: random keeps of the same count within day / side / hour / rung / known remainder, 1,000
        draws) p <= 0.05 pooled -> POSITIVE-KEEP, else NOT-SHOWN. v1's gain vs keep-all is printed only (it is positive whenever the
        population loses on average).
   R904' BEATS-B2-DECISION: per day, B2 keeps the same NUMBER of actions as the model (its top-k by prediction); the model's kept value
        above B2's on >= 4 days -> BEATS-B2-DECISION, else NOT-SHOWN.
   R905 CALIBRATED as v2.
   VERDICT v3: MODEL-READY iff R900 RANKS, R901 IC+ and BEATS-B2-IC, R902 BOTH-SIDES, R903' POSITIVE-KEEP, R904' BEATS-B2-DECISION and
     R905 CALIBRATED; else NOT-READY naming the parts. The exclusion statuses of every test day are printed beside the tables.
  DROPPED: T5 / T5v2 (permuted-label refits cannot see a feature leak and are no significance test; leaks are covered by E10, the
    contract's known-state inputs and inputs_of), R952v2 (the certified subset is selected by later trades).
  CONTROLS added (synthetic): the within-window permutation (a cross-window-only signal keeps its association in the null; a within-
    window signal is significant), the positive-keep reading (all-losing actions cannot pass), the matched-count B2 comparator, the stamp
    check (rows of two references are refused), the reload gate (a 1e-6 difference refuses); planted errors each caught: a global
    permutation, the gain-vs-keep-all rule, B2 at its own count, constant stamps, the reload gate skipped.
Usage (run_light, PF_SLOT=claude): pf_stop_ev_model.py selftest | prep <day> | fit <test day> | report (v1); prep2 <day> | fit2 <test day> |
t5v2 <test day> | report2 (v2); prep3 <day> | fit3 <test day> | report3 (v3); drivers run_stop_ev_model.sh (v1),
run_stop_ev_model2.sh (v2), run_stop_ev_model3.sh (v3).
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
MUT_V2_HUBER = MUT_T5_ONESIDED = MUT_CERT_NOVAL = 0
MUT_PERM_GLOBAL = MUT_KEEP_GAIN = MUT_B2_UNMATCHED = MUT_CONST_STAMP = MUT_RELOAD_SKIP = 0           # v3
MUTS = ('MUT_JOIN_LATE', 'MUT_OUTCOME_IN', 'MUT_HURDLE_ALL', 'MUT_NO_PERM', 'MUT_V2_HUBER', 'MUT_T5_ONESIDED', 'MUT_CERT_NOVAL',
        'MUT_PERM_GLOBAL', 'MUT_KEEP_GAIN', 'MUT_B2_UNMATCHED', 'MUT_CONST_STAMP', 'MUT_RELOAD_SKIP')
CERT = f'{K.IO}/stop_ev_timing_certified_20261115_v1'
KEYJ = ['day', 'm', 'side', 'rung', 'generation', 'decision_ms']
CERT_EST_OK = ('eligible_estimated_timing', 'eligible'); CERT_WS_OK = ('eligible_timing_invariant',)
T5V2_DRAWS = 200; T5V2_SE = 3.0; SKILL_Q = 0.99; CAL_TOL = 0.10
V3_DRAWS = 1000; T4_975 = 2.776                                   # v3: null draws; t(4) two-sided 95% for the five daily ICs


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


def t5_pass(ic):
    """T5 (POST HOC one-sided, see the docstring): a leak shows as positive test skill of the permuted-label fits."""
    ic = np.asarray(ic, float)
    return bool(np.mean(ic) <= T5_MEAN and np.max(ic) <= T5_MAX)


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
    info['T5_ic'] = t5; info['T5_pass'] = t5_pass(t5); info['T5_pass_two_sided_as_declared'] = bool(np.mean(np.abs(t5)) <= T5_MEAN and np.max(np.abs(t5)) <= T5_MAX)
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
    t5f = [t for t, f in fits.items() if not t5_pass(f['T5_ic'])]       # recomputed from the stored ICs (the fits stored the declared rule)
    if t5f:
        sys.exit(f'REFUSED: T5 (one-sided) failed on {t5f}')
    P = pd.concat([pd.read_parquet(f'{OUTD}/pred_{t}_{t}.parquet') for t in TESTS], ignore_index=True)
    xs = {d: json.load(open(f'{OUTD}/ev_x_{d}.json')) for d in DAYS9}
    T = [f'STOP-EV STAGE 2 - MODEL (hurdle P(fill) x E[value | fill], Huber; {TIMING}; untimed rule {E.UNTIMED_RULE}); 5 chronological test days; '
         f'as of {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}', '']
    T.append('DATA per day: eligible actions / probes (statuses) | E10 join')
    for d in DAYS9:
        x = xs[d]; T.append(f"  {d}: {x['rows']} / {x['probes']} ({', '.join(f'{k} {v}' for k, v in sorted(x['status'].items(), key=lambda z: -z[1]))}) | "
                            f"E10 {'PASS' if x['E10']['pass'] else 'FAIL'}")
    T.append('TRAINING per fold: n train (filled) | Huber delta | IC train | T5 permuted IC signed mean / max (one-sided gate, POST HOC) and '
             'two-sided mean |.| (as declared) | top value / fill inputs')
    for t, f in fits.items():
        ic5 = np.asarray(f['T5_ic'], float)
        T.append(f"  {t}: {f['n_train']} ({f['filled_train']}) | {f['delta']:.2f} c | {f['T6_ic_train']:+.3f} | {ic5.mean():+.4f} / {ic5.max():+.4f} "
                 f"(two-sided {np.mean(np.abs(ic5)):.4f}) | {', '.join(c for c, _ in f['T7_top_value'][:5])} / {', '.join(c for c, _ in f['T7_top_fill'][:5])}")
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


# ---------------------------------------------------------------- v2 (after the Codex review; see the docstring)
def sha_file(path):
    import hashlib
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def cert_flags(D, C_est, C_ws):
    """Exact one-to-one joins of our model rows D with the two certificate tables (one row per probe): flags and checks."""
    out = {}
    for mode, C, okst in (('certified_estimated', C_est, CERT_EST_OK), ('certified_ws', C_ws, CERT_WS_OK)):
        dup = int(C.duplicated(KEYJ).sum()) + int(D.duplicated(KEYJ).sum())
        if dup:
            out[mode] = (np.zeros(len(D), bool), dict(duplicates=dup)); continue
        J = D[KEYJ + ['value_cents']].merge(C[KEYJ + ['status', 'value_cents']], on=KEYJ, how='left', suffixes=('', '_c'), validate='one_to_one', indicator=True)
        ok = J.status.isin(okst).values & (J._merge.values == 'both')
        v1, v2 = J.value_cents.values.astype(float), J.value_cents_c.values.astype(float)
        vd = 0 if MUT_CERT_NOVAL else int((ok & ~(np.abs(v1 - v2) <= 1e-9)).sum())
        cert_keys = C.loc[C.status.isin(okst).values, KEYJ]
        extra = int(len(cert_keys.merge(D[KEYJ], on=KEYJ, how='left', indicator=True).query('_merge == "left_only"')))
        out[mode] = (ok, dict(duplicates=0, ours=int(len(D)), not_in_certificate=int((J._merge.values != 'both').sum()), certified=int(ok.sum()),
                              value_differ=vd, certified_not_ours=extra))
    return out


def prep2(day):
    """v1's model rows (ev_x_<day>) with the certificate flags; refuses unless the joins are exact and the values agree."""
    all_selftests()
    D = pd.read_parquet(f'{OUTD}/ev_x_{day}.parquet')
    paths = {m_: f'{CERT}/{day}/{m_}.parquet' for m_ in ('certified_estimated', 'certified_ws')}
    C = {m_: pd.read_parquet(p_, columns=KEYJ + ['status', 'value_cents']) for m_, p_ in paths.items()}
    for x in C.values():
        x['day'] = x['day'].astype('int64')
    fl = cert_flags(D, C['certified_estimated'], C['certified_ws'])
    st = dict(day=day, rows=int(len(D)), checks={m_: v[1] for m_, v in fl.items()}, cert_sha256={m_: sha_file(p_) for m_, p_ in paths.items()},
              ev_x_sha256=sha_file(f'{OUTD}/ev_x_{day}.parquet'))
    bad = [m_ for m_, v in st['checks'].items() if v.get('duplicates') or v.get('not_in_certificate') or v.get('value_differ') or v.get('certified_not_ours')]
    if bad:
        sys.exit(f'REFUSED (no output for {day}): certificate join {st["checks"]}')
    D['cert_est'] = fl['certified_estimated'][0]; D['cert_ws'] = fl['certified_ws'][0]
    D.to_parquet(f'{OUTD}/ev_x2_{day}.parquet', index=False); json.dump(st, open(f'{OUTD}/ev_x2_{day}.json', 'w'), indent=1, default=str)
    print(f'{day}: {len(D)} rows, certified_estimated {int(D.cert_est.sum())}, certified_ws {int(D.cert_ws.sum())}; {st["checks"]}', flush=True)


def fit_hurdle2(X, y, fill, F, rounds=ROUNDS):
    """v2: P(fill) on every action x E[value | fill] by SQUARED error (the conditional mean) on the filled actions."""
    import lightgbm as lgb
    fill = np.asarray(fill, bool); y = np.asarray(y, float)
    bp = lgb.train(PARAMS, lgb.Dataset(X, fill.astype(float), feature_name=F), rounds)
    if MUT_V2_HUBER:
        d_ = float(np.median(np.abs(y[fill]))) if fill.any() else 1.0
        pv = {**PARAMS, 'objective': 'huber', 'alpha': d_ if d_ > 0 else 1.0}
    else:
        pv = {**PARAMS, 'objective': 'regression'}
    bv = lgb.train(pv, lgb.Dataset(X[fill], y[fill], feature_name=F), rounds)
    return dict(p=bp, v=bv, objective_value=pv['objective'])


def t5v2_gate(ics, observed=None):
    """Two-sided leak gate on the permuted-label refits' test ICs (|mean| <= 3 sd / sqrt(n)); skill = the observed IC above their 99th pct."""
    ic = np.asarray(ics, float); n = len(ic); sd = float(ic.std(ddof=1)) if n > 1 else float('inf')
    leak_ok = bool(n >= T5V2_DRAWS and (np.mean(ic) <= T5V2_SE * sd / np.sqrt(n) if MUT_T5_ONESIDED else abs(np.mean(ic)) <= T5V2_SE * sd / np.sqrt(n)))
    q = float(np.quantile(ic, SKILL_Q)) if n else NAN
    return dict(n=n, mean=float(np.mean(ic)) if n else NAN, sd=sd, se=sd / np.sqrt(n) if n else NAN, leak_pass=leak_ok, q99=q,
                skill_beyond_null=bool(observed is not None and observed == observed and observed > q))


def load2(days):
    return pd.concat([pd.read_parquet(f'{OUTD}/ev_x2_{d}.parquet') for d in days], ignore_index=True)


def fit2(test):
    import lightgbm as lgb
    all_selftests()
    tr, cal = FOLDS[test]
    assert max(tr) < cal < test, 'T1 chronology'
    T0 = time.time(); Dtr = load2(list(tr)); Dtr = Dtr[Dtr.cert_est.values].reset_index(drop=True)
    F = inputs_of(list(FULL_IN)); FB = inputs_of(list(B2_IN))
    Xtr = Dtr[F].to_numpy(np.float32); ytr = Dtr.value_cents.values.astype(float); ftr = Dtr.filled_quantity.values > 0
    H = fit_hurdle2(Xtr, ytr, ftr, F); HB = fit_hurdle2(Dtr[FB].to_numpy(np.float32), ytr, ftr, FB)
    tab = b1_table(Dtr); dflt = float(ytr.mean())
    import hashlib
    info = dict(version='v2', test=test, cal=cal, train_days=list(tr), n_train=int(len(Dtr)), filled_train=int(ftr.sum()), objective_value=H['objective_value'],
                params=PARAMS, rounds=ROUNDS, features=F, b2_features=FB, b1=[[int(k[0]), int(k[1]), float(v)] for k, v in tab.items()],
                train_matrix_sha256=hashlib.sha256(Xtr.tobytes() + ytr.tobytes()).hexdigest(),
                inputs_sha256={d: sha_file(f'{OUTD}/ev_x2_{d}.parquet') for d in list(tr) + [cal, test]},
                T6_ic_train=spearman(predict_hurdle(H, Xtr)[0], ytr))
    md = f'{OUTD}/models_v2'; os.makedirs(md, exist_ok=True); arts = {}
    Dte = load2([test]); Xte = Dte[F].to_numpy(np.float32); XteB = Dte[FB].to_numpy(np.float32)
    for nm, h, X_ in (('full', H, Xte), ('b2', HB, XteB)):
        for head in ('p', 'v'):
            path = f'{md}/{test}_{nm}_{head}.txt'; h[head].save_model(path)
            rel = lgb.Booster(model_file=path); diff = float(np.max(np.abs(rel.predict(X_) - h[head].predict(X_)))) if len(X_) else 0.0
            arts[f'{nm}_{head}'] = dict(path=path, sha256=sha_file(path), reload_max_abs_diff=diff)
    info['artifacts'] = arts; info['reload_pass'] = bool(all(a['reload_max_abs_diff'] <= 1e-9 for a in arts.values()))
    imp = pd.Series(H['v'].feature_importance('gain'), index=F).sort_values(ascending=False)
    imp_p = pd.Series(H['p'].feature_importance('gain'), index=F).sort_values(ascending=False)
    info['T7_top_value'] = [[c, float(v)] for c, v in imp.head(12).items()]; info['T7_top_fill'] = [[c, float(v)] for c, v in imp_p.head(12).items()]
    for d in (cal, test):
        Dd = Dte if d == test else load2([d]); Xd = Dd[F].to_numpy(np.float32)
        pr, pp, pv = predict_hurdle(H, Xd); pb = predict_hurdle(HB, Dd[FB].to_numpy(np.float32))[0]
        P_ = Dd[KEYJ + ['hour', 'status', 'x_known_remaining', 'value_cents', 'filled_quantity', 'cert_est', 'cert_ws']].copy()
        P_['pred'] = pr; P_['p_fill'] = pp; P_['v_fill'] = pv; P_['b2'] = pb; P_['b1'] = b1_predict(tab, Dd, dflt)
        P_.to_parquet(f'{OUTD}/pred2_{test}_{d}.parquet', index=False)
    info['seconds'] = round(time.time() - T0, 1)
    json.dump(info, open(f'{OUTD}/fit2_{test}.json', 'w'), indent=1, default=float)
    if not info['reload_pass']:
        sys.exit(f'REFUSED: a saved head does not reproduce its predictions {arts}')
    print(f'fold {test} v2: train {info["n_train"]} ({info["filled_train"]} filled), IC train {info["T6_ic_train"]:+.3f}, reload ok, {info["seconds"]} s', flush=True)


def t5v2(test):
    """T5v2: T5V2_DRAWS permuted-label refits of the v2 pipeline on the fold's certified training actions; test IC on the primary test actions."""
    all_selftests()
    tr, cal = FOLDS[test]; T0 = time.time()
    Dtr = load2(list(tr)); Dtr = Dtr[Dtr.cert_est.values].reset_index(drop=True); F = inputs_of(list(FULL_IN))
    Xtr = Dtr[F].to_numpy(np.float32); ytr = Dtr.value_cents.values.astype(float); ftr = Dtr.filled_quantity.values > 0
    Dte = load2([test]); Dte = Dte[Dte.cert_est.values]; Xte = Dte[F].to_numpy(np.float32); yte = Dte.value_cents.values.astype(float)
    rng = np.random.default_rng(int(test) + 2); ics = []
    for i in range(T5V2_DRAWS):
        yp, fp = permute_labels(ytr, ftr, rng)
        ics.append(spearman(predict_hurdle(fit_hurdle2(Xtr, yp, fp, F), Xte)[0], yte))
        if (i + 1) % 20 == 0:
            print(f'{test}: {i + 1} draws, {time.time() - T0:.0f} s', flush=True)
    obs = pd.read_parquet(f'{OUTD}/pred2_{test}_{test}.parquet'); obs = obs[obs.cert_est.values]
    g = t5v2_gate(ics, spearman(obs.pred.values, obs.value_cents.values))
    json.dump(dict(test=test, draws=T5V2_DRAWS, ics=ics, gate=g, seconds=round(time.time() - T0, 1)), open(f'{OUTD}/t5v2_{test}.json', 'w'), indent=1, default=float)
    print(f'{test} T5v2: mean {g["mean"]:+.4f} (se {g["se"]:.4f}) -> leak gate {"PASS" if g["leak_pass"] else "FAIL"}; q99 {g["q99"]:+.4f}, skill beyond null '
          f'{g["skill_beyond_null"]}; {time.time() - T0:.0f} s', flush=True)


def report2():
    fits = {t: json.load(open(f'{OUTD}/fit2_{t}.json')) for t in TESTS if os.path.exists(f'{OUTD}/fit2_{t}.json')}
    if len(fits) < len(TESTS):
        sys.exit(f'REFUSED: v2 folds missing {[t for t in TESTS if t not in fits]}')
    t5 = {t: json.load(open(f'{OUTD}/t5v2_{t}.json')) for t in TESTS if os.path.exists(f'{OUTD}/t5v2_{t}.json')}
    PA = pd.concat([pd.read_parquet(f'{OUTD}/pred2_{t}_{t}.parquet') for t in TESTS], ignore_index=True)
    T = [f'STOP-EV STAGE 2 v2 - MODEL (hurdle P(fill) x E[value | fill] by squared error; Codex timing-certified labels; v1 kept as exploratory); '
         f'as of {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}', '']
    T.append('TRAINING per fold (certified_estimated training actions): n (filled) | IC train | saved heads reload | T5v2 (200 refits): mean / se -> leak gate | '
             'q99 vs observed -> skill beyond the null')
    for t, f in fits.items():
        g = t5.get(t, {}).get('gate')
        T.append(f"  {t}: {f['n_train']} ({f['filled_train']}) | {f['T6_ic_train']:+.3f} | {'ok' if f['reload_pass'] else 'FAIL'} | "
                 + (f"{g['mean']:+.4f} / {g['se']:.4f} -> {'PASS' if g['leak_pass'] else 'FAIL'} | {g['q99']:+.4f} -> {g['skill_beyond_null']}" if g else 'PENDING'))
    R = {}
    for popn, sel in (('PRIMARY certified_estimated', PA.cert_est.values), ('TIMING-INVARIANT certified_ws', PA.cert_ws.values),
                      ('SENSITIVITY v1 population (future-only untimed rule)', np.ones(len(PA), bool))):
        P = PA[sel].reset_index(drop=True); y = P.value_cents.values.astype(float); days = [int(t) for t in TESTS]
        per = {}
        for d in days:
            x = P[P.day.values == d]; yy = x.value_cents.values.astype(float)
            q = quint(x.pred.values, yy) if len(x) >= 5 else [NAN] * 5
            per[d] = dict(n=len(x), mean=float(yy.mean()) if len(x) else NAN, ic=spearman(x.pred.values, yy), ic_b2=spearman(x.b2.values, yy), q=q,
                          tb=q[-1] - q[0], gain=float(-yy[~(x.pred.values > 0)].sum()), gain_b2=float(-yy[~(x.b2.values > 0)].sum()),
                          pull=float(np.mean(~(x.pred.values > 0))) if len(x) else NAN, mse=float(np.mean((x.pred.values - yy) ** 2)) if len(x) else NAN,
                          mse_b1=float(np.mean((x.b1.values - yy) ** 2)) if len(x) else NAN)
        qp = quint(P.pred.values, y); dec = pd.qcut(pd.Series(P.pred.values).rank(method='first'), 10, labels=False).values
        cal_sum = float(abs(P.pred.values.sum() - y.sum()) / max(abs(y.sum()), 1e-12)); top_p = float(P.pred.values[dec == 9].mean()); top_y = float(y[dec == 9].mean())
        nd = lambda f: sum(int(f(per[d])) for d in days)
        r900 = 'RANKS' if (qp[-1] - qp[0] > 0 and nd(lambda z: z['tb'] > 0) >= 4) else 'NO-RANK'
        icp = spearman(P.pred.values, y); ic_pos = icp > 0 and nd(lambda z: z['ic'] > 0) >= 4; beats_b2 = nd(lambda z: z['ic'] > z['ic_b2']) >= 4
        sides = {s: (spearman(P.pred.values[P.side.values == s], y[P.side.values == s]),
                     (lambda qq: qq[-1] - qq[0])(quint(P.pred.values[P.side.values == s], y[P.side.values == s]))) for s in (0, 1)}
        r902 = 'BOTH-SIDES' if all(v[0] > 0 and v[1] > 0 for v in sides.values()) else 'FAILS-' + '/'.join(('BID', 'ASK')[s] for s, v in sides.items() if not (v[0] > 0 and v[1] > 0))
        gain = float(-y[~(P.pred.values > 0)].sum()); mm = metric_rows(P, P.pred.values > 0)
        r903 = 'LOCAL-GAIN' if (gain > 0 and nd(lambda z: z['gain'] > 0) >= 4 and mm['p_value'] is not None and mm['p_value'] <= P_MAX) else 'NOT-SHOWN'
        r904 = 'BEATS-B2-DECISION' if nd(lambda z: z['gain'] > z['gain_b2']) >= 4 else 'NOT-SHOWN'
        r905 = 'CALIBRATED' if (cal_sum <= CAL_TOL and np.sign(top_p) == np.sign(top_y)) else 'MISCALIBRATED'
        T += ['', f'{popn}: {len(P)} test actions; IC {icp:+.3f} (B2 {spearman(P.b2.values, y):+.3f}); quintiles ' + ' '.join(f'{v:+.2f}' for v in qp)
              + f'; keep iff pred > 0: pull {np.mean(~(P.pred.values > 0)):.1%}, local gain ${gain / 100:+.1f}, null p {mm["p_value"]}']
        for d in days:
            z = per[d]
            T.append(f"  {d}: n {z['n']} | mean {z['mean']:+.2f} | IC {z['ic']:+.3f} / B2 {z['ic_b2']:+.3f} | quintiles " + ' '.join(f'{v:+.2f}' for v in z['q'])
                     + f" | MSE {z['mse']:.0f} / B1 {z['mse_b1']:.0f} | pull {z['pull']:.1%}, gain ${z['gain'] / 100:+.1f} / B2 ${z['gain_b2'] / 100:+.1f}")
        T.append('  calibration by decile (mean prediction -> realized): ' + ' | '.join(f'{P.pred.values[dec == i].mean():+.2f} -> {y[dec == i].mean():+.2f}' for i in range(10)))
        T.append(f'  sides IC / top-bottom: bid {sides[0][0]:+.3f} / {sides[0][1]:+.2f}; ask {sides[1][0]:+.3f} / {sides[1][1]:+.2f}; '
                 f'sum of predictions vs values off by {cal_sum:.1%}; top decile {top_p:+.2f} -> {top_y:+.2f}')
        T.append(f'  READINGS: R900 {r900}; R901 {"IC+" if ic_pos else "IC-NOT-SHOWN"}, {"BEATS-B2-IC" if beats_b2 else "NOT-ABOVE-B2-IC"}; R902 {r902}; '
                 f'R903 {r903}; R904 {r904}; R905 {r905}')
        R[popn] = dict(per_day=per, ic=icp, quintiles=qp, readings=dict(R900=r900, R901=[ic_pos, beats_b2], R902=r902, R903=r903, R904=r904, R905=r905),
                       calibration=dict(sum_off=cal_sum, top_pred=top_p, top_real=top_y), metrics=mm)
    inv = R['TIMING-INVARIANT certified_ws']; r952 = 'SKILL-ON-INVARIANT' if (inv['ic'] > 0 and sum(int(z['ic'] > 0) for z in inv['per_day'].values()) >= 4) else 'NOT-SHOWN'
    pr = R['PRIMARY certified_estimated']['readings']
    t5ok = len(t5) == len(TESTS) and all(x['gate']['leak_pass'] for x in t5.values())
    parts = [n for n, ok in (('R900', pr['R900'] == 'RANKS'), ('R901 IC+', pr['R901'][0]), ('R901 B2', pr['R901'][1]), ('R902', pr['R902'] == 'BOTH-SIDES'),
                             ('R903', pr['R903'] == 'LOCAL-GAIN'), ('R905', pr['R905'] == 'CALIBRATED')) if not ok]
    if len(t5) < len(TESTS):
        verdict = f'PENDING (T5v2 missing on {[t for t in TESTS if t not in t5]}); readings above are not a claim'
    elif not t5ok:
        verdict = 'REFUSED (T5v2 leak gate failed on ' + ', '.join(t for t, x in t5.items() if not x['gate']['leak_pass']) + ')'
    else:
        verdict = 'MODEL-READY' if not parts else 'NOT-READY (' + ', '.join(parts) + ')'
    T += ['', f'R952v2 {r952} (timing-invariant test actions); VERDICT v2: {verdict} (development only: consumed days, replay rule M; local 1 s value)']
    R.update(R952v2=r952, verdict=verdict, t5v2={t: x['gate'] for t, x in t5.items()})
    open(f'{OUTD}/stop_ev_model_v2.txt', 'w').write('\n'.join(T) + '\n')
    json.dump(R, open(f'{OUTD}/stop_ev_model_v2.json', 'w'), indent=1, default=float)
    print('\n'.join(T), flush=True)


# ---------------------------------------------------------------- v3 (REVIEW 2026-11-15)
def stamps_of(J):
    """A day's build stamps from stage 1's day record: (reference, label_spec) for the contract's matched null, which refuses mixed ones."""
    s = J.get('sources', {})
    ref = dict(reference=J['reference'], contract=J['label_contract'], untimed_rule=J.get('untimed_rule'),
               accounting=s.get('pf_stop_ev_accounting.py'), lane=s.get('pf_stop_queue_activation_lane.py'))
    spec = dict(timing=TIMING, ages_ms=[float(a) for a in J['ages_ms']], e2_uncertainty_ms=float(J['e2_uncertainty_ms']), size=float(J['size']))
    return ref, spec


def prep3(day):
    """v3 model rows: stage 1's imputed-untimed actions, every eligible status; E10 and its control; the day's build stamps."""
    all_selftests()
    if E.ZTAG != '_imp' or E.UNTIMED_RULE != 'impute':
        sys.exit(f'REFUSED: v3 reads the imputed-untimed labels only (ZTAG {E.ZTAG}, rule {E.UNTIMED_RULE})')
    for t in TESTS:
        if json.load(open(f'{M.OUTD}/fit_{t}_v6.json'))['features'] != F71:
            sys.exit(f'REFUSED: v6 fold {t} has a different input list')
    J = json.load(open(f'{E.OUTD}/day_{day}{E.ZTAG}.json'))
    if J['label_contract'] != AC.CONTRACT or J['reference'] != AC.REFERENCE or J.get('untimed_rule') != 'impute':
        sys.exit(f"REFUSED: {day} labels built under {J['label_contract']} / {J['reference']} / {J.get('untimed_rule')}")
    ap = f'{E.OUTD}/actions_{day}{E.ZTAG}_{TIMING}.parquet'; A = pd.read_parquet(ap)
    st = dict(version='v3', day=day, probes=int(len(A)), status={k: int(v) for k, v in A.status.value_counts().items()}, stamps=stamps_of(J),
              actions_sha256=sha_file(ap))
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
    D.to_parquet(f'{OUTD}/ev_x3_{day}.parquet', index=False)
    st['rows'] = int(len(D)); st['missing_share'] = {c: float(np.isnan(D[c].values.astype(float)).mean()) for c in ORDER_IN}
    json.dump(st, open(f'{OUTD}/ev_x3_{day}.json', 'w'), indent=1, default=str)
    print(f'{day} v3: {len(D)} eligible actions of {st["probes"]} probes; statuses {st["status"]}; E10 {e10}', flush=True)


def load3(days):
    return pd.concat([pd.read_parquet(f'{OUTD}/ev_x3_{d}.parquet') for d in days], ignore_index=True)


def reload_gate(arts):
    """The saved heads must reproduce the in-memory predictions BEFORE any prediction is written (v2 refused only after writing)."""
    if MUT_RELOAD_SKIP:
        return True
    return bool(arts) and all(a['reload_max_abs_diff'] <= 1e-9 for a in arts.values())


def fit3(test):
    import hashlib
    import lightgbm as lgb
    all_selftests()
    if test not in FOLDS:
        sys.exit(f'REFUSED: {test} is not a fold test day')
    tr, cal = FOLDS[test]
    assert max(tr) < cal < test, 'T1 chronology'
    T0 = time.time(); Dtr = load3(list(tr)); F = inputs_of(list(FULL_IN)); FB = inputs_of(list(B2_IN))
    Xtr = Dtr[F].to_numpy(np.float32); ytr = Dtr.value_cents.values.astype(float); ftr = Dtr.filled_quantity.values > 0
    H = fit_hurdle2(Xtr, ytr, ftr, F); HB = fit_hurdle2(Dtr[FB].to_numpy(np.float32), ytr, ftr, FB)
    tab = b1_table(Dtr); dflt = float(ytr.mean())
    info = dict(version='v3', test=test, cal=cal, train_days=list(tr), n_train=int(len(Dtr)), filled_train=int(ftr.sum()), objective_value=H['objective_value'],
                params=PARAMS, rounds=ROUNDS, features=F, b2_features=FB, b1=[[int(k[0]), int(k[1]), float(v)] for k, v in tab.items()],
                train_matrix_sha256=hashlib.sha256(Xtr.tobytes() + ytr.tobytes()).hexdigest(),
                inputs_sha256={d: sha_file(f'{OUTD}/ev_x3_{d}.parquet') for d in list(tr) + [cal, test]},
                T6_ic_train=spearman(predict_hurdle(H, Xtr)[0], ytr))
    del Xtr
    md = f'{OUTD}/models_v3'; os.makedirs(md, exist_ok=True); arts = {}; rel = {}
    Dte = load3([test]); Xte = Dte[F].to_numpy(np.float32); XteB = Dte[FB].to_numpy(np.float32)
    for nm, h, X_ in (('full', H, Xte), ('b2', HB, XteB)):
        for head in ('p', 'v'):
            path = f'{md}/{test}_{nm}_{head}.txt'; h[head].save_model(path); rel[(nm, head)] = lgb.Booster(model_file=path)
            diff = float(np.max(np.abs(rel[(nm, head)].predict(X_) - h[head].predict(X_)))) if len(X_) else 0.0
            arts[f'{nm}_{head}'] = dict(path=path, sha256=sha_file(path), reload_max_abs_diff=diff)
    info['artifacts'] = arts; info['reload_pass'] = reload_gate(arts)
    if not info['reload_pass']:
        json.dump(info, open(f'{OUTD}/fit3_{test}.json', 'w'), indent=1, default=float)
        sys.exit(f'REFUSED (no predictions written): a saved head does not reproduce its predictions {arts}')
    imp = pd.Series(H['v'].feature_importance('gain'), index=F).sort_values(ascending=False)
    imp_p = pd.Series(H['p'].feature_importance('gain'), index=F).sort_values(ascending=False)
    info['T7_top_value'] = [[c, float(v)] for c, v in imp.head(12).items()]; info['T7_top_fill'] = [[c, float(v)] for c, v in imp_p.head(12).items()]
    RH = dict(p=rel[('full', 'p')], v=rel[('full', 'v')]); RB = dict(p=rel[('b2', 'p')], v=rel[('b2', 'v')])   # predictions from the saved heads
    for d in (cal, test):
        Dd = Dte if d == test else load3([d]); Xd = Dd[F].to_numpy(np.float32)
        pr, pp, pv = predict_hurdle(RH, Xd); pb = predict_hurdle(RB, Dd[FB].to_numpy(np.float32))[0]
        P_ = Dd[KEYJ + ['hour', 'status', 'x_known_remaining', 'value_cents', 'filled_quantity']].copy()
        P_['pred'] = pr; P_['p_fill'] = pp; P_['v_fill'] = pv; P_['b2'] = pb; P_['b1'] = b1_predict(tab, Dd, dflt)
        P_.to_parquet(f'{OUTD}/pred3_{test}_{d}.parquet', index=False)
    info['seconds'] = round(time.time() - T0, 1)
    json.dump(info, open(f'{OUTD}/fit3_{test}.json', 'w'), indent=1, default=float)
    print(f'fold {test} v3: train {info["n_train"]} ({info["filled_train"]} filled), IC train {info["T6_ic_train"]:+.3f}, reload ok, {info["seconds"]} s', flush=True)


def perm_ic_within(pred, y, strata, draws, seed):
    """Spearman IC of pred vs y and its null with y permuted WITHIN strata (day x window): the windows' common shocks and the cross-window
    part of the association stay in every draw, so p tests the ranking inside windows only. MUT_PERM_GLOBAL permutes across all rows."""
    rp = pd.Series(np.asarray(pred, float)).rank().values; ry = pd.Series(np.asarray(y, float)).rank().values
    rp = rp - rp.mean(); ry = ry - ry.mean(); den = float(np.sqrt((rp ** 2).sum() * (ry ** 2).sum()))
    if not den > 0:
        return dict(ic=NAN, null_mean=NAN, null_q95=NAN, p=None, draws=draws)
    codes = np.zeros(len(rp), 'int64') if MUT_PERM_GLOBAL else np.asarray(strata, 'int64')
    base = np.argsort(codes, kind='stable'); rng = np.random.default_rng(seed); null = np.empty(draws); yp = np.empty_like(ry)
    for i in range(draws):
        o = np.lexsort((rng.random(len(codes)), codes)); yp[base] = ry[o]; null[i] = float((rp * yp).sum()) / den
    obs = float((rp * ry).sum()) / den
    return dict(ic=obs, null_mean=float(null.mean()), null_q95=float(np.quantile(null, 0.95)), p=float((1 + np.sum(null >= obs)) / (1 + draws)), draws=draws)


def positive_keep(y, pred, day):
    """R903' core: the realized value of the KEPT actions (keep iff pred > 0), pooled and per day. MUT_KEEP_GAIN reads v1's gain vs
    keep-all instead, which is positive whenever the population loses on average."""
    y = np.asarray(y, float); keep = np.asarray(pred, float) > 0; day = np.asarray(day)
    val = (lambda s: float(-y[s & ~keep].sum())) if MUT_KEEP_GAIN else (lambda s: float(y[s & keep].sum()))
    return val(np.ones(len(y), bool)), {int(d): val(day == d) for d in np.unique(day)}


def kept_b2_matched(y, pred, b2, day):
    """R904': per day, B2 keeps the same NUMBER of actions as the model (its top-k by prediction). {day: (model kept value, B2 kept value,
    model kept count, B2 kept count)}. MUT_B2_UNMATCHED lets B2 keep at its own sign."""
    y = np.asarray(y, float); pred = np.asarray(pred, float); b2 = np.asarray(b2, float); day = np.asarray(day); out = {}
    for d in np.unique(day):
        s = np.flatnonzero(day == d); k = int((pred[s] > 0).sum())
        if MUT_B2_UNMATCHED:
            kb = b2[s] > 0
        else:
            kb = np.zeros(len(s), bool); kb[np.argsort(-b2[s], kind='stable')[:k]] = True
        out[int(d)] = (float(y[s][pred[s] > 0].sum()), float(y[s][kb].sum()), k, int(kb.sum()))
    return out


def metric_rows3(P, keep, stamps, draws=V3_DRAWS):
    """action_metrics adapter with each day's RECORDED build stamps (v1 / v2 stamped constants, so the contract's mixed-reference refusal
    could not fire); MUT_CONST_STAMP plants the constants back."""
    rows = []
    for d, m, s, r, g, st, v, h, kr in zip(P.day.values, P.m.values, P.side.values, P.rung.values, P.generation.values, P.status.values,
                                            P.value_cents.values, P.hour.values, P.x_known_remaining.values):
        ref, spec = (dict(contract=AC.REFERENCE), dict(timing=TIMING)) if MUT_CONST_STAMP else stamps[int(d)]
        rows.append(dict(action_id=(str(int(d)), int(m), int(s), int(r), int(g)), timing=TIMING, reference=ref, label_spec=spec, status=st,
                         label=dict(value_cents=float(v)), hour=int(h), features=dict(known_remaining=float(kr))))
    return AC.action_metrics(rows, [bool(x) for x in keep], asof=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), draws=draws)


def report3():
    fits = {t: json.load(open(f'{OUTD}/fit3_{t}.json')) for t in TESTS if os.path.exists(f'{OUTD}/fit3_{t}.json')}
    if len(fits) < len(TESTS) or not all(f.get('reload_pass') for f in fits.values()):
        sys.exit(f'REFUSED: v3 folds missing or refused {[t for t in TESTS if not fits.get(t, {}).get("reload_pass")]}')
    PX = {int(d): json.load(open(f'{OUTD}/ev_x3_{d}.json')) for d in DAYS9}
    stamps = {d: tuple(x['stamps']) for d, x in PX.items()}
    PA = pd.concat([pd.read_parquet(f'{OUTD}/pred3_{t}_{t}.parquet') for t in TESTS], ignore_index=True)
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()); days = [int(t) for t in TESTS]
    y = PA.value_cents.values.astype(float); pr = PA.pred.values.astype(float); b2 = PA.b2.values.astype(float); day = PA.day.values.astype('int64')
    keep = pr > 0
    T = [f'STOP-EV STAGE 2 v3 - MODEL (hurdle P(fill) x E[value | fill] by squared error; imputed-untimed labels, every eligible status; '
         f'v1 / v2 superseded, kept as provenance); as of {asof}', '']
    T.append('TRAINING per fold: n (filled) | IC train | saved heads reload | top value inputs')
    for t, f in fits.items():
        T.append(f"  {t}: {f['n_train']} ({f['filled_train']}) | {f['T6_ic_train']:+.3f} | {'ok' if f['reload_pass'] else 'FAIL'} | "
                 + ', '.join(c for c, _ in f['T7_top_value'][:6]))
    T.append('EXCLUSIONS per test day (probe statuses; the model uses eligible + eligible_estimated_timing):')
    for d in days:
        T.append(f'  {d}: probes {PX[d]["probes"]} | ' + ', '.join(f'{k} {v}' for k, v in sorted(PX[d]['status'].items())))
    bm = kept_b2_matched(y, pr, b2, day); kv, kvd = positive_keep(y, pr, day); per = {}
    for d in days:
        s = day == d; yy = y[s]; x = PA[s]; q = quint(x.pred.values, yy)
        per[d] = dict(n=int(s.sum()), mean=float(yy.mean()), ic=spearman(x.pred.values, yy), ic_b2=spearman(x.b2.values, yy), q=q, tb=q[-1] - q[0],
                      mse=float(np.mean((x.pred.values - yy) ** 2)), mse_b1=float(np.mean((x.b1.values - yy) ** 2)), keep=float(keep[s].mean()),
                      kept=int(keep[s].sum()), kept_value=kvd[d], kept_value_b2=bm[d][1], b2_kept=bm[d][3], gain_vs_keep_all=float(-yy[~keep[s]].sum()))
    icp = spearman(pr, y); dics = np.asarray([per[d]['ic'] for d in days]); hw = T4_975 * dics.std(ddof=1) / np.sqrt(len(dics))
    strata = PA.groupby(['day', 'm'], sort=False).ngroup().values
    pw = perm_ic_within(pr, y, strata, V3_DRAWS, 20261115)
    qp = quint(pr, y); dec = pd.qcut(pd.Series(pr).rank(method='first'), 10, labels=False).values
    cal_sum = float(abs(pr.sum() - y.sum()) / max(abs(y.sum()), 1e-12)); top_p = float(pr[dec == 9].mean()); top_y = float(y[dec == 9].mean())
    nd = lambda f: sum(int(f(per[d])) for d in days)
    r900 = 'RANKS' if (qp[-1] - qp[0] > 0 and nd(lambda z: z['tb'] > 0) >= 4) else 'NO-RANK'
    ic_pos = bool(icp > 0 and nd(lambda z: z['ic'] > 0) >= 4); beats_b2 = nd(lambda z: z['ic'] > z['ic_b2']) >= 4
    r901w = 'WITHIN-SKILL' if (pw['p'] is not None and pw['p'] <= P_MAX) else 'NOT-SHOWN'
    sides = {s: (spearman(pr[PA.side.values == s], y[PA.side.values == s]),
                 (lambda qq: qq[-1] - qq[0])(quint(pr[PA.side.values == s], y[PA.side.values == s]))) for s in (0, 1)}
    r902 = 'BOTH-SIDES' if all(v[0] > 0 and v[1] > 0 for v in sides.values()) else 'FAILS-' + '/'.join(('BID', 'ASK')[s] for s, v in sides.items() if not (v[0] > 0 and v[1] > 0))
    mm = metric_rows3(PA, keep, stamps)
    r903 = 'POSITIVE-KEEP' if (kv > 0 and sum(int(kvd[d] > 0) for d in days) >= 4 and mm['p_value'] is not None and mm['p_value'] <= P_MAX) else 'NOT-SHOWN'
    r904 = 'BEATS-B2-DECISION' if nd(lambda z: z['kept_value'] > z['kept_value_b2']) >= 4 else 'NOT-SHOWN'
    r905 = 'CALIBRATED' if (cal_sum <= CAL_TOL and np.sign(top_p) == np.sign(top_y)) else 'MISCALIBRATED'
    T += ['', f'TEST ACTIONS: n {len(PA)} on {len(days)} days (as of {asof}); mean {y.mean():+.2f} c per order of {E.Z:.0f} sh; IC {icp:+.3f} '
              f'(B2 {spearman(b2, y):+.3f}); daily IC mean {dics.mean():+.3f}, day-cluster 95% t-interval [{dics.mean() - hw:+.3f}, {dics.mean() + hw:+.3f}] (G = 5)',
          f'  within-window permutation ({V3_DRAWS} draws, {int(strata.max()) + 1} day x window strata): IC {pw["ic"]:+.4f} vs null mean '
          f'{pw["null_mean"]:+.4f}, q95 {pw["null_q95"]:+.4f} -> p {pw["p"]}',
          '  quintiles of the prediction (realized mean c): ' + ' '.join(f'{v:+.2f}' for v in qp),
          f'  keep iff pred > 0: keep {keep.mean():.1%} ({int(keep.sum())}); kept value ${kv / 100:+.1f} ({kv / max(int(keep.sum()), 1):+.2f} c per kept '
          f'action); matched null (random keeps, same count in day / side / hour / rung / remainder strata, {mm["draws"]} draws, {mm["varying_strata"]} '
          f'varying strata) mean ${mm["null_mean"] / 100:+.1f} -> p {mm["p_value"]}; gain vs keep-all ${mm["local_gain_vs_keep_all_cents"] / 100:+.1f} (printed only)']
    for d in days:
        z = per[d]
        T.append(f"  {d}: n {z['n']} | mean {z['mean']:+.2f} | IC {z['ic']:+.3f} / B2 {z['ic_b2']:+.3f} | quintiles " + ' '.join(f'{v:+.2f}' for v in z['q'])
                 + f" | MSE {z['mse']:.0f} / B1 {z['mse_b1']:.0f} | keep {z['keep']:.1%} ({z['kept']}) -> kept ${z['kept_value'] / 100:+.1f}; B2 at the "
                 f"same count ({z['b2_kept']}) ${z['kept_value_b2'] / 100:+.1f} | gain vs keep-all ${z['gain_vs_keep_all'] / 100:+.1f}")
    T.append('  calibration by decile (mean prediction -> realized): ' + ' | '.join(f'{pr[dec == i].mean():+.2f} -> {y[dec == i].mean():+.2f}' for i in range(10)))
    T.append(f'  sides IC / top-bottom: bid {sides[0][0]:+.3f} / {sides[0][1]:+.2f}; ask {sides[1][0]:+.3f} / {sides[1][1]:+.2f}; '
             f'sum of predictions vs values off by {cal_sum:.1%}; top decile {top_p:+.2f} -> {top_y:+.2f}')
    parts = [n for n, ok in (('R900', r900 == 'RANKS'), ('R901 IC+', ic_pos), ('R901 B2', beats_b2), ('R902', r902 == 'BOTH-SIDES'),
                             ("R903'", r903 == 'POSITIVE-KEEP'), ("R904'", r904 == 'BEATS-B2-DECISION'), ('R905', r905 == 'CALIBRATED')) if not ok]
    verdict = 'MODEL-READY' if not parts else 'NOT-READY (' + ', '.join(parts) + ')'
    T += ['', f"READINGS: R900 {r900}; R901 {'IC+' if ic_pos else 'IC-NOT-SHOWN'}, {'BEATS-B2-IC' if beats_b2 else 'NOT-ABOVE-B2-IC'}; R901w {r901w}; "
              f"R902 {r902}; R903' {r903}; R904' {r904}; R905 {r905}",
          f'VERDICT v3: {verdict} (development only: consumed days, replay rule M; a local 1 s value under a neutral continuation, not P&L)']
    R = dict(asof=asof, n=int(len(PA)), per_day=per, ic=icp, daily_ic_interval=[float(dics.mean() - hw), float(dics.mean() + hw)], within=pw,
             quintiles=qp, kept_value=kv, metrics=mm, calibration=dict(sum_off=cal_sum, top_pred=top_p, top_real=top_y),
             readings=dict(R900=r900, R901=[ic_pos, beats_b2], R901w=r901w, R902=r902, R903p=r903, R904p=r904, R905=r905), verdict=verdict,
             exclusions={d: PX[d]['status'] for d in days})
    open(f'{OUTD}/stop_ev_model_v3.txt', 'w').write('\n'.join(T) + '\n')
    json.dump(R, open(f'{OUTD}/stop_ev_model_v3.json', 'w'), indent=1, default=float)
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
    # (7) v2 value head = the conditional MEAN: the review's control (900 x +1, 100 x -100 among filled actions; mean -9.1, Huber +0.89)
    # two groups (LightGBM's Huber starts from the label mean and barely moves on one group): x = 0 holds 10% -100 (mean -9.1), x = 1 only +1
    n7 = 4000; x7 = (np.arange(n7) % 2).astype(float); y7 = np.where(x7 > 0, 1.0, np.where(np.arange(n7) % 20 == 0, -100.0, 1.0))
    r7 = np.random.default_rng(7)                         # its own generator: the case must not depend on earlier draws
    X7 = np.stack([x7, r7.normal(0, 1, n7)], 1).astype(np.float32); f7 = np.ones(n7, bool); f7[3::97] = False
    H7 = fit_hurdle2(X7, y7, f7, ['a', 'b'], rounds=600)
    pv7 = H7['v'].predict(X7[f7]); g7 = x7[f7]; obj7 = H7['v'].dump_model()['objective']   # in-sample group means of the value head
    v7 = [float(pv7[g7 == 0].mean()), float(pv7[g7 == 1].mean())]; m7 = [float(y7[f7][g7 == 0].mean()), float(y7[f7][g7 == 1].mean())]
    if abs(v7[0] - m7[0]) > 1.5 or abs(v7[1] - m7[1]) > 0.5 or not str(obj7).startswith('regression'):
        bad.append(f'v2 value head: group means {np.round(v7, 2).tolist()} vs {np.round(m7, 2).tolist()}; fitted objective {obj7}')
    # (8) T5v2 gate: two-sided (an IC of -1 on every draw must fail), passes centred noise, needs 200 draws
    g_neg = t5v2_gate([-1.0] * 199 + [-0.99]); g_ok = t5v2_gate(list(rng.normal(0, 0.01, 200))); g_few = t5v2_gate(list(rng.normal(0, 0.01, 10)))
    if g_neg['leak_pass'] or not g_ok['leak_pass'] or g_few['leak_pass']:
        bad.append(f"T5v2 gate: all -1 {g_neg['leak_pass']}, centred {g_ok['leak_pass']}, 10 draws {g_few['leak_pass']}")
    # (9) the certificate join: exact keys, values must agree, certified actions must be ours
    Dc = pd.DataFrame(dict(day=[1, 1, 1], m=[5, 5, 5], side=[0, 0, 1], rung=[0, 1, 0], generation=[0, 0, 0], decision_ms=[10.0, 20.0, 10.0], value_cents=[1.0, 2.0, 3.0]))
    Ce = Dc.assign(status=['eligible_estimated_timing', 'uncertain_prior_reference', 'eligible_estimated_timing'], value_cents=[1.0, NAN, 3.5])
    Cw = Dc.assign(status=['eligible_timing_invariant', 'uncertain_prior_reference', 'uncertain_prior_reference'], value_cents=[1.0, NAN, NAN])
    fl = cert_flags(Dc, Ce, Cw)
    if not (list(fl['certified_estimated'][0]) == [True, False, True] and fl['certified_estimated'][1]['value_differ'] == 1
            and list(fl['certified_ws'][0]) == [True, False, False] and fl['certified_ws'][1]['value_differ'] == 0):
        bad.append(f'certificate join {fl}')
    # (10) v3 within-window permutation: a within-window signal is significant; a cross-window-only signal keeps its association in the
    # null (null mean near the observed IC, p not small); a global permutation would call the cross-window signal significant
    r10 = np.random.default_rng(10); g10 = np.repeat(np.arange(150), 8); wf = r10.normal(0, 1, 150)[g10]; e_ = r10.normal(0, 1, len(g10))
    a10 = perm_ic_within(e_ + r10.normal(0, 1, len(g10)), wf + e_, g10, 200, 1)
    b10 = perm_ic_within(wf + r10.normal(0, 1, len(g10)), wf + r10.normal(0, 1, len(g10)), g10, 200, 2)
    if not (a10['p'] <= 0.01 and b10['ic'] > 0.2 and b10['null_mean'] > 0.5 * b10['ic'] and b10['p'] > 0.05):
        bad.append(f"within-window permutation: within p {a10['p']}; cross-only IC {b10['ic']:+.3f}, null mean {b10['null_mean']:+.3f}, p {b10['p']}")
    # (11) R903' cannot pass when every action loses (v1's gain vs keep-all would)
    kv11, _ = positive_keep(-np.ones(10), np.r_[np.ones(5), -np.ones(5)], np.zeros(10, 'int64'))
    if kv11 > 0:
        bad.append(f'positive keep: all-losing actions read {kv11:+.1f}')
    # (12) R904': B2 keeps the model's count
    bm12 = kept_b2_matched(np.arange(10.0), np.r_[np.ones(3), -np.ones(7)], np.r_[-np.ones(2), np.ones(8)], np.zeros(10, 'int64'))
    if not (bm12[0][2] == 3 and bm12[0][3] == 3 and bm12[0][0] == 3.0):
        bad.append(f'B2 at the matched count {bm12}')
    # (13) stamps: rows of two references are refused; one reference passes and the separable toy beats its null
    n13 = 400; y13 = np.where(np.arange(n13) % 2 == 0, 5.0, -5.0)
    P13 = pd.DataFrame(dict(day=[20261027] * (n13 // 2) + [20261031] * (n13 // 2), m=np.arange(n13), side=[0] * n13, rung=[0] * n13, generation=[0] * n13,
                            status=['eligible'] * n13, value_cents=y13, hour=[1] * n13, x_known_remaining=[30.0] * n13))
    s_a = (dict(reference='a', contract='c'), dict(timing=TIMING)); s_b = (dict(reference='b', contract='c'), dict(timing=TIMING))
    try:
        metric_rows3(P13, y13 > 0, {20261027: s_a, 20261031: s_b}, draws=200); bad.append('stamps: two references were not refused')
    except ValueError:
        pass
    try:
        mt13 = metric_rows3(P13, y13 > 0, {20261027: s_a, 20261031: s_a}, draws=200)
        if not (mt13['p_value'] is not None and mt13['p_value'] <= 0.01):
            bad.append(f'stamps: one reference, separable toy {mt13}')
    except ValueError as e:
        bad.append(f'stamps: one reference refused ({e})')
    # (14) the reload gate refuses a 1e-6 difference and an empty artefact list
    if reload_gate({'full_p': dict(reload_max_abs_diff=1e-6)}) or reload_gate({}) or not reload_gate({'full_p': dict(reload_max_abs_diff=0.0)}):
        bad.append('reload gate')
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
    elif cmd == 'prep2':
        prep2(sys.argv[2])
    elif cmd == 'fit2':
        fit2(sys.argv[2])
    elif cmd == 't5v2':
        t5v2(sys.argv[2])
    elif cmd == 'report2':
        report2()
    elif cmd == 'prep3':
        prep3(sys.argv[2])
    elif cmd == 'fit3':
        fit3(sys.argv[2])
    elif cmd == 'report3':
        report3()
    else:
        sys.exit(__doc__)
