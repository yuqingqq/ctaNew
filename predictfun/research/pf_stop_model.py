"""STOP-MODEL (2026-11-14, session aea5bfb9): the keep / cancel model for the ladder's orders - a per-level STOP-PRICE model at every 50 ms step.

USER (2026-11-14): "Proceed to build the keep vs cancel model for multiple ladder orders" / "Build review, then test model performance, make
sure the model is built aligned to our goal, check data distribution, make sure the data is good for training, clear all issues in training,
make sure the model predicts the right thing with good data, then check model's performance".

THE USER'S DESIGN (HANDOFF START HERE 2026-11-14 16:38Z, STATUS direction_20261114; restated per C18): "rest orders there as default, only pull
when real signals happen"; "Pull the risk ones, and keep our protection side at the future best price"; "if we estimate it will stop 1c, we
can rest 49c there and repost 50c ask"; "we need to predict the next stop price and only pull the risky levels". Every ladder level rests by
default (rung 0 included). At every 50 ms step, per side, a STOP-PRICE model predicts from the CURRENT state how far the best will move against
the side; each resting order d ticks behind the known best is cancelled iff it is likely to be RUN OVER (the move goes through its price:
P(M >= d + 1)) and kept otherwise - the order at the predicted stop is the protection side at the new best. No guard trigger: the guard's value
and its age are not inputs. This script builds and checks the MODEL; the policy replay (STOP-POLICY: thresholds, hysteresis, joint lane, caps)
is the next step, on the USER's go.

TARGET (STOP-DEPTH's clean target, now at every step): rows = every 50 ms grid step of the first 60 s, both sides, with a valid known book.
  ref = the side's best in the last frame KNOWN at t (feed >= 13.2 ms, median ~53, + 2 ms decision: K.window_sides' kf / jk).
  D0  = ticks the best went against the side (bids down, asks up) beyond ref over venue frames stamped after the known frame up to t + 22 ms
        (K.L_CANCEL: a cancel sent at t lands at t + 22 ms; these moves cannot be avoided by the decision at t);
  D1  = the same over the state at t + 22 ms and the frames stamped in (t + 22, t + 1000] ms (no new frame = no change).
  For level j = d + 1 (the order d ticks behind ref): status ALREADY_j if D0 >= j (past saving: not a training or test row for j), else
  y_j = [D1 >= j] = the order is run over. M = D1 is the stop: an order at depth M is at the new best. j = 1..6 (rungs 0..5).
INPUTS (82, every one known at t): STOP-DEPTH's NEW inputs without (i) the guard value v (a leave-one-day-out model whose fits include later
  days; C16) and the guard age (steps since the guard came on: derived from v), (ii) 'gap' (reads the window's start price before it is known,
  P16; with it gone, no input needs the start price and the masked steps are rows like any other), plus two level ages (ms since the current
  best price on our / the opposite side started, venue stamps of known frames, capped at 4,000 ms because frames are loaded from ws - 5 s).
FOLDS (C16, ORDER-HARM v3's): test day -> (training days = the days before its calibration day, calibration day = the day before):
  10-27 <- 10-23..25 / 10-26; 10-31 <- ..10-26 / 10-27; 11-01 <- ..10-27 / 10-31; 11-02 <- ..10-31 / 11-01; 11-03 <- ..11-01 / 11-02.
  LightGBM binary per j, PULL-DEPTH's pinned parameters and rounds (no tuning); 60,000 rows drawn per training day (fixed seed), rows
  ALREADY_j dropped per j; predictions made cumulative (P_j = min(P_j, P_j-1), raw values kept for the monotonicity check); the calibration
  day's predictions are saved for STOP-POLICY's thresholds and are not used here except as a diagnostic.
DATA AUDIT ('audit'; decisions from the DEV days 10-23..10-26 only = fold 1's training + calibration days, applied to every fold; the nine
  days are printed as description and never used for a decision): per day and input the missing share, constant flag, quantiles, the PSI of
  each dev day against the other dev days (deciles of the pooled dev days + a missing bin) and the input's own AUC for y_1; DROP an input if on
  any dev day it is > 20% missing or constant, or its PSI >= 0.25 on >= 2 of the 4 dev days; FLAG it if 5-20% missing, PSI 0.10-0.25 on >= 2
  dev days, or its AUC for y_1 is above 0.52 on one dev day and below 0.48 on another. Labels: per day and j the rows, ALREADY share and base
  rate, quiet share; FLAG a j whose dev-day base rates differ by more than 1.5x.
READINGS DECLARED BEFORE THE FIRST RUN (test days = the five fold test days; day rule >= 4 of 5):
 R700 DATA: the audit's DROP / FLAG lists (printed; the drops are applied before any fit).
 R701 TARGET-ALIGNED ('align': the keep-all ladder in the client lane - rungs 0..5 x 30 sh each side, never pulled, no cap - on each test day at
      M 0 / 50 / 250; every 5th step (250 ms), every order client-known resting there, at depth d = 0..5 behind the known best and not
      ALREADY_(d+1); value V = the 1 s markout of its fills in (t + 22, t + 1000] ms, cents per order share): at M 50, mean V in rows with
      y_(d+1) = 1 below mean V in rows with y_(d+1) = 0 pooled AND on >= 4 days, AND mean V | y = 1 < 0 pooled -> ALIGNED; else MISALIGNED.
      Printed: per depth, per trade source (WS / E2, C12) and clock; the share of all negative V in y = 0 rows (harm the target misses) and
      of all positive V in y = 1 rows (good fills it would cancel).
 R702 SKILL (j = 1..6, rows not ALREADY_j): AUC >= 0.70 pooled and >= 0.65 on >= 4 days -> PREDICTABLE, else WEAK; Brier skill against the
      training days' base rate > 0 pooled and on >= 4 days -> BEATS-BASE, else NO-SKILL.
 R703 CALIBRATION (j = 1..3, pooled test days, deciles of P_j): max |mean P - realized rate| <= 0.03 -> CALIBRATED, else MISCALIBRATED.
 R704 VS-GUARD (j = 1..3, rows with a known guard value): AUC of P_j above the AUC of -v on >= 4 days -> BETTER; below on >= 4 -> WORSE;
      else NOT-SHOWN (the guard is leave-one-day-out on these days: an advantaged comparator).
 R705 VS-STOP-DEPTH (j = 1..3, STOP-DEPTH's population: guard-on steps with a known guard value): AUC above STOP-DEPTH NEW's saved
      leave-one-day-out predictions on >= 4 days -> BETTER; below on >= 4 -> WORSE; else NOT-SHOWN (STOP-DEPTH trained with later days).
 R706 KEEP-QUALITY (rung d = 0..5, rows with a known guard value, not ALREADY_(d+1)): the guard keeps the steps with v > -0.10; the model keeps
      the same number with the lowest P_(d+1); the share of kept steps run over (y_(d+1) = 1): model lower on >= 4 days -> FEWER-RUN-OVERS;
      higher on >= 4 -> MORE-RUN-OVERS; else NOT-SHOWN.
 R707 PRICES-HARM ('align', depth 0..5): mean V in the top decile of P_(d+1) (deciles per day and depth) below mean V in the bottom half,
      pooled AND on >= 4 days, at M 50 -> PRICES-HARM; else NOT-SHOWN (M 0 / 250 printed).
 VERDICT: MODEL-READY (a candidate input for STOP-POLICY) iff R701 ALIGNED, R702 PREDICTABLE and BEATS-BASE for j = 1, 2, 3, and R707
      PRICES-HARM; else NOT-READY, naming the failing parts. Development only (C8, C16): consumed days; any use needs a frozen policy and a
      forward test on >= 5 untouched days.
TRAINING CHECKS (printed; T1 / T5 failures refuse the report): T1 chronology (training < calibration < test, asserted; no row of the test or
  calibration day in a fit); T2 sample parity (the training sample's base rate within 0.5 pp of its days' population rate, per j); T3 positives
  per j; T4 the share of rows with raw P_(j+1) > P_j before the cumulative min; T5 permuted-label falsifier (fold 10-27, j = 1, labels
  permuted within day and side: test AUC within 0.5 +- 0.01); T6 AUC on the training sample vs the calibration and test days; T7 top inputs
  per fold.
CONTROLS (any failure -> no output): synthetic selftests with planted errors (target via STOP-DEPTH's; level age reading a later frame; a
  later day in a training set; the cumulative min dropped; horizon sums including fills before t + 22 ms; PSI / AUC helpers incl. NaN rows,
  P17); on every 'rows' day: (a) PARITY - every STOP-DEPTH row (m, side, k) is a row here with all shared inputs and D0 / D1 equal; (b)
  FUTURE-PERTURBATION falsifier on real rows - frames after the known frame and step rows after k randomised: inputs unchanged (0 differ), the
  label changes somewhere (the perturbation reaches the future); (c) INDEPENDENT LABELS - a brute-force recomputation equals D0 / D1 / quiet on
  a sample, and with STOP-DEPTH's planted sign error it differs (the comparison can fail on real data); 'align': lane conservation
  violations 0, terminal reconciliation, every fill matched to exactly one order; memory self-guard 2 GiB.
LIMITS: consumed days (inputs chosen after PULL-DEPTH / STOP-DEPTH on them; exploratory); the target is the venue book's best (snapshots ~50 ms
  apart: a sweep refilled within one snapshot is invisible to it - R701 measures what that misses on our simulated orders); replay rule M for
  'align' (B1a: the queue position is an estimate, C5 / C10); E2 trade anchors +-250 ms (C12); AUC ranks rows, only the policy replay values
  decisions; rows overlap (every 50 ms with a 1 s horizon): per-row statistics describe decision states, never strategy P&L.
Usage (run_light, PF_SLOT=claude): pf_stop_model.py selftest | rows <day> | audit | fit <test_day> | align <test_day> | report;
driver run_stop_model.sh.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_kp_s4 as K                                      # noqa: E402  (load_day, window_sides, clocks)
import pf_qr_queue_race as QR                             # noqa: E402  (runs: valid frames, run starts)
import pf_pull_depth as PD                                # noqa: E402  (feats, pinned LightGBM parameters)
import pf_stop_depth as SD                                # noqa: E402  (target, NEW inputs, saved rows / predictions for parity)
import pf_order_harm_v3 as V3                             # noqa: E402  (chronological folds)
import pf_order_harm_client_lane as CL                    # noqa: E402  (client-known joint lane)
import pf_order_harm_policy as P1                         # noqa: E402  (window_pairs)
import pf_joint_ladder_v3 as J3                           # noqa: E402  (fill clocks, violation names)
import pf_timing_split as TS                              # noqa: E402  (WS / E2 source)
import pf_depth_timing as DT                              # noqa: E402  (STOP / THROUGH)

DAYS9 = list(K.DAYS9)
FOLDS = {d: (tuple(tr), cal) for d, (tr, cal) in V3.FOLDS.items()}
TESTS = tuple(sorted(FOLDS))
DEV_DAYS = tuple(FOLDS[TESTS[0]][0]) + (FOLDS[TESTS[0]][1],)
OUTD = f'{K.IO}/stop_model_20261114'
JS = (1, 2, 3, 4, 5, 6)
H_MS = float(SD.H_MS); L_C = float(K.L_CANCEL)
AGE_CAP = 4000.0
SUB_DAY = 60000; SUB_N = 400000
FZ_EVERY = 25; FZ_STEP = 37; BF_EVERY = 7; BF_STEP = 13
EVERY_ALIGN = 5
CFGS = list(J3.CFGS)
SIZE = float(CL.SIZE); NR = int(CL.NR)
GUARD = float(K.GUARD_PULL); MASK_V = float(SD.MASK_V)
X37 = list(SD.X37)
KEEP_X = [i for i, c in enumerate(X37) if c not in ('gap', 'v')]
PD_KEEP = ['fside', 's1', 's2', 's3', 's4', 's5', 'g2', 'g3', 'g4', 'g5', 'spread', 'c1', 'c2', 'c3']
AGE_F = ['own_lvl_age', 'opp_lvl_age']
FEAT = [X37[i] for i in KEEP_X] + PD_KEEP + list(SD.NEW_EXTRA) + AGE_F
SHARED = [c for c in FEAT if c not in AGE_F]                         # inputs also in STOP-DEPTH's rows
SD_NAME = {**{X37[i]: f'x{i}' for i in KEEP_X}, **{c: c for c in PD_KEEP + list(SD.NEW_EXTRA)}}
KEYS = ['day', 'm', 'side', 'k']
LAB = ['D0', 'D1', 'quiet']
MISS_DROP, MISS_FLAG, PSI_DROP, PSI_FLAG, PSI_DAYS, AUC_FLIP, RATE_RATIO = 0.20, 0.05, 0.25, 0.10, 2, 0.02, 1.5
AUC_POOL, AUC_DAY, DAY_RULE, CAL_TOL, T2_TOL, T5_TOL = 0.70, 0.65, 4, 0.03, 0.005, 0.01
MEM_CAP_MIB = 2048.0
INF = float('inf'); NAN = float('nan')
MUT_AGE_NEXT = MUT_FOLD_LATE = MUT_NO_CUMMIN = MUT_HORIZON_EARLY = MUT_PSI_NONAN = MUT_AUC_KEEPNAN = MUT_RULE_OFF = 0   # planted (selftest)


# ---------------------------------------------------------------- inputs and target
def run_starts(px, valid):
    """Per frame: index of the first frame of the current run of valid frames with the same price (QR.runs' rule, any price series)."""
    px = np.asarray(px); n = len(px); same = np.zeros(n, bool)
    if n > 1:
        same[1:] = valid[1:] & valid[:-1] & (px[1:] == px[:-1])
    idx = np.arange(n)
    return np.maximum.accumulate(np.where(~same, idx, 0)) if n else idx


def level_ages(tf, jk, rs, rso):
    """ms since the current best on our side / the opposite side started (venue stamps of known frames), capped at AGE_CAP."""
    j = min(jk + 1, len(tf) - 1) if MUT_AGE_NEXT else jk
    return min(AGE_CAP, float(tf[j] - tf[rs[j]])), min(AGE_CAP, float(tf[j] - tf[rso[j]]))


def step_feats(w, k, side, rs, rso, nanP):
    """The 82 inputs of step k (FEAT order), from the step's known frame and step row only."""
    jk = int(w['jk'][k])
    fd = dict(zip(SD.PDF, PD.feats(w, k, nanP, 0)))          # PULL-DEPTH's 53 (P = NaN and the guard age are not used)
    sz = [fd[f's{i}'] for i in range(1, 6)]; gaps = [0.0] + [fd[f'g{i}'] for i in range(2, 6)]
    ex = SD.extra_feats(np.asarray(w['X'][k], float), side, sz, gaps) + list(SD.best_moves(w, k, side))
    return [fd[f'x{i}'] for i in KEEP_X] + [fd[c] for c in PD_KEEP] + ex + list(level_ages(w['tf'], jk, rs, rso))


def label_brute(tf, b0, valid, side, jk, t, lc=None, h=None):
    """Independent recomputation of (D0, D1, quiet) by an explicit frame walk (control (c); not used to build labels)."""
    lc = L_C if lc is None else lc; h = H_MS if h is None else h
    ref = int(b0[jk]); n = len(tf)

    def adv(p):
        return (ref - int(p)) if side == 0 else (int(p) - ref)
    D0 = 0; last = ref; i = jk + 1
    while i < n and tf[i] <= t + lc:
        if valid[i]:
            D0 = max(D0, adv(b0[i])); last = b0[i]
        i += 1
    D1 = max(0, adv(last)); quiet = True
    while i < n and tf[i] <= t + h:
        if valid[i]:
            D1 = max(D1, adv(b0[i])); quiet = False
        i += 1
    return float(D0), float(D1), bool(quiet)


def window_rows(w, st):
    """Rows of one window-side: keys, inputs (float32), labels (D0, D1, quiet), guard value (comparison only), known best."""
    n = w['n']
    if not n:
        return None
    side = int(w['side'])
    b0, valid, rs, _ = QR.runs(w['tf'], w['P5'], w['opp'], side)
    rso = run_starts(np.asarray(w['opp'], 'int64'), valid)
    tfl = w['tf'].tolist(); vv = np.asarray(w['vkl'], float); nanP = np.full(n, np.nan)
    ks, X, Y = [], [], []
    for k in range(n):
        st['steps'] += 1
        jk = int(w['jk'][k])
        if jk < 0 or not valid[jk] or w['best'][k] < 0:
            st['invalid_book'] += 1; continue
        t = float(w['tg'][k])
        X.append(step_feats(w, k, side, rs, rso, nanP))
        D0, D1, q = SD.move_clean(b0, valid, tfl, side, jk, t)
        Y.append((D0, D1, float(q))); ks.append(k)
        st['quiet'] += int(q); st['masked'] += int(vv[k] == MASK_V)
    if not ks:
        return None
    ks = np.asarray(ks, 'int64')
    return dict(m=np.full(len(ks), int(w['m']), 'int64'), side=np.full(len(ks), side, 'int64'), k=ks,
                t=np.asarray(w['tg'], float)[ks], v=vv[ks].astype(np.float32), best=np.asarray(w['best'], 'int64')[ks],
                X=np.asarray(X, np.float64).astype(np.float32), Y=np.asarray(Y, np.float32), ctx=(b0, valid))


def perturb_future(w, jk, k, rng):
    """A copy of the window-side with every frame after the known frame jk and every step row after k randomised."""
    w2 = dict(w)
    P5 = np.array(w['P5'], copy=True); S5 = np.array(w['S5'], dtype=float, copy=True); opp = np.array(w['opp'], copy=True)
    X = np.array(w['X'], dtype=float, copy=True); a = slice(jk + 1, None)
    P5[a] = np.where(P5[a] >= 0, np.clip(P5[a] + rng.integers(-3, 4, P5[a].shape), 1, 99), P5[a])
    S5[a] = S5[a] * rng.uniform(0.2, 3.0, S5[a].shape)
    opp[a] = np.where(opp[a] >= 0, np.clip(opp[a] + rng.integers(-3, 4, opp[a].shape), 1, 99), opp[a])
    X[k + 1:] = X[k + 1:] * rng.uniform(0.5, 1.5, X[k + 1:].shape) + rng.normal(0.0, 1.0, X[k + 1:].shape)
    w2.update(P5=P5, S5=S5, opp=opp, X=X, cache={})
    return w2


def falsify_window(w, out, rng, fz, step=FZ_STEP):
    """Control (b): inputs of sampled rows recomputed on a future-perturbed copy must not change; the label should."""
    side = int(w['side']); n = w['n']; tfl = w['tf'].tolist()
    for r_i in range(0, len(out['k']), step):
        k = int(out['k'][r_i]); jk = int(w['jk'][k]); t = float(w['tg'][k])
        w2 = perturb_future(w, jk, k, rng)
        b2, v2, rs2, _ = QR.runs(w2['tf'], w2['P5'], w2['opp'], side)
        rso2 = run_starts(np.asarray(w2['opp'], 'int64'), v2)
        x2 = np.asarray(step_feats(w2, k, side, rs2, rso2, np.full(n, np.nan)), np.float64).astype(np.float32)
        x1 = out['X'][r_i]
        same = (x1 == x2) | (np.isnan(x1) & np.isnan(x2))
        fz['checked'] += 1
        if not same.all():
            fz['feat_differ'] += 1
            fz['cols'] = sorted(set(fz['cols']) | {FEAT[i] for i in np.where(~same)[0]})
        D0, D1, _ = SD.move_clean(b2, v2, tfl, side, jk, t)
        fz['label_differ'] += int((D0, D1) != (float(out['Y'][r_i, 0]), float(out['Y'][r_i, 1])))


def brute_window(w, out, bf, step=BF_STEP):
    """Control (c): brute-force labels equal the stored ones; STOP-DEPTH's planted sign error must differ from them somewhere."""
    side = int(w['side']); tf = np.asarray(w['tf'], float); tfl = tf.tolist(); b0, valid = out['ctx']
    for r_i in range(0, len(out['k']), step):
        k = int(out['k'][r_i]); jk = int(w['jk'][k]); t = float(w['tg'][k])
        D0b, D1b, qb = label_brute(tf, b0, valid, side, jk, t)
        y = out['Y'][r_i]
        bf['checked'] += 1
        bf['differ'] += int((D0b, D1b, float(qb)) != (float(y[0]), float(y[1]), float(y[2])))
        SD.MUT_SIGN = 1
        try:
            D0m, D1m, _ = SD.move_clean(b0, valid, tfl, side, jk, t)
        finally:
            SD.MUT_SIGN = 0
        bf['planted_differ'] += int((D0m, D1m) != (D0b, D1b))


# ---------------------------------------------------------------- statistics helpers
def auc_nan(y, s):
    """Rank AUC with non-finite scores dropped (P17); returns (auc, rows used)."""
    y = np.asarray(y, bool); s = np.asarray(s, float)
    ok = np.ones(len(s), bool) if MUT_AUC_KEEPNAN else np.isfinite(s)
    y = y[ok]; s = s[ok]; n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return NAN, int(ok.sum())
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)), int(ok.sum())


def hist_counts(x, edges):
    """Counts over the bins defined by edges (len(edges) + 1 finite bins) plus a last bin for non-finite values."""
    x = np.asarray(x, float); fin = np.isfinite(x)
    c = np.bincount(np.searchsorted(edges, x[fin], side='right'), minlength=len(edges) + 1).astype(float)
    return np.concatenate([c, [0.0 if MUT_PSI_NONAN else float((~fin).sum())]])


def psi(a, b, eps=1e-4):
    """Population stability index between two count vectors (same bins)."""
    p = np.asarray(a, float); q = np.asarray(b, float)
    if p.sum() <= 0 or q.sum() <= 0:
        return NAN
    p = p / p.sum() + eps; q = q / q.sum() + eps; p /= p.sum(); q /= q.sum()
    return float(((p - q) * np.log(p / q)).sum())


def cummin_pred(P):
    """P (rows x j) made cumulative across j: P_j = min(P_j, P_j-1)."""
    return np.array(P, copy=True) if MUT_NO_CUMMIN else np.minimum.accumulate(np.asarray(P, float), axis=1)


def train_days(test):
    tr = tuple(FOLDS[test][0])
    return tr + (test,) if MUT_FOLD_LATE else tr


def chrono_ok(tr, cal, test):
    return len(tr) > 0 and all(d < cal for d in tr) and cal < test and test not in tr and cal not in tr


def day_rule(a, b, lower_is_better=False):
    """Counts of days where a beats b (and the reverse) for a per-day list; NaN days count for neither."""
    a = np.asarray(a, float); b = np.asarray(b, float); ok = np.isfinite(a) & np.isfinite(b)
    if MUT_RULE_OFF:
        return int(ok.sum()), 0
    better = (a < b) if lower_is_better else (a > b)
    worse = (a > b) if lower_is_better else (a < b)
    return int((better & ok).sum()), int((worse & ok).sum())


def fill_type(side, pt, thr, prices):
    """STOP: our price is the taker order's worst on this side; THROUGH: it went past (POLICY v2's rule)."""
    if thr:
        return 'THROUGH'
    return DT.trade_type(side, pt, prices)


def horizon_sums(ft, cols, t):
    """For fill times ft (sorted) and per-fill columns, the sums over fills executed in (t + L_C, t + H_MS] for each decision time t."""
    ft = np.asarray(ft, float); t = np.asarray(t, float)
    lo = np.searchsorted(ft, t if MUT_HORIZON_EARLY else t + L_C, side='right'); hi = np.searchsorted(ft, t + H_MS, side='right')
    out = []
    for c in cols:
        cs = np.concatenate([[0.0], np.cumsum(np.asarray(c, float))])
        out.append(cs[hi] - cs[lo])
    return out


# ---------------------------------------------------------------- self-tests
def synth_window(rng, n=60, nf=200):
    """A synthetic window-side with every field step_feats / falsify_window read."""
    tf = np.cumsum(rng.uniform(20.0, 80.0, nf)); kf = tf + 60.0
    b = 50 + np.cumsum(rng.integers(-1, 2, nf)); b = np.clip(b, 10, 80)
    P5 = np.stack([b - i for i in range(5)], 1).astype('int64'); S5 = rng.uniform(1.0, 200.0, (nf, 5)); opp = (b + rng.integers(1, 3, nf)).astype('int64')
    tg = np.linspace(tf[5] + 100.0, tf[-40], n); jk = np.searchsorted(kf, tg, side='right') - 1
    X = rng.normal(0.0, 1.0, (n, 37))
    return dict(m=1, side=0, tf=tf, kf=kf, tg=tg, jk=jk, n=n, P5=P5, S5=S5, opp=opp, best=[int(P5[j, 0]) if j >= 0 else -1 for j in jk],
                X=X, vkl=[0.0] * n)


def selftest():
    bad = []

    def chk(nm, c):
        if not c:
            bad.append(nm)
    chk('feature list', len(FEAT) == 82 and len(set(FEAT)) == 82 and not {'v', 'gap', 'gage', 'P'} & set(FEAT)
        and len(SHARED) == 80 and all(c in SD.NEW_FEAT or c in X37 for c in SHARED))
    chk('sd names', all(SD_NAME[c] in SD.NEW_FEAT for c in SHARED))
    # level ages
    tf = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 5000.0]); valid = np.ones(6, bool)
    rs = run_starts(np.array([50, 50, 49, 49, 49, 48]), valid); rso = run_starts(np.array([52, 52, 52, 51, 51, 51]), valid)
    chk(f'level ages {level_ages(tf, 4, rs, rso)}', level_ages(tf, 4, rs, rso) == (200.0, 100.0))
    chk('age cap', level_ages(np.array([0.0, 9000.0]), 1, np.array([0, 0]), np.array([0, 1])) == (AGE_CAP, 0.0))
    v2 = valid.copy(); v2[3] = False
    chk('run broken by an invalid frame', list(run_starts(np.array([50, 50, 49, 49, 49, 48]), v2)) == [0, 0, 2, 3, 4, 5])
    # labels: brute force equals STOP-DEPTH's move_clean on random sequences (both sides, invalid frames, quiet books)
    rng = np.random.default_rng(5); nd = 0
    for _ in range(400):
        nf = int(rng.integers(3, 40)); tfr = np.sort(rng.uniform(0.0, 2000.0, nf)); b0 = 50 + rng.integers(-4, 5, nf)
        val = rng.uniform(0, 1, nf) > 0.15; jk = int(rng.integers(0, nf)); val[jk] = True; side = int(rng.integers(0, 2))
        t = float(tfr[jk] + rng.uniform(13.2, 300.0))
        a = SD.move_clean(b0, val, tfr.tolist(), side, jk, t); b_ = label_brute(tfr, b0, val, side, jk, t)
        nd += int(tuple(map(float, a[:2])) + (bool(a[2]),) != b_)
    chk(f'brute force = move_clean ({nd} differ)', nd == 0)
    # inputs: a synthetic window - the future perturbation leaves every input unchanged and changes some label
    w = synth_window(np.random.default_rng(11)); out = window_rows(w, dict(steps=0, invalid_book=0, quiet=0, masked=0))
    fz = dict(checked=0, feat_differ=0, label_differ=0, cols=[])
    falsify_window(w, out, np.random.default_rng(3), fz, step=3)
    chk(f'falsifier clean {fz}', fz['checked'] > 5 and fz['feat_differ'] == 0 and fz['label_differ'] > 0)
    bf = dict(checked=0, differ=0, planted_differ=0); brute_window(w, out, bf, step=2)
    chk(f'brute window {bf}', bf['checked'] > 5 and bf['differ'] == 0 and bf['planted_differ'] > 0)
    # folds
    chk('folds chronological', all(chrono_ok(train_days(d), FOLDS[d][1], d) for d in TESTS) and len(TESTS) == 5
        and DEV_DAYS == ('20261023', '20261024', '20261025', '20261026'))
    # cumulative predictions
    P = cummin_pred(np.array([[0.5, 0.6, 0.2, 0.3, 0.1, 0.05]]))
    chk(f'cummin {P}', np.allclose(P, [[0.5, 0.5, 0.2, 0.2, 0.1, 0.05]]))
    # PSI / AUC helpers
    e = np.array([1.0, 2.0, 3.0]); xa = np.array([0.5, 1.5, 2.5, 3.5] * 50)
    chk('psi same', abs(psi(hist_counts(xa, e), hist_counts(xa, e))) < 1e-12)
    chk('psi shift', psi(hist_counts(xa, e), hist_counts(xa + 2.0, e)) > PSI_DROP)
    xm = xa.copy(); xm[:100] = np.nan
    chk('psi sees missingness', psi(hist_counts(xa, e), hist_counts(xm, e)) > PSI_FLAG)
    yy = np.array([0, 0, 1, 1, 0, 1], bool); ss = np.array([0.1, 0.2, 0.8, 0.9, np.nan, np.nan])
    a_, n_ = auc_nan(yy, ss)
    chk(f'auc with NaN rows {a_} {n_}', a_ == 1.0 and n_ == 4)
    # day rule
    chk('day rule', day_rule([1, 2, 3, 4, 5], [0, 0, 0, 0, 9]) == (4, 1) and day_rule([1, 1], [2, 2], lower_is_better=True) == (2, 0))
    # horizon sums: fills at decision + 10 (unavoidable), + 500 (counted), + 1500 (outside)
    q_, v_ = horizon_sums([110.0, 600.0, 1600.0], [[1.0, 2.0, 4.0], [-1.0, -2.0, -4.0]], [100.0])
    chk(f'horizon sums {q_} {v_}', float(q_[0]) == 2.0 and float(v_[0]) == -2.0)
    chk('fill type', fill_type(0, 49, False, [51, 50, 49]) == 'STOP' and fill_type(0, 50, False, [51, 50, 49]) == 'THROUGH'
        and fill_type(1, 52, True, [52]) == 'THROUGH')
    return bad


def mutation_check():
    out = {}
    for nm in ('MUT_AGE_NEXT', 'MUT_FOLD_LATE', 'MUT_NO_CUMMIN', 'MUT_HORIZON_EARLY', 'MUT_PSI_NONAN', 'MUT_AUC_KEEPNAN', 'MUT_RULE_OFF'):
        globals()[nm] = 1
        try:
            out[nm] = len(selftest()) > 0
        finally:
            globals()[nm] = 0
    return out


def all_selftests():
    SD.all_selftests()
    bad = selftest()
    if bad:
        sys.exit(f'STOP-MODEL SELFTEST FAILED: {bad}')
    m = mutation_check()
    if not all(m.values()):
        sys.exit(f'STOP-MODEL PLANTED ERROR NOT CAUGHT: {m}')
    print(f'stop-model selftests passed; planted errors caught: {m}', flush=True)
    return m


def mem_guard(what):
    r = K.rss_mib()
    if r > MEM_CAP_MIB:
        sys.exit(f'REFUSED: resident memory {r:.0f} MiB > {MEM_CAP_MIB:.0f} MiB ({what})')


# ---------------------------------------------------------------- rows (one day)
def parity(day, R):
    """Control (a): every STOP-DEPTH row of the day is a row here with equal shared inputs and labels."""
    S = pd.read_parquet(f'{SD.OUTD}/rows_{day}.parquet', columns=['m', 'side', 'k'] + [SD_NAME[c] for c in SHARED] + ['D0', 'D1'])
    S.columns = ['m', 'side', 'k'] + [f'sd_{c}' for c in SHARED] + ['sd_D0', 'sd_D1']
    M = S.merge(R[['m', 'side', 'k'] + SHARED + ['D0', 'D1']], on=['m', 'side', 'k'], how='left', validate='one_to_one', indicator=True)
    pc = dict(sd_rows=int(len(S)), missing=int((M['_merge'] != 'both').sum()), differ={})
    for c in SHARED + ['D0', 'D1']:
        a = M[f'sd_{c}'].to_numpy(np.float32); b = M[c].to_numpy(np.float32)
        nd = int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())
        if nd:
            pc['differ'][c] = nd
    pc['ok'] = pc['sd_rows'] > 0 and pc['missing'] == 0 and not pc['differ']
    return pc


def rows_day(day):
    all_selftests()
    if day not in DAYS9:
        sys.exit(f'REFUSED: {day} is not a consumed replay day')
    T0 = time.time(); D = K.load_day(day)
    st = dict(window_sides=0, steps=0, invalid_book=0, masked=0, quiet=0)
    fz = dict(checked=0, feat_differ=0, label_differ=0, cols=[]); bf = dict(checked=0, differ=0, planted_differ=0)
    rng = np.random.default_rng(int(day)); parts = []
    for w in K.window_sides(D):
        st['window_sides'] += 1
        out = window_rows(w, st)
        if out is None:
            continue
        if st['window_sides'] % FZ_EVERY == 1:
            falsify_window(w, out, rng, fz)
        if st['window_sides'] % BF_EVERY == 0:
            brute_window(w, out, bf)
        out.pop('ctx'); parts.append(out)
        if st['window_sides'] % 80 == 0:
            mem_guard(f'rows {day}')
    R = pd.DataFrame({'day': np.full(sum(len(p['k']) for p in parts), int(day), 'int64'),
                      **{c: np.concatenate([p[c] for p in parts]) for c in ('m', 'side', 'k', 't', 'v', 'best')}})
    X = np.vstack([p['X'] for p in parts]); Y = np.vstack([p['Y'] for p in parts]); del parts
    R = pd.concat([R, pd.DataFrame(X, columns=FEAT), pd.DataFrame(Y, columns=LAB)], axis=1); del X, Y
    R['masked'] = (R.v.values == MASK_V)
    mem_guard(f'rows {day} frame')
    pc = parity(day, R)
    st.update(rows=int(len(R)), load_stat=D['stat'], falsifier=fz, brute=bf, parity=pc, seconds=round(time.time() - T0, 1))
    fails = []
    if not pc['ok']:
        fails.append(f'parity {pc}')
    if fz['checked'] < 50 or fz['feat_differ'] or fz['label_differ'] == 0:
        fails.append(f'falsifier {fz}')
    if bf['checked'] < 500 or bf['differ'] or bf['planted_differ'] == 0:
        fails.append(f'brute {bf}')
    if fails:
        sys.exit(f'REFUSED (no output for {day}): {fails}')
    os.makedirs(OUTD, exist_ok=True)
    R.to_parquet(f'{OUTD}/rows_{day}.parquet', index=False)
    json.dump(st, open(f'{OUTD}/rows_{day}.json', 'w'), indent=1, default=str)
    print(f'{day}: rows {len(R)}; invalid {st["invalid_book"]}; quiet {st["quiet"]}; masked {st["masked"]}; parity {pc["sd_rows"]} rows ok; '
          f'falsifier {fz}; brute {bf}; {time.time() - T0:.0f} s', flush=True)


# ---------------------------------------------------------------- data audit (decisions on the dev days only)
def label_stats(D0, D1):
    out = {}
    for j in JS:
        ok = D0 < j; n = int(ok.sum())
        out[f'j{j}'] = dict(rows=n, already=float(1.0 - ok.mean()), rate=float((D1[ok] >= j).mean()) if n else NAN)
    z = D0 == 0; d = np.minimum(D1[z], 6)
    out['stop_dist_D0_0'] = [float((d == i).mean()) for i in range(7)]
    return out


def audit():
    all_selftests()
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    rj = {d: json.load(open(f'{OUTD}/rows_{d}.json')) for d in DAYS9}
    sub = []
    for d in DEV_DAYS:
        R = pd.read_parquet(f'{OUTD}/rows_{d}.parquet', columns=FEAT)
        ix = np.sort(np.random.default_rng(int(d)).choice(len(R), min(len(R), 100000), replace=False))
        sub.append(R.iloc[ix].to_numpy(np.float64)); del R
    sub = np.vstack(sub)
    edges = {}
    for i, c in enumerate(FEAT):
        x = sub[:, i]; x = x[np.isfinite(x)]
        edges[c] = np.unique(np.quantile(x, np.linspace(0.1, 0.9, 9))) if len(x) else np.zeros(0)
    del sub
    per = {}; H = {}
    for d in DAYS9:
        R = pd.read_parquet(f'{OUTD}/rows_{d}.parquet', columns=FEAT + ['side', 'D0', 'D1', 'quiet', 'masked', 'best'])
        D0 = R.D0.to_numpy(np.float64); D1 = R.D1.to_numpy(np.float64)
        rng = np.random.default_rng(int(d) + 1); ok1 = np.where(D0 < 1)[0]
        si = np.sort(rng.choice(ok1, min(len(ok1), 150000), replace=False)); y1 = D1[si] >= 1
        best = R.best.to_numpy(); side = R.side.to_numpy()
        pd_ = dict(rows=int(len(R)), invalid_book=int(rj[d]['invalid_book']), steps=int(rj[d]['steps']), quiet=float(R.quiet.mean()),
                   masked=float(R.masked.mean()), extreme=float(((best <= 6) | (best >= 94)).mean()),
                   labels=label_stats(D0, D1), labels_by_side={s: label_stats(D0[side == s], D1[side == s]) for s in (0, 1)}, feat={})
        H[d] = {}
        for c in FEAT:
            x = R[c].to_numpy(np.float64); fin = np.isfinite(x); xf = x[fin]
            q = np.quantile(xf, [0.01, 0.5, 0.99]).tolist() if len(xf) else [NAN] * 3
            a, na = auc_nan(y1, x[si])
            pd_['feat'][c] = dict(missing=float(1.0 - fin.mean()), constant=bool(len(xf) == 0 or float(np.ptp(xf)) == 0.0), p01=q[0], p50=q[1],
                                 p99=q[2], mean=float(xf.mean()) if len(xf) else NAN, auc_y1=a, auc_rows=na)
            H[d][c] = hist_counts(x, edges[c])
        per[d] = pd_; del R
        mem_guard(f'audit {d}')
    # decisions (dev days only)
    drop, flag, why = [], [], {}
    psi_dev = {c: {d: psi(H[d][c], sum(H[e][c] for e in DEV_DAYS if e != d)) for d in DEV_DAYS} for c in FEAT}
    for c in FEAT:
        mi = [per[d]['feat'][c]['missing'] for d in DEV_DAYS]; co = [per[d]['feat'][c]['constant'] for d in DEV_DAYS]
        ps = [psi_dev[c][d] for d in DEV_DAYS]; au = [per[d]['feat'][c]['auc_y1'] for d in DEV_DAYS]
        r_ = []
        if max(mi) > MISS_DROP:
            r_.append(f'missing {max(mi):.1%}')
        if any(co):
            r_.append('constant on a dev day')
        if sum(p >= PSI_DROP for p in ps) >= PSI_DAYS:
            r_.append(f'PSI >= {PSI_DROP} on {sum(p >= PSI_DROP for p in ps)} dev days ({", ".join(f"{p:.2f}" for p in ps)})')
        if r_:
            drop.append(c); why[c] = 'DROP: ' + '; '.join(r_); continue
        f_ = []
        if max(mi) > MISS_FLAG:
            f_.append(f'missing {max(mi):.1%}')
        if sum(PSI_FLAG <= p < PSI_DROP for p in ps) >= PSI_DAYS:
            f_.append(f'PSI {", ".join(f"{p:.2f}" for p in ps)}')
        auf = [a for a in au if np.isfinite(a)]
        if auf and max(auf) > 0.5 + AUC_FLIP and min(auf) < 0.5 - AUC_FLIP:
            f_.append(f'AUC for y1 flips {min(auf):.3f}..{max(auf):.3f}')
        if f_:
            flag.append(c); why[c] = 'FLAG: ' + '; '.join(f_)
    lab_flag = {}
    for j in JS:
        rr = [per[d]['labels'][f'j{j}']['rate'] for d in DEV_DAYS]
        if min(rr) > 0 and max(rr) / min(rr) > RATE_RATIO:
            lab_flag[f'j{j}'] = rr
    psi_fold = {t: {c: dict(train_vs_test=psi(sum(H[d][c] for d in FOLDS[t][0]), H[t][c]),
                            train_vs_cal=psi(sum(H[d][c] for d in FOLDS[t][0]), H[FOLDS[t][1]][c])) for c in FEAT} for t in TESTS}
    res = dict(asof=asof, dev_days=list(DEV_DAYS), drop=drop, flag=flag, why=why, label_flags=lab_flag, per_day=per, psi_dev=psi_dev,
               psi_fold=psi_fold, edges={c: e.tolist() for c, e in edges.items()},
               rules=dict(MISS_DROP=MISS_DROP, MISS_FLAG=MISS_FLAG, PSI_DROP=PSI_DROP, PSI_FLAG=PSI_FLAG, PSI_DAYS=PSI_DAYS, AUC_FLIP=AUC_FLIP,
                          RATE_RATIO=RATE_RATIO))
    json.dump(res, open(f'{OUTD}/audit.json', 'w'), indent=1, default=float)
    L = [f'STOP-MODEL DATA AUDIT as of {asof}; decisions from the dev days {", ".join(DEV_DAYS)} only (the other days are description)', '',
         'ROWS AND LABELS per day (rows = valid known book; ALREADY_j = the move reached j before a cancel lands; rate = P(y_j) among the rest)']
    L.append('  day       rows    invalid  quiet  masked extreme | ' + ' | '.join(f'j{j} already / rate' for j in JS))
    for d in DAYS9:
        p = per[d]; lb = p['labels']
        L.append(f'  {d} {p["rows"]:8d} {p["invalid_book"]:8d} {p["quiet"]:6.2%} {p["masked"]:6.2%} {p["extreme"]:6.2%} | '
                 + ' | '.join(f'{lb[f"j{j}"]["already"]:6.2%} / {lb[f"j{j}"]["rate"]:6.2%}' for j in JS) + ('   (dev)' if d in DEV_DAYS else ''))
    L.append('  stop (D1 among D0 = 0 rows) 0 / 1 / 2 / 3 / 4 / 5 / 6+ ticks:')
    for d in DAYS9:
        L.append(f'    {d} ' + ' '.join(f'{x:6.2%}' for x in per[d]['labels']['stop_dist_D0_0']))
    L += ['', f'LABEL FLAGS (dev-day base rate ratio > {RATE_RATIO}): {lab_flag if lab_flag else "none"}', '',
          f'R700 DATA: DROP {len(drop)} {drop}; FLAG {len(flag)} {flag}']
    for c in drop + flag:
        L.append(f'  {c:18s} {why[c]}')
    L += ['', 'INPUTS per day: missing share | median | AUC for y1 (dev days decide; PSI of each dev day vs the other dev days)']
    for c in FEAT:
        L.append(f'  {c:18s} ' + ' '.join(f'{per[d]["feat"][c]["missing"]:5.1%}|{per[d]["feat"][c]["p50"]:9.3g}|{per[d]["feat"][c]["auc_y1"]:.3f}'
                                         for d in DAYS9) + '  PSI dev ' + ' '.join(f'{psi_dev[c][d]:.2f}' for d in DEV_DAYS))
    L += ['', 'PSI by fold (training days vs the test day; description only), top 12 per fold:']
    for t in TESTS:
        top = sorted(FEAT, key=lambda c: -np.nan_to_num(psi_fold[t][c]['train_vs_test']))[:12]
        L.append(f'  {t}: ' + ', '.join(f'{c} {psi_fold[t][c]["train_vs_test"]:.2f}' for c in top))
    open(f'{OUTD}/audit.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L[:40]), flush=True)


# ---------------------------------------------------------------- fit (one chronological fold)
def fit_fold(test):
    import lightgbm as lgb
    all_selftests()
    if test not in TESTS:
        sys.exit(f'REFUSED: {test} is not a fold test day')
    T0 = time.time(); tr_days = train_days(test); cal = FOLDS[test][1]
    if not chrono_ok(tr_days, cal, test):
        sys.exit(f'REFUSED: fold not chronological {tr_days} / {cal} / {test}')
    A = json.load(open(f'{OUTD}/audit.json')); F = [c for c in FEAT if c not in set(A['drop'])]
    parts = []
    for d in tr_days:
        R = pd.read_parquet(f'{OUTD}/rows_{d}.parquet', columns=KEYS + F + ['D0', 'D1'])
        ix = np.sort(np.random.default_rng(int(test) * 31 + int(d)).choice(len(R), min(len(R), SUB_DAY), replace=False))
        parts.append(R.iloc[ix].reset_index(drop=True)); del R
    tr = pd.concat(parts, ignore_index=True); del parts
    if tr.day.isin([int(cal), int(test)]).any():
        sys.exit('REFUSED: a calibration or test row in the training rows')
    ev = {dd: pd.read_parquet(f'{OUTD}/rows_{dd}.parquet', columns=KEYS + F + ['D0', 'D1']) for dd in (cal, test)}
    mem_guard(f'fit {test} load')
    pop = {j: (sum(A['per_day'][d]['labels'][f'j{j}']['rows'] * A['per_day'][d]['labels'][f'j{j}']['rate'] for d in tr_days)
               / max(1, sum(A['per_day'][d]['labels'][f'j{j}']['rows'] for d in tr_days))) for j in JS}
    raw = {dd: np.full((len(ev[dd]), len(JS)), np.nan) for dd in ev}
    info = {}; os.makedirs(f'{OUTD}/models', exist_ok=True)
    for ji, j in enumerate(JS):
        ok = tr.D0.values < j; trj = tr[ok]
        if len(trj) > SUB_N:
            trj = trj.iloc[np.sort(np.random.default_rng(int(test) * 7 + j).choice(len(trj), SUB_N, replace=False))]
        y = (trj.D1.values >= j).astype(int); Xtr = trj[F].to_numpy(np.float32)
        b = lgb.train(PD.LGB_PARAMS, lgb.Dataset(Xtr, y, feature_name=F), PD.N_ROUNDS)
        b.save_model(f'{OUTD}/models/fold{test}_j{j}.txt')
        a_tr, _ = auc_nan(y.astype(bool), b.predict(Xtr)); del Xtr
        inf = dict(train_rows=int(len(trj)), positives=int(y.sum()), base_rate=float(y.mean()), population_rate=float(pop[j]),
                   T2_gap=float(y.mean() - pop[j]), auc_train=a_tr)
        for dd in ev:
            raw[dd][:, ji] = b.predict(ev[dd][F].to_numpy(np.float32))
            okd = ev[dd].D0.values < j
            inf[f'auc_{"cal" if dd == cal else "test"}'] = auc_nan(ev[dd].D1.values[okd] >= j, raw[dd][okd, ji])[0]
        imp = b.feature_importance('gain'); inf['top_gain'] = [(F[i], float(imp[i] / imp.sum())) for i in np.argsort(-imp)[:10]]
        info[f'j{j}'] = inf
        mem_guard(f'fit {test} j{j}')
    T5 = None
    if test == TESTS[0]:                                   # permuted-label falsifier (j = 1)
        ok = tr.D0.values < 1; trj = tr[ok].reset_index(drop=True); y = (trj.D1.values >= 1).astype(int); yp = y.copy()
        rng = np.random.default_rng(99)
        for _, ix in trj.groupby(['day', 'side']).indices.items():
            yp[ix] = rng.permutation(y[ix])
        bp = lgb.train(PD.LGB_PARAMS, lgb.Dataset(trj[F].to_numpy(np.float32), yp, feature_name=F), PD.N_ROUNDS)
        okd = ev[test].D0.values < 1
        a_p, n_p = auc_nan(ev[test].D1.values[okd] >= 1, bp.predict(ev[test][F].to_numpy(np.float32)[okd]))
        T5 = dict(auc=a_p, rows=n_p, ok=bool(abs(a_p - 0.5) <= T5_TOL))
    mono = {}
    for dd in ev:
        P = cummin_pred(raw[dd])
        mono[dd] = {f'j{j + 1}': float((raw[dd][:, j] > raw[dd][:, j - 1] + 1e-12).mean()) for j in range(1, len(JS))}
        out = ev[dd][KEYS].copy()
        for ji, j in enumerate(JS):
            out[f'P{j}'] = P[:, ji].astype(np.float32); out[f'R{j}'] = raw[dd][:, ji].astype(np.float32)
        out.to_parquet(f'{OUTD}/pred_fold{test}_{dd}.parquet', index=False)
    res = dict(test=test, cal=cal, train_days=list(tr_days), features=F, dropped=A['drop'], info=info, T1=True, T5=T5, T4_monotonicity=mono,
               params=PD.LGB_PARAMS, rounds=PD.N_ROUNDS, sub_day=SUB_DAY, sub_n=SUB_N, seconds=round(time.time() - T0, 1))
    json.dump(res, open(f'{OUTD}/fit_{test}.json', 'w'), indent=1, default=float)
    print(f'fit {test}: ' + '; '.join(f'j{j} train {info[f"j{j}"]["train_rows"]} pos {info[f"j{j}"]["positives"]} AUC tr/cal/test '
                                     f'{info[f"j{j}"]["auc_train"]:.3f}/{info[f"j{j}"]["auc_cal"]:.3f}/{info[f"j{j}"]["auc_test"]:.3f}' for j in JS)
          + f'; T5 {T5}; {time.time() - T0:.0f} s', flush=True)


# ---------------------------------------------------------------- align (one test day): the target and the model against our orders' fills
def order_steps(W, r, shf, src, tkv, tkp, st):
    """Decision states (every EVERY_ALIGN-th step) of every client-known resting order and its fills in the horizon after the cancel."""
    cols = {c: [] for c in ('m', 'side', 'k', 'rung', 'depth', 'age', 'q', 'v1', 'q_ws', 'v1_ws', 'q_thr', 'v1_thr')}
    m = int(W[0]['m'])
    for s in (0, 1):
        w = W[s]; tf, mff = w['tf_mff']; sg = float(w['sg']); anc = np.asarray(w['anc'], float); tt = (w['anc'] - shf)
        tg = np.asarray(w['tg'], float); best = np.asarray(w['best'], 'int64')
        for d in range(NR):
            orders = [o for o in r['order_states'][(s, d)] if not o['ref']]
            fl = {id(o): [] for o in orders}
            for f in r['fills'][(s, d)]:
                i, pt, qq, thr = int(f[0]), int(f[1]), float(f[2]), bool(f[3]); t = float(tt[i])
                cand = [o for o in orders if o['pt'] == pt and o['ta'] <= t < o['tr']]
                if len(cand) != 1:
                    st['unmatched_fills'] += 1; continue
                j1 = max(int(np.searchsorted(tf, anc[i] + 1000.0, side='right')) - 1, 0)
                v1 = sg * (float(mff[j1]) - pt)
                ty = fill_type(s, pt, thr, tkp[s][int(tkv[s][i])])
                fl[id(cand[0])].append((t, qq, v1, bool(src[s][i]), ty == 'THROUGH'))
                st['fills'] += 1
            for o in orders:
                F = sorted(fl[id(o)]); cum = 0.0; tfull = INF
                for x in F:
                    cum += x[1]
                    if cum >= SIZE - 1e-9:
                        tfull = x[0] + CL.L_NOTE; break
                t_end = min(o['tc'], tfull)
                k0 = int(np.searchsorted(tg, o['ta'], side='left')); k1 = int(np.searchsorted(tg, t_end, side='left'))
                ks = np.arange(k0, k1); ks = ks[ks % EVERY_ALIGN == 0]
                if not len(ks):
                    continue
                bk = best[ks]; okb = bk >= 0; ks = ks[okb]; bk = bk[okb]
                if not len(ks):
                    continue
                ft = [x[0] for x in F]; qv = [x[1] for x in F]
                q_, v_, qw, vw, qt, vt = horizon_sums(ft, [qv, [x[1] * x[2] for x in F], [x[1] * x[3] for x in F], [x[1] * x[2] * x[3] for x in F],
                                                           [x[1] * x[4] for x in F], [x[1] * x[2] * x[4] for x in F]], tg[ks])
                n_ = len(ks); st['order_steps'] += n_
                cols['m'].append(np.full(n_, m, 'int64')); cols['side'].append(np.full(n_, s, 'int64')); cols['k'].append(ks.astype('int64'))
                cols['rung'].append(np.full(n_, d, 'int64'))
                cols['depth'].append(((bk - o['pt']) if s == 0 else (o['pt'] - bk)).astype('int64'))
                cols['age'].append((tg[ks] - o['ta']).astype(np.float32))
                for c, a in zip(('q', 'v1', 'q_ws', 'v1_ws', 'q_thr', 'v1_thr'), (q_, v_, qw, vw, qt, vt)):
                    cols[c].append(np.asarray(a, np.float32))
    return {c: (np.concatenate(v) if v else np.zeros(0)) for c, v in cols.items()}


def align_day(day):
    all_selftests()
    if day not in TESTS:
        sys.exit(f'REFUSED: {day} is not a fold test day')
    T0 = time.time(); D = K.load_day(day)
    tkf = pd.read_parquet(f'{K.IO}/step1/{day}/legs_tk.parquet', columns=['tk']).tk.values
    LT = D['LT']; ix = LT.index.values; grp = np.where(tkf[ix] >= 0, tkf[ix], -1 - ix)
    WSg = pd.Series(TS.ws_groups(LT.how.values, grp), index=LT.index); TKS = pd.Series(grp, index=LT.index)
    st = dict(windows=0, runs=0, viol={v_: 0 for v_ in J3.VIOL}, unmatched_fills=0, fills=0, order_steps=0, terminal_checks=0)
    out = {f'{q}{int(shf)}': [] for q, shf in CFGS}
    for W in P1.window_pairs(D):
        st['windows'] += 1; m = W[0]['m']; n = W[0]['n']
        src = {}; tkv = {}; tkp = {}
        for s in (0, 1):
            if (m, s) in D['LI']:
                l = D['LT'].iloc[D['LI'][(m, s)]]
                o_ = np.lexsort(((-l.ptick.values if s == 0 else l.ptick.values), l.anc.values)); lrow = l.index.values[o_]
            else:
                lrow = np.zeros(0, 'int64')
            src[s] = WSg.reindex(lrow).values.astype(bool) if len(lrow) else np.zeros(0, bool)
            tkv[s] = TKS.reindex(lrow).values if len(lrow) else np.zeros(0, 'int64'); tkp[s] = {}
            for t_, p_ in zip(tkv[s], W[s]['ptl']):
                tkp[s].setdefault(int(t_), []).append(int(p_))
        aux = {0: [False] * n, 1: [False] * n}
        for q, shf in CFGS:
            r = CL.sim_joint(W, q, shf, None, aux, decide=None, idle_only=0, mode='none', cooldown=CL.REPOST_MS)
            st['runs'] += 1
            for v_ in J3.VIOL:
                st['viol'][v_] += r['stat']['viol'][v_]
            CL.portfolio_terminal(r, W[0]['tf_mff'][0], W[0]['tf_mff'][1]); st['terminal_checks'] += 1
            out[f'{q}{int(shf)}'].append(order_steps(W, r, shf, src, tkv, tkp, st))
        if st['windows'] % 40 == 0:
            mem_guard(f'align {day}')
    if any(st['viol'].values()) or st['unmatched_fills']:
        sys.exit(f'REFUSED (no output for {day}): lane violations / unmatched fills {st}')
    os.makedirs(OUTD, exist_ok=True)
    for cfg, lst in out.items():
        pd.DataFrame({c: np.concatenate([x[c] for x in lst]) for c in lst[0]}).to_parquet(f'{OUTD}/align_{day}_{cfg}.parquet', index=False)
    st['seconds'] = round(time.time() - T0, 1)
    json.dump(st, open(f'{OUTD}/align_{day}.json', 'w'), indent=1, default=str)
    print(f'align {day}: {st}', flush=True)


# ---------------------------------------------------------------- report
def report():
    all_selftests()
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    A = json.load(open(f'{OUTD}/audit.json')); FJ = {t: json.load(open(f'{OUTD}/fit_{t}.json')) for t in TESTS}
    RJ = {d: json.load(open(f'{OUTD}/rows_{d}.json')) for d in DAYS9}
    T5 = FJ[TESTS[0]]['T5']
    if not T5 or not T5['ok']:
        sys.exit(f'REFUSED: permuted-label falsifier failed {T5}')
    E = {}
    for t in TESTS:
        R = pd.read_parquet(f'{OUTD}/rows_{t}.parquet', columns=KEYS + ['D0', 'D1', 'v', 'masked'])
        P = pd.read_parquet(f'{OUTD}/pred_fold{t}_{t}.parquet')
        if not (np.array_equal(R.m.values, P.m.values) and np.array_equal(R.side.values, P.side.values) and np.array_equal(R.k.values, P.k.values)):
            sys.exit(f'REFUSED: predictions of {t} not aligned with its rows')
        for j in JS:
            R[f'P{j}'] = P[f'P{j}'].values
        S = pd.read_parquet(f'{SD.OUTD}/pred_{t}.parquet', columns=['m', 'side', 'k'] + [f'NEW_P{j}' for j in (1, 2, 3)])
        E[t] = R.merge(S, on=['m', 'side', 'k'], how='left', validate='one_to_one')
    L = [f'STOP-MODEL (the keep / cancel model for every ladder level), {len(TESTS)} chronological test days {", ".join(TESTS)}; as of {asof}',
         f'controls: rows parity / falsifier / brute-force passed on 9 days (no output otherwise); T5 permuted-label AUC {T5["auc"]:.4f} '
         f'({T5["rows"]} rows); inputs {len(FJ[TESTS[0]]["features"])} (dropped by the audit: {A["drop"]})', '']
    R_ = {}
    # R702 skill and R703 calibration
    L.append('SKILL per level j (the order j - 1 ticks behind the best is run over), rows not ALREADY_j: AUC | Brier skill vs the training base rate')
    sk = {}
    for j in JS:
        ys, ps, bs, aucs, bss = [], [], [], [], []
        for t in TESTS:
            e = E[t]; ok = e.D0.values < j; y = (e.D1.values[ok] >= j).astype(float); p = e[f'P{j}'].values[ok].astype(float)
            b = FJ[t]['info'][f'j{j}']['base_rate']
            aucs.append(auc_nan(y.astype(bool), p)[0])
            bss.append(1.0 - float(((p - y) ** 2).mean()) / float(((b - y) ** 2).mean()))
            ys.append(y); ps.append(p); bs.append(np.full(len(y), b))
        y = np.concatenate(ys); p = np.concatenate(ps); b = np.concatenate(bs)
        a_pool = auc_nan(y.astype(bool), p)[0]; bss_pool = 1.0 - float(((p - y) ** 2).sum()) / float(((b - y) ** 2).sum())
        dd = int(sum(a >= AUC_DAY for a in aucs)); db = int(sum(x > 0 for x in bss))
        sk[j] = dict(auc_pool=a_pool, auc_days=aucs, bss_pool=bss_pool, bss_days=bss, rows=int(len(y)), rate=float(y.mean()),
                     pred=float(p.mean()), predictable=bool(a_pool >= AUC_POOL and dd >= DAY_RULE), beats_base=bool(bss_pool > 0 and db >= DAY_RULE))
        R_[f'R702_j{j}'] = ('PREDICTABLE' if sk[j]['predictable'] else 'WEAK') + ' / ' + ('BEATS-BASE' if sk[j]['beats_base'] else 'NO-SKILL')
        L.append(f'  j{j}: rows {len(y):8d} rate {y.mean():6.2%} mean P {p.mean():6.2%} | AUC {a_pool:.3f} days '
                 + ' '.join(f'{a:.3f}' for a in aucs) + f' | BSS {bss_pool:+.3f} days ' + ' '.join(f'{x:+.3f}' for x in bss) + f'  -> {R_[f"R702_j{j}"]}')
        if j <= 3:
            qs = np.quantile(p, np.linspace(0.1, 0.9, 9)); bi = np.searchsorted(qs, p, side='right')
            cal = [(float(p[bi == i].mean()), float(y[bi == i].mean()), int((bi == i).sum())) for i in range(10) if (bi == i).any()]
            gap = max(abs(a - c) for a, c, _ in cal)
            R_[f'R703_j{j}'] = 'CALIBRATED' if gap <= CAL_TOL else 'MISCALIBRATED'
            sk[j]['calibration'] = cal
            L.append(f'      calibration deciles (mean P / rate): ' + ' '.join(f'{a:.3f}/{c:.3f}' for a, c, _ in cal) + f'  max gap {gap:.3f} -> {R_[f"R703_j{j}"]}')
    L.append('')
    # R704 vs the guard, R705 vs STOP-DEPTH
    L.append('AGAINST THE GUARD (rows with a known guard value; score -v) and STOP-DEPTH NEW (its guard-on population); per-day AUC model | ref')
    cmp_ = {}
    for j in (1, 2, 3):
        mg, gg, ms, ss = [], [], [], []
        for t in TESTS:
            e = E[t]; ok = (e.D0.values < j) & np.isfinite(e.v.values) & (e.v.values != MASK_V); y = e.D1.values[ok] >= j
            mg.append(auc_nan(y, e[f'P{j}'].values[ok])[0]); gg.append(auc_nan(y, -e.v.values[ok].astype(float))[0])
            o2 = ok & (e.v.values <= GUARD) & np.isfinite(e[f'NEW_P{j}'].values); y2 = e.D1.values[o2] >= j
            ms.append(auc_nan(y2, e[f'P{j}'].values[o2])[0]); ss.append(auc_nan(y2, e[f'NEW_P{j}'].values[o2])[0])
        b1, w1 = day_rule(mg, gg); b2, w2 = day_rule(ms, ss)
        R_[f'R704_j{j}'] = 'BETTER' if b1 >= DAY_RULE else ('WORSE' if w1 >= DAY_RULE else 'NOT-SHOWN')
        R_[f'R705_j{j}'] = 'BETTER' if b2 >= DAY_RULE else ('WORSE' if w2 >= DAY_RULE else 'NOT-SHOWN')
        cmp_[j] = dict(model_all=mg, guard=gg, model_guard_on=ms, stop_depth=ss)
        L.append(f'  j{j} vs guard: ' + ' '.join(f'{a:.3f}|{b:.3f}' for a, b in zip(mg, gg)) + f' -> {R_[f"R704_j{j}"]}; vs STOP-DEPTH (guard-on): '
                 + ' '.join(f'{a:.3f}|{b:.3f}' for a, b in zip(ms, ss)) + f' -> {R_[f"R705_j{j}"]}')
    L.append('')
    # R706 keep quality per rung
    L.append('KEEP QUALITY per rung d (keep the guard\'s count of steps; run-over share of kept steps, model | guard; stopped-at-the-rung share)')
    kq = {}
    for d in range(6):
        j = d + 1; mr, gr, ms_, gs_ = [], [], [], []
        for t in TESTS:
            e = E[t]; ok = (e.D0.values < j) & np.isfinite(e.v.values) & (e.v.values != MASK_V)
            y = e.D1.values[ok] >= j; at = e.D1.values[ok] == d; v = e.v.values[ok]; p = e[f'P{j}'].values[ok]
            gk = v > GUARD; nk = int(gk.sum())
            mk = np.zeros(len(p), bool); mk[np.argsort(p, kind='stable')[:nk]] = True
            mr.append(float(y[mk].mean()) if nk else NAN); gr.append(float(y[gk].mean()) if nk else NAN)
            ms_.append(float(at[mk].mean()) if nk else NAN); gs_.append(float(at[gk].mean()) if nk else NAN)
        b_, w_ = day_rule(mr, gr, lower_is_better=True)
        R_[f'R706_d{d}'] = 'FEWER-RUN-OVERS' if b_ >= DAY_RULE else ('MORE-RUN-OVERS' if w_ >= DAY_RULE else 'NOT-SHOWN')
        kq[d] = dict(model_runover=mr, guard_runover=gr, model_at=ms_, guard_at=gs_)
        L.append(f'  d{d}: run over ' + ' '.join(f'{a:.3f}|{b:.3f}' for a, b in zip(mr, gr)) + '; at the rung '
                 + ' '.join(f'{a:.3f}|{b:.3f}' for a, b in zip(ms_, gs_)) + f' -> {R_[f"R706_d{d}"]}')
    L.append('')
    # R701 / R707 alignment with our orders' fills
    al = {}
    for q, shf in CFGS:
        cfg = f'{q}{int(shf)}'; per_day = {}
        for t in TESTS:
            O = pd.read_parquet(f'{OUTD}/align_{t}_{cfg}.parquet')
            O = O[(O.depth >= 0) & (O.depth <= 5)]
            e = E[t][['m', 'side', 'k', 'D0', 'D1'] + [f'P{j}' for j in JS]]
            O = O.merge(e, on=['m', 'side', 'k'], how='inner', validate='many_to_one')
            j_ = O.depth.values + 1; ok = O.D0.values < j_; O = O[ok]; j_ = j_[ok]
            O = O.assign(y=(O.D1.values >= j_), P=np.choose(j_ - 1, [O[f'P{j}'].values for j in JS]), V=O.v1.values / SIZE,
                         Vws=O.v1_ws.values / SIZE, Ve2=(O.v1.values - O.v1_ws.values) / SIZE)
            dec = np.full(len(O), -1)
            for dpt, ix in O.groupby('depth').indices.items():
                pr = O.P.values[ix]; qs = np.quantile(pr, np.linspace(0.1, 0.9, 9)); dec[ix] = np.searchsorted(qs, pr, side='right')
            O['dec'] = dec
            per_day[t] = O[['depth', 'y', 'P', 'V', 'Vws', 'Ve2', 'q', 'q_thr', 'dec']]
        al[cfg] = per_day
    alr = {}
    for cfg, per_day in al.items():
        X = pd.concat(per_day.values(), ignore_index=True)
        v1d = [float(per_day[t].V[per_day[t].y].mean()) for t in TESTS]; v0d = [float(per_day[t].V[~per_day[t].y].mean()) for t in TESTS]
        tdd = [float(per_day[t].V[per_day[t].dec == 9].mean()) for t in TESTS]; bhd = [float(per_day[t].V[per_day[t].dec <= 4].mean()) for t in TESTS]
        neg = X.V.values < 0; pos = X.V.values > 0
        alr[cfg] = dict(rows=int(len(X)), V_y1=float(X.V[X.y].mean()), V_y0=float(X.V[~X.y].mean()), V_y1_days=v1d, V_y0_days=v0d,
                        missed_harm=float(X.V.values[neg & ~X.y.values].sum() / X.V.values[neg].sum()) if neg.any() else NAN,
                        cancelled_good=float(X.V.values[pos & X.y.values].sum() / X.V.values[pos].sum()) if pos.any() else NAN,
                        V_top=float(X.V[X.dec == 9].mean()), V_bottom_half=float(X.V[X.dec <= 4].mean()), V_top_days=tdd, V_bh_days=bhd,
                        fill_rate_y1=float((X.q[X.y] > 0).mean()), fill_rate_y0=float((X.q[~X.y] > 0).mean()),
                        by_depth={int(d_): dict(rows=int(len(g)), share_y1=float(g.y.mean()), V_y1=float(g.V[g.y].mean()), V_y0=float(g.V[~g.y].mean()),
                                                Vws_y1=float(g.Vws[g.y].mean()), Ve2_y1=float(g.Ve2[g.y].mean()),
                                                Vws_y0=float(g.Vws[~g.y].mean()), Ve2_y0=float(g.Ve2[~g.y].mean()),
                                                V_dec=[float(g.V[g.dec == i].mean()) for i in range(10)])
                                  for d_, g in X.groupby('depth')})
        b1, _ = day_rule(v1d, v0d, lower_is_better=True); b7, _ = day_rule(tdd, bhd, lower_is_better=True)
        alr[cfg]['aligned'] = bool(alr[cfg]['V_y1'] < alr[cfg]['V_y0'] and b1 >= DAY_RULE and alr[cfg]['V_y1'] < 0)
        alr[cfg]['prices_harm'] = bool(alr[cfg]['V_top'] < alr[cfg]['V_bottom_half'] and b7 >= DAY_RULE)
    R_['R701'] = 'ALIGNED' if alr['M50']['aligned'] else 'MISALIGNED'
    R_['R707'] = 'PRICES-HARM' if alr['M50']['prices_harm'] else 'NOT-SHOWN'
    L.append('ALIGNMENT with our orders (keep-all ladder, rungs 0-5 x 30 sh, client lane; every 250 ms; V = 1 s value of fills in (t + 22, t + 1000] '
             'ms, cents per order share; rows overlap: decision states, not P&L)')
    for cfg in alr:
        a = alr[cfg]
        L.append(f'  {cfg}: rows {a["rows"]}; V | run over {a["V_y1"]:+.4f} vs not {a["V_y0"]:+.4f} (days ' + ' '.join(f'{x:+.3f}|{z:+.3f}' for x, z in zip(a['V_y1_days'], a['V_y0_days']))
                 + f'); fill rate {a["fill_rate_y1"]:.1%} vs {a["fill_rate_y0"]:.1%}; missed harm {a["missed_harm"]:.1%}; good value in run-over rows {a["cancelled_good"]:.1%}'
                 + (f'  -> R701 {R_["R701"]}' if cfg == 'M50' else ''))
        L.append(f'        top-risk decile V {a["V_top"]:+.4f} vs bottom half {a["V_bottom_half"]:+.4f} (days ' + ' '.join(f'{x:+.3f}|{z:+.3f}' for x, z in zip(a['V_top_days'], a['V_bh_days']))
                 + ')' + (f'  -> R707 {R_["R707"]}' if cfg == 'M50' else ''))
        for d_, g in a['by_depth'].items():
            L.append(f'        depth {d_}: rows {g["rows"]:7d} run over {g["share_y1"]:6.2%}; V y1 {g["V_y1"]:+.3f} (WS {g["Vws_y1"]:+.3f} / E2 {g["Ve2_y1"]:+.3f}) '
                     f'y0 {g["V_y0"]:+.3f} (WS {g["Vws_y0"]:+.3f} / E2 {g["Ve2_y0"]:+.3f}); V by risk decile ' + ' '.join(f'{x:+.3f}' for x in g['V_dec']))
    L.append('')
    # training checks
    L.append('TRAINING CHECKS per fold: T2 sample - population rate (pp), T3 positives, T6 AUC train / cal / test (j1, j2, j3), T4 raw non-monotone share, T7 top inputs (j1)')
    for t in TESTS:
        f = FJ[t]; inf = f['info']
        L.append(f'  {t} (train {",".join(x[4:] for x in f["train_days"])}; cal {f["cal"][4:]}): T2 ' + ' '.join(f'{100 * inf[f"j{j}"]["T2_gap"]:+.2f}' for j in JS)
                 + ' | T3 ' + ' '.join(str(inf[f'j{j}']['positives']) for j in JS) + ' | T6 '
                 + ' '.join(f'{inf[f"j{j}"]["auc_train"]:.3f}/{inf[f"j{j}"]["auc_cal"]:.3f}/{inf[f"j{j}"]["auc_test"]:.3f}' for j in (1, 2, 3))
                 + ' | T4 ' + ' '.join(f'{x:.3f}' for x in f['T4_monotonicity'][t].values()))
        L.append('      T7 ' + ', '.join(f'{c} {g:.2f}' for c, g in inf['j1']['top_gain'][:8]))
    t2_bad = [(t, j) for t in TESTS for j in JS if abs(FJ[t]['info'][f'j{j}']['T2_gap']) > T2_TOL]
    L.append(f'  T2 sample parity within {100 * T2_TOL:.1f} pp: {"all" if not t2_bad else f"FAILS {t2_bad}"}; T5 permuted-label AUC {T5["auc"]:.4f} -> PASS')
    L.append('')
    miss = [f'R701 {R_["R701"]}'] if R_['R701'] != 'ALIGNED' else []
    miss += [f'R702 j{j} {R_[f"R702_j{j}"]}' for j in (1, 2, 3) if R_[f'R702_j{j}'] != 'PREDICTABLE / BEATS-BASE']
    miss += [f'R707 {R_["R707"]}'] if R_['R707'] != 'PRICES-HARM' else []
    R_['VERDICT'] = 'MODEL-READY' if not miss else 'NOT-READY: ' + '; '.join(miss)
    L.append(f'READINGS {json.dumps(R_)}')
    L.append(f'VERDICT {R_["VERDICT"]} (development only: consumed days, C8 / C16)')
    res = dict(asof=asof, readings=R_, skill=sk, compare=cmp_, keep_quality=kq, align=alr, T5=T5, dropped=A['drop'], flags=A['flag'],
               rows={d: dict(rows=RJ[d]['rows'], parity=RJ[d]['parity'], falsifier=RJ[d]['falsifier'], brute=RJ[d]['brute']) for d in DAYS9},
               fits={t: dict(info=FJ[t]['info'], T4=FJ[t]['T4_monotonicity']) for t in TESTS})
    json.dump(res, open(f'{OUTD}/stop_model.json', 'w'), indent=1, default=float)
    open(f'{OUTD}/stop_model.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L), flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'selftest'
    if cmd == 'selftest':
        all_selftests()
    elif cmd == 'rows':
        rows_day(sys.argv[2])
    elif cmd == 'audit':
        audit()
    elif cmd == 'fit':
        fit_fold(sys.argv[2])
    elif cmd == 'align':
        align_day(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit(f'unknown command {cmd}')
