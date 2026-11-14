"""INFLUENCE (2026-11-14, session ce68ebe0; USER: "We need to reset the target and rebuild the model, to measure the influence effects").

STOP-DEPTH measured, at every guard-on step, how far the price moves in the next 1 s; a cleaner, richer model barely helped (AUC +0.003)
and did not change the keep decision. Two things in that set-up hide the signal's influence: the target starts at every guard-on step and
stops at 1 s, while a signal is one event and 43% of big moves take longer than 1 s to their peak (LEARN-PULL R556); and the signal's size
enters only as raw Binance returns, not in this market's price units. This study resets both.

The influence in price units: step 5's Binance-implied fair price p_bn = 100 N(ln(S x basis / start price) / (sigma sqrt(seconds left)))
(cents; basis-adjusted; sigma = 1 s log-return s.d. over the previous 30 min). Signed against the side (adv = -1 bids, +1 asks; > 0 = against
us):
  dfv250 / dfv1000  change of p_bn over the last 250 ms / 1 s (the value at the latest step at or before t - h, within 60 ms)
  gap_adv           p_bn minus the known Predict mid (step 5's 'gap'): how far the fair price says the price should still go
  sens              cents of p_bn per 1 bp BTC move now, 100 N'(z) 1e-4 / (sigma sqrt(seconds left)); pbn_dist = |p_bn - 50|; tau_rem
9 consumed replay days (10-23..27, 10-31, 11-01..03), first 60 s of each window; steps with the guard value masked (start price not known
yet, FINDINGS P16) are never used as events.

PART A, THE INFLUENCE OF A FAIR-PRICE MOVE ON THE MARKET (no model). Events: 50 ms grid steps with a valid known book where dfv250 >= 0.5 c
against a side, the first step of each run (one side per move). Response on the venue frames from the venue state at the event: the Predict
mid against us at +250 ms / 1 s / 2 s / 3 s (ticks); the side's best, farthest against us within 1 / 2 / 3 s. By dfv250 bins [0.5, 1) /
[1, 2) / [2, 4) / >= 4 c; pass-through = mean mid move / mean dfv250.
PART B, THE SIGNAL'S MOVE (the reset target). Events: our guard's onsets (guard on at the step, off at the step before; neither masked;
valid known book). Reference: the known best at the onset. On the venue frames: D0 = against us by t + 22 ms (our cancel's landing: past
saving); PEAK = farthest against us over the state at t + 22 ms and frames in (t + 22 ms, t + 2 s] (1 s and 3 s printed); END = against us at
t + 3 s (where it settles, signed); time to the peak. For threshold j: status ALREADY_j if D0 >= j, else y_j = [PEAK >= j].
Models at onsets (LightGBM binary per j, PULL-DEPTH's pinned parameters and rounds, leave one day out, all training onsets (<= 300,000)):
  FULL     STOP-DEPTH's NEW inputs + the influence inputs (dfv250, dfv1000, gap_adv, sens, pbn_dist, tau_rem)    j = 1..5
  NOINF    STOP-DEPTH's NEW inputs, on the new target                                                          j = 1..3
  INFONLY  the influence inputs + the side                                                                     j = 1..3
  STEP     STOP-DEPTH's NEW predictions at the same steps (trained on the 1 s step target), as scores
Readings DECLARED BEFORE THE FIRST RUN (day rule >= 7 of 9):
 R590 PRICE-FOLLOWS (A): per day, over events with dfv250 >= 1 c, mean Predict mid move against us at 1 s / mean dfv250 > 0 on >= 7 days ->
      FOLLOWS; else NOT-SHOWN. Pass-through at each horizon and bin printed.
 R591 SIZE-FOLLOWS-INFLUENCE (B): Spearman correlation of gap_adv with PEAK at onsets not ALREADY_1, > 0 on >= 7 days -> FOLLOWS; < 0 on >= 7
      -> REVERSE; else NOT-SHOWN (dfv250 printed beside).
 R592 INFLUENCE-HELPS (B): FULL's out-of-day AUC above NOINF's on >= 7 days for each j = 1, 2, 3 -> HELPS; below on >= 7 -> HURTS; else
      NOT-SHOWN.
 R593 RESET-HELPS (B): FULL's AUC above STEP's, on the new target, on >= 7 days for each j = 1, 2, 3 -> BETTER; below on >= 7 -> WORSE; else
      NOT-SHOWN.
 R594 KEEP-QUALITY (B, rungs 1 and 2 each): on each held-out day STEP keeps rung d at the onsets with P_{d+1} < 0.25 (FB25's line); FULL
      keeps the same number of onsets with the lowest P_{d+1}; the share of kept onsets run over (PEAK >= d + 1): lower on >= 7 days ->
      FEWER-RUN-OVERS; higher on >= 7 -> MORE-RUN-OVERS; else NOT-SHOWN.
Printed, not readings: PEAK / END / time to the peak at onsets by gap_adv and dfv250 bins; AUC for j = 4, 5; calibration; top inputs; statuses.
Controls (any failure -> no output): synthetic cases for the move (peak, end, time to the peak, already, both sides), the signed fair-price
change, the onset rule (a masked previous step or the first grid step is not an onset), the event runs (first step only, one side) and the
sensitivity, with planted errors (dfv's side sign dropped; every guard step an onset; event runs not debounced) that must each fail a case;
STOP-DEPTH's self-tests; p_bn found for every event step used (else a status); no held-out row in a fit (asserted); every onset matched to a
STEP prediction or counted; memory self-guard (2 GiB).
Limits: p_bn's basis and volatility inputs are step 5's (basis drift, FINDINGS / memory; sigma from the previous 30 min); first minute only
(240-300 s left); our guard's onsets; consumed days, exploratory; AUC ranks onsets, the replay values them.
Usage (run_light, PF_SLOT=claude): pf_influence.py selftest | rows <day> | fit <day> | report; driver run_influence.sh
"""
import bisect
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import ndtri

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_kp_s4 as K                                      # noqa: E402
import pf_qr_queue_race as QR                             # noqa: E402
import pf_pull_depth as PD                                # noqa: E402
import pf_stop_depth as SD                                # noqa: E402

DAYS9 = K.DAYS9
OUTD = f'{K.IO}/influence_20261114'
FV_TH = 0.5; FV_BINS = ((0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 1e9)); HS = (250.0, 1000.0, 2000.0, 3000.0)
JS = (1, 2, 3, 4, 5); JS_ABL = (1, 2, 3); SUB_N = 300000; TAU_STEP = 0.25; MASK_V = -1.0; PBN_TOL = 60.0
INF_FEAT = ['dfv250', 'dfv1000', 'gap_adv', 'sens', 'pbn_dist', 'tau_rem']
FULL_FEAT = SD.NEW_FEAT + INF_FEAT; NOINF_FEAT = list(SD.NEW_FEAT); INFONLY_FEAT = INF_FEAT + ['fside']
ARMS = (('FULL', FULL_FEAT, JS), ('NOINF', NOINF_FEAT, JS_ABL), ('INFONLY', INFONLY_FEAT, JS_ABL))
INF = float('inf'); NAN = float('nan')
MUT_DFV_NOSIDE = MUT_ONSET_ALL = MUT_NO_DEBOUNCE = 0      # planted errors (self-test only)
MEM_CAP_MIB = 2048.0


# ---------------------------------------------------------------- pieces
def adv_of(side):
    return 1.0 if (side == 1 or MUT_DFV_NOSIDE) else -1.0


def pbn_at(rt, pv, t):
    j = bisect.bisect_right(rt, t) - 1
    return float(pv[j]) if j >= 0 and t - rt[j] <= PBN_TOL and np.isfinite(pv[j]) else NAN


def sens_of(p, sig, tr):
    if not (np.isfinite(p) and np.isfinite(sig) and np.isfinite(tr)) or sig <= 0 or tr <= 0:
        return NAN
    z = ndtri(min(max(p / 100.0, 1e-6), 1 - 1e-6))
    return float(100.0 * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi) * 1e-4 / (sig * np.sqrt(tr)))


def onsets(v):
    """Guard on at k and off at k - 1, neither masked; the first grid step is never an onset."""
    g = [x <= K.GUARD_PULL for x in v]
    if MUT_ONSET_ALL:
        return [k for k in range(len(v)) if g[k] and v[k] != MASK_V]
    return [k for k in range(1, len(v)) if g[k] and not g[k - 1] and v[k] != MASK_V and v[k - 1] != MASK_V]


def run_starts(x, th=FV_TH):
    """First steps of runs with x >= th (NaN breaks a run)."""
    ok = [bool(np.isfinite(a) and a >= th) for a in x]
    if MUT_NO_DEBOUNCE:
        return [k for k in range(len(ok)) if ok[k]]
    return [k for k in range(len(ok)) if ok[k] and (k == 0 or not ok[k - 1])]


def move_path(b0, valid, tfl, side, jk, t, lc=None):
    """(D0, PEAK1, PEAK2, PEAK3, END, time to PEAK2) against us beyond the known frame jk's best (header)."""
    lc = K.L_CANCEL if lc is None else lc
    D0, P1, _ = SD.move_clean(b0, valid, tfl, side, jk, t, lc, 1000.0)
    _, P2, _ = SD.move_clean(b0, valid, tfl, side, jk, t, lc, 2000.0)
    _, P3, _ = SD.move_clean(b0, valid, tfl, side, jk, t, lc, 3000.0)
    if not np.isfinite(D0):
        return (NAN,) * 6
    ref = int(b0[jk]); a = (lambda x: ref - int(x)) if side == 0 else (lambda x: int(x) - ref)
    je = bisect.bisect_right(tfl, t + 3000.0) - 1
    while je > jk and not valid[je]:
        je -= 1
    END = float(a(b0[je]))
    j0 = bisect.bisect_right(tfl, t + lc); jl = j0 - 1
    while jl > jk and not valid[jl]:
        jl -= 1
    tp = NAN
    if a(b0[jl]) >= P2:
        tp = lc
    else:
        j2 = bisect.bisect_right(tfl, t + 2000.0)
        for j in range(j0, j2):
            if valid[j] and a(b0[j]) >= P2:
                tp = tfl[j] - t; break
    return D0, P1, P2, P3, END, tp


def venue_response(b0, opp, valid, tfl, side, t):
    """From the venue state at t: Predict mid against us at t + 250 ms / 1 s / 2 s / 3 s (ticks) and the side's best, farthest against us
    within 1 / 2 / 3 s. NaN without a valid frame at or before t."""
    j = bisect.bisect_right(tfl, t) - 1
    while j >= 0 and not valid[j]:
        j -= 1
    if j < 0:
        return (NAN,) * 7
    adv = 1.0 if side == 1 else -1.0
    m0 = (b0[j] + opp[j]) / 2.0; ref = int(b0[j]); out = []
    for h in HS:
        jh = bisect.bisect_right(tfl, t + h) - 1
        while jh > j and not valid[jh]:
            jh -= 1
        out.append(float(adv * ((b0[jh] + opp[jh]) / 2.0 - m0)))
    for h in (1000.0, 2000.0, 3000.0):
        jh = bisect.bisect_right(tfl, t + h)
        seg = b0[j:jh][valid[j:jh]]
        out.append(float(max(0, (ref - int(seg.min())) if side == 0 else (int(seg.max()) - ref))) if len(seg) else 0.0)
    return tuple(out)


# ---------------------------------------------------------------- self-tests
def selftest():
    bad = []

    def chk(nm, c):
        if not c:
            bad.append(nm)
    tfl = [0.0, 100.0, 1500.0, 2500.0, 3500.0]; valid = np.ones(5, bool); b0 = np.array([50, 49, 47, 48, 45])
    r = move_path(b0, valid, tfl, 0, 0, 0.0, lc=22.0)
    chk(f'move path {r}', r[0] == 0 and r[1] == 1 and r[2] == 3 and r[3] == 3 and r[4] == 2 and r[5] == 1500.0)
    a0 = np.array([51, 51, 53, 52, 55]); r = move_path(a0, valid, tfl, 1, 0, 0.0, lc=22.0)
    chk(f'ask path {r}', r[0] == 0 and r[2] == 2 and r[4] == 1)
    r = move_path(np.array([50, 48, 48, 48, 48]), valid, [0.0, 10.0, 1500.0, 2500.0, 3500.0], 0, 0, 0.0, lc=22.0)
    chk(f'already {r}', r[0] == 2 and r[2] == 2 and r[5] == 22.0)
    chk('adv sign', adv_of(0) * (48.5 - 50.0) == 1.5 and adv_of(1) * (48.5 - 50.0) == -1.5)
    v = [-1.0, 0.2, -0.5, -0.6, 0.1, -0.3, -1.0, -0.4]
    chk(f'onsets {onsets(v)}', onsets(v) == [2, 5])
    chk(f'run starts {run_starts([0.2, 0.6, 0.7, 0.1, 0.8, NAN, 0.9])}', run_starts([0.2, 0.6, 0.7, 0.1, 0.8, NAN, 0.9]) == [1, 4, 6])
    chk(f'sens {sens_of(50.0, 1e-4, 100.0)}', abs(sens_of(50.0, 1e-4, 100.0) - 3.98942) < 1e-4)
    rt = [0.0, 50.0, 100.0]; pv = np.array([50.0, 49.0, 48.0])
    chk('pbn at', pbn_at(rt, pv, 120.0) == 48.0 and np.isnan(pbn_at(rt, pv, 200.0)) and np.isnan(pbn_at(rt, pv, -1.0)))
    vr = venue_response(np.array([50, 49, 48, 50, 47]), np.array([52, 51, 50, 52, 49]), valid, tfl, 0, 50.0)
    chk(f'venue response {vr}', vr == (1.0, 1.0, 2.0, 0.0, 1.0, 2.0, 2.0))
    return bad


def mutation_check():
    global MUT_DFV_NOSIDE, MUT_ONSET_ALL, MUT_NO_DEBOUNCE
    out = {}
    for nm in ('MUT_DFV_NOSIDE', 'MUT_ONSET_ALL', 'MUT_NO_DEBOUNCE'):
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
        sys.exit(f'INFLUENCE SELFTEST FAILED: {bad}')
    m = mutation_check()
    if not all(m.values()):
        sys.exit(f'INFLUENCE PLANTED ERROR NOT CAUGHT: {m}')
    print(f'influence selftests passed; planted errors caught: {m}', flush=True)
    return m


# ---------------------------------------------------------------- rows (one day)
def rows_day(day):
    all_selftests()
    T0 = time.time(); D = K.load_day(day)
    r = pd.read_parquet(f'{K.IO}/step5/{day}/rows_rl_d2_pk.parquet', columns=['mid', 'tau', 'p_bn', 'tau_rem', 'sigma'])
    r = r[r.mid.isin(D['W0'])]; r = r[r.tau.values < r.mid.map(D['W0']).values + 61000.0].sort_values(['mid', 'tau'], kind='stable')
    PB = {int(m): (g.tau.values.astype(float).tolist(), g.p_bn.values.astype(float), g.tau_rem.values.astype(float), g.sigma.values.astype(float))
          for m, g in r.groupby('mid', sort=False)}
    del r
    st = dict(window_sides=0, no_pbn_window=0, onsets=0, onset_no_pbn=0, onset_invalid=0, fv_events=0, fv_no_frame=0)
    ev = []; on = []
    for w in K.window_sides(D):
        st['window_sides'] += 1; n = w['n']; m = int(w['m']); side = int(w['side'])
        if not n:
            continue
        if m not in PB:
            st['no_pbn_window'] += 1; continue
        rt, pv, trv, sgv = PB[m]; adv = adv_of(side)
        tg = w['tg']; v = [float(x) for x in w['vkl']]
        b0, valid, _, _ = QR.runs(w['tf'], w['P5'], w['opp'], side); tfl = w['tf'].tolist()
        pk = np.array([pbn_at(rt, pv, float(t)) for t in tg])
        p250 = np.array([pbn_at(rt, pv, float(t) - 250.0) for t in tg]); p1000 = np.array([pbn_at(rt, pv, float(t) - 1000.0) for t in tg])
        okb = np.array([(int(w['jk'][k]) >= 0 and bool(valid[int(w['jk'][k])]) and w['best'][k] >= 0) for k in range(n)])
        msk = np.array([x == MASK_V for x in v])
        d250 = np.where(okb & ~msk, adv * (pk - p250), np.nan); d1000 = adv * (pk - p1000)
        # PART A: fair-price events against this side
        for k in run_starts(d250.tolist()):
            t = float(tg[k]); vr = venue_response(b0, w['opp'], valid, tfl, side, t)
            if not np.isfinite(vr[0]):
                st['fv_no_frame'] += 1; continue
            st['fv_events'] += 1
            jj = bisect.bisect_right(rt, t) - 1
            ev.append((int(day), m, side, t - w['ws'], float(d250[k]), float(d1000[k]), adv * float(w['X'][k][31]),
                       sens_of(pk[k], float(sgv[jj]) if jj >= 0 else NAN, float(trv[jj]) if jj >= 0 else NAN), int(v[k] <= K.GUARD_PULL)) + vr)
        # PART B: guard onsets
        nanP = np.full(n, np.nan)
        gl = np.zeros(n, int); g_ = 0
        for k in range(n):
            g_ = g_ + 1 if v[k] <= K.GUARD_PULL else 0; gl[k] = g_
        for k in onsets(v):
            st['onsets'] += 1
            if not okb[k]:
                st['onset_invalid'] += 1; continue
            if not np.isfinite(pk[k]):
                st['onset_no_pbn'] += 1; continue
            t = float(tg[k]); jk = int(w['jk'][k])
            f = PD.feats(w, k, nanP, int(gl[k])); fd = dict(zip(SD.PDF, f))
            sz = [fd[f's{i}'] for i in range(1, 6)]; gaps = [0.0] + [fd[f'g{i}'] for i in range(2, 6)]
            ex = SD.extra_feats(np.asarray(w['X'][k], float), side, sz, gaps) + list(SD.best_moves(w, k, side))
            jj = bisect.bisect_right(rt, t) - 1
            inf = [float(d250[k]), float(d1000[k]), adv * float(w['X'][k][31]),
                   sens_of(pk[k], float(sgv[jj]), float(trv[jj])), abs(float(pk[k]) - 50.0), float(trv[jj])]
            mp = move_path(b0, valid, tfl, side, jk, t)
            on.append([int(day), m, side, k, t - w['ws'], v[k]] + [fd[c] for c in SD.BASE_FEAT] + ex + inf + list(mp))
        if st['window_sides'] % 80 == 0:
            r2 = K.rss_mib()
            if r2 > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r2:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    os.makedirs(OUTD, exist_ok=True)
    pd.DataFrame(ev, columns=['day', 'm', 'side', 't_rel', 'dfv250', 'dfv1000', 'gap_adv', 'sens', 'guard_on', 'mid250', 'mid1000', 'mid2000',
                              'mid3000', 'best1000', 'best2000', 'best3000']).to_parquet(f'{OUTD}/events_{day}.parquet', index=False)
    cols = ['day', 'm', 'side', 'k', 't_rel', 'v'] + FULL_FEAT + ['D0', 'P1', 'P2', 'P3', 'END', 'TP']
    R = pd.DataFrame(on, columns=cols).astype({'day': 'int64', 'm': 'int64', 'side': 'int64', 'k': 'int64'})
    R.to_parquet(f'{OUTD}/onsets_{day}.parquet', index=False)
    st['onset_rows'] = len(R); json.dump(st, open(f'{OUTD}/rows_{day}.json', 'w'))
    print(f'{day}: {st} in {time.time() - T0:.0f} s', flush=True)


# ---------------------------------------------------------------- fit (one held-out day)
def fit_day(d):
    import lightgbm as lgb
    all_selftests()
    T0 = time.time()
    tr = pd.concat([pd.read_parquet(f'{OUTD}/onsets_{x}.parquet') for x in DAYS9 if x != d], ignore_index=True)
    assert not (tr.day == int(d)).any(), 'held-out day in the training rows'
    te = pd.read_parquet(f'{OUTD}/onsets_{d}.parquet')
    pred = dict(m=te.m.values, side=te.side.values, k=te.k.values); info = {}
    for arm, F, js in ARMS:
        prev = np.ones(len(te))
        for j in js:
            ok = (tr.D0.values < j) & np.isfinite(tr.P2.values); trj = tr[ok]
            if len(trj) > SUB_N:
                trj = trj.iloc[np.sort(np.random.default_rng(int(d) * 11 + j).choice(len(trj), SUB_N, replace=False))]
            y = (trj.P2.values >= j).astype(int)
            b = lgb.train(PD.LGB_PARAMS, lgb.Dataset(trj[F].values.astype(np.float32), y, feature_name=F), PD.N_ROUNDS)
            p = np.minimum(b.predict(te[F].values.astype(np.float32)), prev); prev = p; pred[f'{arm}_P{j}'] = p
            if arm == 'FULL':
                imp = b.feature_importance('gain'); info[f'j{j}'] = dict(train=len(trj), base_rate=float(y.mean()), top_gain=[F[i] for i in np.argsort(-imp)[:12]])
                os.makedirs(f'{OUTD}/models', exist_ok=True); b.save_model(f'{OUTD}/models/full_{d}_j{j}.txt')
        r2 = K.rss_mib()
        if r2 > MEM_CAP_MIB:
            sys.exit(f'REFUSED: resident memory {r2:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    pd.DataFrame(pred).to_parquet(f'{OUTD}/pred_{d}.parquet', index=False)
    json.dump(dict(info=info, train_rows=len(tr), params=PD.LGB_PARAMS, rounds=PD.N_ROUNDS), open(f'{OUTD}/fit_{d}.json', 'w'), indent=1)
    print(f'fit {d}: {info} in {time.time() - T0:.0f} s', flush=True)


# ---------------------------------------------------------------- report
def report():
    from scipy.stats import spearmanr
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    rs = {d: json.load(open(f'{OUTD}/rows_{d}.json')) for d in DAYS9}
    E = pd.concat([pd.read_parquet(f'{OUTD}/events_{d}.parquet') for d in DAYS9], ignore_index=True)
    A = []; unmatched = 0
    for d in DAYS9:
        R = pd.read_parquet(f'{OUTD}/onsets_{d}.parquet', columns=['day', 'm', 'side', 'k', 'v', 'dfv250', 'gap_adv', 'D0', 'P1', 'P2', 'P3', 'END', 'TP'])
        P = pd.read_parquet(f'{OUTD}/pred_{d}.parquet')
        S = pd.read_parquet(f'{SD.OUTD}/pred_{d}.parquet', columns=['m', 'side', 'k'] + [f'NEW_P{j}' for j in JS]).rename(
            columns={f'NEW_P{j}': f'STEP_P{j}' for j in JS})
        R = R.merge(P, on=['m', 'side', 'k'], how='left', validate='one_to_one').merge(S, on=['m', 'side', 'k'], how='left', validate='one_to_one')
        unmatched += int(R.STEP_P1.isna().sum()); A.append(R)
    A = pd.concat(A, ignore_index=True)
    R_ = {}
    L = [f'INFLUENCE, 9 consumed replay days, first 60 s; as of {asof}; controls passed; fair-price events {len(E)} (no frame '
         f'{sum(r["fv_no_frame"] for r in rs.values())}); guard onsets {sum(r["onsets"] for r in rs.values())} (invalid book '
         f'{sum(r["onset_invalid"] for r in rs.values())}, no fair price {sum(r["onset_no_pbn"] for r in rs.values())}, used {len(A)}; without a STEP '
         f'prediction {unmatched})', '']
    L.append('PART A - THE MARKET\'S RESPONSE TO A FAIR-PRICE MOVE AGAINST A SIDE (Binance-implied p_bn moved >= 0.5 c in 250 ms; ticks against us from the event)')
    L.append('  dfv250 bin    events  mean dfv | Predict mid moved: 250 ms  1 s  2 s  3 s | pass-through 1 s / 3 s | best moved >= 1 / 2 / 3 ticks in 2 s | guard on')
    for lo, hi in FV_BINS:
        x = E[(E.dfv250 >= lo) & (E.dfv250 < hi)]
        if not len(x):
            continue
        md = x.dfv250.mean()
        L.append(f'  [{lo:g}, {hi:g}){" " * (9 - len(f"{lo:g}{hi:g}"))} {len(x):>7}  {md:6.2f} c | {x.mid250.mean():5.2f} {x.mid1000.mean():5.2f} {x.mid2000.mean():5.2f}'
                 f' {x.mid3000.mean():5.2f} | {x.mid1000.mean() / md:4.2f} / {x.mid3000.mean() / md:4.2f} | {float((x.best2000 >= 1).mean()):5.1%}'
                 f' {float((x.best2000 >= 2).mean()):5.1%} {float((x.best2000 >= 3).mean()):5.1%} | {x.guard_on.mean():5.1%}')
    up = 0; ratios = {}
    for d in DAYS9:
        x = E[(E.day == int(d)) & (E.dfv250 >= 1.0)]
        ratios[d] = float(x.mid1000.mean() / x.dfv250.mean()) if len(x) else NAN; up += int(np.isfinite(ratios[d]) and ratios[d] > 0)
    R_['R590'] = dict(days=up, per_day=ratios, label='FOLLOWS' if up >= 7 else 'NOT-SHOWN')
    L.append(f'  R590: pass-through at 1 s for moves >= 1 c, per day ' + ', '.join(f'{d[4:]} {ratios[d]:.2f}' for d in DAYS9) + f' -> {R_["R590"]["label"]}')
    L.append('')
    Z = A[A.D0 == 0]
    dist = lambda x: ' | '.join(f'{lab} {v:5.1%}' for lab, v in zip(('0', '1', '2', '3', '4', '5+'), [float((np.minimum(x, 5) == i).mean()) for i in range(6)]))
    L.append(f'PART B - THE MOVE AFTER OUR SIGNAL (guard onsets, nothing moved by t + 22 ms, n {len(Z)}; ticks against us)')
    L.append(f'  farthest within 1 s {dist(Z.P1.values)}')
    L.append(f'  farthest within 2 s {dist(Z.P2.values)}   (the target)')
    L.append(f'  farthest within 3 s {dist(Z.P3.values)}')
    L.append(f'  at 3 s (where it settles; negative = in our favour): mean {Z.END.mean():.2f}; below the 2 s peak {float((Z.END < Z.P2).mean()):.1%} of onsets;'
             f' time to the 2 s peak, median by its size: ' + ', '.join(f'{j} tick {Z[Z.P2 == j].TP.median():.0f} ms' for j in (1, 2, 3)))
    for col, bins, nm in (('gap_adv', ((-1e9, -1), (-1, 0), (0, 1), (1, 2), (2, 4), (4, 1e9)), 'fair price minus Predict mid, against us (c)'),
                          ('dfv250', ((-1e9, 0), (0, 0.5), (0.5, 1), (1, 2), (2, 1e9)), 'fair-price move in the last 250 ms, against us (c)')):
        L.append(f'  by {nm}:')
        for lo, hi in bins:
            x = Z[(Z[col] >= lo) & (Z[col] < hi)]
            if len(x):
                L.append(f'    [{lo:g}, {hi:g}) {len(x) / len(Z):5.1%} of onsets: {dist(x.P2.values)}; mean {x.P2.mean():.2f}; moves of >= 1 tick stop at 1:'
                         f' {float((x[x.P2 >= 1].P2 == 1).mean()) if (x.P2 >= 1).any() else NAN:5.1%}')
    sp = {}; up = dn = 0
    for d in DAYS9:
        x = Z[Z.day == int(d)]; sp[d] = float(spearmanr(x.gap_adv.values, x.P2.values, nan_policy='omit')[0]) if len(x) > 30 else NAN
        up += int(sp[d] > 0); dn += int(sp[d] < 0)
    sp2 = float(spearmanr(Z.dfv250.values, Z.P2.values, nan_policy='omit')[0])
    R_['R591'] = dict(per_day=sp, up=up, down=dn, label='FOLLOWS' if up >= 7 else ('REVERSE' if dn >= 7 else 'NOT-SHOWN'))
    L.append(f'  R591: Spearman(gap_adv, peak) per day ' + ', '.join(f'{d[4:]} {sp[d]:+.3f}' for d in DAYS9) + f' -> {R_["R591"]["label"]}; dfv250 pooled {sp2:+.3f}')
    L.append('')
    L.append('AUC OUT OF DAY at onsets, new target (farthest within 2 s >= j; onsets not ALREADY_j): pooled | per day')
    au = {}
    for j in JS:
        Y = A[(A.D0 < j) & A.STEP_P1.notna()]; y = Y.P2.values >= j
        for arm in ('FULL', 'NOINF', 'INFONLY', 'STEP'):
            c = f'{arm}_P{j}'
            if c not in Y:
                continue
            pooled = PD.auc(y, Y[c].values); per = {d: PD.auc(y[Y.day.values == int(d)], Y[c].values[Y.day.values == int(d)]) for d in DAYS9}
            au[(arm, j)] = (pooled, per)
            L.append(f'  j{j} {arm:<7} {pooled:.3f} | ' + ' '.join(f'{per[d]:.3f}' for d in DAYS9) + f'  (base rate {float(y.mean()):.3f}, n {len(Y)})')
    for j in JS_ABL:
        pf = au[('FULL', j)][1]
        for key, other, good, badl in (('R592', 'NOINF', 'HELPS', 'HURTS'), ('R593', 'STEP', 'BETTER', 'WORSE')):
            po = au[(other, j)][1]; u = sum(pf[d] > po[d] for d in DAYS9); dn_ = sum(pf[d] < po[d] for d in DAYS9)
            R_[f'{key}_j{j}'] = dict(full=au[('FULL', j)][0], other=au[(other, j)][0], up=u, down=dn_, label=good if u >= 7 else (badl if dn_ >= 7 else 'NOT-SHOWN'))
    L.append('  ' + '; '.join(f'{k}: {v["label"]} ({v["up"]}/9)' for k, v in R_.items() if k[:4] in ('R592', 'R593')))
    L.append('')
    L.append('CALIBRATION of FULL (deciles: mean predicted / realized)')
    for j in (1, 2, 3):
        Y = A[A.D0 < j]; p = Y[f'FULL_P{j}'].values; yv = (Y.P2.values >= j).astype(float)
        qd = np.quantile(p, np.linspace(0.1, 0.9, 9)); b = np.searchsorted(qd, p)
        L.append(f'  j{j}: ' + ' '.join(f'{p[b == i].mean():.2f}/{yv[b == i].mean():.2f}' for i in range(10) if (b == i).any()))
    L.append('')
    L.append('KEEP DECISION at onsets: kept onsets split into not reached / stopped at the rung / run over (farthest within 2 s)')
    for dd in (1, 2):
        j = dd + 1; Y = A[(A.D0 < j) & A.STEP_P1.notna()]
        ks = Y[f'STEP_P{j}'].values < TAU_STEP; kf = np.zeros(len(Y), bool)
        for d_ in DAYS9:
            ix = np.flatnonzero(Y.day.values == int(d_)); nk = int(ks[ix].sum())
            if nk:
                kf[ix[np.argsort(Y[f'FULL_P{j}'].values[ix], kind='stable')[:nk]]] = True
        L.append(f'  rung {dd} (onsets {len(Y)}):')
        for nm, kk in (('STEP (FB25)', ks), ('FULL (same count)', kf), ('realized', Y.P2.values < j)):
            Dk = Y.P2.values[kk]
            L.append(f'    {nm:<18} kept {kk.mean():5.1%}; not reached {float((Dk < dd).mean()):5.1%} | stopped {float((Dk == dd).mean()):5.1%} | run over {float((Dk >= j).mean()):5.1%}')
        lo = hi = 0
        for d_ in DAYS9:
            ix = Y.day.values == int(d_)
            rs_ = float((Y.P2.values[ix & ks] >= j).mean()) if (ix & ks).any() else NAN
            rf_ = float((Y.P2.values[ix & kf] >= j).mean()) if (ix & kf).any() else NAN
            lo += int(rf_ < rs_); hi += int(rf_ > rs_)
        R_[f'R594_d{dd}'] = dict(lower=lo, higher=hi, label='FEWER-RUN-OVERS' if lo >= 7 else ('MORE-RUN-OVERS' if hi >= 7 else 'NOT-SHOWN'))
        L.append(f'    R594_d{dd}: FULL lower on {lo}, higher on {hi} of 9 -> {R_[f"R594_d{dd}"]["label"]}')
    L.append('')
    gains = {}
    for d in DAYS9:
        fj = json.load(open(f'{OUTD}/fit_{d}.json'))
        for j in (1, 2):
            for f in fj['info'][f'j{j}']['top_gain'][:8]:
                gains[(j, f)] = gains.get((j, f), 0) + 1
    for j in (1, 2):
        top = sorted([(c, f) for (jj, f), c in gains.items() if jj == j], reverse=True)[:12]
        L.append(f'FULL top inputs by gain, j{j} (days in the top 8): ' + ', '.join(f'{f} {c}' for c, f in top))
    L.append('')
    L.append('READINGS: ' + json.dumps({k: v['label'] for k, v in R_.items()}))
    txt = '\n'.join(L) + '\n'
    open(f'{OUTD}/influence.txt', 'w').write(txt)
    json.dump(dict(asof=asof, readings=R_, auc={f'{a}|{j}': dict(pooled=v[0], per_day=v[1]) for (a, j), v in au.items()}),
              open(f'{OUTD}/influence.json', 'w'), indent=1, default=float)
    print(txt)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        all_selftests()
    elif cmd == 'rows':
        rows_day(sys.argv[2])
    elif cmd == 'fit':
        fit_day(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit('usage: pf_influence.py selftest | rows <day> | fit <day> | report')
