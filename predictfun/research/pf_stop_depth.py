"""STOP-DEPTH (2026-11-14, session ce68ebe0; USER: "Ok, so we need to check how large is the influence of the signal, then we just need to pull
50c bid, if we estimate it will stop 1c, we can rest 49c there and repost 50c ask" / "Makes sense. Proceed"). Step 1 of the plan: a better
model of WHERE A DANGER MOVE STOPS, checked offline before any simulation. FUTURE-BEST's rule (keep a rung d ticks behind the best unless
P(move >= d + 1) >= tau) is limited by this prediction: with the realized move it made +$378 a day more than the guard ladder on WS trades,
with PULL-DEPTH's model +$41-71 (not robust).

Rows: as PULL-DEPTH PART A - every 50 ms grid step of the first 60 s with the side's guard on (v <= -0.10) and a valid known book, 9 consumed
replay days (10-23..27, 10-31, 11-01..03) - except steps with the guard value masked (v = -1: the window's start price is not known yet and
the raw features read it ahead, FINDINGS P16), which are a status here.
Target (fixes PULL-DEPTH's, per the other session's review: known-frame target, missing-frame zeros), on the venue frames, in ticks the side's
best goes against us beyond the known frame's best (bids: down; asks: up), invalid frames skipped:
  D0  over frames stamped after the known frame up to t + 22 ms (our cancel's landing): moves that already happened or cannot be avoided
  D1  over the state at t + 22 ms (the last valid frame at or before it; no new frame = no change) and frames stamped in (t + 22, t + 1000] ms
  for a threshold j: status ALREADY_j if D0 >= j (a rung there is past saving), else y_j = [D1 >= j].
Inputs (no fitted model's output: FRONT-PULL's P (fitted with the held-out day: PULL-DEPTH's nested-fold leak) and the guard value v (its
models' fits include 8 of the 9 days) are dropped):
  BASE  PULL-DEPTH's inputs without P and v: the 32 step-5 features, own / opposite 250 ms L1 size change, Binance activity, Binance imbalance
        change, the side, the known frame's sizes at levels 1-5, tick gaps of levels 2-5, the spread, cumulative size to levels 1-3, steps since
        the guard came on
  NEW   BASE + the directional features signed against our side (adv x raw; adv = -1 bids, +1 asks: Binance returns 25 ms-3 s, imbalance,
        flows 250 ms-5 s, imbalance change; Predict micro-price deviation, L1 / L3 imbalance, mid changes 250 ms-5 s), own / opposite L1 size
        and pending settlements (by side), size within 1..5 ticks of the best on our side (what a move must clear), and our side's and the
        opposite side's best moves against us over the last 250 ms / 1 s on the known frames
Models: LightGBM binary for D1 >= j (j = 1..5 NEW, 1..3 BASE), PULL-DEPTH's pinned parameters and rounds, LEAVE-ONE-DAY-OUT; training rows:
45,000 per training day drawn with a fixed seed, then ALREADY_j rows dropped and at most 300,000 kept; predictions made cumulative
(P_j = min(P_j, P_j-1)). OLD = PULL-DEPTH's saved predictions (the model FUTURE-BEST used), evaluated on the same rows and clean target.
Readings DECLARED BEFORE THE FIRST RUN (out of day; rows not ALREADY_j; day rule >= 7 of 9):
 R580 CLEAN-PREDICTABLE (NEW, j = 1, 2, 3 each): AUC >= 0.70 pooled and >= 0.65 on >= 7 days -> PREDICTABLE; else WEAK.
 R581 NEW-VS-OLD (j = 1, 2, 3 each): NEW's AUC above OLD's on >= 7 days -> BETTER; below on >= 7 -> WORSE; else NOT-SHOWN.
 R582 FEATURES (j = 1, 2, 3 each): NEW above BASE on >= 7 days -> FEATURES-HELP; below on >= 7 -> FEATURES-HURT; else NOT-SHOWN.
 R583 KEEP-QUALITY (rung d = 1 and d = 2 each; the USER's case is d = 1: keep 49c when 50c is the best): on each held-out day OLD keeps rung d
      at the steps with P_{d+1} < 0.25 (FUTURE-BEST's FB25 line); NEW keeps the same number of steps, those with the lowest P_{d+1}. The share
      of kept steps where the move goes through the rung (y_{d+1} = 1): NEW lower on >= 7 days -> FEWER-RUN-OVERS; higher on >= 7 ->
      MORE-RUN-OVERS; else NOT-SHOWN.
Printed, not readings: how far moves go after the signal (D1 among rows with D0 = 0: 0 / 1 / 2 / 3 / 4 / 5+ ticks; where moves of >= 1 tick
stop), by guard-value quintile, onset vs continuing steps, and by the Binance 1 s return against us; AUC for j = 4, 5; calibration (deciles)
of NEW; the kept steps split into not reached / stopped at the rung (the good case) / run over, for OLD, NEW and the realized move; FB10's line
(tau 0.10); statuses (masked, invalid, ALREADY_j, quiet = no new frame in the horizon).
Controls (any failure -> no output): synthetic cases for the target (both sides, already-happened moves, invalid frames, the horizon, a quiet
book), the signed and per-tick features and the recent best moves, with planted errors (move sign flipped; frames before t + 22 ms counted in
D1; the side sign of adv dropped) that must each fail a case; K.G32 equals the step-5 list in this order; the fit sees no row of its held-out
day (asserted); every row matches an OLD prediction or is counted; memory self-guard (2 GiB).
Limits: OUR signal's population (guard-on steps); the 1 s horizon is PULL-DEPTH's; consumed days and the inputs were chosen after PULL-DEPTH /
FUTURE-BEST on these days (exploratory: a better model is a candidate for the replay of step 2, not a result); AUC ranks rows, the replay
values them.
Usage (run_light, PF_SLOT=claude): pf_stop_depth.py selftest | rows <day> | fit <day> | report | posthoc; driver run_stop_depth.sh
(selftest, rows, fits, report). 'posthoc' is POST HOC, added after the first report (04:22Z): the broken Binance-return quintiles
replaced by value bins, and where moves stop by signal strength (see posthoc()).
"""
import bisect
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_kp_s4 as K                                      # noqa: E402
import pf_qr_queue_race as QR                             # noqa: E402  (runs: valid frames)
import pf_pull_depth as PD                                # noqa: E402  (feats, auc, LGB_PARAMS, N_ROUNDS, OUTD of OLD)
import pf_model_spec as MS                                # noqa: E402

DAYS9 = K.DAYS9
OUTD = f'{K.IO}/stop_depth_20261114'
JS = (1, 2, 3, 4, 5); JS_BASE = (1, 2, 3); H_MS = 1000.0; MASK_V = -1.0
SUB_DAY = 45000; SUB_N = 300000
TAU_OLD = 0.25; TAU_FB10 = 0.10
G32N = list(MS.SETS['ALL'])
X37 = G32N + ['v', 'own_d250', 'opp_d250', 'act_rel', 'd_bnimb250']
ADV = ['bn_r25', 'bn_r50', 'bn_r100', 'bn_r250', 'bn_r500', 'bn_r1000', 'bn_r3000', 'bn_imb', 'bn_flow250', 'bn_flow1000', 'bn_flow5000',
       'pf_micro_dev', 'pf_imb1', 'pf_imb3', 'pf_dm250', 'pf_dm1000', 'pf_dm5000', 'd_bnimb250']
PDF = [f'x{i}' for i in range(37)] + ['fside', 'P', 's1', 's2', 's3', 's4', 's5', 'g2', 'g3', 'g4', 'g5', 'spread', 'c1', 'c2', 'c3', 'gage']
BASE_FEAT = [f'x{i}' for i in range(37) if i != 32] + ['fside', 's1', 's2', 's3', 's4', 's5', 'g2', 'g3', 'g4', 'g5', 'spread', 'c1', 'c2', 'c3', 'gage']
NEW_EXTRA = [f'adv_{c}' for c in ADV] + ['own_l1', 'opp_l1', 'own_pend', 'opp_pend', 'ct1', 'ct2', 'ct3', 'ct4', 'ct5',
                                         'own_mv250', 'own_mv1000', 'opp_mv250', 'opp_mv1000']
NEW_FEAT = BASE_FEAT + NEW_EXTRA
INF = float('inf'); NAN = float('nan')
MUT_SIGN = MUT_EARLY_IN_D1 = MUT_ADV_NOSIDE = 0           # planted errors (self-test only)
MEM_CAP_MIB = 2048.0


# ---------------------------------------------------------------- target and features
def move_clean(b0, valid, tfl, side, jk, t, lc=None, h=H_MS):
    """(D0, D1, quiet) in ticks against us beyond the known frame jk's best (header)."""
    lc = K.L_CANCEL if lc is None else lc
    if jk < 0 or not valid[jk]:
        return NAN, NAN, False
    ref = int(b0[jk])

    def adv(seg):
        if not len(seg):
            return 0
        m = (ref - int(seg.min())) if side == 0 else (int(seg.max()) - ref)
        return -m if MUT_SIGN else m
    j0 = bisect.bisect_right(tfl, t + lc); j1 = bisect.bisect_right(tfl, t + h)
    s0 = b0[jk + 1:j0][valid[jk + 1:j0]]
    D0 = max(0, adv(s0))
    jl = j0 - 1
    while jl > jk and not valid[jl]:
        jl -= 1
    lo = (jk + 1) if MUT_EARLY_IN_D1 else j0
    s1 = b0[lo:j1][valid[lo:j1]]
    D1 = max(0, adv(np.concatenate([[b0[jl]], s1])))
    return float(D0), float(D1), bool(len(s1) == 0)


def per_tick(sz, gaps):
    """Size within 1..5 ticks of the best on our side: level i (size sz[i], tick gap gaps[i], level 1 gap 0; -1 = no level) counts toward
    ct_j when its gap <= j - 1."""
    return [float(sum(s for s, g in zip(sz, gaps) if 0 <= g <= j - 1)) for j in range(1, 6)]


def best_moves(w, k, side):
    """Our side's and the opposite side's best moves against us (ticks) from the frame known 250 ms / 1 s before the step to the known frame."""
    kf, jk = w['kf'], int(w['jk'][k]); t = float(w['tg'][k]); b0 = w['P5'][:, 0]; opp = w['opp']; out = []
    for h in (250.0, 1000.0):
        jp = int(np.searchsorted(kf, t - h, side='right')) - 1
        if jk < 0 or jp < 0 or b0[jk] < 0 or b0[jp] < 0:
            out.append(NAN)
        else:
            out.append(float((b0[jp] - b0[jk]) if side == 0 else (b0[jk] - b0[jp])))
    for h in (250.0, 1000.0):
        jp = int(np.searchsorted(kf, t - h, side='right')) - 1
        if jk < 0 or jp < 0 or opp[jk] < 0 or opp[jp] < 0:
            out.append(NAN)
        else:
            out.append(float((opp[jp] - opp[jk]) if side == 0 else (opp[jk] - opp[jp])))
    return out[0], out[1], out[2], out[3]


def extra_feats(x37, side, sz, gaps):
    """NEW's additions from the step's 37 inputs (named X37) and the known frame's levels; best moves are appended by the caller."""
    adv = 1.0 if side == 1 else -1.0
    if MUT_ADV_NOSIDE:
        adv = 1.0
    d = dict(zip(X37, x37))
    sgn = [adv * float(d[c]) for c in ADV]
    own_l1, opp_l1 = (d['pf_lbs'], d['pf_las']) if side == 0 else (d['pf_las'], d['pf_lbs'])
    own_p, opp_p = (d['pf_pend_bid1000'], d['pf_pend_ask1000']) if side == 0 else (d['pf_pend_ask1000'], d['pf_pend_bid1000'])
    return sgn + [float(own_l1), float(opp_l1), float(own_p), float(opp_p)] + per_tick(sz, gaps)


# ---------------------------------------------------------------- self-tests
def selftest():
    bad = []

    def chk(nm, c):
        if not c:
            bad.append(nm)
    chk('G32 order', list(K.G32) == G32N and len(X37) == 37 and len(BASE_FEAT) == 51 and len(NEW_FEAT) == 82)
    tfl = [0.0, 100.0, 150.0, 200.0, 900.0, 1500.0]; valid = np.ones(6, bool)
    b0 = np.array([50, 50, 49, 49, 48, 47])
    D0, D1, q = move_clean(b0, valid, tfl, 0, 0, 120.0, lc=22.0)
    chk(f'bid move {D0} {D1}', D0 == 0 and D1 == 2 and not q)
    b0e = np.array([50, 48, 50, 50, 50, 50])
    D0, D1, q = move_clean(b0e, valid, tfl, 0, 0, 120.0, lc=22.0)
    chk(f'already {D0} {D1}', D0 == 2 and D1 == 2)
    b0r = np.array([50, 48, 50, 50, 50, 50]); tfr = [0.0, 100.0, 130.0, 200.0, 900.0, 1500.0]
    D0, D1, q = move_clean(b0r, valid, tfr, 0, 0, 120.0, lc=22.0)
    chk(f'early move not in D1 {D0} {D1}', D0 == 2 and D1 == 0)
    a0 = np.array([51, 51, 52, 53, 53, 60])
    D0, D1, q = move_clean(a0, valid, tfl, 1, 0, 120.0, lc=22.0)
    chk(f'ask move {D0} {D1}', D0 == 0 and D1 == 2)
    vinv = np.array([True, True, False, True, True, True]); b0i = np.array([50, 50, 40, 50, 49, 47])
    D0, D1, q = move_clean(b0i, vinv, tfl, 0, 0, 120.0, lc=22.0)
    chk(f'invalid frame skipped {D1}', D1 == 1)
    D0, D1, q = move_clean(np.array([50, 50, 50, 50, 50, 50]), valid, [0.0, 10.0, 20.0, 30.0, 40.0, 50.0], 0, 2, 100.0, lc=22.0)
    chk(f'quiet {D0} {D1} {q}', D0 == 0 and D1 == 0 and q)
    chk('per tick', per_tick([10, 20, 30, 40, 50], [0, 1, 3, 4, 6]) == [10.0, 30.0, 30.0, 60.0, 100.0])
    x = np.zeros(37); x[X37.index('bn_r250')] = 0.5; x[X37.index('pf_lbs')] = 2.0; x[X37.index('pf_las')] = 3.0
    x[X37.index('pf_pend_bid1000')] = 7.0
    e0 = extra_feats(x, 0, [1, 1, 1, 1, 1], [0, 1, 2, 3, 4]); e1 = extra_feats(x, 1, [1, 1, 1, 1, 1], [0, 1, 2, 3, 4])
    i_r = ADV.index('bn_r250'); n_a = len(ADV)
    chk(f'adv sign {e0[i_r]} {e1[i_r]}', e0[i_r] == -0.5 and e1[i_r] == 0.5)
    chk('own / opp', e0[n_a:n_a + 4] == [2.0, 3.0, 7.0, 0.0] and e1[n_a:n_a + 4] == [3.0, 2.0, 0.0, 7.0])
    w = dict(kf=np.array([0.0, 100.0, 300.0, 1100.0]), jk=np.array([3]), tg=np.array([1200.0]),
             P5=np.array([[50, 49, 48, 47, 46], [50, 49, 48, 47, 46], [49, 48, 47, 46, 45], [48, 47, 46, 45, 44]]), opp=np.array([51, 51, 50, 49]))
    m = best_moves(w, 0, 0)
    chk(f'best moves {m}', m == (1.0, 2.0, 1.0, 2.0))
    return bad


def mutation_check():
    global MUT_SIGN, MUT_EARLY_IN_D1, MUT_ADV_NOSIDE
    out = {}
    for nm in ('MUT_SIGN', 'MUT_EARLY_IN_D1', 'MUT_ADV_NOSIDE'):
        globals()[nm] = 1
        try:
            out[nm] = len(selftest()) > 0
        finally:
            globals()[nm] = 0
    return out


def all_selftests():
    bad = selftest()
    if bad:
        sys.exit(f'STOP-DEPTH SELFTEST FAILED: {bad}')
    m = mutation_check()
    if not all(m.values()):
        sys.exit(f'STOP-DEPTH PLANTED ERROR NOT CAUGHT: {m}')
    print(f'selftests passed; planted errors caught: {m}', flush=True)
    return m


# ---------------------------------------------------------------- rows
def rows_day(day):
    all_selftests()
    T0 = time.time(); D = K.load_day(day)
    out = []; st = dict(window_sides=0, guard_steps=0, masked=0, invalid_book=0, quiet=0)
    nanP = None
    for w in K.window_sides(D):
        st['window_sides'] += 1; n = w['n']
        if not n:
            continue
        side = int(w['side']); vv = np.asarray(w['vkl'], float); vg = vv <= K.GUARD_PULL
        b0, valid, _, _ = QR.runs(w['tf'], w['P5'], w['opp'], side); tfl = w['tf'].tolist(); gage = 0
        nanP = np.full(n, np.nan); rw = []
        for k in range(n):
            if not vg[k]:
                gage = 0; continue
            gage += 1; st['guard_steps'] += 1
            if vv[k] == MASK_V:
                st['masked'] += 1; continue
            jk = int(w['jk'][k])
            if jk < 0 or not valid[jk] or w['best'][k] < 0:
                st['invalid_book'] += 1; continue
            t = float(w['tg'][k])
            f = PD.feats(w, k, nanP, gage)                 # PULL-DEPTH's 53 inputs (P is NaN here and not used)
            fd = dict(zip(PDF, f))
            sz = [fd[f's{i}'] for i in range(1, 6)]; gaps = [0.0] + [fd[f'g{i}'] for i in range(2, 6)]
            ex = extra_feats(np.asarray(w['X'][k], float), side, sz, gaps) + list(best_moves(w, k, side))
            D0, D1, q = move_clean(b0, valid, tfl, side, jk, t)
            st['quiet'] += int(q)
            rw.append([int(day), int(w['m']), side, k, t, int(gage == 1), float(vv[k])] + [fd[c] for c in BASE_FEAT] + ex + [D0, D1])
        if rw:
            out.append(np.asarray(rw, dtype=np.float64))
        if st['window_sides'] % 80 == 0:
            r2 = K.rss_mib()
            if r2 > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r2:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    cols = ['day', 'm', 'side', 'k', 't', 'onset', 'v'] + NEW_FEAT + ['D0', 'D1']
    R = pd.DataFrame(np.vstack(out) if out else np.zeros((0, len(cols))), columns=cols)
    R = R.astype({'day': 'int64', 'm': 'int64', 'side': 'int64', 'k': 'int64', 'onset': 'int64', **{c: 'float32' for c in NEW_FEAT + ['v']}})
    os.makedirs(OUTD, exist_ok=True)
    R.to_parquet(f'{OUTD}/rows_{day}.parquet', index=False)
    st['rows'] = len(R); json.dump(st, open(f'{OUTD}/rows_{day}.json', 'w'))
    print(f'{day}: {st} in {time.time() - T0:.0f} s', flush=True)


# ---------------------------------------------------------------- fit (one held-out day)
def fit_day(d):
    import lightgbm as lgb
    all_selftests()
    T0 = time.time(); cols = ['day', 'm', 'side', 'k', 'D0', 'D1'] + NEW_FEAT
    parts = []
    for x in DAYS9:
        if x == d:
            continue
        R = pd.read_parquet(f'{OUTD}/rows_{x}.parquet', columns=cols)
        rng = np.random.default_rng(int(d) * 31 + int(x))
        parts.append(R.iloc[np.sort(rng.choice(len(R), min(len(R), SUB_DAY), replace=False))])
        del R
    tr = pd.concat(parts, ignore_index=True); del parts
    assert not (tr.day == int(d)).any(), 'held-out day in the training rows'
    te = pd.read_parquet(f'{OUTD}/rows_{d}.parquet', columns=cols)
    pred = dict(m=te.m.values, side=te.side.values, k=te.k.values); prev = {'NEW': np.ones(len(te)), 'BASE': np.ones(len(te))}
    info = {}
    for j in JS:
        ok = (tr.D0.values < j) & np.isfinite(tr.D1.values)
        trj = tr[ok]
        if len(trj) > SUB_N:
            trj = trj.iloc[np.sort(np.random.default_rng(int(d) * 7 + j).choice(len(trj), SUB_N, replace=False))]
        y = (trj.D1.values >= j).astype(int); info[f'j{j}'] = dict(train=len(trj), base_rate=float(y.mean()))
        for arm, F in (('NEW', NEW_FEAT), ('BASE', BASE_FEAT)):
            if arm == 'BASE' and j not in JS_BASE:
                continue
            b = lgb.train(PD.LGB_PARAMS, lgb.Dataset(trj[F].values.astype(np.float32), y, feature_name=F), PD.N_ROUNDS)
            p = np.minimum(b.predict(te[F].values.astype(np.float32)), prev[arm]); prev[arm] = p; pred[f'{arm}_P{j}'] = p
            if arm == 'NEW':
                os.makedirs(f'{OUTD}/models', exist_ok=True); b.save_model(f'{OUTD}/models/new_{d}_j{j}.txt')
                imp = b.feature_importance('gain'); info[f'j{j}']['top_gain'] = [F[i] for i in np.argsort(-imp)[:12]]
        r2 = K.rss_mib()
        if r2 > MEM_CAP_MIB:
            sys.exit(f'REFUSED: resident memory {r2:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    pd.DataFrame(pred).to_parquet(f'{OUTD}/pred_{d}.parquet', index=False)
    json.dump(dict(info=info, train_rows=len(tr), params=PD.LGB_PARAMS, rounds=PD.N_ROUNDS, sub_day=SUB_DAY, sub_n=SUB_N),
              open(f'{OUTD}/fit_{d}.json', 'w'), indent=1)
    print(f'fit {d}: {info} in {time.time() - T0:.0f} s', flush=True)


# ---------------------------------------------------------------- report
def report():
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    rs = {d: json.load(open(f'{OUTD}/rows_{d}.json')) for d in DAYS9}
    A = []
    unmatched = 0
    for d in DAYS9:
        R = pd.read_parquet(f'{OUTD}/rows_{d}.parquet', columns=['day', 'm', 'side', 'k', 'onset', 'v', 'adv_bn_r1000', 'D0', 'D1'])
        P = pd.read_parquet(f'{OUTD}/pred_{d}.parquet')
        O = pd.read_parquet(f'{PD.OUTD}/pred_{d}.parquet').rename(columns={f'P{j}': f'OLD_P{j}' for j in JS})
        R = R.merge(P, on=['m', 'side', 'k'], how='left', validate='one_to_one').merge(O, on=['m', 'side', 'k'], how='left', validate='one_to_one')
        unmatched += int(R.OLD_P1.isna().sum())
        A.append(R)
    A = pd.concat(A, ignore_index=True)
    R_ = {}
    L = [f'STOP-DEPTH, 9 consumed replay days, first 60 s, guard-on steps; as of {asof}; controls passed; rows {len(A)}; masked steps excluded '
         f'{sum(r["masked"] for r in rs.values())}; invalid book {sum(r["invalid_book"] for r in rs.values())}; quiet (no new frame in the horizon) '
         f'{sum(r["quiet"] for r in rs.values())}; rows without an OLD prediction {unmatched}', '']
    # how far moves go
    Z = A[A.D0 == 0]
    dist = lambda x: ' | '.join(f'{lab} {v:5.1%}' for lab, v in zip(('0', '1', '2', '3', '4', '5+'),
                                                                     [float((np.minimum(x, 5) == i).mean()) for i in range(6)]))
    L.append(f'HOW FAR THE MOVE GOES within 1 s after a guard-on step (ticks against us, venue frames after t + 22 ms; rows with nothing moved yet, n {len(Z)})')
    L.append(f'  all            {dist(Z.D1.values)}')
    L.append(f'  onset steps    {dist(Z[Z.onset == 1].D1.values)}  (n {int((Z.onset == 1).sum())})')
    L.append(f'  later steps    {dist(Z[Z.onset == 0].D1.values)}')
    qv = np.nanquantile(Z.v.values, [0.2, 0.4, 0.6, 0.8]); bv = np.searchsorted(qv, Z.v.values)
    for i in range(5):
        L.append(f'  guard value Q{i + 1}{" (most negative)" if i == 0 else ""} {dist(Z.D1.values[bv == i])}')
    qr = np.nanquantile(Z.adv_bn_r1000.values, [0.2, 0.4, 0.6, 0.8]); br = np.searchsorted(qr, Z.adv_bn_r1000.values)
    for i in range(5):
        L.append(f'  Binance 1 s return against us Q{i + 1}{" (most against)" if i == 4 else ""} {dist(Z.D1.values[br == i])}')
    mv = Z[Z.D1 >= 1].D1.values
    L.append(f'  moves of >= 1 tick (n {len(mv)}) stop at 1 tick {float((mv == 1).mean()):.1%}, 2 ticks {float((mv == 2).mean()):.1%}, 3+ {float((mv >= 3).mean()):.1%}')
    L.append('')
    # AUC
    L.append('AUC OUT OF DAY on the clean target (rows not ALREADY_j): pooled | per day; OLD = PULL-DEPTH\'s model (FUTURE-BEST\'s), BASE = its inputs without P / v, NEW')
    auc_day = {}
    for j in JS:
        Y = A[(A.D0 < j) & A.OLD_P1.notna()]; y = (Y.D1.values >= j)
        arms = ['OLD', 'NEW'] + (['BASE'] if j in JS_BASE else [])
        for a in arms:
            pooled = PD.auc(y, Y[f'{a}_P{j}'].values)
            per = {d: PD.auc((Y.D1.values >= j)[Y.day.values == int(d)], Y[f'{a}_P{j}'].values[Y.day.values == int(d)]) for d in DAYS9}
            auc_day[(a, j)] = (pooled, per)
            L.append(f'  j{j} {a:<4} {pooled:.3f} | ' + ' '.join(f'{per[d]:.3f}' for d in DAYS9) + f'  (base rate {float(y.mean()):.3f}, n {len(Y)})')
    for j in JS_BASE:
        pooled, per = auc_day[('NEW', j)]
        lab = 'PREDICTABLE' if pooled >= 0.70 and sum(per[d] >= 0.65 for d in DAYS9) >= 7 else 'WEAK'
        R_[f'R580_j{j}'] = dict(pooled=pooled, days=sum(per[d] >= 0.65 for d in DAYS9), label=lab)
        for key, other, good, badl in (('R581', 'OLD', 'BETTER', 'WORSE'), ('R582', 'BASE', 'FEATURES-HELP', 'FEATURES-HURT')):
            po = auc_day[(other, j)][1]
            up = sum(per[d] > po[d] for d in DAYS9); dn = sum(per[d] < po[d] for d in DAYS9)
            R_[f'{key}_j{j}'] = dict(up=up, down=dn, new=pooled, other=auc_day[(other, j)][0], label=good if up >= 7 else (badl if dn >= 7 else 'NOT-SHOWN'))
    L.append('  ' + '; '.join(f'{k}: {v["label"]}' for k, v in R_.items()))
    L.append('')
    # calibration
    L.append('CALIBRATION of NEW (deciles of the prediction: mean predicted / realized)')
    for j in (1, 2, 3):
        Y = A[A.D0 < j]; p = Y[f'NEW_P{j}'].values; y = (Y.D1.values >= j).astype(float)
        qd = np.quantile(p, np.linspace(0.1, 0.9, 9)); b = np.searchsorted(qd, p)
        L.append(f'  j{j}: ' + ' '.join(f'{p[b == i].mean():.2f}/{y[b == i].mean():.2f}' for i in range(10)))
    L.append('')
    # decision level
    L.append('KEEP DECISION for rung d (keep while the guard is on): kept steps split into not reached (D1 < d) / stopped at the rung (D1 = d, the good case) / run over (D1 >= d + 1)')
    for dd in (1, 2, 3, 4):
        j = dd + 1; Y = A[(A.D0 < j) & A.OLD_P1.notna()]
        rows_ = []
        for tau, nm in ((TAU_OLD, 'FB25'), (TAU_FB10, 'FB10')):
            ko_all = Y[f'OLD_P{j}'].values < tau
            kn_all = np.zeros(len(Y), bool)
            for d_ in DAYS9:
                ix = np.flatnonzero(Y.day.values == int(d_)); nk = int(ko_all[ix].sum())
                if nk:
                    kn_all[ix[np.argsort(Y[f'NEW_P{j}'].values[ix], kind='stable')[:nk]]] = True
            for nm2, kk in (('OLD', ko_all), ('NEW', kn_all), ('realized', Y.D1.values < j)):
                if nm2 == 'realized' and nm != 'FB25':
                    continue
                Dk = Y.D1.values[kk]
                rows_.append(f'{nm} {nm2}: kept {kk.mean():5.1%} of steps; not reached {float((Dk < dd).mean()):5.1%} | stopped {float((Dk == dd).mean()):5.1%} | run over {float((Dk >= j).mean()):5.1%}')
            if nm == 'FB25' and dd in (1, 2):
                lo = hi = 0
                for d_ in DAYS9:
                    ix = Y.day.values == int(d_)
                    ro = float((Y.D1.values[ix & ko_all] >= j).mean()) if (ix & ko_all).any() else NAN
                    rn = float((Y.D1.values[ix & kn_all] >= j).mean()) if (ix & kn_all).any() else NAN
                    lo += int(rn < ro); hi += int(rn > ro)
                R_[f'R583_d{dd}'] = dict(lower=lo, higher=hi, label='FEWER-RUN-OVERS' if lo >= 7 else ('MORE-RUN-OVERS' if hi >= 7 else 'NOT-SHOWN'))
        L.append(f'  rung {dd} (n {len(Y)}):')
        for r_ in rows_:
            L.append(f'    {r_}')
    L.append('  ' + '; '.join(f'{k}: {v["label"]} ({v["lower"]} lower / {v["higher"]} higher)' for k, v in R_.items() if k.startswith('R583')))
    L.append('')
    gains = {}
    for d in DAYS9:
        fj = json.load(open(f'{OUTD}/fit_{d}.json'))
        for j in (1, 2):
            for f in fj['info'][f'j{j}'].get('top_gain', [])[:8]:
                gains[(j, f)] = gains.get((j, f), 0) + 1
    for j in (1, 2):
        top = sorted([(c, f) for (jj, f), c in gains.items() if jj == j], reverse=True)[:12]
        L.append(f'NEW top inputs by gain, j{j} (days in the top 8): ' + ', '.join(f'{f} {c}' for c, f in top))
    L.append('')
    L.append('READINGS: ' + json.dumps({k: v['label'] for k, v in R_.items()}))
    txt = '\n'.join(L) + '\n'
    open(f'{OUTD}/stop_depth.txt', 'w').write(txt)
    json.dump(dict(asof=asof, readings=R_, auc={f'{a}|{j}': dict(pooled=v[0], per_day=v[1]) for (a, j), v in auc_day.items()}),
              open(f'{OUTD}/stop_depth.json', 'w'), indent=1, default=float)
    print(txt)


def posthoc():
    """POST HOC, added after the first report (2026-11-14 04:22Z), descriptive, no readings. The report's Binance-return quintiles were
    broken: 77% of the 1 s returns against us are exactly 0, so every quintile line was 0 (Q2-Q4 empty, 'Q5' = any move against us). Here
    value bins instead (in our favour < 0 / zero / against us in terciles of the positive values) for the 1 s and 250 ms returns, and where
    moves of >= 1 tick stop by guard-value quintile and by those bins."""
    Z = pd.concat([pd.read_parquet(f'{OUTD}/rows_{d}.parquet', columns=['v', 'onset', 'adv_bn_r1000', 'adv_bn_r250', 'D0', 'D1']) for d in DAYS9],
                  ignore_index=True)
    Z = Z[Z.D0 == 0]
    dist = lambda x: ' | '.join(f'{lab} {v:5.1%}' for lab, v in zip(('0', '1', '2', '3', '4', '5+'),
                                                                     [float((np.minimum(x, 5) == i).mean()) for i in range(6)]))
    stop = lambda x: (f'stop at 1 tick {float((x == 1).mean()):5.1%}, 2 ticks {float((x == 2).mean()):5.1%}, 3+ {float((x >= 3).mean()):5.1%}'
                      f' (n {len(x)})') if len(x) else 'n/a'
    L = [f'STOP-DEPTH POST HOC (after the first report; descriptive, no readings): guard-on steps with nothing moved yet, n {len(Z)}; as of '
         f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}', '']
    for col, nm in (('adv_bn_r1000', 'Binance 1 s return against us'), ('adv_bn_r250', 'Binance 250 ms return against us')):
        x = Z[col].values.astype(float); pos = x[x > 0]; t1, t2 = np.quantile(pos, [1 / 3, 2 / 3]) if len(pos) else (NAN, NAN)
        bins = (('in our favour (< 0)', x < 0), ('zero', x == 0), (f'against, low (0, {t1:.2f}]', (x > 0) & (x <= t1)),
                (f'against, mid ({t1:.2f}, {t2:.2f}]', (x > t1) & (x <= t2)), (f'against, high (> {t2:.2f})', x > t2), ('missing', ~np.isfinite(x)))
        L.append(f'HOW FAR THE MOVE GOES by the {nm} (share of steps; ticks within 1 s)')
        for lab, m in bins:
            if m.any():
                L.append(f'  {lab:<28} {m.mean():5.1%} of steps: {dist(Z.D1.values[m])}')
        L.append(f'WHERE MOVES OF >= 1 TICK STOP by the {nm}')
        for lab, m in bins:
            if (m & (Z.D1.values >= 1)).any():
                L.append(f'  {lab:<28} {stop(Z.D1.values[m & (Z.D1.values >= 1)])}')
        L.append('')
    qv = np.nanquantile(Z.v.values, [0.2, 0.4, 0.6, 0.8]); bv = np.searchsorted(qv, Z.v.values)
    L.append('WHERE MOVES OF >= 1 TICK STOP by guard-value quintile (Q1 = most negative, the strongest signal)')
    for i in range(5):
        L.append(f'  Q{i + 1} {stop(Z.D1.values[(bv == i) & (Z.D1.values >= 1)])}')
    for o, nm in ((1, 'onset steps'), (0, 'later steps')):
        L.append(f'  {nm:<11} {stop(Z.D1.values[(Z.onset.values == o) & (Z.D1.values >= 1)])}')
    txt = '\n'.join(L) + '\n'
    open(f'{OUTD}/stop_depth_posthoc.txt', 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'posthoc':
        posthoc(); sys.exit(0)
    if cmd == 'selftest':
        all_selftests()
    elif cmd == 'rows':
        rows_day(sys.argv[2])
    elif cmd == 'fit':
        fit_day(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit('usage: pf_stop_depth.py selftest | rows <day> | fit <day> | report')
