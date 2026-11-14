"""ORDER-HARM v2 (2026-11-14, session ce68ebe0). Supersedes pf_order_harm.py (v1, 67a6e6fb / f8f1db6b), which was stopped before any
result after the USER's review (06:1xZ): "Confirmed look-ahead in queue features. Changing a book update at 300 ms changes the features used for
a decision at 250 ms ... Overlapping labels reuse the same fill ... Economic evaluation must count each physical fill once ... Keep and pull
branches have different ending conditions ... Both branches need identical terminal rules and sufficient future data, or an explicit censored
status. ... The immediate priority is repairing target construction". v1's partial rows (two days) were deleted unread.

REPAIRS against v1:
 (1) Causal order-queue inputs. v1 fed the simulator's queue (a0 raised to the next frame up to 100 ms after activation, then capped by frames
     selected by their stamps) into the model. v2 builds the order's queue inputs only from frames KNOWN by the decision time (pf_kp_s4's
     known time kf): a0_k = the size shown at its price in the frame known at its activation (no raise), ahead_k = a0_k capped by the smallest
     size shown in frames known after the activation up to t, shown_k = the size in the frame known at t, qfrac_k = ahead_k / shown_k (an
     unknown size at activation - price beyond the 5 shown levels - is NaN). The simulator still uses its own queue for fills (that is the fill
     model, not an input).
 (2) No summed economics on overlapping rows. v1's R622 summed DELTA over decision rows every 250 ms whose windows share fills; dropped. Offline
     only ranking diagnostics per day; the economic test is the joint replay, where each physical fill counts once.
 (3) Identical terminal rules. Both branches are cut at the same time T = t + H; a row whose horizon passes the lane's end (t + H > 60 s, where
     the reference lane stops and trades are not loaded) is CENSORED (a status, out of training and evaluation); the 5 s labels likewise
     (NaN, counted). The pull branch is simulated to T, never past the lane's end.

ROWS (decision states): every resting order of the guard ladder (rungs d = 0..5, 30 sh, both sides; the reference path = each rung's S4 rest lane
never pulled, QUEUE-PLACE's sim4 with no trigger) at every 250 ms step (k % 5 == 0) while its side's guard is on (v <= -0.10, not masked) and
STOP-DEPTH has the step's inputs; 9 consumed replay days, first 60 s, fill clock M 50. Rung 0 included.
TWO BRANCHES from the decision at t: KEEP = the reference path; PULL = cancel at t (lands t + 22 ms), nothing placed while the guard stays on,
re-entry at the first step after it has been clear 250 ms at the then-known best - d behind the shown size, never pulled after.
LABEL over (t + 22 ms, t + H], H = 2 s (5 s printed); fills inside the cancel latency on neither branch; values sg (mid(anchor + h) - price) x
shares at h = 1 s / 5 s, per share of the 30-sh order: HARM (negative keep fills; severity), GAIN, PULLV, DELTA = HARM + GAIN - PULLV.
INPUTS at t: STOP-DEPTH's NEW inputs (82), INFLUENCE's fair-price inputs (6), and the order's state: rung, ticks behind the known best, age,
a0_k, ahead_k, shown_k, the size at better prices in the known frame, qfrac_k, the known best's move against us since the guard came on.
MODELS (LightGBM, PULL-DEPTH's pinned parameters and rounds, leave one day out, 37,500 training rows per training day): DELTA at 1 s
(regression, clipped +-10 c/sh), P(harm) at 1 s (binary), severity (regression on harmed rows); saved per held-out day.
Readings DECLARED BEFORE THE FIRST RUN (out of day; day rule >= 7 of 9; rows not censored):
 R620 DELTA-SKILL: Spearman IC of predicted DELTA with DELTA (1 s) > 0 on >= 7 days and pooled >= 0.05 -> SKILL; else NO-SKILL.
 R621 HARM-AUC: AUC of P(harm) >= 0.70 pooled and >= 0.65 on >= 7 days -> PREDICTABLE; else WEAK.
 (No economic reading here: the joint replay values the policy.)
Controls (any failure -> no output): (i) NO-LOOKAHEAD falsifier on real data: on every 20th decision row, every frame not yet known at t is
perturbed (sizes x 3 + 11) and the order inputs recomputed must be identical; (ii) synthetic queue cases where a frame stamped after t, and a
frame stamped before t but known after t, must not change the inputs, with planted errors (selection by stamp; the next-frame raise) that must
each fail; (iii) v1's fork cases (sweep avoided, latency fills on neither branch, re-entry waits for the guard) and planted errors; (iv) the
reference path equals pf_kp_s4.sim_arm per rung on every window-side; (v) no row past the lane's end in the labels (asserted); upstream
self-tests; no held-out row in a fit; memory self-guard (2 GiB).
Limits: replay rule M (B1a); the keep branch never pulls later while the policy re-decides every step; overlapping rows (one order's windows share
fills: per-day diagnostics only); rungs independent here; consumed days (C8); every 250 ms only; the step inputs are STOP-DEPTH's (causal by
construction: live-clock rows and known frames).
Usage (run_light, PF_SLOT=claude): pf_order_harm_v2.py selftest | rows <day> | fit <day> | report; driver run_order_harm_v2.sh
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
import pf_qr_queue_race as QR                             # noqa: E402
import pf_queue_place as QP                               # noqa: E402
import pf_pull_depth as PD                                # noqa: E402
import pf_stop_depth as SD                                # noqa: E402
import pf_influence as IN                                 # noqa: E402
import pf_next_price as NP                                # noqa: E402  (ahead_at)
import pf_joint_ladder as JL                              # noqa: E402  (synthetic contexts)

DAYS9 = K.DAYS9
OUTD = f'{K.IO}/order_harm_v2_20261114'
SIZE = 30.0; NR = 6; H = 2000.0; H5 = 5000.0; EVERY = 5; QCFG = ('M', 50.0); MASK_V = -1.0; SUB_N = 300000; CLIP = 10.0
ORD_FEAT = ['rung', 'depth_now', 'age', 'ahead_k', 'a0_k', 'shown_k', 'size_better', 'qfrac_k', 'mv_since']
CHECK_EVERY = 20
FEAT = SD.NEW_FEAT + IN.INF_FEAT + ORD_FEAT
LAB = ['harm1', 'gain1', 'pull1', 'delta1', 'harm5', 'gain5', 'pull5', 'delta5', 'harm1_h5', 'delta1_h5']
INF = float('inf'); NAN = float('nan')
MUT_NOWAIT = MUT_LAT_KEEP = MUT_HARM_SIGN = MUT_STAMP = MUT_RAISE = 0   # planted errors (self-test only)
MEM_CAP_MIB = 2048.0


# ---------------------------------------------------------------- the two branches
def run_fork(w, k, d, vg, q, shf, t_end):
    """The pull branch from step k: rung d's lane restarted at step k with no order, the guard as its entry line (S4's rest lane, idle-only
    trigger), never pulled after re-entry, to t_end. Returns pf_kp_s4.sim_arm's fills and orders."""
    kend = int(np.searchsorted(w['tg'], t_end, side='right'))
    w2 = dict(w)
    for key in ('tg', 'stn'):
        w2[key] = w[key][k:kend]
    for key in ('vkl', 'best', 'bopp'):
        w2[key] = list(w[key][k:kend])
    w2['n'] = kend - k; w2['te'] = t_end
    aux = [True] + [False] * (kend - k - 1) if MUT_NOWAIT else list(vg[k:kend])
    F, O, _ = K.sim_arm(w2, q, shf, 'rest', aux=aux, idle_only=1, offset=d, size=SIZE)
    return F, O


def branch_values(t0, keep, pull, hh):
    """keep / pull: lists of (sim time, value per fill in c x shares at horizon set hh). Returns HARM, GAIN, PULLV, DELTA per share of SIZE over
    (t0 + latency, t0 + hh] for keep and (t0, t0 + hh] for pull."""
    lo = t0 + (0.0 if MUT_LAT_KEEP else K.L_CANCEL); up = t0 + hh
    kv = [v for (t, v) in keep if lo < t <= up]
    harm = sum(v for v in kv if v < 0); gain = sum(v for v in kv if v > 0)
    if MUT_HARM_SIGN:
        harm = -harm
    pv = sum(v for (t, v) in pull if t0 < t <= up)
    return harm / SIZE, gain / SIZE, pv / SIZE, (harm + gain - pv) / SIZE


def queue_known(o, t, lvp, kfl, tfl):
    """Causal queue inputs of order o at decision time t (header repair 1): (a0_k, ahead_k, shown_k) from frames KNOWN by t only."""
    clock = tfl if MUT_STAMP else kfl
    ja = bisect.bisect_right(clock, o['ta']) - 1; jt = bisect.bisect_right(clock, t) - 1
    if ja < 0 or jt < 0:
        return NAN, NAN, NAN
    a0 = float(lvp[ja])
    if MUT_RAISE and ja + 1 < len(lvp) and tfl[ja + 1] - o['ta'] <= 100.0 and np.isfinite(lvp[ja + 1]) and np.isfinite(a0):
        a0 = max(a0, float(lvp[ja + 1]))
    shown = float(lvp[jt]) if np.isfinite(lvp[jt]) else NAN
    if not np.isfinite(a0):
        return NAN, NAN, shown
    ah = a0
    if jt > ja:
        seg = lvp[ja + 1:jt + 1]; seg = seg[np.isfinite(seg)]
        if len(seg):
            ah = min(ah, float(seg.min()))
    return a0, ah, shown


def order_inputs(o, t, d, side, best_now, best_onset, lvp, S5, P5, jk, kfl, tfl):
    """The order's 9 inputs (ORD_FEAT) at decision time t, from known information only."""
    pt = int(o['pt'])
    a0, ah, shown = queue_known(o, t, lvp, kfl, tfl)
    better = float(sum(S5[jk, i] for i in range(5) if P5[jk, i] >= 0 and ((P5[jk, i] > pt) if side == 0 else (P5[jk, i] < pt)))) if jk >= 0 else NAN
    return [float(d), float((best_now - pt) if side == 0 else (pt - best_now)), t - o['ta'], ah, a0, shown, better,
            (ah / shown) if (np.isfinite(ah) and np.isfinite(shown) and shown > 0) else NAN,
            float((best_onset - best_now) if side == 0 else (best_now - best_onset))]


# ---------------------------------------------------------------- self-tests
def selftest():
    bad = []
    fb = [(0.0, [(50, 100), (49, 100)], 52)]
    # (1) a sweep after the decision: the kept order (rung 0 at 50, queue U) is filled through at 400 ms; the pull at 200 ms avoids it
    w = JL._ctx(0, fb, [(400.0, 48, 100.0)], t_end=3000.0, v=[0.5] * 60)
    vg = [False] * 4 + [True] * 12 + [False] * 44             # guard on 200-750 ms
    Fb, Ob = K.sim_arm(w, 'U', 0.0, 'rest', aux=[False] * 60, offset=0, size=SIZE)[:2]
    keep = [(float(w['anc'][f[0]]), -2.0 * f[2]) for f in Fb]
    Ff, Of = run_fork(w, 4, 0, vg, 'U', 0.0, 200.0 + H)
    pull = [(float(w['anc'][f[0]]), 1.0 * f[2]) for f in Ff]
    hv = branch_values(200.0, keep, pull, H)
    reent = min((o[0] for o in Of), default=INF)
    if not (hv[0] == -2.0 and hv[2] == 0.0 and hv[3] == -2.0):
        bad.append(f'sweep after the decision {hv}')
    # (2) the guard staying on delays the re-entry: first placement after 750 ms + 250 ms clear, live 40 ms later
    if reent != 1000.0 + K.L_PLACE:
        bad.append(f're-entry at {reent}')
    # (3) a fill inside the cancel latency (210 ms, decision at 200 ms) counts on neither branch
    hv = branch_values(200.0, [(210.0, -3.0 * 30)], [], H)
    if hv != (0.0, 0.0, 0.0, 0.0):
        bad.append(f'latency fill {hv}')
    # (4) causal queue inputs: frames stamped [0, 100, 200, 240, 300], known 30 ms later; an order live at 140 ms at our price
    tfl = [0.0, 100.0, 200.0, 240.0, 300.0]; kfl = [30.0, 130.0, 230.0, 270.0, 330.0]; o = dict(pt=50, ta=140.0)
    base = np.array([50.0, 40.0, 60.0, 20.0, 10.0])
    q0 = queue_known(o, 250.0, base, kfl, tfl)
    if q0 != (40.0, 40.0, 60.0):
        bad.append(f'queue known at 250 ms {q0}')
    for j, nm in ((3, 'stamped before t but known after t'), (4, 'stamped after t')):
        pert = base.copy(); pert[j] = 1.0
        if queue_known(o, 250.0, pert, kfl, tfl) != q0:
            bad.append(f'queue inputs changed by a frame {nm}')
    q1 = queue_known(o, 180.0, base, kfl, tfl); pert = base.copy(); pert[2] = 99.0     # the next frame after activation is not known at 180 ms
    if queue_known(o, 180.0, pert, kfl, tfl) != q1:
        bad.append('queue inputs changed by the next frame after activation, unknown at the decision')
    return bad


def mutation_check():
    out = {}
    for nm in ('MUT_NOWAIT', 'MUT_LAT_KEEP', 'MUT_HARM_SIGN', 'MUT_STAMP', 'MUT_RAISE'):
        globals()[nm] = 1
        try:
            out[nm] = bool(selftest())
        finally:
            globals()[nm] = 0
    return out


def all_selftests():
    IN.all_selftests(); QP.all_selftests()
    b = selftest(); m = mutation_check()
    if b or not all(m.values()):
        sys.exit(f'ORDER-HARM v2 SELFTEST FAILED -> no output: {b} planted errors caught {m}')
    print(f'order-harm v2 selftests passed; planted errors caught: {m}', flush=True)
    return m


# ---------------------------------------------------------------- rows (one day)
def rows_day(day):
    all_selftests()
    T0 = time.time(); D = K.load_day(day); q, shf = QCFG
    SR = pd.read_parquet(f'{SD.OUTD}/rows_{day}.parquet', columns=['m', 'side', 'k'] + SD.NEW_FEAT); SRI = SR.groupby(['m', 'side']).indices
    SRV = SR[SD.NEW_FEAT].to_numpy(dtype=np.float64); X31 = SD.NEW_FEAT.index('x31')
    P = pd.read_parquet(f'{K.IO}/step5/{day}/rows_rl_d2_pk.parquet', columns=['mid', 'tau', 'p_bn', 'tau_rem', 'sigma'])
    P = P[P.mid.isin(D['W0'])].sort_values(['mid', 'tau'], kind='stable')
    PB = {int(m): (g.tau.values.astype(float).tolist(), g.p_bn.values.astype(float), g.tau_rem.values.astype(float), g.sigma.values.astype(float))
          for m, g in P.groupby('mid', sort=False)}
    del P
    st = dict(window_sides=0, ref_differ=0, steps=0, no_order=0, no_inputs=0, rows=0, censored=0, censored5=0, lookahead_checks=0, lookahead_bad=0)
    out = []
    for w in K.window_sides(D):
        st['window_sides'] += 1; m, side, sg, n = int(w['m']), int(w['side']), w['sg'], w['n']
        if not n or (m, side) not in SRI:
            continue
        tf, mff = w['tf_mff']; tfl = tf.tolist(); P5, S5, opp = w['P5'], w['S5'], w['opp']; kf = np.asarray(w['kf'], float); kfl = kf.tolist()
        vv = np.asarray(w['vkl'], float); vg = (vv <= K.GUARD_PULL).tolist()
        ii = SRI[(m, side)]; kmap = {int(k_): int(j_) for k_, j_ in zip(SR.k.values[ii], ii)}
        steps = [k for k in range(0, n, EVERY) if vg[k] and vv[k] != MASK_V]
        if not steps:
            continue
        rt, pv, trv, sgv = PB.get(m, ([], np.zeros(0), np.zeros(0), np.zeros(0))); adv = IN.adv_of(side)
        tt = w['anc'] - shf; cache = w.setdefault('cache', {})
        val = lambda t, hh, pt: sg * (mff[max(int(np.searchsorted(tf, t + hh, side='right')) - 1, 0)] - pt)
        onset = np.zeros(n, int); k0 = -1
        for k in range(n):
            if vg[k] and (k == 0 or not vg[k - 1]):
                k0 = k
            onset[k] = k0 if vg[k] else -1
        rw = []
        for d in range(NR):
            Fo, Ot, O, fx = QP.sim4(side, tf, opp, P5, S5, cache, w['tg'], w['vkl'], -INF, w['best'], w['bopp'], tt, w['ptl'], w['shl'], q, w['te'],
                                    [False] * n, d, SIZE)
            Fk, Ok, _ = K.sim_arm(w, q, shf, 'rest', aux=[False] * n, offset=d, size=SIZE)
            st['ref_differ'] += int(Fk != Fo or Ok != Ot)
            keep1 = [(float(tt[f[0]]), val(float(w['anc'][f[0]]), 1000.0, int(f[1])) * f[2]) for f in Fo]
            keep5 = [(float(tt[f[0]]), val(float(w['anc'][f[0]]), 5000.0, int(f[1])) * f[2]) for f in Fo]
            for k in steps:
                st['steps'] += 1; t = float(w['tg'][k])
                o = next((o_ for o_ in O if not o_['ref'] and o_['ta'] <= t < min(o_['tr'], o_['tfill'])), None)
                if o is None:
                    st['no_order'] += 1; continue
                if k not in kmap:
                    st['no_inputs'] += 1; continue
                if t + H > w['te']:                       # repair 3: the horizon passes the lane's end -> censored
                    st['censored'] += 1; continue
                c5 = t + H5 > w['te']; st['censored5'] += int(c5)
                tend = (t + H) if c5 else (t + H5)
                assert tend <= w['te'] + 1e-9
                Ff, _ = run_fork(w, k, d, vg, q, shf, tend)
                pull1 = [(float(tt[f[0]]), val(float(w['anc'][f[0]]), 1000.0, int(f[1])) * f[2]) for f in Ff]
                pull5 = [(float(tt[f[0]]), val(float(w['anc'][f[0]]), 5000.0, int(f[1])) * f[2]) for f in Ff]
                b1 = branch_values(t, keep1, pull1, H); b5 = branch_values(t, keep5, pull5, H)
                b15 = (NAN, NAN, NAN, NAN) if c5 else branch_values(t, keep1, pull1, H5)
                jk = int(w['jk'][k]); b = int(w['best'][k]); pt = int(o['pt'])
                lvp = cache.get(pt)
                if lvp is None:
                    lvp = QR.lvarr(P5, S5, pt, side); cache[pt] = lvp
                k0 = onset[k]; b0k = int(w['best'][k0]) if k0 >= 0 and w['best'][k0] >= 0 else b
                ordf = order_inputs(o, t, d, side, b, b0k, lvp, S5, P5, jk, kfl, tfl)
                if st['rows'] % CHECK_EVERY == 0:          # control (i): every frame not yet known at t perturbed -> identical inputs
                    S5p = S5.copy(); fut = kf > t; S5p[fut] = S5[fut] * 3.0 + 11.0
                    ordp = order_inputs(o, t, d, side, b, b0k, QR.lvarr(P5, S5p, pt, side), S5p, P5, jk, kfl, tfl)
                    st['lookahead_checks'] += 1
                    st['lookahead_bad'] += int(not all((x == y) or (x != x and y != y) for x, y in zip(ordf, ordp)))
                pk = IN.pbn_at(rt, pv, t)
                if np.isfinite(pk):
                    jj = bisect.bisect_right(rt, t) - 1
                    inf = [adv * (pk - IN.pbn_at(rt, pv, t - 250.0)), adv * (pk - IN.pbn_at(rt, pv, t - 1000.0)), adv * float(SRV[kmap[k], X31]),
                           IN.sens_of(pk, float(sgv[jj]), float(trv[jj])), abs(pk - 50.0), float(trv[jj])]
                else:
                    inf = [NAN] * len(IN.INF_FEAT)
                rw.append([int(day), m, side, k, d, t - w['ws']] + list(SRV[kmap[k]]) + inf + ordf +
                           [b1[0], b1[1], b1[2], b1[3], b5[0], b5[1], b5[2], b5[3], b15[0], b15[3]])
                st['rows'] += 1
        if rw:
            out.append(np.asarray(rw, dtype=np.float64))
        w.pop('cache', None)
        if st['window_sides'] % 80 == 0:
            r2 = K.rss_mib(); print(f'{day}: {st["window_sides"]} window-sides, {st["rows"]} rows, {time.time() - T0:.0f} s, rss {r2:.0f} MiB', flush=True)
            if r2 > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r2:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    if st['ref_differ'] or st['lookahead_bad'] or not st['lookahead_checks']:
        sys.exit(f'ORDER-HARM v2 CONTROL FAILED on {day} (reference path / no-look-ahead falsifier) -> no output: {st}')
    os.makedirs(OUTD, exist_ok=True)
    cols = ['day', 'm', 'side', 'k', 'rung_i', 't_rel'] + FEAT + LAB
    R = pd.DataFrame(np.vstack(out) if out else np.zeros((0, len(cols))), columns=cols).astype({'day': 'int64', 'm': 'int64', 'side': 'int64', 'k': 'int64', 'rung_i': 'int64'})
    R.to_parquet(f'{OUTD}/rows_{day}.parquet', index=False); json.dump(st, open(f'{OUTD}/rows_{day}.json', 'w'))
    print(f'{day}: {st} in {time.time() - T0:.0f} s', flush=True)


# ---------------------------------------------------------------- fit (one held-out day)
def fit_day(d):
    import lightgbm as lgb
    all_selftests()
    T0 = time.time()
    cols = ['day', 'm', 'side', 'k', 'rung_i', 'delta1', 'harm1'] + FEAT; parts = []; per_day = SUB_N // (len(DAYS9) - 1)
    for x in DAYS9:                                       # memory: each training day subsampled as it is read (float32)
        if x == d:
            continue
        R_ = pd.read_parquet(f'{OUTD}/rows_{x}.parquet', columns=cols)
        R_ = R_.iloc[np.sort(np.random.default_rng(int(d) * 13 + int(x)).choice(len(R_), min(len(R_), per_day), replace=False))]
        parts.append(R_.astype({c: 'float32' for c in FEAT})); del R_
    tr = pd.concat(parts, ignore_index=True); del parts
    assert not (tr.day == int(d)).any(), 'held-out day in the training rows'
    te = pd.read_parquet(f'{OUTD}/rows_{d}.parquet', columns=cols).astype({c: 'float32' for c in FEAT})
    Xtr = tr[FEAT].values.astype(np.float32); Xte = te[FEAT].values.astype(np.float32)
    preg = dict(PD.LGB_PARAMS, objective='regression'); out = dict(m=te.m.values, side=te.side.values, k=te.k.values, rung_i=te.rung_i.values); info = {}
    b = lgb.train(preg, lgb.Dataset(Xtr, np.clip(tr.delta1.values, -CLIP, CLIP), feature_name=FEAT), PD.N_ROUNDS)
    out['p_delta'] = b.predict(Xte); os.makedirs(f'{OUTD}/models', exist_ok=True); b.save_model(f'{OUTD}/models/delta_{d}.txt')
    imp = b.feature_importance('gain'); info['delta_top'] = [FEAT[i] for i in np.argsort(-imp)[:15]]
    b = lgb.train(PD.LGB_PARAMS, lgb.Dataset(Xtr, (tr.harm1.values < 0).astype(int), feature_name=FEAT), PD.N_ROUNDS)
    out['p_harm'] = b.predict(Xte); b.save_model(f'{OUTD}/models/harm_{d}.txt')
    hr = tr.harm1.values < 0
    b = lgb.train(preg, lgb.Dataset(Xtr[hr], np.clip(-tr.harm1.values[hr], 0, CLIP), feature_name=FEAT), PD.N_ROUNDS)
    out['p_sev'] = b.predict(Xte); b.save_model(f'{OUTD}/models/sev_{d}.txt')
    pd.DataFrame(out).to_parquet(f'{OUTD}/pred_{d}.parquet', index=False)
    json.dump(dict(info=info, train_rows=len(tr), harm_rows=int(hr.sum())), open(f'{OUTD}/fit_{d}.json', 'w'), indent=1)
    print(f'fit {d}: train {len(tr)} in {time.time() - T0:.0f} s', flush=True)


# ---------------------------------------------------------------- report
def report():
    from scipy.stats import spearmanr
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    rs = {d: json.load(open(f'{OUTD}/rows_{d}.json')) for d in DAYS9}
    A = pd.concat([pd.read_parquet(f'{OUTD}/rows_{d}.parquet', columns=['day', 'm', 'side', 'k', 'rung_i', 'gage'] + LAB).merge(
        pd.read_parquet(f'{OUTD}/pred_{d}.parquet'), on=['m', 'side', 'k', 'rung_i'], how='left', validate='one_to_one') for d in DAYS9], ignore_index=True)
    R = {}
    L = [f'ORDER-HARM v2, 9 consumed replay days, first 60 s, guard ladder decision states every 250 ms while the guard is on, M 50; as of {asof}; '
         f'controls passed (no-look-ahead falsifier: {sum(r["lookahead_checks"] for r in rs.values())} rows with every unknown frame perturbed, 0 '
         f'differ); rows {len(A)}; censored (horizon past the lane\'s end) {sum(r["censored"] for r in rs.values())}, 5 s horizon censored '
         f'{sum(r["censored5"] for r in rs.values())}; decision steps without a resting order {sum(r["no_order"] for r in rs.values())}, without '
         f'inputs {sum(r["no_inputs"] for r in rs.values())}; overlapping rows: ranking diagnostics only, no summed value', '']
    L.append('LABEL by rung (c per share of the 30-sh order, 1 s values, H 2 s): rows | P(harm) | severity when harmed | gain | pull value | DELTA (keep - pull) | DELTA 5 s values')
    for dg in range(NR):
        x = A[A.rung_i == dg]; hm = x.harm1 < 0
        L.append(f'  rung {dg}: {len(x):>7} | {hm.mean():5.1%} | {(-x.harm1[hm]).mean():6.2f} | {x.gain1.mean():+6.3f} | {x.pull1.mean():+6.3f} |'
                 f' {x.delta1.mean():+6.3f} | {x.delta5.mean():+6.3f}')
    x = A; hm = x.harm1 < 0
    L.append(f'  all   : {len(x):>7} | {hm.mean():5.1%} | {(-x.harm1[hm]).mean():6.2f} | {x.gain1.mean():+6.3f} | {x.pull1.mean():+6.3f} |'
             f' {x.delta1.mean():+6.3f} | {x.delta5.mean():+6.3f}')
    for lo, hi, nm in ((1, 2, 'first step'), (2, 6, '2-5 steps'), (6, 21, '6-20 steps'), (21, 10 ** 6, '> 20 steps')):
        x = A[(A.gage >= lo) & (A.gage < hi)]
        if len(x):
            L.append(f'  steps since the guard came on {nm:<11}: {len(x):>7} rows, P(harm) {float((x.harm1 < 0).mean()):5.1%}, DELTA {x.delta1.mean():+6.3f}')
    L.append('')
    ic = {}; up = 0
    for d in DAYS9:
        x = A[A.day == int(d)]; ic[d] = float(spearmanr(x.p_delta.values, x.delta1.values)[0]); up += int(ic[d] > 0)
    icp = float(spearmanr(A.p_delta.values, A.delta1.values)[0])
    R['R620'] = dict(pooled=icp, per_day=ic, days=up, label='SKILL' if (up >= 7 and icp >= 0.05) else 'NO-SKILL')
    L.append(f'R620 DELTA-SKILL: Spearman IC pooled {icp:+.3f}; per day ' + ', '.join(f'{d[4:]} {ic[d]:+.3f}' for d in DAYS9) + f' -> {R["R620"]["label"]}')
    y = (A.harm1.values < 0); au = PD.auc(y, A.p_harm.values); aud = {d: PD.auc(y[A.day.values == int(d)], A.p_harm.values[A.day.values == int(d)]) for d in DAYS9}
    R['R621'] = dict(pooled=au, per_day=aud, label='PREDICTABLE' if (au >= 0.70 and sum(aud[d] >= 0.65 for d in DAYS9) >= 7) else 'WEAK')
    L.append(f'R621 HARM-AUC: pooled {au:.3f}; per day ' + ', '.join(f'{d[4:]} {aud[d]:.3f}' for d in DAYS9) + f' -> {R["R621"]["label"]}')
    hm = A.harm1 < 0
    L.append(f'  severity model on harmed rows: Spearman {float(spearmanr(A.p_sev[hm], -A.harm1[hm])[0]):+.3f}')
    L.append('')
    L.append('CALIBRATION (deciles of the prediction: mean predicted / realized)')
    for col, ycol, nm in (('p_delta', 'delta1', 'DELTA'), ('p_harm', None, 'P(harm)')):
        p = A[col].values; yv = A[ycol].values if ycol else (A.harm1.values < 0).astype(float)
        qd = np.quantile(p, np.linspace(0.1, 0.9, 9)); bb = np.searchsorted(qd, p)
        L.append(f'  {nm}: ' + ' '.join(f'{p[bb == i].mean():+.2f}/{yv[bb == i].mean():+.2f}' for i in range(10) if (bb == i).any()))
    tops = {}
    for d in DAYS9:
        for f in json.load(open(f'{OUTD}/fit_{d}.json'))['info']['delta_top'][:10]:
            tops[f] = tops.get(f, 0) + 1
    L.append('DELTA model top inputs (days in the top 10): ' + ', '.join(f'{f} {c}' for f, c in sorted(tops.items(), key=lambda x: -x[1])[:15]))
    L.append('')
    L.append('READINGS: ' + json.dumps({k_: v_['label'] for k_, v_ in R.items()}))
    txt = '\n'.join(L) + '\n'
    open(f'{OUTD}/order_harm_v2.txt', 'w').write(txt); json.dump(dict(asof=asof, readings=R), open(f'{OUTD}/order_harm_v2.json', 'w'), indent=1, default=float)
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
        sys.exit('usage: pf_order_harm_v2.py selftest | rows <day> | fit <day> | report')
