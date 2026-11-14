"""ORDER-HARM (2026-11-14, session ce68ebe0; USER: "the next target should be the probability and severity of harm to each order after
cancellation could take effect, with queue value included in the keep/pull decision. A single expected move distance is insufficient because it
hides continuation risk", after naming three gaps: orders at the best blanket-pulled on the guard; a model learned at signal onset applied
throughout the signal (it must predict the remaining harm from the current decision state); rungs evaluated independently).

ROWS (decision states): every resting order of the guard ladder (rungs d = 0..5, 30 sh, both sides; the reference path = each rung's S4 rest lane
never pulled, QUEUE-PLACE's sim4 with no trigger) at every 250 ms step (k % 5 == 0) while its side's guard is on (v <= -0.10, not masked) and
STOP-DEPTH has the step's inputs; 9 consumed replay days, first 60 s, fill clock M 50. Rung 0, the order at the best, is included.
TWO BRANCHES from the decision at t (S1's construction with the pull branch made policy-consistent - S4's label lacked the stay-out):
  KEEP  the reference path: the order stays, and the lane goes on never pulled
  PULL  the order is cancelled at t (lands t + 22 ms); the lane places nothing while the guard stays on, re-enters at the first step after the
        guard has been clear for 250 ms at the then-known best - d behind the shown size (S4's rest lane with the guard as the entry line),
        and is never pulled after
LABEL over (t + 22 ms, t + H], H = 2 s (5 s printed); fills inside the cancel latency happen on both branches and are excluded; each fill valued
sg (mid(anchor + h) - price) x shares at h = 1 s (primary) and 5 s (FUTURE-BEST's values), per share of the 30-sh order:
  HARM   the keep branch's fills with negative value (summed; severity = its size); P(harm) = [HARM < 0]
  GAIN   the keep branch's fills with positive value
  PULLV  the pull branch's fills (the re-entered order)
  DELTA  = HARM + GAIN - PULLV (keep minus pull: the queue kept and the cost of re-entering behind are inside it)
INPUTS at t (no fitted model's output): STOP-DEPTH's NEW inputs (82, incl. steps since the guard came on), INFLUENCE's fair-price inputs (6),
and the order's own state: rung, ticks behind the known best now, age, rule M's queue ahead now (shown size at activation capped by the
smallest shown since), the shown size at its price, the size at better prices on our side, queue ahead / shown, the known best's move against
us since the guard came on.
MODELS (LightGBM, PULL-DEPTH's pinned parameters and rounds, leave one day out, 300,000 training rows = 37,500 per training day): DELTA at 1 s (regression, clipped
+-10 c/sh), P(harm) at 1 s (binary), severity (regression on rows with harm). Saved per held-out day for the replay.
Readings DECLARED BEFORE THE FIRST RUN (out of day; day rule >= 7 of 9):
 R620 DELTA-SKILL: Spearman IC of predicted DELTA with DELTA (1 s) > 0 on >= 7 days and pooled >= 0.05 -> SKILL; else NO-SKILL.
 R621 HARM-AUC: AUC of P(harm) >= 0.70 pooled and >= 0.65 on >= 7 days -> PREDICTABLE; else WEAK.
 R622 OFFLINE-KEEP (one decision, on the label rows): keep where predicted DELTA >= 0, pull elsewhere. Summed DELTA of the kept rows > 0 on
      >= 7 days (better than pulling every row) AND summed DELTA of the pulled rows < 0 on >= 7 days (better than keeping every row) ->
      BEATS-BOTH; else 'k of 2'.
Printed: label statistics by rung and by steps since the guard came on (P(harm), severity, gain, pull value, DELTA); calibration; top inputs;
statuses (decision steps without a resting order, without inputs, fork windows truncated at the lane's end).
Controls (any failure -> no output): synthetic fork cases (a sweep after the decision: the kept order is harmed and the pull avoids it; a fill
inside the cancel latency counts on neither branch; the guard staying on delays the re-entry) with planted errors (the pull branch re-enters
after 250 ms without waiting for the guard; latency fills counted for keep; the harm sign flipped) that must each fail a case; the reference
path equals pf_kp_s4.sim_arm per rung (fill for fill, order for order) on every window-side; STOP-DEPTH's / INFLUENCE's / QUEUE-PLACE's
self-tests; no held-out row in a fit; memory self-guard (2 GiB).
Limits: replay rule M (B1a); the keep branch never pulls later while the policy re-decides every step (the label prices keeping through H
against leaving now); rungs independent here (the policy is evaluated in the joint replay); consumed days (C8); every 250 ms only.
Usage (run_light, PF_SLOT=claude): pf_order_harm.py selftest | rows <day> | fit <day> | report; driver run_order_harm.sh
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
OUTD = f'{K.IO}/order_harm_20261114'
SIZE = 30.0; NR = 6; H = 2000.0; H5 = 5000.0; EVERY = 5; QCFG = ('M', 50.0); MASK_V = -1.0; SUB_N = 300000; CLIP = 10.0
ORD_FEAT = ['rung', 'depth_now', 'age', 'ahead', 'a0', 'shown_at', 'size_better', 'qfrac', 'mv_since']
FEAT = SD.NEW_FEAT + IN.INF_FEAT + ORD_FEAT
LAB = ['harm1', 'gain1', 'pull1', 'delta1', 'harm5', 'gain5', 'pull5', 'delta5', 'harm1_h5', 'delta1_h5']
INF = float('inf'); NAN = float('nan')
MUT_NOWAIT = MUT_LAT_KEEP = MUT_HARM_SIGN = 0             # planted errors (self-test only)
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
    return bad


def mutation_check():
    out = {}
    for nm in ('MUT_NOWAIT', 'MUT_LAT_KEEP', 'MUT_HARM_SIGN'):
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
        sys.exit(f'ORDER-HARM SELFTEST FAILED -> no output: {b} planted errors caught {m}')
    print(f'order-harm selftests passed; planted errors caught: {m}', flush=True)
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
    st = dict(window_sides=0, ref_differ=0, steps=0, no_order=0, no_inputs=0, rows=0, truncated5=0)
    out = []
    for w in K.window_sides(D):
        st['window_sides'] += 1; m, side, sg, n = int(w['m']), int(w['side']), w['sg'], w['n']
        if not n or (m, side) not in SRI:
            continue
        tf, mff = w['tf_mff']; tfl = tf.tolist(); P5, S5, opp = w['P5'], w['S5'], w['opp']
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
                if t + H5 > w['te']:
                    st['truncated5'] += 1
                Ff, _ = run_fork(w, k, d, vg, q, shf, t + H5)
                pull1 = [(float(tt[f[0]]), val(float(w['anc'][f[0]]), 1000.0, int(f[1])) * f[2]) for f in Ff]
                pull5 = [(float(tt[f[0]]), val(float(w['anc'][f[0]]), 5000.0, int(f[1])) * f[2]) for f in Ff]
                b1 = branch_values(t, keep1, pull1, H); b5 = branch_values(t, keep5, pull5, H); b15 = branch_values(t, keep1, pull1, H5)
                jk = int(w['jk'][k]); b = int(w['best'][k]); pt = int(o['pt'])
                lvp = cache.get(pt)
                if lvp is None:
                    lvp = QR.lvarr(P5, S5, pt, side); cache[pt] = lvp
                shown = float(lvp[jk]) if jk >= 0 and np.isfinite(lvp[jk]) else NAN
                better = float(sum(S5[jk, i] for i in range(5) if P5[jk, i] >= 0 and ((P5[jk, i] > pt) if side == 0 else (P5[jk, i] < pt)))) if jk >= 0 else NAN
                ah = NP.ahead_at(o, t, lvp, tfl)
                k0 = onset[k]; b0k = int(w['best'][k0]) if k0 >= 0 and w['best'][k0] >= 0 else b
                ordf = [float(d), float((b - pt) if side == 0 else (pt - b)), t - o['ta'], ah, float(o['a0']), shown, better,
                        (ah / shown) if shown and shown > 0 else NAN, float((b0k - b) if side == 0 else (b - b0k))]
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
    if st['ref_differ']:
        sys.exit(f'ORDER-HARM CONTROL FAILED on {day}: the reference path differs from sim_arm -> no output: {st}')
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
    L = [f'ORDER-HARM, 9 consumed replay days, first 60 s, guard ladder decision states every 250 ms while the guard is on, M 50; as of {asof}; '
         f'controls passed; rows {len(A)}; decision steps without a resting order {sum(r["no_order"] for r in rs.values())}, without inputs '
         f'{sum(r["no_inputs"] for r in rs.values())}; 5 s windows cut by the lane\'s end {sum(r["truncated5"] for r in rs.values())}', '']
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
    kp = A.p_delta.values >= 0; a1 = a2 = 0; per = []
    for d in DAYS9:
        ix = A.day.values == int(d); sk = float(A.delta1.values[ix & kp].sum()); sp = float(A.delta1.values[ix & ~kp].sum())
        a1 += int(sk > 0); a2 += int(sp < 0); per.append((d[4:], round(sk / max((ix & kp).sum(), 1), 3), round(sp / max((ix & ~kp).sum(), 1), 3)))
    lab = 'BEATS-BOTH' if (a1 >= 7 and a2 >= 7) else f'{int(a1 >= 7) + int(a2 >= 7)} of 2'
    R['R622'] = dict(kept_pos_days=a1, pulled_neg_days=a2, label=lab)
    L.append(f'R622 OFFLINE-KEEP: keeps {kp.mean():.1%} of rows; kept rows DELTA {A.delta1[kp].mean():+.3f} (> 0 summed on {a1}/9 days); pulled rows DELTA '
             f'{A.delta1[~kp].mean():+.3f} (< 0 on {a2}/9) -> {lab}; per day (mean kept / mean pulled) {per}')
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
    open(f'{OUTD}/order_harm.txt', 'w').write(txt); json.dump(dict(asof=asof, readings=R), open(f'{OUTD}/order_harm.json', 'w'), indent=1, default=float)
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
        sys.exit('usage: pf_order_harm.py selftest | rows <day> | fit <day> | report')
