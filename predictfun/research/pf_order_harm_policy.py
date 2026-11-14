"""ORDER-HARM POLICY (2026-11-14, session ce68ebe0): ORDER-HARM v2's per-order models used as a keep / pull policy in JOINT-LADDER v3's lane
(both sides, shared position, decisions on client-known state, hard pre-trade caps, conservation checks, each fill counted once).

USER 2026-11-14: "Orders originally posted at the best still get pulled whenever the guard fires, regardless of the new forecast" / "It needs to
predict the remaining move from the current decision state" / "Preserving several orders needs evaluation under shared inventory limits,
including adverse fills and cancellation/re-entry costs" / "the next target should be the probability and severity of harm to each order after
cancellation could take effect, with queue value included in the keep/pull decision" / "We should have larger cap when we increase the size".

LANE: pf_joint_ladder_v3.sim_joint unchanged (state 'client', rungs 0-5 x 30 sh, both sides); caps hard K = 2 / 5 / 10 x size; fill clocks
M 0 / 50 / 250; values as v3 (WS / E2 / all trades at 1 s without fills on orders < 100 ms old, 5 s, 60 s marks, settlement, order-age split).
ARMS:
  LDG   pull every resting order while the guard is on (reference);
  FB25  FUTURE-BEST's rule: keep rung d >= 1 when P(move >= d + 1) < 0.25; rung 0 at the best always pulled;
  OHH   keep a resting order while the guard is on iff its predicted P(harm within 2 s) < tau_H;            } rung 0 included;
  OHE   keep iff its predicted expected harm P(harm) x severity < eps_E (c/sh);                              } ORDER-HARM v2's models
  OHD   keep iff its predicted DELTA = keep - pull-and-stay-out (c/sh, harm + gain - re-entry value) > theta_D; }
  OHD0  keep iff its predicted DELTA > 0: the model's own economic rule (not matched; its keep share printed);  }
  OFB2  FB25's rule with the realized 2 s move (perfect foresight; a ceiling only).
  A model arm decides when an order first rests with the guard on (any 50 ms step) and then on the label's 250 ms grid (k % 5 == 0); between
  decisions the last decision holds; a new guard episode starts fresh. No inputs at a decision (masked start, no STOP-DEPTH row) -> pull (fail
  closed, counted). Once pulled an order stays out while the guard is on and re-enters at the back 250 ms after it clears (the lane's rule = the
  label's pull branch).
MODELS AND INPUTS: ORDER-HARM v2's boosters of the replayed day (models/{harm,sev,delta}_<day>.txt, fit WITHOUT that day); inputs exactly
  ORDER-HARM v2's FEAT: STOP-DEPTH's step inputs at (window, side, step), INFLUENCE's six fair-price inputs at t (as v2's rows_day), and the
  lane's own order's nine inputs from pf_order_harm_v2.order_inputs (frames known by t only); cast to float32 as in the fit.
THRESHOLDS, MATCHED ON THE DECISION VARIABLE (CLAUDE.md rule 7), fixed before any replay from PREDICTIONS ONLY (no outcome read): on ORDER-HARM
  v2's decision rows (never-pull reference path, 9 days, the saved out-of-day predictions) FB25's rule keeps a share s_FB of rows (PULL-DEPTH's P
  at each row, as fb_dec); tau_H and eps_E = the s_FB quantiles of P(harm) and P(harm) x severity, theta_D = the (1 - s_FB) quantile of DELTA,
  so OHH / OHE / OHD keep as many orders as FB25 on that population (OHD0 is not matched). The lane's realized keep shares (decisions on the
  250 ms grid) are printed.
READINGS DECLARED BEFORE THE FIRST RUN (WS-timed $/day w/o young at 1 s; day rule >= 7 of 9; per model arm and per hard cap):
  R630 OH-VS-FB25: higher than FB25 at M 0 AND M 50 on >= 7 days -> BETTER; lower at both on >= 7 -> WORSE; else NOT-SHOWN.
  R631 OH-VS-LDG:  higher than LDG at M 0 AND M 50 on >= 7 days -> PAYS; lower at both on >= 7 -> HURTS; else NOT-SHOWN.
  R632 RUNG-0 (printed): $/day on rung 0 vs rungs 1-5 per arm (the order at the best, which FB25 always pulls).
  Always printed: M 250, E2, all trades, 60 s marks, settlement, age split (C1 / C12), keep shares by rung, cap refusals, positions.
CONTROLS (any failure -> no output):
  (i) selftests: v3's (with its planted errors), ORDER-HARM v2's (no-look-ahead queue inputs), FUTURE-BEST's, and this policy's decision logic on
      synthetic steps (first-rest decision off the grid, hold between decisions, fresh decision per guard episode, guard off -> keep, no inputs
      -> pull, threshold directions) with planted errors (reversed threshold; a decision held across episodes; no inputs -> keep);
  (ii) INPUT PARITY on 3 days (10-23, 10-31, 11-03): the never-pull joint lane (no cap, M 50, ORDER-HARM's clock) records the model inputs at
      every 250 ms guard-on step; wherever its order is ORDER-HARM v2's row order (same window, side, step, rung, ticks behind the best and age)
      all 97 inputs must be identical and the three predictions equal to the saved out-of-day predictions (|diff| <= 1e-9);
  (iii) NO-LOOK-AHEAD falsifier on every 20th new decision input in the replay: every frame not yet known at t perturbed, the order's inputs
      recomputed must be identical;
  (iv) REGRESSION: LDG / FB25 / OFB2 fills equal v3's saved fills row for row on every day, cap and clock (same lane, harness unchanged);
  (v) v3's conservation / accounting checks on every run, and the true exposure (pf_joint_ladder_v3_exposure.max_exposure) <= K x size on every
      run; memory self-guard (2 GiB).
Limits: replay rule M (B1a); the label's 2 s horizon and blind keep; ORDER-HARM's models were fit on the S4 reference path's order population;
  thresholds matched on that population, not in the lane; the guard ladder, not the frozen v3 (C11); consumed days (C8).
Usage (run_light, PF_SLOT=claude): pf_order_harm_policy.py selftest | thresholds | parity <day> | eval <day> | report;
driver run_order_harm_policy.sh
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
import pf_timing_split as TS                              # noqa: E402
import pf_pull_depth as PD                                # noqa: E402
import pf_future_best as FB                               # noqa: E402
import pf_stop_depth as SD                                # noqa: E402
import pf_influence as IN                                 # noqa: E402
import pf_order_harm_v2 as OH                             # noqa: E402
import pf_joint_ladder_v3 as J3                           # noqa: E402
import pf_joint_ladder_v3_exposure as JX                  # noqa: E402

DAYS9 = K.DAYS9
OUTD = f'{K.IO}/order_harm_policy_20261114'
SIZE = J3.SIZE; NR = J3.NR
CAPS = (('hard', 2), ('hard', 5), ('hard', 10))
CFGS = J3.CFGS
BASE_ARMS = ('LDG', 'FB25', 'OFB2'); OH_ARMS = ('OHH', 'OHE', 'OHD', 'OHD0'); ARMS = ('LDG', 'FB25', 'OHH', 'OHE', 'OHD', 'OHD0', 'OFB2')
EVERY = OH.EVERY                                          # 5 steps = 250 ms, the label's grid
PAR_DAYS = ('20261023', '20261031', '20261103')
CHECK_EVERY = 20
MEM_CAP_MIB = 2048.0
BOOSTERS = ('harm', 'sev', 'delta')
INF = float('inf'); NAN = float('nan')
MUT_SIGN = MUT_HOLD = MUT_NOINPUT_KEEP = 0                # planted errors (selftest only)


# ---------------------------------------------------------------- decision logic
def keep_rule(arm, ph, ps, pdl, thr):
    if arm == 'OHH':
        k_ = ph < thr['tau_H']
    elif arm == 'OHE':
        k_ = ph * ps < thr['eps_E']
    elif arm == 'OHD':
        k_ = pdl > thr['theta_D']
    elif arm == 'OHD0':
        k_ = pdl > 0.0
    else:
        raise ValueError(arm)
    return (not k_) if MUT_SIGN else bool(k_)


def oh_dec(arm, s, d, wi, pred, thr, cnt):
    """Keep / pull callback for rung d of side s (header). Returns True = pull. cnt: decision counters (250 ms grid) and fail-closed count."""
    vg = wi['vg'][s]; onset = wi['onset'][s]; MK = wi['MK'][s]; last = {}

    def dec(k, t, cur):
        if not vg[k]:
            return False
        k0 = int(onset[k]); key = id(cur); held = last.get(key)
        if held is not None and (held[0] == k0 or MUT_HOLD) and k % EVERY != 0:
            return not held[1]
        if k not in MK:
            keep = bool(MUT_NOINPUT_KEEP); cnt['noinput'] += 1
        else:
            ph, ps, pdl = pred(s, d, cur, k, t)
            keep = keep_rule(arm, ph, ps, pdl, thr)
        last[key] = (k0, keep)
        if k % EVERY == 0:
            cnt['keep' if keep else 'pull'][d] += 1
        return not keep
    return dec


def counted(f, vg, d, cnt):
    """Wrap a base arm's callback to count its keep / pull decisions on the 250 ms grid (decisions unchanged; the regression control checks)."""
    def dec(k, t, cur):
        r = f(k, t, cur)
        if vg[k] and k % EVERY == 0:
            cnt['pull' if r else 'keep'][d] += 1
        return r
    return dec


# ---------------------------------------------------------------- self-tests
def selftest():
    bad = []
    vg = [False, True, True, True, True, True, True, False, True, True, True, True]
    onset = []; k0 = -1
    for k, g in enumerate(vg):
        if g and (k == 0 or not vg[k - 1]):
            k0 = k
        onset.append(k0 if g else -1)
    MK = {k: np.zeros(3) for k in range(12) if vg[k] and k != 9}
    wi = dict(vg={0: vg}, onset={0: onset}, MK={0: MK})
    PH = {1: 0.05, 5: 0.5, 8: 0.5, 10: 0.5, 11: 0.05}
    calls = []

    def pred(s, d, o, k, t):
        calls.append(k); return PH.get(k, 0.5), 2.0, (0.1 if PH.get(k, 0.5) < 0.1 else -0.3)
    thr = dict(tau_H=0.10, eps_E=0.2 * 2.0, theta_D=0.0)

    def run(arm, ks, o):
        cnt = dict(keep=[0] * NR, pull=[0] * NR, noinput=0); f = oh_dec(arm, 0, 1, wi, pred, thr, cnt)
        return [f(k, 50.0 * k, o) for k in ks], cnt
    for arm in OH_ARMS:
        calls.clear(); o = dict(pt=49, ta=0.0)
        r, cnt = run(arm, [0, 1, 2, 3, 4, 5], o)
        if r != [False, False, False, False, False, True] or calls != [1, 5]:
            bad.append(f'{arm} episode 1: {r} calls {calls} (keep at the first rest k=1 off the grid, held to 4, re-decided at 5 -> pull)')
        calls.clear(); o2 = dict(pt=48, ta=0.0)
        r, cnt = run(arm, [1, 2, 3, 4, 7, 8], o2)
        if r != [False, False, False, False, False, True]:
            bad.append(f'{arm} new episode: {r} (a fresh decision at k=8 -> pull; the episode-1 keep must not carry over)')
        calls.clear(); o3 = dict(pt=47, ta=0.0)
        r, cnt = run(arm, [9], o3)
        if r != [True] or cnt['noinput'] != 1 or calls:
            bad.append(f'{arm} no inputs: {r} {cnt} (must pull, fail closed)')
        calls.clear(); o4 = dict(pt=46, ta=0.0)
        r, cnt = run(arm, [0, 7], o4)
        if r != [False, False] or calls:
            bad.append(f'{arm} guard off: {r} calls {calls}')
        calls.clear(); o5 = dict(pt=45, ta=0.0)
        r, cnt = run(arm, [10, 11], o5)
        if r != [True] + [True] or cnt['pull'][1] != 1:
            bad.append(f'{arm} grid: {r} {cnt} (pull at 10; the order is gone after a pull)')
    for arm, a, b in (('OHH', (0.09, 1.0, 0.0), (0.11, 1.0, 0.0)), ('OHE', (0.1, 3.9, 0.0), (0.1, 4.1, 0.0)), ('OHD', (0.9, 1.0, 0.01), (0.0, 0.0, -0.01)),
                      ('OHD0', (0.9, 1.0, 0.01), (0.0, 0.0, -0.01))):
        if not (keep_rule(arm, *a, thr) and not keep_rule(arm, *b, thr)):
            bad.append(f'{arm} threshold direction')
    return bad


def mutation_check():
    out = {}
    for nm in ('MUT_SIGN', 'MUT_HOLD', 'MUT_NOINPUT_KEEP'):
        globals()[nm] = 1
        try:
            out[nm] = bool(selftest())
        finally:
            globals()[nm] = 0
    return out


def all_selftests():
    J3.all_selftests(); OH.all_selftests(); JX.run_selftest()
    b = selftest(); m = mutation_check()
    if b or not all(m.values()):
        sys.exit(f'ORDER-HARM POLICY SELFTEST FAILED -> no output: {b} planted errors caught {m}')
    print(f'order-harm policy selftests passed; planted errors caught: {m}', flush=True)
    return m


# ---------------------------------------------------------------- thresholds (predictions only)
def thresholds():
    all_selftests()
    parts = []
    for d in DAYS9:
        R = pd.read_parquet(f'{OH.OUTD}/rows_{d}.parquet', columns=['m', 'side', 'k', 'rung_i', 'depth_now'])
        P = pd.read_parquet(f'{OH.OUTD}/pred_{d}.parquet')
        for c in ('m', 'side', 'k', 'rung_i'):
            if not np.array_equal(P[c].values, R[c].values):
                sys.exit(f'REFUSED: {d} predictions not aligned with the rows on {c}')
        PO = pd.read_parquet(f'{PD.OUTD}/pred_{d}.parquet', columns=['m', 'side', 'k'] + [f'P{j}' for j in PD.JS])
        X = R.merge(PO, on=['m', 'side', 'k'], how='left', validate='many_to_one')
        if len(X) != len(R):
            sys.exit(f'REFUSED: {d} PULL-DEPTH merge changed the row count')
        dn = X.depth_now.values; off = X.rung_i.values
        j = np.clip(np.where(np.isfinite(dn), dn, -1) + 1, 1, len(PD.JS)).astype(int)
        p = X[[f'P{q}' for q in PD.JS]].values[np.arange(len(X)), j - 1]
        keep = ~((~np.isfinite(dn)) | (dn < 0) | ((dn == 0) & (off == 0))) & np.isfinite(p) & (p < 0.25)
        parts.append(pd.DataFrame(dict(day=int(d), rung=off, keep_fb=keep, ph=P.p_harm.values, pe=P.p_harm.values * P.p_sev.values,
                                       pdl=P.p_delta.values)))
    A = pd.concat(parts, ignore_index=True)
    if not np.isfinite(A[['ph', 'pe', 'pdl']].values).all():
        sys.exit('REFUSED: non-finite predictions on the rows')
    s_fb = float(A.keep_fb.mean())
    thr = dict(s_FB=s_fb, tau_H=float(np.quantile(A.ph, s_fb)), eps_E=float(np.quantile(A.pe, s_fb)), theta_D=float(np.quantile(A.pdl, 1.0 - s_fb)),
               rows=int(len(A)))
    A['OHH'] = A.ph < thr['tau_H']; A['OHE'] = A.pe < thr['eps_E']; A['OHD'] = A.pdl > thr['theta_D']; A['OHD0'] = A.pdl > 0.0
    thr['share_by_rung'] = {c: [float(A[A.rung == r][c].mean()) for r in range(NR)] for c in ('keep_fb', 'OHH', 'OHE', 'OHD', 'OHD0')}
    thr['share'] = {c: float(A[c].mean()) for c in ('keep_fb', 'OHH', 'OHE', 'OHD', 'OHD0')}
    os.makedirs(OUTD, exist_ok=True)
    json.dump(thr, open(f'{OUTD}/thresholds.json', 'w'), indent=1)
    print(f'thresholds (predictions only): {json.dumps(thr)}', flush=True)


# ---------------------------------------------------------------- inputs
def day_inputs(day, D):
    SR = pd.read_parquet(f'{SD.OUTD}/rows_{day}.parquet', columns=['m', 'side', 'k'] + SD.NEW_FEAT); SRI = SR.groupby(['m', 'side']).indices
    SRV = SR[SD.NEW_FEAT].to_numpy(dtype=np.float64); SRK = SR.k.values.astype(int)
    P = pd.read_parquet(f'{K.IO}/step5/{day}/rows_rl_d2_pk.parquet', columns=['mid', 'tau', 'p_bn', 'tau_rem', 'sigma'])
    P = P[P.mid.isin(D['W0'])].sort_values(['mid', 'tau'], kind='stable')
    PB = {int(m): (g.tau.values.astype(float).tolist(), g.p_bn.values.astype(float), g.tau_rem.values.astype(float), g.sigma.values.astype(float))
          for m, g in P.groupby('mid', sort=False)}
    return dict(SRI=SRI, SRV=SRV, SRK=SRK, PB=PB, X31=SD.NEW_FEAT.index('x31'))


def window_inputs(W, ctx, DI):
    """Per side: guard flags, episode onsets, and the 88 step inputs (STOP-DEPTH + INFLUENCE, as ORDER-HARM v2's rows_day) at every guard-on,
    unmasked step with a STOP-DEPTH row."""
    m = int(W[0]['m']); wi = dict(vg={}, onset={}, MK={}, lvp={0: {}, 1: {}}, kfl={s: np.asarray(W[s]['kf'], float).tolist() for s in (0, 1)},
                                  tfl={s: ctx[s]['tfl'] for s in (0, 1)})
    rt, pv, trv, sgv = DI['PB'].get(m, ([], np.zeros(0), np.zeros(0), np.zeros(0)))
    for s in (0, 1):
        w = W[s]; vv = np.asarray(w['vkl'], float); vg = ctx[s]['vg']; n = len(vg)
        onset = np.full(n, -1, int); k0 = -1
        for k in range(n):
            if vg[k] and (k == 0 or not vg[k - 1]):
                k0 = k
            onset[k] = k0 if vg[k] else -1
        MK = {}; adv = IN.adv_of(s)
        if (m, s) in DI['SRI']:
            for j_ in DI['SRI'][(m, s)]:
                k = int(DI['SRK'][j_])
                if not (0 <= k < n and vg[k] and vv[k] != OH.MASK_V):
                    continue
                t = float(w['tg'][k]); pk = IN.pbn_at(rt, pv, t)
                if np.isfinite(pk):
                    jj = bisect.bisect_right(rt, t) - 1
                    inf = [adv * (pk - IN.pbn_at(rt, pv, t - 250.0)), adv * (pk - IN.pbn_at(rt, pv, t - 1000.0)),
                           adv * float(DI['SRV'][j_, DI['X31']]), IN.sens_of(pk, float(sgv[jj]), float(trv[jj])), abs(pk - 50.0), float(trv[jj])]
                else:
                    inf = [NAN] * len(IN.INF_FEAT)
                MK[k] = np.concatenate([DI['SRV'][j_], np.asarray(inf, float)])
        wi['vg'][s] = vg; wi['onset'][s] = onset; wi['MK'][s] = MK
    return wi


def make_pred(W, wi, B, st):
    """Predictions for the lane's order o at step k, cached per (side, rung, price, activation, step) within the window (shared by runs)."""
    cache = {}

    def order_x(s, d, o, k, t, S5):
        w = W[s]; pt = int(o['pt'])
        if S5 is w['S5']:
            lvp = wi['lvp'][s].get(pt)
            if lvp is None:
                lvp = QR.lvarr(w['P5'], w['S5'], pt, s); wi['lvp'][s][pt] = lvp
        else:
            lvp = QR.lvarr(w['P5'], S5, pt, s)
        k0 = int(wi['onset'][s][k]); b = int(w['best'][k]); b0k = int(w['best'][k0]) if k0 >= 0 and w['best'][k0] >= 0 else b
        return OH.order_inputs(o, t, d, s, b, b0k, lvp, S5, w['P5'], int(w['jk'][k]), wi['kfl'][s], wi['tfl'][s])

    def pred(s, d, o, k, t):
        key = (s, d, int(o['pt']), round(float(o['ta']), 3), k)
        if key in cache:
            return cache[key]
        ordf = order_x(s, d, o, k, t, W[s]['S5'])
        st['inputs'] += 1
        if st['inputs'] % CHECK_EVERY == 0:              # control (iii): frames not yet known at t perturbed -> identical inputs
            S5 = W[s]['S5']; kf = np.asarray(wi['kfl'][s], float); S5p = S5.copy(); fut = kf > t; S5p[fut] = S5[fut] * 3.0 + 11.0
            ordp = order_x(s, d, o, k, t, S5p)
            st['lookahead_checks'] += 1
            st['lookahead_bad'] += int(not all((x == y) or (x != x and y != y) for x, y in zip(ordf, ordp)))
        x = np.concatenate([wi['MK'][s][k], np.asarray(ordf, float)]).astype(np.float32)[None, :]
        out = tuple(float(B[nm].predict(x)[0]) for nm in BOOSTERS)
        cache[key] = out
        return out
    pred.order_x = order_x
    return pred


def load_boosters(day):
    import lightgbm as lgb
    B = {nm: lgb.Booster(model_file=f'{OH.OUTD}/models/{nm}_{day}.txt') for nm in BOOSTERS}
    for nm, b in B.items():
        if b.feature_name() != OH.FEAT:
            sys.exit(f'REFUSED: booster {nm}_{day} inputs differ from ORDER-HARM v2 FEAT')
    return B


def arm_dec(arm, W, ctx, wi, pred, thr, cnt):
    if arm in BASE_ARMS:
        dec, idle = J3.arm_dec(arm, W, ctx)
        if dec is None:
            return None, idle
        return {(s, d): counted(f, ctx[s]['vg'], d, cnt) for (s, d), f in dec.items()}, idle
    return {(s, d): oh_dec(arm, s, d, wi, pred, thr, cnt) for s in (0, 1) for d in range(NR)}, 1


def day_setup(day):
    D = K.load_day(day)
    PO = pd.read_parquet(f'{PD.OUTD}/pred_{day}.parquet'); POI = PO.groupby(['m', 'side']).indices
    return D, PO, POI, day_inputs(day, D)


def window_pairs(D):
    pend = {}
    for w in K.window_sides(D):
        pend.setdefault(w['m'], {})[w['side']] = w
        if len(pend[w['m']]) < 2:
            continue
        W = pend.pop(w['m'])
        if not np.array_equal(W[0]['tg'], W[1]['tg']) or not np.array_equal(W[0]['tf'], W[1]['tf']):
            sys.exit(f'REFUSED: window {W[0]["m"]}: the two sides do not share the grid / frames')
        yield W


# ---------------------------------------------------------------- control (ii): input parity with ORDER-HARM v2's rows
def parity_day(day):
    all_selftests()
    T0 = time.time(); D, PO, POI, DI = day_setup(day); B = load_boosters(day)
    R = pd.read_parquet(f'{OH.OUTD}/rows_{day}.parquet', columns=['m', 'side', 'k', 'rung_i'] + OH.FEAT)
    P = pd.read_parquet(f'{OH.OUTD}/pred_{day}.parquet')
    RV = R[OH.FEAT].to_numpy(dtype=np.float64); PV = P[['p_harm', 'p_sev', 'p_delta']].to_numpy(dtype=np.float64)
    RI = {(int(a), int(b), int(c), int(e)): i for i, (a, b, c, e) in enumerate(zip(R.m.values, R.side.values, R.k.values, R.rung_i.values))}
    iD, iA = OH.FEAT.index('depth_now'), OH.FEAT.index('age')
    st = dict(windows=0, recorded=0, key_found=0, matched=0, input_differ=0, pred_differ=0, inputs=0, lookahead_checks=0, lookahead_bad=0)
    for W in window_pairs(D):
        st['windows'] += 1; n = W[0]['n']; m = int(W[0]['m']); ctx = J3.window_ctx(W, PO, POI, n); wi = window_inputs(W, ctx, DI)
        pred = make_pred(W, wi, B, st); rec = []

        def recorder(s, d):
            vg = wi['vg'][s]; MK = wi['MK'][s]

            def dec(k, t, cur):
                if vg[k] and k % EVERY == 0 and k in MK:
                    ordf = pred.order_x(s, d, cur, k, t, W[s]['S5'])
                    rec.append((s, k, d, np.concatenate([MK[k], np.asarray(ordf, float)]), pred(s, d, cur, k, t)))
                return False
            return dec
        J3.sim_joint(W, 'M', 50.0, None, {0: [False] * n, 1: [False] * n}, decide={(s, d): recorder(s, d) for s in (0, 1) for d in range(NR)},
                     idle_only=1, mode='none')
        for (s, k, d, x, pr) in rec:
            st['recorded'] += 1; i = RI.get((m, s, k, d))
            if i is None:
                continue
            st['key_found'] += 1
            if not (x[iD] == RV[i, iD] and abs(x[iA] - RV[i, iA]) < 1e-6):
                continue
            st['matched'] += 1
            st['input_differ'] += int(not all((a == b) or (a != a and b != b) for a, b in zip(x, RV[i])))
            st['pred_differ'] += int(max(abs(a - b) for a, b in zip(pr, PV[i])) > 1e-9)
        for s in (0, 1):
            W[s].pop('cache', None)
    os.makedirs(OUTD, exist_ok=True)
    json.dump(st, open(f'{OUTD}/parity_{day}.json', 'w'), indent=1)
    if st['input_differ'] or st['pred_differ'] or st['lookahead_bad'] or st['matched'] < 0.5 * max(st['key_found'], 1) or not st['matched']:
        sys.exit(f'ORDER-HARM POLICY INPUT PARITY FAILED on {day}: {st}')
    print(f'parity {day}: {st} in {time.time() - T0:.0f} s', flush=True)


# ---------------------------------------------------------------- replay
def eval_day(day):
    import pf_gap_top_makers as GP
    all_selftests()
    thr = json.load(open(f'{OUTD}/thresholds.json'))
    T0 = time.time(); D, PO, POI, DI = day_setup(day); B = load_boosters(day); st = D['stat']
    S1d = f'{K.IO}/step1/{day}'
    tkf = pd.read_parquet(f'{S1d}/legs_tk.parquet', columns=['tk']).tk.values
    LT = D['LT']; ix = LT.index.values
    grp = np.where(tkf[ix] >= 0, tkf[ix], -1 - ix)
    WSg = pd.Series(TS.ws_groups(LT.how.values, grp), index=LT.index)
    st.update(windows=0, runs=0, viol={v_: 0 for v_ in J3.VIOL}, maxexp={}, over_cap=0, inputs=0, lookahead_checks=0, lookahead_bad=0)
    cnt = {}; fills = []; inv = []
    for W in window_pairs(D):
        st['windows'] += 1; m = W[0]['m']; n = W[0]['n']
        ctx = J3.window_ctx(W, PO, POI, n); wi = window_inputs(W, ctx, DI); pred = make_pred(W, wi, B, st)
        src = {}
        for s in (0, 1):
            if (m, s) in D['LI']:
                l = D['LT'].iloc[D['LI'][(m, s)]]
                o_ = np.lexsort(((-l.ptick.values if s == 0 else l.ptick.values), l.anc.values)); lrow = l.index.values[o_]
            else:
                lrow = np.zeros(0, 'int64')
            src[s] = WSg.reindex(lrow).values.astype(bool) if len(lrow) else np.zeros(0, bool)
        tend = W[0]['te']; jte = int(np.searchsorted(ctx[0]['tf'], tend, side='right')) - 1; mid_end = ctx[0]['mff'][max(jte, 0)]
        yup = 100.0 * W[0]['yup']
        for q, shf in CFGS:
            cfg = f'{q}{int(shf)}'
            for arm in ARMS:
                for mode, ck in CAPS:
                    cn = J3.cap_name(mode, ck); c_ = cnt.setdefault((cfg, arm, cn), dict(keep=[0] * NR, pull=[0] * NR, noinput=0))
                    dec, idle = arm_dec(arm, W, ctx, wi, pred, thr, c_)
                    r = J3.sim_joint(W, q, shf, ck * SIZE, {0: ctx[0]['vg'], 1: ctx[1]['vg']}, decide=dec, idle_only=idle, mode=mode)
                    st['runs'] += 1
                    for v_ in J3.VIOL:
                        st['viol'][v_] += r['stat']['viol'][v_]
                    e = JX.max_exposure(r); mx = max(e[0], e[1])
                    st['maxexp'][cn] = max(st['maxexp'].get(cn, 0.0), mx); st['over_cap'] += int(mx > ck * SIZE + JX.TOL)
                    inv.append((int(day), cfg, arm, cn, m, r['stat']['maxpos'], r['endpos'], sum(r['stat']['pulls']), sum(r['stat']['skip']),
                                sum(r['stat']['hard_block']), mx))
                    for s in (0, 1):
                        c2 = ctx[s]; sg = c2['sg']; tf, mff, tfl = c2['tf'], c2['mff'], c2['tfl']; anc = W[s]['anc']
                        for d in range(NR):
                            for f in r['fills'][(s, d)]:
                                i, pt, qq, thr_, age, ph = f[0], int(f[1]), f[2], f[3], f[4], f[5]
                                t = float(anc[i])
                                if not (W[s]['ws'] <= t < W[s]['ws'] + 61000.0):
                                    continue
                                d_ = GP.depth_of(tfl, W[s]['P5'], W[s]['opp'], s, t, pt)
                                v1 = sg * (mff[max(int(np.searchsorted(tf, t + 1000.0, side='right')) - 1, 0)] - pt)
                                v5 = sg * (mff[max(int(np.searchsorted(tf, t + 5000.0, side='right')) - 1, 0)] - pt)
                                fills.append((int(day), cfg, arm, cn, s, d, 'WS' if src[s][i] else 'E2', -9 if d_ is None else d_, bool(thr_),
                                              age < 100.0, ph == 'pend', float(qq), v1, v5, sg * (mid_end - pt), sg * (yup - pt), float(age)))
        for s in (0, 1):
            W[s].pop('cache', None)
        if st['windows'] % 24 == 0:
            r2 = K.rss_mib(); print(f'{day}: {st["windows"]} windows, {st["inputs"]} model inputs, {time.time() - T0:.0f} s, rss {r2:.0f} MiB', flush=True)
            if r2 > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r2:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    F = pd.DataFrame(fills, columns=['day', 'cfg', 'arm', 'capk', 'side', 'rung', 'src', 'depth', 'thr', 'young', 'pend', 'q', 'v1', 'v5', 'vend', 'vset', 'age'])
    R3 = pd.read_parquet(f'{J3.OUTD}/fills_{day}.parquet'); reg_bad = {}
    for arm in BASE_ARMS:
        for cfg in ('M0', 'M50', 'M250'):
            for mode, ck in CAPS:
                cn = J3.cap_name(mode, ck)
                X = F[(F.capk == cn) & (F.arm == arm) & (F.cfg == cfg)].drop(columns=['capk']).reset_index(drop=True)
                Y = R3[(R3.capk == cn) & (R3.arm == arm) & (R3.cfg == cfg)].drop(columns=['capk']).reset_index(drop=True)
                if not FB.same_rows(X, Y):
                    reg_bad[f'{arm} {cfg} {cn}'] = (len(X), len(Y))
    st['regression_bad'] = reg_bad
    if any(st['viol'].values()) or reg_bad or st['over_cap'] or st['lookahead_bad'] or not st['lookahead_checks']:
        sys.exit(f'ORDER-HARM POLICY CONTROL FAILED on {day} -> no output: viol {st["viol"]}, regression {reg_bad}, over cap {st["over_cap"]}, '
                 f'look-ahead {st["lookahead_bad"]} of {st["lookahead_checks"]}')
    os.makedirs(OUTD, exist_ok=True)
    F.to_parquet(f'{OUTD}/fills_{day}.parquet', index=False)
    pd.DataFrame(inv, columns=['day', 'cfg', 'arm', 'capk', 'm', 'maxpos', 'endpos', 'pulls', 'skip', 'hard_block', 'maxexp']
                 ).to_parquet(f'{OUTD}/inv_{day}.parquet', index=False)
    st['counts'] = {' '.join(k_): v_ for k_, v_ in cnt.items()}
    json.dump(dict(stat=st, days=len(D['W0']) / 288.0), open(f'{OUTD}/day_{day}.json', 'w'), indent=1, default=float)
    print(f'{day}: windows {st["windows"]}, runs {st["runs"]}, model inputs {st["inputs"]}, violations {st["viol"]}, max exposure {st["maxexp"]} '
          f'in {time.time() - T0:.0f} s', flush=True)


# ---------------------------------------------------------------- report
def report():
    stats = {d: json.load(open(f'{OUTD}/day_{d}.json')) for d in DAYS9}
    par = {d: json.load(open(f'{OUTD}/parity_{d}.json')) for d in PAR_DAYS}
    thr = json.load(open(f'{OUTD}/thresholds.json'))
    A = pd.concat([pd.read_parquet(f'{OUTD}/fills_{d}.parquet') for d in DAYS9], ignore_index=True)
    I = pd.concat([pd.read_parquet(f'{OUTD}/inv_{d}.parquet') for d in DAYS9], ignore_index=True)
    nbad = int((~np.isfinite(A.v1)).sum()); A = A[np.isfinite(A.v1)]
    dd = {d: stats[d]['days'] for d in DAYS9}; ndays = sum(dd.values())
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    CN = [J3.cap_name(mo, k) for mo, k in CAPS]

    def cell(arm, cfg, cn, srcs=('WS',), d=None, h='v1', aged=True, rungs=None):
        x = A[(A.arm == arm) & (A.cfg == cfg) & (A.capk == cn) & A.src.isin(srcs)]
        if aged:
            x = x[~x.young]
        if rungs is not None:
            x = x[x.rung.isin(rungs)]
        if d:
            x = x[x.day == int(d)]
        return float((x.q * x[h]).sum()) / 100.0 / (dd[d] if d else ndays)
    runs = sum(stats[d]['stat']['runs'] for d in DAYS9)
    mx = {}
    for d in DAYS9:
        for cn, v in stats[d]['stat']['maxexp'].items():
            mx[cn] = max(mx.get(cn, 0.0), v)
    keepsh = {}
    for d in DAYS9:
        for k_, v_ in stats[d]['stat']['counts'].items():
            a_ = keepsh.setdefault(k_, dict(keep=[0] * NR, pull=[0] * NR, noinput=0))
            for r_ in range(NR):
                a_['keep'][r_] += v_['keep'][r_]; a_['pull'][r_] += v_['pull'][r_]
            a_['noinput'] += v_['noinput']
    L = [f'ORDER-HARM POLICY in JOINT-LADDER v3\'s lane, 9 consumed replay days, first 60 s, both sides, shared position, client-known decisions, hard '
         f'caps K x {SIZE:.0f} sh; as of {asof}; controls passed: input parity on {len(PAR_DAYS)} days ({sum(p["matched"] for p in par.values())} matched '
         f'decision rows of {sum(p["key_found"] for p in par.values())} with ORDER-HARM v2\'s row present, 0 inputs / predictions differ); no-look-ahead '
         f'falsifier {sum(stats[d]["stat"]["lookahead_checks"] for d in DAYS9)} checks in the replay, 0 differ; LDG / FB25 / OFB2 = v3 row for row; '
         f'conservation / accounting on all {runs} runs, 0 violations; max true exposure {mx} sh (<= cap on every run); fills without a 1 s value {nbad}',
         '', f'THRESHOLDS (predictions only, matched to FB25\'s keep share {thr["s_FB"]:.3f} on {thr["rows"]} ORDER-HARM v2 rows): tau_H {thr["tau_H"]:.4f}, '
         f'eps_E {thr["eps_E"]:.4f} c/sh, theta_D {thr["theta_D"]:+.4f} c/sh; keep share by rung 0-5 on those rows: ' +
         '; '.join(f'{c} ' + ' '.join(f'{v:.2f}' for v in thr['share_by_rung'][c]) for c in ('keep_fb', 'OHH', 'OHE', 'OHD', 'OHD0')),
         '', 'KEEP SHARE IN THE LANE (250 ms grid decisions while resting with the guard on), M 50: arm cap: all | by rung 0-5 | fail-closed (no inputs)']
    for arm in ARMS:
        if arm == 'LDG':
            continue
        for cn in CN:
            a_ = keepsh.get(f'M50 {arm} {cn}')
            if not a_:
                continue
            tk, tp = sum(a_['keep']), sum(a_['pull'])
            L.append(f'  {arm:<4} {cn:<6}: {tk / max(tk + tp, 1):.3f} | ' + ' '.join(f'{a_["keep"][r_] / max(a_["keep"][r_] + a_["pull"][r_], 1):.2f}'
                                                                              for r_ in range(NR)) + f' | {a_["noinput"]}')
    L += ['', 'PART 1: $/day w/o young: WS 1 s | E2 1 s | all trades 1 s | WS 5 s | all fills marked at 60 s | at settlement || pulls a day | placements '
          'refused by the cap a day | max |position| per window median / p95 / max sh']
    for cfg in ('M0', 'M50', 'M250'):
        for arm in ARMS:
            for cn in CN:
                x = I[(I.arm == arm) & (I.cfg == cfg) & (I.capk == cn)]
                L.append(f'  {cfg:<4} {arm:<4} {cn:<6} {cell(arm, cfg, cn):+8.1f} | {cell(arm, cfg, cn, ("E2",)):+8.1f} | '
                         f'{cell(arm, cfg, cn, ("WS", "E2")):+8.1f} | {cell(arm, cfg, cn, h="v5"):+8.1f} | {cell(arm, cfg, cn, ("WS", "E2"), h="vend", aged=False):+8.1f} | '
                         f'{cell(arm, cfg, cn, ("WS", "E2"), h="vset", aged=False):+9.1f} || {x.pulls.sum() / ndays:7.0f} | {x.hard_block.sum() / ndays:7.0f} | '
                         f'{x.maxpos.median():4.0f} / {x.maxpos.quantile(0.95):4.0f} / {x.maxpos.max():4.0f}')
        L.append('')
    R = {}
    L.append('READINGS (R630 vs FB25: BETTER / WORSE; R631 vs LDG: PAYS / HURTS; at M 0 AND M 50 on >= 7 of 9 days)')
    for a in OH_ARMS:
        for b, nm, lab_up, lab_dn in (('FB25', 'R630', 'BETTER', 'WORSE'), ('LDG', 'R631', 'PAYS', 'HURTS')):
            for cn in CN:
                up = {c: sum(cell(a, c, cn, d=d) > cell(b, c, cn, d=d) for d in DAYS9) for c in ('M0', 'M50', 'M250')}
                dn = {c: sum(cell(a, c, cn, d=d) < cell(b, c, cn, d=d) for d in DAYS9) for c in ('M0', 'M50', 'M250')}
                lab = lab_up if min(up['M0'], up['M50']) >= 7 else (lab_dn if min(dn['M0'], dn['M50']) >= 7 else 'NOT-SHOWN')
                diff = {c: cell(a, c, cn) - cell(b, c, cn) for c in ('M0', 'M50', 'M250')}
                R[f'{nm} {a} vs {b} {cn}'] = dict(label=lab, up=up, down=dn, diff=diff)
                L.append(f'  {nm} {a} vs {b:<4} {cn:<6}: {lab:<10} (M 0 {diff["M0"]:+7.1f}, {up["M0"]}/9; M 50 {diff["M50"]:+7.1f}, {up["M50"]}/9; '
                         f'M 250 {diff["M250"]:+8.1f}, {up["M250"]}/9)')
    L += ['', 'R632 RUNG 0 (printed): WS $/day w/o young at 1 s, rung 0 | rungs 1-5, M 0 / M 50']
    for cn in CN:
        for arm in ARMS:
            L.append(f'  {arm:<4} {cn:<6}: ' + ' ; '.join(f'{c} {cell(arm, c, cn, rungs=[0]):+7.1f} | {cell(arm, c, cn, rungs=list(range(1, NR))):+7.1f}'
                                                       for c in ('M0', 'M50')))
    L += ['', '  per day, WS $/day w/o young, M 50, hard5: ' + ' | '.join(ARMS)]
    for d in DAYS9:
        L.append(f'    {d} ' + ' | '.join(f'{cell(a_, "M50", "hard5", d=d):+7.1f}' for a_ in ARMS))
    L += ['', 'C1 / C12: $/day at 1 s by order age, M 50, hard5: WS (< 100 | 100-500 | >= 500 ms) || E2 (same)']
    for arm in ARMS:
        x = A[(A.arm == arm) & (A.cfg == 'M50') & (A.capk == 'hard5')]; parts = []
        for s_ in ('WS', 'E2'):
            y = x[x.src == s_]
            parts.append(' | '.join(f'{float((y.q * y.v1)[(y.age >= lo) & (y.age < hi)].sum()) / 100.0 / ndays:+8.1f}' for _, lo, hi in J3.AGEB))
        L.append(f'  {arm:<4} {parts[0]} || {parts[1]}')
    L.append('')
    L.append('READINGS: ' + json.dumps({k_: v_['label'] for k_, v_ in R.items()}))
    os.makedirs(OUTD, exist_ok=True)
    json.dump(R, open(f'{OUTD}/order_harm_policy.json', 'w'), indent=1, default=float)
    open(f'{OUTD}/order_harm_policy.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        all_selftests()
    elif cmd == 'thresholds':
        thresholds()
    elif cmd == 'parity':
        parity_day(sys.argv[2])
    elif cmd == 'eval':
        eval_day(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit('usage: pf_order_harm_policy.py selftest | thresholds | parity <day> | eval <day> | report')
