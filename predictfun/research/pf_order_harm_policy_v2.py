"""ORDER-HARM POLICY v2 (2026-11-14, session ce68ebe0): the keep-with-option policy (ORDER-HARM v3's chronological models) in JOINT-LADDER v3's
lane, against SAME-CADENCE baselines, judged by a WHOLE-PORTFOLIO acceptance criterion. Supersedes the v1 design (pf_order_harm_policy.py,
73950293 / 041edeee), stopped at its 4th day after the USER's review, outputs deleted unread.

USER review:
 "1. The model comparison changes cancellation frequency too. The new policy generally reconsiders every 250 ms, versus 50 ms for FB25. Any
     improvement could partly come from cancelling less often. Add a baseline using exactly the same cadence."
 "2. The success criterion still covers a favorable subset. It can declare 'BETTER/PAYS' using WS-timed fills while broader trade coverage or
     timing stress performs poorly. Require a separate whole-portfolio acceptance criterion."
 "3. There is still no forward validation. Thresholds use predictions across all nine consumed dates, and leave-one-day-out training includes
     later dates when testing earlier ones. Useful for development, but improvement needs chronological training/calibration followed by
     untouched days."
 "4. The target still differs from the intended action ... 'keep now, retain the option to cancel later', including queue value." (v3)

TEST DAYS: ORDER-HARM v3's chronological test days 10-27, 10-31, 11-01, 11-02, 11-03; each replayed with the models of its fold (trained on the
days before its calibration day; PULL-DEPTH refit on the same days; theta_M set on the calibration day). Consumed days: development only.
LANE: pf_order_harm_client_lane, derived from the frozen JOINT-LADDER v3 engine (client-known decisions, both sides, rungs 0-5 x 30 sh, shared position); caps hard 10 / 20 / 50 x
size and none (USER: "much larger cap"); fill clocks M 0 / 50 / 250.
CADENCE C (every arm marked C, and ORDER-HARM v3's labels): in a guard episode, an order resting is decided at its first resting step and then
on every 50 ms step; a new episode starts fresh; once pulled, the order stays
out while the guard is on and re-enters at the back 250 ms after it clears (the lane's rule).
ARMS:
  LDG     pull every resting order at the guard (cadence-free: it pulls at the onset step under any cadence);
  FB25    FUTURE-BEST's rule at every 50 ms step (the fold's PULL-DEPTH model) - the previous comparison;
  FB25C   the same rule at cadence C                                  -> R651 CADENCE = FB25C - FB25;
  OFB2C   the rule with the realized 2 s move at cadence C (perfect foresight; a ceiling);
  OHQ0    keep iff A_hat > 0 at cadence C (v3's final model: keep now with the option to cancel later; no threshold), rung 0 included;
  OHQM    keep iff A_hat > theta_M at cadence C (theta_M matched to FB25's keep share on the calibration day).
  A model decision without inputs (masked start) -> pull (fail closed), counted.
READINGS DECLARED BEFORE THE FIRST RUN (5 test days; day rule >= 4 of 5):
 R650 WS (a diagnostic, never acceptance): OHQ0 / OHQM vs FB25C and vs LDG, WS-timed $/day without fills on orders < 100 ms old at 1 s, per cap:
      higher at M 0 AND M 50 on >= 4 days -> BETTER (vs FB25C) / PAYS (vs LDG); lower at both on >= 4 -> WORSE / HURTS; else NOT-SHOWN.
 R651 CADENCE (printed): FB25C minus FB25; now both 50 ms, required to match fill for fill.
 R652 PORTFOLIO-ACCEPT (the acceptance criterion, per model arm and cap): the WHOLE PORTFOLIO = every fill on every trade (WS and E2) marked at
      the common 60.2 s mid, after the 60 s stop and cancellation tail (cash plus inventory), counted both with and without fills on orders < 100 ms old. ACCEPT iff,
      against EACH of LDG and FB25C, at EACH fill clock M 0 / M 50 / M 250, under EACH age treatment, the arm is higher pooled over the 5 days
      AND on >= 4 of 5 days (12 conditions); otherwise REJECT, naming every failing condition.
 VERDICT per model arm: DEVELOPMENT-ACCEPT iff R652 ACCEPT at hard 50 (the headline cap); else NOT-ACCEPTED. A development accept is only a
      candidate for a frozen forward test on >= 5 untouched days (none readable here: the forward blocks are reserved; the USER decides).
CONTROLS (any failure -> no output): selftests (v3 lane, ORDER-HARM v3, FUTURE-BEST, exposure) and this script's cadence wrapper with planted
errors (250 ms default cadence; no inputs -> keep; generic cadence-wrapper regressions); the folds are chronological (v3's
fit files and client50 artifact contract); terminal cash/inventory reconciliation on every run; INPUT PARITY on each test day: the keep-all joint lane (no cap, M 50: the labels' rules) records the inputs at cadence-C decisions;
wherever its order is v3's label order (same window, side, step, rung, depth and age) A_hat must equal v3's saved prediction (<= 1e-9) on
every matched decision and all 97 inputs the label row's stored float32 inputs on every 20th (memory: keys only in memory); the no-look-ahead falsifier on every 20th new decision input; LDG equals v3's saved LDG fills at hard 10 and none
row for row; v3's conservation / accounting checks and the true exposure <= the cap on every hard-cap run; memory self-guard (2 GiB).
LIMITS: consumed days (C8), development only; replay rule M (B1a); the guard and FRONT-PULL's P are leave-one-day-out upstream models; labels
from rungs alone (the replay couples them); the 2 s tail bounds the queue value counted.
Usage (run_light, PF_SLOT=claude): pf_order_harm_policy_v2.py selftest | parity <day> | eval <day> | report; driver run_order_harm_policy_v2.sh
"""
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
import pf_future_best as FB                               # noqa: E402
import pf_order_harm_v2 as OH                             # noqa: E402
import pf_order_harm_v3 as V3                             # noqa: E402
import pf_order_harm_client_lane as CL                    # noqa: E402
import pf_joint_ladder_v3 as J3                           # noqa: E402
import pf_joint_ladder_v3_exposure as JX                  # noqa: E402
import pf_order_harm_policy as P1                         # noqa: E402  (day_inputs, window_inputs, window_pairs: unchanged)

TESTS = tuple(V3.FOLDS)
OUTD = f'{K.IO}/order_harm_policy_v2_client50_20261114'
SIZE = J3.SIZE; NR = J3.NR; EVERY = V3.EVERY
assert EVERY * 50 == CL.DECISION_MS
CAPS = (('hard', 10), ('hard', 20), ('hard', 50), ('none', None)); HEAD_CAP = 'hard50'; V3_CAPS = ('hard10', 'none')
CFGS = J3.CFGS
ARMS = ('LDG', 'FB25', 'FB25C', 'OFB2C', 'OHQ0', 'OHQM'); MODEL_ARMS = ('OHQ0', 'OHQM')
CHECK_EVERY = 20
MEM_CAP_MIB = 2048.0
NAN = float('nan')
MUT_GRID = MUT_HOLD = MUT_EVERY_STEP = MUT_NOINPUT_KEEP = MUT_CADENCE_250 = 0  # planted errors (selftest only)


# ---------------------------------------------------------------- cadence C
def cadence(keep_fn, vg, onset, cnt, d, every=None):
    """Wrap keep_fn(k, t, cur) -> keep? into cadence C; returns the lane callback (True = pull)."""
    every = (5 if MUT_CADENCE_250 else EVERY) if every is None else every
    last = {}

    def dec(k, t, cur):
        if not vg[k]:
            return False
        k0 = int(onset[k]); key = id(cur); h = last.get(key)
        same_ep = h is not None and (h[0] == k0 or MUT_HOLD)
        on_grid = (k % every == 0) if MUT_GRID else ((k - k0) % every == 0)
        if same_ep and not on_grid and not MUT_EVERY_STEP:
            return not h[1]
        keep = bool(keep_fn(k, t, cur))
        last[key] = (k0, keep)
        if on_grid or not same_ep:
            cnt['keep' if keep else 'pull'][d] += 1
        return not keep
    return dec


def model_keep(MK, pred, s, d, theta, cnt):
    def f(k, t, cur):
        if k not in MK:
            cnt['noinput'] += 1
            return bool(MUT_NOINPUT_KEEP)
        return pred(s, d, cur, k, t) > theta
    return f


def selftest():
    bad = []
    vg = [False, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True]
    onset = []; k0 = -1
    for k, g in enumerate(vg):
        if g and (k == 0 or not vg[k - 1]):
            k0 = k
        onset.append(k0 if g else -1)
    MK = {k: 1 for k in range(16) if vg[k] and k != 14}
    A = {1: 0.5, 2: -9.0, 6: -0.5, 9: 0.5, 10: -9.0, 14: 0.5}
    calls = []

    def pred(s, d, o, k, t):
        calls.append(k); return A.get(k, -1.0)

    def run(ks, o, theta=0.0):
        cnt = dict(keep=[0] * NR, pull=[0] * NR, noinput=0)
        f = cadence(model_keep(MK, pred, 0, 1, theta, cnt), vg, onset, cnt, 1, every=5)
        return [f(k, 50.0 * k, o) for k in ks], cnt
    calls.clear(); r, c = run([0, 1, 2, 3, 4, 5, 6], dict(pt=49))
    if r != [False, False, False, False, False, False, True] or calls != [1, 6]:
        bad.append(f'episode 1 {r} calls {calls} (decide at the onset 1, hold, re-decide at 6 = onset + 5 -> pull)')
    calls.clear(); r, c = run([1, 2, 8, 9, 10, 11, 13], dict(pt=48))
    if r != [False, False, False, False, False, False, False] or calls != [1, 9]:
        bad.append(f'new episode {r} calls {calls} (fresh decision at its onset 9, held at 10-13)')
    calls.clear(); r, c = run([1, 11], dict(pt=45))
    if r != [False, True] or calls != [1, 11]:
        bad.append(f'a new episode must re-decide even off its grid {r} calls {calls} (episode 1 keep must not carry into episode 2)')
    calls.clear(); r, c = run([14], dict(pt=47))
    if r != [True] or c['noinput'] != 1:
        bad.append(f'no inputs {r} {c}')
    calls.clear(); r, c = run([1, 6], dict(pt=46), theta=-1.0)
    if r != [False, False]:
        bad.append(f'threshold {r}')
    # Production/default cadence must score again on the next 50 ms step.
    calls.clear(); cnt = dict(keep=[0] * NR, pull=[0] * NR, noinput=0)
    f = cadence(model_keep(MK, pred, 0, 1, 0.0, cnt), vg, onset, cnt, 1)
    o = dict(pt=49)
    got = [f(k, 50.0 * k, o) for k in (1, 2)]
    if got != [False, True] or calls != [1, 2]:
        bad.append(f'50 ms default cadence {got}, calls {calls}')
    return bad


def mutation_check():
    out = {}
    for nm in ('MUT_GRID', 'MUT_HOLD', 'MUT_EVERY_STEP', 'MUT_NOINPUT_KEEP', 'MUT_CADENCE_250'):
        globals()[nm] = 1
        try:
            out[nm] = bool(selftest())
        finally:
            globals()[nm] = 0
    return out


def all_selftests():
    J3.all_selftests(); V3.all_selftests(); JX.run_selftest()
    b = selftest(); m = mutation_check()
    if b or not all(m.values()):
        sys.exit(f'ORDER-HARM POLICY v2 SELFTEST FAILED -> no output: {b} planted errors caught {m}')
    print(f'order-harm policy v2 selftests passed; planted errors caught: {m}', flush=True)
    return m


# ---------------------------------------------------------------- the fold's models and inputs
def fold(day):
    import lightgbm as lgb
    fz = json.load(open(f'{V3.OUTD}/fit_{day}.json')); CL.require_contract(fz)
    info = fz['info']; thr = fz['thresholds']; CL.require_contract(thr)
    tr, cal = info['train'], info['cal']
    if not (all(DI < K.DAYS9.index(cal) for DI in (K.DAYS9.index(x) for x in tr)) and K.DAYS9.index(cal) < K.DAYS9.index(day) and info['test'] == day):
        sys.exit(f'REFUSED: fold of {day} not chronological: {tr} / {cal}')
    B = lgb.Booster(model_file=f'{V3.OUTD}/models/a_{day}.txt')
    if B.feature_name() != V3.FEAT:
        sys.exit('REFUSED: the fold model inputs differ from ORDER-HARM v3 FEAT')
    PO = pd.read_parquet(f'{V3.OUTD}/pd_{day}_{day}.parquet')
    return dict(B=B, theta_M=float(thr['theta_M']), PO=PO, POI=PO.groupby(['m', 'side']).indices, train=tr, cal=cal)


def make_pred(W, wi, B, st):
    cache = {}

    def order_x(s, d, o, k, t, S5):
        w = W[s]; pt = int(o['pt'])
        lvp = wi['lvp'][s].get(pt) if S5 is w['S5'] else None
        if lvp is None:
            lvp = QR.lvarr(w['P5'], S5, pt, s)
            if S5 is w['S5']:
                wi['lvp'][s][pt] = lvp
        k0 = int(wi['onset'][s][k]); b = int(w['best'][k]); b0k = int(w['best'][k0]) if k0 >= 0 and w['best'][k0] >= 0 else b
        return OH.order_inputs(o, t, d, s, b, b0k, lvp, S5, w['P5'], int(w['jk'][k]), wi['kfl'][s], wi['tfl'][s])

    def pred(s, d, o, k, t):
        key = (s, d, int(o['pt']), round(float(o['ta']), 3), k)
        if key in cache:
            return cache[key]
        ordf = order_x(s, d, o, k, t, W[s]['S5'])
        st['inputs'] += 1
        if st['inputs'] % CHECK_EVERY == 0:
            S5 = W[s]['S5']; kf = np.asarray(wi['kfl'][s], float); S5p = S5.copy(); fut = kf > t; S5p[fut] = S5[fut] * 3.0 + 11.0
            ordp = order_x(s, d, o, k, t, S5p)
            st['lookahead_checks'] += 1
            st['lookahead_bad'] += int(not all((x == y) or (x != x and y != y) for x, y in zip(ordf, ordp)))
        x = np.concatenate([wi['MK'][s][k], np.asarray(ordf, float)]).astype(np.float32)[None, :]
        out = float(B.predict(x)[0])
        cache[key] = out
        return out
    pred.order_x = order_x
    return pred


def arm_dec(arm, W, ctx, wi, pred, F_, cnt):
    if arm == 'LDG':
        return None, 0
    if arm == 'FB25':
        dec, idle = J3.arm_dec('FB25', W, ctx)
        out = {}
        for (s, d), f in dec.items():
            def g(k, t, cur, f=f, s=s, d=d):          # counted on cadence C's grid for comparable keep shares (decisions unchanged)
                r = f(k, t, cur)
                if ctx[s]['vg'][k] and (k - int(wi['onset'][s][k])) % EVERY == 0:
                    cnt['pull' if r else 'keep'][d] += 1
                return r
            out[(s, d)] = g
        return out, idle
    out = {}
    for s in (0, 1):
        for d in range(NR):
            if arm in ('FB25C', 'OFB2C'):
                fb = (FB.fb_dec(ctx[s]['vg'], W[s]['best'], s, ctx[s]['Pm'], d, tau=0.25) if arm == 'FB25C' else
                      FB.fb_dec(ctx[s]['vg'], W[s]['best'], s, ctx[s]['Pm'], d, mv=ctx[s]['mv2']))
                keep_fn = (lambda fb_: (lambda k, t, cur: not fb_(k, t, cur)))(fb)
            else:
                keep_fn = model_keep(wi['MK'][s], pred, s, d, 0.0 if arm == 'OHQ0' else F_['theta_M'], cnt)
            out[(s, d)] = cadence(keep_fn, ctx[s]['vg'], wi['onset'][s], cnt, d)
    return out, 1


# ---------------------------------------------------------------- control: input parity with v3's labels
def parity_day(day):
    """Control: inputs and A_hat of the keep-all joint lane's decisions against v3's label rows (memory: keys only in memory; predictions
    compared on every matched decision, the 97 stored float32 inputs on every 20th, from the first)."""
    import pyarrow.parquet as pq
    all_selftests()
    T0 = time.time(); F_ = fold(day); D = K.load_day(day); DI = P1.day_inputs(day, D)
    CL.require_contract(json.load(open(f'{V3.OUTD}/rows_{day}.json')))
    RK = pq.read_table(f'{V3.OUTD}/rows_{day}.parquet', columns=['m', 'side', 'k', 'rung_i', 'depth_now', 'age']).to_pandas()
    Pr = pq.read_table(f'{V3.OUTD}/pred_{day}.parquet', columns=['m', 'side', 'k', 'rung_i', 'a_final']).to_pandas()
    for c in ('m', 'side', 'k', 'rung_i'):
        if not np.array_equal(Pr[c].values, RK[c].values):
            sys.exit(f'REFUSED: v3 predictions not aligned with its rows on {c}')
    KEYV = ((RK.m.values.astype(np.int64) * 2 + RK.side.values.astype(np.int64)) * 8192 + RK.k.values.astype(np.int64)) * 8 + RK.rung_i.values.astype(np.int64)
    order = np.argsort(KEYV, kind='stable'); KS = KEYV[order]
    if len(KS) > 1 and (np.diff(KS) == 0).any():
        sys.exit('REFUSED: duplicate label keys')
    DN = RK.depth_now.values.astype(np.float32); AG = RK.age.values.astype(np.float32); PV = Pr.a_final.values.astype(np.float64)
    del RK, Pr
    iD, iA = V3.FEAT.index('depth_now'), V3.FEAT.index('age')
    st = dict(windows=0, recorded=0, key_found=0, matched=0, checked=0, input_differ=0, pred_differ=0, inputs=0, lookahead_checks=0, lookahead_bad=0)
    checks = []
    for W in P1.window_pairs(D):
        st['windows'] += 1; n = W[0]['n']; m = int(W[0]['m']); ctx = J3.window_ctx(W, F_['PO'], F_['POI'], n); wi = P1.window_inputs(W, ctx, DI)
        pred = make_pred(W, wi, F_['B'], st); rec = []
        dcnt = dict(keep=[0] * NR, pull=[0] * NR, noinput=0)

        def recorder(s, d):
            MK = wi['MK'][s]

            def keep_fn(k, t, cur):
                if k in MK:
                    ordf = pred.order_x(s, d, cur, k, t, W[s]['S5'])
                    rec.append((s, k, d, np.concatenate([MK[k], np.asarray(ordf, float)]), pred(s, d, cur, k, t)))
                return True
            return cadence(keep_fn, ctx[s]['vg'], wi['onset'][s], dcnt, d)
        CL.sim_joint(W, 'M', 50.0, None, {0: ctx[0]['vg'], 1: ctx[1]['vg']}, decide={(s, d): recorder(s, d) for s in (0, 1) for d in range(NR)},
                     idle_only=1, mode='none')
        for (s, k, d, x, pr) in rec:
            st['recorded'] += 1
            kv = ((m * 2 + s) * 8192 + k) * 8 + d; p_ = int(np.searchsorted(KS, kv))
            if p_ >= len(KS) or KS[p_] != kv:
                continue
            i = int(order[p_]); st['key_found'] += 1; x32 = x.astype(np.float32)
            if not (x32[iD] == DN[i] and x32[iA] == AG[i]):  # the same order: ticks behind the best and age
                continue
            st['matched'] += 1
            st['pred_differ'] += int(abs(pr - PV[i]) > 1e-9)
            if st['matched'] % CHECK_EVERY == 1:
                checks.append((i, x32))
        for s in (0, 1):
            W[s].pop('cache', None)
    if checks:                                            # the sampled decisions' inputs against the stored label rows (float32: the model's precision)
        checks.sort(key=lambda z: z[0])
        LV = V3.feats_rows(day, np.array([c[0] for c in checks]))
        for (i, x32), lv in zip(checks, LV):
            st['checked'] += 1
            st['input_differ'] += int(not all((a_ == b_) or (a_ != a_ and b_ != b_) for a_, b_ in zip(x32, lv)))
    os.makedirs(OUTD, exist_ok=True)
    json.dump(dict(st, **CL.stamp()), open(f'{OUTD}/parity_{day}.json', 'w'), indent=1)
    if (st['input_differ'] or st['pred_differ'] or st['lookahead_bad'] or not st['matched'] or not st['checked']
            or st['matched'] < 0.5 * st['key_found']):
        sys.exit(f'ORDER-HARM POLICY v2 INPUT PARITY FAILED on {day}: {st}')
    print(f'parity {day}: {st} in {time.time() - T0:.0f} s', flush=True)


# ---------------------------------------------------------------- replay (one test day)
def eval_day(day):
    import pf_gap_top_makers as GP
    all_selftests()
    T0 = time.time(); F_ = fold(day); D = K.load_day(day); DI = P1.day_inputs(day, D); st = D['stat']
    S1d = f'{K.IO}/step1/{day}'
    tkf = pd.read_parquet(f'{S1d}/legs_tk.parquet', columns=['tk']).tk.values
    LT = D['LT']; ix = LT.index.values
    grp = np.where(tkf[ix] >= 0, tkf[ix], -1 - ix)
    WSg = pd.Series(TS.ws_groups(LT.how.values, grp), index=LT.index)
    st.update(windows=0, runs=0, viol={v_: 0 for v_ in J3.VIOL}, maxexp={}, over_cap=0, inputs=0, lookahead_checks=0, lookahead_bad=0,
              theta_M=F_['theta_M'], train=F_['train'], cal=F_['cal'], terminal_checks=0, terminal_bad=0)
    cnt = {}; fills = []; inv = []
    for W in P1.window_pairs(D):
        st['windows'] += 1; m = W[0]['m']; n = W[0]['n']
        ctx = J3.window_ctx(W, F_['PO'], F_['POI'], n); wi = P1.window_inputs(W, ctx, DI); pred = make_pred(W, wi, F_['B'], st)
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
                    dec, idle = arm_dec(arm, W, ctx, wi, pred, F_, c_)
                    capv = None if ck is None else ck * SIZE
                    r = CL.sim_joint(W, q, shf, capv, {0: ctx[0]['vg'], 1: ctx[1]['vg']}, decide=dec, idle_only=idle, mode=mode)
                    st['runs'] += 1
                    terminal = CL.portfolio_terminal(r, ctx[0]['tf'], ctx[0]['mff'])
                    terminal_value_sum = 0.0
                    for v_ in J3.VIOL:
                        st['viol'][v_] += r['stat']['viol'][v_]
                    e = JX.max_exposure(r); mx = max(e[0], e[1])
                    st['maxexp'][cn] = max(st['maxexp'].get(cn, 0.0), mx)
                    if capv is not None:
                        st['over_cap'] += int(mx > capv + JX.TOL)
                    inv.append((int(day), cfg, arm, cn, m, r['stat']['maxpos'], r['endpos'], sum(r['stat']['pulls']), sum(r['stat']['skip']),
                                sum(r['stat']['hard_block']), mx, terminal['cash'], terminal['position'], terminal['mark'], terminal['pnl'], terminal['time']))
                    for s in (0, 1):
                        c2 = ctx[s]; sg = c2['sg']; tf, mff, tfl = c2['tf'], c2['mff'], c2['tfl']; anc = W[s]['anc']
                        for d in range(NR):
                            for f in r['fills'][(s, d)]:
                                i, pt, qq, thr_, age, ph = f[0], int(f[1]), f[2], f[3], f[4], f[5]
                                t = float(anc[i])
                                t_exec = t - shf
                                if not (W[s]['ws'] <= t_exec <= terminal['time']):
                                    raise ValueError('fill outside the portfolio execution interval')
                                vend = sg * (terminal['mark'] - pt)
                                terminal_value_sum += float(qq) * vend / 100.0
                                d_ = GP.depth_of(tfl, W[s]['P5'], W[s]['opp'], s, t, pt)
                                v1 = sg * (mff[max(int(np.searchsorted(tf, t + 1000.0, side='right')) - 1, 0)] - pt)
                                v5 = sg * (mff[max(int(np.searchsorted(tf, t + 5000.0, side='right')) - 1, 0)] - pt)
                                fills.append((int(day), cfg, arm, cn, s, d, 'WS' if src[s][i] else 'E2', -9 if d_ is None else d_, bool(thr_),
                                              age < 100.0, ph == 'pend', float(qq), v1, v5, vend, sg * (yup - pt), float(age),
                                              sg * (mid_end - pt), int(m), t_exec, terminal['time'], pt))
                    st['terminal_checks'] += 1
                    st['terminal_bad'] += int(abs(terminal_value_sum - terminal['pnl']) > 1e-6)
        for s in (0, 1):
            W[s].pop('cache', None)
        if st['windows'] % 24 == 0:
            r2 = K.rss_mib(); print(f'{day}: {st["windows"]} windows, {st["inputs"]} model inputs, {time.time() - T0:.0f} s, rss {r2:.0f} MiB', flush=True)
            if r2 > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r2:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    F = pd.DataFrame(fills, columns=['day', 'cfg', 'arm', 'capk', 'side', 'rung', 'src', 'depth', 'thr', 'young', 'pend', 'q', 'v1', 'v5', 'vend', 'vset', 'age', 'vend60', 'm', 't_exec', 'terminal_ms', 'pt'])
    R3 = pd.read_parquet(f'{J3.OUTD}/fills_{day}.parquet'); reg_bad = {}
    for cfg in ('M0', 'M50', 'M250'):
        for cn in V3_CAPS:
            X = F[(F.capk == cn) & (F.arm == 'LDG') & (F.cfg == cfg)].drop(columns=['capk', 'vend', 'm', 't_exec', 'terminal_ms', 'pt']).rename(columns={'vend60': 'vend'}).reset_index(drop=True)
            Y = R3[(R3.capk == cn) & (R3.arm == 'LDG') & (R3.cfg == cfg)].drop(columns=['capk']).reset_index(drop=True)
            X = X[Y.columns]
            if not FB.same_rows(X, Y):
                reg_bad[f'LDG {cfg} {cn}'] = (len(X), len(Y))
    st['regression_bad'] = reg_bad
    for cfg in ('M0', 'M50', 'M250'):
        for mode, ck in CAPS:
            cn = J3.cap_name(mode, ck)
            a = F[(F.capk == cn) & (F.cfg == cfg) & (F.arm == 'FB25')].drop(columns=['arm']).reset_index(drop=True)
            b = F[(F.capk == cn) & (F.cfg == cfg) & (F.arm == 'FB25C')].drop(columns=['arm']).reset_index(drop=True)
            if not FB.same_rows(a, b):
                reg_bad[f'50ms cadence equivalence {cfg} {cn}'] = (len(a), len(b))
    if (any(st['viol'].values()) or reg_bad or st['over_cap'] or st['lookahead_bad'] or not st['lookahead_checks']
            or st['terminal_bad'] or st['terminal_checks'] != st['runs']):
        sys.exit(f'ORDER-HARM POLICY v2 CONTROL FAILED on {day} -> no output: viol {st["viol"]}, regression {reg_bad}, over cap {st["over_cap"]}, '
                 f'look-ahead {st["lookahead_bad"]} of {st["lookahead_checks"]}; terminal mismatches {st["terminal_bad"]}')
    os.makedirs(OUTD, exist_ok=True)
    F.to_parquet(f'{OUTD}/fills_{day}.parquet', index=False)
    pd.DataFrame(inv, columns=['day', 'cfg', 'arm', 'capk', 'm', 'maxpos', 'endpos', 'pulls', 'skip', 'hard_block', 'maxexp',
                              'terminal_cash', 'terminal_position', 'terminal_mid', 'terminal_pnl', 'terminal_ms']
                 ).to_parquet(f'{OUTD}/inv_{day}.parquet', index=False)
    st['counts'] = {' '.join(k_): v_ for k_, v_ in cnt.items()}
    json.dump(dict(stat=st, days=len(D['W0']) / 288.0, **CL.stamp()), open(f'{OUTD}/day_{day}.json', 'w'), indent=1, default=float)
    print(f'{day}: windows {st["windows"]}, runs {st["runs"]}, model inputs {st["inputs"]}, violations {st["viol"]}, max exposure {st["maxexp"]} '
          f'in {time.time() - T0:.0f} s', flush=True)


# ---------------------------------------------------------------- report
def report():
    stats = {d: json.load(open(f'{OUTD}/day_{d}.json')) for d in TESTS}
    par = {d: json.load(open(f'{OUTD}/parity_{d}.json')) for d in TESTS}
    for receipt in list(stats.values()) + list(par.values()):
        CL.require_contract(receipt)
    A = pd.concat([pd.read_parquet(f'{OUTD}/fills_{d}.parquet') for d in TESTS], ignore_index=True)
    I = pd.concat([pd.read_parquet(f'{OUTD}/inv_{d}.parquet') for d in TESTS], ignore_index=True)
    nbad = int((~np.isfinite(A.v1)).sum())
    if not np.isfinite(A.vend).all() or (A.t_exec > A.terminal_ms).any():
        raise ValueError('invalid terminal portfolio rows')
    dd = {d: stats[d]['days'] for d in TESTS}; ndays = sum(dd.values())
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    CN = [J3.cap_name(mo, k) for mo, k in CAPS]

    def cell(arm, cfg, cn, srcs=('WS',), d=None, h='v1', aged=True):
        x = A[(A.arm == arm) & (A.cfg == cfg) & (A.capk == cn) & A.src.isin(srcs)]
        if aged:
            x = x[~x.young]
        if d:
            x = x[x.day == int(d)]
        return float((x.q * x[h]).sum()) / 100.0 / (dd[d] if d else ndays)

    def port(arm, cfg, cn, aged, d=None):
        return cell(arm, cfg, cn, ('WS', 'E2'), d=d, h='vend', aged=aged)
    runs = sum(stats[d]['stat']['runs'] for d in TESTS)
    mx = {}
    for d in TESTS:
        for cn, v in stats[d]['stat']['maxexp'].items():
            mx[cn] = max(mx.get(cn, 0.0), v)
    keepsh = {}
    for d in TESTS:
        for k_, v_ in stats[d]['stat']['counts'].items():
            a_ = keepsh.setdefault(k_, dict(keep=[0] * NR, pull=[0] * NR, noinput=0))
            for r_ in range(NR):
                a_['keep'][r_] += v_['keep'][r_]; a_['pull'][r_] += v_['pull'][r_]
            a_['noinput'] += v_['noinput']
    L = [f'ORDER-HARM POLICY v2: keep-with-option (ORDER-HARM v3, chronological folds) in JOINT-LADDER v3\'s lane; test days {", ".join(TESTS)} '
         f'(consumed: development only); as of {asof}; controls passed: input parity {sum(p["matched"] for p in par.values())} matched decisions of '
         f'{sum(p["key_found"] for p in par.values())}, 0 differ; look-ahead falsifier {sum(stats[d]["stat"]["lookahead_checks"] for d in TESTS)} '
         f'checks, 0 differ; LDG = v3 row for row; conservation on {runs} runs, 0 violations; max true exposure {mx} sh (<= the cap on every '
         f'hard-cap run); fills without a 1 s value {nbad}',
         '  folds: ' + '; '.join(f'{d[4:]}: train {",".join(x[4:] for x in stats[d]["stat"]["train"])} cal {stats[d]["stat"]["cal"][4:]} theta_M '
                                 f'{stats[d]["stat"]["theta_M"]:+.3f}' for d in TESTS), '',
         'KEEP SHARE (decisions on cadence C\'s grid while resting with the guard on), M 50: arm cap: all | by rung 0-5 | fail-closed']
    for arm in ARMS[1:]:
        for cn in CN:
            a_ = keepsh.get(f'M50 {arm} {cn}')
            if not a_:
                continue
            tk, tp = sum(a_['keep']), sum(a_['pull'])
            L.append(f'  {arm:<5} {cn:<6}: {tk / max(tk + tp, 1):.3f} | ' + ' '.join(f'{a_["keep"][r_] / max(a_["keep"][r_] + a_["pull"][r_], 1):.2f}'
                                                                               for r_ in range(NR)) + f' | {a_["noinput"]}')
    L += ['', 'PART 1: $/day: WS 1 s w/o young | E2 1 s w/o young | all trades 1 s w/o young | PORTFOLIO (all trades, marked after the cancellation tail (60.2 s)): all ages | '
          'w/o young | settlement, all || pulls a day | refused a day | max |position| median / p95 / max sh']
    for cfg in ('M0', 'M50', 'M250'):
        for arm in ARMS:
            for cn in CN:
                x = I[(I.arm == arm) & (I.cfg == cfg) & (I.capk == cn)]
                L.append(f'  {cfg:<4} {arm:<5} {cn:<6} {cell(arm, cfg, cn):+8.1f} | {cell(arm, cfg, cn, ("E2",)):+8.1f} | '
                         f'{cell(arm, cfg, cn, ("WS", "E2")):+8.1f} | {port(arm, cfg, cn, False):+8.1f} | {port(arm, cfg, cn, True):+8.1f} | '
                         f'{cell(arm, cfg, cn, ("WS", "E2"), h="vset", aged=False):+8.1f} || {x.pulls.sum() / ndays:7.0f} | {x.hard_block.sum() / ndays:7.0f} | '
                         f'{x.maxpos.median():4.0f} / {x.maxpos.quantile(0.95):4.0f} / {x.maxpos.max():4.0f}')
        L.append('')
    R = {}
    L.append('R650 WS (diagnostic; >= 4 of 5 days at M 0 AND M 50)')
    for a in MODEL_ARMS:
        for b, lu, ld in (('FB25C', 'BETTER', 'WORSE'), ('LDG', 'PAYS', 'HURTS')):
            for cn in CN:
                up = {c: sum(cell(a, c, cn, d=d) > cell(b, c, cn, d=d) for d in TESTS) for c in ('M0', 'M50', 'M250')}
                dn = {c: sum(cell(a, c, cn, d=d) < cell(b, c, cn, d=d) for d in TESTS) for c in ('M0', 'M50', 'M250')}
                lab = lu if min(up['M0'], up['M50']) >= 4 else (ld if min(dn['M0'], dn['M50']) >= 4 else 'NOT-SHOWN')
                diff = {c: cell(a, c, cn) - cell(b, c, cn) for c in ('M0', 'M50', 'M250')}
                R[f'R650 {a} vs {b} {cn}'] = dict(label=lab, up=up, down=dn, diff=diff)
                L.append(f'  {a} vs {b:<5} {cn:<6}: {lab:<9} (M 0 {diff["M0"]:+7.1f}, {up["M0"]}/5; M 50 {diff["M50"]:+7.1f}, {up["M50"]}/5; '
                         f'M 250 {diff["M250"]:+8.1f}, {up["M250"]}/5)')
    L += ['', 'R651 CADENCE (printed): FB25C minus FB25, $/day: WS 1 s w/o young | all trades 1 s w/o young | portfolio all ages | portfolio w/o young']
    for cn in CN:
        L.append(f'  {cn:<6}: ' + ' ; '.join(f'{c} {cell("FB25C", c, cn) - cell("FB25", c, cn):+7.1f} | '
                                             f'{cell("FB25C", c, cn, ("WS", "E2")) - cell("FB25", c, cn, ("WS", "E2")):+7.1f} | '
                                             f'{port("FB25C", c, cn, False) - port("FB25", c, cn, False):+7.1f} | '
                                             f'{port("FB25C", c, cn, True) - port("FB25", c, cn, True):+7.1f}' for c in ('M0', 'M50', 'M250')))
    L += ['', 'R652 PORTFOLIO-ACCEPT (all trades, marked after the cancellation tail (60.2 s); vs LDG and FB25C x M 0 / 50 / 250 x all ages / w/o young: pooled higher AND >= 4/5 '
          'days, all 12 conditions)']
    for a in MODEL_ARMS:
        for cn in CN:
            fails = []; cond = []
            for b in ('LDG', 'FB25C'):
                for c in ('M0', 'M50', 'M250'):
                    for aged in (False, True):
                        dif = port(a, c, cn, aged) - port(b, c, cn, aged)
                        upd = sum(port(a, c, cn, aged, d) > port(b, c, cn, aged, d) for d in TESTS)
                        ok = dif > 0 and upd >= 4
                        cond.append(dict(vs=b, cfg=c, aged=aged, diff=dif, up=upd, ok=ok))
                        if not ok:
                            fails.append(f'{b} {c} {"w/o young" if aged else "all ages"} ({dif:+.1f}, {upd}/5)')
            lab = 'ACCEPT' if not fails else 'REJECT'
            R[f'R652 {a} {cn}'] = dict(label=lab, conditions=cond)
            L.append(f'  {a} {cn:<6}: {lab}' + ('' if not fails else f' - failing {len(fails)} of 12: ' + '; '.join(fails)))
    L.append('')
    for a in MODEL_ARMS:
        v = 'DEVELOPMENT-ACCEPT' if R[f'R652 {a} {HEAD_CAP}']['label'] == 'ACCEPT' else 'NOT-ACCEPTED'
        R[f'VERDICT {a}'] = dict(label=v)
        L.append(f'VERDICT {a} (headline cap {HEAD_CAP}): {v}' + (' (a candidate for a frozen forward test on >= 5 untouched days only)'
                                                                 if v == 'DEVELOPMENT-ACCEPT' else ''))
    L += ['', f'  per day, M 50, {HEAD_CAP}: portfolio all ages || WS 1 s w/o young: ' + ' | '.join(ARMS)]
    for d in TESTS:
        L.append(f'    {d} ' + ' | '.join(f'{port(a_, "M50", HEAD_CAP, False, d):+7.1f}' for a_ in ARMS) + ' || ' +
                 ' | '.join(f'{cell(a_, "M50", HEAD_CAP, d=d):+7.1f}' for a_ in ARMS))
    L += ['', f'C1 / C12: $/day at 1 s by order age, M 50, {HEAD_CAP}: WS (< 100 | 100-500 | >= 500 ms) || E2 (same)']
    for arm in ARMS:
        x = A[(A.arm == arm) & (A.cfg == 'M50') & (A.capk == HEAD_CAP)]; parts = []
        for s_ in ('WS', 'E2'):
            y = x[x.src == s_]
            parts.append(' | '.join(f'{float((y.q * y.v1)[(y.age >= lo) & (y.age < hi)].sum()) / 100.0 / ndays:+8.1f}' for _, lo, hi in J3.AGEB))
        L.append(f'  {arm:<5} {parts[0]} || {parts[1]}')
    L.append('')
    L.append('READINGS: ' + json.dumps({k_: v_['label'] for k_, v_ in R.items()}))
    os.makedirs(OUTD, exist_ok=True)
    json.dump(dict(readings=R, **CL.stamp()), open(f'{OUTD}/order_harm_policy_v2.json', 'w'), indent=1, default=float)
    open(f'{OUTD}/order_harm_policy_v2.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        all_selftests()
    elif cmd == 'parity':
        parity_day(sys.argv[2])
    elif cmd == 'eval':
        eval_day(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit('usage: pf_order_harm_policy_v2.py selftest | parity <day> | eval <day> | report')
