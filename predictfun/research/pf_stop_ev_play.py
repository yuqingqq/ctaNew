"""STOP-EV-PLAY (2026-11-15, session aea5bfb9; USER: "We can test 1 first"): replay "rest when quiet, pull when the fill model sees
activity" - the v2 fill head (the model's real skill: AUC 0.82-0.86) driving keep / pull / re-entry - in the CORRECTED queue lane, against
keep-all, the guard and the harm-only model. Development only (consumed days); the v2 falsifier (T5v2) has not run.

RULE QR(theta, r): every 250 ms (grid step k % 5 == 0; between them nothing changes), each resting order is pulled when the v2 fill head's
  P(fill within its avoidable 1 s) on its client-known state exceeds theta; an empty rung is (re-)posted only while P(fill) of a FRESH order
  there (age 0, full size, at the back of the shown queue) is below r x theta (hysteresis). No model row at the step -> keep / allow (rest
  by default). Inputs = the model's own: v6's 71 market inputs of the STOP-MODEL row (m, side, k) + the order inputs from the lane order's
  client-known state (pf_stop_ev_accounting.known_state, reproduced incrementally: received frames, acknowledged fills only).
LANE: pf_stop_queue_activation_lane.sim_joint_g (activation-snapshot-queue-v1) for EVERY arm; 6 rungs x 30 sh a side, no cap, re-post at
  once (REPOST_MS 0), unknown queue 0; clocks M 50 (primary) and M 250; whole portfolio at the 60.2 s mark (C15), with / without fills on
  orders < 100 ms old.
ARMS: KA keep-all; LDG the guard (pull while v <= -0.10); MD1000 the harm-only v6 model with its calibration-day (tau_on, r) from
  play_cal_<test>_v6.json (chosen on the OLD lane - stated); QR the rule with (theta, r) chosen on the fold's calibration day; QRS the
  SHIFTED control: QR's rule and parameters with the market inputs of the side's previous window at the same step (the same order-state
  inputs, no information about this window's market).
CHOICE (chronological): per fold, (theta, r) over THETAS x HYST maximising the calibration day's whole-portfolio $ without young fills at
  M 50 (ties -> the larger theta, then the larger r), applied unchanged to the test day.
READINGS DECLARED BEFORE THE RUN (5 fold test days; day rule >= 4 of 5; whole portfolio, w/o young fills unless stated):
 R980 VS-KEEP-ALL: QR - KA > 0 pooled AND on >= 4 days at M 50 AND M 250, with AND without young fills -> BEATS-KEEP-ALL.
 R981 VS-HARM-ONLY: QR - MD1000 > 0 pooled AND on >= 4 days at M 50 AND M 250 -> BEATS-HARM-ONLY; < 0 the same way -> LOSES-TO-HARM-ONLY;
      else NOT-SHOWN.
 R982 VS-GUARD: QR - LDG, the same rule -> BEATS-LDG / LOSES-TO-LDG / NOT-SHOWN.
 R983 INFORMATION: QR - QRS > 0 pooled AND on >= 4 days at M 50 -> INFORMED, else NOT-SHOWN.
 Printed: the WS / E2 split, fills by order age, quote uptime per rung, pulls and gated entries, the chosen parameters.
CONTROLS (any failure -> no output): QR with theta above every probability equals KA fill for fill on every test window (the override
  engaged, never triggering); lane violations 0; the incremental order state equals pf_stop_ev_accounting.known_state on a sample of
  >= 1,000 calls a day (0 mismatches); batched predictions equal single ones; synthetic: the parity on a toy lane, an always-pull rule
  removes every fill, planted errors caught (the venue frame instead of the known frame; the entry gate ignored).
LIMITS: consumed days; replay rule M; E2 point anchors; the fill head is v2 (certified labels, its falsifier pending); MD1000's parameters
  were chosen on the old lane; local model, whole-portfolio evaluation.
Usage (run_light, PF_SLOT=claude): pf_stop_ev_play.py selftest | cal <test day> | test <test day> | report; driver run_stop_ev_play.sh.
"""
import bisect
import collections
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_stop_ev_model as V                              # noqa: E402  (the v2 heads, inputs, folds)
import pf_stop_play as SP                                 # noqa: E402  (the harm-only arms, values, sources)

E, M, K, AC = V.E, V.M, V.K, V.AC
QA = E.QA
OUTD = f'{K.IO}/stop_ev_play_20261115'
THETAS = (0.10, 0.20, 0.30, 0.50); HYST = (1.0, 0.6); COMBOS = [(t, r) for t in THETAS for r in HYST]
CAD = 5; THETA_OFF = 2.0; Z = 30.0; NR = int(M.CL.NR); L_NOTE = float(M.CL.L_NOTE); PARITY_EVERY = 97
CFGS = (('M', 50.0), ('M', 250.0))
MUT_FEAT_VENUE = MUT_GATE_OFF = 0                         # planted errors (selftest only)
MUTS = ('MUT_FEAT_VENUE', 'MUT_GATE_OFF')
ORDER_KEYS = ('side', 'price', 'rung', 'depth', 'age_ms', 'submitted_quantity', 'known_remaining', 'known_filled', 'queue_initial_known',
              'queue_ahead_known', 'shown_known', 'queue_initial_unknown', 'effective_horizon_ms')


def shown(w, j, side, price):
    """pf_stop_ev_accounting.shown_at: the size shown at price in frame j (NaN beyond the 5th shown level)."""
    P = w['P5'][j]; S = w['S5'][j]; last = P[-1]
    if last >= 0 and ((price < last) if side == 0 else (price > last)):
        return math.nan
    return float(np.asarray(S, float)[np.asarray(P) == price].sum())


def order_state(w, cur, k, t, side, rung, terminal):
    """The order's client-known state at t, incremental over received frames (pf_stop_ev_accounting.known_state's definitions)."""
    kf = w['kf']; jt = (bisect.bisect_right(w['tf'], t) - 1) if MUT_FEAT_VENUE else int(w['jk'][k]); pt = cur['pt']
    if '_ev_ja' not in cur:
        ja = bisect.bisect_right(kf, cur['ta']) - 1; cur['_ev_ja'] = ja; cur['_ev_last'] = ja
        ini = shown(w, ja, side, pt) if (ja >= 0 and ja <= jt) else math.nan
        cur['_ev_ini'] = ini; cur['_ev_min'] = ini if math.isfinite(ini) else math.inf
    ja = cur['_ev_ja']
    if ja >= 0 and ja <= jt:
        if cur['_ev_last'] < ja:
            cur['_ev_last'] = ja
        for j in range(cur['_ev_last'] + 1, jt + 1):
            x = shown(w, j, side, pt)
            if math.isfinite(x) and x < cur['_ev_min']:
                cur['_ev_min'] = x
        cur['_ev_last'] = max(cur['_ev_last'], jt)
        ini = cur['_ev_ini']; ahead = cur['_ev_min'] if math.isfinite(cur['_ev_min']) else math.nan   # min of the initial and later finite sizes
    else:
        ini = math.nan; ahead = math.nan
    filled = sum(float(q) for ft, q in cur['fl'] if ft + L_NOTE <= t)
    sh = shown(w, jt, side, pt) if jt >= 0 else math.nan
    best = float(w['P5'][jt][0]) if jt >= 0 else math.nan
    return dict(side=side, price=float(pt), rung=rung, depth=float(best - pt if side == 0 else pt - best), age_ms=t - cur['ta'], submitted_quantity=Z,
                known_remaining=max(0.0, Z - filled), known_filled=filled, queue_initial_known=ini, queue_ahead_known=ahead, shown_known=sh,
                queue_initial_unknown=float(not math.isfinite(ini)), effective_horizon_ms=min(1000.0, terminal - t))


def fresh_state(w, k, t, side, rung, terminal):
    """A fresh order of rung d posted at t: age 0, full size, behind the queue shown at its price in the known frame."""
    jt = int(w['jk'][k]); b = w['best'][k]
    if jt < 0 or b < 0:
        return None
    pt = int(b) - rung if side == 0 else int(b) + rung
    if not 1 <= pt <= 99:
        return None
    sh = shown(w, jt, side, pt); best = float(w['P5'][jt][0])
    return dict(side=side, price=float(pt), rung=rung, depth=float(best - pt if side == 0 else pt - best), age_ms=0.0, submitted_quantity=Z,
                known_remaining=Z, known_filled=0.0, queue_initial_known=sh, queue_ahead_known=sh, shown_known=sh,
                queue_initial_unknown=float(not math.isfinite(sh)), effective_horizon_ms=min(1000.0, terminal - t))


def order_vector(f, scale, pf_mid):
    """The ORDER inputs (pf_stop_ev_model.order_inputs) of one state dict."""
    sg = 1.0 if f['side'] == 0 else -1.0; sc = scale if (scale == scale and scale > 0) else math.nan
    qa, sh, q0 = f['queue_ahead_known'], f['shown_known'], f['queue_initial_known']
    rel = lambda x: (x / sc) if (math.isfinite(x) and sc == sc) else math.nan
    return [float(f['rung']), f['depth'], f['price'], min(f['age_ms'], 60000.0), f['known_remaining'] / f['submitted_quantity'], rel(qa), rel(sh), rel(q0),
            (qa / sh) if (math.isfinite(qa) and math.isfinite(sh) and sh > 0) else math.nan, f['queue_initial_unknown'], sg * (pf_mid - f['price']),
            f['effective_horizon_ms'] / 1000.0]


class FillPolicy:
    """QR(theta, r) for one window run: P(fill) from the fill head on the lane's own orders, batched once per decision step."""

    def __init__(self, booster, W, X, theta, r, terminal, stat, check=None):
        self.b = booster; self.W = W; self.X = X; self.theta = theta; self.off = r * theta; self.term = terminal; self.st = stat
        self.reg = {}; self.cache = {}; self.ck = -1; self.gate_last = {}; self.check = check

    def _row(self, s, k):
        x = self.X.get(s)
        if x is None or k >= len(x[0]) or x[0][k] < 0:
            return None
        return x[1][x[0][k]], x[2][x[0][k]], x[3][x[0][k]]         # market row, scale, pf_mid

    def _vec(self, s, d, k, t, cur):
        row = self._row(s, k)
        if row is None:
            return None
        f = order_state(self.W[s], cur, k, t, s, d, self.term) if cur is not None else fresh_state(self.W[s], k, t, s, d, self.term)
        if f is None:
            return None
        if cur is not None and self.check is not None:
            self.check(self.W, cur, (s, d), t, f)
        return np.concatenate([row[0], np.asarray(order_vector(f, row[1], row[2]), np.float32)])

    def _batch(self, k, t):
        self.ck = k; self.cache = {}; keys, vecs = [], []
        for (s, d), cur in list(self.reg.items()):
            live = cur is not None and cur['tc'] > t and not cur['ref'] and cur['ta'] <= t
            c_ = cur if live else None
            v = self._vec(s, d, k, t, c_)
            if v is not None:
                keys.append(((s, d), id(c_) if c_ is not None else None)); vecs.append(v)
        if vecs:
            p = self.b.predict(np.vstack(vecs)); self.st['batch_rows'] += len(vecs); self.st['batches'] += 1
            for key, pv in zip(keys, p):
                self.cache[key] = float(pv)

    def prob(self, s, d, k, t, cur):
        if k != self.ck:
            self._batch(k, t)
        key = ((s, d), id(cur) if cur is not None else None)
        if key not in self.cache:
            v = self._vec(s, d, k, t, cur)
            self.cache[key] = float(self.b.predict(v[None, :])[0]) if v is not None else math.nan; self.st['single_rows'] += 1
        return self.cache[key]

    def dec(self, s, d):
        def f(k, t, cur):
            self.reg[(s, d)] = cur
            if k % CAD:
                return False
            p = self.prob(s, d, k, t, cur)
            return bool(p == p and p > self.theta)
        return f

    def gate(self, s, d):
        tg = self.W[0]['tg']

        def g(k):
            if MUT_GATE_OFF:
                return False
            if k % CAD:
                return self.gate_last.get((s, d), False)
            if self.reg.get((s, d)) is not None:
                self.reg[(s, d)] = None
            p = self.prob(s, d, k, float(tg[k]), None)
            self.gate_last[(s, d)] = bool(p == p and p >= self.off)
            return self.gate_last[(s, d)]
        return g


def market_arrays(day, keys_only=False):
    """v6's 71 market inputs of every STOP-MODEL row of the day, with per (m, side) the row index of each grid step k."""
    R = pd.read_parquet(f'{M.OUTD}/rows_{day}.parquet', columns=['m', 'side', 'k', 'scale_w30'])
    X = M.day_matrix(day, 'W30', V.F71) if not keys_only else np.zeros((len(R), len(V.F71)), np.float32)
    pm = X[:, V.F71.index('pf_mid')]; sc = R.scale_w30.values.astype(float); out = {}
    for (m, s), ix in R.groupby(['m', 'side'], sort=False).indices.items():
        kk = R.k.values[ix]; pos = np.full(int(kk.max()) + 1, -1, 'int64'); pos[kk] = np.arange(len(ix))
        out[(int(m), int(s))] = (pos, X[ix], sc[ix], pm[ix])
    return out


def play(test, phase):
    import lightgbm as lgb
    all_selftests()
    if test not in M.TESTS:
        sys.exit(f'REFUSED: {test} is not a fold test day')
    day = M.FOLDS[test][1] if phase == 'cal' else test
    if M.QUNK != 'zero':
        sys.exit('REFUSED: unknown queue must be zero (the primary assumption)')
    T0 = time.time(); bst = lgb.Booster(model_file=f'{V.OUTD}/models_v2/{test}_full_p.txt')
    if bst.num_feature() != len(V.FULL_IN):
        sys.exit('REFUSED: the fill head has a different input count')
    MK = market_arrays(day); D = K.load_day(day); cut = M.boundary_cut(day)
    tkf = pd.read_parquet(f'{K.IO}/step1/{day}/legs_tk.parquet', columns=['tk']).tk.values
    LT = D['LT']; ix = LT.index.values; grp = np.where(tkf[ix] >= 0, tkf[ix], -1 - ix)
    WSg = pd.Series(M.TS.ws_groups(LT.how.values, grp), index=LT.index); TKS = pd.Series(grp, index=LT.index)
    st = collections.Counter(); val = {}; live = {}; order = []
    path = f'{M.OUTD}/pred_fold{test}_{day}_v6.parquet'; PA = SP.pred_arrays(path, M.HP)
    mdc = json.load(open(f'{M.OUTD}/play_cal_{test}_v6.json'))['choice'][str(M.HP)]
    choice = json.load(open(f'{OUTD}/cal_{test}.json'))['choice'] if phase == 'test' else None
    pcheck = dict(n=0, bad=0)

    def check(W, cur, key, t, f):
        st['calls'] += 1
        if st['calls'] % PARITY_EVERY:
            return
        g = AC.known_state(W, cur, key + (0,), t, Z, terminal_ms=float(W[0]['te']) + 200.0)
        pcheck['n'] += 1
        for k_ in ORDER_KEYS:
            a, b = f[k_], g[k_]
            if not ((isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)) or abs(float(a) - float(b)) <= 1e-9):
                pcheck['bad'] += 1; break

    def add(cfg, arm, v_, ls=None):
        a = val.setdefault(cfg, {}).setdefault(arm, {})
        for k_, x in v_.items():
            a[k_] = a.get(k_, 0.0) + x
        if ls is not None:
            b = live.setdefault(cfg, {}).setdefault(arm, [0.0] * NR)
            for i, x in enumerate(ls):
                b[i] += x

    def run(W, q, shf, dec=None, gate=None, aux=None):
        n = W[0]['n']; aux = aux or {0: [False] * n, 1: [False] * n}
        r = QA.sim_joint_g(W, q, shf, None, aux, decide=dec, gate=gate, idle_only=0, size=Z, mode='none', cooldown=M.CL.REPOST_MS)
        st['runs'] += 1; st['viol'] += int(sum(r['stat']['viol'].values()))
        return r

    def qr(W, X, theta, r_, chk=None):
        pol = FillPolicy(bst, W, X, theta, r_, float(W[0]['te']) + 200.0, st, chk)
        return {(s, d): pol.dec(s, d) for s in (0, 1) for d in range(NR)}, {(s, d): pol.gate(s, d) for s in (0, 1) for d in range(NR)}

    prev_X = None
    for W in M.P1.window_pairs(D):
        if W[0]['ws'] < cut:
            continue
        m = int(W[0]['m']); st['windows'] += 1; tf, mff = W[0]['tf_mff']; src = SP.window_src(D, W, WSg, TKS); ws, we = float(W[0]['tg'][0]), float(W[0]['te'])
        X = {s: MK.get((m, s)) for s in (0, 1)}
        if phase == 'cal':
            for (th, rr) in COMBOS:
                r = run(W, 'M', 50.0, *qr(W, X, th, rr)); add('M50', f'QR|{th}|{rr}', SP.arm_value2(r, tf, mff, src))
        else:
            th, rr = choice
            for q, shf in CFGS:
                cfg = f'{q}{int(shf)}'; n = W[0]['n']
                rk = run(W, q, shf); add(cfg, 'KA', SP.arm_value2(rk, tf, mff, src), SP.live_share(rk, ws, we))
                vg = {s: (np.asarray(W[s]['vkl'], float) <= M.GUARD).tolist() for s in (0, 1)}
                rl = run(W, q, shf, aux=vg); add(cfg, 'LDG', SP.arm_value2(rl, tf, mff, src), SP.live_share(rl, ws, we))
                P = {s: PA[0].get((m, s)) for s in (0, 1)}
                dec = {(s, d): SP.model_dec(P[s], W[s]['best'], s, mdc[0]) for s in (0, 1) for d in range(NR)}
                gate = {(s, d): SP.model_gate(P[s], d, mdc[0] * mdc[1]) for s in (0, 1) for d in range(NR)}
                rm = run(W, q, shf, dec, gate); add(cfg, 'MD1000', SP.arm_value2(rm, tf, mff, src), SP.live_share(rm, ws, we))
                rq = run(W, q, shf, *qr(W, X, th, rr, check if cfg == 'M50' else None)); add(cfg, 'QR', SP.arm_value2(rq, tf, mff, src), SP.live_share(rq, ws, we))
                if prev_X is not None:
                    rs = run(W, q, shf, *qr(W, prev_X, th, rr)); add(cfg, 'QRS', SP.arm_value2(rs, tf, mff, src), SP.live_share(rs, ws, we))
                    st[f'qrs_windows_{cfg}'] += 1
                else:
                    add(cfg, 'QRS', SP.arm_value2(rq, tf, mff, src), SP.live_share(rq, ws, we)); st[f'qrs_first_window_{cfg}'] += 1
                if cfg == 'M50':
                    rz = run(W, q, shf, *qr(W, X, THETA_OFF, 1.0)); st['offpar_windows'] += 1
                    st['offpar_differ'] += int(rz['fills'] != rk['fills'] or rz['endpos'] != rk['endpos'])
        prev_X = X
        if st['windows'] % 40 == 0:
            E.mem_guard(f'stop-ev play {phase} {day}')
    st['parity_checked'] = pcheck['n']; st['parity_bad'] = pcheck['bad']
    fails = []
    if st['viol'] or st['offpar_differ']:
        fails.append(f"lane violations {st['viol']}, off-parity differences {st['offpar_differ']}")
    if phase == 'test' and (pcheck['n'] < 1000 or pcheck['bad']):
        fails.append(f'order-state parity with known_state: {pcheck}')
    os.makedirs(OUTD, exist_ok=True); res = dict(test=test, day=day, phase=phase, stat=dict(st), value=val, live=live, md1000_choice=mdc,
                                                 seconds=round(time.time() - T0, 1), manifest=V_manifest())
    if fails:
        json.dump(res, open(f'{OUTD}/refused_{phase}_{test}.json', 'w'), indent=1, default=float)
        sys.exit(f'REFUSED (no output): {fails}')
    if phase == 'cal':
        best = max(COMBOS, key=lambda c: (val['M50'][f'QR|{c[0]}|{c[1]}']['pnl_old'], c[0], c[1])); res['choice'] = list(best)
        json.dump(res, open(f'{OUTD}/cal_{test}.json', 'w'), indent=1, default=float)
        print(f'cal {test} (day {day}): choice {best}; M50 w/o young ' + ', '.join(f'{c} {val["M50"][f"QR|{c[0]}|{c[1]}"]["pnl_old"]:+.0f}' for c in COMBOS)
              + f'; {res["seconds"]} s', flush=True)
    else:
        res['choice'] = choice
        json.dump(res, open(f'{OUTD}/test_{test}.json', 'w'), indent=1, default=float)
        print(f'test {test}: ' + '; '.join(f'{cfg} ' + ', '.join(f'{a} {x["pnl_old"]:+.0f}' for a, x in val[cfg].items()) for cfg in val)
              + f'; parity {pcheck}; off-parity {st["offpar_differ"]} of {st["offpar_windows"]}; {res["seconds"]} s', flush=True)


def V_manifest():
    import hashlib
    return {f: hashlib.sha256(open(f'{HERE}/{f}', 'rb').read()).hexdigest()[:16]
            for f in ('pf_stop_ev_play.py', 'pf_stop_ev_model.py', 'pf_stop_play.py', 'pf_stop_queue_activation_lane.py', 'pf_stop_ev_accounting.py')}


def report():
    J = {t: json.load(open(f'{OUTD}/test_{t}.json')) for t in M.TESTS if os.path.exists(f'{OUTD}/test_{t}.json')}
    if len(J) < len(M.TESTS):
        sys.exit(f'REFUSED: test days missing {[t for t in M.TESTS if t not in J]}')
    if len({json.dumps(x['manifest'], sort_keys=True) for x in J.values()}) > 1:
        sys.exit('REFUSED: days run from different sources')
    arms = ('KA', 'LDG', 'MD1000', 'QR', 'QRS'); T = [f'STOP-EV-PLAY - "rest when quiet, pull when the fill model sees activity" (v2 fill head) in the corrected '
                                                        f'queue lane; 5 chronological test days; as of {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}', '']
    T.append('chosen (theta, r) per fold: ' + ', '.join(f"{t} {x['choice']}" for t, x in J.items()) + '; MD1000 (old-lane choice): '
             + ', '.join(f"{t} {x['md1000_choice']}" for t, x in J.items()))
    T.append('WHOLE PORTFOLIO $ a day (mean of 5 days): arm | M50 all / w/o young | M250 all / w/o young | WS / E2 (M50 w/o young) | quote uptime per rung (M50)')
    mean = lambda cfg, a, k_: float(np.mean([J[t]['value'][cfg][a][k_] for t in J]))
    for a in arms:
        up = np.mean([np.asarray(J[t]['live']['M50'][a]) / max(J[t]['stat']['windows'], 1) for t in J], axis=0)
        T.append(f"  {a:7s} {mean('M50', a, 'pnl'):+9.1f} / {mean('M50', a, 'pnl_old'):+9.1f} | {mean('M250', a, 'pnl'):+9.1f} / {mean('M250', a, 'pnl_old'):+9.1f} | "
                 f"{mean('M50', a, 'ws_old'):+8.1f} / {mean('M50', a, 'e2_old'):+8.1f} | " + ' '.join(f'{u:.2f}' for u in up))
    per = lambda cfg, a, k_: np.asarray([J[t]['value'][cfg][a][k_] for t in M.TESTS])

    def rule(b, cfgs, keys):
        ok_pos = all((per(c, 'QR', k_) - per(c, b, k_)).sum() > 0 and int(((per(c, 'QR', k_) - per(c, b, k_)) > 0).sum()) >= 4 for c in cfgs for k_ in keys)
        ok_neg = all((per(c, 'QR', k_) - per(c, b, k_)).sum() < 0 and int(((per(c, 'QR', k_) - per(c, b, k_)) < 0).sum()) >= 4 for c in cfgs for k_ in keys)
        return ok_pos, ok_neg
    r980 = 'BEATS-KEEP-ALL' if rule('KA', ('M50', 'M250'), ('pnl', 'pnl_old'))[0] else 'NOT-SHOWN'
    p_, n_ = rule('MD1000', ('M50', 'M250'), ('pnl_old',)); r981 = 'BEATS-HARM-ONLY' if p_ else 'LOSES-TO-HARM-ONLY' if n_ else 'NOT-SHOWN'
    p_, n_ = rule('LDG', ('M50', 'M250'), ('pnl_old',)); r982 = 'BEATS-LDG' if p_ else 'LOSES-TO-LDG' if n_ else 'NOT-SHOWN'
    r983 = 'INFORMED' if rule('QRS', ('M50',), ('pnl_old',))[0] else 'NOT-SHOWN'
    T.append('per day M50 w/o young (QR | KA | LDG | MD1000 | QRS): ' + '; '.join(
        f"{t} " + ' | '.join(f"{J[t]['value']['M50'][a]['pnl_old']:+.0f}" for a in ('QR', 'KA', 'LDG', 'MD1000', 'QRS')) for t in M.TESTS))
    T.append('per day M250 w/o young (QR | KA | LDG | MD1000 | QRS): ' + '; '.join(
        f"{t} " + ' | '.join(f"{J[t]['value']['M250'][a]['pnl_old']:+.0f}" for a in ('QR', 'KA', 'LDG', 'MD1000', 'QRS')) for t in M.TESTS))
    T.append('pulls a day (M50): ' + ', '.join(f"{a} {mean('M50', a, 'pulls'):.0f}" for a in arms) + '; gated entries a day: '
             + ', '.join(f"{a} {mean('M50', a, 'gated'):.0f}" for a in arms))
    T.append('by order age (M50, all fills): ' + '; '.join(f"{a} <100 {mean('M50', a, 'age_lt100'):+.0f} / 100-500 {mean('M50', a, 'age_100_500'):+.0f} / >=500 "
                                                         f"{mean('M50', a, 'age_ge500'):+.0f}" for a in arms))
    T += ['', f'R980 {r980}; R981 {r981}; R982 {r982}; R983 {r983} (development only: consumed days, replay rule M, E2 point anchors, v2 falsifier pending)']
    open(f'{OUTD}/stop_ev_play.txt', 'w').write('\n'.join(T) + '\n')
    json.dump(dict(readings=dict(R980=r980, R981=r981, R982=r982, R983=r983), days=J), open(f'{OUTD}/stop_ev_play.json', 'w'), indent=1, default=float)
    print('\n'.join(T), flush=True)


# ---------------------------------------------------------------- self-tests
def selftest():
    import lightgbm as lgb
    bad = []
    bid = [(710.0, 50, 30.0, 2), (910.0, 50, 25.0, 4), (1210.0, 47, 40.0, 6), (1260.0, 49, 12.0, 8)]
    ask = [(300.0, 52, 15.0, 3), (800.0, 53, 40.0, 5), (1500.0, 52, 18.0, 7)]
    (w0, _, _, _), (w1, _, _, _) = E.toy_w(0, bid, drop=(500.0, 0, 20.0)), E.toy_w(1, ask)
    W = {0: w0, 1: w1}; n = w0['n']; aux = {0: [False] * n, 1: [False] * n}
    rk = QA.sim_joint_g(W, 'M', 50.0, None, aux, size=Z, mode='none', cooldown=M.CL.REPOST_MS)
    rng = np.random.default_rng(5); F = len(V.FULL_IN)
    Xb = rng.normal(0, 1, (600, F)).astype(np.float32); yb = (rng.random(600) < 0.3).astype(float)
    b = lgb.train({**V.PARAMS, 'min_data_in_leaf': 20}, lgb.Dataset(Xb, yb), 10)
    X = {s: (np.arange(n, dtype='int64'), rng.normal(0, 1, (n, len(V.F71))).astype(np.float32), np.full(n, 200.0), np.full(n, 51.0)) for s in (0, 1)}
    st = collections.Counter(); mis = []

    def chk(W_, cur, key, t, f):
        g = AC.known_state(W_, cur, key + (0,), t, Z, terminal_ms=float(W_[0]['te']) + 200.0)
        for k_ in ORDER_KEYS:
            a, c = f[k_], g[k_]
            if not ((isinstance(a, float) and isinstance(c, float) and math.isnan(a) and math.isnan(c)) or abs(float(a) - float(c)) <= 1e-9):
                mis.append((k_, a, c)); break
    pol = FillPolicy(b, W, X, THETA_OFF, 1.0, float(w0['te']) + 200.0, st, chk)
    rz = QA.sim_joint_g(W, 'M', 50.0, None, aux, decide={(s, d): pol.dec(s, d) for s in (0, 1) for d in range(NR)},
                        gate={(s, d): pol.gate(s, d) for s in (0, 1) for d in range(NR)}, size=Z, mode='none', cooldown=M.CL.REPOST_MS)
    if rz['fills'] != rk['fills'] or rz['endpos'] != rk['endpos']:
        bad.append('off-parity: the override above every probability changed the fills')
    if mis or st['batches'] == 0:
        bad.append(f'order state vs known_state: {mis[:3]}, batches {st["batches"]}')
    v1 = np.vstack([np.concatenate([X[0][1][i], rng.normal(0, 1, len(V.ORDER_IN)).astype(np.float32)]) for i in range(5)])
    if not np.allclose(b.predict(v1), np.asarray([b.predict(v1[i:i + 1])[0] for i in range(5)]), rtol=0, atol=1e-12):
        bad.append('batched predictions differ from single ones')
    pol2 = FillPolicy(b, W, X, -1.0, 1.0, float(w0['te']) + 200.0, collections.Counter())      # always pull, never enter
    ra = QA.sim_joint_g(W, 'M', 50.0, None, aux, decide={(s, d): pol2.dec(s, d) for s in (0, 1) for d in range(NR)},
                        gate={(s, d): pol2.gate(s, d) for s in (0, 1) for d in range(NR)}, size=Z, mode='none', cooldown=M.CL.REPOST_MS)
    nf = sum(len(x) for x in ra['fills'].values())
    if nf:
        bad.append(f'an always-pull rule left {nf} fills')
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
        sys.exit(f'STOP-EV-PLAY SELFTEST FAILED: {bad}')
    m = mutation_check()
    if not all(m.values()):
        sys.exit(f'STOP-EV-PLAY PLANTED ERROR NOT CAUGHT: {m}')
    print(f'stop-ev-play selftests passed; planted errors caught: {m}', flush=True)
    return m


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        all_selftests()
    elif cmd in ('cal', 'test'):
        play(sys.argv[2], cmd)
    elif cmd == 'report':
        report()
    else:
        sys.exit(__doc__)
