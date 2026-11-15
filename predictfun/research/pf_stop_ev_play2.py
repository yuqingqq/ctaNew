"""STOP-EV-PLAY v2 (2026-11-15, session aea5bfb9; USER "Can we fix the critical issues and let the results be correct first", after the
pipeline REVIEW): the replay of "rest when quiet, pull when the fill model sees activity" (QR), rebuilt. v1 (pf_stop_ev_play.py; receipts
in reviews/stop_ev_play_20261115) is VOID AS RUN: its entry gate was stale between decision steps (a pulled rung re-posted at the back of
the queue 50 ms later), its shifted control fed the previous window's scale and mid into the ORDER inputs, its incumbents decided every
50 ms against QR's 250 ms, the floor of its theta grid was chosen, MD1000's parameters came from the old lane, its tape lacked the
untimed legs the labels now carry, and it had no control matched on how much an arm pulls.

FIXES: (1) GATE: decisions every CAD = 5 grid steps (250 ms); a rung the policy pulled stays empty until the next decision step; at every
  decision step the entry gate of EVERY rung is evaluated fresh (P(fill) of a fresh order there >= r x theta -> closed) and between
  decision steps an empty rung (after a fill, a drift / cross cancel) uses that value (at most 200 ms old). (2) QRS: the 71 market inputs
  of the side's previous window at the same step; the ORDER inputs from this window's own scale and pf_mid. (3) CADENCE: the incumbents
  LDG and MD1000 run native (50 ms: the bar) and at QR's cadence with QR's gate semantics (LDG250 / MD250, printed). (4) THETAS (0.02,
  0.05, 0.10, 0.20, 0.30) x HYST (1.0, 0.6); the calibration day's $ of every combination printed with a grid-edge flag. (5) MD1000's
  (tau_on, r) re-chosen on the corrected lane on the fold's calibration day (pf_stop_play's TAUS x HYST, C16 rule). (6) RV, a matched
  random control: QR's decision structure with coin flips at QR's realized rates in the same window, side and clock (pulls per resting
  decision, closed entry gates per rung period); the mean of RV_DRAWS seeded draws. (7) TAPE: the labels' imputed-untimed tape
  (pf_stop_ev.impute_window: untimed legs at their taker group's earliest exec second + the day's median anchor offset; source flag E2).
  (8) c / share and shares a day beside every $.
MODEL: v3's fill head (pf_stop_ev_model models_v3/<test>_full_p.txt: imputed-untimed labels, every eligible status, reload-checked; its
  file hash must equal the fit's record).
LANE: pf_stop_queue_activation_lane.sim_joint_g (activation-snapshot-queue-v1) for every arm; 6 rungs x 30 sh a side, no cap, re-post at
  once (REPOST_MS 0), unknown queue 0; clocks M 50 (primary) and M 250; whole portfolio at the 60.2 s mark (C15), with / without fills on
  orders < 100 ms old (B1a).
CHOICE (chronological): per fold, QR's (theta, r) and MD1000's (tau_on, r) each maximise the calibration day's whole-portfolio $ without
  young fills at M 50 (ties -> the larger threshold, then the larger r), applied unchanged to the test day.
READINGS DECLARED BEFORE THE RUN (5 fold test days; day rule >= 4 of 5; whole portfolio w/o young fills unless stated):
 R980 VS-KEEP-ALL: QR - KA > 0 pooled AND on >= 4 days at M 50 AND M 250, with AND without young fills -> BEATS-KEEP-ALL, else NOT-SHOWN.
 R981 VS-HARM-ONLY (MD1000 native): QR - MD1000 > 0 pooled AND on >= 4 days at M 50 AND M 250 -> BEATS-HARM-ONLY; < 0 the same way ->
      LOSES-TO-HARM-ONLY; else NOT-SHOWN.
 R982 VS-GUARD (LDG native): the same rule -> BEATS-LDG / LOSES-TO-LDG / NOT-SHOWN.
 R983 INFORMATION: QR - QRS > 0 pooled AND on >= 4 days at M 50 -> INFORMED, else NOT-SHOWN.
 R984 TIMING VS RANDOM: QR - RV > 0 pooled AND on >= 4 days at M 50 AND M 250 -> BEATS-RANDOM, else NOT-SHOWN (RV is a matched control,
      the mean of RV_DRAWS seeded draws per window; no null distribution and no p-value is claimed).
 R985 PROFITABLE: QR > 0 pooled AND on >= 4 days at M 50 AND M 250 -> PROFITABLE, else NOT-PROFITABLE.
 Printed, not decided on: c / share and shares a day per arm, LDG250 / MD250, the WS / E2 split, fills by order age, quote uptime per
 rung, pulls and closed entry gates (QR vs RV realized), the calibration-day grids, and SENSITIVITY at M 50 (KA, LDG, MD1000, QR): LAT =
 placement / cancel latency 58 / 37 ms (the measured medians, pf_clock_rl.py) instead of 40 / 22; QINF = unknown queue infinite, not 0.
CONTROLS (any failure -> no output): off-parity (QR with theta above every probability equals KA fill for fill on every test window); lane
  violations 0; the incremental order state equals pf_stop_ev_accounting.known_state on >= 1,000 sampled calls a test day; the imputed
  tape's leg rows align with the loader's legs and its source flags with its legs. Synthetic: off-parity and the order-state parity on a
  toy lane, an always-pull QR and RV at rates 1 / 1 remove every fill, RV at rates 0 / 0 and a cadence wrapper that never triggers equal
  keep-all, no re-post of a pulled rung before the next decision step (QR, RV, the cadence wrapper) with at least one pull, QRS takes the
  market inputs from the previous window and the order inputs from the current one, batched predictions equal single ones; planted
  errors each caught: the venue frame instead of the known frame, the entry gate ignored, the stale gate (a pulled rung re-posted between
  decision steps), QRS order inputs from the previous window.
LIMITS: consumed days (exploratory, USER); replay rule M (B1a); E2 point anchors and imputed untimed times; the fill head is a local 1 s
  model, the evaluation is the whole portfolio; v3's verdict is read separately (stop_ev_model_v3.txt).
Usage (run_light, PF_SLOT=claude): pf_stop_ev_play2.py selftest | cal <test day> | test <test day> | report; driver run_stop_ev_play2.sh.
"""
import collections
import contextlib
import json
import math
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_stop_ev_play as P0                              # noqa: E402  (v1's order-state helpers, unchanged definitions)
import pf_stop_play as SP                                 # noqa: E402  (the harm-only arms, values)

V = P0.V; E, M, K, AC = V.E, V.M, V.K, V.AC
QA = E.QA
OUTD = f'{K.IO}/stop_ev_play2_20261115'
THETAS = (0.02, 0.05, 0.10, 0.20, 0.30); HYST = (1.0, 0.6); COMBOS = [(t, r) for t in THETAS for r in HYST]
CAD = 5; THETA_OFF = 2.0; Z = 30.0; NR = int(M.CL.NR); PARITY_EVERY = 97; RV_DRAWS = 5
CFGS = (('M', 50.0), ('M', 250.0))
SENS = (('LAT', dict(L_PLACE=58.0, L_CANCEL=37.0), None), ('QINF', {}, 'inf'))
ARMS = ('KA', 'LDG', 'LDG250', 'MD1000', 'MD250', 'QR', 'QRS', 'RV')
MUT_GATE_OFF = MUT_STALE_GATE = MUT_QRS_PREV = 0          # planted errors (selftest only); MUT_FEAT_VENUE lives in pf_stop_ev_play
MUTS = ('MUT_FEAT_VENUE', 'MUT_GATE_OFF', 'MUT_STALE_GATE', 'MUT_QRS_PREV')


# ---------------------------------------------------------------- policies
class FillPolicy2:
    """QR(theta, r) on the lane's own orders. X[s] = (pos, market rows, scale, pf_mid) supplies the 71 market inputs and Xo[s] the scale
    and pf_mid of the ORDER inputs (both this window for QR; QRS takes X from the side's previous window). One batch per decision step:
    every registered live order and a fresh order on every rung (the entry gates)."""

    def __init__(self, booster, W, X, Xo, theta, r, terminal, stat, check=None):
        self.b = booster; self.W = W; self.X = X; self.Xo = Xo; self.theta = theta; self.off = r * theta; self.term = terminal
        self.st = stat; self.check = check; self.reg = {}; self.cache = {}; self.ck = -1; self.gv = {}; self.pulled = {}; self.cnt = {}
        self.n = collections.Counter()                     # per side: decisions on resting orders / pulls, gate periods / closed ones

    def _rows(self, s, k):
        x = self.X.get(s); xo = self.Xo.get(s)
        if x is None or xo is None or k >= len(x[0]) or k >= len(xo[0]) or x[0][k] < 0 or xo[0][k] < 0:
            return None
        o = x if MUT_QRS_PREV else xo
        return x[1][x[0][k]], o[2][o[0][k]], o[3][o[0][k]]  # market row, scale, pf_mid

    def _vec(self, s, d, k, t, cur):
        row = self._rows(s, k)
        if row is None:
            return None
        f = P0.order_state(self.W[s], cur, k, t, s, d, self.term) if cur is not None else P0.fresh_state(self.W[s], k, t, s, d, self.term)
        if f is None:
            return None
        if cur is not None and self.check is not None:
            self.check(self.W, cur, (s, d), t, f)
        return np.concatenate([row[0], np.asarray(P0.order_vector(f, row[1], row[2]), np.float32)])

    def _batch(self, k, t, fresh_only=False):
        self.ck = k; self.cache = {}; keys, vecs = [], []
        for s in (0, 1):
            for d in range(NR):
                cur = None if fresh_only else self.reg.get((s, d))
                if cur is not None and cur['tc'] > t and not cur['ref'] and cur['ta'] <= t:
                    v = self._vec(s, d, k, t, cur)
                    if v is not None:
                        keys.append(((s, d), id(cur))); vecs.append(v)
                v = self._vec(s, d, k, t, None)
                if v is not None:
                    keys.append(((s, d), None)); vecs.append(v)
        if vecs:
            p = self.b.predict(np.vstack(vecs)); self.st['batch_rows'] += len(vecs); self.st['batches'] += 1
            for key, pv in zip(keys, p):
                self.cache[key] = float(pv)
        for s in (0, 1):                                  # the fresh entry gate of every rung at this decision step
            for d in range(NR):
                p = self.cache.get(((s, d), None), math.nan)
                self.gv[(s, d)] = bool(p == p and p >= self.off)

    def prob(self, s, d, k, t, cur):
        if k != self.ck:
            self._batch(k, t)
        key = ((s, d), id(cur))
        if key not in self.cache:
            v = self._vec(s, d, k, t, cur)
            self.cache[key] = float(self.b.predict(v[None, :])[0]) if v is not None else math.nan; self.st['single_rows'] += 1
        return self.cache[key]

    def dec(self, s, d):
        def f(k, t, cur):
            self.reg[(s, d)] = cur
            if k % CAD:
                return False
            p = self.prob(s, d, k, t, cur); self.n[f'dec_{s}'] += 1
            x = bool(p == p and p > self.theta)
            if x:
                self.pulled[(s, d)] = k; self.n[f'pull_{s}'] += 1
            return x
        return f

    def gate(self, s, d):
        tg = self.W[0]['tg']

        def g(k):
            if MUT_GATE_OFF:
                return False
            k0 = k - k % CAD
            if self.ck != k0:
                self._batch(k0, float(tg[k0]), fresh_only=(k != k0))
            if self.pulled.get((s, d)) == k0 and not MUT_STALE_GATE:
                return True                               # pulled at this period's decision step: empty until the next one
            x = self.gv.get((s, d), False)
            if self.cnt.get((s, d)) != k0:
                self.cnt[(s, d)] = k0; self.n[f'gate_{s}'] += 1; self.n[f'block_{s}'] += int(x)
            return x
        return g

    def fns(self):
        return ({(s, d): self.dec(s, d) for s in (0, 1) for d in range(NR)}, {(s, d): self.gate(s, d) for s in (0, 1) for d in range(NR)})

    def rates(self):
        pp = {s: (self.n[f'pull_{s}'] / self.n[f'dec_{s}']) if self.n[f'dec_{s}'] else 0.0 for s in (0, 1)}
        pb = {s: (self.n[f'block_{s}'] / self.n[f'gate_{s}']) if self.n[f'gate_{s}'] else 0.0 for s in (0, 1)}
        return pp, pb


class RandomPolicy:
    """RV: QR's decision structure with coin flips - at each decision step a resting order is pulled with probability p_pull[s]; each
    rung's entry gate is closed for a period with probability p_block[s]; a pulled rung stays empty until the next decision step."""

    def __init__(self, p_pull, p_block, seed):
        self.pp = p_pull; self.pb = p_block; self.rng = np.random.default_rng(seed); self.pulled = {}; self.gv = {}
        self.n = collections.Counter()

    def dec(self, s, d):
        def f(k, t, cur):
            if k % CAD:
                return False
            self.n[f'dec_{s}'] += 1; x = bool(self.rng.random() < self.pp[s])
            if x:
                self.pulled[(s, d)] = k; self.n[f'pull_{s}'] += 1
            return x
        return f

    def gate(self, s, d):
        def g(k):
            k0 = k - k % CAD
            if self.pulled.get((s, d)) == k0 and not MUT_STALE_GATE:
                return True
            if self.gv.get((s, d), (None, False))[0] != k0:
                self.gv[(s, d)] = (k0, bool(self.rng.random() < self.pb[s])); self.n[f'gate_{s}'] += 1; self.n[f'block_{s}'] += int(self.gv[(s, d)][1])
            return self.gv[(s, d)][1]
        return g

    def fns(self):
        return ({(s, d): self.dec(s, d) for s in (0, 1) for d in range(NR)}, {(s, d): self.gate(s, d) for s in (0, 1) for d in range(NR)})


def cadence(dec_n, gate_n):
    """A 50 ms incumbent at QR's cadence (LDG250 / MD250): it decides at decision steps only, a rung it pulled stays empty until the next
    decision step, otherwise its entry gate as sampled at the period's decision step."""
    pulled = {}

    def mk_dec(key, f):
        def dd(k, t, cur):
            if k % CAD:
                return False
            x = bool(f(k, t, cur))
            if x:
                pulled[key] = k
            return x
        return dd

    def mk_gate(key, g):
        def gg(k):
            k0 = k - k % CAD
            if pulled.get(key) == k0 and not MUT_STALE_GATE:
                return True
            return bool(g(k0)) if g is not None else False
        return gg
    return {key: mk_dec(key, f) for key, f in dec_n.items()}, {key: mk_gate(key, gate_n.get(key)) for key in dec_n}


def md_fns(Pm, W, tau, r):
    """MD1000 native (pf_stop_play): cancel iff P^H_(d+1)(k) > tau_on; rung d closed while P^H_(d+1)(k) >= r x tau_on."""
    return ({(s, d): SP.model_dec(Pm[s], W[s]['best'], s, tau) for s in (0, 1) for d in range(NR)},
            {(s, d): SP.model_gate(Pm[s], d, tau * r) for s in (0, 1) for d in range(NR)})


def ldg_fns(sig):
    """The guard as decide / gate functions (LDG250's input): pull, and stay out, while the step's guard signal is on."""
    return ({(s, d): (lambda s_: (lambda k, t, cur: bool(sig[s_][k])))(s) for s in (0, 1) for d in range(NR)},
            {(s, d): (lambda s_: (lambda k: bool(sig[s_][k])))(s) for s in (0, 1) for d in range(NR)})


@contextlib.contextmanager
def patched(lat, qunk):
    """SENSITIVITY: the lane's placement / cancel latency and the unknown-queue assumption for the runs inside; restored after."""
    K_ = M.CL.K; old = {k_: getattr(K_, k_) for k_ in lat}; oq = M.QUNK
    try:
        for k_, v_ in lat.items():
            setattr(K_, k_, v_)
        if qunk is not None:
            M.QUNK = qunk
        yield
    finally:
        for k_, v_ in old.items():
            setattr(K_, k_, v_)
        M.QUNK = oq


# ---------------------------------------------------------------- the tape
def tape(D, W, LT, WSg, GRP, RW, off):
    """The window on the labels' imputed-untimed tape (pf_stop_ev.impute_window) and its per-leg WS flags (C12) in the lane's leg order;
    the leg rows must align with the loader's window-side legs (stage 1's check)."""
    m = int(W[0]['m']); LR, SR, GR = {}, {}, {}
    for s in (0, 1):
        w = W[s]
        if (m, s) in D['LI']:
            l = LT.iloc[D['LI'][(m, s)]]
            o_ = np.lexsort(((-l.ptick.values if s == 0 else l.ptick.values), l.anc.values)); lrow = l.index.values[o_]
        else:
            lrow = np.zeros(0, 'int64')
        if len(lrow) != len(w['ptl']) or (len(lrow) and not (np.array_equal(LT.loc[lrow, 'ptick'].values, np.asarray(w['ptl']))
                                                              and np.array_equal(LT.loc[lrow, 'anc'].values, np.asarray(w['anc'])))):
            raise ValueError(f'window {m} side {s}: leg rows not aligned with the window-side legs')
        LR[s] = lrow; SR[s] = WSg.reindex(lrow).values.astype(bool) if len(lrow) else np.zeros(0, bool)
        GR[s] = GRP.reindex(lrow).values.astype('int64') if len(lrow) else np.zeros(0, 'int64')
    W2, LR2, SR2, GR2, IM2 = E.impute_window(W, LR, SR, GR, RW.get(m, {}).get('imp', []), off)
    for s in (0, 1):
        if not (len(SR2[s]) == len(W2[s]['ptl']) == len(W2[s]['shl']) == len(W2[s]['anc']) == len(IM2[s])):
            raise ValueError(f'window {m} side {s}: imputed tape and its flags differ in length')
    return W2, SR2, int(sum(int(IM2[s].sum()) for s in (0, 1)))


# ---------------------------------------------------------------- one day
def play(test, phase):
    import lightgbm as lgb
    all_selftests()
    if test not in M.TESTS:
        sys.exit(f'REFUSED: {test} is not a fold test day')
    if M.QUNK != 'zero':
        sys.exit('REFUSED: unknown queue must be zero (the primary assumption)')
    day = M.FOLDS[test][1] if phase == 'cal' else test
    T0 = time.time(); hp = f'{V.OUTD}/models_v3/{test}_full_p.txt'; fj = json.load(open(f'{V.OUTD}/fit3_{test}.json'))
    if not fj.get('reload_pass') or V.sha_file(hp) != fj['artifacts']['full_p']['sha256']:
        sys.exit('REFUSED: the v3 fill head is refused or changed since its fit')
    bst = lgb.Booster(model_file=hp)
    if bst.num_feature() != len(V.FULL_IN):
        sys.exit('REFUSED: the fill head has a different input count')
    MK = P0.market_arrays(day); D = K.load_day(day); cut = M.boundary_cut(day)
    tkf = pd.read_parquet(f'{K.IO}/step1/{day}/legs_tk.parquet', columns=['tk']).tk.values
    LT = D['LT']; ix = LT.index.values; grp = np.where(tkf[ix] >= 0, tkf[ix], -1 - ix)
    WSg = pd.Series(M.TS.ws_groups(LT.how.values, grp), index=LT.index); GRP = pd.Series(grp, index=LT.index)
    RW, ust = E.raw_windows(day, D)
    st = collections.Counter(); val = {}; live = {}; flow = {}
    PA = SP.pred_arrays(f'{M.OUTD}/pred_fold{test}_{day}_v6.parquet', M.HP)
    if phase == 'test':
        cj = json.load(open(f'{OUTD}/cal_{test}.json')); th, rr = cj['choice']; mdc = cj['md_choice']
    pcheck = dict(n=0, bad=0)

    def check(W, cur, key, t, f):
        st['calls'] += 1
        if st['calls'] % PARITY_EVERY:
            return
        g = AC.known_state(W, cur, key + (0,), t, Z, terminal_ms=float(W[0]['te']) + 200.0)
        pcheck['n'] += 1
        for k_ in P0.ORDER_KEYS:
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

    def addn(cfg, arm, n, w=1.0):
        a = flow.setdefault(cfg, {}).setdefault(arm, {})
        for k_, x in n.items():
            a[k_] = a.get(k_, 0.0) + x * w

    def run(W, q, shf, dec=None, gate=None, aux=None):
        n = W[0]['n']; aux = aux or {0: [False] * n, 1: [False] * n}
        r = QA.sim_joint_g(W, q, shf, None, aux, decide=dec, gate=gate, idle_only=0, size=Z, mode='none', cooldown=M.CL.REPOST_MS)
        st['runs'] += 1; st['viol'] += int(sum(r['stat']['viol'].values()))
        return r

    prev = None
    for W0 in M.P1.window_pairs(D):
        if W0[0]['ws'] < cut:
            st['boundary_windows'] += 1; continue
        m = int(W0[0]['m']); st['windows'] += 1
        W, src, nimp = tape(D, W0, LT, WSg, GRP, RW, ust['imp_off_ms']); st['imputed_legs'] += nimp
        tf, mff = W[0]['tf_mff']; ws, we = float(W[0]['tg'][0]), float(W[0]['te'])
        X = {s: MK.get((m, s)) for s in (0, 1)}; Pm = {s: PA[0].get((m, s)) for s in (0, 1)}
        sig = {s: (np.asarray(W[s]['vkl'], float) <= M.GUARD) for s in (0, 1)}
        av = lambda r: SP.arm_value2(r, tf, mff, src)
        lsh = lambda r: SP.live_share(r, ws, we)
        if phase == 'cal':
            for (t_, r_) in COMBOS:
                add('M50', f'QR|{t_}|{r_}', av(run(W, 'M', 50.0, *FillPolicy2(bst, W, X, X, t_, r_, we + 200.0, st).fns())))
            for (tau, r_) in SP.COMBOS:
                add('M50', f'MD|{tau}|{r_}', av(run(W, 'M', 50.0, *md_fns(Pm, W, tau, r_))))
        else:
            for q, shf in CFGS:
                cfg = f'{q}{int(shf)}'
                rk = run(W, q, shf); add(cfg, 'KA', av(rk), lsh(rk))
                rl = run(W, q, shf, aux={s: sig[s].tolist() for s in (0, 1)}); add(cfg, 'LDG', av(rl), lsh(rl))
                r_ = run(W, q, shf, *cadence(*ldg_fns(sig))); add(cfg, 'LDG250', av(r_), lsh(r_))
                r_ = run(W, q, shf, *md_fns(Pm, W, *mdc)); add(cfg, 'MD1000', av(r_), lsh(r_))
                r_ = run(W, q, shf, *cadence(*md_fns(Pm, W, *mdc))); add(cfg, 'MD250', av(r_), lsh(r_))
                pol = FillPolicy2(bst, W, X, X, th, rr, we + 200.0, st, check if cfg == 'M50' else None)
                rq = run(W, q, shf, *pol.fns()); add(cfg, 'QR', av(rq), lsh(rq)); addn(cfg, 'QR', pol.n)
                pp, pb = pol.rates(); acc, lacc = [], []
                for i in range(RV_DRAWS):
                    rv = RandomPolicy(pp, pb, zlib.crc32(f'{day}|{m}|{cfg}|{i}'.encode()))
                    r_ = run(W, q, shf, *rv.fns()); acc.append(av(r_)); lacc.append(lsh(r_)); addn(cfg, 'RV', rv.n, 1.0 / RV_DRAWS)
                add(cfg, 'RV', {k_: float(np.mean([a_[k_] for a_ in acc])) for k_ in acc[0]}, [float(x) for x in np.mean(lacc, axis=0)])
                if prev is not None:
                    r_ = run(W, q, shf, *FillPolicy2(bst, W, prev, X, th, rr, we + 200.0, st).fns()); add(cfg, 'QRS', av(r_), lsh(r_))
                    st[f'qrs_windows_{cfg}'] += 1
                else:
                    add(cfg, 'QRS', av(rq), lsh(rq)); st[f'qrs_first_window_{cfg}'] += 1
                if cfg == 'M50':
                    rz = run(W, q, shf, *FillPolicy2(bst, W, X, X, THETA_OFF, 1.0, we + 200.0, st).fns()); st['offpar_windows'] += 1
                    st['offpar_differ'] += int(rz['fills'] != rk['fills'] or rz['endpos'] != rk['endpos'])
                    for nm, lat, qk in SENS:
                        with patched(lat, qk):
                            c2 = f'M50{nm}'
                            add(c2, 'KA', av(run(W, q, shf)))
                            add(c2, 'LDG', av(run(W, q, shf, aux={s: sig[s].tolist() for s in (0, 1)})))
                            add(c2, 'MD1000', av(run(W, q, shf, *md_fns(Pm, W, *mdc))))
                            add(c2, 'QR', av(run(W, q, shf, *FillPolicy2(bst, W, X, X, th, rr, we + 200.0, st).fns())))
        prev = X
        if st['windows'] % 40 == 0:
            E.mem_guard(f'stop-ev play2 {phase} {day}')
    st['parity_checked'] = pcheck['n']; st['parity_bad'] = pcheck['bad']
    fails = []
    if st['viol'] or st['offpar_differ']:
        fails.append(f"lane violations {st['viol']}, off-parity differences {st['offpar_differ']}")
    if phase == 'test' and (pcheck['n'] < 1000 or pcheck['bad']):
        fails.append(f'order-state parity with known_state: {pcheck}')
    os.makedirs(OUTD, exist_ok=True)
    res = dict(test=test, day=day, phase=phase, stat=dict(st), value=val, live=live, flow=flow, untimed=ust, seconds=round(time.time() - T0, 1),
               fill_head=dict(path=hp, sha256=fj['artifacts']['full_p']['sha256']), manifest=manifest())
    if fails:
        json.dump(res, open(f'{OUTD}/refused_{phase}_{test}.json', 'w'), indent=1, default=float)
        sys.exit(f'REFUSED (no output): {fails}')
    if phase == 'cal':
        best = max(COMBOS, key=lambda c: (val['M50'][f'QR|{c[0]}|{c[1]}']['pnl_old'], c[0], c[1]))
        mdb = max(SP.COMBOS, key=lambda c: (val['M50'][f'MD|{c[0]}|{c[1]}']['pnl_old'], c[0], c[1]))
        res.update(choice=list(best), md_choice=list(mdb), edge=dict(qr=best[0] in (THETAS[0], THETAS[-1]), md=mdb[0] in (SP.TAUS[0], SP.TAUS[-1])))
        json.dump(res, open(f'{OUTD}/cal_{test}.json', 'w'), indent=1, default=float)
        print(f'cal {test} (day {day}): QR {best}, MD1000 {mdb}; M50 w/o young ' + ', '.join(f'{c} {val["M50"][f"QR|{c[0]}|{c[1]}"]["pnl_old"]:+.0f}'
                                                                                         for c in COMBOS) + f'; {res["seconds"]} s', flush=True)
    else:
        res.update(choice=[th, rr], md_choice=mdc)
        json.dump(res, open(f'{OUTD}/test_{test}.json', 'w'), indent=1, default=float)
        print(f'test {test}: ' + '; '.join(f'{cfg} ' + ', '.join(f'{a} {x["pnl_old"]:+.0f}' for a, x in val[cfg].items()) for cfg in val)
              + f'; parity {pcheck}; off-parity {st["offpar_differ"]} of {st["offpar_windows"]}; {res["seconds"]} s', flush=True)


def manifest():
    import hashlib
    return {f: hashlib.sha256(open(f'{HERE}/{f}', 'rb').read()).hexdigest()[:16]
            for f in ('pf_stop_ev_play2.py', 'pf_stop_ev_play.py', 'pf_stop_ev_model.py', 'pf_stop_ev.py', 'pf_stop_play.py',
                      'pf_stop_queue_activation_lane.py', 'pf_stop_ev_accounting.py', 'pf_queue_activation.py')}


# ---------------------------------------------------------------- report
def report():
    J = {t: json.load(open(f'{OUTD}/test_{t}.json')) for t in M.TESTS if os.path.exists(f'{OUTD}/test_{t}.json')}
    if len(J) < len(M.TESTS):
        sys.exit(f'REFUSED: test days missing {[t for t in M.TESTS if t not in J]}')
    if len({json.dumps(x['manifest'], sort_keys=True) for x in J.values()}) > 1:
        sys.exit('REFUSED: days run from different sources')
    C = {t: json.load(open(f'{OUTD}/cal_{t}.json')) for t in M.TESTS}
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()); nd = len(J)
    S = lambda cfg, a, k_: float(sum(J[t]['value'][cfg][a][k_] for t in J))
    mean = lambda cfg, a, k_: S(cfg, a, k_) / nd
    per = lambda cfg, a, k_: np.asarray([J[t]['value'][cfg][a][k_] for t in M.TESTS])

    def cps(cfg, a):
        sh = S(cfg, a, 'shares') - S(cfg, a, 'shares_young')
        return (100.0 * S(cfg, a, 'pnl_old') / sh if sh > 0 else math.nan), sh / nd
    T = [f'STOP-EV-PLAY v2 - "rest when quiet, pull when the fill model sees activity" (v3 fill head) in the corrected queue lane on the '
         f'imputed-untimed tape; 5 chronological test days; as of {asof}', '']
    T.append('CHOICES on each fold\'s calibration day (M 50 w/o young; GRID EDGE = the choice sits at an end of its grid): QR (theta, r) | '
             'MD1000 (tau_on, r) | QR $ of every (theta / r)')
    for t in M.TESTS:
        g = C[t]['value']['M50']
        T.append(f"  {t}: {C[t]['choice']}{' GRID EDGE' if C[t]['edge']['qr'] else ''} | {C[t]['md_choice']}{' GRID EDGE' if C[t]['edge']['md'] else ''} | "
                 + ', '.join(f"{c[0]}/{c[1]} {g[f'QR|{c[0]}|{c[1]}']['pnl_old']:+.0f}" for c in COMBOS))
    T.append('TAPE per test day: windows (boundary windows excluded) | imputed untimed legs | order-state parity checks (mismatches) | '
             'off-parity windows (differences) | QRS windows (first window = QR)')
    for t in M.TESTS:
        x = J[t]['stat']
        T.append(f"  {t}: {x.get('windows', 0)} ({x.get('boundary_windows', 0)}) | {x.get('imputed_legs', 0)} | {x.get('parity_checked', 0)} "
                 f"({x.get('parity_bad', 0)}) | {x.get('offpar_windows', 0)} ({x.get('offpar_differ', 0)}) | {x.get('qrs_windows_M50', 0)} ({x.get('qrs_first_window_M50', 0)})")
    T.append('WHOLE PORTFOLIO $ a day (mean of 5 days): arm | M50 all / w/o young | M250 all / w/o young | M50 w/o young: c / share on shares '
             'a day | WS / E2 | quote uptime per rung')
    for a in ARMS:
        c_, sh_ = cps('M50', a)
        up = np.mean([np.asarray(J[t]['live']['M50'][a]) / max(J[t]['stat']['windows'], 1) for t in J], axis=0)
        T.append(f"  {a:7s} {mean('M50', a, 'pnl'):+8.1f} / {mean('M50', a, 'pnl_old'):+8.1f} | {mean('M250', a, 'pnl'):+8.1f} / {mean('M250', a, 'pnl_old'):+8.1f} | "
                 f"{c_:+.3f} c on {sh_:,.0f} sh | {mean('M50', a, 'ws_old'):+.1f} / {mean('M50', a, 'e2_old'):+.1f} | " + ' '.join(f'{u:.2f}' for u in up))
    for cfg in ('M50', 'M250'):
        T.append(f'per day {cfg} w/o young (QR | KA | LDG | MD1000 | QRS | RV): ' + '; '.join(
            f"{t} " + ' | '.join(f"{J[t]['value'][cfg][a]['pnl_old']:+.0f}" for a in ('QR', 'KA', 'LDG', 'MD1000', 'QRS', 'RV')) for t in M.TESTS))
    T.append('pulls a day (M50): ' + ', '.join(f"{a} {mean('M50', a, 'pulls'):.0f}" for a in ARMS) + '; closed-gate entries a day: '
             + ', '.join(f"{a} {mean('M50', a, 'gated'):.0f}" for a in ARMS))
    fq = lambda cfg, a, k_: float(sum(J[t]['flow'].get(cfg, {}).get(a, {}).get(k_, 0.0) for t in J)) / nd
    for cfg in ('M50', 'M250'):
        T.append(f'QR vs RV realized ({cfg}, a day): decisions on resting orders / pulls | gate periods / closed: ' + '; '.join(
            f"{a} " + ' / '.join(f"{fq(cfg, a, f'dec_{s}'):.0f}/{fq(cfg, a, f'pull_{s}'):.0f} | {fq(cfg, a, f'gate_{s}'):.0f}/{fq(cfg, a, f'block_{s}'):.0f}"
                                 for s in (0, 1)) for a in ('QR', 'RV')))
    T.append('by order age (M50, all fills, $ a day): ' + '; '.join(f"{a} <100 {mean('M50', a, 'age_lt100'):+.0f} / 100-500 {mean('M50', a, 'age_100_500'):+.0f} / "
                                                                 f">=500 {mean('M50', a, 'age_ge500'):+.0f}" for a in ARMS))
    T.append('SENSITIVITY (M50 w/o young, $ a day | c / share): arm | base | LAT 58 / 37 ms | QINF')
    for a in ('KA', 'LDG', 'MD1000', 'QR'):
        T.append(f'  {a:7s} ' + ' | '.join(f"{mean(c, a, 'pnl_old'):+8.1f} {cps(c, a)[0]:+.3f} c" for c in ('M50', 'M50LAT', 'M50QINF')))

    def rule(b, cfgs, keys):
        d = [per(c, 'QR', k_) - (per(c, b, k_) if b else 0.0) for c in cfgs for k_ in keys]
        return (all(x.sum() > 0 and int((x > 0).sum()) >= 4 for x in d), all(x.sum() < 0 and int((x < 0).sum()) >= 4 for x in d))
    r980 = 'BEATS-KEEP-ALL' if rule('KA', ('M50', 'M250'), ('pnl', 'pnl_old'))[0] else 'NOT-SHOWN'
    p_, n_ = rule('MD1000', ('M50', 'M250'), ('pnl_old',)); r981 = 'BEATS-HARM-ONLY' if p_ else 'LOSES-TO-HARM-ONLY' if n_ else 'NOT-SHOWN'
    p_, n_ = rule('LDG', ('M50', 'M250'), ('pnl_old',)); r982 = 'BEATS-LDG' if p_ else 'LOSES-TO-LDG' if n_ else 'NOT-SHOWN'
    r983 = 'INFORMED' if rule('QRS', ('M50',), ('pnl_old',))[0] else 'NOT-SHOWN'
    r984 = 'BEATS-RANDOM' if rule('RV', ('M50', 'M250'), ('pnl_old',))[0] else 'NOT-SHOWN'
    r985 = 'PROFITABLE' if rule(None, ('M50', 'M250'), ('pnl_old',))[0] else 'NOT-PROFITABLE'
    T += ['', f'R980 {r980}; R981 {r981}; R982 {r982}; R983 {r983}; R984 {r984}; R985 {r985} (development only: consumed days, replay rule M, '
              f'E2 point anchors and imputed untimed times; RV = a matched control, not a null)']
    os.makedirs(OUTD, exist_ok=True)
    open(f'{OUTD}/stop_ev_play2.txt', 'w').write('\n'.join(T) + '\n')
    json.dump(dict(asof=asof, readings=dict(R980=r980, R981=r981, R982=r982, R983=r983, R984=r984, R985=r985), days=J, cal=C),
              open(f'{OUTD}/stop_ev_play2.json', 'w'), indent=1, default=float)
    print('\n'.join(T), flush=True)


# ---------------------------------------------------------------- self-tests
class StubBooster:
    """A fill head for the synthetic cases: P = hi for a resting order (age > 0), lo for a fresh one (age 0)."""

    def __init__(self, hi, lo):
        self.hi = hi; self.lo = lo; self.ia = len(V.F71) + V.ORDER_IN.index('o_age')

    def predict(self, X):
        return np.where(np.asarray(X, float)[:, self.ia] > 0, self.hi, self.lo)


def repost_after_pull(r, tg):
    """(policy pulls, re-posts of a pulled rung before the next decision step) from the lane log."""
    kof = {float(t): i for i, t in enumerate(tg)}; last = {}; n_pull = n_bad = 0
    for (t, s, d, what, pt) in r['log']:
        if what == 'cancel:trig':
            last[(s, d)] = kof[float(t)]; n_pull += 1
        elif what == 'place' and (s, d) in last:
            n_bad += int(kof[float(t)] < last.pop((s, d)) + CAD)
    return n_pull, n_bad


def selftest():
    import lightgbm as lgb
    bad = []
    bid = [(710.0, 50, 30.0, 2), (910.0, 50, 25.0, 4), (1210.0, 47, 40.0, 6), (1260.0, 49, 12.0, 8)]
    ask = [(300.0, 52, 15.0, 3), (800.0, 53, 40.0, 5), (1500.0, 52, 18.0, 7)]
    (w0, _, _, _), (w1, _, _, _) = E.toy_w(0, bid, drop=(500.0, 0, 20.0)), E.toy_w(1, ask)
    W = {0: w0, 1: w1}; n = w0['n']; tg = w0['tg']; aux = {0: [False] * n, 1: [False] * n}; term = float(w0['te']) + 200.0

    def lane(dec=None, gate=None):
        return QA.sim_joint_g(W, 'M', 50.0, None, aux, decide=dec, gate=gate, size=Z, mode='none', cooldown=M.CL.REPOST_MS)
    rk = lane(); nfk = sum(len(x) for x in rk['fills'].values())
    rng = np.random.default_rng(5); F = len(V.FULL_IN)
    Xb = rng.normal(0, 1, (600, F)).astype(np.float32); yb = (rng.random(600) < 0.3).astype(float)
    b = lgb.train({**V.PARAMS, 'min_data_in_leaf': 20}, lgb.Dataset(Xb, yb), 10)
    X = {s: (np.arange(n, dtype='int64'), rng.normal(0, 1, (n, len(V.F71))).astype(np.float32), np.full(n, 200.0), np.full(n, 51.0)) for s in (0, 1)}
    st = collections.Counter(); mis = []

    def chk(W_, cur, key, t, f):
        g = AC.known_state(W_, cur, key + (0,), t, Z, terminal_ms=float(W_[0]['te']) + 200.0)
        for k_ in P0.ORDER_KEYS:
            a, c = f[k_], g[k_]
            if not ((isinstance(a, float) and isinstance(c, float) and math.isnan(a) and math.isnan(c)) or abs(float(a) - float(c)) <= 1e-9):
                mis.append((k_, a, c)); break
    # (1) off-parity: QR with theta above every probability equals keep-all; the incremental order state equals known_state
    rz = lane(*FillPolicy2(b, W, X, X, THETA_OFF, 1.0, term, st, chk).fns())
    if nfk == 0 or rz['fills'] != rk['fills'] or rz['endpos'] != rk['endpos']:
        bad.append(f'off-parity: keep-all fills {nfk}; the override above every probability changed the fills')
    if mis or st['batches'] == 0:
        bad.append(f'order state vs known_state: {mis[:3]}, batches {st["batches"]}')
    # (2) an always-pull QR (every gate closed) and RV at rates 1 / 1 remove every fill; RV at 0 / 0 equals keep-all
    ra = lane(*FillPolicy2(b, W, X, X, -1.0, 1.0, term, collections.Counter()).fns())
    r1 = lane(*RandomPolicy({0: 1.0, 1: 1.0}, {0: 1.0, 1: 1.0}, 1).fns()); r0 = lane(*RandomPolicy({0: 0.0, 1: 0.0}, {0: 0.0, 1: 0.0}, 2).fns())
    nfa = sum(len(x) for x in ra['fills'].values()); nf1 = sum(len(x) for x in r1['fills'].values())
    if nfa or nf1 or r0['fills'] != rk['fills']:
        bad.append(f'always-pull QR left {nfa} fills, RV 1 / 1 left {nf1}, RV 0 / 0 equals keep-all {r0["fills"] == rk["fills"]}')
    # (3) no re-post of a pulled rung before the next decision step (resting orders P 0.9 > theta; fresh orders P 0 -> the gate is open)
    rq = lane(*FillPolicy2(StubBooster(0.9, 0.0), W, X, X, 0.5, 1.0, term, collections.Counter()).fns())
    rv = lane(*RandomPolicy({0: 1.0, 1: 1.0}, {0: 0.0, 1: 0.0}, 3).fns())
    always = {(s, d): (lambda k, t, cur: True) for s in (0, 1) for d in range(NR)}; never_g = {(s, d): (lambda k: False) for s in (0, 1) for d in range(NR)}
    rc = lane(*cadence(always, never_g))
    for nm, r_ in (('QR', rq), ('RV', rv), ('cadence', rc)):
        n_pull, n_bad = repost_after_pull(r_, tg)
        if n_pull == 0 or n_bad:
            bad.append(f'{nm}: {n_pull} pulls, {n_bad} re-posts of a pulled rung before the next decision step')
    never = {(s, d): (lambda k, t, cur: False) for s in (0, 1) for d in range(NR)}
    if lane(*cadence(never, never_g))['fills'] != rk['fills']:
        bad.append('a cadence wrapper that never triggers differs from keep-all')
    # (4) QRS: market inputs from the previous window, ORDER inputs (scale, pf_mid) from the current one
    Xp = {s: (np.arange(n, dtype='int64'), rng.normal(0, 1, (n, len(V.F71))).astype(np.float32), np.full(n, 100.0), np.full(n, 40.0)) for s in (0, 1)}
    pol = FillPolicy2(StubBooster(0.9, 0.0), W, Xp, X, 0.5, 1.0, term, collections.Counter())
    v4 = pol._vec(0, 0, 3, float(tg[3]), None); pt4 = float(w0['best'][3]); ie = len(V.F71) + V.ORDER_IN.index('o_edge')
    if v4 is None or not (np.array_equal(v4[:len(V.F71)], Xp[0][1][3]) and abs(float(v4[ie]) - (51.0 - pt4)) < 1e-6):
        bad.append(f'QRS inputs: edge {None if v4 is None else float(v4[ie])} vs {51.0 - pt4}')
    # (5) batched predictions equal single ones
    v1 = np.vstack([np.concatenate([X[0][1][i], rng.normal(0, 1, len(V.ORDER_IN)).astype(np.float32)]) for i in range(5)])
    if not np.allclose(b.predict(v1), np.asarray([b.predict(v1[i:i + 1])[0] for i in range(5)]), rtol=0, atol=1e-12):
        bad.append('batched predictions differ from single ones')
    return bad


def mutation_check():
    out = {}
    for nm in MUTS:
        mod = P0 if nm == 'MUT_FEAT_VENUE' else sys.modules[__name__]
        setattr(mod, nm, 1)
        try:
            out[nm] = bool(selftest())
        finally:
            setattr(mod, nm, 0)
    return out


def all_selftests():
    bad = selftest()
    if bad:
        sys.exit(f'STOP-EV-PLAY2 SELFTEST FAILED: {bad}')
    m = mutation_check()
    if not all(m.values()):
        sys.exit(f'STOP-EV-PLAY2 PLANTED ERROR NOT CAUGHT: {m}')
    print(f'stop-ev-play2 selftests passed; planted errors caught: {m}', flush=True)
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
