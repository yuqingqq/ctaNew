"""ORDER-HARM v3 (2026-11-14, session ce68ebe0): the value of KEEPING NOW WITH THE OPTION TO CANCEL LATER, per resting order, at the
policy's own decision cadence, fit CHRONOLOGICALLY. Supersedes v2's target for the keep / pull decision (v2 stays as provenance).

REPAIR: shared client-state reference/fork engine, 50 ms decisions, and a new
client50 output namespace. Earlier 250 ms artifacts are incompatible.

USER review of the ORDER-HARM POLICY design (its run stopped, outputs deleted unread):
 "4. The target still differs from the intended action. Labels assume keeping without reconsideration for two seconds; the policy can
     reconsider every 250 ms. We need to estimate 'keep now, retain the option to cancel later', including queue value."
 "3. ... Thresholds use predictions across all nine consumed dates, and leave-one-day-out training includes later dates when testing earlier
     ones. Useful for development, but improvement needs chronological training/calibration followed by untouched days."
 "1. ... The new policy generally reconsiders every 250 ms, versus 50 ms for FB25 ... Add a baseline using exactly the same cadence."

THE LANE (labels): each rung of the guard ladder alone (shared client-state lane, 30 sh, M 50), KEEP-ALL with the guard as the ENTRY line only: a resting
order is never pulled by the guard; an order still in flight when the guard comes on is cancelled; nothing new is placed while the guard is on;
re-entry at the back 250 ms after it clears (CL.reference_lane: the policy engine, active side and rung offset explicit). v2's keep branch was the never-pull
lane, which also placed replacement orders while the guard was on - not what keeping an order does.
DECISIONS (the policy's cadence, also used by every same-cadence arm of the replay): in each guard episode (a maximal run of guard-on steps),
for the rung's order resting when it starts (client-known: activation reached, no cancel sent, no full fill known, i.e. fill + 100 ms), at the
onset step and every 50 ms step after it, while the guard stays on and the order rests. A step without inputs (masked start) is a forced
pull (the policy fails closed).
VALUES (c per share of the 30-sh order; each fill valued sg (mid(anchor + 1 s) - price); fills inside the 22 ms cancel latency on neither
branch): for an episode clearing at t_clear, T_e = t_clear + H_TAIL (2 s), the same terminal for every branch. Episodes reaching the lane's end
before T_e are CENSORED (status).
 R_j  = the keep-all lane's fills of the rung in (t_j + 22, t_{j+1} + 22] (last decision: to T_e) - the kept order's fills in the episode and,
        after it, the kept order (with its queue place) or its re-entered replacement, in the tail;
 P_e  = the pull branch's fills in (t_j + 22, T_e]: cancel at t_j, stay out, re-enter after the clear (run_fork on the shared client-state engine). Pulled at
        ANY decision of the episode, the branch is empty until the same re-entry, so P_e is one number per episode (asserted on a sample);
 A_j  = Q(keep now, decide again in 50 ms) - Q(pull now) = R_j + E[max(0, A_{j+1})]  (last decision: R_j - P_e; before a forced
        pull: R_j). The option to cancel later is the max(0, .) term; queue value enters through R's tail (the kept order's place vs the
        re-entered order behind the queue).
FIT (POLICY ITERATION on realized continuation returns, LightGBM, PULL-DEPTH's pinned parameters, regression clipped +-10 c/sh; review
2026-11-14 10:2xZ): one-step fitted-Q backups (target T0 + max(0, A_hat(s_{j+1}))) move the end-of-episode value back one decision per
iteration, so with 3 iterations only the last 3 decisions of an episode see it - 150 ms at the 50 ms cadence (synthetic 40-step episode worth
+1 to keep: the first decision read "pull"). Instead, with T0_j = R_j (- P_e if last), the advantage of keeping now and then following policy
pi is the sum of T0 from j until pi next pulls (or the episode ends): G_j = T0_j + [pi keeps at s_{j+1}] G_{j+1}. Iteration 0: pi = keep-all
(G = keep through the episode and tail minus pull now); iteration n: pi keeps at s iff A_hat_{n-1}(s) > 0, with A_hat_{n-1} OUT OF FOLD (a model
not trained on that day); 3 iterations; the share of decisions that change between the last two iterations is printed (convergence).
CHRONOLOGICAL FOLDS on the 9 consumed days (C8; development only): for test day D[i], i = 4..8 (10-27, 10-31, 11-01, 11-02, 11-03): train on
D[0..i-2], calibrate on D[i-1], test on D[i]; out-of-fold predictions inside the training block by leaving one training day out. FB25's
PULL-DEPTH model is refit the same way (train D[0..i-2]; its rows, features and parameters unchanged). Calibration on D[i-1]: the share of
decision rows FB25's rule keeps there (with the fold's PULL-DEPTH model), and theta_M = the matching quantile of A_hat on D[i-1].
Residual (stated): the guard values and FRONT-PULL's P (a PULL-DEPTH input) come from leave-one-day-out models fit before this study; only a
frozen forward test on untouched days removes them.
INPUTS: exactly ORDER-HARM v2's FEAT (STOP-DEPTH's step inputs, INFLUENCE's six fair-price inputs, the order's nine known-frame inputs).
Readings DECLARED BEFORE THE FIRST RUN (printed by report; the economic test is pf_order_harm_policy_v2):
 R660 OPTION (printed): the mean continuation term (G_j - T0_j under the final policy) by rung, the share of test-day decisions whose sign
      (keep if A_hat > 0) differs between iteration 0 (keep-all continuation) and the final iteration, and between the last two iterations.
 R661 SKILL: Spearman IC of the final A_hat with a REALIZED, model-free advantage - keeping the order through the rest of the episode and its
      tail minus pulling now (KAV - P_e) - on each test day: > 0 on >= 4 of 5 test days and pooled >= 0.05 -> SKILL; else NO-SKILL. (The
      policy-iteration target contains the model's own decisions, so it is never used to score the model.) IC with the one-step realized
      target (R_j, last: R - P_e) printed.
 Printed: the label with a 5 s tail beside the 2 s one (episodes not censored at 5 s) - how much of keeping's value comes after the episode.
Controls (any failure -> no output): synthetic cases (keep-all lane semantics, re-entry, decision cadence anchored at the onset, client-known
resting, R's partition of (t_0 + 22, T_e], the continuation target on a 40-step episode) with planted errors (slow 250 ms cadence; venue-state
resting; intervals without the cancel latency; the one-step backup); v2's selftests (forks, causal queue inputs); on real data: the keep-all
lane with the recording
callback equals it without, on every window-side and rung; P_e from the episode's last decision equals P_e from its first on every 20th
multi-decision episode; the R partition on every episode; the no-look-ahead falsifier (frames not yet known perturbed, order inputs identical)
on every 20th row; no row past the lane's end; no test or calibration day in a fit; memory self-guard (2 GiB): label rows are written in
batches with the inputs as float32 (the model's precision), training rows are sampled and predictions made batch by batch.
Limits: replay rule M (B1a); rungs alone (the joint replay couples them); the tail assumes keep-all after the episode (a later episode inside
the tail is not decided); queue value beyond T_e not counted; three policy-iteration steps (convergence printed); consumed days.
Usage (run_light, PF_SLOT=claude): pf_order_harm_v3.py selftest | rows <day> | fit <test day> | report; driver run_order_harm_v3.sh
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
import pf_pull_depth as PD                                # noqa: E402
import pf_stop_depth as SD                                # noqa: E402
import pf_influence as IN                                 # noqa: E402
import pf_order_harm_client_lane as CL                    # noqa: E402
import pf_order_harm_v2 as OH                             # noqa: E402  (order_inputs, FEAT, selftests)
import pf_joint_ladder as JL                              # noqa: E402  (synthetic contexts)

DAYS9 = K.DAYS9
OUTD = f'{K.IO}/order_harm_v3_client50_20261114'
SIZE = 30.0; NR = 6; EVERY = 1; H_TAIL = 2000.0; QCFG = ('M', 50.0); CLIP = 10.0; SUB_N = 300000; N_ITER = 3
assert EVERY * 50 == CL.DECISION_MS
FEAT = OH.FEAT
KEYS = ['day', 'm', 'side', 'k', 'rung_i', 'ep', 'j', 'J', 'last', 'next_ok', 't_rel', 'R', 'Pe', 'T0', 'KAV', 'Pe5', 'KAV5']
H_TAIL5 = 5000.0
FOLDS = {DAYS9[i]: (tuple(DAYS9[:i - 1]), DAYS9[i - 1]) for i in range(4, 9)}     # test day -> (training days, calibration day)
CHECK_EVERY = 20
MEM_CAP_MIB = 2048.0
INF = float('inf'); NAN = float('nan')
MUT_SLOW_CADENCE = MUT_VENUE_REST = MUT_NO_LAT = MUT_BOOTSTRAP = 0   # planted errors (selftest only)
INT_KEYS = ('day', 'm', 'side', 'k', 'rung_i', 'ep', 'j', 'J', 'last', 'next_ok')
BATCH = 200000                                                        # rows per parquet batch (memory)


# ---------------------------------------------------------------- building blocks
def run_fork(w, k, d, vg, q, shf, t_end):
    F, O, _, _ = CL.reference_lane(w, q, shf, d, vg, size=SIZE, start_k=k, end_time=t_end)
    return F, O


def episodes(vg):
    """Maximal guard-on runs: [(k_on, k_clear)], k_clear = the first step after the run (None if it reaches the lane's end)."""
    out = []; k = 0; n = len(vg)
    while k < n:
        if vg[k]:
            k0 = k
            while k < n and vg[k]:
                k += 1
            out.append((k0, k if k < n else None))
        else:
            k += 1
    return out


def client_resting(o, t):
    """Our order rests as far as the client knows at t: activation reached (a refusal is known there), no cancel sent, no full fill known."""
    if o['ref'] or o['ta'] > t or t >= o['tc']:
        return False
    return t < (o['tfill'] if MUT_VENUE_REST else o['tfill'] + K.L_FILL)


def decision_steps(k_on, k_clear, n, o, tgl, has_input):
    """The cadence: the onset step and every 50 ms step after it while the guard stays on and o rests (client-known). Returns (steps,
    forced_pull_after): a step without inputs ends the sequence as a forced pull."""
    kc = n if k_clear is None else k_clear
    ks = list(range(k_on, kc, 5 if MUT_SLOW_CADENCE else EVERY))
    steps = []
    for k in ks:
        if not client_resting(o, tgl[k]):
            return steps, False
        if not has_input(k):
            return steps, True
        steps.append(k)
    return steps, False


def dp_fields(ts, t_end, fills):
    """R_j per decision time ts[j]: the value of fills (sim time, c x sh) in (t_j + L_CANCEL, t_{j+1} + L_CANCEL]; the last to t_end."""
    lat = 0.0 if MUT_NO_LAT else K.L_CANCEL
    edges = [t + lat for t in ts] + [t_end]
    R = [0.0] * len(ts)
    for (t, v) in fills:
        if not (edges[0] < t <= edges[-1]):
            continue
        j = bisect.bisect_left(edges, t) - 1
        R[max(min(j, len(ts) - 1), 0)] += v
    return R


def continuation_targets(T0, nxt, keep_next):
    """Policy-iteration target: G_j = T0_j + [the policy keeps at the next decision] G_{j+1}. Rows of one episode are consecutive with j rising;
    nxt[i] = the row of the next decision of the same episode (-1: none, the episode's last decision or a forced pull next); keep_next[i] = the
    policy's decision at that row (True = keep). Keep-all gives the realized keep-through-the-episode-and-tail minus pull-now advantage."""
    G = np.array(T0, dtype=np.float64)
    if MUT_BOOTSTRAP:                                     # planted error: no continuation (one decision only)
        return G
    for i in range(len(G) - 1, -1, -1):
        j = nxt[i]
        if j >= 0 and keep_next[i]:
            G[i] += G[j]
    return G


# ---------------------------------------------------------------- self-tests
def selftest():
    bad = []
    # (1) keep-all lane: rung 0 at 50 (rule U) rests before the guard (on 200-750 ms) and is kept; a sweep at 400 ms fills it; nothing is placed
    #     while the guard is on; re-entry at 750 + 250 ms (live 40 ms later); the pull branch re-enters at the same time from any decision step.
    fb = [(0.0, [(50, 100), (49, 100)], 52)]
    w = JL._ctx(0, fb, [(400.0, 48, 100.0)], t_end=3000.0, v=[0.5] * 60)
    vg = [False] * 4 + [True] * 12 + [False] * 44
    rec = {}

    def recorder(k, t, cur):
        rec.setdefault(id(cur), cur)
        return False
    Fk, Ok, _, _ = CL.reference_lane(w, 'U', 0.0, 0, vg, size=SIZE, decide=recorder)
    F0, O0, _, _ = CL.reference_lane(w, 'U', 0.0, 0, vg, size=SIZE)
    if (Fk, Ok) != (F0, O0):
        bad.append('the recording callback changed the keep-all lane')
    if [o[0] for o in Ok] != [K.L_PLACE, 1000.0 + K.L_PLACE] or sum(f[2] for f in Fk) != 30.0:
        bad.append(f'keep-all lane: orders {[o[0] for o in Ok]}, fills {[f[2] for f in Fk]}')
    Ff1, Of1 = run_fork(w, 4, 0, vg, 'U', 0.0, 2750.0)
    Ff2, Of2 = run_fork(w, 9, 0, vg, 'U', 0.0, 2750.0)
    if [o[0] for o in Of1] != [1000.0 + K.L_PLACE] or [o[0] for o in Of1] != [o[0] for o in Of2] or Ff1 != Ff2:
        bad.append(f'pull branch not constant within the episode: {[o[0] for o in Of1]} vs {[o[0] for o in Of2]}')
    # (2) client-known resting: an order filled at 400 ms rests (to the client) until 500 ms; a cancel sent at 600 ms ends it there
    o = dict(ref=False, ta=100.0, tc=INF, tfill=400.0)
    if [client_resting(o, t) for t in (50.0, 100.0, 450.0, 500.0)] != [False, True, True, False]:
        bad.append('client-known resting (fill)')
    o2 = dict(ref=False, ta=100.0, tc=600.0, tfill=INF)
    if [client_resting(o2, t) for t in (599.0, 600.0)] != [True, False]:
        bad.append('client-known resting (cancel)')
    # (3) 50 ms cadence: every step from 3 through 14; missing inputs force a pull at that step
    tgl = [50.0 * k for k in range(30)]; orest = dict(ref=False, ta=0.0, tc=INF, tfill=INF)
    st, fp = decision_steps(3, 15, 30, orest, tgl, lambda k: True)
    if st != list(range(3, 15)) or fp:
        bad.append(f'cadence {st}')
    st, fp = decision_steps(3, 15, 30, orest, tgl, lambda k: k != 8)
    if st != list(range(3, 8)) or not fp:
        bad.append(f'forced pull at a step without inputs {st} {fp}')
    # (4) R's partition: decisions at 200 / 450 / 700 ms, T_e 2750; a fill at 465 ms (inside the 22 ms after 450) belongs to the first interval
    fills = [(300.0, -60.0), (465.0, 30.0), (480.0, -10.0), (900.0, 15.0), (2800.0, 99.0)]
    R = dp_fields([200.0, 450.0, 700.0], 2750.0, fills)
    if R != [-30.0, -10.0, 15.0] or abs(sum(R) - sum(v for t, v in fills if 222.0 < t <= 2750.0)) > 1e-12:
        bad.append(f'R partition {R}')
    # (5) the continuation target: a 40-decision episode (2 s at 50 ms) worth +1 to keep through, all of it at the end
    J = 40; T0 = np.zeros(J); T0[-1] = 1.0; nxt = np.r_[np.arange(1, J), -1]
    G = continuation_targets(T0, nxt, np.ones(J, bool))
    if not np.allclose(G, 1.0):
        bad.append(f'keep-all continuation reaches {int((G > 0).sum())} of {J} decisions')
    kn = np.ones(J, bool); kn[19] = False                   # the policy pulls at decision 20: the earlier decisions get no continuation past it
    G = continuation_targets(T0, nxt, kn)
    if not (np.allclose(G[:20], 0.0) and np.allclose(G[20:], 1.0)):
        bad.append('continuation not cut at the policy pull')
    G = continuation_targets(np.array([1.0, 2.0, 4.0, 8.0]), np.array([1, -1, 3, -1]), np.ones(4, bool))
    if list(G) != [3.0, 2.0, 12.0, 8.0]:
        bad.append(f'continuation crossed an episode boundary {list(G)}')
    # (6) batched label I/O: rows written in two batches come back in order, inputs float32 in FEAT order, episode links intact, read in batches of 3
    import tempfile
    global OUTD
    old = OUTD
    with tempfile.TemporaryDirectory() as tmp:
        OUTD = tmp
        try:
            nr = 7; A = np.zeros((nr, len(KEYS) + len(FEAT)))
            for c, v in (('m', 5), ('rung_i', [0, 0, 0, 1, 1, 2, 2]), ('ep', [3, 3, 3, 3, 3, 9, 9]), ('j', [0, 1, 2, 0, 1, 0, 1]),
                         ('next_ok', [1, 1, 0, 1, 0, 1, 0]), ('T0', np.arange(nr))):
                A[:, KEYS.index(c)] = v
            A[:, len(KEYS):] = np.arange(nr)[:, None] * 1000.0 + np.arange(len(FEAT))[None, :] + 0.25
            wr = RowWriter(f'{tmp}/rows_x.parquet'); wr.write(A[:3]); wr.write(A[3:]); wr.close()
            with open(f'{tmp}/rows_x.json', 'w') as fh:
                json.dump(dict(stat={}, **CL.stamp()), fh)
            kd = load_keys('x')
            if list(kd['nxt']) != [1, 2, -1, 4, -1, 6, -1] or list(kd['T0']) != list(range(nr)) or wr.rows != nr:
                bad.append(f'batched label keys {list(kd["nxt"])}')
            got = feats_rows('x', np.array([1, 3, 5]), batch=3)
            if not np.array_equal(got, A[[1, 3, 5], len(KEYS):].astype(np.float32)) or got.dtype != np.float32:
                bad.append('batched label inputs (order / precision)')
            pr = predict_day(type('M', (), {'predict': staticmethod(lambda X: X[:, 0].astype(np.float64))})(), 'x', batch=3)
            if not np.array_equal(pr, A[:, len(KEYS)].astype(np.float32).astype(np.float64)):
                bad.append('batched predictions out of order')
        finally:
            OUTD = old
    return bad


def mutation_check():
    out = {}
    for nm in ('MUT_SLOW_CADENCE', 'MUT_VENUE_REST', 'MUT_NO_LAT', 'MUT_BOOTSTRAP'):
        globals()[nm] = 1
        try:
            out[nm] = bool(selftest())
        finally:
            globals()[nm] = 0
    return out


def all_selftests():
    OH.all_selftests()
    b = selftest(); m = mutation_check()
    if b or not all(m.values()):
        sys.exit(f'ORDER-HARM v3 SELFTEST FAILED -> no output: {b} planted errors caught {m}')
    print(f'order-harm v3 selftests passed; planted errors caught: {m}', flush=True)
    return m


# ---------------------------------------------------------------- rows (one day)
class RowWriter:
    """Label rows to parquet batch by batch: keys int64 / float64, the 97 model inputs float32 (the precision the model is fit and applied in)."""
    def __init__(self, path):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.pa = pa
        self.schema = pa.schema([(c, pa.int64() if c in INT_KEYS else pa.float64()) for c in KEYS] + [(c, pa.float32()) for c in FEAT])
        self.w = pq.ParquetWriter(path, self.schema); self.rows = 0

    def write(self, A):
        pa = self.pa; nk = len(KEYS)
        cols = [pa.array(A[:, i].astype(np.int64) if c in INT_KEYS else A[:, i]) for i, c in enumerate(KEYS)]
        cols += [pa.array(A[:, nk + i].astype(np.float32)) for i in range(len(FEAT))]
        self.w.write_table(pa.Table.from_arrays(cols, schema=self.schema)); self.rows += len(A)

    def close(self):
        self.w.close()


def rows_day(day):
    all_selftests()
    T0_ = time.time(); D = K.load_day(day); q, shf = QCFG
    SR = pd.read_parquet(f'{SD.OUTD}/rows_{day}.parquet', columns=['m', 'side', 'k'] + SD.NEW_FEAT); SRI = SR.groupby(['m', 'side']).indices
    SRV = SR[SD.NEW_FEAT].to_numpy(dtype=np.float64); SRK = SR.k.values.astype(int); X31 = SD.NEW_FEAT.index('x31'); del SR
    P = pd.read_parquet(f'{K.IO}/step5/{day}/rows_rl_d2_pk.parquet', columns=['mid', 'tau', 'p_bn', 'tau_rem', 'sigma'])
    P = P[P.mid.isin(D['W0'])].sort_values(['mid', 'tau'], kind='stable')
    PB = {int(m): (g.tau.values.astype(float).tolist(), g.p_bn.values.astype(float), g.tau_rem.values.astype(float), g.sigma.values.astype(float))
          for m, g in P.groupby('mid', sort=False)}
    del P
    st = dict(window_sides=0, episodes=0, rung_episodes=0, censored=0, censored5=0, no_input_at_onset=0, two_resting=0, rows=0, forced_pull=0,
              multi=0, recorder_differ=0, pconst_checks=0, pconst_differ=0, partition_bad=0, lookahead_checks=0, lookahead_bad=0)
    os.makedirs(OUTD, exist_ok=True)                     # memory: rows go to a partial parquet file in batches (inputs as float32)
    part = f'{OUTD}/rows_{day}.parquet.partial'; writer = RowWriter(part); buf = []; nbuf = 0
    for w in K.window_sides(D):
        st['window_sides'] += 1; m, side, sg, n = int(w['m']), int(w['side']), w['sg'], w['n']
        if not n:
            continue
        tf, mff = w['tf_mff']; tfl = tf.tolist(); P5, S5 = w['P5'], w['S5']; kf = np.asarray(w['kf'], float); kfl = kf.tolist()
        tgl = w['tg'].tolist(); vv = np.asarray(w['vkl'], float); vg = (vv <= K.GUARD_PULL).tolist()
        rt, pv, trv, sgv = PB.get(m, ([], np.zeros(0), np.zeros(0), np.zeros(0))); adv = IN.adv_of(side)
        MK = {}
        if (m, side) in SRI:
            for j_ in SRI[(m, side)]:
                k = int(SRK[j_])
                if not (0 <= k < n and vg[k] and vv[k] != OH.MASK_V):
                    continue
                t = tgl[k]; pk = IN.pbn_at(rt, pv, t)
                if np.isfinite(pk):
                    jj = bisect.bisect_right(rt, t) - 1
                    inf = [adv * (pk - IN.pbn_at(rt, pv, t - 250.0)), adv * (pk - IN.pbn_at(rt, pv, t - 1000.0)), adv * float(SRV[j_, X31]),
                           IN.sens_of(pk, float(sgv[jj]), float(trv[jj])), abs(pk - 50.0), float(trv[jj])]
                else:
                    inf = [NAN] * len(IN.INF_FEAT)
                MK[k] = np.concatenate([SRV[j_], np.asarray(inf, float)])
        eps = episodes(vg); st['episodes'] += len(eps)
        tt = w['anc'] - shf; cache = w.setdefault('cache', {})
        val = lambda i, pt, qq: sg * (mff[max(int(np.searchsorted(tf, float(w['anc'][i]) + 1000.0, side='right')) - 1, 0)] - pt) * qq
        rw = []
        for d in range(NR):
            rec = {}

            def recorder(k, t, cur, rec=rec):
                rec.setdefault(id(cur), cur)
                return False
            Fk, Ok, order_states, _ = CL.reference_lane(w, q, shf, d, vg, size=SIZE, decide=recorder)
            F0, O0, _, _ = CL.reference_lane(w, q, shf, d, vg, size=SIZE)
            st['recorder_differ'] += int(Fk != F0 or Ok != O0)
            ka = [(float(tt[f[0]]), val(f[0], int(f[1]), f[2])) for f in Fk]
            orders = order_states
            for (k_on, k_clear) in eps:
                if k_clear is None or tgl[k_clear] + H_TAIL > w['te']:
                    st['censored'] += 1; continue
                T_e = tgl[k_clear] + H_TAIL; t_on = tgl[k_on]; T_e5 = tgl[k_clear] + H_TAIL5; c5 = T_e5 > w['te']
                res = [o for o in orders if client_resting(o, t_on)]
                if not res:
                    continue
                if len(res) > 1:
                    st['two_resting'] += 1; continue
                o = res[0]; st['rung_episodes'] += 1
                steps, forced = decision_steps(k_on, k_clear, n, o, tgl, lambda k: k in MK)
                if not steps:
                    st['no_input_at_onset'] += int(forced); continue
                st['forced_pull'] += int(forced)
                ts = [tgl[k] for k in steps]
                t_last_end = (tgl[k_on + EVERY * len(steps)] + K.L_CANCEL) if forced else T_e
                R = dp_fields(ts, t_last_end, ka)
                st['censored5'] += int(c5)
                Ff, _ = run_fork(w, k_on, d, vg, q, shf, T_e if c5 else T_e5)
                Pe = sum(val(f[0], int(f[1]), f[2]) for f in Ff if ts[0] + K.L_CANCEL < float(tt[f[0]]) <= T_e)
                Pe5 = NAN if c5 else sum(val(f[0], int(f[1]), f[2]) for f in Ff if ts[0] + K.L_CANCEL < float(tt[f[0]]) <= T_e5)
                R5 = None if (c5 or forced) else dp_fields(ts, T_e5, ka)
                tot = sum(v for (t, v) in ka if ts[0] + K.L_CANCEL < t <= t_last_end)
                st['partition_bad'] += int(abs(sum(R) - tot) > 1e-6)
                if len(steps) >= 2:
                    st['multi'] += 1
                    if st['multi'] % CHECK_EVERY == 1:            # control: P_e is the same from the last decision (every 20th, from the first)
                        Ff2, _ = run_fork(w, steps[-1], d, vg, q, shf, T_e)
                        Pe2 = sum(val(f[0], int(f[1]), f[2]) for f in Ff2 if ts[-1] + K.L_CANCEL < float(tt[f[0]]) <= T_e)
                        st['pconst_checks'] += 1; st['pconst_differ'] += int(abs(Pe2 - Pe) > 1e-6)
                pt = int(o['pt']); lvp = cache.get(pt)
                if lvp is None:
                    lvp = QR.lvarr(P5, S5, pt, side); cache[pt] = lvp
                b0k = int(w['best'][k_on]) if w['best'][k_on] >= 0 else int(w['best'][steps[0]])
                J = len(steps); kav = np.cumsum(R[::-1])[::-1]
                kav5 = np.cumsum(R5[::-1])[::-1] if R5 is not None else np.full(J, NAN)
                for j, k in enumerate(steps):
                    t = tgl[k]; jk = int(w['jk'][k])
                    ordf = OH.order_inputs(o, t, d, side, int(w['best'][k]), b0k, lvp, S5, P5, jk, kfl, tfl)
                    if st['rows'] % CHECK_EVERY == 0:          # control: frames not yet known at t perturbed -> identical order inputs
                        S5p = S5.copy(); fut = kf > t; S5p[fut] = S5[fut] * 3.0 + 11.0
                        ordp = OH.order_inputs(o, t, d, side, int(w['best'][k]), b0k, QR.lvarr(P5, S5p, pt, side), S5p, P5, jk, kfl, tfl)
                        st['lookahead_checks'] += 1
                        st['lookahead_bad'] += int(not all((x == y) or (x != x and y != y) for x, y in zip(ordf, ordp)))
                    last = (j == J - 1) and not forced; nok = j < J - 1
                    Rj = R[j] / SIZE; pe = Pe / SIZE
                    T0 = Rj - (pe if last else 0.0)
                    rw.append([int(day), m, side, k, d, k_on, j, J, int(last), int(nok), t - w['ws'], Rj, pe, T0, kav[j] / SIZE, Pe5 / SIZE,
                               kav5[j] / SIZE] + list(MK[k]) + ordf)
                    st['rows'] += 1
        if rw:
            buf.append(np.asarray(rw, dtype=np.float64)); nbuf += len(rw)
            if nbuf >= BATCH:
                writer.write(np.vstack(buf)); buf = []; nbuf = 0
        w.pop('cache', None)
        if st['window_sides'] % 96 == 0:
            r2 = K.rss_mib(); print(f'{day}: {st["window_sides"]} window-sides, {st["rows"]} rows, {time.time() - T0_:.0f} s, rss {r2:.0f} MiB', flush=True)
            if r2 > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r2:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    if buf:
        writer.write(np.vstack(buf)); buf = []
    writer.close()
    if (st['recorder_differ'] or st['pconst_differ'] or st['partition_bad'] or st['lookahead_bad'] or not st['lookahead_checks']
            or not st['pconst_checks'] or writer.rows != st['rows']):
        os.remove(part)
        sys.exit(f'ORDER-HARM v3 CONTROL FAILED on {day} -> no output: {st}, rows written {writer.rows}')
    os.replace(part, f'{OUTD}/rows_{day}.parquet')
    with open(f'{OUTD}/rows_{day}.json', 'w') as handle:
        json.dump(dict(stat=st, **CL.stamp()), handle)
    print(f'{day}: {st} in {time.time() - T0_:.0f} s', flush=True)


# ---------------------------------------------------------------- chronological policy iteration (one test day)
def load_keys(day, full=False):
    """Compact keys of a day's rows: the next-decision link and T0 (training days); plus the columns calibration and diagnostics need (full)."""
    import pyarrow.parquet as pq
    with open(f'{OUTD}/rows_{day}.json') as fh:
        CL.require_contract(json.load(fh))
    cols = ['m', 'side', 'rung_i', 'ep', 'j', 'next_ok', 'T0'] + (['k', 'KAV', 'Pe', 'depth_now'] if full else [])
    R = pq.read_table(f'{OUTD}/rows_{day}.parquet', columns=cols).to_pandas()
    g = R.m.values.astype(np.int64) * 1000 + R.side.values.astype(np.int64) * 100 + R.rung_i.values.astype(np.int64)
    same = np.r_[(g[1:] == g[:-1]) & (R.ep.values[1:] == R.ep.values[:-1]) & (R.j.values[1:] == R.j.values[:-1] + 1), False]
    nxt = np.where(same & (R.next_ok.values == 1), np.arange(len(R)) + 1, -1)
    kd = dict(nxt=nxt, T0=R.T0.values.astype(np.float64), n=len(R))
    if full:
        kd['df'] = R
    return kd


def feat_batches(day, batch=None):
    """The day's 97 inputs as float32 arrays in FEAT order, batch by batch (memory)."""
    import pyarrow.parquet as pq
    for b in pq.ParquetFile(f'{OUTD}/rows_{day}.parquet').iter_batches(batch_size=batch or BATCH, columns=FEAT):
        names = b.schema.names
        yield np.column_stack([b.column(names.index(c)).to_numpy(zero_copy_only=False) for c in FEAT]).astype(np.float32)


def feats_rows(day, idx, batch=None):
    """float32 inputs of the rows idx (sorted), read batch by batch."""
    out = []; lo = 0
    for X in feat_batches(day, batch):
        hi = lo + len(X); sel = idx[(idx >= lo) & (idx < hi)] - lo
        if len(sel):
            out.append(X[sel])
        lo = hi
    return np.vstack(out) if out else np.zeros((0, len(FEAT)), np.float32)


def predict_day(b, day, batch=None):
    return np.concatenate([b.predict(X) for X in feat_batches(day, batch)] or [np.zeros(0)])


def policy_keep_next(kd, a):
    """The policy's decision at each row's next decision: keep-all (a is None) or keep iff A_hat > 0 there."""
    nxt = kd['nxt']
    return (nxt >= 0) if a is None else ((nxt >= 0) & (a[np.clip(nxt, 0, None)] > 0))


def fit_model(days, KD, TG, seed):
    import lightgbm as lgb
    per = SUB_N // max(len(days), 1); Xs = []; ys = []
    for x in days:
        nr = KD[x]['n']; idx = np.sort(np.random.default_rng(seed * 31 + int(x)).choice(nr, min(nr, per), replace=False))
        Xs.append(feats_rows(x, idx)); ys.append(np.clip(TG[x][idx], -CLIP, CLIP))
    preg = dict(PD.LGB_PARAMS, objective='regression')
    return lgb.train(preg, lgb.Dataset(np.vstack(Xs), np.concatenate(ys), feature_name=FEAT), PD.N_ROUNDS)


def fit_fold(test):
    import lightgbm as lgb
    from scipy.stats import spearmanr
    all_selftests(); T0_ = time.time()
    train, cal = FOLDS[test]
    if test in train or cal in train or test == cal or any(DAYS9.index(x) >= DAYS9.index(cal) for x in train):
        sys.exit(f'REFUSED: fold {test} not chronological: train {train}, calibration {cal}')
    KD = {x: load_keys(x) for x in train}; KC = {y: load_keys(y, full=True) for y in (cal, test)}
    oof = None; preds = {cal: [], test: []}; info = dict(train=list(train), cal=cal, test=test, iters=[])
    os.makedirs(f'{OUTD}/models', exist_ok=True)
    for it in range(N_ITER):
        TG = {x: continuation_targets(KD[x]['T0'], KD[x]['nxt'], policy_keep_next(KD[x], None if oof is None else oof[x])) for x in train}
        new = {}
        for x in train:                                    # out of fold inside the training block (leave one training day out)
            b = fit_model([y for y in train if y != x], KD, TG, seed=int(test) % 1000 * 10 + it)
            new[x] = predict_day(b, x)
        ball = fit_model(list(train), KD, TG, seed=int(test) % 1000 * 10 + it + 5)
        for y in (cal, test):
            preds[y].append(predict_day(ball, y))
        if it == N_ITER - 1:
            ball.save_model(f'{OUTD}/models/a_{test}.txt')
        flip = None if oof is None else float(np.mean(np.concatenate([(new[x] > 0) != (oof[x] > 0) for x in train])))
        oof = new
        info['iters'].append(dict(it=it, policy='keep-all' if it == 0 else 'A_hat > 0 (out of fold)',
                                  train_target_mean=float(np.mean(np.concatenate([TG[x] for x in train]))),
                                  oof_keep_share=float(np.mean(np.concatenate([oof[x] > 0 for x in train]))), oof_flip_vs_previous=flip))
    # PULL-DEPTH refit on the same training days (FB25 / FB25-C's model in the replay)
    pdc = ['day', 'm', 'side', 'k', 'mv1000'] + PD.FEAT; parts = []; per = SUB_N // len(train)
    for x in train:                                       # memory: each training day subsampled as it is read
        R_ = pd.read_parquet(f'{PD.OUTD}/rows_{x}.parquet', columns=pdc); R_ = R_[np.isfinite(R_.mv1000.values)]
        R_ = R_.iloc[np.sort(np.random.default_rng(int(test) % 1000 * 7 + int(x) % 1000).choice(len(R_), min(len(R_), per), replace=False))]
        parts.append(R_); del R_
    tr = pd.concat(parts, ignore_index=True); del parts
    if tr.day.isin([int(cal), int(test)]).any():
        sys.exit('REFUSED: a calibration / test day in the PULL-DEPTH refit')
    Xtr = tr[PD.FEAT].values.astype(np.float32); pdm = []
    for j in PD.JS:
        pdm.append(lgb.train(PD.LGB_PARAMS, lgb.Dataset(Xtr, (tr.mv1000.values >= j).astype(int)), PD.N_ROUNDS))
        pdm[-1].save_model(f'{OUTD}/models/pd_{test}_j{j}.txt')
    del Xtr, tr
    for y in (cal, test):
        PY = pd.read_parquet(f'{PD.OUTD}/rows_{y}.parquet', columns=['m', 'side', 'k'] + PD.FEAT)
        X = PY[PD.FEAT].values.astype(np.float32); prev = np.ones(len(X)); out = dict(m=PY.m.values, side=PY.side.values, k=PY.k.values)
        for j, b in zip(PD.JS, pdm):
            p = np.minimum(b.predict(X), prev); prev = p; out[f'P{j}'] = p
        pd.DataFrame(out).to_parquet(f'{OUTD}/pd_{test}_{y}.parquet', index=False)
        del PY, X
    # calibration on the calibration day: FB25's keep share on its decision rows -> theta_M
    C = KC[cal]['df'][['m', 'side', 'k', 'rung_i', 'depth_now']].merge(pd.read_parquet(f'{OUTD}/pd_{test}_{cal}.parquet'), on=['m', 'side', 'k'],
                                                                       how='left', validate='many_to_one')
    dn = C.depth_now.values.astype(np.float64); off = C.rung_i.values
    jx = np.clip(np.where(np.isfinite(dn), dn, -1) + 1, 1, len(PD.JS)).astype(int)
    pj = C[[f'P{j}' for j in PD.JS]].values[np.arange(len(C)), jx - 1]
    keep_fb = ~((~np.isfinite(dn)) | (dn < 0) | ((dn == 0) & (off == 0))) & np.isfinite(pj) & (pj < 0.25)
    a_cal = preds[cal][-1]; s_fb = float(keep_fb.mean())
    thr = dict(**CL.stamp(), test=test, cal=cal, train=list(train), s_FB_cal=s_fb, theta_M=float(np.quantile(a_cal, 1.0 - s_fb)),
               keep_OHQ0_cal=float((a_cal > 0).mean()))
    json.dump(thr, open(f'{OUTD}/thresholds_{test}.json', 'w'), indent=1)
    # diagnostics on the test day (R660 / R661)
    kt = KC[test]; a_fin = preds[test][-1]; a_init = preds[test][0]; a_prev = preds[test][-2] if N_ITER > 1 else a_init
    G = continuation_targets(kt['T0'], kt['nxt'], policy_keep_next(kt, a_fin))
    real = (kt['df'].KAV - kt['df'].Pe).values; cont = G - kt['T0']; rung = kt['df'].rung_i.values
    info.update(ic_final=float(spearmanr(a_fin, real)[0]), ic_final_onestep=float(spearmanr(a_fin, kt['T0'])[0]),
                ic_init=float(spearmanr(a_init, real)[0]), flip_vs_init=float(np.mean((a_fin > 0) != (a_init > 0))),
                flip_last=float(np.mean((a_fin > 0) != (a_prev > 0))), keep_share_test=float(np.mean(a_fin > 0)), cont_mean=float(np.mean(cont)),
                cont_by_rung=[float(np.mean(cont[rung == r])) if (rung == r).any() else NAN for r in range(NR)],
                rows_test=int(len(a_fin)), rows_cal=int(len(a_cal)))
    pd.DataFrame(dict(m=kt['df'].m.values, side=kt['df'].side.values, k=kt['df'].k.values, rung_i=rung, j=kt['df'].j.values,
                      a_final=a_fin, a_init=a_init, target=G, realized=real)).to_parquet(f'{OUTD}/pred_{test}.parquet', index=False)
    json.dump(dict(info=info, thresholds=thr, **CL.stamp()), open(f'{OUTD}/fit_{test}.json', 'w'), indent=1, default=float)
    print(f'fold {test}: train {train}, cal {cal}; IC final {info["ic_final"]:+.3f}, keep share {info["keep_share_test"]:.3f}, flip last '
          f'{info["flip_last"]:.3f}, theta_M {thr["theta_M"]:+.3f}; {time.time() - T0_:.0f} s', flush=True)


# ---------------------------------------------------------------- report
def report():
    import pyarrow.parquet as pq
    from scipy.stats import spearmanr
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    receipts = {d: json.load(open(f'{OUTD}/rows_{d}.json')) for d in DAYS9}
    for receipt in receipts.values():
        CL.require_contract(receipt)
    S = {d: receipt['stat'] for d, receipt in receipts.items()}
    Fz = {t: json.load(open(f'{OUTD}/fit_{t}.json')) for t in FOLDS}
    for receipt in Fz.values():
        CL.require_contract(receipt)
    F0 = pd.concat([pq.read_table(f'{OUTD}/rows_{d}.parquet', columns=['day', 'rung_i', 'j', 'R', 'Pe', 'KAV', 'Pe5', 'KAV5'],
                                  filters=[('j', '=', 0)]).to_pandas() for d in DAYS9], ignore_index=True)      # first decisions only (memory)
    tot = {k_: sum(S[d][k_] for d in DAYS9) for k_ in S[DAYS9[0]]}
    L = [f'ORDER-HARM v3 (keep now with the option to cancel later; policy iteration; chronological folds), 9 consumed days, first 60 s, M 50, '
         f'{CL.DECISION_MS} ms decisions; as of {asof}', f'statuses: {tot}', '',
         'LABELS (c per share of the 30-sh order; first decision of each rung-episode): rows | mean R | keep-all to T_e minus pull (KAV - Pe) | '
         'share KAV > Pe | mean P_e']
    for r in list(range(NR)) + ['all']:
        x = F0 if r == 'all' else F0[F0.rung_i == r]
        L.append(f'  rung {r}: {len(x):7d} | {x.R.mean():+.3f} | {(x.KAV - x.Pe).mean():+.3f} | {(x.KAV > x.Pe).mean():.3f} | {x.Pe.mean():+.3f}')
    L += ['', 'TAIL SENSITIVITY (first decisions not censored at 5 s): keep-all minus pull with a 2 s tail | with a 5 s tail | rows']
    F5 = F0[np.isfinite(F0.KAV5.values) & np.isfinite(F0.Pe5.values)]
    for r in list(range(NR)) + ['all']:
        x = F5 if r == 'all' else F5[F5.rung_i == r]
        L.append(f'  rung {r}: {(x.KAV - x.Pe).mean():+.3f} | {(x.KAV5 - x.Pe5).mean():+.3f} | {len(x)}')
    L += ['', 'FOLDS (train -> calibrate -> test): IC final vs realized keep-all minus pull | IC final vs one-step realized target | IC iteration-0 '
          'model vs realized | keep share (A_hat > 0) test | flip vs iteration 0 | flip vs previous iteration | continuation mean | theta_M (FB25 keep '
          'share on the calibration day)']
    ic = []
    for t in FOLDS:
        i_ = Fz[t]['info']; th = Fz[t]['thresholds']; ic.append(i_['ic_final'])
        L.append(f'  {t}: train {",".join(x[4:] for x in i_["train"])} cal {i_["cal"][4:]} | {i_["ic_final"]:+.3f} | {i_["ic_final_onestep"]:+.3f} | '
                 f'{i_["ic_init"]:+.3f} | {i_["keep_share_test"]:.3f} | {i_["flip_vs_init"]:.3f} | {i_["flip_last"]:.3f} | {i_["cont_mean"]:+.3f} | '
                 f'{th["theta_M"]:+.3f} ({th["s_FB_cal"]:.3f})')
    P = pd.concat([pd.read_parquet(f'{OUTD}/pred_{t}.parquet', columns=['a_final', 'realized']) for t in FOLDS], ignore_index=True)
    pooled = float(spearmanr(P.a_final, P.realized)[0]); up = sum(1 for v in ic if v > 0)
    lab = 'SKILL' if up >= 4 and pooled >= 0.05 else 'NO-SKILL'
    L += ['', f'R660 OPTION (printed): continuation term by rung (fold mean): ' +
          ' '.join(f'{np.nanmean([Fz[t]["info"]["cont_by_rung"][r] for t in FOLDS]):+.3f}' for r in range(NR)) +
          f'; decisions flipped vs iteration 0 {np.mean([Fz[t]["info"]["flip_vs_init"] for t in FOLDS]):.3f}, vs the previous iteration '
          f'{np.mean([Fz[t]["info"]["flip_last"] for t in FOLDS]):.3f} (fold means)',
          f'R661 SKILL: IC > 0 on {up}/5 test days, pooled {pooled:+.3f} -> {lab}']
    open(f'{OUTD}/order_harm_v3.txt', 'w').write('\n'.join(L) + '\n')
    json.dump(dict(R661=lab, pooled_ic=pooled, ic=ic, **CL.stamp()), open(f'{OUTD}/order_harm_v3.json', 'w'), indent=1)
    print('\n'.join(L), flush=True)

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        all_selftests()
    elif cmd == 'rows':
        rows_day(sys.argv[2])
    elif cmd == 'fit':
        fit_fold(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit('usage: pf_order_harm_v3.py selftest | rows <day> | fit <test day> | report')
