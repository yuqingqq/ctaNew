"""THE FOUR-PART REVIEW OF THE PAIRED PIPELINE RUN: VALIDITY, PREDICTION, CONTROLLER, RISK (2026-11-19, session aea5bfb9). USER: "After the run,
Launch a loop: diagnosis first, if there are issues, clear the issues. After the run, we review: Validity: Are labels, timing, populations and
accounting correct? Prediction: Do predicted action advantages match realized outcomes? Controller: Does acting on those predictions improve value
and useful quoting? Risk: Where do severe losses and inventory costs arise?". Declared before any output of this module is read. It reads the
committed R0 run (pf_pipeline_paired v2, pf_pipeline_paired_eval v2), the benchmark v3 predictions (E1), the export v2 rows and the bank's
baseline (B) inputs, for one consumed evaluation day (Oct 26 or Oct 27). Nothing is fitted or chosen.

LEDGER (the review's own second path, at B): per opportunity with a path and per policy (A, A-PA, AF, AM, PI0, PI0-PA, NEU, NEU-PA, E1, E1-PA,
 PA and PI0-ALL = every opportunity under pi0's cancellation): its admission and first pull step (pf_pipeline_paired_eval.admissions / triggers),
 the B fills it keeps (a fill is removed at / after the landing 50 T + L of an executed pull, T < the order's structural end step), the episode
 value V (0 when not posted or refused), the kept quantity, the fill-time windows (before the first review at +100 ms / the earliest-effect
 delay up to +100 ms + L / after it), the spread component sign * (mid at the post - price) * q and the movement component V - spread, the
 time-integrated inventory sum q * (END - t) and the terminal inventory.
VALIDITY (predicates; a FAIL is an open defect for the loop to clear, reported - not a refusal of the review):
 V1 LABELS: for a deterministic order sample (oid % 25 == 0), every labelled RETAIN and POST row's pair_v and pair_q recomputed by a per-row loop
    over the keep / pull run's M50 fills and pi0's saved forecasts equal the saved labels (1e-6)
 V2 IDENTITY: on every saved label row, pair_v = sign * (pf_mid - p) * pair_q + pair_qm (1e-6 c)
 V3 TIMING: on every export row, decision - t = 0 (POST) / 50 j (RETAIN), effective - t = 58 (POST) / 50 j + 37 (RETAIN), endpoint - t = 5,000 and
    decision <= effective < endpoint; every pi0 step T* > j with its landing <= END; every kept B fill within [activation, END]
 V4 POPULATIONS: original = with a path + the no-path statuses; labelled POST rows = with a path; labelled RETAIN rows = eligible RETAIN export
    rows; the predictions cover exactly the labelled rows; every admission has a path; refused posts carry zero labels; nonfinite predictions are
    counted (fallbacks); every A reached row belongs to an A-admitted order
 V5 ACCOUNTING: the ledger's value per policy and side equals the committed evaluation v2 value at B (1e-6 relative); the POST labels sum to the
    ledger value of PI0-ALL
PREDICTION (per head A / AF / AM, side; neutral rows and the head's admission-aware reached rows): groups POST (pred > 0), NO_ENTRY (pred < 0),
 POST tie (pred = 0), KEEP (reached, pred > 0), PULL (reached, pred < 0), KEEP tie, fallback (nonfinite); per group rows, distinct episodes,
 episodes with a nonzero label, raw and episode-balanced mean prediction and label, bias, MAE; the old gate (sign + total ratio in [0.8, 1.25];
 its verdict stands) and, as a diagnostic, plan v2's amount screen (|episode-balanced bias| <= 0.05 c + 0.20 |episode-balanced label|, the sign
 of the episode-balanced label, >= 200 episodes and >= 20 with a nonzero label else INSUFFICIENT_SUPPORT; empty or zero -> NOT_EVALUABLE).
CONTROLLER (the committed evaluation v2): value at B per policy and side, per opportunity, against N0, PI0, NEU, E1 and the same-admission
 PULL-ALL; scenario counts above 0 and above PI0; useful quoting (resting, at-best, both-sides time, median / p90 lifetime, first-review
 reversals, cancellations per 100 filled shares, participation).
RISK (the ledger, per policy and side): unconditional over the opportunities with a path - mean max(-V, 0), P(V < 0), P(V < -30 c), mean
 max(-V - 30, 0); conditional - p95 / p99 of max(-V, 0) among filled episodes and among loss episodes; the worst 1% share of the total loss;
 where the loss and the severe episodes (V < -30 c) arise - by side, fill-time window, minute of the market window (phase 1-5), spread at the
 state (1 / 2 / 3-5 / > 5 c) and volatility (the day's sigma terciles over the opportunities with a path); the spread and movement components;
 the time-integrated and terminal inventory.
ADDED BEFORE THE FIRST RUN (an independent read-only review of the pipeline, 2026-11-19, P2 x3 / P3; nothing of this module had been run on
 data): the practical-gate predicates against the same-admission PULL-ALL are checked for NESTING (A's pulls never land before the first-review
 landing, so per admitted order A keeps a superset of A-PA's fills and live time: at-best time, first-review reversals and cancellations per
 filled share then favour A mechanically) and labelled IMPLIED-BY-NESTING; a non-nested A vs PI0 quoting table is added; the matched null carries
 its exact finite-population variance over the (day, window, side, minute) cells, sum_c k_c (n_c - k_c) / (n_c (n_c - 1)) sum_i (V_i - mean_c)^2,
 from an A-ALL ledger (every opportunity under A's cancellation), with z and a normal-approximation tail (labelled approximate), and 1,000 random
 draws of the same per-cell counts (seed 20261119; rule 6: >= 200) whose one-sided p = (1 + #draws >= A) / 1,001 is the reading MATCHED CONTROL
 WITH ITS SPREAD at p <= 0.05 (the draws' mean and variance must agree with the exact moments: V5); FINDINGS C1 (value
 by fill age < 100 / 100-500 / >= 500 ms after activation, the value without fills on orders < 100 ms old, FV3's B1a-weighted value), C3 (cancel-
 pending fills between the pull request and its landing) and C12 (value by trade-anchor source) per policy; V1 / V2 fail closed on NaN; V4 counts
 the original opportunities from two sources (the keep / pull orders and the export's POST rows) and the RETAIN steps without a forecast (they
 keep: fallback decisions); V5 compares at 1e-6 c absolute, reproduces the evaluation's matched-null mean and checks the evaluation's input hashes;
 pi0's (and A's) first-review share over every eligible opportunity and over its own admissions is computed (the latter = the evaluation's,
 V5); the export's RETAIN steps without a forecast (off_grid / feature_unavailable) on A's reached path and at its first review are counted.
LIMITS: consumed days; single-order episodes; the ledger at B only (the evaluation carries the 19 scenarios); the movement component runs from the
 posting mid to the END mark (no mid is stored per fill); a restricted experiment (POST vs NO_ENTRY to END).
SELFTEST (synthetic tape, no data): the ledger against the bank on planted policies; every validity predicate PASS on clean inputs and FAIL on its
 planted defect (a label off by 1 c, a broken identity, a timing field off by 1 ms, a reached row of an unposted order, a ledger landing moved);
 the prediction groups and screens and the risk tables on planted values. Mutations: pf_pipeline_paired_review_mutants.py.
Usage (run_light, PF_SLOT=claude): pf_pipeline_paired_review.py selftest | day <day>
"""
import contextlib
import gc
import io
import itertools
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_keep_pull as KP                                 # noqa: E402
import pf_keep_pull_scenarios as SC                       # noqa: E402
import pf_keep_pull_variants as V                         # noqa: E402
import pf_pipeline_paired as PQ                           # noqa: E402
import pf_pipeline_paired_eval as PQE                     # noqa: E402
import pf_pipeline_policy as P                            # noqa: E402
import pf_queue_place_test as QP                          # noqa: E402
import pf_queue_place_ledger as QL                        # noqa: E402

C, pd, I = KP.C, KP.pd, KP.I
OUT = Path(I.K.IO) / 'pipeline_paired_review_20261119'
MARK = 'PIPELINE PAIRED REVIEW'
B_NAME = V.B_NAME
POLICIES = {'PA': ('ALL', None), 'PI0-ALL': ('ALL', 'PI0'), 'A-ALL': ('ALL', 'A')}
for _r in PQE.RULES:
    POLICIES[_r] = (_r, _r); POLICIES[f'{_r}-PA'] = (_r, None)        # (admission set, trigger rule; None = pull at the first live step)
SAMPLE = 25                                                # V1's deterministic order sample (oid % SAMPLE == 0)
SPREADS = ((0., 1.), (1., 2.), (2., 5.), (5., np.inf))     # the spread strata, (lo, hi] cents
WINDOWS = ('before the first review', 'earliest-effect delay', 'after the earliest effect')
LAND_SHIFT = 0.                                            # planted-error switch (selftest only): the ledger's landing moved (ms)
LEDGER_NUDGE = 0.                                          # planted-error switch (selftest only): added to the ledger totals of |value| >= 100 c in V5
NULL_DRAWS, NULL_SEED = 1000, 20261119                     # the matched null's random draws (rule 6: >= 200) and their fixed seed
PATH_EXPORT = ('eligible', 'refused_post_only', 'off_grid', 'feature_unavailable')   # the export's POST statuses of an opportunity with a path
NO_FORECAST = ('off_grid', 'feature_unavailable')          # RETAIN steps without a forecast: the controller keeps (a fallback decision)


def ledger(Op, Os, Fs, sc, adm, hit):
    """Per opportunity with a path and per policy at one scenario: V, kept quantity, window values / quantities, spread / movement components,
    time-integrated and terminal inventory -> {policy: DataFrame indexed like Op}."""
    Os_ = Os.rename(columns={c: c + '_s' for c in ('refused', 'v', 'q', 't_full', 'land')})
    E = Op.merge(Os_, on=['day', 'oid'], how='left', validate='one_to_one').reset_index(drop=True)
    n = len(E); el = ~E.refused_s.astype(bool).values; key = pd.MultiIndex.from_arrays([E.day.astype(str).values, E.oid.astype(int).values])
    ix = pd.Series(np.arange(n), index=key); ri = ix.reindex(pd.MultiIndex.from_arrays([Fs.day.astype(str).values, Fs.oid.astype(int).values])).values
    C.require(not np.isnan(ri).any(), 'ledger: a fill of an opportunity without a path'); ri = ri.astype(int)
    C.require(bool(el[ri].all()), 'ledger: a fill of a refused opportunity')
    sg = np.where(E.side.values == 0, 1., -1.); t = Fs.t.values.astype(float); q = Fs.q.values.astype(float); v = q * sg[ri] * (E.mark.values[ri] - E.p.values[ri])
    endj = SC.endj_n(E.t_full_s.values, E.g_struct.values, sc['N']); lat = float(sc['L']); j0 = float(KP.landing(KP.J0, lat))
    win = np.where(t < KP.STEP * KP.J0, 0, np.where(t < j0, 1, 2)); edge = sg * (E.pf_mid_post.values.astype(float) - E.p.values.astype(float))
    age = np.digitize(t - float(sc['A']), SC.AGE_EDGES); src = Fs.src.values.astype(np.int64) if 'src' in Fs else np.full(len(Fs), -1)   # C1 / C12
    out = {}
    for name, (ent, rule) in POLICIES.items():
        a = np.ones(n, bool) if ent == 'ALL' else key.isin(pd.MultiIndex.from_tuples(sorted(adm[ent]))) if adm[ent] else np.zeros(n, bool)
        T = np.full(n, KP.J0) if rule is None else pd.Series({(str(k[0]), int(k[1])): int(x) for k, x in hit[rule].items()}, dtype=float).reindex(key).fillna(KP.NSTEP + 1).values.astype(int) \
            if len(hit[rule]) else np.full(n, KP.NSTEP + 1)
        act = (T >= KP.J0) & (T <= KP.NSTEP) & (T < endj); land = KP.landing(np.clip(T, 1, KP.NSTEP), lat) + LAND_SHIFT
        kept = a[ri] & el[ri] & ~(act[ri] & (t >= land[ri]))
        vv, qq = v * kept, q * kept
        D = pd.DataFrame(dict(V=np.bincount(ri, vv, n), q=np.bincount(ri, qq, n), tint=np.bincount(ri, qq * np.maximum(KP.H - t, 0.), n) / 1e3))
        for w in range(len(WINDOWS)):
            D[f'v_w{w}'] = np.bincount(ri, vv * (win == w), n); D[f'q_w{w}'] = np.bincount(ri, qq * (win == w), n)
        D['spread_comp'] = edge * D.q.values; D['move_comp'] = D.V.values - D.spread_comp.values; D['inv'] = sg * D.q.values
        for b in range(len(SC.AGE_LABELS)):
            D[f'v_a{b}'] = np.bincount(ri, vv * (age == b), n)
        for c_ in range(len(SC.SRC_LABELS)):
            D[f'v_s{c_}'] = np.bincount(ri, vv * (src == c_), n)
        D['v_s_unknown'] = np.bincount(ri, vv * ((src < 0) | (src >= len(SC.SRC_LABELS))), n)
        req = np.where(act, KP.STEP * T, np.inf); pm = kept & (t >= req[ri]) & (t < land[ri])                     # C3: between the request and the landing
        D['pend_v'] = np.bincount(ri, v * pm, n); D['pend_n'] = np.bincount(ri, pm.astype(float), n); D['gross'] = np.bincount(ri, np.abs(vv), n)
        D['admitted'] = a; D['activated'] = a & el; D['side'] = E.side.values.astype(int); D['phase'] = E.phase.values.astype(int)
        D['spread'] = E.spread.values.astype(float); D['sigma'] = E.sigma_post.values.astype(float); D['oid'] = E.oid.values.astype(int)
        D['m'] = E.m.values.astype(int); D['T'] = T; D['act'] = act; D['live0'] = el & (KP.J0 < endj)
        D['kept_t_min'] = pd.Series(np.where(kept, t, np.inf)).groupby(ri).min().reindex(range(n)).fillna(np.inf).values
        D['kept_t_max'] = pd.Series(np.where(kept, t, -np.inf)).groupby(ri).max().reindex(range(n)).fillna(-np.inf).values
        out[name] = D
    return out


def v1_labels(lab, fills, sample=SAMPLE):
    """V1: per-row loop on the sampled orders -> (rows checked, mismatches)."""
    chk = bad = 0; fl = fills.groupby('oid')
    for o, g in lab[lab.oid.astype(int) % sample == 0].groupby('oid'):
        f = fl.get_group(o) if o in fl.groups else fills.iloc[:0]
        rr = g[g.kind.astype(str) == 'retain'].sort_values('j'); pulls = [int(j) for j, x, k in zip(rr.j, rr.pi0_V, rr.pi0_ok) if bool(k) and x < 0]
        for r in rr.itertuples():
            nx = next((k for k in pulls if k > r.j), None); hi = KP.landing(nx) if nx is not None else np.inf; lo = KP.landing(int(r.j)); m = (f.t >= lo) & (f.t < hi)
            chk += 1; bad += not (abs(float(f.v[m].sum()) - r.pair_v) <= 1e-6 and abs(float(f.q[m].sum()) - r.pair_q) <= 1e-6)      # NaN fails
        for r in g[g.kind.astype(str) == 'post'].itertuples():
            hi = KP.landing(pulls[0]) if pulls else np.inf; m = f.t < hi; zero = str(r.status) == 'refused_post_only'
            chk += 1; bad += not (abs((0. if zero else float(f.v[m].sum())) - r.pair_v) <= 1e-6 and abs((0. if zero else float(f.q[m].sum())) - r.pair_q) <= 1e-6)
    return chk, bad


def v2_identity(lab):
    """V2: rows where pair_v != sign * (pf_mid - p) * pair_q + pair_qm."""
    sg = np.where(lab.side.values == 0, 1., -1.)
    return int((~(np.abs(lab.pair_v.values - (sg * (lab.pf_mid.values.astype(float) - lab.p.values.astype(float)) * lab.pair_q.values + lab.pair_qm.values)) <= 1e-6)).sum())   # NaN fails


def v3_timing(ex, lab, kept_t):
    """V3: (export rows with a timing field off, labelled rows whose pi0 step is not after j or lands after END, kept fills outside [activation, END])."""
    post = ex.kind.astype(str).values == 'post'; j = ex.j.values.astype(float); t = ex.t.values.astype(float)
    dec = np.where(post, 0., KP.STEP * j); eff = np.where(post, KP.PLACE, KP.landing(np.maximum(j, 1).astype(int)))
    off = ~(np.isclose(ex.decision_ms.values - t, dec, atol=1e-6) & np.isclose(ex.effective_ms.values - t, eff, atol=1e-6) & np.isclose(ex.endpoint_ms.values - t, KP.H, atol=1e-6)
            & (ex.decision_ms.values <= ex.effective_ms.values) & (ex.effective_ms.values < ex.endpoint_ms.values))
    r = lab[lab.kind.astype(str) == 'retain']; ts = r.tstar.values.astype(float); has = np.isfinite(ts)
    bad_t = int((has & ~((ts > r.j.values) & (KP.landing(np.nan_to_num(ts, nan=KP.J0).astype(int)) <= KP.H))).sum())
    lo, hi = kept_t
    return int(off.sum()), bad_t, int(((lo < KP.PLACE) | (hi > KP.H)).sum())


def null_moments(cell, sel, v):
    """The exact matched null of a selection: random selection of the same count k_c in each cell -> (mean, variance) of the selected total
    (sampling without replacement within a cell: variance sum_c k_c (n_c - k_c) / (n_c (n_c - 1)) sum_i (v_i - mean_c)^2)."""
    d = pd.DataFrame(dict(c=np.asarray(cell), k=np.asarray(sel, bool).astype(float), v=np.asarray(v, float))); g = d.groupby('c')
    n, k, sv = g.v.size(), g.k.sum(), g.v.sum(); mc = sv / n; ss = g.v.apply(lambda x: float(((x - x.mean()) ** 2).sum()))
    var = float((k * (n - k) / (n * (n - 1)).where(n > 1, 1.) * ss).where(n > 1, 0.).sum()); return float((k * mc).sum()), var


def null_draws(cell, sel, v, side, draws=None, seed=None):
    """The matched null by random draws: per draw, k_c opportunities chosen uniformly without replacement in each cell -> the drawn totals
    (draws x [both, bid, ask])."""
    draws = NULL_DRAWS if draws is None else draws; rng = np.random.default_rng(NULL_SEED if seed is None else seed)
    ci = pd.factorize(np.asarray(cell))[0]; n = len(ci); v = np.asarray(v, float); side = np.asarray(side)
    nc = np.bincount(ci); kc = np.bincount(ci, weights=np.asarray(sel, bool).astype(float)).astype(int); start = np.r_[0, np.cumsum(nc)[:-1]]
    out = np.empty((draws, 3)); pos = np.empty(n, int)
    for d_ in range(draws):
        pos[np.lexsort((rng.random(n), ci))] = np.arange(n); m = (pos - start[ci]) < kc[ci]
        out[d_] = (v[m].sum(), v[m & (side == 0)].sum(), v[m & (side == 1)].sum())
    return out


def null_agree(e0, v0, d):
    """Do the draws agree with the exact moments? |mean - e0| <= 4 sd / sqrt(draws) and the variance ratio in [0.8, 1.25] (zero variance: every
    draw equals e0)."""
    d = np.asarray(d, float)
    if v0 <= 0:
        return bool(np.allclose(d, e0, rtol=0, atol=1e-6))
    return bool(abs(float(d.mean()) - e0) <= 4. * math.sqrt(v0 / len(d)) and .8 <= float(d.var(ddof=1)) / v0 <= 1.25)


def spread_verdict(p_draws):
    """MATCHED CONTROL WITH ITS SPREAD: PASS when the draws' one-sided p is at most 0.05."""
    return 'PASS' if p_draws <= .05 else 'FAIL'


def first_review_share(D, mask):
    """The share of the orders in `mask` live at the first review (+100 ms) whose pull executes there."""
    m = np.asarray(mask, bool) & D.live0.values
    return float((m & D.act.values & (D['T'].values == KP.J0)).sum() / m.sum()) if m.any() else None


def nested(DA, DP, adm_mask):
    """Is A's same-admission comparison nested? Per admitted order A keeps at least A-PA's shares and the same fills before the first effect."""
    m = np.asarray(adm_mask, bool)
    return bool(m.any() and (DA.q.values[m] >= DP.q.values[m] - 1e-9).all() and np.allclose(DA.v_w0.values[m], DP.v_w0.values[m], atol=1e-9)
                and np.allclose(DA.v_w1.values[m], DP.v_w1.values[m], atol=1e-9))


def c1_c3_c12(D):
    """FINDINGS C1 (value by fill age; without fills on orders < 100 ms old; B1a-weighted), C3 (cancel-pending) and C12 (by trade-anchor source)."""
    import pf_fv3_check as FV
    va = [float(D[f'v_a{b}'].sum()) for b in range(len(SC.AGE_LABELS))]; V_ = float(D.V.sum())
    b1a = {k: float(sum(w * x for w, x in zip(wt, va))) for k, wt in FV.B1A.items()}
    return dict(C1=dict(by_age=dict(zip(SC.AGE_LABELS, va)), without_young=V_ - va[0], b1a_weighted=b1a, b1a_sign=SC.b1a_sign(V_, b1a) if V_ != 0 else 'ZERO'),
                C3=dict(fills=int(D.pend_n.sum()), value=float(D.pend_v.sum()), share_of_gross=float(np.abs(D.pend_v).sum() / D.gross.sum()) if D.gross.sum() > 0 else None),
                C12={**{sl: float(D[f'v_s{i}'].sum()) for i, sl in enumerate(SC.SRC_LABELS)}, 'unknown': float(D.v_s_unknown.sum())})


def episode_mean(oid, x):
    """The episode-balanced mean: the mean over episodes of each episode's own mean."""
    return float(pd.Series(np.asarray(x, float)).groupby(np.asarray(oid)).mean().mean()) if len(x) else float('nan')


def screen(oid, pred, lab, want_pos):
    """Plan v2's amount screen (a diagnostic here): support, then the sign of the episode-balanced label and |bias| <= 0.05 + 0.20 |label|."""
    n_ep = len(set(np.asarray(oid).tolist())); nz = len(set(np.asarray(oid)[np.asarray(lab) != 0].tolist()))
    if not len(lab) or float(np.sum(lab)) == 0.:
        return 'NOT_EVALUABLE', n_ep, nz
    if n_ep < 200 or nz < 20:
        return 'INSUFFICIENT_SUPPORT', n_ep, nz
    mp, ml = episode_mean(oid, pred), episode_mean(oid, lab)
    ok = (ml > 0 if want_pos else ml < 0) and abs(mp - ml) <= .05 + .20 * abs(ml)
    return ('PASS' if ok else 'FAIL'), n_ep, nz


def groups_of(P_, R_, head):
    """The prediction groups of one head and side: POST / NO_ENTRY / POST tie / fallbacks on the POST rows; KEEP / PULL / KEEP tie on the admission-
    aware reached RETAIN rows and on the neutral rows."""
    pp, pl, po = P_[f'{head}_V'].values.astype(float), P_.pair_v.values.astype(float), P_.oid.values.astype(int)
    kr, kl, ko = R_[f'{head}_V'].values.astype(float), R_.pair_v.values.astype(float), R_.oid.values.astype(int)
    adm = set(po[np.isfinite(pp) & (pp > 0)].tolist()); rch = PQ.reached(ko, R_.j.values, np.where(np.isfinite(kr), kr, 0.), None if REACH_LEAK else adm)
    G = {}
    for name, (o, p_, l_, m, pos) in {'POST (pred > 0)': (po, pp, pl, np.isfinite(pp) & (pp > 0), True), 'NO_ENTRY (pred < 0)': (po, pp, pl, np.isfinite(pp) & (pp < 0), False),
                                      'POST tie (pred = 0)': (po, pp, pl, pp == 0, None), 'POST fallback (nonfinite)': (po, pp, pl, ~np.isfinite(pp), None),
                                      'KEEP reached (pred > 0)': (ko, kr, kl, rch & np.isfinite(kr) & (kr > 0), True), 'PULL reached (pred < 0)': (ko, kr, kl, rch & np.isfinite(kr) & (kr < 0), False),
                                      'KEEP tie reached (pred = 0)': (ko, kr, kl, rch & (kr == 0), None), 'KEEP neutral (pred > 0)': (ko, kr, kl, np.isfinite(kr) & (kr > 0), True),
                                      'PULL neutral (pred < 0)': (ko, kr, kl, np.isfinite(kr) & (kr < 0), False), 'RETAIN fallback (nonfinite)': (ko, kr, kl, ~np.isfinite(kr), None)}.items():
        x = dict(rows=int(m.sum()), episodes=len(set(o[m].tolist())), nonzero_label_episodes=len(set(o[m & (l_ != 0)].tolist())))
        if m.any() and pos is not None:
            x.update(mean_pred=float(p_[m].mean()), mean_label=float(l_[m].mean()), eb_pred=episode_mean(o[m], p_[m]), eb_label=episode_mean(o[m], l_[m]),
                     bias=float((p_[m] - l_[m]).mean()), mae=float(np.abs(p_[m] - l_[m]).mean()), sum_pred=float(p_[m].sum()), sum_label=float(l_[m].sum()),
                     old_gate=PQ.gate_group(p_[m], l_[m], pos), plan_v2_screen=screen(o[m], p_[m], l_[m], pos)[0])
        G[name] = x
    return G, rch, adm


def tails(V_, filled):
    """Unconditional tails over every episode given, conditional among filled and among loss episodes, the worst 1% share."""
    L = np.maximum(-V_, 0.); S = np.maximum(-V_ - KP.SEVERE, 0.); n = len(V_); lf, ll = L[filled], L[V_ < 0]
    q = lambda x, p_: float(np.quantile(x, p_)) if len(x) else None
    k = max(int(np.ceil(.01 * n)), 1); worst = float(np.sort(L)[::-1][:k].sum() / L.sum()) if L.sum() > 0 else None
    return dict(episodes=n, mean_loss=float(L.mean()) if n else None, p_loss=float((V_ < 0).mean()) if n else None, p_severe=float((V_ < -KP.SEVERE).mean()) if n else None,
                mean_severe_excess=float(S.mean()) if n else None, total_loss=float(L.sum()), total_severe_excess=float(S.sum()), severe_episodes=int((V_ < -KP.SEVERE).sum()),
                p95_loss_filled=q(lf, .95), p99_loss_filled=q(lf, .99), p95_loss_loss=q(ll, .95), p99_loss_loss=q(ll, .99), worst_1pct_share_of_loss=worst, largest_loss=float(L.max()) if n else None)


def where_risk(D, tert):
    """Where the loss and the severe episodes arise: per stratum the episodes, total loss, severe episodes / excess and the window values."""
    L = np.maximum(-D.V.values, 0.); S = np.maximum(-D.V.values - KP.SEVERE, 0.); sev = D.V.values < -KP.SEVERE; out = {}
    strata = {'side': np.where(D.side.values == 0, 'bid', 'ask'), 'phase': D.phase.values.astype(str),
              'spread': np.select([(D.spread.values > lo) & (D.spread.values <= hi) for lo, hi in SPREADS], [f'({lo:g}, {hi:g}]' for lo, hi in SPREADS], 'n/a'),
              'sigma tercile': np.select([D.sigma.values <= tert[0], D.sigma.values <= tert[1], D.sigma.values > tert[1]], ['low', 'mid', 'high'], 'n/a')}
    for s_name, lab_ in strata.items():
        out[s_name] = {str(u): dict(episodes=int((lab_ == u).sum()), total_loss=float(L[lab_ == u].sum()), severe_episodes=int((sev & (lab_ == u)).sum()),
                                    severe_excess=float(S[lab_ == u].sum()), value=float(D.V.values[lab_ == u].sum()))
                       for u in sorted(set(lab_.tolist()))}
    out['window (all episodes)'] = {WINDOWS[w]: float(D[f'v_w{w}'].sum()) for w in range(len(WINDOWS))}
    out['window (severe episodes)'] = {WINDOWS[w]: float(D[f'v_w{w}'].values[sev].sum()) for w in range(len(WINDOWS))}
    out['components'] = dict(spread=float(D.spread_comp.sum()), movement=float(D.move_comp.sum()), severe_spread=float(D.spread_comp.values[sev].sum()),
                             severe_movement=float(D.move_comp.values[sev].sum()))
    out['inventory'] = dict(time_integrated_share_s=float(D.tint.sum()), mean_abs_terminal=float(np.abs(D.inv.values[D.admitted.values]).mean()) if D.admitted.any() else None,
                            peak_abs_terminal=float(np.abs(D.inv.values).max()) if len(D) else None, filled_episodes=int((D.q.values > 0).sum()),
                            movement_per_filled_share=float(D.move_comp.sum() / D.q.sum()) if D.q.sum() > 0 else None)
    return out


def run(day, out=None, sc_out=None, kp=None, paired=None, evald=None, bench=None, export=None, labels=None, pred=None, jpred=None, evj=None):  # noqa: C901
    """The four-part review of one evaluation day -> review_<day>.json / .txt / .csv and ledger_<day>.parquet."""
    start = time.time(); out = Path(out or OUT); sc_out = Path(sc_out or SC.OUT); kp = Path(kp or KP.OUT); paired = Path(paired or PQ.OUT)
    evald = Path(evald or PQE.OUT); bench = Path(bench or P.BENCH); export = Path(export or P.EXPORT); C.require(not (out / f'review_{day}.json').exists(), 'output exists')
    C.require(day in (PQ.EVAL if pred is None else KP.DAYS), 'undeclared day')
    pj = json.loads((paired / 'paired.json').read_text()); hashes = {}
    def rd(name, path, cols=None):
        C.require(pj['outputs'].get(path.name) == C.sha(path), f'{name}: {path.name} does not match the paired receipt'); hashes[path.name] = C.sha(path)
        return pd.read_parquet(path, columns=cols)
    lab = labels if labels is not None else rd('labels', paired / f'labels_{day}.parquet', ['oid', 'kind', 'j', 'side', 'status', 'p', 'pf_mid', 'pi0_V', 'pi0_ok', 'tstar',
                                                                                   'pair_v', 'pair_q', 'pair_qm'])
    lab = lab.drop(columns=[c for c in ('day',) if c in lab]); lab['kind'] = lab.kind.astype(str).astype('category'); lab['status'] = lab.status.astype(str).astype('category')   # memory
    prd = pred if pred is not None else rd('predictions', paired / f'pred_{day}.parquet', ['day', 'oid', 'kind', 'j', 'side', 'status', 'pi0_V', 'pi0_ok', 'NEU_V', 'NEU_ok', 'A_V', 'AF_V', 'AM_V', 'pair_v'])
    if jpred is None:
        bj = json.loads((bench / 'bench.json').read_text()); jf = bench / f'pred_{day}.parquet'
        C.require(bj['outputs'].get(jf.name) == C.sha(jf), 'the benchmark predictions do not match their receipt'); jpred = pd.read_parquet(jf, columns=['day', 'oid', 'kind', 'j', 'J_V', 'J_ok'])
    ev = evj if evj is not None else json.loads((evald / f'eval_{day}.json').read_text())
    lab['oid'] = lab.oid.astype(int); lab['j'] = lab.j.astype(int)
    for df in (prd, jpred):
        df['day'] = df.day.astype(str); df['oid'] = df.oid.astype(int); df['j'] = df.j.astype(int)
    prd = prd.merge(jpred, on=['day', 'oid', 'kind', 'j'], how='left', validate='one_to_one'); del jpred; gc.collect()
    prd['kind'] = prd.kind.astype(str).astype('category'); prd['status'] = prd.status.astype(str).astype('category'); prd['day'] = prd.day.astype('category')   # memory
    er = json.loads((export / day / 'receipt.json').read_text()); ex = []
    for x, h in er['outputs'].items():
        C.require(C.sha(export / day / x) == h, f'export part {x} changed since its receipt')
        ex.append(pd.read_parquet(export / day / x, columns=['oid', 'kind', 'j', 'status', 't', 'decision_ms', 'effective_ms', 'endpoint_ms', 'side', 'pf_mid', 'p', 'sigma']))
    ex = pd.concat(ex, ignore_index=True); ex['kind'] = ex.kind.astype(str).astype('category'); ex['status'] = ex.status.astype(str).astype('category')
    O, Fl, S, SF, Op, rec = V.load([day], sc_out, kp); O['day'] = O.day.astype(str); Op = Op.assign(day=Op.day.astype(str))
    pe = ex[ex.kind == 'post'].set_index(ex.oid[ex.kind == 'post'].astype(int).values)
    Op = Op.assign(pf_mid_post=pe.pf_mid.reindex(Op.oid.astype(int).values).values, sigma_post=pe.sigma.reindex(Op.oid.astype(int).values).values)
    post, ret = prd[prd.kind.astype(str) == 'post'], prd[prd.kind.astype(str) == 'retain']
    adm = {'ALL': set(zip(Op.day, Op.oid.astype(int)))}; adm.update({r: PQE.admissions(post, r) for r in PQE.RULES}); hit = {r: PQE.triggers(ret, r) for r in PQE.RULES}
    Os, Fs = V.scen_inputs(SC.B0, Op, Fl, S, SF); Os = Os.assign(day=Os.day.astype(str)); Fs = Fs.assign(day=Fs.day.astype(str)); del S, SF; gc.collect()
    LG = ledger(Op, Os, Fs, SC.B0, adm, hit); del Fl, Os, Fs; gc.collect(); C.guard('pipeline paired review ledger')
    R, F = {}, {}                                                                           # readings and the open defects found
    # ---- VALIDITY
    kpf = PQ.kp_fills(kp, day)[1]; chk, nbad = v1_labels(lab, kpf); F['V1 LABELS'] = nbad
    R['V1 LABELS (sampled orders, a per-row loop over the fills)'] = dict(rows_checked=chk, mismatches=nbad, verdict='PASS' if (nbad == 0 and chk > 0) else 'FAIL')
    n2 = v2_identity(lab); F['V2 IDENTITY'] = n2; R['V2 VALUE IDENTITY (every label row)'] = dict(rows=int(len(lab)), failures=n2, verdict='PASS' if n2 == 0 else 'FAIL')
    kt = (np.concatenate([LG[p_].kept_t_min.values[np.isfinite(LG[p_].kept_t_min.values)] for p_ in LG]), np.concatenate([LG[p_].kept_t_max.values[np.isfinite(LG[p_].kept_t_max.values)] for p_ in LG]))
    t1, t2, t3 = v3_timing(ex, lab, kt); F['V3 TIMING'] = t1 + t2 + t3
    R['V3 TIMING'] = dict(export_rows=int(len(ex)), export_rows_off=t1, pi0_steps_off=t2, kept_fills_outside=t3, verdict='PASS' if t1 + t2 + t3 == 0 else 'FAIL')
    st = O.status.astype(str).value_counts().to_dict(); npath = len(Op)
    ok_post = set(zip(ex.oid[(ex.kind == 'post') & ex.status.isin(('eligible', 'refused_post_only'))].astype(int), ex.j[(ex.kind == 'post') & ex.status.isin(('eligible', 'refused_post_only'))].astype(int)))
    ok_ret = set(zip(ex.oid[(ex.kind == 'retain') & (ex.status == 'eligible')].astype(int), ex.j[(ex.kind == 'retain') & (ex.status == 'eligible')].astype(int)))
    lk = lab.kind.astype(str).values; labp = set(zip(lab.oid.values[lk == 'post'], lab.j.values[lk == 'post'])); labr = set(zip(lab.oid.values[lk == 'retain'], lab.j.values[lk == 'retain']))
    pk = prd.kind.astype(str).values; prp = set(zip(prd.oid.values[pk == 'post'], prd.j.values[pk == 'post'])); prr = set(zip(prd.oid.values[pk == 'retain'], prd.j.values[pk == 'retain']))
    G_A, rch_A, adm_A = groups_of(post, ret, 'A'); rch_orders = set(ret.oid.values[rch_A].tolist())
    xpo = ex[ex.kind == 'post']; xo_ = xpo.oid.astype(int).values; xs_ = xpo.status.astype(str).values                       # the export's side of the counts
    two_orig = bool(len(O) == len(xpo) and set(O.oid.astype(int).tolist()) == set(xo_.tolist())); two_path = bool(set(Op.oid.astype(int).tolist()) == set(xo_[np.isin(xs_, PATH_EXPORT)].tolist()))
    xr = ex[ex.kind == 'retain']; fbi = np.flatnonzero(np.isin(xr.status.astype(str).values, NO_FORECAST)); fA = {int(k[1]): int(v) for k, v in hit['A'].items()}
    aA = {int(o) for (_, o) in adm['A']}; fbo, fbj = xr.oid.values[fbi].astype(int), xr.j.values[fbi].astype(int)
    pop = dict(original=int(len(O)), export_post_rows=int(len(xpo)), original_two_sources=two_orig, with_path=npath, with_path_two_sources=two_path, statuses=st,
               export_post_statuses=pd.Series(xs_).value_counts().to_dict(), export_retain_statuses=xr.status.astype(str).value_counts().to_dict(),
               fallback_retain_steps=int(len(fbi)), fallback_steps_on_A_reached_path=int(sum(1 for o, j in zip(fbo, fbj) if o in aA and j < fA.get(o, KP.NSTEP + 1))),
               fallback_steps_at_A_first_review=int(sum(1 for o, j in zip(fbo, fbj) if o in aA and j == KP.J0)),
               no_path=int(len(O) - npath), labelled_post=len(labp), labelled_retain=len(labr), eligible_retain=len(ok_ret),
               admitted={k: len(v) for k, v in adm.items()}, fallbacks={h: int((~np.isfinite(prd[f'{h}_V'].values.astype(float))).sum()) for h in ('A', 'AF', 'AM', 'pi0', 'NEU')},
               refused_nonzero_labels=int((lab[(lk == 'post') & (lab.status.astype(str).values == 'refused_post_only')].pair_v != 0).sum()),
               A_reached_rows=int(rch_A.sum()), A_reached_orders_not_admitted=len(rch_orders - adm_A))
    v4 = (two_orig and two_path and labp == ok_post and labr == ok_ret and prp == labp and prr == labr
          and all(k in adm['ALL'] for r in PQE.RULES for k in adm[r]) and pop['refused_nonzero_labels'] == 0 and pop['A_reached_orders_not_admitted'] == 0)
    F['V4 POPULATIONS'] = 0 if v4 else 1; R['V4 POPULATIONS'] = dict(pop, verdict='PASS' if v4 else 'FAIL')
    acc = {}
    for p_, D in LG.items():
        if p_ in ev['values']:
            for sd, s in (('both', None), ('bid', 0), ('ask', 1)):
                m = np.ones(len(D), bool) if s is None else D.side.values == s; a, b = float(D.V.values[m].sum()), float(ev['values'][p_][sd][B_NAME])
                a += LEDGER_NUDGE if abs(b) >= 100. else 0.; acc[f'{p_} {sd}'] = (a, b, abs(a - b) <= 1e-6)
    lsum = float(lab.pair_v.values[lk == 'post'].sum()); pall = float(LG['PI0-ALL'].V.sum()); acc['POST labels = PI0-ALL'] = (lsum, pall, abs(lsum - pall) <= 1e-6)
    DAll, DA_ = LG['A-ALL'], LG['A']; sel = DA_.admitted.values; cell = (DAll.m.astype(str) + '|' + DAll.side.astype(str) + '|' + DAll.phase.astype(str)).values
    acc['A = the A-ALL ledger on its admissions'] = (float(DA_.V.sum()), float(DAll.V.values[sel].sum()), abs(float(DA_.V.sum()) - float(DAll.V.values[sel].sum())) <= 1e-6)
    drw = null_draws(cell, sel, DAll.V.values, DAll.side.values); nullr = {}
    for i_, (sd, s) in enumerate((('both', None), ('bid', 0), ('ask', 1))):
        m = np.ones(len(DAll), bool) if s is None else DAll.side.values == s; e0, v0 = null_moments(cell[m], sel[m], DAll.V.values[m]); va = float(DA_.V.values[m].sum())
        sd0 = math.sqrt(v0); z = (va - e0) / sd0 if sd0 > 0 else None; d_ = drw[:, i_]
        nullr[sd] = dict(value=va, null_mean=e0, null_sd=sd0, z=z, p_normal_one_sided=.5 * math.erfc(z / math.sqrt(2.)) if z is not None else None, draws=int(len(d_)),
                         draws_mean=float(d_.mean()), draws_sd=float(d_.std(ddof=1)), p_draws_one_sided=float((1 + (d_ >= va).sum()) / (1 + len(d_))), moments_agree=null_agree(e0, v0, d_))
        acc[f'matched null draws = the exact moments ({sd})'] = (nullr[sd]['draws_mean'], e0, nullr[sd]['moments_agree'])
    nA = float(ev['null_A'][B_NAME]); acc['matched null mean (A, both) = the evaluation'] = (nullr['both']['null_mean'], nA, abs(nullr['both']['null_mean'] - nA) <= 1e-6)
    for p_ in ('A', 'PI0'):
        a, b = first_review_share(LG[p_], LG[p_].admitted.values), ev['metrics'][p_]['both']['first_review_of_live']
        acc[f'{p_} first-review share on its admissions = the evaluation'] = (a, b, a is not None and b is not None and abs(a - b) <= 1e-12)
    ins = ev.get('inputs', {}); pf_, jf_ = paired / f'pred_{day}.parquet', bench / f'pred_{day}.parquet'
    for k_, w_ in dict(paired=C.sha(paired / 'paired.json'), predictions=C.sha(pf_) if pf_.exists() else None, bench=C.sha(bench / 'bench.json'),
                       bench_predictions=C.sha(jf_) if jf_.exists() else None, export_receipt=C.sha(export / day / 'receipt.json')).items():
        acc[f'evaluation input {k_}'] = (ins.get(k_, 'missing'), w_, ins.get(k_, 'missing') == w_)
    es = C.sha(Path(PQE.__file__).resolve()); acc['evaluation source = pf_pipeline_paired_eval.py'] = (ev.get('source_sha256'), es, ev.get('source_sha256') == es)
    nacc = sum(1 for x in acc.values() if not x[2]); F['V5 ACCOUNTING'] = nacc
    R['V5 ACCOUNTING (the ledger vs the evaluation at B; the labels vs PI0-ALL)'] = dict(compared=len(acc), mismatches=nacc, verdict='PASS' if (nacc == 0 and len(acc) > 40) else 'FAIL',
                                                                                       first_mismatches={k: v[:2] for k, v in acc.items() if not v[2]})
    R['VALIDITY (all five)'] = 'PASS' if all(R[k]['verdict'] == 'PASS' for k in R if k.startswith('V')) else 'FAIL'
    # ---- PREDICTION
    pred_t = {}
    for h in ('A', 'AF', 'AM'):
        for s, sd in ((0, 'bid'), (1, 'ask')):
            g_, _, _ = groups_of(post[post.side.values == s], ret[ret.side.values == s], h); pred_t[f'{h} {sd}'] = g_
    R['PREDICTION: A old gate groups'] = {k: {g: x.get('old_gate') for g, x in v.items() if 'old_gate' in x} for k, v in pred_t.items() if k.startswith('A ')}
    R['PREDICTION: A plan v2 screen groups (diagnostic)'] = {k: {g: x.get('plan_v2_screen') for g, x in v.items() if 'plan_v2_screen' in x} for k, v in pred_t.items() if k.startswith('A ')}
    # ---- CONTROLLER (the committed evaluation)
    M = ev['metrics']; ctl = {}
    for p_ in ('A', 'A-PA', 'PI0', 'NEU', 'E1', 'PA'):
        for sd in ('both', 'bid', 'ask'):
            y = M[p_][sd]; ctl[f'{p_} {sd}'] = dict(value=y['value'], per_opportunity=y['per_original_opportunity'], vs_PI0=y['vs_PI0'], vs_same_admission_PULL_ALL=y['vs_same_admission_PULL_ALL'],
                                                    resting=y['resting'], at_best=y['at_best'], lifetime=y['lifetime_p10_p50_p90'], first_review_of_live=y['first_review_of_live'],
                                                    cancels_per_100_filled_shares=y['cancels_per_100_filled_shares'], attempts=y['attempts'], accepted=y['accepted'], filled=y['filled'])
        ctl[f'{p_} pairs'] = M[p_]['pairs']
    vals = ev['values']; scen = list(vals['A']['both'])
    R['CONTROLLER: A above N0 / PI0 / NEU / E1 / A-PA at B (both sides)'] = [vals['A']['both'][B_NAME] > 0] + [vals['A']['both'][B_NAME] > vals[x]['both'][B_NAME] for x in ('PI0', 'NEU', 'E1', 'A-PA')]
    R['CONTROLLER: scenarios with A above 0 / above PI0'] = [sum(vals['A']['both'][nm] > 0 for nm in scen), sum(vals['A']['both'][nm] > vals['PI0']['both'][nm] for nm in scen), len(scen)]
    R['CONTROLLER: evaluation readings (carried)'] = {k: v for k, v in ev['readings'].items() if k.startswith(('ECONOMIC GATE', 'MATCHED', 'PRACTICAL'))}
    nest = nested(DA_, LG['A-PA'], sel)
    R["CONTROLLER: A vs A-PA nested on every admitted order (at least A-PA's shares, the same fills before the first effect)"] = nest
    R['CONTROLLER: the practical predicates vs A-PA'] = 'IMPLIED-BY-NESTING (not evidence)' if nest else 'NOT-NESTED'
    R['CONTROLLER: MATCHED CONTROL WITH ITS SPREAD (draws, one-sided p <= 0.05)'] = {sd: spread_verdict(x['p_draws_one_sided']) for sd, x in nullr.items()}
    qt = {}
    for p_ in ('A', 'PI0', 'NEU', 'E1', 'A-PA'):
        y = M[p_]['both']; qt[p_] = dict(resting=y['resting'], at_best=y['at_best'], at_best_both=M[p_]['pairs'].get('at_best_both'), median_lifetime=(y['lifetime_p10_p50_p90'] or [None] * 3)[1],
                                          first_review_of_live=y['first_review_of_live'], cancels_per_100_filled_shares=y['cancels_per_100_filled_shares'], attempts=y['attempts'], filled=y['filled'])
    gt = lambda a, b: None if a is None or b is None else bool(a > b)
    R['CONTROLLER: quoting, A vs PI0 / NEU / E1 (not nested): more resting / more at-best both sides / longer median life / fewer first-review pulls / fewer cancels per filled share'] = {
        x: [gt(qt['A']['resting'], qt[x]['resting']), gt(qt['A']['at_best_both'], qt[x]['at_best_both']), gt(qt['A']['median_lifetime'], qt[x]['median_lifetime']),
            gt(qt[x]['first_review_of_live'], qt['A']['first_review_of_live']), gt(qt[x]['cancels_per_100_filled_shares'], qt['A']['cancels_per_100_filled_shares'])] for x in ('PI0', 'NEU', 'E1')}
    R['CONTROLLER: pulled at the first review (share of the orders live then)'] = {
        'PI0, every eligible opportunity': first_review_share(LG['PI0-ALL'], np.ones(len(DAll), bool)), 'PI0, its admissions': first_review_share(LG['PI0'], LG['PI0'].admitted.values),
        'A, every eligible opportunity': first_review_share(DAll, np.ones(len(DAll), bool)), 'A, its admissions': first_review_share(DA_, sel)}
    fc = {p_: {sd: c1_c3_c12(LG[p_] if s is None else LG[p_][LG[p_].side.values == s]) for sd, s in (('both', None), ('bid', 0), ('ask', 1))}
          for p_ in ('A', 'A-PA', 'PI0', 'PI0-PA', 'NEU', 'E1', 'PA')}
    R['FINDINGS C1 (A, both): value / without fills on orders < 100 ms old / B1a-weighted sign'] = [float(DA_.V.sum()), fc['A']['both']['C1']['without_young'], fc['A']['both']['C1']['b1a_sign']]
    R['FINDINGS C1: A minus A-PA by fill age (both)'] = {k: fc['A']['both']['C1']['by_age'][k] - fc['A-PA']['both']['C1']['by_age'][k] for k in SC.AGE_LABELS}
    # ---- RISK
    tert = np.nanquantile(Op.sigma_post.values.astype(float), (1 / 3, 2 / 3)); risk = {}
    for p_ in ('A', 'A-PA', 'PI0', 'NEU', 'E1', 'PA', 'PI0-ALL'):
        D = LG[p_]; risk[p_] = {}
        for sd, s in (('both', None), ('bid', 0), ('ask', 1)):
            m = np.ones(len(D), bool) if s is None else D.side.values == s
            risk[p_][sd] = tails(D.V.values[m], D.q.values[m] > 0)
        risk[p_]['where'] = where_risk(D, tert)
    R['RISK: the window carrying most of A\'s severe-episode loss'] = min(risk['A']['where']['window (severe episodes)'].items(), key=lambda kv: kv[1])[0] if risk['A']['both']['severe_episodes'] else 'NO SEVERE EPISODE'
    R['RISK: A movement component per filled share (c)'] = risk['A']['where']['inventory']['movement_per_filled_share']
    R['OPEN DEFECTS (validity)'] = {k: v for k, v in F.items() if v}
    # ---- outputs
    rows = []
    for p_, x in risk.items():
        for sd in ('both', 'bid', 'ask'):
            rows.append(dict(policy=p_, side=sd, **{k: v for k, v in x[sd].items()}))
    L = [f'THE FOUR-PART REVIEW OF THE PAIRED RUN (pf_pipeline_paired_review.py; {day}; restricted experiment: POST vs NO_ENTRY to END; c at END, not P&L)', '', 'VALIDITY']
    L += [f'  {k}: {v}' for k, v in R.items() if k.startswith('V')]
    L += ['', 'PREDICTION (per head and side; the old gate verdict stands, the plan v2 screen is a diagnostic)']
    for k, g_ in pred_t.items():
        for gname, x in g_.items():
            if x['rows']:
                L.append(f"  {k:<9} {gname:<28} rows {x['rows']:>8,} episodes {x['episodes']:>6,} nonzero {x['nonzero_label_episodes']:>6,}"
                         + (f" | mean pred {x['mean_pred']:+.3f} label {x['mean_label']:+.3f} | episode-balanced {x['eb_pred']:+.3f} / {x['eb_label']:+.3f} | MAE {x['mae']:.3f} | sums {x['sum_pred']:+,.0f} / "
                            f"{x['sum_label']:+,.0f} | old gate {x['old_gate']} | plan v2 screen {x['plan_v2_screen']}" if 'mean_pred' in x else ''))
    L += ['', 'CONTROLLER (the committed evaluation v2, at B)'] + [f'  {k}: {v}' for k, v in ctl.items()] + [f'  {k}: {v}' for k, v in R.items() if k.startswith('CONTROLLER')]
    L += ['', 'RISK (the ledger at B)']
    for p_, x in risk.items():
        for sd in ('both', 'bid', 'ask'):
            L.append(f'  {p_:<7} {sd:<4} {x[sd]}')
        L.append(f"  {p_:<7} where: {x['where']}")
    L += [f'  {k}: {v}' for k, v in R.items() if k.startswith('RISK')]
    L += ['', 'MATCHED NULL OF A (exact moments and 1,000 seeded draws; at B)'] + [f'  {sd}: {x}' for sd, x in nullr.items()]
    L += ['', 'QUOTING (the committed evaluation, both sides)'] + [f'  {p_:<5} {x}' for p_, x in qt.items()]
    L += ['', 'FINDINGS C1 / C3 / C12 (the ledger at B)'] + [f'  {p_:<6} {sd:<4} {x}' for p_, y in fc.items() for sd, x in y.items()]
    L += [f'  {k}: {v}' for k, v in R.items() if k.startswith('FINDINGS')] + ['', f"OPEN DEFECTS (validity): {R['OPEN DEFECTS (validity)'] or 'none'}"]
    L += ['', 'LIMITS: consumed day; single-order episodes; ledger at B only; movement from the posting mid to END (no mid per fill); restricted experiment.']
    out.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(out / f'review_{day}.csv', index=False); (out / f'review_{day}.txt').write_text('\n'.join(L) + '\n')
    pd.DataFrame(drw, columns=['both', 'bid', 'ask']).to_csv(out / f'review_null_draws_{day}.csv', index=False)
    pd.DataFrame([dict(policy=p_, side=sd, **{f'C1 {k}': v for k, v in x['C1']['by_age'].items()}, **{'C1 without young': x['C1']['without_young'], 'C1 B1a sign': x['C1']['b1a_sign']},
                       **{f"C1 B1a {k}": v for k, v in x['C1']['b1a_weighted'].items()}, **{f'C3 {k}': v for k, v in x['C3'].items()}, **{f'C12 {k}': v for k, v in x['C12'].items()})
                  for p_, y in fc.items() for sd, x in y.items()]).to_csv(out / f'review_findings_{day}.csv', index=False)
    led = pd.concat([D.assign(policy=p_) for p_, D in LG.items()], ignore_index=True)
    led.drop(columns=['kept_t_min', 'kept_t_max']).to_parquet(out / f'ledger_{day}.parquet', index=False)
    C.dump(out / f'review_{day}.json', dict(asof=C.stamp(), day=day, readings=R, prediction=pred_t, controller=ctl, risk=risk, sigma_terciles=[float(x) for x in tert],
                                            matched_null=nullr, quoting=qt, findings_c1_c3_c12=fc, accounting={k: [v[0], v[1], bool(v[2])] for k, v in acc.items()},
                                            inputs=dict(paired=C.sha(paired / 'paired.json'), evaluation=C.sha(evald / f'eval_{day}.json') if (evald / f'eval_{day}.json').exists() else None,
                                                        parts=hashes, export_receipt=C.sha(export / day / 'receipt.json'), days=rec),
                                            outputs={x: C.sha(out / x) for x in (f'review_{day}.txt', f'review_{day}.csv', f'review_null_draws_{day}.csv', f'review_findings_{day}.csv',
                                                                                 f'ledger_{day}.parquet')},
                                            source_sha256=C.sha(Path(__file__).resolve()), provenance=KP.provenance(), seconds=time.time() - start))
    print('\n'.join(L), flush=True)
    return R, LG, pred_t, risk


REACH_LEAK = False                                         # planted-error switch (selftest only): A's reached rows without its admissions


def selftest():  # noqa: C901
    """Synthetic tape only (see the module's SELFTEST)."""
    global LAND_SHIFT, REACH_LEAK, LEDGER_NUDGE
    bad = []
    tv = tails(np.r_[-50., -10., 0., 0., 20.], np.r_[True, True, False, False, True])          # the tails by hand
    want = dict(mean_loss=12., p_loss=.4, p_severe=.2, mean_severe_excess=4., total_loss=60., severe_episodes=1, worst_1pct_share_of_loss=50. / 60., largest_loss=50.,
                p95_loss_filled=float(np.quantile([50., 10., 0.], .95)))
    if any(abs(tv[k] - v) > 1e-12 for k, v in want.items()):
        bad.append(f'the tails on planted values: { {k: tv[k] for k in want} }')
    tw = tails(np.r_[np.full(100, -1.), -100.], np.ones(101, bool))['worst_1pct_share_of_loss']   # 101 episodes: the worst 2 (1%) carry 101 of 200
    if abs(tw - 101. / 200.) > 1e-12:
        bad.append(f'the worst 1% share on 101 planted episodes: {tw} (want 0.505)')
    o_ = np.r_[1, 1, 1, 2]; pr_ = np.r_[1., 1., 1., 9.]; lb_ = np.r_[2., 2., 2., 2.]           # episode-balanced: (1 + 9) / 2 = 5, not (3 + 9) / 4 = 3
    if not (abs(episode_mean(o_, pr_) - 5.) < 1e-12 and screen(np.arange(300), np.full(300, 2.04), np.full(300, 2.), True)[0] == 'PASS'
            and screen(np.arange(300), np.full(300, 3.), np.full(300, 2.), True)[0] == 'FAIL' and screen(np.arange(10), np.ones(10), np.ones(10), True)[0] == 'INSUFFICIENT_SUPPORT'
            and screen(np.arange(300), np.ones(300), np.zeros(300), True)[0] == 'NOT_EVALUABLE' and screen(np.arange(300), np.full(300, -1.), np.full(300, -1.), True)[0] == 'FAIL'):
        bad.append('the episode-balanced mean or the plan v2 screen on planted values')
    P_ = pd.DataFrame(dict(oid=[1, 2, 3, 4], A_V=[1., -1., 0., np.nan], pair_v=[1., -1., 0., 0.]))
    R_ = pd.DataFrame(dict(oid=[1, 1, 2, 2], j=[2, 3, 2, 3], A_V=[1., -1., -1., 1.], pair_v=[.5, -.5, -1., 1.]))
    g_, rch_, adm_ = groups_of(P_, R_, 'A')
    if not (g_['POST (pred > 0)']['rows'] == 1 and g_['NO_ENTRY (pred < 0)']['rows'] == 1 and g_['POST tie (pred = 0)']['rows'] == 1 and g_['POST fallback (nonfinite)']['rows'] == 1
            and rch_.tolist() == [True, True, False, False] and g_['PULL reached (pred < 0)']['rows'] == 1 and g_['PULL neutral (pred < 0)']['rows'] == 2 and adm_ == {1}):
        bad.append(f'the prediction groups on planted values: { {k: v["rows"] for k, v in g_.items()} } / reached {rch_.tolist()}')
    lab_ = pd.DataFrame(dict(side=[0, 1], pf_mid=[50.5, 50.], p=[50., 50.5], pair_q=[2., 2.], pair_v=[3., 3.], pair_qm=[2., 2.]))   # 0.5*2 + 2 = 3 for both
    lab_bad = lab_.copy(); lab_bad.loc[1, 'pair_qm'] = 2.5; lab_nan = lab_.copy(); lab_nan.loc[0, 'pair_qm'] = np.nan
    if not (v2_identity(lab_) == 0 and v2_identity(lab_bad) == 1 and v2_identity(lab_nan) == 1):
        bad.append(f'the value identity on planted rows (clean / broken / NaN): {v2_identity(lab_)} / {v2_identity(lab_bad)} / {v2_identity(lab_nan)}')
    l1_ = pd.DataFrame(dict(oid=[0, 0], kind=['retain', 'post'], j=[2, 0], status=['eligible', 'eligible'], pi0_V=[1., np.nan], pi0_ok=[True, True], pair_v=[0., 0.], pair_q=[0., 0.]))
    f1_ = pd.DataFrame(dict(oid=np.zeros(0, int), t=np.zeros(0), v=np.zeros(0), q=np.zeros(0))); l1n = l1_.copy(); l1n.loc[0, 'pair_v'] = np.nan
    if not (v1_labels(l1_, f1_) == (2, 0) and v1_labels(l1n, f1_) == (2, 1)):
        bad.append(f'V1 on planted rows (clean / a NaN label): {v1_labels(l1_, f1_)} / {v1_labels(l1n, f1_)}')
    ce, se, ve = np.r_[0, 0, 0, 1, 1, 2], np.r_[True, False, False, True, False, True], np.r_[1., 2., 6., 3., 5., 4.]   # the matched null by enumeration
    tots = [sum(ve[list(c)]) for c in itertools.product(*[itertools.combinations(np.flatnonzero(ce == u), int(se[ce == u].sum())) for u in np.unique(ce)])]
    e_, v_ = null_moments(ce, se, ve)
    if not (abs(e_ - float(np.mean(tots))) < 1e-12 and abs(v_ - float(np.var(tots))) < 1e-12 and abs(e_ - 11.) < 1e-12 and abs(v_ - 17. / 3.) < 1e-12):
        bad.append(f'the matched null moments vs enumeration: {e_}, {v_} vs {np.mean(tots)}, {np.var(tots)} (want 11, 17/3)')
    rg = np.random.default_rng(3); cb = np.repeat(np.arange(60), 25); vb = rg.gamma(1.5, 4., len(cb)) * rg.choice([-1., 1.], len(cb)); sb = np.zeros(len(cb), bool); sdb = cb % 2
    for u in range(60):
        sb[np.flatnonzero(cb == u)[:3 + u % 7]] = True
    eb, vb0 = null_moments(cb, sb, vb); db = null_draws(cb, sb, vb, sdb, draws=400, seed=5)
    top = np.zeros(len(cb), bool)
    for u in range(60):
        ix_ = np.flatnonzero(cb == u); top[ix_[np.argsort(-vb[ix_])][:int(sb[ix_].sum())]] = True
    ok_ = (null_agree(eb, vb0, db[:, 0]) and not null_agree(eb, 2. * vb0, db[:, 0]) and not null_agree(eb + 4. * np.sqrt(vb0), vb0, db[:, 0])
           and abs(db[:, 1] + db[:, 2] - db[:, 0]).max() < 1e-9 and (db[:, 0] >= float(vb[top].sum())).sum() == 0 and int(top.sum()) == int(sb.sum()))
    if not ok_:
        bad.append(f'the matched null draws: agree {null_agree(eb, vb0, db[:, 0])}, mean {db[:, 0].mean():.3f} vs {eb:.3f}, variance {db[:, 0].var(ddof=1):.3f} vs {vb0:.3f}')
    if not (spread_verdict(.05) == 'PASS' and spread_verdict(.0501) == 'FAIL' and spread_verdict(1. / 1001.) == 'PASS'):
        bad.append('the spread verdict at its declared 0.05')
    Dn = pd.DataFrame(dict(live0=[True, True, True, False, True], act=[True, True, False, True, True], T=[KP.J0, KP.J0 + 3, KP.J0, KP.J0, KP.J0]))
    if not (first_review_share(Dn, np.ones(5, bool)) == .5 and first_review_share(Dn, np.r_[True, False, False, False, False]) == 1. and first_review_share(Dn, np.zeros(5, bool)) is None):
        bad.append('the first-review share on planted orders')
    DA0 = pd.DataFrame(dict(q=[3., 0., 2.], v_w0=[1., 0., 0.], v_w1=[0., 0., 1.])); DP0 = pd.DataFrame(dict(q=[1., 0., 2.], v_w0=[1., 0., 0.], v_w1=[0., 0., 1.]))
    if not (nested(DA0, DP0, np.ones(3, bool)) and not nested(DP0, DA0, np.ones(3, bool)) and not nested(DA0, DP0.assign(v_w1=[0., 0., 2.]), np.ones(3, bool))
            and not nested(DA0, DP0, np.zeros(3, bool))):
        bad.append('the nesting predicate on planted orders')
    Dc = pd.DataFrame(dict(V=[10., -6.], v_a0=[4., -8.], v_a1=[1., 1.], v_a2=[5., 1.], pend_v=[0., -2.], pend_n=[0., 1.], gross=[12., 10.],
                           **{f'v_s{i}': [10. if i == 0 else 0., -6. if i == 1 else 0.] for i in range(len(SC.SRC_LABELS))}, v_s_unknown=[0., 0.]))
    cx = c1_c3_c12(Dc); import pf_fv3_check as FV_
    if not (cx['C1']['by_age'] == dict(zip(SC.AGE_LABELS, [-4., 2., 6.])) and cx['C1']['without_young'] == 8. and abs(cx['C3']['share_of_gross'] - 2. / 22.) < 1e-12
            and cx['C3']['fills'] == 1 and cx['C12'][SC.SRC_LABELS[0]] == 10. and cx['C12'][SC.SRC_LABELS[1]] == -6.
            and all(abs(cx['C1']['b1a_weighted'][k] - (-4. * w[0] + 2. * w[1] + 6. * w[2])) < 1e-12 for k, w in FV_.B1A.items())):
        bad.append(f'FINDINGS C1 / C3 / C12 on planted orders: {cx}')
    dd = str(KP.DAYS[0])                                      # the ledger on planted fill times (independent of what the synthetic tape happens to contain)
    Opu = pd.DataFrame(dict(day=dd, oid=[1, 2, 3], side=[0, 1, 0], mark=[52., 49., 50.], p=[50., 50., 50.], g_struct=np.nan, pf_mid_post=[50.5, 49.5, 50.5],
                            phase=[1, 2, 3], spread=[1., 2., 3.], sigma_post=[.1, .2, .3], m=[7, 7, 8]))
    Osu = pd.DataFrame(dict(day=dd, oid=[1, 2, 3], refused=[False, False, True], v=0., q=0., t_full=np.nan, land=np.nan))
    Fsu = pd.DataFrame(dict(day=dd, oid=[1, 1, 1, 1, 1, 1, 2], t=[70., 120., 300., 610., 700., 2000., 3000.], q=[1., 2., 1., 3., 4., 2., 1.], src=[0, 1, 2, 3, 0, 1, 2]))
    admu = {'ALL': {(dd, 1), (dd, 2), (dd, 3)}, **{r: set() for r in PQE.RULES}}; admu['A'] = {(dd, 1)}; hitu = {r: {} for r in PQE.RULES}; hitu['A'] = {(dd, 1): 12}
    LGu = ledger(Opu, Osu, Fsu, SC.B0, admu, hitu); a1, p1 = LGu['A'].iloc[0], LGu['A-PA'].iloc[0]     # A pulls at step 12: request 600, landing 637 ms
    want_a = dict(V=14., q=7., v_w0=2., v_w1=4., v_w2=8., v_a0=6., v_a1=2., v_a2=6., pend_v=6., pend_n=1., v_s0=2., v_s1=4., v_s2=2., v_s3=6., spread_comp=3.5, T=12.)
    want_p = dict(V=6., q=3., v_w0=2., v_w1=4., v_w2=0., pend_v=4., pend_n=1.)                 # A-PA: request 100, landing 137 ms
    if not (all(abs(float(a1[k]) - v) < 1e-12 for k, v in want_a.items()) and all(abs(float(p1[k]) - v) < 1e-12 for k, v in want_p.items())
            and float(LGu['A'].V.iloc[1]) == 0. and abs(float(LGu['A-ALL'].V.iloc[1]) - 1.) < 1e-12 and float(LGu['A-ALL'].V.iloc[2]) == 0. and bool(a1.act) and bool(a1.live0)
            and abs(float(a1.tint) - (1 * 4930. + 2 * 4880. + 1 * 4700. + 3 * 4390.) / 1e3) < 1e-12):
        bad.append(f"the ledger on planted fill times: A {a1[list(want_a)].to_dict()}, A-PA {p1[list(want_p)].to_dict()}, second order A / A-ALL "
                   f"{LGu['A'].V.iloc[1]} / {LGu['A-ALL'].V.iloc[1]}, tint {a1.tint}")
    env = lambda d_: dict(h=600., q01=-900., q99=1050., inputs={}); syn_set = lambda d_: (pd.DataFrame(dict(t_set_ms=np.zeros(0), set_status=np.zeros(0, int))), {})
    src = KP.Syn(); days = list(KP.DAYS); d0 = days[0]
    with tempfile.TemporaryDirectory(dir=str(I.K.IO)) as tmp:
        tmp = Path(tmp)
        with QL._v1_on(src, tmp):
            for d in days:
                QP.day_run(d)
        tim = SC.SynTiming(src, contra=((108, 0, 150500.),))
        with contextlib.redirect_stdout(io.StringIO()):
            for d in days:
                KP.day_run(d, src=src, v1=tmp / 'v1', out=tmp / 'kp')
            SC.day_run(d0, src=src, timing=tim, kp=tmp / 'kp', out=tmp / 'sc', env_of=env, set_of=syn_set, st_lag=150.)
        O = pd.read_parquet(tmp / 'kp' / d0 / 'orders.parquet'); el = O[O.status == 'eligible']; Opx = O[O.status.isin(('eligible', 'refused_post_only'))]
        endj = SC.A.endj_of(el.t_full.values, el.g_struct.values); rng = np.random.default_rng(9); rows = []
        fl = PQ.kp_fills(tmp / 'kp', d0)[1]; fg = {k: v for k, v in fl.groupby('oid')}
        for o, e in zip(el.itertuples(), endj):                                              # planted forecasts; labels from the keep / pull run's own fills
            js = np.arange(KP.J0, max(int(e), KP.J0)); oid = int(o.oid); f = fg.get(oid, fl.iloc[:0]); sg = 1. if o.side == 0 else -1.
            for i_, j in enumerate(js):
                mk = f.t.values >= KP.landing(int(j)); q_, v_ = float(f.q.values[mk].sum()), float(f.v.values[mk].sum()); mid = float(o.p) + sg * (.5 + .01 * i_)
                vals = {r: (-1. if ((oid + n_) % 3 == 0 and i_ >= (oid % max(len(js), 1))) else 1.) for n_, r in enumerate(PQE.RULES)}
                rows.append(dict(day=d0, oid=oid, kind='retain', j=int(j), side=int(o.side), status='eligible', p=float(o.p), pf_mid=mid, t=float(o.t), decision_ms=float(o.t) + KP.STEP * j,
                                 effective_ms=float(o.t) + KP.landing(int(j)), endpoint_ms=float(o.t) + KP.H, sigma=float(rng.normal()), value_m50=v_, q_m50=q_, qm_m50=v_ - sg * (mid - float(o.p)) * q_,
                                 pi0_V=vals['PI0'], pi0_ok=True, NEU_V=vals['NEU'], NEU_ok=True, A_V=vals['A'], AF_V=vals['AF'], AM_V=vals['AM'], J_V=vals['E1'], J_ok=True))
        for o in Opx.itertuples():
            oid = int(o.oid); f = fg.get(oid, fl.iloc[:0]); sg = 1. if o.side == 0 else -1.; ref_ = o.status == 'refused_post_only'
            q_, v_ = (0., 0.) if ref_ else (float(f.q.sum()), float(f.v.sum())); a = 1. if oid % 2 == 0 else -1.
            rows.append(dict(day=d0, oid=oid, kind='post', j=0, side=int(o.side), status=str(o.status), p=float(o.p), pf_mid=float(o.p) + .5 * sg, t=float(o.t), decision_ms=float(o.t),
                             effective_ms=float(o.t) + KP.PLACE, endpoint_ms=float(o.t) + KP.H, sigma=float(rng.normal()), value_m50=v_, q_m50=q_, qm_m50=v_ - .5 * q_,
                             pi0_V=1. if oid % 3 else -1., pi0_ok=True, NEU_V=-a, NEU_ok=True, A_V=a, AF_V=1., AM_V=-1., J_V=a, J_ok=oid % 5 != 0))
        X = pd.DataFrame(rows); r_ = X[X.kind == 'retain'].reset_index(drop=True); p_ = X[X.kind == 'post'].reset_index(drop=True)
        lr, ts, first = PQ.label_retain(r_); r_ = r_.assign(**lr, tstar=ts); lp, ff = PQ.label_post(p_, first, r_); p_ = p_.assign(**lp, tstar=ff)
        lab = pd.concat([r_, p_], ignore_index=True)
        XC = ['oid', 'kind', 'j', 'status', 't', 'decision_ms', 'effective_ms', 'endpoint_ms', 'side', 'pf_mid', 'p', 'sigma']; onp = O[~O.status.isin(('eligible', 'refused_post_only'))]
        npx = pd.DataFrame(dict(oid=onp.oid.astype(int).values, kind='post', j=0, status=onp.status.astype(str).values, t=onp.t.astype(float).values, decision_ms=onp.t.astype(float).values,
                                effective_ms=onp.t.astype(float).values + KP.PLACE, endpoint_ms=onp.t.astype(float).values + KP.H, side=onp.side.astype(int).values, pf_mid=np.nan,
                                p=onp.p.astype(float).values, sigma=np.nan))                      # the no-path orders' POST rows (no labels), as the export writes them
        exp = tmp / 'exp' / d0; exp.mkdir(parents=True); pd.concat([lab[XC], npx[XC]], ignore_index=True).assign(o_depth=0.).to_parquet(exp / 'rows_000.parquet', index=False)
        C.dump(exp / 'receipt.json', dict(outputs={'rows_000.parquet': C.sha(exp / 'rows_000.parquet')}))
        pq_ = tmp / 'pq'; pq_.mkdir(); C.dump(pq_ / 'paired.json', dict(outputs={}, readings={'PREDICTION GATE (A, per day and side)': {d0: {'0': 'FAIL', '1': 'FAIL'}}}))
        bj = tmp / 'bench'; bj.mkdir(); C.dump(bj / 'bench.json', dict(outputs={}))
        jp = lab[['day', 'oid', 'kind', 'j', 'J_V', 'J_ok']].copy(); pr = lab[['day', 'oid', 'kind', 'j', 'side', 'status', 'pi0_V', 'pi0_ok', 'NEU_V', 'NEU_ok', 'A_V', 'AF_V', 'AM_V', 'pair_v', 'value_m50']].copy()
        with contextlib.redirect_stdout(io.StringIO()):
            PQE.run(d0, out=tmp / 'ev', sc_out=tmp / 'sc', kp=tmp / 'kp', paired=pq_, bench=bj, export=tmp / 'exp', pred=pr.copy(), jpred=jp.copy())
        labs = lab[['day', 'oid', 'kind', 'j', 'side', 'status', 'p', 'pf_mid', 'value_m50', 'q_m50', 'qm_m50', 'pi0_V', 'pi0_ok', 'tstar', 'pair_v', 'pair_q', 'pair_qm']]
        def go(out_, **kw):
            with contextlib.redirect_stdout(io.StringIO()):
                return run(d0, out=out_, sc_out=tmp / 'sc', kp=tmp / 'kp', paired=pq_, evald=kw.pop('evald', tmp / 'ev'), bench=bj, export=kw.pop('export', tmp / 'exp'),
                           labels=kw.pop('labels', labs.copy()), pred=kw.pop('pred', pr).drop(columns=['value_m50']).copy(), jpred=jp.copy())
        R = None
        try:
            R, LG, pt, risk = go(tmp / 'r')
        except (SystemExit, ValueError) as e:
            bad.append(f'the synthetic review refused: {str(e)[:400]}')
        if R is not None:
            vs = [R[k]['verdict'] for k in R if k.startswith('V') and isinstance(R[k], dict)]
            if not (R['VALIDITY (all five)'] == 'PASS' and len(vs) == 5):
                bad.append(f"the clean synthetic run: validity {[(k, R[k]['verdict']) for k in R if k.startswith('V') and isinstance(R[k], dict)]}")
            if R['CONTROLLER: the practical predicates vs A-PA'] != 'IMPLIED-BY-NESTING (not evidence)':
                bad.append(f"the clean synthetic run: A vs A-PA not nested ({R['CONTROLLER: the practical predicates vs A-PA']}); A's pulls cannot land before the first review's")
            led = pd.read_parquet(tmp / 'r' / f'ledger_{d0}.parquet'); DA = led[led.policy == 'A']
            Ofl = V.load([d0], tmp / 'sc', tmp / 'kp'); Opl = Ofl[4].assign(day=Ofl[4].day.astype(str)); Os0, Fs0 = V.scen_inputs(SC.B0, Opl, Ofl[1], Ofl[2], Ofl[3])
            ref0 = Os0.set_index(Os0.oid.astype(int)); mo = Opl.set_index(Opl.oid.astype(int)); ad = {int(o) for (_, o) in PQE.admissions(pr[pr.kind == 'post'], 'A')}
            hA = {int(k[1]): int(v) for k, v in PQE.triggers(pr[pr.kind == 'retain'], 'A').items()}; en0 = dict(zip(ref0.index, SC.endj_n(ref0.t_full.values, mo.g_struct.reindex(ref0.index).values, SC.B0['N'])))
            w0_, sp_, ag_, pv_, sr_ = 0., 0., np.zeros(len(SC.AGE_LABELS)), 0., np.zeros(len(SC.SRC_LABELS))   # A by a per-fill loop: window 0, spread, age, pending, source
            pfm = lab[lab.kind == 'post'].set_index(lab[lab.kind == 'post'].oid.astype(int)).pf_mid
            for f in Fs0.itertuples():
                o = int(f.oid)
                if o not in ad or bool(ref0.refused.loc[o]):
                    continue
                T = hA.get(o); ex_ = T is not None and KP.J0 <= T <= KP.NSTEP and T < en0[o]; gone = ex_ and float(f.t) >= KP.STEP * T + SC.B0['L']
                if gone:
                    continue
                sg = 1. if mo.side.loc[o] == 0 else -1.; v = float(f.q) * sg * (float(mo.mark.loc[o]) - float(mo.p.loc[o])); age_ = float(f.t) - float(SC.B0['A'])
                w0_ += v if float(f.t) < KP.STEP * KP.J0 else 0.; sp_ += sg * (float(pfm.loc[o]) - float(mo.p.loc[o])) * float(f.q)
                ag_[0 if age_ < KP.YOUNG else (1 if age_ < 500. else 2)] += v; pv_ += v if (ex_ and KP.STEP * T <= float(f.t)) else 0.; sr_[int(f.src)] += v
            if not (abs(w0_ - DA.v_w0.sum()) < 1e-6 and abs(sp_ - DA.spread_comp.sum()) < 1e-6 and np.allclose(ag_, [DA[f'v_a{b}'].sum() for b in range(len(SC.AGE_LABELS))], rtol=0, atol=1e-6)
                    and abs(pv_ - DA.pend_v.sum()) < 1e-6 and np.allclose(sr_, [DA[f'v_s{i}'].sum() for i in range(len(SC.SRC_LABELS))], rtol=0, atol=1e-6)):
                bad.append(f'the ledger by a per-fill loop: first window {DA.v_w0.sum():,.3f} vs {w0_:,.3f}, spread component {DA.spread_comp.sum():,.3f} vs {sp_:,.3f}, '
                           f'age {[round(DA[f"v_a{b}"].sum(), 3) for b in range(3)]} vs {ag_.round(3).tolist()}, pending {DA.pend_v.sum():,.3f} vs {pv_:,.3f}, '
                           f'sources {[round(DA[f"v_s{i}"].sum(), 3) for i in range(len(SC.SRC_LABELS))]} vs {sr_.round(3).tolist()}')
            if not (len(ag_) == 3 and abs(ag_[0]) > 0 and abs(pv_) > 0 and abs(ag_[1]) + abs(ag_[2]) > 0):
                bad.append(f'the per-fill loop has no young / pending / older fill to compare (age {ag_.tolist()}, pending {pv_})')
            for k_, sd in (('both', None),):
                w_ = sum(DA[f'v_w{w}'].sum() for w in range(3)); sp = DA.spread_comp.sum() + DA.move_comp.sum()
                if not (abs(w_ - DA.V.sum()) < 1e-6 and abs(sp - DA.V.sum()) < 1e-6 and abs(risk['A']['both']['total_loss'] - np.maximum(-DA.V.values, 0.).sum()) < 1e-6
                        and abs(sum(x['total_loss'] for x in risk['A']['where']['phase'].values()) - risk['A']['both']['total_loss']) < 1e-6):
                    bad.append('the ledger windows / components / strata do not add up to the totals')
        def flips(name, which, **kw):
            o_ = tmp / ('z' + str(abs(hash(name)) % 100000))
            if 'export' in kw and 'evald' not in kw:                                       # the evaluation as if it had read this export (V5 checks the provenance)
                j_ = json.loads((tmp / 'ev' / f'eval_{d0}.json').read_text()); j_['inputs']['export_receipt'] = C.sha(kw['export'] / d0 / 'receipt.json')
                ed = tmp / ('x' + str(abs(hash(name)) % 100000)); ed.mkdir(); C.dump(ed / f'eval_{d0}.json', j_); kw['evald'] = ed
            try:
                Rz = go(o_, **kw)[0]
                got = {k: Rz[k]['verdict'] for k in Rz if k.startswith('V') and isinstance(Rz[k], dict)}
                if not (got[which] == 'FAIL' and all(v == 'PASS' for k, v in got.items() if k != which)):
                    bad.append(f'{name}: validity verdicts {got} (only {which} should FAIL)')
            except (SystemExit, ValueError) as e:
                bad.append(f'{name}: the review refused instead of reporting: {str(e)[:200]}')
        l1 = labs.copy(); i1 = l1.index[(l1.oid.astype(int) % SAMPLE == 0) & (l1.kind == 'retain')][0]; l1.loc[i1, 'pair_v'] += 1.; l1.loc[i1, 'pair_qm'] += 1.
        flips('a sampled label off by 1 c', 'V1 LABELS (sampled orders, a per-row loop over the fills)', labels=l1)
        lq = labs.copy(); iq = lq.index[(lq.oid.astype(int) % SAMPLE == 0) & (lq.kind == 'retain')][0]; sgq = 1. if lq.loc[iq, 'side'] == 0 else -1.
        lq.loc[iq, 'pair_q'] += 1.; lq.loc[iq, 'pair_qm'] -= sgq * (float(lq.loc[iq, 'pf_mid']) - float(lq.loc[iq, 'p']))          # the identity kept: only the quantity is wrong
        flips('a sampled paired quantity off by 1 share', 'V1 LABELS (sampled orders, a per-row loop over the fills)', labels=lq)
        l2 = labs.copy(); l2.loc[l2.index[(l2.oid.astype(int) % SAMPLE != 0)][0], 'pair_qm'] += 1.
        flips('a broken value identity', 'V2 VALUE IDENTITY (every label row)', labels=l2)
        e3 = tmp / 'exp3' / d0; e3.mkdir(parents=True); g3 = pd.read_parquet(exp / 'rows_000.parquet'); g3.loc[g3.index[5], 'effective_ms'] += 1.; g3.to_parquet(e3 / 'rows_000.parquet', index=False)
        C.dump(e3 / 'receipt.json', dict(outputs={'rows_000.parquet': C.sha(e3 / 'rows_000.parquet')}))
        flips('an export timing field off by 1 ms', 'V3 TIMING', export=tmp / 'exp3')
        V4N, V5N = 'V4 POPULATIONS', 'V5 ACCOUNTING (the ledger vs the evaluation at B; the labels vs PI0-ALL)'
        def exp_with(name_, g_):
            e_ = tmp / name_ / d0; e_.mkdir(parents=True); g_.to_parquet(e_ / 'rows_000.parquet', index=False); C.dump(e_ / 'receipt.json', dict(outputs={'rows_000.parquet': C.sha(e_ / 'rows_000.parquet')}))
            return tmp / name_
        g4 = pd.read_parquet(exp / 'rows_000.parquet'); x4 = g4[g4.kind == 'post'].iloc[[0]].assign(oid=int(O.oid.max()) + 7, status='missing_endpoint')
        flips('an extra export POST row of no opportunity', V4N, export=exp_with('exp4a', pd.concat([g4, x4], ignore_index=True)))
        nop = g4.index[(g4.kind == 'post') & ~g4.status.isin(PATH_EXPORT)]
        refo = [int(o) for o in g4.oid[(g4.kind == 'post') & (g4.status == 'refused_post_only')] if int(o) % 2 == 1]
        if len(nop):
            g4b = g4.copy(); g4b.loc[nop[0], 'status'] = 'off_grid'; flips('a no-path opportunity exported with a path', V4N, export=exp_with('exp4b', g4b))
        elif refo:
            g4b = g4.copy(); g4b.loc[(g4b.oid == refo[0]) & (g4b.kind == 'post'), 'status'] = 'missing_endpoint'
            flips('an opportunity with a path exported as no-path', V4N, export=exp_with('exp4b', g4b), labels=labs[labs.oid.astype(int) != refo[0]].copy(), pred=pr[pr.oid.astype(int) != refo[0]])
        else:
            bad.append('the synthetic day has neither a no-path nor an unadmitted refused opportunity: the with-path two-source falsifier cannot run')
        evj0 = json.loads((tmp / 'ev' / f'eval_{d0}.json').read_text())
        for nm_, f_ in (('an evaluation input hash changed', lambda j: j['inputs'].__setitem__('export_receipt', 'x' * 64)),
                        ('the evaluation null mean moved by 1 c', lambda j: j['null_A'].__setitem__(B_NAME, j['null_A'][B_NAME] + 1.)),
                        ('the evaluation from another source', lambda j: j.__setitem__('source_sha256', 'y' * 64))):
            j_ = json.loads(json.dumps(evj0)); f_(j_); ed = tmp / ('ev' + str(abs(hash(nm_)) % 100000)); ed.mkdir(); C.dump(ed / f'eval_{d0}.json', j_)
            flips(nm_, V5N, evald=ed)
        if not any(abs(float(evj0['values'][p_][sd][B_NAME])) >= 100. for p_ in evj0['values'] for sd in ('both', 'bid', 'ask')):
            bad.append('no evaluation total of |value| >= 100 c: the 1e-5 c nudge falsifier cannot fire')
        for flag, val, name, which in (('REACH_LEAK', True, 'reached rows of unposted orders', V4N), ('LAND_SHIFT', 1., 'the ledger landing moved by 1 ms', V5N),
                                       ('LEDGER_NUDGE', 1e-5, 'the ledger totals of |value| >= 100 c nudged by 1e-5 c', V5N)):
            old = globals()[flag]; globals()[flag] = val
            try:
                flips(name, which)
            finally:
                globals()[flag] = old
    if bad:
        sys.exit(f'{MARK} SELFTEST FAILED: ' + ' | '.join(bad))
    print('pipeline paired review selftests passed (synthetic only): the tails, the episode-balanced means, the plan v2 screen, the prediction groups and the '
          'value identity on planted values; a clean synthetic run passes all five validity predicates, its ledger windows, components and strata add up; each '
          'planted defect fails only its own predicate: a sampled label off by 1 c (V1), a broken identity (V2), an export timing field off by 1 ms (V3), '
          'reached rows of unposted orders, an extra export row and a path-status disagreement between the two sources (V4), the ledger landing moved, '
          'a 1e-5 c nudge, a changed evaluation input hash / null mean / source (V5); NaN fails V1 / V2; the matched null against enumeration and its '
          'draws; the first-review share, nesting and FINDINGS C1 / C3 / C12 on planted orders and by a per-fill loop; the ledger on planted fill times '
          '(windows, age from activation, cancel-pending, sources, cancellation, admissions)', flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif len(a) == 2 and a[0] == 'day':
        run(a[1])
    else:
        sys.exit('usage: pf_pipeline_paired_review.py selftest | day <day>')
