"""CAUSAL FEATURE / LABEL EXPORT ON THE CORRECTED NEUTRAL JOIN ORDERS (2026-11-19, session aea5bfb9). USER: "Can you check the new commit,
proceed our pipeline model" (after "We should work on the pipeline, a prediction model + value mapping one"). Step 1 of the pipeline plan
committed with the contract c2717dc8 (handoffs/20261119_prediction_pipeline.md): "Export causal market/order inputs on these exact neutral
identities, with timestamps and status completeness; reconcile sampled labels to the kernel. Reuse the richer market representation ...
Record queue age/ahead, remaining quantity, depth/rung, pending operations, inventory and time to the common endpoint. Audit every raw-data
hour boundary." The contract's interfaces (pf_pipeline_contract.py) are imported, never edited. Declared before any data is read; consumed
days Oct 23-27 only; nothing is fitted, chosen or evaluated as a policy.

POPULATION: every committed state of the keep / pull run (pf_keep_pull, keep_pull_20261117 - the scenario replay's identities): one 30 sh
 JOIN order at the known best at t, live at t + 58 ms, the common absolute endpoint END = t + 5000 ms, valued at the END mark; queue rule M.
ROWS (the contract's targets; one row per action, never a sum of overlapping rows):
 POST    one per order: decision t, effective t + 58 (activation), endpoint END; affected = every fill.
 RETAIN  one per live decision step j = J0..98 (d_j = t + 50 j, every 50 ms from +100 ms) while the client knows the order is live at B
         (a full fill is known 100 ms after it; a structural cancel when the client decides it): KEEP versus PULL at d_j; effective = the
         pull's landing d_j + 37 ms; affected = the fills a pull at j can still remove (at or after the landing - the simulator's convention).
 STATUSES (rows are never dropped): the keep / pull run's own (refused_post_only: a POST row with zero value and no RETAIN row;
  missing_endpoint, stale_future_book, price_out_of_range: no labels), off_grid (a decision not on the engine grid) and feature_unavailable
  (no finite causal mid at the decision); counted per day and kind.
INPUTS (known at the decision; the pipeline's own builders, not re-implemented):
 the 71 market features (pf_flow_price_data.feature_rows: book / flow changes, W30 scale, volatility, fair-price response, Binance and
 depth inputs); the 12 order inputs (pf_stop_ev_model.order_inputs of pf_stop_ev_accounting.known_state - rung, depth from the best, price,
 age, known remaining, queue ahead / shown / initial as the client sees them (min-updated shown sizes, rule M), edge to the mid, horizon to
 the common endpoint: the contract's Settings with exposure = END, not the old 1 s); the 6 extras (size, remaining, queue, shown, scale,
 phase); and fair_edge = s * (f_j - p) (A+0's input), t_to_end. known_state is computed vectorised per order (known_steps) and must equal
 the reference function on sampled rows (X5). Pending operations: none in JOIN beyond the structural cancel (a status); inventory: none.
LABELS per row at B (M50) and at the trade-clock shifts M0 / M250 (the keep / pull run's re-simulated branches): affected quantity q,
 value = sum q * s * (mark - p) (the fills' value at the END mark), qm = value - s * (ref - p) * q with ref = the causal mid pf_mid at the
 decision (the contract's E[q * move] term), severe excess max(-value - 30, 0), no-fill; at B q by trade-anchor source (WS / E2 / imputed /
 mixed); live_<clock> (whether the client would still decide at that clock). Realized labels are oracle accounting, never inputs.
CONTROLS (any failure -> no output for the day; parts are written to a partial directory and published only when every control passes):
 X1 every committed state has exactly one POST row; every eligible state has one RETAIN row per live step; no duplicated identity
 X2 POST q / value = the committed order's at B, M0 and M250 (every RETAIN label is recomputed by brute force in the selftest; on real
    data X3 ties the RETAIN labels to the simulator)
 X3 for every declared score policy (A+0 and its five variants) the RETAIN values at the policy's executed pull steps sum to minus the
    committed keep / pull run's improvement of that day, and the pulls count matches (keep_pull.json, M50 / 37 ms): the labels are the
    simulator's valuation of a policy, exactly
 X4 kernel parity: sampled orders (oid % 97 == 0) re-simulated through the kernel (pf_keep_pull_scenarios.fills_g on the window's path) give
    the exported B fills exactly
 X5 causality: on sampled windows (m % 17 == 0) the features at a decision are unchanged when every input after it is perturbed
    (perturb_future / perturb_pb / later depth) and change when same-time information changes (the builder's own falsifier pair); the
    vectorised known state equals known_state on sampled rows (and ignores a fill not yet notified: a selftest unit check); a sampled
    check that never ran is a failure
 X6 hours: every raw hour stem the day's builders read is listed; none is sealed (10-28..30) or forward (>= 11-04)
SELFTEST (synthetic tape, no data): the keep / pull synthetic chain with a causal stub for the market features; known_steps = known_state at
 every synthetic decision; every control passing and a brute recomputation of every label and own input; refusals, each by its own control
 with no output: a leaky feature stub (X5), a stub that ignores the book (X5's positive control), a committed order value (X2) and fill time
 (X4) changed under a matching receipt, a sealed hour (X6), no sampled order (X4 / X5), a landing moved by 37 ms (X3), a dropped RETAIN row
 (X1), the queue ahead not min-updated (X5), a published day. Mutations: pf_pipeline_export_mutants.py.
LIMITS: consumed days; queue rule M only (C5); the replay's fills are not validated on live orders (C10); timing scenarios beyond M0 / M250
 are evaluated by the scenario bank, not labelled here; one order at a time (JOIN), no continuation after END.
Usage (run_light, PF_SLOT=claude): pf_pipeline_export.py selftest | smoke <day> | day <day>
"""
import contextlib
import io
import json
import shutil
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_keep_pull as KP                                 # noqa: E402
import pf_keep_pull_clock_audit as A                      # noqa: E402
import pf_keep_pull_scenarios as SC                       # noqa: E402
import pf_flow_price_common as FC                         # noqa: E402  the pipeline's feature lists and settings
import pf_stop_ev_accounting as AC                        # noqa: E402  known_state (the reference)
import pf_queue_place_test as QP                          # noqa: E402
import pf_queue_place_ledger as QL                        # noqa: E402

C, pd, I = KP.C, KP.pd, KP.I
OUT = Path(I.K.IO) / 'pipeline_export_v2_20261119'       # v2 (review 89caa74c P1): pre-activation POST state from the decision's frame; v1 stays as provenance
MARK = 'PIPELINE EXPORT'
END = KP.H                                                # the common endpoint, ms after posting
SETTINGS = replace(FC.SETTINGS, exposure_ms=END)          # the contract: horizon to the common endpoint, not the old 1 s exposure
CLOCKS = (('m50', 50.), ('m0', 0.), ('m250', 250.))
SRC = ('ws', 'e2', 'imp', 'mixed')                        # the fills' trade-anchor source codes 0 / 1 / 2 / 3
OWN = ['fair_edge', 't_to_end']
INPUTS = list(FC.MARKET) + list(FC.ORDER) + list(FC.EXTRA) + OWN
KNOWN = ('side', 'price', 'rung', 'depth', 'age_ms', 'submitted_quantity', 'known_remaining', 'known_filled', 'queue_initial_known',
         'queue_ahead_known', 'shown_known', 'queue_initial_unknown', 'effective_horizon_ms')
VCOL = {'m50': ('q', 'v', 't_full'), 'm0': ('q_m0', 'v_m0', 't_full_m0'), 'm250': ('q_m250', 'v_m250', 't_full_m250')}   # the committed columns
LANDING_OFFSET = 0.                                       # planted-error switches (selftest only)
DROP_ONE = False
NO_MINUPDATE = False
V1_POST_STATE = False                                     # the v1 pre-activation state (from the activation frame): a planted leak
FREEZE_POST = False                                       # a POST vector that ignores every input (X5b's control cannot fire)


def known_steps(w, side, ta, pt, fl, d, terminal, notice=SETTINGS.notification_ms, exposure=SETTINGS.exposure_ms):
    """AC.known_state for one order at many decision times d (absolute ms), vectorised: fl = (times, quantities) absolute; the queue ahead
    is the minimum of the shown sizes at the price over the frames known from activation to the decision (rule M as the client sees it).
    v2: a decision BEFORE activation (the POST row) uses only the frame known at the decision - the initial queue and the queue ahead are the
    size shown at the price then (known_state's activation-frame convention there reads a future arrival, review 89caa74c P1)."""
    clock = np.asarray(w['kf'], float); d = np.asarray(d, float); ft, fq = (np.asarray(x, float) for x in fl)
    C.require(bool(np.isfinite(clock).all() and (np.diff(clock) >= 0).all()), 'nonmonotone known frame clock')
    ja = int(np.searchsorted(clock, ta, side='right') - 1); jt = (np.searchsorted(clock, d, side='right') - 1).astype(int)
    filled = np.array([sum(float(b) for a, b in zip(ft, fq) if a + notice <= x) for x in d], float)    # AC's sequential sum, exactly
    lo, hi = max(ja, 0), int(jt.max()) if len(jt) else -1
    sh = np.array([AC.shown_at(w, f, side, pt) for f in range(lo, hi + 1)], float) if hi >= lo else np.zeros(0)
    fin = np.where(np.isfinite(sh), sh, np.inf)
    run = fin if NO_MINUPDATE else (np.minimum.accumulate(fin) if len(fin) else fin)
    shown = np.array([(sh[j - lo] if j >= lo else AC.shown_at(w, j, side, pt)) if j >= 0 else np.nan for j in jt], float)
    ok = (ja >= 0) & (ja <= jt)
    initial = np.where(ok, sh[0] if len(sh) else np.nan, np.nan)
    ahead = np.where(ok, np.array([run[j - lo] if (o_ and 0 <= j - lo < len(run)) else np.inf for j, o_ in zip(jt, ok)]), np.nan)
    ahead = np.where(np.isfinite(ahead), ahead, np.nan)
    pre = d < ta                                                                          # v2: before activation the order is not live - its queue is
    if not V1_POST_STATE:                                                                 # the size shown at its price in the frame known at the decision,
        initial, ahead = np.where(pre, shown, initial), np.where(pre, shown, ahead)      # never the (future) activation frame (review 89caa74c P1)
    best = np.array([float(w['P5'][j][0]) if j >= 0 else np.nan for j in jt])
    return dict(side=np.full(len(d), float(side)), price=np.full(len(d), float(pt)), rung=np.zeros(len(d)),
                depth=(best - pt) if side == 0 else (pt - best), age_ms=d - ta, submitted_quantity=np.full(len(d), float(KP.SIZE)),
                known_remaining=np.maximum(0., float(KP.SIZE) - filled), known_filled=filled, queue_initial_known=initial,
                queue_ahead_known=ahead, shown_known=shown, queue_initial_unknown=(~np.isfinite(initial)).astype(float),
                effective_horizon_ms=np.minimum(exposure, terminal - d))


def affected(ft, fq, val, eff):
    """Per effective time (ms after posting): the quantity and value of the fills at or after it (removable / all fills)."""
    o = np.argsort(ft, kind='stable'); ft, fq, val = ft[o], fq[o], val[o]
    cq = np.r_[np.cumsum(fq[::-1])[::-1], 0.]; cv = np.r_[np.cumsum(val[::-1])[::-1], 0.]; i = np.searchsorted(ft, eff, side='left')
    return cq[i], cv[i]


def hour_ok(h):
    """X6: a raw hour stem (YYYYMMDD_HH) a builder reads is neither sealed nor forward."""
    h0 = I._ms(f'{h[:4]}-{h[4:6]}-{h[6:8]}') + int(h[9:11]) * 3600000
    return not (I.SEAL_MS[0] <= h0 < I.SEAL_MS[1]) and h0 < I.FWD_MS


def real_features(W, fair, depth, chosen):
    """The pipeline's own market-feature builder (imported here: it reads the day's step-5 / depth inputs)."""
    import pf_flow_price_data as FD
    return FD.feature_rows(W, fair, depth, chosen)


def real_perturb(W, fair, depth, k, t, rng):
    """The builder's own future perturbation (pf_flow_price_data.build_window): frames / steps after the decision, fair prices stamped
    after it, and the later depth rows."""
    mut = {a: I.M.perturb_future(w, int(w['jk'][k]), k, rng) for a, w in W.items()}
    dep = {a: v.copy() for a, v in depth.items()}; fut = depth['k'] > t
    for a in dep:
        if a != 'k':
            dep[a][fut] += rng.uniform(1, 5, int(fut.sum()))
    return mut, I.M.perturb_pb(fair, t, rng), dep


def real_fire(W, s, k):
    """The builder's same-time positive control: the step row at the decision changed."""
    ch = dict(W[s]); ch['X'] = np.array(W[s]['X'], copy=True); ch['X'][k, 0] += 10.; out = dict(W); out[s] = ch; return out


def causal_check(W, fair, depth, feat_fn, k, s, rng, perturb, fire):
    """X5 on one decision: inputs after it perturbed -> its features unchanged; a same-time change -> they change."""
    chosen = {0: [], 1: []}; chosen[s] = [k]; before = feat_fn(W, fair, depth, chosen)[s][k]; t = float(W[s]['tg'][k])
    mut, fair2, dep2 = perturb(W, fair, depth, k, t, rng); after = feat_fn(mut, fair2, dep2, chosen)[s][k]
    same = bool(np.array_equal(after[0], before[0], equal_nan=True) and np.isclose(after[1], before[1], equal_nan=True))
    fired = not np.array_equal(feat_fn(fire(W, s, k), fair, depth, chosen)[s][k][0], before[0], equal_nan=True)
    return same, fired


def order_block(day, o, js, d, k, on, path, w, F, fills, none, trig, pol, fails):
    """One order's rows as columns: its POST row, then one RETAIN row per live step; identities, statuses, causal inputs, labels."""
    s, t, p, oid = int(o.side), float(o.t), float(o.p), int(o.oid); sg = 1. if s == 0 else -1.; n = 1 + len(js)
    jj = np.r_[0, js].astype(int); eff = np.r_[KP.PLACE, KP.landing(js) + LANDING_OFFSET] if len(js) else np.array([KP.PLACE])
    B = dict(day=np.full(n, day), oid=np.full(n, oid), m=np.full(n, int(o.m)), side=np.full(n, s), t=np.full(n, t), p=np.full(n, p),
             phase=np.full(n, int(o.phase)), mark=np.full(n, float(o.mark)), kind=np.array(['post'] + ['retain'] * len(js)), j=jj,
             decision_ms=d, effective_ms=t + eff, endpoint_ms=np.full(n, t + END))
    if not path:                                                                           # no path: the keep / pull run's status, no labels
        return pd.DataFrame(dict(B, status=np.full(n, str(o.status))))
    for c, _ in CLOCKS:
        ft, fq, fs = fills[c].get(oid, none); val = fq * sg * (float(o.mark) - p); q_, v_ = affected(ft, fq, val, eff)
        qc, vc, tc = VCOL[c]
        if not (abs(q_[0] - float(getattr(o, qc))) < 1e-7 and abs(v_[0] - float(getattr(o, vc))) < 1e-7):     # X2: POST = the committed order
            fails.append(f'X2 order {oid} {c}: POST q / value {q_[0]:.6f} / {v_[0]:.6f} vs the committed {float(getattr(o, qc)):.6f} / {float(getattr(o, vc)):.6f}')
        endc = int(A.endj_of(np.array([float(getattr(o, tc))]), np.array([o.g_struct]))[0])
        B.update({f'q_{c}': q_, f'value_{c}': v_, f'severe_{c}': np.maximum(-v_ - KP.SEVERE, 0.), f'nofill_{c}': (q_ <= 1e-12).astype(float),
                  f'live_{c}': np.r_[True, js < endc]})
        if c == 'm50':
            for i_, nm in enumerate(SRC):
                B[f'q_m50_{nm}'] = affected(ft[fs == i_], fq[fs == i_], val[fs == i_], eff)[0]
            v50 = v_
    st = np.where(on, 'eligible', 'off_grid').astype(object)
    X = np.full((n, len(FC.MARKET)), np.nan, np.float32); scale = np.full(n, np.nan)
    for i_ in np.flatnonzero(on):
        X[i_], scale[i_] = F[s][int(k[i_])][0], F[s][int(k[i_])][1]
    ref = X[:, FC.MARKET.index('pf_mid')].astype(float); st[on & ~np.isfinite(ref)] = 'feature_unavailable'
    if o.status == 'refused_post_only':
        st[0] = 'refused_post_only'
    ft, fq, _ = fills['m50'].get(oid, none); kn = known_steps(w, s, t + KP.PLACE, p, (ft + t, fq), d, t + END)
    oi = FC.V.order_inputs(pd.DataFrame({f'x_{a}': kn[a] for a in KNOWN}), scale, ref)
    B.update(status=st, **{nm: X[:, i_] for i_, nm in enumerate(FC.MARKET)}, **{nm: np.asarray(oi[nm], float) for nm in FC.ORDER},
             fp_size=np.full(n, float(KP.SIZE)), fp_remaining=kn['known_remaining'], fp_queue=kn['queue_ahead_known'], fp_shown=kn['shown_known'],
             fp_scale=scale, fp_phase=np.full(n, float(o.phase)), fair_edge=sg * (np.array([float(getattr(o, f'f{x:02d}')) for x in jj]) - p),
             t_to_end=END - (d - t))
    for c, _ in CLOCKS:
        B[f'qm_{c}'] = B[f'value_{c}'] - sg * (ref - p) * B[f'q_{c}']
    for pp in KP.POLICIES:                                                                 # X3: the policy's executed pull at a live step
        T = trig[pp].get(oid)
        if T is not None and len(js) and js[0] <= T <= js[-1]:
            pol[pp][0] += float(v50[1 + int(T) - int(js[0])]); pol[pp][1] += 1
    return pd.DataFrame(B)


def known_parity(w, s, o, fl, st):
    """X5: the vectorised known state = AC.known_state on sampled live steps of an order (after activation)."""
    bad = []; t = float(o.t); endj = int(A.endj_of(np.array([o.t_full]), np.array([o.g_struct]))[0]); js = np.arange(KP.J0, max(endj, KP.J0))
    d = np.r_[t, t + KP.STEP * js]; order = dict(ta=t + KP.PLACE, pt=float(o.p), fl=[(a + t, b) for a, b in zip(fl[0], fl[1])])
    vec = known_steps(w, s, t + KP.PLACE, float(o.p), (fl[0] + t, fl[1]), d, t + END)
    eq = lambda x, y: (np.isnan(x) and np.isnan(y)) or x == y
    for i in sorted({1 if len(d) > 1 else 0, len(d) // 2, len(d) - 1}):
        if d[i] < t + KP.PLACE:                                                            # v2: pre-activation decisions follow their own rule (X5b)
            continue
        ref = AC.known_state({s: w}, order, (s, 0, int(o.oid)), float(d[i]), KP.SIZE, terminal_ms=t + END, settings=SETTINGS)
        if not all(eq(float(ref[a]), float(vec[a][i])) for a in KNOWN):
            bad.append(f'X5 order {int(o.oid)} decision {i}: the vectorised known state differs from known_state '
                       + str({a: (float(ref[a]), float(vec[a][i])) for a in KNOWN if not eq(float(ref[a]), float(vec[a][i]))}))
        st['known_checked'] += 1
    return bad


def post_vector(W, s, o, fl, fair, depth, feat_fn, k):
    """The POST row's complete input vector at its decision (step k): market features, the known state and the order inputs."""
    if FREEZE_POST:
        return np.zeros(len(FC.MARKET) + 1 + len(KNOWN) + len(FC.ORDER))
    t = float(o.t); chosen = {0: [], 1: []}; chosen[s] = [k]; x, scale = feat_fn(W, fair, depth, chosen)[s][k]
    kn = known_steps(W[s], s, t + KP.PLACE, float(o.p), (fl[0] + t, fl[1]), np.array([t]), t + END); ref = float(x[FC.MARKET.index('pf_mid')])
    oi = FC.V.order_inputs(pd.DataFrame({f'x_{a}': kn[a] for a in KNOWN}), [scale], [ref])
    return np.r_[np.asarray(x, float), float(scale), [float(kn[a][0]) for a in KNOWN], [float(oi[nm][0]) for nm in FC.ORDER]]


def arrive_later(W, k, t, rng, values=True):
    """X5b's perturbation: every frame received after the decision gets a new receipt time in (t, t + 150 ms] (monotone; some cross the
    activation) and, with values, randomised book values; the known-frame index per step is rebuilt from the new receipt clock."""
    out = {}
    for a, w in W.items():
        w2 = dict(w); kf = np.array(w['kf'], float, copy=True); tg = np.asarray(w['tg'], float); j0 = int(np.searchsorted(kf, t, side='right'))
        C.require(np.array_equal(np.searchsorted(kf, tg, side='right') - 1, np.asarray(w['jk'], int)), 'X5b: the known-frame index is not the receipt clock search')
        n = len(kf) - j0
        if n > 0:
            kf[j0:] = np.maximum.accumulate(np.r_[t + np.sort(rng.uniform(1., 150., min(n, 4))), kf[j0 + min(n, 4):]])
            w2['kf'] = kf; w2['jk'] = (np.searchsorted(kf, tg, side='right') - 1).astype(np.asarray(w['jk']).dtype)
            if values:
                for key in ('P5', 'S5', 'opp'):
                    v = np.array(w[key], float, copy=True); v[j0:] = v[j0:] + rng.integers(1, 4, v[j0:].shape) if key != 'S5' else v[j0:] * rng.uniform(.2, 3., v[j0:].shape)
                    w2[key] = v
                if 'X' in w:
                    X2 = np.array(w['X'], float, copy=True); X2[k + 1:] = X2[k + 1:] * rng.uniform(.5, 1.5, X2[k + 1:].shape); w2['X'] = X2
                w2['cache'] = {}
        out[a] = w2
    return out


def post_causal(W, s, o, fl, fair, depth, feat_fn, k, rng):
    """X5b: the complete POST vector is unchanged when later frames arrive at other times with other values, and changes when the size shown
    at the price in the decision's own frame changes (the same-time positive control)."""
    base = post_vector(W, s, o, fl, fair, depth, feat_fn, k); t = float(o.t)
    same = np.array_equal(post_vector(arrive_later(W, k, t, rng), s, o, fl, fair, depth, feat_fn, k), base, equal_nan=True)
    w = dict(W[s]); jt = int(np.searchsorted(np.asarray(w['kf'], float), t, side='right') - 1); S5 = np.array(w['S5'], float, copy=True); P5 = np.asarray(w['P5'])
    hit = np.flatnonzero(P5[jt] == float(o.p)); S5[jt, hit[0] if len(hit) else 0] += 7.; w['S5'] = S5; w['cache'] = {}; W2 = dict(W); W2[s] = w
    fired = not np.array_equal(post_vector(W2, s, o, fl, fair, depth, feat_fn, k), base, equal_nan=True)
    return same, fired


def controls(day, O, st, counts, pol, kp, full):
    """X1 and X3 after the day (X2, X4, X5 run per order / window, X6 before any file opens)."""
    bad = []; el = O[O.status == 'eligible']
    want = int(np.maximum(A.endj_of(el.t_full.values, el.g_struct.values) - KP.J0, 0).sum())
    if not (st['post_rows'] == len(O) and st['retain_rows'] == want and st['ids_unique']):
        bad.append(f"X1: {st['post_rows']:,} POST rows for {len(O):,} states, {st['retain_rows']:,} RETAIN rows for {want:,} live steps, unique ids {st['ids_unique']}")
    if full:
        D = json.loads((kp / 'keep_pull.json').read_text())['D']['M50_L37']; i = KP.DAYS.index(day)
        for p_, (v, n) in pol.items():
            imp, n_ = float(D[p_]['policy_by_day']['imp'][i]), int(round(D[p_]['policy_by_day']['n'][i]))
            if not (n == n_ and abs(-v - imp) <= 1e-6 * max(1., abs(imp))):
                bad.append(f'X3 {p_}: RETAIN values at the executed pulls sum to {-v:,.3f} over {n:,} pulls, the committed keep / pull run {imp:,.3f} over {n_:,}')
    return bad


def real_inputs(day, markets):
    """The day's fair price, depth arrays and the raw hours they read (as pf_flow_price_data.run builds them), with their hashes."""
    import pf_io as IO
    s1 = Path(I.K.IO) / 'step1' / day; tracked = {}
    reg = QP.JD.metadata(day, True)[1]; W0 = {int(x.mid): float(x.ws) * 1000. for x in reg.itertuples() if int(x.mid) in set(markets)}
    C.require(bool(W0) and not (I.SEAL_MS[0] <= min(W0.values()) < I.SEAL_MS[1]) and max(W0.values()) < I.FWD_MS, 'protected date')
    hours = I.X5.depth_hours(day); C.require(all(hour_ok(h) for h in hours), 'X6: a sealed or forward hour in the day\'s builders')   # before any file opens
    fair, _ = I.fair_fix(day, W0)
    for h in hours:
        for p_ in (Path(IO.BN_DIR) / (h + '.csv.gz'), Path(IO.REF_RUN) / 'reference_prices' / (h + '.jsonl.gz'), Path(I.X5.DEPTH_DIR) / (h + '.csv.gz')):
            if p_.exists():
                tracked[str(p_)] = C.sha(p_)
    ck = json.loads((s1 / 'clock_rl_d2.json').read_text()); depth, _ = I.X5.depth_arrays(hours, float(ck['bn_in']) + float(ck['d_dec']))
    C.require(depth is not None, 'depth unavailable')
    return {int(k_): v for k_, v in fair.items()}, depth, hours, tracked


def day_run(day, src=None, kp=None, out=None, feat_fn=None, inputs=None, markets=None, sample_k=97, sample_w=17, perturb=None, fire=None):
    """One day -> out/<day>/rows_<batch>.parquet + receipt.json, published only after X1-X6 pass (written to <day>.partial first)."""
    start = time.time(); C.require(day in KP.DAYS, 'undeclared day'); src = src or KP.Tape(); kp = Path(kp or KP.OUT); out = Path(out or OUT)
    feat_fn, perturb, fire = feat_fn or real_features, perturb or real_perturb, fire or real_fire
    fin, part = out / day, out / (day + '.partial'); C.require(not fin.exists(), 'day output exists')
    r = json.loads((kp / day / 'receipt.json').read_text()); C.require(all(C.sha(kp / day / x) == h for x, h in r['outputs'].items()), 'keep-pull outputs changed since their receipt')
    mk = src.markets(day) if markets is None else list(markets)
    fair, depth, hrs, tracked = (inputs or real_inputs)(day, mk)
    bad = [f'X6 hour {h} is sealed or forward' for h in hrs if not hour_ok(h)]
    if bad:
        sys.exit(f'{MARK} FAILED -> no output: ' + ' | '.join(bad))
    O = pd.read_parquet(kp / day / 'orders.parquet'); Fl = pd.read_parquet(kp / day / 'fills.parquet'); O = O[O.m.isin(mk)].reset_index(drop=True)
    fills = {c: {int(o_): (g.t.values.astype(float), g.q.values.astype(float), g.src.values.astype(int)) for o_, g in Fl[Fl.clock == x].groupby('oid')} for c, x in CLOCKS}
    none = (np.zeros(0), np.zeros(0), np.zeros(0, int)); E = O[O.status == 'eligible']
    T = KP.score_steps(E[[f'f{j:02d}' for j in range(KP.NSTEP + 1)]].values, E.side.values, E.p.values)
    trig = {p_: dict(zip(E.oid.values.astype(int).tolist(), np.asarray(T[p_]).astype(int).tolist())) for p_ in KP.POLICIES}; pol = {p_: [0., 0] for p_ in KP.POLICIES}
    by_m = {int(m): g for m, g in O.groupby('m')}; offset = src.offset(day)
    if part.exists():
        shutil.rmtree(part)
    part.mkdir(parents=True); st = dict(states=len(O), post_rows=0, retain_rows=0, kernel_checked=0, causal_checked=0, known_checked=0, ids_unique=True)
    counts, seen = {}, set()
    for begin in range(0, len(mk), 24):
        data = src.load(day, mk[begin:begin + 24], offset); blocks = []
        for W in src.pairs(data):
            m = int(W[0]['m'])
            if m not in by_m:
                continue
            plan, chosen = [], {0: [], 1: []}
            for o in by_m[m].itertuples():
                s, t = int(o.side), float(o.t); tg = np.asarray(W[s]['tg'], float); path = o.status in ('eligible', 'refused_post_only')
                endj = int(A.endj_of(np.array([o.t_full]), np.array([o.g_struct]))[0]) if o.status == 'eligible' else KP.J0
                js = np.arange(KP.J0, max(endj, KP.J0)) if o.status == 'eligible' else np.zeros(0, int)
                if DROP_ONE and len(js) > 1 and int(o.oid) % 2 == 0:
                    js = js[:-1]
                d = np.r_[t, t + KP.STEP * js]; k = np.searchsorted(tg, d); kk = np.minimum(k, len(tg) - 1); on = (k < len(tg)) & (np.abs(tg[kk] - d) <= 1e-6)
                if path:
                    chosen[s].extend(k[on].tolist())
                plan.append((o, js, d, k, on, path))
            chosen = {a: sorted(set(v)) for a, v in chosen.items()}
            F = feat_fn(W, fair.get(m) if fair is not None else None, depth, chosen); fails = []
            for o, js, d, k, on, path in plan:
                b = order_block(day, o, js, d, k, on, path, W[int(o.side)], F, fills, none, trig, pol, fails); blocks.append(b)
                ids = set(zip(b.oid.tolist(), b.kind.tolist(), b.j.tolist())); st['ids_unique'] &= not (ids & seen) and len(ids) == len(b); seen |= ids
                st['post_rows'] += int((b.kind == 'post').sum()); st['retain_rows'] += int((b.kind == 'retain').sum())
                for (kd, sx), n_ in b.groupby(['kind', 'status']).size().items():
                    counts[f'{kd}:{sx}'] = counts.get(f'{kd}:{sx}', 0) + int(n_)
            bad += fails
            if m % sample_w == 0:                                                         # X5 on a sampled window
                s_ = 0 if chosen[0] else 1
                if chosen[s_]:
                    same, fired = causal_check(W, fair.get(m) if fair is not None else None, depth, feat_fn, chosen[s_][len(chosen[s_]) // 2], s_,
                                               np.random.default_rng(m), perturb, fire)
                    if not (same and fired):
                        bad.append(f'X5 window {m}: ' + ' and '.join(x for x, y in (('future inputs changed a feature', not same), ('the same-time control did not fire', not fired)) if y))
                    st['causal_checked'] += 1
            for o in by_m[m].itertuples():                                                # X4 kernel parity and X5 known-state parity, sampled
                if o.status != 'eligible' or not sample_k or int(o.oid) % sample_k:
                    continue
                s, t, p = int(o.side), float(o.t), float(o.p); base, _ = QP.build_path(W, t, t + KP.H); fl = fills['m50'].get(int(o.oid), none)
                vk, qk, tt, fq = SC.fills_g(base, s, p, KP.PLACE, KP.CANCEL, KP.PENDING); mine = fq > 1e-12
                if not (np.array_equal(tt[mine], fl[0]) and np.array_equal(fq[mine], fl[1])):
                    bad.append(f'X4 order {int(o.oid)}: the kernel gives other fills than the exported B fills')
                st['kernel_checked'] += 1; bad += known_parity(W[s], s, o, fl, st)
                kp_ = int(np.searchsorted(np.asarray(W[s]['tg'], float), t))                            # X5b: the complete POST vector
                same, fired = post_causal(W, s, o, fl, fair.get(m) if fair is not None else None, depth, feat_fn, kp_, np.random.default_rng(int(o.oid)))
                if not (same and fired):
                    bad.append(f'X5b order {int(o.oid)}: ' + ' and '.join(x for x, y in (('a future arrival changed the POST inputs', not same),
                                                                                            ('the same-time POST control did not fire', not fired)) if y))
                st['post_checked'] = st.get('post_checked', 0) + 1
        if blocks:
            pd.concat(blocks, ignore_index=True).to_parquet(part / f'rows_{begin // 24:03d}.parquet', index=False)
        del data, blocks; C.guard('pipeline export')
        print(json.dumps(dict(stage='export', day=day, markets=min(begin + 24, len(mk)), post=st['post_rows'], retain=st['retain_rows'],
                              seconds=round(time.time() - start), rss=round(I.K.rss_mib()))), flush=True)
    bad += controls(day, O, st, counts, pol, kp, markets is None)
    if not (st['kernel_checked'] and st['known_checked'] and st['causal_checked'] and st.get('post_checked', 0)) and len(E):
        bad.append(f"X4 / X5: a sampled check never ran ({st['kernel_checked']} kernel, {st['known_checked']} known-state, {st['causal_checked']} causal)")
    if bad:
        shutil.rmtree(part); sys.exit(f'{MARK} FAILED -> no output: ' + ' | '.join(bad[:12]) + (f' (+{len(bad) - 12} more)' if len(bad) > 12 else ''))
    parts = sorted(x.name for x in part.glob('rows_*.parquet'))
    C.dump(part / 'receipt.json', dict(asof=C.stamp(), day=day, stat=st, statuses=counts, policies={p_: dict(retain_sum=v[0], pulls=v[1]) for p_, v in pol.items()},
                                       hours=hrs, inputs=tracked, keep_pull_receipt=C.sha(kp / day / 'receipt.json'), keep_pull_outputs=r['outputs'],
                                       settings=dict(exposure_ms=SETTINGS.exposure_ms, notification_ms=SETTINGS.notification_ms, cancel_ms=KP.CANCEL, end_ms=END),
                                       input_columns=INPUTS, markets=len(mk), full_day=markets is None, provenance=KP.provenance(),
                                       source_sha256=C.sha(Path(__file__).resolve()), outputs={x: C.sha(part / x) for x in parts}, seconds=time.time() - start))
    part.rename(fin); print(json.dumps(dict(day=day, stat=st, statuses=counts, seconds=round(time.time() - start))), flush=True)
    return fin


def syn_inputs(day, markets):
    """The selftest's inputs: no fair price or depth (the stub ignores them); the day's own 24 hours."""
    return {}, None, [f'{day}_{h:02d}' for h in range(24)], {}


def syn_features(leak=0, const=False):
    """A causal stub of the market-feature builder on the synthetic tape: at step k the book of the frame known at tg[k] (jk[k]) gives pf_mid
    and the first inputs. leak = 1 also reads the next frame (a planted look-ahead that still responds to same-time information); const ignores
    the book (a check that cannot fire)."""
    def f(W, fair, depth, chosen):
        out = {}
        for s, w in W.items():
            P5, S5, opp, jk = (np.asarray(w[x], float) for x in ('P5', 'S5', 'opp', 'jk')); res = {}
            for k in chosen.get(s, []):
                j = int(jk[k]); jl = min(j + leak, len(P5) - 1); x = np.full(len(FC.MARKET), np.nan, np.float32)
                bid, ask = (P5[j, 0], opp[j]) if s == 0 else (opp[j], P5[j, 0])
                x[FC.MARKET.index('pf_mid')] = 50. if const else ((bid + ask) / 2. if bid >= 0 and ask >= 0 else np.nan)
                x[0], x[1], x[2] = (1., 1., 1.) if const else (P5[j, 0], S5[jl, 0], float(k)); res[k] = (x, 10.)   # leak: input 1 from the next frame
            out[s] = res
        return out
    return f


def syn_perturb(W, fair, depth, k, t, rng):
    """The selftest's future perturbation: every frame after the one known at the decision randomised (P5 / S5 / opp)."""
    out = {}
    for a, w in W.items():
        jk = int(np.asarray(w['jk'])[k]); w2 = dict(w); P5, S5, opp = (np.array(w[x], float, copy=True) for x in ('P5', 'S5', 'opp'))
        P5[jk + 1:] += rng.integers(1, 4, P5[jk + 1:].shape); S5[jk + 1:] *= rng.uniform(.2, 3., S5[jk + 1:].shape); opp[jk + 1:] += rng.integers(1, 4, opp[jk + 1:].shape)
        w2.update(P5=P5, S5=S5, opp=opp); out[a] = w2
    return out, fair, depth


def syn_fire(W, s, k):
    """The selftest's same-time control: the frame known at the decision changed."""
    w = dict(W[s]); jk = int(np.asarray(w['jk'])[k]); P5 = np.array(w['P5'], float, copy=True); P5[jk, 0] += 1.; w['P5'] = P5; out = dict(W); out[s] = w; return out


def selftest():
    """Synthetic tape only (see the module's SELFTEST)."""
    global LANDING_OFFSET, DROP_ONE, NO_MINUPDATE, V1_POST_STATE, FREEZE_POST
    bad = []; src = KP.Syn(); days = list(KP.DAYS); d0 = days[0]
    q_, v_ = affected(np.array([100., 137., 136.9, 500.]), np.array([5., 3., 2., 4.]), np.array([1., -2., 4., -8.]), np.array([58., 137., 137.1, 501.]))
    if not (q_.tolist() == [14., 7., 4., 0.] and v_.tolist() == [-5., -10., -8., 0.]):
        bad.append(f'affected: {q_.tolist()} {v_.tolist()}')
    if (hour_ok('20261027_23'), hour_ok('20261028_00'), hour_ok('20261030_23'), hour_ok('20261031_00'), hour_ok('20261103_23'), hour_ok('20261104_00')) != (
            True, False, False, True, True, False):
        bad.append('the sealed / forward hour rule')
    def win(nxt):                                                                         # review 89caa74c P1: identical history and values,
        P5 = np.tile([50., 49., 48., 47., 46.], (4, 1)); S5 = np.tile([12., 5., 5., 5., 5.], (4, 1))   # the next book receipt at +30 / +80 ms
        return dict(kf=np.array([0., 990., 1000. + nxt, 1200.]), P5=P5, S5=S5)
    kq = lambda w_, d_: known_steps(w_, 0, 1058., 50., (np.zeros(0), np.zeros(0)), np.array([d_]), 6000.)
    eqd = lambda a_, b_: all((np.isnan(a_[x][0]) and np.isnan(b_[x][0])) or a_[x][0] == b_[x][0] for x in KNOWN)
    post_same, live_same, post_q = eqd(kq(win(30.), 1000.), kq(win(80.), 1000.)), eqd(kq(win(30.), 1100.), kq(win(80.), 1100.)), kq(win(30.), 1000.)['queue_ahead_known'][0]
    V1_POST_STATE = True
    try:
        v1_same = eqd(kq(win(30.), 1000.), kq(win(80.), 1000.))
    finally:
        V1_POST_STATE = False
    if not (post_same and live_same and post_q == 12. and not v1_same):
        bad.append(f'the POST state against a future arrival: identical at submission {post_same} (queue {post_q}), at +100 ms {live_same}, v1 differs {not v1_same}')
    with tempfile.TemporaryDirectory(dir=str(I.K.IO)) as tmp:
        tmp = Path(tmp)
        with QL._v1_on(src, tmp):
            for d in days:
                QP.day_run(d)
        with contextlib.redirect_stdout(io.StringIO()):
            for d in days:
                KP.day_run(d, src=src, v1=tmp / 'v1', out=tmp / 'kp')
            KP.report(tmp / 'kp')
        O = pd.read_parquet(tmp / 'kp' / d0 / 'orders.parquet'); Fl = pd.read_parquet(tmp / 'kp' / d0 / 'fills.parquet'); Oi = O.set_index('oid')

        def run(out_, **kw):
            with contextlib.redirect_stdout(io.StringIO()):
                return day_run(d0, src=src, kp=kw.pop('kp', tmp / 'kp'), out=out_, feat_fn=kw.pop('feat_fn', syn_features()), inputs=kw.pop('inputs', syn_inputs),
                               sample_k=kw.pop('sample_k', 1), sample_w=1, perturb=syn_perturb, fire=syn_fire, **kw)
        # known_steps = AC.known_state at every decision of every eligible synthetic order, and a fill not yet notified changes nothing
        data = src.load(d0, src.markets(d0), src.offset(d0)); n_kn = diff = late = 0
        for W in src.pairs(data):
            for o in O[(O.m == int(W[0]['m'])) & (O.status == 'eligible')].itertuples():
                s, t = int(o.side), float(o.t); f = Fl[(Fl.oid == o.oid) & (Fl.clock == 50.)]; fl = (f.t.values + t, f.q.values)
                endj = int(A.endj_of(np.array([o.t_full]), np.array([o.g_struct]))[0]); d = np.r_[t, t + KP.STEP * np.arange(KP.J0, max(endj, KP.J0))]
                vec = known_steps(W[s], s, t + KP.PLACE, float(o.p), fl, d, t + END)
                order = dict(ta=t + KP.PLACE, pt=float(o.p), fl=list(zip(fl[0], fl[1])))
                for i, x in enumerate(d):
                    if x < t + KP.PLACE:                                                       # v2: pre-activation decisions follow their own rule
                        continue
                    ref = AC.known_state({s: W[s]}, order, (s, 0, int(o.oid)), float(x), KP.SIZE, terminal_ms=t + END, settings=SETTINGS); n_kn += 1
                    diff += int(not all((np.isnan(float(ref[a])) and np.isnan(vec[a][i])) or float(ref[a]) == float(vec[a][i]) for a in KNOWN))
                ln = known_steps(W[s], s, t + KP.PLACE, float(o.p), (np.r_[fl[0], d[-1] - SETTINGS.notification_ms + 1.], np.r_[fl[1], .01]), d[-1:], t + END)
                late += int(not all((np.isnan(ln[a][0]) and np.isnan(vec[a][-1])) or ln[a][0] == vec[a][-1] for a in KNOWN))
        if not (n_kn > 1000 and diff == 0 and late == 0):
            bad.append(f'known_steps vs known_state: {diff} of {n_kn} decisions differ, {late} orders moved by a fill not yet notified')
        del data
        fin = None
        try:
            fin = run(tmp / 'ex')
        except (SystemExit, ValueError) as e:
            bad.append(f'the synthetic export refused: {str(e)[:400]}')
        if fin is not None:
            R = pd.concat([pd.read_parquet(x) for x in sorted(fin.glob('rows_*.parquet'))], ignore_index=True); rec = json.loads((fin / 'receipt.json').read_text())
            el = O[O.status == 'eligible']; want_r = int(np.maximum(A.endj_of(el.t_full.values, el.g_struct.values) - KP.J0, 0).sum())
            ok_counts = (len(R[R.kind == 'post']) == len(O) and len(R[R.kind == 'retain']) == want_r and not R.duplicated(['oid', 'kind', 'j']).any()
                         and int((R[R.kind == 'post'].status == 'refused_post_only').sum()) == int((O.status == 'refused_post_only').sum())
                         and sum(rec['statuses'].values()) == len(R) and rec['stat']['kernel_checked'] > 0 and rec['stat']['known_checked'] > 0
                         and rec['stat']['causal_checked'] > 0 and (R.status == 'eligible').sum() > 0)
            if not ok_counts:
                bad.append(f"rows / statuses: {len(R):,} rows, {rec['statuses']}, stat {rec['stat']}")
            errs = n = 0; P = R[R.kind.isin(['post', 'retain']) & R.oid.isin(O[O.status.isin(['eligible', 'refused_post_only'])].oid)]
            for oid, g in P.groupby('oid'):                                                  # brute recomputation of every label and own input
                o = Oi.loc[oid]; sg = 1. if o.side == 0 else -1.; eff = np.where(g.kind == 'post', KP.PLACE, KP.STEP * g.j + KP.CANCEL); ref = g.pf_mid.values.astype(float)
                for c, x in CLOCKS:
                    f = Fl[(Fl.oid == oid) & (Fl.clock == x)]; ft, fq = f.t.values, f.q.values; val = fq * sg * (o.mark - o.p)
                    qb = np.array([fq[ft >= e].sum() for e in eff]); vb = np.array([val[ft >= e].sum() for e in eff])
                    tc = VCOL[c][2]; endc = int(A.endj_of(np.array([float(o[tc])]), np.array([float(o.g_struct)]))[0]); liveb = np.where(g.kind == 'post', True, g.j < endc)
                    errs += int((np.abs(qb - g[f'q_{c}']) > 1e-9).sum() + (np.abs(vb - g[f'value_{c}']) > 1e-9).sum()
                                + (~np.isclose(vb - sg * (ref - o.p) * qb, g[f'qm_{c}'], atol=1e-6, equal_nan=True)).sum()
                                + (np.abs(np.maximum(-vb - KP.SEVERE, 0.) - g[f'severe_{c}']) > 1e-9).sum() + (liveb != g[f'live_{c}'].values.astype(bool)).sum())
                fe = sg * (np.array([float(o[f'f{int(j):02d}']) for j in g.j]) - o.p); hz = (o.t + END - g.decision_ms.values) / 1000.
                el_ = g.status.values == 'eligible'
                errs += int((~np.isclose(fe, g.fair_edge, equal_nan=True)).sum() + (np.abs(END - (g.decision_ms - o.t) - g.t_to_end) > 1e-9).sum()
                            + (~np.isclose(hz[el_], g.o_horizon.values[el_], equal_nan=True)).sum()
                            + (~np.isclose(g.fp_remaining.values[el_] / KP.SIZE, g.o_rem.values[el_], equal_nan=True)).sum()
                            + (~np.isclose(g.fp_queue.values[el_] / g.fp_scale.values[el_], g.o_q_rel.values[el_], equal_nan=True)).sum())
                n += len(g)
            if not (n > 0 and errs == 0):
                bad.append(f'brute recomputation: {errs} differences over {n:,} rows')
        def refused(name, msg, **kw):
            o_ = tmp / ('z' + str(abs(hash(name)) % 100000))
            try:
                run(o_, **kw); bad.append(f'the export accepted {name}')
            except (SystemExit, ValueError) as e:
                if msg not in str(e):
                    bad.append(f'{name} refused by another control: {str(e)[:300]}')
            if (o_ / d0).exists() or (o_ / (d0 + '.partial')).exists():
                bad.append(f'a refused export left output ({name})')
        def kp_copy(name, change):
            d_ = tmp / name; shutil.copytree(tmp / 'kp', d_); change(d_ / d0)
            r_ = json.loads((d_ / d0 / 'receipt.json').read_text()); r_['outputs'] = {x: C.sha(d_ / d0 / x) for x in r_['outputs']}
            (d_ / d0 / 'receipt.json').write_text(json.dumps(r_)); return d_
        def bump_v(dd):
            o2 = pd.read_parquet(dd / 'orders.parquet'); i_ = int(np.flatnonzero((o2.status == 'eligible').values & (o2.q.values > 0))[0])
            o2.loc[o2.index[i_], 'v'] = o2.v.iloc[i_] + 1.; o2.to_parquet(dd / 'orders.parquet', index=False)
        def move_fill(dd):
            f2 = pd.read_parquet(dd / 'fills.parquet'); e_ = set(O[O.status == 'eligible'].oid)
            i_ = int(np.flatnonzero((f2.clock.values == 50.) & f2.oid.isin(e_).values & (f2.t.values < 4000.))[0]); f2.loc[f2.index[i_], 't'] = f2.t.iloc[i_] + 7.
            f2.to_parquet(dd / 'fills.parquet', index=False)
        refused('a leaky feature stub', 'future inputs changed a feature', feat_fn=syn_features(leak=1))
        refused('a feature stub that ignores the book (the same-time control cannot fire)', 'the same-time control did not fire', feat_fn=syn_features(const=True))
        refused('a committed order value changed under a matching receipt', 'X2 order', kp=kp_copy('kpv', bump_v))
        refused('a committed fill time changed under a matching receipt', 'X4 order', kp=kp_copy('kpt', move_fill))
        refused('a sealed hour', 'X6 hour', inputs=lambda day, mk: ({}, None, ['20261028_00'], {}))
        refused('no order sampled for the kernel / known-state checks', 'never ran', sample_k=None)
        for flag, name, msg in (('LANDING_OFFSET', 'a landing moved by 37 ms', 'X3'), ('DROP_ONE', 'a dropped RETAIN row', 'X1'),
                                ('NO_MINUPDATE', 'the queue ahead not min-updated', 'X5 order'),
                                ('V1_POST_STATE', 'the v1 POST state (a future arrival)', 'a future arrival changed the POST inputs'),
                                ('FREEZE_POST', 'a POST vector that ignores its inputs', 'the same-time POST control did not fire')):
            globals()[flag] = -37. if flag == 'LANDING_OFFSET' else True
            try:
                refused(name, msg)
            finally:
                LANDING_OFFSET, DROP_ONE, NO_MINUPDATE, V1_POST_STATE, FREEZE_POST = 0., False, False, False, False
        if fin is not None:
            try:
                run(tmp / 'ex'); bad.append('the export overwrote a published day')
            except (SystemExit, ValueError) as e:
                if 'day output exists' not in str(e):
                    bad.append(f'a published day refused by another check: {str(e)[:300]}')
    if bad:
        sys.exit(f'{MARK} SELFTEST FAILED: ' + ' | '.join(bad))
    print('pipeline export selftests passed (synthetic only): affected-fill arithmetic; the sealed / forward hour rule; known_steps = known_state at every '
          'decision of every eligible synthetic order, a fill not yet notified ignored; the production path end to end with a causal feature stub (rows, '
          'statuses, every control passing) and a brute recomputation of every label and own input; refusals by their own control with no output: a leaky '
          'stub (X5), a stub that ignores the book (X5), a committed value (X2) and fill time (X4) changed under a matching receipt, a sealed hour (X6), no '
          'sampled order (X4 / X5), a landing moved by 37 ms (X3), a dropped RETAIN row (X1), the queue ahead not min-updated (X5), the v1 POST '
          'state (X5b: a future arrival) and a frozen POST vector (X5b\'s control), and a published day; the POST state against a future arrival '
          '(review 89caa74c P1: identical at submission and at +100 ms, the v1 rule differs)', flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif len(a) == 2 and a[0] == 'smoke':
        C.require(a[1] in KP.DAYS, 'undeclared day'); day_run(a[1], out=OUT / 'smoke', markets=KP.Tape().markets(a[1])[:2], sample_w=1)   # the smoke checks every window
    elif len(a) == 2 and a[0] == 'day':
        day_run(a[1])
    else:
        sys.exit('usage: pf_pipeline_export.py selftest | smoke <day> | day <day>')
