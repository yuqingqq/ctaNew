"""E2SEP (2026-11-15, session aea5bfb9; USER: "Go" on the recommendation after STOP-EV stage 2): are the local losses - and the stage-2
model's skill - in windows with E2-timed trades a TIMING artifact or ACTIVITY? Stage 2 found every prediction decile losing and, on the
4,572 strictly WS-timed test actions, a mean value of -0.12 c with no skill (IC -0.009); strict-WS windows are also the quietest, so that
contrast is confounded. Descriptive analysis on the consumed days; no model, no policy.

BUILDS: per day, stage 1's PRIMARY labels exactly (pf_stop_ev: the corrected neutral ladder, the Codex ledger, one probe per order
  generation, all_anchors_sensitivity, the USER's untimed rule) rebuilt with every E2-anchored leg's time shifted by DELTA in {-250, 0,
  +250} ms (the C12 uncertainty; WS legs untouched; the legs re-sorted in the lane's order, the provenance following each leg). Beside each
  action's status and value: the shares of the legs that can REACH the order in its avoidable window [t + 22 ms, min(t + 1 s, end)] (its
  side, at or beyond its price) by source, the side's leg shares there by source (activity), and the E2 quantity of its own counted fills
  (ledger tranches).
GROUPS (per build, eligible actions): QUIET = no reaching leg (value 0 by construction); WS-REACHED = every reaching share WS-timed;
  E2-REACHED = at least half of the reaching shares E2-timed; MIXED = the rest.
READINGS DECLARED BEFORE THE RUN (9 consumed days; c per 30 sh order over 1 s):
 R950 TIMING: the E2-REACHED mean value at DELTA -250 / 0 / +250 (each build's own group); its range > 50% of |mean at 0| pooled ->
      TIMING-SENSITIVE (the E2 population's values depend on the assumed trade time), else TIMING-ROBUST; WS-REACHED's range printed
      beside it (path effects of the shift) and the per-day ranges.
 R951 MATCHED ACTIVITY (DELTA 0): within strata rung x side x quintile of the reaching shares (quintile edges over all 9 days' reached
      actions), E2-REACHED minus WS-REACHED mean value, pooled over the strata holding both (weights = their sizes): < 0 pooled and on >= 7
      of 9 days -> E2-WORSE-AT-EQUAL-ACTIVITY; > 0 pooled and on >= 7 -> E2-BETTER-AT-EQUAL-ACTIVITY; else NOT-SHOWN.
 R952 MODEL BY GROUP (stage 2's test-day predictions joined by action id to DELTA 0): IC within WS-REACHED > 0 pooled and on >= 4 of 5
      test days -> SKILL-ON-WS, else NO-SKILL-ON-WS; the IC within E2-REACHED and MIXED printed.
 Printed: group sizes and mean values per build and day, each group's share of the total loss, the E2 share of filled quantity.
CONTROLS (any failure -> no output): DELTA 0 reproduces stage 1's actions exactly (status and value, every action); in every build the
  shift moves each E2 leg by exactly DELTA and no WS leg, and each leg's source row's raw anchor + its shift equals its new anchor (the
  provenance followed the re-sort); the ledger reconciles in every build (0 failures); synthetic: the shift on a toy window keeps the
  lane's order and every leg's identity; planted errors caught: WS legs shifted too, the provenance left unsorted.
LIMITS: a shift moves every E2 leg of the window by the same DELTA (the true errors vary per leg); E2-timed trades may differ in kind
  from WS-timed ones (R951 matches activity, not kind); consumed days; replay rule M.
Usage (run_light, PF_SLOT=claude): pf_stop_ev_e2sep.py selftest | day <day> | report; driver run_stop_ev_e2sep.sh.
"""
import bisect
import collections
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_stop_ev as E                                    # noqa: E402
import pf_stop_ev_model as V                              # noqa: E402  (stage 2's predictions, spearman)

M, K, AC = E.M, E.K, E.AC
OUTD = f'{K.IO}/stop_ev_e2sep_20261115'
DELTAS = (-250.0, 0.0, 250.0); TIMING = 'all_anchors_sensitivity'; ELIG = ('eligible', 'eligible_estimated_timing')
MEM_CAP_MIB = 2048.0; NAN = float('nan')
MUT_SHIFT_WS = MUT_PROV_STALE = 0                         # planted errors (selftest only)
MUTS = ('MUT_SHIFT_WS', 'MUT_PROV_STALE')


def shift_window(W, LR, SR, delta):
    """W with every E2-anchored leg (SR False) moved by delta ms and each side's legs re-sorted in the lane's order (anchor, then price
    priority); the leg rows LR and sources SR follow the legs."""
    W2, LR2, SR2 = {}, {}, {}
    for s in (0, 1):
        w = W[s]; src = np.asarray(SR[s], bool); anc = np.asarray(w['anc'], float)
        mv = np.ones(len(src), bool) if MUT_SHIFT_WS else ~src
        a2 = anc + delta * mv.astype(float)
        pt = np.asarray(w['ptl'], 'int64'); sh = np.asarray(w['shl'], float)
        o = np.lexsort(((-pt if s == 0 else pt), a2)) if len(a2) else np.zeros(0, 'int64')
        W2[s] = dict(w, anc=a2[o], ptl=pt[o].tolist(), shl=sh[o].tolist())
        LR2[s] = np.asarray(LR[s])[o] if not MUT_PROV_STALE else np.asarray(LR[s])
        SR2[s] = src[o] if not MUT_PROV_STALE else src
    return W2, LR2, SR2


def check_shift(W2, LR2, SR2, raw_anc, raw_ws, delta):
    """Violations: a leg whose new anchor differs from its source row's raw anchor + its shift, a WS leg moved, a side not time-sorted."""
    bad = 0
    for s in (0, 1):
        a2 = np.asarray(W2[s]['anc'], float)
        if not len(a2):
            continue
        r0 = np.asarray([raw_anc[int(x)] for x in LR2[s]], float); ws_ = np.asarray([raw_ws[int(x)] for x in LR2[s]], bool)
        bad += int((a2 != r0 + delta * (~ws_).astype(float)).sum())
        bad += int((np.asarray(SR2[s], bool) != ws_).sum())
        bad += int((np.diff(a2) < 0).sum())
    return bad


def build_window(day, W, LR, SR, RW, st, asof, delta, errs):
    """Stage 1's primary labels (pf_stop_ev.actions_window: the same coverage, provenance, ledger, probes and untimed rule) on the window
    with E2 legs shifted by delta, plus each action's reach / activity / fill composition by source. Returns columns or None."""
    W2, LR2, SR2 = shift_window(W, LR, SR, delta)
    m = int(W2[0]['m']); ws = float(W2[0]['ws']); tf = np.asarray(W2[0]['tf'], float)
    r = AC.replay_neutral(W2, size=E.Z, shift_ms=E.SHF, queue='M', unknown_queue='zero', nr=E.NR, cap=None, mode='none')
    prov = {(s, i): AC.LegInfo(str(int(LR2[s][i])), 'ws' if bool(SR2[s][i]) else 'e2', 0.0 if bool(SR2[s][i]) else E.E2_UNC_MS)
            for s in (0, 1) for i in range(len(LR2[s]))}
    rw = RW.get(m, dict(expected=0, missing=[], unt={0: [], 1: []}))
    n_unt = sum(1 for x in rw['missing'] if x[3] == 'untimed'); gi = np.flatnonzero(np.diff(tf) > E.BOOK_GAP_MS)
    cov = AC.Coverage(trade_start_ms=ws - 2000.0 - E.SHF, trade_end_ms=ws + 61000.0 - E.SHF, mark_end_ms=float(tf[-1]),
                      expected_legs=int(rw['expected']) - n_unt, book_start_ms=float(tf[0]), book_end_ms=float(tf[-1]),
                      missing=tuple(AC.MissingLeg(*x) for x in rw['missing'] if x[3] != 'untimed'),
                      book_gaps=tuple(AC.MissingLeg(f'gap_{m}_{i}', float(tf[i]), float(tf[i + 1]), 'feed_gap') for i in gi))
    try:
        led = AC.Ledger(W2, r, submitted_quantity=E.Z, shift_ms=E.SHF, leg_info=prov, coverage=cov)
    except ValueError as e:
        st['ledger_fail'] += 1
        if len(errs) < 20:
            errs.append(f'window {m} delta {delta:g}: {e}')
        return None
    rows, _ = AC.build_actions(led, day=day, asof=asof, ages_ms=E.AGES, settings=AC.Settings(timing=TIMING))
    LEG = {s: (np.asarray(W2[s]['anc'], float) - E.SHF, np.asarray(W2[s]['ptl'], 'int64'), np.asarray(W2[s]['shl'], float), np.asarray(SR2[s], bool))
           for s in (0, 1)}
    c = collections.defaultdict(list)
    for x in rows:
        _, mm, sd, rung, gen = x['action_id']; t = float(x['decision_ms']); status = x['status']; lab = x['label']
        if status in ELIG:                                # the USER's untimed rule, as pf_stop_ev.actions_window applies it
            pr_ = float(x['features']['price']); end_ = min(t + 1000.0, led.end)
            if any(u[0] <= end_ and u[1] >= t and ((u[2] <= pr_) if sd == 0 else (u[2] >= pr_)) for u in rw['unt'][sd]):
                status = 'untimed_reach'; lab = None
        tt, pt, sh, src = LEG[sd]; end = min(t + 1000.0, led.end)
        lo = int(np.searchsorted(tt, t + E.L_C, side='left')); hi = int(np.searchsorted(tt, end, side='right'))
        f_ = x['features'] or {}; pr = float(f_.get('price', NAN))
        rch = ((pt[lo:hi] <= pr) if sd == 0 else (pt[lo:hi] >= pr)) if pr == pr else np.zeros(hi - lo, bool)
        s_ = src[lo:hi]; q_ = sh[lo:hi]
        fe2 = 0.0
        if lab:
            for tr in lab['tranches']:
                if t + E.L_C <= tr['time'] <= end and not bool(src[tr['leg'][1]]):
                    fe2 += float(tr['quantity'])
        for c_, v_ in (('m', int(mm)), ('side', int(sd)), ('rung', int(rung)), ('generation', int(gen)), ('decision_ms', t), ('status', status),
                       ('value_cents', float(lab['value_cents']) if lab else NAN), ('filled_quantity', float(lab['filled_quantity']) if lab else NAN),
                       ('fill_e2', fe2 if lab else NAN), ('reach_ws', float(q_[rch & s_].sum())), ('reach_e2', float(q_[rch & ~s_].sum())),
                       ('act_ws', float(q_[s_].sum())), ('act_e2', float(q_[~s_].sum())), ('price', pr), ('known_remaining', float(f_.get('known_remaining', NAN)))):
            c[c_].append(v_)
        st[f'actions_{delta:g}_{status}'] += 1
    return c


def day_run(day):
    all_selftests()
    if day not in E.DAYS9:
        sys.exit(f'REFUSED: {day} is not a consumed replay day')
    T0 = time.time(); D = K.load_day(day); cut = M.boundary_cut(day); RW, _ = E.raw_windows(day, D)
    tkf = pd.read_parquet(f'{K.IO}/step1/{day}/legs_tk.parquet', columns=['tk']).tk.values
    LT = D['LT']; ix = LT.index.values; g_all = np.where(tkf[ix] >= 0, tkf[ix], -1 - ix)
    WSg = pd.Series(E.TS.ws_groups(LT.how.values, g_all), index=LT.index)
    raw_anc = LT.anc.to_dict(); raw_ws = WSg.to_dict()
    st = collections.Counter(); errs = []; asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()); OUT = {d: [] for d in DELTAS}
    for W in E.P1.window_pairs(D):
        if W[0]['ws'] < cut:
            continue
        m = int(W[0]['m']); LR = {}; SR = {}
        for s in (0, 1):
            if (m, s) in D['LI']:
                l = LT.iloc[D['LI'][(m, s)]]
                o_ = np.lexsort(((-l.ptick.values if s == 0 else l.ptick.values), l.anc.values)); LR[s] = l.index.values[o_]
            else:
                LR[s] = np.zeros(0, 'int64')
            if len(LR[s]) != len(W[s]['ptl']):
                sys.exit(f'REFUSED: {day} window {m}: leg rows not aligned')
            SR[s] = WSg.reindex(LR[s]).values.astype(bool) if len(LR[s]) else np.zeros(0, bool)
        st['windows'] += 1
        for d in DELTAS:
            W2, LR2, SR2 = shift_window(W, LR, SR, d)
            st[f'shift_violations_{d:g}'] += check_shift(W2, LR2, SR2, raw_anc, raw_ws, d)
            st[f'e2_legs_{d:g}'] += int(sum((~np.asarray(SR2[s], bool)).sum() for s in (0, 1)))
            c = build_window(day, W, LR, SR, RW, st, asof, d, errs)
            if c is not None:
                OUT[d].append({k: np.asarray(v) for k, v in c.items()})
        if st['windows'] % 40 == 0:
            E.mem_guard(f'e2sep {day}')
    A0 = pd.read_parquet(f'{E.OUTD}/actions_{day}{E.ZTAG}_{TIMING}.parquet', columns=['m', 'side', 'rung', 'generation', 'status', 'value_cents'])
    B0 = pd.DataFrame({k: np.concatenate([x[k] for x in OUT[0.0]]) for k in OUT[0.0][0]})
    J = A0.merge(B0[['m', 'side', 'rung', 'generation', 'status', 'value_cents']], on=['m', 'side', 'rung', 'generation'], how='outer', suffixes=('', '_b'),
                 indicator=True)
    both = J._merge.values == 'both'; v1, v2 = J.value_cents.values.astype(float), J.value_cents_b.values.astype(float)
    reg = dict(stage1=int(len(A0)), rebuilt=int(len(B0)), unmatched=int((~both).sum()), status_differ=int((J.status.values[both] != J.status_b.values[both]).sum()),
               value_differ=int((~((np.isnan(v1) & np.isnan(v2)) | (np.abs(v1 - v2) <= 1e-9)))[both].sum()))
    fails = []
    if reg['unmatched'] or reg['status_differ'] or reg['value_differ']:
        fails.append(f'DELTA 0 does not reproduce stage 1: {reg}')
    if any(st[f'shift_violations_{d:g}'] for d in DELTAS) or st['ledger_fail']:
        fails.append(f"shift violations {[st[f'shift_violations_{d:g}'] for d in DELTAS]}, ledger failures {st['ledger_fail']} {errs[:3]}")
    rep = dict(day=day, stat=dict(st), regression=reg, asof=asof, seconds=round(time.time() - T0, 1), sources=E.source_ids())
    os.makedirs(OUTD, exist_ok=True)
    if fails:
        json.dump(rep, open(f'{OUTD}/refused_{day}.json', 'w'), indent=1, default=str)
        sys.exit(f'REFUSED (no output for {day}): {fails}')
    for d in DELTAS:
        X = pd.DataFrame({k: np.concatenate([x[k] for x in OUT[d]]) for k in OUT[d][0]}); X.insert(0, 'day', int(day))
        X.to_parquet(f'{OUTD}/e2_{day}_{int(d)}.parquet', index=False)
    json.dump(rep, open(f'{OUTD}/day_{day}.json', 'w'), indent=1, default=str)
    print(f'{day}: windows {st["windows"]}, regression {reg}, shift violations 0, {rep["seconds"]} s', flush=True)


# ---------------------------------------------------------------- report
def groups(X):
    rs = X.reach_ws.values + X.reach_e2.values
    with np.errstate(invalid='ignore', divide='ignore'):
        e2f = np.where(rs > 0, X.reach_e2.values / np.where(rs > 0, rs, 1.0), 0.0)
    return np.where(rs <= 0, 'QUIET', np.where(X.reach_e2.values <= 0, 'WS-REACHED', np.where(e2f >= 0.5, 'E2-REACHED', 'MIXED')))


def report():
    days = [d for d in E.DAYS9 if os.path.exists(f'{OUTD}/day_{d}.json')]
    if len(days) < len(E.DAYS9):
        sys.exit(f'REFUSED: days missing {[d for d in E.DAYS9 if d not in days]}')
    srcs = {json.dumps(json.load(open(f'{OUTD}/day_{d}.json'))['sources'], sort_keys=True) for d in days}
    if len(srcs) > 1:
        sys.exit('REFUSED: days built from different sources')
    X = {d: pd.concat([pd.read_parquet(f'{OUTD}/e2_{day}_{int(d)}.parquet') for day in days], ignore_index=True) for d in DELTAS}
    for d in DELTAS:
        X[d] = X[d][X[d].status.isin(ELIG).values].reset_index(drop=True); X[d]['grp'] = groups(X[d])
    T = [f'E2SEP - timing vs activity in the STOP-EV local values ({TIMING}, the USER\'s untimed rule; 9 consumed days; c per 30 sh order over 1 s; '
         f'as of {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})', '']
    G = ('QUIET', 'WS-REACHED', 'MIXED', 'E2-REACHED')
    T.append('GROUPS per E2 shift (eligible actions): n | mean value | share of the total loss | E2 share of filled quantity')
    for d in DELTAS:
        x = X[d]; tot = float(x.value_cents.sum()); fq = float(np.nansum(x.filled_quantity.values)); fe = float(np.nansum(x.fill_e2.values))
        T.append(f'  DELTA {d:+.0f} ms: all {len(x)} | {x.value_cents.mean():+.3f} | E2 share of filled quantity {fe / max(fq, 1e-12):.1%}')
        for g in G:
            y = x.value_cents.values[x.grp.values == g]
            T.append(f'    {g:10s} {len(y):7d} | {y.mean() if len(y) else NAN:+.3f} | {y.sum() / tot if tot else NAN:6.1%}')
    R = {}
    mean = lambda d, g, dd=None: float(X[d].value_cents.values[(X[d].grp.values == g) & (True if dd is None else (X[d].day.values == dd))].mean())
    e2 = [mean(d, 'E2-REACHED') for d in DELTAS]; wsm = [mean(d, 'WS-REACHED') for d in DELTAS]
    rng_e2 = max(e2) - min(e2); r950 = 'TIMING-SENSITIVE' if rng_e2 > 0.5 * abs(e2[1]) else 'TIMING-ROBUST'
    T += ['', f'R950 TIMING: E2-REACHED mean at DELTA -250 / 0 / +250: ' + ' / '.join(f'{v:+.3f}' for v in e2) + f' (range {rng_e2:.3f} vs 50% of |mean at 0| '
              f'{0.5 * abs(e2[1]):.3f}) -> {r950}; WS-REACHED ' + ' / '.join(f'{v:+.3f}' for v in wsm)]
    T.append('  per day E2-REACHED -250 / 0 / +250: ' + '; '.join(f"{dd} " + ' / '.join(f'{mean(d, "E2-REACHED", int(dd)):+.2f}' for d in DELTAS) for dd in days))
    x0 = X[0.0]; rs = x0.reach_ws.values + x0.reach_e2.values; reached = rs > 0
    edges = np.quantile(rs[reached], [0.2, 0.4, 0.6, 0.8]); qb = np.digitize(rs, edges)
    key = pd.Series(list(zip(x0.rung.values, x0.side.values, qb)))

    def diff(sel):
        num = den = 0.0
        for k_, ix in key[sel].groupby(key[sel]).groups.items():
            ix = np.asarray(list(ix)); g_ = x0.grp.values[ix]; v_ = x0.value_cents.values[ix]
            a, b = v_[g_ == 'E2-REACHED'], v_[g_ == 'WS-REACHED']
            if len(a) and len(b):
                w = len(a) + len(b); num += w * (a.mean() - b.mean()); den += w
        return num / den if den else NAN, den
    base = reached & np.isin(x0.grp.values, ['E2-REACHED', 'WS-REACHED'])
    pooled, wt = diff(base); per = {dd: diff(base & (x0.day.values == int(dd)))[0] for dd in days}
    neg = sum(int(v < 0) for v in per.values()); pos = sum(int(v > 0) for v in per.values())
    r951 = ('E2-WORSE-AT-EQUAL-ACTIVITY' if (pooled < 0 and neg >= 7) else 'E2-BETTER-AT-EQUAL-ACTIVITY' if (pooled > 0 and pos >= 7) else 'NOT-SHOWN')
    T.append(f'R951 MATCHED ACTIVITY (DELTA 0; strata rung x side x reach-share quintile, edges {np.round(edges, 1).tolist()} sh): E2-REACHED - WS-REACHED '
             f'{pooled:+.3f} c pooled over {int(wt)} actions; per day ' + ', '.join(f'{dd} {v:+.2f}' for dd, v in per.items()) + f' -> {r951}')
    P = pd.concat([pd.read_parquet(f'{V.OUTD}/pred_{t}_{t}.parquet') for t in V.TESTS], ignore_index=True)
    J = x0.merge(P[['day', 'm', 'side', 'rung', 'generation', 'pred']], on=['day', 'm', 'side', 'rung', 'generation'], how='inner')
    T.append(f'R952 MODEL BY GROUP (stage 2 test-day predictions joined to DELTA 0: {len(J)} of {len(P)}): group | n | IC pooled | IC per test day')
    ics = {}
    for g in ('WS-REACHED', 'MIXED', 'E2-REACHED'):
        x = J[J.grp.values == g]; per_ = {t: V.spearman(x.pred.values[x.day.values == int(t)], x.value_cents.values[x.day.values == int(t)]) for t in V.TESTS}
        ics[g] = (V.spearman(x.pred.values, x.value_cents.values), per_)
        T.append(f'  {g:10s} {len(x):7d} | {ics[g][0]:+.3f} | ' + ', '.join(f'{t} {v:+.3f}' for t, v in per_.items()))
    ws_ic, ws_per = ics['WS-REACHED']
    r952 = 'SKILL-ON-WS' if (ws_ic > 0 and sum(int(v > 0) for v in ws_per.values()) >= 4) else 'NO-SKILL-ON-WS'
    T += ['', f'READINGS: R950 {r950}; R951 {r951}; R952 {r952}', '(descriptive; consumed days; replay rule M; a shift moves all E2 legs of a window alike)']
    R.update(R950=dict(e2=e2, ws=wsm, reading=r950), R951=dict(pooled=pooled, per_day=per, reading=r951, edges=edges.tolist()),
             R952=dict(ic={g: v[0] for g, v in ics.items()}, per_day={g: v[1] for g, v in ics.items()}, reading=r952))
    open(f'{OUTD}/e2sep.txt', 'w').write('\n'.join(T) + '\n')
    json.dump(R, open(f'{OUTD}/e2sep.json', 'w'), indent=1, default=float)
    print('\n'.join(T), flush=True)


# ---------------------------------------------------------------- self-tests
def selftest():
    bad = []
    legs = [(100.0, 50, 5.0, 1), (100.0, 49, 7.0, 1), (300.0, 50, 4.0, 2), (420.0, 48, 9.0, 3), (420.0, 50, 2.0, 4)]
    w0, lr0, g0, s0 = E.toy_w(0, legs)                    # sources: even groups WS, odd E2 (toy_w)
    w1, lr1, g1, s1 = E.toy_w(1, [(250.0, 52, 3.0, 2), (260.0, 53, 6.0, 5)])
    W = {0: w0, 1: w1}; LR = {0: lr0, 1: lr1 + 100}; SR = {0: s0, 1: s1}
    raw_anc = {int(r): float(a) for s in (0, 1) for r, a in zip(LR[s], W[s]['anc'])}
    raw_ws = {int(r): bool(x) for s in (0, 1) for r, x in zip(LR[s], SR[s])}
    for d in (-250.0, 0.0, 250.0):
        W2, LR2, SR2 = shift_window(W, LR, SR, d)
        v = check_shift(W2, LR2, SR2, raw_anc, raw_ws, d)
        if v:
            bad.append(f'shift {d:g}: {v} violations')
        for s in (0, 1):                                  # identities kept: the same rows with the same prices and shares
            got = sorted(zip(LR2[s].tolist(), W2[s]['ptl'], W2[s]['shl'])); ref = sorted(zip(LR[s].tolist(), W[s]['ptl'], W[s]['shl']))
            if got != ref:
                bad.append(f'shift {d:g} side {s}: leg identities changed')
            if d == 0.0 and not (np.array_equal(W2[s]['anc'], W[s]['anc']) and list(LR2[s]) == list(LR[s])):
                bad.append(f'shift 0 side {s}: not the identity')
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
        sys.exit(f'E2SEP SELFTEST FAILED: {bad}')
    m = mutation_check()
    if not all(m.values()):
        sys.exit(f'E2SEP PLANTED ERROR NOT CAUGHT: {m}')
    print(f'e2sep selftests passed; planted errors caught: {m}', flush=True)
    return m


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        all_selftests()
    elif cmd == 'day':
        day_run(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit(__doc__)
