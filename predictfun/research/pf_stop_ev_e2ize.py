"""E2IZE (2026-11-15, session aea5bfb9; USER: "Go next" after the model's skill split): are the replay's losses on E2-timed trades REAL
adverse flow or MANUFACTURED by the E2 timing method? Counterfactual on trades whose true time we know.

E2 TIMES (step 1, pf_step1_booktime): a cell (market, exec second, side, price) without a WS time is timed at the book frame where the shown
  size at that price DROPPED by the amount closest to the cell's volume, within +-600 ms of the exec second (estimator 'E2'; scored on
  WS-timed cells: 10-23 median |error| 49 ms, p90 589 ms, 72% within 250 ms - the replay assumed +-250 ms).
BUILD: per day, stage 1's primary labels (the same corrected neutral ladder, ledger, probes, all_anchors_sensitivity and the USER's untimed
  rule, via pf_stop_ev_e2sep.build_window) with every WS-ANCHORED taker group re-timed by its own E2 estimate (the group's earliest t_e2,
  as load_day anchors E2 groups); E2-anchored groups unchanged; a WS group without an E2 estimate keeps its WS time (counted). Re-timed legs
  are E2-class in the provenance (+-250 ms), as real E2 legs are.
POPULATION: stage 1's actions under the true timing, grouped by the Codex prefix certificate (INVARIANT = certified_ws eligible: no E2 or
  missing event can reach any accepted order of the side before the label ends - their values rest on WS times only), compared by FIXED
  IDENTITY (one-to-one on day / m / side / rung / generation / decision_ms) with the re-timed build.
READINGS DECLARED BEFORE THE RUN (9 consumed days; c per 30 sh order over 1 s):
 R970 ARTIFACT: on matched INVARIANT actions, mean (re-timed value - true-timing value): <= -1.0 c pooled and < 0 on >= 7 of 9 days ->
      E2-TIMING-MANUFACTURES-LOSS (the timing method alone produces losses on trades whose true times make none); within +-0.3 c pooled ->
      E2-TIMING-NEUTRAL; else NOT-SHOWN. The same difference on the other groups and the unmatched shares are printed.
 R971 E2 ERROR: on WS-timed cells with an E2 estimate, the signed error (E2 time - median WS time of the cell): median, quartiles, p10 / p90,
      share later than +250 ms and earlier than -250 ms (descriptive).
CONTROLS (any failure -> no output): re-timing every WS group by its own WS anchor reproduces the window exactly; re-timed legs keep their
  identities and the lane's order; only WS-anchored groups move; the ledger reconciles (0 failures); synthetic: the toy re-timing and its
  planted errors (E2 groups re-timed too; the legs left unsorted) caught.
LIMITS: WS-timed trades may differ in kind from those that lacked WS times (busier periods, different takers); the WS time itself is a
  feed time, not a verified execution time; consumed days; replay rule M.
Usage (run_light, PF_SLOT=claude): pf_stop_ev_e2ize.py selftest | day <day> | report; driver run_stop_ev_e2ize.sh.
"""
import collections
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_stop_ev_e2sep as X                              # noqa: E402  (build_window, manifest, the certificate keys)

E, M, K, AC = X.E, X.M, X.K, X.AC
OUTD = f'{K.IO}/stop_ev_e2ize_20261115'
KEYJ = X.KEYJ; ELIG = X.ELIG; TIMING = X.TIMING
MUT_RETIME_E2 = MUT_NO_SORT = 0                           # planted errors (selftest only)
MUTS = ('MUT_RETIME_E2', 'MUT_NO_SORT')


def retime_window(W, LR, SR, new_anc):
    """W with every WS-anchored leg (SR True) moved to new_anc[row] (its group's E2 estimate) when finite, re-sorted in the lane's order
    (anchor, then price priority); re-timed legs become E2-class. Returns (W2, LR2, SR2, n_retimed, n_kept)."""
    W2, LR2, SR2 = {}, {}, {}; nr = nk = 0
    for s in (0, 1):
        w = W[s]; src = np.asarray(SR[s], bool); anc = np.asarray(w['anc'], float); rows = np.asarray(LR[s])
        e2 = np.asarray([new_anc.get(int(r), np.nan) for r in rows], float)
        mv = (src | bool(MUT_RETIME_E2)) & np.isfinite(e2)
        a2 = np.where(mv, e2, anc); nr += int(mv.sum()); nk += int((src & ~np.isfinite(e2)).sum())
        pt = np.asarray(w['ptl'], 'int64'); sh = np.asarray(w['shl'], float)
        o = np.lexsort(((-pt if s == 0 else pt), a2)) if len(a2) else np.zeros(0, 'int64')
        if MUT_NO_SORT:
            o = np.arange(len(a2))
        W2[s] = dict(w, anc=a2[o], ptl=pt[o].tolist(), shl=sh[o].tolist()); LR2[s] = rows[o]; SR2[s] = (src & ~mv)[o]
    return W2, LR2, SR2, nr, nk


def group_e2(LT, day):
    """Per leg row: its taker group's earliest E2 estimate (t_e2 over the group's legs), the anchor load_day would give the group if it
    had no WS time."""
    tk = pd.read_parquet(f'{K.IO}/step1/{day}/legs_tk.parquet', columns=['tk']).tk.values.astype('int64')
    T = pd.read_parquet(f'{K.IO}/step1/{day}/legs_timed2.parquet', columns=['t_e2'])
    rid = np.arange(len(T)); grp = np.where(tk >= 0, tk, -1 - rid)
    g = pd.Series(T.t_e2.values.astype(float)).groupby(grp).transform('min').values
    return {int(r): float(g[r]) for r in LT.index.values}


def e2_error(day):
    """R971: on WS-timed cells with an E2 estimate, E2 time - the cell's median WS time (ms)."""
    L = pd.read_parquet(f'{K.IO}/step1/{day}/legs_timed2.parquet', columns=['mid', 'exec_s', 'side', 'ptick', 'how', 't_ms', 't_e2'])
    c = L.groupby(['mid', 'exec_s', 'side', 'ptick']).agg(n=('how', 'size'), nt=('how', lambda h: int((h > 0).sum())), truth=('t_ms', 'median'), e2=('t_e2', 'first'))
    c = c[(c.n == c.nt) & np.isfinite(c.e2.values)]
    return (c.e2.values - c.truth.values).astype(float)


def day_run(day):
    all_selftests()
    if day not in E.DAYS9:
        sys.exit(f'REFUSED: {day} is not a consumed replay day')
    T0 = time.time(); MAN = X.manifest(); D = K.load_day(day); cut = M.boundary_cut(day); RW, _ = E.raw_windows(day, D)
    tkf = pd.read_parquet(f'{K.IO}/step1/{day}/legs_tk.parquet', columns=['tk']).tk.values
    LT = D['LT']; ix = LT.index.values; g_all = np.where(tkf[ix] >= 0, tkf[ix], -1 - ix)
    WSg = pd.Series(E.TS.ws_groups(LT.how.values, g_all), index=LT.index)
    GE2 = group_e2(LT, day); WSA = LT.anc.to_dict()
    st = collections.Counter(); errs = []; asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()); OUT = []
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
            SR[s] = WSg.reindex(LR[s]).values.astype(bool) if len(LR[s]) else np.zeros(0, bool)
        st['windows'] += 1
        Wi, _, _, _, _ = retime_window(W, LR, SR, WSA)           # control: re-timed by their own WS anchors = the window itself
        st['identity_differ'] += int(any(not np.array_equal(Wi[s]['anc'], np.asarray(W[s]['anc'], float)) or Wi[s]['ptl'] != list(W[s]['ptl']) for s in (0, 1)))
        W2, LR2, SR2, nr, nk = retime_window(W, LR, SR, GE2)
        st['retimed_legs'] += nr; st['ws_legs_without_e2'] += nk
        st['identity_lost'] += int(any(sorted(LR2[s].tolist()) != sorted(np.asarray(LR[s]).tolist()) for s in (0, 1)))
        c = X.build_window(day, W2, LR2, SR2, RW, st, asof, 0.0, errs)
        if c is not None:
            OUT.append({k: np.asarray(v) for k, v in c.items()})
        if st['windows'] % 40 == 0:
            E.mem_guard(f'e2ize {day}')
    fails = []
    if st['identity_differ'] or st['identity_lost'] or st['ledger_fail'] or X.manifest() != MAN:
        fails.append(f"identity re-timing {st['identity_differ']}, legs lost {st['identity_lost']}, ledger failures {st['ledger_fail']} {errs[:2]}, "
                     f"manifest changed {X.manifest() != MAN}")
    err = e2_error(day)
    rep = dict(day=day, stat=dict(st), asof=asof, seconds=round(time.time() - T0, 1), manifest=MAN,
               e2_error=dict(n=int(len(err)), q=[float(v) for v in np.quantile(err, [0.1, 0.25, 0.5, 0.75, 0.9])] if len(err) else [],
                             late_250=float(np.mean(err > 250.0)) if len(err) else float('nan'), early_250=float(np.mean(err < -250.0)) if len(err) else float('nan')))
    os.makedirs(OUTD, exist_ok=True)
    if fails:
        json.dump(rep, open(f'{OUTD}/refused_{day}.json', 'w'), indent=1, default=str)
        sys.exit(f'REFUSED (no output for {day}): {fails}')
    A = pd.DataFrame({k: np.concatenate([x[k] for x in OUT]) for k in OUT[0]}); A.insert(0, 'day', int(day))
    A.to_parquet(f'{OUTD}/e2ize_{day}.parquet', index=False)
    json.dump(rep, open(f'{OUTD}/day_{day}.json', 'w'), indent=1, default=str)
    print(f'{day}: windows {st["windows"]}, re-timed legs {st["retimed_legs"]} (WS legs without an E2 estimate {st["ws_legs_without_e2"]}), '
          f'E2 error median {rep["e2_error"]["q"][2] if rep["e2_error"]["q"] else float("nan"):+.0f} ms, {rep["seconds"]} s', flush=True)


def report():
    days = [d for d in E.DAYS9 if os.path.exists(f'{OUTD}/day_{d}.json')]
    if len(days) < len(E.DAYS9):
        sys.exit(f'REFUSED: days missing {[d for d in E.DAYS9 if d not in days]}')
    J = {d: json.load(open(f'{OUTD}/day_{d}.json')) for d in days}
    if len({json.dumps(x['manifest'], sort_keys=True) for x in J.values()}) > 1:
        sys.exit('REFUSED: days built from different sources')
    base = pd.concat([pd.read_parquet(f'{E.OUTD}/actions_{d}{E.ZTAG}_{TIMING}.parquet', columns=KEYJ + ['status', 'value_cents']) for d in days], ignore_index=True)
    base = base[base.status.isin(ELIG).values].reset_index(drop=True)
    C = {}
    for mode, okst in (('certified_ws', ('eligible_timing_invariant',)), ('certified_estimated', ('eligible_estimated_timing', 'eligible'))):
        Y = pd.concat([pd.read_parquet(f'{X.CERT}/{d}/{mode}.parquet', columns=KEYJ + ['status']) for d in days], ignore_index=True)
        Y['day'] = Y['day'].astype('int64'); C[mode] = Y[Y.status.isin(okst).values]
    key = lambda Z: pd.MultiIndex.from_frame(Z[KEYJ])
    inv = key(base).isin(key(C['certified_ws'])); est = key(base).isin(key(C['certified_estimated']))
    base['grp'] = np.where(inv, 'INVARIANT', np.where(est, 'E2-DEPENDENT', 'UNCERTIFIED'))
    RT = pd.concat([pd.read_parquet(f'{OUTD}/e2ize_{d}.parquet') for d in days], ignore_index=True)
    Jn, ok = X.id_join(base, RT)
    dv = Jn.value_cents_b.values - Jn.value_cents.values
    T = [f'E2IZE - WS-timed trades re-timed by their own E2 book-time estimates (9 consumed days; c per 30 sh order over 1 s; as of '
         f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})', '']
    q = np.concatenate([[J[d]['e2_error']['q'][2]] for d in days]); nerr = sum(J[d]['e2_error']['n'] for d in days)
    T.append('R971 E2 ERROR on WS-timed cells (E2 time - WS time, ms) per day: n | p10 / p25 / median / p75 / p90 | later than +250 | earlier than -250')
    for d in days:
        e = J[d]['e2_error']
        T.append(f"  {d}: {e['n']} | " + ' / '.join(f'{v:+.0f}' for v in e['q']) + f" | {e['late_250']:.1%} | {e['early_250']:.1%}")
    T.append(f"  re-timed legs per day: " + ', '.join(f"{d} {J[d]['stat']['retimed_legs']} (kept WS {J[d]['stat']['ws_legs_without_e2']})" for d in days))
    T += ['', 'GROUPS (true-timing eligible actions): n | matched in the re-timed build | mean value true / re-timed (matched) | mean change | share changed']
    R = {}
    for g in ('INVARIANT', 'E2-DEPENDENT', 'UNCERTIFIED'):
        s_ = base.grp.values == g; mb = s_ & ok
        R[g] = dict(n=int(s_.sum()), matched=float(mb.sum() / max(s_.sum(), 1)), true=float(Jn.value_cents.values[mb].mean()) if mb.any() else float('nan'),
                    retimed=float(Jn.value_cents_b.values[mb].mean()) if mb.any() else float('nan'), change=float(dv[mb].mean()) if mb.any() else float('nan'),
                    changed=float(np.mean(np.abs(dv[mb]) > 1e-9)) if mb.any() else float('nan'),
                    per_day={d: float(dv[mb & (base.day.values == int(d))].mean()) if (mb & (base.day.values == int(d))).any() else float('nan') for d in days})
        x = R[g]; T.append(f"  {g:12s} {x['n']:7d} | {x['matched']:6.1%} | {x['true']:+.3f} / {x['retimed']:+.3f} | {x['change']:+.3f} | {x['changed']:.1%}")
    inv_ = R['INVARIANT']; neg = sum(int(v < 0) for v in inv_['per_day'].values() if v == v)
    r970 = ('E2-TIMING-MANUFACTURES-LOSS' if (inv_['change'] <= -1.0 and neg >= 7) else 'E2-TIMING-NEUTRAL' if abs(inv_['change']) <= 0.3 else 'NOT-SHOWN')
    T += ['', f"R970 ARTIFACT: INVARIANT re-timed - true {inv_['change']:+.3f} c pooled; per day " + ', '.join(f'{d} {v:+.2f}' for d, v in inv_['per_day'].items())
          + f' -> {r970}', f'(descriptive; consumed days; replay rule M; {nerr} WS-timed cells scored for R971)']
    R.update(R970=r970)
    open(f'{OUTD}/e2ize.txt', 'w').write('\n'.join(T) + '\n')
    json.dump(R, open(f'{OUTD}/e2ize.json', 'w'), indent=1, default=float)
    print('\n'.join(T), flush=True)


def selftest():
    bad = []
    legs = [(100.0, 50, 5.0, 2), (100.0, 49, 7.0, 2), (300.0, 50, 4.0, 1), (420.0, 48, 9.0, 4)]   # even groups WS, odd E2 (toy_w)
    w0, lr0, g0, s0 = E.toy_w(0, legs); w1, lr1, g1, s1 = E.toy_w(1, [(250.0, 52, 3.0, 2)])
    W = {0: w0, 1: w1}; LR = {0: lr0, 1: lr1 + 100}; SR = {0: s0, 1: s1}
    own = {int(r): float(a) for s in (0, 1) for r, a in zip(LR[s], W[s]['anc'])}
    Wi, LRi, SRi, nr, _ = retime_window(W, LR, SR, own)
    if any(not np.array_equal(Wi[s]['anc'], np.asarray(W[s]['anc'], float)) or Wi[s]['ptl'] != list(W[s]['ptl']) for s in (0, 1)):
        bad.append('re-timing by the own anchors is not the identity')
    ne = {int(LR[0][0]): 450.0, int(LR[0][1]): 450.0, int(LR[0][2]): 50.0, int(LR[1][0]): 260.0}   # the E2 group (row 2) must NOT move
    W2, LR2, SR2, nr, nk = retime_window(W, LR, SR, ne)
    a0 = dict(zip(LR2[0].tolist(), W2[0]['anc'].tolist()))
    if not (a0.get(int(LR[0][0])) == 450.0 and a0.get(int(LR[0][2])) == 300.0 and np.all(np.diff(W2[0]['anc']) >= 0)
            and sorted(LR2[0].tolist()) == sorted(LR[0].tolist()) and not SR2[0][list(LR2[0]).index(int(LR[0][0]))] and nk == 1):
        bad.append(f're-timing: anchors {a0}, sorted {np.all(np.diff(W2[0]["anc"]) >= 0)}, kept WS {nk}')
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
        sys.exit(f'E2IZE SELFTEST FAILED: {bad}')
    m = mutation_check()
    if not all(m.values()):
        sys.exit(f'E2IZE PLANTED ERROR NOT CAUGHT: {m}')
    print(f'e2ize selftests passed; planted errors caught: {m}', flush=True)
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
