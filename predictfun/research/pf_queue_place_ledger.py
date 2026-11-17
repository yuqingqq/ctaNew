"""QUEUE PLACE LEDGER v2: every counted unit of the queue place test gets a row and a status (2026-11-17, session aea5bfb9; USER review:
"the queue receipts count 151,228 causal states, but only 149,174 have saved result rows - 2,054 lack explicit exclusion records. Failed
paths are silently skipped. This needs fixing before treating the comparison as covering the complete population").

DEFECT (pf_queue_place_test.py, v1, frozen - never edited; its outputs stay as provenance): a causal state whose 5 s path cannot be built
is counted and then skipped without a row (lines 240-248); 60 state arm rows lack a 2 s value without a status; arms that are not
feasible (price outside 1..99, IMPROVE at spread < 2), births received after the last decision step, EARLY arms that do not exist and
grid steps without a valid book leave no record; the loader can drop a selected market without a window; the report prints one exclusion.
REPAIR: this file re-walks the same tape with v1's own functions (QP.build_path / evaluate / order / births / keyhash / parity) in v1's
control flow - no estimand, engine or sample changes - and writes a LEDGER (output queue_place_20261117_v2; supersedes v1's accounting):
 markets     one row per selected market: window | no_frames_preds_rows | not_selected_by_loader (+ the window start)
 grid        one row per 100th decision step x side: before_start_margin | expiry_censored | no_known_book | one_sided_book |
             locked_or_crossed | counted (+ the known best / opposite best the step was judged on; + the 5 s and 2 s path status of
             every counted state)
 state_arms  exactly JOIN / REST1 / IMPROVE per counted state; per valuation (5 s, 2 s, M250) a status: eligible | the path status |
             price_out_of_range | spread_lt_2; the 2 s value is kept where the 5 s path fails
 births      one row per new best: before_start_margin | expiry_censored | not_sampled | sampled
 event_arms  exactly FIRST / LATE / EARLY per sampled birth; per valuation a status: eligible | the path status | no_time_before_end |
             seen_after_last_step | gone_before_first | not_applicable_improve_birth | early_no_step | early_no_known_book |
             early_not_behind | early_too_far
 Each eligible arm also carries rej (the order is refused as post-only crossing at its activation frame; valued 0, mirrored from the
 kernel) and event arms the time left to END (an arm with < 58 ms left is never live).
CONTROLS (any failure -> no output):
 (1) conservation - grid rows = grid steps, 3 arm rows per counted state and per sampled birth, one row per birth and per market,
     sampled = min(40, in-window births) per window-side, a value is present iff its status is eligible, every status from the declared
     lists AND equal to its predicate recomputed from the saved columns (grid and births from t / b, the window start and the judged
     book; state arms from best, spread and the state's path status; event arms from b, t, the time left to END, rung and LATE's place);
 (2) reconciliation with v1's saved rows - bit-exact values, statuses and counters on every v1 row, and the v1 gap (counter - saved
     states) must equal the states without an eligible 5 s arm;
 (3) run-time planted errors on the artifact itself must each be flagged: a v1 value moved by 1e-9, a dropped v1 row, a counter + 1,
     a dropped ledger arm row, a dropped ledger grid row, a status without its value, and four the counts cannot see (a swapped grid
     status, a swapped birth status, a moved arm price, a moved time to END - only the predicates of (1) can flag these);
 (4) v1's tape controls repeated (path parity, planted no-shift);
 (5) rej => no fill and value 0 on every row; the mirror itself is checked both ways against the kernel at the start of every day run.
SELFTEST (synthetic tape only): v1's own day_run and report run on a synthetic tape with every status planted, then this file on the same
 tape; planted silent skip (MUT_SILENT_SKIP) must break conservation and leave exactly v1's saved states; post-only mirror both ways
 against the kernel (also at the start of every day run); every status must occur except the path statuses the walk cannot reach
 (declared with reasons in UNREACH_STATE / UNREACH_EVENT: these must NOT occur); planted states and event arms on the scripted market
 must carry the status, decision time, rung and time to END the script was written for; known-bad inputs refused (changed v1 rows,
 changed ledger file, existing output, undeclared day). MUTATION CHECK of the selftest: pf_queue_place_ledger_mutants.py plants code
 errors in a copy of this file one at a time - each must make the selftest fail, the unchanged copy must pass.
READINGS (declared before the run, computed):
 L1 LEDGER-COMPLETE   every conservation identity holds on all five days.
 L2 V1-REPRODUCED     zero reconciliation mismatches on all five days, and v1's eighteen readings recomputed from the ledger are equal.
 L3 GAP-ACCOUNTED     v1 counter - v1 saved states = states without an eligible 5 s arm on every day, listed by status.
 L4 COVERAGE at 2 s   (the only valuation the excluded states can have) COVERAGE-MATTERS-AT-2S if an arm's 2 s mean moves by >= 0.25 c
                      or changes sign between v1's subset (5 s-eligible) and every 2 s-eligible state; else COVERAGE-IMMATERIAL-AT-2S.
 L5 SIGN-ROBUST       per arm (and for the IMPROVE - JOIN pairs): the mean the excluded units would need for the pooled 5 s mean to change
                      sign, beside the most extreme saved value; SIGN-ROBUST-TO-EXCLUDED if it lies beyond that value. The excluded
                      states have no 5 s value by definition: this is a bound, not a measurement.
LIMITS: unchanged from v1 (consumed days Oct23-27; replay fill model; one order at a time held to END; EARLY ex post; FIRST / LATE / EARLY
 differ in exposure as well as queue place - not a measure of queue priority alone; common-endpoint values, not fill + 5 s markouts).
Usage (run_light, PF_SLOT=claude): pf_queue_place_ledger.py selftest | day <day> [limit] | report
"""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_queue_place_test as QP                          # noqa: E402  (v1: frozen; imported, never edited)

F, K, PATH = QP.F, QP.K, QP.PATH
C, pd, I = F.C, F.pd, F.I
V1 = QP.OUT                                               # v1's saved outputs: read only
OUT = Path(I.K.IO) / 'queue_place_20261117_v2'
DAYS, H_MAIN, H_SHORT, LATE_MS, EARLY_MS, PER_SIDE, STATE_EVERY = QP.DAYS, QP.H_MAIN, QP.H_SHORT, QP.LATE_MS, QP.EARLY_MS, QP.PER_SIDE, QP.STATE_EVERY
PLACE_MS = 58.
MUT_SILENT_SKIP = 0                                       # planted error (v1's defect): a counted state with a failed 5 s path leaves no row

MARKET_ST = ('window', 'no_frames_preds_rows', 'not_selected_by_loader')
GRID_ST = ('before_start_margin', 'expiry_censored', 'no_known_book', 'one_sided_book', 'locked_or_crossed', 'counted')
PATH_ST = ('eligible', 'no_known_book', 'invalid_known_book', 'expiry_censored', 'missing_future_book', 'stale_future_book', 'missing_endpoint')
ARM_ST = ('eligible', 'price_out_of_range', 'spread_lt_2')
BIRTH_ST = ('before_start_margin', 'expiry_censored', 'not_sampled', 'sampled')       # order used as a code in window()
NOT_TRIED = ('seen_after_last_step', 'gone_before_first', 'not_applicable_improve_birth', 'early_no_step', 'early_no_known_book', 'early_not_behind', 'early_too_far')
EVENT_ST = PATH_ST + ('no_time_before_end',) + NOT_TRIED
STATE_ARM_ST = PATH_ST + ARM_ST[1:]
ARMS_S, ARMS_E = ('JOIN', 'REST1', 'IMPROVE'), ('FIRST', 'LATE', 'EARLY')
LATE_REL = ('at_or_behind', 'inside', 'no_known_book', 'no_step')       # LATE's price against the known best at its decision step
TAGS = ('', '_2s', '_m250')
VALS = {'': ('v', 'q', 'vy', 'qy', 'rej'), '_2s': ('v_2s', 'q_2s', 'vy_2s', 'qy_2s', 'rej_2s'), '_m250': ('v_m250', 'q_m250')}
ALLV = [c for t in TAGS for c in VALS[t]]
COLS = dict(markets=['day', 'm', 'market_status', 'ws'],
            grid=['day', 'm', 'side', 'k', 't', 'cand_status', 'best', 'opp', 'spread', 'phase', 'path', 'path_2s'],
            state_arms=['day', 'm', 'side', 'k', 't', 'arm', 'rung', 'p', 'spread', 'phase', 'arm_status'] + ['status' + t for t in TAGS] + ALLV,
            births=['day', 'm', 'side', 'bj', 'b', 'p', 'kind', 'birth_status'],
            event_arms=['day', 'm', 'side', 'kind', 'bj', 'b', 'p', 'phase', 'arm', 't', 'rung', 'late_rel', 'ttl'] + ['status' + t for t in TAGS] + ALLV)
NEW_ST = dict(markets=0, windows=0, loader_selected=0, loader_no_frames=0, grid=0, counted=0, births_all=0, sampled=0, first_ok=0,
              parity_checked=0, parity_bad=0, mut_checked=0, mut_differs=0)


def postonly(p, side, price):
    """The kernel's post-only refusal for a new order, mirrored: opposite best at the activation frame (last frame at or before 58 ms)."""
    f = p['frames']; j = max(int(np.searchsorted(f[:, 0], PLACE_MS, side='right') - 1), 0); opp = f[j, 2 + (1 - side) * 10]
    return bool(0 <= opp <= price) if side == 0 else bool(opp >= price)


def valued(pth, side, orders, prices, horizon):
    """v1's two kernel runs (all fills, first-100-ms fills) + the post-only mirror; rej => no fill and value 0 (checked on every order)."""
    v, q = QP.evaluate(pth, orders, horizon); vy, qy = QP.evaluate(pth, orders, horizon, young=True)
    C.require((qy <= q + 1e-9).all(), 'young fills exceed total fills')
    rj = np.array([postonly(pth, side, pr) for pr in prices], float)
    C.require(((q == 0) & (v == 0))[rj > 0].all(), 'post-only mirror: a refused order filled')
    return v, q, vy, qy, rj


def put(rec, tag, n_, vals):
    for c, x in zip(VALS[tag], vals):
        rec[c] = float(x[n_])


def window(W, day, sink, st):
    """One market window: every grid step, birth and arm gets a row and a status; values by v1's functions in v1's control flow."""
    m = int(W[0]['m']); tg = np.asarray(W[0]['tg'], float); kn = np.asarray(W[0]['jk']); ws = W[0]['ws']
    for s in (0, 1):
        sgn = 1 if s == 0 else -1; P5 = np.asarray(W[s]['P5'], float); OP = np.asarray(W[1 - s]['P5'], float)
        # --- births of a new best: the whole population, then v1's sample
        allb = QP.births(W, s)
        bl = [x for x in allb if x[1] >= ws + 2500. and x[1] + H_MAIN <= ws + 298000.]
        bl = sorted(bl, key=lambda x: QP.keyhash(day, m, s, x[1]))[:PER_SIDE]; chosen = {x[0] for x in bl}
        if allb:                                                    # one frame per window-side (the whole birth population is large: no row dicts)
            bj = np.array([x[0] for x in allb], np.int64); bb = np.array([x[1] for x in allb], float)
            code = np.where(~(bb >= ws + 2500.), 0, np.where(~(bb + H_MAIN <= ws + 298000.), 1, np.where(np.isin(bj, sorted(chosen)), 3, 2)))
            sink['births'].append(pd.DataFrame(dict(day=day, m=m, side=s, bj=bj, b=bb, p=np.array([x[2] for x in allb], float), kind=np.array([x[3] for x in allb], object),
                                                    birth_status=np.array(BIRTH_ST, object)[code])))
        st['births_all'] += len(allb); st['sampled'] += len(bl)
        for (j, b, p, kind) in bl:
            base = dict(day=day, m=m, side=s, kind=kind, bj=j, b=b, p=p, phase=min(5, int((b - ws) // 60000) + 1))
            na = 'not_applicable_improve_birth' if kind == 'improve' else None

            def skip(arm, why, base=base, **kw):
                sink['event_arms'].append(dict(base, arm=arm, **{'status' + g: why for g in TAGS}, **kw))
            kf_b = float(W[s]['kf'][j]); kf_ = np.searchsorted(tg, kf_b, side='left')
            if kf_ >= len(tg):
                skip('FIRST', 'seen_after_last_step'); skip('LATE', 'seen_after_last_step'); skip('EARLY', na or 'seen_after_last_step'); continue
            t_f = float(tg[kf_]); jj = int(kn[kf_])
            if jj < 0 or P5[jj, 0] != p:
                skip('FIRST', 'gone_before_first', t=t_f); skip('LATE', 'gone_before_first'); skip('EARLY', na or 'gone_before_first'); continue
            st['first_ok'] += 1
            arms = [('FIRST', t_f, 0), ('LATE', t_f + LATE_MS, None)]; early = na
            if kind == 'deplete':
                ke = int(np.searchsorted(tg, b - EARLY_MS, side='right') - 1)
                if ke < 0:
                    early = 'early_no_step'
                else:
                    t_e = float(tg[ke]); QP.early_ok(b, t_e); je = int(kn[ke])
                    if not (je >= 0 and P5[je, 0] >= 0):
                        early = 'early_no_known_book'
                    else:
                        dist = int(round(sgn * (P5[je, 0] - p)))
                        early = 'early_not_behind' if dist < 1 else 'early_too_far' if dist > 3 else None
                        if early is None:
                            arms.append(('EARLY', t_e, dist))
            if early is not None:
                skip('EARLY', early)
            for arm, t_a, rung in arms:
                rec = dict(base, arm=arm, t=t_a, ttl=b + H_MAIN - t_a)
                if arm == 'LATE':
                    ka = int(np.searchsorted(tg, t_a, side='left'))
                    ja = int(kn[min(ka, len(kn) - 1)]) if ka < len(kn) else -1
                    known = bool(ja >= 0 and P5[ja, 0] >= 0)
                    rung = int(round(sgn * (P5[ja, 0] - p))) if known else 0
                    rec['late_rel'] = 'no_step' if ka >= len(kn) else 'no_known_book' if not known else 'inside' if rung < 0 else 'at_or_behind'
                    rung = max(rung, 0)
                rec['rung'] = rung
                for H, tag in ((H_MAIN, ''), (H_SHORT, '_2s')):
                    if b + H - t_a < 50.:                               # v1: seen too late, no decision time before END
                        rec['status' + tag] = 'no_time_before_end'; continue
                    pth, rec['status' + tag] = QP.build_path(W, t_a, b + H)
                    if pth is None:
                        continue
                    o = [QP.order(s, rung, p)]
                    put(rec, tag, 0, valued(pth, s, o, [p], b + H - t_a))
                    if H == H_MAIN:
                        p250, s250 = QP.build_path(W, t_a, b + H, shift=250.)
                        C.require(p250 is not None and s250 == 'eligible', 'M250 path status differs from the M50 status')
                        put(rec, '_m250', 0, QP.evaluate(p250, o, b + H - t_a))
                rec['status_m250'] = rec['status']
                sink['event_arms'].append(rec)
        # --- causal states: every 100th decision step
        for k in range(0, len(tg), STATE_EVERY):
            t = float(tg[k]); st['grid'] += 1; g = dict(day=day, m=m, side=s, k=k, t=t); jj = int(kn[k])
            best = float(P5[jj, 0]) if jj >= 0 else np.nan; oppb = float(OP[jj, 0]) if jj >= 0 else np.nan
            spread = (oppb - best) if s == 0 else (best - oppb)
            why = ('before_start_margin' if t < ws + 2000. else 'expiry_censored' if t + H_MAIN > ws + 298000. else 'no_known_book' if jj < 0
                   else 'one_sided_book' if best < 0 or oppb < 0 else 'locked_or_crossed' if spread < 1 else 'counted')
            if why != 'counted':
                sink['grid'].append(dict(g, cand_status=why, best=best, opp=oppb, spread=spread)); continue
            st['counted'] += 1; phase = min(5, int((t - ws) // 60000) + 1)
            cands = [('JOIN', 0, best), ('REST1', 1, best - sgn), ('IMPROVE', 0, best + sgn)]
            ast = {a: 'spread_lt_2' if a == 'IMPROVE' and not spread >= 2 else 'price_out_of_range' if not 1 <= pr <= 99 else 'eligible' for a, _, pr in cands}
            live = [c for c in cands if ast[c[0]] == 'eligible']; o = [QP.order(s, r, pr) for _, r, pr in live]; pstat = {}; vals = {}
            for H, tag in ((H_MAIN, ''), (H_SHORT, '_2s')):
                pth, pstat[tag] = QP.build_path(W, t, t + H)
                if pth is None or not live:
                    continue
                vals[tag] = valued(pth, s, o, [c[2] for c in live], H)
                if H == H_MAIN:
                    p250, s250 = QP.build_path(W, t, t + H, shift=250.)
                    C.require(p250 is not None and s250 == 'eligible', 'M250 path status differs from the M50 status')
                    vals['_m250'] = QP.evaluate(p250, o, H)
            if MUT_SILENT_SKIP and pstat[''] != 'eligible':
                continue                                               # planted: v1's silent skip
            sink['grid'].append(dict(g, cand_status='counted', best=best, opp=oppb, spread=spread, phase=phase, path=pstat[''], path_2s=pstat['_2s']))
            for arm, r, pr in cands:
                rec = dict(g, arm=arm, rung=r, p=pr, spread=spread, phase=phase, arm_status=ast[arm])
                n_ = [c[0] for c in live].index(arm) if ast[arm] == 'eligible' else None
                for tag in TAGS:
                    rec['status' + tag] = ast[arm] if n_ is None else pstat['' if tag == '_m250' else tag]
                    if rec['status' + tag] == 'eligible':
                        put(rec, tag, n_, vals[tag])
                sink['state_arms'].append(rec)


def frames(sink):
    fr = {k: pd.DataFrame(sink[k], columns=COLS[k]) for k in COLS if k != 'births'}
    fr['births'] = pd.concat(sink['births'], ignore_index=True)[COLS['births']] if sink['births'] else pd.DataFrame(columns=COLS['births'])
    return fr


def conserve(fr, st):
    """Conservation identities of the ledger -> list of violations (empty: every counted unit has exactly one row and a declared status)."""
    bad = []; MK, G, SA, B, EA = (fr[k] for k in ('markets', 'grid', 'state_arms', 'births', 'event_arms'))

    def need(ok, msg):
        if not bool(ok):
            bad.append(msg)
    need(len(MK) == st['markets'] and not MK.duplicated(['day', 'm']).any() and set(MK.market_status) <= set(MARKET_ST), 'market rows != selected markets')
    n_ = MK.market_status.value_counts()
    need(n_.get('window', 0) == st['windows'] == st['loader_selected'] - st['loader_no_frames'], 'windows != loader selected - loader no-frames')
    need(n_.get('no_frames_preds_rows', 0) == st['loader_no_frames'] and n_.get('not_selected_by_loader', 0) == st['markets'] - st['loader_selected'], 'markets without a window != loader counts')
    kg = ['day', 'm', 'side', 'k']; kb = ['day', 'm', 'side', 'bj']
    need(len(G) == st['grid'] and not G.duplicated(kg).any() and set(G.cand_status) <= set(GRID_ST), 'grid rows != grid steps')
    cnt = G[G.cand_status == 'counted']; need(len(cnt) == st['counted'], 'counted grid rows != counted states')
    need(set(cnt.path) <= set(PATH_ST) and set(cnt.path_2s) <= set(PATH_ST), 'undeclared path status')
    need(len(SA) == 3 * st['counted'], 'state arm rows != 3 x counted states')
    for arm in ARMS_S:
        a = SA[SA.arm == arm]
        need(len(a) == len(cnt) and not a.duplicated(kg).any() and len(a[kg].merge(cnt[kg], on=kg)) == len(cnt), f'{arm}: not exactly one row per counted state')
    need(len(B) == st['births_all'] and not B.duplicated(kb).any() and set(B.birth_status) <= set(BIRTH_ST), 'birth rows != births')
    smp = B[B.birth_status == 'sampled']; need(len(smp) == st['sampled'], 'sampled birth rows != sampled births')
    inw = B[B.birth_status.isin(('sampled', 'not_sampled'))]
    if len(inw):
        c = inw.assign(s=(inw.birth_status == 'sampled').astype(int)).groupby(['day', 'm', 'side']).s.agg(['size', 'sum'])
        need((c['sum'] == np.minimum(PER_SIDE, c['size'])).all(), 'sampled != min(40, in-window births) on a window-side')
    need(len(EA) == 3 * st['sampled'], 'event arm rows != 3 x sampled births')
    for arm in ARMS_E:
        a = EA[EA.arm == arm]
        need(len(a) == len(smp) and not a.duplicated(kb).any() and len(a[kb].merge(smp[kb], on=kb)) == len(smp), f'{arm}: not exactly one row per sampled birth')
    need(int((~EA[EA.arm == 'FIRST'].status.isin(NOT_TRIED)).sum()) == st['first_ok'], 'FIRST rows tried != births with FIRST placeable')
    for T, name, sts in ((SA, 'state', STATE_ARM_ST), (EA, 'event', EVENT_ST)):
        for tag in TAGS:
            s_ = T['status' + tag]; need(s_.notna().all() and set(s_.dropna()) <= set(sts), f'{name} status{tag}: undeclared or missing')
            el = (s_ == 'eligible').values
            for c_ in VALS[tag]:
                need((T[c_].notna().values == el).all(), f'{name} {c_}: value present != status eligible')
        need((T.status_m250 == T.status).all(), f'{name}: status_m250 != status')
    ne = (SA.arm_status != 'eligible').values
    need(set(SA.arm_status) <= set(ARM_ST) and ((SA.status == SA.arm_status) & (SA.status_2s == SA.arm_status)).values[ne].all(), 'state arm: infeasible arm with another status')
    need(not (SA.status.isin(ARM_ST[1:]).values & ~ne).any(), 'state arm: feasible arm with an arm exclusion')
    need((EA.status_2s == EA.status).values[EA.status.isin(NOT_TRIED).values].all() and not EA.status_2s[~EA.status.isin(NOT_TRIED)].isin(NOT_TRIED).any(), 'event arm: not-tried status differs by valuation')
    # --- every status equals its declared predicate, recomputed from the saved columns
    def same(x, y):
        return np.asarray(x, object) == np.asarray(y, object)

    def num(x):
        return np.asarray(x, float)
    if not MK.duplicated(['day', 'm']).any():
        wsm = MK.set_index(['day', 'm']).ws

        def ws_of(T):
            return num(wsm.reindex(pd.MultiIndex.from_frame(T[['day', 'm']])).values)
        w = ws_of(G); t, best, opp, sp = num(G.t), num(G.best), num(G.opp), num(G.spread)
        exp = np.where(t < w + 2000., 'before_start_margin', np.where(t + H_MAIN > w + 298000., 'expiry_censored', np.where(np.isnan(best), 'no_known_book',
                       np.where((best < 0) | (opp < 0), 'one_sided_book', np.where(sp < 1, 'locked_or_crossed', 'counted')))))
        need(not np.isnan(w).any() and same(exp, G.cand_status).all() and (np.isnan(best) | (sp == np.where(num(G.side) == 0, opp - best, best - opp))).all(),
             'grid: a status differs from its predicate (t, ws, best, opp, spread)')
        w = ws_of(B); b = num(B.b)
        exp = np.where(~(b >= w + 2500.), 'before_start_margin', np.where(~(b + H_MAIN <= w + 298000.), 'expiry_censored', 'in_window'))
        need(not np.isnan(w).any() and same(exp, np.where(B.birth_status.isin(('sampled', 'not_sampled')).values, 'in_window', np.asarray(B.birth_status, object))).all(),
             'births: a status differs from its predicate (b, ws)')
    x = SA.merge(cnt[kg + ['best', 'path', 'path_2s']], on=kg, how='left'); sg = np.where(num(x.side) == 0, 1., -1.); f = same(x.arm_status, 'eligible')
    exp = np.where(same(x.arm, 'IMPROVE') & ~(num(x.spread) >= 2), 'spread_lt_2', np.where(~((num(x.p) >= 1) & (num(x.p) <= 99)), 'price_out_of_range', 'eligible'))
    need(len(x) == len(SA) and (num(x.p) == num(x.best) + sg * num(x.arm.map(dict(JOIN=0., REST1=-1., IMPROVE=1.)))).all()
         and (num(x.rung) == num(x.arm.map(dict(JOIN=0., REST1=1., IMPROVE=0.)))).all() and same(exp, x.arm_status).all()
         and same(x.status, x.path)[f].all() and same(x.status_2s, x.path_2s)[f].all(), 'state arm: price, rung, arm status or path status differs from its predicate')
    tr = EA[~EA.status.isin(NOT_TRIED)]; fi, la, ea = (tr[tr.arm == a_] for a_ in ARMS_E)
    need((num(tr.ttl) == num(tr.b + H_MAIN - tr.t)).all() and (same(tr.status, 'no_time_before_end') == (num(tr.ttl) < 50.)).all()
         and (same(tr.status_2s, 'no_time_before_end') == (num(tr.b + H_SHORT - tr.t) < 50.)).all(), 'event arm: time to END or no_time_before_end differs from its predicate (b, t)')
    need((num(fi.rung) == 0).all() and (num(fi.t) >= num(fi.b)).all() and ea.rung.between(1, 3).all() and (num(ea.b - ea.t) >= EARLY_MS).all() and (num(la.rung) >= 0).all()
         and la.late_rel.isin(LATE_REL).all() and (num(la.rung)[~same(la.late_rel, 'at_or_behind')] == 0).all() and tr[tr.arm != 'LATE'].late_rel.isna().all()
         and len(fi) == len(la) and (num(la.set_index(kb).t - fi.set_index(kb).t) == LATE_MS).all(), 'event arm: rung, decision time or LATE place differs from its declaration')
    return bad


V1C = ['day', 'm', 'side', 'kind', 'arm', 'b', 't', 'p', 'phase', 'status', 'v', 'q', 'vy', 'qy', 'v_m250', 'q_m250', 'status_2s', 'v_2s', 'q_2s', 'vy_2s', 'qy_2s', 'spread']
SCOLS = ['p', 'spread', 'phase', 'v', 'q', 'vy', 'qy', 'v_2s', 'q_2s', 'vy_2s', 'qy_2s', 'v_m250']
ECOLS = ['kind', 't', 'phase', 'status', 'status_2s', 'v', 'q', 'vy', 'qy', 'v_2s', 'q_2s', 'vy_2s', 'qy_2s', 'v_m250', 'q_m250']


def _cmp(a, b, key, cols, bad, what):
    """Bit-exact comparison on a unique key (NaN equals NaN); every difference is appended to bad; returns the rows matched."""
    if len(a) == 0 or len(b) == 0:
        if len(a) != len(b):
            bad.append(f'{what}: {len(a)} rows in v1, {len(b)} in the ledger')
        return 0
    if a.duplicated(key).any() or b.duplicated(key).any():
        bad.append(f'{what}: duplicate keys'); return 0
    z = a[key + cols].merge(b[key + cols], on=key, how='outer', suffixes=('_a', '_b'), indicator=True)
    lo, ro = int((z._merge == 'left_only').sum()), int((z._merge == 'right_only').sum())
    if lo or ro:
        bad.append(f'{what}: {lo} rows only in v1, {ro} only in the ledger')
    z = z[z._merge == 'both']
    for c in cols:
        x, y = z[c + '_a'], z[c + '_b']
        if c.startswith('status') or c == 'kind':
            eq = x.astype(object).where(x.notna(), '~').astype(str).values == y.astype(object).where(y.notna(), '~').astype(str).values
        else:
            x, y = x.values.astype(float), y.values.astype(float); eq = (x == y) | (np.isnan(x) & np.isnan(y))
        if not eq.all():
            bad.append(f'{what}: {int((~eq).sum())} rows differ in {c}')
    return int(len(z))


def reconcile(v1, v1stat, fr, st, counters=True):
    """v1's saved rows against the ledger -> (mismatches, summary); any mismatch fails the build. counters=False for a market-limited smoke."""
    bad = []; A = v1.reindex(columns=V1C); SA, EA, G = fr['state_arms'], fr['event_arms'], fr['grid']
    s1 = A[A.kind == 'state']; e1 = A[(A.kind != 'state') & A.b.notna()].copy(); g1 = A[(A.kind != 'state') & A.b.isna()]
    s2 = SA[SA.status == 'eligible']; e2 = EA[~EA.status.isin(NOT_TRIED)].copy(); g2 = EA[(EA.arm == 'FIRST') & (EA.status == 'gone_before_first')]
    ke = ['m', 'side', 'arm', 'b', 'p']; e1['n'] = e1.groupby(ke).cumcount(); e2['n'] = e2.groupby(ke).cumcount()
    out = dict(state_rows=_cmp(s1, s2, ['m', 'side', 'arm', 't'], SCOLS, bad, 'state rows'), event_rows=_cmp(e1, e2, ke + ['n'], ECOLS, bad, 'event rows'), gone_rows=int(len(g1)))
    if not ((g1.arm == 'FIRST') & (g1.status == 'gone_before_first')).all():
        bad.append('v1 rows without a birth time that are not FIRST gone_before_first')

    def tally(g):
        return {(int(a), int(b), str(c)): int(n) for (a, b, c), n in g.groupby(['m', 'side', 'kind']).size().items()} if len(g) else {}
    if tally(g1) != tally(g2):
        bad.append('gone_before_first counts differ by market, side and kind')
    kg = ['m', 'side', 'k']; cnt = G[G.cand_status == 'counted']
    lost = cnt.merge(s2[kg].drop_duplicates().assign(has=1), on=kg, how='left'); lost = lost[lost.has.isna()]
    out['states_without_eligible_5s_arm'] = int(len(lost))
    out['states_without_eligible_5s_arm_by_status'] = {str(k): int(v) for k, v in lost.path.where(lost.path != 'eligible', 'no_feasible_arm').value_counts().items()}
    out['v1_saved_states'] = int(len(s1.drop_duplicates(['m', 'side', 't'])))
    n2 = SA[(SA.status == 'eligible') & (SA.status_2s != 'eligible')]
    out['arm_rows_5s_eligible_2s_not'] = int(len(n2)); out['arm_rows_5s_eligible_2s_not_by_status'] = {str(k): int(v) for k, v in n2.status_2s.value_counts().items()}
    out['v1_state_rows_without_2s_value'] = int(s1.v_2s.isna().sum())
    if out['v1_state_rows_without_2s_value'] != out['arm_rows_5s_eligible_2s_not']:
        bad.append('v1 state rows without a 2 s value != ledger 5 s-eligible arms that are not eligible at 2 s')
    if counters:
        out['v1_counter_states'] = int(v1stat['states']); out['v1_gap'] = out['v1_counter_states'] - out['v1_saved_states']
        if v1stat['states'] != st['counted']:
            bad.append(f'v1 state counter {v1stat["states"]} != ledger counted states {st["counted"]}')
        if v1stat['births'] != st['first_ok']:
            bad.append(f'v1 birth counter {v1stat["births"]} != ledger births with FIRST placeable {st["first_ok"]}')
        if out['v1_gap'] != out['states_without_eligible_5s_arm']:
            bad.append(f'v1 gap {out["v1_gap"]} != ledger states without an eligible 5 s arm {out["states_without_eligible_5s_arm"]}')
    return bad, out


def falsifiers(A, v1stat, fr, st, counters):
    """Planted errors on the artifact itself: each must be flagged by the instrument that is about to certify it."""
    out = {}; a = A.copy(); i = a.index[a.v.notna()][0]; a.loc[i, 'v'] = a.loc[i, 'v'] + 1e-9
    out['v1_value_moved_1e-9'] = bool(reconcile(a, v1stat, fr, st, counters)[0])
    out['v1_row_dropped'] = bool(reconcile(A.drop(index=A.index[A.kind == 'state'][0]), v1stat, fr, st, counters)[0])
    if counters:
        out['v1_counter_plus_1'] = bool(reconcile(A, dict(v1stat, states=v1stat['states'] + 1), fr, st, True)[0])
    f2 = dict(fr); f2['state_arms'] = fr['state_arms'].iloc[1:]; out['ledger_arm_row_dropped'] = bool(conserve(f2, st))
    g = fr['grid']; f2 = dict(fr); f2['grid'] = g.drop(index=g.index[g.cand_status != 'counted'][0]); out['ledger_grid_row_dropped'] = bool(conserve(f2, st))
    sa = fr['state_arms'].copy(); sa.loc[sa.index[sa.status != 'eligible'][0], 'status'] = 'eligible'
    f2 = dict(fr); f2['state_arms'] = sa; out['ledger_status_without_value'] = bool(conserve(f2, st))
    # --- errors the count identities cannot see (the row counts and the status lists stay intact): only the predicates can flag these
    swap = {'before_start_margin': 'expiry_censored', 'expiry_censored': 'before_start_margin'}

    def first(mask, what):
        C.require(bool(mask.any()), f'no row to plant an error on: {what}'); return mask.index[mask.values][0]
    g2 = g.copy(); i = first(g2.cand_status.isin(list(swap)), 'grid status'); g2.loc[i, 'cand_status'] = swap[g2.loc[i, 'cand_status']]
    f2 = dict(fr); f2['grid'] = g2; out['ledger_grid_status_swapped'] = bool(conserve(f2, st))
    b2 = fr['births'].copy(); i = first(b2.birth_status.isin(list(swap)), 'birth status'); b2.loc[i, 'birth_status'] = swap[b2.loc[i, 'birth_status']]
    f2 = dict(fr); f2['births'] = b2; out['ledger_birth_status_swapped'] = bool(conserve(f2, st))
    sa = fr['state_arms'].copy(); sa.loc[first(sa.p.notna(), 'arm price'), 'p'] += 1.
    f2 = dict(fr); f2['state_arms'] = sa; out['ledger_arm_price_moved'] = bool(conserve(f2, st))
    ea = fr['event_arms'].copy(); ea.loc[first(ea.ttl.notna(), 'time to END'), 'ttl'] += 1.
    f2 = dict(fr); f2['event_arms'] = ea; out['ledger_time_to_end_moved'] = bool(conserve(f2, st))
    return out


class Tape:
    """The production source: v1's market list, imputation offset and loader."""
    def markets(self, day):
        return QP.JD.metadata(day, True)[1].mid.astype(int).tolist()

    def offset(self, day):
        return json.loads((Path(C.V.E.OUTD) / f'day_{day}_imp.json').read_text())['untimed']['imp_off_ms']

    def load(self, day, batch, offset):
        return I.load_day(day, set(batch), offset)

    def pairs(self, data):
        return I.pairs(data)


def provenance():
    def git(*a):
        return subprocess.run(('git', '-C', str(HERE)) + a, capture_output=True, text=True).stdout.strip()
    files = [Path(__file__).resolve(), Path(QP.__file__).resolve(), HERE / 'pf_flow_control_kernel.cpp', HERE / 'pf_flow_control_kernel.py']
    return dict(commit=git('rev-parse', 'HEAD'), source={p.name: C.sha(p) for p in files}, uncommitted=git('status', '--porcelain', '--', *map(str, files)))


def day_run(day, limit=None, src=None, v1=None, out=None):
    C.require(day in DAYS, 'undeclared day'); src = src or Tape(); v1 = Path(v1 or V1); out = Path(out or OUT); start = time.time()
    bad = _selftest_postonly(); C.require(not bad, f'post-only mirror differs from the kernel: {bad}')
    dst = out / (f'smoke_{day}' if limit else day); C.require(not dst.exists(), 'day output exists')
    r1 = json.loads((v1 / day / 'receipt.json').read_text()); C.require(C.sha(v1 / day / 'rows.parquet') == r1['outputs']['rows.parquet'], 'v1 rows changed')
    A = pd.read_parquet(v1 / day / 'rows.parquet'); markets = src.markets(day)
    C.require(len(markets) == r1['markets'], 'market list differs from v1'); markets = markets[:limit] if limit else markets
    offset = src.offset(day); sink = {k: [] for k in COLS}; st = dict(NEW_ST)
    for begin in range(0, len(markets), 24):
        batch = markets[begin:begin + 24]; data = src.load(day, batch, offset); seen = set()
        for W in src.pairs(data):
            m = int(W[0]['m']); tg = np.asarray(W[0]['tg'], float); seen.add(m); st['windows'] += 1
            for t in (float(tg[len(tg) // 3]), float(tg[2 * len(tg) // 3])):          # v1's tape controls, repeated
                r = QP.parity(W, t, day)
                if r is None:
                    continue
                st['parity_checked'] += 1; st['parity_bad'] += int(not r[0])
                if r[1] > 0:
                    QP.MUT_NOSHIFT = 1
                    try:
                        st['mut_checked'] += 1; st['mut_differs'] += int(QP.parity(W, t, day)[0] is False)
                    finally:
                        QP.MUT_NOSHIFT = 0
            window(W, day, sink, st)
            print(json.dumps(dict(stage='day', day=day, m=m, state_arms=len(sink['state_arms']), event_arms=len(sink['event_arms']), seconds=round(time.time() - start), rss=round(I.K.rss_mib()))), flush=True)
        st['markets'] += len(batch); st['loader_selected'] += int(data['stat']['selected_windows']); st['loader_no_frames'] += int(data['stat'].get('no_frames_preds_rows', 0))
        sink['markets'] += [dict(day=day, m=m, market_status='window' if m in seen else 'no_frames_preds_rows' if m in data['W0'] else 'not_selected_by_loader', ws=data['W0'].get(m, np.nan)) for m in batch]
        del data; C.guard('queue-place ledger day')
    fr = frames(sink); full = not limit; A = A if full else A[A.m.isin(markets)]
    ctl = bool(st['parity_bad'] or not st['parity_checked'] or not st['mut_checked'] or st['mut_differs'] != st['mut_checked'])
    cons = conserve(fr, st); rec, summ = reconcile(A, r1['stat'], fr, st, full)
    if full:
        rec += [f'tape control counter {k}: v1 {r1["stat"][k]} != ledger {st[k]}' for k in ('parity_checked', 'parity_bad', 'mut_checked', 'mut_differs') if st[k] != r1['stat'][k]]
    planted = falsifiers(A, r1['stat'], fr, st, full) if not (ctl or cons or rec) else {}
    if ctl or cons or rec or not planted or not all(planted.values()):
        sys.exit(f'QUEUE-PLACE LEDGER FAILED on {day} -> no output: controls {st}; conservation {cons}; reconciliation {rec}; planted errors flagged {planted}')
    dst.mkdir(parents=True)
    for k in COLS:
        fr[k].to_parquet(dst / f'{k}.parquet', index=False)
    C.dump(dst / 'receipt.json', dict(asof=C.stamp(), day=day, limit=limit, markets=len(markets), stat=st, conservation_violations=cons, reconciliation_mismatches=rec,
                                      reconciliation=summ, planted_errors_flagged=planted, v1=dict(receipt_asof=r1['asof'], rows=r1['outputs']['rows.parquet'], stat=r1['stat']),
                                      provenance=provenance(), outputs={f'{k}.parquet': C.sha(dst / f'{k}.parquet') for k in COLS}, seconds=time.time() - start))
    print(json.dumps(dict(day=day, stat=st, reconciliation=summ, planted=planted, seconds=round(time.time() - start))), flush=True)


def load(out, v1):
    """The saved ledger (hashes checked, conservation recomputed from the files) and v1's saved rows."""
    L = {k: [] for k in COLS}; R = {}; A = []
    for d in DAYS:
        r = json.loads((out / d / 'receipt.json').read_text()); fr = {}
        for k in COLS:
            C.require(C.sha(out / d / f'{k}.parquet') == r['outputs'][f'{k}.parquet'], 'ledger file changed'); fr[k] = pd.read_parquet(out / d / f'{k}.parquet'); L[k].append(fr[k])
        r['conservation_recomputed'] = conserve(fr, r['stat']); R[d] = r
        r1 = json.loads((v1 / d / 'receipt.json').read_text())
        C.require(C.sha(v1 / d / 'rows.parquet') == r1['outputs']['rows.parquet'] == r['v1']['rows'], 'v1 rows changed'); A.append(pd.read_parquet(v1 / d / 'rows.parquet'))
    return {k: pd.concat(v, ignore_index=True) for k, v in L.items()}, R, pd.concat(A, ignore_index=True)


def to_v1(L):
    """The ledger in v1's row shape (what v1 saved): eligible state arms, tried event arms, FIRST gone rows without a birth time."""
    s = L['state_arms']; s = s[s.status == 'eligible'].assign(kind='state'); e = L['event_arms']
    g = e[(e.arm == 'FIRST') & (e.status == 'gone_before_first')].assign(b=np.nan)
    return pd.concat([e[~e.status.isin(NOT_TRIED)], g, s], ignore_index=True)


def v1_readings(A):
    """v1's eighteen readings (pf_queue_place_test.report: paired / rule) from rows in v1's shape -> {name: (flag, pooled mean)}."""
    ev = A[A.kind != 'state']; stt = A[A.kind == 'state']; out = {}; kev = ['day', 'm', 'side', 'b']; kst = ['day', 'm', 'side', 't']

    def rule(by):
        z = np.concatenate([s.values for s in by.values()]); pooled = float(np.mean(z)) if len(z) else float('nan')
        return bool(pooled > 0 and sum(float(s.mean()) > 0 for s in by.values() if len(s)) >= 4), pooled
    for name, df, a, b, keys in (('F1 FIRST-BEATS-LATE', ev, 'FIRST', 'LATE', kev), ('E1 EARLY-BEATS-FIRST (ex post)', ev[ev.kind == 'deplete'], 'EARLY', 'FIRST', kev),
                                 ('R1 REST1-BEATS-JOIN', stt, 'REST1', 'JOIN', kst), ('R2 IMPROVE-BEATS-JOIN', stt, 'IMPROVE', 'JOIN', kst)):
        d2 = df.assign(v_old=df.v - df.vy)
        for col in ('v', 'v_old'):
            out[f'{name} [{col}]'] = rule({d: (d2[(d2.day == d) & (d2.arm == a)].set_index(keys)[col] - d2[(d2.day == d) & (d2.arm == b)].set_index(keys)[col]).dropna() for d in DAYS})
    for name, df, arm in (('F2 FIRST-PROFITABLE', ev, 'FIRST'), ('E2 EARLY-PROFITABLE (ex post)', ev[ev.kind == 'deplete'], 'EARLY'),
                          ('R3 JOIN-PROFITABLE', stt, 'JOIN'), ('R3 REST1-PROFITABLE', stt, 'REST1'), ('R3 IMPROVE-PROFITABLE', stt, 'IMPROVE')):
        g = df[(df.arm == arm) & (df.status == 'eligible')]; g = g.assign(v_old=g.v - g.vy)
        for col in ('v', 'v_old'):
            out[f'{name} [{col}]'] = rule({d: g[g.day == d][col].dropna() for d in DAYS})
    return out


def vc(s, order):
    c = s.value_counts(); return ' | '.join(f'{k} {int(c[k]):,}' for k in list(order) + [k for k in c.index if k not in order] if k in c.index) or '-'


def bound(inc, n_exc):
    """The mean the excluded units would need for the pooled mean to reach zero, the most extreme saved value on that side, and whether it lies beyond it."""
    if not n_exc or not len(inc):
        return float('nan'), float('nan'), None
    need = -float(inc.sum()) / n_exc; ext = float(inc.max() if need > 0 else inc.min())
    return need, ext, bool(need > ext if need > 0 else need < ext)


def report(v1=None, out=None):
    v1 = Path(v1 or V1); out = Path(out or OUT); L, R, A = load(out, v1); asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    MK, G, SA, B, EA = (L[k] for k in ('markets', 'grid', 'state_arms', 'births', 'event_arms')); cnt = G[G.cand_status == 'counted']
    T = [f'QUEUE PLACE LEDGER v2 (every counted unit has a row and a status; values are v1\'s, reconciled bit for bit); as of {asof}', '',
         'PART A: the ledger (all five days)',
         f'  markets selected {len(MK):,}: {vc(MK.market_status, MARKET_ST)}',
         f'  grid steps (every 100th decision step x side) {len(G):,}: {vc(G.cand_status, GRID_ST)}',
         f'  counted states {len(cnt):,}: 5 s path {vc(cnt.path, PATH_ST)} || 2 s path {vc(cnt.path_2s, PATH_ST)}']
    T += [f'    5 s {a} x 2 s {b}: {int(n):,}' for (a, b), n in cnt[(cnt.path != 'eligible') | (cnt.path_2s != 'eligible')].groupby(['path', 'path_2s']).size().items()]
    T.append('  state arms (one row per arm per counted state): arm | 5 s status | 2 s status | eligible at 5 s but refused as post-only crossing (valued 0)')
    for arm in ARMS_S:
        a = SA[SA.arm == arm]; T.append(f'    {arm:<8} {vc(a.status, STATE_ARM_ST)} || {vc(a.status_2s, STATE_ARM_ST)} || refused {int((a.rej > 0).sum()):,}')
    smp = B[B.birth_status == 'sampled']
    T += [f'  births of a new best {len(B):,}: {vc(B.birth_status, BIRTH_ST)}; sampled by kind: {vc(smp.kind, ("improve", "deplete"))}',
          '  event arms (one row per arm per sampled birth): arm | 5 s status | 2 s status | refused as post-only crossing | eligible with < 58 ms to END (never live)']
    for arm in ARMS_E:
        a = EA[EA.arm == arm]
        T.append(f'    {arm:<6} {vc(a.status, EVENT_ST)} || {vc(a.status_2s, EVENT_ST)} || refused {int((a.rej > 0).sum()):,} | never live {int(((a.status == "eligible") & (a.ttl < PLACE_MS)).sum()):,}')
    la = EA[(EA.arm == 'LATE') & ~EA.status.isin(NOT_TRIED)]
    T.append(f'    LATE price against the known best at its decision: {vc(la.late_rel, ("at_or_behind", "inside", "no_known_book", "no_step"))}')
    T += ['', 'PART B: reconciliation with v1 by day: rows matched state / event / gone | mismatches | v1 counter - v1 saved states = gap | ledger states without an eligible 5 s arm (by status) | '
          '5 s-eligible arm rows not eligible at 2 s | planted errors flagged']
    tot = {}; tot2 = {}
    for d in DAYS:
        r = R[d]; s = r['reconciliation']
        for k, v in s['states_without_eligible_5s_arm_by_status'].items():
            tot[k] = tot.get(k, 0) + v
        for k, v in s['arm_rows_5s_eligible_2s_not_by_status'].items():
            tot2[k] = tot2.get(k, 0) + v
        T.append(f"  {d}: {s['state_rows']:,} / {s['event_rows']:,} / {s['gone_rows']:,} | {len(r['reconciliation_mismatches'])} | {s['v1_counter_states']:,} - {s['v1_saved_states']:,} = {s['v1_gap']:,} | "
                 f"{s['states_without_eligible_5s_arm']:,} ({s['states_without_eligible_5s_arm_by_status']}) | {s['arm_rows_5s_eligible_2s_not']:,} | {sum(r['planted_errors_flagged'].values())} / {len(r['planted_errors_flagged'])}")
    gap = sum(R[d]['reconciliation']['v1_gap'] for d in DAYS)
    T.append(f'  all days: gap {gap:,} = {tot}; 5 s-eligible arm rows without a 2 s value {sum(tot2.values()):,} = {tot2}')
    T += ['', 'PART C: what the excluded states can change (c per order, M50)',
          '  arm | 5 s eligible n, mean | feasible arm excluded at 5 s n | 2 s mean on v1\'s subset (5 s-eligible), n | 2 s mean on every 2 s-eligible state, n | moved | '
          'states excluded at 5 s valued at 2 s n, mean | mean the excluded would need for a zero pooled 5 s mean vs the most extreme saved value']
    read = {}; move = {}; kg = ['day', 'm', 'side', 'k']
    for arm in ARMS_S:
        a = SA[SA.arm == arm]; inc = a[a.status == 'eligible']; exc = a[(a.arm_status == 'eligible') & (a.status != 'eligible')]
        s2v = inc[inc.status_2s == 'eligible']; s2a = a[a.status_2s == 'eligible']; ex2 = exc[exc.status_2s == 'eligible']
        m2v, m2a = float(s2v.v_2s.mean()), float(s2a.v_2s.mean()); need, ext, rob = bound(inc.v, len(exc)); move[arm] = m2a - m2v
        read[f'L4 {arm}'] = bool(abs(m2a - m2v) >= .25 or (m2a > 0) != (m2v > 0)); read[f'L5 {arm}'] = rob
        T.append(f'    {arm:<8} {len(inc):,}, {inc.v.mean():+.3f} | {len(exc):,} | {m2v:+.4f}, {len(s2v):,} | {m2a:+.4f}, {len(s2a):,} | {m2a - m2v:+.4f} | {len(ex2):,}, {ex2.v_2s.mean():+.3f} | {need:+,.1f} vs {ext:+,.1f}')
    for name, x_, y_ in (('R1 REST1 - JOIN', 'REST1', 'JOIN'), ('R2 IMPROVE - JOIN', 'IMPROVE', 'JOIN')):
        x = SA[SA.arm == x_].set_index(kg).sort_index(); y = SA[SA.arm == y_].set_index(kg).sort_index(); C.require(x.index.equals(y.index), 'arm rows not aligned')
        both = (x.arm_status == 'eligible').values & (y.arm_status == 'eligible').values; el = (x.status == 'eligible').values
        d = (x.v - y.v)[both & el]; n_ex = int((both & ~el).sum()); need, ext, rob = bound(d, n_ex); read[f'L5 {name}'] = rob
        T.append(f'    {name}: pairs {len(d):,}, mean {d.mean():+.3f} | pairs excluded at 5 s {n_ex:,} | mean they would need {need:+,.1f} vs the most extreme saved difference {ext:+,.1f}')
    lost = cnt[cnt.path != 'eligible']
    T.append('  counted states without a 5 s path by contract minute: ' + ' | '.join(f'{int(p)}: {int(n):,} of {int((cnt.phase == p).sum()):,}' for p, n in lost.groupby('phase').size().items()))
    T.append('  by day: ' + ' | '.join(f'{d}: {int((lost.day == d).sum()):,} of {int((cnt.day == d).sum()):,}' for d in DAYS))
    j = json.loads((v1 / 'queue_place.json').read_text())['readings']; r1 = v1_readings(A); r2 = v1_readings(to_v1(L))
    impl = set(r1) == set(j) and all(r1[k][0] == j[k] for k in j)
    same = all(r1[k][0] == r2[k][0] and (abs(r1[k][1] - r2[k][1]) < 1e-9 or (np.isnan(r1[k][1]) and np.isnan(r2[k][1]))) for k in r1)
    nm = sum(len(R[d]['reconciliation_mismatches']) for d in DAYS)
    L1 = all(not R[d]['conservation_violations'] and not R[d]['conservation_recomputed'] for d in DAYS); L2 = bool(nm == 0 and impl and same)
    L3 = all(R[d]['reconciliation']['v1_gap'] == R[d]['reconciliation']['states_without_eligible_5s_arm'] for d in DAYS)
    L4 = any(read[f'L4 {a}'] for a in ARMS_S); L5 = {k[3:]: v for k, v in read.items() if k.startswith('L5 ')}
    T += ['', 'READINGS (declared before the run)',
          f'  L1 LEDGER-COMPLETE: {"YES" if L1 else "NO"} (conservation identities at the run and recomputed from the saved files, every day)',
          f'  L2 V1-REPRODUCED: {"YES" if L2 else "NO"} (row mismatches {nm}; v1\'s readings re-implemented here equal v1\'s file: {impl}; recomputed from the ledger equal v1\'s: {same})',
          f'  L3 GAP-ACCOUNTED: {"YES" if L3 else "NO"} ({gap:,} states = {tot})',
          f'  L4 {"COVERAGE-MATTERS-AT-2S" if L4 else "COVERAGE-IMMATERIAL-AT-2S"} (move of the 2 s mean, c per order: ' + ', '.join(f'{a} {move[a]:+.4f}' for a in ARMS_S) + ')',
          '  L5 ' + '; '.join(f'{k}: {"SIGN-ROBUST-TO-EXCLUDED" if v else "NO-EXCLUDED-UNITS" if v is None else "NOT-ROBUST"}' for k, v in L5.items()),
          '', 'LIMITS: as v1 (consumed days; replay fill model, queue M unvalidated on live; one order held to END; EARLY ex post; FIRST / LATE / EARLY differ in exposure as well as queue place; '
          'common-endpoint values, not fill + 5 s markouts). L5 is a bound: the excluded states have no 5 s value by definition.']
    (out / 'queue_place_ledger.txt').write_text('\n'.join(T) + '\n')
    C.dump(out / 'queue_place_ledger.json', dict(asof=asof, L1=L1, L2=L2, L3=L3, L4_coverage_matters_at_2s=L4, L5=L5, gap=gap, gap_by_status=tot, arm_rows_without_2s=tot2,
                                                 move_2s=move, v1_readings_from_ledger={k: v[0] for k, v in r2.items()}, provenance=provenance()))
    print('\n'.join(T), flush=True)
    return dict(L1=L1, L2=L2, L3=L3)


# ---------------------------------------------------------------- synthetic tape (selftest only; no market data is read)
class _Book:
    """A synthetic market in the loader's shape: frames every 200 ms (venue time; received 20 ms later), best bid / ask per frame
    (-1 = empty side), five levels of 100 sh, trades (anchor ms after ws, side whose orders it can fill, price, shares)."""
    def __init__(self, ws, bid, ask, start=-3000., stop=300000.):
        self.ws = ws; self.t = ws + np.arange(start, stop + 1., 200.); n = len(self.t)
        self.b = np.full(n, float(bid)); self.a = np.full(n, float(ask)); self.k = self.t + 20.; self.tr = []

    def _in(self, lo, hi):
        return (self.t >= self.ws + lo) & (self.t < self.ws + hi)

    def book(self, lo, hi, bid=None, ask=None):
        x = self._in(lo, hi)
        if bid is not None:
            self.b[x] = bid
        if ask is not None:
            self.a[x] = ask

    def frame(self, at, bid, ask):                          # one extra frame between the 200 ms frames
        i = int(np.searchsorted(self.t, self.ws + at, side='right'))
        self.t, self.b, self.a, self.k = (np.insert(x, i, v) for x, v in ((self.t, self.ws + at), (self.b, bid), (self.a, ask), (self.k, self.ws + at + 20.)))

    def drop(self, lo, hi):                                 # no frames in [lo, hi): a stale stretch
        x = ~self._in(lo, hi); self.t, self.b, self.a, self.k = self.t[x], self.b[x], self.a[x], self.k[x]

    def stall(self, lo, until):                             # frames from lo on are received no earlier than until
        x = self.t >= self.ws + lo; self.k[x] = np.maximum(self.k[x], self.ws + until)

    def trade(self, at, side, price, shares):
        self.tr.append((self.ws + at, int(side), float(price), float(shares)))

    def sides(self, m, tg=None):
        tg = self.ws + 50. * np.arange(5996) if tg is None else tg; kf = np.maximum.accumulate(self.k)
        jk = np.searchsorted(kf, tg, side='right') - 1; W = {}
        for s in (0, 1):
            best = self.b if s == 0 else self.a; P = best[:, None] + (-1. if s == 0 else 1.) * np.arange(5)[None, :]
            P = np.where((best[:, None] >= 0) & (P >= 1) & (P <= 99), P, -1.).astype(np.int64)
            tr = sorted((x for x in self.tr if x[1] == s), key=lambda x: (x[0], -x[2] if s == 0 else x[2]))
            W[s] = dict(m=m, side=s, ws=self.ws, tf=self.t.copy(), kf=kf, jk=jk, tg=tg, opp=(self.a if s == 0 else self.b).copy(), P5=P, S5=np.where(P >= 0, 100., 0.),
                        anc=np.array([x[0] for x in tr], float), ptl=[x[2] for x in tr], shl=[x[3] for x in tr])
        return W


def _syn_random(ws, rng):
    """101: a random-walk book with random trades - more than 40 births per side (not_sampled), fills, the tape controls' trades."""
    B = _Book(ws, 50, 52); n = len(B.t); mid = np.clip(50 + np.cumsum(rng.choice([-1, 0, 0, 0, 1], n)), 20, 80)
    B.b = mid.astype(float); B.a = (mid + rng.choice([1, 2, 2, 3], n)).astype(float)

    def best(at, s):
        return (B.b if s == 0 else B.a)[int(np.searchsorted(B.t, ws + at, side='right') - 1)]
    for at in np.floor(np.sort(rng.uniform(0., 299000., 500))):
        s = int(rng.integers(2)); B.trade(float(at), s, best(at, s) + (0 if rng.random() < .7 else -1 if s == 0 else 1), float(rng.integers(10, 300)))
    B.trade(100500., 0, best(100500., 0), 200.); B.trade(200500., 1, best(200500., 1), 200.)
    return B


def _syn_scripted(ws):
    """102: one planted status per slot on a constant 50 / 52 book (times in ms after the window start)."""
    B = _Book(ws, 50, 52)
    B.book(1000, 1600, bid=51)                                         # births before the start margin
    B.book(8000, 12000, bid=51)                                        # deplete birth at 12000 with the best one tick above 2 s earlier: EARLY eligible
    B.book(21000, 22000, bid=51)                                       # early_not_behind (the best was p 2 s before the birth)
    B.book(32000, 36000, bid=45)                                       # early_too_far (5 ticks)
    B.book(43600, 44400, bid=-1); B.book(46000, 47000, bid=49)         # early_no_known_book (own side empty 2 s before the birth)
    for at in (56000, 60000):                                          # gone_before_first, twice (another best 10 ms after the birth)
        B.book(at, at + 200, bid=51); B.frame(at + 10, 50, 52)
    B.book(66000, 68000, bid=51); B.frame(66010, 51, -1)               # FIRST invalid_known_book (opposite side empty at its decision)
    B.book(76000, 76200, bid=51)                                       # LATE inside (the best fell back below p)
    B.book(86000, 86200, bid=51); B.book(86200, 86600, bid=-1)         # LATE no_known_book -> invalid_known_book
    B.book(96000, 100000, bid=51); B.stall(96000, 98400)               # 2 s no_time_before_end (birth received 2.4 s late)
    B.book(110800, 112200, ask=-1)                                     # state 110000: missing_endpoint at 2 s only (v1's rows without a 2 s value)
    B.book(123800, 125200, ask=-1)                                     # state 120000: missing_endpoint at 5 s only; state 125000: one_sided_book
    B.book(130800, 135200, ask=-1)                                     # state 130000: missing_endpoint at both
    B.drop(148000, 149200)                                             # state 145000: stale_future_book at 5 s only
    B.drop(155800, 157200)                                             # state 155000: stale_future_book at both
    B.book(164600, 165400, bid=52)                                     # state 165000: locked_or_crossed
    B.book(175000, 176000, bid=48, ask=50)                             # state 175000: bid JOIN / IMPROVE refused as post-only crossing
    B.book(185000, 186000, bid=52, ask=54)                             # state 185000: ask JOIN / IMPROVE refused
    B.book(196000, 196200, bid=51); B.frame(196100, 50, 51)            # FIRST refused (the ask reaches p before the order is live)
    B.book(206000, 210000, bid=51); B.drop(207000, 208200)             # event arms stale_future_book (5 s and 2 s)
    B.book(216000, 222000, bid=51); B.book(219800, 221200, ask=-1)     # event arm missing_endpoint at 5 s
    B.book(236000, 240000, bid=51); B.book(236800, 238200, ask=-1)     # event arm missing_endpoint at 2 s
    B.book(292000, 300001, bid=51); B.stall(292000, 299580)            # 5 s no_time_before_end; LATE has no decision step
    for at, s, px, q in ((25200, 0, 50, 400), (27300, 1, 52, 400), (100500, 0, 50, 400), (151000, 0, 50, 400), (176500, 0, 49, 60), (200400, 1, 52, 400)):
        B.trade(float(at), s, px, q)
    return B


def _syn_day(day):
    """Seven markets: 101-105 with frames, 106 selected without frames, 107 not selected by the loader."""
    ws = 1.7e12 + (int(day) % 100) * 86400000.; rng = np.random.default_rng(int(day)); W = {101: _syn_random(ws, rng).sides(101), 102: _syn_scripted(ws).sides(102)}
    B = _Book(ws, 50, 52, start=7200.); B.trade(30000., 0, 50, 300.); B.trade(100600., 1, 52, 300.); W[103] = B.sides(103)          # grid no_known_book
    B = _Book(ws, 50, 52, stop=206000.); B.book(-3000, 2600, bid=51); B.book(203400, 204600, bid=51); B.trade(102500., 0, 50, 300.)
    W[104] = B.sides(104, tg=ws + 3000. + 50. * np.arange(5936))      # early_no_step (late first step); missing_future_book (tape ends)
    B = _Book(ws, 1, 3); B.book(150000, 300001, bid=97, ask=99); B.book(292000, 300001, bid=96); B.stall(292000, 299780)
    B.trade(50000., 0, 1, 300.); B.trade(250000., 1, 99, 300.); W[105] = B.sides(105)                                              # price_out_of_range; seen_after_last_step
    return dict(ws=ws, W=W)


class Syn:
    """The synthetic source, with the loader's interface (the no-frames counter is filled lazily, as in the loader's generator)."""
    def __init__(self):
        self.days = {}

    def day(self, day):
        if day not in self.days:
            self.days[day] = _syn_day(day)
        return self.days[day]

    def markets(self, day):
        return [101, 102, 103, 104, 105, 106, 107]

    def offset(self, day):
        return 0.

    def load(self, day, batch, offset):
        d = self.day(day); sel = [m for m in sorted(batch) if m != 107]
        return dict(day=day, W0={m: d['ws'] for m in sel}, stat=dict(selected_windows=len(sel), no_frames_preds_rows=0),
                    _w=[d['W'][m] for m in sel if m in d['W']], _nf=sum(m not in d['W'] for m in sel))

    def pairs(self, data):
        yield from data['_w']
        data['stat']['no_frames_preds_rows'] += data['_nf']


@contextlib.contextmanager
def _v1_on(src, tmp):
    """v1's own day_run / report routed to a source and a temp output: its module globals are swapped and restored; v1's file is untouched."""
    keep = {k: getattr(QP, k) for k in ('JD', 'I', 'C', 'OUT')}; imp = tmp / 'imp'; imp.mkdir(parents=True, exist_ok=True)
    for d in DAYS:
        (imp / f'day_{d}_imp.json').write_text(json.dumps(dict(untimed=dict(imp_off_ms=src.offset(d)))))
    QP.JD = types.SimpleNamespace(metadata=lambda day, flag: (None, pd.DataFrame(dict(mid=src.markets(day)))))
    QP.I = types.SimpleNamespace(load_day=src.load, pairs=src.pairs, K=I.K)
    QP.C = types.SimpleNamespace(require=C.require, dump=C.dump, sha=C.sha, stamp=C.stamp, guard=C.guard, V=types.SimpleNamespace(E=types.SimpleNamespace(OUTD=str(imp))))
    QP.OUT = tmp / 'v1'
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        for k, v in keep.items():
            setattr(QP, k, v)


def _walk(src, day):
    """The ledger's walk without tape controls or output (selftest only) -> (frames, counters)."""
    sink = {k: [] for k in COLS}; st = dict(NEW_ST); markets = src.markets(day); data = src.load(day, markets, src.offset(day)); seen = set()
    for W in src.pairs(data):
        seen.add(int(W[0]['m'])); st['windows'] += 1; window(W, day, sink, st)
    st['markets'] = len(markets); st['loader_selected'] = data['stat']['selected_windows']; st['loader_no_frames'] = data['stat']['no_frames_preds_rows']
    sink['markets'] += [dict(day=day, m=m, market_status='window' if m in seen else 'no_frames_preds_rows' if m in data['W0'] else 'not_selected_by_loader', ws=data['W0'].get(m, np.nan)) for m in markets]
    return frames(sink), st


# Path statuses that cannot occur where the walk calls the path builder (the rest must each occur on the synthetic tape):
#  counted state: the grid status already requires a known frame (no_known_book), a two-sided uncrossed book with prices <= 99
#  (invalid_known_book) and t + 5 s <= ws + 298 s (expiry_censored);
#  event arm: its decision is at or after the birth frame is known (no_known_book) and END <= ws + 298 s by the birth filter (expiry_censored).
UNREACH_STATE = ('no_known_book', 'invalid_known_book', 'expiry_censored')
UNREACH_EVENT = ('no_known_book', 'expiry_censored')


def _frame(tv, tk, bid, ask):
    def lv(best, step):
        p = np.array([best + step * i if best >= 0 else -1. for i in range(5)], float); return np.r_[p, np.where(p >= 0, 100., 0.)]
    return np.r_[tv, tk, lv(bid, -1.), lv(ask, 1.)]


def _selftest_postonly():
    """The post-only mirror against the kernel, both ways, on handcrafted paths (a trade through the price fills any live order); a wrong
    mirror must be refused by valued()."""
    bad = []

    def path(second, trade):
        return dict(frames=np.array([_frame(-100., -80., 50, 52), second]), trades=np.array([trade], float), mark=51., mid=51.)
    cases = (('bid, the ask reaches the price at the activation frame', path(_frame(40., 60., 48, 50), (60., 0, 49., 100.)), 0, 50., True),
             ('bid, the ask reaches the price 1 ms after activation', path(_frame(59., 79., 48, 50), (60., 0, 49., 100.)), 0, 50., False),
             ('bid, the opposite side is empty', path(_frame(40., 60., 50, -1), (60., 0, 49., 100.)), 0, 50., False),
             ('ask, the bid reaches the price at the activation frame', path(_frame(40., 60., 52, 54), (60., 1, 53., 100.)), 1, 52., True),
             ('ask, the bid reaches the price 1 ms after activation', path(_frame(59., 79., 52, 54), (60., 1, 53., 100.)), 1, 52., False),
             ('ask, the opposite side is empty', path(_frame(40., 60., -1, 52), (60., 1, 53., 100.)), 1, 52., False))
    for name, p, side, price, refused in cases:
        v, q = QP.evaluate(p, [QP.order(side, 0, price)], 200.); mir = postonly(p, side, price)
        if mir != refused or bool(q[0] == 0) != refused or (refused and v[0] != 0):
            bad.append(f'post-only ({name}): mirror {mir}, kernel filled {q[0]}, expected refused {refused}')
    keep = globals()['postonly']; globals()['postonly'] = lambda *a: True          # a wrong mirror: every order refused
    try:
        try:
            valued(cases[1][1], 0, [QP.order(0, 0, 50.)], [50.], 200.); bad.append('a wrong post-only mirror was accepted')
        except ValueError:
            pass
    finally:
        globals()['postonly'] = keep
    return bad


def _refused(fn):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
    except ValueError:
        return True
    return False


def selftest():
    """Synthetic tape only: v1's own day_run and report, then this file, on the same tape; every reachable status planted; planted errors."""
    import shutil
    start = time.time(); bad = _selftest_postonly(); src = Syn(); d0 = DAYS[0]; ws = src.day(d0)['ws']; W = src.day(d0)['W']
    got = (QP.build_path(W[103], ws + 5000., ws + 10000.)[1], QP.build_path(W[102], ws + 125000., ws + 130000.)[1], QP.build_path(W[102], ws + 294000., ws + 299000.)[1])
    if got != UNREACH_STATE or not set(got) <= set(PATH_ST):
        bad.append(f'the path builder does not return the statuses the walk cannot reach: {got}')
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); v1 = tmp / 'v1'; out = tmp / 'v2'
        with _v1_on(src, tmp):                                         # v1's own code on the synthetic tape
            for d in DAYS:
                QP.day_run(d)
            QP.report()
        with contextlib.redirect_stdout(io.StringIO()):
            for d in DAYS:
                day_run(d, src=src, v1=v1, out=out)
            flags = report(v1=v1, out=out); day_run(d0, limit=3, src=src, v1=v1, out=out)
        L, R, A = load(out, v1); G, SA, B, EA = (L[k] for k in ('grid', 'state_arms', 'births', 'event_arms')); cnt = G[G.cand_status == 'counted']
        if not (flags['L1'] and flags['L2'] and flags['L3']):
            bad.append(f'readings on the synthetic tape: {flags}')
        for name, col, declared, unreach in (
                ('market', L['markets'].market_status, MARKET_ST, ()), ('grid', G.cand_status, GRID_ST, ()), ('5 s path', cnt.path, PATH_ST, UNREACH_STATE),
                ('2 s path', cnt.path_2s, PATH_ST, UNREACH_STATE), ('arm', SA.arm_status, ARM_ST, ()), ('state arm 5 s', SA.status, STATE_ARM_ST, UNREACH_STATE),
                ('state arm 2 s', SA.status_2s, STATE_ARM_ST, UNREACH_STATE), ('birth', B.birth_status, BIRTH_ST, ()), ('event arm 5 s', EA.status, EVENT_ST, UNREACH_EVENT),
                ('event arm 2 s', EA.status_2s, EVENT_ST, UNREACH_EVENT), ('LATE place', EA[(EA.arm == 'LATE') & ~EA.status.isin(NOT_TRIED)].late_rel, LATE_REL, ())):
            seen = set(col.dropna()); miss = sorted(set(declared) - set(unreach) - seen); extra = sorted(seen & set(unreach))
            if miss or extra:
                bad.append(f'{name}: statuses not planted {miss}; declared unreachable but seen {extra}')
        for d in DAYS:
            s = R[d]['reconciliation']; pl = R[d]['planted_errors_flagged']
            if not (s['v1_gap'] > 0 and s['arm_rows_5s_eligible_2s_not'] > 0 and len(pl) == 10 and all(pl.values())):
                bad.append(f'{d}: the synthetic tape does not reproduce the defect, or a planted error passed: {s} {pl}')
        g0 = G[(G.day == d0) & (G.m == 102) & (G.side == 0)].set_index('t'); a0 = SA[(SA.day == d0) & (SA.m == 102)].set_index(['side', 't', 'arm'])
        for t, e in ((110000, ('counted', 'eligible', 'missing_endpoint')), (120000, ('counted', 'missing_endpoint', 'eligible')), (130000, ('counted', 'missing_endpoint', 'missing_endpoint')),
                     (145000, ('counted', 'stale_future_book', 'eligible')), (155000, ('counted', 'stale_future_book', 'stale_future_book')),
                     (125000, ('one_sided_book', None, None)), (165000, ('locked_or_crossed', None, None))):
            r = g0.loc[ws + t]; got = tuple(None if x is None or x != x else x for x in (r.cand_status, r.path, r.path_2s))
            if got != e:
                bad.append(f'planted state {t}: {got} != {e}')
        for s, t, arm, e in ((0, 175000, 'JOIN', 1.), (0, 175000, 'IMPROVE', 1.), (0, 175000, 'REST1', 0.), (1, 175000, 'JOIN', 0.), (1, 185000, 'JOIN', 1.),
                             (1, 185000, 'IMPROVE', 1.), (1, 185000, 'REST1', 0.)):
            if a0.loc[(s, ws + t, arm)].rej != e:
                bad.append(f'planted post-only refusal: side {s} state {t} {arm} rej != {e}')
        f0 = EA[(EA.day == d0) & (EA.m == 102) & (EA.side == 0) & (EA.b == ws + 196000.) & (EA.arm == 'FIRST')]
        if len(f0) != 1 or f0.rej.iloc[0] != 1. or f0.q.iloc[0] != 0:
            bad.append('planted FIRST post-only refusal not recorded')
        e0 = EA[(EA.day == d0) & (EA.m == 102) & (EA.side == 0)].set_index(['b', 'arm'])        # planted event arms: decision time, rung, time to END, statuses, LATE place
        for b_, arm, e in ((12000, 'EARLY', (10000, 1, 7000, 'eligible', 'eligible', None)), (22000, 'EARLY', (None, None, None, 'early_not_behind', 'early_not_behind', None)),
                           (32000, 'EARLY', (None, None, None, 'early_too_far', 'early_too_far', None)), (46000, 'EARLY', (None, None, None, 'early_no_known_book', 'early_no_known_book', None)),
                           (56000, 'FIRST', (56050, None, None, 'gone_before_first', 'gone_before_first', None)), (66000, 'FIRST', (66050, 0, 4950, 'invalid_known_book', 'invalid_known_book', None)),
                           (76000, 'LATE', (76350, 0, 4650, 'eligible', 'eligible', 'inside')), (86000, 'LATE', (86350, 0, 4650, 'invalid_known_book', 'invalid_known_book', 'no_known_book')),
                           (96000, 'FIRST', (98400, 0, 2600, 'eligible', 'no_time_before_end', None)), (96000, 'LATE', (98700, 0, 2300, 'eligible', 'no_time_before_end', 'at_or_behind')),
                           (175000, 'EARLY', (173000, 2, 7000, 'eligible', 'eligible', None)), (206000, 'FIRST', (206050, 0, 4950, 'stale_future_book', 'stale_future_book', None)),
                           (216000, 'FIRST', (216050, 0, 4950, 'missing_endpoint', 'eligible', None)), (236000, 'FIRST', (236050, 0, 4950, 'eligible', 'missing_endpoint', None)),
                           (292000, 'FIRST', (299600, 0, -2600, 'no_time_before_end', 'no_time_before_end', None)),
                           (292000, 'LATE', (299900, 0, -2900, 'no_time_before_end', 'no_time_before_end', 'no_step'))):
            r = e0.loc[(ws + b_, arm)]; got = tuple(None if x is None or x != x else x for x in (r.t - ws, r.rung, r.ttl, r.status, r.status_2s, r.late_rel))
            if got != e:
                bad.append(f'planted event arm {b_} {arm}: {got} != {e}')
        fr, st = _walk(src, d0); globals()['MUT_SILENT_SKIP'] = 1    # planted: v1's defect
        try:
            fm, sm = _walk(src, d0)
        finally:
            globals()['MUT_SILENT_SKIP'] = 0
        saved = len(A[(A.day == d0) & (A.kind == 'state')].drop_duplicates(['m', 'side', 't'])); kept = int((fm['grid'].cand_status == 'counted').sum())
        if conserve(fr, st) or not conserve(fm, sm) or kept != saved or not saved < st['counted']:
            bad.append(f'planted silent skip: clean walk {conserve(fr, st)}, mutant flagged {bool(conserve(fm, sm))}, mutant states {kept} vs v1 saved {saved} of {st["counted"]}')
        shutil.copytree(v1, tmp / 'v1bad'); p = tmp / 'v1bad' / d0 / 'rows.parquet'; x = pd.read_parquet(p); x.loc[x.index[x.v.notna()][0], 'v'] += 1.; x.to_parquet(p, index=False)
        shutil.copytree(out, tmp / 'v2bad'); p = tmp / 'v2bad' / d0 / 'grid.parquet'; x = pd.read_parquet(p).iloc[1:]; x.to_parquet(p, index=False)
        known_bad = dict(day_output_exists=_refused(lambda: day_run(d0, src=src, v1=v1, out=out)), v1_rows_changed=_refused(lambda: day_run(d0, src=src, v1=tmp / 'v1bad', out=tmp / 'x')),
                         ledger_file_changed=_refused(lambda: load(tmp / 'v2bad', v1)), undeclared_day=_refused(lambda: day_run('20261028', src=src, v1=v1, out=out)))
        if not all(known_bad.values()):
            bad.append(f'known-bad input accepted: {known_bad}')
        info = dict(rows={k: int(len(v)) for k, v in L.items()}, v1_gap={d: R[d]['reconciliation']['v1_gap'] for d in DAYS},
                    gap_by_status=R[d0]['reconciliation']['states_without_eligible_5s_arm_by_status'], rows_without_2s=R[d0]['reconciliation']['arm_rows_5s_eligible_2s_not_by_status'],
                    planted_flagged=R[d0]['planted_errors_flagged'], silent_skip=dict(mutant_states=kept, v1_saved=saved, counted=st['counted']), known_bad_refused=known_bad)
    if bad:
        sys.exit('QUEUE-PLACE LEDGER SELFTEST FAILED -> no output:\n  ' + '\n  '.join(bad))
    print('queue-place ledger selftests passed (synthetic tape only; v1 reproduced on it bit for bit; every reachable status planted): '
          + json.dumps(C.clean(dict(info, seconds=round(time.time() - start)))), flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        selftest()
    elif cmd == 'day':
        day_run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
    elif cmd == 'report':
        report()
    else:
        sys.exit('usage: pf_queue_place_ledger.py selftest | day <day> [limit] | report')
