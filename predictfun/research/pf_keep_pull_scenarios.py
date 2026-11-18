"""QUEUE-AWARE SCENARIO REPLAY OF THE KEEP / PULL SELECTION UNDER EXECUTION WINDOWS (2026-11-18, session aea5bfb9). USER: "Still have not
cleared all issues, check the review, make sure you clear all issues" (after "we have some issues for simulator, need to correct these
issues first"). The simulator issues it clears: the execution-windows addendum of the d1de7504 review, items 1 and 2 ("The replay uses
placement 58 ms, cancellation 37 ms, pending landing 95 ms and own-fill notice delay 100 ms ... configured assumptions"; "The current interval
audit holds executions and eligibility fixed ... The execution-window implementation must replay the complete neutral order population,
including unfilled orders and explicit excluded/unknown states"), restated as still open by the review of c23e6f29 (queue / fill /
remaining-size / eligibility re-simulation, orders without an M50 fill, activation and notice timing). This replay RE-SIMULATES the
complete neutral JOIN population - every counted state of the committed keep / pull run with a path, filled at M50 or not; path exclusions
stay statuses - through the validated branch kernel under declared execution scenarios. Queue consumption, fills, remaining size, post-only
refusal, structural cancels and the client's eligibility are recomputed in each, and the declared score A+0 is valued against its exact
STATES null in each. Declared before any data is read; consumed Oct 23-27; nothing is fitted or chosen on results.

SCENARIOS (one factor at a time around the replay's baseline B = M50 trade times, activation +58 ms, cancel latency 37 ms, fill notice
 +100 ms; every value is an assumption range, not a measured distribution: "stable in tested scenarios" is the strongest claim)
 TIMELINES  per EVENT (one taker order: all its legs move together; pf_keep_pull_exec_windows.events_of), every trade's execution time is
            M50    anchor - 50 ms (B: it must reproduce the committed replay exactly)
            S1lo / S1mid / S1hi   the start / middle / end of the event's admissible S1 window (the program's declared range AND the REST
                   evidence cap); S2lo / S2hi the same for the measured envelopes; EVhi the evidence's latest instant ((S + 1) * 1000)
            an event without an admissible window (contradictory) is HELD at its M50 time in every timeline - counted and printed, never
            given an invented window.
            M0 / M250   the committed uniform shifts (a constant offset between the trade stamps' clock and the book's), from the
                   committed replay.
 ACTIVATION the order goes live at +25 / +100 ms instead of +58: post-only refusal and queue place at the activation frame, no fill before
            it. The client's decision schedule stays on its receipt clock (first live decision +100 ms, J0 = 2).
 CANCEL     every cancel - the score's and the null's pulls, the structural cancels and the pending landing (activation + latency) - takes
            effect 0 / 100 ms after its decision instead of 37.
 NOTICE     the client learns of a full fill 0 / 250 ms after it instead of 100 (its eligibility for later decisions; fills unchanged).
 INFORMATION never moves with a latent execution time: the fair inputs and the book frames (receipt clock) are the committed ones in every
            scenario; only the trades' execution times, the order's activation and the cancels' effects move.
VALUES per scenario: KEEP (the orders' END values summed), the score's improvement over KEEP, the exact STATES-null expectation, E = their
 difference (pf_keep_pull_exec_windows.contrib at the scenario's latency), PULL-ALL's improvement; fills, shares, orders filled, eligible
 orders; fills DISCOVERED (orders without a baseline fill that fill here) and LOST; held contradictory events.
READINGS (computed; declared before any data is read)
 Z0 CONTROLS, else no output: (a) B reproduces the committed keep / pull run exactly (every order's value, shares, full-fill time and
    post-only status, every M50 fill); (b) the clock-parameterised kernel call equals the validated one at 37 / 95 ms (sampled orders, bit
    for bit); (c) in every timeline every event's legs share one time and every admissible event sits inside its window (held ones at
    their M50 time); (d) every order's frames, END mark and structural decision are the same in every scenario; (e) per-trade fills sum to
    the order's keep branch in every scenario and none falls before activation or at / after the structural landing; (f) the timing legs
    align with the tape; (g) every committed state with a path is re-simulated; (h) B's E and score improvement equal the clock audit's
    per-fill values on the committed tables (+234,279 / 1,169,671 c on the real run).
 Z1 E ACROSS TESTED SCENARIOS: STABLE-POSITIVE-IN-TESTED-SCENARIOS if E > 0 in every scenario, STABLE-NEGATIVE-IN-TESTED-SCENARIOS if E < 0
    in every one, else NOT-STABLE-IN-TESTED-SCENARIOS (min / max printed).
 Z2 KEEP VALUE ACROSS TESTED SCENARIOS and Z3 THE SCORE'S VALUE (KEEP + its improvement): the same predicate.
SELFTEST (synthetic tape, no data): the keep / pull run on its synthetic tape is the committed baseline (control a); planted fills per
 scenario (an adverse sweep moved earlier / later around a structural cancel, a pending-cancel fill under activation +25 / +100 ms and
 cancel latency 0 / 100 ms, a contradictory event held at M50) and planted landings; the notice rule; a one-order outcome at two latencies;
 the report end to end; a misaligned timing source, a committed fill changed under a matching receipt and a perturbed pending clock each
 refused with no output. Mutations: pf_keep_pull_scenarios_mutants.py.
LIMITS: consumed days; one declared rule; one-factor-at-a-time scenarios plus timelines - no search for the joint extreme (value is not
 monotone in timing); assumption ranges, not measured distributions; the queue rule M stays unvalidated on live orders (B1a); STATES is
 local selection; one order at a time, no re-post after a pull.
Usage (run_light, PF_SLOT=claude): pf_keep_pull_scenarios.py selftest | day <day> | report
"""
import contextlib
import ctypes
import io
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_keep_pull as KP                                 # noqa: E402
import pf_keep_pull_clock_audit as A                      # noqa: E402
import pf_keep_pull_timing_audit as V1                    # noqa: E402
import pf_keep_pull_exec_windows as X                     # noqa: E402
import pf_queue_place_test as QP                          # noqa: E402
import pf_queue_place_ledger as QL                        # noqa: E402
import pf_flow_control_kernel as FK                       # noqa: E402

C, pd, I = KP.C, KP.pd, KP.I
OUT = Path(I.K.IO) / 'keep_pull_scenarios_20261118'
SHIFT = 50.
B0 = dict(tl='M50', A=KP.PLACE, L=KP.CANCEL, N=KP.NOTICE)
TIMELINES = ('M50', 'S1lo', 'S1mid', 'S1hi', 'S2lo', 'S2hi', 'EVhi')
NOTICE_RANGE = (0., 250.)
SCEN = ([dict(B0, tl=x) for x in TIMELINES] + [dict(B0, A=a) for a in X.ACT_RANGE] + [dict(B0, L=x) for x in X.CANCEL_RANGE] + [dict(B0, N=x) for x in NOTICE_RANGE]
        + [dict(B0, tl='M0'), dict(B0, tl='M250')])
PEND_OFFSET = MIRROR_OFFSET = 0.                          # planted-error switches (selftest only): the pending clock / the mirror's landing moved


def name(sc):
    return f"{sc['tl']}|A{sc['A']:g}|L{sc['L']:g}|N{sc['N']:g}"


KSCEN = [sc for sc in SCEN if sc['tl'] in TIMELINES and sc['N'] == KP.NOTICE]          # re-simulated by the kernel (11)


def landing_s(d, act, lat):
    """The kernel's landing of a cancel decided d ms after posting: d + lat, or activation + lat if decided before the order is live."""
    d = np.asarray(d, float)
    return np.where(d < act, np.maximum(d + lat, act + lat + PEND_OFFSET), d + lat)


def endj_n(t_full, g_struct, notice):
    """The client's last decision step + 1 (decisions at j < endj): a full fill is known `notice` ms after it, a structural cancel when the
    client decides it (A.endj_of at notice = 100 ms)."""
    lim = np.fmin(g_struct, t_full + notice); d = KP.STEP * np.arange(1, KP.NSTEP + 1)
    return np.where(np.isnan(lim), KP.NSTEP + 1, np.searchsorted(d, lim, side='left') + 1)


def postonly_at(p, side, price, act):
    """The kernel's post-only refusal mirrored at activation `act` (QL.postonly at 58 ms)."""
    f = p['frames']; j = max(int(np.searchsorted(f[:, 0], act, side='right') - 1), 0); opp = f[j, 2 + (1 - side) * 10]
    return bool(0 <= opp <= price) if side == 0 else bool(opp >= price)


def kernel_eval(p, orders, horizon, cancel, pending):
    """QP.evaluate with the kernel's cancel and pending-cancel clocks as parameters (pf_flow_control_kernel.evaluate's checks repeated).
    At 37 / 95 ms it must equal the validated call bit for bit (control b)."""
    fo = np.array([0, len(p['frames'])], np.int64); to = np.array([0, len(p['trades'])], np.int64)
    f = np.ascontiguousarray(p['frames'], float); t = np.ascontiguousarray(p['trades'], float); delta = np.zeros(1); mark = np.array([p['mark']], float)
    o = np.ascontiguousarray(np.asarray(orders, float).reshape(-1, 9)); acts = np.ascontiguousarray(np.eye(len(o)))
    C.require(len(o) > 0 and np.isfinite(o).all() and (o[:, 3] >= 0).all() and np.isin(o[:, 0], [0, 1]).all(), 'invalid branch state')
    nb = no = len(o); out = np.zeros((1, nb, 6)); qty = np.zeros((1, nb, no))
    dp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)); ip = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
    rc = FK.library().evaluate(1, nb, no, ip(fo), ip(to), dp(f), dp(t), dp(delta), dp(mark), dp(o), dp(acts), 0., float(p['mid']), float(horizon),
                               58., float(cancel), float(pending), 3., dp(out), dp(qty))
    C.require(rc == 0, 'branch kernel refused ' + str(rc))
    C.require(np.isfinite(out).all() and np.isfinite(qty).all() and (qty >= -1e-9).all() and (qty <= o[None, None, :, 3] + 1e-7).all(), 'branch quantity bounds')
    sign = 1 - 2 * o[:, 0]
    C.require(np.allclose(out[:, :, 0], (qty * sign * (mark[:, None, None] - o[None, None, :, 2])).sum(2), atol=1e-7), 'branch cash/position/value reconciliation')
    C.require(np.allclose(out[:, :, 2], (qty * sign).sum(2), atol=1e-7), 'branch position reconciliation')
    return out[0, :, 0].copy(), np.array([qty[0, b, b] for b in range(no)])


def fills_g(p, side, price, act, cancel, pending):
    """KP.fills with the order's activation and the kernel's cancel / pending clocks as parameters -> (keep value, keep shares, fill times,
    fill shares): the same order evaluated with its removal between consecutive trades that can fill it."""
    tr = p['trades']; thru = tr[:, 2] < price if side == 0 else tr[:, 2] > price
    can = np.flatnonzero((tr[:, 1] == side) & ((tr[:, 2] == price) | thru)); tt = np.unique(tr[can, 0])
    cut = np.r_[(tt[:-1] + tt[1:]) / 2., tt[-1:] + 1.]
    o = np.tile(np.asarray(QP.order(side, 0, price, act=act), float), (len(cut) + 1, 1)); o[:len(cut), 6] = cut
    v = np.zeros(len(o)); q = np.zeros(len(o))
    for a in range(0, len(o), 72):
        v[a:a + 72], q[a:a + 72] = kernel_eval(p, o[a:a + 72], KP.H, cancel, pending)
    return float(v[-1]), float(q[-1]), tt, np.diff(np.r_[0., q[:-1]])


def loader_timing(data, m, W):
    """The production timing source: per side, the loader's legs of the window in the tape's order (KP.Tape.sources' sort) with their event
    (taker order) and raw observations; must align with the tape's anchors (control f)."""
    out = []
    for s in (0, 1):
        ix = data['LI'].get((m, s), np.zeros(0, np.int64)); l = data['LT'].iloc[ix]
        o_ = np.lexsort(((-l.ptick.values if s == 0 else l.ptick.values), l.anc.values)); l = l.iloc[o_]
        out.append(pd.DataFrame(dict(side=s, mid=m, group=l.group.values.astype(np.int64), how=l.how.values, t_ms=l.t_ms.values.astype(float),
                                     t_e2=l.t_e2.values.astype(float), exec_s=l.exec_s.values.astype(float), anc=l.anc.values.astype(float))))
    return out


def timelines(legs, W, env, day):
    """Per timeline, per side: every leg's execution time in the tape's order - one time per event; an event without an admissible window
    is held at its M50 time. -> (times, held-event counts per scenario family)."""
    for s in (0, 1):
        C.require(np.array_equal(legs[s].anc.values, np.asarray(W[s]['anc'], float)), 'timing legs not aligned with the window tape')
    L = pd.concat([x.assign(pos=np.arange(len(x))) for x in legs], ignore_index=True)
    held = dict(S1=0, S2=0)
    if not len(L):
        return {tl: [np.zeros(0), np.zeros(0)] for tl in TIMELINES}, held
    L = L.assign(day=day, h=env['h'], q01=env['q01'], q99=env['q99']); EVT = X.events_of(L)
    e = pd.MultiIndex.from_frame(EVT[['day', 'group']]).get_indexer(pd.MultiIndex.from_arrays([L.day.values, L.group.values])); C.require(bool((e >= 0).all()), 'a leg without an event')
    m50 = L.anc.values - SHIFT; tau = {'M50': m50}
    for k in ('S1', 'S2'):
        hold = EVT[f'{k}_st'].values >= 2; lo, hi = EVT[f'{k}_lo'].values, EVT[f'{k}_hi'].values; held[k] = int(hold.sum())
        tau[k + 'lo'] = np.where(hold[e], m50, lo[e]); tau[k + 'hi'] = np.where(hold[e], m50, hi[e])
        if k == 'S1':
            tau['S1mid'] = np.where(hold[e], m50, (lo[e] + hi[e]) / 2.)
        ok = hold[e] | ((tau[k + 'lo'] >= lo[e] - 1e-9) & (tau[k + 'hi'] <= hi[e] + 1e-9) & (tau[k + 'lo'] <= tau[k + 'hi'] + 1e-9))
        C.require(bool(ok.all()), f'a {k} time outside its admissible window')
    tau['EVhi'] = EVT.EV_hi.values[e]
    for tl, x in tau.items():                                                  # control (c): one time per event
        g = pd.Series(x).groupby(e); C.require(bool(((g.max() - g.min()).abs() <= 1e-9).all()), f'{tl}: legs of one event at different times')
    out = {}
    for tl, x in tau.items():
        out[tl] = [x[(L.side.values == s)][np.argsort(L.pos.values[L.side.values == s], kind='stable')] for s in (0, 1)]
    return out, held


def scen_order(W, o, tau, fc, rows, frows, st):
    """One committed state in every kernel scenario; B against the committed row and fills (control a); sampled kernel checks (control b)."""
    s, t, p = int(o.side), float(o.t), float(o.p)
    base, bst = QP.build_path(W, t, t + KP.H)
    C.require(base is not None and base['mark'] == o.mark, 'a committed state lost its path or END mark')
    g, land, cause = KP.struct_cancel(base, s, p)
    C.require((g == o.g_struct or (np.isnan(g) and np.isnan(o.g_struct))) and cause == (o.cause if isinstance(o.cause, str) else ''), 'structural decision differs from the committed one')
    if int(o.oid) % 37 == 0:
        oo = [QP.order(s, 0, p)]; a1, b1 = kernel_eval(base, oo, KP.H, KP.CANCEL, KP.PENDING); a2, b2 = QP.evaluate(base, oo, KP.H)
        C.require(np.array_equal(a1, a2) and np.array_equal(b1, b2), 'the clock-parameterised kernel differs from the validated call'); st['kernel_checked'] += 1
    for sc in KSCEN:
        Wk = [dict(W[0], anc=tau[sc['tl']][0]), dict(W[1], anc=tau[sc['tl']][1])]; pk, _ = QP.build_path(Wk, t, t + KP.H, shift=0.)
        C.require(pk is not None and np.array_equal(pk['frames'], base['frames']) and pk['mark'] == base['mark'], 'a scenario moved more than the trades')
        a_, l_ = sc['A'], sc['L']; rej = postonly_at(pk, s, p, a_); vk, qk, tt, fq = fills_g(pk, s, p, a_, l_, a_ + l_ + PEND_OFFSET)
        lnd = float(landing_s(g, a_, l_)) + MIRROR_OFFSET if np.isfinite(g) else np.nan
        C.require(abs(fq.sum() - qk) <= 1e-9 and (fq >= -1e-12).all(), 'per-trade fills do not sum to the keep branch')
        C.require(not (fq[tt < a_] > 1e-12).any() and not (np.isfinite(lnd) and (fq[tt >= lnd] > 1e-12).any()), 'a fill before activation or at / after the structural landing')
        C.require(not rej or (qk == 0 and vk == 0), 'post-only: a refused order filled')
        full = np.flatnonzero(np.cumsum(fq) >= KP.SIZE - 1e-9); tf = float(tt[full[0]]) if len(full) else np.nan
        rows.append((int(o.oid), name(sc), rej, vk, qk, tf, lnd))
        frows.extend((int(o.oid), name(sc), float(x), float(q_)) for x, q_ in zip(tt, fq) if q_ > 1e-12)
        if sc == B0:                                                           # control (a)
            ct, cq = fc; mine = fq > 1e-12
            same = (vk == o.v and qk == o.q and (tf == o.t_full or (np.isnan(tf) and np.isnan(o.t_full))) and rej == (o.status == 'refused_post_only')
                    and np.array_equal(tt[mine], ct) and np.array_equal(fq[mine], cq))
            C.require(same, f'the baseline scenario differs from the committed replay (order {int(o.oid)})'); st['base_matched'] += 1
    st['orders'] += 1


def day_run(day, src=None, timing=None, kp=None, out=None, env_of=None):
    """One day: every committed state with a path, in every kernel scenario -> orders.parquet (order x scenario) + fills.parquet."""
    start = time.time(); C.require(day in KP.DAYS, 'undeclared day'); src = src or KP.Tape(); timing = timing or loader_timing
    kp = Path(kp or KP.OUT); out = Path(out or OUT); d = out / day; C.require(not d.exists(), 'day output exists')
    r = json.loads((kp / day / 'receipt.json').read_text()); C.require(all(C.sha(kp / day / x) == h for x, h in r['outputs'].items()), 'keep-pull outputs changed since their receipt')
    O = pd.read_parquet(kp / day / 'orders.parquet'); Fl = pd.read_parquet(kp / day / 'fills.parquet'); F50 = Fl[Fl.clock == SHIFT]
    fc = {int(k): (g.t.values.astype(float), g.q.values.astype(float)) for k, g in F50.groupby('oid')}; none = (np.zeros(0), np.zeros(0))
    env = (env_of or X.day_env)(day); markets = src.markets(day); offset = src.offset(day)
    Om = O[O.status.isin(('eligible', 'refused_post_only'))]; by_m = {int(m): g for m, g in Om.groupby('m')}
    rows, frows = [], []; st = dict(windows=0, orders=0, base_matched=0, kernel_checked=0, held_events_S1=0, held_events_S2=0)
    for begin in range(0, len(markets), 24):
        data = src.load(day, markets[begin:begin + 24], offset)
        for W in src.pairs(data):
            m = int(W[0]['m']); st['windows'] += 1
            if m not in by_m:
                continue
            tau, held = timelines(timing(data, m, W), W, env, day); st['held_events_S1'] += held['S1']; st['held_events_S2'] += held['S2']
            for o in by_m[m].itertuples():
                scen_order(W, o, tau, fc.get(int(o.oid), none), rows, frows, st)
        del data; C.guard('keep-pull scenarios day')
        print(json.dumps(dict(stage='scenarios', day=day, markets=min(begin + 24, len(markets)), orders=st['orders'], seconds=round(time.time() - start), rss=round(I.K.rss_mib()))), flush=True)
    C.require(st['orders'] == len(Om) == st['base_matched'], f"re-simulated {st['orders']} / baseline-matched {st['base_matched']} of {len(Om)} committed states with a path")
    C.require(st['kernel_checked'] > 0 or not len(Om), 'no kernel clock check ran')
    S = pd.DataFrame(rows, columns=['oid', 'scen', 'refused', 'v', 'q', 't_full', 'land']); SF = pd.DataFrame(frows, columns=['oid', 'scen', 't', 'q'])
    d.mkdir(parents=True); S.to_parquet(d / 'orders.parquet', index=False); SF.to_parquet(d / 'fills.parquet', index=False)
    C.dump(d / 'receipt.json', dict(asof=C.stamp(), day=day, stat=st, scenarios=[name(sc) for sc in KSCEN], env=env, keep_pull_receipt=C.sha(kp / day / 'receipt.json'),
                                    keep_pull_outputs=r['outputs'], provenance=KP.provenance(), source_sha256=C.sha(Path(__file__).resolve()),
                                    dependencies={p_.name: C.sha(p_) for p_ in (Path(X.__file__), Path(QP.__file__), Path(FK.__file__), HERE / 'pf_flow_control_kernel.cpp')},
                                    outputs={x: C.sha(d / x) for x in ('orders.parquet', 'fills.parquet')}, seconds=time.time() - start))
    print(json.dumps(dict(day=day, stat=st, seconds=round(time.time() - start))), flush=True)
    return d


def outcome(O, Os, Fs, sc):
    """One scenario's values. O: the committed orders (states, fair inputs, side, price, mark, structural decision); Os: per order of O the
    scenario's refused / v / q / t_full / land; Fs: its fills (day, oid, t, q)."""
    Os = Os.rename(columns={c: c + '_s' for c in ('refused', 'v', 'q', 't_full', 'land')})
    M = O.merge(Os, on=['day', 'oid'], how='inner', validate='one_to_one'); E = M[~M.refused_s.astype(bool)].reset_index(drop=True)
    cell = E.groupby(['day', 'm', 'side', 'phase'], sort=True).ngroup().values
    T = KP.score_steps(E[[f'f{j:02d}' for j in range(KP.NSTEP + 1)]].values, E.side.values, E.p.values)[KP.DECLARED]
    endj = endj_n(E.t_full_s.values, E.g_struct.values, sc['N']); act, CW = V1.null_weights(T, cell, endj)
    ix = pd.Series(np.arange(len(E)), index=pd.MultiIndex.from_arrays([E.day.values, E.oid.values]))
    ri = ix.reindex(pd.MultiIndex.from_arrays([Fs.day.values, Fs.oid.values])).values; C.require(not np.isnan(ri).any(), 'a fill of an order outside the eligible set')
    ri = ri.astype(int); sg = np.where(E.side.values == 0, 1., -1.); val = Fs.q.values * sg[ri] * (E.mark.values[ri] - E.p.values[ri]); t = Fs.t.values.astype(float)
    lend = np.where(np.isnan(E.land_s.values[ri]), np.inf, E.land_s.values[ri])
    F = dict(val=val, pulled=act[ri], T=T[ri], s=cell[ri], endj=endj[ri], lend=lend)
    e = float(X.contrib(t, F, CW, sc['L']).sum()); r = KP.landing(np.clip(T[ri], 1, KP.NSTEP), sc['L']); score = float(-(val * (act[ri] & (t >= r))).sum())
    j0 = KP.landing(np.full(len(ri), KP.J0), sc['L']); pa = float(-(val * ((KP.J0 < endj[ri]) & (t >= j0))).sum())
    filled = np.bincount(ri, minlength=len(E)) > 0
    return dict(KEEP=float(E.v_s.sum()), score=score, null=score - e, E=e, PULLALL=pa, fills=int(len(Fs)), shares=float(Fs.q.sum()), eligible=int(len(E)),
                orders_filled=int(filled.sum()), pulls=int(act.sum()), by_day={str(d_): float(X.contrib(t[E.day.values[ri] == d_], {k: v[E.day.values[ri] == d_] for k, v in F.items()}, CW,
                                                                                                    sc['L']).sum()) for d_ in sorted(set(E.day))},
                _filled=set(zip(E.day.values[filled], E.oid.values[filled])))


def stable(lo, hi):
    return 'STABLE-POSITIVE-IN-TESTED-SCENARIOS' if lo > 0 else 'STABLE-NEGATIVE-IN-TESTED-SCENARIOS' if hi < 0 else 'NOT-STABLE-IN-TESTED-SCENARIOS'


def report(out=None, kp=None, days=KP.DAYS, expect=(234279., 1169671.), tol=1.):
    """Every scenario's values pooled over the days -> scenarios.txt / scenarios.json; controls (g, h); readings Z1-Z3."""
    start = time.time(); out = Path(out or OUT); kp = Path(kp or KP.OUT); O, Fl, S, SF, st = [], [], [], [], {}
    for d in days:
        r = json.loads((out / d / 'receipt.json').read_text()); C.require(all(C.sha(out / d / x) == h for x, h in r['outputs'].items()), f'{d}: scenario outputs changed since their receipt')
        rk = json.loads((kp / d / 'receipt.json').read_text()); C.require(r['keep_pull_outputs'] == rk['outputs'], f'{d}: scenario day built on other keep-pull outputs')
        O.append(pd.read_parquet(kp / d / 'orders.parquet')); f = pd.read_parquet(kp / d / 'fills.parquet'); f['day'] = d; Fl.append(f)
        s_ = pd.read_parquet(out / d / 'orders.parquet'); s_['day'] = d; S.append(s_); f_ = pd.read_parquet(out / d / 'fills.parquet'); f_['day'] = d; SF.append(f_); st[d] = r['stat']
    O, Fl, S, SF = (pd.concat(x, ignore_index=True) for x in (O, Fl, S, SF)); Op = O[O.status.isin(('eligible', 'refused_post_only'))]; res = {}
    for sc in SCEN:
        nm = name(sc)
        if sc['tl'] in ('M0', 'M250'):                                          # the committed uniform shifts
            tag = KP.CTAG[float(sc['tl'][1:])]
            Os = pd.DataFrame(dict(day=Op.day.values, oid=Op.oid.values, refused=(Op.status == 'refused_post_only').values, v=Op['v' + tag].values, q=Op['q' + tag].values,
                                   t_full=Op['t_full' + tag].values, land=Op.land_struct.values))
            Fs = Fl[Fl.clock == float(sc['tl'][1:])][['day', 'oid', 't', 'q']]
        else:
            key = name(dict(sc, N=KP.NOTICE)); Os = S[S.scen == key][['day', 'oid', 'refused', 'v', 'q', 't_full', 'land']]; Fs = SF[SF.scen == key][['day', 'oid', 't', 'q']]
        C.require(len(Os) == len(Op), f'{nm}: {len(Os)} order rows for {len(Op)} committed states with a path')
        res[nm] = outcome(O, Os, Fs, sc)
    b = res[name(B0)]; Ee = O[O.status == 'eligible'].reset_index(drop=True); cell = Ee.groupby(['day', 'm', 'side', 'phase'], sort=True).ngroup().values
    T = KP.score_steps(Ee[[f'f{j:02d}' for j in range(KP.NSTEP + 1)]].values, Ee.side.values, Ee.p.values)[KP.DECLARED]
    pf = A.per_fill(T, cell, A.endj_of(Ee.t_full.values, Ee.g_struct.values), A.fills_of(Ee, Fl, SHIFT))
    ok_h = (abs(b['E'] - float(pf['score'].sum() - pf['null'].sum())) < 1e-6 and abs(b['score'] - float(pf['score'].sum())) < 1e-6
            and (expect is None or (abs(b['E'] - expect[0]) <= tol and abs(b['score'] - expect[1]) <= tol)))
    if not ok_h:
        sys.exit(f"KEEP-PULL SCENARIOS FAILED -> no output: B's E {b['E']:,.1f} / score {b['score']:,.1f} differ from the clock audit / {expect}")
    bf = b['_filled']
    for nm, v in res.items():
        v['discovered'] = len(v['_filled'] - bf); v['lost'] = len(bf - v['_filled'])
    Es = [v['E'] for v in res.values()]; Ks = [v['KEEP'] for v in res.values()]; Ss = [v['KEEP'] + v['score'] for v in res.values()]
    read = {'Z0 CONTROLS': 'YES', 'Z1 E ACROSS TESTED SCENARIOS': stable(min(Es), max(Es)), 'Z2 KEEP VALUE ACROSS TESTED SCENARIOS': stable(min(Ks), max(Ks)),
            'Z3 THE SCORE\'S VALUE ACROSS TESTED SCENARIOS': stable(min(Ss), max(Ss))}
    L = ['QUEUE-AWARE SCENARIO REPLAY of the keep / pull selection, declared score A+0 (pf_keep_pull_scenarios.py; every committed state with a path re-simulated '
         f'through the branch kernel; days {", ".join(days)}; c summed)',
         f"  {len(Op):,} states re-simulated ({int((Op.status == 'eligible').sum()):,} eligible at B); held contradictory events by day S1 "
         + ', '.join(f"{d} {st[d]['held_events_S1']:,}" for d in days) + '; S2 ' + ', '.join(f"{d} {st[d]['held_events_S2']:,}" for d in days),
         f"  Z0 (h): B's E {b['E']:+,.0f} and score improvement {b['score']:,.0f} = the clock audit's per-fill values on the committed tables", '',
         f"  {'scenario':<24} {'E':>11} {'score':>11} {'null':>11} {'KEEP':>12} {'KEEP+score':>12} {'PULL-ALL':>11} {'eligible':>9} {'fills':>8} {'filled':>7} {'discov.':>7} {'lost':>6}"]
    for nm, v in res.items():
        L.append(f"  {nm:<24} {v['E']:>+11,.0f} {v['score']:>+11,.0f} {v['null']:>+11,.0f} {v['KEEP']:>+12,.0f} {v['KEEP'] + v['score']:>+12,.0f} {v['PULLALL']:>+11,.0f} "
                 f"{v['eligible']:>9,} {v['fills']:>8,} {v['orders_filled']:>7,} {v['discovered']:>7,} {v['lost']:>6,}")
    L += ['', '  E by day: ' + ' | '.join(f"{nm} " + ', '.join(f'{d} {x:+,.0f}' for d, x in v['by_day'].items()) for nm, v in res.items()), '',
          'READINGS (declared before any data was read)'] + [f'  {k}: {v}' for k, v in read.items()] + [
          f"  E ranges over [{min(Es):+,.0f}, {max(Es):+,.0f}], KEEP over [{min(Ks):+,.0f}, {max(Ks):+,.0f}], KEEP + score over [{min(Ss):+,.0f}, {max(Ss):+,.0f}] in the "
          f'{len(res)} tested scenarios', '',
          'LIMITS: consumed days; one declared rule; one-factor-at-a-time scenarios and timelines (no joint extreme search); assumption ranges, not measured '
          'distributions; queue rule M unvalidated on live orders; STATES is local selection; one order at a time, no re-post after a pull.']
    R = {k: {x: y for x, y in v.items() if not x.startswith('_')} for k, v in res.items()}
    (out / 'scenarios.txt').write_text('\n'.join(L) + '\n')
    C.dump(out / 'scenarios.json', dict(asof=C.stamp(), readings=read, scenarios=R, stat=st, days=list(days), provenance=KP.provenance(),
                                        source_sha256=C.sha(Path(__file__).resolve()), seconds=time.time() - start))
    print('\n'.join(L), flush=True)
    return R, read


class SynTiming:
    """The selftest's timing source on the keep / pull synthetic tape: one event per (market, side, anchor); WS legs (source 0) observed at
    their anchor, E2 legs (1) estimated at it, imputed legs (2) only by their second; executedAt = anchor / 1000 s (the REST cap 1 s later)
    except the planted contradictory events, whose cap precedes their windows."""
    def __init__(self, src, contra=()):
        self.src, self.contra, self.ids = src, set(contra), {}

    def __call__(self, data, m, W):
        d = self.src.day(data['day']); out = []
        for s in (0, 1):
            c, anc = d['codes'][m][s]; anc = np.asarray(anc, float); g = np.array([self.ids.setdefault((m, s, a), len(self.ids)) for a in anc], np.int64)
            ex = anc / 1000. - np.array([2. if (m, s, a - d['ws']) in self.contra else 0. for a in anc])
            out.append(pd.DataFrame(dict(side=s, mid=m, group=g, how=(c == 0).astype(int), t_ms=np.where(c == 0, anc, np.nan), t_e2=np.where(c == 1, anc, np.nan), exec_s=ex, anc=anc)))
        return out


def selftest():
    """Synthetic tape only: the keep / pull run on it is the committed baseline; planted scenario fills and landings; the report; refusals."""
    global PEND_OFFSET, MIRROR_OFFSET
    bad = []; src = KP.Syn(); days = KP.DAYS[:2]; env = lambda d_: dict(h=600., q01=-900., q99=1050., inputs={})
    if not (np.array_equal(endj_n(np.array([1000., np.nan, 1000.]), np.array([np.nan, np.nan, 600.]), KP.NOTICE), A.endj_of(np.array([1000., np.nan, 1000.]), np.array([np.nan, np.nan, 600.])))
            and endj_n(np.full(3, 1000.), np.full(3, np.nan), np.array([0., 100., 250.])).tolist() == [20, 22, 25]):
        bad.append('the notice rule (endj) at 0 / 100 / 250 ms')
    if not (np.array_equal(landing_s(np.arange(1, 99) * 50., KP.PLACE, KP.CANCEL), KP.landing(np.arange(1, 99))) and landing_s(np.array([50., 1050.]), 100., 0.).tolist() == [100., 1050.]):
        bad.append('the scenario landing rule')
    with tempfile.TemporaryDirectory(dir=str(I.K.IO)) as tmp:
        tmp = Path(tmp)
        with QL._v1_on(src, tmp):
            for d in days:
                QP.day_run(d)
        with contextlib.redirect_stdout(io.StringIO()):
            for d in days:
                KP.day_run(d, src=src, v1=tmp / 'v1', out=tmp / 'kp')
        tim = SynTiming(src, contra=((108, 0, 150500.),))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                for d in days:
                    day_run(d, src=src, timing=tim, kp=tmp / 'kp', out=tmp / 'sc', env_of=env)
        except (SystemExit, ValueError) as e:
            bad.append(f'synthetic scenario days refused: {e}')
        if not bad:
            O = pd.read_parquet(tmp / 'kp' / days[0] / 'orders.parquet'); SF = pd.read_parquet(tmp / 'sc' / days[0] / 'fills.parquet')
            S = pd.read_parquet(tmp / 'sc' / days[0] / 'orders.parquet'); ws = src.day(days[0])['ws']

            def fl(t_, sc):
                oid = int(O[(O.m == 108) & (O.side == 0) & (O.t == ws + t_)].oid.iloc[0]); x = SF[(SF.oid == oid) & (SF.scen == sc)]
                return [(float(a), float(b)) for a, b in zip(x.t, x.q)]

            def ld(t_, sc):
                oid = int(O[(O.m == 108) & (O.side == 0) & (O.t == ws + t_)].oid.iloc[0]); return float(S[(S.oid == oid) & (S.scen == sc)].land.iloc[0])
            n_ = lambda **kw: name(dict(B0, **kw)); bfill = [(950., 20.), (1070., 10.)]
            exp = {'108 50 000 bid at B': (fl(50000., n_()), bfill), 'S1lo (both trades earlier)': (fl(50000., n_(tl='S1lo')), [(750., 20.), (870., 10.)]),
                   'S1hi (both after the cross cancel)': (fl(50000., n_(tl='S1hi')), []), 'S2lo': (fl(50000., n_(tl='S2lo')), [(400., 20.), (520., 10.)]),
                   'S2hi': (fl(50000., n_(tl='S2hi')), []), 'EVhi': (fl(50000., n_(tl='EVhi')), []), 'activation 25': (fl(50000., n_(A=25.)), bfill),
                   'activation 100': (fl(50000., n_(A=100.)), bfill), 'cancel 0 ms (the 1 070 fill after the landing)': (fl(50000., n_(L=0.)), [(950., 20.)]),
                   'cancel 100 ms': (fl(50000., n_(L=100.)), bfill),
                   'landings 1 087 / 1 050 / 1 150 / 1 087': ([ld(50000., n_()), ld(50000., n_(L=0.)), ld(50000., n_(L=100.)), ld(50000., n_(A=100.))], [1087., 1050., 1150., 1087.]),
                   '200 000 pending-cancel fill at B': (fl(200000., n_()), [(90., 30.)]), '200 000 activation 25 (landing 87)': (fl(200000., n_(A=25.)), []),
                   '200 000 activation 100 (live after the trade)': (fl(200000., n_(A=100.)), []), '200 000 cancel 0 (pending landing 58)': (fl(200000., n_(L=0.)), []),
                   '200 000 cancel 100 (pending landing 158)': (fl(200000., n_(L=100.)), [(90., 30.)]),
                   '200 000 pending landings 95 / 87 / 137 / 58 / 158': ([ld(200000., n_()), ld(200000., n_(A=25.)), ld(200000., n_(A=100.)), ld(200000., n_(L=0.)), ld(200000., n_(L=100.))],
                                                                         [95., 87., 137., 58., 158.]),
                   '200 000 S1lo (before posting) / S1mid (before activation) / S1hi (after landing)': ([fl(200000., n_(tl=x)) for x in ('S1lo', 'S1mid', 'S1hi')], [[], [], []]),
                   '150 000 contradictory event held at M50': ([fl(150000., n_(tl=x)) for x in ('S1lo', 'S1hi', 'S2lo', 'S2hi')], [[(450., 30.)]] * 4)}
            for k, (got, want) in exp.items():
                if got != want:
                    bad.append(f'{k}: {got} != {want}')
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    R, read = report(out=tmp / 'sc', kp=tmp / 'kp', days=days, expect=None)
                n250 = sum(int((pd.read_parquet(tmp / 'kp' / d / 'fills.parquet').clock == 250.).sum()) for d in days)
                if not (read['Z0 CONTROLS'] == 'YES' and len(R) == len(SCEN) and all(k in read for k in ('Z1 E ACROSS TESTED SCENARIOS', 'Z2 KEEP VALUE ACROSS TESTED SCENARIOS'))
                        and (tmp / 'sc' / 'scenarios.json').exists() and R[n_()]['eligible'] > 0 and R[n_(tl='M250')]['fills'] == n250 > 0
                        and R[n_(N=0.)]['fills'] == R[n_()]['fills'] and R[n_(N=0.)]['null'] != R[n_()]['null']):
                    bad.append(f'report: {read}')
            except (SystemExit, ValueError) as e:
                bad.append(f'synthetic report refused: {e}')
            try:                                                               # control (h) must refuse a baseline that is not the committed one
                with contextlib.redirect_stdout(io.StringIO()):
                    report(out=tmp / 'sc', kp=tmp / 'kp', days=days, expect=(R[n_()]['E'] + 1., R[n_()]['score']), tol=1e-6)
                bad.append('report accepted a baseline E that differs from the expectation')
            except (SystemExit, ValueError, TypeError) as e:
                if 'differ from the clock audit' not in str(e):
                    bad.append(f'the wrong expectation refused by another control: {e}')
            O1 = pd.DataFrame(dict(day='d', oid=[0], m=[1], side=[0], phase=[1], p=[50.], mark=[45.], g_struct=[np.nan], **{f'f{j:02d}': [51. if j < 5 else 49.] for j in range(KP.NSTEP + 1)}))
            Os1 = pd.DataFrame(dict(day='d', oid=[0], refused=[False], v=[-150.], q=[30.], t_full=[300.], land=[np.nan])); Fs1 = pd.DataFrame(dict(day='d', oid=[0], t=[300.], q=[30.]))
            o37, o100 = outcome(O1, Os1, Fs1, B0), outcome(O1, Os1, Fs1, dict(B0, L=100.))
            if not (o37['score'] == 150. and o37['pulls'] == 1 and o100['score'] == 0. and o37['KEEP'] == -150.):
                bad.append(f"one-order outcome: score {o37['score']} / {o100['score']} at 37 / 100 ms (want 150 / 0)")
            # refusals: a misaligned timing source, a committed fill changed under a matching receipt, a perturbed pending clock
            def mis(data, m, W):
                x = tim(data, m, W); x[0] = x[0].assign(anc=x[0].anc + 1.) if len(x[0]) else x[0].iloc[:0]; return x
            shutil.copytree(tmp / 'kp', tmp / 'kpb'); fb = pd.read_parquet(tmp / 'kpb' / days[0] / 'fills.parquet'); j_ = int(np.flatnonzero(fb.clock.values == SHIFT)[0])
            fb.loc[fb.index[j_], 'q'] = fb.q.iloc[j_] - 1.; fb.to_parquet(tmp / 'kpb' / days[0] / 'fills.parquet', index=False)
            rb = json.loads((tmp / 'kpb' / days[0] / 'receipt.json').read_text()); rb['outputs']['fills.parquet'] = C.sha(tmp / 'kpb' / days[0] / 'fills.parquet')
            (tmp / 'kpb' / days[0] / 'receipt.json').write_text(json.dumps(rb))
            for nm_, kw, pend, mo, msg in (('a timing source misaligned with the tape', dict(timing=mis, kp=tmp / 'kp'), 0., 0., 'timing legs not aligned'),
                                           ('a committed fill changed under a matching receipt', dict(timing=tim, kp=tmp / 'kpb'), 0., 0., 'the baseline scenario differs'),
                                           ('a pending clock 10 ms off activation + latency', dict(timing=tim, kp=tmp / 'kp'), -10., 0., 'the baseline scenario differs'),
                                           ('a structural landing mirror 10 ms early (a kernel fill after it)', dict(timing=tim, kp=tmp / 'kp'), 0., -10.,
                                            'at / after the structural landing')):
                o_ = tmp / ('z' + str(abs(hash(nm_)) % 1000)); PEND_OFFSET = pend; MIRROR_OFFSET = mo
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        day_run(days[0], src=src, out=o_, env_of=env, **kw)
                    bad.append(f'scenario day accepted {nm_}')
                except (SystemExit, ValueError) as e:
                    if msg not in str(e):
                        bad.append(f'{nm_} refused by another control: {e}')
                finally:
                    PEND_OFFSET = MIRROR_OFFSET = 0.
                if (o_ / days[0]).exists():
                    bad.append(f'a refused scenario day wrote output ({nm_})')
    if bad:
        sys.exit('KEEP-PULL SCENARIOS SELFTEST FAILED: ' + ' | '.join(bad))
    print('keep-pull scenario selftests passed (synthetic only): the notice and landing rules; the keep / pull run on its synthetic tape reproduced at B; planted fills '
          'and landings under timelines S1 / S2 / EV, activation 25 / 100 ms and cancel latency 0 / 100 ms; a contradictory event held at M50; a one-order outcome at '
          '37 / 100 ms; the report end to end and refused on a wrong baseline expectation; a misaligned timing source, a changed committed fill, a perturbed pending clock '
          'and an early landing mirror each refused by its own control with no output', flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif len(a) == 2 and a[0] == 'day':
        day_run(a[1])
    elif a == ['report']:
        report()
    else:
        sys.exit('usage: pf_keep_pull_scenarios.py selftest | day <day> | report')
