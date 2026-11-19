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
REVIEW OF 1fd77004 (reviews/automatic/20261118_1fd77004/report.md; USER: "it still has time disagreements, and the issues are still there,
resolve these issues, all timestamp related issues, correct these ambiguity"), v2 of this replay before any further data is read: (1) the S3
semantics windows (pf_keep_pull_exec_windows.sem_windows: REST executedAt is the Confirmed second, a match up to 10 s earlier; a WS leg's
window spans every value-consistent pending increment) enter as timelines S3lo / S3hi; (2) every scenario carries its FAMILY - timeline,
trade-clock shift (M0 / M250: every trade stamp moved uniformly), activation delay, cancel latency (L0 / L100), fill-notice delay - and
no scenario is called "zero latency"; (3) the held contradictory events get their denominator (every event per family and day) and their
economic weight (the fills, shares and value at held trade instants per scenario); v1's outputs (keep_pull_scenarios_20261118) stay as
provenance.
v6.1 (2026-11-19, USER "can we run the one day test now" -> the score variants against A+0): outcome() takes the score policy as a parameter
(default the declared A+0; any other of KP.POLICIES; an undeclared one is refused) so pf_keep_pull_variants.py can value every declared policy
from the same day outputs. Day outputs, report and A+0 values are unchanged (the variants module checks A+0 against this report's output).
C1 B1a-WEIGHTED (v6, the readiness loop, added after v5's age bins were read - a checklist report item, not a test): FINDINGS C1 asks
for the headline B1a-weighted beside it. Per scenario E weighted by bin with FV3's B1a precision weights (pf_fv3_check.B1A: the share of
replay fills that were real on live's own orders by age bin, at M 0 and at M 50; one day of live's orders), both weight sets as the earlier
modules print them; reading C1 B1a-WEIGHTED E AT B: HOLDS-SIGN iff both weighted values have B's sign (non-zero), else SIGN-DIFFERS.
C12 / C2 (v6, the readiness loop; declared before v6 ran): every fill carries its trade-anchor source as the committed run assigns it
(KP.trade_index + KP.window's rule: the codes of the candidate trades at the fill's instant, 3 when several), the age table is split by it
(C12), control (j): at B the sources equal the committed fills' exactly; reading C2 E AT M250 vs M50 and C2 KEEP + SCORE AT M250 vs M50:
HOLDS-AT-M250 iff the sign at M250 equals B's, else SIGN-DIFFERS-AT-M250.
C1 / C3 (v5, USER "proceed to fix" with review 8e63f178; declared before v5 ran): the replay reported neither FINDINGS C1 (value split by the
order's age at the fill; the headline without fills on orders < 100 ms old, which were 8-12% real on live's own orders - B1a) nor C3 (the
cancel-pending share). Per scenario now: E, the score's improvement and the fills' value by age bin (< 100 / 100-500 / >= 500 ms after the
scenario's activation; E is additive over fills), the headline without the < 100 ms fills, and the fills landing between the score's pull
decision and its landing (count, shares, value, share of the fills' gross). Control (i): KEEP equals the sum of its fills' values in every
scenario, else no output. Readings Z1-NY / Z3-NY: E and KEEP + score without the < 100 ms fills, STABLE-* / NOT-STABLE as Z1 / Z3. B1a-
weighted values were not computed in v5; v6 computes them (below). Scenario values are otherwise unchanged from v4.
SETTLEMENT-LAG CORRECTION (v4, review 52c45d57): UUID creation lag is event-specific, not a constant offset. The default ST window
has no finite lower bound. STlo is omitted from the default scenario bank, and no -infinity endpoint may enter a replay. SThi remains
an upper-endpoint sensitivity scenario, conditional on creation after matching and aligned clocks; it is not a measured match instant.
The activation / notice / cancel-mixture values inherited from v3 remain explicit, posthoc sensitivity choices; their previous description
as measured latency distributions is withdrawn. Earlier scenario receipts are refused by the new reporter until re-simulated.

SCENARIOS (one factor at a time around the replay's baseline B = M50 trade times, activation +58 ms, cancel latency 37 ms, fill notice
 +100 ms; every value is an assumption range, not a measured distribution: "stable in tested scenarios" is the strongest claim)
 TIMELINES  per EVENT (one taker order: all its legs move together; pf_keep_pull_exec_windows.events_of), every trade's execution time is
            M50    anchor - 50 ms (B: it must reproduce the committed replay exactly)
            S1lo / S1mid / S1hi   the start / middle / end of the event's admissible S1 window (the program's declared range AND the REST
                   evidence cap); S2lo / S2hi the same for the measured envelopes; S3lo / S3hi the same for the semantics windows; EVhi the
                   evidence's latest instant ((S + 1) * 1000)
            an event without an admissible window (contradictory) is HELD at its M50 time in every timeline - counted and printed, never
            given an invented window; per family the held events, every event (the denominator) and the fills at held instants (count,
            shares, value) are printed.
 TRADE-CLOCK SHIFT M0 / M250   the committed uniform shifts of every trade stamp (a constant offset between the trade stamps' clock and the
                   book's; M50 is the baseline's), from the committed replay - not an order latency.
 ACTIVATION the order goes live at +15 / +100 ms instead of +58: post-only refusal and queue place at the activation frame, no fill before
            it. The client's decision schedule stays on its receipt clock (first live decision +100 ms, J0 = 2).
 CANCEL     every cancel - the score's and the null's pulls, the structural cancels and the pending landing (activation + latency) - takes
            effect 0 / 100 ms after its decision instead of 37, or by the mixture (a fixed 13% of the orders at 4 s). The kernel orders a grid
            instant as its trades up to it, then its decisions:
            a structural cancel decided at g never stops a trade at g (at zero latency it acts just after g); the pulls keep the replay's
            at-or-after rule - a tie convention at one instant, not evidence (C20).
 NOTICE     the client learns of a full fill 160 / 660 ms after it instead of 100 (its eligibility for later decisions; fills unchanged).
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
 monotone in timing); assumption ranges, not measured distributions (K5 on live's own orders found cancel acks up to ~6 s); the tie order at
 one instant (trades before decisions in the kernel, at-or-after for the pulls) is a convention; the queue rule M stays unvalidated on live orders (B1a); STATES is
 local selection; one order at a time, no re-post after a pull.
Usage (run_light, PF_SLOT=claude): pf_keep_pull_scenarios.py selftest | day <day> | report [day]
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
import pf_fv3_check as FV                                # noqa: E402  FINDINGS C1's B1a precision weights by age bin (M 0 / M 50)
import pf_keep_pull as KP                                 # noqa: E402
import pf_keep_pull_clock_audit as A                      # noqa: E402
import pf_keep_pull_timing_audit as V1                    # noqa: E402
import pf_keep_pull_exec_windows as X                     # noqa: E402
import pf_queue_place_test as QP                          # noqa: E402
import pf_queue_place_ledger as QL                        # noqa: E402
import pf_flow_control_kernel as FK                       # noqa: E402

C, pd, I = KP.C, KP.pd, KP.I
OUT = Path(I.K.IO) / 'keep_pull_scenarios_v6_20261118'      # v6: C12 (age by trade-anchor source) and the C2 reading at M250; v5 and earlier stay as provenance
SRC_LABELS = ('WS', 'E2', 'imputed', 'mixed')              # KP.window's trade-source codes 0 / 1 / 2 / 3 (several sources at one instant)
AGE_EDGES, AGE_LABELS = (KP.YOUNG, 500.), ('lt100', '100to500', 'ge500')   # FINDINGS C1: the order's age at the fill (fill - activation), ms
SHIFT = 50.
B0 = dict(tl='M50', A=KP.PLACE, L=KP.CANCEL, N=KP.NOTICE, X=None)
TIMELINES = ('M50', 'S1lo', 'S1mid', 'S1hi', 'S2lo', 'S2hi', 'S3lo', 'S3hi', 'SThi', 'EVhi')
TIMELINE_RULES = dict(M50='anchor - 50 ms (the committed replay)', S1lo='start of the admissible S1 window', S1mid='middle of the admissible S1 window',
                      S1hi='end of the admissible S1 window', S2lo='start of the admissible S2 window', S2hi='end of the admissible S2 window',
                      S3lo='start of the admissible S3 (semantics) window', S3hi='end of the admissible S3 (semantics) window',
                      SThi='UUID-creation upper endpoint under the after-match/aligned-clock premises; no finite earliest endpoint is identified',
                      EVhi='the REST evidence cap (the latest admissible instant)', held='an event without an admissible window keeps its M50 time')
NOTICE_RANGE = (160., 660.)                               # explicit sensitivity values inherited from v3; not measured notice bounds
ACT_TEST = (15., 100.)                                    # explicit sensitivity values; base acknowledgment evidence only establishes <=21 ms for one order
CANCEL_MIX = (0.13, 4000.)                                # explicit sensitivity mixture inherited from v3, not an estimated frequency or latency
SCEN = ([dict(B0, tl=x) for x in TIMELINES] + [dict(B0, A=a) for a in ACT_TEST] + [dict(B0, L=x) for x in X.CANCEL_RANGE] + [dict(B0, N=x) for x in NOTICE_RANGE]
        + [dict(B0, X=CANCEL_MIX), dict(B0, tl='M0'), dict(B0, tl='M250')])
HOLD_FAMILIES = ('S1', 'S2', 'S3', 'ST')                  # the timelines whose contradictory events are held at M50
FAMILIES = ('baseline', 'timeline', 'trade-clock shift', 'activation delay', 'cancel latency', 'fill-notice delay')
PEND_OFFSET = MIRROR_OFFSET = 0.                          # planted-error switches (selftest only): the pending clock / the mirror's landing moved


def name(sc):
    x = f"+{100 * sc['X'][0]:g}%x{sc['X'][1]:g}" if sc.get('X') else ''
    return f"{sc['tl']}|A{sc['A']:g}|L{sc['L']:g}{x}|N{sc['N']:g}"


def slow_order(oid, share):
    """The mixture's slow orders: a fixed hash of the order id below `share` (deterministic, independent of the order's state)."""
    return ((np.asarray(oid, np.int64) * 2654435761) % 2 ** 32) / 2. ** 32 < share


def lat_of(sc, oid):
    """An order's cancel latency in scenario sc (the mixture's slow orders take its slow latency)."""
    return float(sc['X'][1]) if sc.get('X') and bool(slow_order(oid, sc['X'][0])) else float(sc['L'])


def family(sc):
    """A scenario's family and plain description (one factor moved from the baseline B)."""
    if sc == B0:
        return 'baseline', 'B: trades at anchor - 50 ms, activation +58 ms, cancel latency 37 ms, fill notice +100 ms'
    if sc['tl'] in ('M0', 'M250'):
        return 'trade-clock shift', f"every trade stamp {sc['tl'][1:]} ms before its anchor (B: 50 ms) - a uniform clock offset, not an order latency"
    if sc['tl'] != 'M50':
        return 'timeline', f"every trade at the {TIMELINE_RULES[sc['tl']]}; contradictory events held at M50"
    if sc.get('X'):
        return 'cancel latency', (f"{100 * sc['X'][0]:g}% of the orders' cancels take effect {sc['X'][1]:g} ms after their decision, the rest {sc['L']:g} ms (own "
                                  'assumed mixture; the former K8-based measured-tail interpretation is withdrawn)')
    if sc['A'] != B0['A']:
        return 'activation delay', f"the order goes live {sc['A']:g} ms after posting (B: 58)"
    if sc['L'] != B0['L']:
        return 'cancel latency', f"every cancel takes effect {sc['L']:g} ms after its decision (B: 37)"
    return 'fill-notice delay', f"the client learns of a full fill {sc['N']:g} ms after it (B: 100)"


KSCEN = [sc for sc in SCEN if sc['tl'] in TIMELINES and sc['N'] == KP.NOTICE]          # re-simulated by the kernel


def landing_s(d, act, lat):
    """The kernel's landing of a cancel decided d ms after posting: d + lat, or activation + lat if decided before the order is live."""
    d = np.asarray(d, float)
    return np.where(d < act, np.maximum(d + lat, act + lat + PEND_OFFSET), d + lat)


def eff_landing(g, act, lat):
    """A structural cancel's effect on fills in the kernel's order of a grid instant (its trades up to it first, then its decisions): a cancel
    decided at g never stops a trade at g, so a landing at or before g (zero latency) acts just after g."""
    x = float(landing_s(g, act, lat))
    return x if x > g else float(np.nextafter(g, np.inf))


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
                                     t_e2=l.t_e2.values.astype(float), exec_s=l.exec_s.values.astype(float), anc=l.anc.values.astype(float),
                                     row_id=l.row_id.values.astype(np.int64))))
    return out


def timelines(legs, W, env, day, sem=None, tset=None, lag=None):
    """Per timeline, per side: every leg's execution time in the tape's order - one time per event; an event without an admissible window
    is held at its M50 time. S3 leg windows come with the legs (S3_llo / S3_lhi) or from the day's sem_windows by row_id. -> (times, held
    events per family, events, per timeline and side the sorted held / free leg times)."""
    for s in (0, 1):
        C.require(np.array_equal(legs[s].anc.values, np.asarray(W[s]['anc'], float)), 'timing legs not aligned with the window tape')
    L = pd.concat([x.assign(pos=np.arange(len(x))) for x in legs], ignore_index=True)
    held = {k: 0 for k in HOLD_FAMILIES}; empty = (np.zeros(0), np.zeros(0))
    if not len(L):
        return {tl: [np.zeros(0), np.zeros(0)] for tl in TIMELINES}, held, 0, {tl: [empty, empty] for tl in TIMELINES}
    if 'S3_llo' not in L:
        C.require(sem is not None and bool(np.isin(L.row_id.values, sem.index.values).all()), 'legs without their S3 windows')
        L['S3_llo'], L['S3_lhi'] = sem.S3_llo.reindex(L.row_id.values).values, sem.S3_lhi.reindex(L.row_id.values).values
    if 'ST_llo' not in L:
        C.require(tset is not None and bool(np.isin(L.row_id.values, tset.index.values).all()), 'legs without their settlement times')
        L['ST_llo'], L['ST_lhi'], _ = X.st_windows(tset.t_set_ms.reindex(L.row_id.values).values, L.S3_llo.values, L.S3_lhi.values, lag)
    L = L.assign(day=day, h=env['h'], q01=env['q01'], q99=env['q99']); EVT = X.events_of(L)
    e = pd.MultiIndex.from_frame(EVT[['day', 'group']]).get_indexer(pd.MultiIndex.from_arrays([L.day.values, L.group.values])); C.require(bool((e >= 0).all()), 'a leg without an event')
    m50 = L.anc.values - SHIFT; tau = {'M50': m50}; hl = {'M50': np.zeros(len(L), bool)}
    for k in HOLD_FAMILIES:
        hold = EVT[f'{k}_st'].values >= 2; lo, hi = EVT[f'{k}_lo'].values, EVT[f'{k}_hi'].values; held[k] = int(hold.sum())
        tau[k + 'lo'] = np.where(hold[e], m50, lo[e]); tau[k + 'hi'] = np.where(hold[e], m50, hi[e]); hl[k + 'lo'] = hl[k + 'hi'] = hold[e]
        if k == 'ST' and not np.isfinite(lo).all():
            del tau['STlo'], hl['STlo']  # No earliest execution can be manufactured from an upper-only window.
        if k == 'S1':
            tau['S1mid'] = np.where(hold[e], m50, (lo[e] + hi[e]) / 2.); hl['S1mid'] = hold[e]
        lower = tau.get(k + 'lo', lo[e])
        ok = hold[e] | ((lower >= lo[e] - 1e-9) & (tau[k + 'hi'] <= hi[e] + 1e-9) & (lower <= tau[k + 'hi'] + 1e-9))
        C.require(bool(ok.all()), f'a {k} time outside its admissible window')
    tau['EVhi'] = EVT.EV_hi.values[e]; hl['EVhi'] = np.zeros(len(L), bool)
    for tl, x in tau.items():                                                  # control (c): one time per event
        C.require(bool(np.isfinite(x).all()), f'{tl}: no finite execution endpoint is identified')
        g = pd.Series(x).groupby(e); C.require(bool(((g.max() - g.min()).abs() <= 1e-9).all()), f'{tl}: legs of one event at different times')
    out, hm = {}, {}
    for tl, x in tau.items():
        out[tl] = [x[(L.side.values == s)][np.argsort(L.pos.values[L.side.values == s], kind='stable')] for s in (0, 1)]
        hm[tl] = [(np.sort(x[(L.side.values == s) & hl[tl]]), np.sort(x[(L.side.values == s) & ~hl[tl]])) for s in (0, 1)]
    return out, held, int(len(EVT)), hm


def held_code(x, times):
    """Per fill time x (absolute): 1 only held legs trade at it, 2 only free ones, 3 both, 0 none (times: sorted held / free leg times)."""
    def at(t_):
        if not len(t_):
            return np.zeros(len(x), bool)
        j = np.clip(np.searchsorted(t_, x), 1, len(t_) - 1) if len(t_) > 1 else np.zeros(len(x), int)
        return (np.abs(t_[j] - x) <= 1e-3) | (np.abs(t_[np.maximum(j - 1, 0)] - x) <= 1e-3)                 # (tau - t) + t round-trips to ~1e-4 ms
    return at(times[0]).astype(np.int8) + 2 * at(times[1]).astype(np.int8)


def scen_order(W, o, tau, fc, rows, frows, st, hm, codes):
    """One committed state in every kernel scenario; B against the committed row and fills (control a); sampled kernel checks (control b);
    each fill marked by the trades at its instant: held (contradictory events kept at M50) / free / both."""
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
        a_, l_ = sc['A'], lat_of(sc, int(o.oid)); rej = postonly_at(pk, s, p, a_); vk, qk, tt, fq = fills_g(pk, s, p, a_, l_, a_ + l_ + PEND_OFFSET)
        lnd = eff_landing(g, a_, l_) + MIRROR_OFFSET if np.isfinite(g) else np.nan
        C.require(abs(fq.sum() - qk) <= 1e-9 and (fq >= -1e-12).all(), 'per-trade fills do not sum to the keep branch')
        C.require(not (fq[tt < a_] > 1e-12).any() and not (np.isfinite(lnd) and (fq[tt >= lnd] > 1e-12).any()), 'a fill before activation or at / after the structural landing')
        C.require(not rej or (qk == 0 and vk == 0), 'post-only: a refused order filled')
        full = np.flatnonzero(np.cumsum(fq) >= KP.SIZE - 1e-9); tf = float(tt[full[0]]) if len(full) else np.nan
        rows.append((int(o.oid), name(sc), rej, vk, qk, tf, lnd))
        mine = fq > 1e-12; hc = held_code(tt[mine] + t, hm[sc['tl']][s])
        trs, ix_ = KP.trade_index(Wk, t, t + KP.H, 0.); C.require(np.array_equal(trs, pk['trades']), 'the trade index differs from the scenario path')
        thru = trs[:, 2] < p if s == 0 else trs[:, 2] > p; can = np.flatnonzero((trs[:, 1] == s) & ((trs[:, 2] == p) | thru))
        sx = []                                                                # C12: each fill's trade-anchor source, as KP.window assigns it
        for x in tt[mine]:
            cs = {int(codes[s][0][ix_[r_][1]]) for r_ in can if trs[r_, 0] == x}; sx.append(cs.pop() if len(cs) == 1 else 3)
        frows.extend((int(o.oid), name(sc), float(x), float(q_), int(c_), int(c2)) for x, q_, c_, c2 in zip(tt[mine], fq[mine], hc, sx))
        if sc == B0:                                                           # control (a); v6 (j): the fills' sources equal the committed ones
            ct, cq, cs_ = fc; mine = fq > 1e-12
            same = (vk == o.v and qk == o.q and (tf == o.t_full or (np.isnan(tf) and np.isnan(o.t_full))) and rej == (o.status == 'refused_post_only')
                    and np.array_equal(tt[mine], ct) and np.array_equal(fq[mine], cq) and np.array_equal(np.asarray(sx, np.int64), cs_))
            C.require(same, f'the baseline scenario differs from the committed replay (order {int(o.oid)})'); st['base_matched'] += 1
    st['orders'] += 1


def _day_sem(day):
    """The day's S3 leg windows by legs_timed2 row (= the loader's row_id) and the input hashes."""
    rows = X.day_rows(day); sem, h = X.day_sem(day, rows); del rows; return sem, h


def day_run(day, src=None, timing=None, kp=None, out=None, env_of=None, sem_of=None, set_of=None, st_lag=None):
    """One day: every committed state with a path, in every kernel scenario -> orders.parquet (order x scenario) + fills.parquet."""
    start = time.time(); C.require(day in KP.DAYS, 'undeclared day'); src = src or KP.Tape(); timing = timing or loader_timing
    kp = Path(kp or KP.OUT); out = Path(out or OUT); d = out / day; C.require(not d.exists(), 'day output exists')
    r = json.loads((kp / day / 'receipt.json').read_text()); C.require(all(C.sha(kp / day / x) == h for x, h in r['outputs'].items()), 'keep-pull outputs changed since their receipt')
    O = pd.read_parquet(kp / day / 'orders.parquet'); Fl = pd.read_parquet(kp / day / 'fills.parquet'); F50 = Fl[Fl.clock == SHIFT]
    fc = {int(k): (g.t.values.astype(float), g.q.values.astype(float), g.src.values.astype(np.int64)) for k, g in F50.groupby('oid')}
    none = (np.zeros(0), np.zeros(0), np.zeros(0, np.int64))
    env = (env_of or X.day_env)(day); markets = src.markets(day); offset = src.offset(day); sem, sem_in = (sem_of or _day_sem)(day)
    tset, ts_in = (set_of or X.set_of)(day); lag_in = {}
    if st_lag is None:
        st_lag, lag_in = X.known_lag()
    Om = O[O.status.isin(('eligible', 'refused_post_only'))]; by_m = {int(m): g for m, g in Om.groupby('m')}
    rows, frows = [], []; st = dict(windows=0, orders=0, base_matched=0, kernel_checked=0, events=0, **{f'held_events_{k}': 0 for k in HOLD_FAMILIES})
    for begin in range(0, len(markets), 24):
        data = src.load(day, markets[begin:begin + 24], offset)
        for W in src.pairs(data):
            m = int(W[0]['m']); st['windows'] += 1
            if m not in by_m:
                continue
            tau, held, nev, hm = timelines(timing(data, m, W), W, env, day, sem, tset, st_lag); st['events'] += nev
            for k in HOLD_FAMILIES:
                st[f'held_events_{k}'] += held[k]
            cd = src.sources(data, m)                                         # the committed run's per-trade source codes, the tape's order
            for o in by_m[m].itertuples():
                scen_order(W, o, tau, fc.get(int(o.oid), none), rows, frows, st, hm, cd)
        del data; C.guard('keep-pull scenarios day')
        print(json.dumps(dict(stage='scenarios', day=day, markets=min(begin + 24, len(markets)), orders=st['orders'], seconds=round(time.time() - start), rss=round(I.K.rss_mib()))), flush=True)
    C.require(st['orders'] == len(Om) == st['base_matched'], f"re-simulated {st['orders']} / baseline-matched {st['base_matched']} of {len(Om)} committed states with a path")
    C.require(st['kernel_checked'] > 0 or not len(Om), 'no kernel clock check ran')
    S = pd.DataFrame(rows, columns=['oid', 'scen', 'refused', 'v', 'q', 't_full', 'land']); SF = pd.DataFrame(frows, columns=['oid', 'scen', 't', 'q', 'held', 'src'])
    d.mkdir(parents=True); S.to_parquet(d / 'orders.parquet', index=False); SF.to_parquet(d / 'fills.parquet', index=False)
    C.dump(d / 'receipt.json', dict(asof=C.stamp(), day=day, stat=st, scenarios=[name(sc) for sc in KSCEN], families={name(sc): family(sc) for sc in SCEN},
                                    timeline_rules=TIMELINE_RULES, env=env, s3_inputs=sem_in, settlement_inputs=dict(ts_in, **lag_in), st_lag_ms=st_lag, settlement_model=X.SETTLEMENT_MODEL, st_lag_role='UNIDENTIFIED' if st_lag is None else 'EXPLICIT_SCENARIO_ASSUMPTION',
                                    keep_pull_receipt=C.sha(kp / day / 'receipt.json'),
                                    keep_pull_outputs=r['outputs'], provenance=KP.provenance(), source_sha256=C.sha(Path(__file__).resolve()),
                                    dependencies={p_.name: C.sha(p_) for p_ in (Path(X.__file__), Path(QP.__file__), Path(FK.__file__), HERE / 'pf_flow_control_kernel.cpp')},
                                    outputs={x: C.sha(d / x) for x in ('orders.parquet', 'fills.parquet')}, seconds=time.time() - start))
    print(json.dumps(dict(day=day, stat=st, seconds=round(time.time() - start))), flush=True)
    return d


def outcome(O, Os, Fs, sc, policy=None):
    """One scenario's values for one score policy (default the declared A+0; v6.1: any of KP.POLICIES, for pf_keep_pull_variants.py). O: the
    committed orders (states, fair inputs, side, price, mark, structural decision); Os: per order of O the scenario's refused / v / q / t_full /
    land; Fs: its fills (day, oid, t, q)."""
    pol = KP.DECLARED if policy is None else policy; C.require(pol in KP.POLICIES, f'undeclared score policy {pol}')
    Os = Os.rename(columns={c: c + '_s' for c in ('refused', 'v', 'q', 't_full', 'land')})
    M = O.merge(Os, on=['day', 'oid'], how='inner', validate='one_to_one'); E = M[~M.refused_s.astype(bool)].reset_index(drop=True)
    cell = E.groupby(['day', 'm', 'side', 'phase'], sort=True).ngroup().values
    T = KP.score_steps(E[[f'f{j:02d}' for j in range(KP.NSTEP + 1)]].values, E.side.values, E.p.values)[pol]
    endj = endj_n(E.t_full_s.values, E.g_struct.values, sc['N']); act, CW = V1.null_weights(T, cell, endj)
    ix = pd.Series(np.arange(len(E)), index=pd.MultiIndex.from_arrays([E.day.values, E.oid.values]))
    ri = ix.reindex(pd.MultiIndex.from_arrays([Fs.day.values, Fs.oid.values])).values; C.require(not np.isnan(ri).any(), 'a fill of an order outside the eligible set')
    ri = ri.astype(int); sg = np.where(E.side.values == 0, 1., -1.); val = Fs.q.values * sg[ri] * (E.mark.values[ri] - E.p.values[ri]); t = Fs.t.values.astype(float)
    lend = np.where(np.isnan(E.land_s.values[ri]), np.inf, E.land_s.values[ri])
    F = dict(val=val, pulled=act[ri], T=T[ri], s=cell[ri], endj=endj[ri], lend=lend)
    lf = np.where(slow_order(E.oid.values[ri], sc['X'][0]), sc['X'][1], sc['L']) if sc.get('X') else np.full(len(ri), float(sc['L']))   # per-fill latency
    groups = [(lf == x, float(x)) for x in np.unique(lf)]
    con = lambda m: sum(float(X.contrib(t[g & m], {k: v[g & m] for k, v in F.items()}, CW, x).sum()) for g, x in groups)
    e = con(np.ones(len(ri), bool)); r = KP.landing(np.clip(T[ri], 1, KP.NSTEP), lf); score = float(-(val * (act[ri] & (t >= r))).sum())
    j0 = KP.landing(np.full(len(ri), KP.J0), lf); pa = float(-(val * ((KP.J0 < endj[ri]) & (t >= j0))).sum())
    filled = np.bincount(ri, minlength=len(E)) > 0; hv = np.isin(Fs.held.values, (1, 3)) if 'held' in Fs else np.zeros(len(Fs), bool)
    # v5 (FINDINGS C1 / C3): every value split by the order's age at the fill, and the fills that land while the score's pull is in flight
    yb = np.digitize(t - float(sc['A']), AGE_EDGES); rm = act[ri] & (t >= r); pend = act[ri] & (t >= KP.STEP * T[ri]) & (t < r)
    by_age = {lab: dict(E=con(yb == b), score=float(-(val * (rm & (yb == b))).sum()), fills_value=float(val[yb == b].sum()), fills=int((yb == b).sum()),
                        shares=float(Fs.q.values[yb == b].sum())) for b, lab in enumerate(AGE_LABELS)}
    gross = max(float(np.abs(val).sum()), 1e-12)
    sc_ = Fs.src.values.astype(np.int64) if 'src' in Fs else np.full(len(Fs), -1)                       # C12 (v6): the age table by trade-anchor source
    by_age_src = {lab: {sl: dict(E=con((yb == b) & (sc_ == c)), fills=int(((yb == b) & (sc_ == c)).sum())) for c, sl in enumerate(SRC_LABELS)}
                  for b, lab in enumerate(AGE_LABELS)}
    return dict(KEEP=float(E.v_s.sum()), score=score, null=score - e, E=e, PULLALL=pa, fills=int(len(Fs)), shares=float(Fs.q.sum()), eligible=int(len(E)),
                fills_value=float(val.sum()), by_age=by_age, by_age_src=by_age_src,
                b1a_E=b1a_of(by_age),
                no_young=dict(E=e - by_age['lt100']['E'], score=score - by_age['lt100']['score'], fills_value=float(val.sum()) - by_age['lt100']['fills_value']),
                cancel_pending=dict(fills=int(pend.sum()), shares=float(Fs.q.values[pend].sum()), value=float(val[pend].sum()),
                                    share_of_fills=float(pend.mean()) if len(pend) else 0., share_of_gross=float(np.abs(val[pend]).sum()) / gross),
                held=dict(fills=int(hv.sum()), shares=float(Fs.q.values[hv].sum()), value=float(val[hv].sum()), gross=float(np.abs(val[hv]).sum()),
                          share_of_gross=float(np.abs(val[hv]).sum() / max(float(np.abs(val).sum()), 1e-12))),
                orders_filled=int(filled.sum()), pulls=int(act.sum()), by_day={str(d_): con(E.day.values[ri] == d_) for d_ in sorted(set(E.day))},
                _filled=set(zip(E.day.values[filled], E.oid.values[filled])))


def keep_not_fills(res):
    """Control (i) of v5: the scenarios whose KEEP is not the sum of their fills' values (the C1 split would then be incomplete)."""
    return [nm for nm, v in res.items() if abs(v['KEEP'] - v['fills_value']) > 1e-6 * max(1., abs(v['KEEP']))]


def b1a_of(by_age):
    """FINDINGS C1: E weighted by age bin with FV3's B1a precision weights, per weight set (M 0 / M 50)."""
    return {k: float(sum(by_age[lab]['E'] * w for lab, w in zip(AGE_LABELS, wt))) for k, wt in FV.B1A.items()}


def b1a_sign(e, w):
    """FINDINGS C1: does B's E keep its sign under every B1a weight set?"""
    return 'HOLDS-SIGN' if e != 0 and all(np.sign(x) == np.sign(e) for x in w.values()) and len(w) > 0 else 'SIGN-DIFFERS'


def at_m250(v50, v250):
    """FINDINGS C2: does a value's sign at the M50 baseline hold at the M250 trade clock?"""
    return 'HOLDS-AT-M250' if np.sign(v50) == np.sign(v250) and v50 != 0 else 'SIGN-DIFFERS-AT-M250'


def stable(lo, hi):
    return 'STABLE-POSITIVE-IN-TESTED-SCENARIOS' if lo > 0 else 'STABLE-NEGATIVE-IN-TESTED-SCENARIOS' if hi < 0 else 'NOT-STABLE-IN-TESTED-SCENARIOS'


def report(out=None, kp=None, days=KP.DAYS, expect='committed', tol=1.):
    """Every scenario's values pooled over the days -> scenarios.txt / scenarios.json; controls (g, h); readings Z1-Z3."""
    start = time.time(); out = Path(out or OUT); kp = Path(kp or KP.OUT); O, Fl, S, SF, st = [], [], [], [], {}; days = list(days)
    expect = X.collapsed_reference(days) if expect == 'committed' else expect; stem = 'scenarios' if days == list(KP.DAYS) else 'scenarios_' + '_'.join(days)
    for d in days:
        r = json.loads((out / d / 'receipt.json').read_text()); C.require(r.get('settlement_model') == X.SETTLEMENT_MODEL, 'legacy settlement timing receipt: rebuild scenarios before reporting')
        C.require(all(C.sha(out / d / x) == h for x, h in r['outputs'].items()), f'{d}: scenario outputs changed since their receipt')
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
            Fs = Fl[Fl.clock == float(sc['tl'][1:])][['day', 'oid', 't', 'q', 'src']]
        else:
            key = name(dict(sc, N=KP.NOTICE)); Os = S[S.scen == key][['day', 'oid', 'refused', 'v', 'q', 't_full', 'land']]
            Fs = SF[SF.scen == key][['day', 'oid', 't', 'q', 'held', 'src']]
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
    bad_k = keep_not_fills(res)                                                           # control (i): KEEP is the sum of its fills
    if bad_k:
        sys.exit(f'KEEP-PULL SCENARIOS FAILED -> no output: KEEP is not the sum of its fills\' values in {bad_k} (the C1 split would be incomplete)')
    Ey = [v['no_young']['E'] for v in res.values()]; Sy = [v['no_young']['fills_value'] + v['no_young']['score'] for v in res.values()]
    timing_identified = all(json.loads((out / d / 'receipt.json').read_text()).get('st_lag_ms') is not None for d in days)
    read = {'TIMING EVIDENCE': 'EXPLICIT-LAG-SCENARIO-ONLY' if timing_identified else 'LOWER-MATCH-BOUND-UNIDENTIFIED', 'Z0 CONTROLS': 'YES', 'Z1 E ACROSS TESTED SCENARIOS': stable(min(Es), max(Es)), 'Z2 KEEP VALUE ACROSS TESTED SCENARIOS': stable(min(Ks), max(Ks)),
            'Z3 THE SCORE\'S VALUE ACROSS TESTED SCENARIOS': stable(min(Ss), max(Ss)),
            'Z1-NY E WITHOUT FILLS ON ORDERS < 100 MS OLD ACROSS TESTED SCENARIOS': stable(min(Ey), max(Ey)),
            'C2 E AT M250 vs M50': at_m250(b['E'], res[name(dict(B0, tl='M250'))]['E']),
            'C1 B1a-WEIGHTED E AT B': b1a_sign(b['E'], b['b1a_E']),
            'C2 KEEP + SCORE AT M250 vs M50': at_m250(b['KEEP'] + b['score'], res[name(dict(B0, tl='M250'))]['KEEP'] + res[name(dict(B0, tl='M250'))]['score']),
            'Z3-NY THE SCORE\'S VALUE WITHOUT FILLS ON ORDERS < 100 MS OLD ACROSS TESTED SCENARIOS': stable(min(Sy), max(Sy))}
    L = ['QUEUE-AWARE SCENARIO REPLAY of the keep / pull selection, declared score A+0 - the fair-edge rule, not a learned model (pf_keep_pull_scenarios.py; every '
         'committed state with a path re-simulated '
         f'through the branch kernel; days {", ".join(days)}; c summed)',
         f"  {len(Op):,} states re-simulated ({int((Op.status == 'eligible').sum()):,} eligible at B); events (taker orders) by day "
         + ', '.join(f"{d} {st[d]['events']:,}" for d in days) + '; held contradictory events (kept at M50) of them: '
         + '; '.join(f"{k} " + ', '.join(f"{d} {st[d][f'held_events_{k}']:,} ({100 * st[d][f'held_events_{k}'] / max(st[d]['events'], 1):.1f}%)" for d in days)
                     for k in HOLD_FAMILIES),
         f"  Z0 (h): B's E {b['E']:+,.0f} and score improvement {b['score']:,.0f} = the clock audit's per-fill values on the committed tables", '',
         f"  {'scenario':<24} {'E':>11} {'score':>11} {'null':>11} {'KEEP':>12} {'KEEP+score':>12} {'PULL-ALL':>11} {'eligible':>9} {'fills':>8} {'filled':>7} {'discov.':>7} {'lost':>6}"]
    for nm, v in res.items():
        L.append(f"  {nm:<24} {v['E']:>+11,.0f} {v['score']:>+11,.0f} {v['null']:>+11,.0f} {v['KEEP']:>+12,.0f} {v['KEEP'] + v['score']:>+12,.0f} {v['PULLALL']:>+11,.0f} "
                 f"{v['eligible']:>9,} {v['fills']:>8,} {v['orders_filled']:>7,} {v['discovered']:>7,} {v['lost']:>6,}")
    L += ['', '  C1 / C3 (FINDINGS checklist): E by the order\'s age at the fill (< 100 / 100-500 / >= 500 ms after activation), the headline without fills on '
          'orders < 100 ms old (live\'s were 8-12% real, B1a), and the fills that land while the score\'s pull is in flight (cancel-pending):',
          f"  {'scenario':<24} {'E <100':>9} {'E 100-500':>10} {'E >=500':>9} {'E w/o <100':>11} {'E B1a M0/M50':>17} {'K+s w/o <100':>13} {'fills <100':>10} {'pending fills / c / gross':>28}"]
    for nm, v in res.items():
        a_, ny, cp = v['by_age'], v['no_young'], v['cancel_pending']
        L.append(f"  {nm:<24} {a_['lt100']['E']:>+9,.0f} {a_['100to500']['E']:>+10,.0f} {a_['ge500']['E']:>+9,.0f} {ny['E']:>+11,.0f} "
                 f"{v['b1a_E']['M0']:>+8,.0f}/{v['b1a_E']['M50']:<+8,.0f} {ny['fills_value'] + ny['score']:>+13,.0f} "
                 f"{100 * a_['lt100']['fills'] / max(v['fills'], 1):>9.1f}% {cp['fills']:>9,} / {cp['value']:>+9,.0f} / {100 * cp['share_of_gross']:>4.1f}%")
    ba = b['by_age_src']
    L += ['', f"  C12 (FINDINGS checklist): B's E by age (rows) and trade-anchor source (columns: {' / '.join(SRC_LABELS)}; E2 anchors are uncertain by ~+-250 ms), "
          'c (fills):'] + [f"  {lab:<10} " + ' | '.join(f"{sl} {ba[lab][sl]['E']:+,.0f} ({ba[lab][sl]['fills']:,})" for sl in SRC_LABELS) for lab in AGE_LABELS]
    L += ['', '  scenario families (one factor moved from B; none of them is an order latency unless named so) and the fills at held instants (trades of '
          'contradictory events kept at M50: count / shares / value c / share of the scenario\'s fill gross):']
    for sc in SCEN:
        nm = name(sc); f_, dsc = family(sc); h = res[nm]['held']
        L.append(f"  {nm:<24} {f_:<18} {dsc}; held {h['fills']:,} / {h['shares']:,.0f} sh / {h['value']:+,.0f} c / {100 * h['share_of_gross']:.1f}%")
    L += ['', '  E by day: ' + ' | '.join(f"{nm} " + ', '.join(f'{d} {x:+,.0f}' for d, x in v['by_day'].items()) for nm, v in res.items()), '',
          'READINGS (declared before any data was read)'] + [f'  {k}: {v}' for k, v in read.items()] + [
          f"  E ranges over [{min(Es):+,.0f}, {max(Es):+,.0f}], KEEP over [{min(Ks):+,.0f}, {max(Ks):+,.0f}], KEEP + score over [{min(Ss):+,.0f}, {max(Ss):+,.0f}] in the "
          f'{len(res)} tested scenarios; without fills on orders < 100 ms old E over [{min(Ey):+,.0f}, {max(Ey):+,.0f}], KEEP + score over [{min(Sy):+,.0f}, {max(Sy):+,.0f}]', '',
          'LIMITS: consumed days; one declared rule; one-factor-at-a-time scenarios and timelines (no joint extreme search); assumption ranges, not measured '
          'distributions; queue rule M unvalidated on live orders; STATES is local selection; one order at a time, no re-post after a pull.']
    R = {k: {x: y for x, y in v.items() if not x.startswith('_')} for k, v in res.items()}
    (out / f'{stem}.txt').write_text('\n'.join(L) + '\n')
    C.dump(out / f'{stem}.json', dict(asof=C.stamp(), readings=read, scenarios=R, families={name(sc): family(sc) for sc in SCEN}, stat=st, days=list(days), settlement_model=X.SETTLEMENT_MODEL, omitted_scenarios={'STlo': 'no finite lower bound from UUID creation'}, provenance=KP.provenance(),
                                        source_sha256=C.sha(Path(__file__).resolve()), seconds=time.time() - start))
    print('\n'.join(L), flush=True)
    return R, read


class SynTiming:
    """The selftest's timing source on the keep / pull synthetic tape: one event per (market, side, anchor); WS legs (source 0) observed at
    their anchor, E2 legs (1) estimated at it, imputed legs (2) only by their second; executedAt = anchor / 1000 s (the REST cap 1 s later)
    except the planted contradictory events, whose cap precedes their windows; S3: a WS leg [anchor - 250, anchor] (its own increment), every
    other leg its REST range (pf_keep_pull_exec_windows.rest_range)."""
    def __init__(self, src, contra=()):
        self.src, self.contra, self.ids = src, set(contra), {}

    def __call__(self, data, m, W):
        d = self.src.day(data['day']); out = []
        for s in (0, 1):
            c, anc = d['codes'][m][s]; anc = np.asarray(anc, float); g = np.array([self.ids.setdefault((m, s, a), len(self.ids)) for a in anc], np.int64)
            ex = anc / 1000. - np.array([2. if (m, s, a - d['ws']) in self.contra else 0. for a in anc])
            r0, r1 = X.rest_range(ex)
            out.append(pd.DataFrame(dict(side=s, mid=m, group=g, how=(c == 0).astype(int), t_ms=np.where(c == 0, anc, np.nan), t_e2=np.where(c == 1, anc, np.nan), exec_s=ex, anc=anc,
                                         S3_llo=np.where(c == 0, anc - 250., r0), S3_lhi=np.where(c == 0, anc, r1), ST_llo=anc - 170., ST_lhi=anc - 20.)))
        return out


def selftest():
    """Synthetic tape only: the keep / pull run on it is the committed baseline; planted scenario fills and landings; the report; refusals."""
    global PEND_OFFSET, MIRROR_OFFSET
    bad = []; src = KP.Syn(); days = KP.DAYS[:2]; env = lambda d_: dict(h=600., q01=-900., q99=1050., inputs={})
    syn_set = lambda d_: (pd.DataFrame(dict(t_set_ms=np.zeros(0), set_status=np.zeros(0, int))), {})   # SynTiming carries its ST windows
    if not (np.array_equal(endj_n(np.array([1000., np.nan, 1000.]), np.array([np.nan, np.nan, 600.]), KP.NOTICE), A.endj_of(np.array([1000., np.nan, 1000.]), np.array([np.nan, np.nan, 600.])))
            and endj_n(np.full(3, 1000.), np.full(3, np.nan), np.array([0., 100., 250.])).tolist() == [20, 22, 25]):
        bad.append('the notice rule (endj) at 0 / 100 / 250 ms')
    if not (np.array_equal(landing_s(np.arange(1, 99) * 50., KP.PLACE, KP.CANCEL), KP.landing(np.arange(1, 99))) and landing_s(np.array([50., 1050.]), 100., 0.).tolist() == [100., 1050.]):
        bad.append('the scenario landing rule')
    fr = np.array([[0., 0., 50, 49, 48, 47, 46] + [100.] * 5 + [52, 53, 54, 55, 56] + [100.] * 5,
                   [1000., 1000., 47, 46, 45, 44, 43] + [100.] * 5 + [49, 50, 51, 52, 53] + [100.] * 5], float)       # a cross for the 50 bid known at 1 000 ms
    p0 = dict(frames=fr, trades=np.array([[1000., 0., 48., 20.], [1020., 0., 48., 20.]]), mark=49., mid=51.)
    g0 = KP.struct_cancel(p0, 0, 50.)[0]; f0, f37 = fills_g(p0, 0, 50., 58., 0., 58.), fills_g(p0, 0, 50., 58., 37., 95.)
    pick = lambda f: [(float(a), float(b)) for a, b in zip(f[2], f[3]) if b > 1e-12]
    if not (g0 == 1000. and pick(f0) == [(1000., 20.)] and pick(f37) == [(1000., 20.), (1020., 10.)] and eff_landing(g0, 58., 0.) == float(np.nextafter(1000., np.inf))
            and eff_landing(g0, 58., 37.) == 1037.):
        bad.append(f'the kernel tie order at zero latency: g {g0}, fills {pick(f0)} / {pick(f37)}, landings {eff_landing(g0, 58., 0.)} / {eff_landing(g0, 58., 37.)}')
    if not all(eff_landing(g_, KP.PLACE, l_) == float(X.struct_landing(np.array([g_]), l_)[0]) for g_ in (50., 100., 1050.) for l_ in (0., KP.CANCEL, 100.)):
        bad.append('the scenario landing rule differs from the ledger\'s struct_landing')
    lg = [pd.DataFrame(dict(side=0, mid=7, group=[1, 1, 2], how=[1, 1, 0], t_ms=[1000., 1100., np.nan], t_e2=np.nan, exec_s=[1., 1., 3.], anc=[1000., 1000., 3500.], row_id=[5, 3, 9])),
          pd.DataFrame(dict(side=1, mid=7, group=[3], how=[1], t_ms=[2000.], t_e2=np.nan, exec_s=[2.], anc=[2000.], row_id=[0]))]
    sem_ = pd.DataFrame(dict(S3_llo=np.zeros(10), S3_lhi=np.full(10, 1e9))); sem_.loc[[5, 3, 9, 0], ['S3_llo', 'S3_lhi']] = [[800., 1000.], [900., 1100.], [-7000., 4000.], [1750., 2000.]]
    Wt = [dict(anc=lg[0].anc.values), dict(anc=lg[1].anc.values)]; env1 = dict(h=600., q01=-900., q99=1050.)
    tset_ = pd.DataFrame(dict(t_set_ms=np.full(10, np.nan), set_status=1)); tset_.loc[[5, 3, 0], 't_set_ms'] = [950., 950., 1900.]; tset_.loc[[5, 3, 0], 'set_status'] = 0
    try:
        tt_, hh_, nev_, hm_ = timelines(lg, Wt, env1, 'd', sem_, tset_, 100.)
        if not (tt_['S3lo'][0].tolist() == [900., 900., -7000.] and tt_['S3hi'][0].tolist() == [1000., 1000., 4000.] and tt_['S3lo'][1].tolist() == [1750.] and nev_ == 3
                and hh_ == dict(S1=0, S2=0, S3=0, ST=0) and hm_['S3lo'][0][0].size == 0 and hm_['S3lo'][0][1].tolist() == [-7000., 900., 900.]
                and tt_['STlo'][0].tolist() == [850., 850., -7000.] and tt_['SThi'][0].tolist() == [950., 950., 4000.] and tt_['STlo'][1].tolist() == [1800.]
                and tt_['SThi'][1].tolist() == [1900.]):
            bad.append(f"timelines with S3 / ST by row id: {tt_['S3lo']} {tt_['S3hi']} {tt_['STlo']} {tt_['SThi']} {hh_} {nev_}")
    except (SystemExit, ValueError) as e:
        bad.append(f'timelines with S3 by row id refused: {e}')
    try:
        timelines(lg, Wt, env1, 'd', sem_.drop(index=9), tset_, 100.); bad.append('timelines accepted a leg without its S3 window')
    except ValueError:
        pass
    try:
        timelines(lg, Wt, env1, 'd', sem_, tset_.drop(index=9), 100.); bad.append('timelines accepted a leg without its settlement row')
    except ValueError:
        pass
    try:
        upper_only = timelines(lg, Wt, env1, 'd', sem_, tset_, None)[0]
    except (SystemExit, ValueError) as e:
        upper_only = None; bad.append(f'upper-only timelines refused: {e}')
    if upper_only is not None and ('STlo' in upper_only or 'STlo' in TIMELINES or not all(np.isfinite(x).all() for x in upper_only['SThi'])):
        bad.append('an unidentified earliest match was replayed or the finite upper endpoint was lost')
    fam = {name(sc): family(sc)[0] for sc in SCEN}; n0 = lambda **kw: name(dict(B0, **kw))
    if not (set(fam.values()) == set(FAMILIES) and fam[n0(tl='M0')] == fam[n0(tl='M250')] == 'trade-clock shift' and fam[n0(L=0.)] == fam[n0(L=100.)] == 'cancel latency'
            and fam[n0()] == 'baseline' and fam[n0(tl='S3lo')] == 'timeline' and fam[n0(A=15.)] == 'activation delay' and fam[n0(X=CANCEL_MIX)] == 'cancel latency' and fam[n0(N=160.)] == 'fill-notice delay'
            and not any('zero latency' in family(sc)[1] for sc in SCEN)):
        bad.append(f'scenario families: {fam}')
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
                    day_run(d, src=src, timing=tim, kp=tmp / 'kp', out=tmp / 'sc', env_of=env, set_of=syn_set, st_lag=150.)
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
                   'S2hi': (fl(50000., n_(tl='S2hi')), []), 'EVhi': (fl(50000., n_(tl='EVhi')), []), 'activation 15': (fl(50000., n_(A=15.)), bfill),
                   'activation 100': (fl(50000., n_(A=100.)), bfill), 'cancel 0 ms (the 1 070 fill after the landing)': (fl(50000., n_(L=0.)), [(950., 20.)]),
                   'cancel 100 ms': (fl(50000., n_(L=100.)), bfill),
                   'landings 1 087 / just after 1 050 / 1 150 / 1 087': ([ld(50000., n_()), ld(50000., n_(L=0.)), ld(50000., n_(L=100.)), ld(50000., n_(A=100.))],
                                                                         [1087., float(np.nextafter(1050., np.inf)), 1150., 1087.]),
                   '200 000 pending-cancel fill at B': (fl(200000., n_()), [(90., 30.)]), '200 000 activation 15 (landing 87)': (fl(200000., n_(A=15.)), []),
                   '200 000 activation 100 (live after the trade)': (fl(200000., n_(A=100.)), []), '200 000 cancel 0 (pending landing 58)': (fl(200000., n_(L=0.)), []),
                   '200 000 cancel 100 (pending landing 158)': (fl(200000., n_(L=100.)), [(90., 30.)]),
                   '200 000 pending landings 95 / 87 / 137 / 58 / 158': ([ld(200000., n_()), ld(200000., n_(A=15.)), ld(200000., n_(A=100.)), ld(200000., n_(L=0.)), ld(200000., n_(L=100.))],
                                                                         [95., 87., 137., 58., 158.]),
                   '200 000 S1lo (before posting) / S1mid (before activation) / S1hi (after landing)': ([fl(200000., n_(tl=x)) for x in ('S1lo', 'S1mid', 'S1hi')], [[], [], []]),
                   '150 000 contradictory event held at M50': ([fl(150000., n_(tl=x)) for x in ('S1lo', 'S1hi', 'S2lo', 'S2hi', 'S3lo', 'S3hi', 'SThi')], [[(450., 30.)]] * 7),
                   'SThi (the sweep 30 ms after M50 fills; the WS trade lands after the cross cancel)': (fl(50000., n_(tl='SThi')), [(980., 20.)]),
                   '200 000 SThi (after the pending landing)': (fl(200000., n_(tl='SThi')), []),
                   'S3lo (the E2 sweep moves before posting - its REST range starts 10 s early - so the 50 sh at 50 at 870 ms meet the queue ahead: no fill)':
                       (fl(50000., n_(tl='S3lo')), []),
                   'S3hi (both after the cross cancel)': (fl(50000., n_(tl='S3hi')), []),
                   '200 000 S3lo (before posting) / S3hi (after the pending landing)': ([fl(200000., n_(tl=x)) for x in ('S3lo', 'S3hi')], [[], []])}
            for k, (got, want) in exp.items():
                if got != want:
                    bad.append(f'{k}: {got} != {want}')
            o150 = int(O[(O.m == 108) & (O.side == 0) & (O.t == ws + 150000.)].oid.iloc[0]); hcode = lambda sc: SF[(SF.oid == o150) & (SF.scen == sc)].held.tolist()
            K50 = pd.read_parquet(tmp / 'kp' / days[0] / 'fills.parquet'); K50 = K50[K50.clock == SHIFT][['oid', 't', 'src']]   # C12: B's sources = the committed ones
            SB = SF[SF.scen == n_()][['oid', 't', 'src']].merge(K50, on=['oid', 't'], how='outer', suffixes=('', '_c'), indicator=True)
            if not (len(SB) > 0 and (SB._merge == 'both').all() and np.array_equal(SB.src.values.astype(int), SB.src_c.values.astype(int)) and SB.src.nunique() >= 2):
                bad.append(f"C12: B's fill sources differ from the committed fills' ({int((SB._merge != 'both').sum())} unmatched, {SB.src.value_counts().to_dict()})")
            st0 = json.loads((tmp / 'sc' / days[0] / 'receipt.json').read_text())['stat']
            if not (hcode(n_()) == [2] and hcode(n_(tl='S1lo')) == [1] and hcode(n_(tl='S3hi')) == [1] and st0['events'] > 0 and st0['held_events_S3'] >= 1
                    and st0['held_events_S1'] >= 1):
                bad.append(f"held fills / denominators: B {hcode(n_())}, S1lo {hcode(n_(tl='S1lo'))}, S3hi {hcode(n_(tl='S3hi'))}, stat {st0}")
            R = None
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    R, read = report(out=tmp / 'sc', kp=tmp / 'kp', days=days, expect=None)
                n250 = sum(int((pd.read_parquet(tmp / 'kp' / d / 'fills.parquet').clock == 250.).sum()) for d in days)
                R1_, _ = report(out=tmp / 'sc', kp=tmp / 'kp', days=days[:1], expect=None)                 # one day first: its own report files
                if not ((tmp / 'sc' / f'scenarios_{days[0]}.json').exists() and R1_[n_()]['eligible'] < R[n_()]['eligible']):
                    bad.append('the one-day report')
                pl = b1a_of({'lt100': dict(E=1.), '100to500': dict(E=10.), 'ge500': dict(E=100.)})
                if not (all(abs(pl[k] - (w[0] + 10. * w[1] + 100. * w[2])) <= 1e-9 for k, w in FV.B1A.items()) and set(pl) == {'M0', 'M50'}
                        and all(R[nm]['b1a_E'] == b1a_of(R[nm]['by_age']) for nm in R)
                        and (b1a_sign(5., dict(M0=1., M50=2.)), b1a_sign(5., dict(M0=1., M50=-2.)), b1a_sign(0., dict(M0=0., M50=0.))) == ('HOLDS-SIGN', 'SIGN-DIFFERS', 'SIGN-DIFFERS')
                        and read.get('C1 B1a-WEIGHTED E AT B') in ('HOLDS-SIGN', 'SIGN-DIFFERS') and FV.B1A['M50'][0] < .2):
                    bad.append(f"C1 B1a-weighted: {({nm: R[nm]['b1a_E'] for nm in list(R)[:2]})} {read.get('C1 B1a-WEIGHTED E AT B')}")
                if not (all(sum(x['fills'] for x in R[nm]['by_age_src'][lab].values()) == R[nm]['by_age'][lab]['fills'] for nm in R for lab in AGE_LABELS)
                        and read.get('C2 E AT M250 vs M50') in ('HOLDS-AT-M250', 'SIGN-DIFFERS-AT-M250')
                        and (at_m250(5., 3.), at_m250(5., -1.), at_m250(0., 0.)) == ('HOLDS-AT-M250', 'SIGN-DIFFERS-AT-M250', 'SIGN-DIFFERS-AT-M250')):
                    bad.append(f"C12 / C2: the age-by-source table does not partition the fills, or the M250 reading: {read.get('C2 E AT M250 vs M50')}")
                if not (read['Z0 CONTROLS'] == 'YES' and len(R) == len(SCEN) and all(k in read for k in ('Z1 E ACROSS TESTED SCENARIOS', 'Z2 KEEP VALUE ACROSS TESTED SCENARIOS'))
                        and (tmp / 'sc' / f"scenarios_{'_'.join(days)}.json").exists() and R[n_()]['eligible'] > 0 and R[n_(tl='M250')]['fills'] == n250 > 0
                        and R[n_(N=660.)]['fills'] == R[n_()]['fills'] and R[n_(N=660.)]['null'] != R[n_()]['null']
                        and R[n_()]['held']['fills'] == 0 and R[n_(tl='S1lo')]['held']['fills'] >= 1 and R[n_(tl='S1lo')]['held']['gross'] > 0):
                    bad.append(f'report: {read}')
            except (SystemExit, ValueError) as e:
                bad.append(f'synthetic report refused: {e}')
            try:                                                               # control (h) must refuse a baseline that is not the committed one
                if R is None:
                    raise ValueError('no synthetic report to perturb (refused above)')
                with contextlib.redirect_stdout(io.StringIO()):
                    report(out=tmp / 'sc', kp=tmp / 'kp', days=days, expect=(R[n_()]['E'] + 1., R[n_()]['score']), tol=1e-6)
                bad.append('report accepted a baseline E that differs from the expectation')
            except (SystemExit, ValueError, TypeError) as e:
                if 'differ from the clock audit' not in str(e):
                    bad.append(f'the wrong expectation refused by another control: {e}')
            O1 = pd.DataFrame(dict(day='d', oid=[0], m=[1], side=[0], phase=[1], p=[50.], mark=[45.], g_struct=[np.nan], **{f'f{j:02d}': [51. if j < 5 else 49.] for j in range(KP.NSTEP + 1)}))
            Os1 = pd.DataFrame(dict(day='d', oid=[0], refused=[False], v=[-150.], q=[30.], t_full=[300.], land=[np.nan])); Fs1 = pd.DataFrame(dict(day='d', oid=[0], t=[300.], q=[30.]))
            o37, o100 = outcome(O1, Os1, Fs1, B0), outcome(O1, Os1, Fs1, dict(B0, L=100.))
            omx0 = outcome(O1, Os1, Fs1, dict(B0, X=CANCEL_MIX)); omx1 = outcome(O1.assign(oid=[1]), Os1.assign(oid=[1]), Fs1.assign(oid=[1]), dict(B0, X=CANCEL_MIX))
            shr = float(slow_order(np.arange(20000), CANCEL_MIX[0]).mean())
            if not (bool(slow_order(0, CANCEL_MIX[0])) and not bool(slow_order(1, CANCEL_MIX[0])) and omx0['score'] == 0. and omx1['score'] == 150.
                    and abs(shr - CANCEL_MIX[0]) < .01 and lat_of(dict(B0, X=CANCEL_MIX), 0) == CANCEL_MIX[1] and lat_of(dict(B0, X=CANCEL_MIX), 1) == KP.CANCEL):
                bad.append(f"cancel-latency mixture: score slow {omx0['score']} / fast {omx1['score']}, share {shr}")
            nmx = name(dict(B0, X=CANCEL_MIX)); Sm = S[S.scen == nmx].merge(O[['oid', 'g_struct']], on='oid'); gm_ = np.isfinite(Sm.g_struct.values)
            want_l = np.array([eff_landing(g_, KP.PLACE, lat_of(dict(B0, X=CANCEL_MIX), o_)) for g_, o_ in zip(Sm.g_struct.values[gm_], Sm.oid.values[gm_])])
            if not (gm_.sum() > 0 and np.allclose(Sm.land.values[gm_], want_l) and bool(slow_order(Sm.oid.values[gm_], CANCEL_MIX[0]).any())):
                bad.append(f'the mixture\'s per-order structural landings: {int(gm_.sum())} orders, slow among them {int(slow_order(Sm.oid.values[gm_], CANCEL_MIX[0]).sum())}')
            if not (o37['score'] == 150. and o37['pulls'] == 1 and o100['score'] == 0. and o37['KEEP'] == -150.):
                bad.append(f"one-order outcome: score {o37['score']} / {o100['score']} at 37 / 100 ms (want 150 / 0)")
            oa1, oa0 = outcome(O1, Os1, Fs1, B0, policy='A-1'), outcome(O1, Os1, Fs1, B0, policy=KP.DECLARED)   # v6.1: the edge never falls below -1 c
            if not (oa1['pulls'] == 0 and oa1['score'] == 0. and oa1['E'] == 0. and oa1['KEEP'] == o37['KEEP'] and oa0['score'] == o37['score'] == 150.
                    and oa0['pulls'] == o37['pulls'] == 1):
                bad.append(f"the policy parameter: A-1 pulls {oa1['pulls']} / score {oa1['score']}, A+0 {oa0['score']}, default {o37['score']}")
            try:
                outcome(O1, Os1, Fs1, B0, policy='A+2'); bad.append('outcome accepted an undeclared score policy')
            except ValueError as e:
                if 'undeclared score policy' not in str(e):
                    bad.append(f'the undeclared policy refused by another check: {e}')
            except KeyError as e:
                bad.append(f'an undeclared policy reached the score lookup: {e}')
            ox = outcome(O1, Os1, Fs1, dict(B0, A=250.))                  # the fill at 300 ms: 242 ms old at activation 58, 50 ms old at 250
            if not (o37['by_age']['100to500']['fills'] == 1 and o37['by_age']['lt100']['fills'] == 0 and ox['by_age']['lt100']['fills'] == 1
                    and abs(sum(a['E'] for a in o37['by_age'].values()) - o37['E']) < 1e-9 and o37['no_young']['E'] == o37['E'] and ox['no_young']['fills_value'] == 0.
                    and o37['cancel_pending']['fills'] == 0 and o100['cancel_pending'] == dict(fills=1, shares=30., value=-150., share_of_fills=1., share_of_gross=1.)
                    and o37['fills_value'] == o37['KEEP'] and ox['no_young']['E'] == 0. and ox['no_young']['score'] == 0.
                    and keep_not_fills({'a': o37, 'b': dict(o37, KEEP=o37['KEEP'] + 1.)}) == ['b']):
                bad.append(f"C1 / C3 on the one-order case: {o37['by_age']} {ox['by_age']} pending {o37['cancel_pending']} / {o100['cancel_pending']}")
            # check of 9a114cc1: the PRODUCTION default - no lag passed, the corrected known receipt read, ST upper-only, no STlo - rehearsed end to end
            kk = tmp / 'k5'; kk.mkdir(); (kk / 'known.json').write_text(json.dumps(dict(R=dict(K8=dict(model=X.SETTLEMENT_MODEL, consistent=True, lag_bound_ms=None)))))
            old_k = X.OUT_K; X.OUT_K = kk
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    day_run(days[0], src=src, timing=tim, kp=tmp / 'kp', out=tmp / 'scu', env_of=env, set_of=syn_set)
                ru = json.loads((tmp / 'scu' / days[0] / 'receipt.json').read_text()); Su = pd.read_parquet(tmp / 'scu' / days[0] / 'orders.parquet')
                if not (ru['st_lag_ms'] is None and ru['st_lag_role'] == 'UNIDENTIFIED' and ru['scenarios'] == [name(sc) for sc in KSCEN]
                        and not any(x.startswith('STlo') for x in ru['scenarios']) and name(dict(B0, tl='SThi')) in set(Su.scen)
                        and str(kk / 'known.json') in ru['settlement_inputs']):
                    bad.append(f"upper-only production day: lag {ru['st_lag_ms']} role {ru['st_lag_role']} scenarios {ru['scenarios']}")
            except (SystemExit, ValueError) as e:
                bad.append(f'upper-only production day refused: {e}')
            finally:
                X.OUT_K = old_k
            # refusals: a misaligned timing source, a committed fill changed under a matching receipt, a perturbed pending clock
            def mis(data, m, W):
                x = tim(data, m, W); x[0] = x[0].assign(anc=x[0].anc + 1.) if len(x[0]) else x[0].iloc[:0]; return x
            shutil.copytree(tmp / 'kp', tmp / 'kpb'); fb = pd.read_parquet(tmp / 'kpb' / days[0] / 'fills.parquet'); j_ = int(np.flatnonzero(fb.clock.values == SHIFT)[0])
            fb.loc[fb.index[j_], 'q'] = fb.q.iloc[j_] - 1.; fb.to_parquet(tmp / 'kpb' / days[0] / 'fills.parquet', index=False)
            rb = json.loads((tmp / 'kpb' / days[0] / 'receipt.json').read_text()); rb['outputs']['fills.parquet'] = C.sha(tmp / 'kpb' / days[0] / 'fills.parquet')
            (tmp / 'kpb' / days[0] / 'receipt.json').write_text(json.dumps(rb))
            shutil.copytree(tmp / 'kp', tmp / 'kpc'); fc_ = pd.read_parquet(tmp / 'kpc' / days[0] / 'fills.parquet'); j_ = int(np.flatnonzero(fc_.clock.values == SHIFT)[0])
            fc_.loc[fc_.index[j_], 'src'] = (int(fc_.src.iloc[j_]) + 1) % 3; fc_.to_parquet(tmp / 'kpc' / days[0] / 'fills.parquet', index=False)   # C12's control (j)
            rc_ = json.loads((tmp / 'kpc' / days[0] / 'receipt.json').read_text()); rc_['outputs']['fills.parquet'] = C.sha(tmp / 'kpc' / days[0] / 'fills.parquet')
            (tmp / 'kpc' / days[0] / 'receipt.json').write_text(json.dumps(rc_))
            for nm_, kw, pend, mo, msg in (('a timing source misaligned with the tape', dict(timing=mis, kp=tmp / 'kp'), 0., 0., 'timing legs not aligned'),
                                           ('a committed fill changed under a matching receipt', dict(timing=tim, kp=tmp / 'kpb'), 0., 0., 'the baseline scenario differs'),
                                           ("a committed fill's trade source changed under a matching receipt", dict(timing=tim, kp=tmp / 'kpc'), 0., 0.,
                                            'the baseline scenario differs'),
                                           ('a pending clock 10 ms off activation + latency', dict(timing=tim, kp=tmp / 'kp'), -10., 0., 'the baseline scenario differs'),
                                           ('a structural landing mirror 10 ms early (a kernel fill after it)', dict(timing=tim, kp=tmp / 'kp'), 0., -10.,
                                            'at / after the structural landing')):
                o_ = tmp / ('z' + str(abs(hash(nm_)) % 1000)); PEND_OFFSET = pend; MIRROR_OFFSET = mo
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        day_run(days[0], src=src, out=o_, env_of=env, set_of=syn_set, st_lag=150., **kw)
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
          '37 / 100 ms; the report end to end and refused on a wrong baseline expectation; a misaligned timing source, a changed committed fill, a changed committed trade source (C12), a perturbed pending clock '
          'and an early landing mirror each refused by its own control with no output; S3 / ST timelines (and their joins by row id, a leg without its '
          'window refused); held fills marked and valued; event denominators; every scenario\'s family (no "zero latency"); the cancel-latency '
          'mixture (its slow share, per-order landings, one-order outcomes); the policy parameter (A-1 never pulls the one-order case, an undeclared policy '
          'refused)', flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif len(a) == 2 and a[0] == 'day':
        day_run(a[1])
    elif a == ['report']:
        report()
    elif len(a) == 2 and a[0] == 'report':                                  # one day first: scenarios_<day>.txt / .json
        C.require(a[1] in KP.DAYS, 'undeclared day'); report(days=[a[1]])
    else:
        sys.exit('usage: pf_keep_pull_scenarios.py selftest | day <day> | report')
