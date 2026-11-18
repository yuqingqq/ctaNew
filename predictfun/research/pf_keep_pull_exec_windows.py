"""EXECUTION-WINDOW LEDGER AND WINDOW ORDERING CLASSIFIER FOR THE KEEP / PULL SELECTION (2026-11-18, session aea5bfb9) - v2 of the
fill-versus-cancel ordering audit. USER: "New review commit there, check it, we have some issues for simulator, need to correct these issues
first". The review of d1de7504 (reviews/automatic/20261118_d1de7504/report.md: P2 the removed-value headline mixes populations; P2 the
linked timing evidence is not retained) and its user-directed addendum (execution_windows_addendum.md), whose one next priority is: "Build
and preserve an evidence-backed execution-window ledger, then validate the ordering classifier on known-order lifecycle cases ... accept
separated windows, mark overlap / unknowns explicitly, and refuse wrong identities, contradictory windows or a delayed status timestamp
substituted for a match. Include cancellation / clock uncertainty and preserve original observations." Declared before any data is read.
The v1 audit (pf_keep_pull_timing_audit.py, d1de7504) and its receipt stay unchanged as provenance; this ledger supersedes its headline.

WHAT CHANGES FROM v1 (the review / addendum item each answers)
 1 Population (P2-1, item 4): v1 split ALL fills of the score's pulled orders and called it removed value. Now a cross-tab: replay ordering
   (REMOVED = the replay's fill at or after the pull's landing, KEPT = before it) x window class; fills / net / gross c per cell; shares on
   each row's own denominator.
 2 Retention (P2-2, item 3): a ledger of every candidate leg (day + its legs_timed2 row; its taker order = pf_txmap tk with tk_status; the raw
   observations executedAt second, first receipt, WS pending source time and how, E2 estimate - never overwritten), every event, every
   fill's candidate events and every order of the neutral population with its status; input, dependency and output hashes.
 3 Shared events (main review, item 3): one taker order is one matching operation, so its legs share ONE instant: their windows INTERSECT (an
   empty intersection is CONTRADICTORY - kept, never repaired). A fill whose anchor carries several events has candidate identities
   (multi_event): classified only where every candidate agrees. The value bounds move each event's instant once for all the orders it fills
   (v1 moved each order's copy independently); multi_event fills move independently over their candidates' windows.
 4 Status timestamps (addendum): REST executedAt is the Confirmed status second (11-09 fill-clock semantics audit: equal to the Confirmed
   stamp on 99 / 99 own legs; 0 / 1 / 10 s after Matched). It bounds the match from ABOVE only (evidence_bound (S + 1) * 1000); window_from
   refuses it as a point; no public source has established match-instant semantics (MATCH_SOURCES is empty).
 5 Cancellation (item 1): the classifier takes a cancel-effect window: CP = the replay's point (decision + 37 ms), CR = [decision, decision
   + 100 ms] (assumption range). Value bounds at cancel latency 0 / 37 / 100 ms. Activation (+58 ms) and the own-fill notice (+100 ms) stay
   the replay's configured points - recorded as assumed, not varied (the queue-aware scenario replay that re-simulates fills varies them).
 6 Ties (addendum): separated except at a touching endpoint = TIE; no venue tie-order rule is established and the replay's at-or-after
   convention is a modelling choice, not evidence.
 7 Known lifecycles (addendum): mode `known` checks the classifier and the windows on live's own 99 confirmed maker legs (K1-K4).

WINDOWS (venue clock; the replay axis is venue time; the client's feed delay is not modelled - an assumption)
 per leg            EV (evidence)                      S1 (the program's declared ranges)    S2 (measured envelopes, per day)
 WS (how > 0)       (-inf, (S + 1) * 1000]             [t_ms - 250, t_ms]                    [t_ms - h, t_ms + h]
 E2 (t_e2 finite)   (-inf, (S + 1) * 1000]             [t_e2 - 250, t_e2 + 250]              [t_e2 - h, t_e2 + h]
 imputed (S only)   (-inf, (S + 1) * 1000]             [1000 S - 500, 1000 S + 500] *        [1000 S + q01, 1000 S + q99] *
 S = executedAt second; t_ms = settlementsPending source time, linked to the leg by value (no signed identity); h = the day's p90 |E2 - WS|;
 q01 / q99 = the day's percentiles of WS - executedAt. * centred on the Confirmed status second: an assumption the known semantics contradict
 (the match can precede it by up to 10 s). Bound types: EV lo unknown, hi evidence_bound; S1 / S2 lo / hi assumption_range.
 per event: S1 / S2 = the intersection over its legs, capped by EV (admissible = assumption AND evidence): ok / truncated_by_evidence /
 contradictory_evidence / contradictory_legs. A contradictory event's instant is unknown: in the value bounds it moves over the whole axis.
ORDERING classify(match window, cancel window): CANCEL_FIRST (cancel_hi < match_lo: the pull avoids the fill), MATCH_FIRST (match_hi <
 cancel_lo: too late), TIE (separated except at a touching endpoint), UNKNOWN (overlap with an open endpoint), AMBIGUOUS (bounded overlap),
 CONTRADICTORY (an empty match window).
VALUE: E = score improvement - exact STATES-null expectation (pf_keep_pull_clock_audit), quantities held at M50 (a stated limit: no fill
 absent at M50 is discovered, no queue re-simulated). E_min / E_max = exact extremes over each event's shared instant in its admissible
 window: the sum of its fills' step functions swept over their breakpoints.

READINGS (computed; declared before any data is read)
 V0 CONTROLS, else no output: (a) collapsed to the replay's own times E = +234,279 c and the score improvement 1,169,671 c (to 1 c); (b) v1's
    own functions on v1's hull windows reproduce v1's committed S1 / S2 bounds (to 1 c) and ordering counts (exactly); (c) every M50 fill
    has a candidate event (v1 and v2 links agree); (d) every event leg's loader fields equal its legs_timed2 row and its event id the
    taker-order rule (tk, else -1 - row); (e) every cross-tab row's cells sum to the replay's removed / kept totals.
 V1 SIGN (event-consistent, admissible windows, cancel latency 37 ms) per scenario S1 / S2 / EV: EXCESS-POSITIVE-IDENTIFIED (E_min > 0),
    EXCESS-NEGATIVE-IDENTIFIED (E_max < 0) or SIGN-NOT-IDENTIFIED; the same predicate at 0 and 100 ms printed beside it.
 V2 EVIDENCE COVERAGE (EV x CP): the share of the replay's removed / kept value (gross) and of the pulled orders whose ordering the evidence
    identifies - printed. Under EV only MATCH_FIRST is identifiable (no evidence bounds the match from below).
 V3 CONTRADICTIONS: events / fills / c whose S1 / S2 window the REST evidence contradicts; fills whose replay time lies after the evidence cap
    - printed by class and day.
 KNOWN (mode known: the 11-09 fill-clock audit's 99 confirmed own maker legs, 86 orders, consumed Nov 8, hash-pinned; lifecycle on the
    bot's clock, REST / Matched / the research anchor Z on the venue clock, offset 0 where they meet - as the 11-09 audits did):
  K0 the 11-09 counts reproduce (REST floor-second non-overlap, Matched floor-second non-overlap), else no output.
  K1 NO-FALSE-AVOIDANCE: match [place, min(first fill notice, cancel ack)] vs cancel effect [cancel request, cancel ack]: YES iff no known
     fill is CANCEL_FIRST and no window is empty.
  K2 DELAYED-STATUS-REFUSED: YES iff REST executedAt used as the match second contradicts >= 1 known lifecycle while its evidence use
     (-inf, (S + 1) * 1000] contradicts none, and window_from refuses it as a point.
  K3 IDENTITY-CHECK-FIRES: the Matched status second against the order's lifecycle: YES iff the true identity contradicts none and >= 95% of
     200 within-run permutations contradict >= 1.
  K4 (printed) the research anchor Z with the S1 / S2 widths (+-250 / +-600 ms, the widest class): legs it misses, known fills such a window
     calls CANCEL_FIRST, and whether one constant clock offset per run reconciles every leg (not fitted, not used).
SELFTEST (synthetic, no data): every class incl. TIE / UNKNOWN / CONTRADICTORY; window_from refusals; event intersection, evidence cap and
 statuses; link statuses; the event sweep = a dense grid of the shared instant, and on singletons v1's bounds; the collapsed control =
 pf_keep_pull_clock_audit.per_fill; K0-K4 on planted cases; an end-to-end rehearsal of run() on a synthetic source into a temporary dir.
 Mutations: pf_keep_pull_exec_windows_mutants.py.
LIMITS: consumed days; one declared rule; STATES is local selection; M50 quantities held (no queue / fill / eligibility re-simulation - the
 next gate); S1 / S2 / CR are assumption ranges; the alignment of the REST, WS and book clocks is not established; one partial day of known
 lifecycles; only same-side legs at the fill's market join an event.
Usage (run_light, PF_SLOT=claude): pf_keep_pull_exec_windows.py selftest | run | known
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_keep_pull as KP                                 # noqa: E402
import pf_keep_pull_clock_audit as A                      # noqa: E402
import pf_keep_pull_timing_audit as V1                    # noqa: E402  v1 (d1de7504): its own functions run the regression control

C, pd, I = KP.C, KP.pd, KP.I
OUT = Path(I.K.IO) / 'keep_pull_exec_windows_20261118'
OUT_K = Path(I.K.IO) / 'keep_pull_exec_windows_known_20261118'
FC = Path(I.K.IO) / 'fill_clock_20261109'                 # the 11-09 fill-clock audit (own legs, lifecycles, status stamps)
LIVE = Path('/home/yuqing/pmmm-data/review-exports/live-samewindow-20261108/live_vs_replay')
CC_SHA = '2f1a580bb6bb3689bf39b52ddbf5a7a443ea004e0d281274510f1193eb108a19'   # clock_cases.parquet (as pinned by pf_b1a_live_orders)
SHIFT = 50.                                               # the primary replay's trade shift (M50)
CANCEL_RANGE = (0., 100.)                                 # CR: the cancel takes effect between the decision and +100 ms (assumption)
LATS = (0., KP.CANCEL, 100.)
MATCH_SOURCES = ()                                        # sources with established match-instant semantics: none
SCEN = ('S1', 'S2', 'EV')
CWIN = ('CP', 'CR')
ORD = ('CANCEL_FIRST', 'MATCH_FIRST', 'AMBIGUOUS', 'TIE', 'UNKNOWN', 'CONTRADICTORY')
EVC = ('WS', 'E2', 'imputed', 'mixed')
WST = ('ok', 'truncated_by_evidence', 'contradictory_evidence', 'contradictory_legs')
LINK = ('unlinked', 'unique_event', 'multi_event')
LEG_COLS = ['mid', 'side', 'anc', 'ptick', 'shares', 'how', 't_ms', 't_e2', 'exec_s', 'row_id', 'group']
V1_COLS = ['mid', 'side', 'anc', 'ptick', 'how', 't_ms', 't_e2', 'exec_s']
RAW_COLS = ['mid', 'exec_s', 'first_recv_ns', 'side', 'ptick', 'shares', 'price_executed', 'signer', 'taker_index', 'taker_quote', 't_ms', 'how', 't_e2']
RECON = ('mid', 'side', 'ptick', 'shares', 'exec_s', 'how', 't_ms', 't_e2')


def classify(m_lo, m_hi, c_lo, c_hi):
    """Ordering class (index into ORD) of match windows [m_lo, m_hi] against cancel-effect windows [c_lo, c_hi] on one clock: closed
    intervals, +-inf an open endpoint, m_lo > m_hi an empty (contradictory) window."""
    m_lo, m_hi, c_lo, c_hi = np.broadcast_arrays(*(np.asarray(x, float) for x in (m_lo, m_hi, c_lo, c_hi)))
    C.require(bool(np.all(c_lo <= c_hi)), 'an empty cancel window')
    cf = c_hi < m_lo; mf = m_hi < c_lo; tie = ~cf & ~mf & ((m_hi == c_lo) | (c_hi == m_lo))
    opn = ~(np.isfinite(m_lo) & np.isfinite(m_hi) & np.isfinite(c_lo) & np.isfinite(c_hi))
    return np.where(m_lo > m_hi, 5, np.where(cf, 0, np.where(mf, 1, np.where(tie, 3, np.where(opn, 4, 2)))))


def window_from(kind, x, point=False):
    """A match window (lo, hi) from one typed observation, vectorized over x. 'rest_confirmed_s': the REST executedAt second (the Confirmed
    status; the match precedes the end of that second); 'matched_status_s': own orders' Matched status second (floor reading - an
    assumption); 'lifecycle': x = (place, upper), own orders' causal interval; 'match_instant': needs a source in MATCH_SOURCES. A status
    asked for a point is refused: a delayed status time is never a match time."""
    if kind in ('rest_confirmed_s', 'matched_status_s') and point:
        raise ValueError(f'REFUSED: {kind} is a status timestamp, not the matching instant')
    if kind == 'rest_confirmed_s':
        x = np.asarray(x, float); return np.full(x.shape, -np.inf), (x + 1.) * 1000.
    if kind == 'matched_status_s':
        x = np.asarray(x, float); return x * 1000., x * 1000. + 999.999
    if kind == 'lifecycle':
        lo, hi = (np.asarray(v, float) for v in x); C.require(bool(np.all(lo <= hi)), 'reversed causal bounds'); return lo, hi
    if kind == 'match_instant':
        raise ValueError(f'REFUSED: no source has established match-instant semantics (MATCH_SOURCES = {MATCH_SOURCES})')
    raise ValueError(f'unknown observation kind {kind}')


def leg_windows(L):
    """Per leg (columns how, t_ms, t_e2, exec_s and its day's h, q01, q99): class (0 WS, 1 E2, 2 imputed) and {scenario: (lo, hi)}."""
    ws = (L.how.values > 0) & np.isfinite(L.t_ms.values); e2 = ~ws & np.isfinite(L.t_e2.values); S = L.exec_s.values * 1000.
    cls = np.where(ws, 0, np.where(e2, 1, 2)); x = np.where(ws, L.t_ms.values, np.where(e2, L.t_e2.values, S))
    h, q01, q99 = L.h.values, L.q01.values, L.q99.values
    return cls, dict(S1=(np.where(cls < 2, x - 250., x - 500.), np.where(ws, x, np.where(e2, x + 250., x + 500.))),
                     S2=(np.where(cls < 2, x - h, x + q01), np.where(cls < 2, x + h, x + q99)), EV=window_from('rest_confirmed_s', L.exec_s.values))


def events_of(L):
    """Per event (day, group): market, side(s), the loader's anchor, legs, class, WS span and per scenario the intersection of its legs'
    windows capped by the evidence, with its status (WST)."""
    cls, W = leg_windows(L)
    g = pd.DataFrame(dict(day=L.day.values, group=L.group.values, m=L.mid.values, side=L.side.values, anc=L.anc.values, cls=cls,
                          wst=np.where(cls == 0, L.t_ms.values, np.nan), EV_hi=W['EV'][1], S1_lo=W['S1'][0], S1_hi=W['S1'][1],
                          S2_lo=W['S2'][0], S2_hi=W['S2'][1]))
    E = g.groupby(['day', 'group'], sort=True).agg(m=('m', 'first'), sides=('side', 'nunique'), side=('side', 'min'), anc=('anc', 'min'),
                                                   anc_max=('anc', 'max'), legs=('cls', 'size'), cmin=('cls', 'min'), cmax=('cls', 'max'),
                                                   ws_lo=('wst', 'min'), ws_hi=('wst', 'max'), EV_hi=('EV_hi', 'min'), S1_lo=('S1_lo', 'max'),
                                                   S1_hi=('S1_hi', 'min'), S2_lo=('S2_lo', 'max'), S2_hi=('S2_hi', 'min')).reset_index()
    C.require(bool((E.anc == E.anc_max).all()), 'an event with two loader anchors')
    E['cls'] = np.where(E.cmin == E.cmax, E.cmin, 3); E['ws_span'] = E.ws_hi - E.ws_lo; E['EV_lo'] = -np.inf; E['EV_st'] = 0
    for k in ('S1', 'S2'):
        legs_empty = (E[f'{k}_lo'] > E[f'{k}_hi']).values; cap = E.EV_hi.values < E[f'{k}_hi'].values
        E[f'{k}_rhi'] = E[f'{k}_hi']; E[f'{k}_hi'] = np.minimum(E[f'{k}_hi'], E.EV_hi)
        E[f'{k}_st'] = np.where(legs_empty, 3, np.where(E[f'{k}_lo'].values > E[f'{k}_hi'].values, 2, np.where(cap, 1, 0)))
    return E.drop(columns=['anc_max', 'cmin', 'cmax', 'ws_lo', 'ws_hi'])


def link(Fd, LT):
    """Candidate events of each fill: the legs of its market and side whose anchor equals the fill's (anchor = replay time + 50 ms + posting
    time, to 0.01 ms, as v1) and whose price can fill the order (at or through it) name the events (taker orders). -> pairs (fid, group) and
    every leg of those events in LT (all prices: one instant)."""
    U = LT[['mid', 'side', 'anc']].drop_duplicates().sort_values('anc'); U = U.assign(mid=U.mid.astype(np.int64), side=U.side.astype(np.int64))
    x = Fd.assign(m=Fd.m.astype(np.int64), side=Fd.side.astype(np.int64)).sort_values('a')
    x = pd.merge_asof(x, U.rename(columns={'mid': 'm'}), left_on='a', right_on='anc', by=['m', 'side'], direction='nearest', tolerance=0.01)
    y = x.dropna(subset=['anc']).merge(LT.assign(m=LT.mid.astype(np.int64), side=LT.side.astype(np.int64))[['m', 'side', 'anc', 'ptick', 'group']],
                                       on=['m', 'side', 'anc'])
    y = y[np.where(y.side.values == 0, y.ptick.values <= y.p.values, y.ptick.values >= y.p.values)]
    pairs = y[['fid', 'group']].drop_duplicates().reset_index(drop=True)
    return pairs, LT[LT.group.isin(pairs.group.unique())]


def day_env(day):
    """The day's measured envelopes (v1's) and the hashes of every file this audit reads for the day."""
    S1d = Path(f'{I.IO}/step1/{day}'); b = json.loads((S1d / 'booktime.json').read_text()); t = json.loads((S1d / 'timing.json').read_text())
    files = (S1d / 'legs_timed2.parquet', S1d / 'legs_tk.parquet', S1d / 'booktime.json', S1d / 'timing.json', Path(C.V.E.OUTD) / f'day_{day}_imp.json')
    return dict(h=float(b['E2']['p90']), q01=float(t['offset_ms_pct']['p01']), q99=float(t['offset_ms_pct']['p99']), inputs={str(p): C.sha(p) for p in files})


def day_rows(day):
    """The day's legs_timed2 rows (the loader's row_id is the row position) with each leg's taker order (pf_txmap: tk, tk_status)."""
    S1d = Path(f'{I.IO}/step1/{day}'); L = pd.read_parquet(S1d / 'legs_timed2.parquet', columns=RAW_COLS)
    tk = pd.read_parquet(S1d / 'legs_tk.parquet', columns=['tk', 'tk_status']); C.require(len(tk) == len(L), f'{day}: legs_tk not row-aligned')
    return pd.concat([L.reset_index(drop=True), tk.reset_index(drop=True)], axis=1)


def link_day(day, Fd, src, env, rows):
    """One day's fills, market batch by market batch through the replay's own loader -> (v1's hull per fill, candidate pairs, the events'
    legs with their legs_timed2 identity and raw observations, identity mismatches)."""
    offset = src.offset(day); H1, P, LG = [], [], []; markets = sorted(set(Fd.m.astype(int)))
    for k in range(0, len(markets), 24):
        batch = markets[k:k + 24]; LT = src.load(day, set(batch), offset)['LT'][LEG_COLS]; f = Fd[Fd.m.isin(batch)]
        H1.append(V1.link(f, LT[V1_COLS], env['h'], env['q01'], env['q99'])); p, lg = link(f, LT); P.append(p); LG.append(lg)
        del LT; C.guard('exec-windows link')
    H1, P = pd.concat(H1), pd.concat(P, ignore_index=True); LG = pd.concat(LG, ignore_index=True).drop_duplicates('row_id').reset_index(drop=True)
    R = rows.iloc[LG.row_id.values.astype(np.int64)].reset_index(drop=True); bad = 0
    for c in RECON:
        a, b = LG[c].values.astype(float), R[c].values.astype(float); bad += int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())
    tk = R.tk.values.astype(np.int64); bad += int((LG.group.values.astype(np.int64) != np.where(tk >= 0, tk, -1 - LG.row_id.values.astype(np.int64))).sum())
    LG = pd.concat([LG, R[[c for c in RAW_COLS if c not in LEG_COLS] + ['tk', 'tk_status']]], axis=1)
    return H1, P, LG, bad


def contrib(tau, F, CW, lat=KP.CANCEL):
    """A fill's contribution to E if its match time were tau (ms after posting) and every cancel landed lat ms after its decision. F: per
    fill val, pulled, T (the score's step), s (stratum), endj, lend (structural landing, inf if none). As v1 at lat = 37 ms."""
    La = KP.landing(np.arange(1, KP.NSTEP + 1), lat); top = np.minimum(np.searchsorted(La, tau, side='right'), F['endj'] - 1)
    W = np.where(top >= KP.J0, CW[F['s'], np.maximum(top, 0)] - CW[F['s'], KP.J0 - 1], 0.)
    c = -F['val'] * (np.where(F['pulled'] & (tau >= KP.landing(np.clip(F['T'], 1, KP.NSTEP), lat)), 1., 0.) - W)
    return np.where((tau < KP.PLACE) | (tau > KP.H) | (tau >= F['lend']), 0., c)


def sweep(ev, wlo, whi, d, F, CW, lat, chunk=4000):
    """Exact min / max over ONE shared instant per event of the sum of its fills' contributions. ev: event index per fill; wlo / whi: the
    event's window on its own axis (+-inf allowed); d: the fill's posting time on that axis (tau_fill = tau_event - d). The sum is a
    right-continuous step function: evaluated at the window's start and after every breakpoint in (wlo, whi], equal instants aggregated."""
    nE = len(wlo); bp = np.r_[KP.landing(np.arange(1, KP.NSTEP + 1), lat), KP.PLACE, KP.H, np.nextafter(KP.H, np.inf)]
    s0 = wlo[ev] - d; fin = np.isfinite(s0); g0 = np.zeros(nE)
    np.add.at(g0, ev, np.where(fin, contrib(np.where(fin, s0, 0.), F, CW, lat), 0.)); mn, mx = g0.copy(), g0.copy()
    o = np.argsort(ev, kind='stable'); cuts = np.r_[0, np.flatnonzero(np.diff(ev[o])) + 1, len(o)]; i = 0
    while i < len(cuts) - 1:
        j = i + 1
        while j < len(cuts) - 1 and cuts[j + 1] - cuts[i] <= chunk:
            j += 1
        idx = o[cuts[i]:cuts[j]]; i = j; e_ = ev[idx]
        B = np.sort(np.c_[np.broadcast_to(bp, (len(idx), len(bp))), F['lend'][idx]], axis=1); Bev = d[idx, None] + B
        dup = np.zeros(B.shape, bool); dup[:, 1:] = B[:, 1:] == B[:, :-1]                     # one jump per breakpoint of a fill (lend can equal a landing)
        fi, bi = np.nonzero(np.isfinite(B) & ~dup & (Bev > wlo[e_, None]) & (Bev <= whi[e_, None]))
        if not len(fi):
            continue
        rel = B[fi, bi]; sub = {k: v[idx][fi] for k, v in F.items()}
        dl = contrib(rel, sub, CW, lat) - contrib(np.nextafter(rel, -np.inf), sub, CW, lat)
        ee, bb = e_[fi], Bev[fi, bi]; q = np.lexsort((bb, ee)); ee, bb, dl = ee[q], bb[q], dl[q]
        st = np.flatnonzero(np.r_[True, (np.diff(ee) != 0) | (np.diff(bb) != 0)]); du = np.add.reduceat(dl, st); eu = ee[st]
        cs = np.cumsum(du); s_ = np.flatnonzero(np.r_[True, np.diff(eu) != 0]); base = np.r_[0., cs[s_[1:] - 1]]
        v = g0[eu] + cs - np.repeat(base, np.diff(np.r_[s_, len(eu)])); E_ = eu[s_]
        mn[E_] = np.minimum(mn[E_], np.minimum.reduceat(v, s_)); mx[E_] = np.maximum(mx[E_], np.maximum.reduceat(v, s_))
        C.guard('exec-windows sweep')
    return mn, mx


def event_axis_windows(EVT, k):
    """Per event, scenario k's window on the event's own axis (its anchor = 0); a contradictory event's instant is unknown: (-inf, inf)."""
    bad = EVT[f'{k}_st'].values >= 2; anc = EVT.anc.values
    return np.where(bad, -np.inf, EVT[f'{k}_lo'].values - anc), np.where(bad, np.inf, EVT[f'{k}_hi'].values - anc)


def combine(fid, cls, n):
    """Per fill: the class every candidate event agrees on, else AMBIGUOUS (identity unknown among candidates that disagree); -1 if none."""
    lo = np.full(n, 99); hi = np.full(n, -1); np.minimum.at(lo, fid, cls); np.maximum.at(hi, fid, cls)
    return np.where(hi < 0, -1, np.where(lo == hi, lo, 2))


def xtab(cls, removed, val, mask):
    """Rows REMOVED / KEPT x ORD: fills, net c (what removing them avoids, -value) and gross c; each row with its own total."""
    out = {}
    for rn, rm in (('REMOVED', removed), ('KEPT', ~removed)):
        m = mask & rm; out[rn] = {'_total': dict(fills=int(m.sum()), net=float(-val[m].sum()), gross=float(np.abs(val[m]).sum()))}
        for c, nm in enumerate(ORD):
            x = m & (cls == c); out[rn][nm] = dict(fills=int(x.sum()), net=float(-val[x].sum()), gross=float(np.abs(val[x]).sum()))
    return out


def conserved(X):
    """Every row's cells sum to its total (fills exactly, c to 1e-6 relative)."""
    for r in X.values():
        t = r['_total']; f = sum(r[nm]['fills'] for nm in ORD); g = sum(r[nm]['gross'] for nm in ORD); nt = sum(r[nm]['net'] for nm in ORD)
        if f != t['fills'] or abs(g - t['gross']) > 1e-6 * max(1., t['gross']) or abs(nt - t['net']) > 1e-6 * max(1., abs(t['gross'])):
            return False
    return True


def sign(mn, mx):
    return 'EXCESS-POSITIVE-IDENTIFIED' if mn > 0 else 'EXCESS-NEGATIVE-IDENTIFIED' if mx < 0 else 'SIGN-NOT-IDENTIFIED'


def v1_reference():
    """v1's committed numbers (its output JSON, hash-equal to the copy committed in d1de7504)."""
    p = V1.OUT / 'timing_audit.json'; q = HERE.parent / 'reviews' / 'keep_pull_timing_audit_20261118' / 'timing_audit.json'
    C.require(C.sha(p) == C.sha(q), 'v1 output differs from its committed copy'); R = json.loads(p.read_text())['R']
    return {k: dict(E_min=R[k]['E_min'], E_max=R[k]['E_max'], order={nm: R[k]['order'][nm]['fills'] for nm in ('AFTER', 'BEFORE', 'AMBIGUOUS')}) for k in ('S1', 'S2')}


def run(out=OUT, src=None, tables=None, rows_of=None, env_of=None, v1=None, days=None, expect=(234279., 1169671.), tol=1.):
    """The ledger on the committed keep / pull run (or, in the selftest's rehearsal, on synthetic tables, source and output dir)."""
    start = time.time(); out = Path(out); C.require(not out.exists(), 'ledger output exists')
    if tables is None:
        selftest()
    O, Fl = tables if tables is not None else KP.load(KP.OUT); days = list(days or KP.DAYS); src = src or KP.Tape()
    rows_of = rows_of or day_rows; env_of = env_of or day_env; v1 = v1 or v1_reference(); La = KP.landing(np.arange(1, KP.NSTEP + 1))
    E = O[O.status == 'eligible'].reset_index(drop=True); cell = E.groupby(['day', 'm', 'side', 'phase'], sort=True).ngroup().values
    T = KP.score_steps(E[[f'f{j:02d}' for j in range(KP.NSTEP + 1)]].values, E.side.values, E.p.values)[KP.DECLARED]
    endj = A.endj_of(E.t_full.values, E.g_struct.values); act, CW = V1.null_weights(T, cell, endj); ri, t, val, srcc = A.fills_of(E, Fl, SHIFT)
    n = len(ri); lend = np.where(np.isnan(E.land_struct.values[ri]), np.inf, E.land_struct.values[ri]); t0 = E.t.values[ri]
    F = dict(val=val, pulled=act[ri], T=T[ri], s=cell[ri], endj=endj[ri], lend=lend); r37 = La[np.clip(T[ri], 1, KP.NSTEP) - 1]
    F1 = dict(val=val, pulled=act[ri], r=r37, s=cell[ri], endj=endj[ri], lend=lend)
    pf = A.per_fill(T, cell, endj, (ri, t, val, srcc)); e0 = float(contrib(t, F, CW).sum()); s0 = float(pf['score'].sum())
    ok_a = abs(e0 - float(pf['score'].sum() - pf['null'].sum())) < 1e-6 and abs(e0 - expect[0]) <= tol and abs(s0 - expect[1]) <= tol
    if not ok_a:
        sys.exit(f'EXEC-WINDOW LEDGER FAILED -> no output: collapsed E {e0:,.1f} / score {s0:,.1f}, not {expect[0]:+,.0f} / {expect[1]:,.0f}')
    Fd = pd.DataFrame(dict(fid=np.arange(n), day=E.day.values[ri], m=E.m.values[ri], side=E.side.values[ri], a=t + SHIFT + t0, p=E.p.values[ri]))
    H1, P, LG, env, recon = [], [], [], {}, {}
    for d in days:
        env[d] = env_of(d); rows = rows_of(d); h_, p_, lg, recon[d] = link_day(d, Fd[Fd.day == d], src, env[d], rows); del rows
        H1.append(h_); P.append(p_.assign(day=d)); LG.append(lg.assign(day=d, h=env[d]['h'], q01=env[d]['q01'], q99=env[d]['q99']))
        print(json.dumps(dict(stage='link', day=d, fills=int((Fd.day == d).sum()), linked=int(p_.fid.nunique()), legs=len(lg), identity_mismatches=recon[d],
                              seconds=round(time.time() - start), rss=round(I.K.rss_mib()))), flush=True)
    H1, P, LG = pd.concat(H1), pd.concat(P, ignore_index=True), pd.concat(LG, ignore_index=True)
    # V0 (b) v1's own functions on v1's hull windows; (c) both links cover every fill; (d) identities
    C.require(H1.index.is_unique, 'a fill linked twice by v1'); H1 = H1.reindex(np.arange(n)); unl = int(H1.legs.isna().sum())
    ok_c = unl == 0 and set(P.fid.unique()) == set(range(n)); pl = F['pulled']
    reg = {}
    for k, (a_, b_) in (('S1', ('lo1', 'hi1')), ('S2', ('lo2', 'hi2'))):
        lo, hi = H1[a_].values - t0, H1[b_].values - t0; mn, mx = V1.bounds(lo, hi, F1, CW); o = np.where(pl, V1.order_of(lo, hi, r37), -1)
        reg[k] = dict(E_min=float(mn.sum()), E_max=float(mx.sum()), order={nm: int(((o == c) & pl).sum()) for c, nm in enumerate(('AFTER', 'BEFORE', 'AMBIGUOUS'))})
    ok_b = all(abs(reg[k][x] - v1[k][x]) <= tol for k in reg for x in ('E_min', 'E_max')) and all(reg[k]['order'] == v1[k]['order'] for k in reg)
    ok_d = sum(recon.values()) == 0
    # events, candidate windows, classes
    EVT = events_of(LG); EVT['e'] = np.arange(len(EVT)); P = P.merge(EVT[['day', 'group', 'e']], on=['day', 'group'], how='left', validate='many_to_one')
    C.require(bool(P.e.notna().all()), 'a candidate event without legs'); pe = P.e.values.astype(np.int64); pf_ = P.fid.values.astype(np.int64)
    ncand = np.bincount(pf_, minlength=n); link_st = np.where(ncand == 0, 0, np.where(ncand == 1, 1, 2))
    cw = dict(CP=(r37, r37), CR=(KP.STEP * np.clip(T[ri], 1, KP.NSTEP) + CANCEL_RANGE[0], KP.STEP * np.clip(T[ri], 1, KP.NSTEP) + CANCEL_RANGE[1]))
    cls = {}
    for k in SCEN:
        lo = EVT[f'{k}_lo'].values[pe] - t0[pf_]; hi = EVT[f'{k}_hi'].values[pe] - t0[pf_]
        for c in CWIN:
            cls[k, c] = np.where(pl, combine(pf_, classify(lo, hi, cw[c][0][pf_], cw[c][1][pf_]), n), -1)
    removed = t >= r37; X = {f'{k}|{c}': xtab(cls[k, c], removed, val, pl) for k in SCEN for c in CWIN}; ok_e = all(conserved(x) for x in X.values())
    V0 = dict(a_collapsed=ok_a, b_v1_regression=ok_b, c_linked=ok_c, d_identity=ok_d, e_conservation=ok_e)
    if not all(V0.values()):
        sys.exit(f'EXEC-WINDOW LEDGER FAILED -> no output: V0 {V0} (v1 regression {reg} vs {v1}; unlinked {unl}; identity mismatches {recon})')
    # value bounds: one shared instant per event (unique fills), each candidate window independently (multi fills)
    uf = np.flatnonzero(link_st == 1); mf = np.flatnonzero(link_st == 2); e_u = np.zeros(n, np.int64); e_u[pf_[ncand[pf_] == 1]] = pe[ncand[pf_] == 1]
    ue, ev_u = np.unique(e_u[uf], return_inverse=True); anc = EVT.anc.values; mp = np.flatnonzero(ncand[pf_] > 1)
    B = {}
    for k in SCEN:
        wl, wh = event_axis_windows(EVT, k)
        for lat in LATS:
            mn_e, mx_e = sweep(ev_u, wl[ue], wh[ue], t0[uf] - anc[e_u[uf]], {x: v[uf] for x, v in F.items()}, CW, lat)
            mn_p, mx_p = sweep(np.arange(len(mp)), wl[pe[mp]], wh[pe[mp]], t0[pf_[mp]] - anc[pe[mp]], {x: v[pf_[mp]] for x, v in F.items()}, CW, lat)
            fmn = np.full(n, np.inf); fmx = np.full(n, -np.inf); np.minimum.at(fmn, pf_[mp], mn_p); np.maximum.at(fmx, pf_[mp], mx_p)
            ev_side, ev_day, ev_cls = EVT.side.values[ue], EVT.day.values[ue], np.array(EVC)[EVT.cls.values[ue]]
            br = dict(side={}, cls={}, day={})
            for nm, key_e, key_f in (('side', np.where(ev_side == 0, 'bid', 'ask'), np.where(Fd.side.values == 0, 'bid', 'ask')), ('cls', ev_cls, np.full(n, 'multi')),
                                     ('day', ev_day, Fd.day.values)):
                for g in sorted(set(key_e) | set(key_f[mf])):
                    br[nm][str(g)] = [float(mn_e[key_e == g].sum() + fmn[mf][key_f[mf] == g].sum()), float(mx_e[key_e == g].sum() + fmx[mf][key_f[mf] == g].sum())]
            B[k, lat] = dict(E_min=float(mn_e.sum() + fmn[mf].sum()), E_max=float(mx_e.sum() + fmx[mf].sum()), by=br)
    # coverage, decisions, contradictions
    cls_name = np.array(EVC + ('multi', 'unlinked'))[np.where(link_st == 1, EVT.cls.values[e_u], np.where(link_st == 2, 4, 5))]
    side_nm = np.where(Fd.side.values == 0, 'bid', 'ask'); gross = np.abs(val); cov = {}
    for k in SCEN:
        ident = np.isin(cls[k, 'CP'], (0, 1)) & pl; cov[k] = {nm: {str(g): float(gross[ident & (key == g)].sum() / max(gross[pl & (key == g)].sum(), 1e-12))
                                                                  for g in sorted(set(key[pl]))} for nm, key in (('day', Fd.day.values), ('side', side_nm), ('cls', cls_name),
                                                                                                                 ('link', np.array(LINK)[link_st]))}
    po = np.flatnonzero(act); nf = np.bincount(ri, minlength=len(E)); dec = {}
    for k in SCEN:
        unev = np.bincount(ri[pl & ~np.isin(cls[k, 'CP'], (0, 1))], minlength=len(E))
        dec[k] = dict(pulled_orders=int(len(po)), no_m50_fill=int((nf[po] == 0).sum()), evaluable=int(((nf[po] > 0) & (unev[po] == 0)).sum()),
                      not_evaluable=int((unev[po] > 0).sum()))
    V3 = {}
    for k in ('S1', 'S2'):
        st = EVT[f'{k}_st'].values; bad_f = (link_st == 1) & (st[e_u] >= 2)
        V3[k] = dict(events={WST[s]: int((st == s).sum()) for s in range(4)}, events_by_cls={EVC[c]: {WST[s]: int(((st == s) & (EVT.cls.values == c)).sum()) for s in range(4)} for c in range(4)},
                     fills_contradictory=int(bad_f.sum()), gross_contradictory=float(gross[bad_f].sum()),
                     by_day={d: int((bad_f & (Fd.day.values == d)).sum()) for d in days})
    late = (link_st == 1) & (t + t0 > EVT.EV_hi.values[e_u]); V3['replay_after_evidence'] = dict(fills=int(late.sum()), gross=float(gross[late].sum()),
                                                                                                 pulled_fills=int((late & pl).sum()))
    read = {'V0 CONTROLS': 'YES'}
    for k in SCEN:
        read[f'V1 SIGN {k}'] = sign(B[k, KP.CANCEL]['E_min'], B[k, KP.CANCEL]['E_max'])
    R = dict(fills=n, eligible=len(E), counted=len(O), status={str(s): int(v) for s, v in O.status.value_counts().items()}, link={LINK[s]: int((link_st == s).sum()) for s in range(3)},
             events=len(EVT), events_by_cls={EVC[c]: int((EVT.cls == c).sum()) for c in range(4)}, events_multi_side=int((EVT.sides > 1).sum()),
             legs=len(LG), legs_tk_status={str(s): int(v) for s, v in LG.tk_status.value_counts().items()}, env=env, collapsed_E=e0, score=s0, v1_regression=reg,
             xtab=X, bounds={f'{k}|{lat:g}': v for (k, lat), v in B.items()}, coverage=cov, decisions=dec, V3=V3, sim_removed=float(-val[pl & removed].sum()),
             sim_kept=float(-val[pl & ~removed].sum()))
    # the ledger tables
    out.mkdir(parents=True); files = {}
    EVT.to_parquet(out / 'ledger_events.parquet', index=False)
    LG.drop(columns=['h', 'q01', 'q99']).to_parquet(out / 'ledger_legs.parquet', index=False)
    P[['fid', 'day', 'group', 'e']].to_parquet(out / 'ledger_fill_events.parquet', index=False)
    ix = pd.Series(np.arange(len(E)), index=pd.MultiIndex.from_arrays([E.day.values, E.oid.values])); f50 = Fl[Fl.clock == SHIFT]
    q50 = f50.q.values[~np.isnan(ix.reindex(pd.MultiIndex.from_arrays([f50.day.values, f50.oid.values])).values)]; C.require(len(q50) == n, 'fill quantities misaligned')
    FL = pd.DataFrame(dict(fid=np.arange(n), day=Fd.day.values, oid=E.oid.values[ri], m=Fd.m.values, side=Fd.side.values, p=Fd.p.values, t0=t0, t_replay=t, anchor=Fd.a.values,
                           q=q50, value=val, src=srcc, pulled=pl, T=T[ri], landing37=r37, replay_removed=removed & pl, link=np.array(LINK)[link_st], n_events=ncand,
                           event=np.where(link_st == 1, e_u, -1), activation_assumed=KP.PLACE, notice_assumed=KP.NOTICE,
                           **{f'cls_{k}_{c}': cls[k, c] for k in SCEN for c in CWIN}))
    FL.to_parquet(out / 'ledger_fills.parquet', index=False)
    OL = O[['day', 'oid', 'm', 'side', 't', 'status']].copy(); ix = pd.MultiIndex.from_arrays([E.day.values, E.oid.values])
    for nm, v in (('T', T), ('pulled', act), ('endj', endj), ('m50_fills', nf), ('m50_value', np.bincount(ri, weights=val, minlength=len(E)))):
        OL[nm] = pd.Series(v, index=ix).reindex(pd.MultiIndex.from_arrays([OL.day.values, OL.oid.values])).values
    for k in SCEN:
        unev = np.bincount(ri[pl & ~np.isin(cls[k, 'CP'], (0, 1))], minlength=len(E))
        ev_ = np.where(~act, 'not_pulled', np.where(nf == 0, 'no_m50_fill', np.where(unev == 0, 'evaluable', 'not_evaluable')))
        OL[f'decision_{k}'] = pd.Series(ev_, index=ix).reindex(pd.MultiIndex.from_arrays([OL.day.values, OL.oid.values])).fillna('not_eligible').values
    OL.to_parquet(out / 'ledger_orders.parquet', index=False)
    for f in sorted(out.glob('*.parquet')):
        files[f.name] = C.sha(f)
    L = report(R, read, days)
    (out / 'exec_windows.txt').write_text('\n'.join(L) + '\n')
    deps = [Path(__file__).resolve(), Path(KP.__file__).resolve(), Path(A.__file__).resolve(), Path(V1.__file__).resolve(), Path(I.__file__).resolve()]
    C.dump(out / 'exec_windows.json', dict(asof=C.stamp(), readings=read, V0=V0, R=R, outputs=files, source_receipts={d: C.sha(KP.OUT / d / 'receipt.json') for d in days}
                                         if tables is None else {}, v1_reference=v1, dependencies={p.name: C.sha(p) for p in deps}, provenance=KP.provenance(),
                                         source_sha256=C.sha(Path(__file__).resolve()), seconds=time.time() - start))
    print('\n'.join(L), flush=True)
    return R, read


def report(R, read, days):
    def cell(x, tot):
        return f"{x['fills']:,} / {x['net']:+,.0f} c ({100 * x['gross'] / max(tot, 1e-12):.1f}% of the row's gross)"
    L = ['EXECUTION-WINDOW LEDGER of the keep / pull selection, declared score A+0 (pf_keep_pull_exec_windows.py; the committed keep / pull run, M50 fills, '
         'consumed Oct23-27; c summed; v2 of the ordering audit - supersedes v1\'s headline)',
         f"  {R['counted']:,} counted states ({', '.join(f'{k} {v:,}' for k, v in R['status'].items())}); {R['fills']:,} M50 fills; links "
         + ', '.join(f'{k} {v:,}' for k, v in R['link'].items()),
         f"  {R['events']:,} events (taker orders; {', '.join(f'{k} {v:,}' for k, v in R['events_by_cls'].items())}; {R['events_multi_side']:,} with legs on both sides); "
         f"{R['legs']:,} legs (tk_status {R['legs_tk_status']})",
         f"  V0 controls: collapsed E {R['collapsed_E']:+,.0f} / score {R['score']:,.0f}; v1 regression " + '; '.join(
             f"{k} [{v['E_min']:+,.0f}, {v['E_max']:+,.0f}] {v['order']}" for k, v in R['v1_regression'].items()), '',
         'PART 1 - the score\'s pulled orders\' fills: replay ordering (row) x window class (cell: fills / net c = what removing them avoids)',
         f"  replay: REMOVED {R['sim_removed']:+,.0f} c, KEPT {R['sim_kept']:+,.0f} c"]
    for key, X in R['xtab'].items():
        L.append(f'  {key}:')
        for rn in ('REMOVED', 'KEPT'):
            tot = X[rn]['_total']['gross']; L.append(f"    {rn:<7} ({X[rn]['_total']['fills']:,} fills, {X[rn]['_total']['net']:+,.0f} c): "
                                                    + ' | '.join(f'{nm} {cell(X[rn][nm], tot)}' for nm in ORD if X[rn][nm]['fills']))
    L += ['', 'PART 2 - coverage: share of the pulled orders\' fill value (gross) whose ordering is identified (CANCEL_FIRST or MATCH_FIRST) against the replay\'s cancel point']
    for k, v in R['coverage'].items():
        L.append(f'  {k}: ' + ' | '.join(f"{nm} " + ', '.join(f'{g} {100 * x:.1f}%' for g, x in d.items()) for nm, d in v.items()))
    L.append('  decisions (pulled orders): ' + ' | '.join(f"{k} evaluable {v['evaluable']:,} / not {v['not_evaluable']:,} / no M50 fill {v['no_m50_fill']:,} of {v['pulled_orders']:,}"
                                                         for k, v in R['decisions'].items()))
    L += ['', 'PART 3 - bounds on E = score improvement - exact STATES-null expectation (one shared instant per event in its admissible window; M50 quantities held)']
    for key, v in R['bounds'].items():
        L.append(f"  {key} ms: [{v['E_min']:+,.0f}, {v['E_max']:+,.0f}] | " + ' | '.join(f'{nm} ' + ', '.join(f'{g} [{a:+,.0f}, {b:+,.0f}]' for g, (a, b) in d.items())
                                                                              for nm, d in v['by'].items()))
    L += ['', 'PART 4 - contradictions (kept, never repaired)']
    for k in ('S1', 'S2'):
        v = R['V3'][k]; L.append(f"  {k}: events {v['events']}; contradictory fills {v['fills_contradictory']:,} ({v['gross_contradictory']:,.0f} c gross); by class {v['events_by_cls']}")
    v = R['V3']['replay_after_evidence']; L.append(f"  replay times after the REST evidence cap: {v['fills']:,} fills ({v['gross']:,.0f} c gross; {v['pulled_fills']:,} on pulled orders)")
    L += ['', 'READINGS (declared before any data was read)'] + [f'  {k}: {v}' for k, v in read.items()] + [
          '', 'MISSING EVIDENCE: no public capture bounds the match instant from below; the REST executedAt is a delayed status (an upper bound only); the WS '
          'pending link is by value; cancel effect and activation are configured points. Order-level match identity and timing (own orders, A4b) is the evidence that '
          'would identify the ordering.', 'LIMITS: consumed days; one declared rule; STATES is local selection; M50 quantities held; S1 / S2 / CR are assumption '
          'ranges; clock alignment between REST, WS and book stamps not established.']
    return L


def known_frame():
    """The 11-09 fill-clock audit's 99 confirmed own maker legs (hash-pinned) with their lifecycles, status stamps and research anchor."""
    sem = json.loads((FC / 'semantics_declaration.json').read_text()); rep = json.loads((FC / 'semantics_report.json').read_text())
    dec = json.loads((FC / 'declaration_v3.json').read_text())['input_sha']
    C.require(C.sha(FC / 'clock_cases.parquet') == CC_SHA == sem['input_sha']['clock_cases.parquet'], 'clock_cases changed since the fill-clock audit')
    C.require(C.sha(FC / 'status_clock_cases.parquet') == rep['artifact_sha'], 'status_clock_cases changed since the semantics audit')
    for r in ('v6', 'basis'):
        p = LIVE / f'core_orders_{r}.parquet'; C.require(C.sha(p) == dec[str(p)], f'{p.name} changed since the fill-clock audit')
    CC = pd.read_parquet(FC / 'clock_cases.parquet'); SC = pd.read_parquet(FC / 'status_clock_cases.parquet')
    core = pd.concat([pd.read_parquet(LIVE / f'core_orders_{r}.parquet', columns=['cid', 't_cancel']) for r in ('v6', 'basis')])
    C.require(core.cid.is_unique, 'a core order id twice')
    k = CC.merge(SC[['run', 'cid', 'order_hash', 'trade_id', 'matched_reported_ms']], on=['run', 'cid', 'order_hash', 'trade_id'], how='left', validate='one_to_one')
    k = k.merge(core, on='cid', how='left', validate='many_to_one')
    K = pd.DataFrame(dict(run=k.run.values, cid=k.cid.values, place=k.place.values, upper=k.upper.values, notice=k.first_positive_notice.values, ack=k.cancel.values,
                          req=k.t_cancel.values, S=k.exec_s.values.astype(float), M=k.matched_reported_ms.values / 1000., Z=k.Z.values))
    return K, dict(rest_floor_nonoverlap=rep['timestamp_checks']['rest_floor_nonoverlap'], matched_floor_nonoverlap=rep['timestamp_checks']['matched_floor_nonoverlap'])


def miss(lo, hi, K):
    """Windows [lo, hi] that miss the leg's lifecycle [place, upper] entirely."""
    return (hi < K.place.values) | (lo > K.upper.values)


def known_checks(K, ref, draws=200, seed=20261118):
    """K0-K4 on known own legs (K: run, cid, place, upper, notice, ack, req, S, M, Z; ref: the 11-09 counts)."""
    lo, hi = window_from('lifecycle', (K.place.values, K.upper.values)); can = np.isfinite(K.ack.values)
    c_lo = np.where(np.isfinite(K.req.values), K.req.values, K.place.values); c_hi = K.ack.values
    rest_pt = (K.S.values * 1000., K.S.values * 1000. + 999.999); rest_ev = window_from('rest_confirmed_s', K.S.values); mt = window_from('matched_status_s', K.M.values)
    k0 = dict(rest_floor_nonoverlap=int(miss(*rest_pt, K).sum()), matched_floor_nonoverlap=int(miss(*mt, K).sum()))
    c1 = classify(lo[can], hi[can], c_lo[can], c_hi[can]); k1 = {ORD[c]: int((c1 == c).sum()) for c in range(len(ORD))}
    try:
        window_from('rest_confirmed_s', K.S.values, point=True); refused = False
    except ValueError:
        refused = True
    c2 = classify(rest_pt[0][can], rest_pt[1][can], c_lo[can], c_hi[can])
    k2 = dict(point_contradictions=int(miss(*rest_pt, K).sum()), evidence_contradictions=int(miss(*rest_ev, K).sum()), point_refused=refused,
              point_cancel_first=int((c2 == 0).sum()), evidence_classes={ORD[c]: int((v == c).sum()) for v in [classify(rest_ev[0][can], rest_ev[1][can], c_lo[can], c_hi[can])]
                                                                         for c in range(len(ORD))})
    rng = np.random.default_rng(seed); hits = []
    for _ in range(draws):
        Kp = K.copy()
        for r in Kp.run.unique():
            ix = np.flatnonzero(Kp.run.values == r); pm = rng.permutation(ix); Kp.loc[Kp.index[ix], ['place', 'upper']] = K[['place', 'upper']].values[pm]
        hits.append(int(miss(*mt, Kp).sum()))
    hits = np.array(hits); k3 = dict(true_contradictions=k0['matched_floor_nonoverlap'], draws=draws, share_draws_firing=float((hits >= 1).mean()), median_hits=float(np.median(hits)))
    k4 = {}
    for nm, w in (('S1', 250.), ('S2', 600.)):
        zl, zh = K.Z.values - w, K.Z.values + w; c4 = classify(zl[can], zh[can], c_lo[can], c_hi[can])
        feas = {}
        for r in sorted(K.run.unique()):
            x = K.run.values == r; dlo = float((K.place.values[x] - zh[x]).max()); dhi = float((K.upper.values[x] - zl[x]).min())
            feas[str(r)] = dict(lo=dlo, hi=dhi, exists=bool(dlo <= dhi))
        k4[nm] = dict(missed=int(miss(zl, zh, K).sum()), cancel_first=int((c4 == 0).sum()), constant_offset=feas)
    read = {'K0 REPRODUCES 11-09': 'YES' if k0 == ref else 'NO',
            'K1 NO-FALSE-AVOIDANCE': 'YES' if k1['CANCEL_FIRST'] == 0 and k1['CONTRADICTORY'] == 0 and bool(np.all(lo <= hi)) else 'NO',
            'K2 DELAYED-STATUS-REFUSED': 'YES' if k2['point_contradictions'] >= 1 and k2['evidence_contradictions'] == 0 and refused else 'NO',
            'K3 IDENTITY-CHECK-FIRES': 'YES' if k3['true_contradictions'] == 0 and k3['share_draws_firing'] >= .95 else 'NO'}
    return dict(legs=len(K), orders=int(K.cid.nunique()), cancelled=int(can.sum()), K0=k0, K0_ref=ref, K1=k1, K2=k2, K3=k3, K4=k4), read


def known(out=OUT_K):
    start = time.time(); out = Path(out); C.require(not out.exists(), 'known-lifecycle output exists'); selftest()
    K, ref = known_frame(); R, read = known_checks(K, ref)
    if read['K0 REPRODUCES 11-09'] != 'YES':
        sys.exit(f"KNOWN-LIFECYCLE CHECK FAILED -> no output: K0 {R['K0']} != the 11-09 counts {ref}")
    L = ['KNOWN-ORDER LIFECYCLE CHECK of the window classifier (pf_keep_pull_exec_windows.py known; the 11-09 fill-clock audit\'s confirmed own maker legs, consumed Nov 8)',
         f"  {R['legs']} legs, {R['orders']} orders, {R['cancelled']} legs on orders with a cancel ack; lifecycle on the bot's clock, REST / Matched / Z on the venue clock (offset 0)",
         f"  K0 the 11-09 counts: {R['K0']} (11-09: {R['K0_ref']})",
         f"  K1 lifecycle match window vs the actual cancel-effect window: {R['K1']}",
         f"  K2 REST executedAt as the match second: {R['K2']['point_contradictions']} lifecycles contradicted, {R['K2']['point_cancel_first']} known fills it would call "
         f"CANCEL_FIRST; as the evidence bound: {R['K2']['evidence_contradictions']} contradicted, classes {R['K2']['evidence_classes']}; refused as a point: {R['K2']['point_refused']}",
         f"  K3 Matched second vs lifecycle: true identity {R['K3']['true_contradictions']} contradicted; {100 * R['K3']['share_draws_firing']:.1f}% of {R['K3']['draws']} "
         f"within-run permutations contradict >= 1 (median {R['K3']['median_hits']:.0f})"]
    for nm, v in R['K4'].items():
        L.append(f"  K4 research anchor Z +-{250 if nm == 'S1' else 600} ms ({nm} width): misses {v['missed']} lifecycles; would call {v['cancel_first']} known fills CANCEL_FIRST; "
                 f"constant offset per run {v['constant_offset']}")
    L += ['', 'READINGS (declared before any data was read)'] + [f'  {k}: {v}' for k, v in read.items()]
    out.mkdir(parents=True); (out / 'known.txt').write_text('\n'.join(L) + '\n'); K.to_parquet(out / 'known_cases.parquet', index=False)
    C.dump(out / 'known.json', dict(asof=C.stamp(), readings=read, R=R, inputs={p: C.sha(FC / p) for p in ('clock_cases.parquet', 'status_clock_cases.parquet')},
                                    outputs={'known_cases.parquet': C.sha(out / 'known_cases.parquet')}, provenance=KP.provenance(),
                                    source_sha256=C.sha(Path(__file__).resolve()), seconds=time.time() - start))
    print('\n'.join(L), flush=True)


def _brute(ev, wlo, whi, d, F, CW, lat):
    """Dense-grid min / max of each event's summed contribution (integer breakpoints in the selftest: a 0.5 ms grid hits every piece)."""
    nE = len(wlo); mn = np.full(nE, np.inf); mx = np.full(nE, -np.inf)
    for e in range(nE):
        f = np.flatnonzero(ev == e)
        if not len(f):
            mn[e] = mx[e] = 0.; continue
        a = wlo[e] if np.isfinite(wlo[e]) else float(d[f].min()) - 10.; b = whi[e] if np.isfinite(whi[e]) else float(d[f].max()) + KP.H + 10.
        g = np.r_[np.arange(np.ceil(a * 2) / 2, b + 1e-9, .5), a, b]; g = g[(g >= a) & (g <= b)]
        tot = sum(contrib(g - d[i], {k: np.full(len(g), v[i]) for k, v in F.items()}, CW, lat) for i in f)
        mn[e], mx[e] = float(tot.min()), float(tot.max())
    return mn, mx


def _synthetic(rng, day='20990101'):
    """A synthetic keep / pull day: orders (incl. two refusals), M50 / M250 fills, and the legs the loader would return (unique, multi and
    singleton events; a leg of a candidate event at a price that cannot fill)."""
    n = 40; side = rng.integers(0, 2, n); p = rng.integers(30, 70, n).astype(float); t = 1.0e6 + 1000. * np.arange(n) + rng.integers(0, 50, n)
    f = p[:, None] + rng.normal(0., 2., (n, KP.NSTEP + 1)); st = np.where(np.arange(n) < 38, 'eligible', 'refused_post_only')
    keep = np.arange(n) % 3 == 0; f[keep] = p[keep, None] + np.where(side[keep] == 0, 1., -1.)[:, None] * (1. + np.abs(rng.normal(0., 2., (int(keep.sum()), KP.NSTEP + 1))))
    O = pd.DataFrame(dict(day=day, oid=np.arange(n), m=np.where(np.arange(n) % 2, 11, 12), side=side, t=t, ws=0., phase=0, p=p, status=st, mark=p + rng.normal(0., 3., n),
                          g_struct=np.where(rng.random(n) < .1, 3000., np.nan), land_struct=np.where(rng.random(n) < .1, 3500., np.nan),
                          t_full=np.where(rng.random(n) < .2, 2500., np.nan), **{f'f{j:02d}': f[:, j] for j in range(KP.NSTEP + 1)}))
    fl, legs, g = [], [], 100
    for o in range(n):
        for _ in range(rng.integers(0, 4)):
            ls_ = O.land_struct[o]; tt = float(rng.integers(60, 4990 if np.isnan(ls_) else int(ls_)))   # the replay never fills after the structural landing
            fl.append((day, o, 50., tt, float(rng.integers(1, 30)), int(rng.integers(0, 4)))); fl.append((day, o, 250., tt + 7., 5., 0))
            a = tt + SHIFT + t[o]; S = np.floor(a / 1000.) + rng.integers(0, 2); kind = rng.integers(0, 3); ngr = 2 if rng.random() < .2 else 1
            for _k in range(ngr):
                g += 1; tk = g if rng.random() < .8 else -1
                for leg in range(rng.integers(1, 3)):
                    ws = kind == 0; tms = a + (0. if leg == 0 else float(rng.integers(0, 300))) if ws else np.nan
                    legs.append(dict(mid=int(O.m[o]), side=int(side[o]), anc=a, ptick=int(p[o]) if leg == 0 else int(p[o]) + (1 if side[o] == 0 else -1),
                                     shares=5., how=1 if ws else 0, t_ms=tms, t_e2=a if kind == 1 else np.nan, exec_s=float(S), tk=tk, tk_status=0 if tk >= 0 else 1,
                                     first_recv_ns=0, price_executed=p[o] / 100., signer='x', taker_index=0, taker_quote=0))
    Fl = pd.DataFrame(fl, columns=['day', 'oid', 'clock', 't', 'q', 'src']); L = pd.DataFrame(legs); L['row_id'] = np.arange(len(L))
    L['group'] = np.where(L.tk.values >= 0, L.tk.values, -1 - L.row_id.values); return O, Fl, L


class _Src:
    def __init__(self, L):
        self.L = L

    def offset(self, day):
        return 0.

    def load(self, day, batch, offset):
        return dict(LT=self.L[self.L.mid.isin(batch)].copy())


def selftest(rehearse=True):
    bad = []; rng = np.random.default_rng(KP.SEED)
    I_ = np.inf; c = classify(np.array([140., 50., 100., 137., 50., -I_, -I_, 300., 80., 150., 90., 100., 100., 210., 200.]),
                              np.array([300., 130., 200., 300., 137., 130., 200., 200., 120., 190., 180., 150., 110., 250., 250.]),
                              np.r_[np.full(11, 137.), 120., 120., 120., 120.], np.r_[np.full(11, 137.), 200., 200., 200., 200.])
    if c.tolist() != [0, 1, 2, 3, 3, 1, 4, 5, 1, 0, 2, 2, 1, 0, 3]:
        bad.append(f'classify: {c.tolist()}')
    for kind, kw in (('rest_confirmed_s', dict(point=True)), ('matched_status_s', dict(point=True)), ('match_instant', {})):
        try:
            window_from(kind, np.array([5.]), **kw); bad.append(f'window_from {kind} {kw} not refused')
        except ValueError:
            pass
    lo, hi = window_from('rest_confirmed_s', np.array([5.]))
    if not (np.isneginf(lo[0]) and hi[0] == 6000.):
        bad.append('REST executedAt evidence window')
    try:
        window_from('lifecycle', (np.array([10.]), np.array([5.]))); bad.append('a reversed lifecycle accepted')
    except ValueError:
        pass
    L = pd.DataFrame(dict(day='d', group=[1, 1, 2, 3, 4, 4, 5, 6], mid=7, side=0, anc=1000., how=[1, 1, 0, 1, 1, 1, 0, 0], t_ms=[1000., 1100., np.nan, 1000., 1000., 1300., np.nan, np.nan],
                          t_e2=[np.nan, np.nan, 1010., np.nan, np.nan, np.nan, np.nan, 1200.], exec_s=[2., 2., 0., 0., 2., 2., 3., 2.], h=600., q01=-900., q99=1050.))
    EV = events_of(L).set_index('group')
    want = {1: (0, 850., 1000., 0), 2: (1, 760., 1000., 1), 3: (0, 750., 1000., 0), 4: (0, 1050., 1000., 3), 5: (2, 2500., 3500., 0), 6: (1, 950., 1450., 0)}
    got = {g: (int(r.cls), float(r.S1_lo), float(r.S1_hi), int(r.S1_st)) for g, r in EV.iterrows()}
    if got != want:
        bad.append(f'events_of: {got} != {want}')
    L2 = L.copy(); L2.loc[L2.group == 3, 'exec_s'] = -1.                                      # evidence cap at 0 ms < the window's start
    if int(events_of(L2).set_index('group').S1_st[3]) != 2:
        bad.append('an evidence-contradicted event not marked contradictory_evidence')
    LT = pd.DataFrame(dict(mid=[7, 7, 7, 7, 7, 8, 7], side=[0, 0, 0, 0, 0, 1, 0], anc=[1000., 1000., 1000., 2000., 1000., 3000., 1000.], ptick=[50, 49, 51, 50, 50, 60, 52],
                           group=[9, 9, 9, 3, 4, 5, 8]))                                        # group 8: at the anchor, but it cannot fill a bid at 50
    Fd = pd.DataFrame(dict(fid=[0, 1, 2], m=[7, 7, 8], side=[0, 0, 1], a=[1000.004, 2000., 2500.], p=[50, 50, 60]))
    pr, lg = link(Fd, LT)
    if sorted(map(tuple, pr[['fid', 'group']].values.tolist())) != [(0, 4), (0, 9), (1, 3)] or sorted(lg.group.unique().tolist()) != [3, 4, 9] or len(lg) != 5:
        bad.append(f'link: {pr.values.tolist()} / {sorted(lg.group.unique().tolist())}')
    if combine(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), 3).tolist() != [2, 1, -1]:
        bad.append('combine: disagreeing candidates must be AMBIGUOUS, none = -1')
    CWp = np.zeros((1, KP.NSTEP + 2)); CWp[0, 5:] = .5; Fp = dict(val=np.array([-10.]), pulled=np.array([True]), T=np.array([5]), s=np.array([0]), endj=np.array([99]),
                                                               lend=np.array([np.inf]))
    got = [float(contrib(np.array([300.]), Fp, CWp, lat)[0]) for lat in LATS]
    if got != [5., 5., 0.]:
        bad.append(f'contrib at cancel latencies 0 / 37 / 100: {got} != [5, 5, 0] (score and null landings move with the latency)')
    wl_, wh_ = event_axis_windows(EV.reset_index(), 'S1'); gi = {g: i for i, g in enumerate(EV.index)}
    if not (np.isneginf(wl_[gi[4]]) and np.isposinf(wh_[gi[4]]) and (wl_[gi[1]], wh_[gi[1]]) == (-150., 0.)):
        bad.append('event-axis windows: a contradictory event must move over the whole axis')
    n = 240; cell = np.repeat(np.arange(20), 12); endj = rng.integers(KP.J0, KP.NSTEP + 2, n); T = np.where(rng.random(n) < .5, rng.integers(KP.J0, 40, n), KP.NSTEP + 1)
    m = 600; ri = rng.integers(0, n, m); t = np.r_[rng.integers(60, 4990, m - 30).astype(float), np.full(30, KP.H)]; val = rng.normal(0., 10., m)
    lend = np.where(rng.random(n) < .2, rng.integers(500, 4000, n).astype(float), np.inf); keep = t < lend[ri]; ri, t, val = ri[keep], t[keep], val[keep]; m = len(ri)
    act, CW = V1.null_weights(T, cell, endj); La = KP.landing(np.arange(1, KP.NSTEP + 1))
    F = dict(val=val, pulled=act[ri], T=T[ri], s=cell[ri], endj=endj[ri], lend=lend[ri]); F1 = dict(val=val, pulled=act[ri], r=La[np.clip(T[ri], 1, KP.NSTEP) - 1], s=cell[ri], endj=endj[ri], lend=lend[ri])
    pf = A.per_fill(T, cell, endj, (ri, t, val, np.zeros(m, int)))
    if abs(contrib(t, F, CW).sum() - (pf['score'].sum() - pf['null'].sum())) > 1e-6:
        bad.append('collapsed contributions differ from the clock audit per fill')
    lo = t - rng.integers(0, 600, m); hi = t + rng.integers(0, 600, m); mn1, mx1 = V1.bounds(lo, hi, F1, CW)
    mn, mx = sweep(np.arange(m), lo, hi, np.zeros(m), F, CW, KP.CANCEL)
    if not (np.allclose(mn, mn1, atol=1e-9) and np.allclose(mx, mx1, atol=1e-9)):
        bad.append('singleton sweep differs from v1 bounds')
    nE = 30; ev = rng.integers(0, nE, m); d = rng.integers(-300, 300, m).astype(float); ctr = rng.integers(0, 5000, nE).astype(float)
    wl = ctr - rng.integers(0, 700, nE); wh = ctr + rng.integers(0, 700, nE); wl[:4] = -np.inf; wh[2:6] = np.inf
    for lat in LATS:
        a1, b1 = sweep(ev, wl, wh, d, F, CW, lat, chunk=50); a2, b2 = _brute(ev, wl, wh, d, F, CW, lat)
        if not (np.allclose(a1, a2, atol=1e-9) and np.allclose(b1, b2, atol=1e-9)):
            bad.append(f'event sweep differs from the dense grid at lat {lat:g}: {int((~np.isclose(a1, a2)).sum())} minima, {int((~np.isclose(b1, b2)).sum())} maxima')
    Fz = dict(val=np.array([-10., 10.]), pulled=np.array([True, True]), T=np.array([5, 5]), s=np.array([0, 0]), endj=np.array([99, 99]), lend=np.array([np.inf, np.inf]))
    mz, xz = sweep(np.array([0, 0]), np.array([0.]), np.array([1000.]), np.zeros(2), Fz, np.zeros((1, KP.NSTEP + 2)), KP.CANCEL)
    if not (mz[0] == 0. and xz[0] == 0.):
        bad.append('two jumps of one event at one instant must be summed before the state is read (true sum 0 everywhere)')
    X = xtab(np.array([0, 1, 2, 2, 4, -1]), np.array([True, True, False, False, True, True]), np.array([-5., 3., 2., -1., 4., 9.]), np.array([True] * 5 + [False]))
    if not (conserved(X) and X['REMOVED']['CANCEL_FIRST']['net'] == 5. and X['KEPT']['AMBIGUOUS']['fills'] == 2 and X['REMOVED']['_total']['fills'] == 3):
        bad.append('cross-tab cells / conservation')
    X['KEPT']['AMBIGUOUS']['fills'] = 1
    if conserved(X):
        bad.append('conservation check does not fire on a lost fill')
    base = 1.79e12; pl_ = base + 1000. * np.arange(8) * 60.
    Kk = pd.DataFrame(dict(run=['a'] * 4 + ['b'] * 4, cid=np.arange(8), place=pl_, upper=pl_ + 160., notice=pl_ + 600., ack=np.r_[pl_[:7] + 160., np.nan],
                           req=pl_ + 100., S=np.floor((pl_ + 80.) / 1000.), M=np.floor((pl_ + 80.) / 1000.), Z=pl_ + 70.))
    Kk.loc[1, 'notice'] = pl_[1] + 90.; Kk.loc[1, 'upper'] = pl_[1] + 90.                      # notice before the cancel request: MATCH_FIRST
    Kk.loc[2, 'S'] = np.floor((pl_[2] + 80.) / 1000.) + 10.                                     # Confirmed 10 s late: the substitution must fire
    Kk.loc[3, 'Z'] = pl_[3] + 10070.; Kk.loc[6, 'Z'] = pl_[6] - 5000.                           # research anchors outside their lifecycles
    ref = dict(rest_floor_nonoverlap=int(miss(Kk.S.values * 1000., Kk.S.values * 1000. + 999.999, Kk).sum()), matched_floor_nonoverlap=0)
    Rk, rk = known_checks(Kk, ref, draws=50)
    if not (all(v == 'YES' for v in rk.values()) and Rk['K1']['MATCH_FIRST'] == 1 and Rk['K1']['AMBIGUOUS'] == 6 and Rk['K1']['CANCEL_FIRST'] == 0 and Rk['K2']['point_contradictions'] == 1
            and Rk['K4']['S1']['missed'] == 2 and Rk['K4']['S1']['cancel_first'] == 1 and not Rk['K4']['S1']['constant_offset']['a']['exists']):
        bad.append(f'known checks on planted cases: {rk} {Rk}')
    Kc = Kk.copy(); Kc.loc[5, 'S'] = np.floor(Kc.place[5] / 1000.) - 5.                       # a REST second ending before the order existed
    if known_checks(Kc, dict(ref, rest_floor_nonoverlap=ref['rest_floor_nonoverlap'] + 1), draws=50)[1]['K2 DELAYED-STATUS-REFUSED'] != 'NO':
        bad.append('an evidence bound that contradicts its lifecycle passes K2')
    Kb = Kk.copy(); Kb.loc[0, 'M'] += 5.                                                       # a wrong identity: Matched second 5 s off its order
    if known_checks(Kb, dict(ref, matched_floor_nonoverlap=1), draws=50)[1]['K3 IDENTITY-CHECK-FIRES'] != 'NO':
        bad.append('a wrong identity passes K3')
    if rehearse and not bad:
        O, Fl, Ls = _synthetic(rng); day = '20990101'; Ls_rows = Ls[RAW_COLS + ['tk', 'tk_status']].reset_index(drop=True)
        env = dict(h=600., q01=-900., q99=1050., inputs={})
        E = O[O.status == 'eligible'].reset_index(drop=True); cell = E.groupby(['day', 'm', 'side', 'phase'], sort=True).ngroup().values
        T_ = KP.score_steps(E[[f'f{j:02d}' for j in range(KP.NSTEP + 1)]].values, E.side.values, E.p.values)[KP.DECLARED]
        endj_ = A.endj_of(E.t_full.values, E.g_struct.values); act_, CW_ = V1.null_weights(T_, cell, endj_); ri_, t_, val_, src_ = A.fills_of(E, Fl, SHIFT)
        pf_ = A.per_fill(T_, cell, endj_, (ri_, t_, val_, src_)); lend_ = np.where(np.isnan(E.land_struct.values[ri_]), np.inf, E.land_struct.values[ri_])
        Fd_ = pd.DataFrame(dict(fid=np.arange(len(ri_)), m=E.m.values[ri_], side=E.side.values[ri_], a=t_ + SHIFT + E.t.values[ri_], p=E.p.values[ri_]))
        H = V1.link(Fd_, Ls[V1_COLS], 600., -900., 1050.).reindex(np.arange(len(ri_))); r_ = La[np.clip(T_[ri_], 1, KP.NSTEP) - 1]
        F1_ = dict(val=val_, pulled=act_[ri_], r=r_, s=cell[ri_], endj=endj_[ri_], lend=lend_); v1 = {}
        for k, (a_, b_) in (('S1', ('lo1', 'hi1')), ('S2', ('lo2', 'hi2'))):
            lo_, hi_ = H[a_].values - E.t.values[ri_], H[b_].values - E.t.values[ri_]; mn_, mx_ = V1.bounds(lo_, hi_, F1_, CW_); o_ = np.where(act_[ri_], V1.order_of(lo_, hi_, r_), -1)
            v1[k] = dict(E_min=float(mn_.sum()), E_max=float(mx_.sum()), order={nm: int(((o_ == c_) & act_[ri_]).sum()) for c_, nm in enumerate(('AFTER', 'BEFORE', 'AMBIGUOUS'))})
        with tempfile.TemporaryDirectory(dir=str(I.K.IO)) as tmp:
            try:
                R, read = run(out=Path(tmp) / 'x', src=_Src(Ls[LEG_COLS]), tables=(O, Fl), rows_of=lambda d_: Ls_rows, env_of=lambda d_: env, v1=v1, days=[day],
                              expect=(float(pf_['score'].sum() - pf_['null'].sum()), float(pf_['score'].sum())), tol=1e-6)
                outs = sorted(p.name for p in (Path(tmp) / 'x').iterdir()); FLx = pd.read_parquet(Path(tmp) / 'x' / 'ledger_fills.parquet')
                OLx = pd.read_parquet(Path(tmp) / 'x' / 'ledger_orders.parquet')
                if not (read['V0 CONTROLS'] == 'YES' and len(FLx) == len(ri_) and len(OLx) == len(O) and R['link']['unlinked'] == 0 and R['link']['multi_event'] > 0
                        and 'ledger_legs.parquet' in outs and 'exec_windows.json' in outs and (FLx.link != 'unlinked').all()):
                    bad.append(f'rehearsal outputs: {outs} {R["link"]}')
                Xs = R['xtab']['S1|CP']; npl = int(act_[ri_].sum())
                if not (Xs['REMOVED']['_total']['fills'] + Xs['KEPT']['_total']['fills'] == npl < len(ri_)
                        and abs(R['sim_removed'] - float(-val_[act_[ri_] & (t_ >= r_)].sum())) < 1e-9):
                    bad.append(f'rehearsal cross-tab population: {npl} pulled-order fills of {len(ri_)}; removed {R["sim_removed"]}')
                g_ = FLx[FLx.pulled].groupby('oid').cls_S1_CP.apply(lambda x: bool(np.isin(x.values, (0, 1)).all()))
                P_ = OLx[OLx.decision_S1.isin(['evaluable', 'not_evaluable', 'no_m50_fill'])]
                want_ = np.where(P_.m50_fills.values == 0, 'no_m50_fill', np.where(g_.reindex(P_.oid.values).fillna(False).values.astype(bool), 'evaluable', 'not_evaluable'))
                if not ((P_.decision_S1.values == want_).all() and R['decisions']['S1']['evaluable'] == int((OLx.decision_S1 == 'evaluable').sum()) > 0):
                    bad.append('rehearsal decisions differ from the ledger fills\' classes')
            except (SystemExit, ValueError) as e:
                bad.append(f'rehearsal refused: {e}')
            rb1 = Ls_rows.copy(); rb1.loc[0, 'shares'] += 1.; rb2 = Ls_rows.copy(); rb2.loc[0, 'tk'] = rb2.tk[0] + 1000 if rb2.tk[0] >= 0 else 7
            v1b = {k: dict(v, E_min=v['E_min'] - 1.) for k, v in v1.items()}; e_ = float(pf_['score'].sum() - pf_['null'].sum()); s_ = float(pf_['score'].sum())
            for nm_, kw in (('a wrong collapsed expectation', dict(expect=(e_ + 1., s_))), ('a wrong v1 reference', dict(v1=v1b)),
                            ('a leg field that differs from its legs_timed2 row', dict(rows_of=lambda d_: rb1)), ('a leg whose event id breaks the taker-order rule', dict(rows_of=lambda d_: rb2))):
                args = dict(src=_Src(Ls[LEG_COLS]), tables=(O, Fl), rows_of=lambda d_: Ls_rows, env_of=lambda d_: env, v1=v1, days=[day], expect=(e_, s_), tol=1e-6); args.update(kw)
                o_ = Path(tmp) / ('y' + str(abs(hash(nm_)) % 1000))
                try:
                    run(out=o_, **args); bad.append(f'run accepted {nm_}')
                except SystemExit:
                    pass
                if o_.exists():
                    bad.append(f'a refused run wrote output ({nm_})')
    if bad:
        sys.exit('EXEC-WINDOW SELFTEST FAILED: ' + ' | '.join(bad))
    print('exec-window selftests passed (synthetic only): classes incl. TIE / UNKNOWN / CONTRADICTORY; status timestamps refused as points; event intersection, '
          'evidence cap and statuses; link candidates; the event sweep = a dense grid (3 latencies) and v1 on singletons; collapsed = the clock audit; cross-tab '
          'conservation (and its falsifier); K0-K4 on planted cases (and a wrong identity refused); run() rehearsed end to end on a synthetic source', flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif a == ['run']:
        run()
    elif a == ['known']:
        known()
    else:
        sys.exit('usage: pf_keep_pull_exec_windows.py selftest | run | known')
