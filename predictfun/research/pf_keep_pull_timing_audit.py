"""SOURCE-AWARE FILL-VERSUS-CANCEL ORDERING AUDIT OF THE KEEP / PULL SELECTION (2026-11-18, session aea5bfb9). USER: "Check new review and
proceed next priority"; the review of 1a1abbaa (reviews/automatic/20261118_1a1abbaa/report.md, its one next priority): "Build a source-aware
audit of fill-versus-cancel ordering for the same neutral decision population. Link each relevant trade/fill to its original identity and
distinguish observed source/receipt times from inferred match times and simulated cancel landing. Report whether timing establishes that
the fill is before cancellation, after cancellation, or ambiguous. Carry ambiguous fills into the full value accounting with explicitly
justified uncertainty; report scenario ranges where genuine bounds cannot be established. Do not silently drop near fills, assume M250 is
ground truth, or fit an offset to improve P&L." Declared before computing. Consumed Oct23-27; nothing fitted; no offset chosen.

POPULATION: the committed keep / pull run (data/predict_fun/research/keep_pull_20261117, hashes checked): its 143,134 eligible resting JOIN
 orders, their M50 replay fills, the declared score A+0 and its exact STATES-null weights at M50 eligibility (pf_keep_pull_clock_audit).
LINK: every M50 fill is joined to the loader's legs (pf_stop_ev_minutes_inputs.load_day, the replay's own source) of its market and side
 whose anchor equals the fill's (anchor = fill time + 50 ms + posting time) and whose price can fill the order (at or through it). A fill
 without such a leg is a status (unlinked), counted - none is expected.
WHAT EACH SOURCE OBSERVES (venue clock, ms) and the match-time interval each scenario allows (no capture records the match instant):
 WS-timed legs (how 1 leg / 2 cell match): t_ms = the venue's settlementsPending source time matched to the leg; the match precedes it by
   an unmeasured lag.               S1 [t_ms - 250, t_ms] (the program's M0..M250 clock range, C2)   S2 [t_ms - h, t_ms + h]
 E2 legs: t_e2 = the book-drop time   S1 [t_e2 - 250, t_e2 + 250] (the program's +-250 ms assumption, C12)  S2 [t_e2 - h, t_e2 + h]
   estimated from book frames (calibrated on WS-timed cells).
 imputed legs: only the REST executedAt second S (rounded).   S1 [S - 500, S + 500]   S2 [S + q01, S + q99]
 h = the day's measured p90 |E2 - WS| (booktime.json, ~550-600 ms); q01 / q99 = the day's 1st / 99th percentile of WS time - executedAt
 (timing.json, ~-900 / +1,050 ms). S1 / S2 are scenario ranges, not established bounds. A fill's interval is the union over its legs.
ORDERING: against the cancel landing r of the score's pull of that order (decision + 37 ms, the replay's venue axis), a fill is AFTER (the
 pull avoids it) if its whole interval lies at or after r, BEFORE if wholly before r, else AMBIGUOUS.
VALUE: each fill's contribution to E = score improvement - exact STATES-null expectation is a step function of its match time tau (the
 score's landing and the null's weighted landings; zero before the order is live, at or after END or its structural cancel landing).
 E_min / E_max = the sum over fills of the minimum / maximum over tau in the fill's interval - exact bounds when each fill's tau is free in
 its interval and its quantity is held at the M50 replay's (a stated limit: the queue would change with tau; the M0 / M250 kernel runs are
 the scenario range with quantities re-simulated).
READINGS:
 T0 CONTROLS: every M50 fill linked; with each interval collapsed to the replay's own M50 time, E = +234,279 c and the score improvement
    1,169,671 c (the committed numbers, to 1 c). Else no output.
 T1 SIGN UNDER S1: EXCESS-POSITIVE-IDENTIFIED if E_min > 0, EXCESS-NEGATIVE-IDENTIFIED if E_max < 0, else SIGN-NOT-IDENTIFIED.
 T2 SIGN UNDER S2: the same.
 Printed: the score's removed value split AFTER / BEFORE / AMBIGUOUS by scenario, source, side and day; E_min / E_max by source, side,
 day; the evidence each source lacks.
SELFTEST (synthetic): classification and bounds of planted fills; the breakpoint bounds equal a dense tau grid; collapsed intervals equal
 pf_keep_pull_clock_audit.per_fill to the cent; a fill without a leg is a status.
LIMITS: consumed days; one declared rule; STATES is local selection; fill quantities held at the M50 replay; the scenario widths are the
 program's declared ranges (S1) and measured disagreement envelopes (S2), not calibrated bounds.
Usage (run_light, PF_SLOT=claude): pf_keep_pull_timing_audit.py selftest | run
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_keep_pull as KP                                 # noqa: E402
import pf_keep_pull_clock_audit as A                      # noqa: E402

C, pd, I = KP.C, KP.pd, KP.I
OUT = Path(I.K.IO) / 'keep_pull_timing_audit_20261118'
SHIFT = 50.                                               # the primary replay's trade shift (M50)
S1_WS, S1_E2, S1_IMP = 250., 250., 500.
CLASS = ('WS', 'E2', 'imputed', 'mixed')


def null_weights(T, cell, endj):
    """The score's executed pulls and the STATES null's cumulative weights CW[s, j] = sum over steps <= j of n / live (as the clock audit)."""
    n = len(T); S = int(cell.max()) + 1; lo = KP.J0; act = (T >= lo) & (T <= KP.NSTEP) & (T < endj)
    cnt = np.zeros((S, KP.NSTEP + 2)); np.add.at(cnt, (cell[act], T[act]), 1.)
    live = np.zeros((S, KP.NSTEP + 3)); np.add.at(live, (cell, np.full(n, lo)), 1.); np.add.at(live, (cell, np.clip(endj, lo, KP.NSTEP + 1)), -1.)
    live = np.cumsum(live, 1)[:, :KP.NSTEP + 2]; C.require(bool((cnt <= live + 1e-9).all()), 'an executed pull outside its risk set')
    return act, np.cumsum(np.where(cnt > 0, cnt / np.maximum(live, 1.), 0.), 1)


def contrib(tau, F, CW):
    """A fill's contribution to E = score improvement - exact null expectation if its match time were tau (ms after posting). F: per-fill
    arrays val, pulled, r (the score's landing), s (stratum), endj, lend (the structural cancel landing, inf if none); CW: null weights.
    No fill before the order is live, after END (a trade at END itself counts, as in the kernel) or at / after the structural landing."""
    La = KP.landing(np.arange(1, KP.NSTEP + 1)); top = np.minimum(np.searchsorted(La, tau, side='right'), F['endj'] - 1)
    W = np.where(top >= KP.J0, CW[F['s'], np.maximum(top, 0)] - CW[F['s'], KP.J0 - 1], 0.)
    c = -F['val'] * (np.where(F['pulled'] & (tau >= F['r']), 1., 0.) - W)
    return np.where((tau < KP.PLACE) | (tau > KP.H) | (tau >= F['lend']), 0., c)


def bounds(lo, hi, F, CW):
    """Per fill: min and max over tau in [lo, hi] of contrib - a right-continuous step function, so evaluated at lo and at every breakpoint
    in (lo, hi]: the landings of steps 1..98 (the score's and the null's), the order going live, its structural landing, END and just after it."""
    La = KP.landing(np.arange(1, KP.NSTEP + 1)); n = len(lo)
    a = np.searchsorted(La, lo, side='right'); k = np.searchsorted(La, hi, side='right') - a
    pos = np.repeat(a, k) + (np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k))
    idx = np.r_[np.arange(n), np.repeat(np.arange(n), k), np.tile(np.arange(n), 4)]
    tau = np.r_[lo, La[pos], np.clip(np.full(n, KP.PLACE), lo, hi), np.clip(F['lend'], lo, hi), np.clip(np.full(n, KP.H), lo, hi),
                np.clip(np.full(n, np.nextafter(KP.H, np.inf)), lo, hi)]
    c = contrib(tau, {k_: v[idx] for k_, v in F.items()}, CW)
    mn = np.full(n, np.inf); mx = np.full(n, -np.inf); np.minimum.at(mn, idx, c); np.maximum.at(mx, idx, c)
    return mn, mx


def order_of(lo, hi, r):
    """AFTER (0) if the whole interval is at or after the landing r, BEFORE (1) if wholly before it, else AMBIGUOUS (2)."""
    return np.where(lo >= r, 0, np.where(hi < r, 1, 2))


def leg_intervals(L, h, q01, q99):
    """Per leg: evidence class (0 WS, 1 E2, 2 imputed) and the S1 / S2 match-time intervals (venue ms)."""
    ws = (L.how.values > 0) & np.isfinite(L.t_ms.values); e2 = ~ws & np.isfinite(L.t_e2.values); S = L.exec_s.values * 1000.
    cls = np.where(ws, 0, np.where(e2, 1, 2)); x = np.where(ws, L.t_ms.values, np.where(e2, L.t_e2.values, S))
    lo1 = np.where(ws, x - S1_WS, np.where(e2, x - S1_E2, x - S1_IMP)); hi1 = np.where(ws, x, np.where(e2, x + S1_E2, x + S1_IMP))
    lo2 = np.where(cls < 2, x - h, x + q01); hi2 = np.where(cls < 2, x + h, x + q99)
    return cls, lo1, hi1, lo2, hi2


def link(Fd, LT, h, q01, q99):
    """Join fills to the loader's legs. Fd: one row per fill with fid, m, side, a (anchor, venue ms), p. LT: the loader's legs (mid, side,
    anc, ptick, how, t_ms, t_e2, exec_s). Returns per fid: legs, class (0 WS / 1 E2 / 2 imputed / 3 mixed), S1 / S2 union intervals."""
    U = LT[['mid', 'side', 'anc']].drop_duplicates().sort_values('anc'); U = U.assign(mid=U.mid.astype(np.int64), side=U.side.astype(np.int64))
    x = Fd.assign(m=Fd.m.astype(np.int64), side=Fd.side.astype(np.int64)).sort_values('a')
    x = pd.merge_asof(x, U.rename(columns={'mid': 'm'}), left_on='a', right_on='anc', by=['m', 'side'], direction='nearest', tolerance=0.01)
    y = x.dropna(subset=['anc']).merge(LT.assign(m=LT.mid.astype(np.int64), side=LT.side.astype(np.int64))[['m', 'side', 'anc', 'ptick', 'how', 't_ms', 't_e2', 'exec_s']],
                                       on=['m', 'side', 'anc'])
    y = y[np.where(y.side.values == 0, y.ptick.values <= y.p.values, y.ptick.values >= y.p.values)]           # legs that can fill the order
    cls, lo1, hi1, lo2, hi2 = leg_intervals(y, h, q01, q99)
    g = pd.DataFrame(dict(fid=y.fid.values, cls=cls, lo1=lo1, hi1=hi1, lo2=lo2, hi2=hi2)).groupby('fid')
    out = g.agg(legs=('cls', 'size'), cmin=('cls', 'min'), cmax=('cls', 'max'), lo1=('lo1', 'min'), hi1=('hi1', 'max'), lo2=('lo2', 'min'), hi2=('hi2', 'max'))
    out['cls'] = np.where(out.cmin == out.cmax, out.cmin, 3)
    return out.drop(columns=['cmin', 'cmax'])


def link_day(day, Fd, src=None):
    """All fills of one day, linked market batch by market batch through the replay's own loader."""
    src = src or KP.Tape(); b = json.load(open(f'{I.IO}/step1/{day}/booktime.json')); t = json.load(open(f'{I.IO}/step1/{day}/timing.json'))
    h, q01, q99 = float(b['E2']['p90']), float(t['offset_ms_pct']['p01']), float(t['offset_ms_pct']['p99']); offset = src.offset(day); outs = []
    markets = sorted(set(Fd.m.astype(int)))
    for k in range(0, len(markets), 24):
        batch = markets[k:k + 24]; data = src.load(day, set(batch), offset)
        LT = data['LT'][['mid', 'side', 'anc', 'ptick', 'how', 't_ms', 't_e2', 'exec_s']]
        outs.append(link(Fd[Fd.m.isin(batch)], LT, h, q01, q99)); del data; C.guard('timing-audit link')
    return pd.concat(outs) if outs else pd.DataFrame(), dict(h=h, q01=q01, q99=q99)


def selftest():
    bad = []; rng = np.random.default_rng(KP.SEED); La = KP.landing(np.arange(1, KP.NSTEP + 1))
    if order_of(np.array([140., 50., 100.]), np.array([300., 130., 200.]), np.full(3, 137.)).tolist() != [0, 1, 2]:
        bad.append('ordering classes AFTER / BEFORE / AMBIGUOUS')
    n = 240; cell = np.repeat(np.arange(20), 12); endj = rng.integers(KP.J0, KP.NSTEP + 2, n); T = np.where(rng.random(n) < .5, rng.integers(KP.J0, 40, n), KP.NSTEP + 1)
    m = 900; ri = rng.integers(0, n, m); t = np.r_[rng.uniform(60., 4990., m - 40), np.full(40, KP.H)]; val = rng.normal(0., 10., m)   # 40 fills at END itself
    lend = np.where(rng.random(n) < .2, rng.uniform(500., 4000., n), np.inf); keep = t < lend[ri]; ri, t, val = ri[keep], t[keep], val[keep]   # never after the landing
    act, CW = null_weights(T, cell, endj); F = dict(val=val, pulled=act[ri], r=La[np.clip(T[ri], 1, KP.NSTEP) - 1], s=cell[ri], endj=endj[ri], lend=lend[ri])
    pf = A.per_fill(T, cell, endj, (ri, t, val, np.zeros(len(ri), int))); mn, mx = bounds(t, t, F, CW)
    if not (np.allclose(mn, mx) and abs(mn.sum() - (pf['score'].sum() - pf['null'].sum())) < 1e-6):
        bad.append(f"collapsed intervals {mn.sum():.6f} != the clock audit's per-fill excess {pf['score'].sum() - pf['null'].sum():.6f}")
    lo = t - rng.uniform(0., 600., len(t)); hi = t + rng.uniform(0., 600., len(t)); mn, mx = bounds(lo, hi, F, CW); gmn = np.full(len(t), np.inf); gmx = -gmn
    for x in np.linspace(0., 1., 1201):                                                       # dense grid (0.5 ms or finer) plus both ends
        c = contrib(lo + x * (hi - lo), F, CW); gmn = np.minimum(gmn, c); gmx = np.maximum(gmx, c)
    if not (np.all(mn <= gmn + 1e-9) and np.all(mx >= gmx - 1e-9) and np.allclose(mn, gmn, atol=1e-9) and np.allclose(mx, gmx, atol=1e-9)):
        bad.append(f'breakpoint bounds differ from a dense grid: {int((~np.isclose(mn, gmn)).sum())} minima, {int((~np.isclose(mx, gmx)).sum())} maxima')
    LT = pd.DataFrame(dict(mid=[7, 7, 7, 7, 8], side=[0, 0, 0, 0, 1], anc=[1000., 1000., 1000., 2000., 3000.], ptick=[50, 49, 51, 50, 60], how=[1, 0, 1, 0, 0],
                           t_ms=[990., np.nan, 995., np.nan, np.nan], t_e2=[np.nan, 1010., np.nan, np.nan, np.nan], exec_s=[1., 1., 1., 2., 3.]))
    Fd = pd.DataFrame(dict(fid=[0, 1, 2], m=[7, 7, 8], side=[0, 0, 1], a=[1000.004, 2000., 2500.], p=[50, 50, 60]))
    lk = link(Fd, LT, 600., -900., 1050.)
    want = {0: (2, 3, 740., 1260.), 1: (1, 2, 1500., 2500.)}                     # fid 0: WS leg at 50 [740, 990] + E2 leg at 49 [760, 1260]; 51 cannot fill a bid at 50
    got = {int(i): (int(r.legs), int(r.cls), float(r.lo1), float(r.hi1)) for i, r in lk.iterrows()}
    if got != want:
        bad.append(f'link: {got} != {want} (fid 2 has no leg at its anchor: a status)')
    cls, lo1, hi1, lo2, hi2 = leg_intervals(LT.iloc[[3]], 600., -900., 1050.)
    if (cls.tolist(), lo1.tolist(), hi1.tolist(), lo2.tolist(), hi2.tolist()) != ([2], [1500.], [2500.], [1100.], [3050.]):
        bad.append('imputed leg intervals')
    if bad:
        sys.exit('KEEP-PULL TIMING AUDIT SELFTEST FAILED -> no output:\n  ' + '\n  '.join(bad))
    print('keep-pull timing audit selftests passed (synthetic only): ordering classes; collapsed intervals = the clock audit per fill; '
          'breakpoint bounds = a dense grid; link (anchor, can-fill price, mixed class, unlinked fill) and leg intervals as planted', flush=True)


def run():
    """The audit on the committed keep / pull run (orders, M50 fills) and the loader's legs -> timing_audit.txt / timing_audit.json."""
    start = time.time(); C.require(not OUT.exists(), 'audit output exists'); selftest()
    O, Fl = KP.load(KP.OUT); E = O[O.status == 'eligible'].reset_index(drop=True); days = list(KP.DAYS); La = KP.landing(np.arange(1, KP.NSTEP + 1))
    cell = E.groupby(['day', 'm', 'side', 'phase'], sort=True).ngroup().values
    T = KP.score_steps(E[[f'f{j:02d}' for j in range(KP.NSTEP + 1)]].values, E.side.values, E.p.values)[KP.DECLARED]
    endj = A.endj_of(E.t_full.values, E.g_struct.values); act, CW = null_weights(T, cell, endj); ri, t, val, srcc = A.fills_of(E, Fl, SHIFT); n = len(ri)
    F = dict(val=val, pulled=act[ri], r=La[np.clip(T[ri], 1, KP.NSTEP) - 1], s=cell[ri], endj=endj[ri], lend=np.where(np.isnan(E.land_struct.values[ri]), np.inf, E.land_struct.values[ri]))
    mn0, mx0 = bounds(t, t, F, CW); pf = A.per_fill(T, cell, endj, (ri, t, val, srcc)); e0 = float(mn0.sum()); s0 = float(pf['score'].sum())
    if not (abs(e0 - 234279.) <= 1. and abs(float(mx0.sum()) - e0) < 1e-6 and abs(e0 - float(pf['score'].sum() - pf['null'].sum())) < 1e-6 and abs(s0 - 1169671.) <= 1.):
        sys.exit(f'KEEP-PULL TIMING AUDIT FAILED -> no output: collapsed intervals give E {e0:,.1f} / score {s0:,.1f}, not +234,279 / 1,169,671')
    Fd = pd.DataFrame(dict(fid=np.arange(n), day=E.day.values[ri], m=E.m.values[ri], side=E.side.values[ri], a=t + SHIFT + E.t.values[ri], p=E.p.values[ri]))
    LK, env = [], {}
    for d in days:
        lk, env[d] = link_day(d, Fd[Fd.day == d]); LK.append(lk)
        print(json.dumps(dict(stage='link', day=d, fills=int((Fd.day == d).sum()), linked=int(len(lk)), seconds=round(time.time() - start), rss=round(I.K.rss_mib()))), flush=True)
    LK = pd.concat(LK); C.require(LK.index.is_unique, 'a fill linked twice'); unl = n - len(LK)
    if unl:
        sys.exit(f'KEEP-PULL TIMING AUDIT FAILED -> no output: {unl:,} of {n:,} M50 fills have no leg at their anchor (status unlinked)')
    LK = LK.reindex(np.arange(n)); t0 = E.t.values[ri]; cls = LK.cls.values.astype(int)
    S = {'S1': (LK.lo1.values - t0, LK.hi1.values - t0), 'S2': (LK.lo2.values - t0, LK.hi2.values - t0)}
    B = {k: bounds(lo, hi, F, CW) for k, (lo, hi) in S.items()}
    pl = F['pulled']; sim = np.where(t >= F['r'], 0, 1)                                          # the replay's own M50 ordering
    side = np.where(E.side.values[ri] == 0, 'bid', 'ask'); day = E.day.values[ri]
    R = dict(fills=n, legs=int(LK.legs.sum()), class_fills={CLASS[c]: int((cls == c).sum()) for c in range(4)}, collapsed_E=e0, score=s0, env=env)
    for k, (lo, hi) in S.items():
        o = np.where(pl, order_of(lo, hi, F['r']), -1); mn, mx = B[k]
        R[k] = dict(E_min=float(mn.sum()), E_max=float(mx.sum()), width_ms_median=float(np.median(hi - lo)),
                    order={nm: dict(fills=int(((o == c) & pl).sum()), removed_c=float(-val[(o == c) & pl].sum()), gross_c=float(np.abs(val[(o == c) & pl]).sum())) for c, nm in enumerate(('AFTER', 'BEFORE', 'AMBIGUOUS'))},
                    by_class={CLASS[c]: dict(E_min=float(mn[cls == c].sum()), E_max=float(mx[cls == c].sum()), ambiguous_c=float(-val[(o == 2) & pl & (cls == c)].sum()),
                                             identified_gross=float(np.abs(val[(o >= 0) & (o < 2) & pl & (cls == c)]).sum()), gross=float(np.abs(val[pl & (cls == c)]).sum())) for c in range(4)},
                    by_side={sd: dict(E_min=float(mn[side == sd].sum()), E_max=float(mx[side == sd].sum())) for sd in ('bid', 'ask')},
                    by_day={d: dict(E_min=float(mn[day == d].sum()), E_max=float(mx[day == d].sum()), E_sim=float(mn0[day == d].sum())) for d in days})
    R['sim_order'] = dict(removed_c=float(-val[(sim == 0) & pl].sum()), kept_c=float(-val[(sim == 1) & pl].sum()))
    sign = lambda k: 'EXCESS-POSITIVE-IDENTIFIED' if R[k]['E_min'] > 0 else 'EXCESS-NEGATIVE-IDENTIFIED' if R[k]['E_max'] < 0 else 'SIGN-NOT-IDENTIFIED'
    read = {'T0 CONTROLS': 'YES', 'T1 SIGN UNDER S1': sign('S1'), 'T2 SIGN UNDER S2': sign('S2')}
    L = ['SOURCE-AWARE FILL-VERSUS-CANCEL ORDERING AUDIT of the declared score A+0 (pf_keep_pull_timing_audit.py; the committed keep / pull run, M50 fills, consumed Oct23-27; c summed)',
         f"  {len(E):,} eligible orders; {n:,} M50 fills, every one linked ({R['legs']:,} legs at the fill's anchor that can fill the order); fills by evidence: "
         + ', '.join(f'{k} {v:,}' for k, v in R['class_fills'].items()),
         f"  T0 control: collapsed to the replay's own M50 times, E = {e0:+,.0f} and the score improvement {s0:,.0f} (the committed numbers)", '',
         'OBSERVED vs INFERRED (venue clock): WS legs observe the settlementsPending source time (match precedes it by an unmeasured lag); E2 legs a book-drop '
         'estimate; imputed legs only the REST executedAt second. Simulated cancel landing = decision + 37 ms on the replay axis.',
         '  S1 (the program\'s declared ranges): WS [t - 250, t], E2 [t - 250, t + 250], imputed [S - 500, S + 500].  S2 (measured envelopes) by day: '
         + ' | '.join(f"{d} h {env[d]['h']:.0f}, q01 {env[d]['q01']:+.0f}, q99 {env[d]['q99']:+.0f}" for d in days), '',
         'PART 1 - the score\'s pulled orders: their fills against the pull\'s landing (removed c = value avoided if AFTER)',
         f"  replay M50 times: removed {R['sim_order']['removed_c']:+,.0f} c by the pulls (fills after the landing), kept {R['sim_order']['kept_c']:+,.0f} c (before)"]
    for k in ('S1', 'S2'):
        o = R[k]['order']; L.append(f"  {k} (median interval {R[k]['width_ms_median']:.0f} ms): " + ' | '.join(f"{nm} {v['fills']:,} fills, {v['removed_c']:+,.0f} c (gross {v['gross_c']:,.0f})" for nm, v in o.items()))
    L += ['', 'PART 2 - bounds on E = score improvement - exact STATES-null expectation (each fill\'s match time free in its interval; M50 quantities held)',
          f"  replay M50 (collapsed): {e0:+,.0f}"]
    for k in ('S1', 'S2'):
        L.append(f"  {k}: [{R[k]['E_min']:+,.0f}, {R[k]['E_max']:+,.0f}] | by side " + ', '.join(f"{sd} [{v['E_min']:+,.0f}, {v['E_max']:+,.0f}]" for sd, v in R[k]['by_side'].items())
                 + ' | by evidence ' + ', '.join(f"{c} [{v['E_min']:+,.0f}, {v['E_max']:+,.0f}]" for c, v in R[k]['by_class'].items()))
        L.append('       by day ' + ', '.join(f"{d} [{v['E_min']:+,.0f}, {v['E_max']:+,.0f}] (replay {v['E_sim']:+,.0f})" for d, v in R[k]['by_day'].items()))
    L += ['', 'READINGS (declared before computing)'] + [f'  {k}: {v}' for k, v in read.items()] + [
          '', 'MISSING EVIDENCE: no capture records the venue match instant; the WS settlementsPending lag after the match is unmeasured; about 10% of WS-timed legs '
          'lie outside +-500 ms of their executedAt second (timing.json), so even the second does not bound every leg; E2 |error| against WS is ~550-600 ms at p90 '
          '(booktime.json). Order-level fill timing from live (C10) is the evidence that would identify the ordering.',
          'LIMITS: consumed days; one declared rule; STATES is local selection; quantities held at the M50 replay; S1 / S2 are scenario ranges, not calibrated bounds.']
    OUT.mkdir(parents=True); (OUT / 'timing_audit.txt').write_text('\n'.join(L) + '\n')
    C.dump(OUT / 'timing_audit.json', dict(asof=C.stamp(), readings=read, R=R, source_receipts={d: C.sha(KP.OUT / d / 'receipt.json') for d in days},
                                           provenance=KP.provenance(), audit_source_sha256=C.sha(Path(__file__).resolve()), seconds=time.time() - start))
    print('\n'.join(L), flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif a == ['run']:
        run()
    else:
        sys.exit('usage: pf_keep_pull_timing_audit.py selftest | run')
