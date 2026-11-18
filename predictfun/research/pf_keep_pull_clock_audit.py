"""PAIRED M50 / M250 FILL-CLOCK AND ELIGIBILITY AUDIT OF THE KEEP / PULL SELECTION (2026-11-18, session aea5bfb9). USER: "Check review,
Proceed next priority"; the review of 717a1a49 (reviews/automatic/20261118_717a1a49/report.md, its one next priority): "Perform a paired
M50/M250 fill-clock and eligibility audit on the same saved orders and decision opportunities. Attribute the lost selection advantage to
changed fill timing/quantity versus changed client eligibility and state-null weights, by side and source, retaining E2, imputed and mixed
fills in the total. ... This gate should establish whether the economic target/control remains stable enough to justify model tuning."
Saved tables of the committed run only (data/predict_fun/research/keep_pull_20261117, hashes checked against its receipts); no tape, no fit.
Declared before computing.

QUANTITY: the declared score A+0's excess over the exact expectation of its STATES null, E(F, G) = score improvement - expected null
 improvement over KEEP, with the fills of clock F (M50 / M250) and the client's eligibility from G (M50 / M250 / COMMON = the decision
 states live at both clocks). G decides which score triggers execute (while the client knows the order is live), the null's counts per
 (stratum, step) and its risk sets; F decides the fills and their values. The STATES null includes each live order of a (stratum, step)
 with probability n / live, so its expectation is exact by linearity - no draws. Every term is a sum over fills, so the splits by side,
 fill source and day are exact. Common END marks, M = trade shift, 37 ms cancel latency, as the committed run.
DECOMPOSITION of E(250, 250) - E(50, 50) (Shapley, sums exactly): FILL = 1/2 [E(250, 50) - E(50, 50) + E(250, 250) - E(50, 250)],
 ELIGIBILITY = 1/2 [E(50, 250) - E(50, 50) + E(250, 250) - E(250, 50)]; both one-sided paths printed; also by side and by fill source.
MECHANISM (printed): the part of each excess carried by fills in the first 200 ms after the pull lands (M250 places a trade 200 ms
 earlier than M50 does).
READINGS:
 A0 REPRODUCED (control): E(50, 50) = +234,279 c, E(250, 250) = -27,124 c and the exact null expectations 935,392 / 1,051,920 c (the
    review's numbers, to 1 c); the score improvements equal the committed report's 1,169,671 / 1,024,796 c. Else no output.
 A1 ATTRIBUTION: FILL-TIMING-DOMINATES if |FILL| >= 2 |ELIGIBILITY|; ELIGIBILITY-DOMINATES if |ELIGIBILITY| >= 2 |FILL|; else BOTH-MATERIAL.
 A2 SAME-OPPORTUNITY STABILITY (the review's gate): E(50, COMMON) and E(250, COMMON) both > 0 pooled and on >= 4 of 5 days ->
    TARGET-STABLE; their pooled signs differ -> TARGET-CLOCK-SENSITIVE (model tuning on this target is not justified); else MIXED.
CONTROLS (any failure -> no output): the saved tables' hashes equal the committed receipts (pf_keep_pull.load); the vectorized per-fill
 expectation equals a brute-force enumeration over (stratum, step) on sampled strata; every executed pull lies in its risk set. SELFTEST
 (synthetic): the analytic STATES expectation equals the mean of the committed sampler (pf_keep_pull.state_picks) within 3 standard errors;
 the brute-force check; a fill that crosses the landing only at M250 is attributed to FILL, a full-fill notice that removes a pull only at
 M250 to ELIGIBILITY, both to the cent.
LIMITS: consumed days; one declared rule; STATES is local selection, not a portfolio; fills are the replay's (queue M, trade anchors).
Usage (run_light, PF_SLOT=claude): pf_keep_pull_clock_audit.py selftest | run
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_keep_pull as KP                                 # noqa: E402

C, pd = KP.C, KP.pd
SRC_DIR = KP.OUT                                          # the committed keep / pull run's tables
OUT = Path(KP.I.K.IO) / 'keep_pull_clock_audit_20261118'
NEAR = 200.
REVIEW = dict(e50=234279., e250=-27124., n50=935392., n250=1051920., s50=1169671., s250=1024796.)   # the review's / report's numbers
SRC = KP.SRC


def endj_of(t_full, g_struct):
    """The client's last decision step + 1 (decisions at j < endj), as pf_keep_pull.variant: a full fill is known NOTICE ms after it, a
    structural cancel when the client decides it."""
    lim = np.fmin(g_struct, t_full + KP.NOTICE); d = KP.STEP * np.arange(1, KP.NSTEP + 1)
    return np.where(np.isnan(lim), KP.NSTEP + 1, np.searchsorted(d, lim, side='left') + 1)


def fills_of(E, Fl, c):
    """The fills of clock c for the orders of E -> (order index, time after posting, value at END in c, source code)."""
    ix = pd.Series(np.arange(len(E)), index=pd.MultiIndex.from_arrays([E.day.values, E.oid.values]))
    f = Fl[Fl.clock == c]; ri = ix.reindex(pd.MultiIndex.from_arrays([f.day.values, f.oid.values])).values; ok = ~np.isnan(ri)
    f = f[ok]; ri = ri[ok].astype(int); sg = np.where(E.side.values == 0, 1., -1.)
    return ri, f.t.values.astype(float), f.q.values * sg[ri] * (E.mark.values[ri] - E.p.values[ri]), f.src.values.astype(int)


def per_fill(T, cell, endj, fl, near=NEAR):
    """Per-fill contributions to the score's improvement over KEEP and to the exact expectation of its STATES null (and their parts carried
    by fills within `near` ms after the landing). A pull at step j removes the fills at or after its landing."""
    ri, t, val, src = fl; n = len(T); S = int(cell.max()) + 1; lo = KP.J0; La = KP.landing(np.arange(1, KP.NSTEP + 1))
    act = (T >= lo) & (T <= KP.NSTEP) & (T < endj)
    cnt = np.zeros((S, KP.NSTEP + 2)); np.add.at(cnt, (cell[act], T[act]), 1.)
    live = np.zeros((S, KP.NSTEP + 3)); hi = np.clip(endj, lo, KP.NSTEP + 1)
    np.add.at(live, (cell, np.full(n, lo)), 1.); np.add.at(live, (cell, hi), -1.); live = np.cumsum(live, 1)[:, :KP.NSTEP + 2]
    C.require(bool((cnt <= live + 1e-9).all()), 'an executed pull outside its risk set')
    CW = np.cumsum(np.where(cnt > 0, cnt / np.maximum(live, 1.), 0.), 1)              # CW[s, j] = sum over steps <= j of n / live
    top = np.minimum(np.searchsorted(La, t, side='right'), endj[ri] - 1); s_ = cell[ri]   # steps j <= top land at or before t
    jn = np.maximum(np.searchsorted(La, t - near, side='right'), lo - 1)                 # steps j <= jn land at or before t - near
    W = np.where(top >= lo, CW[s_, np.maximum(top, 0)] - CW[s_, lo - 1], 0.)
    Wn = np.where(top > jn, CW[s_, np.maximum(top, 0)] - CW[s_, jn], 0.)
    tl = La[np.clip(T[ri], 1, KP.NSTEP) - 1]; rem = act[ri] & (t >= tl)
    return dict(score=np.where(rem, -val, 0.), null=-val * W, score_near=np.where(rem & (t - tl < near), -val, 0.), null_near=-val * Wn,
                pulls=int(act.sum()))


def brute(T, cell, endj, fl, strata):
    """Control: the null's expectation on the listed strata by explicit enumeration of (stratum, step) cells and their live orders."""
    ri, t, val, src = fl; La = KP.landing(np.arange(1, KP.NSTEP + 1)); act = (T >= KP.J0) & (T <= KP.NSTEP) & (T < endj); tot = 0.
    for s in strata:
        mem = np.flatnonzero(cell == s)
        for j in np.unique(T[mem][act[mem]]):
            R = mem[endj[mem] > j]; k = int((act[mem] & (T[mem] == j)).sum()); f = np.isin(ri, R) & (t >= La[j - 1])
            tot += k / len(R) * float(-val[f].sum())
    return tot


def sums(pf, keys, fl):
    """Group sums of the per-fill contributions; keys: per-fill arrays."""
    df = pd.DataFrame({**{k: v for k, v in keys.items()}, **{k: pf[k] for k in ('score', 'null', 'score_near', 'null_near')}})
    return df.groupby(list(keys)).sum() if keys else df.sum()


def excess_table(T, cell, EJ, FL, day_of, side_of):
    """E(F, G) and its parts for F in (M50, M250) x G in (M50, M250, COMMON): pooled, by day, by side, by fill source."""
    R = {}
    for F in (50., 250.):
        ri, t, val, src = FL[F]
        for G in (50., 250., 'C'):
            pf = per_fill(T, cell, EJ[G], FL[F]); df = pd.DataFrame(dict(day=day_of[ri], side=side_of[ri], src=src, score=pf['score'], null=pf['null'],
                                                                       score_near=pf['score_near'], null_near=pf['null_near']))
            g = lambda k: {str(a): dict(score=float(b.score), null=float(b.null), excess=float(b.score - b.null), near=float(b.score_near - b.null_near))
                           for a, b in df.groupby(k)[['score', 'null', 'score_near', 'null_near']].sum().iterrows()}
            s = df[['score', 'null', 'score_near', 'null_near']].sum()
            R[(F, G)] = dict(pulls=pf['pulls'], score=float(s.score), null=float(s.null), excess=float(s.score - s.null), near=float(s.score_near - s.null_near),
                             day=g('day'), side=g('side'), src=g('src'))
    return R


def shapley(R, pick=lambda x: x['excess']):
    """FILL and ELIGIBILITY parts of E(250, 250) - E(50, 50), and the two one-sided paths."""
    e = {k: pick(v) for k, v in R.items() if k[1] in (50., 250.)}
    fill = .5 * ((e[(250., 50.)] - e[(50., 50.)]) + (e[(250., 250.)] - e[(50., 250.)])); elig = .5 * ((e[(50., 250.)] - e[(50., 50.)]) + (e[(250., 250.)] - e[(250., 50.)]))
    return dict(total=e[(250., 250.)] - e[(50., 50.)], fill=fill, eligibility=elig, fill_first=e[(250., 50.)] - e[(50., 50.)], eligibility_first=e[(50., 250.)] - e[(50., 50.)])


def a1(sh):
    f, e = abs(sh['fill']), abs(sh['eligibility'])
    return 'FILL-TIMING-DOMINATES' if f >= 2 * e else 'ELIGIBILITY-DOMINATES' if e >= 2 * f else 'BOTH-MATERIAL'


def a2(R, days):
    x50, x250 = R[(50., 'C')], R[(250., 'C')]
    pos = lambda x: x['excess'] > 0 and sum(x['day'].get(d, dict(excess=0.))['excess'] > 0 for d in days) >= 4
    return 'TARGET-STABLE' if pos(x50) and pos(x250) else 'TARGET-CLOCK-SENSITIVE' if (x50['excess'] > 0) != (x250['excess'] > 0) else 'MIXED'


def _case(fills50, fills250, T, tf50, tf250):
    """One synthetic stratum of bids at 50 (common END mark 49 or 51 set by the fill values): fills [(order, t, value)] per clock."""
    n = len(T); cell = np.zeros(n, int); g = np.full(n, np.nan)
    EJ = {50.: endj_of(np.asarray(tf50, float), g), 250.: endj_of(np.asarray(tf250, float), g)}; EJ['C'] = np.minimum(EJ[50.], EJ[250.])
    FL = {c: (np.array([x[0] for x in fs], int), np.array([x[1] for x in fs], float), np.array([x[2] for x in fs], float), np.zeros(len(fs), int))
          for c, fs in ((50., fills50), (250., fills250))}
    return excess_table(np.asarray(T), cell, EJ, FL, np.zeros(n, int), np.zeros(n, int))


def selftest():
    bad = []; rng = np.random.default_rng(KP.SEED); La = KP.landing(np.arange(1, KP.NSTEP + 1))
    # (1) analytic STATES expectation = the committed sampler's mean; brute-force enumeration = the vectorized expectation
    n = 240; cell = np.repeat(np.arange(20), 12); endj = rng.integers(KP.J0, KP.NSTEP + 2, n); T = np.where(rng.random(n) < .5, rng.integers(KP.J0, 40, n), KP.NSTEP + 1)
    m = 900; fl = (rng.integers(0, n, m), rng.uniform(60., 4990., m), rng.normal(0., 10., m), rng.integers(0, 4, m))
    pf = per_fill(T, cell, endj, fl); an = float(pf['null'].sum())
    Xm = np.zeros((n, KP.NSTEP))
    for i_, t_, v_ in zip(fl[0], fl[1], fl[2]):
        Xm[i_, :] += np.where(t_ >= La, v_, 0.)                        # value of the fills a pull at step j would remove
    draws = np.array([float(-Xm[oi, oj - 1].sum()) for oi, oj in KP.state_picks(endj, T, cell, KP.SEED, 20000)])
    if abs(draws.mean() - an) > 3 * draws.std() / np.sqrt(len(draws)):
        bad.append(f'analytic STATES expectation {an:.3f} vs the sampler mean {draws.mean():.3f} +- {draws.std() / np.sqrt(len(draws)):.3f}')
    if abs(brute(T, cell, endj, fl, range(20)) - an) > 1e-9:
        bad.append('brute-force enumeration differs from the vectorized expectation')
    act = (T >= KP.J0) & (T <= KP.NSTEP) & (T < endj); sc = sum(float(-Xm[i, T[i] - 1]) for i in np.flatnonzero(act))
    if abs(sc - float(pf['score'].sum())) > 1e-9:
        bad.append('score improvement differs from the explicit sum')
    # (2) planted FILL case: order 0's losing fill sits after the landing at M50 (150 ms) and before it at M250 (120 ms); eligibility equal
    R = _case([(0, 150., -10.), (1, 1000., -10.), (2, 1000., 10.), (3, 1000., 10.)], [(0, 120., -10.), (1, 1000., -10.), (2, 1000., 10.), (3, 1000., 10.)],
              [KP.J0, KP.J0, KP.NSTEP + 1, KP.NSTEP + 1], [np.nan] * 4, [np.nan] * 4)
    sh = shapley(R); got = (R[(50., 50.)]['excess'], R[(250., 250.)]['excess'], sh['fill'], sh['eligibility'], R[(50., 50.)]['near'])
    if not np.allclose(got, (20., 15., -5., 0., 5.)):
        bad.append(f'planted fill case: E50 / E250 / FILL / ELIGIBILITY / near {got} != (20, 15, -5, 0, 5)')
    # (3) planted ELIGIBILITY case: equal fills; order 1 is known fully filled (300 ms -> known 400 ms) only at M250, before its trigger (500 ms)
    fs = [(0, 1000., -10.), (1, 1000., -10.), (2, 1000., 10.), (3, 1000., 10.)]
    R = _case(fs, fs, [10, 10, KP.NSTEP + 1, KP.NSTEP + 1], [np.nan] * 4, [np.nan, 300., np.nan, np.nan])
    sh = shapley(R); got = (R[(50., 50.)]['excess'], R[(250., 250.)]['excess'], sh['fill'], sh['eligibility'], R[(50., 'C')]['pulls'])
    if not np.allclose(got, (20., 40. / 3., 0., 40. / 3. - 20., 1)):
        bad.append(f'planted eligibility case: E50 / E250 / FILL / ELIGIBILITY / common pulls {got}')
    if (a1(dict(fill=-5., eligibility=0.)), a1(dict(fill=0., eligibility=-6.)), a1(dict(fill=-3., eligibility=-2.))) != ('FILL-TIMING-DOMINATES', 'ELIGIBILITY-DOMINATES', 'BOTH-MATERIAL'):
        bad.append('A1 labels')
    days = ['a', 'b', 'c', 'd', 'e']; mk = lambda v, dv: dict(excess=v, day={d: dict(excess=x) for d, x in zip(days, dv)})
    if (a2({(50., 'C'): mk(5., [1, 1, 1, 1, -1]), (250., 'C'): mk(3., [1, 1, 1, 1, 1])}, days), a2({(50., 'C'): mk(5., [1] * 5), (250., 'C'): mk(-3., [-1] * 5)}, days),
            a2({(50., 'C'): mk(5., [1, 1, 1, -1, -1]), (250., 'C'): mk(3., [1] * 5)}, days)) != ('TARGET-STABLE', 'TARGET-CLOCK-SENSITIVE', 'MIXED'):
        bad.append('A2 labels')
    if bad:
        sys.exit('KEEP-PULL CLOCK AUDIT SELFTEST FAILED -> no output:\n  ' + '\n  '.join(bad))
    print(f'keep-pull clock audit selftests passed (synthetic only): analytic STATES expectation {an:,.2f} = sampler mean {draws.mean():,.2f} '
          f'+- {draws.std() / np.sqrt(len(draws)):.2f} (20,000 draws); brute force and explicit score sums equal; planted FILL and ELIGIBILITY cases '
          'attributed to the cent; A1 / A2 labels', flush=True)


def run():
    """The audit on the committed run's saved tables -> clock_audit.txt / clock_audit.json."""
    start = time.time(); C.require(not OUT.exists(), 'audit output exists'); selftest()
    O, Fl = KP.load(SRC_DIR); E = O[O.status == 'eligible'].reset_index(drop=True); days = list(KP.DAYS)
    cell = E.groupby(['day', 'm', 'side', 'phase'], sort=True).ngroup().values
    T = KP.score_steps(E[[f'f{j:02d}' for j in range(KP.NSTEP + 1)]].values, E.side.values, E.p.values)[KP.DECLARED]
    EJ = {50.: endj_of(E.t_full.values, E.g_struct.values), 250.: endj_of(E.t_full_m250.values, E.g_struct.values)}; EJ['C'] = np.minimum(EJ[50.], EJ[250.])
    FL = {c: fills_of(E, Fl, c) for c in (50., 250.)}
    R = excess_table(T, cell, EJ, FL, E.day.values, np.where(E.side.values == 0, 'bid', 'ask'))
    rs = np.random.default_rng(KP.SEED).choice(int(cell.max()) + 1, 25, replace=False)                 # control: brute force on 25 strata
    for F, G in ((50., 50.), (250., 'C')):
        pf = per_fill(T, cell, EJ[G], FL[F]); m_ = np.isin(cell[FL[F][0]], rs)
        C.require(abs(brute(T, cell, EJ[G], FL[F], rs) - float(pf['null'][m_].sum())) <= 1e-6, f'brute-force expectation differs at {F} / {G}')
    got = dict(e50=R[(50., 50.)]['excess'], e250=R[(250., 250.)]['excess'], n50=R[(50., 50.)]['null'], n250=R[(250., 250.)]['null'],
               s50=R[(50., 50.)]['score'], s250=R[(250., 250.)]['score'])
    A0 = all(abs(got[k] - v) <= 1. for k, v in REVIEW.items())
    if not A0:
        sys.exit(f'KEEP-PULL CLOCK AUDIT FAILED -> no output: the review / report numbers are not reproduced {got} vs {REVIEW}')
    sh = shapley(R); SH = dict(pooled=sh, side={k: shapley(R, lambda x, k=k: x['side'][k]['excess']) for k in ('bid', 'ask')},
                               src={nm: shapley(R, lambda x, c=str(i): x['src'].get(c, dict(excess=0.))['excess']) for i, nm in enumerate(SRC)})
    read = {'A0 REPRODUCED': 'YES', 'A1 ATTRIBUTION': a1(sh), 'A2 SAME-OPPORTUNITY STABILITY': a2(R, days)}
    lab = {50.: 'M50', 250.: 'M250', 'C': 'COMMON'}
    L = ['PAIRED M50 / M250 FILL-CLOCK AND ELIGIBILITY AUDIT of the declared score A+0 (pf_keep_pull_clock_audit.py; saved tables of the committed keep / pull run, '
         'consumed Oct23-27; c summed over orders; STATES = local selection, not a portfolio)',
         f'  {len(E):,} eligible resting orders; A0 control: the review\'s E(50, 50) {REVIEW["e50"]:+,.0f}, E(250, 250) {REVIEW["e250"]:+,.0f}, null {REVIEW["n50"]:,.0f} / '
         f'{REVIEW["n250"]:,.0f} and the report\'s score {REVIEW["s50"]:,.0f} / {REVIEW["s250"]:,.0f} reproduced to 1 c', '',
         'PART 1 - E(F, G) = score improvement - exact STATES-null expectation; F = fill clock, G = eligibility (COMMON = live at both clocks)',
         '  fills  eligibility | pulls  | score improvement | null expectation | excess E   | of which fills < 200 ms after the landing']
    for F in (50., 250.):
        for G in (50., 250., 'C'):
            x = R[(F, G)]; L.append(f"  {lab[F]:<6} {lab[G]:<11} | {x['pulls']:6,d} | {x['score']:17,.0f} | {x['null']:16,.0f} | {x['excess']:+10,.0f} | {x['near']:+10,.0f}")
    L += ['', 'PART 2 - E(250, 250) - E(50, 50) split (Shapley; one-sided paths in brackets: fills first / eligibility first)',
          f"  pooled: total {sh['total']:+,.0f} = FILL {sh['fill']:+,.0f} + ELIGIBILITY {sh['eligibility']:+,.0f}   [{sh['fill_first']:+,.0f} / {sh['eligibility_first']:+,.0f}]"]
    for k in ('bid', 'ask'):
        x = SH['side'][k]; L.append(f"  {k:<8}: total {x['total']:+,.0f} = FILL {x['fill']:+,.0f} + ELIGIBILITY {x['eligibility']:+,.0f} | E(50, COMMON) {R[(50., 'C')]['side'][k]['excess']:+,.0f}, "
                                    f"E(250, COMMON) {R[(250., 'C')]['side'][k]['excess']:+,.0f}")
    for i, nm in enumerate(SRC):
        x = SH['src'][nm]; e50 = R[(50., 'C')]['src'].get(str(i), dict(excess=0.))['excess']; e250 = R[(250., 'C')]['src'].get(str(i), dict(excess=0.))['excess']
        L.append(f"  {nm:<8}: total {x['total']:+,.0f} = FILL {x['fill']:+,.0f} + ELIGIBILITY {x['eligibility']:+,.0f} | E(50, COMMON) {e50:+,.0f}, E(250, COMMON) {e250:+,.0f}")
    L += ['', 'PART 3 - the same opportunities (COMMON eligibility) by day: E(50, COMMON) | E(250, COMMON)']
    for d in days:
        L.append(f"  {d}: {R[(50., 'C')]['day'].get(d, dict(excess=0.))['excess']:+10,.0f} | {R[(250., 'C')]['day'].get(d, dict(excess=0.))['excess']:+10,.0f}")
    L += ['', 'READINGS (declared before computing)',
          f"  A0 REPRODUCED: {read['A0 REPRODUCED']}",
          f"  A1 ATTRIBUTION: {read['A1 ATTRIBUTION']} (FILL {sh['fill']:+,.0f} vs ELIGIBILITY {sh['eligibility']:+,.0f} of the {sh['total']:+,.0f} change)",
          f"  A2 SAME-OPPORTUNITY STABILITY: {read['A2 SAME-OPPORTUNITY STABILITY']} (E(50, COMMON) {R[(50., 'C')]['excess']:+,.0f}, E(250, COMMON) {R[(250., 'C')]['excess']:+,.0f})",
          '', 'LIMITS: consumed days; one declared rule; STATES is local selection, not a portfolio; the replay\'s fills (queue M, trade anchors); a diagnostic, not acceptance.']
    OUT.mkdir(parents=True); (OUT / 'clock_audit.txt').write_text('\n'.join(L) + '\n')
    C.dump(OUT / 'clock_audit.json', dict(asof=C.stamp(), readings=read, review_numbers=REVIEW, reproduced=got, R={f'{lab[k[0]]}_{lab[k[1]]}': v for k, v in R.items()},
                                          shapley=SH, source_receipts={d: C.sha(SRC_DIR / d / 'receipt.json') for d in days}, provenance=KP.provenance(), audit_source_sha256=C.sha(Path(__file__).resolve()),
                                          seconds=time.time() - start))
    print('\n'.join(L), flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif a == ['run']:
        run()
    else:
        sys.exit('usage: pf_keep_pull_clock_audit.py selftest | run')
