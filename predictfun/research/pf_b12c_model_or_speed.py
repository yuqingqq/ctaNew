"""B12c (2026-11-11, USER "Can you check if they have a better model, or they just react faster"): is the top makers'
edge information that our own model gets a little later (they react faster), or information it does not get (a better
model)? Read-only, descriptive. Same population as B12b part A (pf_b12b_reason.py c882da77): at-best legs of the 220
consumed 11-08 cohort windows, first minute, guard ON / OFF at lag 50, trades that do not go through the level, the
top 5 (T) against ranks >= 6 (O). G = 1 day < 5: point estimates only.

Idea. B12b (post hoc) found that the top makers' within-guard gap sits while our guard's value v is falling, and that
v at the trade explains none of it. Here our model gets extra time: condition on what v says L ms AFTER the trade. If
the gap goes away once our model has had L ms more, the top makers acted on information our model also gets, only
sooner: speed (a faster feed, model update or cancel path). If it stays, they act on information our model does not
get: a better model or other inputs.

Measures. v(t) = the side's guard value at time t from the latest grid point <= 100 ms old (as B12); the warm-up
placeholder and NaN read as missing. Per leg: level = v(anc - 50); past change dP = v(anc - 50) - v(anc - 300); future
change dL = v(anc + L) - v(anc - 50) for L = 0, 250, 500, 1000, 2000 ms. A leg missing any of these is a status (one
common sample for every L). Outcomes: net value at h = 1 s and 5 s (c/sh, as B12).
Cells: quintile of the level x decile of the change (dP for 'past', dL for each L); edges are quantiles of the T + O
legs. WITHIN_G = the within-guard gap (2 cells); WITHIN_L = the gap within the L cells; X(L, h) = (WITHIN_G - WITHIN_L)
/ WITHIN_G = the share of the gap that our model's information up to anc + L accounts for.

Readings DECLARED BEFORE THE FIRST RUN (evaluated in code):
 R53 instrument  INSTRUMENT-OK if X(2000, 1 s) >= 0.5 (two seconds after the trade, our model's value accounts for at
                 least half of a 1 s gap, whatever its source); else INSTRUMENT-WEAK, and R50 / R51 read 'n/a
                 (instrument weak)'.
 R50 source      primary h = 5 s, L = 500 ms (our model half a second later, a tenth of the horizon). If WITHIN_G(5 s) <
                 +0.10 c/sh the primary is h = 1 s, L = 250 ms (label suffixed ' (1 s)'); NO-GAP if that is also <
                 +0.10. SPEED if X >= 0.5; MODEL if X < 0.25 and WITHIN_L is above its null p99 (N4); else MIXED.
 R51 head start  at the primary h: the smallest L in 250 / 500 / 1000 / 2000 ms with X(L) >= 0.5 -> MATCHED-AT-<L>MS
                 (our model matches them with about L ms more); NONE-WITHIN-2S if none.
 Null N4: 500 permutations of the T / O label within (window, side, cell) at the primary cells, seed 20261111.
What the outcomes mean (fixed in advance): SPEED = they act sooner on what our model learns within half a second
(their data, model update or cancel path is faster than ours). MODEL = at the information our model has half a second
later they still do better: they have an input or a model that ours lacks. MIXED = part of each. SPEED does not say
which part of their path (data, decision, cancel) is faster.

Controls (any failure -> no table): B12b's controls (pf_b12b_reason.controls, which include B12's); K0 the rebuilt
frame reproduces the committed b12.json and B12b's part A population (legs of the top 5 and of O; b12b.json hash
checked); F1 the value lookup on a planted series (the latest fresh point; stale, warm-up and NaN read as missing); F2
two planted worlds through the same cells and labels: 'speed' (T is selected on a move that shows in our value from
+300 ms) -> X(500) >= 0.5, SPEED, MATCHED-AT-500MS; 'model' (T is selected on a move that never shows in our value) ->
X(500) < 0.25 and MODEL; F3 the head-start, fallback and NO-GAP rules on planted curves; a NaN reads as n/a.
Limits fixed in advance: one consumed Sunday; trade times are reports (the match time is unknown); our model's value is
the research replay of the guard, not the live engine's; quintile x decile cells are coarse, so part of any explained
share is lost (this biases X toward MODEL; R53 shows how much the instrument can account for); X(L) also credits our
model with what it learns about the next L ms, so a model edge on horizons under L reads as speed.
Usage: predictfun/research/run_light.sh b12c_model_or_speed -- python3 -m predictfun.research.pf_b12c_model_or_speed <out_dir>
       python3 -m predictfun.research.pf_b12c_model_or_speed --selftest
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

from . import pf_b12_makers as B
from . import pf_b12b_reason as BB

B12B_OUT = f'{BB.LV}/b12b_20261111'
LS = (0, 250, 500, 1000, 2000)
PRIMARY = (5, 500); FALLBACK = (1, 250)
N_NULL = 500; SEED = 20261111
NAN = float('nan')
REV = '/home/yuqing/ctaNew/predictfun/reviews/b12b_20261111/SHA256SUMS'


def vlook(tau, v, t, line):
    """The side's value at times t from the latest grid point <= FRESH ms old; warm-up, NaN and stale read as NaN."""
    st, x = B.guard_state(tau, v, t, line)
    return np.where((st == B.G_ON) | (st == B.G_OFF), x, np.nan)


def qe(x, k):
    return np.unique(np.quantile(x, np.arange(1, k) / k))


def cells(level, change, e5, e10):
    return (10 * BB.bucket(level, e5) + BB.bucket(change, e10)).astype(np.int64)


def curve(lev, chg, off, isT, sh, val):
    """WITHIN_G and, per change key, WITHIN / MIX / X over quintile(level) x decile(change) cells."""
    Wg = BB.decompose_n(off, isT, sh, val, 2)['WITHIN']; e5 = qe(lev, 5); out = dict(WITHIN_G=Wg)
    for k, ch in chg.items():
        d = BB.decompose_n(cells(lev, ch, e5, qe(ch, 10)), isT, sh, val, 50)
        out[k] = dict(WITHIN=d['WITHIN'], MIX=d['MIX'], X=(Wg - d['WITHIN']) / Wg if Wg != 0 else NAN, excl_T=d['excl_T'], excl_O=d['excl_O'])
    return out


def label_r53(x):
    return 'n/a' if not BB.fin(x) else ('INSTRUMENT-OK' if x >= 0.5 else 'INSTRUMENT-WEAK')


def pick_primary(Wg5, Wg1):
    if BB.fin(Wg5) and Wg5 >= 0.10:
        return PRIMARY, ''
    if BB.fin(Wg1) and Wg1 >= 0.10:
        return FALLBACK, ' (1 s)'
    return None, ''


def label_r50(X, W, p99, inst, prim, suffix):
    if inst != 'INSTRUMENT-OK':
        return 'n/a (instrument weak)' if inst == 'INSTRUMENT-WEAK' else 'n/a'
    if prim is None:
        return 'NO-GAP'
    if not BB.fin(X, W, p99):
        return 'n/a'
    return ('SPEED' if X >= 0.5 else ('MODEL' if (X < 0.25 and W > p99) else 'MIXED')) + suffix


def label_r51(cv, inst, prim):
    if inst != 'INSTRUMENT-OK':
        return 'n/a (instrument weak)' if inst == 'INSTRUMENT-WEAK' else 'n/a'
    if prim is None:
        return 'NO-GAP'
    for L in (250, 500, 1000, 2000):
        x = cv.get(str(L), {}).get('X', NAN)
        if BB.fin(x) and x >= 0.5:
            return f'MATCHED-AT-{L}MS'
    return 'NONE-WITHIN-2S'


def controls():
    rep = {}; ok = True

    def chk(name, cond):
        nonlocal ok
        rep[name] = bool(cond); ok = ok and bool(cond)
    okb, repb = BB.controls(); chk('B12b_controls', okb)
    # F1 lookup
    tau = np.array([0.0, 50.0, 100.0, 150.0]); v = np.array([-1.0, 0.03, 0.06, np.nan])
    x = vlook(tau, v, np.array([40.0, 60.0, 160.0, 400.0, 120.0]), B.LINE_DECL)
    chk('F1_lookup', np.isnan(x[0]) and x[1] == 0.03 and np.isnan(x[2]) and np.isnan(x[3]) and x[4] == 0.06)
    # F2 planted worlds
    rng = np.random.default_rng(5); n = 20000
    s = rng.normal(0, 1, n); y = s + rng.normal(0, 1, n); lev = rng.normal(0, 1, n); off = (rng.random(n) < 0.85).astype(np.int64)
    isT = rng.random(n) < 1.0 / (1.0 + np.exp(1.5 - 1.0 * s)); sh = 1.0 + rng.random(n)
    noise = {k: rng.normal(0, 1, n) for k in ['past'] + [str(L) for L in LS]}
    res = {}
    for world in ('speed', 'model'):
        chg = {k: (s + 0.2 * noise[k] if (world == 'speed' and k not in ('past', '0', '250')) else noise[k]) for k in noise}
        cv = curve(lev, chg, off, isT, sh, y)
        e5 = qe(lev, 5); c5 = cells(lev, chg['500'], e5, qe(chg['500'], 10))
        blk = pd.factorize(pd.Series(list(zip(np.zeros(n, int), np.zeros(n, int), c5))))[0]
        nul = BB.null_within_n(c5, isT, sh, y, 50, blk, 100, 9)
        res[world] = dict(X500=cv['500']['X'], X250=cv['250']['X'],
                          r50=label_r50(cv['500']['X'], cv['500']['WITHIN'], float(np.percentile(nul, 99)), 'INSTRUMENT-OK', PRIMARY, ''),
                          r51=label_r51(cv, 'INSTRUMENT-OK', PRIMARY))
    chk('F2_worlds', res['speed']['X500'] >= 0.5 and res['speed']['r50'] == 'SPEED' and res['speed']['r51'] == 'MATCHED-AT-500MS'
        and res['model']['X500'] < 0.25 and res['model']['r50'] == 'MODEL' and res['model']['r51'] == 'NONE-WITHIN-2S')
    rep['F2_detail'] = res
    # F3 rules
    cvA = {'250': dict(X=0.2), '500': dict(X=0.6), '1000': dict(X=0.8), '2000': dict(X=0.9)}
    cvB = {'250': dict(X=0.1), '500': dict(X=0.2), '1000': dict(X=0.3), '2000': dict(X=0.4)}
    chk('F3_rules', label_r51(cvA, 'INSTRUMENT-OK', PRIMARY) == 'MATCHED-AT-500MS' and label_r51(cvB, 'INSTRUMENT-OK', PRIMARY) == 'NONE-WITHIN-2S'
        and pick_primary(0.05, 0.2) == (FALLBACK, ' (1 s)') and pick_primary(0.2, 0.0) == (PRIMARY, '') and pick_primary(NAN, 0.05) == (None, '')
        and label_r50(0.6, 0.1, 0.05, 'INSTRUMENT-OK', None, '') == 'NO-GAP' and label_r50(0.6, 0.1, 0.05, 'INSTRUMENT-WEAK', PRIMARY, '') == 'n/a (instrument weak)'
        and label_r50(0.1, 0.3, 0.05, 'INSTRUMENT-OK', FALLBACK, ' (1 s)') == 'MODEL (1 s)' and label_r50(0.3, 0.3, 0.05, 'INSTRUMENT-OK', PRIMARY, '') == 'MIXED'
        and label_r50(NAN, 0.3, 0.05, 'INSTRUMENT-OK', PRIMARY, '') == 'n/a' and label_r53(NAN) == 'n/a' and label_r53(0.49) == 'INSTRUMENT-WEAK')
    rep['B12b_controls_detail'] = repb
    return ok, rep


def main():
    if '--selftest' in sys.argv:
        ok, rep = controls(); print(json.dumps(rep, indent=1, default=float)); print('selftest', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    ok, ctl = controls()
    if not ok:
        sys.exit(f'CONTROL FAILED -> no table: {ctl}')
    sums = dict(line.split()[::-1] for line in open(REV) if line.strip())
    if hashlib.sha256(open(f'{B12B_OUT}/b12b.json', 'rb').read()).hexdigest() != sums['b12b.json']:
        sys.exit('REFUSED: b12b.json differs from the committed copy')
    b12b = json.load(open(f'{B12B_OUT}/b12b.json'))
    X = BB.build(); R = X['R']; G = X['G']; LINE = X['LINE']
    k0 = BB.reconcile(R, X['days'])
    V = R[R.val_ok]; A = V[(V.status == 'ok') & (V.depth == 0) & V.g50.isin([B.G_ON, B.G_OFF]) & (V.sweep == 0)]
    D = A[A.g.isin(['top 5', '6-20', 'rest'])].copy()
    k0['partA_legs'] = dict(top5=int((D.g == 'top 5').sum()), O=int((D.g != 'top 5').sum()))
    k0['partA_ok'] = (k0['partA_legs']['top5'] == b12b['A']['groups']['top 5']['legs'] and k0['partA_legs']['O'] == b12b['A']['groups']['O (ranks >= 6)']['legs'])
    if not (k0['ok'] and k0['partA_ok']):
        sys.exit(f'CONTROL FAILED (K0) -> no table: {k0}')
    offs = {'m300': -300.0, 'm50': -50.0, 'p0': 0.0, 'p250': 250.0, 'p500': 500.0, 'p1000': 1000.0, 'p2000': 2000.0}
    vals = {k: np.full(len(D), np.nan) for k in offs}
    for (m, side), ii in D.groupby(['mid', 'side'], sort=False).indices.items():
        tau, vb, va = G[m]; vs = (vb, va)[side]; ta = D.anc.values[ii]
        for k, o in offs.items():
            vals[k][ii] = vlook(tau, vs, ta + o, LINE)
    fin_all = np.all(np.isfinite(np.vstack(list(vals.values()))), axis=0)
    stat = dict(partA_legs=int(len(D)), missing_value_legs=int((~fin_all).sum()),
                missing_value_shares_T=float(D.shares.values[~fin_all & (D.g.values == 'top 5')].sum()),
                missing_value_shares_O=float(D.shares.values[~fin_all & (D.g.values != 'top 5')].sum()))
    D = D[fin_all]; vv = {k: x[fin_all] for k, x in vals.items()}
    lev = vv['m50']; chg = {'past': vv['m50'] - vv['m300']}
    for L in LS:
        chg[str(L)] = vv[f'p{L}'] - vv['m50']
    off = (D.g50.values == B.G_OFF).astype(np.int64); isT = D.g.values == 'top 5'; sh = D.shares.values
    cv = {h: curve(lev, chg, off, isT, sh, D[f'v{h}'].values) for h in (1, 5)}
    inst = label_r53(cv[1]['2000']['X'])
    prim, suffix = pick_primary(cv[5]['WITHIN_G'], cv[1]['WITHIN_G'])
    null = None
    if prim is not None:
        h, L = prim; val = D[f'v{h}'].values; e5 = qe(lev, 5); c = cells(lev, chg[str(L)], e5, qe(chg[str(L)], 10))
        blk = pd.factorize(pd.Series(list(zip(D.mid.values, D.side.values, c))))[0]
        nul = BB.null_within_n(c, isT, sh, val, 50, blk, N_NULL, SEED); obs = BB.within_n(c, isT, sh, val, 50)
        null = dict(h=h, L=L, observed=obs, p50=float(np.percentile(nul, 50)), p99=float(np.percentile(nul, 99)), draws=N_NULL, seed=SEED,
                    frac_ge=float((nul >= obs).mean()))
        r50 = label_r50(cv[h][str(L)]['X'], cv[h][str(L)]['WITHIN'], null['p99'], inst, prim, suffix)
        r51 = label_r51(cv[h], inst, prim)
    else:
        r50 = label_r50(NAN, NAN, NAN, inst, None, ''); r51 = label_r51({}, inst, None)
    # descriptive: the gap by tercile of the half-second future change (primary h)
    hh = prim[0] if prim else 5; val = D[f'v{hh}'].values; e3 = qe(chg['500'], 3); t3 = BB.bucket(chg['500'], e3); by3 = []
    for i in range(len(e3) + 1):
        mk = t3 == i
        by3.append(dict(wT=float(sh[mk & isT].sum() / sh[isT].sum()), wO=float(sh[mk & ~isT].sum() / sh[~isT].sum()),
                        T=B.wmean(sh[mk & isT], val[mk & isT]), O=B.wmean(sh[mk & ~isT], val[mk & ~isT])))
    res = dict(as_of_note='point estimates, one consumed partial day (220 windows of 2026-11-08, 0.764 window-days)', controls=ctl, K0=k0, stat=stat,
               curves={str(h): c_ for h, c_ in cv.items()}, primary=dict(h=prim[0], L=prim[1]) if prim else None, N4=null,
               by_future_change_tercile=dict(h=hh, edges=[float(x) for x in e3], cells=by3),
               readings=dict(R53=inst, R50=r50, R51=r51))
    json.dump(res, open(f'{out}/b12c.json', 'w'), indent=1, default=float)
    f = lambda v: 'n/a' if not BB.fin(v) else f'{v:+.3f}'
    Lt = [f"B12c better model or faster reaction. {res['as_of_note']}",
          f"Controls: {json.dumps({k: v for k, v in ctl.items() if not k.endswith('detail')})}; planted worlds {json.dumps(ctl['F2_detail'], default=float)}",
          f"K0: {k0['ok']}, part A legs {k0['partA_legs']} ok {k0['partA_ok']}; status {stat}", '',
          'Curves (top 5 against O): h | WITHIN_G | key: WITHIN / X (share explained by our model\'s information up to anc + key ms; past = the 250 ms before)']
    for h, c_ in cv.items():
        Lt.append(f"{h} s | {f(c_['WITHIN_G'])} | " + ' | '.join(f"{k}: {f(c_[k]['WITHIN'])} / {c_[k]['X']:.2f}" for k in ['past'] + [str(L) for L in LS]))
    if null and prim:
        Lt.append(f"N4 null (h {null['h']} s, L {null['L']} ms): WITHIN {f(null['observed'])}, p50 {f(null['p50'])}, p99 {f(null['p99'])}, share >= observed {null['frac_ge']:.3f}")
    Lt.append(f"By tercile of the change of our value over the half second after the trade (h {hh} s; edges {[round(x, 4) for x in e3]}; falling -> rising):")
    Lt += [f"   tercile {i}: wT {c['wT'] * 100:.1f}% wO {c['wO'] * 100:.1f}% T {f(c['T'])} O {f(c['O'])}" for i, c in enumerate(by3)]
    Lt += ['', 'Readings: ' + json.dumps(res['readings'])]
    open(f'{out}/b12c.txt', 'w').write('\n'.join(Lt) + '\n')
    print('\n'.join(Lt))


if __name__ == '__main__':
    main()
