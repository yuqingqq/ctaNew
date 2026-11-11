"""B12c v2 (2026-11-11). B12c v1 (pf_b12c_model_or_speed.py 8bc18b8e, output 04:19Z) read INSTRUMENT-WEAK: our model's
guard value, even 2 s after the trade, accounted for none of the top makers' 1 s edge, so v1 could not tell a better
model from faster reaction. v2 asks the same question with the Binance price path itself as the instrument, and adds
a model-free split of when the edge builds. Declared before this script's first run; v1's output is kept as it is.
Read-only, descriptive. Population and outcomes as v1 (B12b part A: at-best legs of the 220 consumed 11-08 cohort
windows, first minute, guard ON / OFF at lag 50, trades that do not go through the level; the top 5 (T) against ranks
>= 6 (O); net value at h = 1 s and 5 s). G = 1 day < 5: point estimates only.

Instrument. bn(t) = the Binance mid at the latest point <= t of the research grid (step5 rows_rl_d2.bn_mid, 50 ms,
Tokyo-stamped; <= 100 ms old, else missing). Signed return r(a, b) = sg (ln bn(b) - ln bn(a)) 1e4 bps, sg = +1 for
bids and -1 for asks (positive = good for the maker). Per leg: r_past1 = r(anc - 1050, anc - 50); r_past = r(anc - 300,
anc - 50); r_L = r(anc - 50, anc + L) for L = 0, 250, 500, 1000, 2000 ms. delta proxy = |m0 - 50| (m0 the mid before
the trade, c) in terciles. Cells (75): 'past' = quintile(r_past1) x quintile(r_past) x tercile(delta); key L =
quintile(r_past) x quintile(r_L) x tercile(delta) (the last 250 ms before the report and the move after it). X(key, h) = (WITHIN_G - WITHIN_key) / WITHIN_G as v1. A leg
missing any Binance value or any extra mid is a status (one common sample).
Model-free split (descriptive): the within-guard gap of the move of the research mid over [anc, +250], [+250, +500],
[+500, +1000] and [+1000, +5000] ms (the spread before anc is equal for all groups, B12); the parts add up to the 5 s
gap minus the spread gap.

Readings DECLARED BEFORE THE FIRST RUN (evaluated in code, h = 1 s):
 R55 instrument  BN-OK if X(2000) >= 0.5 (the Binance path up to 2 s after the trade accounts for at least half of the
                 gap); else NOT-BINANCE (the edge is not about Binance moves up to 2 s later), and R56 reads n/a.
 R56 timing      VISIBLE if X(past) >= 0.5 (accounted for by Binance moves already in our data 50 ms before the trade
                 report); AHEAD if X(past) < 0.5 <= X(500); LATER if X(500) < 0.5 (and X(2000) >= 0.5).
 R57 main        VISIBLE -> REACT-FASTER; AHEAD -> FASTER-FEED-OR-SHORT-MODEL; LATER -> BETTER-MODEL; NOT-BINANCE ->
                 OTHER-INFORMATION.
 Null N5 (reported, not part of the labels, as v1's SPEED rule): the residual WITHIN at the deciding key under 500
 permutations of the T / O label within (window, side, cell), seed 20261111.
What the outcomes mean (fixed in advance): REACT-FASTER = their edge is about Binance moves that were already visible
in our data when the trade was reported: they get out of the way (or reprice) in time and others do not; our guard
does not use those sub-second moves. FASTER-FEED-OR-SHORT-MODEL = it is about moves our data shows within half a
second after the report: they see Binance sooner than our Tokyo-stamped feed, or predict the next half second.
BETTER-MODEL = it is about moves 0.5-2 s after the report: no feed is that much faster, so they predict them.
OTHER-INFORMATION = not Binance: Predict.fun's own flow or another venue.

Controls (any failure -> no table): v1's controls (pf_b12c_model_or_speed.controls, which include B12b's and B12's);
K0 v1's checks (b12.json reproduced; B12b part A legs; b12b.json hash); K1 the mids rebuilt here reproduce v1 and v5
on every leg of the sample (1e-9); G1 the Binance lookup and the signed return on a planted series (fresh, stale,
bid / ask sign); G2 three planted worlds through the same cells and labels: T selected on a move visible before the
report -> VISIBLE / REACT-FASTER; on a move 0.5-2 s after it -> LATER / BETTER-MODEL; on a non-Binance signal ->
NOT-BINANCE / OTHER-INFORMATION; G3 the label rules on planted curves, a NaN reads as n/a.
Limits fixed in advance: one consumed Sunday; trade times are reports (the match precedes anc by an unknown lag, p50
~0.2 s for our own legs), so 'visible before the report' includes moves just after the match; the Binance grid is the
research collector's clock; net returns and quintile cells are coarse summaries of the path (biases X down); delta is
proxied by the distance of the mid from 50 c only.
Usage: predictfun/research/run_light.sh b12c2_binance -- python3 -m predictfun.research.pf_b12c2_binance <out_dir>
       python3 -m predictfun.research.pf_b12c2_binance --selftest
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

from . import pf_b12_makers as B
from . import pf_b12b_reason as BB
from . import pf_b12c_model_or_speed as C1

LS = (0, 250, 500, 1000, 2000)
SEG = (250.0, 500.0, 1000.0, 5000.0)
N_NULL = 500; SEED = 20261111
NAN = float('nan')
BN = f'{BB.RP}/research/step5/20261108/rows_rl_d2.parquet'


def bnlook(tau, bn, t):
    """Binance mid at times t from the latest grid point <= FRESH ms old, NaN otherwise."""
    j = np.searchsorted(tau, t, side='right') - 1; jj = np.clip(j, 0, None)
    ok = (j >= 0) & (t - tau[jj] <= B.FRESH)
    return np.where(ok, bn[jj], np.nan)


def sret(sg, a, b):
    return sg * (np.log(b) - np.log(a)) * 1e4


def cells75(x1, x2, dl, e1, e2, ed):
    return (15 * BB.bucket(x1, e1) + 3 * BB.bucket(x2, e2) + BB.bucket(dl, ed)).astype(np.int64)


def key_cells(k, r_past1, r_past, rL, dl):
    """'past': quintile(r_past1) x quintile(r_past) x tercile(delta); L: quintile(r_past) x quintile(r_L) x tercile(delta)."""
    x1, x2 = (r_past1, r_past) if k == 'past' else (r_past, rL[int(k)])
    return cells75(x1, x2, dl, C1.qe(x1, 5), C1.qe(x2, 5), C1.qe(dl, 3))


def curve(r_past1, r_past, rL, dl, off, isT, sh, val):
    Wg = BB.decompose_n(off, isT, sh, val, 2)['WITHIN']; out = dict(WITHIN_G=Wg)
    for k in ['past'] + [str(L) for L in LS]:
        d = BB.decompose_n(key_cells(k, r_past1, r_past, rL, dl), isT, sh, val, 75)
        out[k] = dict(WITHIN=d['WITHIN'], MIX=d['MIX'], X=(Wg - d['WITHIN']) / Wg if Wg != 0 else NAN, excl_T=d['excl_T'], excl_O=d['excl_O'])
    return out


def label_r55(x2000):
    return 'n/a' if not BB.fin(x2000) else ('BN-OK' if x2000 >= 0.5 else 'NOT-BINANCE')


def label_r56(cv, r55):
    if r55 != 'BN-OK':
        return 'n/a'
    xp, x5 = cv['past']['X'], cv['500']['X']
    if not BB.fin(xp, x5):
        return 'n/a'
    return 'VISIBLE' if xp >= 0.5 else ('AHEAD' if x5 >= 0.5 else 'LATER')


def label_r57(r55, r56):
    if r55 == 'NOT-BINANCE':
        return 'OTHER-INFORMATION'
    return {'VISIBLE': 'REACT-FASTER', 'AHEAD': 'FASTER-FEED-OR-SHORT-MODEL', 'LATER': 'BETTER-MODEL'}.get(r56, 'n/a')


def planted(world, rng, n=30000):
    a = rng.normal(0, 0.2, n); b = rng.normal(0, 1, n); f1 = rng.normal(0, 0.5, n); f2 = rng.normal(0, 2, n); z = rng.normal(0, 1, n)
    r_past1 = a + b; r_past = b; rL = {0: rng.normal(0, 0.01, n), 250: 0.5 * f1, 500: f1, 1000: f1 + 0.33 * f2, 2000: f1 + f2}
    sel = {'visible': b, 'later': f2, 'other': z}[world]
    y = b + f1 + {'visible': 0.3 * f2, 'later': f2, 'other': 0.3 * f2 + 2.0 * z}[world] + rng.normal(0, 1, n)
    isT = rng.random(n) < 1.0 / (1.0 + np.exp(1.5 - 1.2 * sel)); off = (rng.random(n) < 0.85).astype(np.int64)
    return curve(r_past1, r_past, rL, rng.random(n) * 50, off, isT, 1.0 + rng.random(n), y)


def controls():
    rep = {}; ok = True

    def chk(name, cond):
        nonlocal ok
        rep[name] = bool(cond); ok = ok and bool(cond)
    okc, repc = C1.controls(); chk('B12c_v1_controls', okc)
    tau = np.array([0.0, 50.0, 100.0]); bn = np.array([100.0, 101.0, 102.0])
    x = bnlook(tau, bn, np.array([-10.0, 60.0, 260.0, 120.0]))
    r = sret(np.array([1.0, -1.0]), np.array([100.0, 100.0]), np.array([101.0, 101.0]))
    chk('G1_lookup_sign', np.isnan(x[0]) and x[1] == 101.0 and np.isnan(x[2]) and x[3] == 102.0 and r[0] > 99.0 and abs(r[0] + r[1]) < 1e-12)
    rng = np.random.default_rng(21); res = {}
    for w in ('visible', 'later', 'other'):
        cv = planted(w, rng); r55 = label_r55(cv['2000']['X']); r56 = label_r56(cv, r55)
        res[w] = dict(X_past=cv['past']['X'], X500=cv['500']['X'], X2000=cv['2000']['X'], R55=r55, R56=r56, R57=label_r57(r55, r56))
    chk('G2_worlds', res['visible']['R57'] == 'REACT-FASTER' and res['later']['R57'] == 'BETTER-MODEL' and res['other']['R57'] == 'OTHER-INFORMATION')
    rep['G2_detail'] = res
    cvA = {'past': dict(X=0.2), '500': dict(X=0.7)}; cvB = {'past': dict(X=NAN), '500': dict(X=0.7)}
    chk('G3_rules', label_r56(cvA, 'BN-OK') == 'AHEAD' and label_r57('BN-OK', 'AHEAD') == 'FASTER-FEED-OR-SHORT-MODEL' and label_r56(cvB, 'BN-OK') == 'n/a'
        and label_r56(cvA, 'NOT-BINANCE') == 'n/a' and label_r55(NAN) == 'n/a' and label_r57('n/a', 'n/a') == 'n/a')
    rep['B12c_v1_controls_detail'] = repc
    return ok, rep


def mids_at(D, W0, hs):
    """Research mid (last frame at or before t, forward-filled two-sided mid, as B12) at anc + h for the legs of D."""
    cols = ['mid', 't_ms', 'b1_p', 'a1_p']; parts = []
    for fn in sorted(glob.glob(f'{BB.S1}/books/part_*.parquet')):
        d = pd.read_parquet(fn, columns=cols); parts.append(d[d.mid.isin(W0)])
    Bk = pd.concat(parts, ignore_index=True); del parts
    Bk.sort_values(['mid', 't_ms'], inplace=True, kind='stable'); BI = Bk.groupby('mid', sort=False).indices
    out = {h: np.full(len(D), np.nan) for h in hs}
    for m, ii in D.groupby('mid', sort=False).indices.items():
        b = Bk.iloc[BI[m]]; tf = b.t_ms.values.astype(float); bb = b.b1_p.values.astype('int64'); ba = b.a1_p.values.astype('int64')
        mff = pd.Series(np.where((bb >= 0) & (ba > bb), (ba + bb) / 2.0, np.nan)).ffill().values; ta = D.anc.values[ii]
        for h in hs:
            j1 = np.searchsorted(tf, ta + h, side='right') - 1; out[h][ii] = np.where(j1 >= 0, mff[np.clip(j1, 0, None)], np.nan)
    return out


def main():
    if '--selftest' in sys.argv:
        ok, rep = controls(); print(json.dumps({k: v for k, v in rep.items() if not k.endswith('detail')}, indent=1))
        print(json.dumps(rep['G2_detail'], indent=1, default=float)); print('selftest', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    ok, ctl = controls()
    if not ok:
        sys.exit(f'CONTROL FAILED -> no table: {ctl}')
    sums = dict(line.split()[::-1] for line in open(C1.REV) if line.strip())
    if hashlib.sha256(open(f'{C1.B12B_OUT}/b12b.json', 'rb').read()).hexdigest() != sums['b12b.json']:
        sys.exit('REFUSED: b12b.json differs from the committed copy')
    b12b = json.load(open(f'{C1.B12B_OUT}/b12b.json'))
    X = BB.build(); R = X['R']; W0 = X['W0']
    k0 = BB.reconcile(R, X['days'])
    V = R[R.val_ok]; A = V[(V.status == 'ok') & (V.depth == 0) & V.g50.isin([B.G_ON, B.G_OFF]) & (V.sweep == 0)]
    D = A[A.g.isin(['top 5', '6-20', 'rest'])].copy()
    k0['partA_legs'] = dict(top5=int((D.g == 'top 5').sum()), O=int((D.g != 'top 5').sum()))
    k0['partA_ok'] = (k0['partA_legs']['top5'] == b12b['A']['groups']['top 5']['legs'] and k0['partA_legs']['O'] == b12b['A']['groups']['O (ranks >= 6)']['legs'])
    if not (k0['ok'] and k0['partA_ok']):
        sys.exit(f'CONTROL FAILED (K0) -> no table: {k0}')
    sg = np.where(D.side.values == 0, 1.0, -1.0); p = D.ptick.values.astype(float); m0 = p + sg * D.spr.values
    M = mids_at(D, W0, (250.0, 500.0, 1000.0, 5000.0))
    v1r = sg * (M[1000.0] - p); v5r = sg * (M[5000.0] - p)
    k1 = dict(v1_max_abs_diff=float(np.nanmax(np.abs(v1r - D.v1.values))), v5_max_abs_diff=float(np.nanmax(np.abs(v5r - D.v5.values))))
    k1['ok'] = bool(k1['v1_max_abs_diff'] < 1e-9 and k1['v5_max_abs_diff'] < 1e-9 and np.isfinite(v1r).all())
    if not k1['ok']:
        sys.exit(f'CONTROL FAILED (K1 mids) -> no table: {k1}')
    BNd = pd.read_parquet(BN, columns=['mid', 'tau', 'bn_mid']); BNd = BNd[BNd.mid.isin(W0)].sort_values(['mid', 'tau'], kind='stable')
    BG = {m: (g.tau.values.astype(float), g.bn_mid.values.astype(float)) for m, g in BNd.groupby('mid')}; del BNd
    offs = {'m1050': -1050.0, 'm300': -300.0, 'm50': -50.0, **{f'p{L}': float(L) for L in LS}}
    bv = {k: np.full(len(D), np.nan) for k in offs}
    for m, ii in D.groupby('mid', sort=False).indices.items():
        if m not in BG:
            continue
        tau, bn = BG[m]; ta = D.anc.values[ii]
        for k, o in offs.items():
            bv[k][ii] = bnlook(tau, bn, ta + o)
    okm = np.all(np.isfinite(np.vstack(list(bv.values()) + [M[250.0], M[500.0]])), axis=0) & np.isfinite(m0)
    stat = dict(partA_legs=int(len(D)), missing_legs=int((~okm).sum()), missing_shares_T=float(D.shares.values[~okm & (D.g.values == 'top 5')].sum()),
                missing_shares_O=float(D.shares.values[~okm & (D.g.values != 'top 5')].sum()))
    sel = okm; Dd = D[sel]; sgd = sg[sel]; b_ = {k: v[sel] for k, v in bv.items()}
    r_past1 = sret(sgd, b_['m1050'], b_['m50']); r_past = sret(sgd, b_['m300'], b_['m50'])
    rL = {L: sret(sgd, b_['m50'], b_[f'p{L}']) for L in LS}; dl = np.abs(m0[sel] - 50.0)
    off = (Dd.g50.values == B.G_OFF).astype(np.int64); isT = Dd.g.values == 'top 5'; sh = Dd.shares.values
    cv = {h: curve(r_past1, r_past, rL, dl, off, isT, sh, Dd[f'v{h}'].values) for h in (1, 5)}
    r55 = label_r55(cv[1]['2000']['X']); r56 = label_r56(cv[1], r55); r57 = label_r57(r55, r56)
    dk = {'VISIBLE': 'past', 'AHEAD': '500', 'LATER': '2000'}.get(r56, '2000')
    c = key_cells(dk, r_past1, r_past, rL, dl); val = Dd.v1.values
    blk = pd.factorize(pd.Series(list(zip(Dd.mid.values, Dd.side.values, c))))[0]
    nul = BB.null_within_n(c, isT, sh, val, 75, blk, N_NULL, SEED); obs = BB.within_n(c, isT, sh, val, 75)
    n5 = dict(key=dk, observed=obs, p50=float(np.percentile(nul, 50)), p99=float(np.percentile(nul, 99)), frac_ge=float((nul >= obs).mean()), draws=N_NULL, seed=SEED)
    # model-free split of the gap by when the move happens
    mm = {0.0: m0[sel], **{h: M[h][sel] for h in SEG}}; edges = [0.0] + list(SEG); split = {}
    for a_, b2 in zip(edges[:-1], edges[1:]):
        seg = sgd * (mm[b2] - mm[a_]); split[f'{int(a_)}-{int(b2)} ms'] = BB.decompose_n(off, isT, sh, seg, 2)['WITHIN']
    split['spread'] = BB.decompose_n(off, isT, sh, Dd.spr.values, 2)['WITHIN']; split['net 5 s'] = cv[5]['WITHIN_G']; split['net 1 s'] = cv[1]['WITHIN_G']
    # Binance moves by group: signed past and next-half-second returns (share-weighted means, bps)
    bng = {g: dict(r_past=B.wmean(sh[m_], r_past[m_]), r_past1=B.wmean(sh[m_], r_past1[m_]), r_500=B.wmean(sh[m_], rL[500][m_]), r_2000=B.wmean(sh[m_], rL[2000][m_]))
           for g, m_ in (('top 5', isT), ('O (ranks >= 6)', ~isT))}
    res = dict(as_of_note='point estimates, one consumed partial day (220 windows of 2026-11-08, 0.764 window-days)', controls=ctl, K0=k0, K1=k1, stat=stat,
               curves={str(h): c_ for h, c_ in cv.items()}, N5=n5, split_by_time=split, binance_by_group=bng, readings=dict(R55=r55, R56=r56, R57=r57))
    json.dump(res, open(f'{out}/b12c2.json', 'w'), indent=1, default=float)
    f = lambda v: 'n/a' if not BB.fin(v) else f'{v:+.3f}'
    Lt = [f"B12c v2 (Binance instrument): better model or faster reaction. {res['as_of_note']}",
          f"Controls: {json.dumps({k: v for k, v in ctl.items() if not k.endswith('detail')})}; planted worlds {json.dumps(ctl['G2_detail'], default=float)}",
          f"K0 {k0['ok']} part A {k0['partA_legs']} {k0['partA_ok']}; K1 {k1}; status {stat}", '',
          "Curves (top 5 against O): h | WITHIN_G | key: WITHIN / X (share accounted for by the Binance path; past = up to 50 ms before the report; L = up to anc + L ms)"]
    for h, c_ in cv.items():
        Lt.append(f"{h} s | {f(c_['WITHIN_G'])} | " + ' | '.join(f"{k}: {f(c_[k]['WITHIN'])} / {c_[k]['X']:.2f}" for k in ['past'] + [str(L) for L in LS]))
    Lt.append(f"N5 residual at key {n5['key']} (1 s): WITHIN {f(n5['observed'])}, p50 {f(n5['p50'])}, p99 {f(n5['p99'])}, share >= observed {n5['frac_ge']:.3f}")
    Lt.append('Gap by when the move happens (within guard, c/sh): ' + '; '.join(f"{k} {f(v)}" for k, v in split.items()))
    Lt.append('Signed Binance returns at their fills (bps; + = good for the maker): ' + '; '.join(
        f"{g}: past 250 ms {v['r_past']:+.3f}, past 1 s {v['r_past1']:+.3f}, next 0.5 s {v['r_500']:+.3f}, next 2 s {v['r_2000']:+.3f}" for g, v in bng.items()))
    Lt += ['', 'Readings: ' + json.dumps(res['readings'])]
    open(f'{out}/b12c2.txt', 'w').write('\n'.join(Lt) + '\n')
    print('\n'.join(Lt))


if __name__ == '__main__':
    main()
