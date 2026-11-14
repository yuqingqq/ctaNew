"""LEARN-PULL-TARGET (2026-11-14, session ce68ebe0; POST HOC on LEARN-PULL's observation files; USER: "Why do they pull? You mean they
pull because some trade will take the price level? Or what is their target to train the model to pull").

We cannot see the top makers' model or its target. What we can measure is which outcome their pulls line up with. Two candidates:
  LEVEL-TAKEN  the next trade at their price goes THROUGH it (LEARN-PULL's 'through')
  FILL-LOSES   a fill at their price on that next trade loses at 5 s (v5 < 0), whether the trade goes through or stops there
Our guard's own target, for contrast (pf_badfill_grid_tx.py, the frozen guard's label): v = s (m(t_sw + 1 s) - p) if the known best p is
swept through in (t + 22, t + 522] ms, else 0. That is LEVEL-TAKEN, harm only: a fill that stops at the price counts 0. Pulled at
v <= -0.10 c, the same line for an order at the front or at the back.

Population: LEARN-PULL's observations (obs_<day>.parquet of learn_pull_20261113; 9 consumed days 10-23..27, 10-31, 11-01..03, first 60 s,
WS-timed tape): front orders at the best proven by a sole-maker stop, followed to the next trade at that price within 3 s. Top 5 (the
field printed beside). No raw data is read and nothing is simulated. Before these readings were declared, only column ranges and the v5
sign split were looked at (top 5: v5 < 0 43.9%, = 0 0.3%, > 0 55.9%), not any relation to GONE.
Measures (GONE = their front order was not hit: pulled or moved; values c/sh weighted by the trade's volume at the price, as LEARN-PULL;
interval buckets 0-250 / 250-1000 / 1000-3000 ms to the next trade, the last including 3000):
  T1  GONE share by the next trade's kind (THROUGH / STOP) x the fill's 5 s value sign (v5 < 0 / > 0; v5 = 0 is a status), per interval
      bucket and bucket-standardised (each bucket weighted by its share of all observations: removes "longer interval, more pulls");
      the same for our guard's pull flag (LEARN-PULL's we_pull) on the same observations; 1 s beside
  T2  the value gap, value when they stayed minus when they were GONE, and the part the trade's kind accounts for (each observation's
      v5 replaced by the mean of its kind, the gap recomputed); also with kind x interval bucket cells
  T3  selection at their own pull count: in each day x interval bucket, pull as many observations as they did, chosen by (a) their actual
      pulls, (b) our guard value v 100 ms before the trade (lowest first), (c) FRONT-PULL's P (highest first), (d) random (200 draws),
      (e) the fill's own value (v5 x volume, lowest first: the bound); the value kept ($ a day, c/sh), and the shares of THROUGH trades
      and of bad trades (v5 <= -2 c) among those pulled. Rows without v or P are a status, excluded from T3 for every arm
Readings DECLARED BEFORE THIS SCRIPT'S FIRST RUN (post hoc on LEARN-PULL's data; descriptive; day rule >= 7 of 9):
 R557 BEYOND-SWEEP: within STOP trades, the top 5's GONE share when the fill loses (v5 < 0) over when it wins (v5 > 0), bucket-
      standardised: >= 1.2 pooled and > 1 on >= 7 of 9 days -> PULLS-TRACK-FILL-LOSS; <= 1 / 1.2 pooled and < 1 on >= 7 ->
      PULLS-AVOID-WINNERS; else NOT-SHOWN.
 R558 KIND-SHARE: the share of the stayed - gone value gap that the trade's kind accounts for: < 0.5 pooled and on >= 7 of 9 days ->
      GAP-MOSTLY-WITHIN-KIND (their pulls pick losing fills among trades that stop at their price); >= 0.5 pooled and on >= 7 ->
      GAP-MOSTLY-KIND; else NOT-SHOWN (a day whose gap is <= 0 counts for neither).
 R559 MATCHED-GUARD: value kept by our guard's ranking against their actual pulls, at their pull count: theirs higher on >= 7 of 9 days
      -> THEY-PICK-BETTER; ours higher or equal on >= 7 -> GUARD-PICKS-AS-WELL; else NOT-SHOWN. R559P: the same with P (its second
      label P-PICKS-AS-WELL).
 R560 NULL-CONTROL: their value kept above the random p95 on >= 7 of 9 days; otherwise R559 / R559P read VOID.
Controls (any failure -> no output): synthetic sets with known answers: GONE driven by the kind only must read NOT-SHOWN in R557 and
GAP-MOSTLY-KIND in R558; GONE driven by the fill's loss within STOP must read PULLS-TRACK-FILL-LOSS and GAP-MOSTLY-WITHIN-KIND; the bound
(e) keeps at least as much as every other arm (random: its p95) on every day, in the synthetic sets and in the real data; a planted error (the value sign
flipped in the value-kept function) must fail that check; every arm pulls exactly their count in every cell; memory self-guard (2 GiB).
Limits: fills only (GONE includes moves to another price); their own pull can make the level break (they are about a third of the volume),
so an outcome after a pull is not independent of it; v and P are our signals, read 100 ms before the trade (they may have pulled earlier
in the interval); the population is front orders after small stopping trades, E2-ambiguous intervals excluded; values are 5 s marks of
one fill; consumed days (C8); no replay of our orders here (C7; C9: theirs are real fills, ours a ranking of our signal on their orders).
Usage (run_light, PF_SLOT=claude): pf_learn_pull_target.py selftest | report; driver run_learn_pull_target.sh
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_learn_pull as LP                                # noqa: E402  (OUTD, DAYS9, GAPB, K)

DAYS9 = LP.DAYS9
GAPB = LP.GAPB
OUTT = f'{LP.OUTD}/target'
MEM_CAP_MIB = 2048.0
NDRAW = 200; SEED = 557; BAD = -2.0; RT = 1.2; DAYRULE = 7
NAN = float('nan')
CELLS = (('THROUGH, fill loses', 1, -1), ('THROUGH, fill wins', 1, 1), ('STOP, fill loses', 0, -1), ('STOP, fill wins', 0, 1))


def bucket(gap):
    b = np.full(len(gap), -1, int)
    for i, (_, lo, hi) in enumerate(GAPB):
        b[(gap >= lo) & ((gap < hi) | ((i == len(GAPB) - 1) & (gap <= hi)))] = i
    return b


def vw(x, w):
    return float((x * w).sum() / w.sum()) if len(x) and w.sum() > 0 else NAN


def std_share(y, mask, b, wb):
    """Bucket-standardised mean of y over rows in mask: bucket means weighted by wb (renormalised over buckets present)."""
    s = wt = 0.0
    for i in range(len(GAPB)):
        m = mask & (b == i)
        if m.any():
            s += wb[i] * float(y[m].mean()); wt += wb[i]
    return s / wt if wt > 0 else NAN


def t1_cells(y, through, val, b, wb):
    sg = np.sign(val)
    return {nm: (std_share(y, (through == th) & (sg == s), b, wb), int(((through == th) & (sg == s)).sum())) for nm, th, s in CELLS}


def loss_ratio(y, through, val, b, wb):
    stop = through == 0
    a = std_share(y, stop & (val < 0), b, wb); c = std_share(y, stop & (val > 0), b, wb)
    return a / c if c > 0 else NAN


def gap_share(v, w, gone, cells):
    """Gap = vw(v | stayed) - vw(v | gone); counterfactual: v replaced by its cell's vw mean. Returns gap, gap explained, share."""
    st = gone == 0; go = gone == 1
    gap = vw(v[st], w[st]) - vw(v[go], w[go])
    c = np.empty(len(v))
    for key in np.unique(cells):
        m = cells == key; c[m] = vw(v[m], w[m])
    gcf = vw(c[st], w[st]) - vw(c[go], w[go])
    return gap, gcf, (gcf / gap if gap > 0 else NAN)


def pick(score, cell, kc, rng):
    """Pull the kc[c] highest-score rows of each cell c; ties broken by a random key."""
    pulled = np.zeros(len(score), bool); tie = rng.random(len(score))
    for c, k in kc.items():
        if k <= 0:
            continue
        idx = np.flatnonzero(cell == c)
        o = np.lexsort((tie[idx], -score[idx]))
        pulled[idx[o[:k]]] = True
    return pulled


def kept_usd(v5, w, pulled, sign=1.0):
    return sign * float((v5[~pulled] * w[~pulled]).sum()) / 100.0


def t3(D, rng, sign=1.0, ndraw=NDRAW):
    """Per day: value kept ($) for each arm at their own pull count per day x interval bucket; pooled per-share and recall rows."""
    out = {}; pooled = {a: dict(kv=0.0, kw=0.0, th=0, thp=0, bad=0, badp=0) for a in ('actual', 'guard', 'P', 'bound', 'random')}
    for d in sorted(D.day.unique()):
        x = D[D.day == d]; v5 = x.v5.values; w = x.vol_p.values; cell = x.b.values; gone = x.gone.values.astype(bool)
        kc = {c: int(gone[cell == c].sum()) for c in np.unique(cell)}
        arms = dict(actual=gone, guard=pick(-x.v.values, cell, kc, rng), P=pick(x.P.values, cell, kc, rng),
                    bound=pick(-(v5 * w), cell, kc, rng))
        for a, pl in arms.items():
            for c, k in kc.items():
                assert int(pl[cell == c].sum()) == k, f'{a}: pull count differs in a cell'
        th = x.through.values == 1; bd = v5 <= BAD
        rnd = np.empty(ndraw); pr = pooled['random']
        for j in range(ndraw):
            pl = pick(rng.random(len(v5)), cell, kc, rng); rnd[j] = kept_usd(v5, w, pl, sign)
            pr['kv'] += float((v5[~pl] * w[~pl]).sum()) / ndraw; pr['kw'] += float(w[~pl].sum()) / ndraw
            pr['thp'] += int((th & pl).sum()) / ndraw; pr['badp'] += int((bd & pl).sum()) / ndraw
        pr['th'] += int(th.sum()); pr['bad'] += int(bd.sum())
        r = {a: kept_usd(v5, w, pl, sign) for a, pl in arms.items()}
        r.update(random_mean=float(rnd.mean()), random_p95=float(np.quantile(rnd, 0.95)), n=len(x), pulls=int(gone.sum()))
        out[int(d)] = r
        for a, pl in arms.items():
            p_ = pooled[a]; p_['kv'] += float((v5[~pl] * w[~pl]).sum()); p_['kw'] += float(w[~pl].sum())
            p_['th'] += int(th.sum()); p_['thp'] += int((th & pl).sum()); p_['bad'] += int(bd.sum()); p_['badp'] += int((bd & pl).sum())
    return out, pooled


def bound_ok(per):
    return all(r['bound'] >= max(r['actual'], r['guard'], r['P'], r['random_p95']) - 1e-9 for r in per.values())


def readings(T, Dt3, rng, sign=1.0, ndraw=NDRAW):
    """The declared rules, evaluated. T = top-5 observations (all), Dt3 = T3's population."""
    R = {}
    b = T.b.values; wb = np.bincount(b, minlength=len(GAPB)) / len(b)
    rp = loss_ratio(T.gone.values, T.through.values, T.v5.values, b, wb)
    rd = {}
    for d in sorted(T.day.unique()):
        y = T[T.day == d]; bd = y.b.values; wbd = np.bincount(bd, minlength=len(GAPB)) / len(bd)
        rd[int(d)] = loss_ratio(y.gone.values, y.through.values, y.v5.values, bd, wbd)
    up = sum(r > 1 for r in rd.values()); dn = sum(r < 1 for r in rd.values())
    R['R557'] = dict(pooled=rp, days_up=up, days_down=dn, per_day=rd,
                     label='PULLS-TRACK-FILL-LOSS' if (rp >= RT and up >= DAYRULE) else
                     ('PULLS-AVOID-WINNERS' if (rp <= 1 / RT and dn >= DAYRULE) else 'NOT-SHOWN'))
    g, gk, sh = gap_share(T.v5.values, T.vol_p.values, T.gone.values, T.through.values)
    sd = {}
    for d in sorted(T.day.unique()):
        y = T[T.day == d]; sd[int(d)] = gap_share(y.v5.values, y.vol_p.values, y.gone.values, y.through.values)
    lo = sum(1 for (gd, _, s) in sd.values() if gd > 0 and s < 0.5); hi = sum(1 for (gd, _, s) in sd.values() if gd > 0 and s >= 0.5)
    R['R558'] = dict(gap=g, explained=gk, share=sh, days_within=lo, days_kind=hi, per_day={d: list(v) for d, v in sd.items()},
                     label='GAP-MOSTLY-WITHIN-KIND' if (g > 0 and sh < 0.5 and lo >= DAYRULE) else
                     ('GAP-MOSTLY-KIND' if (g > 0 and sh >= 0.5 and hi >= DAYRULE) else 'NOT-SHOWN'))
    per, pooled = t3(Dt3, rng, sign, ndraw)
    nullok = sum(r['actual'] > r['random_p95'] for r in per.values())
    R['R560'] = dict(days=nullok, label='PASS' if nullok >= DAYRULE else 'FAIL')
    for key, arm in (('R559', 'guard'), ('R559P', 'P')):
        th = sum(r['actual'] > r[arm] for r in per.values()); us = sum(r[arm] >= r['actual'] for r in per.values())
        lab = 'THEY-PICK-BETTER' if th >= DAYRULE else ('GUARD-PICKS-AS-WELL' if us >= DAYRULE else 'NOT-SHOWN')
        if key == 'R559P' and lab == 'GUARD-PICKS-AS-WELL':
            lab = 'P-PICKS-AS-WELL'
        R[key] = dict(days_theirs=th, days_ours=us, label=lab if R['R560']['label'] == 'PASS' else 'VOID')
    R['bound_ok'] = bound_ok(per)
    return R, per, pooled


def synth(kind, seed, n=6000):
    rng = np.random.default_rng(seed); parts = []
    for d in DAYS9:
        through = (rng.random(n) < 0.2).astype(int)
        v5 = np.round(np.where(through == 1, rng.normal(-3.0, 1.0, n), rng.normal(0.5, 1.0, n)) * 2) / 2
        v5[v5 == 0] = 0.5
        if kind == 'kind':
            pg = np.where(through == 1, 0.6, 0.2)
        else:
            pg = np.where(through == 1, 0.25, np.where(v5 < 0, 0.7, 0.1))
        gone = (rng.random(n) < pg).astype(int)
        parts.append(pd.DataFrame(dict(day=int(d), gap=rng.uniform(1.0, 3000.0, n), through=through, v5=v5, vol_p=rng.uniform(1.0, 50.0, n),
                                       gone=gone, v=rng.normal(-0.2, 0.3, n), P=rng.random(n))))
    D = pd.concat(parts, ignore_index=True); D['b'] = bucket(D.gap.values)
    return D


def selftest():
    ok = True
    for kind, want57, want58 in (('kind', 'NOT-SHOWN', 'GAP-MOSTLY-KIND'), ('loss', 'PULLS-TRACK-FILL-LOSS', 'GAP-MOSTLY-WITHIN-KIND')):
        D = synth(kind, 11 if kind == 'kind' else 12)
        R, per, _ = readings(D, D, np.random.default_rng(1), ndraw=40)
        good = R['R557']['label'] == want57 and R['R558']['label'] == want58 and R['bound_ok']
        print(f'selftest {kind}: R557 {R["R557"]["label"]} ({R["R557"]["pooled"]:.2f}), R558 {R["R558"]["label"]} '
              f'({R["R558"]["share"]:.2f}), bound {R["bound_ok"]} -> {"ok" if good else "FAIL"}')
        ok &= good
        Rf, _, _ = readings(D, D, np.random.default_rng(1), sign=-1.0, ndraw=40)
        caught = not Rf['bound_ok']
        print(f'selftest {kind}: planted value-sign flip caught: {caught}')
        ok &= caught
    b = bucket(np.array([0.0, 249.9, 250.0, 999.9, 1000.0, 3000.0, 3000.1]))
    good = b.tolist() == [0, 0, 1, 1, 2, 2, -1]
    print(f'selftest buckets: {b.tolist()} -> {"ok" if good else "FAIL"}'); ok &= good
    if not ok:
        sys.exit('REFUSED: selftest failed')
    print('SELFTEST PASS')


def report():
    os.makedirs(OUTT, exist_ok=True)
    A = pd.concat([pd.read_parquet(f'{LP.OUTD}/obs_{d}.parquet') for d in DAYS9], ignore_index=True)
    r2 = LP.K.rss_mib()
    if r2 > MEM_CAP_MIB:
        sys.exit(f'REFUSED: resident memory {r2:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    A['gone'] = 1 - A.stayed; A['b'] = bucket(A.gap.values)
    assert (A.b >= 0).all(), 'an interval outside 0-3000 ms'
    T = A[A.grp == 'top 5'].reset_index(drop=True); F = A[A.grp == 'field'].reset_index(drop=True)
    ok3 = np.isfinite(T.v.values) & np.isfinite(T.P.values); Dt3 = T[ok3].reset_index(drop=True)
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    rng = np.random.default_rng(SEED)
    R, per, pooled = readings(T, Dt3, rng)
    if not R['bound_ok']:
        sys.exit('REFUSED: the bound arm does not keep the most on some day (control)')
    L = [f'LEARN-PULL-TARGET (post hoc on LEARN-PULL\'s observations), 9 consumed days, first 60 s, WS-timed tape; as of {asof}; controls passed;'
         f' top 5 {len(T)} (field {len(F)}); v5 = 0 (no sign) top 5 {int((T.v5 == 0).sum())}; without v or P (out of T3) {int((~ok3).sum())};'
         f' in the last bucket at exactly 3000 ms {int((A.gap == 3000.0).sum())}', '']
    L.append('T1 GONE share by the next trade\'s kind x the fill\'s value sign (bucket-standardised; n). Ours = our guard\'s pull flag on the top 5\'s observations')
    for hz, col in (('5 s', 'v5'), ('1 s', 'v1')):
        for nm, X, ycol in (('top 5 GONE', T, 'gone'), ('field GONE', F, 'gone'), ('ours (we_pull)', T, 'we_pull')):
            b = X.b.values; wb = np.bincount(b, minlength=len(GAPB)) / len(b)
            c = t1_cells(X[ycol].values, X.through.values, X[col].values, b, wb)
            rat = loss_ratio(X[ycol].values, X.through.values, X[col].values, b, wb)
            L.append(f'  {hz} {nm:<15} ' + ' | '.join(f'{k}: {s:5.1%} ({n_})' for k, (s, n_) in c.items()) + f' | STOP loses / wins {rat:.2f}')
    L.append('  top 5 by interval bucket (5 s): GONE share STOP-loses / STOP-wins / THROUGH-loses / THROUGH-wins')
    for i, (bn, _, _) in enumerate(GAPB):
        y = T[T.b == i]; sg = np.sign(y.v5.values); th = y.through.values; g = y.gone.values
        f_ = lambda m: f'{g[m].mean():5.1%} ({int(m.sum())})' if m.any() else '  n/a'
        L.append(f'    {bn:<12} {f_((th == 0) & (sg < 0))} / {f_((th == 0) & (sg > 0))} / {f_((th == 1) & (sg < 0))} / {f_((th == 1) & (sg > 0))}')
    L.append('  R557 per day (STOP loses / wins, top 5): ' + ', '.join(f'{d % 10000:04d} {r:.2f}' for d, r in R['R557']['per_day'].items()))
    L.append(f'  R557: pooled {R["R557"]["pooled"]:.2f}, > 1 on {R["R557"]["days_up"]} of 9 -> {R["R557"]["label"]}')
    L.append('')
    L.append('T2 VALUE GAP (5 s, c/sh, volume-weighted): stayed - gone, and the part the trade\'s kind accounts for')
    for nm, X in (('top 5', T), ('field', F)):
        g, gk, sh = gap_share(X.v5.values, X.vol_p.values, X.gone.values, X.through.values)
        g2, gk2, sh2 = gap_share(X.v5.values, X.vol_p.values, X.gone.values, X.through.values * 10 + X.b.values)
        st_ = X.gone.values == 0
        L.append(f'  {nm:<6} stayed {vw(X.v5.values[st_], X.vol_p.values[st_]):+.2f}, gone {vw(X.v5.values[~st_], X.vol_p.values[~st_]):+.2f}: gap {g:+.2f};'
                 f' kind accounts for {gk:+.2f} ({sh:.0%}); kind x interval cells {gk2:+.2f} ({sh2:.0%})')
    for th, nm in ((0, 'STOP'), (1, 'THROUGH')):
        y = T[T.through == th]; st_ = y.gone.values == 0
        L.append(f'  top 5 within {nm:<7}: stayed {vw(y.v5.values[st_], y.vol_p.values[st_]):+.2f}, gone {vw(y.v5.values[~st_], y.vol_p.values[~st_]):+.2f}'
                 f' (n {len(y)}, gone {1 - st_.mean():.1%})')
    L.append('  R558 per day (gap, kind share): ' + ', '.join(f'{d % 10000:04d} {v[0]:+.2f} {v[2]:.0%}' for d, v in R['R558']['per_day'].items()))
    L.append(f'  R558: share {R["R558"]["share"]:.0%}, below half on {R["R558"]["days_within"]} of 9 -> {R["R558"]["label"]}')
    L.append('')
    L.append(f'T3 SELECTION AT THEIR OWN PULL COUNT (top 5, {len(Dt3)} observations, {int(Dt3.gone.sum())} pulls = {Dt3.gone.mean():.1%}; per day x interval bucket):')
    L.append('  arm      value kept $/day | c/sh kept | THROUGH trades among the pulled | bad (v5 <= -2) among the pulled')
    for a in ('actual', 'guard', 'P', 'random', 'bound'):
        p_ = pooled[a]; usd = sum(r[a if a != 'random' else 'random_mean'] for r in per.values()) / len(per)
        nm = dict(actual='theirs', guard='our guard v', P='P', random='random', bound='bound (v5)')[a]
        L.append(f'  {nm:<12} {usd:9.1f} | {p_["kv"] / p_["kw"]:+.2f} | {p_["thp"] / max(p_["th"], 1):5.1%} | {p_["badp"] / max(p_["bad"], 1):5.1%}')
    wp = Dt3.we_pull.values == 1; v5 = Dt3.v5.values; w = Dt3.vol_p.values; th = Dt3.through.values == 1; bd = v5 <= BAD
    L.append(f'  ours as run (we_pull, {wp.mean():.1%} pulled, not matched): c/sh kept {vw(v5[~wp], w[~wp]):+.2f} | THROUGH {(th & wp).sum() / max(th.sum(), 1):5.1%}'
             f' | bad {(bd & wp).sum() / max(bd.sum(), 1):5.1%}; value kept $/day {float((v5[~wp] * w[~wp]).sum()) / 100.0 / 9:.1f}')
    L.append('  per day value kept $ (theirs / guard / P / random p95 / bound): ' + '; '.join(
        f'{d % 10000:04d} {r["actual"]:.0f} / {r["guard"]:.0f} / {r["P"]:.0f} / {r["random_p95"]:.0f} / {r["bound"]:.0f}' for d, r in per.items()))
    L.append(f'  R560 null control: theirs above the random p95 on {R["R560"]["days"]} of 9 -> {R["R560"]["label"]}')
    L.append(f'  R559 guard: theirs higher on {R["R559"]["days_theirs"]}, ours higher or equal on {R["R559"]["days_ours"]} -> {R["R559"]["label"]}')
    L.append(f'  R559P P: theirs higher on {R["R559P"]["days_theirs"]}, P higher or equal on {R["R559P"]["days_ours"]} -> {R["R559P"]["label"]}')
    L.append('')
    lab = {k: R[k]['label'] for k in ('R557', 'R558', 'R559', 'R559P', 'R560')}
    L.append('READINGS: ' + json.dumps(lab))
    txt = '\n'.join(L) + '\n'
    open(f'{OUTT}/learn_pull_target.txt', 'w').write(txt)
    json.dump(dict(asof=asof, readings=R, per_day_t3=per, pooled_t3=pooled), open(f'{OUTT}/learn_pull_target.json', 'w'), indent=1, default=float)
    print(txt)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        selftest()
    elif cmd == 'report':
        selftest()
        report()
    else:
        sys.exit('usage: pf_learn_pull_target.py selftest | report')
