"""B12d (2026-11-11, USER "Run it, and give me a clear explanation"): how the top makers get fills whose Predict price
holds. B12c found their edge is earned within 250 ms of the trade report and is not Binance. Three candidate
mechanisms, all on Predict's own data: (1) they react to Predict trades (pull right after being hit, before the
follow-up taker orders); (2) they read the book (avoid trades that come after the level has thinned); (3) their size
holds the price (they stay, and the level survives their fills). Read-only, descriptive. Same population as B12b
part A (pf_b12b_reason.py c882da77): at-best legs of the 220 consumed 11-08 cohort windows, first minute, guard ON /
OFF at lag 50, trades that do not go through the level; the top 5 (T) against ranks >= 6 (O). G = 1 day < 5: point
estimates only.

Measures (research trade time anc, one per taker order; venue matching is price-time, B12b R42):
 runs      taker orders on a (window, side) sorted by anc; an order within 250 ms of the previous one on that side
           continues a run. Position 1 = first in its run, 2, 3+. Follow-ups = taker orders on that side in (anc,
           anc + 500 ms].
 re-hit    an ABSORBED fill = the only maker leg of its taker order at that price (the maker took the whole trade there),
           at the best, not swept. Its opportunity = the next taker order (in anc order) that trades at the same
           price on that side within 500 ms. Under price-time matching a maker who absorbed a trade was at the front
           with size >= the trade, so if it is still there it is filled first by that next order: re-hit = the same
           maker has a leg in it. No re-hit = it cancelled or repriced, or its order was used up exactly.
 thinning  shown size at the trade's price in the frame strictly before anc (Q) against the last frame <= anc - 250
           ms (Q250): new (absent then), shrank (Q / Q250 <= 0.8), stable, grew (>= 1.25); no earlier frame: status.
 holds     the side's best price in the last frame <= anc + 250 ms is the trade's price or better for the maker.
Decompositions as B12b (D = MIX + WITHIN over cells; X = (WITHIN_G - WITHIN_cells) / WITHIN_G, value at 1 s): RUN
cells = guard x position (6); THIN cells = guard x thinning class (8). HOLD: the within-cell gap of the holds
indicator over guard x reach-bucket cells (B12b's reach buckets), in percentage points.

Readings DECLARED BEFORE THE FIRST RUN (evaluated in code):
 R60 runs      share of the group's fill shares that are first in their run: FIRST-HEAVY if the top 5's >= O's + 10
               pp; FOLLOWUP-HEAVY if <= O's - 10 pp; else SAME-POSITION. X_RUN reported.
 R61 book      BOOK-EXPLAINS if X_THIN >= 0.5; else BOOK-NOT-ENOUGH.
 R62 re-hit    rT, rO = re-hit rates over opportunities (n/a under 100 opportunities in either group).
               PULL-AFTER-HIT if rT <= 0.8 rO and rT - rO is below its null p01 (N6); STAY-AFTER-HIT if rT >= 1.25 rO
               and above the null p99; else SAME-STAY.
 R63 holds     HOLDS-MORE if the top 5's within-cell holds gap >= +10 pp; else SAME-HOLD.
 R64 mechanism every label that applies of REACT-TO-TRADES (R62 PULL-AFTER-HIT), HOLD-THE-PRICE (R62 STAY-AFTER-HIT or
               R63 HOLDS-MORE), READ-THE-BOOK (R61 BOOK-EXPLAINS); none: UNRESOLVED.
 Null N6: 500 permutations of the T / O label among re-hit opportunities within (window, side), seed 20261111.
What the outcomes mean (fixed in advance): REACT-TO-TRADES = after being hit they are gone before the next taker order
more often than others, despite larger orders: they pull right after a fill. HOLD-THE-PRICE = they stay (and the level
survives) more than others at the same reach: their size props up the price their fills are marked at. READ-THE-BOOK =
their edge is accounted for by avoiding trades that come after the level thinned. Descriptive: re-hit rates by the time
to the next order (0-100, 100-250, 250-500 ms) show how fast anyone leaves; values of absorbed fills by stay / leave.

Controls (any failure -> no table): B12b's controls (which include B12's); K0 the rebuilt frame reproduces b12.json
and B12b's part A legs (b12b.json hash checked); D1 run positions and follow-up counts on planted order times (ties
continue a run); D2 re-hit on planted legs (same maker on the next same-price order -> 1; another maker -> 0; next
order at another price, or after 500 ms -> no opportunity; a leg sharing its order with another maker at that price ->
not absorbed); D3 thinning classes and holds on planted frames (bid and ask); D4 the null flags a planted halved re-hit
rate and not equal rates; D5 the label rules; a NaN reads as n/a.
Limits fixed in advance: one consumed Sunday; trade times are reports (a few taker orders can share one time); the
shown book is a ~50 ms snapshot; no re-hit cannot tell a cancel from an order used up exactly; holds after a fill is
partly the outcome itself (it says whether the level survived, not who refilled it); values are of the flow the makers
received.
Usage: predictfun/research/run_light.sh b12d_bursts -- python3 -m predictfun.research.pf_b12d_bursts <out_dir>
       python3 -m predictfun.research.pf_b12d_bursts --selftest
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

RUN_GAP = 250.0; FOLLOW = 500.0; NEXT = 500.0; LOOK = 250.0
GAPS = ((0.0, 100.0), (100.0, 250.0), (250.0, 500.0))
POS_LAB = ['first', 'second', 'third+']; THIN_LAB = ['new', 'shrank', 'stable', 'grew']
N_NULL = 500; SEED = 20261111; MIN_OPP = 100
NAN = float('nan')


# ---- instruments ----
def run_pos(anc, gap=RUN_GAP):
    """Position in run (1 = first) for time-sorted order times; an order within `gap` ms of the previous continues it."""
    pos = np.ones(len(anc), np.int64)
    for i in range(1, len(anc)):
        if anc[i] - anc[i - 1] <= gap:
            pos[i] = pos[i - 1] + 1
    return pos


def follow_count(anc, w=FOLLOW):
    """Orders strictly after each one, within w ms (time-sorted input)."""
    return (np.searchsorted(anc, anc + w, side='right') - np.searchsorted(anc, anc, side='right')).astype(np.int64)


def rehit(opp_key, opp_anc, opp_grp, opp_signer, nxt):
    """For absorbed legs: (has opportunity, gap ms, re-hit). nxt maps (mid, side, price) -> (times, grps, signer sets) of
    taker orders trading at that price, sorted by (time, grp)."""
    has = np.zeros(len(opp_anc), bool); gap = np.full(len(opp_anc), np.nan); hit = np.zeros(len(opp_anc), bool)
    for i, (k, a, g, s) in enumerate(zip(opp_key, opp_anc, opp_grp, opp_signer)):
        if k not in nxt:
            continue
        ts, gs, ss = nxt[k]
        j = np.searchsorted(ts, a, side='left')
        while j < len(ts) and (ts[j] < a or (ts[j] == a and gs[j] <= g)):
            j += 1
        if j < len(ts) and ts[j] - a <= NEXT:
            has[i] = True; gap[i] = ts[j] - a; hit[i] = s in ss[j]
    return has, gap, hit


def size_at(Pk, Sk, j, p):
    """Shown size at price p in frame j (0 if not among the shown levels); -1 if no frame."""
    if j < 0:
        return -1.0
    m = Pk[j] == p
    return float(Sk[j][m].sum()) if m.any() else 0.0


def thin_class(q, q250):
    if q250 < 0:
        return 'no earlier frame'
    if q250 == 0:
        return 'new'
    r = q / q250
    return 'shrank' if r <= 0.8 else ('grew' if r >= 1.25 else 'stable')


def holds(best_after, p, side):
    if best_after < 0:
        return False
    return best_after == p or (best_after > p if side == 0 else best_after < p)


def rate_delta(lab, hit):
    a = lab.sum(); b = (~lab).sum()
    if a == 0 or b == 0:
        return NAN, NAN, NAN
    rT = hit[lab].mean(); rO = hit[~lab].mean()
    return rT - rO, rT, rO


def null_rate(lab, hit, block, draws, seed):
    rng = np.random.default_rng(seed); o = np.argsort(block, kind='stable'); res = np.empty(draws)
    for d in range(draws):
        o2 = np.lexsort((rng.random(len(block)), block)); l2 = np.empty_like(lab); l2[o] = lab[o2]
        res[d] = rate_delta(l2, hit)[0]
    return res


# ---- labels ----
def label_r60(fT, fO):
    if not BB.fin(fT, fO):
        return 'n/a'
    return 'FIRST-HEAVY' if fT >= fO + 0.10 else ('FOLLOWUP-HEAVY' if fT <= fO - 0.10 else 'SAME-POSITION')


def label_r61(x):
    return 'n/a' if not BB.fin(x) else ('BOOK-EXPLAINS' if x >= 0.5 else 'BOOK-NOT-ENOUGH')


def label_r62(rT, rO, d, p01, p99, nT, nO):
    if nT < MIN_OPP or nO < MIN_OPP or not BB.fin(rT, rO, d, p01, p99) or rO <= 0:
        return 'n/a'
    if rT <= 0.8 * rO and d < p01:
        return 'PULL-AFTER-HIT'
    if rT >= 1.25 * rO and d > p99:
        return 'STAY-AFTER-HIT'
    return 'SAME-STAY'


def label_r63(w):
    return 'n/a' if not BB.fin(w) else ('HOLDS-MORE' if w >= 0.10 else 'SAME-HOLD')


def label_r64(r61, r62, r63):
    lab = []
    if r62 == 'PULL-AFTER-HIT':
        lab.append('REACT-TO-TRADES')
    if r62 == 'STAY-AFTER-HIT' or r63 == 'HOLDS-MORE':
        lab.append('HOLD-THE-PRICE')
    if r61 == 'BOOK-EXPLAINS':
        lab.append('READ-THE-BOOK')
    return lab or ['UNRESOLVED']


# ---- controls ----
def controls():
    rep = {}; ok = True

    def chk(name, cond):
        nonlocal ok
        rep[name] = bool(cond); ok = ok and bool(cond)
    okb, repb = BB.controls(); chk('B12b_controls', okb)
    t = np.array([0.0, 100.0, 100.0, 600.0, 700.0, 1200.0])
    chk('D1_runs', list(run_pos(t)) == [1, 2, 3, 1, 2, 1] and list(follow_count(t)) == [2, 1, 1, 1, 1, 0])
    # D2 re-hit: key (1, 0, 50): orders at 0 (grp 10, maker A absorbed), 200 (grp 11: A and B), 800 (grp 12: A)
    nxt = {(1, 0, 50): (np.array([0.0, 200.0, 800.0]), np.array([10, 11, 12]), [{'A'}, {'A', 'B'}, {'A'}]),
           (1, 0, 49): (np.array([300.0]), np.array([20]), [{'C'}])}
    h, g, x = rehit([(1, 0, 50), (1, 0, 50), (1, 0, 50), (1, 0, 49), (1, 0, 48), (1, 0, 50)], np.array([0.0, 0.0, 800.0, 300.0, 5.0, 200.0]),
                    np.array([10, 10, 12, 20, 30, 11]), ['A', 'Z', 'A', 'C', 'A', 'A'], nxt)
    d2 = list(h) == [True, True, False, False, False, False] and list(x[:2]) == [True, False] and g[0] == 200.0
    # a leg sharing its taker order with another maker at that price is not absorbed (n_lvl == 2)
    s_, n_ = BB.taker_level(np.array([5, 5, 6]), np.array([0, 0, 0]), np.array([50, 50, 50]), np.array([1.0, 2.0, 3.0]))
    chk('D2_rehit', d2 and list(n_) == [2, 2, 1])
    # D3 thinning and holds
    Pk = np.array([[50, 49, 48, 47, 46], [50, 49, 48, 47, 46]]); Sk = np.array([[100.0, 5, 5, 5, 5], [60.0, 5, 5, 5, 5]])
    d3 = size_at(Pk, Sk, 0, 50) == 100.0 and size_at(Pk, Sk, 0, 45) == 0.0 and size_at(Pk, Sk, -1, 50) == -1.0
    d3 = d3 and thin_class(60.0, 100.0) == 'shrank' and thin_class(100.0, 100.0) == 'stable' and thin_class(130.0, 100.0) == 'grew'
    d3 = d3 and thin_class(10.0, 0.0) == 'new' and thin_class(10.0, -1.0) == 'no earlier frame'
    d3 = d3 and holds(50, 50, 0) and holds(51, 50, 0) and not holds(49, 50, 0) and holds(49, 50, 1) and not holds(51, 50, 1) and not holds(-1, 50, 0)
    chk('D3_thin_holds', d3)
    # D4 null on planted re-hit rates
    rng = np.random.default_rng(13); n = 4000; lab = rng.random(n) < 0.3; blk = rng.integers(0, 50, n); res = []
    for half in (True, False):
        pr = np.where(lab & half, 0.25, 0.5); hit = rng.random(n) < pr
        dl, rT, rO = rate_delta(lab, hit); nul = null_rate(lab, hit, blk, 200, 3)
        res.append(label_r62(rT, rO, dl, np.percentile(nul, 1), np.percentile(nul, 99), int(lab.sum()), int((~lab).sum())))
    chk('D4_null', res == ['PULL-AFTER-HIT', 'SAME-STAY'])
    chk('D5_rules', label_r60(0.5, 0.35) == 'FIRST-HEAVY' and label_r60(0.3, 0.45) == 'FOLLOWUP-HEAVY' and label_r60(NAN, 0.3) == 'n/a'
        and label_r61(0.6) == 'BOOK-EXPLAINS' and label_r61(0.1) == 'BOOK-NOT-ENOUGH' and label_r63(0.12) == 'HOLDS-MORE' and label_r63(NAN) == 'n/a'
        and label_r62(0.2, 0.5, -0.3, -0.05, 0.05, 50, 500) == 'n/a' and label_r62(0.7, 0.5, 0.2, -0.05, 0.05, 500, 500) == 'STAY-AFTER-HIT'
        and label_r64('BOOK-NOT-ENOUGH', 'PULL-AFTER-HIT', 'HOLDS-MORE') == ['REACT-TO-TRADES', 'HOLD-THE-PRICE']
        and label_r64('BOOK-NOT-ENOUGH', 'SAME-STAY', 'SAME-HOLD') == ['UNRESOLVED'])
    rep['B12b_controls_detail'] = repb
    return ok, rep


# ---- data ----
def frames(W0):
    cols = ['mid', 't_ms'] + [f'{s}{i}_{x}' for s in 'ba' for i in range(1, 6) for x in 'ps']; parts = []
    for fn in sorted(glob.glob(f'{BB.S1}/books/part_*.parquet')):
        d = pd.read_parquet(fn, columns=cols); parts.append(d[d.mid.isin(W0)])
    Bk = pd.concat(parts, ignore_index=True); del parts
    Bk.sort_values(['mid', 't_ms'], inplace=True, kind='stable'); out = {}
    for m, b in Bk.groupby('mid', sort=False):
        tf = b.t_ms.values.astype(float)
        out[m] = (tf, {s: (b[[f"{'b' if s == 0 else 'a'}{i}_p" for i in range(1, 6)]].values.astype('int64'),
                           b[[f"{'b' if s == 0 else 'a'}{i}_s" for i in range(1, 6)]].values.astype(float)) for s in (0, 1)})
    return out


def main():
    if '--selftest' in sys.argv:
        ok, rep = controls(); print(json.dumps({k: v for k, v in rep.items() if not k.endswith('detail')}, indent=1)); print('selftest', 'PASS' if ok else 'FAIL')
        sys.exit(0 if ok else 1)
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    ok, ctl = controls()
    if not ok:
        sys.exit(f'CONTROL FAILED -> no table: {ctl}')
    sums = dict(line.split()[::-1] for line in open(C1.REV) if line.strip())
    if hashlib.sha256(open(f'{C1.B12B_OUT}/b12b.json', 'rb').read()).hexdigest() != sums['b12b.json']:
        sys.exit('REFUSED: b12b.json differs from the committed copy')
    b12b = json.load(open(f'{C1.B12B_OUT}/b12b.json'))
    X = BB.build(); R = X['R']; W0 = X['W0']; days = X['days']
    k0 = BB.reconcile(R, days)
    V = R[R.val_ok]
    PA = (V.status == 'ok') & (V.depth == 0) & V.g50.isin([B.G_ON, B.G_OFF]) & (V.sweep == 0)
    k0['partA_legs'] = dict(top5=int((PA & (V.g == 'top 5')).sum()), O=int((PA & V.g.isin(['6-20', 'rest'])).sum()))
    k0['partA_ok'] = (k0['partA_legs']['top5'] == b12b['A']['groups']['top 5']['legs'] and k0['partA_legs']['O'] == b12b['A']['groups']['O (ranks >= 6)']['legs'])
    if not (k0['ok'] and k0['partA_ok']):
        sys.exit(f'CONTROL FAILED (K0) -> no table: {k0}')
    # taker orders of the first minute, runs and follow-ups per (window, side)
    TO = R.groupby('grp', sort=False).agg(mid=('mid', 'first'), side=('side', 'first'), anc=('anc', 'first')).reset_index()
    TO = TO.sort_values(['mid', 'side', 'anc', 'grp'], kind='stable').reset_index(drop=True)
    pos = np.zeros(len(TO), np.int64); fol = np.zeros(len(TO), np.int64)
    for _, ii in TO.groupby(['mid', 'side'], sort=False).indices.items():
        a = TO.anc.values[ii]; pos[ii] = run_pos(a); fol[ii] = follow_count(a)
    TO['pos'] = pos; TO['fol'] = fol
    P = TO.set_index('grp')
    R['pos'] = P.pos.reindex(R.grp.values).values; R['fol'] = P.fol.reindex(R.grp.values).values
    # frames: thinning before and holds after, for at-best legs
    FR = frames(W0); q250 = np.full(len(R), np.nan); hold = np.zeros(len(R), bool)
    for (m, side), ii in R.groupby(['mid', 'side'], sort=False).indices.items():
        tf, S = FR[m]; Pk, Sk = S[side]; ta = R.anc.values[ii]; pt = R.ptick.values[ii].astype('int64')
        j1 = np.searchsorted(tf, ta - LOOK, side='right') - 1; j2 = np.searchsorted(tf, ta + LOOK, side='right') - 1
        q250[ii] = [size_at(Pk, Sk, j, p) for j, p in zip(j1, pt)]
        hold[ii] = [holds(Pk[j, 0] if j >= 0 else -1, p, side) for j, p in zip(j2, pt)]
    R['q250'] = q250; R['hold'] = hold
    R['thin'] = [thin_class(q, r) if np.isfinite(q) else 'n/a' for q, r in zip(R.Q.values, R.q250.values)]
    V = R[R.val_ok]; A = V[(V.status == 'ok') & (V.depth == 0) & V.g50.isin([B.G_ON, B.G_OFF]) & (V.sweep == 0)].copy()
    A['off'] = (A.g50.values == B.G_OFF).astype(np.int64)
    A['pcls'] = np.minimum(A.pos.values, 3) - 1
    tmap = {k: i for i, k in enumerate(THIN_LAB)}; A['tcls'] = A.thin.map(tmap).fillna(-1).astype(np.int64)
    A['rb'] = BB.bucket(A.s_lvl.values / A.Q.values, BB.R_EDGES)
    D = A[A.g.isin(['top 5', '6-20', 'rest'])]; isT = D.g.values == 'top 5'; sh = D.shares.values; v1 = D.v1.values
    Wg = BB.decompose_n(D.off.values, isT, sh, v1, 2)['WITHIN']
    dr = BB.decompose_n(3 * D.off.values + D.pcls.values, isT, sh, v1, 6)
    kt = D.tcls.values >= 0; Wg_t = BB.decompose_n(D.off.values[kt], isT[kt], sh[kt], v1[kt], 2)['WITHIN']
    dt = BB.decompose_n(4 * D.off.values[kt] + D.tcls.values[kt], isT[kt], sh[kt], v1[kt], 8)
    dh = BB.decompose_n(5 * D.off.values + D.rb.values, isT, sh, D.hold.values.astype(float), 10)
    res = dict(as_of_note='point estimates, one consumed partial day (220 windows of 2026-11-08, 0.764 window-days)', controls=ctl, K0=k0,
               runs=dict(taker_orders=int(len(TO)), share_in_runs=float((TO.pos > 1).mean())), groups={})
    for g in BB.GR:
        x = BB.sel(A, g); s_ = x.shares.values; tot = s_.sum()
        row = dict(legs=int(len(x)), shares=float(tot), net1=B.wmean(s_, x.v1.values), by_position={}, by_thinning={}, followed=None,
                   hold_share=float(s_[x.hold.values].sum() / tot) if tot else NAN)
        for i, lab in enumerate(POS_LAB):
            mk = x.pcls.values == i
            row['by_position'][lab] = dict(share=float(s_[mk].sum() / tot) if tot else NAN, net1=B.wmean(s_[mk], x.v1.values[mk]))
        for lab in THIN_LAB + ['no earlier frame']:
            mk = x.thin.values == lab
            row['by_thinning'][lab] = dict(share=float(s_[mk].sum() / tot) if tot else NAN, net1=B.wmean(s_[mk], x.v1.values[mk]))
        fm = x.fol.values > 0
        row['followed'] = dict(share=float(s_[fm].sum() / tot) if tot else NAN, net1_followed=B.wmean(s_[fm], x.v1.values[fm]),
                               net1_alone=B.wmean(s_[~fm], x.v1.values[~fm]))
        res['groups'][g] = row
    fT = res['groups']['top 5']['by_position']['first']['share']; fO = res['groups']['O (ranks >= 6)']['by_position']['first']['share']
    res['decomposition'] = dict(WITHIN_G=Wg, RUN=dict(D=dr['D'], MIX=dr['MIX'], WITHIN=dr['WITHIN'], X=(Wg - dr['WITHIN']) / Wg),
                                THIN=dict(WITHIN_G=Wg_t, D=dt['D'], MIX=dt['MIX'], WITHIN=dt['WITHIN'], X=(Wg_t - dt['WITHIN']) / Wg_t,
                                          excluded_no_frame_legs=int((~kt).sum())),
                                HOLD=dict(D_pp=dh['D'], MIX_pp=dh['MIX'], WITHIN_pp=dh['WITHIN']))
    # re-hit after an absorbed fill (all guard states), per maker
    AB = V[(V.status == 'ok') & (V.depth == 0) & (V.sweep == 0) & (V.n_lvl == 1)].copy()
    gb = R[R.anc.notna()].groupby(['mid', 'side', 'ptick', 'grp'], sort=False)
    lv = gb.anc.first().to_frame('anc').join(gb.signer.apply(lambda s: frozenset(int(v) for v in s)).rename('signers')).reset_index()
    lv = lv.sort_values(['mid', 'side', 'ptick', 'anc', 'grp'], kind='stable'); nxt = {}
    for k, ii in lv.groupby(['mid', 'side', 'ptick'], sort=False).indices.items():
        nxt[(int(k[0]), int(k[1]), int(k[2]))] = (lv.anc.values[ii], lv.grp.values[ii], list(lv.signers.values[ii]))
    keys = [(int(a), int(b), int(c)) for a, b, c in zip(AB.mid.values, AB.side.values, AB.ptick.values)]
    has, gap, hit = rehit(keys, AB.anc.values, AB.grp.values, [int(s) for s in AB.signer.values], nxt)
    AB['has'] = has; AB['gap'] = gap; AB['hit'] = hit
    OP = AB[AB.has & AB.g.isin(['top 5', '6-20', 'rest'])]; labT = OP.g.values == 'top 5'; hv = OP.hit.values
    dl, rT, rO = rate_delta(labT, hv)
    nul = null_rate(labT, hv, pd.factorize(pd.Series(list(zip(OP.mid.values, OP.side.values))))[0], N_NULL, SEED)
    p01, p99 = float(np.percentile(nul, 1)), float(np.percentile(nul, 99))
    rh = dict(absorbed_legs=int(len(AB)), opportunities=int(AB.has.sum()), rT=rT, rO=rO, delta=dl, null_p01=p01, null_p50=float(np.percentile(nul, 50)), null_p99=p99,
              nT=int(labT.sum()), nO=int((~labT).sum()), by_gap={}, by_group={}, value_by_stay={})
    for a_, b_ in GAPS:
        mk = (OP.gap.values >= a_) & (OP.gap.values <= b_) if a_ == 0 else (OP.gap.values > a_) & (OP.gap.values <= b_)
        rh['by_gap'][f'{int(a_)}-{int(b_)} ms'] = dict(nT=int((mk & labT).sum()), rT=float(hv[mk & labT].mean()) if (mk & labT).any() else NAN,
                                                       nO=int((mk & ~labT).sum()), rO=float(hv[mk & ~labT].mean()) if (mk & ~labT).any() else NAN)
    for g in BB.GR:
        x = AB[AB.has] if g != 'O (ranks >= 6)' else AB[AB.has & AB.g.isin(['6-20', 'rest'])]
        x = x[x.g == g] if g != 'O (ranks >= 6)' else x
        rh['by_group'][g] = dict(n=int(len(x)), rate=float(x.hit.mean()) if len(x) else NAN)
        for st, mk in (('stayed', x.hit.values), ('left', ~x.hit.values)):
            rh['value_by_stay'][f'{g}, {st}'] = dict(legs=int(mk.sum()), net1=B.wmean(x.shares.values[mk], x.v1.values[mk]))
    res['rehit'] = rh
    r60 = label_r60(fT, fO); r61 = label_r61(res['decomposition']['THIN']['X'])
    r62 = label_r62(rT, rO, dl, p01, p99, rh['nT'], rh['nO']); r63 = label_r63(dh['WITHIN'])
    res['readings'] = dict(R60=r60, R61=r61, R62=r62, R63=r63, R64=label_r64(r61, r62, r63))
    json.dump(res, open(f'{out}/b12d.json', 'w'), indent=1, default=float)
    f = lambda v: 'n/a' if not BB.fin(v) else f'{v:+.2f}'
    pc = lambda v: 'n/a' if not BB.fin(v) else f'{v * 100:.1f}%'
    Lt = [f"B12d bursts, re-hits, thinning, holds. {res['as_of_note']}", f"Controls: {json.dumps({k: v for k, v in ctl.items() if not k.endswith('detail')})}",
          f"K0 {k0['ok']} part A {k0['partA_legs']} {k0['partA_ok']}; taker orders {res['runs']}", '',
          'Part A legs by group: legs | net 1 s | first / second / third+ in run (share, net) | followed within 0.5 s (share; net followed / alone) | level holds 250 ms later']
    for g, x in res['groups'].items():
        bp = x['by_position']; fw = x['followed']
        Lt.append(f"{g} | {x['legs']} | {f(x['net1'])} | " + ' / '.join(f"{pc(bp[k]['share'])} {f(bp[k]['net1'])}" for k in POS_LAB)
                  + f" | {pc(fw['share'])}; {f(fw['net1_followed'])} / {f(fw['net1_alone'])} | {pc(x['hold_share'])}")
    Lt.append('Thinning before the trade (share, net 1 s): ' + ' || '.join(
        f"{g}: " + '; '.join(f"{k} {pc(v['share'])} {f(v['net1'])}" for k, v in x['by_thinning'].items()) for g, x in res['groups'].items() if g in ('top 5', 'O (ranks >= 6)', 'ours')))
    dc = res['decomposition']
    Lt += [f"Decomposition top 5 against O (c/sh at 1 s): WITHIN_G {f(dc['WITHIN_G'])}; RUN WITHIN {f(dc['RUN']['WITHIN'])} X {dc['RUN']['X']:.2f}; "
           f"THIN WITHIN {f(dc['THIN']['WITHIN'])} X {dc['THIN']['X']:.2f} (of {f(dc['THIN']['WITHIN_G'])}); HOLD within reach {dc['HOLD']['WITHIN_pp'] * 100:+.1f} pp", '',
           f"Re-hit after an absorbed fill: absorbed {rh['absorbed_legs']}, with a next same-price order within 0.5 s {rh['opportunities']}; "
           f"top 5 {pc(rh['rT'])} (n {rh['nT']}) against O {pc(rh['rO'])} (n {rh['nO']}); delta {rh['delta'] * 100:+.1f} pp, null p01 {p01 * 100:+.1f} p99 {p99 * 100:+.1f} pp",
           '  by time to the next order: ' + '; '.join(f"{k}: top 5 {pc(v['rT'])} (n {v['nT']}), O {pc(v['rO'])} (n {v['nO']})" for k, v in rh['by_gap'].items()),
           '  by group: ' + '; '.join(f"{k} {pc(v['rate'])} (n {v['n']})" for k, v in rh['by_group'].items()),
           '  value of the absorbed fill: ' + '; '.join(f"{k} {f(v['net1'])} ({v['legs']})" for k, v in rh['value_by_stay'].items()), '',
           'Readings: ' + json.dumps(res['readings'])]
    open(f'{out}/b12d.txt', 'w').write('\n'.join(Lt) + '\n')
    print('\n'.join(Lt))


if __name__ == '__main__':
    main()
