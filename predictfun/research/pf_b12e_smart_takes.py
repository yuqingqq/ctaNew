"""B12e (2026-11-11, USER "can you check their take, what happens on Binance or what events happen during then?"):
what is special about the moments when the top makers take. B12d (post hoc) found that three of the top 5 makers
(48, 51, 661) also send taker orders, and that the makers they hit lose -1.23 c/sh at 1 s. Read-only, descriptive.
220 consumed 11-08 cohort windows, first minute; G = 1 day < 5: point estimates only.

Events. SMART = taker orders sent by a top-5 maker's address (address from the chain match records joined to the legs as
in pf_b12d_takers.py: unambiguous keys only, modal address per signer; our id must map to our address) that fill at
least one leg at the best (depth 0) in the first minute; one row per taker order: (window, side hit, t = anc). OTHER =
taker orders at the best with a known taker that is not such an address. RANDOM = for each SMART event, 5 times drawn
uniformly in [ws + 1.05 s, ws + 59 s] of the same window, same side (seed 20261111).
Features at t (signed so that + is in the taker's favour: a taker hitting bids gains if the price falls, so sign = -1
for bids, +1 for asks):
  Binance (step5 rows_rl_d2 bn_mid, 50 ms grid, latest point <= 100 ms old): returns in bps over (t - 1050, t - 50]
     (before), (t - 300, t - 50] (just before), (t - 50, t + 500] and (t - 50, t + 2000] (after).
  Predict mid (research frames, two-sided mid forward-filled, last frame at or before): change in c over (t - 1050,
     t - 50] (before) and from the frame before t to t + 250 / t + 1000 (after).
  Our model: pred_1000 (preds_t60k, latest grid point <= t - 50, fresh), signed; and the guard value v of the side hit
     (maker sign: lower = worse for that side's makers).
  Book (frame strictly before t): spread (ticks), size at the best of the side hit, size at the best of the other side.
  Timing: seconds into the window; for SMART, the share that follow another SMART order on that side within 250 ms.
Statistics: means (and the share > 0 for signed returns) for SMART, OTHER, RANDOM; null: 500 permutations of the SMART /
RANDOM label within window (seed 20261111), one-sided p for SMART - RANDOM.

Readings DECLARED BEFORE THE FIRST RUN (evaluated in code):
 R70 binance lead   BINANCE-LEAD if mean Binance (before) at SMART minus RANDOM >= +0.3 bps and above its null p99;
                    else NO-BINANCE-LEAD.
 R71 predict lag    only under BINANCE-LEAD: PREDICT-STALE if the mean Predict move (before) at SMART minus RANDOM <= +0.1
                    c (Binance moved their way and Predict had not followed); else PREDICT-MOVED-TOO. n/a otherwise.
 R72 our model      only if INSTRUMENT-OK (the correlation of pred_1000 with the realised Predict mid change over the next
                    1 s at RANDOM times is >= 0.05): the share of SMART events whose signed pred_1000 is above the 90th
                    percentile of RANDOM's: OUR-MODEL-SEES if >= 0.30; OUR-MODEL-MISSES if <= 0.15; else PARTLY.
 R73 book           THIN-SIDE if the median size at the best of the side hit at SMART <= 0.5 x RANDOM's; WIDE if the mean
                    spread at SMART >= 1.5 x RANDOM's; NORMAL-BOOK if neither.
 R74 after          BINANCE-FOLLOWS if mean Binance (t - 50, t + 2000] at SMART minus RANDOM >= +0.3 bps and above its
                    null p99 (Binance moves their way after they take: they anticipate it); else NO-BINANCE-FOLLOW.
 R75 summary        the labels together.
What the outcomes mean (fixed in advance): BINANCE-LEAD + PREDICT-STALE = latency arbitrage: they take after Binance
moves before Predict's quotes follow; the defence is to reprice on Binance at least as fast. NO-BINANCE-LEAD = their
trigger is not a Binance move we record; with BINANCE-FOLLOWS they anticipate Binance (a better model or an earlier
feed); with neither, their trigger is elsewhere (Predict itself, another venue or the settlement feed).
OUR-MODEL-SEES = our own model flags those moments too: our problem is acting, not seeing.

Controls (any failure -> no table): B12b's controls (which include B12's); K0 b12.json reproduced; KJ the address join
(our id holds our address from who.json); E1 the event statistics on planted events (a planted +1 bps lead ->
BINANCE-LEAD, none -> NO-BINANCE-LEAD, through the same permutation null); E2 the taker sign on planted bid and ask
events; E3 the label rules; a NaN reads as n/a. INSTRUMENT check for R72 on real data as stated.
Limits fixed in advance: one consumed Sunday; trade times are reports (the match is ~0.2 s earlier), so 'before' includes
about 0.2 s after the match; the Binance grid is the research collector's clock; three addresses only; the book is a
~50 ms snapshot.
Usage: predictfun/research/run_light.sh b12e_smart_takes -- python3 -m predictfun.research.pf_b12e_smart_takes <out_dir>
       python3 -m predictfun.research.pf_b12e_smart_takes --selftest
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from . import pf_b12_makers as B
from . import pf_b12b_reason as BB
from . import pf_b12c_model_or_speed as C1
from . import pf_b12c2_binance as C2
from . import pf_b12d_bursts as DB

MATCH = f'{BB.LV}/matches_20261108.parquet'
WHO = f'{BB.LV}/who.json'
PREDS = f'{BB.RP}/research/step5/20261108/preds_t60k.parquet'
N_NULL = 500; SEED = 20261111; K_RAND = 5
NAN = float('nan')
FEAT_BN = {'bn_before': (-1050.0, -50.0), 'bn_just_before': (-300.0, -50.0), 'bn_after_500': (-50.0, 500.0), 'bn_after_2000': (-50.0, 2000.0)}


def tsign(side):
    """+1 when the taker gains from a rise: a taker hitting asks (side 1) buys; hitting bids (side 0) sells."""
    return np.where(np.asarray(side) == 0, -1.0, 1.0)


def diff_null(x, lab, block, draws, seed):
    """SMART - RANDOM mean difference, observed and under the label permuted within blocks."""
    ok = np.isfinite(x); x, lab, block = x[ok], lab[ok], block[ok]
    obs = x[lab].mean() - x[~lab].mean() if lab.any() and (~lab).any() else NAN
    rng = np.random.default_rng(seed); o = np.argsort(block, kind='stable'); res = np.empty(draws)
    for d in range(draws):
        o2 = np.lexsort((rng.random(len(block)), block)); l2 = np.empty_like(lab); l2[o] = lab[o2]
        res[d] = x[l2].mean() - x[~l2].mean()
    return float(obs), float(np.percentile(res, 99)), float((res >= obs).mean())


def label_r70(d, p99):
    return 'n/a' if not BB.fin(d, p99) else ('BINANCE-LEAD' if (d >= 0.3 and d > p99) else 'NO-BINANCE-LEAD')


def label_r71(r70, dpf):
    if r70 != 'BINANCE-LEAD':
        return 'n/a'
    return 'n/a' if not BB.fin(dpf) else ('PREDICT-STALE' if dpf <= 0.1 else 'PREDICT-MOVED-TOO')


def label_r72(corr, share):
    if not BB.fin(corr) or corr < 0.05:
        return 'n/a (instrument weak)' if BB.fin(corr) else 'n/a'
    return 'n/a' if not BB.fin(share) else ('OUR-MODEL-SEES' if share >= 0.30 else ('OUR-MODEL-MISSES' if share <= 0.15 else 'PARTLY'))


def label_r73(qs, qr, ss, sr):
    if not BB.fin(qs, qr, ss, sr):
        return ['n/a']
    lab = (['THIN-SIDE'] if qs <= 0.5 * qr else []) + (['WIDE'] if ss >= 1.5 * sr else [])
    return lab or ['NORMAL-BOOK']


def label_r74(d, p99):
    return 'n/a' if not BB.fin(d, p99) else ('BINANCE-FOLLOWS' if (d >= 0.3 and d > p99) else 'NO-BINANCE-FOLLOW')


def controls():
    rep = {}; ok = True

    def chk(name, cond):
        nonlocal ok
        rep[name] = bool(cond); ok = ok and bool(cond)
    okb, repb = BB.controls(); chk('B12b_controls', okb)
    rng = np.random.default_rng(31); res = []
    for lead in (1.0, 0.0):
        n = 3000; lab = np.r_[np.ones(500, bool), np.zeros(n - 500, bool)]; blk = rng.integers(0, 100, n)
        x = rng.normal(0, 1, n) + np.where(lab, lead, 0.0); d, p99, _ = diff_null(x, lab, blk, 200, 5); res.append(label_r70(d, p99))
    chk('E1_events', res == ['BINANCE-LEAD', 'NO-BINANCE-LEAD'])
    # E2 a taker selling into bids gains when Binance falls: return -2 bps at side 0 -> +2 signed; asks keep the sign
    chk('E2_sign', list(tsign([0, 1]) * np.array([-2.0, 3.0])) == [2.0, 3.0])
    chk('E3_rules', label_r71('BINANCE-LEAD', 0.05) == 'PREDICT-STALE' and label_r71('BINANCE-LEAD', 0.5) == 'PREDICT-MOVED-TOO'
        and label_r71('NO-BINANCE-LEAD', 0.0) == 'n/a' and label_r72(0.01, 0.5) == 'n/a (instrument weak)' and label_r72(0.2, 0.35) == 'OUR-MODEL-SEES'
        and label_r72(0.2, 0.1) == 'OUR-MODEL-MISSES' and label_r72(0.2, 0.2) == 'PARTLY' and label_r72(NAN, 0.3) == 'n/a'
        and label_r73(10, 30, 1, 1) == ['THIN-SIDE'] and label_r73(30, 30, 3, 1) == ['WIDE'] and label_r73(30, 30, 1, 1) == ['NORMAL-BOOK']
        and label_r73(NAN, 1, 1, 1) == ['n/a'] and label_r74(0.5, 0.2) == 'BINANCE-FOLLOWS' and label_r74(0.1, 0.2) == 'NO-BINANCE-FOLLOW'
        and label_r70(NAN, 0.1) == 'n/a')
    rep['B12b_controls_detail'] = repb
    return ok, rep


def main():
    if '--selftest' in sys.argv:
        ok, rep = controls(); print(json.dumps({k: v for k, v in rep.items() if not k.endswith('detail')}, indent=1)); print('selftest', 'PASS' if ok else 'FAIL')
        sys.exit(0 if ok else 1)
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    ok, ctl = controls()
    if not ok:
        sys.exit(f'CONTROL FAILED -> no table: {ctl}')
    X = BB.build(); R = X['R']; W0 = X['W0']; OWN = X['OWN']; G = X['G']; LINE = X['LINE']
    k0 = BB.reconcile(R, X['days'])
    if not k0['ok']:
        sys.exit(f'CONTROL FAILED (K0) -> no table: {k0}')
    # address join (as pf_b12d_takers)
    M = pd.read_parquet(MATCH, columns=['mid', 'exec_s', 'taker', 'maker', 'ptick', 'shares'])
    M = M[M.mid.isin(W0)].copy(); M['exec_s'] = M.exec_s.astype('int64'); M['sh6'] = M.shares.round(6)
    key = ['mid', 'exec_s', 'ptick', 'sh6']
    Mu = M.merge(M.groupby(key).size().rename('n').reset_index().query('n == 1')[key], on=key)
    L = R[['mid', 'exec_s', 'ptick', 'shares', 'signer', 'grp']].copy(); L['exec_s'] = L.exec_s.astype('int64'); L['sh6'] = L.shares.round(6)
    Lu = L.merge(L.groupby(key).size().rename('nl').reset_index().query('nl == 1')[key], on=key)
    J = Lu.merge(Mu[key + ['taker', 'maker']], on=key)
    addr = J.groupby('signer').maker.agg(lambda s: s.value_counts().index[0]).str.lower()
    tk = J.groupby('grp').taker.agg(lambda s: s.value_counts().index[0]).str.lower()
    own_addr = str(json.load(open(WHO))['our_address']).lower()
    kj = dict(own_ok=bool(OWN in addr.index and addr[OWN] == own_addr))
    if not kj['own_ok']:
        sys.exit(f'CONTROL FAILED (KJ) -> no table: {kj}')
    top5 = sorted(int(s) for s in R.signer[R.g == 'top 5'].unique()); smart_addr = {addr[s]: s for s in top5 if s in addr.index}
    R['taker'] = tk.reindex(R.grp.values).values
    A = R[R.val_ok & (R.status == 'ok') & (R.depth == 0) & R.taker.notna()]
    TO = A.groupby('grp').agg(mid=('mid', 'first'), side=('side', 'first'), anc=('anc', 'first'), taker=('taker', 'first'), shares=('shares', 'sum'),
                              v1=('v1', 'mean')).reset_index()
    TO['smart'] = TO.taker.isin(set(smart_addr)); TO['smart_id'] = TO.taker.map(smart_addr)
    kj['smart_orders'] = {str(v): int((TO.smart_id == v).sum()) for v in smart_addr.values()}
    rng = np.random.default_rng(SEED); Sm = TO[TO.smart]
    rr = pd.DataFrame(dict(mid=np.repeat(Sm.mid.values, K_RAND), side=np.repeat(Sm.side.values, K_RAND)))
    rr['anc'] = rr.mid.map(W0).values + rng.uniform(1050.0, 59000.0, len(rr))
    EV = pd.concat([Sm[['mid', 'side', 'anc']].assign(kind='SMART'), TO[~TO.smart][['mid', 'side', 'anc']].assign(kind='OTHER'), rr.assign(kind='RANDOM')], ignore_index=True)
    EV['ws'] = EV.mid.map(W0).values; sgT = tsign(EV.side.values)
    # Binance
    BNd = pd.read_parquet(C2.BN, columns=['mid', 'tau', 'bn_mid', 'bn_age']); BNd = BNd[BNd.mid.isin(W0)].sort_values(['mid', 'tau'], kind='stable')
    BG = {m: (g.tau.values.astype(float), g.bn_mid.values.astype(float), g.bn_age.values.astype(float)) for m, g in BNd.groupby('mid')}; del BNd
    PRd = pd.read_parquet(PREDS, columns=['mid', 'tau', 'pred_1000']); PRd = PRd[PRd.mid.isin(W0)].sort_values(['mid', 'tau'], kind='stable')
    PG = {m: (g.tau.values.astype(float), g.pred_1000.values.astype(float)) for m, g in PRd.groupby('mid')}; del PRd
    FR = DB.frames(W0)
    cols = {k: np.full(len(EV), np.nan) for k in list(FEAT_BN) + ['pf_before', 'pf_after_250', 'pf_after_1000', 'pf_realised_1000', 'pred', 'v_side',
                                                                  'spread', 'q_side', 'q_other', 'bn_age']}
    for m, ii in EV.groupby('mid', sort=False).indices.items():
        t = EV.anc.values[ii]; sd = EV.side.values[ii]
        tau, bn, age = BG[m]
        for k, (a, b) in FEAT_BN.items():
            cols[k][ii] = C2.sret(np.ones(len(ii)), C2.bnlook(tau, bn, t + a), C2.bnlook(tau, bn, t + b))
        j = np.searchsorted(tau, t, side='right') - 1; cols['bn_age'][ii] = np.where(j >= 0, age[np.clip(j, 0, None)], np.nan)
        tf, S = FR[m]
        bb = S[0][0][:, 0]; ba = S[1][0][:, 0]; mff = pd.Series(np.where((bb >= 0) & (ba > bb), (ba + bb) / 2.0, np.nan)).ffill().values
        at = lambda x: np.where(np.searchsorted(tf, x, side='right') - 1 >= 0, mff[np.clip(np.searchsorted(tf, x, side='right') - 1, 0, None)], np.nan)
        j0 = np.searchsorted(tf, t, side='left') - 1; jj = np.clip(j0, 0, None); m0 = np.where(j0 >= 0, mff[jj], np.nan)
        cols['pf_before'][ii] = at(t - 50.0) - at(t - 1050.0); cols['pf_after_250'][ii] = at(t + 250.0) - m0; cols['pf_after_1000'][ii] = at(t + 1000.0) - m0
        cols['pf_realised_1000'][ii] = at(t - 50.0 + 1000.0) - at(t - 50.0)
        cols['spread'][ii] = np.where((j0 >= 0) & (bb[jj] >= 0) & (ba[jj] > bb[jj]), (ba[jj] - bb[jj]).astype(float), np.nan)
        cols['q_side'][ii] = np.where(j0 >= 0, np.where(sd == 0, S[0][1][jj, 0], S[1][1][jj, 0]), np.nan)
        cols['q_other'][ii] = np.where(j0 >= 0, np.where(sd == 0, S[1][1][jj, 0], S[0][1][jj, 0]), np.nan)
        ptau, pv = PG[m]; jp = np.searchsorted(ptau, t - 50.0, side='right') - 1; okp = (jp >= 0) & (t - 50.0 - ptau[np.clip(jp, 0, None)] <= B.FRESH)
        cols['pred'][ii] = np.where(okp, pv[np.clip(jp, 0, None)], np.nan)
        gt = G[m]; vb = C1.vlook(gt[0], gt[1], t - 50.0, LINE); va = C1.vlook(gt[0], gt[2], t - 50.0, LINE); cols['v_side'][ii] = np.where(sd == 0, vb, va)
    for k in list(FEAT_BN) + ['pf_before', 'pf_after_250', 'pf_after_1000']:
        EV[k] = cols[k] * sgT
    EV['pred_s'] = cols['pred'] * sgT; EV['pred_raw'] = cols['pred']; EV['pf_realised_1000'] = cols['pf_realised_1000']
    for k in ('v_side', 'spread', 'q_side', 'q_other', 'bn_age'):
        EV[k] = cols[k]
    EV['t_in'] = (EV.anc - EV.ws) / 1000.0
    # timing of SMART orders: following another SMART order on that side within 250 ms
    S_ = EV[EV.kind == 'SMART'].sort_values(['mid', 'side', 'anc']); prev = S_.groupby(['mid', 'side']).anc.shift(1)
    follow_share = float(((S_.anc - prev) <= 250.0).mean())
    lab_all = EV.kind.values; SR = EV[EV.kind.isin(['SMART', 'RANDOM'])]; lab = SR.kind.values == 'SMART'; blk = pd.factorize(SR.mid)[0]
    feats = list(FEAT_BN) + ['pf_before', 'pf_after_250', 'pf_after_1000', 'pred_s', 'v_side', 'spread', 'q_side', 'q_other', 'bn_age', 't_in']
    table = {}
    for k in feats:
        row = {kd: dict(n=int(np.isfinite(EV[k][EV.kind == kd]).sum()), mean=float(np.nanmean(EV[k][EV.kind == kd])),
                        median=float(np.nanmedian(EV[k][EV.kind == kd])), share_pos=float(np.nanmean(EV[k][EV.kind == kd] > 0)))
               for kd in ('SMART', 'OTHER', 'RANDOM')}
        d, p99, p = diff_null(SR[k].values.astype(float), lab, blk, N_NULL, SEED)
        row['SMART_minus_RANDOM'] = dict(diff=d, null_p99=p99, p_one_sided=p)
        table[k] = row
    rnd = EV[EV.kind == 'RANDOM']; okc = np.isfinite(rnd.pred_raw) & np.isfinite(rnd.pf_realised_1000)
    corr = float(np.corrcoef(rnd.pred_raw[okc], rnd.pf_realised_1000[okc])[0, 1]) if okc.sum() > 30 else NAN
    p90 = float(np.nanpercentile(rnd.pred_s, 90)); sm = EV[EV.kind == 'SMART']; share_hi = float(np.nanmean(sm.pred_s > p90))
    r70 = label_r70(table['bn_before']['SMART_minus_RANDOM']['diff'], table['bn_before']['SMART_minus_RANDOM']['null_p99'])
    r71 = label_r71(r70, table['pf_before']['SMART_minus_RANDOM']['diff'])
    r72 = label_r72(corr, share_hi)
    r73 = label_r73(table['q_side']['SMART']['median'], table['q_side']['RANDOM']['median'], table['spread']['SMART']['mean'], table['spread']['RANDOM']['mean'])
    r74 = label_r74(table['bn_after_2000']['SMART_minus_RANDOM']['diff'], table['bn_after_2000']['SMART_minus_RANDOM']['null_p99'])
    readings = dict(R70=r70, R71=r71, R72=r72, R73=r73, R74=r74, R75=[r70, r71, r72] + r73 + [r74])
    res = dict(as_of_note='point estimates, one consumed partial day (220 windows of 2026-11-08, 0.764 window-days)', controls=ctl, K0=k0['ok'], KJ=kj,
               counts={k: int((EV.kind == k).sum()) for k in ('SMART', 'OTHER', 'RANDOM')}, smart_follow_within_250ms=follow_share,
               smart_taker_value_c_per_sh=float(-B.wmean(A.shares.values[A.taker.isin(set(smart_addr)).values], A.v1.values[A.taker.isin(set(smart_addr)).values])), instrument_corr_pred_1000=corr, smart_share_pred_above_random_p90=share_hi,
               table=table, readings=readings)
    json.dump(res, open(f'{out}/b12e.json', 'w'), indent=1, default=float)
    f = lambda v: 'n/a' if not BB.fin(v) else f'{v:+.3f}'
    Lt = [f"B12e what happens when the top makers take. {res['as_of_note']}", f"Controls: {json.dumps({k: v for k, v in ctl.items() if not k.endswith('detail')})}; KJ {kj}",
          f"Counts {res['counts']}; SMART orders within 250 ms of another SMART order on that side: {follow_share * 100:.1f}%; "
          f"their taker gain on the legs hit (= - maker value at 1 s) {res['smart_taker_value_c_per_sh']:+.2f} c/sh", '',
          'feature (taker-favourable sign for moves) | SMART mean / median / share > 0 | OTHER mean / share > 0 | RANDOM mean / share > 0 | SMART - RANDOM (null p99, p)']
    for k, r in table.items():
        Lt.append(f"{k} | {f(r['SMART']['mean'])} / {f(r['SMART']['median'])} / {r['SMART']['share_pos'] * 100:.0f}% | {f(r['OTHER']['mean'])} / {r['OTHER']['share_pos'] * 100:.0f}% | "
                  f"{f(r['RANDOM']['mean'])} / {r['RANDOM']['share_pos'] * 100:.0f}% | {f(r['SMART_minus_RANDOM']['diff'])} ({f(r['SMART_minus_RANDOM']['null_p99'])}, {r['SMART_minus_RANDOM']['p_one_sided']:.3f})")
    Lt += [f"Instrument: corr(pred_1000, realised 1 s Predict change) at RANDOM {corr:+.3f}; SMART share above RANDOM's 90th percentile of signed pred_1000 {share_hi * 100:.1f}%", '',
           'Readings: ' + json.dumps(readings)]
    open(f'{out}/b12e.txt', 'w').write('\n'.join(Lt) + '\n'); print('\n'.join(Lt))


if __name__ == '__main__':
    main()
