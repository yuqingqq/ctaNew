"""Forward test of the chase line on untouched days (USER 2026-11-27: "Agree, and we test on more dates"; days: "We pick some dates different
from our test days previously"; inputs: "Refit without them"; budget: "Approve, two slots"). P-2026-005, public data, no orders. Declared
before any forward day is read; the freeze (RECEIPT.md with model hashes, thresholds, the primary cell and the PASS rule) is a commit made
after the consumed-day stages and before the first forward stage, which refuses to run without it.
Days: 2026-11-20 .. 2026-11-26 (7 complete UTC days; this line used 10-23, 10-31, 11-01, 11-02). Building a day reads its raw hour files and,
as every step-1 build does, the previous day's last hour and the next day's first two: 11-19 23Z (inside another study's near-holdout block)
and 11-27 00-01Z. The ladder-race (11-04..11-08, 11-09..11-13) and event-pressure (11-14..11-18) blocks are not read. Under the USER's
10-22 rule these reads change no other test's frozen arms; they are recorded in the receipt.
Inputs: CHASE_TAKES' 97 less the other-venue family (the 20 u_* gaps and flows), the fair-value family (8, from Binance spot) and the three
Predict-fill inputs timed on the venue clock (p_pf5, q_net_share5, q_fill_r1s: not known at the decision, PREDICT_INPUTS' caveat): 66 inputs.
Models: the three entry scores refit on the four consumed days with the pinned trees (TM.HGB): MOVE (the 1 s move model; score = predicted
move - half spread - fee), DIRECT_TB_s3 (the conservative value taken back after 1 s at + 3c), DECOMP_s10 (fill x net per filled share at
+ 10c); leave-one-day-out for the consumed-day check, one final fit on all four days for the forward days. Thresholds: the pooled quantiles of
the leave-one-day-out scores (top 10% / top 1%), fixed numbers. One action per move (its first selected tick); the control: random move
onsets (1,250 a day, seeded).
Exits (EXIT_FIXED's code): primary = taking at the 1 s calm (TAKE_C1000); challenger = the maker exit (cancel when Binance moves our way,
re-post at the 1 s calm, take after 10 s); comparators sell after 1 s, keep favourites, hold; conservative entry book; latency + 0 (the
PASS) and + 10 / + 25 ms (reported); entries matched >= 15.022 s before the window's last-30 s cut (EARLY).
Forward-mode patches (declared, in-process only, the same in the consumed-day parity runs): the day guards admit exactly the declared
days; public_score / public_score3's other-venue loops are empty; chase_takes.labels' Binance-spot fair value returns NaN (inputs dropped);
the Predict paths come from a light loader (the frozen loader's book reading and frame clock, the fixed 50 ms grid, no market-making model
inputs; its forward bound lifted for the declared days only); the trade-imputation offset (unused by these paths) is 0.
Stages: selftest | parity_scores | parity_paths <day> | parity_rows <day> [<dir>] | parity_exits <day> [<dir>] | refit | check <day> |
        freeze | build <day> | points <day> | labels <day> | select | exits <day> | read
Outputs (write-once): OUT/chase_forward/..., chase_forward_*.json, predictfun/freeze/chase_forward_1120_1126/RECEIPT.md.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/chase_forward.py <stage> ...
"""
import collections
import datetime as dt
import glob
import hashlib
import json
import os
import pickle
import resource
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import exit_follow as E  # noqa: E402
import exit_maker as EM  # noqa: E402
import exit_maker2 as M2  # noqa: E402
import exit_maker3 as M3  # noqa: E402
import exit_fixed as EF  # noqa: E402
import sharpe_exits as SX  # noqa: E402

CT, C, TR, TK, TE, TM, WT, F, VR, MM = E.CT, E.C, E.TR, E.TK, E.TE, E.TM, E.WT, E.F, E.VR, E.MM
PS, P3, CV = CT.PS, CT.P3, CT.CV
I = MM.KP.QL.I
REPO = H.parents[2]
DAYS0 = tuple(E.DAYS)
FWD = ('20261120', '20261121', '20261122', '20261123', '20261124', '20261125', '20261126')
FOUT = WT.OUT / 'chase_forward'
FREEZE = REPO / 'predictfun' / 'freeze' / 'chase_forward_1120_1126'
Q = E.Q
DROP_Q = ['p_pf5', 'q_net_share5', 'q_fill_r1s']
SELQ = (('MOVE_top10', 'MOVE', .1), ('MOVE_top1', 'MOVE', .01), ('DECOMP_s10_top1', 'DECOMP_s10', .01), ('DIRECT_TB_s3_top10', 'DIRECT_TB_s3', .1))
SELS = tuple(s for s, _, _ in SELQ) + ('CONTROL_onset',)
CELLS = (('MOVE_top10', 3.), ('DIRECT_TB_s3_top10', 3.), ('DECOMP_s10_top1', 3.), ('DECOMP_s10_top1', 10.))
N_NULL, SEED, CONTROL_PER_DAY = 200, 20261127, 1250
LATS = (0., 10., 25.)
GRID_MS, GRID_N = 50., 5996
POL = ('HOLD', 'TB1S', 'FAV_ONLY', 'TAKE_C1000', EF.FIXED)


def feats_all():
    return list(P3.V3) + list(CT.MOVE) + list(CT.PF) + list(CT.FV)


def keep_feats():
    drop = set(c for c in P3.V3 if c.startswith('u_')) | set(CT.FV) | set(DROP_Q)
    k = [f for f in feats_all() if f not in drop]
    C.require(len(k) == 66 and len(drop) == 31, f'inputs: {len(k)} kept, {len(drop)} dropped')
    return k


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------------------------ forward mode
def spot_off(v, day):
    """chase_takes.labels' fair value without Binance spot: one NaN print -> NaN fair-value inputs (dropped)."""
    C.require(v == 'binance_spot', f'forward mode: {v} is off')
    lo, hi = (float(x) for x in WT.LV.day_ms(day))
    return np.array([lo]), np.array([np.nan]), np.array([0.]), np.array([1]), (lo, hi), np.zeros(0)


def light_worlds_all(day, mids, src, off, reg):
    """VR.worlds_all's (market, world, status) from the step-1 books and clock only: I.load_day's book reading (parts of [day - 1 h, day + 24 h)
    outside the sealed block, rows within [ws - preroll, ws + 300 s]) and window_sides' frame clock (kf), the fixed 50 ms grid; no trades."""
    C.require(day in DAYS0 or day in FWD, f'{day}: the light loader serves the consumed and the declared forward days only')
    S1d = f'{I.IO}/step1/{day}'
    batch = set(int(x) for x in mids)
    rg = pd.read_parquet(f'{S1d}/registry.parquet', columns=['mid', 'ws', 'outcome'])
    d0 = I._ms(f'{day[:4]}-{day[4:6]}-{day[6:]}'); d1 = d0 + 86400000
    W0 = {}
    for m, ws, o in zip(rg.mid, rg.ws, rg.outcome):
        w = float(ws) * 1000.0
        if d0 <= w < d1 and not (I.SEAL_MS[0] <= w < I.SEAL_MS[1]) and o in ('Up', 'Down') and int(m) in batch:
            W0[int(m)] = w
    ck = json.load(open(f'{S1d}/clock_rl_d2.json'))
    C.require(abs(ck['d_dec'] - 2.0) < 1e-9, f'{day}: clock built with decision time {ck["d_dec"]}')
    cols = ['mid', 't_ms', 'recv_ns'] + [f'{s}{i}_{x}' for s in 'ba' for i in range(1, 6) for x in 'ps']
    parts = []
    for fn in sorted(glob.glob(f'{S1d}/books/part_*.parquet')):
        b = os.path.basename(fn); dd, hh = b[5:13], b[14:16]
        h0 = I._ms(f'{dd[:4]}-{dd[4:6]}-{dd[6:]}') + int(hh) * 3600000
        if not (d0 - 3600000 <= h0 < d0 + 86400000 and not (I.SEAL_MS[0] <= h0 < I.SEAL_MS[1])):
            continue
        if not W0 or not I.part_overlaps(fn, W0):
            continue
        x = pd.read_parquet(fn, columns=cols); x = x[x.mid.isin(W0)]
        rel = x.t_ms.values - x.mid.map(W0).values
        parts.append(x[(rel >= -I.PREROLL_MS) & (rel <= 300000.0)])
    B = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    BI = B.groupby('mid', sort=False).indices
    seen = set()
    for m in sorted(W0, key=W0.get):
        if m not in BI:
            continue
        ws = W0[m]
        bb_ = B.iloc[BI[m]].sort_values('t_ms', kind='stable')
        tf = bb_.t_ms.values.astype(float)
        kf = np.maximum.accumulate(np.maximum(tf + I.PF_MIN, bb_.recv_ns.values / 1e6 - ck['shift_ms']) + ck['d_dec'])
        bb = bb_.b1_p.values.astype('int64'); ba = bb_.a1_p.values.astype('int64')
        lev = {0: (bb_[[f'b{i}_p' for i in range(1, 6)]].values.astype('int64'), bb_[[f'b{i}_s' for i in range(1, 6)]].values.astype(float)),
               1: (bb_[[f'a{i}_p' for i in range(1, 6)]].values.astype('int64'), bb_[[f'a{i}_s' for i in range(1, 6)]].values.astype(float))}
        tg = ws + GRID_MS * np.arange(GRID_N)
        jk = np.searchsorted(kf, tg, side='right') - 1
        W = tuple(dict(m=m, side=s, ws=ws, tf=tf, kf=kf, jk=jk, tg=tg, opp=(ba, bb)[s], P5=lev[s][0], S5=lev[s][1], anc=np.zeros(0), ptl=[], shl=[])
                  for s in (0, 1))
        seen.add(m)
        yield m, W, ('precision_not_2' if int(reg.loc[m, 'decimal_precision']) != 2 else 'loaded')
    for m in mids:
        if int(m) not in seen:
            yield int(m), None, 'loader_excluded'


def forward_mode(days=()):
    """The declared in-process patches (see the docstring). days: forward days the guards must admit (empty in the consumed-day parity runs)."""
    C.require(set(days) <= set(FWD), 'an undeclared forward day')
    if days:
        C.require((FREEZE / 'RECEIPT.md').exists(), 'no freeze receipt: forward stages refuse to run')
        WT.DAYS = tuple(DAYS0) + tuple(days)
        CT.DAYS = tuple(DAYS0) + tuple(days)
    PS.VEN = ()
    P3.VEN3 = ()
    CV.load = spot_off
    VR.worlds_all = light_worlds_all
    MM.KP.Tape.offset = lambda self, day: 0.0


# ------------------------------------------------------------------------------------------------------------------ models
def targets(X):
    d, lim, md = X.d.to_numpy(float), X.lim.to_numpy(float), X.md.to_numpy(float)
    cost = d * (lim - md) + CT.fee(lim)
    mk = X.m1s.to_numpy(float)
    PA = X[[f'pA_{k}' for k in range(5)]].to_numpy(float); SA = X[[f'sA_{k}' for k in range(5)]].to_numpy(float)
    PB = X[[f'pB_{k}' for k in range(5)]].to_numpy(float); SB = X[[f'sB_{k}' for k in range(5)]].to_numpy(float)
    okB = np.isfinite(X.dtB.to_numpy(float))
    BA = TK.minbook(PA, SA, PB, SB)
    W3, W10 = TK.vwalk(BA[0], BA[1], d, lim + d * 3.), TK.vwalk(BA[0], BA[1], d, lim + d * 10.)
    eP, eS = X[[f'eP1s_{k}' for k in range(5)]].to_numpy(float), X[[f'eS1s_{k}' for k in range(5)]].to_numpy(float)
    pay = X.payout.to_numpy(float)
    filled10 = okB & (W10[0] > 0)
    return dict(cost=cost, mo1=d * (mk - md), okB=okB, ytb=TE.exit_values(*W3, d, eP, eS, pay)[0] / Q, fl10=W10[0] / Q, filled10=filled10,
                net10=np.where(filled10, (d * (mk * W10[0] - W10[1]) - W10[2]) / np.maximum(W10[0], 1e-9), np.nan))


def hgb():
    return HistGradientBoostingRegressor(random_state=0, **TM.HGB)


def scores_lodo(X, feats, T):
    """Leave-one-day-out scores, each model on the original's input form (MOVE on the frame, the fill-aware models on float32 arrays)."""
    day = X.day.to_numpy()
    Xf = X[feats].to_numpy(np.float32)
    pm, ptb, pfl, pnt = (np.full(len(X), np.nan) for _ in range(4))
    okh = np.isfinite(T['mo1'])
    for dh in DAYS0:
        te = day == dh
        tr = okh & ~te
        pm[te] = hgb().fit(X.loc[tr, feats], T['mo1'][tr]).predict(X.loc[te, feats])
        tr = T['okB'] & ~te
        ptb[te] = hgb().fit(Xf[tr], T['ytb'][tr]).predict(Xf[te])
        pfl[te] = hgb().fit(Xf[tr], T['fl10'][tr]).predict(Xf[te])
        tr = T['filled10'] & ~te
        pnt[te] = hgb().fit(Xf[tr], T['net10'][tr]).predict(Xf[te])
        C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    return dict(pred_1s=pm, MOVE=pm - T['cost'], DIRECT_TB_s3=ptb, DECOMP_s10=np.clip(pfl, 0., 1.) * pnt)


def final_models(X, feats, T):
    Xf = X[feats].to_numpy(np.float32)
    okh = np.isfinite(T['mo1'])
    return dict(move=hgb().fit(X.loc[okh, feats], T['mo1'][okh]), tb=hgb().fit(Xf[T['okB']], T['ytb'][T['okB']]),
                fill=hgb().fit(Xf[T['okB']], T['fl10'][T['okB']]), net=hgb().fit(Xf[T['filled10']], T['net10'][T['filled10']]))


def scores_final(Mo, X, feats, cost):
    Xf = X[feats].to_numpy(np.float32)
    pm = Mo['move'].predict(X[feats])
    return dict(pred_1s=pm, MOVE=pm - cost, DIRECT_TB_s3=Mo['tb'].predict(Xf), DECOMP_s10=np.clip(Mo['fill'].predict(Xf), 0., 1.) * Mo['net'].predict(Xf))


def select(Sc, key, mv_k, thr, seeds_by_day=None, day=None):
    """One action per move at its first tick with score >= the fixed threshold; the control: seeded random onsets per day."""
    ok = np.isfinite(Sc['MOVE'])
    flags = {}
    for nm, col, _ in SELQ:
        sc = Sc[col]
        flags[nm] = CT.first_per_move(ok & np.isfinite(sc) & (sc >= thr[nm]), key)
    on = CT.first_per_move(mv_k == 1, key)
    if seeds_by_day is None:                                          # the consumed days: E.selections' control draw
        flags['CONTROL_onset'] = np.sort(np.random.default_rng(SEED).choice(on, min(E.N_CONTROL, len(on)), replace=False))
    else:
        pick = []
        for dh in sorted(set(day)):
            od = on[day[on] == dh]
            pick.append(np.random.default_rng(seeds_by_day[dh]).choice(od, min(CONTROL_PER_DAY, len(od)), replace=False))
        flags['CONTROL_onset'] = np.sort(np.concatenate(pick)) if pick else np.zeros(0, int)
    return flags


def flags_frame(X, flags):
    rows = np.unique(np.concatenate([v for v in flags.values() if len(v)])) if any(len(v) for v in flags.values()) else np.zeros(0, int)
    S = X.iloc[rows][['day', 't', 'd', 'key', 'mid', 'lim', 'payout']].reset_index(drop=True)
    for nm, ix in flags.items():
        S[nm] = np.isin(rows, ix)
    return S


# ------------------------------------------------------------------------------------------------------------------ exits
def exits_day(day, S, od, lats=LATS, extra=None):
    """EXIT_FIXED.labels for the actions S of one day (columns day, t, d, key, mid, lim, payout + selection flags), conservative entry book."""
    S = S[S.day == day].reset_index(drop=True)
    BN = M3.Binance(*CT.mid_series(day))
    LG = EF.legs_who(day)
    LGr, lst = EM.legs_of(day)
    C.require(set(LG) == set(LGr) and all(all(np.array_equal(LG[m][k], LGr[m][k]) for k in range(4)) for m in LG), 'legs_who differs from EM.legs_of')
    del LGr
    reg = MM.registry(day).set_index('mid')
    rg = pd.read_parquet(MM.S1 / day / 'registry.parquet').set_index('mid')
    by_mid = S.groupby('mid').indices
    o = collections.defaultdict(lambda: np.full(len(S), np.nan))
    n_cap = 0
    CALM = {L: M2.calm_times(BN.kn, S.t.to_numpy(float) + TM.FEED + TM.DECIDE + E.NET + L + E.HOLD + M2.LEARN + L, EF.GAP) for L in lats}
    src = MM.KP.Tape(); off = src.offset(day)
    loader = collections.Counter()
    for h, gr in reg.groupby('hour'):
        mids = [int(m) for m in gr.index if int(m) in by_mid]
        if not mids:
            continue
        for m, W, lst_ in VR.worlds_all(day, mids, src, off, reg):
            if lst_ != 'loaded':
                loader[lst_] += len(by_mid[m])
                for i in by_mid[m]:
                    o['status_unloaded'][i] = 1.
                continue
            p_, st, info = VR.vm_path(W, float(reg.loc[m, 'ws']) * 1000., 50.)
            if p_ is None:
                loader[f'path_{st}'] += len(by_mid[m])
                for i in by_mid[m]:
                    o['status_unloaded'][i] = 1.
                continue
            t0, end = float(info['t_start']), float(info['end_rel'])
            f = np.asarray(p_['frames'], float)
            f0, fk = f[:, 0], f[:, 1]
            okb = (f[:, 2] >= 0) & (f[:, 12] <= 100) & (f[:, 2] < f[:, 12])
            midy = np.where(okb, (f[:, 2] + f[:, 12]) / 2., np.nan)
            cut = float(rg.loc[m, 'we']) * 1000. - t0 - E.LAST_MS
            lg = LG.get(int(m))
            for i in by_mid[m]:
                t, d, lim, pay = float(S.t.iat[i]), float(S.d.iat[i]), float(S.lim.iat[i]), float(S.payout.iat[i])
                act0 = t + TM.FEED + TM.DECIDE - t0 + E.NET + E.HOLD
                o['cut_minus_act_ms'][i] = cut - act0
                side = 1 if d > 0 else 0
                for L in lats:
                    act = act0 + L
                    jv = int(np.searchsorted(f0, act, 'right') - 1)
                    pa, sa = TK.side_levels(f[jv], side)
                    pb, sb_ = TK.side_levels(f[jv + 1], side) if jv + 1 < len(f) else (np.full(5, -1.), np.zeros(5))
                    bk = TK.minbook(pa[None], sa[None], pb[None], sb_[None])
                    k0 = int(np.searchsorted(fk, act, 'right'))
                    kt, sm = fk[k0:], (midy[k0:] if d > 0 else 100. - midy[k0:])
                    calm = float(CALM[L][i]) - t0
                    for s in (3., 10.):
                        pre = f'L{int(L)}_AB_s{int(s)}'
                        fl, co, fe_ = (x[0] for x in TK.vwalk(*bk, np.array([d]), np.array([lim + d * s])))
                        o[f'{pre}_fill'][i] = fl
                        o[f'{pre}_entry_fee_c'][i] = fe_
                        if fl <= 0:
                            for pol in POL:
                                o[f'{pre}_{pol}_pnl'][i] = 0.
                                o[f'{pre}_{pol}_exit_fee_c'][i] = 0.
                            continue

                        def put(pol, mq, mc, tq, tc, tf):
                            o[f'{pre}_{pol}_pnl'][i] = d * (mc + tc + (fl - mq - tq) * 100. * pay - co) - fe_ - tf
                            o[f'{pre}_{pol}_exit_fee_c'][i] = tf
                        e = co / fl if d > 0 else 100. - co / fl
                        for pol in ('HOLD', 'TB1S', 'FAV_ONLY'):
                            x = E.decide(pol, E.trigger(kt, sm, e, cut), E.mid_at(kt, sm, act + E.T_MED), E.mid_at(kt, sm, act + E.T_UQ), act, cut)
                            put(pol, 0., 0., *EF.take_exit_L(f, f0, end, d, fl, x, L))
                        put('TAKE_C1000', 0., 0., *EF.take_exit_L(f, f0, end, d, fl, calm if calm <= cut else None, L))
                        res = EF.maker_loop_L(EF.MODE, EF.GAP, EF.PRICE, calm, fl, d, f, f0, fk, lg, t0, cut, BN, L, EF.T_DEAD)
                        n_cap += int(res['capped'])
                        inc = res['incs']
                        mq = float(sum(z['q'] for z in inc)); mc = float(sum(z['q'] * z['px'] for z in inc))
                        x = M2.tail_x('TAKE', kt, sm, res['D']) if (res['D'] > res['t_first'] and fl - mq > 1e-9) else None
                        put(EF.FIXED, mq, mc, *EF.take_exit_L(f, f0, end, d, fl - mq, x, L))
                        o[f'{pre}_maker_sh'][i] = mq
            C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    out = pd.concat([S, pd.DataFrame(dict(o), index=S.index)], axis=1)
    od.mkdir(parents=True, exist_ok=True)
    return out, dict(day=day, actions=int(len(S)), legs=lst, capped_loops=n_cap, unloaded=dict(loader))


def null_values(day, X, s):
    """TAKE_C1000's P&L at + 0 ms on the conservative book at + s c for every labelled tick of the day (the null's population)."""
    X = X[X.day == day].reset_index(drop=True)
    BN = M3.Binance(*CT.mid_series(day))
    calm = M2.calm_times(BN.kn, X.t.to_numpy(float) + TM.FEED + TM.DECIDE + E.NET + E.HOLD + M2.LEARN, EF.GAP)
    reg = MM.registry(day).set_index('mid')
    rg = pd.read_parquet(MM.S1 / day / 'registry.parquet').set_index('mid')
    by_mid = X.groupby('mid').indices
    val, cma = np.full(len(X), np.nan), np.full(len(X), np.nan)
    src = MM.KP.Tape(); off = src.offset(day)
    for h, gr in reg.groupby('hour'):
        mids = [int(m) for m in gr.index if int(m) in by_mid]
        if not mids:
            continue
        for m, W, lst_ in VR.worlds_all(day, mids, src, off, reg):
            if lst_ != 'loaded':
                continue
            p_, st, info = VR.vm_path(W, float(reg.loc[m, 'ws']) * 1000., 50.)
            if p_ is None:
                continue
            t0, end = float(info['t_start']), float(info['end_rel'])
            f = np.asarray(p_['frames'], float)
            f0 = f[:, 0]
            cut = float(rg.loc[m, 'we']) * 1000. - t0 - E.LAST_MS
            for i in by_mid[m]:
                t, d, lim, pay = float(X.t.iat[i]), float(X.d.iat[i]), float(X.lim.iat[i]), float(X.payout.iat[i])
                act = t + TM.FEED + TM.DECIDE - t0 + E.NET + E.HOLD
                cma[i] = cut - act
                jv = int(np.searchsorted(f0, act, 'right') - 1)
                side = 1 if d > 0 else 0
                pa, sa = TK.side_levels(f[jv], side)
                pb, sb_ = TK.side_levels(f[jv + 1], side) if jv + 1 < len(f) else (np.full(5, -1.), np.zeros(5))
                fl, co, fe_ = (x[0] for x in TK.vwalk(*TK.minbook(pa[None], sa[None], pb[None], sb_[None]), np.array([d]), np.array([lim + d * s])))
                if fl <= 0:
                    val[i] = 0.
                    continue
                x = float(calm[i]) - t0
                tq, tc, tf = EF.take_exit_L(f, f0, end, d, fl, x if x <= cut else None, 0.)
                val[i] = d * (tc + (fl - tq) * 100. * pay - co) - fe_ - tf
            C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    return val, cma


# ------------------------------------------------------------------------------------------------------------------ consumed-day stages
def parity_scores():
    """The 97-input leave-one-day-out refit must reproduce CHASE_TAKES' pred_1s and FILL_AWARE's DIRECT_TB_s3 / DECOMP_s10."""
    resource.setrlimit(resource.RLIMIT_CPU, (3000, 3001))
    fn = H / 'chase_forward_parity_scores.json'
    C.require(not fn.exists(), f'{fn.name} exists (write-once)')
    X, feats, _ = CT.frame(CT.OUT)
    C.require(feats == feats_all(), 'the input list differs from CHASE_TAKES')
    T = targets(X)
    Sc = scores_lodo(X, feats, T)
    ref = X[['day', 't', 'd']].merge(pd.read_parquet(CT.OUT / 'predictions.parquet', columns=['day', 't', 'd', 'pred_1s']), on=['day', 't', 'd'], how='left', validate='1:1')
    rfa = X[['day', 't', 'd']].merge(pd.read_parquet(E.FA.OUT / 'predictions.parquet', columns=['day', 't', 'd', 'DIRECT_TB_s3', 'DECOMP_s10', 'MOVE']),
                                    on=['day', 't', 'd'], how='left', validate='1:1')
    diff = dict(pred_1s=float(np.nanmax(np.abs(Sc['pred_1s'] - ref.pred_1s.to_numpy(float)))),
                MOVE=float(np.nanmax(np.abs(Sc['MOVE'] - rfa.MOVE.to_numpy(float)))),
                DIRECT_TB_s3=float(np.nanmax(np.abs(Sc['DIRECT_TB_s3'] - rfa.DIRECT_TB_s3.to_numpy(float)))),
                DECOMP_s10=float(np.nanmax(np.abs(Sc['DECOMP_s10'] - rfa.DECOMP_s10.to_numpy(float)))))
    TR.dump(fn, dict(asof='2026-11-27', rows=int(len(X)), max_abs_diff=diff))
    print('max |diff|', diff, flush=True)
    C.require(all(v < 1e-9 for v in diff.values()), f'the refit does not reproduce the original scores: {diff}')


def parity_paths(day):
    """The light loader must reproduce the frozen loader's path (status, start, end, frames) for every market of a consumed day."""
    C.require(day in DAYS0, f'{day} is not a consumed day')
    resource.setrlimit(resource.RLIMIT_CPU, (1500, 1501))
    reg = MM.registry(day).set_index('mid')
    src = MM.KP.Tape(); off = src.offset(day)
    worlds0 = VR.worlds_all
    cnt = collections.Counter()
    for h, gr in reg.groupby('hour'):
        mids = [int(m) for m in gr.index]
        A = {m: (W, s) for m, W, s in worlds0(day, mids, src, off, reg)}
        B = {m: (W, s) for m, W, s in light_worlds_all(day, mids, src, off, reg)}
        for m in mids:
            (Wa, sa), (Wb, sb) = A[m], B[m]
            if sa != 'loaded' or sb != 'loaded':
                cnt[f'status_{sa}|{sb}'] += 1
                if sa != sb:
                    cnt['differs'] += 1
                continue
            pa, sta, ia = VR.vm_path(Wa, float(reg.loc[m, 'ws']) * 1000., 50.)
            pb, stb, ib = VR.vm_path(Wb, float(reg.loc[m, 'ws']) * 1000., 50.)
            same = (sta == stb) and ((pa is None and pb is None) or (pa is not None and pb is not None and np.array_equal(pa['frames'], pb['frames'])
                                                                     and ia.get('t_start') == ib.get('t_start') and ia.get('end_rel') == ib.get('end_rel')))
            cnt['same' if same else 'differs'] += 1
            if not same:
                cnt[f'differs_{sta}|{stb}'] += 1
        C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    fn = H / f'chase_forward_parity_paths_{day}.json'
    TR.dump(fn, dict(asof='2026-11-27', day=day, counts=dict(cnt)))
    print(day, dict(cnt), flush=True)
    C.require(cnt['differs'] == 0, f'{day}: the light loader differs on {cnt["differs"]} markets')


def parity_rows(day, od):
    """Forward-mode points and labels of a consumed day (patches on, guards as consumed) must equal CHASE_TAKES' kept inputs and labels."""
    C.require(day in DAYS0, f'{day} is not a consumed day')
    resource.setrlimit(resource.RLIMIT_CPU, (2400, 2401))
    od = Path(od)
    forward_mode(())
    CT.points(day, od)
    CT.labels(day, od)
    kf = keep_feats()
    out = {}
    for kind in ('points', 'labels'):
        A = pd.read_parquet(CT.OUT / f'{day}_{kind}.parquet')
        B = pd.read_parquet(od / f'{day}_{kind}.parquet')
        C.require(len(A) == len(B) and (A[['t', 'd']].to_numpy() == B[['t', 'd']].to_numpy()).all(), f'{kind}: rows differ')
        cols = [c for c in A.columns if c in B.columns and not (c.startswith('u_') or c in CT.FV or c in DROP_Q)]
        bad = [c for c in cols if not (A[c].equals(B[c]) or (np.issubdtype(A[c].dtype, np.number) and np.allclose(A[c].to_numpy(float), B[c].to_numpy(float), rtol=0, atol=0, equal_nan=True)))]
        out[kind] = dict(rows=int(len(A)), columns_compared=len(cols), differing=bad, kept_inputs_present=sorted(set(kf) & set(cols)).__len__())
    fn = H / f'chase_forward_parity_rows_{day}.json'
    TR.dump(fn, dict(asof='2026-11-27', day=day, **out))
    print(day, out, flush=True)
    C.require(not out['points']['differing'] and not out['labels']['differing'], 'forward-mode rows differ from CHASE_TAKES')


def parity_exits(day, od):
    """exits_day on EXIT_FIXED's actions of a consumed day must reproduce EXIT_FIXED's P&L for every policy and latency."""
    C.require(day in DAYS0, f'{day} is not a consumed day')
    resource.setrlimit(resource.RLIMIT_CPU, (2400, 2401))
    S = E.selections()
    S = S[S.day == day].reset_index(drop=True)
    L_ = pd.read_parquet(CT.OUT / f'{day}_labels.parquet', columns=['t', 'd', 'mid', 'lim', 'payout'])
    S = S.merge(L_, on=['t', 'd'], how='left', validate='1:1')
    X, st = exits_day(day, S, Path(od))
    R = pd.read_parquet(EF.OUT / f'{day}.parquet')
    C.require((R.t.to_numpy() == X.t.to_numpy()).all(), 'actions differ from EXIT_FIXED')
    mx = 0.
    for L in LATS:
        for s in (3., 10.):
            for pol in POL:
                k = f'L{int(L)}_AB_s{int(s)}_{pol}_pnl'
                a, r = X[k].to_numpy(float), R[k].to_numpy(float)
                C.require(np.allclose(a, r, rtol=0, atol=1e-9), f'{k} not reproduced from EXIT_FIXED')
                mx = max(mx, float(np.nanmax(np.abs(a - r))))
    TR.dump(H / f'chase_forward_parity_exits_{day}.json', dict(asof='2026-11-27', day=day, actions=int(len(X)), max_abs_diff_c=mx, status=st))
    print(day, 'exits reproduced, max |diff|', mx, flush=True)


def refit():
    """The 66-input models: leave-one-day-out scores, thresholds and selections on the consumed days; the final fits for the forward days."""
    resource.setrlimit(resource.RLIMIT_CPU, (3000, 3001))
    od = FOUT / 'refit'
    C.require(not (od / 'models.pkl').exists(), 'refit outputs exist (write-once)')
    X, feats, _ = CT.frame(CT.OUT)
    kf = keep_feats()
    T = targets(X)
    Sc = scores_lodo(X, kf, T)
    ok = np.isfinite(Sc['MOVE'])
    thr = {nm: float(np.quantile(Sc[col][ok & np.isfinite(Sc[col])], 1. - q)) for nm, col, q in SELQ}
    key = X.key.to_numpy()
    flags = select(Sc, key, X.mv_k.to_numpy(float), thr)
    Mo = final_models(X, kf, T)
    od.mkdir(parents=True, exist_ok=True)
    with open(od / 'models.pkl', 'wb') as fh:
        pickle.dump(Mo, fh)
    P = X[['day', 't', 'd', 'key']].copy()
    for k, v in Sc.items():
        P[k] = v
    P.to_parquet(od / 'lodo_scores.parquet', index=False)
    S = flags_frame(X, flags)
    S.to_parquet(od / 'consumed_selections.parquet', index=False)
    rho = dict(MOVE=float(spearmanr(Sc['pred_1s'], T['mo1'], nan_policy='omit').correlation),
               DIRECT_TB_s3=float(spearmanr(Sc['DIRECT_TB_s3'][T['okB']], T['ytb'][T['okB']]).correlation))
    out = dict(asof='2026-11-27', inputs=kf, dropped=[f for f in feats if f not in kf], thresholds=thr, actions={k: int(len(v)) for k, v in flags.items()},
               spearman_lodo=rho, models_sha256=sha(od / 'models.pkl'), rows=int(len(X)))
    TR.dump(H / 'chase_forward_refit.json', out)
    print(json.dumps({k: v for k, v in out.items() if k not in ('inputs', 'dropped')}), flush=True)


def check(day):
    """Exits of the refit selections on one consumed day (the frozen loader; nothing forward)."""
    C.require(day in DAYS0, f'{day} is not a consumed day')
    resource.setrlimit(resource.RLIMIT_CPU, (2400, 2401))
    od = FOUT / 'check'
    fp = od / f'{day}.parquet'
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    S = pd.read_parquet(FOUT / 'refit' / 'consumed_selections.parquet')
    X, st = exits_day(day, S, od)
    X.to_parquet(fp, index=False)
    TR.dump(od / f'{day}.json', st)
    print(day, st, flush=True)


def cells_of(X, keys, dayk, per_day, days, scope_ms, lats):
    """Per selection x limit x latency x policy: totals, Sharpe per window (day equivalent), daily, filled positions (scope: EARLY)."""
    early = X.cut_minus_act_ms.to_numpy(float) >= scope_ms
    wk = (X.day.astype(str) + ':' + X.mid.astype(np.int64).astype(str)).to_numpy()
    out = {}
    for sel in SELS:
        m = X[sel].to_numpy(bool) & early
        for L in lats:
            for s in (3., 10.):
                pre = f'L{int(L)}_AB_s{int(s)}'
                fl = X[f'{pre}_fill'].to_numpy(float)
                for pol in POL:
                    pn = np.nan_to_num(X[f'{pre}_{pol}_pnl'].to_numpy(float)) / 100.
                    st = SX.stats(SX.window_series(pn[m], wk[m], keys), dayk, per_day)
                    out[f'{sel}|{pre}|{pol}'] = dict(total_usd=st['total_usd'], sharpe_day_equiv=st['sharpe_day_equiv'], daily_usd=st['daily_usd'],
                                                     days_positive=int(sum(v > 0 for v in st['daily_usd'])), max_drawdown_usd=st['max_drawdown_usd'],
                                                     actions=int(m.sum()), filled=int((m & (fl > 0)).sum()))
    return out


def freeze():
    """Compose the receipt from the consumed-day stages (thresholds, model hash, the check's cells, the primary cell by the declared rule)."""
    FREEZE.mkdir(parents=True, exist_ok=True)
    fn = FREEZE / 'RECEIPT.md'
    C.require(not fn.exists(), 'RECEIPT.md exists (write-once)')
    R = json.loads((H / 'chase_forward_refit.json').read_text())
    X = pd.concat([pd.read_parquet(FOUT / 'check' / f'{d}.parquet') for d in DAYS0], ignore_index=True)
    keys, dayk = [], []
    for d in DAYS0:
        reg = MM.registry(d).sort_values('ws')
        keys += [f'{d}:{int(m)}' for m in reg.mid]; dayk += [d] * len(reg)
    keys, dayk = np.array(keys), np.array(dayk)
    SX.DAYS = DAYS0
    cells = cells_of(X, keys, dayk, len(keys) / len(DAYS0), DAYS0, EF.EARLY_MS, (0.,))
    cand = {f'{sel}|AB_s{int(s)}': cells[f'{sel}|L0_AB_s{int(s)}|TAKE_C1000'] for sel, s in CELLS}
    elig = {k: v for k, v in cand.items() if v['filled'] >= 30 and v['sharpe_day_equiv'] is not None}
    C.require(len(elig) > 0, 'no candidate cell has 30 filled positions on the consumed days')
    primary = max(elig, key=lambda k: elig[k]['sharpe_day_equiv'])
    TR.dump(H / 'chase_forward_check.json', dict(asof='2026-11-27', cells=cells, candidates=cand, primary=primary))
    code = sha(H / 'chase_forward.py')
    lines = [
        '# Forward test receipt: the chase line on 2026-11-20 .. 2026-11-26 (P-2026-005, declared 2026-11-27)',
        '',
        'USER 2026-11-27: "Agree, and we test on more dates"; days: "We pick some dates different from our test days previously"; inputs:',
        '"Refit without them" (other venues, Binance spot); budget: "Approve, two slots". Public data, no orders.',
        '',
        '## Frozen candidate',
        f'- Builder: predictfun/reviews/take_reprice_20261125/chase_forward.py, sha256 {code} (this commit).',
        f'- Models: OUT/chase_forward/refit/models.pkl, sha256 {R["models_sha256"]}: MOVE / DIRECT_TB_s3 / DECOMP_s10 refit on the four consumed days',
        f'  (10-23, 10-31, 11-01, 11-02) with TM.HGB on {len(R["inputs"])} inputs; dropped: {", ".join(R["dropped"])}.',
        '- Thresholds (pooled leave-one-day-out quantiles, fixed): ' + ', '.join(f'{k} {v:.6f}' for k, v in R['thresholds'].items()) + '.',
        '- Actions: one per move, its first tick at or above the threshold; entries matched >= 15.022 s before the window\'s last-30 s cut.',
        '- Entry: a 50-share IOC at the known best + 3c / + 10c on the conservative book (A\'s levels at their size in the next snapshot), matched',
        '  175.5 ms after the Binance tick (+ L ms of extra latency each way to Predict).',
        f'- Primary cell (declared rule: of {", ".join(cand)} the best consumed-day day-equivalent Sharpe of taking at the 1 s calm, >= 30 filled):',
        f'  **{primary}**.',
        '- Exits: primary = take everything at the first 1 s Binance calm after the fill notice (TAKE_C1000); challenger = the maker exit',
        '  CX_OURS_C1000_JOIN_T10_TAKE (EXIT_FIXED); comparators sell after 1 s, keep favourites, hold.',
        '',
        '## PASS (computed in the read; primary cell, take at the 1 s calm, + 0 ms, all 7 days)',
        '- (a) total P&L > 0; (b) total > the 95th percentile of the null; (c) positive on >= 4 of 7 days; (d) >= 30 filled positions.',
        f'- Null ({N_NULL} seeded draws, seed {SEED}): per day, as many random actions as the primary cell has (EARLY), each a random move at a random',
        '  labelled tick of it (CHASE_TAKES\' random_actions), valued by the same entry and exit; the 95th percentile of the 7-day total.',
        '- Reported, not part of PASS: the other three cells, the maker challenger paired against take-at-calm, + 10 / + 25 ms, the control',
        '  (1,250 random onsets a day), comparators, the day-clustered mean and its spread, selection rates against the consumed days.',
        f'- Multiplicity: {len(CELLS)} entry cells x 2 exits = {2 * len(CELLS)} combinations in this race; one primary.',
        '',
        '## Reads',
        '- Raw hour files of 11-20 .. 11-26, plus 11-19 23Z and 11-27 00-01Z (step-1 padding). 11-19 lies in another study\'s near-holdout',
        '  block; the ladder-race and event-pressure blocks are not read. No other test\'s arms change (USER 10-22 rule).',
        '- Forward-mode patches as in the builder\'s docstring; their consumed-day parity: chase_forward_parity_*.json.',
        '',
        '## Consumed-day check (seen days, for orientation only)',
    ]
    for k, v in cand.items():
        lines.append(f'- {k}: take at the 1 s calm {v["total_usd"]:+.0f} $, day-equivalent Sharpe {v["sharpe_day_equiv"]:+.2f}, {v["filled"]} filled, {v["days_positive"]}/4 days > 0.')
    fn.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


# ------------------------------------------------------------------------------------------------------------------ forward stages
def frozen():
    r = (FREEZE / 'RECEIPT.md').read_text()
    C.require(sha(H / 'chase_forward.py') in r, 'the builder changed after the freeze')
    C.require(sha(FOUT / 'refit' / 'models.pkl') in r, 'the models changed after the freeze')
    return r


def build(day, slot):
    """The day's Predict inputs (pf_step1, timing, booktime, the clock) and the taker join, each a run_light job."""
    C.require(day in FWD, f'{day} is not a declared forward day')
    frozen()
    C.require(not (MM.S1 / day).exists(), f'step1 for {day} exists: refusing to rebuild')
    rl = str(REPO / 'predictfun' / 'research' / 'run_light.sh')
    env = dict(os.environ, PF_SLOT=slot)
    for tag, cmd, extra in ((f'cf_s1_{day}', ['python3', '-m', 'predictfun.research.pf_step1', day], {}),
                            (f'cf_t_{day}', ['python3', '-m', 'predictfun.research.pf_step1_timing', day], {}),
                            (f'cf_bt_{day}', ['python3', '-m', 'predictfun.research.pf_step1_booktime', day], {}),
                            (f'cf_f_{day}', ['python3', '-m', 'predictfun.research.pf_step5_features_rl', day], {'PF_RL_DDEC': '2'}),
                            (f'cf_w_{day}', ['python3', str(H / 'chase_forward.py'), 'winners', day], {})):
        r = subprocess.run([rl, tag, '--'] + cmd, cwd=str(REPO), env=dict(env, **extra), capture_output=True, text=True)
        tail = (r.stdout + r.stderr).strip().splitlines()[-1:] if (r.stdout + r.stderr).strip() else ['']
        C.require(r.returncode == 0, f'{tag} failed: {tail}')
        print(tag, 'ok', flush=True)


def winners(day):
    C.require(day in FWD, f'{day} is not a declared forward day')
    frozen()
    forward_mode((day,))
    WT.legs(day)


def fwd_points(day):
    C.require(day in FWD, f'{day} is not a declared forward day')
    frozen()
    forward_mode((day,))
    CT.points(day, FOUT / 'rows')


def fwd_labels(day):
    C.require(day in FWD, f'{day} is not a declared forward day')
    frozen()
    forward_mode((day,))
    CT.labels(day, FOUT / 'rows')


def fwd_frame(days=FWD):
    src = FOUT / 'rows'
    Pp = pd.concat([pd.read_parquet(src / f'{d}_points.parquet') for d in days], ignore_index=True)
    Lb = pd.concat([pd.read_parquet(src / f'{d}_labels.parquet') for d in days], ignore_index=True)
    X = Lb.merge(Pp, on=['day', 't', 'd'], how='inner', validate='1:1')
    C.require(len(X) == len(Lb), 'a labelled tick without its point')
    X['key'] = X.day.astype(str) + ':' + X.mkey.astype(str)
    return X.sort_values(['day', 'd', 't'], kind='stable').reset_index(drop=True)


def fwd_select():
    resource.setrlimit(resource.RLIMIT_CPU, (1800, 1801))
    frozen()
    fp = FOUT / 'forward_selections.parquet'
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    R = json.loads((H / 'chase_forward_refit.json').read_text())
    with open(FOUT / 'refit' / 'models.pkl', 'rb') as fh:
        Mo = pickle.load(fh)
    kf = R['inputs']
    parts, scs, n_rows, acts, above = [], [], 0, collections.Counter(), collections.Counter()
    for i, d in enumerate(FWD):                                         # day by day: thresholds are fixed, actions are per move
        X = fwd_frame((d,))
        T = targets(X)
        Sc = scores_final(Mo, X, kf, T['cost'])
        flags = select(Sc, X.key.to_numpy(), X.mv_k.to_numpy(float), R['thresholds'], {d: SEED + i}, X.day.to_numpy())
        parts.append(flags_frame(X, flags))
        P = X[['day', 't', 'd', 'key', 'mid', 'lim', 'payout']].copy()
        for k, v in Sc.items():
            P[k] = v
        scs.append(P)
        n_rows += len(X)
        for k, v in flags.items():
            acts[k] += int(len(v))
        for nm, col, _ in SELQ:
            above[nm] += int(np.sum(Sc[col] >= R['thresholds'][nm]))
        C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    pd.concat(parts, ignore_index=True).to_parquet(fp, index=False)
    pd.concat(scs, ignore_index=True).to_parquet(FOUT / 'forward_scores.parquet', index=False)
    rate = {nm: above[nm] / n_rows for nm, _, _ in SELQ}
    TR.dump(FOUT / 'forward_select.json', dict(asof='2026-11-27', rows=int(n_rows), actions=dict(acts), tick_rate_above_threshold=rate))
    print(dict(acts), rate, flush=True)


def fwd_exits(day):
    C.require(day in FWD, f'{day} is not a declared forward day')
    resource.setrlimit(resource.RLIMIT_CPU, (2400, 2401))
    frozen()
    forward_mode((day,))
    od = FOUT / 'exits'
    fp = od / f'{day}.parquet'
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    S = pd.read_parquet(FOUT / 'forward_selections.parquet')
    X, st = exits_day(day, S, od)
    X.to_parquet(fp, index=False)
    Xa = fwd_frame((day,))
    for s in (3., 10.):
        v, cma = null_values(day, Xa, s)
        pd.DataFrame(dict(day=Xa.day[Xa.day == day].to_numpy(), t=Xa.t[Xa.day == day].to_numpy(), d=Xa.d[Xa.day == day].to_numpy(),
                          key=Xa.key[Xa.day == day].to_numpy(), val=v, cut_minus_act_ms=cma)).to_parquet(od / f'{day}_null_s{int(s)}.parquet', index=False)
    TR.dump(od / f'{day}.json', st)
    print(day, st, flush=True)


def fwd_read():
    resource.setrlimit(resource.RLIMIT_CPU, (1800, 1801))
    rc = frozen()
    fn = H / 'chase_forward_result.json'
    C.require(not fn.exists(), f'{fn.name} exists (write-once)')
    import re
    primary = re.search(r'\*\*([A-Za-z0-9_]+\|AB_s\d+)\*\*', rc).group(1)
    sel, pre_s = primary.split('|'); s_p = float(pre_s.split('_s')[1])
    X = pd.concat([pd.read_parquet(FOUT / 'exits' / f'{d}.parquet') for d in FWD], ignore_index=True)
    keys, dayk = [], []
    for d in FWD:
        reg = MM.registry(d).sort_values('ws')
        keys += [f'{d}:{int(m)}' for m in reg.mid]; dayk += [d] * len(reg)
    keys, dayk = np.array(keys), np.array(dayk)
    SX.DAYS = FWD
    cells = cells_of(X, keys, dayk, len(keys) / len(FWD), FWD, EF.EARLY_MS, LATS)
    pc = cells[f'{sel}|L0_AB_s{int(s_p)}|TAKE_C1000']
    # the null: per day as many random EARLY actions as the primary cell, at a random labelled tick of a random move
    early = X.cut_minus_act_ms.to_numpy(float) >= EF.EARLY_MS
    n_d = {d: int((X[sel].to_numpy(bool) & early & (X.day == d).to_numpy()).sum()) for d in FWD}
    rng = np.random.default_rng(SEED)
    tot = np.zeros(N_NULL)
    for d in FWD:
        Nd = pd.read_parquet(FOUT / 'exits' / f'{d}_null_s{int(s_p)}.parquet')
        ok = np.isfinite(Nd.val.to_numpy(float)) & (Nd.cut_minus_act_ms.to_numpy(float) >= EF.EARLY_MS)
        g = CT.move_groups(Nd.key.to_numpy(), ok)
        val = Nd.val.to_numpy(float) / 100.
        for k in range(N_NULL):
            if n_d[d] > 0:
                tot[k] += float(val[CT.random_actions(g, min(n_d[d], len(g[1])), rng)].sum())
    p95 = float(np.quantile(tot, .95))
    PASS = dict(a_total_positive=bool(pc['total_usd'] > 0), b_above_null_p95=bool(pc['total_usd'] > p95),
                c_days_positive_ge_4=bool(pc['days_positive'] >= 4), d_filled_ge_30=bool(pc['filled'] >= 30))
    verdict = 'PASS' if all(PASS.values()) else ('INSUFFICIENT' if not PASS['d_filled_ge_30'] else 'FAIL')
    daily = np.array(pc['daily_usd'])
    out = dict(asof=dt.datetime.now(dt.timezone.utc).isoformat(), primary=primary, verdict=verdict, predicates=PASS, primary_cell=pc,
               null=dict(draws=N_NULL, seed=SEED, p95_usd=p95, mean_usd=float(tot.mean()), actions_per_day=n_d),
               day_cluster=dict(mean_usd=float(daily.mean()), sd_usd=float(daily.std(ddof=1)), t=float(daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily)))) if daily.std(ddof=1) > 0 else None),
               cells=cells)
    TR.dump(fn, out)
    print(verdict, PASS, 'total', round(pc['total_usd'], 2), 'null p95', round(p95, 2), flush=True)


def selftest():
    kf = keep_feats()
    C.require(not any(c.startswith('u_') or c in CT.FV or c in DROP_Q for c in kf), 'a dropped input kept')
    C.require(len(set(kf)) == len(kf), 'an input twice')
    # select(): first tick per move at or above the threshold; the per-day control is seeded and bounded
    key = np.array(['a', 'a', 'a', 'b', 'b', 'c'])
    sc = np.array([0., 2., 3., 1., 5., -1.])
    Sc = dict(MOVE=sc, DECOMP_s10=sc, DIRECT_TB_s3=sc)
    thr = dict(MOVE_top10=2., MOVE_top1=4., DECOMP_s10_top1=4., DIRECT_TB_s3_top10=2.)
    fl = select(Sc, key, np.array([1, 2, 3, 1, 2, 1.]), thr, {'x': 1}, np.array(['x'] * 6))
    assert fl['MOVE_top10'].tolist() == [1, 4] and fl['MOVE_top1'].tolist() == [4], fl
    assert sorted(fl['CONTROL_onset'].tolist()) == [0, 3, 5], fl['CONTROL_onset']
    # spot_off: NaN spot, refuses other venues
    v = spot_off('binance_spot', '20261102')
    assert np.isnan(v[1]).all() and v[4][1] - v[4][0] == 86400000.
    try:
        spot_off('bybit_perp', '20261102')
        raise AssertionError('another venue accepted')
    except ValueError as e_:
        assert 'is off' in str(e_)
    # forward_mode refuses an undeclared day and a forward run without the receipt (when absent)
    try:
        forward_mode(('20261104',))
        raise AssertionError('an undeclared forward day accepted')
    except ValueError as e_:
        assert 'undeclared forward day' in str(e_)
    if not (FREEZE / 'RECEIPT.md').exists():
        try:
            forward_mode(('20261120',))
            raise AssertionError('a forward day accepted without a receipt')
        except ValueError as e_:
            assert 'no freeze receipt' in str(e_)
    for fn_ in (build, winners, fwd_points, fwd_labels, fwd_exits):
        try:
            fn_('20261102', 'claude') if fn_ is build else fn_('20261102')
            raise AssertionError(f'{fn_.__name__} accepted a consumed day')
        except ValueError as e_:
            assert 'not a declared forward day' in str(e_), e_
    for fn_ in (parity_paths, check):
        try:
            fn_('20261120')
            raise AssertionError(f'{fn_.__name__} accepted a forward day')
        except ValueError as e_:
            assert 'not a consumed day' in str(e_), e_
    print('chase_forward selftest PASS')


if __name__ == '__main__':
    cmd, a_ = sys.argv[1], sys.argv[2:]
    dict(selftest=selftest, parity_scores=parity_scores, refit=refit, freeze=freeze, select=fwd_select, read=fwd_read)[cmd]() if cmd in (
        'selftest', 'parity_scores', 'refit', 'freeze', 'select', 'read') else dict(
        parity_paths=lambda: parity_paths(a_[0]), parity_rows=lambda: parity_rows(a_[0], a_[1]), parity_exits=lambda: parity_exits(a_[0], a_[1]),
        check=lambda: check(a_[0]), build=lambda: build(a_[0], a_[1] if len(a_) > 1 else os.environ.get('PF_SLOT', 'claude')),
        winners=lambda: winners(a_[0]), points=lambda: fwd_points(a_[0]), labels=lambda: fwd_labels(a_[0]), exits=lambda: fwd_exits(a_[0]))[cmd]()
