"""The fixed maker exit (cancel when the trend continues, re-post at calm) under added latency, and who fills it (USER 2026-11-27: "Sure we can
cancel maker order and repost when the trend continues, and proceed"). Four consumed days, every market, no orders; POST HOC (the exit was
chosen on these days); declared before this run.
FIXED = CX_OURS_C1000_JOIN_T10_TAKE: chosen by a rule declared before this run from exit_maker3.json: of EXIT_MAKER3's 12 trend-cancel
variants (cancel only when Binance moves our way, i.e. the trend continues; re-post at the next calm) the highest median day-equivalent
Sharpe over its well-filled conservative cells (re-evaluated in the read: a check). The exit: from the first time after our fill notice with
no Binance futures mid change in the last 1 s, a post-only order at the best on the exit side; cancelled when Binance moves our way (the
cancel lands 2 + 22 ms after the change is known), re-posted at the next 1 s calm on the book known then; refused orders re-decided at the
next calm; 10 s after the first order rests the rest is taken. EXIT_MAKER3's venue rules (refusal and queue ahead from the venue frame on
arrival, Predict's prints at venue time t_ms2 - 50 ms, cancels ahead not credited).
Entries: EXIT_FOLLOW's on the conservative AB entry book (+ 3c / + 10c). Comparators: HOLD, TB1S, FAV_ONLY, TAKE_C1000 (everything taken at
the same 1 s calm).
Latency: L = 0 / 10 / 25 ms added to every message between us and Predict, each way: the entry IOC arrives L later (the 150 ms hold
follows), the fill notice L later, every post-only order and cancel lands L later, a refusal is known L later, every exit IOC arrives L
later; the Binance feed is unchanged. L = 0 reproduces EXIT_MAKER3 bit for bit (a check).
Who fills us: the taker of each print that fills the maker order: the top taker (0x7cab1c46a5B0557008198b92697F36df855ac0AD), the day's ten
largest takers by shares other than it, everyone else; their share of our maker shares and the side's mid 5 s after their fills against our price.
Before the move: the share of maker shares filled in the 100 ms before the Binance change that triggers a cancel (its exchange time),
against the share of the order's resting time inside those windows; a ratio above 1 = fills crowd in just before Binance moves.
Scopes ALL / EARLY (matched >= 15.022 s before the cut); counts over the cells with >= 30 filled positions at L = 0.
Outputs (write-once): OUT/exit_fixed/<day>.parquet + .json, exit_fixed.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/exit_fixed.py \
       selftest | labels <day> [<dir>] | read [<dir>]
"""
import collections
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import exit_follow as E  # noqa: E402
import exit_maker as EM  # noqa: E402
import exit_maker2 as M2  # noqa: E402
import exit_maker3 as M3  # noqa: E402
import sharpe_exits as SX  # noqa: E402

CT, C, TR, TK, TE, TM, WT, F, VR, MM = E.CT, E.C, E.TR, E.TK, E.TE, E.TM, E.WT, E.F, E.VR, E.MM
OUT = WT.OUT / 'exit_fixed'
DAYS, Q = E.DAYS, E.Q
LEARN, M50 = M2.LEARN, M2.M50
LATS = (0., 10., 25.)
FIXED, MODE, GAP, PRICE, T_DEAD = 'CX_OURS_C1000_JOIN_T10_TAKE', 'CX_OURS', 1000., 'JOIN', 10000.
COMP = ('HOLD', 'TB1S', 'FAV_ONLY', 'TAKE_C1000')
POLICIES = COMP + (FIXED,)
COMBOS = M3.COMBOS
TOP = '0x7cab1c46a5B0557008198b92697F36df855ac0AD'
PRE_MS = 100.
WHO = ('top', 'top2_10', 'others')
EARLY_MS, MIN_FILLED, STRAT = M3.EARLY_MS, M3.MIN_FILLED, M3.STRAT


def fixed_by_rule(J3):
    cells = J3['counts_early']['cells_list']
    cands = [f'CX_OURS_C{int(g)}_{p}_{t}' for g in M3.GAPS for p in M3.PRICES for t, _ in M3.TAILS]
    med = {c: float(np.median([J3['cells'][f'EARLY|{cell}|{c}']['sharpe_day_equiv'] for cell in cells])) for c in cands}
    return max(cands, key=med.get), med


def legs_who(day):
    """EM.legs_of's prints, plus who took: 2 = the top taker, 1 = the day's ten largest takers by shares other than it, 0 = everyone else."""
    L = pd.read_parquet(WT.OUT / f'winners_legs_{day}.parquet', columns=['mid', 'taker_hash', 'side', 'ptick', 'shares', 't_ms2', 'taker'])
    t = L.t_ms2.fillna(L.groupby('taker_hash').t_ms2.transform('min'))
    top10 = set(L.groupby('taker').shares.sum().nlargest(10).index)
    who = np.where(L.taker.str.lower() == TOP.lower(), 2, np.where(L.taker.isin(top10), 1, 0))
    L = L.assign(t=t, who=who)[t.notna()].sort_values('t', kind='stable')
    return {int(m): (g.t.to_numpy(float), g.side.to_numpy(int), g.ptick.to_numpy(float), g.shares.to_numpy(float), g.who.to_numpy(int))
            for m, g in L.groupby('mid')}


def maker_fill2(lg, t_lo, t_hi, d, p, ahead, size):
    """EM.maker_fill, also returning the index of each print that filled us."""
    if lg is None or size <= 0:
        return 0., [], [], []
    tl, sd, pr, sh = lg[:4]
    a, b = np.searchsorted(tl, t_lo, 'right'), np.searchsorted(tl, t_hi, 'right')
    want = 1 if d > 0 else 0
    q_ahead, got, ts, qs, ks = ahead, 0., [], [], []
    for k in range(a, b):
        if sd[k] != want:
            continue
        at = abs(pr[k] - p) < 1e-9
        beyond = (pr[k] > p + 1e-9) if d > 0 else (pr[k] < p - 1e-9)
        if beyond:
            x = size - got
        elif at:
            use = min(q_ahead, sh[k]); q_ahead -= use
            x = min(size - got, sh[k] - use)
        else:
            continue
        if x > 0:
            got += x; ts.append(float(tl[k])); qs.append(float(x)); ks.append(k)
        if got >= size - 1e-9:
            break
    return got, ts, qs, ks


def take_exit_L(f, f0, end, d, q, x, L):
    """M2.take_exit with the exit IOC arriving L later."""
    if x is None or q <= 1e-9:
        return 0., 0., 0.
    mt = x + E.DEC + E.NET + L + E.HOLD
    if mt > end:
        return 0., 0., 0.
    je = int(np.searchsorted(f0, mt, 'right') - 1)
    eP, eS = TK.side_levels(f[je], 0 if d > 0 else 1)
    r_ = TE.walk_q(eP[None], eS[None], np.array([-d]), np.array([1. if d > 0 else 99.]), np.array([q]))
    return tuple(float(z[0]) for z in r_)


def post_L(f, f0, fk, d, dec, pr_, L):
    """M2.post with the order landing L later."""
    t_rest = dec + E.DEC + LEARN + L
    jk = int(np.searchsorted(fk, dec, 'right') - 1)
    px = M2.price_of(f[jk], d, pr_) if jk >= 0 else np.nan
    if not np.isfinite(px):
        return np.nan, t_rest, False, 0., False
    ja = int(np.searchsorted(f0, t_rest, 'right') - 1)
    return (px, t_rest) + M2.venue_queue(f[ja], d, px)


def maker_loop_L(mode, g, pr_, x0, fl, d, f, f0, fk, lg, t0, cut, BN, L, T):
    """M3.maker_loop with latency L and the deadline T after the first order rests; also who filled us and the pre-move windows."""
    t_first = x0 + E.DEC + LEARN + L
    D = min(t_first + T, cut)
    out = dict(t_first=t_first, D=D, incs=[], decisions=0, cancels=0, refusals=0, capped=False, rest_ms=0., pre_ms=0.)
    if not (np.isfinite(x0) and D > t_first):
        return out
    rem, x = fl, x0
    while rem > 1e-9 and np.isfinite(x):
        t_rest = x + E.DEC + LEARN + L
        if t_rest >= D:
            break
        if out['decisions'] >= M3.CAP:
            out['capped'] = True
            break
        out['decisions'] += 1
        px, _, refused, ahead, _ = post_L(f, f0, fk, d, x, pr_, L)
        if not np.isfinite(px):
            if mode == 'NOCX':
                break
            x = BN.calm(t0 + x + M3.RETRY_MS, g) - t0
            continue
        if refused:
            out['refusals'] += 1
            if mode == 'NOCX':
                break
            x = BN.calm(t0 + t_rest + LEARN + L, g) - t0
            continue
        trig = None if mode == 'NOCX' else BN.next_change(t0 + x, d, mode)
        trig = None if trig is None else trig - t0
        t_end = D if trig is None else min(trig + M3.CANCEL_MS + L, D)
        got, ts, qs, ks = maker_fill2(lg, t0 + t_rest + M50, t0 + t_end + M50, d, px, ahead, rem)
        live_end = (ts[-1] - M50 - t0) if (got >= rem - 1e-9 and ts) else t_end     # a filled order stops resting at its last fill
        out['rest_ms'] += live_end - t_rest
        if trig is not None:
            ev = trig - TM.FEED
            out['pre_ms'] += max(0., min(ev, live_end) - max(t_rest, ev - PRE_MS))
        for tv, qv, kv in zip(ts, qs, ks):
            tvr = tv - M50 - t0
            C.require(tvr > t_rest, 'a maker fill at or before the order reached the venue')
            ev = None if trig is None else trig - TM.FEED
            out['incs'].append(dict(t=tvr, q=qv, px=px, flight=bool(ev is not None and tvr > ev), pre=bool(ev is not None and ev - PRE_MS < tvr <= ev),
                                    who=int(lg[4][kv])))
        rem -= got
        if t_end >= D or rem <= 1e-9:
            break
        out['cancels'] += 1
        x = BN.calm(t0 + trig, g) - t0
    return out


def labels(day, od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (2400, 2401))
    C.require(day in DAYS, f'{day} is not a declared day')
    od = Path(od) if od else OUT
    fp = od / f'{day}.parquet'
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    S = E.selections()
    S = S[S.day == day].reset_index(drop=True)
    L_ = pd.read_parquet(CT.OUT / f'{day}_labels.parquet', columns=['t', 'd', 'mid', 'lim', 'payout'])
    S = S.merge(L_, on=['t', 'd'], how='left', validate='1:1')
    BN = M3.Binance(*CT.mid_series(day))
    LG = legs_who(day)
    LGr, lst = EM.legs_of(day)
    C.require(set(LG) == set(LGr) and all(all(np.array_equal(LG[m][k], LGr[m][k]) for k in range(4)) for m in LG), 'legs_who differs from EM.legs_of')
    del LGr
    reg = MM.registry(day).set_index('mid')
    rg = pd.read_parquet(MM.S1 / day / 'registry.parquet').set_index('mid')
    by_mid = S.groupby('mid').indices
    o = collections.defaultdict(lambda: np.full(len(S), np.nan))
    n_cap = 0
    CALM = {L: M2.calm_times(BN.kn, S.t.to_numpy(float) + TM.FEED + TM.DECIDE + E.NET + L + E.HOLD + LEARN + L, GAP) for L in LATS}
    src = MM.KP.Tape(); off = src.offset(day)
    for h, gr in reg.groupby('hour'):
        mids = [int(m) for m in gr.index if int(m) in by_mid]
        if not mids:
            continue
        for m, W, lst_ in VR.worlds_all(day, mids, src, off, reg):
            C.require(lst_ == 'loaded', f'{m} {lst_}')
            p_, st, info = VR.vm_path(W, float(reg.loc[m, 'ws']) * 1000., 50.)
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
                for L in LATS:
                    act = act0 + L
                    jv = int(np.searchsorted(f0, act, 'right') - 1)
                    pa, sa = TK.side_levels(f[jv], side)
                    pb, sb_ = TK.side_levels(f[jv + 1], side) if jv + 1 < len(f) else (np.full(5, -1.), np.zeros(5))
                    bk = TK.minbook(pa[None], sa[None], pb[None], sb_[None])
                    k0 = int(np.searchsorted(fk, act, 'right'))
                    kt, sm = fk[k0:], (midy[k0:] if d > 0 else 100. - midy[k0:])
                    calm = float(CALM[L][i]) - t0
                    for b, s in COMBOS:
                        pre = f'L{int(L)}_{b}_s{int(s)}'
                        fl, co, fe_ = (x[0] for x in TK.vwalk(*bk, np.array([d]), np.array([lim + d * s])))
                        o[f'{pre}_fill'][i] = fl
                        o[f'{pre}_entry_fee_c'][i] = fe_
                        if fl <= 0:
                            for pol in POLICIES:
                                o[f'{pre}_{pol}_pnl'][i] = 0.
                                o[f'{pre}_{pol}_exit_fee_c'][i] = 0.
                            continue

                        def put(pol, mq, mc, tq, tc, tf):
                            o[f'{pre}_{pol}_pnl'][i] = d * (mc + tc + (fl - mq - tq) * 100. * pay - co) - fe_ - tf
                            o[f'{pre}_{pol}_exit_fee_c'][i] = tf
                        e = co / fl if d > 0 else 100. - co / fl
                        for pol in ('HOLD', 'TB1S', 'FAV_ONLY'):
                            x = E.decide(pol, E.trigger(kt, sm, e, cut), E.mid_at(kt, sm, act + E.T_MED), E.mid_at(kt, sm, act + E.T_UQ), act, cut)
                            put(pol, 0., 0., *take_exit_L(f, f0, end, d, fl, x, L))
                        put('TAKE_C1000', 0., 0., *take_exit_L(f, f0, end, d, fl, calm if calm <= cut else None, L))
                        res = maker_loop_L(MODE, GAP, PRICE, calm, fl, d, f, f0, fk, lg, t0, cut, BN, L, T_DEAD)
                        n_cap += int(res['capped'])
                        inc = res['incs']
                        mq = float(sum(z['q'] for z in inc)); mc = float(sum(z['q'] * z['px'] for z in inc))
                        x = M2.tail_x('TAKE', kt, sm, res['D']) if (res['D'] > res['t_first'] and fl - mq > 1e-9) else None
                        put(FIXED, mq, mc, *take_exit_L(f, f0, end, d, fl - mq, x, L))
                        o[f'{pre}_maker_sh'][i] = mq
                        o[f'{pre}_flight_sh'][i] = sum(z['q'] for z in inc if z['flight'])
                        o[f'{pre}_pre_sh'][i] = sum(z['q'] for z in inc if z['pre'])
                        o[f'{pre}_rest_ms'][i] = res['rest_ms']
                        o[f'{pre}_pre_ms'][i] = res['pre_ms']
                        o[f'{pre}_cancels'][i] = res['cancels']
                        for wi, wn in enumerate(('others', 'top2_10', 'top')):
                            zz = [z for z in inc if z['who'] == wi]
                            o[f'{pre}_{wn}_sh'][i] = sum(z['q'] for z in zz)
                            g5 = [(E.mid_at(kt, sm, z['t'] + 5000.) - (z['px'] if d > 0 else 100. - z['px'])) * z['q'] for z in zz]
                            g5 = [v for v in g5 if np.isfinite(v)]
                            o[f'{pre}_{wn}_gain5_c_sh'][i] = sum(g5)
                            o[f'{pre}_{wn}_gain5_n_sh'][i] = sum(z['q'] for z in zz if np.isfinite(E.mid_at(kt, sm, z['t'] + 5000.)))
            C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    out = S[['day', 't', 'd', 'key', 'mid'] + list(E.SELS)].copy()
    out = pd.concat([out, pd.DataFrame(dict(o), index=out.index)], axis=1)
    od.mkdir(parents=True, exist_ok=True)
    out.to_parquet(fp, index=False)
    TR.dump(od / f'{day}.json', dict(asof='2026-11-27', day=day, actions=int(len(S)), legs=lst, capped_loops=n_cap))
    print(day, len(S), 'capped', n_cap, 'rss', round(F.rss_mib()), flush=True)


def read(od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    reh = od is not None
    od = Path(od) if od else None
    fn = (od or H) / 'exit_fixed.json'
    C.require(not fn.exists(), 'exit_fixed.json exists (write-once)')
    J3 = json.loads((H / 'exit_maker3.json').read_text())
    fx, med = fixed_by_rule(J3)
    C.require(fx == FIXED, f'the declared rule picks {fx}, not {FIXED}')
    src = od or OUT
    present = [dh for dh in DAYS if (src / f'{dh}.parquet').exists()]
    X = pd.concat([pd.read_parquet(src / f'{dh}.parquet') for dh in present], ignore_index=True)
    X3 = pd.concat([pd.read_parquet(M3.OUT / f'{dh}.parquet') for dh in present], ignore_index=True)
    C.require(len(X) == len(X3) and (X.t.to_numpy() == X3.t.to_numpy()).all(), 'the actions differ from EXIT_MAKER3')
    inputs = {dh: json.loads((src / f'{dh}.json').read_text()) for dh in present}
    C.require(sum(v['capped_loops'] for v in inputs.values()) == 0, 'a maker loop hit the decision cap')
    parity = {}
    for b, s in COMBOS:
        for pol in POLICIES:
            a, r = X[f'L0_{b}_s{int(s)}_{pol}_pnl'].to_numpy(float), X3[f'{b}_s{int(s)}_{pol}_pnl'].to_numpy(float)
            C.require(np.allclose(a, r, rtol=0, atol=1e-9), f'L0 {b}_s{int(s)}_{pol} not reproduced from EXIT_MAKER3')
            parity[f'{b}_s{int(s)}_{pol}'] = float(np.max(np.abs(a - r)))
    keys, dayk = [], []
    for dd in present:
        reg = MM.registry(dd).sort_values('ws')
        keys += [f'{dd}:{int(m)}' for m in reg.mid]
        dayk += [dd] * len(reg)
    keys, dayk = np.array(keys), np.array(dayk)
    per_day = len(keys) / len(present)
    wk = (X.day.astype(str) + ':' + X.mid.astype(np.int64).astype(str)).to_numpy()
    early = X.cut_minus_act_ms.to_numpy(float) >= EARLY_MS
    out = dict(asof='2026-11-27', post_hoc=True, rehearsal=reh, days=present, inputs=inputs, parity_max_abs_c=parity, fixed=FIXED,
               rule_median_sharpe=med, latencies_ms=list(LATS), cells={})
    for L in LATS:
        for sel in E.SELS:
            msel = X[sel].to_numpy(bool)
            for scope, sm_ in (('ALL', msel), ('EARLY', msel & early)):
                ii = np.flatnonzero(sm_)
                for b, s in COMBOS:
                    pre = f'L{int(L)}_{b}_s{int(s)}'
                    fl = X[f'{pre}_fill'].to_numpy(float)
                    filled = ii[fl[ii] > 0]
                    efee = float(X[f'{pre}_entry_fee_c'].to_numpy(float)[filled].sum() / 100.)
                    for pol in POLICIES:
                        pn = X[f'{pre}_{pol}_pnl'].to_numpy(float) / 100.
                        st = SX.stats(SX.window_series(pn[sm_], wk[sm_], keys), dayk, per_day)
                        c = dict(total_usd=st['total_usd'], sharpe_day_equiv=st['sharpe_day_equiv'], max_drawdown_usd=st['max_drawdown_usd'],
                                 daily_usd=st['daily_usd'], days_positive=int(sum(v > 0 for v in st['daily_usd'])), filled_entries=int(len(filled)),
                                 filled_shares=float(fl[filled].sum()), entry_fee_usd=efee,
                                 exit_fee_usd=float(X[f'{pre}_{pol}_exit_fee_c'].to_numpy(float)[filled].sum() / 100.))
                        if pol == FIXED and len(filled):
                            col = lambda k: X[f'{pre}_{k}'].to_numpy(float)[filled]  # noqa: E731
                            mq = col('maker_sh').sum()
                            c.update(maker_closed_share=float(mq / fl[filled].sum()), in_flight_share=float(col('flight_sh').sum() / mq) if mq > 0 else None,
                                     pre_move_fill_share=float(col('pre_sh').sum() / mq) if mq > 0 else None,
                                     pre_move_time_share=float(col('pre_ms').sum() / col('rest_ms').sum()) if col('rest_ms').sum() > 0 else None,
                                     cancels_mean=float(col('cancels').mean()))
                            for wn in WHO:
                                n5 = col(f'{wn}_gain5_n_sh').sum()
                                c[f'{wn}_share_of_maker'] = float(col(f'{wn}_sh').sum() / mq) if mq > 0 else None
                                c[f'{wn}_gain5_c'] = float(col(f'{wn}_gain5_c_sh').sum() / n5) if n5 > 0 else None
                        out['cells'][f'L{int(L)}|{scope}|{sel}|{b}_s{int(s)}|{pol}'] = c
    cells = [(sel, f'AB_s{int(s)}') for sel in STRAT for _, s in COMBOS]
    big = [(a, b) for a, b in cells if out['cells'][f'L0|EARLY|{a}|{b}|TB1S']['filled_entries'] >= MIN_FILLED]
    g = lambda L, a, b, p: out['cells'][f'L{int(L)}|EARLY|{a}|{b}|{p}']  # noqa: E731
    sh = lambda L, a, b, p: g(L, a, b, p)['sharpe_day_equiv'] if g(L, a, b, p)['sharpe_day_equiv'] is not None else -99.  # noqa: E731
    cnt = dict(cells=len(big), cells_list=[f'{a}|{b}' for a, b in big])
    for L in LATS:
        for p in COMP:
            cnt[f'L{int(L)}_fixed_beats_{p}_pnl'] = int(sum(g(L, a, b, FIXED)['total_usd'] > g(L, a, b, p)['total_usd'] for a, b in big))
            cnt[f'L{int(L)}_fixed_beats_{p}_sharpe'] = int(sum(sh(L, a, b, FIXED) > sh(L, a, b, p) for a, b in big))
        for p in POLICIES:
            cnt[f'L{int(L)}_{p}_positive'] = int(sum(g(L, a, b, p)['total_usd'] > 0 for a, b in big))
    out['counts_early'] = cnt
    TR.dump(fn, out)
    print('parity max |diff| c', max(parity.values()), 'fixed', FIXED, 'cells', len(big))
    print({k: v for k, v in cnt.items() if k != 'cells_list'})


def selftest():
    # maker_fill2 = EM.maker_fill, with the indices of the filling prints
    lg = (np.array([100., 200., 300., 400., 500.]), np.array([1, 0, 1, 1, 1]), np.array([50., 50., 50., 51., 52.]), np.array([10., 9., 15., 5., 7.]),
          np.array([0, 0, 2, 1, 0]))
    for args in ((50., 1000., 1., 50., 20., 30.), (150., 1000., 1., 50., 0., 12.), (0., 1000., -1., 50., 0., 30.)):
        a, b = maker_fill2(lg, *args), EM.maker_fill(lg[:4], *args)
        assert a[:3] == b, (a, b)
    got, ts, qs, ks = maker_fill2(lg, 50., 1000., 1., 50., 20., 30.)
    assert ks == [2, 3] and qs == [5., 25.] and [lg[4][k] for k in ks] == [2, 1], (ks, qs)
    # take_exit_L / post_L: L = 0 is M2's; with L the exit match / the order's arrival moves L later
    fr = np.full(22, -1.); fr[7:12] = 0.; fr[17:22] = 0.
    fr[2], fr[7] = 48., 10.
    fr[12:17], fr[17:22] = [52., 53., 54., 55., 56.], [5., 6., 7., 8., 9.]
    f = np.stack([fr.copy() for _ in range(2)]); f[:, 0] = [0., 180.]; f[:, 1] = [0., 180.]
    f[1, 2], f[1, 7] = 47., 30.
    assert take_exit_L(f, f[:, 0], 1e9, 1., 5., 0., 0.) == M2.take_exit(f, f[:, 0], 1e9, 1., 5., 0.)
    assert take_exit_L(f, f[:, 0], 1e9, 1., 5., 0., 0.)[1] == 5 * 48. and take_exit_L(f, f[:, 0], 1e9, 1., 5., 0., 10.)[1] == 5 * 47.   # 172 vs 182 ms
    assert post_L(f, f[:, 0], f[:, 1], 1., 100., 'JOIN', 0.) == M2.post(f, f[:, 0], f[:, 1], 1., 100., 'JOIN')
    f2 = f.copy(); f2[1, 2] = 52.; f2[:, 0] = [0., 130.]                     # the venue bid reaches 52 at 130 ms
    assert not post_L(f2, f2[:, 0], f2[:, 1], 1., 100., 'JOIN', 0.)[2] and post_L(f2, f2[:, 0], f2[:, 1], 1., 100., 'JOIN', 10.)[2]   # lands 122 / 132
    # maker_loop_L at L = 0 = M3.maker_loop (its selftest scenario, CX_ANY and CX_OURS); the pre-move window and who
    fM = np.stack([fr.copy() for _ in range(2)]); fM[:, 0] = [0., 50000.]; fM[:, 1] = [0., 50000.]
    lgM = (np.array([110., 300., 520., 700., 2400.]) + M50, np.ones(5, int), np.full(5, 52.), np.array([10., 5., 4., 3., 20.]), np.array([0, 0, 2, 0, 0]))
    BN2 = M3.Binance(np.array([0., 500., 2200.]) - TM.FEED, np.array([10., 11., 12.]))
    for mode in ('CX_ANY', 'CX_OURS', 'NOCX'):
        r3 = M3.maker_loop(mode, 300., 'JOIN', 100., 10., 1., fM, fM[:, 0], fM[:, 1], lgM[:4], 0., 1e9, BN2)
        rL = maker_loop_L(mode, 300., 'JOIN', 100., 10., 1., fM, fM[:, 0], fM[:, 1], lgM, 0., 1e9, BN2, 0., 39000.)
        assert [(z[0], z[1], z[2], z[3]) for z in r3['incs']] == [(z['t'], z['q'], z['px'], z['flight']) for z in rL['incs']], (mode, r3, rL)
        assert (r3['decisions'], r3['cancels'], r3['refusals']) == (rL['decisions'], rL['cancels'], rL['refusals']), mode
    rL = maker_loop_L('CX_OURS', 300., 'JOIN', 100., 10., 1., fM, fM[:, 0], fM[:, 1], lgM, 0., 1e9, BN2, 0., 39000.)
    z = rL['incs'][0]                                                        # the 520 print: after the change at 498.5, the top taker
    assert z['flight'] and not z['pre'] and z['who'] == 2, z
    assert abs(rL['pre_ms'] - 2 * PRE_MS) < 1e-9, rL['pre_ms']              # windows (398.5, 498.5] and (2098.5, 2198.5], both resting
    assert abs(rL['rest_ms'] - ((524. - 122.) + (2224. - 822.) + (39122. - 2522.))) < 1e-9, rL['rest_ms']
    lgP = (np.array([450.]) + M50, np.ones(1, int), np.full(1, 52.), np.array([20.]), np.array([1]))   # a print 48.5 ms before the move
    rP = maker_loop_L('CX_OURS', 300., 'JOIN', 100., 10., 1., fM, fM[:, 0], fM[:, 1], lgP, 0., 1e9, BN2, 0., 39000.)
    assert len(rP['incs']) == 1 and rP['incs'][0]['pre'] and not rP['incs'][0]['flight'] and rP['incs'][0]['q'] == 10. and rP['incs'][0]['who'] == 1, rP
    assert abs(rP['pre_ms'] - (450. - 398.5)) < 1e-9 and abs(rP['rest_ms'] - (450. - 122.)) < 1e-9, rP   # filled at 450: stops resting
    # latency moves the cancel landing: a print at venue 540 fills only when the cancel (change known at 500) lands at 554 (L = 30), not 524
    lgQ = (np.array([300., 540.]) + M50, np.ones(2, int), np.full(2, 52.), np.array([5., 6.]), np.array([0, 0]))
    r0 = maker_loop_L('CX_ANY', 300., 'JOIN', 100., 10., 1., fM, fM[:, 0], fM[:, 1], lgQ, 0., 1e9, BN2, 0., 39000.)
    r30 = maker_loop_L('CX_ANY', 300., 'JOIN', 100., 10., 1., fM, fM[:, 0], fM[:, 1], lgQ, 0., 1e9, BN2, 30., 39000.)
    assert r0['incs'] == [] and len(r30['incs']) == 1 and r30['incs'][0]['q'] == 6. and r30['incs'][0]['flight'] and r30['t_first'] == 152., (r0, r30)
    # latency moves the arrival: a print at venue 140 (10 shares, 5 ahead) fills 5 of an order resting from 122 (L = 0), none from 152 (L = 30)
    lgR = (np.array([140.]) + M50, np.ones(1, int), np.full(1, 52.), np.array([10.]), np.array([0]))
    r0 = maker_loop_L('CX_ANY', 300., 'JOIN', 100., 10., 1., fM, fM[:, 0], fM[:, 1], lgR, 0., 1e9, BN2, 0., 39000.)
    r30 = maker_loop_L('CX_ANY', 300., 'JOIN', 100., 10., 1., fM, fM[:, 0], fM[:, 1], lgR, 0., 1e9, BN2, 30., 39000.)
    assert [z['q'] for z in r0['incs']] == [5.] and r30['incs'] == [], (r0, r30)
    try:
        labels('20261022', '/nonexistent')
        raise AssertionError('an undeclared day accepted')
    except ValueError as e_:
        assert 'not a declared day' in str(e_), e_
    print('exit_fixed selftest PASS')


if __name__ == '__main__':
    cmd, a_ = sys.argv[1], sys.argv[2:]
    if cmd == 'labels':
        labels(a_[0], a_[1] if len(a_) > 1 else None)
    elif cmd == 'read':
        read(a_[0] if a_ else None)
    else:
        dict(selftest=selftest)[cmd]()
