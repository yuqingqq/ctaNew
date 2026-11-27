"""Chase by taking, exit with a maker (post-only) order (USER 2026-11-27: "You mean we exit by taking again? Can we chase Binance move and post
maker order to exit"). Four consumed days, every market, no orders; post hoc on these days (entries and comparators from EXIT_FOLLOW, its
shared policies reproduced: a check); declared before the run.
Entries: EXIT_FOLLOW's (FILL_AWARE's selections, a 50-share IOC at the known best + 3c / + 10c, optimistic and conservative entry books).
Maker exit: we learn the fill 22 ms after the match and our post-only order rests 22 ms later (no speed bump for post-only orders, no maker
fee); one order for the whole filled size, never moved. Price (on the exit side: an ask for a long UP, a bid on UP for a long DOWN), from
the book known when it is sent: JOIN = the best price on that side (queue ahead = its shown size); IMP = one tick inside if the spread is
>= 2c, else JOIN (queue ahead 0 inside); TGT2 / TGT5 / TGT10 = our entry price + 2 / 5 / 10c on our side (whole cent, at most 99c), not
crossing (at least one tick from the other side), queue ahead = the shown size at that price if shown, else 0 (beyond the five shown levels
the queue is unknown: counted). TGT5 / TGT10 added after the one-day rehearsal (JOIN / IMP filled within ~50 ms at about our entry price,
the side's mid 6-10c higher 5 s later: selling back into the chase), declared before the real run.
Fills from Predict's prints after the order rests (winners_legs: taker side and UP-space price; a leg without a time takes its order's
earliest time; legs still without one are dropped, counted per day): prints at our price consume the queue ahead, then us; a print beyond
our price (a buyer paying more than our ask, a seller taking less than our bid) means our level was cleared and fills the rest. Cancels
ahead of us are not credited (pessimistic). Deadline 10 s or 39 s after resting (and 30 s before the window's end); the rest then TAKE
(sold by taking, EXIT_FOLLOW's rule), HOLD (to settlement) or FAV (EXIT_FOLLOW's favourite check at that time: kept if the side's mid >= 60c).
Policies: {JOIN, IMP, TGT} x {T10_TAKE, T39_TAKE, T39_HOLD, T39_FAV}; comparators reproduced from EXIT_FOLLOW: HOLD, TB1S, FAV_ONLY.
Readings: P&L ($, per intended share, by day, days > 0), the per-window Sharpe (SHARPE_EXITS' unit: every 5-minute window, day equivalent)
and max drawdown, the share of the filled entry closed by the maker order, its median wait, and the side's mid 5 s after a maker fill
against our price (what the buyer of our order got).
Outputs (write-once): OUT/exit_maker/<day>.parquet, exit_maker.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/exit_maker.py \
       selftest | labels <day> [<dir>] | read [<dir>]
"""
import collections
import json
import math
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import exit_follow as E  # noqa: E402
import sharpe_exits as SX  # noqa: E402

CT, C, TR, TK, TE, TM, WT, F, VR, MM = E.CT, E.C, E.TR, E.TK, E.TE, E.TM, E.WT, E.F, E.VR, E.MM
OUT = WT.OUT / 'exit_maker'
DAYS, Q = E.DAYS, E.Q
LEARN = 22.
PRICES = ('JOIN', 'IMP', 'TGT2', 'TGT5', 'TGT10')
TAILS = (('T10_TAKE', 10000., 'TAKE'), ('T39_TAKE', 39000., 'TAKE'), ('T39_HOLD', 39000., 'HOLD'), ('T39_FAV', 39000., 'FAV'))
MAKER = [f'{p}_{t}' for p in PRICES for t, _, _ in TAILS]
REPRO = ('HOLD', 'TB1S', 'FAV_ONLY')
POLICIES = REPRO + tuple(MAKER)


def legs_of(day):
    """{market: (times ms, taker side 1 = bought UP, UP price c, shares)} sorted by time; untimed legs take their order's earliest time."""
    L = pd.read_parquet(WT.OUT / f'winners_legs_{day}.parquet', columns=['mid', 'taker_hash', 'side', 'ptick', 'shares', 't_ms2'])
    t = L.t_ms2.fillna(L.groupby('taker_hash').t_ms2.transform('min'))
    st = dict(legs=int(len(L)), untimed_legs=int(t.isna().sum()), untimed_share_of_shares=float(L.shares[t.isna()].sum() / L.shares.sum()))
    L = L.assign(t=t)[t.notna()].sort_values('t', kind='stable')
    return {int(m): (g.t.to_numpy(float), g.side.to_numpy(int), g.ptick.to_numpy(float), g.shares.to_numpy(float)) for m, g in L.groupby('mid')}, st


def maker_fill(lg, t_rest, t_end, d, p, ahead, size):
    """Fills of our resting order (an ask at p for d > 0, a bid at p for d < 0) between t_rest and t_end (epoch ms) -> (filled, times of
    the fill increments, sizes)."""
    if lg is None or size <= 0:
        return 0., [], []
    tl, sd, pr, sh = lg
    a, b = np.searchsorted(tl, t_rest, 'right'), np.searchsorted(tl, t_end, 'right')
    want = 1 if d > 0 else 0                                    # a taker buying UP lifts our ask; a taker selling UP hits our bid
    q_ahead, got, ts, qs = ahead, 0., [], []
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
            got += x; ts.append(float(tl[k])); qs.append(float(x))
        if got >= size - 1e-9:
            break
    return got, ts, qs


def labels(day, od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (1500, 1501))
    C.require(day in DAYS, f'{day} is not a declared day')
    od = Path(od) if od else OUT
    fp = od / f'{day}.parquet'
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    S = E.selections()
    S = S[S.day == day].reset_index(drop=True)
    L = pd.read_parquet(CT.OUT / f'{day}_labels.parquet', columns=['t', 'd', 'mid', 'lim', 'payout'])
    S = S.merge(L, on=['t', 'd'], how='left', validate='1:1')
    LG, lst = legs_of(day)
    reg = MM.registry(day).set_index('mid')
    rg = pd.read_parquet(MM.S1 / day / 'registry.parquet').set_index('mid')
    by_mid = S.groupby('mid').indices
    o = collections.defaultdict(lambda: np.full(len(S), np.nan))
    src = MM.KP.Tape(); off = src.offset(day)
    for h, g in reg.groupby('hour'):
        mids = [int(m) for m in g.index if int(m) in by_mid]
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
                act = t + TM.FEED + TM.DECIDE - t0 + E.NET + E.HOLD
                jv = int(np.searchsorted(f0, act, 'right') - 1)
                side = 1 if d > 0 else 0
                pa, sa = TK.side_levels(f[jv], side)
                pb, sb_ = TK.side_levels(f[jv + 1], side) if jv + 1 < len(f) else (np.full(5, -1.), np.zeros(5))
                bk = {'A': (pa[None], sa[None]), 'AB': TK.minbook(pa[None], sa[None], pb[None], sb_[None])}
                k0 = int(np.searchsorted(fk, act, 'right'))
                kt, sm = fk[k0:], (midy[k0:] if d > 0 else 100. - midy[k0:])
                t_rest = act + 2. * LEARN
                jr = int(np.searchsorted(fk, t_rest, 'right') - 1)
                bid, ask = f[jr, 2], f[jr, 12]
                ok_book = okb[jr]
                for b, s in E.COMBOS:
                    pre = f'{b}_s{int(s)}'
                    fl, co, fe_ = (x[0] for x in TK.vwalk(*bk[b], np.array([d]), np.array([lim + d * s])))
                    o[f'{pre}_fill'][i] = fl
                    if fl <= 0:
                        for pol in POLICIES:
                            o[f'{pre}_{pol}_pnl'][i] = 0.
                        continue
                    e = co / fl if d > 0 else 100. - co / fl
                    for pol in REPRO:                                       # EXIT_FOLLOW's own decisions and exit rule
                        x = E.decide(pol, E.trigger(kt, sm, e, cut), E.mid_at(kt, sm, act + E.T_MED), E.mid_at(kt, sm, act + E.T_UQ), act, cut)
                        xq = xc = xf = 0.
                        if x is not None and x + E.DEC + E.NET + E.HOLD <= end:
                            je = int(np.searchsorted(f0, x + E.DEC + E.NET + E.HOLD, 'right') - 1)
                            eP, eS = TK.side_levels(f[je], 0 if d > 0 else 1)
                            r_ = TE.walk_q(eP[None], eS[None], np.array([-d]), np.array([1. if d > 0 else 99.]), np.array([fl]))
                            xq, xc, xf = (float(z[0]) for z in r_)
                        o[f'{pre}_{pol}_pnl'][i] = d * (xc + (fl - xq) * 100. * pay - co) - fe_ - xf
                    for pr_ in PRICES:
                        if not ok_book:
                            px, ahead = np.nan, 0.
                        elif pr_ == 'JOIN':
                            px, ahead = (ask, f[jr, 17]) if d > 0 else (bid, f[jr, 7])
                        elif pr_ == 'IMP':
                            if ask - bid >= 2. - 1e-9:
                                px, ahead = (ask - 1., 0.) if d > 0 else (bid + 1., 0.)
                            else:
                                px, ahead = (ask, f[jr, 17]) if d > 0 else (bid, f[jr, 7])
                        else:
                            tgt = min(math.ceil(e + float(pr_[3:]) - 1e-9), 99)
                            if d > 0:
                                px = max(float(tgt), bid + 1.)
                                lv = np.flatnonzero(np.isclose(f[jr, 12:17], px))
                                ahead = float(f[jr, 17 + lv[0]]) if len(lv) else 0.
                                deepest = f[jr, 12:17][f[jr, 12:17] >= 0].max() if (f[jr, 12:17] >= 0).any() else -1.
                                o[f'{pre}_{pr_}_beyond_shown'][i] = float(px > deepest)
                            else:
                                px = min(100. - tgt, ask - 1.)
                                lv = np.flatnonzero(np.isclose(f[jr, 2:7], px))
                                ahead = float(f[jr, 7 + lv[0]]) if len(lv) else 0.
                                shown = f[jr, 2:7][f[jr, 2:7] >= 0]
                                o[f'{pre}_{pr_}_beyond_shown'][i] = float(px < shown.min()) if len(shown) else 1.
                        o[f'{pre}_{pr_}_price_its_way'][i] = (px if d > 0 else 100. - px) - e if np.isfinite(px) else np.nan
                        for tl_, T_, tail in TAILS:
                            dl = min(t_rest + T_, cut)
                            got, ts, qs = (maker_fill(lg, t0 + t_rest, t0 + dl, d, px, ahead, fl) if (np.isfinite(px) and dl > t_rest) else (0., [], []))
                            xc_m = got * px if got > 0 else 0.
                            rest = fl - got
                            xq = xc = xf = 0.
                            if rest > 1e-9 and dl > t_rest:
                                sell = tail == 'TAKE'
                                if tail == 'FAV':
                                    mT = E.mid_at(kt, sm, dl)
                                    sell = bool(np.isfinite(mT) and mT < E.FAV)
                                mt = dl + E.DEC + E.NET + E.HOLD
                                if sell and mt <= end:
                                    je = int(np.searchsorted(f0, mt, 'right') - 1)
                                    eP, eS = TK.side_levels(f[je], 0 if d > 0 else 1)
                                    r_ = TE.walk_q(eP[None], eS[None], np.array([-d]), np.array([1. if d > 0 else 99.]), np.array([rest]))
                                    xq, xc, xf = (float(z[0]) for z in r_)
                            pnl = d * (xc_m + xc + (rest - xq) * 100. * pay - co) - fe_ - xf
                            key = f'{pre}_{pr_}_{tl_}'
                            o[f'{key}_pnl'][i] = pnl
                            o[f'{key}_maker_share'][i] = got / fl
                            if got > 0:
                                o[f'{key}_maker_wait_ms'][i] = float(np.average(np.array(ts) - (t0 + t_rest), weights=qs))
                                m5 = E.mid_at(kt, sm, float(ts[-1]) - t0 + 5000.)
                                o[f'{key}_after_fill_5s_c'][i] = (m5 - (px if d > 0 else 100. - px)) if np.isfinite(m5) else np.nan
            C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    out = S[['day', 't', 'd', 'key', 'mid'] + list(E.SELS)].copy()
    for k_, v in o.items():
        out[k_] = v
    od.mkdir(parents=True, exist_ok=True)
    out.to_parquet(fp, index=False)
    TR.dump(od / f'{day}.json', dict(asof='2026-11-27', day=day, actions=int(len(S)), legs=lst))
    print(day, len(S), lst, 'rss', round(F.rss_mib()), flush=True)


def read(od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (600, 601))
    reh = od is not None
    od = Path(od) if od else None
    fn = (od or H) / 'exit_maker.json'
    C.require(not fn.exists(), 'exit_maker.json exists (write-once)')
    src = od or OUT
    present = [dh for dh in DAYS if (src / f'{dh}.parquet').exists()]
    X = pd.concat([pd.read_parquet(src / f'{dh}.parquet') for dh in present], ignore_index=True)
    X1 = pd.concat([pd.read_parquet(E.OUT / f'{dh}.parquet') for dh in present], ignore_index=True)
    C.require(len(X) == len(X1) and (X.t.to_numpy() == X1.t.to_numpy()).all(), 'the actions differ from EXIT_FOLLOW')
    for b, s in E.COMBOS:
        for pol in REPRO:
            k = f'{b}_s{int(s)}_{pol}_pnl'
            C.require(np.allclose(X[k].to_numpy(float), X1[k].to_numpy(float), rtol=0, atol=1e-9), f'{k} not reproduced from EXIT_FOLLOW')
    keys, dayk = [], []
    for d in present:
        reg = MM.registry(d).sort_values('ws')
        keys += [f'{d}:{int(m)}' for m in reg.mid]
        dayk += [d] * len(reg)
    keys, dayk = np.array(keys), np.array(dayk)
    wk = (X.day.astype(str) + ':' + X.mid.astype(np.int64).astype(str)).to_numpy()
    day = X.day.to_numpy()
    out = dict(asof='2026-11-27', post_hoc=True, rehearsal=reh, days=present, legs={dh: json.loads((src / f'{dh}.json').read_text())['legs'] for dh in present}, selections={})
    for sel in E.SELS:
        mm = X[sel].to_numpy(bool)
        ii = np.flatnonzero(mm)
        so = dict(actions=int(len(ii)), cells={})
        for b, s in E.COMBOS:
            pre = f'{b}_s{int(s)}'
            fl = X[f'{pre}_fill'].to_numpy(float)
            filled = ii[fl[ii] > 0]
            for pol in POLICIES:
                pn = X[f'{pre}_{pol}_pnl'].to_numpy(float) / 100.
                st = SX.stats(SX.window_series(pn[mm], wk[mm], keys), dayk, len(keys) / len(present))
                c = dict(total_usd=st['total_usd'], per_intended_c=float(pn[ii].sum() * 100. / (Q * len(ii))), sharpe_day_equiv=st['sharpe_day_equiv'],
                         max_drawdown_usd=st['max_drawdown_usd'], daily_usd=st['daily_usd'], days_positive=int(sum(v > 0 for v in st['daily_usd'])))
                if pol in MAKER:
                    pr_ = pol.split('_')[0]
                    if pr_.startswith('TGT') and f'{pre}_{pr_}_beyond_shown' in X:
                        c['target_beyond_shown_depth'] = float(np.nanmean(X[f'{pre}_{pr_}_beyond_shown'].to_numpy(float)[filled]))
                    ms = X[f'{pre}_{pol}_maker_share'].to_numpy(float)[filled]
                    c.update(maker_closed_share=float(np.nansum(ms * fl[filled]) / fl[filled].sum()) if len(filled) else None,
                             maker_wait_s_median=float(np.nanmedian(X[f'{pre}_{pol}_maker_wait_ms'].to_numpy(float)[filled]) / 1000.) if np.isfinite(X[f'{pre}_{pol}_maker_wait_ms'].to_numpy(float)[filled]).any() else None,
                             after_fill_5s_c=float(np.nanmean(X[f'{pre}_{pol}_after_fill_5s_c'].to_numpy(float)[filled])) if np.isfinite(X[f'{pre}_{pol}_after_fill_5s_c'].to_numpy(float)[filled]).any() else None)
                so['cells'][f'{pre}_{pol}'] = c
        out['selections'][sel] = so
    TR.dump(fn, out)
    for sel, so in out['selections'].items():
        for pre in ('AB_s3', 'AB_s10'):
            row = {pol: (round(so['cells'][f'{pre}_{pol}']['total_usd']), round(so['cells'][f'{pre}_{pol}']['sharpe_day_equiv'] or 0, 2)) for pol in POLICIES}
            print(sel, pre, row)


def selftest():
    lg = (np.array([100., 200., 300., 400.]), np.array([1, 1, 0, 1]), np.array([50., 50., 49., 51.]), np.array([10., 15., 20., 5.]))
    got, ts, qs = maker_fill(lg, 50., 1000., 1., 50., 20., 30.)             # ask at 50, 20 ahead: 10 + 15 at 50 -> 5 to us; 51 clears the rest
    assert got == 30. and qs == [5., 25.] and ts == [200., 400.], (got, ts, qs)
    got, ts, qs = maker_fill(lg, 150., 350., 1., 50., 0., 30.)              # rests after the first print: 15 at 50 -> 15
    assert got == 15. and ts == [200.], (got, ts, qs)
    got, _, _ = maker_fill(lg, 50., 1000., -1., 49., 0., 30.)               # a bid at 49 (long DOWN): the seller at 49 fills 20
    assert got == 20., got
    got, _, _ = maker_fill(lg, 50., 1000., -1., 50., 0., 30.)               # a bid at 50: the seller at 49 went beyond our bid -> cleared: all
    assert got == 30., got
    assert maker_fill(None, 0., 1., 1., 50., 0., 30.)[0] == 0.
    try:
        labels('20261022', '/nonexistent')
        raise AssertionError('an undeclared day accepted')
    except ValueError as e_:
        assert 'not a declared day' in str(e_), e_
    print('exit_maker selftest PASS')


if __name__ == '__main__':
    cmd, a_ = sys.argv[1], sys.argv[2:]
    if cmd == 'labels':
        labels(a_[0], a_[1] if len(a_) > 1 else None)
    elif cmd == 'read':
        read(a_[0] if a_ else None)
    else:
        dict(selftest=selftest)[cmd]()
