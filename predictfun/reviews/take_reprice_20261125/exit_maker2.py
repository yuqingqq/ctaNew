"""Chase by taking; when Binance calms down, exit with a maker order (USER 2026-11-27: "Can we chase Binance move and post maker order to
exit" and "We are not going to post maker exit immediately, we wait for Binance to calm down to start to exit"). Four consumed days, every
market, no orders; POST HOC on these days (entries, comparators, tails and fill rule from EXIT_FOLLOW / EXIT_MAKER); declared before this run.
Supersedes EXIT_MAKER (exit_maker.py b9ceecd51 / 23dae007c, exit_maker.json), which is VOID: (1) it timed Predict's prints on the receipt
clock (t_ms2) against our order's venue-clock arrival, but the replay's trade clock is venue = t_ms2 - 50 ms (about +-30 ms), so prints made
up to 50 ms BEFORE our order arrived filled it (its median wait was 30-50 ms); (2) it priced the order and read the queue ahead from the
book known when the order rests, 22 ms after it was sent; (3) it had no post-only check: an order that would cross the venue's book on
arrival is refused, but later prints beyond its price filled it.
Entries: EXIT_FOLLOW's (FILL_AWARE's selections, a 50-share IOC at the known best + 3c / + 10c, optimistic A and conservative AB entry books).
Calm: the first time, from when our fill is known (22 ms after the match), with no Binance futures mid change known (exchange time + 1.5 ms
feed) in the last G = 0.3 s (the move definition's gap), 1 s or 3 s; the input-only look before this run: 0.4-1.7 mid changes a second, the
median wait from the fill notice to calm 0.14 / 0.9 / 3.1 s. Exit at calm: a post-only order (no speed bump, no maker fee) rests 22 ms after
the decision, one order for the whole filled size, never moved; price from the book known at the decision, on the exit side (an ask for a
long UP, a bid on UP for a long DOWN): JOIN = its best price; IMP = one tick inside if the spread is >= 2c, else JOIN. From the venue's book
at arrival (the last frame at or before it): refused if it would cross (an ask at or below the best bid, a bid at or above the best ask; it
rests nowhere and the position goes to the tail at the deadline; counted); queue ahead = the shown size at our price, 0 inside the spread
or at an unshown price within the shown levels, 0 beyond the deepest of five shown levels (unknown: counted). Fills from Predict's prints at
venue time t_ms2 - 50 ms after arrival (EXIT_MAKER's queue rule: prints at our price consume the queue ahead, then us; a print beyond our
price clears the rest; cancels ahead not credited). Deadline 10 s or 39 s after resting and 30 s before the window's end; the rest then
TAKE (sold by taking, EXIT_FOLLOW's rule), HOLD or FAV (taken if the side's mid known then is < 60c). Nothing is sent after that cut (the
position is held), for the maker and the taking exit alike.
Comparators: TAKE_C<G> = everything sold by taking at the same calm decision (maker vs taker at the same time); HOLD, TB1S, FAV_ONLY
reproduced from EXIT_FOLLOW (a check). References: EXIT_MAKER's JOIN_T10_TAKE reproduced bit for bit (EM: a check of the shared code) and
D0_JOIN_T10_TAKE = the immediate post (decided when the fill is known) with the three corrections (not the USER's design; shows what the
corrections change). Clock sensitivity: JOIN / IMP with the 10 s take tail at every G with prints at t_ms2 - 20 / - 80 ms.
Scopes: ALL = every selected entry (the calm family and its taking comparator obey the same cut); EARLY = entries matched >= 15.022 s before
the cut (a decision-time rule) for comparisons with TB1S / FAV_ONLY / HOLD, which act on late entries differently.
Readings: P&L ($, per intended share, by day, days > 0), the per-window Sharpe (SHARPE_EXITS' unit, day equivalent), max drawdown; for maker
policies the share of the filled entry closed by the maker order, its median wait, the sent / refused / beyond-shown shares, our price vs
entry, and the side's mid 5 s after a maker fill against our price (what the buyer got); per G the wait to calm and the side's mid change
over the 5 s after calm (is Predict still repricing when Binance calms). Untimed prints take their order's earliest time; still untimed are
dropped (counted per day).
Outputs (write-once): OUT/exit_maker2/<day>.parquet + .json, exit_maker2.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/exit_maker2.py \
       selftest | labels <day> [<dir>] | read [<dir>]
"""
import collections
import gc
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
import sharpe_exits as SX  # noqa: E402

CT, C, TR, TK, TE, TM, WT, F, VR, MM = E.CT, E.C, E.TR, E.TK, E.TE, E.TM, E.WT, E.F, E.VR, E.MM
OUT = WT.OUT / 'exit_maker2'
DAYS, Q = E.DAYS, E.Q
LEARN, M50, SHIFTS = E.NET, WT.M50, (20., 80.)
GAPS = (300., 1000., 3000.)
PRICES = ('JOIN', 'IMP')
TAILS = (('T10_TAKE', 10000., 'TAKE'), ('T39_TAKE', 39000., 'TAKE'), ('T39_HOLD', 39000., 'HOLD'), ('T39_FAV', 39000., 'FAV'))
MAKER = [f'C{int(g)}_{p}_{t}' for g in GAPS for p in PRICES for t, _, _ in TAILS]
SENS = [f'C{int(g)}_{p}_T10_TAKE_M{int(s)}' for g in GAPS for p in PRICES for s in SHIFTS]
TAKE_C = tuple(f'TAKE_C{int(g)}' for g in GAPS)
COMP = ('HOLD', 'TB1S', 'FAV_ONLY') + TAKE_C
REF = ('EM_JOIN_T10_TAKE', 'D0_JOIN_T10_TAKE')
POLICIES = COMP + REF + tuple(MAKER) + tuple(SENS)
EARLY_MS = 5000. + LEARN + 10000.
STRAT = [s for s in E.SELS if not s.startswith('CONTROL')]


def calm_times(kn, xmin, G):
    """First x >= xmin with no change known in (x - G, x] (kn = sorted known times of Binance mid changes) -> x per xmin."""
    xmin = np.asarray(xmin, float)
    idx = np.searchsorted(kn, xmin, 'right') - 1
    big = np.flatnonzero(np.r_[kn[1:], np.inf] - kn > G)
    j = np.searchsorted(big, np.maximum(idx, 0), 'left')
    x = np.full(len(xmin), np.inf)
    ok = j < len(big)
    x[ok] = np.maximum(kn[big[j[ok]]] + G, xmin[ok])
    return np.where(idx < 0, xmin, x)


def take_exit(f, f0, end, d, q, x):
    """Sell q shares by taking, decided at x (path clock): EXIT_FOLLOW's rule (the IOC matches DEC + NET + HOLD later, at every shown level
    of the snapshot before the match) -> (sold, UP-space value, fee); nothing if x is None, q <= 0 or the match falls beyond the path."""
    if x is None or q <= 1e-9:
        return 0., 0., 0.
    mt = x + E.DEC + E.NET + E.HOLD
    if mt > end:
        return 0., 0., 0.
    je = int(np.searchsorted(f0, mt, 'right') - 1)
    eP, eS = TK.side_levels(f[je], 0 if d > 0 else 1)
    r_ = TE.walk_q(eP[None], eS[None], np.array([-d]), np.array([1. if d > 0 else 99.]), np.array([q]))
    return tuple(float(z[0]) for z in r_)


def price_of(fr, d, pr_):
    """Our post-only price (UP space) from the known frame fr, or nan without a valid two-sided book."""
    bid, ask = fr[2], fr[12]
    if not (bid >= 0 and 0 <= ask <= 100 and bid < ask):
        return np.nan
    if pr_ == 'IMP' and ask - bid >= 2. - 1e-9:
        return float(ask - 1. if d > 0 else bid + 1.)
    return float(ask if d > 0 else bid)


def venue_queue(fr, d, px):
    """Our order at px against the venue frame at arrival -> (refused, queue ahead, beyond the deepest of five shown levels)."""
    if d > 0:
        opp, P, S = fr[2], fr[12:17], fr[17:22]
        if opp >= 0 and px <= opp + 1e-9:
            return True, 0., False
    else:
        opp, P, S = fr[12], fr[2:7], fr[7:12]
        if opp >= 0 and px >= opp - 1e-9:
            return True, 0., False
    v = P >= 0
    if not v.any():
        return False, 0., False
    hit = np.flatnonzero(v & np.isclose(P, px))
    if len(hit):
        return False, float(S[hit[0]]), False
    deeper = (px > P[v].max() + 1e-9) if d > 0 else (px < P[v].min() - 1e-9)
    return False, 0., bool(v.sum() == 5 and deeper)


def post(f, f0, fk, d, dec, pr_):
    """The decision at dec (path clock) -> (price, resting time, refused, queue ahead, beyond shown); price from the frame known at dec,
    refusal and queue from the venue frame at arrival."""
    t_rest = dec + E.DEC + LEARN
    jk = int(np.searchsorted(fk, dec, 'right') - 1)
    px = price_of(f[jk], d, pr_) if jk >= 0 else np.nan
    if not np.isfinite(px):
        return np.nan, t_rest, False, 0., False
    ja = int(np.searchsorted(f0, t_rest, 'right') - 1)
    return (px, t_rest) + venue_queue(f[ja], d, px)


def tail_x(tail, kt, sm, dl):
    if tail == 'TAKE':
        return dl
    if tail == 'FAV':
        m = E.mid_at(kt, sm, dl)
        return dl if (np.isfinite(m) and m < E.FAV) else None
    return None


def labels(day, od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (2400, 2401))
    C.require(day in DAYS, f'{day} is not a declared day')
    od = Path(od) if od else OUT
    fp = od / f'{day}.parquet'
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    S = E.selections()
    S = S[S.day == day].reset_index(drop=True)
    L = pd.read_parquet(CT.OUT / f'{day}_labels.parquet', columns=['t', 'd', 'mid', 'lim', 'payout'])
    S = S.merge(L, on=['t', 'd'], how='left', validate='1:1')
    Eb, _ = CT.mid_series(day)
    kn = Eb.astype(float) + TM.FEED
    xmin = S.t.to_numpy(float) + TM.FEED + TM.DECIDE + E.NET + E.HOLD + LEARN          # our fill known (epoch ms)
    CALM = {g: calm_times(kn, xmin, g) for g in GAPS}
    n_changes = int(len(kn))
    del Eb, kn
    gc.collect()
    LG, lst = EM.legs_of(day)
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
                o['cut_minus_act_ms'][i] = cut - act
                jv = int(np.searchsorted(f0, act, 'right') - 1)
                side = 1 if d > 0 else 0
                pa, sa = TK.side_levels(f[jv], side)
                pb, sb_ = TK.side_levels(f[jv + 1], side) if jv + 1 < len(f) else (np.full(5, -1.), np.zeros(5))
                bk = {'A': (pa[None], sa[None]), 'AB': TK.minbook(pa[None], sa[None], pb[None], sb_[None])}
                k0 = int(np.searchsorted(fk, act, 'right'))
                kt, sm = fk[k0:], (midy[k0:] if d > 0 else 100. - midy[k0:])
                calm = {gg: float(CALM[gg][i]) - t0 for gg in GAPS}
                for gg in GAPS:
                    o[f'C{int(gg)}_wait_ms'][i] = calm[gg] - (act + LEARN)
                    o[f'C{int(gg)}_side_mid_5s_change_c'][i] = E.mid_at(kt, sm, calm[gg] + 5000.) - E.mid_at(kt, sm, calm[gg]) if np.isfinite(calm[gg]) else np.nan
                for b, s in E.COMBOS:
                    pre = f'{b}_s{int(s)}'
                    fl, co, fe_ = (x[0] for x in TK.vwalk(*bk[b], np.array([d]), np.array([lim + d * s])))
                    o[f'{pre}_fill'][i] = fl
                    if fl <= 0:
                        for pol in POLICIES:
                            o[f'{pre}_{pol}_pnl'][i] = 0.
                        continue
                    e = co / fl if d > 0 else 100. - co / fl

                    def pnl(mq, mc, tq, tc, tf):
                        return d * (mc + tc + (fl - mq - tq) * 100. * pay - co) - fe_ - tf
                    for pol in ('HOLD', 'TB1S', 'FAV_ONLY'):
                        x = E.decide(pol, E.trigger(kt, sm, e, cut), E.mid_at(kt, sm, act + E.T_MED), E.mid_at(kt, sm, act + E.T_UQ), act, cut)
                        o[f'{pre}_{pol}_pnl'][i] = pnl(0., 0., *take_exit(f, f0, end, d, fl, x))
                    # EXIT_MAKER's JOIN_T10_TAKE reproduced (EM); the immediate post with the three corrections (D0)
                    tr0 = act + 2. * LEARN
                    jr = int(np.searchsorted(fk, tr0, 'right') - 1)
                    pxe, ahe = ((f[jr, 12], f[jr, 17]) if d > 0 else (f[jr, 2], f[jr, 7])) if okb[jr] else (np.nan, 0.)
                    dl = min(tr0 + 10000., cut)
                    got = EM.maker_fill(lg, t0 + tr0, t0 + dl, d, pxe, ahe, fl)[0] if (np.isfinite(pxe) and dl > tr0) else 0.
                    o[f'{pre}_EM_JOIN_T10_TAKE_pnl'][i] = pnl(got, got * pxe if got > 0 else 0., *take_exit(f, f0, end, d, fl - got, dl if dl > tr0 else None))
                    runs = [('D0', act + LEARN, 'JOIN', [('T10_TAKE', 10000., 'TAKE', M50, 'D0_JOIN_T10_TAKE')])]
                    for gg in GAPS:
                        dec = calm[gg]
                        o[f'{pre}_TAKE_C{int(gg)}_pnl'][i] = pnl(0., 0., *take_exit(f, f0, end, d, fl, dec if dec <= cut else None))
                        for pr_ in PRICES:
                            rr = [(tl_, T_, tail, M50, f'C{int(gg)}_{pr_}_{tl_}') for tl_, T_, tail in TAILS]
                            rr += [('T10_TAKE', 10000., 'TAKE', sh, f'C{int(gg)}_{pr_}_T10_TAKE_M{int(sh)}') for sh in SHIFTS]
                            runs.append((f'C{int(gg)}', dec, pr_, rr))
                    for nm, dec, pr_, rr in runs:
                        if np.isfinite(dec):
                            px, t_rest, refused, ahead, beyond = post(f, f0, fk, d, dec, pr_)
                        else:
                            px, t_rest, refused, ahead, beyond = np.nan, np.inf, False, 0., False
                        sent = bool(np.isfinite(px) and t_rest < cut)
                        o[f'{pre}_{nm}_{pr_}_sent'][i] = float(sent)
                        o[f'{pre}_{nm}_{pr_}_refused'][i] = float(sent and refused)
                        o[f'{pre}_{nm}_{pr_}_beyond_shown'][i] = float(sent and beyond)
                        o[f'{pre}_{nm}_{pr_}_price_its_way'][i] = ((px if d > 0 else 100. - px) - e) if np.isfinite(px) else np.nan
                        for tl_, T_, tail, M, key in rr:
                            dl = min(t_rest + T_, cut)
                            can = dl > t_rest
                            live = can and np.isfinite(px) and not refused
                            got, ts, qs = EM.maker_fill(lg, t0 + t_rest + M, t0 + dl + M, d, px, ahead, fl) if live else (0., [], [])
                            x = tail_x(tail, kt, sm, dl) if (can and fl - got > 1e-9) else None
                            o[f'{pre}_{key}_pnl'][i] = pnl(got, got * px if got > 0 else 0., *take_exit(f, f0, end, d, fl - got, x))
                            o[f'{pre}_{key}_maker_share'][i] = got / fl
                            if got > 0:
                                o[f'{pre}_{key}_maker_wait_ms'][i] = float(np.average(np.array(ts) - M - (t0 + t_rest), weights=qs))
                                o[f'{pre}_{key}_maker_first_ms'][i] = float(ts[0]) - M - (t0 + t_rest)
                                m5 = E.mid_at(kt, sm, float(ts[-1]) - M - t0 + 5000.)
                                o[f'{pre}_{key}_after_fill_5s_c'][i] = (m5 - (px if d > 0 else 100. - px)) if np.isfinite(m5) else np.nan
            C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    out = S[['day', 't', 'd', 'key', 'mid'] + list(E.SELS)].copy()
    out = pd.concat([out, pd.DataFrame(dict(o), index=out.index)], axis=1)
    od.mkdir(parents=True, exist_ok=True)
    out.to_parquet(fp, index=False)
    TR.dump(od / f'{day}.json', dict(asof='2026-11-27', day=day, actions=int(len(S)), legs=lst, binance_mid_changes=n_changes))
    print(day, len(S), lst, 'rss', round(F.rss_mib()), flush=True)


def read(od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    reh = od is not None
    od = Path(od) if od else None
    fn = (od or H) / 'exit_maker2.json'
    C.require(not fn.exists(), 'exit_maker2.json exists (write-once)')
    src = od or OUT
    present = [dh for dh in DAYS if (src / f'{dh}.parquet').exists()]
    X = pd.concat([pd.read_parquet(src / f'{dh}.parquet') for dh in present], ignore_index=True)
    X1 = pd.concat([pd.read_parquet(E.OUT / f'{dh}.parquet') for dh in present], ignore_index=True)
    XM = pd.concat([pd.read_parquet(EM.OUT / f'{dh}.parquet') for dh in present], ignore_index=True)
    C.require(len(X) == len(X1) == len(XM) and (X.t.to_numpy() == X1.t.to_numpy()).all() and (X.t.to_numpy() == XM.t.to_numpy()).all(),
              'the actions differ from EXIT_FOLLOW / EXIT_MAKER')
    parity = {}
    for b, s in E.COMBOS:
        pre = f'{b}_s{int(s)}'
        for pol, ref, rk in (('HOLD', X1, 'HOLD'), ('TB1S', X1, 'TB1S'), ('FAV_ONLY', X1, 'FAV_ONLY'), ('EM_JOIN_T10_TAKE', XM, 'JOIN_T10_TAKE')):
            a, r = X[f'{pre}_{pol}_pnl'].to_numpy(float), ref[f'{pre}_{rk}_pnl'].to_numpy(float)
            C.require(np.allclose(a, r, rtol=0, atol=1e-9), f'{pre}_{pol} not reproduced')
            parity[f'{pre}_{pol}'] = float(np.max(np.abs(a - r)))
        C.require(np.allclose(X[f'{pre}_fill'].to_numpy(float), X1[f'{pre}_fill'].to_numpy(float), rtol=0, atol=1e-12), f'{pre} fills differ')
    for b, s in E.COMBOS:                                                   # impossible states: a fill before arrival, a refused order filled
        pre = f'{b}_s{int(s)}'
        for pol in MAKER + SENS + ['D0_JOIN_T10_TAKE']:
            nm, pr_ = pol.split('_')[:2]
            fm = X[f'{pre}_{pol}_maker_first_ms'].to_numpy(float) if f'{pre}_{pol}_maker_first_ms' in X else np.array([])
            C.require(not (fm[np.isfinite(fm)] <= 0).any(), f'{pre}_{pol}: a maker fill at or before the order reached the venue')
            ms = X[f'{pre}_{pol}_maker_share'].to_numpy(float)
            rf = X[f'{pre}_{nm}_{pr_}_refused'].to_numpy(float) > 0
            C.require(not (ms[rf] > 0).any(), f'{pre}_{pol}: a refused order filled')
    keys, dayk = [], []
    for d in present:
        reg = MM.registry(d).sort_values('ws')
        keys += [f'{d}:{int(m)}' for m in reg.mid]
        dayk += [d] * len(reg)
    keys, dayk = np.array(keys), np.array(dayk)
    per_day = len(keys) / len(present)
    wk = (X.day.astype(str) + ':' + X.mid.astype(np.int64).astype(str)).to_numpy()
    early = X.cut_minus_act_ms.to_numpy(float) >= EARLY_MS
    legs = {dh: json.loads((src / f'{dh}.json').read_text()) for dh in present}
    out = dict(asof='2026-11-27', post_hoc=True, rehearsal=reh, days=present, inputs=legs, parity_max_abs_c=parity, early_ms=EARLY_MS,
               selections={}, cells={})
    for sel in E.SELS:
        msel = X[sel].to_numpy(bool)
        so = dict(actions=int(msel.sum()), early=int((msel & early).sum()), calm={})
        for gg in GAPS:
            w = X[f'C{int(gg)}_wait_ms'].to_numpy(float)[msel]
            ch = X[f'C{int(gg)}_side_mid_5s_change_c'].to_numpy(float)[msel]
            so['calm'][f'C{int(gg)}'] = dict(wait_s_q10_50_90=[float(v) / 1000. for v in np.nanquantile(np.where(np.isfinite(w), w, np.nan), [.1, .5, .9])],
                                             never_calm=int((~np.isfinite(w)).sum()), side_mid_change_5s_after_calm_c_mean=float(np.nanmean(ch)),
                                             side_mid_change_5s_after_calm_c_median=float(np.nanmedian(ch)))
        out['selections'][sel] = so
        for scope, sm_ in (('ALL', msel), ('EARLY', msel & early)):
            ii = np.flatnonzero(sm_)
            for b, s in E.COMBOS:
                pre = f'{b}_s{int(s)}'
                fl = X[f'{pre}_fill'].to_numpy(float)
                filled = ii[fl[ii] > 0]
                for pol in POLICIES:
                    pn = X[f'{pre}_{pol}_pnl'].to_numpy(float) / 100.
                    st = SX.stats(SX.window_series(pn[sm_], wk[sm_], keys), dayk, per_day)
                    c = dict(total_usd=st['total_usd'], per_intended_c=float(pn[ii].sum() * 100. / (Q * len(ii))) if len(ii) else None,
                             sharpe_day_equiv=st['sharpe_day_equiv'], max_drawdown_usd=st['max_drawdown_usd'], daily_usd=st['daily_usd'],
                             days_positive=int(sum(v > 0 for v in st['daily_usd'])), filled_entries=int(len(filled)))
                    if pol in MAKER or pol in SENS or pol == 'D0_JOIN_T10_TAKE':
                        nm, pr_ = pol.split('_')[:2]
                        w_ = fl[filled]
                        col = lambda k: X[f'{pre}_{pol}_{k}'].to_numpy(float)[filled] if f'{pre}_{pol}_{k}' in X else np.array([np.nan])  # noqa: E731
                        ms, wt, a5 = col('maker_share'), col('maker_wait_ms'), col('after_fill_5s_c')
                        snt = X[f'{pre}_{nm}_{pr_}_sent'].to_numpy(float)[filled] > 0
                        c.update(maker_closed_share=float(np.nansum(ms * w_) / w_.sum()) if len(filled) else None,
                                 maker_wait_s_median=float(np.nanmedian(wt) / 1000.) if np.isfinite(wt).any() else None,
                                 after_fill_5s_c=float(np.nanmean(a5)) if np.isfinite(a5).any() else None,
                                 sent_share=float(snt.mean()) if len(filled) else None,
                                 refused_share_of_sent=float(X[f'{pre}_{nm}_{pr_}_refused'].to_numpy(float)[filled][snt].mean()) if snt.any() else None,
                                 beyond_shown_share_of_sent=float(X[f'{pre}_{nm}_{pr_}_beyond_shown'].to_numpy(float)[filled][snt].mean()) if snt.any() else None,
                                 price_its_way_c_median=float(np.nanmedian(X[f'{pre}_{nm}_{pr_}_price_its_way'].to_numpy(float)[filled][snt])) if snt.any() else None)
                    out['cells'][f'{scope}|{sel}|{pre}|{pol}'] = c
    # predicates, conservative entry book, the four strategy selections
    cmp_ = {}
    for sel in STRAT:
        for s in (3., 10.):
            pre = f'AB_s{int(s)}'
            g = lambda sc, p: out['cells'][f'{sc}|{sel}|{pre}|{p}']  # noqa: E731
            sh = lambda sc, p: g(sc, p)['sharpe_day_equiv'] if g(sc, p)['sharpe_day_equiv'] is not None else -99.  # noqa: E731
            r = {}
            for gg in GAPS:                                                     # maker vs taking at the same calm decision (ALL)
                for pr_ in PRICES:
                    for tl_, _, _ in TAILS:
                        p = f'C{int(gg)}_{pr_}_{tl_}'
                        r[f'{p}_minus_TAKE_C{int(gg)}_usd'] = g('ALL', p)['total_usd'] - g('ALL', f'TAKE_C{int(gg)}')['total_usd']
            bm = max(MAKER, key=lambda p: sh('EARLY', p)); bmp = max(MAKER, key=lambda p: g('EARLY', p)['total_usd'])
            bc = max(COMP, key=lambda p: sh('EARLY', p)); bcp = max(COMP, key=lambda p: g('EARLY', p)['total_usd'])
            r.update(best_maker_by_sharpe=bm, best_maker_sharpe=sh('EARLY', bm), best_maker_sharpe_usd=g('EARLY', bm)['total_usd'],
                     best_maker_by_pnl=bmp, best_maker_pnl_usd=g('EARLY', bmp)['total_usd'], best_maker_pnl_sharpe=sh('EARLY', bmp),
                     best_comparator_by_sharpe=bc, best_comparator_sharpe=sh('EARLY', bc), best_comparator_sharpe_usd=g('EARLY', bc)['total_usd'],
                     best_comparator_by_pnl=bcp, best_comparator_pnl_usd=g('EARLY', bcp)['total_usd'],
                     **{f'{p}_sharpe': sh('EARLY', p) for p in COMP}, **{f'{p}_usd': g('EARLY', p)['total_usd'] for p in COMP},
                     maker_beats_best_comparator_sharpe=bool(sh('EARLY', bm) > sh('EARLY', bc)),
                     maker_beats_best_comparator_pnl=bool(g('EARLY', bmp)['total_usd'] > g('EARLY', bcp)['total_usd']),
                     maker_beats_TB1S_sharpe=bool(sh('EARLY', bm) > sh('EARLY', 'TB1S')))
            cmp_[f'{sel}|{pre}'] = r
    out['compare_AB'] = cmp_
    cnt = dict(cells=len(cmp_), **{k: int(sum(bool(v[k]) for v in cmp_.values())) for k in
                                   ('maker_beats_best_comparator_sharpe', 'maker_beats_best_comparator_pnl', 'maker_beats_TB1S_sharpe')})
    for gg in GAPS:
        for pr_ in PRICES:
            for tl_, _, _ in TAILS:
                k = f'C{int(gg)}_{pr_}_{tl_}_minus_TAKE_C{int(gg)}_usd'
                cnt[f'{k}_positive'] = int(sum(v[k] > 0 for v in cmp_.values()))
    out['counts_AB'] = cnt
    out['clock_AB'] = {f'{sel}|AB_s{int(s)}|C{int(gg)}_{p}': {f'M{int(mm)}': out['cells'][f'ALL|{sel}|AB_s{int(s)}|' + (f'C{int(gg)}_{p}_T10_TAKE' if mm == M50 else f'C{int(gg)}_{p}_T10_TAKE_M{int(mm)}')]['total_usd']
                                                               for mm in (SHIFTS[0], M50, SHIFTS[1])}
                       for sel in STRAT for s in (3., 10.) for gg in GAPS for p in PRICES}
    out['reference_AB'] = {f'{sel}|AB_s{int(s)}': {p: out['cells'][f'ALL|{sel}|AB_s{int(s)}|{p}']['total_usd'] for p in REF + ('TB1S',)}
                           for sel in E.SELS for s in (3., 10.)}
    TR.dump(fn, out)
    print('parity max |diff| c', max(parity.values()))
    print('counts', cnt)
    for k, v in cmp_.items():
        print(k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items() if not a.startswith('C')})


def selftest():
    kn = np.array([0., 100., 150., 600., 2000., 2100.])
    x = calm_times(kn, np.array([120., 160., 700., 2050., -5.]), 300.)
    assert x.tolist() == [450., 450., 900., 2400., -5.], x               # 150 + 300; 600 + 300; 2100 + 300; nothing before -> at once
    assert calm_times(kn, np.array([1000.]), 300.).tolist() == [1000.]     # already quiet 400 ms
    assert calm_times(kn, np.array([1000.]), 1400.).tolist() == [3500.]    # 600 -> 2000 is exactly 1.4 s: a change at the end is not quiet
    x2 = calm_times(np.array([0., 100., 200.]), np.array([150.]), 1000.)
    assert x2.tolist() == [1200.], x2                                    # the last change 200 + 1 s, none after
    assert calm_times(np.array([0., 100., 1099.]), np.array([50.]), 1000.).tolist() == [2099.]   # 100 -> 1099 is < 1 s: not calm
    fr = np.full(22, -1.); fr[7:12] = 0.; fr[17:22] = 0.
    fr[2:5], fr[7:10] = [48., 47., 45.], [10., 20., 30.]                     # bids 48 x 10, 47 x 20, 45 x 30
    fr[12:17], fr[17:22] = [52., 53., 54., 55., 56.], [5., 6., 7., 8., 9.]  # five asks shown
    assert price_of(fr, 1., 'JOIN') == 52. and price_of(fr, -1., 'JOIN') == 48.
    assert price_of(fr, 1., 'IMP') == 51. and price_of(fr, -1., 'IMP') == 49.
    tight = fr.copy(); tight[12] = 49.
    assert price_of(tight, 1., 'IMP') == 49. and price_of(tight, -1., 'IMP') == 48.   # 1c spread: IMP = JOIN
    bad = fr.copy(); bad[12] = 48.
    assert np.isnan(price_of(bad, 1., 'JOIN'))
    assert venue_queue(fr, 1., 53.) == (False, 6., False) and venue_queue(fr, 1., 51.) == (False, 0., False)
    assert venue_queue(fr, 1., 60.) == (False, 0., True)                     # beyond five shown asks: unknown
    assert venue_queue(fr, -1., 44.) == (False, 0., False)                   # beyond three shown bids: nothing there
    assert venue_queue(fr, -1., 46.) == (False, 0., False) and venue_queue(fr, -1., 47.) == (False, 20., False)
    assert venue_queue(fr, 1., 48.)[0] and venue_queue(fr, -1., 52.)[0]       # an ask at the bid / a bid at the ask: refused
    # post(): the price from the frame KNOWN at the decision, the refusal and queue from the venue frame at arrival
    f = np.stack([fr.copy() for _ in range(3)])
    f[:, 0] = [0., 10., 30.]                                                 # venue times
    f[:, 1] = [5., 40., 45.]                                                 # known times
    f[1, 2] = 52.; f[1, 12:17] = [53., 54., 55., 56., 57.]
    f[2, 2] = 52.
    px, tr, refused, ah, _ = post(f, f[:, 0], f[:, 1], 1., 10., 'JOIN')       # decided at 10 ms: knows frame 0 only (ask 52)
    assert px == 52. and tr == 32. and refused, (px, tr, refused)            # rests at 32 ms; the venue bid 52 then refuses it
    f[2, 2] = 48.; f[2, 12:17] = [52., 53., 54., 55., 56.]
    px, tr, refused, ah, _ = post(f, f[:, 0], f[:, 1], 1., 10., 'JOIN')
    assert px == 52. and not refused and ah == 5., (px, refused, ah)
    assert post(f, f[:, 0], f[:, 1], 1., 41., 'JOIN')[0] == 53.              # decided at 41 ms: knows frame 1 (ask 53)
    # the trade clock: a print 30 ms after arrival on the receipt clock is 20 ms BEFORE it at the venue
    lg = (np.array([1030., 1080.]), np.array([1, 1]), np.array([52., 52.]), np.array([5., 50.]))
    got, ts, _ = EM.maker_fill(lg, 1000. + M50, 5000. + M50, 1., 52., 0., 3.)
    assert got == 3. and ts == [1080.], ts
    assert EM.maker_fill(lg, 1000., 5000., 1., 52., 0., 3.)[1] == [1030.]     # EXIT_MAKER's receipt clock took the earlier print
    # take_exit: a long UP sells 20 into the bids 48 x 10, 47 x 20; nothing past the path end or without a decision
    f2 = np.stack([fr.copy() for _ in range(2)]); f2[:, 0] = [0., 100.]
    q, v, fe = take_exit(f2, f2[:, 0], 1000., 1., 20., 0.)
    assert q == 20. and abs(v - (10 * 48. + 10 * 47.)) < 1e-9 and fe > 0, (q, v, fe)
    assert take_exit(f2, f2[:, 0], 100., 1., 20., 0.) == (0., 0., 0.) and take_exit(f2, f2[:, 0], 1000., 1., 20., None) == (0., 0., 0.)
    kt, sm = np.array([0., 10.]), np.array([55., 65.])
    assert tail_x('FAV', kt, sm, 5.) == 5. and tail_x('FAV', kt, sm, 20.) is None and tail_x('HOLD', kt, sm, 5.) is None
    assert tail_x('TAKE', kt, sm, 7.) == 7.
    try:
        labels('20261022', '/nonexistent')
        raise AssertionError('an undeclared day accepted')
    except ValueError as e_:
        assert 'not a declared day' in str(e_), e_
    print('exit_maker2 selftest PASS')


if __name__ == '__main__':
    cmd, a_ = sys.argv[1], sys.argv[2:]
    if cmd == 'labels':
        labels(a_[0], a_[1] if len(a_) > 1 else None)
    elif cmd == 'read':
        read(a_[0] if a_ else None)
    else:
        dict(selftest=selftest)[cmd]()
