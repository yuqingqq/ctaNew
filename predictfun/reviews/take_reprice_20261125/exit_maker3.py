"""Maker exit that rests only while Binance is calm: cancelled when Binance moves again, re-posted at the next calm (USER 2026-11-27:
"Have we calculated taker fee in pnl, and try the maker mode with cancels when Binance moves again"). Four consumed days, every market, no
orders; POST HOC on these days (entries, calm gaps, prices, tails, fill rule and comparators from EXIT_MAKER2); declared before this run.
Entries: EXIT_FOLLOW's on the conservative AB entry book (+ 3c / + 10c limits).
Loop, from the first calm after our fill notice (no Binance futures mid change known in the last G = 0.3 / 1 / 3 s, EXIT_MAKER2's rule):
  decide at a calm time x: one post-only order for the remaining shares, priced on the book known at x (JOIN = the best on the exit side,
  IMP = one tick inside if the spread is >= 2c), resting from x + 22 ms with EXIT_MAKER2's venue rules (refused if it would cross the
  venue's book on arrival, queue ahead = the shown size at our price, fills from Predict's prints at venue time t_ms2 - 50 ms, cancels
  ahead not credited);
  cancel rule: CX_ANY = the first Binance mid change known after x triggers a cancel; CX_OURS = only a change in our direction (our side's
  value up: the move whose chasers would lift our order); the cancel is sent 2 ms after the change is known and lands 22 ms later (no speed
  bump on removals); the order can fill until it lands; the remaining shares are re-posted at the next calm after the triggering change;
  a refused order is re-decided at the first calm after its refusal is known (22 ms after arrival); a decision without a valid book is
  re-tried at the first calm 1 s later; at most 500 decisions per position (the cap is counted, expected 0);
  NOCX = no cancels and no re-posts (EXIT_MAKER2's single order: reproduced bit for bit, a check).
Deadline: 10 s (then the rest is taken) or 39 s (then taken unless the side's mid known then is >= 60c) after the first order rests, and
30 s before the window's end; nothing is sent after the cut (held). The 10 s variant is the 39 s loop truncated at its deadline (same
decisions up to it).
Comparators (reproduced from EXIT_MAKER2, a check): HOLD, TB1S, FAV_ONLY, TAKE_C<G> (everything sold by taking at the first calm).
Fees: the entry IOC pays 0.02 x min(p, 100 - p) per share at each level, every exit by taking the same on what it sells, maker fills 0,
settlement 0 (the P&L of every earlier result). Stored per position: the entry fee and the exit taking fee; reported per cell in $. Fee-in-
shares sensitivity: Predict takes a BUY's fee in shares (fee / price of them; README venue table), so every exit and settlement scales by
(1 - phi / filled), phi = entry fee / entry price on our side (the walk depth of a smaller exit not re-done).
Scopes: ALL; EARLY = matched >= 15.022 s before the cut (EXIT_MAKER2's). Counts over the EARLY cells with >= 30 filled positions (the
four strategy selections x two limits; EXIT_MAKER2's reporting rule).
Readings: P&L ($, per intended share, by day, days > 0), per-window Sharpe (day equivalent), max drawdown; for maker loops the share closed
by maker orders, decisions / cancels / refusals per position, the share of maker shares filled after the triggering Binance change but
before the cancel landed (picked off in flight), the side's mid 5 s after a maker fill against its price (what the buyer got); fee totals.
Outputs (write-once): OUT/exit_maker3/<day>.parquet + .json, exit_maker3.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/exit_maker3.py \
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
import sharpe_exits as SX  # noqa: E402

CT, C, TR, TK, TE, TM, WT, F, VR, MM = E.CT, E.C, E.TR, E.TK, E.TE, E.TM, E.WT, E.F, E.VR, E.MM
OUT = WT.OUT / 'exit_maker3'
DAYS, Q = E.DAYS, E.Q
LEARN, M50, GAPS, PRICES = M2.LEARN, M2.M50, M2.GAPS, M2.PRICES
COMBOS = [('AB', 3.), ('AB', 10.)]
MODES = ('NOCX', 'CX_ANY', 'CX_OURS')
TAILS = (('T10_TAKE', 10000.), ('T39_FAV', 39000.))
CANCEL_MS = TM.DECIDE + LEARN
CAP = 500
RETRY_MS = 1000.
MAKER = [f'{m}_C{int(g)}_{p}_{t}' for m in MODES for g in GAPS for p in PRICES for t, _ in TAILS]
COMP = ('HOLD', 'TB1S', 'FAV_ONLY') + tuple(f'TAKE_C{int(g)}' for g in GAPS)
POLICIES = COMP + tuple(MAKER)
EARLY_MS, MIN_FILLED = M2.EARLY_MS, 30
STRAT = M2.STRAT


class Binance:
    """Binance futures mid changes on our clock (exchange time + feed) with their direction, and the quiet-gap index per G."""

    def __init__(self, Eb, Mb):
        self.kn = np.asarray(Eb, float) + TM.FEED
        dm = np.r_[0., np.diff(np.asarray(Mb, float))]
        self.up, self.dn = self.kn[dm > 0], self.kn[dm < 0]
        nxt = np.r_[self.kn[1:], np.inf] - self.kn
        self.big = {g: np.flatnonzero(nxt > g) for g in GAPS}

    def calm(self, y, g):
        """First x >= y (epoch ms) with no change known in (x - g, x] (scalar calm_times)."""
        idx = int(np.searchsorted(self.kn, y, 'right') - 1)
        if idx < 0:
            return float(y)
        b = self.big[g]
        j = int(np.searchsorted(b, idx, 'left'))
        return float(max(self.kn[b[j]] + g, y)) if j < len(b) else np.inf

    def next_change(self, x, d, mode):
        """Known time of the first change after x: any direction (CX_ANY) or our side's direction (CX_OURS); None if none."""
        arr = self.kn if mode == 'CX_ANY' else (self.up if d > 0 else self.dn)
        j = int(np.searchsorted(arr, x, 'right'))
        return float(arr[j]) if j < len(arr) else None


def maker_loop(mode, g, pr_, x0, fl, d, f, f0, fk, lg, t0, cut, BN):
    """The maker exit from the first calm decision x0 (path clock) to the 39 s deadline -> dict: t_first, D39, fill increments
    [(venue time, shares, price, picked off in flight)], decisions, cancels, refusals, capped."""
    t_first = x0 + E.DEC + LEARN
    D = min(t_first + 39000., cut)
    out = dict(t_first=t_first, D39=D, incs=[], decisions=0, cancels=0, refusals=0, capped=False)
    if not (np.isfinite(x0) and D > t_first):
        return out
    rem, x = fl, x0
    while rem > 1e-9 and np.isfinite(x):
        t_rest = x + E.DEC + LEARN
        if t_rest >= D:
            break
        if out['decisions'] >= CAP:
            out['capped'] = True
            break
        out['decisions'] += 1
        px, _, refused, ahead, _ = M2.post(f, f0, fk, d, x, pr_)
        if not np.isfinite(px):
            if mode == 'NOCX':
                break
            x = BN.calm(t0 + x + RETRY_MS, g) - t0
            continue
        if refused:
            out['refusals'] += 1
            if mode == 'NOCX':
                break
            x = BN.calm(t0 + t_rest + LEARN, g) - t0
            continue
        trig = None if mode == 'NOCX' else BN.next_change(t0 + x, d, mode)
        trig = None if trig is None else trig - t0
        t_end = D if trig is None else min(trig + CANCEL_MS, D)
        got, ts, qs = EM.maker_fill(lg, t0 + t_rest + M50, t0 + t_end + M50, d, px, ahead, rem)
        for tv, qv in zip(ts, qs):
            tvr = tv - M50 - t0
            C.require(tvr > t_rest, 'a maker fill at or before the order reached the venue')
            out['incs'].append((tvr, qv, px, bool(trig is not None and tvr > trig - TM.FEED)))
        rem -= got
        if t_end >= D or rem <= 1e-9:
            break
        out['cancels'] += 1
        x = BN.calm(t0 + trig, g) - t0
    return out


def settle_tail(res, T, tail, fl, d, f, f0, end, kt, sm):
    """Truncate a loop at its deadline (T after the first order rests, capped at the cut) and apply the tail -> (maker shares, maker value,
    taken shares, taken value, taking fee, maker shares picked off in flight, buyers' 5 s gain per maker share)."""
    t_first, D39 = res['t_first'], res['D39']
    D = min(t_first + T, D39) if T < 39000. else D39
    inc = [z for z in res['incs'] if z[0] <= D]
    mq = float(sum(z[1] for z in inc))
    mc = float(sum(z[1] * z[2] for z in inc))
    pk = float(sum(z[1] for z in inc if z[3]))
    g5 = [(E.mid_at(kt, sm, z[0] + 5000.) - (z[2] if d > 0 else 100. - z[2]), z[1]) for z in inc]
    g5 = [(a, b) for a, b in g5 if np.isfinite(a)]
    buy5 = float(np.average([a for a, _ in g5], weights=[b for _, b in g5])) if g5 else np.nan
    x = None
    if D > t_first and fl - mq > 1e-9:
        x = M2.tail_x('TAKE' if tail == 'T10_TAKE' else 'FAV', kt, sm, D)
    tq, tc, tf = M2.take_exit(f, f0, end, d, fl - mq, x)
    return mq, mc, tq, tc, tf, pk, buy5


def labels(day, od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (3000, 3001))
    C.require(day in DAYS, f'{day} is not a declared day')
    od = Path(od) if od else OUT
    fp = od / f'{day}.parquet'
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    S = E.selections()
    S = S[S.day == day].reset_index(drop=True)
    L = pd.read_parquet(CT.OUT / f'{day}_labels.parquet', columns=['t', 'd', 'mid', 'lim', 'payout'])
    S = S.merge(L, on=['t', 'd'], how='left', validate='1:1')
    BN = Binance(*CT.mid_series(day))
    xmin = S.t.to_numpy(float) + TM.FEED + TM.DECIDE + E.NET + E.HOLD + LEARN
    CALM = {g: M2.calm_times(BN.kn, xmin, g) for g in GAPS}
    LG, lst = EM.legs_of(day)
    reg = MM.registry(day).set_index('mid')
    rg = pd.read_parquet(MM.S1 / day / 'registry.parquet').set_index('mid')
    by_mid = S.groupby('mid').indices
    o = collections.defaultdict(lambda: np.full(len(S), np.nan))
    n_cap = 0
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
                act = t + TM.FEED + TM.DECIDE - t0 + E.NET + E.HOLD
                o['cut_minus_act_ms'][i] = cut - act
                jv = int(np.searchsorted(f0, act, 'right') - 1)
                side = 1 if d > 0 else 0
                pa, sa = TK.side_levels(f[jv], side)
                pb, sb_ = TK.side_levels(f[jv + 1], side) if jv + 1 < len(f) else (np.full(5, -1.), np.zeros(5))
                bk = TK.minbook(pa[None], sa[None], pb[None], sb_[None])
                k0 = int(np.searchsorted(fk, act, 'right'))
                kt, sm = fk[k0:], (midy[k0:] if d > 0 else 100. - midy[k0:])
                calm = {g: float(CALM[g][i]) - t0 for g in GAPS}
                for b, s in COMBOS:
                    pre = f'{b}_s{int(s)}'
                    fl, co, fe_ = (x[0] for x in TK.vwalk(*bk, np.array([d]), np.array([lim + d * s])))
                    o[f'{pre}_fill'][i] = fl
                    o[f'{pre}_entry_fee_c'][i] = fe_
                    if fl <= 0:
                        for pol in POLICIES:
                            o[f'{pre}_{pol}_pnl'][i] = 0.
                            o[f'{pre}_{pol}_pnl_sharefee'][i] = 0.
                            o[f'{pre}_{pol}_exit_fee_c'][i] = 0.
                        continue
                    e = co / fl if d > 0 else 100. - co / fl
                    phi_share = (fe_ / e) / fl if e > 0 else 0.

                    def put(pol, mq, mc, tq, tc, tf):
                        pn = d * (mc + tc + (fl - mq - tq) * 100. * pay - co) - fe_ - tf
                        o[f'{pre}_{pol}_pnl'][i] = pn
                        o[f'{pre}_{pol}_exit_fee_c'][i] = tf
                        o[f'{pre}_{pol}_pnl_sharefee'][i] = (1. - phi_share) * (pn + fl * e + fe_) - fl * e
                    for pol in ('HOLD', 'TB1S', 'FAV_ONLY'):
                        x = E.decide(pol, E.trigger(kt, sm, e, cut), E.mid_at(kt, sm, act + E.T_MED), E.mid_at(kt, sm, act + E.T_UQ), act, cut)
                        put(pol, 0., 0., *M2.take_exit(f, f0, end, d, fl, x))
                    for g in GAPS:
                        put(f'TAKE_C{int(g)}', 0., 0., *M2.take_exit(f, f0, end, d, fl, calm[g] if calm[g] <= cut else None))
                    for mode in MODES:
                        for g in GAPS:
                            for pr_ in PRICES:
                                res = maker_loop(mode, g, pr_, calm[g], fl, d, f, f0, fk, lg, t0, cut, BN)
                                n_cap += int(res['capped'])
                                base = f'{pre}_{mode}_C{int(g)}_{pr_}'
                                o[f'{base}_decisions'][i] = res['decisions']
                                o[f'{base}_cancels'][i] = res['cancels']
                                o[f'{base}_refusals'][i] = res['refusals']
                                for tl_, T in TAILS:
                                    mq, mc, tq, tc, tf, pk, buy5 = settle_tail(res, T, tl_, fl, d, f, f0, end, kt, sm)
                                    put(f'{mode}_C{int(g)}_{pr_}_{tl_}', mq, mc, tq, tc, tf)
                                    o[f'{base}_{tl_}_maker_share'][i] = mq / fl
                                    o[f'{base}_{tl_}_picked_off_sh'][i] = pk
                                    o[f'{base}_{tl_}_buyer_5s_c'][i] = buy5
            C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    out = S[['day', 't', 'd', 'key', 'mid'] + list(E.SELS)].copy()
    out = pd.concat([out, pd.DataFrame(dict(o), index=out.index)], axis=1)
    od.mkdir(parents=True, exist_ok=True)
    out.to_parquet(fp, index=False)
    TR.dump(od / f'{day}.json', dict(asof='2026-11-27', day=day, actions=int(len(S)), legs=lst, binance_mid_changes=int(len(BN.kn)), capped_loops=n_cap))
    print(day, len(S), lst, 'capped', n_cap, 'rss', round(F.rss_mib()), flush=True)


def read(od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    reh = od is not None
    od = Path(od) if od else None
    fn = (od or H) / 'exit_maker3.json'
    C.require(not fn.exists(), 'exit_maker3.json exists (write-once)')
    src = od or OUT
    present = [dh for dh in DAYS if (src / f'{dh}.parquet').exists()]
    X = pd.concat([pd.read_parquet(src / f'{dh}.parquet') for dh in present], ignore_index=True)
    X2 = pd.concat([pd.read_parquet(M2.OUT / f'{dh}.parquet') for dh in present], ignore_index=True)
    C.require(len(X) == len(X2) and (X.t.to_numpy() == X2.t.to_numpy()).all(), 'the actions differ from EXIT_MAKER2')
    inputs = {dh: json.loads((src / f'{dh}.json').read_text()) for dh in present}
    C.require(sum(v['capped_loops'] for v in inputs.values()) == 0, 'a maker loop hit the decision cap')
    parity = {}
    for b, s in COMBOS:
        pre = f'{b}_s{int(s)}'
        pairs = [(p, p) for p in COMP] + [(f'NOCX_C{int(g)}_{p}_{t}', f'C{int(g)}_{p}_{t}') for g in GAPS for p in PRICES for t, _ in TAILS]
        for mine, theirs in pairs:
            a, r = X[f'{pre}_{mine}_pnl'].to_numpy(float), X2[f'{pre}_{theirs}_pnl'].to_numpy(float)
            C.require(np.allclose(a, r, rtol=0, atol=1e-9), f'{pre}_{mine} not reproduced from EXIT_MAKER2')
            parity[f'{pre}_{mine}'] = float(np.max(np.abs(a - r)))
        C.require(np.allclose(X[f'{pre}_fill'].to_numpy(float), X2[f'{pre}_fill'].to_numpy(float), rtol=0, atol=1e-12), f'{pre} fills differ')
    keys, dayk = [], []
    for dd in present:
        reg = MM.registry(dd).sort_values('ws')
        keys += [f'{dd}:{int(m)}' for m in reg.mid]
        dayk += [dd] * len(reg)
    keys, dayk = np.array(keys), np.array(dayk)
    per_day = len(keys) / len(present)
    wk = (X.day.astype(str) + ':' + X.mid.astype(np.int64).astype(str)).to_numpy()
    early = X.cut_minus_act_ms.to_numpy(float) >= EARLY_MS
    out = dict(asof='2026-11-27', post_hoc=True, rehearsal=reh, days=present, inputs=inputs, parity_max_abs_c=parity, early_ms=EARLY_MS,
               min_filled=MIN_FILLED, selections={}, cells={})
    for sel in E.SELS:
        msel = X[sel].to_numpy(bool)
        out['selections'][sel] = dict(actions=int(msel.sum()), early=int((msel & early).sum()))
        for scope, sm_ in (('ALL', msel), ('EARLY', msel & early)):
            ii = np.flatnonzero(sm_)
            for b, s in COMBOS:
                pre = f'{b}_s{int(s)}'
                fl = X[f'{pre}_fill'].to_numpy(float)
                filled = ii[fl[ii] > 0]
                efee = float(X[f'{pre}_entry_fee_c'].to_numpy(float)[filled].sum() / 100.)
                for pol in POLICIES:
                    pn = X[f'{pre}_{pol}_pnl'].to_numpy(float) / 100.
                    st = SX.stats(SX.window_series(pn[sm_], wk[sm_], keys), dayk, per_day)
                    psf = X[f'{pre}_{pol}_pnl_sharefee'].to_numpy(float) / 100.
                    sts = SX.stats(SX.window_series(psf[sm_], wk[sm_], keys), dayk, per_day)
                    xfee = float(X[f'{pre}_{pol}_exit_fee_c'].to_numpy(float)[filled].sum() / 100.)
                    c = dict(total_usd=st['total_usd'], per_intended_c=float(pn[ii].sum() * 100. / (Q * len(ii))) if len(ii) else None,
                             sharpe_day_equiv=st['sharpe_day_equiv'], max_drawdown_usd=st['max_drawdown_usd'], daily_usd=st['daily_usd'],
                             days_positive=int(sum(v > 0 for v in st['daily_usd'])), filled_entries=int(len(filled)),
                             entry_fee_usd=efee, exit_fee_usd=xfee, before_fees_usd=st['total_usd'] + efee + xfee,
                             sharefee_total_usd=sts['total_usd'], sharefee_sharpe_day_equiv=sts['sharpe_day_equiv'])
                    if pol in MAKER:
                        base = pol.rsplit('_', 2)[0]
                        tl_ = '_'.join(pol.rsplit('_', 2)[1:])
                        w_ = fl[filled]
                        ms = X[f'{pre}_{base}_{tl_}_maker_share'].to_numpy(float)[filled]
                        pk = X[f'{pre}_{base}_{tl_}_picked_off_sh'].to_numpy(float)[filled]
                        b5 = X[f'{pre}_{base}_{tl_}_buyer_5s_c'].to_numpy(float)[filled]
                        mq = ms * w_
                        c.update(maker_closed_share=float(mq.sum() / w_.sum()) if len(filled) else None,
                                 picked_off_share_of_maker=float(pk.sum() / mq.sum()) if mq.sum() > 0 else None,
                                 buyer_5s_c=float(np.nansum(b5 * mq) / np.sum(mq[np.isfinite(b5)])) if np.isfinite(b5).any() and np.sum(mq[np.isfinite(b5)]) > 0 else None,
                                 decisions_mean=float(X[f'{pre}_{base}_decisions'].to_numpy(float)[filled].mean()) if len(filled) else None,
                                 cancels_mean=float(X[f'{pre}_{base}_cancels'].to_numpy(float)[filled].mean()) if len(filled) else None,
                                 refusals_mean=float(X[f'{pre}_{base}_refusals'].to_numpy(float)[filled].mean()) if len(filled) else None)
                    out['cells'][f'{scope}|{sel}|{pre}|{pol}'] = c
    # predicates: EARLY, the four strategy selections x two limits with >= MIN_FILLED filled positions
    cells = [(sel, f'AB_s{int(s)}') for sel in STRAT for _, s in COMBOS]
    big = [(a, b) for a, b in cells if out['cells'][f'EARLY|{a}|{b}|TB1S']['filled_entries'] >= MIN_FILLED]
    g = lambda a, b, p: out['cells'][f'EARLY|{a}|{b}|{p}']  # noqa: E731
    sh = lambda a, b, p: g(a, b, p)['sharpe_day_equiv'] if g(a, b, p)['sharpe_day_equiv'] is not None else -99.  # noqa: E731
    cnt = dict(cells=len(big), cells_list=[f'{a}|{b}' for a, b in big])
    for mode in ('CX_ANY', 'CX_OURS'):
        for gg in GAPS:
            for p in PRICES:
                for tl_, _ in TAILS:
                    k = f'{mode}_C{int(gg)}_{p}_{tl_}'
                    cnt[f'{k}_beats_NOCX'] = int(sum(g(a, b, k)['total_usd'] > g(a, b, f'NOCX_C{int(gg)}_{p}_{tl_}')['total_usd'] for a, b in big))
                    cnt[f'{k}_beats_TAKE_C{int(gg)}'] = int(sum(g(a, b, k)['total_usd'] > g(a, b, f'TAKE_C{int(gg)}')['total_usd'] for a, b in big))
    best = {}
    for a, b in big:
        bm = max(MAKER, key=lambda p: sh(a, b, p)); bmp = max(MAKER, key=lambda p: g(a, b, p)['total_usd'])
        bc = max(COMP, key=lambda p: sh(a, b, p)); bcp = max(COMP, key=lambda p: g(a, b, p)['total_usd'])
        cx = [p for p in MAKER if p.startswith('CX')]
        bx = max(cx, key=lambda p: sh(a, b, p))
        best[f'{a}|{b}'] = dict(best_maker_by_sharpe=bm, best_maker_by_pnl=bmp, best_comparator_by_sharpe=bc, best_comparator_by_pnl=bcp,
                                best_cancel_maker_by_sharpe=bx,
                                maker_beats_comparator_sharpe=bool(sh(a, b, bm) > sh(a, b, bc)),
                                maker_beats_comparator_pnl=bool(g(a, b, bmp)['total_usd'] > g(a, b, bcp)['total_usd']))
    cnt['maker_beats_best_comparator_sharpe'] = int(sum(v['maker_beats_comparator_sharpe'] for v in best.values()))
    cnt['maker_beats_best_comparator_pnl'] = int(sum(v['maker_beats_comparator_pnl'] for v in best.values()))
    out['counts_early'] = cnt
    out['best_early'] = best
    TR.dump(fn, out)
    print('parity max |diff| c', max(parity.values()), 'cells', len(big))
    print({k: v for k, v in cnt.items() if k.startswith('maker_') or k.endswith('_T10_TAKE_beats_NOCX')})
    for k, v in best.items():
        print(k, v)


def selftest():
    # Binance: changes at 0 (from), 100 (up), 150 (down), 600 (up), 2000 (up)
    BN = Binance(np.array([0., 100., 150., 600., 2000.]) - TM.FEED, np.array([10., 11., 10., 12., 13.]))
    assert BN.calm(120., 300.) == 450. and BN.calm(700., 300.) == 900. and BN.calm(-5., 300.) == -5.
    assert BN.next_change(120., 1., 'CX_ANY') == 150. and BN.next_change(120., 1., 'CX_OURS') == 600. and BN.next_change(120., -1., 'CX_OURS') == 150.
    assert BN.next_change(2000., 1., 'CX_ANY') is None
    # a book: bids 48 x 10, asks 52 x 5 (five shown), valid throughout; venue = known times
    fr = np.full(22, -1.); fr[7:12] = 0.; fr[17:22] = 0.
    fr[2], fr[7] = 48., 10.
    fr[12:17], fr[17:22] = [52., 53., 54., 55., 56.], [5., 6., 7., 8., 9.]
    f = np.stack([fr.copy() for _ in range(2)]); f[:, 0] = [0., 50000.]; f[:, 1] = [0., 50000.]
    # long UP (d = 1) exits with an ask at 52 behind 5; prints (receipt clock = venue + 50) buy UP at 52
    # (the first print, at venue 110, comes before the order rests at 122 and must not touch it)
    lg = (np.array([110., 300., 520., 700., 2400.]) + M50, np.array([1, 1, 1, 1, 1]), np.array([52., 52., 52., 52., 52.]), np.array([10., 5., 4., 3., 20.]))
    BN2 = Binance(np.array([0., 500., 2200.]) - TM.FEED, np.array([10., 11., 12.]))
    # NOCX from a calm at 100: rests 122 ->, the 300 print consumes the 5 ahead, 520 fills 4, 700 fills 3, 2400 fills the last 3
    r = maker_loop('NOCX', 300., 'JOIN', 100., 10., 1., f, f[:, 0], f[:, 1], lg, 0., 1e9, BN2)
    assert [round(z[1], 6) for z in r['incs']] == [4., 3., 3.] and r['decisions'] == 1, r
    # CX_ANY: the change known at 500 cancels, landing at 524; the 520 print still fills 4 (picked off in flight). Re-posted at the next
    # calm (800) it rests from 822, after the 700 print; the change at 2200 cancels it at 2224, before the 2400 print; re-posted at 2500
    # it rests from 2522 with no later print -> 4 maker shares, 2 cancels
    r = maker_loop('CX_ANY', 300., 'JOIN', 100., 10., 1., f, f[:, 0], f[:, 1], lg, 0., 1e9, BN2)
    assert [round(z[1], 6) for z in r['incs']] == [4.] and r['incs'][0][3] and r['cancels'] == 2, r
    # CX_OURS for a long DOWN ignores up-moves: no cancel; its bid at 48 is never sold into here -> nothing
    r = maker_loop('CX_OURS', 300., 'JOIN', 100., 10., -1., f, f[:, 0], f[:, 1], lg, 0., 1e9, BN2)
    assert r['incs'] == [] and r['cancels'] == 0 and r['decisions'] == 1, r
    # refusals: the venue moved to bid 52 / ask 53 at 110 ms, known only at 400 ms. Decisions at 100, 300, 344 and 388 price the ask at
    # the stale 52 and are refused on arrival (re-decided at the first calm after each refusal notice); at 432 the new book is known: the
    # ask at 53 rests (behind 5), the 52 prints never reach it -> 4 refusals, 7 decisions, nothing filled, not capped
    fr1 = fr.copy(); fr1[2] = 52.; fr1[12:17] = [53., 54., 55., 56., 57.]
    fy = np.stack([fr.copy(), fr1]); fy[:, 0] = [0., 110.]; fy[:, 1] = [0., 400.]
    r = maker_loop('CX_ANY', 300., 'JOIN', 100., 10., 1., fy, fy[:, 0], fy[:, 1], lg, 0., 1e9, BN2)
    assert r['refusals'] == 4 and r['decisions'] == 7 and r['incs'] == [] and not r['capped'], r
    rn = maker_loop('NOCX', 300., 'JOIN', 100., 10., 1., fy, fy[:, 0], fy[:, 1], lg, 0., 1e9, BN2)
    assert rn['refusals'] == 1 and rn['decisions'] == 1 and rn['incs'] == [], rn                   # EXIT_MAKER2: refused once, then waits
    # truncation: the 10 s deadline keeps only fills before it
    res = dict(t_first=0., D39=39000., incs=[(500., 4., 52., False), (15000., 6., 52., False)])
    kt, sm = np.array([0.]), np.array([55.])
    mq, mc, tq, tc, tf, pk, _ = settle_tail(res, 10000., 'T10_TAKE', 10., 1., f, f[:, 0], 1e9, kt, sm)
    assert mq == 4. and abs(mc - 208.) < 1e-9 and tq == 6., (mq, tq)
    mq, _, tq, _, _, _, _ = settle_tail(res, 39000., 'T39_FAV', 10., 1., f, f[:, 0], 1e9, kt, sm)
    assert mq == 10. and tq == 0.
    try:
        labels('20261022', '/nonexistent')
        raise AssertionError('an undeclared day accepted')
    except ValueError as e_:
        assert 'not a declared day' in str(e_), e_
    print('exit_maker3 selftest PASS')


if __name__ == '__main__':
    cmd, a_ = sys.argv[1], sys.argv[2:]
    if cmd == 'labels':
        labels(a_[0], a_[1] if len(a_) > 1 else None)
    elif cmd == 'read':
        read(a_[0] if a_ else None)
    else:
        dict(selftest=selftest)[cmd]()
