"""Stop-loss or hold, as the top taker does (USER 2026-11-27: "So smart takers either stop loss by taking or holding to settlement? If we do so,
the profit looks best?"). POST HOC: added after EXIT_FOLLOW's results; same entries, books, exit rule and days (four consumed days); the
policies EXIT_FOLLOW shares are reproduced (a check). Declared before this run.
New policies (the side's prices on the known clock; exits IOCs at every shown level through the 172 ms hold on the snapshot before the exit
match; the rest held to settlement; nothing in the window's last 30 s):
  SL_ONLY: taken back when the side's mid is <= entry - 14c (its average losing take-back), else held to settlement: a stop without a
    profit-take.
  SL_FAV: SL_ONLY, and at 39 s taken back if the side is not a favourite (mid below 60c).
  FAV_BID: EXIT_FOLLOW's FAV_ONLY with the favourite judged on the side's best bid (>= 60c), the price we could sell at: a wide book's mid
    can read 60c with the bid far lower.
  SL_FAV_BID: SL_ONLY with the bid-based favourite check.
Reproduced from EXIT_FOLLOW: HOLD, TB1S, FAV_ONLY, TPSL. Readings as EXIT_FOLLOW (P&L per intended share, $, by day, days > 0, exit mix).
Outputs (write-once): OUT/exit_follow2/<day>.parquet, exit_follow2.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/exit_follow2.py \
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

CT, C, TR, TK, TE, TM, WT, F, VR, MM = E.CT, E.C, E.TR, E.TK, E.TE, E.TM, E.WT, E.F, E.VR, E.MM
OUT = WT.OUT / 'exit_follow2'
DAYS, Q = E.DAYS, E.Q
POLICIES = ('HOLD', 'TB1S', 'FAV_ONLY', 'TPSL', 'SL_ONLY', 'SL_FAV', 'FAV_BID', 'SL_FAV_BID')
REPRO = ('HOLD', 'TB1S', 'FAV_ONLY', 'TPSL')


def stop(kt, sm, e, cut):
    hit = (kt <= cut) & np.isfinite(sm) & (sm <= e - E.SL)
    return float(kt[np.argmax(hit)]) if hit.any() else None


def decide2(policy, kt, sm, sb, e, act, cut):
    if policy in ('HOLD', 'TB1S', 'FAV_ONLY', 'TPSL'):
        return E.decide(policy, E.trigger(kt, sm, e, cut), E.mid_at(kt, sm, act + E.T_MED), E.mid_at(kt, sm, act + E.T_UQ), act, cut)
    ts = stop(kt, sm, e, cut)
    if policy == 'SL_ONLY':
        return ts
    T = act + E.T_MED
    ref = E.mid_at(kt, sb, T) if policy in ('FAV_BID', 'SL_FAV_BID') else E.mid_at(kt, sm, T)
    not_fav = T <= cut and np.isfinite(ref) and ref < E.FAV
    if policy == 'FAV_BID':
        return T if not_fav else None
    if ts is not None and ts <= T:
        return ts
    if not_fav:
        return T
    return ts


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
    reg = MM.registry(day).set_index('mid')
    rg = pd.read_parquet(MM.S1 / day / 'registry.parquet').set_index('mid')
    by_mid = S.groupby('mid').indices
    o = collections.defaultdict(lambda: np.full(len(S), np.nan))
    kind = {(b, s, p): np.full(len(S), '', dtype=object) for b, s in E.COMBOS for p in POLICIES}
    src = MM.KP.Tape(); off = src.offset(day)
    for h, g in reg.groupby('hour'):
        mids = [int(m) for m in g.index if int(m) in by_mid]
        if not mids:
            continue
        for m, W, lst in VR.worlds_all(day, mids, src, off, reg):
            C.require(lst == 'loaded', f'{m} {lst}')
            p, st, info = VR.vm_path(W, float(reg.loc[m, 'ws']) * 1000., 50.)
            t0, end = float(info['t_start']), float(info['end_rel'])
            f = np.asarray(p['frames'], float)
            f0, fk = f[:, 0], f[:, 1]
            okb = (f[:, 2] >= 0) & (f[:, 12] <= 100) & (f[:, 2] < f[:, 12])
            midy = np.where(okb, (f[:, 2] + f[:, 12]) / 2., np.nan)
            bidy, asky = np.where(okb, f[:, 2], np.nan), np.where(okb, f[:, 12], np.nan)
            cut = float(rg.loc[m, 'we']) * 1000. - t0 - E.LAST_MS
            for i in by_mid[m]:
                t, d, lim, pay = float(S.t.iat[i]), float(S.d.iat[i]), float(S.lim.iat[i]), float(S.payout.iat[i])
                act = t + TM.FEED + TM.DECIDE - t0 + E.NET + E.HOLD
                jv = int(np.searchsorted(f0, act, 'right') - 1)
                side = 1 if d > 0 else 0
                pa, sa = TK.side_levels(f[jv], side)
                pb, sb_ = TK.side_levels(f[jv + 1], side) if jv + 1 < len(f) else (np.full(5, -1.), np.zeros(5))
                bk = {'A': (pa[None], sa[None]), 'AB': TK.minbook(pa[None], sa[None], pb[None], sb_[None])}
                k0 = int(np.searchsorted(fk, act, 'right'))
                kt = fk[k0:]
                sm = midy[k0:] if d > 0 else 100. - midy[k0:]
                sbid = bidy[k0:] if d > 0 else 100. - asky[k0:]
                for b, s in E.COMBOS:
                    fl, co, fe_ = (x[0] for x in TK.vwalk(*bk[b], np.array([d]), np.array([lim + d * s])))
                    o[f'{b}_s{int(s)}_fill'][i] = fl
                    if fl <= 0:
                        for pol in POLICIES:
                            o[f'{b}_s{int(s)}_{pol}_pnl'][i] = 0.
                            kind[(b, s, pol)][i] = 'no_fill'
                        continue
                    e = co / fl if d > 0 else 100. - co / fl
                    for pol in POLICIES:
                        x = decide2(pol, kt, sm, sbid, e, act, cut)
                        xq = xc = xf = 0.
                        kd = 'held'
                        if x is not None:
                            mt = x + E.DEC + E.NET + E.HOLD
                            if mt > end:
                                kd = 'held_beyond_path'
                            else:
                                je = int(np.searchsorted(f0, mt, 'right') - 1)
                                eP, eS = TK.side_levels(f[je], 0 if d > 0 else 1)
                                xq_, xc_, xf_ = TE.walk_q(eP[None], eS[None], np.array([-d]), np.array([1. if d > 0 else 99.]), np.array([fl]))
                                xq, xc, xf = float(xq_[0]), float(xc_[0]), float(xf_[0])
                                kd = 'sold'
                        o[f'{b}_s{int(s)}_{pol}_pnl'][i] = d * (xc + (fl - xq) * 100. * pay - co) - fe_ - xf
                        kind[(b, s, pol)][i] = kd
            C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    out = S[['day', 't', 'd', 'key', 'mid'] + list(E.SELS)].copy()
    for k_, v in o.items():
        out[k_] = v
    for (b, s, pol), v in kind.items():
        out[f'{b}_s{int(s)}_{pol}_kind'] = v
    od.mkdir(parents=True, exist_ok=True)
    out.to_parquet(fp, index=False)
    print(day, len(S), 'rss', round(F.rss_mib()), flush=True)


def read(od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (600, 601))
    reh = od is not None
    od = Path(od) if od else None
    fn = (od or H) / 'exit_follow2.json'
    C.require(not fn.exists(), 'exit_follow2.json exists (write-once)')
    src = od or OUT
    X = pd.concat([pd.read_parquet(src / f'{dh}.parquet') for dh in DAYS if (src / f'{dh}.parquet').exists()], ignore_index=True)
    X1 = pd.concat([pd.read_parquet(E.OUT / f'{dh}.parquet') for dh in DAYS if (src / f'{dh}.parquet').exists()], ignore_index=True)
    C.require(len(X) == len(X1) and (X.t.to_numpy() == X1.t.to_numpy()).all(), 'the actions differ from EXIT_FOLLOW')
    for b, s in E.COMBOS:
        for pol in REPRO:
            k = f'{b}_s{int(s)}_{pol}_pnl'
            C.require(np.allclose(X[k].to_numpy(float), X1[k].to_numpy(float), rtol=0, atol=1e-9), f'{k} not reproduced from EXIT_FOLLOW')
    day = X.day.to_numpy()
    out = dict(asof='2026-11-27', post_hoc=True, rehearsal=reh, actions=int(len(X)), reproduced=list(REPRO), selections={})
    for sel in E.SELS:
        ii = np.flatnonzero(X[sel].to_numpy(bool))
        so = dict(actions=int(len(ii)), cells={})
        for b, s in E.COMBOS:
            pre = f'{b}_s{int(s)}'
            fl = X[f'{pre}_fill'].to_numpy(float)
            filled = ii[fl[ii] > 0]
            for pol in POLICIES:
                pn = X[f'{pre}_{pol}_pnl'].to_numpy(float)
                kd = X[f'{pre}_{pol}_kind'].to_numpy()
                by = {dh: float(pn[ii][day[ii] == dh].sum() / 100.) for dh in DAYS if (day[ii] == dh).any()}
                so['cells'][f'{pre}_{pol}'] = dict(per_intended_c=float(pn[ii].sum() / (Q * len(ii))), total_usd=float(pn[ii].sum() / 100.), usd_by_day=by,
                                                  days_positive=int(sum(v > 0 for v in by.values())), sold_share=float(np.mean(kd[filled] == 'sold')) if len(filled) else None)
        out['selections'][sel] = so
    TR.dump(fn, out)
    for sel, so in out['selections'].items():
        for pre in ('AB_s3', 'AB_s10'):
            print(sel, pre, {pol: (round(so['cells'][f'{pre}_{pol}']['total_usd']), so['cells'][f'{pre}_{pol}']['days_positive']) for pol in POLICIES})


def selftest():
    kt = np.array([1000., 2000., 3000., 38000., 50000., 90000.])
    e, act, cut = 50., 0., 100000.
    sm = np.array([50., 45., 40., 62., 61., 63.]); sb = sm - 1.
    assert decide2('SL_ONLY', kt, sm, sb, e, act, cut) is None                          # 40 > 50 - 14: no stop; held
    sm2 = np.array([50., 45., 36., 62., 61., 63.])
    assert decide2('SL_ONLY', kt, sm2, sm2 - 1., e, act, cut) == 3000.                  # 36 <= 36: the stop
    sm3 = np.array([50., 52., 54., 58., 61., 63.])
    assert decide2('SL_FAV', kt, sm3, sm3 - 1., e, act, cut) == 39000.                  # no stop, mid 58 at 39 s: not a favourite
    wide_bid = np.array([50., 52., 54., 40., 41., 43.]); wide_mid = np.array([50., 52., 54., 62., 61., 63.])
    assert decide2('FAV_ONLY', kt, wide_mid, wide_bid, e, act, cut) is None             # the mid says favourite
    assert decide2('FAV_BID', kt, wide_mid, wide_bid, e, act, cut) == 39000.            # the bid says otherwise
    assert decide2('SL_FAV_BID', kt, wide_mid, wide_bid, e, act, cut) == 39000.
    assert decide2('SL_FAV', kt, wide_mid, wide_bid, e, act, cut) is None
    try:
        labels('20261022', '/nonexistent')
        raise AssertionError('an undeclared day accepted')
    except ValueError as e_:
        assert 'not a declared day' in str(e_), e_
    print('exit_follow2 selftest PASS')


if __name__ == '__main__':
    cmd, a_ = sys.argv[1], sys.argv[2:]
    if cmd == 'labels':
        labels(a_[0], a_[1] if len(a_) > 1 else None)
    elif cmd == 'read':
        read(a_[0] if a_ else None)
    else:
        dict(selftest=selftest)[cmd]()
