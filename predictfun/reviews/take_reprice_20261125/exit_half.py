"""Exit half, hold half to settlement (USER 2026-11-27: "We check if we can exit half like top takers, then hold to settlement. Check the
performance"; the top taker sells back about half of what it opens and holds the rest, EXIT_TIMES). POST HOC: added after EXIT_FOLLOW /
EXIT_FOLLOW2; same entries (FILL_AWARE's selections, 50-share IOC at the known best + 3c / + 10c, conservative and optimistic entry books),
same exit rule (IOC at every shown level through the 172 ms hold on the snapshot before the exit match), same days (four consumed days).
Declared before this run.
Policies (sell a fraction of the filled shares, the rest held to settlement; nothing in the window's last 30 s):
  HALF_1S: half sold 1 s after the entry match. HALF_39: half sold 39 s after it (its median take-back time).
  HALF_FAV: at 39 s all of a non-favourite (the side's mid below 60c) sold, half of a favourite.
  Reproduced from EXIT_FOLLOW (a check): HOLD, TB1S, TB39, FAV_ONLY.
Readings per selection x entry: P&L in $ and per intended share, by day, days > 0; risk: the P&L summed per market, its standard deviation
across markets and the share of markets losing; the worst day.
Outputs (write-once): OUT/exit_half/<day>.parquet, exit_half.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/exit_half.py \
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
OUT = WT.OUT / 'exit_half'
DAYS, Q = E.DAYS, E.Q
POLICIES = ('HOLD', 'TB1S', 'TB39', 'FAV_ONLY', 'HALF_1S', 'HALF_39', 'HALF_FAV')
REPRO = ('HOLD', 'TB1S', 'TB39', 'FAV_ONLY')


def decide3(policy, kt, sm, e, act, cut):
    """-> (exit decision time, fraction sold) or None (all held to settlement)."""
    if policy in REPRO:
        x = E.decide(policy, E.trigger(kt, sm, e, cut), E.mid_at(kt, sm, act + E.T_MED), E.mid_at(kt, sm, act + E.T_UQ), act, cut)
        return None if x is None else (x, 1.)
    if policy == 'HALF_1S':
        return (act + 1000., .5)
    T = act + E.T_MED
    if T > cut:
        return None
    if policy == 'HALF_39':
        return (T, .5)
    m = E.mid_at(kt, sm, T)
    return (T, 1.) if (np.isfinite(m) and m < E.FAV) else (T, .5)


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
                for b, s in E.COMBOS:
                    fl, co, fe_ = (x[0] for x in TK.vwalk(*bk[b], np.array([d]), np.array([lim + d * s])))
                    o[f'{b}_s{int(s)}_fill'][i] = fl
                    e = (co / fl if d > 0 else 100. - co / fl) if fl > 0 else np.nan
                    for pol in POLICIES:
                        if fl <= 0:
                            o[f'{b}_s{int(s)}_{pol}_pnl'][i] = 0.
                            continue
                        dec = decide3(pol, kt, sm, e, act, cut)
                        xq = xc = xf = 0.
                        if dec is not None:
                            x, frac = dec
                            mt = x + E.DEC + E.NET + E.HOLD
                            if mt <= end:
                                je = int(np.searchsorted(f0, mt, 'right') - 1)
                                eP, eS = TK.side_levels(f[je], 0 if d > 0 else 1)
                                xq_, xc_, xf_ = TE.walk_q(eP[None], eS[None], np.array([-d]), np.array([1. if d > 0 else 99.]), np.array([frac * fl]))
                                xq, xc, xf = float(xq_[0]), float(xc_[0]), float(xf_[0])
                        o[f'{b}_s{int(s)}_{pol}_pnl'][i] = d * (xc + (fl - xq) * 100. * pay - co) - fe_ - xf
                        o[f'{b}_s{int(s)}_{pol}_sold'][i] = xq / fl
            C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    out = S[['day', 't', 'd', 'key', 'mid'] + list(E.SELS)].copy()
    for k_, v in o.items():
        out[k_] = v
    od.mkdir(parents=True, exist_ok=True)
    out.to_parquet(fp, index=False)
    print(day, len(S), 'rss', round(F.rss_mib()), flush=True)


def read(od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (600, 601))
    reh = od is not None
    od = Path(od) if od else None
    fn = (od or H) / 'exit_half.json'
    C.require(not fn.exists(), 'exit_half.json exists (write-once)')
    src = od or OUT
    X = pd.concat([pd.read_parquet(src / f'{dh}.parquet') for dh in DAYS if (src / f'{dh}.parquet').exists()], ignore_index=True)
    X1 = pd.concat([pd.read_parquet(E.OUT / f'{dh}.parquet') for dh in DAYS if (src / f'{dh}.parquet').exists()], ignore_index=True)
    C.require(len(X) == len(X1) and (X.t.to_numpy() == X1.t.to_numpy()).all(), 'the actions differ from EXIT_FOLLOW')
    for b, s in E.COMBOS:
        for pol in REPRO:
            k = f'{b}_s{int(s)}_{pol}_pnl'
            C.require(np.allclose(X[k].to_numpy(float), X1[k].to_numpy(float), rtol=0, atol=1e-9), f'{k} not reproduced from EXIT_FOLLOW')
    day = X.day.to_numpy()
    mk = (X.day.astype(str) + ':' + X.mid.astype(str)).to_numpy()
    out = dict(asof='2026-11-27', post_hoc=True, rehearsal=reh, actions=int(len(X)), reproduced=list(REPRO), selections={})
    for sel in E.SELS:
        ii = np.flatnonzero(X[sel].to_numpy(bool))
        so = dict(actions=int(len(ii)), markets=int(np.unique(mk[ii]).size), cells={})
        for b, s in E.COMBOS:
            pre = f'{b}_s{int(s)}'
            for pol in POLICIES:
                pn = X[f'{pre}_{pol}_pnl'].to_numpy(float)
                by = {dh: float(pn[ii][day[ii] == dh].sum() / 100.) for dh in DAYS if (day[ii] == dh).any()}
                per_mkt = pd.Series(pn[ii] / 100.).groupby(mk[ii]).sum()
                so['cells'][f'{pre}_{pol}'] = dict(total_usd=float(pn[ii].sum() / 100.), per_intended_c=float(pn[ii].sum() / (Q * len(ii))), usd_by_day=by,
                                                  days_positive=int(sum(v > 0 for v in by.values())), worst_day_usd=float(min(by.values())),
                                                  market_sd_usd=float(per_mkt.std()), markets_losing=float((per_mkt < 0).mean()))
        out['selections'][sel] = so
    TR.dump(fn, out)
    for sel, so in out['selections'].items():
        for pre in ('AB_s3', 'AB_s10'):
            print(sel, pre, {pol: (round(so['cells'][f'{pre}_{pol}']['total_usd']), so['cells'][f'{pre}_{pol}']['days_positive'],
                                   round(so['cells'][f'{pre}_{pol}']['market_sd_usd'], 1), round(so['cells'][f'{pre}_{pol}']['worst_day_usd'])) for pol in POLICIES})


def selftest():
    kt = np.array([1000., 2000., 38000., 50000.])
    sm = np.array([50., 55., 58., 70.]); sm2 = np.array([50., 55., 65., 70.])
    act, cut, e = 0., 100000., 50.
    assert decide3('HALF_1S', kt, sm, e, act, cut) == (1000., .5)
    assert decide3('HALF_39', kt, sm, e, act, cut) == (39000., .5) and decide3('HALF_39', kt, sm, e, act, 30000.) is None
    assert decide3('HALF_FAV', kt, sm, e, act, cut) == (39000., 1.)                      # 58 at 39 s: not a favourite, all sold
    assert decide3('HALF_FAV', kt, sm2, e, act, cut) == (39000., .5)                     # 65: a favourite, half sold
    assert decide3('HOLD', kt, sm, e, act, cut) is None and decide3('TB1S', kt, sm, e, act, cut) == (1000., 1.)
    # selling half walks half the depth: never more than the shares asked
    q, c_, f_ = TE.walk_q(np.array([[60., 59., -1., -1., -1.]]), np.array([[10., 40., 0., 0., 0.]]), np.array([-1.]), np.array([1.]), np.array([25.]))
    assert q[0] == 25. and abs(c_[0] - (10 * 60. + 15 * 59.)) < 1e-9, (q, c_)
    try:
        labels('20261022', '/nonexistent')
        raise AssertionError('an undeclared day accepted')
    except ValueError as e_:
        assert 'not a declared day' in str(e_), e_
    print('exit_half selftest PASS')


if __name__ == '__main__':
    cmd, a_ = sys.argv[1], sys.argv[2:]
    if cmd == 'labels':
        labels(a_[0], a_[1] if len(a_) > 1 else None)
    elif cmd == 'read':
        read(a_[0] if a_ else None)
    else:
        dict(selftest=selftest)[cmd]()
