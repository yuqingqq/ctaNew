"""Busy book, large imbalance, far-side trades: the USER's reading of the top taker's signal, tested directly (USER 2026-11-27: "I don't think
they have extra information, but we have not interpreted the signals well enough, it looks the signal they fire will have active activities
in Orderbook, both ask/bid are changing fast. The imbalance is large, and there are frequent far side trades?"). Four consumed days;
declared before the run.
Points: SIGNAL_VS_NOISE's (public_score3's decision moments, no Binance futures move their way in the previous 300 ms): the top taker's
fires (t = its order's arrival at Predict) and random moments; label: a Binance move their way within 300 ms.
Per point, from the raw Binance futures data (stage points; bookTicker collapsed to the last state per ms, trades de-duplicated by id,
masked prints dropped), on Binance's clock, with far = the side that must empty for a move their way (the ask for an up bet):
book changes on the far side and on the near side (a best price or size change) and aggressive trades hitting the far side (their way) and
the near side (count, BTC) over (t - 1 s, t] and (t - 300 ms, t]; the same over (t, t + 100 ms] for the profile only; best-level imbalance
their way at t.
Readouts (stage read): (1) medians and means for fires, their 20 nearest visual look-alikes (VISUAL_TWINS), their 5 score look-alikes
(SIGNAL_VS_NOISE) and random moments; (2) the USER's rule: far-side changes AND near-side changes over the last 1 s at or above the random
moments' q-quantile (q = 50 / 75 / 90 %), imbalance their way >= 0.6, >= 1 far-side trade in the last 1 s: share of fires meeting it, share
of random moments meeting it, the move rate of those random moments (and of fires meeting it or not), lift over the random base; each
condition dropped in turn; the far-side-trade condition before vs after t.
Outputs (write-once): OUT/fire_activity/<day>.parquet (points), fire_activity.json (read).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/fire_activity.py \
       points <day> [<dir>] | read [<dir>]
"""
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import depletion_cause as DC  # noqa: E402
import fire_events as FE  # noqa: E402
import signal_vs_noise as SVN  # noqa: E402
import visual_twins as VT  # noqa: E402

C, TR, WT, DAYS = SVN.C, SVN.TR, SVN.WT, SVN.DAYS
OUT = SVN.BASE / 'fire_activity'
WINS = (('1s', 1000.), ('300ms', 300.))
QS = (.5, .75, .9)
IMB_MIN = .6


def points_of(day):
    P = pd.read_parquet(SVN.BASE / 'public_score3_points.parquet', columns=['day', 't', 'd', 'kind', 'y', 's_gbm_v3'])
    P = P[P.kind.isin(['fire', 'random'])]
    P = P[~P.duplicated(['day', 't', 'd', 'kind'], keep=False)]
    return P[P.day == day].reset_index(drop=True)


def count(times, lo, hi):
    return np.searchsorted(times, hi, 'right') - np.searchsorted(times, lo, 'right')


def vol(times, cum, lo, hi):
    return cum[np.searchsorted(times, hi, 'right')] - cum[np.searchsorted(times, lo, 'right')]


def points(day, od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (1800, 1801))
    C.require(day in DAYS, f'{day} is not a declared day')
    od = Path(od) if od else OUT
    fp = od / f'{day}.parquet'
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    P = points_of(day)
    t, d = P.t.to_numpy(float), P.d.to_numpy(float)
    ask_ch, bid_ch = [], []
    imb, far0, near0 = np.full(len(P), np.nan), np.full(len(P), np.nan), np.full(len(P), np.nan)
    day0 = WT.OD_day_ms(day)
    hour = ((t - day0) // 3600000).astype(int)
    for h, bk in DC.book_hours(day):
        E, bid, bq, ask, aq = FE.collapse(*bk)
        ch_a = np.r_[False, (ask[1:] != ask[:-1]) | (aq[1:] != aq[:-1])]
        ch_b = np.r_[False, (bid[1:] != bid[:-1]) | (bq[1:] != bq[:-1])]
        ask_ch.append(E[ch_a]); bid_ch.append(E[ch_b])
        sel = np.flatnonzero(hour == h)
        if len(sel):
            j = np.searchsorted(E, t[sel], 'right') - 1
            okj = j >= 0
            jj = np.maximum(j, 0)
            up = d[sel] > 0
            fq, nq = np.where(up, aq[jj], bq[jj]), np.where(up, bq[jj], aq[jj])
            far0[sel] = np.where(okj, fq, np.nan); near0[sel] = np.where(okj, nq, np.nan)
            imb[sel] = np.where(okj, (nq - fq) / np.maximum(nq + fq, 1e-12), np.nan)
        C.require(SVN.TM.F.rss_mib() < SVN.TM.F.RSS_GUARD_MIB, 'RSS over the self-guard')
    A, B = np.unique(np.concatenate(ask_ch)), np.unique(np.concatenate(bid_ch))
    tr = DC.read_trades(day).drop_duplicates('id')
    tr = tr[(tr.p > 0) & (tr.q > 0)].sort_values('E', kind='stable')
    buy = (tr.m.to_numpy() == DC.BUY_AGGR)
    Tb, Qb = tr.E.to_numpy(float)[buy], tr.q.to_numpy(float)[buy]
    Ts, Qs = tr.E.to_numpy(float)[~buy], tr.q.to_numpy(float)[~buy]
    cb, cs = np.r_[0., np.cumsum(Qb)], np.r_[0., np.cumsum(Qs)]
    up = d > 0
    o = dict(day=P.day.to_numpy(), t=t, d=d, kind=P.kind.to_numpy(), y=P.y.to_numpy(float), s=P.s_gbm_v3.to_numpy(float), imb=imb, far_size=far0, near_size=near0)
    for nm, (lo, hi) in [(w, (t - x, t)) for w, x in WINS] + [('after100', (t, t + 100.))]:
        a_n, b_n = count(A, lo, hi), count(B, lo, hi)
        o[f'far_ch_{nm}'] = np.where(up, a_n, b_n); o[f'near_ch_{nm}'] = np.where(up, b_n, a_n)
        bn, sn = count(Tb, lo, hi), count(Ts, lo, hi)
        bv, sv = vol(Tb, cb, lo, hi), vol(Ts, cs, lo, hi)
        o[f'far_tr_n_{nm}'] = np.where(up, bn, sn); o[f'near_tr_n_{nm}'] = np.where(up, sn, bn)
        o[f'far_tr_btc_{nm}'] = np.where(up, bv, sv); o[f'near_tr_btc_{nm}'] = np.where(up, sv, bv)
    L = pd.DataFrame(o)
    od.mkdir(parents=True, exist_ok=True)
    L.to_parquet(fp, index=False)
    print(day, len(L), 'fires', int((L.kind == 'fire').sum()), 'book changes ask/bid', len(A), len(B), 'trades', len(tr), 'imb nan', int(np.isnan(imb).sum()), flush=True)


def stats(L, m, cols):
    return {c: dict(median=float(np.nanmedian(L[c].to_numpy(float)[m])), mean=float(np.nanmean(L[c].to_numpy(float)[m]))) for c in cols}


def read(od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    fn = (Path(od) if od else H) / 'fire_activity.json'
    C.require(not fn.exists(), 'fire_activity.json exists (write-once)')
    L = pd.concat([pd.read_parquet((Path(od) if od else OUT) / f'{dh}.parquet') for dh in DAYS], ignore_index=True)
    fire, rnd = (L.kind == 'fire').to_numpy(), (L.kind == 'random').to_numpy()
    key = pd.Series(np.arange(len(L)), index=pd.MultiIndex.from_arrays([L.day, L.t, L.d, L.kind]))
    # visual twins (VISUAL_TWINS) and score look-alikes (SIGNAL_VS_NOISE), mapped onto these rows
    VP, VZ = VT.load()
    T = VT.twins(VP, VZ)
    vt_rows = np.concatenate([T[f]['any'][0] for f in T])
    vt = key.reindex(pd.MultiIndex.from_arrays([VP.day.to_numpy()[vt_rows], VP.t.to_numpy()[vt_rows], VP.d.to_numpy()[vt_rows], VP.kind.to_numpy()[vt_rows]])).to_numpy()
    X = L[['day', 't', 'd', 'kind']].copy(); X['s_gbm_v3'] = L.s.to_numpy()
    F_, Ct = SVN.match(X)
    sl = Ct[(Ct >= 0).all(1)].ravel()
    C.require(np.isfinite(vt.astype(float)).all(), 'a visual twin missing from the points')
    groups = dict(fires=np.flatnonzero(fire), visual_lookalikes=vt.astype(int), score_lookalikes=sl, random=np.flatnonzero(rnd))
    cols = [c for c in L.columns if c.startswith(('far_', 'near_')) or c == 'imb']
    out = dict(asof='2026-11-27', rows=int(len(L)), fires=int(fire.sum()), random=int(rnd.sum()), profiles={}, rule={})
    for g, ix in groups.items():
        out['profiles'][g] = dict(n=int(len(ix)), move_rate=float(L.y.to_numpy()[ix].mean()), **stats(L.iloc[ix].reset_index(drop=True), np.ones(len(ix), bool), cols))
    y = L.y.to_numpy(float)
    base = float(y[rnd].mean())
    thr = {q: (float(np.quantile(L.far_ch_1s[rnd], q)), float(np.quantile(L.near_ch_1s[rnd], q))) for q in QS}
    conds = lambda q, after=False: dict(far_busy=(L.far_ch_1s >= thr[q][0]).to_numpy(), near_busy=(L.near_ch_1s >= thr[q][1]).to_numpy(),
                                        imbalance=(L.imb >= IMB_MIN).to_numpy(),
                                        far_trades=((L.far_tr_n_after100 if after else L.far_tr_n_1s) >= 1).to_numpy())

    def cell(m):
        mr, mf = m & rnd, m & fire
        return dict(fires_meeting=float(mf.sum() / fire.sum()), random_meeting=float(mr.sum() / rnd.sum()), random_moves_when_met=float(y[mr].mean()) if mr.any() else None,
                    lift=float(y[mr].mean() / base) if mr.any() else None, random_n=int(mr.sum()), fires_moves_when_met=float(y[mf].mean()) if mf.any() else None,
                    fires_moves_when_not=float(y[fire & ~m].mean()) if (fire & ~m).any() else None)
    for q in QS:
        cd = conds(q)
        allm = np.logical_and.reduce(list(cd.values()))
        r = dict(thresholds=dict(far_changes_1s=thr[q][0], near_changes_1s=thr[q][1], imbalance=IMB_MIN, far_trades_1s=1), all=cell(allm), drop_one={},
                 single={k: cell(v) for k, v in cd.items()})
        for k in cd:
            r['drop_one'][k] = cell(np.logical_and.reduce([v for kk, v in cd.items() if kk != k]))
        ca = conds(q, after=True)
        r['far_trades_after_t_instead'] = cell(np.logical_and.reduce(list(ca.values())))
        out['rule'][f'q{int(q * 100)}'] = r
    out['random_base_move_rate'] = base
    TR.dump(fn, out)
    print(json.dumps({g: {k: (v if not isinstance(v, dict) else round(v['median'], 3)) for k, v in p.items()} for g, p in out['profiles'].items()}, indent=1)[:3000])
    for q, r in out['rule'].items():
        print(q, 'thr', r['thresholds'], 'ALL', r['all'])
        for k, v in r['drop_one'].items():
            print('   drop', k, v)
        print('   after-t trades', r['far_trades_after_t_instead'])


if __name__ == '__main__':
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == 'points':
        points(a[0], a[1] if len(a) > 1 else None)
    else:
        read(a[0] if a else None)
