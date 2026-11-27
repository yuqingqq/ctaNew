"""Does the near side keep building against trades before the top taker's fires? (USER 2026-11-27: "I feel they are reading the book better,
if the near side keep accumulating even when there are far side sell trades, it shows a stronger trend for near side"). Four consumed days;
declared before the run. Two readings of the USER's words, both tested:
(A) absorption: aggressive orders against the bet hit the near side and the near side is refilled at least as much as was taken;
(B) two-sided pressure: aggressive orders their way hit the far side while the near side keeps growing.
Points: SIGNAL_VS_NOISE's (fires, random moments; no Binance move their way in the previous 300 ms); label: a move their way within 300 ms.
Per point (stage points), Binance futures best level on Binance's clock (fire_events' decomposition: adds / cancels at an unchanged best price
net of the aggressive volume at that price; a level emptied away = gone), over (t - 1 s, t] and (t - 300 ms, t]: near-side adds, cancels
and trades against (BTC), far-side trades their way, near net change = adds - cancels - trades against; imbalance at t.
Readouts (stage read): profiles (medians, means, share meeting A / B) for fires, their visual and score look-alikes and random moments; rules
A, B, A + imbalance >= 0.6, B + imbalance >= 0.6 (1 s and 300 ms): share of fires meeting, share of random moments meeting, their move rate,
lift, fires' move rate when meeting. Added after the rehearsal (graded versions, 1 s): near net growth in the random moments' top 10 / 1 % with
imbalance >= 0.6 and far-side trades their way; strong absorption = trades against in the random moments' top 10 % fully refilled (adds >=
trades against, near net >= 0), alone and with imbalance >= 0.6.
Outputs (write-once): OUT/absorption/<day>.parquet (points), absorption.json (read).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/absorption.py \
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
import fire_activity as FA  # noqa: E402
import fire_events as FE  # noqa: E402
import signal_vs_noise as SVN  # noqa: E402
import visual_twins as VT  # noqa: E402

C, TR, WT, DAYS = SVN.C, SVN.TR, SVN.WT, SVN.DAYS
OUT = SVN.BASE / 'absorption'
WINS = (('1s', 1000.), ('300ms', 300.))
IMB_MIN = .6


def series(ev):
    e, a = ev
    o = np.argsort(e, kind='stable')
    return e[o], np.r_[0., np.cumsum(a[o])]


def points(day, od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (2400, 2401))
    C.require(day in DAYS, f'{day} is not a declared day')
    od = Path(od) if od else OUT
    fp = od / f'{day}.parquet'
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    P = FA.points_of(day)
    t, d = P.t.to_numpy(float), P.d.to_numpy(float)
    day0 = WT.OD_day_ms(day)
    ix = DC.build_index(DC.read_trades(day), day0)
    parts = {k: ([], []) for k in [(s, kk) for s in ('ask', 'bid') for kk in ('add', 'cancel')]}
    imb = np.full(len(P), np.nan)
    hour = ((t - day0) // 3600000).astype(int)
    for h, bk in DC.book_hours(day):
        h0 = day0 + h * 3600000.
        E, bid, bq, ask, aq = FE.collapse(*bk)
        ev = FE.book_events((E, bid, bq, ask, aq), ix)
        for k in parts:
            e, a = ev[k]
            keep = (e >= h0) & (e < h0 + 3600000.)
            parts[k][0].append(e[keep]); parts[k][1].append(a[keep])
        sel = np.flatnonzero(hour == h)
        if len(sel):
            j = np.maximum(np.searchsorted(E, t[sel], 'right') - 1, 0)
            up = d[sel] > 0
            fq, nq = np.where(up, aq[j], bq[j]), np.where(up, bq[j], aq[j])
            imb[sel] = (nq - fq) / np.maximum(nq + fq, 1e-12)
        C.require(SVN.TM.F.rss_mib() < SVN.TM.F.RSS_GUARD_MIB, 'RSS over the self-guard')
    S = {k: series((np.concatenate(v[0]), np.concatenate(v[1]))) for k, v in parts.items()}
    S.update({k: series(v) for k, v in FE.trade_events(ix).items()})       # ('ask', 'trade') = aggressive buys; ('bid', 'trade') = aggressive sells
    up = d > 0
    o = dict(day=P.day.to_numpy(), t=t, d=d, kind=P.kind.to_numpy(), y=P.y.to_numpy(float), imb=imb)
    for nm, w in WINS:
        lo, hi = t - w, t
        val = {k: FA.vol(e, c, lo, hi) for k, (e, c) in S.items()}
        near = lambda kind: np.where(up, val[('bid', kind)], val[('ask', kind)])
        far = lambda kind: np.where(up, val[('ask', kind)], val[('bid', kind)])
        o[f'near_add_{nm}'] = near('add'); o[f'near_cancel_{nm}'] = near('cancel')
        o[f'near_trade_against_{nm}'] = near('trade'); o[f'far_trade_their_way_{nm}'] = far('trade')
        o[f'far_add_{nm}'] = far('add'); o[f'far_cancel_{nm}'] = far('cancel')
        o[f'near_net_{nm}'] = o[f'near_add_{nm}'] - o[f'near_cancel_{nm}'] - o[f'near_trade_against_{nm}']
    L = pd.DataFrame(o)
    od.mkdir(parents=True, exist_ok=True)
    L.to_parquet(fp, index=False)
    print(day, len(L), 'fires', int((L.kind == 'fire').sum()), {k: len(v[0]) for k, v in S.items()}, flush=True)


def rules(L, nm):
    na, nt, ft, nn = (L[f'near_add_{nm}'].to_numpy(), L[f'near_trade_against_{nm}'].to_numpy(), L[f'far_trade_their_way_{nm}'].to_numpy(), L[f'near_net_{nm}'].to_numpy())
    A = (nt > 0) & (na >= nt)
    B = (ft > 0) & (na > 0) & (nn > 0)
    big = (L.imb >= IMB_MIN).to_numpy()
    out = dict(A_absorption=A, B_two_sided=B, A_and_imbalance=A & big, B_and_imbalance=B & big, imbalance_only=big)
    if nm == '1s':                                                    # graded versions, added after the rehearsal
        rnd = (L.kind == 'random').to_numpy()
        for q in (.9, .99):
            g = nn >= np.quantile(nn[rnd], q)
            out[f'near_growth_top{round((1 - q) * 100):g}_and_imbalance_and_far_trades'] = g & big & (ft > 0)
        strong = (nt >= np.quantile(nt[rnd], .9)) & (nt > 0) & (na >= nt) & (nn >= 0)
        out['strong_absorption'] = strong
        out['strong_absorption_and_imbalance'] = strong & big
    return out


def read(od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    fn = (Path(od) if od else H) / 'absorption.json'
    C.require(not fn.exists(), 'absorption.json exists (write-once)')
    L = pd.concat([pd.read_parquet((Path(od) if od else OUT) / f'{dh}.parquet') for dh in DAYS], ignore_index=True)
    fire, rnd = (L.kind == 'fire').to_numpy(), (L.kind == 'random').to_numpy()
    y = L.y.to_numpy(float)
    base = float(y[rnd].mean())
    key = pd.Series(np.arange(len(L)), index=pd.MultiIndex.from_arrays([L.day, L.t, L.d, L.kind]))
    VP, VZ = VT.load()
    T = VT.twins(VP, VZ)
    vr = np.concatenate([T[f]['any'][0] for f in T])
    vt = key.reindex(pd.MultiIndex.from_arrays([VP.day.to_numpy()[vr], VP.t.to_numpy()[vr], VP.d.to_numpy()[vr], VP.kind.to_numpy()[vr]])).to_numpy()
    C.require(np.isfinite(vt.astype(float)).all(), 'a visual twin missing from the points')
    X = L[['day', 't', 'd', 'kind']].copy()
    X['s_gbm_v3'] = pd.read_parquet(SVN.BASE / 'public_score3_points.parquet', columns=['day', 't', 'd', 'kind', 's_gbm_v3']).drop_duplicates(['day', 't', 'd', 'kind'], keep=False) \
        .merge(L[['day', 't', 'd', 'kind']], on=['day', 't', 'd', 'kind'], how='right').s_gbm_v3.to_numpy()
    _, Ct = SVN.match(X)
    groups = dict(fires=np.flatnonzero(fire), visual_lookalikes=vt.astype(int), score_lookalikes=Ct[(Ct >= 0).all(1)].ravel(), random=np.flatnonzero(rnd))
    out = dict(asof='2026-11-27', rows=int(len(L)), base_move_rate=base, profiles={}, rules={})
    cols = [c for c in L.columns if c.startswith(('near_', 'far_')) or c == 'imb']
    for nm, _ in WINS:
        R = rules(L, nm)
        out['rules'][nm] = {}
        for rn, m in R.items():
            mr, mf = m & rnd, m & fire
            out['rules'][nm][rn] = dict(fires_meeting=float(mf.mean() / fire.mean()), random_meeting=float(mr.sum() / rnd.sum()),
                                        random_moves_when_met=float(y[mr].mean()) if mr.any() else None, lift=float(y[mr].mean() / base) if mr.any() else None,
                                        fires_moves_when_met=float(y[mf].mean()) if mf.any() else None,
                                        lookalikes_meeting=float(m[groups['visual_lookalikes']].mean()))
    for g, ix in groups.items():
        p = dict(n=int(len(ix)), move_rate=float(y[ix].mean()))
        for c in cols:
            v = L[c].to_numpy(float)[ix]
            p[c] = dict(median=float(np.nanmedian(v)), mean=float(np.nanmean(v)), share_positive=float(np.nanmean(v > 0)))
        for nm, _ in WINS:
            for rn, m in rules(L, nm).items():
                p[f'meets_{rn}_{nm}'] = float(m[ix].mean())
        out['profiles'][g] = p
    TR.dump(fn, out)
    for g, p in out['profiles'].items():
        print(g, 'move', round(p['move_rate'], 3), {k: round(v, 3) for k, v in p.items() if k.startswith('meets_') and k.endswith('1s')},
              {c: round(p[c]['mean'], 3) for c in ('near_add_1s', 'near_trade_against_1s', 'far_trade_their_way_1s', 'near_net_1s')})
    for nm, r in out['rules'].items():
        for rn, v in r.items():
            print(nm, rn, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()})


if __name__ == '__main__':
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == 'points':
        points(a[0], a[1] if len(a) > 1 else None)
    else:
        read(a[0] if a else None)
