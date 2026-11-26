"""Which of the top taker's bets pay, and how predictive is its book state at random times? (USER 2026-11-26: "keep these findings recorded,
any other more information we can find"). Readout; four consumed days; no strategy change; descriptive (the state rule is written from
what the top taker's sends showed on these same days, so its precision here is in-sample).
Points: the top taker's decisions (bursts; arrival a = settlement - 150 - c) and random times (30,000 per day, seeded, uniform over hours
with a bookTicker file, >= 1.5 s from hour boundaries), each random time read in both directions.
At t, direction d (fire_events' events; opp = the far side that empties for a move their way): trades their way / against them (count)
and far-side new orders, far-side cancels, near-side new orders (BTC) over (t - 100, t]; imbalance their way and the far / near best size
at t; a move their way (timeline.py's moves) in (t - 300, t] (momentum) and against (bounce); outcome = a move their way within 100 / 150
ms after t. The top taker's 1 s value per share (winners_trigger points, by day / market / M50 send time).
Declared before the result: the STATE = imbalance their way >= 0.3, no trade against in the last 100 ms, far-side new orders in the last
100 ms at or below the random-time median, and no move their way in the last 300 ms. Buckets: imbalance < 0.3 / 0.3-0.6 / >= 0.6 x trades
against 0 / >= 1 x far-side new orders <= / > the random-time median.
Readings: the decisions by class (ahead of a move within 300 ms / ahead, none within 300 ms / just after a move: share, move within 150 ms,
1 s value); decisions and random times in / out of the STATE and per bucket (share, move within 100 / 150 ms, decisions' value).
Output (write-once): state_value.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/state_value.py [dir]
"""
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import fire_events as FE  # noqa: E402
import timeline as TL  # noqa: E402
import timeline2 as T2  # noqa: E402

DC, C, TR, WT = FE.DC, FE.C, FE.TR, FE.WT
W_MS, N_RAND, IMB_STATE = 100., 30000, .3
IMB_B = (('<0.3', -9., .3), ('0.3-0.6', .3, .6), ('>=0.6', .6, 9.))


def vol_in(E, cA, t, w):
    return cA[np.searchsorted(E, t, 'right')] - cA[np.searchsorted(E, t - w, 'right')]


def feats(ev, t, d):
    o = {}
    for name, role, kind, fn in (('buys', 'opp', 'trade', 'n'), ('sells', 'near', 'trade', 'n'), ('far_adds', 'opp', 'add', 'v'),
                                 ('far_cxl', 'opp', 'cancel', 'v'), ('near_adds', 'near', 'add', 'v')):
        x = np.full(len(t), np.nan)
        for dv in (1., -1.):
            m = d == dv
            if m.any():
                E, cA = FE.role_events(ev, role, kind, dv)
                x[m] = FE.count_in(E, t[m], W_MS) if fn == 'n' else vol_in(E, cA, t[m], W_MS)
        o[name] = x
    nxt, mom, bnc = np.full(len(t), np.nan), np.zeros(len(t), bool), np.zeros(len(t), bool)
    for dv in (1., -1.):
        m = d == dv
        if not m.any():
            continue
        Mw, Mo = TL.moves(ev, dv), TL.moves(ev, -dv)
        j = np.searchsorted(Mw, t[m], 'right')
        nxt[m] = np.where(j < len(Mw), Mw[np.minimum(j, max(len(Mw) - 1, 0))] - t[m], np.inf) if len(Mw) else np.inf
        mom[m] = (np.searchsorted(Mw, t[m], 'right') - np.searchsorted(Mw, t[m] - 300., 'right')) > 0
        bnc[m] = (np.searchsorted(Mo, t[m], 'right') - np.searchsorted(Mo, t[m] - 300., 'right')) > 0
    o.update(next_move=nxt, momentum=mom, bounce=bnc)
    return o


def mean(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else None


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (1200, 1201))
    fn = (Path(sys.argv[1]) if len(sys.argv) > 1 else H) / 'state_value.json'
    C.require(not fn.exists(), 'state_value.json exists (write-once)')
    c = float(json.loads((H / 'fire_events.json').read_text())['groups']['TOP1']['arrival_offset_ms'])
    A = pd.read_parquet(WT.OUT / 'fire_times_orders.parquet')
    A = A[A.t_set.notna() & (A['rank'] == 1)]
    B = A.sort_values(['burst', 'fire'], kind='stable').drop_duplicates('burst', keep='first').reset_index(drop=True)
    B['a'] = B.fire - c
    V1 = pd.read_parquet(WT.OUT / 'winners_trigger_points.parquet', columns=['day', 'mid', 's', 'y', 'grp', 'v1'])
    V1 = V1[(V1.y == 1) & (V1.grp == 'TOP1')].drop_duplicates(['day', 'mid', 's'])
    B = B.merge(V1[['day', 'mid', 's', 'v1']].rename(columns={'s': 's_m50'}), on=['day', 'mid', 's_m50'], how='left', validate='many_to_one')
    rng = np.random.default_rng(WT.SEED + 1)
    rows = []
    for day in WT.DAYS:
        Bd = B[B.day == day]
        d0 = float(WT.OD_day_ms(day))
        rt = np.sort(rng.uniform(d0 + 2000., d0 + 86400000. - 2000., N_RAND))
        rt = np.floor(rt[T2.good(day, rt)])
        P = pd.concat([pd.DataFrame(dict(t=Bd.a.to_numpy(float), d=Bd.d.to_numpy(float), kind='send', v1=Bd.v1.to_numpy(float))),
                       pd.DataFrame(dict(t=np.r_[rt, rt], d=np.r_[np.ones(len(rt)), -np.ones(len(rt))], kind='random', v1=np.nan))], ignore_index=True)
        ev, st, traj, _ = FE.day_data(day, P)
        P['status'] = st
        P['imb'], P['far'], P['near'] = traj['imb_0'], traj['opp_0'], traj['near_0']
        for k, v in feats(ev, P.t.to_numpy(float), P.d.to_numpy(float)).items():
            P[k] = v
        P['day'] = day
        rows.append(P[P.status == 'ok'])
        C.require(DC.F.rss_mib() < DC.F.RSS_GUARD_MIB, 'RSS over the self-guard')
        print(day, int((P.kind == 'send').sum()), int((P.kind == 'random').sum()), int((P.status == 'ok').sum()), flush=True)
    X = pd.concat(rows, ignore_index=True)
    R, S = X[X.kind == 'random'], X[X.kind == 'send']
    adds_med = float(R.far_adds.median())
    X['low_refill'] = X.far_adds <= adds_med
    X['state'] = (X.imb >= IMB_STATE) & (X.sells == 0) & X.low_refill & ~X.momentum
    X['m100'], X['m150'] = X.next_move <= 100., X.next_move <= 150.
    R, S = X[X.kind == 'random'], X[X.kind == 'send']

    def blk(Z):
        return dict(n=int(len(Z)), move_100=mean(Z.m100), move_150=mean(Z.m150), v1=mean(Z.v1), v1_n=int(Z.v1.notna().sum()))
    cls = {'ahead_move_300': (~S.momentum) & (S.next_move <= 300.), 'ahead_no_move_300': (~S.momentum) & (S.next_move > 300.), 'after_a_move': S.momentum}
    out = dict(asof='2026-11-26', window_ms=W_MS, far_adds_random_median_btc=adds_med, state_rule=dict(imb_min=IMB_STATE, sells=0, far_adds='<= random median', momentum=False),
               decisions=dict(n=int(len(S)), all=blk(S), by_class={k: dict(share=mean(m), **blk(S[m])) for k, m in cls.items()}),
               state=dict(random_share=mean(R.state), random_in=blk(R[R.state]), random_out=blk(R[~R.state]), decisions_share=mean(S.state),
                          decisions_in=blk(S[S.state]), decisions_out=blk(S[~S.state])),
               buckets={})
    for nm, lo_, hi_ in IMB_B:
        for sl, sm in (('no_sells', R.sells == 0), ('sells', R.sells > 0)):
            for rl in (True, False):
                mr = (R.imb >= lo_) & (R.imb < hi_) & sm & (R.low_refill == rl) & ~R.momentum
                ss = (S.imb >= lo_) & (S.imb < hi_) & ((S.sells == 0) if sl == 'no_sells' else (S.sells > 0)) & (S.low_refill == rl) & ~S.momentum
                key = f"imb {nm} | {sl} | refill {'low' if rl else 'high'}"
                out['buckets'][key] = dict(random_share=mean(mr), random=blk(R[mr]), decisions_share=mean(ss), decisions=blk(S[ss]))
    out['random_baseline'] = blk(R[~R.momentum])
    TR.dump(fn, out)
    print(json.dumps(dict(decisions=out['decisions'], state=out['state']), indent=1))


if __name__ == '__main__':
    main()
