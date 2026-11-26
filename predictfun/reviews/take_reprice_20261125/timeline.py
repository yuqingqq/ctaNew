"""One race on one clock, t = 0 at the Binance move (USER 2026-11-26: "show me a clear timeline of the events ... when Binance happens
something, like top level moves, taker predicts, fire, with clear time, and what we do, I want to check the difference"). Readout; four
consumed days; no strategy change.
The move = the first Binance futures mid change in the taker's direction (its opposite best level empties or its near side improves:
fire_events' gone / improve events), as fill_rate's onset (the first mid tick of the move).
Top taker (every burst; arrival a = settlement - 150 - c, c = fire_events' arrival offset): the next move after a; 'ahead' = no move in the
250 ms before a, 'after a move' = one (time since it). For 'ahead' sends whose move comes within 300 ms: the lag quantiles, and the
imbalance and best sizes at move - 300 .. 0 ms; the same for a seeded sample of all Binance moves of each day (5,000 per direction per
day): does every move have this build-up? Our side (REPRICE-8) and Predict's makers from fill_rate.json, computed there on the REPRICE
events with the first tick as onset: our trigger's lag, + FEED + DECIDE = our decision, + 22 ms (assumed path) = our arrival, + 150 ms =
our match; the pulled quotes' vanish times; the other takers' orders (M50 clock).
Output (write-once): timeline.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/timeline.py [dir]
"""
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
sys.path.insert(0, str(H.parent / 'take_stream_20261125'))
import fire_events as FE  # noqa: E402
import stream_replay as SR  # noqa: E402

DC, C, TR, WT = FE.DC, FE.C, FE.TR, FE.WT
AHEAD_MS, MAX_LAG, N_ALL, NET = 250., 300., 5000, 22.
QS = (.1, .25, .5, .75, .9)


def moves(ev, d):
    """Times of mid changes in direction d: the opposite level empties or the near side improves."""
    a = ev[(FE.side_of('opp', d), 'gone')][0]
    b = ev[(FE.side_of('near', d), 'improve')][0]
    return np.unique(np.r_[a, b])


def qs(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {f'p{int(q * 100)}': float(np.quantile(x, q)) for q in QS} if len(x) else None


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (1200, 1201))
    fn = (Path(sys.argv[1]) if len(sys.argv) > 1 else H) / 'timeline.json'          # an argument: rehearsal dir
    C.require(not fn.exists(), 'timeline.json exists (write-once)')
    FJ = json.loads((H / 'fire_events.json').read_text())
    c = float(FJ['groups']['TOP1']['arrival_offset_ms'])
    A = pd.read_parquet(WT.OUT / 'fire_times_orders.parquet')
    A = A[A.t_set.notna() & (A['rank'] == 1)]
    B = A.sort_values(['burst', 'fire'], kind='stable').drop_duplicates('burst', keep='first').reset_index(drop=True)
    B['a'] = B.fire - c
    rng = np.random.default_rng(WT.SEED)
    rows, traj_rows = [], []
    for day in WT.DAYS:
        Bd = B[B.day == day].reset_index(drop=True)
        if not len(Bd):
            continue
        P = pd.DataFrame(dict(t=Bd.a.to_numpy(float), d=Bd.d.to_numpy(float)))
        ev, st, _, _ = FE.day_data(day, P)
        t, d = P.t.to_numpy(), P.d.to_numpy()
        nxt, prv = np.full(len(t), np.nan), np.full(len(t), np.nan)
        for dv in (1., -1.):
            m = d == dv
            M = moves(ev, dv)
            j = np.searchsorted(M, t[m], 'right')
            nxt[m] = np.where(j < len(M), M[np.minimum(j, len(M) - 1)], np.nan)
            prv[m] = np.where(j > 0, M[np.maximum(j - 1, 0)], np.nan)
        R = pd.DataFrame(dict(day=day, a=t, d=d, status=st, next_move=nxt, prev_move=prv))
        rows.append(R)
        ok = (R.status == 'ok') & ((R.a - R.prev_move > AHEAD_MS) | R.prev_move.isna()) & (R.next_move - R.a <= MAX_LAG)
        pts = [pd.DataFrame(dict(t=R.next_move[ok].to_numpy(), d=R.d[ok].to_numpy(), kind='top1'))]
        for dv in (1., -1.):
            M = moves(ev, dv)
            M = M[np.abs(M - (WT.OD_day_ms(day) + np.rint((M - WT.OD_day_ms(day)) / 3600000.) * 3600000.)) >= FE.EDGE_MS]
            pick = np.sort(rng.choice(M, min(N_ALL, len(M)), replace=False)) if len(M) else M
            pts.append(pd.DataFrame(dict(t=pick, d=dv, kind='all')))
        Q = pd.concat(pts, ignore_index=True)
        _, st2, traj, _ = FE.day_data(day, Q)
        Q['status'] = st2
        for k, v in traj.items():
            Q[k] = v
        traj_rows.append(Q[Q.status == 'ok'])
        C.require(DC.F.rss_mib() < DC.F.RSS_GUARD_MIB, 'RSS over the self-guard')
        print(day, len(R), int(ok.sum()), len(Q), flush=True)
    R = pd.concat(rows, ignore_index=True)
    R = R[R.status == 'ok']
    ahead = (R.a - R.prev_move > AHEAD_MS) | R.prev_move.isna()
    lag = R.next_move - R.a
    T = pd.concat(traj_rows, ignore_index=True)
    fr = json.loads((H / 'fill_rate.json').read_text())['configs']['REPRICE-8-172']
    bl = fr['trigger']['bn_lag_ms']
    dec = [x + SR.FEED + SR.DECIDE for x in bl]
    out = dict(asof='2026-11-26', arrival_offset_ms=c, ahead_ms=AHEAD_MS, max_lag_ms=MAX_LAG,
               top1=dict(sends=int(len(R)), ahead=float(ahead.mean()), after_a_move=float((~ahead).mean()),
                         ahead_move_within_150=float((lag[ahead] <= 150.).mean()), ahead_move_within_300=float((lag[ahead] <= 300.).mean()),
                         ahead_lag_ms=qs(lag[ahead & (lag <= MAX_LAG)]), after_a_move_since_ms=qs((R.a - R.prev_move)[~ahead]),
                         arrival_rel_move_ms=qs(-lag[ahead & (lag <= MAX_LAG)])),
               build_up={k: {x: dict(imb_mean=float(T[T.kind == k][f'imb_{int(x)}'].mean()), opp_median=float(T[T.kind == k][f'opp_{int(x)}'].median()),
                                    near_median=float(T[T.kind == k][f'near_{int(x)}'].median())) for x in FE.LAGS} | {'n': int((T.kind == k).sum())}
                         for k in ('top1', 'all')},
               reprice8=dict(source='fill_rate.json REPRICE-8-172 (REPRICE events, onset = first tick of the move; quantiles p25 / p50 / p75)',
                             trigger_after_move_ms=bl, decision_after_move_ms=dec, feed_ms=SR.FEED, decide_ms=SR.DECIDE, net_ms_assumed=NET,
                             arrival_after_move_ms=[x + NET for x in dec], match_after_move_ms=[x + NET + SR.HOLD for x in dec],
                             predict_makers_pull_after_move_ms=dict(zip(('p5', 'p10', 'p25', 'p50', 'p75'), fr['clock']['pulled_gone_from_onset_ms'])),
                             other_takers_arrival_after_move_ms_m50=dict(zip(('p5', 'p10', 'p25', 'p50', 'p75'), fr['clock']['taken_order_from_onset_ms']))))
    TR.dump(fn, out)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
