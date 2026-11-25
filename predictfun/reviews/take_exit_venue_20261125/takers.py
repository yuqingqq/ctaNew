"""Who took the quotes our 172 ms takes missed, when they sent, and whether their fills pay. Diagnostic on the consumed v3 days.
1. Identity: the day's REST match records, keyed as pf_step1 does (tx + maker hash, earliest receipt), joined to step1 legs_timed2 rows on
   (mid, exec_s, first_recv_ns, side, ptick, shares, price_executed); the join rate and ambiguous keys are reported. Hours read: the day's
   own 24 hours plus the next day's 00-01Z only when that day is permitted (never the sealed block, never 11-04); asserted before any open.
2. Consuming legs: for each missed take (172 ms BASE ledger, reason limit_moved), the legs on the aimed book side at or through the limit
   printed between the decision's known frame and the level's vanish + 100 ms (who_took.follow); the takers of those legs.
3. Timing (legs with a websocket print time, how2 == 1): print - 150 ms = the latest possible send time (every taker waits the hold),
   relative to our decision and to the last Binance update before the decision; the Binance move (r250 / r100 / r50, signed in the
   direction of our take) at that send time.
4. Their fills: value after a 2 % x min(p, 1 - p) fee at +1 s and +10 s (book mid) and at settlement (ties 50 / 50), for the consuming legs
   and for all taker legs of the top takers on the 408 complete markets.
Output (write-once): takers.json.
Usage: PF_SLOT=claude predictfun/research/run_light.sh takers -- python3 predictfun/reviews/take_exit_venue_20261125/takers.py
"""
import collections
import datetime as dt
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import venue_common as V  # noqa: E402
import who_took as WT  # noqa: E402

F, OD, MM, TR, C = V.F, V.OD, V.MM, V.TR, V.C
sys.path.insert(0, str(H.parents[1] / 'research'))
import pf_io as IO  # noqa: E402

S1D = WT.S1D
HOLD = 150.
PERMITTED_NEXT = {'20261023': '20261024', '20261031': '20261101', '20261101': '20261102', '20261102': '20261103'}   # 20261103 -> none
BANNED = ('20261028', '20261029', '20261030', '20261104')
KEY = ['mid', 'exec_s', 'first_recv_ns', 'side', 'ptick', 'shares', 'price_executed']
EPS = V.EPS


def stems(day):
    out = [f'{day}_{h:02d}' for h in range(24)]
    if day in PERMITTED_NEXT:
        out += [f'{PERMITTED_NEXT[day]}_00', f'{PERMITTED_NEXT[day]}_01']
    C.require(not any(s[:8] in BANNED or s[:8] >= '20261104' for s in out), 'a stem outside the permitted hours')
    return out


def taker_table(day):
    """REST match legs of one day with the taker's signer and order hash (pf_step1's key and earliest receipt)."""
    legs = {}
    for h in stems(day):
        if not Path(f'{IO.PF_RUN}/rest/{h}.jsonl.gz').exists():
            continue
        for recv, params, path, pl in IO.rest_rows(h, '/v1/orders/matches'):
            for x in pl.get('data') or []:
                try:
                    ts = int(IO.iso_s(x['executedAt']))
                except (KeyError, TypeError, ValueError):
                    continue
                tx, t = x['transactionHash'][2:18], x['taker']
                for m in x['makers']:
                    k = tx + m['hash'][2:18]
                    old = legs.get(k)
                    if old is not None:
                        if recv < old[2]:
                            legs[k] = old[:2] + (recv,) + old[3:]
                        continue
                    side, py = IO.yes_side(m)
                    legs[k] = ((x.get('market') or {}).get('id'), ts, recv, 0 if side == 'bid' else 1, int(round(py * 100)), int(m['amount']) / IO.E18,
                               int(x['priceExecuted']) / IO.E18, t['signer'], t['hash'])
    return pd.DataFrame(list(legs.values()), columns=KEY + ['taker', 'taker_hash'])


def bn_at(k, m, t, back):
    """Binance mid change (bps) over (t - back, t], from updates with event time <= t."""
    j0, j1 = np.searchsorted(k, t, 'right') - 1, np.searchsorted(k, t - back, 'right') - 1
    return float(1e4 * np.log(m[j0] / m[j1])) if j0 >= 0 and j1 >= 0 else np.nan


def mid_at(f, t):
    j = int(np.searchsorted(f[:, 0], t, 'right') - 1)
    if j < 0:
        return None
    b, a = f[j, 2], f[j, 12]
    return float((b + a) / 2) if 0 <= b < a <= 100 else None


def main():
    fn = H / 'takers.json'
    C.require(not fn.exists(), 'takers.json exists (write-once)')
    sm = json.loads((S1D / 'summary.json').read_text())
    C.require(sm['delay_ms'] == 172., 'not the 172 ms run')
    pay = {(r['day'], int(r['mid'])): r['y'] for r in sm['rows']}
    want = collections.defaultdict(set)
    for (d, m) in pay:
        want[d].add(m)
    src = MM.KP.Tape()
    cons, alltake, join = [], [], collections.Counter()
    for day in OD.DAYS3:
        L = pd.read_parquet(MM.S1 / day / 'legs_timed2.parquet', columns=KEY + ['t_ms2', 'how2'])
        T = taker_table(day)
        dup = T.duplicated(KEY, keep=False)
        amb = T[dup].groupby(KEY).taker.nunique()
        join['ambiguous_keys'] += int((amb > 1).sum())
        T = T.drop_duplicates(KEY)
        J = L.merge(T, on=KEY, how='left')
        C.require(len(J) == len(L), 'the join duplicated legs')
        J = J[J.mid.isin(want[day])]
        join['legs'] += len(J); join['joined'] += int(J.taker.notna().sum())
        bk, bm = OD.seg_arrays(OD.SEG_OF[day])
        reg = MM.registry(day).set_index('mid'); off = src.offset(day)
        by_mid = {m: g for m, g in J.groupby('mid')}
        for h, g in reg.groupby('hour'):
            mids = [int(m) for m in g.index if int(m) in want[day]]
            if not mids:
                continue
            for m, st, p, info, yup in F.worlds_yup(day, mids, src, off, reg):
                C.require(st == 'complete', f'{m} {st}')
                t0 = float(info['t_start']); f = np.asarray(p['frames'], float); y = pay[(day, m)]
                lg = by_mid.get(m)
                if lg is None:
                    continue
                led = json.loads(gzip.decompress((S1D / f'{m}_BASE.json.gz').read_bytes()))
                pt = lg.t_ms2.values.astype(float) - t0                     # print time relative to the market start (venue clock)
                ws = lg.how2.values == 1
                # every taker leg of the market (for the takers' overall fill value)
                for i in range(len(lg)):
                    r = lg.iloc[i]; sg = 1 if r.side == 1 else -1; pr = float(r.ptick); fee = .02 * min(pr, 100. - pr)
                    v1 = mid_at(f, pt[i] + 1000.) if np.isfinite(pt[i]) else None; v10 = mid_at(f, pt[i] + 10000.) if np.isfinite(pt[i]) else None
                    alltake.append((r.taker, float(r.shares), None if v1 is None else sg * (v1 - pr) - fee, None if v10 is None else sg * (v10 - pr) - fee,
                                    sg * (100. * y - pr) - fee))
                for x in led['ioc_results']:
                    if x['reason'] != 'limit_moved':
                        continue
                    fo = WT.follow(p, x)
                    if fo['gone_ms'] is None:
                        continue
                    s, Lp, g = int(x['side']), float(x['limit']), float(x['submit'])
                    opp = 1 - s; jk = int(np.searchsorted(f[:, 1], g, side='right') - 1)
                    lo, hi = f[jk, 0], g + fo['gone_ms'] + 100.
                    sel = (lg.side.values == opp) & ((lg.ptick.values >= Lp) if opp == 1 else (lg.ptick.values <= Lp)) & (pt >= lo) & (pt <= hi)
                    gabs = t0 + g
                    jb = np.searchsorted(bk, gabs - V.MM.B_IN, 'right') - 1            # the last Binance update our decision used
                    e_last = float(bk[jb]) if jb >= 0 else np.nan
                    d_sign = 1. if s == 0 else -1.                                    # our take's direction: buy YES on Binance up
                    for i in np.flatnonzero(sel):
                        r = lg.iloc[i]
                        row = dict(day=day, mid=m, taker=r.taker, shares=float(r.shares), ws_timed=bool(ws[i]), print_rel_decision_ms=float(pt[i] - g))
                        if ws[i]:
                            send = t0 + pt[i] - HOLD
                            row.update(send_rel_decision_ms=float(send - gabs), send_rel_last_bn_ms=float(send - e_last),
                                       r250=d_sign * bn_at(bk, bm, send, 250.), r100=d_sign * bn_at(bk, bm, send, 100.), r50=d_sign * bn_at(bk, bm, send, 50.),
                                       r250_at_decision=d_sign * bn_at(bk, bm, gabs - V.MM.B_IN, 250.))
                        sg = 1 if r.side == 1 else -1; pr = float(r.ptick); fee = .02 * min(pr, 100. - pr)
                        v1 = mid_at(f, pt[i] + 1000.); v10 = mid_at(f, pt[i] + 10000.)
                        row.update(v1=None if v1 is None else sg * (v1 - pr) - fee, v10=None if v10 is None else sg * (v10 - pr) - fee, vset=sg * (100. * y - pr) - fee)
                        cons.append(row)
        print(day, 'legs', join['legs'], 'joined', join['joined'], 'consuming', len(cons), flush=True)
    C.require(join['joined'] / join['legs'] > .99, f"taker join rate {join['joined'] / join['legs']:.3f}")
    # identity
    D = pd.DataFrame(cons)
    vol = D.groupby('taker').shares.sum().sort_values(ascending=False)
    top = list(vol.index[:10])
    A = pd.DataFrame(alltake, columns=['taker', 'shares', 'v1', 'v10', 'vset'])
    wavg = lambda d, c: float((d[c] * d.shares).sum() / d.shares[d[c].notna()].sum()) if d[c].notna().any() else None
    q = lambda a, p: float(np.nanquantile(a, p)) if np.isfinite(a).any() else None
    W = D[D.ws_timed]
    res = dict(
        asof='2026-11-25', source='stage-1 172 ms BASE ledgers (limit_moved takes), 408 complete markets; REST match takers',
        join=dict(join, rate=join['joined'] / join['legs']), consuming_legs=len(D), consuming_shares=float(D.shares.sum()), distinct_takers=int(vol.size),
        top10_share=float(vol.iloc[:10].sum() / vol.sum()), top3_share=float(vol.iloc[:3].sum() / vol.sum()),
        top_takers=[dict(rank=i + 1, taker=f'{t[:6]}…{t[-4:]}', consumed_shares=float(vol[t]), share=float(vol[t] / vol.sum()),
                         all_taker_shares_5d=float(A[A.taker == t].shares.sum()),
                         value_after_fee_c_per_share=dict(consuming=dict(s1=wavg(D[D.taker == t], 'v1'), s10=wavg(D[D.taker == t], 'v10'), settle=wavg(D[D.taker == t], 'vset')),
                                                          all_fills=dict(s1=wavg(A[A.taker == t], 'v1'), s10=wavg(A[A.taker == t], 'v10'), settle=wavg(A[A.taker == t], 'vset'))))
                    for i, t in enumerate(top)],
        consuming_value_after_fee_c_per_share=dict(s1=wavg(D, 'v1'), s10=wavg(D, 'v10'), settle=wavg(D, 'vset')),
        all_takers_value_after_fee_c_per_share=dict(s1=wavg(A, 'v1'), s10=wavg(A, 'v10'), settle=wavg(A, 'vset')),
        timing_ws_timed=dict(
            legs=len(W), shares=float(W.shares.sum()),
            print_rel_decision_ms=[q(W.print_rel_decision_ms.values, p) for p in (.1, .25, .5, .75, .9)],
            send_rel_decision_ms=[q(W.send_rel_decision_ms.values, p) for p in (.1, .25, .5, .75, .9)],
            send_rel_last_binance_update_ms=[q(W.send_rel_last_bn_ms.values, p) for p in (.1, .25, .5, .75, .9)],
            binance_move_at_their_send_bps=dict(r250=[q(W.r250.values, p) for p in (.1, .25, .5, .75, .9)], r100=[q(W.r100.values, p) for p in (.1, .25, .5, .75, .9)],
                                               r50=[q(W.r50.values, p) for p in (.1, .25, .5, .75, .9)]),
            binance_r250_at_our_decision_bps=[q(W.r250_at_decision.values, p) for p in (.1, .25, .5, .75, .9)],
            sent_before_our_decision_share=float((W.send_rel_decision_ms < 0).mean()),
            sent_before_the_last_binance_update_share=float((W.send_rel_last_bn_ms < 0).mean())),
        quantiles=[.1, .25, .5, .75, .9])
    TR.dump(fn, res)
    print(json.dumps({k: v for k, v in res.items() if k not in ('top_takers',)}, indent=1))
    for t in res['top_takers']:
        print(t)


if __name__ == '__main__':
    main()
