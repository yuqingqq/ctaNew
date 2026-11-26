"""Export of the smart takers' fires for a cross-venue check, with worked examples (USER 2026-11-26: "Can you show me some examples of what
happened on Binance Orderbook/trades, and then what's the case when he fires, like what's the probability, remaining time, drift, etc. ...
can you also generate a Md, list the timestamps of the fires, and to check what kind of metrics in other exchanges books, I will give it to
another guy with these data to check the factors"). Readout on the four consumed days; no strategy change.
Rows: every fire of the top taker (fire_breakdown_points, its 5,930 bursts), its control moments (the same times shifted -20 / -10 / +10 /
+20 s, same direction), and every burst of the takers ranked 2-10 (fire_times; first fire per burst). Times: arrival at Predict = settlement
- 150 - 34 ms (Predict's clock, estimate +-15 ms); fire = settlement - 150 ms; Binance times on Binance futures' E clock.
Context per row: the market (window start, time left), Binance futures best bid / ask and sizes at t (bookTicker states), imbalance its way,
depth imbalance (20 levels, depth_deep where computed), the mid drift since the window opened and over the last 30 s (bps its way), the
realized volatility over the previous 300 s and the Binance-implied fair UP probability = N(drift / (vol sqrt(time left))) (winners_trigger's
convention, the Binance mid at the window open as the strike: Predict settles on its own reference, not Binance), the market's payout; for
fires: its average fill price (UP space), shares, 1 s value, the breakdown row, the number of SMART_COMBO's eight factors present, the next
Binance move its way.
Examples (declared before the run): one top-taker fire per breakdown group (no activity, far side shrinking, other changes, after a move),
drawn with seed 20261126 + group index; the Binance best levels and trades over [t - 3 s, t + 0.5 s].
Outputs (write-once): OUT/smart_fires_export/{top_taker_fires,top_taker_controls,ranks2_10_fires}.csv.gz; smart_fires/examples.json and
smart_fires/top_taker_fires_sample.csv (the first 50 fires) in the review folder.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/smart_fires_export.py [<rehearsal dir> <day> ...]
"""
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import fire_events as FE  # noqa: E402

DC, C, TR, WT = FE.DC, FE.C, FE.TR, FE.WT
SHIFTS_C, TOP = (-20000., -10000., 10000., 20000.), '0x7cab1c46a5B0557008198b92697F36df855ac0AD'
EX_GROUPS = (('no activity', ('flat_unchanged', 'flat_small_changes')), ('far side shrinking', ('flat_far_shrank',)),
             ('other changes', ('flat_near_grew', 'flat_other_activity', 'moved_against_only')), ('after a move started', ('moved_its_way',)))
SEED = 20261126


def utc(t):
    return pd.to_datetime(np.asarray(t, float), unit='ms', utc=True).strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]


def context(day, P):
    """Binance context for points P (t, d, ws) of one day."""
    t, d, ws = P.t.to_numpy(float), P.d.to_numpy(float), P.ws.to_numpy(float)
    n = len(t)
    o = {k: np.full(n, np.nan) for k in ('bn_bid', 'bn_ask', 'bn_bid_size', 'bn_ask_size', 'imbalance_its_way', 'far_size_btc', 'near_size_btc')}
    day0 = float(WT.OD_day_ms(day))
    hp = ((t - day0) // 3600000).astype(int)
    for h, bk in DC.book_hours(day):
        sel = np.flatnonzero(hp == h)
        if not len(sel):
            continue
        E, bid, bq, ask, aq = FE.collapse(*bk)
        j = WT.last_idx(E, t[sel]); ok = j >= 0; jj = np.maximum(j, 0)
        up = d[sel] > 0
        far, near = np.where(up, aq[jj], bq[jj]), np.where(up, bq[jj], aq[jj])
        for k, v in (('bn_bid', bid[jj]), ('bn_ask', ask[jj]), ('bn_bid_size', bq[jj]), ('bn_ask_size', aq[jj]), ('far_size_btc', far), ('near_size_btc', near),
                     ('imbalance_its_way', (near - far) / (near + far))):
            o[k][sel] = np.where(ok, v, np.nan)
    bk_, bm = WT.OD.seg_arrays(WT.OD.SEG_OF[day]); bk_ = bk_.astype(float); bm = bm.astype(float)
    cs = np.cumsum(np.r_[0., np.diff(np.log(bm))] ** 2)
    j = WT.last_idx(bk_, t); jj = np.maximum(j, 0)
    jw, j30 = np.maximum(WT.last_idx(bk_, ws), 0), np.maximum(WT.last_idx(bk_, t - 30000.), 0)
    js = np.maximum(WT.last_idx(bk_, t - 300000.), 0)
    x = 1e4 * np.log(bm[jj] / bm[jw])
    sig = 1e4 * np.sqrt(np.maximum(cs[jj] - cs[js], 0.) / 300.)
    tau = (ws + 300000. - t) / 1000.
    inside = (tau > 0) & (tau <= 300.) & (j >= 0)
    o['time_left_s'] = np.where(inside, tau, np.nan)
    o['bn_mid_at_open'] = np.where(inside, bm[jw], np.nan)
    o['drift_since_open_bps_its_way'] = np.where(inside, d * x, np.nan)
    o['drift_last_30s_bps_its_way'] = np.where(j >= 0, d * 1e4 * np.log(bm[jj] / bm[j30]), np.nan)
    o['vol_300s_bps_per_sqrt_s'] = np.where(j >= 0, sig, np.nan)
    o['fair_up_prob'] = np.where(inside & (sig > 0), ndtr(x / np.where(sig > 0, sig * np.sqrt(np.maximum(tau, .5)), 1.)), np.nan)
    return o


def tape(day, t, d):
    """Binance best levels (changes) and trades over [t - 3 s, t + 0.5 s] for one example."""
    day0 = float(WT.OD_day_ms(day)); h = int((t - day0) // 3600000)
    for hh, bk in DC.book_hours(day):
        if hh != h:
            continue
        E, bid, bq, ask, aq = FE.collapse(*bk)
        j0 = WT.last_idx(E, np.array([t - 3000.]))[0]
        k = np.r_[j0, np.flatnonzero((E > t - 3000.) & (E <= t + 500.))]
        book = [[float(E[i] - t), float(bid[i]), float(bq[i]), float(ask[i]), float(aq[i])] for i in k if i >= 0]
        tr = DC.read_trades(day).drop_duplicates('id')
        tr = tr[(tr.E > t - 3000.) & (tr.E <= t + 500.) & (tr.p > 0) & (tr.q > 0)]
        g = tr.groupby(['E', 'm']).agg(q=('q', 'sum'), n=('q', 'size'), pmin=('p', 'min'), pmax=('p', 'max')).reset_index()
        trades = [[float(r.E - t), 'buy' if int(r.m) == DC.BUY_AGGR else 'sell', float(r.q), int(r.n), float(r.pmin), float(r.pmax)] for r in g.itertuples()]
        return dict(book=book, trades=trades)
    return None


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (1500, 1501))
    od = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    out_dir = (od or WT.OUT) / 'smart_fires_export'
    rv = (od or H) / 'smart_fires'
    C.require(not out_dir.exists() and not (rv / 'examples.json').exists(), 'an output exists (write-once)')
    days = tuple(sys.argv[2:]) if len(sys.argv) > 2 else WT.DAYS
    C.require(all(x in WT.DAYS for x in days), 'not a declared day')
    c = float(json.loads((H / 'fire_events.json').read_text())['groups']['TOP1']['arrival_offset_ms'])
    A = pd.read_parquet(WT.OUT / 'fire_times_orders.parquet')
    A = A[A.t_set.notna()].copy()
    A['taker_short'] = A.taker.str[:6] + '…' + A.taker.str[-4:]
    B = A.sort_values(['burst', 'fire'], kind='stable').drop_duplicates('burst', keep='first').reset_index(drop=True)
    B['a'], B['d'] = B.fire - c, B.d.astype(float)
    reg = pd.concat([WT.MM.registry(dd).assign(day=dd) for dd in WT.DAYS])
    reg['ws_ms'] = reg.ws.astype(float) * 1000.
    vs = json.loads((WT.VR.OUT / 'summary.json').read_text())
    pay = {int(r['mid']): r['payout'] for r in vs['rows'] if r['day'] in WT.DAYS}
    # the top taker's fires, numbered as fire_breakdown (state_value's send order)
    S = pd.read_parquet(WT.OUT / 'state_value_points.parquet', columns=['day', 'kind', 't', 'd', 'v1'])
    S = S[S.kind == 'send'].reset_index(drop=True); S['dec'] = np.arange(len(S))
    Bt = B[B['rank'] == 1]
    Fx = S.merge(Bt[['day', 'a', 'd', 'mid', 'burst', 't_set', 'fire']].rename(columns={'a': 't'}), on=['day', 't', 'd'], how='left')
    C.require(Fx.mid.notna().all() and not Fx.dec.duplicated().any(), 'a fire without its burst / a duplicated fire')
    legs = pd.concat([pd.read_parquet(WT.OUT / f'winners_legs_{dd}.parquet', columns=['taker_hash', 'ptick', 'shares', 'how2', 'ambiguous']) for dd in WT.DAYS])
    legs = legs[~legs.ambiguous & legs.how2.isin([1, 3])]
    lg = legs.assign(ps=legs.ptick * legs.shares).groupby('taker_hash').agg(ps=('ps', 'sum'), sh=('shares', 'sum'))
    ob = A[A['rank'] == 1].merge(lg, left_on='taker_hash', right_index=True, how='left').groupby('burst').agg(ps=('ps', 'sum'), sh=('sh', 'sum'), orders=('taker_hash', 'size'))
    Fx = Fx.merge(ob, left_on='burst', right_index=True, how='left')
    Fx['avg_price_up_c'] = Fx.ps / Fx.sh
    FB = pd.read_parquet(WT.OUT / 'fire_breakdown_points.parquet', columns=['dec', 'day', 't', 'd', 'shift_ms', 'row', 'next_ms'])
    DD_ = pd.read_parquet(WT.OUT / 'depth_deep_points.parquet', columns=['dec', 'shift_ms', 'imb_1_20'])
    CB = pd.read_parquet(WT.OUT / 'smart_combo_points.parquet', columns=['day', 't', 'd', 'grp', 'kind', 'n_on'])
    CB = CB[(CB.grp == 'TOP1') & (CB.kind == 'fire')].drop_duplicates(['day', 't', 'd'])
    F0 = FB[FB.shift_ms == 0.].merge(DD_[DD_.shift_ms == 0.][['dec', 'imb_1_20']], on='dec', how='left')
    Fx = Fx.merge(F0[['dec', 'row', 'next_ms', 'imb_1_20']], on='dec', how='left', validate='one_to_one').merge(CB[['day', 't', 'd', 'n_on']], on=['day', 't', 'd'], how='left')
    Fx = Fx.merge(reg[['mid', 'ws_ms']], on='mid', how='left')
    Fx = Fx[Fx.day.isin(days)].reset_index(drop=True)
    Cx = FB[FB.shift_ms.isin(SHIFTS_C)].merge(DD_[['dec', 'shift_ms', 'imb_1_20']], on=['dec', 'shift_ms'], how='left').merge(Fx[['dec', 'mid', 'ws_ms']], on='dec', how='inner')
    Rk = B.loc[B['rank'] > 1, ['day', 'a', 'd', 'rank', 'taker_short', 'mid', 'burst', 't_set', 'fire']].rename(columns={'a': 't'}).merge(reg[['mid', 'ws_ms']], on='mid', how='left')   # select first: B also holds the M50 't'
    Rk = Rk[Rk.day.isin(days)].reset_index(drop=True)
    tabs = {'fires': Fx, 'controls': Cx, 'ranks': Rk}
    for nm, T in tabs.items():
        parts = []
        for day in days:
            Td = T[T.day == day].reset_index(drop=True)
            if not len(Td):
                continue
            for k, v in context(day, Td.assign(ws=Td.ws_ms)).items():
                Td[k] = v
            parts.append(Td)
        tabs[nm] = pd.concat(parts, ignore_index=True)
        print(nm, len(tabs[nm]), flush=True)
    Fx, Cx, Rk = tabs['fires'], tabs['controls'], tabs['ranks']
    for T in (Fx, Cx, Rk):
        T['utc_time'] = utc(T.t)
        T['window_start_utc'] = utc(T.ws_ms)
        T['bet'] = np.where(T.d > 0, 'UP', 'DOWN')
        T['payout_up'] = T.mid.map(pay)
    Fx['settlement_utc'] = utc(Fx.t_set); Fx['binance_move_utc'] = np.where(np.isfinite(Fx.next_ms), utc(Fx.t + np.nan_to_num(Fx.next_ms, posinf=0.)), '')
    Fx['bet_won'] = np.where(Fx.payout_up.isin([0., 1.]), np.where(Fx.d > 0, Fx.payout_up == 1., Fx.payout_up == 0.), np.nan)
    Fx['edge_vs_fair_c'] = Fx.d * (100. * Fx.fair_up_prob - Fx.avg_price_up_c)
    Fx['next_move_ms'] = Fx.next_ms.where(np.isfinite(Fx.next_ms))
    base = ['utc_time', 't', 'bet', 'mid', 'window_start_utc', 'time_left_s', 'bn_bid', 'bn_ask', 'bn_bid_size', 'bn_ask_size', 'imbalance_its_way', 'far_size_btc',
            'near_size_btc', 'imb_1_20', 'drift_since_open_bps_its_way', 'drift_last_30s_bps_its_way', 'vol_300s_bps_per_sqrt_s', 'fair_up_prob', 'payout_up']
    fcols = ['dec'] + base[:1] + ['settlement_utc'] + base[1:] + ['avg_price_up_c', 'sh', 'orders', 'v1', 'edge_vs_fair_c', 'bet_won', 'row', 'n_on', 'next_move_ms', 'binance_move_utc']
    ren = dict(dec='fire_id', t='t_ms', mid='market_id', imb_1_20='depth_imbalance_20_its_way', sh='shares', v1='value_1s_c_per_share', row='binance_300ms_breakdown',
               n_on='factors_present_of_8')
    out_dir.mkdir(parents=True, exist_ok=True); rv.mkdir(parents=True, exist_ok=True)
    Fo = Fx[fcols].rename(columns=ren)
    Fo.to_csv(out_dir / 'top_taker_fires.csv.gz', index=False)
    Fo.head(50).to_csv(rv / 'top_taker_fires_sample.csv', index=False)
    Cx[['dec', 'shift_ms'] + base].rename(columns=dict(ren, shift_ms='shift_from_fire_ms')).to_csv(out_dir / 'top_taker_controls.csv.gz', index=False)
    Rk[['rank', 'taker_short', 'utc_time', 't', 'bet'] + base[3:]].rename(columns=ren).to_csv(out_dir / 'ranks2_10_fires.csv.gz', index=False)
    ex = []
    for gi, (gname, rows) in enumerate(EX_GROUPS):
        pool = Fx[Fx.row.isin(rows)].reset_index(drop=True)
        if not len(pool):
            continue
        r = pool.iloc[int(np.random.default_rng(SEED + gi).integers(len(pool)))]
        ex.append(dict(group=gname, pool=int(len(pool)), **{k: (None if isinstance(v, float) and not np.isfinite(v) else (v.item() if hasattr(v, 'item') else v))
                                                           for k, v in r[fcols].rename(ren).items()}, tape=tape(r.day, float(r.t), float(r.d)), day=r.day))
    TR.dump(rv / 'examples.json', dict(asof='2026-11-26', examples=ex))
    summ = dict(fires=int(len(Fo)), controls=int(len(Cx)), ranks=int(len(Rk)), fires_with_price=float(Fo.avg_price_up_c.notna().mean()),
                fair_coverage=float(Fo.fair_up_prob.notna().mean()), edge_vs_fair_c=dict(mean=float(Fo.edge_vs_fair_c.mean()), median=float(Fo.edge_vs_fair_c.median())),
                bet_won=float(np.nanmean(Fo.bet_won.astype(float))), time_left_s=dict(p25=float(Fo.time_left_s.quantile(.25)), p50=float(Fo.time_left_s.median()), p75=float(Fo.time_left_s.quantile(.75))))
    TR.dump(rv / 'export_summary.json', summ)
    print(json.dumps(summ, indent=1))
    for e in ex:
        print(e['group'], e['utc_time'], e['bet'], 'left', e['time_left_s'], 'price', e['avg_price_up_c'], 'fair', e['fair_up_prob'], 'drift', e['drift_since_open_bps_its_way'], 'row', e['binance_300ms_breakdown'], 'move', e['next_move_ms'], 'won', e['bet_won'], 'v1', e['value_1s_c_per_share'])


if __name__ == '__main__':
    main()
