"""What separates our chase model's top-10 % picks from the top taker's group-3 fires, on every piece of information we hold (USER
2026-11-27: "What's the main dif between our top 10 percentile fires against top taker fires? Check all information we have"). Consumed days;
descriptive; declared before the run.
Sets, on the days it traded >= 100 group-3 fires (both sets on the same days): OURS = CHASE_TAKES' model top-10 % actions; ITS = its group-3
fires at their decision tick (GROUP3_PROFILE's: our last tick of its move decided before its arrival; one row per tick); REF = every move at
its first tick. Rows and inputs are CHASE_TAKES' (counts reproduced: a check).
Information (stage extras, per labelled chase tick): CHASE_TAKES' 97 inputs (families: Binance best level, Binance flows against normal,
Binance depth, other venues, Predict fills, Predict book, fair value and time, the move); ETH futures (Binance trades) return its way over
300 ms / 1 s / 5 s; the top taker's own state in the market (its orders' arrivals before the tick: count, net shares its way, ms since its last
order; not public in real time, its fills are visible after settlement); Predict traded shares in the market over the last 30 s and since
the window opened (every taker's WS-timed legs).
Readings (stage read): (1) per input: medians (OURS, ITS, REF) and the AUC of ITS against OURS; null: the set labels shuffled across the
pooled rows (200 draws), the largest |AUC - 0.5| over the inputs, its 95th percentile; (2) a classifier (public_score's pinned trees) of ITS
against OURS, leave-one-day-out: held-out AUC; per family the held-out AUC drop when the family's columns are shuffled together (5 draws),
with and without the top taker's own state; (3) outcomes per set: Predict's move 1 s after our match from that tick, the Binance move's size
in its first 300 ms, the value of taking there on both books (known best + 0 / 3 / 10c), the share of the optimistic fill gone by the next
snapshot; for ITS its own 1 s value and pay-through; (4) the overlap: moves in both sets against those in one only.
Outputs (write-once): OUT/ours_vs_its/<day>.parquet (extras), ours_vs_its.json (read).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/ours_vs_its.py \
       selftest | extras <day> [<dir>] | read [<dir>]
"""
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import chase_takes as CT  # noqa: E402
import group3_profile as GP  # noqa: E402

C, TR, TK, TM, WT, CV, F = CT.C, CT.TR, CT.TK, CT.TM, CT.WT, CT.CV, CT.F
OUT = WT.OUT / 'ours_vs_its'
DAYS = CT.DAYS
N_NULL, N_PERM, MIN_FIRES, SEED = 200, 5, 100, 20261127
EXTRA = ['eth_ret_300_bps', 'eth_ret_1s_bps', 'eth_ret_5s_bps', 'its_orders_mkt', 'its_net_its_way_sh', 'its_last_order_ms', 'pf_volume_30s_sh', 'pf_volume_open_sh']


def family(c):
    if c.startswith('eth_'):
        return 'ETH futures'
    if c.startswith('its_orders') or c.startswith('its_net') or c.startswith('its_last'):
        return "the top taker's own state"
    if c.startswith('pf_volume'):
        return 'Predict volume'
    if c in ('p_tau', 'chase_implied_c', 'chase_gap_c') or c.startswith('fv_'):
        return 'fair value and time'
    if c.startswith('mv_') or c in ('its_100_btc', 'ag_100_btc'):
        return 'the move'
    if c.startswith('b_'):
        return 'Binance best level'
    if c.startswith('x_'):
        return 'Binance flows'
    if c.startswith(('d_', 'z_')):
        return 'Binance depth'
    if c.startswith('u_'):
        return 'other venues'
    if c in ('p_pf5', 'q_net_share5', 'q_fill_r1s'):
        return 'Predict fills'
    if c.startswith('pf_') or c in ('spread_60s', 'book_updates_60s'):
        return 'Predict book'
    return 'other'


# ---------------------------------------------------------------------------------------------------------------- extras
def ret_its_way(tv, pv, lo_cov, t, d, w):
    j1 = np.searchsorted(tv, t, 'right') - 1
    j0 = np.searchsorted(tv, t - w, 'right') - 1
    ok = (j0 >= 0) & (j1 >= 0) & (t - w >= lo_cov)
    return np.where(ok, d * 1e4 * np.log(pv[np.maximum(j1, 0)] / pv[np.maximum(j0, 0)]), np.nan)


def per_market_state(mid, t, d, om, oa, osgn, osh):
    """The orders (market om, arrival oa, direction osgn, shares osh) before each row (market mid, time t, direction d): count, net shares
    its way, ms since the last one."""
    n = len(t)
    cnt, net, last = np.zeros(n), np.zeros(n), np.full(n, np.nan)
    rows = pd.Series(np.arange(n)).groupby(mid).indices
    for m, g in pd.DataFrame(dict(m=om, a=oa, s=osgn * osh)).groupby('m'):
        r = rows.get(m)
        if r is None:
            continue
        g = g.sort_values('a', kind='stable')
        a, cs = g.a.to_numpy(float), np.r_[0., np.cumsum(g.s.to_numpy(float))]
        k = np.searchsorted(a, t[r], 'left')                              # orders that arrived before the tick
        cnt[r], net[r] = k, d[r] * cs[k]
        last[r] = np.where(k > 0, t[r] - a[np.maximum(k - 1, 0)], np.nan)
    return cnt, net, last


def extras(day, od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (600, 601))
    C.require(day in DAYS, f'{day} is not a declared day')
    od = Path(od) if od else OUT
    fp = od / f'{day}.parquet'
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    L = pd.read_parquet(CT.OUT / f'{day}_labels.parquet', columns=['day', 't', 'd', 'mid'])
    t, d, mid = L.t.to_numpy(float), L.d.to_numpy(float), L.mid.to_numpy()
    o = {}
    tv, pv, _, _, cov, _ = CV.load('eth_fut', day)
    for w, nm in ((300., '300'), (1000., '1s'), (5000., '5s')):
        o[f'eth_ret_{nm}_bps'] = ret_its_way(tv, pv, cov[0], t, d, w)
    del tv, pv
    c = float(json.loads((H / 'fire_events.json').read_text())['groups']['TOP1']['arrival_offset_ms'])
    A = pd.read_parquet(WT.OUT / 'fire_times_orders.parquet', columns=['day', 'rank', 't_set', 'fire', 'd', 'mid', 'shares'])
    A = A[(A.day == day) & (A['rank'] == 1) & A.t_set.notna()]
    o['its_orders_mkt'], o['its_net_its_way_sh'], o['its_last_order_ms'] = per_market_state(mid, t, d, A.mid.to_numpy(), (A.fire - c).to_numpy(float),
                                                                                           A.d.to_numpy(float), A.shares.to_numpy(float))
    Lg = pd.read_parquet(WT.OUT / f'winners_legs_{day}.parquet', columns=['mid', 'shares', 't_ms2'])
    Lg = Lg[Lg.t_ms2.notna()]
    v30, vop = np.zeros(len(t)), np.zeros(len(t))
    rows = pd.Series(np.arange(len(t))).groupby(mid).indices
    ws = np.floor(t / 300000.) * 300000.
    for m, g in Lg.groupby('mid'):
        r = rows.get(m)
        if r is None:
            continue
        g = g.sort_values('t_ms2', kind='stable')
        tt, cs = g.t_ms2.to_numpy(float), np.r_[0., np.cumsum(g.shares.to_numpy(float))]
        hi = np.searchsorted(tt, t[r], 'right')
        v30[r] = cs[hi] - cs[np.searchsorted(tt, t[r] - 30000., 'right')]
        vop[r] = cs[hi] - cs[np.searchsorted(tt, ws[r], 'left')]
    o['pf_volume_30s_sh'], o['pf_volume_open_sh'] = v30, vop
    out = pd.concat([L[['day', 't', 'd']].reset_index(drop=True), pd.DataFrame(o)], axis=1)
    od.mkdir(parents=True, exist_ok=True)
    out.to_parquet(fp, index=False)
    print(day, len(out), {k: round(float(np.nanmedian(v)), 3) for k, v in o.items()}, 'rss', round(F.rss_mib()), flush=True)


# ---------------------------------------------------------------------------------------------------------------- read
def auc_cols(V, y):
    one, A = GP.auc_matrix(V, y)
    return one, A


def read(od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (1500, 1501))
    reh = od is not None
    od = Path(od) if od else None
    fn = (od or H) / 'ours_vs_its.json'
    C.require(not fn.exists(), 'ours_vs_its.json exists (write-once)')
    src = od or OUT
    X, feats, _ = CT.frame(CT.OUT)
    X = X.merge(pd.read_parquet(CT.OUT / 'predictions.parquet', columns=['day', 't', 'd', 'pred_1s']), on=['day', 't', 'd'], how='left', validate='1:1')
    E = pd.concat([pd.read_parquet(src / f'{dh}.parquet') for dh in DAYS if (src / f'{dh}.parquet').exists()], ignore_index=True)
    X = X.merge(E, on=['day', 't', 'd'], how='left', validate='1:1')
    d, lim, md, day = X.d.to_numpy(float), X.lim.to_numpy(float), X.md.to_numpy(float), X.day.to_numpy()
    key = X.key.to_numpy()
    e1 = X.pred_1s.to_numpy(float) - (d * (lim - md) + CT.fee(lim))
    on = CT.first_per_move(X.mv_k.to_numpy(float) == 1, key)
    t10 = CT.first_per_move(e1 >= np.quantile(e1, .9), key)
    CJ = json.loads((H / 'chase_takes.json').read_text())
    C.require(len(on) == CJ['selections']['every_move_onset']['actions'] and len(t10) == CJ['selections']['model1s_top10']['actions'], 'CHASE_TAKES\' selections not reproduced')
    # its group-3 fires at their decision tick
    G = pd.concat([pd.read_parquet(GP.OUT / f'{dh}.parquet') for dh in DAYS], ignore_index=True)
    G3 = G[(G.status == 'ok') & (G.group == 3)].copy()
    G3['key'] = G3.day.astype(str) + ':' + G3.mkey.astype(np.int64).astype(str)
    by_key = pd.Series(np.arange(len(X))).groupby(key).indices
    tt_ = X.t.to_numpy(float)
    dec = np.full(len(G3), -1, np.int64)
    for r, (k_, aa) in enumerate(zip(G3.key, G3.t.to_numpy(float))):
        ii = by_key.get(k_)
        if ii is None:
            continue
        ii = ii[tt_[ii] + TM.FEED + TM.DECIDE <= aa]
        if len(ii):
            dec[r] = ii[np.argmax(tt_[ii])]
    G3['dec'] = dec
    G3 = G3[G3.dec >= 0]
    fires_by_day = G3.day.value_counts()
    D3 = sorted(dh for dh, n_ in fires_by_day.items() if n_ >= MIN_FIRES)
    its_rows = np.unique(G3[G3.day.isin(D3)].dec.to_numpy())
    ours = t10[np.isin(day[t10], D3)]
    ref = on[np.isin(day[on], D3)]
    cols = list(feats) + EXTRA
    C.require(set(cols) <= set(X.columns), 'an input or extra missing')
    out = dict(asof='2026-11-27', rehearsal=reh, days=D3, fires_by_day={k: int(v) for k, v in fires_by_day.items()}, its_fires=int(G3.day.isin(D3).sum()),
               its_rows=int(len(its_rows)), ours_rows=int(len(ours)), ref_rows=int(len(ref)), overlap_rows=int(np.intersect1d(its_rows, ours).size),
               inputs={}, null={}, classifier={}, outcomes={}, overlap={})
    # ---- (1) per input
    rows = np.r_[its_rows, ours]
    y = np.r_[np.ones(len(its_rows)), np.zeros(len(ours))]
    V = X[cols].to_numpy(float)[rows]
    one, A = auc_cols(V, y)
    rng = np.random.default_rng(SEED)
    null = [float(np.nanmax(np.abs(one(rng.permutation(y)) - .5))) for _ in range(N_NULL if not reh else 20)]
    p95 = float(np.quantile(null, .95))
    out['null'] = dict(draws=len(null), p95=p95)
    Vall = X[cols].to_numpy(float)
    for j, c in enumerate(cols):
        med = lambda ix: float(np.nanmedian(Vall[ix, j])) if np.isfinite(Vall[ix, j]).any() else None
        out['inputs'][c] = dict(family=family(c), auc_its_vs_ours=float(A[j]) if np.isfinite(A[j]) else None, beyond_null=bool(np.isfinite(A[j]) and abs(A[j] - .5) > p95),
                                median_its=med(its_rows), median_ours=med(ours), median_ref=med(ref))
    # ---- (2) classifier its vs ours, leave-one-day-out
    dd = day[rows]
    fams = sorted({family(c) for c in cols})
    for nm, use in (('all_information', cols), ('without_its_own_state', [c for c in cols if family(c) != "the top taker's own state"])):
        Xc = pd.DataFrame(X[use].to_numpy(float)[rows], columns=use)
        p = np.full(len(rows), np.nan)
        drops = {f_: [] for f_ in fams if any(family(c) == f_ for c in use)}
        for dh in D3:
            tr, te = dd != dh, dd == dh
            if te.sum() == 0 or len(np.unique(y[tr])) < 2:
                continue
            mdl = HistGradientBoostingClassifier(random_state=0, **TM.HGB).fit(Xc[tr], y[tr])
            p[te] = mdl.predict_proba(Xc[te])[:, 1]
            base = roc_auc_score(y[te], p[te]) if len(np.unique(y[te])) == 2 else np.nan
            for f_ in drops:
                cj = [i for i, c in enumerate(use) if family(c) == f_]
                for _ in range(N_PERM if not reh else 2):
                    Xt = Xc[te].to_numpy().copy()
                    perm = rng.permutation(len(Xt))
                    Xt[:, cj] = Xt[perm][:, cj]
                    drops[f_].append(float(base - roc_auc_score(y[te], mdl.predict_proba(pd.DataFrame(Xt, columns=use))[:, 1])))
        ok = np.isfinite(p)
        out['classifier'][nm] = dict(heldout_auc=float(roc_auc_score(y[ok], p[ok])), by_day={dh: float(roc_auc_score(y[dd == dh], p[dd == dh])) for dh in D3
                                                                                           if len(np.unique(y[dd == dh])) == 2},
                                     family_auc_drop={f_: float(np.mean(v)) for f_, v in sorted(drops.items(), key=lambda kv: -np.mean(kv[1]))})
        C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    # ---- (3) outcomes
    mo1 = d * (X.m1s.to_numpy(float) - md)
    win_size = X.groupby('key').mv_bps.max()
    PA = X[[f'pA_{k}' for k in range(5)]].to_numpy(float); SA = X[[f'sA_{k}' for k in range(5)]].to_numpy(float)
    PB = X[[f'pB_{k}' for k in range(5)]].to_numpy(float); SB = X[[f'sB_{k}' for k in range(5)]].to_numpy(float)
    okB = np.isfinite(X.dtB.to_numpy(float))
    BK = {'A': (PA, SA), 'AB': TK.minbook(PA, SA, PB, SB)}
    mk = X.m1s.to_numpy(float)

    def outcome(ix):
        o = dict(rows=int(len(ix)), predict_move_1s_c=float(mo1[ix].mean()), move_size_first_300ms_bps=float(win_size.loc[key[ix]].mean()))
        for b in ('A', 'AB'):
            ii = ix[okB[ix]] if b == 'AB' else ix
            for s in (0., 3., 10.):
                fl, co, fe = TK.vwalk(BK[b][0][ii], BK[b][1][ii], d[ii], lim[ii] + s * d[ii])
                o[f'{b}_s{int(s)}_c'] = float(((d[ii] * (mk[ii] * fl - co) - fe) / CT.Q).mean())
        fa, fb = TK.vwalk(PA[ix], SA[ix], d[ix], np.where(d[ix] > 0, 99., 1.))[0], TK.vwalk(*BK['AB'], d, np.where(d > 0, 99., 1.))[0][ix]
        o['gone_by_next_snapshot'] = float(1. - fb.sum() / max(fa.sum(), 1e-9))
        return o
    out['outcomes'] = dict(its=outcome(its_rows), ours=outcome(ours), ref=outcome(ref))
    g3d = G3[G3.day.isin(D3)]
    out['outcomes']['its_own'] = dict(v1_c=float(g3d.v1.mean()), pay_through_median_c=float(g3d.pay_through_c.median()), markout_1s_c=float(g3d.markout_1s_c.mean()))
    # ---- (4) overlap: moves in both sets against one only
    ko, ki = set(key[ours]), set(key[its_rows])
    both = np.array([k_ in ki for k_ in key[ours]])
    out['overlap'] = dict(ours_on_its_moves=outcome(ours[both]) if both.any() else None, ours_only=outcome(ours[~both]),
                          its_on_our_moves=outcome(its_rows[np.array([k_ in ko for k_ in key[its_rows]])]),
                          its_only=outcome(its_rows[~np.array([k_ in ko for k_ in key[its_rows]])]))
    TR.dump(fn, out)
    rk = sorted(((c, v['auc_its_vs_ours']) for c, v in out['inputs'].items() if v['auc_its_vs_ours'] is not None), key=lambda x: -abs(x[1] - .5))
    print('null p95', round(p95, 3), 'its rows', len(its_rows), 'ours', len(ours), 'days', D3)
    print([(c, round(a, 3), out['inputs'][c]['median_its'], out['inputs'][c]['median_ours']) for c, a in rk[:25]])
    print(json.dumps(out['classifier'], indent=1)[:2500])
    print(json.dumps(out['outcomes'], indent=1)[:2000])


# ---------------------------------------------------------------------------------------------------------------- selftest
def selftest():
    t = np.array([0., 1000., 2000., 3000.]); p = np.array([100., 101., 101., 100.])
    r = ret_its_way(t, p, 0., np.array([1500., 3000., 500.]), np.array([1., -1., 1.]), 1000.)
    assert abs(r[0] - 1e4 * np.log(101. / 100.)) < 1e-9 and abs(r[1] - (-1e4 * np.log(100. / 101.))) < 1e-9 and np.isnan(r[2]), r     # 500 - 1000 < coverage
    cnt, net, last = per_market_state(np.array([7, 7, 8]), np.array([100., 300., 100.]), np.array([1., -1., 1.]),
                                      np.array([7, 7, 7]), np.array([50., 150., 300.]), np.array([1., -1., 1.]), np.array([30., 10., 5.]))
    assert cnt.tolist() == [1, 2, 0] and net.tolist() == [30., -20., 0.] and last[0] == 50. and last[1] == 150. and np.isnan(last[2]), (cnt, net, last)
    assert family('eth_ret_1s_bps') == 'ETH futures' and family('its_net_its_way_sh') == "the top taker's own state" and family('its_100_btc') == 'the move'
    assert family('b_imb1') == 'Binance best level' and family('pf_mid') == 'Predict book' and family('q_fill_r1s') == 'Predict fills' and family('fv_delta') == 'fair value and time'
    assert family('u_okx_swap_gap_d300') == 'other venues' and family('z_imb_d1s') == 'Binance depth' and family('x_far_r300') == 'Binance flows'
    try:
        extras('20261022', '/nonexistent')
        raise AssertionError('an undeclared day accepted')
    except ValueError as e_:
        assert 'not a declared day' in str(e_), e_
    print('ours_vs_its selftest PASS')


if __name__ == '__main__':
    cmd, a_ = sys.argv[1], sys.argv[2:]
    if cmd == 'extras':
        extras(a_[0], a_[1] if len(a_) > 1 else None)
    elif cmd == 'read':
        read(a_[0] if a_ else None)
    else:
        dict(selftest=selftest)[cmd]()
