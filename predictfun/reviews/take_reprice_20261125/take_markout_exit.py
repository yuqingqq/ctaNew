"""Are TAKE_MARKOUT's 5 s / 30 s markouts collectable? (follow-up to USER 2026-11-27: "1. We can get slippage to fill the taker order,
2. The training target is the mark out price"). TAKE_MARKOUT valued each take at the Predict mid 1 / 5 / 30 s after the match; its 5 s
and 30 s selections came out positive held-out, buying sides priced near 0 late in the window, where a wide book's mid sits nearer 50
than any price one can trade at. Declared before the run; no refit (TAKE_MARKOUT's predictions and selections, reproduced: a check).
Labels (stage labels), the same rows at T = 172 ms: for h = 1 s / 5 s / 30 s the best bid and ask at the match + h (snapshot <= 1 s old,
inside the path) and the exit side's 5 shown levels at the exit's own match, the match + h + 172 ms (a 50-share IOC decided at match + h,
every shown level, the same speed bump; snapshot at or before, TAKE_MODEL's rule); the market's settlement payout (score_replay's rows).
Values per intended share (misses 0), per selection and entry order of TAKE_MARKOUT (book A): (a) the mid after h (TAKE_MARKOUT's; equal
there: a check); (b) take back after h: the filled shares sold (or bought back) at the exit match across the shown levels, fee per level,
any part the levels cannot take held to settlement; (c) hold to settlement: 100 x payout. Also the spread after h (median, p90, share
> 5c) against the decision's, and over all moments the three values by the price of the side bought.
Paying (computed): value per intended share > 0 pooled AND on >= 3 of 4 days AND >= 150 orders; settlement values are binary per market:
noisy (G = 4 days, point estimates).
Outputs (write-once): OUT/take_markout_exit/<day>_labels.parquet + .json (labels), take_markout_exit.json (read).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/take_markout_exit.py \
       selftest | labels <day> [<dir> <n markets>] | read [<dir>]
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
import take_markout as TK  # noqa: E402
import take_model_insample as TI  # noqa: E402

TM, VR, MM, TR, C, F, WT, SRy = TK.TM, TK.VR, TK.MM, TK.TR, TK.C, TK.F, TK.WT, TK.SRy
OUT = WT.OUT / 'take_markout_exit'
DAYS, Q, fee = TK.DAYS, TK.Q, TK.fee
NET, EXIT_DELAY = 22., 172.
HZ = (1000., 5000., 30000.)
ENTRIES = ('s0', 's3', 'any', 'adaptive_m0')
SELS = ('model_top1', 'model_e_positive', 'oracle_top1')
PRICE_BINS = ((0, 5), (5, 10), (10, 20), (20, 35), (35, 65), (65, 80), (80, 90), (90, 95), (95, 100.01))


def labels(day, od=None, limit=None):
    resource.setrlimit(resource.RLIMIT_CPU, (1500, 1501))
    C.require(day in DAYS, f'{day} is not a declared day')
    od = Path(od) if od else OUT
    fp = od / f'{day}_labels.parquet'
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    K0 = pd.read_parquet(TK.OUT / f'{day}_labels.parquet').set_index('row')
    Z = pd.read_parquet(WT.OUT / 'public_score3_points.parquet', columns=['day', 't', 'd', 'kind'])
    Z['row'] = np.arange(len(Z))
    Z = Z[Z.row.isin(set(K0.index))].reset_index(drop=True)
    C.require(len(Z) == len(K0), 'the population differs from TAKE_MARKOUT')
    reg = MM.registry(day).set_index('mid')
    ws_to_mid = {int(float(reg.loc[m, 'ws'])): int(m) for m in reg.index}
    Z['mid'] = (np.floor(Z.t / 300000.) * 300).astype(int).map(ws_to_mid)
    C.require((Z.mid.to_numpy() == K0.loc[Z.row, 'mid'].to_numpy()).all(), 'markets differ from TAKE_MARKOUT')
    pay = {int(r['mid']): float(r['payout']) for r in json.loads((SRy.OUT / f'score_replay_{day}.json').read_text())['rows']}
    by_mid = Z.groupby('mid').indices
    rows, status, parity = [], collections.Counter(), collections.Counter()
    src = MM.KP.Tape(); off = src.offset(day); n_m = 0
    for h, g in reg.groupby('hour'):
        mids = [int(m) for m in g.index if int(m) in by_mid]
        if limit is not None:
            mids = mids[:max(0, limit - n_m)]
        if not mids:
            continue
        for m, W, lst in VR.worlds_all(day, mids, src, off, reg):
            C.require(lst == 'loaded', f'{m} {lst}')
            ws = float(reg.loc[m, 'ws']) * 1000.
            p, st, info = VR.vm_path(W, ws, 50.)
            C.require(st == 'complete', f'{m} {st}')
            n_m += 1
            t0, end = float(info['t_start']), float(info['end_rel'])
            f = np.asarray(p['frames'], float)
            for i in by_mid[m]:
                t, d, rw = float(Z.t.iat[i]), float(Z.d.iat[i]), int(Z.row.iat[i])
                k0 = K0.loc[rw]
                act = t + TM.FEED + TM.DECIDE - t0 + NET + TM.HOLD
                r = dict(row=rw, day=day, mid=m, d=d, payout=pay.get(m, np.nan))
                if m not in pay:
                    status['no_payout'] += 1
                for hz in HZ:
                    hs = f'{int(hz / 1000)}s'
                    x = act + hz
                    ja = int(np.searchsorted(f[:, 0], x, 'right') - 1)
                    ok = x <= end and ja >= 0 and x - f[ja, 0] <= 1000.
                    b_, a_ = (f[ja, 2], f[ja, 12]) if ok else (-1., -1.)
                    r[f'bid_{hs}'], r[f'ask_{hs}'] = b_, a_
                    mk = TM.mid_of(f[ja]) if ok else np.nan
                    parity['checked'] += 1
                    parity['mid_differs'] += int(not ((np.isnan(mk) and np.isnan(k0[f'm{hs}_172'])) or mk == k0[f'm{hs}_172']))
                    xe = x + EXIT_DELAY
                    je = int(np.searchsorted(f[:, 0], xe, 'right') - 1)
                    if xe <= end and je >= 0:
                        pe, se = TK.side_levels(f[je], 0 if d > 0 else 1)      # a long YES exits to the bids, a short to the asks
                    else:
                        pe, se = np.full(5, -1.), np.zeros(5)
                        status[f'no_exit_book_{hs}'] += 1
                    for k in range(5):
                        r[f'eP{hs}_{k}'], r[f'eS{hs}_{k}'] = pe[k], se[k]
                status['ok'] += 1
                rows.append(r)
            C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    L = pd.DataFrame(rows)
    if limit is None:
        C.require(len(L) == len(K0), 'rows lost against TAKE_MARKOUT')
    C.require(parity['mid_differs'] == 0, f'the mid after h differs from TAKE_MARKOUT: {dict(parity)}')
    od.mkdir(parents=True, exist_ok=True)
    L.to_parquet(fp, index=False)
    TR.dump(od / f'{day}_labels.json', dict(asof='2026-11-27', day=day, markets=n_m, status=dict(status), parity=dict(parity)))
    print(day, n_m, dict(status), dict(parity), flush=True)


def walk_q(P, S, d, lim, q):
    """TAKE_MARKOUT's vwalk with a quantity per row."""
    P, S = np.asarray(P, float), np.asarray(S, float)
    dd, ll, qq = np.asarray(d, float)[:, None], np.asarray(lim, float)[:, None], np.asarray(q, float)[:, None]
    ok = (P >= 0) & (S > 0) & np.where(dd > 0, P <= ll + 1e-9, P >= ll - 1e-9)
    ok = np.logical_and.accumulate(ok, axis=1)
    avail = np.where(ok, S, 0.)
    take = np.clip(np.minimum(avail, qq - (np.cumsum(avail, axis=1) - avail)), 0., None)
    Pz = np.where(ok, P, 0.)
    return take.sum(1), (take * Pz).sum(1), (take * fee(Pz)).sum(1)


def exit_values(fl, co, fe, d, eP, eS, pay):
    """Per row in YES space, d = +1 long YES / -1 short: (take back, hold to settlement) in c; the shares the exit levels cannot take settle."""
    xq, xc, xf = walk_q(eP, eS, -d, np.where(d > 0, 1., 99.), fl)
    settle = 100. * pay
    back = d * (xc + (fl - xq) * settle - co) - fe - xf
    hold = d * (fl * settle - co) - fe
    return back, hold, xq


def summary(v, idx, day):
    out = dict(orders=int(len(idx)), value_per_intended_c=float(v[idx].mean() / Q) if len(idx) else None, by_day={})
    for dh in DAYS:
        m = day[idx] == dh
        out['by_day'][dh] = float(v[idx][m].mean() / Q) if m.any() else None
    out['days_positive'] = int(sum(1 for x in out['by_day'].values() if x is not None and x > 0))
    out['paying'] = bool(out['value_per_intended_c'] is not None and out['value_per_intended_c'] > 0 and out['days_positive'] >= 3 and len(idx) >= TK.MIN_ORDERS)
    return out


def read(od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (1200, 1201))
    reh = od is not None
    od = Path(od) if od else None
    fn = (od or H) / 'take_markout_exit.json'
    C.require(not fn.exists(), 'take_markout_exit.json exists (write-once)')
    X, feats = TI.frame()
    K = pd.concat([pd.read_parquet(TK.OUT / f'{d}_labels.parquet') for d in DAYS], ignore_index=True)          # TAKE_MARKOUT's (also in a rehearsal)
    E = pd.concat([pd.read_parquet((od or OUT) / f'{d}_labels.parquet') for d in DAYS if ((od or OUT) / f'{d}_labels.parquet').exists()], ignore_index=True)
    Pr = pd.read_parquet(TK.OUT / 'predictions.parquet')
    C.require(K.row.is_unique and E.row.is_unique and set(E.row) <= set(K.row), 'labels do not align')
    if not reh:
        C.require(len(E) == len(K) == len(X), 'the exit labels do not cover the population')
    X = X[X.row.isin(set(E.row))].reset_index(drop=True)
    X = X.merge(K.drop(columns=['day', 'mid', 't', 'd']), on='row', how='left', validate='1:1')
    X = X.merge(E.drop(columns=['day', 'mid', 'd']), on='row', how='left', validate='1:1').merge(Pr.drop(columns=['day']), on='row', how='left', validate='1:1')
    C.require(X.payout.notna().all(), 'a market without its payout')
    d, lim, md = X.d.to_numpy(float), X.lim.to_numpy(float), X.md.to_numpy(float)
    day, pay = X.day.to_numpy(), X.payout.to_numpy(float)
    cost_known = d * (lim - md) + fee(lim)
    PA = X[[f'pA172_{k}' for k in range(5)]].to_numpy(float); SA = X[[f'sA172_{k}' for k in range(5)]].to_numpy(float)
    ref = json.loads((H / 'take_markout.json').read_text())['targets'] if not reh else None
    out = dict(asof='2026-11-27', rows=int(len(X)), payout_half=int((pay == .5).sum()), label_status={}, targets={}, by_price={})
    for dh in DAYS:
        fj = (od or OUT) / f'{dh}_labels.json'
        if fj.exists():
            out['label_status'][dh] = json.loads(fj.read_text())['status']
    for hz in HZ:
        hs = f'{int(hz / 1000)}s'
        tg = f'172_{hs}'
        mk = X[f'm{hs}_172'].to_numpy(float)
        ok = np.isfinite(mk)
        mo = np.where(ok, d * (mk - md), np.nan)
        pl = X[f'pred_{tg}'].to_numpy(float)
        bid, ask = X[f'bid_{hs}'].to_numpy(float), X[f'ask_{hs}'].to_numpy(float)
        spr = np.where(ok, ask - bid, np.nan)
        eP = X[[f'eP{hs}_{k}' for k in range(5)]].to_numpy(float); eS = X[[f'eS{hs}_{k}' for k in range(5)]].to_numpy(float)
        sels = dict(model_top1=TK.top(pl - cost_known, ok, .01), model_e_positive=np.flatnonzero(ok & (pl - cost_known > 0)),
                    oracle_top1=TK.top(mo - cost_known, ok, .01))
        o = dict(spread_decision_c=dict(median=float(np.median(X.pf_spread[ok])), p90=float(np.quantile(X.pf_spread[ok], .9))),
                 spread_after_all=dict(median=float(np.nanmedian(spr)), p90=float(np.nanquantile(spr, .9)), share_over_5c=float(np.nanmean(spr > 5))), selections={})
        for nm, idx in sels.items():
            ph = pl if nm.startswith('model') else np.where(ok, mo, np.nan)
            s = dict(n=int(len(idx)), spread_after=dict(median=float(np.median(spr[idx])), p90=float(np.quantile(spr[idx], .9)), share_over_5c=float(np.mean(spr[idx] > 5))),
                     time_left_s=[float(x) for x in np.quantile(X.p_tau.to_numpy()[idx], [.1, .5, .9])],
                     its_price_c=[float(x) for x in np.quantile(X.pf_its_price.to_numpy()[idx], [.1, .5, .9])], entries={})
            for en in ENTRIES:
                if en.startswith('adaptive'):
                    L_ = TK.adaptive_limit(md, d, ph, 0.)
                    fire = np.where(d > 0, L_ >= lim, L_ <= lim)
                else:
                    L_ = np.where(d > 0, 99., 1.) if en == 'any' else lim + d * float(en[1:])
                    fire = np.ones(len(X), bool)
                ii = idx[fire[idx]]
                fl, co, fe = (np.zeros(len(X)) for _ in range(3))
                w = TK.vwalk(PA[ii], SA[ii], d[ii], L_[ii])
                fl[ii], co[ii], fe[ii] = w
                vmid = d * (mk * fl - co) - fe
                back, hold, xq = exit_values(fl, co, fe, d, eP, eS, pay)
                e = dict(mid=summary(vmid, ii, day), take_back=summary(back, ii, day), hold_to_settlement=summary(hold, ii, day),
                         exit_filled_share=float(xq[ii].sum() / fl[ii].sum()) if fl[ii].sum() > 0 else None, fill=float(fl[ii].sum() / (Q * len(ii))) if len(ii) else None)
                if ref is not None:
                    rv = ref[tg]['selections'][nm]['orders']['A'][en]
                    C.require(rv['orders'] == len(ii) and abs(rv['value_per_intended_c'] - e['mid']['value_per_intended_c']) < 1e-9,
                              f'TAKE_MARKOUT not reproduced: {tg} {nm} {en}')
                s['entries'][en] = e
            o['selections'][nm] = s
        out['targets'][tg] = o
        print(tg, {nm: {en: (round(e['mid']['value_per_intended_c'], 2), round(e['take_back']['value_per_intended_c'], 2), round(e['hold_to_settlement']['value_per_intended_c'], 2))
                        for en, e in s['entries'].items()} for nm, s in o['selections'].items()}, flush=True)
    # all moments by the price of the side bought: an order at the known best (s0) under each exit
    ip = X.pf_its_price.to_numpy(float)
    fl, co, fe = TK.vwalk(PA, SA, d, lim)
    for hz in HZ:
        hs = f'{int(hz / 1000)}s'
        mk = X[f'm{hs}_172'].to_numpy(float); ok = np.isfinite(mk)
        eP = X[[f'eP{hs}_{k}' for k in range(5)]].to_numpy(float); eS = X[[f'eS{hs}_{k}' for k in range(5)]].to_numpy(float)
        back, hold, _ = exit_values(fl, co, fe, d, eP, eS, pay)
        vmid = d * (mk * fl - co) - fe
        spr = X[f'ask_{hs}'].to_numpy(float) - X[f'bid_{hs}'].to_numpy(float)
        out['by_price'][hs] = {}
        for lo, hi in PRICE_BINS:
            m = ok & (ip >= lo) & (ip < hi)
            out['by_price'][hs][f'{lo}-{hi:g}'] = dict(n=int(m.sum()), mid_c=float(vmid[m].mean() / Q), take_back_c=float(back[m].mean() / Q), hold_c=float(hold[m].mean() / Q),
                                                     spread_after_median_c=float(np.median(spr[m])) if m.any() else None)
    TR.dump(fn, out)
    print(json.dumps(out['by_price'], indent=0)[:3000])


def selftest():
    # long YES: 50 bought at 40 (fee 0.8 each); exit bids 45 x 30, 44 x 10; 10 left settle at payout 1
    eP = np.array([[45., 44., -1., -1., -1.]]); eS = np.array([[30., 10., 0., 0., 0.]])
    fl, co, fe = np.array([50.]), np.array([50 * 40.]), np.array([50 * fee(40.)])
    back, hold, xq = exit_values(fl, co, fe, np.array([1.]), eP, eS, np.array([1.]))
    want_back = 30 * 45 + 10 * 44 + 10 * 100 - 2000 - 50 * fee(40.) - (30 * fee(45.) + 10 * fee(44.))
    assert abs(back[0] - want_back) < 1e-9 and xq[0] == 40., (back, want_back, xq)
    assert abs(hold[0] - (50 * 100 - 2000 - 50 * fee(40.))) < 1e-9
    # short YES: 20 sold at 60; buy back at asks 58 x 50; payout 0
    eP = np.array([[58., 59., -1., -1., -1.]]); eS = np.array([[50., 10., 0., 0., 0.]])
    fl, co, fe = np.array([20.]), np.array([20 * 60.]), np.array([20 * fee(60.)])
    back, hold, xq = exit_values(fl, co, fe, np.array([-1.]), eP, eS, np.array([0.]))
    assert abs(back[0] - (20 * 60 - 20 * 58 - 20 * fee(60.) - 20 * fee(58.))) < 1e-9 and xq[0] == 20.
    assert abs(hold[0] - (20 * 60 - 20 * fee(60.))) < 1e-9
    # nothing filled: nothing to exit
    back, hold, xq = exit_values(np.array([0.]), np.array([0.]), np.array([0.]), np.array([1.]), eP, eS, np.array([1.]))
    assert back[0] == 0. and hold[0] == 0. and xq[0] == 0.
    # walk_q equals vwalk at q = 50
    rng = np.random.default_rng(3)
    P = np.sort(rng.integers(1, 99, (500, 5)).astype(float), axis=1); S = rng.integers(1, 60, (500, 5)).astype(float)
    dd = np.ones(500); L_ = rng.integers(1, 99, 500).astype(float)
    a, b = TK.vwalk(P, S, dd, L_), walk_q(P, S, dd, L_, np.full(500, 50.))
    assert all(np.allclose(x, y) for x, y in zip(a, b))
    s = summary(np.array([100., -50., 25.]), np.array([0, 1, 2]), np.array(['20261023', '20261031', '20261101']))
    assert abs(s['value_per_intended_c'] - 75. / 3 / Q) < 1e-12 and s['days_positive'] == 2 and s['paying'] is False
    print('take_markout_exit selftest PASS')


if __name__ == '__main__':
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == 'labels':
        labels(a[0], a[1] if len(a) > 1 else None, int(a[2]) if len(a) > 2 else None)
    elif cmd == 'read':
        read(a[0] if a else None)
    else:
        dict(selftest=selftest)[cmd]()
