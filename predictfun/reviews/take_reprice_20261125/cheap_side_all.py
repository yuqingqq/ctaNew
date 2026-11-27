"""Cheap sides on every market, not only 'complete' ones (correction to TAKE_MARKOUT / TAKE_MARKOUT_LONGSHOT, found while checking BTC's
drift, USER 2026-11-27: "We check the btc price drift,"). 'Complete' (TAKE_MODEL's population) requires a valid two-sided Predict book
at the path end, which mostly happens when BTC ends near its start: on these days its markets' end move has an sd of 1.2-1.9 bps against
3-16 bps for the other markets. Outcome and price-path results on it are outcome-selected. Declared before the run.
Population: public_score3's random moments (both directions) in every market of the four consumed days whose recorded path (vm_path, any
status) holds the decision (>= 1 s in) and the match; statuses counted (loader, no path, outside the path, invalid known book).
Per moment: TAKE_MODEL's take at T = 172 ms (known best at the decision as the limit, the venue book at the match, 50 shares, fee per
level); held to settlement (registry: 100 Up / 0 Down / 50 tie); sold back after 30 s through the speed bump where the path holds the
exit's match (TAKE_MARKOUT_EXIT's rule; else a status); the market's path status and BTC's end move (registry start / end).
Readouts (stage read): by path status (complete / not complete / all): markets, BTC end move (median |move|, sd); calibration of the side
bought (price paid vs won, filled shares; and over market sides) by price bucket; the value per intended share of sides under 10c and
10-20c held to settlement (by direction, by day) and sold back after 30 s where covered. Check: the complete markets' TAKE_MODEL rows
reproduce TAKE_MARKOUT_LONGSHOT's held values and calibration for sides under 10c.
Outputs (write-once): OUT/cheap_side_all/<day>.parquet + .json (labels), cheap_side_all.json (read).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/cheap_side_all.py \
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
import take_markout_exit as TE  # noqa: E402

TM, VR, MM, TR, C, F, WT = TK.TM, TK.VR, TK.MM, TK.TR, TK.C, TK.F, TK.WT
OUT = WT.OUT / 'cheap_side_all'
DAYS, Q, fee = TK.DAYS, TK.Q, TK.fee
NET, EXIT_DELAY, HOLD_S = 22., 172., 30000.
BINS = ((0, 10), (10, 20), (20, 35), (35, 65), (65, 80), (80, 90), (90, 100.01))
GROUPS = ('complete', 'not_complete', 'all')


def lab(lo, hi):
    return f'{lo}-{min(hi, 100):g}'


def payout(r):
    if float(r['start_price']) == float(r['end_price']):
        return .5
    return 1. if r['outcome'] == 'Up' else 0.


def labels(day, od=None, limit=None):
    resource.setrlimit(resource.RLIMIT_CPU, (2400, 2401))
    C.require(day in DAYS, f'{day} is not a declared day')
    od = Path(od) if od else OUT
    fp = od / f'{day}.parquet'
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    Z = pd.read_parquet(WT.OUT / 'public_score3_points.parquet', columns=['day', 't', 'd', 'kind'])
    Z['row'] = np.arange(len(Z))
    Z = Z[(Z.day == day) & (Z.kind == 'random')].reset_index(drop=True)
    reg = MM.registry(day).set_index('mid')
    rg = pd.read_parquet(MM.S1 / day / 'registry.parquet').set_index('mid')
    ws_to_mid = {int(float(reg.loc[m, 'ws'])): int(m) for m in reg.index}
    Z['mid'] = (np.floor(Z.t / 300000.) * 300).astype(int).map(ws_to_mid)
    status = collections.Counter()
    status['no_market'] = int(Z.mid.isna().sum())
    Z = Z[Z.mid.notna()].reset_index(drop=True)
    Z['mid'] = Z.mid.astype(int)
    tm_rows = set(pd.read_parquet(TM.OUT / f'{day}_labels.parquet', columns=['row']).row)
    by_mid = Z.groupby('mid').indices
    rows, path_st = [], {}
    src = MM.KP.Tape(); off = src.offset(day); n_m = 0
    for h, g in reg.groupby('hour'):
        mids = [int(m) for m in g.index if int(m) in by_mid]
        if limit is not None:
            mids = mids[:max(0, limit - n_m)]
        if not mids:
            continue
        for m, W, lst in VR.worlds_all(day, mids, src, off, reg):
            n_m += 1
            if lst != 'loaded':
                status[f'loader_{lst}'] += len(by_mid[m]); path_st[m] = f'loader_{lst}'
                continue
            ws = float(reg.loc[m, 'ws']) * 1000.
            p, st, info = VR.vm_path(W, ws, 50.)
            path_st[m] = st
            if p is None:
                status[f'path_{st}'] += len(by_mid[m])
                continue
            t0, end = float(info['t_start']), float(info['end_rel'])
            f = np.asarray(p['frames'], float)
            r_ = rg.loc[m]
            y = payout(r_)
            em = 1e4 * np.log(float(r_.end_price) / float(r_.start_price))
            for i in by_mid[m]:
                t, d, rw = float(Z.t.iat[i]), float(Z.d.iat[i]), int(Z.row.iat[i])
                g_ = t + TM.FEED + TM.DECIDE - t0
                act = g_ + NET + TM.HOLD
                if g_ < 1000. or act > end:
                    status['outside_path'] += 1
                    continue
                jk = int(np.searchsorted(f[:, 1], g_, 'right') - 1)
                if jk < 0:
                    status['no_known_frame'] += 1
                    continue
                rk = f[jk]
                bid, ask = rk[2], rk[12]
                if not (0 <= bid < ask <= 100):
                    status['invalid_known_book'] += 1
                    continue
                side = 1 if d > 0 else 0
                lim = ask if d > 0 else bid
                md = (bid + ask) / 2.
                jv = int(np.searchsorted(f[:, 0], act, 'right') - 1)
                pa, sa = TK.side_levels(f[jv], side)
                r = dict(row=rw, day=day, mid=m, d=d, lim=lim, md=md, its_price=md if d > 0 else 100. - md, path_status=st, payout=y, end_move_bps=em,
                         in_take_model=rw in tm_rows)
                for k in range(5):
                    r[f'pA_{k}'], r[f'sA_{k}'] = pa[k], sa[k]
                xe = act + HOLD_S + EXIT_DELAY
                if xe <= end:
                    pe, se = TK.side_levels(f[int(np.searchsorted(f[:, 0], xe, 'right') - 1)], 0 if d > 0 else 1)
                    r['exit_ok'] = True
                else:
                    pe, se = np.full(5, -1.), np.zeros(5)
                    r['exit_ok'] = False
                    status['no_exit_book_30s'] += 1
                for k in range(5):
                    r[f'eP_{k}'], r[f'eS_{k}'] = pe[k], se[k]
                status['ok'] += 1
                rows.append(r)
            C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    L = pd.DataFrame(rows)
    od.mkdir(parents=True, exist_ok=True)
    L.to_parquet(fp, index=False)
    TR.dump(od / f'{day}.json', dict(asof='2026-11-27', day=day, markets=n_m, status=dict(status), path_status=dict(collections.Counter(path_st.values())),
                                     take_model_rows_found=int(L.in_take_model.sum()), take_model_rows=len(tm_rows)))
    print(day, n_m, dict(status), dict(collections.Counter(path_st.values())), 'take_model rows', int(L.in_take_model.sum()), len(tm_rows), flush=True)


def values(L):
    d, lim = L.d.to_numpy(float), L.lim.to_numpy(float)
    PA = L[[f'pA_{k}' for k in range(5)]].to_numpy(float); SA = L[[f'sA_{k}' for k in range(5)]].to_numpy(float)
    eP = L[[f'eP_{k}' for k in range(5)]].to_numpy(float); eS = L[[f'eS_{k}' for k in range(5)]].to_numpy(float)
    fl, co, fe = TK.vwalk(PA, SA, d, lim)
    back, hold, _ = TE.exit_values(fl, co, fe, d, eP, eS, L.payout.to_numpy(float))
    return fl, co, back, hold


def group_masks(L):
    c = (L.path_status == 'complete').to_numpy()
    return dict(complete=c, not_complete=~c, all=np.ones(len(L), bool))


def calibration(L, fl, co, m):
    d, pay, ip = L.d.to_numpy(float), L.payout.to_numpy(float), L.its_price.to_numpy(float)
    win = np.where(d > 0, pay, 1. - pay)
    paid = np.where(fl > 0, np.where(d > 0, co / np.where(fl > 0, fl, 1.), 100. - co / np.where(fl > 0, fl, 1.)), np.nan)
    out = {}
    for lo, hi in BINS:
        b = m & (ip >= lo) & (ip < hi)
        f_ = b & (fl > 0)
        mk = pd.DataFrame(dict(mid=L.mid.to_numpy()[b], d=d[b], w=win[b])).drop_duplicates(['mid', 'd'])
        out[lab(lo, hi)] = dict(moments=int(b.sum()), filled=int(f_.sum()), its_price_paid_c=float(np.average(paid[f_], weights=fl[f_])) if f_.any() else None,
                                won_c_moments=float(100. * np.average(win[f_], weights=fl[f_])) if f_.any() else None,
                                market_sides=int(len(mk)), won_c_market_sides=float(100. * mk.w.mean()) if len(mk) else None)
    return out


def cell(v, m, day, mid):
    if not m.any():
        return dict(n=0)
    by = {dh: (float(v[m & (day == dh)].mean() / Q) if (m & (day == dh)).any() else None) for dh in DAYS}
    return dict(n=int(m.sum()), markets=int(np.unique(mid[m]).size), value_c=float(v[m].mean() / Q), by_day=by,
                days_positive=int(sum(1 for x in by.values() if x is not None and x > 0)))


def read(od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (1200, 1201))
    reh = od is not None
    od = Path(od) if od else None
    fn = (od or H) / 'cheap_side_all.json'
    C.require(not fn.exists(), 'cheap_side_all.json exists (write-once)')
    src = od or OUT
    L = pd.concat([pd.read_parquet(src / f'{dh}.parquet') for dh in DAYS if (src / f'{dh}.parquet').exists()], ignore_index=True)
    C.require(L.row.is_unique, 'a moment twice')
    fl, co, back, hold = values(L)
    day, mid, d, ip = L.day.to_numpy(), L.mid.to_numpy(), L.d.to_numpy(float), L.its_price.to_numpy(float)
    G = group_masks(L)
    out = dict(asof='2026-11-27', rows=int(len(L)), rehearsal=reh, label_status={dh: json.loads((src / f'{dh}.json').read_text()) for dh in DAYS if (src / f'{dh}.json').exists()},
               groups={})
    M = L.drop_duplicates('mid')
    for gn, m in G.items():
        mm = group_masks(M)[gn]
        e = M.end_move_bps.to_numpy(float)[mm]
        o = dict(markets=int(mm.sum()), moments=int(m.sum()), btc_end_move_bps=dict(median_abs=float(np.median(np.abs(e))), sd=float(np.std(e))),
                 calibration=calibration(L, fl, co, m), cheap={})
        ex = L.exit_ok.to_numpy(bool)
        for dn, dm in (('up', d > 0), ('down', d < 0), ('both', np.ones(len(L), bool))):
            for lo, hi in ((0, 10), (10, 20)):
                b = m & dm & (ip >= lo) & (ip < hi)
                o['cheap'][f'{dn}_{lab(lo, hi)}'] = dict(held=cell(hold, b, day, mid), sold_back_30s=cell(back, b & ex, day, mid),
                                                        exit_covered_share=float(ex[b].mean()) if b.any() else None)
        out['groups'][gn] = o
        print(gn, o['markets'], o['btc_end_move_bps'], {k: (v['held'].get('value_c'), v['held'].get('days_positive')) for k, v in o['cheap'].items()}, flush=True)
    # check: TAKE_MODEL's rows of the complete markets reproduce TAKE_MARKOUT_LONGSHOT
    if not reh:
        ref = json.loads((H / 'take_markout_longshot.json').read_text())
        tmr = L.in_take_model.to_numpy(bool)
        C.require(int(tmr.sum()) == ref['rows'], f'TAKE_MODEL rows found {int(tmr.sum())} of {ref["rows"]}')
        for dn, dm in (('up', d > 0), ('down', d < 0)):
            b = tmr & dm & (ip < 10)
            mine, theirs = float(hold[b].mean() / Q), ref['by_direction'][dn]['0-10']['hold_to_settlement']['value_c']
            C.require(abs(mine - theirs) < 1e-9, f'held value not reproduced ({dn}): {mine} vs {theirs}')
        cal = calibration(L, fl, co, tmr)['0-10']
        C.require(abs(cal['won_c_moments'] - ref['calibration']['0-10']['won_c_moments']) < 1e-9 and abs(cal['its_price_paid_c'] - ref['calibration']['0-10']['its_price_paid_c']) < 1e-9,
                  'calibration not reproduced')
        out['reproduced_take_markout_longshot'] = True
    a, c_ = out['groups']['all']['calibration']['0-10'], out['groups']['complete']['calibration']['0-10']
    out['predicates'] = dict(cheap_won_minus_paid_all_c=a['won_c_moments'] - a['its_price_paid_c'], cheap_won_minus_paid_complete_c=c_['won_c_moments'] - c_['its_price_paid_c'],
                             cheap_held_up_all_c=out['groups']['all']['cheap']['up_0-10']['held']['value_c'], cheap_held_down_all_c=out['groups']['all']['cheap']['down_0-10']['held']['value_c'],
                             complete_end_move_sd_below_others=bool(out['groups']['complete']['btc_end_move_bps']['sd'] < out['groups']['not_complete']['btc_end_move_bps']['sd']))
    TR.dump(fn, out)
    print(json.dumps(out['predicates'], indent=1))


def selftest():
    L = pd.DataFrame(dict(row=[1, 2, 3], day=['20261023'] * 3, mid=[10, 10, 11], d=[1., -1., 1.], lim=[5., 95., 50.], md=[4.5, 95.5, 49.5],
                          its_price=[4.5, 4.5, 49.5], path_status=['complete', 'truncated', 'complete'], payout=[1., 1., 0.], end_move_bps=[1., 1., -3.],
                          in_take_model=[True, False, True], exit_ok=[False, False, False]))
    for k in range(5):
        L[f'pA_{k}'] = [5. + k, 95. - k, 50. + k]; L[f'sA_{k}'] = [100., 100., 100.]
        L[f'eP_{k}'] = -1.; L[f'eS_{k}'] = 0.
    fl, co, back, hold = values(L)
    assert fl.tolist() == [50., 50., 50.]
    # buy Up at 5, Up wins: 50 x (100 - 5) - fee; sell Up at 95 (= buy Down at 5), Up wins: 50 x (95 - 100) - fee
    assert abs(hold[0] - (50 * 95 - 50 * fee(5.))) < 1e-9 and abs(hold[1] - (50 * (95 - 100) - 50 * fee(95.))) < 1e-9
    assert np.allclose(back, hold)                                         # no exit book: everything settles
    G = group_masks(L)
    assert G['complete'].tolist() == [True, False, True] and G['all'].all()
    cal = calibration(L, fl, co, G['all'])['0-10']
    assert cal['moments'] == 2 and abs(cal['its_price_paid_c'] - 5.) < 1e-12 and abs(cal['won_c_moments'] - 50.) < 1e-12 and cal['market_sides'] == 2
    print('cheap_side_all selftest PASS')


if __name__ == '__main__':
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == 'labels':
        labels(a[0], a[1] if len(a) > 1 else None, int(a[2]) if len(a) > 2 else None)
    elif cmd == 'read':
        read(a[0] if a else None)
    else:
        dict(selftest=selftest)[cmd]()
