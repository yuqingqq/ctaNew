"""Do other venues' trades show what the top taker acts on? (USER 2026-11-26: "So you recommend to include more exchanges ob dynamics, i
agree"; recorded order books exist only for Binance futures, so the other venues are read from their trade prints.) Readout; the four
consumed days (it traded on three); no strategy change.
Points: fire_breakdown_points.parquet (its 5,930 fires, t = arrival at Predict, and nearby moments = the same times +-5..30 s), with their
breakdown row (what the Binance futures best level did in the 300 ms before) and the Binance futures move its way within 300 ms after.
Venues (trade prints, each exchange's own clock; aggressor side and size): Binance spot (Vision aggTrades), Bybit perp / spot, OKX swap
(contracts x 0.01 BTC) / spot, Coinbase, Kraken, Deribit (+1 buy / -1 sell taker), Binance futures ETHUSDT (cross-asset, ETH), and Binance
futures BTCUSDT as the reference. OKX day files run 16:00-16:00 UTC: a file holding a sealed (10-28..30) or reserved (>= 11-04) day is never
opened (10-31 is covered from 16:00 only); a point outside a venue's coverage or with no trade in the previous 60 s has no value there.
Per point and venue: aggressive volume its way over (t - 50 / 100 / 300, t], against over (t - 100, t], the largest per-ms group its way over
(t - 100, t] and its time, the last-trade return its way over 100 / 300 ms (bps).
Declared before the run: a burst = volume its way over 100 ms above the venue's 99th percentile at nearby moments (also the 95th; and the
100 ms return above its nearby 99th); row groups: quiet (unchanged + small changes), far side shrank, other flat (near grew, other activity,
moved against), moved its way. Readings per venue and for any of the eight other venues: burst share at its fires vs nearby, by row group;
the Binance move within 300 ms after nearby moments with / without a burst among points with no prior Binance move its way (is it a
public signal?), and after its fires; for fires with a burst, the burst's time before t.
Outputs (write-once): cross_venue.json; OUT/cross_venue_points.parquet.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/cross_venue.py \
       [<rehearsal dir> <day>]
"""
import datetime as dt
import json
import resource
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import fire_events as FE  # noqa: E402

DC, C, TR, WT = FE.DC, FE.C, FE.TR, FE.WT
DATA = WT.LV.DATA
EP = DATA / 'exchange_public'
SEALED = {'2026-10-28', '2026-10-29', '2026-10-30'}
FORWARD0 = '2026-11-04'
OTHER = ('binance_spot', 'bybit_perp', 'bybit_spot', 'okx_swap', 'okx_spot', 'coinbase', 'kraken', 'deribit')
VENUES = OTHER + ('eth_fut', 'btc_fut')
WINS, STALE_MS, P_HI, P_LO = (50., 100., 300.), 60000., .99, .95
ROWG = dict(flat_unchanged='quiet', flat_small_changes='quiet', flat_far_shrank='far_shrank', flat_near_grew='other_flat', flat_other_activity='other_flat',
            moved_against_only='other_flat', moved_its_way='moved_its_way')
QS = (.1, .25, .5, .75, .9)


def iso(day, k=0):
    return (dt.datetime.strptime(day, '%Y%m%d') + dt.timedelta(days=k)).strftime('%Y-%m-%d')


def allowed(date_iso):
    return date_iso not in SEALED and date_iso < FORWARD0


def zcsv(p, **kw):
    z = zipfile.ZipFile(p)
    return pd.read_csv(z.open(z.namelist()[0]), **kw)


def load(v, day):
    """(t_ms, price, size, sign +1 buy / -1 sell aggressor) sorted by t, and the covered interval [lo, hi) of UTC day `day`."""
    C.require(day in WT.DAYS and allowed(iso(day)), f'{day} is not a declared day')
    lo, hi = (float(x) for x in WT.LV.day_ms(day))
    if v in ('btc_fut', 'eth_fut'):
        sym = 'BTCUSDT' if v == 'btc_fut' else 'ETHUSDT'
        fs = [DATA / f'mm_hf/raw/trade/{sym}/{day}_{h:02d}.csv.gz' for h in range(24)]
        d = pd.concat([pd.read_csv(f, header=None, usecols=[1, 3, 4, 5, 6], names=['t', 'id', 'p', 'q', 'm']) for f in fs if f.exists()]).drop_duplicates('id')
        d['s'] = np.where(d.m == DC.BUY_AGGR, 1, -1)
    elif v == 'binance_spot':
        d = zcsv(DATA / f'binance_spot_vision/aggTrades/BTCUSDT/BTCUSDT-aggTrades-{iso(day)}.zip', header=None, usecols=[1, 2, 5, 6], names=['p', 'q', 't', 'bm'])
        d['t'] = d.t / 1000.
        d['s'] = np.where(d.bm.astype(str).str.lower() == 'true', -1, 1)
    elif v == 'bybit_perp':
        d = pd.read_csv(EP / f'bybit_perp/BTCUSDT{iso(day)}.csv.gz', usecols=['timestamp', 'side', 'size', 'price']).rename(columns={'timestamp': 't', 'size': 'q', 'price': 'p'})
        d['t'] = d.t * 1000.
        d['s'] = np.where(d.side == 'Buy', 1, -1)
    elif v == 'bybit_spot':
        d = pd.read_csv(EP / f'bybit_spot/BTCUSDT_{iso(day)}.csv.gz', usecols=['timestamp', 'price', 'volume', 'side']).rename(columns={'timestamp': 't', 'price': 'p', 'volume': 'q'})
        d['s'] = np.where(d.side.str.lower() == 'buy', 1, -1)
    elif v in ('okx_swap', 'okx_spot'):
        nm = 'BTC-USDT-SWAP' if v == 'okx_swap' else 'BTC-USDT'
        parts, cov = [], []
        for k, (a, b) in ((0, (iso(day, -1), iso(day))), (1, (iso(day), iso(day, 1)))):        # file k covers a 16:00 .. b 16:00 UTC
            if not (allowed(a) and allowed(b)):
                continue                                                                      # never opened: it holds a sealed / reserved day
            x = zcsv(EP / f'{v}/{nm}-trades-{iso(day, k)}.zip', usecols=['side', 'price', 'size', 'created_time'], encoding_errors='replace')
            x = x[(x.created_time >= lo) & (x.created_time < hi)]
            parts.append(x)
            cov.append((lo, lo + 16 * 3600000.) if k == 0 else (lo + 16 * 3600000., hi))
        C.require(len(parts) > 0, f'{v} {day}: no allowed file')
        d = pd.concat(parts).rename(columns={'created_time': 't', 'price': 'p', 'size': 'q'})
        if v == 'okx_swap':
            d['q'] = d.q * .01
        d['s'] = np.where(d.side == 'buy', 1, -1)
        lo, hi = min(c[0] for c in cov), max(c[1] for c in cov)
    else:
        d = pd.read_csv(EP / f'{v}/{day}.csv.gz', usecols=['t_ms', 'price', 'size_btc', 'taker']).rename(columns={'t_ms': 't', 'price': 'p', 'size_btc': 'q'})
        d['s'] = np.sign(d.taker).astype(int)
    d = d[(d.t >= lo) & (d.t < hi)].sort_values('t', kind='stable')
    C.require(len(d) > 0 and np.isin(d.s.unique(), [-1, 1]).all(), f'{v} {day}: empty or unsided')
    return d.t.to_numpy(float), d.p.to_numpy(float), d.q.to_numpy(float), d.s.to_numpy(int), (lo, hi)


def feats(tv, pv, qv, sv, cov, t, d):
    """Per point (direction d): volumes its way / against, the largest per-ms group its way and its time, returns its way."""
    n = len(t)
    o = {f'v{int(w)}': np.full(n, np.nan) for w in WINS}
    o.update(a100=np.full(n, np.nan), g100=np.full(n, np.nan), g100_t=np.full(n, np.nan), r100=np.full(n, np.nan), r300=np.full(n, np.nan))
    j = np.searchsorted(tv, t, 'right') - 1
    ok = (t - 300. >= cov[0]) & (t < cov[1]) & (j >= 0) & (t - tv[np.maximum(j, 0)] <= STALE_MS)
    for sg in (1, -1):
        m = sv == sg
        te, cq = tv[m], np.r_[0., np.cumsum(qv[m])]
        msec = np.floor(te)
        u, idx = np.unique(msec, return_index=True)
        gq = np.add.reduceat(qv[m], idx) if len(idx) else np.zeros(0)
        for dv in (1., -1.):
            pt = ok & (d == dv)
            if not pt.any():
                continue
            tt = t[pt]
            if sg == int(dv):
                for w in WINS:
                    o[f'v{int(w)}'][pt] = cq[np.searchsorted(te, tt, 'right')] - cq[np.searchsorted(te, tt - w, 'right')]
                a, b = np.searchsorted(u, tt - 100., 'right'), np.searchsorted(u, tt, 'right')
                g, gt = np.zeros(len(tt)), np.full(len(tt), np.nan)
                for i in np.flatnonzero(b > a):
                    k = a[i] + int(np.argmax(gq[a[i]:b[i]]))
                    g[i], gt[i] = gq[k], u[k] - tt[i]
                o['g100'][pt], o['g100_t'][pt] = g, gt
            else:
                o['a100'][pt] = cq[np.searchsorted(te, tt, 'right')] - cq[np.searchsorted(te, tt - 100., 'right')]
    for w in (100., 300.):
        j0 = np.searchsorted(tv, t - w, 'right') - 1
        r = np.where(ok & (j0 >= 0), d * 1e4 * np.log(pv[np.maximum(j, 0)] / pv[np.maximum(j0, 0)]), np.nan)
        o[f'r{int(w)}'] = r
    return o


def qs(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {f'p{int(q * 100)}': float(np.quantile(x, q)) for q in QS} if len(x) else None


def share(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else None


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (1500, 1501))
    od = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    fn, fp = (od or H) / 'cross_venue.json', (od or WT.OUT) / 'cross_venue_points.parquet'
    C.require(not fn.exists() and not fp.exists(), 'an output exists (write-once)')
    days = WT.DAYS if len(sys.argv) < 3 else (sys.argv[2],)
    C.require(all(x in WT.DAYS for x in days), 'not a declared day')
    P = pd.read_parquet(WT.OUT / 'fire_breakdown_points.parquet', columns=['dec', 'day', 't', 'd', 'shift_ms', 'row', 'their_prior', 'move300', 'next_ms'])
    P = P[P.day.isin(days)].reset_index(drop=True)
    P['grp'] = P.row.map(ROWG)
    C.require(P.grp.notna().all(), 'a row without a group')
    P['fire'] = P.shift_ms == 0.
    cols = {}
    for day in days:
        m = (P.day == day).to_numpy()
        t, d = P.t.to_numpy(float)[m], P.d.to_numpy(float)[m]
        for v in VENUES:
            tv, pv, qv, sv, cov = load(v, day)
            for k, x in feats(tv, pv, qv, sv, cov, t, d).items():
                cols.setdefault(f'{v}__{k}', np.full(len(P), np.nan))[m] = x
            del tv, pv, qv, sv
            C.require(DC.F.rss_mib() < DC.F.RSS_GUARD_MIB, 'RSS over the self-guard')
        print(day, int(m.sum()), flush=True)
    Z = pd.concat([P, pd.DataFrame(cols)], axis=1)
    N, Fz = Z[~Z.fire], Z[Z.fire]
    out = dict(asof='2026-11-26', days=list(days), n_fires=int(len(Fz)), n_nearby=int(len(N)), windows_ms=list(WINS), stale_ms=STALE_MS, q_hi=P_HI, q_lo=P_LO, venues={})
    burst_any = np.zeros(len(Z), bool)
    for v in VENUES:
        x = Z[f'{v}__v100']
        thr_hi, thr_lo = float(np.nanquantile(N[f'{v}__v100'], P_HI)), float(np.nanquantile(N[f'{v}__v100'], P_LO))
        rthr = float(np.nanquantile(N[f'{v}__r100'], P_HI))
        b_hi = np.where(x.notna(), x > thr_hi, np.nan)
        Z[f'{v}__burst'] = b_hi
        if v in OTHER:
            burst_any |= np.nan_to_num(b_hi, nan=0.) > 0
        o = dict(coverage_fires=share(x[Z.fire].notna()), coverage_nearby=share(x[~Z.fire].notna()), v100_p99_nearby=thr_hi, v100_p95_nearby=thr_lo, r100_p99_nearby_bps=rthr,
                 fires_burst99=share(b_hi[Z.fire.to_numpy()]), nearby_burst99=share(b_hi[~Z.fire.to_numpy()]),
                 fires_burst95=share(np.where(x.notna(), x > thr_lo, np.nan)[Z.fire.to_numpy()]), nearby_burst95=share(np.where(x.notna(), x > thr_lo, np.nan)[~Z.fire.to_numpy()]),
                 fires_ret99=share(np.where(Z[f'{v}__r100'].notna(), Z[f'{v}__r100'] > rthr, np.nan)[Z.fire.to_numpy()]),
                 nearby_ret99=share(np.where(Z[f'{v}__r100'].notna(), Z[f'{v}__r100'] > rthr, np.nan)[~Z.fire.to_numpy()]),
                 fires_v100_mean=float(np.nanmean(x[Z.fire])), nearby_v100_mean=float(np.nanmean(x[~Z.fire])),
                 burst_time_before_t_ms=qs(-Z.loc[Z.fire & (np.nan_to_num(b_hi, nan=0.) > 0), f'{v}__g100_t']), by_group={})
        for g in ('quiet', 'far_shrank', 'other_flat', 'moved_its_way'):
            mg = (Z.grp == g).to_numpy()
            o['by_group'][g] = dict(fires_burst99=share(b_hi[mg & Z.fire.to_numpy()]), nearby_burst99=share(b_hi[mg & ~Z.fire.to_numpy()]))
        noprior = ~Z.their_prior.to_numpy(bool)
        bb = np.nan_to_num(b_hi, nan=-1.)
        o['follow'] = dict(nearby_burst=share(Z.move300[~Z.fire & noprior & (bb == 1)]), nearby_no_burst=share(Z.move300[~Z.fire & noprior & (bb == 0)]),
                           n_nearby_burst=int((~Z.fire & noprior & (bb == 1)).sum()),
                           fires_burst=share(Z.move300[Z.fire & noprior & (bb == 1)]), fires_no_burst=share(Z.move300[Z.fire & noprior & (bb == 0)]))
        out['venues'][v] = o
    Z['any_burst'] = burst_any
    noprior = ~Z.their_prior.to_numpy(bool)
    out['any_other'] = dict(fires=share(Z.any_burst[Z.fire]), nearby=share(Z.any_burst[~Z.fire]),
                            by_group={g: dict(fires=share(Z.any_burst[Z.fire & (Z.grp == g)]), nearby=share(Z.any_burst[~Z.fire & (Z.grp == g)])) for g in ('quiet', 'far_shrank', 'other_flat', 'moved_its_way')},
                            follow=dict(nearby_burst=share(Z.move300[~Z.fire & noprior & Z.any_burst]), nearby_no_burst=share(Z.move300[~Z.fire & noprior & ~Z.any_burst]),
                                        n_nearby_burst=int((~Z.fire & noprior & Z.any_burst).sum()),
                                        fires_burst=share(Z.move300[Z.fire & noprior & Z.any_burst]), fires_no_burst=share(Z.move300[Z.fire & noprior & ~Z.any_burst])))
    Z.to_parquet(fp, index=False)
    TR.dump(fn, out)
    for v, o in out['venues'].items():
        print(f"{v:13s} cov {o['coverage_fires']:.2f} burst99 fires {o['fires_burst99']:.3f} nearby {o['nearby_burst99']:.3f} | quiet {o['by_group']['quiet']['fires_burst99']} vs {o['by_group']['quiet']['nearby_burst99']} "
              f"| follow nearby {o['follow']['nearby_burst']} vs {o['follow']['nearby_no_burst']} (n {o['follow']['n_nearby_burst']}) | lead {o['burst_time_before_t_ms']}")
    print(json.dumps(out['any_other'], indent=1))


if __name__ == '__main__':
    main()
