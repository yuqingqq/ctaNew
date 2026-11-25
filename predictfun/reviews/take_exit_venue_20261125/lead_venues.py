"""Which venue moved first when the takers who beat our 172 ms takes reached Predict? Diagnostic on four consumed days (no strategy change).
Events: the websocket-timed competitor fills of takers_timing.py, one event per competitor order (mid, print time, book side), arrival at
Predict A = print - 150 ms, direction d = the competitor's (and our take's) BTC direction; class early = no Binance futures mid change in
(A - 250, A], else late. Venues (trade prints, exchange clocks): Binance futures, Binance spot, Bybit perp / spot, OKX swap / spot, Coinbase,
Kraken, Deribit. Per event and venue: the first trade in [A - 500, A + 300] ms whose trailing 250 ms return in direction d reaches the
threshold (our trigger's: up 0.285 / down 0.467 bps; and a low 0.15 bps) -> onset and lead = A - onset. Placebos at A - 10 s and A + 10 s
give the base rate. Days: 20261023, 20261031, 20261101, 20261102 (every venue present). OKX day files run 16:00-16:00 UTC: rows outside the
read day are dropped right after parsing (no 10-30 row kept). Output (write-once): lead_venues.json, events in OUT/lead_events.parquet.
Usage: PF_SLOT=claude predictfun/research/run_light.sh lead_venues -- python3 predictfun/reviews/take_exit_venue_20261125/lead_venues.py
"""
import collections
import datetime as dt
import gzip
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import venue_common as V  # noqa: E402
import who_took as WT  # noqa: E402

F, OD, MM, TR, C = V.F, V.OD, V.MM, V.TR, V.C
DATA = F.X.DATA.parents[1]                       # data/
S1D, HOLD = WT.S1D, 150.
DAYS = ('20261023', '20261031', '20261101', '20261102')
THR = dict(ours={1: 0.285456120967865, -1: 0.4666730940341952}, low={1: 0.15, -1: 0.15})
WIN = (-500., 300.)
PLACEBO = (-10000., 10000.)
OUT = F.X.DATA / 'take_exit_venue_lead_20261125'
VENUES = ('binance_fut', 'binance_spot', 'bybit_perp', 'bybit_spot', 'okx_swap', 'okx_spot', 'coinbase', 'kraken', 'deribit')


def day_ms(day):
    d0 = dt.datetime.strptime(day, '%Y%m%d').replace(tzinfo=dt.timezone.utc)
    return int(d0.timestamp() * 1000), int((d0 + dt.timedelta(days=1)).timestamp() * 1000)


def iso(day, delta=0):
    return (dt.datetime.strptime(day, '%Y%m%d') + dt.timedelta(days=delta)).strftime('%Y-%m-%d')


def zcsv(p, **kw):
    z = zipfile.ZipFile(p)
    return pd.read_csv(z.open(z.namelist()[0]), **kw)


def load_venue(v, day):
    """(t_ms, price) of every trade on venue v in UTC day `day`, sorted; rows outside the day dropped at once."""
    lo, hi = day_ms(day)
    E = DATA / 'exchange_public'
    if v == 'binance_fut':
        fs = [DATA / f'mm_hf/raw/trade/BTCUSDT/{day}_{h:02d}.csv.gz' for h in range(24)]
        d = pd.concat([pd.read_csv(f, header=None, usecols=[2, 4], names=['t', 'p']) for f in fs if f.exists()])
    elif v == 'binance_spot':
        d = zcsv(DATA / f'binance_spot_vision/aggTrades/BTCUSDT/BTCUSDT-aggTrades-{iso(day)}.zip', header=None, usecols=[1, 5], names=['p', 't'])
        d['t'] = d.t / 1000.
    elif v == 'bybit_perp':
        d = pd.read_csv(E / f'bybit_perp/BTCUSDT{iso(day)}.csv.gz', usecols=['timestamp', 'price']).rename(columns={'timestamp': 't', 'price': 'p'}); d['t'] = d.t * 1000.
    elif v == 'bybit_spot':
        d = pd.read_csv(E / f'bybit_spot/BTCUSDT_{iso(day)}.csv.gz', usecols=['timestamp', 'price']).rename(columns={'timestamp': 't', 'price': 'p'})
    elif v in ('okx_swap', 'okx_spot'):
        nm = 'BTC-USDT-SWAP' if v == 'okx_swap' else 'BTC-USDT'
        parts = []
        for k in (0, 1):                                                   # file D covers D-1 16:00 .. D 16:00; file D+1 the rest of D
            x = zcsv(E / f'{v}/{nm}-trades-{iso(day, k)}.zip', usecols=['created_time', 'price'], encoding_errors='replace')
            x = x[(x.created_time >= lo) & (x.created_time < hi)]            # cut to the read day right after parsing (no 10-30 row kept)
            parts.append(x)
        d = pd.concat(parts).rename(columns={'created_time': 't', 'price': 'p'})
    else:
        d = pd.read_csv(E / f'{v}/{day}.csv.gz', usecols=['t_ms', 'price']).rename(columns={'t_ms': 't', 'price': 'p'})
    d = d[(d.t >= lo) & (d.t < hi)].sort_values('t', kind='stable')
    return d.t.to_numpy(float), d.p.to_numpy(float)


def onset(t, p, a, d, thr):
    """First upward crossing of thr (bps) by the trailing 250 ms return in direction d at a trade in [a + WIN0, a + WIN1] -> its time.
    If the return is already at or above thr at the window's first trade, the move began earlier: ALREADY (a + WIN0 - 1). None if no cross."""
    i0, i1 = np.searchsorted(t, a + WIN[0], 'left'), np.searchsorted(t, a + WIN[1], 'right')
    if i1 <= i0:
        return None
    lo = max(0, i0 - 1)                                                      # one trade of context before the window
    tt, pp = t[lo:i1], p[lo:i1]
    jr = np.searchsorted(t, tt - 250., 'right') - 1
    r = np.where(jr >= 0, d * 1e4 * np.log(pp / p[np.maximum(jr, 0)]), np.nan)
    above = r >= thr
    start = i0 - lo                                                          # index of the window's first trade in tt
    if above[start] and (start == 0 or above[start - 1]):
        return a + WIN[0] - 1.                                               # already moving before the window (flag: lead > 500 ms)
    cross = np.flatnonzero(above[start:] & ~np.r_[above[start - 1] if start > 0 else False, above[start:-1]])
    return float(tt[start + cross[0]]) if len(cross) else None


def events():
    """Competitor orders that took the quotes our 172 ms takes missed (websocket-timed fills), one row per order."""
    sm = json.loads((S1D / 'summary.json').read_text())
    C.require(sm['delay_ms'] == 172., 'not the 172 ms run')
    want = collections.defaultdict(set)
    for r in sm['rows']:
        if r['day'] in DAYS:
            want[r['day']].add(int(r['mid']))
    src, rows = MM.KP.Tape(), []
    for day in DAYS:
        L = pd.read_parquet(MM.S1 / day / 'legs_timed2.parquet', columns=['mid', 'side', 'ptick', 'shares', 't_ms2', 'how2'])
        L = L[L.mid.isin(want[day]) & (L.how2 == 1)]
        by_mid = {m: g for m, g in L.groupby('mid')}
        bk, bm = OD.seg_arrays(OD.SEG_OF[day]); bk = bk.astype(float)
        reg = MM.registry(day).set_index('mid'); off = src.offset(day)
        for h, g in reg.groupby('hour'):
            mids = [int(m) for m in g.index if int(m) in want[day]]
            if not mids:
                continue
            for m, st, p, info, yup in F.worlds_yup(day, mids, src, off, reg):
                lg = by_mid.get(m)
                if lg is None:
                    continue
                t0 = float(info['t_start']); f = np.asarray(p['frames'], float); pt = lg.t_ms2.values.astype(float) - t0
                led = json.loads(gzip.decompress((S1D / f'{m}_BASE.json.gz').read_bytes()))
                seen = set()
                for x in led['ioc_results']:
                    if x['reason'] != 'limit_moved':
                        continue
                    fo = WT.follow(p, x)
                    if fo['gone_ms'] is None:
                        continue
                    s, Lp, gd = int(x['side']), float(x['limit']), float(x['submit']); opp = 1 - s
                    jk = int(np.searchsorted(f[:, 1], gd, side='right') - 1)
                    sel = (lg.side.values == opp) & ((lg.ptick.values >= Lp) if opp == 1 else (lg.ptick.values <= Lp)) & (pt >= f[jk, 0]) & (pt <= gd + fo['gone_ms'] + 100.)
                    d = 1 if s == 0 else -1
                    for i in np.flatnonzero(sel):
                        key = (m, float(lg.t_ms2.values[i]), opp)
                        if key in seen:
                            continue
                        seen.add(key)
                        A = t0 + pt[i] - HOLD
                        j0 = int(np.searchsorted(bk, A, 'right') - 1); j1 = int(np.searchsorted(bk, A - 250., 'right') - 1)
                        if j0 < 0 or j1 < 0:
                            continue
                        k = j0 + 1; tf_after = None
                        while k < len(bk) and bk[k] <= A + 2000.:
                            if d * (bm[k] - bm[j0]) > 0:
                                tf_after = float(bk[k]); break
                            k += 1
                        rows.append(dict(day=day, mid=m, A=A, d=d, cls='early' if j0 == j1 else 'late', decision=t0 + gd,
                                         shares=float(lg[(lg.t_ms2 == lg.t_ms2.values[i]) & (lg.side == opp)].shares.sum()), tf_after=tf_after))
        print(day, 'events', len(rows), flush=True)
    return pd.DataFrame(rows)


def main():
    fn = H / 'lead_venues.json'
    C.require(not fn.exists(), 'lead_venues.json exists (write-once)')
    OUT.mkdir(parents=True, exist_ok=True)
    E = events()
    res = collections.defaultdict(dict)
    on = {(v, th, pl): np.full(len(E), np.nan) for v in VENUES for th in THR for pl in ('ev', 'pm', 'pp')}
    for day in DAYS:
        ix = np.flatnonzero(E.day.values == day)
        for v in VENUES:
            t, p = load_venue(v, day)
            res['trades_per_day'][f'{v}|{day}'] = int(len(t))
            for i in ix:
                a, d = E.A.values[i], int(E.d.values[i])
                for th, tb in THR.items():
                    for pl, sh in (('ev', 0.), ('pm', PLACEBO[0]), ('pp', PLACEBO[1])):
                        o = onset(t, p, a + sh, d, tb[d])
                        on[(v, th, pl)][i] = np.nan if o is None else o - (a + sh)         # onset relative to the (placebo) arrival
            del t, p
        print(day, 'venues done', flush=True)
    for (v, th, pl), arr in on.items():
        E[f'{v}|{th}|{pl}'] = arr
    E.to_parquet(OUT / 'lead_events.parquet', index=False)
    out = dict(asof='2026-11-25', days=list(DAYS), events=len(E), events_by_class=dict(collections.Counter(E.cls)),
               window_ms=list(WIN), placebo_shift_ms=list(PLACEBO), thresholds=THR, trades_per_day=res['trades_per_day'], venues={})
    for v in VENUES:
        for th in THR:
            blk = {}
            for c in ('early', 'late'):
                X = E[E.cls == c]
                ev = X[f'{v}|{th}|ev'].values; pl = np.r_[X[f'{v}|{th}|pm'].values, X[f'{v}|{th}|pp'].values]
                before = np.isfinite(ev) & (ev <= 0); pbefore = np.isfinite(pl) & (pl <= 0)
                lead = -ev[before]
                rel_tf = (X.A.values + ev - X.tf_after.values)[before & np.isfinite(X.tf_after.values)] if c == 'early' else None
                blk[c] = dict(events=len(X), onset_before_arrival=float(before.mean()), placebo_rate=float(pbefore.mean()),
                              lift=float(before.mean() / pbefore.mean()) if pbefore.mean() > 0 else None,
                              lead_ms_p25_p50_p75=[float(np.quantile(lead, q)) for q in (.25, .5, .75)] if len(lead) else None,
                              onset_rel_futures_move_ms_p25_p50_p75=[float(np.quantile(rel_tf, q)) for q in (.25, .5, .75)] if rel_tf is not None and len(rel_tf) else None)
            out['venues'][f'{v}|{th}'] = blk
    for th in THR:                                                          # the earliest venue per early event
        X = E[E.cls == 'early']
        M = np.column_stack([np.where(np.isfinite(X[f'{v}|{th}|ev'].values) & (X[f'{v}|{th}|ev'].values <= 0), X[f'{v}|{th}|ev'].values, np.inf) for v in VENUES])
        has = np.isfinite(M.min(1))
        first = collections.Counter(VENUES[j] for j in M[has].argmin(1))
        out[f'earliest_{th}'] = dict(early_events=len(X), with_any_onset_before_arrival=float(has.mean()), first_venue=dict(first.most_common()),
                                     reaction_ms_arrival_minus_earliest_p25_p50_p75=[float(np.quantile(-M[has].min(1), q)) for q in (.25, .5, .75)] if has.any() else None)
    TR.dump(fn, out)
    print(json.dumps(out, indent=1)[:6000])


if __name__ == '__main__':
    main()
