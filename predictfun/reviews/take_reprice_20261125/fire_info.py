"""The information we hold that the fire studies have not used, at the top taker's fires (USER 2026-11-26: "Check the items that we have
information first"; items 2-6 and 9 of the list given). Readout on the four consumed days (the top taker traded on three); no strategy
change.
Points (declared before the run): FIRE = all the top taker's fires (taker orders from exit_times' match-record scan, de-duplicated; one fire
= its orders in one market within fire_times' 2 ms burst rule; arrival at Predict = settlement - 184 ms; direction its bet) inside their
market's window [ws, ws + 300 s); the set must reproduce smart_fires/top_taker_fire_times_all.csv. NEARBY = each fire shifted by +-5, 10,
.., 30 s plus a seeded uniform offset in [-500, 500) ms (sub-second phase randomised), same market and direction, inside the window and
>= 1 s from any of its fires in that market. RANDOM = 20 seeded uniform times per market it traded, inside the window, seeded direction
(selection readings only).
Items:
2 reference: Predict's reference relay (the Chainlink price its site shows; whole-second source stamps; recorded since 10-22): time since
  the last relay price change and to the next (receipt clock); the live Binance spot price (last spot trade) and the futures mid against the
  last relay price, its way (bps), and the spot gap's change over 1 s.
3 Hyperliquid BTC (our P-2026-002 collector, receipt clock): mid return its way over 100 / 300 / 1000 ms, best-level imbalance its way and
  its change over 300 ms, far / near best size change over 300 ms (log ratio), 10-level imbalance its way, trades its way / against over
  300 ms against the point's own normal ((t - 30 s, t - 1 s], log ratio).
4 venues' prices: the futures mid (E clock) and the last trade price of Binance spot and the seven other venues of cross_venue (own
  exchange clocks), return its way over 100 / 300 / 1000 ms; Binance spot aggressive volume its way over 300 ms against normal. Lead test:
  points with no futures mid change in the previous 300 ms, split by a futures move its way within 300 ms after: the share whose venue
  price had already moved its way (>= 0.2 bps; Hyperliquid > 0) in the previous 300 ms, FIRE vs NEARBY.
5 Predict's own book (complete markets, the path loader; receipt clock, what a trader could have seen): best size and 5-level size on the
  side it takes and the other side, change over 300 ms (log ratio); mid change its way over 300 ms / 1 s (c); 5-level imbalance its way;
  spread; frames in the last 300 ms against normal; public fills its way / against seen in the last 300 ms / 1 s (trade time + 50 ms);
  an order of a top-2..10 taker in the same market and direction matched (settlement - 34 ms) 20-1000 ms before its arrival.
6 timing: arrival phase within the second (20 bins of 50 ms) and within 100 ms (20 bins of 5 ms) against uniform (null: 1,000 seeded draws
  of the same n, statistic max |bin share - 1/20|) and against ranks 2-10 orders' arrivals (Predict-side batching control; 1,000 label
  permutations); time left; the live spot price's distance from the window's start price (the relay price at the start second), its way;
  the share betting on the side ahead; FIRE vs RANDOM, and FIRE / RANDOM density by time left x |distance|.
9 clock: Predict's websocket heartbeat (server ms stamp) minus our receipt; websocket request -> reply round trip (our send logged, reply
  receipt); REST round trip; book frames' receipt minus their source stamp (step1 tables; the part file of a sealed hour is never opened).
  The Predict clock offset against ours is bounded by heartbeat lag and round trip; our clock against Binance's from futures bookTicker
  receipt minus event time.
Readings (merge): per feature AUC FIRE vs NEARBY, direction, medians, n; the lead table; relay (FIRE / NEARBY / RANDOM), timing, clock and
Predict-book readings as stated; counts and statuses. Hyperliquid's l2Book is sampled about every 5 s (bbo every ~150-200 ms): its
10-level imbalance exists only within 2 s of a snapshot (coverage reported). Point estimates, consumed days, its filled orders only.
Outputs (write-once): OUT/fire_info/points.parquet + points.json, <day>_feats.parquet, <day>_pbook.parquet, <day>_clock.json;
fire_info.json (merge).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/fire_info.py \
       selftest | points [<dir>] | feats <day> [<dir> <n>] | pbook <day> [<dir> <n>] | clock <day> [<dir>] | merge [<dir> <day> ...]
"""
import collections
import datetime as dt
import gzip
import json
import re
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import cross_venue as CV  # noqa: E402
import stale_quotes as SQ  # noqa: E402

C, TR, WT = CV.C, CV.TR, CV.WT
VR, MM, OD, F = SQ.VR, SQ.MM, SQ.OD, SQ.F
DATA = CV.DATA
OUT = WT.OUT / 'fire_info'
DAYS = SQ.DAYS
REF = DATA / 'predict_reference/runs/20261022T132901Z_8c043d17b6/reference_prices'
HLD = DATA / 'mm_hf/hl_raw'
S1 = DATA / 'predict_fun/research/step1'
BT = DATA / 'mm_hf/raw/bookTicker/BTCUSDT'
HOLD, LAG, BURST, NET_SEEN = 150., 34., 2., 50.
SHIFTS = (-30000., -25000., -20000., -15000., -10000., -5000., 5000., 10000., 15000., 20000., 25000., 30000.)
SEED, N_RANDOM, NULL_DRAWS, NBINS = 20261126, 20, 1000, 20
EPSQ = 1e-3
WINS = (100., 300., 1000.)
TAU_B, DIST_B = [0., 60., 120., 180., 240., 300.], [0., 2., 5., 10., 20., np.inf]
RX_RECV, RX_SENT = re.compile(r'"recv_ns":(\d+)'), re.compile(r'"sent":\{"recv_ns":(\d+)')
RX_HB = re.compile(r'\\?"data\\?":(\d+),\\?"topic\\?":\\?"heartbeat')
RX_REQ = re.compile(r'\\?"requestId\\?":(\d+)')


def iso_of_stem(stem):
    return f'{stem[:4]}-{stem[4:6]}-{stem[6:8]}'


def stems(day):
    """The day's 24 hour stems and the next day's 00Z when that day is allowed (never a sealed or reserved day)."""
    d0 = dt.datetime.strptime(day, '%Y%m%d')
    out = [f'{day}_{h:02d}' for h in range(24)]
    n = (d0 + dt.timedelta(days=1)).strftime('%Y%m%d')
    if CV.allowed(iso_of_stem(n)):
        out.append(f'{n}_00')
    for s in out:
        C.require(CV.allowed(iso_of_stem(s)), f'{s}: not an allowed day')
    return out


def at(ts, vs, t, fresh=np.inf):
    """Last value at or before t (vectorised); nan before the first or older than fresh."""
    j = np.searchsorted(ts, t, 'right') - 1
    ok = (j >= 0) & (t - ts[np.maximum(j, 0)] <= fresh) if len(ts) else np.zeros(len(t), bool)
    return np.where(ok, vs[np.maximum(j, 0)] if len(ts) else np.nan, np.nan)


def age(ts, t):
    j = np.searchsorted(ts, t, 'right') - 1
    return np.where(j >= 0, t - ts[np.maximum(j, 0)], np.nan) if len(ts) else np.full(len(t), np.nan)


def nxt(ts, t):
    j = np.searchsorted(ts, t, 'right')
    return np.where(j < len(ts), ts[np.minimum(j, len(ts) - 1)] - t, np.nan) if len(ts) else np.full(len(t), np.nan)


def wsum(ts, cum, a, b):
    """Sum over (a, b] given cum = r_[0, cumsum(values)] aligned with ts."""
    return cum[np.searchsorted(ts, b, 'right')] - cum[np.searchsorted(ts, a, 'right')]


def flow_ratio(ts, q, s, t, d, cover):
    """log((its-way volume over (t - 300, t] + eps) / (its normal over (t - 30 s, t - 1 s] scaled to 300 ms + eps)), and against."""
    cb, cs = np.r_[0., np.cumsum(q * (s > 0))], np.r_[0., np.cumsum(q * (s < 0))]
    out = {}
    for nm, sg in (('its', 1.), ('against', -1.)):
        buy = (d * sg) > 0
        w300 = np.where(buy, wsum(ts, cb, t - 300., t), wsum(ts, cs, t - 300., t))
        nrm = np.where(buy, wsum(ts, cb, t - 30000., t - 1000.), wsum(ts, cs, t - 30000., t - 1000.)) / 29000. * 300.
        r = np.log((w300 + EPSQ) / (nrm + EPSQ))
        out[nm] = np.where(cover, r, np.nan)
    return out


def next_move(bk, bm, t, d, w):
    """A futures mid change its way within (t, t + w]."""
    j0, j1 = np.searchsorted(bk, t, 'right'), np.searchsorted(bk, t + w, 'right')
    m0 = at(bk, bm, t)
    out = np.zeros(len(t), bool)
    for i in np.flatnonzero(j1 > j0):
        out[i] = bool(np.any(d[i] * (bm[j0[i]:j1[i]] - m0[i]) > 0))
    return out


# ---------------------------------------------------------------- points
def build_fires():
    L = pd.concat([pd.read_parquet(WT.OUT / f'exit_legs_{d}.parquet') for d in DAYS], ignore_index=True)
    L = L.sort_values('recv_ns', kind='stable').drop_duplicates(['tx', 'maker_hash'], keep='first')
    reg = pd.concat([MM.registry(d).reset_index().assign(day=d) for d in DAYS], ignore_index=True)
    reg['mid'] = reg.mid.astype(int)
    L = L[(L.role == 'taker') & L.mid.isin(set(reg.mid))]
    O = L.groupby('taker_hash').agg(mid=('mid', 'first'), t_set=('t_set', 'min'), side=('side', 'first'), sides=('side', 'nunique'))
    C.require((O.sides == 1).all(), 'a taker order with two sides')
    O['fire'] = O.t_set - HOLD
    O = O.sort_values(['mid', 'fire'], kind='stable').reset_index()
    O['burst'] = (~(O.groupby('mid').fire.diff() <= BURST)).cumsum()
    B = O.groupby('burst').agg(fire=('fire', 'min'), side=('side', 'first'), mid=('mid', 'first'), orders=('taker_hash', 'size')).reset_index(drop=True)
    B['t'] = (B.fire - LAG).round()
    ref = pd.read_csv(H / 'smart_fires/top_taker_fire_times_all.csv')
    C.require(len(ref) == len(B) and np.array_equal(np.sort(ref.t_ms.to_numpy()), np.sort(B.t.to_numpy().astype(np.int64))), 'the fires do not reproduce the exported list')
    R = reg.drop_duplicates('mid').set_index('mid')
    B['day'] = B.mid.map(R.day)
    B['ws_ms'] = B.mid.map(R.ws).astype(float) * 1000.
    B['d'] = np.where(B.side > 0, 1, -1)
    return B


def points(od=None):
    od = Path(od) if od else OUT
    fp, fj = od / 'points.parquet', od / 'points.json'
    C.require(not fp.exists() and not fj.exists(), 'points exist (write-once)')
    B = build_fires()
    inw = (B.t >= B.ws_ms) & (B.t < B.ws_ms + 300000.)
    Fz = B[inw].reset_index(drop=True)
    Fz['pid'] = np.arange(len(Fz))
    rng = np.random.default_rng(SEED)
    base = dict(pid=Fz.pid.to_numpy(), day=Fz.day.to_numpy(), mid=Fz.mid.to_numpy(), ws_ms=Fz.ws_ms.to_numpy(), d=Fz.d.to_numpy())
    parts = [pd.DataFrame(dict(kind='FIRE', t=Fz.t.to_numpy(float), shift_ms=0., **base))]
    for s in SHIFTS:
        u = rng.uniform(-500., 500., len(Fz))
        parts.append(pd.DataFrame(dict(kind='NEARBY', t=Fz.t.to_numpy(float) + s + u, shift_ms=s + u, **base)))
    P = pd.concat(parts, ignore_index=True)
    near = (P.kind == 'NEARBY').to_numpy()
    tt, mm = P.t.to_numpy(float), P.mid.to_numpy()
    allf = {m: np.sort(g.t.to_numpy(float)) for m, g in B.groupby('mid')}
    dmin = np.full(len(P), np.inf)
    for m in np.unique(mm[near]):
        ix = np.flatnonzero(near & (mm == m)); a = allf[m]
        j = np.searchsorted(a, tt[ix])
        dmin[ix] = np.minimum(np.abs(tt[ix] - a[np.clip(j - 1, 0, len(a) - 1)]), np.abs(a[np.clip(j, 0, len(a) - 1)] - tt[ix]))
    inwin = (P.t >= P.ws_ms) & (P.t < P.ws_ms + 300000.)
    keep = ~near | (inwin.to_numpy() & (dmin >= 1000.))
    n_near_all = int(near.sum())
    P = P[keep].reset_index(drop=True)
    mk = Fz.drop_duplicates('mid')[['day', 'mid', 'ws_ms']]
    k = len(mk) * N_RANDOM
    Rr = pd.DataFrame(dict(kind='RANDOM', pid=-1, day=np.repeat(mk.day.to_numpy(), N_RANDOM), mid=np.repeat(mk.mid.to_numpy(), N_RANDOM),
                           ws_ms=np.repeat(mk.ws_ms.to_numpy(), N_RANDOM)))
    Rr['t'] = Rr.ws_ms.to_numpy() + rng.uniform(0., 300000., k)
    Rr['d'] = rng.choice(np.array([-1, 1]), k)
    Rr['shift_ms'] = np.nan
    P = pd.concat([P, Rr[P.columns]], ignore_index=True)
    od.mkdir(parents=True, exist_ok=True)
    P.to_parquet(fp, index=False)
    out = dict(asof='2026-11-26', fires=int(len(B)), fires_in_window=int(inw.sum()), fires_before_window=int((B.t < B.ws_ms).sum()),
               fires_after_window=int((B.t >= B.ws_ms + 300000.).sum()), nearby_drawn=n_near_all, nearby_kept=int((P.kind == 'NEARBY').sum()),
               random=int((P.kind == 'RANDOM').sum()), markets=int(len(mk)), by_day={k_: int(v) for k_, v in Fz.day.value_counts().sort_index().items()})
    TR.dump(fj, out)
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------- loaders (day rule applied per file, before it opens)
def load_relay(day):
    rows = []
    for s in stems(day):
        p = REF / f'{s}.jsonl.gz'
        if not p.exists():
            continue
        with gzip.open(p, 'rt') as f:
            for line in f:
                if '"kind":"PRICE"' not in line or '"duplicate_previous_payload":true' in line:
                    continue
                r = json.loads(line)
                rows.append((r['recv_ns'] / 1e6, float(r['price']), float(r['publish_time_s'])))
    A = np.array(sorted(rows), float).reshape(-1, 3)
    return A[:, 0], A[:, 1], A[:, 2]


def load_hl(day, kind):
    parts = []
    for s in stems(day):
        p = HLD / kind / 'BTC' / f'{s}.csv.gz'
        if p.exists():
            if kind == 'bbo':
                parts.append(pd.read_csv(p, header=None, names=['recv', 'hl', 'bp', 'bs', 'ap', 'as_']))
            else:
                parts.append(pd.read_csv(p, header=None, names=['recv', 'hl', 'tid', 'px', 'sz', 'side']))
    D = pd.concat(parts, ignore_index=True)
    D['t'] = D.recv / 1e6
    return D.sort_values('t', kind='stable').reset_index(drop=True)


def hl_depth_imb(day, t, d):
    out = np.full(len(t), np.nan)
    for s in stems(day):
        p = HLD / 'l2Book' / 'BTC' / f'{s}.csv.gz'
        if not p.exists():
            continue
        df = pd.read_csv(p, header=None, names=['recv', 'hl', 'bids', 'asks'], dtype={'bids': str, 'asks': str})
        r = df.recv.to_numpy(np.int64) / 1e6
        if not len(r):
            continue
        sel = np.flatnonzero((t >= r[0]) & (t <= r[-1] + 2000.))
        j = np.searchsorted(r, t[sel], 'right') - 1
        ok = (j >= 0) & (t[sel] - r[np.maximum(j, 0)] <= 2000.)
        for i, jj in zip(sel[ok], j[ok]):
            try:
                b = sum(float(x.split('@')[1]) for x in df.bids.iat[jj].split('|'))
                a = sum(float(x.split('@')[1]) for x in df.asks.iat[jj].split('|'))
            except (AttributeError, IndexError, ValueError):
                continue
            out[i] = d[i] * (b - a) / (b + a) if b + a > 0 else np.nan
    return out


def start_prices(rpub, rp, ws_s):
    """The relay price published at the window's start second (else the last within 5 s before)."""
    o = np.argsort(rpub, kind='stable'); pub, pr = rpub[o], rp[o]
    j = np.searchsorted(pub, ws_s, 'right') - 1
    ok = (j >= 0) & (ws_s - pub[np.maximum(j, 0)] <= 5.)
    exact = np.searchsorted(pub, ws_s, 'left')
    ex = (exact < len(pub)) & (pub[np.minimum(exact, len(pub) - 1)] == ws_s)
    return np.where(ex, pr[np.minimum(exact, len(pub) - 1)], np.where(ok, pr[np.maximum(j, 0)], np.nan))


# ---------------------------------------------------------------- items 2, 3, 4, 6 features
def feats(day, od=None, limit=None):
    resource.setrlimit(resource.RLIMIT_CPU, (1500, 1501))
    C.require(day in DAYS, f'{day} is not a declared day')
    od = Path(od) if od else OUT
    fp = od / f'{day}_feats.parquet'
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    P = pd.read_parquet(OUT / 'points.parquet')
    P = P[P.day == day].reset_index(drop=True)
    if limit:
        P = P.iloc[np.sort(np.random.default_rng(1).choice(len(P), min(limit, len(P)), replace=False))].reset_index(drop=True)
    t, d = P.t.to_numpy(float), P.d.to_numpy(float)
    X = {}
    bk, bm = OD.seg_arrays(OD.SEG_OF[day]); bk = bk.astype(float); bm = bm.astype(float)
    fm = at(bk, bm, t)
    for w in WINS:
        X[f'fut_ret_{int(w)}'] = d * 1e4 * np.log(fm / at(bk, bm, t - w))
    X['fut_quiet_300'] = (np.searchsorted(bk, t, 'right') - np.searchsorted(bk, t - 300., 'right')) == 0
    X['y300'] = next_move(bk, bm, t, d, 300.)
    # venues' last trade prices (own exchange clocks)
    spot_last = None
    for v in CV.OTHER:
        ts, p_, q_, s_, (lo, hi), _ = CV.load(v, day)
        cover = (t - 30000. >= lo) & (t < hi)
        last = np.where(cover, at(ts, p_, t, 5000.), np.nan)
        for w in WINS:
            X[f'{v}_ret_{int(w)}'] = d * 1e4 * np.log(last / np.where(cover, at(ts, p_, t - w, 5000.), np.nan))
        if v == 'binance_spot':
            spot_last = last
            fr = flow_ratio(ts, q_, s_, t, d, cover)
            X['binance_spot_flow_its_r300'], X['binance_spot_flow_against_r300'] = fr['its'], fr['against']
            spot_prev1s = np.where(cover, at(ts, p_, t - 1000., 5000.), np.nan)
    # reference relay
    rr, rp, rpub = load_relay(day)
    chg = np.r_[True, np.diff(rp) != 0]
    rc = rr[chg]
    X['relay_age_ms'] = age(rc, t)
    X['relay_next_ms'] = nxt(rc, t)
    rel_last = at(rr, rp, t, 5000.)
    X['relay_gap_spot_bps'] = d * 1e4 * np.log(spot_last / rel_last)
    X['relay_gap_fut_bps'] = d * 1e4 * np.log(fm / rel_last)
    X['relay_gap_spot_d1s'] = X['relay_gap_spot_bps'] - d * 1e4 * np.log(spot_prev1s / at(rr, rp, t - 1000., 5000.))
    sp0 = start_prices(rpub, rp, np.floor(P.ws_ms.to_numpy() / 1000.))
    X['start_found'] = np.isfinite(sp0)
    X['dist_its_bps'] = d * 1e4 * np.log(spot_last / sp0)
    X['tau_s'] = (P.ws_ms.to_numpy() + 300000. - t) / 1000.
    X['phase1000'] = np.mod(t, 1000.)
    X['phase100'] = np.mod(t, 100.)
    # Hyperliquid
    hb = load_hl(day, 'bbo')
    ht = hb.t.to_numpy(float); hm = ((hb.bp + hb.ap) / 2.).to_numpy(float)
    bs_, as_ = hb.bs.to_numpy(float), hb.as_.to_numpy(float)
    m_t = at(ht, hm, t, 2000.)
    for w in WINS:
        X[f'hl_ret_{int(w)}'] = d * 1e4 * np.log(m_t / at(ht, hm, t - w, 2000.))
    b_t, a_t, b_3, a_3 = at(ht, bs_, t, 2000.), at(ht, as_, t, 2000.), at(ht, bs_, t - 300., 2000.), at(ht, as_, t - 300., 2000.)
    imb_t, imb_3 = d * (b_t - a_t) / (b_t + a_t), d * (b_3 - a_3) / (b_3 + a_3)
    X['hl_imb1'], X['hl_imb1_d300'] = imb_t, imb_t - imb_3
    far_t, far_3 = np.where(d > 0, a_t, b_t), np.where(d > 0, a_3, b_3)
    near_t, near_3 = np.where(d > 0, b_t, a_t), np.where(d > 0, b_3, a_3)
    X['hl_far_d300'], X['hl_near_d300'] = np.log(far_t / far_3), np.log(near_t / near_3)
    X['hl_imb10'] = hl_depth_imb(day, t, d)
    htr = load_hl(day, 'trades')
    hcover = np.isfinite(at(ht, hm, t - 30000., 5000.)) & np.isfinite(m_t)
    fr = flow_ratio(htr.t.to_numpy(float), htr.sz.to_numpy(float), np.where(htr.side.astype(str) == 'B', 1., -1.), t, d, hcover)
    X['hl_flow_its_r300'], X['hl_flow_against_r300'] = fr['its'], fr['against']
    Xd = pd.DataFrame(X)
    Z = pd.concat([P, Xd], axis=1)
    od.mkdir(parents=True, exist_ok=True)
    Z.to_parquet(fp, index=False)
    print(day, len(Z), {k: float(np.nanmedian(v)) if np.issubdtype(np.asarray(v).dtype, np.number) else None for k, v in list(X.items())[:6]}, flush=True)


# ---------------------------------------------------------------- item 5: Predict's own book
def pbook(day, od=None, limit=None):
    resource.setrlimit(resource.RLIMIT_CPU, (1200, 1201))
    C.require(day in DAYS, f'{day} is not a declared day')
    od = Path(od) if od else OUT
    fp = od / f'{day}_pbook.parquet'
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    P = pd.read_parquet(OUT / 'points.parquet')
    vs = json.loads((VR.OUT / 'summary.json').read_text())
    comp = {int(r['mid']) for r in vs['rows'] if r['day'] == day and r['status'] == 'complete'}
    P = P[(P.day == day) & P.kind.isin(['FIRE', 'NEARBY']) & P.mid.isin(comp)].reset_index(drop=True)
    Ok = pd.read_parquet(WT.OUT / 'fire_times_orders.parquet', columns=['day', 'rank', 't_set', 'd', 'mid'])
    Ok = Ok[(Ok.day == day) & (Ok['rank'] > 1) & Ok.t_set.notna()]
    smart = {m: (g.t_set.to_numpy(float) - LAG, g.d.to_numpy(float)) for m, g in Ok.groupby('mid')}
    by_mid = P.groupby('mid').indices
    cols = ['pb_far1_d300', 'pb_near1_d300', 'pb_far5_d300', 'pb_near5_d300', 'pb_mid_ret_300', 'pb_mid_ret_1000', 'pb_imb5', 'pb_spread',
            'pb_frames_r300', 'pb_fills_its_300', 'pb_fills_against_300', 'pb_fills_its_1000', 'pb_smart_before']
    R = {c: np.full(len(P), np.nan) for c in cols}
    src = MM.KP.Tape()
    reg = MM.registry(day).set_index('mid'); off = src.offset(day)
    n_m = 0
    for h, g in reg.groupby('hour'):
        mids = [int(m) for m in g.index if int(m) in comp and int(m) in by_mid]
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
            f = np.asarray(p['frames'], float); tr = np.asarray(p['trades'], float).reshape(-1, 4)
            ix = by_mid[m]
            tl = P.t.to_numpy(float)[ix] - t0; dd = P.d.to_numpy(float)[ix]
            ok = (tl >= 1000.) & (tl <= end)
            kf = f[:, 1]

            def fr_at(x):
                return np.searchsorted(kf, x, 'right') - 1
            j, j3, j10 = fr_at(tl), fr_at(tl - 300.), fr_at(tl - 1000.)
            ok &= (j3 >= 0) & (j10 >= 0)
            jj, jj3, jj10 = np.maximum(j, 0), np.maximum(j3, 0), np.maximum(j10, 0)
            bp, bsz, ap, asz = f[:, 2:7], f[:, 7:12], f[:, 12:17], f[:, 17:22]
            bsz = np.where(bp >= 0, bsz, 0.); asz = np.where(ap >= 0, asz, 0.)
            b1, a1, b5, a5 = bsz[:, 0], asz[:, 0], bsz.sum(1), asz.sum(1)
            far1 = lambda k: np.where(dd > 0, a1[k], b1[k]); near1 = lambda k: np.where(dd > 0, b1[k], a1[k])
            far5 = lambda k: np.where(dd > 0, a5[k], b5[k]); near5 = lambda k: np.where(dd > 0, b5[k], a5[k])
            mid = np.where((bp[:, 0] >= 0) & (ap[:, 0] > bp[:, 0]), (bp[:, 0] + ap[:, 0]) / 2., np.nan)
            vals = dict(pb_far1_d300=np.log((far1(jj) + 1.) / (far1(jj3) + 1.)), pb_near1_d300=np.log((near1(jj) + 1.) / (near1(jj3) + 1.)),
                        pb_far5_d300=np.log((far5(jj) + 1.) / (far5(jj3) + 1.)), pb_near5_d300=np.log((near5(jj) + 1.) / (near5(jj3) + 1.)),
                        pb_mid_ret_300=dd * (mid[jj] - mid[jj3]), pb_mid_ret_1000=dd * (mid[jj] - mid[jj10]),
                        pb_imb5=dd * (b5[jj] - a5[jj]) / np.maximum(b5[jj] + a5[jj], 1e-9), pb_spread=ap[jj, 0] - bp[jj, 0])
            n3 = np.searchsorted(kf, tl, 'right') - np.searchsorted(kf, tl - 300., 'right')
            nn = (np.searchsorted(kf, tl - 1000., 'right') - np.searchsorted(kf, tl - 30000., 'right')) / 29000. * 300.
            vals['pb_frames_r300'] = np.log((n3 + 1.) / (nn + 1.))
            seen = tr[:, 0] + NET_SEEN
            o = np.argsort(seen, kind='stable'); sn, ts_, tq = seen[o], tr[o, 1], tr[o, 3]
            c1, c0 = np.r_[0., np.cumsum(tq * (ts_ == 1))], np.r_[0., np.cumsum(tq * (ts_ == 0))]
            for w, nm in ((300., '300'), (1000., '1000')):
                v1, v0 = wsum(sn, c1, tl - w, tl), wsum(sn, c0, tl - w, tl)
                vals[f'pb_fills_its_{nm}'] = np.where(dd > 0, v1, v0)
                if nm == '300':
                    vals['pb_fills_against_300'] = np.where(dd > 0, v0, v1)
            sm = np.zeros(len(ix))
            if m in smart:
                st_, sd_ = smart[m]
                st_ = st_ - t0
                for k_ in range(len(ix)):
                    sm[k_] = float(np.any((st_ >= tl[k_] - 1000.) & (st_ <= tl[k_] - 20.) & (sd_ == dd[k_])))
            vals['pb_smart_before'] = sm
            for c in cols:
                R[c][ix] = np.where(ok, vals[c], np.nan)
            C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    Z = pd.concat([P, pd.DataFrame(R)], axis=1)
    od.mkdir(parents=True, exist_ok=True)
    Z.to_parquet(fp, index=False)
    print(day, n_m, len(Z), flush=True)


# ---------------------------------------------------------------- item 9: clocks
def clock(day, od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (1200, 1201))
    C.require(day in DAYS, f'{day} is not a declared day')
    od = Path(od) if od else OUT
    fj = od / f'{day}_clock.json'
    C.require(not fj.exists(), f'{fj.name} exists (write-once)')
    run = Path(SQ.XT.IO.PF_RUN)
    hb, rtt_ws, rtt_rest, book = [], [], [], []
    sent = {}
    st_day = [f'{day}_{h:02d}' for h in range(24)]
    for s in st_day:
        C.require(CV.allowed(iso_of_stem(s)), s)
        pe = run / 'events' / f'{s}.jsonl.gz'
        if pe.exists():
            with gzip.open(pe, 'rt') as f:
                for line in f:
                    if '"WS_REQUEST"' in line:
                        r = json.loads(line)
                        sent[(r.get('connection_id'), int(r['request_id']))] = r['recv_ns'] / 1e6
        pw = run / 'predict_ws' / f'{s}.jsonl.gz'
        if pw.exists():
            with gzip.open(pw, 'rt') as f:
                for line in f:
                    if 'heartbeat' in line[:400] or '"type\\":\\"R\\"' in line[:400] or 'requestId' in line[:400]:
                        r = json.loads(line)
                        raw = r.get('raw') or ''
                        rv = r['recv_ns'] / 1e6
                        if '"topic":"heartbeat"' in raw:
                            hb.append(rv - float(json.loads(raw)['data']))
                        elif '"requestId"' in raw:
                            k = (r.get('connection_id'), int(json.loads(raw)['requestId']))
                            if k in sent:
                                rtt_ws.append(rv - sent[k])
        prr = run / 'rest' / f'{s}.jsonl.gz'
        if prr.exists():
            with gzip.open(prr, 'rt') as f:
                for line in f:
                    head = line[:600]
                    a, b = RX_RECV.search(head), RX_SENT.search(head)
                    if a and b:
                        rtt_rest.append((int(a.group(1)) - int(b.group(1))) / 1e6)
        pb = S1 / day / 'books' / f'part_{s}.parquet'
        if pb.exists():
            x = pd.read_parquet(pb, columns=['t_ms', 'recv_ns'])
            book.append(x.recv_ns.to_numpy(np.int64) / 1e6 - x.t_ms.to_numpy(float))
    bn = []
    for s in (f'{day}_06', f'{day}_12', f'{day}_18'):
        p = BT / f'{s}.csv.gz'
        if p.exists():
            x = pd.read_csv(p, header=None, names=['recv', 'E', 'T', 'u', 'bid', 'bq', 'ask', 'aq'], usecols=['recv', 'E'], on_bad_lines='skip')
            x = x.apply(pd.to_numeric, errors='coerce').dropna()
            bn.append(x.recv.to_numpy(np.int64) / 1e6 - x.E.to_numpy(float))

    def qd(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        return dict(n=int(len(v)), min=float(v.min()), p001=float(np.quantile(v, .001)), p01=float(np.quantile(v, .01)), p05=float(np.quantile(v, .05)),
                    p50=float(np.quantile(v, .5)), p95=float(np.quantile(v, .95))) if len(v) else dict(n=0)
    out = dict(asof='2026-11-26', day=day, heartbeat_lag_ms=qd(hb), ws_request_rtt_ms=qd(rtt_ws), rest_rtt_ms=qd(rtt_rest),
               book_frame_lag_ms=qd(np.concatenate(book)) if book else dict(n=0), binance_recv_minus_E_ms=qd(np.concatenate(bn)) if bn else dict(n=0),
               ws_requests_logged=len(sent), book_part_files=len(book))
    od.mkdir(parents=True, exist_ok=True)
    TR.dump(fj, out)
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------- merge
def auc_rows(Fz, N, cols):
    rows = []
    for c in cols:
        a, b = pd.to_numeric(Fz[c], errors='coerce').to_numpy(float), pd.to_numeric(N[c], errors='coerce').to_numpy(float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 30 or len(b) < 30 or np.r_[a, b].std() == 0:
            rows.append(dict(feature=c, n_fires=int(len(a)), n_nearby=int(len(b)), auc=None))
            continue
        au = float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b]))
        rows.append(dict(feature=c, auc=au, separation=max(au, 1. - au), direction='higher at fires' if au >= .5 else 'lower at fires',
                         fires_median=float(np.median(a)), nearby_median=float(np.median(b)), n_fires=int(len(a)), n_nearby=int(len(b))))
    return sorted(rows, key=lambda r: -(r['separation'] if r.get('auc') is not None else 0.))


def phase_test(ph, width, rng):
    n = len(ph)
    h = np.histogram(ph, bins=NBINS, range=(0., width))[0] / n
    stat = float(np.max(np.abs(h - 1. / NBINS)))
    null = np.array([np.max(np.abs(np.histogram(rng.uniform(0., width, n), bins=NBINS, range=(0., width))[0] / n - 1. / NBINS)) for _ in range(NULL_DRAWS)])
    return dict(n=int(n), max_dev=stat, null_p95=float(np.quantile(null, .95)), p_value=float((1 + (null >= stat).sum()) / (1 + NULL_DRAWS)),
                shares=[float(x) for x in h])


def merge(od=None, days=None):
    days = days or DAYS
    fn = (Path(od) if od else H) / 'fire_info.json'
    C.require(not fn.exists(), 'fire_info.json exists (write-once)')
    src = Path(od) if od else OUT
    Z = pd.concat([pd.read_parquet(src / f'{d}_feats.parquet') for d in days], ignore_index=True)
    PBK = pd.concat([pd.read_parquet(src / f'{d}_pbook.parquet') for d in days if (src / f'{d}_pbook.parquet').exists()], ignore_index=True)
    CK = {d: json.loads((src / f'{d}_clock.json').read_text()) for d in days if (src / f'{d}_clock.json').exists()}
    pts = json.loads((OUT / 'points.json').read_text())
    rng = np.random.default_rng(SEED + 1)
    Fz, N, Rn = Z[Z.kind == 'FIRE'], Z[Z.kind == 'NEARBY'], Z[Z.kind == 'RANDOM']
    num = [c for c in Z.columns if c not in ('kind', 'pid', 'day', 'mid', 'ws_ms', 't', 'd', 'shift_ms', 'fut_quiet_300', 'y300', 'start_found', 'phase1000', 'phase100')]
    out = dict(asof='2026-11-26', days=list(days), points=pts, n=dict(fires=int(len(Fz)), nearby=int(len(N)), random=int(len(Rn))))
    out['auc'] = auc_rows(Fz, N, num)
    lead = {}
    for nm, col, thr in [(v, f'{v}_ret_300', .2) for v in CV.OTHER] + [('hyperliquid', 'hl_ret_300', 0.)]:
        r = {}
        for lab, X in (('fires', Fz), ('nearby', N)):
            q = X[X.fut_quiet_300.astype(bool)]
            for ylab, Y in (('move_after', q[q.y300.astype(bool)]), ('no_move_after', q[~q.y300.astype(bool)])):
                v = pd.to_numeric(Y[col], errors='coerce')
                r[f'{lab}_{ylab}'] = dict(n=int(v.notna().sum()), moved_its_way=float((v[v.notna()] > thr).mean()) if v.notna().any() else None,
                                          moved_against=float((v[v.notna()] < -thr).mean()) if v.notna().any() else None)
        lead[nm] = r
    out['lead'] = lead
    for lab, X in (('fires', Fz), ('nearby', N)):
        q = X[X.fut_quiet_300.astype(bool)]
        out.setdefault('quiet_futures', {})[lab] = dict(n=int(len(q)), share_of_points=float(len(q) / len(X)), move_after=float(q.y300.astype(bool).mean()))
    rel = {}
    for lab, X in (('fires', Fz), ('nearby', N), ('random', Rn)):
        a = pd.to_numeric(X.relay_age_ms, errors='coerce'); g = pd.to_numeric(X.relay_gap_spot_bps, errors='coerce')
        rel[lab] = dict(age_ms=[float(x) for x in a.quantile([.1, .25, .5, .75, .9])], next_ms=[float(x) for x in pd.to_numeric(X.relay_next_ms, errors='coerce').quantile([.1, .25, .5, .75, .9])],
                        gap_spot_bps=[float(x) for x in g.quantile([.1, .25, .5, .75, .9])], gap_its_ge1=float((g[g.notna()] >= 1.).mean()), gap_against_ge1=float((g[g.notna()] <= -1.).mean()))
    out['relay'] = rel
    O = pd.read_parquet(WT.OUT / 'fire_times_orders.parquet', columns=['day', 'rank', 't_set'])
    O = O[O.day.isin(days) & (O['rank'] > 1) & O.t_set.notna()]
    ra = O.t_set.to_numpy(float) - HOLD - LAG
    fa = Fz.t.to_numpy(float)
    ph = {}
    for width, nm in ((1000., 'second'), (100., '100ms')):
        pf, pr = np.mod(fa, width), np.mod(ra, width)
        ph[nm] = dict(top=phase_test(pf, width, rng), ranks2_10=phase_test(pr, width, rng))
        hf, hr = np.histogram(pf, NBINS, (0., width))[0] / len(pf), np.histogram(pr, NBINS, (0., width))[0] / len(pr)
        stat = float(np.max(np.abs(hf - hr)))
        allp, nf = np.r_[pf, pr], len(pf)
        null = []
        for _ in range(NULL_DRAWS):
            o = rng.permutation(len(allp))
            null.append(np.max(np.abs(np.histogram(allp[o[:nf]], NBINS, (0., width))[0] / nf - np.histogram(allp[o[nf:]], NBINS, (0., width))[0] / (len(allp) - nf))))
        ph[nm]['top_vs_ranks'] = dict(max_dev=stat, p_value=float((1 + (np.array(null) >= stat).sum()) / (1 + NULL_DRAWS)))
    out['phase'] = ph
    sel = {}
    for lab, X in (('fires', Fz), ('random', Rn)):
        tau = pd.to_numeric(X.tau_s, errors='coerce'); dist = pd.to_numeric(X.dist_its_bps, errors='coerce')
        sel[lab] = dict(n=int(len(X)), tau_s=[float(x) for x in tau.quantile([.1, .25, .5, .75, .9])], last_60s=float((tau <= 60.).mean()), first_60s=float((tau > 240.).mean()),
                        ahead=float((dist[dist.notna()] > 0).mean()), abs_dist_bps=[float(x) for x in dist.abs().quantile([.1, .25, .5, .75, .9])],
                        start_found=float(X.start_found.astype(bool).mean()))
    tb = lambda X: pd.cut(pd.to_numeric(X.tau_s, errors='coerce'), TAU_B, include_lowest=True)
    db = lambda X: pd.cut(pd.to_numeric(X.dist_its_bps, errors='coerce').abs(), DIST_B, include_lowest=True)
    cf = pd.crosstab(tb(Fz), db(Fz), normalize=True); cr = pd.crosstab(tb(Rn), db(Rn), normalize=True)
    sel['density_ratio'] = {f'tau {i}': {f'|dist| {j}': (float(cf.loc[i, j] / cr.loc[i, j]) if (i in cf.index and j in cf.columns and cr.loc[i, j] > 0) else None)
                                         for j in cr.columns} for i in cr.index}
    sel['fires_share'] = {f'tau {i}': {f'|dist| {j}': float(cf.loc[i, j]) if (i in cf.index and j in cf.columns) else 0. for j in cr.columns} for i in cr.index}
    out['selection'] = sel
    if len(PBK):
        pcols = [c for c in PBK.columns if c.startswith('pb_')]
        PF, PN = PBK[PBK.kind == 'FIRE'], PBK[PBK.kind == 'NEARBY']
        out['pbook'] = dict(n_fires=int(len(PF)), n_nearby=int(len(PN)), auc=auc_rows(PF, PN, [c for c in pcols if c != 'pb_smart_before']),
                            smart_before=dict(fires=float(pd.to_numeric(PF.pb_smart_before, errors='coerce').mean()), nearby=float(pd.to_numeric(PN.pb_smart_before, errors='coerce').mean())),
                            fills_its_300_any=dict(fires=float((pd.to_numeric(PF.pb_fills_its_300, errors='coerce') > 0).mean()), nearby=float((pd.to_numeric(PN.pb_fills_its_300, errors='coerce') > 0).mean())))
    if CK:
        def pool(key, stat):
            v = [CK[d][key][stat] for d in CK if CK[d][key].get('n')]
            return (min(v) if stat in ('min', 'p001', 'p01') else float(np.median(v))) if v else None
        out['clock'] = dict(by_day=CK, heartbeat_lag_min=pool('heartbeat_lag_ms', 'min'), heartbeat_lag_p50=pool('heartbeat_lag_ms', 'p50'),
                            ws_rtt_min=pool('ws_request_rtt_ms', 'min'), ws_rtt_p50=pool('ws_request_rtt_ms', 'p50'), rest_rtt_min=pool('rest_rtt_ms', 'min'),
                            book_lag_min=pool('book_frame_lag_ms', 'min'), book_lag_p50=pool('book_frame_lag_ms', 'p50'),
                            binance_recv_minus_E_min=pool('binance_recv_minus_E_ms', 'min'), binance_recv_minus_E_p50=pool('binance_recv_minus_E_ms', 'p50'))
        c = out['clock']
        if c['heartbeat_lag_min'] is not None and c['ws_rtt_min'] is not None:
            c['predict_offset_bounds_ms'] = [-c['heartbeat_lag_min'], c['ws_rtt_min'] - c['heartbeat_lag_min']]
            c['predict_offset_symmetric_estimate_ms'] = c['ws_rtt_min'] / 2. - c['heartbeat_lag_min']
    TR.dump(fn, out)
    print(json.dumps({k: out[k] for k in ('n', 'quiet_futures')}, indent=1))


def selftest():
    ts = np.array([0., 10., 20., 30.]); vs = np.array([1., 2., 3., 4.])
    assert np.array_equal(np.nan_to_num(at(ts, vs, np.array([-1., 0., 16., 31.]), fresh=5.), nan=-9.), np.array([-9., 1., -9., 4.]))
    assert np.allclose(np.nan_to_num(age(ts, np.array([-1., 12., 30.])), nan=-9.), [-9., 2., 0.])
    assert np.allclose(np.nan_to_num(nxt(ts, np.array([5., 30.])), nan=-9.), [5., -9.])
    cum = np.r_[0., np.cumsum(vs)]
    assert np.allclose(wsum(ts, cum, np.array([0., 5.]), np.array([20., 30.])), [5., 9.])          # (0, 20] = 2 + 3; (5, 30] = 2 + 3 + 4
    bk = np.array([0., 100., 200., 400.]); bm = np.array([10., 11., 10., 12.])
    assert next_move(bk, bm, np.array([50., 50., 150., 250.]), np.array([1., -1., 1., 1.]), 300.).tolist() == [True, False, True, True]
    tq = np.array([500., 1000., 29800., 29900.]); qq = np.array([1., 1., 2., 3.]); ss = np.array([1., -1., 1., -1.])
    fr = flow_ratio(tq, qq, ss, np.array([30000.]), np.array([1.]), np.array([True]))
    assert abs(fr['its'][0] - np.log((2. + EPSQ) / (1. / 29000. * 300. + EPSQ))) < 1e-12 and abs(fr['against'][0] - np.log((3. + EPSQ) / (1. / 29000. * 300. + EPSQ))) < 1e-12
    rpub = np.array([99., 100., 100., 101.]); rp = np.array([1., 2., 3., 4.])
    assert np.allclose(np.nan_to_num(start_prices(rpub, rp, np.array([100., 102., 200.])), nan=-9.), [2., 4., -9.])
    rng = np.random.default_rng(0)
    pt = phase_test(rng.uniform(0., 1000., 5000), 1000., rng)
    assert pt['p_value'] > .01, pt['p_value']
    pk = phase_test(np.r_[rng.uniform(0., 1000., 4000), rng.uniform(300., 350., 1000)], 1000., rng)
    assert pk['p_value'] < .01, pk['p_value']                                                  # falsifier: a clustered phase must be flagged
    assert stems('20261102')[-1] == '20261103_00' and stems('20261031')[0] == '20261031_00' and len(stems('20261031')) == 25
    hb = '{"data":1793613609220,"topic":"heartbeat","type":"M"}'
    assert '"topic":"heartbeat"' in hb and float(json.loads(hb)['data']) == 1793613609220.
    line = '{"run_id":"x","seq":1,"source":"rest","kind":"response","recv_ns":2000000000,"recv_mono_ns":5,"path":"/v1/orders/matches","params":{},"sent":{"recv_ns":1000000000,"recv_mono_ns":4},"status":200}'
    assert (int(RX_RECV.search(line).group(1)) - int(RX_SENT.search(line).group(1))) / 1e6 == 1000.
    print('fire_info selftest PASS')


if __name__ == '__main__':
    cmd = sys.argv[1]
    a = sys.argv[2:]
    if cmd == 'points':
        points(a[0] if a else None)
    elif cmd == 'feats':
        feats(a[0], a[1] if len(a) > 1 else None, int(a[2]) if len(a) > 2 else None)
    elif cmd == 'pbook':
        pbook(a[0], a[1] if len(a) > 1 else None, int(a[2]) if len(a) > 2 else None)
    elif cmd == 'clock':
        clock(a[0], a[1] if len(a) > 1 else None)
    elif cmd == 'merge':
        merge(a[0] if a else None, tuple(a[1:]) or None)
    else:
        dict(selftest=selftest)[cmd]()
