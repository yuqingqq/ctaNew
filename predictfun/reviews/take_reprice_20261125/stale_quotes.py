"""Do our fires catch the makers' stale quotes, as the smart takers do? (USER 2026-11-26: "In current fires, do we catch the stale quotes
of makers, it looks smart takers can take those stale quotes, and their mark out prices are much better"). Readout on the four consumed
days over score_replay's saved records and the recorded books; no new replay, no strategy change.
A quote is stale when its price stays on the Predict book while Binance moves the taker's way. Per quote hit (or aimed at) at time t on
book side b at price L (UP space, cents; d = +1 when the taker buys Up): its appearance = the first frame of the run of consecutive frames
showing size at L on side b that contains the frame 25 ms before t (frame clock; 'not_shown' when that frame does not show it: posted
within 25 ms, beyond the 5 shown levels, or the clock; counted and kept in the value tables; 'age_censored' when the run starts at the
path's first frame); staleness (cents) = d x 100 x (Phi(z(t)) - Phi(z(appearance))), z(x) = 1e4 ln(bm(x) / bm(window start)) /
(sigma sqrt(tau)), sigma (300 s ending 250 ms before t) and tau taken at t: reprice_replay's dFV between two instants
(100 x (Phi(z(t)) - Phi(z(t - 250))) = why_markout.dfv_at(t), checked on every row; falsifier: t - 500 must differ); split additively at
the taker's arrival at Predict (t - 150 ms, the speed bump) into stale on arrival and gone stale during the hold; the Binance mid move its
way (bps) over the same span.
Groups (declared before the run):
- fills: TOP = the top taker's taker legs (exit_times' scan, de-duplicated by transaction + maker order) in the day's complete markets, at
  t_set - 34 ms; RANKS2_10 = the legs of the takers ranked 2-10 (winners_legs, not ambiguous) at their order's t_set - 34 ms
  (fire_times_orders); OTHERS = the markets' public trades (M50 clock) not within 100 ms of a TOP / RANKS2_10 leg on the same side and
  price; SCORE1 / RAND1 = our filled takes in score_replay, at act. Per fill the value per share 1 s / 10 s later (why_markout.would_be:
  the take-marks mid, after fee); our fills must reproduce the engine's take marks (falsifier: the mid read 1 s later must mismatch).
- our sent takes (SCORE1, RAND1): why_markout.takes_of unchanged (filled / taken by other takers before us / pulled by makers / other;
  would-be value at our limit on arrival; the other takers' own trades on the quotes they took and their first trade - our decision),
  plus the aimed level's staleness (appearance in the frame known at our decision) on our arrival (decision + 22 ms) and during our hold.
Readings (merge): per group the shares and value per share by staleness bucket (<= 0, 0-0.5, 0.5-1, 1-2, 2-4, > 4 c), share-weighted
quantiles of age / staleness / Binance bps, the share of fill volume on quotes stale by >= 1 c (total, on arrival, during the hold); our
sent takes by class and by staleness during our hold. Point estimates, G = 4 days; the Predict / Binance clock offset is not measured
(TOP and RANKS2_10 +-15 ms, OTHERS M50 +-30 ms).
Outputs (write-once): OUT/stale_quotes/<day>_fills.parquet, <day>_sent.parquet, <day>.json; stale_quotes.json (merge).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/stale_quotes.py \
       selftest | rehearse <scratch dir> <day> <n markets> | day <day> | merge [<rehearsal dir> <day> ...]
"""
import collections
import gzip
import json
import math
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import score_replay as SRy  # noqa: E402
import why_markout as WM  # noqa: E402
import exit_times as XT  # noqa: E402

VR, OD, MM, TR, C, F = SRy.VR, SRy.OD, SRy.MM, SRy.TR, SRy.C, SRy.F
RP, S, WT = WM.RP, WM.S, XT.WT
OUT = WT.OUT / 'stale_quotes'
DAYS, NET = SRy.DAYS, SRy.NET
LAG_SET, PRE_MS, MATCH_MS, HOLD = 34., 25., 100., 150.
BUCKETS = [-np.inf, 0., .5, 1., 2., 4., np.inf]
BLAB = ['<= 0', '0-0.5', '0.5-1', '1-2', '2-4', '> 4']
GROUPS = ('TOP', 'RANKS2_10', 'OTHERS', 'SCORE1', 'RAND1')
QS = (.1, .25, .5, .75, .9)
SQ2 = math.sqrt(2.)


def runs(f, side):
    """Per frame and price (0..100 c): the index of the first frame of the run of consecutive frames showing size at that price on the
    side; -1 where the frame does not show it."""
    b = 2 + 10 * side
    P, Q = f[:, b:b + 5], f[:, b + 5:b + 10]
    n = len(f)
    pres = np.zeros((n, 101), bool)
    r, c = np.nonzero((P >= 0) & (P <= 100) & (Q > WM.EPS))
    pres[r, np.rint(P[r, c]).astype(int)] = True
    rs = np.full((n, 101), -1, np.int32)
    prev = np.full(101, -1, np.int32)
    for j in range(n):
        prev = np.where(pres[j], np.where(prev >= 0, prev, j), -1).astype(np.int32)
        rs[j] = prev
    return rs


def appear(f, rs_b, j, L):
    """The appearance (frame-clock ms) of the level at L in frame j -> (t or None, status)."""
    if j < 0:
        return None, 'before_path'
    Li = int(round(L))
    if not 0 <= Li <= 100:
        return None, 'price_out_of_range'
    r = int(rs_b[j, Li])
    if r < 0:
        return None, 'not_shown'
    return float(f[r, 0]), ('age_censored' if r == 0 else 'shown')


def zfun(btc, ws, t):
    """z(x) of reprice_replay's dFV with sigma and tau at t (absolute ms) -> function, or None where dfv_at is undefined."""
    bk, bm, cs = btc
    jw = int(np.searchsorted(bk, ws, 'right') - 1)
    j0, j1 = int(np.searchsorted(bk, t, 'right') - 1), int(np.searchsorted(bk, t - RP.LAG, 'right') - 1)
    js = int(np.searchsorted(bk, t - RP.LAG - RP.SIG_WIN, 'right') - 1)
    if min(jw, j0, j1) < 0:
        return None
    span = (t - RP.LAG - max(t - RP.LAG - RP.SIG_WIN, bk[0])) / 1000.
    sig = 1e4 * math.sqrt(max(cs[j1] - cs[max(js, 0)], 0.) / max(span, 1.)) if span > 1. else np.nan
    tau = (ws + 300000. - t) / 1000.
    if not (np.isfinite(sig) and sig > 0 and tau >= .5):
        return None
    s, b0 = sig * math.sqrt(tau), bm[jw]

    def z(x):
        j = int(np.searchsorted(bk, x, 'right') - 1)
        return None if j < 0 else 1e4 * math.log(bm[j] / b0) / s
    return z


def phi(z):
    return .5 * (1. + math.erf(z / SQ2))


def stale_parts(btc, ws, d, t_app, t_arr, t, ctl=None):
    """Staleness its way (cents) from the appearance to t, split at the arrival (appearance after the arrival: all of it in the hold);
    Binance bps its way; absolute ms. None where undefined."""
    z = zfun(btc, ws, t)
    if z is None:
        return None
    za, zr, zt = z(t_app), z(max(t_arr, t_app)), z(t)
    if za is None or zr is None or zt is None:
        return None
    pa, pr, pt = phi(za), phi(zr), phi(zt)
    if ctl is not None:
        ref = WM.dfv_at(btc[0], btc[1], btc[2], ws, t)
        zl, zf = z(t - RP.LAG), z(t - 2 * RP.LAG)
        if np.isfinite(ref) and zl is not None:
            ctl['stale_parity_checked'] += 1
            ctl['stale_parity_mismatch'] += int(abs(100. * (pt - phi(zl)) - ref) > 1e-9)
            if zf is not None:
                ctl['stale_falsifier_differs'] += int(abs(100. * (pt - phi(zf)) - ref) > 1e-9)
    bk, bm = btc[0], btc[1]
    ja, jt = int(np.searchsorted(bk, t_app, 'right') - 1), int(np.searchsorted(bk, t, 'right') - 1)
    return dict(st_total=100. * d * (pt - pa), st_arrival=100. * d * (pr - pa), st_hold=100. * d * (pt - pr), bn_bps=d * 1e4 * math.log(bm[jt] / bm[ja]))


def fill_row(f, rs, btc, t0, ws, end, grp, t_rel, b, L, q, ctl, extra=None):
    """One fill: side b consumed at price L at path time t_rel."""
    d = 1 if b == 1 else -1
    r = dict(grp=grp, t_rel=float(t_rel), side_b=int(b), d=d, price=float(L), q=float(q), **(extra or {}))
    if not (0. <= t_rel <= end):
        r['status'] = 'outside_path'
        return r
    wb = WM.would_be(f, d, float(L), float(t_rel), end)
    r.update(v1=wb['1'], v10=wb['10'])
    j = int(np.searchsorted(f[:, 0], t_rel - PRE_MS, side='right') - 1)
    ta, st = appear(f, rs[b], j, L)
    r['status'] = st
    if ta is not None:
        r['age_ms'] = t_rel - ta
        r['posted_during_hold'] = bool(ta > t_rel - HOLD)
        sp = stale_parts(btc, ws, d, t0 + ta, t0 + t_rel - HOLD, t0 + t_rel, ctl)
        if sp is not None:
            r.update(sp)
        else:
            ctl['stale_undefined'] += 1
    return r


def legs_of_day(day, comp):
    """TOP and RANKS2_10 taker legs of the day's complete markets: (mid, t_abs, side_b, price, q, grp, order)."""
    X = pd.read_parquet(WT.OUT / f'exit_legs_{day}.parquet')
    X = X.sort_values('recv_ns', kind='stable').drop_duplicates(['tx', 'maker_hash'], keep='first')
    X = X[(X.role == 'taker') & X.mid.isin(comp)]
    top = pd.DataFrame(dict(mid=X.mid.astype(int), t_abs=X.t_set.astype(float) - LAG_SET, side_b=np.where(X.side > 0, 1, 0), price=X.price_up_c.astype(float),
                            q=X.shares.astype(float), grp='TOP', order=X.taker_hash))
    O = pd.read_parquet(WT.OUT / 'fire_times_orders.parquet', columns=['taker_hash', 'day', 'rank', 't_set'])
    O = O[(O.day == day) & (O['rank'] > 1) & O.t_set.notna()].drop_duplicates('taker_hash').set_index('taker_hash')
    W = pd.read_parquet(WT.OUT / f'winners_legs_{day}.parquet', columns=['mid', 'side', 'ptick', 'shares', 'taker_hash', 'ambiguous'])
    W = W[~W.ambiguous & W.taker_hash.isin(O.index) & W.mid.isin(comp)]
    rk = pd.DataFrame(dict(mid=W.mid.astype(int), t_abs=W.taker_hash.map(O.t_set).astype(float) - LAG_SET, side_b=W.side.astype(int), price=W.ptick.astype(float),
                           q=W.shares.astype(float), grp='RANKS2_10', order=W.taker_hash))
    return pd.concat([top, rk], ignore_index=True)


def day_run(day, out=None, limit=None):
    resource.setrlimit(resource.RLIMIT_CPU, (1200, 1201))
    C.require(day in DAYS, f'{day} is not a declared day')
    od = Path(out) if out else OUT
    fj, ff, fs = od / f'{day}.json', od / f'{day}_fills.parquet', od / f'{day}_sent.parquet'
    C.require(not any(x.exists() for x in (fj, ff, fs)), 'an output exists (write-once)')
    od.mkdir(parents=True, exist_ok=True)
    vs = json.loads((VR.OUT / 'summary.json').read_text())
    comp = {int(r['mid']) for r in vs['rows'] if r['day'] == day and r['status'] == 'complete'}
    bk, bm = OD.seg_arrays(OD.SEG_OF[day]); bk = bk.astype(float); bm = bm.astype(float)
    btc = (bk, bm, np.cumsum(np.r_[0., np.diff(np.log(bm))] ** 2))
    LG = legs_of_day(day, comp)
    LGm = {m: g for m, g in LG.groupby('mid')}
    src = MM.KP.Tape()
    reg = MM.registry(day).set_index('mid'); off = src.offset(day)
    fills, sent, ctl, n_m = [], [], collections.Counter(), 0
    for h, g in reg.groupby('hour'):
        mids = [int(m) for m in g.index if int(m) in comp]
        if limit is not None:
            mids = mids[:max(0, limit - n_m)]
        if not mids:
            continue
        for m, W, lst in VR.worlds_all(day, mids, src, off, reg):
            C.require(lst == 'loaded', f'{m} {lst}')
            ws = float(reg.loc[m, 'ws']) * 1000.
            p, st, info = VR.vm_path(W, ws, 50.)
            C.require(st == 'complete', f'{m} {st}')
            t0, end = float(info['t_start']), float(info['end_rel'])
            f = np.asarray(p['frames'], float); tr = np.asarray(p['trades'], float).reshape(-1, 4)
            rs = {0: runs(f, 0), 1: runs(f, 1)}
            n_m += 1
            # smart takers' legs; public trades near one of them (same side and price within 100 ms) are theirs
            mine = np.zeros(len(tr), bool)
            for x in (LGm[m].itertuples() if m in LGm else ()):
                tl = x.t_abs - t0
                fills.append(dict(fill_row(f, rs, btc, t0, ws, end, x.grp, tl, x.side_b, x.price, x.q, ctl, dict(order=x.order)), day=day, mid=m))
                lo, hi = np.searchsorted(tr[:, 0], tl - MATCH_MS, 'left'), np.searchsorted(tr[:, 0], tl + MATCH_MS, 'right')
                k = np.arange(lo, hi)
                mine[k[(tr[k, 1] == x.side_b) & (np.abs(tr[k, 2] - x.price) < 1e-9)]] = True
            ctl['public_trades'] += len(tr); ctl['public_trades_near_a_smart_leg'] += int(mine.sum())
            for i in np.flatnonzero(~mine):
                fills.append(dict(fill_row(f, rs, btc, t0, ws, end, 'OTHERS', tr[i, 0], int(tr[i, 1]), tr[i, 2], tr[i, 3], ctl), day=day, mid=m))
            # our takes
            for cfg in SRy.CONFIGS:
                rec = json.loads(gzip.decompress((SRy.OUT / f'{m}_{cfg}.json.gz').read_bytes()))
                C.require(rec['config'] == cfg and int(rec['mid']) == m, f'{m} {cfg}: record mismatch')
                marks = {int(x['gid']): x for x in rec['take_marks']}
                for x in rec['ioc_results']:
                    if x['filled'] <= WM.EPS:
                        continue
                    s = int(x['side']); mk = marks[int(x['gid'])]
                    r = fill_row(f, rs, btc, t0, ws, end, cfg, float(x['act']), 1 - s, float(x['price']), float(x['filled']), ctl, dict(order=str(x['gid'])))
                    ctl['ours_parity_checked'] += 1
                    ctl['ours_parity_mismatch'] += int(not ((r.get('v1') is None and mk['m1s_c'] is None) or
                                                            (r.get('v1') is not None and mk['m1s_c'] is not None and abs(r['v1'] - mk['m1s_c']) < 1e-6)))
                    late = WM.would_be(f, r['d'], float(x['price']), float(x['act']) + 1000., end + 1000.)['1']
                    ctl['ours_falsifier_late_mismatch'] += int(late is None or mk['m1s_c'] is None or abs(late - mk['m1s_c']) > 1e-6)
                    fills.append(dict(r, day=day, mid=m))
                rows, cc = WM.takes_of(rec, p, f, tr, t0, end, ws, btc, None)
                ctl.update({f'{cfg}:{k}': v for k, v in cc.items()})
                for r in rows:
                    opp = 1 - r['side']
                    jk = int(np.searchsorted(f[:, 1], r['g'], side='right') - 1)
                    ta, st_ = appear(f, rs[opp], jk, r['L'])
                    r.update(config=cfg, aimed_status=st_, day=day, mid=m)
                    if ta is not None:
                        r['aimed_age_ms'] = r['g'] - ta
                        sp = stale_parts(btc, ws, r['d'], t0 + ta, t0 + r['g'] + NET, t0 + r['a'], ctl)
                        if sp is not None:
                            r.update({f'aimed_{k}': v for k, v in sp.items()})
                    sent.append(r)
            C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    Fd, Sd = pd.DataFrame(fills), pd.DataFrame(sent)
    Fd.to_parquet(ff, index=False); Sd.to_parquet(fs, index=False)
    TR.dump(fj, dict(asof='2026-11-26', day=day, markets=n_m, legs_in_day_markets={k: int(v) for k, v in LG.grp.value_counts().items()},
                     fills={k: int(v) for k, v in Fd.grp.value_counts().items()}, status={f'{g}:{s}': int(v) for (g, s), v in Fd.groupby(['grp', 'status']).size().items()},
                     sent={k: int(v) for k, v in Sd.config.value_counts().items()}, controls=dict(ctl), peak_rss_mib=F.rss_mib()))
    print(day, n_m, dict(Fd.grp.value_counts()), dict(ctl), flush=True)


def wq(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return None
    o = np.argsort(x[ok]); xs, cw = x[ok][o], np.cumsum(w[ok][o])
    return {f'p{int(q * 100)}': float(xs[min(np.searchsorted(cw, q * cw[-1]), len(xs) - 1)]) for q in QS}


def wavg(v, w):
    v, w = pd.to_numeric(v, errors='coerce'), pd.to_numeric(w, errors='coerce')
    ok = v.notna() & w.notna() & (w > 0)
    return float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else None


def merge(od=None, days=None):
    days = days or DAYS
    fn = (Path(od) if od else H) / 'stale_quotes.json'
    C.require(not fn.exists(), 'stale_quotes.json exists (write-once)')
    src = Path(od) if od else OUT
    Fd = pd.concat([pd.read_parquet(src / f'{d}_fills.parquet') for d in days], ignore_index=True)
    Sd = pd.concat([pd.read_parquet(src / f'{d}_sent.parquet') for d in days], ignore_index=True)
    J = [json.loads((src / f'{d}.json').read_text()) for d in days]
    ctl = sum((collections.Counter(j['controls']) for j in J), collections.Counter())
    out = dict(asof='2026-11-26', days=list(days), markets=sum(j['markets'] for j in J), controls=dict(ctl), groups={}, sent={})
    out['controls_pass'] = bool(ctl['stale_parity_mismatch'] == 0 and ctl['stale_parity_checked'] > 0 and ctl['ours_parity_mismatch'] == 0 and ctl['ours_parity_checked'] > 0)
    out['falsifiers_fire'] = bool(ctl['stale_falsifier_differs'] > 0 and ctl['ours_falsifier_late_mismatch'] > 0)
    for g in GROUPS:
        X = Fd[Fd.grp == g]
        if not len(X):
            continue
        q = X.q.astype(float)
        V = X[X.st_total.notna()] if 'st_total' in X else X.iloc[:0]
        o = dict(fills=int(len(X)), shares=float(q.sum()), orders=int(X.order.nunique()),
                 status_share={k: float(v / q.sum()) for k, v in X.groupby('status').q.sum().items()},
                 v1=wavg(X.v1, X.q), v10=wavg(X.v10, X.q), staleness_defined_share=float(V.q.sum() / q.sum()))
        if len(V):
            o.update(age_ms=wq(V.age_ms, V.q), st_total=wq(V.st_total, V.q), st_arrival=wq(V.st_arrival, V.q), st_hold=wq(V.st_hold, V.q), bn_bps=wq(V.bn_bps, V.q),
                     stale_ge1_total=float(V.q[V.st_total >= 1.].sum() / V.q.sum()), stale_ge1_arrival=float(V.q[V.st_arrival >= 1.].sum() / V.q.sum()),
                     stale_ge1_hold=float(V.q[V.st_hold >= 1.].sum() / V.q.sum()), posted_during_hold=float(V.q[V.posted_during_hold.astype(bool)].sum() / V.q.sum()),
                     mean_st_total=wavg(V.st_total, V.q), mean_st_hold=wavg(V.st_hold, V.q), mean_st_arrival=wavg(V.st_arrival, V.q))
            bb = pd.cut(V.st_total.astype(float), BUCKETS, labels=BLAB)
            o['by_total_staleness'] = {str(k): dict(share=float(G.q.sum() / V.q.sum()), v1=wavg(G.v1, G.q), v10=wavg(G.v10, G.q), fills=int(len(G)))
                                       for k, G in V.groupby(bb, observed=True)}
            bh = pd.cut(V.st_hold.astype(float), BUCKETS, labels=BLAB)
            o['by_hold_staleness'] = {str(k): dict(share=float(G.q.sum() / V.q.sum()), v1=wavg(G.v1, G.q), fills=int(len(G))) for k, G in V.groupby(bh, observed=True)}
        out['groups'][g] = o
    for cfg in SRy.CONFIGS:
        X = Sd[Sd.config == cfg]
        n = len(X)
        b = dict(sent=n, classes={})
        for c in ('filled', 'taken', 'pulled', 'other'):
            G = X[X.cls == c]
            hv = pd.to_numeric(G.get('aimed_st_hold'), errors='coerce') if len(G) else pd.Series(dtype=float)
            b['classes'][c] = dict(n=int(len(G)), share=len(G) / n if n else None, wb1=WM_mean(G.get('wb1')), wb10=WM_mean(G.get('wb10')),
                                   aimed_st_arrival=qs3(G.get('aimed_st_arrival')), aimed_st_hold=qs3(G.get('aimed_st_hold')),
                                   aimed_hold_ge1=float((hv >= 1.).sum() / hv.notna().sum()) if hv.notna().any() else None)
        K = X[(X.cls == 'taken') & pd.to_numeric(X.get('c_shares'), errors='coerce').notna()] if 'c_shares' in X else X.iloc[:0]
        b['taken_by_others'] = dict(n=int(len(K)), their_v1=wavg(K.c_v1, K.c_shares) if len(K) else None, our_wb1_same=WM_mean(K.wb1) if len(K) else None,
                                    decided_before_us=float((pd.to_numeric(K.gap_dec) < S.HOLD).mean()) if len(K) else None, gap_dec=qs3(K.gap_dec) if len(K) else None)
        hv = pd.to_numeric(X.get('aimed_st_hold'), errors='coerce')
        bh = pd.cut(hv, BUCKETS, labels=BLAB)
        b['by_aimed_hold_staleness'] = {str(k): dict(n=int(len(G)), filled=float((G.cls == 'filled').mean()), taken=float((G.cls == 'taken').mean()),
                                                     pulled=float((G.cls == 'pulled').mean()), other=float((G.cls == 'other').mean()), wb1=WM_mean(G.wb1))
                                        for k, G in X[hv.notna()].groupby(bh[hv.notna()], observed=True)}
        b['aimed_status'] = {k: int(v) for k, v in X.aimed_status.value_counts().items()}
        out['sent'][cfg] = b
    TR.dump(fn, out)
    print(json.dumps(out, indent=1, default=str)[:6000])


def WM_mean(s):
    if s is None:
        return None
    s = pd.to_numeric(s, errors='coerce').dropna()
    return float(s.mean()) if len(s) else None


def qs3(s):
    if s is None:
        return None
    s = pd.to_numeric(s, errors='coerce').dropna()
    return [float(x) for x in s.quantile([.25, .5, .75])] if len(s) else None


def selftest():
    # level runs: ask 40 shown in frames 2-4 and 6-7, ask 41 in every frame
    n = 8
    f = np.full((n, 22), -1.)
    f[:, 0] = np.arange(n) * 100.; f[:, 1] = f[:, 0] + 5.
    f[:, 17:22] = 0.
    f[:, 12] = 41.; f[:, 17] = 3.
    for j in (2, 3, 4, 6, 7):
        f[j, 13] = 40.; f[j, 18] = 2.
    rs = runs(f, 1)
    assert rs[4, 40] == 2 and rs[5, 40] == -1 and rs[7, 40] == 6 and rs[7, 41] == 0 and rs[3, 39] == -1, (rs[4, 40], rs[5, 40], rs[7, 40], rs[7, 41])
    assert appear(f, rs, 4, 40.) == (200., 'shown') and appear(f, rs, 5, 40.)[1] == 'not_shown' and appear(f, rs, 7, 41.)[1] == 'age_censored'
    j = int(np.searchsorted(f[:, 0], 430. - PRE_MS, side='right') - 1)
    assert j == 4 and appear(f, rs, j, 40.)[0] == 200.
    # staleness: parity with why_markout.dfv_at, additivity, sign
    rng = np.random.default_rng(3)
    bk = np.arange(0., 700000., 50.)
    bm = 100000. * np.exp(np.cumsum(rng.normal(0., 2e-5, len(bk))))
    btc = (bk, bm, np.cumsum(np.r_[0., np.diff(np.log(bm))] ** 2))
    ws, t = 350000., 500000.
    ctl = collections.Counter()
    a = stale_parts(btc, ws, 1, t - 250., t - 150., t, ctl)
    assert ctl['stale_parity_checked'] == 1 and ctl['stale_parity_mismatch'] == 0 and ctl['stale_falsifier_differs'] == 1, dict(ctl)
    assert abs(a['st_total'] - WM.dfv_at(bk, bm, btc[2], ws, t)) < 1e-9
    b = stale_parts(btc, ws, 1, t - 3000., t - 150., t)
    assert abs(b['st_total'] - b['st_arrival'] - b['st_hold']) < 1e-12
    c = stale_parts(btc, ws, -1, t - 3000., t - 150., t)
    assert abs(b['st_total'] + c['st_total']) < 1e-12 and abs(b['bn_bps'] + c['bn_bps']) < 1e-12
    e = stale_parts(btc, ws, 1, t - 100., t - 150., t)                               # posted during the hold: nothing on arrival
    assert e['st_arrival'] == 0. and abs(e['st_hold'] - e['st_total']) < 1e-12
    flat = (bk, np.full(len(bk), 1e5), np.zeros(len(bk)))
    assert stale_parts(flat, ws, 1, t - 3000., t - 150., t) is None                  # no volatility: undefined, as dfv_at
    print('stale_quotes selftest PASS')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'day':
        day_run(sys.argv[2])
    elif cmd == 'rehearse':
        day_run(sys.argv[3], out=sys.argv[2], limit=int(sys.argv[4]))
    elif cmd == 'merge':
        merge(sys.argv[2] if len(sys.argv) > 2 else None, tuple(sys.argv[3:]) or None)
    else:
        dict(selftest=selftest)[cmd]()
