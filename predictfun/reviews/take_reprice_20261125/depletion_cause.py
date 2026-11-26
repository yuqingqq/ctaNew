"""Is the opposite Binance futures best level that empties before the winners' sends traded away or withdrawn? (USER 2026-11-26: "When
opposite price level disappears causing imbalance, does it correspond to a trade, or the best level shrinks because maker withdraws").
Diagnostic on the winners_trigger points (winners_trigger_points.parquet: the TOP1 / TOP2_10 / REST sends and their 12 shifted nulls,
declared with that study); four consumed days; no strategy change.

Per point (send time s, direction d; the opposite level is the ask for d = +1) two windows, the book from the raw bookTicker on E read
exactly as winners_trigger.bt_feats (hour files, the previous hour's last 2 s prepended; its bt_depl300 is recomputed and must match):
- pre (s - 300, s]: the opposite best level at s - 300, price P and size Q. End status: same (the best is still P at s, size left Q'),
  gone (the best moved their way: P emptied, Q' = 0), improved (a better price in front: P hidden, no split), no_book, edge (s + 150
  within 1 s of the hour file's end), trade_gap (a trade-id gap inside the window).
- hold (s, s + 150]: the same from the opposite level at s (the tick their way happens when that level goes).
Split: traded V = aggressive trades at exactly P inside the window (buyer-taker for an ask, seller-taker for a bid; trades on E,
de-duplicated by id); withdrawn W = (Q - Q') - V = cancelled - added at P. On same / gone points bt_depl300 = V / Q + W / Q before its
clip; on gone points V / Q + W / Q = 1.
Checks (refuse on failure): bt_depl300 parity on every point; the stored AUC of bt_depl300 per group reproduced (full run); V against a
brute-force sum on 300 random points; prices on the 0.1 grid; the selftest (trade-only, pull, sweep, pull-out, add-then-trade, improved,
bid side, window ends, trade gap; a flipped-side mutant must fail; the day guard refuses before any file opens).
Reliability, declared before the result: aggressive trades at the best opposite price, each read over (E - 2, E + 1] ms against the
bookTicker; the split is read as reliable when the trade volume left without an equal visible size drop is <= 5 % (hidden liquidity, or
an add in the same ms, would show here; above 5 % the traded share is an upper bound).
Readings per group, sends vs null (point estimates: four days, no interval): status shares; mean size, shrink, traded and withdrawn (BTC);
medians of the feature and its two parts; AUC of each part (send vs null; per day for days with >= 50 sends); among points that lost >= 20 %,
the share where trades are at least half the shrink and with no trade at all; on gone points the traded / withdrawn fractions, swept
(W <= 0.5 mBTC) and pulled (V <= 0.5 mBTC); near-side sizes (what else moves the imbalance).
Outputs (write-once): depletion_cause.json; OUT/depletion_cause_points.parquet (per point, for the predictor step).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/depletion_cause.py \
       selftest <scratch dir> | run | rehearse <scratch dir> <day>
"""
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import winners_trigger as WT  # noqa: E402

C, TR, F, OUT, RAW = WT.C, WT.TR, WT.F, WT.OUT, WT.RAW
PRE, HOLD, EDGE, TOL, LOST, RELIABLE = 300., 150., 1000., 5e-4, .2, .05
SIDE_KEY, SHIFT, N_BRUTE, SEED = 10_000_000, 1 << 27, 300, 20261126
SAME, GONE, IMPROVED, NO_BOOK, EDGE_ST, GAP = range(6)
ST_NAMES = ('same', 'gone', 'improved', 'no_book', 'edge', 'trade_gap')
GROUPS = ('TOP1', 'TOP2_10', 'REST')
BUY_AGGR = 0                                   # m (buyer is maker) of a buyer-taker trade: it consumes asks


def book_hours(day, raw=RAW):
    """(hour, (E, bid, bq, ask, aq)) per hour file of a declared day, the previous hour's last 2 s prepended: winners_trigger.bt_feats."""
    C.require(day in WT.DAYS, f'{day} is not a declared day')
    tail = None
    for h in range(24):
        fn = raw / f'bookTicker/BTCUSDT/{day}_{h:02d}.csv.gz'
        if not fn.exists():
            tail = None
            continue
        b = pd.read_csv(fn, header=None, usecols=[1, 4, 5, 6, 7], names=['E', 'bid', 'bq', 'ask', 'aq'])
        if tail is not None:
            b = pd.concat([tail, b])
        b = b.sort_values('E', kind='stable')
        E = b.E.to_numpy(float)
        tail = b[b.E >= E[-1] - 2000.]
        yield h, tuple(b[c].to_numpy(float) for c in ('E', 'bid', 'bq', 'ask', 'aq'))


def read_trades(day, raw=RAW):
    C.require(day in WT.DAYS, f'{day} is not a declared day')
    parts = [pd.read_csv(raw / f'trade/BTCUSDT/{day}_{h:02d}.csv.gz', header=None, usecols=[1, 3, 4, 5, 6], names=['E', 'id', 'p', 'q', 'm'])
             for h in range(24) if (raw / f'trade/BTCUSDT/{day}_{h:02d}.csv.gz').exists()]
    return pd.concat(parts, ignore_index=True)


def build_index(tr, day0):
    """Aggressive volume by (side, price tick, E): sorted composite keys + cumulative size; trade-id gaps as E intervals."""
    n_raw = len(tr)
    tr = tr.drop_duplicates('id', keep='first').sort_values('id', kind='stable')
    ids, E = tr.id.to_numpy(np.int64), tr.E.to_numpy(np.int64)
    tick = tr.p.to_numpy(float) * 10.
    C.require(len(tr) and np.abs(tick - np.rint(tick)).max() < 1e-6, 'a trade price off the 0.1 grid')
    m = tr.m.to_numpy(np.int64)
    C.require(np.isin(m, [0, 1]).all(), 'trade side not 0/1')
    base = day0 - 10_000_000
    off = E - base
    C.require((off >= 0).all() and (off < SHIFT).all(), 'a trade time outside the index range')
    comp = (m * SIDE_KEY + np.rint(tick).astype(np.int64)) * SHIFT + off
    o = np.argsort(comp, kind='stable')
    gap = np.flatnonzero(np.diff(ids) > 1)
    lo, hi = np.minimum(E[gap], E[gap + 1]).astype(float), np.maximum(E[gap], E[gap + 1]).astype(float)
    g = np.argsort(lo, kind='stable')
    cs = comp[o]
    u = cs[np.r_[True, np.diff(cs) != 0]] if len(cs) else cs                   # trade-ms groups (side, price, E) for the reconciliation
    return dict(comp=cs, grp_e=(u % SHIFT + base).astype(float), grp_m=(u // SHIFT) // SIDE_KEY, grp_tick=(u // SHIFT) % SIDE_KEY, cq=np.r_[0., np.cumsum(tr.q.to_numpy(float)[o])], base=base, g_lo=lo[g], g_hm=np.maximum.accumulate(hi[g]) if len(g) else hi,
                E=E.astype(float), p=tr.p.to_numpy(float), q=tr.q.to_numpy(float), m=m,
                tape=dict(rows=n_raw, duplicates=n_raw - len(tr), missing_ids=int(ids[-1] - ids[0] + 1 - len(ids)), gaps=int(len(gap))))


def traded(ix, side, P, t_a, t_b):
    """Aggressive volume of `side` at exactly price P with E in (t_a, t_b]."""
    key = side.astype(np.int64) * SIDE_KEY + np.rint(P * 10.).astype(np.int64)
    qa, qb = np.floor(t_a) - ix['base'], np.floor(t_b) - ix['base']
    C.require((qa >= 0).all() and (qb < SHIFT).all(), 'a window outside the index range')
    lo = np.searchsorted(ix['comp'], key * SHIFT + qa.astype(np.int64), 'right')
    hi = np.searchsorted(ix['comp'], key * SHIFT + qb.astype(np.int64), 'right')
    return ix['cq'][hi] - ix['cq'][lo]


def split(bk, ix, t_a, t_b, d, edge):
    """The opposite best level at t_a followed to t_b: status, price P, size Q, size left at P, traded V, withdrawn W; bt_feats' depletion."""
    E, bid, bq, ask, aq = bk
    ja, jb = WT.last_idx(E, t_a), WT.last_idx(E, t_b)
    ok = (ja >= 0) & (jb >= 0)
    ja, jb = np.maximum(ja, 0), np.maximum(jb, 0)
    up = d > 0
    Pa, Qa, Pb, Qb = np.where(up, ask[ja], bid[ja]), np.where(up, aq[ja], bq[ja]), np.where(up, ask[jb], bid[jb]), np.where(up, aq[jb], bq[jb])
    dep = np.where(np.abs(Pb - Pa) < 1e-9, (Qa - Qb) / np.maximum(Qa, 1e-9), np.where(d * (Pb - Pa) > 0, 1., -1.))
    dep = np.where(ok, np.clip(dep, -1., 1.), np.nan)
    st = np.where(np.abs(Pb - Pa) < 1e-9, SAME, np.where(d * (Pb - Pa) > 0, GONE, IMPROVED))
    if len(ix['g_lo']):
        st = np.where(np.searchsorted(ix['g_lo'], t_b, 'right') > np.searchsorted(ix['g_hm'], t_a, 'right'), GAP, st)
    st = np.where(~ok | ~(Qa > 0), NO_BOOK, np.where(edge, EDGE_ST, st))
    sp = (st == SAME) | (st == GONE)
    tk = np.where(sp, Pa, 0.) * 10.
    C.require(np.abs(tk - np.rint(tk)).max(initial=0.) < 1e-6, 'a book price off the 0.1 grid')
    left = np.where(st == SAME, Qb, np.where(st == GONE, 0., np.nan))
    V = np.where(sp, traded(ix, np.where(up, BUY_AGGR, 1 - BUY_AGGR), np.where(sp, Pa, 0.), np.where(sp, t_a, t_b), t_b), np.nan)
    return dict(st=st, P=Pa, Q=Qa, left=left, V=V, W=Qa - left - V, dep=dep, near_a=np.where(up, bq[ja], aq[ja]), near_b=np.where(up, bq[jb], aq[jb]), opp_b=Qb)


def reconcile(bk, ix, lo, hi):
    """Aggressive trade-ms at the best opposite price with E in [lo, hi), each over (E - 2, E + 1]: traded volume vs the visible drop."""
    s = (ix['grp_e'] >= lo) & (ix['grp_e'] < hi)
    e, m, tick = ix['grp_e'][s], ix['grp_m'][s], ix['grp_tick'][s]
    d = np.where(m == BUY_AGGR, 1., -1.)
    r = split(bk, ix, e - 2., e + 1., d, np.zeros(len(e), bool))
    at = ((r['st'] == SAME) | (r['st'] == GONE)) & (np.abs(r['P'] * 10. - tick) < 1e-6)
    V, drop = r['V'][at], (r['Q'] - r['left'])[at]
    return dict(groups=int(at.sum()), volume=float(V.sum()), unreflected_volume=float(np.maximum(V - np.maximum(drop, 0.), 0.).sum()),
                exact=int((np.abs(drop - V) <= TOL).sum()), drop_less=int((drop < V - TOL).sum()), drop_more=int((drop > V + TOL).sum()))


def mean(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else None


def median(x):
    x = np.asarray(x, float)
    return float(np.nanmedian(x)) if np.isfinite(x).any() else None


def auc(y, x):
    y, x = np.asarray(y), np.asarray(x, float)
    ok = np.isfinite(x)
    return float(roc_auc_score(y[ok], x[ok])) if ok.sum() >= 100 and len(set(y[ok])) == 2 else None


def gone_block(Q, V, W):
    Q, V, W = (np.asarray(x, float) for x in (Q, V, W))
    return dict(n=int(len(Q)), mean_size_btc=mean(Q), traded_frac_mean=mean(V / Q), withdrawn_frac_mean=mean(W / Q),
                traded_share_of_volume=float(V.sum() / Q.sum()) if Q.sum() > 0 else None, swept=mean(W <= TOL), pulled=mean(V <= TOL))


def block(Z):
    o = dict(n=int(len(Z)), pre_status={k: mean(Z.pre_st == i) for i, k in enumerate(ST_NAMES)}, hold_status={k: mean(Z.hold_st == i) for i, k in enumerate(ST_NAMES)})
    A = Z[Z.pre_st.isin([SAME, GONE])]
    sh = A.pre_Q - A.pre_left
    o['pre'] = dict(n=int(len(A)), mean_size_btc=mean(A.pre_Q), mean_shrink_btc=mean(sh), mean_traded_btc=mean(A.pre_V), mean_withdrawn_btc=mean(A.pre_W),
                    traded_share_of_mean_shrink=mean(A.pre_V) / mean(sh) if len(A) and mean(sh) > 0 else None,
                    med_feature=median(A.dep), med_traded_frac=median(A.pre_V / A.pre_Q), med_withdrawn_frac=median(A.pre_W / A.pre_Q),
                    share_any_trade=mean(A.pre_V > TOL), share_gone=mean(A.pre_st == GONE), share_lost20=mean(A.dep >= LOST))
    L = A[A.dep >= LOST]
    o['pre']['lost20'] = dict(n=int(len(L)), trade_led=mean(L.pre_V >= .5 * (L.pre_Q - L.pre_left)), no_trade=mean(L.pre_V <= TOL))
    Gn = A[A.pre_st == GONE]
    o['pre']['gone'] = gone_block(Gn.pre_Q, Gn.pre_V, Gn.pre_W)
    o['near'] = dict(med_near_start_btc=median(A.near_a), med_near_send_btc=median(A.near_b), med_opp_start_btc=median(A.pre_Q), med_opp_send_btc=median(A.opp_b))
    B = Z[Z.hold_st.isin([SAME, GONE])]
    hs = B.hold_Q - B.hold_left
    o['hold'] = dict(n=int(len(B)), share_gone=mean(B.hold_st == GONE), mean_size_btc=mean(B.hold_Q), mean_shrink_btc=mean(hs), mean_traded_btc=mean(B.hold_V),
                     mean_withdrawn_btc=mean(B.hold_W))
    Hg = B[B.hold_st == GONE]
    o['hold']['gone'] = gone_block(Hg.hold_Q, Hg.hold_V, Hg.hold_W)
    return o


def parts_auc(G):
    A = G[G.pre_st.isin([SAME, GONE])]
    return dict(feature_all=auc(G.y, G.dep), feature=auc(A.y, A.dep), traded_frac=auc(A.y, A.pre_V / A.pre_Q), withdrawn_frac=auc(A.y, A.pre_W / A.pre_Q))


def readings(X):
    out = {}
    for g in GROUPS:
        G = X[X.grp == g]
        if not G.y.sum():
            continue
        r = dict(send=block(G[G.y == 1]), null=block(G[G.y == 0]), auc=parts_auc(G), auc_by_day={})
        for dy, D in G.groupby('day'):
            if ((D.y == 1) & D.pre_st.isin([SAME, GONE])).sum() >= 50:
                r['auc_by_day'][dy] = parts_auc(D)
        a, s = r['auc'], r['send']['pre']
        if a['traded_frac'] is not None and a['withdrawn_frac'] is not None:
            r['withdrawn_part_separates_more'] = abs(a['withdrawn_frac'] - .5) > abs(a['traded_frac'] - .5)
        if s['mean_traded_btc'] is not None and s['mean_withdrawn_btc'] is not None:
            r['withdrawn_exceeds_traded_at_sends'] = s['mean_withdrawn_btc'] > s['mean_traded_btc']
        out[g] = r
    return out


def brute(ix, P, side, t_a, t_b):
    m = (ix['m'] == side) & (np.abs(ix['p'] - P) < 1e-6) & (ix['E'] > np.floor(t_a)) & (ix['E'] <= np.floor(t_b))
    return float(ix['q'][m].sum())


def process(X, days, raw=RAW, rng=None):
    """Per-point split columns for the points of `days` (X rows in place); returns the checks."""
    n = len(X)
    cols = {c: np.full(n, np.nan) for c in ('pre_P', 'pre_Q', 'pre_left', 'pre_V', 'pre_W', 'dep', 'near_a', 'near_b', 'opp_b',
                                            'hold_Q', 'hold_left', 'hold_V', 'hold_W')}
    pre_st, hold_st = np.full(n, NO_BOOK, np.int8), np.full(n, NO_BOOK, np.int8)
    ck = dict(tape={}, reconcile={}, brute=[])
    rng = rng or np.random.default_rng(SEED)
    for day in days:
        day0 = WT.OD_day_ms(day)
        rows = np.flatnonzero((X.day == day).to_numpy())
        s, d = X.s.to_numpy(float)[rows], X.d.to_numpy(float)[rows]
        hour = ((s - day0) // 3600000).astype(int)
        ix = build_index(read_trades(day, raw), day0)
        ck['tape'][day] = ix['tape']
        rc = dict(groups=0, volume=0., unreflected_volume=0., exact=0, drop_less=0, drop_more=0)
        for h, bk in book_hours(day, raw):
            h0 = day0 + h * 3600000.
            for k, v in reconcile(bk, ix, h0 + 2000., h0 + 3600000. - EDGE).items():
                rc[k] += v
            sel = np.flatnonzero(hour == h)
            if not len(sel):
                continue
            t, dd, edge = s[sel], d[sel], s[sel] + HOLD > h0 + 3600000. - EDGE
            a, b = split(bk, ix, t - PRE, t, dd, edge), split(bk, ix, t, t + HOLD, dd, edge)
            r = rows[sel]
            pre_st[r], hold_st[r] = a['st'], b['st']
            for c in ('P', 'Q', 'left', 'V', 'W'):
                cols[f'pre_{c}'][r] = a[c]
            for c in ('Q', 'left', 'V', 'W'):
                cols[f'hold_{c}'][r] = b[c]
            for c in ('dep', 'near_a', 'near_b', 'opp_b'):
                cols[c][r] = a[c]
            sp = np.flatnonzero((a['st'] == SAME) | (a['st'] == GONE))
            for i in rng.choice(sp, min(len(sp), max(1, N_BRUTE // (24 * len(days)))), replace=False) if len(sp) else []:
                side = BUY_AGGR if dd[i] > 0 else 1 - BUY_AGGR
                ck['brute'].append(abs(brute(ix, a['P'][i], side, t[i] - PRE, t[i]) - a['V'][i]))
        rc['unreflected_share'] = rc['unreflected_volume'] / rc['volume'] if rc['volume'] > 0 else None
        ck['reconcile'][day] = rc
        del ix
        C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
        print(day, len(rows), flush=True)
    for c, v in cols.items():
        X[c] = v
    X['pre_st'], X['hold_st'] = pre_st, hold_st
    return ck


def finish(X, ck, days, fj, fp, stored=None):
    sub = X.bt_depl300.to_numpy(np.float32)
    rec = X.dep.to_numpy(float).astype(np.float32)
    mis = ~((sub == rec) | (np.isnan(sub) & np.isnan(rec)))
    C.require(not mis.any(), f'bt_depl300 parity: {int(mis.sum())} mismatches')
    X = X.copy()
    X['dep'] = sub.astype(float)                       # equal to the stored feature: its float32 ties, as in winners_trigger's AUCs
    C.require(len(ck['brute']) and max(ck['brute']) < 1e-9, 'V differs from the brute-force sum')
    R = readings(X)
    if stored is not None:
        for g, r in R.items():
            C.require(abs(r['auc']['feature_all'] - stored[g]['uni']['bt_depl300']['auc']) < 1e-9, f'{g}: the stored bt_depl300 AUC is not reproduced')
    tot = {k: sum(v[k] for v in ck['reconcile'].values()) for k in ('groups', 'volume', 'unreflected_volume', 'exact', 'drop_less', 'drop_more')}
    tot['unreflected_share'] = tot['unreflected_volume'] / tot['volume'] if tot['volume'] > 0 else None
    out = dict(asof='2026-11-26', days=list(days), points=int(len(X)), declared=dict(pre_ms=PRE, hold_ms=HOLD, edge_ms=EDGE, tol_btc=TOL, lost=LOST, reliable_unreflected=RELIABLE),
               checks=dict(parity_points=int(len(X)), parity_mismatches=int(mis.sum()), brute_points=len(ck['brute']), brute_max_diff=max(ck['brute']),
                           stored_auc_reproduced=stored is not None, tape=ck['tape'], reconcile=dict(by_day=ck['reconcile'], all=tot),
                           split_reliable=tot['unreflected_share'] is not None and tot['unreflected_share'] <= RELIABLE),
               groups=R)
    keep = ['day', 'k', 'mid', 'y', 'grp', 'd', 's', 'pre_st', 'hold_st'] + [c for c in X.columns if c.startswith(('pre_', 'hold_')) and c not in ('pre_st', 'hold_st')] + ['dep', 'near_a', 'near_b', 'opp_b']
    P = X[keep].copy()
    for c in keep[9:]:
        P[c] = P[c].astype(np.float32)
    P.to_parquet(fp, index=False)
    TR.dump(fj, out)
    return out


def run(argv):
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    if argv[:1] == ['rehearse']:
        days, fj = [argv[2]], Path(argv[1]) / 'depletion_cause.json'
        fp = Path(argv[1]) / 'depletion_cause_points.parquet'
        C.require(days[0] in WT.DAYS, f'{days[0]} is not a declared day')
    else:
        days, fj, fp = list(WT.DAYS), H / 'depletion_cause.json', OUT / 'depletion_cause_points.parquet'
    C.require(not fj.exists() and not fp.exists(), 'an output exists (write-once)')
    X = pd.read_parquet(OUT / 'winners_trigger_points.parquet', columns=['day', 'k', 'mid', 'y', 'grp', 'd', 's', 'bt_depl300'])
    X = X[X.day.isin(days)].reset_index(drop=True)
    ck = process(X, days)
    stored = json.loads((H / 'winners_trigger.json').read_text())['groups'] if days == list(WT.DAYS) else None
    out = finish(X, ck, days, fj, fp, stored)
    print(json.dumps(dict(checks={k: v for k, v in out['checks'].items() if k not in ('tape', 'reconcile')}, reconcile=out['checks']['reconcile']['all'],
                          tape=out['checks']['tape']), indent=1))
    for g, r in out['groups'].items():
        print(g, json.dumps(dict(auc=r['auc'], send=r['send']['pre'], null=r['null']['pre'], send_hold=r['send']['hold'], near_send=r['send']['near'],
                                 near_null=r['null']['near']), indent=1))


def selftest(argv):
    """Synthetic sequences through split / build_index / reconcile, a flipped-side mutant, the day guard, and every writer end to end."""
    global BUY_AGGR
    scratch = Path(argv[0])
    C.require(scratch.is_dir(), 'selftest needs a scratch dir')

    def bk_of(rows):                                    # rows: (E, bid, bq, ask, aq)
        a = np.asarray(rows, float)
        return tuple(a[:, i] for i in range(5))

    def ix_of(trades):                                  # trades: (E, id, p, q, m)
        return build_index(pd.DataFrame(trades, columns=['E', 'id', 'p', 'q', 'm']).astype({'E': np.int64, 'id': np.int64, 'm': np.int64}), 0)

    def one(bk, ix, t_a, t_b, d):
        r = split(bk, ix, np.array([t_a]), np.array([t_b]), np.array([d]), np.array([False]))
        return {k: (v[0].item() if hasattr(v[0], 'item') else v[0]) for k, v in r.items()}

    B0 = [(0, 99.9, 5, 100.0, 10)]
    cases = {
        'trade_only': (B0 + [(100, 99.9, 5, 100.0, 7), (300, 99.9, 5, 100.0, 4)], [(0, 1, 100.0, 1, 0), (100, 2, 100.0, 3, 0), (300, 3, 100.0, 3, 0), (301, 4, 100.0, 2, 0)],
                       1, dict(st=SAME, Q=10, left=4, V=6, W=0)),
        'pull': (B0 + [(150, 99.9, 5, 100.0, 4)], [(10, 1, 99.9, 1, 1)], 1, dict(st=SAME, Q=10, left=4, V=0, W=6)),
        'sweep': (B0 + [(100, 99.9, 5, 100.2, 3)], [(100, 1, 100.0, 10, 0), (100, 2, 100.1, 2, 0)], 1, dict(st=GONE, Q=10, left=0, V=10, W=0)),
        'pull_out': (B0 + [(100, 99.9, 5, 100.1, 6)], [(50, 1, 99.9, 2, 1)], 1, dict(st=GONE, Q=10, left=0, V=0, W=10)),
        'add_then_trade': (B0 + [(100, 99.9, 5, 100.0, 15), (200, 99.9, 5, 100.0, 7)], [(200, 1, 100.0, 8, 0)], 1, dict(st=SAME, Q=10, left=7, V=8, W=-5)),
        'improved': ([(0, 99.8, 5, 100.0, 10), (100, 99.8, 5, 99.9, 1)], [(50, 1, 100.0, 2, 0)], 1, dict(st=IMPROVED)),
        'bid_side': ([(0, 99.9, 10, 100.0, 5), (100, 99.9, 6, 100.0, 5)], [(100, 1, 99.9, 4, 1), (120, 2, 100.0, 3, 0)], -1, dict(st=SAME, Q=10, left=6, V=4, W=0)),
        'trade_gap': (B0 + [(100, 99.9, 5, 100.0, 7)], [(50, 1, 100.0, 1, 0), (150, 5, 100.0, 2, 0)], 1, dict(st=GAP)),
    }
    for name, (rows, trades, d, want) in cases.items():
        got = one(bk_of(rows), ix_of(trades), 0., 300., d)
        for k, v in want.items():
            C.require(abs(got[k] - v) < 1e-9, f'selftest {name}: {k} = {got[k]}, want {v}')
        if want['st'] in (SAME, GONE):
            C.require(abs(got['V'] / got['Q'] + got['W'] / got['Q'] - (got['Q'] - got['left']) / got['Q']) < 1e-12, f'selftest {name}: parts do not add')
    rows, trades, d, want = cases['trade_only']
    BUY_AGGR = 1                                        # mutant: flipped aggressor side
    try:
        got = one(bk_of(rows), ix_of(trades), 0., 300., d)
    finally:
        BUY_AGGR = 0
    C.require(abs(got['V'] - want['V']) > 1e-9, 'selftest: the flipped-side mutant was not detected')
    try:
        next(book_hours('20200101'))
        C.require(False, 'selftest: the day guard did not refuse')
    except ValueError as e:
        C.require('is not a declared day' in str(e), f'selftest: the day guard refused for another reason: {e}')
    rb = bk_of([(0, 99.9, 5, 100.0, 10), (100, 99.9, 5, 100.0, 7), (200, 99.9, 5, 100.0, 7)])
    rr = reconcile(rb, ix_of([(100, 1, 100.0, 3, 0), (200, 2, 100.0, 2, 0)]), 0., 1000.)
    C.require(rr['groups'] == 2 and rr['exact'] == 1 and rr['drop_less'] == 1 and abs(rr['unreflected_volume'] - 2.) < 1e-9, f'selftest reconcile: {rr}')
    # every writer end to end on a synthetic point set (two groups, sends and nulls, gone / same / improved statuses)
    rng = np.random.default_rng(1)
    n = 600
    X = pd.DataFrame(dict(day='20261101', k=np.arange(n), mid=1, y=(np.arange(n) % 3 == 0).astype(int), grp=np.where(np.arange(n) % 2, 'TOP1', 'REST'),
                          d=1., s=0., bt_depl300=np.float32(np.nan)))
    Q = rng.uniform(1, 10, n)
    left = np.where(X.y == 1, Q * rng.uniform(0, 1, n), Q * rng.uniform(.5, 1.5, n))
    st = np.where(np.arange(n) % 10 == 0, GONE, np.where(np.arange(n) % 10 == 1, IMPROVED, SAME))
    left = np.where(st == GONE, 0., np.where(st == IMPROVED, np.nan, left))
    V = np.where(st == IMPROVED, np.nan, rng.uniform(0, 1, n) * Q)
    for c, v in dict(pre_P=100., pre_Q=Q, pre_left=left, pre_V=V, pre_W=Q - left - V, dep=np.where(st == IMPROVED, -1., np.clip((Q - left) / Q, -1, 1)),
                     near_a=5., near_b=6., opp_b=np.where(st == GONE, 3., left), hold_Q=Q, hold_left=np.where(st == GONE, 0., Q / 2), hold_V=Q / 4, hold_W=Q / 4).items():
        X[c] = v
    X['hold_W'] = X.hold_Q - X.hold_left - X.hold_V
    X['pre_st'], X['hold_st'] = st.astype(np.int8), st.astype(np.int8)
    X['bt_depl300'] = X.dep.astype(np.float32)
    ck = dict(tape={'20261101': dict(rows=1)}, reconcile={'20261101': dict(groups=1, volume=1., unreflected_volume=0., exact=1, drop_less=0, drop_more=0, unreflected_share=0.)},
              brute=[0.])
    fj, fp = scratch / 'selftest_depletion_cause.json', scratch / 'selftest_depletion_cause_points.parquet'
    for f in (fj, fp):
        f.unlink(missing_ok=True)
    out = finish(X, ck, ['20261101'], fj, fp)
    back, P = json.loads(fj.read_text()), pd.read_parquet(fp)
    C.require(set(back['groups']) == {'TOP1', 'REST'} and len(P) == n and back['checks']['parity_mismatches'] == 0, 'selftest: writers')
    g = out['groups']['TOP1']['send']['pre']
    C.require(abs(g['mean_shrink_btc'] - g['mean_traded_btc'] - g['mean_withdrawn_btc']) < 1e-9, 'selftest: shrink != traded + withdrawn')
    X2 = X.copy(); X2.loc[0, 'bt_depl300'] = np.float32(.5)
    try:
        finish(X2, ck, ['20261101'], scratch / 'x.json', scratch / 'x.parquet')
        C.require(False, 'selftest: a parity mismatch was not refused')
    except ValueError as e:
        C.require('bt_depl300 parity' in str(e), f'selftest: parity refused for another reason: {e}')
    print('selftest ok', len(cases), 'cases')


if __name__ == '__main__':
    dict(selftest=lambda: selftest(sys.argv[2:]), run=lambda: run([]), rehearse=lambda: run(['rehearse'] + sys.argv[2:]))[sys.argv[1]]()
