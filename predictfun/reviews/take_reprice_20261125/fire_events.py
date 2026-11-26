"""What happens on Binance just before the winning takers fire, and how fast are they? (USER 2026-11-26: "check their firing timings first,
is it when some trades happen, the top level becomes imbalanced, or when some makers cancel ..., or makers post more ..., or it is a
combination ... we first need to get the time when the takers fire, then we check what happened, to figure out how fast we need to be ...
detect the signals early enough"). Event study around the fire times of fire_times.py (t_set - 150 ms: an upper bound on the venue receipt
under FINDINGS C23's premises; Predict's clock against Binance's E, offset unknown); four consumed days; no strategy change.
Binance futures best-level events (raw bookTicker on E, collapsed to the last state per ms; trades on E, de-duplicated by id), per side:
trade (aggressive volume at any price, timed by the trade), cancel / add (a size change at an unchanged best price net of the aggressive
volume at that price in the interval: drop = cancel, rise = add; the untraded rest of an emptied level is a cancel; adds and cancels inside
one update net out, so both are lower bounds), gone (the best price moves away from the book: the level emptied), improve (a better price
appears). Signed by the taker's direction d: opp = the side that must empty for the price to move their way (the ask for d = +1), near =
the other; toward them: opp trade / cancel / gone, near add / improve; against them: opp add / improve, near trade / cancel / gone.
Points: each burst (one taker, one market, fires within 2 ms; its first fire) of the top taker (TOP1, all) and ranks 2-10 (TOP2_10, a seeded
sample of 20,000 bursts over the four days); null = the same fire shifted by +-5, 10, .., 30 s, same d (declared with winners_trigger). Excluded, counted: points within 1.5 s of an hour file
boundary or in an hour without a bookTicker file.
Readings per group, sends vs null: (1) event rates (count and BTC per ms) by time relative to the fire, 5 ms bins over [-300, +150) and
1 ms bins over [-50, +20); (2) time since the last event of each kind; (3) which of opp trade / opp cancel / opp gone / near add occurred
in the last 10 / 25 / 50 ms: share of sends and of random times per combination, lift, the Binance mid moving their way within 150 ms and
the sends' 1 s value (winners_trigger points, where sampled); (4) imbalance, opposite and near best size at fire - 300 .. 0 ms; (5) the same
1 ms profile around the M50 send time of the same orders, all and by how the first leg was timed (websocket / book drop): which clock
aligns the events more sharply (toward-event excess over random times in the last 20 ms).
Output (write-once): fire_events.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/fire_events.py \
       selftest | run [<rehearsal dir> <day>]
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
import depletion_cause as DC  # noqa: E402

C, TR, WT = DC.C, DC.TR, DC.WT
OUT, RAW, TOL = WT.OUT, WT.RAW, DC.TOL
KINDS = ('trade', 'cancel', 'add', 'gone', 'improve')
ROLES = ('opp', 'near')
PATTERN = (('opp', 'trade'), ('opp', 'cancel'), ('opp', 'gone'), ('near', 'add'))
TOWARD = (('opp', 'trade'), ('opp', 'cancel'), ('opp', 'gone'), ('near', 'add'), ('near', 'improve'))
COARSE, FINE = np.arange(-300., 151., 5.), np.arange(-50., 21., 1.)
LAGS, WINS = (300., 200., 100., 50., 20., 10., 5., 0.), (10., 25., 50.)
EDGE_MS, KEEP, CHUNK, HIT_MS, CAP_2_10 = 1500., 200., 20000, 150., 20000
QS = (.05, .25, .5, .75, .95)


def collapse(E, bid, bq, ask, aq):
    last = np.r_[E[1:] != E[:-1], True]
    return E[last], bid[last], bq[last], ask[last], aq[last]


def book_events(bk, ix):
    """Cancel / add / gone / improve events per side from consecutive per-ms states: {(side, kind): (E, amount)}."""
    E, bid, bq, ask, aq = bk
    ev = {}
    for side, P, Q, m_aggr, away_sign in (('ask', ask, aq, DC.BUY_AGGR, 1.), ('bid', bid, bq, 1 - DC.BUY_AGGR, -1.)):
        p0, p1, q0, q1, e0, e1 = P[:-1], P[1:], Q[:-1], Q[1:], E[:-1], E[1:]
        V = DC.traded(ix, np.full(len(p0), m_aggr), p0, e0, e1)
        same = np.abs(p1 - p0) < 1e-9
        away, better = away_sign * (p1 - p0) > 1e-9, away_sign * (p1 - p0) < -1e-9
        r = (q1 - q0) + V
        rest = q0 - V
        add, cxl, cg = same & (r > TOL), same & (r < -TOL), away & (rest > TOL)
        ce = np.r_[e1[cxl], e1[cg]]
        o = np.argsort(ce, kind='stable')
        ev[(side, 'add')] = (e1[add], r[add])
        ev[(side, 'cancel')] = (ce[o], np.r_[-r[cxl], rest[cg]][o])
        ev[(side, 'gone')] = (e1[away], q0[away])
        ev[(side, 'improve')] = (e1[better], q1[better])
    return ev


def trade_events(ix):
    buy = ix['m'] == DC.BUY_AGGR
    o = np.argsort(ix['E'], kind='stable')
    E, q, b = ix['E'][o], ix['q'][o], buy[o]
    return {('ask', 'trade'): (E[b], q[b]), ('bid', 'trade'): (E[~b], q[~b])}


def side_of(role, d):
    """The book side of a role for direction d: opp is the ask for d = +1."""
    return 'ask' if (role == 'opp') == (d > 0) else 'bid'


def window_sums(E, cA, t, edges):
    n, v = np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)
    for i in range(0, len(t), CHUNK):
        j = np.searchsorted(E, t[i:i + CHUNK, None] + edges[None, :], 'right')
        n += np.diff(j, axis=1).sum(0)
        v += np.diff(cA[j], axis=1).sum(0)
    return n, v


def since_last(E, t):
    if not len(E):
        return np.full(len(t), np.nan)
    j = np.searchsorted(E, t, 'right') - 1
    return np.where(j >= 0, t - E[np.maximum(j, 0)], np.nan)


def count_in(E, t, w):
    return np.searchsorted(E, t, 'right') - np.searchsorted(E, t - w, 'right')


def qs(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {f'p{int(q * 100)}': float(np.quantile(x, q)) for q in QS} if len(x) else None


def mean(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else None


def day_data(day, P):
    """Events of the day and, for the points P (t, d), the book trajectory and the Binance move over the next 150 ms."""
    day0 = WT.OD_day_ms(day)
    ix = DC.build_index(DC.read_trades(day), day0)
    parts = collections.defaultdict(list)
    t, d = P.t.to_numpy(float), P.d.to_numpy(float)
    hour = ((t - day0) // 3600000).astype(int)
    edge = (np.abs(t - (day0 + np.rint((t - day0) / 3600000.) * 3600000.)) < EDGE_MS)
    st = np.where(edge, 'edge', 'no_book').astype(object)
    traj = {f'{k}_{int(x)}': np.full(len(t), np.nan) for k in ('imb', 'opp', 'near') for x in LAGS}
    hit = np.full(len(t), np.nan)
    tail = None
    for h, bk in DC.book_hours(day):
        h0 = day0 + h * 3600000.
        lo = h0 - KEEP if h > 0 and tail == h - 1 else h0
        tail = h
        E, bid, bq, ask, aq = collapse(*bk)
        ev = book_events((E, bid, bq, ask, aq), ix)
        for k, (e, a) in ev.items():
            keep = (e >= lo) & (e < h0 + 3600000. - KEEP)
            parts[k].append((e[keep], a[keep]))
        sel = np.flatnonzero((hour == h) & ~edge)
        if not len(sel):
            continue
        st[sel] = 'ok'
        tt, dd = t[sel], d[sel]
        up = dd > 0
        for x in LAGS:
            j = np.maximum(WT.last_idx(E, tt - x), 0)
            opp, near = np.where(up, aq[j], bq[j]), np.where(up, bq[j], aq[j])
            traj[f'imb_{int(x)}'][sel] = (near - opp) / np.maximum(near + opp, 1e-9)
            traj[f'opp_{int(x)}'][sel], traj[f'near_{int(x)}'][sel] = opp, near
        j0, j1 = np.maximum(WT.last_idx(E, tt), 0), np.maximum(WT.last_idx(E, tt + HIT_MS), 0)
        hit[sel] = (dd * ((bid[j1] + ask[j1]) - (bid[j0] + ask[j0])) > 1e-9).astype(float)
    C.require(len(parts) > 0, f'{day}: no bookTicker hour')
    ev = {k: (np.concatenate([e for e, _ in v]), np.concatenate([a for _, a in v])) for k, v in parts.items()}
    ev.update(trade_events(ix))
    for k, (e, a) in ev.items():
        o = np.argsort(e, kind='stable')
        ev[k] = (e[o], np.r_[0., np.cumsum(a[o])])
    C.require(DC.F.rss_mib() < DC.F.RSS_GUARD_MIB, 'RSS over the self-guard')
    return ev, st, traj, hit


def role_events(ev, role, kind, d):
    return ev[(side_of(role, d), kind)]


def peri(ev, t, d, edges):
    """Mean events per point per bin (count, BTC) for every role / kind, the side chosen by each point's d."""
    out = {}
    for role in ROLES:
        for kind in KINDS:
            n, v = np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)
            for dv in (1., -1.):
                m = d == dv
                if m.any():
                    E, cA = role_events(ev, role, kind, dv)
                    a, b = window_sums(E, cA, t[m], edges)
                    n += a
                    v += b
            out[f'{role}_{kind}'] = (n, v)
    return out


def per_point(ev, t, d, fn):
    """fn(E, t) per point with the side chosen by d, for every role / kind."""
    o = {}
    for role in ROLES:
        for kind in KINDS:
            x = np.full(len(t), np.nan)
            for dv in (1., -1.):
                m = d == dv
                if m.any():
                    x[m] = fn(role_events(ev, role, kind, dv)[0], t[m])
            o[f'{role}_{kind}'] = x
    return o


def run(argv):
    resource.setrlimit(resource.RLIMIT_CPU, (2400, 2401))
    if argv[:1] == ['rehearse']:
        days, fj = [argv[2]], Path(argv[1]) / 'fire_events.json'
        C.require(days[0] in WT.DAYS, f'{days[0]} is not a declared day')
    else:
        days, fj = list(WT.DAYS), H / 'fire_events.json'
    C.require(not fj.exists(), 'fire_events.json exists (write-once)')
    A = pd.read_parquet(OUT / 'fire_times_orders.parquet')
    A = A[A.t_set.notna() & A.day.isin(days)].copy()
    A['grp'] = np.where(A['rank'] == 1, 'TOP1', 'TOP2_10')
    B = A.sort_values(['burst', 'fire'], kind='stable').drop_duplicates('burst', keep='first').reset_index(drop=True)
    n2 = int((B.grp == 'TOP2_10').sum())
    if n2 > CAP_2_10:                                   # ranks 2-10 are the contrast: a seeded sample keeps the run small
        keep = np.zeros(len(B), bool)
        keep[(B.grp == 'TOP1').to_numpy()] = True
        keep[np.random.default_rng(WT.SEED).choice(np.flatnonzero((B.grp == 'TOP2_10').to_numpy()), CAP_2_10, replace=False)] = True
        B = B[keep].reset_index(drop=True)
    V1 = pd.read_parquet(OUT / 'winners_trigger_points.parquet', columns=['day', 'mid', 's', 'y', 'grp', 'v1'])
    V1 = V1[(V1.y == 1) & V1.grp.isin(['TOP1', 'TOP2_10'])].drop_duplicates(['day', 'mid', 's', 'grp'])
    B = B.merge(V1[['day', 'mid', 's', 'grp', 'v1']].rename(columns={'s': 's_m50'}), on=['day', 'mid', 's_m50', 'grp'], how='left', validate='many_to_one')
    pts = [pd.DataFrame(dict(day=B.day, grp=B.grp, y=1, t=B.fire, d=B.d.astype(float), v1=B.v1, t_m50=B.s_m50, how=B.how2_first))]
    pts += [pd.DataFrame(dict(day=B.day, grp=B.grp, y=0, t=B.fire + s, d=B.d.astype(float), v1=np.nan, t_m50=np.nan, how=0)) for s in WT.SHIFTS]
    P = pd.concat(pts, ignore_index=True)
    d0 = P.day.map({x: float(WT.OD_day_ms(x)) for x in days})
    P = P[P.t.between(d0 + 2000., d0 + 86400000. - 2000.)].reset_index(drop=True)
    acc = collections.defaultdict(lambda: collections.defaultdict(float))
    rows = []
    for day in days:
        Pd = P[P.day == day].reset_index(drop=True)
        ev, st, traj, hit = day_data(day, Pd)
        Pd['status'], Pd['hit150'] = st, hit
        for k, v in traj.items():
            Pd[k] = v
        ok = (Pd.status == 'ok').to_numpy()
        for (g, y), ix in Pd[ok].groupby(['grp', 'y']).indices.items():
            G = Pd[ok].iloc[ix]
            t, d = G.t.to_numpy(float), G.d.to_numpy(float)
            key = f'{g}|{y}'
            acc[key]['n'] += len(G)
            for res, edges in (('coarse', COARSE), ('fine', FINE)):
                for k, (n, v) in peri(ev, t, d, edges).items():
                    acc[key][(res, k, 'n')] = acc[key].get((res, k, 'n'), 0.) + n
                    acc[key][(res, k, 'v')] = acc[key].get((res, k, 'v'), 0.) + v
            if y == 1:
                for hw in ('all', 1, 3):                # the first leg timed by websocket (1) or by a book drop (3)
                    m = np.ones(len(G), bool) if hw == 'all' else (G.how.to_numpy() == hw)
                    if not m.any():
                        continue
                    acc[key][f'n_m50_{hw}'] += int(m.sum())
                    for clk, tt in (('m50', G.t_m50.to_numpy(float)[m]), ('fire', t[m])):
                        for k, (n, v) in peri(ev, tt, d[m], FINE).items():
                            acc[key][(f'{clk}_{hw}', k, 'n')] = acc[key].get((f'{clk}_{hw}', k, 'n'), 0.) + n
        Gok = Pd[ok].copy()
        t, d = Gok.t.to_numpy(float), Gok.d.to_numpy(float)
        for k, x in per_point(ev, t, d, since_last).items():
            Gok[f'since_{k}'] = x
        for w in WINS:
            cnt_w = per_point(ev, t, d, lambda E, tt: count_in(E, tt, w))
            for role, kind in PATTERN:
                Gok[f'in{int(w)}_{role}_{kind}'] = cnt_w[f'{role}_{kind}'] > 0
        for c in Gok.columns:
            if Gok[c].dtype == np.float64 and c not in ('t', 't_m50', 'd'):
                Gok[c] = Gok[c].astype(np.float32)
        rows.append(Gok)
        cnt = Pd.groupby(['grp', 'y', 'status']).size()
        acc['status'][day] = {f'{g}|{y}|{s}': int(v) for (g, y, s), v in cnt.items()}
        print(day, len(Pd), int(ok.sum()), flush=True)
    X = pd.concat(rows, ignore_index=True)
    out = dict(asof='2026-11-26', days=days, coarse_edges_ms=COARSE.tolist(), fine_edges_ms=FINE.tolist(), lags_ms=list(LAGS), windows_ms=list(WINS),
               status=dict(acc['status']), groups={})
    for g in ('TOP1', 'TOP2_10'):
        s, n = acc[f'{g}|1'], acc[f'{g}|0']
        if not s.get('n') or not n.get('n'):
            continue
        r = dict(sends=int(s['n']), random=int(n['n']))
        for res, edges in (('coarse', COARSE), ('fine', FINE)):
            w = np.diff(edges)
            r[f'rates_{res}'] = {f'{role}_{kind}': dict(send_n_per_ms=(s[(res, f'{role}_{kind}', 'n')] / s['n'] / w).tolist(),
                                                        random_n_per_ms=(n[(res, f'{role}_{kind}', 'n')] / n['n'] / w).tolist(),
                                                        send_btc_per_ms=(s[(res, f'{role}_{kind}', 'v')] / s['n'] / w).tolist(),
                                                        random_btc_per_ms=(n[(res, f'{role}_{kind}', 'v')] / n['n'] / w).tolist())
                                 for role in ROLES for kind in KINDS}
        tw = lambda src, key, nn: sum(np.asarray(src[(key, f'{ro}_{ki}', 'n')]) for ro, ki in TOWARD) / nn
        base = tw(n, 'fine', n['n'])
        i0, i1 = np.searchsorted(FINE, -20.), np.searchsorted(FINE, 0.)
        r['toward_fine'] = dict(random=base.tolist())
        r['toward_excess_last20ms'] = {}
        for hw in ('all', 1, 3):
            if not s.get(f'n_m50_{hw}'):
                continue
            pf, pm = tw(s, f'fire_{hw}', s[f'n_m50_{hw}']), tw(s, f'm50_{hw}', s[f'n_m50_{hw}'])
            r['toward_fine'][f'around_fire_{hw}'], r['toward_fine'][f'around_m50_{hw}'] = pf.tolist(), pm.tolist()
            r['toward_excess_last20ms'][str(hw)] = dict(sends=int(s[f'n_m50_{hw}']), fire=float((pf - base)[i0:i1].sum()), m50=float((pm - base)[i0:i1].sum()))
        e = r['toward_excess_last20ms']
        r['fire_clock_sharper'] = {k: v['fire'] > v['m50'] for k, v in e.items()}
        G = X[X.grp == g]
        S, N = G[G.y == 1], G[G.y == 0]
        r['since_last_ms'] = {k[6:]: dict(send=qs(S[k]), random=qs(N[k]), send_within={str(int(w)): mean(S[k] <= w) for w in (5., 10., 20., 50., 100.)},
                                         random_within={str(int(w)): mean(N[k] <= w) for w in (5., 10., 20., 50., 100.)})
                              for k in G.columns if k.startswith('since_')}
        r['patterns'] = {}
        for w in WINS:
            code = sum(G[f'in{int(w)}_{ro}_{ki}'].to_numpy().astype(int) << i for i, (ro, ki) in enumerate(PATTERN))
            cs, cn = code[(G.y == 1).to_numpy()], code[(G.y == 0).to_numpy()]
            tab = {}
            for cd in sorted(set(code.tolist())):
                ms, mn = cs == cd, cn == cd
                name = '+'.join(f'{ro}_{ki}' for i, (ro, ki) in enumerate(PATTERN) if cd >> i & 1) or 'none'
                tab[name] = dict(share_sends=mean(ms), share_random=mean(mn), lift=mean(ms) / mean(mn) if mean(mn) else None,
                                 hit_send=mean(S.hit150.to_numpy()[ms]) if ms.any() else None, hit_random=mean(N.hit150.to_numpy()[mn]) if mn.any() else None,
                                 v1_send=mean(S.v1.to_numpy()[ms]) if ms.any() else None, v1_n=int(np.isfinite(S.v1.to_numpy()[ms]).sum()))
            r['patterns'][str(int(w))] = dict(sorted(tab.items(), key=lambda kv: -(kv[1]['share_sends'] or 0.)))
        r['trajectory'] = {k: dict(send_mean=mean(S[k]), random_mean=mean(N[k]), send_median=float(np.nanmedian(S[k])), random_median=float(np.nanmedian(N[k])))
                           for k in G.columns if k.split('_')[0] in ('imb', 'opp', 'near') and k[-1].isdigit()}
        r['hit150'] = dict(send=mean(S.hit150), random=mean(N.hit150))
        out['groups'][g] = r
    TR.dump(fj, out)
    for g, r in out['groups'].items():
        print(g, r['sends'], r['random'], r['hit150'], r['toward_excess_last20ms'], r['fire_clock_sharper'], flush=True)


def selftest():
    """Synthetic book / trades through book_events, trade_events, the role mapping and window_sums; a flipped-role mutant must fail."""
    global side_of
    ix = DC.build_index(pd.DataFrame(dict(E=[100, 300, 300], id=[1, 2, 3], p=[100.0, 100.0, 99.9], q=[2., 10., 1.], m=[0, 0, 1])).astype({'E': np.int64, 'id': np.int64, 'm': np.int64}), 0)
    rows = np.array([(0, 99.9, 5., 100.0, 10.), (100, 99.9, 5., 100.0, 7.), (200, 99.9, 8., 100.0, 9.), (300, 99.9, 7., 100.1, 4.), (400, 100.0, 1., 100.1, 4.)])
    bk = tuple(rows[:, i] for i in range(5))
    ev = book_events(bk, ix)
    want = {('ask', 'cancel'): [(100, 1.), (300, 0.)], ('ask', 'add'): [(200, 2.)], ('ask', 'gone'): [(300, 9.)], ('bid', 'add'): [(200, 3.)],
            ('bid', 'improve'): [(400, 1.)], ('bid', 'cancel'): [], ('ask', 'improve'): [], ('bid', 'gone'): []}
    # t=100: ask 10 -> 7 with 2 traded at 100.0: cancel 1; t=200: ask 7 -> 9: add 2, bid 5 -> 8: add 3; t=300: ask 100.0 -> 100.1 with 10 traded
    # at 100.0 (rest 9 - 10 < 0: no cancel), gone 9; bid 8 -> 7 with 1 sold at 99.9: r = 0, no event; t=400: bid 99.9 -> 100.0: improve 1.
    want[('ask', 'cancel')] = [(100, 1.)]
    for k, v in want.items():
        e, a = ev[k]
        C.require(len(e) == len(v) and all(abs(e[i] - v[i][0]) < 1e-9 and abs(a[i] - v[i][1]) < 1e-9 for i in range(len(v))), f'selftest {k}: {list(zip(e, a))} want {v}')
    tr = trade_events(ix)
    C.require(list(tr[('ask', 'trade')][0]) == [100., 300.] and list(tr[('bid', 'trade')][0]) == [300.], 'selftest: trade events')
    C.require(side_of('opp', 1.) == 'ask' and side_of('opp', -1.) == 'bid' and side_of('near', 1.) == 'bid', 'selftest: role mapping')
    E = np.array([10., 20., 20., 35.])
    cA = np.r_[0., np.cumsum([1., 2., 3., 4.])]
    n, v = window_sums(E, cA, np.array([30.]), np.array([-30., -15., 0., 10.]))
    C.require(list(n) == [1., 2., 1.] and list(v) == [1., 5., 4.], f'selftest window_sums: {n} {v}')
    C.require(abs(since_last(E, np.array([30.]))[0] - 10.) < 1e-9 and count_in(E, np.array([30.]), 15.)[0] == 2, 'selftest since_last / count_in')
    ev2 = {('ask', 'cancel'): (np.array([5.]), np.array([0., 1.])), ('bid', 'cancel'): (np.array([7.]), np.array([0., 1.]))}
    for k in KINDS:
        for sd in ('ask', 'bid'):
            ev2.setdefault((sd, k), (np.array([]), np.array([0.])))
    got = per_point(ev2, np.array([10., 10.]), np.array([1., -1.]), since_last)['opp_cancel']
    C.require(list(got) == [5., 3.], f'selftest per_point roles: {got}')
    orig = side_of
    side_of = lambda role, d: 'ask' if (role == 'opp') == (d < 0) else 'bid'           # mutant: flipped roles
    try:
        bad = per_point(ev2, np.array([10., 10.]), np.array([1., -1.]), since_last)['opp_cancel']
    finally:
        side_of = orig
    C.require(list(bad) != [5., 3.], 'selftest: the flipped-role mutant was not detected')
    print('selftest ok')


if __name__ == '__main__':
    dict(selftest=selftest, run=lambda: run([]), rehearse=lambda: run(['rehearse'] + sys.argv[2:]))[sys.argv[1]]()
