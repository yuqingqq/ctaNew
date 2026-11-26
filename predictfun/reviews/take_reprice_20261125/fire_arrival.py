"""What had happened on Binance when the winning takers' orders reached Predict? (USER 2026-11-26: "we first need to get the time when the
takers fire, then we check what happened, to figure out how fast we need to be ... detect the signals early enough"). Follow-up of
fire_events.py: the same points (TOP1 all bursts; TOP2_10 the same seeded 20,000-burst sample; null = shifts of +-5 .. 30 s, same d), read
at the estimated arrival a = fire - c (c = fire_events' arrival offset: the median settlement lag over the book-stamped trade on websocket-
timed fills, per group), instead of at the fire time. Four consumed days; no strategy change.
Readings per group, sends vs null (point estimates):
(1) depletion_cause's split of the opposite best level over (a - 300, a] (traded at that price vs withdrawn net of re-adds; level kept /
    gone), its AUC parts, and the same split on depletion_cause's M50 send times for the same orders (matched by day, market and M50 send
    time), all and by how the first leg was timed (websocket / book drop);
(2) time since the last event of each kind at a (fire_events' event definitions) and the share within 5 / 10 / 20 / 50 / 100 ms;
(3) which of opp trade / opp cancel / opp gone / near add occurred in the last 10 / 25 / 50 / 100 ms before a: share of sends and of random
    times per combination, lift, and the Binance mid moving their way within 150 ms after a;
(4) imbalance, opposite and near best size at a - 300 .. a.
Output (write-once): fire_arrival.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/fire_arrival.py [<rehearsal dir> <day>]
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
import depletion_cause as DC  # noqa: E402
import fire_events as FE  # noqa: E402

C, TR, WT = DC.C, DC.TR, DC.WT
OUT = WT.OUT
WINS = (10., 25., 50., 100.)
WITHIN = (5., 10., 20., 50., 100.)


def mean(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else None


def auc(y, x):
    y, x = np.asarray(y), np.asarray(x, float)
    ok = np.isfinite(x)
    return float(roc_auc_score(y[ok], x[ok])) if ok.sum() >= 100 and len(set(y[ok])) == 2 else None


def split_block(Z, y_col='y'):
    """depletion_cause's split read on rows Z (pre_* columns): sends vs null means, traded share, kept / gone, AUC of the parts."""
    o = {}
    for y, name in ((1, 'send'), (0, 'null')):
        A = Z[(Z[y_col] == y) & Z.pre_st.isin([DC.SAME, DC.GONE])]
        sh = A.pre_Q - A.pre_left
        o[name] = dict(n=int(len(A)), mean_size_btc=mean(A.pre_Q), mean_shrink_btc=mean(sh), mean_traded_btc=mean(A.pre_V), mean_withdrawn_btc=mean(A.pre_W),
                       traded_share=float(A.pre_V.sum() / sh.sum()) if len(A) and sh.sum() > 0 else None, share_gone=mean(A.pre_st == DC.GONE))
        for st in (DC.SAME, DC.GONE):
            K = A[A.pre_st == st]
            s2 = K.pre_Q - K.pre_left
            o[name][DC.ST_NAMES[st]] = dict(n=int(len(K)), traded_share=float(K.pre_V.sum() / s2.sum()) if len(K) and s2.sum() > 0 else None)
    A = Z[Z.pre_st.isin([DC.SAME, DC.GONE])]
    o['auc'] = dict(traded_frac=auc(A[y_col], A.pre_V / A.pre_Q), withdrawn_frac=auc(A[y_col], A.pre_W / A.pre_Q))
    return o


def run(argv):
    resource.setrlimit(resource.RLIMIT_CPU, (1800, 1801))
    if argv:
        days, fj = [argv[1]], Path(argv[0]) / 'fire_arrival.json'
        C.require(days[0] in WT.DAYS, f'{days[0]} is not a declared day')
    else:
        days, fj = list(WT.DAYS), H / 'fire_arrival.json'
    C.require(not fj.exists(), 'fire_arrival.json exists (write-once)')
    FJ = json.loads((H / 'fire_events.json').read_text())
    c_of = {g: float(FJ['groups'][g]['arrival_offset_ms']) for g in ('TOP1', 'TOP2_10')}
    A = pd.read_parquet(OUT / 'fire_times_orders.parquet')
    A = A[A.t_set.notna()].copy()
    A['grp'] = np.where(A['rank'] == 1, 'TOP1', 'TOP2_10')
    B = A.sort_values(['burst', 'fire'], kind='stable').drop_duplicates('burst', keep='first').reset_index(drop=True)
    n2 = int((B.grp == 'TOP2_10').sum())
    if n2 > FE.CAP_2_10:                                 # the same sample as fire_events
        keep = np.zeros(len(B), bool)
        keep[(B.grp == 'TOP1').to_numpy()] = True
        keep[np.random.default_rng(WT.SEED).choice(np.flatnonzero((B.grp == 'TOP2_10').to_numpy()), FE.CAP_2_10, replace=False)] = True
        B = B[keep].reset_index(drop=True)
    B = B[B.day.isin(days)].reset_index(drop=True)
    B['a'] = B.fire - B.grp.map(c_of)
    pts = [pd.DataFrame(dict(day=B.day, grp=B.grp, y=1, s=B.a, d=B.d.astype(float), mid=B.mid, s_m50=B.s_m50, how=B.how2_first))]
    pts += [pd.DataFrame(dict(day=B.day, grp=B.grp, y=0, s=B.a + sh, d=B.d.astype(float), mid=B.mid, s_m50=np.nan, how=0)) for sh in WT.SHIFTS]
    P = pd.concat(pts, ignore_index=True)
    d0 = P.day.map({x: float(WT.OD_day_ms(x)) for x in days})
    P = P[P.s.between(d0 + 2000., d0 + 86400000. - 2000.)].reset_index(drop=True)
    ck = DC.process(P, days)                              # the split over (a - 300, a] and the hold (a, a + 150]
    C.require(len(ck['brute']) and max(ck['brute']) < 1e-9, 'V differs from the brute-force sum')
    rows = []
    for day in days:
        Pd = P[P.day == day].reset_index(drop=True)
        ev, st, traj, hit = FE.day_data(day, Pd.assign(t=Pd.s))
        Pd['ev_status'], Pd['hit150'] = st, hit
        for k, v in traj.items():
            Pd[k] = v
        t, d = Pd.s.to_numpy(float), Pd.d.to_numpy(float)
        for k, x in FE.per_point(ev, t, d, FE.since_last).items():
            Pd[f'since_{k}'] = x
        for w in WINS:
            cnt = FE.per_point(ev, t, d, lambda E, tt: FE.count_in(E, tt, w))
            for ro, ki in FE.PATTERN:
                Pd[f'in{int(w)}_{ro}_{ki}'] = cnt[f'{ro}_{ki}'] > 0
        rows.append(Pd)
        print(day, len(Pd), int((Pd.ev_status == 'ok').sum()), flush=True)
    X = pd.concat(rows, ignore_index=True)
    DCP = pd.read_parquet(DC.OUT / 'depletion_cause_points.parquet', columns=['day', 'mid', 'y', 'grp', 's', 'pre_st', 'pre_Q', 'pre_left', 'pre_V', 'pre_W'])
    DCP = DCP[(DCP.y == 1)].drop_duplicates(['day', 'mid', 's', 'grp'])
    out = dict(asof='2026-11-26', days=days, arrival_offset_ms=c_of, windows_ms=list(WINS), checks=dict(brute_points=len(ck['brute']), brute_max=max(ck['brute']),
               tape=ck['tape'], reconcile_unreflected={k: v['unreflected_share'] for k, v in ck['reconcile'].items()}), groups={})
    for g in ('TOP1', 'TOP2_10'):
        G = X[(X.grp == g) & (X.ev_status == 'ok')]
        S, N = G[G.y == 1], G[G.y == 0]
        r = dict(sends=int(len(S)), random=int(len(N)), hit150_after_arrival=dict(send=mean(S.hit150), random=mean(N.hit150)))
        r['split_at_arrival'] = split_block(G)
        # the same sends on both clocks (arrival vs the M50 send time of depletion_cause), all and by how the first leg was timed
        M = S.merge(DCP[DCP.grp == g].rename(columns={'s': 's_m50', 'pre_st': 'm_st', 'pre_Q': 'm_Q', 'pre_left': 'm_left', 'pre_V': 'm_V', 'pre_W': 'm_W'})
                    .drop(columns=['y', 'grp']), on=['day', 'mid', 's_m50'], how='inner', validate='many_to_one')
        r['same_orders'] = {}
        for h in ('all', 1, 3):
            Mh = M if h == 'all' else M[M.how == h]
            ka = Mh.pre_st.isin([DC.SAME, DC.GONE]) & Mh.m_st.isin([DC.SAME, DC.GONE])
            K = Mh[ka]
            sa, sm = (K.pre_Q - K.pre_left), (K.m_Q - K.m_left)
            r['same_orders'][str(h)] = dict(n=int(len(K)), arrival=dict(mean_traded_btc=mean(K.pre_V), mean_withdrawn_btc=mean(K.pre_W), traded_share=float(K.pre_V.sum() / sa.sum()) if sa.sum() > 0 else None,
                                                                         share_gone=mean(K.pre_st == DC.GONE)),
                                            m50=dict(mean_traded_btc=mean(K.m_V), mean_withdrawn_btc=mean(K.m_W), traded_share=float(K.m_V.sum() / sm.sum()) if sm.sum() > 0 else None,
                                                     share_gone=mean(K.m_st == DC.GONE)))
        r['since_last_at_arrival_ms'] = {k[6:]: dict(send={f'p{q}': float(np.nanpercentile(S[k], q)) for q in (25, 50, 75)}, random={f'p{q}': float(np.nanpercentile(N[k], q)) for q in (25, 50, 75)},
                                                     send_within={str(int(w)): mean(S[k] <= w) for w in WITHIN}, random_within={str(int(w)): mean(N[k] <= w) for w in WITHIN})
                                         for k in G.columns if k.startswith('since_')}
        r['patterns_at_arrival'] = {}
        for w in WINS:
            code = sum(G[f'in{int(w)}_{ro}_{ki}'].to_numpy().astype(int) << i for i, (ro, ki) in enumerate(FE.PATTERN))
            cs, cn = code[(G.y == 1).to_numpy()], code[(G.y == 0).to_numpy()]
            hs, hn = S.hit150.to_numpy(), N.hit150.to_numpy()
            tab = {}
            for cd in sorted(set(code.tolist())):
                ms, mn = cs == cd, cn == cd
                name = '+'.join(f'{ro}_{ki}' for i, (ro, ki) in enumerate(FE.PATTERN) if cd >> i & 1) or 'none'
                tab[name] = dict(share_sends=mean(ms), share_random=mean(mn), lift=mean(ms) / mean(mn) if mean(mn) else None,
                                 hit_send=mean(hs[ms]) if ms.any() else None, hit_random=mean(hn[mn]) if mn.any() else None)
            any_t = {f'{ro}_{ki}': dict(share_sends=mean(S[f'in{int(w)}_{ro}_{ki}']), share_random=mean(N[f'in{int(w)}_{ro}_{ki}'])) for ro, ki in FE.PATTERN}
            r['patterns_at_arrival'][str(int(w))] = dict(combos=dict(sorted(tab.items(), key=lambda kv: -(kv[1]['share_sends'] or 0.))), any_of_kind=any_t)
        r['trajectory_at_arrival'] = {k: dict(send_mean=mean(S[k]), random_mean=mean(N[k]), send_median=float(np.nanmedian(S[k])), random_median=float(np.nanmedian(N[k])))
                                      for k in G.columns if k.split('_')[0] in ('imb', 'opp', 'near') and k[-1].isdigit()}
        out['groups'][g] = r
    TR.dump(fj, out)
    for g, r in out['groups'].items():
        print(g, r['sends'], json.dumps(r['split_at_arrival']['send']), json.dumps(r['same_orders']), flush=True)


if __name__ == '__main__':
    run(sys.argv[1:])
