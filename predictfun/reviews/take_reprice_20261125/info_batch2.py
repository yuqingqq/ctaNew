"""The remaining information we hold, at the top taker's fires (USER 2026-11-26: "Yes check items 1-6" of the unused-information review).
Readout on the four consumed days; no strategy change. Points, the lead-test population (no BTC futures mid change in the previous 300 ms;
a BTC move its way within 300 ms after) and the flags are fire_info's (FIRE / NEARBY / RANDOM).
Declared before the run:
1 activity: per market of the four days (registry), its fires against the market's conditions: Binance futures BTC realised volatility
  over the window (bps, from the change points) and mid changes, futures traded volume (BTC), Predict traded volume (every taker's legs),
  Predict median spread (ticks) and median best-level size (shares) over the window (step1 book frames, the day's own hour files only).
  Over the three days it traded: Spearman correlation of fires per market with each condition, in all markets and in its active hours
  (UTC hours with >= 1 fire); fires per market and the share of markets with a fire by condition quintile; fires per UTC hour per day.
2 other coins: Binance futures ETH, SOL, XRP, BNB, DOGE mid change points (bookTicker, E clock) and Hyperliquid ETH, SOL mids (bbo, receipt
  clock): return its way over 300 ms / 1 s at every point; the lead test (moved its way first: > 0 over the previous 300 ms), FIRE vs
  NEARBY; AUCs.
3 Polymarket: the same-window BTC 5m market (slug btc-updown-5m-<window start>), Up-token best bid / ask from book and price_change
  messages (Polymarket server ms stamps), on a seeded sample of 300 of the markets with its fires (bounded CPU): mid return its way over
  300 ms / 1 s (c); the lead test (moved first = >= 0.5 c); AUCs. Polymarket resolves on a Chainlink 60 s TWAP, Predict on the top-of-book
  mid: same windows, different settlement.
4 window opening: fire density by seconds since the window opened (0-5, 5-15, 15-30, 30-60, 60-120, 120-240, 240-300) against random
  moments; the spot distance from the start price at fires by phase; the top taker's 1 s value per share by phase (stale_quotes' fill rows
  of its legs, aligned to legs_of_day row by row, prices and shares checked).
5 counterparties: maker signers of every leg (step1 legs, joined to their taker on winners_legs' key, duplicates of the key dropped and
  counted): per taker group (top taker, ranks 2-10, others) the number of makers and the top 1 / 5 / 10 makers' share of its volume; per
  maker of the top taker's ten largest: its share of the top taker's volume, of ranks 2-10's and of all takers' (lift = top / all).
6 order among the top takers: per fire of the top taker in the complete markets (fire_times_orders' rank-1 bursts), same-direction orders of
  ranks 2-10 in the same market (arrival = settlement - 184 ms): the share with one 20-1000 ms before / 20-1000 ms after, against its fire
  times shifted by +-5..30 s (12 per fire, inside the window); lags within +-1 s in 50 ms bins; per rank.
Outputs (write-once): OUT/info2/<day>_coins.parquet, <day>_pm.parquet; info2.json (merge, which computes 1, 4, 5, 6).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/info_batch2.py \
       selftest | coins <day> [<dir>] | pm <day> [<dir> <n markets>] | merge [<dir>]
"""
import collections
import gzip
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import fire_info as FI  # noqa: E402
import fire_times as FT  # noqa: E402

CV, C, TR, WT, OD, MM, VR, SQ = FI.CV, FI.C, FI.TR, FI.WT, FI.OD, FI.MM, FI.VR, FI.SQ
DATA = FI.DATA
OUT = WT.OUT / 'info2'
DAYS = FI.DAYS
TRADED = ('20261031', '20261101', '20261102')
COINS_BN = ('ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT')
COINS_HL = ('ETH', 'SOL')
PMRAW, PMMK = DATA / 'pm_5min/raw', DATA / 'pm_5min/markets.jsonl'
N_PM, SEED = 300, 20261126
OPEN_B = [0., 5., 15., 30., 60., 120., 240., 300.]
SHIFTS = FI.SHIFTS
KEY = ['mid', 'exec_s', 'first_recv_ns', 'side', 'ptick', 'shares', 'price_executed']


def day_stems(day):
    out = [f'{day}_{h:02d}' for h in range(24)]
    for s in out:
        C.require(CV.allowed(FI.iso_of_stem(s)), s)
    return out


def bn_changes(sym, day):
    """Mid change points (E clock) of one symbol-day, reduced per chunk of each hour file (bounded memory)."""
    ks, ms = [], []
    names = ['recv', 'E', 'T', 'u', 'bid', 'bq', 'ask', 'aq']
    for s in day_stems(day):
        p = DATA / 'mm_hf/raw/bookTicker' / sym / f'{s}.csv.gz'
        if not p.exists():
            continue
        for ch in pd.read_csv(p, header=None, names=names, usecols=['E', 'u', 'bid', 'ask'], on_bad_lines='skip', dtype=str, chunksize=200000):
            df = ch.apply(pd.to_numeric, errors='coerce').dropna()
            df = df[(df.bid > 0) & (df.ask > df.bid)].drop_duplicates('u')
            e, m = df.E.to_numpy(float), ((df.bid + df.ask) / 2.).to_numpy(float)
            o = np.argsort(e, kind='stable'); e, m = e[o], m[o]
            keep = np.r_[True, np.diff(m) != 0]
            ks.append(e[keep]); ms.append(m[keep])
        C.require(FI.F.rss_mib() < FI.F.RSS_GUARD_MIB, 'RSS over the self-guard')
    k, m = np.concatenate(ks), np.concatenate(ms)
    o = np.argsort(k, kind='stable'); k, m = k[o], m[o]
    keep = np.r_[True, np.diff(m) != 0]
    return k[keep], m[keep]


def hl_coin(day, coin):
    parts = []
    for s in day_stems(day):
        p = FI.HLD / 'bbo' / coin / f'{s}.csv.gz'
        if p.exists():
            parts.append(pd.read_csv(p, header=None, names=['recv', 'hl', 'bp', 'bs', 'ap', 'as_']))
    D = pd.concat(parts, ignore_index=True)
    D['t'] = D.recv / 1e6
    return D.sort_values('t', kind='stable').reset_index(drop=True)


def coins(day, od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (1500, 1501))
    C.require(day in DAYS, f'{day} is not a declared day')
    od = Path(od) if od else OUT
    fp = od / f'{day}_coins.parquet'
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    Z = pd.read_parquet(FI.OUT / f'{day}_feats.parquet', columns=['kind', 'pid', 'day', 'mid', 't', 'd', 'fut_quiet_300', 'y300'])
    t, d = Z.t.to_numpy(float), Z.d.to_numpy(float)
    X = {}
    for sym in COINS_BN:
        k, m = bn_changes(sym, day)
        cur = FI.at(k, m, t)
        for w in (300., 1000.):
            X[f'{sym}_ret_{int(w)}'] = d * 1e4 * np.log(cur / FI.at(k, m, t - w))
    for coin in COINS_HL:
        hb = hl_coin(day, coin)
        ht, hm = hb.t.to_numpy(float), ((hb.bp + hb.ap) / 2.).to_numpy(float)
        cur = FI.at(ht, hm, t, 2000.)
        for w in (300., 1000.):
            X[f'hl_{coin}_ret_{int(w)}'] = d * 1e4 * np.log(cur / FI.at(ht, hm, t - w, 2000.))
    od.mkdir(parents=True, exist_ok=True)
    pd.concat([Z, pd.DataFrame(X)], axis=1).to_parquet(fp, index=False)
    print(day, len(Z), flush=True)


def pm_up_tokens():
    up = {}
    with open(PMMK) as f:
        for line in f:
            if '"btc-updown-5m-' not in line:
                continue
            r = json.loads(line)
            if r.get('coin') != 'btc':
                continue
            outs = json.loads(r['outcomes']) if isinstance(r['outcomes'], str) else r['outcomes']
            if 'Up' in outs:
                up[r['slug']] = str(r['clobTokenIds'][outs.index('Up')])
    return up


def pm_series(path, up, lo, hi):
    """Up-token (Polymarket stamp ms, mid c) from book and price_change messages received within [lo - 5 s, hi + 5 s]."""
    ts, mids = [], []
    with gzip.open(path, 'rt') as f:
        for line in f:
            tab = line.find('\t')
            if tab < 0:
                continue
            try:
                rv = int(line[:tab]) / 1e6
            except ValueError:
                continue
            if rv < lo - 5000. or rv > hi + 5000.:
                continue
            try:
                arr = json.loads(line[tab + 1:])
            except ValueError:
                continue
            for msg in (arr if isinstance(arr, list) else [arr]):
                if not isinstance(msg, dict):
                    continue
                et = msg.get('event_type')
                if et == 'price_change':
                    for c in msg.get('price_changes') or []:
                        if str(c.get('asset_id')) == up and c.get('best_bid') not in (None, '') and c.get('best_ask') not in (None, ''):
                            b, a = float(c['best_bid']), float(c['best_ask'])
                            if 0 < b < a < 1:
                                ts.append(float(msg['timestamp'])); mids.append(50. * (a + b))
                elif et == 'book' and str(msg.get('asset_id')) == up:
                    bids = [float(x['price']) for x in msg.get('bids') or []]
                    asks = [float(x['price']) for x in msg.get('asks') or []]
                    if bids and asks and 0 < max(bids) < min(asks) < 1:
                        ts.append(float(msg['timestamp'])); mids.append(50. * (max(bids) + min(asks)))
    t, m = np.asarray(ts, float), np.asarray(mids, float)
    o = np.argsort(t, kind='stable')
    return t[o], m[o]


def pm(day, od=None, n=None):
    resource.setrlimit(resource.RLIMIT_CPU, (1500, 1501))
    C.require(day in DAYS, f'{day} is not a declared day')
    od = Path(od) if od else OUT
    fp = od / f'{day}_pm.parquet'
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    P = pd.read_parquet(FI.OUT / 'points.parquet')
    mk = np.sort(P[P.kind == 'FIRE'].mid.unique())
    rng = np.random.default_rng(SEED)
    pick = set(rng.choice(mk, min(N_PM, len(mk)), replace=False).tolist())
    Z = pd.read_parquet(FI.OUT / f'{day}_feats.parquet', columns=['kind', 'pid', 'day', 'mid', 'ws_ms', 't', 'd', 'fut_quiet_300', 'y300'])
    Z = Z[Z.mid.isin(pick) & Z.kind.isin(['FIRE', 'NEARBY'])].reset_index(drop=True)
    up = pm_up_tokens()
    X = {f'pm_ret_{w}': np.full(len(Z), np.nan) for w in (300, 1000)}
    status = collections.Counter()
    mids = Z.mid.unique()
    if n:
        mids = mids[:n]
    for m in mids:
        ix = np.flatnonzero(Z.mid.to_numpy() == m)
        ws = float(Z.ws_ms.iat[ix[0]])
        slug = f'btc-updown-5m-{int(ws // 1000)}'
        dd = pd.Timestamp(ws, unit='ms', tz='UTC').strftime('%Y%m%d')
        C.require(CV.allowed(FI.iso_of_stem(dd)), dd)
        p = PMRAW / dd / f'{slug}.jsonl.gz'
        if slug not in up or not p.exists():
            status['pm_missing'] += 1
            continue
        tt, mm = pm_series(p, up[slug], ws, ws + 300000.)
        if len(tt) < 10:
            status['pm_too_few'] += 1
            continue
        status['pm_ok'] += 1
        t, d = Z.t.to_numpy(float)[ix], Z.d.to_numpy(float)[ix]
        cur = FI.at(tt, mm, t, 5000.)
        for w in (300, 1000):
            X[f'pm_ret_{w}'][ix] = d * (cur - FI.at(tt, mm, t - w, 5000.))
    od.mkdir(parents=True, exist_ok=True)
    pd.concat([Z, pd.DataFrame(X)], axis=1).to_parquet(fp, index=False)
    print(day, dict(status), len(Z), flush=True)


# ---------------------------------------------------------------- merge (1, 4, 5, 6 computed here)
def lead_table(Z, col, thr):
    r = {}
    for lab in ('FIRE', 'NEARBY'):
        q = Z[(Z.kind == lab) & Z.fut_quiet_300.astype(bool)]
        for ylab, Y in (('move_after', q[q.y300.astype(bool)]), ('no_move_after', q[~q.y300.astype(bool)])):
            v = pd.to_numeric(Y[col], errors='coerce'); v = v[v.notna()]
            r[f'{lab.lower()}_{ylab}'] = dict(n=int(len(v)), moved_its_way=float((v > thr).mean()) if len(v) else None, moved_against=float((v < -thr).mean()) if len(v) else None)
    return r


def market_conditions(day):
    reg = MM.registry(day).reset_index()
    reg['mid'] = reg.mid.astype(int)
    ws = reg.ws.to_numpy(float) * 1000.
    bk, bm = OD.seg_arrays(OD.SEG_OF[day]); bk = bk.astype(float); bm = bm.astype(float)
    lr = np.r_[0., np.diff(np.log(bm))] ** 2
    cs = np.r_[0., np.cumsum(lr)]
    a, b = np.searchsorted(bk, ws, 'right'), np.searchsorted(bk, ws + 300000., 'right')
    reg['rv_bps'] = 1e4 * np.sqrt(np.maximum(cs[b] - cs[a], 0.))
    reg['n_changes'] = (b - a).astype(float)
    ts, p_, q_, s_, (lo, hi), _ = CV.load('btc_fut', day)
    cq = np.r_[0., np.cumsum(q_)]
    reg['fut_volume'] = np.where((ws >= lo) & (ws + 300000. <= hi), cq[np.searchsorted(ts, ws + 300000., 'right')] - cq[np.searchsorted(ts, ws, 'right')], np.nan)
    W = pd.read_parquet(WT.OUT / f'winners_legs_{day}.parquet', columns=['mid', 'shares'])
    reg['pf_volume'] = reg.mid.map(W.groupby('mid').shares.sum()).fillna(0.)
    parts = []
    for s in day_stems(day):
        pb = FI.S1 / day / 'books' / f'part_{s}.parquet'
        if pb.exists():
            parts.append(pd.read_parquet(pb, columns=['mid', 't_ms', 'b1_p', 'b1_s', 'a1_p', 'a1_s']))
    B = pd.concat(parts, ignore_index=True)
    B = B.merge(reg[['mid', 'ws']], on='mid')
    B = B[(B.t_ms >= B.ws * 1000.) & (B.t_ms < B.ws * 1000. + 300000.) & (B.b1_p >= 0) & (B.a1_p > B.b1_p)]
    g = B.assign(spread=B.a1_p - B.b1_p, best=np.minimum(B.b1_s, B.a1_s)).groupby('mid')
    reg['pf_spread'] = reg.mid.map(g.spread.median())
    reg['pf_best_size'] = reg.mid.map(g.best.median())
    reg['hour'] = pd.to_datetime(reg.ws, unit='s', utc=True).dt.hour
    reg['day'] = day
    return reg[['day', 'mid', 'ws', 'hour', 'rv_bps', 'n_changes', 'fut_volume', 'pf_volume', 'pf_spread', 'pf_best_size']]


def merge(od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (1500, 1501))
    fn = (Path(od) if od else H) / 'info2.json'
    C.require(not fn.exists(), 'info2.json exists (write-once)')
    src = Path(od) if od else OUT
    out = dict(asof='2026-11-26')
    P = pd.read_parquet(FI.OUT / 'points.parquet')
    # 2 / 3 lead tables
    Zc = pd.concat([pd.read_parquet(src / f'{d}_coins.parquet') for d in DAYS], ignore_index=True)
    cc = [c for c in Zc.columns if c.endswith('_ret_300') or c.endswith('_ret_1000')]
    out['coins'] = dict(lead={c.replace('_ret_300', ''): lead_table(Zc, c, 0.) for c in cc if c.endswith('_ret_300')},
                        auc=FI.auc_rows(Zc[Zc.kind == 'FIRE'], Zc[Zc.kind == 'NEARBY'], cc))
    Zp = pd.concat([pd.read_parquet(src / f'{d}_pm.parquet') for d in DAYS if (src / f'{d}_pm.parquet').exists()], ignore_index=True)
    out['polymarket'] = dict(markets=int(Zp[Zp.pm_ret_300.notna()].mid.nunique()), lead=lead_table(Zp, 'pm_ret_300', .5),
                             auc=FI.auc_rows(Zp[Zp.kind == 'FIRE'], Zp[Zp.kind == 'NEARBY'], ['pm_ret_300', 'pm_ret_1000']))
    # 1 activity
    M = pd.concat([market_conditions(d) for d in DAYS], ignore_index=True)
    fz = P[P.kind == 'FIRE']
    M['fires'] = M.mid.map(fz.groupby('mid').size()).fillna(0.)
    T = M[M.day.isin(TRADED)].copy()
    act_hours = set(map(tuple, T[T.fires > 0][['day', 'hour']].drop_duplicates().to_numpy().tolist()))
    T['active_hour'] = [(d_, h_) in act_hours for d_, h_ in zip(T.day, T.hour)]
    conds = ['rv_bps', 'n_changes', 'fut_volume', 'pf_volume', 'pf_spread', 'pf_best_size']
    act = dict(markets=int(len(T)), markets_with_fires=int((T.fires > 0).sum()), active_hours=len(act_hours), spearman={}, quintiles={})
    for c in conds:
        for lab, X in (('all', T), ('active_hours', T[T.active_hour])):
            v = X[[c, 'fires']].dropna()
            rho = spearmanr(v[c], v.fires).correlation if len(v) > 10 else None
            act['spearman'][f'{c}|{lab}'] = dict(rho=float(rho) if rho is not None else None, n=int(len(v)))
        v = T[T.active_hour][[c, 'fires']].dropna()
        qb = pd.qcut(v[c].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        act['quintiles'][c] = {int(k): dict(median=float(G[c].median()), fires_per_market=float(G.fires.mean()), share_with_fire=float((G.fires > 0).mean()), n=int(len(G))) for k, G in v.groupby(qb, observed=True)}
    act['by_hour'] = {d_: {int(h_): float(G.fires.sum()) for h_, G in T[T.day == d_].groupby('hour')} for d_ in TRADED}
    act['by_day'] = {d_: dict(fires=float(M[M.day == d_].fires.sum()), markets=int((M.day == d_).sum()), markets_with_fires=int(((M.day == d_) & (M.fires > 0)).sum()),
                              rv_bps_median=float(M[M.day == d_].rv_bps.median()), pf_volume_median=float(M[M.day == d_].pf_volume.median())) for d_ in DAYS}
    out['activity'] = act
    # 4 window opening
    Zf = pd.concat([pd.read_parquet(FI.OUT / f'{d}_feats.parquet', columns=['kind', 'mid', 'day', 't', 'ws_ms', 'tau_s', 'dist_its_bps']) for d in DAYS], ignore_index=True)
    Zf['since_open'] = 300. - Zf.tau_s
    ob = lambda X: pd.cut(X.since_open, OPEN_B, include_lowest=True, right=False)
    F_, R_ = Zf[Zf.kind == 'FIRE'], Zf[Zf.kind == 'RANDOM']
    sf, sr = F_.groupby(ob(F_), observed=False).size() / len(F_), R_.groupby(ob(R_), observed=False).size() / len(R_)
    opn = dict(density_ratio={str(k): float(sf[k] / sr[k]) if sr[k] > 0 else None for k in sf.index}, fires_share={str(k): float(sf[k]) for k in sf.index},
               dist_abs_median={str(k): dict(fires=float(G.dist_its_bps.abs().median()), random=float(R_[ob(R_) == k].dist_its_bps.abs().median())) for k, G in F_.groupby(ob(F_), observed=False) if len(G)},
               ahead={str(k): dict(fires=float((G.dist_its_bps > 0).mean()), random=float((R_[ob(R_) == k].dist_its_bps > 0).mean())) for k, G in F_.groupby(ob(F_), observed=False) if len(G)})
    vs = json.loads((VR.OUT / 'summary.json').read_text())
    vals, mism = [], 0
    for d_ in DAYS:
        comp = {int(r['mid']) for r in vs['rows'] if r['day'] == d_ and r['status'] == 'complete'}
        LG = SQ.legs_of_day(d_, comp)
        Fd = pd.read_parquet(SQ.OUT / f'{d_}_fills.parquet', columns=['grp', 'mid', 'price', 'q', 'v1'])
        Fd = Fd[Fd.grp == 'TOP']
        L1 = LG[LG.grp == 'TOP']
        reg = MM.registry(d_).reset_index(); wsm = dict(zip(reg.mid.astype(int), reg.ws.astype(float) * 1000.))
        for m_, g in L1.groupby('mid', sort=False):
            f_ = Fd[Fd.mid == m_]
            if len(f_) != len(g) or not (np.allclose(f_.price.to_numpy(), g.price.to_numpy()) and np.allclose(f_.q.to_numpy(), g.q.to_numpy())):
                mism += 1
                continue
            vals.append(pd.DataFrame(dict(since_open=(g.t_abs.to_numpy() - wsm[int(m_)]) / 1000., v1=f_.v1.to_numpy(), q=f_.q.to_numpy())))
    V = pd.concat(vals, ignore_index=True)
    C.require(mism == 0, f'{mism} markets where the fill rows do not align with their legs')
    V = V[V.v1.notna() & (V.since_open >= 0) & (V.since_open < 300.)]
    vb = pd.cut(V.since_open, OPEN_B, include_lowest=True, right=False)
    opn['value_1s_by_phase'] = {str(k): dict(v1=float((G.v1 * G.q).sum() / G.q.sum()), shares=float(G.q.sum()), legs=int(len(G))) for k, G in V.groupby(vb, observed=False) if len(G)}
    out['opening'] = opn
    # 5 counterparties
    top10 = FT.top10()
    TOP = SQ.XT.TOP.lower()
    C.require(top10[0].lower() == TOP, 'rank 1 is not the top taker')
    r210 = {a.lower() for a in top10[1:]}
    parts, dups = [], 0
    for d_ in DAYS:
        reg = set(MM.registry(d_).reset_index().mid.astype(int))
        S = pd.read_parquet(FI.S1 / d_ / 'legs.parquet', columns=KEY + ['signer'])
        Wl = pd.read_parquet(WT.OUT / f'winners_legs_{d_}.parquet', columns=KEY + ['taker', 'ambiguous'])
        n0 = len(S) + len(Wl)
        S = S[S.mid.isin(reg)].drop_duplicates(KEY, keep=False); Wl = Wl[Wl.mid.isin(reg) & ~Wl.ambiguous & Wl.taker.notna()].drop_duplicates(KEY, keep=False)
        dups += n0 - len(S) - len(Wl)
        parts.append(S.merge(Wl, on=KEY, how='inner').assign(day=d_))
    J = pd.concat(parts, ignore_index=True)
    J['taker'] = J.taker.str.lower(); J['maker'] = J.day + ':' + J.signer.astype(str)          # step1 numbers signers per day: a maker is (day, code)
    J['grp'] = np.where(J.taker == TOP, 'TOP', np.where(J.taker.isin(r210), 'RANKS2_10', 'OTHERS'))
    cp = dict(legs_joined=int(len(J)), rows_dropped_or_outside=int(dups), groups={})
    tot = J.groupby('maker').shares.sum(); tot_all = tot.sum()
    for g_, G in J.groupby('grp'):
        s = G.groupby('maker').shares.sum().sort_values(ascending=False)
        pd_ = {d_: (lambda z: dict(top1=float(z.iloc[:1].sum() / z.sum()), top5=float(z.iloc[:5].sum() / z.sum()), top10=float(z.iloc[:10].sum() / z.sum())))(x.groupby('maker').shares.sum().sort_values(ascending=False))
               for d_, x in G.groupby('day') if x.shares.sum() > 0}
        cp['groups'][g_] = dict(maker_days=int(len(s)), shares=float(s.sum()), top1=float(s.iloc[:1].sum() / s.sum()), top5=float(s.iloc[:5].sum() / s.sum()), top10=float(s.iloc[:10].sum() / s.sum()),
                                per_day=pd_)
    st = J[J.grp == 'TOP'].groupby('maker').shares.sum().sort_values(ascending=False)
    sr2 = J[J.grp == 'RANKS2_10'].groupby('maker').shares.sum()
    cp['top_makers'] = [dict(maker_day=m_, share_of_top=float(v / st.sum()), share_of_ranks2_10=float(sr2.get(m_, 0.) / sr2.sum()), share_of_all=float(tot.get(m_, 0.) / tot_all),
                             lift=float((v / st.sum()) / (tot.get(m_, 0.) / tot_all)) if tot.get(m_, 0.) > 0 else None) for m_, v in st.iloc[:10].items()]
    out['counterparties'] = cp
    # 6 order among the top takers
    O = pd.read_parquet(WT.OUT / 'fire_times_orders.parquet', columns=['mid', 'rank', 'burst', 'fire', 't_set', 'd', 'day'])
    O = O[O.t_set.notna()]
    T1 = O[O['rank'] == 1].sort_values('fire').drop_duplicates('burst', keep='first')
    T1 = T1.assign(a=T1.fire - FI.LAG)
    Rk = O[O['rank'] > 1].assign(a=lambda x: x.t_set - FI.HOLD - FI.LAG)
    regs = pd.concat([MM.registry(d_).reset_index()[['mid', 'ws']] for d_ in DAYS]).drop_duplicates('mid')
    wsm = dict(zip(regs.mid.astype(int), regs.ws.astype(float) * 1000.))
    rk_by = {k_: (g.a.to_numpy(float), g.d.to_numpy(float), g['rank'].to_numpy(int)) for k_, g in Rk.groupby('mid')}
    rng = np.random.default_rng(SEED)

    fine = collections.defaultdict(list)

    def scan(mids, aa, dd, keep_fine=False):
        bef, aft, lags, rb = np.zeros(len(aa), bool), np.zeros(len(aa), bool), [], collections.Counter()
        for i, (m_, a_, d0) in enumerate(zip(mids, aa, dd)):
            if m_ not in rk_by:
                continue
            ra, rd, rr = rk_by[m_]
            lg = ra - a_
            sel = (rd == d0) & (np.abs(lg) <= 1000.)
            bef[i] = bool(np.any(sel & (lg >= -1000.) & (lg <= -20.)))
            aft[i] = bool(np.any(sel & (lg >= 20.) & (lg <= 1000.)))
            lags.extend(lg[sel].tolist())
            if keep_fine:
                for r_, l_ in zip(rr[sel], lg[sel]):
                    fine[int(r_)].append(float(l_))
            for r_ in np.unique(rr[sel & (lg >= -1000.) & (lg <= -20.)]):
                rb[f'before_rank{int(r_)}'] += 1
            for r_ in np.unique(rr[sel & (lg >= 20.) & (lg <= 1000.)]):
                rb[f'after_rank{int(r_)}'] += 1
        return bef, aft, np.asarray(lags), rb
    m1, a1, d1 = T1.mid.to_numpy(int), T1.a.to_numpy(float), T1.d.to_numpy(float)
    bef, aft, lags, rb = scan(m1, a1, d1, keep_fine=True)
    nm_, na_, nd_ = [], [], []
    for s in SHIFTS:
        u = rng.uniform(-500., 500., len(a1))
        x = a1 + s + u
        w0 = np.array([wsm.get(int(m_), np.nan) for m_ in m1])
        ok = (x >= w0) & (x < w0 + 300000.)
        nm_.append(m1[ok]); na_.append(x[ok]); nd_.append(d1[ok])
    nb, nf, nlags, _ = scan(np.concatenate(nm_), np.concatenate(na_), np.concatenate(nd_))
    hist = np.histogram(lags, bins=40, range=(-1000., 1000.))[0]
    out['order'] = dict(fires=int(len(T1)), before=float(bef.mean()), after=float(aft.mean()), null_before=float(nb.mean()), null_after=float(nf.mean()), null_n=int(len(nb)),
                        lag_hist_50ms=[int(x) for x in hist],
                        per_rank_fine={r_: dict(pairs_1s=int(len(v)), within_5ms=float(np.mean(np.abs(v) <= 5.)), within_50ms=float(np.mean(np.abs(v) <= 50.)),
                                                hist_5ms_150=[int(x) for x in np.histogram(v, bins=60, range=(-150., 150.))[0]], median=float(np.median(v)))
                                       for r_, v in sorted(fine.items())},
                        lag_median=float(np.median(lags)) if len(lags) else None, per_rank={k_: float(v / len(T1)) for k_, v in sorted(rb.items())})
    TR.dump(fn, out)
    print(json.dumps({k: (v if k != 'coins' else '...') for k, v in out.items() if k in ('order', 'counterparties')}, indent=1, default=str)[:3000])


def selftest():
    Z = pd.DataFrame(dict(kind=['FIRE'] * 4 + ['NEARBY'] * 4, fut_quiet_300=[True] * 8, y300=[True, True, False, False] * 2, x=[1., -1., 0., 2., 0., 0., 1., 0.]))
    r = lead_table(Z, 'x', 0.)
    assert r['fire_move_after'] == dict(n=2, moved_its_way=.5, moved_against=.5) and r['nearby_no_move_after']['moved_its_way'] == .5, r
    assert day_stems('20261031')[0] == '20261031_00' and len(day_stems('20261031')) == 24
    print('info_batch2 selftest PASS')


if __name__ == '__main__':
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == 'coins':
        coins(a[0], a[1] if len(a) > 1 else None)
    elif cmd == 'pm':
        pm(a[0], a[1] if len(a) > 1 else None, int(a[2]) if len(a) > 2 else None)
    elif cmd == 'merge':
        merge(a[0] if a else None)
    else:
        dict(selftest=selftest)[cmd]()
