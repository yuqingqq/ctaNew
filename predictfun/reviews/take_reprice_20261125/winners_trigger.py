"""What triggers the takers who beat us? (USER 2026-11-25: "Proceed to find their signal"). Diagnostic on four consumed days (20261023,
20261031..1102: every venue present); no strategy change.

Stage `legs <day>`: every BTC 5m maker leg of the day (step1 legs_timed2) joined to its taker (REST match records: taker signer and order hash,
takers.taker_table, pf_step1's key); legs whose key maps to more than one taker are flagged `ambiguous` and kept out of per-taker results.
Output (write-once): OUT/winners_legs_<day>.parquet + OUT/winners_legs_<day>.json (counts).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/winners_trigger.py legs <day>

Stage `analyze`: taker orders (legs grouped by taker order hash; venue time = t_ms2 - 50 ms, the replay's trade-clock convention, about
+-30 ms; direction d = +1 when the order took asks, i.e. bought Up) on the venue-complete markets of the 408 complete set. Takers ranked by
their 1 s value after fee (d x (book mid 1 s later - VWAP) - 2 % x min(p, 1 - p), x shares). Groups: TOP1, TOP2-10 (by 1 s value), REST
(a seeded sample). Features at the latest possible send time s = first trade - 150 ms, signed by d: Binance futures mid returns over 50 / 100 /
250 / 1000 ms, time since the last mid change, mid changes in the last 1 s, best-level imbalance, opposite best size and its change over
300 ms, spread; Binance futures taker flow over 100 / 300 / 1000 ms and trade count over 300 ms; OKX swap, Binance spot, Bybit perp and
Coinbase trade-price returns over 100 / 250 ms; Predict spread, distance of the mid from 50, time left, the mid's 1 s change, opposite
size, and the Binance-implied repricing over 250 / 1000 ms not yet in Predict's mid. Outcomes (not features): the Binance move during the
hold (s .. s + 150 / 300 ms) and the order's 1 s value.
Null (declared before the result): the same features at s shifted by +-5, 10, .., 30 s inside the same market with the same d (12 per order).
Readings: per feature AUC (send vs null) and medians; a logistic model (standardized, balanced) scored leave-one-day-out (AUC per held-out
day); the share of sends followed by a Binance move their way during the hold vs the null.
Output (write-once): winners_trigger.json.
Usage: ... winners_trigger.py analyze
"""
import collections
import json
import math
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
sys.path.insert(0, str(H.parent / 'take_exit_venue_20261125'))
import reprice_replay as RP  # noqa: E402
import takers as TK  # noqa: E402
import lead_venues as LV  # noqa: E402
import why_markout as WM  # noqa: E402

S, VR, F, OD, MM, TR, C = RP.S, RP.VR, RP.F, RP.OD, RP.MM, RP.TR, RP.C
OUT = RP.OUT
DAYS = ('20261023', '20261031', '20261101', '20261102')
RAW = LV.DATA / 'mm_hf/raw'
M50, HOLD, CAP_REST, CAP_TOP, SEED, LR_ROWS = 50., 150., 10000, 15000, 20261125, 150000
SHIFTS = np.array([-30, -25, -20, -15, -10, -5, 5, 10, 15, 20, 25, 30], float) * 1000.
VEN = ('okx_swap', 'binance_spot', 'bybit_perp', 'coinbase')
OUTCOMES = ('bn_f150', 'bn_f300')


def legs(day):
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    C.require(day in DAYS, f'{day} is not a declared day')
    fn, rc = OUT / f'winners_legs_{day}.parquet', OUT / f'winners_legs_{day}.json'
    C.require(not fn.exists() and not rc.exists(), f'{fn.name} exists (write-once)')
    L = pd.read_parquet(MM.S1 / day / 'legs_timed2.parquet', columns=TK.KEY + ['t_ms2', 'how2'])
    T = TK.taker_table(day)
    dup = T.duplicated(TK.KEY, keep=False)
    amb = T[dup].groupby(TK.KEY).taker.nunique()
    amb_keys = set(amb[amb > 1].index)
    T = T.drop_duplicates(TK.KEY)
    J = L.merge(T, on=TK.KEY, how='left')
    C.require(len(J) == len(L), 'the join duplicated legs')
    J['ambiguous'] = [k in amb_keys for k in zip(*(J[c] for c in TK.KEY))]
    J.to_parquet(fn, index=False)
    TR.dump(rc, dict(asof='2026-11-25', day=day, legs=len(J), joined=int(J.taker.notna().sum()), ambiguous_legs=int(J.ambiguous.sum()),
                     takers=int(J.taker.nunique()), orders=int(J.taker_hash.nunique())))
    print(day, len(J), int(J.taker.notna().sum()), int(J.ambiguous.sum()), flush=True)


def last_idx(t, T):
    return np.searchsorted(t, T, 'right') - 1


def bn_feats(bk, bm, cs, T, d, ws):
    j = last_idx(bk, T); ok = j >= 0; jj = np.maximum(j, 0)
    o = {}
    for k in (50., 100., 250., 1000.):
        jk = np.maximum(last_idx(bk, T - k), 0)
        o[f'bn_r{int(k)}'] = np.where(ok, d * 1e4 * np.log(bm[jj] / bm[jk]), np.nan)
    o['bn_age'] = np.where(ok, T - bk[jj], np.nan)
    o['bn_n1s'] = np.where(ok, j - last_idx(bk, T - 1000.), np.nan)
    for k in (150., 300.):
        jf = np.maximum(last_idx(bk, T + k), 0)
        o[f'bn_f{int(k)}'] = np.where(ok, d * 1e4 * np.log(bm[jf] / bm[jj]), np.nan)
    jw = np.maximum(last_idx(bk, ws), 0)
    tau = np.maximum((ws + 300000. - T) / 1000., .5)
    for lag in (250., 1000.):
        j1 = np.maximum(last_idx(bk, T - lag), 0); js = np.maximum(last_idx(bk, T - lag - 300000.), 0)
        sig = 1e4 * np.sqrt(np.maximum(cs[j1] - cs[js], 0.) / 300.)
        sc = np.where(sig > 0, sig * np.sqrt(tau), np.nan)
        x, xl = 1e4 * np.log(bm[jj] / bm[jw]), 1e4 * np.log(bm[j1] / bm[jw])
        o[f'dfv{int(lag)}'] = np.where(ok, d * 100. * (ndtr(x / sc) - ndtr(xl / sc)), np.nan)
    return o


def bt_feats(day, T, d):
    o = {k: np.full(len(T), np.nan) for k in ('bt_imb', 'bt_opp_log', 'bt_depl300', 'bt_spread')}
    hour = ((T - OD_day_ms(day)) // 3600000).astype(int)
    tail = None
    for h in range(24):
        fn = RAW / f'bookTicker/BTCUSDT/{day}_{h:02d}.csv.gz'
        if not fn.exists():
            tail = None; continue
        b = pd.read_csv(fn, header=None, usecols=[1, 4, 5, 6, 7], names=['E', 'bid', 'bq', 'ask', 'aq'])
        if tail is not None:
            b = pd.concat([tail, b])
        b = b.sort_values('E', kind='stable')
        E, bid, bq, ask, aq = (b[c].to_numpy(float) for c in ('E', 'bid', 'bq', 'ask', 'aq'))
        tail = b[b.E >= E[-1] - 2000.]
        sel = np.flatnonzero(hour == h)
        if not len(sel):
            continue
        t, dd = T[sel], d[sel]
        j, j3 = last_idx(E, t), last_idx(E, t - 300.)
        ok = (j >= 0) & (j3 >= 0); j, j3 = np.maximum(j, 0), np.maximum(j3, 0)
        opp_now, opp_prev = np.where(dd > 0, aq[j], bq[j]), np.where(dd > 0, aq[j3], bq[j3])
        p_now, p_prev = np.where(dd > 0, ask[j], bid[j]), np.where(dd > 0, ask[j3], bid[j3])
        dep = np.where(np.abs(p_now - p_prev) < 1e-9, (opp_prev - opp_now) / np.maximum(opp_prev, 1e-9), np.where(dd * (p_now - p_prev) > 0, 1., -1.))
        o['bt_imb'][sel] = np.where(ok, dd * (bq[j] - aq[j]) / np.maximum(bq[j] + aq[j], 1e-9), np.nan)
        o['bt_opp_log'][sel] = np.where(ok, np.log(np.maximum(opp_now, 1e-6)), np.nan)
        o['bt_depl300'][sel] = np.where(ok, np.clip(dep, -1., 1.), np.nan)
        o['bt_spread'][sel] = np.where(ok, (ask[j] - bid[j]) / .1, np.nan)
    return o


def OD_day_ms(day):
    return LV.day_ms(day)[0]


def flow_feats(tt, Fq, T, d):
    o = {}
    for k in (100., 300., 1000.):
        o[f'fl{int(k)}'] = d * (Fq[np.searchsorted(tt, T, 'right')] - Fq[np.searchsorted(tt, T - k, 'right')])
    o['nt300'] = (np.searchsorted(tt, T, 'right') - np.searchsorted(tt, T - 300., 'right')).astype(float)
    return o


def ven_feats(px, T, d):
    o = {}
    for v in VEN:
        t, p = px[v]
        j = last_idx(t, T); ok = j >= 0; jj = np.maximum(j, 0)
        for k in (100., 250.):
            jk = last_idx(t, T - k)
            o[f'{v}_r{int(k)}'] = np.where(ok & (jk >= 0), d * 1e4 * np.log(p[jj] / p[np.maximum(jk, 0)]), np.nan)
    return o


def pf_feats(pb, T, d, ws):
    rel = T - pb['t0']
    j = last_idx(pb['t'], rel); ok = j >= 0; jj = np.maximum(j, 0)
    b, a = pb['b'][jj], pb['a'][jj]; good = ok & (b >= 0) & (b < a) & (a <= 100)
    mid = np.where(good, (b + a) / 2., np.nan)
    j1 = np.maximum(last_idx(pb['t'], rel - 1000.), 0); b1, a1 = pb['b'][j1], pb['a'][j1]
    mid1 = np.where((b1 >= 0) & (b1 < a1) & (a1 <= 100), (b1 + a1) / 2., np.nan)
    return dict(p_spread=np.where(good, a - b, np.nan), p_dist50=np.abs(mid - 50.), p_tau=(ws + 300000. - T) / 1000., p_r1000=d * (mid - mid1),
                p_opp_log=np.where(good, np.log(np.maximum(np.where(d > 0, pb['as'][jj], pb['bs'][jj]), 1e-6)), np.nan))


def analyze():
    resource.setrlimit(resource.RLIMIT_CPU, (1500, 1501))
    fn = H / 'winners_trigger.json'
    C.require(not fn.exists(), 'winners_trigger.json exists (write-once)')
    vs = json.loads((VR.OUT / 'summary.json').read_text())
    src, rng = MM.KP.Tape(), np.random.default_rng(SEED)
    O_all, feats = [], []
    for day in DAYS:
        comp = {int(r['mid']) for r in vs['rows'] if r['day'] == day and r['status'] == 'complete'}
        J = pd.read_parquet(OUT / f'winners_legs_{day}.parquet')
        n0 = len(J)
        J = J[J.taker.notna() & ~J.ambiguous & J.mid.isin(comp) & J.how2.isin([1, 3])].copy()
        J['t_v'] = J.t_ms2.astype(float) - M50; J['d'] = np.where(J.side == 1, 1, -1); J['ps'] = J.ptick * J.shares
        g = J.groupby('taker_hash')
        O = g.agg(taker=('taker', 'first'), mid=('mid', 'first'), t=('t_v', 'min'), d=('d', 'first'), dn=('d', 'nunique'), shares=('shares', 'sum'), ps=('ps', 'sum'),
                  n_legs=('shares', 'size')).reset_index()
        O = O[O.dn == 1].copy(); O['vwap'] = O.ps / O.shares; O['day'] = day
        reg = MM.registry(day).set_index('mid'); off = src.offset(day)
        O['ws'] = O.mid.map(lambda m: float(reg.loc[m, 'ws']) * 1000.)
        O['v1'] = np.nan
        paths = {}
        for h, gg in reg.groupby('hour'):
            mids = [int(m) for m in gg.index if int(m) in comp]
            if not mids:
                continue
            for m, W, lst in VR.worlds_all(day, mids, src, off, reg):
                C.require(lst == 'loaded', f'{m} {lst}')
                ws = float(reg.loc[m, 'ws']) * 1000.
                p, st, info = VR.vm_path(W, ws, 50.)
                C.require(st == 'complete', f'{m} {st}')
                f = np.asarray(p['frames'], float); t0 = float(info['t_start'])
                paths[m] = dict(t=f[:, 0].copy(), b=f[:, 2].copy(), a=f[:, 12].copy(), bs=f[:, 7].copy(), **{'as': f[:, 17].copy()}, t0=t0)
                sel = np.flatnonzero(O.mid.to_numpy() == m)
                v = []
                for i in sel:
                    mm = WM.mid_ok(f, O.t.iat[i] - t0 + 1000.)
                    v.append(np.nan if mm is None else O.d.iat[i] * (mm - O.vwap.iat[i]) - WM.fee(O.vwap.iat[i]))
                O.iloc[sel, O.columns.get_loc('v1')] = v
        O_all.append(O)
        print(day, 'legs', n0, 'orders', len(O), flush=True)
        feats.append((day, O, paths))
    A = pd.concat(O_all, ignore_index=True)
    A['val'] = A.v1 * A.shares
    tk = A.groupby('taker').agg(orders=('t', 'size'), shares=('shares', 'sum'), value_usd=('val', lambda x: float(np.nansum(x)) / 100.),
                                v1_per_share=('val', lambda x: np.nan)).reset_index()
    tk['v1_per_share'] = tk.value_usd * 100. / tk.shares
    by_val = tk.sort_values('value_usd', ascending=False).reset_index(drop=True)
    top = list(by_val.taker[:10])
    grp_of = {top[0]: 'TOP1', **{t: 'TOP2_10' for t in top[1:]}}
    out = dict(asof='2026-11-25', days=list(DAYS), m50_ms=M50, takers=int(len(tk)), orders=int(len(A)),
               top10_by_value=[dict(rank=i + 1, taker=r.taker[:6] + '…' + r.taker[-4:], orders=int(r.orders), shares=float(r.shares), value_usd=float(r.value_usd),
                                    v1_c_per_share=float(r.v1_per_share)) for i, r in by_val.head(10).iterrows()],
               top10_by_shares=[dict(taker=r.taker[:6] + '…' + r.taker[-4:], orders=int(r.orders), shares=float(r.shares), value_usd=float(r.value_usd),
                                     v1_c_per_share=float(r.v1_per_share)) for _, r in tk.sort_values('shares', ascending=False).head(10).iterrows()],
               all_takers_v1_c_per_share=float(np.nansum(A.val) / A.shares[A.v1.notna()].sum()))
    rows = []
    for day, O, paths in feats:
        O = O.copy(); O['grp'] = O.taker.map(grp_of).fillna('REST')
        pick = []
        for gname, cap in (('TOP1', CAP_TOP), ('TOP2_10', CAP_TOP), ('REST', CAP_REST)):
            ix = np.flatnonzero(O.grp.to_numpy() == gname)
            pick.append(rng.choice(ix, cap, replace=False) if len(ix) > cap else ix)
        O = O.iloc[np.sort(np.concatenate(pick))].reset_index(drop=True)
        s0 = O.t.to_numpy() - HOLD
        pts = [pd.DataFrame(dict(k=np.arange(len(O)), y=1, s=s0))] + [pd.DataFrame(dict(k=np.arange(len(O)), y=0, s=s0 + sh)) for sh in SHIFTS]
        P = pd.concat(pts, ignore_index=True)
        P['mid'] = O.mid.to_numpy()[P.k]; P['d'] = O.d.to_numpy()[P.k].astype(float); P['ws'] = O.ws.to_numpy()[P.k]; P['grp'] = O.grp.to_numpy()[P.k]
        P = P[(P.s >= P.ws + 1000.) & (P.s <= P.ws + 299000.)].reset_index(drop=True)
        T, dd, ws = P.s.to_numpy(float), P.d.to_numpy(float), P.ws.to_numpy(float)
        bk, bm = OD.seg_arrays(OD.SEG_OF[day]); bk = bk.astype(float); bm = bm.astype(float)
        cs = np.cumsum(np.r_[0., np.diff(np.log(bm))] ** 2)
        Fe = bn_feats(bk, bm, cs, T, dd, ws)
        Fe.update(bt_feats(day, T, dd))
        tr = pd.concat([pd.read_csv(RAW / f'trade/BTCUSDT/{day}_{h:02d}.csv.gz', header=None, usecols=[2, 5, 6], names=['t', 'q', 'm'])
                        for h in range(24) if (RAW / f'trade/BTCUSDT/{day}_{h:02d}.csv.gz').exists()]).sort_values('t', kind='stable')
        tt = tr.t.to_numpy(float); Fq = np.r_[0., np.cumsum(tr.q.to_numpy(float) * np.where(tr.m.to_numpy() == 0, 1., -1.))]
        Fe.update(flow_feats(tt, Fq, T, dd)); del tr
        Fe.update(ven_feats({v: LV.load_venue(v, day) for v in VEN}, T, dd))
        pf = {k: np.full(len(T), np.nan) for k in ('p_spread', 'p_dist50', 'p_tau', 'p_r1000', 'p_opp_log')}
        for m, ix in P.groupby('mid').indices.items():
            if m not in paths:
                continue
            o = pf_feats(paths[m], T[ix], dd[ix], ws[ix])
            for k in pf:
                pf[k][ix] = o[k]
        Fe.update(pf)
        Fe['unpriced1000'] = Fe['dfv1000'] - pf['p_r1000']
        for k, v in Fe.items():
            P[k] = np.asarray(v, np.float32)
        P['day'] = day; P['v1'] = O.v1.to_numpy()[P.k]
        rows.append(P)
        C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
        print(day, 'points', len(P), flush=True)
    X = pd.concat(rows, ignore_index=True)
    X.to_parquet(OUT / 'winners_trigger_points.parquet', index=False)
    FEAT = [c for c in X.columns if c.startswith(('bn_', 'bt_', 'fl', 'nt', 'dfv', 'p_', 'unpriced')) or c.endswith(('_r100', '_r250'))]
    FEAT = [c for c in dict.fromkeys(FEAT) if c not in OUTCOMES]
    out['features'] = FEAT; out['groups'] = {}
    for gname in ('TOP1', 'TOP2_10', 'REST'):
        G = X[X.grp == gname]
        if not G.y.sum():
            continue
        b = dict(sends=int(G.y.sum()), null=int((G.y == 0).sum()), v1_c_per_share=float(np.nanmean(G[G.y == 1].v1)),
                 bet=dict(move_their_way_150=float((G[G.y == 1].bn_f150 > 0).mean()), null_150=float((G[G.y == 0].bn_f150 > 0).mean()),
                          move_their_way_300=float((G[G.y == 1].bn_f300 > 0).mean()), null_300=float((G[G.y == 0].bn_f300 > 0).mean()),
                          mean_f150_bps=float(np.nanmean(G[G.y == 1].bn_f150)), null_mean_f150_bps=float(np.nanmean(G[G.y == 0].bn_f150))), uni={})
        for c in FEAT:
            ok = G[c].notna()
            if ok.sum() < 100 or G.y[ok].nunique() < 2:
                continue
            b['uni'][c] = dict(auc=float(roc_auc_score(G.y[ok], G[c][ok])), send_med=float(G[c][ok & (G.y == 1)].median()), null_med=float(G[c][ok & (G.y == 0)].median()))
        Z = G[FEAT].copy(); med = Z[G.y == 0].median()
        Z = Z.fillna(med); mu, sd = Z.mean(), Z.std().replace(0, 1.)
        Z = ((Z - mu) / sd).clip(-6, 6)
        cv, yv = {}, G.y.to_numpy()

        def cap(mask):
            ix = np.flatnonzero(mask)
            return np.sort(rng.choice(ix, LR_ROWS, replace=False)) if len(ix) > LR_ROWS else ix
        for dy in DAYS:
            tr_, te_ = cap((G.day != dy).to_numpy()), cap((G.day == dy).to_numpy())
            if len(set(yv[te_])) < 2 or len(set(yv[tr_])) < 2:
                continue
            lr = LogisticRegression(max_iter=1000, C=1., class_weight='balanced').fit(Z.iloc[tr_], yv[tr_])
            cv[dy] = float(roc_auc_score(yv[te_], lr.predict_proba(Z.iloc[te_])[:, 1]))
        ia = cap(np.ones(len(G), bool))
        lr = LogisticRegression(max_iter=1000, C=1., class_weight='balanced').fit(Z.iloc[ia], yv[ia])
        b['multi'] = dict(cv_auc_by_day=cv, cv_auc_mean=float(np.mean(list(cv.values()))) if cv else None,
                          coef=dict(sorted(((c, float(w)) for c, w in zip(FEAT, lr.coef_[0])), key=lambda kv: -abs(kv[1]))))
        out['groups'][gname] = b
    TR.dump(fn, out)
    print(json.dumps({g: dict(sends=b['sends'], v1=b['v1_c_per_share'], bet=b['bet'], cv=b['multi']['cv_auc_mean'],
                              top_uni=sorted(((c, round(u['auc'], 3)) for c, u in b['uni'].items()), key=lambda kv: -abs(kv[1] - .5))[:8]) for g, b in out['groups'].items()}, indent=1))


if __name__ == '__main__':
    dict(legs=lambda: legs(sys.argv[2]), analyze=analyze)[sys.argv[1]]()
