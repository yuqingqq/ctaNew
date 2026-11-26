"""The Binance futures book behind the best level before the top taker's fires (USER 2026-11-26: "an we check deeper level in binance
first"). Readout; the four consumed days (it traded on three); no strategy change.
Points: fire_breakdown_points.parquet (its 5,930 fires, t = arrival at Predict, and nearby moments = the same times +-5..30 s) with their
breakdown row (row groups as cross_venue: quiet / far side shrank / other flat / moved its way).
Depth: the depth20 stream (top 20 levels per side, a snapshot every ~102 ms, E clock); per point the last snapshot at or before t (S1),
t - 300 ms (S0) and t - 1 s (Sm); a point whose S1 is more than 250 ms old has no value (status counted). Check: the snapshot's level-1
prices against the bookTicker best at the snapshot's E (share equal, both sides).
Per point (far side = the side that empties for a move its way): far / near size over levels 2-5 and 2-20 (behind the best), the size
within $2 of the best (levels 2-20), depth imbalance its way over levels 1-5 and 1-20 ((near - far) / (near + far)); their change from S0
and from Sm (ratios for sizes, differences for imbalance).
Declared before the run: deciles from nearby moments. Readings per feature and row group: fires vs nearby quantiles; the share of fires
below the nearby 10th / above the 90th percentile (same group); at nearby moments with no prior Binance move, the Binance move within
300 ms in the bottom decile / deciles 2-9 / top decile (deciles over those moments), and after its fires in the same bins.
Outputs (write-once): depth_deep.json; OUT/depth_deep_points.parquet.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/depth_deep.py \
       [<rehearsal dir> <day>]
"""
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import fire_events as FE  # noqa: E402
import cross_venue as CV  # noqa: E402

DC, C, TR, WT = FE.DC, FE.C, FE.TR, FE.WT
NL, STALE, CUSH = 20, 250., 2.
LAGS = (('1', 0.), ('0', 300.), ('m', 1000.))
GROUPS = ('quiet', 'far_shrank', 'other_flat', 'moved_its_way')
FEATS = ('far_2_5', 'far_2_20', 'near_2_5', 'near_2_20', 'far_cush', 'near_cush', 'imb_1_5', 'imb_1_20',
         'r300_far_2_20', 'r300_near_2_20', 'd300_imb_1_20', 'r1s_far_2_20', 'd1s_imb_1_20')
QS = (.1, .25, .5, .75, .9)


def parse(strs):
    """'p@q|p@q|..' x n -> prices, sizes (n, NL); NaN where a row has fewer levels."""
    P, Q = np.full((len(strs), NL), np.nan), np.full((len(strs), NL), np.nan)
    for i, s in enumerate(strs):
        for k, x in enumerate(s.split('|')[:NL]):
            p, q = x.split('@')
            P[i, k], Q[i, k] = float(p), float(q)
    return P, Q


def side_feats(Pf, Qf, Pn, Qn):
    """Features of one snapshot set: far side (Pf, Qf) and near side (Pn, Qn), level 1 first."""
    o = dict(far_2_5=np.nansum(Qf[:, 1:5], 1), far_2_20=np.nansum(Qf[:, 1:], 1), near_2_5=np.nansum(Qn[:, 1:5], 1), near_2_20=np.nansum(Qn[:, 1:], 1))
    o['far_cush'] = np.nansum(np.where(np.abs(Pf[:, 1:] - Pf[:, :1]) <= CUSH + 1e-9, Qf[:, 1:], 0.), 1)
    o['near_cush'] = np.nansum(np.where(np.abs(Pn[:, 1:] - Pn[:, :1]) <= CUSH + 1e-9, Qn[:, 1:], 0.), 1)
    for k in (5, 20):
        f, n = np.nansum(Qf[:, :k], 1), np.nansum(Qn[:, :k], 1)
        o[f'imb_1_{k}'] = (n - f) / np.maximum(n + f, 1e-12)
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
    fn, fp = (od or H) / 'depth_deep.json', (od or WT.OUT) / 'depth_deep_points.parquet'
    C.require(not fn.exists() and not fp.exists(), 'an output exists (write-once)')
    days = WT.DAYS if len(sys.argv) < 3 else (sys.argv[2],)
    C.require(all(x in WT.DAYS for x in days), 'not a declared day')
    P = pd.read_parquet(WT.OUT / 'fire_breakdown_points.parquet', columns=['dec', 'day', 't', 'd', 'shift_ms', 'row', 'their_prior', 'move300', 'far1', 'near1'])
    P = P[P.day.isin(days)].reset_index(drop=True)
    P['grp'] = P.row.map(CV.ROWG)
    P['fire'] = P.shift_ms == 0.
    cols = {f'{k}_{lag}': np.full(len(P), np.nan) for k in ('far_2_5', 'far_2_20', 'near_2_5', 'near_2_20', 'far_cush', 'near_cush', 'imb_1_5', 'imb_1_20') for lag, _ in LAGS}
    cols.update(age1=np.full(len(P), np.nan), l1_match=np.full(len(P), np.nan))
    for day in days:
        day0 = float(WT.OD_day_ms(day))
        m_day = np.flatnonzero((P.day == day).to_numpy())
        hp = ((P.t.to_numpy()[m_day] - day0) // 3600000).astype(int)
        for h, bk in DC.book_hours(day):
            sel = m_day[hp == h]
            if not len(sel):
                continue
            fd = DC.RAW / f'depth20/BTCUSDT/{day}_{h:02d}.csv.gz'
            C.require(fd.exists(), f'{day} {h}: no depth20 file')
            D = pd.read_csv(fd, header=None, usecols=[1, 4, 5], names=['E', 'b', 'a']).sort_values('E', kind='stable')
            E = D.E.to_numpy(float)
            t, d = P.t.to_numpy(float)[sel], P.d.to_numpy(float)[sel]
            idx = {lag: np.searchsorted(E, t - w, 'right') - 1 for lag, w in LAGS}
            u = np.unique(np.concatenate([v[v >= 0] for v in idx.values()]))
            bP, bQ = parse(D.b.to_numpy()[u])
            aP, aQ = parse(D.a.to_numpy()[u])
            pos = {lag: np.searchsorted(u, np.maximum(v, 0)) for lag, v in idx.items()}
            up = d > 0
            for lag, _ in LAGS:
                ok = idx[lag] >= 0
                k = pos[lag]
                Pf, Qf = np.where(up[:, None], aP[k], bP[k]), np.where(up[:, None], aQ[k], bQ[k])
                Pn, Qn = np.where(up[:, None], bP[k], aP[k]), np.where(up[:, None], bQ[k], aQ[k])
                for name, x in side_feats(Pf, Qf, Pn, Qn).items():
                    cols[f'{name}_{lag}'][sel] = np.where(ok, x, np.nan)
            age = t - E[np.maximum(idx['1'], 0)]
            cols['age1'][sel] = np.where(idx['1'] >= 0, age, np.nan)
            Eb, bid, bq, ask, aq = FE.collapse(*bk)
            j = WT.last_idx(Eb, E[np.maximum(idx['1'], 0)])
            k1 = pos['1']
            cols['l1_match'][sel] = np.where((j >= 0) & (idx['1'] >= 0), ((np.abs(bP[k1, 0] - bid[np.maximum(j, 0)]) < 1e-9) & (np.abs(aP[k1, 0] - ask[np.maximum(j, 0)]) < 1e-9)).astype(float), np.nan)
            C.require(DC.F.rss_mib() < DC.F.RSS_GUARD_MIB, 'RSS over the self-guard')
        print(day, len(m_day), flush=True)
    Z = pd.concat([P, pd.DataFrame(cols)], axis=1)
    stale = ~(Z.age1 <= STALE)
    Z.loc[stale, [c for c in cols if c not in ('age1', 'l1_match')]] = np.nan
    for k in ('far_2_5', 'far_2_20', 'near_2_5', 'near_2_20', 'far_cush', 'near_cush', 'imb_1_5', 'imb_1_20'):
        Z[k] = Z[f'{k}_1']
    Z['r300_far_2_20'] = Z.far_2_20_1 / Z.far_2_20_0.where(Z.far_2_20_0 > 0)
    Z['r300_near_2_20'] = Z.near_2_20_1 / Z.near_2_20_0.where(Z.near_2_20_0 > 0)
    Z['d300_imb_1_20'] = Z.imb_1_20_1 - Z.imb_1_20_0
    Z['r1s_far_2_20'] = Z.far_2_20_1 / Z.far_2_20_m.where(Z.far_2_20_m > 0)
    Z['d1s_imb_1_20'] = Z.imb_1_20_1 - Z.imb_1_20_m
    Fz, N = Z[Z.fire], Z[~Z.fire]
    npr = ~Z.their_prior.astype(bool)
    out = dict(asof='2026-11-26', days=list(days), n_fires=int(len(Fz)), n_nearby=int(len(N)), stale_ms=STALE, cushion_usd=CUSH,
               status=dict(fires_no_depth=int(stale[Z.fire].sum()), nearby_no_depth=int(stale[~Z.fire].sum())),
               snapshot_age_ms=dict(fires=qs(Fz.age1), nearby=qs(N.age1)), level1_matches_bookticker=dict(fires=share(Fz.l1_match), nearby=share(N.l1_match)),
               features={})
    NN = Z[~Z.fire & npr]
    for f in FEATS:
        lo, hi = np.nanquantile(NN[f], .1), np.nanquantile(NN[f], .9)
        o = dict(nearby_no_prior_p10=float(lo), nearby_no_prior_p90=float(hi), all=dict(fires=qs(Fz[f]), nearby=qs(N[f])), by_group={}, follow={})
        for g in GROUPS:
            Fg, Ng = Fz[Fz.grp == g], N[N.grp == g]
            gl, gh = np.nanquantile(Ng[f], .1), np.nanquantile(Ng[f], .9)
            o['by_group'][g] = dict(fires=qs(Fg[f]), nearby=qs(Ng[f]), fires_below_nearby_p10=share(np.where(Fg[f].notna(), Fg[f] < gl, np.nan)),
                                    fires_above_nearby_p90=share(np.where(Fg[f].notna(), Fg[f] > gh, np.nan)))
        for b, cond in (('bottom_decile', Z[f] < lo), ('middle', (Z[f] >= lo) & (Z[f] <= hi)), ('top_decile', Z[f] > hi)):
            cn, cf = ~Z.fire & npr & cond, Z.fire & npr & cond
            o['follow'][b] = dict(nearby=share(Z.move300[cn]), nearby_n=int(cn.sum()), fires=share(Z.move300[cf]), fires_n=int(cf.sum()))
        out['features'][f] = o
    Z.to_parquet(fp, index=False)
    TR.dump(fn, out)
    print(json.dumps({k: out[k] for k in ('status', 'snapshot_age_ms', 'level1_matches_bookticker')}, indent=0))
    for f, o in out['features'].items():
        q = o['by_group']['quiet']
        print(f"{f:16s} quiet med {q['fires']['p50'] if q['fires'] else None:.3f} vs {q['nearby']['p50'] if q['nearby'] else None:.3f} tails {q['fires_below_nearby_p10']:.3f}/{q['fires_above_nearby_p90']:.3f} "
              f"| follow nearby {o['follow']['bottom_decile']['nearby']:.3f} {o['follow']['middle']['nearby']:.3f} {o['follow']['top_decile']['nearby']:.3f} "
              f"| fires {o['follow']['bottom_decile']['fires']} {o['follow']['middle']['fires']:.3f} {o['follow']['top_decile']['fires']}")


if __name__ == '__main__':
    main()
