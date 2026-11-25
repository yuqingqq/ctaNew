"""Why do the winners' takes mark out so much higher than ours? (USER 2026-11-25: "check the selection rule, why others have a much higher
mark out profit"). Diagnostic on the TAKE-REPRICE v1 ledgers (REPRICE-4/8/16-172) and STREAM-172: 408 complete markets, five consumed days.

Every take we SENT is valued as if we had filled it at our limit L on arrival a (= decision + 172 ms):
would-be value at h = d x (mid(a + h) - L) - fee(L), with the take-marks mid (venue frame at the time, both sides valid, <= 1 s old) and
fee 2 % x min(L, 100 - L). The same quote is worth the same to whoever fills it at that price within milliseconds, so splitting our sent
takes by who got the aimed quote separates the race from the signal:
- filled: we got shares (filled / partial);
- taken: public trades consumed the aimed level before we arrived (reason empty, or limit_moved with trades at or through L of at least
  half the shown size before the level vanished: who_took.follow, tight window);
- pulled: the level vanished without that much trading (limit_moved otherwise; strict_cancelled = gone in the next frame without trades);
- other: invalid / stale book, price boundary.
For taken quotes the other takers' own trades (aimed side, at or through L, decision frame .. level gone + 100 ms) are valued the same way at
their own time and price; timing = their first trade minus our decision (< 150 ms: they decided before us, the hold alone takes 150 ms);
the implied repricing they could have seen = dFV at their first trade - 150 ms (the latest instant they can have decided).
Room left at our limit: e = d x (m_pre - L) + d x dFV - fee(L), m_pre = mid known 250 ms before the decision, dFV = reprice_replay's
implied repricing recomputed at the decision breakpoint.
Controls: our filled takes at L reproduce the engine's take marks exactly (falsifier: the mid read 1 s later must mismatch); every
REPRICE-D take has d x dFV >= D (falsifier: dFV read 300 ms earlier must fall below D somewhere).
Output (write-once): why_markout.json; per-take table OUT/why_markout_takes.parquet.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh why_markout -- python3 predictfun/reviews/take_reprice_20261125/why_markout.py
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
sys.path.insert(0, str(H.parent / 'take_exit_venue_20261125'))
import reprice_replay as RP  # noqa: E402
import who_took as WT  # noqa: E402

S, VR, F, OD, MM, TR, C = RP.S, RP.VR, RP.F, RP.OD, RP.MM, RP.TR, RP.C
EPS = 1e-9
CFG = {'REPRICE-4-172': (RP.OUT, 4.), 'REPRICE-8-172': (RP.OUT, 8.), 'REPRICE-16-172': (RP.OUT, 16.), 'STREAM-172': (S.OUT, None)}
E_BINS = [-np.inf, 0., 2., 4., 8., np.inf]
GAP_BINS = [-np.inf, 100., 150., 172., np.inf]


def fee(p):
    return .02 * min(p, 100. - p)


def mid_ok(f, t):
    j = int(np.searchsorted(f[:, 0], t, side='right')) - 1
    if j < 0:
        return None
    b, a = f[j, 2], f[j, 12]
    return float((b + a) / 2) if (0 <= b < a <= 100 and t - f[j, 0] <= 1000) else None


def mid_known(f, t):
    j = int(np.searchsorted(f[:, 1], t, side='right')) - 1
    if j < 0:
        return None
    b, a = f[j, 2], f[j, 12]
    return float((b + a) / 2) if 0 <= b < a <= 100 else None


def dfv_at(bk, bm, cs, ws, t):
    """reprice_replay's dFV (cents, signed: up > 0) at one breakpoint t (absolute ms); nan where the trigger would skip it."""
    jw, j0 = int(np.searchsorted(bk, ws, 'right') - 1), int(np.searchsorted(bk, t, 'right') - 1)
    j1, js = int(np.searchsorted(bk, t - RP.LAG, 'right') - 1), int(np.searchsorted(bk, t - RP.LAG - RP.SIG_WIN, 'right') - 1)
    if min(jw, j0, j1) < 0:
        return np.nan
    span = (t - RP.LAG - max(t - RP.LAG - RP.SIG_WIN, bk[0])) / 1000.
    sig = 1e4 * math.sqrt(max(cs[j1] - cs[max(js, 0)], 0.) / max(span, 1.)) if span > 1. else np.nan
    tau = (ws + 300000. - t) / 1000.
    if not (np.isfinite(sig) and sig > 0 and tau >= .5):
        return np.nan
    s = sig * math.sqrt(tau)
    z = np.array([1e4 * math.log(bm[j0] / bm[jw]) / s, 1e4 * math.log(bm[j1] / bm[jw]) / s])
    ph = RP.Phi(z)
    return float(100. * (ph[0] - ph[1]))


def would_be(f, d, L, t, end):
    out = {}
    for h in (1000., 10000.):
        m = mid_ok(f, t + h) if t + h <= end else None
        out[f'{int(h // 1000)}'] = None if m is None else d * (m - L) - fee(L)
    return out


def takes_of(rec, p, f, tr, t0, end, ws, btc, D):
    bk, bm, cs = btc
    marks = {int(x['gid']): x for x in rec['take_marks']}
    rows, ctl = [], collections.Counter()
    for x in rec['ioc_results']:
        s, L, g, a = int(x['side']), float(x['limit']), float(x['submit']), float(x['act']); d = 1 if s == 0 else -1; opp = 1 - s
        wb = would_be(f, d, L, a, end)
        r = dict(gid=int(x['gid']), side=s, d=d, L=L, g=g, a=a, reason=x['reason'], q=float(x['q']), filled=float(x['filled']), wb1=wb['1'], wb10=wb['10'])
        dfv = dfv_at(bk, bm, cs, ws, t0 + g - S.FEED - S.DECIDE)
        m_pre = mid_known(f, g - 250.)
        r.update(ddfv=d * dfv if np.isfinite(dfv) else np.nan, m_pre=m_pre)
        r['e'] = d * (m_pre - L) + d * dfv - fee(L) if (m_pre is not None and np.isfinite(dfv)) else np.nan
        if D is not None:
            ctl['dfv_checked'] += int(np.isfinite(dfv)); ctl['dfv_below_D'] += int(np.isfinite(dfv) and d * dfv < D - 1e-6)
            early = dfv_at(bk, bm, cs, ws, t0 + g - S.FEED - S.DECIDE - 300.)
            ctl['falsifier_early_below_D'] += int(np.isfinite(early) and d * early < D - 1e-6)
        if x['filled'] > EPS:
            r['cls'] = 'filled'; mk = marks[int(x['gid'])]; r.update(price=float(x['price']), v1=mk['m1s_c'], v10=mk['m10s_c'])
            if abs(float(x['price']) - L) < EPS:
                ctl['parity_checked'] += 1
                ctl['parity_mismatch'] += int(not ((wb['1'] is None and mk['m1s_c'] is None) or (wb['1'] is not None and mk['m1s_c'] is not None and abs(wb['1'] - mk['m1s_c']) < 1e-6)))
                late = would_be(f, d, L, a + 1000., end + 1000.)['1']
                ctl['falsifier_late_mismatch'] += int(late is None or mk['m1s_c'] is None or abs(late - mk['m1s_c']) > 1e-6)
        else:
            fo = WT.follow(p, x)
            if x['reason'] == 'empty':
                r['cls'] = 'taken'
            elif x['reason'] == 'limit_moved':
                r['cls'] = 'taken' if fo['shown'] > EPS and fo['traded_tight'] >= .5 * fo['shown'] else 'pulled'
            elif x['reason'] == 'strict_cancelled':
                r['cls'] = 'pulled'
            else:
                r['cls'] = 'other'
            r.update(shown=fo['shown'], traded_tight=fo['traded_tight'], gone_ms=fo['gone_ms'])
            if r['cls'] == 'taken':
                jk = int(np.searchsorted(f[:, 1], g, side='right') - 1)
                stop = (g + fo['gone_ms'] if fo['gone_ms'] is not None else a) + 100.
                m = (tr[:, 1] == opp) & ((tr[:, 2] >= L) if opp == 1 else (tr[:, 2] <= L)) & (tr[:, 0] >= f[jk, 0]) & (tr[:, 0] <= stop)
                if m.any():
                    T_ = tr[m]; vals = [would_be(f, d, float(P), float(t), end) for t, _, P, _ in T_]
                    w = T_[:, 3]
                    for h in ('1', '10'):
                        ok = np.array([v[h] is not None for v in vals])
                        r[f'c_v{h}'] = float(np.sum([v[h] * q for v, q in zip(vals, w) if v[h] is not None]) / w[ok].sum()) if ok.any() else None
                    r.update(c_shares=float(w.sum()), gap_dec=float(T_[:, 0].min() - g),
                             c_ddfv_latest=d * dfv_at(bk, bm, cs, ws, t0 + float(T_[:, 0].min()) - S.HOLD))
        rows.append(r)
    return rows, ctl


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    fn = H / 'why_markout.json'
    C.require(not fn.exists(), 'why_markout.json exists (write-once)')
    vs = json.loads((VR.OUT / 'summary.json').read_text())
    src, rows, ctl = MM.KP.Tape(), [], collections.Counter()
    for day in OD.DAYS3:
        comp = {int(r['mid']) for r in vs['rows'] if r['day'] == day and r['status'] == 'complete'}
        bk, bm = OD.seg_arrays(OD.SEG_OF[day]); bk = bk.astype(float); bm = bm.astype(float)
        btc = (bk, bm, np.cumsum(np.r_[0., np.diff(np.log(bm))] ** 2))
        reg = MM.registry(day).set_index('mid'); off = src.offset(day)
        for h, g in reg.groupby('hour'):
            mids = [int(m) for m in g.index if int(m) in comp]
            if not mids:
                continue
            for m, W, lst in VR.worlds_all(day, mids, src, off, reg):
                C.require(lst == 'loaded', f'{m} {lst}')
                ws = float(reg.loc[m, 'ws']) * 1000.
                p, st, info = VR.vm_path(W, ws, 50.)
                C.require(st == 'complete', f'{m} {st}')
                t0, end = float(info['t_start']), float(info['end_rel'])
                f = np.asarray(p['frames'], float); tr = np.asarray(p['trades'], float).reshape(-1, 4)
                for name, (od, D) in CFG.items():
                    rec = json.loads(gzip.decompress((od / f'{m}_{name}.json.gz').read_bytes()))
                    rr, cc = takes_of(rec, p, f, tr, t0, end, ws, btc, D)
                    ctl.update({f'{name}:{k}': v for k, v in cc.items()})
                    rows += [dict(r, day=day, mid=m, config=name) for r in rr]
                C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
        print(day, len(rows), flush=True)
    T = pd.DataFrame(rows)
    T.to_parquet(RP.OUT / 'why_markout_takes.parquet', index=False)
    num = lambda s: pd.to_numeric(s, errors='coerce')

    def mean(s):
        s = num(s).dropna(); return float(s.mean()) if len(s) else None

    def wavg(v, w):
        v, w = num(v), num(w); ok = v.notna() & w.notna() & (w > 0); return float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else None

    def q3(s):
        s = num(s).dropna(); return [float(x) for x in s.quantile([.25, .5, .75])] if len(s) else None
    out = dict(asof='2026-11-25', markets=int(T[['day', 'mid']].drop_duplicates().shape[0]), controls=dict(ctl), configs={})
    for name in CFG:
        X = T[T.config == name]; n = len(X)
        b = dict(sent=n, all_sent=dict(wb1=mean(X.wb1), wb10=mean(X.wb10)), classes={})
        for c in ('filled', 'taken', 'pulled', 'other'):
            G = X[X.cls == c]
            b['classes'][c] = dict(n=len(G), share=len(G) / n if n else None, wb1=mean(G.wb1), wb10=mean(G.wb10), ddfv_q=q3(G.ddfv), e_q=q3(G.e))
        Fd = X[X.cls == 'filled']; K = X[(X.cls == 'taken') & num(X.get('c_shares')).notna()] if 'c_shares' in X else X.iloc[:0]
        b['filled_realised'] = dict(shares=float(num(Fd.filled).sum()), v1=wavg(Fd.v1, Fd.filled), v10=wavg(Fd.v10, Fd.filled))
        b['taken_by_others'] = dict(n=len(K), their_shares=float(num(K.c_shares).sum()) if len(K) else 0., their_v1=wavg(K.c_v1, K.c_shares), their_v10=wavg(K.c_v10, K.c_shares),
                                    our_wb1_same=mean(K.wb1), our_wb10_same=mean(K.wb10), gap_dec_q=q3(K.gap_dec), their_ddfv_latest_q=q3(K.c_ddfv_latest),
                                    our_ddfv_q=q3(K.ddfv), decided_before_us=float((num(K.gap_dec) < S.HOLD).mean()) if len(K) else None,
                                    gap_dec_bins={str(k): int(v) for k, v in pd.cut(num(K.gap_dec), GAP_BINS).value_counts(sort=False).items()} if len(K) else {})
        Xe = X[num(X.e).notna()].assign(eb=lambda z: pd.cut(num(z.e), E_BINS))
        b['by_room'] = {str(k): dict(n=len(G), filled=float((G.cls == 'filled').mean()), taken=float((G.cls == 'taken').mean()), pulled=float((G.cls == 'pulled').mean()),
                                     wb1_all=mean(G.wb1), wb1_filled=mean(G[G.cls == 'filled'].wb1), wb1_taken=mean(G[G.cls == 'taken'].wb1))
                        for k, G in Xe.groupby('eb', observed=True)}
        out['configs'][name] = b
    ok = all(ctl[f'{n}:parity_mismatch'] == 0 for n in CFG) and all(ctl[f'{n}:dfv_below_D'] == 0 for n in CFG if CFG[n][1] is not None)
    fires = all(ctl[f'{n}:falsifier_late_mismatch'] > 0 for n in CFG) and all(ctl[f'{n}:falsifier_early_below_D'] > 0 for n in CFG if CFG[n][1] is not None)
    out['controls_pass'], out['falsifiers_fire'] = bool(ok), bool(fires)
    TR.dump(fn, out)
    print(json.dumps(out, indent=1, default=str))


if __name__ == '__main__':
    main()
