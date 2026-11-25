"""How do the takers who win the stale quotes select them? Diagnostic on four consumed days (20261023, 20261031..1102).
Two fill sets: WINNERS = the websocket-timed competitor legs that took the quotes our 172 ms grid takes missed (lead_venues' consuming legs),
state read at their arrival A = print - 150 ms; OURS = the filled takes of the STREAM-172 ledgers, state at our decision.
State from Binance futures mids (event time) and the registry: time left tau; BTC move since the window start x = ln(F_t / F_ws) (bps;
the futures-vs-Chainlink basis cancels to first order); realised vol sigma over the previous 300 s (bps / sqrt(s)); binary fair value
FV = Phi(x / (sigma sqrt(tau))); the take's edge after fee = (FV - p) for a buy, (p - FV) for a sell, minus 2 % x min(p, 1 - p), in cents;
the Binance move r250 in the take's direction. Realised value after fee at +1 s / +10 s (Predict book mid) and at settlement (ties 50 / 50).
Output (write-once): selection.json; per-fill table OUT/selection_fills.parquet.
Usage: PF_SLOT=claude predictfun/research/run_light.sh selection -- python3 predictfun/reviews/take_stream_20261125/selection.py
"""
import collections
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
sys.path.insert(0, str(H.parent / 'take_exit_venue_20261125'))
import stream_replay as S  # noqa: E402
import lead_venues as LV  # noqa: E402
import who_took as WT  # noqa: E402

V, F, OD, MM, TR, C = S.V, S.F, S.OD, S.MM, S.TR, S.C
DAYS = LV.DAYS
SIG_WIN = 300000.
EDGE_BINS = [-np.inf, -2., 0., 1., 2., 4., np.inf]


def phi(z):
    return 0.5 * (1. + math.erf(z / math.sqrt(2.)))


class Btc:
    def __init__(self, day):
        bk, bm = OD.seg_arrays(OD.SEG_OF[day])
        self.k, self.m = bk.astype(float), bm.astype(float)
        self.lr = np.r_[0., np.diff(np.log(self.m))]
        self.cs = np.cumsum(self.lr ** 2)

    def mid(self, t):
        j = int(np.searchsorted(self.k, t, 'right') - 1)
        return float(self.m[j]) if j >= 0 else np.nan

    def sigma(self, t):
        j1, j0 = int(np.searchsorted(self.k, t, 'right') - 1), int(np.searchsorted(self.k, t - SIG_WIN, 'right') - 1)
        if j1 < 0 or j0 < 0:
            return np.nan
        return 1e4 * math.sqrt(max(self.cs[j1] - self.cs[j0], 0.) / (SIG_WIN / 1000.))

    def state(self, t, ws, d, p):
        tau = max((ws + 300000. - t) / 1000., 1e-3)
        x = 1e4 * math.log(self.mid(t) / self.mid(ws))
        sg = self.sigma(t)
        fv = 100. * phi(x / (sg * math.sqrt(tau))) if sg > 0 else (100. if x > 0 else 0. if x < 0 else 50.)
        fee = 2. * min(p, 100. - p) / 100.
        edge = (fv - p if d == 1 else p - fv) - fee
        r250 = d * 1e4 * math.log(self.mid(t) / self.mid(t - 250.))
        return dict(tau=tau, x_bps=x, sigma=sg, z=x / (sg * math.sqrt(tau)) if sg > 0 else np.nan, fv=fv, edge_c=edge, r250=r250, price=p)


def mid_at(f, t):
    j = int(np.searchsorted(f[:, 0], t, 'right') - 1)
    if j < 0:
        return None
    b, a = f[j, 2], f[j, 12]
    return float((b + a) / 2) if 0 <= b < a <= 100 else None


def main():
    fn = H / 'selection.json'
    C.require(not fn.exists(), 'selection.json exists (write-once)')
    sm = json.loads((WT.S1D / 'summary.json').read_text())
    pay = {(r['day'], int(r['mid'])): r['y'] for r in sm['rows']}
    want = collections.defaultdict(set)
    for (d, m) in pay:
        if d in DAYS:
            want[d].add(m)
    src, rows = MM.KP.Tape(), []
    for day in DAYS:
        btc = Btc(day)
        L = pd.read_parquet(MM.S1 / day / 'legs_timed2.parquet', columns=['mid', 'side', 'ptick', 'shares', 't_ms2', 'how2'])
        L = L[L.mid.isin(want[day]) & (L.how2 == 1)]
        by_mid = {m: g for m, g in L.groupby('mid')}
        reg = MM.registry(day).set_index('mid'); off = src.offset(day)
        for h, g in reg.groupby('hour'):
            mids = [int(m) for m in g.index if int(m) in want[day]]
            if not mids:
                continue
            for m, st, p, info, yup in F.worlds_yup(day, mids, src, off, reg):
                C.require(st == 'complete', f'{m} {st}')
                t0 = float(info['t_start']); ws = float(reg.loc[m, 'ws']) * 1000.; f = np.asarray(p['frames'], float); y = 100. * pay[(day, m)]
                # WINNERS: the consuming legs of our missed 172 ms grid takes
                lg = by_mid.get(m)
                led = json.loads(gzip.decompress((WT.S1D / f'{m}_BASE.json.gz').read_bytes()))
                seen = set()
                if lg is not None:
                    pt = lg.t_ms2.values.astype(float) - t0
                    for x in led['ioc_results']:
                        if x['reason'] != 'limit_moved':
                            continue
                        fo = WT.follow(p, x)
                        if fo['gone_ms'] is None:
                            continue
                        s, Lp, gd = int(x['side']), float(x['limit']), float(x['submit']); opp = 1 - s
                        jk = int(np.searchsorted(f[:, 1], gd, side='right') - 1)
                        sel = (lg.side.values == opp) & ((lg.ptick.values >= Lp) if opp == 1 else (lg.ptick.values <= Lp)) & (pt >= f[jk, 0]) & (pt <= gd + fo['gone_ms'] + 100.)
                        dd = 1 if s == 0 else -1
                        for i in np.flatnonzero(sel):
                            key = (float(lg.t_ms2.values[i]), int(lg.side.values[i]), float(lg.ptick.values[i]))
                            if key in seen:
                                continue
                            seen.add(key)
                            pr = float(lg.ptick.values[i]); A = t0 + pt[i] - LV.HOLD
                            stt = btc.state(A, ws, dd, pr)
                            v1, v10 = mid_at(f, pt[i] + 1000.), mid_at(f, pt[i] + 10000.)
                            fee = 2. * min(pr, 100. - pr) / 100.
                            rows.append(dict(group='winners', day=day, mid=m, shares=float(lg[(lg.t_ms2 == lg.t_ms2.values[i]) & (lg.side == lg.side.values[i])].shares.sum()), **stt,
                                             v1=None if v1 is None else dd * (v1 - pr) - fee, v10=None if v10 is None else dd * (v10 - pr) - fee, vset=dd * (y - pr) - fee))
                # OURS: STREAM-172 filled takes
                rec = json.loads(gzip.decompress((S.OUT / f'{m}_STREAM-172.json.gz').read_bytes()))
                tm = {int(x['gid']): x for x in rec['take_marks']}
                for x in rec['ioc_results']:
                    if x['filled'] <= S.EPS:
                        continue
                    dd = 1 if x['side'] == 0 else -1; pr = float(x['price'])
                    stt = btc.state(t0 + float(x['submit']), ws, dd, pr)
                    mk = tm.get(int(x['gid']), {})
                    fee = x['fee_c'] / x['filled']
                    rows.append(dict(group='ours', day=day, mid=m, shares=float(x['filled']), **stt, v1=mk.get('m1s_c'), v10=mk.get('m10s_c'), vset=dd * (y - pr) - fee))
        print(day, collections.Counter(r['group'] for r in rows), flush=True)
    D = pd.DataFrame(rows)
    S.OUT.mkdir(parents=True, exist_ok=True)
    D.to_parquet(S.OUT / 'selection_fills.parquet', index=False)

    def wq(d, c, qs=(.25, .5, .75)):
        x = d[c].to_numpy(float); w = d.shares.to_numpy(float); ok = np.isfinite(x)
        if not ok.any():
            return None
        o = np.argsort(x[ok]); cw = np.cumsum(w[ok][o]) / w[ok].sum()
        return [float(x[ok][o][min(np.searchsorted(cw, q), ok.sum() - 1)]) for q in qs]

    def wavg(d, c):
        x = d[c].to_numpy(float); w = d.shares.to_numpy(float); ok = np.isfinite(x)
        return float((x[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else None
    out = dict(asof='2026-11-25', days=list(DAYS), fills=dict(collections.Counter(D.group)), shares={g: float(D[D.group == g].shares.sum()) for g in ('winners', 'ours')},
               state_quantiles={}, value_by_edge={}, shares_by_edge={})
    for g in ('winners', 'ours'):
        X = D[D.group == g]
        out['state_quantiles'][g] = {c: wq(X, c) for c in ('edge_c', 'tau', 'price', 'z', 'r250', 'sigma')}
        X = X.assign(eb=pd.cut(X.edge_c, EDGE_BINS))
        out['value_by_edge'][g] = {str(b): dict(shares=float(G.shares.sum()), v1=wavg(G, 'v1'), v10=wavg(G, 'v10'), vset=wavg(G, 'vset')) for b, G in X.groupby('eb', observed=True)}
        out['shares_by_edge'][g] = {str(b): float(G.shares.sum() / X.shares.sum()) for b, G in X.groupby('eb', observed=True)}
    TR.dump(fn, out)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
