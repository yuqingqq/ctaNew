"""Are the "early" takers predicting the move, or are taker orders simply always in flight? (USER 2026-11-25: find the earlier signals).
Diagnostic on the REPRICE-8-172 sent takes (fill_rate_takes.parquet), 408 complete markets, five consumed days.

For a window (t, t + 150 ms] (an order arriving in it was sent before t, every taker waits the 150 ms hold): the best level on the side a
take in direction d would consume (the ask for d = +1), from the book at t (venue stamps); HIT = public trades at or through that price
inside the window reach half its shown size; ANY = at least one such trade; their value = d x (mid 1 s later - price) - 2 % x min(p, 1 - p).
Windows: t = the Binance onset of each event (the quote is stale from here on, so an arriving order was sent before the move); t = onset -
150 (orders that arrived just before the move); null = t shifted by +-5, 10, .., 30 s inside the same market (12 per event), same d.
If taker orders are always in flight, HIT after the onset equals the null rate; the excess over the null is what anticipation (or activity
clustering) must explain. Output (write-once): inflight.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh inflight -- python3 predictfun/reviews/take_reprice_20261125/inflight.py
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
import reprice_replay as RP  # noqa: E402
import why_markout as WM  # noqa: E402

S, VR, F, OD, MM, TR, C = RP.S, RP.VR, RP.F, RP.OD, RP.MM, RP.TR, RP.C
WIN, CFG = 150., 'REPRICE-8-172'
SHIFTS = np.array([-30, -25, -20, -15, -10, -5, 5, 10, 15, 20, 25, 30], float) * 1000.


def window(f, tr, end, d, t):
    """HIT / ANY / value of the trades on the stale side's best level at t inside (t, t + WIN]; None when no valid book."""
    if t < 0 or t + WIN + 1000. > end:
        return None
    j = int(np.searchsorted(f[:, 0], t, 'right') - 1)
    if j < 0:
        return None
    opp = 1 if d == 1 else 0
    P = float(f[j, 2 + opp * 10]); q0 = float(f[j, 7 + opp * 10])
    if not (0 < P < 100 and q0 > 0 and 0 <= f[j, 2] < f[j, 12] <= 100):
        return None
    m = (tr[:, 1] == opp) & ((tr[:, 2] >= P) if opp == 1 else (tr[:, 2] <= P)) & (tr[:, 0] > t) & (tr[:, 0] <= t + WIN)
    qty = float(tr[m, 3].sum())
    v = []
    for tt, _, pp, qq in tr[m]:
        mid = WM.mid_ok(f, tt + 1000.)
        if mid is not None:
            v.append((d * (mid - pp) - WM.fee(pp), qq))
    return dict(hit=qty >= .5 * q0, any=qty > 0, val=v)


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    fn = H / 'inflight.json'
    C.require(not fn.exists(), 'inflight.json exists (write-once)')
    T = pd.read_parquet(RP.OUT / 'fill_rate_takes.parquet')
    T = T[(T.config == CFG) & T.cls.isin(['taken', 'filled', 'pulled']) & T.bn_lag.notna()]
    vs = json.loads((VR.OUT / 'summary.json').read_text())
    src, acc = MM.KP.Tape(), collections.defaultdict(lambda: dict(n=0, hit=0, any=0, v=0., q=0.))
    for day, G in T.groupby('day'):
        comp = {int(r['mid']) for r in vs['rows'] if r['day'] == day and r['status'] == 'complete'}
        reg = MM.registry(day).set_index('mid'); off = src.offset(day)
        by_mid = {m: g for m, g in G.groupby('mid')}
        for h, g in reg.groupby('hour'):
            mids = [int(m) for m in g.index if int(m) in comp and int(m) in by_mid]
            if not mids:
                continue
            for m, W, lst in VR.worlds_all(day, mids, src, off, reg):
                C.require(lst == 'loaded', f'{m} {lst}')
                ws = float(reg.loc[m, 'ws']) * 1000.
                p, st, info = VR.vm_path(W, ws, 50.)
                C.require(st == 'complete', f'{m} {st}')
                end = float(info['end_rel']); f = np.asarray(p['frames'], float); tr = np.asarray(p['trades'], float).reshape(-1, 4)
                for r in by_mid[m].itertuples():
                    t_on = r.g - S.FEED - S.DECIDE - r.bn_lag
                    grp = 'taken' if r.cls == 'taken' else 'not_taken'
                    for key, t in [(f'onset|{grp}', t_on), (f'pre_onset|{grp}', t_on - WIN)] + [(f'null|{grp}', t_on + s) for s in SHIFTS]:
                        w = window(f, tr, end, int(r.d), t)
                        if w is None:
                            continue
                        a = acc[key]; a['n'] += 1; a['hit'] += int(w['hit']); a['any'] += int(w['any'])
                        for v, q in w['val']:
                            a['v'] += v * q; a['q'] += q
                C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
        print(day, flush=True)
    out = dict(asof='2026-11-25', config=CFG, window_ms=WIN, shifts_s=(SHIFTS / 1000.).tolist(), windows={})
    for k, a in sorted(acc.items()):
        out['windows'][k] = dict(n=a['n'], hit=a['hit'] / a['n'] if a['n'] else None, any=a['any'] / a['n'] if a['n'] else None,
                                 traded_shares=a['q'], value_c_per_share=a['v'] / a['q'] if a['q'] else None)
    for g in ('taken', 'not_taken'):
        o, n = out['windows'].get(f'onset|{g}'), out['windows'].get(f'null|{g}')
        if o and n:
            out[f'excess_hit_{g}'] = o['hit'] - n['hit']
    TR.dump(fn, out)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
