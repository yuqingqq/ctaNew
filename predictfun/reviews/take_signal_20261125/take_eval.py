"""TAKE-SIGNAL v1 stage 2 (contract.md): every qualifying BTC trigger valued as a hypothetical IOC take on the Predict book, Oct 24-27.
Inputs: bn_changes.parquet (stage 1), maker_trace_v2 markets_<day> / tob_<day>. Output (write-once): takes.parquet (one row per kept trigger
per clock) and events_all.parquet (every qualifying event with market, direction, kept flag) + take_eval.json.
Usage: run_light.sh ts_take -- python3 predictfun/reviews/take_signal_20261125/take_eval.py
"""
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

H = Path(__file__).resolve().parent
R = H.parents[2]
V = R / 'data/predict_fun/research/maker_trace_v2_20261123'
OUT = R / 'data/predict_fun/research/take_signal_20261125'
DAYS = ('20261024', '20261025', '20261026', '20261027')
UP_EDGE, DOWN_EDGE = 0.285456120967865, 0.4666730940341952       # frozen pressure-guard edges (asks, bids), never recomputed
COOLDOWN_MS = 2000
HORIZONS = (1000, 3000, 10000, 30000)
CLOCKS = {'tokyo': dict(b_in=5, p_in=63, arrive=42), 'slow': dict(b_in=10, p_in=126, arrive=86)}
KEY = 10 ** 13
BAD = ('bad_frame', 'mid_mismatch', 'out_of_order', 'off_grid')
TBE = [0, 30, 60, 180, 240, 270, 298]
TBL = ['000-030', '030-060', '060-180', '180-240', '240-270', '270-298']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_tob(day, mindex):
    t = pq.read_table(V / f'tob_{day}.parquet', columns=['mid', 'upd_ms', 'recv_ns', 'best_bid', 'best_bid_sh', 'best_ask', 'best_ask_sh',
                                                         'status'], use_threads=False).to_pandas()
    t['midx'] = t.mid.map(mindex)
    assert t.midx.notna().all()
    t = t.sort_values(['midx', 'upd_ms', 'recv_ns'], kind='mergesort')
    return dict(key=t.midx.to_numpy(np.int64) * KEY + t.upd_ms.to_numpy(np.int64), midx=t.midx.to_numpy(np.int64), upd=t.upd_ms.to_numpy(np.int64),
                bid=t.best_bid.to_numpy(float), ask=t.best_ask.to_numpy(float), bsh=t.best_bid_sh.to_numpy(float), ash=t.best_ask_sh.to_numpy(float),
                bad=t.status.fillna('').astype(str).str.contains('|'.join(BAD), regex=True).to_numpy(bool))


def frame_last(T, midx, x):
    """Index of the last frame with upd_ms <= x in market midx (-1 if none)."""
    i = np.searchsorted(T['key'], midx * KEY + x, 'right') - 1
    ok = (i >= 0)
    j = np.where(ok, i, 0)
    return np.where(ok & (T['midx'][j] == midx), i, -1)


def frame_next(T, midx, x):
    """Index of the first frame with upd_ms > x in market midx (-1 if none)."""
    i = np.searchsorted(T['key'], midx * KEY + x, 'right')
    ok = i < len(T['key'])
    j = np.where(ok, i, 0)
    return np.where(ok & (T['midx'][j] == midx), i, -1)


def mid_ok(T, i, x):
    """Two-sided ok mid of frame i read at clock x (econ MID rule), NaN otherwise."""
    j = np.where(i >= 0, i, 0)
    b, a = T['bid'][j], T['ask'][j]
    ok = (i >= 0) & np.isfinite(b) & np.isfinite(a) & (b < a) & ~T['bad'][j] & (x - T['upd'][j] <= 1000)
    return np.where(ok, (b + a) / 2, np.nan), ok


def side_px(T, i, up):
    j = np.where(i >= 0, i, 0)
    px = np.where(up, T['ask'][j], T['bid'][j])
    sh = np.where(up, T['ash'][j], T['bsh'][j])
    ok = (i >= 0) & np.isfinite(px) & ~T['bad'][j]
    return np.where(ok, px, np.nan), np.where(ok, sh, np.nan)


def events_for(bk, bm, b_in, lo, hi):
    """BTC change points (decision clock t = E + b_in) in [lo, hi) with bn_r250 / bn_r1000."""
    t = bk + b_in
    sel = (t >= lo) & (t < hi)
    ti = t[sel]
    mi = bm[sel]
    j250 = np.searchsorted(t, ti - 250, 'right') - 1
    j1000 = np.searchsorted(t, ti - 1000, 'right') - 1
    r250 = np.where(j250 >= 0, 1e4 * np.log(mi / bm[np.maximum(j250, 0)]), np.nan)
    r1000 = np.where(j1000 >= 0, 1e4 * np.log(mi / bm[np.maximum(j1000, 0)]), np.nan)
    return ti, r250, r1000


def main():
    t0 = time.process_time()
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    outs = {n: OUT / n for n in ('takes.parquet', 'events_all.parquet')}
    for p in outs.values():
        assert not p.exists(), f'write-once: {p.name}'
    bn = pd.read_parquet(OUT / 'bn_changes.parquet')
    bk, bm = bn.k_ms.to_numpy(np.int64), bn.mid.to_numpy(float)
    rows, evs, rec = [], [], dict(days={})
    for day in DAYS:
        mk = pd.read_parquet(V / f'markets_{day}.parquet', columns=['mid', 'ws', 'we', 'outcome', 'feeRateBps', 'status'])
        mindex = {int(m): i for i, m in enumerate(sorted(mk.mid))}
        T = load_tob(day, mindex)
        cens = mk.status.astype(str).str.contains('boundary_censored')
        dr = dict(markets=int((~cens).sum()), censored=int(cens.sum()), outcomes=mk.outcome.astype(str).value_counts().to_dict())
        for clk, c in CLOCKS.items():
            for r in mk[~cens].itertuples():
                ws, end = int(r.ws) * 1000, int(r.we) * 1000 - 2000
                ti, r250, r1000 = events_for(bk, bm, c['b_in'], ws, end)
                up, dn = r250 >= UP_EDGE, r250 <= -DOWN_EDGE
                q = up | dn
                if not q.any():
                    continue
                ti, r250, r1000, d = ti[q], r250[q], r1000[q], np.where(up[q], 1, -1)
                keep = np.zeros(len(ti), bool)
                for s in (1, -1):
                    w = np.where(d == s)[0]
                    if len(w):
                        gap = np.r_[np.inf, np.diff(ti[w])]
                        keep[w[gap >= COOLDOWN_MS]] = True
                if clk == 'tokyo':
                    evs.append(pd.DataFrame(dict(day=day, mid=int(r.mid), t=ti, t_rel_s=(ti - ws) / 1000, dir=d, r250=r250, r1000=r1000, kept=keep)))
                ti, r250, r1000, d = ti[keep], r250[keep], r1000[keep], d[keep]
                n = len(ti)
                midx = np.full(n, mindex[int(r.mid)], np.int64)
                upb = d == 1
                xk = ti - c['p_in']
                ik = frame_last(T, midx, xk)
                kmid, kok = mid_ok(T, ik, xk)
                kpx, _ = side_px(T, ik, upb)
                jk = np.where(ik >= 0, ik, 0)
                kspr = np.where(kok, (T['ask'][jk] - T['bid'][jk]) * 100, np.nan)
                ip = frame_last(T, midx, xk - 1000)
                pmid, _ = mid_ok(T, ip, xk - 1000)
                ta = ti + c['arrive']
                f1, f2 = frame_last(T, midx, ta), frame_next(T, midx, ta)
                p1, s1 = side_px(T, f1, upb)
                p2, s2 = side_px(T, f2, upb)
                lim = kpx
                exe1 = np.where(upb, p1 <= lim + 1e-9, p1 >= lim - 1e-9)
                exe2 = np.where(upb, p2 <= lim + 1e-9, p2 >= lim - 1e-9)
                st = np.full(n, 'filled', dtype=object)
                st[~(exe1 & exe2)] = 'not_executable'
                st[f2 < 0] = 'no_next_frame'
                st[~kok | ~np.isfinite(kpx)] = 'no_known_book'
                avail = np.where(st == 'filled', np.minimum(s1, s2), 0.0)
                price = lim
                fee = r.feeRateBps / 1e4 * np.minimum(price, 1 - price)
                vv = {'Up': 1.0, 'Down': 0.0}.get(str(r.outcome), 0.5 if str(r.outcome).lower() in ('tie', 'draw') else np.nan)
                out = dict(day=day, mid=int(r.mid), clock=clk, t=ti, t_rel_s=(ti - ws) / 1000, dir=d, r250=r250, r1000=r1000, status=st,
                           known_mid=kmid, known_spread_c=kspr, limit=lim, avail_sh=avail, fee=fee, V=vv,
                           stale_c=d * (kmid - pmid) * 100)
                for h in HORIZONS:
                    x = ta + h
                    ih = frame_last(T, midx, x)
                    mh, okh = mid_ok(T, ih, x)
                    cen = x > end
                    out[f'mark_{h // 1000}s'] = np.where(cen | ~okh, np.nan, d * (mh - price) - fee)
                    ex, _ = side_px(T, ih, ~upb)
                    exf = r.feeRateBps / 1e4 * np.minimum(ex, 1 - ex)
                    out[f'rt_{h // 1000}s'] = np.where(cen | ~okh | ~np.isfinite(ex), np.nan, d * (ex - price) - fee - exf)
                    out[f'cens_{h // 1000}s'] = cen
                ie = frame_last(T, midx, np.full(n, end))
                me, oke = mid_ok(T, ie, np.full(n, end))
                out['end'] = np.where(oke, d * (me - price) - fee, np.nan)
                out['res'] = d * (vv - price) - fee
                rows.append(pd.DataFrame(out))
        rec['days'][day] = dr
        del T
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 < 2500, 'VmHWM guard'
    takes = pd.concat(rows, ignore_index=True)
    ev = pd.concat(evs, ignore_index=True)
    takes['strength'] = pd.cut(takes.r250.abs(), [0, 0.5, 1, 2, np.inf], right=False, labels=['edge-0.5', '0.5-1', '1-2', '>=2']).astype(str)
    takes['tb'] = pd.cut(takes.t_rel_s, TBE, right=False, labels=TBL).astype(str)
    takes['money'] = np.where((takes.known_mid > 0.10) & (takes.known_mid < 0.90), 'NEAR', np.where(takes.known_mid.notna(), 'EXTREME', 'na'))
    rec.update(takes=len(takes), by_clock_status={f'{a}|{b}': int(v) for (a, b), v in takes.groupby(['clock', 'status']).size().items()},
               events_all=len(ev), events_kept=int(ev.kept.sum()), inputs={'bn_changes.parquet': sha(OUT / 'bn_changes.parquet')},
               prior_attempt='2026-11-25T03:27:44Z ts_take rc=1: receipt json refused tuple keys after the two outputs were written; '
                             'those partial outputs were removed and the run repeated with string keys (no input or rule change)')
    json.dumps(rec, default=str)                                   # the receipt must serialise BEFORE any output is written
    takes.to_parquet(outs['takes.parquet'], index=False)
    ev.to_parquet(outs['events_all.parquet'], index=False)
    rec.update(cpu_s=round(time.process_time() - t0, 2), vmhwm_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
               outputs={k: sha(p) for k, p in outs.items()})
    (H / 'take_eval.json').write_text(json.dumps(rec, indent=1, default=str) + '\n')
    print(json.dumps({k: v for k, v in rec.items() if k != 'days'}, default=str))
    print(json.dumps(rec['days']))


if __name__ == '__main__':
    main()
