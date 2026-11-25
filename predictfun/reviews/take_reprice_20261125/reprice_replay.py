"""TAKE-REPRICE v1 (declaration.md): takes only when the Binance move implies a large repricing of the binary (dFV >= D cents), decided
on the stream, through stream_replay.StreamBranch under the venue-matched harness. 408 complete markets, five consumed days.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/reprice_replay.py \
       selftest | day <day>
"""
import collections
import json
import math
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
R = H.parents[2]
sys.path.insert(0, str(H.parent / 'take_stream_20261125'))
import stream_replay as S  # noqa: E402  (frozen at 4c5cf27de / ad8e368e6: StreamBranch, one_stream, the venue harness)

VR, V, F, OD, MM, TR, C = S.VR, S.V, S.F, S.OD, S.MM, S.TR, S.C
OUT = S.X.DATA / 'take_reprice_20261125'
SIG_WIN, QUIET, LAG = 300000., 2000., 250.
CONFIGS = {'REPRICE-4-172': (4., 22.), 'REPRICE-8-172': (8., 22.), 'REPRICE-16-172': (16., 22.), 'REPRICE-8-192': (8., 42.)}
DAY_CPU_CAP, TOTAL_CPU_CAP, FIRST_DAY = 900., 2800., '20261023'
FILES = ['declaration.md', 'reprice_replay.py']
ERF = np.vectorize(math.erf)


def Phi(z):
    return 0.5 * (1. + ERF(z / math.sqrt(2.)))


def reprice_triggers(bk, bm, cs, ws, lo, hi, D):
    """dFV triggers of one market: breakpoints (change points, change points + 250 ms) in [lo - QUIET, hi]; sigma over the 300 s ending
    250 ms before (the judged move excluded); OFF->ON with no ON on that side in the previous 2 s -> sorted [(t_break, dir)] with t >= lo."""
    jw = int(np.searchsorted(bk, ws, 'right') - 1)
    if jw < 0:
        return []
    f_ws = bm[jw]
    i0, i1 = int(np.searchsorted(bk, lo - QUIET - LAG, 'left')), int(np.searchsorted(bk, hi, 'right'))
    k = bk[i0:i1]
    bp = np.unique(np.r_[k, k + LAG]); bp = bp[(bp >= lo - QUIET) & (bp <= hi)]
    j0, j1 = np.searchsorted(bk, bp, 'right') - 1, np.searchsorted(bk, bp - LAG, 'right') - 1
    js = np.searchsorted(bk, bp - LAG - SIG_WIN, 'right') - 1
    ok = (j0 >= 0) & (j1 >= 0)
    x = np.where(ok, 1e4 * np.log(bm[np.maximum(j0, 0)] / f_ws), np.nan)
    x250 = np.where(ok, 1e4 * np.log(bm[np.maximum(j1, 0)] / f_ws), np.nan)
    span = (bp - LAG - np.maximum(bp - LAG - SIG_WIN, bk[0])) / 1000.
    var = cs[np.maximum(j1, 0)] - cs[np.maximum(js, 0)]
    sig = np.where(ok & (span > 1.), 1e4 * np.sqrt(np.maximum(var, 0.) / np.maximum(span, 1.)), np.nan)
    tau = (ws + 300000. - bp) / 1000.
    good = ok & np.isfinite(sig) & (sig > 0) & (tau >= .5)
    s = np.where(good, sig * np.sqrt(np.maximum(tau, .5)), np.nan)
    dfv = np.where(good, 100. * (Phi(x / s) - Phi(x250 / s)), np.nan)
    out = []
    for d, on in ((1, good & (dfv >= D)), (-1, good & (dfv <= -D))):
        last_end, prev = -np.inf, False
        for i in range(len(bp)):
            if on[i]:
                if not prev and bp[i] - last_end >= QUIET and bp[i] >= lo:
                    out.append((float(bp[i]), d))
                last_end = float(bp[i + 1]) if i + 1 < len(bp) else np.inf
            prev = bool(on[i])
    return sorted(out)


def bound():
    for f in FILES:
        C.require(subprocess.check_output(['git', 'show', 'HEAD:' + str((H / f).relative_to(R))], cwd=R) == (H / f).read_bytes(), f'{f} is not committed')
    S.bound()


def day_run(day):
    resource.setrlimit(resource.RLIMIT_CPU, (int(DAY_CPU_CAP) + 60, int(DAY_CPU_CAP) + 61))
    bound()
    C.require(day in OD.DAYS3, f'{day} is not a declared day')
    fn = OUT / f'day_{day}.json'
    C.require(not fn.exists(), f'{fn.name} exists (write-once)')
    if day != FIRST_DAY:
        pj = json.loads((H / 'projection.json').read_text())
        C.require(pj['projected_cpu_s'] <= TOTAL_CPU_CAP, 'the first-day projection exceeds the cap')
    OUT.mkdir(parents=True, exist_ok=True)
    vs = json.loads((VR.OUT / 'summary.json').read_text())
    comp = {int(r['mid']): r['payout'] for r in vs['rows'] if r['day'] == day and r['status'] == 'complete'}
    bk, bm = OD.seg_arrays(OD.SEG_OF[day]); bk = bk.astype(float); bm = bm.astype(float)
    cs = np.cumsum(np.r_[0., np.diff(np.log(bm))] ** 2)
    edges, src = TR.T.P.frozen_edges(), MM.KP.Tape()
    reg = MM.registry(day).set_index('mid'); off = src.offset(day); t0c = time.process_time()
    rows = []
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
            cache = OD.make_cache3(t0, end, OD.SEG_OF[day])
            old = dict(end=end, ident=MM.ident_of(day, m, ws, info, p))
            y = comp[m]
            row = dict(day=day, mid=m, hour=int(h), payout=y, arms={}, n_triggers={})
            trig = {}
            for D in sorted({c[0] for c in CONFIGS.values()}):
                tr = [(x + S.FEED + S.DECIDE - t0, dd) for x, dd in reprice_triggers(bk, bm, cs, ws, t0 - S.FEED - S.DECIDE, t0 + end, D)]
                tr = [(x, dd) for x, dd in tr if 0. <= x < end]
                trig[D] = (np.array([x for x, _ in tr], float), np.array([dd for _, dd in tr], int), np.array([f'reprice{int(D)}'] * len(tr)))
                row['n_triggers'][str(int(D))] = len(tr)
            for name, (D, net) in CONFIGS.items():
                rec = S.one_stream(p, cache, old, edges, 200., V.BASE_SLACK, trig[D], net)
                rec.update(day=day, mid=m, config=name)
                sha = F.write_once_gz(OUT / f'{m}_{name}.json.gz', rec)
                a = VR.native_account(rec, y)
                a.update(sha256=sha, sig_status=dict(collections.Counter(r_['status'] for r_ in rec['signal_log'])),
                         ioc_status=rec['metrics']['ioc_status'], take_value=S.take_value(rec))
                row['arms'][name] = a
            rows.append(row)
            C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    spent = time.process_time() - t0c
    if day == FIRST_DAY:
        pfn = H / 'projection.json'
        C.require(not pfn.exists(), 'projection.json exists (write-once)')
        TR.dump(pfn, dict(asof='2026-11-25', first_day=day, spent_cpu_s=spent, markets=len(rows), projected_cpu_s=spent * 408 / max(len(rows), 1) * 1.1,
                          cap_cpu_s=TOTAL_CPU_CAP))
    TR.dump(fn, dict(asof='2026-11-25', day=day, rows=rows, cpu_s=spent, peak_rss_mib=F.rss_mib()))
    print(f'{day}: {len(rows)} markets, cpu {spent:.0f} s', flush=True)


def selftest():
    rng = np.random.default_rng(7)
    t = np.arange(0., 400000., 100.)                                          # 100 ms change points, small noise (sigma ~ 0.1 bps / sqrt s)
    lr = rng.normal(0., .1e-4 * math.sqrt(.1), len(t))
    bm = 80000. * np.exp(np.cumsum(lr))
    jump = int(np.searchsorted(t, 350000.))
    bm2 = bm.copy(); bm2[jump:] *= np.exp(3e-4)                                 # +3 bps at 350 s, then a second +3 bps at 351 s
    bm2[jump + 10:] *= np.exp(3e-4)
    cs = np.cumsum(np.r_[0., np.diff(np.log(bm2))] ** 2)
    near = reprice_triggers(t, bm2, cs, ws=200000., lo=200000., hi=399000., D=8.)
    assert any(abs(x - 350000.) < 1. and d == 1 for x, d in near), near         # near the strike (window started at 200 s): fires
    assert not any(350500. < x < 352500. and d == 1 for x, d in near), near    # the second jump inside the 2 s quiet: no repeat
    far = bm2.copy(); far[int(np.searchsorted(t, 210000.)):] *= np.exp(30e-4)   # 30 bps above the strike from 10 s into the window
    csf = np.cumsum(np.r_[0., np.diff(np.log(far))] ** 2)
    farr = reprice_triggers(t, far, csf, ws=200000., lo=200500., hi=399000., D=8.)
    assert not any(abs(x - 350000.) < 1. for x, d in farr), farr               # the same jump far from the strike: no fire
    print('reprice selftest PASS')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'day':
        day_run(sys.argv[2])
    else:
        dict(selftest=selftest)[cmd]()
