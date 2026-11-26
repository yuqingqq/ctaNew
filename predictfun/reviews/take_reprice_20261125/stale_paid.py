"""What price did each fill pay against the book its order found on arrival? (USER 2026-11-26: "do we catch the stale quotes of makers ...
smart takers can take those stale quotes"; STALE_QUOTES left open whether the top taker's fills on quotes posted during its hold were paid
above the best it found on arrival, i.e. a limit through the book, or were price improvements). Readout over stale_quotes' fill rows.
Per fill (book side b consumed at price L, path time t, arrival t - 150 ms, frame clock): the best opposite price in the last frame at or
before the arrival (the ask for b = 1, the bid for b = 0); paid through = d x (L - that best) in cents (> 0: paid more than the best on
arrival; < 0: price improvement); whether that best price still showed size 25 ms before the fill. 'no_arrival_frame' when no frame
precedes the arrival or its best is empty (counted). Declared before the run: per group (TOP, RANKS2_10, OTHERS, SCORE1, RAND1) and per
stale_quotes partition part: the volume shares paid through > 0 / = 0 / < 0, share-weighted quantiles of paid through, value per share
1 s later by paid-through bucket (< 0, 0, 1, 2, >= 3 c), and the share whose arrival best had left the book before the fill.
Output (write-once): stale_paid.json; OUT/stale_quotes/<day>_paid.parquet.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/stale_paid.py \
       day <day> [<rehearsal dir> <n markets>] | merge [<rehearsal dir> <day> ...]
"""
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import stale_quotes as SQ  # noqa: E402

VR, MM, TR, C, F, WM = SQ.VR, SQ.MM, SQ.TR, SQ.C, SQ.F, SQ.WM
OUT = SQ.OUT
PB = [-np.inf, -.5, .5, 1.5, 2.5, np.inf]
PLAB = ['< 0', '0', '1', '2', '>= 3']


def day_run(day, od=None, limit=None):
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    C.require(day in SQ.DAYS, f'{day} is not a declared day')
    od = Path(od) if od else OUT
    fp = od / f'{day}_paid.parquet'
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    Fd = pd.read_parquet(OUT / f'{day}_fills.parquet')
    Fd = Fd[Fd.status != 'outside_path'].reset_index(drop=True)
    vs = json.loads((VR.OUT / 'summary.json').read_text())
    comp = {int(r['mid']) for r in vs['rows'] if r['day'] == day and r['status'] == 'complete'}
    src = MM.KP.Tape()
    reg = MM.registry(day).set_index('mid'); off = src.offset(day)
    best, still, n_m = np.full(len(Fd), np.nan), np.full(len(Fd), np.nan), 0
    by_mid = {m: g.index.to_numpy() for m, g in Fd.groupby('mid')}
    for h, g in reg.groupby('hour'):
        mids = [int(m) for m in g.index if int(m) in comp]
        if limit is not None:
            mids = mids[:max(0, limit - n_m)]
        if not mids:
            continue
        for m, W, lst in VR.worlds_all(day, mids, src, off, reg):
            C.require(lst == 'loaded', f'{m} {lst}')
            ws = float(reg.loc[m, 'ws']) * 1000.
            p, st, info = VR.vm_path(W, ws, 50.)
            C.require(st == 'complete', f'{m} {st}')
            n_m += 1
            if m not in by_mid:
                continue
            f = np.asarray(p['frames'], float)
            ix = by_mid[m]
            t, b = Fd.t_rel.to_numpy(float)[ix], Fd.side_b.to_numpy(int)[ix]
            ja = np.searchsorted(f[:, 0], t - SQ.HOLD, side='right') - 1
            jm = np.searchsorted(f[:, 0], t - SQ.PRE_MS, side='right') - 1
            col = np.where(b == 1, 12, 2)
            ok = ja >= 0
            bb = np.where(ok, f[np.maximum(ja, 0), col], np.nan)
            bb = np.where(bb >= 0, bb, np.nan)
            best[ix] = bb
            for k, i in enumerate(ix):
                if np.isfinite(bb[k]) and jm[k] >= 0:
                    still[i] = float(SQ.WM.WT.level(f[jm[k]], int(b[k]), float(bb[k])) > WM.EPS)
            C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    Fd['best_on_arrival'] = best
    Fd['paid_through'] = Fd.d * (Fd.price - Fd.best_on_arrival)
    Fd['arrival_best_still_there'] = still
    Fd['paid_status'] = np.where(Fd.best_on_arrival.isna(), 'no_arrival_frame', 'ok')
    od.mkdir(parents=True, exist_ok=True)
    Fd[['day', 'mid', 'grp', 'order', 't_rel', 'side_b', 'd', 'price', 'q', 'status', 'best_on_arrival', 'paid_through', 'arrival_best_still_there', 'paid_status']].to_parquet(fp, index=False)
    print(day, n_m, Fd.groupby('grp').paid_through.median().to_dict(), flush=True)


def merge(od=None, days=None):
    days = days or SQ.DAYS
    fn = (Path(od) if od else H) / 'stale_paid.json'
    C.require(not fn.exists(), 'stale_paid.json exists (write-once)')
    src = Path(od) if od else OUT
    P = pd.concat([pd.read_parquet(src / f'{d}_paid.parquet') for d in days], ignore_index=True)
    Fd = pd.concat([pd.read_parquet(OUT / f'{d}_fills.parquet') for d in days], ignore_index=True)
    Fd = Fd[Fd.status != 'outside_path'].reset_index(drop=True)
    C.require(len(Fd) == len(P) and (Fd.t_rel.to_numpy() == P.t_rel.to_numpy()).all() and (Fd.grp.to_numpy() == P.grp.to_numpy()).all(), 'paid rows do not align with the fill rows')
    Fd['paid_through'] = P.paid_through.to_numpy(); Fd['still'] = P.arrival_best_still_there.to_numpy(); Fd['paid_status'] = P.paid_status.to_numpy()
    Fd['part'] = SQ.part_of(Fd).to_numpy()
    out = dict(asof='2026-11-26', days=list(days), groups={})

    def block(X):
        Y = X[X.paid_status == 'ok']
        if not len(Y):
            return dict(fills=int(len(X)))
        w = Y.q.astype(float); pt = Y.paid_through.astype(float)
        bk = pd.cut(pt, PB, labels=PLAB)
        return dict(fills=int(len(X)), no_arrival_frame_share=float(X.q[X.paid_status != 'ok'].sum() / X.q.sum()),
                    paid_more=float(w[pt > .5].sum() / w.sum()), paid_same=float(w[pt.abs() <= .5].sum() / w.sum()), improved=float(w[pt < -.5].sum() / w.sum()),
                    paid_through=SQ.wq(pt, w), mean_paid_through=SQ.wavg(pt, w),
                    arrival_best_gone=float(w[Y.still == 0.].sum() / w[Y.still.notna()].sum()) if Y.still.notna().any() else None,
                    by_paid={str(k): dict(share=float(G.q.sum() / w.sum()), v1=SQ.wavg(G.v1, G.q), fills=int(len(G))) for k, G in Y.groupby(bk, observed=True)})
    for g in SQ.GROUPS:
        X = Fd[Fd.grp == g]
        if not len(X):
            continue
        o = block(X)
        o['by_part'] = {k: block(G) for k, G in X.groupby('part')}
        out['groups'][g] = o
    TR.dump(fn, out)
    print(json.dumps(out, indent=1)[:5000])


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'day':
        day_run(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None, int(sys.argv[4]) if len(sys.argv) > 4 else None)
    elif cmd == 'merge':
        merge(sys.argv[2] if len(sys.argv) > 2 else None, tuple(sys.argv[3:]) or None)
