"""Why do we fill only 2-3 % of the REPRICE takes: late on the signal, or slow to the venue? (USER 2026-11-25: "Can you check why we only
fill 2-3%, is it we miss some good signals or the latency issue"). Diagnostic on why_markout_takes.parquet (every take we sent, 408
complete markets, five consumed days) and the Binance futures change points the triggers used (event time).

1. When was each missed quote lost, relative to our decision g? An order decided at g reaches the book at g + network + 150 ms. Lost = the
   venue time the aimed level vanished from the book before our arrival (who_took.follow). Bins: before g; g .. g + 150 (no order decided
   at g arrives in time at any network speed); g + 150 .. g + 172 (inside our 22 ms network: the latency share); at our arrival (the level
   was still shown, the queue / strict rules gave us nothing: reasons empty, strict_cancelled).
2. How late is our trigger on its own data? In the take's direction over the 1 s before g: how long the 250 ms dFV had been >= 2 / 4 c in
   the run that ends in our trigger, and when the mid first moved inside the 250 ms window the trigger measured.
3. Did the other takers decide before Binance moved? At the latest instant they can have decided (their first trade - 150 ms): Binance mid
   change points in the 250 ms before, and the time to the next mid move in the take's direction; placebo = the same 2 s earlier.
4. Clock calibration: makers' cancels are not held, so the time the pulled quotes vanished after the Binance onset of the move (the first mid
   tick in the take's direction inside the trigger's 250 ms window) bounds the offset between the Binance event-time clock and Predict's
   venue clock; the other takers' orders (first trade - 150 ms) are timed against the same onset.
Controls: t0 = the registry window start reproduces d x dFV >= D at every REPRICE-D decision (falsifier: t0 shifted by 300 ms breaks it).
Output (write-once): fill_rate.json; per-take table OUT/fill_rate_takes.parquet.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh fill_rate -- python3 predictfun/reviews/take_reprice_20261125/fill_rate.py
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

S, OD, MM, TR, C = RP.S, RP.OD, RP.MM, RP.TR, RP.C
CFG = {'REPRICE-4-172': 4., 'REPRICE-8-172': 8., 'REPRICE-16-172': 16., 'STREAM-172': None}
NET, PLACEBO, LOOK = 22., 2000., 1000.


def dfv_vec(bk, bm, cs, ws, t):
    """reprice_replay.reprice_triggers' dFV (cents, up > 0) at the times t (absolute ms); nan where the trigger skips."""
    t = np.atleast_1d(np.asarray(t, float))
    jw = int(np.searchsorted(bk, ws, 'right') - 1)
    j0, j1 = np.searchsorted(bk, t, 'right') - 1, np.searchsorted(bk, t - RP.LAG, 'right') - 1
    js = np.searchsorted(bk, t - RP.LAG - RP.SIG_WIN, 'right') - 1
    ok = (j0 >= 0) & (j1 >= 0) & (jw >= 0)
    a0, a1, w = np.maximum(j0, 0), np.maximum(j1, 0), max(jw, 0)
    x, x250 = 1e4 * np.log(bm[a0] / bm[w]), 1e4 * np.log(bm[a1] / bm[w])
    span = (t - RP.LAG - np.maximum(t - RP.LAG - RP.SIG_WIN, bk[0])) / 1000.
    sig = np.where(ok & (span > 1.), 1e4 * np.sqrt(np.maximum(cs[a1] - cs[np.maximum(js, 0)], 0.) / np.maximum(span, 1.)), np.nan)
    tau = (ws + 300000. - t) / 1000.
    good = ok & np.isfinite(sig) & (sig > 0) & (tau >= .5)
    s = np.where(good, sig * np.sqrt(np.maximum(tau, .5)), 1.)
    return np.where(good, 100. * (RP.Phi(x / s) - RP.Phi(x250 / s)), np.nan)


def loss_bin(r):
    if r.cls == 'filled':
        return 'filled'
    if r.cls == 'other':
        return 'other'
    gm = r.gone_ms
    if gm is not None and np.isfinite(gm):
        return 'before_decision' if gm < 0 else 'speed_bump' if gm < S.HOLD else 'network' if gm < S.HOLD + NET else 'at_arrival'
    return 'at_arrival' if r.reason in ('empty', 'strict_cancelled') else 'unknown'


def onset(bk, bm, cs, ws, d, ga):
    k = bk[(bk >= ga - LOOK - RP.LAG) & (bk <= ga)].astype(float)
    bp = np.unique(np.r_[k, k + RP.LAG, ga]); bp = bp[(bp >= ga - LOOK) & (bp <= ga)]
    v = d * dfv_vec(bk, bm, cs, ws, bp)
    out = {}
    for thr in (2., 4.):
        if not v[-1] >= thr:
            out[f'on{int(thr)}'] = np.nan; continue
        below = np.flatnonzero(~(v >= thr))
        out[f'on{int(thr)}'] = ga - (bp[below[-1] + 1] if len(below) else bp[0])          # = LOOK when censored
    jr = int(np.searchsorted(bk, ga - RP.LAG, 'right') - 1)
    sel = np.flatnonzero((bk > ga - RP.LAG) & (bk <= ga))
    mv = sel[d * np.log(bm[sel] / bm[jr]) > 0] if jr >= 0 else sel[:0]
    out['bn_lag'] = ga - float(bk[mv[0]]) if len(mv) else np.nan
    out['bn_ticks'] = len(sel)
    tot = d * np.log(bm[sel[-1]] / bm[jr]) if len(sel) and jr >= 0 else np.nan
    out['last_tick_share'] = d * np.log(bm[sel[-1]] / bm[sel[-2] if len(sel) > 1 else jr]) / tot if (len(sel) and tot and np.isfinite(tot)) else np.nan
    return out


def before_move(bk, bm, d, t):
    j = int(np.searchsorted(bk, t, 'right') - 1)
    n = int(np.searchsorted(bk, t, 'right') - np.searchsorted(bk, t - RP.LAG, 'right'))
    nx = np.flatnonzero((bk > t) & (bk <= t + LOOK))
    nx = nx[d * np.log(bm[nx] / bm[j]) > 0] if j >= 0 else nx[:0]
    return n, (float(bk[nx[0]]) - t) if len(nx) else np.nan


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (600, 601))
    fn = H / 'fill_rate.json'
    C.require(not fn.exists(), 'fill_rate.json exists (write-once)')
    T = pd.read_parquet(RP.OUT / 'why_markout_takes.parquet')
    T['bin'] = [loss_bin(r) for r in T.itertuples()]
    rows, ctl = [], collections.Counter()
    for day, G in T.groupby('day'):
        bk, bm = OD.seg_arrays(OD.SEG_OF[day]); bk = bk.astype(np.int64); bm = bm.astype(float)
        cs = np.cumsum(np.r_[0., np.diff(np.log(bm))] ** 2); bkf = bk.astype(float)
        reg = MM.registry(day).set_index('mid')
        for r in G.itertuples():
            t0 = float(reg.loc[r.mid, 'ws']) * 1000.; ws = t0
            ga = t0 + r.g - S.FEED - S.DECIDE
            D = CFG[r.config]
            o = dict(idx=r.Index)
            if D is not None:
                ctl['t0_checked'] += 1
                ctl['t0_below_D'] += int(not (r.d * dfv_vec(bkf, bm, cs, ws, ga)[0] >= D - 1e-6))
                ctl['falsifier_shift_below_D'] += int(not (r.d * dfv_vec(bkf, bm, cs, ws, ga + 300.)[0] >= D - 1e-6))
                o.update(onset(bkf, bm, cs, ws, r.d, ga))
            if r.cls == 'taken' and np.isfinite(r.gap_dec):
                lat = t0 + r.g + r.gap_dec - S.HOLD
                for nm, t in (('act', lat), ('plc', lat - PLACEBO)):
                    n, dl = before_move(bkf, bm, r.d, t)
                    o[f'{nm}_ticks250'], o[f'{nm}_next_move'] = n, dl
            rows.append(o)
        print(day, len(rows), flush=True)
    T = T.join(pd.DataFrame(rows).set_index('idx'))
    T['onset_rel_g'] = -(S.FEED + S.DECIDE) - T.bn_lag
    T['gone_from_onset'] = T.gone_ms - T.onset_rel_g
    T['trade_from_onset'] = T.gap_dec - T.onset_rel_g
    T.to_parquet(RP.OUT / 'fill_rate_takes.parquet', index=False)

    def q3(s, qs=(.25, .5, .75)):
        s = pd.to_numeric(s, errors='coerce').dropna(); return [float(x) for x in s.quantile(list(qs))] if len(s) else None

    def mean(s):
        s = pd.to_numeric(s, errors='coerce').dropna(); return float(s.mean()) if len(s) else None
    out = dict(asof='2026-11-25', takes=len(T), controls=dict(ctl), configs={})
    out['controls_pass'] = ctl['t0_below_D'] == 0 and ctl['t0_checked'] > 0
    out['falsifier_fires'] = ctl['falsifier_shift_below_D'] > 0
    order = ['filled', 'before_decision', 'speed_bump', 'network', 'at_arrival', 'unknown', 'other']
    for name in CFG:
        X = T[T.config == name]; n = len(X)
        b = dict(sent=n, bins={k: dict(n=int((X.bin == k).sum()), share=float((X.bin == k).mean()), wb1=mean(X[X.bin == k].wb1),
                                      taken=int(((X.bin == k) & (X.cls == 'taken')).sum()), pulled=int(((X.bin == k) & (X.cls == 'pulled')).sum())) for k in order})
        if CFG[name] is not None:
            b['trigger'] = dict(on2_ms=q3(X.on2), on4_ms=q3(X.on4), on2_ge_150=float((X.on2 >= S.HOLD).mean()), on4_ge_150=float((X.on4 >= S.HOLD).mean()),
                                bn_lag_ms=q3(X.bn_lag), bn_ticks=q3(X.bn_ticks), last_tick_share=q3(X.last_tick_share))
            P = X[(X.cls == 'pulled') & X.gone_from_onset.notna()]; K2 = X[(X.cls == 'taken') & X.trade_from_onset.notna()]
            b['clock'] = dict(pulled_n=len(P), pulled_gone_from_onset_ms=q3(P.gone_from_onset, (.05, .1, .25, .5, .75)),
                              pulled_before_onset=float((P.gone_from_onset < 0).mean()) if len(P) else None,
                              taken_n=len(K2), taken_order_from_onset_ms=q3(K2.trade_from_onset - S.HOLD, (.05, .1, .25, .5, .75)),
                              taken_order_before_onset=float((K2.trade_from_onset - S.HOLD < 0).mean()) if len(K2) else None)
        K = X[(X.cls == 'taken') & X.gap_dec.notna()]
        if len(K):
            b['others'] = dict(n=len(K), no_tick_250_before=float((K.act_ticks250 == 0).mean()), no_tick_250_placebo=float((K.plc_ticks250 == 0).mean()),
                               next_move_ms=q3(K.act_next_move), next_move_placebo_ms=q3(K.plc_next_move),
                               moved_within_150=float((K.act_next_move <= S.HOLD).mean()), moved_within_150_placebo=float((K.plc_next_move <= S.HOLD).mean()))
        out['configs'][name] = b
    TR.dump(fn, out)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
