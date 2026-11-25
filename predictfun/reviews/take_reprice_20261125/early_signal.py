"""What earlier signal do the takers who beat our REPRICE takes use? (USER 2026-11-25: "You mean other takers have earlier signals, then check
how do we find the earlier signals"). Diagnostic on four consumed days (20261023, 20261031..1102: every venue present); no strategy change.

Events (fill_rate_takes.parquet; REPRICE-8-172 primary, REPRICE-16-172 secondary): our sent takes on a large Binance futures move; t_on = the
move's first Binance futures mid tick in the take's direction d (event time E). TAKEN = other takers consumed the aimed quote before we
arrived; their latest possible order time t_ord = their first trade (Predict book stamp) - 150 ms. NOT TAKEN = we filled or a maker pulled it.
Clocks: Predict book stamps vs Binance E are aligned to ~10 ms (Tokyo receipt - stamp >= 13 ms; makers' unheld pulls follow t_on).
Signals over the 400 ms ending at T (T = t_ord for TAKEN, t_on for both groups), each venue on its own exchange clock:
- trade prices on nine venues (lead_venues.load_venue): the largest move in direction d from the last trade at or before T - 400 (bps);
- Binance futures aggressive flow: taker volume in direction d minus against (BTC; raw trade stream, aggressor flag, trade time);
- Binance futures best-level depletion: the share of the best opposite size (the ask for d = +1) at T - 400 gone by T at that price (1 when the
  price level itself was taken out), from the raw bookTicker (E).
Null (declared before the result): the same statistic at T shifted by +-5, 10, .., 30 s (12 shifts per event: >= 18,000 draws per group);
thresholds = the null's 90th / 95th percentiles per group, exceedance read as >= (depletion saturates at 1); lift = exceedance / null exceedance. A signal the takers use must exceed before t_ord
far more often than its null, and more often in TAKEN than in NOT TAKEN events at t_on.
Controls: positive -- Binance futures trade prices in the 400 ms ending 50 ms after our trigger must exceed their null q95 for >= 80 % of
events (the move is visible to the machinery); the null's own exceedance must be ~10 % / 5 % by construction.
Output (write-once): early_signal.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh early_signal -- python3 predictfun/reviews/take_reprice_20261125/early_signal.py
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
sys.path.insert(0, str(H.parent / 'take_exit_venue_20261125'))
import reprice_replay as RP  # noqa: E402
import lead_venues as LV  # noqa: E402

S, OD, MM, TR, C, F = RP.S, RP.OD, RP.MM, RP.TR, RP.C, RP.F
RAW = LV.DATA / 'mm_hf/raw'
WIN, MARGIN = 400., 5000.
SHIFTS = np.array([-30, -25, -20, -15, -10, -5, 5, 10, 15, 20, 25, 30], float) * 1000.
CFGS = ('REPRICE-8-172', 'REPRICE-16-172')
SIGNALS = tuple(f'px_{v}' for v in LV.VENUES) + ('flow_binance_fut', 'depl_binance_fut')


def excursion(t, p, d, T):
    """Largest move in direction d (bps) of the trades in (T - WIN, T] from the last trade at or before T - WIN; first time it reaches thr."""
    j0, j1 = np.searchsorted(t, T - WIN, 'right') - 1, np.searchsorted(t, T, 'right')
    out = np.full(len(T), np.nan)
    for i in range(len(T)):
        if j0[i] < 0:
            continue
        seg = p[j0[i] + 1: j1[i]]
        out[i] = max(0., float((d[i] * 1e4 * np.log(seg / p[j0[i]])).max())) if len(seg) else 0.
    return out


def first_cross(t, p, d, T, thr):
    j0, j1 = np.searchsorted(t, T - WIN, 'right') - 1, np.searchsorted(t, T, 'right')
    out = np.full(len(T), np.nan)
    for i in range(len(T)):
        if j0[i] < 0 or not np.isfinite(thr):
            continue
        seg = d[i] * 1e4 * np.log(p[j0[i] + 1: j1[i]] / p[j0[i]])
        k = np.flatnonzero(seg >= thr)
        if len(k):
            out[i] = T[i] - t[j0[i] + 1 + k[0]]
    return out


def flow(tt, F_, d, T):
    return d * (F_[np.searchsorted(tt, T, 'right')] - F_[np.searchsorted(tt, T - WIN, 'right')])


def depletion(bt, d, T):
    E, bid, bq, ask, aq = bt
    j0, j1 = np.searchsorted(E, T - WIN, 'right') - 1, np.searchsorted(E, T, 'right')
    out = np.full(len(T), np.nan)
    for i in range(len(T)):
        if j0[i] < 0 or E[j0[i]] < T[i] - WIN - MARGIN:
            continue
        if d[i] == 1:
            P0, Q0, pr, qq = ask[j0[i]], aq[j0[i]], ask[j0[i] + 1: j1[i]], aq[j0[i] + 1: j1[i]]
            gone = (pr > P0).any()
        else:
            P0, Q0, pr, qq = bid[j0[i]], bq[j0[i]], bid[j0[i] + 1: j1[i]], bq[j0[i] + 1: j1[i]]
            gone = (pr < P0).any()
        if gone:
            out[i] = 1.
        else:
            at = qq[pr == P0]
            out[i] = float(max(0., 1. - at.min() / Q0)) if len(at) and Q0 > 0 else 0.
    return out


def needed_mask(E, Ts):
    k = np.searchsorted(Ts, E, 'left')
    ok = k < len(Ts)
    keep = np.zeros(len(E), bool)
    keep[ok] = E[ok] >= Ts[k[ok]] - WIN - MARGIN
    return keep


def load_binance(day, Ts):
    tr = []
    for h in range(24):
        fn = RAW / f'trade/BTCUSDT/{day}_{h:02d}.csv.gz'
        if fn.exists():
            tr.append(pd.read_csv(fn, header=None, usecols=[2, 5, 6], names=['t', 'q', 'm']))
    tr = pd.concat(tr).sort_values('t', kind='stable')
    tt = tr.t.to_numpy(float); sg = np.where(tr.m.to_numpy() == 0, 1., -1.)
    F_ = np.r_[0., np.cumsum(tr.q.to_numpy(float) * sg)]
    parts = []
    for h in range(24):
        fn = RAW / f'bookTicker/BTCUSDT/{day}_{h:02d}.csv.gz'
        if not fn.exists():
            continue
        b = pd.read_csv(fn, header=None, usecols=[1, 4, 5, 6, 7], names=['E', 'bid', 'bq', 'ask', 'aq'])
        b = b[needed_mask(b.E.to_numpy(float), Ts)]
        parts.append(b)
    b = pd.concat(parts).sort_values('E', kind='stable')
    bt = tuple(b[c].to_numpy(float) for c in ('E', 'bid', 'bq', 'ask', 'aq'))
    return (tt, F_), bt


def stats_at(day_data, d, T):
    px, (tt, F_), bt = day_data
    out = {f'px_{v}': excursion(px[v][0], px[v][1], d, T) for v in LV.VENUES}
    out['flow_binance_fut'] = flow(tt, F_, d, T)
    out['depl_binance_fut'] = depletion(bt, d, T)
    return out


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    fn = H / 'early_signal.json'
    C.require(not fn.exists(), 'early_signal.json exists (write-once)')
    T = pd.read_parquet(RP.OUT / 'fill_rate_takes.parquet')
    T = T[T.config.isin(CFGS) & T.day.isin(LV.DAYS) & T.cls.isin(['taken', 'filled', 'pulled']) & T.bn_lag.notna()].copy()
    T['grp'] = np.where(T.cls == 'taken', 'taken', 'not_taken')
    T = T[(T.grp == 'not_taken') | T.gap_dec.notna()].copy()
    res = collections.defaultdict(dict)
    for day, G in T.groupby('day'):
        reg = MM.registry(day).set_index('mid')
        t0 = G.mid.map(lambda m: float(reg.loc[m, 'ws']) * 1000.).to_numpy()
        ga = t0 + G.g.to_numpy() - S.FEED - S.DECIDE
        t_on = ga - G.bn_lag.to_numpy()
        t_ord = np.where(G.grp == 'taken', t0 + G.g.to_numpy() + G.gap_dec.fillna(0.).to_numpy() - S.HOLD, np.nan)
        d = G.d.to_numpy().astype(float)
        bk, bm = OD.seg_arrays(OD.SEG_OF[day]); bk = bk.astype(float); bm = bm.astype(float)
        jb, ja = np.searchsorted(bk, t_on, 'left') - 1, np.searchsorted(bk, ga, 'right') - 1
        move = d * 1e4 * np.log(bm[ja] / bm[np.maximum(jb, 0)])
        ends = dict(on=t_on, ord=t_ord, ctl=ga + 50.)
        allT = np.concatenate([e[np.isfinite(e)][:, None] + np.r_[0., SHIFTS][None, :] for e in ends.values()]).ravel()
        Ts = np.unique(allT)
        px = {v: LV.load_venue(v, day) for v in LV.VENUES}
        fl, bt = load_binance(day, Ts)
        dd = (px, fl, bt)
        C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
        for ek, e in ends.items():
            ok = np.isfinite(e)
            act = stats_at(dd, d[ok], e[ok])
            plc = [stats_at(dd, d[ok], e[ok] + s) for s in SHIFTS]
            res[day][ek] = dict(idx=G.index.to_numpy()[ok], act=act, plc={k: np.concatenate([p_[k] for p_ in plc]) for k in SIGNALS})
        res[day]['meta'] = dict(idx=G.index.to_numpy(), move=move, t_on=t_on, t_ord=t_ord, d=d)
        print(day, len(G), flush=True)
    rows = []
    for day, r in res.items():
        m = r['meta']
        base = pd.DataFrame(dict(idx=m['idx'], move=m['move'], t_on=m['t_on'], t_ord=m['t_ord'], d=m['d'], day=day)).set_index('idx')
        for ek in ('on', 'ord', 'ctl'):
            a = r[ek]
            for k in SIGNALS:
                base.loc[a['idx'], f'{ek}:{k}'] = a['act'][k]
        rows.append(base)
    X = T.join(pd.concat(rows), rsuffix='_ev')
    out = dict(asof='2026-11-25', days=list(LV.DAYS), window_ms=WIN, shifts_s=(SHIFTS / 1000.).tolist(), configs={})
    q = lambda s, qs=(.25, .5, .75): [float(v) for v in np.nanquantile(s, qs)] if np.isfinite(s).any() else None
    for cfg in CFGS:
        Xc = X[X.config == cfg]; o = dict(events={g: int((Xc.grp == g).sum()) for g in ('taken', 'not_taken')},
                                          move_bps=q(Xc.move.to_numpy(float)), order_before_onset_ms=q((Xc.t_on - Xc.t_ord).dropna().to_numpy(float)), signals={})
        union = {}
        for k in SIGNALS:
            s = {}
            for ek, grp in (('ord', 'taken'), ('on', 'taken'), ('on', 'not_taken'), ('ctl', 'all')):
                sel = Xc if grp == 'all' else Xc[Xc.grp == grp]
                cols = {dy: np.isin(res[dy][ek]['idx'], sel.index.to_numpy()) for dy in res if ek in res[dy]}
                raw_null = np.concatenate([res[dy][ek]['plc'][k].reshape(len(SHIFTS), -1)[:, cols[dy]].ravel() for dy in cols])
                null = raw_null[np.isfinite(raw_null)]
                v = sel[f'{ek}:{k}'].to_numpy(float); v = v[np.isfinite(v)]
                if not len(null) or not len(v):
                    continue
                q90, q95 = np.quantile(null, .9), np.quantile(null, .95)
                if q95 <= 0:
                    continue
                s[f'{ek}|{grp}'] = dict(n=len(v), null_n=len(null), q90=float(q90), q95=float(q95), med=float(np.median(v)), null_med=float(np.median(null)),
                                        exc90=float((v >= q90).mean()), null_exc90=float((null >= q90).mean()), exc95=float((v >= q95).mean()), null_exc95=float((null >= q95).mean()),
                                        lift95=float((v >= q95).mean() / max((null >= q95).mean(), 1e-9)))
                if ek == 'ord' and grp == 'taken':
                    union[k] = (sel[f'{ek}:{k}'].to_numpy(float) >= q95, raw_null >= q95)
            o['signals'][k] = s
        if union:
            act_any = np.any([e for e, _ in union.values()], axis=0); null_any = np.any([n for _, n in union.values()], axis=0)
            o['any_signal_q95'] = dict(taken_before_order=float(act_any.mean()), null=float(null_any.mean()), signals=sorted(union))
        out['configs'][cfg] = o
    ctl = out['configs']['REPRICE-8-172']['signals']['px_binance_fut'].get('ctl|all')
    out['positive_control_pass'] = bool(ctl is not None and ctl['exc95'] >= .8)
    out['null_calibrated_continuous'] = all(abs(v['null_exc95'] - .05) < .02 and abs(v['null_exc90'] - .10) < .03 for o in out['configs'].values()
                                            for k, s in o['signals'].items() if not k.startswith('depl') for v in s.values() if v['null_n'] > 1000)
    TR.dump(fn, out)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
