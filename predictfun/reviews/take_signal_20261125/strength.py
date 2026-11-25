"""TAKE-SIGNAL v1 amendment 2: the signal-strength curve (amendment_2_strength.md). Every BTC mid change with |bn_r250| >= 0.05 bps in a
market window, valued with the parity-checked take valuation (controls.value, which reproduced takes.parquet exactly), plus causal
strength features: bn_r1000 / bn_r3000, sigma300, the binary's sensitivity and the implied Predict move in cents.
Output (write-once): data/.../take_signal_20261125/strength_events.parquet + strength.json + tables/s_*.csv.
Usage: run_light.sh ts_strength -- python3 predictfun/reviews/take_signal_20261125/strength.py
"""
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtri

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import controls as CT  # noqa: E402
import take_eval as TE  # noqa: E402

OUT = TE.OUT
T = H / 'tables'
FLOOR = 0.05
Q = 50
CLK = TE.CLOCKS['tokyo']
FINE = [0.05, 0.1, 0.2, TE.UP_EDGE, 0.5, 0.75, 1, 1.5, 2, 3, np.inf]
FINE_L = ['0.05-0.1', '0.1-0.2', '0.2-0.29', '0.29-0.5', '0.5-0.75', '0.75-1', '1-1.5', '1.5-2', '2-3', '>=3']
IMB = [0, 0.5, 1, 1.5, 2, 3, 5, 10, np.inf]
IML = ['0-0.5', '0.5-1', '1-1.5', '1.5-2', '2-3', '3-5', '5-10', '>=10']
EDB = [-np.inf, -1, 0, 1, 2, 5, np.inf]
EDL = ['<-1', '-1-0', '0-1', '1-2', '2-5', '>=5']
THETAS = [0.05, 0.1, 0.15, 0.2, 0.25, TE.UP_EDGE, 0.35, TE.DOWN_EDGE, 0.5, 0.75, 1, 1.5, 2, 3]
VALS = ['mark_1s', 'mark_3s', 'mark_10s', 'rt_3s', 'end', 'res']
HALF = {'20261024': 'Oct24-25', '20261025': 'Oct24-25', '20261026': 'Oct26-27', '20261027': 'Oct26-27'}


def sigma_series(bk, bm):
    """Whole-second BTC log returns (bps) on the decision clock (E + 5) and their cumulative squares."""
    t = bk + CLK['b_in']
    s0, s1 = t[0] // 1000 + 1, t[-1] // 1000
    secs = np.arange(s0, s1 + 1)
    j = np.searchsorted(t, secs * 1000, 'right') - 1
    m = bm[j]
    r = np.r_[np.nan, 1e4 * np.diff(np.log(m))]
    r2 = np.where(np.isfinite(r), r * r, 0.0)
    return secs, np.cumsum(r2)


def sigma_at(secs, cs, tt):
    k = tt // 1000                                   # the last whole second <= t
    i = np.searchsorted(secs, k, 'right') - 1
    lo = i - 300
    ok = lo >= 1
    v = np.where(ok, (cs[np.maximum(i, 0)] - cs[np.maximum(lo, 0)]) / 300.0, np.nan)
    return np.sqrt(v)


def agg(d, keys):
    d = d.assign(sh=np.minimum(Q, d.avail_sh))

    def f(x):
        o = dict(events=len(x), filled=int((x.status == 'filled').sum()), markets=int(x[x.status == 'filled'].mid.nunique()),
                 fill_rate=float((x.status == 'filled').mean()), im_c=float(x.im_c.mean()), cost_c=float(x.cost_c.mean()))
        for v in VALS:
            ok = x[v].notna() & (x.sh > 0)
            o[f'{v}_c'] = float(100 * (x[v][ok] * x.sh[ok]).sum() / x.sh[ok].sum()) if ok.any() else np.nan
        ok = x.mark_3s.notna() & (x.sh > 0)
        o['usd_3s'] = float((x.mark_3s[ok] * x.sh[ok]).sum())
        return pd.Series(o)
    return d.groupby(keys, observed=True).apply(f, include_groups=False).reset_index()


def triggers(ev, theta_up, theta_dn):
    """OFF->ON with the 2 s cooldown per market x direction for the rule |bn_r250| >= theta (direction-specific thresholds)."""
    keep = np.zeros(len(ev), bool)
    q = ((ev.dir == 1) & (ev.r250 >= theta_up - 1e-12)) | ((ev.dir == -1) & (-ev.r250 >= theta_dn - 1e-12))
    sub = ev[q]
    for (day, mid, d), g in sub.groupby(['day', 'mid', 'dir']):
        tt = g.t.to_numpy()
        gap = np.r_[np.inf, np.diff(tt)]
        keep[g.index[gap >= TE.COOLDOWN_MS]] = True
    return keep


def main():
    t0 = time.process_time()
    resource.setrlimit(resource.RLIMIT_CPU, (1200, 1201))
    out = OUT / 'strength_events.parquet'
    assert not out.exists(), 'write-once'
    T.mkdir(exist_ok=True)
    bn = pd.read_parquet(OUT / 'bn_changes.parquet')
    bk, bm = bn.k_ms.to_numpy(np.int64), bn.mid.to_numpy(float)
    secs, cs = sigma_series(bk, bm)
    rows = []
    for day in TE.DAYS:
        mk = pd.read_parquet(TE.V / f'markets_{day}.parquet', columns=['mid', 'ws', 'we', 'outcome', 'feeRateBps', 'status'])
        mindex = {int(m): i for i, m in enumerate(sorted(mk.mid))}
        Tb = TE.load_tob(day, mindex)
        for r in mk[~mk.status.astype(str).str.contains('boundary_censored')].itertuples():
            ws, we = int(r.ws) * 1000, int(r.we) * 1000
            end = we - 2000
            ti, r250, r1000 = TE.events_for(bk, bm, CLK['b_in'], ws, end)
            q = np.abs(r250) >= FLOOR
            if not q.any():
                continue
            ti, r250, r1000 = ti[q], r250[q], r1000[q]
            t3 = bk + CLK['b_in']
            j3 = np.searchsorted(t3, ti - 3000, 'right') - 1
            jn = np.searchsorted(t3, ti, 'right') - 1
            r3000 = np.where(j3 >= 0, 1e4 * np.log(bm[jn] / bm[np.maximum(j3, 0)]), np.nan)
            d = np.where(r250 > 0, 1, -1).astype(np.int64)
            row = type('R', (), dict(ws=r.ws, we=r.we, outcome=r.outcome, feeRateBps=r.feeRateBps))
            v = CT.value(Tb, mindex[int(r.mid)], row, ti, d, CLK)
            midx = np.full(len(ti), mindex[int(r.mid)], np.int64)
            xk = ti - CLK['p_in']
            ik = TE.frame_last(Tb, midx, xk)
            kmid, kok = TE.mid_ok(Tb, ik, xk)
            kpx, _ = TE.side_px(Tb, ik, d == 1)
            jk = np.where(ik >= 0, ik, 0)
            spr = np.where(kok, (Tb['ask'][jk] - Tb['bid'][jk]) * 100, np.nan)
            sig = sigma_at(secs, cs, ti)
            tau = (we - ti) / 1000.0
            z = ndtri(np.clip(kmid, 0.01, 0.99))
            sens = 100 * np.exp(-z * z / 2) / np.sqrt(2 * np.pi) / (sig * np.sqrt(tau))
            fee = r.feeRateBps / 1e4 * np.minimum(kpx, 1 - kpx)
            cost = d * (kpx - kmid) * 100 + fee * 100
            im = sens * np.abs(r250)
            v = v.assign(day=day, mid=int(r.mid), t_rel_s=(ti - ws) / 1000, r250=r250, r1000=r1000, r3000=r3000, sigma300=sig, tau_s=tau,
                         known_mid=kmid, limit=kpx, spread_c=spr, sens_c_per_bps=sens, im_c=im, cost_c=cost, edge_c=im - cost)
            rows.append(v)
        del Tb
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 < 2500, 'VmHWM guard'
    ev = pd.concat(rows, ignore_index=True)
    ev['fine'] = pd.cut(ev.r250.abs(), FINE, right=False, labels=FINE_L).astype(str)
    ev['imb'] = pd.cut(ev.im_c, IMB, right=False, labels=IML).astype(str)
    ev['edb'] = pd.cut(ev.edge_c, EDB, right=False, labels=EDL).astype(str)
    ev['half'] = ev.day.map(HALF)
    same = np.sign(ev.r3000) == ev.dir
    ev['trend'] = np.where(same & (ev.r3000.abs() >= ev.r250.abs()), 'CONTINUATION', np.where(np.sign(ev.r3000) == -ev.dir, 'REVERSAL', 'OTHER'))
    ev['coarse'] = np.where(ev.r250.abs() < TE.UP_EDGE, '<0.29', np.where(ev.r250.abs() < 1, '0.29-1', '>=1'))
    # parity control: the guard's own rule must reproduce takes.parquet (tokyo) row for row
    g = ev[triggers(ev, TE.UP_EDGE, TE.DOWN_EDGE)].sort_values(['day', 'mid', 't', 'dir']).reset_index(drop=True)
    tk = pd.read_parquet(OUT / 'takes.parquet')
    tk = tk[tk.clock == 'tokyo'].sort_values(['day', 'mid', 't', 'dir']).reset_index(drop=True)
    par = len(g) == len(tk) and (g.t.to_numpy() == tk.t.to_numpy()).all() and (g.status.to_numpy() == tk.status.to_numpy()).all()
    for col in ['avail_sh'] + VALS:
        a, b = g[col].to_numpy(float), tk[col].to_numpy(float)
        par = par and bool(((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) < 1e-12)).all())
    assert par, 'parity with takes.parquet failed'
    rec = dict(events=len(ev), parity_rows=len(g), parity_ok=bool(par))
    tabs = {}
    for name, keys in (('fine', ['fine']), ('imb', ['imb']), ('edb', ['edb'])):
        a = agg(ev, keys)
        b = agg(ev, ['half'] + keys)
        a.to_csv(T / f's_{name}.csv', index=False)
        b.to_csv(T / f's_{name}_half.csv', index=False)
        tabs[name], tabs[name + '_half'] = a, b
    tr = agg(ev, ['half', 'coarse', 'trend'])
    tr.to_csv(T / 's_trend_half.csv', index=False)
    agg(ev, ['coarse', 'trend']).to_csv(T / 's_trend.csv', index=False)
    agg(ev[(ev.known_mid > 0.1) & (ev.known_mid < 0.9)], ['fine']).to_csv(T / 's_fine_near.csv', index=False)
    agg(ev.assign(tw=pd.cut(ev.t_rel_s, [0, 60, 180, 270, 298], right=False, labels=['0-60', '60-180', '180-270', '270-298']).astype(str)),
        ['tw', 'imb']).to_csv(T / 's_imb_time.csv', index=False)
    curve = []
    for th in THETAS:
        k = triggers(ev, th, th)
        for half in ('all', 'Oct24-25', 'Oct26-27'):
            x = ev[k & ((ev.half == half) if half != 'all' else True)]
            a = agg(x.assign(_k=1), ['_k']).iloc[0]
            curve.append(dict(theta_bps=th, half=half, triggers=int(a.events), fill_rate=a.fill_rate, mark_1s_c=a.mark_1s_c, mark_3s_c=a.mark_3s_c,
                              mark_10s_c=a.mark_10s_c, rt_3s_c=a.rt_3s_c, usd_3s_q50=a.usd_3s, im_c=a.im_c, cost_c=a.cost_c))
    curve = pd.DataFrame(curve)
    curve.to_csv(T / 's_threshold_curve.csv', index=False)
    # readings
    fh = tabs['fine_half'].pivot(index='fine', columns='half', values='mark_3s_c').reindex(FINE_L)
    both = (fh > 0).all(axis=1).to_numpy()
    be = None
    for i in range(len(FINE_L)):
        if both[i:].all():
            be = FINE_L[i]
            break
    rec['a_break_even_bin'] = be
    rec['a_bins_positive_both_halves'] = dict(zip(FINE_L, map(bool, both)))
    fp = tabs['fine'].set_index('fine').reindex(FINE_L).mark_3s_c.to_numpy()
    rec['b_monotone_fine_mark_3s'] = bool(np.all(np.diff(fp) > 0))
    rec['b_fine_mark_3s_c'] = dict(zip(FINE_L, map(float, fp)))
    eh = tabs['edb_half'].pivot(index='edb', columns='half', values='mark_3s_c').reindex(EDL)
    assert list(eh.columns) == ['Oct24-25', 'Oct26-27'], eh.columns
    rec['c_calibration'] = dict(positive_when_edge_ge_0=bool((eh.loc[['0-1', '1-2', '2-5', '>=5']] > 0).all().all()),
                                nonpositive_when_edge_lt_minus1=bool((eh.loc[['<-1']] <= 0).all().all()),
                                by_bin={k: {h: float(eh.loc[k, h]) for h in eh.columns} for k in EDL})
    trp = tr.pivot_table(index=['half', 'coarse'], columns='trend', values='mark_3s_c')
    rec['d_continuation_beats_reversal_all'] = bool((trp['CONTINUATION'] > trp['REVERSAL']).all())
    rec['d_table'] = {f'{a}|{b}': {c: float(trp.loc[(a, b), c]) for c in trp.columns} for a, b in trp.index}
    rec['cpu_s'] = round(time.process_time() - t0, 2)
    json.dumps(rec, default=float)                                  # the receipt must serialise before the output is written
    ev.to_parquet(out, index=False)
    (H / 'strength.json').write_text(json.dumps(rec, indent=1, default=float) + '\n')
    print(json.dumps(rec, indent=1, default=float))


if __name__ == '__main__':
    main()
