"""B12b POST HOC (2026-11-11, written after reading b12b.txt; descriptive, no reading). Runs the committed
pf_b12b_reason build unchanged (c882da77) and adds:
 PH1 the top 5 against O on Part A's legs (at the best, guard ON / OFF at lag 50, not swept): does their within-guard
     gap line up with our guard's own value? Cells: V10 = decile of v at anc - 50 ms; V5D3 = quintile of v x tercile of
     its change over the 250 ms before (dv = v(anc - 50) - v(anc - 300); legs without both values excluded, count
     reported); T4G = seconds into the window (0-5, 5-15, 15-30, 30-60) x guard. Edges are quantiles of the T + O legs.
     X = (WITHIN_G - WITHIN_cells) / WITHIN_G on the same legs. Legs whose v at anc - 50 is NaN (B12's lookup reads
     them as OFF; 23 of 72,913 at-best legs, none of the top 5's or ours) are excluded and counted.
 PH2 our at-best first-minute legs (guard OFF and ON at lag 50) against the fill-clock audit's match interval [lower,
     upper] (clock_cases: lower = kernel place, upper = min(first positive fill notice, confirmed cancel)): where the
     research trade time sits (inside, after upper, before lower); the research guard state at upper - 50 ms; the share of
     [lower - 50, upper - 50] with the guard OFF (10 ms steps); our cancel request (core t_cancel) against upper.
 PH3 slow cancels: our OFF legs split by whether their order's cancel took > 500 ms from request to cancel ack;
     for all cancelled live orders of the cohort, the slow share by cancel second and whether a throttle was recorded.
Usage: predictfun/research/run_light.sh b12b_posthoc -- python3 -m predictfun.research.pf_b12b_posthoc <out_dir>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from . import pf_b12_makers as B
from . import pf_b12b_reason as BB


def qedges(x, k):
    return np.unique(np.quantile(x, np.arange(1, k) / k))


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    ok, ctl = BB.controls()
    if not ok:
        sys.exit(f'CONTROL FAILED -> no table: {ctl}')
    X = BB.build(); R = X['R']; G = X['G']; LINE = X['LINE']
    k0 = BB.reconcile(R, X['days'])
    if not k0['ok']:
        sys.exit(f'CONTROL FAILED (K0) -> no table: {k0}')
    v50 = np.full(len(R), np.nan); v300 = np.full(len(R), np.nan)
    for (m, side), ii in R.groupby(['mid', 'side'], sort=False).indices.items():
        if m not in G:
            continue
        tau, vb, va = G[m]; vs = (vb, va)[side]; ta = R.anc.values[ii]
        v50[ii] = B.guard_state(tau, vs, ta - 50.0, LINE)[1]; v300[ii] = B.guard_state(tau, vs, ta - 300.0, LINE)[1]
    R['v50'] = np.where(v50 == -1.0, np.nan, v50); R['dv'] = R.v50.values - np.where(v300 == -1.0, np.nan, v300)
    res = dict(note='POST HOC, descriptive; same build and statuses as b12b.json', K0=k0['ok'])
    # PH1
    V = R[R.val_ok]; A = V[(V.status == 'ok') & (V.depth == 0) & V.g50.isin([B.G_ON, B.G_OFF]) & (V.sweep == 0)]
    D = A[A.g.isin(['top 5', '6-20', 'rest'])].copy(); n_nan = int(np.isnan(D.v50.values).sum())
    D = D[np.isfinite(D.v50.values)]; D['off'] = (D.g50.values == B.G_OFF).astype(np.int64)   # a NaN v reads as OFF in B12's lookup: excluded here
    isT = D.g.values == 'top 5'; sh = D.shares.values; val = D.v1.values
    Wg = BB.decompose_n(D.off.values, isT, sh, val, 2)['WITHIN']
    e10 = qedges(D.v50.values, 10); b10 = BB.bucket(D.v50.values, e10); n10 = len(e10) + 1
    d10 = BB.decompose_n(b10, isT, sh, val, n10)
    ph1 = dict(excluded_nan_v_legs=n_nan, WITHIN_G=Wg, V10=dict(edges=[float(x) for x in e10], D=d10['D'], MIX=d10['MIX'], WITHIN=d10['WITHIN'], X=(Wg - d10['WITHIN']) / Wg,
                                     cells=[dict(wT=float(d10['wT'][i]), wO=float(d10['wO'][i]), vT=float(d10['vT'][i]), vO=float(d10['vO'][i])) for i in range(n10)]))
    fd = np.isfinite(D.dv.values); Dd = D[fd]; tD = isT[fd]
    Wgd = BB.decompose_n(Dd.off.values, tD, Dd.shares.values, Dd.v1.values, 2)['WITHIN']
    e5 = qedges(Dd.v50.values, 5); ed = qedges(Dd.dv.values, 3)
    c53 = (len(ed) + 1) * BB.bucket(Dd.v50.values, e5) + BB.bucket(Dd.dv.values, ed); n53 = (len(e5) + 1) * (len(ed) + 1)
    d53 = BB.decompose_n(c53, tD, Dd.shares.values, Dd.v1.values, n53)
    wd = BB.bucket(Dd.dv.values, ed)
    by_dv = []
    for i in range(len(ed) + 1):
        mk = wd == i; s_ = Dd.shares.values
        by_dv.append(dict(T=B.wmean(s_[mk & tD], Dd.v1.values[mk & tD]), O=B.wmean(s_[mk & ~tD], Dd.v1.values[mk & ~tD]),
                          wT=float(s_[mk & tD].sum() / s_[tD].sum()), wO=float(s_[mk & ~tD].sum() / s_[~tD].sum())))
    ph1['V5D3'] = dict(excluded_legs=int((~fd).sum()), WITHIN_G_subset=Wgd, dv_edges=[float(x) for x in ed], D=d53['D'], MIX=d53['MIX'], WITHIN=d53['WITHIN'],
                       X=(Wgd - d53['WITHIN']) / Wgd, by_dv_tercile=by_dv)
    tw = BB.bucket(D.anc.values - D.ws.values, np.array([5000.0, 15000.0, 30000.0])); c4 = 4 * D.off.values + tw
    d4 = BB.decompose_n(c4, isT, sh, val, 8)
    ph1['T4G'] = dict(D=d4['D'], MIX=d4['MIX'], WITHIN=d4['WITHIN'], X=(Wg - d4['WITHIN']) / Wg,
                      cells=[dict(wT=float(d4['wT'][i]), wO=float(d4['wO'][i]), vT=float(d4['vT'][i]), vO=float(d4['vO'][i])) for i in range(8)])
    res['PH1'] = ph1
    # PH2
    U = R[R.val_ok & (R.status == 'ok') & (R.depth == 0) & (R.g == 'ours') & R.g50.isin([B.G_ON, B.G_OFF])].copy()
    cid, lst = BB.link_orders(U, X['own']); U['cid'] = cid; U['link'] = lst
    cc = X['own'][BB.KEYS + ['lower', 'upper', 'first_positive_notice', 'cancel', 'place', 'accepted', 'Z']]
    U = U.merge(cc, on=BB.KEYS, how='left', suffixes=('', '_cc'))
    if len(U) != (R.val_ok & (R.status == 'ok') & (R.depth == 0) & (R.g == 'ours') & R.g50.isin([B.G_ON, B.G_OFF])).sum():
        sys.exit('REFUSED: an own leg matched two clock_cases rows')
    core = []
    for r in BB.RUNS:
        K = pd.read_parquet(f'{BB.LV}/core_orders_{r}.parquet'); core.append(K[K.t_Accepted.notna() & K.refused.isna() & K.market.isin(list(X['W0']))])
    CI = pd.concat(core, ignore_index=True).set_index('cid')
    o = CI.reindex(U.cid.values); U['t_cancel'] = o.t_cancel.values; U['t_done'] = o.t_CancelDone.values
    pos = []; st_up = []; off_share = []
    for r in U.itertuples():
        pos.append('inside' if r.lower - 1.0 <= r.anc <= r.upper + 1.0 else ('after upper' if r.anc > r.upper + 1.0 else 'before lower'))
        tau, vb, va = G[r.mid]; vs = (vb, va)[r.side]
        st_up.append(int(B.guard_state(tau, vs, np.array([r.upper - 50.0]), LINE)[0][0]))
        grid = np.arange(r.lower - 50.0, r.upper - 50.0 + 1e-9, 10.0)
        s_ = B.guard_state(tau, vs, grid, LINE)[0] if len(grid) else np.array([], np.int8)
        off_share.append(float(np.mean(s_ == B.G_OFF)) if len(s_) else np.nan)
    U['pos'] = pos; U['g_upper'] = st_up; U['off_share'] = off_share
    ph2 = {}
    for name, gv in (('OFF', B.G_OFF), ('ON', B.G_ON)):
        x = U[U.g50 == gv]; s_ = x.shares.values
        ph2[name] = dict(legs=int(len(x)), shares=float(s_.sum()),
                         research_time_position={k: dict(legs=int((x.pos == k).sum()), shares=float(s_[x.pos.values == k].sum())) for k in ('inside', 'after upper', 'before lower')},
                         guard_at_upper_minus_50={lab: dict(legs=int((x.g_upper == v).sum()), shares=float(s_[x.g_upper.values == v].sum()))
                                                  for lab, v in (('ON', B.G_ON), ('OFF', B.G_OFF), ('warm-up', B.G_WARM), ('none', B.G_NONE))},
                         interval_off_share_mean=B.wmean(s_, x.off_share.values),
                         upper_is_cancel=int((np.abs(x.upper.values - x.cancel.values) < 1e-6).sum()),
                         cancel_request_before_upper=int((x.t_cancel.values < x.upper.values).sum()),
                         net1=B.wmean(s_, x.v1.values),
                         legs_detail=[dict(t=float(r.anc - r.ws), shares=float(r.shares), pos=r.pos, interval_ms=float(r.upper - r.lower),
                                           anc_minus_upper=float(r.anc - r.upper), cancel_req_minus_upper=float(r.t_cancel - r.upper),
                                           audit_cancel_minus_req=float(r.cancel - r.t_cancel), done_minus_req=float(r.t_done - r.t_cancel),
                                           notice_minus_req=float(r.first_positive_notice - r.t_cancel), guard_at_upper=int(r.g_upper),
                                           interval_off_share=float(r.off_share), v1=float(r.v1)) for r in x.sort_values('anc').itertuples()])
    res['PH2'] = ph2
    # PH3 slow cancels: our OFF legs whose order's cancel took > 500 ms (request to cancel ack), and all live orders
    x = U[U.g50 == B.G_OFF]; slow = (x.t_done.values - x.t_cancel.values) > 500.0; s_ = x.shares.values; days = X['days']
    ph3 = dict(off_legs={k: dict(legs=int(m_.sum()), shares=float(s_[m_].sum()), windows=int(x.mid[m_].nunique()), net1=B.wmean(s_[m_], x.v1.values[m_]),
                                 usd_day1=float((s_[m_] * x.v1.values[m_]).sum() / 100.0 / days), t_in_window_s=sorted(round(float(v) / 1000.0, 1) for v in (x.anc - x.ws)[m_]))
                         for k, m_ in (('cancel > 500 ms', slow), ('cancel <= 500 ms', ~slow))})
    Ca = pd.concat(core, ignore_index=True); Ca = Ca[Ca.t_cancel.notna() & Ca.t_CancelDone.notna()].copy()
    Ca['ws'] = Ca.market.map(X['W0']).astype(float); Ca['dc'] = Ca.t_CancelDone - Ca.t_cancel; Ca['tw'] = (Ca.t_cancel - Ca.ws) / 1000.0
    slow_o = Ca.dc > 500.0
    ph3['live_orders'] = dict(cancelled=int(len(Ca)), slow=int(slow_o.sum()), slow_share=float(slow_o.mean()), slow_windows=int(Ca.market[slow_o].nunique()),
                              slow_throttled=int(Ca.t_Throttled[slow_o].notna().sum()), throttled_all=int(Ca.t_Throttled.notna().sum()),
                              slow_rejected=int(Ca.t_Rejected[slow_o].notna().sum()),
                              slow_share_by_cancel_second={f'{a_}-{b_} s': float(slow_o[(Ca.tw >= a_) & (Ca.tw < b_)].mean()) if ((Ca.tw >= a_) & (Ca.tw < b_)).any() else float('nan')
                                                           for a_, b_ in ((0, 30), (30, 50), (50, 60), (60, 120), (120, 300))},
                              slow_dc_p50_p90=[float(v) for v in np.percentile(Ca.dc[slow_o], [50, 90])] if slow_o.any() else [])
    res['PH3'] = ph3
    json.dump(res, open(f'{out}/b12b_posthoc.json', 'w'), indent=1, default=float)
    f = lambda v: 'n/a' if not np.isfinite(v) else f'{v:+.2f}'
    L = ['B12b POST HOC (written after reading b12b.txt; descriptive). c/sh at 1 s.', '',
         f"PH1 top 5 against O, Part A legs with a finite v ({n_nan} legs with NaN v excluded): WITHIN_G {f(Wg)}",
         f"  V10 (v decile): MIX {f(d10['MIX'])} WITHIN {f(d10['WITHIN'])} X {ph1['V10']['X']:.2f}; edges {[round(x, 4) for x in e10]}"]
    L += [f"    decile {i}: wT {c['wT'] * 100:.1f}% wO {c['wO'] * 100:.1f}% vT {f(c['vT'])} vO {f(c['vO'])}" for i, c in enumerate(ph1['V10']['cells'])]
    L += [f"  V5D3 (v quintile x dv tercile; {ph1['V5D3']['excluded_legs']} legs without dv excluded): WITHIN_G {f(Wgd)} MIX {f(d53['MIX'])} WITHIN {f(d53['WITHIN'])} X {ph1['V5D3']['X']:.2f}; dv edges {[round(x, 4) for x in ed]}"]
    L += [f"    dv tercile {i} (falling -> rising): wT {c['wT'] * 100:.1f}% wO {c['wO'] * 100:.1f}% T {f(c['T'])} O {f(c['O'])}" for i, c in enumerate(by_dv)]
    L += [f"  T4G (0-5 / 5-15 / 15-30 / 30-60 s x guard ON, OFF): MIX {f(d4['MIX'])} WITHIN {f(d4['WITHIN'])} X {ph1['T4G']['X']:.2f}"]
    L += [f"    {'ON' if i < 4 else 'OFF'} {['0-5', '5-15', '15-30', '30-60'][i % 4]} s: wT {c['wT'] * 100:.1f}% wO {c['wO'] * 100:.1f}% vT {f(c['vT'])} vO {f(c['vO'])}" for i, c in enumerate(ph1['T4G']['cells'])]
    L += ['', 'PH2 our at-best legs against the audit match interval [place, min(fill notice, confirmed cancel)]']
    for name, x in ph2.items():
        L.append(f"{name}: legs {x['legs']} sh {x['shares']:.1f} net {f(x['net1'])}; research time {x['research_time_position']}; guard at upper - 50 {x['guard_at_upper_minus_50']}; "
                 f"mean share of the interval OFF {x['interval_off_share_mean']:.2f}; upper = audit cancel in {x['upper_is_cancel']}; cancel request before upper in {x['cancel_request_before_upper']}")
        if name == 'OFF':
            for r in x['legs_detail']:
                L.append(f"   t {r['t'] / 1000:.1f} s {r['shares']:.1f} sh | {r['pos']} | interval {r['interval_ms']:.0f} ms | anc - upper {r['anc_minus_upper']:.0f} | req - upper {r['cancel_req_minus_upper']:.0f} | "
                         f"audit cancel - req {r['audit_cancel_minus_req']:.0f} | done - req {r['done_minus_req']:.0f} | notice - req {r['notice_minus_req']:.0f} | guard at upper {r['guard_at_upper']} | "
                         f"OFF share {r['interval_off_share']:.2f} | v1 {r['v1']:+.2f}")
    L += ['', 'PH3 slow cancels (request to cancel ack > 500 ms)']
    for k, v in ph3['off_legs'].items():
        L.append(f"OFF legs, {k}: legs {v['legs']} sh {v['shares']:.1f} windows {v['windows']} net {f(v['net1'])} $/day {v['usd_day1']:+.2f}; seconds into window {v['t_in_window_s']}")
    L.append(f"All cancelled live orders: {json.dumps(ph3['live_orders'])}")
    open(f'{out}/b12b_posthoc.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
