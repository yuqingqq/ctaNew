"""P&L of the replay fills by per-minute guard percentile (descriptive; user 2026-11-05: "what's the pnl performance and other metrics
about the percentile test"). No new replay: reads the central kJ run (real taker-order key + within-frame time).
Each fill gets
  v      guard value of its side (out-of-fold preds_t60k, latest row with tau <= fill time), c above today's global pull line
  pct    bucket of v within the fill's minute, edges = that minute's percentiles of all guard rows (8 read days, bid + ask pooled,
         as pf_minute_pct (A))
  stop   the fill's share of the window's mark@stop: sign x q x (mid at stop + 100 ms - price); mid from the book frames exactly as the
         engine's mid_at (last valid frame with t_ms <= ws + stop + 100 ms). Reconciliation: per window sum(stop) == win.mtm_stop
         (refuses on mismatch > 1e-6 $).
  mo1 / settle  the engine's +1 s mark and hold-to-settle value ($), rebate not included.
Buckets below today's pull line hold only fills of orders still live while the pull was in flight; the policy never quotes there,
so those rows are not an estimate of quoting at a lower line. Fill-level sums ignore path effects (inventory caps, queue).
Usage: python3 -m predictfun.research.pf_minute_pnl
"""
import glob, json, numpy as np, pandas as pd
from . import pf_io as io

D8 = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']; BESIDE = '20261103'
RUN = f'{io.OUT}/step4/live_tokyoreal_custom_e2_preds_t60k_kJ'
PRE = 'MICRO_G4_G2_G6_G5_G3_G1_V0_XNA_XMIN1_XV_'
ARMS = {'C1 S60': 'S60__mid_szA1-5-25-0', 'C2 S60': 'S60__mid_szA1-20-100-0', 'C4 S60': 'S60__mid_szAV-5-25-0.06',
        'C3 S60': 'S60__mid_szAV-20-100-0.06', 'C4 S240': 'S240__mid_szAV-5-25-0.06', 'C3 S240': 'S240__mid_szAV-20-100-0.06'}
EDGES = [0, 50, 60, 70, 80, 90, 95, 100]
assert all(d not in ('20261028', '20261029', '20261030') and d < '20261104' for d in D8 + [BESIDE])


def preds(d):
    p = pd.read_parquet(f'{io.OUT}/step5/{d}/preds_t60k.parquet', columns=['mid', 'tau', 'v_bid', 'v_ask'])
    ws = pd.read_parquet(f'{io.OUT}/step1/{d}/registry.parquet', columns=['mid', 'ws']).set_index('mid').ws
    p['rel'] = (p.tau.values - p.mid.map(ws).values * 1000.0) / 1000.0
    return p


def day_mids(d):
    """valid book mids of day d's own-date frames, sorted by t_ms (pf_step4_live.mid_at's frames)."""
    B = pd.concat([pd.read_parquet(f, columns=['mid', 't_ms', 'b1_p', 'a1_p']) for f in sorted(glob.glob(f'{io.OUT}/step1/{d}/books/part_{d}_*.parquet'))])
    B = B[(B.b1_p.values >= 0) & (B.a1_p.values > B.b1_p.values)]
    return pd.DataFrame(dict(mid=B.mid.values, t_ms=B.t_ms.values.astype(float), m=(B.a1_p.values + B.b1_p.values) / 2.0)).sort_values('t_ms', kind='stable')


def stop_mids(B, W, stop_s):
    """mid at ws + stop + 100 ms per window, as pf_step4_live.mid_at (last valid frame with t_ms <= t)."""
    T = pd.DataFrame(dict(mid=W.mid.values, t_ms=W.ws.values.astype(float) + stop_s * 1000.0 + 100.0)).sort_values('t_ms')
    return pd.merge_asof(T, B, on='t_ms', by='mid', direction='backward', allow_exact_matches=True).set_index('mid').m


def main():
    P = {d: preds(d) for d in D8 + [BESIDE]}
    allv = pd.concat([P[d] for d in D8]); allv = allv[(allv.rel >= 0) & (allv.rel < 240)]
    mi = np.minimum((allv.rel.values // 60).astype(int), 3)
    CUT = {}
    for i in range(4):
        v = np.r_[allv.v_bid.values[mi == i], allv.v_ask.values[mi == i]]; v = v[np.isfinite(v)]
        CUT[i] = np.percentile(v, EDGES[1:-1])
    del allv
    rows, recon, status = [], {}, {}
    for d in D8 + [BESIDE]:
        B = day_mids(d)
        for name, arm in ARMS.items():
            stop_s = int(arm.split('__')[0][1:])
            F = pd.read_parquet(f'{RUN}/{d}/fills/{PRE}{arm}.parquet'); W = pd.read_parquet(f'{RUN}/{d}/win/{PRE}{arm}.parquet')
            W = W[W.n_fills > 0]
            ms = stop_mids(B, W, stop_s)
            sg = np.where(F.side.values == 0, 1.0, -1.0)
            F['stop'] = sg * (F.mid.map(ms).values - F.pt.values) * F.q.values / 100.0
            chk = F.groupby('mid').stop.sum().reindex(W.mid).values - W.mtm_stop.values
            ok = np.isfinite(W.mtm_stop.values)
            bad = int(np.sum(~np.isfinite(chk[ok]) | (np.abs(chk[ok]) > 1e-6)))
            recon[f'{name} {d}'] = dict(windows=int(ok.sum()), mismatched=bad, nan_mtm=int((~ok).sum()))
            assert bad == 0, (name, d, bad)
            # falsifier: the same check on mids one tick off must flag every window holding inventory at the stop
            ctl = (sg * (F.mid.map(ms).values + 1.0 - F.pt.values) * F.q.values / 100.0)
            ctl = pd.Series(ctl, index=F.index).groupby(F.mid).sum().reindex(W.mid).values - W.mtm_stop.values
            held = ok & (np.abs(W.net_end.values) > 1e-9)
            assert np.all(np.abs(ctl[held]) > 1e-9), (name, d, 'control did not fire')
            recon[f'{name} {d}']['control_flagged'] = int(held.sum())
            # guard value of the fill's side, latest pred row at or before the fill
            Q = P[d][['mid', 'tau', 'v_bid', 'v_ask']].astype({'tau': float}).sort_values('tau')
            wsm = W.set_index('mid').ws
            F['tau'] = F.mid.map(wsm).values + F.t.values * 1000.0
            J = pd.merge_asof(F.reset_index().sort_values('tau'), Q, on='tau', by='mid', direction='backward').set_index('index').sort_index()
            F['v'] = np.where(F.side.values == 0, J.v_bid.values, J.v_ask.values)
            F['m'] = np.minimum(np.maximum(F.t.values // 60, 0).astype(int), 3)
            k = np.full(len(F), -1)
            for i in range(4):
                s = (F.m.values == i) & np.isfinite(F.v.values)
                k[s] = np.searchsorted(CUT[i], F.v.values[s], side='right')
            status[f'{name} {d}'] = dict(fills=len(F), no_pred=int((k < 0).sum()), stop_nan=int(F.stop.isna().sum()))
            F['b'] = k; F['day'] = d; F['arm'] = name
            rows.append(F[['arm', 'day', 'mid', 't', 'side', 'q', 'through', 'stop', 'mo1', 'settle', 'v', 'm', 'b']])
    A = pd.concat(rows, ignore_index=True)
    A.to_parquet(f'{io.OUT}/step4/minute_pnl_fills_kJ.parquet')
    out = dict(run=RUN, days=D8, beside=BESIDE, edges=EDGES, cuts={i: [round(float(x), 4) for x in CUT[i]] for i in range(4)},
               reconciliation=recon, status=status, tables={})
    for name in ARMS:
        X = A[(A.arm == name) & (A.b >= 0) & A.stop.notna()]
        R8, R3 = X[X.day != BESIDE], X[X.day == BESIDE]
        tot = R8.stop.sum() / len(D8)
        print(f'\n{name}: 8 read days, $ / day mark@stop {tot:+.1f} (fills {len(R8)}, shares {R8.q.sum():.0f}); '
              f'11-03 {R3.stop.sum():+.1f}; excluded no_pred {int((A[(A.arm == name)].b < 0).sum())}')
        print(f"{'minute':8s} {'pct':>7s} {'fills':>6s} {'sh %':>6s} {'$/day':>7s} {'c/sh':>7s} {'+1s c/sh':>8s} {'settle c/sh':>11s} "
              f"{'swept %':>7s} {'days>0':>6s} {'11-03 $':>8s}")
        tab = []
        for i in range(int(X.m.max()) + 1):
            for j in range(len(EDGES) - 1):
                g = R8[(R8.m == i) & (R8.b == j)]; g3 = R3[(R3.m == i) & (R3.b == j)]
                if len(g) == 0:
                    continue
                sh = g.q.sum(); per = g.groupby('day').stop.sum().reindex(D8, fill_value=0.0)
                r = dict(minute=i, pct=f'{EDGES[j]}-{EDGES[j + 1]}', fills=len(g), share_of_shares=sh / R8.q.sum(),
                         usd_day=g.stop.sum() / len(D8), c_sh=100 * g.stop.sum() / sh, mo1_c_sh=100 * g.mo1.sum() / sh,
                         settle_c_sh=100 * g.settle.sum() / sh, swept=float((g.q * g.through).sum() / sh),
                         days_pos=int((per > 0).sum()), usd_1103=float(g3.stop.sum()))
                tab.append(r)
                print(f"{60 * i:3d}-{60 * i + 60:<3d}s {r['pct']:>7s} {r['fills']:6d} {r['share_of_shares']:6.1%} {r['usd_day']:+7.1f} "
                      f"{r['c_sh']:+7.2f} {r['mo1_c_sh']:+8.2f} {r['settle_c_sh']:+11.2f} {r['swept']:7.1%} {r['days_pos']:4d}/8 {r['usd_1103']:+8.1f}")
        out['tables'][name] = tab
    json.dump(out, open(f'{io.OUT}/step4/report_minute_pnl.json', 'w'), indent=1, default=float)
    print('\nreconciliation: windows', sum(v['windows'] for v in recon.values()), 'mismatched', sum(v['mismatched'] for v in recon.values()))


if __name__ == '__main__':
    main()
