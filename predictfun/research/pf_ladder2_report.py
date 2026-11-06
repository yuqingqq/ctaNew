"""Report of the ladder / improve / per-minute-line / fresh-refresh replays (run_ladder2.sh; 2026-11-05; exploratory, IN-SAMPLE:
the per-minute lines and the guard were fit on the same 8 read days; 11-03 beside). Money = mark@stop $ / day without rebate.
Per arm: 8-day mean, days > 0, 11-03; per lane (touch 0/1 = bid/ask, rung 2/3, improve = touch fills at a price inside the
venue spread at placement is not stored, so improve shows as a touch-lane change between arms): shares, mark@stop c/sh (each fill
valued at the mid at stop + 100 ms from the book frames; per window the lanes sum to the engine's mtm_stop, refused otherwise),
+1 s c/sh, swept share, median queue ahead at activation; requests / window, throttled, invariants (must be 0).
Comparisons (computed): day-clustered mean difference with a t interval (7 df, 2.365); 'excludes 0' evaluated, not asserted.
Cross-check: arm 'today (no GS)' at 50 ms predictions and central timing equals the fixed live engine's C4 of the same round (kJ<RND>) on every day (refused otherwise).
Usage: python3 -m predictfun.research.pf_ladder2_report
"""
import json, os
import numpy as np, pandas as pd
from . import pf_io as io
from .pf_minute_pnl import day_mids, stop_mids

S4 = f'{io.OUT}/step4'
D8 = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']; BES = '20261103'
B = 'MICRO+G4+G2+G6+G5+G3+G1+'; T = '+XNA+XMIN1+XV+S60/mid@'
ARMS = {'today (no GS)': B + 'V0' + T + 'AV:5:25:0.06', 'today': B + 'V0+GS' + T + 'AV:5:25:0.06', 'pm': B + 'PM90+GS' + T + 'AM:5:25:95',
        'pm+ladder': B + 'PM90+GS+LD2' + T + 'AM:5:25:95', 'pm+improve': B + 'PM90+GS+IM95' + T + 'AM:5:25:95',
        'pm+both': B + 'PM90+GS+LD2+IM95' + T + 'AM:5:25:95', 'today+ladder': B + 'V0+GS+LD2' + T + 'AV:5:25:0.06',
        'pm+ladder 20/100': B + 'PM90+GS+LD2' + T + 'AM:20:100:95', 'pm+ladder hold': B + 'PM90+GS+LD2+LH' + T + 'AM:5:25:95'}
RND = os.environ.get('PF_LAD2_ROUND', 'i')   # run_ladder2.sh round suffix: i = after review-4 R7 / R8 fixes; round 3 = 'h' (its lJ / lX had no suffix)
RUNS = {(p, k): f'{S4}/ladder_tokyoreal_custom_e2_{p}_{k}{RND if RND != "h" else ""}' for p in ('preds_t60k', 'preds_t60kE') for k in ('lJ', 'lX')}
LIVEKJ = f'{S4}/live_tokyoreal_custom_e2_preds_t60k_kJ{RND}'   # the fixed live engine of the same round
stem = lambda n: n.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')
CMP = [('today', 'today (no GS)'), ('pm', 'today'), ('pm+ladder', 'pm'), ('pm+improve', 'pm'), ('pm+both', 'pm'), ('today+ladder', 'today'), ('pm+ladder 20/100', 'pm+ladder'), ('pm+ladder hold', 'pm'), ('pm+ladder hold', 'pm+ladder')]


def ci(x):
    x = np.asarray(x, float); m = x.mean(); h = 2.365 * x.std(ddof=1) / np.sqrt(len(x))
    return dict(mean=round(m, 1), lo=round(m - h, 1), hi=round(m + h, 1), excludes_0=bool(m - h > 0 or m + h < 0), days_pos=int((x > 0).sum()))


def main():
    out = dict(note='exploratory, in-sample; mark@stop $/day without rebate', runs={}, cross_check={}, recon={})
    MIDS = {}
    for (p, k), R in RUNS.items():
        if not all(os.path.exists(f'{R}/{d}/summary.json') for d in D8 + [BES]):
            print('missing run', p, k); continue
        res = {}
        for lab, name in ARMS.items():
            per = {}; lanes = {}
            for d in D8 + [BES]:
                s = json.load(open(f'{R}/{d}/summary.json'))['arms'][name]
                F = pd.read_parquet(f'{R}/{d}/fills/{stem(name)}.parquet'); W = pd.read_parquet(f'{R}/{d}/win/{stem(name)}.parquet')
                W = W[W.n_fills > 0]
                if d not in MIDS:
                    MIDS[d] = day_mids(d)
                ms = stop_mids(MIDS[d], W, 60)
                sg = np.where(F.side.values == 0, 1.0, -1.0)
                F['stop'] = sg * (F.mid.map(ms).values - F.pt.values) * F.q.values / 100.0
                chk = F.groupby('mid').stop.sum().reindex(W.mid).values - W.mtm_stop.values
                bad = int(np.sum(np.abs(chk[np.isfinite(W.mtm_stop.values)]) > 1e-6)); out['recon'][f'{p} {k} {lab} {d}'] = bad
                assert bad == 0, (p, k, lab, d, bad)
                c = s['live']['counters']
                assert c.get('own_dup_price', 0) == 0 and c.get('own_cross', 0) == 0, (p, k, lab, d, 'invariant')
                per[d] = dict(usd=s['mtm_stop_usd'], req_per_win=sum(s['live']['requests'].values()) / max(s['windows'], 1),
                              throttled=c.get('throttled', 0), improve_wanted=c.get('improve_wanted', 0))
                F['rung'] = F.lane >= 2; F['day'] = d
                lanes[d] = F[['day', 'rung', 'q', 'stop', 'mo1', 'through', 'ahead0']]
            L = pd.concat(lanes.values()); L8 = L[L.day != BES]
            x = np.array([per[d]['usd'] for d in D8])
            lane_tab = {}
            for rg, g in L8.groupby('rung'):
                lane_tab['rung' if rg else 'touch'] = dict(shares_day=round(g.q.sum() / 8, 1), usd_day=round(g.stop.sum() / 8, 1),
                                                          c_sh=round(100 * g.stop.sum() / g.q.sum(), 2), mo1_c_sh=round(100 * g.mo1.sum() / g.q.sum(), 2),
                                                          swept=round(float((g.q * g.through).sum() / g.q.sum()), 3), ahead0_median=round(float(g.ahead0.median()), 1))
            res[lab] = dict(usd_day=ci(x), per_day={d: round(per[d]['usd'], 1) for d in D8}, d1103=round(per[BES]['usd'], 1),
                            req_per_window=round(np.mean([per[d]['req_per_win'] for d in D8]), 1), throttled_day=round(np.mean([per[d]['throttled'] for d in D8]), 1),
                            improve_wanted_day=round(np.mean([per[d]['improve_wanted'] for d in D8]), 1), lanes=lane_tab)
        res['_compare'] = {f'{a} - {b}': ci([res[a]['per_day'][d] - res[b]['per_day'][d] for d in D8]) | dict(d1103=round(res[a]['d1103'] - res[b]['d1103'], 1)) for a, b in CMP}
        out['runs'][f'{p} {k}'] = res
        if (p, k) == ('preds_t60k', 'lJ'):                     # cross-check against the live engine
            for d in D8 + [BES]:
                a = json.load(open(f'{LIVEKJ}/{d}/summary.json'))['arms'][ARMS['today (no GS)']]['mtm_stop_usd']
                b = res['today (no GS)']['per_day'].get(d, res['today (no GS)']['d1103'] if d == BES else None)
                out['cross_check'][d] = (a, b); assert abs(round(a, 1) - b) < 0.051, ('cross-check', d, a, b)
    for lab in ARMS:                                            # fresh minus grid, per timing
        for k in ('lJ', 'lX'):
            g, e = out['runs'].get(f'preds_t60k {k}'), out['runs'].get(f'preds_t60kE {k}')
            if g and e:
                out.setdefault('fresh_minus_grid', {})[f'{lab} {k}'] = ci([e[lab]['per_day'][d] - g[lab]['per_day'][d] for d in D8]) | dict(d1103=round(e[lab]['d1103'] - g[lab]['d1103'], 1))
    io.write_json(f'{S4}/report_ladder2' + ('' if RND == 'h' else f'_{RND}') + '.json', out)
    for rk, res in out['runs'].items():
        print(f'\n== {rk}')
        for lab in ARMS:
            r = res[lab]; u = r['usd_day']
            ln = ' | '.join(f"{n}: {v['shares_day']:.0f} sh/day {v['usd_day']:+.1f}$ {v['c_sh']:+.2f}c/sh +1s {v['mo1_c_sh']:+.2f} swept {v['swept']:.0%} ahead {v['ahead0_median']:.0f}" for n, v in r['lanes'].items())
            print(f"{lab:17s} {u['mean']:+7.1f} $/day ({u['days_pos']}/8) 11-03 {r['d1103']:+7.1f} req/win {r['req_per_window']:6.1f} throttled/day {r['throttled_day']:7.1f} | {ln}")
        for c, v in res['_compare'].items():
            print(f"   {c:28s} {v['mean']:+7.1f} ({v['lo']:+.1f}..{v['hi']:+.1f}) excl0 {v['excludes_0']} {v['days_pos']}/8 11-03 {v['d1103']:+.1f}")
    for c, v in out.get('fresh_minus_grid', {}).items():
        print(f"fresh - grid {c:24s} {v['mean']:+7.1f} ({v['lo']:+.1f}..{v['hi']:+.1f}) excl0 {v['excludes_0']} {v['days_pos']}/8 11-03 {v['d1103']:+.1f}")


if __name__ == '__main__':
    main()
