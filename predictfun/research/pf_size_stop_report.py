"""Size x stop grid report (declared in STATUS.yml 2026-11-04 10:32Z; user: "check if we can increase size and quote beyond 60s").
Ours: tag gz runs (run_size_stop.sh), trained guard V0 at pull k, arms MICRO+G4+G2+G6+G5+G3+G1+V0+XNA+XMIN1+S<stop> x size A1:lo:hi:0;
8 read days and 11-03 apart. Per arm: shares / day, +1 s c/sh from the fill price, rebate (25% of the 2% taker fee, UNVERIFIED), bad %
(+1 s <= -2 c), bad$/good$, mark@stop and settlement $ / day, window sd and worst day. Per 60 s segment of fill time (S240 arms).
Day-clustered deltas (G = 8, t 2.365) of longer stops vs S60 and bigger sizes vs A1:5:25:0, on mark@stop + rebate and settlement + rebate.
Market makers: timed legs by the same segments, bands by day volume over 0-240 s (top 5 / 6-20 / rest), +1 s mark-out as pf_rebate_compare.
Dynamic sizes (declared 10:37Z, tag gzd): fixed 25 / 50 / 100, AV:lo:hi:par (trained-guard value), A3:5:100:0.2 (depth), stop 60 / 240 s;
each sized arm (A1 of gz, AV, A3) against the fixed arms at the SAME shares (per day, linear interpolation of mark@stop + rebate and
settlement + rebate over the three fixed sizes on shares; outside their range = a status), and AV against A1 at the same lo / hi.
Touch: time-weighted displayed size at the best (b1_s, a1_s) per segment, live 0-240 s - our hi size against it (the fill model has no
own impact: our order is never shown to others, so large sizes are optimistic). Parity: gz S60 A1:5:25:0 = gd run (V0, same k).
Usage: python3 -m predictfun.research.pf_size_stop_report
"""
import json, os
import numpy as np, pandas as pd
from . import pf_io as io

D8 = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']; FR = ['20261103']
S4 = f'{io.OUT}/step4'; B = 'MICRO+G4+G2+G6+G5+G3+G1'; STOPS = (60, 120, 180, 240); SIZES = ('A1:5:25:0', 'A1:10:50:0', 'A1:20:100:0')
SEG = ((0, 60), (60, 120), (120, 180), (180, 240))
DYN = ('25', '50', '100', 'AV:10:50:0.03', 'AV:20:100:0.03', 'AV:20:100:0.06', 'A3:5:100:0.2'); DSTOPS = (60, 240)
stem = lambda n: n.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')
arm = lambda s, sz: f'{B}+V0+XNA+XMIN1+S{s}/mid@{sz}'
reb = lambda p, q: 0.25 * 0.02 * np.minimum(p, 100 - p) / 100.0 * q
seg_of = lambda t: np.minimum((np.asarray(t) // 60).astype(int), 3)


def qual(q, mo1, rb):
    c = 100 * mo1 / q; bad = c <= -2; good = c >= 0; Q = q.sum()
    return dict(shares=float(Q), mo1_usd=float(mo1.sum()), mo1_c=float(100 * mo1.sum() / Q) if Q else np.nan, rebate=float(rb.sum()),
                rebate_c=float(100 * rb.sum() / Q) if Q else np.nan, bad=float(q[bad].sum() / Q) if Q else np.nan,
                ratio=float(-mo1[bad].sum() / mo1[good].sum()) if mo1[good].sum() > 0 else np.nan)


def ci(dd):
    dd = np.asarray(dd, float); se = dd.std(ddof=1) / np.sqrt(len(dd))
    return dict(mean=float(dd.mean()), lo=float(dd.mean() - 2.365 * se), hi=float(dd.mean() + 2.365 * se), days_better=int((dd > 0).sum()), n=len(dd))


def load(k, days, tag='gz', grid=None):
    base = f'{S4}/live_tokyoreal_custom_e2_preds_g{k}_{tag}'; grid = grid or [(s, sz) for s in STOPS for sz in SIZES]
    if not all(os.path.exists(f'{base}/{d}/summary.json') for d in days):
        return None
    out = {}
    for s, sz in grid:
        if True:
            a = arm(s, sz); S = [json.load(open(f'{base}/{d}/summary.json'))['arms'][a] for d in days]
            f = pd.concat([pd.read_parquet(f'{base}/{d}/fills/{stem(a)}.parquet', columns=['t', 'q', 'pt', 'mo1', 'rebate', 'osz']).assign(day=d) for d in days])
            w = pd.concat([pd.read_parquet(f'{base}/{d}/win/{stem(a)}.parquet', columns=['pnl', 'mtm_stop']).assign(day=d) for d in days])
            r = qual(f.q.values, f.mo1.values, f.rebate.values)
            rb_day = f.groupby('day').rebate.sum().reindex(days, fill_value=0.0).values
            r.update(days=len(days), mtm=float(sum(x['mtm_stop_usd'] for x in S)), gross=float(sum(x['gross_usd'] for x in S)),
                     mtm_day=(np.array([x['mtm_stop_usd'] for x in S]) + rb_day).tolist(), gross_day=(np.array([x['gross_usd'] for x in S]) + rb_day).tolist(),
                     win_sd=float(w.pnl.std()), win_worst=float(w.pnl.min()), day_worst=float(min(np.array([x['gross_usd'] for x in S]) + rb_day)),
                     abs_net_stop=float(np.mean([x['abs_net_stop_mean'] for x in S])),
                     rebate_check=float(reb(f.pt.values.astype(float), f.q.values).sum()),
                     hi_share=float(f.q[f.osz >= float(sz.split(':')[2] if ':' in sz else sz) - 1e-9].sum() / f.q.sum()),
                     mean_osz=float((f.q * f.osz).sum() / f.q.sum()), shares_day=f.groupby('day').q.sum().reindex(days, fill_value=0.0).tolist())
            r['seg'] = {i: qual(g.q.values, g.mo1.values, g.rebate.values) for i, g in f.groupby(seg_of(f.t.values))}
            out[(s, sz)] = r
    if tag != 'gz':
        return out
    # parity with the gd run (same k, V0, S60, A1:5:25:0)
    a = arm(60, SIZES[0]); par = []
    for d in days:
        p = f'{S4}/live_tokyoreal_custom_e2_preds_g{k}_gd/{d}/summary.json'
        if os.path.exists(p):
            x = json.load(open(p))['arms'][a]; y = json.load(open(f'{base}/{d}/summary.json'))['arms'][a]
            par.append(abs(x['gross_usd'] - y['gross_usd']) < 1e-6 and x['fills'] == y['fills'] and abs(x['shares'] - y['shares']) < 1e-6)
    out['parity'] = dict(checked=len(par), equal=int(sum(par)))
    return out


DG = [(s, sz) for s in DSTOPS for sz in DYN]
res = {k: dict(read=load(k, D8), fresh=load(k, FR), dread=load(k, D8, 'gzd', DG), dfresh=load(k, FR, 'gzd', DG)) for k in (60, 50, 70)}
res = {k: v for k, v in res.items() if v['read'] is not None}
assert res, 'no complete k yet'

# market makers and the touch, by segment
acc, tch = [], []
for d in D8:
    S1 = f'{io.OUT}/step1/{d}'
    reg = pd.read_parquet(f'{S1}/registry.parquet', columns=['mid', 'ws']); WS = dict(zip(reg.mid.values, 1000.0 * reg.ws.values.astype(float)))   # registry ws is in seconds
    bk = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=['mid', 't_ms', 'b1_p', 'b1_s', 'a1_p', 'a1_s']) for f in sorted(os.listdir(f'{S1}/books'))])
    bk = bk[(bk.b1_p >= 0) & (bk.a1_p > bk.b1_p)].sort_values(['mid', 't_ms'], kind='stable')
    BY = {}
    for m, g in bk.groupby('mid', sort=False):
        t = g.t_ms.values.astype(float); BY[m] = (t, (g.b1_p.values + g.a1_p.values) / 2.0)
        if m in WS:
            rel = (t - WS[m]) / 1000.0; dur = np.minimum(np.diff(np.r_[t, t[-1]]), 5000.0); ok = (rel >= 0) & (rel < 240) & (dur > 0)
            tch.append(pd.DataFrame(dict(seg=seg_of(rel[ok]), w=dur[ok], b=g.b1_s.values[ok].astype(float), a=g.a1_s.values[ok].astype(float))))
    del bk
    lt = pd.read_parquet(f'{S1}/legs_timed2.parquet', columns=['mid', 'side', 'ptick', 'shares', 't_ms', 'how', 't_e2', 'signer', 't_rel'])
    lt = lt[(lt.t_rel >= 0) & (lt.t_rel < 240)].copy()
    vol = lt.groupby('signer').shares.sum().sort_values(ascending=False); rk = pd.Series(np.arange(1, len(vol) + 1), index=vol.index)
    r_ = lt.signer.map(rk).values; lt['band'] = np.where(r_ <= 5, 'top 5', np.where(r_ <= 20, 'ranks 6-20', 'rest'))
    lt = lt[(lt.how.values > 0) | np.isfinite(lt.t_e2.values)].copy(); lt['ta'] = np.where(lt.how.values > 0, lt.t_ms.values, lt.t_e2.values)
    for mid, g in lt.groupby('mid', sort=False):
        v = BY.get(mid)
        if v is None:
            continue
        tv, mv = v; i1 = np.searchsorted(tv, g.ta.values + 1000.0, side='right') - 1; ok = i1 >= 0
        s = np.where(g.side.values == 0, 1.0, -1.0); c = s * (mv[np.clip(i1, 0, None)] - g.ptick.values)
        acc.append(pd.DataFrame(dict(day=d, band=g.band.values, seg=seg_of(g.t_rel.values), q=g.shares.values, c=c,
                                     rb=reb(g.ptick.values.astype(float), g.shares.values)))[ok])
    del BY, lt
    io.mem_check(f'day {d}')
M = pd.concat(acc); del acc
mk = {(b, int(sg)): qual(g.q.values, g.q.values * g.c.values / 100, g.rb.values) for (b, sg), g in M.groupby(['band', 'seg'])}
T = pd.concat(tch); del tch


def wmed(x, w):
    o = np.argsort(x); cw = np.cumsum(w[o]); return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])


touch = {int(sg): dict(bid=wmed(g.b.values, g.w.values), ask=wmed(g.a.values, g.w.values)) for sg, g in T.groupby('seg')}

# print
H = f"{'':28s} {'sh/day':>7s} {'+1s c/sh':>8s} {'+reb':>6s} {'bad':>6s} {'ratio':>6s} | {'mark@stop+reb $/d':>17s} {'settle+reb $/d':>14s} {'win sd':>7s} {'worst day':>9s} {'|net|@stop':>10s} {'hi-size fills':>13s}"
pred = {}
for k, v in res.items():
    R = v['read']
    print(f'\n=== pull {k}% (trained guard V0), 8 read days; parity with gd: {R["parity"]}; rebate check ' +
          ('ok' if all(abs(R[(s, sz)]['rebate'] - R[(s, sz)]['rebate_check']) < 1e-6 * max(1, R[(s, sz)]['rebate']) for s in STOPS for sz in SIZES) else 'MISMATCH'))
    print(H)
    for sz in SIZES:
        for s in STOPS:
            r = R[(s, sz)]; dd = r['days']
            print(f"{'S' + str(s) + ' ' + sz:28s} {r['shares'] / dd:7.0f} {r['mo1_c']:+8.3f} {r['mo1_c'] + r['rebate_c']:+6.2f} {r['bad']:6.1%} {r['ratio']:6.0%} | "
                  f"{np.mean(r['mtm_day']):17.1f} {np.mean(r['gross_day']):14.1f} {r['win_sd']:7.2f} {r['day_worst']:9.1f} {r['abs_net_stop']:10.1f} {r['hi_share']:13.0%}")
    print('  longer stop vs S60 (same size), $/day, day-clustered 95% CI, days better of 8:')
    for sz in SIZES:
        for s in STOPS[1:]:
            a, b = R[(s, sz)], R[(60, sz)]
            cm = ci(np.array(a['mtm_day']) - b['mtm_day']); cg = ci(np.array(a['gross_day']) - b['gross_day'])
            pred[f'k{k}_S{s}_vs_S60_{sz}'] = dict(mtm=cm, gross=cg, adds_mtm=bool(cm['lo'] > 0), adds_gross=bool(cg['lo'] > 0))
            print(f"    {sz:12s} S{s:<3d} mark@stop {cm['mean']:+7.1f} ({cm['lo']:+.1f}..{cm['hi']:+.1f}) {cm['days_better']}/8   settle {cg['mean']:+7.1f} ({cg['lo']:+.1f}..{cg['hi']:+.1f}) {cg['days_better']}/8")
    print('  bigger size vs A1:5:25:0 (same stop):')
    for s in STOPS:
        for sz in SIZES[1:]:
            a, b = R[(s, sz)], R[(s, SIZES[0])]
            cm = ci(np.array(a['mtm_day']) - b['mtm_day']); cg = ci(np.array(a['gross_day']) - b['gross_day'])
            pred[f'k{k}_{sz}_vs_x1_S{s}'] = dict(mtm=cm, gross=cg, adds_mtm=bool(cm['lo'] > 0), adds_gross=bool(cg['lo'] > 0))
            print(f"    S{s:<3d} {sz:12s} mark@stop {cm['mean']:+7.1f} ({cm['lo']:+.1f}..{cm['hi']:+.1f}) {cm['days_better']}/8   settle {cg['mean']:+7.1f} ({cg['lo']:+.1f}..{cg['hi']:+.1f}) {cg['days_better']}/8")
    print('  by segment of fill time (S240 arms) vs top 5 in the same segment: shares/day, +1 s c/sh (+ rebate), bad, ratio; meets top 5 = c/sh >= and bad <= and ratio <=')
    for sz in SIZES:
        for i in range(4):
            r = R[(240, sz)]['seg'].get(i); m5 = mk[('top 5', i)]
            if r is None:
                continue
            meets = bool(r['mo1_c'] >= m5['mo1_c'] and r['bad'] <= m5['bad'] and r['ratio'] <= m5['ratio']); pred[f'k{k}_{sz}_seg{i}_meets_top5'] = meets
            print(f"    {sz:12s} {SEG[i][0]:3d}-{SEG[i][1]:3d} s  {r['shares'] / 8:7.0f} {r['mo1_c']:+7.3f} ({r['mo1_c'] + r['rebate_c']:+.2f}) {r['bad']:6.1%} {r['ratio']:5.0%}  meets top 5: {meets}")
    F = v['fresh']
    if F is not None:
        print('  11-03 (seen once, for the pull choice): mark@stop + rebate / settle + rebate $, +1 s c/sh')
        for sz in SIZES:
            print('    ' + sz.ljust(12) + '  '.join(f"S{s} {F[(s, sz)]['mtm_day'][0]:+6.1f} / {F[(s, sz)]['gross_day'][0]:+6.1f} ({F[(s, sz)]['mo1_c']:+.2f})" for s in STOPS))


def matched(R, D, s, r, key):
    """Per day: the fixed arms' money interpolated at this arm's shares; None when outside the fixed range."""
    out = []
    for j in range(len(r['shares_day'])):
        xs = np.array([D[(s, z)]['shares_day'][j] for z in ('25', '50', '100')]); ys = np.array([D[(s, z)][key][j] for z in ('25', '50', '100')])
        o = np.argsort(xs); x = r['shares_day'][j]
        out.append(float(np.interp(x, xs[o], ys[o])) if xs.min() <= x <= xs.max() else None)
    return out


for k, v in res.items():
    D = v['dread']
    if D is None:
        continue
    G = v['read']; print(f'\n=== dynamic sizes, pull {k}%, 8 read days' + '' + ' (A1 arms from the grid)')
    print(H.replace('hi-size fills', 'mean order sz'))
    for s in DSTOPS:
        for nm, r in [(sz, D[(s, sz)]) for sz in DYN] + [(sz, G[(s, sz)]) for sz in SIZES]:
            dd = r['days']
            print(f"{'S' + str(s) + ' ' + nm:28s} {r['shares'] / dd:7.0f} {r['mo1_c']:+8.3f} {r['mo1_c'] + r['rebate_c']:+6.2f} {r['bad']:6.1%} {r['ratio']:6.0%} | "
                  f"{np.mean(r['mtm_day']):17.1f} {np.mean(r['gross_day']):14.1f} {r['win_sd']:7.2f} {r['day_worst']:9.1f} {r['abs_net_stop']:10.1f} {r['mean_osz']:13.1f}")
    print('  sized arm minus fixed sizes at the same shares (per-day interpolation), $/day, 95% CI, days better; n = days inside the fixed range')
    for s in DSTOPS:
        for nm, r in [(sz, D[(s, sz)]) for sz in DYN[3:]] + [(sz, G[(s, sz)]) for sz in SIZES]:
            cells = []
            for key, lab in (('mtm_day', 'mark@stop'), ('gross_day', 'settle')):
                m_ = matched(G, D, s, r, key); dd = [a - b for a, b in zip(r[key], m_) if b is not None]
                if len(dd) >= 5:
                    c = ci(dd); pred[f'k{k}_S{s}_{nm}_vs_fixed_{key}'] = dict(c, beats=bool(c['lo'] > 0))
                    cells.append(f"{lab} {c['mean']:+7.1f} ({c['lo']:+.1f}..{c['hi']:+.1f}) {c['days_better']}/{c['n']}")
                else:
                    cells.append(f'{lab} n={len(dd)} (outside the fixed range: status, no estimate)')
            print(f"    S{s:<3d} {nm:16s} " + '   '.join(cells))
    print('  AV minus A1 at the same lo / hi:')
    for s in DSTOPS:
        for av, a1 in (('AV:10:50:0.03', 'A1:10:50:0'), ('AV:20:100:0.03', 'A1:20:100:0'), ('AV:20:100:0.06', 'A1:20:100:0')):
            cm = ci(np.array(D[(s, av)]['mtm_day']) - G[(s, a1)]['mtm_day']); cg = ci(np.array(D[(s, av)]['gross_day']) - G[(s, a1)]['gross_day'])
            pred[f'k{k}_S{s}_{av}_vs_{a1}'] = dict(mtm=cm, gross=cg, beats_mtm=bool(cm['lo'] > 0))
            print(f"    S{s:<3d} {av:16s} vs {a1:12s} mark@stop {cm['mean']:+7.1f} ({cm['lo']:+.1f}..{cm['hi']:+.1f}) {cm['days_better']}/8   settle {cg['mean']:+7.1f} ({cg['lo']:+.1f}..{cg['hi']:+.1f}) {cg['days_better']}/8")
    print('  by segment of fill time (S240 dynamic arms): shares/day, +1 s c/sh (+ rebate), bad, ratio, meets top 5')
    for sz in DYN:
        for i in range(4):
            r = D[(240, sz)]['seg'].get(i); m5 = mk[('top 5', i)]
            if r is None:
                continue
            meets = bool(r['mo1_c'] >= m5['mo1_c'] and r['bad'] <= m5['bad'] and r['ratio'] <= m5['ratio']); pred[f'k{k}_{sz}_seg{i}_meets_top5'] = meets
            print(f"    {sz:16s} {SEG[i][0]:3d}-{SEG[i][1]:3d} s  {r['shares'] / 8:7.0f} {r['mo1_c']:+7.3f} ({r['mo1_c'] + r['rebate_c']:+.2f}) {r['bad']:6.1%} {r['ratio']:5.0%}  meets top 5: {meets}")
    F = v['dfresh']
    if F is not None:
        print('  11-03: mark@stop + rebate / settle + rebate $, +1 s c/sh')
        for sz in DYN:
            print('    ' + sz.ljust(16) + '  '.join(f"S{s} {F[(s, sz)]['mtm_day'][0]:+6.1f} / {F[(s, sz)]['gross_day'][0]:+6.1f} ({F[(s, sz)]['mo1_c']:+.2f})" for s in DSTOPS))
print('\nmarket makers by segment, 8 read days: shares/day (per top-5 maker), +1 s c/sh (+ rebate), bad, ratio')
for b in ('top 5', 'ranks 6-20', 'rest'):
    for i in range(4):
        m = mk[(b, i)]; per = m['shares'] / 8 / (5 if b == 'top 5' else 15 if b == 'ranks 6-20' else 1)
        print(f"  {b:10s} {SEG[i][0]:3d}-{SEG[i][1]:3d} s  {m['shares'] / 8:8.0f} ({per:7.0f}) {m['mo1_c']:+7.3f} ({m['mo1_c'] + m['rebate_c']:+.2f}) {m['bad']:6.1%} {m['ratio']:5.0%}")
print('\ndisplayed size at the best, time-weighted median, live windows (shares): ' + ', '.join(f"{SEG[i][0]}-{SEG[i][1]} s bid {touch[i]['bid']:.0f} ask {touch[i]['ask']:.0f}" for i in range(4)))
print('our hi sizes 25 / 50 / 100 as a share of the 0-60 s median touch: ' + ', '.join(f"{h / (0.5 * (touch[0]['bid'] + touch[0]['ask'])):.0%}" for h in (25, 50, 100)))


def js(o):
    if isinstance(o, dict):
        return {(f'S{k[0]}|{k[1]}' if isinstance(k, tuple) and len(k) == 2 and isinstance(k[0], int) and isinstance(k[1], str) and k[1].startswith('A')
                 else '|'.join(map(str, k)) if isinstance(k, tuple) else str(k)): js(v) for k, v in o.items()}
    return o


io.write_json(f'{S4}/report_size_stop.json', js(dict(ours=res, market=mk, touch=touch, predicates=pred)))
