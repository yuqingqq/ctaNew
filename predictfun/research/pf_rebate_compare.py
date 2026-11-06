"""P&L with the maker rebate, and ours vs the top makers (2026-11-04; user: "check the pnl after rebate, then how is the performance
against top makers"). Rebate = the replay's formula: 25% of a 2% taker fee on min(p, 1 - p) per share (Predict's September maker-rebate
promotion; UNVERIFIED for November - reported beside the no-rebate numbers, never in place of them). Liquidity rewards not included.
Ours: P60 arms of the stage-3 runs (current guard M025; trained guard V0 instead of M025 at pull 50 / 60 / 70%), 8 read days and 11-03.
Market makers: timed legs in the first 60 s of each window, bands by day volume (top 5 / 6-20 / rest), +1 s mark-out from the fill
price (as pf_maker_bad_fills) and the same rebate formula on their fills.
Usage: python3 -m predictfun.research.pf_rebate_compare
"""
import json, os
import numpy as np, pandas as pd
from . import pf_io as io

D8 = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']; S4 = f'{io.OUT}/step4'; B = 'MICRO+G4+G2+G6+G5+G3+G1'
stem = lambda n: n.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')
arm = lambda tok: f'{B}+{tok}+XNA+XMIN1+S60/mid@A1:5:25:0'
reb = lambda p, q: 0.25 * 0.02 * np.minimum(p, 100 - p) / 100.0 * q


def ours(days, k, tok):
    S = [json.load(open(f'{S4}/live_tokyoreal_custom_e2_preds_g{k}_gd/{d}/summary.json'))['arms'][arm(tok)] for d in days]
    f = pd.concat([pd.read_parquet(f'{S4}/live_tokyoreal_custom_e2_preds_g{k}_gd/{d}/fills/{stem(arm(tok))}.parquet', columns=['q', 'pt', 'mo1', 'rebate']) for d in days])
    q = f.q.sum(); c = 100 * f.mo1 / f.q; bad = c <= -2; good = c >= 0; r = float(f.rebate.sum())
    chk = float(reb(f.pt.values.astype(float), f.q.values).sum())
    return dict(shares=float(q), mtm=float(sum(s['mtm_stop_usd'] for s in S)), gross=float(sum(s['gross_usd'] for s in S)), rebate=r, rebate_check=chk,
                mo1_usd=float(f.mo1.sum()), mo1_c=float(100 * f.mo1.sum() / q), rebate_c=float(100 * r / q), bad=float(f.q[bad].sum() / q),
                ratio=float(-f.mo1[bad].sum() / f.mo1[good].sum()), days=len(days))


rows = {}
for lab, k, tok in (('current guard (M025)', 20, 'M025'), ('trained guard 50%', 50, 'V0'), ('trained guard 60%', 60, 'V0'), ('trained guard 70%', 70, 'V0')):
    rows[lab] = dict(read=ours(D8, k, tok), fresh=ours(['20261103'], k, tok))
# market makers, first 60 s
acc = []
for d in D8:
    S1 = f'{io.OUT}/step1/{d}'
    bk = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=['mid', 't_ms', 'b1_p', 'a1_p']) for f in sorted(os.listdir(f'{S1}/books'))])
    bk = bk[(bk.b1_p >= 0) & (bk.a1_p > bk.b1_p)].sort_values(['mid', 't_ms'], kind='stable')
    BY = {m: (g.t_ms.values.astype(float), (g.b1_p.values + g.a1_p.values) / 2.0) for m, g in bk.groupby('mid', sort=False)}; del bk
    lt = pd.read_parquet(f'{S1}/legs_timed2.parquet', columns=['mid', 'side', 'ptick', 'shares', 't_ms', 'how', 't_e2', 'signer', 't_rel'])
    lt = lt[(lt.t_rel >= 0) & (lt.t_rel < 240)].copy()
    vol = lt.groupby('signer').shares.sum().sort_values(ascending=False); rk = pd.Series(np.arange(1, len(vol) + 1), index=vol.index)
    r_ = lt.signer.map(rk).values; lt['band'] = np.where(r_ <= 5, 'top 5', np.where(r_ <= 20, 'ranks 6-20', 'rest'))
    lt = lt[(lt.t_rel < 60) & ((lt.how.values > 0) | np.isfinite(lt.t_e2.values))].copy(); lt['ta'] = np.where(lt.how.values > 0, lt.t_ms.values, lt.t_e2.values)
    for mid, g in lt.groupby('mid', sort=False):
        v = BY.get(mid)
        if v is None:
            continue
        tv, mv = v; i1 = np.searchsorted(tv, g.ta.values + 1000.0, side='right') - 1; ok = i1 >= 0
        s = np.where(g.side.values == 0, 1.0, -1.0); c = s * (mv[np.clip(i1, 0, None)] - g.ptick.values)
        acc.append(pd.DataFrame(dict(day=d, band=g.band.values, q=g.shares.values, c=c, rb=reb(g.ptick.values.astype(float), g.shares.values)))[ok])
    del BY
    io.mem_check(f'day {d}')
M = pd.concat(acc); mk = {}
for b, g in M.groupby('band'):
    q = g.q.sum(); u = g.q * g.c / 100; bad = g.c <= -2; good = g.c >= 0
    mk[b] = dict(shares=float(q), mo1_usd=float(u.sum()), mo1_c=float(100 * u.sum() / q), rebate=float(g.rb.sum()), rebate_c=float(100 * g.rb.sum() / q),
                 bad=float(g.q[bad].sum() / q), ratio=float(-u[bad].sum() / u[good].sum()), days=8)
print('rebate check (ours): stored', {k: round(v['read']['rebate'], 2) for k, v in rows.items()}, 'recomputed', {k: round(v['read']['rebate_check'], 2) for k, v in rows.items()})
print('\nfirst 60 s of each window, 8 read days; c/sh at 1 s from the fill price; rebate = 25% of the 2% taker fee (UNVERIFIED)')
print(f"{'':26s} {'shares/day':>10s} {'+1 s c/sh':>9s} {'rebate c/sh':>11s} {'+1 s + reb':>10s} {'bad %':>6s} {'ratio':>6s} | {'$/day +1 s':>10s} {'+ rebate':>9s} | {'mark@stop $/day':>15s} {'+ rebate':>9s}")
for lab, r in [(k, v['read']) for k, v in rows.items()] + [(f'market: {b}', mk[b]) for b in ('top 5', 'ranks 6-20', 'rest')]:
    dd = r['days']; ms = f"{r['mtm'] / dd:15.1f} {(r['mtm'] + r['rebate']) / dd:9.1f}" if 'mtm' in r else f"{'-':>15s} {'-':>9s}"
    print(f"{lab:26s} {r['shares'] / dd:10.0f} {r['mo1_c']:+9.3f} {r['rebate_c']:11.3f} {r['mo1_c'] + r['rebate_c']:+10.3f} {r['bad']:6.1%} {r['ratio']:6.0%} | "
          f"{r['mo1_usd'] / dd:10.1f} {(r['mo1_usd'] + r['rebate']) / dd:9.1f} | {ms}")
print('\n11-03 (unseen):')
for lab, v in rows.items():
    r = v['fresh']
    print(f"  {lab:24s} shares {r['shares']:7.0f}  +1 s {r['mo1_c']:+.3f} + rebate {r['rebate_c']:.3f} = {r['mo1_c'] + r['rebate_c']:+.3f} c/sh  mark@stop {r['mtm']:+7.1f} -> {r['mtm'] + r['rebate']:+7.1f}  "
          f"gross {r['gross']:+7.1f} -> {r['gross'] + r['rebate']:+7.1f}")
io.write_json(f'{S4}/report_rebate_compare.json', dict(ours=rows, market=mk))
