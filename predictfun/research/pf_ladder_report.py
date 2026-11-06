"""Ladder test report (declared in STATUS.yml 2026-11-03 15:55Z; logic docs §35).

Per arm over the six read days and the two fresh days: mark-at-stop (spread minus adverse selection, no outcome), gross,
carry sd per window, spread and 1 s informed cost per share, the fill mix (queue fills at our level vs swept through)
and their 1 s value, mean leftover, requests and throttle refusals. Rule (computed): pick the arm with the largest read-day
mark-at-stop sum among arms with mark-at-stop > 0 on >= 4/6 read days; promising iff its fresh-day mark-at-stop sum > 0.
Reference: the single-quote S60 + CAP10 arm of round LC (same clock).
Usage: python3 -m predictfun.research.pf_ladder_report
"""
import json
import numpy as np, pandas as pd
from . import pf_io as io

READ = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031']; FRESH = ['20261101', '20261102']; ALL = READ + FRESH
S4 = f'{io.OUT}/step4'


def ddir(d):
    return f"{S4}/ladder_{'preds_c3' if d in FRESH else 'preds_o3'}_lad1/{d}"


SUM = {d: json.load(open(f'{ddir(d)}/summary.json'))['arms'] for d in ALL}
NAMES = list(SUM[ALL[0]].keys())


def stem(n):
    return n.replace('=', '').replace(',', '_')


def arm_row(n):
    v = {d: SUM[d][n] for d in ALL}
    W = pd.concat([pd.read_parquet(f'{ddir(d)}/win/{stem(n)}.parquet', columns=['mtm', 'carry', 'net_end']) for d in ALL])
    f = pd.concat([pd.read_parquet(f'{ddir(d)}/fills/{stem(n)}.parquet', columns=['q', 'through', 'spread', 'informed']) for d in ALL])
    sh = max(f.q.sum(), 1e-9); qf = f[~f.through]; tf = f[f.through]
    m = [v[d]['mtm_stop_usd'] for d in ALL]
    return dict(params=v[ALL[0]]['params'], mtm_by_day=m, read=round(sum(m[:6]), 2), read_pos=sum(x > 0 for x in m[:6]), fresh=round(sum(m[6:]), 2),
                all=round(sum(m), 2), gross=round(sum(v[d]['gross_usd'] for d in ALL), 2), carry_sd=round(float(W.carry.std()), 2),
                abs_net=round(float(W.net_end.abs().mean()), 1), shares=round(float(sh)), spread_c=round(100 * f.spread.sum() / sh, 3),
                informed_c=round(100 * f.informed.sum() / sh, 3), through_share=round(float(tf.q.sum() / sh), 3),
                queue_fill_c=round(100 * (qf.spread.sum() + qf.informed.sum()) / max(qf.q.sum(), 1e-9), 3),
                swept_fill_c=round(100 * (tf.spread.sum() + tf.informed.sum()) / max(tf.q.sum(), 1e-9), 3),
                requests=sum(v[d]['place'] + v[d]['cancel'] for d in ALL), throttled=sum(v[d]['throttled'] for d in ALL))


tab = {n: arm_row(n) for n in NAMES}
elig = [n for n, r in tab.items() if r['read_pos'] >= 4]
best = max(elig, key=lambda n: tab[n]['read']) if elig else None
lc = 'MICRO+G4+G2+G6+G5+G3+G1+M025+XNA+XMIN1+XS0+XV+S60+CAP10/mid@A1:5:25:0'
ref = [json.load(open(f"{S4}/live_tokyoreal_custom_e2_{'preds_c3' if d in FRESH else 'preds_o3'}_lc/{d}/summary.json"))['arms'][lc]['mtm_stop_usd'] for d in ALL]
out = dict(table=tab, eligible=elig, best=best, promising=bool(best and tab[best]['fresh'] > 0),
           reference_S60_CAP10=dict(mtm_by_day=ref, read=round(sum(ref[:6]), 2), fresh=round(sum(ref[6:]), 2), all=round(sum(ref), 2)))
io.write_json(f'{S4}/report_ladder_lad1.json', out)
short = lambda n: ','.join(x for x in n.split(',') if x.split('=')[0] in ('K', 'z', 'S', 'TH'))
print(f"{'arm':26s} {'read mtm':>9s} {'d+':>3s} {'fresh':>7s} {'8d':>7s} {'gross':>7s} {'sd/win':>6s} | {'spread':>6s} {'inform':>6s} {'swept%':>6s} {'queue c':>7s} {'swept c':>7s} | {'|net|':>5s} {'shares':>7s} {'req':>7s} {'thr':>6s}")
for n, r in sorted(tab.items(), key=lambda kv: -kv[1]['read']):
    print(f"{short(n):26s} {r['read']:9.0f} {r['read_pos']:3d} {r['fresh']:7.0f} {r['all']:7.0f} {r['gross']:7.0f} {r['carry_sd']:6.1f} | {r['spread_c']:6.2f} {r['informed_c']:6.2f} "
          f"{100 * r['through_share']:6.1f} {r['queue_fill_c']:7.2f} {r['swept_fill_c']:7.2f} | {r['abs_net']:5.1f} {r['shares']:7d} {r['requests']:7d} {r['throttled']:6d}")
print(f"reference single quote S60 + CAP10: read {out['reference_S60_CAP10']['read']}, fresh {out['reference_S60_CAP10']['fresh']}, 8 days {out['reference_S60_CAP10']['all']}")
print('eligible (mark-at-stop > 0 on >= 4/6 read days):', [short(n) for n in elig])
print('selected:', short(best) if best else None, '| promising (fresh mark-at-stop > 0):', out['promising'])
if best:
    print('  selected mark-at-stop by day:', [round(x) for x in tab[best]['mtm_by_day']])
