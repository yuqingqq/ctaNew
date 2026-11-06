"""Latency test L1 report (declared in STATUS.yml 2026-11-03 04:21Z): live-like C2-P at the measured Tokyo clock.

Columns: T20 = tokyo20 clock with preds_lodo6 (the replay so far); RA = Tokyo clock, model A (tokyo20-trained, the frozen
live spec, applied to Tokyo-clock features); RB = Tokyo clock, model B (refit on Tokyo-clock features). All out of day.
Rows: design (concurrent = replace beside, sequential = XS0) x throttle (none, 1/s, 2/s, venue XV), plus D1 alone and
D1 + M025 at 5 shares. Per cell: 6-day gross, days > 0, +1 s c/sh, shares, max |net| in a window, requests per quoting
minute (mean, share of minutes > 500) and seconds > 40. Parity: T20 arms rerun here must equal rounds 2 / 5 to the cent.
Usage: python3 -m predictfun.research.pf_latency_report
"""
import json, os
import numpy as np, pandas as pd
from . import pf_io as io

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031']
S4 = f'{io.OUT}/step4'
DIRS = dict(RA='live_tokyoreal_custom_e2_preds_rl_a_l1', RB='live_tokyoreal_custom_e2_preds_rl_b_l1', T20='live_tokyo20_custom_e2_preds_lodo6_l1',
            R2='live_tokyo20_custom_e2_preds_lodo6_livedev2', R5='live_tokyo20_custom_e2_preds_lodo6_livedev5a')
SUM = {k: {d: json.load(open(f'{S4}/{v}/{d}/summary.json'))['arms'] for d in DAYS} for k, v in DIRS.items() if os.path.isdir(f'{S4}/{v}')}
ST = 'MICRO+G4+G2+G6+G5+G3+G1'; PL = f'{ST}+M025+XNA+XMIN1'; A1 = '/mid@A1:5:25:0'
ROWS = [('concurrent, none', f'{PL}{A1}'), ('concurrent, 1/s', f'{PL}+XT1{A1}'), ('concurrent, 2/s', f'{PL}+XT2{A1}'), ('concurrent, venue', f'{PL}+XV{A1}'),
        ('sequential, none', f'{PL}+XS0{A1}'), ('sequential, 1/s', f'{PL}+XS0+XT1{A1}'), ('sequential, 2/s', f'{PL}+XS0+XT2{A1}'),
        ('sequential, venue', f'{PL}+XS0+XV{A1}'), ('D1 alone, 5 sh', f'{ST}+XNA+XMIN1/mid'), ('D1 + M025, 5 sh', f'{ST}+M025+XNA+XMIN1/mid')]
T20_ALT = {f'{PL}+XT1{A1}': ('R5', f'{PL}+XQ0+XT1{A1}'), f'{PL}+XT2{A1}': ('R5', f'{PL}+XQ0+XT2{A1}'),
           f'{PL}+XS0+XT1{A1}': ('R2', f'{PL}+XS0+XT1{A1}'), f'{PL}+XS0+XT2{A1}': ('R2', f'{PL}+XS0+XT2{A1}')}


def stem(n):
    return n.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')


def cell(src, name):
    if src not in SUM or not all(name in SUM[src][d] for d in DAYS):
        return None
    v = [SUM[src][d][name] for d in DAYS]; g = [x['gross_usd'] for x in v]; sh = sum(x['shares'] for x in v)
    out = dict(src=src, sum=round(sum(g), 2), by_day=g, pos=sum(x > 0 for x in g), shares=round(sh),
               mo1_c_sh=round(100 * sum(x['mo1_usd'] for x in v) / max(sh, 1e-9), 3))
    base = f'{S4}/{DIRS[src]}'; wm = []; qs = []; W = []
    for d in DAYS:
        f = pd.read_parquet(f'{base}/{d}/fills/{stem(name)}.parquet', columns=['mid', 't', 'side', 'q']).sort_values(['mid', 't'], kind='stable')
        if len(f):
            c = pd.Series(np.where(f.side.values == 0, 1.0, -1.0) * f.q.values).groupby(f.mid.values).cumsum().abs()
            wm.append(c.groupby(f.mid.values).max().values)
        qs.append(pd.read_parquet(f'{base}/{d}/requests/{stem(name)}.parquet', columns=['t_ms'])); W.append(pd.read_parquet(f'{base}/{d}/win/{stem(name)}.parquet', columns=['ws']))
    wm = np.concatenate(wm) if wm else np.zeros(1); q = pd.concat(qs); W = pd.concat(W)
    qm = np.concatenate([np.arange(int(w // 60000), int(w // 60000) + 4) for w in W.ws.values])
    qsec = np.concatenate([np.arange(int(w // 1000), int(w // 1000) + 240) for w in W.ws.values])
    cm = pd.Series((q.t_ms // 60000).astype(np.int64)).value_counts().reindex(qm, fill_value=0).values
    cs = pd.Series((q.t_ms // 1000).astype(np.int64)).value_counts().reindex(qsec, fill_value=0).values
    out.update(max_abs_net=round(float(wm.max()), 1), req_min_mean=round(float(cm.mean()), 1), min_gt500=round(float((cm > 500).mean()), 4),
               sec_gt40=round(float((cs > 40).mean()), 6))
    return out


tab = {}
for lab, name in ROWS:
    t20 = cell('T20', name)
    if t20 is None and name in T20_ALT:
        t20 = cell(*T20_ALT[name])
    tab[lab] = dict(arm=name, T20=t20, RA=cell('RA', name), RB=cell('RB', name))
par = {}
for lab, name in (('concurrent, none', f'{PL}{A1}'), ('sequential, none', f'{PL}+XS0{A1}')):
    a, b = cell('T20', name), cell('R2', name)
    par[lab] = bool(a and b and a['by_day'] == b['by_day'])
best = {}
for m in ('RA', 'RB'):
    ok = [(lab, v[m]) for lab, v in tab.items() if v[m] and lab.startswith(('concurrent', 'sequential'))]
    elig = [(lab, x) for lab, x in ok if x['pos'] >= 4 and x['sum'] > 0]
    best[m] = max(elig, key=lambda z: z[1]['sum'])[0] if elig else None
lodo = json.load(open(f'{io.OUT}/step5/lodo_rl_report.json')) if os.path.exists(f'{io.OUT}/step5/lodo_rl_report.json') else None
out = dict(days=DAYS, clock='pf_clock_rl (Tokyo measured)', parity_T20_rerun=par, table=tab, best_positive_4of6=best, model_scores=lodo)
io.write_json(f'{S4}/{DIRS["RA"]}/report_latency_l1.json', out)
print('parity of the tokyo20 reruns vs round 2:', par)
print(f"{'arm':20s} | {'T20 $':>9s} {'d+':>2s} | {'RA $':>9s} {'d+':>2s} {'mo1':>5s} {'max|n|':>6s} {'req/m':>5s} {'>500':>5s} | {'RB $':>9s} {'d+':>2s} {'mo1':>5s} {'max|n|':>6s}")
for lab, v in tab.items():
    t, a, b = v['T20'], v['RA'], v['RB']
    f = lambda x, k, w, p=0: (f"{x[k]:{w}.{p}f}" if x else ' ' * (w - 3) + 'n/a')
    print(f"{lab:20s} | {f(t,'sum',9,0)} {f(t,'pos',2)} | {f(a,'sum',9,0)} {f(a,'pos',2)} {f(a,'mo1_c_sh',5,2)} {f(a,'max_abs_net',6,0)} {f(a,'req_min_mean',5,0)} "
          f"{f(a,'min_gt500',5,3)} | {f(b,'sum',9,0)} {f(b,'pos',2)} {f(b,'mo1_c_sh',5,2)} {f(b,'max_abs_net',6,0)}")
print('best (sum > 0, >= 4/6 days):', best)
if lodo:
    print('model out-of-day R2 (Tokyo-clock target): A', [lodo[d]['a']['r2'] for d in DAYS], 'B', [lodo[d]['b']['r2'] for d in DAYS],
          '| lodo6 on tokyo20', [lodo[d]['lodo6_on_tokyo20']['r2'] for d in DAYS])
for lab in ('concurrent, none', 'sequential, venue', 'concurrent, venue'):
    for m in ('RA', 'RB'):
        if tab[lab][m]:
            print(f'  {lab} {m} by day', tab[lab][m]['by_day'])
