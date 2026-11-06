"""Lock-length grid for live-like C2-P under the SEQUENTIAL design (rounds 2 + 3; user 2026-11-02 "Test between 0-200ms").
Rows L = 0 / 50 / 100 / 150 / 200 / 500 / 1000 ms; columns: no throttle, exec-port throttle 1 place/s, 2 places/s.
Cells: 6-day gross (read days 10-23..27, 10-31), days > 0, +1 s mark c/sh, shares. Break-even L per column is computed
by linear interpolation between the two grid points where the 6-day sum changes sign (None if it never does).
Usage: python3 -m predictfun.research.pf_live_report3
"""
import json, os
from . import pf_io as io

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031']
B = f'{io.OUT}/step4/live_tokyo20_custom_e2_preds_lodo6_'
PL = 'MICRO+G4+G2+G6+G5+G3+G1+M025+XNA+XMIN1'; A1 = '/mid@A1:5:25:0'
LS = [0, 50, 100, 150, 200, 500, 1000]; TH = {'none': '', '1/s': '+XT1', '2/s': '+XT2'}
S = {r: {d: json.load(open(f'{B}{r}/{d}/summary.json'))['arms'] for d in DAYS} for r in ('livedev2', 'livedev3') if os.path.isdir(f'{B}{r}')}


def cell(L, th):
    name = f'{PL}+XS{L}{th}{A1}'
    for r in ('livedev2', 'livedev3'):
        if r in S and all(name in S[r][d] for d in DAYS):
            v = [S[r][d][name] for d in DAYS]; g = [x['gross_usd'] for x in v]; sh = sum(x['shares'] for x in v)
            return dict(src=r, sum=round(sum(g), 2), by_day=g, pos=sum(x > 0 for x in g), shares=round(sh),
                        mo1_c_sh=round(100 * sum(x['mo1_usd'] for x in v) / max(sh, 1e-9), 3))
    return None


grid = {c: {L: cell(L, th) for L in LS} for c, th in TH.items()}
be = {}
for c, col in grid.items():
    pts = [(L, col[L]['sum']) for L in LS if col[L] is not None]
    be[c] = None
    for (l0, s0), (l1, s1) in zip(pts, pts[1:]):
        if s0 > 0 >= s1:
            be[c] = round(l0 + (l1 - l0) * s0 / (s0 - s1), 1); break
v = [S['livedev2'][d][f'{PL}{A1}'] for d in DAYS]
out = dict(days=DAYS, design='sequential', reference_no_lock_concurrent=round(sum(x['gross_usd'] for x in v), 2),
           grid={c: {str(L): x for L, x in col.items()} for c, col in grid.items()}, break_even_lock_ms=be,
           note='break-even = linear interpolation of the 6-day sum between adjacent grid points; grid spacing 50 ms below 200')
io.write_json(f'{B}livedev3/report_lock_grid.json', out)
print('sequential lock grid, live-like C2-P, 6-day gross $ (days > 0 of 6) [+1 s c/sh]; no-lock concurrent reference', out['reference_no_lock_concurrent'])
print(f"{'L ms':>6s} " + ' '.join(f'{c:>28s}' for c in TH))
for L in LS:
    print(f'{L:6d} ' + ' '.join(f"{x['sum']:12.2f} ({x['pos']}/6) [{x['mo1_c_sh']:6.2f}]".rjust(28) if x else ' ' * 22 + 'n/a'.rjust(6) for x in (grid[c][L] for c in TH)))
print('break-even lock (ms):', be)
for c in TH:
    print(c, 'by day:', {L: grid[c][L]['by_day'] for L in LS if grid[c][L] and L <= 200})
