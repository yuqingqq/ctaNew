"""B12 POST HOC (2026-11-10, written after reading b12.txt; descriptive, no reading): is the top makers' advantage spread
(pricing) or move (smaller adverse move)? Runs the committed pf_b12_makers build unchanged (its report() is replaced by a
capture) and splits net 1 s / 5 s into spread + move for:
  P1 the at-best WITHIN gap (top 5 against ranks >= 6, lag 50 cells): WITHIN = WITHIN_spread + WITHIN_move;
  P2 legs behind the best (depth 1, 2+) and inside the spread, by group;
  P3 at the best by guard state (lag 50), by group, ours included.
Usage: predictfun/research/run_light.sh b12_posthoc -- python3 -m predictfun.research.pf_b12_posthoc <out_dir>
"""
import json
import sys

import numpy as np

from . import pf_b12_makers as B

CAP = {}


def capture(out, R, ctl, stat, days, top5, OWN, n_makers):
    CAP.update(R=R, days=days, ctl=ctl)


def parts(x):
    sh = x.shares.values; s = sh.sum()
    if s <= 0:
        return dict(sh=0.0, legs=0, spread=float('nan'), move1=float('nan'), net1=float('nan'), move5=float('nan'), net5=float('nan'))
    return dict(sh=float(s), legs=int(len(x)), spread=float((sh * x.spr.values).sum() / s), move1=float((sh * (x.v1.values - x.spr.values)).sum() / s),
                net1=float((sh * x.v1.values).sum() / s), move5=float((sh * (x.v5.values - x.spr.values)).sum() / s), net5=float((sh * x.v5.values).sum() / s))


def main():
    out = sys.argv[1]
    B.report = capture
    sys.argv = [sys.argv[0], out]
    B.main()
    R = CAP['R']; V = R[R.val_ok]; days = CAP['days']
    grp = {'top 5': V.g == 'top 5', 'O (ranks >= 6)': V.g.isin(['6-20', 'rest']), '6-20': V.g == '6-20', 'rest': V.g == 'rest', 'ours': V.g == 'ours'}
    ok = V.status.values == 'ok'
    res = dict(note='POST HOC, descriptive; same population and statuses as b12.json', days=days)
    # P1 the WITHIN gap at the best split into spread and move
    A = V[ok & (V.depth.values == 0)]
    D = A[A.g.isin(['top 5', '6-20', 'rest']) & A.g50.isin([B.G_ON, B.G_OFF]) & (A.sweep >= 0)]
    cell = (2 * (D.g50.values == B.G_OFF) + D.sweep.values).astype(np.int64); isT = D.g.values == 'top 5'
    p1 = {}
    for name, col in (('net1', D.v1.values), ('spread', D.spr.values), ('move1', D.v1.values - D.spr.values)):
        wT, vT = B.shares_vals(cell[isT], D.shares.values[isT], col[isT], 4); wO, vO = B.shares_vals(cell[~isT], D.shares.values[~isT], col[~isT], 4)
        p1[name] = dict(WITHIN=float(np.sum(wT * (vT - vO))), cells={B.CELL_LAB[i]: dict(T=float(vT[i]), O=float(vO[i]), wT=float(wT[i])) for i in range(4)})
    p1['identity_ok'] = abs(p1['net1']['WITHIN'] - p1['spread']['WITHIN'] - p1['move1']['WITHIN']) < 1e-9
    res['P1'] = p1
    # P2 behind the best and inside the spread
    res['P2'] = {}
    for gname, gm in grp.items():
        x = V[gm.values]; xo = x.status.values == 'ok'
        res['P2'][gname] = {'1 tick behind': parts(x[xo & (x.depth.values == 1)]), '2+ behind': parts(x[xo & (x.depth.values >= 2)]),
                            'inside the spread': parts(x[x.status.values == 'inside the spread']), 'at best': parts(x[xo & (x.depth.values == 0)])}
    # P3 at the best by guard state
    res['P3'] = {}
    for gname, gm in grp.items():
        x = A[gm.reindex(A.index).values]
        res['P3'][gname] = {k: parts(x[x.g50.values == v]) for k, v in (('ON', B.G_ON), ('OFF', B.G_OFF), ('warm-up', B.G_WARM))}
    json.dump(res, open(f'{out}/b12_posthoc.json', 'w'), indent=1, default=float)
    f = lambda v: 'n/a' if not np.isfinite(v) else f'{v:+.2f}'
    L = ['B12 POST HOC (written after reading b12.txt; descriptive). c/sh; spread + move = net.', '',
         f"P1 at-best WITHIN gap, top 5 against ranks >= 6 (lag 50): net {f(p1['net1']['WITHIN'])} = spread {f(p1['spread']['WITHIN'])} + "
         f"move {f(p1['move1']['WITHIN'])} (identity {p1['identity_ok']})"]
    for c in B.CELL_LAB:
        L.append(f"  {c}: wT {p1['net1']['cells'][c]['wT'] * 100:.1f}%; spread T {f(p1['spread']['cells'][c]['T'])} O {f(p1['spread']['cells'][c]['O'])}; "
                 f"move T {f(p1['move1']['cells'][c]['T'])} O {f(p1['move1']['cells'][c]['O'])}")
    L += ['', 'P2 by depth: group | bucket | sh/day | spread | move 1 s | net 1 s | move 5 s | net 5 s']
    for gname, d in res['P2'].items():
        for k, x in d.items():
            L.append(f"{gname} | {k} | {x['sh'] / days:,.0f} | {f(x['spread'])} | {f(x['move1'])} | {f(x['net1'])} | {f(x['move5'])} | {f(x['net5'])}")
    L += ['', 'P3 at the best by guard state (lag 50): group | state | sh/day | spread | move 1 s | net 1 s | net 5 s']
    for gname, d in res['P3'].items():
        for k, x in d.items():
            L.append(f"{gname} | {k} | {x['sh'] / days:,.0f} | {f(x['spread'])} | {f(x['move1'])} | {f(x['net1'])} | {f(x['net5'])}")
    open(f'{out}/b12_posthoc.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
